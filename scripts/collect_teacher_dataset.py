from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from dotenv import load_dotenv

from grid2op_env import GridAction, GridEnv
from grid2op_env.inference import (
    BaselineConfig,
    SimulationOutcome,
    _build_llm_client,
    _chat_completion_kwargs,
    _default_model_name,
    _llm_api_base_url,
    build_proposal_prompt,
    filter_selectable_simulations,
    parse_json_action,
    serialize_simulation_outcome,
    constrain_redispatch_delta,
)
from grid2op_env.models import GridObservation, RedispatchGeneratorContext, TaskId
from grid2op_env.server.tasks import TASKS, benchmark_tiers_for_task


DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TEACHER_MODEL = os.environ.get("GRID2OP_TEACHER_MODEL", "Qwen/Qwen3.6-27B")
DEFAULT_TEACHER_API_BASE_URL = os.environ.get(
    "GRID2OP_TEACHER_API_BASE_URL",
    os.environ.get("API_BASE_URL", DEFAULT_GROQ_BASE_URL),
)
DEFAULT_GRID_MESSAGE_TIMEOUT_S = float(os.environ.get("GRID2OP_MESSAGE_TIMEOUT_S", "180"))
DEFAULT_GRID_CONNECT_TIMEOUT_S = float(os.environ.get("GRID2OP_CONNECT_TIMEOUT_S", "30"))


def _load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (Path.cwd() / ".env", repo_root / ".env"):
        if path.exists():
            load_dotenv(path, override=False)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _observation_summary(observation: GridObservation) -> dict[str, Any]:
    stressed_lines = sorted(
        (
            {
                "line_id": line_id,
                "rho": round(float(rho), 6),
                "status": bool(observation.line_status[line_id]),
                "timestep_overflow": int(observation.timestep_overflow[line_id])
                if line_id < len(observation.timestep_overflow)
                else 0,
            }
            for line_id, rho in enumerate(observation.rho)
            if float(rho) >= 0.7 or not bool(observation.line_status[line_id])
        ),
        key=lambda item: (not item["status"], item["rho"], item["timestep_overflow"]),
        reverse=True,
    )[:12]
    return {
        "max_rho": round(max((float(value) for value in observation.rho), default=0.0), 6),
        "stressed_lines": stressed_lines,
        "disconnected_lines": [
            line_id for line_id, status in enumerate(observation.line_status) if not status
        ],
        "sensitivity_guidance": observation.sensitivity_guidance[:5],
    }


def _action_type(action: GridAction) -> str:
    if action.do_nothing:
        return "do_nothing"
    if action.redispatch:
        return "redispatch"
    if action.line_set:
        statuses = [int(value) for value in action.line_set.values()]
        if statuses and statuses[0] == 1:
            return "reconnect_line"
        if statuses and statuses[0] == -1:
            return "disconnect_line"
        return "line_set"
    return "empty"


def _candidate_summary(
    observation: GridObservation,
    simulations: Sequence[SimulationOutcome],
) -> list[dict[str, Any]]:
    current_max_rho = max((float(value) for value in observation.rho), default=0.0)
    return [
        {
            "candidate_index": outcome.candidate_index,
            "action": outcome.action.model_dump(),
            "action_type": _action_type(outcome.action),
            "simulated_reward": round(float(outcome.simulated_reward), 6),
            "max_rho": round(float(outcome.max_rho), 6),
            "delta_max_rho": round(float(outcome.max_rho) - current_max_rho, 6),
            "redispatch_magnitude": round(redispatch_magnitude(outcome), 6),
            "overloaded_line_count": len(outcome.overloaded_line_ids),
            "disconnected_line_count": len(outcome.disconnected_lines),
            "max_timestep_overflow": max_simulated_overflow(outcome),
            "convergence_failed": outcome.convergence_failed,
            "done": outcome.done,
        }
        for outcome in simulations
    ]


def build_sft_prompt(
    task_id: TaskId,
    step: int,
    max_steps: int,
    observation: GridObservation,
    graph_intelligence: dict[str, Any],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
    simulations: Sequence[SimulationOutcome],
) -> str:
    """Prompt used for local SFT: choose the best verified GridAction JSON."""

    task_objective_lines: list[str] = []
    if task_id == "single_fault":
        task_objective_lines = [
            "single_fault_objective=bring max_rho below 0.80; choose redispatch while above target if it lowers max_rho; choose do_nothing mainly after target is met.",
        ]
    elif task_id == "n_minus_1":
        task_objective_lines = [
            "n_minus_1_objective=score rewards early emergency clearing and steady security, not only immediate simulated_reward.",
            "n_minus_1_thresholds=steps 1-5 should reduce max_rho below 0.92; steps 6-20 should keep max_rho below 0.90.",
            "n_minus_1_action_rule=prefer safe reconnect when it restores the disconnected line; while max_rho is above the active threshold, prefer redispatch that lowers max_rho even if immediate reward is slightly lower than do_nothing.",
        ]

    return "\n".join(
        [
            "You are a power-grid control policy.",
            "Choose exactly one action for the current Grid2Op state.",
            "Return only a valid GridAction JSON object with keys: line_set, redispatch, do_nothing.",
            "Do not include markdown, prose, reasoning, or extra keys.",
            f"task_id={task_id}",
            f"step={step}/{max_steps}",
            *task_objective_lines,
            "observation_summary=" + _json_dumps(_observation_summary(observation)),
            "redispatch_generator_bounds="
            + _json_dumps([context.model_dump() for context in redispatch_generators]),
            "graph_intelligence="
            + _json_dumps(
                {
                    "bridge_lines": graph_intelligence.get("bridge_lines", []),
                    "safe_to_disconnect": graph_intelligence.get("safe_to_disconnect", []),
                    "n_minus_1_critical_lines": graph_intelligence.get(
                        "n_minus_1_critical_lines", []
                    ),
                    "stressed_lines": graph_intelligence.get("stressed_lines", []),
                    "congestion_corridor": graph_intelligence.get(
                        "congestion_corridor", {}
                    ),
                }
            ),
            "verified_simulation_results="
            + _json_dumps([serialize_simulation_outcome(outcome) for outcome in simulations]),
            "candidate_comparison_summary="
            + _json_dumps(_candidate_summary(observation, simulations)),
        ]
    )


def max_simulated_overflow(outcome: SimulationOutcome) -> int:
    overflow = outcome.raw_result.get("timestep_overflow", [])
    if not isinstance(overflow, list):
        return 0
    return max((int(value) for value in overflow), default=0)


def redispatch_magnitude(outcome: SimulationOutcome) -> float:
    return sum(abs(float(delta)) for delta in outcome.action.redispatch.values())


def select_dataset_label(
    task_id: TaskId,
    observation: GridObservation,
    simulations: Sequence[SimulationOutcome],
) -> SimulationOutcome | None:
    """Choose the supervised target from simulator-verified candidates.

    The dataset label should teach the model the best verified action for the
    task objective, not merely the teacher's first candidate. This selector is
    intentionally task-objective based and does not contain line-specific rules.
    """

    selectable = filter_selectable_simulations(simulations)
    if not selectable:
        return None

    no_overload = [
        outcome for outcome in selectable if not outcome.overloaded_line_ids
    ]
    candidates = no_overload or selectable

    if task_id == "single_fault":
        current_max_rho = max((float(value) for value in observation.rho), default=0.0)
        do_nothing_candidates = [
            outcome for outcome in candidates if outcome.action.do_nothing
        ]
        active_redispatch = [
            outcome
            for outcome in candidates
            if not outcome.action.do_nothing and outcome.action.redispatch
        ]
        improving_active = [
            outcome
            for outcome in active_redispatch
            if float(outcome.max_rho) < current_max_rho - 0.002
        ]
        if current_max_rho > 0.80 and improving_active:
            return min(
                improving_active,
                key=lambda outcome: (
                    float(outcome.max_rho),
                    -float(outcome.simulated_reward),
                    redispatch_magnitude(outcome),
                    len(outcome.disconnected_lines),
                    outcome.candidate_index,
                ),
            )
        if current_max_rho <= 0.80 and do_nothing_candidates:
            return max(
                do_nothing_candidates,
                key=lambda outcome: (
                    float(outcome.simulated_reward),
                    -float(outcome.max_rho),
                    -len(outcome.disconnected_lines),
                    -outcome.candidate_index,
                ),
            )
        strong_active = [
            outcome
            for outcome in active_redispatch
            if float(outcome.max_rho) < current_max_rho - 0.01
        ]
        if strong_active:
            return min(
                strong_active,
                key=lambda outcome: (
                    float(outcome.max_rho),
                    -float(outcome.simulated_reward),
                    redispatch_magnitude(outcome),
                    len(outcome.disconnected_lines),
                    outcome.candidate_index,
                ),
            )
        return max(
            candidates,
            key=lambda outcome: (
                float(outcome.simulated_reward),
                -float(outcome.max_rho),
                -len(outcome.disconnected_lines),
                -len(outcome.action.line_set),
                -outcome.candidate_index,
            ),
        )

    if task_id == "n_minus_1":
        current_max_rho = max((float(value) for value in observation.rho), default=0.0)
        active_redispatch = [
            outcome
            for outcome in candidates
            if outcome.action.redispatch
            and (
                float(outcome.max_rho) < current_max_rho - 0.005
                or float(outcome.simulated_reward)
                >= max(float(candidate.simulated_reward) for candidate in candidates) - 0.05
            )
        ]
        if active_redispatch and current_max_rho >= 0.80:
            return min(
                active_redispatch,
                key=lambda outcome: (
                    float(outcome.max_rho),
                    -float(outcome.simulated_reward),
                    redispatch_magnitude(outcome),
                    outcome.candidate_index,
                ),
            )
        safe_reconnect = [
            outcome
            for outcome in candidates
            if outcome.action.line_set
            and any(int(status) == 1 for status in outcome.action.line_set.values())
            and not outcome.overloaded_line_ids
        ]
        if safe_reconnect:
            return max(
                safe_reconnect,
                key=lambda outcome: (
                    float(outcome.simulated_reward),
                    -float(outcome.max_rho),
                    -len(outcome.disconnected_lines),
                    -outcome.candidate_index,
                ),
            )
        return max(
            candidates,
            key=lambda outcome: (
                float(outcome.simulated_reward),
                -float(outcome.max_rho),
                -len(outcome.disconnected_lines),
                -len(outcome.action.line_set),
                -outcome.candidate_index,
            ),
        )

    current_max_overflow = max(
        (int(value) for value in observation.timestep_overflow), default=0
    )
    current_max_rho = max((float(value) for value in observation.rho), default=0.0)
    active_cascade = current_max_overflow > 0 or current_max_rho > 1.0
    if active_cascade:
        return min(
            candidates,
            key=lambda outcome: (
                max_simulated_overflow(outcome),
                len(outcome.overloaded_line_ids),
                float(outcome.max_rho),
                -float(outcome.simulated_reward),
                len(outcome.disconnected_lines),
                len(outcome.action.line_set),
                outcome.candidate_index,
            ),
        )

    return max(
        candidates,
        key=lambda outcome: (
            float(outcome.simulated_reward),
            -float(outcome.max_rho),
            -len(outcome.disconnected_lines),
            -len(outcome.action.line_set),
            -outcome.candidate_index,
        ),
    )


def parse_teacher_only_proposals(
    raw_output: str,
    task_id: TaskId,
    n_line: int,
    n_gen: int,
    redispatchable_generators: Sequence[int],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
) -> tuple[list[tuple[GridAction, dict[str, Any]]], dict[str, Any]]:
    payload = parse_json_action(raw_output)
    if task_id == "single_fault":
        raw_candidates = [
            payload.get("primary_action"),
            payload.get("backup_action_1"),
            payload.get("backup_action_2"),
        ]
    else:
        raw_candidates = payload.get("candidates", [])

    candidates: list[tuple[GridAction, dict[str, Any]]] = []
    raw_candidate_count = 0
    if isinstance(raw_candidates, list):
        for item in raw_candidates[:3]:
            if not isinstance(item, dict):
                continue
            raw_candidate_count += 1
            parsed = parse_teacher_candidate(
                payload=item,
                task_id=task_id,
                n_line=n_line,
                n_gen=n_gen,
                redispatchable_generators=redispatchable_generators,
                redispatch_generators=redispatch_generators,
            )
            if parsed is None:
                continue
            action, trace = parsed
            candidates.append((action, {**trace, "source": "teacher"}))

    deduped: list[tuple[GridAction, dict[str, Any]]] = []
    seen: set[str] = set()
    for action, trace in candidates:
        key = json.dumps(action.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, trace))

    return deduped[:3], {
        "raw_candidate_count": raw_candidate_count,
        "parsed_candidate_count": len(candidates),
        "deduped_candidate_count": len(deduped[:3]),
        "teacher_only": True,
    }


def parse_teacher_candidate(
    payload: dict[str, Any],
    task_id: TaskId,
    n_line: int,
    n_gen: int,
    redispatchable_generators: Sequence[int],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
) -> tuple[GridAction, dict[str, Any]] | None:
    action_type = payload.get("action_type")
    allowed_redispatch = {int(gen_id) for gen_id in redispatchable_generators}
    redispatch_context_by_id = {int(context.gen_id): context for context in redispatch_generators}

    if action_type == "do_nothing":
        return GridAction(do_nothing=True), {
            "decision": "do_nothing",
            "reason": str(payload.get("reason", "teacher_do_nothing")),
        }

    if action_type == "redispatch":
        gen_id = payload.get("gen_id")
        delta = payload.get("delta_mw")
        if gen_id is None or delta is None:
            return None
        gen_id_int = int(gen_id)
        if gen_id_int < 0 or gen_id_int >= n_gen or gen_id_int not in allowed_redispatch:
            return None
        if gen_id_int not in redispatch_context_by_id:
            return None
        constrained = constrain_redispatch_delta(float(delta), redispatch_context_by_id[gen_id_int])
        if constrained is None:
            return None
        return GridAction(redispatch={gen_id_int: constrained}), {
            "decision": "redispatch",
            "reason": str(payload.get("reason", "")),
        }

    if action_type in {"disconnect_line", "reconnect_line"}:
        if task_id == "single_fault":
            return None
        line_id = payload.get("line_id")
        if line_id is None:
            return None
        line_id_int = int(line_id)
        if line_id_int < 0 or line_id_int >= n_line:
            return None
        status = -1 if action_type == "disconnect_line" else 1
        return GridAction(line_set={line_id_int: status}), {
            "decision": action_type,
            "reason": str(payload.get("reason", "")),
        }

    return None


def filter_teacher_candidate_proposals(
    task_id: TaskId,
    observation: GridObservation,
    graph_intelligence: dict[str, Any],
    proposal_candidates: Sequence[tuple[GridAction, dict[str, Any]]],
) -> tuple[list[tuple[GridAction, dict[str, Any]]], dict[str, Any]]:
    filtered: list[tuple[GridAction, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    safe_disconnect = {
        int(line_id) for line_id in graph_intelligence.get("safe_to_disconnect", [])
    }

    for action, trace in proposal_candidates:
        rejection_reasons: list[str] = []
        for line_id_raw, status_raw in action.line_set.items():
            line_id = int(line_id_raw)
            status = int(status_raw)
            currently_connected = bool(observation.line_status[line_id])
            if task_id == "single_fault":
                rejection_reasons.append(f"topology_blocked:{line_id}:{status}")
            elif status == -1 and task_id == "n_minus_1":
                rejection_reasons.append(f"new_disconnect_blocked:{line_id}")
            elif status == -1 and task_id in {"cascade_prevent", "multi_stage_cascade"}:
                if line_id not in safe_disconnect:
                    rejection_reasons.append(f"unsafe_disconnect_filtered:{line_id}")
            elif status == 1 and currently_connected:
                rejection_reasons.append(f"reconnect_already_connected:{line_id}")

        if rejection_reasons:
            rejected.append(
                {
                    "action": action.model_dump(),
                    "reason": ",".join(sorted(rejection_reasons)),
                }
            )
            continue
        filtered.append((action, trace))

    deduped: list[tuple[GridAction, dict[str, Any]]] = []
    seen: set[str] = set()
    for action, trace in filtered:
        key = json.dumps(action.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, trace))

    return deduped[:3], {
        "prefiltered_candidate_count": len(deduped[:3]),
        "prefilter_rejections": rejected,
    }


def collect_teacher_dataset(
    base_url: str,
    output_path: Path,
    task_ids: Sequence[TaskId],
    episodes_per_task: int,
    max_steps_per_episode: int | None,
    seed_start: int,
    scenario_mode: str,
    model: str,
    max_tokens: int,
    temperature: float,
    connect_timeout_s: float,
    message_timeout_s: float,
) -> dict[str, Any]:
    llm_config = BaselineConfig(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=float(os.environ.get("TOP_P", "0.8")),
        presence_penalty=float(os.environ.get("PRESENCE_PENALTY", "0.0")),
        top_k=int(os.environ.get("TOP_K", "20")),
        min_p=float(os.environ.get("MIN_P", "0.0")),
        repetition_penalty=float(os.environ.get("REPETITION_PENALTY", "1.0")),
        enable_thinking=False,
        num_seeds=episodes_per_task,
        seed_start=seed_start,
        scenario_mode=scenario_mode,  # type: ignore[arg-type]
    )
    client = _build_llm_client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "base_url": base_url,
        "api_base_url": _llm_api_base_url(),
        "model": model,
        "output_path": str(output_path),
        "rows_written": 0,
        "episodes": 0,
        "teacher_calls": 0,
        "skipped_model_output": 0,
        "skipped_no_candidates": 0,
        "skipped_no_selectable": 0,
        "tasks": {},
    }

    started_at = perf_counter()
    with GridEnv(
        base_url=base_url,
        connect_timeout_s=connect_timeout_s,
        message_timeout_s=message_timeout_s,
    ).sync() as env, output_path.open(
        "a", encoding="utf-8"
    ) as handle:
        for task_id in task_ids:
            task = TASKS[task_id]
            benchmark_tiers = benchmark_tiers_for_task(task_id)
            stats["tasks"].setdefault(
                task_id,
                {
                    "rows": 0,
                    "episodes": 0,
                    "skipped_model_output": 0,
                    "skipped_no_candidates": 0,
                    "skipped_no_selectable": 0,
                },
            )
            for episode_index in range(episodes_per_task):
                seed = seed_start + episode_index
                benchmark_tier = benchmark_tiers[episode_index % len(benchmark_tiers)]
                result = env.reset(
                    task_id=task_id,
                    seed=seed,
                    difficulty_level=episode_index + 1,
                    scenario_mode=scenario_mode,  # type: ignore[arg-type]
                    benchmark_tier=benchmark_tier,
                )
                state = env.state()
                stats["episodes"] += 1
                stats["tasks"][task_id]["episodes"] += 1

                step_limit = min(task.max_steps, max_steps_per_episode or task.max_steps)
                for step_idx in range(step_limit):
                    if result.done:
                        break

                    planning_context = env.planning_context(state.episode_id)
                    graph_intelligence = planning_context.graph_intelligence
                    redispatch_generators = planning_context.redispatch_generators
                    proposal_prompt = build_proposal_prompt(
                        task_id=task_id,
                        observation=result.observation,
                        graph_intelligence=graph_intelligence,
                        redispatchable_generators=planning_context.redispatchable_generators,
                        redispatch_generators=redispatch_generators,
                        step_count=step_idx,
                        max_steps=task.max_steps,
                        include_task_description=(step_idx == 0),
                    )
                    response = client.chat.completions.create(
                        **_chat_completion_kwargs(
                            llm_config=llm_config,
                            prompt=proposal_prompt,
                        )
                    )
                    stats["teacher_calls"] += 1
                    proposal_raw_output = response.choices[0].message.content or ""
                    try:
                        proposal_candidates, proposal_trace = parse_teacher_only_proposals(
                            proposal_raw_output,
                            task_id=task_id,
                            n_line=len(result.observation.line_status),
                            n_gen=len(result.observation.gen_p),
                            redispatchable_generators=planning_context.redispatchable_generators,
                            redispatch_generators=redispatch_generators,
                        )
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        stats["skipped_model_output"] += 1
                        stats["tasks"][task_id]["skipped_model_output"] += 1
                        print(
                            json.dumps(
                                {
                                    "event": "skip_model_output",
                                    "task_id": task_id,
                                    "seed": seed,
                                    "benchmark_tier": benchmark_tier,
                                    "step": step_idx + 1,
                                    "error": str(exc),
                                    "raw_output_prefix": proposal_raw_output[:300],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        break
                    proposal_candidates, prefilter_trace = filter_teacher_candidate_proposals(
                        task_id=task_id,
                        observation=result.observation,
                        graph_intelligence=graph_intelligence,
                        proposal_candidates=proposal_candidates,
                    )
                    if not proposal_candidates:
                        stats["skipped_no_candidates"] += 1
                        stats["tasks"][task_id]["skipped_no_candidates"] += 1
                        break
                    simulation_response = env.simulate_candidates(
                        state.episode_id,
                        [action for action, _trace in proposal_candidates],
                    )
                    simulations = [
                        SimulationOutcome(
                            candidate_index=index,
                            action=sim_result.action,
                            trace=proposal_candidates[index - 1][1],
                            done=sim_result.done,
                            simulated_reward=sim_result.simulated_reward,
                            max_rho=sim_result.max_rho,
                            overloaded_line_ids=sim_result.overloaded_line_ids,
                            disconnected_lines=sim_result.disconnected_lines,
                            convergence_failed=sim_result.convergence_failed,
                            exceptions=sim_result.exceptions,
                            raw_result=sim_result.raw_result,
                        )
                        for index, sim_result in enumerate(
                            simulation_response.results, start=1
                        )
                    ]
                    selectable = filter_selectable_simulations(simulations)
                    selected = select_dataset_label(
                        task_id=task_id,
                        observation=result.observation,
                        simulations=simulations,
                    )
                    if selected is None:
                        stats["skipped_no_selectable"] += 1
                        stats["tasks"][task_id]["skipped_no_selectable"] += 1
                        break
                    sft_prompt = build_sft_prompt(
                        task_id=task_id,
                        step=step_idx + 1,
                        max_steps=task.max_steps,
                        observation=result.observation,
                        graph_intelligence=graph_intelligence,
                        redispatch_generators=redispatch_generators,
                        simulations=simulations,
                    )
                    completion = _json_dumps(selected.action.model_dump())
                    row = {
                        "messages": [
                            {"role": "user", "content": sft_prompt},
                            {"role": "assistant", "content": completion},
                        ],
                        "metadata": {
                            "task_id": task_id,
                            "seed": seed,
                            "benchmark_tier": benchmark_tier,
                            "scenario_mode": scenario_mode,
                            "episode_id": state.episode_id,
                            "step": step_idx + 1,
                            "env_name": state.env_name,
                            "teacher_model": model,
                            "teacher_prompt": proposal_prompt,
                            "teacher_raw_output": proposal_raw_output,
                            "proposal_trace": {**proposal_trace, **prefilter_trace},
                            "label_policy": "task_objective_verified_simulation",
                            "selected_candidate": selected.candidate_index,
                            "selected_action": selected.action.model_dump(),
                            "simulations": [
                                serialize_simulation_outcome(outcome)
                                for outcome in simulations
                            ],
                            "observation_summary": _observation_summary(
                                result.observation
                            ),
                            "scenario_metadata": state.scenario_metadata,
                        },
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    stats["rows_written"] += 1
                    stats["tasks"][task_id]["rows"] += 1

                    result = env.step(selected.action)
                    state = env.state()

    stats["wall_time_s"] = round(perf_counter() - started_at, 6)
    return stats


def main() -> None:
    _load_env()
    os.environ.setdefault("API_BASE_URL", DEFAULT_TEACHER_API_BASE_URL)
    os.environ.setdefault("MODEL_NAME", DEFAULT_TEACHER_MODEL)
    if not os.environ.get("API_KEY") and os.environ.get("GROQ_API_KEY"):
        os.environ.setdefault("API_KEY", os.environ["GROQ_API_KEY"])

    parser = argparse.ArgumentParser(
        description="Collect verified teacher-action data for Grid2Op SFT."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRID2OP_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--teacher-api-base-url",
        default=os.environ.get("API_BASE_URL", DEFAULT_TEACHER_API_BASE_URL),
        help="OpenAI-compatible API endpoint for the teacher model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/datasets/teacher_actions.jsonl"),
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        nargs="+",
        choices=sorted(TASKS.keys()),
        default=["single_fault"],
    )
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--scenario-mode",
        choices=["benchmark", "curriculum"],
        default=os.environ.get("GRID2OP_SCENARIO_MODE", "benchmark"),
    )
    parser.add_argument(
        "--model",
        default=_default_model_name(),
        help="Teacher model served from the configured OpenAI-compatible API.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "300")))
    parser.add_argument(
        "--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.2"))
    )
    parser.add_argument("--connect-timeout-s", type=float, default=DEFAULT_GRID_CONNECT_TIMEOUT_S)
    parser.add_argument("--message-timeout-s", type=float, default=DEFAULT_GRID_MESSAGE_TIMEOUT_S)
    args = parser.parse_args()

    os.environ["API_BASE_URL"] = args.teacher_api_base_url
    os.environ["MODEL_NAME"] = args.model

    stats = collect_teacher_dataset(
        base_url=args.base_url,
        output_path=args.output,
        task_ids=args.task_ids,
        episodes_per_task=args.episodes_per_task,
        max_steps_per_episode=args.max_steps_per_episode,
        seed_start=args.seed_start,
        scenario_mode=args.scenario_mode,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        connect_timeout_s=args.connect_timeout_s,
        message_timeout_s=args.message_timeout_s,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
