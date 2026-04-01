from __future__ import annotations

import csv
import json
import logging
import os
import re
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Sequence

import requests
from dotenv import load_dotenv
from openai import OpenAI

from grid2op_env import BaselineRequest, BaselineScores, GridAction, GridEnv
from grid2op_env.models import GridObservation, RedispatchGeneratorContext, ScenarioMode, TaskId
from grid2op_env.server.tasks import TASKS, benchmark_tiers_for_task


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_env() -> None:
    env_dir = Path(__file__).resolve().parent
    candidate_paths = [
        Path.cwd() / ".env",
        env_dir / ".env",
        env_dir.parent / ".env",
    ]
    for path in candidate_paths:
        if path.exists():
            load_dotenv(path, override=False)


_load_env()
configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    top_k: int
    min_p: float
    repetition_penalty: float
    enable_thinking: bool
    num_seeds: int
    seed_start: int
    scenario_mode: ScenarioMode


@dataclass
class SimulationOutcome:
    candidate_index: int
    action: GridAction
    trace: dict[str, Any]
    done: bool
    simulated_reward: float
    max_rho: float
    overloaded_line_ids: list[int]
    disconnected_lines: list[int]
    convergence_failed: bool
    exceptions: list[str]
    raw_result: dict[str, Any]


def run_baseline_suite(
    base_url: str,
    config: BaselineRequest | None = None,
    task_ids: Sequence[TaskId] | None = None,
) -> BaselineScores:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = prepare_run_paths(timestamp)
    attach_file_logger(run_paths["log"])

    request_config = config or BaselineRequest(
        model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-9B")
    )
    llm_config = BaselineConfig(
        model=request_config.model,
        max_tokens=request_config.max_tokens,
        temperature=request_config.temperature,
        top_p=request_config.top_p,
        presence_penalty=request_config.presence_penalty,
        top_k=request_config.top_k,
        min_p=request_config.min_p,
        repetition_penalty=request_config.repetition_penalty,
        enable_thinking=request_config.enable_thinking,
        num_seeds=request_config.num_seeds,
        seed_start=request_config.seed_start,
        scenario_mode=request_config.scenario_mode,
    )

    client = OpenAI()
    selected_task_ids = list(task_ids) if task_ids is not None else list(TASKS.keys())
    scores: Dict[TaskId, float] = {}
    episode_lengths: Dict[TaskId, int] = {}
    evaluation_records: list[dict[str, Any]] = []
    logger.info(
        "Starting baseline suite base_url=%s model=%s num_seeds=%s seed_start=%s",
        base_url,
        llm_config.model,
        llm_config.num_seeds,
        llm_config.seed_start,
    )

    with GridEnv(base_url=base_url).sync() as env:
        task_metrics: Dict[TaskId, list[dict[str, Any]]] = {task_id: [] for task_id in selected_task_ids}
        task_episode_counts: Dict[TaskId, int] = {task_id: 0 for task_id in selected_task_ids}
        for task_id in selected_task_ids:
            task = TASKS[task_id]
            benchmark_tiers = benchmark_tiers_for_task(task_id)
            for benchmark_tier in benchmark_tiers:
                for seed in range(llm_config.seed_start, llm_config.seed_start + llm_config.num_seeds):
                    task_episode_counts[task_id] += 1
                    curriculum_episode = task_episode_counts[task_id]
                    logger.info(
                        "Baseline episode start task_id=%s seed=%s curriculum_episode=%s benchmark_tier=%s max_steps=%s",
                        task_id,
                        seed,
                        curriculum_episode,
                        benchmark_tier,
                        task.max_steps,
                    )
                    result = env.reset(
                        task_id=task_id,
                        seed=seed,
                        difficulty_level=curriculum_episode,
                        scenario_mode=llm_config.scenario_mode,
                        benchmark_tier=benchmark_tier,
                    )
                    state = env.state()
                    logger.info(
                        "Initial state task_id=%s seed=%s episode_id=%s scenario_metadata=%s max_rho=%.4f disconnected=%s",
                        task_id,
                        seed,
                        state.episode_id,
                        state.scenario_metadata,
                        float(max(result.observation.rho)) if result.observation.rho else 0.0,
                        [idx for idx, status in enumerate(result.observation.line_status) if not status],
                    )

                    step_idx = 0
                    do_nothing_steps = 0
                    raw_outputs: list[dict[str, Any]] = []
                    while not result.done and step_idx < task.max_steps:
                        action, planning_trace = choose_action_with_qwen(
                            client=client,
                            env=env,
                            episode_id=state.episode_id,
                            task_id=task_id,
                            observation=result.observation,
                            step_count=step_idx,
                            max_steps=task.max_steps,
                            include_task_description=(step_idx == 0),
                            llm_config=llm_config,
                        )
                        raw_outputs.append(
                            {
                                "step": step_idx + 1,
                                "proposal_prompt": planning_trace["proposal_prompt"],
                                "proposal_raw_output": planning_trace["proposal_raw_output"],
                                "proposal_trace": planning_trace["proposal_trace"],
                                "graph_intelligence": planning_trace["graph_intelligence"],
                                "simulations": planning_trace["simulations"],
                                "final_prompt": planning_trace["final_prompt"],
                                "final_raw_output": planning_trace["final_raw_output"],
                                "final_trace": planning_trace["final_trace"],
                                "selected_action": action.model_dump(),
                                "observation_summary": {
                                    "max_rho": max(result.observation.rho) if result.observation.rho else 0.0,
                                    "disconnected_lines": [
                                        idx
                                        for idx, status in enumerate(result.observation.line_status)
                                        if not status
                                    ],
                                    "timestep_overflow": result.observation.timestep_overflow,
                                },
                            }
                        )
                        if action.do_nothing:
                            do_nothing_steps += 1
                        logger.info(
                            "Baseline action task_id=%s seed=%s step=%s action=%s trace=%s",
                            task_id,
                            seed,
                            step_idx + 1,
                            action.model_dump(),
                            planning_trace["final_trace"],
                        )
                        result = env.step(action)
                        step_idx += 1

                    state = env.state()
                    logger.info(
                        "Baseline episode finished task_id=%s seed=%s step_count=%s done=%s last_reward=%.4f",
                        task_id,
                        seed,
                        state.step_count,
                        state.done,
                        state.last_reward,
                    )
                    response = requests.post(
                        f"{base_url}/grader",
                        json={
                            "task_id": task_id,
                            "episode_log": [entry.model_dump() for entry in state.episode_log],
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    score_payload = response.json()
                    episode_score = float(score_payload["score"])
                    episode_length = int(state.step_count)
                    record = {
                        "task_id": task_id,
                        "seed": seed,
                        "curriculum_episode": curriculum_episode,
                        "benchmark_tier": benchmark_tier,
                        "score": episode_score,
                        "episode_length": episode_length,
                        "done": state.done,
                        "do_nothing_steps": do_nothing_steps,
                        "non_do_nothing_steps": max(0, episode_length - do_nothing_steps),
                        "episode_log": [entry.model_dump() for entry in state.episode_log],
                        "raw_outputs": raw_outputs,
                        "scenario_metadata": state.scenario_metadata,
                    }
                    evaluation_records.append(record)
                    task_metrics[task_id].append(record)
                    logger.info(
                        "Baseline score task_id=%s seed=%s benchmark_tier=%s score=%.6f episode_length=%s do_nothing_steps=%s",
                        task_id,
                        seed,
                        benchmark_tier,
                        episode_score,
                        episode_length,
                        do_nothing_steps,
                    )

        for task_id, records in task_metrics.items():
            task_scores = [float(record["score"]) for record in records]
            task_lengths = [int(record["episode_length"]) for record in records]
            scores[task_id] = round(mean(task_scores), 6) if task_scores else 0.0
            episode_lengths[task_id] = round(mean(task_lengths)) if task_lengths else 0

    logger.info("Baseline suite complete scores=%s", scores)
    baseline_scores = BaselineScores(
        model=llm_config.model,
        scores=scores,
        episode_lengths=episode_lengths,
    )
    write_evaluation_outputs(
        timestamp=timestamp,
        run_paths=run_paths,
        model=llm_config.model,
        base_url=base_url,
        llm_config=llm_config,
        baseline_scores=baseline_scores,
        evaluation_records=evaluation_records,
        selected_task_ids=selected_task_ids,
    )
    append_evaluation_markdown(
        timestamp=timestamp,
        model=llm_config.model,
        llm_config=llm_config,
        baseline_scores=baseline_scores,
        evaluation_records=evaluation_records,
        run_paths=run_paths,
        selected_task_ids=selected_task_ids,
    )
    return baseline_scores


def choose_action_with_qwen(
    client: OpenAI,
    env: GridEnv,
    episode_id: str,
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    include_task_description: bool,
    llm_config: BaselineConfig,
) -> tuple[GridAction, dict[str, Any]]:
    planning_context = env.planning_context(episode_id)
    graph_intelligence = planning_context.graph_intelligence
    redispatchable_generators = planning_context.redispatchable_generators
    redispatch_generators = planning_context.redispatch_generators
    logger.info(
        "Graph intelligence task_id=%s step=%s bridges=%s safe_disconnect=%s central_buses=%s islanded=%s corridor=%s stressed=%s",
        task_id,
        step_count + 1,
        graph_intelligence.get("bridge_lines", []),
        graph_intelligence.get("safe_to_disconnect", []),
        graph_intelligence.get("high_centrality_buses", []),
        graph_intelligence.get("islanded_clusters", []),
        graph_intelligence.get("congestion_corridor", "none"),
        graph_intelligence.get("stressed_lines", []),
    )
    proposal_prompt = build_proposal_prompt(
        task_id=task_id,
        observation=observation,
        graph_intelligence=graph_intelligence,
        redispatchable_generators=redispatchable_generators,
        redispatch_generators=redispatch_generators,
        step_count=step_count,
        max_steps=max_steps,
        include_task_description=include_task_description,
    )

    response = client.chat.completions.create(
        model=llm_config.model,
        messages=[{"role": "user", "content": proposal_prompt}],
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        presence_penalty=llm_config.presence_penalty,
        extra_body={
            "top_k": llm_config.top_k,
            "min_p": llm_config.min_p,
            "repetition_penalty": llm_config.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": llm_config.enable_thinking},
        },
    )

    proposal_raw_output = response.choices[0].message.content or ""
    logger.info(
        "Received proposal response task_id=%s chars=%s content=%r",
        task_id,
        len(proposal_raw_output),
        proposal_raw_output,
    )
    proposal_candidates, proposal_trace = parse_candidate_proposals(
        proposal_raw_output,
        task_id=task_id,
        n_line=len(observation.line_status),
        n_gen=len(observation.gen_p),
        redispatchable_generators=redispatchable_generators,
        redispatch_generators=redispatch_generators,
    )
    simulation_response = env.simulate_candidates(
        episode_id,
        [action for action, _ in proposal_candidates],
    )
    simulations = [
        SimulationOutcome(
            candidate_index=index,
            action=result.action,
            trace=proposal_candidates[index - 1][1],
            done=result.done,
            simulated_reward=result.simulated_reward,
            max_rho=result.max_rho,
            overloaded_line_ids=result.overloaded_line_ids,
            disconnected_lines=result.disconnected_lines,
            convergence_failed=result.convergence_failed,
            exceptions=result.exceptions,
            raw_result=result.raw_result,
        )
        for index, result in enumerate(simulation_response.results, start=1)
    ]
    logger.info(
        "Simulation results task_id=%s step=%s results=%s",
        task_id,
        step_count + 1,
        [
            {
                "candidate_index": outcome.candidate_index,
                "action": outcome.action.model_dump(),
                "max_rho": outcome.max_rho,
                "done": outcome.done,
                "convergence_failed": outcome.convergence_failed,
                "overloaded_line_ids": outcome.overloaded_line_ids,
                "exceptions": outcome.exceptions,
            }
            for outcome in simulations
        ],
    )

    final_prompt = build_final_selection_prompt(
        task_id=task_id,
        observation=observation,
        step_count=step_count,
        max_steps=max_steps,
        simulations=simulations,
    )
    final_response = client.chat.completions.create(
        model=llm_config.model,
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        presence_penalty=llm_config.presence_penalty,
        extra_body={
            "top_k": llm_config.top_k,
            "min_p": llm_config.min_p,
            "repetition_penalty": llm_config.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": llm_config.enable_thinking},
        },
    )
    final_raw_output = final_response.choices[0].message.content or ""
    logger.info(
        "Received final selection task_id=%s chars=%s content=%r",
        task_id,
        len(final_raw_output),
        final_raw_output,
    )
    action, final_trace = select_final_action(
        task_id=task_id,
        observation=observation,
        final_raw_output=final_raw_output,
        simulations=simulations,
        n_line=len(observation.line_status),
        n_gen=len(observation.gen_p),
    )
    return action, {
        "proposal_prompt": proposal_prompt,
        "proposal_raw_output": proposal_raw_output,
        "proposal_trace": proposal_trace,
        "graph_intelligence": graph_intelligence,
        "simulations": [serialize_simulation_outcome(outcome) for outcome in simulations],
        "final_prompt": final_prompt,
        "final_raw_output": final_raw_output,
        "final_trace": final_trace,
    }


def build_proposal_prompt(
    task_id: TaskId,
    observation: GridObservation,
    graph_intelligence: dict[str, Any],
    redispatchable_generators: Sequence[int],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
    step_count: int,
    max_steps: int,
    include_task_description: bool,
) -> str:
    line_count = len(observation.line_status)
    gen_count = len(observation.gen_p)
    stressed_lines = summarize_lines(observation.rho, limit=8, minimum_rho=0.7)
    sensitivity_guidance = observation.sensitivity_guidance[:3]
    disconnected = [
        {"line_id": idx, "status": "disconnected"}
        for idx, status in enumerate(observation.line_status)
        if not status
    ]
    generator_summary = summarize_generators(observation.gen_p, limit=6)
    lines = [
        "You are a grid operator proposing actions for a deterministic simulator.",
        "Propose exactly 3 candidate actions to test in the physics sandbox.",
        "Allowed action types: disconnect_line, reconnect_line, redispatch, do_nothing.",
        "Return a single JSON object only.",
        'Use this exact schema: {"candidates":[{"action_type":"disconnect_line|reconnect_line|redispatch|do_nothing","line_id":null|int,"gen_id":null|int,"delta_mw":null|float,"reason":"short string"}]}',
        "Rules: no markdown, no prose, no code fences, no extra keys, exactly 3 candidates.",
        "Diversity rule: use at least two different action types when plausible.",
        "CRITICAL PHYSICS RULE: You must prioritize candidates from the sensitivity_guidance list. These actions have been mathematically verified by power-flow sensitivity factors to reduce the load on the stressed line.",
        f"task_id={task_id}",
        f"step={step_count + 1}/{max_steps}",
        f"max_rho={max(observation.rho):.3f}" if observation.rho else "max_rho=0.0",
        f"line_count={line_count}",
        f"generator_count={gen_count}",
        "redispatchable_generators=" + json.dumps([int(x) for x in redispatchable_generators], separators=(",", ":")),
        "redispatch_generator_bounds="
        + json.dumps(
            [context.model_dump() for context in redispatch_generators],
            separators=(",", ":"),
        ),
        "stressed_lines=" + json.dumps(stressed_lines, separators=(",", ":")),
        "sensitivity_guidance=" + json.dumps(sensitivity_guidance, separators=(",", ":")),
        "disconnected_lines=" + json.dumps(disconnected, separators=(",", ":")),
        "generators=" + json.dumps(generator_summary, separators=(",", ":")),
        "timestep_overflow=" + json.dumps(observation.timestep_overflow, separators=(",", ":")),
        "grid_topology_intelligence=" + json.dumps(graph_intelligence, separators=(",", ":")),
    ]
    if task_id == "single_fault":
        lines.insert(
            6,
            "TASK RULE: For single_fault, do not propose disconnect_line or reconnect_line. Use redispatch and do_nothing only. Solve congestion by shifting generation, not by cutting topology.",
        )
    if include_task_description:
        lines.append("task_description=" + TASKS[task_id].description)
    lines.append(
        'example_1={"candidates":[{"action_type":"disconnect_line","line_id":10,"gen_id":null,"delta_mw":null,"reason":"line 10 appears to be the stress bottleneck"},{"action_type":"redispatch","line_id":null,"gen_id":0,"delta_mw":5.0,"reason":"pick one allowed positive delta from redispatch_generator_bounds"},{"action_type":"do_nothing","line_id":null,"gen_id":null,"delta_mw":null,"reason":"keep a safe fallback"}]}'
    )
    lines.append(
        'example_2={"candidates":[{"action_type":"reconnect_line","line_id":0,"gen_id":null,"delta_mw":null,"reason":"restore missing transfer capacity"},{"action_type":"redispatch","line_id":null,"gen_id":1,"delta_mw":-5.0,"reason":"pick one allowed negative delta from redispatch_generator_bounds"},{"action_type":"do_nothing","line_id":null,"gen_id":null,"delta_mw":null,"reason":"baseline if both interventions are dangerous"}]}'
    )
    return "\n".join(lines)


def build_final_selection_prompt(
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    simulations: Sequence[SimulationOutcome],
) -> str:
    lines = [
        "You are a grid operator choosing a final action after reviewing simulator outcomes.",
        "Select the safest candidate that reduces stress without ending the episode.",
        "You must select one simulated candidate or explicit do_nothing.",
        "Return a single JSON object only.",
        'Use this exact schema: {"selected_candidate":1|2|3|0,"reason":"short string"}',
        "Use selected_candidate=0 only if every candidate is unsafe.",
        "Rules: no markdown, no prose, no code fences, no extra keys.",
        f"task_id={task_id}",
        f"step={step_count + 1}/{max_steps}",
        f"current_max_rho={max(observation.rho):.3f}" if observation.rho else "current_max_rho=0.0",
        "simulation_results=" + json.dumps(
            [serialize_simulation_outcome(outcome) for outcome in simulations],
            separators=(",", ":"),
        ),
    ]
    if task_id == "single_fault":
        lines.insert(
            7,
            "RULE: If a simulated candidate safely reduces max_rho compared to the current state, you MUST select it over do_nothing, no matter how small the reduction is. Do not choose do_nothing unless every other candidate increases max_rho or causes a failure. Safe, incremental redispatch improvements are the only way to win.",
        )
    return "\n".join(lines)


def summarize_lines(rho: Sequence[float], limit: int, minimum_rho: float) -> list[dict[str, Any]]:
    return sorted(
        (
            {"line_id": idx, "rho": round(float(value), 4)}
            for idx, value in enumerate(rho)
            if float(value) >= minimum_rho
        ),
        key=lambda item: item["rho"],
        reverse=True,
    )[:limit]


def summarize_generators(gen_p: Sequence[float], limit: int) -> list[dict[str, Any]]:
    return sorted(
        (
            {"gen_id": idx, "p_mw": round(float(value), 4)}
            for idx, value in enumerate(gen_p)
        ),
        key=lambda item: abs(item["p_mw"]),
        reverse=True,
    )[:limit]


def parse_json_action(content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"do_nothing": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"do_nothing": True}


def parse_selected_candidate(content: str) -> int | None:
    match = re.search(r'"selected_candidate"\s*:\s*(-?\d+)', content)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def parse_candidate_proposals(
    content: str,
    n_line: int,
    n_gen: int,
    redispatchable_generators: Sequence[int] | None = None,
    redispatch_generators: Sequence[RedispatchGeneratorContext] | None = None,
    task_id: TaskId = "n_minus_1",
) -> tuple[list[tuple[GridAction, dict[str, Any]]], dict[str, Any]]:
    payload = parse_json_action(content)
    raw_candidates = payload.get("candidates", [])
    candidates: list[tuple[GridAction, dict[str, Any]]] = []
    if isinstance(raw_candidates, list):
        for item in raw_candidates[:3]:
            if isinstance(item, dict):
                candidates.append(
                    validate_baseline_action(
                        item,
                        task_id=task_id,
                        n_line=n_line,
                        n_gen=n_gen,
                        redispatchable_generators=redispatchable_generators,
                        redispatch_generators=redispatch_generators,
                    )
                )

    deduped: list[tuple[GridAction, dict[str, Any]]] = []
    seen: set[str] = set()
    for action, trace in candidates:
        key = json.dumps(action.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, trace))

    fallback_pool = [(GridAction(do_nothing=True), {"decision": "do_nothing", "reason": "fallback_baseline"})]
    if redispatch_generators:
        for context in redispatch_generators:
            for delta in context.allowed_deltas:
                if abs(delta) < 1e-9:
                    continue
                fallback_pool.append(
                    (
                        GridAction(redispatch={int(context.gen_id): float(delta)}),
                        {
                            "decision": "redispatch",
                            "reason": f"fallback_generator_{int(context.gen_id)}_{float(delta):.4f}",
                        },
                    )
                )
    else:
        fallback_pool.extend(
            [
                (GridAction(redispatch={0: 10.0}), {"decision": "redispatch", "reason": "fallback_generator_0_up"}),
                (GridAction(redispatch={0: -10.0}), {"decision": "redispatch", "reason": "fallback_generator_0_down"}),
            ]
        )
    for action, trace in fallback_pool:
        if len(deduped) >= 3:
            break
        key = json.dumps(action.model_dump(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((action, trace))

    return deduped[:3], {
        "parsed_candidate_count": len(candidates),
        "deduped_candidate_count": len(deduped[:3]),
    }


def select_final_action(
    final_raw_output: str,
    simulations: Sequence[SimulationOutcome],
    n_line: int,
    n_gen: int,
    task_id: TaskId = "n_minus_1",
    observation: GridObservation | None = None,
) -> tuple[GridAction, dict[str, Any]]:
    deterministic_best = choose_best_simulation(task_id, observation, simulations)
    payload = parse_json_action(final_raw_output)
    reason = payload.get("reason", "")
    selected_candidate = payload.get("selected_candidate")

    try:
        selected_candidate = int(selected_candidate)
    except (TypeError, ValueError):
        selected_candidate = parse_selected_candidate(final_raw_output)

    if selected_candidate == 0:
        return GridAction(do_nothing=True), {
            "decision": "do_nothing",
            "reason": reason or "model_rejected_all_candidates",
        }

    if selected_candidate is not None:
        for outcome in simulations:
            if outcome.candidate_index == selected_candidate:
                return outcome.action, {
                    "decision": "simulated_candidate_by_index",
                    "reason": reason or outcome.trace.get("reason", ""),
                    "selected_candidate": selected_candidate,
                }

    return deterministic_best.action, {
        "decision": "fallback_best_simulation",
        "reason": reason or "invalid_final_selection",
        "selected_candidate": deterministic_best.candidate_index,
    }


def choose_best_simulation(
    task_id: TaskId,
    observation: GridObservation | None,
    simulations: Sequence[SimulationOutcome],
) -> SimulationOutcome:
    safe = [
        outcome
        for outcome in simulations
        if not outcome.done and not outcome.convergence_failed
    ]
    if safe:
        current_max_rho = max(observation.rho) if observation and observation.rho else None
        if task_id == "single_fault":
            improving = [
                outcome
                for outcome in safe
                if current_max_rho is not None
                and outcome.max_rho < current_max_rho - 1e-9
                and not outcome.action.do_nothing
            ]
            if improving:
                return min(
                    improving,
                    key=lambda outcome: (
                        outcome.max_rho,
                        outcome.action.do_nothing,
                        len(outcome.overloaded_line_ids),
                        len(outcome.action.line_set),
                        outcome.candidate_index,
                    ),
                )
        return min(
            safe,
            key=lambda outcome: (
                outcome.max_rho,
                outcome.action.do_nothing,
                len(outcome.overloaded_line_ids),
                len(outcome.action.line_set),
                outcome.candidate_index,
            ),
        )
    nonfatal = [outcome for outcome in simulations if not outcome.convergence_failed]
    if nonfatal:
        return min(
            nonfatal,
            key=lambda outcome: (
                outcome.done,
                outcome.max_rho,
                outcome.action.do_nothing,
                len(outcome.action.line_set),
                outcome.candidate_index,
            ),
        )
    return min(simulations, key=lambda outcome: outcome.candidate_index)


def serialize_simulation_outcome(outcome: SimulationOutcome) -> dict[str, Any]:
    return {
        "candidate_index": outcome.candidate_index,
        "action": outcome.action.model_dump(),
        "proposal_trace": outcome.trace,
        "done": outcome.done,
        "simulated_reward": outcome.simulated_reward,
        "max_rho": outcome.max_rho,
        "overloaded_line_ids": outcome.overloaded_line_ids,
        "disconnected_lines": outcome.disconnected_lines,
        "convergence_failed": outcome.convergence_failed,
        "exceptions": outcome.exceptions,
        "raw_result": outcome.raw_result,
    }


def actions_equivalent(lhs: GridAction, rhs: GridAction) -> bool:
    return lhs.model_dump() == rhs.model_dump()


def validate_baseline_action(
    payload: Dict[str, Any],
    task_id: TaskId,
    n_line: int,
    n_gen: int,
    redispatchable_generators: Sequence[int] | None = None,
    redispatch_generators: Sequence[RedispatchGeneratorContext] | None = None,
) -> tuple[GridAction, dict[str, Any]]:
    allowed_redispatch = (
        {int(gen_id) for gen_id in redispatchable_generators}
        if redispatchable_generators is not None
        else set(range(n_gen))
    )
    redispatch_context_by_id = (
        {int(context.gen_id): context for context in redispatch_generators}
        if redispatch_generators is not None
        else {}
    )
    action_type = payload.get("action_type")
    if payload.get("do_nothing") or action_type == "do_nothing":
        return GridAction(do_nothing=True), {"decision": "do_nothing", "reason": payload.get("reason", "explicit_do_nothing")}

    if task_id == "single_fault" and action_type in {"disconnect_line", "reconnect_line"}:
        return GridAction(do_nothing=True), {
            "decision": "fallback_do_nothing",
            "reason": f"{action_type}_forbidden_for_single_fault",
        }

    line_set_payload = payload.get("line_set") or {}
    if not isinstance(line_set_payload, dict):
        line_set_payload = {}

    valid_line_set = {}
    for key, value in line_set_payload.items():
        try:
            line_id = int(key)
            status = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= line_id < n_line and status in (-1, 1):
            valid_line_set[line_id] = status

    redispatch_payload = payload.get("redispatch") or {}
    if not isinstance(redispatch_payload, dict):
        redispatch_payload = {}

    if action_type == "disconnect_line":
        line_id = payload.get("line_id")
        try:
            line_id = int(line_id)
        except (TypeError, ValueError):
            line_id = None
        if line_id is not None and 0 <= line_id < n_line:
            return (
                GridAction(line_set={line_id: -1}, redispatch={}),
                {"decision": "disconnect_line", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_disconnect_line"}

    if action_type == "reconnect_line":
        line_id = payload.get("line_id")
        try:
            line_id = int(line_id)
        except (TypeError, ValueError):
            line_id = None
        if line_id is not None and 0 <= line_id < n_line:
            return (
                GridAction(line_set={line_id: 1}, redispatch={}),
                {"decision": "reconnect_line", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_reconnect_line"}

    if action_type == "redispatch":
        gen_id = payload.get("gen_id")
        delta = payload.get("delta_mw")
        try:
            gen_id = int(gen_id)
            delta = float(delta)
        except (TypeError, ValueError):
            gen_id = None
            delta = None
        if (
            gen_id is not None
            and 0 <= gen_id < n_gen
            and gen_id in allowed_redispatch
            and delta is not None
        ):
            if gen_id in redispatch_context_by_id:
                delta = constrain_redispatch_delta(delta, redispatch_context_by_id[gen_id])
                if delta is None:
                    return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_redispatch"}
            return (
                GridAction(redispatch={gen_id: delta}, line_set={}),
                {"decision": "redispatch", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_redispatch"}

    valid_redispatch = {}
    for key, value in redispatch_payload.items():
        try:
            gen_id = int(key)
            delta = float(value)
        except (TypeError, ValueError):
            continue
        if 0 <= gen_id < n_gen and gen_id in allowed_redispatch:
            if gen_id in redispatch_context_by_id:
                delta = constrain_redispatch_delta(delta, redispatch_context_by_id[gen_id])
                if delta is None:
                    continue
            valid_redispatch[gen_id] = delta

    if not valid_line_set and not valid_redispatch:
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "empty_or_invalid_payload"}
    return (
        GridAction(line_set=valid_line_set, redispatch=valid_redispatch),
        {"decision": "legacy_payload", "reason": payload.get("reason", "")},
    )


def constrain_redispatch_delta(
    delta: float,
    context: RedispatchGeneratorContext,
) -> float | None:
    if context.allowed_delta_min == 0.0 and context.allowed_delta_max == 0.0:
        return None
    clamped = min(max(float(delta), float(context.allowed_delta_min)), float(context.allowed_delta_max))
    feasible = [
        float(value)
        for value in context.allowed_deltas
        if float(context.allowed_delta_min) <= float(value) <= float(context.allowed_delta_max)
    ]
    if feasible:
        return min(feasible, key=lambda value: (abs(value - clamped), abs(value), value))
    if abs(clamped) < 1e-9:
        return None
    return round(clamped, 4)


def write_evaluation_outputs(
    timestamp: str,
    run_paths: dict[str, Path],
    model: str,
    base_url: str,
    llm_config: BaselineConfig,
    baseline_scores: BaselineScores,
    evaluation_records: list[dict[str, Any]],
    selected_task_ids: Sequence[TaskId],
) -> None:
    json_path = run_paths["json"]
    csv_path = run_paths["csv"]

    aggregate: dict[str, Any] = {}
    for task_id in selected_task_ids:
        records = [record for record in evaluation_records if record["task_id"] == task_id]
        scores = [float(record["score"]) for record in records]
        lengths = [int(record["episode_length"]) for record in records]
        do_nothing = [int(record["do_nothing_steps"]) for record in records]
        aggregate[task_id] = {
            "num_episodes": len(records),
            "score_mean": round(mean(scores), 6) if scores else 0.0,
            "score_std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
            "episode_length_mean": round(mean(lengths), 6) if lengths else 0.0,
            "episode_length_std": round(pstdev(lengths), 6) if len(lengths) > 1 else 0.0,
            "do_nothing_steps_mean": round(mean(do_nothing), 6) if do_nothing else 0.0,
        }
    aggregate_by_tier: dict[str, Any] = {}
    for benchmark_tier in sorted({record["benchmark_tier"] for record in evaluation_records}):
        records = [record for record in evaluation_records if record["benchmark_tier"] == benchmark_tier]
        scores = [float(record["score"]) for record in records]
        lengths = [int(record["episode_length"]) for record in records]
        do_nothing = [int(record["do_nothing_steps"]) for record in records]
        aggregate_by_tier[benchmark_tier] = {
            "num_episodes": len(records),
            "score_mean": round(mean(scores), 6) if scores else 0.0,
            "score_std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
            "episode_length_mean": round(mean(lengths), 6) if lengths else 0.0,
            "do_nothing_steps_mean": round(mean(do_nothing), 6) if do_nothing else 0.0,
        }

    payload = {
        "timestamp": timestamp,
        "model": model,
        "base_url": base_url,
        "sampling": {
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "top_k": llm_config.top_k,
            "min_p": llm_config.min_p,
            "presence_penalty": llm_config.presence_penalty,
            "repetition_penalty": llm_config.repetition_penalty,
            "enable_thinking": llm_config.enable_thinking,
            "max_tokens": llm_config.max_tokens,
            "num_seeds": llm_config.num_seeds,
            "seed_start": llm_config.seed_start,
            "scenario_mode": llm_config.scenario_mode,
        },
        "summary": baseline_scores.model_dump(),
        "aggregate": aggregate,
        "aggregate_by_tier": aggregate_by_tier,
        "episodes": evaluation_records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "seed",
                "curriculum_episode",
                "benchmark_tier",
                "score",
                "episode_length",
                "done",
                "do_nothing_steps",
                "non_do_nothing_steps",
            ],
        )
        writer.writeheader()
        for record in evaluation_records:
            writer.writerow(
                {
                    "task_id": record["task_id"],
                    "seed": record["seed"],
                    "curriculum_episode": record["curriculum_episode"],
                    "benchmark_tier": record["benchmark_tier"],
                    "score": record["score"],
                    "episode_length": record["episode_length"],
                    "done": record["done"],
                    "do_nothing_steps": record["do_nothing_steps"],
                    "non_do_nothing_steps": record["non_do_nothing_steps"],
                }
            )
    logger.info("Wrote evaluation outputs json=%s csv=%s", json_path, csv_path)


def prepare_run_paths(timestamp: str) -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parent
    eval_dir = base_dir / "outputs" / "evals"
    log_dir = base_dir / "outputs" / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": eval_dir / f"baseline_eval_{timestamp}.json",
        "csv": eval_dir / f"baseline_eval_{timestamp}.csv",
        "log": log_dir / f"baseline_run_{timestamp}.log",
    }


def attach_file_logger(log_path: Path) -> None:
    root_logger = logging.getLogger()
    if any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in root_logger.handlers
    ):
        return
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    logger.info("Attached file logger log_path=%s", log_path)


def append_evaluation_markdown(
    timestamp: str,
    model: str,
    llm_config: BaselineConfig,
    baseline_scores: BaselineScores,
    evaluation_records: list[dict[str, Any]],
    run_paths: dict[str, Path],
    selected_task_ids: Sequence[TaskId],
) -> None:
    eval_md = Path(__file__).resolve().parent.parent / "evaluation.md"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in evaluation_records:
        grouped.setdefault(record["task_id"], []).append(record)

    lines = [
        f"## Run {timestamp}",
        "",
        f"- Model: `{model}`",
        f"- Tasks: `{', '.join(selected_task_ids)}`",
        f"- Seeds: `{llm_config.seed_start}` to `{llm_config.seed_start + llm_config.num_seeds - 1}`",
        f"- Scenario mode: `{llm_config.scenario_mode}`",
        f"- Sampling: `temperature={llm_config.temperature}`, `top_p={llm_config.top_p}`, `top_k={llm_config.top_k}`, `min_p={llm_config.min_p}`, `presence_penalty={llm_config.presence_penalty}`, `repetition_penalty={llm_config.repetition_penalty}`",
        f"- JSON output: [{run_paths['json']}]({run_paths['json']})",
        f"- CSV output: [{run_paths['csv']}]({run_paths['csv']})",
        f"- Log file: [{run_paths['log']}]({run_paths['log']})",
        "",
        "| Task | Tier | Mean Score | Mean Episode Length | Mean Do-Nothing Steps |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for task_id, records in grouped.items():
        tier_groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            tier_groups.setdefault(record["benchmark_tier"], []).append(record)
        for benchmark_tier, tier_records in tier_groups.items():
            scores = [float(record["score"]) for record in tier_records]
            lengths = [int(record["episode_length"]) for record in tier_records]
            do_nothing = [int(record["do_nothing_steps"]) for record in tier_records]
            lines.append(
                f"| `{task_id}` | `{benchmark_tier}` | `{mean(scores):.6f}` | `{mean(lengths):.2f}` | `{mean(do_nothing):.2f}` |"
            )

    lines.extend(
        [
            "",
            "Summary scores:",
            f"```json\n{baseline_scores.model_dump_json(indent=2)}\n```",
            "",
        ]
    )

    with eval_md.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        nargs="+",
        choices=sorted(TASKS.keys()),
        help="Run only the selected task ids. Defaults to all tasks.",
    )
    args = parser.parse_args()

    base_url = os.environ.get("GRID2OP_BASE_URL", "http://127.0.0.1:7860")
    result = run_baseline_suite(base_url=base_url, task_ids=args.task_ids)
    print(result.model_dump_json(indent=2))
