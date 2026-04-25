from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from grid2op_env import GridAction, GridEnv
from grid2op_env.inference import SimulationOutcome, filter_selectable_simulations, serialize_simulation_outcome
from grid2op_env.models import TaskId
from grid2op_env.server.tasks import TASKS, benchmark_tiers_for_task

try:
    from scripts.collect_teacher_dataset import build_sft_prompt
    from scripts.diagnose_action_space import _reset_and_replay
    from scripts.task_objectives import objective_completion_score
except ImportError:
    from collect_teacher_dataset import build_sft_prompt
    from diagnose_action_space import _reset_and_replay
    from task_objectives import objective_completion_score

try:
    from ft_inference import (
        FineTunedPolicy,
        action_key,
        choose_ft_action,
        generate_legal_candidates,
        rank_simulations,
    )
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from ft_inference import (  # type: ignore[no-redef]
        FineTunedPolicy,
        action_key,
        choose_ft_action,
        generate_legal_candidates,
        rank_simulations,
    )


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _log_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _simulation_map(simulations: Sequence[SimulationOutcome]) -> dict[str, SimulationOutcome]:
    return {action_key(outcome.action): outcome for outcome in simulations}


def _serialize_with_lookahead(
    simulations: Sequence[SimulationOutcome],
    lookahead_values: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in simulations:
        payload = serialize_simulation_outcome(outcome)
        payload["lookahead_value"] = round(float(lookahead_values.get(action_key(outcome.action), 0.0)), 6)
        rows.append(payload)
    return rows


def _observation_summary_for_rollout(task_id: TaskId, step: int, observation: Any) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "step": step,
        "max_rho": max((float(value) for value in observation.rho), default=0.0),
        "disconnected_lines": [
            line_id for line_id, status in enumerate(observation.line_status) if not bool(status)
        ],
        "timestep_overflow": [int(value) for value in observation.timestep_overflow],
        "sensitivity_guidance": observation.sensitivity_guidance[:5],
    }


def _rollout_value(
    replay_env: GridEnv,
    *,
    policy: FineTunedPolicy,
    task_id: TaskId,
    seed: int,
    benchmark_tier: str,
    pre_actions: Sequence[GridAction],
    first_action: GridAction,
    lookahead_horizon: int,
    candidate_count: int,
) -> float:
    result, state = _reset_and_replay(
        replay_env,
        task_id=task_id,
        seed=seed,
        benchmark_tier=benchmark_tier,
        pre_actions=pre_actions,
    )
    if result.done:
        return 0.0

    cumulative_reward = 0.0
    result = replay_env.step(first_action)
    state = replay_env.state()
    cumulative_reward += float(result.reward)
    if result.done:
        return cumulative_reward

    for step_idx in range(1, lookahead_horizon):
        next_action, _trace = choose_ft_action(
            policy=policy,
            env=replay_env,
            episode_id=state.episode_id,
            task_id=task_id,
            observation=result.observation,
            step=state.step_count,
            max_steps=TASKS[task_id].max_steps,
            candidate_count=candidate_count,
        )
        result = replay_env.step(next_action)
        state = replay_env.state()
        cumulative_reward += float(result.reward)
        if result.done:
            break
    return cumulative_reward


def collect_rollout_dataset(
    *,
    base_url: str,
    output_path: Path,
    model: str,
    adapter: Path | None,
    task_ids: Sequence[TaskId],
    episodes_per_task: int,
    seed_start: int,
    prompt_candidate_count: int,
    lookahead_first_actions: int,
    lookahead_horizon: int,
    min_advantage: float,
    precision: str,
    use_4bit: bool,
    max_new_tokens: int,
    attn_implementation: str | None,
) -> dict[str, Any]:
    policy = FineTunedPolicy(
        base_model=model,
        adapter=adapter,
        precision=precision,
        use_4bit=use_4bit,
        attn_implementation=attn_implementation,
        max_new_tokens=max_new_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "adapter": None if adapter is None else str(adapter),
        "output_path": str(output_path),
        "episodes": 0,
        "rows_written": 0,
        "skipped_small_gap": 0,
        "skipped_policy_best": 0,
        "tasks": {},
    }
    started_at = perf_counter()

    with GridEnv(base_url=base_url).sync() as main_env, output_path.open(
        "a", encoding="utf-8"
    ) as handle:
        for task_id in task_ids:
            benchmark_tiers = benchmark_tiers_for_task(task_id)
            stats["tasks"][task_id] = {
                "episodes": 0,
                "rows": 0,
                "skipped_small_gap": 0,
                "skipped_policy_best": 0,
            }
            for episode_index in range(episodes_per_task):
                seed = seed_start + episode_index
                benchmark_tier = benchmark_tiers[episode_index % len(benchmark_tiers)]
                _log_event(
                    {
                        "event": "episode_start",
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "seed": seed,
                        "benchmark_tier": benchmark_tier,
                    }
                )
                result = main_env.reset(
                    task_id=task_id,
                    seed=seed,
                    difficulty_level=episode_index + 1,
                    scenario_mode="benchmark",
                    benchmark_tier=benchmark_tier,
                )
                state = main_env.state()
                history: list[GridAction] = []
                stats["episodes"] += 1
                stats["tasks"][task_id]["episodes"] += 1

                while not result.done:
                    current_step = int(state.step_count)
                    current_max_rho = max((float(value) for value in result.observation.rho), default=0.0)
                    _log_event(
                        {
                            "event": "step_start",
                            "task_id": task_id,
                            "seed": seed,
                            "benchmark_tier": benchmark_tier,
                            "step": current_step,
                            "current_max_rho": round(current_max_rho, 6),
                            "rows_written_so_far": stats["rows_written"],
                        }
                    )
                    planning_context = main_env.planning_context(state.episode_id)
                    candidate_actions = generate_legal_candidates(
                        task_id=task_id,
                        observation=result.observation,
                        graph_intelligence=planning_context.graph_intelligence,
                        redispatch_generators=planning_context.redispatch_generators,
                    )
                    simulation_response = main_env.simulate_candidates(state.episode_id, candidate_actions)
                    simulations = [
                        SimulationOutcome(
                            candidate_index=index,
                            action=sim_result.action,
                            trace={"source": "rollout_enumeration"},
                            done=sim_result.done,
                            simulated_reward=sim_result.simulated_reward,
                            max_rho=sim_result.max_rho,
                            overloaded_line_ids=sim_result.overloaded_line_ids,
                            disconnected_lines=sim_result.disconnected_lines,
                            convergence_failed=sim_result.convergence_failed,
                            exceptions=sim_result.exceptions,
                            raw_result=sim_result.raw_result,
                        )
                        for index, sim_result in enumerate(simulation_response.results, start=1)
                    ]
                    selectable = filter_selectable_simulations(simulations)
                    if not selectable:
                        _log_event(
                            {
                                "event": "step_no_selectable",
                                "task_id": task_id,
                                "seed": seed,
                                "step": current_step,
                                "candidate_count": len(simulations),
                            }
                        )
                        break

                    policy_action, trace = choose_ft_action(
                        policy=policy,
                        env=main_env,
                        episode_id=state.episode_id,
                        task_id=task_id,
                        observation=result.observation,
                        step=state.step_count,
                        max_steps=TASKS[task_id].max_steps,
                        candidate_count=prompt_candidate_count,
                    )
                    simulation_by_key = _simulation_map(simulations)
                    policy_sim = simulation_by_key.get(action_key(policy_action))
                    if policy_sim is None:
                        raise ValueError("Policy selected action missing from enumerated simulations")

                    shortlist = rank_simulations(task_id, result.observation, simulations)[:lookahead_first_actions]
                    shortlist_by_key = {action_key(outcome.action): outcome for outcome in shortlist}
                    shortlist_by_key[action_key(policy_sim.action)] = policy_sim
                    for outcome in simulations:
                        if outcome.action.do_nothing:
                            shortlist_by_key[action_key(outcome.action)] = outcome
                            break
                    shortlisted = list(shortlist_by_key.values())
                    observation_summary = _observation_summary_for_rollout(
                        task_id=task_id,
                        step=current_step,
                        observation=result.observation,
                    )
                    _log_event(
                        {
                            "event": "step_shortlist",
                            "task_id": task_id,
                            "seed": seed,
                            "step": current_step,
                            "candidate_count": len(simulations),
                            "shortlist_count": len(shortlisted),
                            "policy_action": policy_action.model_dump(),
                        }
                    )

                    lookahead_values = {
                        action_key(outcome.action): _rollout_value(
                            main_env,
                            policy=policy,
                            task_id=task_id,
                            seed=seed,
                            benchmark_tier=benchmark_tier,
                            pre_actions=history,
                            first_action=outcome.action,
                            lookahead_horizon=lookahead_horizon,
                            candidate_count=prompt_candidate_count,
                        )
                        for outcome in shortlisted
                    }
                    serialized_shortlisted = _serialize_with_lookahead(shortlisted, lookahead_values)
                    best_payload = max(
                        serialized_shortlisted,
                        key=lambda payload: objective_completion_score(
                            task_id=task_id,
                            simulation=payload,
                            simulations=serialized_shortlisted,
                            observation_summary=observation_summary,
                        ),
                    )
                    best_outcome = shortlist_by_key[action_key(GridAction.model_validate(best_payload["action"]))]
                    policy_payload = next(
                        payload
                        for payload in serialized_shortlisted
                        if action_key(GridAction.model_validate(payload["action"])) == action_key(policy_sim.action)
                    )
                    best_value = objective_completion_score(
                        task_id=task_id,
                        simulation=best_payload,
                        simulations=serialized_shortlisted,
                        observation_summary=observation_summary,
                    )
                    policy_value = objective_completion_score(
                        task_id=task_id,
                        simulation=policy_payload,
                        simulations=serialized_shortlisted,
                        observation_summary=observation_summary,
                    )
                    _log_event(
                        {
                            "event": "step_lookahead_scored",
                            "task_id": task_id,
                            "seed": seed,
                            "step": current_step,
                            "policy_action": policy_action.model_dump(),
                            "policy_value": round(policy_value, 6),
                            "best_action": best_outcome.action.model_dump(),
                            "best_value": round(best_value, 6),
                            "advantage": round(best_value - policy_value, 6),
                        }
                    )

                    result, state = _reset_and_replay(
                        main_env,
                        task_id=task_id,
                        seed=seed,
                        benchmark_tier=benchmark_tier,
                        pre_actions=history,
                    )
                    if result.done:
                        break

                    if action_key(best_outcome.action) == action_key(policy_sim.action):
                        stats["skipped_policy_best"] += 1
                        stats["tasks"][task_id]["skipped_policy_best"] += 1
                        _log_event(
                            {
                                "event": "step_skip_policy_best",
                                "task_id": task_id,
                                "seed": seed,
                                "step": current_step,
                                "policy_value": round(policy_value, 6),
                            }
                        )
                    elif best_value - policy_value < min_advantage:
                        stats["skipped_small_gap"] += 1
                        stats["tasks"][task_id]["skipped_small_gap"] += 1
                        _log_event(
                            {
                                "event": "step_skip_small_gap",
                                "task_id": task_id,
                                "seed": seed,
                                "step": current_step,
                                "advantage": round(best_value - policy_value, 6),
                                "min_advantage": min_advantage,
                            }
                        )
                    else:
                        prompt = build_sft_prompt(
                            task_id=task_id,
                            step=state.step_count,
                            max_steps=TASKS[task_id].max_steps,
                            observation=result.observation,
                            graph_intelligence=planning_context.graph_intelligence,
                            redispatch_generators=planning_context.redispatch_generators,
                            simulations=shortlisted,
                        )
                        row = {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": _json_dumps(best_outcome.action.model_dump())},
                            ],
                            "metadata": {
                                "task_id": task_id,
                                "benchmark_tier": benchmark_tier,
                                "seed": seed,
                                "step": state.step_count,
                                "observation_summary": observation_summary,
                                "simulations": serialized_shortlisted,
                                "selected_action": best_outcome.action.model_dump(),
                                "policy_action": policy_action.model_dump(),
                                "policy_action_lookahead_value": round(policy_value, 6),
                                "selected_action_lookahead_value": round(best_value, 6),
                                "lookahead_advantage": round(best_value - policy_value, 6),
                                "label_policy": "rollout_short_horizon",
                                "lookahead_horizon": lookahead_horizon,
                            },
                        }
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        stats["rows_written"] += 1
                        stats["tasks"][task_id]["rows"] += 1
                        _log_event(
                            {
                                "event": "row_written",
                                "task_id": task_id,
                                "seed": seed,
                                "step": current_step,
                                "rows_written": stats["rows_written"],
                                "advantage": round(best_value - policy_value, 6),
                                "selected_action": best_outcome.action.model_dump(),
                                "policy_action": policy_action.model_dump(),
                            }
                        )

                    result = main_env.step(policy_action)
                    state = main_env.state()
                    history.append(policy_action)

                _log_event(
                    {
                        "event": "episode_end",
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "seed": seed,
                        "rows_written_so_far": stats["rows_written"],
                        "task_rows_written": stats["tasks"][task_id]["rows"],
                        "task_skipped_policy_best": stats["tasks"][task_id]["skipped_policy_best"],
                        "task_skipped_small_gap": stats["tasks"][task_id]["skipped_small_gap"],
                    }
                )

    stats["wall_time_s"] = round(perf_counter() - started_at, 6)
    _log_event({"event": "collection_complete", **stats})
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect rollout-derived hard states for GRPO from the current policy."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/models/grid2op-qwen3-4b-sft-3k-v1"))
    parser.add_argument("--task-id", dest="task_ids", nargs="+", choices=list(TASKS), default=["multi_stage_cascade"])
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--prompt-candidate-count", type=int, default=3)
    parser.add_argument("--lookahead-first-actions", type=int, default=5)
    parser.add_argument("--lookahead-horizon", type=int, default=3)
    parser.add_argument("--min-advantage", type=float, default=0.5)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--attn-implementation", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = collect_rollout_dataset(
        base_url=args.base_url,
        output_path=args.output_path,
        model=args.model,
        adapter=None if str(args.adapter).lower() == "none" else args.adapter,
        task_ids=args.task_ids,
        episodes_per_task=args.episodes_per_task,
        seed_start=args.seed_start,
        prompt_candidate_count=args.prompt_candidate_count,
        lookahead_first_actions=args.lookahead_first_actions,
        lookahead_horizon=args.lookahead_horizon,
        min_advantage=args.min_advantage,
        precision=args.precision,
        use_4bit=args.use_4bit,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
