from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import combinations, product
import json
from typing import Any

from grid2op_env import GridAction, GridEnv
from grid2op_env.models import GridObservation, RedispatchGeneratorContext, TaskId


SINGLE_FAULT_TARGET_RHO = 0.80


def _json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _max_rho(observation: GridObservation) -> float:
    return max((float(value) for value in observation.rho), default=0.0)


def _disconnected_lines(observation: GridObservation) -> list[int]:
    return [
        line_id
        for line_id, status in enumerate(observation.line_status)
        if not bool(status)
    ]


def _overflow_lines(observation: GridObservation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_id, rho in enumerate(observation.rho):
        overflow = (
            int(observation.timestep_overflow[line_id])
            if line_id < len(observation.timestep_overflow)
            else 0
        )
        if float(rho) > 1.0 or overflow > 0:
            rows.append(
                {
                    "line_id": line_id,
                    "rho": round(float(rho), 6),
                    "timestep_overflow": overflow,
                }
            )
    return sorted(rows, key=lambda item: (item["timestep_overflow"], item["rho"]), reverse=True)


def _action_key(action: GridAction) -> str:
    return _json(action.model_dump())


def _candidate_actions(
    task_id: TaskId,
    observation: GridObservation,
    graph_intelligence: dict[str, Any],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
    *,
    include_compound_redispatch: bool,
    max_compound_redispatch_size: int,
) -> list[GridAction]:
    actions: list[GridAction] = [GridAction(do_nothing=True)]

    for context in redispatch_generators:
        for delta in context.allowed_deltas:
            delta_value = float(delta)
            if abs(delta_value) < 1e-9:
                continue
            actions.append(GridAction(redispatch={int(context.gen_id): delta_value}))

    if include_compound_redispatch:
        actions.extend(
            _compound_redispatch_actions(
                redispatch_generators,
                max_compound_redispatch_size=max_compound_redispatch_size,
            )
        )

    if task_id in {"n_minus_1", "cascade_prevent", "multi_stage_cascade"}:
        for line_id in _disconnected_lines(observation):
            actions.append(GridAction(line_set={line_id: 1}))

    if task_id in {"cascade_prevent", "multi_stage_cascade"}:
        safe_to_disconnect = {
            int(line_id) for line_id in graph_intelligence.get("safe_to_disconnect", [])
        }
        blocked_disconnect = {
            int(line_id) for line_id in graph_intelligence.get("bridge_lines", [])
        } | {
            int(line_id)
            for line_id in graph_intelligence.get("n_minus_1_critical_lines", [])
        }
        for line_id, status in enumerate(observation.line_status):
            if bool(status) and line_id in safe_to_disconnect - blocked_disconnect:
                actions.append(GridAction(line_set={line_id: -1}))

    deduped: list[GridAction] = []
    seen: set[str] = set()
    for action in actions:
        key = _action_key(action)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _compound_redispatch_actions(
    redispatch_generators: Sequence[RedispatchGeneratorContext],
    *,
    max_compound_redispatch_size: int,
) -> list[GridAction]:
    actions: list[GridAction] = []
    usable = [
        context
        for context in redispatch_generators
        if any(abs(float(delta)) > 1e-9 for delta in context.allowed_deltas)
    ]
    max_size = max(2, min(max_compound_redispatch_size, len(usable)))
    for size in range(2, max_size + 1):
        for contexts in combinations(usable, size):
            delta_options = [
                [
                    float(delta)
                    for delta in context.allowed_deltas
                    if abs(float(delta)) > 1e-9
                ]
                for context in contexts
            ]
            for deltas in product(*delta_options):
                actions.append(
                    GridAction(
                        redispatch={
                            int(context.gen_id): float(delta)
                            for context, delta in zip(contexts, deltas, strict=True)
                        }
                    )
                )
    return actions


def _action_label(action: GridAction) -> str:
    if action.do_nothing:
        return "do_nothing"
    if action.redispatch:
        return "redispatch " + _json(action.redispatch)
    if action.line_set:
        return "line_set " + _json(action.line_set)
    return "empty_action"


def _row_from_simulation(index: int, simulation: Any) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "action": simulation.action.model_dump(),
        "action_label": _action_label(simulation.action),
        "done": simulation.done,
        "convergence_failed": simulation.convergence_failed,
        "simulated_reward": round(float(simulation.simulated_reward), 6),
        "max_rho": round(float(simulation.max_rho), 6),
        "target_reached": float(simulation.max_rho) < SINGLE_FAULT_TARGET_RHO,
        "overloaded_line_ids": simulation.overloaded_line_ids,
        "disconnected_lines": simulation.disconnected_lines,
        "exceptions": simulation.exceptions,
        "timestep_overflow": simulation.raw_result.get("timestep_overflow", []),
    }


def _survivors(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not row["done"] and not row["convergence_failed"]
    ]


def _sort_by_single_fault_objective(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get("target_reached", False)),
            len(row["overloaded_line_ids"]),
            float(row["max_rho"]),
            -float(row["simulated_reward"]),
            len(row["disconnected_lines"]),
            row["candidate_index"],
        ),
    )


def _reset_and_replay(
    env: GridEnv,
    *,
    task_id: TaskId,
    seed: int,
    benchmark_tier: str,
    pre_actions: Sequence[GridAction],
) -> tuple[Any, Any]:
    result = env.reset(
        task_id=task_id,
        seed=seed,
        difficulty_level=1,
        scenario_mode="benchmark",
        benchmark_tier=benchmark_tier,
    )
    state = env.state()
    for action in pre_actions:
        result = env.step(action)
        state = env.state()
        if result.done:
            break
    return result, state


def _simulate_rows(
    env: GridEnv,
    episode_id: str,
    actions: Sequence[GridAction],
    *,
    batch_size: int,
    event_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(actions), batch_size):
        batch = list(actions[start : start + batch_size])
        simulation_response = env.simulate_candidates(episode_id, batch)
        for offset, simulation in enumerate(simulation_response.results, start=start + 1):
            row = _row_from_simulation(offset, simulation)
            rows.append(row)
            print(
                _json(
                    {
                        "event": f"{event_prefix}_candidate_result",
                        **row,
                    }
                ),
                flush=True,
            )
        print(
            _json(
                {
                    "event": f"{event_prefix}_simulate_batch_done",
                    "completed": min(start + batch_size, len(actions)),
                    "candidate_count": len(actions),
                }
            ),
            flush=True,
        )
    return rows


def run_diagnostic(
    *,
    base_url: str,
    task_id: TaskId,
    seed: int,
    benchmark_tier: str,
    pre_actions: list[GridAction],
    top_n: int,
    batch_size: int,
    include_compound_redispatch: bool,
    max_compound_redispatch_size: int,
    lookahead_depth: int,
    lookahead_first_actions: int,
) -> None:
    with GridEnv(base_url=base_url).sync() as env:
        print(
            _json(
                {
                    "event": "reset_start",
                    "base_url": base_url,
                    "task_id": task_id,
                    "seed": seed,
                    "benchmark_tier": benchmark_tier,
                }
            ),
            flush=True,
        )
        result = env.reset(
            task_id=task_id,
            seed=seed,
            difficulty_level=1,
            scenario_mode="benchmark",
            benchmark_tier=benchmark_tier,
        )
        state = env.state()
        print(
            _json(
                {
                    "event": "reset_done",
                    "episode_id": state.episode_id,
                    "max_rho": round(_max_rho(result.observation), 6),
                    "overflow_lines": _overflow_lines(result.observation),
                    "disconnected_lines": _disconnected_lines(result.observation),
                }
            ),
            flush=True,
        )

        executed_pre_actions: list[dict[str, Any]] = []
        for index, action in enumerate(pre_actions, start=1):
            result = env.step(action)
            state = env.state()
            executed_pre_actions.append(
                {
                    "index": index,
                    "action": action.model_dump(),
                    "reward": result.reward,
                    "done": result.done,
                    "max_rho": round(_max_rho(result.observation), 6),
                    "overflow_lines": _overflow_lines(result.observation),
                    "disconnected_lines": _disconnected_lines(result.observation),
                }
            )
            print(
                _json(
                    {
                        "event": "pre_action_done",
                        "index": index,
                        "action": action.model_dump(),
                        "done": result.done,
                        "max_rho": round(_max_rho(result.observation), 6),
                        "overflow_lines": _overflow_lines(result.observation),
                        "disconnected_lines": _disconnected_lines(result.observation),
                    }
                ),
                flush=True,
            )
            if result.done:
                break

        print(_json({"event": "planning_context_start"}), flush=True)
        planning_context = env.planning_context(state.episode_id)
        graph_intelligence = planning_context.graph_intelligence
        actions = _candidate_actions(
            task_id,
            result.observation,
            graph_intelligence,
            planning_context.redispatch_generators,
            include_compound_redispatch=include_compound_redispatch,
            max_compound_redispatch_size=max_compound_redispatch_size,
        )
        print(
            _json(
                {
                    "event": "simulate_start",
                    "candidate_count": len(actions),
                    "batch_size": batch_size,
                    "include_compound_redispatch": include_compound_redispatch,
                    "max_compound_redispatch_size": max_compound_redispatch_size,
                }
            ),
            flush=True,
        )

        rows = _simulate_rows(
            env,
            state.episode_id,
            actions,
            batch_size=batch_size,
            event_prefix="one_step",
        )

        survivors = _survivors(rows)
        survivors_sorted = _sort_by_single_fault_objective(survivors)
        failures = [row for row in rows if row["done"] or row["convergence_failed"]]
        lookahead_rows = []
        if lookahead_depth >= 2:
            first_actions = [
                GridAction.model_validate(row["action"])
                for row in survivors_sorted[:lookahead_first_actions]
            ]
            print(
                _json(
                    {
                        "event": "lookahead_start",
                        "depth": 2,
                        "first_action_count": len(first_actions),
                    }
                ),
                flush=True,
            )
            for first_index, first_action in enumerate(first_actions, start=1):
                replay_result, replay_state = _reset_and_replay(
                    env,
                    task_id=task_id,
                    seed=seed,
                    benchmark_tier=benchmark_tier,
                    pre_actions=[*pre_actions, first_action],
                )
                if replay_result.done:
                    continue
                replay_context = env.planning_context(replay_state.episode_id)
                second_actions = _candidate_actions(
                    task_id,
                    replay_result.observation,
                    replay_context.graph_intelligence,
                    replay_context.redispatch_generators,
                    include_compound_redispatch=include_compound_redispatch,
                    max_compound_redispatch_size=max_compound_redispatch_size,
                )
                second_rows = _simulate_rows(
                    env,
                    replay_state.episode_id,
                    second_actions,
                    batch_size=batch_size,
                    event_prefix=f"lookahead_{first_index}",
                )
                for row in second_rows:
                    row["first_action_index"] = first_index
                    row["first_action"] = first_action.model_dump()
                    row["first_step_max_rho"] = round(_max_rho(replay_result.observation), 6)
                lookahead_rows.extend(second_rows)

        lookahead_survivors_sorted = _sort_by_single_fault_objective(
            _survivors(lookahead_rows)
        )

        report = {
            "base_url": base_url,
            "task_id": task_id,
            "seed": seed,
            "benchmark_tier": benchmark_tier,
            "episode_id": state.episode_id,
            "pre_actions": executed_pre_actions,
            "current_state": {
                "max_rho": round(_max_rho(result.observation), 6),
                "overflow_lines": _overflow_lines(result.observation),
                "disconnected_lines": _disconnected_lines(result.observation),
                "bridge_lines": graph_intelligence.get("bridge_lines", []),
                "n_minus_1_critical_lines": graph_intelligence.get(
                    "n_minus_1_critical_lines", []
                ),
                "safe_to_disconnect": graph_intelligence.get("safe_to_disconnect", []),
                "congestion_corridor": graph_intelligence.get("congestion_corridor"),
            },
            "candidate_count": len(rows),
            "include_compound_redispatch": include_compound_redispatch,
            "max_compound_redispatch_size": max_compound_redispatch_size,
            "survivor_count": len(survivors),
            "failure_count": len(failures),
            "target_threshold": SINGLE_FAULT_TARGET_RHO,
            "target_reached_count": sum(1 for row in rows if row["target_reached"]),
            "best_survivors": survivors_sorted[:top_n],
            "lookahead_depth": lookahead_depth,
            "lookahead_candidate_count": len(lookahead_rows),
            "lookahead_target_reached_count": sum(
                1 for row in lookahead_rows if row["target_reached"]
            ),
            "best_lookahead_survivors": lookahead_survivors_sorted[:top_n],
            "failed_actions": failures[:top_n],
        }
        print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate and simulate legal single actions from a reproduced episode state."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8018")
    parser.add_argument("--task-id", default="cascade_prevent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-tier", default="cascade_prevent_extreme")
    parser.add_argument(
        "--pre-action",
        action="append",
        default=[],
        help="GridAction JSON to apply before enumeration. Can be repeated.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--include-compound-redispatch",
        action="store_true",
        help="Also enumerate legal pairwise redispatch actions.",
    )
    parser.add_argument(
        "--max-compound-redispatch-size",
        type=int,
        default=2,
        help="Maximum number of generators in compound redispatch candidates.",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        choices=[1, 2],
        default=1,
        help="Depth 2 replays the best first actions and enumerates second-step actions.",
    )
    parser.add_argument(
        "--lookahead-first-actions",
        type=int,
        default=5,
        help="Number of best one-step survivors to expand when --lookahead-depth=2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pre_actions = [
        GridAction.model_validate(json.loads(raw_action))
        for raw_action in args.pre_action
    ]
    run_diagnostic(
        base_url=args.base_url,
        task_id=args.task_id,
        seed=args.seed,
        benchmark_tier=args.benchmark_tier,
        pre_actions=pre_actions,
        top_n=args.top_n,
        batch_size=args.batch_size,
        include_compound_redispatch=args.include_compound_redispatch,
        max_compound_redispatch_size=args.max_compound_redispatch_size,
        lookahead_depth=args.lookahead_depth,
        lookahead_first_actions=args.lookahead_first_actions,
    )


if __name__ == "__main__":
    main()
