from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _selected_action_type(action: dict[str, Any]) -> str:
    if action.get("do_nothing"):
        return "do_nothing"
    if action.get("redispatch"):
        return "redispatch"
    if action.get("line_set"):
        statuses = [int(value) for value in action["line_set"].values()]
        if statuses and statuses[0] == 1:
            return "reconnect_line"
        if statuses and statuses[0] == -1:
            return "disconnect_line"
        return "line_set"
    return "empty"


def _selectable_simulations(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        simulation
        for simulation in simulations
        if not simulation.get("done") and not simulation.get("convergence_failed")
    ]


def _best_indices(simulations: list[dict[str, Any]]) -> dict[str, int | None]:
    selectable = _selectable_simulations(simulations)
    if not selectable:
        return {
            "best_reward": None,
            "best_rho": None,
            "best_safe_reward": None,
        }

    safe = [
        simulation
        for simulation in selectable
        if not simulation.get("overloaded_line_ids")
    ]
    best_reward = max(
        selectable, key=lambda simulation: float(simulation["simulated_reward"])
    )["candidate_index"]
    best_rho = min(
        selectable, key=lambda simulation: float(simulation["max_rho"])
    )["candidate_index"]
    best_safe_reward = max(
        safe or selectable,
        key=lambda simulation: float(simulation["simulated_reward"]),
    )["candidate_index"]
    return {
        "best_reward": int(best_reward),
        "best_rho": int(best_rho),
        "best_safe_reward": int(best_safe_reward),
    }


def check_dataset(path: Path) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "no_selectable_simulations": 0,
            "selected_not_best_reward": 0,
            "selected_not_best_rho": 0,
            "selected_not_best_safe_reward": 0,
            "actions": Counter(),
            "label_policies": Counter(),
            "benchmark_tiers": Counter(),
        }
    )
    global_counts: dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "invalid_json_rows": 0,
        "missing_metadata_rows": 0,
        "missing_selected_action_rows": 0,
        "missing_simulations_rows": 0,
    }
    row_issues: list[dict[str, Any]] = []

    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            global_counts["invalid_json_rows"] += 1
            row_issues.append(
                {"line": line_no, "issue": "invalid_json", "error": str(exc)}
            )
            continue

        global_counts["rows"] += 1
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            global_counts["missing_metadata_rows"] += 1
            row_issues.append({"line": line_no, "issue": "missing_metadata"})
            continue

        task_id = str(metadata.get("task_id", "unknown"))
        task_stats = by_task[task_id]
        task_stats["rows"] += 1
        task_stats["label_policies"][str(metadata.get("label_policy", "missing"))] += 1
        task_stats["benchmark_tiers"][str(metadata.get("benchmark_tier", "missing"))] += 1

        selected_action = metadata.get("selected_action")
        if not isinstance(selected_action, dict):
            global_counts["missing_selected_action_rows"] += 1
            row_issues.append(
                {"line": line_no, "task_id": task_id, "issue": "missing_selected_action"}
            )
            continue
        task_stats["actions"][_selected_action_type(selected_action)] += 1

        simulations = metadata.get("simulations")
        if not isinstance(simulations, list) or not simulations:
            global_counts["missing_simulations_rows"] += 1
            row_issues.append(
                {"line": line_no, "task_id": task_id, "issue": "missing_simulations"}
            )
            continue

        selected_candidate = metadata.get("selected_candidate")
        best = _best_indices(simulations)
        if best["best_reward"] is None:
            task_stats["no_selectable_simulations"] += 1
            row_issues.append(
                {"line": line_no, "task_id": task_id, "issue": "no_selectable_simulations"}
            )
            continue

        if selected_candidate != best["best_reward"]:
            task_stats["selected_not_best_reward"] += 1
        if selected_candidate != best["best_rho"]:
            task_stats["selected_not_best_rho"] += 1
        if selected_candidate != best["best_safe_reward"]:
            task_stats["selected_not_best_safe_reward"] += 1

    task_output: dict[str, Any] = {}
    for task_id, stats in sorted(by_task.items()):
        task_output[task_id] = {
            "rows": stats["rows"],
            "no_selectable_simulations": stats["no_selectable_simulations"],
            "selected_not_best_reward": stats["selected_not_best_reward"],
            "selected_not_best_rho": stats["selected_not_best_rho"],
            "selected_not_best_safe_reward": stats["selected_not_best_safe_reward"],
            "actions": dict(sorted(stats["actions"].items())),
            "label_policies": dict(sorted(stats["label_policies"].items())),
            "benchmark_tiers": dict(sorted(stats["benchmark_tiers"].items())),
        }

    return {
        "summary": global_counts,
        "tasks": task_output,
        "row_issues": row_issues[:100],
        "row_issue_count": len(row_issues),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Grid2Op teacher dataset JSONL quality."
    )
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(check_dataset(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
