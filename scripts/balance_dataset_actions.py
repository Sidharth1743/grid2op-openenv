from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def selected_action_type(row: dict[str, Any]) -> str:
    action = row.get("metadata", {}).get("selected_action", {})
    if action.get("do_nothing"):
        return "do_nothing"
    if action.get("redispatch"):
        return "redispatch"
    line_set = action.get("line_set") or {}
    if line_set:
        statuses = [int(value) for value in line_set.values()]
        if statuses and statuses[0] == 1:
            return "reconnect_line"
        if statuses and statuses[0] == -1:
            return "disconnect_line"
        return "line_set"
    return "empty"


def parse_action_caps(values: list[str]) -> dict[str, int]:
    caps: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected ACTION=COUNT, got: {value}")
        action, count_text = value.split("=", 1)
        caps[action.strip()] = int(count_text)
    return caps


def parse_task_action_caps(values: list[str]) -> dict[tuple[str, str], int]:
    caps: dict[tuple[str, str], int] = {}
    for value in values:
        if "=" not in value or ":" not in value:
            raise ValueError(f"Expected TASK:ACTION=COUNT, got: {value}")
        key, count_text = value.split("=", 1)
        task_id, action = key.split(":", 1)
        caps[(task_id.strip(), action.strip())] = int(count_text)
    return caps


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Row {line_no} is not a JSON object")
        rows.append(row)
    return rows


def balance_rows(
    rows: list[dict[str, Any]],
    task_id: str | None,
    action_caps: dict[str, int],
    task_action_caps: dict[tuple[str, str], int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    untouched_rows: list[dict[str, Any]] = []

    for row in rows:
        row_task = str(row.get("metadata", {}).get("task_id", "unknown"))
        if task_id is not None and row_task != task_id:
            untouched_rows.append(row)
        else:
            target_rows.append(row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        row_task = str(row.get("metadata", {}).get("task_id", "unknown"))
        grouped[(row_task, selected_action_type(row))].append(row)

    action_summary: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for (row_task, action), action_rows in sorted(grouped.items()):
        cap = task_action_caps.get((row_task, action), action_caps.get(action))
        original_count = len(action_rows)
        selected_rows = list(action_rows)
        if cap is not None and original_count > cap:
            selected_rows = rng.sample(action_rows, cap)
        kept.extend(selected_rows)
        action_summary[row_task][action] = {
            "original": original_count,
            "kept": len(selected_rows),
        }

    kept.sort(
        key=lambda row: (
            str(row.get("metadata", {}).get("task_id", "")),
            int(row.get("metadata", {}).get("seed", 0)),
            int(row.get("metadata", {}).get("step", 0)),
            selected_action_type(row),
        )
    )

    output_rows = untouched_rows + kept
    summary = {
        "input_rows": len(rows),
        "output_rows": len(output_rows),
        "untouched_rows": len(untouched_rows),
        "balanced_rows": len(kept),
        "task_id": task_id,
        "action_caps": action_caps,
        "task_action_caps": {
            f"{task}:{action}": cap
            for (task, action), cap in sorted(task_action_caps.items())
        },
        "actions": action_summary,
        "output_action_counts": dict(Counter(selected_action_type(row) for row in output_rows)),
        "output_task_action_counts": {
            task: dict(counter)
            for task, counter in sorted(
                task_action_counts(output_rows).items()
            )
        },
    }
    return output_rows, summary


def task_action_counts(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        task = str(row.get("metadata", {}).get("task_id", "unknown"))
        counts[task][selected_action_type(row)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample overrepresented action labels in a JSONL dataset.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--max-action-rows", nargs="*", default=[])
    parser.add_argument("--max-task-action-rows", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    action_caps = parse_action_caps(args.max_action_rows)
    task_action_caps = parse_task_action_caps(args.max_task_action_rows)
    output_rows, summary = balance_rows(
        rows=rows,
        task_id=args.task_id,
        action_caps=action_caps,
        task_action_caps=task_action_caps,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
