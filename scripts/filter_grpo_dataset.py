from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_grpo_verifier import filter_grpo_rows, load_grpo_rows, summarize_grpo_rows


def _load_raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Row {line_no} is not a JSON object")
        rows.append(row)
    return rows


def _build_keep_keys(filtered_rows: list[dict[str, Any]]) -> set[tuple[str, int, int, str]]:
    keys: set[tuple[str, int, int, str]] = set()
    for row in filtered_rows:
        keys.add(
            (
                str(row.get("task_id", "unknown")),
                int(row.get("seed_value", -1)),
                int(row.get("step_value", -1)),
                str(row.get("benchmark_tier", "unknown")),
            )
        )
    return keys


def _filter_raw_rows(
    raw_rows: list[dict[str, Any]],
    keep_keys: set[tuple[str, int, int, str]],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in raw_rows:
        metadata = row.get("metadata") or {}
        key = (
            str(metadata.get("task_id", "unknown")),
            int(metadata.get("seed", -1)),
            int(metadata.get("step", -1)),
            str(metadata.get("benchmark_tier", "unknown")),
        )
        if key in keep_keys:
            kept.append(row)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the filtered GRPO training subset into a new JSONL file."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-style", choices=["compact", "original"], default="compact")
    parser.add_argument("--task-filter", nargs="*", default=None)
    parser.add_argument("--require-noop-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-noop-gap", type=float, default=0.01)
    parser.add_argument("--informative-multistage-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = _load_raw_rows(args.input)
    trainer_rows = load_grpo_rows(args.input, prompt_style=args.prompt_style)
    filtered_rows = filter_grpo_rows(
        trainer_rows,
        task_filter=args.task_filter,
        require_noop_baseline=args.require_noop_baseline,
        min_noop_gap=args.min_noop_gap,
        informative_multistage_only=args.informative_multistage_only,
    )
    keep_keys = _build_keep_keys(filtered_rows)
    output_rows = _filter_raw_rows(raw_rows, keep_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input_path": str(args.input),
                "output_path": str(args.output),
                "input_rows": len(raw_rows),
                "filtered_rows": len(output_rows),
                "filters": {
                    "task_filter": args.task_filter,
                    "require_noop_baseline": args.require_noop_baseline,
                    "min_noop_gap": args.min_noop_gap,
                    "informative_multistage_only": args.informative_multistage_only,
                    "prompt_style": args.prompt_style,
                },
                "summary": summarize_grpo_rows(filtered_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
