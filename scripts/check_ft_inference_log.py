from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


START_RE = re.compile(r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>.+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>\{.*\}) "
    r"reward=(?P<reward>-?\d+(?:\.\d+)?) done=(?P<done>true|false) error=(?P<error>.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) "
    r"score=(?P<score>-?\d+(?:\.\d+)?) rewards=(?P<rewards>.*)$"
)


def action_type(action: dict[str, Any]) -> str:
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


def parse_final_summary(lines: list[str]) -> dict[str, Any] | None:
    for index, line in enumerate(lines):
        if not line.startswith("{"):
            continue
        candidate = "\n".join(lines[index:])
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "tasks" in payload and "episodes" in payload:
            return payload
    return None


def parse_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    summary: dict[str, Any] = {
        "path": str(path),
        "episodes": 0,
        "failures": [],
        "tasks": {},
        "final_summary": parse_final_summary(lines),
    }
    current_task: str | None = None
    current_steps: list[dict[str, Any]] = []
    task_episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for line_no, line in enumerate(lines, 1):
        start_match = START_RE.match(line)
        if start_match:
            current_task = start_match.group("task")
            current_steps = []
            continue

        if line.startswith("[FT_FAIL] "):
            payload_text = line[len("[FT_FAIL] ") :]
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                payload = {"raw": payload_text}
            payload["line_no"] = line_no
            summary["failures"].append(payload)
            continue

        step_match = STEP_RE.match(line)
        if step_match and current_task:
            action = json.loads(step_match.group("action"))
            reward = float(step_match.group("reward"))
            error = step_match.group("error")
            current_steps.append(
                {
                    "step": int(step_match.group("step")),
                    "action": action,
                    "action_type": action_type(action),
                    "reward": reward,
                    "done": step_match.group("done") == "true",
                    "error": None if error == "null" else error,
                }
            )
            continue

        end_match = END_RE.match(line)
        if end_match and current_task:
            rewards_text = end_match.group("rewards")
            rewards = [
                float(value)
                for value in rewards_text.split(",")
                if value.strip()
            ]
            episode = {
                "success": end_match.group("success") == "true",
                "steps": int(end_match.group("steps")),
                "score": float(end_match.group("score")),
                "rewards": rewards,
                "step_count_from_log": len(current_steps),
                "actions": current_steps,
            }
            task_episodes[current_task].append(episode)
            summary["episodes"] += 1
            current_task = None
            current_steps = []

    for task_id, episodes in sorted(task_episodes.items()):
        action_counts: Counter[str] = Counter()
        scores: list[float] = []
        steps: list[int] = []
        reward_sums: list[float] = []
        negative_terminal_rewards = 0
        errored_steps = 0
        invalid_step_counts = 0

        for episode in episodes:
            scores.append(float(episode["score"]))
            steps.append(int(episode["steps"]))
            reward_sums.append(sum(float(value) for value in episode["rewards"]))
            if episode["rewards"] and episode["rewards"][-1] <= -5.0:
                negative_terminal_rewards += 1
            if episode["steps"] != episode["step_count_from_log"]:
                invalid_step_counts += 1
            for step in episode["actions"]:
                action_counts[step["action_type"]] += 1
                if step["error"] is not None:
                    errored_steps += 1

        summary["tasks"][task_id] = {
            "episodes": len(episodes),
            "successes": sum(1 for episode in episodes if episode["success"]),
            "mean_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "mean_steps": round(sum(steps) / len(steps), 3) if steps else 0.0,
            "mean_reward_sum": round(sum(reward_sums) / len(reward_sums), 6)
            if reward_sums
            else 0.0,
            "action_counts": dict(sorted(action_counts.items())),
            "errored_steps": errored_steps,
            "negative_terminal_rewards": negative_terminal_rewards,
            "invalid_step_counts": invalid_step_counts,
        }

    summary["safety"] = {
        "pass": not summary["failures"]
        and all(
            task["errored_steps"] == 0 and task["invalid_step_counts"] == 0
            for task in summary["tasks"].values()
        ),
        "failure_count": len(summary["failures"]),
        "errored_step_count": sum(
            task["errored_steps"] for task in summary["tasks"].values()
        ),
        "negative_terminal_episode_count": sum(
            task["negative_terminal_rewards"] for task in summary["tasks"].values()
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ft_inference.py terminal logs.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(parse_log(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
