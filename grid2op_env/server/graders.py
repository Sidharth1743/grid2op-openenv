from __future__ import annotations

import logging
from typing import Iterable

try:
    from ..models import EpisodeStepLog, TaskId
except ImportError:
    from models import EpisodeStepLog, TaskId

logger = logging.getLogger(__name__)


def grade_episode(task_id: TaskId, episode_log: Iterable[EpisodeStepLog]) -> float:
    logs = list(episode_log)
    logger.info("Grading episode task_id=%s steps=%s", task_id, len(logs))
    if task_id == "single_fault":
        return grade_single_fault(logs)
    if task_id == "n_minus_1":
        return grade_n_minus_1(logs)
    if task_id == "cascade_prevent":
        return grade_cascade_prevent(logs)
    raise ValueError(f"Unsupported task_id: {task_id}")


def grade_single_fault(episode_log: list[EpisodeStepLog]) -> float:
    if not episode_log:
        return 0.0

    max_steps = 10
    survival_ratio = min(1.0, len(episode_log) / max_steps)
    achieved_target = any(entry.all_lines_below_target or entry.all_lines_below_80 for entry in episode_log)
    target_bonus = 0.5 if achieved_target else 0.0
    legacy_success_score = 0.0
    for entry in episode_log:
        if entry.all_lines_below_target or entry.all_lines_below_80:
            legacy_success_score = round(max(0.0, 1.0 - (0.08 * max(0, entry.step - 1))), 6)
            break

    final_entry = episode_log[-1]
    final_rho = float(final_entry.max_rho)
    target_threshold = float(final_entry.single_fault_target_threshold or 0.8)
    if final_rho < target_threshold:
        final_bonus = 0.3
    elif final_rho < target_threshold + 0.05:
        final_bonus = 0.15
    elif final_rho < target_threshold + 0.10:
        final_bonus = 0.05
    else:
        final_bonus = 0.0

    score = (survival_ratio * 0.7) + target_bonus + final_bonus
    return round(min(1.0, max(0.0, score, legacy_success_score)), 6)


def grade_n_minus_1(episode_log: list[EpisodeStepLog], max_steps: int = 20) -> float:
    if not episode_log:
        return 0.0
    survived_steps = min(max_steps, max(entry.step for entry in episode_log))
    return round(min(1.0, survived_steps / max_steps), 6)


def grade_cascade_prevent(
    episode_log: list[EpisodeStepLog], max_steps: int = 30
) -> float:
    if not episode_log:
        return 0.0

    terminated_early = any(entry.done and entry.step < max_steps for entry in episode_log)
    convergence_failed = any(entry.convergence_failed for entry in episode_log)
    survived_full = (
        1.0
        if episode_log[-1].step >= max_steps and not terminated_early and not convergence_failed
        else 0.0
    )
    safety_ratio = sum(1 for entry in episode_log if entry.all_lines_below_100) / max_steps

    stabilize_step = next(
        (entry.step for entry in episode_log if entry.all_lines_below_100),
        None,
    )
    stabilization_score = (
        (max_steps - stabilize_step + 1) / max_steps if stabilize_step is not None else 0.0
    )

    score = (0.5 * survived_full) + (0.3 * safety_ratio) + (0.2 * stabilization_score)
    return round(min(1.0, max(0.0, score)), 6)
