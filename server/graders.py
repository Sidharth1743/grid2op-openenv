from __future__ import annotations

import logging
from typing import Iterable

try:
    from ..models import EpisodeStepLog, TaskId
except ImportError:
    from models import EpisodeStepLog, TaskId

logger = logging.getLogger(__name__)


def _clamp_submission_score(raw_score: float) -> float:
    return round(max(0.01, min(0.99, raw_score)), 6)


def grade_episode(task_id: TaskId, episode_log: Iterable[EpisodeStepLog]) -> float:
    logs = list(episode_log)
    logger.info("Grading episode task_id=%s steps=%s", task_id, len(logs))
    if task_id == "single_fault":
        return grade_single_fault(logs)
    if task_id == "n_minus_1":
        return grade_n_minus_1(logs)
    if task_id == "cascade_prevent":
        return grade_cascade_prevent(logs)
    if task_id == "multi_stage_cascade":
        return grade_multi_stage_cascade(logs)
    raise ValueError(f"Unsupported task_id: {task_id}")


def grade_single_fault(episode_log: list[EpisodeStepLog]) -> float:
    if not episode_log:
        return 0.01

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
    return _clamp_submission_score(max(score, legacy_success_score))


def grade_n_minus_1(episode_log: list[EpisodeStepLog], max_steps: int = 20) -> float:
    if not episode_log:
        return 0.01
    emergency_clear_step = next(
        (entry.step for entry in episode_log[:5] if float(entry.max_rho) < 0.92),
        None,
    )
    emergency_score = (
        max(0.0, 1.0 - (0.2 * max(0, emergency_clear_step - 1)))
        if emergency_clear_step is not None
        else 0.0
    )

    phase2_logs = [entry for entry in episode_log if entry.step >= 6]
    security_ratio = (
        sum(1 for entry in phase2_logs if float(entry.max_rho) < 0.90) / 15.0
        if phase2_logs
        else 0.0
    )

    reconnection_score = 1.0 if any(0 not in entry.disconnected_lines for entry in episode_log) else 0.0

    survival_ratio = min(max_steps, max(entry.step for entry in episode_log)) / max_steps
    mastery_score = (0.30 * emergency_score) + (0.50 * security_ratio) + (0.20 * reconnection_score)
    final_score = mastery_score * survival_ratio
    return _clamp_submission_score(final_score)


def grade_cascade_prevent(
    episode_log: list[EpisodeStepLog], max_steps: int = 30
) -> float:
    if not episode_log:
        return 0.01
    containment_ratio = sum(1 for entry in episode_log if not entry.auto_trip_detected) / max_steps

    containment_steps = [entry for entry in episode_log if not entry.auto_trip_detected]
    stability_ratio = (
        sum(1 for entry in containment_steps if entry.all_lines_below_100) / len(containment_steps)
        if containment_steps
        else 0.0
    )

    first_overload_step = next(
        (entry.step for entry in episode_log if not entry.all_lines_below_100),
        None,
    )
    if first_overload_step is None:
        recovery_score = 1.0
    else:
        stabilize_step = next(
            (
                entry.step
                for entry in episode_log
                if entry.step >= first_overload_step and entry.all_lines_below_100
            ),
            None,
        )
        if stabilize_step is None:
            recovery_score = 0.0
        else:
            recovery_score = max(0.0, 1.0 - ((stabilize_step - first_overload_step) / 10.0))

    score = (0.5 * containment_ratio) + (0.3 * stability_ratio) + (0.2 * recovery_score)
    return _clamp_submission_score(score)


def grade_multi_stage_cascade(
    episode_log: list[EpisodeStepLog], max_steps: int = 25
) -> float:
    if not episode_log:
        return 0.01

    reached_stage_2 = any(entry.step >= 10 for entry in episode_log)
    reached_stage_3 = any(entry.step >= 20 for entry in episode_log)
    ended_without_blackout = bool(episode_log[-1].step >= max_steps and not episode_log[-1].convergence_failed)
    stage_completion = (
        float(reached_stage_2) + float(reached_stage_3) + float(ended_without_blackout)
    ) / 3.0

    final_entry = episode_log[-1]
    load_preservation = max(0.0, min(1.0, float(final_entry.available_load_ratio)))

    boundary_logs = [
        entry for entry in episode_log if entry.stage_boundary_assessed
    ]
    island_quality = (
        sum(1 for entry in boundary_logs if entry.majority_islands_available) / 2.0
        if boundary_logs
        else 0.0
    )
    island_quality = max(0.0, min(1.0, island_quality))

    stage_ranges = [(1, 10), (11, 20), (21, 30)]
    stage_speed_scores: list[float] = []
    for start_step, end_step in stage_ranges:
        stable_step = next(
            (
                entry.step
                for entry in episode_log
                if start_step <= entry.step <= end_step and entry.all_lines_below_100
            ),
            None,
        )
        if stable_step is None:
            stage_speed_scores.append(0.0)
            continue
        steps_to_stable = stable_step - start_step
        stage_speed_scores.append(max(0.0, (10.0 - steps_to_stable) / 10.0))
    speed_score = sum(stage_speed_scores) / len(stage_speed_scores)

    score = (
        0.30 * stage_completion
        + 0.40 * load_preservation
        + 0.20 * island_quality
        + 0.10 * speed_score
    )
    return _clamp_submission_score(score)
