from __future__ import annotations

from typing import Any


def objective_value(simulation: dict[str, Any]) -> float:
    if simulation.get("lookahead_value") is not None:
        return float(simulation.get("lookahead_value", 0.0))
    return float(simulation.get("simulated_reward", 0.0))


def simulation_action_type(simulation: dict[str, Any]) -> str:
    action = simulation.get("action", {}) or {}
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
    return "unknown"


def is_safe(simulation: dict[str, Any]) -> bool:
    return (
        not bool(simulation.get("done"))
        and not bool(simulation.get("convergence_failed"))
        and not bool(simulation.get("exceptions"))
    )


def current_overflow_from_summary(observation_summary: dict[str, Any]) -> int:
    overflow = observation_summary.get("timestep_overflow") or []
    if not isinstance(overflow, list):
        return 0
    return max((int(value) for value in overflow), default=0)


def objective_completion_score(
    *,
    task_id: str,
    simulation: dict[str, Any],
    simulations: list[dict[str, Any]],
    observation_summary: dict[str, Any],
) -> float:
    safe = is_safe(simulation)
    if not safe:
        return -100.0

    current_max_rho = float(observation_summary.get("max_rho", 0.0))
    current_overflow = current_overflow_from_summary(observation_summary)
    selected_max_rho = float(simulation.get("max_rho", 999.0))
    selected_reward = objective_value(simulation)
    action_kind = simulation_action_type(simulation)
    overloaded_count = len(simulation.get("overloaded_line_ids") or [])
    max_overflow = max((int(value) for value in (simulation.get("raw_result", {}) or {}).get("timestep_overflow", []) or []), default=0)

    safe_sims = [candidate for candidate in simulations if is_safe(candidate)]
    noop_sim = next((candidate for candidate in simulations if simulation_action_type(candidate) == "do_nothing"), None)
    noop_value = objective_value(noop_sim) if noop_sim is not None else selected_reward
    best_non_noop = max(
        (objective_value(candidate) for candidate in safe_sims if simulation_action_type(candidate) != "do_nothing"),
        default=noop_value,
    )

    if task_id == "single_fault":
        target_reached = selected_max_rho < 0.80
        best_rho = min((float(candidate.get("max_rho", 999.0)) for candidate in safe_sims), default=selected_max_rho)
        score = 8.0 if target_reached else (current_max_rho - selected_max_rho) * 40.0
        score -= max(0.0, selected_max_rho - 0.80) * 25.0
        if action_kind == "do_nothing" and current_max_rho > 0.80 and best_rho < selected_max_rho - 1e-4:
            score -= 4.0
        return score

    if task_id == "n_minus_1":
        threshold = 0.92 if current_max_rho >= 0.92 else 0.90
        score = (current_max_rho - selected_max_rho) * 30.0
        if selected_max_rho < threshold:
            score += 5.0
        score -= overloaded_count * 2.0
        if action_kind == "reconnect_line":
            score += 1.5
        if action_kind == "do_nothing" and current_max_rho >= threshold and best_non_noop > noop_value + 0.02:
            score -= 3.0
        return score + (selected_reward * 0.1)

    if task_id == "cascade_prevent":
        score = (current_overflow - max_overflow) * 4.0
        score -= max_overflow * 3.0
        score -= overloaded_count * 2.0
        score -= max(0.0, selected_max_rho - 1.0) * 15.0
        if max_overflow == 0:
            score += 4.0
        if action_kind == "do_nothing" and (current_overflow > 0 or current_max_rho > 1.0) and best_non_noop > noop_value + 0.02:
            score -= 3.0
        return score + (selected_reward * 0.1)

    if task_id == "multi_stage_cascade":
        score = selected_reward * 1.5
        score += (current_max_rho - selected_max_rho) * 12.0
        score -= overloaded_count * 1.5
        score -= max_overflow * 1.5
        score -= max(0.0, selected_max_rho - 0.85) * 12.0
        if action_kind != "do_nothing":
            score += 0.5
        if action_kind == "do_nothing" and current_max_rho > 0.80 and best_non_noop > noop_value + 0.02:
            score -= 4.0
        return score

    return selected_reward
