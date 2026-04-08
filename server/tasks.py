from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from grid2op.Exceptions import Grid2OpException

try:
    from ..models import GridObservation, ScenarioMode, TaskId, TaskInfo
except ImportError:
    from models import GridObservation, ScenarioMode, TaskId, TaskInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSpec:
    task_id: TaskId
    difficulty: str
    description: str
    max_steps: int


TASKS: Dict[TaskId, TaskSpec] = {
    "single_fault": TaskSpec(
        task_id="single_fault",
        difficulty="easy",
        description=(
            "One line is already approaching its thermal limit. Stabilize the grid "
            "within 10 steps and bring all lines below 80% loading."
        ),
        max_steps=10,
    ),
    "n_minus_1": TaskSpec(
        task_id="n_minus_1",
        difficulty="medium",
        description=(
            "One line is disconnected at reset. Survive 20 steps and redistribute "
            "power flows without causing a blackout."
        ),
        max_steps=20,
    ),
    "cascade_prevent": TaskSpec(
        task_id="cascade_prevent",
        difficulty="hard",
        description=(
            "Two lines are disconnected and load is increased by 15%. Prevent a "
            "cascade within 30 steps before overloads trigger more trips."
        ),
        max_steps=30,
    ),
    "multi_stage_cascade": TaskSpec(
        task_id="multi_stage_cascade",
        difficulty="hard",
        description=(
            "Three lines are disconnected and load is increased by 20%. Manage the "
            "guaranteed three-stage cascade for 25 steps and preserve as much load "
            "as possible across stage boundaries."
        ),
        max_steps=25,
    ),
}

CASCADE_LINE_PAIRS: List[tuple[int, int]] = [
    (0, 8),
    (0, 7),
    (0, 3),
]
MULTI_STAGE_LINE_TRIPLETS: List[tuple[int, int, int]] = [
    (2, 4, 14),
    (2, 4, 15),
    (4, 14, 16),
]

BENCHMARK_TIERS: Dict[TaskId, List[str]] = {
    "single_fault": ["single_fault_easy", "single_fault_moderate", "single_fault_severe"],
    "n_minus_1": ["n_minus_1_fixed"],
    "cascade_prevent": [
        "cascade_prevent_easy",
        "cascade_prevent_medium",
        "cascade_prevent_hard",
        "cascade_prevent_extreme",
    ],
    "multi_stage_cascade": ["multi_stage_cascade_expert"],
}


def task_list() -> List[TaskInfo]:
    return [
        TaskInfo(
            task_id=spec.task_id,
            difficulty=spec.difficulty,  # type: ignore[arg-type]
            description=spec.description,
            max_steps=spec.max_steps,
        )
        for spec in TASKS.values()
    ]


def inject_scenario_raw(
    env,
    task_id: TaskId,
    seed: int | None = None,
    max_attempts: int = 3,
    difficulty_level: int | None = None,
    scenario_mode: ScenarioMode = "curriculum",
    benchmark_tier: str | None = None,
):
    """Initialize the environment according to the selected task and return the raw Grid2Op observation."""

    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")

    effective_attempts = max_attempts
    if task_id == "single_fault":
        effective_attempts = max(max_attempts, 8)
        if scenario_mode == "benchmark":
            effective_attempts = max(effective_attempts, 48)
    if task_id == "multi_stage_cascade":
        effective_attempts = max(max_attempts, _available_time_series_count(env))

    for attempt in range(effective_attempts):
        try:
            logger.info(
                "Injecting scenario task_id=%s attempt=%s seed=%s difficulty_level=%s",
                task_id,
                attempt + 1,
                seed,
                difficulty_level,
            )
            if task_id == "single_fault":
                return _reset_single_fault(
                    env,
                    seed=seed,
                    attempt=attempt,
                    difficulty_level=difficulty_level,
                    scenario_mode=scenario_mode,
                    benchmark_tier=benchmark_tier,
                )
            if task_id == "n_minus_1":
                return _reset_n_minus_1(
                    env,
                    seed=seed,
                    difficulty_level=difficulty_level,
                    scenario_mode=scenario_mode,
                    benchmark_tier=benchmark_tier,
                )
            if task_id == "multi_stage_cascade":
                return _reset_multi_stage_cascade(
                    env,
                    seed=seed,
                    attempt=attempt,
                    difficulty_level=difficulty_level,
                    scenario_mode=scenario_mode,
                    benchmark_tier=benchmark_tier,
                )
            return _reset_cascade_prevent(
                env,
                seed=seed,
                attempt=attempt,
                difficulty_level=difficulty_level,
                scenario_mode=scenario_mode,
                benchmark_tier=benchmark_tier,
            )
        except Grid2OpException as exc:
            logger.warning(
                "Scenario injection failed task_id=%s attempt=%s error=%s",
                task_id,
                attempt + 1,
                exc,
            )
            if attempt == effective_attempts - 1:
                raise
    raise RuntimeError("Unreachable")


def inject_scenario(
    env,
    task_id: TaskId,
    seed: int | None = None,
    max_attempts: int = 3,
    difficulty_level: int | None = None,
    scenario_mode: ScenarioMode = "curriculum",
    benchmark_tier: str | None = None,
) -> tuple[GridObservation, Dict[str, Any]]:
    """Initialize the environment according to the selected task."""

    raw_obs, metadata = inject_scenario_raw(
        env,
        task_id,
        seed=seed,
        max_attempts=max_attempts,
        difficulty_level=difficulty_level,
        scenario_mode=scenario_mode,
        benchmark_tier=benchmark_tier,
    )
    return _convert(raw_obs), metadata


def replay_scenario_raw(
    env,
    task_id: TaskId,
    seed: int | None,
    scenario_metadata: dict[str, Any],
):
    """Replay the exact scenario selected by the server using persisted metadata."""

    if task_id == "single_fault":
        time_series_id = scenario_metadata.get("time_series_id")
        warmup_steps = scenario_metadata.get("warmup_steps")
        if time_series_id is None or warmup_steps is None:
            raise Grid2OpException(
                "single_fault scenario metadata is missing time_series_id or warmup_steps; restart the env server with the current code"
            )
        return _replay_single_fault_state(
            env=env,
            seed=seed,
            options={"time serie id": int(time_series_id)},
            warmup_steps=int(warmup_steps),
        )

    if task_id == "n_minus_1":
        faulted_lines = [int(line_id) for line_id in scenario_metadata.get("faulted_lines", [0])]
        return env.reset(
            seed=seed,
            options={"init state": {"set_line_status": [(line_id, -1) for line_id in faulted_lines]}},
        )

    if task_id == "cascade_prevent":
        faulted_lines = scenario_metadata.get("faulted_lines")
        load_scale = scenario_metadata.get("load_scale")
        if faulted_lines is None or load_scale is None:
            raise Grid2OpException(
                "cascade_prevent scenario metadata is missing faulted_lines or load_scale; restart the env server with the current code"
            )
        base_obs = env.reset(seed=seed)
        load_p = [float(v) for v in (base_obs.load_p * float(load_scale)).astype(float).tolist()]
        return env.reset(
            seed=seed,
            options={
                "init state": {
                    "set_line_status": [(int(line_id), -1) for line_id in faulted_lines],
                    "injection": {"load_p": load_p},
                }
            },
        )

    if task_id == "multi_stage_cascade":
        faulted_lines = scenario_metadata.get("faulted_lines")
        load_scale = scenario_metadata.get("load_scale")
        time_series_id = scenario_metadata.get("time_series_id")
        if faulted_lines is None or load_scale is None:
            raise Grid2OpException(
                "multi_stage_cascade scenario metadata is missing faulted_lines or load_scale; restart the env server with the current code"
            )
        _set_overflow_window(env, allowed_steps=2)
        reset_options = {"time serie id": int(time_series_id)} if time_series_id is not None else None
        base_obs = env.reset(seed=seed, options=reset_options)
        load_p = [float(v) for v in (base_obs.load_p * float(load_scale)).astype(float).tolist()]
        return env.reset(
            seed=seed,
            options={
                **({"time serie id": int(time_series_id)} if time_series_id is not None else {}),
                "init state": {
                    "set_line_status": [(int(line_id), -1) for line_id in faulted_lines],
                    "injection": {"load_p": load_p},
                }
            },
        )

    raise ValueError(f"Unsupported task_id for replay: {task_id}")


def _curriculum_episode(difficulty_level: int | None) -> int:
    if difficulty_level is None or difficulty_level < 1:
        return 1
    return int(difficulty_level)


def _distance_to_range(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def _single_fault_profile(difficulty_level: int | None) -> tuple[str, float, float]:
    episode = _curriculum_episode(difficulty_level)
    if episode <= 3:
        return "mild", 0.90, 0.94
    if episode <= 6:
        return "moderate", 0.94, 0.97
    return "severe", 0.96, 0.99


def _single_fault_benchmark_profile(benchmark_tier: str | None) -> tuple[str, float, float]:
    if benchmark_tier == "single_fault_easy":
        return "benchmark_easy", 0.82, 0.85
    if benchmark_tier == "single_fault_moderate":
        return "benchmark_moderate", 0.86, 0.89
    if benchmark_tier == "single_fault_severe":
        return "benchmark_severe", 0.90, 0.93
    raise ValueError(f"Unsupported single_fault benchmark_tier: {benchmark_tier}")


def _cascade_profile(difficulty_level: int | None, seed: int | None) -> tuple[str, list[int], float]:
    episode = _curriculum_episode(difficulty_level)
    pair_index = ((episode - 1) + (0 if seed is None else int(seed))) % len(CASCADE_LINE_PAIRS)
    selected_pair = list(CASCADE_LINE_PAIRS[pair_index])
    if episode <= 3:
        return "one_line_5pct", [selected_pair[0]], 1.05
    if episode <= 6:
        return "one_line_10pct", [selected_pair[0]], 1.10
    if episode <= 9:
        return "two_lines_10pct", selected_pair, 1.10
    return "two_lines_15pct", selected_pair, 1.15


def _cascade_benchmark_profile(benchmark_tier: str | None, seed: int | None) -> tuple[str, list[int], float]:
    pair_index = 0 if seed is None else int(seed) % len(CASCADE_LINE_PAIRS)
    selected_pair = list(CASCADE_LINE_PAIRS[pair_index])
    if benchmark_tier == "cascade_prevent_easy":
        return "benchmark_easy", [selected_pair[0]], 1.05
    if benchmark_tier == "cascade_prevent_medium":
        return "benchmark_medium", [selected_pair[0]], 1.10
    if benchmark_tier == "cascade_prevent_hard":
        return "benchmark_hard", selected_pair, 1.10
    if benchmark_tier == "cascade_prevent_extreme":
        return "benchmark_extreme", selected_pair, 1.15
    raise ValueError(f"Unsupported cascade_prevent benchmark_tier: {benchmark_tier}")


def _multi_stage_profile(seed: int | None) -> tuple[str, list[int], float]:
    triplet_index = 0 if seed is None else int(seed) % len(MULTI_STAGE_LINE_TRIPLETS)
    selected_triplet = list(MULTI_STAGE_LINE_TRIPLETS[triplet_index])
    return "expert_three_stage", selected_triplet, 1.20


def benchmark_tiers_for_task(task_id: TaskId) -> list[str]:
    return list(BENCHMARK_TIERS[task_id])


def _available_time_series_count(env) -> int:
    real_data = getattr(env.chronics_handler, "real_data", None)
    subpaths = getattr(real_data, "subpaths", None)
    if subpaths is None:
        return 1
    return max(1, int(len(subpaths)))


def _single_fault_time_series_id(env, seed: int | None, attempt: int, difficulty_level: int | None) -> int:
    total = _available_time_series_count(env)
    base = _curriculum_episode(difficulty_level) - 1
    seed_offset = 0 if seed is None else int(seed) * 131
    return int((base + seed_offset + attempt * 17) % total)


def _reset_single_fault(
    env,
    seed: int | None,
    attempt: int,
    difficulty_level: int | None,
    scenario_mode: ScenarioMode,
    benchmark_tier: str | None,
):
    minimum_acceptable_fallback_rho = 0.80
    if scenario_mode == "benchmark":
        stage, min_rho, max_rho_target = _single_fault_benchmark_profile(benchmark_tier)
    else:
        stage, min_rho, max_rho_target = _single_fault_profile(difficulty_level)
    options = {
        "time serie id": _single_fault_time_series_id(
            env=env,
            seed=seed,
            attempt=attempt,
            difficulty_level=difficulty_level,
        )
    }
    obs = env.reset(seed=seed, options=options)
    max_warmup = 2000
    warmup_steps = 0
    best_obs = obs
    best_warmup_steps = 0
    best_max_rho = float(max(obs.rho))
    best_distance = _distance_to_range(best_max_rho, min_rho, max_rho_target)
    best_stable_obs = obs if best_max_rho < 1.0 else None
    best_stable_warmup_steps = 0
    best_stable_max_rho = best_max_rho
    best_stable_distance = best_distance if best_max_rho < 1.0 else float("inf")

    while warmup_steps < max_warmup:
        max_rho = float(max(obs.rho))
        distance = _distance_to_range(max_rho, min_rho, max_rho_target)
        if distance < best_distance:
            best_obs = obs
            best_warmup_steps = warmup_steps
            best_max_rho = max_rho
            best_distance = distance
        if max_rho < 1.0 and distance < best_stable_distance:
            best_stable_obs = obs
            best_stable_warmup_steps = warmup_steps
            best_stable_max_rho = max_rho
            best_stable_distance = distance
        if min_rho <= max_rho <= max_rho_target:
            logger.info(
                "Selected single_fault warmup state stage=%s warmup_steps=%s max_rho=%.4f target=[%.2f, %.2f]",
                stage,
                warmup_steps,
                max_rho,
                min_rho,
                max_rho_target,
            )
            return obs, {
                    "curriculum_episode": _curriculum_episode(difficulty_level),
                    "curriculum_stage": stage,
                    "scenario_mode": scenario_mode,
                    "benchmark_tier": benchmark_tier,
                    "time_series_id": int(options["time serie id"]),
                    "target_rho_range": [min_rho, max_rho_target],
                    "warmup_steps": warmup_steps,
                    "target_matched": True,
                    "benchmark_valid": True,
                    "scenario": "high_loading",
                }
        obs, _, done, _ = env.step(env.action_space())
        warmup_steps += 1
        max_rho = float(max(obs.rho))
        distance = _distance_to_range(max_rho, min_rho, max_rho_target)
        if distance < best_distance:
            best_obs = obs
            best_warmup_steps = warmup_steps
            best_max_rho = max_rho
            best_distance = distance
        if max_rho < 1.0 and distance < best_stable_distance:
            best_stable_obs = obs
            best_stable_warmup_steps = warmup_steps
            best_stable_max_rho = max_rho
            best_stable_distance = distance
        if done:
            break

    if best_stable_obs is not None and best_stable_max_rho >= minimum_acceptable_fallback_rho:
        logger.warning(
            "Falling back to closest stable single_fault warmup state stage=%s best_warmup_steps=%s best_max_rho=%.4f target=[%.2f, %.2f]",
            stage,
            best_stable_warmup_steps,
            best_stable_max_rho,
            min_rho,
            max_rho_target,
        )
        replayed_obs = _replay_single_fault_state(
            env=env,
            seed=seed,
            options=options,
            warmup_steps=best_stable_warmup_steps,
        )
        return replayed_obs, {
            "curriculum_episode": _curriculum_episode(difficulty_level),
            "curriculum_stage": stage,
            "scenario_mode": scenario_mode,
            "benchmark_tier": benchmark_tier,
            "time_series_id": int(options["time serie id"]),
            "target_rho_range": [min_rho, max_rho_target],
            "warmup_steps": best_stable_warmup_steps,
            "target_matched": False,
            "benchmark_valid": scenario_mode != "benchmark",
            "stable_fallback_used": True,
            "scenario": "high_loading_closest_stable_match",
        }
    raise Grid2OpException(
        f"Could not find a stable single-fault warmup state in target range [{min_rho:.2f}, {max_rho_target:.2f}] after {max_warmup} steps"
    )


def _replay_single_fault_state(env, seed: int | None, options: dict[str, Any], warmup_steps: int):
    """Reset and replay the deterministic warmup so env backend matches the returned observation."""

    obs = env.reset(seed=seed, options=options)
    for _ in range(warmup_steps):
        obs, _, done, _ = env.step(env.action_space())
        if done:
            raise Grid2OpException(
                f"Could not replay single_fault warmup to step {warmup_steps}: episode terminated during replay"
            )
    return obs


def _reset_n_minus_1(
    env,
    seed: int | None,
    difficulty_level: int | None,
    scenario_mode: ScenarioMode,
    benchmark_tier: str | None,
):
    obs = env.reset(
        seed=seed,
        options={"init state": {"set_line_status": [(0, -1)]}},
    )
    logger.info(
        "Initialized n_minus_1 with faulted_lines=[0] curriculum_episode=%s",
        _curriculum_episode(difficulty_level),
    )
    return obs, {
        "faulted_lines": [0],
        "curriculum_episode": _curriculum_episode(difficulty_level),
        "curriculum_stage": "fixed_n_minus_1",
        "scenario_mode": scenario_mode,
        "benchmark_tier": benchmark_tier or "n_minus_1_fixed",
        "benchmark_valid": True,
    }


def _reset_cascade_prevent(
    env,
    seed: int | None,
    attempt: int,
    difficulty_level: int | None,
    scenario_mode: ScenarioMode,
    benchmark_tier: str | None,
):
    base_obs = env.reset(seed=seed)
    del attempt
    if scenario_mode == "benchmark":
        stage, faulted_lines, load_scale = _cascade_benchmark_profile(benchmark_tier, seed)
    else:
        stage, faulted_lines, load_scale = _cascade_profile(difficulty_level, seed)
    load_p = [float(v) for v in (base_obs.load_p * load_scale).astype(float).tolist()]
    obs = env.reset(
        seed=seed,
        options={
            "init state": {
                "set_line_status": [(line_id, -1) for line_id in faulted_lines],
                "injection": {"load_p": load_p},
            }
        },
    )
    logger.info(
        "Initialized cascade_prevent with stage=%s faulted_lines=%s load_scale=%.2f max_rho=%.4f",
        stage,
        faulted_lines,
        load_scale,
        float(max(obs.rho)),
    )
    return obs, {
        "faulted_lines": faulted_lines,
        "load_scale": load_scale,
        "curriculum_episode": _curriculum_episode(difficulty_level),
        "curriculum_stage": stage,
        "scenario_mode": scenario_mode,
        "benchmark_tier": benchmark_tier,
        "benchmark_valid": True,
    }


def _set_overflow_window(env, allowed_steps: int) -> None:
    from grid2op.Parameters import Parameters

    params = Parameters()
    params.init_from_dict(env.parameters.to_dict())
    params.NB_TIMESTEP_OVERFLOW_ALLOWED = int(allowed_steps)
    env.change_parameters(params)


def _reset_multi_stage_cascade(
    env,
    seed: int | None,
    attempt: int,
    difficulty_level: int | None,
    scenario_mode: ScenarioMode,
    benchmark_tier: str | None,
):
    del difficulty_level
    stage, faulted_lines, load_scale = _multi_stage_profile(seed)
    _set_overflow_window(env, allowed_steps=2)
    total = _available_time_series_count(env)
    for offset in range(total):
        time_series_id = int(((0 if seed is None else int(seed) * 131) + attempt + offset) % total)
        options = {"time serie id": time_series_id}
        base_obs = env.reset(seed=seed, options=options)
        load_p = [float(v) for v in (base_obs.load_p * load_scale).astype(float).tolist()]
        scenario_options = {
            "time serie id": time_series_id,
            "init state": {
                "set_line_status": [(line_id, -1) for line_id in faulted_lines],
                "injection": {"load_p": load_p},
            },
        }
        obs = env.reset(seed=seed, options=scenario_options)
        if _multi_stage_survives_probe(env, min_steps=5):
            replayed = env.reset(seed=seed, options=scenario_options)
            logger.info(
                "Initialized multi_stage_cascade with stage=%s faulted_lines=%s load_scale=%.2f time_series_id=%s max_rho=%.4f overloaded=%s",
                stage,
                faulted_lines,
                load_scale,
                time_series_id,
                float(max(replayed.rho)),
                sum(1 for value in replayed.rho.tolist() if float(value) > 1.0),
            )
            return replayed, {
                "faulted_lines": faulted_lines,
                "load_scale": load_scale,
                "time_series_id": time_series_id,
                "curriculum_episode": 1,
                "curriculum_stage": stage,
                "scenario_mode": scenario_mode,
                "benchmark_tier": benchmark_tier or "multi_stage_cascade_expert",
                "benchmark_valid": True,
                "stage_count": 3,
                "stage_length": 10,
                "initial_total_load_mw": round(float(sum(load_p)), 6),
                "overflow_window": 2,
                "do_nothing_probe_steps": 5,
            }

    raise Grid2OpException(
        "Could not find a viable multi_stage_cascade chronic where do_nothing survives 5 steps under the calibrated 3-line +15% load scenario"
    )


def _multi_stage_survives_probe(env, min_steps: int) -> bool:
    obs = None
    for _ in range(int(min_steps)):
        obs, _, done, _ = env.step(env.action_space())
        if done:
            return False
        if obs is None or not obs.rho.tolist():
            return False
        if max(float(value) for value in obs.rho.tolist()) <= 0.0:
            return False
    return obs is not None


def _convert(obs) -> GridObservation:
    return GridObservation(
        rho=[float(x) for x in obs.rho.tolist()],
        gen_p=[float(x) for x in obs.gen_p.tolist()],
        load_p=[float(x) for x in obs.load_p.tolist()],
        line_status=[bool(x) for x in obs.line_status.tolist()],
        timestep_overflow=[int(x) for x in obs.timestep_overflow.tolist()],
        sensitivity_guidance=[],
        reward=0.0,
        done=False,
    )
