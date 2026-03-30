from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from grid2op.Exceptions import Grid2OpException

try:
    from ..models import GridObservation, TaskId, TaskInfo
except ImportError:
    from models import GridObservation, TaskId, TaskInfo

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
}

CASCADE_LINE_PAIRS: List[tuple[int, int]] = [
    (0, 8),
    (0, 7),
    (0, 3),
]


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
):
    """Initialize the environment according to the selected task and return the raw Grid2Op observation."""

    if task_id not in TASKS:
        raise ValueError(f"Unsupported task_id: {task_id}")

    for attempt in range(max_attempts):
        try:
            logger.info(
                "Injecting scenario task_id=%s attempt=%s seed=%s",
                task_id,
                attempt + 1,
                seed,
            )
            if task_id == "single_fault":
                return _reset_single_fault(env, seed=seed, attempt=attempt)
            if task_id == "n_minus_1":
                return _reset_n_minus_1(env, seed=seed)
            return _reset_cascade_prevent(env, seed=seed, attempt=attempt)
        except Grid2OpException as exc:
            logger.warning(
                "Scenario injection failed task_id=%s attempt=%s error=%s",
                task_id,
                attempt + 1,
                exc,
            )
            if attempt == max_attempts - 1:
                raise
    raise RuntimeError("Unreachable")


def inject_scenario(
    env,
    task_id: TaskId,
    seed: int | None = None,
    max_attempts: int = 3,
) -> tuple[GridObservation, Dict[str, Any]]:
    """Initialize the environment according to the selected task."""

    raw_obs, metadata = inject_scenario_raw(
        env,
        task_id,
        seed=seed,
        max_attempts=max_attempts,
    )
    return _convert(raw_obs), metadata


def _reset_single_fault(env, seed: int | None, attempt: int):
    options = {"time serie id": attempt}
    obs = env.reset(seed=seed, options=options)
    max_warmup = 48
    warmup_steps = 0

    while warmup_steps < max_warmup:
        max_rho = float(max(obs.rho))
        if 0.90 <= max_rho <= 0.98:
            logger.info(
                "Selected single_fault warmup state warmup_steps=%s max_rho=%.4f",
                warmup_steps,
                max_rho,
            )
            return obs, {
                "warmup_steps": warmup_steps,
                "scenario": "high_loading",
            }
        obs, _, done, _ = env.step(env.action_space())
        warmup_steps += 1
        if done:
            break

    raise Grid2OpException(
        f"Could not find a single-fault warmup state after {max_warmup} steps"
    )


def _reset_n_minus_1(env, seed: int | None):
    obs = env.reset(
        seed=seed,
        options={"init state": {"set_line_status": [(0, -1)]}},
    )
    logger.info("Initialized n_minus_1 with faulted_lines=[0]")
    return obs, {"faulted_lines": [0]}


def _reset_cascade_prevent(env, seed: int | None, attempt: int):
    base_obs = env.reset(seed=seed)
    line_pair = CASCADE_LINE_PAIRS[attempt % len(CASCADE_LINE_PAIRS)]
    load_p = [float(v) for v in (base_obs.load_p * 1.15).astype(float).tolist()]
    obs = env.reset(
        seed=seed,
        options={
            "init state": {
                "set_line_status": [(line_pair[0], -1), (line_pair[1], -1)],
                "injection": {"load_p": load_p},
            }
        },
    )
    logger.info(
        "Initialized cascade_prevent with faulted_lines=%s load_scale=1.15 max_rho=%.4f",
        list(line_pair),
        float(max(obs.rho)),
    )
    return obs, {"faulted_lines": list(line_pair), "load_scale": 1.15}


def _convert(obs) -> GridObservation:
    return GridObservation(
        rho=[float(x) for x in obs.rho.tolist()],
        gen_p=[float(x) for x in obs.gen_p.tolist()],
        load_p=[float(x) for x in obs.load_p.tolist()],
        line_status=[bool(x) for x in obs.line_status.tolist()],
        timestep_overflow=[int(x) for x in obs.timestep_overflow.tolist()],
        reward=0.0,
        done=False,
    )
