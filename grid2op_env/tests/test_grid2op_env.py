from __future__ import annotations

from grid2op_env.models import EpisodeStepLog, GridAction, GridState
from grid2op_env.server.graders import (
    grade_cascade_prevent,
    grade_n_minus_1,
    grade_single_fault,
)
from grid2op_env.server.grid_environment import GridEnvironment
from grid2op_env.server.tasks import TASKS


def test_task_resets_and_logs():
    env = GridEnvironment()
    try:
        for task_id, spec in TASKS.items():
            obs = env.reset(task_id=task_id)
            assert obs.rho
            assert env.state.task_id == task_id
            assert env.state.max_steps == spec.max_steps

            next_obs = env.step(GridAction(do_nothing=True))
            assert isinstance(next_obs.reward, float)
            assert env.state.episode_log
            assert env.state.episode_log[-1].task_id == task_id
    finally:
        env.close()


def test_state_model_validation():
    state = GridState(
        episode_id="ep",
        step_count=1,
        task_id="single_fault",
        max_steps=10,
        n_line=20,
        n_gen=6,
    )
    assert state.n_line == 20


def test_graders_are_deterministic():
    single_fault_log = [
        EpisodeStepLog(
            step=1,
            task_id="single_fault",
            reward=1.0,
            raw_reward=0.0,
            done=False,
            max_rho=0.85,
            all_lines_below_90=True,
            all_lines_below_100=True,
        )
    ]
    assert grade_single_fault(single_fault_log) == 1.0

    n_minus_1_log = [
        EpisodeStepLog(
            step=idx,
            task_id="n_minus_1",
            reward=0.0,
            raw_reward=0.0,
            done=False,
            max_rho=0.95,
            all_lines_below_100=True,
        )
        for idx in range(1, 21)
    ]
    assert grade_n_minus_1(n_minus_1_log) == 1.0

    cascade_log = [
        EpisodeStepLog(
            step=idx,
            task_id="cascade_prevent",
            reward=0.0,
            raw_reward=0.0,
            done=(idx == 30),
            max_rho=0.99,
            all_lines_below_100=True,
        )
        for idx in range(1, 31)
    ]
    assert grade_cascade_prevent(cascade_log) == 1.0
