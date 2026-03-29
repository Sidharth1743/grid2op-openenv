from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from uuid import uuid4

from grid2op.Exceptions import BackendError, Grid2OpException
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import EpisodeStepLog, GridAction, GridObservation, GridState, TaskId
    from .tasks import TASKS, inject_scenario
except ImportError:
    from models import EpisodeStepLog, GridAction, GridObservation, GridState, TaskId
    from server.tasks import TASKS, inject_scenario

try:
    from lightsim2grid.solver import SolverError
except ImportError:  # pragma: no cover
    SolverError = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CONVERGENCE_EXCEPTIONS = (Grid2OpException, BackendError) + (
    (SolverError,) if SolverError is not None else ()
)


class GridEnvironment(Environment[GridAction, GridObservation, GridState]):
    """Core OpenEnv adapter around Grid2Op."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, env_name: str = "l2rpn_case14_sandbox"):
        import grid2op

        super().__init__()
        self._env_name = env_name
        self._env = grid2op.make(env_name)
        self._last_obs = None
        self._task_id: TaskId = "single_fault"
        self._max_steps = TASKS[self._task_id].max_steps
        self._action_history: deque[str] = deque(maxlen=3)
        self._state = GridState(
            episode_id=str(uuid4()),
            env_name=env_name,
            task_id=self._task_id,
            max_steps=self._max_steps,
            n_line=int(self._env.n_line),
            n_gen=int(self._env.n_gen),
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: TaskId = "single_fault",
        **kwargs,
    ) -> GridObservation:
        del kwargs
        self._task_id = task_id
        self._max_steps = TASKS[task_id].max_steps
        self._action_history.clear()
        logger.info(
            "Resetting environment env_name=%s task_id=%s seed=%s episode_id=%s",
            self._env_name,
            task_id,
            seed,
            episode_id,
        )

        observation, scenario_metadata = inject_scenario(self._env, task_id, seed=seed)
        self._last_obs = observation
        self._state = GridState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            env_name=self._env_name,
            task_id=task_id,
            max_steps=self._max_steps,
            n_line=int(self._env.n_line),
            n_gen=int(self._env.n_gen),
            last_reward=0.0,
            done=False,
            episode_log=[],
            scenario_metadata=scenario_metadata,
        )
        logger.info(
            "Reset complete episode_id=%s task_id=%s max_steps=%s scenario_metadata=%s initial_max_rho=%.4f",
            self._state.episode_id,
            task_id,
            self._max_steps,
            scenario_metadata,
            float(max(observation.rho)) if observation.rho else 0.0,
        )
        return observation

    def step(
        self,
        action: GridAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> GridObservation:
        del timeout_s, kwargs

        sanitized_action, invalid_action, invalid_reason, signature = self._sanitize_action(
            action
        )
        self._action_history.append(signature)

        invalid_penalty = (
            -0.1 if invalid_action and sanitized_action.do_nothing and not action.do_nothing else 0.0
        )
        logger.info(
            "Executing step episode_id=%s task_id=%s next_step=%s action=%s sanitized_action=%s invalid_action=%s",
            self._state.episode_id,
            self._task_id,
            self._state.step_count + 1,
            action.model_dump(),
            sanitized_action.model_dump(),
            invalid_action,
        )
        try:
            grid_action = self._to_grid2op_action(sanitized_action)
            obs, raw_reward, env_done, info = self._env.step(grid_action)
            observation = self._convert_observation(
                obs,
                reward=0.0,
                done=False,
                metadata={"exceptions": [str(exc) for exc in info.get("exception", [])]},
            )
            convergence_failed = False
        except CONVERGENCE_EXCEPTIONS as exc:
            observation = self._handle_convergence_failure(exc)
            raw_reward = 0.0
            env_done = True
            convergence_failed = True
            info = {"exception": [exc]}
            logger.exception(
                "Convergence failure episode_id=%s task_id=%s step=%s",
                self._state.episode_id,
                self._task_id,
                self._state.step_count + 1,
            )

        self._state.step_count += 1
        reached_time_limit = self._state.step_count >= self._max_steps
        done = bool(env_done or reached_time_limit or observation.done)

        shaped_reward = self._shape_reward(
            observation=observation,
            done=done,
            reached_time_limit=reached_time_limit,
            invalid_penalty=invalid_penalty,
        )
        observation.reward = shaped_reward
        observation.done = done
        observation.metadata.update(
            {
                "task_id": self._task_id,
                "step_count": self._state.step_count,
                "max_steps": self._max_steps,
                "raw_reward": float(raw_reward),
            }
        )

        log_entry = self._build_log_entry(
            observation=observation,
            raw_reward=float(raw_reward),
            action=sanitized_action,
            invalid_action=invalid_action,
            invalid_reason=invalid_reason,
            convergence_failed=convergence_failed,
        )
        self._state.episode_log.append(log_entry)
        self._state.last_reward = shaped_reward
        self._state.done = done
        self._last_obs = observation
        logger.info(
            "Step complete episode_id=%s task_id=%s step=%s reward=%.4f raw_reward=%.4f done=%s max_rho=%.4f overloaded=%s exceptions=%s",
            self._state.episode_id,
            self._task_id,
            self._state.step_count,
            shaped_reward,
            float(raw_reward),
            done,
            float(max(observation.rho)) if observation.rho else 0.0,
            log_entry.overloaded_line_ids,
            observation.metadata.get("exceptions", []),
        )
        return observation

    @property
    def state(self) -> GridState:
        return self._state

    def close(self) -> None:
        logger.info(
            "Closing environment episode_id=%s task_id=%s",
            self._state.episode_id,
            self._task_id,
        )
        self._env.close()

    def _sanitize_action(
        self, action: GridAction
    ) -> tuple[GridAction, bool, str | None, str]:
        if action.do_nothing:
            signature = json.dumps({"do_nothing": True}, sort_keys=True)
            return GridAction(do_nothing=True), False, None, signature

        valid_line_set = {}
        invalid_parts = []
        for line_id, status in action.line_set.items():
            if 0 <= int(line_id) < int(self._env.n_line) and int(status) in (-1, 1):
                valid_line_set[int(line_id)] = int(status)
            else:
                invalid_parts.append(f"line_set[{line_id}]={status}")

        valid_redispatch = {}
        for gen_id, delta in action.redispatch.items():
            if 0 <= int(gen_id) < int(self._env.n_gen):
                valid_redispatch[int(gen_id)] = float(delta)
            else:
                invalid_parts.append(f"redispatch[{gen_id}]={delta}")

        if not valid_line_set and not valid_redispatch:
            if invalid_parts:
                sanitized = GridAction(do_nothing=True)
                signature = json.dumps({"do_nothing": True}, sort_keys=True)
                return sanitized, True, ", ".join(invalid_parts), signature
            sanitized = GridAction(do_nothing=True)
            signature = json.dumps({"do_nothing": True}, sort_keys=True)
            return sanitized, False, None, signature

        sanitized = GridAction(
            line_set=valid_line_set,
            redispatch=valid_redispatch,
            do_nothing=False,
        )
        signature = json.dumps(
            {
                "line_set": sanitized.line_set,
                "redispatch": sanitized.redispatch,
                "do_nothing": sanitized.do_nothing,
            },
            sort_keys=True,
        )
        return sanitized, bool(invalid_parts), ", ".join(invalid_parts) or None, signature

    def _to_grid2op_action(self, action: GridAction):
        if action.do_nothing:
            return self._env.action_space()

        payload = {}
        if action.line_set:
            payload["set_line_status"] = sorted(action.line_set.items())
        if action.redispatch:
            payload["redispatch"] = sorted(action.redispatch.items())
        return self._env.action_space(payload)

    def _convert_observation(
        self,
        obs,
        reward: float,
        done: bool,
        metadata: dict | None = None,
    ) -> GridObservation:
        if isinstance(obs, GridObservation):
            obs.reward = reward
            obs.done = done
            obs.metadata.update(metadata or {})
            return obs

        return GridObservation(
            rho=[float(x) for x in obs.rho.tolist()],
            gen_p=[float(x) for x in obs.gen_p.tolist()],
            load_p=[float(x) for x in obs.load_p.tolist()],
            line_status=[bool(x) for x in obs.line_status.tolist()],
            timestep_overflow=[int(x) for x in obs.timestep_overflow.tolist()],
            reward=reward,
            done=done,
            metadata=metadata or {},
        )

    def _handle_convergence_failure(self, exc: Exception) -> GridObservation:
        if self._last_obs is None:
            raise
        return self._convert_observation(
            self._last_obs,
            reward=-10.0,
            done=True,
            metadata={"exceptions": [str(exc)], "convergence_failed": True},
        )

    def _shape_reward(
        self,
        observation: GridObservation,
        done: bool,
        reached_time_limit: bool,
        invalid_penalty: float,
    ) -> float:
        overloaded = [rho for rho in observation.rho if rho > 1.0]
        reward = 0.0
        if observation.rho and max(observation.rho) < 0.8:
            reward += 0.1
        reward -= 0.2 * len(overloaded)
        reward += invalid_penalty

        if len(self._action_history) == 3 and len(set(self._action_history)) == 1:
            reward -= 0.05

        if observation.metadata.get("convergence_failed"):
            reward -= 10.0
        elif done and not reached_time_limit:
            reward -= 10.0
        elif done and reached_time_limit:
            reward += 5.0

        return float(reward)

    def _build_log_entry(
        self,
        observation: GridObservation,
        raw_reward: float,
        action: GridAction,
        invalid_action: bool,
        invalid_reason: str | None,
        convergence_failed: bool,
    ) -> EpisodeStepLog:
        overloaded_ids = [
            idx for idx, rho in enumerate(observation.rho) if float(rho) > 1.0
        ]
        disconnected = [
            idx for idx, status in enumerate(observation.line_status) if not status
        ]
        return EpisodeStepLog(
            step=self._state.step_count,
            task_id=self._task_id,
            reward=float(observation.reward or 0.0),
            raw_reward=raw_reward,
            done=bool(observation.done),
            max_rho=float(max(observation.rho)) if observation.rho else 0.0,
            overloaded_line_ids=overloaded_ids,
            all_lines_below_80=all(rho < 0.8 for rho in observation.rho),
            all_lines_below_90=all(rho < 0.9 for rho in observation.rho),
            all_lines_below_100=all(rho < 1.0 for rho in observation.rho),
            disconnected_lines=disconnected,
            timestep_overflow=observation.timestep_overflow,
            invalid_action=invalid_action,
            invalid_action_reason=invalid_reason,
            convergence_failed=convergence_failed,
            action=action.model_dump(),
        )


def smoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="single_fault")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    env = GridEnvironment()
    try:
        obs = env.reset(task_id=args.task_id)
        result = {
            "env_name": env.state.env_name,
            "task_id": args.task_id,
            "n_line": env.state.n_line,
            "n_gen": env.state.n_gen,
            "initial_max_rho": max(obs.rho) if obs.rho else 0.0,
            "steps": [],
        }
        for index in range(args.steps):
            next_obs = env.step(GridAction(do_nothing=True))
            result["steps"].append(
                {
                    "step": index + 1,
                    "reward": next_obs.reward,
                    "done": next_obs.done,
                    "max_rho": max(next_obs.rho) if next_obs.rho else 0.0,
                    "exceptions": next_obs.metadata.get("exceptions", []),
                }
            )
            if next_obs.done:
                break
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        env.close()
