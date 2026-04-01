from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import deque
from uuid import uuid4

from grid2op.Exceptions import BackendError, Grid2OpException
from openenv.core.env_server.interfaces import Environment

try:
    from ..graph_analysis import analyze_grid_topology
    from ..models import (
        EpisodeStepLog,
        GridAction,
        GridObservation,
        GridState,
        PlanningContextResponse,
        RedispatchGeneratorContext,
        ScenarioMode,
        SimulationResult,
        TaskId,
    )
    from .tasks import TASKS, inject_scenario_raw
except ImportError:
    from graph_analysis import analyze_grid_topology
    from models import (
        EpisodeStepLog,
        GridAction,
        GridObservation,
        GridState,
        PlanningContextResponse,
        RedispatchGeneratorContext,
        ScenarioMode,
        SimulationResult,
        TaskId,
    )
    from server.tasks import TASKS, inject_scenario_raw

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
    SINGLE_FAULT_REDISPATCH_PENALTY_PER_MW: float = 0.01
    _instances_by_episode_id: dict[str, "GridEnvironment"] = {}
    _instances_lock = threading.RLock()

    def __init__(self, env_name: str = "l2rpn_case14_sandbox"):
        import grid2op

        super().__init__()
        self._env_name = env_name
        self._env = grid2op.make(env_name)
        self._last_obs = None
        self._last_raw_obs = None
        self._task_id: TaskId = "single_fault"
        self._max_steps = TASKS[self._task_id].max_steps
        self._action_history: deque[str] = deque(maxlen=3)
        self._previous_max_rho: float | None = None
        self._previous_topology_change_count: int = 0
        self._instance_lock = threading.RLock()
        self._state = GridState(
            episode_id=str(uuid4()),
            env_name=env_name,
            task_id=self._task_id,
            max_steps=self._max_steps,
            n_line=int(self._env.n_line),
            n_gen=int(self._env.n_gen),
        )
        self._register_instance(self._state.episode_id)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: TaskId = "single_fault",
        difficulty_level: int | None = None,
        scenario_mode: ScenarioMode = "curriculum",
        benchmark_tier: str | None = None,
        **kwargs,
    ) -> GridObservation:
        del kwargs
        with self._instance_lock:
            previous_episode_id = self._state.episode_id
            self._task_id = task_id
            self._max_steps = TASKS[task_id].max_steps
            self._action_history.clear()
            self._previous_max_rho = None
            self._previous_topology_change_count = 0
            logger.info(
                "Resetting environment env_name=%s task_id=%s seed=%s episode_id=%s difficulty_level=%s",
                self._env_name,
                task_id,
                seed,
                episode_id,
                difficulty_level,
            )

            raw_observation, scenario_metadata = inject_scenario_raw(
                self._env,
                task_id,
                seed=seed,
                difficulty_level=difficulty_level,
                scenario_mode=scenario_mode,
                benchmark_tier=benchmark_tier,
            )
            observation = self._convert_observation(
                raw_observation,
                reward=0.0,
                done=False,
                metadata={},
            )
            self._last_raw_obs = raw_observation
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
            self._unregister_instance(previous_episode_id)
            self._register_instance(self._state.episode_id)
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

        with self._instance_lock:
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
                previous_observation = self._last_obs
                grid_action = self._to_grid2op_action(sanitized_action)
                raw_obs, raw_reward, env_done, info = self._env.step(grid_action)
                observation = self._convert_observation(
                    raw_obs,
                    reward=0.0,
                    done=False,
                    metadata={"exceptions": [str(exc) for exc in info.get("exception", [])]},
                )
                convergence_failed = False
            except CONVERGENCE_EXCEPTIONS as exc:
                previous_observation = self._last_obs
                observation = self._handle_convergence_failure(exc)
                raw_reward = 0.0
                env_done = True
                convergence_failed = True
                info = {"exception": [exc]}
                raw_obs = None
                logger.exception(
                    "Convergence failure episode_id=%s task_id=%s step=%s",
                    self._state.episode_id,
                    self._task_id,
                    self._state.step_count + 1,
                )

            self._state.step_count += 1
            reached_time_limit = self._state.step_count >= self._max_steps
            single_fault_target_threshold = self._single_fault_success_threshold()
            all_lines_below_target = bool(observation.rho) and all(
                rho < single_fault_target_threshold for rho in observation.rho
            )
            all_lines_below_80 = bool(observation.rho) and all(rho < 0.8 for rho in observation.rho)
            topology_change_count = self._compute_topology_change_count(previous_observation, observation)
            auto_trip_detected = self._detect_auto_trip(previous_observation, observation, sanitized_action)
            if self._task_id == "single_fault" and all_lines_below_target:
                done = True
            else:
                done = bool(env_done or reached_time_limit or observation.done)

            shaped_reward = self._shape_reward(
                observation=observation,
                done=done,
                reached_time_limit=reached_time_limit,
                invalid_penalty=invalid_penalty,
                auto_trip_detected=auto_trip_detected,
                topology_change_count=topology_change_count,
                action=sanitized_action,
            )
            observation.reward = shaped_reward
            observation.done = done
            observation.metadata.update(
                {
                    "task_id": self._task_id,
                    "step_count": self._state.step_count,
                    "max_steps": self._max_steps,
                    "raw_reward": float(raw_reward),
                    "redispatch_mw": self._redispatch_magnitude(sanitized_action),
                    "action_penalty": self._action_penalty(sanitized_action),
                }
            )

            log_entry = self._build_log_entry(
                observation=observation,
                raw_reward=float(raw_reward),
                action=sanitized_action,
                invalid_action=invalid_action,
                invalid_reason=invalid_reason,
                convergence_failed=convergence_failed,
                topology_change_count=topology_change_count,
                auto_trip_detected=auto_trip_detected,
            )
            self._state.episode_log.append(log_entry)
            self._state.last_reward = shaped_reward
            self._state.done = done
            self._last_obs = observation
            if raw_obs is not None:
                self._last_raw_obs = raw_obs
            self._previous_max_rho = float(max(observation.rho)) if observation.rho else None
            self._previous_topology_change_count = topology_change_count
            logger.info(
                "Step complete episode_id=%s task_id=%s step=%s reward=%.4f raw_reward=%.4f done=%s max_rho=%.4f overloaded=%s auto_trip=%s topo_changes=%s exceptions=%s",
                self._state.episode_id,
                self._task_id,
                self._state.step_count,
                shaped_reward,
                float(raw_reward),
                done,
                float(max(observation.rho)) if observation.rho else 0.0,
                log_entry.overloaded_line_ids,
                auto_trip_detected,
                topology_change_count,
                observation.metadata.get("exceptions", []),
            )
            return observation

    @property
    def state(self) -> GridState:
        return self._state

    def close(self) -> None:
        with self._instance_lock:
            logger.info(
                "Closing environment episode_id=%s task_id=%s",
                self._state.episode_id,
                self._task_id,
            )
            self._unregister_instance(self._state.episode_id)
            self._env.close()

    def get_planning_context(self) -> PlanningContextResponse:
        with self._instance_lock:
            if self._last_raw_obs is None:
                raise RuntimeError("No active raw observation available for planning context")
            graph_intelligence = analyze_grid_topology(
                self._last_raw_obs,
                line_or_to_subid=self._env.line_or_to_subid.tolist(),
                line_ex_to_subid=self._env.line_ex_to_subid.tolist(),
                n_sub=int(self._env.n_sub),
            )
            redispatchable_generators = [
                idx for idx, allowed in enumerate(self._env.gen_redispatchable.tolist()) if bool(allowed)
            ]
            redispatch_generators = [
                RedispatchGeneratorContext(
                    gen_id=int(gen_id),
                    p_mw=bounds["p_mw"],
                    max_ramp_up=bounds["max_ramp_up"],
                    max_ramp_down=bounds["max_ramp_down"],
                    allowed_delta_min=bounds["allowed_delta_min"],
                    allowed_delta_max=bounds["allowed_delta_max"],
                    allowed_deltas=bounds["allowed_deltas"],
                )
                for gen_id in redispatchable_generators
                for bounds in [self._redispatch_bounds_for_gen(gen_id)]
            ]
            return PlanningContextResponse(
                episode_id=self._state.episode_id,
                graph_intelligence=graph_intelligence,
                redispatchable_generators=redispatchable_generators,
                redispatch_generators=redispatch_generators,
            )

    def simulate_actions(self, actions: list[GridAction]) -> list[SimulationResult]:
        with self._instance_lock:
            if self._last_raw_obs is None:
                raise RuntimeError("No active raw observation available for simulation")
            results: list[SimulationResult] = []
            for action in actions:
                sanitized_action, _, _, _ = self._sanitize_action(action)
                try:
                    sim_obs, sim_reward, sim_done, sim_info = self._last_raw_obs.simulate(
                        self._to_grid2op_action(sanitized_action)
                    )
                    exceptions = [str(exc) for exc in sim_info.get("exception", [])]
                    max_rho = float(max(sim_obs.rho)) if len(sim_obs.rho) else 0.0
                    overloaded = [
                        idx for idx, rho in enumerate(sim_obs.rho.tolist()) if float(rho) > 1.0
                    ]
                    disconnected = [
                        idx for idx, status in enumerate(sim_obs.line_status.tolist()) if not bool(status)
                    ]
                    results.append(
                        SimulationResult(
                            action=sanitized_action,
                            max_rho=max_rho,
                            done=bool(sim_done),
                            simulated_reward=float(sim_reward),
                            overloaded_line_ids=overloaded,
                            disconnected_lines=disconnected,
                            convergence_failed=bool(
                                exceptions
                                or sim_info.get("is_illegal")
                                or sim_info.get("is_ambiguous")
                                or sim_info.get("is_done")
                                or max_rho == 0.0
                            ),
                            exceptions=exceptions,
                            raw_result={
                                "done": bool(sim_done),
                                "simulated_reward": float(sim_reward),
                                "max_rho": max_rho,
                                "overloaded_line_ids": overloaded,
                                "disconnected_lines": disconnected,
                                "exceptions": exceptions,
                                "timestep_overflow": [int(x) for x in sim_obs.timestep_overflow.tolist()],
                            },
                        )
                    )
                except CONVERGENCE_EXCEPTIONS as exc:
                    results.append(
                        SimulationResult(
                            action=sanitized_action,
                            max_rho=999.0,
                            done=True,
                            simulated_reward=-10.0,
                            overloaded_line_ids=[],
                            disconnected_lines=[],
                            convergence_failed=True,
                            exceptions=[str(exc)],
                            raw_result={
                                "done": True,
                                "simulated_reward": -10.0,
                                "max_rho": 999.0,
                                "overloaded_line_ids": [],
                                "disconnected_lines": [],
                                "exceptions": [str(exc)],
                                "timestep_overflow": [],
                            },
                        )
                    )
            return results

    @classmethod
    def get_active_instance(cls, episode_id: str) -> "GridEnvironment" | None:
        with cls._instances_lock:
            return cls._instances_by_episode_id.get(episode_id)

    def _register_instance(self, episode_id: str) -> None:
        with self._instances_lock:
            self._instances_by_episode_id[episode_id] = self

    def _unregister_instance(self, episode_id: str) -> None:
        with self._instances_lock:
            existing = self._instances_by_episode_id.get(episode_id)
            if existing is self:
                self._instances_by_episode_id.pop(episode_id, None)

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
        redispatchable = {
            idx for idx, allowed in enumerate(self._env.gen_redispatchable.tolist()) if bool(allowed)
        }
        for gen_id, delta in action.redispatch.items():
            gen_id_int = int(gen_id)
            if 0 <= gen_id_int < int(self._env.n_gen) and gen_id_int in redispatchable:
                bounds = self._redispatch_bounds_for_gen(gen_id_int)
                constrained_delta = self._constrain_redispatch_delta(
                    float(delta),
                    bounds["allowed_delta_min"],
                    bounds["allowed_delta_max"],
                    bounds["allowed_deltas"],
                )
                if constrained_delta is None:
                    invalid_parts.append(f"redispatch[{gen_id}]={delta}")
                else:
                    valid_redispatch[gen_id_int] = constrained_delta
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

        sensitivity_guidance = self._build_sensitivity_guidance(obs)

        return GridObservation(
            rho=[float(x) for x in obs.rho.tolist()],
            gen_p=[float(x) for x in obs.gen_p.tolist()],
            load_p=[float(x) for x in obs.load_p.tolist()],
            line_status=[bool(x) for x in obs.line_status.tolist()],
            timestep_overflow=[int(x) for x in obs.timestep_overflow.tolist()],
            sensitivity_guidance=sensitivity_guidance,
            reward=reward,
            done=done,
            metadata=metadata or {},
        )

    def _handle_convergence_failure(self, exc: Exception) -> GridObservation:
        if self._last_obs is None:
            return GridObservation(
                rho=[0.0 for _ in range(int(self._env.n_line))],
                gen_p=[0.0 for _ in range(int(self._env.n_gen))],
                load_p=[],
                line_status=[False for _ in range(int(self._env.n_line))],
                timestep_overflow=[0 for _ in range(int(self._env.n_line))],
                sensitivity_guidance=[],
                reward=-10.0,
                done=True,
                metadata={"exceptions": [str(exc)], "convergence_failed": True},
            )
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
        auto_trip_detected: bool,
        topology_change_count: int,
        action: GridAction,
    ) -> float:
        max_rho = float(max(observation.rho)) if observation.rho else 0.0
        single_fault_target_threshold = self._single_fault_success_threshold()
        overloaded_count = sum(1 for rho in observation.rho if rho > 1.0)
        safe_line_ratio = (
            sum(1 for rho in observation.rho if rho < 0.8) / len(observation.rho)
            if observation.rho
            else 0.0
        )
        reward = 0.0

        if self._task_id == "single_fault":
            if observation.rho and all(rho < single_fault_target_threshold for rho in observation.rho):
                reward += 1.0 / max(1, self._state.step_count)
            reward += 0.05 * max(0.0, 1.0 - max_rho)
            reward -= 0.2 * overloaded_count
            reward -= self._action_penalty(action)
            if reached_time_limit and not all(rho < single_fault_target_threshold for rho in observation.rho):
                reward -= 5.0

        elif self._task_id == "n_minus_1":
            reward += 0.1 * safe_line_ratio
            if self._previous_max_rho is not None:
                if max_rho < self._previous_max_rho:
                    reward += 0.05
                elif max_rho > self._previous_max_rho:
                    reward -= 0.05
            reward -= 0.3 * overloaded_count
            if reached_time_limit and not observation.metadata.get("convergence_failed"):
                reward += 3.0
            elif done and not reached_time_limit:
                reward -= 8.0

        elif self._task_id == "cascade_prevent":
            if not auto_trip_detected:
                reward += 0.2
            reward -= 0.1 * sum(int(value) for value in observation.timestep_overflow)
            if topology_change_count == 0:
                reward += 0.05
            if auto_trip_detected:
                reward -= 2.0
            if observation.metadata.get("convergence_failed"):
                reward -= 10.0
            elif done and not reached_time_limit:
                reward -= 10.0
            elif reached_time_limit:
                reward += 5.0

        reward += invalid_penalty
        return float(reward)

    def _build_log_entry(
        self,
        observation: GridObservation,
        raw_reward: float,
        action: GridAction,
        invalid_action: bool,
        invalid_reason: str | None,
        convergence_failed: bool,
        topology_change_count: int,
        auto_trip_detected: bool,
    ) -> EpisodeStepLog:
        overloaded_ids = [
            idx for idx, rho in enumerate(observation.rho) if float(rho) > 1.0
        ]
        disconnected = [
            idx for idx, status in enumerate(observation.line_status) if not status
        ]
        single_fault_target_threshold = self._single_fault_success_threshold()
        redispatch_mw = self._redispatch_magnitude(action)
        action_penalty = self._action_penalty(action)
        return EpisodeStepLog(
            step=self._state.step_count,
            task_id=self._task_id,
            reward=float(observation.reward or 0.0),
            raw_reward=raw_reward,
            done=bool(observation.done),
            max_rho=float(max(observation.rho)) if observation.rho else 0.0,
            redispatch_mw=redispatch_mw,
            action_penalty=action_penalty,
            overloaded_line_ids=overloaded_ids,
            single_fault_target_threshold=single_fault_target_threshold,
            all_lines_below_target=all(rho < single_fault_target_threshold for rho in observation.rho),
            all_lines_below_80=all(rho < 0.8 for rho in observation.rho),
            all_lines_below_90=all(rho < 0.9 for rho in observation.rho),
            all_lines_below_100=all(rho < 1.0 for rho in observation.rho),
            disconnected_lines=disconnected,
            timestep_overflow=observation.timestep_overflow,
            safe_line_ratio=(
                sum(1 for rho in observation.rho if rho < 0.8) / len(observation.rho)
                if observation.rho
                else 0.0
            ),
            topology_change_count=topology_change_count,
            auto_trip_detected=auto_trip_detected,
            invalid_action=invalid_action,
            invalid_action_reason=invalid_reason,
            convergence_failed=convergence_failed,
            action=action.model_dump(),
        )

    def _single_fault_success_threshold(self) -> float:
        if self._task_id != "single_fault":
            return 0.8
        benchmark_tier = self._state.scenario_metadata.get("benchmark_tier")
        if benchmark_tier == "single_fault_severe":
            return 0.9
        return 0.8

    @staticmethod
    def _redispatch_magnitude(action: GridAction) -> float:
        if action.do_nothing or not action.redispatch:
            return 0.0
        return float(sum(abs(float(delta)) for delta in action.redispatch.values()))

    def _action_penalty(self, action: GridAction) -> float:
        if self._task_id != "single_fault":
            return 0.0
        return self.SINGLE_FAULT_REDISPATCH_PENALTY_PER_MW * self._redispatch_magnitude(action)

    def _compute_topology_change_count(
        self,
        previous_observation: GridObservation | None,
        observation: GridObservation,
    ) -> int:
        if previous_observation is None:
            return 0
        previous_status = previous_observation.line_status
        current_status = observation.line_status
        return sum(
            1
            for previous, current in zip(previous_status, current_status)
            if bool(previous) != bool(current)
        )

    def _redispatch_bounds_for_gen(
        self,
        gen_id: int,
        obs=None,
    ) -> dict[str, float | list[float]]:
        raw_obs = obs if obs is not None else self._last_raw_obs
        if raw_obs is None:
            raise RuntimeError("No active raw observation available for redispatch bounds")

        current_p = float(raw_obs.gen_p[gen_id])
        pmin = float(self._env.gen_pmin[gen_id])
        pmax = float(self._env.gen_pmax[gen_id])
        max_ramp_up = float(self._env.gen_max_ramp_up[gen_id])
        max_ramp_down = float(self._env.gen_max_ramp_down[gen_id])
        allowed_up = max(0.0, min(max_ramp_up, pmax - current_p))
        allowed_down = max(0.0, min(max_ramp_down, current_p - pmin))
        allowed_delta_min = -allowed_down
        allowed_delta_max = allowed_up
        allowed_deltas = self._discretize_allowed_deltas(allowed_down, allowed_up)
        return {
            "p_mw": round(current_p, 4),
            "max_ramp_up": round(max_ramp_up, 4),
            "max_ramp_down": round(max_ramp_down, 4),
            "allowed_delta_min": round(allowed_delta_min, 4),
            "allowed_delta_max": round(allowed_delta_max, 4),
            "allowed_deltas": allowed_deltas,
        }

    @staticmethod
    def _discretize_allowed_deltas(allowed_down: float, allowed_up: float) -> list[float]:
        candidates = [0.0]
        if allowed_down > 0.0:
            candidates.extend([-allowed_down, -(allowed_down / 2.0)])
        if allowed_up > 0.0:
            candidates.extend([allowed_up / 2.0, allowed_up])
        deduped = sorted({round(value, 4) for value in candidates if abs(value) > 1e-9})
        return deduped

    @staticmethod
    def _constrain_redispatch_delta(
        delta: float,
        delta_min: float,
        delta_max: float,
        allowed_deltas: list[float],
    ) -> float | None:
        if delta_min == 0.0 and delta_max == 0.0:
            return None
        clamped = min(max(float(delta), float(delta_min)), float(delta_max))
        feasible = [float(value) for value in allowed_deltas if delta_min <= float(value) <= delta_max]
        if feasible:
            return min(feasible, key=lambda value: (abs(value - clamped), abs(value), value))
        if abs(clamped) < 1e-9:
            return None
        return round(clamped, 4)

    def _build_sensitivity_guidance(self, obs) -> list[dict[str, float | int | str]]:
        rho_values = [float(value) for value in obs.rho.tolist()]
        if not rho_values:
            return []

        current_global_max_rho = max(rho_values)
        candidates: list[dict[str, float | int | str]] = []

        allow_topology_guidance = self._task_id != "single_fault"
        if allow_topology_guidance:
            line_status = [bool(value) for value in obs.line_status.tolist()]
            for line_id, is_connected in enumerate(line_status):
                if not is_connected:
                    continue
                try:
                    sim_obs, _, sim_done, sim_info = obs.simulate(
                        self._env.action_space({"set_line_status": [(line_id, -1)]})
                    )
                except CONVERGENCE_EXCEPTIONS:
                    continue

                sim_exceptions = sim_info.get("exception", [])
                if bool(sim_done) or sim_exceptions:
                    continue

                sim_rho_values = [float(value) for value in sim_obs.rho.tolist()]
                if not sim_rho_values:
                    continue
                global_rho_change = float(max(sim_rho_values) - current_global_max_rho)
                if global_rho_change >= 0.0:
                    continue
                candidates.append(
                    {
                        "action_type": "disconnect_line",
                        "target_id": int(line_id),
                        "expected_rho_change": round(global_rho_change, 6),
                    }
                )

        for gen_id, allowed in enumerate(self._env.gen_redispatchable.tolist()):
            if not bool(allowed):
                continue
            bounds = self._redispatch_bounds_for_gen(int(gen_id), obs=obs)
            for delta in bounds["allowed_deltas"]:
                delta_value = float(delta)
                if abs(delta_value) < 1e-9:
                    continue
                try:
                    sim_obs, _, sim_done, sim_info = obs.simulate(
                        self._env.action_space({"redispatch": [(int(gen_id), delta_value)]})
                    )
                except CONVERGENCE_EXCEPTIONS:
                    continue

                sim_exceptions = sim_info.get("exception", [])
                if bool(sim_done) or sim_exceptions:
                    continue

                sim_rho_values = [float(value) for value in sim_obs.rho.tolist()]
                if not sim_rho_values:
                    continue
                global_rho_change = float(max(sim_rho_values) - current_global_max_rho)
                if global_rho_change >= 0.0:
                    continue
                candidates.append(
                    {
                        "action_type": "redispatch",
                        "target_id": int(gen_id),
                        "expected_rho_change": round(global_rho_change, 6),
                    }
                )

        return sorted(
            candidates,
            key=lambda item: float(item["expected_rho_change"]),
        )[:3]

    def _detect_auto_trip(
        self,
        previous_observation: GridObservation | None,
        observation: GridObservation,
        action: GridAction,
    ) -> bool:
        if previous_observation is None:
            return False
        requested_disconnects = {
            int(line_id)
            for line_id, status in action.line_set.items()
            if int(status) == -1
        }
        for idx, (before, after) in enumerate(
            zip(previous_observation.line_status, observation.line_status)
        ):
            if bool(before) and not bool(after) and idx not in requested_disconnects:
                return True
        return False


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
