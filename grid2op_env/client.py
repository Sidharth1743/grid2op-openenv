from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import GridAction, GridObservation, GridState


class GridEnv(EnvClient[GridAction, GridObservation, GridState]):
    """Client for the standalone Grid2Op environment server."""

    def _step_payload(self, action: GridAction) -> Dict[str, Any]:
        return {
            "line_set": action.line_set,
            "redispatch": action.redispatch,
            "do_nothing": action.do_nothing,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GridObservation]:
        obs_data = payload.get("observation", {})
        observation = GridObservation(
            rho=obs_data.get("rho", []),
            gen_p=obs_data.get("gen_p", []),
            load_p=obs_data.get("load_p", []),
            line_status=obs_data.get("line_status", []),
            timestep_overflow=obs_data.get("timestep_overflow", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GridState:
        return GridState.model_validate(payload)


# Backward-compatible alias for the earlier bootstrap name.
Grid2OpEnv = GridEnv
