from __future__ import annotations

from typing import Any, Dict
import httpx

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    GridAction,
    GridObservation,
    GridState,
    PlanningContextResponse,
    SimulationResponse,
)


class GridEnv(EnvClient[GridAction, GridObservation, GridState]):
    """Client for the standalone Grid2Op environment server."""

    def __init__(self, base_url: str, *args: Any, **kwargs: Any):
        super().__init__(base_url=base_url, *args, **kwargs)
        self._http_base_url = (
            base_url.rstrip("/")
            .replace("ws://", "http://")
            .replace("wss://", "https://")
        )

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

    async def planning_context(self, episode_id: str) -> PlanningContextResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._http_base_url}/planning_context",
                json={"episode_id": episode_id},
                timeout=self._message_timeout,
            )
            response.raise_for_status()
            return PlanningContextResponse.model_validate(response.json())

    async def simulate_candidates(
        self,
        episode_id: str,
        actions: list[GridAction],
    ) -> SimulationResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._http_base_url}/simulate",
                json={
                    "episode_id": episode_id,
                    "actions": [action.model_dump() for action in actions],
                },
                timeout=self._message_timeout,
            )
            response.raise_for_status()
            return SimulationResponse.model_validate(response.json())


# Backward-compatible alias for the earlier bootstrap name.
Grid2OpEnv = GridEnv
