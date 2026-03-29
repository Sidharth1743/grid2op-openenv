from __future__ import annotations

from typing import Any, Dict, List, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


TaskId = Literal["single_fault", "n_minus_1", "cascade_prevent"]


class GridAction(Action):
    """JSON-serializable subset of Grid2Op actions."""

    line_set: Dict[int, int] = Field(
        default_factory=dict,
        description="Map line id to status. Use -1 to disconnect and 1 to reconnect.",
    )
    redispatch: Dict[int, float] = Field(
        default_factory=dict,
        description="Map generator id to redispatch delta in MW.",
    )
    do_nothing: bool = Field(
        default=False,
        description="When true, ignore other fields and apply the native no-op action.",
    )


class GridObservation(Observation):
    """Typed subset of the Grid2Op observation surface."""

    rho: List[float] = Field(default_factory=list)
    gen_p: List[float] = Field(default_factory=list)
    load_p: List[float] = Field(default_factory=list)
    line_status: List[bool] = Field(default_factory=list)
    timestep_overflow: List[int] = Field(default_factory=list)


class EpisodeStepLog(BaseModel):
    """Structured per-step trace used by graders and debugging."""

    step: int
    task_id: TaskId
    reward: float
    raw_reward: float
    done: bool
    max_rho: float
    overloaded_line_ids: List[int] = Field(default_factory=list)
    all_lines_below_80: bool = False
    all_lines_below_90: bool = False
    all_lines_below_100: bool = False
    disconnected_lines: List[int] = Field(default_factory=list)
    timestep_overflow: List[int] = Field(default_factory=list)
    invalid_action: bool = False
    invalid_action_reason: str | None = None
    convergence_failed: bool = False
    action: Dict[str, Any] = Field(default_factory=dict)


class GridState(State):
    """Environment state for the current Grid2Op episode."""

    env_name: str = "l2rpn_case14_sandbox"
    task_id: TaskId = "single_fault"
    max_steps: int = 0
    n_line: int = 0
    n_gen: int = 0
    last_reward: float = 0.0
    done: bool = False
    episode_log: List[EpisodeStepLog] = Field(default_factory=list)
    scenario_metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    task_id: TaskId
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]
    action_schema: Dict[str, Any]


class GraderRequest(BaseModel):
    task_id: TaskId
    episode_log: List[EpisodeStepLog]


class GraderResponse(BaseModel):
    task_id: TaskId
    score: float


class BaselineRequest(BaseModel):
    model: str = Field(default="Qwen/Qwen3.5-9B")
    max_tokens: int = Field(default=32768, ge=1)
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.5
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = False
    num_seeds: int = Field(default=5, ge=1)
    seed_start: int = Field(default=0, ge=0)


class BaselineScores(BaseModel):
    model: str
    scores: Dict[TaskId, float]
    episode_lengths: Dict[TaskId, int]
