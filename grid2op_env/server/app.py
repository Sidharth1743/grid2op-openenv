from __future__ import annotations

import logging
from fastapi import Request
from openenv.core.env_server.http_server import create_app

try:
    from ..models import (
        BaselineRequest,
        BaselineScores,
        GraderRequest,
        GraderResponse,
        GridAction,
        GridObservation,
        TaskListResponse,
    )
    from .graders import grade_episode
    from .grid_environment import GridEnvironment
    from .logging_utils import configure_logging
    from .tasks import task_list
except ImportError:
    from models import (
        BaselineRequest,
        BaselineScores,
        GraderRequest,
        GraderResponse,
        GridAction,
        GridObservation,
        TaskListResponse,
    )
    from server.graders import grade_episode
    from server.grid_environment import GridEnvironment
    from server.logging_utils import configure_logging
    from server.tasks import task_list

configure_logging()
logger = logging.getLogger(__name__)

app = create_app(
    GridEnvironment,
    GridAction,
    GridObservation,
    env_name="grid2op_env",
    max_concurrent_envs=2,
)


@app.get("/tasks", response_model=TaskListResponse)
def get_tasks() -> TaskListResponse:
    logger.info("Serving /tasks")
    return TaskListResponse(
        tasks=task_list(),
        action_schema=GridAction.model_json_schema(),
    )


@app.post("/grader", response_model=GraderResponse)
def post_grader(payload: GraderRequest) -> GraderResponse:
    logger.info(
        "Serving /grader task_id=%s steps=%s",
        payload.task_id,
        len(payload.episode_log),
    )
    return GraderResponse(
        task_id=payload.task_id,
        score=grade_episode(payload.task_id, payload.episode_log),
    )


@app.post("/baseline", response_model=BaselineScores)
def run_baseline_route(payload: BaselineRequest, request: Request) -> BaselineScores:
    from ..inference import run_baseline_suite

    base_url = str(request.base_url).rstrip("/")
    logger.info("Serving /baseline model=%s base_url=%s", payload.model, base_url)
    return run_baseline_suite(base_url=base_url, config=payload)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()
    logger.info("Starting Grid2Op FastAPI server host=%s port=%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
