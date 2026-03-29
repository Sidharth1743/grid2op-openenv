from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from openai import OpenAI

from grid2op_env import BaselineRequest, BaselineScores, GridAction, GridEnv
from grid2op_env.models import GridObservation, TaskId
from grid2op_env.server.tasks import TASKS

KNOWN_LINE_IDS = set(range(20))
KNOWN_GEN_IDS = set(range(6))


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_env() -> None:
    env_dir = Path(__file__).resolve().parent
    candidate_paths = [
        Path.cwd() / ".env",
        env_dir / ".env",
        env_dir.parent / ".env",
    ]
    for path in candidate_paths:
        if path.exists():
            load_dotenv(path, override=False)


_load_env()
configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    top_k: int
    enable_thinking: bool


def run_baseline_suite(
    base_url: str,
    config: BaselineRequest | None = None,
) -> BaselineScores:
    request_config = config or BaselineRequest(
        model=os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.5-9B")
    )
    llm_config = BaselineConfig(
        model=request_config.model,
        max_tokens=request_config.max_tokens,
        temperature=request_config.temperature,
        top_p=request_config.top_p,
        presence_penalty=request_config.presence_penalty,
        top_k=request_config.top_k,
        enable_thinking=request_config.enable_thinking,
    )

    client = OpenAI()
    scores: Dict[TaskId, float] = {}
    episode_lengths: Dict[TaskId, int] = {}
    logger.info("Starting baseline suite base_url=%s model=%s", base_url, llm_config.model)

    with GridEnv(base_url=base_url).sync() as env:
        for task_id, task in TASKS.items():
            logger.info(
                "Baseline episode start task_id=%s max_steps=%s",
                task_id,
                task.max_steps,
            )
            result = env.reset(task_id=task_id)
            step_idx = 0
            while not result.done and step_idx < task.max_steps:
                action = choose_action_with_qwen(
                    client=client,
                    task_id=task_id,
                    observation=result.observation,
                    step_count=step_idx,
                    max_steps=task.max_steps,
                    include_task_description=(step_idx == 0),
                    llm_config=llm_config,
                )
                logger.info(
                    "Baseline action task_id=%s step=%s action=%s",
                    task_id,
                    step_idx + 1,
                    action.model_dump(),
                )
                result = env.step(action)
                step_idx += 1

            state = env.state()
            logger.info(
                "Baseline episode finished task_id=%s step_count=%s done=%s",
                task_id,
                state.step_count,
                state.done,
            )
            response = requests.post(
                f"{base_url}/grader",
                json={
                    "task_id": task_id,
                    "episode_log": [entry.model_dump() for entry in state.episode_log],
                },
                timeout=60,
            )
            response.raise_for_status()
            score_payload = response.json()
            scores[task_id] = float(score_payload["score"])
            episode_lengths[task_id] = int(state.step_count)
            logger.info(
                "Baseline score task_id=%s score=%.6f episode_length=%s",
                task_id,
                scores[task_id],
                episode_lengths[task_id],
            )

    logger.info("Baseline suite complete scores=%s", scores)
    return BaselineScores(
        model=llm_config.model,
        scores=scores,
        episode_lengths=episode_lengths,
    )


def choose_action_with_qwen(
    client: OpenAI,
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    include_task_description: bool,
    llm_config: BaselineConfig,
) -> GridAction:
    prompt = build_prompt(
        task_id=task_id,
        observation=observation,
        step_count=step_count,
        max_steps=max_steps,
        include_task_description=include_task_description,
    )

    response = client.chat.completions.create(
        model=llm_config.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        presence_penalty=llm_config.presence_penalty,
        extra_body={
            "top_k": llm_config.top_k,
            "chat_template_kwargs": {"enable_thinking": llm_config.enable_thinking},
        },
    )

    content = response.choices[0].message.content or ""
    logger.info("Received model response task_id=%s chars=%s", task_id, len(content))
    parsed = parse_json_action(content)
    return validate_baseline_action(parsed)


def build_prompt(
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    include_task_description: bool,
) -> str:
    overloaded = [
        f"line {idx}: rho={rho:.3f}"
        for idx, rho in enumerate(observation.rho)
        if rho >= 0.8
    ]
    disconnected = [
        str(idx) for idx, status in enumerate(observation.line_status) if not status
    ]
    overflow_warnings = [
        f"line {idx} overloaded for {count} steps"
        for idx, count in enumerate(observation.timestep_overflow)
        if count > 2
    ]

    lines = [
        "You are a power grid operator. Respond with JSON only.",
        'Use schema: {"do_nothing": bool, "line_set": {"line_id": -1 or 1}, "redispatch": {"gen_id": float}}',
        f"task_id={task_id}",
        f"step={step_count + 1}/{max_steps}",
        f"max_rho={max(observation.rho):.3f}" if observation.rho else "max_rho=0.0",
        "lines_above_80=" + (", ".join(overloaded) if overloaded else "none"),
        "disconnected_lines=" + (", ".join(disconnected) if disconnected else "none"),
    ]
    if overflow_warnings:
        lines.append("imminent_trip=" + ", ".join(overflow_warnings))
    if include_task_description:
        lines.append("task_description=" + TASKS[task_id].description)
    lines.append("If unsure, use do_nothing=true.")
    return "\n".join(lines)


def parse_json_action(content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"do_nothing": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"do_nothing": True}


def validate_baseline_action(payload: Dict[str, Any]) -> GridAction:
    if payload.get("do_nothing"):
        return GridAction(do_nothing=True)

    valid_line_set = {}
    for key, value in payload.get("line_set", {}).items():
        try:
            line_id = int(key)
            status = int(value)
        except (TypeError, ValueError):
            continue
        if line_id in KNOWN_LINE_IDS and status in (-1, 1):
            valid_line_set[line_id] = status

    valid_redispatch = {}
    for key, value in payload.get("redispatch", {}).items():
        try:
            gen_id = int(key)
            delta = float(value)
        except (TypeError, ValueError):
            continue
        if gen_id in KNOWN_GEN_IDS:
            valid_redispatch[gen_id] = delta

    if not valid_line_set and not valid_redispatch:
        return GridAction(do_nothing=True)
    return GridAction(line_set=valid_line_set, redispatch=valid_redispatch)


if __name__ == "__main__":
    base_url = os.environ.get("GRID2OP_BASE_URL", "http://127.0.0.1:7860")
    result = run_baseline_suite(base_url=base_url)
    print(result.model_dump_json(indent=2))

