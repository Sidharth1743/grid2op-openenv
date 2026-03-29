from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
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
    min_p: float
    repetition_penalty: float
    enable_thinking: bool
    num_seeds: int
    seed_start: int


def run_baseline_suite(
    base_url: str,
    config: BaselineRequest | None = None,
) -> BaselineScores:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = prepare_run_paths(timestamp)
    attach_file_logger(run_paths["log"])

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
        min_p=request_config.min_p,
        repetition_penalty=request_config.repetition_penalty,
        enable_thinking=request_config.enable_thinking,
        num_seeds=request_config.num_seeds,
        seed_start=request_config.seed_start,
    )

    client = OpenAI()
    scores: Dict[TaskId, float] = {}
    episode_lengths: Dict[TaskId, int] = {}
    evaluation_records: list[dict[str, Any]] = []
    logger.info(
        "Starting baseline suite base_url=%s model=%s num_seeds=%s seed_start=%s",
        base_url,
        llm_config.model,
        llm_config.num_seeds,
        llm_config.seed_start,
    )

    with GridEnv(base_url=base_url).sync() as env:
        task_metrics: Dict[TaskId, list[dict[str, Any]]] = {task_id: [] for task_id in TASKS}
        for task_id, task in TASKS.items():
            for seed in range(llm_config.seed_start, llm_config.seed_start + llm_config.num_seeds):
                logger.info(
                    "Baseline episode start task_id=%s seed=%s max_steps=%s",
                    task_id,
                    seed,
                    task.max_steps,
                )
                result = env.reset(task_id=task_id, seed=seed)
                state = env.state()
                logger.info(
                    "Initial state task_id=%s seed=%s episode_id=%s scenario_metadata=%s max_rho=%.4f disconnected=%s",
                    task_id,
                    seed,
                    state.episode_id,
                    state.scenario_metadata,
                    float(max(result.observation.rho)) if result.observation.rho else 0.0,
                    [idx for idx, status in enumerate(result.observation.line_status) if not status],
                )

                step_idx = 0
                do_nothing_steps = 0
                raw_outputs: list[dict[str, Any]] = []
                while not result.done and step_idx < task.max_steps:
                    action, raw_output, trace = choose_action_with_qwen(
                        client=client,
                        task_id=task_id,
                        observation=result.observation,
                        step_count=step_idx,
                        max_steps=task.max_steps,
                        include_task_description=(step_idx == 0),
                        llm_config=llm_config,
                    )
                    raw_outputs.append(
                        {
                            "step": step_idx + 1,
                            "raw_model_output": raw_output,
                            "action_trace": trace,
                            "action": action.model_dump(),
                            "observation_summary": {
                                "max_rho": max(result.observation.rho) if result.observation.rho else 0.0,
                                "disconnected_lines": [
                                    idx
                                    for idx, status in enumerate(result.observation.line_status)
                                    if not status
                                ],
                                "timestep_overflow": result.observation.timestep_overflow,
                            },
                        }
                    )
                    if action.do_nothing:
                        do_nothing_steps += 1
                    logger.info(
                        "Baseline action task_id=%s seed=%s step=%s action=%s trace=%s",
                        task_id,
                        seed,
                        step_idx + 1,
                        action.model_dump(),
                        trace,
                    )
                    result = env.step(action)
                    step_idx += 1

                state = env.state()
                logger.info(
                    "Baseline episode finished task_id=%s seed=%s step_count=%s done=%s last_reward=%.4f",
                    task_id,
                    seed,
                    state.step_count,
                    state.done,
                    state.last_reward,
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
                episode_score = float(score_payload["score"])
                episode_length = int(state.step_count)
                record = {
                    "task_id": task_id,
                    "seed": seed,
                    "score": episode_score,
                    "episode_length": episode_length,
                    "done": state.done,
                    "do_nothing_steps": do_nothing_steps,
                    "non_do_nothing_steps": max(0, episode_length - do_nothing_steps),
                    "episode_log": [entry.model_dump() for entry in state.episode_log],
                    "raw_outputs": raw_outputs,
                    "scenario_metadata": state.scenario_metadata,
                }
                evaluation_records.append(record)
                task_metrics[task_id].append(record)
                logger.info(
                    "Baseline score task_id=%s seed=%s score=%.6f episode_length=%s do_nothing_steps=%s",
                    task_id,
                    seed,
                    episode_score,
                    episode_length,
                    do_nothing_steps,
                )

        for task_id, records in task_metrics.items():
            task_scores = [float(record["score"]) for record in records]
            task_lengths = [int(record["episode_length"]) for record in records]
            scores[task_id] = round(mean(task_scores), 6) if task_scores else 0.0
            episode_lengths[task_id] = round(mean(task_lengths)) if task_lengths else 0

    logger.info("Baseline suite complete scores=%s", scores)
    baseline_scores = BaselineScores(
        model=llm_config.model,
        scores=scores,
        episode_lengths=episode_lengths,
    )
    write_evaluation_outputs(
        timestamp=timestamp,
        run_paths=run_paths,
        model=llm_config.model,
        base_url=base_url,
        llm_config=llm_config,
        baseline_scores=baseline_scores,
        evaluation_records=evaluation_records,
    )
    append_evaluation_markdown(
        timestamp=timestamp,
        model=llm_config.model,
        llm_config=llm_config,
        baseline_scores=baseline_scores,
        evaluation_records=evaluation_records,
        run_paths=run_paths,
    )
    return baseline_scores


def choose_action_with_qwen(
    client: OpenAI,
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    include_task_description: bool,
    llm_config: BaselineConfig,
) -> tuple[GridAction, str, dict[str, Any]]:
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
            "min_p": llm_config.min_p,
            "repetition_penalty": llm_config.repetition_penalty,
            "chat_template_kwargs": {"enable_thinking": llm_config.enable_thinking},
        },
    )

    content = response.choices[0].message.content or ""
    logger.info("Received model response task_id=%s chars=%s content=%r", task_id, len(content), content)
    parsed = parse_json_action(content)
    action, trace = validate_baseline_action(parsed)
    return action, content, trace


def build_prompt(
    task_id: TaskId,
    observation: GridObservation,
    step_count: int,
    max_steps: int,
    include_task_description: bool,
) -> str:
    candidate_actions = build_candidate_actions(observation)
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
        "You are a power grid operator. Respond with a single JSON object only.",
        "Choose exactly one control action from the provided candidates.",
        'Use this exact schema: {"action_type":"disconnect_line|reconnect_line|redispatch|do_nothing","line_id":null|int,"gen_id":null|int,"delta_mw":null|float,"reason":"short string"}',
        "Rules: no markdown, no prose, no code fences, no extra keys.",
        "RULE 1: If a line is currently disconnected, your highest priority is reconnect_line to restore grid capacity. Do not disconnect healthy lines while the grid is in a degraded state.",
        "RULE 2: If lines are highly loaded (0.8 to 0.98) but overflow is 0, the grid is stable. The safest action is do_nothing. Prematurely disconnecting lines will cause a fatal cascade.",
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
    lines.append("candidate_actions=" + json.dumps(candidate_actions, separators=(",", ":")))
    lines.append(
        'example_1={"action_type":"disconnect_line","line_id":9,"gen_id":null,"delta_mw":null,"reason":"line 9 is the most overloaded"}'
    )
    lines.append(
        'example_2={"action_type":"reconnect_line","line_id":0,"gen_id":null,"delta_mw":null,"reason":"restore capacity through a disconnected line"}'
    )
    lines.append(
        'example_3={"action_type":"redispatch","line_id":null,"gen_id":2,"delta_mw":10.0,"reason":"increase generator 2 to relieve overloaded corridors"}'
    )
    lines.append(
        'If none of the candidates is safe or useful, return {"action_type":"do_nothing","line_id":null,"gen_id":null,"delta_mw":null,"reason":"no safe candidate"}'
    )
    return "\n".join(lines)


def build_candidate_actions(observation: GridObservation) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    overloaded_lines = sorted(
        ((idx, rho) for idx, rho in enumerate(observation.rho) if rho >= 0.8),
        key=lambda item: item[1],
        reverse=True,
    )
    disconnected_lines = [idx for idx, status in enumerate(observation.line_status) if not status]

    for idx in disconnected_lines[:2]:
        candidates.append(
            {
                "action_type": "reconnect_line",
                "line_id": idx,
                "reason_hint": "highest priority: reconnect a previously disconnected line to restore capacity",
            }
        )

    if overloaded_lines and not disconnected_lines:
        top_gen_candidates = [0, 1, 2]
        for gen_id in top_gen_candidates:
            candidates.append(
                {
                    "action_type": "redispatch",
                    "gen_id": gen_id,
                    "delta_mw": 10.0,
                    "reason_hint": "increase generation to relieve stressed lines",
                }
            )
            candidates.append(
                {
                    "action_type": "redispatch",
                    "gen_id": gen_id,
                    "delta_mw": -10.0,
                    "reason_hint": "decrease generation if it appears to worsen congestion",
                }
            )

    for idx, rho in overloaded_lines[:3]:
        overflow = observation.timestep_overflow[idx] if idx < len(observation.timestep_overflow) else 0
        if rho > 0.99 and overflow > 0 and not disconnected_lines:
            candidates.append(
                {
                    "action_type": "disconnect_line",
                    "line_id": idx,
                    "reason_hint": f"emergency-only disconnect because rho={rho:.3f} and overflow={overflow}",
                }
            )

    candidates.append(
        {
            "action_type": "do_nothing",
            "reason_hint": "preferred when the grid is hot but stable or when reconnect is already done and no overflow emergency exists",
        }
    )
    return candidates


def parse_json_action(content: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {"do_nothing": True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"do_nothing": True}


def validate_baseline_action(payload: Dict[str, Any]) -> tuple[GridAction, dict[str, Any]]:
    action_type = payload.get("action_type")
    if payload.get("do_nothing") or action_type == "do_nothing":
        return GridAction(do_nothing=True), {"decision": "do_nothing", "reason": payload.get("reason", "explicit_do_nothing")}

    line_set_payload = payload.get("line_set") or {}
    if not isinstance(line_set_payload, dict):
        line_set_payload = {}

    valid_line_set = {}
    for key, value in line_set_payload.items():
        try:
            line_id = int(key)
            status = int(value)
        except (TypeError, ValueError):
            continue
        if line_id in KNOWN_LINE_IDS and status in (-1, 1):
            valid_line_set[line_id] = status

    redispatch_payload = payload.get("redispatch") or {}
    if not isinstance(redispatch_payload, dict):
        redispatch_payload = {}

    if action_type == "disconnect_line":
        line_id = payload.get("line_id")
        try:
            line_id = int(line_id)
        except (TypeError, ValueError):
            line_id = None
        if line_id in KNOWN_LINE_IDS:
            return (
                GridAction(line_set={line_id: -1}, redispatch={}),
                {"decision": "disconnect_line", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_disconnect_line"}

    if action_type == "reconnect_line":
        line_id = payload.get("line_id")
        try:
            line_id = int(line_id)
        except (TypeError, ValueError):
            line_id = None
        if line_id in KNOWN_LINE_IDS:
            return (
                GridAction(line_set={line_id: 1}, redispatch={}),
                {"decision": "reconnect_line", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_reconnect_line"}

    if action_type == "redispatch":
        gen_id = payload.get("gen_id")
        delta = payload.get("delta_mw")
        try:
            gen_id = int(gen_id)
            delta = float(delta)
        except (TypeError, ValueError):
            gen_id = None
            delta = None
        if gen_id in KNOWN_GEN_IDS and delta is not None:
            return (
                GridAction(redispatch={gen_id: delta}, line_set={}),
                {"decision": "redispatch", "reason": payload.get("reason", "")},
            )
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "invalid_redispatch"}

    valid_redispatch = {}
    for key, value in redispatch_payload.items():
        try:
            gen_id = int(key)
            delta = float(value)
        except (TypeError, ValueError):
            continue
        if gen_id in KNOWN_GEN_IDS:
            valid_redispatch[gen_id] = delta

    if not valid_line_set and not valid_redispatch:
        return GridAction(do_nothing=True), {"decision": "fallback_do_nothing", "reason": "empty_or_invalid_payload"}
    return (
        GridAction(line_set=valid_line_set, redispatch=valid_redispatch),
        {"decision": "legacy_payload", "reason": payload.get("reason", "")},
    )


def write_evaluation_outputs(
    timestamp: str,
    run_paths: dict[str, Path],
    model: str,
    base_url: str,
    llm_config: BaselineConfig,
    baseline_scores: BaselineScores,
    evaluation_records: list[dict[str, Any]],
) -> None:
    json_path = run_paths["json"]
    csv_path = run_paths["csv"]

    aggregate: dict[str, Any] = {}
    for task_id in TASKS:
        records = [record for record in evaluation_records if record["task_id"] == task_id]
        scores = [float(record["score"]) for record in records]
        lengths = [int(record["episode_length"]) for record in records]
        do_nothing = [int(record["do_nothing_steps"]) for record in records]
        aggregate[task_id] = {
            "num_episodes": len(records),
            "score_mean": round(mean(scores), 6) if scores else 0.0,
            "score_std": round(pstdev(scores), 6) if len(scores) > 1 else 0.0,
            "episode_length_mean": round(mean(lengths), 6) if lengths else 0.0,
            "episode_length_std": round(pstdev(lengths), 6) if len(lengths) > 1 else 0.0,
            "do_nothing_steps_mean": round(mean(do_nothing), 6) if do_nothing else 0.0,
        }

    payload = {
        "timestamp": timestamp,
        "model": model,
        "base_url": base_url,
        "sampling": {
            "temperature": llm_config.temperature,
            "top_p": llm_config.top_p,
            "top_k": llm_config.top_k,
            "min_p": llm_config.min_p,
            "presence_penalty": llm_config.presence_penalty,
            "repetition_penalty": llm_config.repetition_penalty,
            "enable_thinking": llm_config.enable_thinking,
            "max_tokens": llm_config.max_tokens,
            "num_seeds": llm_config.num_seeds,
            "seed_start": llm_config.seed_start,
        },
        "summary": baseline_scores.model_dump(),
        "aggregate": aggregate,
        "episodes": evaluation_records,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "seed",
                "score",
                "episode_length",
                "done",
                "do_nothing_steps",
                "non_do_nothing_steps",
            ],
        )
        writer.writeheader()
        for record in evaluation_records:
            writer.writerow(
                {
                    "task_id": record["task_id"],
                    "seed": record["seed"],
                    "score": record["score"],
                    "episode_length": record["episode_length"],
                    "done": record["done"],
                    "do_nothing_steps": record["do_nothing_steps"],
                    "non_do_nothing_steps": record["non_do_nothing_steps"],
                }
            )
    logger.info("Wrote evaluation outputs json=%s csv=%s", json_path, csv_path)


def prepare_run_paths(timestamp: str) -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parent
    eval_dir = base_dir / "outputs" / "evals"
    log_dir = base_dir / "outputs" / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "json": eval_dir / f"baseline_eval_{timestamp}.json",
        "csv": eval_dir / f"baseline_eval_{timestamp}.csv",
        "log": log_dir / f"baseline_run_{timestamp}.log",
    }


def attach_file_logger(log_path: Path) -> None:
    root_logger = logging.getLogger()
    if any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path
        for handler in root_logger.handlers
    ):
        return
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    logger.info("Attached file logger log_path=%s", log_path)


def append_evaluation_markdown(
    timestamp: str,
    model: str,
    llm_config: BaselineConfig,
    baseline_scores: BaselineScores,
    evaluation_records: list[dict[str, Any]],
    run_paths: dict[str, Path],
) -> None:
    eval_md = Path(__file__).resolve().parent.parent / "evaluation.md"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in evaluation_records:
        grouped.setdefault(record["task_id"], []).append(record)

    lines = [
        f"## Run {timestamp}",
        "",
        f"- Model: `{model}`",
        f"- Seeds: `{llm_config.seed_start}` to `{llm_config.seed_start + llm_config.num_seeds - 1}`",
        f"- Sampling: `temperature={llm_config.temperature}`, `top_p={llm_config.top_p}`, `top_k={llm_config.top_k}`, `min_p={llm_config.min_p}`, `presence_penalty={llm_config.presence_penalty}`, `repetition_penalty={llm_config.repetition_penalty}`",
        f"- JSON output: [{run_paths['json']}]({run_paths['json']})",
        f"- CSV output: [{run_paths['csv']}]({run_paths['csv']})",
        f"- Log file: [{run_paths['log']}]({run_paths['log']})",
        "",
        "| Task | Mean Score | Mean Episode Length | Mean Do-Nothing Steps |",
        "| --- | ---: | ---: | ---: |",
    ]

    for task_id, records in grouped.items():
        scores = [float(record["score"]) for record in records]
        lengths = [int(record["episode_length"]) for record in records]
        do_nothing = [int(record["do_nothing_steps"]) for record in records]
        lines.append(
            f"| `{task_id}` | `{mean(scores):.6f}` | `{mean(lengths):.2f}` | `{mean(do_nothing):.2f}` |"
        )

    lines.extend(
        [
            "",
            "Summary scores:",
            f"```json\n{baseline_scores.model_dump_json(indent=2)}\n```",
            "",
        ]
    )

    with eval_md.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


if __name__ == "__main__":
    base_url = os.environ.get("GRID2OP_BASE_URL", "http://127.0.0.1:7860")
    result = run_baseline_suite(base_url=base_url)
    print(result.model_dump_json(indent=2))
