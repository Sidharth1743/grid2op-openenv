from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from grid2op_env import GridAction, GridEnv
from grid2op_env.inference import (
    SimulationOutcome,
    log_end,
    log_start,
    log_step,
    serialize_simulation_outcome,
)
from grid2op_env.models import GridObservation, RedispatchGeneratorContext, TaskId
from grid2op_env.server.tasks import TASKS, benchmark_tiers_for_task
from scripts.collect_teacher_dataset import build_sft_prompt
from scripts.train_sft import QWEN_CHATML_TRAINING_TEMPLATE, resolve_precision


DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER = "outputs/models/grid2op-qwen3-4b-sft-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8018"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def action_key(action: GridAction) -> str:
    return _json_dumps(action.model_dump())


def action_type(action: GridAction) -> str:
    if action.do_nothing:
        return "do_nothing"
    if action.redispatch:
        return "redispatch"
    if action.line_set:
        statuses = [int(value) for value in action.line_set.values()]
        if statuses and statuses[0] == 1:
            return "reconnect_line"
        if statuses and statuses[0] == -1:
            return "disconnect_line"
    return "unknown"


def max_overflow(outcome: SimulationOutcome) -> int:
    overflow = outcome.raw_result.get("timestep_overflow", [])
    if not isinstance(overflow, list):
        return 0
    return max((int(value) for value in overflow), default=0)


def generate_legal_candidates(
    task_id: TaskId,
    observation: GridObservation,
    graph_intelligence: dict[str, Any],
    redispatch_generators: Sequence[RedispatchGeneratorContext],
) -> list[GridAction]:
    candidates: list[GridAction] = [GridAction(do_nothing=True)]

    for context in redispatch_generators:
        for delta in context.allowed_deltas:
            delta_value = float(delta)
            if abs(delta_value) < 1e-9:
                continue
            if not (
                float(context.allowed_delta_min)
                <= delta_value
                <= float(context.allowed_delta_max)
            ):
                continue
            candidates.append(GridAction(redispatch={int(context.gen_id): delta_value}))

    disconnected_lines = [
        line_id for line_id, status in enumerate(observation.line_status) if not status
    ]
    if task_id in {"n_minus_1", "cascade_prevent", "multi_stage_cascade"}:
        for line_id in disconnected_lines:
            candidates.append(GridAction(line_set={line_id: 1}))

    if task_id in {"cascade_prevent", "multi_stage_cascade"}:
        safe_disconnect = {
            int(line_id) for line_id in graph_intelligence.get("safe_to_disconnect", [])
        }
        blocked_disconnect = {
            int(line_id) for line_id in graph_intelligence.get("bridge_lines", [])
        } | {
            int(line_id)
            for line_id in graph_intelligence.get("n_minus_1_critical_lines", [])
        }
        for line_id in sorted(safe_disconnect - blocked_disconnect):
            if 0 <= line_id < len(observation.line_status) and observation.line_status[line_id]:
                candidates.append(GridAction(line_set={line_id: -1}))

    deduped: list[GridAction] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = action_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def rank_simulations(
    task_id: TaskId,
    observation: GridObservation,
    simulations: Sequence[SimulationOutcome],
) -> list[SimulationOutcome]:
    viable = [outcome for outcome in simulations if not outcome.convergence_failed]
    if not viable:
        return []

    no_overload = [outcome for outcome in viable if not outcome.overloaded_line_ids]
    candidates = no_overload or viable
    current_max_rho = max((float(value) for value in observation.rho), default=0.0)

    if task_id == "single_fault":
        active_threshold = 0.80
        return sorted(
            candidates,
            key=lambda outcome: (
                float(outcome.max_rho) >= active_threshold,
                action_type(outcome.action) == "do_nothing"
                and current_max_rho > active_threshold,
                float(outcome.max_rho),
                -float(outcome.simulated_reward),
                len(outcome.disconnected_lines),
                outcome.candidate_index,
            ),
        )

    if task_id == "n_minus_1":
        active_threshold = 0.92 if len(candidates) and current_max_rho >= 0.92 else 0.90
        return sorted(
            candidates,
            key=lambda outcome: (
                float(outcome.max_rho) >= active_threshold,
                action_type(outcome.action) == "do_nothing"
                and current_max_rho >= active_threshold,
                float(outcome.max_rho),
                -float(outcome.simulated_reward),
                len(outcome.disconnected_lines),
                outcome.candidate_index,
            ),
        )

    active_cascade = (
        max((int(value) for value in observation.timestep_overflow), default=0) > 0
        or current_max_rho > 1.0
    )
    if active_cascade:
        return sorted(
            candidates,
            key=lambda outcome: (
                max_overflow(outcome),
                len(outcome.overloaded_line_ids),
                float(outcome.max_rho),
                -float(outcome.simulated_reward),
                len(outcome.disconnected_lines),
                action_type(outcome.action) == "do_nothing",
                outcome.candidate_index,
            ),
        )

    return sorted(
        candidates,
        key=lambda outcome: (
            -float(outcome.simulated_reward),
            float(outcome.max_rho),
            len(outcome.disconnected_lines),
            action_type(outcome.action) == "do_nothing",
            outcome.candidate_index,
        ),
    )


def select_prompt_candidates(
    task_id: TaskId,
    observation: GridObservation,
    simulations: Sequence[SimulationOutcome],
    candidate_count: int,
) -> list[SimulationOutcome]:
    ranked = rank_simulations(task_id, observation, simulations)
    selected = ranked[:candidate_count]
    if task_id == "single_fault":
        current_max_rho = max((float(value) for value in observation.rho), default=0.0)
        best_do_nothing = next(
            (outcome for outcome in ranked if outcome.action.do_nothing),
            None,
        )
        do_nothing_next_rho = (
            float(best_do_nothing.max_rho)
            if best_do_nothing is not None
            else current_max_rho
        )
        improving_redispatch = [
            outcome
            for outcome in ranked
            if outcome.action.redispatch
            and float(outcome.max_rho) < do_nothing_next_rho - 0.0001
        ]
        if current_max_rho > 0.80 and improving_redispatch:
            return improving_redispatch[:candidate_count]
        if best_do_nothing is not None and all(
            not outcome.action.do_nothing for outcome in selected
        ):
            if len(selected) >= candidate_count:
                selected = selected[:-1]
            selected.append(best_do_nothing)
    elif task_id == "n_minus_1":
        current_max_rho = max((float(value) for value in observation.rho), default=0.0)
        active_threshold = 0.92 if current_max_rho >= 0.92 else 0.90
        if current_max_rho >= active_threshold:
            active_improvers = [
                outcome
                for outcome in ranked
                if not outcome.action.do_nothing
                and float(outcome.max_rho) < current_max_rho - 0.0001
            ]
            best_reconnect = next(
                (
                    outcome
                    for outcome in active_improvers
                    if action_type(outcome.action) == "reconnect_line"
                ),
                None,
            )
            best_redispatch = next(
                (
                    outcome
                    for outcome in active_improvers
                    if action_type(outcome.action) == "redispatch"
                ),
                None,
            )
            forced = [
                outcome
                for outcome in (best_reconnect, best_redispatch)
                if outcome is not None
            ]
            for forced_outcome in forced:
                if all(outcome.candidate_index != forced_outcome.candidate_index for outcome in selected):
                    if len(selected) >= candidate_count:
                        selected = selected[:-1]
                    selected.append(forced_outcome)
    return selected


def parse_grid_action(raw_output: str, simulations: Sequence[SimulationOutcome]) -> GridAction:
    payload = parse_first_json_object(raw_output)
    allowed_keys = {"line_set", "redispatch", "do_nothing", "metadata"}
    extra_keys = set(payload) - allowed_keys
    if extra_keys:
        raise ValueError(f"Model output has extra keys: {sorted(extra_keys)}")

    action = GridAction.model_validate(payload)
    simulated_by_key = {action_key(outcome.action): outcome.action for outcome in simulations}
    key = action_key(action)
    if key not in simulated_by_key:
        raise ValueError(
            "Model selected an action outside verified_simulation_results: "
            f"{action.model_dump()}"
        )
    return simulated_by_key[key]


def parse_first_json_object(raw_output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    cleaned = raw_output.strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    payload, _end = decoder.raw_decode(cleaned[start:])
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object")
    return payload


class FineTunedPolicy:
    def __init__(
        self,
        base_model: str,
        adapter: Path | None,
        precision: str,
        use_4bit: bool,
        attn_implementation: str | None,
        max_new_tokens: int,
    ) -> None:
        dtype, _, _, _ = resolve_precision(precision)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        if "Qwen" in base_model:
            self.tokenizer.chat_template = QWEN_CHATML_TRAINING_TEMPLATE

        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": dtype,
            "device_map": "auto",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if adapter is None:
            self.model = base
        else:
            self.model = PeftModel.from_pretrained(base, str(adapter))
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def choose(self, prompt: str) -> tuple[GridAction, str]:
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        first_param = next(self.model.parameters())
        encoded = {key: value.to(first_param.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[-1] :]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GridAction(), raw_output


def choose_ft_action(
    policy: FineTunedPolicy,
    env: GridEnv,
    episode_id: str,
    task_id: TaskId,
    observation: GridObservation,
    step: int,
    max_steps: int,
    candidate_count: int,
) -> tuple[GridAction, dict[str, Any]]:
    planning_context = env.planning_context(episode_id)
    candidate_actions = generate_legal_candidates(
        task_id=task_id,
        observation=observation,
        graph_intelligence=planning_context.graph_intelligence,
        redispatch_generators=planning_context.redispatch_generators,
    )
    simulation_response = env.simulate_candidates(episode_id, candidate_actions)
    simulations = [
        SimulationOutcome(
            candidate_index=index,
            action=result.action,
            trace={"source": "enumerated_action_space", "decision": action_type(result.action)},
            done=result.done,
            simulated_reward=result.simulated_reward,
            max_rho=result.max_rho,
            overloaded_line_ids=result.overloaded_line_ids,
            disconnected_lines=result.disconnected_lines,
            convergence_failed=result.convergence_failed,
            exceptions=result.exceptions,
            raw_result=result.raw_result,
        )
        for index, result in enumerate(simulation_response.results, start=1)
    ]
    ranked = select_prompt_candidates(
        task_id=task_id,
        observation=observation,
        simulations=simulations,
        candidate_count=candidate_count,
    )
    if not ranked:
        raise ValueError("No verified non-converged simulation candidates available")

    prompt = build_sft_prompt(
        task_id=task_id,
        step=step,
        max_steps=max_steps,
        observation=observation,
        graph_intelligence=planning_context.graph_intelligence,
        redispatch_generators=planning_context.redispatch_generators,
        simulations=ranked,
    )
    _placeholder, raw_output = policy.choose(prompt)
    trace = {
        "prompt": prompt,
        "raw_output": raw_output,
        "candidate_count": len(candidate_actions),
        "verified_candidates": [
            serialize_simulation_outcome(outcome) for outcome in ranked
        ],
    }
    action = parse_grid_action(raw_output, ranked)
    trace["selected_action"] = action.model_dump()
    return action, trace


def run_ft_episodes(args: argparse.Namespace) -> dict[str, Any]:
    policy = FineTunedPolicy(
        base_model=args.model,
        adapter=None if str(args.adapter).lower() == "none" else args.adapter,
        precision=args.precision,
        use_4bit=args.use_4bit,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    selected_task_ids: list[TaskId] = args.task_ids or list(TASKS)
    stats: dict[str, Any] = {"episodes": 0, "tasks": {}}
    started_at = perf_counter()

    with GridEnv(base_url=args.base_url).sync() as env:
        for task_id in selected_task_ids:
            task = TASKS[task_id]
            task_scores: list[float] = []
            stats["tasks"][task_id] = {"episodes": 0, "scores": []}
            benchmark_tiers = benchmark_tiers_for_task(task_id)
            for episode_index in range(args.episodes_per_task):
                seed = args.seed_start + episode_index
                benchmark_tier = benchmark_tiers[episode_index % len(benchmark_tiers)]
                rewards: list[float] = []
                steps_taken = 0
                score = 0.0
                success = False
                model_label = args.model if str(args.adapter).lower() == "none" else str(args.adapter)
                log_start(task=task_id, env=args.base_url, model=model_label)
                result = env.reset(
                    task_id=task_id,
                    seed=seed,
                    difficulty_level=episode_index + 1,
                    scenario_mode=args.scenario_mode,
                    benchmark_tier=benchmark_tier,
                )
                state = env.state()
                try:
                    for step_idx in range(task.max_steps):
                        if result.done:
                            break
                        action, trace = choose_ft_action(
                            policy=policy,
                            env=env,
                            episode_id=state.episode_id,
                            task_id=task_id,
                            observation=result.observation,
                            step=step_idx + 1,
                            max_steps=task.max_steps,
                            candidate_count=args.candidate_count,
                        )
                        if args.verbose_trace:
                            print(
                                "[FT_PLAN] "
                                + json.dumps(
                                    {
                                        "task_id": task_id,
                                        "step": step_idx + 1,
                                        **trace,
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                flush=True,
                            )
                        result = env.step(action)
                        reward = float(result.reward or 0.0)
                        rewards.append(reward)
                        steps_taken = step_idx + 1
                        log_step(
                            step=steps_taken,
                            action=action,
                            reward=reward,
                            done=bool(result.done),
                            error=None,
                        )

                    state = env.state()
                    response = requests.post(
                        f"{args.base_url.rstrip('/')}/grader",
                        json={
                            "task_id": task_id,
                            "episode_log": [
                                entry.model_dump() for entry in state.episode_log
                            ],
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    score = float(response.json()["score"])
                    score = max(0.0, min(1.0, score))
                    success = score >= args.success_threshold
                    task_scores.append(score)
                except Exception as exc:
                    raw_output = None
                    if "trace" in locals() and isinstance(trace, dict):
                        raw_output = trace.get("raw_output")
                    print(
                        "[FT_FAIL] "
                        + json.dumps(
                            {
                                "task_id": task_id,
                                "seed": seed,
                                "benchmark_tier": benchmark_tier,
                                "step": steps_taken + 1,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "raw_output": raw_output,
                                "traceback": traceback.format_exc(),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                finally:
                    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
                    stats["episodes"] += 1
                    stats["tasks"][task_id]["episodes"] += 1
                    stats["tasks"][task_id]["scores"].append(score)

            stats["tasks"][task_id]["mean_score"] = (
                sum(task_scores) / len(task_scores) if task_scores else 0.0
            )

    stats["wall_time_s"] = round(perf_counter() - started_at, 6)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grid2Op inference with a local SFT LoRA adapter.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=Path(DEFAULT_ADAPTER))
    parser.add_argument("--task-id", dest="task_ids", nargs="+", choices=list(TASKS), default=None)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--scenario-mode", choices=["benchmark", "curriculum"], default="benchmark")
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--success-threshold", type=float, default=0.1)
    parser.add_argument("--verbose-trace", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = run_ft_episodes(args)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
