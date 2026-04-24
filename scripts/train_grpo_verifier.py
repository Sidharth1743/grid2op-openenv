from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from grid2op_env import GridAction
try:
    from scripts.train_sft import (
        QWEN_CHATML_TRAINING_TEMPLATE,
        resolve_precision,
        selected_action_type,
    )
except ModuleNotFoundError:
    from train_sft import (  # type: ignore[no-redef]
        QWEN_CHATML_TRAINING_TEMPLATE,
        resolve_precision,
        selected_action_type,
    )


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET = "outputs/datasets/grid2op_teacher_wide_balanced_v2.jsonl"
DEFAULT_SFT_ADAPTER = "outputs/models/grid2op-qwen3-4b-sft-3k-v1"
DEFAULT_OUTPUT_DIR = "outputs/models/grid2op-qwen3-4b-grpo-v1"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_json_loads(text: str) -> Any:
    return json.loads(text)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model text."""
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = _safe_json_loads(text[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("JSON payload is not an object")
                return payload
    raise ValueError("Unterminated JSON object")


def _canonical_action_dict(action: dict[str, Any]) -> dict[str, Any]:
    line_set = action.get("line_set") or {}
    redispatch = action.get("redispatch") or {}
    if not isinstance(line_set, dict):
        line_set = {}
    if not isinstance(redispatch, dict):
        redispatch = {}
    clean_line_set: dict[str, int] = {}
    for key, value in line_set.items():
        if value is None:
            continue
        try:
            clean_line_set[str(int(key))] = int(value)
        except (TypeError, ValueError):
            continue
    clean_redispatch: dict[str, float] = {}
    for key, value in redispatch.items():
        if value is None:
            continue
        try:
            delta = float(value)
            if abs(delta) > 1e-9:
                clean_redispatch[str(int(key))] = round(delta, 6)
        except (TypeError, ValueError):
            continue
    return {
        "do_nothing": bool(action.get("do_nothing", False)),
        "line_set": clean_line_set,
        "redispatch": clean_redispatch,
    }


def _action_key(action: dict[str, Any]) -> str:
    return _json_dumps(_canonical_action_dict(action))


def _parse_grid_action_from_completion(completion: Any) -> tuple[GridAction | None, dict[str, Any] | None, str | None]:
    text = _completion_text(completion)
    try:
        payload = _extract_first_json_object(text)
        allowed_keys = {"line_set", "redispatch", "do_nothing", "metadata"}
        extra_keys = set(payload) - allowed_keys
        if extra_keys:
            return None, payload, f"extra_keys:{sorted(extra_keys)}"
        action = GridAction.model_validate(payload)
        return action, payload, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}:{exc}"


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _simulation_action_type(simulation: dict[str, Any]) -> str:
    return selected_action_type(simulation.get("action", {}))


def _safe_simulations(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        simulation
        for simulation in simulations
        if not bool(simulation.get("done"))
        and not bool(simulation.get("convergence_failed"))
        and not simulation.get("exceptions")
    ]


def _sim_by_action_key(simulations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for simulation in simulations:
        action = simulation.get("action")
        if isinstance(action, dict):
            mapping[_action_key(action)] = simulation
    return mapping


def _best_safe_reward(simulations: list[dict[str, Any]]) -> float | None:
    safe = _safe_simulations(simulations)
    if not safe:
        return None
    return max(float(simulation.get("simulated_reward", -1e9)) for simulation in safe)


def _best_safe_rho(simulations: list[dict[str, Any]]) -> float | None:
    safe = _safe_simulations(simulations)
    if not safe:
        return None
    return min(float(simulation.get("max_rho", 999.0)) for simulation in safe)


def _compact_candidate_summary(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, simulation in enumerate(simulations, start=1):
        raw_result = simulation.get("raw_result") or {}
        overflow = raw_result.get("timestep_overflow") or []
        max_overflow = (
            max((int(value) for value in overflow), default=0)
            if isinstance(overflow, list)
            else 0
        )
        rows.append(
            {
                "id": index,
                "action": _canonical_action_dict(simulation.get("action") or {}),
                "action_type": _simulation_action_type(simulation),
                "safe": not bool(simulation.get("done"))
                and not bool(simulation.get("convergence_failed"))
                and not bool(simulation.get("exceptions")),
                "max_rho": round(float(simulation.get("max_rho", 999.0)), 6),
                "reward": round(float(simulation.get("simulated_reward", 0.0)), 6),
                "overloaded": simulation.get("overloaded_line_ids") or [],
                "disconnected": simulation.get("disconnected_lines") or [],
                "max_overflow_countdown": max_overflow,
            }
        )
    return rows


def _task_objective_text(task_id: str, current_max_rho: float) -> str:
    if task_id == "single_fault":
        return (
            "core objective: bring max_rho below 0.80 when reachable; "
            "if no candidate reaches 0.80, choose the safe candidate with lowest max_rho."
        )
    if task_id == "n_minus_1":
        threshold = 0.92 if current_max_rho >= 0.92 else 0.90
        return (
            f"core objective: keep topology alive and prefer max_rho below {threshold:.2f}; "
            "safe reconnect is useful when verified."
        )
    if task_id == "cascade_prevent":
        return (
            "core objective: prevent overload countdown trips first, then maximize survival and safety."
        )
    if task_id == "multi_stage_cascade":
        return (
            "core objective: survive stage transitions, preserve load/islands, and avoid unsafe topology."
        )
    return "core objective: choose the safest verified candidate."


def build_compact_grpo_prompt(row: dict[str, Any]) -> str:
    task_id = str(row.get("task_id", "unknown"))
    obs_summary = row.get("observation_summary") or {}
    current_max_rho = float(obs_summary.get("max_rho", 0.0))
    candidates = _compact_candidate_summary(row["verified_simulation_results"])
    return "\n".join(
        [
            "You are a power-grid control policy.",
            "Choose exactly one action from verified_candidates.",
            "Return only the selected GridAction JSON with keys: line_set, redispatch, do_nothing.",
            "No markdown, prose, reasoning, candidate id, or extra keys.",
            f"task_id={task_id}",
            f"step={row.get('step_value', -1)}",
            f"current_max_rho={current_max_rho:.6f}",
            _task_objective_text(task_id, current_max_rho),
            "verified_candidates=" + _json_dumps(candidates),
        ]
    )


def _selected_simulation(
    completion: Any,
    verified_simulation_results: list[dict[str, Any]],
) -> tuple[GridAction | None, dict[str, Any] | None, str | None]:
    action, _payload, error = _parse_grid_action_from_completion(completion)
    if action is None:
        return None, None, error
    simulation = _sim_by_action_key(verified_simulation_results).get(_action_key(action.model_dump()))
    if simulation is None:
        return action, None, "not_verified_candidate"
    return action, simulation, None


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = _completion_text(completion).strip()
        try:
            _extract_first_json_object(text)
            rewards.append(0.2 if text.startswith("{") else 0.05)
        except Exception:
            rewards.append(-1.0)
    return rewards


def schema_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        action, _payload, error = _parse_grid_action_from_completion(completion)
        rewards.append(0.3 if action is not None and error is None else -1.0)
    return rewards


def verified_candidate_reward(
    completions: list[Any],
    verified_simulation_results: list[list[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, simulations in zip(completions, verified_simulation_results, strict=True):
        _action, simulation, error = _selected_simulation(completion, simulations)
        if simulation is not None:
            rewards.append(1.0)
        elif error == "not_verified_candidate":
            rewards.append(-1.0)
        else:
            rewards.append(-1.5)
    return rewards


def safety_reward(
    completions: list[Any],
    verified_simulation_results: list[list[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, simulations in zip(completions, verified_simulation_results, strict=True):
        _action, simulation, _error = _selected_simulation(completion, simulations)
        if simulation is None:
            rewards.append(-1.0)
            continue
        if bool(simulation.get("convergence_failed")) or bool(simulation.get("done")):
            rewards.append(-1.0)
            continue
        if simulation.get("exceptions"):
            rewards.append(-1.0)
            continue
        overloaded = simulation.get("overloaded_line_ids") or []
        if overloaded:
            rewards.append(-0.5)
            continue
        rewards.append(1.0)
    return rewards


def task_objective_reward(
    completions: list[Any],
    task_id: list[str],
    observation_summary: list[dict[str, Any]],
    verified_simulation_results: list[list[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, current_task, obs_summary, simulations in zip(
        completions,
        task_id,
        observation_summary,
        verified_simulation_results,
        strict=True,
    ):
        _action, simulation, _error = _selected_simulation(completion, simulations)
        if simulation is None:
            rewards.append(-1.0)
            continue
        safe = _safe_simulations(simulations)
        if not safe:
            rewards.append(-0.5)
            continue
        max_rho = float(simulation.get("max_rho", 999.0))
        best_rho = _best_safe_rho(simulations)
        best_reward = _best_safe_reward(simulations)
        current_max_rho = float(obs_summary.get("max_rho", 0.0))
        action_kind = _simulation_action_type(simulation)
        reward = 0.0

        if current_task == "single_fault":
            target_exists = any(float(candidate.get("max_rho", 999.0)) < 0.80 for candidate in safe)
            if target_exists:
                reward += 2.0 if max_rho < 0.80 else -0.5
            else:
                # Do not punish the model for unreachable targets; reward best thermal progress.
                reward += max(0.0, min(1.0, (current_max_rho - max_rho) / 0.03))
            if best_rho is not None and max_rho <= best_rho + 1e-4:
                reward += 1.0
            if current_max_rho > 0.80 and action_kind == "do_nothing" and best_rho is not None and best_rho < max_rho - 1e-4:
                reward -= 1.0
        elif current_task == "n_minus_1":
            threshold = 0.92 if current_max_rho >= 0.92 else 0.90
            reward += 1.5 if max_rho < threshold else -0.25
            if action_kind == "reconnect_line":
                reward += 0.4
            if current_max_rho >= threshold and action_kind == "do_nothing" and best_rho is not None and best_rho < max_rho - 1e-4:
                reward -= 1.0
            if best_rho is not None and max_rho <= best_rho + 1e-4:
                reward += 0.6
        elif current_task == "cascade_prevent":
            overflow = simulation.get("raw_result", {}).get("timestep_overflow", [])
            max_overflow = max((int(value) for value in overflow), default=0) if isinstance(overflow, list) else 0
            reward += 1.0 if max_overflow == 0 else -1.0
            reward += 0.5 if max_rho < 1.0 else -0.5
            if best_reward is not None and float(simulation.get("simulated_reward", -1e9)) >= best_reward - 1e-3:
                reward += 0.5
        elif current_task == "multi_stage_cascade":
            reward += 1.0 if not simulation.get("done") else -1.0
            reward += 0.5 if max_rho < 1.0 else -0.5
            if action_kind in {"reconnect_line", "disconnect_line", "redispatch"}:
                reward += 0.2
            if best_reward is not None and float(simulation.get("simulated_reward", -1e9)) >= best_reward - 1e-3:
                reward += 0.5
        else:
            if best_rho is not None and max_rho <= best_rho + 1e-4:
                reward += 1.0

        rewards.append(max(-2.0, min(3.0, reward)))
    return rewards


def anti_hacking_reward(
    completions: list[Any],
    verified_simulation_results: list[list[dict[str, Any]]],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, simulations in zip(completions, verified_simulation_results, strict=True):
        action, payload, error = _parse_grid_action_from_completion(completion)
        if action is None or payload is None:
            rewards.append(-1.0)
            continue
        penalty = 0.0
        if action.do_nothing and (action.line_set or action.redispatch):
            penalty -= 1.0
        if len(_completion_text(completion)) > 512:
            penalty -= 0.5
        verified_keys = set(_sim_by_action_key(simulations))
        if _action_key(action.model_dump()) not in verified_keys:
            penalty -= 1.0
        if error is not None:
            penalty -= 0.5
        rewards.append(penalty)
    return rewards


REWARD_FUNCS = [
    format_reward,
    schema_reward,
    verified_candidate_reward,
    safety_reward,
    task_objective_reward,
    anti_hacking_reward,
]


def load_grpo_rows(path: Path, prompt_style: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        metadata = row.get("metadata", {})
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Row {line_no} has no valid messages")
        if messages[0].get("role") != "user":
            raise ValueError(f"Row {line_no} first message must be user")
        simulations = metadata.get("simulations") or []
        if not isinstance(simulations, list) or not simulations:
            raise ValueError(f"Row {line_no} missing metadata.simulations")
        observation_summary = metadata.get("observation_summary") or {}
        if not isinstance(observation_summary, dict):
            observation_summary = {}
        row_out = {
            "prompt": [{"role": "user", "content": str(messages[0].get("content", ""))}],
            "task_id": str(metadata.get("task_id", "unknown")),
            "benchmark_tier": str(metadata.get("benchmark_tier", "unknown")),
            "seed_value": int(metadata.get("seed", -1)),
            "step_value": int(metadata.get("step", -1)),
            "observation_summary": observation_summary,
            "verified_simulation_results": simulations,
            "selected_action": metadata.get("selected_action", {}),
        }
        if prompt_style == "compact":
            row_out["prompt"] = [
                {"role": "user", "content": build_compact_grpo_prompt(row_out)}
            ]
        elif prompt_style != "original":
            raise ValueError(f"Unsupported prompt style: {prompt_style}")
        rows.append(row_out)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def summarize_grpo_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    task_actions: dict[str, Counter[str]] = defaultdict(Counter)
    candidate_counts: list[int] = []
    target_reachable = 0
    for row in rows:
        task = row["task_id"]
        tasks[task] += 1
        tiers[row["benchmark_tier"]] += 1
        action = selected_action_type(row.get("selected_action", {}))
        actions[action] += 1
        task_actions[task][action] += 1
        simulations = row["verified_simulation_results"]
        candidate_counts.append(len(simulations))
        if task == "single_fault" and any(float(sim.get("max_rho", 999.0)) < 0.80 for sim in _safe_simulations(simulations)):
            target_reachable += 1
    return {
        "rows": len(rows),
        "tasks": dict(sorted(tasks.items())),
        "benchmark_tiers": dict(sorted(tiers.items())),
        "selected_actions": dict(sorted(actions.items())),
        "task_actions": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(task_actions.items())
        },
        "avg_candidate_count": sum(candidate_counts) / len(candidate_counts),
        "max_candidate_count": max(candidate_counts),
        "single_fault_target_reachable_rows": target_reachable,
    }


def build_datasets(rows: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    if eval_ratio <= 0 or len(dataset) < 4:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verifier-reward GRPO training for Grid2Op action selection."
    )
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="grid2op-qwen3-4b-grpo-v1")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "grid2op-openenv-grpo"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-style", choices=["compact", "original"], default="compact")
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-type", choices=["grpo", "bnpo", "dr_grpo", "dapo", "sapo"], default="dapo")
    parser.add_argument("--scale-rewards", choices=["group", "batch", "none"], default="group")
    parser.add_argument("--multi-objective-aggregation", choices=["sum_then_normalize", "normalize_then_sum"], default="sum_then_normalize")
    parser.add_argument("--reward-weights", default="0.5,0.5,2.0,3.0,3.0,2.0")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-completions-to-print", type=int, default=4)
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--torch-empty-cache-steps", type=int, default=25)
    parser.add_argument("--new-lora-r", type=int, default=32)
    parser.add_argument("--new-lora-alpha", type=int, default=64)
    parser.add_argument("--new-lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def _parse_reward_weights(raw: str) -> list[float]:
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(weights) != len(REWARD_FUNCS):
        raise ValueError(f"Expected {len(REWARD_FUNCS)} reward weights, got {len(weights)}")
    return weights


def main() -> None:
    args = parse_args()
    if args.use_liger_kernel and args.device_map != "none":
        raise ValueError(
            "--use-liger-kernel is not safe with --device-map auto/sharded loading in this QLoRA setup. "
            "Use --no-use-liger-kernel for device_map=auto training, or --device-map none if the model fits per process."
        )

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    rows = load_grpo_rows(args.dataset, prompt_style=args.prompt_style)
    dataset_summary = summarize_grpo_rows(rows)
    train_dataset, eval_dataset = build_datasets(rows, args.eval_ratio, args.seed)
    reward_weights = _parse_reward_weights(args.reward_weights)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if "Qwen" in args.model:
        tokenizer.chat_template = QWEN_CHATML_TRAINING_TEMPLATE

    model_dtype, bf16, fp16, resolved_precision = resolve_precision(args.precision)
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_use_double_quant=True,
        )

    device_map = None if args.device_map == "none" else args.device_map
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": model_dtype,
        "device_map": device_map,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    peft_config = None
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.new_lora_r,
            lora_alpha=args.new_lora_alpha,
            lora_dropout=args.new_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        report_to=["wandb"],
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch_fused",
        fp16=fp16,
        bf16=bf16,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        reward_weights=reward_weights,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        multi_objective_aggregation=args.multi_objective_aggregation,
        mask_truncated_completions=args.mask_truncated_completions,
        use_liger_kernel=args.use_liger_kernel,
        pad_to_multiple_of=args.pad_to_multiple_of,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
    )

    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config={
            "base_model": args.model,
            "adapter": args.adapter,
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "dataset_summary": dataset_summary,
            "prompt_style": args.prompt_style,
            "reward_functions": [func.__name__ for func in REWARD_FUNCS],
            "reward_weights": reward_weights,
            "loss_type": args.loss_type,
            "scale_rewards": args.scale_rewards,
            "multi_objective_aggregation": args.multi_objective_aggregation,
            "num_generations": args.num_generations,
            "precision": resolved_precision,
            "torch_dtype": str(model_dtype),
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "qlora_4bit": args.use_4bit,
            "use_liger_kernel": args.use_liger_kernel,
            "mask_truncated_completions": args.mask_truncated_completions,
            "prompt_style": args.prompt_style,
        },
    )
    wandb.summary.update(
        {
            f"dataset/{key}": value
            for key, value in dataset_summary.items()
            if not isinstance(value, dict)
        }
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
