from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET = "outputs/datasets/grid2op_sft_v1.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/models/grid2op-qwen3-4b-sft-v1"
QWEN_CHATML_TRAINING_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' }}
{%- if message['role'] == 'assistant' %}
{% generation %}{{- message['content'] }}{% endgeneration %}
{%- else %}
{{- message['content'] }}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
"""


def resolve_precision(precision: str) -> tuple[torch.dtype, bool, bool, str]:
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "fp16"
    if precision == "bf16":
        return torch.bfloat16, True, False, precision
    if precision == "fp16":
        return torch.float16, False, True, precision
    if precision == "fp32":
        return torch.float32, False, False, precision
    raise ValueError(f"Unsupported precision: {precision}")


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Row {line_no} has no valid messages list")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Row {line_no} final message must be assistant")
        content = messages[-1].get("content")
        if not isinstance(content, str):
            raise ValueError(f"Row {line_no} assistant content must be a string")
        json.loads(content)
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    task_actions: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_chars: list[int] = []
    completion_chars: list[int] = []

    for row in rows:
        metadata = row.get("metadata", {})
        task_id = str(metadata.get("task_id", "unknown"))
        tasks[task_id] += 1
        tiers[str(metadata.get("benchmark_tier", "unknown"))] += 1
        policies[str(metadata.get("label_policy", "unknown"))] += 1

        selected_action = metadata.get("selected_action", {})
        action_type = selected_action_type(selected_action)
        actions[action_type] += 1
        task_actions[task_id][action_type] += 1

        messages = row["messages"]
        prompt_chars.append(sum(len(str(message.get("content", ""))) for message in messages[:-1]))
        completion_chars.append(len(str(messages[-1].get("content", ""))))

    return {
        "rows": len(rows),
        "tasks": dict(sorted(tasks.items())),
        "actions": dict(sorted(actions.items())),
        "benchmark_tiers": dict(sorted(tiers.items())),
        "label_policies": dict(sorted(policies.items())),
        "task_actions": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(task_actions.items())
        },
        "avg_prompt_chars": sum(prompt_chars) / len(prompt_chars),
        "max_prompt_chars": max(prompt_chars),
        "avg_completion_chars": sum(completion_chars) / len(completion_chars),
        "max_completion_chars": max(completion_chars),
    }


def selected_action_type(action: dict[str, Any]) -> str:
    if action.get("do_nothing"):
        return "do_nothing"
    if action.get("redispatch"):
        return "redispatch"
    if action.get("line_set"):
        statuses = [int(value) for value in action["line_set"].values()]
        if statuses and statuses[0] == 1:
            return "reconnect_line"
        if statuses and statuses[0] == -1:
            return "disconnect_line"
        return "line_set"
    return "empty"


def add_token_lengths(dataset: Dataset, tokenizer, max_samples: int) -> dict[str, Any]:
    lengths: list[int] = []
    sample_count = min(len(dataset), max_samples)
    for index in range(sample_count):
        text = tokenizer.apply_chat_template(
            dataset[index]["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    if not lengths:
        return {"sampled_token_lengths": 0}
    lengths_sorted = sorted(lengths)
    p95_index = min(len(lengths_sorted) - 1, int(0.95 * (len(lengths_sorted) - 1)))
    return {
        "sampled_token_lengths": len(lengths),
        "avg_tokens": sum(lengths) / len(lengths),
        "max_tokens": max(lengths),
        "p95_tokens": lengths_sorted[p95_index],
    }


def build_datasets(rows: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    dataset = Dataset.from_list([{"messages": row["messages"], "metadata": row.get("metadata", {})} for row in rows])
    dataset = dataset.shuffle(seed=seed)
    if eval_ratio <= 0 or len(dataset) < 4:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Grid2Op chat-action data.")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default="grid2op-qwen3-4b-sft-v1")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "grid2op-openenv-sft"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-qwen-training-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-liger-kernel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--torch-empty-cache-steps", type=int, default=25)
    parser.add_argument("--token-length-samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_liger_kernel and args.device_map != "none":
        raise ValueError(
            "--use-liger-kernel is not safe with --device-map auto/sharded loading in this QLoRA setup. "
            "Use --no-use-liger-kernel for two-GPU device_map=auto training, or use --device-map none "
            "only when the full model fits on one GPU/process."
        )
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    rows = load_jsonl_rows(args.dataset)
    dataset_summary = summarize_rows(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if args.assistant_only_loss and args.patch_qwen_training_template and "Qwen" in args.model:
        tokenizer.chat_template = QWEN_CHATML_TRAINING_TEMPLATE

    train_dataset, eval_dataset = build_datasets(
        rows=rows,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    token_summary = add_token_lengths(
        Dataset.from_list([{"messages": row["messages"]} for row in rows]),
        tokenizer=tokenizer,
        max_samples=args.token_length_samples,
    )
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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        run_name=args.run_name,
        report_to=["wandb"],
        max_length=args.max_length,
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
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=fp16,
        bf16=bf16,
        packing=args.packing,
        pad_to_multiple_of=args.pad_to_multiple_of,
        assistant_only_loss=args.assistant_only_loss,
        use_liger_kernel=args.use_liger_kernel,
        torch_empty_cache_steps=args.torch_empty_cache_steps,
        remove_unused_columns=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config={
            "base_model": args.model,
            "dataset": str(args.dataset),
            "output_dir": str(args.output_dir),
            "dataset_summary": dataset_summary,
            "token_summary": token_summary,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            },
            "qlora_4bit": args.use_4bit,
            "max_length": args.max_length,
            "precision": resolved_precision,
            "torch_dtype": str(model_dtype),
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "use_liger_kernel": args.use_liger_kernel,
            "patch_qwen_training_template": args.patch_qwen_training_template,
            "pad_to_multiple_of": args.pad_to_multiple_of,
            "torch_empty_cache_steps": args.torch_empty_cache_steps,
        },
    )
    wandb.summary.update({f"dataset/{key}": value for key, value in dataset_summary.items() if not isinstance(value, dict)})
    wandb.summary.update({f"tokens/{key}": value for key, value in token_summary.items()})

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    wandb.finish()


if __name__ == "__main__":
    main()
