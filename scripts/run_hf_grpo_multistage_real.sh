#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your shell before running this script.}"
: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-hack-submission-clean}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-grid2op-openenv-grpo}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l4x1}"
DATASET_PATH="${DATASET_PATH:-/mnt/datasets/grpo_multistage_informative_v1.jsonl}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/models/grid2op-qwen3-4b-sft-3k-v1}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/runs/grid2op-qwen3-4b-grpo-multistage-v1}"
RUN_NAME="${RUN_NAME:-grid2op-qwen3-4b-grpo-multistage-v1}"

REMOTE_CMD="apt-get update && \
apt-get install -y git curl && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
export PATH=\"\$HOME/.local/bin:\$PATH\" && \
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace && \
cd /workspace && \
uv sync --frozen --no-dev && \
uv pip install torch datasets transformers trl peft accelerate bitsandbytes wandb && \
uv run python scripts/train_grpo_verifier.py \
  --dataset ${DATASET_PATH} \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter ${ADAPTER_PATH} \
  --output-dir ${OUTPUT_PATH} \
  --run-name ${RUN_NAME} \
  --reward-mode relative_multistage \
  --task-filter multi_stage_cascade \
  --require-noop-baseline \
  --informative-multistage-only \
  --min-noop-gap 0.01 \
  --prompt-style compact \
  --max-steps 200 \
  --num-generations 2 \
  --max-completion-length 48 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 5e-7 \
  --precision auto \
  --device-map auto \
  --loss-type dapo \
  --scale-rewards none \
  --eval-ratio 0.1 \
  --eval-steps 25 \
  --logging-steps 5 \
  --save-steps 25 \
  --judge-eval-max-rows 59 \
  --no-use-liger-kernel"

export HF_USERNAME BRANCH_NAME WANDB_PROJECT_NAME HF_JOB_FLAVOR DATASET_PATH ADAPTER_PATH OUTPUT_PATH RUN_NAME REMOTE_CMD

python - <<'PY'
import os
from huggingface_hub import Volume, run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-lc", os.environ["REMOTE_CMD"]],
    env={"WANDB_PROJECT": os.environ["WANDB_PROJECT_NAME"]},
    secrets={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        "HF_TOKEN": os.environ["HF_TOKEN"],
    },
    flavor=os.environ["HF_JOB_FLAVOR"],
    timeout="8h",
    volumes=[
        Volume(
            type="bucket",
            source=f'{os.environ["HF_USERNAME"]}/grid2op-finals',
            mount_path="/mnt",
        )
    ],
    token=os.environ["HF_TOKEN"],
)
print(f"Job started with ID: {job.id}")
print(f"View at: {job.url}")
PY
