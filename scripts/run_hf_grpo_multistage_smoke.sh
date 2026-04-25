#!/usr/bin/env bash
set -euo pipefail

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-hack-submission-clean}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-grid2op-openenv-grpo}"
DATASET_PATH="${DATASET_PATH:-/mnt/datasets/grpo_multistage_informative_v1.jsonl}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/models/grid2op-qwen3-4b-sft-3k-v1}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/runs/grid2op-qwen3-4b-grpo-multistage-smoke}"
RUN_NAME="${RUN_NAME:-grid2op-qwen3-4b-grpo-multistage-smoke}"

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
  --max-steps 25 \
  --num-generations 2 \
  --max-completion-length 48 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-7 \
  --precision auto \
  --device-map auto \
  --loss-type dapo \
  --scale-rewards none \
  --eval-ratio 0.1 \
  --eval-steps 10 \
  --logging-steps 2 \
  --save-steps 10 \
  --judge-eval-max-rows 59 \
  --no-use-liger-kernel"

hf jobs run \
  --flavor l4x1 \
  --timeout 2h \
  -s WANDB_API_KEY="${WANDB_API_KEY}" \
  -s HF_TOKEN="${HF_TOKEN}" \
  -e WANDB_PROJECT="${WANDB_PROJECT_NAME}" \
  -v "hf://buckets/${HF_USERNAME}/grid2op-finals:/mnt" \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  -- \
  bash -lc "${REMOTE_CMD}"
