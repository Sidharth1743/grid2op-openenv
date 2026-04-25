#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your shell before running this script.}"
: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-ieee118-port}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-grid2op-openenv-ieee118-sft}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l40sx1}"
DATASET_PATH="${DATASET_PATH:-/mnt/datasets/ieee118_teacher_actions_v1.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/models/grid2op-qwen3-4b-sft-ieee118-v1}"
RUN_NAME="${RUN_NAME:-grid2op-qwen3-4b-sft-ieee118-v1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_STEPS="${MAX_STEPS:-800}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
EVAL_RATIO="${EVAL_RATIO:-0.05}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PRECISION="${PRECISION:-auto}"

REMOTE_CMD=$(cat <<EOF
set -euxo pipefail
apt-get update
apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace
cd /workspace
uv sync --frozen --no-dev
uv pip install torch datasets transformers trl peft accelerate bitsandbytes wandb
uv run python scripts/train_sft.py \
  --dataset ${DATASET_PATH} \
  --model ${BASE_MODEL} \
  --output-dir ${OUTPUT_PATH} \
  --run-name ${RUN_NAME} \
  --wandb-project ${WANDB_PROJECT_NAME} \
  --eval-ratio ${EVAL_RATIO} \
  --max-length ${MAX_LENGTH} \
  --max-steps ${MAX_STEPS} \
  --learning-rate ${LEARNING_RATE} \
  --per-device-train-batch-size ${TRAIN_BATCH_SIZE} \
  --per-device-eval-batch-size ${EVAL_BATCH_SIZE} \
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
  --logging-steps ${LOGGING_STEPS} \
  --save-steps ${SAVE_STEPS} \
  --eval-steps ${EVAL_STEPS} \
  --precision ${PRECISION} \
  --device-map auto \
  --use-4bit \
  --assistant-only-loss \
  --patch-qwen-training-template \
  --no-use-liger-kernel
EOF
)

export HF_USERNAME BRANCH_NAME WANDB_PROJECT_NAME HF_JOB_FLAVOR DATASET_PATH OUTPUT_PATH RUN_NAME BASE_MODEL MAX_STEPS LEARNING_RATE TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS LOGGING_STEPS SAVE_STEPS EVAL_STEPS EVAL_RATIO MAX_LENGTH PRECISION REMOTE_CMD

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
    timeout="12h",
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
print(f"Will write model to: {os.environ['OUTPUT_PATH']}")
PY
