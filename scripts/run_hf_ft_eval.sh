#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-hack-submission-clean}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l4x1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/runs/grid2op-qwen3-4b-grpo-multistage-v1}"
RUN_LABEL="${RUN_LABEL:-grid2op-qwen3-4b-grpo-multistage-v1}"
TASK_IDS="${TASK_IDS:-multi_stage_cascade}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
SEED_START="${SEED_START:-0}"
PRECISION="${PRECISION:-auto}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.1}"
LOG_DIR="${LOG_DIR:-/mnt/evals}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_LABEL}_seed${SEED_START}_${EPISODES_PER_TASK}eps.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${LOG_DIR}/${RUN_LABEL}_seed${SEED_START}_${EPISODES_PER_TASK}eps.summary.json}"

REMOTE_CMD="apt-get update && \
apt-get install -y git curl && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
export PATH=\"\$HOME/.local/bin:\$PATH\" && \
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace && \
mkdir -p ${LOG_DIR} && \
cd /workspace && \
uv sync --frozen --no-dev && \
uv pip install torch datasets transformers peft accelerate bitsandbytes && \
uv run python ft_inference.py \
  --model ${BASE_MODEL} \
  --adapter ${ADAPTER_PATH} \
  --task-id ${TASK_IDS} \
  --episodes-per-task ${EPISODES_PER_TASK} \
  --seed-start ${SEED_START} \
  --precision ${PRECISION} \
  --success-threshold ${SUCCESS_THRESHOLD} \
  2>&1 | tee ${LOG_PATH} && \
uv run python scripts/check_ft_inference_log.py ${LOG_PATH} | tee ${SUMMARY_PATH}"

export HF_USERNAME BRANCH_NAME HF_JOB_FLAVOR BASE_MODEL ADAPTER_PATH RUN_LABEL TASK_IDS EPISODES_PER_TASK SEED_START PRECISION SUCCESS_THRESHOLD LOG_DIR LOG_PATH SUMMARY_PATH REMOTE_CMD

python - <<'PY'
import os
from huggingface_hub import Volume, run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-lc", os.environ["REMOTE_CMD"]],
    env={},
    secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    flavor=os.environ["HF_JOB_FLAVOR"],
    timeout="4h",
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
print(f"Will write log to: {os.environ['LOG_PATH']}")
print(f"Will write summary to: {os.environ['SUMMARY_PATH']}")
PY
