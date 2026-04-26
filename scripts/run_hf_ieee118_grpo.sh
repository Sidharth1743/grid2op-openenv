#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your shell before running this script.}"
: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-ieee118-port}"
GIT_REF="${GIT_REF:-}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/Sidharth1743/grid2op-openenv.git}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-grid2op-openenv-ieee118-grpo}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l40sx1}"
DATASET_PATH="${DATASET_PATH:-/mnt/datasets/ieee118_teacher_actions_v1.jsonl}"
ADAPTER_PATH="${ADAPTER_PATH:-Sidharth1743/grid2op-qwen3-4b-sft-final}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/runs/grid2op-qwen3-4b-grpo-ieee118-v1}"
RUN_NAME="${RUN_NAME:-grid2op-qwen3-4b-grpo-ieee118-v1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_STEPS="${MAX_STEPS:-300}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-96}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-25}"
EVAL_RATIO="${EVAL_RATIO:-0.05}"
PRECISION="${PRECISION:-auto}"
LOSS_TYPE="${LOSS_TYPE:-dapo}"
SCALE_REWARDS="${SCALE_REWARDS:-none}"
PROMPT_STYLE="${PROMPT_STYLE:-compact}"
JUDGE_EVAL_MAX_ROWS="${JUDGE_EVAL_MAX_ROWS:-96}"
FOLLOW_HF_JOB_LOGS="${FOLLOW_HF_JOB_LOGS:-true}"

REMOTE_CMD=$(cat <<EOF
set -euxo pipefail
echo "[phase] apt"
apt-get update
apt-get install -y git curl
echo "[phase] install_uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
command -v uv
echo "[phase] clone_repo"
if [ -n "${GIT_REF}" ]; then
  git clone ${GIT_REPO_URL} /workspace
  cd /workspace
  git checkout ${GIT_REF}
else
  git clone --depth 1 --branch ${BRANCH_NAME} ${GIT_REPO_URL} /workspace
  cd /workspace
fi
echo "[phase] repo_state"
git rev-parse HEAD
git log -1 --oneline
echo "[phase] sync_env"
uv sync --frozen --no-dev
uv pip install torch datasets transformers trl peft accelerate bitsandbytes wandb
echo "[phase] dataset_probe"
ls -lh ${DATASET_PATH}
uv run python - <<'PY'
from pathlib import Path

path = Path("${DATASET_PATH}")
print({"dataset_exists": path.exists(), "dataset_path": str(path), "size_bytes": path.stat().st_size if path.exists() else None})
if path.exists():
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            if idx > 3:
                break
            print({"dataset_sample_line": idx, "preview": line[:240]})
PY
mkdir -p ${OUTPUT_PATH}
echo "[phase] run_grpo"
uv run python scripts/train_grpo_verifier.py \
  --dataset ${DATASET_PATH} \
  --model ${BASE_MODEL} \
  --adapter ${ADAPTER_PATH} \
  --output-dir ${OUTPUT_PATH} \
  --run-name ${RUN_NAME} \
  --wandb-project ${WANDB_PROJECT_NAME} \
  --reward-mode legacy \
  --no-require-noop-baseline \
  --no-informative-multistage-only \
  --prompt-style ${PROMPT_STYLE} \
  --max-steps ${MAX_STEPS} \
  --num-generations ${NUM_GENERATIONS} \
  --max-completion-length ${MAX_COMPLETION_LENGTH} \
  --per-device-train-batch-size ${TRAIN_BATCH_SIZE} \
  --per-device-eval-batch-size ${EVAL_BATCH_SIZE} \
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
  --learning-rate ${LEARNING_RATE} \
  --precision ${PRECISION} \
  --device-map auto \
  --loss-type ${LOSS_TYPE} \
  --scale-rewards ${SCALE_REWARDS} \
  --eval-ratio ${EVAL_RATIO} \
  --eval-steps ${EVAL_STEPS} \
  --logging-steps ${LOGGING_STEPS} \
  --save-steps ${SAVE_STEPS} \
  --judge-eval-max-rows ${JUDGE_EVAL_MAX_ROWS} \
  --no-use-liger-kernel
EOF
)

export HF_USERNAME BRANCH_NAME GIT_REF GIT_REPO_URL WANDB_PROJECT_NAME HF_JOB_FLAVOR DATASET_PATH ADAPTER_PATH OUTPUT_PATH RUN_NAME BASE_MODEL MAX_STEPS NUM_GENERATIONS MAX_COMPLETION_LENGTH LEARNING_RATE TRAIN_BATCH_SIZE EVAL_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS LOGGING_STEPS SAVE_STEPS EVAL_STEPS EVAL_RATIO PRECISION LOSS_TYPE SCALE_REWARDS PROMPT_STYLE JUDGE_EVAL_MAX_ROWS FOLLOW_HF_JOB_LOGS REMOTE_CMD

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python - <<'PY'
import os
from huggingface_hub import HfApi, Volume, run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-lc", os.environ["REMOTE_CMD"]],
    env={"WANDB_PROJECT": os.environ["WANDB_PROJECT_NAME"]},
    secrets={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        "HF_TOKEN": os.environ["HF_TOKEN"],
    },
    flavor=os.environ["HF_JOB_FLAVOR"],
    timeout="10h",
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
print(f"Will read dataset from: {os.environ['DATASET_PATH']}")
print(f"Will initialize from adapter: {os.environ['ADAPTER_PATH']}")
print(f"Will write adapter to: {os.environ['OUTPUT_PATH']}")

follow_logs = os.environ.get("FOLLOW_HF_JOB_LOGS", "true").lower() == "true"
if follow_logs:
    print("[phase] follow_job_logs")
    api = HfApi(token=os.environ["HF_TOKEN"])
    for line in api.fetch_job_logs(job_id=job.id, follow=True):
        print(line, end="")
    final_info = api.inspect_job(job_id=job.id)
    print()
    print(
        {
            "job_id": job.id,
            "final_status": str(getattr(final_info, "status", None)),
            "final_stage": str(getattr(final_info, "stage", None)),
            "job_url": job.url,
        }
    )
PY
