#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-hack-submission-clean}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l4x1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_PATH="${ADAPTER_PATH:-/mnt/models/grid2op-qwen3-4b-sft-3k-v1}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/datasets/grpo_multistage_rollout_h3_v1.jsonl}"
TASK_IDS="${TASK_IDS:-multi_stage_cascade}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-12}"
SEED_START="${SEED_START:-0}"
PROMPT_CANDIDATE_COUNT="${PROMPT_CANDIDATE_COUNT:-3}"
LOOKAHEAD_FIRST_ACTIONS="${LOOKAHEAD_FIRST_ACTIONS:-5}"
LOOKAHEAD_HORIZON="${LOOKAHEAD_HORIZON:-3}"
MIN_ADVANTAGE="${MIN_ADVANTAGE:-0.5}"
PRECISION="${PRECISION:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

REMOTE_CMD=$(cat <<EOF
set -euxo pipefail
apt-get update
apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
command -v curl
command -v uv
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace
mkdir -p "$(dirname "${OUTPUT_PATH}")"
cd /workspace
uv sync --frozen --no-dev
uv pip install torch datasets transformers trl peft accelerate bitsandbytes fastapi uvicorn gradio pandas
uv run python -m grid2op_env.server.app --host 127.0.0.1 --port 8018 > /tmp/grid2op_collect_server.log 2>&1 &
SERVER_PID=\$!
echo "server pid: \$SERVER_PID"
for i in \$(seq 1 60); do
  if curl -fsS http://127.0.0.1:8018/health >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://127.0.0.1:8018/health >/dev/null; then
  echo '===== server log ====='
  cat /tmp/grid2op_collect_server.log
  ps -p "\$SERVER_PID" -f || true
  exit 1
fi
uv run python scripts/collect_rollout_grpo_dataset.py \
  --base-url http://127.0.0.1:8018 \
  --output-path ${OUTPUT_PATH} \
  --model ${BASE_MODEL} \
  --adapter ${ADAPTER_PATH} \
  --task-id ${TASK_IDS} \
  --episodes-per-task ${EPISODES_PER_TASK} \
  --seed-start ${SEED_START} \
  --prompt-candidate-count ${PROMPT_CANDIDATE_COUNT} \
  --lookahead-first-actions ${LOOKAHEAD_FIRST_ACTIONS} \
  --lookahead-horizon ${LOOKAHEAD_HORIZON} \
  --min-advantage ${MIN_ADVANTAGE} \
  --precision ${PRECISION} \
  --max-new-tokens ${MAX_NEW_TOKENS}
uv run python scripts/check_dataset_quality.py ${OUTPUT_PATH}
wc -l ${OUTPUT_PATH}
EOF
)

export HF_USERNAME BRANCH_NAME HF_JOB_FLAVOR BASE_MODEL ADAPTER_PATH OUTPUT_PATH TASK_IDS EPISODES_PER_TASK SEED_START PROMPT_CANDIDATE_COUNT LOOKAHEAD_FIRST_ACTIONS LOOKAHEAD_HORIZON MIN_ADVANTAGE PRECISION MAX_NEW_TOKENS REMOTE_CMD

python - <<'PY'
import os
from huggingface_hub import Volume, run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-lc", os.environ["REMOTE_CMD"]],
    env={},
    secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    flavor=os.environ["HF_JOB_FLAVOR"],
    timeout="6h",
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
print(f"Will write dataset to: {os.environ['OUTPUT_PATH']}")
PY
