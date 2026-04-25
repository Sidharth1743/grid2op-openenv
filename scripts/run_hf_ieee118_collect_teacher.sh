#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-ieee118-port}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-a100-large}"
GRID2OP_ENV_NAME="${GRID2OP_ENV_NAME:-l2rpn_idf_2023}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.6-27B}"
TEACHER_PORT="${TEACHER_PORT:-8001}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-16384}"
TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.90}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/datasets/ieee118_teacher_actions_v1.jsonl}"
TASK_IDS="${TASK_IDS:-single_fault n_minus_1 cascade_prevent multi_stage_cascade}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-8}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-6}"
SEED_START="${SEED_START:-0}"
MAX_TOKENS="${MAX_TOKENS:-320}"
TEMPERATURE="${TEMPERATURE:-0.2}"
GRID_MESSAGE_TIMEOUT_S="${GRID_MESSAGE_TIMEOUT_S:-240}"
GRID_CONNECT_TIMEOUT_S="${GRID_CONNECT_TIMEOUT_S:-30}"

REMOTE_CMD=$(cat <<EOF
set -euxo pipefail
apt-get update
apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
export GRID2OP_ENV_NAME="${GRID2OP_ENV_NAME}"
export GRID2OP_MESSAGE_TIMEOUT_S="${GRID_MESSAGE_TIMEOUT_S}"
export GRID2OP_CONNECT_TIMEOUT_S="${GRID_CONNECT_TIMEOUT_S}"
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace
mkdir -p "$(dirname "${OUTPUT_PATH}")"
cd /workspace
uv sync --frozen --no-dev
uv pip install torch datasets transformers trl peft accelerate bitsandbytes fastapi uvicorn gradio pandas "vllm>=0.19.0"
uv run python - <<'PY'
import grid2op

env = grid2op.make("${GRID2OP_ENV_NAME}")
print({"preloaded_env": "${GRID2OP_ENV_NAME}", "n_line": int(env.n_line), "n_gen": int(env.n_gen)})
env.close()
PY
export API_KEY="EMPTY"
export GRID2OP_TEACHER_MODEL="${TEACHER_MODEL}"
export GRID2OP_TEACHER_API_BASE_URL="http://127.0.0.1:${TEACHER_PORT}/v1"
python -m vllm.entrypoints.openai.api_server \
  --model ${TEACHER_MODEL} \
  --host 127.0.0.1 \
  --port ${TEACHER_PORT} \
  --max-model-len ${TEACHER_MAX_MODEL_LEN} \
  --gpu-memory-utilization ${TEACHER_GPU_MEMORY_UTILIZATION} \
  --language-model-only \
  > /tmp/grid2op_ieee118_teacher_vllm.log 2>&1 &
VLLM_PID=\$!
echo "teacher pid: \$VLLM_PID"
for i in \$(seq 1 180); do
  if curl -fsS http://127.0.0.1:${TEACHER_PORT}/v1/models >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://127.0.0.1:${TEACHER_PORT}/v1/models >/dev/null; then
  echo '===== teacher log ====='
  cat /tmp/grid2op_ieee118_teacher_vllm.log
  ps -p "\$VLLM_PID" -f || true
  exit 1
fi
uv run python -m grid2op_env.server.app --host 127.0.0.1 --port 8018 > /tmp/grid2op_ieee118_collect_server.log 2>&1 &
SERVER_PID=\$!
echo "server pid: \$SERVER_PID"
for i in \$(seq 1 90); do
  if curl -fsS http://127.0.0.1:8018/health >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://127.0.0.1:8018/health >/dev/null; then
  echo '===== server log ====='
  cat /tmp/grid2op_ieee118_collect_server.log
  ps -p "\$SERVER_PID" -f || true
  exit 1
fi
uv run python scripts/collect_teacher_dataset.py \
  --base-url http://127.0.0.1:8018 \
  --teacher-api-base-url http://127.0.0.1:${TEACHER_PORT}/v1 \
  --output ${OUTPUT_PATH} \
  --task-id ${TASK_IDS} \
  --episodes-per-task ${EPISODES_PER_TASK} \
  --max-steps-per-episode ${MAX_STEPS_PER_EPISODE} \
  --seed-start ${SEED_START} \
  --scenario-mode benchmark \
  --model ${TEACHER_MODEL} \
  --max-tokens ${MAX_TOKENS} \
  --temperature ${TEMPERATURE} \
  --connect-timeout-s ${GRID_CONNECT_TIMEOUT_S} \
  --message-timeout-s ${GRID_MESSAGE_TIMEOUT_S}
uv run python scripts/check_dataset_quality.py ${OUTPUT_PATH}
wc -l ${OUTPUT_PATH}
EOF
)

export HF_USERNAME BRANCH_NAME HF_JOB_FLAVOR GRID2OP_ENV_NAME TEACHER_MODEL TEACHER_PORT TEACHER_MAX_MODEL_LEN TEACHER_GPU_MEMORY_UTILIZATION OUTPUT_PATH TASK_IDS EPISODES_PER_TASK MAX_STEPS_PER_EPISODE SEED_START MAX_TOKENS TEMPERATURE GRID_MESSAGE_TIMEOUT_S GRID_CONNECT_TIMEOUT_S REMOTE_CMD

python - <<'PY'
import os
from huggingface_hub import Volume, run_job

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "-lc", os.environ["REMOTE_CMD"]],
    env={},
    secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
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
print(f"Will write dataset to: {os.environ['OUTPUT_PATH']}")
PY
