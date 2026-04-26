#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your shell before running this script.}"

HF_USERNAME="${HF_USERNAME:-Sidharth1743}"
BRANCH_NAME="${BRANCH_NAME:-ieee118-port}"
HF_JOB_FLAVOR="${HF_JOB_FLAVOR:-l40sx1}"
GRID2OP_ENV_NAME="${GRID2OP_ENV_NAME:-l2rpn_idf_2023}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_PATH="${ADAPTER_PATH:-none}"
RUN_LABEL="${RUN_LABEL:-ieee118-qwen3-4b-base-benchmark}"
TASK_IDS="${TASK_IDS:-single_fault n_minus_1 cascade_prevent multi_stage_cascade}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
SEED_START="${SEED_START:-0}"
PRECISION="${PRECISION:-auto}"
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.1}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
GRID_MESSAGE_TIMEOUT_S="${GRID_MESSAGE_TIMEOUT_S:-240}"
GRID_CONNECT_TIMEOUT_S="${GRID_CONNECT_TIMEOUT_S:-30}"
LOG_DIR="${LOG_DIR:-/mnt/evals}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_LABEL}_seed${SEED_START}_${EPISODES_PER_TASK}eps.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${LOG_DIR}/${RUN_LABEL}_seed${SEED_START}_${EPISODES_PER_TASK}eps.summary.json}"

REMOTE_CMD=$(cat <<EOF
set -euxo pipefail
echo "[phase] apt"
apt-get update
apt-get install -y git curl
echo "[phase] install_uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
export GRID2OP_ENV_NAME="${GRID2OP_ENV_NAME}"
export GRID2OP_MESSAGE_TIMEOUT_S="${GRID_MESSAGE_TIMEOUT_S}"
export GRID2OP_CONNECT_TIMEOUT_S="${GRID_CONNECT_TIMEOUT_S}"
echo "[phase] clone_repo"
git clone --depth 1 --branch ${BRANCH_NAME} https://github.com/Sidharth1743/grid2op-openenv.git /workspace
mkdir -p ${LOG_DIR}
cd /workspace
echo "[phase] repo_state"
git rev-parse HEAD
git log -1 --oneline
echo "[phase] sync_env"
uv sync --frozen --no-dev
uv pip install torch datasets transformers trl peft accelerate bitsandbytes fastapi uvicorn gradio pandas
echo "[phase] preload_grid2op_env"
uv run python - <<'PY'
import grid2op

env = grid2op.make("${GRID2OP_ENV_NAME}")
print({"preloaded_env": "${GRID2OP_ENV_NAME}", "n_line": int(env.n_line), "n_gen": int(env.n_gen)})
env.close()
PY
echo "[phase] start_env_server"
uv run python -m grid2op_env.server.app --host 127.0.0.1 --port 8018 > /tmp/grid2op_ieee118_eval_server.log 2>&1 &
SERVER_PID=\$!
echo "server pid: \$SERVER_PID"
for i in \$(seq 1 90); do
  echo "[wait_server] attempt=\$i"
  if curl -fsS http://127.0.0.1:8018/health >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -fsS http://127.0.0.1:8018/health >/dev/null; then
  echo '===== server log ====='
  cat /tmp/grid2op_ieee118_eval_server.log
  ps -p "\$SERVER_PID" -f || true
  exit 1
fi
echo "[phase] run_eval"
uv run python ft_inference.py \
  --base-url http://127.0.0.1:8018 \
  --model ${BASE_MODEL} \
  --adapter ${ADAPTER_PATH} \
  --task-id ${TASK_IDS} \
  --episodes-per-task ${EPISODES_PER_TASK} \
  --seed-start ${SEED_START} \
  --scenario-mode benchmark \
  --candidate-count ${CANDIDATE_COUNT} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --precision ${PRECISION} \
  --success-threshold ${SUCCESS_THRESHOLD} \
  --connect-timeout-s ${GRID_CONNECT_TIMEOUT_S} \
  --message-timeout-s ${GRID_MESSAGE_TIMEOUT_S} \
  2>&1 | tee ${LOG_PATH} || {
    echo '===== server log ====='
    tail -n 200 /tmp/grid2op_ieee118_eval_server.log || true
    exit 1
  }
echo "[phase] summarize_eval"
uv run python scripts/check_ft_inference_log.py ${LOG_PATH} | tee ${SUMMARY_PATH}
EOF
)

export HF_USERNAME BRANCH_NAME HF_JOB_FLAVOR GRID2OP_ENV_NAME BASE_MODEL ADAPTER_PATH RUN_LABEL TASK_IDS EPISODES_PER_TASK SEED_START PRECISION SUCCESS_THRESHOLD CANDIDATE_COUNT MAX_NEW_TOKENS GRID_MESSAGE_TIMEOUT_S GRID_CONNECT_TIMEOUT_S LOG_DIR LOG_PATH SUMMARY_PATH REMOTE_CMD

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
print(f"Will write log to: {os.environ['LOG_PATH']}")
print(f"Will write summary to: {os.environ['SUMMARY_PATH']}")
PY
