# Grid2Op Environment

Standalone OpenEnv environment package for the full `PROJECT.md` design.

## File structure

```text
grid2op_env/
├── .dockerignore
├── .env
├── .gitignore
├── __init__.py
├── models.py
├── client.py
├── inference.py
├── README.md
├── openenv.yaml
├── outputs/
│   ├── logs/
│   └── evals/
├── pyproject.toml
└── server/
    ├── grid_environment.py
    ├── tasks.py
    ├── graders.py
    ├── app.py
    ├── requirements.txt
    └── Dockerfile
```

The top-level package now follows the canonical OpenEnv environment layout:

- `.dockerignore`
- `__init__.py`
- `models.py`
- `client.py`
- `README.md`
- `openenv.yaml`
- `pyproject.toml`
- `outputs/logs`
- `outputs/evals`
- `server/`

Supporting files outside the minimum template remain for quality and verification:

- `inference.py`
- `tests/test_grid2op_env.py`
- helper server modules such as `tasks.py`, `graders.py`, and `logging_utils.py`

## What is implemented

- Grid2Op core simulator using `l2rpn_case14_sandbox`
- Typed `GridAction`, `GridObservation`, and `GridState`
- Three tasks: `single_fault`, `n_minus_1`, `cascade_prevent`
- Reset-time scenario injection and retry logic for non-convergent starts
- Shaped reward, episode logging, and deterministic graders
- OpenEnv WebSocket interface plus `/tasks`, `/grader`, and `/baseline`
- Qwen3.5 baseline using the Chat Completions API
- Local Docker workflow with dataset pre-download

## Local Docker workflow

Build:

```bash
cd grid2op_env
docker build -t grid2op-env:local -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 7860:7860 grid2op-env:local
```

If your Qwen-compatible API is running on the host machine, use:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=EMPTY \
  -p 7860:7860 \
  grid2op-env:local
```

## Local UV workflow

```bash
cd grid2op_env
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev grid2op-smoke --task-id single_fault --steps 1
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev server --port 7860
env UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/test_grid2op_env.py -q
```

## Qwen baseline

The baseline uses the OpenAI Python SDK against a local Chat Completions API.

```bash
cat > .env <<'EOF'
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
EOF

env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev inference.py
```
