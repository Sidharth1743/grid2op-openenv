# Grid2Op Environment Implementation

This document describes the implemented `grid2op_env` package, verifies it against `PROJECT.md`, and provides the exact run procedures for local development, local Docker, and baseline evaluation.

## Scope

The implementation is a standalone OpenEnv environment package at [grid2op_env](/home/sidharth/Desktop/Openenv_modules/grid2op_env) with the inference runner at [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py).


## Implemented file structure

```text
grid2op_env/
├── .dockerignore
├── .env
├── .gitignore
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── outputs/
│   ├── evals/
│   └── logs/
├── pyproject.toml
├── README.md
├── tests/
│   └── test_grid2op_env.py
└── server/
    ├── __init__.py
    ├── app.py
    ├── Dockerfile
    ├── graders.py
    ├── grid_environment.py
    ├── logging_utils.py
    ├── requirements.txt
    └── tasks.py
```

The canonical OpenEnv package structure is present at the top level:

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

Additional support files are retained beyond the minimum template:

- `tests/`
- `inference.py`
- helper modules for tasks, graders, and logging

## Verification against PROJECT.md

### Big picture

`PROJECT.md` specifies two cooperating pieces:

- a server that owns Grid2Op and exposes the environment over OpenEnv
- a client package that agents use via `reset()`, `step()`, and `state()`

Implemented:

- server: [grid2op_env/server/app.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/app.py)
- adapter: [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- client: [grid2op_env/client.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/client.py)

### Layer 1: Grid2Op physics engine

Required by `PROJECT.md`:

- use `l2rpn_case14_sandbox`
- expose `rho`, `gen_p`, `load_p`, `line_status`, `timestep_overflow`
- support do-nothing, line status changes, and redispatch

Implemented:

- Grid2Op environment creation in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- dataset pre-download in [grid2op_env/server/Dockerfile](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/Dockerfile)
- typed observation conversion in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- action translation in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)

Verification:

- smoke tests passed for `single_fault`, `n_minus_1`, and `cascade_prevent`
- local Docker build completed successfully and downloaded `l2rpn_case14_sandbox`

### Layer 2: OpenEnv adapter

Required by `PROJECT.md`:

- typed `GridAction`
- typed `GridObservation`
- typed `state()`
- JSON-safe translation between Grid2Op and OpenEnv
- maintain `episode_log`

Implemented:

- models in [grid2op_env/models.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/models.py)
- adapter in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- `episode_log` stored on `GridState`

Verification:

- direct test coverage in [grid2op_env/tests/test_grid2op_env.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/tests/test_grid2op_env.py)
- `3 passed` in focused pytest run

### Layer 3: task and scenario system

Required by `PROJECT.md`:

- `single_fault`, `n_minus_1`, `cascade_prevent`
- `single_fault` should warm up to a 90-95% style high-loading state
- `n_minus_1` should reset with one disconnected line
- `cascade_prevent` should reset with two disconnected lines and 15% load increase
- reset-time convergence retry logic for hard cases

Implemented:

- task metadata and reset injection in [grid2op_env/server/tasks.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/tasks.py)
- `single_fault`: warmup search until high line loading is found
- `n_minus_1`: `set_line_status=[(0, -1)]`
- `cascade_prevent`: two-line fault plus `load_p * 1.15`
- retry loop on `Grid2OpException`

Verification:

- `single_fault` smoke run succeeded
- `n_minus_1` smoke run succeeded
- `cascade_prevent` smoke run succeeded

### Layer 4: reward function

Required by `PROJECT.md`:

- safe-step bonus `+0.1` when all lines are below 80%
- overload penalty `-0.2` per overloaded line
- blackout terminal penalty `-10.0`
- survival bonus `+5.0`
- oscillation penalty `-0.05` when repeating the same action three times
- invalid-action fallback penalty `-0.1` from the later blindspot section

Implemented in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py):

- safe-step bonus
- overload penalty
- terminal penalty on early failure or convergence failure
- survival bonus on reaching `max_steps`
- oscillation penalty via three-entry action history
- invalid action penalty only when invalid input collapses to do-nothing

### Layer 5: deterministic graders

Required by `PROJECT.md`:

- `single_fault`: score based on getting all lines below 90%, scaled by speed
- `n_minus_1`: survival ratio
- `cascade_prevent`: weighted survival, safety, stabilization composite
- deterministic behavior

Implemented in [grid2op_env/server/graders.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/graders.py):

- `grade_single_fault`
- `grade_n_minus_1`
- `grade_cascade_prevent`
- top-level `grade_episode`

Verification:

- direct grader assertions in [grid2op_env/tests/test_grid2op_env.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/tests/test_grid2op_env.py)

### Layer 6: FastAPI server

Required by `PROJECT.md`:

- OpenEnv WebSocket support
- `/tasks`
- `/grader`
- `/baseline`

Implemented in [grid2op_env/server/app.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/app.py):

- `create_app(...)` with `GridEnvironment`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`

Verification:

- FastAPI import check succeeded
- local Docker image built successfully

### Layer 7: client package

Required by `PROJECT.md`:

- `GridEnv`
- `reset()`, `step()`, `state()`
- typed access to `GridAction` and `GridObservation`

Implemented:

- [grid2op_env/client.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/client.py)
- exports in [grid2op_env/__init__.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/__init__.py)

### Layer 8: baseline script

Required by `PROJECT.md`:

- reset -> prompt -> LLM -> parse action -> step -> `/grader`
- invalid JSON should fall back to `do_nothing=True`
- baseline should run all tasks and print scores

Implemented:

- [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py)
- `.env` loading via `load_dotenv()`
- OpenAI-compatible Chat Completions API support
- Qwen-compatible extra parameters
- baseline-side action validation against 20 lines and 6 generators

Verified runtime output:

```json
{
  "model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
  "scores": {
    "single_fault": 0.6111111111111112,
    "n_minus_1": 1.0,
    "cascade_prevent": 0.196667
  },
  "episode_lengths": {
    "single_fault": 10,
    "n_minus_1": 20,
    "cascade_prevent": 3
  }
}
```

This satisfies the spec requirement that the baseline runs without error and produces scores.

## Blindspot handling from PROJECT.md

### 1. LLM hallucination and action validation

Implemented:

- baseline-side validation in [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py)
- server-side revalidation in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- invalid-action fallback to do-nothing
- invalid-action penalty `-0.1` when fallback occurs

### 2. Power flow non-convergence

Implemented:

- catches `Grid2OpException`, `BackendError`, and `lightsim2grid` solver error when available
- returns the last valid observation
- marks `done=True`
- applies terminal penalty via the shaped reward path
- hard task reset retries up to 3 times

### 3. WebSocket state cleanup

OpenEnv handles the WebSocket session lifecycle in its server implementation. This environment uses OpenEnv factory mode with per-session environment instances and proper `close()` cleanup in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py).

### 4. Context window bloat

Implemented in [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py):

- always include lines above 80%
- include disconnected lines
- include current step and max steps
- include task description only on the first step
- include overload-duration warnings when `timestep_overflow > 2`
- omit full low-signal state dumps

## Logging

Timestamped logging was added to the full flow.

Server logs:

- startup
- `/tasks`, `/grader`, `/baseline`
- scenario injection attempts
- reset completion
- each step input and output
- convergence failures
- environment close

Files:

- [grid2op_env/server/logging_utils.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/logging_utils.py)
- [grid2op_env/server/app.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/app.py)
- [grid2op_env/server/tasks.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/tasks.py)
- [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- [grid2op_env/server/graders.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/graders.py)

Baseline logs:

- suite start
- task start
- chosen actions
- model response receipt
- task completion
- grader score receipt
- suite completion

File:

- [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py)

## How to run

### Option 1: local UV workflow

Start the environment server:

```bash
cd /home/sidharth/Desktop/Openenv_modules/grid2op_env
source .venv/bin/activate
env UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-dev server --port 7860
```

Quick checks:

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7860/tasks
```

### Option 2: local Docker workflow

Build:

```bash
cd /home/sidharth/Desktop/Openenv_modules/grid2op_env
docker build -t grid2op-env:local -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 7860:7860 grid2op-env:local
```

This workflow has already been verified. The image successfully built and pre-downloaded `l2rpn_case14_sandbox` at image build time.

## Running the Qwen baseline

Create [grid2op_env/.env](/home/sidharth/Desktop/Openenv_modules/grid2op_env/.env):

```env
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
GRID2OP_BASE_URL=http://127.0.0.1:7860
```

Run the baseline:

```bash
cd /home/sidharth/Desktop/Openenv_modules/grid2op_env
source .venv/bin/activate
env UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-dev inference.py
```

## Testing

Focused test suite:

```bash
cd /home/sidharth/Desktop/Openenv_modules/grid2op_env
source .venv/bin/activate
env UV_CACHE_DIR=/tmp/uv-cache uv run --active --extra dev pytest tests/test_grid2op_env.py -q
```

Verified result:

```text
3 passed, 1 warning in 3.62s
```

## Notes

- `Numba cannot be loaded` is currently only a performance warning from Grid2Op, not a correctness issue.
- `grid2op_env` is constrained to Python `<3.13` in [grid2op_env/pyproject.toml](/home/sidharth/Desktop/Openenv_modules/grid2op_env/pyproject.toml).
- the working baseline and environment runtime were verified using Python 3.12.
