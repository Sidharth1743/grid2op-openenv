# Grid2Op Environment Implementation

This document describes the implemented `grid2op_env` package, verifies it against `PROJECT.md`, and provides the exact run procedures for local development, local Docker, and baseline evaluation. It reflects the current server-side simulation architecture.

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

### Recent fixes

1. **Benchmark ranges corrected** ([tasks.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/tasks.py) lines 248-255):
   - `single_fault_easy`: 0.82-0.85 (was mathematically impossible 0.90-0.94)
   - `single_fault_moderate`: 0.86-0.89 (was 0.94-0.97)
   - `single_fault_severe`: 0.90-0.93 (was 0.96-0.99)

2. **Redispatch penalty added** ([grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py) line 58):
   - `SINGLE_FAULT_REDISPATCH_PENALTY_PER_MW = 0.01` per MW to discourage large interventions

3. **Survival-focused grading** ([graders.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/graders.py)):
   - 70% weight on survival ratio + bonuses

4. **Task 2 (n_minus_1) redesign** - Based on RL2Grid paper and L2RPN winning agents:
   - Three-component reward: `R = 0.3·R_survive + 0.6·R_overload + 0.1·R_cost`
   - `R_survive`: +1.0 per step (constant survival signal)
   - `R_overload`: `(1/n) × Σ clip(1-ρ, -1, 1)` (loading margin)
   - `R_cost`: -0.05 × Σ|ΔMW|/ramp (redispatch cost)
   - Reconnection bonus: +2.0 when reconnecting faulted line safely
   - Terminal: +10×(s/m)² quadratic survival, -15 blackout
   - Phase-aware grader: 30% emergency + 50% security + 20% reconnection
   - Two thresholds: EMERGENCY ≥0.92, WARNING 0.80-0.92, SAFE <0.80
   - N-1 security score (bridge line analysis) in prompt
   - **Grading fix**: Score = survival_ratio × mastery_score (no legacy override)
   - Latest eval: score=0.952 (honest grading, was 1.0 with override)

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
- live-session planning interfaces:
  - `PlanningContextRequest`
  - `PlanningContextResponse`
  - `SimulationRequest`
  - `SimulationResponse`

Verification:

- direct test coverage in [grid2op_env/tests/test_grid2op_env.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/tests/test_grid2op_env.py)
- current focused pytest run: `11 passed`

### Layer 3: task and scenario system

Required by `PROJECT.md`:

- `single_fault`, `n_minus_1`, `cascade_prevent`, `multi_stage_cascade`
- `single_fault` should warm up to a 90-95% style high-loading state
- `n_minus_1` should reset with one disconnected line
- `cascade_prevent` should reset with two disconnected lines and 15% load increase
- `multi_stage_cascade` should reset with three disconnected lines and 15% load increase, with three explicit stages (10 steps each)
- reset-time convergence retry logic for hard cases

Implemented:

- task metadata and reset injection in [grid2op_env/server/tasks.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/tasks.py)
- `single_fault`: warmup search until high line loading is found
- `n_minus_1`: `set_line_status=[(0, -1)]`
- `cascade_prevent`: two-line fault plus `load_p * 1.15`
- `multi_stage_cascade`: three-line fault plus `load_p * 1.15`, with overflow_window=2 and do-nothing probe (5 steps)
- retry loop on `Grid2OpException`

Verification:

- `single_fault` smoke run succeeded
- `n_minus_1` smoke run succeeded
- `cascade_prevent` smoke run succeeded
- `multi_stage_cascade` smoke run succeeded (3 lines + 15% load)

### Layer 4: reward function

Required by the current spec:

- `single_fault`
  - speed bonus `1 / step_number` when fixed
  - safe margin bonus `0.05 * (1 - rho_max)`
  - overload penalty `-0.2` per overloaded line
  - failure penalty `-5.0` if max steps reached without fixing
- `n_minus_1` (redesigned based on RL2Grid paper)
  - **Three-component reward**: `R = 0.3·R_survive + 0.6·R_overload + 0.1·R_cost`
  - `R_survive`: +1.0 per step (constant survival signal)
  - `R_overload`: `(1/n) × Σ clip(1 - ρ, -1, 1)` for each line
  - `R_cost`: -0.05 × Σ|ΔMW|/max_ramp (normalized redispatch cost)
  - Reconnection bonus: +2.0 when reconnecting faulted line safely
  - Terminal: +10.0 × (steps_survived/max_steps)², -15.0 blackout penalty
- `cascade_prevent`
  - cascade prevention bonus `+0.2`
  - overflow duration penalty `-0.1 * timestep_overflow`
  - topology efficiency bonus `+0.05` when `topology_change_count == 0`
  - cascade event penalty `-2.0`
  - blackout penalty `-10.0`
  - survival bonus `+5.0`
- `multi_stage_cascade` (Task 4)
  - Generation cost penalty: `-0.02 × (total_gen / initial_load)`
  - Convergence reward: `+0.5 × available_island_ratio` (every step)
  - Load loss penalty: `-5.0 × (1 - available_load_ratio)` (only at stage boundaries step 10, 20)
  - Terminal win: `+8.0 × (available_load_ratio)²` (only if >=50% load at step 30)
  - Terminal blackout: `-12.0` (early termination or convergence failure)

Implemented in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py).

### Layer 5: deterministic graders

Required by `PROJECT.md`:

- `single_fault`: score based on getting all lines below 90%, scaled by speed
- `n_minus_1`: phase-aware three-component (emergency 30% + security 50% + reconnection 20%)
- `cascade_prevent`: weighted survival, safety, stabilization composite
- `multi_stage_cascade`: stage completion (30%) + load preservation (40%) + island quality (20%) + speed bonus (10%)
- deterministic behavior

Implemented in [grid2op_env/server/graders.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/graders.py):

- `grade_single_fault`
- `grade_n_minus_1` (phase-aware: emergency response + sustained security + reconnection)
- `grade_cascade_prevent`
- `grade_multi_stage_cascade` (four-component grading)
- top-level `grade_episode`

Verification:

- direct grader assertions in [grid2op_env/tests/test_grid2op_env.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/tests/test_grid2op_env.py)

### Layer 6: FastAPI server

Required by `PROJECT.md`:

- OpenEnv WebSocket support
- `/tasks`
- `/grader`
- `/baseline`
- live simulation support for the planner

Implemented in [grid2op_env/server/app.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/app.py):

- `create_app(...)` with `GridEnvironment`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `POST /planning_context`
- `POST /simulate`

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
- baseline-side action validation against live `n_line`, `n_gen`, and redispatchable generators
- server-side planning context lookup by `episode_id`
- server-side candidate simulation on the active live session

Current planner flow:

1. `reset(task_id, seed, difficulty_level)`
2. `state()` to get `episode_id`
3. `planning_context(episode_id)`
4. LLM proposes 3 actions
5. `simulate_candidates(episode_id, actions)`
6. LLM selects final action
7. `step(action)`
8. `/grader`

Previously, `inference.py` used a local replay mirror. That path was removed from the active planning loop because it caused state drift on `n_minus_1` and `cascade_prevent`.

Latest verified runtime output:

```json
{
  "model": "cyankiwi/Qwen3.5-9B-AWQ-4bit",
  "scores": {
    "single_fault": 0.752,
    "n_minus_1": 0.952,
    "cascade_prevent": 0.798,
    "multi_stage_cascade": 0.929
  },
  "episode_lengths": {
    "single_fault": 3,
    "n_minus_1": 20,
    "cascade_prevent": 30,
    "multi_stage_cascade": 30
  }
}
```

This satisfies the spec requirement that the baseline runs without error and produces scores. It also verifies that the active planner path now uses server-side simulation rather than a replayed local sandbox.

Important caveat:

- the latest `cascade_prevent = 1.0` run only covered early curriculum episodes in the 5-seed sweep
- `single_fault` still sometimes falls back to easier stable warmup states outside the intended target range

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
env UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-dev server --port 8000
```

Quick checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
```

### Option 2: local Docker workflow

Build:

```bash
cd /home/sidharth/Desktop/Openenv_modules/grid2op_env
docker build -t grid2op-env:local -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 grid2op-env:local
```

This workflow has already been verified. The image successfully built and pre-downloaded `l2rpn_case14_sandbox` at image build time.

## Running the Qwen baseline

Create [grid2op_env/.env](/home/sidharth/Desktop/Openenv_modules/grid2op_env/.env):

```env
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
GRID2OP_BASE_URL=http://127.0.0.1:8000
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

## Architecture Documentation

- [architecture/task_1_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_1_architecture.md) - Task 1 (single_fault) detailed walkthrough
- [architecture/task_2_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_2_architecture.md) - Task 2 (n_minus_1) N-1 contingency management
- [architecture/task_3_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_3_architecture.md) - Task 3 (cascade_prevent) cascade prevention
- [architecture/task_4_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_4_architecture.md) - Task 4 (multi_stage_cascade) multi-stage cascade management
- [architecture/architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/architecture.md) - Overall system architecture
