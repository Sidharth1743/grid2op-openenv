# Grid2Op Environment

Standalone OpenEnv environment package for the full `PROJECT.md` design.

The current planner uses server-side simulation on the live Grid2Op session. It does not rely on a replayed local mirror.

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
- Server-side planner support via:
  - `POST /planning_context`
  - `POST /simulate`
- Qwen3.5 baseline using the Chat Completions API
- Local Docker workflow with dataset pre-download

## Recent fixes

1. **Benchmark ranges corrected** (tasks.py lines 248-255):
   - `single_fault_easy`: 0.82-0.85 (was mathematically impossible 0.90-0.94)
   - `single_fault_moderate`: 0.86-0.89 (was 0.94-0.97)
   - `single_fault_severe`: 0.90-0.93 (was 0.96-0.99)

2. **Redispatch penalty added** (grid_environment.py line 58):
   - `SINGLE_FAULT_REDISPATCH_PENALTY_PER_MW = 0.01` per MW to discourage large interventions

3. **Survival-focused grading** (graders.py):
   - 70% weight on survival ratio + bonuses

4. **Task 2 (n_minus_1) redesign** based on RL2Grid paper:
   - Three-component reward: 0.3×R_survive + 0.6×R_overload + 0.1×R_cost
   - Reconnection bonus: +2.0 when safely reconnecting faulted line
   - Terminal: +10×(s/m)² quadratic survival, -15 blackout
   - Phase-aware grader: 30% emergency + 50% security + 20% reconnection
   - N-1 security score (bridge lines) in prompt
   - **Grading now honest**: score = survival_ratio × mastery_score (no override)
   - Latest eval: 0.952 (was 1.0 with old override)

## Planner architecture

`inference.py` now uses this flow:

1. `reset()` live episode
2. `state()` to obtain `episode_id`
3. `planning_context(episode_id)` for graph intelligence and redispatchable generators
4. LLM proposes 3 candidate actions
5. `simulate_candidates(episode_id, actions)` on the live server session
6. LLM selects the safest simulated action
7. `step(action)`

This avoids the old replay-mirror drift problem.

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

## Important runtime note

After changing server code, restart the Grid2Op server before running `inference.py`. The planner depends on the live server routes `/planning_context` and `/simulate`.

## Latest verified result

Latest saved run:

- `single_fault`: `0.752`
- `n_minus_1`: `1.0`
- `cascade_prevent`: `1.0`

This confirms the server-side simulation path is active. One benchmark caveat remains: the latest `cascade_prevent` score was achieved on the easier early curriculum slice of the 5-seed run, not yet on the hardest late curriculum stages.

## Architecture Documentation

- [architecture/task_1_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_1_architecture.md) - Task 1 detailed walkthrough
- [architecture/task_2_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_2_architecture.md) - Task 2 N-1 contingency management
- [architecture/architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/architecture.md) - Overall system architecture
