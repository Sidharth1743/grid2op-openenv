---
title: Grid2Op Environment
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

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
    ├── environment.py
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
- Four tasks: `single_fault`, `n_minus_1`, `cascade_prevent`, `multi_stage_cascade`
- Reset-time scenario injection and retry logic for non-convergent starts
- Shaped reward, episode logging, and deterministic graders
- OpenEnv WebSocket interface plus `/tasks`, `/grader`, and `/baseline`
- Server-side planner support via:
  - `POST /planning_context`
  - `POST /simulate`
- Qwen3.5 baseline using the Chat Completions API
- Local Docker workflow with dataset pre-download

## Recent fixes

1. **Task 1 (single_fault) benchmark ranges corrected** (tasks.py:297-304):
   - `single_fault_easy`: 0.82-0.85 (was mathematically impossible 0.90-0.94)
   - `single_fault_moderate`: 0.86-0.89 (was 0.94-0.97)
   - `single_fault_severe`: 0.90-0.93 (was 0.96-0.99)
   - Warmup phase finds high-loading state in chronics, then agent has 10 steps to solve

2. **Task 1 reward function** (grid_environment.py:589-596):
   - Target achieved bonus: `1.0 / step_count` (rewards early solution)
   - Safe margin bonus: `0.05 × max(0.0, 1.0 - max_rho)`
   - Overload penalty: `0.2 × overloaded_count` (lines > 100%)
   - Redispatch penalty: `0.01 × MW` (discourages large interventions)
   - Failure penalty: `-5.0` if time limit reached without target

3. **Task 1 grading** (graders.py:28-55):
   - 70% weight on survival ratio
   - 50% target achieved bonus
   - Final state bonus (0.3 if below target, 0.15/+0.05, 0.05/+0.10)
   - Legacy success score for early completion: `1.0 - 0.08 × (step - 1)`

4. **Task 2 (n_minus_1) redesign** based on RL2Grid paper (grid_environment.py:598-609):
   - Three-component reward: `0.3×R_survive + 0.6×R_overload + 0.1×R_cost`
   - `R_survive`: +1.0 per step (constant survival signal)
   - `R_overload`: `(1/n) × Σ clip(1-ρ, -1, 1)` - loading margin quality
   - `R_cost`: `-0.05 × Σ|ΔMW|/max_ramp` (normalized redispatch cost)
   - Reconnection bonus: +2.0 when safely reconnecting (grid_environment.py:853-869)
   - Terminal: +10×(s/m)² quadratic survival, -15 blackout
   - Phase-aware grader (graders.py:58-83):
     - Emergency response (30%): cleared within 5 steps at rho < 0.92
     - Sustained security (50%): steps 6-20 at rho < 0.90
     - Reconnection (20%): did agent reconnect line 0?
   - N-1 security score (bridge lines) in prompt
   - **Grading now honest**: score = survival_ratio × mastery_score (no override)
   - Latest eval: 0.952 (was 1.0 with old override)

5. **Task 3 (cascade_prevent)** (grid_environment.py:611-628):
   - 1-2 lines disconnected at reset + 5-15% load increase
   - Key metric: `timestep_overflow` countdowns (not just max_rho)
   - Quadratic overflow penalty: `-0.05 × Σ(overflow²)` - line at overflow=2 is 4x more urgent than overflow=1
   - Reward components:
     - Cascade prevention: +0.3 if no auto-trip, -2.5 if auto-trip
     - Thermal margin: +0.1 × mean(clip(1-ρ, -1, 1))
     - Terminal: +5.0 × (1 - auto_trips/5)² survival bonus, -12.0 blackout
   - Grading (graders.py:86-121):
     - Cascade containment (50%): steps without auto-trips / 30
     - Thermal stability (30%): safe_steps / containment_steps
     - Recovery speed (20%): how fast recovered from first overload
   - Latest eval: 0.798 (hard/extreme tiers challenging)

6. **Task 4 (multi_stage_cascade)** (tasks.py:334-337, grid_environment.py:630-647):
   - 3 lines disconnected at reset + **20% load increase** (not 15%)
   - Three explicit stages (10 steps each) with stage boundaries at step 10 and 20
   - Overflow window: 2 (faster cascades than default 3)
   - Do-nothing survival probe: 5 steps minimum
   - Island availability assessment at stage boundaries (grid_environment.py:767-814)
   - Candidate filtering (inference.py:1003-1030): filters unsafe topology disconnects
   - Reward (grid_environment.py:630-647):
     - Generation cost: -0.02 × (total_gen / initial_load)
     - Convergence: +0.5 × available_island_ratio
     - Load loss penalty: -5.0 × (1 - available_load_ratio) at boundaries only
     - Terminal win: +8.0 × (available_load_ratio)² if ≥50% load at step 30
     - Terminal blackout: -12.0
   - Grading (graders.py:124-174):
     - Stage completion (30%): survived stages 1, 2, 3
     - Load preservation (40%): available_load_ratio at end
     - Island quality (20%): majority islands viable at boundaries
     - Speed bonus (10%): how fast stability returned each stage
   - Latest eval: 0.929 (31x improvement from 0.027)

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
- `n_minus_1`: `0.952`
- `cascade_prevent`: `0.798`
- `multi_stage_cascade`: `0.929`

This confirms the server-side simulation path is active.

## Architecture Documentation

- [architecture/task_1_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_1_architecture.md) - Task 1 detailed walkthrough
- [architecture/task_2_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_2_architecture.md) - Task 2 N-1 contingency management
- [architecture/task_3_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_3_architecture.md) - Task 3 cascade prevention
- [architecture/task_4_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_4_architecture.md) - Task 4 multi-stage cascade management
- [architecture/architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/architecture.md) - Overall system architecture
