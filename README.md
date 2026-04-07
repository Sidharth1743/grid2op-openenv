---
title: Grid2Op Environment
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Grid2Op OpenEnv Environment

This repository provides an **AI-ready power-grid control environment** built on `Grid2Op` and packaged for `OpenEnv`.

It is a standardized testbed for evaluating LLM and RL agents on power-grid operations. A FastAPI server hosts the live `l2rpn_case14_sandbox` simulation, and agents interact through `reset()`, `step()`, `state()`, `/planning_context`, and `/simulate`.

**Key design principle:** the server is the source of truth. Planning is done against the live simulator session instead of a replayed local mirror.

## 📑 Table of Contents

1. [Quick Start](#-quick-start)
2. [Repository Layout](#-repository-layout)
3. [The Evaluation Suite](#-the-evaluation-suite)
4. [Agent Planning Flow](#-agent-planning-flow)
5. [Docker and Deployment](#-docker-and-deployment)
6. [Further Reading](#-further-reading)

## 🚀 Quick Start

**1. Create the virtual environment**

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

**2. Run the server locally**

```bash
ENABLE_WEB_INTERFACE=true uv run server --port 7860
```

The server listens on `http://127.0.0.1:7860`.

**3. Smoke-test the API**

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:7860/tasks
```

**4. Run the baseline agent**

Create a `.env` file in the repo root:

```env
GRID2OP_BASE_URL=http://127.0.0.1:7860
API_BASE_URL=https://router.huggingface.co/v1
HF_TOKEN=hf_your_token
MODEL_NAME=openai/gpt-oss-20b:groq
```

Then run:

```bash
python inference.py --task-id single_fault
```

Optional local checks:

```bash
uv run grid2op-smoke --task-id single_fault --steps 1
uv run --extra dev pytest tests/test_grid2op_env.py -q
```

The browser UI is available at `http://127.0.0.1:7860/web/`.

## 📂 Repository Layout

- [server/app.py](/home/sidharth/Desktop/grid2op-openenv/server/app.py): FastAPI and OpenEnv entrypoint
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py): live Grid2Op adapter, reward shaping, planning support, and episode state
- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py): reset logic, benchmark tiers, and scenario injection
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py): deterministic grading formulas
- [server/gradio_ui.py](/home/sidharth/Desktop/grid2op-openenv/server/gradio_ui.py): custom OpenEnv `/web` UI tab
- [models.py](/home/sidharth/Desktop/grid2op-openenv/models.py): typed actions, observations, state, and logs
- [client.py](/home/sidharth/Desktop/grid2op-openenv/client.py): environment client wrapper
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py): baseline Think-Simulate-Act runner
- [architecture/architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/architecture.md): system-level design

## 🎯 The Evaluation Suite

The environment contains four tasks that scale from overload relief to staged cascade damage control.

### Task 1: `single_fault`

Starts from an intact but highly stressed grid. The server replays a real chronic until it finds a stable timestep where one or more lines are already hot, then gives the agent **10 steps** to cool the system down.

- Reset: warmup search over a real time series until `max_rho` reaches the target band
- Objective: bring all lines below the task threshold, usually `0.80` and `0.90` for the severe benchmark
- Reward: early-fix bonus, safe-margin bonus, overload penalty, and redispatch penalty
- Grader: rewards survival, hitting the threshold, and ending in a strong final state
- Agent behavior: redispatch-first; topology actions are intentionally blocked in the planner
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_1_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_1_architecture.md)
- Grounding: Dwivedi et al. (2024), *RL for Mitigating Cascading Failures: Targeted Exploration via Sensitivity Factors*; Donnot (2020), *Grid2Op*

### Task 2: `n_minus_1`

Starts with line `0` already disconnected. The remaining network must absorb the rerouted flow immediately. The agent has **20 steps** to stabilize the degraded topology and reconnect the missing line safely.

- Reset: fixed N-1 outage injected at step 0
- Objective: clear the emergency, maintain secure operation, and reconnect the line when safe
- Reward: `0.3 * R_survive + 0.6 * R_overload + 0.1 * R_cost`, plus a safe reconnection bonus
- Grader: 30% emergency response, 50% sustained security, 20% reconnection
- Agent behavior: first 5 steps are treated as an emergency window, then the planner shifts toward sustained operation
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_2_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_2_architecture.md)
- Grounding: Marchesini et al. (2025), *RL2Grid*; Yoon et al. (2021), *Winning the L2RPN Challenge*; van der Sar et al. (2025); Marot et al. (2021)

### Task 3: `cascade_prevent`

Starts from a grid already moving toward a cascade. One or two lines are disconnected, load is increased, and several remaining lines may already be near or above their limits. The critical signal is `timestep_overflow`, which tracks how close overloaded lines are to tripping automatically.

- Reset: calibrated line outages plus load scaling from `+5%` to `+15%`
- Objective: prevent overflow countdowns from turning into automatic trips over a 30-step horizon
- Reward: no-trip bonus, auto-trip penalty, quadratic overflow penalty, and thermal-margin shaping
- Grader: 50% cascade containment, 30% thermal stability, 20% recovery speed
- Agent behavior: triage the most urgent countdowns first, then improve overall thermal margin
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_3_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_3_architecture.md)
- Grounding: Marchesini et al. (2025), *RL2Grid*; Dwivedi et al. (2024); Ramapuram Matavalam et al. (2023)

### Task 4: `multi_stage_cascade`

This is the hardest task. It does not ask whether the agent can stop the cascade completely. It asks whether the agent can preserve as much **viable load** as possible while the grid passes through a severe staged failure.

- Reset: three line outages, `+20%` load, overflow window `2`, and a do-nothing viability probe before acceptance
- Objective: preserve viable islands and maximize available load across three explicit 10-step stages
- Reward: generation-cost penalty, island-availability reward, stage-boundary load-loss penalty, and terminal load-preservation bonus
- Grader: 30% stage completion, 40% load preservation, 20% island quality, 10% speed to stability
- Agent behavior: plan across stage boundaries, not just the current overload; survivable islands matter more than one-step `max_rho`
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_4_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_4_architecture.md)
- Grounding: Meng et al. (2025), *Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation*; Zhu (2021)

## 🧠 Agent Planning Flow

The baseline agent in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py) uses a server-backed Think-Simulate-Act loop:

1. `reset()` the live episode and get the current `episode_id`
2. call `state()` to read task metadata and episode status
3. call `/planning_context` to gather graph intelligence and redispatch bounds
4. ask the LLM for candidate actions
5. call `/simulate` on the live server session to test those candidates
6. choose the safest verified action
7. execute `step(action)`
8. score the finished episode through `/grader`

This avoids replay drift and keeps the planner tied to the live simulator state.

## 🐳 Docker and Deployment

Build locally:

```bash
docker build -t grid2op-env:local .
docker run --rm -p 7860:7860 grid2op-env:local
```

The same container shape is used for Hugging Face Spaces. Deployment metadata is defined in the frontmatter above, and the runtime manifest is defined in [openenv.yaml](/home/sidharth/Desktop/grid2op-openenv/openenv.yaml).

## 📚 Further Reading

- [architecture/architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/architecture.md)
- [architecture/task_1_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_1_architecture.md)
- [architecture/task_2_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_2_architecture.md)
- [architecture/task_3_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_3_architecture.md)
- [architecture/task_4_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_4_architecture.md)
