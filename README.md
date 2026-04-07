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

This repository exposes a **user-driven power-grid control environment** built on `Grid2Op` and packaged for `OpenEnv`.

It is designed for agent evaluation, prompt iteration, and hackathon-style deployment:
- a FastAPI server owns the live `l2rpn_case14_sandbox` simulation
- agents interact through `reset()`, `step()`, `state()`, `/planning_context`, and `/simulate`
- four tasks cover overload relief, N-1 contingency handling, cascade prevention, and staged cascade damage control

The key design choice is simple: **the server is the source of truth**. Planning uses simulator-backed rollouts on the live episode instead of reconstructing a local mirror.

## Quick Start with `uv`

### 1. Create the environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Run the server locally

```bash
uv run server --port 7860
```

The server listens on `http://127.0.0.1:7860`.

### 3. Smoke-test the API

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{}'
curl http://127.0.0.1:7860/tasks
```

### 4. Run the baseline agent

Create a `.env` file:

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

## Repository Layout

- [server/app.py](/home/sidharth/Desktop/grid2op-openenv/server/app.py): FastAPI/OpenEnv entrypoint
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py): live Grid2Op adapter, rewards, planner support
- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py): task reset logic and scenario injection
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py): deterministic episode grading
- [models.py](/home/sidharth/Desktop/grid2op-openenv/models.py): typed actions, observations, state, logs
- [client.py](/home/sidharth/Desktop/grid2op-openenv/client.py): environment client
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py): baseline LLM runner
- [architecture/architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/architecture.md): system-level architecture

## Tasks

### Task 1: `single_fault`

This task starts from an **intact but highly stressed grid**. The server replays a real chronic until it finds a stable timestep where one or more lines are already running hot, then gives the agent **10 steps** to cool the grid down. In practice, this is the cleanest overload-relief task in the suite.

- Reset: warmup search on a real time series until `max_rho` reaches the target band.
- Objective: bring all lines below the task threshold, usually `0.80` and `0.90` for the severe benchmark.
- Reward: early-fix bonus, safe-margin bonus, overload penalty, and redispatch penalty.
- Grader: rewards surviving the horizon, achieving the threshold, and ending in a good final state.
- Agent behavior: redispatch-first; topology edits are intentionally blocked in the planner.
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_1_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_1_architecture.md)
- Research grounding:
  - Primary: Dwivedi, A., Tajer, A., Paternain, S., & Virani, N. (2024). *RL for Mitigating Cascading Failures: Targeted Exploration via Sensitivity Factors*. NeurIPS 2024 Workshop, arXiv:2411.18050. This is the direct source for the `(1 - rho^2)` family of overload-relief shaping, sensitivity-guided control ideas, and switching-cost framing that inspired the simpler Task 1 control philosophy.
  - Supporting: Donnot, B. (2020). *Grid2Op: A Testbed Platform to Model Sequential Decision Making in Power Systems*. GitHub. <https://github.com/rte-france/grid2op>

### Task 2: `n_minus_1`

This task starts with **line 0 already disconnected**. The grid is still operating, but one important corridor is gone, so the remaining lines must absorb the rerouted flow. The agent has **20 steps** to stabilize the degraded topology and reconnect the missing line when it becomes safe.

- Reset: fixed N-1 outage with `line 0` disconnected at step 0.
- Objective: clear the emergency, maintain secure operation, and safely reconnect the line.
- Reward: `0.3 * R_survive + 0.6 * R_overload + 0.1 * R_cost`, plus a safe reconnection bonus.
- Grader: 30% emergency response, 50% sustained security, 20% reconnection.
- Agent behavior: first 5 steps are treated as an emergency window; later steps favor safe operation and reconnection.
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_2_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_2_architecture.md)
- Research grounding:
  - Primary: Marchesini, E., Marzari, L., & Leofante, F. (2025). *RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations*. arXiv:2503.23101. This is the direct source for the three-part `R_survive + R_overload + R_cost` reward structure.
  - Supporting: Yoon, D., Hong, S., Lee, B.-J., & Kim, K.-E. (2021). *Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic*. ICLR 2021. This motivates the activation-threshold idea used in the Task 2 planner.
  - Supporting: van der Sar, E. et al. (2025). *Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control*. ACM e-Energy 2025, arXiv:2502.08681. This informs the structural N-1 security framing and overload-margin component.
  - Supporting: Marot, A., Donnot, B. et al. (2021). *Learning to Run a Power Network Challenge: A Retrospective Analysis*. NeurIPS 2021. This provides the broader L2RPN context for reconnection and post-contingency operation.

### Task 3: `cascade_prevent`

This task starts from a grid that is **already moving toward a cascade**. One or two lines are disconnected, load is increased, and some remaining lines are near or above their limits. The key signal is not just `max_rho`; it is `timestep_overflow`, which tells you how long each line has been overloaded and how close it is to tripping automatically.

- Reset: calibrated line outages plus load increase from `+5%` to `+15%` depending on tier.
- Objective: stop overload countdowns from turning into automatic trips and finish the 30-step horizon.
- Reward: no-trip bonus, auto-trip penalty, quadratic overflow penalty, thermal-margin term, and terminal survival logic.
- Grader: 50% cascade containment, 30% thermal stability, 20% recovery speed.
- Agent behavior: triage the most urgent countdown first, then improve overall thermal margin.
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_3_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_3_architecture.md)
- Research grounding:
  - Primary: Marchesini, E., Marzari, L., & Leofante, F. (2025). *RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations*. arXiv:2503.23101. This is the basis for the thermal-margin shaping term adapted here.
  - Supporting: Dwivedi, A., Tajer, A., Paternain, S., & Virani, N. (2024). *RL for Mitigating Cascading Failures: Targeted Exploration via Sensitivity Factors*. NeurIPS 2024 Workshop, arXiv:2411.18050. This motivates treating overload timing as a cascade-control signal and using stronger penalties near failure.
  - Supporting: Ramapuram Matavalam, A. R., Guddanti, K. P., Weng, Y., & Ajjarapu, V. (2023). *Curriculum Based Reinforcement Learning of Grid Topology Controllers to Prevent Thermal Cascading*. IEEE Transactions on Power Systems, 38(5), 4206-4220. This supports the task’s curriculum-style difficulty progression and thermal-cascade framing.

### Task 4: `multi_stage_cascade`

This is the hardest task. It does not ask, “Can you stop the cascade entirely?” It asks, **“If the grid is already headed into a severe staged failure, how much viable load can you keep alive?”** Reset disconnects three lines, adds `20%` load, shortens the overflow window to `2`, and only accepts scenarios that survive a 5-step do-nothing probe. The episode is then evaluated over **three explicit 10-step stages**.

- Reset: three fixed line outages, `+20%` load, overflow window `2`, viability probe before acceptance.
- Objective: preserve viable islands and as much load as possible across stage boundaries.
- Reward: generation-cost penalty, island-availability reward, stage-boundary load-loss penalty, and terminal load-preservation bonus.
- Grader: 30% stage completion, 40% load preservation, 20% island quality, 10% speed to stability.
- Agent behavior: think across stage boundaries, not just the current overload; preserving survivable islands matters more than chasing one-step `max_rho`.
- Relevant files: [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py), [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py), [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py), [architecture/task_4_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_4_architecture.md)
- Research grounding:
  - Primary: Meng, B., Xu, C., & Zhu, Y. (2025). *Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation*. ICLR 2025, arXiv:2505.09012. This is the direct source for the multi-stage cascading-failure formulation, island-availability rule, and four-part reward structure.
  - Supporting: Zhu, Y. (2021). *Power Grid Cascading Failure Mitigation by Reinforcement Learning*. arXiv:2108.10424. This provides the earlier cascading-failure MDP and corrective-control foundation extended by Task 4.

## Planning Flow

The baseline agent in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py) uses this loop:

1. `reset()` the live episode
2. `state()` to get `episode_id`
3. `planning_context(episode_id)` for graph and action context
4. ask the LLM for candidate actions
5. `simulate` those candidates on the live server session
6. choose the safest simulated action
7. `step(action)`
8. score the episode with `/grader`

This avoids replay drift and makes the baseline easier to debug.

## Docker and Deployment

Build locally:

```bash
docker build -t grid2op-env:local .
docker run --rm -p 7860:7860 grid2op-env:local
```

The same image shape is used for Hugging Face Spaces. The deployment metadata is defined in the frontmatter above and the runtime manifest in [openenv.yaml](/home/sidharth/Desktop/grid2op-openenv/openenv.yaml).

## Further Reading

- [architecture/architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/architecture.md)
- [architecture/task_1_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_1_architecture.md)
- [architecture/task_2_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_2_architecture.md)
- [architecture/task_3_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_3_architecture.md)
- [architecture/task_4_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_4_architecture.md)
