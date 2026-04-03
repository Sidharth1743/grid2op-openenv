# Grid2Op Environment Architecture

## Overview

`grid2op_env` is an OpenEnv-compatible power-grid environment built on Grid2Op. The environment server owns the live Grid2Op session, and the LLM planner interacts with it through a typed client. Candidate evaluation is now done on the live server session, not with a reconstructed local mirror.

## Current Architecture

```text
LLM Planner
  |
  | proposal prompt / final selection prompt
  v
grid2op_env/inference.py
  |
  | GridEnv client
  | - reset()
  | - step()
  | - state()
  | - planning_context(episode_id)
  | - simulate_candidates(episode_id, actions)
  v
FastAPI / OpenEnv server
  |
  | GridEnvironment session bound to episode_id
  | - reset()
  | - step()
  | - state
  | - get_planning_context()
  | - simulate_actions()
  v
Grid2Op live environment
  |
  | l2rpn_case14_sandbox
  v
power-flow physics
```

## Why the Architecture Changed

The earlier planner used a local replayed sandbox in `inference.py`. That caused state drift for `n_minus_1` and `cascade_prevent`, because local replay did not always reconstruct the exact remote chronic and live state. The current design removes that failure mode by making the server session the single source of truth for:

- graph intelligence
- redispatchable generator metadata
- candidate simulation

## Main Components

### 1. Environment Server

Implemented in [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py).

Responsibilities:

- owns the live Grid2Op environment instance
- injects reset scenarios for `single_fault`, `n_minus_1`, and `cascade_prevent`
- shapes rewards and stores `episode_log`
- tracks the active raw Grid2Op observation
- registers active sessions by `episode_id`
- exposes:
  - `get_planning_context()`
  - `simulate_actions(actions)`

### 2. FastAPI Layer

Implemented in [grid2op_env/server/app.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/app.py).

Routes:

- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `POST /planning_context`
- `POST /simulate`
- `WS /ws`

`/planning_context` and `/simulate` resolve the exact active environment instance by `episode_id`, so simulation runs on the same live session the agent is controlling.

### 3. Client Layer

Implemented in [grid2op_env/client.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/client.py).

Core methods:

- `reset(...)`
- `step(action)`
- `state()`
- `planning_context(episode_id)`
- `simulate_candidates(episode_id, actions)`

### 4. Planner

Implemented in [grid2op_env/inference.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/inference.py).

Loop:

1. `reset()` live episode
2. fetch `planning_context(episode_id)`
3. ask LLM for 3 candidates
4. send candidates to `/simulate` on the same live session
5. ask LLM to choose among simulated outcomes
6. execute chosen action with `step()`
7. send `episode_log` to `/grader`

This is the current `think -> simulate -> act` implementation.

### 5. Graph Analysis

Implemented in [grid2op_env/graph_analysis.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/graph_analysis.py).

Uses:

- `obs.get_energy_graph()`
- `obs.flow_bus_matrix(active_flow=True)`
- `networkx`

Produces:

- `bridge_lines`
- `safe_to_disconnect`
- `parallel_groups`
- `high_centrality_buses`
- `islanded_clusters`
- `congestion_corridor`
- `flow_clusters`
- `stressed_lines`

This graph intelligence is computed on the live server observation and returned through `/planning_context`.

## Task System

Implemented in [grid2op_env/server/tasks.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/tasks.py).

Tasks:

- `single_fault` - See [task_1_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_1_architecture.md) for detailed walkthrough
- `n_minus_1` - See [task_2_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_2_architecture.md) for N-1 contingency management
- `cascade_prevent` - See [task_3_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_3_architecture.md) for cascade prevention
- `multi_stage_cascade` - See [task_4_architecture.md](/home/sidharth/Desktop/Openenv_modules/architecture/task_4_architecture.md) for multi-stage cascade management

Curriculum:

- `single_fault`
  - episodes `1-3`: mild
  - episodes `4-6`: moderate
  - episodes `7+`: severe
- `cascade_prevent`
  - episodes `1-3`: one line, `+5%`
  - episodes `4-6`: one line, `+10%`
  - episodes `7-9`: two lines, `+10%`
  - episodes `10+`: two lines, `+15%`
- `multi_stage_cascade` (Task 4)
  - 3 lines disconnected at reset
  - +15% load increase
  - Three stages: 10 steps each
  - Overflow window: 2 (faster cascades)
  - Do-nothing probe: must survive 5 steps

## Reward and Grading

Implemented in:

- [grid2op_env/server/grid_environment.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/grid_environment.py)
- [grid2op_env/server/graders.py](/home/sidharth/Desktop/Openenv_modules/grid2op_env/server/graders.py)

Reward shaping is task-specific. Grading is deterministic.

## Verified Status

Current verified properties:

- server-side simulation path is implemented
- local replay mirror is no longer used by the active planner
- `single_fault`, `n_minus_1`, `cascade_prevent`, and `multi_stage_cascade` all run through the same live-session simulation path
- unit tests pass for the server-side simulation and planning context path
- benchmark ranges corrected to be mathematically achievable (tasks.py lines 248-255)
- redispatch penalty implemented (grid_environment.py line 58)
- survival-focused grading implemented (graders.py)
- **Task 2 (n_minus_1) fully redesigned** per RL2Grid paper:
  - three-component reward (0.3×survive + 0.6×overload + 0.1×cost)
  - reconnection bonus (+2.0) implemented
  - phase-aware grader (emergency 30% + security 50% + reconnection 20%)
  - N-1 security score in prompt (bridge line analysis)
  - two-threshold framing (EMERGENCY/WARNING/SAFE)
  - **Grading fix**: score = survival_ratio × mastery_score (no legacy override)
  - latest eval: score=0.952 (honest grading, was 1.0 with override)
- **Task 4 (multi_stage_cascade) added**:
  - 3 lines disconnected at reset
  - +15% load increase
  - Three-stage cascade structure (10 steps each)
  - Candidate filtering prevents grid collapse actions
  - Island availability assessment at stage boundaries
  - Latest eval: score=0.929 (31x improvement from 0.027 with candidate filtering)

Current benchmark caveat:

- the latest `cascade_prevent = 1.0` result is valid for the evaluated curriculum slice, but that slice only covered the easier early curriculum stages in the 5-seed run
- `single_fault` still has some fallback states that are easier than the nominal target range
