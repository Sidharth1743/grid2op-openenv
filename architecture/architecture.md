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

Implemented in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py).

Responsibilities:

- owns the live Grid2Op environment instance
- injects reset scenarios for all four tasks
- shapes rewards and stores `episode_log`
- tracks the active raw Grid2Op observation
- registers active sessions by `episode_id`
- exposes:
  - `get_planning_context()`
  - `simulate_actions(actions)`

### 2. FastAPI Layer

Implemented in [server/app.py](/home/sidharth/Desktop/grid2op-openenv/server/app.py).

Routes:

- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `POST /planning_context`
- `POST /simulate`
- `WS /ws`

`/planning_context` and `/simulate` resolve the exact active environment instance by `episode_id`, so simulation runs on the same live session the agent is controlling.

### 3. Client Layer

Implemented in [client.py](/home/sidharth/Desktop/grid2op-openenv/client.py).

Core methods:

- `reset(...)`
- `step(action)`
- `state()`
- `planning_context(episode_id)`
- `simulate_candidates(episode_id, actions)`

### 4. Planner

Implemented in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py).

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

Implemented in [graph_analysis.py](/home/sidharth/Desktop/grid2op-openenv/graph_analysis.py).

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

Implemented in [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py).

Tasks:

- `single_fault` - See [task_1_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_1_architecture.md) for the detailed walkthrough
- `n_minus_1` - See [task_2_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_2_architecture.md) for N-1 contingency management
- `cascade_prevent` - See [task_3_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_3_architecture.md) for cascade prevention
- `multi_stage_cascade` - See [task_4_architecture.md](/home/sidharth/Desktop/grid2op-openenv/architecture/task_4_architecture.md) for multi-stage cascade management

Curriculum:

- `single_fault`
  - episodes `1-3`: mild
  - episodes `4-6`: moderate
  - episodes `7+`: severe
  - reset searches one chronic until `max_rho` falls inside the target band
  - benchmark tiers use calibrated achievable ranges (`0.82-0.93`)
  - fallback uses the closest stable state with `max_rho >= 0.80`
- `cascade_prevent`
  - episodes `1-3`: one line, `+5%`
  - episodes `4-6`: one line, `+10%`
  - episodes `7-9`: two lines, `+10%`
  - episodes `10+`: two lines, `+15%`
- `multi_stage_cascade` (Task 4)
  - 3 lines disconnected at reset
  - +20% load increase
  - Three stages: 10 steps each
  - Overflow window: 2 (faster cascades)
  - Do-nothing probe: must survive 5 steps

## Reward and Grading

Implemented in:

- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py)

Reward shaping is task-specific. Grading is deterministic.

## Verified Status

Current verified properties:

- server-side simulation path is implemented
- local replay mirror is no longer used by the active planner
- `single_fault`, `n_minus_1`, `cascade_prevent`, and `multi_stage_cascade` all run through the same live-session simulation path
- unit tests pass for the server-side simulation and planning context path
- benchmark ranges corrected to be mathematically achievable
- redispatch penalty implemented for `single_fault`
- `single_fault` planner now uses redispatch-only control: topology edits are rejected for this task
- `single_fault` severe benchmark uses a `0.90` success threshold; other variants use `0.80`
- survival-focused grading implemented (graders.py)
- **Task 2 (`n_minus_1`) is modeled as a fixed single-line contingency**:
  - line `0` starts disconnected
  - reward uses survival, loading quality, and redispatch cost
  - safe reconnection can earn a `+2.0` bonus
  - grading uses emergency handling, sustained security, and reconnection
  - prompt context includes bridge-line / N-1 security information
  - latest eval: score=0.952
- **Task 3 (`cascade_prevent`) is modeled as an active cascade-prevention problem**:
  - reset disconnects one or two lines and increases load by `5-15%`
  - `timestep_overflow` is the primary urgency signal
  - reward penalizes countdown growth quadratically and punishes auto-trips
  - grading measures containment, thermal stability, and recovery speed
  - latest eval: score=0.798
- **Task 4 (`multi_stage_cascade`) is modeled as staged load-preservation under unavoidable cascade pressure**:
  - reset disconnects three lines and increases load by `20%`
  - only chronics that survive a 5-step do-nothing probe are used
  - the episode is divided into three explicit 10-step stages
  - island viability is assessed from connected components, local load, and local generation capacity
  - planning favors redispatch and survivable island structure over short-term cosmetic improvements
  - latest eval: score=0.929

Current benchmark caveat:

- `cascade_prevent` score can vary sharply between benchmark tiers because the extreme tier starts with two missing lines and active overload countdown pressure
- `single_fault` can still fall back to the closest stable high-loading state when an exact benchmark-band match is unavailable
