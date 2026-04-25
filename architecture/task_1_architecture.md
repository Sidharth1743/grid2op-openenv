# Task 1: `single_fault`

## Overview

`single_fault` is the simplest task operationally, but it is the cleanest way to understand how this environment works.

The grid is **not** reset into a blackout and **not** reset with a line already disconnected. Instead, the server replays one real Grid2Op time series until it finds a timestep where the grid is intact but one or more lines are already running hot. The agent then has **10 steps** to bring the whole grid back under control.

In plain language:

- the network is still connected
- the physics are still stable
- one line is too close to its thermal limit
- the agent must cool the grid down quickly, mainly by redispatching generation

This task teaches “relieve overload before it becomes a failure.”

## 1. How Reset Works

Task 1 uses a **warmup search**, not a direct fault injection.

The reset logic in [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py) does this:

1. pick a deterministic time series id
2. reset the Grid2Op environment on that time series
3. step forward with do-nothing actions
4. measure `max_rho = max(obs.rho)` at each warmup step
5. stop when `max_rho` enters the target range for the requested difficulty

If the exact target range is never found, the server keeps the **closest stable** state it saw and replays that warmup exactly so the backend and the returned observation stay synchronized.

### Target ranges

There are two modes:

**Curriculum mode**

- episodes `1-3`: `0.90-0.94`
- episodes `4-6`: `0.94-0.97`
- episodes `7+`: `0.96-0.99`

These are intentionally aggressive and are mostly for internal progression logic.

**Benchmark mode**

- `single_fault_easy`: `0.82-0.85`
- `single_fault_moderate`: `0.86-0.89`
- `single_fault_severe`: `0.90-0.93`

These are the calibrated public benchmark bands.

### Returned metadata

Reset stores metadata such as:

- `time_series_id`
- `warmup_steps`
- `target_rho_range`
- `target_matched`
- `stable_fallback_used` when applicable
- `scenario = "high_loading"` or `"high_loading_closest_stable_match"`

That metadata matters because the same warmup can be replayed later if needed.

## 2. What Counts as Success

Task 1 does **not** use the same threshold in every variant.

The success threshold is computed in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py):

- default threshold: `0.80`
- severe benchmark threshold: `0.90`

This means:

- easy benchmark, moderate benchmark, and normal curriculum runs are solved when **all lines are below `0.80`**
- severe benchmark is solved when **all lines are below `0.90`**

The episode ends early as soon as the threshold is satisfied.

## 3. What the Agent Can Do

Although the environment itself supports line switching and redispatch, the Task 1 planner intentionally treats this as a **redispatch-first control problem**.

In [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py):

- `disconnect_line` and `reconnect_line` proposals are rejected for `single_fault`
- the prompt tells the model to use only `redispatch` or `do_nothing`
- after simulation, the best improving candidate is selected directly
- there is no second LLM “final choice” call for this task

This is an important design choice. Task 1 is supposed to be the cleanest overload-relief problem, not a topology-exploration problem.

## 4. What the Agent Sees

The observation includes:

- `rho`: per-line loading ratio
- `gen_p`: generator outputs
- `load_p`: load values
- `line_status`
- `timestep_overflow`
- `sensitivity_guidance`

For Task 1, `sensitivity_guidance` is especially useful. The server simulates small redispatch moves and returns the ones that reduce the current global `max_rho`.

Typical guidance item:

```json
{
  "action_type": "redispatch",
  "target_id": 5,
  "delta_mw": -15.0,
  "expected_rho_change": -0.0214
}
```

That means “moving generator 5 by `-15 MW` is expected to reduce the current worst line loading by about `0.0214`.”

## 5. Reward Function

Task 1 uses a simple shaped reward in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py):

### A. Early-fix bonus

If the agent gets every line below the success threshold, it earns:

```text
1.0 / step_count
```

So solving at step 1 is better than solving at step 8.

### B. Safe-margin bonus

Each step also gets:

```text
0.05 × max(0, 1 - max_rho)
```

This encourages lower stress even before the exact success condition is met.

### C. Overload penalty

For each line above `1.0`, the reward loses:

```text
0.2 × overloaded_line_count
```

### D. Redispatch penalty

Task 1 discourages oversized generator moves:

```text
0.01 × total_redispatch_mw
```

This penalty is specific to Task 1.

### E. Failure penalty

If the agent reaches the 10-step limit without solving the task:

```text
-5.0
```

## 6. Grading

The grader in [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py) combines three ideas.

### A. Survival ratio

The agent gets credit for lasting through the task horizon:

```text
0.7 × (steps_survived / 10)
```

### B. Target bonus

If the threshold was achieved at any point:

```text
+0.5
```

### C. Final-state bonus

The last state gives extra credit:

- `+0.3` if final `max_rho < target_threshold`
- `+0.15` if within `+0.05`
- `+0.05` if within `+0.10`

### D. Legacy early-success score

There is also a preserved “solve early” score:

```text
1.0 - 0.08 × (step - 1)
```

The grader keeps the better of:

- the composite score above
- the legacy early-success score

Finally, the score is clamped to `(0.01, 0.99)` for submission safety.

## 7. Typical Episode Flow

Here is the intended control loop:

1. reset into a high-loading intact grid
2. inspect the highest `rho`
3. look at `sensitivity_guidance`
4. propose a few redispatch candidates
5. simulate them on the live server session
6. execute the safest improvement
7. stop early if every line is below the task threshold

For Task 1, the best behavior is usually:

- avoid random line cuts
- prefer simulator-backed redispatch actions
- reduce the worst line quickly
- avoid oscillating giant redispatchs without improvement

## 8. What Makes Task 1 Tricky

Task 1 is easy to describe, but several details matter:

- the benchmark target bands are lower than the old curriculum bands
- severe benchmark is judged against `0.90`, not `0.80`
- reset may return a stable fallback state if no exact target-band state exists
- the planner is intentionally restricted to redispatch-style actions

So the right mental model is:

“Start from a realistic high-loading operating point, then use small, informed generation shifts to cool the grid below the task threshold before time runs out.”

## 9. Key Files

- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py)
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py)
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py)
