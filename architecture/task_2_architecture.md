# Task 2: `n_minus_1`

## Overview

`n_minus_1` is the standard degraded-grid task in this environment.

The name comes from power-systems practice: if a network has `N` important components, an “N-1” event means one of them has failed and the system must continue operating safely anyway.

In this implementation:

- line `0` is disconnected at reset
- the rest of the grid stays online
- the agent has **20 steps** to keep the network stable
- the agent should reconnect the missing line when that becomes safe

In plain language:

- Task 1 asks “can you cool down one stressed intact grid?”
- Task 2 asks “can you operate safely after a real line outage?”

## 1. How Reset Works

Task 2 does not use a warmup search.

The reset logic in [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py) simply resets the environment with one fixed line already disconnected:

```python
options={"init state": {"set_line_status": [(0, -1)]}}
```

That means:

- line `0` starts out of service
- the benchmark tier is fixed as `n_minus_1_fixed`
- the starting problem is structural, not just a high-loading snapshot

The returned metadata includes:

- `faulted_lines = [0]`
- `curriculum_stage = "fixed_n_minus_1"`
- `scenario_mode`
- `benchmark_tier`

## 2. What the Agent Is Trying to Do

The agent has three jobs at once:

1. survive the immediate disturbance
2. keep the remaining network in a safe operating region
3. reconnect the missing line when that is allowed and safe

This is why the planner treats the task in two phases:

### Phase 1: Emergency window

The first five steps are treated as “clear the danger quickly.”

The key target is:

```text
max_rho < 0.92
```

If the grid is above that, the planner should prefer active control over passive waiting.

### Phase 2: Sustained secure operation

After the emergency is cleared, the goal shifts to:

- keep the grid under control for the rest of the 20-step horizon
- spend as many steps as possible with `max_rho < 0.90`
- reconnect line `0` when cooldown allows and simulation shows it is safe

## 3. What the Agent Sees

Task 2 uses the same typed observation fields as the other tasks:

- `rho`
- `gen_p`
- `load_p`
- `line_status`
- `timestep_overflow`
- `sensitivity_guidance`

In addition, the planner gets useful metadata through `planning_context`:

- `n1_security_score`
- `bridge_lines`
- graph centrality and corridor information
- redispatchable generator bounds

That matters because Task 2 is not just about “lower one number.” The agent needs to understand how the missing line changes the structure of the network.

## 4. Reward Function

Task 2 uses a three-part reward in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py).

### A. Survival signal

Every step starts with:

```text
R_survive = +1.0
```

This keeps the reward dense and gives the agent constant credit for staying alive.

### B. Loading quality

The reward also measures how healthy the line loadings are:

```text
R_overload = mean(clip(1 - rho, -1, 1))
```

This is positive when lines are comfortably loaded and worse when lines are near or above their limits.

### C. Redispatch cost

Redispatch is useful, but not free:

```text
R_cost = -0.05 × Σ|ΔMW|/max_ramp
```

Large redispatch moves cost more than smaller ones.

### Combined reward

The final per-step combination is:

```text
0.3 × R_survive + 0.6 × R_overload + 0.1 × R_cost
```

### Reconnection bonus

If the agent asks to reconnect a line and that reconnection really succeeds, the reward can get:

```text
+2.0
```

But only if the reconnect is also judged safe enough:

- the line was actually restored
- `max_rho` did not worsen by more than `0.1`

### Terminal terms

- if the agent survives all 20 steps: quadratic survival bonus
- if the episode ends early: `-15.0`

So Task 2 rewards calm, sustained secure operation much more than one flashy move.

## 5. Reconnection Logic

Reconnection is one of the main differences between Task 2 and the other tasks.

The environment checks two things:

### A. Did the reconnection actually happen?

The agent may request a reconnect, but the grid only gets credit if the line status truly changed from disconnected to connected.

### B. Was the reconnection safe enough?

The reconnect bonus is given only when the new `max_rho` is not much worse than the old one.

The current rule is:

```text
current_max_rho <= previous_max_rho + 0.1
```

This prevents the agent from scoring well by reconnecting recklessly.

## 6. Planner Behavior

The inference path in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py) gives Task 2 a task-specific prompt and selection policy.

The prompt tells the model:

- line `0` is the faulted line
- the first five steps are an emergency phase
- in the emergency phase, active redispatch is preferred over passive waiting
- after cooldown, reconnect line `0` when it is safe

The planner also includes:

- `EMERGENCY_LINES` with `rho >= 0.92`
- `WARNING_LINES` with `0.80 <= rho < 0.92`
- `RECONNECT_WINDOW_LINES`
- `n1_security_score`

This is intended to push the model toward realistic grid-operator behavior:

- first stop the dangerous state
- then maintain secure operation
- then restore missing transfer capacity

## 7. Grading

The grader in [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py) is phase-aware.

### A. Emergency response: 30%

Did the agent get below `0.92` within the first five steps?

Faster is better.

### B. Sustained security: 50%

Among steps `6-20`, how many stayed below `0.90`?

This is the largest part of the score because the task is about operating safely after the fault, not just surviving the first few moments.

### C. Reconnection: 20%

Did the agent reconnect line `0` at any point?

### Final score

The grader multiplies the mastery score by survival ratio:

```text
final_score = survival_ratio × mastery_score
```

Then it clamps the result to `(0.01, 0.99)` for submission safety.

## 8. Typical Episode Flow

A good Task 2 episode usually looks like this:

1. start with line `0` disconnected
2. identify the most stressed remaining lines
3. use redispatch or a safe line action to reduce the emergency
4. keep `max_rho` under control for the rest of the horizon
5. reconnect line `0` once the grid can handle it
6. finish all 20 steps without blackout

The strong behavior is not “never touch topology” or “always reconnect immediately.” The strong behavior is:

- act quickly when the grid is in danger
- do not make unnecessary risky changes
- reconnect when the simulation says it is safe

## 9. What Makes Task 2 Difficult

Task 2 is harder than Task 1 because:

- one real line is already gone
- the power flows have already been redistributed
- the agent must think about both present stress and future reconnection
- some actions that help the emergency can hurt the later steady state

So the correct mental model is:

“Operate a real grid in degraded mode after one transmission-line outage, then restore it safely.”

## 10. Key Files

- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py)
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py)
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py)
