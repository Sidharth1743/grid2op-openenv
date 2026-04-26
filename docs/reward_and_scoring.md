# Reward And Scoring

This document is the authoritative reference for how the environment rewards each step and how the final benchmark score is computed for each task.

The key design choice is that **training reward** and **submission score** are related but not identical:

- the environment emits a shaped reward at every step to provide dense learning signal
- the official benchmark score is computed afterward by a task-specific grader over the full episode log

That separation is deliberate. It keeps the training signal useful without letting a convenient shaped reward quietly replace the real operational objective. [1][2]

Core implementation:
- reward shaping: [server/environment.py](../server/environment.py)
- graders: [server/graders.py](../server/graders.py)
- logged per-step fields: [models.py](../models.py)

## Global Scoring Rule

All task graders end by clamping the reported submission score into the range:

- minimum: `0.01`
- maximum: `0.99`

This happens in:
- [server/graders.py](../server/graders.py) via `_clamp_submission_score(...)`

So when we report a score such as `0.990`, that is the capped top-end submission score rather than an uncapped raw metric.

## Why There Are Two Layers

The environment reward answers:

> was this step locally useful and safe?

The grader answers:

> did the whole episode accomplish the actual task objective?

That matters because:

- a policy can collect acceptable step rewards while still missing the deeper objective
- a policy can be locally conservative but globally strong
- different tasks need different definitions of success

This is especially important for cascade tasks, where short-horizon survival and long-horizon stabilization are not the same thing. [2][3]

## Task 1: `single_fault`

### Environment Reward

Source:
- [server/environment.py](../server/environment.py)

Episode horizon:
- `10` steps

Success threshold:
- default benchmark target: all lines below `max_rho < 0.80`
- severe tier override: all lines below `max_rho < 0.90`

The shaped reward is:

1. Early success bonus:
   - if all lines are below the task threshold, add `1.0 / step_count`
   - this gives higher reward for solving the task earlier

2. Margin bonus:
   - add `0.05 * max(0, 1 - max_rho)`
   - this rewards lower peak loading even before full success

3. Overload penalty:
   - subtract `0.2 * overloaded_line_count`

4. Redispatch penalty:
   - subtract `0.01 * total_abs_redispatch_mw`
   - this discourages unnecessarily large redispatch interventions

5. Timeout penalty:
   - if the episode reaches the horizon without clearing the target, subtract `5.0`

In short, Task 1 reward pushes the policy toward:

- fast threshold clearance
- lower thermal stress
- smaller redispatch when possible

### Final Benchmark Score

Source:
- [server/graders.py](../server/graders.py) via `grade_single_fault(...)`

The final score combines:

1. Survival ratio:
   - `len(episode_log) / 10`
   - weighted by `0.7`

2. Target bonus:
   - `+0.5` if the target was achieved at any step

3. Final-state bonus:
   - `+0.3` if final `max_rho < threshold`
   - `+0.15` if final `max_rho < threshold + 0.05`
   - `+0.05` if final `max_rho < threshold + 0.10`

4. Legacy early-success score:
   - if the target is reached, score can also use:
   - `1.0 - 0.08 * (step - 1)`
   - the grader takes the better of the composite score and this legacy early-success rule

Interpretation:

- Task 1 rewards both solving the problem and solving it early
- the final state still matters, even if the target was not fully reached

## Task 2: `n_minus_1`

### Environment Reward

Source:
- [server/environment.py](../server/environment.py)

Episode horizon:
- `20` steps

Important thresholds:
- emergency threshold: `max_rho < 0.92`
- secure phase threshold: `max_rho < 0.90`

The shaped reward is:

1. Survival term:
   - `R_survive = 1.0`

2. Thermal margin term:
   - for each line, compute `clip(1 - rho, -1, 1)`
   - average this over all lines
   - call the result `R_overload`

3. Redispatch cost term:
   - normalize each redispatch by the generator’s max ramp
   - sum the absolute normalized redispatch magnitudes
   - multiply by `0.05`
   - use the negative of that as `R_cost`

4. Weighted combination:
   - reward += `0.3 * R_survive + 0.6 * R_overload + 0.1 * R_cost`

5. Safe reconnection bonus:
   - add `2.0` when a reconnection succeeds and does not worsen peak loading by more than `0.1`

6. Terminal shaping:
   - if the full horizon is survived without convergence failure:
     - add `10.0 * (step_count / max_steps)^2`
   - if the episode ends early:
     - subtract `15.0`

Interpretation:

- this reward explicitly mixes survival, line loading, and intervention cost
- reconnection is encouraged, but only when it remains operationally safe

### Final Benchmark Score

Source:
- [server/graders.py](../server/graders.py) via `grade_n_minus_1(...)`

The final score is:

1. Emergency response score:
   - look at steps `1..5`
   - find the first step where `max_rho < 0.92`
   - score = `1.0 - 0.2 * (clear_step - 1)`
   - if never cleared in the emergency window, score = `0`

2. Phase-2 security ratio:
   - look at steps `6..20`
   - compute fraction of those steps with `max_rho < 0.90`

3. Reconnection score:
   - `1.0` if the faulted line is reconnected successfully during the episode
   - else `0.0`

4. Mastery score:
   - `0.30 * emergency_score + 0.50 * security_ratio + 0.20 * reconnection_score`

5. Survival multiplier:
   - multiply mastery score by `survival_ratio`

Interpretation:

- early emergency clearance matters
- keeping the degraded grid secure matters even more
- reconnection is important, but it is not allowed to dominate the whole task

## Task 3: `cascade_prevent`

### Environment Reward

Source:
- [server/environment.py](../server/environment.py)

Episode horizon:
- `30` steps

The shaped reward is:

1. Auto-trip control term:
   - `+0.3` if no automatic trip is detected that step
   - `-2.5` if an automatic trip is detected

2. Overflow-countdown penalty:
   - subtract `0.05 * sum(timestep_overflow_i^2)`
   - this makes prolonged overload countdowns increasingly expensive

3. Thermal margin term:
   - compute `clip(1 - rho, -1, 1)` for each line
   - average over lines
   - add `0.1 * thermal_margin`

4. Terminal penalties:
   - if convergence fails: subtract `12.0`
   - if the episode ends early: subtract `12.0`

5. Full-horizon completion bonus:
   - if the agent survives to the horizon:
   - count automatic trips across the episode
   - add `5.0 * max(0, 1 - auto_trips / 5)^2`

Interpretation:

- this reward focuses on preventing overload-driven propagation, not merely lowering one scalar metric
- overflow countdowns matter because they are the mechanism by which a stressful state becomes a cascade

### Final Benchmark Score

Source:
- [server/graders.py](../server/graders.py) via `grade_cascade_prevent(...)`

The final score uses:

1. Containment ratio:
   - fraction of `30` steps with no automatic trip
   - weighted `0.5`

2. Stability ratio:
   - among containment steps, fraction with all lines below `100%` loading
   - weighted `0.3`

3. Recovery score:
   - if overloads appear, find how fast the episode returns to all lines below `100%`
   - `1.0` if already stable
   - otherwise `1 - (stabilize_step - first_overload_step) / 10`
   - weighted `0.2`

Interpretation:

- containment matters most
- thermal stability comes next
- speed of recovery is rewarded, but only after containment and stability are handled

## Task 4: `multi_stage_cascade`

### Environment Reward

Source:
- [server/environment.py](../server/environment.py)

Episode horizon:
- `30` steps

Important metadata fields:
- `available_load_ratio`
- `available_island_ratio`
- `stage_boundary_assessed`
- `majority_islands_available`

The shaped reward is:

1. Generation cost penalty:
   - subtract `0.02 * (total_generation / initial_total_load)`

2. Island-availability term:
   - add `0.5 * available_island_ratio`

3. Stage-boundary load-loss penalty:
   - whenever a stage boundary is assessed:
   - subtract `5.0 * max(0, 1 - available_load_ratio)`

4. Terminal penalties:
   - if convergence fails: subtract `12.0`
   - if the episode ends early: subtract `12.0`

5. Full-horizon win term:
   - if the full horizon is survived and `available_load_ratio >= 0.5`
   - add `8.0 * available_load_ratio^2`

Interpretation:

- this reward is stage-aware
- preserving available load and preserving usable islands both matter
- the horizon bonus only pays out when the policy gets deep enough into the task and keeps enough load alive

### Final Benchmark Score

Source:
- [server/graders.py](../server/graders.py) via `grade_multi_stage_cascade(...)`

The final score is:

1. Stage completion:
   - `1/3` credit for reaching step `10`
   - `1/3` credit for reaching step `20`
   - `1/3` credit for reaching the full horizon without blackout
   - weighted `0.30`

2. Load preservation:
   - final `available_load_ratio`
   - weighted `0.40`

3. Island quality:
   - count how often stage boundaries retain majority-available islands
   - normalized to `[0,1]`
   - weighted `0.20`

4. Stage speed:
   - for each 10-step stage, measure how quickly the agent returns to all lines below `100%`
   - weighted `0.10`

Interpretation:

- this grader cares first about finishing stages and preserving load
- it also rewards structurally useful islanding and faster within-stage stabilization

## Episode Log Fields Used By Graders

The grader logic depends on structured fields stored in [EpisodeStepLog](../models.py), including:

- `max_rho`
- `reward`
- `done`
- `all_lines_below_target`
- `all_lines_below_80`
- `all_lines_below_100`
- `disconnected_lines`
- `auto_trip_detected`
- `reconnect_successful`
- `available_load_ratio`
- `available_island_ratio`
- `stage_boundary_assessed`
- `majority_islands_available`

This is important because the benchmark is not inferred from free text after the fact. It is computed from explicit, typed, per-step episode traces.

## Summary

- `single_fault` rewards fast thermal-target achievement
- `n_minus_1` rewards early emergency clearing, secure operation, and safe reconnection
- `cascade_prevent` rewards prevention of auto-trips and stable recovery
- `multi_stage_cascade` rewards stage completion, load preservation, island quality, and stabilization

That is why a single generic reward would not have been enough for this environment.

## References

[1] Grid2Op reward documentation:  
https://grid2op.readthedocs.io/en/v1.9.8/reward.html

[2] Learning to run a power network challenge for training topology controllers:  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

[3] RL2Grid benchmark framing for power-grid RL evaluation:  
https://huggingface.co/papers/2503.23101

[4] Local implementation:  
[server/environment.py](../server/environment.py), [server/graders.py](../server/graders.py), [models.py](../models.py)
