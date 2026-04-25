# Task 3: `cascade_prevent`

## Overview

`cascade_prevent` is the first task in the suite where the grid is no longer merely stressed or degraded. It is already moving toward a cascade.

At reset:

- one or two lines are disconnected
- demand is scaled up
- some remaining lines are already close to or above their limits

The agent has **30 steps** to stop the situation from turning into automatic line trips and a broader collapse.

In plain language:

- Task 2 is “keep a damaged grid stable”
- Task 3 is “stop the damage from spreading”

## 1. How Reset Works

Task 3 starts from a fresh reset, then injects both topology damage and extra load.

The reset logic in [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py) does this:

1. reset the base Grid2Op environment
2. choose a fault profile from curriculum or benchmark mode
3. multiply `load_p` by a load scale
4. reset again with:
   - selected lines disconnected
   - the scaled load injection applied

### Curriculum profiles

- episodes `1-3`: one line down, `+5%` load
- episodes `4-6`: one line down, `+10%` load
- episodes `7-9`: two lines down, `+10%` load
- episodes `10+`: two lines down, `+15%` load

### Benchmark profiles

- `cascade_prevent_easy`: one line down, `+5%`
- `cascade_prevent_medium`: one line down, `+10%`
- `cascade_prevent_hard`: two lines down, `+10%`
- `cascade_prevent_extreme`: two lines down, `+15%`

### Faulted line pairs

The current reset logic chooses from a small set of predefined fault pairs:

- `(0, 8)`
- `(0, 7)`
- `(0, 3)`

So Task 3 is not random chaos. It is a calibrated family of difficult but reproducible cascading-risk states.

## 2. Why `timestep_overflow` Matters

Task 3 is built around one Grid2Op concept:

```text
timestep_overflow[line_id]
```

This is the number of consecutive steps that a line has remained overloaded.

That makes Task 3 different from a simple “reduce max_rho” problem.

Example:

- a line at `rho = 0.98` with overflow `0` is stressed but not immediately dying
- a line at `rho = 1.03` with overflow `2` is urgent, because it may trip very soon

So in Task 3, urgency is line-specific and time-sensitive.

The planner is explicitly told to prioritize active countdowns over small improvements in global line loading.

## 3. What the Agent Is Trying to Do

The agent is trying to:

1. prevent automatic line trips
2. bring overloaded lines back under control
3. keep the grid stable long enough to finish the 30-step horizon

This is a triage problem.

The best action is not always the one that gives the lowest immediate `max_rho`. The best action is often the one that prevents the next trip.

## 4. What the Agent Sees

Task 3 uses the standard observation fields:

- `rho`
- `gen_p`
- `load_p`
- `line_status`
- `timestep_overflow`
- `sensitivity_guidance`

The planner prompt in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py) adds Task 3-specific emphasis:

- lines with non-zero overflow are listed explicitly
- candidates are expected to focus on the most urgent countdowns
- the model is told that a line with `overflow=2` matters more than a merely high `rho` with `overflow=0`

This is the core behavioral difference from Task 2.

## 5. Reward Function

Task 3 uses a reward in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py) that directly reflects cascade urgency.

### A. Auto-trip signal

Each step gets:

```text
+0.3 if no automatic trip happened
-2.5 if an automatic trip happened
```

This is the main control objective.

### B. Overflow urgency penalty

The reward penalizes active overload countdowns quadratically:

```text
-0.05 × Σ overflow²
```

This is important:

- overflow `1` is bad
- overflow `2` is much worse
- the penalty grows faster as the line gets closer to tripping

### C. Thermal margin signal

The reward also includes:

```text
+0.1 × mean(clip(1 - rho, -1, 1))
```

This pushes the agent toward healthier margins overall.

### D. Terminal terms

- convergence failure or early collapse: `-12.0`
- if the full horizon is reached: a survival bonus based on how many auto-trips occurred

The survival bonus shrinks as trips accumulate, so the task rewards true containment rather than merely surviving the aftermath.

## 6. Grading

The grader in [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py) measures three things.

### A. Cascade containment: 50%

How many of the 30 steps avoided auto-trips?

```text
steps_without_auto_trip / 30
```

This is the largest part of the score because it reflects the task’s main purpose.

### B. Thermal stability: 30%

Among steps with no auto-trip, how many had all lines below `100%`?

This distinguishes “barely hanging on” from genuinely stable control.

### C. Recovery speed: 20%

After the first overloaded state appeared, how fast did the agent bring the grid back below `100%` everywhere?

Faster recovery gets more credit.

### Final score

The grader combines those three components and then clamps the result to `(0.01, 0.99)` for submission safety.

## 7. Planner Behavior

The planner logic for Task 3 is built around urgency.

In [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py), the prompt tells the model:

- prioritize lines with active overflow countdowns
- stop automatic trips first
- improve overall thermal margin second

This matters because a strategy that only minimizes `max_rho` can still fail if it ignores the line that is about to trip next.

## 8. Typical Episode Flow

A strong Task 3 episode usually looks like this:

1. start with one or two lines already out
2. identify which remaining lines are overloaded
3. check which lines have active overflow countdowns
4. apply redispatch or safe topology changes to stop the most urgent line from tripping
5. keep repeating that triage until the grid settles
6. finish all 30 steps with as few automatic trips as possible

The correct mindset is:

- not “optimize one global metric”
- but “stabilize the most dangerous part of the network before it cascades”

## 9. What Makes Task 3 Difficult

Task 3 is difficult because it combines:

- structural damage from missing lines
- increased load
- per-line countdown pressure
- a longer horizon with many chances for the grid to destabilize again

An action can help one overloaded corridor while making another one worse. That is why Task 3 is the first task that really feels like cascade management rather than simple overload relief.

So the right mental model is:

“A few lines are already gone, several others are in danger, and every step you wait increases the chance of a chain reaction.”

## 10. Key Files

- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py)
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py)
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py)
