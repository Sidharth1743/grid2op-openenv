# Task 4: `multi_stage_cascade`

## Overview

`multi_stage_cascade` is the hardest task in this environment.

It is not just a bigger version of Task 3. The point is different.

Task 3 asks:

- can the agent stop a cascade from spreading?

Task 4 asks:

- if the grid is already headed into a severe multi-stage failure, can the agent preserve as much viable load as possible across the whole event?

This is why Task 4 is best understood as a **damage-control and survivability** task, not a pure overload-reduction task.

At reset:

- three lines are already disconnected
- load is increased by `20%`
- overflow tolerance is shortened
- the environment expects the system to become structurally fragmented

The agent has **30 steps**, divided into three 10-step stages, to keep as much of the grid usable as possible.

## 1. How Reset Works

Task 4 uses a calibrated reset, not a random catastrophic state.

The reset logic in [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py) does this:

1. choose one predefined triplet of faulted lines
2. set Grid2Op’s overflow window to `2`
3. choose a time series
4. scale the load up by `20%`
5. reset with:
   - three lines disconnected
   - the scaled load injected
6. test whether that scenario survives **5 do-nothing steps**

Only scenarios that survive that probe are accepted.

That means Task 4 is deliberately severe, but it is still intended to be controllable.

### Faulted line triplets

The reset logic chooses from these calibrated triplets:

- `(2, 4, 14)`
- `(2, 4, 15)`
- `(4, 14, 16)`

### Metadata stored at reset

The task stores:

- `faulted_lines`
- `load_scale`
- `time_series_id`
- `stage_count = 3`
- `stage_length = 10`
- `initial_total_load_mw`
- `overflow_window = 2`
- `do_nothing_probe_steps = 5`

This metadata is used later for stage accounting and island scoring.

## 2. The Three-Stage Structure

Task 4 has an explicit three-stage episode:

- Stage 1: steps `1-10`
- Stage 2: steps `11-20`
- Stage 3: steps `21-30`

At each step, the environment computes:

- `stage_index`
- `steps_to_stage_boundary`
- `stage_boundary_assessed`
- `available_load_ratio`
- `available_island_ratio`
- `majority_islands_available`

The important point is that Task 4 is not judged only at the final step. The environment keeps track of whether the grid is still breaking into islands that can actually survive as stages progress.

## 3. What “Available Load” Means

This is the core idea of Task 4.

As the network breaks apart, the grid may split into several connected components, or **islands**.

Some of those islands are viable:

- they contain enough generation capacity to support the load inside them

Some are not:

- they contain more load than the local generators can support

Task 4 measures this explicitly.

### Island viability rule

For each connected component:

```text
if local_generation_capacity >= local_load:
    island is available
else:
    island is unavailable
```

From that, the environment computes:

### `available_load_ratio`

How much of the original total load is still located in viable islands.

This is the most intuitive Task 4 metric.

Example:

- initial load = 100 MW
- viable islands still contain 68 MW of load
- then `available_load_ratio = 0.68`

### `available_island_ratio`

What fraction of the evaluated islands are viable.

Example:

- 4 islands exist
- 3 of them are self-sustaining
- then `available_island_ratio = 0.75`

### `majority_islands_available`

Whether more than half of the evaluated islands are viable.

This is used by the grader.

## 4. Why Task 4 Is Different

Task 4 is not solved by asking:

- which action gives the lowest `max_rho` right now?

That question is too narrow.

The better question is:

- if the grid fragments further, will the remaining islands still have enough generation to carry their own load?

So Task 4 is about preserving **future survivability**.

That is why the planner prompt emphasizes:

- transferable generation
- controlled islanding only when justified
- not cutting topology unless it clearly helps more than redispatch

## 5. What the Agent Sees

Task 4 uses the standard observation plus Task 4-specific metadata:

- `rho`
- `line_status`
- `timestep_overflow`
- `available_load_ratio`
- `available_island_ratio`
- `stage_index`
- `steps_to_stage_boundary`
- `stage_boundary_assessed`
- `majority_islands_available`

The planning prompt in [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py) explicitly tells the model:

- which stage it is in
- how many steps remain before the next boundary
- how much load is still in viable islands
- whether controlled-islanding candidates exist
- whether redispatch candidates exist

This gives the model a concrete picture of whether the grid is merely stressed or structurally losing viable load.

## 6. Reward Function

Task 4 uses a reward in [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py) that reflects cost, survivability, and stage-boundary damage.

### A. Generation cost penalty

Each step pays a cost for total generation relative to the initial total load:

```text
-0.02 × (total_generation / initial_total_load)
```

This discourages wasteful control.

### B. Island-availability reward

Each step also gets:

```text
+0.5 × available_island_ratio
```

This rewards keeping more islands viable.

### C. Boundary load-loss penalty

At stage boundaries only, the environment penalizes lost viable load:

```text
-5.0 × (1 - available_load_ratio)
```

So if a lot of load is stranded in dead islands at the boundary, the task score suffers sharply.

### D. Terminal success bonus

If the agent survives the full 30 steps **and** at least `50%` of the original load remains available:

```text
+8.0 × (available_load_ratio²)
```

This strongly rewards preserving a large fraction of usable load by the end.

### E. Failure penalty

If the episode ends early or the power flow fails to converge:

```text
-12.0
```

## 7. Grading

The grader in [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py) measures four things.

### A. Stage completion: 30%

Did the agent survive into Stage 2, into Stage 3, and then all the way to the end?

This is not a pure survival reward; it measures progress across the structured event.

### B. Load preservation: 40%

What is the final `available_load_ratio`?

This is the single largest part of the Task 4 score, because preserving usable load is the central objective.

### C. Island quality: 20%

At the stage boundaries, did most islands remain viable?

This is measured using `majority_islands_available`.

### D. Speed bonus: 10%

Within each stage, how quickly did the grid reach a state where all lines were below `100%`?

This rewards not only surviving, but stabilizing early enough to leave room for the next stage.

### Final score

The final score is the weighted sum of those four components, then clamped to `(0.01, 0.99)` for submission safety.

## 8. Planner Behavior

Task 4 uses stronger task-specific instructions than the earlier tasks.

The planner is told to:

- assume the cascade will continue across stages
- think beyond the current step
- preserve islands that can still support their own load
- prefer redispatch over risky topology cuts unless the topology action is clearly justified

The inference code also filters unsafe topology-disconnect candidates before final selection.

That matters because in Task 4, one bad topology cut can strand a large amount of load in an island that no longer has enough generation.

## 9. Typical Episode Flow

A strong Task 4 episode often looks like this:

1. start in a badly damaged grid with three missing lines
2. identify which islands or corridors are at risk
3. use redispatch to keep as much generation-load balance as possible inside survivable parts of the network
4. avoid topology changes that create dead islands
5. cross the step-10 boundary with a healthy `available_load_ratio`
6. repeat the same logic for the next stage
7. finish at step 30 with as much viable load preserved as possible

The right mindset is:

- not “win every local overload fight”
- but “make sure the parts of the grid that remain connected can still stand on their own”

## 10. What Makes Task 4 Hard

Task 4 is hard because it mixes:

- severe initial structural damage
- heavier load than any other task
- faster overflow progression
- explicit stage boundaries
- island viability reasoning
- long-horizon planning

An action can reduce stress on one line right now while silently destroying the viability of a future island.

That is why Task 4 is the most strategic task in the suite.

The correct mental model is:

“The cascade is not fully avoidable, so preserve the parts of the grid that can still keep real load alive through all three stages.”

## 11. Key Files

- [server/tasks.py](/home/sidharth/Desktop/grid2op-openenv/server/tasks.py)
- [server/environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)
- [server/graders.py](/home/sidharth/Desktop/grid2op-openenv/server/graders.py)
- [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py)
