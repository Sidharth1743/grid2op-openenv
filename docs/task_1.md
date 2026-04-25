# Task 1 — `single_fault`

## Purpose

Task 1 is the simplest benchmark task, but it is still operationally meaningful. The topology remains intact, and the agent has to reduce line loading under a tight `10`-step horizon. The core target is to bring all lines below the single-fault threshold, which is usually `max_rho < 0.80`.

Current task definition in code:
- description and tiers: [server/tasks.py](../server/tasks.py)
- reward: [server/environment.py](../server/environment.py)
- grader: [server/graders.py](../server/graders.py)

## What We Changed

Task 1 started as a generic congestion-management setup. We tightened it so it better reflects the intended objective.

Main changes:
- warm-started episodes into realistic stressed states instead of trivial resets
- added benchmark tiers for `easy`, `moderate`, and `severe`
- restricted the action family to `redispatch` and `do_nothing`
- explicitly banned topology cuts for this task in the inference prompt
- changed the prompt to rank a primary action and fallback actions
- updated the reward and grader to focus on hitting the thermal target rather than only surviving

## Current Implementation

Reset:
- stressed single-fault warm-start
- `10`-step horizon

Prompt-side task rules:
- do not use `disconnect_line` or `reconnect_line`
- solve congestion with redispatch
- output ranked action candidates

Current reward shape:
- early success bonus if all lines go below the target threshold
- positive term for lower `max_rho`
- penalty for overloaded lines
- action penalty
- strong timeout penalty if the horizon is reached without clearing the target

Current grader:
- survival ratio
- target-achievement bonus
- final-state bonus depending on how close the final `max_rho` is to the threshold

## What We Observed

This task improved less than the others.

Base result, seed block `0..4`:
- `0.856`

Final SFT result, seed block `0..4`:
- `0.856`

Final SFT result, unseen seeds `100..102`:
- `0.830`

Interpretation:
- SFT clearly improved action validity across the project
- but for Task 1 the main remaining bottleneck appears to be candidate reachability, not output formatting

In weak seeds, the available one-step redispatch candidates often do not expose an action that actually drives the grid below the task threshold. This is why Task 1 stayed the most difficult task to improve materially.

## Current Limitation

Task 1 is still constrained by the redispatch candidate space. That means the model can be correct about the best candidate and still fail to hit the objective if the candidate generator does not surface a threshold-clearing action.

## References

[1] Grid2Op `L2RPNReward` documentation:  
https://grid2op.readthedocs.io/en/latest/_modules/grid2op/Reward/l2RPNReward.html

[2] RL2Grid benchmark overview:  
https://huggingface.co/papers/2503.23101

[3] Local implementation:  
[server/tasks.py](../server/tasks.py), [server/environment.py](../server/environment.py), [server/graders.py](../server/graders.py), [inference.py](../inference.py)
