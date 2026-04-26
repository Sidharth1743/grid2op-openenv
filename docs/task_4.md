# Task 4 — `multi_stage_cascade`

## Purpose

Task 4 is the hardest benchmark task. Three lines are disconnected, load is increased, and the task is designed around guaranteed multi-stage degradation rather than full prevention. The goal is to preserve as much load as possible across stage boundaries over `30` steps.

Current task definition in code:
- description and scenario setup: [server/tasks.py](../server/tasks.py)
- reward: [server/environment.py](../server/environment.py)
- grader: [server/graders.py](../server/graders.py)

## What We Changed

This task changed the most during development.

Main changes:
- made it explicitly stage-aware in both prompt and reward logic
- added stage boundary assessment and island-availability tracking
- added controlled-islanding and redispatch guidance to the prompt
- changed the reward to penalize load loss at stage boundaries instead of only rewarding short-term survival
- changed the grader to score stage completion, load preservation, island quality, and stage-wise stabilization speed

These changes were motivated by multi-stage cascading-failure literature, where greedy single-stage action selection is known to miss later-stage consequences. [1][2]

## Current Implementation

Reset:
- three lines disconnected
- load increase around the expert benchmark setting
- overflow window tightened for faster propagation
- `30` steps split into three conceptual stages

Prompt-side task rules now include:
- explicit stage context such as `stage_1_of_3`
- steps remaining to the next boundary
- `available_load_ratio`
- `available_island_ratio`
- guidance to avoid stranding load in islands without enough generation
- separate controlled-islanding and redispatch guidance blocks

Current reward shape:
- generation cost penalty
- positive term for available-island convergence
- load-loss penalty at stage boundaries
- heavy penalties for blackout or convergence failure
- positive win reward at the horizon if enough load is preserved

Current grader:
- `30%` stage completion
- `40%` load preservation
- `20%` island quality
- `10%` speed of stabilization within each stage

## What We Observed

This is the task where the SFT model most clearly outperformed the base model while staying safe.

Final SFT result, seed block `0..4`:
- `0.9156444`

Final SFT result, unseen seeds `100..102`:
- `0.9069863`

Base-model result on seed block `0..4`:
- `0.000`

This task also exposed the limit of the original offline GRPO setup:
- completed GRPO runs preserved the strong SFT behavior
- they validated the RL stack on the hardest benchmark task

## Design Scope

The task is intentionally constrained by verified candidate actions and a fixed benchmark horizon. That makes it robust, reproducible, and well suited for controlled comparison across model variants.

## References

[1] Mitigating Multi-Stage Cascading Failure by Reinforcement Learning (ISGT Asia 2019):  
https://vbn.aau.dk/en/publications/mitigating-multi-stage-cascading-failure-by-reinforcement-learnin/

[2] Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation:  
https://www.climatechange.ai/papers/iclr2025/1

[3] Local implementation:  
[server/tasks.py](../server/tasks.py), [server/environment.py](../server/environment.py), [server/graders.py](../server/graders.py), [inference.py](../inference.py)
