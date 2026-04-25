# Task 3 — `cascade_prevent`

## Purpose

Task 3 is the first explicitly time-critical cascade-control task in the benchmark. Two lines are disconnected and load is increased at reset. The agent must prevent additional overload-driven trips over a `30`-step horizon.

Current task definition in code:
- description and tiers: [server/tasks.py](../server/tasks.py)
- reward: [server/environment.py](../server/environment.py)
- grader: [server/graders.py](../server/graders.py)

## What We Changed

The original version was too close to a generic “keep max rho down” task. We changed it so the model focuses on actual cascade prevention.

Main changes:
- added tiered benchmark variants: `easy`, `medium`, `hard`, `extreme`
- made prompt logic explicitly prioritize `timestep_overflow`
- changed reward to penalize active overflow countdowns quadratically
- made automatic trips the main negative event
- changed the grader to track containment, thermal stability, and recovery speed

This follows the intuition from cascading-failure RL work: preventing line trips is not the same as simply lowering one summary metric. [1][2]

## Current Implementation

Reset:
- two lines disconnected
- load increase determined by benchmark tier
- `30`-step horizon

Prompt-side task rules now emphasize:
- countdown urgency through `timestep_overflow`
- “trip prevention first, margin improvement second”
- explicit overflow triage instead of purely global `max_rho` optimization

Current reward shape:
- `+0.3` when no automatic trip occurs in the step
- `-2.5` when an automatic trip is detected
- quadratic penalty over `timestep_overflow`
- small positive thermal-margin term
- strong negative penalties for convergence failure or blackout
- end-of-episode bonus if the horizon is survived with few auto-trips

Current grader:
- `50%` containment ratio
- `30%` stability ratio
- `20%` recovery score

## What We Observed

This task is one of the clearest SFT wins in the project.

Final SFT result, seed block `0..4`:
- `0.990`

Final SFT result, unseen seeds `100..102`:
- `0.990`

Base-model result on seed block `0..4`:
- `0.000`

This gap is one of the strongest pieces of evidence that the verified-candidate SFT pipeline learned the environment-specific action protocol and avoided invalid unsafe behavior.

## Current Limitation

This task still uses a simplified cascade-prevention setting rather than a full utility-grade remedial-action pipeline. That is intentional. The benchmark is designed to be reproducible and to isolate control quality under constrained action selection.

## References

[1] Mitigating cascading failure in power grids with deep reinforcement learning-based remedial actions:  
https://www.sciencedirect.com/science/article/pii/S0951832024003156

[2] Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation:  
https://www.climatechange.ai/papers/iclr2025/1

[3] Local implementation:  
[server/tasks.py](../server/tasks.py), [server/environment.py](../server/environment.py), [server/graders.py](../server/graders.py), [inference.py](../inference.py)
