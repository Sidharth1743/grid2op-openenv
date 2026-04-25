# Benchmark Notes

## Benchmark Goal

This benchmark is designed to test whether an LLM-driven controller can operate a small but nontrivial power grid safely under progressively harder disturbance settings. The benchmark is implemented as an OpenEnv environment backed by Grid2Op, with task-specific resets, shaped environment rewards, and separate per-task graders. [1][2][3]

Core implementation:
- task setup: [server/tasks.py](../server/tasks.py)
- environment and shaped rewards: [server/environment.py](../server/environment.py)
- official submission scoring: [server/graders.py](../server/graders.py)
- verified-candidate inference and evaluation: [ft_inference.py](../ft_inference.py)

## Tasks And Variants

The benchmark currently contains four tasks:
- `single_fault`
- `n_minus_1`
- `cascade_prevent`
- `multi_stage_cascade`

Current benchmark variants:
- `single_fault_easy`
- `single_fault_moderate`
- `single_fault_severe`
- `n_minus_1_fixed`
- `cascade_prevent_easy`
- `cascade_prevent_medium`
- `cascade_prevent_hard`
- `cascade_prevent_extreme`
- `multi_stage_cascade_expert`

This gives the benchmark both breadth across tasks and controlled difficulty variation within tasks.

## Why The Benchmark Is Varied

Each task tests a different operational skill:

- `single_fault`: congestion relief under a short redispatch-only horizon
- `n_minus_1`: secure operation after a fixed contingency, including safe reconnection
- `cascade_prevent`: time-critical prevention of auto-trips
- `multi_stage_cascade`: load preservation across stage boundaries under guaranteed degradation

These are not cosmetic variants. They differ in:
- reset structure
- allowed action families
- horizon length
- prompt guidance
- reward shape
- grading logic

That is why the same model can succeed on one task and still struggle on another.

## Why The Benchmark Is Robust

### 1. Same evaluation pipeline for every model

All compared models use the same verified-candidate inference pipeline:
- simulate legal actions
- prompt the model with verified outcomes
- require a valid `GridAction`
- require exact match to a verified candidate
- execute and grade

This makes base vs SFT vs GRPO comparisons much more defensible.

### 2. Strong protection against invalid-action wins

A model does not get credit for:
- invalid JSON
- malformed actions
- invented actions outside the verified set

This is one of the benchmark’s strongest robustness features.

### 3. Separate train-time reward and benchmark score

The environment has shaped rewards, but the official task score comes from dedicated graders. This prevents us from treating a convenient training reward as the benchmark truth.

### 4. Seen-seed and unseen-seed evaluation

We did not only test on one small seed block. The current evaluation story includes:
- seed block `0..4`
- unseen seed block `100..102`

This does not prove full generalization, but it is much stronger than a single-block claim.

### 5. Distinct task-specific scoring

The graders are different on purpose:
- target completion for `single_fault`
- emergency/steady-state/reconnection for `n_minus_1`
- containment/recovery for `cascade_prevent`
- stage completion/load preservation/island quality for `multi_stage_cascade`

This makes it harder for one generic exploit to score well across the entire benchmark.

## Current Results

Best completed submission model:
- `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Main seed block `0..4`, `5` episodes per task:
- `single_fault`: `0.856`
- `n_minus_1`: `0.990`
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9156444`
- failures: `0`

Unseen seed block `100..102`, `3` episodes per task:
- `single_fault`: `0.830`
- `n_minus_1`: `0.9222223`
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9069863`
- failures: `0`

These results show that the benchmark is strong enough to separate:
- an unreliable base model
- a strong SFT model
- a safe but flat GRPO follow-up

## What The Benchmark Still Does Not Claim

We should be careful about overclaiming.

This benchmark does not prove:
- utility-grade deployment readiness
- full contingency coverage
- universal transfer to much larger grids

What it does provide is:
- a reproducible four-task control suite
- realistic Grid2Op dynamics
- verified-candidate action enforcement
- task-specific grading
- seen and unseen seed evaluation

That is a strong benchmark package for a hackathon submission.

## References

[1] OpenEnv integration for training and evaluation:  
https://huggingface.co/docs/trl/openenv

[2] Learning to run a power network challenge for training topology controllers:  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

[3] RL2Grid benchmark paper:  
https://huggingface.co/papers/2503.23101

[4] Local implementation:  
[server/tasks.py](../server/tasks.py), [server/environment.py](../server/environment.py), [server/graders.py](../server/graders.py), [ft_inference.py](../ft_inference.py)
