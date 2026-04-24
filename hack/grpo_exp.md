# GRPO Experiment Notes

## Goal

The goal of this experiment was to move beyond supervised fine-tuning and test whether GRPO-style reinforcement learning with verifier rewards could improve the Grid2Op control policy while preserving strict action safety and output validity.

The intended design was:
- start from the SFT adapter
- train with TRL GRPO on verified offline trajectories
- use a verifier reward instead of trusting raw environment reward alone
- evaluate against the same seed blocks already used for the SFT baseline

This document records what was tried, what broke, what was fixed, and what the actual outcome was.

## Base Model And Starting Point

Base model:
- `Qwen/Qwen3-4B-Instruct-2507`

Starting adapter:
- `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Primary balanced GRPO dataset:
- `outputs/datasets/grid2op_teacher_wide_balanced_v2.jsonl`

Dataset size:
- `3373` rows

Balanced action distribution by task:
- `cascade_prevent`: 503 rows
- `multi_stage_cascade`: 1199 rows
- `n_minus_1`: 1200 rows
- `single_fault`: 471 rows

Important observation before GRPO:
- the SFT model was already strong and safe
- the main weakness remained `single_fault`
- `single_fault` is structurally hard because many seeds do not admit a one-step or even short-horizon action that actually achieves the threshold target

## Why GRPO Was Added

The benchmark score alone is not enough guidance for a robust controller.

Known issues in this project:
- fixed episode horizons let passive survival look better than it really is
- `do_nothing` can be over-selected in survivable states
- a policy can learn valid output format without learning the real task objective
- the official reward is useful, but too coarse for research-level shaping

The GRPO plan was to keep benchmark evaluation unchanged while adding a stronger training-time verifier reward that explicitly rewards:
- valid JSON format
- valid schema
- exact match to verified candidates
- safety
- task-objective alignment
- anti-hacking behavior

## GRPO Script

Training script:
- [train_grpo_verifier.py](/home/sidharth/Desktop/grid2op-openenv/scripts/train_grpo_verifier.py)

The script trains offline on dataset rows that already contain verified simulator results.

Instead of letting the model invent arbitrary actions and directly calling the live simulator during RL, the reward function maps the generated action back to a verified candidate in the stored rollout set.

This turns GRPO into a constrained offline verifier setup.

## Reward Design

The GRPO reward is composed of multiple reward functions:

- `format_reward`
- `schema_reward`
- `verified_candidate_reward`
- `safety_reward`
- `task_objective_reward`
- `anti_hacking_reward`

Interpretation:
- `format_reward`: rewards strict output formatting
- `schema_reward`: rewards producing a valid `GridAction` structure
- `verified_candidate_reward`: rewards choosing an action exactly present in the verified simulation set
- `safety_reward`: rewards actions that do not create overloads, divergence, or invalid outcomes
- `task_objective_reward`: rewards actions that actually help the task objective
- `anti_hacking_reward`: discourages exploiting formatting gaps or invalid selection behavior

Important design principle:
- the live environment reward was not changed
- GRPO added a training-time verifier reward layer

This matters because the benchmark score remains comparable while the RL training signal becomes more research-focused.

## Prompt Compression

The initial GRPO setup used long prompt text close to the SFT prompt format. This was too heavy for the hardware and unnecessary for offline verifier training.

To fix this, a compact prompt style was added.

Compact prompt contents:
- task id
- current step
- current `max_rho`
- concise task objective
- a compact list of verified candidate actions with simulator outcomes

Measured prompt size improvement:
- original prompt average over first 100 rows: about `4871` chars
- compact prompt average over first 100 rows: about `1034` chars

This reduction was important for fitting GRPO into the available GPUs.

## Hardware Constraints

Available hardware:
- `2 x RTX 3060 12GB`

This immediately constrained the design.

What did not work well:
- naive multi-GPU DDP replication
- large generation count
- long prompt context
- long completion length
- Liger plus extra memory-heavy settings in this setup

Observed problems:
- out-of-memory during GRPO training
- per-rank duplication of the full model
- evaluator batch constraints from TRL GRPO config

## Early Failures

### 1. Chat Template / SFT Issues

Before GRPO, the SFT pipeline already needed a training-compatible chat template because TRL rejected the tokenizer’s native template.

This was solved in the SFT path first.

### 2. GRPO Eval Batch Divisibility Error

A GRPO launch failed with:

```text
ValueError: The global eval batch size (1 * 2) must be divisible by the number of generations used for evaluation (4).
```

Interpretation:
- TRL GRPO requires generation count and eval batch sizing to be compatible

Fix:
- disable eval during smoke runs with `--eval-ratio 0`
- reduce generation pressure

### 3. OOM With Multi-GPU DDP

Several runs failed with CUDA OOM errors on the 3060s.

Representative failures:
- attempted allocation around `550 MiB`
- attempted allocation around `1.21 GiB`
- attempted allocation around `1.25 GiB`

Interpretation:
- DDP replicated too much state across both GPUs
- the effective memory pressure was still too high for this model plus LoRA plus GRPO generation

Conclusion:
- full DDP was not the safest route on this machine

### 4. Reward Canonicalization Crash

Once the compact setup finally reached actual GRPO reward computation, training crashed in the verifier logic:

```text
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

Root cause:
- some verified candidate actions had `line_set` entries with `null`
- the canonicalization helper tried to cast them directly with `int(None)`

Fix:
- normalize and skip invalid null values in `line_set` and `redispatch`
- make canonical action mapping robust to malformed or partially empty candidate payloads

This was an important fix because it proved the training design itself was viable; the crash was in our reward utility code, not in GRPO itself.

## Compact GRPO Smoke Run

After patching the verifier canonicalization and moving to compact prompts, the following smoke run completed successfully:

Core characteristics:
- compact prompts
- short completions
- `num_generations=2`
- `device_map=auto`
- no eval ratio
- no Liger
- low learning rate

Result:
- the run trained to completion
- adapter saved successfully
- checkpoints and completions parquet files were produced

Smoke adapter path:
- `outputs/models/grid2op-qwen3-4b-grpo-smoke-compact`

This was the first proof that GRPO was feasible on the available hardware.

## Real GRPO Run

After smoke validation, a larger compact GRPO run was launched and completed.

Final adapter path:
- `outputs/models/grid2op-qwen3-4b-grpo-compact-v1`

Observed saved checkpoints:
- `checkpoint-100`
- `checkpoint-150`
- `checkpoint-200`

This means the training pipeline itself was operational end-to-end.

## Evaluation Protocol

To compare fairly with the SFT baseline, the GRPO model was evaluated using the same seed blocks:

- seed block `0..4` with `5` episodes per task
- seed block `100..102` with `3` episodes per task

The evaluation script used was:
- [ft_inference.py](/home/sidharth/Desktop/grid2op-openenv/ft_inference.py)

The analysis script used was:
- [check_ft_inference_log.py](/home/sidharth/Desktop/grid2op-openenv/scripts/check_ft_inference_log.py)

## SFT Baseline For Comparison

### SFT seed `0..4`

- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.915644`
- `n_minus_1`: `0.990`
- `single_fault`: `0.856`
- failures: `0`
- safety pass: `true`

### SFT seed `100..102`

- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.906986`
- `n_minus_1`: `0.922222`
- `single_fault`: `0.830`
- failures: `0`
- safety pass: `true`

## GRPO Results

### GRPO seed `0..4`

Evaluation log:
- `outputs/logs/grpo_compact_v1_seed0_5eps.log`

Summary:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9156444`
- `n_minus_1`: `0.990`
- `single_fault`: `0.856`
- failures: `0`
- safety pass: `true`

Task behavior:
- `cascade_prevent`: action mix exactly in the same pattern as strong SFT behavior
- `multi_stage_cascade`: same mean score and safe action mix
- `n_minus_1`: nearly identical to SFT, one extra redispatch and one fewer `do_nothing`
- `single_fault`: identical mean score to SFT

Interpretation:
- GRPO did not improve the policy
- GRPO also did not damage the policy
- it mostly preserved the SFT solution

### GRPO seed `100..102`

Evaluation log:
- `outputs/logs/grpo_compact_v1_seed100_3eps.log`

Summary:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9069863333333332`
- `n_minus_1`: `0.9222223333333334`
- `single_fault`: `0.7833333333333333`
- failures: `0`
- safety pass: `true`

Task behavior:
- `cascade_prevent`: unchanged
- `multi_stage_cascade`: unchanged
- `n_minus_1`: unchanged
- `single_fault`: worse than SFT

Interesting detail:
- `single_fault` chose `redispatch` on all `30` steps across these 3 episodes
- negative terminal episode count remained `3`

Interpretation:
- GRPO kept safety and validity
- GRPO did not improve generalization
- GRPO slightly hurt `single_fault`

## Direct Comparison

### Seed `0..4`

```text
Task                  SFT              GRPO
cascade_prevent       0.990            0.990
multi_stage_cascade   0.915644         0.915644
n_minus_1             0.990            0.990
single_fault          0.856            0.856
failures              0                0
```

### Seed `100..102`

```text
Task                  SFT              GRPO
cascade_prevent       0.990            0.990
multi_stage_cascade   0.906986         0.906986
n_minus_1             0.922222         0.922222
single_fault          0.830            0.783333
failures              0                0
```

## Main Conclusion

The GRPO experiment was a technical success but not yet a policy improvement.

What succeeded:
- verifier-based GRPO training ran successfully on the available hardware
- reward parsing and canonicalization issues were fixed
- compact prompts made the setup feasible
- the resulting GRPO model stayed safe
- no failures were introduced in the evaluated seed blocks

What did not succeed:
- GRPO did not outperform the SFT model on the benchmark seed sets
- `single_fault` got worse on the `100..102` seed block
- there is no evidence yet that current verifier-GRPO improves the submission model

## Why GRPO Likely Did Not Help Yet

The likely issue is not “GRPO is bad”.

The likely issue is that the current reward is still too close to:
- choose a valid action
- choose a verified action
- choose a safe action

and not sharp enough on:
- choose the action that best advances the core task objective

This is most visible in `single_fault`.

The current GRPO setup reinforces legality and safe verified selection very well, but that is not enough if the hard part of the task is objective-driven control quality rather than output validity.

## Single Fault Remains The Hardest Problem

Important earlier diagnostic:
- some `single_fault` seeds have `target_reached_count = 0`
- some also have `lookahead_target_reached_count = 0`

That means the failure is not always because the model is dumb. Sometimes the reachable action space itself does not admit the desired threshold achievement under the current task structure.

So `single_fault` is not a normal formatting problem. It is an objective and reachability problem.

This matters because GRPO reward must not blindly punish the policy for failing to hit an impossible threshold. Instead it should reward:
- best achievable `max_rho` reduction
- meaningful progress toward the threshold
- avoidance of useless redispatch oscillation

## Recommended Use Of GRPO Result

Current recommendation:
- keep `outputs/models/grid2op-qwen3-4b-sft-3k-v1` as the main submission candidate
- treat `outputs/models/grid2op-qwen3-4b-grpo-compact-v1` as a research success, not a final competition model

This is the correct engineering interpretation:
- the pipeline works
- the training idea is viable
- the current reward design is not yet strong enough to beat SFT

## Next GRPO Direction

If GRPO is continued, it should be much more targeted.

Best next step:
- patch the `single_fault` objective reward so it rewards best achievable progress, not just valid verified action selection

Suggested improvements:
- stronger reward for minimal `max_rho`
- explicit relative advantage versus `do_nothing`
- penalty for redispatch actions that are valid but do not meaningfully improve the state
- stronger distinction between objective progress and merely surviving to the horizon

Recommended training strategy:
- do not immediately rerun all tasks
- first run a targeted `single_fault` GRPO experiment
- only promote a new GRPO adapter if it improves `single_fault` without harming cascade and `n_minus_1`

## Final Practical Judgment

Right now:
- SFT is the best submission candidate
- GRPO is a promising research extension
- the current GRPO result proves infrastructure quality, not policy superiority

That is still valuable.

Many projects fail before they even get a stable RL training loop. This one now has:
- a working SFT baseline
- a working evaluation harness
- a working verifier-GRPO trainer
- safe compact training on consumer GPUs
- clear evidence about what reward shaping still needs improvement

That is a strong place to be in before the hackathon.
