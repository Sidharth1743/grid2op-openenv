# GRPO Experiment Notes

## Goal

The GRPO work was added after the SFT model was already stable and strong. The goal was not to replace the benchmark or change the task definition. The goal was to test whether verifier-guided offline RL could improve the policy further while preserving the strict safety and verified-action constraints already used at inference time.

Starting point:
- base model: `Qwen/Qwen3-4B-Instruct-2507`
- starting adapter: `outputs/models/grid2op-qwen3-4b-sft-3k-v1`
- main GRPO training script: [train_grpo_verifier.py](/home/sidharth/Desktop/grid2op-openenv/scripts/train_grpo_verifier.py)

## What We Changed For GRPO

The main GRPO design decisions were:
- train from the SFT adapter instead of from the base model
- keep training offline on rows with verified simulator candidates
- reward only actions that map back to verified candidates
- add verifier-style reward terms instead of trusting raw environment reward alone
- compress prompts so the setup fits on limited hardware

The reward design focused on:
- output format correctness
- valid `GridAction` schema
- exact match to a verified candidate
- action safety
- task-objective alignment
- anti-hacking behavior

The important engineering fixes during this phase were:
- compact GRPO prompt format to reduce prompt length from roughly `4871` chars to about `1034` chars on the first `100` rows
- robust canonicalization of `line_set` and `redispatch` fields so malformed candidate payloads did not crash reward computation
- stable low-memory settings for local GPUs
- HF Jobs training and evaluation flow with bucket-backed artifacts

## Datasets And Hardware

Primary balanced GRPO dataset:
- `outputs/datasets/grid2op_teacher_wide_balanced_v2.jsonl`
- `3373` rows

Balanced row counts by task:
- `cascade_prevent`: `503`
- `multi_stage_cascade`: `1199`
- `n_minus_1`: `1200`
- `single_fault`: `471`

Later targeted dataset for cloud multistage GRPO:
- filtered informative `multi_stage_cascade` subset

Hardware used:
- local: `2 x RTX 3060 12GB`
- cloud: Hugging Face Jobs with bucket-mounted datasets, adapters, and outputs

## Experiments Run

### 1. Compact GRPO Smoke Run

Adapter:
- `outputs/models/grid2op-qwen3-4b-grpo-smoke-compact`

Purpose:
- verify that compact offline verifier-GRPO was feasible on the local hardware

Outcome:
- run completed
- adapter and checkpoints were saved
- this proved the pipeline worked technically

### 2. Compact Full GRPO Run

Adapter:
- `outputs/models/grid2op-qwen3-4b-grpo-compact-v1`

Saved checkpoints:
- `checkpoint-100`
- `checkpoint-150`
- `checkpoint-200`

This was the main completed local GRPO run.

Seed block `0..4`, `5` episodes per task:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9156444`
- `n_minus_1`: `0.990`
- `single_fault`: `0.856`
- failures: `0`
- safety pass: `true`

Action counts:
- `cascade_prevent`: `disconnect_line=10`, `do_nothing=132`, `reconnect_line=8`
- `multi_stage_cascade`: `disconnect_line=7`, `do_nothing=126`, `reconnect_line=17`
- `n_minus_1`: `do_nothing=15`, `reconnect_line=5`, `redispatch=80`
- `single_fault`: `do_nothing=2`, `redispatch=44`

Seed block `100..102`, `3` episodes per task:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9069863`
- `n_minus_1`: `0.9222223`
- `single_fault`: `0.7833333`
- failures: `0`
- safety pass: `true`

Interpretation:
- the GRPO model stayed safe
- `cascade_prevent`, `multi_stage_cascade`, and `n_minus_1` matched the SFT baseline
- `single_fault` regressed slightly on unseen seeds

### 3. Hugging Face Jobs Multistage GRPO

Cloud run:
- `grid2op-qwen3-4b-grpo-multistage-v1`

Tracking:
- W&B: `https://wandb.ai/sidhu1743/grid2op-openenv-grpo/runs/yq5rgzg0`

Cloud pipeline pieces that worked:
- private HF bucket storage for datasets, adapters, checkpoints, and eval outputs
- HF Jobs launch from Python API
- OpenEnv server started inside eval jobs
- cloud training and cloud evaluation both completed end-to-end

Focused `multi_stage_cascade` evaluation result:
- mean score: `0.9156444`
- failures: `0`
- safety pass: `true`

Action counts:
- `disconnect_line=7`
- `do_nothing=126`
- `reconnect_line=17`

Interpretation:
- the HF Jobs run matched the SFT multistage behavior exactly
- the cloud setup was successful
- the policy still did not improve over SFT

## Comparison Against SFT

SFT reference scores, seed block `0..4`:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9156444`
- `n_minus_1`: `0.990`
- `single_fault`: `0.856`

SFT reference scores, seed block `100..102`:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9069863`
- `n_minus_1`: `0.9222223`
- `single_fault`: `0.830`

Direct result:
- completed GRPO runs did not beat SFT on the evaluated seed blocks
- one completed GRPO run slightly hurt `single_fault`

## What Worked

The GRPO phase still produced real engineering progress:
- compact prompts made verifier-GRPO feasible on limited hardware
- reward parsing and canonicalization were made robust
- offline verifier-GRPO training ran to completion locally
- HF Jobs training and HF Jobs evaluation both worked
- safety and output validity were preserved

These were not fake results. The GRPO pipeline is real and operational.

## Why GRPO Stayed Flat

The completed GRPO setup did not fail because of cloud issues or broken training loops. It stayed flat because the learning signal was weak.

The main reasons were:
- the offline dataset already encoded the best safe one-step verified choice
- the SFT policy already matched that choice distribution well
- GRPO was still training on one-step candidate selection, not on long-horizon improvement
- reward variance for a strong SFT policy was small
- the narrow action space and low generation count gave little relative ranking signal

In short:
- the completed GRPO setup was good at preserving a safe policy
- it was not strong enough to produce a clearly better policy

## Final Position

Final submission model:
- `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Honest conclusion:
- SFT is the strongest completed model
- GRPO is a successful research and engineering extension
- completed GRPO runs did not provide evaluated evidence of policy improvement over SFT

That is the result we should report.
