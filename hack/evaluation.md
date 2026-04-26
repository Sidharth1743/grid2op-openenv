# Evaluation Notes

This document records the final evaluation status for the Grid2Op OpenEnv submission and the reason the SFT adapter is the chosen model.

## Submission Decision

Final submission model:
- `Qwen/Qwen3-4B-Instruct-2507` + LoRA adapter `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Why:
- it is the strongest completed model
- it clearly beats the base model on the hardest tasks
- it stays safe on both the main and unseen seed blocks
- completed GRPO runs reinforced it as the strongest evaluated operating point

## Evaluation Setup

All evaluated models used the same verified-candidate inference pipeline in [ft_inference.py](/home/sidharth/Desktop/grid2op-openenv/ft_inference.py):

1. reset a task episode
2. enumerate legal Grid2Op actions
3. simulate candidate actions
4. prompt the model with verified candidate outcomes
5. require a valid `GridAction` JSON output
6. require the selected action to exactly match one verified candidate
7. execute the action and grade the episode

This means the comparison is controlled. The base model, SFT model, and GRPO models all saw the same style of prompt and the same verified-action constraint.

Models:
- base: `Qwen/Qwen3-4B-Instruct-2507`
- SFT: `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Log analysis:
- [check_ft_inference_log.py](/home/sidharth/Desktop/grid2op-openenv/scripts/check_ft_inference_log.py)

## What Improved In SFT

The SFT gain came from a combination of model training and environment-facing prompt/control improvements:
- verified-candidate prompting, so the model chose from simulator-checked actions instead of inventing arbitrary ones
- stricter action-schema learning, which sharply reduced invalid JSON and malformed action payloads
- task-specific prompt guidance
- threshold-aware candidate ranking for `n_minus_1`
- safer and more conservative behavior on cascade tasks, where validity matters more than risky action invention

This was not just a cosmetic formatting gain. It changed actual benchmark performance.

## Base Vs SFT: Main Seed Block

Seed block:
- `0..4`
- `5` episodes per task

Scores:

| Task | Base | SFT |
|---|---:|---:|
| `single_fault` | `0.856` | `0.856` |
| `n_minus_1` | `0.952` | `0.990` |
| `cascade_prevent` | `0.000` | `0.990` |
| `multi_stage_cascade` | `0.000` | `0.9156444` |

Safety:
- base failures: `10`
- SFT failures: `0`
- SFT safety pass: `true`

Most important result:
- the base model collapsed on the hard cascade tasks because it often produced invalid or unverified actions
- the SFT model completed all evaluated episodes safely

## Final SFT Scores To Report

Main seed block `0..4`, `5` episodes per task:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9156444`
- `n_minus_1`: `0.990`
- `single_fault`: `0.856`
- failures: `0`
- safety pass: `true`

Unseen seed block `100..102`, `3` episodes per task:
- `cascade_prevent`: `0.990`
- `multi_stage_cascade`: `0.9069863`
- `n_minus_1`: `0.9222223`
- `single_fault`: `0.830`
- failures: `0`
- safety pass: `true`

## Action Behavior

The final SFT action profile was sensible for the constrained verified-candidate setup.

Main seed block action counts:
- `single_fault`: `do_nothing=2`, `redispatch=44`
- `n_minus_1`: `do_nothing=16`, `reconnect_line=5`, `redispatch=79`
- `cascade_prevent`: `disconnect_line=10`, `do_nothing=132`, `reconnect_line=8`
- `multi_stage_cascade`: `disconnect_line=7`, `do_nothing=126`, `reconnect_line=17`

Interpretation:
- `n_minus_1` became much more active and threshold-aware than earlier versions
- cascade tasks remained conservative, but in a useful way: safe verified actions instead of invalid ones

## Task Interpretation

`single_fault` remains the most demanding task in the current benchmark suite.

Current evidence suggests the bottleneck is not just the model. In many weak seeds, the available one-step redispatch candidates do not expose an action that actually clears the target `max_rho < 0.80`.

So the result is best described as:
- SFT fixed action validity and protocol adherence
- `single_fault` remains the clearest benchmark for future candidate-space expansion

## GRPO Outcome In Context

Completed GRPO runs were technically successful and consistently preserved the strong SFT operating point.

Completed GRPO results:
- local compact GRPO matched SFT on the main seed block
- local compact GRPO remained close to SFT on unseen seeds
- focused HF Jobs `multi_stage_cascade` GRPO matched the SFT multistage score exactly

This means:
- SFT is the best submission model
- GRPO is a real and working extension of the project
- completed GRPO runs provided a stable RL baseline for the next stage of experimentation

## Final Conclusion

The final result is:
- the base model is inconsistent on the hard Grid2Op tasks
- the SFT model fixes the action protocol problem and strongly improves benchmark performance
- the SFT model stays safe on unseen seeds
- completed GRPO work strengthened the project technically and established a reliable RL foundation

This is the final evaluation summary for the submission.
