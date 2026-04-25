# Plot Notes

The following plots were generated for the submission and stored in [hack/assets](./assets).

## 1. Main benchmark comparison

File:
- [hack/assets/benchmark_task_scores.png](./assets/benchmark_task_scores.png)

Use this to show:
- base vs SFT vs completed GRPO on the main seed block
- the strongest task-level gains from SFT
- the fact that completed GRPO preserved SFT behavior rather than improving it

## 2. Seen vs unseen robustness

File:
- [hack/assets/generalization_seen_vs_unseen.png](./assets/generalization_seen_vs_unseen.png)

Use this to show:
- SFT remains strong on unseen seeds
- GRPO stayed close on most tasks but did not beat SFT
- the submission is not based only on a single seed block

## 3. Safety / failure count

File:
- [hack/assets/safety_failures.png](./assets/safety_failures.png)

Use this to show:
- the base model failed frequently on the hard tasks
- SFT fixed the action-protocol problem
- GRPO preserved safety

## 4. Focused multistage DAPO-loss comparison

File:
- [hack/assets/multistage_dapo_focus.png](./assets/multistage_dapo_focus.png)

Use this to show:
- the HF Jobs GRPO run trained with DAPO loss did not improve `multi_stage_cascade`
- this is a useful honest result, not a negative to hide

Important note:
- this is a task-specific plot, not an all-task GRPO-vs-DAPO comparison
- we do not currently have a separate full-benchmark DAPO-only result that should be plotted as a distinct global model point

## 5. High-level performance vs effort

File:
- [hack/assets/performance_vs_effort.png](./assets/performance_vs_effort.png)

Use this to show:
- why SFT is the main submission model
- why GRPO is still worth mentioning as an engineering extension
- the project-level tradeoff between reliability and extra post-training effort
