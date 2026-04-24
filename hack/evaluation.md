# Evaluation Notes

This document summarizes the current base-model versus SFT-model evaluation for the Grid2Op OpenEnv fine-tuning work.

## Evaluation Setup

Both models were evaluated through the same `ft_inference.py` verified-candidate pipeline:

1. Reset a task episode.
2. Enumerate legal Grid2Op actions.
3. Simulate candidates through the environment.
4. Prompt the model with verified candidate outcomes.
5. Require the model to output a valid `GridAction` JSON.
6. Validate that the selected action exactly matches one verified simulation candidate.
7. Execute the action and grade the episode.

This means the comparison is not between different environments or different action filters. The base model and SFT model see the same style of prompt and must obey the same verified-action constraint.

Models:

- Base: `Qwen/Qwen3-4B-Instruct-2507`
- SFT: `Qwen/Qwen3-4B-Instruct-2507` + LoRA adapter `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Evaluation command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python ft_inference.py \
  --base-url http://127.0.0.1:8018 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter <none-or-sft-adapter> \
  --task-id single_fault n_minus_1 cascade_prevent multi_stage_cascade \
  --episodes-per-task 5 \
  --seed-start 0 \
  --precision auto \
  --attn-implementation kernels-community/flash-attn2
```

Logs analyzed:

- Base: `outputs/logs/base_qwen3_4b_seed0_5eps.log`
- SFT: `outputs/logs/sft_3k_v1_seed0_5eps.log`

Analysis command:

```bash
python scripts/check_ft_inference_log.py <log-path>
```

## Score Comparison

| Task | Base Mean Score | SFT Mean Score | Interpretation |
|---|---:|---:|---|
| `single_fault` | `0.856` | `0.856` | Same score |
| `n_minus_1` | `0.952` | `0.990` | SFT improves reliability and reaches max score across seeds |
| `cascade_prevent` | `0.000` | `0.990` | SFT is dramatically better |
| `multi_stage_cascade` | `0.000` | `0.916` | SFT is dramatically better |

The most important result is not just higher mean score. It is that the base model fails complex cascade tasks due to invalid or unverified actions, while the SFT model completes every evaluated episode.

## Safety And Validity

| Metric | Base | SFT |
|---|---:|---:|
| Episodes | `20` | `20` |
| Failures | `10` | `0` |
| Errored steps | `0` | `0` |
| Safety pass | `false` | `true` |

The base model failed `10/20` episodes. Most failures came from selecting actions outside `verified_simulation_results` or emitting invalid action structure.

Examples of base failures:

- selected reconnect actions that were not in verified candidates,
- emitted redispatch in the wrong shape,
- selected compound or unavailable actions,
- selected line actions not present in the candidate list.

The SFT model had:

```text
failure_count: 0
errored_step_count: 0
invalid_step_counts: 0
```

This shows fine-tuning substantially improved action-format reliability and adherence to the verified-action protocol.

## Action Distribution

Base action counts:

```text
single_fault:
  redispatch: 46

n_minus_1:
  reconnect_line: 4
  redispatch: 96

cascade_prevent:
  disconnect_line: 4
  reconnect_line: 2

multi_stage_cascade:
  disconnect_line: 3
  do_nothing: 5
  reconnect_line: 4
  redispatch: 2
```

SFT action counts:

```text
single_fault:
  do_nothing: 2
  redispatch: 44

n_minus_1:
  do_nothing: 16
  reconnect_line: 5
  redispatch: 79

cascade_prevent:
  disconnect_line: 10
  do_nothing: 132
  reconnect_line: 8

multi_stage_cascade:
  disconnect_line: 7
  do_nothing: 126
  reconnect_line: 17
```

The SFT model is more conservative on cascade tasks, but this conservatism is useful: it avoids invalid or unsafe actions and survives the full horizon.

## Interpretation

The fine-tuned model did not improve every task equally. `single_fault` stayed at the same mean score as the base model. However, the overall result strongly supports fine-tuning because the SFT model is reliable across all tasks:

```text
Base model: sometimes strong on simpler tasks, unstable on complex tasks.
SFT model: stable, valid, verified-action behavior across all tasks.
```

The strongest evidence for fine-tuning benefit is:

- base model cascade score: `0.0`
- SFT cascade score: `0.99`
- base multi-stage score: `0.0`
- SFT multi-stage score: `0.916`
- base failures: `10`
- SFT failures: `0`

This means SFT learned the environment-specific action protocol and avoided the invalid/unverified decisions that break the base model.

## Single-Fault Limitation

`single_fault` remains the weak improvement area. The base and SFT model both average:

```text
single_fault mean_score = 0.856
```

Trace inspection showed that, for some seeds, the available one-step redispatch candidates do not push `max_rho` below the task target of `0.80`. Even strict redispatch candidate exposure could not always reach the target.

This suggests the limitation may be in the current candidate/action space rather than only the model. Future work should test compound redispatch candidates and/or multi-step lookahead for `single_fault`.

## N-1 Improvement

Earlier SFT evaluation had `n_minus_1` stuck around `0.60` because the model chose mostly `do_nothing`.

The fix was to make n-1 candidate ranking and prompt objectives threshold-aware:

- steps 1-5: prioritize clearing `max_rho < 0.92`,
- steps 6-20: prioritize keeping `max_rho < 0.90`,
- prefer safe reconnect,
- show redispatch candidates when they help threshold clearing.

After this, `n_minus_1` improved to:

```text
mean_score = 0.990
```

The final SFT action distribution for `n_minus_1` over 5 episodes:

```text
do_nothing: 16
reconnect_line: 5
redispatch: 79
```

This is a healthy active-control profile.

## Unseen-Seed Evaluation

The SFT model was also evaluated on unseen seeds:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python ft_inference.py \
  --base-url http://127.0.0.1:8018 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/models/grid2op-qwen3-4b-sft-3k-v1 \
  --task-id single_fault n_minus_1 cascade_prevent multi_stage_cascade \
  --episodes-per-task 3 \
  --seed-start 100 \
  --precision auto \
  --attn-implementation kernels-community/flash-attn2 \
  2>&1 | tee outputs/logs/sft_3k_v1_seed100_3eps.log
```

Results:

| Task | Mean Score | Min Score | Max Score | Interpretation |
|---|---:|---:|---:|---|
| `single_fault` | `0.830` | `0.750` | `0.990` | Mixed; still limited by single-fault candidate/action space |
| `n_minus_1` | `0.922` | `0.787` | `0.990` | Strong generalization, one weaker seed |
| `cascade_prevent` | `0.990` | `0.990` | `0.990` | Excellent |
| `multi_stage_cascade` | `0.907` | `0.895` | `0.913` | Strong and stable |

Safety:

```text
failure_count: 0
errored_step_count: 0
invalid_step_counts: 0
safety_pass: true
```

Action counts on unseen seeds:

```text
single_fault:
  redispatch: 30

n_minus_1:
  do_nothing: 17
  reconnect_line: 3
  redispatch: 40

cascade_prevent:
  disconnect_line: 2
  do_nothing: 83
  reconnect_line: 5

multi_stage_cascade:
  disconnect_line: 2
  do_nothing: 80
  reconnect_line: 8
```

This is important because it addresses overfitting risk. It does not prove that the model cannot overfit, but it shows that the learned policy transfers to unseen seeds without invalid-action failures.

The main generalization finding:

```text
The SFT model remains safe and high-scoring on unseen seeds.
```

Important nuance:

```text
The SFT model generalizes the verified-action protocol better than the base model, but single_fault still has objective-performance gaps.
```

On unseen seeds, `single_fault` produced only `redispatch` actions:

```text
single_fault action_counts:
  redispatch: 30
```

So the weak `single_fault` scores are not caused by the model passively choosing only `do_nothing`. The current problem is that the active redispatch choices available to the model often reduce or manage loading, but do not reliably clear the task target of `max_rho < 0.80`.

## Single-Fault Improvement Plan

`single_fault` should not be treated as a reward-maximization problem. Its core objective is:

```text
bring all line loadings below max_rho < 0.80
```

The current one-step reward can prefer actions that are locally cheap, but the grader cares about satisfying the thermal target. Therefore, the candidate selector for `single_fault` should prioritize target achievement and thermal reduction, not immediate `simulated_reward`.

This explains why logs can look confusing:

```text
step reward can be small or even negative
episode score can still be high or low depending on whether the task objective is achieved
```

For `single_fault`, a redispatch action may reduce thermal risk but receive a worse immediate reward than `do_nothing` because redispatch has an operational cost. That does not automatically make it the wrong action. The real question is whether it moves the grid toward `max_rho < 0.80`.

Current problem:

- On some seeds, SFT selects redispatch every step but still scores only `0.75`.
- Trace inspection showed that the best one-step redispatch candidates often do not reach `max_rho < 0.80`.
- Therefore, the bottleneck is likely the candidate/action space, not just the model.
- The seed-100 run confirms this: `single_fault` used active redispatch on all 30 steps across 3 episodes, but still averaged only `0.830`.

Observed trace pattern:

```text
current max_rho: ~0.826 to ~0.86
best one-step candidate max_rho: still ~0.83 to ~0.86
target: <0.80
```

So even a correct model cannot choose a candidate that is not present.

Recommended improvements:

1. Add compound redispatch candidates.

Current candidate generation mostly offers one-generator redispatch actions. `single_fault` may need coordinated multi-generator actions, for example:

```json
{"redispatch": {"5": -15.0, "1": 10.0}}
{"redispatch": {"5": -15.0, "0": 5.0}}
{"redispatch": {"1": 10.0, "0": -5.0}}
```

These should be generated from redispatch bounds and sensitivity guidance, not hardcoded to specific lines.

The generator should build combinations mechanically:

```text
pick top redispatch generators from usable sensitivity guidance when present
otherwise use redispatchable_generators and allowed_deltas
simulate pairwise combinations that respect each generator bound
keep only candidates that are legal and converge
rank by target achievement first
```

2. Rank candidates by objective distance.

For `single_fault`, ranking should use:

```text
primary: whether max_rho < 0.80
secondary: lowest max_rho
tertiary: lower redispatch magnitude
last: simulated_reward
```

The reward should be a tie-breaker, not the objective.

3. Add two-step lookahead diagnostics before retraining.

Some actions may not satisfy the target in one step but may create a better state for the next step. A diagnostic should test whether repeated/compound redispatch can ever reach the 0.80 target for difficult seeds.

The diagnostic should answer this directly:

```text
For each weak single_fault seed, is there any legal one-step or two-step action sequence that reaches max_rho < 0.80?
```

If yes, expose those candidate patterns to the SFT inference prompt and add similar rows to the dataset. If no, then the task may require a broader action space or a different environment-side intervention.

4. Keep `do_nothing` only after stabilization.

`do_nothing` is useful when:

```text
max_rho <= 0.80
```

or when no active redispatch improves the next simulated state. It should not be preferred above target simply because it has higher immediate reward.

5. Do not retrain until candidate diagnostics prove target reachability.

If no candidate or short action sequence can push the grid below `0.80`, more SFT data will not solve the task. The action generator must first expose target-achieving actions.

## Next Single-Fault Experiment

Before another SFT run, run a candidate-space diagnostic for `single_fault`:

```text
goal: find whether legal compound redispatch can clear max_rho < 0.80
scope: same seeds where current SFT scores 0.70 or 0.75
compare: current single redispatch vs compound redispatch vs two-step lookahead
success condition: at least one verified candidate or short sequence reaches max_rho < 0.80
```

If compound redispatch works, the next code change should be:

- add general compound redispatch candidate generation to `ft_inference.py`,
- add the same candidate family to teacher dataset collection,
- regenerate targeted `single_fault` rows,
- rebalance so `single_fault` includes more objective-clearing redispatch labels,
- retrain or continue SFT from the current adapter.

If compound redispatch does not work, changing labels will not help. The model cannot learn to select a target-clearing action that the candidate generator never provides.

## Single-Fault Diagnostic Result

The action-space diagnostic was extended to support:

- legal `single_fault` action filtering,
- pairwise compound redispatch candidates,
- objective-first sorting by `max_rho < 0.80`,
- optional depth-2 lookahead by replaying the best first-step actions.

Diagnostic logs:

```text
outputs/logs/single_fault_action_space_seed0_onestep.log
outputs/logs/single_fault_action_space_seed0_depth2.log
outputs/logs/single_fault_action_space_seed0_full_redispatch_onestep.log
```

Seed tested:

```text
task_id=single_fault
benchmark_tier=single_fault_easy
seed=0
initial_max_rho=0.826042
target=max_rho < 0.80
```

One-step compound redispatch result:

```text
candidate_count=61
target_reached_count=0
best_max_rho=0.830701
best_actions included:
  redispatch {"5": -15.0}
  redispatch {"0": 5.0, "1": 10.0}
  redispatch {"0": 5.0, "5": -15.0}
  redispatch {"1": 10.0, "5": -15.0}
```

Depth-2 result, expanding the two best first actions:

```text
target_reached_count=0
lookahead_target_reached_count=0
best_first_step_max_rho=0.822918
best_second_step_max_rho=0.833879
```

Full one-step redispatch-space result, including all three redispatchable generators:

```text
candidate_count=125
max_compound_redispatch_size=3
target_reached_count=0
best_max_rho=0.830701
best_full_compound_action=redispatch {"0": 5.0, "1": 10.0, "5": -15.0}
```

Interpretation:

```text
For this weak single_fault seed, legal single, pairwise, and three-generator redispatch actions do not expose a target-clearing candidate.
```

This means the `single_fault` score ceiling is probably not caused by the SFT model choosing the wrong candidate. The model cannot select a candidate that reaches `max_rho < 0.80` because the candidate generator does not currently produce one.

The next useful diagnostic is to test more weak seeds and longer action sequences:

- test seeds that scored `0.70` or `0.75`,
- test longer scripted redispatch sequences,
- inspect whether the environment allows reaching `<0.80` at all under legal redispatch constraints.

## Current Conclusion

The SFT model is better than the base model for this environment because it is reliable and task-safe:

- no invalid action failures,
- no verified-candidate selection failures,
- strong cascade performance,
- improved n-1 performance,
- stable full-horizon execution.

The unseen-seed run makes this a defensible training-improvement result for the hackathon demo. The remaining work is not basic action validity; it is improving the `single_fault` candidate/action space so the model can pursue the actual thermal target.
