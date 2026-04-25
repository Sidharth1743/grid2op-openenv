# Environment Design

## Current Environment Framing

This project uses a Grid2Op-based control benchmark wrapped through OpenEnv-style tasks. The agent does not interact with the raw simulator directly. Instead, it receives a structured task prompt that summarizes the current grid condition in a control-oriented format.

The environment should be understood in this order:

## 1. What The Agent Observes

The agent observes a compact state summary, not the full simulator internals.

Typical observation fields include:
- `task_id`
- current step and horizon
- `max_rho`
- stressed lines and overflow counters
- disconnected lines
- redispatchable generators and allowed deltas
- graph or topology intelligence
- sensitivity guidance when available

In teacher-data and constrained-selection pipelines, the prompt may also include verified simulation results for a small candidate set. That turns the problem from unconstrained action invention into selection among simulator-checked actions.

This is useful because it keeps the model focused on operational decisions instead of forcing it to reconstruct power-system structure from raw tensors.

## 2. What Actions The Agent Can Take

The action space is task-dependent, but the main action families are:
- `do_nothing`
- `redispatch`
- `disconnect_line`
- `reconnect_line`

Each task restricts which actions are legal:
- `single_fault`: mostly `redispatch` and `do_nothing`
- `n_minus_1`: `redispatch`, `reconnect_line`, `do_nothing`
- `cascade_prevent`: may include redispatch and topology changes
- `multi_stage_cascade`: may include redispatch and topology changes

For safe inference, the best setup is to make the model choose from verified candidate actions rather than invent arbitrary actions outside the simulator-checked set.

## 3. What Ends An Episode

Right now, an episode ends in one of three ways:
- the fixed task horizon is reached
- the grid reaches a terminal failure state
- the runner stops because the model output is invalid or unusable

Current fixed horizons are:
- `single_fault`: 10 steps
- `n_minus_1`: 20 steps
- `cascade_prevent`: 30 steps
- `multi_stage_cascade`: 30 steps

This means the benchmark is fundamentally a fixed-horizon survivability task.

## 4. How Reward Is Computed

There are two reward layers in the project.

### Environment Reward

The simulator and task grader provide the official benchmark reward. In practice this usually looks like:
- modest per-step rewards for maintaining acceptable operation
- a larger terminal reward when the task survives to the horizon
- bad terminal outcomes when the grid fails

This is the reward that determines benchmark score.

### Training Reward

For GRPO, we do not directly trust raw environment reward as the only signal. We add a verifier-style reward layer that checks:
- output format correctness
- schema validity
- exact match to a verified candidate action
- safety of the chosen action
- alignment with task objective
- anti-hacking penalties

So the benchmark reward stays official, while RL training uses a safer research-oriented reward proxy.

## 5. How Abuse, Loops, And Cheating Are Controlled

Several controls already exist:
- fixed horizons prevent infinite loops
- task-specific legality rules block forbidden actions
- parser and schema validation reject malformed outputs
- verified candidate matching prevents arbitrary action invention
- simulator verification filters out unsafe or non-convergent actions
- log-based checks track failures, invalid outputs, and unsafe terminal behavior

This gives a reasonable baseline against reward hacking, but it is not perfect.

## What Is Good About Hard Episode Caps

The hard cap is not arbitrary. It gives real benefits:
- every team is evaluated on the same finite decision window
- scores are comparable across runs
- the task stays focused on survivability under disturbance
- the agent cannot inflate reward by dragging the episode on forever

For a hackathon benchmark, this is a sensible default.

## What Is Bad About Hard Episode Caps

The hard cap also creates predictable distortions:
- a policy can learn to coast to the horizon instead of truly stabilizing the grid
- `do_nothing` can look deceptively good in already survivable states
- the benchmark may under-reward extra safety margin once survival is already secured
- two policies can get similar scores even if one is much more robust

This is exactly why our analysis kept surfacing high `do_nothing` ratios in some tasks despite acceptable scores.

## Better Ways To End An Episode

We should not remove the benchmark horizon for official evaluation, because comparability matters. But we can improve the environment logic in research mode and in our diagnostics.

### Option 1: Success Early-Stop

End the episode early if the grid has been stably healthy for a sustained window.

Example:
- all lines below a task threshold for 3 to 5 consecutive steps
- no active overflow countdown
- no disconnected line that should still be restored

This would reward agents that solve the problem quickly instead of merely surviving until timeout.

### Option 2: Stability Margin Early-Stop

End the episode early when the agent not only survives, but creates durable margin.

Example conditions:
- `max_rho < 0.75` for several consecutive steps
- no topological fragility signal
- no pending emergency condition

This is stronger than simple survival because it requires the agent to move the grid into a genuinely safer state.

### Option 3: No-Progress Termination

Terminate the episode if the agent is trapped in repetitive non-improving behavior.

Example:
- repeated `do_nothing` or equivalent low-impact actions
- no improvement in `max_rho`
- no restoration progress
- no countdown reduction

This is useful in research mode to expose policy collapse and “survive by drifting” behavior.

### Option 4: Recovery-Phase Split

Treat emergency control and steady-state recovery as separate sub-phases.

Example:
- phase A ends when emergency overload is cleared
- phase B evaluates whether the policy maintains safe operation and restores topology

This is attractive for `n_minus_1` and cascade tasks because it separates “stop the fire” from “restore normal operation”.

### Option 5: Variable Horizon With Ceiling

Keep a maximum horizon, but let the environment stop earlier if success or failure becomes obvious.

Example:
- minimum required control window: 5 steps
- early success if grid is stably safe
- hard maximum at 20 or 30 steps

This is probably the best compromise if we want richer behavior without losing reproducibility.

## Recommended Direction

For the hackathon submission, keep the official benchmark horizon unchanged.

For our internal research pipeline, add two extra diagnostics:
- `target_reached_count`: whether the core task threshold was actually achieved
- `stable_recovery_count`: whether the grid stayed under a stronger safety margin for several consecutive steps

If we later build a research-only environment variant, the safest upgrade path is:
1. retain the hard maximum horizon
2. add early success termination for sustained safe recovery
3. add no-progress detection for repeated non-improving behavior

That preserves benchmark comparability while making training and analysis much less vulnerable to passive survival hacks.

## Practical Interpretation

Right now, a high score means:
- the policy usually survived the benchmark window

It does not always mean:
- the policy truly stabilized the system as strongly as possible
- the policy chose the best control strategy
- the policy avoided lazy `do_nothing` behavior whenever action was available

That is why our project needs both:
- benchmark evaluation
- additional safety and objective diagnostics

The benchmark score tells us if the agent survives.
The extra diagnostics tell us whether the agent is actually behaving like a competent operator.
