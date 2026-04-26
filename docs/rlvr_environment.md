# RLVR Environment Design

This project is best understood as an **RLVR-style power-grid environment**:

- the grid physics are simulated explicitly
- the model is not trusted to invent arbitrary control actions
- proposed actions are verified through the simulator
- reward and score are computed from typed, inspectable episode traces

In other words, the environment does not ask an LLM to “guess a good answer.” It asks the model to operate inside a physics-backed control loop with verifiable outcomes.

Core implementation:
- environment adapter: [server/environment.py](../server/environment.py)
- task resets: [server/tasks.py](../server/tasks.py)
- topology and flow analysis: [graph_analysis.py](../graph_analysis.py)
- typed action and state models: [models.py](../models.py)
- inference loop: [inference.py](../inference.py)
- verified-candidate evaluation: [ft_inference.py](../ft_inference.py)

## What RLVR Means Here

In this repository, RLVR does not mean “reward from a human preference model.” It means:

1. the environment exposes a real simulator-backed control problem
2. legal actions are enumerated or verified by the simulator
3. the policy is rewarded only through simulator-grounded consequences
4. final benchmark scores come from explicit, inspectable grader logic

That makes the feedback loop verifiable in a way that free-form text generation is not. [1][2]

## Physics Simulation Layer

The environment is built on **Grid2Op**, which provides:

- power-flow simulation
- thermal loading values `rho`
- line status and cooldown tracking
- redispatch constraints
- automatic progression of stressed scenarios

At every step, the simulator tells us:

- which lines are overloaded
- whether an action converged
- whether a line disconnected
- how redispatch changed the state
- whether the grid remained operational

The simulator is therefore the ground truth for:

- legality
- safety
- short-horizon action consequences

That is the foundation that makes this an RLVR environment instead of a pure prompting benchmark. [2][3]

## Verified-Action Loop

The environment uses a verified-candidate loop rather than unconstrained action invention.

High-level flow:

1. observe the current grid state
2. enumerate or construct legal candidate actions
3. simulate those candidates with Grid2Op
4. present the model with verified candidate outcomes
5. require the selected action to match one verified candidate exactly
6. execute the chosen action in the live episode

That means the model is evaluated on:

- choosing well among verified options
- respecting the task objective
- remaining safe under constrained control

and not on:

- hallucinating arbitrary topology changes
- inventing impossible redispatch values
- exploiting a text-only reward shortcut

This verified-candidate design is one of the most important quality controls in the whole project.

## Graph Intelligence Layer

The environment does not expose raw simulator arrays alone. It also computes structured **graph intelligence** from the live grid topology.

Current graph analysis outputs include:

- `bridge_lines`
- `safe_to_disconnect`
- `n_minus_1_critical_lines`
- `parallel_groups`
- `high_centrality_buses`
- `islanded_clusters`
- `congestion_corridor`
- `flow_clusters`
- `stressed_lines`
- `graph_density`

These are computed in:
- [graph_analysis.py](../graph_analysis.py)

What they mean operationally:

- `bridge_lines`: disconnecting these lines would split the active network
- `safe_to_disconnect`: connected lines whose removal does not immediately fragment the active graph
- `high_centrality_buses`: structurally important buses in the active topology
- `islanded_clusters`: disconnected graph components that already exist
- `congestion_corridor`: a compact summary of likely exporter-to-importer stress direction
- `stressed_lines`: the highest-loading lines with overflow context

This graph layer helps the model reason about:

- whether a topology cut is structurally dangerous
- where congestion is flowing through the grid
- which lines are bottlenecks rather than just noisy symptoms

That is a major improvement over exposing only flat `rho` arrays.

## Why Graph Intelligence Matters For LLMs

An LLM is much better at using structured relational hints than at reverse-engineering a power-network graph from raw vectors alone.

For example:

- “line 17 is a bridge line” is a meaningful control warning
- “safe_to_disconnect = [3, 8, 10]” narrows the safe topology search space
- “export buses [2, 5] -> import buses [9, 12] via lines [7, 11, 18]” gives a directional congestion picture

These are the kinds of abstractions human operators also use:

- bottlenecks
- critical corridors
- reconnection windows
- islands

So graph intelligence is not decorative metadata. It is the environment’s way of making the physics legible to the model.

## Why This Is Stronger Than Free-Form Prompting

A purely prompt-based grid benchmark would let the model:

- emit invalid JSON
- invent actions outside the simulator action set
- appear “smart” in text while failing operationally

This project blocks that path:

- actions are typed
- candidate outcomes are simulator-verified
- unsafe or malformed actions do not get free credit
- the final score comes from explicit graders

That makes the benchmark much closer to a real control stack and much harder to game.

## Where RL Fits

SFT and GRPO both operate inside this same RLVR environment design.

SFT learns:

- the action protocol
- how to map structured verified candidates to good control decisions

GRPO then attempts to improve policy quality while preserving:

- action validity
- candidate fidelity
- simulator-grounded reward structure

The important point is that RL is not happening in a vacuum. It is layered on top of:

- physics simulation
- verified candidates
- graph-aware prompts
- task-specific graders

That is why the environment is a stronger research platform than a plain text-only RL loop.

## Core Design Summary

The environment combines Grid2Op physics, graph-derived topology intelligence, verified candidate simulation, typed action enforcement, and task-specific grading into an RLVR-style control loop for LLMs.

## References

[1] OpenEnv integration for training and evaluation:  
https://huggingface.co/docs/trl/openenv

[2] Grid2Op documentation:  
https://grid2op.readthedocs.io

[3] Learning to run a power network challenge for training topology controllers:  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

[4] RL2Grid benchmark framing:  
https://huggingface.co/papers/2503.23101

[5] Local implementation:  
[server/environment.py](../server/environment.py), [server/tasks.py](../server/tasks.py), [graph_analysis.py](../graph_analysis.py), [models.py](../models.py), [inference.py](../inference.py), [ft_inference.py](../ft_inference.py)
