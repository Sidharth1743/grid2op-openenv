# Reward Hacking Notes

## Why We Wrote This Note

One of the main lessons from this project is that reward and score can drift apart if the environment only rewards “survive this step” or “choose a valid action” while the real task objective is stronger. That is the core reward-hacking risk in power-grid control as well as in LLM post-training. [1][2][3]

This note records:
- where reward hacking pressure appeared in our system
- how it showed up in practice
- what we changed to reduce it

## Where The Risk Came From

### 1. Fixed-horizon survival can over-reward passivity

With fixed episode caps, an agent can sometimes get decent reward by drifting safely to the horizon instead of actually solving the task. This was most visible in cascade tasks, where repeated `do_nothing` could look acceptable if the grid remained barely survivable. The benchmark still needed fixed horizons for comparability, but that created a real incentive to coast. [4][5]

Relevant code:
- [server/tasks.py](../server/tasks.py)
- [server/environment.py](../server/environment.py)
- [server/graders.py](../server/graders.py)

### 2. One-step verified rewards can overvalue legality

Our verified-candidate setup is deliberately safe: the model only chooses from simulator-checked candidates. That prevents arbitrary invalid actions, but it also means an RL reward can become too focused on:
- valid formatting
- schema correctness
- matching a verified candidate
- avoiding immediate unsafe actions

If those terms dominate, the model learns “be valid and safe” without necessarily learning “best advance the task objective.” This is a classic offline-RL signal problem rather than a bug in the parser. [1][6]

### 3. `single_fault` exposed the mismatch most clearly

For `single_fault`, redispatch often has a real operational cost while `do_nothing` can look cheap locally. If the reward overweights immediate local reward and underweights target completion, the model can settle into behavior that is legal and sometimes stable but still misses the real objective `max_rho < 0.80`.

Relevant code:
- [server/environment.py](../server/environment.py)
- [server/graders.py](../server/graders.py)
- [inference.py](../inference.py)

## How Reward Hacking Showed Up In Practice

We saw three concrete symptoms:

1. High `do_nothing` ratios in hard tasks while benchmark scores were still acceptable.
2. Original offline GRPO preserving the SFT policy instead of learning a stronger one.
3. Cases where step reward looked reasonable but the real task objective was not truly completed.

The clearest example was the original GRPO path:
- it trained cleanly
- it preserved safety
- but it did not improve benchmark performance over SFT

That strongly suggested the reward was not giving the policy enough pressure to move beyond “safe verified selection.”

## What We Changed To Reduce It

### 1. Kept benchmark grading separate from training reward

We did not redefine success by the shaped reward alone. The environment has shaped rewards for training, but the benchmark score remains task-specific and grader-based.

This is important because it prevents us from claiming improvement just because the training reward went up. [5]

Relevant code:
- [server/environment.py](../server/environment.py)
- [server/graders.py](../server/graders.py)

### 2. Added task-specific graders

Instead of a single generic survivability score, each task now has its own grader:
- `single_fault`: target achievement plus final-threshold proximity
- `n_minus_1`: emergency clearing, secure phase-2 operation, reconnection
- `cascade_prevent`: containment, stability, recovery
- `multi_stage_cascade`: stage completion, load preservation, island quality, speed

This was one of the most important anti-hacking changes in the project.

Relevant code:
- [server/graders.py](../server/graders.py)

### 3. Made prompts objective-aware

We changed the prompts so the model is told the real task rule, not just shown candidate outcomes.

Examples:
- `single_fault`: use redispatch, not topology cuts
- `n_minus_1`: clear the emergency window and reconnect safely
- `cascade_prevent`: prioritize lines with active overflow countdowns
- `multi_stage_cascade`: optimize across stages, not just the current step

Relevant code:
- [inference.py](../inference.py)

### 4. Changed shaped rewards to reflect each task’s real structure

The final environment rewards are no longer generic.

Examples:
- `n_minus_1` now has a reconnect bonus and a structured overload term
- `cascade_prevent` penalizes actual overflow countdown pressure and auto-trips
- `multi_stage_cascade` penalizes load loss at stage boundaries and rewards preserved available load

Relevant code:
- [server/environment.py](../server/environment.py)

### 5. Enforced verified-candidate action selection

This is not only a safety measure. It is also an anti-hacking measure. The model cannot invent a superficially attractive but unverified action and still pass evaluation.

Relevant code:
- [ft_inference.py](../ft_inference.py)
- [inference.py](../inference.py)

### 6. Evaluated on unseen seeds

This is the final guardrail against reward-hacking narratives. A model only counts as improved if the benchmark score stays strong on held-out seeds, not just on the seed block seen during development.

Observed SFT result:
- seed `0..4`: strong and safe
- seed `100..102`: still strong and safe

Observed GRPO result:
- technically stable
- did not beat SFT

## What We Learned

The original RL failure was not mainly an infrastructure failure. It was a signal-design problem.

More concretely:
- if reward mostly says “be valid and safe,” RL preserves the current safe policy
- if objective-completion pressure is too weak, RL does not find a stronger control strategy
- if evaluation is stricter than training reward, benchmark score is the truth signal

That is why the final submission keeps:
- SFT as the main model
- GRPO as a real but non-winning extension

## Practical Takeaway

The main anti-reward-hacking lesson from this project is:

> legality, safety, and short-term reward are necessary, but not sufficient; the benchmark must still test objective completion directly.

That principle is now reflected in the current task prompts, environment rewards, and per-task graders.

## References

[1] GRPO trainer documentation and metrics such as `frac_reward_zero_std`:  
https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md

[2] DeepSeekMath and the original GRPO formulation:  
https://huggingface.co/papers/2402.03300

[3] DAPO: an RL system paper emphasizing that loss changes alone do not replace strong training signal design:  
https://arxiv.org/abs/2503.14476

[4] Grid2Op reward documentation:  
https://grid2op.readthedocs.io/en/v1.9.8/reward.html

[5] Grid2Op/L2RPN benchmark framing:  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

[6] RL2Grid benchmark motivation for aligning reward and operational constraints:  
https://huggingface.co/papers/2503.23101
