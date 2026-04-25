# Task 2 — `n_minus_1`

## Purpose

Task 2 tests post-contingency control rather than single-line repair. One line is already disconnected at reset, and the agent has to operate the degraded topology safely for `20` steps without blacking out the grid. This is closer to classical N-1 security reasoning than Task 1, where the topology is still intact. [1][2]

Current task definition in code:
- description and horizon: [server/tasks.py](../server/tasks.py)
- reward: [server/environment.py](../server/environment.py)
- grader: [server/graders.py](../server/graders.py)

## What We Changed

The earlier version of Task 2 behaved too much like a generic survivability problem. We updated it to better reflect the actual N-1 objective.

Main changes:
- added explicit emergency and steady-state phases in the prompt
- added reconnection guidance for the faulted line
- added structural graph guidance through `n1_security_score` and bridge-line analysis
- changed the reward to combine survival, overload margin, redispatch cost, and a reconnect bonus
- changed the grader to score emergency clearing, sustained secure operation, and reconnection success separately

Relevant prompt and candidate logic:
- [inference.py](../inference.py)
- [graph_analysis.py](../graph_analysis.py)

## Current Implementation

Reset:
- line `0` is disconnected at reset
- episode length is `20` steps

Prompt-side task rules now include:
- emergency thresholding with `rho >= 0.92`
- steady-state target with `rho < 0.90`
- reconnect guidance once cooldown allows
- explicit warning that passive `do_nothing` is not enough in the emergency window

Current reward shape:
- constant survival term
- clipped thermal-margin term over line loadings
- redispatch cost penalty
- reconnect bonus when a reconnection succeeds without worsening the state too much
- strong positive terminal bonus for surviving the full horizon
- strong negative penalty for blackout

Current grader:
- `30%` emergency response quality
- `50%` sustained security in phase 2
- `20%` reconnection achievement

This is much closer to how N-1 security is discussed in the Grid2Op and L2RPN ecosystem, where reconnecting safely and keeping the degraded topology secure matters more than merely surviving step to step. [1][3][4]

## What We Observed

This task improved materially after the prompt and ranking updates.

Final SFT result, seed block `0..4`:
- `0.990`

Final SFT result, unseen seeds `100..102`:
- `0.9222223`

Key improvement:
- earlier runs overused `do_nothing`
- the final system became more active, using redispatch and reconnect actions more effectively

Main action counts for the final SFT model on seed block `0..4`:
- `do_nothing=16`
- `reconnect_line=5`
- `redispatch=79`

## Current Limitation

Task 2 is now in a good place compared with the other tasks. The main remaining risk is that it still uses a simplified structural proxy for some N-1 reasoning, not a full online contingency analysis engine. That is acceptable for the hackathon benchmark, but it is worth stating clearly.

## References

[1] Grid2Op reward documentation and built-in reward classes:  
https://grid2op.readthedocs.io/en/v1.9.8/reward.html

[2] Learning to run a power network challenge for training topology controllers:  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

[3] Grid2Op `LinesReconnectedReward` documentation:  
https://grid2op.readthedocs.io/en/v1.10.5/_modules/grid2op/Reward/linesReconnectedReward.html

[4] L2RPN 2023 winning-agent writeup with greedy reconnection discussion:  
https://lajavaness.medium.com/how-we-built-the-winning-real-time-autonomous-agent-for-power-grid-management-in-the-l2rpn-41ab3cfaddbd
