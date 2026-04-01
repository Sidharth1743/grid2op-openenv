Now I have everything I need. Let me synthesize what the literature says and design Task 2 properly.

---

## What the literature tells us about N-1 specifically

Several key findings from the sources above:

The multi-agent paper introduces a new action space specifically designed to ensure structural N-1 security — meaning the grid remains operational and avoids automatic blackout in the event of a single contingency such as an unplanned powerline outage.

RL2Grid defines a three-component reward: `R = α·R_survive + β·R_overload + η·R_cost`, where `R_survive` is a constant per-step survival bonus, `R_overload` measures line loading margins, and `R_cost` assigns economic cost to redispatching actions and penalizes energy losses.

The L2RPN literature uses a reward combining three weighted components: a constant per-step survival bonus encouraging sustained operation, a line loading component minimizing overloads and disconnections, and an economic cost for redispatching.

A critical insight from the L2RPN survey: after a line disconnection by overcurrent, lines face a cooldown before reconnection. A key real-world behavior is that disconnected lines should be reconnected once conditions allow — this is the "greedy reconnection" heuristic used by winning agents. The winning agent from L2RPN 2023 specifically implements a greedy reconnection module that evaluates which reconnection most improves the grid's state.

The 2023 winning agent used two thresholds: `ρ_danger` (when to act urgently) and `ρ_safe` (target state after acting), which are the primary hyperparameters controlling the balance between redispatching, curtailment, and topological action.

The ICLR 2021 winning agent activates only when `ρ_max ≥ δ_h = 0.9` and stacks the last 6 states as input to capture temporal correlation — a key insight: N-1 contingency management requires understanding trajectory, not just the current snapshot.

---

## What makes N-1 fundamentally different from single_fault

This is the conceptual core you need to internalize before designing the reward.

In `single_fault`, there is one stressed line in an otherwise intact topology. The optimal action is usually local — fix the one problem line.

In `n_minus_1`, one line is already **gone**. The topology is permanently changed for the duration of the episode. The agent cannot restore the original grid — it must find a **new stable operating point** for a degraded topology. The literature calls this "post-contingency redispatch" or "corrective control." The objective shifts from "fix this line" to "operate safely in a degraded topology for 20 steps while the chronic keeps evolving."

This distinction drives everything about the reward and the grader.

---

## The core problem with your current n_minus_1 reward

Your current reward:

```
safe_line_ratio × 0.1       ← safe step bonus
delta_rho ±0.05             ← trending signal
overload_penalty −0.3       ← per overloaded line
survival_bonus +3.0         ← terminal
blackout_penalty −8.0       ← terminal
```

**What's wrong**:

**First — no reconnection incentive.** When the faulted line (line 0) disconnects, other lines become overloaded. The natural corrective action in the literature is to reconnect line 0 once its cooldown expires (`time_before_cooldown_line[0] == 0`). Your reward has zero signal for this. An agent that reconnects line 0 at step 3 gets the same per-step reward as one that ignores it. The 2023 winning agent's greedy reconnection module exists precisely because this signal is missing from the base reward.

**Second — delta_rho is noisy and reward-hackable.** The `±0.05` delta signal rewards trending toward safety but also rewards oscillating actions — disconnect a line (rho goes up slightly, -0.05), reconnect it (rho goes down, +0.05), repeat. The net reward is 0.0 but the agent has wasted two steps and two cooldown cycles. This is the oscillation problem from a different angle.

**Third — the survival ratio grader conflates two different failure modes.** An agent that dies at step 17 because it never reconnected line 0 and the chronic finally pushed a remaining line over 100% scores `17/20 = 0.85`. An agent that died at step 17 because it made a catastrophic topology error at step 16 also scores `0.85`. These are fundamentally different failure modes — one is passive (didn't act), one is active (acted badly). The grader doesn't distinguish them.

---

## Redesigned Task 2 — N-1 Contingency Management

### What the scenario tests

The real N-1 problem is a **two-phase** challenge that maps directly to how grid operators actually work:

**Phase 1 — Emergency response (steps 1–5)**: The line just tripped. Several remaining lines are now above 80% loading. The agent must take corrective actions immediately — topology changes and/or redispatch — to bring all lines below the `ρ_danger` threshold.

**Phase 2 — Sustained secure operation (steps 6–20)**: Emergency addressed. Now the agent must maintain N-1 secure operation — meaning the current degraded topology must be able to withstand **one more** line trip without blackout. This is the structural N-1 security concept from the multi-agent paper. The agent should also reconnect the faulted line when its cooldown expires.

This two-phase structure is what the literature is modeling. Your current task only has Phase 2 implicitly.

---

### Revised reward function — three-component RL2Grid structure

Adopt directly from RL2Grid's `R = α·R_survive + β·R_overload + η·R_cost`:

**Component 1 — `R_survive` (survival signal)**

Constant `+1.0` per step the agent is alive. Normalized by `max_steps` at episode end.

This is the simplest component but critically important — it ensures the agent always has a dense positive signal for staying alive. Without this, early episodes with all-negative rewards cause the agent to give up.

```
R_survive = +1.0 per step (always)
```

**Component 2 — `R_overload` (loading margin signal)**

The scaled L2RPN reward used by virtually every competitive paper. Not the `(1-ρ²)` from the PG-RL paper — that's for Task 1 where you want speed. For N-1, you want margin quality over the full 20-step horizon.

From the multi-agent paper: the scaled L2RPN reward calculates the squared margin of the current flow relative to the current limit for each powerline — the larger the margin, the higher the reward, encouraging states where powerlines are not operating near their maximum capacity.

```
R_overload = (1/n_line) × Σ_ℓ clip(1 - ρ_ℓ, -1, 1)
```

Note: `clip` at -1 means a line at 200% loading doesn't produce -100 and dominate the signal. Clipping keeps it bounded at `[-1, +1]` per line, summing to `[-1, +1]` normalized.

**Component 3 — `R_cost` (economic cost signal)**

The cost component assigns an economic cost to redispatching actions and penalizes energy losses, ranging in `[-1, 0]`.

For your system: any redispatch action incurs a cost proportional to the absolute MW change. Topology-only actions are free (as the literature consistently notes, topology is the "zero-cost" action preferred by operators).

```
R_cost = -λ × Σ_g |delta_mw_g| / max_ramp_g   (normalized by ramp limit)
λ = 0.05 for Task 2  (mild cost — N-1 sometimes requires redispatch)
```

**Reconnection bonus — novel addition from heuristic literature**

Not in RL2Grid but in every winning L2RPN agent. Explicit reward for reconnecting a previously faulted line when conditions allow:

```
R_reconnect = +2.0 if agent successfully reconnects a disconnected line this step
             AND reconnection doesn't increase ρ_max by more than 0.1
```

The condition prevents reward hacking via reconnecting lines that worsen loading.

**Total reward**:

```
R[n] = α·R_survive + β·R_overload + η·R_cost + R_reconnect

weights: α=0.3, β=0.6, η=0.1
```

The weights reflect the primary objective hierarchy: overload management > survival > cost minimization.

**Terminal signals** (not per-step, applied once):

```
Survival bonus: +10.0 × (steps_survived / max_steps)² — quadratic to reward full survival more
Blackout penalty: -15.0
```

Quadratic survival bonus: surviving 18/20 steps gives `10 × 0.81 = 8.1`. Surviving 20/20 gives `10.0`. This creates a strong incentive for the last few steps that a linear formula doesn't provide.

---

### N-1 security check — the structural component from the multi-agent paper

Structural N-1 security means the grid remains operational and avoids automatic blackout in the event of a single contingency — an unplanned powerline outage.

Add a per-step **N-1 security score** to the observation and the prompt. Using your graph analysis module, for each connected line, simulate its removal in NetworkX and check if the graph remains connected. Count the fraction of lines whose removal would disconnect the graph (bridge lines). This is a graph-only operation — no power flow needed.

```
n1_security_score = 1.0 - (bridge_line_count / n_connected_lines)
```

A score of 1.0 means no single line trip would island the grid (maximum structural security). A score of 0.5 means half the lines are bridges (highly vulnerable).

Include this in the observation and in the LLM prompt: "Current N-1 structural security: 0.73 (8 of 19 connected lines are safe to trip). Bridge lines that would cause islanding: [4, 11, 15]."

This is the key information an N-1 operator needs that your current prompt doesn't provide.

---

### Two-threshold activation from ICLR 2021 winner

The winning agent only activates when `ρ_max ≥ 0.9` — below this threshold, do-nothing is the optimal action since acting risks worsening the grid without benefit.

Add an explicit **activation threshold** to Task 2:

```
ρ_danger = 0.92   (act urgently — any line above this is an emergency)
ρ_safe   = 0.80   (target — all lines below this = episode in safe mode)
```

In the LLM prompt, replace the current "lines above 80%" framing with:

- **EMERGENCY**: Lines above `ρ_danger=0.92` — must act this step
- **WARNING**: Lines between 0.80–0.92 — monitor, consider acting
- **SAFE**: Lines below 0.80 — no action needed

This maps directly to how grid operators think and gives the LLM a clear decision framework.

---

### Revised grader — phase-aware

The current survival ratio `steps_survived / 20` doesn't capture quality of operation. Replace with a three-component grader:

**Component A — Emergency response quality (30% weight)**

Did the agent bring all lines below `ρ_danger = 0.92` within the first 5 steps?

```
emergency_score = max(0, 1.0 - 0.2 × steps_to_reach_safe_state)
  → Cleared in step 1: 1.0
  → Cleared in step 5: 0.2 (minimum passing)
  → Never cleared: 0.0
```

**Component B — Sustained secure operation quality (50% weight)**

Among steps 6–20, what fraction had `ρ_max < 0.90`?

```
security_ratio = safe_steps_in_phase2 / 15
```

This directly measures the quality of post-contingency operation, not just survival.

**Component C — Reconnection achievement (20% weight)**

Did the agent successfully reconnect the faulted line during the episode?

```
reconnection_score = 1.0 if faulted_line_was_reconnected else 0.0
```

Simple binary but important — reconnection is a concrete measurable achievement that requires the agent to understand cooldown mechanics and choose the right moment.

**Total grader score**:

```
score = 0.30 × emergency_score + 0.50 × security_ratio + 0.20 × reconnection_score
```

This gives partial credit at every level: an agent that handles the emergency but fails phase 2 scores 0.30. An agent that manages phase 2 well but never reconnects scores up to 0.80. Full marks require emergency response + sustained security + reconnection.

---

### What changes in the code

| File | Change | Complexity |
|------|--------|------------|
| `grid_environment.py` `_shape_reward` | Replace Task 2 reward with three-component RL2Grid formulation + reconnect bonus | Medium |
| `grid_environment.py` `step()` | Track reconnection events (was line previously disconnected, now connected, and not by fault) | Small |
| `graph_analysis.py` | Add `compute_n1_security_score()` — bridge line count / total lines | Small — data engineer |
| `inference.py` prompt builder | Add two-threshold framing (EMERGENCY/WARNING/SAFE) + N-1 security score + reconnection window | Medium |
| `graders.py` | Replace survival ratio with phase-aware three-component grader | Medium |
| `tasks.py` | No change to scenario injection — line 0 disconnection stays the same | None |

---

### Connection to existing Task 1 improvements

The quadratic `(1-ρ²)` signal from Task 1 does NOT carry over to Task 2. Task 1 uses it because speed matters — the quadratic penalizes staying near the limit. Task 2 uses the clipped linear margin from RL2Grid because sustained quality over 20 steps matters — the linear form is more stable for long horizons.

This is the right differentiation. Each task's reward should reflect its specific objective, not be a variation of the same formula.