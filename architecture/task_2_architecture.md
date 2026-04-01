# Task 2: `n_minus_1` - N-1 Contingency Management

## Overview

Task 2 (`n_minus_1`) simulates an **N-1 contingency** - one transmission line has failed and the grid must operate in a degraded topology for 20 steps without blacking out. This task is based on the **RL2Grid paper** and **L2RPN winning agent heuristics**.

Key difference from Task 1:
- **Task 1**: One line is stressed but topology is intact - "fix this line"
- **Task 2**: One line is already GONE - "operate safely in degraded topology"

---

## 1. Reset Phase

```python
# tasks.py:434-456
def _reset_n_minus_1(env, seed, difficulty_level, scenario_mode, benchmark_tier):
    obs = env.reset(
        seed=seed,
        options={"init state": {"set_line_status": [(0, -1)]}},
    )
    return obs, {
        "faulted_lines": [0],
        "curriculum_stage": "fixed_n_minus_1",
        ...
    }
```

**What happens:**
- Line 0 is **disconnected** at reset (status = -1)
- No warmup search (unlike Task 1)
- Straight reset with fault already applied

**Initial state:**
```
┌─────────────────────────────────────┐
│  GRID - STEP 0 (after reset)        │
├─────────────────────────────────────┤
│                                     │
│    LINE 0: DISCONNECTED ❌         │
│    (was carrying ~15% of power)     │
│                                     │
│    Other 19 lines: connected        │
│    max_rho: ~0.75-0.90              │
│                                     │
│  TASK: Survive 20 steps             │
│  GOAL: Don't black out              │
└─────────────────────────────────────┘
```

---

## 2. Two-Phase Structure

Based on how grid operators actually work:

### Phase 1: Emergency Response (Steps 1-5)
- The line just tripped, remaining lines may be above 80%
- Agent must take corrective actions (topology/redispatch)
- Target: get all lines below `ρ_danger = 0.92`

### Phase 2: Sustained Secure Operation (Steps 6-20)
- Emergency addressed, now maintain safe operation
- Agent should reconnect faulted line when cooldown expires
- Maintain N-1 security (can withstand another line trip)

---

## 3. Reward Function (RL2Grid-inspired)

```python
# grid_environment.py:588-599
elif self._task_id == "n_minus_1":
    r_survive = 1.0
    clipped_margins = [max(-1.0, min(1.0, 1.0 - float(rho))) for rho in observation.rho]
    r_overload = sum(clipped_margins) / len(clipped_margins)
    r_cost = -self._n_minus_1_redispatch_cost(action)
    reward += (0.3 * r_survive) + (0.6 * r_overload) + (0.1 * r_cost)
    if reconnect_successful and self._reconnection_within_margin(...):
        reward += 2.0
    if reached_time_limit:
        reward += 10.0 * ((step / max_steps) ** 2)  # quadratic survival
    elif done:
        reward -= 15.0  # blackout penalty
```

### Components

| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| `R_survive` | +1.0 per step | 0.3 | Constant survival signal |
| `R_overload` | (1/n) × Σ clip(1-ρ, -1, 1) | 0.6 | Loading margin quality |
| `R_cost` | -0.05 × Σ\|ΔMW\|/ramp | 0.1 | Economic cost of redispatch |
| `R_reconnect` | +2.0 if safe reconnection | - | Heuristic from winning agents |
| Terminal | +10×(s/m)² / -15 | - | Quadratic survival / blackout |

### Reconnection Bonus Logic

```python
# grid_environment.py:689-698
def _reconnection_within_margin(self, previous_observation, observation):
    previous_max = max(previous_observation.rho)
    current_max = max(observation.rho)
    return current_max <= previous_max + 0.1  # don't worsen by >10%
```

---

## 4. Two-Threshold Framing (ICLR 2021 Winner)

In the prompt, lines are categorized:

```
ρ_danger = 0.92   → EMERGENCY (must act)
ρ_safe   = 0.80   → WARNING (monitor)
           < 0.80 → SAFE (no action needed)
```

This maps directly to how grid operators think.

---

## 5. N-1 Security Score

Computed in `graph_analysis.py`:

```python
n1_security_score = 1.0 - (bridge_line_count / n_connected_lines)
```

- **1.0**: No single line trip would island the grid
- **0.5**: Half the lines are bridges (highly vulnerable)

Included in prompt:
```
N-1 STRUCTURAL SECURITY: score=0.941; bridge_lines=[4, 11, 15]
```

---

## 6. Grading (Phase-Aware)

```python
# graders.py:56-80
def grade_n_minus_1(episode_log, max_steps=20):
    # Survival gates the score
    survival_ratio = min(1.0, len(episode_log) / max_steps)
    
    # Component A: Emergency response (30%)
    emergency_clear_step = next(
        (entry.step for entry in episode_log[:5] if max_rho < 0.92),
        None
    )
    emergency_score = max(0, 1.0 - 0.2 × max(0, emergency_clear_step - 1))
    
    # Component B: Sustained security (50%)
    phase2_logs = [e for e in episode_log if e.step >= 6]
    security_ratio = sum(1 for e in phase2_logs if max_rho < 0.90) / 15
    
    # Component C: Reconnection (20%)
    reconnection_score = 1.0 if any(0 not in e.disconnected_lines) else 0.0
    
    mastery_score = 0.30 × emergency + 0.50 × security + 0.20 × reconnect
    
    # Final: survival × mastery (no legacy override)
    return survival_ratio × mastery_score
```

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Emergency response | 30% | Cleared within 5 steps? |
| Sustained security | 50% | Steps 6-20 with rho < 0.90? |
| Reconnection | 20% | Did agent reconnect line 0? |

**Note**: Previously, a `legacy_survival_score` override would return 1.0 for any full-length episode regardless of mastery. This was removed to ensure honest grading - the score now properly reflects both survival AND quality of operation.

---

## 7. Example Episode Walkthrough

```
Step 1:
  - Line 0 already disconnected
  - max_rho = 0.957 (EMERGENCY: line 17 at 95.7%)
  - Action: disconnect line 14 (topology action)
  - max_rho → 0.888 ✓ (below 0.92)
  - Reward: 0.3×1.0 + 0.6×0.89 + 0.1×0 = 0.834

Step 2:
  - max_rho = 0.881 (WARNING zone)
  - Action: do_nothing
  - Reward: 0.648

Step 3:
  - max_rho = 0.834
  - Action: disconnect line 12 (protect against cascade)
  - Reward: 0.647

Step 5:
  - RECONNECT_WINDOW: line 0 cooldown expired
  - Action: reconnect line 0 ✓
  - Reward: +2.0 bonus (reconnection success!)

Step 12:
  - RECONNECT_WINDOW: line 12 cooldown expired
  - Action: reconnect line 12 ✓
  - Reward: +2.0 bonus again!

Step 20:
  - max_rho = 0.815 (SAFE)
  - reached_time_limit = True
  - Terminal bonus: +10.0 × (20/20)² = +10.0
  - Total score: 1.0
```

---

## 8. Key Design Decisions

### Why Three-Component Reward?
- `R_survive`: Ensures agent always has positive signal for staying alive
- `R_overload`: The L2RPN standard - encourages margin over quantity
- `R_cost`: Economic signal - topology changes are "free", redispatch costs

### Why Quadratic Terminal?
- Linear: 18/20 = 0.9
- Quadratic: 18/20 → 10 × 0.81 = 8.1 bonus

The last few steps matter more - quadratic creates stronger incentive to finish.

### Why Reconnection Bonus?
From L2RPN 2023 winning agent: "greedy reconnection module"
- Disconnected lines should be reconnected when cooldown expires
- Without explicit reward, agent ignores this action

---

## 9. Latest Evaluation Results

```
20260401_230222 (after grading fix):
  n_minus_1: score_mean=0.952, score_std=0.070
  episode_length: 20 (full survival)
  do_nothing_steps: 9.4/20 (47%)
  redispatch_mw: 23.5 MW total
  
Grading breakdown (seed 0, score 0.94):
  survival_ratio = 20/20 = 1.0
  emergency_score = 0.8 (cleared at step 2, not step 1)
  security_ratio = 1.0 (all 15 phase-2 steps < 0.90)
  reconnection_score = 1.0 (line 0 at step 1, line 12 at step 14)
  mastery = 0.30×0.8 + 0.50×1.0 + 0.20×1.0 = 0.94
  final = 1.0 × 0.94 = 0.94

Behavior observed:
  - Two successful reconnections (line 0 at step 1, line 12 at step 14)
  - N-1 security maintained at 1.0 after recovery
  - Agent learns to stay passive when safe
  - Grading is now honest (survival × mastery, no override)
```

**Note**: Before the grading fix, score was 1.0 due to a `legacy_survival_score` override that returned 1.0 for any full-length episode. The fix removed this override so the score properly reflects both survival AND quality of operation.

---

## 10. Files Reference

| File | Purpose |
|------|---------|
| `tasks.py:434` | Scenario injection (line 0 disconnection) |
| `grid_environment.py:588` | Three-component reward function |
| `grid_environment.py:689` | Reconnection margin check |
| `graph_analysis.py:129` | N-1 security score calculation |
| `graders.py:56` | Phase-aware grader |
| `inference.py:521` | Prompt with two-threshold framing |

---

## 11. Literature Reference

- **RL2Grid** (arXiv:2212.04069): Three-component reward `R = α·R_survive + β·R_overload + η·R_cost`
- **L2RPN 2023**: Greedy reconnection heuristic
- **ICLR 2021 Winner**: Activate only when `ρ_max ≥ 0.9`, two-threshold system