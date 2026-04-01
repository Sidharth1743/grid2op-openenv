# Task 1: `single_fault` - Complete Architecture & Walkthrough

## Overview

`single_fault` is the first and most important task in the Grid2Op environment. It simulates a pre-overload power grid scenario where the agent must stabilize the grid within 10 timesteps by bringing all transmission lines below a target threshold (80% for easy/curriculum, 82-93% for benchmark tiers).

---

## 1. BEFORE THE GAME STARTS (Reset Phase)

The environment performs a **warmup** phase to find a suitable starting state:

```
Grid2Op loads: l2rpn_case14_sandbox (14-bus power grid)
     │
     ├── ChronicsHandler starts at step 0
     │
     ▼
Env steps forward: 0 → 1 → 2 → ... → N
     │
     ▼
At each step, calculate: max_rho = max(all line loadings)
     │
     ▼
Find the step where: target_min ≤ max_rho ≤ target_max
     │
     │  Difficulty levels:
     │  - easy/curriculum: 0.90-0.94 → 0.82-0.85 (benchmark)
     │  - moderate: 0.94-0.97 → 0.86-0.89 (benchmark)
     │  - severe: 0.96-0.99 → 0.90-0.93 (benchmark)
     │
     ▼
STOP at that step - this is your starting state
     │
     ▼
Return observation + scenario metadata
```

**Scenario Metadata** (returned to track what was used):
- `curriculum_episode`: Episode number in curriculum
- `curriculum_stage`: "mild", "moderate", "severe", or benchmark equivalent
- `scenario_mode`: "curriculum" or "benchmark"
- `benchmark_tier`: Which benchmark tier if in benchmark mode
- `time_series_id`: Which chronic was used
- `target_rho_range`: [min, max] that was searched for
- `warmup_steps`: How many steps were taken to find the state
- `target_matched`: True if exact target found, False if fallback used

---

## 2. GRID PHYSICS: What Is `l2rpn_case14_sandbox`?

### The Power Grid

```
┌─────────────────────────────────────────────────────────────────────┐
│                    14-BUS POWER GRID                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    GEN1 (100 MW)                                                    │
│        │                                                             │
│        ╰── LINE 0 (65%) ── BUS 1 ── LINE 1 (70%) ── BUS 2 ──...    │
│                                                                      │
│    GEN2 (80 MW)                                                     │
│        │                                                             │
│        ╰── LINE 5 (87%) ── BUS 3 ── LINE 6 (55%) ── BUS 5 ──...    │
│                              ↑                                       │
│                              │ PROBLEM LINE                          │
│    GEN3 (60 MW)                                                     │
│        │                                                             │
│        ╰── LINE 12 (45%) ── BUS 6 ── LINE 13 (60%) ── BUS 9        │
│                                                                      │
│    GEN4 (45 MW)                                                     │
│    GEN5 (30 MW)                                                     │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Total: 5 Generators, 20 Lines, 14 Substations, ~240 MW Load        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Value | Description |
|----------|-------|-------------|
| `n_line` | 20 | Number of transmission lines |
| `n_gen` | 5 | Number of generators |
| `n_sub` | 14 | Number of substations/buses |
| `n_load` | 11 | Number of loads (cities/factories) |
| `gen_redispatchable` | [True, True, True, False, False] | Which generators can be moved |

### What Is `rho`?

`rho` = **thermal loading ratio** = actual power flow / maximum capacity

```
rho = 0.65  → Line is at 65% of its thermal limit
rho = 0.87  → Line is at 87% of its thermal limit (WARNING!)
rho = 1.05  → Line is overloaded by 5% (PENALTY!)
```

---

## 3. STEP 0: Initial State (After Reset)

After the warmup finds a valid state, the agent receives this observation:

```
┌─────────────────────────────────────────────────────┐
│  POWER GRID - STEP 0 (Initial State)                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│    GEN1 ●───────────╮                               │
│        (100 MW)     ╰─── LINE 0 ───● SUBSTATION 1  │
│                         (65%)                       │
│                                                     │
│    GEN2 ●───────────╮                               │
│        (80 MW)     ╰─── LINE 5 ───● SUBSTATION 3  │
│                         (87%)  ← PROBLEM LINE      │
│                                                     │
│    GEN3 ●───────────╮                               │
│        (60 MW)     ╰─── LINE 12 ──● SUBSTATION 5   │
│                         (45%)                       │
│                                                     │
│  Load: 240 MW  |  Lines: 20  |  Gen: 5             │
│  MAX_RHO = 0.87 (Line 5)                            │
│  TARGET: Get ALL lines < 0.80                      │
└─────────────────────────────────────────────────────┘
```

### Observation JSON Structure

```json
{
  "rho": [0.65, 0.72, 0.55, 0.81, 0.45, 0.87, 0.62, 0.50, 0.78, 0.41, 
          0.33, 0.89, 0.45, 0.60, 0.72, 0.38, 0.55, 0.68, 0.44, 0.52],
  "gen_p": [100.0, 80.0, 60.0, 45.0, 30.0],
  "load_p": [45.0, 55.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 5.0],
  "line_status": [true, true, true, true, true, true, true, true, true, true,
                  true, true, true, true, true, true, true, true, true, true],
  "timestep_overflow": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "sensitivity_guidance": [
    {"action_type": "redispatch", "target_id": 1, "expected_rho_change": -0.023},
    {"action_type": "redispatch", "target_id": 0, "expected_rho_change": -0.015}
  ],
  "reward": 0.0,
  "done": false
}
```

### What the Agent Sees

1. **Line loadings** (`rho` array): 20 values, one per line
   - Line 5 is at 87% - ABOVE the 80% target!
   
2. **Generator outputs** (`gen_p` array): 5 values in MW
   - Generators 0, 1, 2 are redispatchable (can change output)
   - Generators 3, 4 are fixed

3. **Load** (`load_p` array): 11 values showing current consumption
   - This will INCREASE each step as chronics advance

4. **Line status**: All lines are connected (true)

5. **Sensitivity guidance**: Hints about which actions might help
   - Redispatch generator 1 might reduce rho by 0.023

---

## 4. LLM REASONING (Planner Phase)

The LLM receives additional context through `/planning_context`:

### Graph Analysis
```
{
  "bridge_lines": [5, 12],
  "safe_to_disconnect": [3, 7, 15],
  "parallel_groups": [[0, 1], [8, 9]],
  "high_centrality_buses": [1, 3, 5],
  "islanded_clusters": [],
  "congestion_corridor": [4, 5, 6],
  "stressed_lines": [5, 11]
}
```

### Redispatchable Generators
```json
{
  "redispatchable_generators": [0, 1, 2],
  "redispatch_generators": [
    {
      "gen_id": 0,
      "p_mw": 100.0,
      "max_ramp_up": 20.0,
      "max_ramp_down": 15.0,
      "allowed_delta_min": -15.0,
      "allowed_delta_max": 20.0,
      "allowed_deltas": [-15.0, -7.5, 0.0, 7.5, 20.0]
    },
    {
      "gen_id": 1,
      "p_mw": 80.0,
      "max_ramp_up": 15.0,
      "max_ramp_down": 12.0,
      "allowed_delta_min": -12.0,
      "allowed_delta_max": 15.0,
      "allowed_deltas": [-12.0, -6.0, 0.0, 6.0, 15.0]
    }
  ]
}
```

### LLM Internal Reasoning

```
Observation analysis:
- Line 5 at 87% - ABOVE 80% target
- Generator 0: can shift -15 to +20 MW
- Generator 1: can shift -12 to +15 MW

Strategy options:
1. Do nothing - wait for grid to stabilize
   → Risk: chronics increase load, rho goes UP
   
2. Redispatch Generator 0 (-10 MW) and Generator 1 (+10 MW)
   → Expected: Line 5 drops to ~82%
   → Risk: not enough, might still be above target
   
3. More aggressive: Gen0 (-15), Gen1 (+15)
   → Expected: Line 5 drops to ~77%
   → Cost: penalty = 0.01 × 30 = 0.3

Decision: Go with option 3 - more aggressive
Action: {"redispatch": {"0": -10.0, "1": 10.0}}
```

---

## 5. STEP 1: Action Execution

### Action JSON

```json
{
  "line_set": {},
  "redispatch": {"0": -10.0, "1": 10.0},
  "do_nothing": false
}
```

### What This Means

- **Generator 0**: Reduce output by 10 MW (100 → 90 MW)
- **Generator 1**: Increase output by 10 MW (80 → 90 MW)
- Total generation stays the same (200 MW)
- Power flow paths change through the network

### Grid Physics Computation

```
BEFORE action:
  Gen0: 100 MW, Gen1: 80 MW
  Line 5 power flow: 87% of capacity

AFTER action:
  Gen0: 90 MW (-10), Gen1: 90 MW (+10)
  
Power flow solver recomputes:
  → New line impedances
  → New power paths
  → New line loadings

Result:
  Line 0: 65% → 60%
  Line 5: 87% → 82%  ↓ (improved!)
  Line 12: 45% → 48%
```

### New State After Step 1

```
┌─────────────────────────────────────────────────────┐
│  POWER GRID - AFTER STEP 1                          │
├─────────────────────────────────────────────────────┤
│  Line 5: 87% → 82%  ↓                               │
│  MAX_RHO = 0.82  (still above 0.80 target!)        │
│  All lines < 1.0? YES (no overload)                 │
│                                                     │
│  Reward: 0.05 × (1-0.82) - 0.01 × 20 = 0.009 - 0.2 │
│  Net: -0.191                                        │
└─────────────────────────────────────────────────────┘
```

### Reward Breakdown (Step 1)

```
Safe margin bonus:  0.05 × (1.0 - 0.82) = 0.05 × 0.18 = 0.009
Overload penalty:   0 (no lines > 1.0)
Redispatch penalty: 0.01 × |−10| + 0.01 × |10| = 0.01 × 20 = 0.2
                                                    ─────────────────
Total reward:      0.009 - 0.2 = -0.191
```

---

## 6. STEP 2: Chronics Advance (The Challenge)

### What Are Chronics?

Chronics are **time-series data** representing the grid evolving over time:
- Load patterns change (morning → afternoon → evening)
- Generation schedules change
- Each step advances time by one unit

### What Happens in Step 2

```
CHRONICS TICK FORWARD: step 2

Load increases due to time progression:
  City A (bus 1):    45 → 47 MW (+2)
  City B (bus 3):    55 → 57 MW (+2)
  City C (bus 5):    40 → 42 MW (+2)
  City D (bus 7):    35 → 36 MW (+1)
  ...and so on
Total load increase: ~+6-8 MW

Grid physics recomputes:
  More power must flow through the network
  → Line loadings INCREASE
  
  Line 5: 82% → 85%  ↑ (back up!)
  Line 0: 60% → 63%
  Line 12: 48% → 51%
```

### The Core Challenge

```
┌─────────────────────────────────────────────────────┐
│  THE PROBLEM: Load Keeps Growing                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: Redispatch fixed the problem              │
│          max_rho: 0.87 → 0.82                       │
│                                                     │
│  Step 2: Chronics tick forward                      │
│          Load increases +6 MW                       │
│          max_rho: 0.82 → 0.85 (WORSE!)              │
│                                                     │
│  The agent must keep up with the growing load!    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### New State After Step 2

```
┌─────────────────────────────────────────────────────┐
│  POWER GRID - AFTER STEP 2                          │
├─────────────────────────────────────────────────────┤
│  Line 5: 82% → 85%  ↑ (back up!)                    │
│  MAX_RHO = 0.85  (still above 0.80 target)         │
│                                                     │
│  Reward: -0.05 (max_rho went UP!)                  │
│  Net: -0.25                                         │
└─────────────────────────────────────────────────────┘
```

### Reward Breakdown (Step 2)

```
Margin bonus:    0.05 × (1-0.85) = 0.05 × 0.15 = 0.0075
Overload:       0 (no lines > 1.0)
Redispatch:     0.01 × 20 = 0.2
Delta penalty:  max_rho went UP → -0.05
                                        ──────────────
Total:          0.0075 - 0.2 - 0.05 = -0.2425
```

---

## 7. STEP 3: More Aggressive Action

### LLM Sees

- max_rho went UP from 0.82 to 0.85
- Need stronger intervention
- Previous redispatch wasn't enough

### LLM Decision

```json
{
  "redispatch": {"0": -15.0, "1": 15.0}
}
```
(Bigger shift: -15/+15 = 30 MW total)

### Grid Physics

```
Gen0: 90 → 75 MW (-15)
Gen1: 90 → 105 MW (+15)

Line 5: 85% → 77%  ↓↓ (significant improvement!)
```

### New State

```
┌─────────────────────────────────────────────────────┐
│  POWER GRID - AFTER STEP 3                          │
├─────────────────────────────────────────────────────┤
│  Line 5: 85% → 77%  ↓↓ SUCCESS!                    │
│  MAX_RHO = 0.77  (< 0.80 target!)                   │
│  ALL LINES BELOW TARGET!                            │
│                                                     │
│  EPISODE ENDS HERE (done=True)                     │
└─────────────────────────────────────────────────────┘
```

### Episode Complete

When `all(rho < target)`, the episode terminates early:

```python
# From grid_environment.py line 215-218
if self._task_id == "single_fault" and all_lines_below_target:
    done = True
```

---

## 8. GRADING (Score Calculation)

### Episode Log

```json
[
  {"step": 1, "max_rho": 0.82, "all_lines_below_target": false},
  {"step": 2, "max_rho": 0.85, "all_lines_below_target": false},
  {"step": 3, "max_rho": 0.77, "all_lines_below_target": true}
]
```

### Grader Calculation (from graders.py)

```python
def grade_single_fault(episode_log):
    # 1. Survival ratio (70% weight)
    survival_ratio = min(1.0, len(episode_log) / 10)  # 3/10 = 0.3
    survival_score = survival_ratio * 0.7  # = 0.21
    
    # 2. Target achieved bonus (50%)
    achieved_target = any(entry.all_lines_below_target for entry in episode_log)
    target_bonus = 0.5 if achieved_target else 0.0  # = 0.5
    
    # 3. Final state bonus
    final_rho = 0.77
    target_threshold = 0.80
    if final_rho < target_threshold:
        final_bonus = 0.3  # = 0.3
    elif final_rho < target_threshold + 0.05:
        final_bonus = 0.15
    else:
        final_bonus = 0.0
    
    # Total
    score = survival_score + target_bonus + final_bonus
    score = min(1.0, max(0.0, score))
    
    return score

# Calculation:
# survival_score = 0.3 × 0.7 = 0.21
# target_bonus = 0.5
# final_bonus = 0.3
# 
# TOTAL = 0.21 + 0.5 + 0.3 = 1.01 → capped at 1.0
```

---

## 9. Visual Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TASK 1 COMPLETE FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

RESET PHASE
│
├─► Load chronics (l2rpn_case14_sandbox)
│
├─► Warmup loop: step forward until max_rho in target range
│
├─► Found: max_rho = 0.87 at step 47
│
└─► Return observation to agent

STEP 1
│
├─► Agent sees: max_rho = 0.87, Line 5 at 87%
│
├─► Agent thinks: "Redispatch Gen0→Gen1 by 10 MW"
│
├─► Execute: {"redispatch": {"0": -10.0, "1": 10.0}}
│
├─► Grid physics: Line 5 → 82%
│
├─► Reward: -0.191 (penalty for redispatch)
│
└─► State: max_rho = 0.82 (still > 0.80)

STEP 2
│
├─► Chronics tick: load +6 MW
│
├─► Grid physics: Line 5 → 85% (load pushed it up!)
│
├─► Reward: -0.25 (max_rho went UP!)
│
└─► State: max_rho = 0.85 (still > 0.80)

STEP 3
│
├─► Agent sees: max_rho went UP, need more!
│
├─► Agent thinks: "More aggressive: -15/+15 MW"
│
├─► Execute: {"redispatch": {"0": -15.0, "1": 15.0}}
│
├─► Grid physics: Line 5 → 77%  ↓↓
│
├─► Check: all(rho < 0.80)? YES!
│
├─► EPISODE DONE! (done = True)
│
└─► Grading: 0.21 + 0.5 + 0.3 = 1.0

┌─────────────────────────────────────────────────────────────────────┐
│                              SCORE = 1.0                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Difficulty Levels

### Curriculum Mode (Progressive)

| Episode | Stage | Target Range | Description |
|---------|-------|--------------|-------------|
| 1-3 | mild | 0.90-0.94 | Easy warmup |
| 4-6 | moderate | 0.94-0.97 | Medium challenge |
| 7+ | severe | 0.96-0.99 | Hardest |

### Benchmark Mode (Fixed Tiers)

| Tier | Target Range | Fixed in Code |
|------|--------------|---------------|
| `single_fault_easy` | 0.82-0.85 | tasks.py:250 |
| `single_fault_moderate` | 0.86-0.89 | tasks.py:252 |
| `single_fault_severe` | 0.90-0.93 | tasks.py:254 |

**Note**: The original benchmark ranges (0.90-0.94, etc.) were mathematically impossible because generators could only reduce ~0.03-0.05 rho per step. Fixed in recent updates.

---

## 11. Key Files Reference

| File | Purpose |
|------|---------|
| `server/tasks.py` | Scenario injection, warmup logic, benchmark profiles |
| `server/grid_environment.py` | Environment wrapper, reward shaping, action execution |
| `server/graders.py` | Score calculation (survival + target bonus + final bonus) |
| `inference.py` | LLM planner with simulate-then-act loop |
| `models.py` | Typed data classes (GridAction, GridObservation, etc.) |
| `graph_analysis.py` | Topology analysis (bridges, stressed lines, etc.) |

---

## 12. Why It's Challenging

| Challenge | Explanation |
|-----------|-------------|
| **Load keeps growing** | Each chronics step adds ~5-10 MW, pushing rho back up |
| **Limited generator ramp** | Can only shift ~30-50 MW per step |
| **Target threshold** | Must get ALL lines (not just max) below 80% |
| **Penalty for trying** | 0.01 × MW redispatched discourages large interventions |
| **Time limit** | Only 10 steps to solve |

The agent must find the **optimal balance**: enough redispatch to push max_rho below target, but not so much that the penalty outweighs the rewards.