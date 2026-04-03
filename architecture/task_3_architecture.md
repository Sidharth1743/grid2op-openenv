# Task 3: `cascade_prevent` - Cascading Failure Prevention

## Overview

Task 3 (`cascade_prevent`) is the third task in the Grid2Op environment. It simulates an **active cascading failure scenario** where the grid is mid-collapse - multiple lines are already disconnected, load is elevated, and remaining lines are actively approaching their thermal limits. Unlike Task 2 (N-1) where the grid is stable, Task 3 requires the agent to **prevent the cascade from propagating**.

This task is based on research from the **Multi-Stage Cascading Failure (MSCF)** paper, which emphasizes that cascade prevention requires understanding individual line countdowns (`timestep_overflow`), not just global max_rho.

---

## 1. Task Definition

From `tasks.py:44-52`:

```python
"cascade_prevent": TaskSpec(
    task_id="cascade_prevent",
    difficulty="hard",
    description=(
        "Two lines are disconnected and load is increased by 15%. Prevent a "
        "cascade within 30 steps before overloads trigger more trips."
    ),
    max_steps=30,
),
```

**Key Characteristics:**
- Difficulty: Hard
- Max steps: 30
- Goal: Prevent cascade propagation

---

## 2. Reset Phase

From `tasks.py:509-548`:

### Scenario Injection

```python
def _reset_cascade_prevent(env, seed, attempt, difficulty_level, scenario_mode, benchmark_tier):
    base_obs = env.reset(seed=seed)
    
    # Get profile based on mode (curriculum or benchmark)
    if scenario_mode == "benchmark":
        stage, faulted_lines, load_scale = _cascade_benchmark_profile(benchmark_tier, seed)
    else:
        stage, faulted_lines, load_scale = _cascade_profile(difficulty_level, seed)
    
    # Increase load by the configured percentage
    load_p = [float(v) for v in (base_obs.load_p * load_scale).astype(float).tolist()]
    
    # Disconnect faulted lines
    obs = env.reset(
        seed=seed,
        options={
            "init state": {
                "set_line_status": [(line_id, -1) for line_id in faulted_lines],
                "injection": {"load_p": load_p},
            }
        },
    )
```

### Difficulty Profiles

#### Curriculum Mode (`tasks.py:302-312`):

```python
def _cascade_profile(difficulty_level: int | None, seed: int | None) -> tuple[str, list[int], float]:
    episode = _curriculum_episode(difficulty_level)
    pair_index = ((episode - 1) + (0 if seed is None else int(seed))) % len(CASCADE_LINE_PAIRS)
    selected_pair = list(CASCADE_LINE_PAIRS[pair_index])
    
    if episode <= 3:
        return "one_line_5pct", [selected_pair[0]], 1.05      # 1 line, +5%
    if episode <= 6:
        return "one_line_10pct", [selected_pair[0]], 1.10    # 1 line, +10%
    if episode <= 9:
        return "two_lines_10pct", selected_pair, 1.10         # 2 lines, +10%
    return "two_lines_15pct", selected_pair, 1.15            # 2 lines, +15%
```

#### Benchmark Mode (`tasks.py:315-326`):

```python
def _cascade_benchmark_profile(benchmark_tier: str | None, seed: int | None) -> tuple[str, list[int], float]:
    pair_index = 0 if seed is None else int(seed) % len(CASCADE_LINE_PAIRS)
    selected_pair = list(CASCADE_LINE_PAIRS[pair_index])
    
    if benchmark_tier == "cascade_prevent_easy":
        return "benchmark_easy", [selected_pair[0]], 1.05
    if benchmark_tier == "cascade_prevent_medium":
        return "benchmark_medium", [selected_pair[0]], 1.10
    if benchmark_tier == "cascade_prevent_hard":
        return "benchmark_hard", selected_pair, 1.10
    if benchmark_tier == "cascade_prevent_extreme":
        return "benchmark_extreme", selected_pair, 1.15
```

### Line Pairs

From `tasks.py:65-69`:

```python
CASCADE_LINE_PAIRS: List[tuple[int, int]] = [
    (0, 8),
    (0, 7),
    (0, 3),
]
```

These are the possible combinations of lines that will be disconnected at reset.

### Difficulty Progression Summary

| Curriculum Episode | Benchmark Tier | Lines Disconnected | Load Increase |
|--------------------|----------------|--------------------|----------------|
| 1-3 | easy | 1 line | +5% |
| 4-6 | medium | 1 line | +10% |
| 7-9 | hard | 2 lines | +10% |
| 10+ | extreme | 2 lines | +15% |

---

## 3. The Cascade Mechanism

### What is timestep_overflow?

In Grid2Op, each transmission line has a **thermal time constant**. When a line exceeds 100% loading (rho > 1.0), Grid2Op starts counting how long it's been overloaded:

```
timestep_overflow[line_id] = number of consecutive steps the line has been >100%
```

When this counter reaches the limit (`NB_TIMESTEP_OVERFLOW_ALLOWED`), the line **trips automatically** (disconnects). This is the cascade mechanism.

### Visual Example

```
Step 1:  Line 5 at 105% → overflow[5] = 1
Step 2:  Line 5 at 107% → overflow[5] = 2
Step 3:  Line 5 at 110% → overflow[5] = 3
Step 4:  Line 5 at 112% → overflow[5] = 4 (if limit is 4, line trips!)
         ↓
Line trips → power reroutes → other lines may now exceed 100%
         ↓
More lines trip → cascade continues
```

### Why Task 3 is Different from Task 2

| Aspect | Task 2 (n_minus_1) | Task 3 (cascade_prevent) |
|--------|-------------------|-------------------------|
| Grid state | Stable, degraded | Unstable, actively collapsing |
| Key metric | max_rho trending | timestep_overflow countdowns |
| Agent action | Find new stable point | Prevent lines from reaching trip threshold |
| Failure mode | Gradual deterioration | Sudden cascade at specific steps |
| Urgency | Methodical | Immediate - lines trip if not acted upon |

---

## 4. Reward Function

From `grid_environment.py:611-628`:

```python
elif self._task_id == "cascade_prevent":
    # Component 1: Cascade prevention bonus
    if not auto_trip_detected:
        reward += 0.3
    else:
        reward -= 2.5
    
    # Component 2: Overflow urgency penalty (quadratic)
    reward -= 0.05 * sum(int(value) ** 2 for value in observation.timestep_overflow)
    
    # Component 3: Thermal margin signal
    clipped_margins = [max(-1.0, min(1.0, 1.0 - float(rho))) for rho in observation.rho]
    thermal_margin = (sum(clipped_margins) / len(clipped_margins)) if clipped_margins else 0.0
    reward += 0.1 * thermal_margin
    
    # Terminal signals
    if observation.metadata.get("convergence_failed"):
        reward -= 12.0
    elif done and not reached_time_limit:
        reward -= 12.0
    elif reached_time_limit:
        auto_trips = sum(
            1 for entry in self._state.episode_log if bool(entry.auto_trip_detected)
        ) + (1 if auto_trip_detected else 0)
        reward += 5.0 * max(0.0, 1.0 - (auto_trips / 5.0)) ** 2
```

### Reward Components Breakdown

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Cascade prevention** | +0.3 if no auto-trip, -2.5 if auto-trip | Primary objective - don't let lines trip |
| **Overflow urgency** | -0.05 × Σ(overflow²) | Quadratic penalty for countdown urgency |
| **Thermal margin** | +0.1 × mean(clip(1-ρ, -1, 1)) | Keep lines below thermal limits |
| **Terminal - survival** | +5.0 × (1 - auto_trips/5)² | Quadratic bonus based on auto-trip count |
| **Terminal - blackout** | -12.0 | Convergence failure or early termination |

### Key Design Decisions

1. **Quadratic overflow penalty**: overflow=1 costs 0.05, overflow=2 costs 0.20, overflow=3 costs 0.45. This matches real urgency - a line at overflow=2 will trip next step unless acted upon.

2. **Anti-reward-hacking terminal**: The survival bonus is reduced by auto-trip count. An agent cannot score well by "managing the aftermath" of cascades - it must prevent them.

3. **Differentiated from Task 2**: Task 2 uses linear signals (R_survive + R_overload + R_cost). Task 3 uses quadratic to model the immediate urgency of line countdowns.

---

## 5. Grading

From `graders.py:86-121`:

```python
def grade_cascade_prevent(episode_log: list[EpisodeStepLog], max_steps: int = 30) -> float:
    if not episode_log:
        return 0.0
    
    # Component A: Cascade containment (50%)
    containment_ratio = sum(1 for entry in episode_log if not entry.auto_trip_detected) / max_steps
    
    # Component B: Thermal stability (30%)
    containment_steps = [entry for entry in episode_log if not entry.auto_trip_detected]
    stability_ratio = (
        sum(1 for entry in containment_steps if entry.all_lines_below_100) / len(containment_steps)
        if containment_steps
        else 0.0
    )
    
    # Component C: Recovery speed (20%)
    first_overload_step = next(
        (entry.step for entry in episode_log if not entry.all_lines_below_100),
        None,
    )
    if first_overload_step is None:
        recovery_score = 1.0
    else:
        stabilize_step = next(
            (
                entry.step
                for entry in episode_log
                if entry.step >= first_overload_step and entry.all_lines_below_100
            ),
            None,
        )
        if stabilize_step is None:
            recovery_score = 0.0
        else:
            recovery_score = max(0.0, 1.0 - ((stabilize_step - first_overload_step) / 10.0))
    
    score = (0.5 * containment_ratio) + (0.3 * stability_ratio) + (0.2 * recovery_score)
    return round(min(1.0, max(0.0, score)), 6)
```

### Grading Components

| Component | Weight | Formula | What it measures |
|-----------|--------|---------|------------------|
| **Cascade containment** | 50% | steps_without_auto_trip / 30 | Primary metric - how many steps had zero line trips |
| **Thermal stability** | 30% | safe_steps / containment_steps | Among non-trip steps, what fraction had all lines <100% |
| **Recovery speed** | 20% | max(0, 1 - (steps_to_stabilize / 10)) | How fast did the agent recover from first overload |

### Grading Examples

| Scenario | Containment | Stability | Recovery | Score |
|----------|-------------|-----------|----------|-------|
| No auto-trips, always <100% | 1.0 | 1.0 | 1.0 | 1.0 |
| 1 auto-trip at step 15 | 0.5 | 0.5 | 0.5 | 0.5 |
| 5 auto-trips spread out | 0.17 | 0.1 | 0.0 | 0.12 |
| Full survival but always above 90% | 1.0 | 0.0 | 0.5 | 0.6 |

---

## 6. Agent Strategy

From `inference.py:549-562`:

The prompt includes special instructions for Task 3:

```python
if task_id == "cascade_prevent":
    prompt_parts.append(
        "TASK RULE: In cascade_prevent, prioritize lines with active overflow countdowns. "
        "A line with timestep_overflow=2 is more urgent than a line with high rho but overflow=0."
    )
```

### What the LLM Should Do

1. **Identify urgent lines**: Check `timestep_overflow` for each line - not just rho
2. **Triage**: A line at overflow=2 will trip next step - act immediately
3. **Prevent**: Use redispatch or topology changes to reduce overflow before trip
4. **Maintain**: Keep max_rho below 1.0 to prevent new overflows starting
5. **Survive**: Reach step 30 without cascade events

### Action Types Available

- **Redispatch**: Shift generation between generators to change power flow patterns
- **Topology changes**: Disconnect lines to reroute power (risky - can cause more trips)
- **Do nothing**: Let the grid evolve (risky if lines are near trip threshold)

---

## 7. Latest Benchmark Results

From eval `baseline_eval_20260402_080302.json`:

### Overall Results
```json
{
  "cascade_prevent": {
    "score_mean": 0.798,
    "episode_length_mean": 23.25
  }
}
```

### By Tier

| Tier | Score | Episode Length | Notes |
|------|-------|-----------------|-------|
| **easy** (1 line, +5%) | **1.0** | 30 (full) | Perfect |
| **medium** (1 line, +10%) | **1.0** | 30 (full) | Perfect |
| **hard** (2 lines, +10%) | 0.516 | 13.8 | Struggling |
| **extreme** (2 lines, +15%) | 0.677 | 19.2 | Challenging |

### Key Observations

1. **Easy/Medium tiers are achievable**: Full survival, score 1.0
2. **Hard/Extreme tiers are genuinely harder**: Lower scores reflect actual difficulty
3. **No cascading failures**: Agent avoids causing auto-trips
4. **Cannot fully stabilize**: max_rho stays at 0.88-0.90 in extreme tier
5. **Avoids dangerous actions**: Doesn't disconnect bridge lines that would cause blackout

---

## 8. Unique Distinction from Other Tasks

| Dimension | Task 1 (single_fault) | Task 2 (n_minus_1) | Task 3 (cascade_prevent) |
|-----------|----------------------|-------------------|-------------------------|
| **Faults at reset** | 0 (stress only) | 1 line | 1-2 lines |
| **Load** | baseline | baseline | +5% to +15% |
| **Max steps** | 10 | 20 | 30 |
| **Goal** | Fix max_rho < 80% | Survive 20 steps | Stop cascade propagation |
| **Key metric** | max_rho | survival | timestep_overflow |
| **Urgency** | None | None | Immediate (countdowns) |
| **Reward shape** | Quadratic (1-ρ²) | Linear (RL2Grid) | Quadratic (overflow²) |
| **Agent mindset** | Fix the problem | Keep it stable | Prevent the domino chain |

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `tasks.py:44-52` | Task definition |
| `tasks.py:65-69` | Line pairs for cascade scenarios |
| `tasks.py:302-326` | Curriculum and benchmark profiles |
| `tasks.py:509-548` | Reset function |
| `grid_environment.py:611-628` | Reward function |
| `graders.py:86-121` | Grading function |
| `inference.py:549-562` | LLM prompt |

---

## 10. Research Foundation

The task design is based on insights from power grid cascading failure literature:

1. **MSCF Paper**: Stages are interdependent - solving one stage may make the next unrecoverable
2. **L2RPN Competition**: Winning agents use overflow countdowns to prioritize urgent actions
3. **ICLR 2021 Winner**: Two-threshold system (act when rho >= 0.9)
4. **Quadratic Urgency**: Lines at overflow=2 are 4x more urgent than overflow=1 - this matches real thermal dynamics

The key insight: Task 3 tests **reactive speed under time pressure**, not just optimal steady-state operation.