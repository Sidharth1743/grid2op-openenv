## Task 4 — Multi-Stage Cascading Failure Mitigation (Expert)

### What Is MSCF?

Traditional cascading failure research treats each stage independently. The MSCF paper's key insight is that stages are interdependent — an action that fixes stage 1 may make stage 2 unrecoverable, and the agent must reason across stage boundaries holistically.

Task 3 is reactive — the agent watches for line trips and prevents them. Task 4 is **multi-horizon** — the grid goes through discrete failure stages, each one changing the topology permanently, and the agent must plan across all stages simultaneously.

---

### What Is a Stage?

A stage is a discrete failure event where one or more lines trip automatically due to sustained overload. After a stage, Grid2Op runs a new power flow, the island detection assessment runs, and the agent then acts in the new degraded topology before the next stage begins.

Task 4 has exactly **three stages**. The episode doesn't end after stage 1 — the grid continues operating (possibly in a degraded state) through stages 2 and 3. This is the "multi-stage" in MSCF — the agent cannot treat each stage in isolation.

---

### Why It's Harder Than Task 3

In Task 3, the agent prevents stages from happening. If a line trips, it's a failure event (-2.5 reward) but the episode continues.

In Task 4, **stages are guaranteed to happen**. The scenario injects an initial fault severe enough that at least one more line will trip during the episode regardless of agent action. The agent cannot prevent all stages — it must minimize what is lost at each stage and position the remaining grid to survive the next.

This requires a fundamentally different strategy:

- Task 3: "Keep all lines below 100%" 
- Task 4: "Given that this cascade will proceed through three stages, manage what's recoverable and sacrifice what isn't"

---

### Reset Phase

The reset injects a severe initial fault: **three lines simultaneously disconnected** (not two like Task 3) with **20% load increase**. This is beyond the threshold where full prevention is possible — some cascade propagation is physically inevitable.

Grid2Op's `NB_TIMESTEP_OVERFLOW_ALLOWED` is set to **2** for this task (reduced from the default 3), making cascades faster and more unforgiving.

The initial state has multiple lines above 100% loading from step 0. The agent cannot "fix" this — it can only manage how the cascade unfolds.

---

### Stage Structure

Each stage runs for a maximum of 10 steps. At the end of each stage, Grid2Op's automatic overflow disconnection fires, tripping any lines that stayed above 100% for the full period.

```
Stage 1 (steps 1–10):   Agent acts, some lines auto-trip at step 10
Stage 2 (steps 11–20):  New topology, island assessment runs, agent acts again
Stage 3 (steps 21–30):  Final degraded state, agent must survive
```

The island availability assessment from the paper image you shared fires at each stage boundary:

```
For each disconnected component after auto-trip:
  If max_gen_total >= load_total: island is AVAILABLE (can self-sustain)
  If max_gen_total < load_total: island is UNAVAILABLE (will collapse)
```

Unavailable islands are declared as load shed — that load is permanently lost. The agent knows before acting in stage 2 which parts of the grid survived stage 1.

---

### What the Agent Controls

The paper's action design uses generation coefficients — the output of the i-th generator is the product of its coefficient `a_i` and its maximum power capacity.

For your system: the agent controls **generator redispatch** (continuous) and **topology switches** (discrete). The key distinction from Tasks 1–3 is that the agent must think about which generators to prioritize across stage boundaries, not just the current step.

The agent should be explicitly told in the prompt: "You are in Stage X of 3. The next stage boundary is in Y steps. Plan your actions to position the grid favorably for stage 2, not just to fix the current step."

---

### Reward Function

The paper defines four reward components. Adapted for your system:

**Component 1 — Generation cost (always present)**

Cost of total generation across available islands. Penalizes wasteful redispatch:

```
-c₁ × total_gen_cost   (c₁ = 0.01, scaled to be same magnitude as other components)
```

**Component 2 — Load loss penalty (stage-boundary signal)**

Applied only at stage boundaries when islands are assessed. Penalizes load shed from unavailable islands:

```
-BaseReward₁ × (P_loss / P_total_initial)
  where P_loss = load MW in unavailable islands
```

**Component 3 — Convergence reward (per step)**

Positive signal when power flow converges — the grid is still physically stable:

```
+BaseReward₂ × (converged_islands / total_islands)
```

**Component 4 — Win reward (terminal)**

Applied only when the agent successfully survives all three stages with at least 50% of initial load preserved:

```
+BaseReward₃ × (P_available / P_total_initial)²
  where P_available = load MW in available islands at end
```

The paper's key design principle: the constants BaseReward₁, BaseReward₂, BaseReward₃ should be in the same order of magnitude so no single component dominates.

For the 14-bus system with ~11 loads averaging ~10 MW each ≈ 110 MW total:

```
BaseReward₁ = 5.0    (load loss penalty scale)
BaseReward₂ = 0.5    (convergence reward scale per step)
BaseReward₃ = 8.0    (win reward scale)
c₁          = 0.02   (generation cost scale)
```

This keeps all components in the [-5, +8] range — same order of magnitude.

---

### Grader

Four components, one per paper reward element, plus a stage completion component:

**A — Stage completion (30% weight)**

```
stage_1_survived = 0.10 (stage 2 was reached)
stage_2_survived = 0.10 (stage 3 was reached)
stage_3_survived = 0.10 (episode ended without full blackout)
```

**B — Load preservation (40% weight)**

The central metric — how much load was kept energized across all three stages:

```
load_preservation = total_MW_preserved / total_MW_initial
```

An agent that preserved 90% of load scores 0.36. An agent that preserved 50% scores 0.20.

**C — Island quality (20% weight)**

Fraction of stage boundaries where majority of islands were classified as available:

```
island_quality = available_stages / 2 (two stage boundaries)
```

**D — Speed bonus (10% weight)**

How quickly did the agent achieve a stable configuration in each stage:

```
speed_score = average of (10 - steps_to_stable) / 10 across stages
```

**Total:**

```
score = 0.30 × stage_completion + 0.40 × load_preservation 
      + 0.20 × island_quality + 0.10 × speed_score
```

---

## The Uniqueness Map — What Makes Each Task Distinct

| Dimension | Task 3 | Task 4 |
|-----------|--------|--------|
| **Core question** | Stop the cascade from propagating | Minimize damage across guaranteed multi-stage collapse |
| **Agent mindset** | Prevention | Damage control and triage |
| **Stage structure** | No explicit stages — continuous episode | Three explicit stages with boundary assessments |
| **Faults at reset** | 2 lines, +10–15% load | 3 lines, +20% load |
| **Can cascade be stopped?** | Yes — full prevention is achievable | No — some cascade progression is physically inevitable |
| **Key observation** | `timestep_overflow` countdown per line | `P_available / P_total` load preservation ratio |
| **Reward uniqueness** | Quadratic overflow penalty + cascade containment | Generation cost + load loss + convergence + win reward |
| **Grader uniqueness** | Containment ratio + stability ratio | Stage completion + load preservation + island quality |
| **Island detection** | Used in graph prompts (advisory) | Used as hard stage boundary assessor (deterministic) |
| **LLM planning horizon** | Per-step urgency | Cross-stage strategic planning |
| **Failure mode** | Allowing the domino chain to start | Losing too much load at each stage boundary |

The two tasks test genuinely different skills. Task 3 tests reactive speed under time pressure. Task 4 tests strategic planning across discrete failure horizons with permanent consequences at each boundary.