Let me fetch the paper properly first.Good. The paper maps "each stage" of the MSCF problem to "each step" in the RL context to investigate it holistically, rather than decomposing into independent single-stage subproblems. The key insight is that stages are interdependent — solving stage 1 greedily may make stage 2 unrecoverable. I now have enough to design both tasks properly.

---

## First — Task 3 Redesigned (Cascade Prevention)

### What Is Cascade Prevention?

Task 3 takes the N-1 scenario from Task 2 and makes it active and dynamic. In Task 2, one line was already disconnected and the grid found a new stable operating point. In Task 3, the grid is **mid-collapse** — two lines are already gone, load is elevated, and the remaining lines are actively approaching their thermal limits. The cascade hasn't finished yet. Every step, Grid2Op counts how long each overloaded line has been in overflow. When that counter hits the limit, the line trips automatically and the cascade propagates.

The agent is not managing a stable-but-degraded grid. It is racing against a countdown timer on multiple lines simultaneously.

---

### Reset Phase

Two lines are disconnected at reset and load is increased by 5–15% depending on curriculum level. The initial state has several lines between 85–100% loading — not yet overloaded, but trending toward it. The chronic then keeps evolving underneath, naturally increasing stress.

The difficulty progression is:

| Curriculum | Lines Down | Load Increase | Starting max_rho |
|------------|------------|---------------|-----------------|
| Early | 1 line | +5% | ~0.85 |
| Mid | 1 line | +10% | ~0.90 |
| Hard | 2 lines | +10% | ~0.92 |
| Full | 2 lines | +15% | ~0.95–1.0 |

---

### What Makes It Harder Than Task 2

Task 2 had one line down and a stable new operating point. Task 3 has:

- Two lines down — more capacity lost, power redistribution is more complex
- Elevated load — chronic stress on top of structural stress
- Active overflow countdowns — Grid2Op is autonomously tripping lines unless the agent intervenes
- 30 steps — longer horizon means more chances for the chronic to push new lines into danger
- Cascade interdependence — each auto-trip redistributes flow and immediately threatens other lines

The key difficulty is **temporal pressure**. In Task 2, the agent can observe, think, and act methodically. In Task 3, lines with `timestep_overflow = 2` will trip next step no matter what unless the agent acts right now. The urgency is not just about keeping max_rho below 1.0 — it's about the specific countdown on each individual line.

---

### What the Agent Must Understand

The LLM needs to understand `timestep_overflow` not as a general stress indicator but as an **individual countdown per line**. A line at `rho=0.95` with `overflow=0` is less urgent than a line at `rho=1.01` with `overflow=2`. The second one trips next step.

This changes the planning horizon. The agent is not just minimizing global max_rho — it is triaging specific lines based on their individual trip timers.

---

### Reward Function

Three components, each measuring a distinct aspect of cascade prevention:

**Component 1 — Cascade prevention bonus (per step)**

Positive reward for every step where no automatic line trip occurred. This is the primary objective signal — keeping the domino chain from falling.

```
+0.3 per step if no auto-trip detected
-2.5 per auto-trip detected
```

Raised from the current `+0.2 / -2.0` because with 30 steps and two lines already down, the signal needs more separation between safe and unsafe steps.

**Component 2 — Overflow urgency penalty (per step)**

From the paper's design philosophy: the physical countdown drives urgency. A line at overflow step 1 is concerning. A line at overflow step 2 is critical. The penalty escalates non-linearly to match the real urgency:

```
-Σ_ℓ (timestep_overflow[ℓ]²) × 0.05
```

Squaring the overflow counter means: overflow=1 costs 0.05, overflow=2 costs 0.20, overflow=3 costs 0.45 (but the line would trip at 3, so this is the terminal penalty window). This is distinct from Task 1 and Task 2's linear signals — the quadratic escalation models actual urgency correctly.

**Component 3 — Thermal margin signal (per step)**

Same RL2Grid formulation as Task 2 but weighted less, because cascade prevention is more about avoiding trips than maintaining comfortable margins:

```
(1/n_line) × Σ_ℓ clip(1 - ρ_ℓ, -1, 1) × 0.1
```

**Terminal signals:**

```
Full survival bonus: +5.0 × (1 - auto_trips_count / 5)²
  → Zero auto-trips: +5.0
  → One auto-trip survived: +3.2 (penalizes but rewards managing aftermath)
  → Five auto-trips: +0.0
Blackout penalty: -12.0
```

The survival bonus being reduced by auto-trip count is the key anti-reward-hacking mechanism. An agent cannot score well by surviving 30 steps if it allowed cascades to happen — it just managed the aftermath. Cascade prevention is the primary objective.

**Total per-step reward:**

```
R[n] = cascade_signal + overflow_urgency + thermal_margin
```

---

### Grader

Three components measuring cascade quality, not just survival:

**A — Cascade containment (50% weight)**: What fraction of the 30 steps had zero auto-trips? This is the primary metric — an agent that prevents all auto-trips in 28 out of 30 steps scores very differently from one that allowed trips throughout.

```
containment_ratio = steps_without_auto_trip / 30
```

**B — Thermal stability (30% weight)**: Among steps with no auto-trips, what fraction had all lines below 100%?

```
stability_ratio = safe_steps / containment_steps
```

**C — Recovery speed (20% weight)**: After the first overload appeared, how many steps did it take to bring all lines below 100%?

```
recovery_score = max(0, 1 - (steps_to_stabilize / 10))
```

**Total:**

```
score = 0.5 × containment_ratio + 0.3 × stability_ratio + 0.2 × recovery_score
```

---

### Key Distinction from Task 2

| Dimension | Task 2 (N-1) | Task 3 (Cascade Prevent) |
|-----------|-------------|--------------------------|
| **Core question** | Can you manage a degraded-but-stable grid? | Can you stop an active collapse in progress? |
| **Urgency** | Methodical — find new stable point | Immediate — line trips are imminent |
| **Key signal** | max_rho trending | timestep_overflow countdowns |
| **Failure mode** | Gradual deterioration | Sudden cascade at specific steps |
| **Reward shape** | Dense, gradual | Sparse rewards, sharp penalties |
| **What agent learns** | Post-contingency redispatch | Triage under time pressure |

---

