# Grid2Op OpenEnv Environment

> **Power grid topology control for reinforcement learning — four tasks, from overload relief to multi-stage cascade damage control.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Grid2Op](https://img.shields.io/badge/Grid2Op-l2rpn__case14__sandbox-green)](https://grid2op.readthedocs.io)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)](https://huggingface.co/spaces/Sidharth1743/grid2op-openenv)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## What This Is

This environment wraps the **IEEE 14-bus power grid** (`l2rpn_case14_sandbox`) as a fully compliant [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment. It exposes four tasks of increasing difficulty, each grounded in real power systems research, with deterministic graders and a physics-backed simulation endpoint.

The design principle is simple: **the server owns the simulation state**. Planning uses `obs.simulate()` on the live session rather than a local mirror, so simulation results are always exact.

The included baseline agent uses a **Think → Simulate → Act** loop: the LLM proposes candidate actions, the server validates each one via physics simulation, and the LLM selects the safest option. This is a physics-grounded LLM planner, not a zero-shot guesser.

---

## Quick Start

### Prerequisites

- Python 3.10–3.12
- `uv` (recommended) or `pip`
- Docker (for containerised deployment)

### 1. Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2. Run the server

```bash
uv run server --port 7860
```

The server listens at `http://127.0.0.1:7860`.

### 3. Smoke test

```bash
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{}'

curl http://127.0.0.1:7860/tasks
```

### 4. Run the baseline agent

Create a `.env` file:

```env
GRID2OP_BASE_URL=http://127.0.0.1:7860
API_BASE_URL=https://router.huggingface.co/v1
HF_TOKEN=hf_your_token
MODEL_NAME=openai/gpt-oss-20b:groq
```

Then run against any task:

```bash
python inference.py --task-id single_fault
python inference.py --task-id n_minus_1
python inference.py --task-id cascade_prevent
python inference.py --task-id multi_stage_cascade
```

### 5. Run tests

```bash
uv run --extra dev pytest tests/test_grid2op_env.py -q
```

---

## Docker

```bash
docker build -t grid2op-env:local .
docker run --rm -p 7860:7860 grid2op-env:local
```

The image pre-downloads `l2rpn_case14_sandbox` at build time so runtime startup is instant.

---

## Grid

All four tasks use the same underlying grid:

| Property | Value |
|---|---|
| Environment | `l2rpn_case14_sandbox` (IEEE 14-bus) |
| Substations | 14 |
| Transmission lines | 20 |
| Generators | 6 |
| Loads | 11 |
| Power flow backend | `lightsim2grid` (AC) |
| Time resolution | 5 minutes per step |
| Scenario data | Pre-recorded chronics (load + generation time series) |

---

## Action Space

Agents submit a `GridAction` with any combination of:

| Field | Type | Description |
|---|---|---|
| `line_set` | `dict[int, int]` | Map of `line_id → status` where `1` = connect, `-1` = disconnect |
| `redispatch` | `dict[int, float]` | Map of `gen_id → delta_mw` (subject to ramp limits) |
| `do_nothing` | `bool` | Explicit no-op |

```json
{
  "line_set": {"4": -1},
  "redispatch": {"2": 15.0},
  "do_nothing": false
}
```

All actions are validated server-side. Invalid indices are silently stripped. Ramp limits are enforced. Bridge lines that would island the network are rejected before reaching the physics solver.

---

## Observation Space

Each step returns a `GridObservation`:

| Field | Shape | Description |
|---|---|---|
| `rho` | `[20]` | Line loading ratio — `1.0` = thermal limit |
| `gen_p` | `[6]` | Generator active power output (MW) |
| `load_p` | `[11]` | Load active power demand (MW) |
| `line_status` | `[20]` | `true` = connected, `false` = disconnected |
| `timestep_overflow` | `[20]` | Consecutive steps each line has been overloaded |
| `reward` | `float` | Shaped reward for this step |
| `done` | `bool` | Whether the episode has ended |
| `metadata` | `dict` | Task-specific fields (stage index, available load ratio, etc.) |

---

## Tasks

### Task 1 — `single_fault` · Easy · 10 steps

**Scenario**: The grid is intact but one or more lines are running hot (90–98% loading). No lines have tripped yet, but the chronic is trending toward overload.

**Objective**: Bring all lines below the safe threshold within 10 steps.

**What makes it easy**: Single problem region, full topology intact, multiple redundant paths available.

**Reward**:
- `+(1 - ρ²)` per line per step — quadratic physics reward that penalises proximity to limits more sharply than linear alternatives
- `−µ_line × n_switches` — switching cost that prevents reward hacking via excessive topology changes
- Speed bonus on resolution, decaying with each step used

**Grader**: Score based on whether all lines reached below threshold and how many steps it took. Faster resolution = higher score.

**Research basis**: Physics reward formulation from Dwivedi et al. (2024) [arXiv:2411.18050]. Switching cost design from the same paper's `µ_line × c_line × W_ℓ[n]` formulation.

---

### Task 2 — `n_minus_1` · Medium · 20 steps

**Scenario**: Line 0 is disconnected at reset (N-1 contingency). The remaining 19 lines absorb the rerouted flow. Several lines are immediately pushed to 70–90% loading.

**Objective**: Clear the emergency, maintain N-1 secure operation for 20 steps, and reconnect the faulted line when its cooldown expires.

**What makes it medium**: The agent must manage sustained stress across multiple lines over a longer horizon, and must understand that the cooldown on line 0 creates a reconnection opportunity that should not be missed.

**Reward**: Three-component RL2Grid structure:

```
R = 0.3 × R_survive + 0.6 × R_overload + 0.1 × R_cost
```

- `R_survive`: constant +1.0 per step alive
- `R_overload`: clipped thermal margin `mean(clip(1 - ρ, -1, 1))`
- `R_cost`: economic penalty for redispatch proportional to `|delta_mw| / max_ramp`
- Reconnection bonus: +2.0 when a faulted line is successfully reconnected without worsening loading

**Grader**:
- 30% emergency response quality — did all lines reach below `ρ_danger = 0.92` within 5 steps?
- 50% sustained secure operation — fraction of steps 6–20 with `ρ_max < 0.90`
- 20% reconnection achievement — binary, did line 0 get reconnected?

**Research basis**: Three-component reward from Marchesini et al. (2025) [arXiv:2503.23101]. Activation threshold pattern from Yoon et al. (2021), ICLR 2021. Reconnection heuristic from the L2RPN 2023 winning agent (LJNAgent).

---

### Task 3 — `cascade_prevent` · Hard · 30 steps

**Scenario**: One or two lines are already disconnected and load is elevated by 5–15%. Several remaining lines are near or above their thermal limits. Grid2Op is counting down to automatic line trips — and each trip redistributes flow, potentially overloading more lines.

**Objective**: Prevent automatic line trips from propagating into a cascade for 30 steps.

**What makes it hard**: The key signal is not `max_rho` but `timestep_overflow` — the per-line countdown to automatic disconnection. A line at 103% with `overflow=2` is more urgent than a line at 95% with `overflow=0`, even though the second has higher absolute loading. Triage under time pressure, not global optimisation.

**Reward**:
- `+0.3` per step with no automatic trip
- `−2.5` per automatic trip detected
- `−0.05 × Σ overflow²` — quadratic overflow penalty that escalates with each step of inaction
- `+0.1 × mean(clip(1 - ρ, -1, 1))` — thermal margin signal
- Terminal: survival bonus reduced proportionally by auto-trip count, blackout penalty `−12.0`

**Grader**:
- 50% cascade containment — `steps_without_auto_trip / 30`
- 30% thermal stability — fraction of safe steps with all lines below 100%
- 20% recovery speed — how fast the grid reached `ρ_max < 1.0` after initial stress

**Research basis**: Cascade prevention framing and curriculum profiles from Ramapuram Matavalam et al. (2023), IEEE Transactions on Power Systems. Thermal margin shaping from RL2Grid [arXiv:2503.23101]. Quadratic overflow penalty from physics urgency analysis.

---

### Task 4 — `multi_stage_cascade` · Expert · 30 steps (3 × 10)

**Scenario**: Three lines are simultaneously disconnected and load is increased by 20%. The overflow window is shortened to 2 steps. The grid will fragment — the question is not whether a cascade occurs but how much viable load survives it.

**Objective**: Preserve as much load in self-sustaining grid islands as possible across three 10-step stages.

**What makes it expert**: Cascade propagation is physically inevitable. Actions that appear beneficial in Stage 1 can destroy island viability in Stage 2. The agent must plan across stage boundaries, not just the current timestep.

**Island viability rule**: For each connected component after fragmentation:
- `gen_total ≥ load_total` → island is available (self-sustaining)
- `gen_total < load_total` → island is unavailable (will collapse)

**Key metric**: `available_load_ratio` — fraction of original total load still located in viable islands at each step.

**Reward** (four-component MSCF structure):
- `−0.02 × (total_gen / initial_load)` — generation cost penalty per step
- `+0.5 × available_island_ratio` — reward for keeping more islands viable
- `−5.0 × (1 − available_load_ratio)` — stage-boundary load-loss penalty (applied at steps 10, 20)
- `+8.0 × (available_load_ratio²)` — terminal win reward if ≥50% load preserved at step 30
- `−12.0` — early collapse or convergence failure

**Grader**:
- 30% stage completion — did the agent cross step 10, step 20, and step 30?
- 40% load preservation — `available_load_ratio` at episode end (largest component)
- 20% island quality — fraction of stage boundaries where majority of islands were viable
- 10% speed — how fast each stage reached all lines below 100%

**Research basis**: MSCF formulation, island availability assessment (`Max_Gen_Total ≥ Load_Total`), and four-component reward structure from Meng, Xu & Zhu (2025) [arXiv:2505.09012], ICLR 2025. Stage-interdependence principle and continuous action design from the same paper. Earlier MSCF MDP formulation from Zhu (2021) [arXiv:2108.10424].

---

## Task Summary

| | Task 1 | Task 2 | Task 3 | Task 4 |
|---|---|---|---|---|
| **Name** | `single_fault` | `n_minus_1` | `cascade_prevent` | `multi_stage_cascade` |
| **Difficulty** | Easy | Medium | Hard | Expert |
| **Max steps** | 10 | 20 | 30 | 30 |
| **Lines down at reset** | 0 | 1 | 1–2 | 3 |
| **Load increase** | 0% | 0% | +5% to +15% | +20% |
| **Core question** | Relieve overload | Survive degraded grid | Stop cascade propagation | Preserve viable load |
| **Key signal** | `max_rho` | `ρ_max` trending | `timestep_overflow` | `available_load_ratio` |
| **Unique reward** | Quadratic physics reward | Three-component RL2Grid | Quadratic overflow penalty | Load preservation + island quality |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset the environment; accepts `task_id`, `seed`, `difficulty_level` |
| `/step` | POST | Execute a `GridAction`; returns `GridObservation` |
| `/state` | GET | Current episode metadata and `episode_id` |
| `/tasks` | GET | List all tasks with descriptions and action schema |
| `/simulate` | POST | Simulate candidate actions on the live session without advancing state |
| `/planning_context` | GET | Graph topology intelligence for the current episode |
| `/grader` | POST | Score a completed episode using the deterministic grader |
| `/baseline` | POST | Run the full baseline agent against all tasks and return scores |
| `/ws` | WebSocket | OpenEnv-compliant persistent session interface |

---

## Baseline Agent — Think → Simulate → Act

The agent in `inference.py` implements a physics-grounded LLM planning loop:

```
reset()
  └─ state() → episode_id
      └─ planning_context() → graph topology, safe actions, LODF guidance
          └─ LLM proposes 3 candidate actions
              └─ /simulate → physics validation for each candidate
                  └─ LLM selects safest validated action
                      └─ step(action)
                          └─ /grader at episode end
```

Key properties:
- Candidates with `convergence_failed=True` are filtered before LLM final selection
- Bridge lines (whose removal would island the network) are excluded from the candidate pool
- Ramp limits are exposed in the prompt to prevent duplicate redispatch proposals after sanitisation
- Context window is kept compact: only lines above 80%, active overflow countdowns, and stage-specific metadata are included

---

## Baseline Scores

Scores from the baseline agent (`Qwen2.5-14B-Instruct` via HuggingFace Inference API):

| Task | Score | Episode Length |
|---|---|---|
| `single_fault` | 0.752 | 7 / 10 |
| `n_minus_1` | 1.000 | 20 / 20 |
| `cascade_prevent` | 0.546 | 18 / 30 |
| `multi_stage_cascade` | ~0.15 | In progress |

Scores are deterministic — same seed, same model, same score. The `/baseline` endpoint reproduces these results end-to-end.

---

## Repository Layout

```
grid2op-openenv/
├── server/
│   ├── app.py             # FastAPI + OpenEnv entrypoint, WebSocket, extra HTTP routes
│   ├── environment.py     # Grid2Op adapter, reward shaping, planning support
│   ├── tasks.py           # Reset logic and scenario injection for all four tasks
│   └── graders.py         # Deterministic episode graders
├── models.py              # GridAction, GridObservation, GridState, EpisodeStepLog
├── client.py              # EnvClient (WebSocket)
├── inference.py           # Baseline LLM agent
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml
├── Dockerfile
├── architecture/          # Per-task architecture notes
└── docs/                  # Per-task research grounding and design notes
```

---

## References

1. Meng, B., Xu, C., & Zhu, Y. (2025). *Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation*. ICLR 2025. [arXiv:2505.09012](https://arxiv.org/abs/2505.09012)

2. Dwivedi, A., Tajer, A., Paternain, S., & Virani, N. (2024). *RL for Mitigating Cascading Failures: Targeted Exploration via Sensitivity Factors*. NeurIPS 2024 Workshop. [arXiv:2411.18050](https://arxiv.org/abs/2411.18050)

3. Marchesini, E., Marzari, L., & Leofante, F. (2025). *RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations*. [arXiv:2503.23101](https://arxiv.org/abs/2503.23101)

4. Yoon, D., Hong, S., Lee, B.-J., & Kim, K.-E. (2021). *Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic*. ICLR 2021. [OpenReview](https://openreview.net/forum?id=LmUJqB1Cz8)

5. Ramapuram Matavalam, A. R., Guddanti, K. P., Weng, Y., & Ajjarapu, V. (2023). *Curriculum Based Reinforcement Learning of Grid Topology Controllers to Prevent Thermal Cascading*. IEEE Transactions on Power Systems, 38(5), 4206–4220.

6. van der Sar, E. et al. (2025). *Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control*. ACM e-Energy 2025. [arXiv:2502.08681](https://arxiv.org/abs/2502.08681)

7. Zhu, Y. (2021). *Power Grid Cascading Failure Mitigation by Reinforcement Learning*. [arXiv:2108.10424](https://arxiv.org/abs/2108.10424)

8. Donnot, B. (2020). *Grid2Op: A Testbed Platform to Model Sequential Decision Making in Power Systems*. [GitHub](https://github.com/rte-france/grid2op)

---

## License

MIT