---
title: Grid2Op Environment
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---
# Grid2Op OpenEnv Environment

> An OpenEnv-compatible power-grid control environment built on Grid2Op, with four benchmark tasks, a verified-action inference pipeline, and a strong SFT submission model.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Grid2Op](https://img.shields.io/badge/Grid2Op-l2rpn__case14__sandbox-green)](https://grid2op.readthedocs.io)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Sidharth1743/grid2op-openenv)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

## Problem

Modern power grids must keep electricity flowing even when lines overload, contingencies occur, or cascades begin. In practice, that means an operator must repeatedly answer a hard question:

> given the current grid state, what control action is both safe and operationally useful right now?

This repository turns that problem into an OpenEnv environment built on **Grid2Op**. The environment exposes four progressively harder control tasks on the IEEE 14-bus sandbox grid, ranging from simple overload relief to multi-stage cascade management.

The central design choice is simple:

- the simulator stays authoritative
- actions are checked against physics
- the model is evaluated on safe, verified grid control rather than free-form guessing

## What This Repository Contains

This repo includes:

- an OpenEnv-compatible server around `l2rpn_case14_sandbox`
- four benchmark tasks with task-specific rewards and graders
- a verified-candidate inference pipeline for LLM control
- a strong supervised fine-tuned submission model
- completed GRPO experiments and cloud training/eval infrastructure
- benchmark docs, plots, and submission notes

## Deliverables

### Public Hugging Face Space

- Space: https://huggingface.co/spaces/Sidharth1743/grid2op-openenv

### Training Code

- SFT / inference pipeline: [inference.py](/home/sidharth/Desktop/grid2op-openenv/inference.py)
- Verified-candidate evaluation: [ft_inference.py](/home/sidharth/Desktop/grid2op-openenv/ft_inference.py)
- GRPO trainer: [train_grpo_verifier.py](/home/sidharth/Desktop/grid2op-openenv/scripts/train_grpo_verifier.py)
- Dataset and experiment notes:
  - [evaluation.md](/home/sidharth/Desktop/grid2op-openenv/hack/evaluation.md)
  - [grpo_exp.md](/home/sidharth/Desktop/grid2op-openenv/hack/grpo_exp.md)
  - [benchmark.md](/home/sidharth/Desktop/grid2op-openenv/hack/benchmark.md)
  - [reward_hack.md](/home/sidharth/Desktop/grid2op-openenv/hack/reward_hack.md)

### Key Plots

- Main benchmark comparison: [benchmark_task_scores.png](/home/sidharth/Desktop/grid2op-openenv/hack/assets/benchmark_task_scores.png)
- Seen vs unseen seeds: [generalization_seen_vs_unseen.png](/home/sidharth/Desktop/grid2op-openenv/hack/assets/generalization_seen_vs_unseen.png)
- Safety / failures: [safety_failures.png](/home/sidharth/Desktop/grid2op-openenv/hack/assets/safety_failures.png)
- Focused multistage GRPO plot: [multistage_dapo_focus.png](/home/sidharth/Desktop/grid2op-openenv/hack/assets/multistage_dapo_focus.png)
- Project-level tradeoff view: [performance_vs_effort.png](/home/sidharth/Desktop/grid2op-openenv/hack/assets/performance_vs_effort.png)
- Plot notes: [plots.md](/home/sidharth/Desktop/grid2op-openenv/hack/plots.md)

## Final Model Choice

The final submission model is:

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- adapter: `outputs/models/grid2op-qwen3-4b-sft-3k-v1`

Why this is the final model:

- it clearly improves over the base model on the hard tasks
- it is safe on both the main seed block and unseen seeds
- completed GRPO runs were technically successful, but they did not beat the SFT model

## Results

### Final SFT Benchmark Scores

Seed block `0..4`, `5` episodes per task:

| Task | Score |
|---|---:|
| `single_fault` | `0.856` |
| `n_minus_1` | `0.990` |
| `cascade_prevent` | `0.990` |
| `multi_stage_cascade` | `0.9156444` |

Safety:

- failures: `0`
- safety pass: `true`

Unseen seed block `100..102`, `3` episodes per task:

| Task | Score |
|---|---:|
| `single_fault` | `0.830` |
| `n_minus_1` | `0.9222223` |
| `cascade_prevent` | `0.990` |
| `multi_stage_cascade` | `0.9069863` |

Safety:

- failures: `0`
- safety pass: `true`

### Base Vs SFT

Main seed block `0..4`:

| Task | Base | SFT |
|---|---:|---:|
| `single_fault` | `0.856` | `0.856` |
| `n_minus_1` | `0.952` | `0.990` |
| `cascade_prevent` | `0.000` | `0.990` |
| `multi_stage_cascade` | `0.000` | `0.9156444` |

Most important change:

- the base model often failed on the hard tasks because it produced invalid or unverified actions
- the SFT model learned the environment-specific action protocol and completed the evaluated episodes safely

## How The System Works

The environment uses a **verified-candidate control loop**:

1. reset the task
2. enumerate legal grid actions
3. simulate those actions with Grid2Op
4. prompt the model with verified candidate outcomes
5. require the model to output a valid `GridAction`
6. require the selected action to match one verified candidate exactly
7. execute and grade

This is important because the model is not rewarded for inventing arbitrary actions. It must operate inside a simulator-checked action set.

## Benchmark Tasks

All tasks run on the IEEE 14-bus sandbox grid:

| Task | Horizon | Core Objective |
|---|---:|---|
| `single_fault` | `10` | relieve overload through redispatch |
| `n_minus_1` | `20` | operate safely after a contingency and reconnect well |
| `cascade_prevent` | `30` | stop automatic trips from propagating |
| `multi_stage_cascade` | `30` | preserve load across staged degradation |

Task-specific notes:

- [task_1.md](/home/sidharth/Desktop/grid2op-openenv/docs/task_1.md)
- [task_2.md](/home/sidharth/Desktop/grid2op-openenv/docs/task_2.md)
- [task_3.md](/home/sidharth/Desktop/grid2op-openenv/docs/task_3.md)
- [task_4.md](/home/sidharth/Desktop/grid2op-openenv/docs/task_4.md)

## Why The Environment Is Strong

The benchmark is strong for three reasons:

1. every model uses the same verified-candidate evaluation path
2. each task has its own grader, aligned to its real objective
3. the project evaluates on both seen and unseen seed blocks

That makes the comparison between Base, SFT, and GRPO much more meaningful than a single reward number or a single cherry-picked trajectory.

## Quick Start

### Prerequisites

- Python 3.10–3.12
- `uv`
- Docker for containerized deployment

### Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Run the server

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev server --port 8000
```

Server URL:

- `http://127.0.0.1:8000`

### Smoke test

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

curl http://127.0.0.1:8000/tasks
```

### Run the baseline agent

Create `.env`:

```env
GRID2OP_BASE_URL=http://127.0.0.1:8000
API_BASE_URL=https://router.huggingface.co/v1
HF_TOKEN=hf_your_token
MODEL_NAME=openai/gpt-oss-20b:groq
```

Then run:

```bash
python inference.py --task-id single_fault
python inference.py --task-id n_minus_1
python inference.py --task-id cascade_prevent
python inference.py --task-id multi_stage_cascade
```

### Run tests

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/test_grid2op_env.py -q
```

## API

Important endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | reset a task episode |
| `/step` | POST | apply a `GridAction` |
| `/state` | GET | inspect current state |
| `/simulate` | POST | test candidate actions without advancing state |
| `/planning_context` | GET | graph and topology intelligence |
| `/grader` | POST | deterministic task scoring |
| `/tasks` | GET | task list and descriptions |
| `/ws` | WebSocket | OpenEnv-compatible persistent session |

## Graph And Topology Intelligence

The environment exposes structural guidance computed from the live grid state, including:

- `bridge_lines`
- `safe_to_disconnect`
- `n_minus_1_critical_lines`
- high-centrality buses
- congestion corridors
- islanded clusters

This helps the planner reason about grid structure, not just line overload numbers.

Main implementation:

- [graph_analysis.py](/home/sidharth/Desktop/grid2op-openenv/graph_analysis.py)
- [environment.py](/home/sidharth/Desktop/grid2op-openenv/server/environment.py)

## Project Structure

```text
grid2op-openenv/
├── server/                # OpenEnv + FastAPI environment server
├── models.py              # Shared Pydantic models
├── client.py              # Environment client helpers
├── inference.py           # Baseline agent and benchmark runner
├── ft_inference.py        # Verified-candidate fine-tuned evaluation runner
├── graph_analysis.py      # Graph intelligence and topology analysis
├── tests/                 # Pytest suite
├── docs/                  # Task and implementation notes
├── hack/                  # Submission notes, plots, and benchmark writeups
└── openenv.yaml           # OpenEnv manifest
```

## References

1. Donnot, B. et al. Grid2Op: sequential decision making in power systems.  
https://github.com/rte-france/grid2op

2. Learning to run a power network challenge for topology control.  
https://www.sciencedirect.com/science/article/abs/pii/S0378779620304387

3. RL2Grid benchmark paper.  
https://huggingface.co/papers/2503.23101

4. Multi-stage cascading failure mitigation with reinforcement learning.  
https://www.climatechange.ai/papers/iclr2025/1

5. OpenEnv / TRL environment integration.  
https://huggingface.co/docs/trl/openenv

## License

MIT
