---
title: Grid2Op Environment
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# OpenEnv: Grid2op Environments

### Power grid topology control for reinforcement learning — four tasks, from overload relief to multi-stage cascade damage control.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Grid2Op](https://img.shields.io/badge/Grid2Op-l2rpn__case14__sandbox-green)](https://grid2op.readthedocs.io)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/Sidharth1743/grid2op-openenv)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Project write-up: https://medium.com/@jayasreeselvam37/openenv-grid2op-environment-c45ffbcb2cc4

## 📖 What is Grid2Op?

[Grid2Op](https://github.com/Grid2op/grid2op) is an open-source Python platform developed by _RTE France_ (the French transmission system operator) for modelling sequential decision-making on power grids. It powers the [L2RPN](https://l2rpn.chalearn.org/) — Learning to Run a Power Network — competition series used by researchers worldwide.

Think of it as _Gymnasium, but for electricity grids._ Every step() runs a full AC power flow simulation. Every observation reflects real physical quantities — line loading ratios, generator outputs, reactive power. Every action you take propagates through the grid via Kirchhoff's laws, not a lookup table.

This environment wraps l2rpn_case14_sandbox (the standard IEEE 14-bus benchmark)

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
- a public HF dataset repo for the main training datasets
- a public HF model repo for the final SFT adapter
- completed GRPO experiments and cloud training/eval infrastructure
- benchmark docs, plots, and submission notes

## Quick Start

### Prerequisites

- Python 3.10–3.12
- `uv`
- Docker for containerized deployment

### Install

```bash
uv venv
source .venv/bin/activate
env UV_CACHE_DIR=/tmp/uv-cache uv sync --no-dev
```

### Run the environment server

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev server --port 8000
```

Server URL:

- `http://127.0.0.1:8000`

### Smoke test the environment

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev grid2op-smoke --task-id single_fault --steps 1

curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'

curl http://127.0.0.1:8000/tasks
```

### Run the baseline agent locally

Create `.env`:

```env
GRID2OP_BASE_URL=http://127.0.0.1:8000
API_BASE_URL=https://router.huggingface.co/v1
HF_TOKEN=hf_your_token
MODEL_NAME=openai/gpt-oss-20b:groq
```

Then run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python inference.py --task-id single_fault
env UV_CACHE_DIR=/tmp/uv-cache uv run python inference.py --task-id n_minus_1
env UV_CACHE_DIR=/tmp/uv-cache uv run python inference.py --task-id cascade_prevent
env UV_CACHE_DIR=/tmp/uv-cache uv run python inference.py --task-id multi_stage_cascade
```

### Run tests

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/test_grid2op_env.py -q
```

### Run the medium-cost IEEE 118 SFT benchmark on HF Jobs

Script:

- [run_hf_ieee118_eval.sh](scripts/run_hf_ieee118_eval.sh)

Command:

```bash
export HF_TOKEN="your_hf_token"

HF_JOB_FLAVOR=t4-small \
ADAPTER_PATH=/mnt/models/grid2op-qwen3-4b-sft-3k-v1 \
EPISODES_PER_TASK=3 \
SEED_START=0 \
RUN_LABEL=ieee118-qwen3-4b-sft-transfer-3eps \
bash scripts/run_hf_ieee118_eval.sh
```

### Launch IEEE 118 GRPO training on HF Jobs

Notebook:

- [grid2op_training_colab.ipynb](submission/grid2op_training_colab.ipynb)

Direct launcher script:

- [run_hf_ieee118_grpo.sh](scripts/run_hf_ieee118_grpo.sh)

Required assets before launch:

- dataset in HF bucket:
  - `/mnt/datasets/ieee118_teacher_actions_v1.jsonl`
- public SFT adapter:
  - `Sidharth1743/grid2op-qwen3-4b-sft-final`

Command:

```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"

HF_JOB_FLAVOR=l40sx1 \
DATASET_PATH=/mnt/datasets/ieee118_teacher_actions_v1.jsonl \
ADAPTER_PATH=Sidharth1743/grid2op-qwen3-4b-sft-final \
RUN_NAME=grid2op-qwen3-4b-grpo-ieee118-v1 \
OUTPUT_PATH=/mnt/runs/grid2op-qwen3-4b-grpo-ieee118-v1 \
bash scripts/run_hf_ieee118_grpo.sh
```

## API

Important endpoints:

| Endpoint            | Method    | Purpose                                        |
| ------------------- | --------- | ---------------------------------------------- |
| `/reset`            | POST      | reset a task episode                           |
| `/step`             | POST      | apply a `GridAction`                           |
| `/state`            | GET       | inspect current state                          |
| `/simulate`         | POST      | test candidate actions without advancing state |
| `/planning_context` | GET       | graph and topology intelligence                |
| `/grader`           | POST      | deterministic task scoring                     |
| `/tasks`            | GET       | task list and descriptions                     |
| `/ws`               | WebSocket | OpenEnv-compatible persistent session          |

## 🌐 Environment Overview

### The Grid

All four tasks simulate the same physical network — the **IEEE 14-bus system**, a standard benchmark used in power systems research since the 1960s and adopted by RTE France for their L2RPN competitions.

                    BUS 1 (Slack)
                   /      \
            BUS 2           BUS 5
           /    \          /     \
        BUS 3   BUS 4 ── BUS 7   BUS 6
          |       |       |         |
        BUS 11  BUS 9   BUS 8    BUS 11
          |       |       |
        BUS 10  BUS 14  BUS 13 ── BUS 12

| Property           | Value                          |
| ------------------ | ------------------------------ |
| Environment ID     | `l2rpn_case14_sandbox`         |
| Grid standard      | IEEE 14-bus                    |
| Substations        | 14                             |
| Transmission lines | 20                             |
| Generators         | 6                              |
| Loads              | 11                             |
| Power flow solver  | `lightsim2grid` (full AC)      |
| Time resolution    | 5 minutes per step             |
| Episode length     | Up to 8,065 steps (~4.7 weeks) |
| Scenario pool      | 1,014 pre-recorded chronics    |

## The Scenario Dataset

Our companion repository, **[grid2op-data](https://github.com/Jayashree1743/grid2op-data)**, provides comprehensive data intelligence for all `1,014` scenarios.

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

- [graph_analysis.py](graph_analysis.py)
- [environment.py](server/environment.py)

### The Four Tasks at a Glance

Difficulty ────────────────────────────────────────────────────► Expert
│ │ │ │
single_fault n_minus_1 cascade_prevent multi_stage_cascade
│ │ │ │
10 steps 20 steps 30 steps 30 steps (3 × 10)
│ │ │ │
0 lines down 1 line down 1–2 lines 3 lines down + load +5–15% + load +20%
│ │ │ │
Relieve Survive N-1 Stop cascade Preserve viable
overload + reconnect propagation load islands

|                   |    `single_fault`     |         `n_minus_1`         |  `cascade_prevent`  |    `multi_stage_cascade`    |
| ----------------- | :-------------------: | :-------------------------: | :-----------------: | :-------------------------: |
| **Difficulty**    |        🟢 Easy        |          🟡 Medium          |       🔴 Hard       |          ⚫ Expert          |
| **Horizon**       |       10 steps        |          20 steps           |      30 steps       |          30 steps           |
| **Key signal**    |       `max_rho`       |        `ρ_max` trend        | `timestep_overflow` |   `available_load_ratio`    |
| **Win condition** | All lines < threshold | N-1 secure + reconnect line |   Zero auto-trips   | ≥50% load in viable islands |

## 🗂️ Project Structure

```
grid2op-openenv/
│
├── README.md                          # Main documentation
├── AGENTS.md                          # Development guidelines and conventions
├── pyproject.toml                     # Python package configuration
├── openenv.yaml                       # OpenEnv manifest (FastAPI, port 8000)
├── Dockerfile                         # Root container build
├── .env.example                       # Environment variable template
├── .gitignore / .dockerignore         # Version control + Docker exclusions
├── synthticdata_checklist.md          # Data validation checklist
├── _init_.py                        # Package initialization
│
├── models.py                          # Pydantic schemas — GridAction, GridObservation, GridState
├── client.py                          # OpenEnv client wrapper for API interaction
├── inference.py                       # Baseline LLM agent — Think → Simulate → Act
├── ft_inference.py                    # Fine-tuned model inference pipeline
├── graph_analysis.py                  # Topology intelligence — bridge lines, congestion analysis
│
├── server/
│   ├── app.py                         # FastAPI/OpenEnv entrypoint + HTTP routes
│   ├── environment.py                 # Grid2Op adapter with live AC simulation
│   ├── tasks.py                       # Task definitions + scenario injection
│   ├── graders.py                     # Deterministic per-task evaluation logic
│   ├── gradio_ui.py                   # Optional Gradio web UI
│   ├── logging_utils.py               # Server logging configuration
│   ├── requirements.txt               # Server-specific dependencies
│   └── Dockerfile                     # Server-focused container build
│
├── scripts/
│   ├── train_sft.py                   # Supervised fine-tuning pipeline
│   ├── train_grpo_verifier.py         # GRPO reinforcement learning training
│   ├── collect_teacher_dataset.py     # Dataset curation for SFT training
│   ├── filter_grpo_dataset.py         # GRPO dataset filtering
│   ├── check_dataset_quality.py       # Training data validation
│   ├── balance_dataset_actions.py     # Action distribution balancing
│   ├── diagnose_action_space.py       # Action space analysis + debugging
│   └── check_ft_inference_log.py      # Fine-tuning evaluation analysis
│
├── tests/
│   └── test_grid2op_env.py            # Integration tests — environment, graders, parsing
│
├── outputs/
│   ├── logs/                          # Training and evaluation logs
│   ├── evals/                         # Evaluation results and metrics
│   └── models/
│       ├── grid2op-qwen3-4b-sft-3k-v1/          # SFT-trained model
│       ├── grid2op-qwen3-4b-grpo-compact-v1/     # GRPO-trained model
│       └── grid2op-qwen3-4b-grpo-smoke-compact/  # Smoke test model
│
└── architecture/
    ├── architecture.md                # Full system architecture overview
    ├── task_1_architecture.md         # single_fault design notes
    ├── task_2_architecture.md         # n_minus_1 design notes
    ├── task_3_architecture.md         # cascade_prevent design notes
    └── task_4_architecture.md         # multi_stage_cascade design notes

```

## Deliverables

### Public Hugging Face Space

- Space: https://huggingface.co/spaces/Sidharth1743/grid2op-openenv

### Submission Files

- notebook launcher: [grid2op_training_colab.ipynb](submission/grid2op_training_colab.ipynb)
- submission readme: [README.md](submission/README.md)
- submission guide: [docs.md](submission/docs.md)
- submission validation script: [pre_validation.sh](submission/pre_validation.sh)
- sample inference helper: [sample_inference.py](submission/sample_inference.py)

### Training Code

- SFT / inference pipeline: [inference.py](inference.py)
- Verified-candidate evaluation: [ft_inference.py](ft_inference.py)
- GRPO trainer: [train_grpo_verifier.py](scripts/train_grpo_verifier.py)
- HF Jobs eval launcher: [run_hf_ieee118_eval.sh](scripts/run_hf_ieee118_eval.sh)
- HF Jobs GRPO launcher: [run_hf_ieee118_grpo.sh](scripts/run_hf_ieee118_grpo.sh)
- HF Jobs teacher-data launcher: [run_hf_ieee118_collect_teacher.sh](scripts/run_hf_ieee118_collect_teacher.sh)
- Public SFT model repo: https://huggingface.co/Sidharth1743/grid2op-qwen3-4b-sft-final
- Public dataset repo: https://huggingface.co/datasets/Sidharth1743/grid2op-openenv-datasets
- SFT training workspace: https://wandb.ai/sidhu1743/grid2op-openenv-sft/runs/olfjebdn?nw=nwusersid250581
- Completed compact GRPO run: https://wandb.ai/sidhu1743/grid2op-openenv-grpo/runs/swrnbnml?nw=nwusersid250581
- Dataset and experiment notes:
  - [evaluation.md](hack/evaluation.md)
  - [grpo_exp.md](hack/grpo_exp.md)
  - [benchmark.md](hack/benchmark.md)
  - [reward_hack.md](hack/reward_hack.md)

### Key Plots

- Main benchmark comparison: [benchmark_task_scores.png](hack/assets/benchmark_task_scores.png)
- Seen-vs-unseen generalization: [generalization_seen_vs_unseen.png](hack/assets/generalization_seen_vs_unseen.png)
- Focused multistage GRPO plot: [multistage_dapo_focus.png](hack/assets/multistage_dapo_focus.png)
- Project-level tradeoff view: [performance_vs_effort.png](hack/assets/performance_vs_effort.png)
- Safety failure summary: [safety_failures.png](hack/assets/safety_failures.png)
- Plot notes: [plots.md](hack/plots.md)

### Training Plot Exports

The repository also includes exported training curves under [training_plots](training_plots):

- [sft_train_loss.png](training_plots/sft_train_loss.png)
- [sft_eval_loss.png](training_plots/sft_eval_loss.png)
- [sft_train_entropy.png](training_plots/sft_train_entropy.png)
- [sft_eval_entropy.png](training_plots/sft_eval_entropy.png)

These are the committed image artifacts that back the SFT training story in addition to the W&B workspace.

Relevant benchmark and GRPO figures are also committed under [hack/assets](hack/assets).

### Environment and Scoring Docs

- reward and grader reference: [reward_and_scoring.md](docs/reward_and_scoring.md)
- RLVR and physics-verification design: [rlvr_environment.md](docs/rlvr_environment.md)
- task 1 spec: [task_1.md](docs/task_1.md)
- task 2 spec: [task_2.md](docs/task_2.md)
- task 3 spec: [task_3.md](docs/task_3.md)
- task 4 spec: [task_4.md](docs/task_4.md)

## Final Model Choice

The final submission model is:

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- adapter: `outputs/models/grid2op-qwen3-4b-sft-3k-v1`
- public model artifact: https://huggingface.co/Sidharth1743/grid2op-qwen3-4b-sft-final

Why this is the final model:

- it clearly improves over the base model on the hard tasks
- it is safe on both the main seed block and unseen seeds
- it remains the strongest fully evaluated model in the project

## Results

### Final SFT Benchmark Scores

Seed block `0..4`, `5` episodes per task:

| Task                  |       Score |
| --------------------- | ----------: |
| `single_fault`        |     `0.856` |
| `n_minus_1`           |     `0.990` |
| `cascade_prevent`     |     `0.990` |
| `multi_stage_cascade` | `0.9156444` |

Safety:

- failures: `0`
- safety pass: `true`

Unseen seed block `100..102`, `3` episodes per task:

| Task                  |       Score |
| --------------------- | ----------: |
| `single_fault`        |     `0.830` |
| `n_minus_1`           | `0.9222223` |
| `cascade_prevent`     |     `0.990` |
| `multi_stage_cascade` | `0.9069863` |

Safety:

- failures: `0`
- safety pass: `true`

### Base Vs SFT

Main seed block `0..4`:

| Task                  |    Base |         SFT |
| --------------------- | ------: | ----------: |
| `single_fault`        | `0.856` |     `0.856` |
| `n_minus_1`           | `0.952` |     `0.990` |
| `cascade_prevent`     | `0.000` |     `0.990` |
| `multi_stage_cascade` | `0.000` | `0.9156444` |

Most important change:

- the base model struggled on the hard tasks because it produced invalid or unverified actions
- the SFT model learned the environment-specific action protocol and completed the evaluated episodes safely

## Training Evidence

For reviewers who want the training trace directly:

- W&B SFT workspace: https://wandb.ai/sidhu1743/grid2op-openenv-sft/runs/olfjebdn?nw=nwusersid250581
- W&B compact GRPO run: https://wandb.ai/sidhu1743/grid2op-openenv-grpo/runs/swrnbnml?nw=nwusersid250581
- W&B DAPO-loss multistage GRPO run: https://wandb.ai/sidhu1743/grid2op-openenv-grpo/runs/yq5rgzg0?nw=nwusersid250581
- committed plot exports: [training_plots](training_plots)
- committed benchmark and GRPO figures: [hack/assets](hack/assets)

## Benchmark Tasks

All tasks run on the IEEE 14-bus sandbox grid:

| Task                  | Horizon | Core Objective                                        |
| --------------------- | ------: | ----------------------------------------------------- |
| `single_fault`        |    `10` | relieve overload through redispatch                   |
| `n_minus_1`           |    `20` | operate safely after a contingency and reconnect well |
| `cascade_prevent`     |    `30` | stop automatic trips from propagating                 |
| `multi_stage_cascade` |    `30` | preserve load across staged degradation               |

Task-specific notes:

- [task_1.md](docs/task_1.md)
- [task_2.md](docs/task_2.md)
- [task_3.md](docs/task_3.md)
- [task_4.md](docs/task_4.md)
- [reward_and_scoring.md](docs/reward_and_scoring.md)
- [rlvr_environment.md](docs/rlvr_environment.md)

## References

| Paper                                                                                                                | Venue                       | Task                                         |
| -------------------------------------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------- |
| Meng, Xu & Zhu — [Deep RL for Power Grid Multi-Stage Cascading Failure Mitigation](https://arxiv.org/abs/2505.09012) | ICLR 2025                   | Task 4 — MSCF reward + island viability rule |
| Dwivedi et al. — [RL for Mitigating Cascading Failures via Sensitivity Factors](https://arxiv.org/abs/2411.18050)    | NeurIPS 2024 WS             | Task 1 — physics reward + switching cost     |
| Marchesini et al. — [RL2Grid: Benchmarking RL in Power Grid Operations](https://arxiv.org/abs/2503.23101)            | 2025                        | Task 2 — three-component reward structure    |
| Yoon et al. — [Winning L2RPN: Semi-Markov Afterstate Actor-Critic](https://openreview.net/forum?id=LmUJqB1Cz8)       | ICLR 2021                   | Task 2 — activation threshold pattern        |
| Ramapuram Matavalam et al. — Curriculum RL for Cascade Prevention                                                    | IEEE Trans. Power Sys. 2023 | Task 3 — curriculum profiles                 |
| van der Sar et al. — [Centrally Coordinated MARL for Grid Topology Control](https://arxiv.org/abs/2502.08681)        | ACM e-Energy 2025           | Multi-agent context                          |
| Zhu — [Power Grid Cascading Failure Mitigation by RL](https://arxiv.org/abs/2108.10424)                              | 2021                        | Task 4 — MSCF MDP formulation                |
| Donnot — [Grid2Op](https://github.com/Grid2op/grid2op)                                                               | 2020                        | Core simulation platform                     |

## License

MIT
