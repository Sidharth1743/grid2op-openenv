# OpenEnv Hackathon Submission Requirements

## Overview

This document outlines all requirements for submitting an environment to the OpenEnv Hackathon. All submissions must meet the criteria defined in this document to be evaluated.

---

## 1. Task Requirements

### 1.1 Real-World Task Simulation
- The environment must simulate a task **humans actually do**
- **NOT** games or toys
- Examples of acceptable domains: email triage, code review, data cleaning, scheduling, customer support, content moderation, power grid management

### 1.2 OpenEnv Spec Compliance
- Implement the full OpenEnv interface:
  - Typed `Observation`, `Action`, and `Reward` Pydantic models
  - `step(action)` → returns `observation, reward, done, info`
  - `reset()` → returns initial observation
  - `state()` → returns current state
- Include `openenv.yaml` with metadata
- Tested via `openenv validate`

### 1.3 Minimum 3 Tasks with Agent Graders
- **Each task** must have:
  - A concrete objective an agent must accomplish
  - A programmatic grader that scores performance (0.0–1.0)
  - Clear, deterministic success/failure criteria
- **Difficulty progression**: easy → medium → hard

### 1.4 Meaningful Reward Function
- Provides signal over the full trajectory (not just binary end-of-episode)
- Rewards partial progress toward task completion
- Penalizes clearly undesirable behavior (e.g., infinite loops, destructive actions)

---

## 2. Functional Requirements

### 2.1 Baseline Inference Script
- Must be named `inference.py` and placed in the **root directory**
- Use the OpenAI API client to run a model against the environment
- Read API credentials from environment variables:
  - `API_BASE_URL` - The API endpoint for the LLM
  - `MODEL_NAME` - The model identifier to use for inference
  - `HF_TOKEN` - Your Hugging Face / API key
- Produce a reproducible baseline score on all tasks

### 2.2 Structured Logging
- Emit structured stdout logs strictly following the format:
  - `[START]` - Episode start
  - `[STEP]` - Each step
  - `[END]` - Episode end
- Any deviation in field names, ordering, or formatting will result in incorrect evaluation

---

## 3. Deployment Requirements

### 3.1 Hugging Face Spaces
- Environment must run as a containerized HF Space tagged with `openenv`
- Automated ping to the Space URL — must return 200 and respond to `reset()`

### 3.2 Containerized Execution
- Must include a working `Dockerfile`
- The environment should start cleanly with `docker build && docker run`

### 3.3 Infrastructure Restrictions
- Runtime of inference script should be **less than 20 minutes**
- Must run on a machine with `vcpu=2, memory=8gb`

---

## 4. Documentation Requirements

### 4.1 README
Must include:
- Environment description and motivation
- Action and observation space definitions
- Task descriptions with expected difficulty
- Setup and usage instructions
- Baseline scores

---

## 5. Evaluation Criteria

### 5.1 Parameter Weights

| Parameter | Weight | Description |
|-----------|--------|-------------|
| **Real-world utility** | 30% | Does the environment model a genuine task? Would someone actually use this to train or evaluate agents? |
| **Task & grader quality** | 25% | Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression? |
| **Environment design** | 20% | Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries |
| **Code quality & spec compliance** | 15% | Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works |
| **Creativity & novelty** | 10% | Novel problem domain, interesting mechanics, clever reward design, original approach |

### 5.2 Scoring Breakdown

#### Real-world utility (30%)
- 0–5: Toy/artificial problem with no practical application
- 6–15: Valid domain but shallow modeling of the real task
- 16–25: Good domain modeling, would be useful for agent evaluation
- 26–30: Excellent — fills a real gap, immediate value for the RL/agent community

#### Task & grader quality (25%)
- ✅ 3+ tasks with difficulty range?
- ✅ Graders produce scores between 0.0–1.0?
- ✅ Graders deterministic and reproducible?
- ✅ Hard task genuinely challenges frontier models?

#### Environment design (20%)
- ✅ `reset()` produces clean state?
- ✅ Action/observation types well-designed and documented?
- ✅ Reward function provides useful varying signal (not just sparse)?
- ✅ Episode boundaries sensible?dob

#### Code quality & spec compliance (15%)
- ✅ `openenv validate` passes?
- ✅ `docker build && docker run` works?
- ✅ HF Space deploys and responds?
- ✅ Baseline script runs and reproduces scores?

#### Creativity & novelty (10%)
- ✅ Domain we haven't seen in OpenEnv before?
- ✅ Reward design has interesting properties?
- ✅ Clever mechanics that make the environment engaging?

---

## 6. Validation Checklist

Before submitting, ensure:

- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] HF Space deploys and responds to `reset()`
- [ ] Baseline inference script runs without error
- [ ] 3+ tasks with graders (scores in 0.0–1.0 range)
- [ ] `inference.py` named correctly and in root directory
- [ ] Environment variables defined: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] Structured logs follow `[START]`, `[STEP]`, `[END]` format
- [ ] Runtime under 20 minutes
- [ ] Works on 2 vCPU, 8GB RAM machine

---

## 7. Judging Phases

### Phase 1: Automated Validation (Pass/Fail)
- HF Space deploys
- OpenEnv spec compliance
- Dockerfile builds
- Baseline reproduces
- 3+ tasks with graders

### Phase 2: Agentic Evaluation (Scored)
- Baseline agent re-run
- Standard Open LLM agent (e.g., Nemotron 3 Super) run against all environments
- Score variance check

### Phase 3: Human Review
- Top submissions reviewed by Meta and HuggingFace engineers
- Real-world utility check
- Creativity check
- Exploit checks

---

## 8. Disqualification Criteria

The following will result in disqualification:

- Environment does not deploy or respond
- Plagiarized or trivially modified existing environments
- Graders that always return the same score

---

## 9. Example: Power Grid Environment (Reference)

Your environment should follow a similar structure:

```
project/
├── inference.py           # Baseline inference script
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container configuration
├── README.md             # Documentation
├── src/
│   ├── tasks.py          # Task definitions (3+ tasks)
│   ├── graders.py        # Task graders (scores 0.0-1.0)
│   ├── environment.py    # Environment implementation
│   └── models.py         # Typed Observation/Action models
└── requirements.txt       # Dependencies
```

---

## Summary Checklist

| Requirement | Mandatory? |
|-------------|-------------|
| Real-world task (not games) | ✅ Yes |
| OpenEnv spec compliance | ✅ Yes |
| 3 tasks (easy→medium→hard) | ✅ Yes |
| Graders (0.0–1.0 scores) | ✅ Yes |
| Meaningful reward function | ✅ Yes |
| `inference.py` in root | ✅ Yes |
| HF_TOKEN, MODEL_NAME, API_BASE_URL | ✅ Yes |
| Structured logs [START/STEP/END] | ✅ Yes |
| HF Space deploys | ✅ Yes |
| Dockerfile works | ✅ Yes |
| Runtime < 20 min | ✅ Yes |
| 2 vCPU, 8GB RAM | ✅ Yes |
| README with setup instructions | ✅ Yes |