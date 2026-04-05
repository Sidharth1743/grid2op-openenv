# Task 4: `multi_stage_cascade` - Multi-Stage Cascading Failure Mitigation (Expert)

## Overview

Task 4 (`multi_stage_cascade`) is the **hardest** task in the Grid2Op environment. It simulates a **guaranteed multi-stage cascading failure** where cascade stages are pre-determined to progress - the agent cannot prevent them, only minimize damage across each stage boundary.

The task is based on the **Multi-Stage Cascading Failure (MSCF)** concept, emphasizing that stages are interdependent: an action that helps in Stage 1 may make Stage 2 unrecoverable. The agent must plan holistically across all three stages.

---

## 1. Task Definition

From `grid2op_env/server/tasks.py:53-62`:

```python
"multi_stage_cascade": TaskSpec(
    task_id="multi_stage_cascade",
    difficulty="hard",
    description=(
        "Three lines are disconnected and load is increased by 20%. Manage the "
        "guaranteed three-stage cascade for 30 steps and preserve as much load "
        "as possible across stage boundaries."
    ),
    max_steps=30,
),
```

**Key Characteristics:**
- **Difficulty**: Hard (Expert level)
- **Max steps**: 30 (3 stages × 10 steps each)
- **Goal**: Preserve maximum load across guaranteed cascade stages
- **Load increase**: **20%** (updated from 15% in architecture)

---

## 2. Key Distinction: Task 3 vs Task 4

| Dimension | Task 3 (cascade_prevent) | Task 4 (multi_stage_cascade) |
|-----------|--------------------------|------------------------------|
| **Core question** | Stop the cascade from propagating | Minimize damage across guaranteed cascade |
| **Can cascade be stopped?** | Yes - full prevention is achievable | No - cascade progression is guaranteed |
| **Agent mindset** | Prevention | Damage control and triage |
| **Stage structure** | No explicit stages | Three explicit stages with boundaries |
| **Faults at reset** | 1-2 lines, +5-15% load | **3 lines, +20% load** |
| **Key observation** | `timestep_overflow` countdowns | `available_load_ratio`, `available_island_ratio` |
| **Failure mode** | Allowing dominoes to start | Losing too much load at each stage |

---

## 3. Reset Phase

From `grid2op_env/server/tasks.py:566-620`:

### Scenario Configuration

```python
def _reset_multi_stage_cascade(
    env,
    seed: int | None,
    attempt: int,
    difficulty_level: int | None,
    scenario_mode: ScenarioMode,
    benchmark_tier: str | None,
):
    del difficulty_level
    stage, faulted_lines, load_scale = _multi_stage_profile(seed)
    
    # Set overflow window to 2 (faster cascades)
    _set_overflow_window(env, allowed_steps=2)
    
    total = _available_time_series_count(env)
    for offset in range(total):
        time_series_id = int(((0 if seed is None else int(seed) * 131) + attempt + offset) % total)
        options = {"time serie id": time_series_id}
        base_obs = env.reset(seed=seed, options=options)
        
        # Increase load by 20% (load_scale = 1.20)
        load_p = [float(v) for v in (base_obs.load_p * load_scale).astype(float).tolist()]
        
        # Disconnect THREE lines
        scenario_options = {
            "time serie id": time_series_id,
            "init state": {
                "set_line_status": [(line_id, -1) for line_id in faulted_lines],
                "injection": {"load_p": load_p},
            },
        }
        obs = env.reset(seed=seed, options=scenario_options)
        
        # Survival probe: must survive 5 steps with do_nothing
        if _multi_stage_survives_probe(env, min_steps=5):
            replayed = env.reset(seed=seed, options=scenario_options)
            return replayed, {
                "faulted_lines": faulted_lines,
                "load_scale": load_scale,
                "time_series_id": time_series_id,
                "curriculum_episode": 1,
                "curriculum_stage": stage,
                "scenario_mode": scenario_mode,
                "benchmark_tier": benchmark_tier or "multi_stage_cascade_expert",
                "benchmark_valid": True,
                "stage_count": 3,
                "stage_length": 10,
                "initial_total_load_mw": round(float(sum(load_p)), 6),
                "overflow_window": 2,
                "do_nothing_probe_steps": 5,
            }

    raise Grid2OpException(
        "Could not find a viable multi_stage_cascade chronic where do_nothing survives 5 steps under the calibrated 3-line +20% load scenario"
    )
```

### Line Triplets

From `grid2op_env/server/tasks.py:70-74`:

```python
MULTI_STAGE_LINE_TRIPLETS: List[tuple[int, int, int]] = [
    (2, 4, 14),
    (2, 4, 15),
    (4, 14, 16),
]
```

**Three lines are disconnected at reset** - more severe than Task 3's 1-2 lines.

### Profile Function

From `grid2op_env/server/tasks.py:335-338`:

```python
def _multi_stage_profile(seed: int | None) -> tuple[str, list[int], float]:
    triplet_index = 0 if seed is None else int(seed) % len(MULTI_STAGE_LINE_TRIPLETS)
    selected_triplet = list(MULTI_STAGE_LINE_TRIPLETS[triplet_index])
    return "expert_three_stage", selected_triplet, 1.20  # 20% load increase
```

### Survival Probe

From `grid2op_env/server/tasks.py:623-633`:

```python
def _multi_stage_survives_probe(env, min_steps: int) -> bool:
    obs = None
    for _ in range(int(min_steps)):
        obs, _, done, _ = env.step(env.action_space())
        if done:
            return False
        if obs is None or not obs.rho.tolist():
            return False
        if max(float(value) for value in obs.rho.tolist()) <= 0.0:
            return False
    return obs is not None
```

### Reset Parameters Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Lines disconnected | **3** | Severe (more than Task 3) |
| Load increase | **20%** | Updated (was 15%) |
| Stage count | 3 | Explicit multi-stage structure |
| Stage length | 10 | 10 steps per stage |
| Overflow window | 2 | Faster than default (3) - cascades happen quicker |
| Do-nothing probe | 5 steps | Must survive 5 steps with no action |
| Time series selection | Calibrated | Tested per chronic for viability |

---

## 4. Stage Structure

### Stage Boundaries

The 30-step episode is divided into three stages:

```
Stage 1: steps 1-10   →  Automatic overflow disconnection at step 10
Stage 2: steps 11-20  →  Island assessment → automatic overflow disconnection at step 20
Stage 3: steps 21-30 →  Final stage → episode ends
```

### Stage Metadata Computation

From `grid2op_env/server/grid_environment.py:750-765`:

```python
def _multi_stage_metadata(self, obs) -> dict[str, float | int | bool]:
    stage_length = int(self._state.scenario_metadata.get("stage_length", 10))
    step_count = max(0, int(self._state.step_count))
    stage_index = min(3, max(1, (step_count // stage_length) + 1))
    stage_boundary_assessed = step_count in {stage_length, 2 * stage_length}  # True at steps 10, 20
    stage_end = min(stage_index * stage_length, int(self._max_steps))
    steps_to_stage_boundary = max(0, stage_end - step_count)
    island_metrics = self._assess_island_availability(obs)
    return {
        "stage_index": stage_index,                      # 1, 2, or 3
        "steps_to_stage_boundary": steps_to_stage_boundary,
        "available_load_ratio": round(float(island_metrics["available_load_ratio"]), 6),
        "available_island_ratio": round(float(island_metrics["available_island_ratio"]), 6),
        "stage_boundary_assessed": stage_boundary_assessed,  # True at step 10 or 20
        "majority_islands_available": bool(island_metrics["majority_islands_available"]),
    }
```

---

## 5. Island Assessment

This is the **core mechanism** that distinguishes Task 4 - the ability to assess whether each connected component (island) can sustain itself.

From `grid2op_env/server/grid_environment.py:767-814`:

```python
def _assess_island_availability(self, obs) -> dict[str, float | bool]:
    initial_total_load = max(
        float(self._state.scenario_metadata.get("initial_total_load_mw", sum(obs.load_p.tolist()))),
        1e-6,
    )
    components = self._connected_components_from_obs(obs)
    if not components:
        return {
            "available_load_ratio": 0.0,
            "available_island_ratio": 0.0,
            "majority_islands_available": False,
        }

    gens_by_sub: dict[int, list[int]] = defaultdict(list)
    loads_by_sub: dict[int, list[int]] = defaultdict(list)
    for gen_id, sub_id in enumerate(self._env.gen_to_subid.tolist()):
        gens_by_sub[int(sub_id)].append(int(gen_id))
    for load_id, sub_id in enumerate(self._env.load_to_subid.tolist()):
        loads_by_sub[int(sub_id)].append(int(load_id))

    available_load_mw = 0.0
    available_islands = 0
    evaluated_islands = 0
    for component in components:
        component_load = 0.0
        component_gen_capacity = 0.0
        for sub_id in component:
            component_load += sum(float(obs.load_p[load_id]) for load_id in loads_by_sub.get(int(sub_id), []))
            component_gen_capacity += sum(
                float(self._env.gen_pmax[gen_id]) for gen_id in gens_by_sub.get(int(sub_id), [])
            )
        if component_load <= 0.0 and component_gen_capacity <= 0.0:
            continue
        evaluated_islands += 1
        
        # Island is AVAILABLE if: gen_capacity >= load
        # Island is UNAVAILABLE if: gen_capacity < load (will collapse)
        if component_gen_capacity + 1e-6 >= component_load:
            available_islands += 1
            available_load_mw += component_load

    available_island_ratio = (
        float(available_islands) / float(evaluated_islands) if evaluated_islands else 0.0
    )
    return {
        "available_load_ratio": available_load_mw / initial_total_load,
        "available_island_ratio": available_island_ratio,
        "majority_islands_available": bool(
            evaluated_islands > 0 and available_islands > (evaluated_islands / 2.0)
        ),
    }
```

### Island Detection Algorithm

From `grid2op_env/server/grid_environment.py:816-849`:

```python
def _connected_components_from_obs(self, obs) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {sub_id: set() for sub_id in range(int(self._env.n_sub))}
    line_status = [bool(value) for value in obs.line_status.tolist()]
    for line_id, connected in enumerate(line_status):
        if not connected:
            continue
        origin = int(self._env.line_or_to_subid[line_id])
        extremity = int(self._env.line_ex_to_subid[line_id])
        adjacency[origin].add(extremity)
        adjacency[extremity].add(origin)

    active_substations = {
        int(sub_id)
        for sub_id in range(int(self._env.n_sub))
        if any(int(load_sub) == int(sub_id) for load_sub in self._env.load_to_subid.tolist())
        or any(int(gen_sub) == int(sub_id) for gen_sub in self._env.gen_to_subid.tolist())
        or adjacency[int(sub_id)]
    }
    components: list[set[int]] = []
    visited: set[int] = set()
    for root in sorted(active_substations):
        if root in visited:
            continue
        stack = [root]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components
```

### Island Viability Rule

```
Island is VIABLE if:  gen_max_capacity >= total_load
Island is UNVIABLE if:  gen_max_capacity < total_load (will collapse at next stage boundary)
```

---

## 6. Reward Function

From `grid2op_env/server/grid_environment.py:630-647`:

```python
elif self._task_id == "multi_stage_cascade":
    initial_total_load = max(
        float(self._state.scenario_metadata.get("initial_total_load_mw", 0.0)),
        1e-6,
    )
    available_load_ratio = float(observation.metadata.get("available_load_ratio", 0.0))
    available_island_ratio = float(observation.metadata.get("available_island_ratio", 0.0))
    total_generation = sum(max(0.0, float(value)) for value in observation.gen_p)
    
    # Component 1: Generation cost penalty
    reward -= self.MULTI_STAGE_GENERATION_COST_SCALE * (total_generation / initial_total_load)
    
    # Component 2: Convergence reward
    reward += self.MULTI_STAGE_CONVERGENCE_SCALE * available_island_ratio
    
    # Component 3: Load loss penalty (only at stage boundaries)
    if bool(observation.metadata.get("stage_boundary_assessed")):
        reward -= self.MULTI_STAGE_LOAD_LOSS_SCALE * max(0.0, 1.0 - available_load_ratio)
    
    # Terminal signals
    if observation.metadata.get("convergence_failed"):
        reward -= 12.0
    elif done and not reached_time_limit:
        reward -= 12.0
    elif reached_time_limit and available_load_ratio >= 0.5:
        reward += self.MULTI_STAGE_WIN_SCALE * (available_load_ratio ** 2)
```

### Reward Scaling Constants

From `grid2op_env/server/grid_environment.py:63-66`:

```python
MULTI_STAGE_GENERATION_COST_SCALE: float = 0.02   # Generation cost penalty
MULTI_STAGE_LOAD_LOSS_SCALE: float = 5.0          # Load loss penalty at boundaries
MULTI_STAGE_CONVERGENCE_SCALE: float = 0.5        # Convergence reward
MULTI_STAGE_WIN_SCALE: float = 8.0                # Win reward at step 30
```

### Reward Components Breakdown

| Component | Formula | When Applied | Purpose |
|-----------|---------|--------------|---------|
| **Generation cost** | -0.02 × (total_gen / initial_load) | Every step | Penalize wasteful redispatch |
| **Convergence reward** | +0.5 × available_island_ratio | Every step | Reward stable islands |
| **Load loss penalty** | -5.0 × (1 - available_load_ratio) | Only at steps 10, 20 | Penalize load shed at stage boundaries |
| **Terminal - win** | +8.0 × (available_load_ratio)² | Step 30 only if ≥50% load | Reward preserving enough load |
| **Terminal - blackout** | -12.0 | Early termination | Major failure |

This ensures no single component dominates the reward signal.

---

## 7. Episode Logging

From `grid2op_env/server/grid_environment.py:652-699`:

```python
def _build_log_entry(
    self,
    observation: GridObservation,
    raw_reward: float,
    action: GridAction,
    invalid_action: bool,
    invalid_reason: str | None,
    convergence_failed: bool,
    topology_change_count: int,
    auto_trip_detected: bool,
    reconnect_successful: bool,
) -> EpisodeStepLog:
    # ... other fields ...
    return EpisodeStepLog(
        step=self._state.step_count,
        task_id=self._task_id,
        reward=float(observation.reward or 0.0),
        raw_reward=raw_reward,
        done=bool(observation.done),
        max_rho=float(max(observation.rho)) if observation.rho else 0.0,
        stage_index=int(observation.metadata.get("stage_index", 1)),
        steps_to_stage_boundary=int(observation.metadata.get("steps_to_stage_boundary", 0))),
        available_load_ratio=float(observation.metadata.get("available_load_ratio", 1.0)),
        available_island_ratio=float(observation.metadata.get("available_island_ratio", 1.0)),
        stage_boundary_assessed=bool(observation.metadata.get("stage_boundary_assessed", False)),
        majority_islands_available=bool(observation.metadata.get("majority_islands_available", False)),
        # ... other fields ...
    )
```

### EpisodeStepLog Fields for Task 4

From `grid2op_env/models.py:54-59`:

```python
stage_index: int = 1
steps_to_stage_boundary: int = 0
available_load_ratio: float = 1.0
available_island_ratio: float = 1.0
stage_boundary_assessed: bool = False
majority_islands_available: bool = False
```

---

## 8. Grading

From `grid2op_env/server/graders.py:124-174`:

```python
def grade_multi_stage_cascade(
    episode_log: list[EpisodeStepLog], max_steps: int = 30
) -> float:
    if not episode_log:
        return 0.0

    # Component A: Stage completion (30%)
    reached_stage_2 = any(entry.step >= 10 for entry in episode_log)
    reached_stage_3 = any(entry.step >= 20 for entry in episode_log)
    ended_without_blackout = bool(episode_log[-1].step >= max_steps and not episode_log[-1].convergence_failed)
    stage_completion = (
        float(reached_stage_2) + float(reached_stage_3) + float(ended_without_blackout)
    ) / 3.0

    # Component B: Load preservation (40%)
    final_entry = episode_log[-1]
    load_preservation = max(0.0, min(1.0, float(final_entry.available_load_ratio)))

    # Component C: Island quality (20%)
    boundary_logs = [
        entry for entry in episode_log if entry.stage_boundary_assessed
    ]
    island_quality = (
        sum(1 for entry in boundary_logs if entry.majority_islands_available) / 2.0
        if boundary_logs
        else 0.0
    )
    island_quality = max(0.0, min(1.0, island_quality))

    # Component D: Speed bonus (10%)
    stage_ranges = [(1, 10), (11, 20), (21, 30)]
    stage_speed_scores: list[float] = []
    for start_step, end_step in stage_ranges:
        stable_step = next(
            (
                entry.step
                for entry in episode_log
                if start_step <= entry.step <= end_step and entry.all_lines_below_100
            ),
            None,
        )
        if stable_step is None:
            stage_speed_scores.append(0.0)
            continue
        steps_to_stable = stable_step - start_step
        stage_speed_scores.append(max(0.0, (10.0 - steps_to_stable) / 10.0))
    speed_score = sum(stage_speed_scores) / len(stage_speed_scores)

    score = (
        0.30 * stage_completion
        + 0.40 * load_preservation
        + 0.20 * island_quality
        + 0.10 * speed_score
    )
    return round(min(1.0, max(0.0, score)), 6)
```

### Grading Components

| Component | Weight | Formula | What it measures |
|-----------|--------|---------|------------------|
| **Stage completion** | 30% | (reached_stage2 + reached_stage3 + completed) / 3 | Did the agent survive each stage? |
| **Load preservation** | 40% | available_load_ratio at end | How much load is still energized? |
| **Island quality** | 20% | majority_available_stages / 2 | At stage boundaries, were most islands viable? |
| **Speed bonus** | 10% | avg((10 - steps_to_stable) / 10) | How fast did stability return in each stage? |

### Grading Examples

| Scenario | Completion | Load Pres. | Island Qty | Speed | Score |
|----------|-------------|-------------|------------|-------|-------|
| Full survival, 80% load, good islands | 1.0 | 0.8 | 1.0 | 0.8 | **0.90** |
| Stage 3 failed, 60% load | 0.67 | 0.6 | 0.5 | 0.5 | 0.58 |
| Blackout at step 25 | 0.67 | 0.0 | 0.5 | 0.3 | 0.30 |
| Survived but only 30% load | 1.0 | 0.3 | 0.0 | 0.5 | 0.47 |

---

## 9. Agent Prompt Engineering

From `grid2op_env/inference.py:606-648`:

The prompt includes special cross-stage planning instructions:

```python
if task_id == "multi_stage_cascade":
    overflow_urgent = [
        {
            "line_id": idx,
            "rho": round(float(observation.rho[idx]), 4),
            "timestep_overflow": int(value),
        }
        for idx, value in enumerate(observation.timestep_overflow)
        if int(value) > 0
    ]
    overflow_urgent.sort(key=lambda item: (item["timestep_overflow"], item["rho"]), reverse=True)
    lines.insert(
        6,
        "TASK RULE: In multi_stage_cascade, assume the collapse will continue across three stages. "
        "Do not optimize only for this step; position the grid so later stages keep more load available.",
    )
    lines.insert(
        7,
        f"STAGE_CONTEXT=stage_{stage_index}_of_3; steps_to_stage_boundary={steps_to_stage_boundary}; "
        f"available_load_ratio={available_load_ratio:.4f}; available_island_ratio={available_island_ratio:.4f}",
    )
    lines.insert(
        8,
        f"BOUNDARY_STATUS=assessed:{str(stage_boundary_assessed).lower()}; "
        f"majority_islands_available:{str(majority_islands_available).lower()}",
    )
    lines.insert(
        9,
        "MSCF RULE: Prefer actions that preserve transferable generation and keep islands "
        "self-sustaining at the next boundary. Avoid short-term fixes that strand load "
        "in islands with insufficient generation.",
    )
    lines.insert(
        10,
        "TASK RULE: With multiple overloaded lines, topology cuts risk bus isolation. "
        "Prioritize redispatch over disconnect_line unless the line is explicitly "
        "safe_to_disconnect and the action preserves connectivity.",
    )
    lines.insert(
        11,
        f"CONTROLLED_ISLANDING_CANDIDATES={json.dumps(topology_guidance, separators=(',', ':'))}",
    )
    lines.insert(
        12,
        f"REDISPATCH_CANDIDATES={json.dumps(redispatch_guidance, separators=(',', ':'))}",
    )
    lines.insert(
        13,
        f"OVERFLOW_COUNTDOWNS={json.dumps(overflow_urgent[:8], separators=(',', ':'))}",
    )
```

### Key Prompt Variables for Task 4

```
STAGE_CONTEXT=stage_2_of_3; steps_to_stage_boundary=5; available_load_ratio=0.8500; available_island_ratio=0.6667
BOUNDARY_STATUS=assessed:false; majority_islands_available:true
MSCF RULE: Prefer actions that preserve transferable generation and keep islands self-sustaining at the next boundary.
```

### What the LLM Should Do

1. **Understand stages**: Recognize that cascade is guaranteed - can't stop it, only manage it
2. **Triage**: Identify which islands can survive and focus on those
3. **Cross-stage planning**: Think about how actions in Stage 1 affect Stage 2 and 3
4. **Preserve load**: The key metric is `available_load_ratio`, not `max_rho`
5. **Survive all stages**: Reach step 30 with at least 50% load to get win reward

### Key Metrics to Monitor

- `available_load_ratio`: Fraction of initial load still energized (0.0 to 1.0)
- `available_island_ratio`: Fraction of islands that can self-sustain (0.0 to 1.0)
- `stage_boundary_assessed`: True at step 10, 20 (stage boundary)
- `majority_islands_available`: True if >50% of islands can survive

---

## 10. Unique Distinction from All Other Tasks

| Dimension | Task 1 | Task 2 | Task 3 | Task 4 |
|-----------|--------|--------|--------|--------|
| **Core** | Fix max_rho | Survive N-1 | Prevent cascade | Manage guaranteed cascade |
| **Lines at reset** | 0 | 1 | 1-2 | **3** |
| **Load increase** | 0% | 0% | +5-15% | **+20%** |
| **Max steps** | 10 | 20 | 30 | 30 |
| **Stages** | None | None | None | **3 explicit** |
| **Key metric** | max_rho | survival | timestep_overflow | **available_load_ratio** |
| **Can succeed by** | Reducing stress | Staying alive | No auto-trips | Preserving load |
| **Reward focus** | Speed + margin | Survival | No trips | **Load preservation** |

---

## 11. Files Reference

| File | Line Numbers | Purpose |
|------|------------|---------|
| `grid2op_env/server/tasks.py` | 53-62 | Task definition |
| `grid2op_env/server/tasks.py` | 70-74 | Line triplets: (2,4,14), (2,4,15), (4,14,16) |
| `grid2op_env/server/tasks.py` | 334-337 | Profile function: returns 1.20 (20% load) |
| `grid2op_env/server/tasks.py` | 565-620 | Reset function with survival probe |
| `grid2op_env/server/tasks.py` | 623-633 | Survival probe function |
| `grid2op_env/server/grid_environment.py` | 63-66 | Reward constants (0.02, 5.0, 0.5, 8.0) |
| `grid2op_env/server/grid_environment.py` | 630-647 | Reward function |
| `grid2op_env/server/grid_environment.py` | 750-765 | Stage metadata computation |
| `grid2op_env/server/grid_environment.py` | 767-814 | Island availability assessment |
| `grid2op_env/server/grid_environment.py` | 816-849 | Connected components detection |
| `grid2op_env/server/graders.py` | 124-174 | Four-component grading |
| `grid2op_env/inference.py` | 606-648 | LLM prompt with stage context |
| `grid2op_env/inference.py` | 1003-1030 | Candidate filtering (removes unsafe disconnects) |
| `grid2op_env/models.py` | 54-59 | EpisodeStepLog fields for Task 4 |

---

## 12. Research Foundation

The task design is based on key insights:

1. **Stages are interdependent**: An action that solves Stage 1 may make Stage 2 unrecoverable
2. **Guaranteed progression**: Unlike Task 3 where prevention is possible, Task 4 cascade WILL progress
3. **Island assessment**: Deterministic evaluation at stage boundaries determines load loss
4. **Multi-horizon planning**: Agent must think across stage boundaries, not just current step
5. **Load preservation**: The key metric is how much load survives, not max_rho

The task tests **strategic planning under permanent consequences** - each stage boundary permanently reduces available load, and decisions have lasting effects.