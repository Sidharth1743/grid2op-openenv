Let me read the actual existing OpenEnv environments to understand the real patterns before explaining.Good. Now I have enough. Let me give you the full conceptual architecture — no code, just deep explanation of every layer.

---

## What we're building — the big picture

Two separate things that work together:

**1. The Server** — lives in Docker, deployed to HF Spaces. Contains Grid2Op + all the environment logic. Agents never touch this directly.

**2. The Client** — a Python package that agents import. Talks to the server over WebSocket. Exposes `reset()`, `step()`, `state()`.

The agent only ever talks to the client. The client talks to the server. The server talks to Grid2Op. Grid2Op does the physics.

---

## Layer by layer

### Layer 1 — Grid2Op (the physics engine, innermost layer)

Grid2Op is a complete power grid simulator maintained by RTE France. You never modify it. You just use it.

When it starts up, it loads the `l2rpn_case14_sandbox` — which is a real IEEE 14-bus network topology bundled as a dataset. This includes the exact electrical parameters: line resistances, bus connections, generator capacities, load positions.

Alongside the topology, it loads **chronics** — these are pre-recorded CSV files of how load and generation evolved hour-by-hour over several weeks of real grid operation. Think of each chronic as one "scenario tape." When you call `reset()`, Grid2Op picks one of these tapes and starts playing it from the beginning. Each `step()` advances the tape by one timestep (typically 5 minutes of real time).

What Grid2Op gives you after each step:
- `rho` — how loaded each line is as a fraction of its thermal limit. 1.0 = exactly at limit, >1.0 = overloaded
- `gen_p` — how much power each generator is producing
- `load_p` — how much power each load is consuming
- `line_status` — which lines are connected vs disconnected
- `timestep_overflow` — how many consecutive steps each line has been in overflow
- A done flag — True if Grid2Op detects a game-ending condition (island formation, voltage collapse, sustained overload)

What Grid2Op accepts as actions:
- Topology changes: connect/disconnect specific lines, or reconfigure which bus a line connects to within a substation
- Redispatch: tell a generator to produce more or less MW (within its ramp rate limits)
- Do nothing: explicitly pass without acting

---

### Layer 2 — The OpenEnv Adapter (translation layer)

Grid2Op's internal objects are not JSON-serializable and not typed in the Pydantic sense. The adapter's entire job is translation — nothing more.

**Observation translation**: Grid2Op gives you numpy arrays. The adapter converts those arrays into a `GridObservation` dataclass with named, typed fields. The line loading array becomes `rho: list[float]`. The boolean line status array becomes `line_status: list[bool]`. This is what the agent actually sees.

**Action translation**: The agent sends a `GridAction` dataclass — a clean JSON-serializable object with fields like `line_set` (a dict mapping line IDs to connect/disconnect) and `redispatch` (a dict mapping generator IDs to delta MW). The adapter converts this into Grid2Op's internal action format before passing it to the simulator.

**Reward shaping**: Grid2Op returns a raw reward signal, but it's not well-shaped for learning. The adapter replaces this with a custom shaped reward that gives signal throughout the episode, not just at termination. More on this in the reward section below.

**State tracking**: The adapter maintains episode metadata — episode ID (UUID), step count, which task is running, the episode log (a list of per-step summaries used by graders). This is what `state()` returns.

---

### Layer 3 — Task and Scenario System

Tasks are configurations that define what situation the agent faces at the start of an episode. There are three, in increasing difficulty.

**Task 1: single_fault (Easy)**
The agent resets into a chronic where one line is already approaching its thermal limit (~90–95% loading). No line has been tripped yet, but if the chronic continues progressing, that line will overflow. The agent has 10 steps to bring all lines below 80% loading by switching topology or redispatching generation. The scenario is mild enough that a single topology switch (disconnecting the stressed line and letting flow reroute) is sufficient.

**Task 2: n_minus_1 (Medium)**
At reset, one line is immediately tripped (forced disconnected). This is the classical N-1 contingency studied by every grid operator. The remaining 19 lines must now carry the load that the disconnected line was handling. Some lines will be pushed toward their limits. The agent has 20 steps to redispatch generators and adjust topology to bring all lines to safe loading without triggering a cascade. This requires understanding that disconnecting one line stresses different lines depending on the network topology — it's not obvious which action helps.

**Task 3: cascade_prevent (Hard)**
At reset, two lines are tripped simultaneously and load is artificially increased by 15% to simulate a demand spike. This creates a genuinely dangerous scenario — multiple lines approach their limits at once, and if even one additional line trips automatically (Grid2Op disconnects lines that stay above 100% loading for too long), the resulting redistribution can push more lines over the edge, creating a cascade. The agent has 30 steps and must combine topology switches + redispatch to stabilize the grid before the first automatic disconnection triggers. The hard part: some actions that fix one line worsen another, so the agent needs to find a coordinated solution.

**How fault injection works**: Grid2Op's `reset()` accepts an `options` dict that lets you set the initial state before the chronic starts playing. For N-1 and cascade tasks, the adapter passes a line disconnection command in this options dict. The chronic then starts from that pre-faulted state.

---

### Layer 4 — Reward Function

The reward function is applied by the adapter every step, replacing Grid2Op's default reward. It has four components:

**Safe step bonus (+0.1 per step)**: Every step where all lines are below 80% loading, the agent gets a small positive reward. This encourages the agent to maintain healthy margins, not just avoid the immediate crisis.

**Overload penalty (−0.2 per overloaded line per step)**: Any line above 100% loading subtracts from the reward. Proportional to the number of overloaded lines, so two overloaded lines is worse than one. This pushes the agent to prioritize reducing overloads.

**Blackout terminal penalty (−10.0)**: If the episode ends because Grid2Op detects a grid collapse — island formation where part of the network is cut off, or the done flag triggers — a large negative reward is applied. This is the worst outcome.

**Survival bonus (+5.0)**: If the episode ends because the agent reached `max_steps` without a blackout — meaning it kept the grid stable for the whole episode — a large positive reward is applied. This is the best outcome.

**Oscillation penalty (−0.05)**: If the agent takes the exact same action three steps in a row, it gets penalized. This prevents degenerate behavior where the agent loops on one action hoping it works.

The shaped reward gives useful gradient signal at every single step. Without it, an agent would only learn from terminal blackouts, which is sparse and slow.

---

### Layer 5 — Graders

Graders are separate from the reward function. They run after an episode completes and produce a score from 0.0 to 1.0 for the hackathon's automated evaluation. They use the `episode_log` — a list of per-step summaries the adapter has been recording throughout the episode.

**single_fault grader**: Did the agent get all lines below 90% loading? If yes, score starts at 1.0 and decreases slightly for every extra step it took. Optimal score = 1.0 (fixed in step 1). Minimum passing score = 0.5 (fixed by step 10). Failure (never fixed) = 0.0.

**n_minus_1 grader**: What fraction of the maximum allowed steps did the agent survive without triggering a blackout? 20 steps survived out of 20 = 1.0. 10 steps = 0.5. Immediate blackout = 0.0. This is a survival ratio, which produces a continuous score naturally.

**cascade_prevent grader**: Weighted composite. 50% weight on whether the agent survived the full 30 steps. 30% weight on what fraction of steps had all lines below 100% loading (safety ratio). 20% weight on how fast the agent stabilized (steps used vs max). This gives partial credit even if the agent couldn't fully stabilize — maybe it kept 90% of steps safe before eventually failing.

All three graders are deterministic — same episode log always produces same score. This is a hard requirement from the hackathon spec.

---

### Layer 6 — FastAPI Server

The server exposes the environment over two protocols simultaneously:

**WebSocket at `/ws`**: This is the core OpenEnv interface. The client connects here and maintains a persistent session. Each WebSocket connection gets its own isolated `GridEnvironment` instance — so multiple agents can run simultaneously without interfering. The WebSocket handles `reset`, `step`, and `state` as message types. This is what `openenv validate` tests against.

**HTTP at `/tasks`**: Returns a JSON list of all three tasks, their descriptions, difficulties, and the full JSON schema of what a valid `GridAction` looks like. This lets an agent inspect what actions are possible before acting.

**HTTP at `/grader`**: Accepts a POST with a `task_id` and an `episode_log` (collected during an episode), runs the appropriate grader function, and returns the score. The client calls this at the end of each episode during baseline evaluation.

**HTTP at `/baseline`**: Triggers the baseline inference script server-side and returns the scores for all three tasks. This is what the hackathon auto-validator will ping.

---

### Layer 7 — Client Package

The client is a thin Python package that wraps the WebSocket connection. It's what an agent or training loop imports.

It provides `GridEnv` — a class that connects to a running server URL. Calling `reset()` sends a reset message over the WebSocket and returns a `GridObservation`. Calling `step(action)` sends a step message and returns the next `GridObservation`. Calling `state()` returns the current episode metadata.

The client also holds references to `GridAction` and `GridObservation` — the typed dataclasses — so the agent can construct valid actions and interpret observations without knowing anything about Grid2Op.

---

### Layer 8 — Baseline Script

The baseline script runs a local LLM (via Ollama's OpenAI-compatible endpoint) as a naive agent against all three tasks.

The flow: reset → format observation as natural language prompt → send to LLM → parse LLM response as `GridAction` → step → repeat → at end, POST the episode log to `/grader` → record score.

The prompt tells the LLM it's a grid operator, gives it the current state (max line loading, which lines are overloaded, which lines are disconnected), and asks it to respond with a JSON action. If the LLM returns invalid JSON or an invalid action, the agent defaults to `do_nothing=True` — so the baseline always runs to completion without crashing.

The baseline is expected to score low (0.1–0.3 range) — its purpose is reproducibility, not performance. The spec just requires it runs without error and produces scores.

---

## Full data flow for one episode

```
Agent calls reset(task_id="n_minus_1")
  → Client sends WebSocket message to server
  → Server adapter calls grid2op.reset(options={"set_line_status": [(0, -1)]})
  → Grid2Op loads a chronic, starts from line 0 disconnected
  → Adapter converts obs arrays → GridObservation
  → Server sends GridObservation back over WebSocket
  → Client returns GridObservation to agent

Agent inspects obs: line 4 is at 94% loading, line 7 at 88%
Agent calls step(GridAction(line_set={4: -1}, redispatch={2: 15.0}))
  → Client sends step message over WebSocket
  → Server adapter translates: disconnect line 4, redispatch gen 2 +15MW
  → Grid2Op applies action, advances chronic by 1 timestep, runs power flow
  → Adapter computes shaped reward: one overloaded line still → -0.2, others safe → +0.1 × 19 = net positive
  → Adapter logs step summary to episode_log
  → Converts new obs → GridObservation
  → Returns to client

... (more steps) ...

Episode ends (survived 20 steps)
Agent calls POST /grader with episode_log
  → n_minus_1 grader: survived 20/20 steps → score 1.0
```

---

## File structure with purpose of each file

```
grid2op_env/
│
├── models.py              ← GridAction + GridObservation dataclasses. 
│                            Both sides import from here.
│
├── client.py              ← GridEnv class. Wraps WebSocket. 
│                            This is what agents import.
│
├── __init__.py            ← Exports GridAction, GridObservation, GridEnv
│
├── openenv.yaml           ← Environment metadata for openenv validate
│
├── pyproject.toml         ← Package deps + install config
│
└── server/
    │
    ├── grid_environment.py  ← The core adapter. Holds Grid2Op instance.
    │                          Implements reset(), step(), state().
    │                          Translates Grid2Op ↔ OpenEnv types.
    │                          Computes shaped reward.
    │                          Maintains episode_log.
    │
    ├── tasks.py             ← TASKS dict (metadata per task).
    │                          inject_scenario() function that sets
    │                          initial fault conditions per task_id.
    │
    ├── graders.py           ← Three grader functions.
    │                          Each takes episode_log, returns float 0–1.
    │                          Completely deterministic.
    │
    ├── app.py               ← Creates the FastAPI app.
    │                          Mounts WebSocket via OpenEnv's create_app().
    │                          Adds /tasks, /grader, /baseline HTTP routes.
    │
    ├── requirements.txt     ← Server-only deps for Docker
    │
    └── Dockerfile           ← Python 3.11 slim base.
                               Installs deps.
                               Pre-downloads l2rpn_case14_sandbox at build time.
                               Starts uvicorn on port 7860.

baseline/
└── run_baseline.py        ← Ollama-compatible agent loop.
                             Runs all 3 tasks, POSTs to /grader, prints JSON scores.
```

---

These are all valid — and they're exactly the kind of things that cause silent failures during automated judging. Let me address each one properly.

---

## 1. LLM Hallucination & Action Validation

This happens at the **boundary between Layer 8 (baseline) and Layer 2 (adapter)**. Two separate defenses needed, not one.

**Defense A — Client side (baseline script)**: Before even sending the action to the server, validate the action against known grid constants. The `l2rpn_case14_sandbox` has exactly 20 lines and 6 generators. These are fixed, knowable at import time. Any line ID outside 0–19 or gen ID outside 0–5 gets stripped from the action dict before the request is sent. If stripping leaves an empty action, it becomes `do_nothing=True`. This catches hallucinations before they ever hit the network.

**Defense B — Server side (adapter)**: The adapter is the last line of defense. Even if something slips through the client, the adapter re-validates every field in `GridAction` against the actual Grid2Op environment's `n_line` and `n_gen` attributes — which are ground truth at runtime. Invalid indices get silently dropped. If the entire action becomes empty after stripping, the adapter converts it to Grid2Op's native do-nothing action and applies a small fixed penalty (`-0.1`) to the reward. This penalty signals to a training loop that bad actions are costly, without crashing anything.

**Why both**: The client defense handles the LLM case (fast, no server round-trip). The server defense handles any future agent type — RL policies can also produce out-of-bounds actions during early training. Never assume the client is trustworthy.

---

## 2. Power Flow Non-Convergence

This is the most dangerous one because it can kill the entire server process, not just one episode.

The `try/except` in the adapter needs to catch specifically `grid2op.exceptions.Grid2OpException` and the underlying `lightsim2grid` backend errors, not a bare `Exception`. A bare except would also swallow programming errors in your own adapter code and make them invisible.

When a convergence failure is caught:
- Apply the maximum terminal penalty (`-10.0`) to the reward
- Set `done = True`
- Return the **last valid observation** that was stored before the failed step — not a zeroed-out observation, because zeros look like a valid healthy grid to an agent
- Log the failure with the step number and task ID for debugging
- The episode ends cleanly from the client's perspective

The `cascade_prevent` task is the one most likely to hit this. The two-line fault plus demand spike can push the grid into a physically inconsistent state before the agent even acts. Consider adding a convergence check immediately after `inject_scenario()` in `reset()` as well — if the initial faulted state itself doesn't converge, pick a different chronic and retry up to 3 times before raising.

---

## 3. WebSocket State Desync

The `finally` block is correct but incomplete on its own. There are actually three cases to handle:

**Case A — Clean disconnect**: Agent calls reset, runs episode, disconnects normally. The `finally` block runs, `env.close()` is called, Grid2Op releases its internal state. Clean.

**Case B — Mid-episode disconnect**: Agent drops the connection without finishing. The `finally` block still runs because the WebSocket exception propagates up. Same cleanup path. The issue here is if you have a session registry (a dict mapping session IDs to env instances) — you need to explicitly remove the entry from that registry inside the `finally`, not just call `env.close()`. Otherwise the dict holds a reference to the dead env object and memory leaks.

**Case C — Server-side timeout**: Agent connects, calls reset, then goes silent for 10+ minutes (hung process on client side). The `finally` block never runs because no exception was raised — the WebSocket is technically still open. Fix: implement a step timeout. If no `step()` call arrives within N seconds after the last action, the server closes the WebSocket from its side, which triggers the `finally`. A reasonable N is 60 seconds for the baseline, 300 seconds for interactive use.

The session registry pattern also lets you expose a `/sessions` debug endpoint showing how many active environments are running — useful for spotting leaks during testing.

---

## 4. Context Window Bloat

The fix you described is right but there's a more principled framing: the prompt should contain only **actionable information**.

A grid operator looking at 20 line loadings doesn't read them all — they look at the alarm panel which only shows lines above threshold. Your prompt should work the same way.

**What to always include**: lines above 80% loading (with their IDs and exact rho values), disconnected lines (lines where `line_status = False`), current step number, max steps for this task, and the task description on the first step only.

**What to never include**: lines below 80% (irrelevant), raw generator outputs (the agent can't use them without knowing ramp rates), load values (same problem), the full topology vector.

**What to include conditionally**: if a line has been in overflow for 2+ steps (`timestep_overflow > 2`), flag it explicitly — "Line 4 has been overloaded for 3 consecutive steps, automatic disconnection imminent." This is the highest-priority information and an LLM won't infer it from raw numbers.

The result is a prompt that's typically 5–8 lines regardless of grid size, and it actually contains more useful signal than a full dump would.

---

## How these four interact

One more thing worth noting: blindspots 1 and 2 interact with each other. A hallucinated invalid action (blindspot 1) that slips through to Grid2Op can itself cause a non-convergence (blindspot 2). So the validation in the adapter (Defense B) is also a convergence protection — invalid topology actions are one of the primary causes of solver failure. Handling them in order (validate first, then attempt power flow, then catch convergence errors) gives you layered protection rather than depending on any single check.