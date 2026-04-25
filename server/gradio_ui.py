from __future__ import annotations

import inspect
import json
from typing import Any

import gradio as gr
from openenv.core.env_server.serialization import serialize_observation

try:
    from ..models import GridAction
    from .tasks import benchmark_tiers_for_task, task_list
except ImportError:
    from models import GridAction
    from server.tasks import benchmark_tiers_for_task, task_list


TASK_CHOICES = [(task.task_id, task.task_id) for task in task_list()]
SCENARIO_MODE_CHOICES = [("benchmark", "benchmark"), ("curriculum", "curriculum")]

DEFAULT_ACTION_JSON = json.dumps(
    {
        "do_nothing": True,
        "line_set": {},
        "redispatch": {},
    },
    indent=2,
)


def _extract_payload(data: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = data or {}
    observation = payload.get("observation", {})
    metadata = observation.get("metadata", {}) if isinstance(observation, dict) else {}
    return observation if isinstance(observation, dict) else {}, metadata if isinstance(metadata, dict) else {}


def _format_status_markdown(state: dict[str, Any] | None, data: dict[str, Any] | None) -> str:
    observation, metadata = _extract_payload(data)
    rho = observation.get("rho", []) or []
    max_rho = max(rho) if rho else 0.0
    reward = payload_reward(data)
    done = payload_done(data)
    episode_id = (state or {}).get("episode_id", "not started")
    step_count = (state or {}).get("step_count", 0)
    task_id = (state or {}).get("task_id", "unknown")
    max_steps = (state or {}).get("max_steps", "?")
    stage_index = metadata.get("stage_index", "-")
    steps_to_boundary = metadata.get("steps_to_stage_boundary", "-")
    available_load_ratio = metadata.get("available_load_ratio")
    available_island_ratio = metadata.get("available_island_ratio")

    lines = [
        "### Episode Status",
        f"- Episode: `{episode_id}`",
        f"- Task: `{task_id}`",
        f"- Step: `{step_count}` / `{max_steps}`",
        f"- Reward: `{reward:.3f}`",
        f"- Done: `{str(done).lower()}`",
        f"- Max rho: `{max_rho:.3f}`",
        f"- Stage: `{stage_index}`",
        f"- Steps to boundary: `{steps_to_boundary}`",
    ]
    if available_load_ratio is not None:
        lines.append(f"- Available load ratio: `{float(available_load_ratio):.3f}`")
    if available_island_ratio is not None:
        lines.append(f"- Available island ratio: `{float(available_island_ratio):.3f}`")
    return "\n".join(lines)


def _hot_lines_dataframe(data: dict[str, Any] | None) -> list[list[Any]]:
    observation, _ = _extract_payload(data)
    rho = observation.get("rho", []) or []
    overflow = observation.get("timestep_overflow", []) or []
    line_status = observation.get("line_status", []) or []
    rows: list[list[Any]] = []
    ranked = sorted(
        range(len(rho)),
        key=lambda idx: float(rho[idx]),
        reverse=True,
    )[:8]
    for idx in ranked:
        rows.append(
            [
                idx,
                round(float(rho[idx]), 4),
                int(overflow[idx]) if idx < len(overflow) else 0,
                "connected" if idx < len(line_status) and bool(line_status[idx]) else "disconnected",
            ]
        )
    return rows


def _guidance_dataframe(data: dict[str, Any] | None) -> list[list[Any]]:
    observation, _ = _extract_payload(data)
    guidance = observation.get("sensitivity_guidance", []) or []
    rows: list[list[Any]] = []
    for item in guidance[:8]:
        rows.append(
            [
                item.get("action_type", ""),
                item.get("target_id", ""),
                item.get("delta_mw", ""),
                item.get("expected_rho_change", ""),
            ]
        )
    return rows


def _action_examples(task_id: str) -> str:
    examples = {
        "single_fault": {
            "do_nothing": False,
            "line_set": {},
            "redispatch": {"5": -15.0},
        },
        "n_minus_1": {
            "do_nothing": False,
            "line_set": {"0": 1},
            "redispatch": {},
        },
        "cascade_prevent": {
            "do_nothing": False,
            "line_set": {},
            "redispatch": {"1": -10.0, "5": 7.5},
        },
        "multi_stage_cascade": {
            "do_nothing": False,
            "line_set": {},
            "redispatch": {"0": -5.0, "1": 5.0},
        },
    }
    return json.dumps(examples.get(task_id, json.loads(DEFAULT_ACTION_JSON)), indent=2)


def _benchmark_tier_choices(task_id: str, scenario_mode: str) -> list[tuple[str, str]]:
    if scenario_mode != "benchmark":
        return [("default", "")]
    return [("default", "")] + [(tier, tier) for tier in benchmark_tiers_for_task(task_id)]


def _default_benchmark_tier(task_id: str) -> str | None:
    tiers = benchmark_tiers_for_task(task_id)
    return tiers[0] if tiers else None


def payload_reward(data: dict[str, Any] | None) -> float:
    if not data:
        return 0.0
    return float(data.get("reward", 0.0) or 0.0)


def payload_done(data: dict[str, Any] | None) -> bool:
    if not data:
        return False
    return bool(data.get("done", False))


def build_grid2op_gradio_app(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    del action_fields, is_chat_env, quick_start_md

    async def reset_with_kwargs(reset_payload: dict[str, Any]) -> dict[str, Any]:
        reset_signature = inspect.signature(web_manager.env.reset)
        accepted_payload = {
            key: value
            for key, value in reset_payload.items()
            if key in reset_signature.parameters
        }
        observation = await web_manager._run_sync_in_thread_pool(
            web_manager.env.reset,
            **accepted_payload,
        )
        state = web_manager.env.state
        serialized = serialize_observation(observation)
        web_manager.episode_state.episode_id = state.episode_id
        web_manager.episode_state.step_count = 0
        web_manager.episode_state.current_observation = serialized["observation"]
        web_manager.episode_state.action_logs = []
        web_manager.episode_state.is_reset = True
        await web_manager._send_state_update()
        return serialized

    async def reset_selected_task(
        task_id: str,
        scenario_mode: str,
        benchmark_tier: str | None,
        seed_value: float | None,
    ):
        reset_payload: dict[str, Any] = {
            "task_id": task_id,
            "scenario_mode": scenario_mode,
        }
        if scenario_mode == "benchmark":
            resolved_benchmark_tier = benchmark_tier or _default_benchmark_tier(task_id)
            if resolved_benchmark_tier:
                reset_payload["benchmark_tier"] = resolved_benchmark_tier
        if seed_value is not None:
            reset_payload["seed"] = int(seed_value)
        data = await reset_with_kwargs(reset_payload)
        state = web_manager.get_state()
        return (
            _format_status_markdown(state, data),
            _hot_lines_dataframe(data),
            _guidance_dataframe(data),
            json.dumps(data, indent=2),
            _action_examples(task_id),
        )

    async def step_with_action_json(action_json: str):
        try:
            action_data = json.loads(action_json)
        except json.JSONDecodeError as exc:
            return (
                f"### Episode Status\n- Error: invalid action JSON\n- Detail: `{exc}`",
                gr.update(),
                gr.update(),
                "",
                action_json,
            )

        normalized = GridAction.model_validate(action_data).model_dump(exclude={"metadata"})
        data = await web_manager.step_environment(normalized)
        state = web_manager.get_state()
        return (
            _format_status_markdown(state, data),
            _hot_lines_dataframe(data),
            _guidance_dataframe(data),
            json.dumps(data, indent=2),
            json.dumps(normalized, indent=2),
        )

    def refresh_state():
        state = web_manager.get_state()
        observation = state.get("current_observation", {})
        data = {
            "observation": observation,
            "reward": state.get("last_reward", 0.0),
            "done": state.get("done", False),
        }
        return (
            _format_status_markdown(state, data),
            _hot_lines_dataframe(data),
            _guidance_dataframe(data),
            json.dumps(state, indent=2),
        )

    async def do_nothing_step():
        data = await web_manager.step_environment({"do_nothing": True, "line_set": {}, "redispatch": {}})
        state = web_manager.get_state()
        return (
            _format_status_markdown(state, data),
            _hot_lines_dataframe(data),
            _guidance_dataframe(data),
            json.dumps(data, indent=2),
            DEFAULT_ACTION_JSON,
        )

    def update_benchmark_dropdown(task_id: str, scenario_mode: str):
        choices = _benchmark_tier_choices(task_id, scenario_mode)
        interactive = scenario_mode == "benchmark"
        value = "" if scenario_mode != "benchmark" else (_default_benchmark_tier(task_id) or "")
        return gr.update(choices=choices, value=value, interactive=interactive)

    with gr.Blocks(title=f"{title} Custom UI") as demo:
        gr.Markdown(
            f"""
            # Grid Operations Console

            This tab is a **user-driven control panel** for `{getattr(metadata, "name", title)}`.
            Start a task, inspect the live grid, and send `do_nothing`, redispatch, or line-status actions
            without leaving the browser.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                task_id = gr.Dropdown(
                    choices=TASK_CHOICES,
                    value="single_fault",
                    label="Task",
                )
            with gr.Column(scale=3):
                scenario_mode = gr.Dropdown(
                    choices=SCENARIO_MODE_CHOICES,
                    value="benchmark",
                    label="Scenario Mode",
                )
            with gr.Column(scale=3):
                benchmark_tier = gr.Dropdown(
                    choices=_benchmark_tier_choices("single_fault", "benchmark"),
                    value=_default_benchmark_tier("single_fault"),
                    label="Benchmark Tier",
                    allow_custom_value=False,
                    interactive=True,
                )
            with gr.Column(scale=2):
                seed_text = gr.Number(
                    label="Seed",
                    precision=0,
                    value=None,
                )

        with gr.Row():
            reset_btn = gr.Button("Start Episode", variant="primary")
            refresh_btn = gr.Button("Refresh State")
            do_nothing_btn = gr.Button("Do Nothing Step")

        with gr.Row():
            status_md = gr.Markdown("### Episode Status\n- No active episode yet.")
            raw_json = gr.Code(
                label="Latest JSON",
                language="json",
                interactive=False,
                value="",
            )

        with gr.Row():
            hot_lines = gr.Dataframe(
                headers=["line_id", "rho", "overflow", "status"],
                datatype=["number", "number", "number", "str"],
                label="Most Stressed Lines",
                interactive=False,
                row_count=8,
                col_count=(4, "fixed"),
            )
            guidance = gr.Dataframe(
                headers=["action_type", "target_id", "delta_mw", "expected_rho_change"],
                datatype=["str", "number", "str", "str"],
                label="Sensitivity Guidance",
                interactive=False,
                row_count=8,
                col_count=(4, "fixed"),
            )

        with gr.Row():
            with gr.Column(scale=5):
                action_json = gr.Code(
                    label="Action JSON",
                    language="json",
                    interactive=True,
                    value=DEFAULT_ACTION_JSON,
                )
            with gr.Column(scale=3):
                gr.Markdown(
                    """
                    ### Action format

                    Use the same schema as the API:
                    - `do_nothing: true`
                    - `line_set: {"0": 1}` to reconnect line 0
                    - `line_set: {"4": -1}` to disconnect line 4
                    - `redispatch: {"5": -15.0}` for MW adjustments
                    """
                )
                step_btn = gr.Button("Apply Action", variant="primary")

        reset_btn.click(
            fn=reset_selected_task,
            inputs=[task_id, scenario_mode, benchmark_tier, seed_text],
            outputs=[status_md, hot_lines, guidance, raw_json, action_json],
        )
        step_btn.click(
            fn=step_with_action_json,
            inputs=[action_json],
            outputs=[status_md, hot_lines, guidance, raw_json, action_json],
        )
        do_nothing_btn.click(
            fn=do_nothing_step,
            outputs=[status_md, hot_lines, guidance, raw_json, action_json],
        )
        refresh_btn.click(
            fn=refresh_state,
            outputs=[status_md, hot_lines, guidance, raw_json],
        )
        task_id.change(
            fn=_action_examples,
            inputs=[task_id],
            outputs=[action_json],
        )
        task_id.change(
            fn=update_benchmark_dropdown,
            inputs=[task_id, scenario_mode],
            outputs=[benchmark_tier],
        )
        scenario_mode.change(
            fn=update_benchmark_dropdown,
            inputs=[task_id, scenario_mode],
            outputs=[benchmark_tier],
        )

    return demo
