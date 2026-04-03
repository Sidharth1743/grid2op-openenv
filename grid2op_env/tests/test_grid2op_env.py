from __future__ import annotations

from grid2op_env.models import (
    EpisodeStepLog,
    GridAction,
    GridObservation,
    GridState,
    RedispatchGeneratorContext,
)
from grid2op_env.inference import (
    build_final_selection_prompt,
    build_proposal_prompt,
    filter_candidate_proposals,
    parse_candidate_proposals,
    select_final_action,
    SimulationOutcome,
    constrain_redispatch_delta,
)
from grid2op_env.graph_analysis import analyze_grid_topology
from grid2op_env.server.tasks import _cascade_profile, benchmark_tiers_for_task, replay_scenario_raw
from grid2op_env.server.graders import (
    grade_cascade_prevent,
    grade_multi_stage_cascade,
    grade_n_minus_1,
    grade_single_fault,
)
from grid2op_env.server.grid_environment import GridEnvironment
from grid2op_env.server.tasks import TASKS, inject_scenario_raw


def test_task_resets_and_logs():
    env = GridEnvironment()
    try:
        for task_id, spec in TASKS.items():
            obs = env.reset(task_id=task_id)
            assert obs.rho
            assert isinstance(obs.sensitivity_guidance, list)
            assert env.state.task_id == task_id
            assert env.state.max_steps == spec.max_steps

            next_obs = env.step(GridAction(do_nothing=True))
            assert isinstance(next_obs.reward, float)
            assert env.state.episode_log
            assert env.state.episode_log[-1].task_id == task_id
    finally:
        env.close()


def test_state_model_validation():
    state = GridState(
        episode_id="ep",
        step_count=1,
        task_id="single_fault",
        max_steps=10,
        n_line=20,
        n_gen=6,
    )
    assert state.n_line == 20


def test_graders_are_deterministic():
    single_fault_log = [
        EpisodeStepLog(
            step=1,
            task_id="single_fault",
            reward=1.0,
            raw_reward=0.0,
            done=False,
            max_rho=0.75,
            all_lines_below_80=True,
            all_lines_below_90=True,
            all_lines_below_100=True,
        )
    ]
    assert grade_single_fault(single_fault_log) == 1.0

    n_minus_1_log = [
        EpisodeStepLog(
            step=idx,
            task_id="n_minus_1",
            reward=0.0,
            raw_reward=0.0,
            done=False,
            max_rho=0.75 if idx >= 6 else 0.85,
            all_lines_below_100=True,
            disconnected_lines=[],
        )
        for idx in range(1, 21)
    ]
    assert grade_n_minus_1(n_minus_1_log) == 1.0

    cascade_log = [
        EpisodeStepLog(
            step=idx,
            task_id="cascade_prevent",
            reward=0.0,
            raw_reward=0.0,
            done=(idx == 30),
            max_rho=0.99,
            all_lines_below_100=True,
        )
        for idx in range(1, 31)
    ]
    assert grade_cascade_prevent(cascade_log) == 1.0

    multi_stage_log = [
        EpisodeStepLog(
            step=idx,
            task_id="multi_stage_cascade",
            reward=0.0,
            raw_reward=0.0,
            done=(idx == 30),
            max_rho=0.95 if idx in (10, 20, 30) else 0.85,
            all_lines_below_100=True,
            stage_boundary_assessed=idx in (10, 20),
            majority_islands_available=True,
            available_load_ratio=0.9,
        )
        for idx in range(1, 31)
    ]
    assert grade_multi_stage_cascade(multi_stage_log) == 0.96


def test_candidate_proposals_are_parsed_and_deduped():
    payload = """
    {
      "candidates": [
        {"action_type": "reconnect_line", "line_id": 0, "gen_id": null, "delta_mw": null, "reason": "restore capacity"},
        {"action_type": "redispatch", "line_id": null, "gen_id": 2, "delta_mw": 10.0, "reason": "shift flow"},
        {"action_type": "redispatch", "line_id": null, "gen_id": 2, "delta_mw": 10.0, "reason": "duplicate"}
      ]
    }
    """
    candidates, trace = parse_candidate_proposals(
        payload,
        n_line=20,
        n_gen=6,
        redispatchable_generators=[0, 1, 5],
        redispatch_generators=[
            RedispatchGeneratorContext(
                gen_id=0,
                p_mw=81.4,
                max_ramp_up=5.0,
                max_ramp_down=5.0,
                allowed_delta_min=-5.0,
                allowed_delta_max=5.0,
                allowed_deltas=[-5.0, -2.5, 2.5, 5.0],
            )
        ],
    )
    assert len(candidates) == 3
    assert trace["parsed_candidate_count"] == 3
    assert candidates[0][0].line_set == {0: 1}
    assert candidates[1][0].do_nothing is True
    assert candidates[2][0].redispatch == {0: -5.0}


def test_final_selection_prefers_simulated_candidate():
    simulations = [
        SimulationOutcome(
            candidate_index=1,
            action=GridAction(line_set={0: 1}),
            trace={"decision": "reconnect_line", "reason": "restore"},
            done=False,
            simulated_reward=1.0,
            max_rho=0.82,
            overloaded_line_ids=[],
            disconnected_lines=[],
            convergence_failed=False,
            exceptions=[],
            raw_result={},
        ),
        SimulationOutcome(
            candidate_index=2,
            action=GridAction(do_nothing=True),
            trace={"decision": "do_nothing", "reason": "fallback"},
            done=False,
            simulated_reward=0.5,
            max_rho=0.91,
            overloaded_line_ids=[],
            disconnected_lines=[],
            convergence_failed=False,
            exceptions=[],
            raw_result={},
        ),
    ]
    payload = """
    {
      "selected_candidate": 1,
      "action": {"action_type": "reconnect_line", "line_id": 0, "gen_id": null, "delta_mw": null, "reason": "choose the restored path"},
      "reason": "simulation 1 is the safest"
    }
    """
    action, trace = select_final_action(payload, simulations, n_line=20, n_gen=6)
    assert action.line_set == {0: 1}
    assert trace["selected_candidate"] == 1


def test_graph_analysis_returns_expected_keys():
    import grid2op

    env = grid2op.make("l2rpn_case14_sandbox")
    try:
        raw_obs, _ = inject_scenario_raw(env, "single_fault", seed=0)
        summary = analyze_grid_topology(
            raw_obs,
            line_or_to_subid=env.line_or_to_subid.tolist(),
            line_ex_to_subid=env.line_ex_to_subid.tolist(),
            n_sub=int(env.n_sub),
        )
        assert "bridge_lines" in summary
        assert "safe_to_disconnect" in summary
        assert "high_centrality_buses" in summary
        assert "congestion_corridor" in summary
        assert isinstance(summary["stressed_lines"], list)
        assert summary["flow_clusters"]["export_buses"]
        assert summary["flow_clusters"]["import_buses"]
        assert "unknown" not in summary["congestion_corridor"]
    finally:
        env.close()


def test_curriculum_metadata_progression():
    import grid2op

    env = grid2op.make("l2rpn_case14_sandbox")
    try:
        _, single_mild = inject_scenario_raw(env, "single_fault", seed=0, difficulty_level=1)
        _, single_severe = inject_scenario_raw(env, "single_fault", seed=0, difficulty_level=8)
        _, cascade_easy = inject_scenario_raw(env, "cascade_prevent", seed=0, difficulty_level=2)
        _, cascade_hard = inject_scenario_raw(env, "cascade_prevent", seed=0, difficulty_level=10)
        _, multi_stage = inject_scenario_raw(env, "multi_stage_cascade", seed=0, difficulty_level=1)

        assert single_mild["curriculum_stage"] == "mild"
        assert single_severe["curriculum_stage"] == "severe"
        assert single_mild["target_rho_range"] == [0.9, 0.94]
        assert single_severe["target_rho_range"] == [0.96, 0.99]

        assert cascade_easy["curriculum_stage"] == "one_line_5pct"
        assert cascade_easy["load_scale"] == 1.05
        assert len(cascade_easy["faulted_lines"]) == 1

        assert cascade_hard["curriculum_stage"] == "two_lines_15pct"
        assert cascade_hard["load_scale"] == 1.15
        assert len(cascade_hard["faulted_lines"]) == 2
        assert multi_stage["curriculum_stage"] == "expert_three_stage"
        assert multi_stage["load_scale"] == 1.2
        assert len(multi_stage["faulted_lines"]) == 3
        assert multi_stage["overflow_window"] == 2
        assert multi_stage["do_nothing_probe_steps"] == 5
    finally:
        env.close()


def test_replay_scenario_raw_matches_single_fault_metadata():
    import grid2op

    env = grid2op.make("l2rpn_case14_sandbox")
    mirror = grid2op.make("l2rpn_case14_sandbox")
    try:
        raw_obs, metadata = inject_scenario_raw(env, "single_fault", seed=3, difficulty_level=4)
        replayed = replay_scenario_raw(
            mirror,
            task_id="single_fault",
            seed=3,
            scenario_metadata=metadata,
        )
        assert max(raw_obs.rho.tolist()) == max(replayed.rho.tolist())
        assert raw_obs.line_status.tolist() == replayed.line_status.tolist()
    finally:
        env.close()
        mirror.close()


def test_cascade_profile_is_deterministic_for_same_seed_and_difficulty():
    first = _cascade_profile(difficulty_level=10, seed=7)
    second = _cascade_profile(difficulty_level=10, seed=7)
    assert first == second


def test_multi_stage_prompt_includes_stage_context():
    prompt = build_proposal_prompt(
        task_id="multi_stage_cascade",
        observation=GridObservation(
            rho=[1.2, 0.95],
            gen_p=[50.0, 20.0],
            load_p=[30.0, 20.0],
            line_status=[True, False],
            timestep_overflow=[2, 0],
            sensitivity_guidance=[
                {"action_type": "disconnect_line", "target_id": 7, "expected_rho_change": -0.1},
                {"action_type": "redispatch", "target_id": 1, "expected_rho_change": -0.05},
            ],
            metadata={
                "stage_index": 2,
                "steps_to_stage_boundary": 4,
                "available_load_ratio": 0.72,
                "available_island_ratio": 0.5,
                "stage_boundary_assessed": True,
                "majority_islands_available": False,
            },
        ),
        graph_intelligence={},
        redispatchable_generators=[0],
        redispatch_generators=[],
        step_count=11,
        max_steps=30,
        include_task_description=True,
    )
    assert "stage_2_of_3" in prompt
    assert "available_load_ratio=0.7200" in prompt
    assert "majority_islands_available:false" in prompt
    assert "CONTROLLED_ISLANDING_CANDIDATES=" in prompt
    assert "REDISPATCH_CANDIDATES=" in prompt


def test_multi_stage_final_prompt_prefers_redispatch_over_do_nothing_when_better():
    prompt = build_final_selection_prompt(
        task_id="multi_stage_cascade",
        observation=GridObservation(
            rho=[0.95, 0.85],
            gen_p=[50.0],
            load_p=[20.0],
            line_status=[True, True],
            timestep_overflow=[0, 0],
            sensitivity_guidance=[],
        ),
        step_count=3,
        max_steps=30,
        simulations=[
            SimulationOutcome(
                candidate_index=1,
                action=GridAction(redispatch={0: -5.0}),
                trace={},
                done=False,
                simulated_reward=1.0,
                max_rho=0.84,
                overloaded_line_ids=[],
                disconnected_lines=[],
                convergence_failed=False,
                exceptions=[],
                raw_result={},
            ),
            SimulationOutcome(
                candidate_index=2,
                action=GridAction(do_nothing=True),
                trace={},
                done=False,
                simulated_reward=0.9,
                max_rho=0.85,
                overloaded_line_ids=[],
                disconnected_lines=[],
                convergence_failed=False,
                exceptions=[],
                raw_result={},
            ),
        ],
    )
    assert "prefer_redispatch_indices" in prompt
    assert "choose that redispatch instead of do_nothing" in prompt


def test_handle_convergence_failure_without_last_obs_does_not_raise():
    env = GridEnvironment()
    try:
        env._last_obs = None
        fallback = env._handle_convergence_failure(RuntimeError("boom"))
        assert fallback.done is True
        assert fallback.metadata["convergence_failed"] is True
    finally:
        env.close()


def test_planning_context_and_server_side_simulation_use_active_episode():
    env = GridEnvironment()
    try:
        obs = env.reset(task_id="single_fault", seed=0, difficulty_level=1)
        episode_id = env.state.episode_id
        active = GridEnvironment.get_active_instance(episode_id)
        assert active is env

        planning = env.get_planning_context()
        assert planning.episode_id == episode_id
        assert planning.graph_intelligence
        assert planning.redispatchable_generators == [0, 1, 5]
        assert planning.redispatch_generators
        assert planning.redispatch_generators[0].allowed_deltas
        assert planning.redispatch_generators[0].allowed_delta_max > 0.0

        results = env.simulate_actions([GridAction(do_nothing=True)])
        assert len(results) == 1
        assert results[0].action.do_nothing is True
        assert results[0].max_rho >= 0.0
        assert isinstance(obs.rho, list)
    finally:
        env.close()


def test_single_fault_observation_includes_negative_sensitivity_guidance():
    env = GridEnvironment()
    try:
        obs = env.reset(task_id="single_fault", seed=0, difficulty_level=1)
        assert isinstance(obs.sensitivity_guidance, list)
        assert len(obs.sensitivity_guidance) <= 3
        for item in obs.sensitivity_guidance:
            assert item["action_type"] in {"disconnect_line", "redispatch"}
            assert isinstance(item["target_id"], int)
            assert float(item["expected_rho_change"]) < 0.0
    finally:
        env.close()


def test_proposal_prompt_includes_sensitivity_guidance_rule():
    observation = GridObservation(
        rho=[0.97, 0.75],
        gen_p=[80.0, 70.0],
        load_p=[50.0],
        line_status=[True, True],
        timestep_overflow=[0, 0],
        sensitivity_guidance=[
            {
                "action_type": "disconnect_line",
                "target_id": 1,
                "expected_rho_change": -0.12,
            }
        ],
    )
    prompt = build_proposal_prompt(
        task_id="single_fault",
        observation=observation,
        graph_intelligence={},
        redispatchable_generators=[0],
        redispatch_generators=[],
        step_count=0,
        max_steps=10,
        include_task_description=True,
    )
    assert "CRITICAL PHYSICS RULE" in prompt
    assert '"action_type":"disconnect_line"' in prompt
    assert '"expected_rho_change":-0.12' in prompt


def test_constrain_redispatch_delta_clamps_to_feasible_discrete_choice():
    context = RedispatchGeneratorContext(
        gen_id=0,
        p_mw=81.4,
        max_ramp_up=5.0,
        max_ramp_down=5.0,
        allowed_delta_min=-5.0,
        allowed_delta_max=5.0,
        allowed_deltas=[-5.0, -2.5, 2.5, 5.0],
    )
    assert constrain_redispatch_delta(10.0, context) == 5.0
    assert constrain_redispatch_delta(-9.0, context) == -5.0


def test_benchmark_tiers_are_declared_for_each_task():
    assert benchmark_tiers_for_task("single_fault") == [
        "single_fault_easy",
        "single_fault_moderate",
        "single_fault_severe",
    ]
    assert benchmark_tiers_for_task("cascade_prevent") == [
        "cascade_prevent_easy",
        "cascade_prevent_medium",
        "cascade_prevent_hard",
        "cascade_prevent_extreme",
    ]
    assert benchmark_tiers_for_task("multi_stage_cascade") == [
        "multi_stage_cascade_expert",
    ]


def test_multi_stage_candidate_prefilter_blocks_unsafe_disconnects():
    candidates = [
        (GridAction(line_set={14: -1}), {"decision": "disconnect_line"}),
        (GridAction(redispatch={0: -5.0}), {"decision": "redispatch"}),
        (GridAction(do_nothing=True), {"decision": "do_nothing"}),
    ]
    filtered, trace = filter_candidate_proposals(
        task_id="multi_stage_cascade",
        observation=GridObservation(
            rho=[1.1, 0.9],
            gen_p=[50.0],
            load_p=[20.0],
            line_status=[True, True],
            timestep_overflow=[0, 0],
            sensitivity_guidance=[],
        ),
        graph_intelligence={"safe_to_disconnect": [3, 4]},
        proposal_candidates=candidates,
    )
    assert len(filtered) == 2
    assert filtered[0][0].redispatch == {0: -5.0}
    assert trace["prefilter_rejections"]


def test_final_selection_never_returns_failed_candidate():
    simulations = [
        SimulationOutcome(
            candidate_index=1,
            action=GridAction(line_set={13: -1}),
            trace={"decision": "disconnect_line", "reason": "bad"},
            done=True,
            simulated_reward=-10.0,
            max_rho=0.0,
            overloaded_line_ids=[],
            disconnected_lines=[],
            convergence_failed=True,
            exceptions=["boom"],
            raw_result={},
        ),
        SimulationOutcome(
            candidate_index=2,
            action=GridAction(redispatch={0: -5.0}),
            trace={"decision": "redispatch", "reason": "good"},
            done=False,
            simulated_reward=1.0,
            max_rho=1.02,
            overloaded_line_ids=[13],
            disconnected_lines=[],
            convergence_failed=False,
            exceptions=[],
            raw_result={},
        ),
    ]
    action, trace = select_final_action(
        final_raw_output='{"selected_candidate":1,"reason":"pick first"}',
        simulations=simulations,
        n_line=20,
        n_gen=6,
        task_id="multi_stage_cascade",
    )
    assert action.redispatch == {0: -5.0}
    assert trace["decision"] == "fallback_best_simulation"


def test_benchmark_single_fault_requires_target_match():
    from grid2op.Exceptions import Grid2OpException
    from unittest.mock import patch

    with patch("grid2op_env.server.tasks._reset_single_fault", side_effect=Grid2OpException("missed target")):
        try:
            inject_scenario_raw(
                object(),
                "single_fault",
                seed=0,
                max_attempts=1,
                scenario_mode="benchmark",
                benchmark_tier="single_fault_moderate",
            )
        except Grid2OpException:
            pass
        else:
            raise AssertionError("benchmark single_fault should reject invalid fallback scenarios")
