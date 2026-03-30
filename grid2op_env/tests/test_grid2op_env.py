from __future__ import annotations

from grid2op_env.models import EpisodeStepLog, GridAction, GridState
from grid2op_env.inference import (
    parse_candidate_proposals,
    select_final_action,
    SimulationOutcome,
)
from grid2op_env.graph_analysis import analyze_grid_topology
from grid2op_env.server.graders import (
    grade_cascade_prevent,
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
            max_rho=0.85,
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
            max_rho=0.95,
            all_lines_below_100=True,
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
    candidates, trace = parse_candidate_proposals(payload, n_line=20, n_gen=6)
    assert len(candidates) == 3
    assert trace["parsed_candidate_count"] == 3
    assert candidates[0][0].line_set == {0: 1}
    assert candidates[1][0].redispatch == {2: 10.0}
    assert candidates[2][0].do_nothing or candidates[2][0].redispatch == {0: 10.0}


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
