import numpy as np
import pytest

from quickexp_v3.autocal.budget import (
    BudgetExceeded,
    BudgetModel,
    BudgetTracker,
)
from quickexp_v3.autocal.gp_optimizer import bayesian_optimize
from quickexp_v3.autocal.graph import (
    NODE_REGISTRY,
    TARGETS,
    change_exceeds_invalidation_threshold,
    downstream,
    invalidated_nodes,
    target_nodes,
    validate_graph,
)
from quickexp_v3.autocal.session import AutocalSession, replay_decisions
from quickexp_v3.experiments.base import ExperimentPlan


def test_graph_is_explicit_acyclic_and_targets_are_topological():
    validate_graph()
    assert tuple(NODE_REGISTRY) == (
        "N0",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "N7",
        "N8",
        "N9",
        "N10r",
        "N11",
        "N12",
        "N13",
        "N14",
    )
    for target, node_ids in TARGETS.items():
        assert tuple(spec.node_id for spec in target_nodes(target)) == node_ids
        positions = {node_id: index for index, node_id in enumerate(node_ids)}
        for node_id in node_ids:
            for dependency in NODE_REGISTRY[node_id].dependencies:
                if dependency in positions:
                    assert positions[dependency] < positions[node_id]
    assert {"N9", "N11", "N12", "N13"} <= set(downstream("N8"))
    assert {"N5", "N8", "N9", "N11", "N12", "N13"} <= set(
        invalidated_nodes("N4")
    )
    assert change_exceeds_invalidation_threshold("N8", 0.1, 0.116)
    assert not change_exceeds_invalidation_threshold("N8", 0.1, 0.114)


def test_session_state_is_atomic_resumable_and_events_are_append_only(tmp_path):
    session = AutocalSession.create_or_resume(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        z_gain=0.0,
        node_ids=("N11", "N12", "N13"),
        calibration_revision=2,
        session_name="resume-me",
    )
    session.update_node("N11", status="done", attempts=1, last_values={"t1": 6.2})
    session.set_working_values({"derived.t1": 6.2})
    session.event(
        "fit_evaluated",
        node="N11",
        decision="accept",
        reason="test",
        gates={"r_squared": {"value": 0.99, "passed": True}},
    )

    resumed = AutocalSession.create_or_resume(
        tmp_path,
        target="coherence_only",
        autonomy_level=0,
        z_gain=0.0,
        node_ids=("N11", "N12", "N13"),
        calibration_revision=2,
        session_name="resume-me",
    )
    assert resumed.node("N11")["status"] == "done"
    assert resumed.state["working_values"]["derived.t1"] == 6.2
    assert replay_decisions(resumed.directory) == resumed.events()
    assert [event["event"] for event in resumed.events()] == [
        "session_started",
        "fit_evaluated",
    ]


def test_budget_model_and_hard_caps():
    plan = ExperimentPlan(
        name="t1",
        quick_class="T1",
        title="budget",
        variables={
            "time": np.linspace(0.0, 30.0, 301),
            "hard_avg": 1000,
            "soft_avg": 2,
            "r_relax": 60.0,
        },
        axes=("time",),
        signal_names=("amplitude", "phase", "i", "q"),
    )
    estimate = BudgetModel(fixed_overhead_seconds=2.0).estimate(plan)
    assert estimate > 2.0
    tracker = BudgetTracker(
        max_wall_clock_seconds=estimate + 1.0,
        max_total_runs=1,
    )
    tracker.check(estimate)
    tracker.record(0.5)
    with pytest.raises(BudgetExceeded, match="run cap"):
        tracker.check(0.1)


def test_gp_finds_correlated_optimum_and_beats_sequential_1d_baseline():
    optimum = np.asarray([0.75, 0.725, 0.425])
    scales = np.asarray([0.22, 0.16, 0.18])

    def fidelity(point):
        x, y, z = point
        return float(
            -((x - 0.75) / scales[0]) ** 2
            - ((y - (0.2 + 0.7 * x)) / scales[1]) ** 2
            - ((z - (0.8 - 0.5 * x)) / scales[2]) ** 2
        )

    result = bayesian_optimize(
        fidelity,
        [(0.0, 1.0)] * 3,
        length_scales=scales,
        max_evaluations=30,
        initial_points=8,
        seed=0,
    )
    assert np.all(np.abs(result.x - optimum) <= scales)
    assert result.x_history.shape == (30, 3)

    sequential = np.full(3, 0.5)
    for dimension in range(3):
        candidates = np.linspace(0.0, 1.0, 8)
        values = []
        for candidate in candidates:
            point = sequential.copy()
            point[dimension] = candidate
            values.append(fidelity(point))
        sequential[dimension] = candidates[int(np.argmax(values))]
    assert result.y > fidelity(sequential)


def test_node_outcome_carries_a_failure_classification_field():
    from quickexp_v3.autocal.nodes import NodeOutcome

    outcome = NodeOutcome("retake", "gate failed", {})
    assert outcome.classification is None

    classified = NodeOutcome(
        "retake",
        "gate failed",
        {},
        classification={
            "failure_class": "A",
            "coverage_reasons": ("detectability",),
            "candidate_count": 2,
            "proposed_remediation": "averaging",
        },
    )
    assert classified.classification["failure_class"] == "A"


def test_multiple_candidates_classify_as_identity_ambiguity():
    from pathlib import Path

    from quickexp_v3.autocal.hp.candidates import Candidate
    from quickexp_v3.autocal.hp.coverage import CoverageAssessment
    from quickexp_v3.autocal.nodes import classify_failure

    def _candidate(center, contrast, rank):
        return Candidate(
            candidate_id="c{0}".format(rank),
            center_mhz=center,
            fwhm_mhz=1.0,
            contrast=contrast,
            center_uncertainty_mhz=0.05,
            local_snr=contrast / 0.01,
            rank=rank,
            source_csv=Path("t.csv"),
            window_mhz=(5500.0, 5700.0),
            statistics={"rmse": 0.01},
        )

    sufficient = CoverageAssessment(True, (), 1.0, 20.0, 0.03, 10.0)
    candidates = (_candidate(5600.0, 0.5, 0), _candidate(5560.0, 0.45, 1))
    assert classify_failure(candidates, sufficient)["failure_class"] == "B"


def test_insufficient_coverage_classifies_as_instrument_limited():
    from quickexp_v3.autocal.hp.coverage import CoverageAssessment
    from quickexp_v3.autocal.nodes import classify_failure

    insufficient = CoverageAssessment(
        False,
        ("detectability",),
        1.0,
        20.0,
        0.03,
        10.0,
    )
    result = classify_failure((), insufficient)
    assert result["failure_class"] == "A"
    assert result["proposed_remediation"] == "averaging"
