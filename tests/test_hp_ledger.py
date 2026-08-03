import pytest

from quickexp_v3.autocal.hp.ledger import (
    BacktrackLimitExceeded,
    DiscrepancyLedger,
    HypothesisLedger,
    render_discrepancy_report,
)


def _ranking(prefix="c"):
    return (
        {"candidate_id": prefix + "1", "hypothesis_id": "qubit_01", "score": 0.0},
        {"candidate_id": prefix + "2", "hypothesis_id": "qubit_01", "score": -1.0},
        {"candidate_id": prefix + "3", "hypothesis_id": "qubit_01", "score": -2.0},
    )


def test_hypothesis_ledger_demotes_without_repromoting_in_session():
    ledger = HypothesisLedger(max_session_backtracks=3, max_address_backtracks=2)
    ledger.record("defaults.q_freq", _ranking(), evidence={"probe": "P1"})
    assert ledger.leader("defaults.q_freq")["candidate_id"] == "c1"

    first = ledger.upstream_doubt("defaults.q_freq", {"node": "N8"})
    second = ledger.upstream_doubt("defaults.q_freq", {"node": "N12"})
    assert first.promoted_candidate_id == "c2"
    assert second.promoted_candidate_id == "c3"
    assert ledger.demoted("defaults.q_freq") == ("c1", "c2")

    # A later scorecard cannot silently put a demoted option back on top.
    ledger.record("defaults.q_freq", _ranking(), evidence={"probe": "P3"})
    assert ledger.leader("defaults.q_freq")["candidate_id"] == "c3"


def test_joint_retune_is_tried_before_candidate_backtrack():
    ledger = HypothesisLedger(max_session_backtracks=3, max_address_backtracks=2)
    ledger.record("defaults.q_freq", _ranking())
    decision = ledger.upstream_doubt(
        "defaults.q_freq",
        {"node": "N8", "reason": "weak Rabi"},
        joint_retune_available=True,
    )
    assert decision.action == "retune_joint_operating_point"
    assert decision.promoted_candidate_id == "c1"
    assert ledger.total_backtracks == 0


def test_backtrack_caps_raise_instead_of_thrashing():
    ledger = HypothesisLedger(max_session_backtracks=1, max_address_backtracks=1)
    ledger.record("defaults.q_freq", _ranking())
    ledger.upstream_doubt("defaults.q_freq", {})
    with pytest.raises(BacktrackLimitExceeded):
        ledger.upstream_doubt("defaults.q_freq", {})


def test_exhausted_candidate_list_does_not_demote_the_only_candidate():
    ledger = HypothesisLedger(max_session_backtracks=3, max_address_backtracks=2)
    ledger.record("defaults.q_freq", _ranking()[:1])
    before = ledger.as_dict()
    with pytest.raises(BacktrackLimitExceeded, match="no alternate"):
        ledger.upstream_doubt("defaults.q_freq", {})
    after = ledger.as_dict()
    assert after["total_backtracks"] == before["total_backtracks"]
    assert after["addresses"]["defaults.q_freq"]["demoted"] == []


def test_discrepancy_ledger_computes_residual_and_renders_report():
    ledger = DiscrepancyLedger()
    consistent = ledger.record(
        "rabi_gain_linearity",
        predicted=1.0,
        measured=1.1,
        sigma=0.2,
        model_assumptions=("weak drive",),
        sources=("N5",),
    )
    deviant = ledger.record(
        "t2_bound",
        predicted=12.0,
        measured=20.0,
        sigma=2.0,
        model_assumptions=("Markovian decay",),
        sources=("N11", "N12"),
    )
    untestable = ledger.record(
        "flux_period_agreement",
        predicted=None,
        measured=None,
        sigma=None,
        model_assumptions=("shared SQUID loop",),
    )
    assert consistent.verdict == "consistent"
    assert consistent.residual == pytest.approx(0.5)
    assert deviant.verdict == "deviant"
    assert untestable.verdict == "untestable"
    report = render_discrepancy_report(ledger)
    assert "rabi_gain_linearity" in report
    assert "deviant" in report
    assert "shared SQUID loop" in report


def test_discrepancy_ids_are_upserted_and_upper_bounds_are_one_sided():
    ledger = DiscrepancyLedger()
    ledger.record("chi", 1.0, 1.5, 0.1, ("first",))
    ledger.record("chi", 1.0, 1.1, 0.1, ("updated",))
    consistent = ledger.record_upper_bound(
        "t2_bound",
        upper_bound=12.0,
        measured=2.0,
        sigma=0.5,
        model_assumptions=("T2 <= 2*T1",),
    )
    deviant = ledger.record_upper_bound(
        "t2_bound_deviant",
        upper_bound=12.0,
        measured=20.0,
        sigma=1.0,
        model_assumptions=("T2 <= 2*T1",),
    )
    assert ledger.prediction_ids().count("chi") == 1
    assert consistent.residual == 0.0
    assert consistent.verdict == "consistent"
    assert deviant.verdict == "deviant"


def test_ledgers_round_trip_through_builtin_state():
    hypotheses = HypothesisLedger(3, 2)
    hypotheses.record("defaults.q_freq", _ranking())
    hypotheses.upstream_doubt("defaults.q_freq", {})
    restored = HypothesisLedger.from_dict(hypotheses.as_dict())
    assert restored.as_dict() == hypotheses.as_dict()

    discrepancies = DiscrepancyLedger()
    discrepancies.record("chi", 1.0, 0.8, 0.1, ("dispersive",))
    restored_discrepancies = DiscrepancyLedger.from_dict(discrepancies.as_dict())
    assert restored_discrepancies.as_dict() == discrepancies.as_dict()
