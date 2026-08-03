from pathlib import Path

from quickexp_v3.autocal.hp.candidates import Candidate
from quickexp_v3.autocal.hp.coverage import CoverageAssessment
from quickexp_v3.autocal.hp.engine import HypothesisNodeSpec, run


def _candidate(identifier="candidate"):
    return Candidate(
        candidate_id=identifier,
        center_mhz=5600.0,
        fwhm_mhz=1.0,
        contrast=0.5,
        center_uncertainty_mhz=0.05,
        local_snr=10.0,
        rank=0,
        source_csv=Path("trace.csv"),
        window_mhz=(5590.0, 5610.0),
        statistics={"rmse": 0.05},
    )


def _coverage(sufficient=True):
    return CoverageAssessment(
        sufficient=sufficient,
        reasons=() if sufficient else ("resolution",),
        prior_coverage=1.0,
        points_per_fwhm=10.0 if sufficient else 2.0,
        detectable_contrast=0.1,
        edge_margin_fwhm=5.0,
    )


def _context(**extra):
    context = {
        "coverage": _coverage(),
        "margin_threshold": 2.0,
        "probe_budget_seconds": 1000.0,
        "power_exponent_tolerance": 0.2,
        "flux_slope_tolerance_mhz_per_z": 100.0,
        "flux_curvature_tolerance_mhz_per_z2": 100.0,
        "rabi_exponent_tolerance": 0.2,
        "rabi_contrast_tolerance": 0.1,
        "dispersive_shift_tolerance_mhz": 0.2,
        "qubit_flux_slope_mhz_per_z": -1000.0,
        "qubit_flux_curvature_mhz_per_z2": -5000.0,
        "expected_rabi_contrast": 0.7,
        "estimated_probe_run_seconds": 1.0,
    }
    context.update(extra)
    return context


def _spec(probes=("drive_power_ladder", "rabi_ping")):
    return HypothesisNodeSpec(
        node_id="N5",
        acquire=lambda _ctx, _attempt: Path("trace.csv"),
        extract=lambda _path: (_candidate(),),
        hypotheses=(
            "qubit_01",
            "neighbor_qubit",
            "f02_two_photon",
            "spurious",
            "novel",
        ),
        wanted="qubit_01",
        probes=probes,
        predictions=("rabi_gain_linearity",),
        product_address="defaults.q_freq",
    )


def test_engine_does_not_probe_when_coverage_is_insufficient():
    calls = []
    result = run(
        _spec(),
        _context(
            coverage=_coverage(False),
            probe_runner=lambda *_args: calls.append(True),
        ),
    )
    assert result.adjudication.action == "remediate"
    assert result.probes_run == ()
    assert calls == []


def test_engine_runs_probes_in_cost_order_and_exits_on_margin():
    calls = []

    def runner(_ctx, probe, _candidate, _runs):
        calls.append(probe.probe_id)
        if probe.probe_id == "drive_power_ladder":
            return {"contrast_power_exponent": 1.0}
        return {"rabi_gain_exponent": 1.0, "rabi_contrast": 0.7}

    result = run(_spec(), _context(probe_runner=runner))
    assert result.adjudication.action == "accept"
    assert result.adjudication.hypothesis_id == "qubit_01"
    assert calls == ["drive_power_ladder", "rabi_ping"]
    assert result.probes_run == ("drive_power_ladder", "rabi_ping")


def test_engine_escalates_before_exceeding_separate_probe_budget():
    result = run(
        _spec(("drive_power_ladder",)),
        _context(probe_budget_seconds=3.0, probe_runner=lambda *_args: {}),
    )
    assert result.adjudication.action == "escalate"
    assert result.probes_run == ()
    assert result.probe_seconds == 0.0


def test_completed_physics_prediction_must_be_inside_declared_tolerance():
    def runner(_ctx, probe, _candidate, _runs):
        if probe.probe_id == "drive_power_ladder":
            return {"contrast_power_exponent": 1.0}
        return {"rabi_gain_exponent": 1.25, "rabi_contrast": 0.7}

    result = run(_spec(), _context(probe_runner=runner))
    assert result.scorecard.margin >= 2.0
    assert result.adjudication.action == "consult"
    assert "consistency" in result.adjudication.reason
