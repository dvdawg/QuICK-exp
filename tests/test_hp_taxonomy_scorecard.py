from pathlib import Path

from quickexp_v3.autocal.hp.candidates import Candidate
from quickexp_v3.autocal.hp.coverage import CoverageAssessment
from quickexp_v3.autocal.hp.scorecard import adjudicate, build_scorecard
from quickexp_v3.autocal.hp.taxonomy import (
    hypothesis_ids,
    hypotheses_for,
)


def _candidate(candidate_id="c0", center=5600.0, rank=0):
    return Candidate(
        candidate_id=candidate_id,
        center_mhz=center,
        fwhm_mhz=1.0,
        contrast=0.5,
        center_uncertainty_mhz=0.05,
        local_snr=20.0,
        rank=rank,
        source_csv=Path("trace.csv"),
        window_mhz=(5500.0, 5700.0),
        statistics={"rmse": 0.025},
    )


def _context():
    return {
        "qubit_contrast_exponent": 1.0,
        "power_exponent_tolerance": 0.2,
        "qubit_flux_slope_mhz_per_z": 120.0,
        "neighbor_flux_slope_mhz_per_z": -80.0,
        "flux_slope_tolerance_mhz_per_z": 10.0,
        "qubit_flux_curvature_mhz_per_z2": -400.0,
        "neighbor_flux_curvature_mhz_per_z2": 180.0,
        "flux_curvature_tolerance_mhz_per_z2": 40.0,
        "rabi_exponent_tolerance": 0.2,
        "expected_rabi_contrast": 0.7,
        "rabi_contrast_tolerance": 0.1,
        "expected_dispersive_shift_mhz": 1.0,
        "dispersive_shift_tolerance_mhz": 0.15,
    }


def test_taxonomy_contains_declared_qubit_and_resonator_hypotheses():
    assert hypothesis_ids("qubit") == (
        "qubit_01",
        "f02_two_photon",
        "higher_transition",
        "tls",
        "neighbor_qubit",
        "readout_leakage",
        "spurious",
        "novel",
    )
    assert hypothesis_ids("resonator") == (
        "readout_resonator",
        "neighbor_resonator",
        "package_mode",
        "spurious",
        "novel",
    )


def test_every_declared_tolerance_is_a_scale_aware_callable():
    context = _context()
    for hypothesis in hypotheses_for("qubit") + hypotheses_for("resonator"):
        for signature in hypothesis.signatures:
            assert callable(signature.tolerance)
            first = signature.tolerance(context)
            scaled = dict(context)
            for key in tuple(scaled):
                if "tolerance" in key:
                    scaled[key] = 2.0 * scaled[key]
            second = signature.tolerance(scaled)
            assert first > 0.0
            assert second >= first


def test_qubit_like_perturbation_response_wins_by_margin():
    responses = {
        "c0": {
            "drive_power_ladder": {"contrast_power_exponent": 1.0},
            "flux_nudge": {
                "flux_slope_mhz_per_z": 120.0,
                "flux_curvature_mhz_per_z2": -400.0,
            },
            "rabi_ping": {
                "rabi_gain_exponent": 1.0,
                "rabi_contrast": 0.7,
            },
            "dispersive_response": {"dispersive_shift_mhz": 1.0},
        }
    }
    scorecard = build_scorecard(
        (_candidate(),),
        hypothesis_ids("qubit"),
        responses,
        _context(),
    )
    assert scorecard.leader.hypothesis_id == "qubit_01"
    assert scorecard.margin > 2.0


def test_two_photon_response_routes_to_derived_retry():
    responses = {
        "c0": {
            "drive_power_ladder": {"contrast_power_exponent": 2.0},
            "flux_nudge": {
                "flux_slope_mhz_per_z": 120.0,
                "flux_curvature_mhz_per_z2": -400.0,
            },
            "rabi_ping": {
                "rabi_gain_exponent": 2.0,
                "rabi_contrast": 0.35,
            },
            "dispersive_response": {"dispersive_shift_mhz": 0.5},
        }
    }
    scorecard = build_scorecard(
        (_candidate(),),
        hypothesis_ids("qubit"),
        responses,
        _context(),
    )
    verdict = adjudicate(
        scorecard,
        CoverageAssessment(True, (), 1.0, 10.0, 0.02, 10.0),
        wanted="qubit_01",
        margin_threshold=2.0,
        probes_remaining=False,
    )
    assert scorecard.leader.hypothesis_id == "f02_two_photon"
    assert verdict.action == "derive_and_retry"


def test_coverage_failure_is_always_remediation_not_rejection():
    scorecard = build_scorecard(
        (_candidate(),),
        hypothesis_ids("qubit"),
        {},
        _context(),
    )
    verdict = adjudicate(
        scorecard,
        CoverageAssessment(
            False,
            ("resolution",),
            1.0,
            2.0,
            0.02,
            10.0,
        ),
        wanted="qubit_01",
        margin_threshold=2.0,
        probes_remaining=True,
    )
    assert verdict.action == "remediate"
    assert verdict.failure_class == "A"


def test_unresolved_margin_requests_another_probe_then_consultation():
    scorecard = build_scorecard(
        (_candidate(),),
        hypothesis_ids("qubit"),
        {},
        _context(),
    )
    coverage = CoverageAssessment(True, (), 1.0, 10.0, 0.02, 10.0)
    assert adjudicate(
        scorecard,
        coverage,
        wanted="qubit_01",
        margin_threshold=2.0,
        probes_remaining=True,
    ).action == "probe"
    assert adjudicate(
        scorecard,
        coverage,
        wanted="qubit_01",
        margin_threshold=2.0,
        probes_remaining=False,
    ).action == "consult"
