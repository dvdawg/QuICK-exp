from dataclasses import fields
from pathlib import Path

from quickexp_v3.autocal.hp.candidates import Candidate
from quickexp_v3.autocal.hp.coverage import assess_coverage
from quickexp_v3.autocal.hp.remediation import next_remediation


def _candidate(center, fwhm=1.0, contrast=0.5, window=(5500.0, 5700.0)):
    return Candidate(
        candidate_id="test",
        center_mhz=center,
        fwhm_mhz=fwhm,
        contrast=contrast,
        center_uncertainty_mhz=0.05,
        local_snr=contrast / 0.01,
        rank=0,
        source_csv=Path("test.csv"),
        window_mhz=window,
        statistics={"rmse": 0.01},
    )


def _as_dict(candidate):
    return {item.name: getattr(candidate, item.name) for item in fields(candidate)}


def test_full_coverage_is_sufficient():
    assessment = assess_coverage(
        candidates=(_candidate(5600.0),),
        prior_window=(5550.0, 5650.0),
        scan_window=(5500.0, 5700.0),
        points=2001,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.5,
    )
    assert assessment.sufficient
    assert assessment.reasons == ()


def test_scan_missing_prior_mass_is_insufficient():
    assessment = assess_coverage(
        candidates=(_candidate(5560.0, window=(5550.0, 5570.0)),),
        prior_window=(5500.0, 5700.0),
        scan_window=(5550.0, 5570.0),
        points=201,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.5,
    )
    assert not assessment.sufficient
    assert "prior_coverage" in assessment.reasons


def test_coarse_resolution_is_insufficient():
    assessment = assess_coverage(
        candidates=(_candidate(5600.0),),
        prior_window=(5550.0, 5650.0),
        scan_window=(5500.0, 5700.0),
        points=41,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.5,
    )
    assert not assessment.sufficient
    assert "resolution" in assessment.reasons


def test_candidate_at_the_window_edge_is_insufficient():
    assessment = assess_coverage(
        candidates=(_candidate(5500.4),),
        prior_window=(5500.0, 5700.0),
        scan_window=(5500.0, 5700.0),
        points=2001,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.5,
    )
    assert not assessment.sufficient
    assert assessment.reasons == ("edge_proximity",)


def test_undetectable_expected_contrast_is_insufficient():
    weak = _candidate(5600.0, contrast=0.001)
    weak = Candidate(**dict(_as_dict(weak), statistics={"rmse": 0.5}))
    assessment = assess_coverage(
        candidates=(weak,),
        prior_window=(5550.0, 5650.0),
        scan_window=(5500.0, 5700.0),
        points=2001,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.05,
    )
    assert not assessment.sufficient
    assert "detectability" in assessment.reasons


def test_any_real_candidate_at_an_edge_marks_coverage_insufficient():
    candidates = (
        _candidate(5600.0, contrast=0.6),
        Candidate(
            **dict(
                _as_dict(_candidate(5699.5, contrast=0.4)),
                candidate_id="edge",
                rank=1,
            )
        ),
    )
    assessment = assess_coverage(
        candidates=candidates,
        prior_window=(5500.0, 5700.0),
        scan_window=(5500.0, 5700.0),
        points=2001,
        expected_fwhm_mhz=1.0,
        expected_contrast=0.5,
    )
    assert "edge_proximity" in assessment.reasons


def _insufficient(*reasons):
    from quickexp_v3.autocal.hp.coverage import CoverageAssessment

    return CoverageAssessment(
        sufficient=False,
        reasons=tuple(reasons),
        prior_coverage=0.5,
        points_per_fwhm=2.0,
        detectable_contrast=0.03,
        edge_margin_fwhm=0.5,
    )


def test_detectability_failure_escalates_averaging_first():
    step = next_remediation(
        _insufficient("detectability"),
        attempted=(),
        current_overrides={"hard_avg": 100},
    )
    assert step.step_id == "averaging"
    assert step.overrides["hard_avg"] == 200


def test_prior_coverage_failure_widens_the_window():
    step = next_remediation(
        _insufficient("prior_coverage"),
        attempted=(),
        current_overrides={},
    )
    assert step.step_id == "window"


def test_attempted_steps_are_not_repeated():
    step = next_remediation(
        _insufficient("detectability"),
        attempted=("averaging",),
        current_overrides={"hard_avg": 200},
    )
    assert step.step_id != "averaging"


def test_exhausted_ladder_returns_none():
    step = next_remediation(
        _insufficient("detectability"),
        attempted=(
            "averaging",
            "timing",
            "readout_power",
            "window",
            "held_flux",
        ),
        current_overrides={},
    )
    assert step is None


def test_sufficient_assessment_needs_no_remediation():
    from quickexp_v3.autocal.hp.coverage import CoverageAssessment

    assessment = CoverageAssessment(True, (), 1.0, 20.0, 0.01, 10.0)
    assert next_remediation(assessment, attempted=(), current_overrides={}) is None
