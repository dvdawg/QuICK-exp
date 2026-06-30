"""Tests for the advisory result-diagnosis layer (labreadout.diagnostics).

Pure: feed it the arrays + a fit-like object and assert the decision-tree verdict.
Synthetic signals give known SNR / over-range conditions; a couple of vendored
real CSVs anchor it to today's lab data.
"""

import os
import types

import numpy as np
import pytest

from labreadout import diagnostics

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _fit(f0, gof, valid=True):
    return types.SimpleNamespace(f0=f0, gof=gof, valid=valid)


def _clean_resonance(seed=0):
    rng = np.random.default_rng(seed)
    f = np.linspace(6580, 6590, 600)
    hwhm = 0.3
    mag = 1 - 0.7 * hwhm**2 / ((f - 6585.0) ** 2 + hwhm**2)
    mag = mag + rng.normal(0, 0.004, f.size)
    return f, mag


def _pure_noise(seed=1):
    rng = np.random.default_rng(seed)
    f = np.linspace(6580, 6590, 600)
    return f, 1.0 + rng.normal(0, 0.3, f.size)


# -- SNR primitives ---------------------------------------------------------- #
def test_snr_high_for_clean_feature():
    f, mag = _clean_resonance()
    assert diagnostics.estimate_snr(mag) > 20


def test_snr_low_for_pure_noise():
    f, mag = _pure_noise()
    assert diagnostics.estimate_snr(mag) < 3


# -- decision tree ----------------------------------------------------------- #
def test_clean_resonance_diagnoses_ok():
    f, mag = _clean_resonance()
    d = diagnostics.diagnose_spectroscopy(f, mag, _fit(6585.0, gof=0.98))
    assert d.status == "ok"
    assert d.snr > 20


def test_noise_dominated_is_flagged_with_actionable_advice():
    f, mag = _pure_noise()
    d = diagnostics.diagnose_spectroscopy(f, mag, _fit(6583.1, gof=0.1, valid=False))
    assert d.status == "bad"
    assert "noise" in d.likely_cause.lower()
    assert any(w in d.suggested_action.lower() for w in ("averag", "power", "path"))


def test_value_outside_expected_range_is_flagged():
    f, mag = _clean_resonance()
    # Good signal, good fit, but the resonance is far outside the operator's
    # known band -> probably the wrong feature or the wrong port.
    d = diagnostics.diagnose_spectroscopy(
        f, mag, _fit(6585.0, gof=0.98), expected_range=(6900, 6950)
    )
    assert d.status == "warn"
    assert "range" in d.likely_cause.lower() or "implausible" in d.likely_cause.lower()


def test_in_range_value_passes_sanity():
    f, mag = _clean_resonance()
    d = diagnostics.diagnose_spectroscopy(
        f, mag, _fit(6585.0, gof=0.98), expected_range=(6580, 6590)
    )
    assert d.status == "ok"


# -- over-range / railing ---------------------------------------------------- #
def test_over_range_fraction_detects_railing():
    counts = np.array([10.0, 20.0, 1500.0, 1499.0, 30.0])
    assert diagnostics.over_range_fraction(counts, fullscale=1500.0) == pytest.approx(0.4)


def test_over_range_takes_priority_over_noise():
    f, mag = _clean_resonance()
    railed = mag.copy()
    railed[:50] = 1500.0  # ADC pinned at full scale on part of the sweep
    d = diagnostics.diagnose_spectroscopy(
        f, railed, _fit(6585.0, gof=0.9), fullscale=1500.0
    )
    assert d.status == "bad"
    assert "range" in d.likely_cause.lower() or "rail" in d.likely_cause.lower()
    assert "reduce" in d.suggested_action.lower()


# -- loopback verdict -------------------------------------------------------- #
def test_diagnose_loopback_over_range_is_bad():
    d = diagnostics.diagnose_loopback("all_over_range", peak=3000, fullscale=1500)
    assert d.status == "bad"
    assert "reduce" in d.suggested_action.lower()


def test_diagnose_loopback_weak_advises_path_check():
    d = diagnostics.diagnose_loopback("weak", peak=12, fullscale=1500)
    assert d.status in ("warn", "bad")
    assert "path" in d.suggested_action.lower() or "connect" in d.suggested_action.lower()


def test_diagnose_loopback_ok_is_healthy():
    d = diagnostics.diagnose_loopback("ok", peak=1180, fullscale=1500)
    assert d.status == "ok"


# -- report string ----------------------------------------------------------- #
def test_report_contains_status_snr_and_action():
    f, mag = _clean_resonance()
    d = diagnostics.diagnose_spectroscopy(f, mag, _fit(6585.0, gof=0.98))
    text = d.report()
    assert "SNR" in text
    assert d.status.upper() in text or d.status in text


# -- anchored to real lab data ----------------------------------------------- #
def _rotated_iq(I, Q):
    pts = np.column_stack([I - I.mean(), Q - Q.mean()])
    _u, _s, vh = np.linalg.svd(pts, full_matrices=False)
    return pts @ vh[0]


def test_real_clean_resonator_is_not_noise_dominated():
    # 2026-06-29 #00064: the session's final converged fine scan. The resonator
    # step diagnoses the dB amplitude column (the natural readout).
    path = os.path.join(DATA_DIR, "2026-06-29_00064_resonator_fine.csv")
    freq, amp_db, _ph, _I, _Q = np.loadtxt(path, delimiter=",").T
    assert diagnostics.estimate_snr(amp_db) > 8


def test_real_repeated_qubit_spec_is_noise_dominated():
    # 2026-06-29 #00015: one of the repeated qubit_spec_coarse scans that kept
    # being retaken in lab -- the qubit feature is lost in the noise.
    path = os.path.join(DATA_DIR, "2026-06-29_00015_qubit_spec_noisy.csv")
    freq, _amp, _ph, I, Q = np.loadtxt(path, delimiter=",").T
    assert diagnostics.estimate_snr(_rotated_iq(I, Q)) < diagnostics.SNR_MIN
