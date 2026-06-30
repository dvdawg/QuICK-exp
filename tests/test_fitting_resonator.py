import os
import numpy as np
import pytest

from labreadout import fitting

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _synthetic_resonator(f0, q, depth_db, fspan=20.0, npts=1000, seed=0):
    """Build a noisy complex notch with gain, delay, and mismatch phase."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(f0 - fspan / 2, f0 + fspan / 2, npts)
    coupling = 1.0 - 10 ** (-depth_db / 20.0)
    qc = q / coupling
    z = fitting._complex_notch(
        freq, amp=1.15, alpha=0.4, tau=0.018, f0=f0,
        ql=q, qc=qc, phi=0.08,
    )
    I = z.real + rng.normal(0, 0.002, npts)
    Q = z.imag + rng.normal(0, 0.002, npts)
    return freq, I, Q


def test_recovers_known_center_frequency():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.f0 == pytest.approx(6584.5, abs=0.05)


def test_recovers_known_quality_factor():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.q == pytest.approx(2000, rel=0.20)


def test_reports_loaded_and_coupling_quality_factors():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.params["ql"] == pytest.approx(2000, rel=0.20)
    assert fit.params["qc"] > fit.params["ql"]


def test_reports_goodness_of_fit():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.gof > 0.9  # r-squared on clean synthetic data
    assert fit.valid


def test_provides_uncertainties_for_each_parameter():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert set(fit.uncertainties) == set(fit.params)
    assert all(np.isfinite(v) for v in fit.uncertainties.values())


def test_fits_real_resonator_csv_in_range():
    path = os.path.join(
        DATA_DIR, "00021 - (ResonatorSpectroscopy)resonator_fine_repeat0_-0.2V.csv"
    )
    freq, _amp, _ph, I, Q = np.loadtxt(path, delimiter=",").T
    fit = fitting.fit_resonator(freq, I, Q)
    assert freq.min() <= fit.f0 <= freq.max()
    assert np.isfinite(fit.q) and fit.q > 0


def test_window_restricts_fit_region():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0, fspan=40)
    fit = fitting.fit_resonator(freq, I, Q, window=(6582.0, 6587.0))
    assert fit.f0 == pytest.approx(6584.5, abs=0.05)
