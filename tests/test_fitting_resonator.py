import os
import numpy as np
import pytest

from labreadout import fitting

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "2026-06-28_MET_ver191"
)


def _synthetic_resonator(f0, q, depth_db, fspan=20.0, npts=1000, seed=0):
    """Build a noisy inverted-Lorentzian magnitude dip with known f0/Q."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(f0 - fspan / 2, f0 + fspan / 2, npts)
    hwhm = f0 / (2 * q)
    # linear magnitude baseline 1.0, dip down by depth (in linear terms)
    depth_lin = 1.0 - 10 ** (-depth_db / 20.0)
    mag = 1.0 - depth_lin * hwhm**2 / ((freq - f0) ** 2 + hwhm**2)
    phase = rng.normal(0, 0.01, npts)
    I = mag * np.cos(phase) + rng.normal(0, 0.003, npts)
    Q = mag * np.sin(phase) + rng.normal(0, 0.003, npts)
    return freq, I, Q


def test_recovers_known_center_frequency():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.f0 == pytest.approx(6584.5, abs=0.05)


def test_recovers_known_quality_factor():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.q == pytest.approx(2000, rel=0.20)


def test_reports_goodness_of_fit():
    freq, I, Q = _synthetic_resonator(f0=6584.5, q=2000, depth_db=12.0)
    fit = fitting.fit_resonator(freq, I, Q)
    assert fit.gof > 0.9  # r-squared on clean synthetic data


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
