import os
import numpy as np
import pytest

from labreadout import fitting

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "2026-06-28_MET_ver191"
)


def _synthetic_qubit(f0, linewidth, sign=+1.0, fspan=600.0, npts=400, seed=1):
    """Noisy Lorentzian feature carried on one IQ quadrature."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(f0 - fspan / 2, f0 + fspan / 2, npts)
    hwhm = linewidth / 2.0
    feature = sign * hwhm**2 / ((freq - f0) ** 2 + hwhm**2)
    I = 0.2 + feature + rng.normal(0, 0.02, npts)
    Q = -0.1 + 0.5 * feature + rng.normal(0, 0.02, npts)
    return freq, I, Q


def test_recovers_peak_center():
    freq, I, Q = _synthetic_qubit(f0=5022.0, linewidth=8.0, sign=+1.0)
    fit = fitting.fit_qubit_peak(freq, I, Q)
    assert fit.f0 == pytest.approx(5022.0, abs=0.5)


def test_recovers_dip_center():
    freq, I, Q = _synthetic_qubit(f0=4815.0, linewidth=6.0, sign=-1.0)
    fit = fitting.fit_qubit_peak(freq, I, Q)
    assert fit.f0 == pytest.approx(4815.0, abs=0.5)


def test_recovers_linewidth():
    freq, I, Q = _synthetic_qubit(f0=5022.0, linewidth=8.0)
    fit = fitting.fit_qubit_peak(freq, I, Q)
    assert fit.linewidth == pytest.approx(8.0, rel=0.30)


def test_reports_goodness_of_fit():
    freq, I, Q = _synthetic_qubit(f0=5022.0, linewidth=8.0)
    fit = fitting.fit_qubit_peak(freq, I, Q)
    assert fit.gof > 0.7


def test_fits_real_qubit_csv_in_range():
    path = os.path.join(
        DATA_DIR, "00035 - (QubitSpectroscopy)qubit_spec_coarse_-0.2V.csv"
    )
    freq, _amp, _ph, I, Q = np.loadtxt(path, delimiter=",").T
    fit = fitting.fit_qubit_peak(freq, I, Q)
    assert freq.min() <= fit.f0 <= freq.max()
    assert np.isfinite(fit.linewidth) and fit.linewidth > 0
