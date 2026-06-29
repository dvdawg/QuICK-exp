import numpy as np
import pytest

from labreadout import fitting


def _synthetic_rabi(period, tau, x_max, npts=200, seed=2):
    """Decaying cosine starting at a maximum (population vs drive)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, x_max, npts)
    y = 0.5 + 0.4 * np.exp(-x / tau) * np.cos(2 * np.pi * x / period)
    y = y + rng.normal(0, 0.01, npts)
    return x, y


def test_recovers_oscillation_period():
    x, y = _synthetic_rabi(period=0.24, tau=2.0, x_max=1.0)
    fit = fitting.fit_rabi(x, y)
    assert fit.period == pytest.approx(0.24, rel=0.10)


def test_pi_value_is_half_period():
    x, y = _synthetic_rabi(period=0.24, tau=2.0, x_max=1.0)
    fit = fitting.fit_rabi(x, y)
    assert fit.pi_value == pytest.approx(0.12, rel=0.10)


def test_reports_goodness_of_fit():
    x, y = _synthetic_rabi(period=0.24, tau=2.0, x_max=1.0)
    fit = fitting.fit_rabi(x, y)
    assert fit.gof > 0.9


def test_provides_uncertainties():
    x, y = _synthetic_rabi(period=0.24, tau=2.0, x_max=1.0)
    fit = fitting.fit_rabi(x, y)
    assert set(fit.uncertainties) == set(fit.params)
    assert all(np.isfinite(v) for v in fit.uncertainties.values())
