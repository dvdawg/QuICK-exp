import numpy as np
import pytest

from labreadout import fitting


def test_t1_recovers_decay_constant():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 30, 200)
    y = 0.1 + 0.8 * np.exp(-t / 8.0) + rng.normal(0, 0.01, t.size)
    fit = fitting.fit_t1(t, y)
    assert fit.t1 == pytest.approx(8.0, rel=0.10)
    assert fit.gof > 0.95


def test_t1_provides_uncertainty():
    t = np.linspace(0, 30, 200)
    y = 0.1 + 0.8 * np.exp(-t / 8.0)
    fit = fitting.fit_t1(t, y)
    assert np.isfinite(fit.uncertainties["t1"])


def test_t2_recovers_decay_and_detuning():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 20, 300)
    period = 2.0
    y = 0.5 + 0.4 * np.exp(-t / 6.0) * np.cos(2 * np.pi * t / period)
    y = y + rng.normal(0, 0.01, t.size)
    fit = fitting.fit_t2(t, y)
    assert fit.t2 == pytest.approx(6.0, rel=0.15)
    assert fit.detuning == pytest.approx(1.0 / period, rel=0.10)
