import numpy as np
import pytest

from quickexp_v3.fit_stats import (
    bic,
    bootstrap_1d,
    oriented_rotate_iq,
    pinned_parameters,
    residual_ripple_fraction,
    r_squared,
)


def test_shared_fit_statistics_cover_core_diagnostics():
    assert r_squared(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)
    assert pinned_parameters(
        {"center": 1.0, "width": 2.0},
        {"center": 0.0, "width": 2.0},
        {"center": 1.0, "width": 4.0},
    ) == ["center", "width"]
    x = np.linspace(-1.0, 1.0, 80)
    linear = 0.2 + 1.7 * x
    quadratic = linear + 0.8 * x**2
    linear_rss = np.sum((quadratic - np.polyval(np.polyfit(x, quadratic, 1), x)) ** 2)
    quadratic_rss = np.sum((quadratic - np.polyval(np.polyfit(x, quadratic, 2), x)) ** 2)
    assert bic(linear_rss, len(x), 2) - bic(quadratic_rss, len(x), 3) > 10


def test_bootstrap_and_ripple_are_deterministic():
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 100)
    y = 1.2 + 2.5 * x + rng.normal(0.0, 0.03, x.size)

    def fit(values, observed):
        slope, intercept = np.polyfit(values, observed, 1)
        return np.array([intercept, slope])

    result = bootstrap_1d(x, y, fit, n_resamples=50, seed=4)
    assert result["samples"].shape == (50, 2)
    assert result["ci_low"][1] < 2.5 < result["ci_high"][1]
    ripple = np.sin(2 * np.pi * 6 * x)
    noise = rng.normal(size=x.size)
    assert residual_ripple_fraction(x, ripple) > 0.9
    assert residual_ripple_fraction(x, noise) < 0.2


def test_oriented_iq_projection_has_stable_sign_and_angle():
    coordinate = np.linspace(-2.0, 1.0, 101)
    iq = coordinate * np.exp(0.7j)
    projected, details = oriented_rotate_iq(iq)
    assert projected[np.argmax(np.abs(projected - np.median(projected)))] > 0
    assert -np.pi / 2 <= details["angle_rad"] < np.pi / 2

