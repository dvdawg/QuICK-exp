import numpy as np
import pytest

from labreadout import fitting


def _two_blobs(sep, sigma, n=4000, seed=3):
    """Ground/excited IQ clouds separated along a tilted axis."""
    rng = np.random.default_rng(seed)
    angle = 0.6
    d = 0.5 * sep * np.array([np.cos(angle), np.sin(angle)])
    g = rng.normal(0, sigma, (n, 2)) - d
    e = rng.normal(0, sigma, (n, 2)) + d
    return g[:, 0], g[:, 1], e[:, 0], e[:, 1]


def test_well_separated_blobs_give_high_fidelity():
    Ig, Qg, Ie, Qe = _two_blobs(sep=6.0, sigma=1.0)
    fit = fitting.fit_iq_threshold(Ig, Qg, Ie, Qe)
    assert fit.fidelity > 0.99


def test_overlapping_blobs_give_low_fidelity():
    Ig, Qg, Ie, Qe = _two_blobs(sep=0.5, sigma=1.0)
    fit = fitting.fit_iq_threshold(Ig, Qg, Ie, Qe)
    assert fit.fidelity < 0.7


def test_threshold_lies_between_projected_centers():
    Ig, Qg, Ie, Qe = _two_blobs(sep=6.0, sigma=1.0)
    fit = fitting.fit_iq_threshold(Ig, Qg, Ie, Qe)
    assert min(fit.center_g, fit.center_e) < fit.threshold < max(
        fit.center_g, fit.center_e
    )


def test_fidelity_is_a_probability():
    Ig, Qg, Ie, Qe = _two_blobs(sep=3.0, sigma=1.0)
    fit = fitting.fit_iq_threshold(Ig, Qg, Ie, Qe)
    assert 0.0 <= fit.fidelity <= 1.0
