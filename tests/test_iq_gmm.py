from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.iq_gmm import fit_iq_gmm, fit_readout_optimization


REAL_SOURCE = Path(
    "/Users/dvdkm/Documents/code/qdg/data/2026-07-21_MET_ver191_qubit2/"
    "00078 - (DispersiveSpectroscopy)Zp0p0000_dispersive_q5603.910.csv"
)


def test_gmm_handles_anisotropic_clouds_and_reports_leakage():
    rng = np.random.default_rng(9)
    covariance_ground = np.array([[0.22, 0.12], [0.12, 0.12]])
    covariance_excited = np.array([[0.08, -0.03], [-0.03, 0.30]])
    ground = rng.multivariate_normal([-0.8, -0.2], covariance_ground, 2500)
    excited = rng.multivariate_normal([0.8, 0.3], covariance_excited, 2500)
    leaked = rng.multivariate_normal([-0.8, -0.2], covariance_ground, 125)
    excited[:125] = leaked
    fit = fit_iq_gmm(
        ground[:, 0],
        ground[:, 1],
        excited[:, 0],
        excited[:, 1],
        seed=2,
    )
    assert fit.assignment_fidelity > 0.85
    assert fit.cross_validated_fidelity >= fit.cross_validated_baseline_fidelity - 0.01
    assert fit.leakage["excited_as_ground_posterior_gt_0p9"] >= 0.04


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="local real-data mirror absent")
def test_real_dispersive_pair_has_finite_readout_optimum():
    fit = fit_readout_optimization(REAL_SOURCE)
    assert np.isfinite(fit.optimum_frequency_mhz)
    assert fit.snr_at_optimum > 0

