import numpy as np
import pytest

from quickexp_v3.qubit_flux_fit import fit_transmon_flux, transmon_frequency


def test_transmon_flux_fit_recovers_synthetic_model():
    rng = np.random.default_rng(7)
    z = np.linspace(-0.35, 0.35, 31)
    frequency = transmon_frequency(
        z,
        f_max_mhz=4763.0,
        period_z=0.30,
        sweet_spot_z=-0.02,
        asymmetry=0.19,
        ec_mhz=180.0,
    )
    frequency += rng.normal(0.0, 0.2, z.size)
    parameters, fitted, statistics, identifiable = fit_transmon_flux(
        z,
        frequency,
        uncertainty_mhz=np.full(z.size, 0.2),
        ec_mhz=180.0,
        period_hint=0.30,
    )
    assert parameters["f_max_mhz"] == pytest.approx(4763.0, abs=1.0)
    assert parameters["period_z"] == pytest.approx(0.30, abs=0.005)
    assert parameters["asymmetry"] == pytest.approx(0.19, abs=0.03)
    assert statistics["r_squared"] > 0.999
    assert identifiable["period"]

