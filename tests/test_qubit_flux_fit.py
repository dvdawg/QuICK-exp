from types import MappingProxyType

import numpy as np
import pytest

import quickexp_v3.qubit_flux_fit as qubit_flux_fit
from quickexp_v3.errors import AnalysisError
from quickexp_v3.native_map import NativeMap
from quickexp_v3.qubit_flux_fit import (
    fit_qubit_flux,
    fit_transmon_flux,
    transmon_frequency,
)


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


def _native_map(tmp_path):
    outer = np.linspace(-0.3, 0.3, 7)
    inner = np.linspace(100.0, 120.0, 21)
    signal = np.arange(outer.size * inner.size, dtype=float).reshape(
        outer.size, inner.size
    )
    return NativeMap(
        source_csv=tmp_path / "map.csv",
        outer_label="Z gain",
        outer_unit="a.u.",
        outer=outer,
        inner_label="Qubit pulse frequency",
        inner_unit="MHz",
        inner=inner,
        signals=MappingProxyType(
            {"amplitude": signal, "i": signal, "q": np.zeros_like(signal)}
        ),
        row_metadata=MappingProxyType({}),
        incomplete_outer=np.asarray([]),
        metadata=MappingProxyType({}),
    )


def test_qubit_flux_fit_applies_frequency_and_flux_windows(monkeypatch, tmp_path):
    native = _native_map(tmp_path)
    extracted_frequency_axes = []

    monkeypatch.setattr(qubit_flux_fit, "load_native_map", lambda _path: native)

    def extract_row(frequencies, iq):
        extracted_frequency_axes.append(frequencies.copy())
        center = 110.0 + float(iq[0].real) / 1000.0
        return center, 0.1, 0.9, 5.0, iq.real

    monkeypatch.setattr(qubit_flux_fit, "_extract_row", extract_row)
    monkeypatch.setattr(
        qubit_flux_fit,
        "fit_transmon_flux",
        lambda z, centers, **_kwargs: (
            {
                "f_max_mhz": 120.0,
                "period_z": 0.4,
                "sweet_spot_z": 0.0,
                "asymmetry": 0.2,
                "ec_mhz": 180.0,
            },
            np.asarray(centers),
            {"r_squared": 1.0, "rmse_mhz": 0.0},
            {"period": True},
        ),
    )

    fit = fit_qubit_flux(
        tmp_path / "map.csv",
        frequency_window_mhz=(116.0, 104.0),
        flux_window_z=(0.2, -0.2),
    )

    assert fit.frequencies_mhz.tolist() == list(np.arange(104.0, 117.0))
    assert fit.map_z_gain.tolist() == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])
    assert fit.z_gain.tolist() == pytest.approx(fit.map_z_gain)
    assert fit.signal_map.shape == (5, 13)
    assert fit.statistics["frequency_fit_window_mhz"] == [104.0, 116.0]
    assert fit.statistics["flux_fit_window_z"] == pytest.approx([-0.2, 0.2])
    assert len(extracted_frequency_axes) == 5
    assert all(
        axis.tolist() == fit.frequencies_mhz.tolist()
        for axis in extracted_frequency_axes
    )


@pytest.mark.parametrize(
    ("frequency_window_mhz", "flux_window_z", "message"),
    [
        ((100.0, 105.0), None, "frequency fit window contains fewer than 12"),
        (None, (-0.1, 0.1), "flux fit window contains fewer than 4"),
    ],
)
def test_qubit_flux_fit_rejects_too_small_windows(
    monkeypatch,
    tmp_path,
    frequency_window_mhz,
    flux_window_z,
    message,
):
    monkeypatch.setattr(
        qubit_flux_fit,
        "load_native_map",
        lambda _path: _native_map(tmp_path),
    )
    with pytest.raises(AnalysisError, match=message):
        fit_qubit_flux(
            tmp_path / "map.csv",
            frequency_window_mhz=frequency_window_mhz,
            flux_window_z=flux_window_z,
        )
