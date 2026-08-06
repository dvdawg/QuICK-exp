from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.notch_fit import (
    complex_notch_model,
    fit_notch,
    fit_spectroscopy_features,
)
from test_native_fit import write_pair


REAL_ROOT = Path(
    "/Users/dvdkm/Documents/code/qdg/data/2026-07-21_MET_ver191_qubit3"
)


def _write_complex(path, frequency, iq):
    np.savetxt(
        path,
        np.column_stack(
            (frequency, np.abs(iq), np.angle(iq), iq.real, iq.imag)
        ),
        delimiter=",",
    )
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [["Readout Pulse Frequency", "MHz"]],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": "ResonatorSpectroscopy",
                    "var": {"z_gain": 0.0},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_complex_notch_recovers_synthetic_center_and_linewidth(tmp_path):
    rng = np.random.default_rng(4)
    frequency = np.linspace(6878.0, 6890.0, 301)
    parameters = [6884.2, 0.8, 1.0, 0.002, 0.3, 0.3, 0.7, 0.25]
    iq = complex_notch_model(frequency, parameters)
    iq += 0.004 * (
        rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    )
    source = tmp_path / "notch.csv"
    _write_complex(source, frequency, iq)
    fit = fit_notch(source, allow_fallback=False)
    assert fit.center_mhz == pytest.approx(6884.2, abs=0.2 * np.diff(frequency)[0])
    assert fit.parameters["fwhm_mhz"] == pytest.approx(0.8, rel=0.1)
    assert fit.statistics["r_squared_complex"] > 0.98


def test_complex_notch_recovers_magnitude_peak(tmp_path):
    rng = np.random.default_rng(8)
    frequency = np.linspace(6878.0, 6890.0, 301)
    parameters = [6884.2, 0.8, 1.0, 0.002, 0.3, 0.3, -0.55, 0.08]
    iq = complex_notch_model(frequency, parameters)
    iq += 0.002 * (
        rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    )
    source = tmp_path / "peak.csv"
    _write_complex(source, frequency, iq)

    fit = fit_notch(source, allow_fallback=False)

    assert fit.center_mhz == pytest.approx(6884.2, abs=0.05)
    assert fit.parameters["feature_polarity"] == "peak"
    assert fit.statistics["contrast_snr"] > 5.0


def test_two_feature_selection_requires_explicit_window(tmp_path):
    frequency = np.linspace(5590.0, 5610.0, 401)
    signal = (
        0.1
        + 1.0 / (1.0 + ((frequency - 5597.0) / 0.5) ** 2)
        + 0.6 / (1.0 + ((frequency - 5603.0) / 0.6) ** 2)
    )
    source = write_pair(
        tmp_path / "qubit.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    fit = fit_spectroscopy_features(source)
    assert fit.statistics["multi_feature"]
    assert not fit.passes(
        minimum_r_squared=0.5,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.3,
    )
    windowed = fit_spectroscopy_features(
        source,
        window_mhz=(5594.0, 5600.0),
    )
    assert windowed.passes(
        minimum_r_squared=0.5,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.3,
    )
    amplitude_fit = fit_spectroscopy_features(
        source,
        signal="amplitude",
    )
    assert amplitude_fit.statistics["multi_feature"]


def test_overlapping_two_component_bic_gain_is_not_two_features(tmp_path):
    frequency = np.linspace(5590.0, 5610.0, 401)
    signal = (
        0.1
        + 1.0 / (1.0 + ((frequency - 5600.0) / 0.8) ** 2)
        + 0.15 / (1.0 + ((frequency - 5600.2) / 2.5) ** 2)
    )
    source = write_pair(
        tmp_path / "overlap.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    fit = fit_spectroscopy_features(source, signal="amplitude")
    assert not fit.statistics["multi_feature"]
    # The detector, not the information criterion, decides how many features
    # there are. Splitting a narrow line on a broad pedestal into two
    # displaced components still cuts the residual sum enormously, so the
    # two-component fit remains "resolved" on the local window; what must not
    # happen is that gain being reported as a second feature.
    assert fit.statistics["detected_candidates"] == 1


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="local real-data mirror absent")
def test_real_notches_reproduce_named_features():
    fine = fit_notch(
        REAL_ROOT / "00021 - (ResonatorSpectroscopy)VNA-test.csv"
    )
    assert fine.center_mhz == pytest.approx(6883.67, abs=0.05)
    assert fine.statistics["r_squared_complex"] >= 0.96
    neighbor = fit_notch(
        REAL_ROOT / "00019 - (ResonatorSpectroscopy)VNA-test.csv"
    )
    assert neighbor.center_mhz == pytest.approx(6817.6, abs=0.1)
