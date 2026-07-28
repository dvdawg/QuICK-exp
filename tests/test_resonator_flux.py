from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from quickexp_v3.errors import AnalysisError, ConfigError
from quickexp_v3.resonator_flux import (
    accept_fit,
    calibration_record,
    fit_resonator_flux,
    frequency_from_calibration_record,
    load_scan,
    plot_resonator_flux_fit,
)


ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_scan(path: Path, *, incomplete_index=None) -> dict:
    parameters = {
        "center_frequency": 6884.2,
        "amplitude": 0.7,
        "period": 0.23,
        "peak_bias": -0.06,
    }
    z_values = np.linspace(-0.4, 0.4, 21)
    frequencies = np.arange(6882.8, 6885.61, 0.02)
    rows = []
    for index, z_gain in enumerate(z_values):
        center = (
            parameters["center_frequency"]
            + parameters["amplitude"]
            * np.cos(
                2
                * np.pi
                * (z_gain - parameters["peak_bias"])
                / parameters["period"]
            )
        )
        amplitude = -12.0 * np.exp(
            -0.5 * ((frequencies - center) / 0.08) ** 2
        )
        row_frequencies = frequencies
        row_amplitude = amplitude
        if index == incomplete_index:
            row_frequencies = frequencies[:-1]
            row_amplitude = amplitude[:-1]
        phase = np.zeros_like(row_frequencies)
        i_data = 10 ** (row_amplitude / 20)
        q_data = np.zeros_like(row_frequencies)
        rows.append(
            np.column_stack(
                [
                    np.full_like(row_frequencies, z_gain),
                    row_frequencies,
                    row_amplitude,
                    phase,
                    i_data,
                    q_data,
                ]
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")
    return parameters


def test_notebook_style_fit_recovers_frequency_curve(tmp_path):
    source = tmp_path / "00025 - ResVsZ_held_bias.csv"
    expected = write_synthetic_scan(source)

    fit = fit_resonator_flux(source, smooth_sigma_bins=1.0)

    probe = np.linspace(-0.39, 0.39, 31)
    expected_frequency = (
        expected["center_frequency"]
        + expected["amplitude"]
        * np.cos(
            2
            * np.pi
            * (probe - expected["peak_bias"])
            / expected["period"]
        )
    )
    assert np.max(np.abs(fit.frequency(probe) - expected_frequency)) < 0.03
    assert fit.statistics["r_squared"] > 0.99
    assert fit.statistics["rmse_mhz"] < 0.02
    assert fit.passes(minimum_r_squared=0.95, maximum_rmse_mhz=0.2)

    figure = plot_resonator_flux_fit(fit)
    assert len(figure.axes) == 4
    plt.close(figure)


def test_incomplete_z_row_is_dropped_but_reported(tmp_path):
    source = tmp_path / "scan.csv"
    write_synthetic_scan(source, incomplete_index=4)

    z_gain, _, _, dropped = load_scan(source)

    assert len(z_gain) == 20
    assert dropped == pytest.approx([-0.24])


def test_duplicate_grid_point_is_rejected(tmp_path):
    source = tmp_path / "scan.csv"
    write_synthetic_scan(source)
    data = np.loadtxt(source, delimiter=",")
    np.savetxt(source, np.vstack([data, data[0]]), delimiter=",")

    with pytest.raises(AnalysisError, match="duplicate"):
        load_scan(source)


def test_accepted_fit_is_atomic_calibration_record(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    source = tmp_path / "00025 - ResVsZ_held_bias.csv"
    write_synthetic_scan(source)
    fit = fit_resonator_flux(source, smooth_sigma_bins=1.0)

    target = accept_fit(
        tmp_path,
        fit,
        minimum_r_squared=0.95,
        maximum_rmse_mhz=0.2,
    )

    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    record = document["records"]["lookups"]["resonator_vs_flux"]
    assert record["status"] == "accepted"
    assert record["provenance"]["source"] == str(source.resolve())
    assert document["history"]
    assert frequency_from_calibration_record(record, 0.0) == pytest.approx(
        fit.frequency(0.0)
    )


def test_calibration_record_enforces_measured_domain(tmp_path):
    source = tmp_path / "scan.csv"
    write_synthetic_scan(source)
    record = calibration_record(
        fit_resonator_flux(source, smooth_sigma_bins=1.0)
    )

    with pytest.raises(ConfigError, match="outside accepted"):
        frequency_from_calibration_record(record, 0.41)
