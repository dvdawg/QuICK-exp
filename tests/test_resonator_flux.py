from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from quickexp_v3.errors import AnalysisError, ConfigError
from quickexp_v3.flux_lookup import frequency_from_record
from quickexp_v3.resonator_flux import (
    LOOKUP_MODEL_NAME,
    accept_fit,
    calibration_record,
    fit_resonator_flux,
    frequency_from_calibration_record,
    load_scan,
    plot_resonator_flux_fit,
)


ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_scan(
    path: Path,
    *,
    incomplete_index=None,
    polarity: float = -1.0,
) -> dict:
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
        amplitude = polarity * 12.0 * np.exp(
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


@pytest.mark.parametrize(
    ("polarity", "expected_polarity"),
    [(-1.0, "dip"), (1.0, "peak")],
)
def test_notebook_style_fit_recovers_frequency_curve(
    tmp_path,
    polarity,
    expected_polarity,
):
    source = tmp_path / f"00025 - ResVsZ_held_bias_{expected_polarity}.csv"
    expected = write_synthetic_scan(source, polarity=polarity)

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
    assert fit.feature_polarity == expected_polarity
    assert fit.passes(minimum_r_squared=0.95, maximum_rmse_mhz=0.2)

    figure = plot_resonator_flux_fit(fit)
    assert len(figure.axes) == 4
    plt.close(figure)


def write_extremum_scan(path: Path, *, polarity: float, rows: int = 9):
    """Write a flux map whose per-Z feature is a clean single extremum."""
    z_values = np.linspace(-0.4, 0.4, rows)
    frequencies = np.arange(6882.8, 6885.61, 0.02)
    expected_bins = []
    stacked = []
    for z_gain in z_values:
        center = 6884.2 + 0.8 * z_gain
        feature = polarity * 12.0 * np.exp(
            -0.5 * ((frequencies - center) / 0.08) ** 2
        )
        expected_index = (
            int(np.argmax(feature)) if polarity > 0 else int(np.argmin(feature))
        )
        expected_bins.append(frequencies[expected_index])
        stacked.append(
            np.column_stack(
                [
                    np.full_like(frequencies, z_gain),
                    frequencies,
                    feature,
                ]
            )
        )
    np.savetxt(path, np.vstack(stacked), delimiter=",")
    return z_values, np.asarray(expected_bins)


@pytest.mark.parametrize(
    ("method", "polarity"),
    [("min", -1.0), ("max", 1.0)],
)
def test_sampled_extremum_lookup_interpolates_selected_bins(
    tmp_path,
    method,
    polarity,
):
    source = tmp_path / f"{method}_scan.csv"
    z_gain, expected_bins = write_extremum_scan(source, polarity=polarity)

    fit = fit_resonator_flux(source, fit_method=method, smooth_sigma_bins=0.0)

    assert fit.fit_method == method
    assert fit.model == LOOKUP_MODEL_NAME
    assert fit.extracted_frequencies_mhz == pytest.approx(expected_bins)
    midpoints = 0.5 * (z_gain[:-1] + z_gain[1:])
    expected_midpoints = 0.5 * (expected_bins[:-1] + expected_bins[1:])
    assert fit.frequency(midpoints) == pytest.approx(expected_midpoints)
    assert fit.passes(minimum_r_squared=1.0, maximum_rmse_mhz=0.0)

    record = calibration_record(fit)
    assert record["model"] == LOOKUP_MODEL_NAME
    assert record["value"]["selection_method"] == method
    assert frequency_from_calibration_record(
        record,
        midpoints,
    ) == pytest.approx(expected_midpoints)
    # The shared flux-lookup dispatcher must resolve it too, since that is
    # what later measurements and the device report actually call.
    assert frequency_from_record(record, midpoints) == pytest.approx(
        expected_midpoints
    )

    figure = plot_resonator_flux_fit(fit)
    assert len(figure.axes) == 4
    plt.close(figure)


def test_sampled_lookup_accepts_fewer_rows_than_the_cosine_fit(tmp_path):
    source = tmp_path / "short_scan.csv"
    write_extremum_scan(source, polarity=-1.0, rows=3)

    fit = fit_resonator_flux(source, fit_method="min", smooth_sigma_bins=0.0)
    assert len(fit.z_gain) == 3

    with pytest.raises(AnalysisError, match="complete finite Z rows"):
        fit_resonator_flux(source, fit_method="fit")


def test_unknown_fit_method_is_rejected(tmp_path):
    source = tmp_path / "scan.csv"
    write_synthetic_scan(source)

    with pytest.raises(AnalysisError, match="fit_method"):
        fit_resonator_flux(source, fit_method="median")


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


def test_accepted_sampled_lookup_is_available_to_later_measurements(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    source = tmp_path / "00025 - ResVsZ_held_bias.csv"
    write_extremum_scan(source, polarity=-1.0)
    fit = fit_resonator_flux(source, fit_method="min", smooth_sigma_bins=0.0)

    target = accept_fit(
        tmp_path,
        fit,
        minimum_r_squared=1.0,
        maximum_rmse_mhz=0.0,
    )

    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    record = document["records"]["lookups"]["resonator_vs_flux"]
    assert record["model"] == LOOKUP_MODEL_NAME
    assert record["quality"]["selection_method"] == "min"
    assert frequency_from_record(record, 0.05) == pytest.approx(
        fit.frequency(0.05)
    )


def test_sampled_lookup_rejects_out_of_domain_z(tmp_path):
    source = tmp_path / "scan.csv"
    write_extremum_scan(source, polarity=-1.0)
    record = calibration_record(
        fit_resonator_flux(source, fit_method="min", smooth_sigma_bins=0.0)
    )

    with pytest.raises(ConfigError, match="outside accepted"):
        frequency_from_record(record, 0.41)


def test_calibration_record_enforces_measured_domain(tmp_path):
    source = tmp_path / "scan.csv"
    write_synthetic_scan(source)
    record = calibration_record(
        fit_resonator_flux(source, smooth_sigma_bins=1.0)
    )

    with pytest.raises(ConfigError, match="outside accepted"):
        frequency_from_calibration_record(record, 0.41)
