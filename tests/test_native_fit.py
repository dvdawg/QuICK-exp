from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.native_fit import (
    find_latest_native,
    fit_loopback,
    fit_ramsey,
    fit_spectroscopy,
    fit_t1,
)


def write_pair(
    path: Path,
    *,
    quick_class: str,
    axis_label: str,
    axis_unit: str,
    x: np.ndarray,
    signal: np.ndarray,
    var=None,
) -> Path:
    angle = 0.47
    i_data = signal * np.cos(angle)
    q_data = signal * np.sin(angle)
    matrix = np.column_stack(
        (
            x,
            np.hypot(i_data, q_data),
            np.angle(i_data + 1j * q_data),
            i_data,
            q_data,
        )
    )
    np.savetxt(path, matrix, delimiter=",")
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [[axis_label, axis_unit]],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": quick_class,
                    "var": dict(var or {}),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "kind,quick_class,axis_label,center",
    [
        (
            "resonator",
            "ResonatorSpectroscopy",
            "Readout Pulse Frequency",
            6884.2,
        ),
        ("qubit", "QubitSpectroscopy", "Qubit Pulse Frequency", 5603.9),
    ],
)
def test_spectroscopy_fit_recovers_center(
    tmp_path,
    kind,
    quick_class,
    axis_label,
    center,
):
    frequency = np.linspace(center - 5.0, center + 5.0, 201)
    signal = (
        0.2
        + 0.003 * (frequency - center)
        - 0.8 / (1.0 + ((frequency - center) / 0.4) ** 2)
    )
    source = write_pair(
        tmp_path / f"{quick_class}.csv",
        quick_class=quick_class,
        axis_label=axis_label,
        axis_unit="MHz",
        x=frequency,
        signal=signal,
        var={"z_gain": 0.1},
    )
    fit = fit_spectroscopy(source, kind=kind, signal="IQ")
    assert fit.center_mhz == pytest.approx(center, abs=1e-4)
    assert fit.statistics["r_squared"] > 0.999


def test_t1_fit_recovers_lifetime(tmp_path):
    time = np.linspace(0.0, 30.0, 301)
    signal = 0.05 + 0.8 * np.exp(-time / 6.2)
    source = write_pair(
        tmp_path / "T1.csv",
        quick_class="T1",
        axis_label="Delay Time",
        axis_unit="us",
        x=time,
        signal=signal,
    )
    fit = fit_t1(source, signal="IQ")
    assert fit.t1_us == pytest.approx(6.2, rel=1e-5)
    assert fit.statistics["r_squared"] > 0.999


def test_ramsey_fit_recovers_decay_fringe_and_frequency_options(tmp_path):
    time = np.linspace(0.0, 5.0, 501)
    signal = (
        0.05
        + 0.8
        * np.exp(-time / 1.8)
        * np.cos(2.0 * np.pi * 5.2 * time + 0.3)
    )
    source = write_pair(
        tmp_path / "T2Ramsey.csv",
        quick_class="T2Ramsey",
        axis_label="Delay Time",
        axis_unit="us",
        x=time,
        signal=signal,
        var={"q_freq": 5600.0, "fringe_freq": 5.0},
    )
    fit = fit_ramsey(source, signal="IQ")
    assert fit.t2_star_us == pytest.approx(1.8, rel=1e-4)
    assert fit.parameters["fitted_fringe_mhz"] == pytest.approx(
        5.2,
        rel=1e-5,
    )
    assert fit.parameters["predicted_drive_mhz"] == pytest.approx(5600.2)
    assert fit.parameters["alternate_drive_mhz"] == pytest.approx(5599.8)


def test_loopback_fit_recovers_edge_and_adds_existing_offset(tmp_path):
    time = np.linspace(0.0, 2.0, 1001)
    signal = 0.05 + 1.2 * 0.5 * (
        1.0 + np.tanh((time - 0.49) / (2.0 * 0.008))
    )
    source = write_pair(
        tmp_path / "LoopBack.csv",
        quick_class="LoopBack",
        axis_label="Time",
        axis_unit="us",
        x=time,
        signal=signal,
        var={"r_offset": 0.1},
    )
    fit = fit_loopback(source, smooth_sigma_bins=1.0)
    assert fit.parameters["edge_in_trace_us"] == pytest.approx(0.49, abs=0.002)
    assert fit.recommended_r_offset_us == pytest.approx(0.59, abs=0.002)
    assert fit.statistics["r_squared"] > 0.999


def test_find_latest_native_skips_wrong_class_and_two_dimensional_file(
    tmp_path,
):
    frequency = np.linspace(6880.0, 6885.0, 21)
    signal = np.ones_like(frequency)
    expected = write_pair(
        tmp_path / "00001.csv",
        quick_class="ResonatorSpectroscopy",
        axis_label="Readout Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    wrong = write_pair(
        tmp_path / "00002.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    metadata = yaml.safe_load(wrong.with_suffix(".yml").read_text())
    metadata["parameters"]["quick_experiment"] = "ResonatorSpectroscopy"
    metadata["independent"].append(["Readout Power", "dB"])
    wrong.with_suffix(".yml").write_text(yaml.safe_dump(metadata))

    found = find_latest_native(
        tmp_path,
        quick_class="ResonatorSpectroscopy",
        axis_text="readout pulse frequency",
    )
    assert found == expected.resolve()
