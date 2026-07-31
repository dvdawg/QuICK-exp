from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import AnalysisError
from quickexp_v3.rabi_fit import (
    accept_rabi_fit,
    find_latest_rabi,
    fit_rabi,
)


ROOT = Path(__file__).resolve().parents[1]


def write_rabi_pair(
    directory: Path,
    *,
    stem: str = "00001 - (Rabi)test",
    variable: str = "q_length",
) -> Path:
    x = np.linspace(0.0, 1.0, 101)
    frequency = 2.0
    signal = (
        0.1
        + 0.8 * np.exp(-x / 3.0) * np.cos(2 * np.pi * frequency * x)
        + 0.01 * x
    )
    angle = 0.63
    i = signal * np.cos(angle)
    q = signal * np.sin(angle)
    amplitude = np.hypot(i, q)
    phase = np.angle(i + 1j * q)
    csv_path = directory / f"{stem}.csv"
    np.savetxt(
        csv_path,
        np.column_stack((x, amplitude, phase, i, q)),
        delimiter=",",
    )
    label, unit = (
        ("Qubit Pulse Length", "us")
        if variable == "q_length"
        else ("Qubit Pulse Gain", "a.u.")
    )
    metadata = {
        "independent": [[label, unit]],
        "dependent": [
            ["Amplitude", ""],
            ["Phase", "rad"],
            ["I", ""],
            ["Q", ""],
        ],
        "parameters": {
            "quick_experiment": "Rabi",
            "var": {
                "q_freq": 3939.5,
                "q_gain": 0.2,
                "q_length": 0.115,
                "z_gain": 0.14,
            },
        },
    }
    csv_path.with_suffix(".yml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    return csv_path


@pytest.mark.parametrize("variable", ("q_length", "q_gain"))
def test_native_rabi_fit_finds_phase_corrected_pi_value(tmp_path, variable):
    source = write_rabi_pair(tmp_path, variable=variable)

    fit = fit_rabi(source, variable=variable)

    assert fit.variable == variable
    assert fit.pi_value == pytest.approx(0.25, abs=0.003)
    assert fit.parameters["half_period"] == pytest.approx(0.25, abs=0.003)
    assert fit.statistics["r_squared"] > 0.999
    assert fit.passes(
        minimum_r_squared=0.95,
        minimum_oscillations=1.0,
        maximum_relative_pi_uncertainty=0.1,
    )


def test_find_latest_rabi_filters_by_native_axis(tmp_path):
    length = write_rabi_pair(
        tmp_path,
        stem="00001 - (Rabi)time",
        variable="q_length",
    )
    gain = write_rabi_pair(
        tmp_path,
        stem="00002 - (Rabi)power",
        variable="q_gain",
    )

    assert find_latest_rabi(tmp_path, variable="q_length") == length.resolve()
    assert find_latest_rabi(tmp_path, variable="q_gain") == gain.resolve()


def test_accept_rabi_fit_updates_calibration_atomically(tmp_path):
    for name in (
        "hardware.example.yml",
        "presets.example.yml",
        "calibration.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    fit = fit_rabi(write_rabi_pair(tmp_path), variable="q_length")

    target = accept_rabi_fit(
        tmp_path,
        fit,
        minimum_r_squared=0.95,
        minimum_oscillations=1.0,
        maximum_relative_pi_uncertainty=0.1,
    )

    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    record = document["records"]["defaults"]["q_length"]
    assert record["status"] == "accepted"
    assert record["value"] == pytest.approx(fit.pi_value)
    assert record["valid_domain"]["z_gain"] == [0.14, 0.14]
    repository = ConfigRepository.from_files(
        tmp_path / "hardware.example.yml",
        target,
        tmp_path / "presets.example.yml",
    )
    parameters = repository.resolve("t1").expanded_parameters()
    assert parameters["q_length"] == pytest.approx(fit.pi_value)


def test_force_write_bypasses_fit_gates_and_is_auditable(tmp_path):
    for name in (
        "hardware.example.yml",
        "presets.example.yml",
        "calibration.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    fit = fit_rabi(write_rabi_pair(tmp_path), variable="q_length")
    thresholds = {
        "minimum_r_squared": 1.1,
        "minimum_oscillations": 1.0,
        "maximum_relative_pi_uncertainty": 0.1,
    }

    with pytest.raises(AnalysisError, match="not accepted"):
        accept_rabi_fit(tmp_path, fit, **thresholds)

    target = accept_rabi_fit(
        tmp_path,
        fit,
        **thresholds,
        force_write=True,
    )

    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    quality = document["records"]["defaults"]["q_length"]["quality"]
    assert quality["force_written"] is True
    assert quality["acceptance_gates_passed"] is False
