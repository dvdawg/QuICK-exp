from pathlib import Path
import runpy
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from test_launchers import EXPECTED
from test_native_fit import write_pair


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = ROOT / "experiments"


def prepare_project(tmp_path):
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, tmp_path / name)
    (tmp_path / "data").mkdir()


def write_native_fit_fixtures(data_directory):
    loopback_time = np.linspace(0.0, 2.0, 1001)
    loopback_signal = 0.05 + 0.6 * (
        1.0 + np.tanh((loopback_time - 0.49) / (2.0 * 0.008))
    )
    loopback = write_pair(
        data_directory / "00001 - (LoopBack)test.csv",
        quick_class="LoopBack",
        axis_label="Time",
        axis_unit="us",
        x=loopback_time,
        signal=loopback_signal,
        var={"r_offset": 0.0},
    )

    resonator_frequency = np.linspace(6879.2, 6889.2, 201)
    resonator_signal = (
        0.2
        + 0.003 * (resonator_frequency - 6884.2)
        - 0.8 / (1.0 + ((resonator_frequency - 6884.2) / 0.4) ** 2)
    )
    resonator = write_pair(
        data_directory / "00002 - (ResonatorSpectroscopy)test.csv",
        quick_class="ResonatorSpectroscopy",
        axis_label="Readout Pulse Frequency",
        axis_unit="MHz",
        x=resonator_frequency,
        signal=resonator_signal,
        var={"z_gain": 0.0},
    )

    qubit_frequency = np.linspace(5598.9, 5608.9, 201)
    qubit_signal = (
        0.1
        - 0.002 * (qubit_frequency - 5603.9)
        + 0.7 / (1.0 + ((qubit_frequency - 5603.9) / 0.45) ** 2)
    )
    qubit = write_pair(
        data_directory / "00003 - (QubitSpectroscopy)test.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=qubit_frequency,
        signal=qubit_signal,
        var={"z_gain": 0.0},
    )

    delay = np.linspace(0.0, 30.0, 301)
    t1 = write_pair(
        data_directory / "00004 - (T1)test.csv",
        quick_class="T1",
        axis_label="Delay Time",
        axis_unit="us",
        x=delay,
        signal=0.05 + 0.8 * np.exp(-delay / 6.2),
    )

    ramsey_time = np.linspace(0.0, 5.0, 501)
    ramsey = write_pair(
        data_directory / "00005 - (T2Ramsey)test.csv",
        quick_class="T2Ramsey",
        axis_label="Delay Time",
        axis_unit="us",
        x=ramsey_time,
        signal=(
            0.05
            + 0.8
            * np.exp(-ramsey_time / 1.8)
            * np.cos(2.0 * np.pi * 5.2 * ramsey_time + 0.3)
        ),
        var={"q_freq": 5600.0, "fringe_freq": 5.0},
    )
    return {
        "02b_fit_loopback.py": loopback,
        "05e_fit_resonator_spectroscopy.py": resonator,
        "06d_fit_qubit_spectroscopy.py": qubit,
        "11b_fit_t1.py": t1,
        "13b_fit_ramsey.py": ramsey,
    }


def write_flux_scan(path):
    z_values = np.linspace(-0.4, 0.4, 21)
    frequencies = np.arange(6883.0, 6885.41, 0.04)
    rows = []
    for z_gain in z_values:
        center = 6884.2 + 0.7 * np.cos(
            2 * np.pi * (z_gain + 0.06) / 0.23
        )
        amplitude = -10 * np.exp(-0.5 * ((frequencies - center) / 0.1) ** 2)
        rows.append(
            np.column_stack(
                [
                    np.full_like(frequencies, z_gain),
                    frequencies,
                    amplitude,
                    np.zeros_like(frequencies),
                    10 ** (amplitude / 20),
                    np.zeros_like(frequencies),
                ]
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")


def write_rabi_scan(path):
    x = np.linspace(0.0, 1.0, 101)
    signal = 0.1 + 0.8 * np.exp(-x / 3.0) * np.cos(4 * np.pi * x)
    i = signal * np.cos(0.4)
    q = signal * np.sin(0.4)
    np.savetxt(
        path,
        np.column_stack((x, np.hypot(i, q), np.angle(i + 1j * q), i, q)),
        delimiter=",",
    )
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [["Qubit Pulse Length", "us"]],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": "Rabi",
                    "var": {
                        "q_length": 0.0,
                        "q_gain": 0.2,
                        "z_gain": 0.0,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_every_numbered_file_runs_to_completion_offline(tmp_path):
    prepare_project(tmp_path)
    native_fit_fixtures = write_native_fit_fixtures(tmp_path / "data")
    for filename in EXPECTED:
        namespace = runpy.run_path(
            str(LAUNCHER_DIR / filename),
            run_name=f"offline_{filename[:-3]}",
        )
        module_globals = namespace["main"].__globals__
        module_globals["PROJECT_ROOT"] = tmp_path
        module_globals["LIVE_HARDWARE"] = False
        if filename == "90_measurement_queue.py":
            module_globals["PROJECT_ROOT"] = ROOT
        if "SHOW_PLOT" in module_globals:
            module_globals["SHOW_PLOT"] = False
        if "SHOTS" in module_globals:
            module_globals["SHOTS"] = 40
        if filename in native_fit_fixtures:
            module_globals["INPUT_CSV"] = native_fit_fixtures[filename]
            module_globals["WRITE_ACCEPTED_FIT"] = False
        if filename == "05d_fit_resonator_vs_flux.py":
            scan_path = tmp_path / "00001 - ResVsZ_held_bias.csv"
            write_flux_scan(scan_path)
            module_globals["INPUT_CSV"] = scan_path
            module_globals["WRITE_ACCEPTED_FIT"] = False
        if filename == "08c_fit_rabi.py":
            scan_path = tmp_path / "00002 - (Rabi)test.csv"
            write_rabi_scan(scan_path)
            module_globals["INPUT_CSV"] = scan_path
            module_globals["WRITE_ACCEPTED_FIT"] = False
        for name, value in list(module_globals.items()):
            if isinstance(value, np.ndarray) and value.ndim == 1 and value.size > 3:
                module_globals[name] = value[:3]

        result = namespace["main"]()
        if filename == "00_connect_and_ports.py":
            assert result is None
        elif filename == "01_configure_experiment.py":
            assert result is False
        elif filename == "05d_fit_resonator_vs_flux.py":
            assert result.statistics["r_squared"] > 0.95
        elif filename == "08c_fit_rabi.py":
            assert result.statistics["r_squared"] > 0.95
        elif filename in native_fit_fixtures:
            assert result.statistics["r_squared"] > 0.95
        elif isinstance(result, list):
            assert len(result) == 3
            assert all(row.status.startswith("completed") for row in result)
        else:
            assert result.status.startswith("completed")
        plt.close("all")
