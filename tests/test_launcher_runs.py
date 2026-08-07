from dataclasses import replace
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
from quickexp_v3.punchout_fit import punchout_model
from quickexp_v3.qubit_flux_fit import transmon_frequency


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
        1.0
        + 0.003 * (resonator_frequency - 6884.2)
        - 0.7 / (1.0 + ((resonator_frequency - 6884.2) / 0.4) ** 2)
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


def write_native_map(path, outer, inner, signal_rows, *, quick_class, labels, var=None):
    rows = []
    for outer_value, signal in zip(outer, signal_rows):
        iq = np.asarray(signal, dtype=float) * np.exp(0.4j)
        rows.append(
            np.column_stack(
                (
                    np.full_like(inner, outer_value),
                    inner,
                    np.abs(iq),
                    np.angle(iq),
                    iq.real,
                    iq.imag,
                )
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [
                    [labels[0], labels[1]],
                    [labels[2], labels[3]],
                ],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": quick_class,
                    "var": dict(var or {}),
                    "z_gain_sweep": list(map(float, outer)),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def write_extended_fit_fixtures(data_directory):
    powers = np.linspace(-40.0, -5.0, 8)
    readout_frequencies = np.arange(6881.0, 6887.01, 0.1)
    centers = punchout_model(powers, 6884.0, 1.2, -20.0, 4.0)
    punchout_signal = [
        1.0 - 0.8 / (1.0 + ((readout_frequencies - center) / 0.25) ** 2)
        for center in centers
    ]
    punchout = write_native_map(
        data_directory / "00010 - (ResonatorSpectroscopy)Punchout.csv",
        powers,
        readout_frequencies,
        punchout_signal,
        quick_class="ResonatorSpectroscopy",
        labels=("Readout Pulse Power", "dB", "Readout Pulse Frequency", "MHz"),
    )

    z_gain = np.linspace(-0.3, 0.3, 9)
    qubit_frequencies = np.linspace(3300.0, 4800.0, 751)
    ridge = transmon_frequency(
        z_gain,
        f_max_mhz=4763.0,
        period_z=0.30,
        sweet_spot_z=0.0,
        asymmetry=0.19,
        ec_mhz=180.0,
    )
    qubit_signal = [
        0.1 + 0.9 / (1.0 + ((qubit_frequencies - center) / 5.0) ** 2)
        for center in ridge
    ]
    qubit_flux = write_native_map(
        data_directory / "00011 - QubitSpecVsZ_fitted_readout.csv",
        z_gain,
        qubit_frequencies,
        qubit_signal,
        quick_class="QubitSpectroscopy",
        labels=("Z Gain", "a.u.", "Qubit Frequency", "MHz"),
        var={"q_delta": -180.0},
    )

    q_gain = np.linspace(0.02, 0.4, 9)
    gain_frequencies = np.linspace(4550.0, 4650.0, 201)
    gain_ridge = 4590.0 + 35.0 * q_gain
    gain_signal = [
        0.1 + 0.9 / (1.0 + ((gain_frequencies - center) / 3.0) ** 2)
        for center in gain_ridge
    ]
    qubit_gain = write_native_map(
        data_directory / "00011b - QubitSpecVsQGain.csv",
        q_gain,
        gain_frequencies,
        gain_signal,
        quick_class="QubitSpectroscopy",
        labels=(
            "Qubit Pulse Gain",
            "a.u.",
            "Qubit Pulse Frequency",
            "MHz",
        ),
        var={"z_gain": -0.18},
    )

    drive_frequencies = np.linspace(5598.0, 5602.0, 9)
    time = np.linspace(0.0, 3.0, 121)
    rabi_rates = np.sqrt(1.2**2 + (drive_frequencies - 5600.0) ** 2)
    rabi_signal = [
        0.1 + 0.8 * np.exp(-time / 8.0) * np.cos(2 * np.pi * rate * time + 0.2)
        for rate in rabi_rates
    ]
    rabi_chevron = write_native_map(
        data_directory / "00012 - (Rabi)chevron.csv",
        drive_frequencies,
        time,
        rabi_signal,
        quick_class="Rabi",
        labels=("Qubit Pulse Frequency", "MHz", "Qubit Pulse Length", "us"),
    )

    fringe = 1.0
    ramsey_rates = np.maximum(
        np.abs(fringe + (5600.0 - drive_frequencies)),
        0.35,
    )
    ramsey_signal = [
        0.1 + 0.8 * np.exp(-time / 8.0) * np.cos(2 * np.pi * rate * time + 0.2)
        for rate in ramsey_rates
    ]
    ramsey_chevron = write_native_map(
        data_directory / "00013 - (T2Ramsey)chevron.csv",
        drive_frequencies,
        time,
        ramsey_signal,
        quick_class="T2Ramsey",
        labels=("Qubit Pulse Frequency", "MHz", "Delay Time", "us"),
        var={"fringe_freq": fringe, "q_freq": 5600.0},
    )

    rng = np.random.default_rng(4)
    ground = rng.normal(-0.5, 0.1, (200, 2))
    excited = rng.normal(0.5, 0.1, (200, 2))
    iq_path = data_directory / "00014 - (IQScatter)blobs.csv"
    np.savetxt(
        iq_path,
        np.column_stack(
            (ground[:, 0], ground[:, 1], excited[:, 0], excited[:, 1])
        ),
        delimiter=",",
    )
    iq_path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [],
                "dependent": [
                    ["I Ground", ""],
                    ["Q Ground", ""],
                    ["I Excited", ""],
                    ["Q Excited", ""],
                ],
                "parameters": {"quick_experiment": "IQScatter", "var": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dispersive_frequency = np.linspace(6878.0, 6888.0, 101)
    ground_trace = 1.0 - 0.3 / (
        1.0 + 2j * (dispersive_frequency - 6883.0) / 1.2
    )
    excited_trace = 1.0 - 0.3 / (
        1.0 + 2j * (dispersive_frequency - 6884.0) / 1.2
    )
    dispersive = data_directory / "00015 - (DispersiveSpectroscopy)test.csv"
    np.savetxt(
        dispersive,
        np.column_stack(
            (
                dispersive_frequency,
                np.abs(ground_trace),
                np.angle(ground_trace),
                ground_trace.real,
                ground_trace.imag,
                np.abs(excited_trace),
                np.angle(excited_trace),
                excited_trace.real,
                excited_trace.imag,
            )
        ),
        delimiter=",",
    )
    dispersive.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [["Readout Pulse Frequency", "MHz"]],
                "dependent": [
                    ["Amplitude 0", ""],
                    ["Phase 0", "rad"],
                    ["I 0", ""],
                    ["Q 0", ""],
                    ["Amplitude 1", ""],
                    ["Phase 1", "rad"],
                    ["I 1", ""],
                    ["Q 1", ""],
                ],
                "parameters": {
                    "quick_experiment": "DispersiveSpectroscopy",
                    "var": {},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    echo_time = np.linspace(0.0, 20.0, 101)
    echo = write_pair(
        data_directory / "00016 - (T2Echo)test.csv",
        quick_class="T2Echo",
        axis_label="Delay Time",
        axis_unit="us",
        x=echo_time,
        signal=0.1 + 0.8 * np.exp(-((echo_time / 5.0) ** 1.5)),
        var={"cycle": 1},
    )
    return {
        "05f_fit_punchout.py": punchout,
        "06e_fit_qubit_vs_flux.py": qubit_flux,
        "06g_design_qubit_sweep_path.py": qubit_flux,
        "07c_fit_rabi_chevron.py": rabi_chevron,
        "09b_fit_iq_blobs.py": iq_path,
        "10b_fit_readout_optimization.py": dispersive,
        "12b_fit_ramsey_chevron.py": ramsey_chevron,
        "14b_fit_echo.py": echo,
    }


def test_every_numbered_file_runs_to_completion_offline(tmp_path):
    prepare_project(tmp_path)
    native_fit_fixtures = write_native_fit_fixtures(tmp_path / "data")
    extended_fit_fixtures = write_extended_fit_fixtures(tmp_path / "data")
    for filename in EXPECTED:
        namespace = runpy.run_path(
            str(LAUNCHER_DIR / filename),
            run_name=f"offline_{filename[:-3]}",
        )
        module_globals = namespace["main"].__globals__
        module_globals["PROJECT_ROOT"] = tmp_path
        module_globals["LIVE_HARDWARE"] = False
        if filename == "01_configure_experiment.py":
            # The operator latch may legitimately be True in a live
            # workspace; executing main() here must never write the YAML.
            module_globals["WRITE_CHANGES"] = False
        if filename == "90_measurement_queue.py":
            module_globals["PROJECT_ROOT"] = ROOT
        if filename == "91_autocal.py":
            module_globals["TARGET"] = "coherence_only"
        if "SHOW_PLOT" in module_globals:
            module_globals["SHOW_PLOT"] = False
        if "SHOTS" in module_globals:
            module_globals["SHOTS"] = 40
        if "SWEEP_PATH_YML" in module_globals:
            module_globals["SWEEP_PATH_YML"] = None
        if filename in native_fit_fixtures:
            module_globals["INPUT_CSV"] = native_fit_fixtures[filename]
            module_globals["WRITE_ACCEPTED_FIT"] = False
        if filename in {
            "06e_fit_qubit_vs_flux.py",
            "06g_design_qubit_sweep_path.py",
        }:
            module_globals["FIT_FREQUENCY_WINDOW_MHZ"] = None
        if filename == "06e_fit_qubit_vs_flux.py":
            module_globals["FIT_FLUX_WINDOW_Z"] = None
        if filename == "06g_design_qubit_sweep_path.py":
            module_globals["PATH_METHOD"] = "fit_margin"
            module_globals["OUTPUT_YML"] = tmp_path / "analysis_cache/path.yml"
            module_globals["SHOW_PREVIEW"] = False
        if filename in extended_fit_fixtures:
            module_globals["INPUT_CSV"] = extended_fit_fixtures[filename]
            if "WRITE_ACCEPTED_FIT" in module_globals:
                module_globals["WRITE_ACCEPTED_FIT"] = False
        if filename == "14b_fit_echo.py":
            module_globals["BOOTSTRAP_RESAMPLES"] = 10
        if filename == "17b_fit_flux_iir.py":
            module_globals["USE_SYNTHETIC_DEMO"] = True
            module_globals["WRITE_CANDIDATE"] = False
            module_globals["MODEL_ORDERS"] = (1, 2, 3, 4)
            module_globals["MULTISTARTS"] = 3
        if filename == "17c_cryoscope.py":
            module_globals["SCHEDULE_JSON"] = (
                tmp_path / "analysis_cache/cryoscope_schedule.json"
            )
            module_globals["HARD_AVG"] = 32
        if filename == "17d_fit_flux_fir.py":
            module_globals["USE_SYNTHETIC_DEMO"] = True
            module_globals["WRITE_CANDIDATE"] = False
            module_globals["FIR_COEFFICIENT_COUNT"] = 12
            module_globals["INVERSE_FIR_LENGTH"] = 12
            module_globals["MAXIMUM_EVALUATIONS"] = 200
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
        if filename == "17c_cryoscope.py":
            module_globals["RAMSEY_PHASE_DEG"] = np.asarray(
                [0.0, 90.0, 180.0, 270.0]
            )
        if filename == "17a_flux_step_spectroscopy.py":
            parameters = module_globals["PARAMETERS"]
            module_globals["PARAMETERS"] = replace(
                parameters,
                live_hardware=False,
                probe_times_us=np.asarray(parameters.probe_times_us)[:3],
                q_frequency_offsets_mhz=np.asarray(
                    parameters.q_frequency_offsets_mhz
                )[:8],
                q_frequency_centers_mhz=5200.0,
                show_plot=False,
            )
        if filename == "18a_resonator_flux_transient.py":
            parameters = module_globals["PARAMETERS"]
            module_globals["PARAMETERS"] = replace(
                parameters,
                live_hardware=False,
                probe_times_us=np.geomspace(1.0, 100.0, 4),
                reference_probe_points=9,
                show_plot=False,
            )

        result = namespace["main"]()
        if filename == "00_connect_and_ports.py":
            assert result is None
        elif filename == "01_configure_experiment.py":
            assert result is False
        elif filename == "05d_fit_resonator_vs_flux.py":
            assert result.statistics["r_squared"] > 0.95
        elif filename == "08c_fit_rabi.py":
            assert result.statistics["r_squared"] > 0.95
        elif filename == "05e_fit_resonator_spectroscopy.py":
            assert (
                result.statistics.get(
                    "r_squared_complex",
                    result.statistics.get("r_squared"),
                )
                > 0.95
            )
        elif filename in native_fit_fixtures:
            assert result.statistics["r_squared"] > 0.95
        elif filename in extended_fit_fixtures:
            assert result is not None
        elif filename == "17b_fit_flux_iir.py":
            assert result.model_order == 3
            assert result.statistics["r_squared"] > 0.999
        elif filename == "17d_fit_flux_fir.py":
            assert result.statistics["r_squared"] > 0.999
        elif filename == "95_device_report.py":
            assert result.markdown_path.is_file()
        elif filename == "92_review_proposals.py":
            assert isinstance(result, list)
        elif filename == "91_autocal.py":
            assert result.status in {"completed", "completed_with_escalations"}
        elif filename == "18a_resonator_flux_transient.py":
            # Offline the synthetic backend writes no native CSV, so the
            # launcher acquires and then stops before the inversion.
            campaign, trace, fit, inverse = result
            assert campaign.transient_result is not None
            assert (trace, fit, inverse) == (None, None, None)
        elif isinstance(result, list):
            assert len(result) == 3
            assert all(row.status.startswith("completed") for row in result)
        else:
            assert result.status.startswith("completed")
        plt.close("all")
