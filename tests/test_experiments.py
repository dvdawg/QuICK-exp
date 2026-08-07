from pathlib import Path

import numpy as np

from quickexp_v3.config import ConfigRepository
from quickexp_v3.experiments.registry import get, names


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def test_registry_has_explicit_standard_and_chevron_adapters():
    assert names() == [
        "cryoscope",
        "dispersive_spectroscopy",
        "echo",
        "flux_step_spectroscopy",
        "iq_blobs",
        "loopback",
        "qubit_spectroscopy",
        "rabi",
        "rabi_chevron",
        "ramsey",
        "ramsey_chevron",
        "resonator_flux_transient",
        "resonator_spectroscopy",
        "t1",
        "t1_zpa",
        "two_tone_zpa",
    ]


def test_t1_translates_delay_to_quick_time():
    plan = get("t1").build(repository().resolve("t1"))
    assert "delay" not in plan.variables
    assert isinstance(plan.variables["time"], np.ndarray)
    assert plan.axes == ("time",)


def test_ramsey_translates_local_quick_names():
    plan = get("ramsey").build(repository().resolve("ramsey"))
    assert "delay" not in plan.variables
    assert "fringe_frequency_mhz" not in plan.variables
    assert plan.variables["fringe_freq"] == 5.0


def test_echo_translates_extra_cycles():
    plan = get("echo").build(repository().resolve("cpmg_4"))
    assert plan.variables["cycle"] == 3
    assert "pulse_count" not in plan.variables


def test_rabi_plan_has_exactly_one_sweep():
    repo = repository()
    amplitude = get("rabi").build(repo.resolve("rabi_amplitude"))
    length = get("rabi").build(repo.resolve("rabi_length"))
    assert amplitude.axes == ("q_gain",)
    assert length.axes == ("q_length",)


def test_chevrons_have_two_native_quick_axes():
    repo = repository()
    duration = get("rabi_chevron").build(repo.resolve("rabi_chevron_duration"))
    amplitude = get("rabi_chevron").build(repo.resolve("rabi_chevron_amplitude"))
    ramsey = get("ramsey_chevron").build(repo.resolve("ramsey_chevron"))
    assert duration.axes == ("q_length", "q_freq")
    assert amplitude.axes == ("q_gain", "q_freq")
    assert ramsey.axes == ("time", "q_freq")


def test_two_dimensional_sweeps_step_frequency_innermost():
    # Quick registers the first constructor sweep as the outer loop. Frequency
    # must therefore be declared last so each slow-axis point is held while a
    # full spectrum is taken, not the reverse.
    repo = repository()
    punchout = get("resonator_spectroscopy").build(repo.resolve("resonator_power"))
    gain_scan = get("qubit_spectroscopy").build(
        repo.resolve(
            "qubit_fine",
            overrides={"q_gain": {"start": 0.02, "stop": 1.0, "points": 9}},
        )
    )
    flux_scan = get("two_tone_zpa").build(repo.resolve("two_tone_zpa"))
    assert punchout.axes == ("r_power", "r_freq")
    assert gain_scan.axes == ("q_gain", "q_freq")
    assert flux_scan.axes == ("z_gain", "q_freq")


def test_flux_and_delay_maps_hold_the_bias_on_the_outer_axis():
    plan = get("t1_zpa").build(repository().resolve("t1_zpa"))
    assert plan.axes == ("z_gain", "time")


def test_flux_compensation_maps_put_probe_axis_innermost():
    repo = repository()
    step = get("flux_step_spectroscopy").build(
        repo.resolve("flux_step_spectroscopy")
    )
    cryoscope = get("cryoscope").build(repo.resolve("cryoscope"))
    assert step.axes == ("probe_time", "q_freq")
    assert cryoscope.axes == ("flux_time", "ramsey_phase")
