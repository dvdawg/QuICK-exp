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
        "dispersive_spectroscopy",
        "echo",
        "iq_blobs",
        "loopback",
        "qubit_spectroscopy",
        "rabi",
        "rabi_chevron",
        "ramsey",
        "ramsey_chevron",
        "resonator_spectroscopy",
        "t1",
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
    assert duration.axes == ("q_freq", "q_length")
    assert amplitude.axes == ("q_freq", "q_gain")
    assert ramsey.axes == ("q_freq", "time")
