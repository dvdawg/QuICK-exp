from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.lab import make_held_flux_controller


ROOT = Path(__file__).resolve().parents[1]


class FakeBaseExperiment:
    captured_var = None
    captured_kwargs = None

    def __init__(self, **kwargs):
        FakeBaseExperiment.captured_var = dict(kwargs["var"])
        FakeBaseExperiment.captured_kwargs = dict(kwargs)

    def run(self, **kwargs):
        return self


class FakeSoc:
    def reset_gens(self):
        pass


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


@pytest.mark.parametrize(
    "preset,overrides",
    [
        ("resonator_power", {}),
        (
            "qubit_fine",
            {
                "q_freq": np.linspace(5400.0, 5500.0, 11),
                "q_gain": np.linspace(0.02, 0.8, 9),
            },
        ),
        ("rabi_chevron_duration", {}),
        ("rabi_chevron_amplitude", {}),
        ("ramsey_chevron", {}),
    ],
)
def test_held_z_scalarizes_every_native_2d_sweep(preset, overrides):
    config = repository()
    parameters = config.resolve(preset, overrides=overrides).expanded_parameters()
    # Exercise a readout array too; this reproduces the original r_power crash.
    parameters["r_length"] = np.array([50.0, 10.0, 20.0])
    parameters["z_length"] = 0.37
    parameters["z_settle"] = 6.25
    expected_frequency = np.asarray(parameters["r_freq"]).ravel()
    expected_frequency = expected_frequency[expected_frequency.size // 2]
    expected_power = float(np.min(np.asarray(parameters["r_power"])))

    quick = SimpleNamespace(
        experiment=SimpleNamespace(
            configs={},
            BaseExperiment=FakeBaseExperiment,
        )
    )
    connection = SimpleNamespace(
        quick=quick,
        soc=FakeSoc(),
        soccfg=object(),
    )
    controller = make_held_flux_controller(
        connection,
        config,
        parameters,
    )
    controller.set(0.0)

    held = FakeBaseExperiment.captured_var
    assert held["r_freq"] == expected_frequency
    assert held["r_power"] == expected_power
    assert held["r_length"] == 2.0
    assert held["r_relax"] == 2.0
    assert held["z_length"] == 0.37
    assert held["z_settle"] == 6.25
    assert FakeBaseExperiment.captured_kwargs["hard_avg"] == 1
    assert FakeBaseExperiment.captured_kwargs["soft_avg"] == 1
    for key in ("hard_avg", "soft_avg", "rep"):
        assert key not in held
    for name, value in held.items():
        if value is None or isinstance(value, (str, bytes, dict)):
            continue
        assert np.asarray(value).ndim == 0, f"{name} leaked a sweep array"
