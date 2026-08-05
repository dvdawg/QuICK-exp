from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quickexp_v3.backend import (
    QuickBackend,
    register_program_variables,
)
from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.experiments.registry import get
from quickexp_v3.lab import install_authored_programs
from quickexp_v3.programs import (
    CRYOSCOPE,
    FLUX_STEP_SPECTROSCOPY,
    T1_ZPA,
    TWO_TONE_ZPA,
)
from quickexp_v3.synthetic_device import DeviceModel
from quickexp_v3.backend import SyntheticBackend


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def soccfg():
    generators = [
        {
            "f_fabric": 599.04,
            "samps_per_clk": 16,
            "maxlen": 16384,
        }
        for _ in range(16)
    ]
    return {"gens": generators}


def test_authored_program_plans_preflight_and_decode_synthetic_maps():
    repo = repository()
    spectroscopy = get("two_tone_zpa")
    spectroscopy_plan = spectroscopy.build(repo.resolve("two_tone_zpa"))
    assert set(spectroscopy_plan.axes) == {"q_freq", "z_gain"}
    assert spectroscopy_plan.metadata["preflight"] == TWO_TONE_ZPA.preflight
    report = TWO_TONE_ZPA.preflight(soccfg(), spectroscopy_plan.variables)
    assert report.ok
    assert report.details["envelopes"]["q_length"]["required_samples"] == 3824

    device = DeviceModel()
    backend = SyntheticBackend(seed=11, device=device)
    acquired = backend.acquire(spectroscopy_plan)
    decoded = spectroscopy.decode(spectroscopy_plan, acquired)
    assert decoded.points == 101 * 21
    assert np.ptp(acquired.metadata["device_model"]["qubit_center_mhz"]) > 100

    t1 = get("t1_zpa")
    t1_plan = t1.build(repo.resolve("t1_zpa"))
    assert set(t1_plan.axes) == {"z_gain", "time"}
    assert T1_ZPA.preflight(soccfg(), t1_plan.variables).ok
    t1_data = t1.decode(t1_plan, backend.acquire(t1_plan))
    assert t1_data.points == 9 * 61

    step = get("flux_step_spectroscopy")
    step_plan = step.build(repo.resolve("flux_step_spectroscopy"))
    assert FLUX_STEP_SPECTROSCOPY.preflight(soccfg(), step_plan.variables).ok
    step_data = step.decode(step_plan, backend.acquire(step_plan))
    assert step_data.points == 13 * 201

    cryoscope = get("cryoscope")
    cryoscope_plan = cryoscope.build(repo.resolve("cryoscope"))
    assert CRYOSCOPE.preflight(soccfg(), cryoscope_plan.variables).ok
    cryoscope_data = cryoscope.decode(
        cryoscope_plan,
        backend.acquire(cryoscope_plan),
    )
    assert cryoscope_data.points == 43 * 16


def test_two_tone_zpa_accepts_one_frequency_row_at_scalar_z():
    repo = repository()
    spectroscopy = get("two_tone_zpa")
    plan = spectroscopy.build(
        repo.resolve(
            "two_tone_zpa",
            overrides={
                "z_gain": 0.125,
                "q_freq": np.linspace(4750.0, 4770.0, 21),
            },
        )
    )

    assert plan.axes == ("q_freq",)
    assert plan.variables["z_gain"] == 0.125
    decoded = spectroscopy.decode(plan, SyntheticBackend(seed=12).acquire(plan))
    assert decoded.points == 21


def test_program_registration_rejects_config_override_collisions():
    with pytest.raises(ConfigError, match="collide.*rep"):
        register_program_variables("BadAuthoredProgram", {"q_freq", "rep"})


def test_install_all_authored_programs_is_idempotent():
    class BaseExperiment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return self

    quick = SimpleNamespace(
        experiment=SimpleNamespace(
            configs={},
            BaseExperiment=BaseExperiment,
        )
    )
    first = install_authored_programs(
        quick,
        channel_variables={"r": 4, "rr": 2, "q": 5, "z": 2},
    )
    second = install_authored_programs(quick)

    assert set(first) == {
        "Cryoscope",
        "FluxStepSpectroscopy",
        "TwoTone_ZPA",
        "T1_zpa",
    }
    assert first == second
    instance = first["TwoTone_ZPA"]()
    assert instance.var["r"] == 4
    assert instance.var["rr"] == 2
    assert instance.var["q"] == 5


def test_quick_backend_routes_authored_sweeps_and_config_separately():
    repo = repository()
    plan = get("two_tone_zpa").build(repo.resolve("two_tone_zpa"))

    class FakeAuthored:
        captured_var = None
        captured_kwargs = None

        def __init__(self, *, var, soc, soccfg, data_path, title, **kwargs):
            type(self).captured_var = dict(var)
            type(self).captured_kwargs = dict(kwargs)
            self.var = dict(var)
            self.sweep = {
                name: value
                for name, value in kwargs.items()
                if name in var and np.asarray(value).size > 1
            }
            axes = [np.asarray(self.sweep[name]) for name in self.sweep]
            grids = [
                value.ravel()
                for value in np.meshgrid(*axes, indexing="ij")
            ]
            iq = np.full(grids[0].size, 0.5 + 0.1j)
            self.data = np.column_stack(
                (*grids, np.abs(iq), np.angle(iq), iq.real, iq.imag)
            )

        def run(self, silent=False, dB=False, population=False):
            return self

    quick = SimpleNamespace(
        __version__="test",
        experiment=SimpleNamespace(TwoTone_ZPA=FakeAuthored),
    )
    result = QuickBackend(
        soc=object(),
        soccfg=soccfg(),
        quick_module=quick,
    ).acquire(plan)

    assert tuple(result.metadata["quick_sweep"]) == plan.axes
    assert set(plan.axes) <= set(FakeAuthored.captured_kwargs)
    assert FakeAuthored.captured_kwargs["hard_avg"] == 1000
    assert FakeAuthored.captured_kwargs["rep"] == 0
    assert "hard_avg" not in FakeAuthored.captured_var
    assert "rep" not in FakeAuthored.captured_var
    assert result.payload.shape[1] == len(plan.axes) + 4
