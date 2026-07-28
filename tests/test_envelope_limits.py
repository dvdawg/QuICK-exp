from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quickexp_v3.backend import (
    QuickBackend,
    validate_quick_envelope_memory,
)
from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.experiments.registry import get
import quickexp_v3.ide as ide
from quickexp_v3.runtime import ExperimentRunner


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def zcu216_soccfg():
    generators = [{} for _ in range(16)]
    generators[1] = {
        "type": "axis_signal_gen_v6",
        "f_fabric": 599.04,
        "samps_per_clk": 16,
        "maxlen": 16384,
    }
    return {"gens": generators}


def test_time_rabi_default_fits_verified_q_generator_memory():
    plan = get("rabi").build(repository().resolve("rabi_length"))

    details = validate_quick_envelope_memory(plan, zcu216_soccfg())

    assert details["required_samples"] == 16000
    assert details["available_samples"] == 16384
    assert details["maximum_single_envelope_us"] == pytest.approx(
        1.7094017094
    )


def test_time_rabi_rejects_first_known_failing_length():
    plan = get("rabi").build(repository().resolve("rabi_length"))
    plan = plan.with_variables(q_length=np.array([1.67, 1.72]))

    with pytest.raises(
        ConfigError,
        match=r"16480 samples.*16384.*1\.720000 us",
    ):
        validate_quick_envelope_memory(plan, zcu216_soccfg())


def test_quick_backend_rejects_unsafe_envelope_before_constructor():
    created = {"value": False}

    class UnexpectedRabi:
        def __init__(self, **kwargs):
            created["value"] = True
            raise AssertionError("unsafe plan must fail before Quick construction")

    plan = get("rabi").build(repository().resolve("rabi_length"))
    plan = plan.with_variables(q_length=np.array([1.67, 1.72]))
    backend = QuickBackend(
        soc=object(),
        soccfg=zcu216_soccfg(),
        quick_module=SimpleNamespace(
            experiment=SimpleNamespace(Rabi=UnexpectedRabi)
        ),
    )

    with pytest.raises(ConfigError, match="Shorten the pulse-length sweep"):
        backend.acquire(plan)
    assert created["value"] is False


def test_config_error_is_not_retried():
    class InvalidBackend:
        def __init__(self):
            self.calls = 0
            self.recoveries = 0

        def connect(self):
            pass

        def acquire(self, plan):
            self.calls += 1
            raise ConfigError("deterministic invalid waveform")

        def recover(self, error, attempt):
            self.recoveries += 1

        def close(self):
            pass

    backend = InvalidBackend()
    runner = ExperimentRunner(repository(), backend)

    with pytest.raises(ConfigError, match="deterministic"):
        runner.run("rabi", "rabi_amplitude")
    assert backend.calls == 1
    assert backend.recoveries == 0


def test_ide_preflight_runs_before_nonzero_held_z(monkeypatch):
    class InvalidBackend:
        def __init__(self):
            self.closed = False

        def validate(self, plan):
            raise ConfigError("unsafe before held Z")

        def close(self):
            self.closed = True

    backend = InvalidBackend()
    connection = SimpleNamespace(backend=backend)
    monkeypatch.setattr(
        ide,
        "connect_quick",
        lambda repository: connection,
    )

    def unexpected_held_z(*args, **kwargs):
        raise AssertionError("held Z must not be established before preflight")

    monkeypatch.setattr(
        ide,
        "make_held_flux_controller",
        unexpected_held_z,
    )

    with pytest.raises(ConfigError, match="unsafe before held Z"):
        ide.run_experiment(
            ROOT,
            experiment="rabi",
            preset="rabi_length",
            overrides={"q_length": np.array([1.67, 1.72])},
            live_hardware=True,
            fixed_z_gain=0.14,
            show_plot=False,
        )
    assert backend.closed is True
