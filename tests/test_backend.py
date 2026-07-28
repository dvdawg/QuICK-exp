from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quickexp_v3.backend import QuickBackend
from quickexp_v3.config import ConfigRepository
from quickexp_v3.experiments.registry import get


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


class FakeSoc:
    def __init__(self):
        self.reset_calls = 0
        self.clear_calls = 0

    def reset_gens(self):
        self.reset_calls += 1

    def clear_interrupts(self):
        self.clear_calls += 1


class FakeRabi:
    captured_kwargs = None
    captured_var = None
    captured_run_arguments = None

    def __init__(self, *, var, soc, soccfg, data_path, title, **kwargs):
        type(self).captured_kwargs = kwargs
        type(self).captured_var = var
        self.var = dict(var)
        self.sweep = {}
        for name, value in kwargs.items():
            if name in self.var:
                self.var[name] = value
                if np.iterable(value) and not isinstance(value, str):
                    self.sweep[name] = value
                    self.var[name] = value[0]
        x = np.asarray(kwargs["q_gain"])
        iq = 0.5 + 0.2 * np.cos(2 * np.pi * x)
        self.data = np.column_stack(
            (x, np.abs(iq), np.angle(iq + 0j), iq, np.zeros_like(iq))
        )

    def run(self, silent=False, population=False):
        type(self).captured_run_arguments = {
            "silent": silent,
            "population": population,
        }
        return self


def test_quick_backend_separates_sweeps_from_mercator_config():
    repo = repository()
    plan = get("rabi").build(repo.resolve("rabi_amplitude"))
    quick = SimpleNamespace(
        __version__="test",
        experiment=SimpleNamespace(Rabi=FakeRabi),
    )
    backend = QuickBackend(soc=FakeSoc(), soccfg=object(), quick_module=quick)
    result = backend.acquire(plan)

    assert list(result.metadata["quick_sweep"]) == ["q_gain"]
    assert "q_gain" in FakeRabi.captured_kwargs
    assert np.asarray(FakeRabi.captured_var["q_gain"]).size == 161
    assert FakeRabi.captured_kwargs["hard_avg"] == 1000
    assert FakeRabi.captured_kwargs["soft_avg"] == 1
    assert FakeRabi.captured_kwargs["rep"] == 1000
    assert FakeRabi.captured_var["z"] == 15
    assert FakeRabi.captured_var["z_gain"] == 0.0
    assert FakeRabi.captured_var["z_length"] == 0.2
    assert FakeRabi.captured_var["z_settle"] == 5.0
    assert FakeRabi.captured_run_arguments["silent"] is False
    assert FakeRabi.captured_run_arguments["population"] is False
    assert result.metadata["native_files"] == []


def test_quick_backend_close_resets_generators_and_interrupts():
    soc = FakeSoc()
    quick = SimpleNamespace(__version__="test", experiment=SimpleNamespace())
    backend = QuickBackend(soc=soc, soccfg=object(), quick_module=quick)
    backend.connect()
    backend.close()
    assert not backend.connected
    assert soc.reset_calls == 1
    assert soc.clear_calls == 1
