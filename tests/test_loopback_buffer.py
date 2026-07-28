from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quickexp_v3.backend import QuickBackend
from quickexp_v3.config import ConfigRepository
from quickexp_v3.experiments.registry import get


ROOT = Path(__file__).resolve().parents[1]


class FakeLoopBack:
    captured_kwargs = None

    def __init__(self, *, var, soc, soccfg, data_path, title, **kwargs):
        type(self).captured_kwargs = kwargs
        self.var = dict(var)
        self.sweep = {}
        time = np.linspace(0.0, 1.0, 5)
        self.data = np.column_stack(
            (time, np.ones(5), np.zeros(5), np.ones(5), np.zeros(5))
        )

    def run(self, silent=False, dB=False):
        return self


def test_loopback_uses_soft_rounds_without_hard_buffer_repetitions():
    repository = ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )
    plan = get("loopback").build(repository.resolve("loopback"))
    assert plan.variables["soft_avg"] == 100
    assert "hard_avg" not in plan.variables
    assert "rep" not in plan.variables

    quick = SimpleNamespace(
        __version__="0.7.2",
        experiment=SimpleNamespace(LoopBack=FakeLoopBack),
    )
    backend = QuickBackend(soc=object(), soccfg=object(), quick_module=quick)
    backend.acquire(plan)
    assert FakeLoopBack.captured_kwargs == {"soft_avg": 100}
