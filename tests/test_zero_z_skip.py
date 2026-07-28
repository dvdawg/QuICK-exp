from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quickexp_v3.backend import SyntheticBackend
import quickexp_v3.ide as ide


ROOT = Path(__file__).resolve().parents[1]


def test_zero_z_uses_connection_reset_without_running_hold_helper(monkeypatch):
    backend = SyntheticBackend(seed=19)
    connection = SimpleNamespace(backend=backend)
    monkeypatch.setattr(ide, "connect_quick", lambda repository: connection)

    def unexpected_hold_helper(*args, **kwargs):
        raise AssertionError("zero Z must not run the held-Z acquisition")

    monkeypatch.setattr(
        ide,
        "make_held_flux_controller",
        unexpected_hold_helper,
    )
    completed = ide.run_experiment(
        ROOT,
        experiment="resonator_spectroscopy",
        preset="resonator_power",
        title="Punchout_Zp0p0000_r6815p000",
        overrides={
            "r_freq": np.asarray([6814.0, 6815.0, 6816.0]),
            "r_power": np.asarray([-40.0, -35.0]),
            "hard_avg": 1,
            "soft_avg": 1,
        },
        live_hardware=True,
        fixed_z_gain=0.0,
        analyze=False,
        show_plot=False,
    )
    assert completed.status == "completed"
    assert backend.calls == 1
