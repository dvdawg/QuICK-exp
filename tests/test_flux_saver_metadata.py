from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quickexp_v3.backend import SyntheticBackend
import quickexp_v3.ide as ide


ROOT = Path(__file__).resolve().parents[1]


class FakeSaver:
    instance = None

    def __init__(
        self,
        title,
        data_path,
        *,
        indep_params,
        dep_params,
        params,
    ):
        self.title = title
        self.data_path = data_path
        self.indep_params = indep_params
        self.dep_params = dep_params
        self.params = params
        self.file_name = str(Path(data_path) / title)
        self.rows = []
        self.has_data = False
        self.wrote_yml = False
        FakeSaver.instance = self

    def write_data(self, rows):
        self.rows.append(np.asarray(rows))
        self.has_data = True

    def write_yml(self):
        self.wrote_yml = True


class FakeFlux:
    def __init__(self):
        self.values = []
        self.parked = False
        self.safe_for_acquisition = False

    def set(self, value):
        self.values.append(float(value))
        self.safe_for_acquisition = True

    def park(self):
        self.parked = True


def test_flux_saver_records_all_fitted_readouts_and_provenance(
    tmp_path,
    monkeypatch,
):
    backend = SyntheticBackend(seed=22)
    backend.data_path = str(tmp_path)
    flux = FakeFlux()

    def sweep(config, sweep_config, progressBar):
        assert progressBar is True
        return [
            {"z_gain": float(value)}
            for value in sweep_config["z_gain"]
        ]

    connection = SimpleNamespace(
        backend=backend,
        quick=SimpleNamespace(Saver=FakeSaver, Sweep=sweep),
    )
    monkeypatch.setattr(ide, "connect_quick", lambda repository: connection)
    monkeypatch.setattr(
        ide,
        "make_held_flux_controller",
        lambda *args, **kwargs: flux,
    )

    z_gain = np.array([-0.1, 0.0, 0.1])
    readout_metadata = {
        "model": "cosine",
        "provenance": {"source": "test ResVsZ CSV"},
    }
    callbacks = []
    rows = ide.run_flux_sweep(
        ROOT,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        flux_values=z_gain,
        readout_frequency=lambda z: 6884.0 + z,
        readout_metadata=readout_metadata,
        overrides={
            "q_freq": np.array([5500.0, 5501.0, 5502.0]),
            "hard_avg": 1,
            "soft_avg": 1,
        },
        live_hardware=True,
        show_plot=False,
        before_row=lambda index, value: callbacks.append(
            ("before", index, value)
        ),
        after_row=lambda index, value, completed: callbacks.append(
            ("after", index, value, completed.status)
        ),
    )

    saver = FakeSaver.instance
    assert saver.params["r_freq_by_z"] == [6883.9, 6884.0, 6884.1]
    assert saver.params["readout_frequency_calibration"] == readout_metadata
    assert saver.wrote_yml is True
    assert len(saver.rows) == len(z_gain)
    assert [row[0, 2] for row in saver.rows] == [
        6883.9,
        6884.0,
        6884.1,
    ]
    assert len(rows) == len(z_gain)
    assert flux.values == [-0.1, 0.0, 0.1]
    assert flux.parked is True
    assert callbacks == [
        ("before", 0, -0.1),
        ("after", 0, -0.1, "completed"),
        ("before", 1, 0.0),
        ("after", 1, 0.0, "completed"),
        ("before", 2, 0.1),
        ("after", 2, 0.1, "completed"),
    ]


def test_row_overrides_move_the_inner_sweep_per_z(tmp_path, monkeypatch):
    backend = SyntheticBackend(seed=22)
    backend.data_path = str(tmp_path)
    flux = FakeFlux()

    def sweep(config, sweep_config, progressBar):
        return [
            {"z_gain": float(value)}
            for value in sweep_config["z_gain"]
        ]

    connection = SimpleNamespace(
        backend=backend,
        quick=SimpleNamespace(Saver=FakeSaver, Sweep=sweep),
    )
    monkeypatch.setattr(ide, "connect_quick", lambda repository: connection)
    monkeypatch.setattr(
        ide,
        "make_held_flux_controller",
        lambda *args, **kwargs: flux,
    )

    z_gain = np.array([-0.1, 0.0, 0.1])
    rows = ide.run_flux_sweep(
        ROOT,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        flux_values=z_gain,
        readout_frequency=lambda z: 6884.0 + z,
        row_overrides=lambda z: {
            "q_freq": np.array([5500.0 + z, 5501.0 + z, 5502.0 + z]),
            "p1_nqz": 1,
        },
        row_override_metadata={"model": "linear_interpolation"},
        overrides={"hard_avg": 1, "soft_avg": 1},
        live_hardware=True,
        show_plot=False,
    )

    saver = FakeSaver.instance
    assert saver.params["row_override_calibration"] == {
        "model": "linear_interpolation"
    }
    assert saver.params["row_overrides_by_z"][1]["q_freq"] == [
        5500.0,
        5501.0,
        5502.0,
    ]
    assert saver.params["row_overrides_by_z"][1]["p1_nqz"] == 1
    # Each acquired row must use its own shifted inner axis.
    assert [float(row.data.axes["q_freq"][0]) for row in rows] == [
        5499.9,
        5500.0,
        5500.1,
    ]


def test_row_overrides_cannot_replace_the_swept_z_gain(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ide,
        "connect_quick",
        lambda repository: pytest.fail("must fail before connecting"),
    )

    with pytest.raises(ValueError, match="cannot replace the swept z_gain"):
        ide.run_flux_sweep(
            ROOT,
            experiment="qubit_spectroscopy",
            preset="qubit_fine",
            flux_values=np.array([-0.1, 0.0, 0.1]),
            row_overrides=lambda z: {"z_gain": 0.5},
            live_hardware=False,
            show_plot=False,
            backend=SyntheticBackend(seed=1),
        )
