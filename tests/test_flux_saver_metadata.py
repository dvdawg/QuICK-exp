from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quickexp_v3.backend import SyntheticBackend
import quickexp_v3.ide as ide
from quickexp_v3.sweep_path import SweepPath


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


def test_zpa_path_rows_use_program_z_without_held_controller(tmp_path, monkeypatch):
    backend = SyntheticBackend(seed=23)
    backend.data_path = str(tmp_path)

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
        lambda *args, **kwargs: pytest.fail("held controller must not be created"),
    )

    rows = ide.run_flux_sweep(
        ROOT,
        experiment="two_tone_zpa",
        preset="two_tone_zpa",
        flux_values=np.array([-0.1, 0.0, 0.1]),
        row_overrides=lambda z: {
            "q_freq": np.linspace(4750.0 + 100.0 * z, 4770.0 + 100.0 * z, 11)
        },
        row_override_metadata={"method": "test_path"},
        overrides={"hard_avg": 1, "soft_avg": 1},
        use_held_flux_controller=False,
        live_hardware=True,
        show_plot=False,
    )

    saver = FakeSaver.instance
    assert saver.params["z_gain_control"] == "experiment_program"
    assert saver.params["row_override_calibration"] == {"method": "test_path"}
    assert len(rows) == 3
    assert all(tuple(row.data.axes) == ("q_freq",) for row in rows)
    assert [row.data.axes["q_freq"].size for row in rows] == [11, 11, 11]


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


def test_generic_q_gain_path_saves_variable_frequency_row_lengths(
    tmp_path,
    monkeypatch,
):
    backend = SyntheticBackend(seed=24)
    backend.data_path = str(tmp_path)
    flux = FakeFlux()

    def sweep(config, sweep_config, progressBar):
        assert progressBar is True
        return [
            {"q_gain": float(value)}
            for value in sweep_config["q_gain"]
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
    path = SweepPath(
        method="ui_polygon",
        outer_name="q_gain",
        inner_name="q_freq",
        outer_values=[0.1, 0.2, 0.3],
        lower_inner_values=[5500.0, 5500.0, 5500.0],
        upper_inner_values=[5502.0, 5503.0, 5504.0],
        points_per_row=None,
        inner_resolution=1.0,
        metadata={"source_csv": "background.csv"},
        outer_label="Qubit Pulse Gain",
        outer_unit="a.u.",
        inner_label="Qubit Pulse Frequency",
        inner_unit="MHz",
    )

    rows = ide.run_sweep_path(
        ROOT,
        path=path,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        overrides={"hard_avg": 1, "soft_avg": 1},
        fixed_z_gain=-0.1,
        live_hardware=True,
        show_plot=False,
    )

    saver = FakeSaver.instance
    assert [saved.shape[0] for saved in saver.rows] == [3, 4, 5]
    assert [row.data.axes["q_freq"].size for row in rows] == [3, 4, 5]
    assert [saved[0, 0] for saved in saver.rows] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert saver.params["sweep_path"]["inner"]["resolution"] == 1.0
    assert saver.params["fixed_z_gain"] == -0.1
    assert saver.wrote_yml is True
    assert flux.values == [-0.1]
    assert flux.parked is True


def test_generic_z_path_maps_outer_to_held_flux_and_row_overrides(
    tmp_path,
    monkeypatch,
):
    backend = SyntheticBackend(seed=25)
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
    path = SweepPath(
        method="ui_polygon",
        outer_name="z_gain",
        inner_name="q_freq",
        outer_values=[-0.1, 0.0, 0.1],
        lower_inner_values=[5500.0, 5501.0, 5502.0],
        upper_inner_values=[5502.0, 5504.0, 5506.0],
        points_per_row=None,
        inner_resolution=1.0,
        inner_segments=[
            [[5500.0, 5502.0]],
            [[5501.0, 5502.0], [5503.0, 5504.0]],
            [[5502.0, 5506.0]],
        ],
        metadata={},
        outer_label="Z Gain",
        outer_unit="a.u.",
        inner_label="Qubit Frequency",
        inner_unit="MHz",
    )
    calibration = {"model": "test_readout_vs_flux"}

    rows = ide.run_sweep_path(
        ROOT,
        path=path,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        row_overrides=lambda z: {"r_freq": 6884.0 + z},
        row_override_metadata=calibration,
        overrides={"hard_avg": 1, "soft_avg": 1},
        outer_control="held_flux",
        live_hardware=True,
        show_plot=False,
    )

    saver = FakeSaver.instance
    assert [row.data.axes["q_freq"].size for row in rows] == [3, 4, 5]
    assert rows[1].data.axes["q_freq"] == pytest.approx(
        [5501.0, 5502.0, 5503.0, 5504.0]
    )
    assert flux.values == pytest.approx([-0.1, 0.0, 0.1])
    assert flux.parked is True
    assert saver.params["outer_control"] == "held_flux"
    assert saver.params["row_overrides_by_outer"][2] == {
        "z_gain": 0.1,
        "r_freq": 6884.1,
    }
    assert saver.params["row_override_calibration"] == calibration
