from pathlib import Path
import runpy
from types import MappingProxyType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

import quickexp_v3.sweep_path as sweep_path_module
import quickexp_v3.qubit_flux_fit as qubit_flux_fit_module
from quickexp_v3.errors import ConfigError
from quickexp_v3.native_map import NativeMap
from quickexp_v3.sweep_path import (
    FrequencySweepPath,
    SweepPath,
    frequency_sweep_path_from_fit,
    frequency_sweep_path_from_polygon,
    load_frequency_sweep_path,
    load_sweep_path,
    plot_frequency_sweep_path,
    save_frequency_sweep_path,
    save_sweep_path,
    sweep_path_from_native_ridge,
    sweep_path_from_polygon,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frequency_sweep_path_interpolates_and_round_trips(tmp_path):
    path = FrequencySweepPath(
        method="test",
        z_gain=[-0.2, 0.0, 0.2],
        lower_frequency_mhz=[4700.0, 4800.0, 4750.0],
        upper_frequency_mhz=[4720.0, 4830.0, 4790.0],
        points_per_row=5,
        metadata={"source": "synthetic"},
    )

    assert path.frequency_sweep(0.1) == pytest.approx(
        [4775.0, 4783.75, 4792.5, 4801.25, 4810.0]
    )
    assert path.frequency_matrix().shape == (3, 5)
    assert path.total_points == 15

    output = save_frequency_sweep_path(path, tmp_path / "path.yml")
    loaded = load_frequency_sweep_path(output)
    assert loaded.method == "test"
    assert loaded.frequency_matrix() == pytest.approx(path.frequency_matrix())
    assert loaded.metadata["source"] == "synthetic"


def test_resolution_path_preserves_spacing_with_variable_rows_and_round_trips(
    tmp_path,
):
    path = SweepPath(
        method="ui_polygon",
        outer_name="q_gain",
        inner_name="q_freq",
        outer_values=[0.1, 0.2, 0.3],
        lower_inner_values=[4700.0, 4710.0, 4720.0],
        upper_inner_values=[4704.0, 4713.0, 4722.0],
        points_per_row=None,
        inner_resolution=1.0,
        metadata={"source": "background"},
    )

    assert path.point_counts.tolist() == [5, 4, 3]
    assert path.inner_sweeps()[1] == pytest.approx(
        [4710.0, 4711.0, 4712.0, 4713.0]
    )
    assert path.total_points == 12
    with pytest.raises(ConfigError, match="variable row lengths"):
        path.inner_matrix()

    output = save_sweep_path(path, tmp_path / "resolution_path.yml")
    loaded = load_sweep_path(output)
    assert loaded.points_per_row is None
    assert loaded.inner_resolution == pytest.approx(1.0)
    assert loaded.point_counts.tolist() == [5, 4, 3]


def test_fitted_ridge_defaults_to_background_frequency_resolution(
    monkeypatch,
    tmp_path,
):
    outer = np.array([0.1, 0.2, 0.3])
    inner = np.arange(4700.0, 4730.0, 2.0)
    signal = np.ones((outer.size, inner.size))
    native = NativeMap(
        source_csv=tmp_path / "background.csv",
        outer_label="Qubit Pulse Gain",
        outer_unit="a.u.",
        outer=outer,
        inner_label="Qubit Pulse Frequency",
        inner_unit="MHz",
        inner=inner,
        signals=MappingProxyType({"i": signal, "q": signal}),
        row_metadata=MappingProxyType({}),
        incomplete_outer=np.asarray([]),
        metadata=MappingProxyType({}),
    )
    centers = iter((4710.0, 4712.0, 4714.0))
    monkeypatch.setattr(sweep_path_module, "load_native_map", lambda _path: native)
    monkeypatch.setattr(
        qubit_flux_fit_module,
        "_extract_row",
        lambda frequencies, iq: (next(centers), 0.2, 0.9, 5.0, iq.real),
    )

    path = sweep_path_from_native_ridge(
        tmp_path / "background.csv",
        margin=4.0,
    )

    assert path.outer_name == "q_gain"
    assert path.inner_name == "q_freq"
    assert path.points_per_row is None
    assert path.inner_resolution == pytest.approx(2.0)
    assert np.diff(path.inner_sweeps()[0]) == pytest.approx(2.0)


def test_fit_margin_builds_asymmetric_corridor():
    fit = SimpleNamespace(
        source_csv="fit.csv",
        map_z_gain=np.array([-0.2, 0.0, 0.2]),
        z_gain=np.array([-0.2, 0.0, 0.2]),
        frequency=lambda z: 4800.0 + 100.0 * np.asarray(z),
        parameters={"period_z": 0.4},
        statistics={"r_squared": 0.99},
    )

    path = frequency_sweep_path_from_fit(
        fit,
        margin_mhz=(5.0, 12.0),
        points_per_row=18,
    )

    assert path.center_frequency_mhz == pytest.approx([4783.5, 4803.5, 4823.5])
    assert path.lower_frequency_mhz == pytest.approx([4775.0, 4795.0, 4815.0])
    assert path.upper_frequency_mhz == pytest.approx([4792.0, 4812.0, 4832.0])
    assert path.points_per_row == 18


def test_polygon_region_becomes_variable_row_bounds():
    vertices = [
        (-0.2, 4700.0),
        (0.2, 4720.0),
        (0.2, 4820.0),
        (-0.2, 4780.0),
    ]
    path = frequency_sweep_path_from_polygon(
        vertices,
        np.linspace(-0.3, 0.3, 7),
        points_per_row=11,
    )

    assert path.z_gain == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])
    assert path.lower_frequency_mhz == pytest.approx(
        [4700.0, 4705.0, 4710.0, 4715.0, 4720.0]
    )
    assert path.upper_frequency_mhz == pytest.approx(
        [4780.0, 4790.0, 4800.0, 4810.0, 4820.0]
    )
    assert path.frequency_matrix().shape == (5, 11)


def test_concave_polygon_preserves_disjoint_intervals_and_skips_gap(tmp_path):
    # A C-shaped polygon has two separate frequency bands in its right arm.
    vertices = [
        (0.0, 4700.0),
        (2.0, 4700.0),
        (2.0, 4710.0),
        (1.0, 4710.0),
        (1.0, 4730.0),
        (2.0, 4730.0),
        (2.0, 4740.0),
        (0.0, 4740.0),
    ]
    path = sweep_path_from_polygon(
        vertices,
        [0.5, 1.5],
        outer_name="z_gain",
        inner_name="q_freq",
        points_per_row=None,
        inner_resolution=5.0,
    )

    assert path.has_disjoint_intervals is True
    assert path.inner_intervals(0.5) == ((4700.0, 4740.0),)
    assert path.inner_intervals(1.5) == (
        (4700.0, 4710.0),
        (4730.0, 4740.0),
    )
    assert path.inner_sweep(1.5) == pytest.approx(
        [4700.0, 4705.0, 4710.0, 4730.0, 4735.0, 4740.0]
    )
    assert not np.any(
        (path.inner_sweep(1.5) > 4710.0)
        & (path.inner_sweep(1.5) < 4730.0)
    )
    assert path.point_counts.tolist() == [9, 6]

    output = save_sweep_path(path, tmp_path / "concave.yml")
    loaded = load_sweep_path(output)
    assert loaded.inner_segments == path.inner_segments
    assert loaded.inner_sweep(1.5) == pytest.approx(path.inner_sweep(1.5))

    legacy_document = path.as_dict()
    legacy_document["schema_version"] = 3
    legacy_document["inner"].pop("segments")
    legacy_output = tmp_path / "legacy_concave.yml"
    legacy_output.write_text(
        yaml.safe_dump(legacy_document, sort_keys=False),
        encoding="utf-8",
    )
    migrated = load_sweep_path(legacy_output)
    assert migrated.metadata["segments_reconstructed_from_polygon"] is True
    assert migrated.inner_intervals(1.5) == path.inner_intervals(1.5)
    assert migrated.inner_sweep(1.5) == pytest.approx(path.inner_sweep(1.5))

    fixed_count = sweep_path_from_polygon(
        vertices,
        [1.5],
        outer_name="z_gain",
        inner_name="q_freq",
        points_per_row=10,
    )
    assert fixed_count.inner_sweep(1.5).size == 10
    assert not np.any(
        (fixed_count.inner_sweep(1.5) > 4710.0)
        & (fixed_count.inner_sweep(1.5) < 4730.0)
    )


def test_polygon_preview_uses_exact_vertices_not_discrete_row_envelope(
    monkeypatch,
    tmp_path,
):
    vertices = [
        (-0.25, 4700.0),
        (0.25, 4720.0),
        (0.25, 4820.0),
        (-0.25, 4780.0),
    ]
    path = frequency_sweep_path_from_polygon(
        vertices,
        [-0.2, 0.0, 0.2],
        points_per_row=11,
    )
    outer = np.linspace(-0.3, 0.3, 7)
    inner = np.linspace(4600.0, 4900.0, 9)
    signal = np.zeros((outer.size, inner.size))
    native = NativeMap(
        source_csv=tmp_path / "background.csv",
        outer_label="Z gain",
        outer_unit="a.u.",
        outer=outer,
        inner_label="Qubit frequency",
        inner_unit="MHz",
        inner=inner,
        signals=MappingProxyType({"phase": signal}),
        row_metadata=MappingProxyType({}),
        incomplete_outer=np.asarray([]),
        metadata=MappingProxyType({}),
    )
    monkeypatch.setattr(
        sweep_path_module,
        "load_native_map",
        lambda _path: native,
    )

    figure = plot_frequency_sweep_path(path, tmp_path / "background.csv")
    highlighted = figure.axes[0].patches[0].get_xy()
    assert highlighted[:, 0].min() == pytest.approx(-0.25)
    assert highlighted[:, 0].max() == pytest.approx(0.25)
    assert highlighted[:, 1].min() == pytest.approx(4700.0)
    assert highlighted[:, 1].max() == pytest.approx(4820.0)
    plt.close(figure)


def test_frequency_sweep_path_rejects_invalid_bounds():
    with pytest.raises(ConfigError, match="upper.*bound"):
        FrequencySweepPath(
            method="invalid",
            z_gain=[0.0],
            lower_frequency_mhz=[4800.0],
            upper_frequency_mhz=[4790.0],
            points_per_row=11,
            metadata={},
        )


def test_06g_ui_polygon_reports_generic_sweep_path(tmp_path):
    namespace = runpy.run_path(
        str(ROOT / "experiments" / "06g_design_qubit_sweep_path.py"),
        run_name="test_06g_generic_path",
    )
    module_globals = namespace["main"].__globals__
    generic = SweepPath(
        method="ui_polygon",
        outer_name="q_gain",
        inner_name="q_freq",
        outer_values=[0.1, 0.2, 0.3],
        lower_inner_values=[4700.0, 4710.0, 4720.0],
        upper_inner_values=[4750.0, 4760.0, 4770.0],
        points_per_row=None,
        inner_resolution=1.0,
        metadata={"polygon_vertices": []},
    )
    module_globals.update(
        {
            "INPUT_CSV": tmp_path / "background.csv",
            "PATH_METHOD": "ui_polygon",
            "OUTPUT_YML": tmp_path / "path.yml",
            "SHOW_PREVIEW": False,
            "load_native_map": lambda _path: SimpleNamespace(
                outer_label="Qubit Pulse Gain",
                inner_label="Qubit Pulse Frequency",
            ),
            "design_sweep_path_ui": lambda *args, **kwargs: generic,
            "save_sweep_path": lambda path, output: Path(output),
            "plot_sweep_path": lambda *args, **kwargs: None,
        }
    )

    assert namespace["main"]() is generic


def test_06b_loads_z_gain_frequency_path(tmp_path):
    namespace = runpy.run_path(
        str(ROOT / "experiments" / "06b_qubit_spectroscopy_vs_flux.py"),
        run_name="test_06b_path",
    )
    module_globals = namespace["main"].__globals__
    path = SweepPath(
        method="ui_polygon",
        outer_name="z_gain",
        inner_name="q_freq",
        outer_values=[-0.1, 0.0],
        lower_inner_values=[4700.0, 4710.0],
        upper_inner_values=[4702.0, 4713.0],
        points_per_row=None,
        inner_resolution=1.0,
        metadata={},
    )
    captured = {}
    marker = object()

    def run_path(*args, **kwargs):
        captured.update(kwargs)
        return marker

    module_globals.update(
        {
            "SWEEP_PATH_YML": tmp_path / "path.yml",
            "TRACK_READOUT_FROM_ACCEPTED_FLUX_FIT": False,
            "load_sweep_path": lambda _path: path,
            "run_sweep_path": run_path,
            "SHOW_PLOT": False,
            "LIVE_HARDWARE": False,
        }
    )

    assert namespace["main"]() is marker
    assert captured["path"] is path
    assert captured["outer_control"] == "held_flux"
    assert captured["row_overrides"](-0.1) == {
        "r_freq": module_globals["FIXED_READOUT_FREQUENCY_MHZ"]
    }
    assert captured["experiment"] == "qubit_spectroscopy"
