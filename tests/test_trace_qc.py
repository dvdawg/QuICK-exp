import numpy as np

from quickexp_v3.native_map import NativeMap
from quickexp_v3.trace_qc import qc_map, qc_trace


def test_trace_qc_detects_spikes_drift_nonuniformity_and_clipping():
    rng = np.random.default_rng(8)
    x = np.linspace(0.0, 1.0, 400)
    clean = np.sin(2 * np.pi * x) + 1j * rng.normal(0.0, 0.005, x.size)
    quality = qc_trace(x, clean)
    assert quality.axis_uniform
    assert quality.snr_estimate > 10

    bad = clean.copy()
    bad[100] += 20
    bad[-80:] += np.linspace(0.0, 2.0, 80)
    quality = qc_trace(np.r_[x[:-1], 1.2], bad)
    assert not quality.axis_uniform
    assert quality.spike_count >= 1
    assert quality.baseline_drift > 0.1

    clipped = np.clip(clean.real, -0.5, 0.5) + 1j * clean.imag
    assert qc_trace(x, clipped).clipping_suspected


def test_qc_map_returns_one_result_per_outer_row(tmp_path):
    x = np.linspace(0.0, 1.0, 30)
    signals = {
        "i": np.vstack((np.sin(x), np.cos(x))),
        "q": np.zeros((2, x.size)),
    }
    native = NativeMap(
        source_csv=tmp_path / "map.csv",
        outer_label="outer",
        outer_unit="",
        outer=np.array([0.0, 1.0]),
        inner_label="inner",
        inner_unit="",
        inner=x,
        signals=signals,
        row_metadata={},
        incomplete_outer=np.array([]),
        metadata={},
    )
    assert set(qc_map(native)) == {0.0, 1.0}

