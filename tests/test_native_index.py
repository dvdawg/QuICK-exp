import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.native_index import NativeIndex
from test_native_fit import write_pair


REAL_DATA = Path(
    "/Users/dvdkm/Documents/code/qdg/data/2026-07-21_MET_ver191_qubit3"
)


def _pair(path, quick_class, axis, offset):
    x = np.linspace(0.0, 1.0, 11)
    return write_pair(
        path,
        quick_class=quick_class,
        axis_label=axis,
        axis_unit="MHz",
        x=x,
        signal=x + offset,
        var={"z_gain": 0.1},
    )


def test_index_selection_incremental_refresh_and_skip_list(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    first = _pair(data / "00001 - first.csv", "Rabi", "Qubit Pulse Length", 0)
    _pair(data / "00002 - second.csv", "T1", "Delay Time", 1)
    newest = _pair(data / "00003 - newest.csv", "Rabi", "Qubit Pulse Length", 2)
    stub = _pair(data / "00004 - stub.csv", "Rabi", "Qubit Pulse Length", 3)
    np.savetxt(stub, np.ones((1, 5)), delimiter=",")
    (data / "empty.csv").touch()
    (data / "labweb.yml").write_text("server: test\n", encoding="utf-8")
    os.utime(newest, (newest.stat().st_atime, newest.stat().st_mtime + 2))

    calls = {"count": 0}
    original = yaml.safe_load

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", counted)
    index = NativeIndex(data, cache_root=tmp_path / "cache").refresh()
    assert len(index.records()) == 3
    assert index.latest(
        quick_class="Rabi",
        axis_text="pulse length",
        n_axes=1,
    ).csv_path == newest.resolve()
    assert any("stub" in warning for warning in index.warnings)
    assert any("labweb.yml" in warning for warning in index.warnings)

    calls["count"] = 0
    first.touch()
    index.refresh()
    assert calls["count"] == 1


@pytest.mark.skipif(not REAL_DATA.exists(), reason="local real-data mirror absent")
def test_real_native_index_has_expected_complete_records(tmp_path):
    index = NativeIndex(REAL_DATA, cache_root=tmp_path / "cache").refresh()
    assert len(index.records()) == 30
    assert len(index.select(quick_class="LoopBack")) == 6
    assert any("00027" in warning for warning in index.warnings)
    assert any("00029" in warning for warning in index.warnings)
    assert any("labweb.yml" in warning for warning in index.warnings)

