import os
import numpy as np

from labreadout import state


def test_load_missing_file_returns_empty(tmp_path):
    assert state.load(str(tmp_path / "nope.yml")) == {}


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "cal.yml")
    state.save(path, {"r_freq": 6584.5, "q_freq": 5022.0})
    assert state.load(path) == {"r_freq": 6584.5, "q_freq": 5022.0}


def test_record_merges_and_persists(tmp_path):
    path = str(tmp_path / "cal.yml")
    state.save(path, {"r_freq": 6584.5})
    updated = state.record(path, {"q_freq": 5022.0})
    assert updated == {"r_freq": 6584.5, "q_freq": 5022.0}
    assert state.load(path) == {"r_freq": 6584.5, "q_freq": 5022.0}


def test_record_overwrites_existing_key(tmp_path):
    path = str(tmp_path / "cal.yml")
    state.save(path, {"r_freq": 6584.5})
    state.record(path, {"r_freq": 6590.0})
    assert state.load(path)["r_freq"] == 6590.0


def test_merged_var_overrides_only_known_keys(tmp_path):
    base = {"r_freq": 6000.0, "q_freq": 5000.0, "q": 8}
    merged = state.merged_var(base, {"r_freq": 6584.5})
    assert merged["r_freq"] == 6584.5
    assert merged["q_freq"] == 5000.0
    assert merged["q"] == 8
    # base dict is not mutated
    assert base["r_freq"] == 6000.0


def test_record_coerces_numpy_scalars_to_plain_python(tmp_path):
    path = str(tmp_path / "cal.yml")
    state.record(path, {"r_freq": np.float64(6584.5)})
    # Re-loading must not require numpy and yields a plain float.
    loaded = state.load(path)
    assert isinstance(loaded["r_freq"], float)
    assert loaded["r_freq"] == 6584.5
