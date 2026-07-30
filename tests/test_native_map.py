from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.native_map import load_native_map


REAL_ROOT = Path(
    "/Users/dvdkm/Documents/code/qdg/data/2026-07-21_MET_ver191_qubit3"
)


def _write_map(path, *, with_readout=False, incomplete=False):
    outer = np.array([-1.0, 0.0, 1.0])
    inner = np.linspace(10.0, 12.0, 5)
    rows = []
    for outer_index, outer_value in enumerate(outer):
        for inner_index, inner_value in enumerate(inner):
            if incomplete and outer_index == 1 and inner_index == 2:
                continue
            iq = (outer_value + inner_value / 20.0) * np.exp(0.2j)
            row = [outer_value, inner_value]
            if with_readout:
                row.append(6800.0 + outer_value)
            row.extend([abs(iq), np.angle(iq), iq.real, iq.imag])
            rows.append(row)
    np.savetxt(path, rows, delimiter=",")
    dependent = []
    if with_readout:
        dependent.append(["Readout Frequency", "MHz"])
    dependent.extend(
        [["Amplitude", ""], ["Phase", "rad"], ["I", ""], ["Q", ""]]
    )
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [["Outer", "a.u."], ["Inner", "MHz"]],
                "dependent": dependent,
                "parameters": {"quick_experiment": "Example"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_native_map_loads_cartesian_and_combined_shapes(tmp_path):
    cartesian = tmp_path / "cartesian.csv"
    _write_map(cartesian)
    loaded = load_native_map(cartesian)
    assert loaded.signals["amplitude"].shape == (3, 5)
    assert loaded.complex_signal.shape == (3, 5)

    combined = tmp_path / "combined.csv"
    _write_map(combined, with_readout=True, incomplete=True)
    loaded = load_native_map(combined)
    assert loaded.outer.tolist() == [-1.0, 1.0]
    assert loaded.incomplete_outer.tolist() == [0.0]
    assert loaded.row_metadata["readout_frequency"].tolist() == [6799.0, 6801.0]


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="local real-data mirror absent")
def test_real_native_maps_have_verified_shapes():
    punchout = load_native_map(
        REAL_ROOT / "00023 - (ResonatorSpectroscopy)Punchout.csv"
    )
    assert punchout.signals["amplitude"].shape == (8, 30)
    flux = load_native_map(REAL_ROOT / "00025 - ResVsZ_held_bias.csv")
    assert flux.signals["amplitude"].shape == (21, 150)
    qubit = load_native_map(
        REAL_ROOT / "00031 - QubitSpecVsZ_fitted_readout.csv"
    )
    assert qubit.signals["amplitude"].shape == (9, 3000)
    assert qubit.row_metadata["readout_frequency"].shape == (9,)

