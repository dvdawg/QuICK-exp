import os
import numpy as np
import yaml

from labreadout import results, fitting


def _resonator_fit():
    rng = np.random.default_rng(0)
    f = np.linspace(6580, 6590, 500)
    hwhm = 0.5
    mag = 1 - 0.5 * hwhm**2 / ((f - 6584.5) ** 2 + hwhm**2)
    I = mag + rng.normal(0, 0.002, f.size)
    Q = rng.normal(0, 0.002, f.size)
    return fitting.fit_resonator(f, I, Q)


def test_summary_has_scalars_no_arrays():
    summary = results.summarize(_resonator_fit())
    assert summary["f0"] != 0
    assert "params" in summary and "uncertainties" in summary
    assert "gof" in summary
    # no raw data arrays leak into the summary
    for value in summary.values():
        assert not isinstance(value, np.ndarray)


def test_sidecar_path_replaces_csv_suffix():
    p = results.sidecar_path("/data/00021 - (ResonatorSpectroscopy)foo.csv")
    assert p == "/data/00021 - (ResonatorSpectroscopy)foo.fit.yml"


def test_write_sidecar_creates_yaml_next_to_csv(tmp_path):
    csv = tmp_path / "00021 - (ResonatorSpectroscopy)foo.csv"
    csv.write_text("6580,1,0,1,0\n")  # pretend raw data
    before = csv.read_text()

    out = results.write_sidecar(str(csv), _resonator_fit())

    assert os.path.exists(out)
    loaded = yaml.safe_load(open(out))
    assert "f0" in loaded and "gof" in loaded
    # raw CSV is left untouched
    assert csv.read_text() == before


def test_write_sidecar_accepts_extra_metadata(tmp_path):
    csv = tmp_path / "00021 - (ResonatorSpectroscopy)foo.csv"
    csv.write_text("x\n")
    out = results.write_sidecar(
        str(csv), _resonator_fit(), extra={"experiment": "ResonatorSpectroscopy"}
    )
    loaded = yaml.safe_load(open(out))
    assert loaded["experiment"] == "ResonatorSpectroscopy"
