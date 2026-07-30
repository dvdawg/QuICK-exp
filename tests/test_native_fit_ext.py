import numpy as np
import pytest

from quickexp_v3.native_fit_ext import fit_echo
from test_native_fit import write_pair


def test_stretched_echo_selects_exponent_and_pure_decay_stays_simple(tmp_path):
    time = np.linspace(0.0, 20.0, 201)
    stretched = 0.1 + 0.8 * np.exp(-((time / 5.0) ** 1.8))
    source = write_pair(
        tmp_path / "echo.csv",
        quick_class="T2Echo",
        axis_label="Delay Time",
        axis_unit="us",
        x=time,
        signal=stretched,
        var={"cycle": 4},
    )
    fit = fit_echo(source, bootstrap_resamples=20)
    assert fit.parameters["decay_us"] == pytest.approx(5.0, rel=0.03)
    assert fit.parameters["exponent"] == pytest.approx(1.8, abs=0.2)
    assert fit.statistics["selected_stretched"]

    exponential = 0.1 + 0.8 * np.exp(-time / 5.0)
    source = write_pair(
        tmp_path / "echo_simple.csv",
        quick_class="T2Echo",
        axis_label="Delay Time",
        axis_unit="us",
        x=time,
        signal=exponential,
        var={"cycle": 1},
    )
    fit = fit_echo(source, bootstrap_resamples=20)
    assert not fit.statistics["selected_stretched"]
    assert fit.parameters["exponent"] == 1.0

