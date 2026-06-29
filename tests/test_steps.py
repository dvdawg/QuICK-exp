import os
import numpy as np
import pytest

from labreadout import steps, config, state, fitting


def _cfg(tmp_path):
    p = tmp_path / "hardware.yml"
    p.write_text(
        "ip: 1.2.3.4\n"
        "data_path: '%s/'\n"
        "channels: {q: 8, r: 10, rr: 1}\n"
        "rf_board: {}\n"
        "bias: {channel: 0, default: -0.2}\n"
        "var: {r_freq: 6000.0, q_freq: 5000.0, r_power: -10}\n" % tmp_path
    )
    return config.load_config(str(p))


def test_module_imports_without_quick():
    # steps must import even though `quick` is absent off the lab PC
    assert hasattr(steps, "Session")
    assert hasattr(steps, "StepResult")


def test_session_merges_calibration_state_into_var(tmp_path):
    cal = str(tmp_path / "cal.yml")
    state.save(cal, {"r_freq": 6584.5})
    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=lambda *a, **k: None)
    assert sess.var["r_freq"] == 6584.5  # state overrides config default
    assert sess.var["q_freq"] == 5000.0  # untouched config default


def test_step_result_accept_commits_mapping_and_updates_var(tmp_path):
    cal = str(tmp_path / "cal.yml")
    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=lambda *a, **k: None)
    fake_fit = type("F", (), {"f0": 6584.5})()
    csv = tmp_path / "00001 - (ResonatorSpectroscopy)x.csv"
    csv.write_text("raw\n")
    result = steps.StepResult(
        fit=fake_fit, csv_path=str(csv), session=sess, commit_map={"r_freq": "f0"}
    )
    result.accept()
    assert sess.var["r_freq"] == 6584.5
    assert state.load(cal)["r_freq"] == 6584.5


def test_resonator_step_runs_fits_and_writes_sidecar(tmp_path):
    cal = str(tmp_path / "cal.yml")
    csv_path = str(tmp_path / "00001 - (ResonatorSpectroscopy)res.csv")

    # Fake runner returns synthetic resonance data + the CSV path quick would write.
    def fake_runner(experiment, var, title, data_path, run_kwargs, **overrides):
        f = np.linspace(6580, 6590, 500)
        hwhm = 0.5
        mag = 1 - 0.6 * hwhm**2 / ((f - 6584.5) ** 2 + hwhm**2)
        data = np.column_stack([f, 20 * np.log10(mag), np.zeros_like(f), mag, np.zeros_like(f)])
        open(csv_path, "w").write("raw\n")  # pretend quick saved the CSV
        return data, csv_path

    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=fake_runner)
    result = sess.resonator_spectroscopy(r_freq=np.linspace(6580, 6590, 500))

    assert result.fit.f0 == pytest.approx(6584.5, abs=0.1)
    assert "resonance" in result.suggestion.lower()
    assert os.path.exists(steps.results.sidecar_path(csv_path))


def test_unaccepted_step_does_not_change_state(tmp_path):
    cal = str(tmp_path / "cal.yml")
    csv_path = str(tmp_path / "00001 - (ResonatorSpectroscopy)res.csv")

    def fake_runner(experiment, var, title, data_path, run_kwargs, **overrides):
        f = np.linspace(6580, 6590, 300)
        mag = 1 - 0.6 * 0.25 / ((f - 6584.5) ** 2 + 0.25)
        data = np.column_stack([f, mag, np.zeros_like(f), mag, np.zeros_like(f)])
        open(csv_path, "w").write("raw\n")
        return data, csv_path

    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=fake_runner)
    sess.resonator_spectroscopy(r_freq=np.linspace(6580, 6590, 300))  # no accept()
    assert state.load(cal) == {}  # nothing committed until accept()
