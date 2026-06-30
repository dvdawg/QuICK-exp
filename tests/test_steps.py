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


class _FakeSoccfg:
    """Minimal stand-in for a live soccfg: indexable gens/readouts + str() card map."""

    def __init__(self):
        self._d = {
            "gens": [{"dac": (i // 4, i % 4)} for i in range(12)],  # gen i -> DAC i
            "readouts": [
                {"adc": (1, 0), "ro_type": "ro"},  # index 0 -> ADC 0 (direct)
                {"adc": (2, 0), "ro_type": "ro"},  # index 1 -> ADC 4 (RF In)
            ],
        }

    def __getitem__(self, key):
        return self._d[key]

    def __str__(self):
        return "RF Out card has ports [8, 9, 10, 11]\nRF In card has ports [4, 5, 6, 7]\n"


def test_module_imports_without_quick():
    # steps must import even though `quick` is absent off the lab PC
    assert hasattr(steps, "Session")
    assert hasattr(steps, "StepResult")


def test_check_ports_resolves_logical_indices_to_physical_ports(tmp_path):
    sess = steps.Session(
        _cfg(tmp_path), calibration_path=str(tmp_path / "cal.yml"),
        soc=None, soccfg=_FakeSoccfg(), runner=lambda *a, **k: None,
    )
    resolved = sess.check_ports()
    assert resolved["r"].port == 10   # gen index 10 -> DAC 10
    assert resolved["rr"].port == 4   # ro index 1 -> ADC 4


def test_check_ports_strict_raises_on_declared_port_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.expected_ports = {"r": 8}  # operator believes r=10 is DAC 8; it's actually DAC 10
    sess = steps.Session(
        cfg, calibration_path=str(tmp_path / "cal.yml"),
        soc=None, soccfg=_FakeSoccfg(), runner=lambda *a, **k: None,
    )
    with pytest.raises(ValueError, match="mismatch|MISMATCH|declares"):
        sess.check_ports(strict=True)


def test_check_ports_without_soccfg_is_a_clear_error(tmp_path):
    sess = steps.Session(
        _cfg(tmp_path), calibration_path=str(tmp_path / "cal.yml"),
        soc=None, soccfg=None, runner=lambda *a, **k: None,
    )
    with pytest.raises(RuntimeError, match="soccfg|connect"):
        sess.check_ports()


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
    # The step attaches an advisory diagnosis (clean synthetic -> healthy).
    assert result.diagnosis is not None
    assert result.diagnosis.status == "ok"


def test_loopback_calibrate_finds_offset_and_safe_power(tmp_path):
    cal = str(tmp_path / "cal.yml")
    csv_path = str(tmp_path / "00001 - (LoopBack)cal.csv")

    def fake_runner(experiment, var, title, data_path, run_kwargs, **overrides):
        # Synthetic loopback: a pulse plateau at t in [0.5, 1.0); peak counts grow
        # with r_power (10 counts per dB above -40), railing past full scale at 0 dB.
        t = np.linspace(0, 2, 400)
        power = overrides.get("r_power", var.get("r_power", -30))
        height = max(1.0, 10.0 * (power + 40))  # -40->1, -10->300, 0->400... scaled below
        height = {-40: 20, -30: 120, -20: 500, -10: 1200, 0: 3000}.get(power, height)
        amp = np.full(t.size, 2.0)
        amp[(t >= 0.5) & (t < 1.0)] = height
        data = np.column_stack([t, amp, np.zeros_like(t), amp, np.zeros_like(t)])
        open(csv_path, "w").write("raw\n")
        return data, csv_path

    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=fake_runner)
    res = sess.loopback_calibrate(powers=[-40, -30, -20, -10, 0], fullscale=1500)

    assert res.fit.r_offset == pytest.approx(0.5, abs=0.05)
    assert res.fit.r_power == -10  # highest power under 0.8*1500 = 1200
    assert res.recommendations == {"r_offset": pytest.approx(0.5, abs=0.05), "r_power": -10}

    res.accept()
    assert sess.var["r_offset"] == pytest.approx(0.5, abs=0.05)
    assert state.load(cal)["r_power"] == -10


def test_rabi_recommends_the_pi_knob_that_was_swept(tmp_path):
    cal = str(tmp_path / "cal.yml")
    csv_path = str(tmp_path / "00001 - (Rabi)rabi.csv")

    def fake_runner(experiment, var, title, data_path, run_kwargs, **overrides):
        x = np.linspace(0, 0.5, 200)
        # Rabi oscillation in rotated IQ; first min (pi) near 0.1 us. A little
        # noise keeps the covariance well-conditioned (no OptimizeWarning).
        rng = np.random.default_rng(0)
        sig = np.cos(2 * np.pi * x / 0.2) + rng.normal(0, 0.02, x.size)
        data = np.column_stack([x, sig, np.zeros_like(x), sig, np.zeros_like(x)])
        with open(csv_path, "w") as fh:
            fh.write("raw\n")
        return data, csv_path

    sess = steps.Session(_cfg(tmp_path), calibration_path=cal, soc=None, runner=fake_runner)
    res = sess.rabi(x=np.linspace(0, 0.5, 200), xlabel="Pulse length (us)")
    # length was swept -> recommend q_length for the pi pulse
    assert "q_length" in res.recommendations
    assert res.recommendations["q_length"] == pytest.approx(res.fit.pi_value, rel=1e-6)


def test_apply_pushes_recommendations_into_live_var(tmp_path):
    sess = steps.Session(
        _cfg(tmp_path), calibration_path=str(tmp_path / "cal.yml"),
        soc=None, runner=lambda *a, **k: None,
    )
    sess.apply({"r_power": -12, "q_gain": 0.3})
    assert sess.var["r_power"] == -12 and sess.var["q_gain"] == 0.3
    # manual override still wins: apply does not persist, just stages live var
    assert state.load(str(sess.calibration_path)) == {}


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
