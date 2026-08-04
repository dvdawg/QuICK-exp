import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from quickexp_v3.errors import ConfigError
from quickexp_v3.lab import HELD_Z_CONFIG
from quickexp_v3.mercator import (
    Expr,
    MercatorProgram,
    Var,
    held_z_program,
    install_program,
)


def _mask_templates(text):
    return re.sub(r"\{[^{}\n]+\}", "__TEMPLATE__", text)


def _soccfg():
    generators = [
        {
            "f_fabric": 599.04,
            "samps_per_clk": 16,
            "maxlen": 16384,
        }
        for _ in range(16)
    ]
    return {"gens": generators}


def _readout_program(name="ReadoutProbe"):
    program = MercatorProgram(name)
    for declaration in (
        ("r", 0, "Readout generator", ""),
        ("rr", 0, "Readout channel", ""),
        ("r_freq", 6884.0, "Readout frequency", "MHz"),
        ("r_length", 2.0, "Readout length", "us"),
        ("r_power", -35.0, "Readout power", "dB"),
        ("r_phase", 0.0, "Readout phase", "deg"),
        ("r_offset", 0.5, "Trigger offset", "us"),
        ("r_relax", 20.0, "Relax", "us"),
    ):
        program.declare_variable(*declaration)
    return (
        program.config(hard_avg=1, soft_avg=1, rep=0)
        .pulse(
            0,
            freq=Var("r_freq"),
            mixer=Expr("int(r_freq/1000)*1000+500"),
            length=Var("r_length"),
            power=Var("r_power"),
        )
        .readout(
            Var("rr"),
            p=0,
            length=Var("r_length"),
            phase=Var("r_phase"),
        )
        .step_pulse(0, Var("r"))
        .step_trigger(Var("r_offset"))
        .step_delay_auto(Var("r_relax"))
    )


def test_held_z_builder_matches_existing_template_key_for_key():
    built = yaml.safe_load(_mask_templates(held_z_program().render()))
    existing = yaml.safe_load(_mask_templates(HELD_Z_CONFIG))

    assert built == existing


def test_render_is_deterministic_and_unknown_fields_name_nearest_valid():
    program = _readout_program()
    assert program.render() == program.render()
    with pytest.raises(ConfigError, match=r"stlye.*style"):
        MercatorProgram("Typo").pulse(
            0,
            freq=1,
            length=1,
            stlye="const",
        )
    with pytest.raises(ConfigError, match="both gain and power"):
        MercatorProgram("Conflict").pulse(
            0,
            freq=1,
            length=1,
            gain=0.2,
            power=-20,
        )


def test_install_program_is_idempotent_and_carries_variables_and_labels():
    class BaseExperiment:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return kwargs

    quick = SimpleNamespace(
        experiment=SimpleNamespace(
            configs={},
            BaseExperiment=BaseExperiment,
        )
    )
    program = _readout_program()
    first = install_program(quick, program)
    second = install_program(quick, program)
    instance = first(test=True)

    assert first is second
    assert quick.experiment.ReadoutProbe is first
    assert "r{rr}_p" in quick.experiment.configs["ReadoutProbe"]
    assert instance.var["r_freq"] == 6884.0
    assert instance.var_label["r_freq"] == ("Readout frequency", "MHz")


def test_preflight_checks_mixer_continuity_and_clipping():
    program = _readout_program()
    variables = program.default_variables()
    variables["r_freq"] = np.linspace(6875.0, 6890.0, 16)
    report = program.preflight(_soccfg(), variables)
    assert report.ok
    assert report.details["mixers"][0]["mixer_values_mhz"] == [6500.0]

    variables["r_freq"] = np.linspace(6990.0, 7010.0, 21)
    report = program.preflight(_soccfg(), variables)
    assert not report.ok
    assert any(
        "mixer changes" in error and "6500" in error and "7500" in error
        for error in report.errors
    )

    variables["r_freq"] = 6884.0
    variables["r_power"] = 0.1
    report = program.preflight(_soccfg(), variables)
    assert any("0 dB" in error for error in report.errors)


def test_flat_top_memory_uses_conservative_ramp_budget_and_missing_term_fails():
    program = MercatorProgram("FlatTopProbe")
    for declaration in (
        ("q", 1, "Qubit generator", ""),
        ("q_freq", 5600.0, "Qubit frequency", "MHz"),
        ("q_gain", 0.4, "Qubit gain", ""),
        ("q_length", 10.0, "Qubit length", "us"),
        ("sigma", 0.05, "Ramp sigma", "us"),
    ):
        program.declare_variable(*declaration)
    program.declare_envelope_term("q_length", 1)
    program.pulse(
        1,
        freq=Var("q_freq"),
        length=Var("q_length"),
        gain=Var("q_gain"),
        style="flat_top",
        sigma=Var("sigma"),
    ).step_pulse(1, Var("q"))
    report = program.preflight(_soccfg(), program.default_variables())

    assert report.ok
    assert report.details["envelopes"]["q_length"]["required_samples"] == 3824
    assert any("8σ" in warning for warning in report.warnings)

    missing = MercatorProgram("MissingEnvelope")
    for declaration in (
        ("q", 1, "Qubit generator", ""),
        ("q_length", 0.2, "Length", "us"),
    ):
        missing.declare_variable(*declaration)
    missing.pulse(
        0,
        freq=5600,
        length=Var("q_length"),
        gain=0.2,
        style="gaussian",
    ).step_pulse(0, Var("q"))
    report = missing.preflight(_soccfg(), missing.default_variables())
    assert not report.ok
    assert any("no declared envelope term" in error for error in report.errors)


def test_preflight_rejects_subclock_timing_and_unbound_sweep():
    program = _readout_program()
    variables = program.default_variables()
    variables["r_length"] = 0.0005
    report = program.preflight(_soccfg(), variables)
    assert not report.ok
    assert any("one fabric clock" in error for error in report.errors)

    variables = program.default_variables()
    variables["unused_sweep"] = np.arange(3)
    report = program.preflight(_soccfg(), variables)
    assert not report.ok
    assert any("unbound swept" in error for error in report.errors)
