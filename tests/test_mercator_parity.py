from pathlib import Path
import re

import pytest
import yaml

from quickexp_v3.mercator import Expr, MercatorProgram, Var


DATA = Path("/Users/dvdkm/Documents/code/qdg/data")
ORACLES = {
    "LoopBack": DATA
    / "2026-07-21_MET_ver191_qubit3"
    / "00000 - (LoopBack)DAC8-ADC0.yml",
    "QubitSpectroscopy": DATA
    / "2026-07-21_MET_ver191_qubit3"
    / "00026 - (QubitSpectroscopy)QubitSpec_Zp0p0000_r6883p702.yml",
    "Rabi": DATA
    / "2026-07-21_MET_ver191_qubit2"
    / "00059 - (Rabi)Zp0p0000_rabi_q5606.000.yml",
    "T1": DATA
    / "2026-07-21_MET_ver191_qubit2"
    / "00064 - (T1)Zp0p0000_T1_q5605.500.yml",
    "T2Ramsey": DATA
    / "2026-07-21_MET_ver191_qubit2"
    / "00067 - (T2Ramsey)Zp0p0000_T2Ramsey_q5606.500.yml",
    "T2Echo": DATA
    / "2026-07-21_MET_ver191_qubit2"
    / "00077 - (T2Echo)Zp0p0000_T2Echo_q5603.910.yml",
    "DispersiveSpectroscopy": DATA
    / "2026-07-21_MET_ver191_qubit2"
    / "00078 - (DispersiveSpectroscopy)Zp0p0000_dispersive_q5603.910.yml",
}


def _load(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["parameters"]["var"], document["parameters"]["config"]


def _base(kind, variables, expected):
    program = MercatorProgram(f"Parity{kind}")
    for name, value in variables.items():
        program.declare_variable(name, value, name, "")
    program.config(
        hard_avg=expected.get("hard_avg"),
        soft_avg=expected.get("soft_avg"),
        rep=expected.get("rep"),
    )
    return program


def _readout(program, *, mixer=False):
    arguments = {}
    if mixer:
        arguments["mixer"] = Expr("int(r_freq/1000)*1000+500")
    program.pulse(
        0,
        freq=Var("r_freq"),
        length=Var("r_length"),
        power=Var("r_power"),
        **arguments,
    ).readout(
        Var("rr"),
        p=0,
        length=Var("r_length"),
        phase=Var("r_phase"),
    )


def _program(kind, variables, expected):
    program = _base(kind, variables, expected)
    if kind == "LoopBack":
        _readout(program)
        return (
            program.step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_delay_auto(Var("r_relax"))
        )

    _readout(program, mixer=kind == "DispersiveSpectroscopy")
    if kind in {"QubitSpectroscopy", "DispersiveSpectroscopy"}:
        program.pulse(
            1,
            style="flat_top" if kind == "QubitSpectroscopy" else "gaussian",
            freq=Var("q_freq"),
            length=Var("q_length"),
            sigma=0.05 if kind == "QubitSpectroscopy" else None,
            gain=Var("q_gain"),
        )
        return (
            program.step_pulse(1, Var("q"))
            .step_delay_auto()
            .step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_delay_auto(Var("r_relax"))
        )
    if kind == "Rabi":
        program.pulse(
            1,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length"),
            gain=Var("q_gain"),
        )
        return (
            program.step_pulse(1, Var("q"))
            .step_delay_auto(0.002)
            .step_goto(0, rep=Var("cycle"))
            .step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_delay_auto(Var("r_relax"))
        )
    if kind == "T1":
        program.pulse(
            1,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length"),
            gain=Var("q_gain"),
        )
        return (
            program.step_pulse(1, Var("q"))
            .step_delay_auto(Var("time"))
            .step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_goto(7, t=Var("r_reset"))
            .step_delay_auto(Var("r_reset"))
            .step_pulse(
                1,
                Var("q"),
                r=Var("rr"),
                threshold=Var("r_threshold"),
            )
            .step_delay_auto(Var("r_relax"))
        )
    if kind == "T2Ramsey":
        program.pulse(
            1,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length"),
            gain=Var("q_gain"),
        ).pulse(
            2,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length_2"),
            gain=Var("q_gain_2"),
        ).pulse(
            3,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length_2"),
            gain=Var("q_gain_2"),
            phase=Expr("360*fringe_freq*time"),
        )
        return (
            program.step_pulse(2, Var("q"))
            .step_delay_auto(Var("r_reset"))
            .step_pulse(3, Var("q"))
            .step_delay_auto()
            .step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_goto(9, t=Var("r_reset"))
            .step_delay_auto(Var("r_reset"))
            .step_pulse(
                1,
                Var("q"),
                r=Var("rr"),
                threshold=Var("r_threshold"),
            )
            .step_delay_auto(Var("r_relax"))
        )
    if kind == "T2Echo":
        program.pulse(
            1,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length"),
            gain=Var("q_gain"),
            phase=90,
        ).pulse(
            2,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length_2"),
            gain=Var("q_gain_2"),
        ).pulse(
            3,
            style="gaussian",
            freq=Var("q_freq"),
            length=Var("q_length_2"),
            gain=Var("q_gain_2"),
            phase=Expr("360*fringe_freq*time"),
        )
        return (
            program.step_pulse(2, Var("q"))
            .step_delay_auto(Var("time"))
            .step_pulse(1, Var("q"))
            .step_delay_auto(Var("time"))
            .step_goto(1, rep=Var("cycle"))
            .step_pulse(3, Var("q"))
            .step_delay_auto()
            .step_pulse(0, Var("r"))
            .step_trigger(Var("r_offset"))
            .step_goto(12, t=Var("r_reset"))
            .step_delay_auto(Var("r_reset"))
            .step_pulse(
                1,
                Var("q"),
                r=Var("rr"),
                threshold=Var("r_threshold"),
            )
            .step_delay_auto(Var("r_relax"))
        )
    raise AssertionError(kind)


def _resolve(program, variables):
    pattern = re.compile(r"\{([^{}\n]+)\}")

    def replacement(match):
        source = match.group(1)
        if source in variables:
            value = variables[source]
        else:
            value = eval(
                source,
                {"__builtins__": {}, "int": int},
                dict(variables),
            )
        return yaml.safe_dump(value, default_flow_style=True).strip().splitlines()[0]

    rendered = pattern.sub(replacement, program.render())
    document = yaml.safe_load(rendered)
    steps = document.pop("steps")
    for index, step in enumerate(steps):
        for name, value in step.items():
            document[f"{index}_{name}"] = value
    return document


def _normalize_power_gain(config):
    normalized = dict(config)
    for key in tuple(normalized):
        match = re.fullmatch(r"p(\d+)_power", key)
        if match:
            normalized.pop(f"p{match.group(1)}_gain", None)
    return normalized


@pytest.mark.skipif(
    not all(path.exists() for path in ORACLES.values()),
    reason="local real-data parity oracles absent",
)
@pytest.mark.parametrize("kind", tuple(ORACLES))
def test_reconstructed_program_matches_real_resolved_config(kind):
    variables, expected = _load(ORACLES[kind])
    program = _program(kind, variables, expected)
    resolved = _resolve(program, variables)

    assert _normalize_power_gain(resolved) == _normalize_power_gain(expected)
