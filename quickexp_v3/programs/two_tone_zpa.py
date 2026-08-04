"""Native two-tone spectroscopy while a finite Z pulse is active."""

from __future__ import annotations

from ..mercator import Expr, MercatorProgram, Var


PROGRAM = MercatorProgram("TwoTone_ZPA")

for _declaration in (
    ("r", 0, "Readout DAC Generator", ""),
    ("rr", 0, "Readout ADC Channel", ""),
    ("q", 1, "Qubit DAC Generator", ""),
    ("z", 2, "Z DAC Generator", ""),
    ("r_freq", 6884.0, "Readout Pulse Frequency", "MHz"),
    ("r_power", -35.0, "Readout Pulse Power", "dB"),
    ("r_length", 2.0, "Readout Pulse Length", "us"),
    ("r_phase", 0.0, "Readout Phase", "deg"),
    ("r_offset", 0.5, "Readout Trigger Offset", "us"),
    ("r_relax", 20.0, "Relax Time", "us"),
    ("q_freq", 5600.0, "Qubit Pulse Frequency", "MHz"),
    ("q_gain", 0.3, "Qubit Pulse Gain", "a.u."),
    ("q_length", 10.0, "Qubit Pulse Length", "us"),
    ("z_gain", 0.0, "Z Pulse Gain", "a.u."),
    ("z_length", 20.0, "Z Pulse Length", "us"),
    ("z_settle", 5.0, "Z Settle Time", "us"),
):
    PROGRAM.declare_variable(*_declaration)

(
    PROGRAM.config(hard_avg=1000, soft_avg=1, rep=0)
    .pulse(
        0,
        freq=Var("r_freq"),
        mixer=Expr("int(r_freq/1000)*1000+500"),
        length=Var("r_length"),
        power=Var("r_power"),
    )
    .pulse(
        1,
        style="flat_top",
        freq=Var("q_freq"),
        length=Var("q_length"),
        sigma=0.05,
        gain=Var("q_gain"),
    )
    .pulse(
        9,
        style="const",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("z_length"),
        gain=Var("z_gain"),
    )
    .readout(
        Var("rr"),
        p=0,
        length=Var("r_length"),
        phase=Var("r_phase"),
    )
    .step_pulse(9, Var("z"))
    .step_delay_auto(Var("z_settle"))
    .step_pulse(1, Var("q"))
    .step_delay_auto()
    .step_pulse(0, Var("r"))
    .step_trigger(Var("r_offset"))
    .step_delay_auto(Var("r_relax"))
    .declare_envelope_term("q_length", 1)
)
