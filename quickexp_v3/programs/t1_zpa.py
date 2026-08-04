"""T1 measurement with a finite Z pulse applied during the decay delay."""

from __future__ import annotations

from ..mercator import Expr, MercatorProgram, Var


PROGRAM = MercatorProgram("T1_zpa")

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
    ("r_relax", 60.0, "Relax Time", "us"),
    ("q_freq", 5600.0, "Qubit Pulse Frequency", "MHz"),
    ("q_gain", 0.4, "Qubit Pulse Gain", "a.u."),
    ("q_length", 0.1, "Qubit Pulse Length", "us"),
    ("z_gain", 0.0, "Z Pulse Gain", "a.u."),
    ("time", 5.0, "Z-biased Delay Time", "us"),
):
    PROGRAM.declare_variable(*_declaration)

(
    PROGRAM.config(hard_avg=1000, soft_avg=1, rep=1)
    .pulse(
        0,
        freq=Var("r_freq"),
        mixer=Expr("int(r_freq/1000)*1000+500"),
        length=Var("r_length"),
        power=Var("r_power"),
    )
    .pulse(
        1,
        style="gaussian",
        freq=Var("q_freq"),
        length=Var("q_length"),
        gain=Var("q_gain"),
    )
    .pulse(
        9,
        style="const",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("time"),
        gain=Var("z_gain"),
    )
    .readout(
        Var("rr"),
        p=0,
        length=Var("r_length"),
        phase=Var("r_phase"),
    )
    .step_pulse(1, Var("q"))
    .step_delay_auto()
    .step_pulse(9, Var("z"))
    .step_delay_auto()
    .step_pulse(0, Var("r"))
    .step_trigger(Var("r_offset"))
    .step_delay_auto(Var("r_relax"))
    .declare_envelope_term("q_length", 1)
)
