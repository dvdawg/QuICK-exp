"""Ramsey cryoscope with a Gaussian-smoothed rectangular flux pulse."""

from __future__ import annotations

from ..mercator import Expr, MercatorProgram, Var


PROGRAM = MercatorProgram("Cryoscope")

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
    ("q_freq", 5600.0, "Ramsey Drive Frequency", "MHz"),
    ("q_gain", 0.2, "Pi-over-Two Pulse Gain", "a.u."),
    ("q_length", 0.04, "Pi-over-Two Gaussian Length", "us"),
    ("z_gain", 0.05, "Flux Pulse Gain", "a.u."),
    ("z_sigma", 0.002, "Flux Edge Gaussian Sigma", "us"),
    ("flux_time", 0.1, "Flux Pulse Duration", "us"),
    ("ramsey_phase", 0.0, "Second Ramsey Pulse Phase", "deg"),
):
    PROGRAM.declare_variable(*_declaration)

(
    PROGRAM.config(hard_avg=4096, soft_avg=1, rep=1)
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
        phase=0.0,
    )
    .pulse(
        2,
        style="gaussian",
        freq=Var("q_freq"),
        length=Var("q_length"),
        gain=Var("q_gain"),
        phase=Var("ramsey_phase"),
    )
    .pulse(
        9,
        style="flat_top",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("flux_time"),
        sigma=Var("z_sigma"),
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
    .step_pulse(2, Var("q"))
    .step_delay_auto()
    .step_pulse(0, Var("r"))
    .step_trigger(Var("r_offset"))
    .step_delay_auto(Var("r_relax"))
    .declare_envelope_term("q_length", 2)
    .declare_envelope_term("flux_time", 1)
)
