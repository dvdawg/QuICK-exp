"""Pulsed spectroscopy of a held flux step versus observation time."""

from __future__ import annotations

from ..mercator import Expr, MercatorProgram, Var


PROGRAM = MercatorProgram("FluxStepSpectroscopy")

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
    ("r_relax", 300.0, "Relax and Bias-Tee Reset Time", "us"),
    ("q_freq", 5600.0, "Qubit Probe Frequency", "MHz"),
    ("q_gain", 0.15, "Qubit Probe Gain", "a.u."),
    ("q_length", 0.04, "Gaussian Qubit Probe Length", "us"),
    ("z_step_gain", 0.05, "Flux Step Gain", "a.u."),
    ("z_return_gain", 0.0, "Flux Return Gain", "a.u."),
    ("z_idle_gain", 0.0, "Post-Readout Fast-Line Gain", "a.u."),
    ("z_set_length", 0.004, "Persistent Z Set Pulse Length", "us"),
    ("probe_time", 1.0, "Flux Step Observation Time", "us"),
    ("post_probe_to_return", 0.07, "Probe-End to Flux Return", "us"),
    ("return_guard", 0.11, "Flux Return to Readout Guard", "us"),
):
    PROGRAM.declare_variable(*_declaration)

(
    PROGRAM.config(hard_avg=2048, soft_avg=1, rep=0)
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
        mode="last",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("z_set_length"),
        gain=Var("z_step_gain"),
    )
    .pulse(
        10,
        style="const",
        mode="last",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("z_set_length"),
        gain=Var("z_return_gain"),
    )
    .pulse(
        11,
        style="const",
        mode="last",
        freq=0,
        mixer=0,
        nqz=1,
        length=Var("z_set_length"),
        gain=Var("z_idle_gain"),
    )
    .readout(
        Var("rr"),
        p=0,
        length=Var("r_length"),
        phase=Var("r_phase"),
    )
    .step_pulse(9, Var("z"))
    .step_delay_auto(Var("probe_time"))
    .step_pulse(1, Var("q"))
    .step_delay_auto(Var("post_probe_to_return"))
    .step_pulse(10, Var("z"))
    .step_delay_auto(Var("return_guard"))
    .step_pulse(0, Var("r"))
    .step_trigger(Var("r_offset"))
    .step_delay_auto()
    .step_pulse(11, Var("z"))
    .step_delay_auto(Var("r_relax"))
    .declare_envelope_term("q_length", 1)
)
