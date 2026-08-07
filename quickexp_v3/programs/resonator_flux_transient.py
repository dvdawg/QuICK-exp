"""Readout-resonator response to a held flux step, versus observation time.

The resonator is the sensor, so unlike ``FluxStepSpectroscopy`` this program
does **not** return the flux line to its baseline before reading out. Each shot
steps the line, waits ``probe_time``, and probes the resonator while the line is
still held at the stepped level. There is no qubit pulse; the qubit stays in
|g> and enters only through whatever dispersive shift it already contributes to
the empirical ``r_freq(z)`` calibration.

The post-readout ``z_baseline_gain`` pulse both parks the line and provides the
settle interval: the line rests at the baseline for the whole of ``r_relax``, so
the next shot's step edge starts from a settled line. That makes ``r_relax`` the
baseline-settle time as well as the qubit-reset time, and it must exceed several
times the longest flux time constant being measured.
"""

from __future__ import annotations

from ..mercator import Expr, MercatorProgram, Var


PROGRAM = MercatorProgram("ResonatorFluxTransient")

for _declaration in (
    ("r", 0, "Readout DAC Generator", ""),
    ("rr", 0, "Readout ADC Channel", ""),
    ("z", 2, "Z DAC Generator", ""),
    ("r_freq", 5879.2, "Readout Probe Frequency", "MHz"),
    ("r_power", -35.0, "Readout Pulse Power", "dB"),
    ("r_length", 2.0, "Readout Pulse Length", "us"),
    ("r_phase", 0.0, "Readout Phase", "deg"),
    ("r_offset", 0.5, "Readout Trigger Offset", "us"),
    ("r_relax", 300.0, "Relax and Baseline Settle Time", "us"),
    ("z_step_gain", -0.065, "Stepped Flux Level", "a.u."),
    ("z_baseline_gain", -0.080, "Baseline Flux Level", "a.u."),
    ("z_set_length", 0.008, "Persistent Z Set Pulse Length", "us"),
    ("probe_time", 1.0, "Flux Step Observation Time", "us"),
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
    # Slots 9 and 10 keep the level-setting convention used by
    # FluxStepSpectroscopy: constant mode="last" pulses that latch the
    # generator output and hold it until the next one.
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
        gain=Var("z_baseline_gain"),
    )
    .readout(
        Var("rr"),
        p=0,
        length=Var("r_length"),
        phase=Var("r_phase"),
    )
    # The step edge. The line was resting at z_baseline_gain through the
    # preceding r_relax, so this is a genuine step from a settled baseline.
    .step_pulse(9, Var("z"))
    .step_delay_auto(Var("probe_time"))
    # Probe while the line is still stepped: no return-to-baseline here.
    .step_pulse(0, Var("r"))
    .step_trigger(Var("r_offset"))
    .step_delay_auto()
    .step_pulse(10, Var("z"))
    .step_delay_auto(Var("r_relax"))
)
