"""Adapter for time-resolved pulsed spectroscopy of a flux step."""

from __future__ import annotations

from dataclasses import replace

from ..errors import ConfigError
from ..programs import FLUX_STEP_SPECTROSCOPY
from .base import Experiment


class FluxStepSpectroscopy(Experiment):
    name = "flux_step_spectroscopy"
    quick_class = FLUX_STEP_SPECTROSCOPY.name
    required = (
        "r_freq",
        "r_power",
        "r_length",
        "r_offset",
        "r_relax",
        "q_freq",
        "q_gain",
        "q_length",
        "z_step_gain",
        "z_return_gain",
        "z_idle_gain",
        "z_set_length",
        "probe_time",
        "post_probe_to_return",
        "return_guard",
    )
    axis_candidates = ("probe_time", "q_freq")
    default_run_options = {"silent": True, "dB": False, "population": False}

    def build(self, config):
        plan = super().build(config)
        if plan.axes not in {
            ("probe_time", "q_freq"),
            ("q_freq",),
        }:
            raise ConfigError(
                "flux_step_spectroscopy requires q_freq, optionally with "
                "probe_time as the outer axis"
            )
        return replace(
            plan,
            metadata={
                **dict(plan.metadata),
                "preflight": FLUX_STEP_SPECTROSCOPY.preflight,
                "authored_program": FLUX_STEP_SPECTROSCOPY.name,
            },
        )

    def axis_units(self, config, axes):
        defaults = {"probe_time": "us", "q_freq": "MHz"}
        configured = super().axis_units(config, axes)
        return {
            name: configured.get(name) or defaults.get(name, "")
            for name in axes
        }


EXPERIMENT = FluxStepSpectroscopy()
