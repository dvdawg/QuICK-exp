"""Amplitude- or duration-Rabi experiment."""

from __future__ import annotations

from dataclasses import replace

from ..analysis import fit_damped_oscillation
from ..errors import ConfigError
from .base import Experiment


class Rabi(Experiment):
    name = "rabi"
    quick_class = "Rabi"
    required = (
        "r_freq",
        "r_power",
        "r_length",
        "r_offset",
        "r_relax",
        "q_freq",
        "q_gain",
        "q_length",
    )
    axis_candidates = ("q_gain", "q_length")
    default_run_options = {"silent": True, "population": True}

    def build(self, config):
        plan = super().build(config)
        if len(plan.axes) != 1:
            raise ConfigError(
                "rabi requires exactly one swept axis: q_gain or q_length"
            )
        return plan

    def axis_units(self, config, axes):
        defaults = {"q_gain": "a.u.", "q_length": "us"}
        configured = super().axis_units(config, axes)
        return {name: configured.get(name) or defaults.get(name, "") for name in axes}

    def analyze(self, data, config):
        axis_name = next(iter(data.axes))
        result = fit_damped_oscillation(
            data.axes[axis_name],
            data.iq,
            model_name="decaying_rabi",
        )
        recommendation_name = (
            "defaults.q_gain" if axis_name == "q_gain" else "defaults.q_length"
        )
        return self.validate_analysis(
            replace(
                result,
                recommendation={
                    recommendation_name: float(result.values["half_period"])
                },
            ),
            config,
        )


EXPERIMENT = Rabi()



