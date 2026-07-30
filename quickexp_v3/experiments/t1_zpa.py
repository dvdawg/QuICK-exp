"""Adapter for T1 versus finite Z-pulse bias."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..errors import ConfigError
from ..programs import T1_ZPA
from .base import Experiment


class T1ZPA(Experiment):
    name = "t1_zpa"
    quick_class = T1_ZPA.name
    required = (
        "r_freq",
        "r_power",
        "r_length",
        "r_offset",
        "r_relax",
        "q_freq",
        "q_gain",
        "q_length",
        "z_gain",
    )
    axis_candidates = ("z_gain", "time")
    default_run_options = {"silent": True, "population": False}

    def _parameters(self, config):
        parameters = super()._parameters(config)
        delay = parameters.pop("delay", None)
        if delay is not None:
            if "time" in parameters and isinstance(parameters["time"], np.ndarray):
                raise ConfigError("t1_zpa cannot declare both delay and time sweeps")
            parameters["time"] = delay
        if "time" not in parameters:
            raise ConfigError("t1_zpa requires a delay or time sweep")
        return parameters

    def build(self, config):
        plan = super().build(config)
        if set(plan.axes) != {"z_gain", "time"}:
            raise ConfigError("t1_zpa requires native z_gain and time sweeps")
        return replace(
            plan,
            metadata={
                **dict(plan.metadata),
                "preflight": T1_ZPA.preflight,
                "authored_program": T1_ZPA.name,
            },
        )

    def axis_units(self, config, axes):
        defaults = {"z_gain": "a.u.", "time": "us"}
        configured = super().axis_units(config, axes)
        return {
            name: configured.get(name) or defaults.get(name, "")
            for name in axes
        }

    def analyze(self, data, config):
        return None


EXPERIMENT = T1ZPA()
