"""Ramsey dephasing experiment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..analysis import fit_damped_oscillation
from ..errors import ConfigError
from .base import Experiment


class Ramsey(Experiment):
    name = "ramsey"
    quick_class = "T2Ramsey"
    required = (
        "r_freq",
        "r_power",
        "r_length",
        "r_offset",
        "r_relax",
        "q_freq",
        "q_gain",
        "q_length",
        "q_gain_2",
    )
    axis_candidates = ("time",)
    default_run_options = {"silent": True, "population": True}

    def _parameters(self, config):
        parameters = super()._parameters(config)
        delay = parameters.pop("delay", None)
        if delay is not None:
            if "time" in parameters and isinstance(parameters["time"], np.ndarray):
                raise ConfigError("ramsey cannot declare both delay and time sweeps")
            parameters["time"] = delay
        if "fringe_frequency_mhz" in parameters:
            parameters["fringe_freq"] = parameters.pop("fringe_frequency_mhz")
        if "time" not in parameters:
            raise ConfigError("ramsey requires a delay or time sweep")
        return parameters

    def axis_units(self, config, axes):
        return {"time": "us"}

    def analyze(self, data, config):
        hint = config.parameters.get(
            "fringe_frequency_mhz",
            config.parameters.get("fringe_freq"),
        )
        result = fit_damped_oscillation(
            data.axes["time"],
            data.iq,
            frequency_hint=float(hint) if hint is not None else None,
            model_name="decaying_ramsey",
        )
        return self.validate_analysis(
            replace(
                result,
                recommendation={"derived.t2_ramsey": float(result.values["decay"])},
            ),
            config,
        )


EXPERIMENT = Ramsey()



