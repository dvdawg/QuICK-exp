"""Energy-relaxation (T1) experiment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..analysis import fit_exponential_decay
from ..errors import ConfigError
from .base import Experiment


class T1(Experiment):
    name = "t1"
    quick_class = "T1"
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
    axis_candidates = ("time",)
    default_run_options = {"silent": True, "population": True}

    def _parameters(self, config):
        parameters = super()._parameters(config)
        delay = parameters.pop("delay", None)
        if delay is not None:
            if "time" in parameters and isinstance(parameters["time"], np.ndarray):
                raise ConfigError("t1 cannot declare both delay and time sweeps")
            parameters["time"] = delay
        if "time" not in parameters:
            raise ConfigError("t1 requires a delay or time sweep")
        return parameters

    def axis_units(self, config, axes):
        return {"time": "us"}

    def analyze(self, data, config):
        result = fit_exponential_decay(data.axes["time"], data.iq)
        return self.validate_analysis(
            replace(
                result,
                recommendation={"derived.t1": float(result.values["decay"])},
            ),
            config,
        )


EXPERIMENT = T1()



