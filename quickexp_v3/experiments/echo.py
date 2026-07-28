"""Hahn-echo / fixed-cycle CPMG experiment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..analysis import fit_exponential_decay
from ..errors import ConfigError
from .base import Experiment


class Echo(Experiment):
    name = "echo"
    quick_class = "T2Echo"
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
                raise ConfigError("echo cannot declare both delay and time sweeps")
            parameters["time"] = delay
        if "pulse_count" in parameters:
            parameters["cycle"] = int(parameters.pop("pulse_count"))
        parameters.setdefault("cycle", 1)
        if isinstance(parameters["cycle"], np.ndarray):
            raise ConfigError(
                "echo runs one pulse count at a time; use separate presets/runs "
                "for a restart-safe CPMG series"
            )
        if "time" not in parameters:
            raise ConfigError("echo requires a delay or time sweep")
        return parameters

    def axis_units(self, config, axes):
        return {"time": "us"}

    def analyze(self, data, config):
        result = fit_exponential_decay(data.axes["time"], data.iq)
        cycle = int(config.parameters.get("pulse_count", config.parameters.get("cycle", 1)))
        return self.validate_analysis(
            replace(
                result,
                recommendation={
                    f"derived.t2_echo.cycle_{cycle}": float(result.values["decay"])
                },
            ),
            config,
        )


EXPERIMENT = Echo()



