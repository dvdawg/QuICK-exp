"""Adapter for Ramsey cryoscope acquisition."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..errors import ConfigError
from ..programs import CRYOSCOPE
from .base import Experiment


class Cryoscope(Experiment):
    name = "cryoscope"
    quick_class = CRYOSCOPE.name
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
        "z_sigma",
        "flux_time",
        "ramsey_phase",
    )
    axis_candidates = ("flux_time", "ramsey_phase")
    default_run_options = {"silent": True, "dB": False, "population": False}

    def build(self, config):
        plan = super().build(config)
        if plan.axes != ("flux_time", "ramsey_phase"):
            raise ConfigError(
                "cryoscope requires flux_time as the outer axis and "
                "ramsey_phase as the inner axis"
            )
        minimum_time = float(np.min(np.asarray(plan.variables["flux_time"])))
        sigma = float(plan.variables["z_sigma"])
        phases = np.asarray(plan.variables["ramsey_phase"], dtype=float).ravel()
        if phases.size < 4 or np.unique(np.mod(phases, 360.0)).size < 4:
            raise ConfigError(
                "cryoscope requires at least four distinct Ramsey phases"
            )
        if sigma <= 0 or minimum_time < 8.0 * sigma:
            raise ConfigError(
                "cryoscope flux_time must be at least 8*z_sigma so the "
                "Gaussian-smoothed flat-top pulse has complete ramps"
            )
        return replace(
            plan,
            metadata={
                **dict(plan.metadata),
                "preflight": CRYOSCOPE.preflight,
                "authored_program": CRYOSCOPE.name,
            },
        )

    def axis_units(self, config, axes):
        defaults = {"flux_time": "us", "ramsey_phase": "deg"}
        configured = super().axis_units(config, axes)
        return {
            name: configured.get(name) or defaults.get(name, "")
            for name in axes
        }


EXPERIMENT = Cryoscope()
