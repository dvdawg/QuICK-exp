"""Resonator spectroscopy experiment and resonator-frequency analysis."""

from __future__ import annotations

from dataclasses import replace

from ..analysis import fit_spectral_feature
from ..data import AnalysisResult
from .base import Experiment


class ResonatorSpectroscopy(Experiment):
    name = "resonator_spectroscopy"
    quick_class = "ResonatorSpectroscopy"
    required = ("r_freq", "r_power", "r_length", "r_offset", "r_relax")
    axis_candidates = ("r_power", "r_freq")
    default_run_options = {"silent": True, "dB": True}

    def axis_units(self, config, axes):
        defaults = {"r_freq": "MHz", "r_power": "dBFS"}
        configured = super().axis_units(config, axes)
        return {name: configured.get(name) or defaults.get(name, "") for name in axes}

    def analyze(self, data, config):
        if "r_freq" not in data.axes:
            return None
        # A two-dimensional power/frequency run is intentionally persisted raw;
        # row-wise branch fitting belongs in a dedicated power-scan module.
        if len(data.axes) != 1:
            return None
        result = fit_spectral_feature(data.axes["r_freq"], data.iq)
        center = float(result.values["center"])
        return self.validate_analysis(
            replace(
                result,
                recommendation={"defaults.r_freq": center},
            ),
            config,
        )


EXPERIMENT = ResonatorSpectroscopy()



