"""Qubit spectroscopy experiment and signed spectral-feature analysis."""

from __future__ import annotations

from dataclasses import replace

from ..analysis import fit_spectral_feature
from .base import Experiment


class QubitSpectroscopy(Experiment):
    name = "qubit_spectroscopy"
    quick_class = "QubitSpectroscopy"
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
    axis_candidates = ("q_gain", "q_freq")
    default_run_options = {"silent": True, "dB": False, "population": False}

    def axis_units(self, config, axes):
        defaults = {"q_freq": "MHz", "q_gain": "a.u."}
        configured = super().axis_units(config, axes)
        return {name: configured.get(name) or defaults.get(name, "") for name in axes}

    def analyze(self, data, config):
        if "q_freq" not in data.axes or len(data.axes) != 1:
            return None
        result = fit_spectral_feature(data.axes["q_freq"], data.iq)
        return self.validate_analysis(
            replace(
                result,
                recommendation={"defaults.q_freq": float(result.values["center"])},
            ),
            config,
        )


EXPERIMENT = QubitSpectroscopy()



