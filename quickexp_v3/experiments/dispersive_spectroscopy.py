"""Readout spectroscopy with qubit-state preparation."""

from __future__ import annotations

import numpy as np

from ..analysis import feature_extremum
from .base import Experiment


class DispersiveSpectroscopy(Experiment):
    name = "dispersive_spectroscopy"
    quick_class = "DispersiveSpectroscopy"
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
    axis_candidates = ("r_freq",)
    default_signal_names = (
        "amplitude_ground",
        "phase_ground",
        "i_ground",
        "q_ground",
        "amplitude_excited",
        "phase_excited",
        "i_excited",
        "q_excited",
    )
    default_run_options = {"silent": True, "dB": False, "population": False}

    def signal_names(self, config):
        return self.default_signal_names

    def axis_units(self, config, axes):
        return {"r_freq": "MHz"}

    def analyze(self, data, config):
        separation = (
            data.signals["i_excited"] + 1j * data.signals["q_excited"]
            - data.signals["i_ground"] - 1j * data.signals["q_ground"]
        )
        return self.validate_analysis(
            feature_extremum(
                data.axes["r_freq"],
                np.abs(separation),
                "maximum_readout_contrast_candidate",
            ),
            config,
        )


EXPERIMENT = DispersiveSpectroscopy()



