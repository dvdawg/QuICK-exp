"""Ground/excited single-shot IQ acquisition and threshold analysis."""

from __future__ import annotations

from ..analysis import fit_iq_classifier
from .base import Experiment


class IQBlobs(Experiment):
    name = "iq_blobs"
    quick_class = "IQScatter"
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
    axis_candidates = ()
    default_signal_names = ("i_ground", "q_ground", "i_excited", "q_excited")
    default_run_options = {"silent": True}

    def _parameters(self, config):
        parameters = super()._parameters(config)
        shots = parameters.pop("shots", None)
        if shots is not None:
            parameters["rep"] = int(shots)
        return parameters

    def signal_names(self, config):
        return self.default_signal_names

    def analyze(self, data, config):
        result = fit_iq_classifier(
            data.signals["i_ground"],
            data.signals["q_ground"],
            data.signals["i_excited"],
            data.signals["q_excited"],
        )
        return self.validate_analysis(result, config)


EXPERIMENT = IQBlobs()



