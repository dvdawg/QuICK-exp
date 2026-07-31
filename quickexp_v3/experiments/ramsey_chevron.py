"""Ramsey idle-time versus drive-frequency chevron."""

from __future__ import annotations

from ..errors import ConfigError
from .base import Experiment
from .ramsey import Ramsey


class RamseyChevron(Ramsey):
    name = "ramsey_chevron"
    axis_candidates = ("time", "q_freq")

    def build(self, config):
        plan = Experiment.build(self, config)
        if set(plan.axes) != {"q_freq", "time"}:
            raise ConfigError(
                "ramsey_chevron requires q_freq and delay/time sweeps"
            )
        return plan

    def analyze(self, data, config):
        return None


EXPERIMENT = RamseyChevron()

