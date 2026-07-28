"""Two-dimensional Rabi chevron using Quick's native Cartesian sweep."""

from __future__ import annotations

from ..errors import ConfigError
from .base import Experiment
from .rabi import Rabi


class RabiChevron(Rabi):
    name = "rabi_chevron"
    axis_candidates = ("q_freq", "q_gain", "q_length")

    def build(self, config):
        plan = Experiment.build(self, config)
        if len(plan.axes) != 2 or "q_freq" not in plan.axes:
            raise ConfigError(
                "rabi_chevron requires q_freq plus q_gain or q_length sweeps"
            )
        return plan

    def analyze(self, data, config):
        # Ridge/contour analysis should operate on the persisted rectangular
        # grid; do not apply the one-dimensional Rabi fitter.
        return None


EXPERIMENT = RabiChevron()

