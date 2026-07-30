"""Adapter for native two-tone spectroscopy under a finite Z pulse."""

from __future__ import annotations

from dataclasses import replace

from ..errors import ConfigError
from ..programs import TWO_TONE_ZPA
from .base import Experiment


class TwoToneZPA(Experiment):
    name = "two_tone_zpa"
    quick_class = TWO_TONE_ZPA.name
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
        "z_length",
        "z_settle",
    )
    axis_candidates = ("q_freq", "z_gain")
    default_run_options = {"silent": True, "dB": False, "population": False}

    def build(self, config):
        plan = super().build(config)
        if set(plan.axes) != {"q_freq", "z_gain"}:
            raise ConfigError(
                "two_tone_zpa requires native q_freq and z_gain sweeps"
            )
        return replace(
            plan,
            metadata={
                **dict(plan.metadata),
                "preflight": TWO_TONE_ZPA.preflight,
                "authored_program": TWO_TONE_ZPA.name,
            },
        )

    def axis_units(self, config, axes):
        defaults = {"q_freq": "MHz", "z_gain": "a.u."}
        configured = super().axis_units(config, axes)
        return {
            name: configured.get(name) or defaults.get(name, "")
            for name in axes
        }

    def analyze(self, data, config):
        return None


EXPERIMENT = TwoToneZPA()
