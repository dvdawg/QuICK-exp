"""Adapter for the resonator-sensed flux-step transient."""

from __future__ import annotations

from dataclasses import replace

from ..errors import ConfigError
from ..programs import RESONATOR_FLUX_TRANSIENT
from .base import Experiment


class ResonatorFluxTransient(Experiment):
    name = "resonator_flux_transient"
    quick_class = RESONATOR_FLUX_TRANSIENT.name
    required = (
        "r_freq",
        "r_power",
        "r_length",
        "r_offset",
        "r_relax",
        "z_step_gain",
        "z_baseline_gain",
        "z_set_length",
        "probe_time",
    )
    # probe_time is the outer loop so a full mini-spectrum is acquired at each
    # observation time while the flux level is held; r_freq steps fastest.
    axis_candidates = ("probe_time", "r_freq")
    # dB=False: the inversion needs linear complex S21, not log magnitude.
    default_run_options = {"silent": True, "dB": False}

    def build(self, config):
        plan = super().build(config)
        if plan.axes not in {
            ("probe_time", "r_freq"),
            ("r_freq",),
        }:
            raise ConfigError(
                "resonator_flux_transient requires r_freq, optionally with "
                "probe_time as the outer axis"
            )
        return replace(
            plan,
            metadata={
                **dict(plan.metadata),
                "preflight": RESONATOR_FLUX_TRANSIENT.preflight,
                "authored_program": RESONATOR_FLUX_TRANSIENT.name,
            },
        )

    def axis_units(self, config, axes):
        defaults = {"probe_time": "us", "r_freq": "MHz"}
        configured = super().axis_units(config, axes)
        return {
            name: configured.get(name) or defaults.get(name, "")
            for name in axes
        }

    def analyze(self, data, config):
        # The transient is inverted against the flux calibration in
        # quickexp_v3.resonator_transient, which needs the whole 2-D map.
        # Per-row spectral fitting here would be misleading.
        return None


EXPERIMENT = ResonatorFluxTransient()
