"""Time-of-flight / raw decimated loopback experiment."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..data import AnalysisResult, BackendResult, ExperimentData
from ..errors import ExperimentError
from .base import Experiment, ExperimentPlan


class Loopback(Experiment):
    name = "loopback"
    quick_class = "LoopBack"
    required = ("r_freq", "r_power", "r_length", "r_offset", "r_relax")
    axis_candidates = ()
    default_run_options = {"silent": True, "dB": False}

    def _parameters(self, config):
        parameters = super()._parameters(config)
        # Quick LoopBack acquires decimated traces. Keep hardware repetitions
        # at one so the trace fits the QICK buffer; average complete traces
        # with soft_avg as Quick's installed template does.
        parameters.pop("hard_avg", None)
        parameters.pop("rep", None)
        return parameters

    def build(self, config):
        plan = super().build(config)
        return ExperimentPlan(
            name=plan.name,
            quick_class=plan.quick_class,
            title=plan.title,
            variables=plan.variables,
            axes=("time",),
            signal_names=plan.signal_names,
            run_options=plan.run_options,
            axis_units={"time": "us"},
            signal_units=plan.signal_units,
            metadata=plan.metadata,
        )

    def decode(self, plan: ExperimentPlan, result: BackendResult) -> ExperimentData:
        data = super().decode(plan, result)
        if data.points < 3:
            raise ExperimentError("loopback trace has fewer than three samples")
        return data

    def analyze(self, data, config) -> Optional[AnalysisResult]:
        time = data.axes["time"]
        magnitude = np.abs(data.iq)
        baseline_points = max(3, magnitude.size // 10)
        baseline = float(np.median(magnitude[:baseline_points]))
        peak = float(np.max(magnitude))
        threshold = baseline + 0.25 * (peak - baseline)
        crossings = np.flatnonzero(magnitude >= threshold)
        if crossings.size == 0:
            return AnalysisResult(
                model="pulse_edge_threshold",
                values={"r_offset": None},
                quality={"baseline": baseline, "peak": peak},
                valid=False,
            )
        offset = float(time[int(crossings[0])])
        return AnalysisResult(
            model="pulse_edge_threshold",
            values={"r_offset": offset},
            quality={"baseline": baseline, "peak": peak, "threshold": threshold},
            valid=True,
            recommendation={"defaults.r_offset": offset},
        )


EXPERIMENT = Loopback()



