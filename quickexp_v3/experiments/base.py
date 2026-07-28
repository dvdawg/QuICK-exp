"""Contract shared by every explicit experiment module."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..backend import logical_channel_variables
from ..config import ResolvedConfig
from ..data import AnalysisResult, BackendResult, ExperimentData
from ..errors import ConfigError, ExperimentError


@dataclass(frozen=True)
class ExperimentPlan:
    """Fully resolved request passed to an acquisition backend."""

    name: str
    quick_class: str
    title: str
    variables: Mapping[str, Any]
    axes: Tuple[str, ...]
    signal_names: Tuple[str, ...]
    run_options: Mapping[str, Any] = field(default_factory=dict)
    axis_units: Mapping[str, str] = field(default_factory=dict)
    signal_units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_variables(self, **changes: Any) -> "ExperimentPlan":
        variables = deepcopy(dict(self.variables))
        variables.update(changes)
        return replace(self, variables=variables)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "quick_class": self.quick_class,
            "title": self.title,
            "variables": deepcopy(dict(self.variables)),
            "axes": list(self.axes),
            "signal_names": list(self.signal_names),
            "run_options": deepcopy(dict(self.run_options)),
            "axis_units": dict(self.axis_units),
            "signal_units": dict(self.signal_units),
            "metadata": deepcopy(dict(self.metadata)),
        }


class Experiment:
    """Base behavior; subclasses own build and analysis semantics."""

    name = ""
    quick_class = ""
    required: Tuple[str, ...] = ()
    axis_candidates: Tuple[str, ...] = ()
    default_signal_names: Tuple[str, ...] = ("amplitude", "phase", "i", "q")
    default_run_options: Mapping[str, Any] = {"silent": True}

    def _parameters(self, config: ResolvedConfig) -> dict:
        if config.experiment != self.name:
            raise ConfigError(
                f"resolved preset targets {config.experiment!r}, not {self.name!r}"
            )
        parameters = config.expanded_parameters()
        missing = [
            key for key in self.required
            if key not in parameters or parameters[key] is None
        ]
        if missing:
            raise ConfigError(
                f"{self.name} is missing required parameters: {', '.join(missing)}"
            )
        parameters.update(logical_channel_variables(config.hardware))
        return parameters

    def _axes(self, parameters: Mapping[str, Any]) -> Tuple[str, ...]:
        candidates = set(self.axis_candidates)
        axes = []
        # Quick builds its sweep list from variable insertion order. Preserve
        # that order so decoded columns are named exactly as Quick emitted them.
        for name, value in parameters.items():
            if (
                name in candidates
                and isinstance(value, np.ndarray)
                and value.ndim == 1
                and value.size > 1
            ):
                axes.append(name)
        if not axes and self.axis_candidates:
            raise ConfigError(
                f"{self.name} requires a swept parameter among "
                + ", ".join(self.axis_candidates)
            )
        return tuple(axes)

    def _run_options(self, config: ResolvedConfig) -> dict:
        allowed = {"silent", "dB", "population"}
        options = dict(self.default_run_options)
        options.update(
            {
                key: value
                for key, value in config.run.items()
                if key in allowed
            }
        )
        return options

    def build(self, config: ResolvedConfig) -> ExperimentPlan:
        parameters = self._parameters(config)
        axes = self._axes(parameters)
        signal_names = self.signal_names(config)
        units = self.axis_units(config, axes)
        return ExperimentPlan(
            name=self.name,
            quick_class=self.quick_class,
            title=config.preset_name or self.name,
            variables=parameters,
            axes=axes,
            signal_names=signal_names,
            run_options=self._run_options(config),
            axis_units=units,
            signal_units=self.signal_units(config, signal_names),
            metadata={
                "preset": config.preset_name,
                "config_fingerprint": config.config_fingerprint,
            },
        )

    def signal_names(self, config: ResolvedConfig) -> Tuple[str, ...]:
        if bool(self._run_options(config).get("population", False)):
            return ("population",) + tuple(self.default_signal_names)
        return tuple(self.default_signal_names)

    def axis_units(
        self,
        config: ResolvedConfig,
        axes: Sequence[str],
    ) -> dict:
        configured = config.run.get("axis_units", {})
        configured = configured if isinstance(configured, Mapping) else {}
        return {name: str(configured.get(name, "")) for name in axes}

    def signal_units(
        self,
        config: ResolvedConfig,
        signals: Sequence[str],
    ) -> dict:
        configured = config.run.get("signal_units", {})
        configured = configured if isinstance(configured, Mapping) else {}
        result = {name: str(configured.get(name, "")) for name in signals}
        if "phase" in result and not result["phase"]:
            result["phase"] = "rad"
        if "amplitude" in result and bool(self._run_options(config).get("dB", False)):
            result["amplitude"] = "dB"
        return result

    def decode(self, plan: ExperimentPlan, result: BackendResult) -> ExperimentData:
        matrix = np.asarray(result.payload)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ExperimentError(
                f"{self.name} expected a two-dimensional result, got {matrix.shape}"
            )
        expected_columns = len(plan.axes) + len(plan.signal_names)
        if matrix.shape[1] != expected_columns:
            raise ExperimentError(
                f"{self.name} expected {expected_columns} columns "
                f"({list(plan.axes) + list(plan.signal_names)}), got "
                f"{matrix.shape[1]}"
            )
        axes = {
            name: matrix[:, index]
            for index, name in enumerate(plan.axes)
        }
        signals = {
            name: matrix[:, len(plan.axes) + index]
            for index, name in enumerate(plan.signal_names)
        }
        units = dict(plan.axis_units)
        units.update(plan.signal_units)
        metadata = dict(result.metadata)
        metadata.update(plan.metadata)
        return ExperimentData(
            axes=axes,
            signals=signals,
            units=units,
            metadata=metadata,
        )

    def analyze(
        self,
        data: ExperimentData,
        config: ResolvedConfig,
    ) -> Optional[AnalysisResult]:
        return None

    def validate_analysis(
        self,
        result: AnalysisResult,
        config: ResolvedConfig,
    ) -> AnalysisResult:
        quality = dict(result.quality)
        valid = bool(result.valid)
        r2_min = config.acceptance.get("r2_min")
        if r2_min is not None and "r_squared" in quality:
            passed = float(quality["r_squared"]) >= float(r2_min)
            quality["r2_pass"] = passed
            valid = valid and passed
        fidelity_min = config.acceptance.get("fidelity_min")
        if fidelity_min is not None and "fidelity" in result.values:
            passed = float(result.values["fidelity"]) >= float(fidelity_min)
            quality["fidelity_pass"] = passed
            valid = valid and passed
        return AnalysisResult(
            model=result.model,
            values=result.values,
            quality=quality,
            valid=valid,
            recommendation=result.recommendation,
        )

