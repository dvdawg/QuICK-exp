"""Common experiment planning, acquisition retry, decoding, and analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from typing import Any, Mapping, Optional, Union

from .config import ConfigRepository, ResolvedConfig
from .data import AnalysisResult, ExperimentData
from .errors import AcquisitionError, SafetyError
from .experiments.base import Experiment, ExperimentPlan
from .experiments.registry import get as get_experiment


@dataclass(frozen=True)
class PlannedRun:
    experiment: Experiment
    config: ResolvedConfig
    plan: ExperimentPlan

    def as_dict(self) -> dict:
        return {
            "experiment": self.experiment.name,
            "config": self.config.as_manifest(),
            "plan": self.plan.as_dict(),
        }


@dataclass(frozen=True)
class CompletedRun:
    """One decoded acquisition.

    Live data persistence belongs to Quick's native Saver. Any CSV/YML paths
    created by that Saver are copied into ``data.metadata["native_files"]``.
    """

    data: ExperimentData
    analysis: Optional[AnalysisResult]
    status: str = "completed"


class ExperimentRunner:
    """Resolve, acquire with exact retry, decode, and analyze in memory."""

    def __init__(
        self,
        config: ConfigRepository,
        backend: Any,
        *,
        flux: Any = None,
    ):
        self.config = config
        self.backend = backend
        self.flux = flux

    def plan(
        self,
        experiment_name: str,
        preset: Union[str, Mapping[str, Any]],
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
    ) -> PlannedRun:
        experiment = get_experiment(experiment_name)
        resolved = self.config.resolve(
            preset,
            experiment=experiment.name,
            overrides=overrides,
        )
        plan = experiment.build(resolved)
        if title:
            plan = replace(plan, title=str(title))
        return PlannedRun(experiment, resolved, plan)

    def run(
        self,
        experiment_name: str,
        preset: Union[str, Mapping[str, Any]],
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        analyze: bool = True,
        close_backend: bool = True,
        park_flux_on_exit: bool = True,
    ) -> CompletedRun:
        planned = self.plan(
            experiment_name,
            preset,
            overrides=overrides,
            title=title,
        )
        connect = getattr(self.backend, "connect", None)
        if callable(connect):
            connect()
        try:
            result = self._acquire(planned)
            data = planned.experiment.decode(planned.plan, result)
            analysis_result = None
            status = "completed"
            if analyze:
                try:
                    analysis_result = planned.experiment.analyze(
                        data,
                        planned.config,
                    )
                except Exception as error:
                    status = "completed_with_analysis_error"
                    print(
                        "Analysis warning: "
                        f"{type(error).__name__}: {error}"
                    )
            return CompletedRun(data, analysis_result, status)
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_errors = []
            if self.flux is not None and park_flux_on_exit:
                try:
                    self.flux.park()
                except BaseException as cleanup:
                    cleanup_errors.append(cleanup)
            if close_backend:
                closer = getattr(self.backend, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except BaseException as cleanup:
                        cleanup_errors.append(cleanup)
            if cleanup_errors:
                message = "; ".join(str(error) for error in cleanup_errors)
                if primary_error is not None:
                    if hasattr(primary_error, "add_note"):
                        primary_error.add_note(
                            f"run cleanup also failed: {message}"
                        )
                else:
                    raise SafetyError(f"run cleanup failed: {message}")

    def _acquire(self, planned: PlannedRun):
        retry = planned.config.run.get("retry", {})
        if not isinstance(retry, Mapping):
            raise AcquisitionError("run.retry must be a mapping")
        max_attempts = int(retry.get("max_attempts", 1))
        if max_attempts < 1:
            raise AcquisitionError("run.retry.max_attempts must be at least 1")
        configured_max = planned.config.hardware.get("limits", {}).get(
            "maximum_retry_attempts"
        )
        if configured_max is not None:
            maximum = int(configured_max[1])
            if max_attempts > maximum:
                raise AcquisitionError(
                    f"requested {max_attempts} attempts exceeds hardware limit "
                    f"{maximum}"
                )
        last_error: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            if self.flux is not None and not bool(
                getattr(self.flux, "safe_for_acquisition", False)
            ):
                raise SafetyError("flux line is not armed and safe for acquisition")
            try:
                return self.backend.acquire(planned.plan)
            except BaseException as error:
                last_error = error
                if attempt >= max_attempts:
                    break
                print(
                    f"Acquisition attempt {attempt}/{max_attempts} failed; "
                    "running Quick/QICK recovery before retry."
                )
                if self.flux is not None:
                    notify = getattr(self.flux, "notify_generator_reset", None)
                    if callable(notify):
                        notify()
                recover = getattr(self.backend, "recover", None)
                if not callable(recover):
                    break
                recover(error, attempt)
                if self.flux is not None:
                    rearm = getattr(self.flux, "recover_after_reset", None)
                    if callable(rearm):
                        rearm()
        raise AcquisitionError(
            f"{planned.experiment.name} failed after {max_attempts} attempt(s): "
            f"{last_error}"
        ) from last_error
