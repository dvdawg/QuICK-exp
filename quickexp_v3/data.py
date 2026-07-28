"""Backend-neutral acquired data and analysis result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .errors import ExperimentError
from .util import to_builtin


@dataclass(frozen=True)
class ExperimentData:
    """Named axes and signals decoded from one backend acquisition."""

    axes: Mapping[str, np.ndarray]
    signals: Mapping[str, np.ndarray]
    units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_axes = {
            str(name): np.asarray(value) for name, value in self.axes.items()
        }
        normalized_signals = {
            str(name): np.asarray(value) for name, value in self.signals.items()
        }
        if not normalized_signals:
            raise ExperimentError("decoded acquisition has no signals")
        sizes = {
            int(value.shape[0])
            for value in list(normalized_axes.values()) + list(normalized_signals.values())
            if value.ndim >= 1
        }
        if len(sizes) > 1:
            raise ExperimentError(
                f"decoded columns have inconsistent leading dimensions: {sorted(sizes)}"
            )
        object.__setattr__(self, "axes", normalized_axes)
        object.__setattr__(self, "signals", normalized_signals)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def points(self) -> int:
        first = next(iter(self.signals.values()))
        return int(first.shape[0]) if first.ndim else 1

    @property
    def iq(self) -> np.ndarray:
        if "i" not in self.signals or "q" not in self.signals:
            raise ExperimentError("this acquisition has no i/q signal pair")
        return np.asarray(self.signals["i"]) + 1j * np.asarray(self.signals["q"])

    def arrays(self) -> dict:
        result = {f"axis__{key}": value for key, value in self.axes.items()}
        result.update({f"signal__{key}": value for key, value in self.signals.items()})
        return result

    def summary(self) -> dict:
        return {
            "points": self.points,
            "axes": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.axes.items()
            },
            "signals": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.signals.items()
            },
            "units": dict(self.units),
            "metadata": to_builtin(self.metadata),
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Serializable fit/QC result produced by an experiment module."""

    model: str
    values: Mapping[str, Any]
    quality: Mapping[str, Any] = field(default_factory=dict)
    valid: bool = True
    recommendation: Optional[Mapping[str, Any]] = None

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "values": to_builtin(self.values),
            "quality": to_builtin(self.quality),
            "valid": bool(self.valid),
            "recommendation": (
                to_builtin(self.recommendation)
                if self.recommendation is not None
                else None
            ),
        }


@dataclass(frozen=True)
class BackendResult:
    """Raw backend payload plus a snapshot of its execution metadata."""

    payload: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

