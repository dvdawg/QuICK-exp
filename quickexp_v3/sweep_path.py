"""Design, persist, and evaluate row-dependent frequency sweep paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import yaml

from .errors import AnalysisError, ConfigError
from .native_map import NativeMap, load_native_map
from .util import to_builtin, utc_now


SWEEP_PATH_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class SweepPath:
    """One or more inner-axis intervals sampled for each outer-axis row."""

    method: str
    outer_name: str
    inner_name: str
    outer_values: np.ndarray
    lower_inner_values: np.ndarray
    upper_inner_values: np.ndarray
    points_per_row: Optional[int]
    metadata: Mapping[str, Any]
    outer_label: str = ""
    outer_unit: str = ""
    inner_label: str = ""
    inner_unit: str = ""
    inner_resolution: Optional[float] = None
    inner_segments: Optional[Any] = None

    def __post_init__(self):
        outer = np.asarray(self.outer_values, dtype=float)
        lower = np.asarray(self.lower_inner_values, dtype=float)
        upper = np.asarray(self.upper_inner_values, dtype=float)
        points = (
            None
            if self.points_per_row is None
            else int(self.points_per_row)
        )
        resolution = (
            None
            if self.inner_resolution is None
            else float(self.inner_resolution)
        )
        if (
            outer.ndim != 1
            or outer.size == 0
            or lower.shape != outer.shape
            or upper.shape != outer.shape
        ):
            raise ConfigError(
                "sweep path requires matching, non-empty 1D outer values and "
                "inner bounds"
            )
        if not (
            np.all(np.isfinite(outer))
            and np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
        ):
            raise ConfigError("sweep path values must all be finite")
        if outer.size > 1 and not np.all(np.diff(outer) > 0):
            raise ConfigError("sweep path outer values must be strictly increasing")
        if np.any(upper <= lower):
            raise ConfigError(
                "every sweep-path upper inner bound must exceed its lower bound"
            )
        segments = None
        if self.inner_segments is not None:
            try:
                if len(self.inner_segments) != outer.size:
                    raise ConfigError(
                        "sweep-path inner segments must contain one row per "
                        "outer value"
                    )
            except TypeError as error:
                raise ConfigError(
                    "sweep-path inner segments must be a sequence of rows"
                ) from error
            normalized_rows = []
            for row_index, row in enumerate(self.inner_segments):
                intervals = np.asarray(row, dtype=float)
                if (
                    intervals.ndim != 2
                    or intervals.shape[0] == 0
                    or intervals.shape[1] != 2
                    or not np.all(np.isfinite(intervals))
                    or np.any(intervals[:, 1] <= intervals[:, 0])
                ):
                    raise ConfigError(
                        "each sweep-path segment row requires one or more "
                        "finite [lower, upper] intervals"
                    )
                intervals = intervals[np.argsort(intervals[:, 0])]
                merged = []
                for interval_lower, interval_upper in intervals:
                    interval_lower = float(interval_lower)
                    interval_upper = float(interval_upper)
                    if merged:
                        tolerance = max(
                            1.0,
                            abs(merged[-1][1]),
                            abs(interval_lower),
                        ) * 1e-12
                        if interval_lower <= merged[-1][1] + tolerance:
                            merged[-1][1] = max(
                                merged[-1][1],
                                interval_upper,
                            )
                            continue
                    merged.append([interval_lower, interval_upper])
                tolerance = max(
                    1.0,
                    abs(lower[row_index]),
                    abs(upper[row_index]),
                ) * 1e-10
                if (
                    abs(merged[0][0] - lower[row_index]) > tolerance
                    or abs(merged[-1][1] - upper[row_index]) > tolerance
                ):
                    raise ConfigError(
                        "sweep-path lower/upper bounds must match the segment "
                        "envelope for every row"
                    )
                normalized_rows.append(
                    tuple((float(start), float(stop)) for start, stop in merged)
                )
            segments = tuple(normalized_rows)
        if points is not None and points < 2:
            raise ConfigError("sweep path requires at least two points per row")
        if resolution is not None and (
            not np.isfinite(resolution) or resolution <= 0
        ):
            raise ConfigError("sweep-path inner resolution must be positive and finite")
        if points is None and resolution is None:
            raise ConfigError(
                "sweep path requires either points_per_row or inner_resolution"
            )
        if points is not None and resolution is not None:
            raise ConfigError(
                "sweep path cannot set both points_per_row and inner_resolution"
            )
        method = str(self.method).strip()
        if not method:
            raise ConfigError("sweep path method cannot be empty")
        outer_name = str(self.outer_name).strip()
        inner_name = str(self.inner_name).strip()
        if not outer_name or not inner_name or outer_name == inner_name:
            raise ConfigError(
                "sweep path requires distinct, non-empty outer and inner names"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "outer_name", outer_name)
        object.__setattr__(self, "inner_name", inner_name)
        object.__setattr__(self, "outer_values", outer.copy())
        object.__setattr__(self, "lower_inner_values", lower.copy())
        object.__setattr__(self, "upper_inner_values", upper.copy())
        object.__setattr__(self, "points_per_row", points)
        object.__setattr__(self, "inner_resolution", resolution)
        object.__setattr__(self, "inner_segments", segments)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def center_inner_values(self) -> np.ndarray:
        return 0.5 * (self.lower_inner_values + self.upper_inner_values)

    @property
    def total_points(self) -> int:
        return int(sum(row.size for row in self.inner_sweeps()))

    @property
    def point_counts(self) -> np.ndarray:
        return np.asarray([row.size for row in self.inner_sweeps()], dtype=int)

    @property
    def has_disjoint_intervals(self) -> bool:
        return bool(
            self.inner_segments is not None
            and any(len(row) > 1 for row in self.inner_segments)
        )

    def inner_intervals(self, outer_value: float) -> tuple[tuple[float, float], ...]:
        """Return all selected inner-axis intervals for one outer row."""
        value = float(outer_value)
        if not np.isfinite(value):
            raise ConfigError("requested sweep-path outer value must be finite")
        minimum = float(self.outer_values[0])
        maximum = float(self.outer_values[-1])
        tolerance = max(1.0, abs(minimum), abs(maximum)) * 1e-12
        if value < minimum - tolerance or value > maximum + tolerance:
            raise ConfigError(
                f"requested {self.outer_name}={value} is outside sweep-path domain "
                f"[{minimum}, {maximum}]"
            )
        value = float(np.clip(value, minimum, maximum))
        if self.inner_segments is not None:
            differences = np.abs(self.outer_values - value)
            row_index = int(np.argmin(differences))
            if differences[row_index] > tolerance:
                raise ConfigError(
                    "segmented sweep paths are defined only at their saved "
                    f"{self.outer_name} rows"
                )
            return self.inner_segments[row_index]
        lower = float(
            np.interp(value, self.outer_values, self.lower_inner_values)
        )
        upper = float(
            np.interp(value, self.outer_values, self.upper_inner_values)
        )
        return ((lower, upper),)

    def inner_sweep(self, outer_value: float) -> np.ndarray:
        """Return only the selected inner values for one outer-axis row."""
        intervals = self.inner_intervals(outer_value)
        if self.inner_resolution is not None:
            sampled = []
            for lower, upper in intervals:
                ratio = (upper - lower) / self.inner_resolution
                tolerance = max(1.0, abs(ratio)) * 1e-12
                count = int(np.floor(ratio + tolerance)) + 1
                if count < 2:
                    raise ConfigError(
                        "a sweep-path interval is narrower than the configured "
                        "inner resolution"
                    )
                sampled.append(
                    lower
                    + self.inner_resolution * np.arange(count, dtype=float)
                )
            return np.concatenate(sampled)
        interval_count = len(intervals)
        if self.points_per_row < 2 * interval_count:
            raise ConfigError(
                "points_per_row must provide at least two points for every "
                "disjoint interval"
            )
        counts = np.full(interval_count, 2, dtype=int)
        remaining = self.points_per_row - int(np.sum(counts))
        if remaining:
            widths = np.asarray([upper - lower for lower, upper in intervals])
            shares = remaining * widths / float(np.sum(widths))
            extras = np.floor(shares).astype(int)
            counts += extras
            leftover = remaining - int(np.sum(extras))
            if leftover:
                order = np.argsort(-(shares - extras))
                counts[order[:leftover]] += 1
        return np.concatenate(
            [
                np.linspace(lower, upper, int(count))
                for (lower, upper), count in zip(intervals, counts)
            ]
        )

    def inner_sweeps(self) -> tuple[np.ndarray, ...]:
        return tuple(self.inner_sweep(value) for value in self.outer_values)

    def inner_matrix(self) -> np.ndarray:
        rows = self.inner_sweeps()
        if len({row.size for row in rows}) != 1:
            raise ConfigError(
                "resolution-based sweep path has variable row lengths; "
                "use inner_sweeps()"
            )
        return np.vstack(rows)

    def as_dict(self) -> dict:
        return {
            "schema_version": SWEEP_PATH_SCHEMA_VERSION,
            "kind": "row_dependent_sweep_path",
            "method": self.method,
            "outer": {
                "name": self.outer_name,
                "label": self.outer_label,
                "unit": self.outer_unit,
                "values": self.outer_values.tolist(),
            },
            "inner": {
                "name": self.inner_name,
                "label": self.inner_label,
                "unit": self.inner_unit,
                "lower": self.lower_inner_values.tolist(),
                "upper": self.upper_inner_values.tolist(),
                "points_per_row": self.points_per_row,
                "resolution": self.inner_resolution,
                "segments": (
                    None
                    if self.inner_segments is None
                    else [
                        [[lower, upper] for lower, upper in row]
                        for row in self.inner_segments
                    ]
                ),
            },
            "metadata": to_builtin(dict(self.metadata)),
        }


class FrequencySweepPath(SweepPath):
    """Backward-compatible Z-gain/qubit-frequency sweep path."""

    def __init__(
        self,
        method,
        z_gain,
        lower_frequency_mhz,
        upper_frequency_mhz,
        points_per_row,
        metadata,
        inner_resolution_mhz=None,
        inner_segments=None,
    ):
        super().__init__(
            method=method,
            outer_name="z_gain",
            inner_name="q_freq",
            outer_values=z_gain,
            lower_inner_values=lower_frequency_mhz,
            upper_inner_values=upper_frequency_mhz,
            points_per_row=points_per_row,
            metadata=metadata,
            outer_label="Z gain",
            outer_unit="a.u.",
            inner_label="Qubit frequency",
            inner_unit="MHz",
            inner_resolution=inner_resolution_mhz,
            inner_segments=inner_segments,
        )

    @property
    def z_gain(self):
        return self.outer_values

    @property
    def lower_frequency_mhz(self):
        return self.lower_inner_values

    @property
    def upper_frequency_mhz(self):
        return self.upper_inner_values

    @property
    def center_frequency_mhz(self):
        return self.center_inner_values

    def frequency_sweep(self, z_gain):
        return self.inner_sweep(z_gain)

    def frequency_matrix(self):
        return self.inner_matrix()


def save_sweep_path(
    path: SweepPath,
    output_path: Path,
) -> Path:
    """Atomically write a reusable sweep path as YAML."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(path.as_dict(), handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def save_frequency_sweep_path(path: SweepPath, output_path: Path) -> Path:
    """Backward-compatible alias for :func:`save_sweep_path`."""
    return save_sweep_path(path, output_path)


def load_sweep_path(path: Path) -> SweepPath:
    """Load and validate a saved frequency sweep path."""
    source = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"could not read sweep path {source}: {error}") from error
    if not isinstance(document, Mapping):
        raise ConfigError(f"sweep path {source} must contain a YAML mapping")
    try:
        schema_version = int(document.get("schema_version", -1))
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid sweep-path schema version in {source}") from error
    if schema_version not in {1, 2, 3, SWEEP_PATH_SCHEMA_VERSION}:
        raise ConfigError(
            f"unsupported sweep-path schema version in {source}: "
            f"{document.get('schema_version')!r}"
        )
    try:
        metadata = dict(document.get("metadata") or {})
        metadata.setdefault("loaded_from", str(source))
        if schema_version == 1:
            if document.get("kind") != "qubit_frequency_vs_flux":
                raise ConfigError(f"unsupported legacy sweep-path kind in {source}")
            return FrequencySweepPath(
                method=document["method"],
                z_gain=document["z_gain"],
                lower_frequency_mhz=document["lower_frequency_mhz"],
                upper_frequency_mhz=document["upper_frequency_mhz"],
                points_per_row=document["points_per_row"],
                metadata=metadata,
            )
        if document.get("kind") != "row_dependent_sweep_path":
            raise ConfigError(f"unsupported sweep-path kind in {source}")
        outer = document["outer"]
        inner = document["inner"]
        outer_values = list(outer["values"])
        lower_values = list(inner["lower"])
        upper_values = list(inner["upper"])
        inner_segments = inner.get("segments")
        polygon_vertices = metadata.get("polygon_vertices")
        if inner_segments is None and polygon_vertices is not None:
            polygon = np.asarray(polygon_vertices, dtype=float)
            reconstructed = []
            resolution = inner.get("resolution")
            for outer_value in outer_values:
                intervals = list(
                    _polygon_intervals_at_outer(polygon, float(outer_value))
                )
                if resolution is not None:
                    intervals = [
                        (lower, upper)
                        for lower, upper in intervals
                        if upper - lower
                        + max(1.0, abs(lower), abs(upper)) * 1e-12
                        >= float(resolution)
                    ]
                if not intervals:
                    reconstructed = []
                    break
                reconstructed.append(intervals)
            if len(reconstructed) == len(outer_values):
                inner_segments = reconstructed
                lower_values = [row[0][0] for row in reconstructed]
                upper_values = [row[-1][1] for row in reconstructed]
                metadata["segments_reconstructed_from_polygon"] = True
        return SweepPath(
            method=document["method"],
            outer_name=outer["name"],
            inner_name=inner["name"],
            outer_values=outer_values,
            lower_inner_values=lower_values,
            upper_inner_values=upper_values,
            points_per_row=inner["points_per_row"],
            metadata=metadata,
            outer_label=outer.get("label", ""),
            outer_unit=outer.get("unit", ""),
            inner_label=inner.get("label", ""),
            inner_unit=inner.get("unit", ""),
            inner_resolution=inner.get("resolution"),
            inner_segments=inner_segments,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"invalid sweep path {source}: {error}") from error


def load_frequency_sweep_path(path: Path) -> FrequencySweepPath:
    """Load a Z-gain/qubit-frequency path with its legacy convenience API."""
    loaded = load_sweep_path(path)
    if isinstance(loaded, FrequencySweepPath):
        return loaded
    if loaded.outer_name != "z_gain" or loaded.inner_name != "q_freq":
        raise ConfigError(
            "frequency sweep path must use z_gain as its outer axis and "
            "q_freq as its inner axis"
        )
    return FrequencySweepPath(
        method=loaded.method,
        z_gain=loaded.outer_values,
        lower_frequency_mhz=loaded.lower_inner_values,
        upper_frequency_mhz=loaded.upper_inner_values,
        points_per_row=loaded.points_per_row,
        metadata=loaded.metadata,
        inner_resolution_mhz=loaded.inner_resolution,
        inner_segments=loaded.inner_segments,
    )


def _margin_pair(margin_mhz: Any) -> tuple[float, float]:
    if np.isscalar(margin_mhz):
        below = above = float(margin_mhz)
    else:
        try:
            if len(margin_mhz) != 2:
                raise ConfigError("fit margin must be one value or (below, above)")
            below, above = map(float, margin_mhz)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "fit margin must be one finite value or a finite (below, above) pair"
            ) from error
    if not np.isfinite(below) or not np.isfinite(above) or below < 0 or above < 0:
        raise ConfigError("fit margins must be finite and non-negative")
    if below + above <= 0:
        raise ConfigError("at least one fit margin must be positive")
    return below, above


def sweep_path_from_center(
    *,
    outer_name: str,
    inner_name: str,
    outer_values: Sequence[float],
    center_values: Any,
    margin: Any,
    points_per_row: Optional[int] = 31,
    inner_resolution: Optional[float] = None,
    method: str = "fit_margin",
    metadata: Optional[Mapping[str, Any]] = None,
    outer_label: str = "",
    outer_unit: str = "",
    inner_label: str = "",
    inner_unit: str = "",
) -> SweepPath:
    """Build a generic corridor around sampled or callable center values."""
    outer = np.asarray(outer_values, dtype=float)
    if outer.ndim != 1 or outer.size == 0 or not np.all(np.isfinite(outer)):
        raise ConfigError("path outer values must be a finite, non-empty 1D sequence")
    outer = np.unique(outer)
    center_source: Callable = center_values if callable(center_values) else None
    center = np.asarray(
        center_source(outer) if center_source is not None else center_values,
        dtype=float,
    )
    if center.shape != outer.shape or not np.all(np.isfinite(center)):
        raise ConfigError("path center must contain one finite value per outer row")
    below, above = _margin_pair(margin)
    details = dict(metadata or {})
    details.update(
        {
            "created_at": utc_now(),
            "margin_below": below,
            "margin_above": above,
        }
    )
    return SweepPath(
        method=method,
        outer_name=outer_name,
        inner_name=inner_name,
        outer_values=outer,
        lower_inner_values=center - below,
        upper_inner_values=center + above,
        points_per_row=points_per_row,
        metadata=details,
        outer_label=outer_label,
        outer_unit=outer_unit,
        inner_label=inner_label,
        inner_unit=inner_unit,
        inner_resolution=inner_resolution,
    )


def sweep_path_from_native_ridge(
    background_csv: Path,
    *,
    margin: Any = 10.0,
    points_per_row: Optional[int] = None,
    inner_resolution: Optional[float] = None,
    outer_values: Optional[Sequence[float]] = None,
    frequency_window: Optional[Sequence[float]] = None,
    minimum_row_r_squared: float = 0.05,
    minimum_row_contrast_snr: float = 1.0,
) -> SweepPath:
    """Fit a spectral ridge row by row and build a margin around it."""
    from .qubit_flux_fit import _extract_row

    native = load_native_map(background_csv)
    if "freq" not in native.inner_label.lower():
        raise AnalysisError(
            "fitted sweep-path backgrounds require frequency as the inner axis"
        )
    frequency_mask = np.ones(native.inner.shape, dtype=bool)
    if frequency_window is not None:
        try:
            if len(frequency_window) != 2:
                raise AnalysisError("frequency fit window must contain two values")
            lower, upper = sorted(float(value) for value in frequency_window)
        except (TypeError, ValueError) as error:
            raise AnalysisError(
                "frequency fit window must contain two finite values"
            ) from error
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise AnalysisError(
                "frequency fit window must contain two finite values"
            )
        frequency_mask = (native.inner >= lower) & (native.inner <= upper)
        if np.count_nonzero(frequency_mask) < 12:
            raise AnalysisError(
                "frequency fit window must contain at least twelve points"
            )
    frequencies = native.inner[frequency_mask]
    iq_map = native.complex_signal[:, frequency_mask]
    fitted_outer = []
    fitted_center = []
    fitted_rows = []
    for row_index, outer_value in enumerate(native.outer):
        extracted = _extract_row(frequencies, iq_map[row_index])
        if extracted is None:
            continue
        center, uncertainty, row_r2, contrast, _projected = extracted
        if (
            row_r2 < float(minimum_row_r_squared)
            or contrast < float(minimum_row_contrast_snr)
        ):
            continue
        fitted_outer.append(float(outer_value))
        fitted_center.append(float(center))
        fitted_rows.append(
            {
                "outer_value": float(outer_value),
                "center": float(center),
                "uncertainty": float(uncertainty),
                "r_squared": float(row_r2),
                "contrast_snr": float(contrast),
            }
        )
    if len(fitted_outer) < 2:
        raise AnalysisError("fewer than two spectral-ridge rows could be fitted")
    fitted_outer = np.asarray(fitted_outer, dtype=float)
    fitted_center = np.asarray(fitted_center, dtype=float)
    requested_outer = (
        native.outer
        if outer_values is None
        else np.asarray(outer_values, dtype=float)
    )
    if (
        requested_outer.ndim != 1
        or requested_outer.size == 0
        or not np.all(np.isfinite(requested_outer))
    ):
        raise ConfigError("path outer values must be finite and non-empty")
    requested_outer = np.unique(requested_outer)
    supported = requested_outer[
        (requested_outer >= fitted_outer[0])
        & (requested_outer <= fitted_outer[-1])
    ]
    if supported.size == 0:
        raise AnalysisError("requested path rows do not overlap the fitted ridge")
    if points_per_row is None and inner_resolution is None:
        inner_resolution = float(np.median(np.diff(native.inner)))
    centers = np.interp(supported, fitted_outer, fitted_center)
    return sweep_path_from_center(
        outer_name=_parameter_name(native.outer_label),
        inner_name=_parameter_name(native.inner_label),
        outer_values=supported,
        center_values=centers,
        margin=margin,
        points_per_row=points_per_row,
        inner_resolution=inner_resolution,
        method="fit_margin",
        metadata={
            "source_csv": str(native.source_csv),
            "ridge_rows": fitted_rows,
            "background_inner_resolution": float(
                np.median(np.diff(native.inner))
            ),
            "frequency_fit_window": [
                float(frequencies.min()),
                float(frequencies.max()),
            ],
        },
        outer_label=native.outer_label,
        outer_unit=native.outer_unit,
        inner_label=native.inner_label,
        inner_unit=native.inner_unit,
    )


def frequency_sweep_path_from_fit(
    fit: Any,
    *,
    z_gain: Optional[Sequence[float]] = None,
    margin_mhz: Any = 10.0,
    points_per_row: Optional[int] = 31,
    inner_resolution_mhz: Optional[float] = None,
) -> FrequencySweepPath:
    """Generate a row-dependent sweep corridor around a fitted flux curve."""
    source_z = getattr(fit, "map_z_gain", fit.z_gain)
    z = np.asarray(source_z if z_gain is None else z_gain, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ConfigError("fit-path Z values must be a finite, non-empty 1D sequence")
    z = np.unique(z)
    center = np.asarray(fit.frequency(z), dtype=float)
    if center.shape != z.shape or not np.all(np.isfinite(center)):
        raise AnalysisError("fitted curve did not return one finite frequency per Z")
    below, above = _margin_pair(margin_mhz)
    return FrequencySweepPath(
        method="fit_margin",
        z_gain=z,
        lower_frequency_mhz=center - below,
        upper_frequency_mhz=center + above,
        points_per_row=points_per_row,
        metadata={
            "created_at": utc_now(),
            "source_csv": str(fit.source_csv),
            "fit_parameters": dict(fit.parameters),
            "fit_statistics": dict(fit.statistics),
            "margin_below_mhz": below,
            "margin_above_mhz": above,
        },
        inner_resolution_mhz=inner_resolution_mhz,
    )


def _polygon_intervals_at_outer(
    vertices: np.ndarray,
    outer_value: float,
) -> tuple[tuple[float, float], ...]:
    """Return all even-odd filled intervals in one vertical polygon slice."""
    value = float(outer_value)
    outer_coordinates = vertices[:, 0]
    scale = max(1.0, float(np.max(np.abs(outer_coordinates))), abs(value))
    tolerance = scale * 1e-10
    minimum = float(np.min(outer_coordinates))
    maximum = float(np.max(outer_coordinates))
    if value < minimum - tolerance or value > maximum + tolerance:
        return ()

    # A half-open edge test avoids double-counting vertices. When a requested
    # row lands exactly on a polygon vertex, choose the immediately adjacent
    # interior topology while evaluating the interval endpoints at the actual
    # requested outer coordinate.
    probe = value
    if np.any(np.abs(outer_coordinates - value) <= tolerance):
        direction = -1.0 if value >= maximum - tolerance else 1.0
        probe = value + direction * tolerance

    intersections = []
    closed = np.vstack((vertices, vertices[0]))
    for start, stop in zip(closed[:-1], closed[1:]):
        x0, y0 = map(float, start)
        x1, y1 = map(float, stop)
        if abs(x1 - x0) <= tolerance:
            continue
        crosses = (x0 <= probe < x1) or (x1 <= probe < x0)
        if not crosses:
            continue
        fraction = float(np.clip((value - x0) / (x1 - x0), 0.0, 1.0))
        intersections.append(y0 + fraction * (y1 - y0))
    if len(intersections) < 2:
        return ()
    intersections.sort()
    groups = [[float(intersections[0])]]
    for candidate in intersections[1:]:
        candidate = float(candidate)
        inner_tolerance = max(1.0, abs(candidate)) * 1e-10
        if abs(candidate - groups[-1][-1]) <= inner_tolerance:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    # Coincident pairs are tangencies and toggle the even-odd fill twice, so
    # they contribute no interval boundary.
    boundaries = [
        float(np.mean(group))
        for group in groups
        if len(group) % 2
    ]
    if len(boundaries) < 2:
        return ()
    if len(boundaries) % 2:
        raise AnalysisError(
            "polygon slice produced an unmatched boundary; redraw the polygon "
            "without self-intersecting edges"
        )
    return tuple(
        (float(boundaries[index]), float(boundaries[index + 1]))
        for index in range(0, len(boundaries), 2)
        if boundaries[index + 1] > boundaries[index]
    )


def _polygon_bounds_at_z(vertices: np.ndarray, z_gain: float) -> Optional[tuple]:
    """Backward-compatible envelope helper for one polygon slice."""
    intervals = _polygon_intervals_at_outer(vertices, z_gain)
    if not intervals:
        return None
    return intervals[0][0], intervals[-1][1]


def sweep_path_from_polygon(
    vertices: Sequence[Sequence[float]],
    outer_values: Sequence[float],
    *,
    outer_name: str,
    inner_name: str,
    points_per_row: Optional[int] = 31,
    inner_resolution: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    outer_label: str = "",
    outer_unit: str = "",
    inner_label: str = "",
    inner_unit: str = "",
) -> SweepPath:
    """Convert a polygon into exact, possibly disjoint per-row intervals."""
    polygon = np.asarray(vertices, dtype=float)
    sampled_outer = np.asarray(outer_values, dtype=float)
    if (
        polygon.ndim != 2
        or polygon.shape[1:] != (2,)
        or polygon.shape[0] < 3
        or not np.all(np.isfinite(polygon))
    ):
        raise ConfigError(
            "path polygon requires at least three finite outer/inner vertices"
        )
    if (
        sampled_outer.ndim != 1
        or sampled_outer.size == 0
        or not np.all(np.isfinite(sampled_outer))
    ):
        raise ConfigError("polygon path requires a finite, non-empty outer axis")
    sampled_outer = np.unique(sampled_outer)
    selected_outer = []
    lower = []
    upper = []
    segment_rows = []
    for outer_value in sampled_outer:
        intervals = list(
            _polygon_intervals_at_outer(polygon, float(outer_value))
        )
        if inner_resolution is not None:
            retained = []
            for interval_lower, interval_upper in intervals:
                width = interval_upper - interval_lower
                tolerance = max(
                    1.0,
                    abs(width),
                    abs(inner_resolution),
                ) * 1e-12
                if width + tolerance >= inner_resolution:
                    retained.append((interval_lower, interval_upper))
            intervals = retained
        if not intervals:
            continue
        if points_per_row is not None and points_per_row < 2 * len(intervals):
            raise ConfigError(
                "points_per_row must provide at least two points for every "
                "polygon interval"
            )
        selected_outer.append(float(outer_value))
        lower.append(intervals[0][0])
        upper.append(intervals[-1][1])
        segment_rows.append(intervals)
    if not selected_outer:
        raise AnalysisError("drawn polygon does not contain any outer-axis rows")
    details = dict(metadata or {})
    details.update(
        {
            "created_at": utc_now(),
            "polygon_vertices": polygon.tolist(),
        }
    )
    return SweepPath(
        method="ui_polygon",
        outer_name=outer_name,
        inner_name=inner_name,
        outer_values=selected_outer,
        lower_inner_values=lower,
        upper_inner_values=upper,
        points_per_row=points_per_row,
        metadata=details,
        outer_label=outer_label,
        outer_unit=outer_unit,
        inner_label=inner_label,
        inner_unit=inner_unit,
        inner_resolution=inner_resolution,
        inner_segments=segment_rows,
    )


def frequency_sweep_path_from_polygon(
    vertices: Sequence[Sequence[float]],
    z_gain: Sequence[float],
    *,
    points_per_row: Optional[int] = 31,
    inner_resolution_mhz: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FrequencySweepPath:
    """Backward-compatible Z/frequency polygon helper."""
    generic = sweep_path_from_polygon(
        vertices,
        z_gain,
        outer_name="z_gain",
        inner_name="q_freq",
        points_per_row=points_per_row,
        inner_resolution=inner_resolution_mhz,
        metadata=metadata,
        outer_label="Z gain",
        outer_unit="a.u.",
        inner_label="Qubit frequency",
        inner_unit="MHz",
    )
    return FrequencySweepPath(
        method=generic.method,
        z_gain=generic.outer_values,
        lower_frequency_mhz=generic.lower_inner_values,
        upper_frequency_mhz=generic.upper_inner_values,
        points_per_row=generic.points_per_row,
        metadata=generic.metadata,
        inner_resolution_mhz=generic.inner_resolution,
        inner_segments=generic.inner_segments,
    )


def _background_signal(native: NativeMap, signal: str) -> tuple[np.ndarray, str]:
    name = str(signal).strip().lower()
    if name == "phase" and name not in native.signals:
        return np.angle(native.complex_signal), "Phase (rad)"
    if name == "amplitude" and name not in native.signals:
        return np.abs(native.complex_signal), "Amplitude"
    if name not in native.signals:
        choices = ", ".join(sorted(native.signals))
        raise ConfigError(f"background signal must be one of: {choices}")
    label = "Phase (rad)" if name == "phase" else name.capitalize()
    return np.asarray(native.signals[name], dtype=float), label


def _plot_background(axis, native: NativeMap, signal: str):
    values, label = _background_signal(native, signal)
    mesh = axis.pcolormesh(
        native.outer,
        native.inner,
        values.T,
        shading="auto",
        cmap="turbo",
    )
    outer_label = native.outer_label or "Outer axis"
    inner_label = native.inner_label or "Inner axis"
    if native.outer_unit:
        outer_label += f" ({native.outer_unit})"
    if native.inner_unit:
        inner_label += f" ({native.inner_unit})"
    axis.set(xlabel=outer_label, ylabel=inner_label)
    return mesh, label


def plot_sweep_path(
    path: SweepPath,
    background_csv: Path,
    *,
    signal: str = "phase",
):
    """Plot a saved/generated path over its reference measurement."""
    native = load_native_map(background_csv)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    mesh, label = _plot_background(axis, native, signal)
    polygon_vertices = path.metadata.get("polygon_vertices")
    if polygon_vertices is not None:
        region = PolygonPatch(
            np.asarray(polygon_vertices, dtype=float),
            closed=True,
            facecolor="white",
            edgecolor="white",
            linewidth=1.5,
            alpha=0.24,
            label="sweep region",
        )
        axis.add_patch(region)
    else:
        axis.fill_between(
            path.outer_values,
            path.lower_inner_values,
            path.upper_inner_values,
            color="white",
            alpha=0.22,
            label="sweep region",
        )
        axis.plot(
            path.outer_values,
            path.lower_inner_values,
            "w-",
            linewidth=1.5,
        )
        axis.plot(
            path.outer_values,
            path.upper_inner_values,
            "w-",
            linewidth=1.5,
        )
    if path.has_disjoint_intervals:
        interval_label_pending = True
        for row_index, (outer_value, intervals) in enumerate(
            zip(path.outer_values, path.inner_segments)
        ):
            for lower, upper in intervals:
                axis.plot(
                    [outer_value, outer_value],
                    [lower, upper],
                    "w-",
                    linewidth=1.0,
                    alpha=0.7,
                    label=(
                        "sampled intervals"
                        if interval_label_pending
                        else None
                    ),
                )
                interval_label_pending = False
    else:
        axis.plot(
            path.outer_values,
            path.center_inner_values,
            "w--",
            linewidth=1.2,
            label="path center",
        )
    if path.inner_resolution is not None:
        sampling = (
            f"{path.inner_resolution:g} {path.inner_unit or 'units'} resolution; "
            f"{path.point_counts.min()}-{path.point_counts.max()} points per row"
        )
    else:
        sampling = f"{path.points_per_row} points per row"
    if path.has_disjoint_intervals:
        maximum_intervals = max(len(row) for row in path.inner_segments)
        sampling += f"; up to {maximum_intervals} disjoint intervals"
    axis.set_title(
        f"{path.method}: {sampling} ({path.total_points} total points)"
    )
    axis.legend(loc="best")
    figure.colorbar(mesh, ax=axis, label=label)
    return figure


def plot_frequency_sweep_path(
    path: SweepPath,
    background_csv: Path,
    *,
    signal: str = "phase",
):
    """Backward-compatible alias for :func:`plot_sweep_path`."""
    return plot_sweep_path(path, background_csv, signal=signal)


def _parameter_name(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")
    if "freq" in text and any(word in text for word in ("qubit", "q_pulse")):
        return "q_freq"
    if "freq" in text and any(
        word in text for word in ("readout", "resonator", "r_pulse")
    ):
        return "r_freq"
    if "gain" in text and (text.startswith("z") or "flux" in text):
        return "z_gain"
    if "gain" in text and (text.startswith("q") or "qubit" in text):
        return "q_gain"
    if "power" in text and any(word in text for word in ("readout", "resonator")):
        return "r_power"
    return text or "axis"


def design_sweep_path_ui(
    background_csv: Path,
    *,
    signal: str = "phase",
    points_per_row: Optional[int] = None,
    inner_resolution: Optional[float] = None,
    outer_values: Optional[Sequence[float]] = None,
    outer_name: Optional[str] = None,
    inner_name: Optional[str] = None,
) -> SweepPath:
    """Interactively draw a sweep polygon over a prior native map."""
    from matplotlib.widgets import Button, PolygonSelector

    native = load_native_map(background_csv)
    if points_per_row is not None and inner_resolution is not None:
        raise ConfigError(
            "UI path design cannot set both points_per_row and inner_resolution"
        )
    if inner_resolution is not None and (
        not np.isfinite(float(inner_resolution))
        or float(inner_resolution) <= 0
    ):
        raise ConfigError("UI path inner resolution must be positive and finite")
    if points_per_row is None and inner_resolution is None:
        inner_resolution = float(np.median(np.diff(native.inner)))
    target_outer = (
        native.outer
        if outer_values is None
        else np.asarray(outer_values, dtype=float)
    )
    resolved_outer_name = outer_name or _parameter_name(native.outer_label)
    resolved_inner_name = inner_name or _parameter_name(native.inner_label)
    figure, axis = plt.subplots(figsize=(10, 7))
    figure.subplots_adjust(bottom=0.14)
    mesh, label = _plot_background(axis, native, signal)
    figure.colorbar(mesh, ax=axis, label=label)
    axis.set_title(
        "Click around the desired region and close the polygon; "
        "then choose Use region"
    )
    state = {"path": None, "accepted": False, "artists": []}

    def clear_preview():
        for artist in state["artists"]:
            artist.remove()
        state["artists"] = []

    def on_select(vertices):
        clear_preview()
        try:
            selected = sweep_path_from_polygon(
                vertices,
                target_outer,
                outer_name=resolved_outer_name,
                inner_name=resolved_inner_name,
                points_per_row=points_per_row,
                inner_resolution=inner_resolution,
                metadata={
                    "source_csv": str(native.source_csv),
                    "background_signal": str(signal),
                },
                outer_label=native.outer_label,
                outer_unit=native.outer_unit,
                inner_label=native.inner_label,
                inner_unit=native.inner_unit,
            )
        except (AnalysisError, ConfigError) as error:
            state["path"] = None
            axis.set_title(f"Invalid region: {error}")
            figure.canvas.draw_idle()
            return
        state["path"] = selected
        region = PolygonPatch(
            np.asarray(vertices, dtype=float),
            closed=True,
            facecolor="white",
            edgecolor="white",
            linewidth=1.5,
            alpha=0.28,
        )
        axis.add_patch(region)
        state["artists"] = [region]
        maximum_intervals = max(
            len(row) for row in selected.inner_segments
        )
        interval_note = (
            f", up to {maximum_intervals} disjoint intervals per row"
            if maximum_intervals > 1
            else ""
        )
        axis.set_title(
            f"Preview: {selected.outer_values.size} {selected.outer_name} rows, "
            f"{selected.point_counts.min()}-{selected.point_counts.max()} "
            f"{selected.inner_name} values per row{interval_note}. "
            "Choose Use region to save."
        )
        figure.canvas.draw_idle()

    selector = PolygonSelector(
        axis,
        on_select,
        useblit=True,
        props={"color": "white", "linewidth": 1.5, "alpha": 0.9},
    )
    use_axis = figure.add_axes((0.71, 0.025, 0.13, 0.055))
    clear_axis = figure.add_axes((0.85, 0.025, 0.10, 0.055))
    use_button = Button(use_axis, "Use region")
    clear_button = Button(clear_axis, "Clear")

    def accept(_event):
        if state["path"] is None:
            axis.set_title("Close a valid polygon before choosing Use region")
            figure.canvas.draw_idle()
            return
        state["accepted"] = True
        plt.close(figure)

    def clear(_event):
        selector.clear()
        clear_preview()
        state["path"] = None
        axis.set_title(
            "Click around the desired region and close the polygon; "
            "then choose Use region"
        )
        figure.canvas.draw_idle()

    use_button.on_clicked(accept)
    clear_button.on_clicked(clear)
    plt.show(block=True)
    if not state["accepted"] or state["path"] is None:
        raise AnalysisError("sweep-path editor was closed without accepting a region")
    return state["path"]


def design_frequency_sweep_path_ui(
    background_csv: Path,
    *,
    signal: str = "phase",
    points_per_row: Optional[int] = None,
    frequency_resolution_mhz: Optional[float] = None,
    z_gain: Optional[Sequence[float]] = None,
) -> SweepPath:
    """Backward-compatible Z-gain/qubit-frequency UI helper."""
    return design_sweep_path_ui(
        background_csv,
        signal=signal,
        points_per_row=points_per_row,
        inner_resolution=frequency_resolution_mhz,
        outer_values=z_gain,
        outer_name="z_gain",
        inner_name="q_freq",
    )
