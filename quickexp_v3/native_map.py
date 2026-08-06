"""Load two-dimensional native Quick and combined Saver maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml

from .errors import AnalysisError


@dataclass(frozen=True)
class NativeMap:
    source_csv: Path
    outer_label: str
    outer_unit: str
    outer: np.ndarray
    inner_label: str
    inner_unit: str
    inner: np.ndarray
    signals: Mapping[str, np.ndarray]
    row_metadata: Mapping[str, np.ndarray]
    incomplete_outer: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def complex_signal(self) -> np.ndarray:
        if "i" not in self.signals or "q" not in self.signals:
            raise AnalysisError("native map has no I/Q signal pair")
        return self.signals["i"] + 1j * self.signals["q"]


@dataclass(frozen=True)
class NativeRowMap:
    """A two-axis native map whose inner sweep may differ for every row."""

    source_csv: Path
    outer_label: str
    outer_unit: str
    outer: np.ndarray
    inner_label: str
    inner_unit: str
    inner_rows: tuple[np.ndarray, ...]
    signals: Mapping[str, tuple[np.ndarray, ...]]
    row_metadata: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def complex_signal_rows(self) -> tuple[np.ndarray, ...]:
        if "i" not in self.signals or "q" not in self.signals:
            raise AnalysisError("native row map has no I/Q signal pair")
        return tuple(
            i_row + 1j * q_row
            for i_row, q_row in zip(self.signals["i"], self.signals["q"])
        )

    @property
    def point_counts(self) -> np.ndarray:
        return np.asarray([row.size for row in self.inner_rows], dtype=int)


def _entry(value: Any, label: str) -> tuple:
    if not isinstance(value, (list, tuple)) or not value:
        raise AnalysisError(f"native Quick YML has an invalid {label}")
    return str(value[0]), str(value[1]) if len(value) > 1 else ""


def _key(label: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    aliases = {
        "amplitude": "amplitude",
        "phase": "phase",
        "i": "i",
        "q": "q",
        "readout_frequency": "readout_frequency",
    }
    return aliases.get(text, text)


def load_native_map(csv_path: Path) -> NativeMap:
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"native Quick CSV does not exist or is empty: {source}")
    yml_path = source.with_suffix(".yml")
    if not yml_path.is_file():
        raise FileNotFoundError(f"paired Quick YML does not exist: {yml_path}")
    metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AnalysisError(f"{yml_path} must contain a YAML mapping")
    independent = metadata.get("independent")
    if not isinstance(independent, list) or len(independent) != 2:
        raise AnalysisError("native map fitting requires exactly two independent axes")
    outer_label, outer_unit = _entry(independent[0], "outer axis")
    inner_label, inner_unit = _entry(independent[1], "inner axis")
    dependent = metadata.get("dependent")
    if not isinstance(dependent, list) or not dependent:
        raise AnalysisError("native map YML has no dependent columns")
    dependent_entries = [_entry(value, "dependent column") for value in dependent]
    try:
        matrix = np.loadtxt(source, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(source)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    expected_columns = 2 + len(dependent_entries)
    if matrix.ndim != 2 or matrix.shape[1] < expected_columns:
        raise AnalysisError(
            f"native map expected at least {expected_columns} columns, got {matrix.shape}"
        )
    finite_axes = np.isfinite(matrix[:, 0]) & np.isfinite(matrix[:, 1])
    matrix = matrix[finite_axes]
    if matrix.size == 0:
        raise AnalysisError("native map has no finite axis rows")
    outer_all = np.unique(matrix[:, 0])
    inner = np.unique(matrix[:, 1])
    if outer_all.size < 1 or inner.size < 2:
        raise AnalysisError("native map requires at least one row and two inner points")
    inner_steps = np.diff(inner)
    inner_step = float(np.median(inner_steps))
    if inner_step <= 0 or not np.allclose(
        inner_steps,
        inner_step,
        rtol=1e-5,
        atol=max(abs(inner_step) * 1e-8, 1e-12),
    ):
        raise AnalysisError("native map inner axis must be uniformly spaced")
    outer_index = np.searchsorted(outer_all, matrix[:, 0])
    inner_index = np.searchsorted(inner, matrix[:, 1])
    counts = np.zeros((outer_all.size, inner.size), dtype=int)
    np.add.at(counts, (outer_index, inner_index), 1)
    if np.any(counts > 1):
        raise AnalysisError("native map contains duplicate grid points")
    complete = np.all(counts == 1, axis=1)
    incomplete = outer_all[~complete]
    outer = outer_all[complete]
    if outer.size == 0:
        raise AnalysisError("native map has no complete outer rows")

    signals = {}
    row_metadata = {}
    for offset, (label, _unit) in enumerate(dependent_entries):
        name = _key(label)
        grid = np.full((outer_all.size, inner.size), np.nan)
        grid[outer_index, inner_index] = matrix[:, 2 + offset]
        grid = grid[complete]
        if name == "readout_frequency":
            values = []
            for row in grid:
                finite = row[np.isfinite(row)]
                if finite.size == 0 or not np.allclose(finite, finite[0]):
                    raise AnalysisError(
                        "readout frequency metadata must be constant within each outer row"
                    )
                values.append(float(finite[0]))
            row_metadata[name] = np.asarray(values, dtype=float)
        else:
            signals[name] = grid
    if not signals:
        raise AnalysisError("native map has no signal columns")
    return NativeMap(
        source_csv=source,
        outer_label=outer_label,
        outer_unit=outer_unit,
        outer=outer,
        inner_label=inner_label,
        inner_unit=inner_unit,
        inner=inner,
        signals=MappingProxyType(signals),
        row_metadata=MappingProxyType(row_metadata),
        incomplete_outer=incomplete,
        metadata=MappingProxyType(dict(metadata)),
    )


def load_native_row_map(csv_path: Path) -> NativeRowMap:
    """Load a native map without requiring a rectangular inner-axis grid."""
    source = Path(csv_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"native Quick CSV does not exist or is empty: {source}")
    yml_path = source.with_suffix(".yml")
    if not yml_path.is_file():
        raise FileNotFoundError(f"paired Quick YML does not exist: {yml_path}")
    metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AnalysisError(f"{yml_path} must contain a YAML mapping")
    independent = metadata.get("independent")
    if not isinstance(independent, list) or len(independent) != 2:
        raise AnalysisError("native row-map fitting requires two independent axes")
    outer_label, outer_unit = _entry(independent[0], "outer axis")
    inner_label, inner_unit = _entry(independent[1], "inner axis")
    dependent = metadata.get("dependent")
    if not isinstance(dependent, list) or not dependent:
        raise AnalysisError("native row-map YML has no dependent columns")
    dependent_entries = [_entry(value, "dependent column") for value in dependent]
    try:
        matrix = np.loadtxt(source, delimiter=",")
    except ValueError:
        matrix = np.loadtxt(source)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    expected_columns = 2 + len(dependent_entries)
    if matrix.ndim != 2 or matrix.shape[1] < expected_columns:
        raise AnalysisError(
            f"native row map expected at least {expected_columns} columns, "
            f"got {matrix.shape}"
        )
    finite_axes = np.isfinite(matrix[:, 0]) & np.isfinite(matrix[:, 1])
    matrix = matrix[finite_axes]
    if matrix.size == 0:
        raise AnalysisError("native row map has no finite axis rows")
    if np.unique(matrix[:, :2], axis=0).shape[0] != matrix.shape[0]:
        raise AnalysisError("native row map contains duplicate axis points")

    outer = np.unique(matrix[:, 0])
    inner_rows = []
    row_matrices = []
    for outer_value in outer:
        row = matrix[matrix[:, 0] == outer_value]
        row = row[np.argsort(row[:, 1])]
        inner = np.asarray(row[:, 1], dtype=float)
        if inner.size < 2 or np.any(np.diff(inner) <= 0):
            raise AnalysisError(
                "every native row-map row requires at least two unique inner values"
            )
        inner_rows.append(inner)
        row_matrices.append(row)

    signals = {}
    row_metadata = {}
    for offset, (label, _unit) in enumerate(dependent_entries):
        name = _key(label)
        rows = tuple(
            np.asarray(row[:, 2 + offset], dtype=float)
            for row in row_matrices
        )
        if name == "readout_frequency":
            values = []
            for row in rows:
                finite = row[np.isfinite(row)]
                if finite.size == 0 or not np.allclose(finite, finite[0]):
                    raise AnalysisError(
                        "readout frequency metadata must be constant within each row"
                    )
                values.append(float(finite[0]))
            row_metadata[name] = np.asarray(values, dtype=float)
        else:
            signals[name] = rows
    if not signals:
        raise AnalysisError("native row map has no signal columns")
    return NativeRowMap(
        source_csv=source,
        outer_label=outer_label,
        outer_unit=outer_unit,
        outer=np.asarray(outer, dtype=float),
        inner_label=inner_label,
        inner_unit=inner_unit,
        inner_rows=tuple(inner_rows),
        signals=MappingProxyType(signals),
        row_metadata=MappingProxyType(row_metadata),
        metadata=MappingProxyType(dict(metadata)),
    )
