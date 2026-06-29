"""Persistent calibration state.

Holds the latest fitted values (``r_freq``, ``q_freq``, pi-pulse, thresholds,
T1/T2, ...) in a YAML file so calibrated numbers survive kernel restarts instead
of living in notebook cell state. Loaded at session start and merged over the
base ``var`` defaults; updated only when the operator accepts a fit.

Pure file + dict logic with no hardware dependency, so it is unit-tested offline.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def _coerce(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python for clean YAML output."""
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if hasattr(value, "tolist"):  # numpy array
        return value.tolist()
    return value


def load(path: str) -> Dict[str, Any]:
    """Load calibration state, returning an empty dict if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def save(path: str, data: Dict[str, Any]) -> None:
    """Write calibration state to YAML (numpy-safe), creating parent dirs."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(_coerce(dict(data)), fh, default_flow_style=False, sort_keys=True)


def record(path: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``updates`` into the persisted state and save. Returns merged dict."""
    current = load(path)
    current.update(_coerce(updates))
    save(path, current)
    return current


def merged_var(base_var: Dict[str, Any], cal_state: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay calibration state onto base ``var`` defaults without mutating either.

    Only keys already present in ``base_var`` are overridden, so stray state keys
    never silently introduce new variables.
    """
    merged = dict(base_var)
    for key, value in cal_state.items():
        if key in merged:
            merged[key] = value
    return merged
