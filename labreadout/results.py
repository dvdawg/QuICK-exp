"""Robust analysis outputs that sit *alongside* the raw data.

The raw measurement CSV is written by the ``quick`` framework and must stay
byte-for-byte identical (external plotting software depends on it). This module
only ever writes a small fit-result *sidecar* YAML next to each CSV; it never
opens or modifies the CSV itself.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from typing import Any, Dict, Optional

import numpy as np

from .state import _coerce


def summarize(fit: Any) -> Dict[str, Any]:
    """Flatten a fit dataclass to scalars + params, dropping raw data arrays."""
    summary: Dict[str, Any] = {}
    for f in dataclasses.fields(fit):
        value = getattr(fit, f.name)
        if isinstance(value, np.ndarray):
            continue  # raw freq/signal/projection arrays do not belong here
        summary[f.name] = _coerce(value)
    summary["model"] = type(fit).__name__
    return summary


def sidecar_path(csv_path: str) -> str:
    """Path of the fit sidecar for a given raw CSV (``...csv`` -> ``...fit.yml``)."""
    if csv_path.endswith(".csv"):
        return csv_path[: -len(".csv")] + ".fit.yml"
    return csv_path + ".fit.yml"


def write_sidecar(
    csv_path: str, fit: Any, extra: Optional[Dict[str, Any]] = None
) -> str:
    """Write the fit summary as YAML next to ``csv_path``. Returns the sidecar path.

    The raw CSV is never read or written here.
    """
    import yaml

    payload = summarize(fit)
    payload["analyzed"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if extra:
        payload.update(_coerce(extra))

    out = sidecar_path(csv_path)
    with open(out, "w") as fh:
        yaml.safe_dump(payload, fh, default_flow_style=False, sort_keys=True)
    return out
