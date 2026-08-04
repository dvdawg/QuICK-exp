"""Naming helpers for Quick's native CSV/YML Saver titles."""

from __future__ import annotations

from typing import Any

import numpy as np


def representative(value: Any) -> float:
    """Return the center element of a scalar or one-dimensional setting."""
    values = np.asarray(value, dtype=float).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("title values must be finite and non-empty")
    return float(values[values.size // 2])


def number_tag(value: Any, decimals: int = 3) -> str:
    """Return a filesystem-friendly decimal tag."""
    text = f"{representative(value):.{int(decimals)}f}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def z_tag(value: Any) -> str:
    number = representative(value)
    sign = "p" if number >= 0 else "m"
    return "Z" + sign + f"{abs(number):.4f}".replace(".", "p")
