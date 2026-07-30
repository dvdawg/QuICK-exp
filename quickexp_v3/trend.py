"""Regenerable calibration trends, repeatability, and drift diagnostics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

from .errors import AnalysisError, ConfigError
from .util import utc_now


TREND_COLUMNS = (
    "time_utc",
    "value",
    "uncertainty",
    "source_csv",
    "calibration_revision",
    "origin",
    "source_mtime_ns",
    "source_size",
)
_SAFE_ADDRESS = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class TrendStatistics:
    count: int
    repeatability_sigma: Optional[float]
    drift_per_day: Optional[float]
    steps: tuple

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "repeatability_sigma": self.repeatability_sigma,
            "drift_per_day": self.drift_per_day,
            "steps": list(self.steps),
        }


def _timestamp(value: Any, fallback: Optional[str] = None) -> str:
    candidate = value or fallback or utc_now()
    text = str(candidate)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AnalysisError(f"invalid trend timestamp {text!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _primary_value(record: Mapping[str, Any]) -> tuple:
    value = record.get("value")
    if isinstance(value, Mapping):
        parameters = value.get("parameters")
        candidates = parameters if isinstance(parameters, Mapping) else value
        preferred = (
            "center_frequency",
            "center_mhz",
            "frequency_mhz",
            "decay_us",
            "decay",
            "threshold",
            "fidelity",
            "value",
        )
        for name in preferred:
            number = _float_or_none(candidates.get(name))
            if number is not None:
                return name, number
        for name in sorted(candidates):
            number = _float_or_none(candidates[name])
            if number is not None:
                return str(name), number
        raise AnalysisError("calibration record has no scalar value to trend")
    number = _float_or_none(value)
    if number is None:
        raise AnalysisError("calibration record has no scalar value to trend")
    return "value", number


def _primary_uncertainty(record: Mapping[str, Any], value_name: str) -> Optional[float]:
    uncertainty = record.get("uncertainty")
    direct = _float_or_none(uncertainty)
    if direct is not None:
        return abs(direct)
    if not isinstance(uncertainty, Mapping):
        return None
    stem = value_name
    for suffix in ("_mhz", "_us", "_db", "_z"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    preferred = (
        f"{value_name}_stderr",
        f"{stem}_stderr",
        f"{value_name}_stderr_mhz",
        f"{stem}_stderr_mhz",
        f"{value_name}_uncertainty",
        f"{stem}_uncertainty",
        "stderr",
        "rmse_mhz",
        "rmse",
    )
    for name in preferred:
        number = _float_or_none(uncertainty.get(name))
        if number is not None:
            return abs(number)
    for name in sorted(uncertainty):
        if "stderr" in str(name).lower() or "uncertainty" in str(name).lower():
            number = _float_or_none(uncertainty[name])
            if number is not None:
                return abs(number)
    return None


def _source_snapshot(source_value: Any, base_directory: Path) -> tuple:
    if source_value is None:
        return "", "", ""
    source = str(source_value)
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    try:
        stat = candidate.stat()
    except (OSError, ValueError):
        return source, "", ""
    return source, str(int(stat.st_mtime_ns)), str(int(stat.st_size))


def _record_row(
    record: Mapping[str, Any],
    *,
    origin: str,
    revision: int,
    fallback_time: Optional[str],
    base_directory: Path,
) -> dict:
    value_name, value = _primary_value(record)
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    fitted_at = (
        provenance.get("fitted_at")
        or record.get("accepted_at")
        or fallback_time
    )
    source, source_mtime, source_size = _source_snapshot(
        provenance.get("source"),
        base_directory,
    )
    uncertainty = _primary_uncertainty(record, value_name)
    return {
        "time_utc": _timestamp(fitted_at),
        "value": f"{value:.17g}",
        "uncertainty": "" if uncertainty is None else f"{uncertainty:.17g}",
        "source_csv": source,
        "calibration_revision": str(int(revision)),
        "origin": origin,
        "source_mtime_ns": source_mtime,
        "source_size": source_size,
    }


def _accepted_records(node: Mapping[str, Any], prefix: str = "") -> Iterable[tuple]:
    for key, value in node.items():
        if not isinstance(value, Mapping):
            continue
        address = f"{prefix}.{key}" if prefix else str(key)
        if "value" in value:
            if value.get("status", "accepted") == "accepted":
                yield address, value
            continue
        yield from _accepted_records(value, address)


def trend_path(cache_directory: Path, address: str) -> Path:
    """Return the safe CSV path for one dotted calibration address."""
    address = str(address)
    if not address or not _SAFE_ADDRESS.fullmatch(address) or ".." in address:
        raise ConfigError(f"unsafe calibration record address: {address!r}")
    return Path(cache_directory).expanduser().resolve() / f"{address}.csv"


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TREND_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in TREND_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def harvest_calibration_trends(
    calibration_path: Path,
    cache_directory: Optional[Path] = None,
) -> Mapping[str, Path]:
    """Rebuild trend CSVs from accepted records and calibration history."""
    source = Path(calibration_path).expanduser().resolve()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping")
    records = loaded.get("records")
    if not isinstance(records, Mapping):
        raise ConfigError("calibration records must be a mapping")
    history = loaded.get("history", [])
    if not isinstance(history, list):
        raise ConfigError("calibration history must be a list")
    cache = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else source.parent / "analysis_cache" / "trends"
    )
    revision = int(loaded.get("revision", 0))
    result = {}
    for address, current in _accepted_records(records):
        historical = [
            item
            for item in history
            if isinstance(item, Mapping)
            and item.get("record") == address
            and isinstance(item.get("previous"), Mapping)
        ]
        rows = []
        total = len(historical)
        for index, item in enumerate(historical):
            previous = item["previous"]
            previous_revision = previous.get("accepted_revision")
            if previous_revision is None:
                previous_revision = max(0, revision - total + index)
            rows.append(
                _record_row(
                    previous,
                    origin="history",
                    revision=int(previous_revision),
                    fallback_time=item.get("superseded_at"),
                    base_directory=source.parent,
                )
            )
        current_revision = current.get("accepted_revision", revision)
        rows.append(
            _record_row(
                current,
                origin="accepted",
                revision=int(current_revision),
                fallback_time=loaded.get("updated_at"),
                base_directory=source.parent,
            )
        )
        path = trend_path(cache, address)
        _atomic_write_csv(path, rows)
        result[address] = path
    return result


def read_trend(path: Path) -> tuple:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(TREND_COLUMNS[:6]).difference(reader.fieldnames or ())
        if missing:
            raise AnalysisError(
                f"{source} is missing trend columns: {sorted(missing)}"
            )
        return tuple(dict(row) for row in reader)


def append_refit_trend(
    cache_directory: Path,
    address: str,
    *,
    value: float,
    uncertainty: Optional[float],
    source_csv: Path,
    time_utc: Optional[str] = None,
    calibration_revision: int = 0,
) -> Path:
    """Append an opt-in refit result without touching calibration."""
    path = trend_path(cache_directory, address)
    rows = list(read_trend(path)) if path.exists() else []
    source = Path(source_csv).expanduser().resolve()
    stat = source.stat()
    row = {
        "time_utc": _timestamp(time_utc),
        "value": f"{float(value):.17g}",
        "uncertainty": (
            "" if uncertainty is None else f"{abs(float(uncertainty)):.17g}"
        ),
        "source_csv": str(source),
        "calibration_revision": str(int(calibration_revision)),
        "origin": "refit",
        "source_mtime_ns": str(int(stat.st_mtime_ns)),
        "source_size": str(int(stat.st_size)),
    }
    identity = (row["time_utc"], row["source_csv"], row["origin"])
    rows = [
        existing
        for existing in rows
        if (
            existing.get("time_utc"),
            existing.get("source_csv"),
            existing.get("origin"),
        )
        != identity
    ]
    rows.append(row)
    rows.sort(key=lambda item: item["time_utc"])
    _atomic_write_csv(path, rows)
    return path


def _robust_slope(days: np.ndarray, values: np.ndarray) -> Optional[float]:
    slopes = []
    for right in range(1, values.size):
        delta_time = days[right] - days[:right]
        valid = delta_time > 0
        if np.any(valid):
            slopes.extend(
                ((values[right] - values[:right][valid]) / delta_time[valid]).tolist()
            )
    return float(np.median(slopes)) if slopes else None


def trend_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    settings: Optional[Sequence[Any]] = None,
) -> TrendStatistics:
    """Compute adjacent repeatability, Theil-style drift, and large steps."""
    parsed = []
    for index, row in enumerate(rows):
        value = _float_or_none(row.get("value"))
        if value is None:
            continue
        time = datetime.fromisoformat(
            str(row.get("time_utc")).replace("Z", "+00:00")
        )
        if time.tzinfo is None:
            time = time.replace(tzinfo=timezone.utc)
        uncertainty = _float_or_none(row.get("uncertainty"))
        setting = settings[index] if settings is not None else None
        parsed.append((time.timestamp(), value, uncertainty, setting, dict(row)))
    parsed.sort(key=lambda item: item[0])
    if not parsed:
        return TrendStatistics(0, None, None, ())
    values = np.asarray([item[1] for item in parsed], dtype=float)
    seconds = np.asarray([item[0] for item in parsed], dtype=float)
    days = (seconds - seconds[0]) / 86400.0
    differences = []
    for index in range(1, len(parsed)):
        if settings is None or parsed[index - 1][3] == parsed[index][3]:
            differences.append(values[index] - values[index - 1])
    if differences:
        deltas = np.asarray(differences, dtype=float)
        if deltas.size >= 3:
            center = float(np.median(deltas))
            repeatability = float(
                np.median(np.abs(deltas - center))
                / (0.6744897501960817 * np.sqrt(2.0))
            )
        else:
            repeatability = float(np.sqrt(np.mean(deltas**2) / 2.0))
    else:
        repeatability = None
    drift = _robust_slope(days, values)
    steps = []
    for index in range(1, len(parsed)):
        uncertainties = [
            value
            for value in (
                repeatability,
                parsed[index - 1][2],
                parsed[index][2],
            )
            if value is not None and value > 0
        ]
        if not uncertainties:
            continue
        delta = float(values[index] - values[index - 1])
        threshold = 5.0 * max(uncertainties)
        if abs(delta) > threshold:
            steps.append(
                {
                    "time_utc": parsed[index][4].get("time_utc"),
                    "delta": delta,
                    "threshold": threshold,
                }
            )
    return TrendStatistics(
        count=len(parsed),
        repeatability_sigma=repeatability,
        drift_per_day=drift,
        steps=tuple(steps),
    )


def source_changed(row: Mapping[str, Any], base_directory: Optional[Path] = None) -> bool:
    """Return true when a harvested source's recorded mtime/size no longer match."""
    source = str(row.get("source_csv", ""))
    recorded_mtime = row.get("source_mtime_ns", "")
    recorded_size = row.get("source_size", "")
    if not source or recorded_mtime in ("", None) or recorded_size in ("", None):
        return False
    path = Path(source).expanduser()
    if not path.is_absolute() and base_directory is not None:
        path = Path(base_directory) / path
    try:
        stat = path.stat()
    except OSError:
        return True
    return (
        int(recorded_mtime) != int(stat.st_mtime_ns)
        or int(recorded_size) != int(stat.st_size)
    )


def sparkline(values: Sequence[float]) -> str:
    """Render a compact Unicode trend without requiring an image viewer."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    span = float(np.ptp(array))
    if span <= np.finfo(float).eps:
        return blocks[3] * array.size
    indices = np.rint((array - np.min(array)) / span * (len(blocks) - 1)).astype(int)
    return "".join(blocks[index] for index in indices)
