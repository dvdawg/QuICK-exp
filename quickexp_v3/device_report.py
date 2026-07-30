"""Generate a local Markdown snapshot of accepted device state and data health."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .errors import AnalysisError, ConfigError
from .flux_lookup import frequency_from_record
from .native_index import NativeIndex
from .native_map import load_native_map
from .trace_qc import qc_map, qc_trace
from .trend import (
    harvest_calibration_trends,
    read_trend,
    source_changed,
    sparkline,
    trend_statistics,
)


@dataclass(frozen=True)
class ReportArtifacts:
    markdown_path: Path
    figure_paths: tuple
    warnings: tuple


def _record_leaves(node: Mapping[str, Any], prefix: str = "") -> Iterable[tuple]:
    for key, value in node.items():
        if not isinstance(value, Mapping):
            continue
        address = f"{prefix}.{key}" if prefix else str(key)
        if "value" in value:
            yield address, value
        else:
            yield from _record_leaves(value, address)


def _format_value(value: Any) -> str:
    if isinstance(value, Mapping):
        parameters = value.get("parameters")
        content = parameters if isinstance(parameters, Mapping) else value
        pieces = []
        for key, item in content.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                pieces.append(f"{key}={float(item):.6g}")
            elif isinstance(item, str):
                pieces.append(f"{key}={item}")
        return ", ".join(pieces) if pieces else json.dumps(value, sort_keys=True)
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def _format_uncertainty(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, Mapping):
        pieces = []
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                pieces.append(f"{key}={float(item):.3g}")
        return ", ".join(pieces[:3]) or "—"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def _record_time(record: Mapping[str, Any]) -> Optional[datetime]:
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    raw = record.get("accepted_at") or provenance.get("fitted_at")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _age(record: Mapping[str, Any], now: datetime) -> str:
    timestamp = _record_time(record)
    if timestamp is None:
        return "unknown"
    days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    return f"{days:.1f} d"


def _proposal_leaves(node: Mapping[str, Any], prefix: str = "") -> Iterable[tuple]:
    for key, value in node.items():
        if not isinstance(value, Mapping):
            continue
        address = f"{prefix}.{key}" if prefix else str(key)
        if "record" in value or "proposal_id" in value or "value" in value:
            yield address, value
        else:
            yield from _proposal_leaves(value, address)


def _native_qc(record) -> dict:
    if record.n_axes == 1 and record.csv_columns >= 5:
        try:
            matrix = np.atleast_2d(np.loadtxt(record.csv_path, delimiter=","))
        except ValueError:
            matrix = np.atleast_2d(np.loadtxt(record.csv_path))
        quality = qc_trace(matrix[:, 0], matrix[:, -2] + 1j * matrix[:, -1])
        return {
            "traces": 1,
            "spikes": quality.spike_count,
            "nonuniform": int(not quality.axis_uniform),
            "clipped": int(quality.clipping_suspected),
        }
    if record.n_axes == 2:
        native_map = load_native_map(record.csv_path)
        rows = tuple(qc_map(native_map).values())
        return {
            "traces": len(rows),
            "spikes": sum(item.spike_count for item in rows),
            "nonuniform": sum(not item.axis_uniform for item in rows),
            "clipped": sum(item.clipping_suspected for item in rows),
        }
    return {"traces": 0, "spikes": 0, "nonuniform": 0, "clipped": 0}


def _flux_figure(
    record: Mapping[str, Any],
    output_path: Path,
) -> Optional[Path]:
    domain = record.get("valid_domain")
    z_domain = domain.get("z_gain") if isinstance(domain, Mapping) else None
    if not isinstance(z_domain, (list, tuple)) or len(z_domain) != 2:
        return None
    z = np.linspace(float(z_domain[0]), float(z_domain[1]), 401)
    try:
        frequency = np.asarray(frequency_from_record(record, z), dtype=float)
    except (ConfigError, AnalysisError, ValueError, TypeError):
        return None
    figure, axis = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    axis.plot(z, frequency, color="#275d86", linewidth=2)
    axis.set(
        title="Accepted resonator-versus-flux lookup",
        xlabel="Z gain",
        ylabel="Readout frequency (MHz)",
    )
    axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def generate_device_report(
    project_root: Path,
    *,
    calibration_path: Optional[Path] = None,
    data_directory: Optional[Path] = None,
    output_directory: Optional[Path] = None,
    trend_directory: Optional[Path] = None,
    latest_runs: int = 20,
    report_date: Optional[date] = None,
) -> ReportArtifacts:
    """Generate the device report entirely from local YAML and native pairs."""
    root = Path(project_root).expanduser().resolve()
    calibration = (
        Path(calibration_path).expanduser().resolve()
        if calibration_path is not None
        else (
            root / "calibration.yml"
            if (root / "calibration.yml").exists()
            else root / "calibration.example.yml"
        )
    )
    document = yaml.safe_load(calibration.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ConfigError(f"{calibration} must contain a YAML mapping")
    records = document.get("records", {})
    if not isinstance(records, Mapping):
        raise ConfigError("calibration records must be a mapping")
    trends = harvest_calibration_trends(calibration, trend_directory)
    output = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else root / "analysis_out"
    )
    output.mkdir(parents=True, exist_ok=True)
    day = report_date or date.today()
    markdown_path = output / f"report_{day.isoformat()}.md"
    now = datetime.now(timezone.utc)
    warnings = []
    figures = []

    lines = [
        f"# Device report — {day.isoformat()}",
        "",
        (
            f"Calibration revision **{int(document.get('revision', 0))}**; "
            f"generated from `{calibration.name}`."
        ),
        "",
        "## Current accepted calibration",
        "",
        "| Address | Value | Unit | Uncertainty | Age | Revision |",
        "|---|---:|---|---|---:|---:|",
    ]
    accepted = []
    for address, record in _record_leaves(records):
        if record.get("status", "accepted") != "accepted":
            continue
        accepted.append((address, record))
        lines.append(
            "| "
            + " | ".join(
                (
                    address,
                    _format_value(record.get("value")).replace("|", "\\|"),
                    str(record.get("unit", "")),
                    _format_uncertainty(record.get("uncertainty")).replace("|", "\\|"),
                    _age(record, now),
                    str(
                        record.get(
                            "accepted_revision",
                            document.get("revision", 0),
                        )
                    ),
                )
            )
            + " |"
        )
    if not accepted:
        lines.append("| _none_ | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Trends",
            "",
            "| Address | Samples | Sparkline | Repeatability σ | Drift/day | Steps |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for address, path in sorted(trends.items()):
        rows = read_trend(path)
        statistics = trend_statistics(rows)
        values = [float(row["value"]) for row in rows]
        changed = [
            row
            for row in rows
            if source_changed(row, base_directory=calibration.parent)
        ]
        if changed:
            warnings.append(
                f"{address}: {len(changed)} harvested source file(s) changed or disappeared"
            )
        repeatability = (
            "—"
            if statistics.repeatability_sigma is None
            else f"{statistics.repeatability_sigma:.3g}"
        )
        drift = (
            "—"
            if statistics.drift_per_day is None
            else f"{statistics.drift_per_day:.3g}"
        )
        lines.append(
            f"| {address} | {statistics.count} | {sparkline(values)} | "
            f"{repeatability} | {drift} | {len(statistics.steps)} |"
        )

    lines.extend(["", "## Recent native-data QC", ""])
    hardware_path = (
        root / "hardware.yml"
        if (root / "hardware.yml").exists()
        else root / "hardware.example.yml"
    )
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    hardware = hardware if isinstance(hardware, Mapping) else {}
    if data_directory is None:
        storage = hardware.get("storage", {})
        raw_data = storage.get("quick_native_root", "./data") if isinstance(storage, Mapping) else "./data"
        candidate = Path(str(raw_data)).expanduser()
        data_directory = candidate if candidate.is_absolute() else root / candidate
    data_root = Path(data_directory).expanduser().resolve()
    if data_root.exists():
        index = NativeIndex(
            data_root,
            cache_root=root / "analysis_cache" / "native_index",
        ).refresh()
        recent = tuple(index.records())[-max(0, int(latest_runs)) :]
        classes = Counter(record.quick_class or "unknown" for record in recent)
        aggregate = {"traces": 0, "spikes": 0, "nonuniform": 0, "clipped": 0}
        qc_failures = []
        for record in recent:
            try:
                row = _native_qc(record)
            except (OSError, ValueError, AnalysisError) as error:
                qc_failures.append(f"{record.csv_path.name}: {error}")
                continue
            for key in aggregate:
                aggregate[key] += int(row[key])
        lines.extend(
            [
                f"- Indexed complete pairs: {len(index.records())}",
                f"- Last {len(recent)} classes: "
                + (
                    ", ".join(f"{name}={count}" for name, count in sorted(classes.items()))
                    if classes
                    else "none"
                ),
                (
                    "- QC traces: "
                    f"{aggregate['traces']}; spikes: {aggregate['spikes']}; "
                    f"nonuniform axes: {aggregate['nonuniform']}; "
                    f"clipping flags: {aggregate['clipped']}"
                ),
                f"- Skipped/incomplete pair warnings: {len(index.warnings)}",
            ]
        )
        warnings.extend(index.warnings)
        warnings.extend(qc_failures)
    else:
        lines.append(f"- Native data directory is absent: `{data_root}`")

    lines.extend(["", "## Open calibration proposals", ""])
    proposals = document.get("proposals", {})
    proposal_rows = (
        tuple(_proposal_leaves(proposals))
        if isinstance(proposals, Mapping)
        else ()
    )
    if proposal_rows:
        lines.extend(
            [
                "| Address | Proposal | Status | Value |",
                "|---|---|---|---:|",
            ]
        )
        for address, proposal in proposal_rows:
            lines.append(
                f"| {address} | {proposal.get('proposal_id', '—')} | "
                f"{proposal.get('status', 'open')} | "
                f"{_format_value(proposal.get('record', proposal).get('value'))} |"
            )
    else:
        lines.append("_No open proposals._")

    lookup = None
    for address, record in accepted:
        if address == "lookups.resonator_vs_flux":
            lookup = record
            break
    if lookup is not None:
        figure_path = output / f"report_{day.isoformat()}_flux.png"
        generated = _flux_figure(lookup, figure_path)
        if generated is not None:
            figures.append(generated)
            lines.extend(
                [
                    "",
                    "## Flux lookup",
                    "",
                    f"![Accepted resonator-versus-flux lookup]({generated.name})",
                ]
            )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("_None._")
    lines.append("")
    _atomic_text(markdown_path, "\n".join(lines))
    return ReportArtifacts(
        markdown_path=markdown_path,
        figure_paths=tuple(figures),
        warnings=tuple(warnings),
    )
