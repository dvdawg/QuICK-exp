"""YAML configuration with deterministic, provenance-preserving overlays.

The mutable lab configuration is deliberately not Python code.  Hardware facts,
accepted calibrations, named presets, and one-run overrides are separate layers:

    hardware defaults < accepted calibration < preset parameters < overrides

Only the hardware file may define routing, instruments, limits, and safety
policy.  Experiment modules consume a :class:`ResolvedConfig`; they never mutate
the source documents.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

from .errors import ConfigError
from .util import deep_merge, dotted_get, fingerprint


SCHEMA_VERSION = 3
PROTECTED_ROOTS = frozenset(
    {
        "schema_version",
        "site",
        "device",
        "storage",
        "qick",
        "channels",
        "rf_board",
        "external_instruments",
        "flux_lines",
        "limits",
        "safety",
        "autocal",
        "expected",
    }
)
SWEEP_KEYS = frozenset(
    {"start", "stop", "points", "center", "center_from", "span", "values"}
)
CALIBRATION_RECORD_KEYS = frozenset(
    {
        "value",
        "unit",
        "uncertainty",
        "provenance",
        "quality",
        "valid_domain",
        "model",
        "notes",
        "status",
        "proposal_id",
        "created_at",
        "accepted_at",
        "accepted_by",
        "accepted_revision",
    }
)


def _read_yaml(path: Optional[Union[str, Path]], *, required: bool) -> dict:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        if required:
            raise ConfigError(f"configuration file does not exist: {source}")
        return {}
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {source}: {error}") from error
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{source} must contain a YAML mapping")
    return deepcopy(dict(loaded))


def _check_schema(document: Mapping[str, Any], label: str) -> None:
    if not document:
        return
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"{label}.schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )


def _reject_protected(layer: Mapping[str, Any], label: str) -> None:
    forbidden = sorted(PROTECTED_ROOTS.intersection(layer))
    if forbidden:
        raise ConfigError(
            f"{label} cannot override hardware-controlled roots: "
            + ", ".join(forbidden)
        )


def _record_tree(node: Mapping[str, Any]) -> dict:
    """Extract accepted leaves while preserving their dotted tree."""
    result: dict = {}
    for key, value in node.items():
        if not isinstance(value, Mapping):
            continue
        if "value" in value and set(value).issubset(CALIBRATION_RECORD_KEYS):
            if value.get("status", "accepted") == "accepted":
                result[key] = deepcopy(value["value"])
            continue
        child = _record_tree(value)
        if child:
            result[key] = child
    return result


def accepted_calibration_values(document: Mapping[str, Any]) -> dict:
    if not document:
        return {}
    records = document.get("records")
    if isinstance(records, Mapping):
        return _record_tree(records)
    values = document.get("values")
    if isinstance(values, Mapping):
        return deepcopy(dict(values))
    return {}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ConfigError(f"{label} must be a finite number")
    return number


def is_sweep(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = set(value)
    return bool(keys.intersection(SWEEP_KEYS))


def _merge_parameter_layers(*layers: Mapping[str, Any]) -> dict:
    """Merge ordinary nested parameters but replace sweep specs atomically."""
    result: dict = {}
    for layer in layers:
        for key, value in layer.items():
            existing = result.get(key)
            if (
                isinstance(value, Mapping)
                and not is_sweep(value)
                and isinstance(existing, Mapping)
                and not is_sweep(existing)
            ):
                result[key] = _merge_parameter_layers(existing, value)
            else:
                result[key] = deepcopy(value)
    return result


def expand_sweep(
    value: Any,
    context: Mapping[str, Any],
    *,
    label: str,
) -> Any:
    """Expand one sweep mapping, leaving ordinary dictionaries untouched."""
    if not is_sweep(value):
        return deepcopy(value)
    spec = dict(value)
    unit = spec.pop("unit", None)
    unknown = set(spec).difference(SWEEP_KEYS)
    if unknown:
        raise ConfigError(f"{label} has unknown sweep keys: {sorted(unknown)}")
    if "values" in spec:
        if set(spec) != {"values"}:
            raise ConfigError(f"{label}.values cannot be mixed with range keys")
        array = np.asarray(spec["values"])
    else:
        points = int(spec.get("points", 0))
        if points < 1:
            raise ConfigError(f"{label}.points must be at least 1")
        if {"start", "stop"} <= set(spec):
            allowed = {"start", "stop", "points"}
            if not set(spec) <= allowed:
                raise ConfigError(
                    f"{label} start/stop form cannot contain center/span keys"
                )
            start = _finite_number(spec["start"], f"{label}.start")
            stop = _finite_number(spec["stop"], f"{label}.stop")
        elif "span" in spec:
            allowed = {"center", "center_from", "span", "points"}
            if not set(spec) <= allowed:
                raise ConfigError(
                    f"{label} center/span form cannot contain start/stop keys"
                )
            if ("center" in spec) == ("center_from" in spec):
                raise ConfigError(
                    f"{label} requires exactly one of center or center_from"
                )
            if "center_from" in spec:
                center = dotted_get(context, str(spec["center_from"]))
                if center is None:
                    raise ConfigError(
                        f"{label} cannot resolve {spec['center_from']!r}"
                    )
            else:
                center = spec["center"]
            center_value = _finite_number(center, f"{label}.center")
            span = _finite_number(spec["span"], f"{label}.span")
            if span < 0:
                raise ConfigError(f"{label}.span cannot be negative")
            start = center_value - span / 2.0
            stop = center_value + span / 2.0
        else:
            raise ConfigError(
                f"{label} must use values, start/stop/points, or "
                "center/center_from/span/points"
            )
        array = np.linspace(start, stop, points)
    if array.ndim != 1 or array.size == 0:
        raise ConfigError(f"{label} must expand to a non-empty one-dimensional axis")
    if np.issubdtype(array.dtype, np.number):
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{label} contains a non-numeric value") from error
        if not np.all(np.isfinite(numeric)):
            raise ConfigError(f"{label} contains a non-finite value")
    # Unit is metadata in YAML, not a Quick variable.
    _ = unit
    return array


def _validate_hardware(document: Mapping[str, Any]) -> None:
    _check_schema(document, "hardware")
    for required in ("storage", "channels", "defaults", "limits"):
        if not isinstance(document.get(required), Mapping):
            raise ConfigError(f"hardware.{required} must be a mapping")
    channels = document["channels"]
    endpoints = {"r": "gen", "q": "gen", "z": "gen", "rr": "ro"}
    for logical, endpoint in endpoints.items():
        channel = channels.get(logical)
        if not isinstance(channel, Mapping) or endpoint not in channel:
            raise ConfigError(f"hardware.channels.{logical}.{endpoint} is required")
        try:
            index = int(channel[endpoint])
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"hardware.channels.{logical}.{endpoint} must be an integer"
            ) from error
        if index < 0:
            raise ConfigError(
                f"hardware.channels.{logical}.{endpoint} cannot be negative"
            )
    _validate_autocal_policy(
        document.get("autocal"),
        limits=document["limits"],
    )


def _validate_autocal_policy(
    value: Any,
    *,
    limits: Mapping[str, Any],
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigError("hardware.autocal must be a mapping")
    allowed_roots = {
        "hard_stop_records",
        "auto_accept",
        "budgets",
        "caps",
        "ramsey_sign",
    }
    unknown_roots = set(value).difference(allowed_roots)
    if unknown_roots:
        raise ConfigError(
            "hardware.autocal has unknown keys: "
            + ", ".join(sorted(unknown_roots))
        )
    hard_stops = value.get("hard_stop_records", [])
    if (
        not isinstance(hard_stops, list)
        or any(not isinstance(item, str) or not item for item in hard_stops)
    ):
        raise ConfigError(
            "hardware.autocal.hard_stop_records must be a list of addresses"
        )
    tolerances = value.get("auto_accept", {})
    if not isinstance(tolerances, Mapping):
        raise ConfigError("hardware.autocal.auto_accept must be a mapping")
    allowed_tolerances = {
        "absolute_mhz",
        "absolute_db",
        "absolute",
        "relative",
        "always",
    }
    for address, tolerance in tolerances.items():
        if not isinstance(address, str) or not address:
            raise ConfigError("autocal auto_accept addresses must be strings")
        if not isinstance(tolerance, Mapping) or len(tolerance) != 1:
            raise ConfigError(
                f"hardware.autocal.auto_accept.{address} must contain "
                "exactly one tolerance"
            )
        kind, amount = next(iter(tolerance.items()))
        if kind not in allowed_tolerances:
            raise ConfigError(
                f"hardware.autocal.auto_accept.{address} has unknown "
                f"tolerance {kind!r}"
            )
        if kind == "always":
            if amount is not True:
                raise ConfigError(
                    f"hardware.autocal.auto_accept.{address}.always must be true"
                )
        elif _finite_number(
            amount,
            f"hardware.autocal.auto_accept.{address}.{kind}",
        ) < 0:
            raise ConfigError(
                f"hardware.autocal.auto_accept.{address}.{kind} "
                "cannot be negative"
            )
    budgets = value.get("budgets", {})
    if not isinstance(budgets, Mapping):
        raise ConfigError("hardware.autocal.budgets must be a mapping")
    allowed_budgets = {
        "max_wall_clock_hours",
        "max_node_attempts",
        "max_total_runs",
    }
    if set(budgets).difference(allowed_budgets):
        raise ConfigError("hardware.autocal.budgets has unknown keys")
    for name, amount in budgets.items():
        number = _finite_number(
            amount,
            f"hardware.autocal.budgets.{name}",
        )
        if number <= 0:
            raise ConfigError(
                f"hardware.autocal.budgets.{name} must be positive"
            )
        if (
            name in {"max_node_attempts", "max_total_runs"}
            and not number.is_integer()
        ):
            raise ConfigError(
                f"hardware.autocal.budgets.{name} must be an integer"
            )
    caps = value.get("caps", {})
    if not isinstance(caps, Mapping):
        raise ConfigError("hardware.autocal.caps must be a mapping")
    allowed_caps = {"q_gain_max", "r_power_max_db"}
    if set(caps).difference(allowed_caps):
        raise ConfigError("hardware.autocal.caps has unknown keys")
    if "q_gain_max" in caps:
        q_gain_max = _finite_number(
            caps["q_gain_max"],
            "hardware.autocal.caps.q_gain_max",
        )
        if q_gain_max <= 0:
            raise ConfigError(
                "hardware.autocal.caps.q_gain_max must be positive"
            )
        q_gain_limits = limits.get("q_gain")
        if (
            isinstance(q_gain_limits, Sequence)
            and not isinstance(q_gain_limits, (str, bytes))
            and len(q_gain_limits) == 2
        ):
            lower = _finite_number(
                q_gain_limits[0],
                "limits.q_gain[0]",
            )
            upper = _finite_number(
                q_gain_limits[1],
                "limits.q_gain[1]",
            )
            if lower > -q_gain_max or upper < q_gain_max:
                raise ConfigError(
                    "hardware.autocal.caps.q_gain_max does not fit symmetrically "
                    "inside hardware.limits.q_gain"
                )
    if "r_power_max_db" in caps:
        r_power_max = _finite_number(
            caps["r_power_max_db"],
            "hardware.autocal.caps.r_power_max_db",
        )
        if r_power_max > 0:
            raise ConfigError(
                "hardware.autocal.caps.r_power_max_db cannot be positive"
            )
        r_power_limits = limits.get("r_power")
        if (
            isinstance(r_power_limits, Sequence)
            and not isinstance(r_power_limits, (str, bytes))
            and len(r_power_limits) == 2
        ):
            lower = _finite_number(
                r_power_limits[0],
                "limits.r_power[0]",
            )
            upper = _finite_number(
                r_power_limits[1],
                "limits.r_power[1]",
            )
            if not lower <= r_power_max <= upper:
                raise ConfigError(
                    "hardware.autocal.caps.r_power_max_db is outside "
                    "hardware.limits.r_power"
                )
    ramsey = value.get("ramsey_sign", {})
    if not isinstance(ramsey, Mapping):
        raise ConfigError("hardware.autocal.ramsey_sign must be a mapping")
    if set(ramsey).difference({"require_two_point_confirmation"}):
        raise ConfigError("hardware.autocal.ramsey_sign has unknown keys")
    confirmation = ramsey.get("require_two_point_confirmation", True)
    if not isinstance(confirmation, bool):
        raise ConfigError(
            "hardware.autocal.ramsey_sign.require_two_point_confirmation "
            "must be boolean"
        )


def _validate_limits(parameters: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    for name, bounds in limits.items():
        if name not in parameters:
            continue
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, (str, bytes))
            or len(bounds) != 2
        ):
            raise ConfigError(f"hardware.limits.{name} must be [minimum, maximum]")
        minimum = _finite_number(bounds[0], f"limits.{name}[0]")
        maximum = _finite_number(bounds[1], f"limits.{name}[1]")
        if minimum > maximum:
            raise ConfigError(f"hardware.limits.{name} has reversed bounds")
        value = parameters[name]
        if isinstance(value, Mapping):
            candidates = [
                value[key]
                for key in ("start", "stop", "center")
                if key in value
            ]
        elif isinstance(value, np.ndarray):
            candidates = value.ravel().tolist()
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            candidates = [value]
        for candidate in candidates:
            number = _finite_number(candidate, f"parameter {name}")
            if not minimum <= number <= maximum:
                raise ConfigError(
                    f"parameter {name}={number} is outside [{minimum}, {maximum}]"
                )


@dataclass(frozen=True)
class ResolvedConfig:
    """Immutable view of one selected experiment invocation."""

    experiment: str
    preset_name: Optional[str]
    parameters: dict
    run: dict
    analysis: dict
    acceptance: dict
    references: dict
    hardware: dict
    calibration: dict
    sources: dict
    overrides: dict
    config_fingerprint: str

    def get(self, path: str, default: Any = None) -> Any:
        context = {
            "parameters": self.parameters,
            "defaults": self.parameters,
            "run": self.run,
            "analysis": self.analysis,
            "acceptance": self.acceptance,
            "hardware": self.hardware,
        }
        return dotted_get(context, path, default)

    def expanded_parameters(self) -> dict:
        context = deep_merge(
            self.hardware,
            {"defaults": self.references, "parameters": self.parameters},
        )
        return {
            key: expand_sweep(value, context, label=f"parameters.{key}")
            for key, value in self.parameters.items()
        }

    def as_manifest(self) -> dict:
        return {
            "experiment": self.experiment,
            "preset": self.preset_name,
            "parameters": deepcopy(self.parameters),
            "expanded_parameters": self.expanded_parameters(),
            "run": deepcopy(self.run),
            "analysis": deepcopy(self.analysis),
            "acceptance": deepcopy(self.acceptance),
            "reference_values": deepcopy(self.references),
            "hardware": deepcopy(self.hardware),
            "calibration": deepcopy(self.calibration),
            "sources": deepcopy(self.sources),
            "overrides": deepcopy(self.overrides),
            "config_fingerprint": self.config_fingerprint,
        }


class ConfigRepository:
    """Validated source documents from which immutable runs are resolved."""

    def __init__(
        self,
        hardware: Mapping[str, Any],
        calibration: Optional[Mapping[str, Any]] = None,
        presets: Optional[Mapping[str, Any]] = None,
        *,
        sources: Optional[Mapping[str, Optional[str]]] = None,
    ):
        self.hardware = deepcopy(dict(hardware))
        self.calibration = deepcopy(dict(calibration or {}))
        presets_document = deepcopy(dict(presets or {}))
        _validate_hardware(self.hardware)
        _check_schema(self.calibration, "calibration")
        _check_schema(presets_document, "presets")
        raw_presets = presets_document.get("presets", presets_document)
        if not isinstance(raw_presets, Mapping):
            raise ConfigError("presets.presets must be a mapping")
        self.presets = deepcopy(dict(raw_presets))
        self.sources = deepcopy(dict(sources or {}))

    @classmethod
    def from_files(
        cls,
        hardware_path: Union[str, Path],
        calibration_path: Optional[Union[str, Path]] = None,
        presets_path: Optional[Union[str, Path]] = None,
    ) -> "ConfigRepository":
        hardware = _read_yaml(hardware_path, required=True)
        calibration = _read_yaml(calibration_path, required=False)
        presets = _read_yaml(presets_path, required=False)
        return cls(
            hardware,
            calibration,
            presets,
            sources={
                "hardware": str(Path(hardware_path).resolve()),
                "calibration": (
                    str(Path(calibration_path).resolve())
                    if calibration_path is not None
                    else None
                ),
                "presets": (
                    str(Path(presets_path).resolve())
                    if presets_path is not None
                    else None
                ),
            },
        )

    def preset_names(self) -> list:
        return sorted(self.presets)

    def resolve(
        self,
        preset: Union[str, Mapping[str, Any]],
        *,
        experiment: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> ResolvedConfig:
        if isinstance(preset, str):
            if preset not in self.presets:
                choices = ", ".join(self.preset_names()) or "<none>"
                raise ConfigError(
                    f"unknown preset {preset!r}; available presets: {choices}"
                )
            preset_name = preset
            selected = deepcopy(dict(self.presets[preset]))
        elif isinstance(preset, Mapping):
            preset_name = None
            selected = deepcopy(dict(preset))
        else:
            raise ConfigError("preset must be a name or mapping")

        preset_experiment = selected.get("experiment")
        experiment_name = experiment or preset_experiment
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ConfigError("a preset must declare a non-empty experiment")
        if experiment and preset_experiment and experiment != preset_experiment:
            raise ConfigError(
                f"preset {preset_name!r} targets {preset_experiment!r}, "
                f"not {experiment!r}"
            )
        parameters = selected.get("parameters", {})
        run = selected.get("run", {})
        analysis = selected.get("analysis", {})
        acceptance = selected.get("acceptance", {})
        for label, value in (
            ("parameters", parameters),
            ("run", run),
            ("analysis", analysis),
            ("acceptance", acceptance),
        ):
            if not isinstance(value, Mapping):
                raise ConfigError(f"preset.{label} must be a mapping")

        accepted = accepted_calibration_values(self.calibration)
        _reject_protected(accepted, "accepted calibration")
        _reject_protected(parameters, "preset parameters")
        override_layer = deepcopy(dict(overrides or {}))
        _reject_protected(override_layer, "runtime overrides")
        base_parameters = self.hardware.get("defaults", {})
        reference_values = deep_merge(
            base_parameters,
            accepted.get("defaults", accepted),
        )
        merged_parameters = _merge_parameter_layers(
            reference_values,
            parameters,
            override_layer,
        )
        limits = self.hardware.get("limits", {})
        _validate_limits(merged_parameters, limits)
        resolution_context = deep_merge(
            self.hardware,
            {"defaults": reference_values, "parameters": merged_parameters},
        )
        expanded_parameters = {
            key: expand_sweep(
                value,
                resolution_context,
                label=f"parameters.{key}",
            )
            for key, value in merged_parameters.items()
        }
        _validate_limits(expanded_parameters, limits)
        payload = {
            "experiment": experiment_name,
            "preset": preset_name,
            "parameters": merged_parameters,
            "run": run,
            "analysis": analysis,
            "acceptance": acceptance,
            "hardware": self.hardware,
            "calibration_revision": self.calibration.get("revision"),
            "sources": self.sources,
            "overrides": override_layer,
        }
        return ResolvedConfig(
            experiment=experiment_name,
            preset_name=preset_name,
            parameters=deepcopy(merged_parameters),
            run=deepcopy(dict(run)),
            analysis=deepcopy(dict(analysis)),
            acceptance=deepcopy(dict(acceptance)),
            references=deepcopy(reference_values),
            hardware=deepcopy(self.hardware),
            calibration=deepcopy(self.calibration),
            sources=deepcopy(self.sources),
            overrides=override_layer,
            config_fingerprint=fingerprint(payload),
        )
