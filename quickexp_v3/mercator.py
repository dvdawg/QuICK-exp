"""Typed authoring and preflight for the Mercator config format used by Quick."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
import difflib
import math
import re
from typing import Any, Mapping, Optional

import numpy as np
import yaml

from .errors import ConfigError


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CHANNEL_ROLES = frozenset({"r", "q", "z", "rr"})
_PULSE_FIELDS = (
    "freq",
    "gain",
    "power",
    "length",
    "style",
    "sigma",
    "mixer",
    "nqz",
    "mode",
    "phase",
)
_PULSE_ORDER = (
    "style",
    "mode",
    "freq",
    "mixer",
    "nqz",
    "length",
    "sigma",
    "gain",
    "power",
    "phase",
)
_CONFIG_FIELDS = ("hard_avg", "soft_avg", "rep")
_ALLOWED_STYLES = frozenset({"const", "gaussian", "flat_top"})
_ALLOWED_EXPR_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
)


@dataclass(frozen=True)
class Var:
    """A declared Mercator variable rendered as ``{name}``."""

    name: str

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(str(self.name)):
            raise ConfigError(f"invalid Mercator variable name {self.name!r}")


@dataclass(frozen=True)
class Expr:
    """A deliberately small arithmetic template expression."""

    source: str

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        if not source:
            raise ConfigError("Mercator expression cannot be empty")
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise ConfigError(f"invalid Mercator expression {source!r}") from error
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_EXPR_NODES):
                raise ConfigError(
                    f"Mercator expression uses unsupported syntax "
                    f"{type(node).__name__}: {source!r}"
                )
            if isinstance(node, ast.Call):
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id != "int"
                    or len(node.args) != 1
                    or node.keywords
                ):
                    raise ConfigError(
                        "Mercator expressions may call only int(value)"
                    )
        object.__setattr__(self, "source", source)


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    errors: tuple
    warnings: tuple
    details: Mapping[str, Any]


def _unknown_field(kind: str, name: str, valid: tuple) -> ConfigError:
    nearest = difflib.get_close_matches(str(name), valid, n=1)
    suggestion = f"; did you mean {nearest[0]!r}?" if nearest else ""
    return ConfigError(
        f"unknown Mercator {kind} field {name!r}{suggestion}"
    )


def _template(value: Any) -> str:
    if isinstance(value, Var):
        return "{" + value.name + "}"
    if isinstance(value, Expr):
        return "{" + value.source + "}"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ConfigError("Mercator config values must be finite")
        return repr(number)
    if isinstance(value, str):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
            return value
        return yaml.safe_dump(value, default_flow_style=True).strip().splitlines()[0]
    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
    return dumped[:-4].rstrip() if dumped.endswith("\n...") else dumped


def _expression_names(source: str) -> tuple:
    tree = ast.parse(source, mode="eval")
    return tuple(
        sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id != "int"
            }
        )
    )


def _safe_eval_scalar(source: str, variables: Mapping[str, Any]) -> Any:
    tree = ast.parse(source, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ConfigError(
                    f"Mercator expression references undeclared variable {node.id!r}"
                )
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.FloorDiv: lambda: left // right,
                ast.Mod: lambda: left % right,
                ast.Pow: lambda: left**right,
            }
            return operations[type(node.op)]()
        if isinstance(node, ast.Call):
            return int(evaluate(node.args[0]))
        raise ConfigError(
            f"unsupported Mercator expression node {type(node).__name__}"
        )

    return evaluate(tree)


def _evaluate(value: Any, variables: Mapping[str, Any]) -> np.ndarray:
    if isinstance(value, Var):
        if value.name not in variables:
            raise ConfigError(
                f"Mercator variable {value.name!r} has no runtime value"
            )
        return np.asarray(variables[value.name])
    if not isinstance(value, Expr):
        return np.asarray(value)
    names = _expression_names(value.source)
    arrays = {
        name: np.asarray(variables[name])
        for name in names
        if name in variables
    }
    missing = set(names).difference(arrays)
    if missing:
        raise ConfigError(
            "Mercator expression has no runtime value for "
            + ", ".join(sorted(missing))
        )
    swept = {
        name: array.ravel()
        for name, array in arrays.items()
        if array.size > 1
    }
    if not swept:
        scalars = {
            name: array.ravel()[0]
            for name, array in arrays.items()
        }
        return np.asarray(_safe_eval_scalar(value.source, scalars))
    sweep_names = tuple(swept)
    mesh = np.meshgrid(
        *(swept[name] for name in sweep_names),
        indexing="ij",
    )
    results = []
    for index in range(mesh[0].size):
        context = {}
        for name, array in arrays.items():
            if name in swept:
                context[name] = mesh[sweep_names.index(name)].ravel()[index]
            else:
                context[name] = array.ravel()[0]
        results.append(_safe_eval_scalar(value.source, context))
    return np.asarray(results)


def _finite_values(
    value: Any,
    variables: Mapping[str, Any],
    *,
    label: str,
) -> np.ndarray:
    try:
        array = np.asarray(_evaluate(value, variables), dtype=float).ravel()
    except (TypeError, ValueError, OverflowError) as error:
        raise ConfigError(f"{label} must resolve to finite numbers") from error
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ConfigError(f"{label} must resolve to finite numbers")
    return array


class MercatorProgram:
    """Small, deterministic builder for the observed Quick Mercator schema."""

    def __init__(self, name: str):
        normalized = str(name).strip()
        if not _IDENTIFIER.fullmatch(normalized):
            raise ConfigError(f"invalid Mercator program name {name!r}")
        self.name = normalized
        self._config = {}
        self._pulses = {}
        self._readouts = []
        self._steps = []
        self._extras = {}
        self._variables = {}
        self._labels = {}
        self._envelope_terms = []

    def declare_variable(
        self,
        name: str,
        default: Any,
        label: str,
        unit: str,
    ) -> "MercatorProgram":
        key = str(name)
        if not _IDENTIFIER.fullmatch(key):
            raise ConfigError(f"invalid Mercator variable name {name!r}")
        if key in self._variables:
            raise ConfigError(f"Mercator variable {key!r} is already declared")
        self._variables[key] = deepcopy(default)
        self._labels[key] = (str(label), str(unit))
        return self

    def declare_envelope_term(
        self,
        variable_name: str,
        count: int,
    ) -> "MercatorProgram":
        name = str(variable_name)
        if name not in self._variables:
            raise ConfigError(
                f"envelope term references undeclared variable {name!r}"
            )
        if int(count) < 1:
            raise ConfigError("envelope count must be at least one")
        if any(existing[0] == name for existing in self._envelope_terms):
            raise ConfigError(f"envelope term {name!r} is already declared")
        self._envelope_terms.append((name, int(count)))
        return self

    def config(
        self,
        hard_avg: Any = None,
        soft_avg: Any = None,
        rep: Any = None,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field("config", next(iter(unknown)), _CONFIG_FIELDS)
        for name, value in (
            ("hard_avg", hard_avg),
            ("soft_avg", soft_avg),
            ("rep", rep),
        ):
            if value is not None:
                self._config[name] = value
        return self

    def pulse(
        self,
        slot: int,
        *,
        freq: Any,
        length: Any,
        style: Optional[str] = None,
        gain: Any = None,
        power: Any = None,
        sigma: Any = None,
        mixer: Any = None,
        nqz: Any = None,
        mode: Optional[str] = None,
        phase: Any = None,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field("pulse", next(iter(unknown)), _PULSE_FIELDS)
        number = int(slot)
        if number < 0 or number in self._pulses:
            raise ConfigError(
                f"Mercator pulse slot {number} must be non-negative and unique"
            )
        if gain is not None and power is not None:
            raise ConfigError(
                f"pulse slot {number} cannot declare both gain and power"
            )
        if style is not None and style not in _ALLOWED_STYLES:
            raise ConfigError(
                f"pulse slot {number} style must be one of "
                + ", ".join(sorted(_ALLOWED_STYLES))
            )
        if sigma is not None and style != "flat_top":
            raise ConfigError(
                f"pulse slot {number} sigma is valid only for flat_top"
            )
        fields = {
            "freq": freq,
            "length": length,
            "style": style,
            "gain": gain,
            "power": power,
            "sigma": sigma,
            "mixer": mixer,
            "nqz": nqz,
            "mode": mode,
            "phase": phase,
        }
        self._pulses[number] = {
            name: value for name, value in fields.items() if value is not None
        }
        return self

    def readout(
        self,
        channel: Any,
        *,
        p: int,
        length: Any,
        phase: Any,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field(
                "readout",
                next(iter(unknown)),
                ("p", "length", "phase"),
            )
        if int(p) not in self._pulses:
            raise ConfigError(f"readout references undefined pulse slot {p}")
        if isinstance(channel, Var) and channel.name != "rr":
            raise ConfigError("readout channel Var must be Var('rr')")
        if not isinstance(channel, (Var, int, np.integer)):
            raise ConfigError("readout channel must be an integer or Var('rr')")
        self._readouts.append(
            (channel, {"p": int(p), "length": length, "phase": phase})
        )
        return self

    def step_pulse(
        self,
        p: int,
        g: Any,
        *,
        r: Any = None,
        threshold: Any = None,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field(
                "pulse step",
                next(iter(unknown)),
                ("p", "g", "r", "threshold"),
            )
        if int(p) not in self._pulses:
            raise ConfigError(f"pulse step references undefined slot {p}")
        if isinstance(g, Var) and g.name not in _CHANNEL_ROLES:
            raise ConfigError(
                f"pulse-step generator role {g.name!r} must be one of "
                + ", ".join(sorted(_CHANNEL_ROLES))
            )
        if not isinstance(g, (Var, int, np.integer)):
            raise ConfigError("pulse-step generator must be an integer or channel Var")
        if (r is None) != (threshold is None):
            raise ConfigError(
                "conditional pulse steps require both r and threshold"
            )
        step = {"type": "pulse", "p": int(p), "g": g}
        if r is not None:
            step.update({"r": r, "threshold": threshold})
        self._steps.append(step)
        return self

    def step_trigger(self, t: Any, **unknown: Any) -> "MercatorProgram":
        if unknown:
            raise _unknown_field("trigger step", next(iter(unknown)), ("t",))
        self._steps.append({"type": "trigger", "t": t})
        return self

    def step_delay_auto(
        self,
        t: Any = None,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field("delay step", next(iter(unknown)), ("t",))
        step = {"type": "delay_auto"}
        if t is not None:
            step["t"] = t
        self._steps.append(step)
        return self

    def step_goto(
        self,
        i: int,
        *,
        rep: Any = None,
        t: Any = None,
        **unknown: Any,
    ) -> "MercatorProgram":
        if unknown:
            raise _unknown_field(
                "goto step",
                next(iter(unknown)),
                ("i", "rep", "t"),
            )
        step = {"type": "goto", "i": int(i)}
        if rep is not None:
            step["rep"] = rep
        if t is not None:
            step["t"] = t
        self._steps.append(step)
        return self

    def extra(self, key: str, value: Any) -> "MercatorProgram":
        name = str(key).strip()
        if not name or "\n" in name or name in self._extras:
            raise ConfigError(f"invalid or duplicate Mercator extra key {key!r}")
        self._extras[name] = value
        return self

    def _references(self) -> tuple:
        values = list(self._config.values()) + list(self._extras.values())
        for fields in self._pulses.values():
            values.extend(fields.values())
        for channel, fields in self._readouts:
            values.append(channel)
            values.extend(fields.values())
        for step in self._steps:
            values.extend(step.values())
        names = set()
        for value in values:
            if isinstance(value, Var):
                names.add(value.name)
            elif isinstance(value, Expr):
                names.update(_expression_names(value.source))
        return tuple(sorted(names))

    def _validate_references(self) -> None:
        missing = set(self._references()).difference(self._variables)
        if missing:
            raise ConfigError(
                f"{self.name} references undeclared variables: "
                + ", ".join(sorted(missing))
            )

    def render(self) -> str:
        self._validate_references()
        lines = []
        for name in _CONFIG_FIELDS:
            if name in self._config:
                lines.append(f"{name}: {_template(self._config[name])}")
        if lines:
            lines.append("")
        for slot in sorted(self._pulses):
            fields = self._pulses[slot]
            for name in _PULSE_ORDER:
                if name in fields:
                    lines.append(
                        f"p{slot}_{name}: {_template(fields[name])}"
                    )
        if self._pulses:
            lines.append("")
        for channel, fields in self._readouts:
            rendered_channel = _template(channel)
            for name in ("p", "length", "phase"):
                lines.append(
                    f"r{rendered_channel}_{name}: {_template(fields[name])}"
                )
        if self._readouts:
            lines.append("")
        for name in sorted(self._extras):
            lines.append(f"{name}: {_template(self._extras[name])}")
        if self._extras:
            lines.append("")
        lines.append("steps:")
        for step in self._steps:
            lines.append(f"- type: {step['type']}")
            for name in ("p", "g", "r", "threshold", "t", "i", "rep"):
                if name in step:
                    lines.append(f"  {name}: {_template(step[name])}")
        return "\n".join(lines).rstrip() + "\n"

    def default_variables(self) -> dict:
        return deepcopy(self._variables)

    def variable_labels(self) -> dict:
        return deepcopy(self._labels)

    def envelope_terms(self) -> tuple:
        return tuple(self._envelope_terms)

    def _generator_for_slot(
        self,
        slot: int,
        variables: Mapping[str, Any],
    ) -> Optional[int]:
        channels = []
        for step in self._steps:
            if step.get("type") == "pulse" and step.get("p") == slot:
                values = _finite_values(
                    step["g"],
                    variables,
                    label=f"{self.name} pulse slot {slot} generator",
                )
                channels.extend(int(value) for value in values)
        return channels[0] if channels and len(set(channels)) == 1 else None

    @staticmethod
    def _generator_details(soccfg: Any, channel: int) -> Optional[Mapping[str, Any]]:
        try:
            value = soccfg["gens"][int(channel)]
        except (KeyError, TypeError, IndexError):
            return None
        return value if isinstance(value, Mapping) else None

    def preflight(
        self,
        soccfg: Any,
        variables: Mapping[str, Any],
    ) -> PreflightReport:
        """Validate memory, clocks, clipping, mixer continuity, and duty cycle."""
        self._validate_references()
        runtime = self.default_variables()
        runtime.update(dict(variables))
        errors = []
        warnings = []
        details = {
            "program": self.name,
            "envelopes": {},
            "timing": {},
            "mixers": {},
            "unverified_keys": tuple(sorted(self._extras)),
            "simultaneous_gain_check": "not implemented; schema has no observed parallel step",
        }
        if self._extras:
            warnings.append(
                "unverified Mercator extra keys: "
                + ", ".join(sorted(self._extras))
            )

        swept = {
            name
            for name, value in runtime.items()
            if np.asarray(value).ndim == 1 and np.asarray(value).size > 1
        }
        explicit_bindings = set(self._references())
        implicit_delays = sum(
            step["type"] == "delay_auto" and "t" not in step
            for step in self._steps
        )
        unresolved_sweeps = swept.difference(explicit_bindings)
        if len(unresolved_sweeps) > implicit_delays:
            errors.append(
                f"{self.name} has {len(unresolved_sweeps)} unbound swept "
                f"variable(s) {sorted(unresolved_sweeps)} but "
                f"{implicit_delays} implicit delay slot(s)"
            )

        declared_terms = dict(self._envelope_terms)
        for slot, fields in self._pulses.items():
            style = str(fields.get("style", "const"))
            length = fields["length"]
            variable_name = length.name if isinstance(length, Var) else None
            if style != "const" and variable_name not in declared_terms:
                errors.append(
                    f"pulse slot {slot} uses {style} but its length has no "
                    "declared envelope term"
                )
        memory_by_channel = {}
        for variable_name, count in self._envelope_terms:
            matching = [
                (slot, fields)
                for slot, fields in self._pulses.items()
                if isinstance(fields.get("length"), Var)
                and fields["length"].name == variable_name
            ]
            if not matching:
                errors.append(
                    f"envelope term {variable_name!r} is not used by a pulse length"
                )
                continue
            slot, fields = matching[0]
            channel = self._generator_for_slot(slot, runtime)
            if channel is None:
                errors.append(
                    f"pulse slot {slot} envelope has no unique generator channel"
                )
                continue
            generator = self._generator_details(soccfg, channel)
            if generator is None:
                warnings.append(
                    f"generator {channel} has no memory metadata; envelope "
                    "capacity was not checked"
                )
                continue
            try:
                f_fabric = float(generator["f_fabric"])
                samples_per_clock = int(generator["samps_per_clk"])
                available = int(generator["maxlen"])
            except (KeyError, TypeError, ValueError):
                warnings.append(
                    f"generator {channel} has incomplete memory metadata"
                )
                continue
            lengths = _finite_values(
                fields["length"],
                runtime,
                label=f"{self.name}.p{slot}_length",
            )
            style = str(fields.get("style", "const"))
            budget_length = float(np.max(lengths))
            if style == "flat_top":
                sigma = _finite_values(
                    fields.get("sigma"),
                    runtime,
                    label=f"{self.name}.p{slot}_sigma",
                )
                budget_length = min(
                    budget_length,
                    8.0 * float(np.max(sigma)),
                )
                warnings.append(
                    f"pulse slot {slot} flat_top memory assumes 8σ ramps "
                    "(hardware gate V4 pending)"
                )
            elif style == "const":
                budget_length = 0.0
                warnings.append(
                    f"pulse slot {slot} const memory assumes no envelope RAM "
                    "(hardware gate V5 pending)"
                )
            samples = int(budget_length * f_fabric) * samples_per_clock * int(count)
            memory_by_channel[channel] = memory_by_channel.get(channel, 0) + samples
            details["envelopes"][variable_name] = {
                "slot": slot,
                "generator": channel,
                "style": style,
                "budget_length_us": budget_length,
                "count": int(count),
                "required_samples": samples,
                "available_samples": available,
            }
            if memory_by_channel[channel] > available:
                errors.append(
                    f"generator {channel} envelope memory requires "
                    f"{memory_by_channel[channel]} samples but only "
                    f"{available} are available"
                )

        def check_timing(label, raw, channel=None):
            values = _finite_values(raw, runtime, label=label)
            positive = values[values > 0]
            if positive.size == 0:
                return
            generator = (
                self._generator_details(soccfg, int(channel))
                if channel is not None
                else None
            )
            f_fabric = (
                float(generator["f_fabric"])
                if generator is not None and "f_fabric" in generator
                else None
            )
            if f_fabric is None:
                return
            clocks = positive * f_fabric
            details["timing"][label] = {
                "minimum_clocks": float(np.min(clocks)),
                "maximum_clocks": float(np.max(clocks)),
            }
            if np.min(clocks) < 1.0:
                errors.append(
                    f"{label} is shorter than one fabric clock "
                    f"({float(np.min(clocks)):.3g} clocks)"
                )
            elif np.min(clocks) < 3.0:
                warnings.append(
                    f"{label} is shorter than three fabric clocks "
                    f"({float(np.min(clocks)):.3g} clocks)"
                )

        for slot, fields in self._pulses.items():
            channel = self._generator_for_slot(slot, runtime)
            check_timing(
                f"p{slot}_length",
                fields["length"],
                channel=channel,
            )
            if "sigma" in fields:
                check_timing(
                    f"p{slot}_sigma",
                    fields["sigma"],
                    channel=channel,
                )
            if "gain" in fields:
                gain = _finite_values(
                    fields["gain"],
                    runtime,
                    label=f"p{slot}_gain",
                )
                if np.any(np.abs(gain) > 1.0):
                    errors.append(f"p{slot}_gain exceeds full scale |gain| <= 1")
            if "power" in fields:
                power = _finite_values(
                    fields["power"],
                    runtime,
                    label=f"p{slot}_power",
                )
                if np.any(power > 0):
                    errors.append(f"p{slot}_power exceeds the 0 dB full-scale limit")
            if "mixer" in fields:
                frequency = _finite_values(
                    fields["freq"],
                    runtime,
                    label=f"p{slot}_freq",
                )
                mixer = _finite_values(
                    fields["mixer"],
                    runtime,
                    label=f"p{slot}_mixer",
                )
                unique_mixers = np.unique(np.round(mixer, decimals=12))
                details["mixers"][slot] = {
                    "frequencies_mhz": (
                        float(np.min(frequency)),
                        float(np.max(frequency)),
                    ),
                    "mixer_values_mhz": unique_mixers.tolist(),
                }
                if unique_mixers.size > 1:
                    errors.append(
                        f"p{slot}_mixer changes across the sweep: "
                        + ", ".join(f"{value:g}" for value in unique_mixers)
                        + " MHz"
                    )
                if mixer.size == 1:
                    mixer = np.full(frequency.size, mixer[0])
                if frequency.size == 1:
                    frequency = np.full(mixer.size, frequency[0])
                if frequency.size == mixer.size and np.any(
                    np.abs(frequency - mixer) > 500.0 + 1e-9
                ):
                    errors.append(
                        f"p{slot} frequency is more than 500 MHz from its mixer"
                    )

        timing_channel = None
        for slot in sorted(self._pulses):
            timing_channel = self._generator_for_slot(slot, runtime)
            if timing_channel is not None:
                break
        sequence_us = 0.0
        for fields in self._pulses.values():
            sequence_us += float(
                np.max(
                    _finite_values(
                        fields["length"],
                        runtime,
                        label="pulse length",
                    )
                )
            )
        for index, step in enumerate(self._steps):
            if "t" in step:
                check_timing(
                    f"step{index}_t",
                    step["t"],
                    channel=timing_channel,
                )
                sequence_us += float(
                    np.max(
                        _finite_values(
                            step["t"],
                            runtime,
                            label=f"step{index}_t",
                        )
                    )
                )
        relax = runtime.get("r_relax")
        relax_values = (
            _finite_values(relax, runtime, label="r_relax")
            if relax is not None
            else np.asarray([])
        )
        if relax_values.size and np.max(relax_values) > 0:
            duty = sequence_us / (sequence_us + float(np.max(relax_values)))
            details["sequence"] = {
                "duration_upper_bound_us": sequence_us,
                "r_relax_us": float(np.max(relax_values)),
                "duty_cycle_upper_bound": duty,
            }
            if duty > 0.2:
                warnings.append(
                    f"sequence duty-cycle upper bound is {100*duty:.1f}% (>20%)"
                )
        return PreflightReport(
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            details=details,
        )


def install_program(
    quick_module: Any,
    program: MercatorProgram,
    *,
    channel_variables: Optional[Mapping[str, int]] = None,
):
    """Install one authored template and BaseExperiment subclass idempotently."""
    experiment_module = quick_module.experiment
    existing = getattr(experiment_module, program.name, None)
    if existing is not None:
        return existing
    configs = getattr(experiment_module, "configs", None)
    if not isinstance(configs, dict):
        raise ConfigError("quick.experiment.configs must be a mutable mapping")
    base_experiment = getattr(experiment_module, "BaseExperiment", None)
    if base_experiment is None:
        raise ConfigError("quick.experiment.BaseExperiment is unavailable")
    defaults = program.default_variables()
    if channel_variables:
        for role in _CHANNEL_ROLES:
            if role in channel_variables and role in defaults:
                defaults[role] = int(channel_variables[role])
    labels = program.variable_labels()
    configs[program.name] = program.render()

    class AuthoredExperiment(base_experiment):
        def __init__(self, **kwargs):
            self.var = deepcopy(defaults)
            self.var_label = deepcopy(labels)
            super().__init__(**kwargs)

        def run(self, silent=False, dB=False, **kwargs):
            return super().run(silent=silent, dB=dB, **kwargs)

    AuthoredExperiment.__name__ = program.name
    AuthoredExperiment.__qualname__ = program.name
    setattr(experiment_module, program.name, AuthoredExperiment)
    return AuthoredExperiment


def held_z_program(name: str = "SetZBiasAndWaitV3") -> MercatorProgram:
    """Reconstruct the existing held-Z helper for offline parity checks."""
    program = MercatorProgram(name)
    declarations = (
        ("r", 0, "Readout DAC Generator", ""),
        ("rr", 0, "Readout ADC Channel", ""),
        ("z", 15, "Z DAC Generator", ""),
        ("r_freq", 6883.0, "Readout Frequency", "MHz"),
        ("r_length", 2.0, "Readout Length", "us"),
        ("r_power", -35.0, "Readout Power", "dB"),
        ("r_phase", 0.0, "Readout Phase", "deg"),
        ("r_offset", 0.495, "Readout Trigger Offset", "us"),
        ("r_relax", 20.0, "Readout Relax Time", "us"),
        ("z_gain", 0.0, "Z Gain", "a.u."),
        ("z_length", 0.2, "Z Hold Pulse Length", "us"),
        ("z_settle", 5.0, "Z Settle Time", "us"),
    )
    for declaration in declarations:
        program.declare_variable(*declaration)
    return (
        program.config(hard_avg=1, soft_avg=1)
        .pulse(
            0,
            freq=Var("r_freq"),
            mixer=Expr("int(r_freq/1000)*1000+500"),
            length=Var("r_length"),
            power=Var("r_power"),
        )
        .pulse(
            9,
            style="const",
            mode="last",
            freq=0,
            mixer=0,
            nqz=1,
            length=Var("z_length"),
            gain=Var("z_gain"),
        )
        .readout(
            Var("rr"),
            p=0,
            length=Var("r_length"),
            phase=Var("r_phase"),
        )
        .step_pulse(9, Var("z"))
        .step_delay_auto(Var("z_settle"))
        .step_pulse(0, Var("r"))
        .step_trigger(Var("r_offset"))
        .step_delay_auto(Var("r_relax"))
    )
