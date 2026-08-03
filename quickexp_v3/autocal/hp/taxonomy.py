"""Declared physical hypotheses and their perturbation signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class Signature:
    probe_id: str
    observable: str
    predicted: Callable[[Mapping[str, Any]], float]
    tolerance: Callable[[Mapping[str, Any]], float]
    weight: float = 1.0


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    family: str
    meaning: str
    signatures: Tuple[Signature, ...] = ()


def _value(
    context: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = float(context.get(key, default))
    if not np.isfinite(value):
        raise ValueError("hypothesis context value must be finite: " + key)
    return value


def _prediction(key: str, default: float) -> Callable[[Mapping[str, Any]], float]:
    return lambda context: _value(context, key, default)


def _scaled_prediction(
    key: str,
    default: float,
    factor: float,
) -> Callable[[Mapping[str, Any]], float]:
    return lambda context: factor * _value(context, key, default)


def _zero(_context: Mapping[str, Any]) -> float:
    return 0.0


def _tolerance(key: str) -> Callable[[Mapping[str, Any]], float]:
    def evaluate(context: Mapping[str, Any]) -> float:
        value = abs(_value(context, key, float("nan")))
        if value <= np.finfo(float).eps:
            raise ValueError("hypothesis tolerance must be positive: " + key)
        return value

    return evaluate


def _qubit_signatures(
    *,
    power_exponent: Callable[[Mapping[str, Any]], float],
    flux_slope: Callable[[Mapping[str, Any]], float],
    flux_curvature: Callable[[Mapping[str, Any]], float],
    rabi_exponent: Callable[[Mapping[str, Any]], float],
    rabi_contrast: Callable[[Mapping[str, Any]], float],
    dispersive_shift: Callable[[Mapping[str, Any]], float],
) -> Tuple[Signature, ...]:
    return (
        Signature(
            "drive_power_ladder",
            "contrast_power_exponent",
            power_exponent,
            _tolerance("power_exponent_tolerance"),
            1.5,
        ),
        Signature(
            "flux_nudge",
            "flux_slope_mhz_per_z",
            flux_slope,
            _tolerance("flux_slope_tolerance_mhz_per_z"),
            1.0,
        ),
        Signature(
            "flux_nudge",
            "flux_curvature_mhz_per_z2",
            flux_curvature,
            _tolerance("flux_curvature_tolerance_mhz_per_z2"),
            0.75,
        ),
        Signature(
            "rabi_ping",
            "rabi_gain_exponent",
            rabi_exponent,
            _tolerance("rabi_exponent_tolerance"),
            1.5,
        ),
        Signature(
            "rabi_ping",
            "rabi_contrast",
            rabi_contrast,
            _tolerance("rabi_contrast_tolerance"),
            1.0,
        ),
        Signature(
            "dispersive_response",
            "dispersive_shift_mhz",
            dispersive_shift,
            _tolerance("dispersive_shift_tolerance_mhz"),
            1.5,
        ),
    )


_QUBIT_01 = Hypothesis(
    "qubit_01",
    "qubit",
    "wanted |0> to |1> transition",
    _qubit_signatures(
        power_exponent=_prediction("qubit_contrast_exponent", 1.0),
        flux_slope=_prediction("qubit_flux_slope_mhz_per_z", 0.0),
        flux_curvature=_prediction("qubit_flux_curvature_mhz_per_z2", 0.0),
        rabi_exponent=_prediction("qubit_rabi_gain_exponent", 1.0),
        rabi_contrast=_prediction("expected_rabi_contrast", 0.7),
        dispersive_shift=_prediction("expected_dispersive_shift_mhz", 1.0),
    ),
)

_F02 = Hypothesis(
    "f02_two_photon",
    "qubit",
    "two-photon |0> to |2> transition",
    _qubit_signatures(
        power_exponent=_prediction("f02_contrast_exponent", 2.0),
        flux_slope=_prediction("qubit_flux_slope_mhz_per_z", 0.0),
        flux_curvature=_prediction("qubit_flux_curvature_mhz_per_z2", 0.0),
        rabi_exponent=_prediction("f02_rabi_gain_exponent", 2.0),
        rabi_contrast=_scaled_prediction("expected_rabi_contrast", 0.7, 0.5),
        dispersive_shift=_scaled_prediction(
            "expected_dispersive_shift_mhz",
            1.0,
            0.5,
        ),
    ),
)

_HIGHER = Hypothesis(
    "higher_transition",
    "qubit",
    "higher transmon ladder transition",
    _qubit_signatures(
        power_exponent=_prediction("higher_transition_exponent", 1.5),
        flux_slope=_prediction("qubit_flux_slope_mhz_per_z", 0.0),
        flux_curvature=_prediction("qubit_flux_curvature_mhz_per_z2", 0.0),
        rabi_exponent=_prediction("higher_rabi_gain_exponent", 1.5),
        rabi_contrast=_scaled_prediction("expected_rabi_contrast", 0.7, 0.5),
        dispersive_shift=_scaled_prediction(
            "expected_dispersive_shift_mhz",
            1.0,
            0.25,
        ),
    ),
)

_TLS = Hypothesis(
    "tls",
    "qubit",
    "flux-insensitive saturating two-level-system defect",
    _qubit_signatures(
        power_exponent=_zero,
        flux_slope=_zero,
        flux_curvature=_zero,
        rabi_exponent=_zero,
        rabi_contrast=_zero,
        dispersive_shift=_zero,
    ),
)

_NEIGHBOR_QUBIT = Hypothesis(
    "neighbor_qubit",
    "qubit",
    "another qubit on the chip",
    _qubit_signatures(
        power_exponent=_prediction("qubit_contrast_exponent", 1.0),
        flux_slope=_prediction("neighbor_flux_slope_mhz_per_z", 0.0),
        flux_curvature=_prediction("neighbor_flux_curvature_mhz_per_z2", 0.0),
        rabi_exponent=_prediction("neighbor_rabi_gain_exponent", 1.0),
        rabi_contrast=_prediction("expected_neighbor_rabi_contrast", 0.0),
        dispersive_shift=_zero,
    ),
)

_READOUT_LEAKAGE = Hypothesis(
    "readout_leakage",
    "qubit",
    "feature induced by the readout tone",
    _qubit_signatures(
        power_exponent=_zero,
        flux_slope=_zero,
        flux_curvature=_zero,
        rabi_exponent=_zero,
        rabi_contrast=_zero,
        dispersive_shift=_zero,
    ),
)

_QUBIT_SPURIOUS = Hypothesis(
    "spurious",
    "qubit",
    "noise, interference, or a non-systematic artifact",
    _qubit_signatures(
        power_exponent=_zero,
        flux_slope=_zero,
        flux_curvature=_zero,
        rabi_exponent=_zero,
        rabi_contrast=_zero,
        dispersive_shift=_zero,
    ),
)

_QUBIT_NOVEL = Hypothesis(
    "novel",
    "qubit",
    "matches no declared qubit signature",
)


def _resonator_signatures(
    slope: Callable[[Mapping[str, Any]], float],
    curvature: Callable[[Mapping[str, Any]], float],
) -> Tuple[Signature, ...]:
    return (
        Signature(
            "flux_nudge",
            "flux_slope_mhz_per_z",
            slope,
            _tolerance("flux_slope_tolerance_mhz_per_z"),
        ),
        Signature(
            "flux_nudge",
            "flux_curvature_mhz_per_z2",
            curvature,
            _tolerance("flux_curvature_tolerance_mhz_per_z2"),
        ),
    )


_RESONATOR_HYPOTHESES: Tuple[Hypothesis, ...] = (
    Hypothesis(
        "readout_resonator",
        "resonator",
        "wanted readout resonator",
        _resonator_signatures(
            _prediction("resonator_flux_slope_mhz_per_z", 0.0),
            _prediction("resonator_flux_curvature_mhz_per_z2", 0.0),
        ),
    ),
    Hypothesis(
        "neighbor_resonator",
        "resonator",
        "another resonator on the chip",
        _resonator_signatures(
            _prediction("neighbor_flux_slope_mhz_per_z", 0.0),
            _prediction("neighbor_flux_curvature_mhz_per_z2", 0.0),
        ),
    ),
    Hypothesis(
        "package_mode",
        "resonator",
        "flux-independent package mode",
        _resonator_signatures(_zero, _zero),
    ),
    Hypothesis(
        "spurious",
        "resonator",
        "noise, interference, or a non-systematic artifact",
        _resonator_signatures(_zero, _zero),
    ),
    Hypothesis(
        "novel",
        "resonator",
        "matches no declared resonator signature",
    ),
)

_QUBIT_HYPOTHESES: Tuple[Hypothesis, ...] = (
    _QUBIT_01,
    _F02,
    _HIGHER,
    _TLS,
    _NEIGHBOR_QUBIT,
    _READOUT_LEAKAGE,
    _QUBIT_SPURIOUS,
    _QUBIT_NOVEL,
)


def hypotheses_for(family: str) -> Tuple[Hypothesis, ...]:
    name = str(family).strip().lower()
    if name == "qubit":
        return _QUBIT_HYPOTHESES
    if name == "resonator":
        return _RESONATOR_HYPOTHESES
    raise KeyError("unknown hypothesis family " + repr(name))


def hypothesis_ids(family: str) -> Tuple[str, ...]:
    return tuple(item.hypothesis_id for item in hypotheses_for(family))


def get_hypothesis(hypothesis_id: str, family: str = "qubit") -> Hypothesis:
    wanted = str(hypothesis_id)
    for hypothesis in hypotheses_for(family):
        if hypothesis.hypothesis_id == wanted:
            return hypothesis
    raise KeyError(
        "unknown {0} hypothesis {1!r}".format(str(family), wanted)
    )
