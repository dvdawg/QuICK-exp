"""Domain-checked model dispatch for flux-dependent calibration records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np

from .errors import ConfigError
from .resonator_flux import (
    LOOKUP_MODEL_NAME,
    MODEL_NAME,
    extract_feature_centers,
    frequency_from_calibration_record,
    sampled_frequency,
)


def _evaluate_cosine(record: Mapping[str, Any], z_gain: Any):
    return frequency_from_calibration_record(record, z_gain)


def _evaluate_sampled_lookup(record: Mapping[str, Any], z_gain: Any):
    """Interpolate any ``z_gain``/``*_freq_mhz`` sampled lookup record."""
    value = record.get("value")
    if not isinstance(value, Mapping):
        raise ConfigError("sampled flux lookup record has no value mapping")
    frequency_keys = sorted(
        key for key in value if str(key).endswith("_freq_mhz")
    )
    if len(frequency_keys) != 1:
        raise ConfigError(
            "sampled flux lookup value must carry exactly one "
            "'*_freq_mhz' array"
        )
    return sampled_frequency(
        z_gain,
        sampled_z_gain=value["z_gain"],
        sampled_frequencies_mhz=value[frequency_keys[0]],
    )


EVALUATORS = {
    MODEL_NAME: _evaluate_cosine,
    LOOKUP_MODEL_NAME: _evaluate_sampled_lookup,
}


@dataclass(frozen=True)
class NotchCentersQC:
    centers_mhz: np.ndarray
    depths: np.ndarray
    smoothed: np.ndarray
    included: np.ndarray
    edge_pinned: np.ndarray
    low_depth: np.ndarray
    discontinuity_rows: np.ndarray
    feature_polarity: str = "unknown"

    @property
    def quality(self) -> dict:
        return {
            "feature_polarity": self.feature_polarity,
            "edge_pinned_rows": np.flatnonzero(self.edge_pinned).tolist(),
            "low_depth_rows": np.flatnonzero(self.low_depth).tolist(),
            "discontinuity_rows": self.discontinuity_rows.tolist(),
        }

    def passes(self, *, allow_branch_jumps: bool = False) -> bool:
        return bool(
            np.count_nonzero(self.included) >= 6
            and not np.any(self.edge_pinned & self.included)
            and (
                allow_branch_jumps
                or self.discontinuity_rows.size == 0
            )
        )


def extract_notch_centers_qc(
    frequencies_mhz: Any,
    amplitude: Any,
    smooth_sigma_bins: float = 2.0,
    feature_polarity: str = "auto",
) -> NotchCentersQC:
    """Extract an automatically polarized resonance ridge with row QC."""
    frequency = np.asarray(frequencies_mhz, dtype=float)
    values = np.asarray(amplitude, dtype=float)
    centers, depths, smoothed, selected_polarity = extract_feature_centers(
        frequency,
        values,
        smooth_sigma_bins,
        feature_polarity=feature_polarity,
    )
    selected_indices = np.argmin(
        np.abs(frequency[None, :] - centers[:, None]),
        axis=1,
    )
    edge_pinned = (selected_indices == 0) | (
        selected_indices == frequency.size - 1
    )
    row_noise = np.median(
        np.abs(np.diff(values, axis=1) - np.median(np.diff(values, axis=1), axis=1)[:, None]),
        axis=1,
    ) / (0.6744897501960817 * np.sqrt(2.0))
    low_depth = depths < 3.0 * np.maximum(row_noise, np.finfo(float).eps)
    included = ~(edge_pinned | low_depth)
    included_indices = np.flatnonzero(included)
    discontinuities = []
    if included_indices.size >= 3:
        changes = np.abs(np.diff(centers[included_indices]))
        positive = changes[changes > 0]
        typical = float(np.median(positive)) if positive.size else 0.0
        if typical > 0:
            for offset in np.flatnonzero(changes > 5.0 * typical):
                discontinuities.append(int(included_indices[offset + 1]))
    return NotchCentersQC(
        centers_mhz=centers,
        depths=depths,
        smoothed=smoothed,
        feature_polarity=selected_polarity,
        included=included,
        edge_pinned=edge_pinned,
        low_depth=low_depth,
        discontinuity_rows=np.asarray(discontinuities, dtype=int),
    )


def register_model(name: str, evaluator) -> None:
    normalized = str(name).strip()
    if not normalized:
        raise ConfigError("flux lookup model name cannot be empty")
    if not callable(evaluator):
        raise ConfigError("flux lookup evaluator must be callable")
    existing = EVALUATORS.get(normalized)
    if existing is not None and existing is not evaluator:
        raise ConfigError(f"flux lookup model {normalized!r} is already registered")
    EVALUATORS[normalized] = evaluator


def _check_domain(record: Mapping[str, Any], z_gain: Any) -> None:
    domain = record.get("valid_domain")
    domain = domain.get("z_gain") if isinstance(domain, Mapping) else None
    if domain is None:
        return
    try:
        minimum, maximum = map(float, domain)
    except (TypeError, ValueError) as error:
        raise ConfigError("flux lookup valid_domain.z_gain must be [min, max]") from error
    values = np.asarray(z_gain, dtype=float)
    if np.any(values < minimum) or np.any(values > maximum):
        raise ConfigError(
            f"requested z_gain is outside accepted flux-lookup domain [{minimum}, {maximum}]"
        )


def frequency_from_record(
    record: Mapping[str, Any],
    z_gain: Any,
) -> Union[float, np.ndarray]:
    """Evaluate one accepted lookup through its registered model."""
    if not isinstance(record, Mapping) or record.get("status", "accepted") != "accepted":
        raise ConfigError("flux lookup record is not accepted")
    model = record.get("model")
    evaluator = EVALUATORS.get(str(model))
    if evaluator is None:
        raise ConfigError(f"no evaluator is registered for flux lookup model {model!r}")
    _check_domain(record, z_gain)
    return evaluator(record, z_gain)
