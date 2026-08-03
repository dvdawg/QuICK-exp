"""Ordered class-A remediation requests for existing acquisition presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


LADDER: Tuple[str, ...] = (
    "averaging",
    "timing",
    "readout_power",
    "window",
    "held_flux",
)

_REASON_PREFERENCE = {
    "detectability": (
        "averaging",
        "timing",
        "readout_power",
        "window",
        "held_flux",
    ),
    "resolution": (
        "window",
        "averaging",
        "timing",
        "readout_power",
        "held_flux",
    ),
    "prior_coverage": (
        "window",
        "averaging",
        "timing",
        "readout_power",
        "held_flux",
    ),
    "edge_proximity": (
        "window",
        "averaging",
        "timing",
        "readout_power",
        "held_flux",
    ),
}


@dataclass(frozen=True)
class RemediationStep:
    step_id: str
    rationale: str
    cost_multiplier: float
    overrides: Dict[str, Any] = field(default_factory=dict)


def _build(step_id: str, current: Mapping[str, Any]) -> RemediationStep:
    if step_id == "averaging":
        current_avg = int(current.get("hard_avg", 100) or 100)
        return RemediationStep(
            step_id="averaging",
            rationale="signal is below the detectability floor; double averaging",
            cost_multiplier=2.0,
            overrides={"hard_avg": max(current_avg, 1) * 2},
        )
    if step_id == "timing":
        return RemediationStep(
            step_id="timing",
            rationale="short relaxation or pulse lengths can suppress contrast",
            cost_multiplier=1.5,
            overrides={
                "r_relax": float(current.get("r_relax", 10.0) or 10.0) * 2.0,
                "q_length": float(current.get("q_length", 2.0) or 2.0) * 2.0,
                "r_length": float(current.get("r_length", 2.0) or 2.0) * 2.0,
            },
        )
    if step_id == "readout_power":
        current_power = float(current.get("r_power", -35.0))
        return RemediationStep(
            step_id="readout_power",
            rationale="readout power may be dephasing or under-driving the feature",
            cost_multiplier=3.0,
            overrides={
                "r_power_bracket": (
                    current_power - 4.0,
                    current_power,
                    current_power + 4.0,
                )
            },
        )
    if step_id == "window":
        return RemediationStep(
            step_id="window",
            rationale="scan coverage, resolution, or edge clearance is insufficient",
            cost_multiplier=1.0,
            overrides={"widen_factor": 2.0, "recenter_on_prior": True},
        )
    return RemediationStep(
        step_id="held_flux",
        rationale="feature may be cleaner at another Z; use run_flux_sweep",
        cost_multiplier=4.0,
        overrides={"probe_z_offsets": (-0.05, 0.05)},
    )


def next_remediation(
    assessment: Any,
    attempted: Sequence[str],
    current_overrides: Mapping[str, Any],
) -> Optional[RemediationStep]:
    """Return the next unattempted remediation request, or ``None``."""
    if getattr(assessment, "sufficient", False):
        return None
    done = {str(item) for item in attempted}
    reasons = tuple(getattr(assessment, "reasons", ()))
    order = _REASON_PREFERENCE.get(reasons[0] if reasons else "", LADDER)
    for step_id in order:
        if step_id not in done:
            return _build(step_id, current_overrides)
    return None
