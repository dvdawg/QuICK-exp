"""Margin-based adjudication across candidate and physical hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .taxonomy import get_hypothesis


NOVEL_SCORE_FLOOR = -4.5


@dataclass(frozen=True)
class ScoreRow:
    candidate_id: str
    hypothesis_id: str
    total_score: float
    evidence_count: int
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Scorecard:
    rows: Tuple[ScoreRow, ...]
    leader: ScoreRow
    runner_up: Optional[ScoreRow]
    margin: float

    def as_dict(self) -> dict:
        return {
            "leader": self.leader.__dict__,
            "runner_up": (
                self.runner_up.__dict__ if self.runner_up is not None else None
            ),
            "margin": float(self.margin),
            "rows": [row.__dict__ for row in self.rows],
        }


@dataclass(frozen=True)
class Adjudication:
    action: str
    reason: str
    failure_class: Optional[str]
    candidate_id: Optional[str]
    hypothesis_id: Optional[str]
    margin: float


def build_scorecard(
    candidates: Sequence,
    hypothesis_ids: Sequence[str],
    responses: Mapping[str, Mapping[str, Mapping[str, float]]],
    device_context: Mapping[str, Any],
    *,
    family: str = "qubit",
    novel_score_floor: float = NOVEL_SCORE_FLOOR,
) -> Scorecard:
    """Score every candidate/hypothesis pair using completed probes only."""
    rows = []
    hypothesis_order = {
        str(hypothesis_id): index
        for index, hypothesis_id in enumerate(hypothesis_ids)
    }
    for candidate in candidates:
        candidate_responses = responses.get(candidate.candidate_id, {})
        for hypothesis_id in hypothesis_ids:
            hypothesis = get_hypothesis(hypothesis_id, family=family)
            if hypothesis.hypothesis_id == "novel":
                rows.append(
                    ScoreRow(
                        candidate.candidate_id,
                        hypothesis.hypothesis_id,
                        float(novel_score_floor),
                        0,
                        {},
                    )
                )
                continue
            if getattr(candidate, "is_null", False):
                score = (
                    float(device_context.get("null_candidate_score", -2.0))
                    if hypothesis.hypothesis_id == "spurious"
                    else float("-inf")
                )
                rows.append(
                    ScoreRow(
                        candidate.candidate_id,
                        hypothesis.hypothesis_id,
                        score,
                        0,
                        {},
                    )
                )
                continue

            context = dict(device_context)
            context["candidate"] = candidate
            context["responses"] = candidate_responses
            total = 0.0
            evidence_count = 0
            components = {}
            for signature in hypothesis.signatures:
                observed_by_probe = candidate_responses.get(signature.probe_id)
                if not isinstance(observed_by_probe, Mapping):
                    continue
                if signature.observable not in observed_by_probe:
                    continue
                observed = float(observed_by_probe[signature.observable])
                predicted = float(signature.predicted(context))
                tolerance = abs(float(signature.tolerance(context)))
                if not np.all(np.isfinite([observed, predicted, tolerance])):
                    continue
                if tolerance <= np.finfo(float).eps:
                    raise ValueError("scorecard signature tolerance must be positive")
                component = (
                    -0.5
                    * ((observed - predicted) / tolerance) ** 2
                    * float(signature.weight)
                )
                key = signature.probe_id + ":" + signature.observable
                components[key] = float(component)
                total += float(component)
                evidence_count += 1
            rows.append(
                ScoreRow(
                    candidate.candidate_id,
                    hypothesis.hypothesis_id,
                    float(total),
                    evidence_count,
                    components,
                )
            )

    if not rows:
        raise ValueError("scorecard requires at least one candidate/hypothesis pair")
    candidate_rank = {
        candidate.candidate_id: int(candidate.rank) for candidate in candidates
    }
    rows.sort(
        key=lambda row: (
            -row.total_score,
            candidate_rank.get(row.candidate_id, 10**9),
            hypothesis_order.get(row.hypothesis_id, 10**9),
        )
    )
    leader = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    margin = (
        float(leader.total_score - runner_up.total_score)
        if runner_up is not None
        else float("inf")
    )
    return Scorecard(tuple(rows), leader, runner_up, margin)


def adjudicate(
    scorecard: Scorecard,
    coverage: Any,
    *,
    wanted: str,
    margin_threshold: float,
    probes_remaining: bool,
    consistency_passes: bool = True,
) -> Adjudication:
    """Apply the design's ordered, margin-only verdict rules."""
    if not getattr(coverage, "sufficient", False):
        return Adjudication(
            "remediate",
            "measurement coverage was insufficient",
            "A",
            None,
            None,
            float(scorecard.margin),
        )
    leader = scorecard.leader
    if leader.hypothesis_id == "novel":
        return Adjudication(
            "consult",
            "no declared signature explains the observed response",
            "B",
            leader.candidate_id,
            leader.hypothesis_id,
            float(scorecard.margin),
        )
    if float(scorecard.margin) < float(margin_threshold):
        return Adjudication(
            "probe" if probes_remaining else "consult",
            "hypothesis margin is unresolved",
            "B",
            leader.candidate_id,
            leader.hypothesis_id,
            float(scorecard.margin),
        )
    if leader.hypothesis_id == str(wanted) and consistency_passes:
        return Adjudication(
            "accept",
            "wanted hypothesis wins by the configured margin",
            None,
            leader.candidate_id,
            leader.hypothesis_id,
            float(scorecard.margin),
        )
    if leader.hypothesis_id == str(wanted):
        return Adjudication(
            "consult",
            "wanted hypothesis conflicts with consistency predictions",
            "C",
            leader.candidate_id,
            leader.hypothesis_id,
            float(scorecard.margin),
        )
    if leader.hypothesis_id == "spurious" and leader.candidate_id:
        return Adjudication(
            "backtrack",
            "leading candidate is spurious",
            "B",
            leader.candidate_id,
            leader.hypothesis_id,
            float(scorecard.margin),
        )
    return Adjudication(
        "derive_and_retry",
        "a declared non-target hypothesis wins",
        "B",
        leader.candidate_id,
        leader.hypothesis_id,
        float(scorecard.margin),
    )
