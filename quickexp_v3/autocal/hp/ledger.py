"""Persistent hypothesis backtracking and circuit-QED discrepancy ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...util import to_builtin


class BacktrackLimitExceeded(RuntimeError):
    """Backtracking cannot continue without violating a hardware-owned cap."""


@dataclass(frozen=True)
class BacktrackDecision:
    action: str
    address: str
    demoted_candidate_id: Optional[str]
    promoted_candidate_id: Optional[str]
    evidence: Mapping[str, Any]


class HypothesisLedger:
    def __init__(
        self,
        max_session_backtracks: int = 3,
        max_address_backtracks: int = 2,
    ):
        self.max_session_backtracks = max(int(max_session_backtracks), 0)
        self.max_address_backtracks = max(int(max_address_backtracks), 0)
        if self.max_address_backtracks > self.max_session_backtracks:
            raise ValueError("per-address backtrack cap cannot exceed session cap")
        self.total_backtracks = 0
        self._addresses: Dict[str, Dict[str, Any]] = {}

    def _entry(self, address: str) -> Dict[str, Any]:
        return self._addresses.setdefault(
            str(address),
            {
                "ranking": [],
                "evidence": [],
                "demoted": [],
                "backtracks": 0,
                "joint_retune_attempted": False,
                "doubts": [],
            },
        )

    def record(
        self,
        address: str,
        ranking: Sequence[Mapping[str, Any]],
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        entry = self._entry(address)
        demoted = set(entry["demoted"])
        normalized = [
            to_builtin(dict(row))
            for row in ranking
            if str(row.get("candidate_id", "")) not in demoted
        ]
        if not normalized:
            raise BacktrackLimitExceeded(
                "no non-demoted hypothesis candidate remains for " + str(address)
            )
        entry["ranking"] = normalized
        if evidence is not None:
            entry["evidence"].append(to_builtin(dict(evidence)))

    def leader(self, address: str) -> Optional[Mapping[str, Any]]:
        ranking = self._entry(address)["ranking"]
        return dict(ranking[0]) if ranking else None

    def demoted(self, address: str) -> Tuple[str, ...]:
        return tuple(str(item) for item in self._entry(address)["demoted"])

    def upstream_doubt(
        self,
        address: str,
        evidence: Mapping[str, Any],
        *,
        joint_retune_available: bool = False,
    ) -> BacktrackDecision:
        entry = self._entry(address)
        normalized_evidence = to_builtin(dict(evidence))
        entry["doubts"].append(normalized_evidence)
        leader = self.leader(address)
        current = None if leader is None else str(leader.get("candidate_id"))
        if joint_retune_available and not entry["joint_retune_attempted"]:
            entry["joint_retune_attempted"] = True
            return BacktrackDecision(
                "retune_joint_operating_point",
                str(address),
                None,
                current,
                normalized_evidence,
            )
        if self.total_backtracks >= self.max_session_backtracks:
            raise BacktrackLimitExceeded("session backtrack cap is exhausted")
        if int(entry["backtracks"]) >= self.max_address_backtracks:
            raise BacktrackLimitExceeded(
                "backtrack cap is exhausted for " + str(address)
            )
        if leader is None:
            raise BacktrackLimitExceeded(
                "no hypothesis candidate is recorded for " + str(address)
            )
        remaining = [
            row
            for row in entry["ranking"]
            if str(row.get("candidate_id")) != current
        ]
        if not remaining:
            raise BacktrackLimitExceeded(
                "no alternate hypothesis candidate remains for " + str(address)
            )
        entry["demoted"].append(current)
        entry["ranking"] = remaining
        entry["backtracks"] = int(entry["backtracks"]) + 1
        self.total_backtracks += 1
        promoted = str(entry["ranking"][0].get("candidate_id"))
        return BacktrackDecision(
            "backtrack",
            str(address),
            current,
            promoted,
            normalized_evidence,
        )

    def as_dict(self) -> dict:
        return {
            "max_session_backtracks": int(self.max_session_backtracks),
            "max_address_backtracks": int(self.max_address_backtracks),
            "total_backtracks": int(self.total_backtracks),
            "addresses": to_builtin(self._addresses),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HypothesisLedger":
        ledger = cls(
            int(raw.get("max_session_backtracks", 3)),
            int(raw.get("max_address_backtracks", 2)),
        )
        ledger.total_backtracks = int(raw.get("total_backtracks", 0))
        addresses = raw.get("addresses", {})
        if isinstance(addresses, Mapping):
            ledger._addresses = {
                str(address): dict(to_builtin(entry))
                for address, entry in addresses.items()
                if isinstance(entry, Mapping)
            }
        return ledger


@dataclass(frozen=True)
class DiscrepancyEntry:
    prediction_id: str
    predicted: Optional[float]
    measured: Optional[float]
    residual: Optional[float]
    sigma: Optional[float]
    model_assumptions: Tuple[str, ...]
    sources: Tuple[str, ...]
    verdict: str

    def as_dict(self) -> dict:
        return to_builtin(self.__dict__)


class DiscrepancyLedger:
    def __init__(self, entries: Sequence[DiscrepancyEntry] = ()):
        self.entries = list(entries)

    def record(
        self,
        prediction_id: str,
        predicted: Optional[float],
        measured: Optional[float],
        sigma: Optional[float],
        model_assumptions: Sequence[str],
        sources: Sequence[str] = (),
        *,
        allowed_sigma: float = 3.0,
    ) -> DiscrepancyEntry:
        finite = bool(
            predicted is not None
            and measured is not None
            and sigma is not None
            and np.all(
                np.isfinite(
                    [float(predicted), float(measured), float(sigma)]
                )
            )
            and abs(float(sigma)) > np.finfo(float).eps
        )
        if finite:
            residual = (float(measured) - float(predicted)) / abs(float(sigma))
            verdict = (
                "consistent"
                if abs(residual) <= abs(float(allowed_sigma))
                else "deviant"
            )
        else:
            residual = None
            verdict = "untestable"
        entry = DiscrepancyEntry(
            str(prediction_id),
            None if predicted is None else float(predicted),
            None if measured is None else float(measured),
            None if residual is None else float(residual),
            None if sigma is None else float(sigma),
            tuple(str(item) for item in model_assumptions),
            tuple(str(item) for item in sources),
            verdict,
        )
        self.entries = [
            existing
            for existing in self.entries
            if existing.prediction_id != entry.prediction_id
        ]
        self.entries.append(entry)
        return entry

    def record_upper_bound(
        self,
        prediction_id: str,
        upper_bound: Optional[float],
        measured: Optional[float],
        sigma: Optional[float],
        model_assumptions: Sequence[str],
        sources: Sequence[str] = (),
        *,
        allowed_sigma: float = 3.0,
    ) -> DiscrepancyEntry:
        """Record a one-sided prediction such as T2 <= 2*T1."""
        finite = bool(
            upper_bound is not None
            and measured is not None
            and sigma is not None
            and np.all(
                np.isfinite(
                    [float(upper_bound), float(measured), float(sigma)]
                )
            )
            and abs(float(sigma)) > np.finfo(float).eps
        )
        if finite:
            residual = max(
                (float(measured) - float(upper_bound)) / abs(float(sigma)),
                0.0,
            )
            verdict = (
                "consistent"
                if residual <= abs(float(allowed_sigma))
                else "deviant"
            )
        else:
            residual = None
            verdict = "untestable"
        entry = DiscrepancyEntry(
            str(prediction_id),
            None if upper_bound is None else float(upper_bound),
            None if measured is None else float(measured),
            None if residual is None else float(residual),
            None if sigma is None else float(sigma),
            tuple(str(item) for item in model_assumptions),
            tuple(str(item) for item in sources),
            verdict,
        )
        self.entries = [
            existing
            for existing in self.entries
            if existing.prediction_id != entry.prediction_id
        ]
        self.entries.append(entry)
        return entry

    def prediction_ids(self) -> Tuple[str, ...]:
        return tuple(entry.prediction_id for entry in self.entries)

    def as_dict(self) -> dict:
        return {"entries": [entry.as_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DiscrepancyLedger":
        entries = []
        for item in raw.get("entries", ()):
            if not isinstance(item, Mapping):
                continue
            entries.append(
                DiscrepancyEntry(
                    str(item.get("prediction_id", "")),
                    item.get("predicted"),
                    item.get("measured"),
                    item.get("residual"),
                    item.get("sigma"),
                    tuple(item.get("model_assumptions", ())),
                    tuple(item.get("sources", ())),
                    str(item.get("verdict", "untestable")),
                )
            )
        return cls(entries)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "{0:.6g}".format(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_discrepancy_report(ledger: DiscrepancyLedger) -> str:
    lines = [
        "# Circuit-QED discrepancy report",
        "",
        "| Prediction | Predicted | Measured | Residual (σ) | Verdict | Sources |",
        "|---|---:|---:|---:|---|---|",
    ]
    for entry in ledger.entries:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                _cell(entry.prediction_id),
                _cell(entry.predicted),
                _cell(entry.measured),
                _cell(entry.residual),
                _cell(entry.verdict),
                _cell(", ".join(entry.sources)),
            )
        )
        if entry.model_assumptions:
            lines.append(
                "| ↳ assumptions |  |  |  | {0} |  |".format(
                    _cell(", ".join(entry.model_assumptions))
                )
            )
    if not ledger.entries:
        lines.append("| — | — | — | — | untestable | — |")
    return "\n".join(lines) + "\n"
