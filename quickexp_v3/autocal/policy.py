"""Load and apply the hardware-owned automated-calibration policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ..errors import ConfigError


_MANDATORY_HARD_STOP_RECORDS = frozenset(
    {
        "defaults.r_offset",
        "lookups.resonator_vs_flux",
        "lookups.qubit_vs_flux",
    }
)


@dataclass(frozen=True)
class PolicyDecision:
    promote: bool
    reason: str


@dataclass(frozen=True)
class AutocalPolicy:
    hard_stop_records: frozenset
    auto_accept: Mapping[str, Mapping[str, Any]]
    max_wall_clock_hours: float
    max_node_attempts: int
    max_total_runs: int
    q_gain_max: float
    r_power_max_db: float
    require_ramsey_confirmation: bool
    auto_promotion_configured: bool
    hypothesis_nodes: frozenset
    adaptive_nodes: frozenset
    margin_threshold: float
    probe_budget_seconds: float
    top_k_candidates: int
    candidate_prominence_ratio: float
    max_backtracks_per_session: int
    max_backtracks_per_address: int
    adaptive_initial_rows: int
    adaptive_max_rows: int
    adaptive_abort_after_rows: int
    advisor_mode: str
    advisor_model: str
    advisor_timeout_seconds: float

    def hypothesis_enabled(self, node_id: str) -> bool:
        return str(node_id) in self.hypothesis_nodes

    def adaptive_enabled(self, node_id: str) -> bool:
        return str(node_id) in self.adaptive_nodes

    def clamp_overrides(self, overrides: Mapping[str, Any]) -> dict:
        result = dict(overrides)
        if "q_gain" in result:
            gain = np.asarray(result["q_gain"], dtype=float)
            gain = np.clip(gain, -self.q_gain_max, self.q_gain_max)
            result["q_gain"] = float(gain) if gain.ndim == 0 else gain
        if "r_power" in result:
            power = np.asarray(result["r_power"], dtype=float)
            power = np.minimum(power, self.r_power_max_db)
            result["r_power"] = float(power) if power.ndim == 0 else power
        return result

    @staticmethod
    def _domain_contains(
        proposal: Mapping[str, Any],
        working_z_gain: Optional[float],
    ) -> bool:
        if working_z_gain is None:
            return True
        domain = proposal.get("valid_domain")
        z_domain = domain.get("z_gain") if isinstance(domain, Mapping) else None
        if z_domain is None:
            return True
        try:
            minimum, maximum = map(float, z_domain)
            working = float(working_z_gain)
        except (TypeError, ValueError):
            return False
        if (
            not np.all(np.isfinite([minimum, maximum, working]))
            or minimum > maximum
        ):
            return False
        return minimum <= working <= maximum

    @staticmethod
    def _tolerance_passes(
        tolerance: Mapping[str, Any],
        current_value: Any,
        new_value: Any,
    ) -> bool:
        kind, amount = next(iter(tolerance.items()))
        if kind == "always":
            return bool(amount)
        try:
            current = float(current_value)
            new = float(new_value)
            limit = float(amount)
        except (TypeError, ValueError):
            return False
        delta = abs(new - current)
        if kind == "relative":
            return delta <= limit * max(abs(current), np.finfo(float).eps)
        return delta <= limit

    def promotion_decision(
        self,
        *,
        autonomy_level: int,
        proposal: Mapping[str, Any],
        current_value: Any,
        gates_pass: bool,
        working_z_gain: Optional[float] = None,
        ramsey_sign_confirmed: bool = False,
    ) -> PolicyDecision:
        level = int(autonomy_level)
        if level not in (0, 1, 2):
            raise ConfigError("autonomy level must be 0, 1, or 2")
        if level > 0 and not self.auto_promotion_configured:
            return PolicyDecision(
                False,
                "hardware.autocal is absent; promotion is proposal-only",
            )
        address = str(proposal.get("record", ""))
        if not gates_pass:
            return PolicyDecision(False, "fit gates did not all pass")
        if address in self.hard_stop_records:
            return PolicyDecision(False, f"{address} is a hard-stop record")
        if not self._domain_contains(proposal, working_z_gain):
            return PolicyDecision(
                False,
                "proposal valid domain excludes the working point",
            )
        provenance = proposal.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        node = str(provenance.get("autocal_node", ""))
        analysis = str(provenance.get("analysis", "")).lower()
        if address == "defaults.r_freq" and node == "N10r":
            return PolicyDecision(
                False,
                "readout-optimization frequency is always human-reviewed",
            )
        if (
            address == "defaults.q_freq"
            and self.require_ramsey_confirmation
            and ("ramsey" in analysis or node == "N12")
            and not ramsey_sign_confirmed
        ):
            return PolicyDecision(False, "Ramsey detuning sign is unconfirmed")
        if level == 0:
            return PolicyDecision(False, "L0 is proposal-only")
        tolerance = self.auto_accept.get(address)
        if tolerance is None:
            # A policy entry for a structured derived quantity applies to its
            # cycle-indexed leaves (for example derived.t2_echo.cycle_0).
            ancestors = [
                (configured, candidate)
                for configured, candidate in self.auto_accept.items()
                if address.startswith(str(configured) + ".")
            ]
            if ancestors:
                _configured, tolerance = max(
                    ancestors,
                    key=lambda item: len(str(item[0])),
                )
        if level == 1 and tolerance is None:
            return PolicyDecision(False, f"{address} is not in the L1 allowlist")
        if tolerance is not None and not self._tolerance_passes(
            tolerance,
            current_value,
            proposal.get("value"),
        ):
            return PolicyDecision(False, "change exceeds the policy tolerance")
        return PolicyDecision(True, f"L{level} policy permits promotion")


def load_autocal_policy(hardware: Mapping[str, Any]) -> AutocalPolicy:
    """Load policy defaults; a missing root is safely L0-only."""
    configured = "autocal" in hardware and hardware.get("autocal") is not None
    raw = hardware.get("autocal", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError("hardware.autocal must be a mapping")
    budgets = raw.get("budgets", {})
    caps = raw.get("caps", {})
    ramsey = raw.get("ramsey_sign", {})
    hypothesis = raw.get("hypothesis", {})
    backtracking = raw.get("backtracking", {})
    adaptive = raw.get("adaptive", {})
    advisor = raw.get("advisor", {})
    return AutocalPolicy(
        hard_stop_records=(
            _MANDATORY_HARD_STOP_RECORDS
            | frozenset(raw.get("hard_stop_records", ()))
        ),
        auto_accept=dict(raw.get("auto_accept", {})),
        max_wall_clock_hours=float(budgets.get("max_wall_clock_hours", 8.0)),
        max_node_attempts=int(budgets.get("max_node_attempts", 3)),
        max_total_runs=int(budgets.get("max_total_runs", 200)),
        q_gain_max=float(caps.get("q_gain_max", 0.8)),
        r_power_max_db=float(caps.get("r_power_max_db", -20.0)),
        require_ramsey_confirmation=bool(
            ramsey.get("require_two_point_confirmation", True)
        ),
        auto_promotion_configured=bool(configured),
        hypothesis_nodes=frozenset(raw.get("hypothesis_nodes", ())),
        adaptive_nodes=frozenset(raw.get("adaptive_nodes", ())),
        margin_threshold=float(hypothesis.get("margin_threshold", 2.0)),
        probe_budget_seconds=float(
            hypothesis.get("probe_budget_seconds", 600.0)
        ),
        top_k_candidates=int(hypothesis.get("top_k_candidates", 3)),
        candidate_prominence_ratio=float(
            hypothesis.get("candidate_prominence_ratio", 0.5)
        ),
        max_backtracks_per_session=int(
            backtracking.get("max_backtracks_per_session", 3)
        ),
        max_backtracks_per_address=int(
            backtracking.get("max_backtracks_per_address", 2)
        ),
        adaptive_initial_rows=int(adaptive.get("initial_rows", 5)),
        adaptive_max_rows=int(adaptive.get("max_rows", 7)),
        adaptive_abort_after_rows=int(
            adaptive.get("abort_after_rows", 5)
        ),
        advisor_mode=str(advisor.get("mode", "null")),
        advisor_model=str(advisor.get("model", "claude-sonnet-5")),
        advisor_timeout_seconds=float(advisor.get("timeout_seconds", 60.0)),
    )
