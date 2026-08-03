"""Resumable dependency scheduler for policy-governed calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..backend import SyntheticBackend
from ..errors import AcquisitionError, ConfigError
from ..ide import load_repository
from ..synthetic_device import DeviceModel
from ..util import to_builtin
from .budget import BudgetExceeded, BudgetModel, BudgetTracker
from .graph import (
    NODE_REGISTRY,
    change_exceeds_invalidation_threshold,
    invalidated_nodes,
    target_nodes,
)
from .nodes import (
    NodeOutcome,
    SessionContext,
    StopRequested,
    consult_advisor,
    run_node,
)
from .policy import load_autocal_policy
from .replay import verify_session_replay
from .session import AutocalSession, replay_decisions


@dataclass(frozen=True)
class AutocalSummary:
    session_id: str
    session_directory: Path
    target: str
    status: str
    node_status: Mapping[str, str]
    proposals: tuple
    total_runs: int
    spent_seconds: float

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_directory": str(self.session_directory),
            "target": self.target,
            "status": self.status,
            "node_status": dict(self.node_status),
            "proposals": list(self.proposals),
            "total_runs": int(self.total_runs),
            "spent_seconds": float(self.spent_seconds),
        }


@dataclass(frozen=True)
class ReplaySummary:
    session_directory: Path
    events: tuple
    verified_fits: tuple

    @property
    def status(self) -> str:
        return "replayed"


def _default_device() -> DeviceModel:
    """Use the accepted real-data cosine as the simulation's nominal device."""
    return DeviceModel(
        resonator_base_mhz=6884.186011,
        resonator_flux_amplitude_mhz=0.620565,
        resonator_flux_period_z=0.184257,
        resonator_flux_peak_z=-0.072305,
        resonator_linewidth_mhz=0.45,
        punchout_transition_power_db=-30.0,
        punchout_width_db=3.0,
        qubit_max_frequency_mhz=5606.5,
        qubit_power_broadening_mhz_per_gain=4.0,
        t1_us=6.2,
        t2_ramsey_us=1.8,
        t2_echo_us=5.0,
    )


def _summary(session: AutocalSession, budget: BudgetTracker) -> AutocalSummary:
    node_status = {
        node_id: str(details.get("status", "pending"))
        for node_id, details in session.state.get("nodes", {}).items()
    }
    proposal_ids = []
    for event in session.events():
        if event.get("event") == "proposal_written":
            proposal_ids.append(str(event.get("proposal_id")))
    return AutocalSummary(
        session_id=session.session_id,
        session_directory=session.directory,
        target=str(session.state["target"]),
        status=str(session.state["status"]),
        node_status=node_status,
        proposals=tuple(dict.fromkeys(proposal_ids)),
        total_runs=int(budget.total_runs),
        spent_seconds=float(budget.spent_seconds),
    )


def _selected_dependency_blocked(
    node_id: str,
    dependencies: tuple,
    selected: set,
    states: Mapping[str, Any],
) -> Optional[str]:
    for dependency in dependencies:
        if dependency not in selected:
            continue
        status = states.get(dependency, {}).get("status")
        if status == "blocked":
            return dependency
    return None


def _recover_failed_rabi_identity(
    context: SessionContext,
    outcome: NodeOutcome,
) -> bool:
    """Try joint retuning, then the next N5 ledger candidate, within caps."""
    from .hp.ledger import BacktrackLimitExceeded, HypothesisLedger

    raw = context.session.state.get("hypothesis_ledger")
    if not isinstance(raw, Mapping) or not raw:
        return False
    ledger = HypothesisLedger.from_dict(raw)
    address = "defaults.q_freq"
    try:
        decision = ledger.upstream_doubt(
            address,
            {
                "node": "N8",
                "reason": outcome.reason,
                "last_csv": outcome.last_csv,
                "gates": to_builtin(outcome.gates or {}),
            },
            joint_retune_available=True,
        )
    except BacktrackLimitExceeded as error:
        context.session.event(
            "backtrack_exhausted",
            node="N8",
            decision="escalate",
            reason=str(error),
            address=address,
        )
        return False
    context.session.state["hypothesis_ledger"] = ledger.as_dict()
    if decision.action == "retune_joint_operating_point":
        current_gain = abs(float(context.working_value("defaults.q_gain", 0.4)))
        increased = min(current_gain * 1.25, context.policy.q_gain_max)
        next_gain = (
            increased
            if increased > current_gain + 1.0e-12
            else current_gain * 0.8
        )
        context.session.set_working_values({"defaults.q_gain": next_gain})
        reason = "retry Rabi after bounded q_gain joint retune"
    else:
        leader = ledger.leader(address)
        if leader is None or "center_mhz" not in leader:
            return False
        context.session.set_working_values(
            {address: float(leader["center_mhz"])}
        )
        reason = "retry Rabi with the next non-demoted N5 candidate"
    context.session.update_node(
        "N8",
        status="pending",
        attempts=0,
        reason=reason,
    )
    context.session.event(
        "upstream_doubt",
        node="N8",
        decision=decision.action,
        reason=reason,
        address=address,
        demoted_candidate_id=decision.demoted_candidate_id,
        promoted_candidate_id=decision.promoted_candidate_id,
        working_q_freq=context.working_value(address),
        working_q_gain=context.working_value("defaults.q_gain"),
        total_backtracks=ledger.total_backtracks,
    )
    return True


def _finalize_discrepancy_ledger(
    context: SessionContext,
    hardware: Mapping[str, Any],
) -> None:
    """Complete the operator report, explicitly marking unavailable tests."""
    raw = context.session.state.get("discrepancy_ledger")
    if not isinstance(raw, Mapping):
        return
    from .hp.ledger import DiscrepancyLedger, render_discrepancy_report

    ledger = DiscrepancyLedger.from_dict(raw)
    working = context.session.state.get("working_values", {})
    working = working if isinstance(working, Mapping) else {}
    expected = hardware.get("expected", {})
    expected = expected if isinstance(expected, Mapping) else {}
    q_band = expected.get("q_freq_mhz")
    q_frequency = working.get("defaults.q_freq")
    if isinstance(q_band, (list, tuple)) and len(q_band) == 2:
        low, high = sorted(float(value) for value in q_band)
        ledger.record(
            "f_q_band",
            0.5 * (low + high),
            None if q_frequency is None else float(q_frequency),
            max((high - low) / 6.0, 1.0e-9),
            ("design band represented as a three-sigma prior",),
            ("hardware.expected", "N5"),
        )

    t1 = working.get("derived.t1")
    t2_values = [
        working.get("derived.t2_ramsey"),
        working.get("derived.t2_echo.cycle_0"),
    ]
    finite_t2 = [
        float(value)
        for value in t2_values
        if value is not None and np.isfinite(float(value))
    ]
    if t1 is not None and finite_t2:
        bound = 2.0 * float(t1)
        ledger.record_upper_bound(
            "t2_bound",
            bound,
            max(finite_t2),
            max(0.05 * abs(bound), 1.0e-9),
            ("T2 cannot exceed twice T1 for a passive two-level system",),
            ("N11", "N12", "N13"),
        )

    required = {
        "f_q_band": (
            "design band and session qubit spectroscopy are both available",
            ("hardware.expected", "N5"),
        ),
        "chi": (
            "dispersive approximation and candidate-state preparation",
            ("N5",),
        ),
        "flux_period_agreement": (
            "qubit and readout share a SQUID-loop period",
            ("N3", "N5"),
        ),
        "anharmonicity": (
            "a resolved f02 two-photon shadow is present",
            ("N5",),
        ),
        "t2_bound": (
            "T1 and at least one T2 estimate are available",
            ("N11", "N12", "N13"),
        ),
        "rabi_gain_linearity": (
            "two weak-drive Rabi rates are resolved",
            ("N5",),
        ),
        "readout_fidelity_vs_snr": (
            "two Gaussian readout clouds describe the labeled shots",
            ("N9",),
        ),
    }
    present = set(ledger.prediction_ids())
    for prediction_id, (assumption, sources) in required.items():
        if prediction_id not in present:
            ledger.record(
                prediction_id,
                None,
                None,
                None,
                (assumption,),
                sources,
            )
    context.session.state["discrepancy_ledger"] = ledger.as_dict()
    context.session.save()
    report_path = context.session.directory / "discrepancy-report.md"
    report_path.write_text(
        render_discrepancy_report(ledger),
        encoding="utf-8",
    )
    context.session.event(
        "discrepancy_ledger_finalized",
        node="SESSION",
        decision="report",
        reason="all declared circuit-QED predictions are represented",
        report=str(report_path),
        prediction_count=len(ledger.entries),
        deviant=[
            entry.prediction_id
            for entry in ledger.entries
            if entry.verdict == "deviant"
        ],
        untestable=[
            entry.prediction_id
            for entry in ledger.entries
            if entry.verdict == "untestable"
        ],
    )


def _record_leaves(node: Mapping[str, Any], prefix: str = "") -> dict:
    leaves = {}
    for name, raw in node.items():
        if not isinstance(raw, Mapping):
            continue
        address = f"{prefix}.{name}" if prefix else str(name)
        if "value" in raw:
            leaves[address] = raw
        else:
            leaves.update(_record_leaves(raw, address))
    return leaves


def _record_at_revision(
    calibration: Mapping[str, Any],
    address: str,
    revision: int,
) -> Optional[Mapping[str, Any]]:
    history = calibration.get("history", ())
    if not isinstance(history, list):
        return None
    versioned = []
    unversioned = []
    for index, entry in enumerate(history):
        if (
            isinstance(entry, Mapping)
            and entry.get("record") == address
            and isinstance(entry.get("previous"), Mapping)
        ):
            previous = entry["previous"]
            accepted_revision = previous.get("accepted_revision")
            try:
                accepted_revision = int(accepted_revision)
            except (TypeError, ValueError):
                unversioned.append((index, previous))
                continue
            if accepted_revision <= int(revision):
                versioned.append(
                    (accepted_revision, index, previous)
                )
    if versioned:
        return max(versioned, key=lambda item: (item[0], item[1]))[2]
    # A single legacy record predates accepted_revision stamping and is the
    # reconstructable baseline. Multiple unversioned versions are ambiguous,
    # so the caller will invalidate conservatively.
    if len(unversioned) == 1:
        return unversioned[0][1]
    return None


def _record_owner(
    address: str,
    record: Mapping[str, Any],
) -> Optional[str]:
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    stated = str(provenance.get("autocal_node", ""))
    if (
        stated in NODE_REGISTRY
        and address in NODE_REGISTRY[stated].products
    ):
        return stated
    owners = [
        node_id
        for node_id, spec in NODE_REGISTRY.items()
        if address in spec.products
    ]
    return owners[0] if owners else None


def _recompute_staleness(
    session: AutocalSession,
    calibration: Mapping[str, Any],
    *,
    previous_revision: int,
    selected_ids: tuple,
) -> tuple:
    records = calibration.get("records", {})
    leaves = _record_leaves(records) if isinstance(records, Mapping) else {}
    changed = {
        address: record
        for address, record in leaves.items()
        if int(record.get("accepted_revision", -1)) > previous_revision
    }
    selected = set(selected_ids)
    affected = set()
    reasons = []
    for address, record in changed.items():
        owner = _record_owner(address, record)
        previous = _record_at_revision(
            calibration,
            address,
            previous_revision,
        )
        if owner is None:
            affected.update(selected)
            reasons.append(f"{address}: conservative")
            continue
        spec = NODE_REGISTRY[owner]
        if (
            spec.invalidation_products
            and address not in spec.invalidation_products
        ):
            reasons.append(f"{address}: no downstream acquisition impact")
            continue
        if previous is None:
            nodes = set(invalidated_nodes(owner)).intersection(selected)
            affected.update(nodes)
            reasons.append(
                f"{address}: new accepted record invalidates {sorted(nodes)}"
            )
            continue
        if change_exceeds_invalidation_threshold(
            owner,
            previous.get("value"),
            record.get("value"),
        ):
            nodes = set(invalidated_nodes(owner)).intersection(selected)
            affected.update(nodes)
            reasons.append(f"{address}: {owner} invalidates {sorted(nodes)}")
    if not changed:
        # Direct/manual writers may not stamp accepted_revision. A revision
        # mismatch without attributable records therefore fails safe.
        affected.update(selected)
        reasons.append("unattributed revision change: conservative")

    invalidated = []
    for node_id in selected_ids:
        if node_id not in affected:
            continue
        node = session.node(node_id)
        if node.get("status") not in {"done", "blocked"}:
            continue
        node.update(
            {
                "status": "stale",
                "attempts": 0,
                "reason": "accepted calibration changed after this result",
            }
        )
        invalidated.append(node_id)
    if invalidated:
        session.save()
    return tuple(invalidated), tuple(reasons)


def run_autocal(
    project_root: Path,
    *,
    target: str = "full_cold_start",
    autonomy_level: int = 0,
    z_gain: float = 0.0,
    session_name: Optional[str] = None,
    max_wall_clock_hours: float = 8.0,
    live_hardware: bool = False,
    replay_session: Optional[Path] = None,
    backend: Any = None,
    budget_model: Optional[BudgetModel] = None,
) -> Any:
    """Create/resume and run one calibration target, or replay its audit log."""
    root = Path(project_root).expanduser().resolve()
    if replay_session is not None:
        directory = Path(replay_session).expanduser().resolve()
        replayed_session = AutocalSession.load(directory)
        return ReplaySummary(
            directory,
            tuple(replay_decisions(directory)),
            tuple(verify_session_replay(replayed_session)),
        )
    if int(autonomy_level) not in (0, 1, 2):
        raise ConfigError("AUTONOMY_LEVEL must be 0, 1, or 2")
    if live_hardware and backend is not None:
        raise ConfigError("custom backends are only allowed in simulation mode")

    specs = target_nodes(target)
    selected_ids = tuple(spec.node_id for spec in specs)
    repository = load_repository(root)
    policy = load_autocal_policy(repository.hardware)
    user_hours = float(max_wall_clock_hours)
    if user_hours <= 0:
        raise ConfigError("MAX_WALL_CLOCK_HOURS must be positive")
    session = AutocalSession.create_or_resume(
        root,
        target=target,
        autonomy_level=int(autonomy_level),
        z_gain=float(z_gain),
        node_ids=selected_ids,
        calibration_revision=int(repository.calibration.get("revision", 0)),
        session_name=session_name,
    )
    current_revision = int(repository.calibration.get("revision", 0))
    previous_revision = int(session.state.get("calibration_revision", 0))
    if current_revision != previous_revision:
        invalidated, invalidation_reasons = _recompute_staleness(
            session,
            repository.calibration,
            previous_revision=previous_revision,
            selected_ids=selected_ids,
        )
        session.event(
            "calibration_revision_changed",
            decision="recompute_staleness",
            reason="accepted calibration changed between session turns",
            previous_revision=previous_revision,
            current_revision=current_revision,
            invalidated_nodes=list(invalidated),
            invalidation_reasons=list(invalidation_reasons),
        )
        session.state["calibration_revision"] = current_revision
        session.save()

    allowed_hours = min(user_hours, policy.max_wall_clock_hours)
    budget = BudgetTracker.from_state(
        session.state,
        max_wall_clock_hours=allowed_hours,
        max_total_runs=policy.max_total_runs,
    )
    if not live_hardware and backend is None:
        backend = SyntheticBackend(seed=17, device=_default_device())
    context = SessionContext(
        project_root=root,
        session=session,
        policy=policy,
        budget=budget,
        budget_model=budget_model or BudgetModel(),
        autonomy_level=int(autonomy_level),
        z_gain=float(z_gain),
        live_hardware=bool(live_hardware),
        backend=backend,
    )
    session.state["status"] = "running"
    session.save()
    if not session.state.get("advisor_session_start_complete", False):
        consult_advisor(
            context,
            trigger="session_start",
            node_id="SESSION",
            device_context={
                "expected": to_builtin(
                    repository.hardware.get("expected", {})
                ),
                "working_values": to_builtin(
                    session.state.get("working_values", {})
                ),
                "target": target,
                "z_gain": float(z_gain),
            },
        )
        session.state["advisor_session_start_complete"] = True
        session.save()

    selected = set(selected_ids)
    for spec in specs:
        node_state = session.node(spec.node_id)
        if node_state.get("status") == "done":
            continue
        blocked_by = _selected_dependency_blocked(
            spec.node_id,
            spec.dependencies,
            selected,
            session.state.get("nodes", {}),
        )
        if blocked_by is not None:
            reason = f"hard dependency {blocked_by} is blocked"
            session.update_node(spec.node_id, status="blocked", reason=reason)
            session.event(
                "escalated",
                node=spec.node_id,
                decision="blocked",
                reason=reason,
            )
            continue

        while int(node_state.get("attempts", 0)) < policy.max_node_attempts:
            attempt = int(node_state.get("attempts", 0)) + 1
            session.update_node(
                spec.node_id,
                status="pending",
                attempts=attempt,
                reason="",
            )
            session.event(
                "node_started",
                node=spec.node_id,
                decision="acquire",
                reason=spec.name,
                attempt=attempt,
            )
            try:
                outcome = run_node(context, spec, attempt=attempt)
            except StopRequested as error:
                session.update_node(
                    spec.node_id,
                    status="pending",
                    reason=str(error),
                )
                session.event(
                    "stopped_by_operator",
                    node=spec.node_id,
                    decision="stop",
                    reason=str(error),
                )
                session.finish("stopped")
                return _summary(session, budget)
            except BudgetExceeded as error:
                session.update_node(
                    spec.node_id,
                    status="pending",
                    reason=str(error),
                )
                session.event(
                    "budget_exceeded",
                    node=spec.node_id,
                    decision="stop",
                    reason=str(error),
                    budget=budget.as_dict(),
                )
                session.finish("budget_exceeded")
                return _summary(session, budget)
            except KeyboardInterrupt:
                session.update_node(
                    spec.node_id,
                    status="pending",
                    reason="KeyboardInterrupt",
                )
                session.event(
                    "stopped_by_operator",
                    node=spec.node_id,
                    decision="stop",
                    reason="KeyboardInterrupt",
                )
                session.finish("stopped")
                raise
            except ConfigError as error:
                outcome = NodeOutcome(
                    "blocked",
                    f"ConfigError: {error}",
                    {},
                )
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
                critical_live_failure = bool(
                    live_hardware
                    and (
                        spec.node_id == "N0"
                        or isinstance(
                            error,
                            (AcquisitionError, ConnectionError),
                        )
                    )
                    and attempt >= policy.max_node_attempts
                )
                if critical_live_failure:
                    critical_reason = (
                        f"{reason}; live acquisition/reconnect attempts are "
                        "exhausted. A lost link cannot prove that an RF-held "
                        "Z line was parked."
                    )
                    session.update_node(
                        spec.node_id,
                        status="blocked",
                        reason=critical_reason,
                    )
                    session.event(
                        "critical_abort",
                        node=spec.node_id,
                        decision="abort",
                        reason=critical_reason,
                    )
                    session.finish("critical_abort")
                    return _summary(session, budget)
                if attempt < policy.max_node_attempts:
                    session.event(
                        "retake",
                        node=spec.node_id,
                        decision="retry",
                        reason=reason,
                        attempt=attempt,
                    )
                    node_state = session.node(spec.node_id)
                    continue
                outcome = NodeOutcome(
                    "blocked",
                    reason,
                    {},
                )

            if outcome.status == "retake":
                session.update_node(
                    spec.node_id,
                    status="pending",
                    last_csv=outcome.last_csv,
                    last_values=outcome.values,
                    reason=outcome.reason,
                    classification=outcome.classification,
                )
                if attempt < policy.max_node_attempts:
                    session.event(
                        "retake",
                        node=spec.node_id,
                        decision="retry",
                        reason=outcome.reason,
                        attempt=attempt,
                        gates=to_builtin(outcome.gates or {}),
                        classification=to_builtin(outcome.classification),
                    )
                    node_state = session.node(spec.node_id)
                    continue
                outcome = NodeOutcome(
                    "blocked",
                    f"{outcome.reason}; attempt cap reached",
                    outcome.values,
                    outcome.proposals,
                    outcome.last_csv,
                    outcome.gates,
                    outcome.classification,
                )

            if (
                spec.node_id == "N8"
                and outcome.status == "blocked"
                and attempt >= policy.max_node_attempts
                and policy.hypothesis_enabled("N5")
                and _recover_failed_rabi_identity(context, outcome)
            ):
                node_state = session.node(spec.node_id)
                continue

            final_status = "done" if outcome.status in {"done", "skipped"} else "blocked"
            session.update_node(
                spec.node_id,
                status=final_status,
                last_csv=outcome.last_csv,
                last_values=outcome.values,
                reason=outcome.reason,
                proposals=list(outcome.proposals),
                classification=outcome.classification,
            )
            if final_status == "blocked":
                session.event(
                    "escalated",
                    node=spec.node_id,
                    decision="blocked",
                    reason=outcome.reason,
                    gates=to_builtin(outcome.gates or {}),
                    classification=to_builtin(outcome.classification),
                )
            else:
                session.event(
                    "node_completed",
                    node=spec.node_id,
                    decision="done",
                    reason=outcome.reason,
                    proposals=list(outcome.proposals),
                    classification=to_builtin(outcome.classification),
                )
            break

    _finalize_discrepancy_ledger(context, repository.hardware)
    statuses = {
        details.get("status")
        for details in session.state.get("nodes", {}).values()
    }
    final_status = (
        "completed_with_escalations"
        if "blocked" in statuses
        else "completed"
    )
    consult_advisor(
        context,
        trigger="session_end",
        node_id="SESSION",
        scorecard={
            "status": final_status,
            "nodes": to_builtin(session.state.get("nodes", {})),
            "budget": budget.as_dict(),
        },
        device_context={
            "expected": to_builtin(repository.hardware.get("expected", {})),
            "working_values": to_builtin(
                session.state.get("working_values", {})
            ),
            "target": target,
            "z_gain": float(z_gain),
        },
    )
    session.event(
        "session_completed",
        decision=final_status,
        reason=(
            "one or more nodes require human review"
            if final_status != "completed"
            else "all target nodes completed"
        ),
        budget=budget.as_dict(),
    )
    session.finish(final_status)
    return _summary(session, budget)
