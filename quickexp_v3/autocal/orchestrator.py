"""Resumable dependency scheduler for policy-governed calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

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
from .nodes import NodeOutcome, SessionContext, StopRequested, run_node
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
                )
                if attempt < policy.max_node_attempts:
                    session.event(
                        "retake",
                        node=spec.node_id,
                        decision="retry",
                        reason=outcome.reason,
                        attempt=attempt,
                        gates=to_builtin(outcome.gates or {}),
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
                )

            final_status = "done" if outcome.status in {"done", "skipped"} else "blocked"
            session.update_node(
                spec.node_id,
                status=final_status,
                last_csv=outcome.last_csv,
                last_values=outcome.values,
                reason=outcome.reason,
                proposals=list(outcome.proposals),
            )
            if final_status == "blocked":
                session.event(
                    "escalated",
                    node=spec.node_id,
                    decision="blocked",
                    reason=outcome.reason,
                    gates=to_builtin(outcome.gates or {}),
                )
            else:
                session.event(
                    "node_completed",
                    node=spec.node_id,
                    decision="done",
                    reason=outcome.reason,
                    proposals=list(outcome.proposals),
                )
            break

    statuses = {
        details.get("status")
        for details in session.state.get("nodes", {}).values()
    }
    final_status = (
        "completed_with_escalations"
        if "blocked" in statuses
        else "completed"
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
