"""Acquisition, fitting, gates, and proposal creation for autocal nodes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .. import ide
from ..backend import SyntheticBackend
from ..config import accepted_calibration_values
from ..data import BackendResult
from ..errors import AnalysisError, ConfigError
from ..fit_calibration import (
    promote_proposal,
    write_calibration_proposals,
)
from ..fit_stats import pi_consistency
from ..iq_gmm import fit_iq_gmm, iq_calibration_records, load_iq_shots
from ..native_fit import (
    fit_loopback,
    fit_ramsey,
    fit_t1,
)
from ..native_fit_ext import echo_calibration_record, fit_echo
from ..notch_fit import (
    fit_complex_notch,
    fit_spectroscopy_features,
    notch_calibration_record,
)
from ..punchout_fit import fit_punchout, punchout_calibration_record
from ..rabi_fit import calibration_record as rabi_calibration_record
from ..rabi_fit import fit_rabi
from ..resonator_flux import (
    calibration_record as resonator_flux_calibration_record,
)
from ..resonator_flux import (
    cosine_frequency,
    fit_resonator_flux,
    frequency_from_calibration_record,
)
from ..runtime import ExperimentRunner
from ..synthetic_device import write_native_pair
from ..util import dotted_get, to_builtin, utc_now
from .budget import BudgetModel, BudgetTracker
from .graph import NodeSpec
from .policy import AutocalPolicy
from .search import centered_sweep, expected_center, search_attempt
from .session import AutocalSession


class StopRequested(RuntimeError):
    """The operator-created STOP sentinel was observed between acquisitions."""


@dataclass(frozen=True)
class NodeOutcome:
    status: str
    reason: str
    values: Mapping[str, Any]
    proposals: Tuple[str, ...] = ()
    last_csv: Optional[str] = None
    gates: Mapping[str, Any] = None
    classification: Optional[Mapping[str, Any]] = None


@dataclass
class SessionContext:
    project_root: Path
    session: AutocalSession
    policy: AutocalPolicy
    budget: BudgetTracker
    budget_model: BudgetModel
    autonomy_level: int
    z_gain: float
    live_hardware: bool = False
    backend: Any = None

    def repository(self):
        return ide.load_repository(self.project_root)

    def current_value(self, address: str) -> Any:
        accepted = accepted_calibration_values(self.repository().calibration)
        return dotted_get(accepted, address)

    def working_value(self, address: str, default: Any = None) -> Any:
        working = self.session.state.get("working_values", {})
        if address in working:
            return working[address]
        current = self.current_value(address)
        return default if current is None else current


def _gate(value: Any, threshold: str, passed: bool) -> dict:
    try:
        scalar = float(value)
        value = scalar if np.isfinite(scalar) else str(scalar)
    except (TypeError, ValueError):
        value = to_builtin(value)
    return {
        "value": value,
        "threshold": str(threshold),
        "passed": bool(passed),
    }


def classify_failure(candidates: Any, assessment: Any) -> dict:
    """Classify failed evidence as instrument, identity, or estimation.

    Coverage failures are class A and carry the next acquisition remediation.
    Comparable real candidates are class B identity ambiguity. A sufficiently
    covered measurement with one clear candidate is class C estimation error.
    """
    from .hp.remediation import next_remediation

    real = sorted(
        (item for item in candidates if not item.is_null),
        key=lambda item: int(item.rank),
    )
    if not getattr(assessment, "sufficient", False):
        step = next_remediation(
            assessment,
            attempted=(),
            current_overrides={},
        )
        return {
            "failure_class": "A",
            "coverage_reasons": tuple(
                getattr(assessment, "reasons", ())
            ),
            "candidate_count": len(real),
            "proposed_remediation": (
                step.step_id if step is not None else None
            ),
        }

    ambiguous = bool(
        len(real) >= 2
        and abs(float(real[1].contrast))
        >= 0.5 * abs(float(real[0].contrast))
    )
    return {
        "failure_class": "B" if ambiguous else "C",
        "coverage_reasons": (),
        "candidate_count": len(real),
        "proposed_remediation": None,
    }


def _averaging_value(
    ctx: SessionContext,
    preset: str,
    attempt: int,
    *,
    name: str = "hard_avg",
) -> int:
    repository = ctx.repository()
    base = int(
        repository.presets.get(preset, {})
        .get("parameters", {})
        .get(name, 1)
    )
    requested = base * min(2 ** max(int(attempt) - 1, 0), 4)
    limits = repository.hardware.get("limits", {}).get(name)
    maximum = int(limits[1]) if isinstance(limits, (list, tuple)) else requested
    return max(1, min(requested, maximum))


def _resonator_lookup_prediction(
    ctx: SessionContext,
) -> tuple[Optional[float], Optional[float], str]:
    """Evaluate the session or accepted resonator lookup at the working Z."""
    working = ctx.session.state.get("working_values", {})
    working = working if isinstance(working, Mapping) else {}
    proposed = working.get("lookups.resonator_vs_flux")
    if isinstance(proposed, Mapping):
        try:
            parameters = proposed["parameters"]
            minimum = float(
                working["session.resonator_lookup_z_min"]
            )
            maximum = float(
                working["session.resonator_lookup_z_max"]
            )
            rmse = float(
                working["session.resonator_lookup_rmse_mhz"]
            )
            if not minimum <= float(ctx.z_gain) <= maximum:
                return (
                    None,
                    None,
                    "working Z is outside the newly fitted resonator lookup",
                )
            predicted = float(
                cosine_frequency(float(ctx.z_gain), **dict(parameters))
            )
        except (KeyError, TypeError, ValueError, ConfigError):
            return None, None, "new resonator lookup is incomplete"
        if not np.isfinite(rmse) or rmse <= 0:
            return None, None, "new resonator lookup has no positive RMSE"
        return predicted, rmse, "newly fitted resonator lookup"

    try:
        record = (
            ctx.repository()
            .calibration["records"]["lookups"]["resonator_vs_flux"]
        )
        predicted = float(
            frequency_from_calibration_record(record, float(ctx.z_gain))
        )
        uncertainty = record.get("uncertainty", {})
        rmse = float(uncertainty["rmse_mhz"])
    except (KeyError, TypeError, ValueError, ConfigError):
        return None, None, "no accepted in-domain resonator lookup"
    if not np.isfinite(rmse) or rmse <= 0:
        return None, None, "accepted resonator lookup has no positive RMSE"
    return predicted, rmse, "accepted resonator lookup"


def _record(
    *,
    value: Any,
    unit: str,
    source_csv: Path,
    analysis: str,
    quality: Mapping[str, Any],
    uncertainty: Any = None,
    valid_domain: Optional[Mapping[str, Any]] = None,
    model: str,
    notes: str = "",
) -> dict:
    return {
        "value": to_builtin(value),
        "unit": str(unit),
        "uncertainty": to_builtin(uncertainty),
        "provenance": {
            "source": str(source_csv),
            "source_yml": str(Path(source_csv).with_suffix(".yml")),
            "fitted_at": utc_now(),
            "analysis": str(analysis),
        },
        "quality": to_builtin(dict(quality)),
        "valid_domain": to_builtin(dict(valid_domain or {})),
        "model": str(model),
        "notes": str(notes),
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def _native_matrix(planned: Any, completed: Any) -> np.ndarray:
    columns = []
    for name in planned.plan.axes:
        columns.append(np.asarray(completed.data.axes[name]))
    for name in planned.plan.signal_names:
        columns.append(np.asarray(completed.data.signals[name]))
    return np.column_stack(columns)


def _materialized_native_directory(ctx: SessionContext) -> Path:
    """Keep live derived native pairs beside Quick data, never session state."""
    if not ctx.live_hardware:
        return ctx.session.native_directory
    configured = ctx.repository().hardware.get("storage", {}).get(
        "quick_native_root"
    )
    if not configured:
        raise ConfigError("live autocal requires storage.quick_native_root")
    return Path(str(configured)).expanduser().resolve()


def _native_path(
    ctx: SessionContext,
    *,
    node_id: str,
    planned: Any,
    completed: Any,
) -> Path:
    native_files = completed.data.metadata.get("native_files", ())
    if native_files:
        source = Path(native_files[0]).expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise AnalysisError(f"native acquisition is missing or empty: {source}")
        return source
    index = ctx.budget.total_runs
    return write_native_pair(
        ctx.session.native_directory,
        planned.plan,
        BackendResult(payload=_native_matrix(planned, completed)),
        index=index,
        title=f"{ctx.session.session_id}_{node_id}_{planned.plan.title}",
    )


def _acquire(
    ctx: SessionContext,
    *,
    node_id: str,
    experiment: str,
    preset: str,
    overrides: Optional[Mapping[str, Any]] = None,
    run_options: Optional[Mapping[str, Any]] = None,
) -> tuple:
    if ctx.session.stop_requested():
        raise StopRequested("autocal_runs/STOP was found")
    repository = ctx.repository()
    safe_overrides = ctx.policy.clamp_overrides(dict(overrides or {}))
    if "z_gain" not in safe_overrides:
        safe_overrides["z_gain"] = float(ctx.z_gain)
    acquisition_z_gain = float(safe_overrides["z_gain"])
    planner_backend = ctx.backend or SyntheticBackend(seed=0)
    planned = ExperimentRunner(repository, planner_backend).plan(
        experiment,
        preset,
        overrides=safe_overrides,
        run_options=run_options,
        title=f"{ctx.session.session_id}_{node_id}",
    )
    predicted = ctx.budget_model.estimate(planned.plan)
    ctx.budget.check(predicted)
    started = time.monotonic()
    try:
        completed = ide.run_experiment(
            ctx.project_root,
            experiment=experiment,
            preset=preset,
            overrides=safe_overrides,
            run_options=run_options,
            title=f"{ctx.session.session_id}_{node_id}",
            live_hardware=ctx.live_hardware,
            fixed_z_gain=acquisition_z_gain,
            analyze=False,
            show_plot=False,
            backend=ctx.backend,
        )
    finally:
        measured = time.monotonic() - started
        ctx.budget.record(measured)
        ctx.budget_model.observe(predicted, measured)
        ctx.session.set_budget(ctx.budget.as_dict())
        plt.close("all")
    csv_path = _native_path(
        ctx,
        node_id=node_id,
        planned=planned,
        completed=completed,
    )
    if planned.plan.quick_class == "IQScatter":
        expected_points = int(planned.plan.variables.get("rep", completed.data.points))
    else:
        expected_points = 1
        for axis in planned.plan.axes:
            expected_points *= np.asarray(
                planned.plan.variables.get(axis, completed.data.axes[axis])
            ).size
    if completed.data.points != expected_points:
        raise AnalysisError(
            f"{node_id} acquired {completed.data.points} points; "
            f"the plan declared {expected_points}"
        )
    ctx.session.event(
        "acquisition_completed",
        node=node_id,
        decision="fit",
        reason="native acquisition is complete",
        csv=str(csv_path),
        points=int(completed.data.points),
        predicted_seconds=float(predicted),
        measured_seconds=float(measured),
        total_runs=int(ctx.budget.total_runs),
    )
    return completed, planned, csv_path


def _proposal_id(session_id: str, node_id: str, address: str) -> str:
    identity = f"{session_id}\0{node_id}\0{address}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:8]
    safe_address = address.replace(".", "-").replace("_", "-")
    return f"{session_id}-{node_id}-{safe_address}-{digest}"


def _propose(
    ctx: SessionContext,
    *,
    node_id: str,
    records: Mapping[str, Mapping[str, Any]],
    gates_pass: bool,
    gate_table: Optional[Mapping[str, Any]] = None,
    ramsey_sign_confirmed: bool = False,
) -> tuple:
    identifiers = []
    working = {}
    for address, raw in records.items():
        proposal = deepcopy(dict(raw))
        for key in (
            "accepted_at",
            "accepted_by",
            "accepted_revision",
            "proposal_id",
            "created_at",
        ):
            proposal.pop(key, None)
        provenance = proposal.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
            proposal["provenance"] = provenance
        provenance.update(
            {
                "autocal_session": ctx.session.session_id,
                "autocal_node": node_id,
                "working_z_gain": float(ctx.z_gain),
            }
        )
        quality = proposal.setdefault("quality", {})
        if not isinstance(quality, dict):
            quality = {"fit_quality": to_builtin(quality)}
            proposal["quality"] = quality
        quality["autocal_gates"] = to_builtin(dict(gate_table or {}))
        quality["autocal_gates_passed"] = bool(gates_pass)
        proposal["record"] = str(address)
        proposal["status"] = "proposed"
        identifier = _proposal_id(ctx.session.session_id, node_id, address)
        write_calibration_proposals(
            ctx.project_root,
            {identifier: proposal},
        )
        identifiers.append(identifier)
        working[address] = proposal.get("value")
        ctx.session.event(
            "proposal_written",
            node=node_id,
            decision="proposed",
            reason="fit passed and proposal is inert until policy promotion",
            proposal_id=identifier,
            record=address,
            value=proposal.get("value"),
        )
        decision = ctx.policy.promotion_decision(
            autonomy_level=ctx.autonomy_level,
            proposal=proposal,
            current_value=ctx.current_value(address),
            gates_pass=gates_pass,
            working_z_gain=ctx.z_gain,
            ramsey_sign_confirmed=ramsey_sign_confirmed,
        )
        ctx.session.event(
            "promotion_evaluated",
            node=node_id,
            decision="promote" if decision.promote else "retain_proposal",
            reason=decision.reason,
            proposal_id=identifier,
            record=address,
        )
        if decision.promote:
            promote_proposal(
                ctx.project_root,
                identifier,
                accepted_by=f"autocal-L{ctx.autonomy_level}",
            )
            ctx.session.event(
                "auto_promoted",
                node=node_id,
                decision="accepted",
                reason=decision.reason,
                proposal_id=identifier,
                record=address,
            )
            ctx.session.state["calibration_revision"] = int(
                ctx.repository().calibration.get("revision", 0)
            )
    ctx.session.set_working_values(working)
    return tuple(identifiers), working


def _fit_event(
    ctx: SessionContext,
    node_id: str,
    *,
    csv_path: Path,
    gates: Mapping[str, Any],
    passed: bool,
    reason: str,
) -> None:
    ctx.session.event(
        "fit_evaluated",
        node=node_id,
        decision="accept" if passed else "retake",
        reason=reason,
        csv=str(csv_path),
        gates=to_builtin(dict(gates)),
    )


def consult_advisor(
    ctx: SessionContext,
    *,
    trigger: str,
    node_id: str,
    candidates: Tuple[Mapping[str, Any], ...] = (),
    probe_responses: Tuple[Mapping[str, Any], ...] = (),
    scorecard: Optional[Mapping[str, Any]] = None,
    device_context: Optional[Mapping[str, Any]] = None,
    images: Tuple[Path, ...] = (),
) -> Any:
    """Call and validate the configured out-of-band advisor, never execute it."""
    from .hp.advisor import (
        AdvisoryRequest,
        AdvisorError,
        ClaudeAdvisor,
        NullAdvisor,
        ReplayAdvisor,
        request_hash,
        validate_response,
    )
    from .hp.taxonomy import hypothesis_ids

    discrepancies = ctx.session.state.get("discrepancy_ledger", {}).get(
        "entries", ()
    )
    advisory_request = AdvisoryRequest(
        trigger=str(trigger),
        node_id=str(node_id),
        candidates=tuple(candidates),
        probe_responses=tuple(probe_responses),
        scorecard=dict(scorecard or {}),
        discrepancies=tuple(discrepancies),
        device_context=dict(device_context or {}),
        images=tuple(images),
    )
    audit_path = ctx.session.directory / "advisory.jsonl"
    if ctx.policy.advisor_mode == "claude":
        advisor = ClaudeAdvisor(
            model=ctx.policy.advisor_model,
            audit_path=audit_path,
            timeout_seconds=ctx.policy.advisor_timeout_seconds,
        )
    elif ctx.policy.advisor_mode == "replay":
        advisor = ReplayAdvisor(audit_path)
    else:
        advisor = NullAdvisor()
    response = None
    policy_logged = False
    try:
        response = advisor.advise(advisory_request)
        repository = ctx.repository()
        experiment_presets = {}
        for preset_name, preset in repository.presets.items():
            experiment = (
                preset.get("experiment") if isinstance(preset, Mapping) else None
            )
            if experiment:
                experiment_presets.setdefault(str(experiment), []).append(
                    str(preset_name)
                )
        action = response.proposed_action or {}
        estimate = 0.0
        if "experiment" in action:
            planned = ExperimentRunner(
                repository,
                ctx.backend or SyntheticBackend(seed=0),
            ).plan(
                str(action["experiment"]),
                str(action["preset"]),
                overrides=dict(action.get("overrides", {})),
                title=f"{ctx.session.session_id}_{node_id}_advisory_preview",
            )
            estimate = ctx.budget_model.estimate(planned.plan)
        all_hypotheses = tuple(
            dict.fromkeys(hypothesis_ids("qubit") + hypothesis_ids("resonator"))
        )
        remaining = max(
            ctx.budget.max_wall_clock_seconds - ctx.budget.spent_seconds,
            0.0,
        )
        validate_response(
            response,
            hypothesis_ids=all_hypotheses,
            experiment_presets=experiment_presets,
            limits=repository.hardware.get("limits", {}),
            remaining_budget_seconds=remaining,
            estimated_action_seconds=estimate,
        )
        proposal_accepted = bool("experiment" in action)
        if isinstance(advisor, ClaudeAdvisor):
            advisor.audit_policy_decision(
                advisory_request,
                policy_accepted=proposal_accepted,
            )
            policy_logged = True
        if proposal_accepted:
            proposals = ctx.session.state.setdefault("advisory_proposals", [])
            proposals.append(
                {
                    "request_hash": request_hash(advisory_request),
                    "node_id": node_id,
                    "trigger": trigger,
                    "response": response.as_dict(),
                    "estimated_seconds": float(estimate),
                    "status": "validated_not_executed",
                }
            )
            ctx.session.save()
        for divergence in getattr(advisor, "divergences", ()):
            divergence_details = {
                str(key): value
                for key, value in dict(divergence).items()
                if str(key) != "event"
            }
            ctx.session.event(
                "advisory_divergence",
                node=node_id,
                decision="escalate",
                reason="ReplayAdvisor found no matching request hash",
                **to_builtin(divergence_details),
            )
        ctx.session.event(
            "advisor_completed",
            node=node_id,
            decision=("proposed" if proposal_accepted else "escalate"),
            reason=response.rationale,
            trigger=trigger,
            request_hash=request_hash(advisory_request),
            model=getattr(advisor, "model", ctx.policy.advisor_mode),
            response=response.as_dict(),
            policy_accepted=proposal_accepted,
            executed=False,
            replay_divergences=to_builtin(
                getattr(advisor, "divergences", ())
            ),
        )
        return response
    except (AdvisorError, ConfigError, ValueError) as error:
        if (
            isinstance(advisor, ClaudeAdvisor)
            and response is not None
            and not policy_logged
        ):
            advisor.audit_policy_decision(
                advisory_request,
                policy_accepted=False,
            )
        ctx.session.event(
            "advisor_failed",
            node=node_id,
            decision="escalate",
            reason=f"{type(error).__name__}: {error}",
            trigger=trigger,
            request_hash=request_hash(advisory_request),
        )
        return NullAdvisor().advise(advisory_request)


def _hypothesis_overlay(
    ctx: SessionContext,
    result: Any,
) -> Tuple[Path, ...]:
    directory = ctx.session.directory / "advisory_images"
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    seen = set()
    for candidate in result.candidates:
        if candidate.is_null or candidate.source_csv in seen:
            continue
        seen.add(candidate.source_csv)
        fit = fit_spectroscopy_features(
            candidate.source_csv,
            kind="qubit",
            signal="amplitude",
        )
        figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        axis.plot(fit.x, fit.measured, ".", markersize=3, label="measured")
        axis.plot(fit.x, fit.fitted, "-", linewidth=1.5, label="fit")
        axis.axvline(candidate.center_mhz, color="tab:red", linestyle="--")
        axis.set(
            xlabel="Qubit frequency (MHz)",
            ylabel=fit.signal_label,
            title="N5 candidate " + candidate.candidate_id,
        )
        axis.legend()
        axis.grid(alpha=0.25)
        path = directory / (candidate.candidate_id + ".png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        images.append(path)
    return tuple(images)


def _n0(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    repository = ctx.repository()
    connection = ide.inspect_connection(
        ctx.project_root,
        live_hardware=ctx.live_hardware,
    )
    facts = {
        "backend": (
            getattr(ctx.backend, "snapshot", lambda: {"backend": "live"})()
            if ctx.backend is not None
            else {"backend": "live" if ctx.live_hardware else "synthetic"}
        ),
        "bitfile_sha256": repository.hardware.get("qick", {}).get(
            "bitfile_sha256"
        ),
        "reference_clock_mhz": repository.hardware.get("qick", {}).get(
            "reference_clock_mhz"
        ),
        "adc_fullscale_counts": repository.hardware.get(
            "expected", {}
        ).get("adc_fullscale_counts"),
        "soccfg_sha256": None,
        "generator_metadata_sha256": None,
    }
    if connection is not None:
        soccfg_text = str(connection.soccfg)
        facts["soccfg_sha256"] = hashlib.sha256(
            soccfg_text.encode("utf-8")
        ).hexdigest()
        try:
            generator_metadata = connection.soccfg["gens"]
        except (KeyError, TypeError):
            generator_metadata = getattr(connection.soccfg, "gens", None)
        if generator_metadata is not None:
            facts["generator_metadata_sha256"] = hashlib.sha256(
                repr(generator_metadata).encode("utf-8")
            ).hexdigest()
        closer = getattr(connection, "close", None)
        if not callable(closer):
            closer = getattr(getattr(connection, "backend", None), "close", None)
        if callable(closer):
            closer()
    facts["missing_hardware_facts"] = [
        name
        for name in (
            "bitfile_sha256",
            "reference_clock_mhz",
            "adc_fullscale_counts",
        )
        if facts[name] is None
    ]
    ctx.session.state["facts"] = to_builtin(facts)
    ctx.session.save()
    return NodeOutcome("done", "connection and configured ports inspected", facts)


def _n1(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    overrides = {
        "soft_avg": _averaging_value(
            ctx,
            "loopback",
            attempt,
            name="soft_avg",
        )
    }
    if attempt > 1:
        overrides["r_offset"] = 0.0
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="loopback",
        preset="loopback",
        overrides=overrides,
    )
    fit = fit_loopback(csv_path)
    checks = fit.acceptance_gates(
        minimum_edge_snr=5.0,
        minimum_r_squared=0.85,
        maximum_edge_uncertainty_us=0.02,
    )
    edge = float(fit.parameters["edge_in_trace_us"])
    step = float(np.median(np.diff(fit.time_us)))
    edge_margin = min(edge - fit.time_us[0], fit.time_us[-1] - edge)
    gates = {
        "edge_snr": _gate(fit.statistics["edge_snr"], ">= 5", checks["edge_snr"]),
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.85", checks["r_squared"]),
        "edge_uncertainty_us": _gate(
            fit.parameters["edge_uncertainty_us"],
            "<= 0.02",
            checks["edge_uncertainty_us"],
        ),
        "edge_inside_usable_record": _gate(
            edge_margin,
            f"> {2.0 * step:.9g} us (two sample bins from either edge)",
            checks["edge_inside_usable_record"],
        ),
    }
    passed = bool(all(checks.values()))
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="loopback edge gates passed" if passed else "loopback edge gates failed",
    )
    if not passed:
        return NodeOutcome("retake", "loopback edge gates failed", {}, last_csv=str(csv_path), gates=gates)
    record = _record(
        value=fit.recommended_r_offset_us,
        unit="us",
        source_csv=csv_path,
        analysis="quickexp_v3.native_fit.fit_loopback",
        quality=fit.statistics,
        uncertainty={
            "edge_us": fit.parameters["edge_uncertainty_us"],
            "rise_10_to_90_us": fit.parameters["rise_10_to_90_us"],
        },
        model="smoothed_iq_logistic_edge",
    )
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.r_offset": record},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome("done", "loopback timing fitted", values, proposals, str(csv_path), gates)


def _n2(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    if ctx.policy.adaptive_enabled("N2"):
        return _n2_adaptive(ctx, spec, attempt)
    center = float(ctx.working_value("defaults.r_freq", 6884.0))
    frequency = centered_sweep(
        center,
        30.0,
        121 if attempt == 1 else 241,
        bounds=ctx.repository().hardware["limits"]["r_freq"],
    )
    # The hardware-owned autocal cap is -20 dB, so search both plateaus on
    # the safe side of that cap rather than clipping an unsafe grid into
    # duplicate points.
    powers = np.linspace(-45.0, -20.0, 11 if attempt == 1 else 21)
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="resonator_spectroscopy",
        preset="resonator_power",
        overrides={
            "r_freq": frequency,
            "r_power": powers,
            "hard_avg": _averaging_value(
                ctx,
                "resonator_power",
                attempt,
            ),
        },
    )
    fit = fit_punchout(csv_path, prior_linewidth_mhz=0.5)
    passed = fit.passes(
        minimum_plateau_rows=2,
        minimum_shift_over_step=2.0,
        maximum_transition_width_db=15.0,
    )
    gates = {
        "status_resolved": _gate(fit.status, "resolved", fit.status == "resolved"),
        "shift_over_step": _gate(
            fit.statistics["shift_over_frequency_step"],
            ">= 2",
            fit.statistics["shift_over_frequency_step"] >= 2,
        ),
        "low_plateau_rows": _gate(
            fit.statistics["low_plateau_rows"],
            ">= 2",
            fit.statistics["low_plateau_rows"] >= 2,
        ),
        "high_plateau_rows": _gate(
            fit.statistics["high_plateau_rows"],
            ">= 2",
            fit.statistics["high_plateau_rows"] >= 2,
        ),
        "transition_width_db": _gate(
            fit.parameters.get("transition_width_db", "unavailable"),
            "<= 15",
            (
                fit.status == "resolved"
                and fit.parameters["transition_width_db"] <= 15.0
            ),
        ),
        "parameters_not_pinned": _gate(
            fit.statistics.get("pinned_parameters", []),
            "empty",
            not fit.statistics.get("pinned_parameters"),
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="punchout plateaus resolved" if passed else "punchout needs a denser retake",
    )
    if not passed:
        return NodeOutcome("retake", "punchout plateaus unresolved", {}, last_csv=str(csv_path), gates=gates)
    record = punchout_calibration_record(fit)
    plateau = float(fit.parameters["f_low_mhz"])
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.r_power": record},
        gates_pass=True,
        gate_table=gates,
    )
    values = {**values, "session.f_r_plateau": plateau}
    ctx.session.set_working_values({"session.f_r_plateau": plateau})
    return NodeOutcome("done", "punchout plateaus fitted", values, proposals, str(csv_path), gates)


def _store_adaptive_schedule(
    ctx: SessionContext,
    node_id: str,
    scheduler: Any,
    *,
    fixed_grid_rows: int,
) -> None:
    schedules = ctx.session.state.setdefault("adaptive_sweeps", {})
    schedules[node_id] = scheduler.as_dict()
    ctx.session.save()
    ctx.session.event(
        "adaptive_map_completed",
        node=node_id,
        decision="abort" if scheduler.aborted else "fit",
        reason=(
            scheduler.abort_reason
            if scheduler.aborted
            else "adaptive row cap reached"
        ),
        rows=len(scheduler.rows),
        fixed_grid_rows=int(fixed_grid_rows),
        row_fraction=float(len(scheduler.rows) / max(fixed_grid_rows, 1)),
        schedule=scheduler.as_dict(),
    )


def _n2_adaptive(
    ctx: SessionContext,
    spec: NodeSpec,
    attempt: int,
) -> NodeOutcome:
    from .hp.adaptive import AdaptiveRowScheduler, tracked_frequency_axis

    repository = ctx.repository()
    max_rows = min(ctx.policy.adaptive_max_rows, 6)
    scheduler = AdaptiveRowScheduler(
        (-45.0, -20.0),
        min(ctx.policy.adaptive_initial_rows, max_rows),
        max_rows,
        min(ctx.policy.adaptive_abort_after_rows, max_rows),
    )
    base_center = float(ctx.working_value("defaults.r_freq", 6884.0))
    row_matrices = []
    row_axes = []
    first_plan = None
    last_csv = None
    while not scheduler.done:
        if len(scheduler.rows) == scheduler.initial_rows:
            # Punchout needs both asymptotic plateaus, so the one adaptive
            # follow-up resolves the sparsely sampled high-power side of the
            # observed transition instead of spending it on another low row.
            measured_powers = sorted(row.value for row in scheduler.rows)
            power = 0.5 * (measured_powers[-2] + measured_powers[-1])
        else:
            power = scheduler.next_row()
        frequency = tracked_frequency_axis(
            base_center,
            30.0,
            121 if attempt == 1 else 241,
            repository.hardware["limits"]["r_freq"],
        )
        completed, planned, last_csv = _acquire(
            ctx,
            node_id=spec.node_id,
            experiment="resonator_spectroscopy",
            preset="resonator_fine",
            overrides={
                "r_freq": frequency,
                "r_power": float(power),
                "hard_avg": _averaging_value(
                    ctx,
                    "resonator_fine",
                    attempt,
                ),
            },
        )
        if first_plan is None:
            first_plan = planned
        matrix = _native_matrix(planned, completed)
        row_matrices.append(
            np.column_stack((np.full(matrix.shape[0], power), matrix))
        )
        row_axes.append(np.asarray(frequency, dtype=float))
        try:
            row_fit = fit_complex_notch(last_csv)
            center = float(row_fit.center_mhz)
            trackable = bool(
                np.isfinite(center)
                and float(frequency[0]) < center < float(frequency[-1])
            )
            uncertainty = float(
                row_fit.parameters.get("center_uncertainty_mhz", np.nan)
            )
        except (AnalysisError, ValueError, FloatingPointError):
            center = None
            uncertainty = None
            trackable = False
        scheduler.record(
            power,
            center_mhz=center,
            trackable=trackable,
            uncertainty_mhz=uncertainty,
        )
    _store_adaptive_schedule(ctx, spec.node_id, scheduler, fixed_grid_rows=11)
    if scheduler.aborted:
        return NodeOutcome(
            "retake",
            scheduler.abort_reason,
            {},
            last_csv=str(last_csv) if last_csv else None,
            classification={
                "failure_class": "A",
                "coverage_reasons": ("trackability",),
                "candidate_count": sum(row.trackable for row in scheduler.rows),
                "proposed_remediation": "readout_power",
            },
        )
    powers = np.asarray([row.value for row in scheduler.rows], dtype=float)
    map_plan = replace(
        first_plan.plan,
        axes=("r_power", "r_freq"),
        variables={
            **dict(first_plan.plan.variables),
            "r_power": powers,
            "r_freq": np.vstack(row_axes),
        },
        axis_units={"r_power": "dB", "r_freq": "MHz"},
        title=f"{ctx.session.session_id}_{spec.node_id}_adaptive_map",
    )
    csv_path = write_native_pair(
        _materialized_native_directory(ctx),
        map_plan,
        BackendResult(payload=np.vstack(row_matrices)),
        index=ctx.budget.total_runs + 1,
        title=map_plan.title,
        extra_metadata={"adaptive_schedule": scheduler.as_dict()},
    )
    fit = fit_punchout(csv_path, prior_linewidth_mhz=0.5)
    passed = fit.passes(
        minimum_plateau_rows=2,
        minimum_shift_over_step=2.0,
        maximum_transition_width_db=15.0,
    )
    gates = {
        "status_resolved": _gate(fit.status, "resolved", fit.status == "resolved"),
        "shift_over_step": _gate(
            fit.statistics["shift_over_frequency_step"],
            ">= 2",
            fit.statistics["shift_over_frequency_step"] >= 2,
        ),
        "low_plateau_rows": _gate(
            fit.statistics["low_plateau_rows"],
            ">= 2",
            fit.statistics["low_plateau_rows"] >= 2,
        ),
        "high_plateau_rows": _gate(
            fit.statistics["high_plateau_rows"],
            ">= 2",
            fit.statistics["high_plateau_rows"] >= 2,
        ),
        "transition_width_db": _gate(
            fit.parameters.get("transition_width_db", "unavailable"),
            "<= 15",
            bool(
                fit.status == "resolved"
                and fit.parameters["transition_width_db"] <= 15.0
            ),
        ),
        "parameters_not_pinned": _gate(
            fit.statistics.get("pinned_parameters", []),
            "empty",
            not fit.statistics.get("pinned_parameters"),
        ),
        "adaptive_rows": _gate(len(scheduler.rows), "<= 7", len(scheduler.rows) <= 7),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="adaptive punchout resolved" if passed else "adaptive punchout unresolved",
    )
    if not passed:
        return NodeOutcome(
            "retake",
            "adaptive punchout plateaus unresolved",
            {},
            last_csv=str(csv_path),
            gates=gates,
        )
    record = punchout_calibration_record(fit)
    plateau = float(fit.parameters["f_low_mhz"])
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.r_power": record},
        gates_pass=True,
        gate_table=gates,
    )
    session_values = {"session.f_r_plateau": plateau}
    ctx.session.set_working_values(session_values)
    return NodeOutcome(
        "done",
        "adaptive punchout plateaus fitted",
        {**values, **session_values},
        proposals,
        str(csv_path),
        gates,
    )


def _n3(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    if ctx.policy.adaptive_enabled("N3"):
        return _n3_adaptive(ctx, spec, attempt)
    z_values = np.linspace(-0.30, 0.30, 13)
    rows = []
    first_plan = None
    last_csv = None
    repository = ctx.repository()
    try:
        lookup = repository.calibration["records"]["lookups"]["resonator_vs_flux"]
    except (KeyError, TypeError):
        lookup = None
    try:
        map_center = float(lookup["value"]["parameters"]["center_frequency"])
    except (KeyError, TypeError, ValueError):
        map_center = float(ctx.working_value("defaults.r_freq", 6884.0))
    frequency = centered_sweep(
        map_center,
        6.0,
        101,
        bounds=repository.hardware["limits"]["r_freq"],
    )
    if ctx.live_hardware:
        if ctx.session.stop_requested():
            raise StopRequested("autocal_runs/STOP was found")
        preview = ExperimentRunner(repository, SyntheticBackend(seed=0)).plan(
            "resonator_spectroscopy",
            "resonator_fine",
            overrides={
                "r_freq": frequency,
                "z_gain": float(z_values[0]),
                "hard_avg": _averaging_value(
                    ctx,
                    "resonator_fine",
                    attempt,
                ),
            },
            title=f"{ctx.session.session_id}_{spec.node_id}_map",
        )
        predicted = ctx.budget_model.estimate(preview.plan)
        starts = {}
        observations = []

        def before_row(row_index: int, z_gain: float) -> None:
            if ctx.session.stop_requested():
                raise StopRequested("autocal_runs/STOP was found")
            ctx.budget.check(predicted)
            starts[row_index] = time.monotonic()

        def after_row(
            row_index: int,
            z_gain: float,
            completed: Optional[Any],
        ) -> None:
            measured = time.monotonic() - starts.get(row_index, time.monotonic())
            ctx.budget.record(measured)
            ctx.budget_model.observe(predicted, measured)
            ctx.session.set_budget(ctx.budget.as_dict())
            observations.append(
                {
                    "row": int(row_index),
                    "z_gain": float(z_gain),
                    "points": (
                        int(completed.data.points)
                        if completed is not None
                        else 0
                    ),
                    "predicted_seconds": float(predicted),
                    "measured_seconds": float(measured),
                }
            )

        completed_rows = ide.run_flux_sweep(
            ctx.project_root,
            experiment="resonator_spectroscopy",
            preset="resonator_fine",
            flux_values=z_values,
            overrides={
                "r_freq": frequency,
                "hard_avg": _averaging_value(
                    ctx,
                    "resonator_fine",
                    attempt,
                ),
            },
            title=f"{ctx.session.session_id}_{spec.node_id}_map",
            live_hardware=True,
            analyze_rows=False,
            show_plot=False,
            before_row=before_row,
            after_row=after_row,
        )
        if len(completed_rows) != len(z_values):
            raise AnalysisError(
                f"{spec.node_id} acquired {len(completed_rows)} of "
                f"{len(z_values)} flux rows"
            )
        native_files = completed_rows[0].data.metadata.get("native_files", ())
        if not native_files:
            raise AnalysisError("live resonator flux map has no native Saver path")
        csv_path = Path(native_files[0]).expanduser().resolve()
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            raise AnalysisError(
                f"live resonator flux map is missing or empty: {csv_path}"
            )
        for observation in observations:
            ctx.session.event(
                "acquisition_completed",
                node=spec.node_id,
                decision="fit",
                reason="native flux-map row is complete",
                csv=str(csv_path),
                total_runs=int(ctx.budget.total_runs),
                **observation,
            )
        plt.close("all")
    else:
        csv_path = None
    for row_index, z_gain in enumerate(z_values):
        if ctx.live_hardware:
            break
        completed, planned, last_csv = _acquire(
            ctx,
            node_id=spec.node_id,
            experiment="resonator_spectroscopy",
            preset="resonator_fine",
            overrides={
                "r_freq": frequency,
                "z_gain": float(z_gain),
                "hard_avg": _averaging_value(
                    ctx,
                    "resonator_fine",
                    attempt,
                ),
            },
        )
        if first_plan is None:
            first_plan = planned
        matrix = _native_matrix(planned, completed)
        rows.append(
            np.column_stack(
                (
                    np.full(matrix.shape[0], float(z_gain)),
                    matrix,
                )
            )
        )
    if not ctx.live_hardware:
        map_plan = replace(
            first_plan.plan,
            axes=("z_gain", "r_freq"),
            variables={
                **dict(first_plan.plan.variables),
                "z_gain": z_values,
                "r_freq": np.asarray(first_plan.plan.variables["r_freq"]),
            },
            axis_units={"z_gain": "", "r_freq": "MHz"},
            title=f"{ctx.session.session_id}_{spec.node_id}_map",
        )
        csv_path = write_native_pair(
            ctx.session.native_directory,
            map_plan,
            BackendResult(payload=np.vstack(rows)),
            index=ctx.budget.total_runs + 1,
            title=map_plan.title,
        )
    fit = fit_resonator_flux(
        csv_path,
        period_min=0.12,
        period_max=0.30,
    )
    passed = fit.passes(minimum_r_squared=0.95, maximum_rmse_mhz=0.2)
    gates = {
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.95", fit.statistics["r_squared"] >= 0.95),
        "rmse_mhz": _gate(fit.statistics["rmse_mhz"], "<= 0.2", fit.statistics["rmse_mhz"] <= 0.2),
        "complete_z_rows": _gate(len(fit.z_gain), ">= 6", len(fit.z_gain) >= 6),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="resonator flux cosine passed" if passed else "resonator flux cosine failed",
    )
    if not passed:
        return NodeOutcome("retake", "resonator flux fit failed", {}, last_csv=str(csv_path), gates=gates)
    record = resonator_flux_calibration_record(fit)
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"lookups.resonator_vs_flux": record},
        gates_pass=True,
        gate_table=gates,
    )
    lookup_session_values = {
        "session.resonator_lookup_rmse_mhz": float(
            fit.statistics["rmse_mhz"]
        ),
        "session.resonator_lookup_z_min": float(np.min(fit.z_gain)),
        "session.resonator_lookup_z_max": float(np.max(fit.z_gain)),
    }
    ctx.session.set_working_values(lookup_session_values)
    values = {**values, **lookup_session_values}
    return NodeOutcome("done", "resonator flux lookup fitted", values, proposals, str(csv_path), gates)


def _n3_adaptive(
    ctx: SessionContext,
    spec: NodeSpec,
    attempt: int,
) -> NodeOutcome:
    from .hp.adaptive import AdaptiveRowScheduler, tracked_frequency_axis

    repository = ctx.repository()
    scheduler = AdaptiveRowScheduler(
        (-0.30, 0.30),
        ctx.policy.adaptive_initial_rows,
        ctx.policy.adaptive_max_rows,
        ctx.policy.adaptive_abort_after_rows,
    )
    try:
        lookup = repository.calibration["records"]["lookups"][
            "resonator_vs_flux"
        ]
        map_center = float(lookup["value"]["parameters"]["center_frequency"])
    except (KeyError, TypeError, ValueError):
        map_center = float(ctx.working_value("defaults.r_freq", 6884.0))
    previous_center = map_center
    row_matrices = []
    row_axes = []
    first_plan = None
    last_csv = None
    while not scheduler.done:
        z_gain = scheduler.next_row()
        frequency = tracked_frequency_axis(
            previous_center,
            6.0,
            101,
            repository.hardware["limits"]["r_freq"],
        )
        completed, planned, last_csv = _acquire(
            ctx,
            node_id=spec.node_id,
            experiment="resonator_spectroscopy",
            preset="resonator_fine",
            overrides={
                "r_freq": frequency,
                "z_gain": float(z_gain),
                "hard_avg": _averaging_value(
                    ctx,
                    "resonator_fine",
                    attempt,
                ),
            },
        )
        if first_plan is None:
            first_plan = planned
        matrix = _native_matrix(planned, completed)
        row_matrices.append(
            np.column_stack((np.full(matrix.shape[0], z_gain), matrix))
        )
        row_axes.append(np.asarray(frequency, dtype=float))
        try:
            row_fit = fit_complex_notch(last_csv)
            center = float(row_fit.center_mhz)
            trackable = bool(
                np.isfinite(center)
                and float(frequency[0]) < center < float(frequency[-1])
            )
            uncertainty = float(
                row_fit.parameters.get("center_uncertainty_mhz", np.nan)
            )
        except (AnalysisError, ValueError, FloatingPointError):
            center = None
            uncertainty = None
            trackable = False
        scheduler.record(
            z_gain,
            center_mhz=center,
            trackable=trackable,
            uncertainty_mhz=uncertainty,
        )
        if trackable and center is not None:
            previous_center = center
    _store_adaptive_schedule(ctx, spec.node_id, scheduler, fixed_grid_rows=13)
    if scheduler.aborted:
        return NodeOutcome(
            "retake",
            scheduler.abort_reason,
            {},
            last_csv=str(last_csv) if last_csv else None,
            classification={
                "failure_class": "A",
                "coverage_reasons": ("trackability",),
                "candidate_count": sum(row.trackable for row in scheduler.rows),
                "proposed_remediation": "window",
            },
        )
    z_values = np.asarray([row.value for row in scheduler.rows], dtype=float)
    map_plan = replace(
        first_plan.plan,
        axes=("z_gain", "r_freq"),
        variables={
            **dict(first_plan.plan.variables),
            "z_gain": z_values,
            "r_freq": np.vstack(row_axes),
        },
        axis_units={"z_gain": "", "r_freq": "MHz"},
        title=f"{ctx.session.session_id}_{spec.node_id}_adaptive_map",
    )
    csv_path = write_native_pair(
        _materialized_native_directory(ctx),
        map_plan,
        BackendResult(payload=np.vstack(row_matrices)),
        index=ctx.budget.total_runs + 1,
        title=map_plan.title,
        extra_metadata={"adaptive_schedule": scheduler.as_dict()},
    )
    fit = fit_resonator_flux(
        csv_path,
        period_min=0.12,
        period_max=0.30,
    )
    passed = fit.passes(minimum_r_squared=0.95, maximum_rmse_mhz=0.2)
    gates = {
        "r_squared": _gate(
            fit.statistics["r_squared"],
            ">= 0.95",
            fit.statistics["r_squared"] >= 0.95,
        ),
        "rmse_mhz": _gate(
            fit.statistics["rmse_mhz"],
            "<= 0.2",
            fit.statistics["rmse_mhz"] <= 0.2,
        ),
        "adaptive_z_rows": _gate(
            len(fit.z_gain),
            "6 to 7 rows",
            6 <= len(fit.z_gain) <= 7,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="adaptive resonator flux cosine passed"
        if passed
        else "adaptive resonator flux cosine failed",
    )
    if not passed:
        return NodeOutcome(
            "retake",
            "adaptive resonator flux fit failed",
            {},
            last_csv=str(csv_path),
            gates=gates,
        )
    record = resonator_flux_calibration_record(fit)
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"lookups.resonator_vs_flux": record},
        gates_pass=True,
        gate_table=gates,
    )
    lookup_session_values = {
        "session.resonator_lookup_rmse_mhz": float(fit.statistics["rmse_mhz"]),
        "session.resonator_lookup_z_min": float(np.min(fit.z_gain)),
        "session.resonator_lookup_z_max": float(np.max(fit.z_gain)),
    }
    ctx.session.set_working_values(lookup_session_values)
    return NodeOutcome(
        "done",
        "adaptive resonator flux lookup fitted",
        {**values, **lookup_session_values},
        proposals,
        str(csv_path),
        gates,
    )


def _n4(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    lookup_frequency, lookup_rmse, lookup_source = (
        _resonator_lookup_prediction(ctx)
    )
    if lookup_frequency is None or lookup_rmse is None:
        return NodeOutcome(
            "blocked",
            f"{lookup_source}; N4 will not acquire without a physics cross-check",
            {},
        )
    ctx.session.set_working_values(
        {
            "session.n4_lookup_frequency_mhz": float(lookup_frequency),
            "session.n4_lookup_rmse_mhz": float(lookup_rmse),
            "session.n4_lookup_source": str(lookup_source),
        }
    )
    center = float(
        ctx.working_value(
            "session.f_r_plateau",
            lookup_frequency,
        )
    )
    frequency = centered_sweep(
        center,
        search_attempt(10.0, attempt),
        201,
        bounds=ctx.repository().hardware["limits"]["r_freq"],
    )
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="resonator_spectroscopy",
        preset="resonator_fine",
        overrides={
            "r_freq": frequency,
            "hard_avg": _averaging_value(
                ctx,
                "resonator_fine",
                attempt,
            ),
        },
    )
    fit = fit_complex_notch(csv_path)
    fit_passed = fit.passes(
        minimum_r_squared=0.60,
        minimum_contrast_snr=4.0,
        minimum_edge_distance_over_fwhm=1.0,
    )
    lookup_delta = abs(float(fit.center_mhz) - lookup_frequency)
    lookup_sigma = lookup_delta / lookup_rmse
    lookup_consistent = lookup_sigma <= 3.0
    passed = bool(fit_passed and lookup_consistent)
    gates = {
        "r_squared_complex": _gate(
            fit.statistics["r_squared_complex"],
            ">= 0.60",
            fit.statistics["r_squared_complex"] >= 0.60,
        ),
        "contrast_snr": _gate(
            fit.statistics["contrast_snr"],
            ">= 4",
            fit.statistics["contrast_snr"] >= 4,
        ),
        "edge_distance_over_fwhm": _gate(
            fit.statistics["edge_distance_over_fwhm"],
            ">= 1",
            fit.statistics["edge_distance_over_fwhm"] >= 1,
        ),
        "parameters_not_pinned": _gate(
            fit.statistics.get("pinned_parameters", []),
            "empty",
            not fit.statistics.get("pinned_parameters"),
        ),
        "lookup_consistency_sigma": _gate(
            lookup_sigma,
            f"<= 3 ({lookup_source}, RMSE={lookup_rmse:.6g} MHz)",
            lookup_consistent,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason=(
            "complex notch and lookup-consistency gates passed"
            if passed
            else "notch window/SNR or lookup-consistency retake needed"
        ),
    )
    if not passed:
        if fit_passed and lookup_sigma > 5.0:
            return NodeOutcome(
                "blocked",
                "notch center disagrees with the resonator lookup by more than 5 sigma",
                {},
                last_csv=str(csv_path),
                gates=gates,
            )
        return NodeOutcome(
            "retake",
            "complex notch or lookup-consistency gates failed",
            {},
            last_csv=str(csv_path),
            gates=gates,
        )
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.r_freq": notch_calibration_record(fit)},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome("done", "working-point notch fitted", values, proposals, str(csv_path), gates)


def _spectroscopy_record(fit: Any, z_gain: float) -> dict:
    return _record(
        value=fit.center_mhz,
        unit="MHz",
        source_csv=fit.source_csv,
        analysis="quickexp_v3.notch_fit.fit_spectroscopy_features",
        quality=fit.statistics,
        uncertainty={
            "center_mhz": fit.parameters["center_uncertainty_mhz"],
            "fwhm_mhz": fit.parameters["fwhm_mhz"],
        },
        valid_domain={"z_gain": [float(z_gain), float(z_gain)]},
        model="lorentzian_feature_with_bic_model_selection",
    )


def _candidate_payload(candidate: Any) -> dict:
    return {
        "candidate_id": str(candidate.candidate_id),
        "center_mhz": float(candidate.center_mhz),
        "fwhm_mhz": float(candidate.fwhm_mhz),
        "contrast": float(candidate.contrast),
        "center_uncertainty_mhz": float(candidate.center_uncertainty_mhz),
        "local_snr": float(candidate.local_snr),
        "rank": int(candidate.rank),
        "source_csv": str(candidate.source_csv),
        "window_mhz": list(candidate.window_mhz),
        "is_null": bool(candidate.is_null),
        "statistics": to_builtin(dict(candidate.statistics)),
    }


def _n5_hp_device_context(
    ctx: SessionContext,
    candidates: Tuple[Any, ...],
) -> dict:
    repository = ctx.repository()
    defaults = repository.hardware.get("defaults", {})
    expected = repository.hardware.get("expected", {})
    real = [candidate for candidate in candidates if not candidate.is_null]
    frequency_scale = max(
        [abs(float(candidate.center_mhz)) for candidate in real] or [5600.0]
    )
    period = abs(float(expected.get("flux_period_z", 0.3)))
    period = max(period, 1.0e-6)
    linewidths = [abs(float(candidate.fwhm_mhz)) for candidate in real]
    linewidth = float(np.median(linewidths)) if linewidths else 1.0
    null_candidates = [candidate for candidate in candidates if candidate.is_null]
    detectable = (
        float(null_candidates[0].statistics.get("detectable_contrast", linewidth))
        if null_candidates
        else linewidth
    )
    prominence = max(
        [abs(float(candidate.contrast)) for candidate in real] or [0.0]
    )
    q_gain = min(
        max(0.15, 0.5 * abs(float(defaults.get("q_gain", 0.3)))),
        ctx.policy.q_gain_max,
    )
    return {
        "q_gain": q_gain,
        "q_gain_max": float(ctx.policy.q_gain_max),
        "z_gain": float(ctx.z_gain),
        "r_freq": float(ctx.working_value("defaults.r_freq", 6884.0)),
        "resonator_linewidth_mhz": float(
            expected.get("resonator_linewidth_mhz", 0.5)
        ),
        "resonator_flux_period_z": period,
        "q_delta_mhz": float(defaults.get("q_delta", -180.0)),
        "power_exponent_tolerance": 0.35,
        "flux_slope_tolerance_mhz_per_z": max(
            4.0 * frequency_scale / period,
            linewidth / max(period / 1500.0, 1.0e-9),
        ),
        "flux_curvature_tolerance_mhz_per_z2": max(
            4.0 * frequency_scale / (period ** 2),
            linewidth / max((period / 1500.0) ** 2, 1.0e-12),
        ),
        "rabi_exponent_tolerance": 0.35,
        "rabi_contrast_tolerance": 0.20,
        "dispersive_shift_tolerance_mhz": max(
            0.5 * float(expected.get("dispersive_shift_mhz", 1.0)),
            0.1,
        ),
        "qubit_flux_slope_mhz_per_z": float(
            expected.get("qubit_flux_slope_mhz_per_z", 0.0)
        ),
        "qubit_flux_curvature_mhz_per_z2": float(
            expected.get("qubit_flux_curvature_mhz_per_z2", 0.0)
        ),
        "neighbor_flux_slope_mhz_per_z": float(
            expected.get("neighbor_flux_slope_mhz_per_z", 0.0)
        ),
        "neighbor_flux_curvature_mhz_per_z2": float(
            expected.get("neighbor_flux_curvature_mhz_per_z2", 0.0)
        ),
        "expected_rabi_contrast": float(
            expected.get("rabi_contrast", 0.7)
        ),
        "expected_neighbor_rabi_contrast": 0.0,
        "expected_dispersive_shift_mhz": float(
            expected.get("dispersive_shift_mhz", 1.0)
        ),
        # The null explanation is penalized by measured detectability, not a
        # fit-quality gate or a device-specific constant.
        "null_candidate_score": float(
            -0.5
            * (prominence / max(detectable, np.finfo(float).eps)) ** 2
        ),
        "estimated_probe_run_seconds": float(
            max(ctx.budget_model.fixed_overhead_seconds, 1.0)
        ),
    }


def _acquire_flux_probe(
    ctx: SessionContext,
    *,
    node_id: str,
    probe: Any,
    runs: Tuple[Mapping[str, Any], ...],
) -> Tuple[Path, ...]:
    """Acquire P2 through the reviewed held-flux sweep and split native rows."""
    if not runs:
        return ()
    repository = ctx.repository()
    safe_runs = [ctx.policy.clamp_overrides(dict(run)) for run in runs]
    z_values = np.asarray([float(run["z_gain"]) for run in safe_runs])
    base = dict(safe_runs[0])
    base.pop("z_gain", None)
    base["hard_avg"] = _averaging_value(ctx, probe.preset, 1)
    planned_rows = []
    predicted_rows = []
    planner_backend = ctx.backend or SyntheticBackend(seed=0)
    for run in safe_runs:
        overrides = dict(base)
        overrides.update(run)
        planned = ExperimentRunner(repository, planner_backend).plan(
            probe.experiment,
            probe.preset,
            overrides=overrides,
            title=f"{ctx.session.session_id}_{node_id}_{probe.probe_id}",
        )
        planned_rows.append(planned)
        predicted_rows.append(ctx.budget_model.estimate(planned.plan))
    starts = {}
    observations = []

    def before_row(row_index: int, _z_gain: float) -> None:
        if ctx.session.stop_requested():
            raise StopRequested("autocal_runs/STOP was found")
        ctx.budget.check(predicted_rows[row_index])
        starts[row_index] = time.monotonic()

    def after_row(row_index: int, z_gain: float, completed: Optional[Any]) -> None:
        measured = time.monotonic() - starts.get(row_index, time.monotonic())
        predicted = predicted_rows[row_index]
        ctx.budget.record(measured)
        ctx.budget_model.observe(predicted, measured)
        ctx.session.set_budget(ctx.budget.as_dict())
        observations.append((row_index, z_gain, measured, completed))

    completed_rows = ide.run_flux_sweep(
        ctx.project_root,
        experiment=probe.experiment,
        preset=probe.preset,
        flux_values=z_values,
        overrides=base,
        title=f"{ctx.session.session_id}_{node_id}_{probe.probe_id}",
        live_hardware=ctx.live_hardware,
        analyze_rows=False,
        show_plot=False,
        backend=ctx.backend,
        before_row=before_row,
        after_row=after_row,
    )
    plt.close("all")
    if len(completed_rows) != len(planned_rows):
        raise AnalysisError(
            f"{node_id} {probe.probe_id} acquired "
            f"{len(completed_rows)} of {len(planned_rows)} rows"
        )
    first_index = ctx.budget.total_runs - len(completed_rows) + 1
    paths = []
    for row_index, (planned, completed) in enumerate(
        zip(planned_rows, completed_rows)
    ):
        path = write_native_pair(
            _materialized_native_directory(ctx),
            planned.plan,
            BackendResult(payload=_native_matrix(planned, completed)),
            index=first_index + row_index,
            title=planned.plan.title + "_row",
        )
        paths.append(path)
        measured = observations[row_index][2]
        ctx.session.event(
            "acquisition_completed",
            node=node_id,
            decision="fit",
            reason="held-flux hypothesis probe row is complete",
            csv=str(path),
            probe_id=probe.probe_id,
            z_gain=float(z_values[row_index]),
            points=int(completed.data.points),
            predicted_seconds=float(predicted_rows[row_index]),
            measured_seconds=float(measured),
            total_runs=int(ctx.budget.total_runs),
        )
    return tuple(paths)


def _persist_hypothesis_result(
    ctx: SessionContext,
    result: Any,
    *,
    product_address: str,
    device_context: Mapping[str, Any],
    coverage_inputs: Mapping[str, Any],
    probe_files: Mapping[str, Mapping[str, Tuple[str, ...]]],
    hypothesis_ids_in_play: Tuple[str, ...],
    predictions: Tuple[str, ...],
) -> None:
    from .hp.ledger import (
        DiscrepancyLedger,
        HypothesisLedger,
        render_discrepancy_report,
    )

    raw_hypotheses = ctx.session.state.get("hypothesis_ledger", {})
    hypotheses = (
        HypothesisLedger.from_dict(raw_hypotheses)
        if isinstance(raw_hypotheses, Mapping) and raw_hypotheses
        else HypothesisLedger(
            ctx.policy.max_backtracks_per_session,
            ctx.policy.max_backtracks_per_address,
        )
    )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in result.candidates
    }
    ranking = []
    seen = set()
    for row in result.scorecard.rows:
        candidate = candidate_by_id.get(row.candidate_id)
        if candidate is None or candidate.is_null or row.candidate_id in seen:
            continue
        seen.add(row.candidate_id)
        ranking.append(
            {
                "candidate_id": row.candidate_id,
                "hypothesis_id": row.hypothesis_id,
                "score": float(row.total_score),
                "center_mhz": float(candidate.center_mhz),
                "source_csv": str(candidate.source_csv),
            }
        )
    if ranking:
        hypotheses.record(
            product_address,
            ranking,
            evidence={
                "node_id": result.node_id,
                "probes_run": list(result.probes_run),
                "scorecard": result.scorecard.as_dict(),
            },
        )
    raw_discrepancies = ctx.session.state.get("discrepancy_ledger", {})
    discrepancies = (
        DiscrepancyLedger.from_dict(raw_discrepancies)
        if isinstance(raw_discrepancies, Mapping)
        else DiscrepancyLedger()
    )
    leader_responses = result.responses.get(
        result.adjudication.candidate_id,
        {},
    )
    context = _n5_hp_device_context(ctx, result.candidates)
    rabi = leader_responses.get("rabi_ping", {})
    if "rabi_gain_exponent" in rabi:
        discrepancies.record(
            "rabi_gain_linearity",
            1.0,
            float(rabi["rabi_gain_exponent"]),
            float(context["rabi_exponent_tolerance"]),
            ("weak-drive linear response",),
            (result.node_id,),
        )
    flux = leader_responses.get("flux_nudge", {})
    if "flux_slope_mhz_per_z" in flux:
        discrepancies.record(
            "flux_period_agreement",
            float(context["qubit_flux_slope_mhz_per_z"]),
            float(flux["flux_slope_mhz_per_z"]),
            float(context["flux_slope_tolerance_mhz_per_z"]),
            ("qubit and readout share the configured SQUID-loop period",),
            (result.node_id,),
        )
    dispersive = leader_responses.get("dispersive_response", {})
    if "dispersive_shift_mhz" in dispersive:
        discrepancies.record(
            "chi",
            0.5 * float(context["expected_dispersive_shift_mhz"]),
            0.5 * float(dispersive["dispersive_shift_mhz"]),
            0.5 * float(context["dispersive_shift_tolerance_mhz"]),
            ("dispersive approximation", "candidate prepares this qubit"),
            (result.node_id,),
        )
    real_candidates = [
        candidate for candidate in result.candidates if not candidate.is_null
    ]
    leader_candidate = candidate_by_id.get(result.adjudication.candidate_id)
    if leader_candidate is not None and len(real_candidates) >= 2:
        q_delta = float(device_context.get("q_delta_mhz", -180.0))
        alternate = min(
            (
                candidate
                for candidate in real_candidates
                if candidate.candidate_id != leader_candidate.candidate_id
            ),
            key=lambda candidate: abs(
                abs(candidate.center_mhz - leader_candidate.center_mhz)
                - abs(q_delta) / 2.0
            ),
        )
        measured_delta = -2.0 * abs(
            alternate.center_mhz - leader_candidate.center_mhz
        )
        sigma = max(
            abs(q_delta) * 0.10,
            2.0
            * (
                abs(leader_candidate.center_uncertainty_mhz)
                + abs(alternate.center_uncertainty_mhz)
            ),
        )
        discrepancies.record(
            "anharmonicity",
            q_delta,
            measured_delta,
            sigma,
            ("the secondary feature is the f02 two-photon shadow",),
            (result.node_id,),
        )
    ctx.session.state["hypothesis_ledger"] = hypotheses.as_dict()
    ctx.session.state["discrepancy_ledger"] = discrepancies.as_dict()
    ctx.session.save()
    report_path = ctx.session.directory / "discrepancy-report.md"
    report_path.write_text(
        render_discrepancy_report(discrepancies),
        encoding="utf-8",
    )
    ctx.session.event(
        "hypothesis_adjudicated",
        node=result.node_id,
        decision=result.adjudication.action,
        reason=result.adjudication.reason,
        source_csv=str(result.source_csv),
        candidates=[_candidate_payload(item) for item in result.candidates],
        coverage=to_builtin(result.coverage.__dict__),
        coverage_inputs=to_builtin(dict(coverage_inputs)),
        responses=to_builtin(result.responses),
        scorecard=result.scorecard.as_dict(),
        adjudication=to_builtin(result.adjudication.__dict__),
        probes_run=list(result.probes_run),
        probe_seconds=float(result.probe_seconds),
        product_address=product_address,
        probe_files=to_builtin(probe_files),
        device_context=to_builtin(device_context),
        hypotheses=list(hypothesis_ids_in_play),
        predictions=list(predictions),
        wanted="qubit_01",
        margin_threshold=float(ctx.policy.margin_threshold),
        discrepancy_report=str(report_path),
    )


def _n5_hypothesis(
    ctx: SessionContext,
    spec: NodeSpec,
    attempt: int,
) -> NodeOutcome:
    from .hp.candidates import extract_candidates
    from .hp.coverage import CoverageAssessment, assess_coverage
    from .hp.engine import HypothesisNodeSpec, run as run_hypothesis
    from .hp.taxonomy import hypothesis_ids

    repository = ctx.repository()
    q_delta_mhz = float(repository.hardware["defaults"].get("q_delta", -180.0))
    ctx.session.set_working_values({"session.q_delta_mhz": q_delta_mhz})
    limits = repository.hardware["limits"]["q_freq"]
    expected_q = repository.hardware.get("expected", {}).get("q_freq_mhz")
    if isinstance(expected_q, (list, tuple)) and len(expected_q) == 2:
        prior_window = tuple(sorted(float(value) for value in expected_q))
    else:
        accepted = float(ctx.working_value("defaults.q_freq", 5600.0))
        prior_window = (accepted - 1000.0, accepted + 1000.0)
    working = ctx.session.state.get("working_values", {})
    forced_center = (
        working.get("session.n5_hp_derived_center_mhz")
        if isinstance(working, Mapping)
        else None
    )
    if forced_center is not None:
        coarse_center = float(forced_center)
        coarse_span = min(400.0, float(np.ptp(limits)))
        active_prior = (
            coarse_center - coarse_span / 2.0,
            coarse_center + coarse_span / 2.0,
        )
    else:
        coarse_center = 0.5 * (prior_window[0] + prior_window[1])
        coarse_span = min(float(np.ptp(prior_window)), float(np.ptp(limits)))
        active_prior = prior_window
    if attempt > 1 and forced_center is None:
        coarse_span = min(
            max(3.0 * coarse_span, 4000.0),
            float(np.ptp(limits)),
        )
    coarse_points = min(max(int(round(coarse_span)) + 1, 801), 4001)
    coarse_axis = centered_sweep(
        coarse_center,
        coarse_span,
        coarse_points,
        bounds=limits,
    )
    _completed, _planned, coarse_csv = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="qubit_spectroscopy",
        preset="qubit_coarse",
        overrides={
            "q_freq": coarse_axis,
            "hard_avg": _averaging_value(ctx, "qubit_coarse", attempt),
        },
    )
    coarse_fit = fit_spectroscopy_features(
        coarse_csv,
        kind="qubit",
        signal="amplitude",
    )
    coarse_candidates = [
        candidate
        for candidate in extract_candidates(coarse_fit)
        if not candidate.is_null
    ][: ctx.policy.top_k_candidates]
    fine_candidates = []
    fine_assessments = []
    fine_measurements = []
    null_candidate = None
    fine_gain = min(
        max(
            0.15,
            0.5 * abs(float(repository.hardware["defaults"].get("q_gain", 0.3))),
        ),
        ctx.policy.q_gain_max,
    )
    for coarse_candidate in coarse_candidates:
        half_width = max(5.0 * abs(float(coarse_candidate.fwhm_mhz)), 5.0)
        fine_axis = centered_sweep(
            coarse_candidate.center_mhz,
            2.0 * half_width,
            201,
            bounds=limits,
        )
        _fine_run, _fine_plan, fine_csv = _acquire(
            ctx,
            node_id=spec.node_id,
            experiment="qubit_spectroscopy",
            preset="qubit_fine",
            overrides={
                "q_freq": fine_axis,
                "q_gain": fine_gain,
                "hard_avg": _averaging_value(ctx, "qubit_fine", attempt),
            },
        )
        fine_fit = fit_spectroscopy_features(
            fine_csv,
            kind="qubit",
            signal="amplitude",
        )
        fine_measurements.append(
            {
                "source_csv": str(fine_csv),
                "coarse_center_mhz": float(coarse_candidate.center_mhz),
                "prior_window": [float(fine_axis[0]), float(fine_axis[-1])],
                "scan_window": [float(fine_axis[0]), float(fine_axis[-1])],
                "points": int(fine_axis.size),
            }
        )
        extracted = extract_candidates(fine_fit)
        real = [candidate for candidate in extracted if not candidate.is_null]
        if real:
            selected = min(
                real,
                key=lambda candidate: abs(
                    float(candidate.center_mhz) - float(coarse_candidate.center_mhz)
                ),
            )
            fine_candidates.append(selected)
        if null_candidate is None:
            null_candidate = next(
                (candidate for candidate in extracted if candidate.is_null),
                None,
            )
        fine_assessments.append(
            assess_coverage(
                extracted,
                prior_window=(float(fine_axis[0]), float(fine_axis[-1])),
                scan_window=(float(fine_axis[0]), float(fine_axis[-1])),
                points=int(fine_axis.size),
                expected_fwhm_mhz=float(fine_fit.parameters["fwhm_mhz"]),
                expected_contrast=abs(float(fine_fit.parameters["amplitude"])),
            )
        )
    if not fine_candidates or null_candidate is None:
        return NodeOutcome(
            "retake",
            "no fine spectroscopy candidate could be extracted",
            {},
            last_csv=str(coarse_csv),
            classification={
                "failure_class": "A",
                "coverage_reasons": ("detectability",),
                "candidate_count": 0,
                "proposed_remediation": "averaging",
            },
        )
    fine_candidates.sort(key=lambda candidate: abs(float(candidate.contrast)), reverse=True)
    deduplicated = []
    for candidate in fine_candidates:
        if any(
            abs(float(candidate.center_mhz) - float(existing.center_mhz))
            <= 0.5 * max(abs(float(candidate.fwhm_mhz)), abs(float(existing.fwhm_mhz)))
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    ranked = tuple(
        replace(candidate, rank=index)
        for index, candidate in enumerate(deduplicated)
    )
    candidates = ranked + (replace(null_candidate, rank=len(ranked)),)
    prior_low, prior_high = sorted(active_prior)
    scan_low, scan_high = float(coarse_axis[0]), float(coarse_axis[-1])
    prior_coverage = max(
        0.0,
        min(prior_high, scan_high) - max(prior_low, scan_low),
    ) / max(prior_high - prior_low, np.finfo(float).eps)
    reasons = set()
    if prior_coverage < 0.9:
        reasons.add("prior_coverage")
    for assessment in fine_assessments:
        reasons.update(assessment.reasons)
    coverage = CoverageAssessment(
        sufficient=not reasons,
        reasons=tuple(sorted(reasons)),
        prior_coverage=float(prior_coverage),
        points_per_fwhm=float(
            min(item.points_per_fwhm for item in fine_assessments)
        ),
        detectable_contrast=float(
            max(item.detectable_contrast for item in fine_assessments)
        ),
        edge_margin_fwhm=float(
            min(item.edge_margin_fwhm for item in fine_assessments)
        ),
    )
    device_context = _n5_hp_device_context(ctx, candidates)

    probe_files = {}

    def probe_runner(
        _engine_context: Any,
        probe: Any,
        probe_candidate: Any,
        runs: Tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, float]:
        if probe.probe_id == "flux_nudge":
            paths = _acquire_flux_probe(
                ctx,
                node_id=spec.node_id,
                probe=probe,
                runs=tuple(runs),
            )
        else:
            acquired = []
            for run_overrides in runs:
                overrides = dict(run_overrides)
                averaging_name = (
                    "soft_avg"
                    if probe.probe_id == "drive_power_ladder"
                    else "hard_avg"
                )
                overrides[averaging_name] = _averaging_value(
                    ctx,
                    probe.preset,
                    attempt,
                    name=averaging_name,
                )
                _run, _plan, path = _acquire(
                    ctx,
                    node_id=spec.node_id,
                    experiment=probe.experiment,
                    preset=probe.preset,
                    overrides=overrides,
                )
                acquired.append(path)
            paths = tuple(acquired)
        probe_files.setdefault(probe_candidate.candidate_id, {})[
            probe.probe_id
        ] = tuple(str(path) for path in paths)
        return probe.extract_response(paths)

    engine_context = {
        **device_context,
        "coverage": coverage,
        "device_context": device_context,
        "margin_threshold": float(ctx.policy.margin_threshold),
        "probe_budget_seconds": float(ctx.policy.probe_budget_seconds),
        "top_k_candidates": int(ctx.policy.top_k_candidates),
        "candidate_prominence_ratio": float(
            ctx.policy.candidate_prominence_ratio
        ),
        "probe_runner": probe_runner,
    }
    hypotheses_in_play = hypothesis_ids("qubit")
    predictions = (
        "rabi_gain_linearity",
        "flux_period_agreement",
        "dispersive_shift",
    )
    hp_spec = HypothesisNodeSpec(
        node_id=spec.node_id,
        acquire=lambda _context, _attempt: Path(ranked[0].source_csv),
        extract=lambda _path: candidates,
        hypotheses=hypotheses_in_play,
        wanted="qubit_01",
        probes=(
            "drive_power_ladder",
            "flux_nudge",
            "dispersive_response",
            "rabi_ping",
        ),
        predictions=predictions,
        product_address="defaults.q_freq",
    )
    result = run_hypothesis(hp_spec, engine_context, attempt=attempt)
    coverage_inputs = {
        "active_prior": [float(active_prior[0]), float(active_prior[1])],
        "coarse_scan_window": [
            float(coarse_axis[0]),
            float(coarse_axis[-1]),
        ],
        "minimum_prior_coverage": 0.9,
        "fine_measurements": fine_measurements,
    }
    _persist_hypothesis_result(
        ctx,
        result,
        product_address="defaults.q_freq",
        device_context=device_context,
        coverage_inputs=coverage_inputs,
        probe_files=probe_files,
        hypothesis_ids_in_play=tuple(hypotheses_in_play),
        predictions=predictions,
    )
    classification = {
        "failure_class": result.adjudication.failure_class,
        "candidate_count": len(ranked),
        "coverage_reasons": tuple(result.coverage.reasons),
        "hypothesis": result.adjudication.hypothesis_id,
        "hypothesis_margin": float(result.adjudication.margin),
        "probes_run": tuple(result.probes_run),
        "proposed_remediation": (
            "averaging"
            if result.adjudication.action == "remediate"
            else None
        ),
    }
    if result.adjudication.action == "consult":
        flattened_responses = tuple(
            {
                "candidate_id": candidate_id,
                "probe_id": probe_id,
                **dict(response),
            }
            for candidate_id, by_probe in result.responses.items()
            for probe_id, response in by_probe.items()
        )
        advisor_response = consult_advisor(
            ctx,
            trigger=(
                "signature_mismatch"
                if result.adjudication.hypothesis_id == "novel"
                else "unresolved_scorecard"
            ),
            node_id=spec.node_id,
            candidates=tuple(
                _candidate_payload(candidate) for candidate in result.candidates
            ),
            probe_responses=flattened_responses,
            scorecard=result.scorecard.as_dict(),
            device_context=device_context,
            images=_hypothesis_overlay(ctx, result),
        )
        classification["advisor_hypothesis"] = (
            advisor_response.hypothesis_label
        )
        classification["advisor_action"] = to_builtin(
            advisor_response.proposed_action
        )
    if result.adjudication.action == "remediate":
        return NodeOutcome(
            "retake",
            result.adjudication.reason,
            {},
            last_csv=str(result.source_csv),
            classification=classification,
        )
    if result.adjudication.action == "derive_and_retry":
        leader = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id == result.adjudication.candidate_id
        )
        if result.adjudication.hypothesis_id == "f02_two_photon":
            derived = float(leader.center_mhz) - q_delta_mhz / 2.0
            ctx.session.set_working_values(
                {"session.n5_hp_derived_center_mhz": derived}
            )
            return NodeOutcome(
                "retake",
                "derived qubit_01 frequency from the f02 two-photon response",
                {"session.n5_hp_derived_center_mhz": derived},
                last_csv=str(result.source_csv),
                classification=classification,
            )
    if result.adjudication.action != "accept":
        return NodeOutcome(
            "blocked",
            result.adjudication.reason,
            {},
            last_csv=str(result.source_csv),
            classification=classification,
        )
    leader = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == result.adjudication.candidate_id
    )
    accepted_fit = fit_spectroscopy_features(
        leader.source_csv,
        kind="qubit",
        signal="amplitude",
        window_mhz=leader.window_mhz,
    )
    gates = {
        "coverage_sufficient": _gate(True, "true", True),
        "wanted_hypothesis": _gate(
            result.adjudication.hypothesis_id,
            "qubit_01",
            result.adjudication.hypothesis_id == "qubit_01",
        ),
        "hypothesis_margin": _gate(
            result.adjudication.margin,
            ">= hardware.autocal.hypothesis.margin_threshold",
            result.adjudication.margin >= ctx.policy.margin_threshold,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=Path(leader.source_csv),
        gates=gates,
        passed=True,
        reason="qubit identity accepted by perturbation-response margin",
    )
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={
            "defaults.q_freq": _spectroscopy_record(accepted_fit, ctx.z_gain)
        },
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome(
        "done",
        "qubit frequency accepted by hypothesis margin",
        values,
        proposals,
        str(leader.source_csv),
        gates,
        classification,
    )


def _n5(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    repository = ctx.repository()
    q_delta_mhz = float(
        repository.hardware["defaults"].get("q_delta", -180.0)
    )
    ctx.session.set_working_values(
        {"session.q_delta_mhz": q_delta_mhz}
    )
    accepted_center = float(ctx.working_value("defaults.q_freq", 5600.0))
    expected_q = repository.hardware.get("expected", {}).get("q_freq_mhz")
    expected_span = 2000.0
    if isinstance(expected_q, (list, tuple)) and len(expected_q) == 2:
        try:
            expected_values = np.asarray(expected_q, dtype=float)
            if (
                np.all(np.isfinite(expected_values))
                and expected_values[0] < expected_values[1]
            ):
                expected_span = float(np.ptp(expected_values))
        except (TypeError, ValueError):
            pass
    coarse_center = expected_center(
        repository.hardware,
        "q_freq_mhz",
        accepted_center,
    )
    session_working = ctx.session.state.get("working_values", {})
    session_working = (
        session_working if isinstance(session_working, Mapping) else {}
    )
    prior_candidate = session_working.get("session.n5_coarse_center_mhz")
    if attempt > 1 and prior_candidate is not None:
        coarse_center = float(prior_candidate)
        coarse_span = 400.0
    elif attempt > 1:
        coarse_span = min(
            max(3.0 * expected_span, 4000.0),
            float(np.ptp(repository.hardware["limits"]["q_freq"])),
        )
    else:
        coarse_span = expected_span
    coarse_points = min(
        max(int(round(coarse_span)) + 1, 801),
        4001,
    )
    coarse = centered_sweep(
        coarse_center,
        coarse_span,
        coarse_points,
        bounds=repository.hardware["limits"]["q_freq"],
    )
    _coarse_run, _coarse_plan, coarse_csv = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="qubit_spectroscopy",
        preset="qubit_coarse",
        overrides={
            "q_freq": coarse,
            "hard_avg": _averaging_value(
                ctx,
                "qubit_coarse",
                attempt,
            ),
        },
    )
    # The synthetic and native readout response is a complex dispersive arc.
    # Its deterministic IQ projection can be odd-symmetric on a centered fine
    # window; amplitude is the launcher-supported scalar channel that retains
    # the qubit feature without a rotation-sign ambiguity.
    coarse_fit = fit_spectroscopy_features(
        coarse_csv,
        kind="qubit",
        signal="amplitude",
    )
    from .hp.candidates import extract_candidates
    from .hp.coverage import assess_coverage

    coarse_candidates = extract_candidates(coarse_fit)
    coarse_assessment = assess_coverage(
        candidates=coarse_candidates,
        prior_window=(
            float(coarse_center - coarse_span / 2.0),
            float(coarse_center + coarse_span / 2.0),
        ),
        scan_window=(float(coarse[0]), float(coarse[-1])),
        points=int(coarse.size),
        expected_fwhm_mhz=float(
            coarse_fit.parameters.get("fwhm_mhz", 1.0)
        ),
        expected_contrast=abs(
            float(coarse_fit.parameters.get("amplitude", 0.0))
        ),
    )
    coarse_classification = classify_failure(
        coarse_candidates,
        coarse_assessment,
    )
    multi_feature = bool(coarse_fit.statistics.get("multi_feature"))
    shadow_recognized = not multi_feature
    if multi_feature:
        features = list(coarse_fit.parameters.get("features", ()))
        expected_shadow_separation = abs(q_delta_mhz) / 2.0
        if len(features) == 2:
            measured_separation = abs(
                float(features[0]["center_mhz"])
                - float(features[1]["center_mhz"])
            )
            shadow_recognized = (
                abs(measured_separation - expected_shadow_separation)
                <= max(10.0, 0.20 * expected_shadow_separation)
            )
        if shadow_recognized:
            selected_center = float(coarse_fit.center_mhz)
            half_window = min(
                max(5.0 * float(coarse_fit.parameters["fwhm_mhz"]), 10.0),
                0.40 * expected_shadow_separation,
            )
            coarse_fit = fit_spectroscopy_features(
                coarse_csv,
                kind="qubit",
                signal="amplitude",
                window_mhz=(
                    selected_center - half_window,
                    selected_center + half_window,
                ),
            )
        else:
            gates = {
                "coarse_single_or_f02_shadow": _gate(
                    coarse_fit.statistics["delta_bic_two_vs_one"],
                    "single feature or second feature within 20% of |q_delta|/2",
                    False,
                ),
                "coarse_r_squared": _gate(
                    coarse_fit.statistics["r_squared"],
                    ">= 0.50",
                    coarse_fit.statistics["r_squared"] >= 0.50,
                ),
                "coarse_contrast_snr": _gate(
                    coarse_fit.statistics["contrast_snr"],
                    ">= 3",
                    coarse_fit.statistics["contrast_snr"] >= 3,
                ),
                "coarse_parameters_not_pinned": _gate(
                    coarse_fit.statistics.get("pinned_parameters", []),
                    "empty",
                    not coarse_fit.statistics.get("pinned_parameters"),
                ),
            }
            _fit_event(
                ctx,
                spec.node_id,
                csv_path=coarse_csv,
                gates=gates,
                passed=False,
                reason="coarse qubit scan contains unresolved multiple features",
            )
            return NodeOutcome(
                "blocked",
                "coarse qubit features are physically ambiguous",
                {},
                last_csv=str(coarse_csv),
                gates=gates,
                classification=coarse_classification,
            )
    coarse_pass = coarse_fit.passes(
        minimum_r_squared=0.50,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.30,
    )
    if not coarse_pass:
        gates = {
            "coarse_single_or_f02_shadow": _gate(
                coarse_fit.statistics["delta_bic_two_vs_one"],
                "single feature or recognized f02/2 shadow",
                shadow_recognized,
            ),
            "coarse_r_squared": _gate(coarse_fit.statistics["r_squared"], ">= 0.50", coarse_fit.statistics["r_squared"] >= 0.50),
            "coarse_contrast_snr": _gate(coarse_fit.statistics["contrast_snr"], ">= 3", coarse_fit.statistics["contrast_snr"] >= 3),
            "coarse_parameters_not_pinned": _gate(
                coarse_fit.statistics.get("pinned_parameters", []),
                "empty",
                not coarse_fit.statistics.get("pinned_parameters"),
            ),
        }
        _fit_event(
            ctx,
            spec.node_id,
            csv_path=coarse_csv,
            gates=gates,
            passed=False,
            reason="coarse qubit feature did not pass",
        )
        return NodeOutcome(
            "retake",
            "coarse qubit feature failed",
            {},
            last_csv=str(coarse_csv),
            gates=gates,
            classification=coarse_classification,
        )
    ctx.session.set_working_values(
        {"session.n5_coarse_center_mhz": float(coarse_fit.center_mhz)}
    )
    fine = centered_sweep(
        coarse_fit.center_mhz,
        20.0,
        201,
        bounds=repository.hardware["limits"]["q_freq"],
    )
    q_gain = 0.5 * float(
        np.asarray(
            ctx.policy.clamp_overrides(
                {"q_gain": repository.presets["qubit_coarse"]["parameters"]["q_gain"]}
            )["q_gain"]
        )
    )
    q_gain = max(q_gain, 0.1)
    _fine_run, _fine_plan, fine_csv = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="qubit_spectroscopy",
        preset="qubit_fine",
        overrides={
            "q_freq": fine,
            "q_gain": q_gain,
            "hard_avg": _averaging_value(
                ctx,
                "qubit_fine",
                attempt,
            ),
        },
    )
    fit = fit_spectroscopy_features(
        fine_csv,
        kind="qubit",
        signal="amplitude",
    )
    fit_pass = fit.passes(
        minimum_r_squared=0.50,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.30,
    )
    center_consistent = (
        abs(fit.center_mhz - coarse_fit.center_mhz)
        <= max(float(coarse_fit.parameters["fwhm_mhz"]), float(np.median(np.diff(coarse))))
    )
    width_reduced = (
        float(fit.parameters["fwhm_mhz"])
        <= 0.70 * float(coarse_fit.parameters["fwhm_mhz"])
    )
    passed = bool(fit_pass and center_consistent and width_reduced)
    gates = {
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.50", fit.statistics["r_squared"] >= 0.50),
        "contrast_snr": _gate(fit.statistics["contrast_snr"], ">= 3", fit.statistics["contrast_snr"] >= 3),
        "single_feature": _gate(
            {
                "delta_bic_two_vs_one": fit.statistics[
                    "delta_bic_two_vs_one"
                ],
                "two_feature_resolved": fit.statistics[
                    "two_feature_resolved"
                ],
            },
            "no resolved two-feature BIC win",
            not fit.statistics["multi_feature"],
        ),
        "parameters_not_pinned": _gate(
            fit.statistics.get("pinned_parameters", []),
            "empty",
            not fit.statistics.get("pinned_parameters"),
        ),
        "center_uncertainty_over_fwhm": _gate(
            fit.statistics["center_uncertainty_fraction_of_fwhm"],
            "<= 0.30",
            fit.statistics["center_uncertainty_fraction_of_fwhm"] <= 0.30,
        ),
        "coarse_fine_center_consistent": _gate(
            abs(fit.center_mhz - coarse_fit.center_mhz),
            "<= coarse FWHM or one coarse bin",
            center_consistent,
        ),
        "fine_not_broader_than_coarse": _gate(
            fit.parameters["fwhm_mhz"] / coarse_fit.parameters["fwhm_mhz"],
            "<= 0.70",
            width_reduced,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=fine_csv,
        gates=gates,
        passed=passed,
        reason="coarse-to-fine qubit feature passed" if passed else "qubit fine-fit evidence failed",
    )
    if not passed:
        return NodeOutcome(
            "retake",
            "qubit spectroscopy evidence failed",
            {},
            last_csv=str(fine_csv),
            gates=gates,
            classification=coarse_classification,
        )
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.q_freq": _spectroscopy_record(fit, ctx.z_gain)},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome(
        "done",
        "qubit frequency fitted coarse-to-fine",
        values,
        proposals,
        str(fine_csv),
        gates,
        coarse_classification,
    )


def _n8(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="rabi",
        preset="rabi_length",
        overrides={
            "q_freq": float(ctx.working_value("defaults.q_freq", 5606.5)),
            "q_gain": float(ctx.working_value("defaults.q_gain", 0.4)),
            "q_length": np.linspace(0.02, 1.67, 34),
            "hard_avg": _averaging_value(
                ctx,
                "rabi_length",
                attempt,
            ),
        },
    )
    fit = fit_rabi(csv_path, variable="q_length")
    fit_passed = fit.passes(
        minimum_r_squared=0.70,
        minimum_oscillations=1.0,
        maximum_relative_pi_uncertainty=0.25,
    )
    consistency = pi_consistency(fit)
    contrast_passed = (
        consistency["measured_contrast_at_pi"] >= 0.60
    )
    odd_multiple_passed = bool(consistency["odd_multiple_consistent"])
    passed = bool(
        fit_passed and contrast_passed and odd_multiple_passed
    )
    gates = {
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.70", fit.statistics["r_squared"] >= 0.70),
        "oscillations": _gate(fit.statistics["oscillations"], ">= 1", fit.statistics["oscillations"] >= 1),
        "relative_pi_uncertainty": _gate(
            fit.statistics["relative_pi_uncertainty"],
            "<= 0.25",
            fit.statistics["relative_pi_uncertainty"] <= 0.25,
        ),
        "pi_inside_sweep": _gate(
            fit.pi_value,
            "inside acquired q_length axis",
            float(np.min(fit.x)) <= fit.pi_value <= float(np.max(fit.x)),
        ),
        "measured_contrast_at_pi": _gate(
            consistency["measured_contrast_at_pi"],
            ">= 0.60",
            contrast_passed,
        ),
        "pi_to_odd_half_period": _gate(
            consistency["pi_to_half_period_ratio"],
            "within 0.15 of an odd integer",
            odd_multiple_passed,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="Rabi pi gates passed" if passed else "Rabi retake needed",
    )
    if not passed:
        return NodeOutcome("retake", "Rabi fit gates failed", {}, last_csv=str(csv_path), gates=gates)
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"defaults.q_length": rabi_calibration_record(fit)},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome("done", "pi length fitted", values, proposals, str(csv_path), gates)


def _n9(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    shots = 2000 * (2 if attempt > 1 else 1)
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="iq_blobs",
        preset="iq_blobs",
        overrides={
            "shots": shots,
            "q_freq": float(ctx.working_value("defaults.q_freq", 5606.5)),
            "q_length": float(ctx.working_value("defaults.q_length", 0.115)),
            "r_freq": float(ctx.working_value("defaults.r_freq", 6883.11)),
            "r_power": float(ctx.working_value("defaults.r_power", -35.0)),
        },
    )
    source, _metadata, arrays = load_iq_shots(csv_path)
    fit = fit_iq_gmm(*arrays)
    passed = fit.passes(
        minimum_fidelity=0.80,
        minimum_shots_per_state=2000,
        maximum_angle_bootstrap_std=0.20,
    )
    gates = {
        "assignment_fidelity": _gate(fit.assignment_fidelity, ">= 0.80", fit.assignment_fidelity >= 0.80),
        "shots_per_state": _gate(fit.shots_per_state, ">= 2000", fit.shots_per_state >= 2000),
        "rotation_stability": _gate(fit.rotation_stability, "<= 0.20", fit.rotation_stability <= 0.20),
        "gmm_beats_centroid": _gate(
            fit.cross_validated_fidelity - fit.cross_validated_baseline_fidelity,
            ">= -0.001",
            fit.cross_validated_fidelity >= fit.cross_validated_baseline_fidelity - 0.001,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="IQ discrimination gates passed" if passed else "IQ discrimination retake needed",
    )
    if not passed:
        status = "blocked" if fit.assignment_fidelity < 0.60 else "retake"
        return NodeOutcome(status, "IQ discrimination gates failed", {}, last_csv=str(csv_path), gates=gates)
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records=iq_calibration_records(fit, source),
        gates_pass=True,
        gate_table=gates,
    )
    if "discrepancy_ledger" in ctx.session.state:
        from math import erf, sqrt

        from .hp.ledger import DiscrepancyLedger, render_discrepancy_report

        direction = np.asarray(fit.means[1] - fit.means[0], dtype=float)
        separation = float(np.linalg.norm(direction))
        unit = direction / max(separation, np.finfo(float).eps)
        variances = [
            float(unit @ covariance @ unit)
            for covariance in np.asarray(fit.covariances, dtype=float)
        ]
        pooled_sigma = sqrt(
            max(0.5 * sum(variances), np.finfo(float).eps)
        )
        discriminability = separation / pooled_sigma
        predicted_fidelity = 0.5 * (
            1.0 + erf(discriminability / (2.0 * sqrt(2.0)))
        )
        ledger = DiscrepancyLedger.from_dict(
            ctx.session.state.get("discrepancy_ledger", {})
        )
        ledger.record(
            "readout_fidelity_vs_snr",
            predicted_fidelity,
            float(fit.assignment_fidelity),
            max(
                float(fit.fidelity_uncertainty),
                1.0 / sqrt(max(2 * fit.shots_per_state, 1)),
            ),
            ("two equal-prior Gaussian readout clouds",),
            (spec.node_id,),
        )
        ctx.session.state["discrepancy_ledger"] = ledger.as_dict()
        ctx.session.save()
        (ctx.session.directory / "discrepancy-report.md").write_text(
            render_discrepancy_report(ledger),
            encoding="utf-8",
        )
    return NodeOutcome("done", "IQ threshold and fidelity fitted", values, proposals, str(csv_path), gates)


def _n10r(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    fidelity = float(ctx.working_value("derived.readout_fidelity", 0.0))
    if fidelity >= 0.80:
        return NodeOutcome(
            "done",
            "readout optimization skipped because fidelity already meets target",
            {"derived.readout_fidelity": fidelity},
        )
    return NodeOutcome(
        "blocked",
        "fidelity is below target; N10r proposals require a supervised hardware search",
        {},
    )


def _n11(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    stop = 29.9 if attempt == 1 else 59.8
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="t1",
        preset="t1",
        overrides={
            "delay": np.linspace(0.0, stop, 300),
            "q_freq": float(ctx.working_value("defaults.q_freq", 5606.5)),
            "q_length": float(ctx.working_value("defaults.q_length", 0.115)),
            "hard_avg": _averaging_value(ctx, "t1", attempt),
        },
        run_options={"population": False},
    )
    fit = fit_t1(csv_path, signal="IQ")
    passed = fit.passes(
        minimum_r_squared=0.70,
        minimum_span_over_t1=0.75,
        maximum_relative_t1_uncertainty=0.25,
    )
    gates = {
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.70", fit.statistics["r_squared"] >= 0.70),
        "span_over_t1": _gate(fit.statistics["span_over_t1"], ">= 0.75", fit.statistics["span_over_t1"] >= 0.75),
        "relative_t1_uncertainty": _gate(
            fit.statistics["relative_t1_uncertainty"],
            "<= 0.25",
            fit.statistics["relative_t1_uncertainty"] <= 0.25,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="T1 gates passed" if passed else "T1 delay span needs a retake",
    )
    if not passed:
        return NodeOutcome("retake", "T1 fit gates failed", {}, last_csv=str(csv_path), gates=gates)
    record = _record(
        value=fit.t1_us,
        unit="us",
        source_csv=csv_path,
        analysis="quickexp_v3.native_fit.fit_t1",
        quality=fit.statistics,
        uncertainty={"t1_us": fit.parameters["t1_uncertainty_us"]},
        valid_domain={"z_gain": [ctx.z_gain, ctx.z_gain]},
        model="bounded_exponential",
    )
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"derived.t1": record},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome("done", "T1 fitted", values, proposals, str(csv_path), gates)


def _ramsey_record(fit: Any, *, value: float, analysis: str) -> dict:
    return _record(
        value=value,
        unit="us" if "t2" in analysis else "MHz",
        source_csv=fit.source_csv,
        analysis="quickexp_v3.native_fit.fit_ramsey",
        quality=fit.statistics,
        uncertainty={
            "t2_star_us": fit.parameters["t2_star_uncertainty_us"],
            "fringe_mhz": fit.parameters["fitted_fringe_uncertainty_mhz"],
        },
        valid_domain={
            "z_gain": [
                float(
                    fit.metadata.get("parameters", {})
                    .get("var", {})
                    .get("z_gain", 0.0)
                )
            ]
            * 2
        },
        model=analysis,
    )


def _n12(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    working = float(ctx.working_value("defaults.q_freq", 5606.5))
    drives = (working + 0.5, working + 1.5)
    fits = []
    csv_paths = []
    for index, drive in enumerate(drives):
        _completed, _planned, csv_path = _acquire(
            ctx,
            node_id=spec.node_id,
            experiment="ramsey",
            preset="ramsey",
            overrides={
                "q_freq": drive,
                "fringe_frequency_mhz": 5.0,
                "delay": np.linspace(0.0, 4.99, 500),
                "q_length": float(ctx.working_value("defaults.q_length", 0.115)),
                "q_length_2": float(ctx.working_value("defaults.q_length", 0.115)),
                "hard_avg": _averaging_value(ctx, "ramsey", attempt),
            },
            run_options={"population": False},
        )
        fits.append(fit_ramsey(csv_path, signal="IQ"))
        csv_paths.append(csv_path)
    individual = [
        fit.passes(
            minimum_r_squared=0.70,
            minimum_oscillations=1.0,
            maximum_relative_t2_uncertainty=0.30,
        )
        for fit in fits
    ]
    frequencies = [float(fit.parameters["fitted_fringe_mhz"]) for fit in fits]
    sign_confirmed = bool(frequencies[1] < frequencies[0])
    passed = bool(all(individual) and sign_confirmed)
    gates = {
        "first_r_squared": _gate(fits[0].statistics["r_squared"], ">= 0.70", fits[0].statistics["r_squared"] >= 0.70),
        "second_r_squared": _gate(fits[1].statistics["r_squared"], ">= 0.70", fits[1].statistics["r_squared"] >= 0.70),
        "first_oscillations": _gate(fits[0].statistics["oscillations"], ">= 1", fits[0].statistics["oscillations"] >= 1),
        "second_oscillations": _gate(fits[1].statistics["oscillations"], ">= 1", fits[1].statistics["oscillations"] >= 1),
        "first_relative_t2_uncertainty": _gate(
            fits[0].statistics["relative_t2_uncertainty"],
            "<= 0.30",
            fits[0].statistics["relative_t2_uncertainty"] <= 0.30,
        ),
        "second_relative_t2_uncertainty": _gate(
            fits[1].statistics["relative_t2_uncertainty"],
            "<= 0.30",
            fits[1].statistics["relative_t2_uncertainty"] <= 0.30,
        ),
        "detuning_sign_confirmed": _gate(
            frequencies[0] - frequencies[1],
            "> 0 when drive is increased",
            sign_confirmed,
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_paths[-1],
        gates=gates,
        passed=passed,
        reason="two-point Ramsey sign confirmed" if passed else "Ramsey sign protocol failed",
    )
    if not passed:
        return NodeOutcome("retake", "Ramsey sign confirmation failed", {}, last_csv=str(csv_paths[-1]), gates=gates)
    second = fits[1]
    corrected_q = float(
        second.parameters["drive_frequency_mhz"]
        + second.parameters["detuning_mhz"]
    )
    records = {
        "derived.t2_ramsey": _ramsey_record(
            second,
            value=second.t2_star_us,
            analysis="decaying_cosine_t2",
        ),
        "defaults.q_freq": _ramsey_record(
            second,
            value=corrected_q,
            analysis="ramsey_fringe_frequency_correction",
        ),
    }
    records["defaults.q_freq"]["unit"] = "MHz"
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records=records,
        gates_pass=True,
        gate_table=gates,
        ramsey_sign_confirmed=True,
    )
    return NodeOutcome("done", "Ramsey T2* and detuning sign fitted", values, proposals, str(csv_paths[-1]), gates)


def _n13(ctx: SessionContext, spec: NodeSpec, attempt: int) -> NodeOutcome:
    stop = 49.0 if attempt == 1 else 98.0
    _completed, _planned, csv_path = _acquire(
        ctx,
        node_id=spec.node_id,
        experiment="echo",
        preset="echo",
        overrides={
            "delay": np.linspace(0.0, stop, 99),
            "pulse_count": 0,
            "q_freq": float(ctx.working_value("defaults.q_freq", 5606.5)),
            "q_length": float(ctx.working_value("defaults.q_length", 0.115)),
            "q_length_2": float(ctx.working_value("defaults.q_length", 0.115)),
            "hard_avg": _averaging_value(ctx, "echo", attempt),
        },
        run_options={"population": False},
    )
    fit = fit_echo(csv_path, signal="IQ", bootstrap_resamples=20)
    passed = fit.passes(
        minimum_r_squared=0.70,
        minimum_span_over_t=0.75,
        maximum_relative_t_uncertainty=0.25,
    )
    gates = {
        "r_squared": _gate(fit.statistics["r_squared"], ">= 0.70", fit.statistics["r_squared"] >= 0.70),
        "span_over_decay": _gate(fit.statistics["span_over_decay"], ">= 0.75", fit.statistics["span_over_decay"] >= 0.75),
        "relative_decay_uncertainty": _gate(
            fit.statistics["relative_decay_uncertainty"],
            "<= 0.25",
            fit.statistics["relative_decay_uncertainty"] <= 0.25,
        ),
        "parameters_not_pinned": _gate(
            fit.statistics.get("pinned_parameters", []),
            "empty",
            not fit.statistics.get("pinned_parameters"),
        ),
        "exponent_not_pinned": _gate(
            fit.statistics.get("n_pinned", False),
            "false",
            not fit.statistics.get("n_pinned", False),
        ),
    }
    _fit_event(
        ctx,
        spec.node_id,
        csv_path=csv_path,
        gates=gates,
        passed=passed,
        reason="echo gates passed" if passed else "echo span/model retake needed",
    )
    if not passed:
        return NodeOutcome("retake", "echo fit gates failed", {}, last_csv=str(csv_path), gates=gates)
    record = echo_calibration_record(fit)
    record["valid_domain"] = {"z_gain": [ctx.z_gain, ctx.z_gain]}
    proposals, values = _propose(
        ctx,
        node_id=spec.node_id,
        records={"derived.t2_echo.cycle_0": record},
        gates_pass=True,
        gate_table=gates,
    )
    return NodeOutcome("done", "echo fitted", values, proposals, str(csv_path), gates)


_HANDLERS: Mapping[str, Callable[[SessionContext, NodeSpec, int], NodeOutcome]] = {
    "N0": _n0,
    "N1": _n1,
    "N2": _n2,
    "N3": _n3,
    "N4": _n4,
    "N5": _n5,
    "N8": _n8,
    "N9": _n9,
    "N10r": _n10r,
    "N11": _n11,
    "N12": _n12,
    "N13": _n13,
}


def run_node(
    ctx: SessionContext,
    spec: NodeSpec,
    *,
    attempt: int,
) -> NodeOutcome:
    """Execute one node attempt through its registered launcher-grade path."""
    if spec.node_id == "N5" and ctx.policy.hypothesis_enabled("N5"):
        return _n5_hypothesis(ctx, spec, int(attempt))
    handler = _HANDLERS.get(spec.node_id)
    if handler is None:
        return NodeOutcome(
            "skipped" if spec.optional else "blocked",
            f"{spec.node_id} is not implemented in the v1 offline graph",
            {},
        )
    return handler(ctx, spec, int(attempt))
