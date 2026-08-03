"""Read-only reconstruction of autocal fit decisions from native data."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..errors import AnalysisError, ConfigError
from ..fit_stats import pi_consistency
from ..ide import load_repository
from ..iq_gmm import fit_iq_gmm, load_iq_shots
from ..native_fit import fit_loopback, fit_ramsey, fit_t1
from ..native_fit_ext import fit_echo
from ..notch_fit import fit_complex_notch, fit_spectroscopy_features
from ..punchout_fit import fit_punchout
from ..rabi_fit import fit_rabi
from ..resonator_flux import (
    cosine_frequency,
    fit_resonator_flux,
    frequency_from_calibration_record,
)
from .session import AutocalSession
from .hp.candidates import Candidate, extract_candidates
from .hp.advisor import AdvisoryResponse
from .hp.coverage import CoverageAssessment, assess_coverage
from .hp.probes import get_probe
from .hp.scorecard import adjudicate, build_scorecard
from .hp.engine import consistency_passes


@dataclass(frozen=True)
class ReplayVerification:
    """One fit decision reproduced from immutable native inputs."""

    event_index: int
    node_id: str
    csv_paths: tuple
    recorded_decision: str
    replayed_decision: str
    gate_passes: Mapping[str, bool]

    @property
    def matches(self) -> bool:
        return self.recorded_decision == self.replayed_decision

    def as_dict(self) -> dict:
        return {
            "event_index": int(self.event_index),
            "node_id": self.node_id,
            "csv_paths": [str(path) for path in self.csv_paths],
            "recorded_decision": self.recorded_decision,
            "replayed_decision": self.replayed_decision,
            "gate_passes": dict(self.gate_passes),
            "matches": self.matches,
        }


def _native_pair(path: Any) -> Path:
    source = Path(str(path)).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(
            f"replay native CSV does not exist or is empty: {source}"
        )
    sidecar = source.with_suffix(".yml")
    if not sidecar.is_file() or sidecar.stat().st_size == 0:
        raise FileNotFoundError(
            f"replay native YML does not exist or is empty: {sidecar}"
        )
    return source


def _recorded_gate_passes(event: Mapping[str, Any]) -> dict:
    gates = event.get("gates", {})
    gates = gates if isinstance(gates, Mapping) else {}
    return {
        str(name): bool(details.get("passed"))
        for name, details in gates.items()
        if isinstance(details, Mapping)
    }


def _lookup_prediction(
    session: AutocalSession,
) -> tuple[float, float]:
    working = session.state.get("working_values", {})
    working = working if isinstance(working, Mapping) else {}
    snapshot_frequency = working.get("session.n4_lookup_frequency_mhz")
    snapshot_rmse = working.get("session.n4_lookup_rmse_mhz")
    if snapshot_frequency is not None and snapshot_rmse is not None:
        frequency = float(snapshot_frequency)
        rmse = float(snapshot_rmse)
        if not np.all(np.isfinite([frequency, rmse])) or rmse <= 0:
            raise AnalysisError("replay N4 lookup snapshot is invalid")
        return frequency, rmse
    proposed = working.get("lookups.resonator_vs_flux")
    z_gain = float(session.state.get("z_gain", 0.0))
    if isinstance(proposed, Mapping):
        try:
            minimum = float(working["session.resonator_lookup_z_min"])
            maximum = float(working["session.resonator_lookup_z_max"])
            rmse = float(working["session.resonator_lookup_rmse_mhz"])
            parameters = proposed["parameters"]
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError(
                "replay state has an incomplete resonator lookup"
            ) from error
        if not minimum <= z_gain <= maximum:
            raise AnalysisError(
                "replay working Z is outside its session resonator lookup"
            )
        return float(cosine_frequency(z_gain, **dict(parameters))), rmse

    repository = load_repository(session.directory.parents[1])
    try:
        record = repository.calibration["records"]["lookups"][
            "resonator_vs_flux"
        ]
        predicted = float(frequency_from_calibration_record(record, z_gain))
        rmse = float(record["uncertainty"]["rmse_mhz"])
    except (KeyError, TypeError, ValueError, ConfigError) as error:
        raise AnalysisError(
            "replay has no accepted in-domain resonator lookup"
        ) from error
    return predicted, rmse


def _evaluate_n1(paths: Sequence[Path]) -> tuple[bool, dict]:
    fit = fit_loopback(paths[-1])
    gates = fit.acceptance_gates(
        minimum_edge_snr=5.0,
        minimum_r_squared=0.85,
        maximum_edge_uncertainty_us=0.02,
    )
    return bool(all(gates.values())), dict(gates)


def _evaluate_n2(
    paths: Sequence[Path],
    recorded: Optional[Mapping[str, bool]] = None,
) -> tuple[bool, dict]:
    fit = fit_punchout(paths[-1], prior_linewidth_mhz=0.5)
    gates = {
        "status_resolved": fit.status == "resolved",
        "shift_over_step": (
            fit.statistics["shift_over_frequency_step"] >= 2.0
        ),
        "low_plateau_rows": fit.statistics["low_plateau_rows"] >= 2,
        "high_plateau_rows": fit.statistics["high_plateau_rows"] >= 2,
        "transition_width_db": (
            fit.status == "resolved"
            and fit.parameters["transition_width_db"] <= 15.0
        ),
        "parameters_not_pinned": not fit.statistics.get(
            "pinned_parameters"
        ),
    }
    if recorded is not None and "adaptive_rows" in recorded:
        gates["adaptive_rows"] = len(fit.powers_db) <= 7
    return fit.passes(
        minimum_plateau_rows=2,
        minimum_shift_over_step=2.0,
        maximum_transition_width_db=15.0,
    ), gates


def _evaluate_n3(
    paths: Sequence[Path],
    recorded: Optional[Mapping[str, bool]] = None,
) -> tuple[bool, dict]:
    fit = fit_resonator_flux(
        paths[-1],
        period_min=0.12,
        period_max=0.30,
    )
    gates = {
        "r_squared": fit.statistics["r_squared"] >= 0.95,
        "rmse_mhz": fit.statistics["rmse_mhz"] <= 0.2,
        (
            "adaptive_z_rows"
            if recorded is not None and "adaptive_z_rows" in recorded
            else "complete_z_rows"
        ): (
            6 <= len(fit.z_gain) <= 7
            if recorded is not None and "adaptive_z_rows" in recorded
            else len(fit.z_gain) >= 6
        ),
    }
    return fit.passes(
        minimum_r_squared=0.95,
        maximum_rmse_mhz=0.2,
    ), gates


def _evaluate_n4(
    session: AutocalSession,
    paths: Sequence[Path],
) -> tuple[bool, dict]:
    fit = fit_complex_notch(paths[-1])
    predicted, rmse = _lookup_prediction(session)
    sigma = abs(float(fit.center_mhz) - predicted) / rmse
    gates = {
        "r_squared_complex": (
            fit.statistics["r_squared_complex"] >= 0.60
        ),
        "contrast_snr": fit.statistics["contrast_snr"] >= 4.0,
        "edge_distance_over_fwhm": (
            fit.statistics["edge_distance_over_fwhm"] >= 1.0
        ),
        "parameters_not_pinned": not fit.statistics.get(
            "pinned_parameters"
        ),
        "lookup_consistency_sigma": sigma <= 3.0,
    }
    fit_passed = fit.passes(
        minimum_r_squared=0.60,
        minimum_contrast_snr=4.0,
        minimum_edge_distance_over_fwhm=1.0,
    )
    return bool(fit_passed and sigma <= 3.0), gates


def _evaluate_n5(
    session: AutocalSession,
    paths: Sequence[Path],
    recorded: Mapping[str, bool],
) -> tuple[bool, dict]:
    coarse = fit_spectroscopy_features(
        paths[0],
        kind="qubit",
        signal="amplitude",
    )
    multi_feature = bool(coarse.statistics.get("multi_feature"))
    shadow_recognized = not multi_feature
    if multi_feature:
        working = session.state.get("working_values", {})
        working = working if isinstance(working, Mapping) else {}
        q_delta = working.get("session.q_delta_mhz")
        if q_delta is None:
            repository = load_repository(session.directory.parents[1])
            q_delta = repository.hardware["defaults"].get(
                "q_delta",
                -180.0,
            )
        expected_separation = abs(float(q_delta)) / 2.0
        features = list(coarse.parameters.get("features", ()))
        if len(features) == 2:
            measured_separation = abs(
                float(features[0]["center_mhz"])
                - float(features[1]["center_mhz"])
            )
            shadow_recognized = (
                abs(measured_separation - expected_separation)
                <= max(10.0, 0.20 * expected_separation)
            )
        if shadow_recognized:
            center = float(coarse.center_mhz)
            half_window = min(
                max(5.0 * float(coarse.parameters["fwhm_mhz"]), 10.0),
                0.40 * expected_separation,
            )
            coarse = fit_spectroscopy_features(
                paths[0],
                kind="qubit",
                signal="amplitude",
                window_mhz=(
                    center - half_window,
                    center + half_window,
                ),
            )
    coarse_gates = {
        "coarse_single_or_f02_shadow": shadow_recognized,
        "coarse_r_squared": coarse.statistics["r_squared"] >= 0.50,
        "coarse_contrast_snr": (
            coarse.statistics["contrast_snr"] >= 3.0
        ),
        "coarse_parameters_not_pinned": not coarse.statistics.get(
            "pinned_parameters"
        ),
    }
    coarse_passed = coarse.passes(
        minimum_r_squared=0.50,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.30,
    )
    if "coarse_r_squared" in recorded or len(paths) == 1:
        return bool(coarse_passed and shadow_recognized), coarse_gates
    if len(paths) < 2:
        raise AnalysisError("N5 replay needs coarse and fine native pairs")
    fine = fit_spectroscopy_features(
        paths[-1],
        kind="qubit",
        signal="amplitude",
    )
    center_consistent = (
        abs(fine.center_mhz - coarse.center_mhz)
        <= max(
            float(coarse.parameters["fwhm_mhz"]),
            float(np.median(np.diff(coarse.x))),
        )
    )
    width_reduced = (
        float(fine.parameters["fwhm_mhz"])
        <= 0.70 * float(coarse.parameters["fwhm_mhz"])
    )
    gates = {
        "r_squared": fine.statistics["r_squared"] >= 0.50,
        "contrast_snr": fine.statistics["contrast_snr"] >= 3.0,
        "single_feature": not fine.statistics["multi_feature"],
        "parameters_not_pinned": not fine.statistics.get(
            "pinned_parameters"
        ),
        "center_uncertainty_over_fwhm": (
            fine.statistics["center_uncertainty_fraction_of_fwhm"] <= 0.30
        ),
        "coarse_fine_center_consistent": center_consistent,
        "fine_not_broader_than_coarse": width_reduced,
    }
    fine_passed = fine.passes(
        minimum_r_squared=0.50,
        minimum_contrast_snr=3.0,
        maximum_center_uncertainty_fraction_of_fwhm=0.30,
    )
    return bool(
        coarse_passed and fine_passed and center_consistent and width_reduced
    ), gates


def _evaluate_n8(paths: Sequence[Path]) -> tuple[bool, dict]:
    fit = fit_rabi(paths[-1], variable="q_length")
    consistency = pi_consistency(fit)
    gates = {
        "r_squared": fit.statistics["r_squared"] >= 0.70,
        "oscillations": fit.statistics["oscillations"] >= 1.0,
        "relative_pi_uncertainty": (
            fit.statistics["relative_pi_uncertainty"] <= 0.25
        ),
        "pi_inside_sweep": (
            float(np.min(fit.x)) <= fit.pi_value <= float(np.max(fit.x))
        ),
        "measured_contrast_at_pi": (
            consistency["measured_contrast_at_pi"] >= 0.60
        ),
        "pi_to_odd_half_period": bool(
            consistency["odd_multiple_consistent"]
        ),
    }
    fit_passed = fit.passes(
        minimum_r_squared=0.70,
        minimum_oscillations=1.0,
        maximum_relative_pi_uncertainty=0.25,
    )
    return bool(
        fit_passed
        and gates["measured_contrast_at_pi"]
        and gates["pi_to_odd_half_period"]
    ), gates


def _evaluate_n9(paths: Sequence[Path]) -> tuple[bool, dict]:
    _source, _metadata, arrays = load_iq_shots(paths[-1])
    fit = fit_iq_gmm(*arrays)
    gates = {
        "assignment_fidelity": fit.assignment_fidelity >= 0.80,
        "shots_per_state": fit.shots_per_state >= 2000,
        "rotation_stability": fit.rotation_stability <= 0.20,
        "gmm_beats_centroid": (
            fit.cross_validated_fidelity
            >= fit.cross_validated_baseline_fidelity - 0.001
        ),
    }
    return fit.passes(
        minimum_fidelity=0.80,
        minimum_shots_per_state=2000,
        maximum_angle_bootstrap_std=0.20,
    ), gates


def _evaluate_n11(paths: Sequence[Path]) -> tuple[bool, dict]:
    fit = fit_t1(paths[-1], signal="IQ")
    gates = {
        "r_squared": fit.statistics["r_squared"] >= 0.70,
        "span_over_t1": fit.statistics["span_over_t1"] >= 0.75,
        "relative_t1_uncertainty": (
            fit.statistics["relative_t1_uncertainty"] <= 0.25
        ),
    }
    return fit.passes(
        minimum_r_squared=0.70,
        minimum_span_over_t1=0.75,
        maximum_relative_t1_uncertainty=0.25,
    ), gates


def _evaluate_n12(paths: Sequence[Path]) -> tuple[bool, dict]:
    if len(paths) < 2:
        raise AnalysisError("N12 replay needs both Ramsey native pairs")
    fits = [fit_ramsey(path, signal="IQ") for path in paths[-2:]]
    individual = [
        fit.passes(
            minimum_r_squared=0.70,
            minimum_oscillations=1.0,
            maximum_relative_t2_uncertainty=0.30,
        )
        for fit in fits
    ]
    frequencies = [
        float(fit.parameters["fitted_fringe_mhz"]) for fit in fits
    ]
    sign_confirmed = frequencies[1] < frequencies[0]
    gates = {
        "first_r_squared": fits[0].statistics["r_squared"] >= 0.70,
        "second_r_squared": fits[1].statistics["r_squared"] >= 0.70,
        "first_oscillations": fits[0].statistics["oscillations"] >= 1.0,
        "second_oscillations": fits[1].statistics["oscillations"] >= 1.0,
        "first_relative_t2_uncertainty": (
            fits[0].statistics["relative_t2_uncertainty"] <= 0.30
        ),
        "second_relative_t2_uncertainty": (
            fits[1].statistics["relative_t2_uncertainty"] <= 0.30
        ),
        "detuning_sign_confirmed": sign_confirmed,
    }
    return bool(all(individual) and sign_confirmed), gates


def _evaluate_n13(paths: Sequence[Path]) -> tuple[bool, dict]:
    fit = fit_echo(paths[-1], signal="IQ", bootstrap_resamples=20)
    gates = {
        "r_squared": fit.statistics["r_squared"] >= 0.70,
        "span_over_decay": fit.statistics["span_over_decay"] >= 0.75,
        "relative_decay_uncertainty": (
            fit.statistics["relative_decay_uncertainty"] <= 0.25
        ),
        "parameters_not_pinned": not fit.statistics.get(
            "pinned_parameters"
        ),
        "exponent_not_pinned": not fit.statistics.get("n_pinned", False),
    }
    return fit.passes(
        minimum_r_squared=0.70,
        minimum_span_over_t=0.75,
        maximum_relative_t_uncertainty=0.25,
    ), gates


def _evaluate(
    session: AutocalSession,
    node_id: str,
    paths: Sequence[Path],
    recorded: Mapping[str, bool],
) -> tuple[bool, dict]:
    evaluators = {
        "N1": _evaluate_n1,
        "N2": lambda values: _evaluate_n2(values, recorded),
        "N3": lambda values: _evaluate_n3(values, recorded),
        "N5": lambda values: _evaluate_n5(session, values, recorded),
        "N8": _evaluate_n8,
        "N9": _evaluate_n9,
        "N11": _evaluate_n11,
        "N12": _evaluate_n12,
        "N13": _evaluate_n13,
    }
    if node_id == "N4":
        return _evaluate_n4(session, paths)
    evaluator = evaluators.get(node_id)
    if evaluator is None:
        raise AnalysisError(
            f"no read-only replay evaluator is registered for {node_id}"
        )
    return evaluator(paths)


def _hp_candidate(raw: Mapping[str, Any]) -> Candidate:
    return Candidate(
        candidate_id=str(raw["candidate_id"]),
        center_mhz=float(raw["center_mhz"]),
        fwhm_mhz=float(raw["fwhm_mhz"]),
        contrast=float(raw["contrast"]),
        center_uncertainty_mhz=float(raw["center_uncertainty_mhz"]),
        local_snr=float(raw["local_snr"]),
        rank=int(raw["rank"]),
        source_csv=_native_pair(raw["source_csv"]),
        window_mhz=tuple(float(value) for value in raw["window_mhz"]),
        is_null=bool(raw.get("is_null", False)),
        statistics=dict(raw.get("statistics", {})),
    )


def _assert_response_matches(
    recorded: Mapping[str, Any],
    replayed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if set(recorded) != set(replayed):
        raise AnalysisError(
            "hypothesis replay response fields changed for " + label
        )
    for name in recorded:
        try:
            matches = np.isclose(
                float(recorded[name]),
                float(replayed[name]),
                rtol=1.0e-9,
                atol=1.0e-9,
                equal_nan=True,
            )
        except (TypeError, ValueError):
            matches = recorded[name] == replayed[name]
        if not bool(matches):
            raise AnalysisError(
                "hypothesis replay response changed for "
                + label
                + ":"
                + str(name)
            )


def _assert_candidate_matches(
    recorded: Candidate,
    replayed: Candidate,
) -> None:
    label = recorded.candidate_id
    if (
        recorded.candidate_id != replayed.candidate_id
        or recorded.rank != replayed.rank
        or recorded.is_null != replayed.is_null
        or recorded.source_csv != replayed.source_csv
    ):
        raise AnalysisError(
            "hypothesis replay candidate identity changed for " + label
        )
    for name in (
        "center_mhz",
        "fwhm_mhz",
        "contrast",
        "center_uncertainty_mhz",
        "local_snr",
    ):
        if not np.isclose(
            float(getattr(recorded, name)),
            float(getattr(replayed, name)),
            rtol=1.0e-9,
            atol=1.0e-9,
            equal_nan=True,
        ):
            raise AnalysisError(
                "hypothesis replay candidate field changed for "
                + label
                + ":"
                + name
            )
    if not np.allclose(
        np.asarray(recorded.window_mhz, dtype=float),
        np.asarray(replayed.window_mhz, dtype=float),
        rtol=1.0e-9,
        atol=1.0e-9,
        equal_nan=True,
    ):
        raise AnalysisError(
            "hypothesis replay candidate window changed for " + label
        )
    _assert_response_matches(
        recorded.statistics,
        replayed.statistics,
        label=label + ":statistics",
    )


def _replay_candidates_and_coverage(
    event: Mapping[str, Any],
    recorded_candidates: Sequence[Candidate],
    recorded_coverage: CoverageAssessment,
) -> tuple[tuple[Candidate, ...], CoverageAssessment, tuple[Path, ...]]:
    """Re-extract N5 candidates and coverage from immutable fine traces."""
    inputs = event.get("coverage_inputs")
    if not isinstance(inputs, Mapping):
        # Sessions created before coverage inputs were logged remain readable;
        # new sessions take the full reconstruction path below.
        return tuple(recorded_candidates), recorded_coverage, ()
    measurements = inputs.get("fine_measurements", ())
    if not isinstance(measurements, Sequence) or isinstance(
        measurements,
        (str, bytes),
    ):
        raise AnalysisError("hypothesis replay fine measurements must be a list")

    selected_candidates = []
    assessments = []
    null_candidate = None
    source_paths = []
    for measurement in measurements:
        if not isinstance(measurement, Mapping):
            raise AnalysisError(
                "hypothesis replay fine measurement must be a mapping"
            )
        source = _native_pair(measurement.get("source_csv"))
        source_paths.append(source)
        fit = fit_spectroscopy_features(
            source,
            kind="qubit",
            signal="amplitude",
        )
        extracted = extract_candidates(fit)
        real = [candidate for candidate in extracted if not candidate.is_null]
        if real:
            coarse_center = float(measurement["coarse_center_mhz"])
            selected_candidates.append(
                min(
                    real,
                    key=lambda candidate: abs(
                        float(candidate.center_mhz) - coarse_center
                    ),
                )
            )
        if null_candidate is None:
            null_candidate = next(
                (candidate for candidate in extracted if candidate.is_null),
                None,
            )
        assessments.append(
            assess_coverage(
                extracted,
                prior_window=tuple(
                    float(value) for value in measurement["prior_window"]
                ),
                scan_window=tuple(
                    float(value) for value in measurement["scan_window"]
                ),
                points=int(measurement["points"]),
                expected_fwhm_mhz=float(fit.parameters["fwhm_mhz"]),
                expected_contrast=abs(float(fit.parameters["amplitude"])),
            )
        )
    if not selected_candidates or null_candidate is None or not assessments:
        raise AnalysisError(
            "hypothesis replay could not reconstruct fine candidates"
        )

    selected_candidates.sort(
        key=lambda candidate: abs(float(candidate.contrast)),
        reverse=True,
    )
    deduplicated = []
    for candidate in selected_candidates:
        if any(
            abs(float(candidate.center_mhz) - float(existing.center_mhz))
            <= 0.5
            * max(
                abs(float(candidate.fwhm_mhz)),
                abs(float(existing.fwhm_mhz)),
            )
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    ranked = tuple(
        replace(candidate, rank=index)
        for index, candidate in enumerate(deduplicated)
    )
    replayed_candidates = ranked + (
        replace(null_candidate, rank=len(ranked)),
    )
    if len(replayed_candidates) != len(recorded_candidates):
        raise AnalysisError("hypothesis replay candidate count changed")
    for recorded, replayed in zip(recorded_candidates, replayed_candidates):
        _assert_candidate_matches(recorded, replayed)

    prior_low, prior_high = sorted(
        float(value) for value in inputs["active_prior"]
    )
    scan_low, scan_high = sorted(
        float(value) for value in inputs["coarse_scan_window"]
    )
    prior_coverage = max(
        0.0,
        min(prior_high, scan_high) - max(prior_low, scan_low),
    ) / max(prior_high - prior_low, np.finfo(float).eps)
    reasons = set()
    if prior_coverage < float(inputs.get("minimum_prior_coverage", 0.9)):
        reasons.add("prior_coverage")
    for assessment in assessments:
        reasons.update(assessment.reasons)
    replayed_coverage = CoverageAssessment(
        sufficient=not reasons,
        reasons=tuple(sorted(reasons)),
        prior_coverage=float(prior_coverage),
        points_per_fwhm=float(
            min(item.points_per_fwhm for item in assessments)
        ),
        detectable_contrast=float(
            max(item.detectable_contrast for item in assessments)
        ),
        edge_margin_fwhm=float(
            min(item.edge_margin_fwhm for item in assessments)
        ),
    )
    if (
        replayed_coverage.sufficient != recorded_coverage.sufficient
        or replayed_coverage.reasons != recorded_coverage.reasons
    ):
        raise AnalysisError("hypothesis replay coverage verdict changed")
    for name in (
        "prior_coverage",
        "points_per_fwhm",
        "detectable_contrast",
        "edge_margin_fwhm",
    ):
        if not np.isclose(
            float(getattr(replayed_coverage, name)),
            float(getattr(recorded_coverage, name)),
            rtol=1.0e-9,
            atol=1.0e-9,
            equal_nan=True,
        ):
            raise AnalysisError(
                "hypothesis replay coverage field changed: " + name
            )
    return replayed_candidates, replayed_coverage, tuple(source_paths)


def _verify_hypothesis_event(
    event_index: int,
    event: Mapping[str, Any],
) -> ReplayVerification:
    candidates = tuple(
        _hp_candidate(item)
        for item in event.get("candidates", ())
        if isinstance(item, Mapping)
    )
    if not candidates:
        raise AnalysisError("hypothesis replay event has no candidates")
    coverage_raw = event.get("coverage", {})
    coverage = CoverageAssessment(
        bool(coverage_raw.get("sufficient", False)),
        tuple(coverage_raw.get("reasons", ())),
        float(coverage_raw.get("prior_coverage", 0.0)),
        float(coverage_raw.get("points_per_fwhm", 0.0)),
        float(coverage_raw.get("detectable_contrast", "nan")),
        float(coverage_raw.get("edge_margin_fwhm", 0.0)),
    )
    candidates, coverage, replayed_sources = _replay_candidates_and_coverage(
        event,
        candidates,
        coverage,
    )
    recorded_responses = event.get("responses", {})
    probe_files = event.get("probe_files", {})
    replayed_responses = {}
    paths = [candidate.source_csv for candidate in candidates]
    paths.extend(replayed_sources)
    for candidate_id, by_probe in probe_files.items():
        if not isinstance(by_probe, Mapping):
            raise AnalysisError("hypothesis replay probe files must be mappings")
        for probe_id, raw_paths in by_probe.items():
            native_paths = tuple(_native_pair(path) for path in raw_paths)
            paths.extend(native_paths)
            replayed = get_probe(probe_id).extract_response(native_paths)
            recorded = recorded_responses.get(candidate_id, {}).get(probe_id)
            if not isinstance(recorded, Mapping):
                raise AnalysisError(
                    "hypothesis replay is missing recorded response for "
                    + str(candidate_id)
                    + ":"
                    + str(probe_id)
                )
            _assert_response_matches(
                recorded,
                replayed,
                label=str(candidate_id) + ":" + str(probe_id),
            )
            replayed_responses.setdefault(str(candidate_id), {})[
                str(probe_id)
            ] = replayed
    scorecard = build_scorecard(
        candidates,
        tuple(event.get("hypotheses", ())),
        replayed_responses,
        dict(event.get("device_context", {})),
        family="qubit",
    )
    replayed = adjudicate(
        scorecard,
        coverage,
        wanted=str(event.get("wanted", "qubit_01")),
        margin_threshold=float(event.get("margin_threshold", 2.0)),
        probes_remaining=False,
        consistency_passes=consistency_passes(
            tuple(event.get("predictions", ())),
            str(event.get("wanted", "qubit_01")),
            "qubit",
            scorecard,
        ),
    )
    recorded_adjudication = event.get("adjudication", {})
    changed = {
        "action": (recorded_adjudication.get("action"), replayed.action),
        "candidate_id": (
            recorded_adjudication.get("candidate_id"),
            replayed.candidate_id,
        ),
        "hypothesis_id": (
            recorded_adjudication.get("hypothesis_id"),
            replayed.hypothesis_id,
        ),
    }
    changed = {
        name: values for name, values in changed.items() if values[0] != values[1]
    }
    recorded_margin = float(recorded_adjudication.get("margin", "nan"))
    if not np.isclose(
        recorded_margin,
        replayed.margin,
        rtol=1.0e-9,
        atol=1.0e-9,
        equal_nan=True,
    ):
        changed["margin"] = (recorded_margin, replayed.margin)
    if changed:
        raise AnalysisError(
            "hypothesis replay diverged at event "
            + str(event_index)
            + ": "
            + str(changed)
        )
    return ReplayVerification(
        event_index=event_index,
        node_id=str(event.get("node", "")),
        csv_paths=tuple(dict.fromkeys(paths)),
        recorded_decision=str(recorded_adjudication.get("action")),
        replayed_decision=replayed.action,
        gate_passes={
            "coverage_sufficient": bool(coverage.sufficient),
            "margin_passes": bool(
                replayed.margin >= float(event.get("margin_threshold", 2.0))
            ),
        },
    )


def _verify_advisory_audit(session: AutocalSession) -> None:
    """Verify logged advisor responses without making an external request."""
    response_records = {}
    policy_records = {}
    audit_path = session.directory / "advisory.jsonl"
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("request_hash", ""))
            if not key:
                raise AnalysisError(
                    "advisory audit contains an empty request hash"
                )
            if isinstance(record.get("response"), Mapping):
                response_records.setdefault(key, record)
            if record.get("record_type") == "policy_validation":
                policy_records[key] = bool(record.get("policy_accepted"))

    for index, event in enumerate(session.events()):
        if event.get("event") != "advisor_completed":
            continue
        key = str(event.get("request_hash", ""))
        if not key:
            raise AnalysisError(
                "advisor replay event has no request hash at event " + str(index)
            )
        event_response = event.get("response")
        if not isinstance(event_response, Mapping):
            raise AnalysisError(
                "advisor replay event has no typed response at event " + str(index)
            )
        # Parsing is itself a schema replay check.
        parsed = AdvisoryResponse.from_mapping(event_response).as_dict()
        model = str(event.get("model", ""))
        if model == "null":
            expected_null = {
                "hypothesis_label": "novel",
                "proposed_action": {"action": "escalate"},
                "confidence": 0.0,
                "rationale": (
                    "No external advisor is configured; deterministic "
                    "escalation required."
                ),
                "discrepancy_notes": [],
                "novel_program_sketch": None,
            }
            if parsed != expected_null:
                raise AnalysisError(
                    "NullAdvisor replay changed its escalation response"
                )
            continue
        record = response_records.get(key)
        if record is None:
            raise AnalysisError(
                "advisor replay has no audit response for request hash " + key
            )
        recorded = AdvisoryResponse.from_mapping(record["response"]).as_dict()
        if recorded != parsed:
            raise AnalysisError(
                "advisor replay response changed for request hash " + key
            )
        audit_accepted = policy_records.get(
            key,
            bool(record.get("policy_accepted", False)),
        )
        if audit_accepted != bool(event.get("policy_accepted", False)):
            raise AnalysisError(
                "advisor replay policy decision changed for request hash " + key
            )


def verify_session_replay(session: AutocalSession) -> tuple:
    """Re-fit every logged decision and fail if any verdict changes."""
    _verify_advisory_audit(session)
    acquisitions: dict[str, list[Path]] = {}
    verifications = []
    hypothesis_nodes = set()
    for index, event in enumerate(session.events()):
        node_id = str(event.get("node", ""))
        if event.get("event") == "acquisition_completed":
            source = _native_pair(event.get("csv"))
            acquisitions.setdefault(node_id, []).append(source)
            continue
        if event.get("event") == "hypothesis_adjudicated":
            verifications.append(_verify_hypothesis_event(index, event))
            hypothesis_nodes.add(node_id)
            acquisitions[node_id] = []
            continue
        if event.get("event") != "fit_evaluated":
            continue
        if node_id in hypothesis_nodes:
            # The immediately preceding hypothesis event already replayed the
            # identity verdict and every native probe response. This fit event
            # is only the existing proposal/audit compatibility envelope.
            continue

        recorded = _recorded_gate_passes(event)
        event_source = _native_pair(event.get("csv"))
        paths = list(acquisitions.get(node_id, ()))
        if node_id == "N3":
            paths = [event_source]
        elif not paths or paths[-1] != event_source:
            paths.append(event_source)
        passed, replayed_gates = _evaluate(
            session,
            node_id,
            paths,
            recorded,
        )
        replayed_decision = "accept" if passed else "retake"
        recorded_decision = str(event.get("decision"))
        mismatched_gates = {
            name: (recorded[name], bool(replayed_gates[name]))
            for name in set(recorded).intersection(replayed_gates)
            if recorded[name] != bool(replayed_gates[name])
        }
        missing_recorded_gates = sorted(
            set(replayed_gates).difference(recorded)
        )
        missing_replayed_gates = sorted(
            set(recorded).difference(replayed_gates)
        )
        if (
            replayed_decision != recorded_decision
            or mismatched_gates
            or missing_recorded_gates
            or missing_replayed_gates
        ):
            raise AnalysisError(
                f"replay diverged at event {index} ({node_id}): "
                f"decision {recorded_decision!r} -> "
                f"{replayed_decision!r}, gate changes={mismatched_gates}, "
                f"missing recorded gates={missing_recorded_gates}, "
                f"missing replayed gates={missing_replayed_gates}"
            )
        verifications.append(
            ReplayVerification(
                event_index=index,
                node_id=node_id,
                csv_paths=tuple(paths),
                recorded_decision=recorded_decision,
                replayed_decision=replayed_decision,
                gate_passes={
                    name: bool(value)
                    for name, value in replayed_gates.items()
                },
            )
        )
        acquisitions[node_id] = []
    return tuple(verifications)
