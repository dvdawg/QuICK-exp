"""Read-only reconstruction of autocal fit decisions from native data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _evaluate_n2(paths: Sequence[Path]) -> tuple[bool, dict]:
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
    return fit.passes(
        minimum_plateau_rows=2,
        minimum_shift_over_step=2.0,
        maximum_transition_width_db=15.0,
    ), gates


def _evaluate_n3(paths: Sequence[Path]) -> tuple[bool, dict]:
    fit = fit_resonator_flux(
        paths[-1],
        period_min=0.12,
        period_max=0.30,
    )
    gates = {
        "r_squared": fit.statistics["r_squared"] >= 0.95,
        "rmse_mhz": fit.statistics["rmse_mhz"] <= 0.2,
        "complete_z_rows": len(fit.z_gain) >= 6,
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
        "N2": _evaluate_n2,
        "N3": _evaluate_n3,
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


def verify_session_replay(session: AutocalSession) -> tuple:
    """Re-fit every logged decision and fail if any verdict changes."""
    acquisitions: dict[str, list[Path]] = {}
    verifications = []
    for index, event in enumerate(session.events()):
        node_id = str(event.get("node", ""))
        if event.get("event") == "acquisition_completed":
            source = _native_pair(event.get("csv"))
            acquisitions.setdefault(node_id, []).append(source)
            continue
        if event.get("event") != "fit_evaluated":
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
