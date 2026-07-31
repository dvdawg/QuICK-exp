"""Numpy/SciPy IQ-shot discrimination and readout-frequency scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp
from scipy.stats import norm
import yaml

from .errors import AnalysisError
from .fit_calibration import annotate_forced_write, write_calibration_records
from .util import utc_now


def _log_gaussian(data: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    regularized = covariance + np.eye(2) * 1e-9
    sign, logdet = np.linalg.slogdet(regularized)
    if sign <= 0:
        regularized = regularized + np.eye(2) * 1e-6
        _, logdet = np.linalg.slogdet(regularized)
    inverse = np.linalg.pinv(regularized)
    centered = data - mean
    quadratic = np.einsum("ni,ij,nj->n", centered, inverse, centered)
    return -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + quadratic)


def _em(
    data: np.ndarray,
    means: np.ndarray,
    *,
    max_iterations: int,
) -> tuple:
    weights = np.asarray([0.5, 0.5], dtype=float)
    shared = np.cov(data.T) + np.eye(2) * 1e-6
    covariances = np.stack((shared.copy(), shared.copy()))
    previous = -np.inf
    for _ in range(int(max_iterations)):
        log_probability = np.column_stack(
            [
                np.log(max(weights[index], np.finfo(float).tiny))
                + _log_gaussian(data, means[index], covariances[index])
                for index in range(2)
            ]
        )
        normalization = logsumexp(log_probability, axis=1)
        responsibilities = np.exp(log_probability - normalization[:, None])
        counts = np.sum(responsibilities, axis=0)
        weights = counts / data.shape[0]
        means = (responsibilities.T @ data) / counts[:, None]
        for index in range(2):
            centered = data - means[index]
            covariances[index] = (
                (responsibilities[:, index, None] * centered).T @ centered
                / counts[index]
                + np.eye(2) * 1e-8
            )
        likelihood = float(np.sum(normalization))
        if abs(likelihood - previous) <= 1e-8 * max(abs(likelihood), 1.0):
            break
        previous = likelihood
    return means, covariances, weights


def _posterior(
    data: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    logs = np.column_stack(
        [
            np.log(max(weights[index], np.finfo(float).tiny))
            + _log_gaussian(data, means[index], covariances[index])
            for index in range(2)
        ]
    )
    return np.exp(logs - logsumexp(logs, axis=1)[:, None])


def _threshold(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    direction = means[1] - means[0]
    separation = float(np.linalg.norm(direction))
    if separation <= 0:
        raise AnalysisError("IQ mixture components have identical means")
    direction = direction / separation
    projected_means = means @ direction
    variances = np.asarray(
        [direction @ covariance @ direction for covariance in covariances]
    )
    variances = np.maximum(variances, np.finfo(float).eps)
    m0, m1 = projected_means
    v0, v1 = variances
    a = 0.5 / v1 - 0.5 / v0
    b = m0 / v0 - m1 / v1
    c = (
        0.5 * m1**2 / v1
        - 0.5 * m0**2 / v0
        + np.log(
            max(weights[0], np.finfo(float).tiny) * np.sqrt(v1)
            / (max(weights[1], np.finfo(float).tiny) * np.sqrt(v0))
        )
    )
    if abs(a) < 1e-12:
        roots = [-c / b] if abs(b) > 1e-12 else []
    else:
        roots = np.roots([a, b, c])
    between = [
        float(np.real(root))
        for root in roots
        if abs(np.imag(root)) < 1e-8
        and min(m0, m1) <= np.real(root) <= max(m0, m1)
    ]
    threshold = (
        between[0]
        if between
        else float(0.5 * (m0 + m1))
    )
    angle = float(np.arctan2(direction[1], direction[0]))
    return direction, threshold, angle, projected_means, variances


def _threshold_fidelity(
    ground: np.ndarray,
    excited: np.ndarray,
    direction: np.ndarray,
    threshold: float,
    projected_means: np.ndarray,
) -> float:
    sign = 1.0 if projected_means[1] > projected_means[0] else -1.0
    ground_error = np.mean(sign * (ground @ direction - threshold) > 0)
    excited_error = np.mean(sign * (excited @ direction - threshold) <= 0)
    return float(1.0 - 0.5 * (ground_error + excited_error))


def _centroid_classifier(ground: np.ndarray, excited: np.ndarray) -> tuple:
    means = np.vstack((np.mean(ground, axis=0), np.mean(excited, axis=0)))
    direction = means[1] - means[0]
    direction = direction / max(float(np.linalg.norm(direction)), np.finfo(float).eps)
    projected = means @ direction
    threshold = float(np.mean(projected))
    fidelity = _threshold_fidelity(
        ground,
        excited,
        direction,
        threshold,
        projected,
    )
    return fidelity, direction, threshold


@dataclass(frozen=True)
class IqGmmFit:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    assignment_fidelity: float
    overlap_fidelity: float
    threshold: float
    angle_rad: float
    rotation_stability: float
    baseline_fidelity: float
    cross_validated_fidelity: float
    cross_validated_baseline_fidelity: float
    leakage: Mapping[str, float]
    thermal_population: float
    fidelity_uncertainty: float
    shots_per_state: int

    def passes(
        self,
        *,
        minimum_fidelity: float = 0.80,
        minimum_shots_per_state: int = 2000,
        maximum_angle_bootstrap_std: float = 0.2,
    ) -> bool:
        return bool(
            self.assignment_fidelity >= minimum_fidelity
            and self.shots_per_state >= minimum_shots_per_state
            and self.rotation_stability <= maximum_angle_bootstrap_std
            and self.cross_validated_fidelity
            >= self.cross_validated_baseline_fidelity - 0.001
        )


def _fit_once(
    ground: np.ndarray,
    excited: np.ndarray,
    max_iterations: int,
) -> tuple:
    initial_means = np.vstack((np.mean(ground, axis=0), np.mean(excited, axis=0)))
    data = np.vstack((ground, excited))
    means, covariances, weights = _em(
        data,
        initial_means,
        max_iterations=max_iterations,
    )
    ground_mean = np.mean(ground, axis=0)
    if np.linalg.norm(means[1] - ground_mean) < np.linalg.norm(means[0] - ground_mean):
        means = means[::-1]
        covariances = covariances[::-1]
        weights = weights[::-1]
    direction, threshold, angle, projected_means, variances = _threshold(
        means,
        covariances,
        weights,
    )
    return (
        means,
        covariances,
        weights,
        direction,
        threshold,
        angle,
        projected_means,
        variances,
    )


def fit_iq_gmm(
    i_ground: Any,
    q_ground: Any,
    i_excited: Any,
    q_excited: Any,
    *,
    max_iterations: int = 200,
    seed: int = 0,
) -> IqGmmFit:
    arrays = [
        np.asarray(value, dtype=float).ravel()
        for value in (i_ground, q_ground, i_excited, q_excited)
    ]
    if arrays[0].size != arrays[1].size or arrays[2].size != arrays[3].size:
        raise AnalysisError("I/Q shot columns must have matched lengths per state")
    ground = np.column_stack(arrays[:2])
    excited = np.column_stack(arrays[2:])
    ground = ground[np.all(np.isfinite(ground), axis=1)]
    excited = excited[np.all(np.isfinite(excited), axis=1)]
    if min(len(ground), len(excited)) < 20:
        raise AnalysisError("IQ GMM fitting requires at least 20 shots per state")
    (
        means,
        covariances,
        weights,
        direction,
        threshold,
        angle,
        projected_means,
        variances,
    ) = _fit_once(ground, excited, max_iterations)
    fidelity = _threshold_fidelity(
        ground,
        excited,
        direction,
        threshold,
        projected_means,
    )
    baseline, _baseline_direction, _baseline_threshold = _centroid_classifier(
        ground,
        excited,
    )
    sign = 1.0 if projected_means[1] > projected_means[0] else -1.0
    ground_error = (
        1.0 - norm.cdf(sign * (threshold - projected_means[0]) / np.sqrt(variances[0]))
    )
    excited_error = norm.cdf(
        sign * (threshold - projected_means[1]) / np.sqrt(variances[1])
    )
    overlap_fidelity = float(1.0 - 0.5 * (ground_error + excited_error))

    folds = 5
    cv = []
    baseline_cv = []
    for fold in range(folds):
        ground_test = np.arange(len(ground)) % folds == fold
        excited_test = np.arange(len(excited)) % folds == fold
        trained = _fit_once(
            ground[~ground_test],
            excited[~excited_test],
            max_iterations,
        )
        cv.append(
            _threshold_fidelity(
                ground[ground_test],
                excited[excited_test],
                trained[3],
                trained[4],
                trained[6],
            )
        )
        base_train = _centroid_classifier(
            ground[~ground_test],
            excited[~excited_test],
        )
        projected_train = np.asarray(
            [
                np.mean(ground[~ground_test] @ base_train[1]),
                np.mean(excited[~excited_test] @ base_train[1]),
            ]
        )
        baseline_cv.append(
            _threshold_fidelity(
                ground[ground_test],
                excited[excited_test],
                base_train[1],
                base_train[2],
                projected_train,
            )
        )

    posterior_ground = _posterior(ground, means, covariances, weights)
    posterior_excited = _posterior(excited, means, covariances, weights)
    leakage = {
        "ground_as_excited_posterior_gt_0p9": float(
            np.mean(posterior_ground[:, 1] > 0.9)
        ),
        "excited_as_ground_posterior_gt_0p9": float(
            np.mean(posterior_excited[:, 0] > 0.9)
        ),
    }
    rng = np.random.default_rng(int(seed))
    angles = []
    for _ in range(50):
        ground_mean = np.mean(
            ground[rng.integers(0, len(ground), len(ground))],
            axis=0,
        )
        excited_mean = np.mean(
            excited[rng.integers(0, len(excited), len(excited))],
            axis=0,
        )
        delta = excited_mean - ground_mean
        angles.append(float(np.arctan2(delta[1], delta[0])))
    unwrapped = np.unwrap(np.asarray(angles))
    stability = float(np.std(unwrapped))
    effective_shots = min(len(ground), len(excited))
    fidelity_uncertainty = float(
        np.sqrt(max(fidelity * (1.0 - fidelity), 0.0) / (2.0 * effective_shots))
    )
    return IqGmmFit(
        means=means,
        covariances=covariances,
        weights=weights,
        assignment_fidelity=fidelity,
        overlap_fidelity=overlap_fidelity,
        threshold=float(threshold),
        angle_rad=angle,
        rotation_stability=stability,
        baseline_fidelity=baseline,
        cross_validated_fidelity=float(np.mean(cv)),
        cross_validated_baseline_fidelity=float(np.mean(baseline_cv)),
        leakage=leakage,
        thermal_population=float(np.mean(posterior_ground[:, 1])),
        fidelity_uncertainty=fidelity_uncertainty,
        shots_per_state=effective_shots,
    )


def load_iq_shots(csv_path: Path) -> tuple:
    source = Path(csv_path).expanduser().resolve()
    yml_path = source.with_suffix(".yml")
    metadata = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
    quick_class = (
        metadata.get("parameters", {}).get("quick_experiment")
        if isinstance(metadata, Mapping)
        else None
    )
    if quick_class != "IQScatter":
        raise AnalysisError(f"{source.name} is {quick_class!r}, not 'IQScatter'")
    matrix = np.atleast_2d(np.loadtxt(source, delimiter=","))
    if matrix.shape[1] < 4:
        raise AnalysisError("IQScatter CSV must contain four shot columns")
    return source, metadata, tuple(matrix[:, index] for index in range(4))


def plot_iq_gmm(
    fit: IqGmmFit,
    i_ground: Any,
    q_ground: Any,
    i_excited: Any,
    q_excited: Any,
):
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.scatter(i_ground, q_ground, s=4, alpha=0.25, label="ground")
    axis.scatter(i_excited, q_excited, s=4, alpha=0.25, label="excited")
    axis.scatter(fit.means[:, 0], fit.means[:, 1], marker="x", s=100, color="black")
    direction = np.asarray([np.cos(fit.angle_rad), np.sin(fit.angle_rad)])
    normal = np.asarray([-direction[1], direction[0]])
    point = fit.threshold * direction
    extent = max(np.ptp(np.r_[i_ground, i_excited]), np.ptp(np.r_[q_ground, q_excited]))
    line = point[None, :] + np.linspace(-extent, extent, 2)[:, None] * normal
    axis.plot(line[:, 0], line[:, 1], "k--", label="threshold")
    axis.set(xlabel="I", ylabel="Q", title=f"GMM fidelity {fit.assignment_fidelity:.3%}")
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    return figure


def iq_calibration_records(fit: IqGmmFit, source_csv: Path) -> dict:
    provenance = {
        "source": str(source_csv),
        "fitted_at": utc_now(),
        "analysis": "quickexp_v3.iq_gmm.fit_iq_gmm",
    }
    quality = {
        "assignment_fidelity": fit.assignment_fidelity,
        "overlap_fidelity": fit.overlap_fidelity,
        "baseline_fidelity": fit.baseline_fidelity,
        "cross_validated_fidelity": fit.cross_validated_fidelity,
        "cross_validated_baseline_fidelity": fit.cross_validated_baseline_fidelity,
        "rotation_stability": fit.rotation_stability,
        "leakage": dict(fit.leakage),
        "thermal_population": fit.thermal_population,
    }
    return {
        "defaults.r_threshold": {
            "value": fit.threshold,
            "unit": "a.u.",
            "uncertainty": {
                "fidelity": fit.fidelity_uncertainty,
                "angle_rad": fit.rotation_stability,
            },
            "provenance": provenance,
            "quality": quality,
            "model": "two_component_full_covariance_gmm",
            "status": "accepted",
            "accepted_at": utc_now(),
        },
        "derived.readout_fidelity": {
            "value": fit.assignment_fidelity,
            "unit": "fraction",
            "uncertainty": {"fidelity": fit.fidelity_uncertainty},
            "provenance": provenance,
            "quality": quality,
            "model": "labeled_assignment_fidelity",
            "status": "accepted",
            "accepted_at": utc_now(),
        },
    }


def accept_iq_gmm(
    project_root: Path,
    fit: IqGmmFit,
    source_csv: Path,
    *,
    minimum_fidelity: float = 0.80,
    minimum_shots_per_state: int = 2000,
    maximum_angle_bootstrap_std: float = 0.2,
    force_write: bool = False,
) -> Path:
    gates_passed = fit.passes(
        minimum_fidelity=minimum_fidelity,
        minimum_shots_per_state=minimum_shots_per_state,
        maximum_angle_bootstrap_std=maximum_angle_bootstrap_std,
    )
    if not gates_passed and not force_write:
        raise AnalysisError("IQ GMM fit did not pass acceptance gates")
    updates = annotate_forced_write(
        iq_calibration_records(fit, source_csv),
        force_write=force_write,
        gates_passed=gates_passed,
    )
    return write_calibration_records(project_root, updates)


@dataclass(frozen=True)
class ReadoutOptimizationFit:
    source_csv: Path
    frequency_mhz: np.ndarray
    separation: np.ndarray
    noise: np.ndarray
    snr: np.ndarray
    optimum_frequency_mhz: float
    snr_at_optimum: float
    optimum_offset_from_notch_mhz: Optional[float]
    metadata: Mapping[str, Any]


def fit_readout_optimization(
    csv_path: Path,
    *,
    notch_frequency_mhz: Optional[float] = None,
) -> ReadoutOptimizationFit:
    source = Path(csv_path).expanduser().resolve()
    metadata = yaml.safe_load(source.with_suffix(".yml").read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise AnalysisError("dispersive YML must contain a mapping")
    if metadata.get("parameters", {}).get("quick_experiment") != "DispersiveSpectroscopy":
        raise AnalysisError("readout optimization requires DispersiveSpectroscopy data")
    matrix = np.atleast_2d(np.loadtxt(source, delimiter=","))
    dependent = metadata.get("dependent", [])
    columns = {}
    for index, entry in enumerate(dependent):
        if isinstance(entry, (list, tuple)) and entry:
            columns[str(entry[0]).strip().lower().replace(" ", "_")] = index + 1
    try:
        i_ground = matrix[:, columns["i_0"]]
        q_ground = matrix[:, columns["q_0"]]
        i_excited = matrix[:, columns["i_1"]]
        q_excited = matrix[:, columns["q_1"]]
    except (KeyError, IndexError) as error:
        raise AnalysisError("dispersive CSV/YML is missing I/Q state columns") from error
    frequency = matrix[:, 0]
    ground = i_ground + 1j * q_ground
    excited = i_excited + 1j * q_excited
    separation = np.abs(excited - ground)
    smooth_ground = gaussian_filter1d(ground.real, 2.0) + 1j * gaussian_filter1d(ground.imag, 2.0)
    smooth_excited = gaussian_filter1d(excited.real, 2.0) + 1j * gaussian_filter1d(excited.imag, 2.0)
    noise = np.sqrt(
        np.abs(ground - smooth_ground) ** 2
        + np.abs(excited - smooth_excited) ** 2
    )
    noise_floor = max(float(np.median(noise)), np.finfo(float).eps)
    score = separation / np.maximum(noise, noise_floor)
    index = int(np.argmax(score))
    optimum = float(frequency[index])
    optimum_score = float(score[index])
    if 0 < index < frequency.size - 1:
        offsets = frequency[index - 1 : index + 2] - frequency[index]
        quadratic, linear, constant = np.polyfit(offsets, score[index - 1 : index + 2], 2)
        if quadratic < 0:
            correction = float(np.clip(-linear / (2.0 * quadratic), offsets[0], offsets[-1]))
            optimum += correction
            optimum_score = float(quadratic * correction**2 + linear * correction + constant)
    return ReadoutOptimizationFit(
        source_csv=source,
        frequency_mhz=frequency,
        separation=separation,
        noise=noise,
        snr=score,
        optimum_frequency_mhz=optimum,
        snr_at_optimum=optimum_score,
        optimum_offset_from_notch_mhz=(
            optimum - float(notch_frequency_mhz)
            if notch_frequency_mhz is not None
            else None
        ),
        metadata=metadata,
    )


def plot_readout_optimization(fit: ReadoutOptimizationFit):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(fit.frequency_mhz, fit.separation)
    axes[0].set(xlabel="Readout frequency (MHz)", ylabel="|S1-S0|", title="IQ separation")
    axes[1].plot(fit.frequency_mhz, fit.snr)
    axes[1].axvline(fit.optimum_frequency_mhz, color="tab:red", linestyle="--")
    axes[1].set(xlabel="Readout frequency (MHz)", ylabel="SNR", title="Noise-normalized score")
    for axis in axes:
        axis.grid(alpha=0.3)
    return figure
