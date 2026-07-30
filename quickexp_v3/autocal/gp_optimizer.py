"""Small pure-numpy exact GP optimizer for readout-fidelity searches."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..errors import ConfigError


@dataclass(frozen=True)
class OptimizationResult:
    x: np.ndarray
    y: float
    x_history: np.ndarray
    y_history: np.ndarray


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))


def _normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * values**2) / sqrt(2.0 * pi)


def _latin_hypercube(count: int, dimensions: int, rng: Any) -> np.ndarray:
    result = np.empty((count, dimensions), dtype=float)
    for dimension in range(dimensions):
        result[:, dimension] = (
            rng.permutation(count) + rng.random(count)
        ) / count
    return result


def _rbf(left: np.ndarray, right: np.ndarray, scales: np.ndarray) -> np.ndarray:
    differences = (
        left[:, None, :] - right[None, :, :]
    ) / scales[None, None, :]
    return np.exp(-0.5 * np.sum(differences**2, axis=2))


def bayesian_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: Sequence[Sequence[float]],
    *,
    length_scales: Optional[Sequence[float]] = None,
    max_evaluations: int = 30,
    initial_points: int = 8,
    candidate_points: int = 2048,
    observation_noise: float = 1.0e-4,
    seed: int = 0,
) -> OptimizationResult:
    """Maximize a bounded objective with RBF-ARD GP expected improvement."""
    limits = np.asarray(bounds, dtype=float)
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ConfigError("GP bounds must be shaped (dimensions, 2)")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ConfigError("GP bounds must be finite and strictly ordered")
    evaluations = int(max_evaluations)
    initial = int(initial_points)
    if evaluations < 1 or not 1 <= initial <= evaluations:
        raise ConfigError("GP evaluation counts must satisfy 1 <= initial <= maximum")
    dimensions = limits.shape[0]
    scales = np.asarray(
        length_scales if length_scales is not None else np.full(dimensions, 0.2),
        dtype=float,
    )
    if scales.shape != (dimensions,) or np.any(scales <= 0):
        raise ConfigError("GP length scales must be one positive value per dimension")

    width = limits[:, 1] - limits[:, 0]
    normalized_scales = scales / width
    normalized_scales = np.clip(normalized_scales, 1.0e-3, 10.0)
    rng = np.random.default_rng(int(seed))
    x_normalized = _latin_hypercube(initial, dimensions, rng)
    y_values = [
        float(objective(limits[:, 0] + point * width))
        for point in x_normalized
    ]

    while len(y_values) < evaluations:
        training = np.asarray(x_normalized, dtype=float)
        observations = np.asarray(y_values, dtype=float)
        y_mean = float(np.mean(observations))
        y_scale = max(float(np.std(observations)), 1.0e-6)
        normalized_y = (observations - y_mean) / y_scale
        covariance = _rbf(training, training, normalized_scales)
        covariance.flat[:: covariance.shape[0] + 1] += max(
            float(observation_noise),
            1.0e-10,
        )
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            covariance.flat[:: covariance.shape[0] + 1] += 1.0e-6
            cholesky = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, normalized_y),
        )

        candidates = rng.random((max(int(candidate_points), 64), dimensions))
        candidates = np.vstack(
            (
                candidates,
                np.clip(
                    training[int(np.argmax(observations))]
                    + rng.normal(0.0, normalized_scales / 2.0, (256, dimensions)),
                    0.0,
                    1.0,
                ),
            )
        )
        cross = _rbf(candidates, training, normalized_scales)
        mean = cross @ alpha
        solved = np.linalg.solve(cholesky, cross.T)
        variance = np.maximum(1.0 - np.sum(solved**2, axis=0), 1.0e-12)
        deviation = np.sqrt(variance)
        improvement = mean - float(np.max(normalized_y)) - 0.01
        score = improvement / deviation
        expected_improvement = (
            improvement * _normal_cdf(score)
            + deviation * _normal_pdf(score)
        )
        point = candidates[int(np.argmax(expected_improvement))]
        x_normalized = np.vstack((x_normalized, point))
        y_values.append(float(objective(limits[:, 0] + point * width)))

    physical = limits[:, 0] + np.asarray(x_normalized) * width
    values = np.asarray(y_values, dtype=float)
    best = int(np.nanargmax(values))
    return OptimizationResult(
        x=physical[best].copy(),
        y=float(values[best]),
        x_history=physical,
        y_history=values,
    )
