"""Measure adaptive resonator-map row use and lookup accuracy on the zoo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from quickexp_v3.autocal.hp.adaptive import AdaptiveRowScheduler
from quickexp_v3.resonator_flux import cosine_frequency, fit_cosine
from quickexp_v3.zoo import ZooChip, generate_zoo


@dataclass(frozen=True)
class AdaptiveLookupResult:
    chip_id: str
    defect_class: str
    adaptive_rows: int
    fixed_rows: int
    adaptive_rmse_mhz: float
    fixed_rmse_mhz: float

    @property
    def row_fraction(self) -> float:
        return float(self.adaptive_rows / max(self.fixed_rows, 1))


def _true_centers(chip: ZooChip, z_gain: Sequence[float]) -> np.ndarray:
    return np.asarray(
        chip.device.resonator_frequency(-35.0, np.asarray(z_gain, dtype=float)),
        dtype=float,
    )


def _lookup_rmse(chip: ZooChip, z_gain: np.ndarray) -> float:
    centers = _true_centers(chip, z_gain)
    parameters, _fitted, _statistics = fit_cosine(
        np.asarray(z_gain, dtype=float),
        centers,
        frequency_step_mhz=0.06,
        period_min=0.12,
        period_max=0.30,
    )
    dense_z = np.linspace(-0.30, 0.30, 1001)
    predicted = np.asarray(
        cosine_frequency(
            dense_z,
            center_frequency=float(parameters[0]),
            amplitude=float(parameters[1]),
            period=float(parameters[2]),
            peak_bias=float(parameters[3]),
        ),
        dtype=float,
    )
    truth = _true_centers(chip, dense_z)
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def evaluate_chip(chip: ZooChip) -> AdaptiveLookupResult:
    scheduler = AdaptiveRowScheduler(
        (-0.30, 0.30),
        initial_rows=5,
        max_rows=7,
        abort_after_rows=5,
    )
    while not scheduler.done:
        z_gain = scheduler.next_row()
        center = float(_true_centers(chip, (z_gain,))[0])
        scheduler.record(
            z_gain,
            center_mhz=center,
            trackable=True,
            uncertainty_mhz=0.0,
        )
    adaptive_z = np.asarray([row.value for row in scheduler.rows], dtype=float)
    fixed_z = np.linspace(-0.30, 0.30, 13)
    return AdaptiveLookupResult(
        chip.chip_id,
        chip.defect_class,
        len(adaptive_z),
        len(fixed_z),
        _lookup_rmse(chip, adaptive_z),
        _lookup_rmse(chip, fixed_z),
    )


def summarize(results: Sequence[AdaptiveLookupResult]) -> Dict[str, float]:
    adaptive = np.asarray(
        [result.adaptive_rmse_mhz for result in results],
        dtype=float,
    )
    fixed = np.asarray(
        [result.fixed_rmse_mhz for result in results],
        dtype=float,
    )
    fractions = np.asarray(
        [result.row_fraction for result in results],
        dtype=float,
    )
    return {
        "count": float(len(results)),
        "median_adaptive_rmse_mhz": float(np.median(adaptive)),
        "median_fixed_rmse_mhz": float(np.median(fixed)),
        "maximum_row_fraction": float(np.max(fractions)),
        "noninferior_fraction": float(
            np.mean(adaptive <= fixed + 1.0e-6)
        ),
    }


def run_adaptive_zoo(
    count: int = 210,
    seed: int = 0,
) -> Tuple[Dict[str, float], Tuple[AdaptiveLookupResult, ...]]:
    chips = generate_zoo(int(count), seed=int(seed))
    results = tuple(evaluate_chip(chip) for chip in chips)
    return summarize(results), results


def main() -> None:
    metrics, _results = run_adaptive_zoo()
    for name in sorted(metrics):
        print("{0}={1:.9g}".format(name, metrics[name]))


if __name__ == "__main__":
    main()
