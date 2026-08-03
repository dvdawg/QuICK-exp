"""Score the legacy threshold-gate path on the adversarial device zoo."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.autocal.budget import BudgetModel
from quickexp_v3.experiments.base import ExperimentPlan
from quickexp_v3.notch_fit import fit_spectroscopy_features
from quickexp_v3.synthetic_device import write_native_pair
from quickexp_v3.zoo import ZooChip

from .zoo_metrics import DecisionResult, format_report, run_zoo


LEGACY_MINIMUM_R_SQUARED = 0.95
LEGACY_MINIMUM_CONTRAST_SNR = 8.0
LEGACY_MAXIMUM_CENTER_UNCERTAINTY_FRACTION = 0.25
LEGACY_Q_DELTA_MHZ = -180.0
TRACE_POINTS = 1201


def _qubit_plan(chip: ZooChip, q_gain: float) -> ExperimentPlan:
    low, high = chip.prior["q_freq_mhz"]
    frequency = np.linspace(float(low), float(high), TRACE_POINTS)
    return ExperimentPlan(
        name="qubit_spectroscopy",
        quick_class="QubitSpectroscopy",
        title="zoo_" + chip.chip_id,
        variables={
            "q_freq": frequency,
            "q_gain": float(q_gain),
            "q_length": 10.0,
            "r_length": 2.0,
            "r_relax": 20.0,
            "hard_avg": 5000,
            "soft_avg": 3,
            "z_gain": 0.0,
        },
        axes=("q_freq",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"q_freq": "MHz"},
    )


def acquire_qubit_trace(
    chip: ZooChip,
    destination: Path,
    q_gain: float = 0.1,
) -> Path:
    """Acquire one native qubit trace over a chip's declared prior window."""
    plan = _qubit_plan(chip, q_gain)
    backend = SyntheticBackend(
        seed=int(chip.seed) % 100_000,
        device=chip.device,
    )
    result = backend.acquire(plan)
    return write_native_pair(destination, plan, result, title=plan.title)


def decide_legacy(chip: ZooChip) -> DecisionResult:
    """Apply the existing N5-style coarse-to-fine threshold gates."""
    with tempfile.TemporaryDirectory() as scratch:
        csv_path = acquire_qubit_trace(chip, Path(scratch))
        coarse_fit = fit_spectroscopy_features(
            csv_path,
            kind="qubit",
            signal="amplitude",
        )
        if coarse_fit.statistics.get("multi_feature"):
            features = list(coarse_fit.parameters.get("features", ()))
            recognized_shadow = False
            if len(features) == 2:
                separation = abs(
                    float(features[0]["center_mhz"])
                    - float(features[1]["center_mhz"])
                )
                expected = abs(LEGACY_Q_DELTA_MHZ) / 2.0
                recognized_shadow = abs(separation - expected) <= max(
                    10.0,
                    0.20 * expected,
                )
            if not recognized_shadow:
                return DecisionResult(
                    chip.chip_id,
                    chip.defect_class,
                    "escalate",
                    None,
                )
        elif not coarse_fit.passes(
            minimum_r_squared=0.50,
            minimum_contrast_snr=3.0,
            maximum_center_uncertainty_fraction_of_fwhm=0.30,
        ):
            return DecisionResult(
                chip.chip_id,
                chip.defect_class,
                "escalate",
                None,
            )

        half_window = max(
            10.0,
            5.0 * float(coarse_fit.parameters["fwhm_mhz"]),
        )
        fit = fit_spectroscopy_features(
            csv_path,
            kind="qubit",
            signal="amplitude",
            window_mhz=(
                float(coarse_fit.center_mhz) - half_window,
                float(coarse_fit.center_mhz) + half_window,
            ),
        )
        passed = fit.passes(
            minimum_r_squared=LEGACY_MINIMUM_R_SQUARED,
            minimum_contrast_snr=LEGACY_MINIMUM_CONTRAST_SNR,
            maximum_center_uncertainty_fraction_of_fwhm=(
                LEGACY_MAXIMUM_CENTER_UNCERTAINTY_FRACTION
            ),
        )
        if passed:
            return DecisionResult(
                chip.chip_id,
                chip.defect_class,
                "accept",
                float(fit.center_mhz),
                simulated_seconds=BudgetModel().estimate(
                    _qubit_plan(chip, 0.1)
                ),
            )
        return DecisionResult(chip.chip_id, chip.defect_class, "escalate", None)


def main() -> None:
    metrics, _results = run_zoo(
        decide_legacy,
        count=210,
        seed=0,
        tolerance_mhz=1.0,
    )
    print(format_report(metrics))


if __name__ == "__main__":
    main()
