"""Score p1 candidates and coverage on the legacy baseline zoo."""

from __future__ import annotations

import tempfile
from pathlib import Path

from quickexp_v3.autocal.budget import BudgetModel
from quickexp_v3.autocal.hp.candidates import extract_candidates
from quickexp_v3.autocal.hp.coverage import assess_coverage
from quickexp_v3.autocal.nodes import classify_failure
from quickexp_v3.notch_fit import fit_spectroscopy_features
from quickexp_v3.zoo import ZooChip

from .baseline_legacy import _qubit_plan, acquire_qubit_trace
from .zoo_metrics import DecisionResult, format_report, run_zoo


def decide_hp(chip: ZooChip) -> DecisionResult:
    """Apply Phase 1 classification without any Phase 2 identity probes."""
    with tempfile.TemporaryDirectory() as scratch:
        csv_path = acquire_qubit_trace(chip, Path(scratch))
        fit = fit_spectroscopy_features(
            csv_path,
            kind="qubit",
            signal="amplitude",
        )
        candidates = extract_candidates(fit)
        low, high = chip.prior["q_freq_mhz"]
        assessment = assess_coverage(
            candidates=candidates,
            prior_window=(float(low), float(high)),
            scan_window=(float(fit.x[0]), float(fit.x[-1])),
            points=int(fit.x.size),
            expected_fwhm_mhz=float(
                fit.parameters.get("fwhm_mhz", 1.0)
            ),
            expected_contrast=abs(
                float(fit.parameters.get("amplitude", 0.0))
            ),
        )
        classification = classify_failure(candidates, assessment)
        if classification["failure_class"] in {"A", "B"}:
            return DecisionResult(
                chip.chip_id,
                chip.defect_class,
                "escalate",
                None,
            )
        real = [item for item in candidates if not item.is_null]
        if not real:
            return DecisionResult(
                chip.chip_id,
                chip.defect_class,
                "escalate",
                None,
            )
        return DecisionResult(
            chip.chip_id,
            chip.defect_class,
            "accept",
            float(real[0].center_mhz),
            simulated_seconds=BudgetModel().estimate(
                _qubit_plan(chip, 0.1)
            ),
        )


def main() -> None:
    metrics, _results = run_zoo(
        decide_hp,
        count=210,
        seed=0,
        tolerance_mhz=1.0,
    )
    print(format_report(metrics))


if __name__ == "__main__":
    main()
