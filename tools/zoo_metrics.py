"""Headless zoo runner and calibration-decision quality metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from quickexp_v3.zoo import ZooChip, generate_zoo


@dataclass(frozen=True)
class DecisionResult:
    chip_id: str
    defect_class: str
    verdict: str
    value: Optional[float]
    simulated_seconds: float = 0.0
    wrong_value_propagated: bool = False
    hypothesis_margin: Optional[float] = None
    hypothesis_id: Optional[str] = None


def _blank() -> dict:
    return {
        "count": 0.0,
        "false_accepts": 0.0,
        "false_rejects": 0.0,
        "escalations": 0.0,
        "errors": 0.0,
        "wrong_value_propagations": 0.0,
        "calibration_times": [],
    }


def score_results(
    results: Sequence[DecisionResult],
    chips: Sequence[ZooChip],
    tolerance_mhz: float,
) -> Dict[str, Dict[str, float]]:
    """Return all five decision metrics per defect class and overall.

    A false accept is an accepted value farther than ``tolerance_mhz`` from
    truth. An escalation is a false reject only when the prior included the
    correct answer. Time-to-calibration is reported for accepted runs; the
    wrong-value propagation metric is explicit because later phases can run a
    downstream subtree while this Phase-0 harness cannot infer that event.
    """
    tolerance = float(tolerance_mhz)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_mhz must be finite and non-negative")

    truth_by_id = {chip.chip_id: chip for chip in chips}
    if len(truth_by_id) != len(chips):
        raise ValueError("chip ids must be unique")
    seen = set()
    buckets = {}
    for result in results:
        if result.chip_id in seen:
            raise ValueError("duplicate result for chip " + repr(result.chip_id))
        seen.add(result.chip_id)
        chip = truth_by_id.get(result.chip_id)
        if chip is None:
            raise KeyError("result references unknown chip " + repr(result.chip_id))
        if result.defect_class != chip.defect_class:
            raise ValueError(
                "result defect class does not match chip " + repr(result.chip_id)
            )

        keys = (chip.defect_class, "overall")
        for key in keys:
            buckets.setdefault(key, _blank())
            buckets[key]["count"] += 1.0

        truth = float(chip.truth["q_freq_mhz"])
        low, high = chip.prior["q_freq_mhz"]
        answer_was_available = bool(float(low) <= truth <= float(high))
        if result.verdict == "accept":
            wrong = result.value is None or abs(float(result.value) - truth) > tolerance
            for key in keys:
                buckets[key]["calibration_times"].append(
                    max(float(result.simulated_seconds), 0.0)
                )
            if wrong:
                for key in keys:
                    buckets[key]["false_accepts"] += 1.0
                    if result.wrong_value_propagated:
                        buckets[key]["wrong_value_propagations"] += 1.0
        elif result.verdict == "escalate":
            for key in keys:
                buckets[key]["escalations"] += 1.0
                if answer_was_available:
                    buckets[key]["false_rejects"] += 1.0
        elif result.verdict == "error":
            for key in keys:
                buckets[key]["errors"] += 1.0
        else:
            raise ValueError("unknown decision verdict " + repr(result.verdict))

    report = {}
    for key, bucket in buckets.items():
        count = max(float(bucket["count"]), 1.0)
        times = np.asarray(bucket["calibration_times"], dtype=float)
        median_time = float(np.median(times)) if times.size else float("nan")
        report[key] = {
            "count": float(bucket["count"]),
            "false_accept_rate": float(bucket["false_accepts"]) / count,
            "false_reject_rate": float(bucket["false_rejects"]) / count,
            "wrong_value_propagation_rate": (
                float(bucket["wrong_value_propagations"]) / count
            ),
            "median_time_to_calibration_seconds": median_time,
            "escalation_rate": float(bucket["escalations"]) / count,
            "error_rate": float(bucket["errors"]) / count,
        }
    return report


def run_zoo(
    decide: Callable[[ZooChip], DecisionResult],
    count: int = 200,
    seed: int = 0,
    tolerance_mhz: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], Tuple[DecisionResult, ...]]:
    """Run ``decide`` over a generated zoo and return metrics and raw results."""
    chips = generate_zoo(int(count), seed=int(seed))
    results = []
    for chip in chips:
        try:
            result = decide(chip)
            if not isinstance(result, DecisionResult):
                raise TypeError("decide must return DecisionResult")
            results.append(result)
        except Exception:
            results.append(
                DecisionResult(chip.chip_id, chip.defect_class, "error", None)
            )
    return score_results(results, chips, tolerance_mhz), tuple(results)


def format_report(metrics: Dict[str, Dict[str, float]]) -> str:
    """Render a deterministic fixed-width table, with the overall row last."""
    keys = sorted(key for key in metrics if key != "overall")
    if "overall" in metrics:
        keys.append("overall")
    lines = [
        (
            "{0:<18} {1:>6} {2:>10} {3:>10} {4:>10} "
            "{5:>10} {6:>10} {7:>8}"
        ).format(
            "class",
            "n",
            "false_acc",
            "false_rej",
            "propagate",
            "median_s",
            "escalate",
            "error",
        )
    ]
    for key in keys:
        row = metrics[key]
        lines.append(
            (
                "{0:<18} {1:>6.0f} {2:>10.3f} {3:>10.3f} {4:>10.3f} "
                "{5:>10.3f} {6:>10.3f} {7:>8.3f}"
            ).format(
                key,
                row["count"],
                row["false_accept_rate"],
                row["false_reject_rate"],
                row["wrong_value_propagation_rate"],
                row["median_time_to_calibration_seconds"],
                row["escalation_rate"],
                row["error_rate"],
            )
        )
    return "\n".join(lines)
