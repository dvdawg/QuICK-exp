"""Score and calibrate the hypothesis-and-probe N5 path on the device zoo."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np
import yaml

from quickexp_v3.autocal.budget import BudgetModel, BudgetTracker
from quickexp_v3.autocal.graph import NODE_REGISTRY
from quickexp_v3.autocal.nodes import SessionContext, run_node
from quickexp_v3.autocal.policy import load_autocal_policy
from quickexp_v3.autocal.session import AutocalSession
from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.ide import load_repository
from quickexp_v3.zoo import ZooChip, generate_zoo

from .zoo_metrics import DecisionResult, format_report, run_zoo


ROOT = Path(__file__).resolve().parents[1]


def _project(destination: Path, chip: ZooChip, margin_threshold: float) -> None:
    for name in (
        "hardware.example.yml",
        "calibration.example.yml",
        "presets.example.yml",
    ):
        shutil.copy2(ROOT / name, destination / name)
    hardware_path = destination / "hardware.example.yml"
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    low, high = chip.prior["q_freq_mhz"]
    hardware["expected"]["q_freq_mhz"] = [float(low), float(high)]
    hardware["expected"]["flux_period_z"] = float(
        chip.device.qubit_flux_period_z
    )
    hardware["expected"]["resonator_linewidth_mhz"] = float(
        chip.device.resonator_linewidth_mhz
    )
    hardware["expected"]["dispersive_shift_mhz"] = float(
        chip.device.dispersive_shift_mhz
    )
    hardware["defaults"]["q_freq"] = 0.5 * (float(low) + float(high))
    hardware["defaults"]["r_freq"] = float(chip.truth["r_freq_mhz"])
    hardware["defaults"]["q_delta"] = -abs(float(chip.device.ec_mhz))
    hardware["autocal"]["hypothesis_nodes"] = ["N5"]
    hardware["autocal"]["hypothesis"]["margin_threshold"] = float(
        margin_threshold
    )
    hardware["autocal"]["advisor"]["mode"] = "null"
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False),
        encoding="utf-8",
    )


def _predicted_seconds(session: AutocalSession) -> float:
    return float(
        sum(
            max(float(event.get("predicted_seconds", 0.0)), 0.0)
            for event in session.events()
            if event.get("event") == "acquisition_completed"
        )
    )


def _ledger_leader_value(session: AutocalSession) -> Optional[float]:
    try:
        ranking = session.state["hypothesis_ledger"]["addresses"][
            "defaults.q_freq"
        ]["ranking"]
        return float(ranking[0]["center_mhz"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def decide_hypothesis(
    chip: ZooChip,
    *,
    margin_threshold: float = 2.0,
) -> DecisionResult:
    """Run the real N5 implementation through native CSV/YML materialization."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        _project(root, chip, float(margin_threshold))
        repository = load_repository(root)
        policy = load_autocal_policy(repository.hardware)
        session = AutocalSession.create_or_resume(
            root,
            target="zoo_n5",
            autonomy_level=0,
            z_gain=0.0,
            node_ids=("N5",),
            calibration_revision=int(repository.calibration.get("revision", 0)),
            session_name="zoo-" + chip.chip_id,
        )
        session.set_working_values(
            {
                "defaults.r_freq": float(chip.truth["r_freq_mhz"]),
                "defaults.q_gain": float(
                    repository.hardware["defaults"].get("q_gain", 0.4)
                ),
            }
        )
        budget = BudgetTracker.from_state(
            session.state,
            max_wall_clock_hours=policy.max_wall_clock_hours,
            max_total_runs=policy.max_total_runs,
        )
        context = SessionContext(
            project_root=root,
            session=session,
            policy=policy,
            budget=budget,
            budget_model=BudgetModel(),
            autonomy_level=0,
            z_gain=0.0,
            backend=SyntheticBackend(
                seed=int(chip.seed) % 100_000,
                device=chip.device,
            ),
        )
        outcome = None
        # Acquisition helpers print launcher diagnostics. The metrics table is
        # intentionally the only CLI output from this headless runner.
        with redirect_stdout(io.StringIO()):
            for attempt in range(1, policy.max_node_attempts + 1):
                outcome = run_node(context, NODE_REGISTRY["N5"], attempt=attempt)
                if outcome.status != "retake":
                    break
        if outcome is None:
            raise RuntimeError("N5 produced no outcome")
        classification = outcome.classification or {}
        accepted = outcome.status == "done"
        value = outcome.values.get("defaults.q_freq") if accepted else None
        if value is None:
            value = _ledger_leader_value(session)
        margin = classification.get("hypothesis_margin")
        wrong_would_propagate = bool(
            accepted
            and value is not None
            and abs(float(value) - float(chip.truth["q_freq_mhz"])) > 1.0
        )
        return DecisionResult(
            chip.chip_id,
            chip.defect_class,
            "accept" if accepted else "escalate",
            None if value is None else float(value),
            simulated_seconds=_predicted_seconds(session),
            wrong_value_propagated=wrong_would_propagate,
            hypothesis_margin=(
                None if margin is None else float(margin)
            ),
            hypothesis_id=(
                None
                if classification.get("hypothesis") is None
                else str(classification["hypothesis"])
            ),
        )


def calibrate_margin_threshold(
    results: Sequence[DecisionResult],
    chips: Sequence[ZooChip],
    *,
    tolerance_mhz: float = 1.0,
    target_false_accept_rate: float = 0.0,
    minimum_validated_threshold: float = 0.0,
) -> float:
    """Choose the least restrictive *validated* margin meeting the FA target.

    Results from a run at threshold ``T`` cannot validate a lower threshold:
    lowering it can stop the probe battery earlier and therefore change the
    evidence and identity verdict.  Callers must pass that run's ``T`` as the
    validation floor.  Higher cutoffs can still be screened conservatively
    from the recorded final margins.
    """
    chip_by_id = {chip.chip_id: chip for chip in chips}
    validation_floor = max(float(minimum_validated_threshold), 0.0)
    finite_margins = sorted(
        {
            max(float(result.hypothesis_margin), 0.0)
            for result in results
            if result.hypothesis_margin is not None
            and np.isfinite(float(result.hypothesis_margin))
            and float(result.hypothesis_margin) >= validation_floor
        }
    )
    candidates = [validation_floor]
    candidates.extend(
        margin
        for margin in finite_margins
        if margin > validation_floor
    )
    if finite_margins:
        candidates.append(float(np.nextafter(finite_margins[-1], np.inf)))
    count = max(len(results), 1)
    for threshold in candidates:
        false_accepts = 0
        for result in results:
            chip = chip_by_id[result.chip_id]
            accepts = bool(
                result.hypothesis_id == "qubit_01"
                and result.hypothesis_margin is not None
                and float(result.hypothesis_margin) >= threshold
                and result.value is not None
            )
            if accepts and abs(
                float(result.value) - float(chip.truth["q_freq_mhz"])
            ) > float(tolerance_mhz):
                false_accepts += 1
        if false_accepts / float(count) <= float(target_false_accept_rate):
            return float(threshold)
    return float("inf")


def run_hypothesis_zoo(
    *,
    count: int = 210,
    seed: int = 0,
    margin_threshold: float = 2.0,
    tolerance_mhz: float = 1.0,
) -> Tuple[dict, Tuple[DecisionResult, ...]]:
    return run_zoo(
        lambda chip: decide_hypothesis(
            chip,
            margin_threshold=float(margin_threshold),
        ),
        count=int(count),
        seed=int(seed),
        tolerance_mhz=float(tolerance_mhz),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=210)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin-threshold", type=float, default=2.0)
    parser.add_argument("--target-false-accept-rate", type=float, default=0.0)
    arguments = parser.parse_args()
    metrics, results = run_hypothesis_zoo(
        count=arguments.count,
        seed=arguments.seed,
        margin_threshold=arguments.margin_threshold,
    )
    print(format_report(metrics))
    chips = generate_zoo(arguments.count, seed=arguments.seed)
    suggested = calibrate_margin_threshold(
        results,
        chips,
        target_false_accept_rate=arguments.target_false_accept_rate,
        minimum_validated_threshold=arguments.margin_threshold,
    )
    print("suggested_margin_threshold={0:.6g}".format(suggested))


if __name__ == "__main__":
    main()
