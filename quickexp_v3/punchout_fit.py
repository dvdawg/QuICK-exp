"""Readout-resonator punchout analysis for power-by-frequency maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from .errors import AnalysisError
from .fit_calibration import write_calibration_records
from .fit_stats import pinned_parameters, r_squared
from .native_map import NativeMap, load_native_map
from .resonator_flux import extract_notch_centers
from .trace_qc import qc_map
from .util import utc_now


def punchout_model(
    power_db: Any,
    f_bare_mhz: float,
    shift_mhz: float,
    transition_power_db: float,
    transition_width_db: float,
) -> np.ndarray:
    power = np.asarray(power_db, dtype=float)
    width = max(abs(float(transition_width_db)), np.finfo(float).eps)
    return float(f_bare_mhz) + float(shift_mhz) * expit(
        (float(transition_power_db) - power) / width
    )


@dataclass(frozen=True)
class PunchoutFit:
    source_csv: Path
    status: str
    powers_db: np.ndarray
    frequencies_mhz: np.ndarray
    amplitude: np.ndarray
    centers_mhz: np.ndarray
    fitted_centers_mhz: Optional[np.ndarray]
    parameters: Mapping[str, Any]
    statistics: Mapping[str, Any]
    recommendation: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def passes(
        self,
        *,
        minimum_plateau_rows: int = 2,
        minimum_shift_over_step: float = 2.0,
        maximum_transition_width_db: float = 15.0,
    ) -> bool:
        return bool(
            self.status == "resolved"
            and self.statistics["low_plateau_rows"] >= minimum_plateau_rows
            and self.statistics["high_plateau_rows"] >= minimum_plateau_rows
            and self.statistics["shift_over_frequency_step"] >= minimum_shift_over_step
            and self.parameters["transition_width_db"] <= maximum_transition_width_db
            and not self.statistics.get("pinned_parameters")
        )


def _power_frequency_map(native: NativeMap) -> tuple:
    outer_name = native.outer_label.lower()
    inner_name = native.inner_label.lower()
    if "power" in outer_name and "freq" in inner_name:
        return native.outer, native.inner, native.signals["amplitude"]
    if "freq" in outer_name and "power" in inner_name:
        return native.inner, native.outer, native.signals["amplitude"].T
    raise AnalysisError("punchout map axes must be power and frequency")


def fit_punchout(
    csv_path: Path,
    *,
    smooth_sigma_bins: float = 1.5,
    prior_linewidth_mhz: Optional[float] = None,
) -> PunchoutFit:
    native = load_native_map(csv_path)
    powers, frequencies, amplitude = _power_frequency_map(native)
    order = np.argsort(powers)
    powers = np.asarray(powers[order], dtype=float)
    amplitude = np.asarray(amplitude[order], dtype=float)
    centers, depths, _ = extract_notch_centers(
        np.asarray(frequencies, dtype=float),
        amplitude,
        smooth_sigma_bins,
    )
    frequency_step = float(np.median(np.diff(frequencies)))
    observed_shift = float(
        np.mean(centers[:2]) - np.mean(centers[-2:])
    ) if centers.size >= 4 else float(np.ptp(centers))
    shift_over_step = abs(observed_shift) / frequency_step
    recommendation = {
        "frequency_step_mhz_max": float(
            prior_linewidth_mhz / 5.0
            if prior_linewidth_mhz is not None
            else frequency_step / 2.0
        ),
        "power_scan": (
            "Extend the power range until both frequency plateaus contain "
            "at least two rows; use 2.5 dB spacing through the transition."
        ),
    }
    base_statistics = {
        "observed_shift_mhz": observed_shift,
        "shift_over_frequency_step": float(shift_over_step),
        "frequency_step_mhz": frequency_step,
        "notch_depth_minimum": float(np.min(depths)),
        "qc": {
            str(power): quality.as_dict()
            for power, quality in qc_map(native).items()
        },
    }
    if centers.size < 5 or shift_over_step < 2.0:
        return PunchoutFit(
            source_csv=native.source_csv,
            status="unresolved",
            powers_db=powers,
            frequencies_mhz=np.asarray(frequencies),
            amplitude=amplitude,
            centers_mhz=centers,
            fitted_centers_mhz=None,
            parameters={},
            statistics={
                **base_statistics,
                "reason": "insufficient frequency resolution or power rows",
                "low_plateau_rows": min(2, centers.size),
                "high_plateau_rows": min(2, centers.size),
                "pinned_parameters": [],
            },
            recommendation=recommendation,
            metadata=native.metadata,
        )

    f_bare_guess = float(np.mean(centers[-2:]))
    shift_guess = float(np.mean(centers[:2]) - f_bare_guess)
    power_step = float(np.median(np.diff(powers)))
    initial = np.asarray(
        [
            f_bare_guess,
            shift_guess,
            float(np.mean(powers)),
            5.0,
        ]
    )
    frequency_span = max(float(np.ptp(frequencies)), frequency_step)
    lower = np.asarray(
        [
            float(np.min(frequencies) - frequency_span),
            -2.0 * frequency_span,
            float(np.min(powers)),
            max(power_step / 10.0, 0.05),
        ]
    )
    upper = np.asarray(
        [
            float(np.max(frequencies) + frequency_span),
            2.0 * frequency_span,
            float(np.max(powers)),
            max(2.0 * float(np.ptp(powers)), power_step),
        ]
    )
    result = least_squares(
        lambda parameters: punchout_model(powers, *parameters) - centers,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(frequency_step, 0.01),
        max_nfev=20_000,
    )
    if not result.success:
        raise AnalysisError(f"punchout fit failed: {result.message}")
    fitted = punchout_model(powers, *result.x)
    residual = centers - fitted
    dof = max(centers.size - result.x.size, 1)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * float(residual @ residual) / dof
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    f_bare, shift, transition_power, transition_width = map(float, result.x)
    f_low = f_bare + shift
    tolerance = max(
        float(prior_linewidth_mhz) / 4.0
        if prior_linewidth_mhz is not None
        else frequency_step,
        frequency_step,
    )
    low_plateau = np.abs(fitted - f_low) < tolerance
    high_plateau = np.abs(fitted - f_bare) < tolerance
    eligible = powers[low_plateau]
    recommended_power = (
        float(np.max(eligible) - 6.0)
        if eligible.size
        else float(np.min(powers))
    )
    recommended_power = max(float(np.min(powers)), recommended_power)
    names = (
        "f_bare_mhz",
        "punchout_shift_mhz",
        "transition_power_db",
        "transition_width_db",
    )
    parameters = {
        "f_low_mhz": float(f_low),
        "f_bare_mhz": f_bare,
        "punchout_shift_mhz": shift,
        "transition_power_db": transition_power,
        "transition_width_db": abs(transition_width),
        "recommended_r_power_db": recommended_power,
        "stderr": dict(zip(names, map(float, stderr))),
    }
    statistics = {
        **base_statistics,
        "r_squared": r_squared(centers, fitted),
        "rmse_mhz": float(np.sqrt(np.mean(residual**2))),
        "low_plateau_rows": int(np.count_nonzero(low_plateau)),
        "high_plateau_rows": int(np.count_nonzero(high_plateau)),
        "pinned_parameters": pinned_parameters(
            dict(zip(names, result.x)),
            dict(zip(names, lower)),
            dict(zip(names, upper)),
        ),
    }
    return PunchoutFit(
        source_csv=native.source_csv,
        status="resolved",
        powers_db=powers,
        frequencies_mhz=np.asarray(frequencies),
        amplitude=amplitude,
        centers_mhz=centers,
        fitted_centers_mhz=fitted,
        parameters=parameters,
        statistics=statistics,
        recommendation=recommendation,
        metadata=native.metadata,
    )


def plot_punchout_fit(fit: PunchoutFit):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        fit.frequencies_mhz,
        fit.powers_db,
        fit.amplitude,
        shading="auto",
    )
    axes[0].plot(fit.centers_mhz, fit.powers_db, "wo-", markeredgecolor="black")
    axes[0].set(xlabel="Readout frequency (MHz)", ylabel="Readout power (dB)", title="Punchout map")
    figure.colorbar(mesh, ax=axes[0])
    axes[1].plot(fit.powers_db, fit.centers_mhz, "o", label="extracted")
    if fit.fitted_centers_mhz is not None:
        axes[1].plot(fit.powers_db, fit.fitted_centers_mhz, "-", label="logistic plateaus")
    axes[1].set(
        xlabel="Readout power (dB)",
        ylabel="Notch center (MHz)",
        title=f"Status: {fit.status}",
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.suptitle(f"Fit from {fit.source_csv.name}")
    return figure


def punchout_calibration_record(fit: PunchoutFit) -> dict:
    if fit.status != "resolved":
        raise AnalysisError("an unresolved punchout scan has no calibration value")
    return {
        "value": float(fit.parameters["recommended_r_power_db"]),
        "unit": "dB",
        "uncertainty": {
            "transition_width_db": float(fit.parameters["transition_width_db"]),
            "rmse_mhz": float(fit.statistics["rmse_mhz"]),
        },
        "provenance": {
            "source": str(fit.source_csv),
            "fitted_at": utc_now(),
            "analysis": "quickexp_v3.punchout_fit.fit_punchout",
        },
        "quality": dict(fit.statistics),
        "model": "two_plateau_logistic",
        "status": "accepted",
        "accepted_at": utc_now(),
    }


def accept_punchout_fit(
    project_root: Path,
    fit: PunchoutFit,
    *,
    minimum_plateau_rows: int = 2,
    minimum_shift_over_step: float = 2.0,
    maximum_transition_width_db: float = 15.0,
) -> Path:
    if not fit.passes(
        minimum_plateau_rows=minimum_plateau_rows,
        minimum_shift_over_step=minimum_shift_over_step,
        maximum_transition_width_db=maximum_transition_width_db,
    ):
        raise AnalysisError("punchout fit did not pass acceptance gates")
    return write_calibration_records(
        project_root,
        {"defaults.r_power": punchout_calibration_record(fit)},
    )
