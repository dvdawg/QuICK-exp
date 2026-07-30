"""A coherent virtual device for offline closed-loop calibration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional

import numpy as np
import yaml

from .data import BackendResult
from .errors import AcquisitionError
from .util import to_builtin, utc_now


@dataclass
class DeviceModel:
    """Ground truth shared by every acquisition made by a SyntheticBackend."""

    resonator_base_mhz: float = 6884.0
    resonator_flux_amplitude_mhz: float = 0.7
    resonator_flux_period_z: float = 0.23
    resonator_flux_peak_z: float = -0.06
    resonator_linewidth_mhz: float = 0.45
    punchout_shift_mhz: float = 1.2
    punchout_transition_power_db: float = -20.0
    punchout_width_db: float = 4.0
    qubit_max_frequency_mhz: float = 5606.5
    qubit_flux_period_z: float = 0.30
    qubit_sweet_spot_z: float = 0.0
    qubit_asymmetry: float = 0.19
    ec_mhz: float = 180.0
    qubit_linewidth_mhz: float = 0.7
    qubit_power_broadening_mhz_per_gain: float = 0.0
    rabi_rate_per_gain_mhz: float = 3.0
    t1_us: float = 6.2
    t1_flux_modulation_fraction: float = 0.20
    t1_flux_feature_z: float = 0.10
    t1_flux_feature_width_z: float = 0.03
    t2_ramsey_us: float = 1.8
    t2_echo_us: float = 5.0
    dispersive_shift_mhz: float = 1.0
    readout_ground_mean: tuple = (-0.5, -0.08)
    readout_excited_mean: tuple = (0.5, 0.08)
    readout_ground_covariance: tuple = ((0.018, 0.008), (0.008, 0.010))
    readout_excited_covariance: tuple = ((0.012, -0.005), (-0.005, 0.020))
    thermal_population: float = 0.02
    leakage_probability: float = 0.02
    resonator_drift_mhz_per_hour: float = 0.0
    qubit_drift_mhz_per_hour: float = 0.0
    coherence_fraction_per_hour: float = 0.0
    elapsed_hours: float = 0.0
    failure_hook: Optional[Callable[[str, int], bool]] = None
    _failures: dict = field(default_factory=dict, init=False, repr=False)
    _failure_checks: int = field(default=0, init=False, repr=False)

    def resonator_frequency(self, r_power: Any, z_gain: Any) -> np.ndarray:
        power = np.asarray(r_power, dtype=float)
        z = np.asarray(z_gain, dtype=float)
        flux = self.resonator_base_mhz + self.resonator_flux_amplitude_mhz * np.cos(
            2.0
            * np.pi
            * (z - self.resonator_flux_peak_z)
            / self.resonator_flux_period_z
        )
        width = max(abs(float(self.punchout_width_db)), np.finfo(float).eps)
        punchout = self.punchout_shift_mhz / (
            1.0
            + np.exp(
                -np.clip(
                    (power - self.punchout_transition_power_db) / width,
                    -60.0,
                    60.0,
                )
            )
        )
        return (
            flux
            + punchout
            + self.resonator_drift_mhz_per_hour * self.elapsed_hours
        )

    def qubit_frequency(self, z_gain: Any) -> np.ndarray:
        z = np.asarray(z_gain, dtype=float)
        ec = float(self.ec_mhz)
        phase = (
            np.pi
            * (z - self.qubit_sweet_spot_z)
            / self.qubit_flux_period_z
        )
        squid_shape = np.sqrt(
            np.cos(phase) ** 2
            + self.qubit_asymmetry**2 * np.sin(phase) ** 2
        )
        ej_sum = (self.qubit_max_frequency_mhz + ec) ** 2 / (8.0 * ec)
        return (
            np.sqrt(8.0 * ec * ej_sum * squid_shape)
            - ec
            + self.qubit_drift_mhz_per_hour * self.elapsed_hours
        )

    def rabi_rate(self, q_gain: Any) -> np.ndarray:
        return np.abs(np.asarray(q_gain, dtype=float)) * self.rabi_rate_per_gain_mhz

    def qubit_linewidth(self, q_gain: Any) -> np.ndarray:
        """Return the intrinsic plus drive-power-broadened linewidth."""
        return (
            abs(float(self.qubit_linewidth_mhz))
            + abs(float(self.qubit_power_broadening_mhz_per_gain))
            * np.abs(np.asarray(q_gain, dtype=float))
        )

    def coherence_time(self, kind: str) -> float:
        base = {
            "t1": self.t1_us,
            "ramsey": self.t2_ramsey_us,
            "echo": self.t2_echo_us,
        }[str(kind)]
        scale = max(
            0.05,
            1.0 + self.coherence_fraction_per_hour * self.elapsed_hours,
        )
        return float(base) * scale

    def t1_at_flux(self, z_gain: Any) -> np.ndarray:
        z = np.asarray(z_gain, dtype=float)
        width = max(abs(self.t1_flux_feature_width_z), np.finfo(float).eps)
        modulation = self.t1_flux_modulation_fraction * np.exp(
            -0.5 * ((z - self.t1_flux_feature_z) / width) ** 2
        )
        return self.coherence_time("t1") * np.maximum(0.05, 1.0 - modulation)

    def advance(self, hours: float) -> None:
        self.elapsed_hours += float(hours)

    def fail_next(self, kind: str, count: int = 1) -> None:
        """Inject one or more failures for an experiment name or Quick class."""
        if int(count) < 1:
            raise ValueError("failure count must be at least one")
        key = str(kind)
        self._failures[key] = self._failures.get(key, 0) + int(count)

    def consume_failure(self, kind: str) -> bool:
        key = str(kind)
        count = int(self._failures.get(key, 0))
        if count > 0:
            if count == 1:
                self._failures.pop(key, None)
            else:
                self._failures[key] = count - 1
            return True
        self._failure_checks += 1
        return bool(
            self.failure_hook is not None
            and self.failure_hook(key, self._failure_checks)
        )


_AXIS_LABELS = {
    "r_freq": ("Readout Pulse Frequency", "MHz"),
    "r_power": ("Readout Pulse Power", "dB"),
    "q_freq": ("Qubit Pulse Frequency", "MHz"),
    "q_gain": ("Qubit Pulse Gain", "a.u."),
    "q_length": ("Qubit Pulse Length", "us"),
    "time": ("Delay Time", "us"),
    "cycle": ("Cycle", ""),
    "z_gain": ("Z Gain", "a.u."),
}
_SIGNAL_LABELS = {
    "amplitude": ("Amplitude", ""),
    "phase": ("Phase", "rad"),
    "i": ("I", ""),
    "q": ("Q", ""),
    "population": ("Population", ""),
    "i_ground": ("I Ground", ""),
    "q_ground": ("Q Ground", ""),
    "i_excited": ("I Excited", ""),
    "q_excited": ("Q Excited", ""),
    "amplitude_ground": ("Amplitude 0", ""),
    "phase_ground": ("Phase 0", "rad"),
    "i_ground_trace": ("I 0", ""),
    "q_ground_trace": ("Q 0", ""),
    "amplitude_excited": ("Amplitude 1", ""),
    "phase_excited": ("Phase 1", "rad"),
    "i_excited_trace": ("I 1", ""),
    "q_excited_trace": ("Q 1", ""),
}


def _safe_title(value: str) -> str:
    cleaned = re.sub(r"[/\\:\x00-\x1f]+", "_", str(value)).strip()
    return cleaned or "synthetic"


def write_native_pair(
    destination: Path,
    plan: Any,
    result: Any,
    *,
    index: int = 1,
    title: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist a synthetic payload in the native Quick CSV/YML pair shape."""
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() != ".csv":
        target.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{int(index):05d} - ({plan.quick_class})"
            f"{_safe_title(title or plan.title)}"
        )
        target = target / f"{stem}.csv"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = result.payload if isinstance(result, BackendResult) else result
    matrix = np.asarray(payload, dtype=float)
    if matrix.ndim != 2:
        raise AcquisitionError(
            f"native synthetic persistence requires a matrix, got {matrix.shape}"
        )
    expected = len(plan.axes) + len(plan.signal_names)
    if matrix.shape[1] != expected:
        raise AcquisitionError(
            f"synthetic payload has {matrix.shape[1]} columns; plan declares {expected}"
        )
    np.savetxt(target, matrix, delimiter=",")
    independent = [
        list(_AXIS_LABELS.get(name, (str(name), plan.axis_units.get(name, ""))))
        for name in plan.axes
    ]
    dependent = [
        list(
            _SIGNAL_LABELS.get(
                name,
                (str(name), plan.signal_units.get(name, "")),
            )
        )
        for name in plan.signal_names
    ]
    metadata = {
        "title": title or plan.title,
        "created_at": utc_now(),
        "independent": independent,
        "dependent": dependent,
        "parameters": {
            "quick_experiment": plan.quick_class,
            "var": to_builtin(dict(plan.variables)),
            "config": to_builtin(dict(plan.metadata)),
        },
    }
    if extra_metadata:
        metadata.update(to_builtin(dict(extra_metadata)))
    target.with_suffix(".yml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
