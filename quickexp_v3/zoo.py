"""Seeded adversarial devices for calibration-decision measurements."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import numpy as np

from .synthetic_device import DeviceModel, SpuriousFeature


DEFECT_CLASSES: Tuple[str, ...] = (
    "clean",
    "f02_shadow",
    "tls",
    "neighbor_qubit",
    "package_mode",
    "wrong_prior",
    "low_snr",
)


@dataclass(frozen=True)
class ZooChip:
    """One reproducible device, its truth, and its starting search prior."""

    chip_id: str
    defect_class: str
    seed: int
    device: DeviceModel
    truth: Dict[str, float] = field(default_factory=dict)
    prior: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def _base_device(rng: np.random.RandomState) -> DeviceModel:
    return DeviceModel(
        qubit_max_frequency_mhz=float(rng.uniform(5400.0, 5800.0)),
        resonator_base_mhz=float(rng.uniform(6700.0, 7100.0)),
        qubit_linewidth_mhz=float(rng.uniform(0.4, 2.5)),
        qubit_power_broadening_mhz_per_gain=float(rng.uniform(1.0, 6.0)),
        ec_mhz=float(rng.uniform(150.0, 230.0)),
        qubit_sweet_spot_z=0.0,
    )


def generate_chip(defect_class: str, seed: int) -> ZooChip:
    """Build one deterministic chip for a declared defect class."""
    name = str(defect_class)
    if name not in DEFECT_CLASSES:
        raise ValueError(
            "unknown defect class "
            + repr(name)
            + "; choose from "
            + ", ".join(DEFECT_CLASSES)
        )

    numeric_seed = int(seed)
    rng = np.random.RandomState(numeric_seed)
    device = _base_device(rng)
    q_truth = float(np.ravel(device.qubit_frequency(0.0))[0])
    r_truth = float(np.ravel(device.resonator_frequency(-35.0, 0.0))[0])
    features = ()
    prior_span = 300.0
    prior_center = q_truth + float(rng.uniform(-40.0, 40.0))

    if name == "f02_shadow":
        features = (
            SpuriousFeature(
                kind="qubit",
                center_mhz=q_truth - device.ec_mhz / 2.0,
                fwhm_mhz=device.qubit_linewidth_mhz * 1.2,
                amplitude=0.9,
                power_exponent=2.0,
                label="f02_two_photon",
            ),
        )
    elif name == "tls":
        features = (
            SpuriousFeature(
                kind="qubit",
                center_mhz=q_truth + float(rng.uniform(-25.0, 25.0)),
                fwhm_mhz=0.25,
                amplitude=1.4,
                saturation_gain=0.04,
                label="tls",
            ),
        )
    elif name == "neighbor_qubit":
        offset = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(35.0, 120.0))
        features = (
            SpuriousFeature(
                kind="qubit",
                center_mhz=q_truth + offset,
                fwhm_mhz=device.qubit_linewidth_mhz,
                amplitude=0.85,
                flux_period_z=float(device.qubit_flux_period_z * 1.7),
                flux_amplitude_mhz=60.0,
                label="neighbor_qubit",
            ),
        )
    elif name == "package_mode":
        features = (
            SpuriousFeature(
                kind="resonator",
                center_mhz=r_truth + float(rng.uniform(-6.0, 6.0)),
                fwhm_mhz=0.9,
                amplitude=0.55,
                reference_gain=10.0 ** (-35.0 / 20.0),
                label="package_mode",
            ),
        )
    elif name == "wrong_prior":
        prior_center = q_truth + float(rng.choice([-1.0, 1.0])) * float(
            rng.uniform(350.0, 600.0)
        )
        prior_span = 200.0
    elif name == "low_snr":
        device = replace(
            device,
            spectroscopy_noise_std=0.08,
            readout_ground_covariance=((0.14, 0.02), (0.02, 0.12)),
            readout_excited_covariance=((0.14, 0.02), (0.02, 0.12)),
        )

    device = replace(device, spurious_features=features)
    return ZooChip(
        chip_id="{0}-{1:05d}".format(name, numeric_seed),
        defect_class=name,
        seed=numeric_seed,
        device=device,
        truth={"q_freq_mhz": q_truth, "r_freq_mhz": r_truth},
        prior={
            "q_freq_mhz": (
                prior_center - prior_span / 2.0,
                prior_center + prior_span / 2.0,
            ),
            "r_freq_mhz": (r_truth - 15.0, r_truth + 15.0),
        },
    )


def generate_zoo(count: int, seed: int = 0) -> Tuple[ZooChip, ...]:
    """Build ``count`` chips while cycling through every defect class."""
    total = int(count)
    if total < len(DEFECT_CLASSES):
        raise ValueError(
            "zoo needs at least {0} chips to cover every class".format(
                len(DEFECT_CLASSES)
            )
        )
    return tuple(
        generate_chip(
            DEFECT_CLASSES[index % len(DEFECT_CLASSES)],
            seed=int(seed) * 10_000 + index,
        )
        for index in range(total)
    )
