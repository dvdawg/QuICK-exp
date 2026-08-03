from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.autocal.hp.candidates import Candidate
from quickexp_v3.autocal.hp.probes import (
    expand_probe_runs,
    get_probe,
    probe_ids,
)
from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.experiments.base import ExperimentPlan
from quickexp_v3.synthetic_device import (
    DeviceModel,
    SpuriousFeature,
    write_native_pair,
)


def _candidate(center):
    return Candidate(
        candidate_id="probe",
        center_mhz=center,
        fwhm_mhz=1.0,
        contrast=0.5,
        center_uncertainty_mhz=0.05,
        local_snr=20.0,
        rank=0,
        source_csv=Path("source.csv"),
        window_mhz=(center - 10.0, center + 10.0),
        statistics={"rmse": 0.025},
    )


def _write_qubit_trace(tmp_path, device, center, gain, index, z_gain=0.0):
    frequency = np.linspace(center - 4.0, center + 4.0, 401)
    plan = ExperimentPlan(
        name="qubit_spectroscopy",
        quick_class="QubitSpectroscopy",
        title="probe",
        variables={
            "q_freq": frequency,
            "q_gain": float(gain),
            "z_gain": float(z_gain),
        },
        axes=("q_freq",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"q_freq": "MHz"},
    )
    result = SyntheticBackend(seed=100 + index, device=device).acquire(plan)
    return write_native_pair(tmp_path, plan, result, index=index)


def _write_rabi_trace(tmp_path, device, center, gain, index):
    duration = np.linspace(0.02, 1.6, 160)
    plan = ExperimentPlan(
        name="rabi",
        quick_class="Rabi",
        title="rabi probe",
        variables={
            "q_freq": float(center),
            "q_gain": float(gain),
            "q_length": duration,
            "z_gain": 0.0,
        },
        axes=("q_length",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"q_length": "us"},
    )
    result = SyntheticBackend(seed=200 + index, device=device).acquire(plan)
    return write_native_pair(tmp_path, plan, result, index=index)


def test_probe_registry_is_in_declared_cost_order_and_uses_existing_experiments():
    assert probe_ids() == (
        "drive_power_ladder",
        "flux_nudge",
        "dispersive_response",
        "rabi_ping",
    )
    assert get_probe("drive_power_ladder").experiment == "qubit_spectroscopy"
    assert get_probe("flux_nudge").experiment == "qubit_spectroscopy"
    assert get_probe("dispersive_response").experiment == "dispersive_spectroscopy"
    assert get_probe("rabi_ping").experiment == "rabi"


def test_drive_power_overrides_expand_to_bounded_scalar_runs():
    runs = expand_probe_runs(
        get_probe("drive_power_ladder"),
        _candidate(5500.0),
        {"q_gain": 0.1, "q_gain_max": 0.25},
    )
    assert 3 <= len(runs) <= 5
    assert all(np.isscalar(run["q_gain"]) for run in runs)
    assert max(abs(float(run["q_gain"])) for run in runs) <= 0.25
    assert all(np.asarray(run["q_freq"]).size > 20 for run in runs)


def test_power_ladder_recovers_two_photon_quadratic_scaling(tmp_path):
    center = 5500.0
    device = DeviceModel(
        qubit_max_frequency_mhz=5600.0,
        spurious_features=(
            SpuriousFeature(
                kind="qubit",
                center_mhz=center,
                fwhm_mhz=0.8,
                amplitude=0.35,
                power_exponent=2.0,
                reference_gain=0.1,
            ),
        ),
    )
    gains = (0.04, 0.07, 0.10, 0.16)
    paths = tuple(
        _write_qubit_trace(tmp_path, device, center, gain, index + 1)
        for index, gain in enumerate(gains)
    )
    response = get_probe("drive_power_ladder").extract_response(paths)
    assert response["contrast_power_exponent"] == pytest.approx(2.0, abs=0.25)
    assert response["gain_levels"] == 4.0


def test_power_ladder_recovers_wanted_single_photon_scaling(tmp_path):
    center = 5600.0
    device = DeviceModel(
        qubit_max_frequency_mhz=center,
        qubit_spectroscopy_saturation_gain=0.3,
    )
    gains = (0.06, 0.10, 0.16, 0.24)
    paths = tuple(
        _write_qubit_trace(tmp_path, device, center, gain, index + 1)
        for index, gain in enumerate(gains)
    )
    response = get_probe("drive_power_ladder").extract_response(paths)
    assert response["contrast_power_exponent"] == pytest.approx(1.0, abs=0.2)


def test_flux_nudge_extracts_slope_and_curvature(tmp_path):
    device = DeviceModel(
        qubit_max_frequency_mhz=5600.0,
        qubit_sweet_spot_z=0.0,
        qubit_flux_period_z=0.30,
    )
    # Keep every displaced line inside the probe's narrow spectroscopy window.
    z_values = (0.0498, 0.0499, 0.05, 0.0501, 0.0502)
    center = float(device.qubit_frequency(0.05))
    paths = tuple(
        _write_qubit_trace(
            tmp_path,
            device,
            center,
            0.1,
            index + 1,
            z_gain=z_gain,
        )
        for index, z_gain in enumerate(z_values)
    )
    response = get_probe("flux_nudge").extract_response(paths)
    assert response["flux_slope_mhz_per_z"] < 0.0
    assert response["flux_curvature_mhz_per_z2"] < 0.0
    assert response["flux_levels"] == 5.0


def test_rabi_ping_recovers_linear_gain_scaling(tmp_path):
    device = DeviceModel(
        qubit_max_frequency_mhz=5600.0,
        rabi_rate_per_gain_mhz=3.0,
    )
    paths = (
        _write_rabi_trace(tmp_path, device, 5600.0, 0.2, 1),
        _write_rabi_trace(tmp_path, device, 5600.0, 0.4, 2),
    )
    response = get_probe("rabi_ping").extract_response(paths)
    assert response["rabi_gain_exponent"] == pytest.approx(1.0, abs=0.15)
    assert response["rabi_contrast"] > 0.4


def test_dispersive_response_recovers_signed_readout_shift(tmp_path):
    device = DeviceModel(
        qubit_max_frequency_mhz=5600.0,
        resonator_base_mhz=6884.0,
        resonator_flux_amplitude_mhz=0.0,
        dispersive_shift_mhz=1.2,
    )
    frequency = np.linspace(6879.0, 6889.0, 501)
    plan = ExperimentPlan(
        name="dispersive_spectroscopy",
        quick_class="DispersiveSpectroscopy",
        title="P4",
        variables={
            "r_freq": frequency,
            "r_power": -35.0,
            "q_freq": 5600.0,
            "q_gain": 0.4,
            "q_length": 0.2,
            "z_gain": 0.0,
        },
        axes=("r_freq",),
        signal_names=(
            "amplitude_ground",
            "phase_ground",
            "i_ground",
            "q_ground",
            "amplitude_excited",
            "phase_excited",
            "i_excited",
            "q_excited",
        ),
        axis_units={"r_freq": "MHz"},
    )
    path = write_native_pair(
        tmp_path,
        plan,
        SyntheticBackend(seed=44, device=device).acquire(plan),
    )
    response = get_probe("dispersive_response").extract_response((path,))
    assert response["dispersive_shift_mhz"] == pytest.approx(1.2, abs=0.08)
    assert response["dispersive_snr"] > 1.0
