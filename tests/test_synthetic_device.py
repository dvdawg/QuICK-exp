import numpy as np
import pytest

from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.errors import AcquisitionError
from quickexp_v3.experiments.base import ExperimentPlan
from quickexp_v3.native_index import NativeIndex
from quickexp_v3.synthetic_device import DeviceModel, write_native_pair


def _spectroscopy_plan(z_gain):
    frequency = np.linspace(6879.0, 6889.0, 501)
    return ExperimentPlan(
        name="resonator_spectroscopy",
        quick_class="ResonatorSpectroscopy",
        title="synthetic flux trace",
        variables={
            "r_freq": frequency,
            "r_power": -40.0,
            "z_gain": z_gain,
        },
        axes=("r_freq",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"r_freq": "MHz"},
    )


def test_device_model_resonator_flux_is_consistent_across_acquisitions():
    device = DeviceModel(
        resonator_flux_amplitude_mhz=1.5,
        resonator_flux_period_z=0.4,
        resonator_flux_peak_z=0.0,
        punchout_shift_mhz=0.0,
    )
    backend = SyntheticBackend(seed=8, device=device)
    first = backend.acquire(_spectroscopy_plan(0.0)).payload
    second = backend.acquire(_spectroscopy_plan(0.2)).payload
    first_center = first[np.argmin(first[:, 1]), 0]
    second_center = second[np.argmin(second[:, 1]), 0]

    assert first_center == pytest.approx(
        device.resonator_frequency(-40.0, 0.0),
        abs=0.03,
    )
    assert second_center == pytest.approx(
        device.resonator_frequency(-40.0, 0.2),
        abs=0.03,
    )
    assert first_center - second_center == pytest.approx(3.0, abs=0.05)


def test_device_model_rabi_chevron_has_detuning_hyperbola():
    device = DeviceModel(
        qubit_max_frequency_mhz=5600.0,
        rabi_rate_per_gain_mhz=3.0,
    )
    frequency = np.asarray([5600.0, 5603.0])
    duration = np.linspace(0.0, 0.8, 801)
    plan = ExperimentPlan(
        name="rabi_chevron",
        quick_class="Rabi",
        title="synthetic chevron",
        variables={
            "q_freq": frequency,
            "q_length": duration,
            "q_gain": 0.4,
            "z_gain": 0.0,
        },
        axes=("q_freq", "q_length"),
        signal_names=("amplitude", "phase", "i", "q"),
    )
    matrix = SyntheticBackend(seed=9, device=device).acquire(plan).payload
    signal = matrix[:, -2].reshape(frequency.size, duration.size)
    first_peak = []
    for row in signal:
        smoothed = np.convolve(row, np.ones(21) / 21.0, mode="same")
        search = smoothed[:500]
        first_peak.append(duration[int(np.argmax(search))])

    assert first_peak[1] < first_peak[0]
    expected_ratio = 1.2 / np.sqrt(1.2**2 + 3.0**2)
    assert first_peak[1] / first_peak[0] == pytest.approx(
        expected_ratio,
        rel=0.12,
    )


def test_device_model_qubit_linewidth_tracks_drive_power_broadening():
    device = DeviceModel(
        qubit_linewidth_mhz=0.7,
        qubit_power_broadening_mhz_per_gain=4.0,
    )
    assert device.qubit_linewidth(0.4) == pytest.approx(2.3)
    assert device.qubit_linewidth(0.2) == pytest.approx(1.5)


def test_failure_injection_drift_and_native_pair_persistence(tmp_path):
    device = DeviceModel(resonator_drift_mhz_per_hour=0.2)
    device.fail_next("resonator_spectroscopy")
    backend = SyntheticBackend(seed=10, device=device)
    plan = _spectroscopy_plan(0.0)
    with pytest.raises(AcquisitionError, match="injected failure"):
        backend.acquire(plan)

    before = device.resonator_frequency(-40.0, 0.0)
    device.advance(5.0)
    after = device.resonator_frequency(-40.0, 0.0)
    assert after - before == pytest.approx(1.0)

    result = backend.acquire(plan)
    csv_path = write_native_pair(tmp_path / "data", plan, result, index=7)
    index = NativeIndex(
        csv_path.parent,
        cache_root=tmp_path / "cache",
    ).refresh()
    record = index.latest(
        quick_class="ResonatorSpectroscopy",
        n_axes=1,
    )
    assert record.index == 7
    assert record.csv_rows == 501
