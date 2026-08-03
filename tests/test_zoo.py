import numpy as np
import pytest

from quickexp_v3.backend import SyntheticBackend
from quickexp_v3.experiments.base import ExperimentPlan
from quickexp_v3.synthetic_device import DeviceModel, SpuriousFeature
from quickexp_v3.zoo import DEFECT_CLASSES, ZooChip, generate_chip, generate_zoo
from tools.zoo_metrics import DecisionResult, score_results


def _qubit_plan(frequency, q_gain=0.1, z_gain=0.0):
    return ExperimentPlan(
        name="qubit_spectroscopy",
        quick_class="QubitSpectroscopy",
        title="zoo qubit trace",
        variables={"q_freq": frequency, "q_gain": q_gain, "z_gain": z_gain},
        axes=("q_freq",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"q_freq": "MHz"},
    )


def _resonator_plan(frequency, r_power=-35.0, z_gain=0.0):
    return ExperimentPlan(
        name="resonator_spectroscopy",
        quick_class="ResonatorSpectroscopy",
        title="zoo resonator trace",
        variables={
            "r_freq": frequency,
            "r_power": r_power,
            "z_gain": z_gain,
        },
        axes=("r_freq",),
        signal_names=("amplitude", "phase", "i", "q"),
        axis_units={"r_freq": "MHz"},
    )


def test_flux_independent_feature_does_not_move_with_z():
    feature = SpuriousFeature(
        kind="qubit",
        center_mhz=5590.0,
        fwhm_mhz=0.4,
        amplitude=0.30,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.linspace(5585.0, 5595.0, 401)
    at_zero = np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.1))
    at_shifted = np.abs(
        device.extra_spectral_response("qubit", frequency, 0.15, 0.1)
    )
    assert frequency[np.argmax(at_zero)] == frequency[np.argmax(at_shifted)]


def test_two_photon_feature_scales_quadratically_with_drive_gain():
    feature = SpuriousFeature(
        kind="qubit",
        center_mhz=5516.0,
        fwhm_mhz=0.5,
        amplitude=0.40,
        power_exponent=2.0,
        reference_gain=0.1,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.linspace(5514.0, 5518.0, 201)
    low = np.max(
        np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.1))
    )
    high = np.max(
        np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.2))
    )
    assert high / low == pytest.approx(4.0, rel=0.15)


def test_amplitude_is_the_strength_at_the_reference_gain():
    feature = SpuriousFeature(
        kind="qubit",
        center_mhz=5516.0,
        fwhm_mhz=0.5,
        amplitude=0.40,
        power_exponent=2.0,
        reference_gain=0.1,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.linspace(5514.0, 5518.0, 201)
    at_reference = np.max(
        np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.1))
    )
    assert at_reference == pytest.approx(0.40, rel=0.05)


def test_saturating_feature_stops_growing_above_its_saturation_gain():
    feature = SpuriousFeature(
        kind="qubit",
        center_mhz=5516.0,
        fwhm_mhz=0.3,
        amplitude=1.4,
        saturation_gain=0.04,
        reference_gain=0.1,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.linspace(5515.0, 5517.0, 201)
    low = np.max(
        np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.05))
    )
    high = np.max(
        np.abs(device.extra_spectral_response("qubit", frequency, 0.0, 0.5))
    )
    assert high == pytest.approx(low, rel=0.02)


def test_resonator_features_are_ignored_for_qubit_kind():
    feature = SpuriousFeature(
        kind="resonator",
        center_mhz=6880.0,
        fwhm_mhz=0.3,
        amplitude=0.5,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.linspace(5585.0, 5595.0, 201)
    response = device.extra_spectral_response("qubit", frequency, 0.0, 0.1)
    assert np.allclose(response, 0.0)


def test_flux_dependent_feature_broadcasts_over_a_sweep():
    feature = SpuriousFeature(
        kind="qubit",
        center_mhz=5590.0,
        fwhm_mhz=0.4,
        amplitude=0.3,
        flux_period_z=0.2,
        flux_amplitude_mhz=2.0,
    )
    device = DeviceModel(spurious_features=(feature,))
    frequency = np.asarray([5592.0, 5588.0])
    z_gain = np.asarray([0.0, 0.1])
    response = np.abs(
        device.extra_spectral_response("qubit", frequency, z_gain, 0.1)
    )
    assert response == pytest.approx([0.3, 0.3])


def test_injected_feature_appears_in_synthesized_trace():
    device = DeviceModel(
        qubit_max_frequency_mhz=5606.5,
        qubit_sweet_spot_z=0.0,
        spurious_features=(
            SpuriousFeature(
                kind="qubit",
                center_mhz=5580.0,
                fwhm_mhz=0.6,
                amplitude=2.0,
            ),
        ),
    )
    backend = SyntheticBackend(seed=3, device=device)
    frequency = np.linspace(5570.0, 5615.0, 901)
    payload = backend.acquire(_qubit_plan(frequency)).payload
    peak_frequency = payload[np.argmax(payload[:, 1]), 0]
    assert abs(peak_frequency - 5580.0) < 1.0


def test_absent_injection_leaves_the_clean_device_unchanged():
    clean = SyntheticBackend(seed=3, device=DeviceModel())
    frequency = np.linspace(5590.0, 5620.0, 601)
    payload = clean.acquire(_qubit_plan(frequency)).payload
    peak_frequency = payload[np.argmax(payload[:, 1]), 0]
    assert abs(peak_frequency - 5606.5) < 1.0


def test_resonator_power_db_is_converted_to_linear_feature_gain():
    reference_gain = 10.0 ** (-35.0 / 20.0)
    device = DeviceModel(
        spurious_features=(
            SpuriousFeature(
                kind="resonator",
                center_mhz=6878.0,
                fwhm_mhz=0.6,
                amplitude=2.0,
                reference_gain=reference_gain,
            ),
        )
    )
    frequency = np.linspace(6875.0, 6890.0, 601)
    payload = SyntheticBackend(seed=3, device=device).acquire(
        _resonator_plan(frequency)
    ).payload
    injected_peak = payload[np.argmax(payload[:, 1])]
    assert abs(injected_peak[0] - 6878.0) < 0.5
    assert injected_peak[1] < 4.0


def test_every_defect_class_generates_a_chip_with_truth_and_prior():
    for defect_class in DEFECT_CLASSES:
        chip = generate_chip(defect_class, seed=11)
        assert isinstance(chip, ZooChip)
        assert chip.defect_class == defect_class
        assert np.isfinite(chip.truth["q_freq_mhz"])
        assert np.isfinite(chip.truth["r_freq_mhz"])
        low, high = chip.prior["q_freq_mhz"]
        assert low < high


def test_generation_is_deterministic_for_a_seed():
    first = generate_chip("f02_shadow", seed=5)
    second = generate_chip("f02_shadow", seed=5)
    assert first.truth == second.truth
    assert first.device.spurious_features == second.device.spurious_features


def test_clean_chip_has_no_injected_features():
    chip = generate_chip("clean", seed=1)
    assert chip.device.spurious_features == ()


def test_wrong_prior_chip_places_truth_outside_its_window():
    chip = generate_chip("wrong_prior", seed=7)
    low, high = chip.prior["q_freq_mhz"]
    assert not (low <= chip.truth["q_freq_mhz"] <= high)


def test_low_snr_chip_has_more_spectroscopy_noise_than_clean_chip():
    clean = generate_chip("clean", seed=7)
    noisy = generate_chip("low_snr", seed=7)
    assert noisy.device.spectroscopy_noise_std > clean.device.spectroscopy_noise_std


def test_generate_zoo_covers_all_classes_and_is_sized():
    chips = generate_zoo(60, seed=2)
    assert len(chips) == 60
    assert set(chip.defect_class for chip in chips) == set(DEFECT_CLASSES)


def test_false_accept_counts_a_wrong_promoted_value():
    chips = (generate_chip("clean", seed=1),)
    truth = chips[0].truth["q_freq_mhz"]
    results = (
        DecisionResult(
            chip_id=chips[0].chip_id,
            defect_class="clean",
            verdict="accept",
            value=truth + 50.0,
        ),
    )
    metrics = score_results(results, chips, tolerance_mhz=1.0)
    assert metrics["overall"]["false_accept_rate"] == 1.0
    assert metrics["overall"]["false_reject_rate"] == 0.0


def test_correct_accept_scores_clean():
    chips = (generate_chip("clean", seed=1),)
    truth = chips[0].truth["q_freq_mhz"]
    results = (
        DecisionResult(
            chip_id=chips[0].chip_id,
            defect_class="clean",
            verdict="accept",
            value=truth + 0.2,
        ),
    )
    metrics = score_results(results, chips, tolerance_mhz=1.0)
    assert metrics["overall"]["false_accept_rate"] == 0.0
    assert metrics["overall"]["escalation_rate"] == 0.0


def test_escalation_on_a_resolvable_chip_is_a_false_reject():
    chips = (generate_chip("clean", seed=1),)
    results = (
        DecisionResult(
            chip_id=chips[0].chip_id,
            defect_class="clean",
            verdict="escalate",
            value=None,
        ),
    )
    metrics = score_results(results, chips, tolerance_mhz=1.0)
    assert metrics["overall"]["false_reject_rate"] == 1.0
    assert metrics["overall"]["escalation_rate"] == 1.0


def test_metrics_are_reported_per_defect_class():
    chips = (generate_chip("clean", seed=1), generate_chip("tls", seed=2))
    results = (
        DecisionResult(
            chips[0].chip_id,
            "clean",
            "accept",
            chips[0].truth["q_freq_mhz"],
        ),
        DecisionResult(chips[1].chip_id, "tls", "escalate", None),
    )
    metrics = score_results(results, chips, tolerance_mhz=1.0)
    assert metrics["clean"]["false_accept_rate"] == 0.0
    assert metrics["tls"]["false_reject_rate"] == 1.0
    assert metrics["overall"]["count"] == 2


def test_wrong_value_propagation_and_time_are_reported():
    chips = (generate_chip("clean", seed=1),)
    result = DecisionResult(
        chips[0].chip_id,
        "clean",
        "accept",
        chips[0].truth["q_freq_mhz"] + 20.0,
        simulated_seconds=12.5,
        wrong_value_propagated=True,
    )
    metrics = score_results((result,), chips, tolerance_mhz=1.0)
    assert metrics["overall"]["wrong_value_propagation_rate"] == 1.0
    assert metrics["overall"]["median_time_to_calibration_seconds"] == 12.5
