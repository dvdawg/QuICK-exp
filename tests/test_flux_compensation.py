from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.errors import AnalysisError, ConfigError
from quickexp_v3.flux_compensation import (
    CryoscopePhaseTrace,
    apply_filter_bundles,
    apply_predistortion,
    cryoscope_frequency,
    design_iir_inverse,
    design_inverse_fir,
    extract_cryoscope_phases,
    extract_step_spectroscopy,
    fit_forward_fir,
    fit_step_response,
    flux_command_from_phi0_fractions,
    frequency_to_flux,
    make_cryoscope_schedule,
    monotonic_branch_for_flux_step,
    recommended_shots_per_phase,
    read_step_campaign_manifest,
    read_step_campaign_metadata,
    settling_metrics,
    upload_predistorted_waveform,
    validate_waveform,
    write_filter_bundle,
    write_step_campaign_manifest,
)
from quickexp_v3.qubit_flux_fit import transmon_frequency


def synthetic_step_fit():
    time_us = np.geomspace(0.025, 50.0, 70)
    expected_taus = np.asarray([0.055, 1.3, 20.0])
    expected_alphas = np.asarray([0.10, 0.22, 0.68])
    response = np.sum(
        expected_alphas[None, :]
        * np.exp(-time_us[:, None] / expected_taus[None, :]),
        axis=1,
    )
    response += np.random.default_rng(9).normal(0.0, 8e-4, time_us.size)
    fit = fit_step_response(
        time_us,
        response,
        model_orders=(1, 2, 3, 4),
        dc_gain=0.0,
        tau_bounds_us=(0.005, 200.0),
        multistarts=4,
        seed=4,
    )
    return fit, expected_taus, expected_alphas


def write_native_map(path, outer, inner, iq_rows, *, quick_class, labels):
    rows = []
    for outer_value, iq in zip(outer, iq_rows):
        rows.append(
            np.column_stack(
                (
                    np.full(inner.size, outer_value),
                    inner,
                    np.abs(iq),
                    np.angle(iq),
                    np.real(iq),
                    np.imag(iq),
                )
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [labels[0], labels[1]],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {"quick_experiment": quick_class, "var": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_step_fit_selects_and_recovers_three_physical_timescales():
    fit, expected_taus, expected_alphas = synthetic_step_fit()
    assert fit.model_order == 3
    assert fit.statistics["r_squared"] > 0.999
    assert fit.taus_us == pytest.approx(expected_taus, rel=0.08)
    assert fit.alphas == pytest.approx(expected_alphas, abs=0.025)
    assert fit.dc_gain + np.sum(fit.alphas) == pytest.approx(1.0, abs=0.01)


def test_step_fit_does_not_force_amplitudes_to_unit_sum():
    time_us = np.geomspace(0.01, 20.0, 60)
    expected_alphas = np.asarray([0.35, 0.85])
    expected_taus = np.asarray([0.08, 4.0])
    response = np.sum(
        expected_alphas[None, :]
        * np.exp(-time_us[:, None] / expected_taus[None, :]),
        axis=1,
    )
    fit = fit_step_response(
        time_us,
        response,
        model_orders=(2,),
        dc_gain=0.0,
        tau_bounds_us=(0.005, 100.0),
        multistarts=3,
        seed=5,
    )
    assert fit.alphas == pytest.approx(expected_alphas, rel=1e-5)
    assert np.sum(fit.alphas) == pytest.approx(1.2, rel=1e-5)


def test_matched_z_inverse_has_correct_negative_zeros_and_optional_leak():
    fit, _, _ = synthetic_step_fit()
    faithful = design_iir_inverse(fit, sample_interval_ns=0.4167)
    assert np.all(np.real(faithful.continuous_zeros_per_us) < 0)
    assert faithful.marginal
    assert faithful.maximum_pole_radius == pytest.approx(1.0)
    assert np.isrealobj(faithful.sos)

    leaky = design_iir_inverse(
        fit,
        sample_interval_ns=0.4167,
        leak_tau_us=1000.0,
    )
    assert leaky.stable
    command = np.r_[np.zeros(5), np.ones(200)]
    predistorted = apply_predistortion(command, iir=leaky)
    assert predistorted.shape == command.shape
    assert np.all(np.isfinite(predistorted))


def test_frequency_inversion_is_restricted_to_explicit_monotonic_branch():
    parameters = {
        "f_max_mhz": 5600.0,
        "period_z": 0.4,
        "sweet_spot_z": 0.0,
        "asymmetry": 0.2,
        "ec_mhz": 180.0,
    }
    record = {
        "status": "accepted",
        "model": "transmon_f01",
        "valid_domain": {"z_gain": [-0.2, 0.2]},
        "value": {"parameters": parameters},
    }
    expected_z = np.asarray([-0.18, -0.14, -0.10, -0.06])
    frequency = transmon_frequency(expected_z, **parameters)
    recovered = frequency_to_flux(
        record,
        frequency,
        branch_z=(-0.2, -0.02),
    )
    assert recovered == pytest.approx(expected_z, abs=2e-6)
    with pytest.raises(AnalysisError, match="not monotonic"):
        frequency_to_flux(record, frequency, branch_z=(-0.1, 0.1))

    baseline_z, step_z = flux_command_from_phi0_fractions(
        record,
        baseline_phi0=-0.127,
        step_phi0=-0.217,
    )
    assert baseline_z == pytest.approx(-0.0508)
    assert step_z == pytest.approx(-0.0868)
    branch = monotonic_branch_for_flux_step(
        record,
        baseline_z=baseline_z,
        commanded_step_z=step_z,
    )
    assert branch[0] < baseline_z + step_z < baseline_z < branch[1]


def test_cryoscope_schedule_and_phase_difference_recover_constant_detuning():
    centers = np.linspace(0.02, 0.09, 12)
    schedule = make_cryoscope_schedule(
        centers,
        pulse_duration_us=0.1,
        sample_interval_ns=1.6693,
    )
    duration = schedule.acquisition_time_us
    detuning_mhz = -42.5
    trace = CryoscopePhaseTrace(
        source_csv=Path("synthetic.csv"),
        duration_us=duration,
        accumulated_phase_rad=2.0 * np.pi * detuning_mhz * duration,
        phase_uncertainty_rad=np.full(duration.size, 0.001),
        contrast=np.ones(duration.size),
        row_r_squared=np.ones(duration.size),
    )
    recovered = cryoscope_frequency(trace, schedule)
    assert recovered.detuning_mhz == pytest.approx(detuning_mhz, abs=1e-9)
    shots = recommended_shots_per_phase(
        schedule.delta_time_us,
        target_frequency_sigma_mhz=0.75,
        phase_count=4,
    )
    assert shots.shape == schedule.delta_time_us.shape
    assert np.all(shots >= 64)


def test_native_step_and_cryoscope_maps_recover_synthetic_features(tmp_path):
    time_us = np.geomspace(0.025, 50.0, 12)
    frequency = np.linspace(4900.0, 5100.0, 201)
    response = (
        0.75 * np.exp(-time_us / 20.0)
        + 0.25 * np.exp(-time_us / 0.8)
    )
    centers = 5000.0 + 45.0 * (2.0 * response - 1.0)
    step_iq = []
    for row_index, center in enumerate(centers):
        if row_index % 2:
            profile = np.exp(-0.5 * ((frequency - center) / 3.0) ** 2)
        else:
            profile = 1.0 / (1.0 + 1j * (frequency - center) / 1.0)
        step_iq.append(0.1 + 0.8 * profile)
    step_path = tmp_path / "step.csv"
    write_native_map(
        step_path,
        time_us,
        frequency,
        step_iq,
        quick_class="FluxStepSpectroscopy",
        labels=(
            ["Flux Step Observation Time", "us"],
            ["Qubit Probe Frequency", "MHz"],
        ),
    )
    extracted_step = extract_step_spectroscopy(
        step_path,
        minimum_row_r_squared=0.0,
        minimum_contrast_snr=0.1,
    )
    expected_centers = np.interp(extracted_step.time_us, time_us, centers)
    assert extracted_step.center_mhz == pytest.approx(expected_centers, abs=0.2)
    assert {row.line_shape for row in extracted_step.rows} == {
        "gaussian",
        "lorentzian",
    }

    schedule = make_cryoscope_schedule(
        np.linspace(0.02, 0.09, 12),
        pulse_duration_us=0.1,
        sample_interval_ns=1.6693,
    )
    phase_deg = np.linspace(0.0, 360.0, 16, endpoint=False)
    detuning = -738.0
    accumulated = 2.0 * np.pi * detuning * schedule.acquisition_time_us
    cryoscope_iq = [
        (0.5 + 0.42 * np.cos(np.deg2rad(phase_deg) - phase))
        * np.exp(0.27j)
        for phase in accumulated
    ]
    cryoscope_path = tmp_path / "cryoscope.csv"
    write_native_map(
        cryoscope_path,
        schedule.acquisition_time_us,
        phase_deg,
        cryoscope_iq,
        quick_class="Cryoscope",
        labels=(
            ["Flux Pulse Duration", "us"],
            ["Second Ramsey Pulse Phase", "deg"],
        ),
    )
    extracted_phase = extract_cryoscope_phases(
        cryoscope_path,
        phase_prior_detuning_mhz=detuning,
    )
    extracted_frequency = cryoscope_frequency(extracted_phase, schedule)
    assert extracted_frequency.detuning_mhz == pytest.approx(detuning, abs=1e-7)


def test_forward_and_inverse_fir_regularized_solves():
    sample_interval_ns = 1.0
    command = np.full(100, -0.2)
    true_forward = np.asarray([0.78, 0.16, 0.06, 0.0, 0.0])
    actual = np.convolve(command, true_forward, mode="full")[: command.size]

    def frequency_model(z):
        return 5600.0 + 900.0 * np.asarray(z)

    time_us = np.arange(command.size) * sample_interval_ns / 1000.0
    fit = fit_forward_fir(
        command,
        sample_interval_ns=sample_interval_ns,
        measured_time_us=time_us,
        measured_frequency_mhz=frequency_model(actual),
        frequency_model=frequency_model,
        baseline_z=0.0,
        coefficient_count=5,
        energy_regularization=1e-10,
        tail_regularization=1e-10,
        dc_regularization=1e-6,
        maximum_evaluations=200,
    )
    assert fit.statistics["r_squared"] > 0.999999
    assert fit.coefficients[:3] == pytest.approx(true_forward[:3], abs=2e-4)
    inverse = design_inverse_fir(
        fit.coefficients,
        sample_interval_ns=sample_interval_ns,
        inverse_length=12,
        gaussian_sigma_ns=1.0,
        derivative_regularization=1e-4,
    )
    assert inverse.statistics["rmse"] < 0.01
    assert np.all(np.isfinite(inverse.coefficients))


def test_waveform_checks_bundle_and_upload_guard(tmp_path):
    fit, _, _ = synthetic_step_fit()
    iir = design_iir_inverse(
        fit,
        sample_interval_ns=0.4167,
        leak_tau_us=1000.0,
    )
    waveform = iir.apply(np.r_[np.zeros(5), np.ones(50) * 0.2])
    check = validate_waveform(
        waveform,
        sample_interval_ns=0.4167,
        full_scale=2.5,
        maximum_fraction_of_full_scale=0.24,
    )
    assert check.passes
    bundle = write_filter_bundle(tmp_path / "candidate.json", step_fit=fit, iir=iir)
    assert bundle.is_file()
    reapplied = apply_filter_bundles(waveform, [bundle])
    assert reapplied.shape == waveform.shape
    assert np.all(np.isfinite(reapplied))
    row_csv = tmp_path / "row.csv"
    row_csv.write_text("0,0\n", encoding="utf-8")
    manifest = write_step_campaign_manifest(
        tmp_path / "campaign.json",
        [
            {
                "csv_path": row_csv,
                "probe_time_us": 0.1,
                "predicted_center_mhz": 5000.0,
            }
        ],
        metadata={"baseline_z": -0.1, "commanded_step_z": -0.2},
    )
    assert read_step_campaign_manifest(manifest) == (row_csv.resolve(),)
    assert read_step_campaign_metadata(manifest) == {
        "baseline_z": -0.1,
        "commanded_step_z": -0.2,
    }
    with pytest.raises(ConfigError, match="no verified arbitrary-waveform"):
        upload_predistorted_waveform(
            object(),
            waveform,
            channel=2,
            sample_interval_ns=0.4167,
            name="candidate",
        )


def test_settling_metric_uses_excursion_relative_threshold():
    metrics = settling_metrics(
        [0.0, 0.01, 0.015, 0.02],
        [5000.0, 5002.0, 5000.4, 4999.7],
        target_frequency_mhz=5000.0,
        full_excursion_mhz=738.0,
        settle_after_us=0.015,
    )
    assert metrics["passes"]
    assert metrics["maximum_relative_error"] < 0.001
