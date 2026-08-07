"""Round-trip and guard tests for the resonator-sensed flux transient."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from quickexp_v3.data import ExperimentData
from quickexp_v3.errors import AnalysisError, ConfigError
from quickexp_v3.notch_fit import complex_notch_model
from quickexp_v3.resonator_transient import (
    ResonatorFluxCalibration,
    ResonatorTransientParameters,
    TransientCampaign,
    analysis_mask,
    calibration_from_accepted_record,
    cavity_time_constant_us,
    fit_transient_tail,
    invert_transient,
    preflight,
)


# The 2026-08 fit of the 5879 MHz resonator, as used by 18a.
CALIBRATION = ResonatorFluxCalibration(
    center_frequency_mhz=5879.22,
    amplitude_mhz=0.825,
    period_z=0.36,
    peak_bias_z=-0.12,
    rmse_mhz=0.493,
    domain_z=(-0.30, 0.30),
)

BASELINE_Z = -0.080
STEP_Z = 0.015
LINEWIDTH_MHZ = 0.8
# Baseline amplitude/slope/phase/delay and complex coupling.
NOTCH_TAIL = (1.0, 0.002, 0.3, 0.3, 0.7, 0.25)


def _write_notch_csv(path: Path, frequency: np.ndarray, iq: np.ndarray) -> None:
    np.savetxt(
        path,
        np.column_stack((frequency, np.abs(iq), np.angle(iq), iq.real, iq.imag)),
        delimiter=",",
    )
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [["Readout Pulse Frequency", "MHz"]],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                # The reference and settled sweeps are static held-Z
                # measurements acquired through resonator_spectroscopy.
                "parameters": {
                    "quick_experiment": "ResonatorSpectroscopy",
                    "var": {"z_gain": BASELINE_Z},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _parameters(**overrides) -> ResonatorTransientParameters:
    defaults = {
        "baseline_z": BASELINE_Z,
        "commanded_step_z": STEP_Z,
        "calibration": CALIBRATION,
        "probe_times_us": np.geomspace(0.025, 100.0, 70),
        "transient_probe_points": 5,
        "readout_length_us": 2.0,
        "cavity_lifetimes_to_mask": 5.0,
        "fit_dc_gain": None,
    }
    defaults.update(overrides)
    return ResonatorTransientParameters(**defaults)


def _synthetic_campaign(
    tmp_path: Path,
    *,
    taus_us,
    deficits,
    noise=0.0,
    seed=11,
    parameters=None,
):
    """Build a campaign whose flux trajectory has known time constants.

    The line is DC-coupled, so the flux settles to the commanded step level:
    ``z(t) = baseline + dz * (1 - sum_i deficit_i * exp(-t/tau_i))``.
    """
    parameters = parameters or _parameters()
    checks = preflight(parameters, CALIBRATION)
    rng = np.random.default_rng(seed)

    # Reference sweep at the baseline, from which the line shape is fitted.
    reference_frequency = np.linspace(
        checks.probe_centre_mhz - 2.5, checks.probe_centre_mhz + 2.5, 41
    )
    reference_notch = (checks.baseline_frequency_mhz, LINEWIDTH_MHZ) + NOTCH_TAIL
    reference_iq = complex_notch_model(reference_frequency, reference_notch)
    if noise:
        reference_iq = reference_iq + noise * (
            rng.normal(size=reference_frequency.size)
            + 1j * rng.normal(size=reference_frequency.size)
        )
    reference_csv = tmp_path / "reference.csv"
    _write_notch_csv(reference_csv, reference_frequency, reference_iq)

    # Transient map: probe_time outer, r_freq inner.
    times = np.asarray(parameters.probe_times_us, dtype=float)
    span = parameters.transient_span_linewidths * LINEWIDTH_MHZ
    probes = checks.probe_centre_mhz + np.linspace(
        -0.5 * span, 0.5 * span, parameters.transient_probe_points
    )
    deficit = np.zeros_like(times)
    for amplitude, tau in zip(deficits, taus_us):
        deficit = deficit + amplitude * np.exp(-times / tau)
    flux = BASELINE_Z + STEP_Z * (1.0 - deficit)
    resonance = CALIBRATION.frequency(flux)

    grid_time, grid_probe = np.meshgrid(times, probes, indexing="ij")
    iq = np.empty(grid_time.shape, dtype=complex)
    for index, centre in enumerate(resonance):
        iq[index] = complex_notch_model(
            probes, (centre, LINEWIDTH_MHZ) + NOTCH_TAIL
        )
    if noise:
        iq = iq + noise * (
            rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape)
        )

    data = ExperimentData(
        axes={"probe_time": grid_time.ravel(), "r_freq": grid_probe.ravel()},
        signals={"i": iq.ravel().real, "q": iq.ravel().imag},
    )
    campaign = TransientCampaign(
        calibration=CALIBRATION,
        preflight=checks,
        reference_csv=reference_csv,
        settled_csv=None,
        transient_csv=None,
        reference_result=None,
        settled_result=None,
        transient_result=SimpleNamespace(data=data),
    )
    return campaign, parameters, flux


# ----------------------------------------------------------------------------
# Calibration geometry
# ----------------------------------------------------------------------------


def test_calibration_matches_hand_computed_operating_point():
    assert CALIBRATION.frequency(-0.080) == pytest.approx(5879.852, abs=1e-3)
    assert CALIBRATION.frequency(-0.065) == pytest.approx(5879.693, abs=1e-3)
    excursion = CALIBRATION.frequency(-0.065) - CALIBRATION.frequency(-0.080)
    assert excursion == pytest.approx(-0.1588, abs=1e-3)
    assert CALIBRATION.maximum_slope_mhz_per_z == pytest.approx(14.399, rel=1e-3)


def test_maximum_slope_biases_are_quarter_period_from_the_extremum():
    biases = CALIBRATION.maximum_slope_biases()
    assert np.allclose(np.sort(biases), [-0.21, -0.03, 0.15], atol=1e-9)
    slopes = np.abs(CALIBRATION.slope_mhz_per_z(biases))
    assert np.allclose(slopes, CALIBRATION.maximum_slope_mhz_per_z, rtol=1e-9)


def test_frequency_inversion_round_trips_on_the_branch():
    z = np.linspace(-0.119, 0.059, 97)
    frequency = CALIBRATION.frequency(z)
    recovered = CALIBRATION.flux_from_frequency(frequency, side=+1.0)
    assert np.allclose(recovered, z, atol=1e-9)


def test_step_across_the_extremum_is_rejected():
    # -0.16 -> -0.10 straddles the extremum at -0.12, where f_r(z) is even.
    with pytest.raises(ConfigError, match="extremum"):
        CALIBRATION.branch(-0.16, -0.10)


def test_step_leaving_the_half_period_is_rejected():
    with pytest.raises(ConfigError, match="half-period"):
        CALIBRATION.branch(-0.10, 0.30)


def test_accepted_record_round_trips():
    record = {
        "value": {
            "parameters": {
                "center_frequency": 5879.22,
                "amplitude": 0.825,
                "period": 0.36,
                "peak_bias": -0.12,
            }
        },
        "uncertainty": {"rmse_mhz": 0.493},
        "valid_domain": {"z_gain": [-0.30, 0.30]},
    }
    assert calibration_from_accepted_record(record) == CALIBRATION


# ----------------------------------------------------------------------------
# Preflight and masking
# ----------------------------------------------------------------------------


def test_preflight_reports_the_operating_point():
    checks = preflight(_parameters(), CALIBRATION)
    assert checks.excursion_mhz == pytest.approx(-0.1588, abs=1e-3)
    assert checks.branch_z == pytest.approx((-0.12, 0.06))
    assert 0.6 < checks.slope_fraction_of_maximum < 0.8
    # f_r is close to linear across a step this small.
    assert checks.curvature_percent < 5.0
    assert "monotonic" in checks.report()


def test_preflight_rejects_a_step_off_the_calibration_domain():
    with pytest.raises(ConfigError, match="outside the calibration domain"):
        preflight(_parameters(baseline_z=0.29, commanded_step_z=0.05), CALIBRATION)


def test_preflight_rejects_a_zero_step():
    with pytest.raises(ConfigError, match="non-zero"):
        preflight(_parameters(commanded_step_z=0.0), CALIBRATION)


def test_cavity_time_constant_matches_two_over_kappa():
    # tau_r = 2/kappa = 1/(pi*FWHM); 0.8 MHz -> 398 ns.
    assert cavity_time_constant_us(0.8) == pytest.approx(0.3979, rel=1e-3)
    assert cavity_time_constant_us(6884.19 / 10000) == pytest.approx(0.4624, rel=1e-3)


def test_mask_takes_the_binding_constraint():
    times = np.asarray([0.1, 1.0, 3.0, 30.0])
    # 5*tau_r = 1.99 us, shorter than a 2 us readout, so the readout binds.
    mask, threshold = analysis_mask(
        times, linewidth_mhz=0.8, readout_length_us=2.0, cavity_lifetimes=5.0
    )
    assert threshold == pytest.approx(2.0)
    assert mask.tolist() == [False, False, True, True]
    # A narrow resonator makes the cavity bind instead.
    _, narrow = analysis_mask(
        times, linewidth_mhz=0.05, readout_length_us=2.0, cavity_lifetimes=5.0
    )
    assert narrow == pytest.approx(5.0 / (np.pi * 0.05))


# ----------------------------------------------------------------------------
# Inversion round trip
# ----------------------------------------------------------------------------


def test_inversion_recovers_a_noiseless_flux_trajectory(tmp_path):
    campaign, parameters, truth = _synthetic_campaign(
        tmp_path, taus_us=(5.0, 30.0), deficits=(0.15, 0.10)
    )
    trace = invert_transient(campaign, parameters)
    assert trace.linewidth_mhz == pytest.approx(LINEWIDTH_MHZ, rel=0.05)
    assert not np.any(trace.clipped)
    # The flux axis is recovered to well under a percent of the step.
    error = np.abs(trace.flux_z - truth)
    assert np.max(error[trace.mask]) < 0.01 * abs(STEP_Z)


def _dominant_poles(fit, count):
    """The ``count`` time constants carrying the most amplitude.

    BIC may add extra low-amplitude terms to chase inversion systematics, so
    presence-with-weight is the meaningful check rather than the sorted extremes.
    """
    order = np.argsort(np.abs(fit.alphas))[::-1][:count]
    return np.sort(np.asarray(fit.taus_us)[order])


def test_inversion_recovers_injected_time_constants(tmp_path):
    taus = (5.0, 30.0)
    deficits = (0.15, 0.10)
    campaign, parameters, _ = _synthetic_campaign(
        tmp_path, taus_us=taus, deficits=deficits
    )
    trace = invert_transient(campaign, parameters)
    fit, inverse = fit_transient_tail(trace, parameters)

    dominant = _dominant_poles(fit, 2)
    assert dominant[0] == pytest.approx(taus[0], rel=0.10)
    assert dominant[1] == pytest.approx(taus[1], rel=0.10)

    # The two dominant amplitudes are the injected deficits, carried up by the
    # boxcar-centroid time shift: alpha_i = -deficit_i * exp(shift/tau_i).
    shift = 0.5 * parameters.readout_length_us
    weights = np.sort(np.abs(fit.alphas))[::-1][:2]
    expected = np.sort(
        [amplitude * np.exp(shift / tau) for amplitude, tau in zip(deficits, taus)]
    )[::-1]
    assert weights == pytest.approx(expected, rel=0.15)

    # A DC-coupled line settles to the commanded level.
    assert fit.dc_gain == pytest.approx(1.0, abs=0.05)
    assert np.isfinite(inverse.sos).all()


def test_time_constants_are_bounded_to_the_measured_window(tmp_path):
    """An unbounded fit puts poles far outside the data into the inverse."""
    campaign, parameters, _ = _synthetic_campaign(
        tmp_path, taus_us=(5.0, 30.0), deficits=(0.15, 0.10)
    )
    trace = invert_transient(campaign, parameters)
    fit, _ = fit_transient_tail(trace, parameters)
    fitted = trace.fitted_times_us
    window = (float(np.min(fitted)), float(np.max(fitted)))
    assert np.all(fit.taus_us >= window[0] * (1 - 1e-9))
    assert np.all(fit.taus_us <= window[1] * (1 + 1e-9))


def test_inversion_survives_measurement_noise(tmp_path):
    taus = (5.0, 30.0)
    campaign, parameters, _ = _synthetic_campaign(
        tmp_path, taus_us=taus, deficits=(0.15, 0.10), noise=0.002, seed=5
    )
    trace = invert_transient(campaign, parameters)
    fit, _ = fit_transient_tail(trace, parameters)
    recovered = np.sort(fit.taus_us)
    assert recovered[-1] == pytest.approx(taus[1], rel=0.35)
    assert np.all(trace.normalized_uncertainty[trace.mask] > 0)


def test_inversion_is_insensitive_to_a_complex_gain_drift(tmp_path):
    """A gain and phase change between reference and transient must not alias
    into flux. It is fitted out, not absorbed into z_hat."""
    campaign, parameters, truth = _synthetic_campaign(
        tmp_path, taus_us=(5.0, 30.0), deficits=(0.15, 0.10)
    )
    drift = 1.35 * np.exp(0.7j)
    data = campaign.transient_result.data
    drifted = ExperimentData(
        axes=dict(data.axes),
        signals={"i": (data.iq * drift).real, "q": (data.iq * drift).imag},
    )
    drifted_campaign = TransientCampaign(
        **{
            **campaign.__dict__,
            "transient_result": SimpleNamespace(data=drifted),
        }
    )
    trace = invert_transient(drifted_campaign, parameters)
    error = np.abs(trace.flux_z - truth)
    assert np.max(error[trace.mask]) < 0.01 * abs(STEP_Z)


def test_inversion_requires_a_reference_sweep(tmp_path):
    campaign, parameters, _ = _synthetic_campaign(
        tmp_path, taus_us=(5.0, 30.0), deficits=(0.15, 0.10)
    )
    without_reference = TransientCampaign(
        **{**campaign.__dict__, "reference_csv": None}
    )
    with pytest.raises(AnalysisError, match="no reference CSV"):
        invert_transient(without_reference, parameters)


def test_fit_refuses_when_the_mask_leaves_too_few_rows(tmp_path):
    parameters = _parameters(
        probe_times_us=np.geomspace(0.025, 1.0, 12),
        readout_length_us=2.0,
    )
    campaign, parameters, _ = _synthetic_campaign(
        tmp_path, taus_us=(5.0, 30.0), deficits=(0.15, 0.10), parameters=parameters
    )
    with pytest.raises(AnalysisError, match="cavity/readout limit"):
        invert_transient(campaign, parameters)
