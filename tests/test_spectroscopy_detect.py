"""Feature detection: cable delay, thresholds, geometry, and the null case."""

from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.errors import AnalysisError
from quickexp_v3.native_fit import fit_spectroscopy
from quickexp_v3.spectroscopy_detect import (
    detect_features,
    estimate_electrical_delay,
    prominence_cut,
    remove_electrical_delay,
    residual_group_delay,
    resonance_projection,
)
from test_native_fit import write_pair


REAL_ROOT = Path("/Users/dvdkm/Documents/code/qdg/data")


def notch_trace(frequency, center, fwhm, *, delay_ns=0.0, depth=0.8, noise=0.0, seed=0):
    """A notch resonance carried on a cable delay, as measured."""
    detuning = 2.0 * (frequency - center) / fwhm
    response = 1.0 - depth / (1.0 + 1j * detuning)
    slope = -2.0 * np.pi * delay_ns * 1e-3
    values = response * np.exp(1j * slope * (frequency - np.mean(frequency)))
    if noise:
        rng = np.random.default_rng(seed)
        values = values + noise * (
            rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
        )
    return values


def qubit_trace(frequency, center, fwhm, *, contrast=0.3, noise=0.0, seed=0):
    """A drive-frequency sweep: the readout point slides along one direction."""
    population = 1.0 / (1.0 + (2.0 * (frequency - center) / fwhm) ** 2)
    ground = 1.0 + 0.5j
    displacement = contrast * (0.6 - 0.8j)
    values = ground + displacement * population
    if noise:
        rng = np.random.default_rng(seed)
        values = values + noise * (
            rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
        )
    return values


# --------------------------------------------------------------------------
# electrical delay
# --------------------------------------------------------------------------


@pytest.mark.parametrize("delay_ns", (0.0, 50.0, -200.0, 1000.0))
def test_electrical_delay_is_recovered_without_unwrapping(delay_ns):
    """What matters is how much rotation is left, not the delay in nanoseconds.

    The resonance's own phase swing biases any estimator slightly; the
    requirement is that what remains is a small fraction of a turn, so the
    resonance arc is the dominant structure left in the I/Q plane.
    """
    frequency = np.linspace(6850.0, 6870.0, 401)
    span = float(np.ptp(frequency))
    values = notch_trace(frequency, 6860.0, 1.0, delay_ns=delay_ns)
    estimate = estimate_electrical_delay(frequency, values)
    assert estimate.coherence > 0.9

    residual_ns = estimate.delay_ns - delay_ns
    residual_turns = abs(residual_ns * 1e-3 * span)
    assert residual_turns < 0.1
    assert estimate.delay_ns == pytest.approx(delay_ns, rel=0.05, abs=3.0)


def test_noise_only_phase_yields_no_coherent_delay():
    """Unwrapping noise manufactures a large slope; the resultant does not."""
    rng = np.random.default_rng(11)
    frequency = np.linspace(3000.0, 4000.0, 1000)
    values = rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    estimate = estimate_electrical_delay(frequency, values)
    assert estimate.coherence < 0.3
    assert not estimate.applied


def test_removing_delay_collapses_the_off_resonant_circle():
    """Many turns of delay become one resonance circle once removed."""
    frequency = np.linspace(6850.0, 6890.0, 801)
    values = notch_trace(frequency, 6870.0, 1.0, delay_ns=800.0)
    estimate = estimate_electrical_delay(frequency, values)
    assert estimate.turns > 10
    corrected, _ = remove_electrical_delay(frequency, values)
    off_resonance = np.abs(frequency - 6870.0) > 5.0
    # Off resonance the corrected trace sits still instead of circling.
    assert np.ptp(np.angle(corrected[off_resonance])) < 0.2
    assert np.ptp(np.angle(values[off_resonance])) > 3.0


# --------------------------------------------------------------------------
# what the channels are for
# --------------------------------------------------------------------------


def test_group_delay_peaks_on_centre_where_phase_peaks_beside_it():
    """Phase is dispersive; seeding from it lands a linewidth off centre."""
    frequency = np.linspace(6855.0, 6865.0, 1001)
    center, fwhm = 6860.0, 0.5
    values = notch_trace(frequency, center, fwhm)
    corrected, _ = remove_electrical_delay(frequency, values)

    phase = np.unwrap(np.angle(corrected))
    phase = phase - np.polyval(np.polyfit(frequency, phase, 1), frequency)
    phase_peak = frequency[int(np.argmax(np.abs(phase)))]

    group = residual_group_delay(frequency, corrected)
    group_peak = frequency[int(np.argmax(np.abs(group - np.median(group))))]

    assert group_peak == pytest.approx(center, abs=2.0 * np.diff(frequency)[0])
    # The point of the channel: group delay lands on the centre while the
    # phase extremum sits well away from it.
    assert abs(phase_peak - center) > 10.0 * abs(group_peak - center)


def test_resonance_projection_is_symmetric_where_principal_axis_is_not():
    """The max-variance axis mixes quadratures into a Fano shape."""
    frequency = np.linspace(6855.0, 6865.0, 1001)
    center = 6860.0
    values = notch_trace(frequency, center, 0.5)
    corrected, _ = remove_electrical_delay(frequency, values)
    projected, _ = resonance_projection(frequency, corrected, center)
    peak = int(np.argmax(np.abs(projected - np.median(projected))))
    assert frequency[peak] == pytest.approx(center, abs=2.0 * np.diff(frequency)[0])
    # Symmetric about the centre: equal-and-opposite detunings agree.
    left = np.interp(center - 1.0, frequency, projected)
    right = np.interp(center + 1.0, frequency, projected)
    assert left == pytest.approx(right, rel=0.05)


# --------------------------------------------------------------------------
# detection thresholds
# --------------------------------------------------------------------------


def test_pure_noise_reports_no_feature_rather_than_a_centre():
    rng = np.random.default_rng(5)
    frequency = np.linspace(4800.0, 4900.0, 1000)
    values = 1.0 + 0.05 * (
        rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    )
    detection = detect_features(frequency, values, kind="qubit")
    assert not detection.found
    with pytest.raises(AnalysisError, match="no credible|no spectroscopy feature"):
        detection.require_best()


def test_a_spike_never_yields_a_sub_sample_linewidth(tmp_path):
    """The original failure: a lone noisy sample fitted as a delta function.

    Smoothing turns a one-sample spike into a bump a few samples wide, so it
    cannot always be told apart from a critically sampled line -- but the
    reported linewidth must never fall below the sampling, which is what used
    to happen (0.002 MHz on a 0.01 MHz grid).
    """
    rng = np.random.default_rng(7)
    frequency = np.linspace(4800.0, 4820.0, 2000)
    step = float(np.median(np.diff(frequency)))
    signal = 1.0 + 0.01 * rng.normal(size=frequency.size)
    signal[1000] += 5.0
    source = write_pair(
        tmp_path / "spike.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    try:
        fit = fit_spectroscopy(source, kind="qubit", signal="amplitude")
    except AnalysisError:
        return
    assert fit.parameters["fwhm_mhz"] >= step


def test_prominence_cut_grows_with_trace_length():
    """More samples means more chances for noise to peak."""
    assert prominence_cut(100, 1.0) < prominence_cut(10000, 1.0)
    # Calibrated against simulated white noise, whose maximum
    # prominence-to-noise ratio runs from about 6.3 to about 8.2 over
    # this range.
    assert 6.0 < prominence_cut(100, 1.0) < 9.0
    assert 8.0 < prominence_cut(10000, 3.0) < 11.0


@pytest.mark.parametrize("delay_ns", (0.0, 400.0))
def test_resonance_is_found_through_cable_delay(delay_ns):
    frequency = np.linspace(6850.0, 6870.0, 601)
    center = 6861.3
    values = notch_trace(frequency, center, 0.8, delay_ns=delay_ns, noise=0.01, seed=3)
    detection = detect_features(frequency, values, kind="resonator")
    assert detection.found
    assert detection.best.center_mhz == pytest.approx(center, abs=0.25)


def test_qubit_line_is_found_and_noise_is_not():
    frequency = np.linspace(4600.0, 4700.0, 1000)
    center = 4655.0
    found = detect_features(
        frequency,
        qubit_trace(frequency, center, 4.0, noise=0.02, seed=1),
        kind="qubit",
    )
    assert found.found
    assert found.best.center_mhz == pytest.approx(center, abs=1.0)
    # The same trace without the line must not produce a centre.
    rng = np.random.default_rng(1)
    flat = (1.0 + 0.5j) + 0.02 * (
        rng.normal(size=frequency.size) + 1j * rng.normal(size=frequency.size)
    )
    assert not detect_features(frequency, flat, kind="qubit").found


def test_iq_geometry_separates_a_line_from_noise():
    """Deviations near a real line share one direction; noise does not."""
    frequency = np.linspace(4600.0, 4700.0, 1000)
    line = detect_features(
        frequency,
        qubit_trace(frequency, 4655.0, 4.0, noise=0.02, seed=2),
        kind="qubit",
    )
    assert line.best.geometry_score > 0.5


def test_baseline_side_lobes_are_not_extra_features():
    """A wide baseline under a tall line leaves dips on both flanks."""
    frequency = np.linspace(5590.0, 5610.0, 801)
    values = qubit_trace(frequency, 5600.0, 0.6, contrast=3.0, noise=0.005, seed=4)
    detection = detect_features(frequency, values, kind="qubit")
    assert detection.found
    assert len(detection.candidates) == 1


def test_marginal_candidates_are_reported_when_none_are_accepted():
    """A near miss is more useful than a bare "nothing found"."""
    frequency = np.linspace(4600.0, 4700.0, 1000)
    values = qubit_trace(frequency, 4655.0, 4.0, contrast=0.02, noise=0.05, seed=6)
    detection = detect_features(frequency, values, kind="qubit")
    if not detection.found:
        assert detection.marginal
        with pytest.raises(AnalysisError, match="strongest candidate"):
            detection.require_best()


# --------------------------------------------------------------------------
# the fitters that sit on top
# --------------------------------------------------------------------------


def test_fit_declines_a_noise_only_trace_instead_of_inventing_a_line(tmp_path):
    rng = np.random.default_rng(9)
    frequency = np.linspace(4800.0, 4820.0, 2000)
    signal = 1.0 + 0.01 * rng.normal(size=frequency.size)
    source = write_pair(
        tmp_path / "noise.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    with pytest.raises(AnalysisError):
        fit_spectroscopy(source, kind="qubit", signal="amplitude")


def test_fitted_width_stays_between_one_sample_and_the_sweep(tmp_path):
    """The old bounds allowed both a delta function and a sweep-wide "line"."""
    rng = np.random.default_rng(10)
    frequency = np.linspace(4800.0, 4900.0, 1000)
    step = float(np.median(np.diff(frequency)))
    signal = (
        1.0
        + 0.002 * (frequency - 4850.0)
        + 0.5 / (1.0 + ((frequency - 4851.7) / 1.5) ** 2)
        + 0.01 * rng.normal(size=frequency.size)
    )
    source = write_pair(
        tmp_path / "line.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    fit = fit_spectroscopy(source, kind="qubit", signal="amplitude")
    assert fit.center_mhz == pytest.approx(4851.7, abs=0.2)
    assert fit.parameters["fwhm_mhz"] > step
    assert fit.parameters["fwhm_mhz"] < float(np.ptp(frequency))
    assert fit.parameters["fwhm_mhz"] == pytest.approx(3.0, rel=0.3)


def test_fit_uses_a_local_window_not_the_whole_sweep(tmp_path):
    """A straight background only describes the neighbourhood of the line."""
    frequency = np.linspace(4600.0, 5000.0, 2000)
    signal = (
        1.0
        + 0.4 * np.sin(2.0 * np.pi * (frequency - 4600.0) / 260.0)
        + 0.5 / (1.0 + ((frequency - 4802.0) / 2.0) ** 2)
    )
    source = write_pair(
        tmp_path / "curved.csv",
        quick_class="QubitSpectroscopy",
        axis_label="Qubit Pulse Frequency",
        axis_unit="MHz",
        x=frequency,
        signal=signal,
    )
    fit = fit_spectroscopy(source, kind="qubit", signal="amplitude")
    assert fit.center_mhz == pytest.approx(4802.0, abs=0.5)
    assert fit.parameters["fwhm_mhz"] < 0.25 * float(np.ptp(frequency))
    assert fit.statistics["points"] < fit.statistics["trace_points"]


# --------------------------------------------------------------------------
# real acquisitions
# --------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="local real-data mirror absent")
def test_repeated_sweeps_of_one_resonator_agree():
    """Five acquisitions of the same resonator, taken across two days."""
    sources = [
        REAL_ROOT / "2026-07-02_MET_ver191/00028 - (ResonatorSpectroscopy)broad scan.csv",
        REAL_ROOT / "2026-07-02_MET_ver191/00029 - (ResonatorSpectroscopy)broad scan.csv",
        REAL_ROOT / "2026-07-03_MET_ver191/00001 - (ResonatorSpectroscopy)broad scan.csv",
    ]
    available = [path for path in sources if path.is_file()]
    if len(available) < 2:
        pytest.skip("resonator repeat mirror absent")
    centers = []
    for path in available:
        matrix = np.atleast_2d(np.loadtxt(path, delimiter=","))
        detection = detect_features(
            matrix[:, 0], matrix[:, 3] + 1j * matrix[:, 4], kind="resonator"
        )
        assert detection.found, f"{path.name} lost its resonance"
        centers.append(detection.best.center_mhz)
    assert np.ptp(centers) < 1.0, f"repeat sweeps disagree: {centers}"


@pytest.mark.skipif(not REAL_ROOT.exists(), reason="local real-data mirror absent")
def test_wide_real_sweeps_are_never_fitted_as_one_enormous_line():
    """These traces used to return linewidths as wide as the sweep itself.

    Declining is an acceptable outcome -- several of these carry only a
    marginal feature -- but returning a "line" a third of the sweep across is
    not.
    """
    from quickexp_v3.notch_fit import fit_spectroscopy_features

    sources = sorted(
        (REAL_ROOT / "2026-07-11_MET_ver191").glob("*(QubitSpectroscopy)*.csv")
    )
    if not sources:
        pytest.skip("qubit sweep mirror absent")
    checked = 0
    for source in sources:
        matrix = np.atleast_2d(np.loadtxt(source, delimiter=","))
        span = float(np.ptp(matrix[:, 0]))
        try:
            fit = fit_spectroscopy_features(source, kind="qubit", signal="IQ")
        except AnalysisError:
            continue
        checked += 1
        assert fit.parameters["fwhm_mhz"] < 0.25 * span, (
            f"{source.name} fitted a line {fit.parameters['fwhm_mhz']:.1f} MHz "
            f"wide across a {span:.1f} MHz sweep"
        )
    assert checked, "no sweep in the mirror produced a fit to check"
