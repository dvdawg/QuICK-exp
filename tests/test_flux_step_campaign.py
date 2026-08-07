from pathlib import Path
import shutil

import numpy as np
import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_step_campaign import (
    LongTimescaleFluxParameters,
    _fast_line_levels,
    _readout_return_gain,
    _resolve_campaign,
    _resolve_flux_command,
    run_long_timescale_flux_step,
)


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def test_offline_phi0_fallback_applies_commissioning_scale():
    parameters = LongTimescaleFluxParameters(
        baseline_phi0=-0.127,
        step_phi0=-0.217,
        step_scale=0.20,
    )

    resolved = _resolve_flux_command(parameters, record=None)

    assert resolved.baseline_z == pytest.approx(-0.127)
    assert resolved.commanded_step_z == pytest.approx(-0.217 * 0.20)
    assert resolved.source == "offline Phi0-as-Z fallback"


@pytest.mark.parametrize("fabric_mhz", [430.08, 599.04])
def test_default_z_set_pulse_is_at_least_three_fabric_clocks(fabric_mhz):
    parameters = LongTimescaleFluxParameters()

    assert parameters.z_set_length_us * fabric_mhz >= 3.0


def test_custom_centers_and_local_flux_parameters_resolve_without_lookup():
    parameters = LongTimescaleFluxParameters(
        baseline_z=-0.11,
        commanded_step_z=-0.02,
        probe_times_us=[0.01, 0.1, 1.0],
        q_frequency_offsets_mhz=np.linspace(-4.0, 4.0, 9),
        q_frequency_centers_mhz=[5200.0, 5195.0, 5190.0],
        q_nyquist_zone=2,
    )

    resolved = _resolve_campaign(parameters, repository())

    np.testing.assert_allclose(
        resolved.q_frequency_centers_mhz,
        [5200.0, 5195.0, 5190.0],
    )
    assert resolved.flux.baseline_z == pytest.approx(-0.11)
    assert resolved.flux.commanded_step_z == pytest.approx(-0.02)
    assert resolved.q_nyquist_zone == 2


def test_center_count_must_match_probe_times():
    parameters = LongTimescaleFluxParameters(
        baseline_z=-0.11,
        commanded_step_z=-0.02,
        probe_times_us=[0.01, 0.1, 1.0],
        q_frequency_offsets_mhz=np.linspace(-4.0, 4.0, 9),
        q_frequency_centers_mhz=[5200.0, 5195.0],
    )

    with pytest.raises(ConfigError, match="one per probe time"):
        _resolve_campaign(parameters, repository())


def test_custom_step_must_stay_inside_fast_line_limits():
    parameters = LongTimescaleFluxParameters(
        baseline_z=-0.11,
        commanded_step_z=-0.81,
        probe_times_us=[0.01],
        q_frequency_offsets_mhz=np.linspace(-4.0, 4.0, 9),
        q_frequency_centers_mhz=5200.0,
    )

    with pytest.raises(ConfigError, match="outside the fast-line Z limits"):
        _resolve_campaign(parameters, repository())


def test_readout_return_uses_configured_probe_timing():
    parameters = LongTimescaleFluxParameters(
        q_length_us=0.060,
        post_probe_to_return_us=0.090,
        bias_tee_tau_us=20.0,
    )

    gain = _readout_return_gain(parameters, 1.0, -0.2)

    assert gain == pytest.approx(-0.2 * (1.0 - np.exp(-1.15 / 20.0)))


def single_line_parameters(**changes):
    base = {
        "baseline_on_fast_line": True,
        "first_uncompensated_pass": False,
        "baseline_z": -0.0798,
        "commanded_step_z": 0.0189,
        "probe_times_us": [0.025, 1.0],
        "q_frequency_offsets_mhz": np.linspace(-4.0, 4.0, 9),
        "q_frequency_centers_mhz": 4577.0,
        "q_nyquist_zone": 1,
    }
    base.update(changes)
    return LongTimescaleFluxParameters(**base)


def test_single_line_mode_commands_absolute_levels():
    parameters = single_line_parameters()
    campaign = _resolve_campaign(parameters, repository())

    levels = _fast_line_levels(parameters, campaign.flux)

    assert levels.step == pytest.approx(-0.0798 + 0.0189)
    assert levels.readout_rest == pytest.approx(-0.0798)
    assert levels.idle == pytest.approx(-0.0798)


def test_single_line_readout_return_holds_the_baseline():
    parameters = single_line_parameters()

    gain = _readout_return_gain(parameters, 1.0, 0.0189, -0.0798)

    assert gain == pytest.approx(-0.0798)


def test_two_path_mode_still_rests_the_fast_line_at_zero():
    parameters = LongTimescaleFluxParameters(
        baseline_z=-0.11,
        commanded_step_z=-0.02,
        probe_times_us=[0.01, 1.0],
        q_frequency_offsets_mhz=np.linspace(-4.0, 4.0, 9),
        q_frequency_centers_mhz=5200.0,
    )
    campaign = _resolve_campaign(parameters, repository())

    levels = _fast_line_levels(parameters, campaign.flux)

    assert levels.step == pytest.approx(-0.02)
    assert levels.readout_rest == 0.0
    assert levels.idle == 0.0


def test_single_line_mode_rejects_the_bias_tee_return_model():
    parameters = single_line_parameters(first_uncompensated_pass=True)

    with pytest.raises(ConfigError, match="no bias-tee droop"):
        _resolve_campaign(parameters, repository())


def test_single_line_mode_rejects_a_conflicting_idle_level():
    parameters = single_line_parameters(z_idle_gain=-0.05)

    with pytest.raises(ConfigError, match="leave z_idle_gain at 0.0"):
        _resolve_campaign(parameters, repository())


def test_single_line_step_level_is_range_checked_absolutely():
    # The delta alone is well inside +/-0.8; only baseline + step leaves it.
    parameters = single_line_parameters(baseline_z=-0.79, commanded_step_z=-0.02)

    with pytest.raises(ConfigError, match="outside the fast-line Z limits"):
        _resolve_campaign(parameters, repository())


def test_single_line_campaign_runs_offline_without_a_baseline_latch(tmp_path):
    for stem in ("hardware", "calibration", "presets"):
        shutil.copyfile(
            ROOT / f"{stem}.example.yml",
            tmp_path / f"{stem}.example.yml",
        )
    parameters = single_line_parameters(
        use_accepted_resonator_flux_fit=False,
        readout_relax_us=1.0,
        hard_avg=2,
        show_plot=False,
        campaign_manifest=tmp_path / "campaign.json",
    )

    results = run_long_timescale_flux_step(tmp_path, parameters)

    assert len(results) == 2


def test_small_custom_campaign_runs_offline(tmp_path):
    for stem in ("hardware", "calibration", "presets"):
        shutil.copyfile(
            ROOT / f"{stem}.example.yml",
            tmp_path / f"{stem}.example.yml",
        )
    parameters = LongTimescaleFluxParameters(
        baseline_z=-0.10,
        commanded_step_z=-0.01,
        probe_times_us=[0.01, 0.1],
        q_frequency_offsets_mhz=np.linspace(-4.0, 4.0, 9),
        q_frequency_centers_mhz=[5200.0, 5198.0],
        q_nyquist_zone=2,
        use_accepted_resonator_flux_fit=False,
        readout_relax_us=1.0,
        hard_avg=2,
        first_uncompensated_pass=False,
        show_plot=False,
        campaign_manifest=tmp_path / "campaign.json",
    )

    results = run_long_timescale_flux_step(tmp_path, parameters)

    assert len(results) == 2
    assert [result.data.points for result in results] == [9, 9]
