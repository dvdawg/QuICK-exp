from pathlib import Path
import shutil

import numpy as np
import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_step_campaign import (
    LongTimescaleFluxParameters,
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
