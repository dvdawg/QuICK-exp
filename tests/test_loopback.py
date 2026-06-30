"""Tests for the pure loopback-calibration helpers (labreadout.loopback).

These encode the two things that dominated the 2026-06-29 session: finding the
readout-window delay (r_offset, starting from 0) and choosing a drive/receive
level that sits above noise but below ADC over-range (the ~30 level-test files).
"""

import numpy as np
import pytest

from labreadout import loopback


def _pulse_trace(onset=0.5, width=0.5, baseline=2.0, height=40.0, n=600, span=2.0):
    """Time trace with a flat baseline and a raised plateau (the returned pulse)."""
    t = np.linspace(0, span, n)
    amp = np.full(n, baseline)
    amp[(t >= onset) & (t < onset + width)] = baseline + height
    return t, amp


# -- offset finding ---------------------------------------------------------- #
def test_find_offset_locates_pulse_onset():
    t, amp = _pulse_trace(onset=0.5)
    assert loopback.find_offset(t, amp) == pytest.approx(0.5, abs=0.05)


def test_find_offset_handles_a_later_pulse():
    t, amp = _pulse_trace(onset=1.1)
    assert loopback.find_offset(t, amp) == pytest.approx(1.1, abs=0.05)


def test_find_offset_flat_trace_returns_zero():
    t = np.linspace(0, 2, 400)
    amp = np.full(400, 3.0)  # no pulse above baseline
    assert loopback.find_offset(t, amp) == 0.0


# -- level selection --------------------------------------------------------- #
def test_select_power_picks_highest_safe_level():
    powers = [-40, -30, -20, -10, 0]
    peaks = [50, 150, 500, 1200, 3000]  # 0 dB rails past full scale
    best, status = loopback.select_power(powers, peaks, fullscale=1500, headroom=0.8)
    assert best == -10  # highest power whose peak stays <= 0.8*1500 = 1200
    assert status == "ok"


def test_select_power_all_over_range_backs_off_to_lowest():
    best, status = loopback.select_power([-10, 0], [2000, 3000], fullscale=1500)
    assert best == -10  # the lowest, since everything else over-ranges
    assert status == "all_over_range"


def test_select_power_all_weak_flags_weak():
    # Every level is far below full scale -- signal is weak (possible path issue).
    best, status = loopback.select_power([-40, -30, -20], [5, 8, 12], fullscale=1500)
    assert status == "weak"


# -- calibration object ------------------------------------------------------ #
def test_calibration_commits_offset_and_power():
    cal = loopback.LoopbackCalibration(
        r_offset=0.52, r_power=-10, peak=1180.0, fullscale=1500.0,
        powers=[-20, -10, 0], peaks=[500, 1180, 3000], valid=True,
    )
    assert cal.valid
    assert "offset" in cal.suggestion().lower()
    # commit_map-style attributes exist for accept()
    assert cal.r_offset == 0.52 and cal.r_power == -10
