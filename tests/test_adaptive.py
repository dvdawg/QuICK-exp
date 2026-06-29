import numpy as np
import pytest

from labreadout import adaptive


def test_window_is_centered_on_the_feature():
    lo, hi = adaptive.next_window(center=6584.5, width=0.2)
    assert (lo + hi) / 2 == pytest.approx(6584.5)


def test_span_scales_with_feature_width():
    lo, hi = adaptive.next_window(center=100.0, width=2.0, span_factor=8)
    assert (hi - lo) == pytest.approx(16.0)


def test_span_respects_minimum():
    lo, hi = adaptive.next_window(center=100.0, width=1e-6, min_span=0.5)
    assert (hi - lo) == pytest.approx(0.5)


def test_window_clips_to_allowed_bounds():
    lo, hi = adaptive.next_window(
        center=6500.0, width=50.0, span_factor=8, bounds=(6400.0, 6520.0)
    )
    assert lo >= 6400.0 and hi <= 6520.0


def test_step_gives_requested_point_count():
    # Inclusive linspace: N points over a span S have step S/(N-1).
    step = adaptive.step_for_points(span=16.0, points=1601)
    assert step == pytest.approx(0.01)


def test_scan_array_covers_window_inclusive():
    arr = adaptive.scan_array(center=6584.5, width=0.5, points=101, span_factor=8)
    assert arr[0] == pytest.approx(6584.5 - 2.0)
    assert arr[-1] == pytest.approx(6584.5 + 2.0)
    assert len(arr) == 101


def test_converged_when_width_below_resolution():
    assert adaptive.converged(width=0.005, step=0.01) is True
    assert adaptive.converged(width=1.0, step=0.01) is False
