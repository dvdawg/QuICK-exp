import numpy as np

from quickexp_v3.autocal.hp.adaptive import (
    AdaptiveRowScheduler,
    spanning_rows,
    tracked_frequency_axis,
)
from tools.adaptive_zoo import run_adaptive_zoo


def test_spanning_rows_cover_bounds_and_center():
    rows = spanning_rows((-0.3, 0.3), 5)
    assert np.allclose(rows, [-0.3, -0.15, 0.0, 0.15, 0.3])


def test_scheduler_selects_unmeasured_informative_rows_up_to_cap():
    scheduler = AdaptiveRowScheduler((-0.3, 0.3), initial_rows=5, max_rows=7)
    selected = []
    while not scheduler.done:
        value = scheduler.next_row()
        assert value not in selected
        selected.append(value)
        center = 6884.0 + 0.6 * np.cos(2.0 * np.pi * (value + 0.07) / 0.18)
        scheduler.record(value, center_mhz=center, trackable=True)
    assert len(selected) == 7
    assert set(np.round(selected[:5], 6)) == set(
        np.round(spanning_rows((-0.3, 0.3), 5), 6)
    )


def test_scheduler_aborts_a_doomed_map_after_reviewed_row_count():
    scheduler = AdaptiveRowScheduler(
        (-0.3, 0.3),
        initial_rows=5,
        max_rows=7,
        abort_after_rows=5,
    )
    for _index in range(5):
        value = scheduler.next_row()
        scheduler.record(value, center_mhz=None, trackable=False)
    assert scheduler.aborted
    assert scheduler.done
    assert "trackable" in scheduler.abort_reason


def test_tracking_axis_recenters_and_respects_hardware_bounds():
    axis = tracked_frequency_axis(
        previous_center_mhz=5000.2,
        span_mhz=4.0,
        points=101,
        bounds=(5000.0, 9000.0),
    )
    assert axis.size == 101
    assert axis[0] == 5000.0
    assert axis[-1] == 5004.0


def test_scheduler_round_trip_preserves_abort_and_rows():
    scheduler = AdaptiveRowScheduler((-1.0, 1.0), 3, 5, 3)
    for _index in range(3):
        value = scheduler.next_row()
        scheduler.record(value, center_mhz=10.0 + value, trackable=True)
    restored = AdaptiveRowScheduler.from_dict(scheduler.as_dict())
    assert restored.as_dict() == scheduler.as_dict()
    assert restored.next_row() == scheduler.next_row()


def test_adaptive_zoo_uses_at_most_sixty_percent_without_lookup_loss():
    metrics, results = run_adaptive_zoo(count=7, seed=0)
    assert len(results) == 7
    assert metrics["maximum_row_fraction"] <= 0.60
    assert metrics["noninferior_fraction"] == 1.0
