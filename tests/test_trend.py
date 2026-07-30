from pathlib import Path

import numpy as np

from quickexp_v3.trend import (
    TREND_COLUMNS,
    append_refit_trend,
    harvest_calibration_trends,
    read_trend,
    source_changed,
    sparkline,
    trend_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_harvest_repo_flux_lookup_has_three_revision_ordered_points(tmp_path):
    generated = harvest_calibration_trends(
        ROOT / "calibration.yml",
        tmp_path / "trends",
    )
    path = generated["lookups.resonator_vs_flux"]
    rows = read_trend(path)

    assert path.name == "lookups.resonator_vs_flux.csv"
    assert len(rows) == 3
    assert tuple(rows[0]) == TREND_COLUMNS
    assert [row["origin"] for row in rows] == ["history", "history", "accepted"]
    assert [int(row["calibration_revision"]) for row in rows] == [2, 3, 4]
    assert np.allclose(
        [float(row["value"]) for row in rows],
        [6884.186011, 6884.186010629018, 6884.186010629018],
    )
    assert float(rows[-1]["uncertainty"]) > 0


def test_refit_append_source_manifest_statistics_and_sparkline(tmp_path):
    source = tmp_path / "native.csv"
    source.write_text("0,1\n1,2\n", encoding="utf-8")
    cache = tmp_path / "trends"
    path = append_refit_trend(
        cache,
        "defaults.q_freq",
        value=5000.0,
        uncertainty=0.1,
        source_csv=source,
        time_utc="2026-01-01T00:00:00+00:00",
        calibration_revision=2,
    )
    append_refit_trend(
        cache,
        "defaults.q_freq",
        value=5000.1,
        uncertainty=0.1,
        source_csv=source,
        time_utc="2026-01-02T00:00:00+00:00",
        calibration_revision=2,
    )
    append_refit_trend(
        cache,
        "defaults.q_freq",
        value=5002.0,
        uncertainty=0.1,
        source_csv=source,
        time_utc="2026-01-03T00:00:00+00:00",
        calibration_revision=2,
    )
    rows = read_trend(path)
    statistics = trend_statistics(rows, settings=("same", "same", "changed"))

    assert statistics.count == 3
    np.testing.assert_allclose(
        statistics.repeatability_sigma,
        0.1 / np.sqrt(2.0),
        rtol=1e-10,
    )
    assert statistics.drift_per_day > 0
    assert len(statistics.steps) == 1
    assert not source_changed(rows[0])
    source.write_text("changed\n", encoding="utf-8")
    assert source_changed(rows[0])
    assert sparkline([1.0, 1.0, 2.0]).endswith("█")
