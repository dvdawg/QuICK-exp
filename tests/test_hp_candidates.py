from pathlib import Path

import numpy as np

from quickexp_v3.autocal.hp.candidates import Candidate, extract_candidates
from quickexp_v3.notch_fit import fit_spectroscopy_features
from quickexp_v3.zoo import generate_chip
from tools.baseline_legacy import acquire_qubit_trace


def _fit_for(chip, scratch: Path):
    csv_path = acquire_qubit_trace(chip, scratch)
    return fit_spectroscopy_features(csv_path, kind="qubit", signal="amplitude")


def test_two_feature_trace_yields_two_real_candidates_plus_null(tmp_path):
    chip = generate_chip("f02_shadow", seed=4)
    candidates = extract_candidates(_fit_for(chip, tmp_path))
    real = [item for item in candidates if not item.is_null]
    assert len(real) >= 2
    assert candidates[-1].is_null


def test_candidates_are_ranked_by_prominence(tmp_path):
    chip = generate_chip("tls", seed=6)
    candidates = extract_candidates(_fit_for(chip, tmp_path))
    real = [item for item in candidates if not item.is_null]
    contrasts = [abs(item.contrast) for item in real]
    assert contrasts == sorted(contrasts, reverse=True)
    assert [item.rank for item in real] == list(range(len(real)))


def test_weak_candidates_are_never_thresholded_away(tmp_path):
    chip = generate_chip("clean", seed=9)
    candidates = extract_candidates(_fit_for(chip, tmp_path), max_candidates=5)
    assert len([item for item in candidates if not item.is_null]) >= 1
    assert all(
        np.isfinite(item.local_snr) for item in candidates if not item.is_null
    )


def test_candidate_ids_are_stable_across_repeated_extraction(tmp_path):
    chip = generate_chip("clean", seed=9)
    fit = _fit_for(chip, tmp_path)
    first = extract_candidates(fit)
    second = extract_candidates(fit)
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]


def test_null_candidate_carries_the_detectability_limit(tmp_path):
    fit = _fit_for(generate_chip("clean", seed=12), tmp_path)
    null = extract_candidates(fit)[-1]
    assert isinstance(null, Candidate)
    assert null.is_null
    assert np.isnan(null.center_mhz)
    assert null.statistics["detectable_contrast"] > 0.0
