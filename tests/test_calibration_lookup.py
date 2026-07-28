from pathlib import Path

import numpy as np
import pytest

from quickexp_v3.config import ConfigRepository
from quickexp_v3.ide import resonator_frequency_from_flux


ROOT = Path(__file__).resolve().parents[1]


def repository():
    return ConfigRepository.from_files(
        ROOT / "hardware.example.yml",
        ROOT / "calibration.example.yml",
        ROOT / "presets.example.yml",
    )


def test_resonator_flux_lookup_uses_accepted_notebook_cosine():
    frequencies = resonator_frequency_from_flux(
        repository(),
        np.array([-0.4, 0.0, 0.4]),
    )
    assert frequencies.shape == (3,)
    assert np.all(np.isfinite(frequencies))


def test_resonator_flux_lookup_refuses_extrapolation():
    with pytest.raises(RuntimeError, match="outside accepted"):
        resonator_frequency_from_flux(repository(), 0.41)
