from pathlib import Path

import numpy as np
import pytest
import yaml

from quickexp_v3.punchout_fit import fit_punchout, punchout_model


REAL_SOURCE = Path(
    "/Users/dvdkm/Documents/code/qdg/data/2026-07-21_MET_ver191_qubit3/"
    "00023 - (ResonatorSpectroscopy)Punchout.csv"
)


def _write_punchout(path, powers, frequencies, centers, noise=0.0):
    rng = np.random.default_rng(2)
    rows = []
    for power, center in zip(powers, centers):
        amplitude = -12.0 / (1.0 + ((frequencies - center) / 0.25) ** 2)
        amplitude += rng.normal(0.0, noise, frequencies.size)
        iq = 10 ** (amplitude / 20.0)
        rows.append(
            np.column_stack(
                (
                    np.full_like(frequencies, power),
                    frequencies,
                    amplitude,
                    np.zeros_like(frequencies),
                    iq,
                    np.zeros_like(frequencies),
                )
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [
                    ["Readout Pulse Power", "dB"],
                    ["Readout Pulse Frequency", "MHz"],
                ],
                "dependent": [
                    ["Amplitude", "dB"],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": "ResonatorSpectroscopy"
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_punchout_resolves_synthetic_and_rejects_coarse_grid(tmp_path):
    powers = np.linspace(-40.0, -5.0, 12)
    frequencies = np.arange(6881.0, 6887.01, 0.1)
    centers = punchout_model(powers, 6884.0, 1.2, -20.0, 4.0)
    source = tmp_path / "resolved.csv"
    _write_punchout(source, powers, frequencies, centers)
    fit = fit_punchout(source)
    assert fit.status == "resolved"
    assert fit.parameters["punchout_shift_mhz"] == pytest.approx(1.2, rel=0.15)
    assert fit.parameters["transition_power_db"] == pytest.approx(-20.0, rel=0.15)

    coarse_frequency = np.arange(6880.0, 6889.0, 1.2)
    coarse = tmp_path / "coarse.csv"
    _write_punchout(coarse, powers, coarse_frequency, centers)
    assert fit_punchout(coarse).status == "unresolved"


@pytest.mark.skipif(not REAL_SOURCE.exists(), reason="local real-data mirror absent")
def test_real_coarse_punchout_reports_unresolved():
    assert fit_punchout(REAL_SOURCE).status == "unresolved"

