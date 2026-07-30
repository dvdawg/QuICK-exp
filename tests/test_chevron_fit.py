import numpy as np
import pytest
import yaml

from quickexp_v3.chevron_fit import fit_rabi_chevron, fit_ramsey_chevron


def _write_chevron(path, frequencies, times, rates, quick_class, *, fringe=0.0, q_freq=0.0):
    rows = []
    for frequency, rate in zip(frequencies, rates):
        signal = 0.1 + 0.8 * np.exp(-times / (2 * times[-1])) * np.cos(
            2 * np.pi * rate * times + 0.2
        )
        iq = signal * np.exp(0.4j)
        rows.append(
            np.column_stack(
                (
                    np.full_like(times, frequency),
                    times,
                    np.abs(iq),
                    np.angle(iq),
                    iq.real,
                    iq.imag,
                )
            )
        )
    np.savetxt(path, np.vstack(rows), delimiter=",")
    path.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "independent": [
                    ["Qubit Pulse Frequency", "MHz"],
                    ["Delay Time" if quick_class == "T2Ramsey" else "Qubit Pulse Length", "us"],
                ],
                "dependent": [
                    ["Amplitude", ""],
                    ["Phase", "rad"],
                    ["I", ""],
                    ["Q", ""],
                ],
                "parameters": {
                    "quick_experiment": quick_class,
                    "var": {"fringe_freq": fringe, "q_freq": q_freq},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_rabi_and_ramsey_chevrons_recover_frequency_and_sign(tmp_path):
    frequencies = np.linspace(5598.0, 5602.0, 13)
    times = np.linspace(0.0, 3.0, 241)
    rabi_rates = np.sqrt(1.2**2 + (frequencies - 5600.1) ** 2)
    rabi = tmp_path / "rabi.csv"
    _write_chevron(rabi, frequencies, times, rabi_rates, "Rabi")
    fit = fit_rabi_chevron(rabi)
    assert fit.f0_mhz == pytest.approx(5600.1, abs=0.1)
    assert fit.parameters["omega0_mhz"] == pytest.approx(1.2, abs=0.1)

    for sign in (-1, 1):
        fringe = 1.0
        q_frequency = 5600.0
        rates = np.abs(fringe + sign * (q_frequency - frequencies))
        rates = np.maximum(rates, 0.35)
        ramsey = tmp_path / f"ramsey_{sign}.csv"
        _write_chevron(
            ramsey,
            frequencies,
            times,
            rates,
            "T2Ramsey",
            fringe=fringe,
            q_freq=q_frequency,
        )
        result = fit_ramsey_chevron(
            ramsey,
            expected_q_frequency_mhz=q_frequency,
        )
        assert result.parameters["detuning_sign_convention"] == sign
        assert result.qubit_frequency_mhz == pytest.approx(q_frequency, abs=0.15)
