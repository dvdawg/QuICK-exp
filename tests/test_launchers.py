from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_DIR = ROOT / "experiments"
EXPECTED = [
    "00_connect_and_ports.py",
    "01_configure_experiment.py",
    "02_raw_adc_loopback.py",
    "02b_fit_loopback.py",
    "05a_resonator_spectroscopy_vs_power.py",
    "05b_resonator_spectroscopy_fixed_flux.py",
    "05c_resonator_spectroscopy_vs_flux.py",
    "05d_fit_resonator_vs_flux.py",
    "05e_fit_resonator_spectroscopy.py",
    "05f_fit_punchout.py",
    "06a_qubit_spectroscopy.py",
    "06b_qubit_spectroscopy_vs_flux.py",
    "06c_qubit_spectroscopy_vs_gain.py",
    "06d_fit_qubit_spectroscopy.py",
    "06e_fit_qubit_vs_flux.py",
    "06f_qubit_spectroscopy_zpa.py",
    "06g_design_qubit_sweep_path.py",
    "07a_rabi_chevron_duration.py",
    "07b_rabi_chevron_amplitude.py",
    "07c_fit_rabi_chevron.py",
    "08a_time_rabi.py",
    "08b_power_rabi.py",
    "08c_fit_rabi.py",
    "09a_iq_blobs.py",
    "09b_fit_iq_blobs.py",
    "10a_readout_frequency_optimization.py",
    "10b_fit_readout_optimization.py",
    "11_t1.py",
    "11b_fit_t1.py",
    "11c_t1_vs_flux.py",
    "12_ramsey_chevron.py",
    "12b_fit_ramsey_chevron.py",
    "13a_ramsey.py",
    "13b_fit_ramsey.py",
    "14_echo.py",
    "14b_fit_echo.py",
    "16_two_photon_spectroscopy.py",
    "17a_flux_step_spectroscopy.py",
    "17b_fit_flux_iir.py",
    "17c_cryoscope.py",
    "17d_fit_flux_fir.py",
    "18a_resonator_flux_transient.py",
    "90_measurement_queue.py",
    "91_autocal.py",
    "92_review_proposals.py",
    "95_device_report.py",
]


def test_numbered_launcher_set_is_explicit_and_opx_ordered():
    assert sorted(path.name for path in LAUNCHER_DIR.glob("*.py")) == EXPECTED


def test_every_launcher_imports_offline_and_exposes_main():
    for filename in EXPECTED:
        namespace = runpy.run_path(
            str(LAUNCHER_DIR / filename),
            run_name=f"launcher_{filename[:-3]}",
        )
        if filename == "01_configure_experiment.py":
            # This is an operator-controlled latch and may intentionally be
            # True in a live workspace; importing must not execute main().
            assert isinstance(namespace["WRITE_CHANGES"], bool)
        else:
            assert isinstance(namespace["LIVE_HARDWARE"], bool)
        assert callable(namespace["main"])
