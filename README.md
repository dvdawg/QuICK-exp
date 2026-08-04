# QuICK-exp

QuICK-exp is a superconducting-qubit measurement workflow with shared YAML configuration, port verification, calibration precedence, recovery, and held-flux safety.

## First use

1. Open `experiments/01_configure_experiment.py`, edit shared values such as
  `q_freq`, `r_freq`, the native data directory, connection, or logical
   channels, and run it once with `WRITE_CHANGES = False`.
2. When the preview is correct, set `WRITE_CHANGES = True`, run it, then set it
  back to `False`.
3. Run `00_connect_and_ports.py` to connect and verify the live `soccfg`
  against the `r/rr/q/z` map.
4. Run 02 to verify the raw ADC trace/readout trigger offset before proceeding
  through spectroscopy and time-domain measurements.

Installation is optional because each launcher adds the project root to
`sys.path`. To run tests:

```sh
python -m pytest -q
```



## Experimental order


| File                                       | Purpose                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `00_connect_and_ports.py`                  | connect, print `soccfg`, verify routing                                   |
| `01_configure_experiment.py`               | edit/validate/write common YAML settings                                  |
| `02_raw_adc_loopback.py`                   | raw decimated trace and readout-offset check                              |
| `05a_resonator_spectroscopy_vs_power.py`   | native power-by-frequency punchout                                        |
| `05b_resonator_spectroscopy_fixed_flux.py` | resonator scan at held Z                                                  |
| `05c_resonator_spectroscopy_vs_flux.py`    | resonator spectroscopy versus Z                                           |
| `05d_fit_resonator_vs_flux.py`             | fit/accept the resonator-versus-Z readout curve                           |
| `06a_qubit_spectroscopy.py`                | qubit spectroscopy                                                        |
| `06b_qubit_spectroscopy_vs_flux.py`        | qubit spectroscopy versus Z                                               |
| `06c_qubit_spectroscopy_vs_gain.py`        | gain-by-frequency power spectroscopy                                      |
| `06f_qubit_spectroscopy_zpa.py`            | authored native frequency-by-Z map (simulation until hardware gates pass) |
| `07a_rabi_chevron_duration.py`             | frequency-by-duration Rabi chevron                                        |
| `07b_rabi_chevron_amplitude.py`            | frequency-by-gain Rabi chevron                                            |
| `08a_time_rabi.py`                         | pulse-duration Rabi                                                       |
| `08b_power_rabi.py`                        | pulse-gain Rabi                                                           |
| `08c_fit_rabi.py`                          | fit/accept a time- or power-Rabi pi value                                 |
| `09a_iq_blobs.py`                          | ground/excited IQ clouds                                                  |
| `10a_readout_frequency_optimization.py`    | dispersive readout scan                                                   |
| `11_t1.py`                                 | energy relaxation                                                         |
| `11c_t1_vs_flux.py`                        | authored finite-Z-pulse T1 map (simulation until hardware gates pass)     |
| `12_ramsey_chevron.py`                     | frequency-by-delay Ramsey chevron                                         |
| `13a_ramsey.py`                            | Ramsey dephasing                                                          |
| `14_echo.py`                               | Hahn echo / fixed-cycle CPMG                                              |
| `16_two_photon_spectroscopy.py`            | high-power two-photon search                                              |
| `90_measurement_queue.py`                  | run selected numbered files sequentially                                  |
| `91_autocal.py`                            | run/resume a policy-governed calibration target                           |
| `92_review_proposals.py`                   | inspect, promote, or reject inert proposals                               |
| `95_device_report.py`                      | local calibration/trend/QC report                                         |


The order is a suggested workflow. Adapt it to the device, firmware, and  
calibration state in use.

## Fitting and Write Latches

Every fitter has the same `INPUT_CSV` setting:

```python
# Newest matching one-dimensional native Quick CSV/YML pair:
INPUT_CSV = None

# Or one specific run:
INPUT_CSV = r"/path/to/data/run.csv"
```

Point to the CSV itself, not its directory. The paired YML must have the same base filename and remain beside it. Use a raw string (`r"..."`), forward slashes, or a raw UNC path:

```python
INPUT_CSV = r"\\file-server\share\path\to\run.csv"
```

When `INPUT_CSV = None`, two-dimensional files and files belonging to another Quick experiment class are ignored.

All write latches default to `False`. First inspect the plotted fit, residuals, uncertainty, and printed quality gates. Then set:

```python
WRITE_ACCEPTED_FIT = True
```

The update is quality-gated and written atomically to `calibration.yml`; a superseded record is retained in `history`.

Some fitting scripts have unique behaviors; for example, Ramsey always reports both possible drive-frequency corrections because a single scalar Ramsey trace does not independently determine the detuning sign. Writing `q_freq` therefore requires both:

```python
WRITE_ACCEPTED_FIT = True
UPDATE_Q_FREQUENCY = True
# Choose a sign, then verify it experimentally.
Q_FREQUENCY_CORRECTION_SIGN = +1
```

In many cases fits will be correct but fail acceptance gates. In this case, the file will refuse to write the accepted fit until the user manually overrides this using `FORCE_WRITE`. 

```
FORCE_WRITE = True
```

After using the write latches, it is always good practice to set them back to `FALSE` and save to avoid accidental writes.

## Configuration and data

Settings resolve in this order:

```text
hardware defaults < accepted calibration < preset < launcher overrides
```

- `hardware.yml` contains the connection, routing, bounds, shared defaults, and
native Quick output directory.
- `calibration.yml` contains accepted values, history, provenance, and inert
autocal proposals. Only accepted records participate in resolution. The 01
editor updates accepted `r_freq`, `q_freq`, and `r_offset` records with their
matching defaults so precedence stays intuitive.
- `presets.yml` contains reusable experiment starting points.

Live runs use only Quick's native numbered CSV/YML Saver. Ordinary run titles
identify the experiment and relevant sweep parameters.

Held-Z maps are combined into one native pair, for example
`ResVsZ_held_bias.csv` or `QubitSpecVsZ_fitted_readout.csv`. No new
`runs/.../manifest.json` or `raw.npz` tree is created. The existing `runs`
directory is retained as legacy acquired data.

Quick progress is enabled by `qick.show_progress: true`.
`qick.progress_mode: terminal` is recommended for terminal or IDE
execution; use `notebook` inside Jupyter. Each native `quick.Sweep` bar shows
percentage, elapsed/estimated time, and iterations per second. Multi-Z scans
use the same renderer for outer rows.

Before a live run applies held Z, QuICK-exp checks Gaussian qubit-pulse memory against the connected generator's `f_fabric`, `samps_per_clk`, and `maxlen`. The check also accounts for the cumulative Gaussian envelopes used by T1, Ramsey, Echo, IQ Scatter, and Dispersive Spectroscopy.

## Live hardware behavior

Logical `r`, `q`, `rr`, and `z` roles and their generator, DAC, readout, and ADC
indexes are installation-specific and configured in `hardware.yml`. Run
`00_connect_and_ports.py` to verify that routing before acquisition.

At zero Z, fixed-Z scripts use the generator reset performed during connection
and skip the auxiliary acquisition. Nonzero fixed-Z scripts establish the held
bias on the configured Z output using scalar settings independent of the main
sweep. Connections, acquisitions, and closes clear persistent QICK streamer
state so an interrupted process cannot affect the next sweep. A retry resets
the configured generators and reapplies held Z. A normal exit attempts to
return Z to zero bias; a lost hardware link can prevent parking.

RF-board programming remains disabled until its attenuation and filter settings  
have been reviewed for the active hardware and wiring.

## Using the Measurement Queue

Open `90_measurement_queue.py`, list enabled files in `TASKS`, and run it. The default queue runs Time Rabi followed by Power Rabi. `SHOW_PLOT=False` lets the next task begin without waiting for a plot window to close. Each task still uses its own normal `main()`, Quick CSV/YML Saver, connection, flux parking, and cleanup. A task may override an `EDIT THESE` value without modifying the source:

```python
{
    "file": "08a_time_rabi.py",
    "label": "Time Rabi at alternate gain",
    "enabled": True,
    "settings": {"Q_GAIN": ...},  # replace with a safe value for the device
}
```

Duplicate files are allowed. `STOP_ON_ERROR=True` stops at the first failure;  
set it to `False` to continue and receive a `completed_with_errors` summary.