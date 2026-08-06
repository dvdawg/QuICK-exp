# QuICK-exp

QuICK-exp is a superconducting-qubit measurement workflow with shared YAML configuration, port verification, calibration precedence, recovery, and held-flux safety.

## Startup

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



## Experiment Listing


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
| `06g_design_qubit_sweep_path.py`           | design a row-dependent sweep path from a prior two-axis map               |
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


The order is a suggested workflow. Adapt it to the device, firmware, and calibration state in use.

## Fitting and Write Latches



### Using Fitters

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

It's also useful to limit the axes that are used for fitting to avoid strange features or any other data that might distort the fit. Some fitters such as `06e_fit_qubit_vs_flux.py`, can restrict axes independently so a broad map can be fitted around one feature. `None` uses the full acquired axis, and bounds are inclusive:

```python
FIT_FREQUENCY_WINDOW_MHZ = (MIN_MHZ, MAX_MHZ)
FIT_FLUX_WINDOW_Z = (MIN_Z_GAIN, MAX_Z_GAIN)
```



### Write Latches

After fitting, it's useful to use the fitted model for future measurements (i.e. for resonator frequency vs flux dependence). All write latches default to `False`. First inspect the plotted fit, residuals, uncertainty, and printed quality gates. Then set:

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

## Measurement Features

### Following a Feature with a Custom Sweep Path

A rectangular scan spends most of its acquisition time far from the feature of interest. `06g_design_qubit_sweep_path.py` runs after a broad two-dimensional scan and stores one or more selected inner-axis intervals per outer-axis row, so a later run measures only the corridor containing the feature. It infers generic parameter names from the background axes; for a Z-by-qubit-frequency map it produces a `z_gain`/`q_freq` path that `06b` can run.

Two design modes are available:

```python
# Fit the spectroscopy ridge and add a margin on both sides.
PATH_METHOD = "fit_margin"
FIT_MARGIN_MHZ = 10.0  # or (below_mhz, above_mhz)

# Draw a polygon directly over the phase/amplitude map from the prior sweep.
PATH_METHOD = "ui_polygon"
BACKGROUND_SIGNAL = "phase"
```

In UI mode, click around the feature to close a polygon, inspect the white previewed region, then choose **Use region**. The prior sweep remains visible as the background throughout editing.

Concave polygons are retained as one or more disjoint intervals at each outer row. If a vertical slice crosses two selected frequency bands with a gap between them, the saved path jumps between those bands; it does not sweep the unused frequencies inside the overall lower/upper envelope.

By default each selected row keeps the frequency spacing of the background map, so narrow rows contain fewer points without changing the MHz resolution. To choose a different spacing:

```python
FREQUENCY_RESOLUTION_MHZ = None  # preserve the background resolution
# FREQUENCY_RESOLUTION_MHZ = 0.5  # optional manual override
```

To acquire the saved path, point `06b_qubit_spectroscopy_vs_flux.py` at the YAML:

```python
SWEEP_PATH_YML = (
    PROJECT_ROOT / "analysis_cache" / "qubit_flux_frequency_sweep_path.yml"
)
```

`06b` then maps each saved `z_gain` through the held-flux controller, runs that row's saved `q_freq` values, and retains fitted readout-frequency tracking. All rows are still written as one native CSV/YML pair even though the path is not rectangular. Setting `SWEEP_PATH_YML = None` retains the original rectangular native sweep.

Analyze either kind of result with `06e_fit_qubit_vs_flux.py`. With `INPUT_CSV = None`, it selects the latest `QubitSpecVsZ` run. Custom-path rows are kept at their acquired lengths and split at unswept gaps; the fitter finds several local candidates in each acquired band and selects one globally transmon-like ridge. The left plot displays the actual acquired points rather than filling the holes. Leave both fit-window settings at `None` to use every acquired point, or set either one to restrict the analysis further.

At the library level, `SweepPath` and `run_sweep_path` are axis-generic: the YAML declares `outer.name` and `inner.name`, and the runner maps those names to the corresponding experiment overrides. Launcher-specific code only validates that a path contains axes appropriate for that experiment and selects any required physical outer-axis control, such as held flux for `z_gain`.

### Using the Measurement Queue

Open `90_measurement_queue.py`, list enabled files in `TASKS`, and run it. The default queue runs Time Rabi followed by Power Rabi. `SHOW_PLOT=False` lets the next task begin without waiting for a plot window to close. Each task still uses its own normal `main()`, Quick CSV/YML Saver, connection, flux parking, and cleanup. A task may override an `EDIT THESE` value without modifying the source:

```python
{
    "file": "08a_time_rabi.py",
    "label": "Time Rabi at alternate gain",
    "enabled": True,
    "settings": {"Q_GAIN": ...},  # replace with a safe value for the device
}
```

Duplicate files are allowed. `STOP_ON_ERROR=True` stops at the first failure; set it to `False` to continue and receive a `completed_with_errors` summary.

## Configuration and data

Settings resolve in this order:

```text
hardware defaults < accepted calibration < preset < launcher overrides
```

- `hardware.yml` contains the connection, routing, bounds, shared defaults, and native Quick output directory.
- `calibration.yml` contains accepted values, history, provenance, and inert autocal proposals. Only accepted records participate in resolution. The 01 editor updates accepted `r_freq`, `q_freq`, and `r_offset` records with their matching defaults so precedence stays intuitive.
- `presets.yml` contains reusable experiment starting points.

Live runs use only Quick's native numbered CSV/YML Saver. Ordinary run titles identify the experiment and relevant sweep parameters.

Held-Z maps are combined into one native pair, for example `ResVsZ_held_bias.csv` or `QubitSpecVsZ_fitted_readout.csv`. No new `runs/.../manifest.json` or `raw.npz` tree is created. The existing `runs` directory is retained as legacy acquired data.

Quick progress is enabled by `qick.show_progress: true`. `qick.progress_mode: terminal` is recommended for terminal or IDE execution; use `notebook` inside Jupyter. Each native `quick.Sweep` bar shows percentage, elapsed/estimated time, and iterations per second. Multi-Z scans use the same renderer for outer rows.

Before a live run applies held Z, QuICK-exp checks Gaussian qubit-pulse memory against the connected generator's `f_fabric`, `samps_per_clk`, and `maxlen`. The check also accounts for the cumulative Gaussian envelopes used by T1, Ramsey, Echo, IQ Scatter, and Dispersive Spectroscopy.

## Live hardware behavior

Logical `r`, `q`, `rr`, and `z` roles and their generator, DAC, readout, and ADC indexes are installation-specific and configured in `hardware.yml`. Run`00_connect_and_ports.py` to verify that routing before acquisition.

At zero Z, fixed-Z scripts use the generator reset performed during connection and skip the auxiliary acquisition. Nonzero fixed-Z scripts establish the held bias on the configured Z output using scalar settings independent of the main sweep. Connections, acquisitions, and closes clear persistent QICK streamer state so an interrupted process cannot affect the next sweep. A retry resets the configured generators and reapplies held Z. A normal exit attempts to return Z to zero bias; a  lost hardware link can prevent parking.

RF-board programming remains disabled until its attenuation and filter settings have been reviewed for the active hardware and wiring.

## Autocal

Currently, autocal is still in development and is not recommended for use.