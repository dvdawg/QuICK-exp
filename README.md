# QuICK-exp v3

QuICK-exp v3 is a superconducting-qubit measurement workflow with shared YAML
configuration, port verification, calibration precedence, recovery, and
held-flux safety.

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
| `17a_flux_step_spectroscopy.py`            | adaptive long-time flux-step spectroscopy                                |
| `17b_fit_flux_iir.py`                      | fit/export the candidate matched-z inverse IIR                            |
| `17c_cryoscope.py`                         | short-time Ramsey cryoscope acquisition                                  |
| `17d_fit_flux_fir.py`                      | fit/export regularized forward and inverse FIR filters                    |
| `90_measurement_queue.py`                  | run selected numbered files sequentially                                  |
| `91_autocal.py`                            | run/resume a policy-governed calibration target                           |
| `92_review_proposals.py`                   | inspect, promote, or reject inert proposals                               |
| `95_device_report.py`                      | local calibration/trend/QC report                                         |


The order is a suggested workflow. Adapt it to the device, firmware, and
calibration state in use.

The flux-line sequence is an implementation of Hellings *et al.*, Phys. Rev.
Research 7, 043142 (2025). Its exact paper settings, in-lab safety gates,
iteration order, efficient variants, hardware-upload boundary, and validation
holdouts are in
[docs/FLUX_COMPENSATION.md](docs/FLUX_COMPENSATION.md).

## Measurement queue

Open `90_measurement_queue.py`, list enabled files in `TASKS`, and run it.
The default queue runs Time Rabi followed by Power Rabi. `SHOW_PLOT=False` lets
the next task begin without waiting for a plot window to close. Each task still
uses its own normal `main()`, Quick CSV/YML Saver, connection, flux parking, and
cleanup. A task may override an `EDIT THESE` value without modifying the source:

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

## Automated calibration

Open `91_autocal.py`, keep `LIVE_HARDWARE=False` for the first run, choose a
target, and run it. The default L0 autonomy level always writes complete
fit records under `calibration.yml`'s inert top-level `proposals` mapping; it
does not replace accepted records. Each session is restartable from
`autocal_runs/<session_id>/state.yml`, and its append-only `decisions.jsonl`
records every acquisition, gate, retake, escalation, proposal, and promotion
decision without copying signal arrays.

Create `autocal_runs/STOP` to stop cleanly between acquisitions. Resume by
putting the prior session directory name in `SESSION_NAME`, after removing the
sentinel. Use `92_review_proposals.py` to promote or reject L0 results. L1 and
L2 promotion remain bounded by the hardware-owned `autocal` policy;
`defaults.r_offset`, `lookups.resonator_vs_flux`, and
`lookups.qubit_vs_flux` are hard stops at every level.

The full synthetic cold-start graph, STOP/resume, budget exhaustion,
failure escalation, proposal promotion, and read-only replay with native-pair
re-fitting are covered offline. Live rollout is deliberately staged: run
supervised L0 sessions before enabling L1. See
[docs/AUTOCAL.md](docs/AUTOCAL.md) for the operator runbook.

The hardware policy can opt N5 into perturbation-based identity adjudication
and N2/N3 into adaptive maps. These paths preserve the native acquisition and
safety stack, keep goodness-of-fit out of the identity verdict, record bounded
downstream backtracking, and produce a seven-prediction circuit-QED discrepancy
report. The optional external advisor is out-of-band and never executes its
own proposal; `mode: null` is the network-free default.

## Resonator-versus-Z fitted readout

1. Run `05c_resonator_spectroscopy_vs_flux.py` to create the native Quick
  `ResVsZ_held_bias.csv`/YML pair.
2. Run `05d_fit_resonator_vs_flux.py` with `WRITE_ACCEPTED_FIT = False`.
  It selects the newest matching CSV by default, extracts each notch with the
   Gaussian-smoothing/parabolic-refinement method, fits the robust
   cosine, and shows the map, fit, and residuals.
   `FIT_METHOD = "fit"` keeps that behavior. If the cosine fit is unreliable,
   use `"min"` or `"max"` to select the corresponding smoothed frequency bin
   in every Z row and build a piecewise-linear lookup instead. Set
   `SMOOTH_SIGMA_BINS = 0` if the extrema should come from the raw rows. A
   sampled lookup needs only two complete Z rows and ignores the R^2/RMSE
   gates, so `WRITE_ACCEPTED_FIT` is its only acceptance check.
3. After checking the diagnostics and quality gates, set
  `WRITE_ACCEPTED_FIT = True` for one run. The accepted parameters, measured Z
   domain, quality, uncertainty, and source CSV are written atomically to
   `calibration.yml`; the previous lookup is retained in `history`.
4. Set the latch back to `False`. Fixed-Z launchers now use
  `USE_ACCEPTED_RESONATOR_FLUX_FIT = True` by default. `06b` evaluates a
   different fitted `r_freq` for every Z row. Set the option to `False` in any
   launcher to use its explicit manual fallback.

The fit step does not create another experiment-data format. It reads Quick's
native CSV and only updates the accepted calibration when explicitly enabled.
Extrapolation outside the measured Z domain is rejected.

## Qubit spectroscopy and DAC Nyquist zones

One acquisition must stay wholly inside one physical DAC Nyquist zone. For this
reason, configure `Q_NYQUIST_BOUNDARY_MHZ` for the active hardware and firmware.
`06b_qubit_spectroscopy_vs_flux.py` selects `p1_nqz` from
`Q_FREQUENCY_MHZ` and rejects a range that crosses the configured boundary
instead of silently measuring the mirror image. Split a broad search into
separate runs for each zone; the chosen zone is recorded in the run title.

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

## Rabi fitting and acceptance

1. Run `08a_time_rabi.py` or `08b_power_rabi.py`.
2. Run `08c_fit_rabi.py` with `FIT_VARIABLE="q_length"` for Time Rabi or
  `FIT_VARIABLE="q_gain"` for Power Rabi. With `INPUT_CSV=None`, it selects
   the newest matching one-dimensional Quick CSV/YML pair and excludes chevrons.
3. Inspect the projected-IQ fit, residuals, IQ trajectory, pi recommendation,
  uncertainty, oscillation count, and quality gates.
4. For one deliberate run, set `WRITE_ACCEPTED_FIT=True`. The fitted pi value
  is written atomically to `records.defaults.q_length` or `q_gain` in
   `calibration.yml`; any prior record is retained in `history`. Reset the latch
   to `False` afterward.

The analysis never connects to QICK and does not create another data format.
The recommendation is phase-corrected: it uses the first fitted oscillation
extremum opposite the extrapolated zero-pulse state, while also reporting the
ordinary half-period for comparison.

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

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for implementation boundaries.
