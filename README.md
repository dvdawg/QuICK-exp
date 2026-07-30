# QuICK-exp v3

QuICK-exp v3 follows the numbered, one-file-per-measurement workflow of
`opx-expcode` while retaining shared YAML configuration, port verification,
calibration precedence, recovery, and held-flux safety from v2.

The workflow is IDE-first—there is no CLI. Open a numbered file in
`experiments/`, edit its `EDIT THESE` block, and press Run.

## First use

1. Select `C:\Users\quant\anaconda3\envs\qcodes\python.exe` in the IDE.
2. Open `experiments/01_configure_experiment.py`, edit shared values such as
   `q_freq`, `r_freq`, the native data directory, connection, or logical
   channels, and run it once with `WRITE_CHANGES = False`.
3. When the preview is correct, set `WRITE_CHANGES = True`, run it, then set it
   back to `False`.
4. Run `00_connect_and_ports.py` to connect and verify the live `soccfg`
   against the `r/rr/q/z` map.
5. Run 02 to verify the raw ADC trace/readout trigger offset before proceeding
   through spectroscopy and time-domain measurements.

Installation is optional because each launcher adds the project root to
`sys.path`. To run tests:

```powershell
C:\Users\quant\anaconda3\envs\qcodes\python.exe -m pytest -q
```

## Experimental order

| File | Purpose |
|---|---|
| `00_connect_and_ports.py` | connect, print `soccfg`, verify routing |
| `01_configure_experiment.py` | edit/validate/write common YAML settings |
| `02_raw_adc_loopback.py` | raw decimated trace and readout-offset check |
| `05a_resonator_spectroscopy_vs_power.py` | native power-by-frequency punchout |
| `05b_resonator_spectroscopy_fixed_flux.py` | resonator scan at held Z |
| `05c_resonator_spectroscopy_vs_flux.py` | resonator spectroscopy versus Z |
| `05d_fit_resonator_vs_flux.py` | fit/accept the resonator-versus-Z readout curve |
| `06a_qubit_spectroscopy.py` | qubit spectroscopy |
| `06b_qubit_spectroscopy_vs_flux.py` | qubit spectroscopy versus Z |
| `06c_qubit_spectroscopy_vs_gain.py` | gain-by-frequency power spectroscopy |
| `06f_qubit_spectroscopy_zpa.py` | authored native frequency-by-Z map (simulation until hardware gates pass) |
| `07a_rabi_chevron_duration.py` | frequency-by-duration Rabi chevron |
| `07b_rabi_chevron_amplitude.py` | frequency-by-gain Rabi chevron |
| `08a_time_rabi.py` | pulse-duration Rabi |
| `08b_power_rabi.py` | pulse-gain Rabi |
| `08c_fit_rabi.py` | fit/accept a time- or power-Rabi pi value |
| `09a_iq_blobs.py` | ground/excited IQ clouds |
| `10a_readout_frequency_optimization.py` | dispersive readout scan |
| `11_t1.py` | energy relaxation |
| `11c_t1_vs_flux.py` | authored finite-Z-pulse T1 map (simulation until hardware gates pass) |
| `12_ramsey_chevron.py` | frequency-by-delay Ramsey chevron |
| `13a_ramsey.py` | Ramsey dephasing |
| `14_echo.py` | Hahn echo / fixed-cycle CPMG |
| `16_two_photon_spectroscopy.py` | high-power two-photon search |
| `90_measurement_queue.py` | run selected numbered files sequentially |
| `91_autocal.py` | run/resume a policy-governed calibration target |
| `92_review_proposals.py` | inspect, promote, or reject inert proposals |
| `95_device_report.py` | local calibration/trend/QC report |

The order and experiment variants follow `opx-expcode`; Quick classes, routing,
and held-Z behavior were checked against `2026-07-21 MET v191.ipynb` and
installed Quick 0.7.2.

## Measurement queue

Open `90_measurement_queue.py`, list enabled files in `TASKS`, and press Run.
The default queue runs Time Rabi followed by Power Rabi. `SHOW_PLOT=False` lets
the next task begin without waiting for a plot window to close. Each task still
uses its own normal `main()`, Quick CSV/YML Saver, connection, flux parking, and
cleanup. A task may override an `EDIT THESE` value without modifying the source:

```python
{
    "file": "08a_time_rabi.py",
    "label": "Time Rabi at alternate gain",
    "enabled": True,
    "settings": {"Q_GAIN": 0.25},
}
```

Duplicate files are allowed. `STOP_ON_ERROR=True` stops at the first failure;
set it to `False` to continue and receive a `completed_with_errors` summary.

## Automated calibration

Open `91_autocal.py`, keep `LIVE_HARDWARE=False` for the first run, choose a
target, and press Run. The default L0 autonomy level always writes complete
fit records under `calibration.yml`'s inert top-level `proposals` mapping; it
does not replace accepted records. Each session is restartable from
`autocal_runs/<session_id>/state.yml`, and its append-only `decisions.jsonl`
records every acquisition, gate, retake, escalation, proposal, and promotion
decision without copying signal arrays.

Create `autocal_runs/STOP` to stop cleanly between acquisitions. Resume by
putting the prior session directory name in `SESSION_NAME`, after removing the
sentinel. Use `92_review_proposals.py` to promote or reject L0 results. L1 and
L2 promotion remain bounded by the hardware-owned `autocal` policy; cabling
timing and flux lookup models are hard stops at every level.

The full synthetic cold-start graph, STOP/resume, budget exhaustion,
failure escalation, proposal promotion, and read-only replay with native-pair
re-fitting are covered offline. Live rollout is deliberately staged: run
supervised L0 sessions before enabling L1. See
[AUTOCAL.md](AUTOCAL.md) for the operator runbook.

## Resonator-versus-Z fitted readout

1. Run `05c_resonator_spectroscopy_vs_flux.py` to create the native Quick
   `ResVsZ_held_bias.csv`/YML pair.
2. Run `05d_fit_resonator_vs_flux.py` with `WRITE_ACCEPTED_FIT = False`.
   It selects the newest matching CSV by default, extracts each notch with the
   notebook's Gaussian-smoothing/parabolic-refinement method, fits the robust
   cosine, and shows the map, fit, and residuals.
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

Live runs use only Quick's native numbered CSV/YML Saver. Ordinary runs get
descriptive titles such as:

```text
00035 - (QubitSpectroscopy)QubitSpec_Zp0p1400_r6884p544.csv
00065 - (T1)Zp0p0000_T1_q5606p500.csv
```

Held-Z maps are combined into one native pair, for example
`ResVsZ_held_bias.csv` or `QubitSpecVsZ_fitted_readout.csv`. No new
`runs/.../manifest.json` or `raw.npz` tree is created. The existing `runs`
directory is retained as legacy acquired data.

Quick progress is enabled by `qick.show_progress: true`. `qick.progress_mode: terminal`
is the correct renderer for an IDE Run button or PowerShell; use `notebook` only
inside Jupyter. Each native `quick.Sweep` bar shows percentage, elapsed/estimated
time, and iterations per second. Multi-Z scans use the same renderer for outer rows.

Before a live run applies held Z, v3 checks Gaussian qubit-pulse memory against
the connected generator's `f_fabric`, `samps_per_clk`, and `maxlen`. On the
current ZCU216 bitfile, q generator 1 holds 16,384 samples: the 1.67 us Time
Rabi endpoint uses 16,000 samples, while 1.72 us would use 16,480 and is
rejected immediately. This is a configuration error, so it is not retried.
The check also accounts for the cumulative Gaussian envelopes used by
T1, Ramsey, Echo, IQ Scatter, and Dispersive Spectroscopy.

## Rabi notebook parity

The `07a`, `07b`, `08a`, and `08b` launchers expose the complete working
notebook recipe in their `EDIT THESE` blocks: `Z_GAIN`, `Z_LENGTH_US`,
`Z_SETTLE_US`, readout power/length/offset/phase/relaxation, `REP`, and
`POPULATION`. The MET notebook's Rabi program uses `rep: 1000`,
`population=False`, `r_power=-30 dB`, a `0.2 us` held-Z pulse, and a `5.0 us`
program settle. These are explicit Rabi defaults; the generic hardware
fallback `rep: 1` must not silently replace Quick's Rabi repetition count.

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

## MET live behavior

The reviewed logical map is:

```text
r  = generator 0  -> DAC0  axis_signal_gen_v6
q  = generator 1  -> DAC1  axis_signal_gen_v6
rr = readout 0    -> ADC4  axis_dyn_readout_v1
z  = generator 15 -> DAC15 axis_sg_int4_v2
```

At zero Z, fixed-Z scripts use the generator reset performed during connection
and skip the auxiliary acquisition entirely. Nonzero fixed-Z scripts establish
a minimal one-average `p9_mode: last` pulse on generator 15. The helper
scalarizes every numeric sweep parameter before compiling this auxiliary
program, so no 2D readout/qubit list can leak into Mercator. Every connection,
acquisition, and close also stops/drains/flushes persistent QICK streamer state,
so LoopBack or an aborted process cannot poison the next integrated sweep. On
retry the runtime resets generators and reapplies held Z. On exit it parks Z at
zero.

RF-board programming stays disabled until attenuation/filter settings have
been deliberately reviewed for the current wiring.

See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation boundaries.
