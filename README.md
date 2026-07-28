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
| `06a_qubit_spectroscopy.py` | qubit spectroscopy |
| `06b_qubit_spectroscopy_vs_flux.py` | qubit spectroscopy versus Z |
| `06c_qubit_spectroscopy_vs_gain.py` | gain-by-frequency power spectroscopy |
| `07a_rabi_chevron_duration.py` | frequency-by-duration Rabi chevron |
| `07b_rabi_chevron_amplitude.py` | frequency-by-gain Rabi chevron |
| `08a_time_rabi.py` | pulse-duration Rabi |
| `08b_power_rabi.py` | pulse-gain Rabi |
| `09a_iq_blobs.py` | ground/excited IQ clouds |
| `10a_readout_frequency_optimization.py` | dispersive readout scan |
| `11_t1.py` | energy relaxation |
| `12_ramsey_chevron.py` | frequency-by-delay Ramsey chevron |
| `13a_ramsey.py` | Ramsey dephasing |
| `14_echo.py` | Hahn echo / fixed-cycle CPMG |
| `16_two_photon_spectroscopy.py` | high-power two-photon search |

The order and experiment variants follow `opx-expcode`; Quick classes, routing,
and held-Z behavior were checked against `2026-07-21 MET v191.ipynb` and
installed Quick 0.7.2.

## Configuration and data

Settings resolve in this order:

```text
hardware defaults < accepted calibration < preset < launcher overrides
```

- `hardware.yml` contains the connection, routing, bounds, shared defaults, and
  native Quick output directory.
- `calibration.yml` contains accepted values and provenance. The 01 editor
  updates accepted `r_freq`, `q_freq`, and `r_offset` records with their matching
  defaults so precedence stays intuitive.
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

Quick progress is enabled by `qick.show_progress: true`; each native sweep shows
the `quick.Sweep` bar with percentage, elapsed/estimated time, and iterations
per second. Multi-Z scans also use a Quick progress bar for their outer rows.

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
