"""The fit-and-suggest hardware layer -- the only module that touches QICK.

Each ``Session`` step applies the current ``var`` + per-call overrides, runs a
``quick`` experiment (which saves the raw CSV exactly as today), fits the
result, writes a fit sidecar, and returns a :class:`StepResult` carrying the
fit, a suggested next scan, and a ``plot`` method. Nothing is committed to the
persistent calibration state until the operator calls ``result.accept()``.

The experiment *runner* is injectable so the orchestration and commit logic are
unit-tested offline with a fake; the real runner (which imports ``quick``) is
exercised only on the lab PC. ``quick`` is imported lazily so this module loads
anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from . import (
    adaptive,
    config as _config,
    diagnostics,
    fitting,
    loopback as _loopback,
    ports,
    results,
    state,
)


# --------------------------------------------------------------------------- #
# Result of a single fit-and-suggest step
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    fit: Any
    csv_path: Optional[str]
    session: "Session"
    commit_map: Dict[str, str] = field(default_factory=dict)
    suggestion: str = ""
    diagnosis: Any = None
    recommendations: Dict[str, Any] = field(default_factory=dict)

    def plot(self, ax=None):
        return self.fit.plot(ax=ax)

    def accept(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Commit the mapped fit values to calibration state and the live ``var``.

        ``commit_map`` maps a ``var`` key to a fit attribute, e.g.
        ``{"r_freq": "f0"}``. Extra key/value pairs may be committed directly.
        """
        if getattr(self.fit, "valid", True) is False:
            raise ValueError("refusing to accept an invalid fit; adjust the fit window and rescan")
        updates: Dict[str, Any] = {
            var_key: getattr(self.fit, attr) for var_key, attr in self.commit_map.items()
        }
        if extra:
            updates.update(extra)
        for key, value in updates.items():
            self.session.var[key] = value
        return state.record(self.session.calibration_path, updates)


# --------------------------------------------------------------------------- #
# Session: holds the connected board, the live var, and the calibration file
# --------------------------------------------------------------------------- #
class Session:
    def __init__(
        self,
        cfg: _config.HardwareConfig,
        calibration_path: str,
        soc=None,
        soccfg=None,
        runner: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.calibration_path = calibration_path
        self.soc = soc
        self.soccfg = soccfg
        self.runner = runner or _make_quick_runner()
        base = _config.build_var(cfg)
        self.var = state.merged_var(base, state.load(calibration_path))

    # -- connection (lab PC only) -------------------------------------------- #
    @classmethod
    def connect(cls, config_path: str, calibration_path: str) -> "Session":
        """Load config, connect to the QICK board, apply RF/bias setup."""
        import quick  # lazy: only available on the lab PC

        cfg = _config.load_config(config_path)
        soccfg, soc = quick.connect(cfg.ip)
        _config.apply_to_soc(cfg, soc)
        sess = cls(cfg, calibration_path, soc=soc, soccfg=soccfg)
        sess.check_ports(strict=True)  # print the map and stop on a declared mismatch
        return sess

    # -- port sanity (hardware cannot be rewired -- verify, don't assume) ----- #
    def check_ports(self, strict: bool = False):
        """Resolve + print the logical->physical port map; return the resolved dict.

        Catches the recurring "is r=1 really DAC 10?" doubt: it shows the physical
        DAC/ADC port each logical index lands on, warns on direct/no-RF-card paths,
        and -- if ``hardware.yml`` declares the expected port -- flags a mismatch.
        With ``strict=True`` a declared-port mismatch raises instead of warning.
        """
        if self.soccfg is None:
            raise RuntimeError(
                "check_ports needs a live soccfg; use Session.connect(...) on the lab PC."
            )
        resolved = ports.from_soccfg(self.soccfg, self.cfg.channels, self.cfg.expected_ports)
        print(ports.report(resolved))
        if strict:
            errs = ports.errors(resolved)
            if errs:
                raise ValueError(
                    "port check failed:\n  " + "\n  ".join(e.message for e in errs)
                )
        return resolved

    # -- recommend-then-override (operator in the loop) ---------------------- #
    def apply(self, recommendations: Dict[str, Any], **overrides) -> Dict[str, Any]:
        """Stage recommended knob values into the live ``var`` for the next step.

        Recommendations come from a previous step's ``.recommendations``; explicit
        ``**overrides`` win. This only updates the in-memory ``var`` -- nothing is
        persisted to calibration state until you ``.accept()`` a fit.
        """
        merged = {**recommendations, **overrides}
        self.var.update(merged)
        return merged

    # -- internal step engine ------------------------------------------------ #
    def _run(self, experiment: str, title: str, run_kwargs: Dict[str, Any], **overrides):
        var = dict(self.var)
        data, csv_path = self.runner(
            experiment, var, title, self.cfg.data_path, run_kwargs, **overrides
        )
        return np.asarray(data, float), csv_path

    def _finish(self, fit, csv_path, commit_map, experiment,
                diagnosis=None, recommendations=None):
        if csv_path is not None:
            extra = {"experiment": experiment}
            if diagnosis is not None:
                extra["diagnosis"] = diagnosis.status
                extra["snr"] = round(float(diagnosis.snr), 2)
            results.write_sidecar(csv_path, fit, extra=extra)
        if diagnosis is not None:
            print(diagnosis.report())  # advisory health report, surfaced in the cell
        return StepResult(
            fit=fit,
            csv_path=csv_path,
            session=self,
            commit_map=commit_map,
            suggestion=fit.suggestion(),
            diagnosis=diagnosis,
            recommendations=dict(recommendations or {}),
        )

    def _expected_range(self, key):
        rng = self.cfg.expected.get(key)
        return tuple(rng) if rng else None

    # -- calibration chain steps -------------------------------------------- #
    def loopback_calibrate(
        self, powers=None, fullscale=None, offset_power=-30,
        title="loopback_calibrate", **overrides
    ) -> StepResult:
        """Bring-up: measure the readout-window delay and pick a safe drive level.

        Starts at ``r_offset=0``, finds the returned-pulse onset from the time
        trace, then sweeps ``powers`` and selects the highest level whose peak
        stays below ADC over-range. ``accept()`` commits both ``r_offset`` and the
        chosen ``r_power``. ``fullscale`` defaults to the configured
        ``expected.adc_fullscale_counts``.
        """
        if powers is None:
            powers = list(range(-40, 1, 5))
        if fullscale is None:
            fullscale = self.cfg.expected.get("adc_fullscale_counts")
        if not fullscale:
            raise ValueError(
                "loopback_calibrate needs a full-scale count: pass fullscale=... or "
                "set expected.adc_fullscale_counts in hardware.yml"
            )

        # 1) Offset from the returned-pulse onset, with the readout window wide open.
        data, csv_path = self._run(
            "LoopBack", f"{title}_offset", {}, r_offset=0, r_power=offset_power, **overrides
        )
        time, amp = data[:, 0], data[:, 1]
        r_offset = _loopback.find_offset(time, amp)

        # 2) Power sweep at the found offset; record peak counts per level.
        peaks = []
        for power in powers:
            d, csv_path = self._run(
                "LoopBack", f"{title}_p{power}", {},
                r_offset=r_offset, r_power=power, **overrides,
            )
            peaks.append(float(np.max(np.abs(d[:, 1]))))

        best_power, status = _loopback.select_power(powers, peaks, fullscale)
        peak = peaks[list(powers).index(best_power)]
        cal = _loopback.LoopbackCalibration(
            r_offset=r_offset, r_power=best_power, peak=peak, fullscale=float(fullscale),
            powers=list(powers), peaks=peaks, valid=(status == "ok"), status=status,
            time=time, amplitude=amp,
        )
        diag = diagnostics.diagnose_loopback(status, peak, float(fullscale))
        recs = {"r_offset": r_offset, "r_power": best_power}
        return self._finish(
            cal, csv_path, {"r_offset": "r_offset", "r_power": "r_power"}, "LoopBack",
            diagnosis=diag, recommendations=recs,
        )

    def resonator_spectroscopy(self, r_freq, window=None, **overrides) -> StepResult:
        data, csv_path = self._run(
            "ResonatorSpectroscopy", overrides.pop("title", "resonator"),
            {"dB": True}, r_freq=r_freq, **overrides,
        )
        freq, amp_db, _ph, I, Q = data.T
        fit = fitting.fit_resonator(freq, I, Q, window=window)
        # Resonator readout is diagnosed on the dB amplitude -- the natural,
        # robust readout (a deep narrow notch is marginal in linear magnitude).
        diag = diagnostics.diagnose_spectroscopy(
            freq, amp_db, fit, expected_range=self._expected_range("r_freq"),
            recommendations={"r_freq": round(float(fit.f0), 4)} if getattr(fit, "valid", True) else {},
        )
        return self._finish(
            fit, csv_path, {"r_freq": "f0"}, "ResonatorSpectroscopy",
            diagnosis=diag, recommendations=diag.recommendations,
        )

    def qubit_spectroscopy(self, q_freq, window=None, **overrides) -> StepResult:
        data, csv_path = self._run(
            "QubitSpectroscopy", overrides.pop("title", "qubit_spec"),
            {"dB": False}, q_freq=q_freq, **overrides,
        )
        freq, _amp, _ph, I, Q = data.T
        fit = fitting.fit_qubit_peak(freq, I, Q, window=window)
        signal = fitting.rotated_iq(I, Q)
        diag = diagnostics.diagnose_spectroscopy(
            freq, signal, fit, expected_range=self._expected_range("q_freq"),
            recommendations={"q_freq": round(float(fit.f0), 4)},
        )
        return self._finish(
            fit, csv_path, {"q_freq": "f0"}, "QubitSpectroscopy",
            diagnosis=diag, recommendations=diag.recommendations,
        )

    def rabi(self, x, xlabel="Drive", **overrides) -> StepResult:
        data, csv_path = self._run(
            "Rabi", overrides.pop("title", "rabi"), {}, **overrides
        )
        signal = fitting.rotated_iq(data[:, 3], data[:, 4])
        fit = fitting.fit_rabi(data[:, 0], signal, xlabel=xlabel)
        # Recommend the pi-pulse value for whichever knob was swept (gain or length).
        knob = "q_gain" if "gain" in xlabel.lower() else "q_length"
        recs = {knob: float(fit.pi_value)}
        return self._finish(fit, csv_path, {}, "Rabi", recommendations=recs)

    def t1(self, time, **overrides) -> StepResult:
        data, csv_path = self._run(
            "T1", overrides.pop("title", "T1"), {"population": False},
            time=time, **overrides,
        )
        signal = fitting.rotated_iq(data[:, 3], data[:, 4])
        fit = fitting.fit_t1(data[:, 0], signal)
        return self._finish(fit, csv_path, {}, "T1")

    def t2(self, time, experiment="T2Ramsey", **overrides) -> StepResult:
        data, csv_path = self._run(
            experiment, overrides.pop("title", experiment), {}, time=time, **overrides
        )
        signal = fitting.rotated_iq(data[:, 3], data[:, 4])
        fit = fitting.fit_t2(data[:, 0], signal)
        return self._finish(fit, csv_path, {}, experiment)


# --------------------------------------------------------------------------- #
# Real runner (lab PC) -- imports quick lazily; verify path handling on hardware
# --------------------------------------------------------------------------- #
def _make_quick_runner():
    """Build the default runner that drives a real ``quick`` experiment.

    Returns ``(data_array, csv_path)``. The CSV is written by ``quick``'s own
    Saver in the unchanged format; we only locate its path to attach a sidecar.
    """

    def runner(experiment, var, title, data_path, run_kwargs, **overrides):
        import glob
        import os
        import quick

        exp_cls = getattr(quick.experiment, experiment)
        # quick.connect(), called by Session.connect(), populates quick's client-
        # side connection cache.  Let BaseExperiment consume that cache. Passing
        # the Pyro proxy explicitly can cause board-only driver types to be
        # deserialized on the Windows client (and therefore import ``pynq``).
        exp = exp_cls(var=var, data_path=data_path, title=title, **overrides)
        try:
            result = exp.run(**run_kwargs)
        except ModuleNotFoundError as exc:
            if exc.name != "pynq":
                raise
            # RFQickSoc raises a board-only ADCInterruptError when any ADC is
            # over-range. Pyro/pickle tries to import qick.rfboard on Windows
            # to reconstruct it, which masks the real fault as missing pynq.
            _, connected_soc = quick.helper.getSoc()
            interrupts_cleared = connected_soc.clear_interrupts(
                max_attempts=5,
                error_on_interrupt=False,
                error_on_persist=False,
            )
            if interrupts_cleared:
                # RFQickSoc aborts a round if it finds even a transient interrupt,
                # including one it successfully clears. Retry once; a recurring
                # interrupt remains a hard failure below.
                try:
                    result = exp.run(**run_kwargs)
                except ModuleNotFoundError as retry_exc:
                    if retry_exc.name != "pynq":
                        raise
                    exc = retry_exc
                else:
                    exc = None
            if exc is not None:
                raise RuntimeError(
                    "The QICK board repeatedly reported an ADC interrupt/over-range "
                    "condition after one clear-and-retry. Reduce the input level; "
                    "pynq is board-only and should not be installed on this PC."
                ) from exc

        # Locate the CSV quick just wrote (best-effort: explicit attr, else newest).
        saver = getattr(exp, "s", None)
        saver_base = getattr(saver, "file_name", None)
        csv_path = (
            (saver_base + ".csv") if saver_base else None
        ) or getattr(exp, "path", None) or getattr(result, "path", None)
        if csv_path is None and data_path:
            csvs = glob.glob(os.path.join(data_path, "*.csv"))
            csv_path = max(csvs, key=os.path.getmtime) if csvs else None

        return np.asarray(result.data, float), csv_path

    return runner
