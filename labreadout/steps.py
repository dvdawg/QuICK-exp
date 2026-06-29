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

from . import adaptive, config as _config, fitting, results, state


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

    def plot(self, ax=None):
        return self.fit.plot(ax=ax)

    def accept(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Commit the mapped fit values to calibration state and the live ``var``.

        ``commit_map`` maps a ``var`` key to a fit attribute, e.g.
        ``{"r_freq": "f0"}``. Extra key/value pairs may be committed directly.
        """
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
        self.runner = runner or _make_quick_runner(soc)
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
        return cls(cfg, calibration_path, soc=soc, soccfg=soccfg)

    # -- internal step engine ------------------------------------------------ #
    def _run(self, experiment: str, title: str, run_kwargs: Dict[str, Any], **overrides):
        var = dict(self.var)
        data, csv_path = self.runner(
            experiment, var, title, self.cfg.data_path, run_kwargs, **overrides
        )
        return np.asarray(data, float), csv_path

    def _finish(self, fit, csv_path, commit_map, experiment):
        if csv_path is not None:
            results.write_sidecar(csv_path, fit, extra={"experiment": experiment})
        return StepResult(
            fit=fit,
            csv_path=csv_path,
            session=self,
            commit_map=commit_map,
            suggestion=fit.suggestion(),
        )

    # -- calibration chain steps -------------------------------------------- #
    def resonator_spectroscopy(self, r_freq, window=None, **overrides) -> StepResult:
        data, csv_path = self._run(
            "ResonatorSpectroscopy", overrides.pop("title", "resonator"),
            {"dB": True}, r_freq=r_freq, **overrides,
        )
        _f, _amp, _ph, I, Q = data.T
        fit = fitting.fit_resonator(data[:, 0], I, Q, window=window)
        return self._finish(fit, csv_path, {"r_freq": "f0"}, "ResonatorSpectroscopy")

    def qubit_spectroscopy(self, q_freq, window=None, **overrides) -> StepResult:
        data, csv_path = self._run(
            "QubitSpectroscopy", overrides.pop("title", "qubit_spec"),
            {"dB": False}, q_freq=q_freq, **overrides,
        )
        _f, _amp, _ph, I, Q = data.T
        fit = fitting.fit_qubit_peak(data[:, 0], I, Q, window=window)
        return self._finish(fit, csv_path, {"q_freq": "f0"}, "QubitSpectroscopy")

    def rabi(self, x, xlabel="Drive", **overrides) -> StepResult:
        data, csv_path = self._run(
            "Rabi", overrides.pop("title", "rabi"), {}, **overrides
        )
        signal = fitting.rotated_iq(data[:, 3], data[:, 4])
        fit = fitting.fit_rabi(data[:, 0], signal, xlabel=xlabel)
        return self._finish(fit, csv_path, {}, "Rabi")

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
def _make_quick_runner(soc):
    """Build the default runner that drives a real ``quick`` experiment.

    Returns ``(data_array, csv_path)``. The CSV is written by ``quick``'s own
    Saver in the unchanged format; we only locate its path to attach a sidecar.
    """

    def runner(experiment, var, title, data_path, run_kwargs, **overrides):
        import glob
        import os
        import quick

        exp_cls = getattr(quick.experiment, experiment)
        exp = exp_cls(var=var, data_path=data_path, title=title, **overrides)
        result = exp.run(**run_kwargs)

        # Locate the CSV quick just wrote (best-effort: explicit attr, else newest).
        csv_path = getattr(exp, "path", None) or getattr(result, "path", None)
        if csv_path is None and data_path:
            csvs = glob.glob(os.path.join(data_path, "*.csv"))
            csv_path = max(csvs, key=os.path.getmtime) if csvs else None

        return np.asarray(result.data, float), csv_path

    return runner
