"""Acquisition backends for installed Quick and deterministic offline work."""

from __future__ import annotations

from copy import deepcopy
import inspect
from typing import Any, Mapping, Optional

import numpy as np

from .data import BackendResult
from .errors import AcquisitionError, ConfigError


LOGICAL_CHANNELS = {
    "r": ("r", "gen"),
    "rr": ("rr", "ro"),
    "q": ("q", "gen"),
    "z": ("z", "gen"),
}

# Quick 0.7.2 treats these as experiment variables. Constructor keywords not
# in this set are literal Mercator config overrides (for example hard_avg).
QUICK_BASE_VARIABLES = frozenset(
    {
        "rr", "r", "q", "r_freq", "r_power", "r_length", "r_phase",
        "r_offset", "r_threshold", "r_relax", "r_reset", "q_freq",
        "q_length", "q_length_2", "q_delta", "q_gain", "q_gain_2",
    }
)
QUICK_CLASS_VARIABLES = {
    "Rabi": frozenset({"cycle"}),
    "T1": frozenset({"time"}),
    "T2Ramsey": frozenset({"time", "fringe_freq"}),
    "T2Echo": frozenset({"time", "cycle", "fringe_freq"}),
    "IQScatter": frozenset({"rr_length"}),
}
QUICK_CONFIG_OVERRIDES = ("hard_avg", "soft_avg", "rep")
QUICK_CONFIG_OVERRIDES_BY_CLASS = {
    # Decimated data for every hard repetition must fit in the on-board
    # accumulation buffer. The installed LoopBack template uses soft_avg and
    # leaves hard_avg/reps at one, matching the MET notebook.
    "LoopBack": ("soft_avg",),
}


def logical_channel_variables(hardware: Mapping[str, Any]) -> dict:
    channels = hardware.get("channels", {})
    result = {}
    for logical, (channel_name, endpoint) in LOGICAL_CHANNELS.items():
        channel = channels.get(channel_name) if isinstance(channels, Mapping) else None
        if not isinstance(channel, Mapping) or endpoint not in channel:
            raise ConfigError(
                f"hardware.channels.{channel_name}.{endpoint} is required"
            )
        result[logical] = int(channel[endpoint])
    return result

def clean_qick_acquisition_state(
    soc: Any,
    *,
    reset_generators: bool = False,
    clear_interrupts: bool = False,
) -> dict:
    """Stop and drain persistent QICK readout state left by prior processes."""
    actions = []
    warnings = []

    def attempt_action(label, callback):
        if not callable(callback):
            return
        try:
            callback()
            actions.append(label)
        except Exception as error:
            warnings.append(f"{label}: {error}")

    stop_tproc = getattr(soc, "stop_tproc", None)
    if not callable(stop_tproc):
        stop_tproc = getattr(soc, "stop_tproc_counter", None)
    attempt_action("stop_tproc", stop_tproc)

    streamer = getattr(soc, "streamer", None)
    if streamer is not None:
        running = getattr(streamer, "readout_running", None)
        try:
            should_stop = bool(running()) if callable(running) else True
        except Exception as error:
            warnings.append(f"streamer.readout_running: {error}")
            should_stop = True
        if should_stop:
            attempt_action(
                "streamer.stop_readout",
                getattr(streamer, "stop_readout", None),
            )
            done_flag = getattr(streamer, "done_flag", None)
            wait = getattr(done_flag, "wait", None)
            if callable(wait):
                attempt_action(
                    "streamer.done_flag.wait",
                    lambda: wait(timeout=1.0),
                )

    poll_data = getattr(soc, "poll_data", None)
    attempt_action(
        "poll_data_flush",
        (lambda: poll_data(totaltime=-1, timeout=0.05))
        if callable(poll_data)
        else None,
    )
    if reset_generators:
        attempt_action("reset_gens", getattr(soc, "reset_gens", None))
    if clear_interrupts:
        clear = getattr(soc, "clear_interrupts", None)

        def clear_safely():
            try:
                return clear(
                    max_attempts=5,
                    error_on_interrupt=False,
                    error_on_persist=False,
                )
            except TypeError:
                try:
                    return clear()
                except TypeError:
                    return clear(ch_list=None)

        attempt_action("clear_interrupts", clear_safely if callable(clear) else None)
    return {"actions": actions, "warnings": warnings}

class QuickBackend:
    """Thin, lazy bridge to the locally installed ``quick`` package.

    The backend performs acquisition only.  Decoding, persistence, fitting, and
    calibration recommendations remain owned by the v3 experiment/runtime.
    """

    def __init__(
        self,
        *,
        soc: Any,
        soccfg: Any,
        quick_module: Any = None,
        data_path: Optional[str] = None,
        show_progress: bool = True,
    ):
        if quick_module is None:
            try:
                import quick as quick_module  # type: ignore
            except ImportError as error:
                raise AcquisitionError(
                    "the live backend requires the local 'quick' package"
                ) from error
        self.quick = quick_module
        self.soc = soc
        self.soccfg = soccfg
        self.data_path = data_path
        self.show_progress = bool(show_progress)
        self.connected = False

    def connect(self) -> None:
        if self.soc is None or self.soccfg is None:
            raise AcquisitionError("QuickBackend requires both soc and soccfg")
        self.connected = True

    def acquire(self, plan: Any) -> BackendResult:
        if not self.connected:
            self.connect()
        self.last_preflight = clean_qick_acquisition_state(self.soc)
        for warning in self.last_preflight["warnings"]:
            print(f"QICK readout preflight warning: {warning}")
        try:
            experiment_class = getattr(self.quick.experiment, plan.quick_class)
        except AttributeError as error:
            raise AcquisitionError(
                f"installed quick has no experiment class {plan.quick_class!r}"
            ) from error
        variables = deepcopy(dict(plan.variables))
        variable_names = QUICK_BASE_VARIABLES.union(
            QUICK_CLASS_VARIABLES.get(plan.quick_class, frozenset())
        )
        quick_variables = {
            name: value
            for name, value in variables.items()
            if name in variable_names
        }
        # Quick 0.7.2 only registers iterable constructor keyword arguments as
        # sweeps. Arrays present only inside ``var`` silently acquire one point.
        sweep_arguments = {
            name: variables[name]
            for name in plan.axes
            if name in variable_names
        }
        # hard_avg/soft_avg/rep are literal Mercator keys in the installed
        # templates, not entries in quick.experiment.var. They must remain
        # outside ``var`` so BaseExperiment routes them to config_update.
        config_keys = QUICK_CONFIG_OVERRIDES_BY_CLASS.get(
            plan.quick_class,
            QUICK_CONFIG_OVERRIDES,
        )
        config_arguments = {
            name: variables[name]
            for name in config_keys
            if name in variables
        }
        constructor_arguments = {**config_arguments, **sweep_arguments}
        experiment = experiment_class(
            var=quick_variables,
            soc=self.soc,
            soccfg=self.soccfg,
            data_path=self.data_path,
            title=plan.title,
            **constructor_arguments,
        )
        expected_sweeps = tuple(sweep_arguments)
        actual_sweeps = tuple(getattr(experiment, "sweep", {}))
        if actual_sweeps != expected_sweeps:
            raise AcquisitionError(
                f"{plan.quick_class} registered sweeps {actual_sweeps}, "
                f"expected {expected_sweeps}"
            )
        candidates = deepcopy(dict(plan.run_options))
        signature = inspect.signature(experiment.run)
        accepts_kwargs = any(
            value.kind is inspect.Parameter.VAR_KEYWORD
            for value in signature.parameters.values()
        )
        arguments = {
            key: value
            for key, value in candidates.items()
            if accepts_kwargs or key in signature.parameters
        }
        if accepts_kwargs or "silent" in signature.parameters:
            arguments["silent"] = not self.show_progress
        try:
            completed = experiment.run(**arguments)
        except Exception as error:
            raise AcquisitionError(
                f"{plan.name} acquisition through {plan.quick_class} failed: {error}"
            ) from error
        saver = getattr(completed, "s", None)
        native_stem = getattr(saver, "file_name", None)
        native_files = (
            [native_stem + ".csv", native_stem + ".yml"]
            if native_stem
            else []
        )
        return BackendResult(
            payload=np.asarray(completed.data),
            metadata={
                "backend": "quick",
                "quick_version": getattr(self.quick, "__version__", None),
                "quick_class": plan.quick_class,
                "quick_sweep": list(getattr(completed, "sweep", [])),
                "requested_var": variables,
                "resolved_var": deepcopy(getattr(completed, "var", variables)),
                "run_options": arguments,
                "native_files": native_files,
            },
        )

    def recover(self, error: BaseException, attempt: int) -> dict:
        """Use the MET notebook's stop, drain, reset recovery order."""
        details = clean_qick_acquisition_state(
            self.soc,
            reset_generators=True,
            clear_interrupts=True,
        )
        return {
            "attempt": attempt,
            "error": str(error),
            **details,
        }
    def snapshot(self) -> dict:
        return {
            "backend": "quick",
            "quick_version": getattr(self.quick, "__version__", None),
            "connected": self.connected,
            "show_progress": self.show_progress,
            "soc_type": type(self.soc).__name__,
            "soccfg_type": type(self.soccfg).__name__,
        }

    def close(self) -> None:
        self.last_cleanup = clean_qick_acquisition_state(
            self.soc,
            reset_generators=True,
            clear_interrupts=True,
        )
        for warning in self.last_cleanup["warnings"]:
            print(f"QICK close cleanup warning: {warning}")
        self.connected = False


class SyntheticBackend:
    """Deterministic backend for config checks, demos, and tests."""

    def __init__(self, seed: int = 7, fail_attempts: int = 0):
        self.seed = int(seed)
        self.fail_attempts = int(fail_attempts)
        self.calls = 0
        self.recoveries = 0
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def acquire(self, plan: Any) -> BackendResult:
        if not self.connected:
            self.connect()
        self.calls += 1
        if self.calls <= self.fail_attempts:
            raise AcquisitionError(f"synthetic transient failure {self.calls}")
        rng = np.random.default_rng(self.seed + self.calls)
        axes = list(plan.axes)
        if plan.quick_class == "LoopBack":
            x = np.linspace(0.0, 8.0, 500)
            envelope = ((x > 1.0) & (x < 3.0)).astype(float)
            iq = envelope * np.exp(0.2j) + 0.01 * (
                rng.normal(size=x.size) + 1j * rng.normal(size=x.size)
            )
            data = np.column_stack(
                (x, np.abs(iq), np.angle(iq), iq.real, iq.imag)
            )
        elif plan.quick_class == "IQScatter":
            shots = int(plan.variables.get("rep", 2000))
            ground = -0.5 + 0.12 * (
                rng.normal(size=shots) + 1j * rng.normal(size=shots)
            )
            excited = 0.5 + 0.12 * (
                rng.normal(size=shots) + 1j * rng.normal(size=shots)
            )
            data = np.column_stack(
                (ground.real, ground.imag, excited.real, excited.imag)
            )
        else:
            if not axes:
                raise AcquisitionError("synthetic swept experiment has no axis")
            axis_arrays = [
                np.asarray(plan.variables[name], dtype=float) for name in axes
            ]
            grids = [
                grid.ravel()
                for grid in np.meshgrid(*axis_arrays, indexing="ij")
            ]
            signal_axis = next(
                (index for index, name in enumerate(axes) if "freq" in name),
                0,
            )
            x = grids[signal_axis]
            center = float(np.mean(axis_arrays[signal_axis]))
            span = max(float(np.ptp(axis_arrays[signal_axis])), np.finfo(float).eps)
            if plan.quick_class in {
                "ResonatorSpectroscopy",
                "QubitSpectroscopy",
                "DispersiveSpectroscopy",
            }:
                feature = 0.4 / (1 + ((x - center) / (span / 15)) ** 2)
                iq = 1.0 - feature * np.exp(0.4j)
            elif plan.quick_class == "T1":
                iq = 0.2 + 0.8 * np.exp(-(x - x.min()) / (span / 3))
            else:
                iq = 0.5 + 0.4 * np.exp(-(x - x.min()) / max(span, 1e-9)) * np.cos(
                    2 * np.pi * 4 * (x - x.min()) / span + 0.2
                )
            iq = iq + 0.005 * (
                rng.normal(size=x.size) + 1j * rng.normal(size=x.size)
            )
            columns = list(grids)
            if plan.quick_class == "DispersiveSpectroscopy":
                ground = iq
                excited = iq + 0.15 * np.exp(-0.3j) * np.exp(
                    -((x - center) / (span / 6)) ** 2
                )
                columns.extend(
                    [
                        np.abs(ground),
                        np.angle(ground),
                        ground.real,
                        ground.imag,
                        np.abs(excited),
                        np.angle(excited),
                        excited.real,
                        excited.imag,
                    ]
                )
            else:
                if bool(plan.run_options.get("population", False)):
                    population = np.clip(iq.real, 0.0, 1.0)
                    columns.append(population)
                columns.extend([np.abs(iq), np.angle(iq), iq.real, iq.imag])
            data = np.column_stack(columns)
        return BackendResult(
            payload=data,
            metadata={
                "backend": "synthetic",
                "seed": self.seed,
                "call": self.calls,
            },
        )

    def recover(self, error: BaseException, attempt: int) -> dict:
        self.recoveries += 1
        return {"attempt": attempt, "error": str(error), "actions": ["synthetic_recover"]}

    def snapshot(self) -> dict:
        return {
            "backend": "synthetic",
            "seed": self.seed,
            "calls": self.calls,
            "recoveries": self.recoveries,
        }

    def close(self) -> None:
        self.connected = False
