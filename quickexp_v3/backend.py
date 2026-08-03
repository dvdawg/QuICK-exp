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
        "z", "z_gain", "z_length", "z_settle",
    }
)
QUICK_CLASS_VARIABLES = {
    "Rabi": frozenset({"cycle"}),
    "T1": frozenset({"time"}),
    "T2Ramsey": frozenset({"time", "fringe_freq"}),
    "T2Echo": frozenset({"time", "cycle", "fringe_freq"}),
    "IQScatter": frozenset({"rr_length"}),
}
QUICK_CONFIG_OVERRIDES = (
    "hard_avg",
    "soft_avg",
    "rep",
    # Mercator pulse-generator settings are literal config keys rather than
    # BaseExperiment variables. Keeping them here lets launchers select the
    # physical DAC Nyquist image without editing the installed Quick package.
    "p0_nqz",
    "p1_nqz",
    "p2_nqz",
    "p3_nqz",
)
QUICK_CONFIG_OVERRIDES_BY_CLASS = {
    # Decimated data for every hard repetition must fit in the on-board
    # accumulation buffer. The installed LoopBack template uses soft_avg and
    # leaves hard_avg/reps at one, matching the MET notebook.
    "LoopBack": ("soft_avg",),
}

# Quick 0.7.2 uploads these Gaussian pulse envelopes into the q-generator
# waveform memory. T2Ramsey/T2Echo allocate one reset/pi envelope plus two
# pi/2 envelopes, so their lengths are cumulative.
QUICK_Q_ENVELOPE_TERMS = {
    "Rabi": (("q_length", 1),),
    "T1": (("q_length", 1),),
    "IQScatter": (("q_length", 1),),
    "DispersiveSpectroscopy": (("q_length", 1),),
    "T2Ramsey": (("q_length", 1), ("q_length_2", 2)),
    "T2Echo": (("q_length", 1), ("q_length_2", 2)),
}
_REGISTERED_CLASS_VARIABLES = {}
_REGISTERED_ENVELOPE_TERMS = {}


def _nyquist_zone(name: str, value: Any) -> int:
    """Coerce a ``p*_nqz`` override to the DAC's zone 1 or zone 2."""
    try:
        zone = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be Nyquist zone 1 or 2") from error
    if zone not in (1, 2) or float(value) != zone:
        raise ConfigError(f"{name} must be Nyquist zone 1 or 2")
    return zone


def register_program_variables(quick_class: str, names: Any) -> frozenset:
    """Register constructor variables for one authored Quick class."""
    class_name = str(quick_class).strip()
    normalized = frozenset(str(name).strip() for name in names)
    if not class_name or not normalized or "" in normalized:
        raise ConfigError("authored program variables require non-empty names")
    collisions = normalized.intersection(QUICK_CONFIG_OVERRIDES)
    if collisions:
        raise ConfigError(
            "authored program variables collide with Mercator config overrides: "
            + ", ".join(sorted(collisions))
        )
    existing = _REGISTERED_CLASS_VARIABLES.get(class_name)
    if existing is not None and existing != normalized:
        raise ConfigError(
            f"variables for authored program {class_name!r} are already registered"
        )
    _REGISTERED_CLASS_VARIABLES[class_name] = normalized
    return normalized


def register_envelope_terms(quick_class: str, terms: Any) -> tuple:
    """Register the legacy-compatible envelope-memory terms for a program."""
    class_name = str(quick_class).strip()
    normalized = []
    for term in terms:
        if not isinstance(term, (tuple, list)) or len(term) != 2:
            raise ConfigError("envelope terms must be (variable_name, count) pairs")
        name, count = str(term[0]).strip(), int(term[1])
        if not name or count < 1:
            raise ConfigError("envelope terms require a name and positive count")
        normalized.append((name, count))
    result = tuple(normalized)
    existing = _REGISTERED_ENVELOPE_TERMS.get(class_name)
    if existing is not None and existing != result:
        raise ConfigError(
            f"envelope terms for authored program {class_name!r} are already registered"
        )
    _REGISTERED_ENVELOPE_TERMS[class_name] = result
    return result


def _maximum_envelope_samples(
    value: Any,
    *,
    f_fabric: float,
    samples_per_clock: int,
    label: str,
) -> int:
    durations = np.asarray(value, dtype=float)
    if durations.size == 0 or not np.all(np.isfinite(durations)):
        raise ConfigError(f"{label} must contain finite pulse lengths")
    if np.any(durations <= 0):
        raise ConfigError(f"{label} pulse lengths must be positive")
    # This intentionally matches quick.mercator.generate_waveform:
    # int(length_us * f_fabric) * samps_per_clk.
    fabric_clocks = np.asarray(
        [int(float(length) * f_fabric) for length in durations.ravel()],
        dtype=int,
    )
    return int(np.max(fabric_clocks) * samples_per_clock)


def validate_quick_envelope_memory(plan: Any, soccfg: Any) -> Optional[dict]:
    """Reject Quick Gaussian waveforms that cannot fit the q-generator RAM."""
    terms = QUICK_Q_ENVELOPE_TERMS.get(
        plan.quick_class,
        _REGISTERED_ENVELOPE_TERMS.get(plan.quick_class),
    )
    if terms is None:
        return None
    variables = dict(plan.variables)
    try:
        q_channel = int(variables["q"])
        generator = soccfg["gens"][q_channel]
        f_fabric = float(generator["f_fabric"])
        samples_per_clock = int(generator["samps_per_clk"])
        available_samples = int(generator["maxlen"])
    except (KeyError, TypeError, ValueError, IndexError):
        # Lightweight fake soccfg objects used by offline tests do not expose
        # generator memory metadata. Live QICK soccfg always does.
        return None

    required_samples = 0
    contributions = {}
    primary_length = variables.get("q_length")
    for name, multiplier in terms:
        value = variables.get(name)
        if name == "q_length_2" and (
            value is None or np.all(np.asarray(value) == 0)
        ):
            value = primary_length
        if value is None:
            raise ConfigError(f"{plan.name} requires {name}")
        samples = _maximum_envelope_samples(
            value,
            f_fabric=f_fabric,
            samples_per_clock=samples_per_clock,
            label=f"{plan.name}.{name}",
        )
        contributions[name] = {
            "samples_per_envelope": samples,
            "envelope_count": int(multiplier),
        }
        required_samples += int(multiplier) * samples

    maximum_single_envelope_us = (
        available_samples / (f_fabric * samples_per_clock)
    )
    details = {
        "q_channel": q_channel,
        "required_samples": required_samples,
        "available_samples": available_samples,
        "maximum_single_envelope_us": maximum_single_envelope_us,
        "contributions": contributions,
    }
    if required_samples > available_samples:
        largest_requested = float(np.max(np.asarray(primary_length, dtype=float)))
        raise ConfigError(
            f"{plan.name} Gaussian envelope request needs "
            f"{required_samples} samples on q generator {q_channel}, but its "
            f"waveform memory holds {available_samples}. Requested q_length "
            f"reaches {largest_requested:.6f} us; one envelope is nominally "
            f"limited to {maximum_single_envelope_us:.6f} us on this bitfile. "
            "Shorten the pulse-length sweep before running."
        )
    return details


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

    def validate(self, plan: Any) -> Optional[dict]:
        """Validate hardware-dependent plan limits without acquiring."""
        callback = plan.metadata.get("preflight")
        if not callable(callback):
            return validate_quick_envelope_memory(plan, self.soccfg)
        # Authored programs carry a style-aware validator. Running the legacy
        # class table as well would incorrectly charge a flat-top's full
        # duration instead of its waveform ramps.
        envelope = None
        report = callback(self.soccfg, plan.variables)
        errors = tuple(getattr(report, "errors", ()))
        if errors:
            raise ConfigError(
                f"{plan.name} Mercator preflight failed: " + "; ".join(errors)
            )
        return {
            "envelope": envelope,
            "program": dict(getattr(report, "details", {})),
            "warnings": list(getattr(report, "warnings", ())),
        }

    def acquire(self, plan: Any) -> BackendResult:
        if not self.connected:
            self.connect()
        self.validate(plan)
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
        ).union(
            _REGISTERED_CLASS_VARIABLES.get(plan.quick_class, frozenset())
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
        for name, value in config_arguments.items():
            if name.endswith("_nqz"):
                config_arguments[name] = _nyquist_zone(name, value)
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
    """Deterministic backend for config checks, demos, and loop tests."""

    def __init__(
        self,
        seed: int = 7,
        fail_attempts: int = 0,
        device: Any = None,
    ):
        self.seed = int(seed)
        self.fail_attempts = int(fail_attempts)
        self.device = device
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
        if self.device is not None:
            names = (str(plan.name), str(plan.quick_class))
            if any(
                self.device.consume_failure(name)
                for name in dict.fromkeys(names)
            ):
                raise AcquisitionError(
                    f"synthetic device injected failure for {plan.name}"
                )
        rng = np.random.default_rng(self.seed + self.calls)
        if self.device is not None:
            return self._acquire_device(plan, rng)
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
            signal_axis = (
                axes.index("time")
                if plan.quick_class in {"T1", "T1_zpa"} and "time" in axes
                else next(
                    (index for index, name in enumerate(axes) if "freq" in name),
                    0,
                )
            )
            x = grids[signal_axis]
            center = float(np.mean(axis_arrays[signal_axis]))
            span = max(float(np.ptp(axis_arrays[signal_axis])), np.finfo(float).eps)
            if plan.quick_class in {
                "ResonatorSpectroscopy",
                "QubitSpectroscopy",
                "TwoTone_ZPA",
                "DispersiveSpectroscopy",
            }:
                feature = 0.4 / (1 + ((x - center) / (span / 15)) ** 2)
                iq = 1.0 - feature * np.exp(0.4j)
            elif plan.quick_class in {"T1", "T1_zpa"}:
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

    def _acquire_device(self, plan: Any, rng: Any) -> BackendResult:
        device = self.device
        axes = list(plan.axes)
        truth = {
            "elapsed_hours": float(device.elapsed_hours),
        }
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
            ground_mean = np.asarray(device.readout_ground_mean, dtype=float)
            excited_mean = np.asarray(device.readout_excited_mean, dtype=float)
            ground = rng.multivariate_normal(
                ground_mean,
                np.asarray(device.readout_ground_covariance, dtype=float),
                size=shots,
            )
            excited = rng.multivariate_normal(
                excited_mean,
                np.asarray(device.readout_excited_covariance, dtype=float),
                size=shots,
            )
            thermal = rng.random(shots) < float(device.thermal_population)
            leakage = rng.random(shots) < float(device.leakage_probability)
            if np.any(thermal):
                ground[thermal] = rng.multivariate_normal(
                    excited_mean,
                    np.asarray(device.readout_excited_covariance, dtype=float),
                    size=int(np.count_nonzero(thermal)),
                )
            if np.any(leakage):
                excited[leakage] = rng.multivariate_normal(
                    1.8 * excited_mean - 0.8 * ground_mean,
                    1.5 * np.asarray(device.readout_excited_covariance, dtype=float),
                    size=int(np.count_nonzero(leakage)),
                )
            data = np.column_stack(
                (ground[:, 0], ground[:, 1], excited[:, 0], excited[:, 1])
            )
            truth.update(
                {
                    "thermal_population": float(device.thermal_population),
                    "leakage_probability": float(device.leakage_probability),
                }
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
            points = grids[0].size

            def variable(name, default=0.0):
                if name in axes:
                    return grids[axes.index(name)]
                value = np.asarray(plan.variables.get(name, default), dtype=float)
                if value.size == 1:
                    return np.full(points, float(value.ravel()[0]))
                if value.size == points:
                    return value.ravel()
                return np.full(points, float(value.ravel()[0]))

            noise_scale = (
                abs(float(device.spectroscopy_noise_std))
                if plan.quick_class
                in {"ResonatorSpectroscopy", "QubitSpectroscopy", "TwoTone_ZPA"}
                else 0.005
            )
            noise = noise_scale * (
                rng.normal(size=points) + 1j * rng.normal(size=points)
            )
            if plan.quick_class == "ResonatorSpectroscopy":
                frequency = variable("r_freq")
                center = device.resonator_frequency(
                    variable("r_power", -35.0),
                    variable("z_gain", 0.0),
                )
                detuning = (
                    2.0
                    * (frequency - center)
                    / max(float(device.resonator_linewidth_mhz), 1e-9)
                )
                iq = (
                    1.0
                    - 0.65 / (1.0 + 1j * detuning)
                    + device.extra_spectral_response(
                        "resonator",
                        frequency,
                        variable("z_gain", 0.0),
                        10.0 ** (variable("r_power", -35.0) / 20.0),
                    )
                    + noise
                )
                truth["resonator_center_mhz"] = np.asarray(center).tolist()
            elif plan.quick_class in {"QubitSpectroscopy", "TwoTone_ZPA"}:
                frequency = variable("q_freq")
                center = device.qubit_frequency(variable("z_gain", 0.0))
                linewidth = np.maximum(
                    device.qubit_linewidth(variable("q_gain", 0.0)),
                    1e-9,
                )
                detuning = (
                    2.0
                    * (frequency - center)
                    / linewidth
                )
                iq = (
                    0.1
                    + device.qubit_spectroscopy_strength(
                        variable("q_gain", 0.0)
                    )
                    / (1.0 + 1j * detuning)
                    + device.extra_spectral_response(
                        "qubit",
                        frequency,
                        variable("z_gain", 0.0),
                        variable("q_gain", 0.0),
                    )
                    + noise
                )
                truth["qubit_center_mhz"] = np.asarray(center).tolist()
                truth["qubit_linewidth_mhz"] = np.asarray(
                    linewidth
                ).tolist()
            elif plan.quick_class == "DispersiveSpectroscopy":
                frequency = variable("r_freq")
                center = device.resonator_frequency(
                    variable("r_power", -35.0),
                    variable("z_gain", 0.0),
                )
                true_qubit = device.qubit_frequency(variable("z_gain", 0.0))
                qubit_detuning = variable("q_freq", true_qubit) - true_qubit
                excitation_width = np.maximum(
                    device.qubit_linewidth(variable("q_gain", 0.0)),
                    1.0e-9,
                )
                excitation = 1.0 / (
                    1.0 + (2.0 * qubit_detuning / excitation_width) ** 2
                )
                observed_shift = float(device.dispersive_shift_mhz) * excitation
                width = max(float(device.resonator_linewidth_mhz), 1e-9)
                ground = 1.0 - 0.5 / (
                    1.0 + 2j * (frequency - center) / width
                ) + noise
                excited = 1.0 - 0.5 / (
                    1.0
                    + 2j
                    * (
                        frequency
                        - center
                        - observed_shift
                    )
                    / width
                ) + noise
                columns = list(grids)
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
                data = np.column_stack(columns)
                truth["dispersive_shift_mhz"] = np.asarray(
                    observed_shift
                ).tolist()
                return BackendResult(
                    payload=data,
                    metadata={
                        "backend": "synthetic",
                        "seed": self.seed,
                        "call": self.calls,
                        "device_model": truth,
                    },
                )
            elif plan.quick_class == "Rabi":
                q_frequency = variable(
                    "q_freq",
                    float(device.qubit_frequency(variable("z_gain", 0.0))[0]),
                )
                true_frequency = device.qubit_frequency(variable("z_gain", 0.0))
                detuning = q_frequency - true_frequency
                omega = device.rabi_rate(variable("q_gain", 0.4))
                generalized = np.sqrt(omega**2 + detuning**2)
                duration = variable("q_length", 0.115)
                population = (
                    omega**2
                    / np.maximum(generalized**2, np.finfo(float).eps)
                    * np.sin(np.pi * generalized * duration) ** 2
                    * np.exp(-duration / max(4.0 * device.t1_us, 1e-9))
                )
                iq = (0.12 + 0.75 * population) * np.exp(0.3j) + noise
                truth.update(
                    {
                        "qubit_center_mhz": np.asarray(true_frequency).tolist(),
                        "rabi_rate_mhz": np.asarray(omega).tolist(),
                    }
                )
            elif plan.quick_class in {"T1", "T1_zpa"}:
                time = variable("time")
                decay = (
                    device.t1_at_flux(variable("z_gain", 0.0))
                    if plan.quick_class == "T1_zpa"
                    else device.coherence_time("t1")
                )
                population = 0.08 + 0.82 * np.exp(
                    -(time - np.min(time)) / np.maximum(decay, 1e-9)
                )
                iq = population * np.exp(0.25j) + noise
                truth["t1_us"] = np.asarray(decay).tolist()
            elif plan.quick_class == "T2Ramsey":
                time = variable("time")
                true_frequency = device.qubit_frequency(variable("z_gain", 0.0))
                drive = variable("q_freq", true_frequency)
                fringe = variable("fringe_freq", 0.0)
                oscillation = fringe + true_frequency - drive
                decay = device.coherence_time("ramsey")
                population = (
                    0.5
                    + 0.42
                    * np.exp(-(time - np.min(time)) / max(decay, 1e-9))
                    * np.cos(2.0 * np.pi * oscillation * time + 0.25)
                )
                iq = population * np.exp(0.25j) + noise
                truth.update(
                    {
                        "t2_ramsey_us": decay,
                        "ramsey_frequency_mhz": np.asarray(oscillation).tolist(),
                    }
                )
            elif plan.quick_class == "T2Echo":
                time = variable("time")
                cycles = np.maximum(variable("cycle", 1.0), 0.0)
                decay = device.coherence_time("echo") * np.sqrt(cycles + 1.0)
                population = 0.08 + 0.82 * np.exp(
                    -(time - np.min(time)) / np.maximum(decay, 1e-9)
                )
                iq = population * np.exp(0.25j) + noise
                truth["t2_echo_us"] = np.asarray(decay).tolist()
            else:
                signal_axis = next(
                    (index for index, name in enumerate(axes) if "freq" in name),
                    0,
                )
                x = grids[signal_axis]
                span = max(float(np.ptp(axis_arrays[signal_axis])), 1e-9)
                iq = (
                    0.5
                    + 0.4
                    * np.exp(-(x - x.min()) / span)
                    * np.cos(2 * np.pi * 4 * (x - x.min()) / span + 0.2)
                    + noise
                )
            columns = list(grids)
            if bool(plan.run_options.get("population", False)):
                columns.append(np.clip(iq.real, 0.0, 1.0))
            columns.extend([np.abs(iq), np.angle(iq), iq.real, iq.imag])
            data = np.column_stack(columns)
        return BackendResult(
            payload=data,
            metadata={
                "backend": "synthetic",
                "seed": self.seed,
                "call": self.calls,
                "device_model": truth,
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
            "device_model": self.device is not None,
            "elapsed_hours": (
                float(self.device.elapsed_hours)
                if self.device is not None
                else None
            ),
        }

    def close(self) -> None:
        self.connected = False
