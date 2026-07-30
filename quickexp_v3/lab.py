"""Live Quick/QICK connection, port verification, RF setup, and held Z bias."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .backend import (
    QuickBackend,
    clean_qick_acquisition_state,
    logical_channel_variables,
)
from .config import ConfigRepository
from .errors import ConfigError, SafetyError
from .ports import errors as port_errors
from .ports import from_soccfg, report as port_report
from .safety import CallbackFluxController, FluxReading


HELD_Z_EXPERIMENT_NAME = "SetZBiasAndWaitV3"
HELD_Z_CONFIG = """
hard_avg: 1
soft_avg: 1

p0_freq: {r_freq}
p0_mixer: {int(r_freq/1000)*1000+500}
p0_length: {r_length}
p0_power: {r_power}
r{rr}_p: 0
r{rr}_length: {r_length}
r{rr}_phase: {r_phase}

p9_style: const
p9_mode: last
p9_freq: 0
p9_mixer: 0
p9_nqz: 1
p9_length: {z_length}
p9_gain: {z_gain}

steps:
- type: pulse
  p: 9
  g: {z}
- type: delay_auto
  t: {z_settle}
- type: pulse
  p: 0
  g: {r}
- type: trigger
  t: {r_offset}
- type: delay_auto
  t: {r_relax}
"""

def configure_quick_progress(quick_module: Any, mode: str = "terminal") -> str:
    """Bind Quick's Sweep to a tqdm renderer suitable for the run surface."""
    normalized = str(mode).strip().lower()
    if normalized == "terminal":
        from tqdm.std import tqdm as progress
    elif normalized == "notebook":
        from tqdm.notebook import tqdm as progress
    elif normalized == "auto":
        from tqdm.auto import tqdm as progress
    else:
        raise ConfigError(
            "qick.progress_mode must be terminal, notebook, or auto"
        )
    helper = getattr(quick_module, "helper", None)
    if helper is None:
        return "unavailable (test/dummy Quick module)"
    helper.tqdm = progress
    return f"{progress.__module__}.{progress.__name__}"

@dataclass
class QuickConnection:
    quick: Any
    soccfg: Any
    soc: Any
    backend: QuickBackend
    channels: Mapping[str, Any]

    def close(self) -> None:
        self.backend.close()


def _call_clear_interrupts(soc: Any) -> None:
    callback = getattr(soc, "clear_interrupts", None)
    if not callable(callback):
        return
    try:
        cleared = callback(
            max_attempts=5,
            error_on_interrupt=False,
            error_on_persist=False,
        )
    except TypeError:
        cleared = callback()
    if cleared is False:
        raise SafetyError(
            "ADC interrupt flags persist after clearing; reduce the signal "
            "into the ADC and check the readout path"
        )


def apply_rf_board_settings(
    soc: Any,
    repository: ConfigRepository,
) -> None:
    config = repository.hardware.get("rf_board", {})
    if not bool(config.get("apply_on_connect", False)):
        print("RF-board programming disabled by hardware.yml.")
        return
    channels = repository.hardware["channels"]
    for role, settings in config.get("generators", {}).items():
        if not isinstance(settings, Mapping) or not settings.get("enabled", True):
            continue
        channel = int(settings.get("channel", channels[role]["gen"]))
        soc.rfb_set_gen_rf(
            channel,
            settings.get("attenuator_1_db", 0),
            settings.get("attenuator_2_db", 0),
        )
        filter_settings = settings.get("filter")
        if isinstance(filter_settings, Mapping):
            soc.rfb_set_gen_filter(
                channel,
                fc=filter_settings.get("center_ghz", 0),
                ftype=filter_settings.get("type", "bypass"),
                bw=filter_settings.get("bandwidth_ghz"),
            )
    for role, settings in config.get("readouts", {}).items():
        if not isinstance(settings, Mapping) or not settings.get("enabled", True):
            continue
        channel = int(settings.get("channel", channels[role]["ro"]))
        soc.rfb_set_ro_rf(channel, settings.get("attenuation_db", 0))
        filter_settings = settings.get("filter")
        if isinstance(filter_settings, Mapping):
            soc.rfb_set_ro_filter(
                channel,
                fc=filter_settings.get("center_ghz", 0),
                ftype=filter_settings.get("type", "bypass"),
                bw=filter_settings.get("bandwidth_ghz", 1),
            )
    for settings in config.get("biases", {}).values():
        if isinstance(settings, Mapping) and settings.get("enabled", False):
            soc.rfb_set_bias(int(settings["channel"]), float(settings["value"]))


def connect_quick(
    repository: ConfigRepository,
    *,
    strict_ports: bool = True,
    apply_rf_board: Optional[bool] = None,
) -> QuickConnection:
    try:
        import quick
    except ImportError as error:
        raise RuntimeError(
            "LIVE_HARDWARE requires the qcodes environment with quick installed"
        ) from error
    qick = repository.hardware["qick"]
    actual_version = str(getattr(quick, "__version__", "unknown"))
    expected_version = qick.get("qick_version")
    if (
        expected_version not in (None, "", "pin-at-session-start")
        and str(expected_version) != actual_version
    ):
        raise ConfigError(
            f"hardware.yml expects Quick {expected_version}, "
            f"but this interpreter has {actual_version}"
        )
    print(f"Using Quick {actual_version}.")
    if hasattr(quick, "experiment"):
        install_authored_programs(
            quick,
            channel_variables=logical_channel_variables(repository.hardware),
        )
    progress_binding = configure_quick_progress(
        quick,
        qick.get("progress_mode", "terminal"),
    )
    if bool(qick.get("show_progress", True)):
        print(f"Quick progress renderer: {progress_binding}")
    host = str(qick["host"])
    port = int(qick.get("pyro_port", 8888))
    proxy_name = str(qick.get("proxy_name", "qick"))
    print(f"Connecting to Quick/QICK at {host}:{port} ({proxy_name!r})...")
    soccfg, soc = quick.connect(host, port=port, proxy_name=proxy_name)
    print(str(soccfg).splitlines()[0])

    resolved = from_soccfg(soccfg, repository.hardware["channels"])
    print(port_report(resolved))
    problems = port_errors(resolved)
    if problems and strict_ports:
        raise ConfigError(
            "connected QICK topology does not match hardware.yml: "
            + "; ".join(problems)
        )

    readout_cleanup = clean_qick_acquisition_state(soc)
    print("QICK acquisition engine stopped/drained before experiment setup.")
    for warning in readout_cleanup["warnings"]:
        print(f"QICK setup cleanup warning: {warning}")

    if bool(qick.get("reset_generators_on_connect", True)):
        soc.reset_gens()
        print("Generators reset before experiment setup.")
    if bool(qick.get("clear_interrupts_on_connect", True)):
        _call_clear_interrupts(soc)
        print("ADC interrupt flags checked/cleared.")

    if apply_rf_board is None:
        apply_rf_board_settings(soc, repository)
    elif apply_rf_board:
        rf_config = repository.hardware.setdefault("rf_board", {})
        original = rf_config.get("apply_on_connect")
        rf_config["apply_on_connect"] = True
        try:
            apply_rf_board_settings(soc, repository)
        finally:
            rf_config["apply_on_connect"] = original

    storage = repository.hardware.get("storage", {})
    native_output = bool(storage.get("quick_native_output", True))
    native_data_path = None
    if native_output:
        native_root = Path(
            storage.get("quick_native_root", "./data")
        ).expanduser().resolve()
        native_root.mkdir(parents=True, exist_ok=True)
        native_data_path = str(native_root)
        print(f"Quick native CSV/YML output: {native_data_path}")

    backend = QuickBackend(
        soc=soc,
        soccfg=soccfg,
        quick_module=quick,
        data_path=native_data_path,
        show_progress=bool(qick.get("show_progress", True)),
    )
    backend.connect()
    return QuickConnection(quick, soccfg, soc, backend, resolved)


def install_held_z_experiment(quick_module: Any):
    existing = getattr(quick_module.experiment, HELD_Z_EXPERIMENT_NAME, None)
    if existing is not None:
        return existing
    quick_module.experiment.configs[HELD_Z_EXPERIMENT_NAME] = HELD_Z_CONFIG
    base_experiment = quick_module.experiment.BaseExperiment

    class SetZBiasAndWaitV3(base_experiment):
        def __init__(self, **kwargs):
            self.var = {
                "z": 15,
                "z_gain": 0.0,
                "z_length": 0.2,
                "z_settle": 5.0,
            }
            self.var_label = {
                "z": ("Z DAC Generator", ""),
                "z_gain": ("Z Gain", "a.u."),
                "z_length": ("Z Hold Pulse Length", "us"),
                "z_settle": ("Z Settle Time", "us"),
            }
            super().__init__(**kwargs)

        def run(self, silent=False, dB=True):
            return super().run(silent=silent, dB=dB)

    SetZBiasAndWaitV3.__name__ = HELD_Z_EXPERIMENT_NAME
    setattr(quick_module.experiment, HELD_Z_EXPERIMENT_NAME, SetZBiasAndWaitV3)
    return SetZBiasAndWaitV3


def install_authored_programs(
    quick_module: Any,
    *,
    channel_variables: Optional[Mapping[str, int]] = None,
) -> Mapping[str, Any]:
    """Install every v3-authored Mercator program idempotently."""
    from .mercator import install_program
    from .programs import PROGRAMS

    return {
        program.name: install_program(
            quick_module,
            program,
            channel_variables=channel_variables,
        )
        for program in PROGRAMS
    }


class QuickHeldFluxController(CallbackFluxController):
    """Notebook-compatible held-Z controller that resets after parking at zero."""

    def __init__(self, *args: Any, soc: Any, **kwargs: Any):
        self.soc = soc
        super().__init__(*args, reset_sensitive=True, **kwargs)

    def park(self) -> FluxReading:
        reading = super().park()
        self.soc.reset_gens()
        self.notify_generator_reset()
        return reading


def _scalarize_held_z_variables(parameters: Mapping[str, Any]) -> dict:
    """Remove every numeric sweep array from the auxiliary held-Z program."""
    variables = dict(parameters)
    for name, value in list(variables.items()):
        if isinstance(value, (str, bytes, Mapping)):
            continue
        try:
            values = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if values.ndim == 0:
            continue
        values = values.ravel()
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ConfigError(
                f"held-Z helper requires finite, non-empty {name} values"
            )
        if name.endswith("_freq"):
            selected = values[values.size // 2]
        elif name.endswith(("_power", "_gain", "_length")):
            selected = np.min(values)
        else:
            selected = values[0]
        variables[name] = float(selected)
    return variables

def make_held_flux_controller(
    connection: QuickConnection,
    repository: ConfigRepository,
    parameters: Mapping[str, Any],
    *,
    line_name: str = "z_rf",
) -> QuickHeldFluxController:
    lines = repository.hardware.get("flux_lines", {})
    if line_name not in lines:
        raise ConfigError(f"unknown hardware flux line {line_name!r}")
    line = lines[line_name]
    if line.get("backend") != "rf_dac":
        raise ConfigError(f"flux line {line_name!r} is not an rf_dac line")
    experiment_class = install_held_z_experiment(connection.quick)
    # This auxiliary program only establishes a held Z value. Scalarize every
    # numeric sweep first so no 2D experiment can leak a list into Mercator.
    variables = _scalarize_held_z_variables(parameters)
    # This program is only a carrier for the persistent Z pulse. Keep its
    # dummy acquisition independent of the measurement sweep's long pulse and
    # relax settings, matching the notebook's one-average helper.
    variables["r_length"] = min(
        float(variables["r_length"]),
        float(line.get("dummy_readout_length_us", 2.0)),
    )
    variables["r_relax"] = min(
        float(variables["r_relax"]),
        float(line.get("dummy_relax_us", 2.0)),
    )
    variables.update(
        {
            "r": int(repository.hardware["channels"]["r"]["gen"]),
            "rr": int(repository.hardware["channels"]["rr"]["ro"]),
            "q": int(repository.hardware["channels"]["q"]["gen"]),
            "z": int(repository.hardware["channels"]["z"]["gen"]),
            "z_length": float(
                variables.get("z_length", line.get("hold_pulse_us", 0.2))
            ),
            "z_settle": float(
                variables.get("z_settle", line.get("program_settle_us", 5.0))
            ),
        }
    )
    if not np.isfinite(variables["z_length"]) or variables["z_length"] <= 0:
        raise ConfigError("z_length must be finite and greater than zero")
    if not np.isfinite(variables["z_settle"]) or variables["z_settle"] < 0:
        raise ConfigError("z_settle must be finite and non-negative")
    carrier_frequency = float(variables["r_freq"])
    print(
        "Held-Z helper: "
        f"length={variables['z_length']:.6f} us, "
        f"program settle={variables['z_settle']:.6f} us; "
        "dummy readout "
        f"{carrier_frequency:.6f} MHz, "
        f"{variables['r_power']:.3f} dB, "
        f"{variables['r_length']:.6f} us"
    )

    def set_value(value: float) -> Any:
        run_variables = dict(variables)
        run_variables["z_gain"] = float(value)
        for key in ("hard_avg", "soft_avg", "rep"):
            run_variables.pop(key, None)
        print(f"Establishing held Z gain {value:+.6f}...")
        completed = experiment_class(
            var=run_variables,
            soc=connection.soc,
            soccfg=connection.soccfg,
            data_path=None,
            title=f"SetZ_{value:+.6f}",
            z_gain=float(value),
            r_freq=np.asarray([carrier_frequency]),
            hard_avg=1,
            soft_avg=1,
        ).run(dB=True, silent=True)
        print(f"Held Z gain {value:+.6f} established.")
        return completed

    return QuickHeldFluxController(
        set_value,
        soc=connection.soc,
        minimum=float(line["minimum"]),
        maximum=float(line["maximum"]),
        park_value=float(line.get("park", 0.0)),
        unit=str(line.get("unit", "dac_gain")),
        settle_seconds=0.0,
    )
