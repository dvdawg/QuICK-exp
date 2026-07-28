"""Edit the common lab settings here, then press the IDE Run button.

Set ``WRITE_CHANGES = True`` only when the values below are ready.  The script
validates the complete hardware/calibration/preset overlay before replacing any
YAML file.  Accepted r_freq, q_freq, and r_offset calibration records are kept
in sync because those records otherwise take precedence over hardware defaults.
"""

from copy import deepcopy
from datetime import date
import os
from pathlib import Path
import sys
import tempfile

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quickexp_v3.config import ConfigRepository


# ============================ EDIT THESE ====================================
# Safety latch: False prints and validates a preview; True writes the YAML.
WRITE_CHANGES = False

# Quick's native Saver writes one numbered .csv and .yml pair here.
DATA_OUTPUT_DIRECTORY = r"Z:\David\Data\2026-07-21_MET_ver191_qubit3"
QUICK_NATIVE_OUTPUT = True
SHOW_PROGRESS = True

DEVICE = {
    "name": "MET_v191",
    "chip": "MET",
    "package": "v191",
    "cooldown": "2026-07-21",
    "qubits": ["q3"],
    "resonators": ["r3"],
}

QICK_CONNECTION = {
    "host": "192.168.1.17",
    "pyro_port": 8888,
    "proxy_name": "qick",
    "qick_version": "0.7.2",
    "bitfile": "MET_v191",
}

# Logical Quick/QICK indices, not physical connector numbers.
CHANNEL_INDICES = {
    "r": 0,   # DAC0
    "rr": 0,  # ADC4
    "q": 1,   # DAC1
    "z": 15,  # DAC15
}

DEFAULT_PARAMETERS = {
    "r_freq": 6883.11,
    "r_power": -35.0,
    "r_length": 2.0,
    "r_offset": 0.495,
    "r_phase": 0.0,
    "r_relax": 20.0,
    "r_reset": 0,
    "r_threshold": 0.0,
    "q_freq": 5606.5,
    "q_gain": 0.4,
    "q_length": 0.115,
    "q_delta": -180.0,
    "q_gain_2": 0.2,
    "q_length_2": 0.115,
    "z_gain": 0.0,
    "hard_avg": 1000,
    "soft_avg": 1,
    "rep": 1,
}

HELD_Z = {
    "park": 0.0,
    "minimum": -0.8,
    "maximum": 0.8,
    "hold_pulse_us": 0.2,
    "program_settle_us": 5.0,
    "dummy_readout_length_us": 2.0,
    "dummy_relax_us": 2.0,
}

RF_BOARD_APPLY_ON_CONNECT = False

# Optional named-preset edits. Values here replace only the listed parameters.
# Example:
# PRESET_PARAMETER_UPDATES = {
#     "qubit_fine": {
#         "q_freq": {"center": 5606.5, "span": 20.0, "points": 201, "unit": "MHz"},
#         "hard_avg": 5000,
#     },
# }
PRESET_PARAMETER_UPDATES = {}
# ============================================================================


def _load(target: Path, example: Path) -> dict:
    source = target if target.exists() else example
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{source} must contain a YAML mapping")
    return loaded


def _atomic_yaml(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            yaml.safe_dump(
                document,
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _documents() -> tuple:
    hardware_path = PROJECT_ROOT / "hardware.yml"
    calibration_path = PROJECT_ROOT / "calibration.yml"
    presets_path = PROJECT_ROOT / "presets.yml"
    hardware = _load(hardware_path, PROJECT_ROOT / "hardware.example.yml")
    calibration = _load(
        calibration_path,
        PROJECT_ROOT / "calibration.example.yml",
    )
    presets = _load(presets_path, PROJECT_ROOT / "presets.example.yml")
    return (
        (hardware_path, calibration_path, presets_path),
        (hardware, calibration, presets),
    )


def main():
    paths, originals = _documents()
    hardware, calibration, presets = map(deepcopy, originals)

    hardware["storage"] = {
        "quick_native_output": bool(QUICK_NATIVE_OUTPUT),
        "quick_native_root": str(Path(DATA_OUTPUT_DIRECTORY).expanduser()),
    }
    hardware["device"].update(deepcopy(DEVICE))
    hardware["qick"].update(deepcopy(QICK_CONNECTION))
    hardware["qick"]["show_progress"] = bool(SHOW_PROGRESS)
    hardware["channels"]["r"]["gen"] = int(CHANNEL_INDICES["r"])
    hardware["channels"]["rr"]["ro"] = int(CHANNEL_INDICES["rr"])
    hardware["channels"]["q"]["gen"] = int(CHANNEL_INDICES["q"])
    hardware["channels"]["z"]["gen"] = int(CHANNEL_INDICES["z"])
    hardware["defaults"].update(deepcopy(DEFAULT_PARAMETERS))
    hardware["flux_lines"]["z_rf"].update(deepcopy(HELD_Z))
    hardware["rf_board"]["apply_on_connect"] = bool(
        RF_BOARD_APPLY_ON_CONNECT
    )

    accepted_defaults = (
        calibration.setdefault("records", {}).setdefault("defaults", {})
    )
    calibration_changed = False
    for name, record in accepted_defaults.items():
        if (
            name in DEFAULT_PARAMETERS
            and isinstance(record, dict)
            and record.get("status", "accepted") == "accepted"
        ):
            new_value = deepcopy(DEFAULT_PARAMETERS[name])
            if record.get("value") != new_value:
                record["value"] = new_value
                calibration_changed = True
    if calibration_changed:
        calibration["revision"] = int(calibration.get("revision", 0)) + 1
        calibration["updated_at"] = date.today().isoformat()

    available_presets = presets.setdefault("presets", {})
    for preset_name, changes in PRESET_PARAMETER_UPDATES.items():
        if preset_name not in available_presets:
            raise KeyError(f"unknown preset {preset_name!r}")
        if not isinstance(changes, dict):
            raise TypeError(
                f"PRESET_PARAMETER_UPDATES[{preset_name!r}] must be a dict"
            )
        available_presets[preset_name].setdefault("parameters", {}).update(
            deepcopy(changes)
        )

    # Constructing the repository validates routing, sweep syntax, and schema.
    ConfigRepository(hardware, calibration, presets)
    print("Configuration preview validated.")
    print(f"  Native Quick data: {hardware['storage']['quick_native_root']}")
    print(f"  Quick progress: {hardware['qick']['show_progress']}")
    print(
        "  Channels: "
        f"r={CHANNEL_INDICES['r']}, rr={CHANNEL_INDICES['rr']}, "
        f"q={CHANNEL_INDICES['q']}, z={CHANNEL_INDICES['z']}"
    )
    print(
        f"  r_freq={DEFAULT_PARAMETERS['r_freq']} MHz, "
        f"q_freq={DEFAULT_PARAMETERS['q_freq']} MHz, "
        f"r_offset={DEFAULT_PARAMETERS['r_offset']} us"
    )

    if not WRITE_CHANGES:
        print("WRITE_CHANGES=False: no YAML files were changed.")
        return False

    for path, original, updated in zip(
        paths,
        originals,
        (hardware, calibration, presets),
    ):
        if updated != original or not path.exists():
            _atomic_yaml(path, updated)
            print(f"Updated {path}")
        else:
            print(f"Unchanged {path}")
    print("Configuration write complete. Set WRITE_CHANGES=False again.")
    return True


if __name__ == "__main__":
    WROTE_CONFIGURATION = main()
