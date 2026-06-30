"""Inspect QICK port mappings and optionally test one low-power loopback path.

The default mode is read-only.  Test mode emits a short RF pulse, so connect a
known-safe loopback cable/attenuator and name the logical generator/readout
indices explicitly::

    python diagnose_ports.py
    python diagnose_ports.py --test 1 1

The logical<->physical mapping logic lives in ``labreadout.ports`` (pure, unit
-tested); this script is the thin CLI/hardware wrapper around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import quick

from labreadout import ports
from labreadout.config import load_config


def print_map(soccfg, configured: dict, expected_ports: dict) -> None:
    description = str(soccfg)
    rf_out = ports.parse_card_ports(description, "RF Out")
    rf_in = ports.parse_card_ports(description, "RF In")

    print("\nGenerators (logical index -> physical DAC port)")
    for index, gen in enumerate(soccfg["gens"]):
        tile, block = gen["dac"]
        port = ports.dac_port(tile, block)
        tags = []
        if index == configured.get("r"):
            tags.append("configured r")
        if index == configured.get("q"):
            tags.append("configured q")
        tags.append("RF-card" if port in rf_out else "direct/no RF card")
        print(f"  gen {index:2d} -> DAC {port:2d}  {', '.join(tags)}")

    print("\nReadouts (logical index -> physical ADC port)")
    for index, ro in enumerate(soccfg["readouts"]):
        tile, block = ro["adc"]
        port = ports.adc_port(tile, block)
        tags = []
        if index == configured.get("rr"):
            tags.append("configured rr")
        tags.append("RF-card" if port in rf_in else "direct/no RF card")
        print(f"  ro  {index:2d} -> ADC {port:2d}  {ro['ro_type']}, {', '.join(tags)}")

    print()
    resolved = ports.from_soccfg(soccfg, configured, expected_ports)
    print(ports.report(resolved))


def run_test(tx: int, rx: int, frequency: float, power: float, yes: bool) -> None:
    if not yes:
        answer = input(
            f"Emit a 2 us, {power:g} dB pulse at {frequency:g} MHz on logical "
            f"generator {tx}, reading logical channel {rx}? [y/N] "
        )
        if answer.strip().lower() not in {"y", "yes"}:
            print("Cancelled.")
            return

    var = dict(quick.experiment.var)
    var.update(
        r=tx, rr=rx, r_freq=frequency, r_power=power,
        r_length=2, r_offset=0, r_relax=2,
    )
    _, soc = quick.helper.getSoc()
    soc.clear_interrupts(max_attempts=5, error_on_interrupt=False, error_on_persist=False)
    try:
        result = quick.experiment.LoopBack(
            var=var, data_path=None, title="port-diagnostic", soft_avg=1
        ).run(silent=True)
    except ModuleNotFoundError as exc:
        if exc.name == "pynq":
            raise RuntimeError(
                "The board raised an ADC over-range interrupt. Reduce/disconnect "
                "the input signal; do not install pynq on this PC."
            ) from exc
        raise

    data = np.asarray(result.data, float)
    amplitude = data[:, 1]
    print(
        f"Captured {len(data)} samples: RMS amplitude="
        f"{np.sqrt(np.mean(amplitude**2)):.6g}, peak={np.max(np.abs(amplitude)):.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("hardware.yml")
    )
    parser.add_argument(
        "--test", nargs=2, type=int, metavar=("TX", "RX"),
        help="test one logical generator/readout pair",
    )
    parser.add_argument("--frequency", type=float, default=6000.0)
    parser.add_argument("--power", type=float, default=-40.0)
    parser.add_argument("--yes", action="store_true", help="skip RF-emission prompt")
    args = parser.parse_args()

    cfg = load_config(str(args.config))
    soccfg, _ = quick.connect(cfg.ip)
    print_map(soccfg, cfg.channels, cfg.expected_ports)
    if args.test:
        run_test(*args.test, args.frequency, args.power, args.yes)


if __name__ == "__main__":
    main()
