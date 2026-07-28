"""Callback-based flux control with bounded set, recovery, and parking.

Instrument- and QICK-release-specific calls stay outside this package.  The
controller owns the safety state machine and makes reset-sensitive RF holds
explicit to the runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterator, Mapping, Optional

from .errors import SafetyError
from .util import utc_now


@dataclass(frozen=True)
class FluxReading:
    requested: float
    applied: float
    readback: Optional[float]
    unit: str
    verified: bool
    time: str

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "applied": self.applied,
            "readback": self.readback,
            "unit": self.unit,
            "verified": self.verified,
            "time": self.time,
        }


class CallbackFluxController:
    """One reviewed flux line backed by injected hardware callbacks."""

    def __init__(
        self,
        set_value: Callable[[float], Any],
        *,
        minimum: float,
        maximum: float,
        park_value: float,
        unit: str,
        read_value: Optional[Callable[[], float]] = None,
        quantize: Optional[Callable[[float], float]] = None,
        settle_seconds: float = 0.0,
        tolerance: Optional[float] = None,
        max_step: Optional[float] = None,
        step_delay: float = 0.0,
        reset_sensitive: bool = False,
        close_callback: Optional[Callable[[], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.set_callback = set_value
        self.read_callback = read_value
        self.quantize = quantize
        self.minimum = self._finite(minimum, "minimum")
        self.maximum = self._finite(maximum, "maximum")
        self.park_value = self._finite(park_value, "park_value")
        if self.minimum > self.maximum:
            raise SafetyError("flux minimum cannot exceed maximum")
        if not self.minimum <= self.park_value <= self.maximum:
            raise SafetyError("flux park value is outside configured limits")
        self.unit = str(unit)
        self.settle_seconds = max(0.0, self._finite(settle_seconds, "settle_seconds"))
        self.tolerance = (
            None if tolerance is None else max(0.0, self._finite(tolerance, "tolerance"))
        )
        self.max_step = (
            None if max_step is None else self._finite(max_step, "max_step")
        )
        if self.max_step is not None and self.max_step <= 0:
            raise SafetyError("flux max_step must be positive")
        self.step_delay = max(0.0, self._finite(step_delay, "step_delay"))
        self.reset_sensitive = bool(reset_sensitive)
        self.close_callback = close_callback
        self.sleep = sleep
        self.last_reading: Optional[FluxReading] = None
        self.target: Optional[float] = None
        self.valid = False
        self.closed = False

    @staticmethod
    def _finite(value: Any, label: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise SafetyError(f"flux {label} must be a finite number") from error
        if not math.isfinite(number):
            raise SafetyError(f"flux {label} must be a finite number")
        return number

    def _bounded(self, value: Any) -> float:
        requested = self._finite(value, "setpoint")
        if not self.minimum <= requested <= self.maximum:
            raise SafetyError(
                f"flux setpoint {requested} {self.unit} is outside "
                f"[{self.minimum}, {self.maximum}]"
            )
        return requested

    def _steps(self, target: float) -> list:
        if self.max_step is None:
            return [target]
        if self.last_reading is not None:
            start = self.last_reading.applied
        elif self.read_callback is not None:
            # An external source may already be biased when this process starts.
            # Ramp from measured state instead of making an unbounded first jump.
            start = self._finite(self.read_callback(), "initial readback")
            self._bounded(start)
        else:
            # With no readback there is no trustworthy initial point from which
            # to synthesize a bounded ramp; the command-only jump is explicit.
            return [target]
        if abs(target - start) <= self.max_step:
            return [target]
        count = int(math.ceil(abs(target - start) / self.max_step))
        return [start + (target - start) * index / count for index in range(1, count + 1)]

    def set(self, value: Any, *, verify: bool = True) -> FluxReading:
        if self.closed:
            raise SafetyError("flux controller is closed")
        requested = self._bounded(value)
        applied = (
            self._bounded(self.quantize(requested))
            if self.quantize is not None
            else requested
        )
        self.valid = False
        steps = self._steps(applied)
        for index, step in enumerate(steps):
            self.set_callback(step)
            if index + 1 < len(steps) and self.step_delay:
                self.sleep(self.step_delay)
        if self.settle_seconds:
            self.sleep(self.settle_seconds)
        readback = None
        verified = False
        if self.read_callback is not None:
            readback = self._finite(self.read_callback(), "readback")
            tolerance = self.tolerance if self.tolerance is not None else 0.0
            verified = abs(readback - applied) <= tolerance
            if verify and not verified:
                raise SafetyError(
                    f"flux readback {readback} differs from {applied} "
                    f"by more than {tolerance} {self.unit}"
                )
        elif verify:
            # Command-only RF lines are explicit in the reading.
            verified = False
        reading = FluxReading(
            requested=requested,
            applied=applied,
            readback=readback,
            unit=self.unit,
            verified=verified,
            time=utc_now(),
        )
        self.target = requested
        self.last_reading = reading
        self.valid = True
        return reading

    def notify_generator_reset(self) -> None:
        if self.reset_sensitive:
            self.valid = False

    def recover_after_reset(self) -> FluxReading:
        if not self.reset_sensitive:
            if self.last_reading is None:
                raise SafetyError("flux has no known setpoint to recover")
            return self.last_reading
        if self.target is None:
            raise SafetyError("reset-sensitive flux has no known target to re-arm")
        return self.set(self.target)

    @property
    def safe_for_acquisition(self) -> bool:
        return bool(self.valid and not self.closed)

    def park(self) -> FluxReading:
        return self.set(self.park_value)

    def close(self) -> None:
        if self.closed:
            return
        if self.close_callback is not None:
            self.close_callback()
        self.closed = True
        self.valid = False

    @contextmanager
    def session(
        self,
        initial_value: Optional[float] = None,
        *,
        park_on_exit: bool = True,
    ) -> Iterator["CallbackFluxController"]:
        error: Optional[BaseException] = None
        try:
            if initial_value is not None:
                self.set(initial_value)
            yield self
        except BaseException as caught:
            error = caught
            raise
        finally:
            if park_on_exit:
                try:
                    self.park()
                except BaseException as cleanup:
                    if error is None:
                        raise
                    if hasattr(error, "add_note"):
                        error.add_note(f"flux parking also failed: {cleanup}")


def controller_from_hardware(
    line: Mapping[str, Any],
    *,
    set_value: Callable[[float], Any],
    read_value: Optional[Callable[[], float]] = None,
    close_callback: Optional[Callable[[], Any]] = None,
) -> CallbackFluxController:
    """Construct a controller from one validated ``hardware.flux_lines`` entry."""
    backend = str(line.get("backend", ""))
    return CallbackFluxController(
        set_value,
        minimum=line["minimum"],
        maximum=line["maximum"],
        park_value=line.get("park", 0.0),
        unit=line.get("unit", ""),
        read_value=read_value,
        settle_seconds=line.get("settle_seconds", 0.0),
        tolerance=line.get("readback_tolerance"),
        max_step=line.get("max_step") if backend == "external_dc" else None,
        step_delay=line.get("step_delay", 0.0),
        reset_sensitive=backend == "rf_dac",
        close_callback=close_callback,
    )

