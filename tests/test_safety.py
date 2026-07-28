import pytest

from quickexp_v3.errors import SafetyError
from quickexp_v3.safety import CallbackFluxController


def test_flux_bounds_reset_rearm_and_park():
    writes = []
    flux = CallbackFluxController(
        writes.append,
        minimum=-0.5,
        maximum=0.5,
        park_value=0.0,
        unit="dac_gain",
        reset_sensitive=True,
        sleep=lambda _: None,
    )
    flux.set(0.2)
    assert flux.safe_for_acquisition
    flux.notify_generator_reset()
    assert not flux.safe_for_acquisition
    flux.recover_after_reset()
    assert flux.safe_for_acquisition
    flux.park()
    assert writes == [0.2, 0.2, 0.0]


def test_external_flux_ramps_and_verifies_readback():
    writes = []
    current = {"value": 0.0}

    def set_value(value):
        writes.append(value)
        current["value"] = value

    flux = CallbackFluxController(
        set_value,
        read_value=lambda: current["value"],
        minimum=-1.0,
        maximum=1.0,
        park_value=0.0,
        unit="V",
        max_step=0.1,
        tolerance=1e-6,
        sleep=lambda _: None,
    )
    flux.set(0.25)
    flux.set(-0.05)
    assert max(
        abs(right - left)
        for left, right in zip([0.0] + writes[:-1], writes)
    ) <= 0.1000001


def test_flux_refuses_out_of_range_setpoint():
    flux = CallbackFluxController(
        lambda _: None,
        minimum=-0.5,
        maximum=0.5,
        park_value=0.0,
        unit="V",
    )
    with pytest.raises(SafetyError, match="outside"):
        flux.set(0.6)
