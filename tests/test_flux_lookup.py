import numpy as np
import pytest

from quickexp_v3.errors import ConfigError
from quickexp_v3.flux_lookup import frequency_from_record, register_model


def test_flux_lookup_dispatch_is_domain_checked_and_extensible():
    cosine = {
        "status": "accepted",
        "model": "cosine",
        "value": {
            "parameters": {
                "center_frequency": 100.0,
                "amplitude": 2.0,
                "period": 1.0,
                "peak_bias": 0.0,
            }
        },
        "valid_domain": {"z_gain": [-0.5, 0.5]},
    }
    assert frequency_from_record(cosine, 0.0) == pytest.approx(102.0)
    with pytest.raises(ConfigError, match="outside accepted"):
        frequency_from_record(cosine, 0.6)

    def linear(record, z_gain):
        values = np.asarray(z_gain, dtype=float)
        result = record["value"]["offset"] + record["value"]["slope"] * values
        return float(result) if result.ndim == 0 else result

    register_model("test_linear", linear)
    custom = {
        "status": "accepted",
        "model": "test_linear",
        "value": {"offset": 2.0, "slope": 3.0},
    }
    assert frequency_from_record(custom, 4.0) == pytest.approx(14.0)
