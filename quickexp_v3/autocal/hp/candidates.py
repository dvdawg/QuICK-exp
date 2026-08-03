"""Turn a spectroscopy fit into ranked feature candidates without gates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    center_mhz: float
    fwhm_mhz: float
    contrast: float
    center_uncertainty_mhz: float
    local_snr: float
    rank: int
    source_csv: Path
    window_mhz: Tuple[float, float]
    is_null: bool = False
    statistics: Dict[str, float] = field(default_factory=dict)


def _candidate_id(source_csv: Path, center_mhz: float, rank: int) -> str:
    source = Path(source_csv).expanduser().resolve(strict=False).as_posix()
    identity = "{0}|{1:.6f}|{2}".format(
        source,
        float(center_mhz),
        int(rank),
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def extract_candidates(
    fit: Any,
    max_candidates: int = 5,
) -> Tuple[Candidate, ...]:
    """Return prominent-to-weak candidates followed by ``no_feature``.

    Model order has already been selected by BIC in
    :func:`fit_spectroscopy_features`. This function deliberately applies no
    fit-quality or signal-strength threshold: every selected component remains
    available for later perturbation-based adjudication.
    """
    limit = max(int(max_candidates), 1)
    x = np.asarray(fit.x, dtype=float)
    window = (float(np.min(x)), float(np.max(x)))
    rmse = float(fit.statistics.get("rmse", float("nan")))
    noise = (
        rmse
        if np.isfinite(rmse) and rmse > np.finfo(float).eps
        else np.finfo(float).eps
    )
    features = list(fit.parameters.get("features", ()))
    features.sort(
        key=lambda item: abs(float(item["amplitude"])),
        reverse=True,
    )

    candidates = []
    for rank, feature in enumerate(features[:limit]):
        center = float(feature["center_mhz"])
        fwhm = 2.0 * abs(float(feature["hwhm_mhz"]))
        contrast = float(feature["amplitude"])
        candidates.append(
            Candidate(
                candidate_id=_candidate_id(fit.source_csv, center, rank),
                center_mhz=center,
                fwhm_mhz=fwhm,
                contrast=contrast,
                center_uncertainty_mhz=float(
                    feature["center_uncertainty_mhz"]
                ),
                local_snr=abs(contrast) / noise,
                rank=rank,
                source_csv=Path(fit.source_csv),
                window_mhz=window,
                statistics={
                    "r_squared": float(
                        fit.statistics.get("r_squared", float("nan"))
                    ),
                    "rmse": rmse,
                    "edge_distance_mhz": float(
                        min(center - window[0], window[1] - center)
                    ),
                    "delta_bic_two_vs_one": float(
                        fit.statistics.get(
                            "delta_bic_two_vs_one",
                            float("nan"),
                        )
                    ),
                },
            )
        )

    null_rank = len(candidates)
    candidates.append(
        Candidate(
            candidate_id=_candidate_id(
                fit.source_csv,
                float("nan"),
                null_rank,
            ),
            center_mhz=float("nan"),
            fwhm_mhz=float("nan"),
            contrast=0.0,
            center_uncertainty_mhz=float("nan"),
            local_snr=0.0,
            rank=null_rank,
            source_csv=Path(fit.source_csv),
            window_mhz=window,
            is_null=True,
            statistics={
                "detectable_contrast": 3.0 * noise,
                "rmse": rmse,
            },
        )
    )
    return tuple(candidates)
