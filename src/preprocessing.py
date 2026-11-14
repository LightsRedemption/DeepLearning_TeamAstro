"""Gaia XP preprocessing shared by training and Sagittarius inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

N_COEFFICIENTS = 55
N_FEATURES = 108
TARGET_NAMES = ("TEFF", "LOGG", "FE_H", "N_O", "AL_FE")


@dataclass(frozen=True)
class PreprocessingResult:
    features: np.ndarray
    valid_mask: np.ndarray
    rejection_reasons: tuple[str, ...]


def _as_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size != N_COEFFICIENTS:
        raise ValueError(
            f"{name} must contain {N_COEFFICIENTS} coefficients; got {array.size}"
        )
    return array


def preprocess_one(bp_coefficients, rp_coefficients) -> np.ndarray:
    """Convert one 55+55 XP coefficient pair into 108 normalized features."""
    bp = _as_vector(bp_coefficients, "BP")
    rp = _as_vector(rp_coefficients, "RP")
    if not np.isfinite(bp).all() or not np.isfinite(rp).all():
        raise ValueError("BP and RP coefficients must all be finite")
    if bp[0] == 0.0 or rp[0] == 0.0:
        raise ValueError("The first BP and RP coefficients must be nonzero")
    features = np.concatenate((bp[1:] / bp[0], rp[1:] / rp[0]))
    if features.shape != (N_FEATURES,):
        raise RuntimeError(f"Expected {N_FEATURES} features; got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Normalized features contain nonfinite values")
    return features


def preprocess_many(bp_rows: Iterable, rp_rows: Iterable) -> PreprocessingResult:
    """Preprocess rows while returning a mask that preserves table alignment."""
    bp_rows = list(bp_rows)
    rp_rows = list(rp_rows)
    if len(bp_rows) != len(rp_rows):
        raise ValueError("BP and RP row counts differ")

    valid_mask = np.zeros(len(bp_rows), dtype=bool)
    accepted: list[np.ndarray] = []
    reasons: list[str] = []
    for index, (bp, rp) in enumerate(zip(bp_rows, rp_rows)):
        try:
            accepted.append(preprocess_one(bp, rp))
            valid_mask[index] = True
            reasons.append("")
        except (TypeError, ValueError, RuntimeError) as error:
            reasons.append(str(error))

    features = (
        np.vstack(accepted).astype(np.float32, copy=False)
        if accepted
        else np.empty((0, N_FEATURES), dtype=np.float32)
    )
    return PreprocessingResult(features, valid_mask, tuple(reasons))
