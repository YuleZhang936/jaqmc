# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Small spectral post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Peak:
    """A local maximum of a sampled spectrum."""

    energy: float
    intensity: float
    index: int


def find_spectrum_peaks(
    omega: ArrayLike,
    intensity: ArrayLike,
    *,
    min_height: float | None = None,
    min_height_fraction: float = 0.0,
    max_peaks: int | None = None,
) -> list[Peak]:
    """Find local maxima in a one-dimensional sampled spectrum."""
    omega_arr = np.asarray(omega, dtype=np.float64)
    intensity_arr = np.asarray(intensity, dtype=np.float64)
    if omega_arr.ndim != 1 or intensity_arr.ndim != 1:
        msg = "omega and intensity must be one-dimensional arrays"
        raise ValueError(msg)
    if omega_arr.shape != intensity_arr.shape:
        msg = (
            "omega and intensity shapes differ: "
            f"{omega_arr.shape}, {intensity_arr.shape}"
        )
        raise ValueError(msg)
    if omega_arr.size < 3:
        return []
    if np.any(np.diff(omega_arr) <= 0):
        msg = "omega grid must be strictly increasing"
        raise ValueError(msg)

    threshold = -np.inf if min_height is None else float(min_height)
    if min_height_fraction > 0:
        threshold = max(
            threshold, float(min_height_fraction) * float(np.nanmax(intensity_arr))
        )

    peaks: list[Peak] = []
    for idx in range(1, omega_arr.size - 1):
        left = float(intensity_arr[idx - 1])
        center = float(intensity_arr[idx])
        right = float(intensity_arr[idx + 1])
        if center < threshold or center < left or center <= right:
            continue
        denominator = left - 2 * center + right
        if denominator == 0:
            energy = float(omega_arr[idx])
            value = center
        else:
            step = float((omega_arr[idx + 1] - omega_arr[idx - 1]) / 2)
            offset = 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
            energy = float(omega_arr[idx] + offset * step)
            value = float(center - 0.25 * (left - right) * offset)
        peaks.append(Peak(energy=energy, intensity=value, index=idx))

    peaks.sort(key=lambda peak: peak.intensity, reverse=True)
    if max_peaks is not None:
        peaks = peaks[:max_peaks]
    return sorted(peaks, key=lambda peak: peak.energy)
