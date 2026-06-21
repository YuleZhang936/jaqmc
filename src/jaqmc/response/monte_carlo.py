# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Monte Carlo weak-form matrix estimators for projected response theory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class WeakMatrixEstimate:
    """Monte Carlo estimates of projected response matrices.

    Attributes:
        overlap: ``S_ij = <chi_i|chi_j>``.
        hamiltonian: ``K_ij = <chi_i|H-E0|chi_j>`` evaluated in weak form.
        source: ``p_i^a = <chi_i|Phi_a>``.
    """

    overlap: NDArray[np.complex128]
    hamiltonian: NDArray[np.complex128]
    source: NDArray[np.complex128]


def systematic_resample(
    probabilities: ArrayLike,
    n_samples: int,
    *,
    seed: int,
) -> NDArray[np.int64]:
    """Draw low-variance resampling indices from a discrete distribution.

    Args:
        probabilities: Nonnegative unnormalized probabilities for the
            candidate points.
        n_samples: Number of indices to draw.
        seed: Random seed for the single systematic offset.

    Returns:
        Integer indices into ``probabilities`` with shape ``(n_samples,)``.

    Raises:
        ValueError: If probabilities are invalid or ``n_samples`` is not
            positive.
    """
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 1:
        msg = f"probabilities must be one-dimensional, got {weights.shape}"
        raise ValueError(msg)
    if int(n_samples) < 1:
        msg = "n_samples must be positive"
        raise ValueError(msg)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        msg = "probabilities must be finite and nonnegative"
        raise ValueError(msg)
    total = float(np.sum(weights))
    if not (np.isfinite(total) and total > 0.0):
        msg = "at least one probability must be positive"
        raise ValueError(msg)
    normalized = weights / total
    cdf = np.cumsum(normalized)
    cdf[-1] = 1.0
    rng = np.random.default_rng(int(seed))
    positions = (rng.random() + np.arange(int(n_samples))) / int(n_samples)
    return np.searchsorted(cdf, positions, side="right").astype(np.int64)


def estimate_weak_matrices(
    basis_values: ArrayLike,
    basis_gradients: ArrayLike,
    source_values: ArrayLike,
    potential_minus_ground_energy: ArrayLike,
    sampling_density: ArrayLike,
) -> WeakMatrixEstimate:
    """Estimate ``S``, ``K``, and ``p`` from samples distributed as ``pi``.

    The estimator assumes samples are drawn from a normalized density
    ``pi(R)`` with respect to coordinate volume. For any integral
    ``int F(R) dR``, it uses ``mean[F(R_n) / pi(R_n)]``.

    Args:
        basis_values: Basis values ``chi_i(R_n)`` with shape
            ``(n_samples, n_basis)``.
        basis_gradients: Basis gradients with shape
            ``(n_samples, n_basis, n_coordinates)``.
        source_values: Source values ``Phi_a(R_n)`` with shape ``(n_samples,)``
            or ``(n_samples, n_sources)``.
        potential_minus_ground_energy: ``V(R_n) - E0`` values.
        sampling_density: Normalized sampling density ``pi(R_n)``.

    Returns:
        Estimated overlap, weak-form Hamiltonian, and source matrices.

    Raises:
        ValueError: If input shapes are inconsistent or a sampling density is
            not positive.
    """
    values = np.asarray(basis_values, dtype=np.complex128)
    gradients = np.asarray(basis_gradients, dtype=np.complex128)
    sources = np.asarray(source_values, dtype=np.complex128)
    potential_shift = np.asarray(potential_minus_ground_energy, dtype=np.float64)
    density = np.asarray(sampling_density, dtype=np.float64)

    if values.ndim != 2:
        msg = f"basis_values must have shape (n_samples, n_basis), got {values.shape}"
        raise ValueError(msg)
    if gradients.ndim != 3 or gradients.shape[:2] != values.shape:
        msg = (
            "basis_gradients must have shape "
            f"{(*values.shape, 'n_coordinates')}, got {gradients.shape}"
        )
        raise ValueError(msg)
    if sources.ndim == 1:
        sources = sources[:, np.newaxis]
    if sources.ndim != 2 or sources.shape[0] != values.shape[0]:
        msg = (
            "source_values must have shape (n_samples,) or "
            f"(n_samples, n_sources), got {sources.shape}"
        )
        raise ValueError(msg)
    if potential_shift.shape != (values.shape[0],):
        msg = (
            "potential_minus_ground_energy must have shape "
            f"({values.shape[0]},), got {potential_shift.shape}"
        )
        raise ValueError(msg)
    if density.shape != (values.shape[0],):
        msg = (
            f"sampling_density must have shape ({values.shape[0]},), "
            f"got {density.shape}"
        )
        raise ValueError(msg)
    if np.any(density <= 0):
        msg = "sampling_density must be positive at every sample"
        raise ValueError(msg)

    weights = 1 / density / values.shape[0]
    overlap = np.einsum("n,ni,nj->ij", weights, values.conj(), values)
    source = np.einsum("n,ni,na->ia", weights, values.conj(), sources)
    kinetic_density = 0.5 * np.einsum("nid,njd->nij", gradients.conj(), gradients)
    potential_density = potential_shift[:, np.newaxis, np.newaxis] * (
        values.conj()[:, :, np.newaxis] * values[:, np.newaxis, :]
    )
    hamiltonian = np.einsum("n,nij->ij", weights, kinetic_density + potential_density)

    return WeakMatrixEstimate(
        overlap=np.asarray(overlap, dtype=np.complex128),
        hamiltonian=np.asarray(hamiltonian, dtype=np.complex128),
        source=np.asarray(source, dtype=np.complex128),
    )
