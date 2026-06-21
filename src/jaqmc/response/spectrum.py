# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Projected resolvent and spectral-response utilities.

This module implements the finite response-subspace algebra used by the
BF-NKSR construction:

    G_ab(z) = (p^b)^dagger (z S - K)^-1 p^a

with overlap whitening, pole/weight extraction, Lorentzian broadening, and
peak picking on a sampled spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Peak:
    """A local maximum of a sampled spectrum."""

    energy: float
    intensity: float
    index: int


@dataclass(frozen=True)
class ProjectedSpectrum:
    """Poles and source-channel weights of a projected resolvent.

    Attributes:
        excitation_energies: Ritz excitation energies, the poles Omega_nu.
        weights: Spectral weights. Shape is ``(n_roots,)`` for one source and
            ``(n_roots, n_sources, n_sources)`` for multiple sources.
        retained_overlap_eigenvalues: Eigenvalues of S retained by whitening.
        whitened_hamiltonian: Hermitian K matrix in the retained orthonormal
            basis.
        whitened_sources: Source vectors in the retained orthonormal basis.
    """

    excitation_energies: NDArray[np.float64]
    weights: NDArray[np.complex128]
    retained_overlap_eigenvalues: NDArray[np.float64]
    whitened_hamiltonian: NDArray[np.complex128]
    whitened_sources: NDArray[np.complex128]


def _as_square_matrix(name: str, value: ArrayLike) -> NDArray[np.complex128]:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"{name} must be a square matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    return matrix


def _as_source_matrix(
    source: ArrayLike, basis_size: int
) -> tuple[NDArray[np.complex128], bool]:
    source_arr = np.asarray(source, dtype=np.complex128)
    single_source = source_arr.ndim == 1
    if single_source:
        source_arr = source_arr[:, np.newaxis]
    if source_arr.ndim != 2 or source_arr.shape[0] != basis_size:
        msg = (
            "source must have shape (n_basis,) or (n_basis, n_sources), "
            f"got {source_arr.shape} for n_basis={basis_size}"
        )
        raise ValueError(msg)
    return source_arr, single_source


def hermitize(matrix: ArrayLike) -> NDArray[np.complex128]:
    """Return the Hermitian part of a square matrix."""
    matrix_arr = _as_square_matrix("matrix", matrix)
    return (matrix_arr + matrix_arr.conj().T) / 2


def whiten_projected_matrices(
    overlap: ArrayLike,
    hamiltonian: ArrayLike,
    source: ArrayLike,
    *,
    overlap_cutoff: float = 1e-10,
    relative_cutoff: bool = True,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64]]:
    """Whiten S, K, and p for the projected resolvent.

    The retained basis is obtained from ``S = U Lambda U^dagger`` and
    ``Lambda^-1/2 U^dagger``. Directions with small overlap are discarded.

    Args:
        overlap: Response-basis overlap matrix S.
        hamiltonian: Shifted Hamiltonian matrix K = <chi_i|H-E0|chi_j>.
        source: Source matrix p with one or more probe channels.
        overlap_cutoff: Minimum retained overlap eigenvalue. If
            ``relative_cutoff`` is true, this is multiplied by the largest
            overlap eigenvalue.
        relative_cutoff: Whether ``overlap_cutoff`` is relative to the largest
            overlap eigenvalue.

    Returns:
        ``(K_tilde, p_tilde, retained_lambdas)`` in an orthonormal basis.

    Raises:
        ValueError: If matrix shapes are inconsistent or no overlap direction is
            retained.
    """
    overlap_arr = hermitize(overlap)
    hamiltonian_arr = hermitize(hamiltonian)
    if hamiltonian_arr.shape != overlap_arr.shape:
        msg = (
            "hamiltonian and overlap must have the same shape, got "
            f"{hamiltonian_arr.shape} and {overlap_arr.shape}"
        )
        raise ValueError(msg)
    source_arr, _ = _as_source_matrix(source, overlap_arr.shape[0])

    diag = np.real(np.diag(overlap_arr))
    positive_diag = np.where(np.isfinite(diag) & (diag > 0), diag, 1.0)
    scale = 1.0 / np.sqrt(np.maximum(positive_diag, 1e-300))
    overlap_arr = (scale[:, np.newaxis] * overlap_arr) * scale[np.newaxis, :]
    hamiltonian_arr = (scale[:, np.newaxis] * hamiltonian_arr) * scale[np.newaxis, :]
    source_arr = scale[:, np.newaxis] * source_arr

    overlap_evals, overlap_evecs = np.linalg.eigh(overlap_arr)
    if overlap_evals.size == 0 or float(overlap_evals[-1]) <= 0:
        msg = "overlap must have at least one positive eigenvalue"
        raise ValueError(msg)

    cutoff = float(overlap_cutoff)
    if relative_cutoff:
        cutoff *= float(overlap_evals[-1])
    keep = overlap_evals > cutoff
    if not np.any(keep):
        msg = f"all overlap directions were discarded by cutoff {cutoff:g}"
        raise ValueError(msg)

    lambdas = np.asarray(overlap_evals[keep], dtype=np.float64)
    retained_evecs = overlap_evecs[:, keep]
    whitening = retained_evecs / np.sqrt(lambdas)
    k_tilde = whitening.conj().T @ hamiltonian_arr @ whitening
    p_tilde = whitening.conj().T @ source_arr
    return hermitize(k_tilde), p_tilde, lambdas


def projected_spectrum(
    overlap: ArrayLike,
    hamiltonian: ArrayLike,
    source: ArrayLike,
    *,
    overlap_cutoff: float = 1e-10,
    relative_cutoff: bool = True,
) -> ProjectedSpectrum:
    """Compute poles and weights of the whitened projected resolvent.

    Returns:
        Poles, weights, and whitened matrices for the projected response.
    """
    source_arr, single_source = _as_source_matrix(source, np.asarray(overlap).shape[0])
    k_tilde, p_tilde, lambdas = whiten_projected_matrices(
        overlap,
        hamiltonian,
        source_arr,
        overlap_cutoff=overlap_cutoff,
        relative_cutoff=relative_cutoff,
    )
    energies, vectors = np.linalg.eigh(k_tilde)
    amplitudes = vectors.conj().T @ p_tilde
    weights = amplitudes[:, :, np.newaxis].conj() * amplitudes[:, np.newaxis, :]
    if single_source:
        weights = weights[:, 0, 0]
    return ProjectedSpectrum(
        excitation_energies=np.asarray(energies, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.complex128),
        retained_overlap_eigenvalues=lambdas,
        whitened_hamiltonian=k_tilde,
        whitened_sources=p_tilde,
    )


def resolvent(
    omega: ArrayLike,
    eta: float,
    overlap: ArrayLike,
    hamiltonian: ArrayLike,
    source: ArrayLike,
    *,
    overlap_cutoff: float = 1e-10,
    relative_cutoff: bool = True,
) -> NDArray[np.complex128]:
    """Evaluate ``p^dagger (z S - K)^-1 p`` for one or more frequencies.

    Returns:
        Complex resolvent values. The result is scalar-valued for one source
        and matrix-valued for multiple sources.

    Raises:
        ValueError: If ``eta`` is not positive or matrix inputs are invalid.
    """
    if eta <= 0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)

    spectrum = projected_spectrum(
        overlap,
        hamiltonian,
        source,
        overlap_cutoff=overlap_cutoff,
        relative_cutoff=relative_cutoff,
    )
    omega_arr = np.asarray(omega, dtype=np.float64)
    z = omega_arr[..., np.newaxis] + 1j * float(eta)
    poles = spectrum.excitation_energies
    if spectrum.weights.ndim == 1:
        return np.sum(spectrum.weights / (z - poles), axis=-1)
    return np.einsum("...n,nab->...ab", 1 / (z - poles), spectrum.weights)


def lorentzian_spectrum(
    omega: ArrayLike,
    excitation_energies: ArrayLike,
    weights: ArrayLike,
    eta: float,
) -> NDArray[np.float64]:
    """Evaluate the Lorentzian-broadened spectral measure.

    For multiple source channels, ``weights`` can have shape
    ``(n_roots, n_sources, n_sources)``; the returned spectrum then has shape
    ``omega.shape + (n_sources, n_sources)``.

    Returns:
        Real Lorentzian-broadened spectral values.

    Raises:
        ValueError: If ``eta`` is not positive.
    """
    if eta <= 0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)
    omega_arr = np.asarray(omega, dtype=np.float64)
    poles = np.asarray(excitation_energies, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.complex128)
    kernel = (eta / np.pi) / ((omega_arr[..., np.newaxis] - poles) ** 2 + eta**2)
    if weights_arr.ndim == 1:
        return np.real(np.sum(kernel * weights_arr, axis=-1))
    return np.real(np.einsum("...n,nab->...ab", kernel, weights_arr))


def find_spectrum_peaks(
    omega: ArrayLike,
    intensity: ArrayLike,
    *,
    min_height: float | None = None,
    min_height_fraction: float = 0.0,
    max_peaks: int | None = None,
) -> list[Peak]:
    """Find local maxima in a one-dimensional sampled spectrum.

    A quadratic interpolation through each maximum and its two neighbors is
    used to reduce grid bias in the reported peak location.

    Returns:
        Peaks sorted by increasing energy. If ``max_peaks`` is provided, the
        strongest peaks are selected first and then returned in energy order.

    Raises:
        ValueError: If the arrays are not one-dimensional, shape-compatible, or
            strictly increasing in energy.
    """
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
