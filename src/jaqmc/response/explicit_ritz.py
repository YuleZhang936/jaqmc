# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Source-adapted explicitly correlated Ritz carrier response basis.

This module implements the atom-level carrier space used to test the
production-certified linear BF-NKSR path:

    C_mu(R) = Q0[Phi_mu^{S Gamma}(R) G_q(R)].

The implementation is intentionally linear: no CAS teacher, no Krylov seed,
and no neural dressing are used in the primary carrier construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.response.ferminet_bfnksr import (
    FermiNetGround,
    _ground_density_np,
    _optimize_matrix_leverage_mixture_weights,
    ground_values_and_gradients,
    one_electron_pz_envelope_mixture_density,
    potential_shift,
    retained_weak_matrix_blocks_from_precomputed_values,
    sample_one_electron_pz_envelope_mixture_sobol_antithetic,
    sample_source_envelope_pz_sobol_antithetic,
    source_envelope_pz_density,
    source_values_and_gradients,
)
from jaqmc.response.monte_carlo import systematic_resample
from jaqmc.response.spectrum import ProjectedSpectrum, projected_spectrum


@dataclass(frozen=True)
class ExplicitCarrierSpec:
    """One explicit response carrier.

    Attributes:
        channel: 0 for spectator-relaxed ``s p + p s`` carriers and 1 for
            coupled ``(p d + d p)_{L=1,M=0}`` carriers.
        p_decay: Slater/Sturmian p-orbital decay zeta.
        laguerre_order: Associated Laguerre order n in L_n^4(2 zeta r).
        s_decay: Core s-orbital decay for two-electron singlet-P carriers.
        s_laguerre_order: Associated Laguerre order n in L_n^2(2 zeta r).
        d_decay: Slater/Sturmian d-orbital decay zeta for ``p d`` carriers.
        d_laguerre_order: Associated Laguerre order n in L_n^6(2 zeta r).
        geminal_kind: 0 for G=1, 1 for r12 exp(-gamma r12),
            2 for (1-exp(-gamma r12))/gamma, 3 for exp(-gamma r12),
            and 4 for r12.
        geminal_gamma: Positive geminal decay for exponential geminals.
    """

    p_decay: float
    laguerre_order: int
    channel: int = 0
    s_decay: float = 0.0
    s_laguerre_order: int = 0
    d_decay: float = 0.0
    d_laguerre_order: int = 0
    geminal_kind: int = 0
    geminal_gamma: float = 0.0


@dataclass(frozen=True)
class ExplicitRitzResult:
    """Matrices and spectrum for an explicit carrier Ritz calculation."""

    overlap: np.ndarray
    hamiltonian: np.ndarray
    source: np.ndarray
    spectrum: ProjectedSpectrum
    samples: np.ndarray
    density: np.ndarray
    block_overlaps: np.ndarray
    block_hamiltonians: np.ndarray
    block_sources: np.ndarray
    block_counts: np.ndarray
    carrier_labels: tuple[str, ...]
    whitening_eigenvalues: np.ndarray | None = None
    whitening_retained: np.ndarray | None = None
    whitening_transform: np.ndarray | None = None
    ritz_carrier_coefficients: np.ndarray | None = None
    raw_overlap: np.ndarray | None = None
    raw_hamiltonian: np.ndarray | None = None
    raw_source: np.ndarray | None = None
    raw_block_overlaps: np.ndarray | None = None
    raw_block_hamiltonians: np.ndarray | None = None
    raw_block_sources: np.ndarray | None = None
    sampling_stats: dict[str, Any] | None = None


def hydrogen_p_carrier_specs(
    decays: Sequence[float],
    *,
    laguerre_orders: Sequence[int] = (0,),
) -> tuple[ExplicitCarrierSpec, ...]:
    """Return H ``P_z`` carriers ``z L_n^4(2 zeta r) exp(-zeta r)``."""
    return tuple(
        ExplicitCarrierSpec(float(decay), int(order))
        for decay in decays
        for order in laguerre_orders
    )


def helium_singlet_p_carrier_specs(
    p_decays: Sequence[float],
    *,
    s_decays: Sequence[float] = (1.6875,),
    s_laguerre_orders: Sequence[int] = (0,),
    d_decays: Sequence[float] = (),
    d_laguerre_orders: Sequence[int] = (0,),
    laguerre_orders: Sequence[int] = (0,),
    geminal_gammas: Sequence[float] = (),
    include_f12: bool = True,
    include_exp_geminals: bool = False,
    include_r12_geminal: bool = False,
    include_pd: bool = False,
) -> tuple[ExplicitCarrierSpec, ...]:
    """Return He singlet ``P_z`` carriers with optional F12/Hylleraas factors."""
    specs: list[ExplicitCarrierSpec] = []
    geminals = _explicit_geminal_specs(
        geminal_gammas,
        include_f12=bool(include_f12),
        include_exp_geminals=bool(include_exp_geminals),
        include_r12_geminal=bool(include_r12_geminal),
    )
    _append_sp_carrier_specs(
        specs,
        p_decays=p_decays,
        s_decays=s_decays,
        s_laguerre_orders=s_laguerre_orders,
        laguerre_orders=laguerre_orders,
        geminals=geminals,
    )
    if bool(include_pd):
        _append_pd_carrier_specs(
            specs,
            p_decays=p_decays,
            d_decays=d_decays,
            d_laguerre_orders=d_laguerre_orders,
            laguerre_orders=laguerre_orders,
            geminals=geminals,
        )
    return tuple(specs)


def _append_sp_carrier_specs(
    specs: list[ExplicitCarrierSpec],
    *,
    p_decays: Sequence[float],
    s_decays: Sequence[float],
    s_laguerre_orders: Sequence[int],
    laguerre_orders: Sequence[int],
    geminals: Sequence[tuple[int, float]],
) -> None:
    for s_decay, s_order, p_decay, order, geminal in product(
        s_decays,
        s_laguerre_orders,
        p_decays,
        laguerre_orders,
        geminals,
    ):
        geminal_kind, gamma = geminal
        specs.append(
            ExplicitCarrierSpec(
                p_decay=float(p_decay),
                laguerre_order=int(order),
                channel=0,
                s_decay=float(s_decay),
                s_laguerre_order=int(s_order),
                geminal_kind=int(geminal_kind),
                geminal_gamma=float(gamma),
            )
        )


def _append_pd_carrier_specs(
    specs: list[ExplicitCarrierSpec],
    *,
    p_decays: Sequence[float],
    d_decays: Sequence[float],
    d_laguerre_orders: Sequence[int],
    laguerre_orders: Sequence[int],
    geminals: Sequence[tuple[int, float]],
) -> None:
    for d_decay, d_order, p_decay, order, geminal in product(
        d_decays,
        d_laguerre_orders,
        p_decays,
        laguerre_orders,
        geminals,
    ):
        geminal_kind, gamma = geminal
        specs.append(
            ExplicitCarrierSpec(
                p_decay=float(p_decay),
                laguerre_order=int(order),
                channel=1,
                d_decay=float(d_decay),
                d_laguerre_order=int(d_order),
                geminal_kind=int(geminal_kind),
                geminal_gamma=float(gamma),
            )
        )


def _explicit_geminal_specs(
    geminal_gammas: Sequence[float],
    *,
    include_f12: bool,
    include_exp_geminals: bool,
    include_r12_geminal: bool,
) -> list[tuple[int, float]]:
    geminals: list[tuple[int, float]] = [(0, 0.0)]
    if bool(include_r12_geminal):
        geminals.append((4, 0.0))
    for raw_gamma in geminal_gammas:
        gamma = float(raw_gamma)
        if gamma <= 0:
            continue
        if bool(include_exp_geminals):
            geminals.append((3, gamma))
        geminals.append((1, gamma))
        if bool(include_f12):
            geminals.append((2, gamma))
    return geminals


def carrier_labels(
    specs: Sequence[ExplicitCarrierSpec],
    *,
    electron_count: int,
) -> tuple[str, ...]:
    """Return human-readable labels for explicit carrier columns."""
    labels = []
    for spec in specs:
        if electron_count == 1:
            labels.append(f"pz(zeta={spec.p_decay:g},L={spec.laguerre_order})")
            continue
        geminal = _geminal_label(spec)
        if int(spec.channel) == 0:
            labels.append(
                "singletPsp("
                f"s={spec.s_decay:g},Ls={spec.s_laguerre_order},"
                f"p={spec.p_decay:g},Lp={spec.laguerre_order},{geminal})"
            )
        else:
            labels.append(
                "singletPpd("
                f"p={spec.p_decay:g},Lp={spec.laguerre_order},"
                f"d={spec.d_decay:g},Ld={spec.d_laguerre_order},{geminal})"
            )
    return tuple(labels)


def _geminal_label(spec: ExplicitCarrierSpec) -> str:
    if spec.geminal_kind == 0:
        return "G=1"
    if spec.geminal_kind == 1:
        return f"G=r12exp(-{spec.geminal_gamma:g}r12)"
    if spec.geminal_kind == 2:
        return f"G=F12({spec.geminal_gamma:g})"
    if spec.geminal_kind == 3:
        return f"G=exp(-{spec.geminal_gamma:g}r12)"
    return "G=r12"


def _associated_laguerre(order: int, alpha: float, x: jax.Array) -> jax.Array:
    if int(order) == 0:
        return jnp.ones_like(x)
    if int(order) == 1:
        return 1.0 + float(alpha) - x
    lm2 = jnp.ones_like(x)
    lm1 = 1.0 + float(alpha) - x
    for n in range(2, int(order) + 1):
        nf = float(n)
        value = (
            (2 * nf - 1 + float(alpha) - x) * lm1 - (nf - 1 + float(alpha)) * lm2
        ) / nf
        lm2, lm1 = lm1, value
    return lm1


def _centered_radius(
    point: jax.Array,
    center: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    shifted = point - center
    radius = jnp.sqrt(jnp.sum(shifted**2) + 1e-24)
    return shifted, radius


def _p_sturmian_z(
    point: jax.Array,
    center: jax.Array,
    decay: float,
    order: int,
) -> jax.Array:
    return _p_sturmian_vector(point, center, decay, order)[2]


def _p_sturmian_vector(
    point: jax.Array,
    center: jax.Array,
    decay: float,
    order: int,
) -> jax.Array:
    shifted, radius = _centered_radius(point, center)
    x = 2.0 * float(decay) * radius
    return (
        shifted
        * _associated_laguerre(int(order), 4.0, x)
        * jnp.exp(-float(decay) * radius)
    )


def _s_sturmian(
    point: jax.Array,
    center: jax.Array,
    decay: float,
    order: int,
) -> jax.Array:
    _, radius = _centered_radius(point, center)
    x = 2.0 * float(decay) * radius
    return _associated_laguerre(int(order), 2.0, x) * jnp.exp(-float(decay) * radius)


def _d_sturmian_tensor(
    point: jax.Array,
    center: jax.Array,
    decay: float,
    order: int,
) -> jax.Array:
    shifted, radius = _centered_radius(point, center)
    x = 2.0 * float(decay) * radius
    radial = _associated_laguerre(int(order), 6.0, x) * jnp.exp(-float(decay) * radius)
    trace_part = jnp.eye(3, dtype=point.dtype) * (radius**2 / 3.0)
    return (jnp.outer(shifted, shifted) - trace_part) * radial


def _sp_singlet_pz(
    electron_a: jax.Array,
    electron_b: jax.Array,
    center: jax.Array,
    spec: ExplicitCarrierSpec,
) -> jax.Array:
    p_a = _p_sturmian_z(
        electron_a,
        center,
        decay=spec.p_decay,
        order=spec.laguerre_order,
    )
    p_b = _p_sturmian_z(
        electron_b,
        center,
        decay=spec.p_decay,
        order=spec.laguerre_order,
    )
    s_a = _s_sturmian(
        electron_a,
        center,
        decay=spec.s_decay,
        order=spec.s_laguerre_order,
    )
    s_b = _s_sturmian(
        electron_b,
        center,
        decay=spec.s_decay,
        order=spec.s_laguerre_order,
    )
    return s_a * p_b + p_a * s_b


def _pd_coupled_singlet_pz(
    electron_a: jax.Array,
    electron_b: jax.Array,
    center: jax.Array,
    spec: ExplicitCarrierSpec,
) -> jax.Array:
    p_a = _p_sturmian_vector(
        electron_a,
        center,
        decay=spec.p_decay,
        order=spec.laguerre_order,
    )
    p_b = _p_sturmian_vector(
        electron_b,
        center,
        decay=spec.p_decay,
        order=spec.laguerre_order,
    )
    d_a = _d_sturmian_tensor(
        electron_a,
        center,
        decay=spec.d_decay,
        order=spec.d_laguerre_order,
    )
    d_b = _d_sturmian_tensor(
        electron_b,
        center,
        decay=spec.d_decay,
        order=spec.d_laguerre_order,
    )
    return jnp.dot(d_b[2, :], p_a) + jnp.dot(d_a[2, :], p_b)


def _geminal(point: jax.Array, spec: ExplicitCarrierSpec) -> jax.Array:
    if int(spec.geminal_kind) == 0:
        return jnp.asarray(1.0, dtype=point.dtype)
    r12 = jnp.sqrt(jnp.sum((point[0] - point[1]) ** 2) + 1e-24)
    if int(spec.geminal_kind) == 4:
        return r12
    gamma = max(float(spec.geminal_gamma), 1e-12)
    if int(spec.geminal_kind) == 3:
        return jnp.exp(-gamma * r12)
    if int(spec.geminal_kind) == 1:
        return r12 * jnp.exp(-gamma * r12)
    return (1.0 - jnp.exp(-gamma * r12)) / gamma


def explicit_carrier_values_single(
    point: jax.Array,
    atoms: jax.Array,
    specs: Sequence[ExplicitCarrierSpec],
) -> jax.Array:
    """Evaluate all explicit carriers at one electronic configuration.

    Returns:
        One value per explicit carrier.

    Raises:
        ValueError: If the electron count is unsupported.
    """
    center = atoms[0]
    values = []
    if point.shape[0] == 1:
        electron = point[0]
        for spec in specs:
            values.append(
                _p_sturmian_z(
                    electron,
                    center,
                    decay=spec.p_decay,
                    order=spec.laguerre_order,
                )
            )
        return jnp.stack(values)
    if point.shape[0] != 2:
        msg = "explicit carrier smoke path currently supports one or two electrons"
        raise ValueError(msg)
    electron_a = point[0]
    electron_b = point[1]
    for spec in specs:
        if int(spec.channel) == 0:
            carrier = _sp_singlet_pz(electron_a, electron_b, center, spec)
        else:
            carrier = _pd_coupled_singlet_pz(electron_a, electron_b, center, spec)
        values.append(carrier * _geminal(point, spec))
    return jnp.stack(values)


def explicit_carrier_value_gradient_blocks(
    ground: FermiNetGround,
    point_blocks: Sequence[np.ndarray],
    specs: Sequence[ExplicitCarrierSpec],
    *,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Evaluate source plus explicit carrier values and gradients by block.

    Returns:
        Value and gradient blocks.  Column 0 is the physical dipole source and
        the remaining columns are explicit carriers.

    Raises:
        ValueError: If no explicit carriers are provided.
    """
    if not specs:
        msg = "at least one explicit carrier is required"
        raise ValueError(msg)
    atoms = jnp.asarray(ground.atoms)

    def carrier_values(point: jax.Array) -> jax.Array:
        return explicit_carrier_values_single(point, atoms, specs)

    carrier_grad = jax.jacfwd(carrier_values)
    carrier_values_batch = jax.jit(jax.vmap(carrier_values))
    carrier_grad_batch = jax.jit(jax.vmap(carrier_grad))
    value_blocks: list[np.ndarray] = []
    gradient_blocks: list[np.ndarray] = []
    for raw_points in point_blocks:
        points_np = np.asarray(raw_points, dtype=np.float64)
        values_pieces = []
        gradient_pieces = []
        for chunk in _make_batches(points_np.shape[0], int(batch_size)):
            chunk_points = jnp.asarray(points_np[chunk])
            source_values, source_gradients = source_values_and_gradients(
                ground,
                chunk_points,
            )
            carrier_vals = carrier_values_batch(chunk_points)
            carrier_grads = carrier_grad_batch(chunk_points)
            values_pieces.append(
                np.concatenate(
                    [
                        np.asarray(source_values)[:, None],
                        np.asarray(carrier_vals),
                    ],
                    axis=1,
                )
            )
            gradient_pieces.append(
                np.concatenate(
                    [
                        np.asarray(source_gradients)[:, None, :, :],
                        np.asarray(carrier_grads),
                    ],
                    axis=1,
                )
            )
        value_blocks.append(np.concatenate(values_pieces, axis=0))
        gradient_blocks.append(np.concatenate(gradient_pieces, axis=0))
    return value_blocks, gradient_blocks


def sample_explicit_carrier_points(
    ground: FermiNetGround,
    *,
    n_samples: int,
    p_decays: Sequence[float],
    core_decay: float,
    diffuse_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a source-adapted Sobol proposal for H/He explicit carriers.

    Returns:
        Sampled electron coordinates and the normalized proposal density.
    """
    if ground.electron_shape == (1, 3):
        decays = np.unique(np.asarray([*p_decays, diffuse_decay], dtype=np.float64))
        weights = np.ones_like(decays)
        points, density, _ = sample_one_electron_pz_envelope_mixture_sobol_antithetic(
            n_samples=n_samples,
            decays=decays,
            weights=weights,
            spherical_decays=np.asarray([core_decay], dtype=np.float64),
            spherical_weights=np.asarray([0.1], dtype=np.float64),
            electron_shape=ground.electron_shape,
            seed=seed,
        )
        return points, density
    points, density, _ = sample_source_envelope_pz_sobol_antithetic(
        n_samples=n_samples,
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
        electron_shape=ground.electron_shape,
        seed=seed,
    )
    return points, density


def explicit_carrier_auxiliary_density(
    ground: FermiNetGround,
    points: np.ndarray,
    *,
    p_decays: Sequence[float],
    core_decay: float,
    diffuse_decay: float,
) -> np.ndarray:
    """Evaluate the normalized auxiliary density used by explicit carriers.

    Returns:
        One density value per sampled electronic configuration.

    Raises:
        ValueError: If the electron count is unsupported.
    """
    points_np = np.asarray(points, dtype=np.float64)
    if ground.electron_shape == (1, 3):
        decays = np.unique(np.asarray([*p_decays, diffuse_decay], dtype=np.float64))
        weights = np.ones_like(decays)
        return one_electron_pz_envelope_mixture_density(
            points_np,
            decays=decays,
            weights=weights,
            spherical_decays=np.asarray([core_decay], dtype=np.float64),
            spherical_weights=np.asarray([0.1], dtype=np.float64),
        )
    if ground.electron_shape == (2, 3):
        return source_envelope_pz_density(
            points_np,
            core_decay=float(core_decay),
            diffuse_decay=float(diffuse_decay),
        )
    msg = (
        "explicit carrier auxiliary density currently supports one or two "
        f"electrons, got {ground.electron_shape}"
    )
    raise ValueError(msg)


def _estimate_explicit_ritz_from_points(
    ground: FermiNetGround,
    *,
    specs: Sequence[ExplicitCarrierSpec],
    points: np.ndarray,
    density: np.ndarray,
    n_blocks: int,
    batch_size: int,
    overlap_cutoff: float,
    fixed_whitening: bool = False,
    block_stable_metric: bool = False,
    block_mode_min: float = 0.0,
    sampling_stats: dict[str, Any] | None = None,
) -> ExplicitRitzResult:
    """Assemble and diagonalize the explicit-carrier Ritz problem.

    Returns:
        Matrices, sampled data, block diagnostics, and projected spectrum.

    Raises:
        ValueError: If the block count is invalid.
    """
    if int(n_blocks) < 1:
        msg = "n_blocks must be positive"
        raise ValueError(msg)
    points = np.asarray(points, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != ground.electron_shape:
        msg = (
            "explicit Ritz points must have shape "
            f"(n_samples, {ground.electron_shape}), got {points.shape}"
        )
        raise ValueError(msg)
    if density.shape != (points.shape[0],):
        msg = (
            "explicit Ritz density must have shape "
            f"({points.shape[0]},), got {density.shape}"
        )
        raise ValueError(msg)
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        msg = "explicit Ritz density must be finite and positive"
        raise ValueError(msg)
    order = np.arange(points.shape[0], dtype=np.int64)
    point_blocks = [
        points[block]
        for block in np.array_split(order, min(int(n_blocks), points.shape[0]))
        if block.size
    ]
    density_blocks = [
        density[block]
        for block in np.array_split(order, min(int(n_blocks), points.shape[0]))
        if block.size
    ]
    value_blocks, gradient_blocks = explicit_carrier_value_gradient_blocks(
        ground,
        point_blocks,
        specs,
        batch_size=int(batch_size),
    )
    (
        overlap,
        hamiltonian,
        source,
        block_overlaps,
        block_hamiltonians,
        block_sources,
        block_counts,
        *_,
    ) = retained_weak_matrix_blocks_from_precomputed_values(
        ground,
        point_blocks,
        density_blocks,
        value_blocks,
        gradient_blocks,
        batch_size=int(batch_size),
        return_block_statistics=True,
    )
    raw_overlap = np.asarray(overlap, dtype=np.float64)
    raw_hamiltonian = np.asarray(hamiltonian, dtype=np.float64)
    raw_source = np.asarray(source, dtype=np.float64)
    raw_block_overlaps = np.asarray(block_overlaps, dtype=np.float64)
    raw_block_hamiltonians = np.asarray(block_hamiltonians, dtype=np.float64)
    raw_block_sources = np.asarray(block_sources, dtype=np.float64)
    whitening_eigenvalues: np.ndarray | None = None
    whitening_retained: np.ndarray | None = None
    whitening_transform: np.ndarray | None = None
    if bool(fixed_whitening):
        (
            overlap,
            hamiltonian,
            source,
            block_overlaps,
            block_hamiltonians,
            block_sources,
            whitening_eigenvalues,
            whitening_retained,
            whitening_transform,
        ) = apply_fixed_metric_whitening(
            overlap,
            hamiltonian,
            source,
            block_overlaps,
            block_hamiltonians,
            block_sources,
            overlap_cutoff=float(overlap_cutoff),
            block_stable=bool(block_stable_metric),
            block_mode_min=float(block_mode_min),
        )
        spectrum = projected_spectrum(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=1e-12,
            relative_cutoff=False,
        )
    else:
        spectrum = projected_spectrum(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=float(overlap_cutoff),
        )
    ritz_carrier_coefficients = _ritz_coefficients_in_raw_carrier_basis(
        raw_overlap,
        raw_hamiltonian,
        raw_source,
        overlap,
        hamiltonian,
        source,
        whitening_transform,
        fixed_whitening=bool(fixed_whitening),
        overlap_cutoff=float(overlap_cutoff),
    )
    return ExplicitRitzResult(
        overlap=np.asarray(overlap, dtype=np.float64),
        hamiltonian=np.asarray(hamiltonian, dtype=np.float64),
        source=np.asarray(source, dtype=np.float64),
        spectrum=spectrum,
        samples=np.asarray(points, dtype=np.float64),
        density=np.asarray(density, dtype=np.float64),
        block_overlaps=np.asarray(block_overlaps, dtype=np.float64),
        block_hamiltonians=np.asarray(block_hamiltonians, dtype=np.float64),
        block_sources=np.asarray(block_sources, dtype=np.float64),
        block_counts=np.asarray(block_counts, dtype=np.float64),
        carrier_labels=carrier_labels(specs, electron_count=ground.electron_shape[0]),
        whitening_eigenvalues=whitening_eigenvalues,
        whitening_retained=whitening_retained,
        whitening_transform=whitening_transform,
        ritz_carrier_coefficients=ritz_carrier_coefficients,
        raw_overlap=raw_overlap,
        raw_hamiltonian=raw_hamiltonian,
        raw_source=raw_source,
        raw_block_overlaps=raw_block_overlaps,
        raw_block_hamiltonians=raw_block_hamiltonians,
        raw_block_sources=raw_block_sources,
        sampling_stats=sampling_stats,
    )


def run_explicit_ritz(
    ground: FermiNetGround,
    *,
    specs: Sequence[ExplicitCarrierSpec],
    n_samples: int,
    n_blocks: int,
    core_decay: float,
    diffuse_decay: float,
    seed: int,
    batch_size: int,
    overlap_cutoff: float,
    fixed_whitening: bool = False,
    block_stable_metric: bool = False,
    block_mode_min: float = 0.0,
    bright_influence_sampling: bool = False,
    bright_influence_pilot_samples: int = 0,
    bright_influence_candidate_factor: int = 2,
    bright_influence_max_candidates: int = 32768,
    bright_influence_min_weight: float = 0.01,
    bright_influence_max_states: int = 1,
    bright_influence_gradient_weight: float = 1.0,
    bright_influence_potential_weight: float = 1.0,
    bright_influence_source_weight: float = 1.0,
    bright_influence_winsor_quantile: float = 0.995,
    bright_influence_floor_fraction: float = 1e-8,
    bright_influence_direct_leverage_component: bool = False,
    bright_influence_pair_resampling: bool = False,
) -> ExplicitRitzResult:
    """Sample, assemble, and diagonalize the explicit-carrier Ritz problem.

    Returns:
        Matrices, samples, diagnostics, and projected spectrum.

    Raises:
        ValueError: If sampling or matrix parameters are invalid.
    """
    if int(n_blocks) < 1:
        msg = "n_blocks must be positive"
        raise ValueError(msg)
    p_decays = [spec.p_decay for spec in specs]
    if bool(bright_influence_sampling):
        pilot_samples = (
            int(n_samples)
            if int(bright_influence_pilot_samples) <= 0
            else int(bright_influence_pilot_samples)
        )
        pilot_points, pilot_density = sample_explicit_carrier_points(
            ground,
            n_samples=pilot_samples,
            p_decays=p_decays,
            core_decay=float(core_decay),
            diffuse_decay=float(diffuse_decay),
            seed=int(seed),
        )
        pilot_result = _estimate_explicit_ritz_from_points(
            ground,
            specs=specs,
            points=pilot_points,
            density=pilot_density,
            n_blocks=n_blocks,
            batch_size=int(batch_size),
            overlap_cutoff=float(overlap_cutoff),
            fixed_whitening=bool(fixed_whitening),
            block_stable_metric=bool(block_stable_metric),
            block_mode_min=float(block_mode_min),
            sampling_stats={"sampler": "explicit_ritz_pilot_source_sobol"},
        )
        points, density, sampling_stats = (
            sample_explicit_ritz_bright_influence_distribution(
                ground,
                specs,
                pilot_result,
                n_samples=int(n_samples),
                core_decay=float(core_decay),
                diffuse_decay=float(diffuse_decay),
                batch_size=int(batch_size),
                seed=int(seed) + 1009,
                candidate_factor=int(bright_influence_candidate_factor),
                max_candidate_samples=int(bright_influence_max_candidates),
                min_bright_weight=float(bright_influence_min_weight),
                max_bright_states=int(bright_influence_max_states),
                gradient_weight=float(bright_influence_gradient_weight),
                potential_weight=float(bright_influence_potential_weight),
                source_weight=float(bright_influence_source_weight),
                winsor_quantile=float(bright_influence_winsor_quantile),
                floor_fraction=float(bright_influence_floor_fraction),
                direct_leverage_component=bool(
                    bright_influence_direct_leverage_component
                ),
                pair_resampling=bool(bright_influence_pair_resampling),
            )
        )
        sampling_stats["pilot_raw_first_pole"] = (
            float(pilot_result.spectrum.excitation_energies[0])
            if pilot_result.spectrum.excitation_energies.size
            else float("nan")
        )
        sampling_stats["pilot_first_bright_pole"] = _first_bright_pole(
            pilot_result.spectrum,
            min_weight=float(bright_influence_min_weight),
        )
    else:
        points, density = sample_explicit_carrier_points(
            ground,
            n_samples=int(n_samples),
            p_decays=p_decays,
            core_decay=float(core_decay),
            diffuse_decay=float(diffuse_decay),
            seed=int(seed),
        )
        sampling_stats = {"sampler": "explicit_ritz_source_sobol"}
    return _estimate_explicit_ritz_from_points(
        ground,
        specs=specs,
        points=points,
        density=density,
        n_blocks=n_blocks,
        batch_size=int(batch_size),
        overlap_cutoff=float(overlap_cutoff),
        fixed_whitening=bool(fixed_whitening),
        block_stable_metric=bool(block_stable_metric),
        block_mode_min=float(block_mode_min),
        sampling_stats=sampling_stats,
    )


def sample_explicit_ritz_bright_influence_distribution(  # noqa: C901
    ground: FermiNetGround,
    specs: Sequence[ExplicitCarrierSpec],
    pilot_result: ExplicitRitzResult,
    *,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    batch_size: int,
    seed: int,
    candidate_factor: int = 2,
    max_candidate_samples: int = 32768,
    min_bright_weight: float = 0.01,
    max_bright_states: int = 1,
    gradient_weight: float = 1.0,
    potential_weight: float = 1.0,
    source_weight: float = 1.0,
    winsor_quantile: float = 0.995,
    floor_fraction: float = 1e-8,
    direct_leverage_component: bool = False,
    pair_resampling: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw production samples from a Ritz bright-influence proposal.

    The pilot Ritz state identifies the bright carrier combination.  A larger
    Sobol candidate pool is then scored by the pointwise weak residual
    ``k_b(R) - Omega_b s_b(R)`` and source-strength influence.  The final
    proposal is an optimized positive mixture of evaluable density components,
    so the production weak matrices remain importance-sampled estimators.

    Returns:
        Sampled points, positive proposal-density values, and sampler diagnostics.

    Raises:
        ValueError: If sampling parameters or pilot Ritz data are invalid.
    """
    if int(n_samples) < 1:
        msg = "bright-influence explicit Ritz sampling needs positive n_samples"
        raise ValueError(msg)
    if int(batch_size) < 1:
        msg = "bright-influence explicit Ritz batch_size must be positive"
        raise ValueError(msg)
    if int(candidate_factor) < 1:
        msg = "bright-influence explicit Ritz candidate_factor must be positive"
        raise ValueError(msg)
    if int(max_candidate_samples) < 0:
        msg = "bright-influence explicit Ritz max_candidate_samples must be nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(gradient_weight)
        and np.isfinite(potential_weight)
        and np.isfinite(source_weight)
        and gradient_weight >= 0.0
        and potential_weight >= 0.0
        and source_weight >= 0.0
    ):
        msg = "bright-influence explicit Ritz weights must be finite nonnegative"
        raise ValueError(msg)
    if not (np.isfinite(winsor_quantile) and 0.0 < winsor_quantile <= 1.0):
        msg = "bright-influence explicit Ritz winsor_quantile must be in (0, 1]"
        raise ValueError(msg)
    if not (np.isfinite(floor_fraction) and floor_fraction >= 0.0):
        msg = "bright-influence explicit Ritz floor_fraction must be nonnegative"
        raise ValueError(msg)
    if pilot_result.ritz_carrier_coefficients is None:
        msg = "pilot explicit Ritz result has no carrier-basis Ritz coefficients"
        raise ValueError(msg)
    if pilot_result.raw_source is None:
        msg = "pilot explicit Ritz result has no raw source vector"
        raise ValueError(msg)

    bright_indices = _bright_indices(
        pilot_result.spectrum,
        min_weight=float(min_bright_weight),
        max_states=int(max_bright_states),
    )
    bright_coefficients = np.asarray(
        pilot_result.ritz_carrier_coefficients[:, bright_indices],
        dtype=np.float64,
    )
    if bright_coefficients.ndim == 1:
        bright_coefficients = bright_coefficients[:, None]
    bright_roots = np.asarray(
        pilot_result.spectrum.excitation_energies[bright_indices],
        dtype=np.float64,
    )
    bright_source_weights = np.maximum(
        _spectrum_weights_1d(pilot_result.spectrum)[bright_indices],
        0.0,
    )
    if not np.any(bright_source_weights > 0.0):
        bright_rho = np.full(
            bright_indices.shape,
            1.0 / float(bright_indices.size),
            dtype=np.float64,
        )
    else:
        bright_rho = bright_source_weights / float(np.sum(bright_source_weights))

    candidate_samples = max(int(n_samples), int(n_samples) * int(candidate_factor))
    if int(max_candidate_samples) > 0:
        candidate_samples = max(
            int(n_samples),
            min(candidate_samples, int(max_candidate_samples)),
        )
    p_decays = [spec.p_decay for spec in specs]
    proposal_points, proposal_density = sample_explicit_carrier_points(
        ground,
        n_samples=candidate_samples,
        p_decays=p_decays,
        core_decay=float(core_decay),
        diffuse_decay=float(diffuse_decay),
        seed=int(seed) + 17,
    )
    proposal_density = np.maximum(
        np.asarray(proposal_density, dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    value_blocks, gradient_blocks = explicit_carrier_value_gradient_blocks(
        ground,
        [proposal_points],
        specs,
        batch_size=int(batch_size),
    )
    values_raw = np.asarray(value_blocks[0], dtype=np.float64)
    gradients_raw = np.asarray(gradient_blocks[0], dtype=np.float64)
    projected_values, projected_gradients, potential_shift_np = (
        _project_precomputed_values_against_ground_np(
            ground,
            proposal_points,
            proposal_density,
            values_raw,
            gradients_raw,
            batch_size=int(batch_size),
        )
    )
    source_projected = projected_values[:, 0]
    head_values = projected_values[:, 1:]
    head_gradients = projected_gradients[:, 1:]
    bright_values = head_values @ bright_coefficients
    bright_gradients = np.einsum(
        "nhei,hb->nbei",
        head_gradients,
        bright_coefficients,
        optimize=True,
    )
    bright_flat_gradients = bright_gradients.reshape(
        proposal_points.shape[0],
        bright_coefficients.shape[1],
        -1,
    )
    bright_overlap_integrand = bright_values**2
    bright_kinetic_integrand = 0.5 * np.sum(bright_flat_gradients**2, axis=2)
    bright_potential_integrand = potential_shift_np[:, None] * (
        bright_overlap_integrand
    )
    bright_hamiltonian_integrand = (
        float(gradient_weight) * bright_kinetic_integrand
        + float(potential_weight) * bright_potential_integrand
    )
    bright_energy_influence = (
        bright_hamiltonian_integrand - bright_roots[None, :] * bright_overlap_integrand
    )
    influence_second_moment = np.sum(
        bright_rho[None, :] * bright_energy_influence**2,
        axis=1,
    )
    source_amplitudes = (
        np.asarray(pilot_result.raw_source[:, 0], dtype=np.float64)
        @ bright_coefficients
    )
    source_strength_influence = (
        2.0 * source_amplitudes[None, :] * source_projected[:, None] * bright_values
    )
    influence_second_moment += float(source_weight) * np.sum(
        bright_rho[None, :] * source_strength_influence**2,
        axis=1,
    )
    leverage = np.sqrt(np.maximum(influence_second_moment, 0.0))
    finite_positive = leverage[np.isfinite(leverage) & (leverage > 0.0)]
    if finite_positive.size == 0:
        msg = "bright-influence explicit Ritz scores are all zero or nonfinite"
        raise ValueError(msg)
    winsor_limit = float(np.quantile(finite_positive, float(winsor_quantile)))
    leverage = np.where(np.isfinite(leverage), leverage, winsor_limit)
    leverage = np.minimum(leverage, winsor_limit)
    floor = float(floor_fraction) * max(float(np.mean(finite_positive)), 1e-300)
    leverage = np.maximum(leverage, max(floor, np.finfo(np.float64).tiny))

    ground_raw = _ground_density_np(
        ground,
        proposal_points,
        batch_size=int(batch_size),
    )
    source_raw = source_projected**2
    bright_raw = np.sum(bright_rho[None, :] * bright_overlap_integrand, axis=1)
    aux_density = explicit_carrier_auxiliary_density(
        ground,
        proposal_points,
        p_decays=p_decays,
        core_decay=float(core_decay),
        diffuse_decay=float(diffuse_decay),
    )
    ground_component, ground_norm = _normalized_positive_component(
        ground_raw,
        proposal_density,
        fallback_density=aux_density,
    )
    source_component, source_norm = _normalized_positive_component(
        source_raw,
        proposal_density,
        fallback_density=aux_density,
    )
    bright_component, bright_norm = _normalized_positive_component(
        bright_raw,
        proposal_density,
        fallback_density=aux_density,
    )
    leverage_component, leverage_norm = _normalized_positive_component(
        leverage,
        proposal_density,
        fallback_density=aux_density,
    )
    aux_component = np.maximum(aux_density, np.finfo(np.float64).tiny)
    component_names = ["ground", "source", "bright", "auxiliary"]
    component_columns = [
        ground_component,
        source_component,
        bright_component,
        aux_component,
    ]
    if bool(direct_leverage_component):
        component_names.insert(3, "leverage")
        component_columns.insert(3, leverage_component)
    component_densities = np.column_stack(component_columns)
    component_weights, leverage_objective = _optimize_matrix_leverage_mixture_weights(
        component_densities, leverage
    )
    mixture_density = np.maximum(
        component_densities @ component_weights,
        np.finfo(np.float64).tiny,
    )
    log_weights = np.log(mixture_density) - np.log(proposal_density)
    log_weights_shift = float(np.max(log_weights))
    relative_weights = np.exp(log_weights - log_weights_shift)
    weight_sum = float(np.sum(relative_weights))
    probabilities = relative_weights / weight_sum
    mixture_normalizer = float(
        np.exp(log_weights_shift) * weight_sum / float(candidate_samples)
    )
    mixture_normalizer = max(mixture_normalizer, np.finfo(np.float64).tiny)
    proposal_ess = float(weight_sum**2 / np.sum(relative_weights**2))
    used_pair_resampling = (
        bool(pair_resampling)
        and int(n_samples) % 2 == 0
        and int(candidate_samples) % 2 == 0
    )
    rng = np.random.default_rng(int(seed) + 53)
    if used_pair_resampling:
        half_candidates = int(candidate_samples) // 2
        pair_weights = (
            relative_weights[:half_candidates]
            + relative_weights[half_candidates : 2 * half_candidates]
        )
        pair_probabilities = pair_weights / float(np.sum(pair_weights))
        pair_indices = systematic_resample(
            pair_probabilities,
            int(n_samples) // 2,
            seed=int(seed) + 31,
        )
        pair_indices = pair_indices[rng.permutation(pair_indices.size)]
        indices = np.empty((int(n_samples),), dtype=np.int64)
        indices[0::2] = pair_indices
        indices[1::2] = pair_indices + half_candidates
        pair_weight_sum = float(np.sum(pair_weights))
        pair_proposal_ess = float(pair_weight_sum**2 / np.sum(pair_weights**2))
    else:
        indices = systematic_resample(
            probabilities,
            int(n_samples),
            seed=int(seed) + 31,
        )
        indices = indices[rng.permutation(int(n_samples))]
        pair_proposal_ess = float("nan")
    unique_fraction = float(np.unique(indices).size / int(n_samples))
    samples = np.asarray(proposal_points[indices], dtype=np.float64)
    selected_density = np.asarray(mixture_density[indices], dtype=np.float64)
    selected_density = selected_density / mixture_normalizer
    final_log_density = np.log(np.maximum(selected_density, np.finfo(np.float64).tiny))
    density_log_shift = float(np.max(final_log_density))
    density = np.exp(final_log_density - density_log_shift)
    density = np.maximum(density, np.finfo(np.float64).tiny)
    stats = {
        "sampler": "explicit_ritz_bright_influence_sobol_resampling",
        "density_log_shift": density_log_shift,
        "proposal_samples": int(candidate_samples),
        "proposal_ess": proposal_ess,
        "proposal_ess_fraction": proposal_ess / float(candidate_samples),
        "pair_proposal_ess": pair_proposal_ess,
        "pair_proposal_ess_fraction": (
            pair_proposal_ess / float(candidate_samples // 2)
            if used_pair_resampling
            else float("nan")
        ),
        "proposal_max_weight_fraction": float(np.max(probabilities)),
        "resampling_unique_fraction": unique_fraction,
        "pair_resampling": bool(used_pair_resampling),
        "leverage_normalizer": mixture_normalizer,
        "leverage_mean": float(np.mean(leverage)),
        "leverage_max": float(np.max(leverage)),
        "leverage_winsor_limit": winsor_limit,
        "leverage_floor": max(floor, np.finfo(np.float64).tiny),
        "leverage_gradient_weight": float(gradient_weight),
        "leverage_potential_weight": float(potential_weight),
        "leverage_source_weight": float(source_weight),
        "leverage_winsor_quantile": float(winsor_quantile),
        "leverage_floor_fraction": float(floor_fraction),
        "leverage_mixture_objective": leverage_objective,
        "leverage_component_weights": component_weights,
        "leverage_component_names": np.asarray(component_names),
        "leverage_bright_indices": bright_indices,
        "leverage_bright_roots": bright_roots,
        "leverage_bright_source_weights": bright_source_weights,
        "leverage_bright_rho": bright_rho,
        "log_weight_shift": log_weights_shift,
        "ground_norm": float(ground_norm),
        "source_norm": float(source_norm),
        "bright_norm": float(bright_norm),
        "leverage_norm": float(leverage_norm),
        "ground_weight": _component_weight(
            component_names,
            component_weights,
            "ground",
        ),
        "source_component_weight": _component_weight(
            component_names,
            component_weights,
            "source",
        ),
        "bright_weight": _component_weight(
            component_names,
            component_weights,
            "bright",
        ),
        "leverage_weight": _component_weight(
            component_names,
            component_weights,
            "leverage",
        ),
        "aux_weight": _component_weight(
            component_names,
            component_weights,
            "auxiliary",
        ),
        "direct_leverage_component": bool(direct_leverage_component),
        "resampling_factor": int(candidate_factor),
        "max_candidate_samples": int(max_candidate_samples),
    }
    return samples, density, stats


def _component_weight(
    names: Sequence[str],
    weights: np.ndarray,
    name: str,
) -> float:
    if name not in names:
        return 0.0
    return float(np.asarray(weights, dtype=np.float64)[list(names).index(name)])


def scalar_rayleigh_quotient(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Evaluate the fixed-state Rayleigh quotient ``c^T K c / c^T S c``.

    Returns:
        Scalar excitation energy for the fixed coefficient vector.
    """
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    overlap_np = np.asarray(overlap, dtype=np.float64)
    hamiltonian_np = np.asarray(hamiltonian, dtype=np.float64)
    denominator = float(coeff @ overlap_np @ coeff)
    numerator = float(coeff @ hamiltonian_np @ coeff)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-300:
        return float("nan")
    return numerator / denominator


def scalar_rayleigh_diagnostics(  # noqa: C901
    block_overlaps: np.ndarray,
    block_hamiltonians: np.ndarray,
    block_counts: np.ndarray,
    coefficients: np.ndarray,
    *,
    full_overlap: np.ndarray | None = None,
    full_hamiltonian: np.ndarray | None = None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, np.ndarray | float | int]:
    """Certify a fixed Ritz state with scalar block Rayleigh quotients.

    Returns:
        Full-sample, per-block, leave-one-out, and bootstrap scalar quotient
        diagnostics.  Bootstrap spread is reported as the sample standard
        deviation, matching the existing final-pole diagnostic convention.

    Raises:
        ValueError: If matrix blocks are inconsistent.
    """
    overlaps = np.asarray(block_overlaps, dtype=np.float64)
    hamiltonians = np.asarray(block_hamiltonians, dtype=np.float64)
    counts = np.asarray(block_counts, dtype=np.float64).reshape(-1)
    if overlaps.ndim != 3 or hamiltonians.shape != overlaps.shape:
        msg = "scalar Rayleigh diagnostics require matching matrix block arrays"
        raise ValueError(msg)
    if counts.shape != (overlaps.shape[0],):
        msg = "scalar Rayleigh block_counts must match the number of blocks"
        raise ValueError(msg)
    if overlaps.shape[0] < 1:
        msg = "scalar Rayleigh diagnostics require at least one block"
        raise ValueError(msg)
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coeff.shape != (overlaps.shape[1],):
        msg = "scalar Rayleigh coefficients must match matrix block size"
        raise ValueError(msg)
    full_overlap_np = (
        _weighted_block_matrix(overlaps, counts)
        if full_overlap is None
        else np.asarray(full_overlap, dtype=np.float64)
    )
    full_hamiltonian_np = (
        _weighted_block_matrix(hamiltonians, counts)
        if full_hamiltonian is None
        else np.asarray(full_hamiltonian, dtype=np.float64)
    )
    full_value = scalar_rayleigh_quotient(
        full_overlap_np,
        full_hamiltonian_np,
        coeff,
    )
    block_values = np.asarray(
        [
            scalar_rayleigh_quotient(overlap, hamiltonian, coeff)
            for overlap, hamiltonian in zip(overlaps, hamiltonians, strict=True)
        ],
        dtype=np.float64,
    )
    loo_values = []
    if overlaps.shape[0] > 1:
        for block_index in range(overlaps.shape[0]):
            keep = np.arange(overlaps.shape[0]) != block_index
            loo_overlap = _weighted_block_matrix(overlaps[keep], counts[keep])
            loo_hamiltonian = _weighted_block_matrix(hamiltonians[keep], counts[keep])
            loo_values.append(
                scalar_rayleigh_quotient(loo_overlap, loo_hamiltonian, coeff)
            )
    loo_values_np = np.asarray(loo_values, dtype=np.float64)
    loo_finite = loo_values_np[np.isfinite(loo_values_np)]
    if loo_finite.size > 1:
        loo_mean = float(np.mean(loo_finite))
        loo_jackknife_se = float(
            np.sqrt(
                (loo_finite.size - 1)
                / loo_finite.size
                * np.sum((loo_finite - loo_mean) ** 2)
            )
        )
    elif loo_finite.size == 1:
        loo_mean = float(loo_finite[0])
        loo_jackknife_se = float("nan")
    else:
        loo_mean = float("nan")
        loo_jackknife_se = float("nan")

    bootstrap_values = []
    if overlaps.shape[0] > 1 and int(bootstrap_replicates) > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        for _ in range(int(bootstrap_replicates)):
            indices = rng.integers(0, overlaps.shape[0], size=overlaps.shape[0])
            boot_overlap = _weighted_block_matrix(overlaps[indices], counts[indices])
            boot_hamiltonian = _weighted_block_matrix(
                hamiltonians[indices],
                counts[indices],
            )
            bootstrap_values.append(
                scalar_rayleigh_quotient(boot_overlap, boot_hamiltonian, coeff)
            )
    bootstrap_np = np.asarray(bootstrap_values, dtype=np.float64)
    bootstrap_finite = bootstrap_np[np.isfinite(bootstrap_np)]
    if bootstrap_finite.size > 1:
        bootstrap_mean = float(np.mean(bootstrap_finite))
        bootstrap_std = float(np.std(bootstrap_finite, ddof=1))
        bootstrap_min = float(np.min(bootstrap_finite))
        bootstrap_max = float(np.max(bootstrap_finite))
    elif bootstrap_finite.size == 1:
        bootstrap_mean = float(bootstrap_finite[0])
        bootstrap_std = float("nan")
        bootstrap_min = float(bootstrap_finite[0])
        bootstrap_max = float(bootstrap_finite[0])
    else:
        bootstrap_mean = float("nan")
        bootstrap_std = float("nan")
        bootstrap_min = float("nan")
        bootstrap_max = float("nan")
    return {
        "full": float(full_value),
        "block_values": block_values,
        "loo_values": loo_values_np,
        "loo_mean": loo_mean,
        "loo_jackknife_se": loo_jackknife_se,
        "bootstrap_values": bootstrap_np,
        "bootstrap_count": int(bootstrap_finite.size),
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,
        "bootstrap_min": bootstrap_min,
        "bootstrap_max": bootstrap_max,
    }


def _weighted_block_matrix(blocks: np.ndarray, counts: np.ndarray) -> np.ndarray:
    block_arr = np.asarray(blocks, dtype=np.float64)
    count_arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(np.sum(count_arr))
    if not np.isfinite(total) or total <= 0.0:
        msg = "block counts must have positive finite sum"
        raise ValueError(msg)
    weights = count_arr / total
    return np.tensordot(weights, block_arr, axes=(0, 0))


def _normalized_positive_component(
    raw_values: np.ndarray,
    proposal_density: np.ndarray,
    *,
    fallback_density: np.ndarray,
) -> tuple[np.ndarray, float]:
    tiny = np.finfo(np.float64).tiny
    raw = np.asarray(raw_values, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    proposal = np.maximum(np.asarray(proposal_density, dtype=np.float64), tiny)
    norm = float(np.mean(raw / proposal))
    if not (np.isfinite(norm) and norm > 0.0):
        fallback = np.maximum(np.asarray(fallback_density, dtype=np.float64), tiny)
        return fallback, 1.0
    return np.maximum(raw / norm, tiny), norm


def _project_precomputed_values_against_ground_np(
    ground: FermiNetGround,
    points: np.ndarray,
    density: np.ndarray,
    values: np.ndarray,
    gradients: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float64)
    density_np = np.maximum(
        np.asarray(density, dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    ground_value_pieces = []
    ground_gradient_pieces = []
    potential_pieces = []
    for chunk in _make_batches(points_np.shape[0], int(batch_size)):
        chunk_points = jnp.asarray(points_np[chunk])
        ground_values, ground_gradients = ground_values_and_gradients(
            ground,
            chunk_points,
        )
        ground_value_pieces.append(np.asarray(ground_values, dtype=np.float64))
        ground_gradient_pieces.append(np.asarray(ground_gradients, dtype=np.float64))
        potential_pieces.append(
            np.asarray(potential_shift(ground, chunk_points), dtype=np.float64)
        )
    ground_values_np = np.concatenate(ground_value_pieces, axis=0)
    ground_gradients_np = np.concatenate(ground_gradient_pieces, axis=0)
    potential_shift_np = np.concatenate(potential_pieces, axis=0)
    weights = 1.0 / density_np / float(points_np.shape[0])
    denominator = float(
        np.einsum("n,n,n->", weights, ground_values_np, ground_values_np)
    )
    denominator = max(denominator, np.finfo(np.float64).tiny)
    coeff = np.einsum("n,n,ni->i", weights, ground_values_np, values) / denominator
    projected_values = values - ground_values_np[:, None] * coeff[None, :]
    projected_gradients = (
        gradients - ground_gradients_np[:, None, :, :] * coeff[None, :, None, None]
    )
    return projected_values, projected_gradients, potential_shift_np


def _spectrum_weights_1d(spectrum: ProjectedSpectrum) -> np.ndarray:
    weights = np.asarray(spectrum.weights, dtype=np.complex128)
    if weights.ndim == 1:
        return np.maximum(np.asarray(weights.real, dtype=np.float64), 0.0)
    return np.maximum(np.asarray(weights[:, 0, 0].real, dtype=np.float64), 0.0)


def _bright_indices(
    spectrum: ProjectedSpectrum,
    *,
    min_weight: float,
    max_states: int,
) -> np.ndarray:
    roots = np.asarray(spectrum.excitation_energies, dtype=np.float64)
    weights = _spectrum_weights_1d(spectrum)
    max_weight = max(float(np.max(weights)) if weights.size else 0.0, 1e-300)
    keep = (roots > 0.0) & (weights / max_weight >= float(min_weight))
    indices = np.flatnonzero(keep).astype(np.int64)
    if indices.size == 0:
        positive = np.flatnonzero(roots > 0.0).astype(np.int64)
        if positive.size:
            indices = positive[:1]
        elif roots.size:
            indices = np.asarray([int(np.argmax(weights))], dtype=np.int64)
    if int(max_states) > 0 and indices.size > int(max_states):
        indices = indices[: int(max_states)]
    if indices.size == 0:
        msg = "explicit Ritz spectrum has no bright state for influence sampling"
        raise ValueError(msg)
    return indices


def _first_bright_pole(spectrum: ProjectedSpectrum, *, min_weight: float) -> float:
    indices = _bright_indices(spectrum, min_weight=float(min_weight), max_states=1)
    return float(np.asarray(spectrum.excitation_energies)[indices[0]])


def _ritz_coefficients_in_raw_carrier_basis(
    raw_overlap: np.ndarray,
    raw_hamiltonian: np.ndarray,
    raw_source: np.ndarray,
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    whitening_transform: np.ndarray | None,
    *,
    fixed_whitening: bool,
    overlap_cutoff: float,
) -> np.ndarray:
    if fixed_whitening:
        if whitening_transform is None:
            msg = "fixed-whitening Ritz coefficients require a whitening transform"
            raise ValueError(msg)
        _, _, transformed_coefficients = _projected_ritz_coefficients(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=1e-12,
            relative_cutoff=False,
        )
        return np.asarray(whitening_transform @ transformed_coefficients)
    _, _, raw_coefficients = _projected_ritz_coefficients(
        raw_overlap,
        raw_hamiltonian,
        raw_source,
        overlap_cutoff=float(overlap_cutoff),
        relative_cutoff=True,
    )
    return np.asarray(raw_coefficients, dtype=np.float64)


def _projected_ritz_coefficients(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    relative_cutoff: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap_arr = np.asarray(overlap, dtype=np.float64)
    hamiltonian_arr = np.asarray(hamiltonian, dtype=np.float64)
    source_arr = np.asarray(source, dtype=np.float64)
    if source_arr.ndim == 1:
        source_arr = source_arr[:, None]
    overlap_arr = (overlap_arr + overlap_arr.T) / 2
    hamiltonian_arr = (hamiltonian_arr + hamiltonian_arr.T) / 2
    diag = np.real(np.diag(overlap_arr))
    positive_diag = np.where(np.isfinite(diag) & (diag > 0), diag, 1.0)
    scale = 1.0 / np.sqrt(np.maximum(positive_diag, 1e-300))
    overlap_scaled = (scale[:, None] * overlap_arr) * scale[None, :]
    hamiltonian_scaled = (scale[:, None] * hamiltonian_arr) * scale[None, :]
    source_scaled = scale[:, None] * source_arr
    overlap_evals, overlap_evecs = np.linalg.eigh(
        (overlap_scaled + overlap_scaled.T) / 2
    )
    if overlap_evals.size == 0 or float(overlap_evals[-1]) <= 0.0:
        msg = "Ritz coefficients require at least one positive overlap direction"
        raise ValueError(msg)
    cutoff = float(overlap_cutoff)
    if bool(relative_cutoff):
        cutoff *= float(overlap_evals[-1])
    keep = overlap_evals > cutoff
    if not np.any(keep):
        msg = f"Ritz coefficients discarded every direction by cutoff {cutoff:g}"
        raise ValueError(msg)
    lambdas = overlap_evals[keep]
    whitening = overlap_evecs[:, keep] / np.sqrt(lambdas)[None, :]
    hamiltonian_tilde = whitening.T @ hamiltonian_scaled @ whitening
    source_tilde = whitening.T @ source_scaled
    energies, vectors = np.linalg.eigh((hamiltonian_tilde + hamiltonian_tilde.T) / 2)
    amplitudes = vectors.T @ source_tilde
    weights = np.asarray(np.abs(amplitudes[:, 0]) ** 2, dtype=np.float64)
    coefficients_scaled = whitening @ vectors
    coefficients = scale[:, None] * coefficients_scaled
    return (
        np.asarray(energies, dtype=np.float64),
        weights,
        np.asarray(coefficients, dtype=np.float64),
    )


def apply_fixed_metric_whitening(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    block_overlaps: np.ndarray,
    block_hamiltonians: np.ndarray,
    block_sources: np.ndarray,
    *,
    overlap_cutoff: float,
    block_stable: bool = False,
    block_mode_min: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Apply one reference metric whitening transform to all estimators.

    Returns:
        Full and block matrices in the same canonical carrier gauge, plus the
        retained metric eigenvalues and raw mode indices.

    """
    overlap_np = np.asarray(overlap, dtype=np.float64)
    hamiltonian_np = np.asarray(hamiltonian, dtype=np.float64)
    source_np = np.asarray(source, dtype=np.float64)
    block_overlaps_np = np.asarray(block_overlaps, dtype=np.float64)
    block_hamiltonians_np = np.asarray(block_hamiltonians, dtype=np.float64)
    block_sources_np = np.asarray(block_sources, dtype=np.float64)
    retained_evals, retained_indices, transform = _fixed_metric_whitening_transform(
        overlap_np,
        block_overlaps_np,
        overlap_cutoff=float(overlap_cutoff),
        block_stable=bool(block_stable),
        block_mode_min=float(block_mode_min),
    )
    overlap_w = transform.T @ overlap_np @ transform
    hamiltonian_w = transform.T @ hamiltonian_np @ transform
    source_w = transform.T @ source_np
    block_overlaps_w = np.asarray(
        [transform.T @ matrix @ transform for matrix in block_overlaps_np],
        dtype=np.float64,
    )
    block_hamiltonians_w = np.asarray(
        [transform.T @ matrix @ transform for matrix in block_hamiltonians_np],
        dtype=np.float64,
    )
    block_sources_w = np.asarray(
        [transform.T @ matrix for matrix in block_sources_np],
        dtype=np.float64,
    )
    return (
        (overlap_w + overlap_w.T) / 2,
        (hamiltonian_w + hamiltonian_w.T) / 2,
        source_w,
        block_overlaps_w,
        block_hamiltonians_w,
        block_sources_w,
        retained_evals,
        retained_indices,
        transform,
    )


def _fixed_metric_whitening_transform(
    overlap: np.ndarray,
    block_overlaps: np.ndarray,
    *,
    overlap_cutoff: float,
    block_stable: bool,
    block_mode_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric = (np.asarray(overlap, dtype=np.float64) + np.asarray(overlap).T) / 2
    evals, evecs = np.linalg.eigh(metric)
    if evals.size == 0 or float(evals[-1]) <= 0:
        msg = "reference explicit carrier metric has no positive direction"
        raise ValueError(msg)
    keep = evals > float(overlap_cutoff) * float(evals[-1])
    if bool(block_stable):
        keep &= _block_stable_mode_mask(
            evals,
            evecs,
            np.asarray(block_overlaps, dtype=np.float64),
            overlap_cutoff=float(overlap_cutoff),
            block_mode_min=float(block_mode_min),
        )
    if not np.any(keep):
        msg = "fixed metric whitening discarded every explicit carrier direction"
        raise ValueError(msg)
    retained_evals = np.asarray(evals[keep], dtype=np.float64)
    retained_indices = np.flatnonzero(keep).astype(np.int64)
    transform = evecs[:, keep] / np.sqrt(retained_evals)[None, :]
    return retained_evals, retained_indices, np.asarray(transform, dtype=np.float64)


def _block_stable_mode_mask(
    evals: np.ndarray,
    evecs: np.ndarray,
    block_overlaps: np.ndarray,
    *,
    overlap_cutoff: float,
    block_mode_min: float,
) -> np.ndarray:
    mask = np.ones(evals.shape, dtype=bool)
    eps = np.finfo(np.float64).tiny
    for block_overlap in np.asarray(block_overlaps, dtype=np.float64):
        block_metric = (block_overlap + block_overlap.T) / 2
        block_evals = np.linalg.eigvalsh(block_metric)
        block_max = max(float(block_evals[-1]) if block_evals.size else 0.0, eps)
        mode_norms = np.einsum(
            "mi,mn,ni->i",
            evecs,
            block_metric,
            evecs,
            optimize=True,
        )
        mask &= mode_norms > float(overlap_cutoff) * block_max
        if block_mode_min > 0:
            mask &= mode_norms / np.maximum(evals, eps) > float(block_mode_min)
    return mask


def _make_batches(n_items: int, batch_size: int):
    for start in range(0, int(n_items), int(batch_size)):
        yield slice(start, min(start + int(batch_size), int(n_items)))
