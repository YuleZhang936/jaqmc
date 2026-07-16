# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Symmetry-orbit proposals for Metropolis-Hastings sampling.

A uniform draw from a complete finite group gives a symmetric proposal kernel:
for every move ``X -> gX``, the inverse move is drawn with the same probability.
Mixing that kernel with the usual symmetric Gaussian all-electron proposal
therefore needs no Hastings correction beyond the target-density ratio already
used by :class:`~jaqmc.sampler.mcmc.MCMCSampler`.

The factory in this module validates the finite group on the host once and
captures only JAX arrays in the returned proposal.  Each walker independently
chooses between a Gaussian move and a uniformly sampled group operation.  The
group cost is only a 3-by-3 coordinate transform per walker; it does not scale
the wavefunction evaluation cost with the order of the group.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey, PyTree

from .mcmc import SamplingProposal, gaussian_proposal

__all__ = [
    "make_haar_orthogonal_proposal",
    "make_linear_haar_proposal",
    "make_symmetry_mixture_proposal",
    "sample_finite_group_indices",
    "sample_haar_orthogonal_operations",
    "sample_linear_orthogonal_operations",
    "transform_electron_coordinates",
]


def sample_finite_group_indices(
    rngs: PRNGKey,
    group_order: int,
    batch_shape: Sequence[int] = (),
) -> jnp.ndarray:
    """Sample independent uniformly distributed finite-group indices.

    Args:
        rngs: JAX random key.
        group_order: Number of operations in the finite group.
        batch_shape: Shape of the walker batch.  An empty shape returns a
            scalar index for one unbatched configuration.

    Returns:
        Integer indices with shape ``batch_shape``.

    Raises:
        ValueError: If ``group_order`` is not a positive integer.
    """
    if int(group_order) != group_order or group_order < 1:
        raise ValueError(
            f"group_order must be a positive integer, got {group_order!r}."
        )
    return jax.random.randint(
        rngs,
        shape=tuple(batch_shape),
        minval=0,
        maxval=int(group_order),
    )


def transform_electron_coordinates(
    electrons: jnp.ndarray,
    operations: jnp.ndarray | np.ndarray | Sequence[Sequence[Sequence[float]]],
    operation_indices: jnp.ndarray,
    center: jnp.ndarray | np.ndarray | Sequence[float],
) -> jnp.ndarray:
    r"""Transform each walker by its selected orthogonal operation.

    Coordinates use the column-vector convention
    ``r' = G @ (r - center) + center``.  ``electrons`` may be unbatched with
    shape ``(n_electrons, 3)`` or have arbitrary leading walker dimensions,
    ``(..., n_electrons, 3)``.  ``operation_indices`` must have exactly those
    leading dimensions.

    Args:
        electrons: Electronic coordinates.
        operations: Stack of matrices with shape ``(group_order, 3, 3)``.
        operation_indices: Selected operation for each walker.
        center: Fixed point of the spatial operations.

    Returns:
        Transformed coordinates with the same shape and dtype as ``electrons``.

    Raises:
        ValueError: If coordinate, operation, index, or center shapes are
            inconsistent.
    """
    coordinates = jnp.asarray(electrons)
    if coordinates.ndim < 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"electrons must have shape (..., n_electrons, 3), got {coordinates.shape}."
        )
    matrices = jnp.asarray(operations, dtype=coordinates.dtype)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            f"operations must have shape (group_order, 3, 3), got {matrices.shape}."
        )
    indices = jnp.asarray(operation_indices)
    batch_shape = coordinates.shape[:-2]
    if indices.shape != batch_shape:
        raise ValueError(
            "operation_indices must match the leading walker dimensions "
            f"{batch_shape}, got {indices.shape}."
        )
    center_array = jnp.asarray(center, dtype=coordinates.dtype)
    if center_array.shape != (3,):
        raise ValueError(f"center must have shape (3,), got {center_array.shape}.")

    selected = matrices[indices]
    return (
        jnp.einsum(
            "...ij,...ej->...ei",
            selected,
            coordinates - center_array,
            precision=jax.lax.Precision.HIGHEST,
        )
        + center_array
    )


def sample_haar_orthogonal_operations(
    rngs: PRNGKey,
    batch_shape: Sequence[int] = (),
    *,
    include_improper: bool = False,
) -> jnp.ndarray:
    """Draw Haar-uniform operations from ``SO(3)`` or ``O(3)``.

    Unit quaternions obtained by normalizing isotropic four-dimensional
    Gaussians induce Haar measure on ``SO(3)``.  When ``include_improper`` is
    true, an independent fair sign multiplies the rotation and hence samples
    both determinant components of ``O(3)`` equally.

    This helper is useful for continuous atomic symmetry diagnostics.  The
    finite-group mixture factory deliberately accepts only a validated finite
    group, for which proposal symmetry can be checked exactly at setup time.

    Args:
        rngs: JAX random key.
        batch_shape: Number and arrangement of operations to draw.
        include_improper: If true, sample Haar-uniform ``O(3)``; otherwise
            sample Haar-uniform ``SO(3)``.

    Returns:
        Orthogonal matrices with shape ``(*batch_shape, 3, 3)``.
    """
    shape = tuple(batch_shape)
    quaternion_rng, parity_rng = jax.random.split(rngs)
    quaternions = jax.random.normal(quaternion_rng, shape=(*shape, 4))
    tiny = jnp.finfo(quaternions.dtype).tiny
    quaternions /= jnp.maximum(
        jnp.linalg.norm(quaternions, axis=-1, keepdims=True),
        tiny,
    )
    w, x, y, z = jnp.moveaxis(quaternions, -1, 0)
    operations = jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*shape, 3, 3))
    if include_improper:
        parity = jax.random.bernoulli(parity_rng, shape=shape)
        sign = jnp.where(parity, -1, 1).astype(operations.dtype)
        operations *= sign[..., None, None]
    return operations


def sample_linear_orthogonal_operations(
    rngs: PRNGKey,
    axis: np.ndarray | Sequence[float],
    batch_shape: Sequence[int] = (),
    *,
    allow_axis_reversal: bool,
) -> jnp.ndarray:
    """Draw Haar operations that preserve an oriented or unoriented line.

    The perpendicular plane is sampled from ``O(2)`` using a uniform angle and
    a fair reflection bit.  For a reversible labeled nuclear geometry, an
    independent fair bit maps the molecular axis to its negative.  The result
    is Haar measure on the compact line stabilizer and is invariant under
    inversion, so it defines a symmetric Metropolis-Hastings proposal.

    Returns:
        Orthogonal matrices shaped ``(*batch_shape, 3, 3)``.
    """
    basis = _linear_axis_basis(axis)
    angle_rng, reflection_rng, reversal_rng = jax.random.split(rngs, 3)
    shape = tuple(batch_shape)
    angles = jax.random.uniform(
        angle_rng,
        shape=shape,
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )
    cosine = jnp.cos(angles)
    sine = jnp.sin(angles)
    reflection = jax.random.bernoulli(reflection_rng, shape=shape)
    o00 = cosine
    o01 = jnp.where(reflection, sine, -sine)
    o10 = sine
    o11 = jnp.where(reflection, -cosine, cosine)
    axis_sign = (
        jnp.where(jax.random.bernoulli(reversal_rng, shape=shape), -1.0, 1.0)
        if allow_axis_reversal
        else jnp.ones(shape)
    )
    zeros = jnp.zeros(shape)
    local = jnp.stack(
        (o00, o01, zeros, o10, o11, zeros, zeros, zeros, axis_sign),
        axis=-1,
    ).reshape((*shape, 3, 3))
    basis_jax = jnp.asarray(basis, dtype=local.dtype)
    return jnp.einsum(
        "ij,...jk,lk->...il",
        basis_jax,
        local,
        basis_jax,
        precision=jax.lax.Precision.HIGHEST,
    )


def make_haar_orthogonal_proposal(
    center: np.ndarray | Sequence[float],
    *,
    include_improper: bool = True,
) -> SamplingProposal:
    """Build a pure Haar ``SO(3)``/``O(3)`` orbit proposal for an atom.

    Returns:
        A proposal compatible with :class:`MCMCSampler`.
    """
    center_array = _validate_center(center)

    def proposal(rngs: PRNGKey, x: PyTree, stddev: float | jnp.ndarray) -> PyTree:
        del stddev
        batch_shape = _validate_coordinate_tree(x)
        operations = sample_haar_orthogonal_operations(
            rngs,
            batch_shape,
            include_improper=include_improper,
        )
        return _transform_tree_with_operations(x, operations, center_array)

    return proposal


def make_linear_haar_proposal(
    axis: np.ndarray | Sequence[float],
    center: np.ndarray | Sequence[float],
    *,
    allow_axis_reversal: bool,
) -> SamplingProposal:
    """Build a pure Haar orbit proposal for a linear labeled geometry.

    Returns:
        A proposal compatible with :class:`MCMCSampler`.
    """
    axis_array = np.asarray(axis, dtype=np.float64)
    _linear_axis_basis(axis_array)
    center_array = _validate_center(center)

    def proposal(rngs: PRNGKey, x: PyTree, stddev: float | jnp.ndarray) -> PyTree:
        del stddev
        batch_shape = _validate_coordinate_tree(x)
        operations = sample_linear_orthogonal_operations(
            rngs,
            axis_array,
            batch_shape,
            allow_axis_reversal=allow_axis_reversal,
        )
        return _transform_tree_with_operations(x, operations, center_array)

    return proposal


def make_symmetry_mixture_proposal(
    operations: np.ndarray | Sequence[Sequence[Sequence[float]]],
    center: np.ndarray | Sequence[float],
    mix_probability: float,
    *,
    tolerance: float = 1e-6,
) -> SamplingProposal:
    """Build a Gaussian/finite-symmetry mixture proposal.

    The returned callable has the same ``(rngs, x, stddev)`` signature as
    :func:`~jaqmc.sampler.mcmc.gaussian_proposal`.  ``x`` may be a coordinate
    array or a PyTree whose leaves are coordinate arrays with a common walker
    batch shape, such as ``{"electrons": electrons}`` in the molecule
    workflow.

    The operations are checked for orthogonality, identity, uniqueness,
    closure, and inverse closure.  They are then sampled uniformly, which is
    what makes the discrete orbit proposal symmetric.  At probability zero the
    factory returns :func:`gaussian_proposal` itself, preserving its exact RNG
    path and introducing no runtime overhead.

    Args:
        operations: Complete finite orthogonal group, shaped ``(order, 3, 3)``.
        center: Common fixed point of the operations.
        mix_probability: Probability that each walker uses a group move rather
            than a Gaussian move.
        tolerance: Absolute tolerance for host-side group validation.

    Returns:
        A proposal compatible with
        :class:`~jaqmc.sampler.mcmc.MCMCSampler`.

    Raises:
        ValueError: If the inputs do not define a complete finite orthogonal
            group and a valid mixture probability.
    """
    operations_array, center_array = _validate_finite_group(
        operations,
        center,
        tolerance=tolerance,
    )
    probability = float(mix_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "mix_probability must be finite and lie in [0, 1], got "
            f"{mix_probability!r}."
        )
    if probability <= 0.0:
        return gaussian_proposal

    operations_jax = jnp.asarray(operations_array)
    center_jax = jnp.asarray(center_array)

    def proposal(rngs: PRNGKey, x: PyTree, stddev: float | jnp.ndarray) -> PyTree:
        leaves = jax.tree.leaves(x)
        if not leaves:
            raise ValueError("x must contain at least one coordinate array.")
        batch_shape = _coordinate_batch_shape(leaves[0])
        for leaf in leaves[1:]:
            leaf_batch_shape = _coordinate_batch_shape(leaf)
            if leaf_batch_shape != batch_shape:
                raise ValueError(
                    "All coordinate leaves must have the same leading walker "
                    f"dimensions; got {batch_shape} and {leaf_batch_shape}."
                )

        gaussian_rng, operation_rng, mixture_rng = jax.random.split(rngs, 3)
        operation_indices = sample_finite_group_indices(
            operation_rng,
            operations_jax.shape[0],
            batch_shape,
        )
        symmetry_state = jax.tree.map(
            lambda coordinates: transform_electron_coordinates(
                coordinates,
                operations_jax,
                operation_indices,
                center_jax,
            ),
            x,
        )
        if probability >= 1.0:
            return symmetry_state

        gaussian_state = gaussian_proposal(gaussian_rng, x, stddev)
        use_symmetry = jax.random.bernoulli(
            mixture_rng,
            p=probability,
            shape=batch_shape,
        )
        return jax.tree.map(
            lambda gaussian, symmetry: jnp.where(
                use_symmetry.reshape(
                    (*batch_shape, *((1,) * (gaussian.ndim - len(batch_shape))))
                ),
                symmetry,
                gaussian,
            ),
            gaussian_state,
            symmetry_state,
        )

    return proposal


def _coordinate_batch_shape(coordinates: Any) -> tuple[int, ...]:
    shape = tuple(coordinates.shape)
    if len(shape) < 2 or shape[-1] != 3:
        raise ValueError(
            f"Every coordinate leaf must have shape (..., n_electrons, 3), got {shape}."
        )
    return tuple(shape[:-2])


def _validate_coordinate_tree(x: PyTree) -> tuple[int, ...]:
    leaves = jax.tree.leaves(x)
    if not leaves:
        raise ValueError("x must contain at least one coordinate array.")
    batch_shape = _coordinate_batch_shape(leaves[0])
    for leaf in leaves[1:]:
        leaf_batch_shape = _coordinate_batch_shape(leaf)
        if leaf_batch_shape != batch_shape:
            raise ValueError(
                "All coordinate leaves must have the same leading walker "
                f"dimensions; got {batch_shape} and {leaf_batch_shape}."
            )
    return batch_shape


def _transform_tree_with_operations(
    x: PyTree,
    operations: jnp.ndarray,
    center: np.ndarray,
) -> PyTree:
    center_jax = jnp.asarray(center)

    def transform(coordinates):
        local_center = center_jax.astype(coordinates.dtype)
        local_operations = operations.astype(coordinates.dtype)
        return (
            jnp.einsum(
                "...ij,...ej->...ei",
                local_operations,
                coordinates - local_center,
                precision=jax.lax.Precision.HIGHEST,
            )
            + local_center
        )

    return jax.tree.map(transform, x)


def _validate_center(center: np.ndarray | Sequence[float]) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (3,) or not np.all(np.isfinite(center_array)):
        raise ValueError("center must be a finite length-3 vector.")
    return center_array


def _linear_axis_basis(axis: np.ndarray | Sequence[float]) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64)
    if axis_array.shape != (3,) or not np.all(np.isfinite(axis_array)):
        raise ValueError("axis must be a finite length-3 vector.")
    norm = np.linalg.norm(axis_array)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("axis must have nonzero norm.")
    direction = axis_array / norm
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return np.stack((first, second, direction), axis=1)


def _validate_finite_group(
    operations: np.ndarray | Sequence[Sequence[Sequence[float]]],
    center: np.ndarray | Sequence[float],
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive and finite, got {tolerance!r}.")
    matrices = np.asarray(operations, dtype=np.float64)
    fixed_point = np.asarray(center, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or matrices.shape[0] < 1:
        raise ValueError(
            "operations must have shape (group_order, 3, 3) with positive "
            f"group_order, got {matrices.shape}."
        )
    if fixed_point.shape != (3,):
        raise ValueError(f"center must have shape (3,), got {fixed_point.shape}.")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(fixed_point)):
        raise ValueError("operations and center must contain only finite values.")

    _validate_orthogonal_identity(matrices, tolerance)
    _validate_unique_operations(matrices, tolerance)
    _validate_inverse_closure(matrices, tolerance)
    _validate_product_closure(matrices, tolerance)
    return matrices, fixed_point


def _validate_orthogonal_identity(matrices: np.ndarray, tolerance: float) -> None:
    identity = np.eye(3)
    orthogonality_error = np.max(
        np.abs(np.einsum("nji,njk->nik", matrices, matrices) - identity),
        axis=(1, 2),
    )
    if np.any(orthogonality_error > tolerance):
        index = int(np.argmax(orthogonality_error))
        raise ValueError(
            f"Operation {index} is not orthogonal within tolerance {tolerance}."
        )
    if not _contains_matrix(matrices, identity, tolerance):
        raise ValueError("A finite group must contain the identity operation.")


def _validate_unique_operations(matrices: np.ndarray, tolerance: float) -> None:
    for first in range(matrices.shape[0]):
        for second in range(first):
            if np.allclose(
                matrices[first],
                matrices[second],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"Operations {second} and {first} are duplicates within "
                    f"tolerance {tolerance}."
                )


def _validate_inverse_closure(matrices: np.ndarray, tolerance: float) -> None:
    for index, matrix in enumerate(matrices):
        if not _contains_matrix(matrices, matrix.T, tolerance):
            raise ValueError(f"Operation {index} has no inverse in the supplied set.")


def _validate_product_closure(matrices: np.ndarray, tolerance: float) -> None:
    for first, left in enumerate(matrices):
        for second, right in enumerate(matrices):
            if not _contains_matrix(matrices, left @ right, tolerance):
                raise ValueError(
                    "The supplied operations are not closed: product of "
                    f"operations {first} and {second} is missing."
                )


def _contains_matrix(
    matrices: np.ndarray,
    candidate: np.ndarray,
    tolerance: float,
) -> bool:
    return bool(
        np.any(
            np.all(
                np.isclose(matrices, candidate, rtol=0.0, atol=tolerance),
                axis=(1, 2),
            )
        )
    )
