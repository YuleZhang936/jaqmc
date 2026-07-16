# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: DOC201,DOC501

r"""Pure JAX kernels for spatial-symmetry-aware ground-state training.

The helpers in this module deliberately know nothing about a particular
wavefunction, optimizer, or sampler.  They provide the numerically delicate
pieces needed by those layers: a phase-aware scalar covariance residual,
source-relevance weights, and stable weighted reductions.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
from jax import numpy as jnp


def scalar_log_covariance_residual(
    log_psi: jnp.ndarray,
    transformed_log_psi: jnp.ndarray,
    *,
    character: complex | jnp.ndarray = 1.0 + 0.0j,
    epsilon: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Return the pointwise scalar covariance residual from log amplitudes.

    For a one-dimensional (possibly complex) character :math:`\chi(g)`, the
    residual is

    .. math::

        \frac{|\psi(gX)-\chi(g)\psi(X)|^2}
        {|\psi(gX)|^2+|\chi(g)\psi(X)|^2}.

    ``log_psi`` and ``transformed_log_psi`` may have any identical shape; the
    output has that same shape.  A shared real log scale is removed independently
    for every element before exponentiation.  The result is therefore invariant
    under a common complex rescaling and remains finite for representable log
    amplitudes of arbitrarily large magnitude.

    An exact zero amplitude may be encoded as ``-inf + finite * 1j``.  Two zero
    amplitudes have zero residual.  NaNs, positive infinity, an invalid complex
    phase, or a zero/non-finite character fail closed by producing NaN at the
    affected element.  ``character`` is required to be a scalar because it is a
    one-dimensional group-representation character, not a per-walker weight.
    """
    log_psi_array = jnp.asarray(log_psi)
    transformed_array = jnp.asarray(transformed_log_psi)
    character_array = jnp.asarray(character)
    if log_psi_array.shape != transformed_array.shape:
        msg = (
            "log_psi and transformed_log_psi must have identical shapes, got "
            f"{log_psi_array.shape} and {transformed_array.shape}."
        )
        raise ValueError(msg)
    if character_array.ndim != 0:
        msg = f"character must be a complex scalar, got shape {character_array.shape}."
        raise ValueError(msg)

    complex_dtype = jnp.result_type(
        log_psi_array.dtype,
        transformed_array.dtype,
        character_array.dtype,
        jnp.complex64,
    )
    log_psi_array = log_psi_array.astype(complex_dtype)
    transformed_array = transformed_array.astype(complex_dtype)
    character_array = character_array.astype(complex_dtype)

    psi_finite, psi_zero = _finite_or_encoded_zero(log_psi_array)
    transformed_finite, transformed_zero = _finite_or_encoded_zero(transformed_array)
    log_inputs_valid = (psi_finite | psi_zero) & (transformed_finite | transformed_zero)

    character_abs = jnp.abs(character_array)
    character_valid = (
        jnp.isfinite(jnp.real(character_array))
        & jnp.isfinite(jnp.imag(character_array))
        & (character_abs > 0.0)
    )
    safe_log_character_abs = jnp.where(
        character_valid,
        jnp.log(character_abs),
        jnp.asarray(0.0, dtype=character_abs.dtype),
    )

    psi_target_real_log = jnp.where(
        psi_finite,
        jnp.real(log_psi_array) + safe_log_character_abs,
        -jnp.inf,
    )
    transformed_real_log = jnp.where(
        transformed_finite,
        jnp.real(transformed_array),
        -jnp.inf,
    )
    log_scale = jnp.maximum(psi_target_real_log, transformed_real_log)
    log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
    log_scale = jax.lax.stop_gradient(log_scale)

    psi_delta = jnp.where(
        psi_finite,
        log_psi_array - log_scale,
        jnp.asarray(0.0 + 0.0j, dtype=complex_dtype),
    )
    transformed_delta = jnp.where(
        transformed_finite,
        transformed_array - log_scale,
        jnp.asarray(0.0 + 0.0j, dtype=complex_dtype),
    )
    psi = jnp.where(psi_finite, jnp.exp(psi_delta), 0.0 + 0.0j)
    psi_at_transformed = jnp.where(
        transformed_finite,
        jnp.exp(transformed_delta),
        0.0 + 0.0j,
    )
    target = character_array * psi
    numerator = jnp.abs(psi_at_transformed - target) ** 2
    denominator = jnp.abs(psi_at_transformed) ** 2 + jnp.abs(target) ** 2

    if epsilon is None:
        epsilon_array = jnp.asarray(
            16.0 * jnp.finfo(denominator.dtype).eps,
            dtype=denominator.dtype,
        )
    else:
        epsilon_array = jnp.asarray(epsilon, dtype=denominator.dtype)
    residual = numerator / jnp.maximum(denominator, epsilon_array)
    valid = log_inputs_valid & character_valid
    return jnp.where(valid, residual, jnp.asarray(jnp.nan, residual.dtype))


def scalar_log_covariance_loss(
    log_psi: jnp.ndarray,
    transformed_log_psi: jnp.ndarray,
    *,
    character: complex | jnp.ndarray = 1.0 + 0.0j,
    weights: jnp.ndarray | None = None,
    epsilon: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Average the phase-aware scalar covariance residual.

    When ``weights`` is supplied, this calls :func:`weighted_mean`; otherwise
    it returns the ordinary mean over every input element.  Scalar inputs are
    supported and return their single residual directly.
    """
    residual = scalar_log_covariance_residual(
        log_psi,
        transformed_log_psi,
        character=character,
        epsilon=epsilon,
    )
    if weights is None:
        return jnp.mean(residual)
    return weighted_mean(residual, weights)


def source_relevance_weights(
    dipole: jnp.ndarray,
    reference_dipole: jnp.ndarray | Sequence[float],
    *,
    beta: float | jnp.ndarray,
    minimum_weight: float | jnp.ndarray = 1.0e-3,
    maximum_weight: float | jnp.ndarray = 10.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Build clipped source-relevance weights and their effective sample size.

    Given a Cartesian dipole ``dipole`` with shape ``(..., 3)``, this computes

    .. math::

        t=\|D-D_0\|^2,\qquad
        w=(1-\beta)+\beta\frac{t}{\langle t\rangle},

    then clips the final weights to the strictly positive interval
    ``[minimum_weight, maximum_weight]``.  The returned ESS is the conventional
    absolute effective sample size ``sum(w)**2 / sum(w**2)`` and lies between
    one and the number of samples for valid positive weights.

    If every source displacement is exactly zero, relevance is undefined but
    harmless; the function falls back to uniform unit weights.  Non-finite
    inputs or invalid dynamic parameters fail closed with NaN weights and ESS.
    ``beta`` must lie in ``[0, 1]`` and the clip bounds must satisfy
    ``0 < minimum_weight <= maximum_weight``.
    """
    dipole_array = jnp.asarray(dipole)
    reference_array = jnp.asarray(reference_dipole)
    if dipole_array.ndim < 1 or dipole_array.shape[-1] != 3:
        msg = f"dipole must have shape (..., 3), got {dipole_array.shape}."
        raise ValueError(msg)
    if reference_array.shape not in {(3,), dipole_array.shape}:
        try:
            jnp.broadcast_shapes(dipole_array.shape, reference_array.shape)
        except ValueError as error:
            msg = (
                "reference_dipole must broadcast to dipole, got "
                f"{reference_array.shape} and {dipole_array.shape}."
            )
            raise ValueError(msg) from error

    real_dtype = jnp.result_type(
        jnp.real(dipole_array).dtype,
        jnp.real(reference_array).dtype,
        jnp.float32,
    )
    beta_array = jnp.asarray(beta, dtype=real_dtype)
    minimum_array = jnp.asarray(minimum_weight, dtype=real_dtype)
    maximum_array = jnp.asarray(maximum_weight, dtype=real_dtype)
    if beta_array.ndim != 0:
        raise ValueError(f"beta must be scalar, got shape {beta_array.shape}.")
    if minimum_array.ndim != 0 or maximum_array.ndim != 0:
        raise ValueError("minimum_weight and maximum_weight must be scalars.")

    displacement = dipole_array - reference_array
    relevance = jnp.sum(jnp.abs(displacement) ** 2, axis=-1).astype(real_dtype)
    relevance_scale = jnp.max(relevance)
    safe_relevance_scale = jnp.where(relevance_scale > 0.0, relevance_scale, 1.0)
    safe_relevance_scale = jax.lax.stop_gradient(safe_relevance_scale)
    scaled_relevance = relevance / safe_relevance_scale
    mean_scaled_relevance = jnp.mean(scaled_relevance)
    normalized_relevance = scaled_relevance / jnp.where(
        mean_scaled_relevance > 0.0,
        mean_scaled_relevance,
        1.0,
    )
    normalized_relevance = jnp.where(
        mean_scaled_relevance > 0.0,
        normalized_relevance,
        jnp.ones_like(normalized_relevance),
    )
    raw_weights = (1.0 - beta_array) + beta_array * normalized_relevance
    weights = jnp.clip(raw_weights, minimum_array, maximum_array)

    valid = (
        jnp.all(jnp.isfinite(relevance))
        & jnp.isfinite(beta_array)
        & (beta_array >= 0.0)
        & (beta_array <= 1.0)
        & jnp.isfinite(minimum_array)
        & jnp.isfinite(maximum_array)
        & (minimum_array > 0.0)
        & (maximum_array >= minimum_array)
    )
    weights = jnp.where(valid, weights, jnp.full_like(weights, jnp.nan))
    return weights, effective_sample_size(weights)


def weighted_mean(
    values: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Return a stable nonnegative-weighted mean.

    Weights and values follow NumPy broadcasting rules.  Rescaling all weights
    by an arbitrary positive constant leaves the result unchanged, including
    for very large or very small finite scales.  A reduction containing a
    negative/non-finite weight, or no positive weight, returns NaN.
    """
    value_array = jnp.asarray(values)
    weight_array = jnp.asarray(weights)
    if jnp.issubdtype(weight_array.dtype, jnp.complexfloating):
        raise ValueError("weights must be real-valued.")
    value_array, weight_array = jnp.broadcast_arrays(value_array, weight_array)
    result_dtype = jnp.result_type(value_array.dtype, weight_array.dtype, jnp.float32)
    value_array = value_array.astype(result_dtype)
    weight_array = weight_array.astype(jnp.real(jnp.zeros((), result_dtype)).dtype)

    weight_scale = jnp.max(weight_array, axis=axis, keepdims=True)
    safe_scale = jnp.where(weight_scale > 0.0, weight_scale, 1.0)
    safe_scale = jax.lax.stop_gradient(safe_scale)
    scaled_weights = weight_array / safe_scale
    numerator = jnp.sum(
        scaled_weights * value_array,
        axis=axis,
        keepdims=keepdims,
    )
    denominator = jnp.sum(scaled_weights, axis=axis, keepdims=keepdims)
    weights_valid = jnp.all(
        jnp.isfinite(weight_array) & (weight_array >= 0.0),
        axis=axis,
        keepdims=keepdims,
    )
    valid = weights_valid & jnp.isfinite(denominator) & (denominator > 0.0)
    safe_denominator = jnp.where(valid, denominator, 1.0)
    result = numerator / safe_denominator
    return jnp.where(valid, result, jnp.asarray(jnp.nan, result_dtype))


def effective_sample_size(
    weights: jnp.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> jnp.ndarray:
    r"""Return ``sum(w)**2 / sum(w**2)`` for nonnegative weights.

    The computation first removes a common weight scale.  Invalid reductions
    and reductions without a positive weight return NaN.
    """
    weight_array = jnp.asarray(weights)
    if jnp.issubdtype(weight_array.dtype, jnp.complexfloating):
        raise ValueError("weights must be real-valued.")
    result_dtype = jnp.result_type(weight_array.dtype, jnp.float32)
    weight_array = weight_array.astype(result_dtype)
    weight_scale = jnp.max(weight_array, axis=axis, keepdims=True)
    safe_scale = jnp.where(weight_scale > 0.0, weight_scale, 1.0)
    safe_scale = jax.lax.stop_gradient(safe_scale)
    scaled_weights = weight_array / safe_scale
    weight_sum = jnp.sum(scaled_weights, axis=axis, keepdims=keepdims)
    weight_square_sum = jnp.sum(
        scaled_weights**2,
        axis=axis,
        keepdims=keepdims,
    )
    weights_valid = jnp.all(
        jnp.isfinite(weight_array) & (weight_array >= 0.0),
        axis=axis,
        keepdims=keepdims,
    )
    valid = weights_valid & jnp.isfinite(weight_square_sum) & (weight_square_sum > 0.0)
    safe_square_sum = jnp.where(valid, weight_square_sum, 1.0)
    ess = weight_sum**2 / safe_square_sum
    return jnp.where(valid, ess, jnp.asarray(jnp.nan, result_dtype))


def _finite_or_encoded_zero(
    log_amplitude: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    real_part = jnp.real(log_amplitude)
    imaginary_part = jnp.imag(log_amplitude)
    finite = jnp.isfinite(real_part) & jnp.isfinite(imaginary_part)
    encoded_zero = jnp.isneginf(real_part) & jnp.isfinite(imaginary_part)
    return finite, encoded_zero
