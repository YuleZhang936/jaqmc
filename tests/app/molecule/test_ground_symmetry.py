# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.ground_symmetry import (
    effective_sample_size,
    scalar_log_covariance_loss,
    scalar_log_covariance_residual,
    source_relevance_weights,
    weighted_mean,
)


def test_scalar_covariance_supports_complex_character_and_batch_residuals():
    psi = jnp.asarray(
        [1.0 + 0.2j, -0.4 + 0.7j, 0.3 - 1.1j],
        dtype=jnp.complex64,
    )
    character = jnp.exp(0.71j)
    transformed = character * psi

    residual = scalar_log_covariance_residual(
        jnp.log(psi),
        jnp.log(transformed),
        character=character,
    )
    incorrect = scalar_log_covariance_loss(
        jnp.log(psi),
        jnp.log(transformed),
        character=1.0 + 0.0j,
    )

    assert residual.shape == psi.shape
    np.testing.assert_allclose(residual, 0.0, atol=2e-6)
    assert float(incorrect) > 0.1


def test_scalar_covariance_is_scale_invariant_stable_jittable_and_differentiable():
    log_psi = jnp.asarray([0.3 + 0.2j, -2.1 - 0.5j], dtype=jnp.complex64)
    transformed = jnp.asarray([-0.2 + 0.9j, -1.7 + 0.3j], dtype=jnp.complex64)
    character = jnp.exp(-0.37j)

    reference = scalar_log_covariance_loss(
        log_psi,
        transformed,
        character=character,
    )
    huge_scale = scalar_log_covariance_loss(
        log_psi + 1.0e4 + 1.23j,
        transformed + 1.0e4 + 1.23j,
        character=character,
    )
    jitted = jax.jit(scalar_log_covariance_loss)(
        log_psi,
        transformed,
        character=character,
    )
    gradient = jax.grad(
        lambda shift: scalar_log_covariance_loss(
            log_psi + shift,
            transformed,
            character=character,
        )
    )(jnp.asarray(0.13, dtype=jnp.float32))

    np.testing.assert_allclose(huge_scale, reference, atol=5e-5)
    np.testing.assert_allclose(jitted, reference, atol=2e-6)
    assert bool(jnp.isfinite(gradient))
    assert abs(float(gradient)) > 1.0e-5


def test_scalar_covariance_distinguishes_encoded_zero_from_invalid_logs():
    encoded_zero = jnp.asarray(complex(float("-inf"), 0.0), dtype=jnp.complex64)
    finite = jnp.asarray(0.2 + 0.3j, dtype=jnp.complex64)

    both_zero = scalar_log_covariance_loss(encoded_zero, encoded_zero)
    one_zero = scalar_log_covariance_loss(encoded_zero, finite)
    invalid_log = scalar_log_covariance_loss(jnp.inf + 0.0j, finite)
    invalid_phase = scalar_log_covariance_loss(jnp.nan + 0.0j, finite)
    invalid_character = scalar_log_covariance_loss(
        finite,
        finite,
        character=0.0 + 0.0j,
    )

    np.testing.assert_allclose(both_zero, 0.0)
    np.testing.assert_allclose(one_zero, 1.0, atol=2e-6)
    assert np.isnan(float(invalid_log))
    assert np.isnan(float(invalid_phase))
    assert np.isnan(float(invalid_character))


def test_scalar_covariance_weighted_loss_uses_samplewise_source_weights():
    log_psi = jnp.log(jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.complex64))
    transformed = jnp.log(jnp.asarray([1.0, -1.0, 1.0j], dtype=jnp.complex64))
    weights = jnp.asarray([1.0, 2.0, 3.0])

    residual = scalar_log_covariance_residual(log_psi, transformed)
    loss = scalar_log_covariance_loss(log_psi, transformed, weights=weights)

    np.testing.assert_allclose(
        loss,
        jnp.sum(weights * residual) / jnp.sum(weights),
        rtol=2.0e-6,
    )


def test_source_relevance_weights_follow_formula_clip_and_report_absolute_ess():
    dipole = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    beta = 0.6

    weights, ess = source_relevance_weights(
        dipole,
        jnp.zeros(3),
        beta=beta,
        minimum_weight=0.2,
        maximum_weight=1.5,
    )
    relevance = jnp.asarray([0.0, 1.0, 4.0])
    expected = jnp.clip(
        (1.0 - beta) + beta * relevance / jnp.mean(relevance),
        0.2,
        1.5,
    )
    expected_ess = jnp.sum(expected) ** 2 / jnp.sum(expected**2)

    np.testing.assert_allclose(weights, expected, rtol=2e-6)
    np.testing.assert_allclose(ess, expected_ess, rtol=2e-6)
    assert bool(jnp.all(weights > 0.0))


def test_source_relevance_zero_source_is_uniform_and_jittable():
    dipole = jnp.zeros((5, 3), dtype=jnp.float32)
    build_weights = jax.jit(
        lambda local_dipole, local_beta: source_relevance_weights(
            local_dipole,
            jnp.zeros(3),
            beta=local_beta,
            minimum_weight=0.1,
            maximum_weight=4.0,
        )
    )

    weights, ess = build_weights(dipole, jnp.asarray(0.7))

    np.testing.assert_array_equal(weights, jnp.ones(5))
    np.testing.assert_allclose(ess, 5.0)


def test_source_relevance_invalid_dynamic_inputs_fail_closed():
    dipole = jnp.asarray([[0.0, 0.0, 0.0], [jnp.nan, 1.0, 0.0]])

    invalid_dipole_weights, invalid_dipole_ess = source_relevance_weights(
        dipole,
        jnp.zeros(3),
        beta=0.2,
    )
    invalid_beta_weights, invalid_beta_ess = source_relevance_weights(
        jnp.zeros((2, 3)),
        jnp.zeros(3),
        beta=1.1,
    )

    assert bool(jnp.all(jnp.isnan(invalid_dipole_weights)))
    assert np.isnan(float(invalid_dipole_ess))
    assert bool(jnp.all(jnp.isnan(invalid_beta_weights)))
    assert np.isnan(float(invalid_beta_ess))


def test_weighted_reductions_are_stable_complex_jittable_and_differentiable():
    values = jnp.asarray([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j])
    weights = jnp.asarray([1.0e30, 2.0e30, 3.0e30])
    expected = jnp.sum(jnp.asarray([1.0, 2.0, 3.0]) * values) / 6.0

    result = jax.jit(weighted_mean)(values, weights)
    ess = jax.jit(effective_sample_size)(weights)
    gradient = jax.grad(
        lambda local_weights: jnp.real(weighted_mean(values, local_weights))
    )(weights / 1.0e30)

    np.testing.assert_allclose(result, expected, rtol=2e-6)
    np.testing.assert_allclose(ess, 36.0 / 14.0, rtol=2e-6)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize(
    "weights",
    [jnp.asarray([0.0, 0.0]), jnp.asarray([1.0, -0.1]), jnp.asarray([1.0, jnp.inf])],
)
def test_weighted_reductions_fail_closed_for_invalid_weights(weights):
    values = jnp.asarray([1.0, 2.0])

    assert np.isnan(float(weighted_mean(values, weights)))
    assert np.isnan(float(effective_sample_size(weights)))


def test_public_helpers_reject_incompatible_static_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        scalar_log_covariance_loss(jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="complex scalar"):
        scalar_log_covariance_loss(jnp.ones(2), jnp.ones(2), character=jnp.ones(2))
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        source_relevance_weights(jnp.ones((2, 2)), jnp.zeros(3), beta=0.2)
