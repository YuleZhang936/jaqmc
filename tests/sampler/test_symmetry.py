# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.sampler.mcmc import gaussian_proposal
from jaqmc.sampler.symmetry import (
    make_haar_orthogonal_proposal,
    make_linear_haar_proposal,
    make_symmetry_mixture_proposal,
    sample_finite_group_indices,
    sample_haar_orthogonal_operations,
    sample_linear_orthogonal_operations,
    transform_electron_coordinates,
)


def _c4_operations() -> np.ndarray:
    return np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )


def test_transform_coordinates_uses_center_and_arbitrary_leading_batch():
    operations = _c4_operations()
    center = jnp.asarray([0.4, -0.3, 1.2])
    electrons = jnp.asarray(
        np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(2, 3, 2, 3) / 7.0
    )
    indices = jnp.asarray([[0, 1, 2], [3, 2, 1]])

    transformed = transform_electron_coordinates(
        electrons,
        operations,
        indices,
        center,
    )

    expected = np.empty_like(np.asarray(electrons))
    for i in range(2):
        for j in range(3):
            expected[i, j] = (np.asarray(electrons[i, j]) - center) @ operations[
                indices[i, j]
            ].T + center
    np.testing.assert_allclose(transformed, expected, atol=1e-6)
    np.testing.assert_allclose(
        jnp.linalg.norm(transformed - center, axis=-1),
        jnp.linalg.norm(electrons - center, axis=-1),
        atol=1e-6,
    )


def test_finite_group_indices_are_uniform_and_jittable():
    draw = jax.jit(lambda key: sample_finite_group_indices(key, 4, (40_000,)))
    indices = draw(jax.random.key(17))

    assert indices.shape == (40_000,)
    counts = np.bincount(np.asarray(indices), minlength=4) / indices.size
    np.testing.assert_allclose(counts, np.full(4, 0.25), atol=0.01)


def test_zero_probability_is_exact_original_gaussian_rng_path():
    proposal = make_symmetry_mixture_proposal(
        _c4_operations(),
        center=[0.2, -0.1, 0.3],
        mix_probability=0.0,
    )
    key = jax.random.key(3)
    electrons = jnp.arange(30, dtype=jnp.float32).reshape(5, 2, 3)

    assert proposal is gaussian_proposal
    actual = proposal(key, {"electrons": electrons}, 0.17)
    expected = gaussian_proposal(key, {"electrons": electrons}, 0.17)
    np.testing.assert_array_equal(actual["electrons"], expected["electrons"])


def test_pure_group_proposal_supports_array_and_electron_pytree():
    operations = _c4_operations()
    center = np.asarray([0.2, -0.4, 0.7])
    electrons = jnp.asarray(
        [
            [[0.5, 0.1, 1.1], [-0.8, 0.2, 0.3]],
            [[1.2, -0.1, -0.5], [0.4, 0.8, 1.3]],
            [[-0.2, -0.7, 0.9], [0.1, -0.5, -0.6]],
        ]
    )
    proposal = make_symmetry_mixture_proposal(operations, center, 1.0)
    key = jax.random.key(9)

    transformed_array = jax.jit(proposal)(key, electrons, 99.0)
    transformed_tree = jax.jit(proposal)(key, {"electrons": electrons}, 0.0)

    np.testing.assert_array_equal(
        transformed_array,
        transformed_tree["electrons"],
    )
    for walker, transformed in zip(electrons, transformed_array, strict=True):
        assert any(
            np.allclose(
                transformed,
                (np.asarray(walker) - center) @ operation.T + center,
                atol=1e-6,
            )
            for operation in operations
        )


def test_mixture_decision_is_independent_per_walker():
    operations = np.asarray([np.eye(3), -np.eye(3)])
    center = np.zeros(3)
    electrons = jnp.ones((256, 1, 3))
    proposal = make_symmetry_mixture_proposal(operations, center, 0.5)

    result = jax.jit(proposal)(jax.random.key(42), electrons, 0.0)

    # With zero Gaussian width, Gaussian walkers remain +1.  Group walkers are
    # either +1 (identity) or -1 (inversion), so observing both signs verifies
    # that the group branch is selected walker-by-walker rather than globally.
    unique = np.unique(np.asarray(result[:, 0, 0]))
    np.testing.assert_array_equal(unique, np.asarray([-1.0, 1.0]))
    np.testing.assert_array_equal(result[..., 0], result[..., 1])
    np.testing.assert_array_equal(result[..., 1], result[..., 2])


@pytest.mark.parametrize("include_improper", [False, True])
def test_haar_operations_are_orthogonal_with_expected_determinants(include_improper):
    operations = jax.jit(
        lambda key: sample_haar_orthogonal_operations(
            key,
            (512,),
            include_improper=include_improper,
        )
    )(jax.random.key(25))

    products = jnp.einsum("...ji,...jk->...ik", operations, operations)
    np.testing.assert_allclose(
        products,
        np.broadcast_to(np.eye(3), products.shape),
        atol=1e-5,
    )
    determinants = np.linalg.det(np.asarray(operations))
    if include_improper:
        assert np.any(determinants < 0.0)
        assert np.any(determinants > 0.0)
        assert abs(np.mean(determinants)) < 0.15
    else:
        np.testing.assert_allclose(determinants, np.ones(512), atol=1e-5)


def test_atomic_haar_proposal_preserves_radii_and_uses_improper_component():
    center = np.asarray([0.3, -0.2, 0.5])
    electrons = jnp.asarray(
        np.arange(512 * 2 * 3, dtype=np.float32).reshape(512, 2, 3) / 100.0
    )
    proposal = make_haar_orthogonal_proposal(center, include_improper=True)

    transformed = jax.jit(proposal)(jax.random.key(73), electrons, 99.0)

    np.testing.assert_allclose(
        jnp.linalg.norm(transformed - center, axis=-1),
        jnp.linalg.norm(electrons - center, axis=-1),
        rtol=2e-5,
        atol=2e-5,
    )
    assert not np.allclose(transformed, electrons)


@pytest.mark.parametrize("allow_axis_reversal", [False, True])
def test_linear_haar_operations_preserve_labeled_line(allow_axis_reversal):
    axis = np.asarray([1.0, 2.0, -0.7])
    axis /= np.linalg.norm(axis)
    operations = sample_linear_orthogonal_operations(
        jax.random.key(81),
        axis,
        (1024,),
        allow_axis_reversal=allow_axis_reversal,
    )

    products = jnp.einsum("...ji,...jk->...ik", operations, operations)
    mapped_axis = jnp.einsum("...ij,j->...i", operations, axis)
    np.testing.assert_allclose(
        products,
        np.broadcast_to(np.eye(3), products.shape),
        atol=2e-5,
    )
    if allow_axis_reversal:
        signs = np.asarray(mapped_axis @ axis)
        assert np.any(signs < -0.99)
        assert np.any(signs > 0.99)
    else:
        np.testing.assert_allclose(
            mapped_axis,
            np.broadcast_to(axis, mapped_axis.shape),
            atol=2e-5,
        )


def test_linear_haar_proposal_preserves_distances_to_unoriented_axis():
    axis = np.asarray([0.2, -0.4, 1.0])
    axis /= np.linalg.norm(axis)
    center = np.asarray([-0.1, 0.7, 0.2])
    electrons = jnp.asarray(
        np.arange(128 * 2 * 3, dtype=np.float32).reshape(128, 2, 3) / 50.0
    )
    proposal = make_linear_haar_proposal(
        axis,
        center,
        allow_axis_reversal=True,
    )

    transformed = jax.jit(proposal)(jax.random.key(91), electrons, 0.0)
    before = electrons - center
    after = transformed - center
    before_axial = jnp.abs(jnp.einsum("...j,j->...", before, axis))
    after_axial = jnp.abs(jnp.einsum("...j,j->...", after, axis))
    before_radial = jnp.linalg.norm(
        before - jnp.einsum("...i,j->...ij", before @ axis, axis),
        axis=-1,
    )
    after_radial = jnp.linalg.norm(
        after - jnp.einsum("...i,j->...ij", after @ axis, axis),
        axis=-1,
    )

    np.testing.assert_allclose(after_axial, before_axial, atol=2e-5)
    np.testing.assert_allclose(after_radial, before_radial, atol=2e-5)


@pytest.mark.parametrize(
    ("operations", "message"),
    [
        (np.asarray([np.eye(3), np.diag([2.0, 1.0, 1.0])]), "not orthogonal"),
        (
            np.asarray([np.eye(3), _c4_operations()[1]]),
            "has no inverse",
        ),
        (np.asarray([np.eye(3), np.eye(3)]), "duplicates"),
        (
            np.asarray(
                [
                    np.eye(3),
                    np.diag([1.0, -1.0, -1.0]),
                    np.diag([-1.0, 1.0, -1.0]),
                ]
            ),
            "not closed",
        ),
    ],
)
def test_factory_rejects_sets_that_are_not_complete_finite_orthogonal_groups(
    operations,
    message,
):
    with pytest.raises(ValueError, match=message):
        make_symmetry_mixture_proposal(operations, np.zeros(3), 0.2)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan])
def test_factory_rejects_invalid_mixture_probability(probability):
    with pytest.raises(ValueError, match="mix_probability"):
        make_symmetry_mixture_proposal(_c4_operations(), np.zeros(3), probability)
