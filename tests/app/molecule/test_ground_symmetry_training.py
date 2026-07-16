# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.ground_symmetry import scalar_log_covariance_loss
from jaqmc.app.molecule.ground_symmetry_training import (
    GroundSymmetryConfig,
    GroundSymmetryRuntime,
    GroundSymmetrySpecification,
    _huber_loss,
    build_ground_symmetry_specification,
)
from jaqmc.data import BatchedData


def test_atomic_spec_uses_full_finite_group_and_broad_o3_training_bank():
    config = GroundSymmetryConfig(atom_random_rotation_quartets=2)

    specification = build_ground_symmetry_specification(
        np.zeros((1, 3)),
        np.asarray([2.0]),
        config,
    )

    assert specification.label.startswith("atom_Oh+O3bank_")
    assert len(specification.finite_group_operations) == 48
    assert len(specification.operations) > 47
    np.testing.assert_allclose(
        specification.invariant_vector_projector,
        np.zeros((3, 3)),
        atol=1e-8,
    )
    np.testing.assert_allclose(specification.characters, 1.0)


def test_generic_c1_geometry_has_exact_zero_overhead_specification():
    atoms = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.3],
            [-0.4, 1.3, 0.5],
            [0.2, -0.7, 1.8],
        ]
    )
    charges = np.asarray([1.0, 2.0, 3.0, 4.0])

    specification = build_ground_symmetry_specification(
        atoms,
        charges,
        GroundSymmetryConfig(),
    )

    assert specification.is_trivial
    assert len(specification.finite_group_operations) == 1
    np.testing.assert_allclose(
        specification.invariant_vector_projector,
        np.eye(3),
        atol=1e-8,
    )


def test_linear_spec_adds_continuous_o2_bank_that_preserves_labeled_geometry():
    atoms = np.asarray([[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]])
    charges = np.asarray([1.0, 1.0])
    specification = build_ground_symmetry_specification(
        atoms,
        charges,
        GroundSymmetryConfig(
            linear_axial_order=4,
            linear_random_operation_pairs=3,
        ),
    )

    assert specification.label.startswith("linear_D4h+O2bank_")
    assert (
        len(specification.operations) > len(specification.finite_group_operations) - 1
    )
    center = np.asarray(specification.center)
    for operation in specification.operations:
        transformed = (atoms - center) @ np.asarray(operation).T + center
        for position in transformed:
            assert np.min(np.linalg.norm(atoms - position, axis=1)) < 1.0e-7


def test_log_huber_keeps_gradient_when_bounded_covariance_is_saturated():
    large_error = jnp.asarray(9.0)
    huber_gradient = jax.grad(lambda x: _huber_loss(x, 1.0))(large_error)
    covariance_gradient = jax.grad(
        lambda x: scalar_log_covariance_loss(
            jnp.asarray(0.0 + 0.0j),
            x.astype(jnp.complex64),
        )
    )(large_error)

    np.testing.assert_allclose(huber_gradient, 1.0)
    assert abs(float(covariance_gradient)) < 1.0e-3


def test_runtime_gradient_reduces_anisotropic_log_amplitude_objective():
    swap_xy = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    specification = GroundSymmetrySpecification(
        label="Cs_swap",
        center=(0.0, 0.0, 0.0),
        operations=(swap_xy,),
        characters=(1.0 + 0.0j,),
        finite_group_operations=(identity, swap_xy),
        invariant_vector_projector=(
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )

    def phase_logpsi(params, data):
        log_abs = -jnp.sum(params * data.electrons**2)
        return jnp.asarray(1.0), log_abs

    runtime = GroundSymmetryRuntime(
        phase_logpsi=phase_logpsi,
        specification=specification,
        config=GroundSymmetryConfig(
            atom_random_rotation_quartets=0,
            update_batch_size=4,
            train_source_weight_beta=0.2,
        ),
    )
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.2, 1.1, 0.3]],
                    [[1.4, -0.1, 0.5]],
                    [[0.7, 1.8, -0.2]],
                    [[-1.2, 0.4, 0.8]],
                ]
            ),
            atoms=jnp.zeros((1, 3)),
            charges=jnp.ones(1),
        ),
        fields_with_batch=["electrons"],
    )
    params = jnp.asarray([0.4, 2.2, 0.7])

    loss, grads, stats = runtime.loss_and_grad(
        params,
        batch,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.2),
    )
    next_loss, _, _ = runtime.loss_and_grad(
        params - 1.0e-2 * grads,
        batch,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.2),
    )

    assert float(loss) > 0.0
    assert float(next_loss) < float(loss)
    assert float(stats["ground_symmetry_covariance_loss"]) > 0.0
    assert 0.0 < float(stats["ground_symmetry_source_ess_fraction"]) <= 1.0
