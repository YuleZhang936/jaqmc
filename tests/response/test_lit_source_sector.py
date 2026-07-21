# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jaqmc.app.molecule.lit_workflow as lit_workflow_module
from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _project_source_center_to_invariant_subspace,
    _resolve_atomic_parity_sector,
    _shard_batched_data_across_local_devices,
)
from jaqmc.data import BatchedData
from jaqmc.response.source_sector import SourceSector


def _one_electron_data(electrons) -> MoleculeData:
    return MoleculeData(
        electrons=jnp.asarray(electrons, dtype=jnp.float32),
        atoms=jnp.zeros((1, 3), dtype=jnp.float32),
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )


def _atomic_batch(atom_center) -> BatchedData:
    center = jnp.asarray(atom_center, dtype=jnp.float32)
    relative_electrons = jnp.asarray(
        [
            [[0.7, 0.2, -0.3]],
            [[-0.6, 0.4, 0.2]],
            [[0.3, -0.5, 0.6]],
            [[-0.4, -0.3, -0.5]],
        ],
        dtype=jnp.float32,
    )
    return BatchedData(
        data=MoleculeData(
            electrons=relative_electrons + center,
            atoms=center[None, :],
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )


def _even_atomic_ground(atom_center):
    center = jnp.asarray(atom_center, dtype=jnp.float32)

    def ground_logpsi(_params, data):
        return -0.5 * jnp.sum((data.electrons - center) ** 2)

    return ground_logpsi


def _odd_atomic_ground(atom_center):
    center = jnp.asarray(atom_center, dtype=jnp.float32)

    def ground_logpsi(_params, data):
        relative = data.electrons - center
        signed_coordinate = relative[0, 0]
        phase = jnp.where(signed_coordinate < 0.0, jnp.pi, 0.0)
        return (
            -0.5 * jnp.sum(relative**2)
            + jnp.log(jnp.abs(signed_coordinate))
            + 1j * phase
        )

    return ground_logpsi


def test_even_atomic_ground_selects_nonzero_direct_hard_odd_response():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.system_config = SimpleNamespace(electron_spins=(1, 0))
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_response_ndets=1,
        nqs_response_hidden_dims_single=(4,),
        nqs_response_hidden_dims_double=(2,),
    )
    atom_center = jnp.asarray([0.4, -0.3, 0.2], dtype=jnp.float32)
    data = MoleculeData(
        electrons=atom_center[None, :]
        + jnp.asarray([[0.7, 0.2, -0.3]], dtype=jnp.float32),
        atoms=atom_center[None, :],
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )
    ground_logpsi = _even_atomic_ground(atom_center)
    pending_sector = workflow._configured_source_sector(data)
    parity = workflow._resolve_atomic_parity(
        ground_logpsi,
        {},
        _atomic_batch(atom_center),
        pending_sector,
    )
    sector = _resolve_atomic_parity_sector(
        pending_sector,
        parity.response_parity,
    )

    scalar_apply, params = workflow._make_response_ansatz(
        data,
        jax.random.PRNGKey(7),
        {},
        source_sector=sector,
        response_parity=parity.response_parity,
    )

    assert parity.response_parity == -1
    assert set(params) == {"params"}
    assert "source_coefficient" not in params
    assert "residual_log_scale" not in params
    inverted = data.merge({"electrons": 2.0 * atom_center[None, :] - data.electrons})
    amplitude = jnp.exp(scalar_apply(params, data))
    inverted_amplitude = jnp.exp(scalar_apply(params, inverted))
    assert np.isfinite(np.asarray(amplitude))
    assert float(jnp.abs(amplitude)) > 1e-10
    np.testing.assert_allclose(
        np.asarray(inverted_amplitude),
        np.asarray(-amplitude),
        rtol=2e-5,
        atol=2e-7,
    )


def test_c1_response_uses_only_direct_raw_parameter_tree():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.system_config = SimpleNamespace(electron_spins=(1, 0))
    workflow.lit_config = MolecularLITConfig(
        nqs_response_ndets=1,
        nqs_response_hidden_dims_single=(4,),
        nqs_response_hidden_dims_double=(2,),
    )
    atoms = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.3, 1.3, 0.4],
            [0.2, -0.5, 1.7],
        ],
        dtype=jnp.float32,
    )
    data = MoleculeData(
        electrons=jnp.asarray([[0.3, -0.2, 0.4]], dtype=jnp.float32),
        atoms=atoms,
        charges=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
    )
    sector = workflow._configured_source_sector(data)

    scalar_apply, params = workflow._make_response_ansatz(
        data,
        jax.random.PRNGKey(8),
        {},
        source_sector=sector,
    )

    assert sector.is_trivial
    assert set(params) == {"params"}
    assert scalar_apply(params, data).shape == ()
    assert np.isfinite(np.asarray(scalar_apply(params, data)))


def test_workflow_source_sector_c1_and_pending_atomic_parity_configuration():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig()
    generic = MoleculeData(
        electrons=jnp.zeros((1, 3), dtype=jnp.float32),
        atoms=jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.2, -0.1],
                [-0.3, 1.3, 0.4],
                [0.2, -0.5, 1.7],
            ],
            dtype=jnp.float32,
        ),
        charges=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
    )

    c1_sector = workflow._configured_source_sector(generic)
    assert c1_sector.is_trivial
    helium = MoleculeData(
        electrons=jnp.zeros((2, 3), dtype=jnp.float32),
        atoms=jnp.zeros((1, 3), dtype=jnp.float32),
        charges=jnp.asarray([2.0], dtype=jnp.float32),
    )

    pending_sector = workflow._configured_source_sector(helium)
    assert pending_sector.label == "atom_parity_pending"
    assert pending_sector.order == 2

    for response_parity, expected_label in (
        (-1, "atom_odd_hard"),
        (1, "atom_even_hard"),
    ):
        atom_sector = _resolve_atomic_parity_sector(
            pending_sector,
            response_parity,
        )
        assert atom_sector.label == expected_label
        assert atom_sector.order == 2
        np.testing.assert_allclose(
            np.asarray(atom_sector.operations[1]), -np.eye(3), atol=1e-7
        )


def test_atomic_parity_diagnosis_rejects_mixed_ground_checkpoint():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_parity_eval_batch_size=4,
        nqs_atomic_ground_parity_max_loss=1e-3,
    )
    atom_center = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float32)
    batch = _atomic_batch(atom_center)
    sector = workflow._configured_source_sector(batch.data)

    def mixed_ground_logpsi(_params, data):
        relative = data.electrons - atom_center
        return -0.5 * jnp.sum(relative**2) + jnp.log(1.0 + 0.5 * relative[0, 0])

    with pytest.raises(RuntimeError, match="not a clean inversion-parity eigenstate"):
        workflow._resolve_atomic_parity(
            mixed_ground_logpsi,
            {},
            batch,
            sector,
        )


def test_c1_parity_diagnosis_bypasses_ground_evaluation():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig()
    atoms = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.2, -0.1],
            [-0.3, 1.3, 0.4],
            [0.2, -0.5, 1.7],
        ],
        dtype=jnp.float32,
    )
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.zeros((2, 1, 3), dtype=jnp.float32),
            atoms=atoms,
            charges=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    sector = workflow._configured_source_sector(batch.data)

    def should_not_run(*_args):
        raise AssertionError("C1 parity diagnosis evaluated the ground model")

    parity = workflow._resolve_atomic_parity(should_not_run, {}, batch, sector)

    assert parity.ground_parity == 0
    assert parity.response_parity == 0
    assert np.isnan(parity.even_loss)
    assert np.isnan(parity.odd_loss)
    assert np.isnan(parity.selected_loss)
    assert _resolve_atomic_parity_sector(sector, 0) is sector


def test_serial_scan_discovers_source_sector_from_physical_geometry(monkeypatch):
    """Unsupported physical symmetry must fail before ground-state restore."""
    atoms = jnp.asarray(
        [[0.2, -0.1, -0.7], [0.2, -0.1, 0.9]],
        dtype=jnp.float32,
    )
    charges = jnp.asarray([6.0, 8.0], dtype=jnp.float32)
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.zeros((2, 2, 3), dtype=jnp.float32),
            atoms=atoms,
            charges=charges,
        ),
        fields_with_batch=("electrons",),
    )
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(seed=7, batch_size=2)
    workflow.system_config = SimpleNamespace()
    workflow.sampler = object()
    workflow.lit_config = MolecularLITConfig(
        axes="x",
        omega_values=(0.5,),
        nqs_burn_in=0,
    )

    monkeypatch.setattr(lit_workflow_module, "data_init", lambda *_args: batch)

    ground_restore_called = False

    def ground_restore_must_not_run(*_args):
        nonlocal ground_restore_called
        ground_restore_called = True
        raise AssertionError("ground-state restore ran before symmetry admission")

    monkeypatch.setattr(
        workflow,
        "_resolve_nqs_ground_state",
        ground_restore_must_not_run,
    )

    with pytest.raises(NotImplementedError, match="linear_C4v"):
        workflow._run_serial_scan()
    assert not ground_restore_called


def test_multicenter_finite_non_c1_is_rejected():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig()
    water = MoleculeData(
        electrons=jnp.zeros((10, 3), dtype=jnp.float32),
        atoms=jnp.asarray(
            [[0.0, 0.0, 0.0], [0.8, 0.6, 0.0], [-0.8, 0.6, 0.0]],
            dtype=jnp.float32,
        ),
        charges=jnp.asarray([8.0, 1.0, 1.0], dtype=jnp.float32),
    )

    with pytest.raises(NotImplementedError, match="finite_O3_4"):
        workflow._configured_source_sector(water)


@pytest.mark.parametrize(
    ("ground_kind", "response_parity", "expected_label"),
    [
        ("even", -1, "atom_odd_hard"),
        ("odd", 1, "atom_even_hard"),
    ],
)
def test_atomic_pure_source_guard_accepts_both_response_parities(
    ground_kind,
    response_parity,
    expected_label,
):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_parity_eval_batch_size=4,
        nqs_atomic_source_parity_max_loss=1e-4,
    )
    atom_center = jnp.asarray([0.2, -0.3, 0.1], dtype=jnp.float32)
    batch = _atomic_batch(atom_center)
    pending_sector = workflow._configured_source_sector(batch.data)
    sector = _resolve_atomic_parity_sector(
        pending_sector,
        response_parity,
    )
    assert sector.label == expected_label
    ground_logpsi = (
        _even_atomic_ground(atom_center)
        if ground_kind == "even"
        else _odd_atomic_ground(atom_center)
    )

    parity_loss = workflow._validate_atomic_source_parity(
        ground_logpsi,
        {},
        batch,
        sector,
        -atom_center,
        axis=0,
        response_parity=response_parity,
    )
    np.testing.assert_allclose(parity_loss, 0.0, atol=2e-7)


def test_atomic_pure_source_guard_accepts_sharded_heldout_pool():
    device_count = jax.local_device_count()
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_parity_eval_batch_size=4 * device_count,
        nqs_atomic_source_parity_max_loss=1e-4,
    )
    atom_center = jnp.asarray([0.2, -0.3, 0.1], dtype=jnp.float32)
    base_batch = _atomic_batch(atom_center)
    eval_pool = BatchedData(
        data=base_batch.data.merge(
            {
                "electrons": jnp.tile(
                    base_batch.data.electrons,
                    (2 * device_count, 1, 1),
                )
            }
        ),
        fields_with_batch=("electrons",),
    )
    eval_pool = _shard_batched_data_across_local_devices(eval_pool)
    sector = _resolve_atomic_parity_sector(
        workflow._configured_source_sector(eval_pool.data),
        -1,
    )

    parity_loss = workflow._validate_atomic_source_parity(
        _even_atomic_ground(atom_center),
        {},
        eval_pool,
        sector,
        -atom_center,
        axis=0,
        response_parity=-1,
    )

    np.testing.assert_allclose(parity_loss, 0.0, atol=2e-7)


def test_atomic_pure_source_guard_rejects_parity_leakage():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_parity_eval_batch_size=4,
        nqs_atomic_source_parity_max_loss=1e-4,
    )
    atom_center = jnp.asarray([0.2, -0.3, 0.1], dtype=jnp.float32)
    batch = _atomic_batch(atom_center)
    sector = _resolve_atomic_parity_sector(
        workflow._configured_source_sector(batch.data),
        -1,
    )
    symmetric_ground = _even_atomic_ground(atom_center)

    def asymmetric_ground(_params, data):
        return symmetric_ground({}, data) + 0.5 * (
            data.electrons[0, 0] - atom_center[0]
        )

    with pytest.raises(RuntimeError, match="pure-source held-out parity"):
        workflow._validate_atomic_source_parity(
            asymmetric_ground,
            {},
            batch,
            sector,
            -atom_center,
            axis=0,
            response_parity=-1,
        )


def test_vector_source_statistics_use_one_shared_sampling_pass():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_source_center_steps=2)
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[1.0, 2.0, 3.0]],
                    [[3.0, 4.0, 5.0]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )

    class CountingPlan:
        def __init__(self):
            self.steps = 0

        def step(self, params, data, state, rng):
            del params, rng
            self.steps += 1
            return data, None, state

    plan = CountingPlan()
    centers, norms, returned_batch, state, _ = workflow._estimate_vector_source_stats(
        {},
        batch,
        "state",
        plan,
        jax.random.PRNGKey(3),
    )

    assert plan.steps == 2
    assert returned_batch is batch
    assert state == "state"
    np.testing.assert_allclose(centers, [-2.0, -3.0, -4.0])
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0])

    workflow.lit_config = MolecularLITConfig(
        nqs_source_center_override=(0.1, -0.2, 0.3),
        nqs_source_norm_override=(1.1, 1.2, 1.3),
    )
    override_plan = CountingPlan()
    override_centers, override_norms, *_ = workflow._estimate_vector_source_stats(
        {},
        batch,
        "state",
        override_plan,
        jax.random.PRNGKey(4),
    )
    assert override_plan.steps == 0
    np.testing.assert_allclose(override_centers, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(override_norms, [1.1, 1.2, 1.3])


def test_source_center_projection_uses_joint_affine_invariant_subspace():
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    reflect_x = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    reflect_y = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    sector = SourceSector(
        center=(0.5, -1.0, 2.0),
        operations=(identity, reflect_x, reflect_y),
        label="two_mirrors",
    )

    projected = _project_source_center_to_invariant_subspace(
        np.asarray([2.0, 3.0, 4.0]),
        sector,
        electron_count=2,
    )

    np.testing.assert_allclose(projected, [-1.0, 2.0, 4.0], atol=1e-14)
    affine_q = projected + 2.0 * np.asarray(sector.center)
    for operation in sector.operations:
        np.testing.assert_allclose(
            np.asarray(operation) @ affine_q,
            affine_q,
            atol=1e-14,
        )

    c1 = SourceSector(center=(5.0, 6.0, 7.0), operations=(identity,), label="C1")
    np.testing.assert_array_equal(
        _project_source_center_to_invariant_subspace(
            np.asarray([2.0, 3.0, 4.0]),
            c1,
            electron_count=2,
        ),
        [2.0, 3.0, 4.0],
    )


def test_vector_source_statistics_project_center_before_finalizing_norms():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_source_center_steps=2)
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[1.0, 2.0, 3.0]],
                    [[3.0, 4.0, 5.0]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )

    class CountingPlan:
        def __init__(self):
            self.steps = 0

        def step(self, params, data, state, rng):
            del params, rng
            self.steps += 1
            return data, None, state

    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, inversion),
        label="inversion",
    )
    plan = CountingPlan()
    centers, norms, *_ = workflow._estimate_vector_source_stats(
        {},
        batch,
        "state",
        plan,
        jax.random.PRNGKey(5),
        source_sector=sector,
    )

    assert plan.steps == 2
    np.testing.assert_allclose(centers, [0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(norms, [5.0, 10.0, 17.0])

    workflow.lit_config = MolecularLITConfig(
        nqs_source_center_override=(0.1, -0.2, 0.3),
        nqs_source_norm_override=(1.1, 1.2, 1.3),
    )
    override_plan = CountingPlan()
    centers, norms, *_ = workflow._estimate_vector_source_stats(
        {},
        batch,
        "state",
        override_plan,
        jax.random.PRNGKey(6),
        source_sector=sector,
    )
    assert override_plan.steps == 0
    np.testing.assert_allclose(centers, [0.0, 0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(norms, [1.1, 1.2, 1.3])
