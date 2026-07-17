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
    _calibrated_residual_log_scale,
    _DirectPsiCarry,
    _NQSUpdateCarry,
    _project_source_center_to_invariant_subspace,
    _resolve_atomic_parity_sector,
    _SpringState,
    _symmetry_gradient_updates,
    _vector_covariance_penalty_gradient,
)
from jaqmc.data import BatchedData
from jaqmc.response.nqs_lit import local_action_ratio, nqs_lit_stats_from_source_sums
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


def test_source_aligned_residual_scale_is_relative_to_raw_head_gauge():
    raw_amplitudes = jnp.asarray(
        [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=jnp.complex64,
    )
    raw_logs = jnp.log(raw_amplitudes)
    dipoles = jnp.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    log_scale = _calibrated_residual_log_scale(
        raw_logs,
        jnp.zeros(2, dtype=jnp.complex64),
        dipoles,
        jnp.zeros(3),
        1.0 + 0.0j,
        target_ratio=1e-4,
    )

    np.testing.assert_allclose(log_scale, np.log(1e-5), rtol=2e-6)


def test_even_atomic_ground_selects_source_aligned_hard_odd_response():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.system_config = SimpleNamespace(electron_spins=(1, 0))
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_response_ndets=1,
        nqs_response_hidden_dims_single=(4,),
        nqs_response_hidden_dims_double=(2,),
        nqs_source_aligned=True,
        nqs_source_aligned_residual_scale=1e-10,
        nqs_source_symmetry_mode="inversion",
        nqs_source_symmetry_weight=1.0,
    )
    atom_center = jnp.asarray([0.4, -0.3, 0.2], dtype=jnp.float32)
    data = MoleculeData(
        electrons=jnp.asarray([[0.7, -0.1, 0.5]], dtype=jnp.float32),
        atoms=atom_center[None, :],
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )
    pending_sector = workflow._configured_source_sector(data)
    assert pending_sector.label == "atom_parity_pending"
    np.testing.assert_allclose(pending_sector.center, atom_center, atol=1e-7)
    ground_logpsi = _even_atomic_ground(atom_center)
    parity = workflow._resolve_atomic_parity(
        ground_logpsi,
        {},
        _atomic_batch(atom_center),
        pending_sector,
    )
    assert parity.ground_parity == 1
    assert parity.response_parity == -1
    np.testing.assert_allclose(parity.even_loss, 0.0, atol=2e-7)
    np.testing.assert_allclose(parity.odd_loss, 2.0, rtol=2e-6)
    sector = _resolve_atomic_parity_sector(
        pending_sector,
        parity.response_parity,
    )
    assert sector.label == "atom_odd_hard"

    scalar_apply, vector_apply, params = workflow._make_response_ansatz(
        data,
        jax.random.PRNGKey(7),
        {},
        axis=0,
        source_center=-float(atom_center[0]),
        source_sector=sector,
        response_parity=parity.response_parity,
        ground_logpsi=ground_logpsi,
        initial_omega=-4.0,
    )

    expected_coefficient = 1.0 / (4.0 - 0.02j)
    coefficient_parts = np.asarray(params["source_coefficient"])
    np.testing.assert_allclose(
        coefficient_parts,
        [expected_coefficient.real, expected_coefficient.imag],
        rtol=2e-6,
    )
    assert all(not jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(params))
    assert vector_apply is None

    inverted = data.merge({"electrons": 2.0 * atom_center[None, :] - data.electrons})
    amplitude = jnp.exp(scalar_apply(params, data))
    inverted_amplitude = jnp.exp(scalar_apply(params, inverted))
    np.testing.assert_allclose(
        np.asarray(inverted_amplitude),
        np.asarray(-amplitude),
        rtol=2e-6,
        atol=2e-7,
    )

    action, response_ratio, local_energy = local_action_ratio(
        scalar_apply,
        params,
        ground_logpsi,
        {},
        data,
        ground_energy=-0.5,
        omega=0.3,
        eta=0.02,
    )
    assert np.all(
        np.isfinite(
            np.asarray([action, response_ratio, local_energy], dtype=np.complex64)
        )
    )


def test_odd_signed_atomic_ground_selects_hard_even_response_end_to_end():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.system_config = SimpleNamespace(electron_spins=(1, 0))
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_response_ndets=1,
        nqs_response_hidden_dims_single=(4,),
        nqs_response_hidden_dims_double=(2,),
        nqs_source_aligned=True,
        nqs_source_aligned_residual_scale=1e-10,
    )
    atom_center = jnp.asarray([-0.2, 0.3, 0.1], dtype=jnp.float32)
    data = MoleculeData(
        electrons=atom_center[None, :]
        + jnp.asarray([[0.6, -0.2, 0.4]], dtype=jnp.float32),
        atoms=atom_center[None, :],
        charges=jnp.asarray([1.0], dtype=jnp.float32),
    )
    ground_logpsi = _odd_atomic_ground(atom_center)
    pending_sector = workflow._configured_source_sector(data)
    parity = workflow._resolve_atomic_parity(
        ground_logpsi,
        {},
        _atomic_batch(atom_center),
        pending_sector,
    )

    assert parity.ground_parity == -1
    assert parity.response_parity == 1
    np.testing.assert_allclose(parity.even_loss, 2.0, rtol=2e-6)
    np.testing.assert_allclose(parity.odd_loss, 0.0, atol=2e-7)
    sector = _resolve_atomic_parity_sector(
        pending_sector,
        parity.response_parity,
    )
    assert sector.label == "atom_even_hard"

    scalar_apply, vector_apply, params = workflow._make_response_ansatz(
        data,
        jax.random.PRNGKey(9),
        {},
        axis=0,
        source_center=-float(atom_center[0]),
        source_sector=sector,
        response_parity=parity.response_parity,
        ground_logpsi=ground_logpsi,
        initial_omega=-4.0,
    )

    assert vector_apply is None
    inverted = data.merge({"electrons": 2.0 * atom_center[None, :] - data.electrons})
    amplitude = jnp.exp(scalar_apply(params, data))
    inverted_amplitude = jnp.exp(scalar_apply(params, inverted))
    np.testing.assert_allclose(
        np.asarray(inverted_amplitude),
        np.asarray(amplitude),
        rtol=2e-6,
        atol=2e-7,
    )

    action, response_ratio, local_energy = local_action_ratio(
        scalar_apply,
        params,
        ground_logpsi,
        {},
        data,
        ground_energy=-0.5,
        omega=0.3,
        eta=0.02,
    )
    assert np.all(
        np.isfinite(
            np.asarray([action, response_ratio, local_energy], dtype=np.complex64)
        )
    )


def test_workflow_source_aligned_c1_uses_scalar_response_and_batch_calibration():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.system_config = SimpleNamespace(electron_spins=(1, 0))
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_response_ndets=1,
        nqs_response_hidden_dims_single=(4,),
        nqs_response_hidden_dims_double=(2,),
        nqs_source_aligned=True,
        nqs_source_aligned_residual_scale=1e-10,
        nqs_source_symmetry_mode="auto",
        nqs_source_symmetry_weight=1.0,
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
    charges = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    data = MoleculeData(
        electrons=jnp.asarray([[0.3, -0.2, 0.4]], dtype=jnp.float32),
        atoms=atoms,
        charges=charges,
    )
    initialization_data = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [[[0.3, -0.2, 0.4]], [[0.5, 0.1, -0.3]]],
                dtype=jnp.float32,
            ),
            atoms=atoms,
            charges=charges,
        ),
        fields_with_batch=("electrons",),
    )
    sector = workflow._configured_source_sector(data)
    assert sector.is_trivial
    assert workflow._active_source_sector_operations(sector) == ()

    def ground_logpsi(_params, point):
        return -jnp.linalg.norm(point.electrons[0])

    source_center = 0.1
    scalar_apply, vector_apply, params = workflow._make_response_ansatz(
        data,
        jax.random.PRNGKey(8),
        {},
        axis=0,
        source_center=source_center,
        source_centers=jnp.asarray([9.0, 8.0, 7.0]),
        source_sector=sector,
        ground_logpsi=ground_logpsi,
        initialization_data=initialization_data,
        initial_omega=-4.0,
    )

    assert vector_apply is None
    scalar_logpsi = scalar_apply(params, data)
    assert scalar_logpsi.shape == ()
    coefficient_parts = np.asarray(params["source_coefficient"])
    coefficient = coefficient_parts[0] + 1j * coefficient_parts[1]
    source = (
        coefficient
        * (-jnp.sum(data.electrons[:, 0]) - source_center)
        * jnp.exp(ground_logpsi({}, data))
    )
    np.testing.assert_allclose(
        np.asarray(jnp.exp(scalar_logpsi)),
        np.asarray(source),
        rtol=2e-5,
        atol=2e-7,
    )
    assert all(not jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(params))


def test_workflow_vector_covariance_penalty_has_parameter_gradient():
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.4, 0.0, 0.0]],
                    [[-0.2, 0.5, -0.4]],
                    [[0.6, -0.3, 0.5]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, inversion),
        label="inversion",
    )
    params = {"even_leakage": jnp.asarray(0.25, dtype=jnp.float32)}

    def vector_apply(local_params, data):
        odd = data.electrons[0].astype(jnp.complex64)
        even = local_params["even_leakage"] * jnp.asarray(
            [1.0, 0.5, -0.3],
            dtype=jnp.complex64,
        )
        return jnp.log(odd + even)

    loss, gradient = _vector_covariance_penalty_gradient(
        vector_apply,
        params,
        batch,
        sector,
        jnp.asarray(inversion),
    )

    assert np.isfinite(float(loss))
    assert float(loss) > 0.0
    assert gradient.shape == (1,)
    assert np.isfinite(float(gradient[0]))
    assert abs(float(gradient[0])) > 1e-5

    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_learning_rate=0.01,
        nqs_sr_max_norm=None,
        nqs_source_symmetry_weight=0.7,
    )
    score = jnp.asarray([[0.2], [-0.1], [0.4]], dtype=jnp.complex64)
    ratio = jnp.asarray([1.0, 1.1, 0.9], dtype=jnp.complex64)
    source_weight = jnp.ones(3, dtype=jnp.float32)
    spring = _SpringState(jnp.zeros(1, dtype=jnp.float32))
    baseline, _, _, _ = workflow._weighted_sr_updates_from_scores(
        params,
        score,
        ratio,
        source_weight,
        spring,
    )
    symmetry_updates = _symmetry_gradient_updates(
        params,
        gradient,
        weight=0.7,
        learning_rate=0.01,
        max_norm=1e-3,
    )
    penalized = jax.tree.map(jnp.add, baseline, symmetry_updates)
    assert not np.allclose(
        np.asarray(baseline["even_leakage"]),
        np.asarray(penalized["even_leakage"]),
    )
    assert float(gradient[0] * symmetry_updates["even_leakage"]) < 0.0
    assert abs(float(symmetry_updates["even_leakage"])) <= 1e-3 + 1e-8


def test_invalid_covariance_loss_cannot_write_nonfinite_parameter_updates():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_learning_rate=0.01,
        nqs_source_symmetry_max_norm=1e-3,
    )
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray([[[0.4, 0.3, 0.2]]], dtype=jnp.float32),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, inversion),
        label="inversion",
    )
    params = {"value": jnp.asarray(0.2, dtype=jnp.float32)}

    def invalid_vector_apply(local_params, _data):
        return jnp.full(
            (3,),
            jnp.asarray(jnp.nan + 0.0j, dtype=jnp.complex64)
            + 0.0 * local_params["value"],
        )

    loss, updates = workflow._source_sector_penalty_updates(
        invalid_vector_apply,
        params,
        batch,
        sector,
        jnp.asarray(inversion),
    )

    assert np.isnan(float(loss))
    np.testing.assert_array_equal(np.asarray(updates["value"]), 0.0)


def test_workflow_source_sector_c1_and_pending_atomic_parity_configuration():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_mode="auto",
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_max_operations=16,
    )
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
    assert workflow._active_source_sector_operations(c1_sector) == ()

    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_mode="inversion",
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_max_operations=2,
    )
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
        active = workflow._active_source_sector_operations(atom_sector)
        guard = workflow._source_guard_operations(atom_sector)
        assert atom_sector.label == expected_label
        assert active == ()
        assert len(guard) == 1
        np.testing.assert_allclose(np.asarray(guard[0]), -np.eye(3), atol=1e-7)


def test_atomic_parity_diagnosis_rejects_mixed_ground_checkpoint():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_eval_batch_size=4,
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
        nqs_source_symmetry_mode="auto",
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_max_operations=16,
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


def test_multicenter_finite_non_c1_is_rejected_even_when_legacy_mode_is_off():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_mode="off",
        nqs_source_symmetry_weight=0.0,
    )
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
        nqs_source_symmetry_eval_batch_size=4,
        nqs_source_symmetry_max_covariance=1e-4,
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

    covariance = workflow._validate_pure_source_covariance(
        ground_logpsi,
        {},
        batch,
        sector,
        -atom_center,
        axis=0,
        response_parity=response_parity,
    )
    np.testing.assert_allclose(float(covariance.mean_loss), 0.0, atol=2e-7)
    np.testing.assert_allclose(float(covariance.max_loss), 0.0, atol=2e-7)
    assert int(covariance.worst_operation_index) == 0


def test_atomic_pure_source_guard_rejects_parity_leakage():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_eval_batch_size=4,
        nqs_source_symmetry_max_covariance=1e-4,
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
        workflow._validate_pure_source_covariance(
            asymmetric_ground,
            {},
            batch,
            sector,
            -atom_center,
            axis=0,
            response_parity=-1,
        )


def test_pure_source_covariance_guard_uses_worst_operation_not_mean():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_mode="general",
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_eval_batch_size=4,
        nqs_source_symmetry_max_covariance=100.0,
    )
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.4, 0.3, 0.2]],
                    [[-0.2, 0.5, -0.4]],
                    [[0.6, -0.3, 0.5]],
                    [[-0.7, -0.2, 0.3]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    reflect_x = tuple(
        tuple(float(value) for value in row)
        for row in np.diag(np.asarray([-1.0, 1.0, 1.0]))
    )
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, reflect_x, inversion),
        label="two-operation-test",
    )

    def ground_with_y_leakage(_params, data):
        return -0.5 * jnp.sum(data.electrons**2) + 2.0 * data.electrons[0, 1]

    metrics = workflow._validate_pure_source_covariance(
        ground_with_y_leakage,
        {},
        batch,
        sector,
        jnp.zeros(3),
        axis=0,
    )
    assert float(metrics.mean_loss) < float(metrics.max_loss)
    worst_operation = workflow._active_source_sector_operations(sector)[
        int(metrics.worst_operation_index)
    ]
    np.testing.assert_allclose(np.asarray(worst_operation), -np.eye(3), atol=1e-7)

    workflow.lit_config.nqs_source_symmetry_max_covariance = 0.5 * (
        float(metrics.mean_loss) + float(metrics.max_loss)
    )
    with pytest.raises(RuntimeError, match="worst-operation"):
        workflow._validate_pure_source_covariance(
            ground_with_y_leakage,
            {},
            batch,
            sector,
            jnp.zeros(3),
            axis=0,
        )


def test_pure_source_covariance_guard_is_disabled_without_maximum():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_source_symmetry_mode="inversion",
        nqs_source_symmetry_weight=1.0,
        nqs_source_symmetry_max_covariance=None,
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, inversion),
        label="inversion",
    )

    def should_not_run(*_args):
        raise AssertionError("disabled source guard evaluated the ground model")

    metrics = workflow._validate_pure_source_covariance(
        should_not_run,
        {},
        None,
        sector,
        jnp.zeros(3),
        axis=0,
    )
    np.testing.assert_allclose(float(metrics.mean_loss), 0.0)
    np.testing.assert_allclose(float(metrics.max_loss), 0.0)
    assert int(metrics.worst_operation_index) == -1


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


def test_direct_fallback_keeps_separate_source_sector_update_and_loss():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=3)
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_direct_psi_train=True,
        nqs_reweight_ess_fraction_min=0.5,
        nqs_train_update_batch_size=3,
        nqs_eval_batch_size=3,
        nqs_source_symmetry_mode="inversion",
        nqs_source_symmetry_weight=0.7,
        nqs_source_symmetry_learning_rate=0.01,
        nqs_source_symmetry_max_norm=1e-3,
    )
    batch = BatchedData(
        data=MoleculeData(
            electrons=jnp.asarray(
                [
                    [[0.4, 0.3, 0.2]],
                    [[0.2, 0.5, 0.4]],
                    [[0.6, 0.3, 0.5]],
                ],
                dtype=jnp.float32,
            ),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.asarray([1.0], dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    inversion = tuple(tuple(float(value) for value in row) for row in -np.eye(3))
    sector = SourceSector(
        center=(0.0, 0.0, 0.0),
        operations=(identity, inversion),
        label="inversion",
    )
    params = {"even_leakage": jnp.asarray(0.25, dtype=jnp.float32)}

    def vector_apply(local_params, data):
        odd = data.electrons[0].astype(jnp.complex64)
        even = local_params["even_leakage"] * jnp.asarray(
            [1.0, 0.5, -0.3],
            dtype=jnp.complex64,
        )
        return jnp.log(odd + even)

    def response_apply(local_params, data):
        return vector_apply(local_params, data)[0]

    def ground_logpsi(_params, data):
        return -0.5 * jnp.sum(data.electrons**2)

    def make_trivial_collector(*_args, **_kwargs):
        def collect(_params, direct_data, sampler_state, rng, _omega):
            return direct_data, direct_data, sampler_state, rng

        return collect

    def fake_weighted_updates(
        _response_apply,
        local_params,
        *_args,
        spring_state,
        **_kwargs,
    ):
        return (
            jax.tree.map(jnp.zeros_like, local_params),
            spring_state,
            jnp.asarray(1e-3, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            lit_workflow_module._empty_spring_optimizer_diagnostics(local_params),
        )

    def fake_direct_updates(
        _response_apply,
        local_params,
        _ground_logpsi,
        _ground_params,
        source_sums,
        _psi_batch,
        *,
        spring_state,
        source_norm,
        omega,
        **_kwargs,
    ):
        stats = nqs_lit_stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=workflow.lit_config.eta,
        )._replace(
            loss=jnp.asarray(0.0, dtype=jnp.float32),
            fidelity=jnp.asarray(1.0, dtype=jnp.float32),
            reverse_kl=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return (
            stats,
            jax.tree.map(jnp.zeros_like, local_params),
            spring_state,
            jnp.asarray(1e-3, dtype=jnp.float32),
        )

    workflow._make_direct_psi_pool_collector = make_trivial_collector
    workflow._weighted_sr_updates = fake_weighted_updates
    workflow._direct_sr_stats_and_updates_from_source_sums = fake_direct_updates

    covariance_loss, covariance_updates = workflow._source_sector_penalty_updates(
        vector_apply,
        params,
        batch,
        sector,
        jnp.asarray(inversion),
    )
    update = workflow._make_nqs_update_step(
        response_apply,
        {},
        ground_logpsi,
        -0.5,
        response_vector_apply=vector_apply,
        source_sector=sector,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
    )
    heldout_covariance_loss = update.source_covariance_loss(params, batch)
    np.testing.assert_allclose(
        float(heldout_covariance_loss),
        float(covariance_loss),
        rtol=2e-6,
    )
    carry = _NQSUpdateCarry(
        direct=_DirectPsiCarry(
            batched_data=batch,
            sampler_state={},
            rng=jax.random.PRNGKey(5),
            initialized=jnp.asarray(True),
            use_direct=jnp.asarray(True),
        ),
        spring=_SpringState(previous_direction=jnp.zeros(1, dtype=jnp.float32)),
    )

    next_params, stats, next_carry = update(
        params,
        batch,
        jnp.asarray(0.3, dtype=jnp.float32),
        carry,
        batch_index=1,
    )

    expected_params = jax.tree.map(jnp.add, params, covariance_updates)
    np.testing.assert_allclose(
        np.asarray(next_params["even_leakage"]),
        np.asarray(expected_params["even_leakage"]),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        float(stats.loss),
        0.7 * float(covariance_loss),
        rtol=2e-5,
    )
    assert bool(next_carry.direct.use_direct)
