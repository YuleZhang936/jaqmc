# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the single-frequency NQS-LIT data-parallel kernels."""

from functools import reduce
from operator import itemgetter
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.sharding import PartitionSpec

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _add_source_sums,
    _merge_source_sums_across_devices,
    _NQSUpdateCarry,
    _regularized_action_gradient,
    _solve_sr_direction_chunked,
    _solve_sr_direction_data_parallel,
    _spring_direction_chunked,
    _spring_direction_data_parallel,
    _SpringState,
)
from jaqmc.data import BatchedData
from jaqmc.response.nqs_lit import NQSLITSourceSums, WeightedComplexMoments
from jaqmc.utils import parallel_jax


def _replicated_specs(tree):
    return jax.tree.map(lambda _: parallel_jax.SHARE_PARTITION, tree)


def _data_specs(tree):
    return jax.tree.map(lambda _: parallel_jax.DATA_PARTITION, tree)


def _device_put_with_specs(values, specs):
    return jax.device_put(values, parallel_jax.make_sharding(specs))


def _assert_tree_allclose(actual, expected, *, rtol=2e-5, atol=2e-6):
    actual_leaves, actual_structure = jax.tree.flatten(actual)
    expected_leaves, expected_structure = jax.tree.flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


def _per_device_source_sums(device_count: int) -> NQSLITSourceSums:
    index = jnp.arange(device_count, dtype=jnp.float32)
    # Deliberately give each shard a different local overflow-protection scale.
    ratio_scale = 10.0 ** jnp.linspace(-30.0, 30.0, device_count)
    ratio_sum = (1.0 + 0.25j) * (index + 1.0)
    ratio_abs2_sum = 0.5 + 0.2 * index
    psi_weight_sum = 0.8 + 0.3 * index
    psi_weight_sq_sum = 0.4 + 0.1 * index
    psi_log_ratio_abs2_sum = (-0.2 + 0.15 * index) * psi_weight_sum
    return NQSLITSourceSums(
        sample_count=3.0 + index,
        weight_sum=2.0 + 0.5 * index,
        valid_sample_count=2.0 + index,
        ratio_scale=ratio_scale,
        ratio_sum=ratio_sum.astype(jnp.complex64),
        ratio_abs2_sum=ratio_abs2_sum,
        psi_weight_sum=psi_weight_sum,
        psi_weight_sq_sum=psi_weight_sq_sum,
        psi_log_ratio_abs2_sum=psi_log_ratio_abs2_sum,
        response_conj_over_source_sum=(0.2 - 0.3j) * (index + 1.0),
        ground_energy_sum=-1.0 - 0.2 * index,
        response_over_source_moments=WeightedComplexMoments(
            weight_sum=2.0 + 0.5 * index,
            origin=(0.7 - 0.2j) * (index + 1.0),
            mean_offset=jnp.zeros_like(index, dtype=jnp.complex64),
            centered_abs2_sum=0.3 + 0.1 * index,
        ),
        hbar_over_source_moments=WeightedComplexMoments(
            weight_sum=2.0 + 0.5 * index,
            origin=(-0.4 + 0.1j) * (index + 1.0),
            mean_offset=jnp.zeros_like(index, dtype=jnp.complex64),
            centered_abs2_sum=1.2 + 0.4 * index,
        ),
        psi_weight_max=0.25 + 0.05 * index,
    )


def test_merge_source_sums_matches_ordered_serial_merge():
    device_count = jax.local_device_count()
    per_device = _per_device_source_sums(device_count)

    def merge(local_arrays):
        local_sums = jax.tree.map(itemgetter(0), local_arrays)
        return _merge_source_sums_across_devices(local_sums)

    merge_sharded = parallel_jax.jit_sharded(
        merge,
        in_specs=(_data_specs(per_device),),
        out_specs=_replicated_specs(per_device),
    )
    (sharded_per_device,) = _device_put_with_specs(
        (per_device,),
        (_data_specs(per_device),),
    )
    parallel_result = merge_sharded(sharded_per_device)

    serial_shards = [
        jax.tree.map(lambda value, i=i: value[i], per_device)
        for i in range(device_count)
    ]
    serial_result = reduce(_add_source_sums, serial_shards)

    _assert_tree_allclose(parallel_result, serial_result)


def test_regularized_action_gradient_matches_serial_global_batch():
    device_count = jax.local_device_count()
    local_batch = 4
    batch_size = local_batch * device_count
    parameter_count = 5
    key_score, key_ratio, key_weight = jax.random.split(
        jax.random.PRNGKey(921),
        3,
    )
    score = jax.random.normal(key_score, (batch_size, parameter_count)) + 1j * (
        0.4
        * jax.random.normal(
            jax.random.fold_in(key_score, 1),
            (batch_size, parameter_count),
        )
    )
    ratio = jax.random.normal(key_ratio, (batch_size,)) + 1j * (
        0.3 * jax.random.normal(jax.random.fold_in(key_ratio, 1), (batch_size,))
    )
    source_weight = 0.2 + jax.random.uniform(key_weight, (batch_size,))
    # Exercise the global validity mask as well as the global max/sum reductions.
    ratio = ratio.at[local_batch].set(jnp.asarray(jnp.nan + 0.0j, ratio.dtype))
    source_weight = source_weight.at[-1].set(jnp.asarray(-1.0, source_weight.dtype))
    reverse_kl_weight = 0.17
    eps = 1e-7

    serial_result = _regularized_action_gradient(
        score,
        ratio,
        source_weight,
        reverse_kl_weight=reverse_kl_weight,
        eps=eps,
    )

    def distributed(local_score, local_ratio, local_source_weight):
        return _regularized_action_gradient(
            local_score,
            local_ratio,
            local_source_weight,
            reverse_kl_weight=reverse_kl_weight,
            eps=eps,
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )

    distributed_sharded = parallel_jax.jit_sharded(
        distributed,
        in_specs=(
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
        out_specs=(
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
        ),
    )
    sharded_args = _device_put_with_specs(
        (score, ratio, source_weight),
        (
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
            parallel_jax.DATA_PARTITION,
        ),
    )
    parallel_result = distributed_sharded(*sharded_args)

    _assert_tree_allclose(parallel_result, serial_result, rtol=3e-5, atol=3e-6)


def _parallel_sr_solve(
    score,
    grad,
    damping,
    *,
    kernel_null_vectors=None,
):
    device_count = jax.local_device_count()

    if kernel_null_vectors is None:

        def solve(local_score, replicated_grad, replicated_damping):
            return _solve_sr_direction_data_parallel(
                local_score,
                replicated_grad,
                replicated_damping,
                device_count=device_count,
            )

        in_specs = (
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
        )
        args = (score, grad, damping)
    else:

        def solve(
            local_score,
            replicated_grad,
            replicated_damping,
            local_null_vectors,
        ):
            return _solve_sr_direction_data_parallel(
                local_score,
                replicated_grad,
                replicated_damping,
                device_count=device_count,
                local_kernel_null_vectors=local_null_vectors,
            )

        in_specs = (
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            parallel_jax.SHARE_PARTITION,
            PartitionSpec(None, parallel_jax.BATCH_AXIS_NAME),
        )
        args = (score, grad, damping, kernel_null_vectors)

    solve_sharded = parallel_jax.jit_sharded(
        solve,
        in_specs=in_specs,
        out_specs=parallel_jax.SHARE_PARTITION,
    )
    sharded_args = _device_put_with_specs(args, in_specs)
    return solve_sharded(*sharded_args)


def _serial_sr_solve(score, grad, damping, *, kernel_null_vectors=None):
    device_count = jax.local_device_count()
    chunk_rows = (score.shape[0] // device_count,) * device_count

    def score_chunk(index):
        row_count = chunk_rows[index]
        start = index * row_count
        return score[start : start + row_count]

    return _solve_sr_direction_chunked(
        chunk_rows,
        score_chunk,
        grad,
        damping,
        kernel_null_vectors=kernel_null_vectors,
    )


def test_data_parallel_sr_primal_matches_chunked_solver():
    device_count = jax.local_device_count()
    sample_count = 4 * device_count
    parameter_count = 3
    score = jax.random.normal(
        jax.random.PRNGKey(31),
        (sample_count, parameter_count),
    )
    grad = jax.random.normal(jax.random.PRNGKey(32), (parameter_count,))
    damping = jnp.asarray(0.13, dtype=score.dtype)

    parallel_result = _parallel_sr_solve(score, grad, damping)
    serial_result = _serial_sr_solve(score, grad, damping)

    np.testing.assert_allclose(parallel_result, serial_result, rtol=2e-5, atol=2e-6)


def test_data_parallel_sr_dual_with_null_vector_matches_chunked_solver():
    device_count = jax.local_device_count()
    sample_count = 3 * device_count
    parameter_count = sample_count + 5
    score = jax.random.normal(
        jax.random.PRNGKey(51),
        (sample_count, parameter_count),
    )
    grad = jax.random.normal(jax.random.PRNGKey(52), (parameter_count,))
    damping = jnp.asarray(0.21, dtype=score.dtype)
    null_vectors = jnp.stack(
        (
            jnp.ones(sample_count, dtype=score.dtype),
            jnp.linspace(-1.0, 1.0, sample_count, dtype=score.dtype),
        )
    )

    parallel_result = _parallel_sr_solve(
        score,
        grad,
        damping,
        kernel_null_vectors=null_vectors,
    )
    serial_result = _serial_sr_solve(
        score,
        grad,
        damping,
        kernel_null_vectors=null_vectors,
    )

    np.testing.assert_allclose(parallel_result, serial_result, rtol=3e-5, atol=3e-6)


def test_data_parallel_spring_preserves_nonzero_history():
    device_count = jax.local_device_count()
    sample_count = 3 * device_count
    parameter_count = sample_count + 4
    score = jax.random.normal(
        jax.random.PRNGKey(71),
        (sample_count, parameter_count),
    )
    grad = jax.random.normal(jax.random.PRNGKey(72), (parameter_count,))
    previous = 0.1 * jax.random.normal(
        jax.random.PRNGKey(73),
        (parameter_count,),
    )
    state = _SpringState(previous_direction=previous)
    null_vectors = jnp.ones((1, sample_count), dtype=score.dtype)
    epsilon_scale = 0.08
    damping_floor = 1e-4
    decay = 0.85
    qfi_trace = jnp.sum(score**2)
    chunk_rows = (sample_count // device_count,) * device_count

    def score_chunk(index):
        row_count = chunk_rows[index]
        start = index * row_count
        return score[start : start + row_count]

    serial_result = _spring_direction_chunked(
        chunk_rows,
        score_chunk,
        grad,
        state,
        epsilon_scale=epsilon_scale,
        damping_floor=damping_floor,
        decay=decay,
        qfi_trace=qfi_trace,
        kernel_null_vectors=null_vectors,
    )

    def distributed(
        local_score,
        replicated_grad,
        replicated_state,
        replicated_qfi_trace,
        local_null_vectors,
    ):
        return _spring_direction_data_parallel(
            local_score,
            replicated_grad,
            replicated_state,
            epsilon_scale=epsilon_scale,
            damping_floor=damping_floor,
            decay=decay,
            device_count=device_count,
            qfi_trace=replicated_qfi_trace,
            local_kernel_null_vectors=local_null_vectors,
        )

    distributed_sharded = parallel_jax.jit_sharded(
        distributed,
        in_specs=(
            parallel_jax.DATA_PARTITION,
            parallel_jax.SHARE_PARTITION,
            _replicated_specs(state),
            parallel_jax.SHARE_PARTITION,
            PartitionSpec(None, parallel_jax.BATCH_AXIS_NAME),
        ),
        out_specs=(
            parallel_jax.SHARE_PARTITION,
            _replicated_specs(state),
            parallel_jax.SHARE_PARTITION,
        ),
    )
    in_specs = (
        parallel_jax.DATA_PARTITION,
        parallel_jax.SHARE_PARTITION,
        _replicated_specs(state),
        parallel_jax.SHARE_PARTITION,
        PartitionSpec(None, parallel_jax.BATCH_AXIS_NAME),
    )
    parallel_result = distributed_sharded(
        *_device_put_with_specs(
            (score, grad, state, qfi_trace, null_vectors),
            in_specs,
        )
    )

    _assert_tree_allclose(parallel_result, serial_result, rtol=3e-5, atol=3e-6)


def _hydrogen_ground_logpsi(_params, data: MoleculeData):
    return -jnp.linalg.norm(data.electrons[0] - data.atoms[0])


def _hydrogen_2pz_logpsi(_params, data: MoleculeData):
    relative = data.electrons[0] - data.atoms[0]
    sign_phase = jnp.where(relative[2] < 0.0, jnp.pi, 0.0)
    return (
        jnp.log(jnp.abs(relative[2]))
        - 0.5 * jnp.linalg.norm(relative)
        + 1j * sign_phase
    )


def _scaled_hydrogen_response(params, data: MoleculeData):
    return params["scale"] * _hydrogen_2pz_logpsi({}, data)


def _hydrogen_batch(batch_size: int) -> BatchedData[MoleculeData]:
    index = jnp.arange(batch_size, dtype=jnp.float32)
    signs = jnp.where(index % 2 == 0, 1.0, -1.0)
    electrons = jnp.stack(
        (
            signs * (0.11 + 0.017 * index),
            0.07 * jnp.sin(0.4 + index),
            signs * (0.45 + 0.031 * index),
        ),
        axis=1,
    )[:, None, :]
    return BatchedData(
        data=MoleculeData(
            electrons=electrons,
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )


def _lit_workflow(
    data_parallel: str,
    *,
    train_batch_size: int,
    eval_batch_size: int,
) -> MoleculeLITWorkflow:
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=train_batch_size, seed=123)
    workflow.lit_config = MolecularLITConfig(
        eta=0.02,
        nqs_data_parallel=data_parallel,
        nqs_train_update_batch_size=train_batch_size,
        nqs_eval_batch_size=eval_batch_size,
        nqs_source_floor=1e-4,
        nqs_reverse_kl_weight=0.17,
        nqs_learning_rate=0.02,
        nqs_spring_epsilon=0.08,
        nqs_spring_decay=0.85,
        nqs_spring_damping_floor=1e-4,
        nqs_sr_max_norm=None,
        nqs_sr_score_eps=1e-7,
    )
    workflow._nqs_chunk_sums_kernel_cache = {}
    return workflow


def test_data_parallel_batch_must_be_divisible_by_devices(monkeypatch):
    device_count = 8
    monkeypatch.setattr(jax, "local_device_count", lambda: device_count)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=device_count + 1,
        eval_batch_size=device_count + 1,
    )
    batch = _hydrogen_batch(device_count + 1)

    with pytest.raises(ValueError, match="positive multiple"):
        workflow._validate_data_parallel_batch(batch, purpose="training")


@pytest.mark.parametrize("batch_size", [0, 1, 7])
def test_data_parallel_empty_and_small_batches_are_rejected(monkeypatch, batch_size):
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=8,
        eval_batch_size=8,
    )

    with pytest.raises(ValueError, match=rf"batch size {batch_size}.*8 local devices"):
        workflow._validate_data_parallel_batch(
            _hydrogen_batch(batch_size),
            purpose="training",
        )


@pytest.mark.parametrize(
    ("workflow_batch", "train_batch", "eval_batch", "purpose", "batch_size"),
    [
        (0, 0, 8, "training", 1),
        (0, 8, 0, "evaluation", 1),
        (1024, 4, 8, "training", 4),
        (1024, 10, 8, "training", 10),
        (1024, 8, 4, "evaluation", 4),
        (1024, 8, 10, "evaluation", 10),
    ],
)
def test_data_parallel_config_rejects_invalid_effective_batches_before_run(
    monkeypatch,
    workflow_batch,
    train_batch,
    eval_batch,
    purpose,
    batch_size,
):
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=train_batch,
        eval_batch_size=eval_batch,
    )
    workflow.config = SimpleNamespace(batch_size=workflow_batch, seed=123)

    with pytest.raises(
        ValueError,
        match=rf"{purpose} batch size {batch_size}.*8 local devices",
    ):
        workflow._validate_data_parallel_config()


def test_data_parallel_config_accepts_formal_1024_on_eight_devices(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=1024,
        eval_batch_size=4096,
    )

    workflow._validate_data_parallel_config()


def test_per_device_batches_scale_only_in_local_devices_mode(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=0,
        eval_batch_size=0,
    )
    workflow.lit_config.nqs_train_update_batch_size_per_device = 512
    workflow.lit_config.nqs_eval_batch_size_per_device = 256

    assert workflow._nqs_train_update_batch_size() == 4096
    assert workflow._nqs_eval_batch_size() == 2048

    workflow.lit_config.nqs_data_parallel = "off"
    assert workflow._nqs_train_update_batch_size() == 512
    assert workflow._nqs_eval_batch_size() == 256


def test_global_and_per_device_batches_are_mutually_exclusive():
    workflow = _lit_workflow(
        "off",
        train_batch_size=16,
        eval_batch_size=8,
    )
    workflow.lit_config.nqs_train_update_batch_size_per_device = 4

    with pytest.raises(ValueError, match="mutually exclusive"):
        workflow._validate_chunk_config()


def test_yaml_boolean_false_is_normalized_to_off_but_true_is_rejected():
    workflow = _lit_workflow(
        "off",
        train_batch_size=8,
        eval_batch_size=8,
    )
    setattr(workflow.lit_config, "nqs_data_parallel", False)

    workflow._validate_data_parallel_config()
    assert not workflow._nqs_data_parallel_enabled()

    setattr(workflow.lit_config, "nqs_data_parallel", True)
    with pytest.raises(ValueError, match="nqs_data_parallel"):
        workflow._validate_data_parallel_config()


def test_data_parallel_rejects_multiple_jax_processes(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=1024,
        eval_batch_size=4096,
    )

    with pytest.raises(ValueError, match="requires one JAX process"):
        workflow._validate_data_parallel_config()


def test_reused_heldout_pool_with_partial_chunk_is_rejected(monkeypatch):
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    workflow = _lit_workflow(
        "local_devices",
        train_batch_size=16,
        eval_batch_size=16,
    )

    with pytest.raises(
        ValueError,
        match=r"held-out evaluation source pool size 17.*divisible.*16",
    ):
        workflow._validate_source_pool_chunks(
            _hydrogen_batch(32),
            _hydrogen_batch(17),
        )


def test_local_devices_hydrogen_stats_match_serial_including_one_device():
    device_count = jax.local_device_count()
    batch_size = 4 * device_count
    eval_batch_size = 2 * device_count
    batch = _hydrogen_batch(batch_size)
    response_params = {"scale": jnp.asarray(0.93, dtype=jnp.float32)}
    common = dict(
        response_apply=_scaled_hydrogen_response,
        response_params=response_params,
        ground_logpsi=_hydrogen_ground_logpsi,
        ground_params={},
        batched_data=batch,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-0.5,
        omega=jnp.asarray(0.36, dtype=jnp.float32),
    )
    serial_workflow = _lit_workflow(
        "off",
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )
    parallel_workflow = _lit_workflow(
        "local_devices",
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )

    serial_stats = serial_workflow._nqs_stats_chunked(**common)
    parallel_stats = parallel_workflow._nqs_stats_chunked(**common)

    _assert_tree_allclose(parallel_stats, serial_stats, rtol=4e-5, atol=4e-6)


def _dummy_update_carry(batch, params, *, seed: int) -> _NQSUpdateCarry:
    del batch
    flat_params, _ = ravel_pytree(params)
    return _NQSUpdateCarry(
        spring=_SpringState(previous_direction=jnp.zeros_like(flat_params)),
        rng=jax.random.PRNGKey(seed),
    )


def test_local_devices_source_distillation_matches_serial_global_batch():
    device_count = jax.local_device_count()
    batch_size = 4 * device_count
    eval_batch_size = 2 * device_count
    train_pool = _hydrogen_batch(batch_size)
    eval_pool = _hydrogen_batch(batch_size)
    serial_workflow = _lit_workflow(
        "off",
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )
    parallel_workflow = _lit_workflow(
        "local_devices",
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
    )
    for workflow in (serial_workflow, parallel_workflow):
        workflow.lit_config.nqs_source_distillation_iterations = 2
        workflow.lit_config.nqs_selection_interval = 1
        workflow.lit_config.nqs_log_interval = 0
    initial_params = {"scale": jnp.asarray(0.93, dtype=jnp.float32)}

    serial_params, _ = serial_workflow._distill_response_from_source(
        _scaled_hydrogen_response,
        initial_params,
        _hydrogen_ground_logpsi,
        {},
        train_pool,
        eval_pool,
        jax.random.PRNGKey(29),
        axis=2,
        source_center=0.0,
    )
    parallel_params, _ = parallel_workflow._distill_response_from_source(
        _scaled_hydrogen_response,
        initial_params,
        _hydrogen_ground_logpsi,
        {},
        train_pool,
        eval_pool,
        jax.random.PRNGKey(29),
        axis=2,
        source_center=0.0,
    )
    serial_stats = serial_workflow._evaluate_source_distillation(
        _scaled_hydrogen_response,
        serial_params,
        _hydrogen_ground_logpsi,
        {},
        eval_pool,
        axis=2,
        source_center=0.0,
    )
    parallel_stats = parallel_workflow._evaluate_source_distillation(
        _scaled_hydrogen_response,
        parallel_params,
        _hydrogen_ground_logpsi,
        {},
        eval_pool,
        axis=2,
        source_center=0.0,
    )

    _assert_tree_allclose(parallel_params, serial_params, rtol=5e-5, atol=5e-6)
    _assert_tree_allclose(parallel_stats, serial_stats, rtol=5e-5, atol=5e-6)


def test_local_devices_source_updates_match_serial_including_one_device():
    device_count = jax.local_device_count()
    batch_size = 4 * device_count
    batch = _hydrogen_batch(batch_size)
    serial_workflow = _lit_workflow(
        "off",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    parallel_workflow = _lit_workflow(
        "local_devices",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    serial_update = serial_workflow._make_nqs_update_step(
        _scaled_hydrogen_response,
        {},
        _hydrogen_ground_logpsi,
        -0.5,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
    )
    parallel_update = parallel_workflow._make_nqs_update_step(
        _scaled_hydrogen_response,
        {},
        _hydrogen_ground_logpsi,
        -0.5,
        axis=2,
        source_center=0.0,
        source_norm=1.0,
    )
    serial_params = {"scale": jnp.asarray(0.93, dtype=jnp.float32)}
    parallel_params = jax.tree.map(jnp.array, serial_params)
    serial_carry = _dummy_update_carry(batch, serial_params, seed=19)
    parallel_carry = _dummy_update_carry(batch, parallel_params, seed=19)
    omega = jnp.asarray(0.36, dtype=jnp.float32)

    for batch_index in range(3):
        serial_params, serial_stats, serial_carry = serial_update(
            serial_params,
            batch,
            omega,
            serial_carry,
            batch_index=batch_index,
        )
        parallel_params, parallel_stats, parallel_carry = parallel_update(
            parallel_params,
            batch,
            omega,
            parallel_carry,
            batch_index=batch_index,
        )
        _assert_tree_allclose(
            parallel_params,
            serial_params,
            rtol=6e-5,
            atol=6e-6,
        )
        _assert_tree_allclose(
            parallel_stats,
            serial_stats,
            rtol=6e-5,
            atol=6e-6,
        )
        _assert_tree_allclose(
            parallel_carry.spring,
            serial_carry.spring,
            rtol=6e-5,
            atol=6e-6,
        )
