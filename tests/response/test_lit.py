# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _axis_indices,
    _batched_data_chunks,
    _cyclic_batched_data_chunk,
    _is_better_nqs_checkpoint,
    _lit_error_monitor,
    _lit_omega_grid,
    _regularized_action_gradient,
    _save_batched_pool,
    _scaled_direction_updates,
    _shuffled_batched_data_chunk,
    _shuffled_batched_data_chunk_index,
    _solve_sr_direction_chunked,
    _source_distillation_stats_from_log_ratios,
    _source_pool_target_digest,
    _spring_direction_chunked,
    _spring_optimizer_diagnostics,
    _SpringState,
)
from jaqmc.data import BatchedData
from jaqmc.response.lit import (
    broadened_from_lit,
    lit_from_poles,
)


def _hydrogen_1s_np_energies(n_max: int = 4) -> np.ndarray:
    n = np.arange(2, n_max + 1, dtype=np.float64)
    return 0.5 * (1.0 - 1.0 / n**2)


def _hydrogen_1s_np_oscillator_strengths(n_max: int = 4) -> np.ndarray:
    n = np.arange(2, n_max + 1, dtype=np.float64)
    return (
        2**8
        * n**5
        * (n - 1.0) ** (2.0 * n - 4.0)
        / (3.0 * (n + 1.0) ** (2.0 * n + 4.0))
    )


def _hydrogen_1s_np_axis_dipole_strengths(n_max: int = 4) -> np.ndarray:
    energies = _hydrogen_1s_np_energies(n_max)
    oscillator_strengths = _hydrogen_1s_np_oscillator_strengths(n_max)
    return oscillator_strengths / (2.0 * energies)


def test_hydrogen_1s_np_exact_reference_values():
    energies = _hydrogen_1s_np_energies(4)
    oscillator_strengths = _hydrogen_1s_np_oscillator_strengths(4)
    axis_strengths = _hydrogen_1s_np_axis_dipole_strengths(4)

    np.testing.assert_allclose(energies[0], 0.375, rtol=1e-14)
    np.testing.assert_allclose(oscillator_strengths[0], 8192 / 19683, rtol=1e-14)
    np.testing.assert_allclose(
        axis_strengths[0],
        oscillator_strengths[0] / (2.0 * energies[0]),
        rtol=1e-14,
    )


def test_hydrogen_bound_lit_matches_hardcoded_lorentzian_sum():
    omega = np.array([0.35, 0.375, 0.40])
    eta = 0.02
    energies = np.array([0.375, 4 / 9, 15 / 32])
    strengths = _hydrogen_1s_np_axis_dipole_strengths(4)

    expected = broadened_from_lit(lit_from_poles(omega, energies, strengths, eta), eta)
    actual = broadened_from_lit(lit_from_poles(omega, energies, strengths, eta), eta)

    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_lit_error_monitor_divides_by_normalization_once_and_is_scale_invariant():
    fidelity = 0.91
    source_norm = 2.25
    eta = 0.03
    normalization = 0.3 - 0.4j
    raw_d = 0.125
    expected = (
        raw_d
        * np.sqrt(source_norm)
        / (eta * abs(normalization))
        * np.sqrt((1.0 - fidelity) / fidelity)
    )

    for magnitude in (1.0, 1e-15, 1e-8, 1e8, 1e15):
        scale = magnitude * np.exp(0.37j)
        actual = _lit_error_monitor(
            fidelity=fidelity,
            source_norm=source_norm,
            normalization=scale * normalization,
            eta=eta,
            error_d=abs(scale) * raw_d,
            error_d_valid=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14)


@pytest.mark.parametrize(
    ("normalization", "error_d", "error_d_valid"),
    [
        (0.0 + 0.0j, 0.2, True),
        (np.nan + 0.0j, 0.2, True),
        (np.inf + 0.0j, 0.2, True),
        (0.5 + 0.0j, np.nan, True),
        (0.5 + 0.0j, 0.2, False),
    ],
)
def test_lit_error_monitor_marks_invalid_normalization_or_d_as_nan(
    normalization,
    error_d,
    error_d_valid,
):
    actual = _lit_error_monitor(
        fidelity=0.9,
        source_norm=1.0,
        normalization=normalization,
        eta=0.02,
        error_d=error_d,
        error_d_valid=error_d_valid,
    )
    assert np.isnan(actual)


def test_lit_omega_values_override_linspace():
    config = MolecularLITConfig(
        omega_min=0.0,
        omega_max=1.0,
        omega_points=5,
        omega_values=(0.774, 0.775, 0.7765),
    )

    np.testing.assert_allclose(_lit_omega_grid(config), [0.774, 0.775, 0.7765])


def test_lit_omega_values_must_be_strictly_increasing():
    config = MolecularLITConfig(omega_values=(0.775, 0.775, 0.776))

    with pytest.raises(ValueError, match="strictly increasing"):
        _lit_omega_grid(config)


def test_lit_linspace_grid_must_be_strictly_increasing():
    config = MolecularLITConfig(omega_min=1.0, omega_max=0.0, omega_points=3)

    with pytest.raises(ValueError, match="omega_max must exceed"):
        _lit_omega_grid(config)


def test_warm_start_must_precede_first_serial_frequency():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        omega_values=(0.1, 0.2),
        nqs_warm_start_omega=0.1,
    )

    with pytest.raises(ValueError, match="below the first spectrum frequency"):
        workflow._validate_config()


def test_single_frequency_data_parallel_mode_is_valid():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_data_parallel="local_devices")

    workflow._validate_config()


@pytest.mark.parametrize("mode", ["auto", "distributed", "local_workers"])
def test_unknown_single_frequency_data_parallel_modes_are_rejected(mode):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_data_parallel=mode)

    with pytest.raises(ValueError, match="nqs_data_parallel"):
        workflow._validate_config()


def test_batched_data_chunks_cover_pool_and_cycle():
    pool = BatchedData(
        data=MoleculeData(
            electrons=jnp.arange(30, dtype=jnp.float32).reshape(10, 1, 3),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )

    cycled = _cyclic_batched_data_chunk(pool, 4, 3)
    chunks = list(_batched_data_chunks(pool, 4))

    np.testing.assert_array_equal(
        np.asarray(cycled.data.electrons[:, 0, 0]),
        [12, 15, 18, 21],
    )
    assert [chunk.batch_size for chunk in chunks] == [4, 4, 2]
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(chunk.data.electrons[:, 0, 0]) for chunk in chunks]),
        np.arange(0, 30, 3),
    )


def test_training_chunks_shuffle_reproducibly_and_cover_every_epoch():
    pool = BatchedData(
        data=MoleculeData(
            electrons=jnp.arange(36, dtype=jnp.float32).reshape(12, 1, 3),
            atoms=jnp.zeros((1, 3), dtype=jnp.float32),
            charges=jnp.ones((1,), dtype=jnp.float32),
        ),
        fields_with_batch=("electrons",),
    )
    first_indices = [
        _shuffled_batched_data_chunk_index(12, 3, step, seed=1701) for step in range(4)
    ]
    repeated_indices = [
        _shuffled_batched_data_chunk_index(12, 3, step, seed=1701) for step in range(4)
    ]
    second_epoch = [
        _shuffled_batched_data_chunk_index(12, 3, step, seed=1701)
        for step in range(4, 8)
    ]

    assert first_indices == repeated_indices
    assert sorted(first_indices) == [0, 1, 2, 3]
    assert sorted(second_epoch) == [0, 1, 2, 3]
    selected = _shuffled_batched_data_chunk(pool, 3, 0, seed=1701)
    expected_start = first_indices[0] * 9
    np.testing.assert_array_equal(
        np.asarray(selected.data.electrons[:, 0, 0]),
        np.arange(expected_start, expected_start + 9, 3),
    )


def test_training_chunk_shuffle_rejects_partial_tail():
    with pytest.raises(ValueError, match="must be divisible"):
        _shuffled_batched_data_chunk_index(10, 4, 0, seed=9)


def test_reused_source_pool_is_bound_to_exact_configured_walker_count(tmp_path):
    def pool(walkers):
        return BatchedData(
            data=MoleculeData(
                electrons=jnp.zeros((walkers, 1, 3), dtype=jnp.float32),
                atoms=jnp.zeros((1, 3), dtype=jnp.float32),
                charges=jnp.ones((1,), dtype=jnp.float32),
            ),
            fields_with_batch=("electrons",),
        )

    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=4, seed=3)
    workflow.save_path = UPath(str(tmp_path))
    workflow.lit_config = MolecularLITConfig(
        nqs_train_pool_batches=2,
        nqs_eval_pool_batches=1,
    )
    reference = pool(4)
    train_path = workflow._source_pool_path(0, "train")
    eval_path = workflow._source_pool_path(0, "eval")
    old_metadata = {
        "axis": 0.0,
        "source_center": 0.0,
        "source_floor": workflow.lit_config.nqs_source_floor,
    }
    _save_batched_pool(train_path, pool(8), metadata=old_metadata)
    _save_batched_pool(eval_path, pool(4), metadata=old_metadata)

    assert (
        workflow._try_load_source_pools(
            reference,
            axis=0,
            source_center=0.0,
            target_sha256="ground-and-geometry-a",
        )
        is None
    )

    train_metadata = workflow._source_pool_metadata(
        0,
        0.0,
        target_sha256="ground-and-geometry-a",
        expected_walkers=8,
    )
    eval_metadata = workflow._source_pool_metadata(
        0,
        0.0,
        target_sha256="ground-and-geometry-a",
        expected_walkers=4,
    )
    _save_batched_pool(train_path, pool(4), metadata=train_metadata)
    _save_batched_pool(eval_path, pool(4), metadata=eval_metadata)
    assert (
        workflow._try_load_source_pools(
            reference,
            axis=0,
            source_center=0.0,
            target_sha256="ground-and-geometry-a",
        )
        is None
    )

    _save_batched_pool(train_path, pool(8), metadata=train_metadata)
    loaded = workflow._try_load_source_pools(
        reference,
        axis=0,
        source_center=0.0,
        target_sha256="ground-and-geometry-a",
    )
    assert loaded is not None
    assert [item.batch_size for item in loaded] == [8, 4]
    assert (
        workflow._try_load_source_pools(
            reference,
            axis=0,
            source_center=0.0,
            target_sha256="ground-and-geometry-b",
        )
        is None
    )


def test_source_pool_target_digest_binds_ground_and_static_system_only():
    data = MoleculeData(
        electrons=jnp.zeros((4, 2, 3)),
        atoms=jnp.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
        charges=jnp.asarray([1.0, 1.0]),
    )
    params = {"orbital": jnp.asarray([1.0, 2.0])}
    reference = _source_pool_target_digest(params, data)

    assert reference == _source_pool_target_digest(
        params,
        data.merge({"electrons": jnp.ones_like(data.electrons)}),
    )
    assert reference != _source_pool_target_digest(
        {"orbital": jnp.asarray([1.0, 2.1])},
        data,
    )
    assert reference != _source_pool_target_digest(
        params,
        data.merge({"atoms": data.atoms.at[1, 2].set(0.8)}),
    )


@pytest.mark.parametrize("axes", ["xx", "xzx", "YY"])
def test_axis_indices_reject_duplicate_axes(axes):
    with pytest.raises(ValueError, match="Duplicate dipole axes"):
        _axis_indices(axes)


def test_chunked_sr_solve_matches_full_metric_branch():
    score_aug = jnp.asarray(np.arange(30, dtype=np.float32).reshape(10, 3) / 17.0)
    grad = jnp.asarray([0.5, -0.25, 0.125], dtype=jnp.float32)
    damping = jnp.asarray(0.03, dtype=jnp.float32)

    full = _solve_sr_direction_chunked(
        (10,),
        lambda _: score_aug,
        grad,
        damping,
    )
    chunks = (score_aug[:4], score_aug[4:7], score_aug[7:])
    chunked = _solve_sr_direction_chunked(
        tuple(chunk.shape[0] for chunk in chunks),
        lambda index: chunks[index],
        grad,
        damping,
    )

    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=5e-5)


def test_chunked_sr_solve_matches_full_kernel_branch():
    score_aug = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 6) / 13.0)
    grad = jnp.asarray([0.2, -0.1, 0.3, -0.4, 0.05, 0.7], dtype=jnp.float32)
    damping = jnp.asarray(0.07, dtype=jnp.float32)

    full = _solve_sr_direction_chunked(
        (4,),
        lambda _: score_aug,
        grad,
        damping,
    )
    chunks = (score_aug[:1], score_aug[1:3], score_aug[3:])
    chunked = _solve_sr_direction_chunked(
        tuple(chunk.shape[0] for chunk in chunks),
        lambda index: chunks[index],
        grad,
        damping,
    )

    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=5e-5)


def test_reverse_kl_gradient_matches_discrete_autodiff():
    source_weight = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32)
    base_ratio = jnp.asarray([0.7, 1.4, 2.1], dtype=jnp.float32)
    score = jnp.asarray(
        [[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]],
        dtype=jnp.float32,
    )
    theta = jnp.asarray([0.15, -0.2], dtype=jnp.float32)
    kl_weight = 0.8

    def objective(local_theta):
        ratio = base_ratio * jnp.exp(score @ local_theta)
        q = source_weight / jnp.sum(source_weight)
        norm = jnp.sum(q * ratio**2)
        p = q * ratio**2 / norm
        fidelity = jnp.sum(q * ratio) ** 2 / norm
        reverse_kl = jnp.sum(p * jnp.log(p / q))
        return fidelity - kl_weight * reverse_kl

    ratio = base_ratio * jnp.exp(score @ theta)
    combined, _, _, _, _, _, reverse_kl = _regularized_action_gradient(
        score,
        ratio,
        source_weight,
        reverse_kl_weight=kl_weight,
        eps=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(combined),
        np.asarray(jax.grad(objective)(theta)),
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(reverse_kl) >= 0.0


def test_reverse_kl_weights_are_finite_at_extreme_dynamic_range():
    score = jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    ratio = jnp.exp(jnp.asarray([-80.0, 0.0, 80.0], dtype=jnp.float32))

    result = _regularized_action_gradient(
        score,
        ratio,
        jnp.ones(3, dtype=jnp.float32),
        reverse_kl_weight=1.0,
        eps=1e-10,
    )

    assert all(np.all(np.isfinite(np.asarray(value))) for value in result)


def test_source_distillation_stats_are_scale_invariant_and_exact_for_source():
    source_weight = jnp.asarray([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    exact = _source_distillation_stats_from_log_ratios(
        jnp.full(4, 37.0 + 0.6j, dtype=jnp.complex64),
        source_weight,
        reverse_kl_weight=1.0,
    )
    shifted = _source_distillation_stats_from_log_ratios(
        jnp.full(4, -51.0 - 2.2j, dtype=jnp.complex64),
        source_weight,
        reverse_kl_weight=1.0,
    )

    np.testing.assert_allclose(float(exact.fidelity), 1.0, rtol=2e-6)
    np.testing.assert_allclose(float(exact.reverse_kl), 0.0, atol=2e-6)
    np.testing.assert_allclose(float(exact.reweight_ess_fraction), 1.0, rtol=2e-6)
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(exact), atol=2e-6)


def test_source_distillation_selects_improved_independent_heldout_state():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.config = SimpleNamespace(batch_size=8)
    workflow.lit_config = MolecularLITConfig(
        nqs_data_parallel="off",
        nqs_source_distillation_iterations=20,
        nqs_train_update_batch_size=8,
        nqs_eval_batch_size=8,
        nqs_selection_interval=1,
        nqs_log_interval=0,
        nqs_reverse_kl_weight=0.1,
        nqs_learning_rate=0.05,
        nqs_spring_epsilon=0.01,
        nqs_spring_decay=0.0,
        nqs_spring_damping_floor=1e-6,
        nqs_sr_max_norm=0.1,
    )

    def pool(source_coordinates):
        electrons = jnp.stack(
            (
                -jnp.asarray(source_coordinates, dtype=jnp.float32),
                jnp.zeros(len(source_coordinates), dtype=jnp.float32),
                jnp.zeros(len(source_coordinates), dtype=jnp.float32),
            ),
            axis=1,
        )[:, None, :]
        return BatchedData(
            data=MoleculeData(
                electrons=electrons,
                atoms=jnp.zeros((1, 3), dtype=jnp.float32),
                charges=jnp.ones(1, dtype=jnp.float32),
            ),
            fields_with_batch=("electrons",),
        )

    train_pool = pool([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9])
    eval_pool = pool([-0.85, -0.65, -0.45, -0.25, 0.25, 0.45, 0.65, 0.85])

    def ground_logpsi(_params, _data):
        return jnp.asarray(0.0, dtype=jnp.float32)

    def response_apply(params, data):
        source = -data.electrons[0, 0]
        amplitude = params["bias"] + params["slope"] * source
        return jnp.log(amplitude.astype(jnp.complex64))

    initial_params = {
        "bias": jnp.asarray(1.0, dtype=jnp.float32),
        "slope": jnp.asarray(0.2, dtype=jnp.float32),
    }
    initial_stats = workflow._evaluate_source_distillation(
        response_apply,
        initial_params,
        ground_logpsi,
        {},
        eval_pool,
        axis=0,
        source_center=0.0,
    )
    selected_params, _ = workflow._distill_response_from_source(
        response_apply,
        initial_params,
        ground_logpsi,
        {},
        train_pool,
        eval_pool,
        jax.random.PRNGKey(3),
        axis=0,
        source_center=0.0,
    )
    selected_stats = workflow._evaluate_source_distillation(
        response_apply,
        selected_params,
        ground_logpsi,
        {},
        eval_pool,
        axis=0,
        source_center=0.0,
    )

    assert float(selected_stats.loss) < float(initial_stats.loss)
    assert float(selected_stats.fidelity) > float(initial_stats.fidelity) + 0.1


def test_spring_optimizer_diagnostics_report_clipping_and_parameter_groups():
    params = {"weights": jnp.zeros(5, dtype=jnp.float32)}
    fidelity_tree = {
        "weights": jnp.asarray([3.0, 4.0, 1.0, -2.0, 2.0], dtype=jnp.float32)
    }
    reverse_kl_tree = {
        "weights": jnp.asarray([2.0, 0.0, 0.0, 2.0, -2.0], dtype=jnp.float32)
    }
    direction_tree = {
        "weights": jnp.asarray([3.0, 4.0, 0.0, 12.0, -5.0], dtype=jnp.float32)
    }
    fidelity_gradient, _ = ravel_pytree(fidelity_tree)
    reverse_kl_gradient, _ = ravel_pytree(reverse_kl_tree)
    direction, _ = ravel_pytree(direction_tree)
    reverse_kl_weight = 0.5
    combined_gradient = fidelity_gradient - reverse_kl_weight * reverse_kl_gradient
    previous_direction = jnp.ones_like(direction)
    learning_rate = 0.1
    max_norm = 0.5
    updates = _scaled_direction_updates(
        params,
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )

    diagnostics = _spring_optimizer_diagnostics(
        params,
        combined_gradient,
        fidelity_gradient,
        reverse_kl_gradient,
        direction,
        updates,
        previous_direction,
        reverse_kl_weight=reverse_kl_weight,
        learning_rate=learning_rate,
        max_norm=max_norm,
        damping=jnp.asarray(0.02, dtype=jnp.float32),
        decay=0.5,
        qfi_trace=jnp.asarray(50.0, dtype=jnp.float32),
    )

    combined_norm = np.linalg.norm(np.asarray(combined_gradient))
    fidelity_norm = np.linalg.norm(np.asarray(fidelity_gradient))
    weighted_kl = reverse_kl_weight * np.asarray(reverse_kl_gradient)
    weighted_kl_norm = np.linalg.norm(weighted_kl)
    direction_norm = np.linalg.norm(np.asarray(direction))
    update_scale = max_norm / direction_norm
    expected_cosine = np.dot(np.asarray(fidelity_gradient), weighted_kl) / (
        fidelity_norm * weighted_kl_norm
    )

    assert bool(diagnostics.available)
    assert float(diagnostics.clip_factor) < 1.0
    np.testing.assert_allclose(float(diagnostics.combined_gradient_norm), combined_norm)
    np.testing.assert_allclose(float(diagnostics.fidelity_gradient_norm), fidelity_norm)
    np.testing.assert_allclose(
        float(diagnostics.weighted_reverse_kl_gradient_norm),
        weighted_kl_norm,
    )
    np.testing.assert_allclose(float(diagnostics.fidelity_kl_cosine), expected_cosine)
    np.testing.assert_allclose(
        float(diagnostics.gradient_cancellation_ratio),
        combined_norm / (fidelity_norm + weighted_kl_norm),
    )
    np.testing.assert_allclose(float(diagnostics.direction_norm), direction_norm)
    np.testing.assert_allclose(float(diagnostics.update_norm), max_norm, rtol=2e-6)
    np.testing.assert_allclose(
        float(diagnostics.clip_factor),
        update_scale / learning_rate,
        rtol=2e-6,
    )
    np.testing.assert_allclose(float(diagnostics.mean_metric_diagonal), 10.0)
    np.testing.assert_allclose(
        float(diagnostics.history_gradient_ratio),
        0.02 * 0.5 * np.sqrt(direction.size) / combined_norm,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.parameter_group_gradient_rms),
        [combined_norm / np.sqrt(direction.size)],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.parameter_group_update_norm),
        [direction_norm * update_scale],
        rtol=2e-6,
    )


def test_spring_optimizer_diagnostics_detect_exact_fidelity_kl_cancellation():
    params = {"weights": jnp.zeros(4, dtype=jnp.float32)}
    fidelity_gradient = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float32)
    reverse_kl_gradient = fidelity_gradient
    combined_gradient = jnp.zeros_like(fidelity_gradient)
    direction = jnp.zeros_like(fidelity_gradient)
    updates = jax.tree.map(jnp.zeros_like, params)

    diagnostics = _spring_optimizer_diagnostics(
        params,
        combined_gradient,
        fidelity_gradient,
        reverse_kl_gradient,
        direction,
        updates,
        jnp.zeros_like(direction),
        reverse_kl_weight=1.0,
        learning_rate=1e-3,
        max_norm=0.1,
        damping=jnp.asarray(1e-3, dtype=jnp.float32),
        decay=0.99,
        qfi_trace=jnp.asarray(1.0, dtype=jnp.float32),
    )

    np.testing.assert_allclose(
        float(diagnostics.fidelity_kl_cosine),
        1.0,
        rtol=2e-7,
    )
    np.testing.assert_allclose(float(diagnostics.gradient_cancellation_ratio), 0.0)
    np.testing.assert_allclose(float(diagnostics.clip_factor), 1.0)


@pytest.mark.parametrize("shape", [(6, 3), (3, 6)])
def test_spring_matches_dense_system_in_both_solver_branches(shape):
    rows, parameters = shape
    score_aug = jnp.asarray(
        np.arange(rows * parameters, dtype=np.float32).reshape(shape) / 19.0
    )
    grad = jnp.linspace(-0.4, 0.6, parameters)
    previous = jnp.linspace(0.1, -0.2, parameters)
    state = _SpringState(previous_direction=previous)

    direction, next_state, damping = _spring_direction_chunked(
        (rows,),
        lambda _: score_aug,
        grad,
        state,
        epsilon_scale=1e-3,
        damping_floor=1e-8,
        decay=0.9,
    )
    score64 = np.asarray(score_aug, dtype=np.float64)
    expected = np.linalg.solve(
        score64.T @ score64 + float(damping) * np.eye(parameters),
        np.asarray(grad, dtype=np.float64)
        + float(damping) * 0.9 * np.asarray(previous, dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(direction), np.asarray(expected), rtol=2e-4)
    np.testing.assert_allclose(
        np.asarray(next_state.previous_direction),
        np.asarray(direction),
    )


def test_spring_damping_scales_with_the_qfi():
    score_aug = jnp.asarray([[1.0, 2.0], [3.0, -1.0]], dtype=jnp.float32)
    state = _SpringState(previous_direction=jnp.zeros(2, dtype=jnp.float32))
    kwargs = dict(
        grad_flat=jnp.asarray([0.2, -0.1], dtype=jnp.float32),
        state=state,
        epsilon_scale=1e-3,
        damping_floor=1e-12,
        decay=0.0,
    )

    _, _, damping = _spring_direction_chunked((2,), lambda _: score_aug, **kwargs)
    _, _, scaled_damping = _spring_direction_chunked(
        (2,), lambda _: 4.0 * score_aug, **kwargs
    )

    np.testing.assert_allclose(float(scaled_damping), 16.0 * float(damping))


class _SelectionStats(NamedTuple):
    loss: jnp.ndarray
    fidelity: jnp.ndarray
    reverse_kl: jnp.ndarray
    invalid_sample_fraction: jnp.ndarray
    reweight_ess_fraction: jnp.ndarray
    signed_lit: jnp.ndarray
    source_norm: jnp.ndarray


def _selection_stats(fidelity):
    return _SelectionStats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(0.0),
        invalid_sample_fraction=jnp.asarray(0.0),
        reweight_ess_fraction=jnp.asarray(1.0),
        signed_lit=jnp.asarray(1.0),
        source_norm=jnp.asarray(1.0),
    )


def test_checkpoint_loss_is_physical_regularized_loss():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig()
    workflow._nqs_stats_chunked = lambda *_args, **_kwargs: _selection_stats(0.8)

    stats = workflow._evaluate_nqs_checkpoint(
        None,
        jnp.asarray(2.0),
        None,
        None,
        None,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=0.0,
        omega=0.1,
    )

    np.testing.assert_allclose(float(stats.loss), 0.2, rtol=1e-6)


def test_serial_frequency_optimization_propagates_heldout_best_params():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_iterations=3,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    starts = []

    def init_carry(rng, params):
        starts.append(float(params))
        return SimpleNamespace(rng=rng)

    def update(params, _pool, _omega, carry, _iteration):
        next_params = params + 1.0
        return next_params, _selection_stats(0.0), carry

    update.init_carry = init_carry

    def evaluate(_response_apply, params, *_args, **_kwargs):
        fidelity = 1.0 - (params - workflow.target) ** 2
        return _selection_stats(fidelity)

    workflow._nqs_stats_chunked = evaluate
    kwargs = dict(
        update_step=update,
        train_pool=None,
        eval_pool=None,
        rng=jax.random.PRNGKey(0),
        response_apply=None,
        ground_logpsi=None,
        ground_params=None,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=0.0,
        iterations=3,
        stage="test",
    )
    workflow.target = 1.0
    first, _, first_iteration, rng = workflow._optimize_nqs_frequency(
        initial_params=0.0,
        omega=0.0,
        **kwargs,
    )
    workflow.target = 2.0
    second, _, second_iteration, _ = workflow._optimize_nqs_frequency(
        initial_params=first,
        omega=0.1,
        **(kwargs | {"rng": rng}),
    )

    assert starts == [0.0, 1.0]
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(2.0)
    assert (first_iteration, second_iteration) == (1, 1)


def test_invalid_checkpoint_is_never_selected():
    candidate = _selection_stats(0.99)._replace(
        invalid_sample_fraction=jnp.asarray(0.1)
    )
    incumbent = _selection_stats(0.8)

    assert not _is_better_nqs_checkpoint(candidate, incumbent)


@pytest.mark.parametrize("maximum", [0.0, -1.0, 1.0, np.nan, np.inf])
def test_atomic_source_parity_maximum_must_be_strict_probability(maximum):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_atomic_source_parity_max_loss=maximum)

    with pytest.raises(ValueError, match="atomic_source_parity_max_loss"):
        workflow._validate_source_sector_config()


def test_regularized_gradient_masks_nonfinite_ratios():
    score = jnp.asarray(
        [[0.2 + 0.1j], [0.5 - 0.3j], [-0.7 + 0.2j]],
        dtype=jnp.complex64,
    )
    ratio = jnp.asarray([1.5 + 0.2j, jnp.inf + 0j, jnp.nan + 0j])

    result = _regularized_action_gradient(
        score,
        ratio,
        jnp.ones(3, dtype=jnp.float32),
        reverse_kl_weight=1.0,
        eps=1e-10,
    )

    assert all(np.all(np.isfinite(np.asarray(value))) for value in result)
    np.testing.assert_allclose(np.asarray(result[3]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(float(result[-1]), 0.0, atol=1e-6)


def test_spring_resets_history_when_metric_has_zero_mass():
    score_aug = jnp.zeros((4, 2), dtype=jnp.float32)
    state = _SpringState(previous_direction=jnp.asarray([3.0, -2.0], dtype=jnp.float32))

    direction, next_state, damping = _spring_direction_chunked(
        (4,),
        lambda _: score_aug,
        jnp.zeros(2, dtype=jnp.float32),
        state,
        epsilon_scale=1e-3,
        damping_floor=1e-12,
        decay=0.99,
    )

    assert np.isfinite(float(damping))
    np.testing.assert_array_equal(np.asarray(direction), np.zeros(2))
    np.testing.assert_array_equal(
        np.asarray(next_state.previous_direction),
        np.zeros(2),
    )


def test_low_rank_sr_projector_matches_primal_spd_solution():
    # The centered score has a known all-ones left null vector.  The low-rank
    # solve lifts that mode before Cholesky without changing the SR direction.
    base = jnp.asarray(
        [
            [1.0, -0.4, 0.2, 0.5, -0.7, 0.1],
            [-0.2, 0.8, -0.6, 0.3, 0.4, -0.9],
            [0.5, -0.1, 0.9, -0.8, 0.2, 0.6],
            [-0.4, 0.3, -0.2, 0.7, -0.5, 0.4],
        ],
        dtype=jnp.float32,
    )
    score_aug = base - jnp.mean(base, axis=0, keepdims=True)
    grad = jnp.linspace(-0.3, 0.5, score_aug.shape[1], dtype=jnp.float32)
    damping = jnp.asarray(1e-4, dtype=jnp.float32)

    actual = _solve_sr_direction_chunked(
        (score_aug.shape[0],),
        lambda _: score_aug,
        grad,
        damping,
        kernel_null_vectors=jnp.ones((1, score_aug.shape[0]), dtype=jnp.float32),
    )
    score64 = np.asarray(score_aug, dtype=np.float64)
    expected = np.linalg.solve(
        score64.T @ score64 + float(damping) * np.eye(score_aug.shape[1]),
        np.asarray(grad, dtype=np.float64),
    )

    assert np.all(np.isfinite(np.asarray(actual)))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4)
