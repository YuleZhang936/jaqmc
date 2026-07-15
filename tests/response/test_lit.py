# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _batched_data_chunks,
    _cyclic_batched_data_chunk,
    _is_better_nqs_checkpoint,
    _lit_omega_grid,
    _regularized_action_gradient,
    _solve_sr_direction_chunked,
    _spring_direction_chunked,
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


@pytest.mark.parametrize("mode", ["auto", "local_devices", "distributed"])
def test_parallel_frequency_modes_are_rejected(mode):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(scan_parallel=mode)

    with pytest.raises(ValueError, match="serial continuation"):
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


def _selection_stats(fidelity):
    return _SelectionStats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(0.0),
        invalid_sample_fraction=jnp.asarray(0.0),
        reweight_ess_fraction=jnp.asarray(1.0),
    )


def test_serial_frequency_optimization_propagates_heldout_best_params():
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_iterations=3,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    starts = []

    def init_carry(_data, rng, params):
        starts.append(float(params))
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

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
        fallback_data=None,
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
