# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jaqmc.response.inversion import (
    _nonnegative_least_squares,
    forward_lit,
    initialize_lit_poles,
    invert_signed_lit,
    lit_block_statistics,
    lit_linear_continuum_kernel,
    lit_pole_kernel,
)
from jaqmc.response.lit import lit_from_poles


def test_analytic_continuum_kernel_resolves_subinterval_lorentzian():
    omega = np.asarray([0.173, 0.5, 0.827])
    eta = np.asarray([1e-3, 3e-4, 1e-3])
    grid = np.asarray([0.0, 1.0])

    kernel = lit_linear_continuum_kernel(omega, eta, grid)
    actual = kernel @ np.ones(grid.size)
    expected = (
        np.arctan((grid[-1] - omega) / eta) - np.arctan((grid[0] - omega) / eta)
    ) / eta

    assert np.all(kernel >= 0.0)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-11)


def test_analytic_continuum_partition_of_unity_on_irregular_grid():
    grid = np.asarray([0.2, 0.23, 0.7, 1.4, 2.0])
    omega = np.asarray([-0.1, 0.225, 0.61, 1.8, 2.3])
    eta = np.asarray([0.03, 1e-4, 0.007, 0.02, 0.05])

    kernel = lit_linear_continuum_kernel(omega, eta, grid)
    actual = np.sum(kernel, axis=-1)
    expected = (
        np.arctan((grid[-1] - omega) / eta) - np.arctan((grid[0] - omega) / eta)
    ) / eta

    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)


def test_nnls_is_invariant_to_tiny_global_problem_scale():
    for scale in (1e-20, 1e-6, 1.0, 1e6, 1e20):
        result = _nonnegative_least_squares(
            np.asarray([[scale]], dtype=np.float64),
            np.asarray([scale], dtype=np.float64),
            tolerance=1e-10,
            max_iterations=None,
        )
        assert result.success
        np.testing.assert_allclose(result.x, [1.0], rtol=2e-14, atol=2e-14)
        assert result.optimality <= 1e-12


def test_nnls_is_invariant_to_independent_positive_column_scaling():
    column_scales = np.asarray([1e-8, 1e8], dtype=np.float64)
    matrix = np.diag(column_scales)
    expected = 1.0 / column_scales
    target = matrix @ expected

    result = _nonnegative_least_squares(
        matrix,
        target,
        tolerance=1e-10,
        max_iterations=None,
    )

    assert result.success
    np.testing.assert_allclose(result.x, expected, rtol=2e-14, atol=0.0)
    assert result.optimality <= 1e-12


def test_pole_kernel_and_forward_lit_match_direct_lorentz_sum():
    omega = np.linspace(0.1, 0.8, 31)
    eta = 0.04
    energies = np.array([0.25, 0.55])
    strengths = np.array([0.7, 0.2])

    kernel = lit_pole_kernel(omega, eta, energies)
    actual = forward_lit(
        omega,
        eta,
        pole_energies=energies,
        pole_strengths=strengths,
    )

    assert kernel.shape == (omega.size, energies.size)
    np.testing.assert_allclose(
        actual,
        lit_from_poles(omega, energies, strengths, eta),
        rtol=1e-13,
        atol=1e-13,
    )


def test_block_statistics_preserve_cross_frequency_covariance_of_mean():
    blocks = np.asarray(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]],
            [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
        ]
    )

    stats = lit_block_statistics(blocks)

    expected = np.stack([np.cov(axis, ddof=1) / blocks.shape[-1] for axis in blocks])
    np.testing.assert_allclose(stats.mean, np.mean(blocks, axis=-1))
    np.testing.assert_allclose(stats.covariance, expected)
    np.testing.assert_allclose(
        stats.standard_error,
        np.sqrt(np.diagonal(expected, axis1=-2, axis2=-1)),
    )
    assert stats.block_count == 4


def test_fixed_poles_are_recovered_from_one_full_rank_eta():
    omega = np.linspace(0.05, 0.75, 101)
    energies = np.array([0.22, 0.48])
    strengths = np.array([0.8, 0.35])
    signed_lit = forward_lit(
        omega,
        0.035,
        pole_energies=energies,
        pole_strengths=strengths,
    )

    result = invert_signed_lit(
        omega,
        0.035,
        signed_lit,
        threshold=0.8,
        pole_energies=energies,
    )

    np.testing.assert_allclose(result.pole_strengths[0], strengths, rtol=1e-11)
    np.testing.assert_allclose(result.fitted_lit[0], signed_lit, rtol=1e-11)
    assert result.diagnostics.solver_success == (True,)
    assert result.diagnostics.unique_eta_count == 1
    assert not result.diagnostics.cross_width_validated
    assert not result.diagnostics.underdetermined
    assert not result.diagnostics.underdetermined_reasons


def test_piecewise_linear_continuum_is_recovered():
    omega = np.linspace(0.2, 1.45, 121)
    grid = np.linspace(0.7, 1.3, 9)
    density = np.exp(-(((grid - 1.0) / 0.18) ** 2))
    signed_lit = forward_lit(
        omega,
        0.045,
        continuum_grid=grid,
        continuum_density=density,
    )

    result = invert_signed_lit(
        omega,
        0.045,
        signed_lit,
        threshold=grid[0],
        continuum_grid=grid,
    )

    kernel = lit_linear_continuum_kernel(omega, 0.045, grid)
    assert kernel.shape == (omega.size, grid.size)
    assert np.all(kernel >= 0)
    np.testing.assert_allclose(result.continuum_density[0], density, rtol=1e-10)
    np.testing.assert_allclose(result.fitted_lit[0], signed_lit, rtol=1e-11)


def test_mixed_multi_eta_axes_can_be_merged_as_flat_observations():
    omega_block = np.linspace(0.08, 1.35, 81)
    omega = np.tile(omega_block, 2)
    eta = np.repeat(np.array([0.03, 0.075]), omega_block.size)
    energies = np.array([0.28, 0.52])
    grid = np.linspace(0.72, 1.32, 8)
    strengths = np.array([[0.8, 0.25], [0.3, 0.65]])
    density = np.array(
        [
            np.linspace(0.05, 0.5, grid.size),
            np.linspace(0.45, 0.1, grid.size),
        ]
    )
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=energies,
        pole_strengths=strengths,
        continuum_grid=grid,
        continuum_density=density,
    )

    result = invert_signed_lit(
        omega,
        eta,
        signed_lit,
        threshold=grid[0],
        pole_energies=energies,
        continuum_grid=grid,
    )

    assert signed_lit.shape == (2, omega.size)
    assert result.fitted_lit.shape == signed_lit.shape
    np.testing.assert_allclose(result.pole_strengths, strengths, rtol=1e-10)
    np.testing.assert_allclose(result.continuum_density, density, rtol=1e-10)
    assert result.diagnostics.unique_eta_count == 2
    assert result.diagnostics.cross_width_validated
    assert not result.diagnostics.underdetermined


def test_covariance_and_standard_deviation_weighted_residual_diagnostics():
    rng = np.random.default_rng(12)
    omega = np.linspace(0.1, 0.65, 45)
    eta = 0.04
    energy = np.array([0.33])
    exact = forward_lit(
        omega,
        eta,
        pole_energies=energy,
        pole_strengths=np.array([0.7]),
    )
    indices = np.arange(omega.size)
    covariance = 2e-5 * 0.55 ** np.abs(indices[:, np.newaxis] - indices)
    covariance_cholesky = np.linalg.cholesky(covariance)
    noisy = exact + covariance_cholesky @ rng.normal(size=omega.size)

    covariance_result = invert_signed_lit(
        omega,
        eta,
        noisy,
        threshold=0.8,
        pole_energies=energy,
        covariance=covariance,
    )
    expected_covariance_norm = np.linalg.norm(
        np.linalg.solve(covariance_cholesky, covariance_result.residual[0])
    )
    np.testing.assert_allclose(
        covariance_result.diagnostics.weighted_residual_norms[0],
        expected_covariance_norm,
        rtol=1e-12,
    )
    assert covariance_result.diagnostics.statistically_weighted
    np.testing.assert_array_equal(
        covariance_result.diagnostics.covariance_effective_ranks,
        [omega.size],
    )
    assert covariance_result.diagnostics.covariance_truncated == (False,)

    standard_deviation = np.linspace(0.002, 0.006, omega.size)
    standard_deviation_result = invert_signed_lit(
        omega,
        eta,
        noisy,
        threshold=0.8,
        pole_energies=energy,
        standard_deviation=standard_deviation,
    )
    expected_standard_deviation_norm = np.linalg.norm(
        standard_deviation_result.residual[0] / standard_deviation
    )
    np.testing.assert_allclose(
        standard_deviation_result.diagnostics.weighted_residual_norms[0],
        expected_standard_deviation_norm,
        rtol=1e-12,
    )
    assert standard_deviation_result.diagnostics.covariance_effective_ranks is None
    assert standard_deviation_result.diagnostics.covariance_truncated is None


def test_singular_empirical_covariance_uses_only_its_effective_subspace():
    rng = np.random.default_rng(29)
    n_observations = 18
    n_blocks = 5
    omega = np.linspace(0.1, 0.7, n_observations)
    eta = np.where(np.arange(n_observations) % 2, 0.035, 0.07)
    energy = np.array([0.36])
    exact = forward_lit(
        omega,
        eta,
        pole_energies=energy,
        pole_strengths=np.array([0.62]),
    )
    block_samples = rng.normal(scale=0.003, size=(n_blocks, n_observations))
    centered_samples = block_samples - np.mean(block_samples, axis=0)
    covariance = centered_samples.T @ centered_samples / (n_blocks - 1)
    noisy = exact + centered_samples[0]

    result = invert_signed_lit(
        omega,
        eta,
        noisy,
        threshold=0.8,
        pole_energies=energy,
        covariance=covariance,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    retained = eigenvalues > 1e-10 * np.max(np.abs(eigenvalues))
    whitener = (
        eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])[np.newaxis, :]
    ).T
    expected_norm = np.linalg.norm(whitener @ result.residual[0])
    np.testing.assert_allclose(
        result.diagnostics.weighted_residual_norms[0],
        expected_norm,
        rtol=1e-11,
    )
    np.testing.assert_array_equal(
        result.diagnostics.covariance_effective_ranks,
        [n_blocks - 1],
    )
    assert result.diagnostics.covariance_truncated == (True,)
    assert result.diagnostics.solver_success == (True,)


def test_negative_noisy_lit_is_not_clipped_but_solution_remains_nonnegative():
    omega = np.linspace(0.1, 1.2, 60)
    grid = np.linspace(0.7, 1.2, 7)
    signed_lit = -np.ones(omega.size)

    result = invert_signed_lit(
        omega,
        0.05,
        signed_lit,
        threshold=grid[0],
        pole_energies=np.array([0.35]),
        continuum_grid=grid,
        continuum_regularization=1e-4,
    )

    assert np.all(result.pole_strengths >= 0)
    assert np.all(result.continuum_density >= 0)
    np.testing.assert_allclose(result.pole_strengths, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.continuum_density, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.residual[0], np.ones(omega.size))


def test_pole_energy_fit_is_shared_across_axes():
    omega_block = np.linspace(0.12, 0.68, 61)
    omega = np.tile(omega_block, 2)
    eta = np.repeat(np.array([0.025, 0.065]), omega_block.size)
    true_energy = np.array([0.371])
    strengths = np.array([[0.8], [0.27], [0.51]])
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=true_energy,
        pole_strengths=strengths,
    )

    result = invert_signed_lit(
        omega,
        eta,
        signed_lit,
        threshold=0.8,
        pole_energies=np.array([0.34]),
        fit_pole_energies=True,
        pole_energy_bounds=np.array([[0.3, 0.44]]),
        pole_fit_tolerance=1e-7,
    )

    np.testing.assert_allclose(result.pole_energies, true_energy, atol=2e-7)
    np.testing.assert_allclose(result.pole_strengths, strengths, rtol=2e-6)
    assert result.pole_strengths.shape == (3, 1)
    assert result.diagnostics.pole_fit_success
    assert result.diagnostics.unique_eta_count == 2


def test_five_ordered_bounded_poles_are_recovered():
    omega_block = np.linspace(0.12, 0.72, 91)
    omega = np.tile(omega_block, 2)
    eta = np.repeat(np.array([0.012, 0.035]), omega_block.size)
    true_energies = np.array([0.20, 0.29, 0.39, 0.50, 0.63])
    strengths = np.array(
        [
            [0.75, 0.52, 0.38, 0.24, 0.15],
            [0.18, 0.33, 0.47, 0.61, 0.74],
        ]
    )
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=true_energies,
        pole_strengths=strengths,
    )
    initial_energies = true_energies + np.array([0.014, -0.012, 0.011, -0.013, 0.012])
    bounds = np.column_stack((true_energies - 0.025, true_energies + 0.025))

    result = invert_signed_lit(
        omega,
        eta,
        signed_lit,
        threshold=0.75,
        pole_energies=initial_energies,
        fit_pole_energies=True,
        pole_energy_bounds=bounds,
        max_fitted_poles=5,
        pole_fit_tolerance=2e-7,
        pole_fit_max_iterations=100,
    )

    assert result.pole_energies.shape == (5,)
    assert np.all(np.diff(result.pole_energies) > 0.0)
    np.testing.assert_allclose(result.pole_energies, true_energies, atol=3e-7)
    np.testing.assert_allclose(result.pole_strengths, strengths, rtol=3e-6)
    assert result.diagnostics.pole_fit_success
    assert result.diagnostics.unique_eta_count == 2


def test_five_poles_are_initialized_blindly_from_single_eta_data():
    omega = np.linspace(0.75, 0.90, 601)
    eta = 0.003
    true_energies = np.array([0.77975, 0.84843, 0.87250, 0.88367, 0.88974])
    strengths = np.array([1.0, 0.25, 0.10, 0.055, 0.035])
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=true_energies,
        pole_strengths=strengths,
    )

    initialization = initialize_lit_poles(
        omega,
        eta,
        signed_lit,
        threshold=0.904,
        pole_count=5,
        candidate_grid_points=601,
    )
    result = invert_signed_lit(
        omega,
        eta,
        signed_lit,
        threshold=0.904,
        pole_energies=initialization.pole_energies,
        fit_pole_energies=True,
        pole_energy_bounds=initialization.pole_energy_bounds,
        max_fitted_poles=5,
        pole_fit_tolerance=1e-9,
        pole_fit_max_iterations=100,
    )

    assert initialization.pole_energies.shape == (5,)
    assert np.all(np.diff(initialization.pole_energy_bounds.ravel()) > 0.0)
    np.testing.assert_allclose(result.pole_energies, true_energies, atol=2e-8)
    np.testing.assert_allclose(result.pole_strengths[0], strengths, rtol=2e-7)
    assert not result.diagnostics.cross_width_validated
    assert not result.diagnostics.underdetermined


def test_curvature_regularization_is_reported_and_preserves_nonnegativity():
    rng = np.random.default_rng(21)
    omega = np.linspace(0.3, 1.4, 80)
    grid = np.linspace(0.7, 1.3, 11)
    density = 0.2 + np.exp(-(((grid - 1.0) / 0.16) ** 2))
    signed_lit = forward_lit(
        omega,
        0.05,
        continuum_grid=grid,
        continuum_density=density,
    )
    signed_lit += rng.normal(scale=0.01, size=omega.size)

    result = invert_signed_lit(
        omega,
        0.05,
        signed_lit,
        threshold=grid[0],
        continuum_grid=grid,
        standard_deviation=0.01,
        continuum_regularization=1e-3,
    )

    assert np.all(result.continuum_density >= 0)
    assert result.diagnostics.regularization_norms[0] > 0
    assert np.isfinite(result.diagnostics.condition_numbers[0])
