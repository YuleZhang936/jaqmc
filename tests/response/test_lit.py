# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jaqmc.response.inversion import (
    fit_lit_basis_expansion,
    lit_basis_transform,
    lit_response_basis,
)
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


def test_regularized_basis_inversion_refits_synthetic_lit():
    response_omega = np.linspace(0.0, 5.0, 1200)
    omega0 = np.linspace(0.2, 3.5, 24)
    eta = 0.2
    basis = lit_response_basis(
        response_omega,
        threshold=0.0,
        basis_count=3,
        alpha1=1.0,
        alpha2=2.0,
    )
    coefficients = np.array([0.8, 0.25, 0.05])
    lit = lit_basis_transform(omega0, response_omega, basis, eta) @ coefficients

    result = fit_lit_basis_expansion(
        omega0,
        lit,
        eta,
        response_omega=response_omega,
        basis_count=3,
        alpha1_grid=(1.0,),
        alpha2_grid=(2.0,),
        l2_grid=(1e-10,),
    )

    np.testing.assert_allclose(result.fit_lit, lit, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(result.response, basis @ coefficients, rtol=1e-5)
    assert result.chi2 < 1e-10
