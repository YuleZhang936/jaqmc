# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jaqmc.response.spectrum import (
    find_spectrum_peaks,
    lorentzian_spectrum,
    projected_spectrum,
    resolvent,
)


def test_projected_spectrum_recovers_diagonal_poles_and_weights():
    overlap = np.eye(3)
    hamiltonian = np.diag([0.375, 4 / 9, 15 / 32])
    source = np.array([2.0, 0.5, 0.0])

    spectrum = projected_spectrum(overlap, hamiltonian, source)

    np.testing.assert_allclose(spectrum.excitation_energies, [0.375, 4 / 9, 15 / 32])
    np.testing.assert_allclose(spectrum.weights.real, [4.0, 0.25, 0.0])
    np.testing.assert_allclose(spectrum.weights.imag, 0.0, atol=1e-14)


def test_projected_spectrum_is_invariant_to_nonorthogonal_basis_transform():
    poles = np.array([0.375, 4 / 9, 15 / 32])
    source_orthonormal = np.array([1.0, -0.25, 0.5])
    transform = np.array(
        [
            [1.0, 0.3, -0.2],
            [0.2, 1.4, 0.1],
            [-0.1, 0.4, 0.8],
        ]
    )
    overlap = transform.T @ transform
    hamiltonian = transform.T @ np.diag(poles) @ transform
    source = transform.T @ source_orthonormal

    spectrum = projected_spectrum(overlap, hamiltonian, source)

    np.testing.assert_allclose(spectrum.excitation_energies, poles, atol=1e-12)
    np.testing.assert_allclose(
        np.sort(spectrum.weights.real), np.sort(source_orthonormal**2), atol=1e-12
    )


def test_resolvent_matches_lorentzian_spectrum_definition():
    overlap = np.eye(2)
    hamiltonian = np.diag([0.375, 4 / 9])
    source = np.array([1.5, 0.25])
    omega = np.linspace(0.34, 0.47, 40)
    eta = 0.01

    response = resolvent(omega, eta, overlap, hamiltonian, source)
    spectrum = projected_spectrum(overlap, hamiltonian, source)
    broadened = lorentzian_spectrum(
        omega, spectrum.excitation_energies, spectrum.weights, eta
    )

    np.testing.assert_allclose(broadened, -np.imag(response) / np.pi)


def test_find_spectrum_peaks_returns_peak_positions_in_energy_order():
    grid = np.linspace(0.3, 0.5, 2001)
    poles = np.array([0.375, 4 / 9])
    weights = np.array([1.0, 0.4])
    intensity = lorentzian_spectrum(grid, poles, weights, eta=0.001)

    peaks = find_spectrum_peaks(grid, intensity, min_height_fraction=0.05)

    assert len(peaks) == 2
    np.testing.assert_allclose([peak.energy for peak in peaks], poles, atol=2e-5)
