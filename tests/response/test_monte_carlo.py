# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from jaqmc.response.monte_carlo import estimate_weak_matrices, systematic_resample


def test_estimate_weak_matrices_matches_uniform_quadrature():
    n_samples = 20_000
    x = (np.arange(n_samples) + 0.5) / n_samples
    values = np.stack([np.ones_like(x), x], axis=1)
    gradients = np.zeros((n_samples, 2, 1))
    gradients[:, 1, 0] = 1
    source = x
    potential_shift = np.zeros_like(x)
    sampling_density = np.ones_like(x)

    estimate = estimate_weak_matrices(
        values, gradients, source, potential_shift, sampling_density
    )

    np.testing.assert_allclose(
        estimate.overlap.real,
        [[1.0, 0.5], [0.5, 1 / 3]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        estimate.hamiltonian.real,
        [[0.0, 0.0], [0.0, 0.5]],
        atol=1e-12,
    )
    np.testing.assert_allclose(estimate.source[:, 0].real, [0.5, 1 / 3], atol=1e-9)


def test_systematic_resample_is_deterministic_and_low_variance():
    probabilities = np.asarray([0.1, 0.2, 0.7])

    indices_a = systematic_resample(probabilities, 10, seed=7)
    indices_b = systematic_resample(probabilities, 10, seed=7)

    np.testing.assert_array_equal(indices_a, indices_b)
    np.testing.assert_array_equal(np.bincount(indices_a, minlength=3), [1, 2, 7])
