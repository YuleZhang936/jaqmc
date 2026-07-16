# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.sampler.symmetry import make_symmetry_mixture_proposal


def _always_rejected_local_proposal(rngs, x, stddev):
    del rngs, stddev
    return jax.tree.map(lambda value: value + 10.0, x)


def _log_probability(data):
    return -100.0 * jnp.sum(data["electrons"] ** 2, axis=(-2, -1))


def test_separate_symmetry_jump_does_not_bias_gaussian_width_adaptation():
    electrons = {"electrons": jnp.zeros((8, 1, 3))}
    operations = np.asarray([np.eye(3), -np.eye(3)])
    symmetry_proposal = make_symmetry_mixture_proposal(
        operations,
        np.zeros(3),
        1.0,
    )
    baseline = MCMCSampler(
        steps=1,
        initial_width=0.3,
        adapt_frequency=1,
        sampling_proposal=_always_rejected_local_proposal,
    )
    symmetry = MCMCSampler(
        steps=1,
        initial_width=0.3,
        adapt_frequency=1,
        sampling_proposal=_always_rejected_local_proposal,
        global_sampling_proposal=symmetry_proposal,
        global_proposal_interval=1,
    )
    key = jax.random.key(4)

    _, baseline_stats, baseline_state = baseline.step(
        _log_probability,
        electrons,
        baseline.init(electrons, key),
        key,
    )
    _, symmetry_stats, symmetry_state = symmetry.step(
        _log_probability,
        electrons,
        symmetry.init(electrons, key),
        key,
    )

    np.testing.assert_allclose(baseline_stats["pmove"], 0.0)
    np.testing.assert_allclose(symmetry_stats["pmove"], 0.0)
    np.testing.assert_allclose(symmetry_stats["symmetry_pmove"], 1.0)
    np.testing.assert_allclose(symmetry_stats["symmetry_move_active"], 1.0)
    np.testing.assert_allclose(symmetry_state.stddev, baseline_state.stddev)


def test_symmetry_jump_interval_skips_without_changing_sampler_state_layout():
    electrons = {"electrons": jnp.ones((4, 1, 3))}
    operations = np.asarray([np.eye(3), -np.eye(3)])
    sampler = MCMCSampler(
        steps=1,
        sampling_proposal=lambda rngs, x, stddev: x,
        global_sampling_proposal=make_symmetry_mixture_proposal(
            operations,
            np.zeros(3),
            1.0,
        ),
        global_proposal_interval=2,
    )
    key = jax.random.key(8)
    state = sampler.init(electrons, key)

    _, first_stats, state = sampler.step(_log_probability, electrons, state, key)
    _, second_stats, state = sampler.step(_log_probability, electrons, state, key)

    np.testing.assert_allclose(first_stats["symmetry_pmove"], 0.0)
    np.testing.assert_allclose(first_stats["symmetry_move_active"], 0.0)
    assert np.isfinite(float(second_stats["symmetry_pmove"]))
    np.testing.assert_allclose(second_stats["symmetry_move_active"], 1.0)
    assert state._fields == ("stddev", "pmoves", "counter")
