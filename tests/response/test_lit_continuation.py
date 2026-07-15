# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import NamedTuple

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc.app.molecule.lit_workflow import (
    MolecularLITConfig,
    MoleculeLITWorkflow,
    _continuation_min_step,
    _physics_continuation_step,
)


class _BridgeStats(NamedTuple):
    loss: jax.Array
    fidelity: jax.Array
    reverse_kl: jax.Array
    invalid_sample_fraction: jax.Array
    reweight_ess_fraction: jax.Array
    lit: jax.Array
    source_norm: jax.Array


def _bridge_stats(fidelity, *, invalid=0.0, lit=1.0, source_norm=1.0):
    return _BridgeStats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(0.0),
        invalid_sample_fraction=jnp.asarray(invalid),
        reweight_ess_fraction=jnp.asarray(1.0),
        lit=jnp.asarray(lit),
        source_norm=jnp.asarray(source_norm),
    )


def _mock_bridge_workflow(config):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = config
    optimized = []
    starts = []
    probes = []

    def init_carry(_data, rng, params):
        starts.append(float(params))
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

    def update(_params, _pool, omega, carry, _iteration):
        optimized.append(float(omega))
        return omega, _bridge_stats(1.0), carry

    update.init_carry = init_carry

    def evaluate(_response_apply, params, *_args, **kwargs):
        omega = float(kwargs["omega"])
        probes.append(omega)
        gap = abs(float(params) - omega)
        return _bridge_stats(np.exp(-0.4 * gap))

    workflow._nqs_stats_chunked = evaluate
    return workflow, update, starts, optimized, probes


def _run_bridge(workflow, update, *, target, current_stats=None):
    if current_stats is None:
        current_stats = _bridge_stats(1.0)
    return workflow._continue_nqs_to_spectrum(
        update,
        jnp.asarray(0.0),
        current_stats,
        None,
        None,
        None,
        jax.random.PRNGKey(0),
        response_apply=None,
        ground_logpsi=None,
        ground_params=None,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=0.0,
        target_omega=target,
        spectrum_omega=np.asarray([target, target + 0.1]),
    )


def test_continuation_default_min_step_uses_finer_spectrum_spacing():
    config = MolecularLITConfig(eta=0.003, nqs_continuation_min_step=None)

    assert _continuation_min_step(
        config, np.asarray([0.772, 0.77225])
    ) == pytest.approx(0.00025)


def test_physics_continuation_step_uses_lit_residual_scale():
    stats = _bridge_stats(1.0, lit=4.0, source_norm=1.0)

    step = _physics_continuation_step(
        stats,
        gap=1.0,
        fraction=0.2,
        min_step=0.01,
    )

    assert step == pytest.approx(0.1)


def test_adaptive_continuation_bisects_and_propagates_bridge_best():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=10.0,
        nqs_continuation_fidelity_retention=0.95,
        nqs_continuation_min_step=0.01,
        nqs_continuation_max_points=20,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, probes = _mock_bridge_workflow(config)

    params, _, records, _ = _run_bridge(workflow, update, target=1.0)
    bridge_omega = [record.omega for record in records if record.optimized]

    assert bridge_omega
    assert np.all(np.diff(bridge_omega) > 0.0)
    assert bridge_omega[-1] < 1.0
    assert starts == pytest.approx([0.0, *bridge_omega[:-1]])
    assert optimized == pytest.approx(bridge_omega)
    assert float(params) == pytest.approx(bridge_omega[-1])
    assert records[-1].omega == pytest.approx(1.0)
    assert not records[-1].optimized
    assert probes[0] == pytest.approx(1.0)
    assert probes[-1] == pytest.approx(1.0)


def test_bridge_point_cap_still_allows_final_target_probe():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.2,
        nqs_continuation_fidelity_retention=0.1,
        nqs_continuation_min_step=0.01,
        nqs_continuation_max_points=1,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, _, optimized, probes = _mock_bridge_workflow(config)

    _, _, records, _ = _run_bridge(workflow, update, target=0.4)

    assert [record.omega for record in records if record.optimized] == pytest.approx(
        [0.2]
    )
    assert records[-1].omega == pytest.approx(0.4)
    assert not records[-1].optimized
    assert optimized == pytest.approx([0.2])
    assert probes[-1] == pytest.approx(0.4)


def test_adaptive_continuation_rejects_invalid_probe_at_min_step():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=1.0,
        nqs_continuation_fidelity_retention=0.95,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
    )
    workflow, update, *_ = _mock_bridge_workflow(config)

    def invalid_evaluate(*_args, **_kwargs):
        return _bridge_stats(0.0, invalid=1.0)

    workflow._nqs_stats_chunked = invalid_evaluate

    with pytest.raises(RuntimeError, match="non-finite/invalid"):
        _run_bridge(workflow, update, target=0.4)
