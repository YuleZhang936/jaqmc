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
    _continuation_probe_is_acceptable,
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


def _bridge_stats(
    fidelity,
    *,
    invalid=0.0,
    ess=1.0,
    lit=1.0,
    reverse_kl=0.0,
    source_norm=1.0,
):
    return _BridgeStats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(reverse_kl),
        invalid_sample_fraction=jnp.asarray(invalid),
        reweight_ess_fraction=jnp.asarray(ess),
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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("nqs_stage_fidelity_min", -1e-6),
        ("nqs_stage_fidelity_min", 1.0 + 1e-6),
        ("nqs_stage_fidelity_min", np.nan),
        ("nqs_stage_reweight_ess_fraction_min", -1e-6),
        ("nqs_stage_reweight_ess_fraction_min", 1.0 + 1e-6),
        ("nqs_stage_reweight_ess_fraction_min", np.nan),
        ("nqs_stage_fidelity_gain_min", -1e-6),
        ("nqs_stage_fidelity_gain_min", 1.0 + 1e-6),
        ("nqs_stage_fidelity_gain_min", np.nan),
    ],
)
def test_stage_gate_config_rejects_values_outside_unit_interval(
    field_name,
    value,
):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        workflow._validate_continuation_config()


@pytest.mark.parametrize("value", [0.0, 1.0])
@pytest.mark.parametrize(
    "field_name",
    [
        "nqs_stage_fidelity_min",
        "nqs_stage_reweight_ess_fraction_min",
        "nqs_stage_fidelity_gain_min",
    ],
)
def test_stage_gate_config_accepts_unit_interval_boundaries(field_name, value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(**{field_name: value})

    workflow._validate_continuation_config()


def test_continuation_probe_requires_absolute_fidelity_floor():
    current = _bridge_stats(0.999, ess=0.8)
    candidate = _bridge_stats(0.98, ess=0.8)

    # The candidate retains far more than 95% of the current fidelity, but it
    # is still scientifically inadmissible under the absolute stage floor.
    assert candidate.fidelity >= 0.95 * current.fidelity
    assert not _continuation_probe_is_acceptable(
        current,
        candidate,
        retention=0.95,
        min_fidelity=0.99,
        min_reweight_ess_fraction=0.05,
    )


def test_continuation_probe_requires_absolute_ess_floor():
    current = _bridge_stats(0.999, ess=0.8)
    low_ess = _bridge_stats(0.995, ess=0.049)
    boundary = _bridge_stats(0.995, ess=0.05)

    assert not _continuation_probe_is_acceptable(
        current,
        low_ess,
        retention=0.95,
        min_fidelity=0.99,
        min_reweight_ess_fraction=0.05,
    )
    assert _continuation_probe_is_acceptable(
        current,
        boundary,
        retention=0.95,
        min_fidelity=0.99,
        min_reweight_ess_fraction=0.05,
    )


def test_min_step_probe_failure_raises_before_optimizer_when_override_disabled():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.1,
        nqs_continuation_fidelity_retention=0.99,
        nqs_stage_fidelity_min=0.99,
        nqs_continuation_allow_min_step_override=False,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, _ = _mock_bridge_workflow(config)

    with pytest.raises(RuntimeError, match="minimum step"):
        _run_bridge(workflow, update, target=0.2)

    assert starts == []
    assert optimized == []


def test_default_min_step_override_preserves_legacy_propagation():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.1,
        nqs_continuation_fidelity_retention=0.99,
        nqs_stage_fidelity_min=0.99,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, _ = _mock_bridge_workflow(config)

    _, _, records, _ = _run_bridge(workflow, update, target=0.2)

    assert config.nqs_continuation_allow_min_step_override is True
    assert starts == pytest.approx([0.0])
    assert optimized == pytest.approx([0.1])
    assert records[0].optimized
    assert not records[0].probe_accepted
    assert records[0].min_step_override
    assert not records[-1].optimized
    assert not records[-1].probe_accepted
    assert records[-1].min_step_override


def _mock_stage_optimizer(config, candidate_fidelity):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = config

    def evaluate(_response_apply, params, *_args, **_kwargs):
        fidelity = 0.8 if float(params) < 0.5 else candidate_fidelity
        return _bridge_stats(fidelity, ess=0.5)

    workflow._nqs_stats_chunked = evaluate

    def init_carry(_data, rng, _params):
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

    def update(params, _pool, _omega, carry, _iteration):
        return params + 1.0, _bridge_stats(candidate_fidelity, ess=0.5), carry

    update.init_carry = init_carry
    return workflow, update


def _run_stage_optimizer(workflow, update):
    return workflow._optimize_nqs_frequency(
        update,
        initial_params=jnp.asarray(0.0),
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
        omega=-1.0,
        iterations=1,
        stage="test",
    )


def test_stage_optimizer_rejects_best_checkpoint_below_required_fidelity():
    config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.85,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.1,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update = _mock_stage_optimizer(config, candidate_fidelity=0.87)

    # Initial F=0.8 is below the floor, so the recovery requirement is
    # max(0.85, 0.8 + 0.1) = 0.9.  Merely crossing the floor is insufficient.
    with pytest.raises(RuntimeError, match=r"quality gate.*required=0\.900000"):
        _run_stage_optimizer(workflow, update)


def test_stage_optimizer_accepts_checkpoint_reaching_required_fidelity():
    config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.85,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.1,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update = _mock_stage_optimizer(config, candidate_fidelity=0.91)

    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(1.0)
    assert float(stats.fidelity) == pytest.approx(0.91)
    assert selected_iteration == 1


def _mock_stage_optimizer_with_stats(config, initial_stats, candidate_stats):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = config

    def evaluate(_response_apply, params, *_args, **_kwargs):
        return initial_stats if float(params) < 0.5 else candidate_stats

    workflow._nqs_stats_chunked = evaluate

    def init_carry(_data, rng, _params):
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

    def update(params, _pool, _omega, carry, _iteration):
        return params + 1.0, candidate_stats, carry

    update.init_carry = init_carry
    return workflow, update


def test_gate_eligible_initial_is_not_displaced_by_lower_loss_ineligible_candidate():
    config = MolecularLITConfig(
        nqs_reverse_kl_weight=1.0,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.995, ess=0.5, reverse_kl=0.2)
    candidate = _bridge_stats(0.98, ess=0.5, reverse_kl=0.0)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    # The candidate has the lower regularized loss (0.02 versus 0.205), but it
    # must not displace an iteration-zero checkpoint that passes every gate.
    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.995)
    assert selected_iteration == 0


def test_gate_eligible_candidate_is_saved_despite_worse_regularized_loss():
    config = MolecularLITConfig(
        nqs_reverse_kl_weight=1.0,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.8, ess=0.5, reverse_kl=0.0)
    candidate = _bridge_stats(0.995, ess=0.5, reverse_kl=0.5)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    # The candidate has the worse regularized loss (0.505 versus 0.2), but it
    # is the only gate-eligible checkpoint and therefore must be propagated.
    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(1.0)
    assert float(stats.fidelity) == pytest.approx(0.995)
    assert selected_iteration == 1
