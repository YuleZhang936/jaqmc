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
    _continuation_checkpoint_digests,
    _continuation_min_step,
    _continuation_probe_is_acceptable,
    _ContinuationRecord,
    _empty_nqs_lit_stats,
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


def _run_bridge(
    workflow,
    update,
    *,
    target,
    current_stats=None,
    response_params=0.0,
    resume_omega=None,
    existing_records=(),
    rng=None,
    checkpoint_callback=None,
):
    if current_stats is None:
        current_stats = _bridge_stats(1.0)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    return workflow._continue_nqs_to_spectrum(
        update,
        jnp.asarray(response_params),
        current_stats,
        None,
        None,
        None,
        rng,
        response_apply=None,
        ground_logpsi=None,
        ground_params=None,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=0.0,
        target_omega=target,
        spectrum_omega=np.asarray([target, target + 0.1]),
        resume_omega=resume_omega,
        existing_records=existing_records,
        checkpoint_callback=checkpoint_callback,
    )


def _post_gate_bridge_workflow(
    config,
    *,
    large_optimized_fidelity=0.978273,
    optimized_ess=0.8,
):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = config
    starts = []
    rng_starts = []
    attempts = []

    def init_carry(_data, rng, params):
        starts.append(float(params))
        rng_starts.append(np.asarray(rng).copy())
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

    def update(_params, _pool, omega, carry, _iteration):
        omega_value = float(omega)
        attempts.append(omega_value)
        next_rng = jax.random.fold_in(
            carry.direct.rng,
            round(1000.0 * omega_value),
        )
        next_carry = SimpleNamespace(direct=SimpleNamespace(rng=next_rng))
        return omega, _bridge_stats(1.0), next_carry

    update.init_carry = init_carry

    def evaluate(_response_apply, params, *_args, **kwargs):
        omega = float(kwargs["omega"])
        optimized = np.isclose(float(params), omega)
        if omega > 0.3:
            fidelity = large_optimized_fidelity if optimized else 0.974661
        else:
            fidelity = 0.995 if optimized else 0.985
        ess = optimized_ess if optimized else 0.8
        return _bridge_stats(fidelity, ess=ess)

    workflow._nqs_stats_chunked = evaluate
    return workflow, update, starts, rng_starts, attempts


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


def test_post_optimizer_fidelity_failure_backtracks_from_last_good_state():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, rng_starts, attempts = _post_gate_bridge_workflow(config)
    initial_rng = jax.random.PRNGKey(17)
    checkpoints = []

    params, stats, records, rng = _run_bridge(
        workflow,
        update,
        target=0.6,
        current_stats=_bridge_stats(0.990259, ess=0.971),
        rng=initial_rng,
        checkpoint_callback=lambda *args: checkpoints.append(
            (*args[:-1], list(args[-1]))
        ),
    )

    # This reproduces the formal failure shape: the relative probe accepts the
    # inherited F=0.974661 at the large step, but optimization only reaches
    # F=0.978273.  The failed state is discarded and the half-step succeeds.
    assert attempts == pytest.approx([0.4, 0.2])
    assert starts == pytest.approx([0.0, 0.0])
    assert len(rng_starts) == 2
    np.testing.assert_array_equal(rng_starts[0], initial_rng)
    np.testing.assert_array_equal(rng_starts[1], initial_rng)
    assert float(params) == pytest.approx(0.2)
    assert float(stats.fidelity) == pytest.approx(0.995)
    np.testing.assert_array_equal(rng, jax.random.fold_in(initial_rng, 200))

    optimized_records = [record for record in records if record.optimized]
    assert len(optimized_records) == 1
    assert optimized_records[0].omega == pytest.approx(0.2)
    assert optimized_records[0].bisections == 1
    assert records[-1].omega == pytest.approx(0.6)
    assert not records[-1].optimized

    # A failed trial is never committed to the durable continuation chain.
    assert len(checkpoints) == 1
    checkpoint_params, _, checkpoint_rng, checkpoint_omega, checkpoint_records = (
        checkpoints[0]
    )
    assert float(checkpoint_params) == pytest.approx(0.2)
    assert checkpoint_omega == pytest.approx(0.2)
    np.testing.assert_array_equal(
        checkpoint_rng,
        jax.random.fold_in(initial_rng, 200),
    )
    assert checkpoint_records[-1].omega == pytest.approx(0.2)
    assert checkpoint_records[-1].optimized


def test_post_optimizer_fidelity_failure_at_min_step_fails_closed():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_continuation_allow_min_step_override=True,
        nqs_continuation_min_step=0.4,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, attempts = _post_gate_bridge_workflow(config)
    checkpoints = []

    with pytest.raises(RuntimeError, match=r"quality gate.*required=0\.990000"):
        _run_bridge(
            workflow,
            update,
            target=0.6,
            current_stats=_bridge_stats(0.990259, ess=0.971),
            checkpoint_callback=lambda *args: checkpoints.append(args),
        )

    assert attempts == pytest.approx([0.4])
    assert starts == pytest.approx([0.0])
    assert checkpoints == []


def test_post_optimizer_ess_failure_is_not_backtracked():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, attempts = _post_gate_bridge_workflow(
        config,
        large_optimized_fidelity=0.995,
        optimized_ess=0.01,
    )
    checkpoints = []

    with pytest.raises(RuntimeError, match=r"ESS fraction=.*required=0\.050000"):
        _run_bridge(
            workflow,
            update,
            target=0.6,
            current_stats=_bridge_stats(0.990259, ess=0.971),
            checkpoint_callback=lambda *args: checkpoints.append(args),
        )

    assert attempts == pytest.approx([0.4])
    assert starts == pytest.approx([0.0])
    assert checkpoints == []


def test_post_optimizer_runtime_error_is_not_backtracked():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_fidelity_min=0.99,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, _ = _post_gate_bridge_workflow(config)
    checkpoints = []

    def fail_update(*_args, **_kwargs):
        raise RuntimeError("optimizer backend sentinel")

    fail_update.init_carry = update.init_carry

    with pytest.raises(RuntimeError, match="optimizer backend sentinel"):
        _run_bridge(
            workflow,
            fail_update,
            target=0.6,
            current_stats=_bridge_stats(0.990259),
            checkpoint_callback=lambda *args: checkpoints.append(args),
        )

    assert starts == pytest.approx([0.0])
    assert checkpoints == []


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


def test_continuation_probe_uses_relative_fidelity_not_stage_floor():
    current = _bridge_stats(0.990063, ess=0.708)
    candidate = _bridge_stats(0.989922, ess=0.704427)

    # This is the exact failed formal-run boundary.  The inherited parameters
    # are an excellent initialization even though they have not yet recovered
    # the absolute post-optimization science floor.
    assert candidate.fidelity >= 0.95 * current.fidelity
    assert _continuation_probe_is_acceptable(
        current,
        candidate,
        retention=0.95,
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
        min_reweight_ess_fraction=0.05,
    )
    assert _continuation_probe_is_acceptable(
        current,
        boundary,
        retention=0.95,
        min_reweight_ess_fraction=0.05,
    )


def test_target_probe_below_stage_floor_is_kept_for_spectrum_recovery():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=10.0,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_min_step=0.01,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, _ = _mock_bridge_workflow(config)
    current = _bridge_stats(0.990063, ess=0.708)
    inherited = _bridge_stats(0.989922, ess=0.704427)
    workflow._nqs_stats_chunked = lambda *_args, **_kwargs: inherited

    params, _, records, _ = _run_bridge(
        workflow,
        update,
        target=0.1,
        current_stats=current,
    )

    assert float(params) == pytest.approx(0.0)
    assert starts == []
    assert optimized == []
    assert len(records) == 1
    assert not records[0].optimized
    assert records[0].probe_accepted
    assert records[0].stats.fidelity == pytest.approx(0.989922)


def test_min_step_relative_retention_failure_respects_disabled_recovery():
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


def test_continuation_resume_keeps_history_and_skips_completed_bridges():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.2,
        nqs_continuation_fidelity_retention=0.95,
        nqs_continuation_min_step=0.01,
        nqs_continuation_max_points=20,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, _ = _mock_bridge_workflow(config)
    saved_stats = _bridge_stats(1.0)
    saved_record = _ContinuationRecord(
        omega=0.2,
        optimized=True,
        selected_iteration=1,
        stats=saved_stats,
        inherited_fidelity=0.99,
        step=0.2,
        bisections=0,
        probe_accepted=True,
        min_step_override=False,
    )

    _, _, records, _ = _run_bridge(
        workflow,
        update,
        target=1.0,
        current_stats=saved_stats,
        response_params=0.2,
        resume_omega=0.2,
        existing_records=(saved_record,),
    )

    assert records[0] is saved_record
    assert starts
    assert min(starts) >= 0.2
    assert optimized
    assert min(optimized) > 0.2
    assert sum(record.optimized for record in records) == 1 + len(optimized)


def test_restore_root_allows_axes_not_reached_before_interruption(tmp_path):
    old_run = tmp_path / "old"
    checkpoint_root = old_run / "continuation_checkpoints"
    (checkpoint_root / "axis_x").mkdir(parents=True)
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_continuation_restore_path=str(old_run))

    y_path, y_required = workflow._continuation_checkpoint_restore_path(1)

    assert str(y_path) == str(checkpoint_root / "axis_y")
    assert not y_required

    workflow.lit_config = MolecularLITConfig(
        nqs_continuation_restore_path=str(checkpoint_root / "axis_x")
    )
    _, exact_axis_required = workflow._continuation_checkpoint_restore_path(0)
    assert exact_axis_required

    workflow.lit_config = MolecularLITConfig(
        nqs_continuation_restore_path=str(tmp_path / "missing")
    )
    _, missing_root_required = workflow._continuation_checkpoint_restore_path(0)
    assert missing_root_required


def test_continuation_checkpoint_round_trip_across_run_directories(tmp_path):
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
    )
    response_params = {"w": jnp.asarray([1.0 + 2.0j])}
    ground_params = {"g": jnp.asarray([3.0])}
    stats = _empty_nqs_lit_stats()._replace(
        loss=jnp.asarray(0.005),
        fidelity=jnp.asarray(0.995),
        reverse_kl=jnp.asarray(0.002),
        lit=jnp.asarray(1.0),
        source_norm=jnp.asarray(1.0),
        reweight_ess_fraction=jnp.asarray(0.8),
        invalid_sample_fraction=jnp.asarray(0.0),
        source_covariance_loss=jnp.asarray(0.0),
        source_covariance_max_loss=jnp.asarray(0.0),
    )
    record = _ContinuationRecord(
        omega=0.2,
        optimized=True,
        selected_iteration=100,
        stats=stats,
        inherited_fidelity=0.989922,
        step=0.2,
        bisections=0,
        probe_accepted=True,
        min_step_override=False,
    )
    digest_args = dict(
        response_params=response_params,
        ground_params=ground_params,
        train_pool={"electrons": jnp.asarray([[0.1, 0.2, 0.3]])},
        eval_pool={"electrons": jnp.asarray([[0.4, 0.5, 0.6]])},
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        ground_checkpoint_step=7,
        response_parity=-1,
        target_omega=0.4,
        spectrum_omega=np.asarray([0.4, 0.5]),
    )
    state_fingerprint, full_digest = _continuation_checkpoint_digests(
        config,
        **digest_args,
    )
    saver = object.__new__(MoleculeLITWorkflow)
    saver.lit_config = config
    saver.save_path = old_run
    rng = jax.random.PRNGKey(17)
    saver._save_nqs_continuation_checkpoint(
        response_params,
        stats,
        rng,
        0.2,
        [record],
        axis=0,
        target_omega=0.4,
        ground_checkpoint_step=7,
        ground_energy=-2.0,
        source_center=0.0,
        source_norm=1.0,
        response_parity=-1,
        state_fingerprint=state_fingerprint,
        full_config_digest=full_digest,
        warm_start_selected_iteration=1500,
    )

    restore_config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_continuation_restore_path=str(old_run),
    )
    restore_fingerprint, restore_full_digest = _continuation_checkpoint_digests(
        restore_config,
        **digest_args,
    )
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = restore_config
    workflow.save_path = new_run
    revalidated = stats._replace(fidelity=jnp.asarray(0.996))
    workflow._evaluate_nqs_checkpoint = lambda **_kwargs: revalidated

    restored = workflow._restore_nqs_continuation_checkpoint(
        {"w": jnp.asarray([0.0 + 0.0j])},
        jax.random.PRNGKey(0),
        None,
        SimpleNamespace(),
        response_apply=None,
        ground_logpsi=None,
        ground_params=ground_params,
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        ground_checkpoint_step=7,
        response_parity=-1,
        target_omega=0.4,
        state_fingerprint=restore_fingerprint,
        full_config_digest=restore_full_digest,
    )

    assert restored is not None
    np.testing.assert_allclose(np.asarray(restored.response_params["w"]), [1 + 2j])
    np.testing.assert_array_equal(np.asarray(restored.rng), np.asarray(rng))
    assert restored.current_omega == pytest.approx(0.2)
    assert restored.current_stats.fidelity == pytest.approx(0.996)
    assert restored.warm_start_selected_iteration == 1500
    assert len(restored.records) == 1
    assert restored.records[0].omega == pytest.approx(0.2)
    assert restored.records[0].stats.fidelity == pytest.approx(0.996)

    workflow._evaluate_nqs_checkpoint = lambda **_kwargs: stats._replace(
        fidelity=jnp.asarray(0.9901)
    )
    with pytest.raises(RuntimeError, match=r"required=0\.990922"):
        workflow._restore_nqs_continuation_checkpoint(
            {"w": jnp.asarray([0.0 + 0.0j])},
            jax.random.PRNGKey(0),
            None,
            SimpleNamespace(),
            response_apply=None,
            ground_logpsi=None,
            ground_params=ground_params,
            axis=0,
            source_center=0.0,
            source_norm=1.0,
            ground_energy=-2.0,
            ground_checkpoint_step=7,
            response_parity=-1,
            target_omega=0.4,
            state_fingerprint=restore_fingerprint,
            full_config_digest=restore_full_digest,
        )


def test_continuation_state_fingerprint_allows_new_gates_but_not_new_ansatz():
    params = {"w": jnp.asarray([1.0])}
    ground = {"g": jnp.asarray([2.0])}
    digest_args = dict(
        response_params=params,
        ground_params=ground,
        train_pool={"electrons": jnp.asarray([[0.1]])},
        eval_pool={"electrons": jnp.asarray([[0.2]])},
        axis=0,
        source_center=0.0,
        source_norm=1.0,
        ground_energy=-2.0,
        ground_checkpoint_step=7,
        response_parity=-1,
        target_omega=0.4,
        spectrum_omega=np.asarray([0.4, 0.5]),
    )
    old_config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.99,
        nqs_continuation_allow_min_step_override=False,
    )
    recovered_config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.995,
        nqs_continuation_allow_min_step_override=True,
    )
    changed_ansatz = MolecularLITConfig(nqs_response_ndets=32)
    relocated_inputs = MolecularLITConfig(
        nqs_stage_fidelity_min=0.99,
        nqs_continuation_allow_min_step_override=False,
        nqs_checkpoint_path="/moved/ground",
        nqs_source_pool_dir="/moved/source_pools",
        nqs_reuse_source_pool=False,
        nqs_save_source_pool=False,
    )

    old_state, old_full = _continuation_checkpoint_digests(
        old_config,
        **digest_args,
    )
    recovered_state, recovered_full = _continuation_checkpoint_digests(
        recovered_config,
        **digest_args,
    )
    changed_state, _ = _continuation_checkpoint_digests(
        changed_ansatz,
        **digest_args,
    )
    relocated_state, relocated_full = _continuation_checkpoint_digests(
        relocated_inputs,
        **digest_args,
    )

    assert recovered_state == old_state
    assert recovered_full != old_full
    assert changed_state != old_state
    assert relocated_state == old_state
    assert relocated_full != old_full

    changed_ground_state, _ = _continuation_checkpoint_digests(
        old_config,
        **{**digest_args, "ground_params": {"g": jnp.asarray([999.0])}},
    )
    changed_pool_state, _ = _continuation_checkpoint_digests(
        old_config,
        **{
            **digest_args,
            "eval_pool": {"electrons": jnp.asarray([[999.0]])},
        },
    )
    assert changed_ground_state != old_state
    assert changed_pool_state != old_state


def test_min_step_low_ess_cannot_bypass_recovery_gate():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.1,
        nqs_continuation_fidelity_retention=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_allow_min_step_override=True,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, optimized, _ = _mock_bridge_workflow(config)

    def low_ess_evaluate(*_args, **_kwargs):
        return _bridge_stats(0.999, ess=0.049)

    workflow._nqs_stats_chunked = low_ess_evaluate

    with pytest.raises(RuntimeError, match=r"insufficient.*ESS"):
        _run_bridge(workflow, update, target=0.2)

    assert starts == []
    assert optimized == []


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


def test_exact_failed_probe_can_recover_through_strict_post_optimizer_gate():
    config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.989922, ess=0.704427)
    candidate = _bridge_stats(0.99095, ess=0.70)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(1.0)
    assert float(stats.fidelity) == pytest.approx(0.99095)
    assert selected_iteration == 1


def test_exact_failed_probe_is_rejected_if_recovery_misses_required_gain():
    config = MolecularLITConfig(
        nqs_stage_fidelity_min=0.99,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_stage_fidelity_gain_min=0.001,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.989922, ess=0.704427)
    candidate = _bridge_stats(0.99090, ess=0.70)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    with pytest.raises(RuntimeError, match=r"quality gate.*required=0\.990922"):
        _run_stage_optimizer(workflow, update)


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
