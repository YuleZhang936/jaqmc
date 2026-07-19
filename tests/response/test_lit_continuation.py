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
    _continuation_capacity_diagnostics,
    _continuation_checkpoint_digests,
    _continuation_history_step_cap,
    _continuation_min_step,
    _continuation_probe_is_acceptable,
    _ContinuationRecord,
    _empty_nqs_lit_stats,
    _FidelityPlateauTracker,
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


def _post_optimization_bridge_workflow(
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
            large_attempt_count = sum(
                np.isclose(attempt, omega) for attempt in attempts
            )
            fidelity = (
                large_optimized_fidelity
                if optimized and large_attempt_count == 1
                else (0.995 if optimized else 0.974661)
            )
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


def test_continuation_capacity_diagnostics_matches_formal_run_budget():
    diagnostics = _continuation_capacity_diagnostics(
        remaining_gap=0.772 - 0.098059,
        optimized_count=99,
        maximum=256,
        chosen_step=0.001227,
    )

    assert diagnostics.remaining_gap == pytest.approx(0.673941)
    assert diagnostics.remaining_bridge_slots == 157
    assert diagnostics.required_mean_step == pytest.approx(0.004265449367088608)
    assert diagnostics.capacity_ratio == pytest.approx(0.287660195773814)


@pytest.mark.parametrize(
    ("chosen_step", "expected_ratio"),
    [(0.4, 1.0), (0.1, 0.25)],
)
def test_continuation_capacity_with_no_bridge_slots_compares_target_step(
    chosen_step,
    expected_ratio,
):
    diagnostics = _continuation_capacity_diagnostics(
        remaining_gap=0.4,
        optimized_count=256,
        maximum=256,
        chosen_step=chosen_step,
    )

    assert diagnostics.remaining_bridge_slots == 0
    assert diagnostics.required_mean_step == pytest.approx(0.4)
    assert diagnostics.capacity_ratio == pytest.approx(expected_ratio)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"remaining_gap": 0.0}, "remaining_gap"),
        ({"remaining_gap": np.nan}, "remaining_gap"),
        ({"chosen_step": 0.0}, "chosen_step"),
        ({"chosen_step": np.inf}, "chosen_step"),
        ({"optimized_count": -1}, "point counts"),
        ({"optimized_count": 3, "maximum": 2}, "point counts"),
        ({"maximum": -1}, "point counts"),
    ],
)
def test_continuation_capacity_diagnostics_rejects_invalid_inputs(
    overrides,
    message,
):
    arguments = {
        "remaining_gap": 0.4,
        "optimized_count": 1,
        "maximum": 2,
        "chosen_step": 0.1,
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        _continuation_capacity_diagnostics(**arguments)


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


def test_post_optimizer_fidelity_has_no_absolute_gate_or_backtrack():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, rng_starts, attempts = _post_optimization_bridge_workflow(
        config,
        large_optimized_fidelity=0.989996,
    )
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

    # This is the formal-run boundary that used to fail the arbitrary 0.99
    # floor.  It is a healthy best held-out checkpoint and is committed without
    # retrying a smaller frequency step.
    assert attempts == pytest.approx([0.4])
    assert starts == pytest.approx([0.0])
    assert len(rng_starts) == 1
    np.testing.assert_array_equal(rng_starts[0], initial_rng)
    assert float(params) == pytest.approx(0.4)
    assert float(stats.fidelity) == pytest.approx(0.989996)
    np.testing.assert_array_equal(rng, jax.random.fold_in(initial_rng, 400))

    optimized_records = [record for record in records if record.optimized]
    assert [record.omega for record in optimized_records] == pytest.approx([0.4])
    assert [record.bisections for record in optimized_records] == [0]
    assert float(optimized_records[0].stats.fidelity) == pytest.approx(0.989996)
    assert records[-1].omega == pytest.approx(0.6)
    assert not records[-1].optimized

    assert len(checkpoints) == 1
    checkpoint_params, _, checkpoint_rng, checkpoint_omega, checkpoint_records = (
        checkpoints[0]
    )
    assert float(checkpoint_params) == pytest.approx(0.4)
    assert checkpoint_omega == pytest.approx(0.4)
    np.testing.assert_array_equal(
        checkpoint_rng,
        jax.random.fold_in(initial_rng, 400),
    )
    assert checkpoint_records[-1].omega == pytest.approx(0.4)
    assert checkpoint_records[-1].optimized


def test_continuation_runs_configured_maximum_without_plateau():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=2,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, updates = _post_optimization_bridge_workflow(
        config,
        large_optimized_fidelity=0.995,
    )

    _, _, records, _ = _run_bridge(
        workflow,
        update,
        target=0.6,
        current_stats=_bridge_stats(0.995, ess=0.8),
    )

    assert updates == pytest.approx([0.4, 0.4])
    assert starts == pytest.approx([0.0])
    optimized_records = [record for record in records if record.optimized]
    assert len(optimized_records) == 1
    assert optimized_records[0].omega == pytest.approx(0.4)


def test_inherited_probe_bisection_is_held_then_clean_success_grows_cautiously():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_step_growth_factor=1.25,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, attempts = _post_optimization_bridge_workflow(config)
    checkpoints = []

    def evaluate(_response_apply, params, *_args, **kwargs):
        omega = float(kwargs["omega"])
        optimized = np.isclose(float(params), omega)
        first_probe_too_large = (
            not optimized and np.isclose(float(params), 0.0) and omega > 0.3
        )
        fidelity = 0.90 if first_probe_too_large else (0.995 if optimized else 0.985)
        return _bridge_stats(fidelity, ess=0.8)

    workflow._nqs_stats_chunked = evaluate

    _, _, records, _ = _run_bridge(
        workflow,
        update,
        target=0.9,
        current_stats=_bridge_stats(0.995, ess=0.8),
        checkpoint_callback=lambda *args: checkpoints.append(args),
    )

    # The inherited 0.4 probe fails relative retention and bisects to 0.2.
    # That recovered step is held once; only the next clean success grows it.
    assert attempts == pytest.approx([0.2, 0.4, 0.65])
    assert starts == pytest.approx([0.0, 0.2, 0.4])
    optimized_records = [record for record in records if record.optimized]
    assert [record.step for record in optimized_records] == pytest.approx(
        [0.2, 0.2, 0.25]
    )
    assert [record.bisections for record in optimized_records] == [1, 0, 0]
    assert len(checkpoints) == 3
    assert records[-1].omega == pytest.approx(0.9)
    assert not records[-1].optimized


@pytest.mark.parametrize(
    ("record_kwargs", "expected_candidate"),
    [
        ({"bisections": 1}, 0.4),
        ({"probe_accepted": False}, 0.4),
        ({"min_step_override": True}, 0.4),
        ({}, 0.45),
    ],
)
def test_resume_reconstructs_step_cap_from_latest_success(
    record_kwargs,
    expected_candidate,
):
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_step_growth_factor=1.25,
        nqs_continuation_fidelity_retention=0.1,
        nqs_continuation_min_step=0.01,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, _, optimized, _ = _mock_bridge_workflow(config)
    saved_stats = _bridge_stats(1.0)
    fields = dict(
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
    fields.update(record_kwargs)
    saved_record = _ContinuationRecord(**fields)

    _run_bridge(
        workflow,
        update,
        target=0.9,
        current_stats=saved_stats,
        response_params=0.2,
        resume_omega=0.2,
        existing_records=(saved_record,),
    )

    assert optimized[0] == pytest.approx(expected_candidate)


def test_resume_step_cap_is_clamped_to_current_minimum_step():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_step_growth_factor=1.25,
        nqs_continuation_fidelity_retention=0.1,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, _, optimized, _ = _mock_bridge_workflow(config)
    saved_stats = _bridge_stats(1.0)
    saved_record = _ContinuationRecord(
        omega=0.2,
        optimized=True,
        selected_iteration=1,
        stats=saved_stats,
        inherited_fidelity=0.99,
        step=0.02,
        bisections=1,
        probe_accepted=True,
        min_step_override=False,
    )

    _run_bridge(
        workflow,
        update,
        target=0.6,
        current_stats=saved_stats,
        response_params=0.2,
        resume_omega=0.2,
        existing_records=(saved_record,),
    )

    assert optimized[0] == pytest.approx(0.3)


def test_history_step_cap_ignores_unoptimized_tail_record():
    stats = _bridge_stats(1.0)
    accepted = _ContinuationRecord(
        omega=0.2,
        optimized=True,
        selected_iteration=1,
        stats=stats,
        inherited_fidelity=0.99,
        step=0.2,
        bisections=1,
        probe_accepted=True,
        min_step_override=False,
    )
    target_probe = accepted._replace(
        omega=0.9,
        optimized=False,
        selected_iteration=-1,
        step=0.7,
    )

    assert _continuation_history_step_cap(
        [accepted, target_probe],
        growth_factor=1.25,
        min_step=0.01,
    ) == pytest.approx(0.2)


def test_post_optimizer_low_ess_candidate_returns_inherited_healthy_checkpoint():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, attempts = _post_optimization_bridge_workflow(
        config,
        large_optimized_fidelity=0.995,
        optimized_ess=0.01,
    )
    checkpoints = []

    params, stats, records, _ = _run_bridge(
        workflow,
        update,
        target=0.6,
        current_stats=_bridge_stats(0.990259, ess=0.971),
        checkpoint_callback=lambda *args: checkpoints.append(args),
    )

    assert attempts == pytest.approx([0.4])
    assert starts == pytest.approx([0.0])
    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.974661)
    optimized_records = [record for record in records if record.optimized]
    assert len(optimized_records) == 1
    assert float(optimized_records[0].stats.reweight_ess_fraction) == pytest.approx(0.8)
    assert len(checkpoints) == 1


def test_post_optimizer_runtime_error_is_not_backtracked():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=0.4,
        nqs_continuation_fidelity_retention=0.95,
        nqs_continuation_min_step=0.1,
        nqs_continuation_max_points=10,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, starts, _, _ = _post_optimization_bridge_workflow(config)
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


@pytest.mark.parametrize("value", [-1e-6, 1.0 + 1e-6, np.nan])
def test_stage_ess_config_rejects_values_outside_unit_interval(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_stage_reweight_ess_fraction_min=value)

    with pytest.raises(ValueError, match="nqs_stage_reweight_ess_fraction_min"):
        workflow._validate_continuation_config()


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_stage_ess_config_accepts_unit_interval_boundaries(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_stage_reweight_ess_fraction_min=value)

    workflow._validate_continuation_config()


@pytest.mark.parametrize("value", [0.999, 2.001, np.nan, np.inf])
def test_continuation_step_growth_rejects_unsafe_values(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_continuation_step_growth_factor=value)

    with pytest.raises(ValueError, match="step_growth_factor"):
        workflow._validate_continuation_config()


@pytest.mark.parametrize("value", [1.0, 2.0])
def test_continuation_step_growth_accepts_safe_boundaries(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(nqs_continuation_step_growth_factor=value)

    workflow._validate_continuation_config()


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("nqs_fidelity_plateau_start_iteration", -1),
        ("nqs_fidelity_plateau_start_iteration", 1.5),
        ("nqs_fidelity_plateau_start_iteration", True),
        ("nqs_fidelity_plateau_patience_iterations", -1),
        ("nqs_fidelity_plateau_patience_iterations", 1.5),
        ("nqs_fidelity_plateau_patience_iterations", True),
    ],
)
def test_fidelity_plateau_integer_config_rejects_invalid_values(field_name, value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        workflow._validate_nqs_iteration_config()


@pytest.mark.parametrize("value", [-1e-6, 1.0 + 1e-6, np.nan, np.inf])
def test_fidelity_plateau_min_delta_rejects_invalid_values(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_fidelity_plateau_min_delta=value,
    )

    with pytest.raises(ValueError, match="nqs_fidelity_plateau_min_delta"):
        workflow._validate_nqs_iteration_config()


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_fidelity_plateau_min_delta_accepts_unit_interval_boundaries(value):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = MolecularLITConfig(
        nqs_fidelity_plateau_min_delta=value,
    )

    workflow._validate_nqs_iteration_config()


def test_continuation_probe_uses_relative_fidelity_only():
    current = _bridge_stats(0.990063, ess=0.708)
    candidate = _bridge_stats(0.989922, ess=0.704427)

    # This is the exact old formal-run boundary.  There is deliberately no
    # absolute post-optimization fidelity floor.
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


def test_target_probe_below_previous_absolute_floor_is_kept_for_spectrum():
    config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_continuation_iterations=1,
        nqs_continuation_step_fraction=10.0,
        nqs_continuation_fidelity_retention=0.95,
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
        nqs_stage_reweight_ess_fraction_min=0.05,
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
        # Execution-only policy changes are excluded from the physical-state
        # fingerprint, so a legacy full-config digest remains resumable.
        full_config_digest=f"legacy-gain-config-{full_digest}",
        warm_start_selected_iteration=1500,
    )

    restore_config = MolecularLITConfig(
        nqs_warm_start_omega=0.0,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_continuation_restore_path=str(old_run),
    )
    restore_fingerprint, restore_full_digest = _continuation_checkpoint_digests(
        restore_config,
        **digest_args,
    )
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = restore_config
    workflow.save_path = new_run

    def restore():
        return workflow._restore_nqs_continuation_checkpoint(
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

    revalidated = stats._replace(fidelity=jnp.asarray(0.996))
    workflow._evaluate_nqs_checkpoint = lambda **_kwargs: revalidated
    restored = restore()

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
        fidelity=jnp.asarray(0.990001)
    )
    assert restore().current_stats.fidelity == pytest.approx(0.990001)

    workflow._evaluate_nqs_checkpoint = lambda **_kwargs: stats._replace(
        fidelity=jnp.asarray(0.989996)
    )
    assert restore().current_stats.fidelity == pytest.approx(0.989996)

    workflow._evaluate_nqs_checkpoint = lambda **_kwargs: stats._replace(
        reweight_ess_fraction=jnp.asarray(0.049)
    )
    with pytest.raises(RuntimeError, match=r"ESS fraction=.*required=0\.050000"):
        restore()


def test_continuation_state_fingerprint_allows_new_execution_policy_not_ansatz():
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
        nqs_fidelity_plateau_start_iteration=1500,
        nqs_fidelity_plateau_patience_iterations=1000,
        nqs_fidelity_plateau_min_delta=1e-5,
        nqs_continuation_allow_min_step_override=False,
    )
    recovered_config = MolecularLITConfig(
        nqs_fidelity_plateau_start_iteration=1200,
        nqs_fidelity_plateau_patience_iterations=500,
        nqs_fidelity_plateau_min_delta=2e-5,
        nqs_continuation_allow_min_step_override=True,
    )
    changed_ansatz = MolecularLITConfig(nqs_response_ndets=32)
    relocated_inputs = MolecularLITConfig(
        nqs_continuation_allow_min_step_override=False,
        nqs_checkpoint_path="/moved/ground",
        nqs_source_pool_dir="/moved/source_pools",
        nqs_reuse_source_pool=False,
        nqs_save_source_pool=False,
    )
    changed_controller = MolecularLITConfig(
        nqs_continuation_allow_min_step_override=False,
        nqs_continuation_step_growth_factor=1.5,
    )
    changed_plateau_policy = MolecularLITConfig(
        nqs_fidelity_plateau_start_iteration=100,
        nqs_fidelity_plateau_patience_iterations=200,
        nqs_fidelity_plateau_min_delta=5e-5,
        nqs_continuation_allow_min_step_override=False,
    )
    changed_execution_layout = MolecularLITConfig(
        nqs_continuation_allow_min_step_override=False,
        nqs_data_parallel="local_devices",
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
    controller_state, controller_full = _continuation_checkpoint_digests(
        changed_controller,
        **digest_args,
    )
    plateau_state, plateau_full = _continuation_checkpoint_digests(
        changed_plateau_policy,
        **digest_args,
    )
    execution_state, execution_full = _continuation_checkpoint_digests(
        changed_execution_layout,
        **digest_args,
    )

    assert recovered_state == old_state
    assert recovered_full != old_full
    assert changed_state != old_state
    assert relocated_state == old_state
    assert relocated_full != old_full
    assert controller_state == old_state
    assert controller_full != old_full
    assert plateau_state == old_state
    assert plateau_full != old_full
    assert execution_state == old_state
    assert execution_full != old_full

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


def _run_stage_optimizer(
    workflow,
    update,
    *,
    iterations=1,
):
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
        iterations=iterations,
        stage="test",
    )


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


def _mock_optimizer_trajectory(config, stats_by_iteration):
    workflow = object.__new__(MoleculeLITWorkflow)
    workflow.lit_config = config
    init_calls = []
    updates = []

    def stats_at(params):
        return stats_by_iteration[round(float(params))]

    def evaluate(_response_apply, params, *_args, **_kwargs):
        return stats_at(params)

    workflow._nqs_stats_chunked = evaluate

    def init_carry(_data, rng, params):
        init_calls.append((float(params), np.asarray(rng).copy()))
        return SimpleNamespace(direct=SimpleNamespace(rng=rng))

    def update(params, _pool, _omega, carry, iteration):
        updates.append((float(params), iteration))
        next_params = params + 1.0
        next_rng = jax.random.fold_in(carry.direct.rng, iteration + 1)
        next_carry = SimpleNamespace(direct=SimpleNamespace(rng=next_rng))
        return next_params, stats_at(next_params), next_carry

    update.init_carry = init_carry
    return workflow, update, init_calls, updates


def test_plateau_tracker_establishes_baseline_before_counting_patience():
    tracker = _FidelityPlateauTracker(
        start_iteration=1500,
        patience_iterations=1000,
        min_delta=1e-5,
    )

    assert not tracker.observe(1499, 0.99)
    assert tracker.reference_fidelity is None
    assert not tracker.observe(1500, 0.99)
    assert not tracker.observe(2499, 0.99 + 1e-5)
    assert tracker.observe(2500, 0.99 + 1e-5)


def test_plateau_tracker_resets_only_after_cumulative_significant_gain():
    tracker = _FidelityPlateauTracker(
        start_iteration=2,
        patience_iterations=2,
        min_delta=0.01,
    )

    assert not tracker.observe(2, 0.80)
    assert not tracker.observe(3, 0.805)
    assert not tracker.observe(4, 0.811)
    assert tracker.last_significant_iteration == 4
    assert not tracker.observe(5, 0.811)
    assert tracker.observe(6, 0.811)


def test_optimizer_plateau_cannot_stop_at_baseline_iteration():
    config = MolecularLITConfig(
        nqs_fidelity_plateau_start_iteration=2,
        nqs_fidelity_plateau_patience_iterations=2,
        nqs_fidelity_plateau_min_delta=1e-5,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, init_calls, updates = _mock_optimizer_trajectory(
        config,
        {iteration: _bridge_stats(0.989996, ess=0.8) for iteration in range(6)},
    )

    params, stats, selected_iteration, _ = _run_stage_optimizer(
        workflow,
        update,
        iterations=5,
    )

    assert len(init_calls) == 1
    assert len(updates) == 4
    assert [iteration for _, iteration in updates] == [0, 1, 2, 3]
    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.989996)
    assert selected_iteration == 0


def test_optimizer_significant_gain_resets_plateau_patience():
    config = MolecularLITConfig(
        nqs_fidelity_plateau_start_iteration=1,
        nqs_fidelity_plateau_patience_iterations=2,
        nqs_fidelity_plateau_min_delta=0.01,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    fidelities = (0.80, 0.80, 0.805, 0.811, 0.811, 0.811, 0.811)
    workflow, update, _, updates = _mock_optimizer_trajectory(
        config,
        {
            iteration: _bridge_stats(fidelity, ess=0.8)
            for iteration, fidelity in enumerate(fidelities)
        },
    )

    params, stats, selected_iteration, _ = _run_stage_optimizer(
        workflow,
        update,
        iterations=6,
    )

    assert len(updates) == 5
    assert float(params) == pytest.approx(3.0)
    assert float(stats.fidelity) == pytest.approx(0.811)
    assert selected_iteration == 3


def test_optimizer_returns_highest_fidelity_despite_worse_regularized_loss():
    config = MolecularLITConfig(
        nqs_reverse_kl_weight=1.0,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.8, ess=0.5, reverse_kl=0.0)
    candidate = _bridge_stats(0.995, ess=0.5, reverse_kl=0.5)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    # Candidate loss is 0.505 versus 0.2, but held-out fidelity is the primary
    # transfer-learning selection criterion.
    assert float(params) == pytest.approx(1.0)
    assert float(stats.fidelity) == pytest.approx(0.995)
    assert selected_iteration == 1


def test_optimizer_keeps_higher_fidelity_initial_checkpoint():
    config = MolecularLITConfig(
        nqs_reverse_kl_weight=1.0,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.995, ess=0.5, reverse_kl=0.2)
    candidate = _bridge_stats(0.98, ess=0.5, reverse_kl=0.0)
    workflow, update = _mock_stage_optimizer_with_stats(config, initial, candidate)

    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.995)
    assert selected_iteration == 0


@pytest.mark.parametrize(
    "candidate",
    [
        _bridge_stats(0.995, ess=0.01),
        _bridge_stats(0.995, invalid=1.0),
    ],
    ids=["low_ess", "invalid"],
)
def test_optimizer_unhealthy_live_checkpoint_returns_earlier_healthy_best(candidate):
    config = MolecularLITConfig(
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    initial = _bridge_stats(0.99, ess=0.8)
    workflow, update = _mock_stage_optimizer_with_stats(
        config,
        initial,
        candidate,
    )

    params, stats, selected_iteration, _ = _run_stage_optimizer(workflow, update)

    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.99)
    assert selected_iteration == 0


@pytest.mark.parametrize(
    ("stats", "message"),
    [
        (_bridge_stats(0.995, ess=0.01), "ESS"),
        (_bridge_stats(0.995, invalid=1.0), "healthy"),
    ],
    ids=["low_ess", "invalid"],
)
def test_optimizer_fails_when_entire_trajectory_has_no_healthy_checkpoint(
    stats,
    message,
):
    config = MolecularLITConfig(
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update = _mock_stage_optimizer_with_stats(config, stats, stats)

    with pytest.raises(RuntimeError, match=message):
        _run_stage_optimizer(workflow, update)


def test_unhealthy_candidate_resets_plateau_patience():
    config = MolecularLITConfig(
        nqs_fidelity_plateau_start_iteration=1,
        nqs_fidelity_plateau_patience_iterations=2,
        nqs_fidelity_plateau_min_delta=1e-5,
        nqs_stage_reweight_ess_fraction_min=0.05,
        nqs_selection_interval=1,
        nqs_log_interval=0,
    )
    workflow, update, _, updates = _mock_optimizer_trajectory(
        config,
        {
            0: _bridge_stats(0.99, ess=0.8),
            1: _bridge_stats(0.99, ess=0.8),
            2: _bridge_stats(0.995, ess=0.01),
            3: _bridge_stats(0.99, ess=0.8),
            4: _bridge_stats(0.99, ess=0.8),
            5: _bridge_stats(0.99, ess=0.8),
        },
    )

    params, stats, selected_iteration, _ = _run_stage_optimizer(
        workflow,
        update,
        iterations=5,
    )

    # Without a reset the iteration-2 unhealthy sample would be skipped and
    # the stale iteration-1 baseline would stop at iteration 3.
    assert len(updates) == 4
    assert float(params) == pytest.approx(0.0)
    assert float(stats.fidelity) == pytest.approx(0.99)
    assert selected_iteration == 0
