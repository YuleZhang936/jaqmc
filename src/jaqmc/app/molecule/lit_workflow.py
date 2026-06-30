# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole NQS-LIT workflow."""

from __future__ import annotations

import logging
import operator
import os
import re
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from upath import UPath

from jaqmc.app.molecule.data import data_init
from jaqmc.app.molecule.workflow import configure_system
from jaqmc.data import BatchedData
from jaqmc.response.lit import lit_error_bound
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    SourceRatioApply,
    ground_local_energy,
    local_action_ratio,
    molecular_electronic_dipole,
    nqs_lit_source_sampled_stats,
    nqs_lit_source_sampled_sums,
    nqs_lit_stats_from_source_sums,
    restore_params_from_checkpoint,
)
from jaqmc.response.spectrum import find_spectrum_peaks
from jaqmc.response.symmetry import (
    SymmetryProjector,
    identity_spatial_projector,
    make_dipole_spatial_projectors,
    make_ground_spatial_projector,
    make_spin_projector,
    project_value,
    projected_log_apply,
    safe_complex_log,
)
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc.wavefunction.output.envelope import EnvelopeType
from jaqmc.workflow.base import Workflow

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


@configurable_dataclass
class MolecularLITConfig:
    """Configuration for molecular dipole NQS-LIT spectra."""

    eta: float = 0.02
    omega_min: float = 0.0
    omega_max: float = 1.0
    omega_points: int = 501
    omega_values: str = ""
    axes: str = "xyz"
    peak_min_height_fraction: float = 0.05
    output_filename: str = "lit_spectrum.npz"
    preview_peaks: int = 5
    scan_parallel: str = "auto"
    scan_parallel_workers: int = 0
    scan_parallel_procs_per_device: int = 1
    scan_parallel_min_points_per_worker: int = 2
    scan_parallel_worker: bool = False
    nqs_checkpoint_path: str = ""
    nqs_allow_untrained_ground: bool = False
    nqs_ground_energy: float | None = None
    nqs_source_center_steps: int = 4
    nqs_source_center_override: float | None = None
    nqs_source_norm_override: float | None = None
    nqs_source_norm_min: float = 1e-10
    nqs_source_burn_in: int = 20
    nqs_source_floor: float = 1e-4
    nqs_train_pool_batches: int = 32
    nqs_eval_pool_batches: int = 8
    nqs_pool_stride: int = 1
    nqs_train_update_batch_size: int = 0
    nqs_sr_score_batch_size: int = 0
    nqs_eval_batch_size: int = 0
    nqs_projected_sr_score_batch_cap: int = 16
    nqs_projected_action_batch_cap: int = 128
    nqs_source_pool_dir: str = ""
    nqs_parallel_shared_source_pool: bool = False
    nqs_reuse_source_pool: bool = True
    nqs_save_source_pool: bool = True
    nqs_reweight_ess_fraction_min: float = 0.05
    nqs_direct_psi_burn_in: int = 5
    nqs_direct_psi_batches: int = 1
    nqs_direct_psi_stride: int = 1
    nqs_energy_steps: int = 2
    nqs_burn_in: int = 20
    nqs_iterations: int = 200
    nqs_learning_rate: float = 1e-3
    nqs_sr_damping: float = 1e-3
    nqs_sr_max_norm: float | None = 0.1
    nqs_sr_score_eps: float = 1e-10
    nqs_warm_start_omega: float | None = None
    nqs_warm_start_iterations: int = 0
    nqs_response_ndets: int = 16
    nqs_response_hidden_dims_single: tuple[int, ...] = field(
        default_factory=lambda: (256, 256, 256, 256)
    )
    nqs_response_hidden_dims_double: tuple[int, ...] = field(
        default_factory=lambda: (32, 32, 32, 32)
    )
    nqs_response_use_last_layer: bool = False
    nqs_response_envelope: EnvelopeType = EnvelopeType.abs_isotropic
    nqs_response_orbitals_spin_split: bool = True
    nqs_symmetry_projectors: bool = True
    nqs_ground_spatial_projector: str = "auto"
    nqs_ground_spatial_irrep: str | None = None
    nqs_spatial_projector: str = "auto"
    nqs_response_spatial_irreps: str = ""
    nqs_spin_projector: str = "auto"
    nqs_ground_spin: float | None = None
    nqs_response_spin: float | None = None
    nqs_symmetry_tolerance: float = 1e-5
    nqs_projection_eps: float = 1e-12
    nqs_projection_chunk_size: int = 4
    nqs_so3_quadrature_order: int = 4
    nqs_so2_quadrature_order: int = 24
    nqs_leakage_diagnostic_samples: int = 8
    isotropic_average: bool = True
    nqs_log_interval: int = 50


@dataclass(frozen=True)
class _ParallelWorker:
    index: int
    block: np.ndarray
    path: UPath
    log_path: UPath
    process: subprocess.Popen[bytes]
    device: str


class _JittedSamplePlan:
    """Shard-map/JIT wrapper for repeated LIT sampling steps."""

    def __init__(self, sample_plan: SamplePlan, batched_data: BatchedData) -> None:
        self._sample_plan = sample_plan
        self._step = parallel_jax.jit_sharded(
            sample_plan.step,
            in_specs=(
                parallel_jax.SHARE_PARTITION,
                batched_data.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.DATA_PARTITION,
            ),
            out_specs=(
                batched_data.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
            ),
            check_vma=True,
        )
        self._run_steps: dict[int, object] = {}

    def init(self, batched_data, rngs):
        return self._sample_plan.init(batched_data, rngs)

    def step(self, params, batched_data, state, rngs):
        shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
        params = jax.device_put(params, shared)
        batched_data = _device_put_batched_data(batched_data)
        state = jax.device_put(state, shared)
        device_rngs = jax.random.split(rngs, jax.device_count()).flatten()
        device_rngs = jax.device_put(
            device_rngs,
            parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
        )
        return self._step(params, batched_data, state, device_rngs)

    def run_steps(self, params, batched_data, state, rngs, steps: int):
        steps = max(0, int(steps))
        if steps == 0:
            return batched_data, state
        run_steps = self._run_steps.get(steps)
        if run_steps is None:

            def run(local_params, local_data, local_state, local_rngs):
                def body(_, carry):
                    data, sampler_state, step_rngs = carry
                    step_rngs, sub_rngs = jax.random.split(step_rngs)
                    data, _, sampler_state = self._sample_plan.step(
                        local_params,
                        data,
                        sampler_state,
                        sub_rngs,
                    )
                    return data, sampler_state, step_rngs

                data, sampler_state, _ = jax.lax.fori_loop(
                    0,
                    steps,
                    body,
                    (local_data, local_state, local_rngs),
                )
                return data, sampler_state

            run_steps = parallel_jax.jit_sharded(
                run,
                in_specs=(
                    parallel_jax.SHARE_PARTITION,
                    batched_data.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.DATA_PARTITION,
                ),
                out_specs=(
                    batched_data.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                ),
                check_vma=True,
            )
            self._run_steps[steps] = run_steps

        shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)
        params = jax.device_put(params, shared)
        batched_data = _device_put_batched_data(batched_data)
        state = jax.device_put(state, shared)
        device_rngs = jax.random.split(rngs, jax.device_count()).flatten()
        device_rngs = jax.device_put(
            device_rngs,
            parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
        )
        return run_steps(params, batched_data, state, device_rngs)


@dataclass(frozen=True)
class _PreparedSourceSampler:
    plan: _JittedSamplePlan
    state: object


def _device_put_batched_data(batched_data: BatchedData) -> BatchedData:
    return jax.device_put(
        batched_data,
        parallel_jax.make_sharding(batched_data.partition_spec),
    )


def _host_batched_data(batched_data: BatchedData) -> BatchedData:
    updates = {}
    for field_name in batched_data.data.field_names:
        updates[field_name] = jax.tree.map(
            lambda leaf: (
                np.asarray(jax.device_get(leaf))
                if isinstance(leaf, jax.Array)
                else leaf
            ),
            getattr(batched_data.data, field_name),
        )
    return batched_data.__class__(
        data=batched_data.data.merge(updates),
        fields_with_batch=batched_data.fields_with_batch,
    )


def _local_device_batched_data(batched_data: BatchedData) -> BatchedData:
    return jax.device_put(_host_batched_data(batched_data), jax.local_devices()[0])


def _run_sample_steps(
    sample_plan: _JittedSamplePlan,
    params,
    batched_data,
    sampler_state,
    rng,
    steps: int,
    *,
    max_chunk: int = 16,
):
    remaining = max(0, int(steps))
    max_chunk = max(1, int(max_chunk))
    while remaining > 0:
        chunk = min(remaining, max_chunk)
        rng, sample_rng = jax.random.split(rng)
        batched_data, sampler_state = sample_plan.run_steps(
            params,
            batched_data,
            sampler_state,
            sample_rng,
            chunk,
        )
        remaining -= chunk
    return batched_data, sampler_state, rng


class _ReweightedDoubleMCSRComponents(NamedTuple):
    """Projected-sector double-MC SR ingredients using pi_Phi reweighting."""

    gradient: jnp.ndarray
    score_aug: jnp.ndarray
    normalization: jnp.ndarray
    fidelity: jnp.ndarray
    reweight_ess: jnp.ndarray
    reweight_ess_fraction: jnp.ndarray


class MoleculeLITWorkflow(Workflow):
    """Compute a molecular dipole response spectrum with NQS-LIT."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        self.lit_config = cfg.get("lit", MolecularLITConfig)
        self.system_config, self.wf = configure_system(cfg)
        self.sampler = cfg.get("sampler", MCMCSampler)
        self._validate_config()

    def run(self) -> None:
        if self._should_run_parallel_scan():
            self._run_parallel_scan()
            return
        self._run_serial_scan()

    def _run_serial_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = _omega_grid_from_config(self.lit_config)
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        rng = jax.random.PRNGKey(seed)
        rng, data_rng, ground_rng, response_rng, sample_rng = jax.random.split(rng, 5)
        batched_data = data_init(self.system_config, self.config.batch_size, data_rng)
        example = batched_data.unbatched_example()

        checkpoint_step, ground_params, ground_logpsi = self._resolve_nqs_ground_state(
            example, ground_rng
        )
        ground_sample_plan = SamplePlan(ground_logpsi, {"electrons": self.sampler})
        sampler_state = ground_sample_plan.init(batched_data, sample_rng)
        ground_sample_plan = _JittedSamplePlan(ground_sample_plan, batched_data)
        batched_data, sampler_state, rng = _run_sample_steps(
            ground_sample_plan,
            ground_params,
            batched_data,
            sampler_state,
            rng,
            self.lit_config.nqs_burn_in,
        )

        ground_energy, batched_data, sampler_state, rng = self._resolve_ground_energy(
            ground_logpsi,
            ground_params,
            batched_data,
            sampler_state,
            ground_sample_plan,
            rng,
        )
        logger.info("Using NQS-LIT ground energy %.10f Ha", ground_energy)

        lit = np.zeros((len(axes), len(omega)), dtype=np.float64)
        broadened = np.zeros_like(lit)
        fidelity = np.zeros_like(lit)
        residual_norm = np.zeros_like(lit)
        action_norm = np.zeros_like(lit)
        source_norm = np.zeros_like(lit)
        error_bound_monitor = np.zeros_like(lit)
        error_d = np.zeros_like(lit)
        reweight_ess = np.zeros_like(lit)
        reweight_ess_fraction = np.zeros_like(lit)
        estimator_mode = np.zeros_like(lit, dtype=np.int64)
        normalization = np.zeros((len(axes), len(omega)), dtype=np.complex128)
        correction_overlap = np.zeros_like(normalization)
        source_centers = np.zeros(len(axes), dtype=np.float64)
        axis_source_norm = np.zeros(len(axes), dtype=np.float64)
        ground_projector = self._make_ground_projector()
        sector_labels: list[str] = []
        sector_axes: list[int] = []
        sector_source_norms: list[float] = []
        sector_source_elastic_overlaps: list[complex] = []
        sector_lit: list[np.ndarray] = []
        sector_broadened: list[np.ndarray] = []
        sector_fidelity: list[np.ndarray] = []
        sector_residual_norm: list[np.ndarray] = []
        sector_action_norm: list[np.ndarray] = []
        sector_error_bound_monitor: list[np.ndarray] = []
        sector_error_d: list[np.ndarray] = []
        sector_reweight_ess: list[np.ndarray] = []
        sector_reweight_ess_fraction: list[np.ndarray] = []
        sector_estimator_mode: list[np.ndarray] = []
        sector_normalization: list[np.ndarray] = []
        sector_correction_overlap: list[np.ndarray] = []
        sector_response_leakage: list[np.ndarray] = []
        sector_carrier_leakage: list[np.ndarray] = []

        for axis_pos, axis in enumerate(axes):
            (
                source_center,
                batched_data,
                sampler_state,
                rng,
            ) = self._load_or_estimate_source_center(
                ground_params,
                batched_data,
                sampler_state,
                ground_sample_plan,
                rng,
                axis=axis,
            )
            source_centers[axis_pos] = source_center
            axis_projectors = self._make_response_projectors(axis)
            axis_sector_count = 0
            for response_projector in axis_projectors:
                raw_source_ratio_apply = self._make_projected_source_ratio(
                    ground_logpsi,
                    ground_params,
                    axis=axis,
                    source_center=source_center,
                    projector=response_projector,
                    elastic_overlap=0.0,
                )
                (
                    source_elastic_overlap,
                    batched_data,
                    sampler_state,
                    rng,
                ) = self._load_or_estimate_source_elastic_overlap(
                    ground_params,
                    batched_data,
                    sampler_state,
                    ground_sample_plan,
                    rng,
                    axis=axis,
                    sector_label=response_projector.label,
                    source_center=source_center,
                    source_ratio_apply=raw_source_ratio_apply,
                    response_projector=response_projector,
                    ground_projector=ground_projector,
                )
                source_ratio_apply = self._make_projected_source_ratio(
                    ground_logpsi,
                    ground_params,
                    axis=axis,
                    source_center=source_center,
                    projector=response_projector,
                    elastic_overlap=source_elastic_overlap,
                )
                (
                    axis_phi_norm,
                    batched_data,
                    sampler_state,
                    rng,
                ) = self._load_or_estimate_source_norm(
                    ground_params,
                    batched_data,
                    sampler_state,
                    ground_sample_plan,
                    rng,
                    axis=axis,
                    sector_label=response_projector.label,
                    source_center=source_center,
                    source_elastic_overlap=source_elastic_overlap,
                    source_ratio_apply=source_ratio_apply,
                )
                if self._skip_source_sector(
                    axis_phi_norm,
                    response_projector,
                    candidate_count=len(axis_projectors),
                ):
                    logger.info(
                        "axis=%s sector=%s skipped source_norm=%.8e",
                        _AXIS_NAMES[axis],
                        response_projector.label,
                        axis_phi_norm,
                    )
                    continue
                axis_sector_count += 1
                axis_source_norm[axis_pos] += axis_phi_norm
                sector_label = response_projector.label
                sector_labels.append(sector_label)
                sector_axes.append(axis)
                sector_source_norms.append(axis_phi_norm)
                sector_source_elastic_overlaps.append(complex(source_elastic_overlap))
                logger.info(
                    "axis=%s source_center=%.8e source_norm=%.8e "
                    "elastic_overlap=%.8e%+.8ej sector=%s",
                    _AXIS_NAMES[axis],
                    source_center,
                    axis_phi_norm,
                    float(np.real(source_elastic_overlap)),
                    float(np.imag(source_elastic_overlap)),
                    sector_label,
                )

                rng, response_rng = jax.random.split(rng)
                response_apply, response_params, raw_response_apply = (
                    self._make_response_ansatz(
                        example,
                        response_rng,
                        ground_params,
                        projector=response_projector,
                    )
                )
                action_ratio_apply = self._make_local_action_ratio_apply(
                    response_apply,
                    raw_response_apply,
                    response_projector,
                    ground_logpsi,
                    ground_params,
                    ground_energy,
                )
                axis_batched_data = batched_data
                source_sampler: _PreparedSourceSampler | None = None
                train_pool, source_sampler, axis_batched_data, rng = (
                    self._load_or_collect_source_pool(
                        source_sampler,
                        ground_params,
                        axis_batched_data,
                        rng,
                        ground_logpsi=ground_logpsi,
                        source_ratio_apply=source_ratio_apply,
                        axis=axis,
                        sector_label=sector_label,
                        source_center=source_center,
                        source_elastic_overlap=source_elastic_overlap,
                        split="train",
                        batches=self.lit_config.nqs_train_pool_batches,
                    )
                )
                eval_pool, source_sampler, axis_batched_data, rng = (
                    self._load_or_collect_source_pool(
                        source_sampler,
                        ground_params,
                        axis_batched_data,
                        rng,
                        ground_logpsi=ground_logpsi,
                        source_ratio_apply=source_ratio_apply,
                        axis=axis,
                        sector_label=sector_label,
                        source_center=source_center,
                        source_elastic_overlap=source_elastic_overlap,
                        split="eval",
                        batches=self.lit_config.nqs_eval_pool_batches,
                    )
                )
                logger.info(
                    "axis=%s sector=%s source_pool train=%d eval=%d",
                    _AXIS_NAMES[axis],
                    sector_label,
                    train_pool.batch_size,
                    eval_pool.batch_size,
                )
                update_batch_size = self._nqs_train_update_batch_size()
                logger.info(
                    "axis=%s sector=%s nqs_batches update=%d score=%d "
                    "action=%d eval=%d",
                    _AXIS_NAMES[axis],
                    sector_label,
                    update_batch_size,
                    self._nqs_sr_score_batch_size(update_batch_size),
                    self._nqs_action_batch_size(update_batch_size),
                    self._nqs_action_batch_size(eval_pool.batch_size),
                )

                update_step = self._make_nqs_update_step(
                    response_apply,
                    ground_params,
                    ground_logpsi,
                    ground_energy,
                    source_ratio_apply=source_ratio_apply,
                    source_norm=axis_phi_norm,
                    action_ratio_apply=action_ratio_apply,
                )
                response_params, axis_batched_data, rng = self._warm_start_axis(
                    update_step,
                    response_params,
                    train_pool,
                    axis_batched_data,
                    rng,
                    axis=axis,
                    sector_label=sector_label,
                )
                axis_start_response_params = response_params
                axis_start_batched_data = axis_batched_data
                logger.info(
                    "axis=%s sector=%s omega_points independent_response_start=true",
                    _AXIS_NAMES[axis],
                    sector_label,
                )

                local_lit = np.zeros(len(omega), dtype=np.float64)
                local_broadened = np.zeros(len(omega), dtype=np.float64)
                local_fidelity = np.zeros(len(omega), dtype=np.float64)
                local_residual_norm = np.zeros(len(omega), dtype=np.float64)
                local_action_norm = np.zeros(len(omega), dtype=np.float64)
                local_error_bound = np.zeros(len(omega), dtype=np.float64)
                local_error_d = np.zeros(len(omega), dtype=np.float64)
                local_ess = np.zeros(len(omega), dtype=np.float64)
                local_ess_fraction = np.zeros(len(omega), dtype=np.float64)
                local_estimator_mode = np.zeros(len(omega), dtype=np.int64)
                local_normalization = np.zeros(len(omega), dtype=np.complex128)
                local_correction_overlap = np.zeros(len(omega), dtype=np.complex128)
                local_response_leakage = np.zeros(len(omega), dtype=np.float64)
                local_carrier_leakage = np.zeros(len(omega), dtype=np.float64)

                for omega_pos, omega_value in enumerate(omega):
                    point_response_params = axis_start_response_params
                    point_batched_data = axis_start_batched_data
                    point_estimator_mode = 0
                    stats = None
                    for iteration in range(self.lit_config.nqs_iterations):
                        point_response_params, stats, point_batched_data, rng = (
                            update_step(
                                point_response_params,
                                train_pool,
                                jnp.asarray(float(omega_value)),
                                point_batched_data,
                                rng,
                                iteration,
                            )
                        )
                        point_estimator_mode = max(
                            point_estimator_mode,
                            int(jax.device_get(stats.estimator_mode)),
                        )
                        if (
                            self.lit_config.nqs_log_interval > 0
                            and (iteration + 1) % self.lit_config.nqs_log_interval == 0
                        ):
                            logger.info(
                                "axis=%s sector=%s omega=%.6f iter=%d "
                                "loss=%.6e fidelity=%.6f lit=%.6e "
                                "ess=%.3f mode=%d",
                                _AXIS_NAMES[axis],
                                sector_label,
                                float(omega_value),
                                iteration + 1,
                                float(stats.loss),
                                float(stats.fidelity),
                                float(stats.lit),
                                float(stats.reweight_ess_fraction),
                                int(jax.device_get(stats.estimator_mode)),
                            )
                    if stats is None:
                        stats = self._nqs_stats_chunked(
                            response_apply,
                            point_response_params,
                            ground_logpsi,
                            ground_params,
                            train_pool,
                            source_ratio_apply=source_ratio_apply,
                            source_norm=axis_phi_norm,
                            ground_energy=ground_energy,
                            omega=float(omega_value),
                            action_ratio_apply=action_ratio_apply,
                        )
                    stats = self._nqs_eval_stats(
                        response_apply,
                        point_response_params,
                        ground_logpsi,
                        ground_params,
                        eval_pool,
                        source_ratio_apply=source_ratio_apply,
                        source_norm=axis_phi_norm,
                        ground_energy=ground_energy,
                        omega=jnp.asarray(float(omega_value)),
                        action_ratio_apply=action_ratio_apply,
                    )
                    host_stats = jax.device_get(stats)
                    local_lit[omega_pos] = float(host_stats.lit)
                    local_broadened[omega_pos] = float(host_stats.broadened)
                    local_fidelity[omega_pos] = float(host_stats.fidelity)
                    local_residual_norm[omega_pos] = float(host_stats.residual_norm)
                    local_action_norm[omega_pos] = float(host_stats.action_norm)
                    local_error_bound[omega_pos] = _lit_error_monitor(
                        fidelity=float(host_stats.fidelity),
                        source_norm=float(host_stats.source_norm),
                        normalization=complex(host_stats.normalization),
                        eta=float(self.lit_config.eta),
                        error_d=float(host_stats.error_d),
                    )
                    local_error_d[omega_pos] = float(host_stats.error_d)
                    local_ess[omega_pos] = float(host_stats.reweight_ess)
                    local_ess_fraction[omega_pos] = float(
                        host_stats.reweight_ess_fraction
                    )
                    local_estimator_mode[omega_pos] = int(point_estimator_mode)
                    local_normalization[omega_pos] = complex(host_stats.normalization)
                    local_correction_overlap[omega_pos] = complex(
                        host_stats.correction_overlap
                    )
                    (
                        local_response_leakage[omega_pos],
                        local_carrier_leakage[omega_pos],
                    ) = self._maybe_projection_leakage(
                        response_apply,
                        point_response_params,
                        ground_logpsi,
                        ground_params,
                        eval_pool,
                        response_projector,
                        ground_energy=ground_energy,
                        omega=jnp.asarray(float(omega_value)),
                        batch_index=omega_pos,
                        action_ratio_apply=action_ratio_apply,
                    )

                lit[axis_pos] += local_lit
                broadened[axis_pos] += local_broadened
                fidelity[axis_pos] = np.maximum(fidelity[axis_pos], local_fidelity)
                residual_norm[axis_pos] += local_residual_norm
                action_norm[axis_pos] += local_action_norm
                source_norm[axis_pos] += axis_phi_norm
                error_bound_monitor[axis_pos] += local_error_bound
                error_d[axis_pos] += local_error_d
                reweight_ess[axis_pos] = np.maximum(reweight_ess[axis_pos], local_ess)
                reweight_ess_fraction[axis_pos] = np.maximum(
                    reweight_ess_fraction[axis_pos],
                    local_ess_fraction,
                )
                estimator_mode[axis_pos] = np.maximum(
                    estimator_mode[axis_pos],
                    local_estimator_mode,
                )
                normalization[axis_pos] += local_normalization
                correction_overlap[axis_pos] += local_correction_overlap

                sector_lit.append(local_lit)
                sector_broadened.append(local_broadened)
                sector_fidelity.append(local_fidelity)
                sector_residual_norm.append(local_residual_norm)
                sector_action_norm.append(local_action_norm)
                sector_error_bound_monitor.append(local_error_bound)
                sector_error_d.append(local_error_d)
                sector_reweight_ess.append(local_ess)
                sector_reweight_ess_fraction.append(local_ess_fraction)
                sector_estimator_mode.append(local_estimator_mode)
                sector_normalization.append(local_normalization)
                sector_correction_overlap.append(local_correction_overlap)
                sector_response_leakage.append(local_response_leakage)
                sector_carrier_leakage.append(local_carrier_leakage)
            if axis_sector_count == 0:
                logger.warning(
                    "axis=%s has no nonzero projected source sector.",
                    _AXIS_NAMES[axis],
                )

        axis_average_factor = self._axis_average_factor(axes)
        total_broadened = np.sum(broadened, axis=0) / axis_average_factor
        peaks = find_spectrum_peaks(
            omega,
            total_broadened,
            min_height_fraction=self.lit_config.peak_min_height_fraction,
        )
        output_path = self.save_path / self.lit_config.output_filename
        _save_npz(
            output_path,
            backend="nqs_lit",
            omega=omega,
            eta=self.lit_config.eta,
            axes=self.lit_config.axes,
            axis_indices=np.asarray(axes, dtype=np.int64),
            lit=lit,
            broadened=broadened,
            total_broadened=total_broadened,
            fidelity=fidelity,
            residual_norm=residual_norm,
            action_norm=action_norm,
            source_norm=source_norm,
            error_bound_monitor=error_bound_monitor,
            error_d=error_d,
            reweight_ess=reweight_ess,
            reweight_ess_fraction=reweight_ess_fraction,
            estimator_mode=estimator_mode,
            normalization=normalization,
            correction_overlap=correction_overlap,
            ground_energy=ground_energy,
            ground_checkpoint_step=checkpoint_step,
            nqs_train_pool_batches=self.lit_config.nqs_train_pool_batches,
            nqs_eval_pool_batches=self.lit_config.nqs_eval_pool_batches,
            nqs_pool_stride=self.lit_config.nqs_pool_stride,
            nqs_parallel_shared_source_pool=bool(
                self.lit_config.nqs_parallel_shared_source_pool
            ),
            nqs_sr_estimator=np.asarray(
                "double_mc_centered_jacobian_reweighted_or_direct",
                dtype=str,
            ),
            nqs_reweight_ess_fraction_min=(
                self.lit_config.nqs_reweight_ess_fraction_min
            ),
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            nqs_ground_spatial_irrep=np.asarray(
                _optional_projector_label(self.lit_config.nqs_ground_spatial_irrep)
                or "",
                dtype=str,
            ),
            nqs_response_spatial_irreps=np.asarray(
                _projector_label_list(self.lit_config.nqs_response_spatial_irreps),
                dtype=str,
            ),
            nqs_so3_quadrature_order=self.lit_config.nqs_so3_quadrature_order,
            nqs_so2_quadrature_order=self.lit_config.nqs_so2_quadrature_order,
            nqs_leakage_diagnostic_samples=(
                self.lit_config.nqs_leakage_diagnostic_samples
            ),
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
            ground_symmetry_projector=ground_projector.label,
            response_symmetry_projectors=np.asarray(sector_labels, dtype=str),
            sector_labels=np.asarray(sector_labels, dtype=str),
            sector_axes=np.asarray(sector_axes, dtype=np.int64),
            sector_source_norm=np.asarray(sector_source_norms, dtype=np.float64),
            sector_source_elastic_overlap=np.asarray(
                sector_source_elastic_overlaps,
                dtype=np.complex128,
            ),
            sector_lit=_stack_sector_arrays(sector_lit, len(omega), np.float64),
            sector_broadened=_stack_sector_arrays(
                sector_broadened,
                len(omega),
                np.float64,
            ),
            sector_fidelity=_stack_sector_arrays(
                sector_fidelity,
                len(omega),
                np.float64,
            ),
            sector_residual_norm=_stack_sector_arrays(
                sector_residual_norm,
                len(omega),
                np.float64,
            ),
            sector_action_norm=_stack_sector_arrays(
                sector_action_norm,
                len(omega),
                np.float64,
            ),
            sector_error_bound_monitor=_stack_sector_arrays(
                sector_error_bound_monitor,
                len(omega),
                np.float64,
            ),
            sector_error_d=_stack_sector_arrays(sector_error_d, len(omega), np.float64),
            sector_reweight_ess=_stack_sector_arrays(
                sector_reweight_ess,
                len(omega),
                np.float64,
            ),
            sector_reweight_ess_fraction=_stack_sector_arrays(
                sector_reweight_ess_fraction,
                len(omega),
                np.float64,
            ),
            sector_estimator_mode=_stack_sector_arrays(
                sector_estimator_mode,
                len(omega),
                np.int64,
            ),
            sector_normalization=_stack_sector_arrays(
                sector_normalization,
                len(omega),
                np.complex128,
            ),
            sector_correction_overlap=_stack_sector_arrays(
                sector_correction_overlap,
                len(omega),
                np.complex128,
            ),
            sector_response_leakage=_stack_sector_arrays(
                sector_response_leakage,
                len(omega),
                np.float64,
            ),
            sector_carrier_leakage=_stack_sector_arrays(
                sector_carrier_leakage,
                len(omega),
                np.float64,
            ),
            axis_average_factor=axis_average_factor,
            peak_energies=np.asarray([peak.energy for peak in peaks]),
            peak_intensities=np.asarray([peak.intensity for peak in peaks]),
            peak_indices=np.asarray([peak.index for peak in peaks]),
        )
        self._log_nqs_summary(str(output_path), peaks, fidelity)

    def _validate_config(self) -> None:
        scan_parallel = self.lit_config.scan_parallel.lower()
        if scan_parallel not in ("auto", "off", "false", "none", "0", "local_devices"):
            msg = (
                "lit.scan_parallel must be one of 'auto', 'off', or "
                f"'local_devices', got {self.lit_config.scan_parallel!r}."
            )
            raise ValueError(msg)
        _omega_grid_from_config(self.lit_config)
        if self.lit_config.scan_parallel_workers < 0:
            msg = "lit.scan_parallel_workers must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.scan_parallel_procs_per_device < 1:
            msg = "lit.scan_parallel_procs_per_device must be positive."
            raise ValueError(msg)
        if self.lit_config.scan_parallel_min_points_per_worker < 1:
            msg = "lit.scan_parallel_min_points_per_worker must be positive."
            raise ValueError(msg)
        self._validate_chunk_config()
        self._validate_symmetry_config()

    def _validate_symmetry_config(self) -> None:
        if self.lit_config.nqs_symmetry_tolerance <= 0.0:
            msg = "lit.nqs_symmetry_tolerance must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_projection_eps <= 0.0:
            msg = "lit.nqs_projection_eps must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_projection_chunk_size < 1:
            msg = "lit.nqs_projection_chunk_size must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_so3_quadrature_order < 1:
            msg = "lit.nqs_so3_quadrature_order must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_so2_quadrature_order < 1:
            msg = "lit.nqs_so2_quadrature_order must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_source_norm_min < 0.0:
            msg = "lit.nqs_source_norm_min must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_leakage_diagnostic_samples < 0:
            msg = "lit.nqs_leakage_diagnostic_samples must be nonnegative."
            raise ValueError(msg)
        self._validate_direct_psi_config()
        spin_mode = self.lit_config.nqs_spin_projector.lower()
        allowed_spin_modes = {
            "auto",
            "on",
            "true",
            "1",
            "off",
            "false",
            "none",
            "0",
            "identity",
        }
        if spin_mode not in allowed_spin_modes:
            msg = (
                "lit.nqs_spin_projector must be auto/on/off/identity, got "
                f"{self.lit_config.nqs_spin_projector!r}."
            )
            raise ValueError(msg)

    def _validate_direct_psi_config(self) -> None:
        if self.lit_config.nqs_reweight_ess_fraction_min < 0.0:
            msg = "lit.nqs_reweight_ess_fraction_min must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_direct_psi_burn_in < 0:
            msg = "lit.nqs_direct_psi_burn_in must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_direct_psi_batches < 1:
            msg = "lit.nqs_direct_psi_batches must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_direct_psi_stride < 1:
            msg = "lit.nqs_direct_psi_stride must be positive."
            raise ValueError(msg)

    def _validate_chunk_config(self) -> None:
        if self.lit_config.nqs_train_update_batch_size < 0:
            msg = "lit.nqs_train_update_batch_size must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_sr_score_batch_size < 0:
            msg = "lit.nqs_sr_score_batch_size must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_eval_batch_size < 0:
            msg = "lit.nqs_eval_batch_size must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_projected_sr_score_batch_cap < 1:
            msg = "lit.nqs_projected_sr_score_batch_cap must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_projected_action_batch_cap < 1:
            msg = "lit.nqs_projected_action_batch_cap must be positive."
            raise ValueError(msg)

    def _projection_chunk_size(self) -> int:
        return _projection_chunk_size_from_config(self.lit_config)

    def _make_response_ansatz(
        self,
        example,
        response_rng,
        ground_params,
        *,
        projector: SymmetryProjector,
    ):
        response = MolecularResponseFermiNet(
            nspins=_two_spin_tuple(self.system_config.electron_spins),
            ndets=int(self.lit_config.nqs_response_ndets),
            hidden_dims_single=tuple(self.lit_config.nqs_response_hidden_dims_single),
            hidden_dims_double=tuple(self.lit_config.nqs_response_hidden_dims_double),
            use_last_layer=bool(self.lit_config.nqs_response_use_last_layer),
            envelope=self.lit_config.nqs_response_envelope,
            orbitals_spin_split=bool(self.lit_config.nqs_response_orbitals_spin_split),
        )
        response_params = response.init(response_rng, example)
        response_params = _copy_matching_parameters(response_params, ground_params)
        response_apply = projected_log_apply(
            response.apply,
            projector,
            eps=float(self.lit_config.nqs_projection_eps),
            chunk_size=_projection_chunk_size_from_config(self.lit_config),
        )
        return response_apply, response_params, response.apply

    def _make_local_action_ratio_apply(
        self,
        response_apply,
        raw_response_apply,
        projector: SymmetryProjector,
        ground_logpsi,
        ground_params,
        ground_energy: float,
    ):
        eta = self.lit_config.eta
        projection_eps = float(self.lit_config.nqs_projection_eps)
        chunk_size = _projection_chunk_size_from_config(self.lit_config)

        if projector.is_identity:

            def identity_action_ratio(response_params, data, omega):
                return local_action_ratio(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    data,
                    ground_energy=ground_energy,
                    omega=omega,
                    eta=eta,
                )

            return identity_action_ratio

        def projected_action_ratio(response_params, data, omega):
            ground_value = jnp.exp(ground_logpsi(ground_params, data))
            safe_ground = jnp.exp(safe_complex_log(ground_value, eps=projection_eps))

            def response_value(local_data):
                return jnp.exp(raw_response_apply(response_params, local_data))

            def carrier_value(local_data):
                action_ratio, _, _ = local_action_ratio(
                    raw_response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    local_data,
                    ground_energy=ground_energy,
                    omega=omega,
                    eta=eta,
                )
                return action_ratio * jnp.exp(ground_logpsi(ground_params, local_data))

            response_value_projected = project_value(
                response_value,
                data,
                projector,
                chunk_size=chunk_size,
            )
            carrier_value_projected = project_value(
                carrier_value,
                data,
                projector,
                chunk_size=chunk_size,
            )
            response_ratio = response_value_projected / safe_ground
            action_ratio = carrier_value_projected / safe_ground
            safe_response_ratio = jnp.where(
                jnp.abs(response_ratio) > projection_eps,
                response_ratio,
                jnp.asarray(projection_eps, dtype=response_ratio.real.dtype) + 0j,
            )
            shift = jnp.asarray(
                omega,
                dtype=response_ratio.real.dtype,
            ) + 1j * jnp.asarray(eta, dtype=response_ratio.real.dtype)
            local_energy = (
                action_ratio / safe_response_ratio
                + jnp.asarray(ground_energy, dtype=action_ratio.dtype)
                + shift
            )
            return action_ratio, response_ratio, local_energy

        return projected_action_ratio

    def _call_action_ratio_apply(
        self,
        action_ratio_apply,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        data,
        *,
        ground_energy: float,
        omega,
    ):
        if action_ratio_apply is not None:
            return action_ratio_apply(response_params, data, omega)
        return local_action_ratio(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            data,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
        )

    def _prepare_source_sampler(
        self,
        sampler,
        batched_data,
        ground_params,
        ground_logpsi,
        rng,
        *,
        source_ratio_apply: SourceRatioApply,
    ):
        source_plan = SamplePlan(
            self._make_source_log_amplitude(
                source_ratio_apply,
                ground_logpsi,
            ),
            {"electrons": sampler},
        )
        rng, source_rng = jax.random.split(rng)
        source_state = source_plan.init(batched_data, source_rng)
        source_plan = _JittedSamplePlan(source_plan, batched_data)
        batched_data, source_state, rng = _run_sample_steps(
            source_plan,
            ground_params,
            batched_data,
            source_state,
            rng,
            self.lit_config.nqs_source_burn_in,
        )
        return source_plan, source_state, batched_data, rng

    def _collect_sample_pool(
        self,
        sample_plan: SamplePlan,
        params,
        batched_data,
        sampler_state,
        rng,
        *,
        batches: int,
        stride: int | None = None,
    ):
        pool = []
        stride = max(
            1,
            int(self.lit_config.nqs_pool_stride if stride is None else stride),
        )
        for _ in range(max(1, int(batches))):
            rng, sample_rng = jax.random.split(rng)
            batched_data, sampler_state = sample_plan.run_steps(
                params,
                batched_data,
                sampler_state,
                sample_rng,
                stride,
            )
            pool.append(batched_data)
        return (
            _host_batched_data(_concat_batched_data(pool)),
            batched_data,
            sampler_state,
            rng,
        )

    def _load_or_collect_source_pool(
        self,
        source_sampler: _PreparedSourceSampler | None,
        params,
        batched_data,
        rng,
        *,
        ground_logpsi,
        source_ratio_apply: SourceRatioApply,
        axis: int,
        sector_label: str,
        source_center: float,
        source_elastic_overlap: complex,
        split: str,
        batches: int,
    ):
        pool_path = self._source_pool_path(axis, sector_label, split)
        metadata = self._source_pool_metadata(
            axis,
            sector_label,
            source_center,
            source_elastic_overlap=source_elastic_overlap,
        )
        if self._source_cache_enabled(pool_path):
            with _file_lock(_cache_lock_path(pool_path)):
                loaded = self._try_load_source_pool(
                    pool_path,
                    batched_data,
                    metadata,
                    axis=axis,
                    split=split,
                )
                if loaded is not None:
                    return loaded, source_sampler, batched_data, rng
                source_sampler, batched_data, rng = self._ensure_source_sampler(
                    source_sampler,
                    params,
                    batched_data,
                    rng,
                    ground_logpsi=ground_logpsi,
                    source_ratio_apply=source_ratio_apply,
                )
                pool, batched_data, sampler_state, rng = self._collect_sample_pool(
                    source_sampler.plan,
                    params,
                    batched_data,
                    source_sampler.state,
                    rng,
                    batches=batches,
                )
                source_sampler = _PreparedSourceSampler(
                    plan=source_sampler.plan,
                    state=sampler_state,
                )
                if self.lit_config.nqs_save_source_pool:
                    _save_batched_pool(pool_path, pool, metadata=metadata)
                    logger.info(
                        "Saved %s source pool for axis=%s to %s",
                        split,
                        _AXIS_NAMES[axis],
                        pool_path,
                    )
                return pool, source_sampler, batched_data, rng

        source_sampler, batched_data, rng = self._ensure_source_sampler(
            source_sampler,
            params,
            batched_data,
            rng,
            ground_logpsi=ground_logpsi,
            source_ratio_apply=source_ratio_apply,
        )
        pool, batched_data, sampler_state, rng = self._collect_sample_pool(
            source_sampler.plan,
            params,
            batched_data,
            source_sampler.state,
            rng,
            batches=batches,
        )
        source_sampler = _PreparedSourceSampler(
            plan=source_sampler.plan,
            state=sampler_state,
        )
        return pool, source_sampler, batched_data, rng

    def _ensure_source_sampler(
        self,
        source_sampler: _PreparedSourceSampler | None,
        ground_params,
        batched_data,
        rng,
        *,
        ground_logpsi,
        source_ratio_apply: SourceRatioApply,
    ):
        if source_sampler is not None:
            return source_sampler, batched_data, rng
        source_plan, source_state, batched_data, rng = self._prepare_source_sampler(
            self.sampler,
            batched_data,
            ground_params,
            ground_logpsi,
            rng,
            source_ratio_apply=source_ratio_apply,
        )
        return (
            _PreparedSourceSampler(plan=source_plan, state=source_state),
            batched_data,
            rng,
        )

    def _try_load_source_pool(
        self,
        pool_path: UPath,
        batched_data,
        metadata: dict[str, float],
        *,
        axis: int,
        split: str,
    ):
        if not self.lit_config.nqs_reuse_source_pool or not pool_path.exists():
            return None
        try:
            pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning(
                "Ignoring incompatible %s source pool %s: %s",
                split,
                pool_path,
                exc,
            )
            return None
        logger.info(
            "Loaded %s source pool for axis=%s from %s",
            split,
            _AXIS_NAMES[axis],
            pool_path,
        )
        return pool

    def _source_cache_enabled(self, path: UPath) -> bool:
        if self.lit_config.nqs_save_source_pool:
            return True
        return bool(self.lit_config.nqs_reuse_source_pool and path.exists())

    def _source_pool_root(self) -> UPath:
        return (
            UPath(self.lit_config.nqs_source_pool_dir)
            if self.lit_config.nqs_source_pool_dir
            else self.save_path / "source_pools"
        )

    def _source_pool_path(self, axis: int, sector_label: str, split: str) -> UPath:
        sector_slug = _safe_label(sector_label)
        return self._source_pool_root() / (
            f"axis_{_AXIS_NAMES[axis]}_{sector_slug}_{split}.npz"
        )

    def _source_center_path(self, axis: int) -> UPath:
        return self._source_pool_root() / f"axis_{_AXIS_NAMES[axis]}_source_center.npz"

    def _source_norm_path(self, axis: int, sector_label: str) -> UPath:
        sector_slug = _safe_label(sector_label)
        return self._source_pool_root() / (
            f"axis_{_AXIS_NAMES[axis]}_{sector_slug}_source_norm.npz"
        )

    def _source_elastic_path(self, axis: int, sector_label: str) -> UPath:
        sector_slug = _safe_label(sector_label)
        return self._source_pool_root() / (
            f"axis_{_AXIS_NAMES[axis]}_{sector_slug}_source_elastic_overlap.npz"
        )

    def _source_center_metadata(self, axis: int) -> dict[str, float]:
        return {
            "axis": float(axis),
            "symmetry_projectors": float(bool(self.lit_config.nqs_symmetry_projectors)),
        }

    def _source_norm_metadata(
        self,
        axis: int,
        sector_label: str,
        source_center: float,
        source_elastic_overlap: complex,
    ) -> dict[str, float]:
        metadata = self._source_pool_metadata(
            axis,
            sector_label,
            source_center,
            source_elastic_overlap=source_elastic_overlap,
        )
        metadata["source_norm_min"] = float(self.lit_config.nqs_source_norm_min)
        return metadata

    def _source_elastic_metadata(
        self,
        axis: int,
        sector_label: str,
        source_center: float,
    ) -> dict[str, float]:
        return {
            "axis": float(axis),
            "sector_hash": float(_stable_label_hash(sector_label)),
            "source_projection_version": 2.0,
            "source_center": float(source_center),
            "symmetry_projectors": float(bool(self.lit_config.nqs_symmetry_projectors)),
        }

    def _source_pool_metadata(
        self,
        axis: int,
        sector_label: str,
        source_center: float,
        *,
        source_elastic_overlap: complex,
    ) -> dict[str, float]:
        return {
            "axis": float(axis),
            "sector_hash": float(_stable_label_hash(sector_label)),
            "source_projection_version": 2.0,
            "source_center": float(source_center),
            "source_elastic_overlap_real": float(np.real(source_elastic_overlap)),
            "source_elastic_overlap_imag": float(np.imag(source_elastic_overlap)),
            "source_floor": float(self.lit_config.nqs_source_floor),
            "symmetry_projectors": float(bool(self.lit_config.nqs_symmetry_projectors)),
        }

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        fallback_data,
        rng,
        *,
        axis: int,
        sector_label: str,
    ):
        if (
            self.lit_config.nqs_warm_start_omega is None
            or self.lit_config.nqs_warm_start_iterations <= 0
        ):
            return response_params, fallback_data, rng
        stats = None
        for iteration in range(self.lit_config.nqs_warm_start_iterations):
            response_params, stats, fallback_data, rng = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(self.lit_config.nqs_warm_start_omega)),
                fallback_data,
                rng,
                iteration,
            )
        if stats is not None:
            logger.info(
                "axis=%s warm_start omega=%.6f iterations=%d "
                "sector=%s fidelity=%.6f lit=%.6e ess=%.3f",
                _AXIS_NAMES[axis],
                float(self.lit_config.nqs_warm_start_omega),
                self.lit_config.nqs_warm_start_iterations,
                sector_label,
                float(stats.fidelity),
                float(stats.lit),
                float(stats.reweight_ess_fraction),
            )
        return response_params, fallback_data, rng

    def _resolve_nqs_ground_state(self, example, ground_rng):
        fallback_ground_params = self.wf.init_params(example, ground_rng)
        checkpoint_path = self.lit_config.nqs_checkpoint_path or str(self.restore_path)
        try:
            checkpoint_step, ground_params = restore_params_from_checkpoint(
                checkpoint_path,
                fallback_ground_params,
            )
            logger.info(
                "Restored ground-state parameters from %s at step %d",
                checkpoint_path,
                checkpoint_step,
            )
        except FileNotFoundError:
            if not self.lit_config.nqs_allow_untrained_ground:
                raise
            checkpoint_step = -1
            ground_params = fallback_ground_params
            logger.warning(
                "No ground checkpoint found at %s; using untrained ground "
                "parameters because lit.nqs_allow_untrained_ground=true.",
                checkpoint_path,
            )
        # The trained ground-state NQS is the carrier distribution for MCMC.
        # Applying the numerical ground-sector projector inside every
        # Metropolis proposal would multiply the cost of all source/energy
        # sampling by the spatial quadrature size.  Source and response
        # sectors are still projected explicitly where they enter the LIT
        # equations.
        return checkpoint_step, ground_params, self._ground_complex_logpsi

    def _ground_complex_logpsi(self, params, data) -> jnp.ndarray:
        phase, log_abs = self.wf.phase_logpsi(params, data)
        return log_abs + 1j * _phase_angle(phase, log_abs.dtype)

    def _make_ground_projector(self) -> SymmetryProjector:
        if not self.lit_config.nqs_symmetry_projectors:
            return SymmetryProjector(
                spatial=identity_spatial_projector("c1"),
                spin=make_spin_projector(
                    _two_spin_tuple(self.system_config.electron_spins),
                    target_s=None,
                    enabled=False,
                ),
                label="identity",
            )
        atoms, charges = self._nuclear_arrays()
        spatial = make_ground_spatial_projector(
            atoms,
            charges,
            mode=self.lit_config.nqs_ground_spatial_projector,
            irrep_label=_optional_projector_label(
                self.lit_config.nqs_ground_spatial_irrep
            ),
            tolerance=float(self.lit_config.nqs_symmetry_tolerance),
            so3_quadrature_order=int(self.lit_config.nqs_so3_quadrature_order),
            so2_quadrature_order=int(self.lit_config.nqs_so2_quadrature_order),
        )
        spin = make_spin_projector(
            _two_spin_tuple(self.system_config.electron_spins),
            target_s=self._ground_spin_target(),
            enabled=self._spin_projection_enabled(),
        )
        return SymmetryProjector(
            spatial=spatial,
            spin=spin,
            label=f"ground:{spatial.label},{spin.label}",
        )

    def _make_response_projectors(self, axis: int) -> tuple[SymmetryProjector, ...]:
        if not self.lit_config.nqs_symmetry_projectors:
            return (
                SymmetryProjector(
                    spatial=identity_spatial_projector("c1"),
                    spin=make_spin_projector(
                        _two_spin_tuple(self.system_config.electron_spins),
                        target_s=None,
                        enabled=False,
                    ),
                    label=f"axis_{_AXIS_NAMES[axis]}:identity",
                ),
            )
        atoms, charges = self._nuclear_arrays()
        spatial_projectors = make_dipole_spatial_projectors(
            atoms,
            charges,
            mode=self.lit_config.nqs_spatial_projector,
            axis=axis,
            irrep_labels=_projector_label_list(
                self.lit_config.nqs_response_spatial_irreps
            ),
            tolerance=float(self.lit_config.nqs_symmetry_tolerance),
            so3_quadrature_order=int(self.lit_config.nqs_so3_quadrature_order),
            so2_quadrature_order=int(self.lit_config.nqs_so2_quadrature_order),
        )
        spin = make_spin_projector(
            _two_spin_tuple(self.system_config.electron_spins),
            target_s=self._response_spin_target(),
            enabled=self._spin_projection_enabled(),
        )
        return tuple(
            SymmetryProjector(
                spatial=spatial,
                spin=spin,
                label=f"axis_{_AXIS_NAMES[axis]}:{spatial.label},{spin.label}",
            )
            for spatial in spatial_projectors
        )

    def _skip_source_sector(
        self,
        source_norm: float,
        projector: SymmetryProjector,
        *,
        candidate_count: int,
    ) -> bool:
        if candidate_count <= 1:
            return False
        del projector
        return float(source_norm) <= float(self.lit_config.nqs_source_norm_min)

    def _axis_average_factor(self, axes: tuple[int, ...]) -> float:
        if not self.lit_config.isotropic_average:
            return 1.0
        return 3.0 if set(axes) == {0, 1, 2} else 1.0

    def _ground_spin_target(self) -> float | None:
        if self.lit_config.nqs_ground_spin is not None:
            return float(self.lit_config.nqs_ground_spin)
        return _minimum_compatible_spin(
            _two_spin_tuple(self.system_config.electron_spins)
        )

    def _response_spin_target(self) -> float | None:
        if self.lit_config.nqs_response_spin is not None:
            return float(self.lit_config.nqs_response_spin)
        return self._ground_spin_target()

    def _spin_projection_enabled(self) -> bool:
        mode = self.lit_config.nqs_spin_projector.lower()
        return mode not in {"off", "false", "none", "0", "identity"}

    def _nuclear_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        atoms = np.asarray(
            [atom.coords_array for atom in self.system_config.atoms],
            dtype=np.float64,
        )
        charges = np.asarray([atom.charge for atom in self.system_config.atoms])
        return atoms, charges

    def _resolve_ground_energy(
        self,
        ground_logpsi,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
    ):
        if self.lit_config.nqs_ground_energy is not None:
            return (
                float(self.lit_config.nqs_ground_energy),
                batched_data,
                sampler_state,
                rng,
            )
        energy_values = []
        for _ in range(max(1, self.lit_config.nqs_energy_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            local = jax.vmap(
                lambda one: ground_local_energy(ground_logpsi, ground_params, one),
                in_axes=(batched_data.vmap_axis,),
            )(batched_data.data)
            energy_values.append(float(jnp.mean(local)))
        return float(np.mean(energy_values)), batched_data, sampler_state, rng

    def _estimate_source_center(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
    ):
        mean_values = []
        for _ in range(max(1, self.lit_config.nqs_source_center_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            dipole = jax.vmap(
                lambda one: molecular_electronic_dipole(one, axis),
                in_axes=(batched_data.vmap_axis,),
            )(batched_data.data)
            mean_values.append(float(jnp.mean(dipole)))
        mean = float(np.mean(mean_values))
        center = mean
        if self.lit_config.nqs_source_center_override is not None:
            center = float(self.lit_config.nqs_source_center_override)
        elif self._ground_dipole_component_forbidden(axis):
            center = 0.0
        return center, batched_data, sampler_state, rng

    def _load_or_estimate_source_center(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
    ):
        path = self._source_center_path(axis)
        if not self._source_cache_enabled(path):
            return self._estimate_source_center(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                axis=axis,
            )
        metadata = self._source_center_metadata(axis)
        with _file_lock(_cache_lock_path(path)):
            if self.lit_config.nqs_reuse_source_pool and path.exists():
                try:
                    center = float(
                        _load_scalar_cache(
                            path,
                            "source_center",
                            metadata=metadata,
                        )
                    )
                    logger.info(
                        "Loaded source center for axis=%s from %s",
                        _AXIS_NAMES[axis],
                        path,
                    )
                    return center, batched_data, sampler_state, rng
                except (KeyError, ValueError, OSError) as exc:
                    logger.warning(
                        "Ignoring incompatible source center cache %s: %s",
                        path,
                        exc,
                    )
            center, batched_data, sampler_state, rng = self._estimate_source_center(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                axis=axis,
            )
            if self.lit_config.nqs_save_source_pool:
                _save_scalar_cache(
                    path,
                    source_center=float(center),
                    metadata=metadata,
                )
                logger.info(
                    "Saved source center for axis=%s to %s",
                    _AXIS_NAMES[axis],
                    path,
                )
            return center, batched_data, sampler_state, rng

    def _estimate_source_elastic_overlap(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        source_ratio_apply: SourceRatioApply,
    ):
        overlap_values = []
        source_overlap_stat = self._make_source_ratio_stat(
            source_ratio_apply,
            batched_data,
            squared_abs=False,
        )
        for _ in range(max(1, self.lit_config.nqs_source_center_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            source_overlap = source_overlap_stat(_device_put_batched_data(batched_data))
            overlap_values.append(complex(jax.device_get(source_overlap)))
        overlap = complex(np.mean(overlap_values))
        return overlap, batched_data, sampler_state, rng

    def _load_or_estimate_source_elastic_overlap(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
        sector_label: str,
        source_center: float,
        source_ratio_apply: SourceRatioApply,
        response_projector: SymmetryProjector,
        ground_projector: SymmetryProjector,
    ):
        if not _same_symmetry_sector(response_projector, ground_projector):
            return 0.0 + 0.0j, batched_data, sampler_state, rng
        path = self._source_elastic_path(axis, sector_label)
        if not self._source_cache_enabled(path):
            return self._estimate_source_elastic_overlap(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                source_ratio_apply=source_ratio_apply,
            )
        metadata = self._source_elastic_metadata(axis, sector_label, source_center)
        with _file_lock(_cache_lock_path(path)):
            if self.lit_config.nqs_reuse_source_pool and path.exists():
                try:
                    overlap = _load_scalar_cache(
                        path,
                        "source_elastic_overlap",
                        metadata=metadata,
                    )
                    logger.info(
                        "Loaded source elastic overlap for axis=%s sector=%s from %s",
                        _AXIS_NAMES[axis],
                        sector_label,
                        path,
                    )
                    return complex(overlap), batched_data, sampler_state, rng
                except (KeyError, ValueError, OSError) as exc:
                    logger.warning(
                        "Ignoring incompatible source elastic cache %s: %s",
                        path,
                        exc,
                    )
            overlap, batched_data, sampler_state, rng = (
                self._estimate_source_elastic_overlap(
                    ground_params,
                    batched_data,
                    sampler_state,
                    sample_plan,
                    rng,
                    source_ratio_apply=source_ratio_apply,
                )
            )
            if self.lit_config.nqs_save_source_pool:
                _save_scalar_cache(
                    path,
                    source_elastic_overlap=complex(overlap),
                    metadata=metadata,
                )
                logger.info(
                    "Saved source elastic overlap for axis=%s sector=%s to %s",
                    _AXIS_NAMES[axis],
                    sector_label,
                    path,
                )
            return overlap, batched_data, sampler_state, rng

    def _ground_dipole_component_forbidden(self, axis: int) -> bool:
        if not self.lit_config.nqs_symmetry_projectors:
            return False
        projector = self._make_ground_projector()
        if projector.spatial.is_identity:
            return False
        component_norm = _spatial_projector_vector_component_norm(
            projector.spatial,
            axis,
        )
        return component_norm <= float(self.lit_config.nqs_symmetry_tolerance)

    def _make_source_ratio_stat(
        self,
        source_ratio_apply: SourceRatioApply,
        batched_data: BatchedData,
        *,
        squared_abs: bool,
    ):
        def source_ratio_stat(data):
            source_ratio = jax.vmap(
                source_ratio_apply,
                in_axes=(data.vmap_axis,),
            )(data.data)
            if squared_abs:
                value = jnp.mean(jnp.abs(source_ratio) ** 2)
            else:
                value = jnp.mean(source_ratio)
            return parallel_jax.pmean(value)

        return parallel_jax.jit_sharded(
            source_ratio_stat,
            in_specs=(batched_data.partition_spec,),
            out_specs=parallel_jax.SHARE_PARTITION,
            check_vma=True,
        )

    def _estimate_source_norm(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        source_ratio_apply: SourceRatioApply,
    ):
        norm_values = []
        source_norm_stat = self._make_source_ratio_stat(
            source_ratio_apply,
            batched_data,
            squared_abs=True,
        )
        for _ in range(max(1, self.lit_config.nqs_source_center_steps)):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )
            source_ratio_norm = source_norm_stat(_device_put_batched_data(batched_data))
            norm_values.append(float(jax.device_get(source_ratio_norm)))
        norm = float(max(float(np.mean(norm_values)), 1e-12))
        if self.lit_config.nqs_source_norm_override is not None:
            norm = float(self.lit_config.nqs_source_norm_override)
        return norm, batched_data, sampler_state, rng

    def _load_or_estimate_source_norm(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
        sector_label: str,
        source_center: float,
        source_elastic_overlap: complex,
        source_ratio_apply: SourceRatioApply,
    ):
        path = self._source_norm_path(axis, sector_label)
        if not self._source_cache_enabled(path):
            return self._estimate_source_norm(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                source_ratio_apply=source_ratio_apply,
            )
        metadata = self._source_norm_metadata(
            axis,
            sector_label,
            source_center,
            source_elastic_overlap,
        )
        with _file_lock(_cache_lock_path(path)):
            if self.lit_config.nqs_reuse_source_pool and path.exists():
                try:
                    norm = float(
                        _load_scalar_cache(
                            path,
                            "source_norm",
                            metadata=metadata,
                        )
                    )
                    logger.info(
                        "Loaded source norm for axis=%s sector=%s from %s",
                        _AXIS_NAMES[axis],
                        sector_label,
                        path,
                    )
                    return norm, batched_data, sampler_state, rng
                except (KeyError, ValueError, OSError) as exc:
                    logger.warning(
                        "Ignoring incompatible source norm cache %s: %s",
                        path,
                        exc,
                    )
            norm, batched_data, sampler_state, rng = self._estimate_source_norm(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                source_ratio_apply=source_ratio_apply,
            )
            if self.lit_config.nqs_save_source_pool:
                _save_scalar_cache(path, source_norm=float(norm), metadata=metadata)
                logger.info(
                    "Saved source norm for axis=%s sector=%s to %s",
                    _AXIS_NAMES[axis],
                    sector_label,
                    path,
                )
            return norm, batched_data, sampler_state, rng

    def _make_source_log_amplitude(
        self,
        source_ratio_apply: SourceRatioApply,
        ground_logpsi,
    ):
        floor = float(self.lit_config.nqs_source_floor)

        def log_amplitude(params, data):
            source = source_ratio_apply(data)
            return ground_logpsi(params, data) + jnp.log(
                jnp.maximum(jnp.abs(source), floor)
            )

        return log_amplitude

    def _make_projected_source_ratio(
        self,
        ground_logpsi,
        ground_params,
        *,
        axis: int,
        source_center: float,
        projector: SymmetryProjector,
        elastic_overlap: complex,
    ) -> SourceRatioApply:
        eps = float(self.lit_config.nqs_projection_eps)

        def source_ratio(data):
            ground_value = jnp.exp(ground_logpsi(ground_params, data))

            def source_value(local_data):
                dipole = molecular_electronic_dipole(local_data, axis)
                centered = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
                return centered * jnp.exp(ground_logpsi(ground_params, local_data))

            projected_source = project_value(
                source_value,
                data,
                projector,
                chunk_size=_projection_chunk_size_from_config(self.lit_config),
            )
            safe_ground = jnp.exp(safe_complex_log(ground_value, eps=eps))
            elastic = jnp.asarray(elastic_overlap, dtype=projected_source.dtype)
            return projected_source / safe_ground - elastic

        return source_ratio

    def _make_nqs_update_step(  # noqa: C901
        self,
        response_apply,
        ground_params,
        ground_logpsi,
        ground_energy: float,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        action_ratio_apply,
    ):
        @jax.jit
        def reweighted_score_chunk(response_params, batched_data, omega):
            return self._source_sampled_action_scores(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )

        @jax.jit
        def reweighted_finish(response_params, score, ratio, source_weight):
            updates = self._reweighted_double_mc_sr_updates_from_scores(
                response_params,
                score,
                ratio,
                source_weight,
            )
            return _apply_updates(response_params, updates)

        @jax.jit
        def reweighted_update(response_params, batched_data, omega):
            _, stats = self._nqs_loss(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                source_ratio_apply=source_ratio_apply,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )
            updates = self._reweighted_double_mc_sr_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )
            response_params = _apply_updates(response_params, updates)
            loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
            return response_params, stats._replace(loss=loss)

        @jax.jit
        def direct_source_chunk(response_params, batched_data, omega):
            return self._source_sampled_action_ratios(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )

        @jax.jit
        def direct_score_chunk(response_params, batched_data, omega):
            return self._source_sampled_action_scores(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )

        @jax.jit
        def direct_finish(
            response_params,
            source_ratio,
            source_weight,
            score,
            ratio,
        ):
            updates = self._direct_double_mc_sr_updates_from_scores(
                response_params,
                source_ratio,
                source_weight,
                score,
                ratio,
            )
            return _apply_updates(response_params, updates)

        @jax.jit
        def direct_update(
            response_params,
            source_batched_data,
            psi_batched_data,
            omega,
        ):
            stats = self._nqs_direct_stats(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                source_batched_data,
                psi_batched_data,
                source_ratio_apply=source_ratio_apply,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )
            updates = self._direct_double_mc_sr_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                source_batched_data,
                psi_batched_data,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            )
            response_params = _apply_updates(response_params, updates)
            loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
            return response_params, stats._replace(loss=loss)

        def update(
            response_params,
            batched_data,
            omega,
            fallback_data,
            rng,
            batch_index: int = 0,
        ):
            update_batch = _cyclic_batched_data_chunk(
                batched_data,
                self._nqs_train_update_batch_size(),
                batch_index,
            )
            original_response_params = response_params
            if self._use_nqs_sr_score_chunks(update_batch.batch_size):
                return self._run_nqs_score_chunked_update(
                    original_response_params,
                    update_batch,
                    omega,
                    fallback_data,
                    rng,
                    response_apply=response_apply,
                    ground_logpsi=ground_logpsi,
                    ground_params=ground_params,
                    ground_energy=ground_energy,
                    source_ratio_apply=source_ratio_apply,
                    source_norm=source_norm,
                    action_ratio_apply=action_ratio_apply,
                    reweighted_score_chunk=reweighted_score_chunk,
                    reweighted_finish=reweighted_finish,
                    direct_source_chunk=direct_source_chunk,
                    direct_score_chunk=direct_score_chunk,
                    direct_finish=direct_finish,
                )

            candidate_response_params, stats = reweighted_update(
                response_params,
                update_batch,
                omega,
            )
            if self._should_use_direct_psi(stats):
                psi_pool, fallback_data, rng = self._collect_direct_psi_pool(
                    response_apply,
                    original_response_params,
                    ground_logpsi,
                    ground_params,
                    fallback_data,
                    rng,
                    ground_energy=ground_energy,
                    omega=omega,
                    batches=self.lit_config.nqs_direct_psi_batches,
                    action_ratio_apply=action_ratio_apply,
                )
                response_params, stats = direct_update(
                    original_response_params,
                    update_batch,
                    psi_pool,
                    omega,
                )
            else:
                response_params = candidate_response_params
            return response_params, stats, fallback_data, rng

        return update

    def _run_nqs_score_chunked_update(
        self,
        response_params,
        update_batch,
        omega,
        fallback_data,
        rng,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        ground_energy: float,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        action_ratio_apply,
        reweighted_score_chunk,
        reweighted_finish,
        direct_source_chunk,
        direct_score_chunk,
        direct_finish,
    ):
        stats = self._nqs_chunked_reweighted_update_stats(
            response_params,
            update_batch,
            omega,
            response_apply=response_apply,
            ground_logpsi=ground_logpsi,
            ground_params=ground_params,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            action_ratio_apply=action_ratio_apply,
        )
        if not self._should_use_direct_psi(stats):
            score, ratio, source_weight = self._nqs_sr_score_chunks(
                reweighted_score_chunk,
                response_params,
                update_batch,
                omega,
            )
            response_params = reweighted_finish(
                response_params,
                score,
                ratio,
                source_weight,
            )
            return response_params, stats, fallback_data, rng

        psi_pool, fallback_data, rng = self._collect_direct_psi_pool(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            fallback_data,
            rng,
            ground_energy=ground_energy,
            omega=omega,
            batches=self.lit_config.nqs_direct_psi_batches,
            action_ratio_apply=action_ratio_apply,
        )
        source_ratio, source_weight = self._nqs_sr_source_ratio_chunks(
            direct_source_chunk,
            response_params,
            update_batch,
            omega,
        )
        score, ratio, _ = self._nqs_sr_score_chunks(
            direct_score_chunk,
            response_params,
            psi_pool,
            omega,
        )
        source_stats = self._nqs_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            update_batch,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            chunk_size=self._nqs_action_batch_size(update_batch.batch_size),
            local_device_chunks=True,
            action_ratio_apply=action_ratio_apply,
        )
        stats = self._nqs_direct_stats_from_ratio(
            source_stats,
            ratio,
            source_norm=source_norm,
        )
        response_params = direct_finish(
            response_params,
            source_ratio,
            source_weight,
            score,
            ratio,
        )
        return response_params, stats, fallback_data, rng

    def _nqs_chunked_reweighted_update_stats(
        self,
        response_params,
        batched_data,
        omega,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        action_ratio_apply,
    ):
        stats = self._nqs_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            chunk_size=self._nqs_action_batch_size(batched_data.batch_size),
            local_device_chunks=True,
            action_ratio_apply=action_ratio_apply,
        )
        loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
        return stats._replace(loss=loss)

    def _reweighted_double_mc_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        ground_energy: float,
        omega,
        action_ratio_apply,
    ):
        score, ratio, source_weight = self._source_sampled_action_scores(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            source_ratio_apply=source_ratio_apply,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )
        components = _reweighted_double_mc_sr_components(
            score,
            ratio,
            source_weight,
            eps=self.lit_config.nqs_sr_score_eps,
        )

        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(
            self.lit_config.nqs_sr_damping,
            dtype=components.gradient.dtype,
        )
        damping = jnp.maximum(
            damping,
            jnp.asarray(1e-12, dtype=components.gradient.dtype),
        )
        preconditioned = self._solve_sr_direction(
            components.score_aug,
            components.gradient,
            damping,
        )
        scale = jnp.asarray(
            self.lit_config.nqs_learning_rate,
            dtype=components.gradient.dtype,
        )
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=components.gradient.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm
                / (update_norm + jnp.asarray(1e-12, dtype=components.gradient.dtype)),
            )
        return unravel_fn(scale * preconditioned)

    def _direct_double_mc_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_batched_data,
        psi_batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        ground_energy: float,
        omega,
        action_ratio_apply,
    ):
        source_ratio, source_weight = self._source_sampled_action_ratios(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_batched_data,
            source_ratio_apply=source_ratio_apply,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )
        source_components = _reweighted_double_mc_sr_components(
            jnp.zeros((source_ratio.shape[0], 1), dtype=source_ratio.dtype),
            source_ratio,
            source_weight,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        score, ratio, _ = self._source_sampled_action_scores(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            psi_batched_data,
            source_ratio_apply=source_ratio_apply,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )
        components = _direct_double_mc_sr_components(
            score,
            ratio,
            source_components.normalization,
            eps=self.lit_config.nqs_sr_score_eps,
        )

        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(
            self.lit_config.nqs_sr_damping,
            dtype=components.gradient.dtype,
        )
        damping = jnp.maximum(
            damping,
            jnp.asarray(1e-12, dtype=components.gradient.dtype),
        )
        preconditioned = self._solve_sr_direction(
            components.score_aug,
            components.gradient,
            damping,
        )
        scale = jnp.asarray(
            self.lit_config.nqs_learning_rate,
            dtype=components.gradient.dtype,
        )
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=components.gradient.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm
                / (update_norm + jnp.asarray(1e-12, dtype=components.gradient.dtype)),
            )
        return unravel_fn(scale * preconditioned)

    def _solve_sr_direction(self, score_aug, grad_flat, damping):
        parameter_count = grad_flat.shape[0]
        sample_count = score_aug.shape[0]
        if parameter_count <= sample_count:
            metric = score_aug.T @ score_aug
            metric = metric + damping * jnp.eye(parameter_count, dtype=metric.dtype)
            return jnp.linalg.solve(metric, grad_flat)
        kernel = score_aug @ score_aug.T
        kernel = kernel + damping * jnp.eye(sample_count, dtype=kernel.dtype)
        rhs = score_aug @ grad_flat
        projected = score_aug.T @ jnp.linalg.solve(kernel, rhs)
        return (grad_flat - projected) / damping

    def _source_sampled_action_scores(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        data = batched_data.data
        score_eps = float(self.lit_config.nqs_sr_score_eps)

        def action_and_score(params, one):
            action, _, _ = self._call_action_ratio_apply(
                action_ratio_apply,
                response_apply,
                params,
                ground_logpsi,
                ground_params,
                one,
                ground_energy=ground_energy,
                omega=omega,
            )

            def split_log_action(local_params):
                local_action, _, _ = self._call_action_ratio_apply(
                    action_ratio_apply,
                    response_apply,
                    local_params,
                    ground_logpsi,
                    ground_params,
                    one,
                    ground_energy=ground_energy,
                    omega=omega,
                )
                safe_action = jnp.where(
                    jnp.abs(local_action) > score_eps,
                    local_action,
                    jnp.asarray(score_eps, dtype=local_action.real.dtype) + 0j,
                )
                value = jnp.log(safe_action)
                return jnp.stack([jnp.real(value), jnp.imag(value)])

            jac = jax.jacrev(split_log_action)(params)
            score_tree = jax.tree.map(lambda leaf: leaf[0] + 1j * leaf[1], jac)
            return action, score_tree

        action, score_tree = jax.vmap(
            lambda one: action_and_score(response_params, one),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        score = _flatten_batched_tree(score_tree, action.shape[0])
        source = jax.vmap(
            source_ratio_apply,
            in_axes=(batched_data.vmap_axis,),
        )(data)
        safe_source = jnp.where(
            jnp.abs(source) > score_eps,
            source,
            jnp.asarray(score_eps, dtype=jnp.real(source).dtype) + 0j,
        )
        sampled_source = jnp.maximum(
            jnp.abs(source),
            jnp.asarray(self.lit_config.nqs_source_floor, dtype=jnp.real(source).dtype),
        )
        source_weight = (jnp.abs(source) / jnp.maximum(sampled_source, score_eps)) ** 2
        ratio = action / safe_source
        finite_score = jnp.all(
            jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
            axis=1,
        )
        finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
        finite_weight = jnp.isfinite(source_weight)
        finite = finite_score & finite_ratio & finite_weight
        score = jnp.where(finite[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
        ratio = jnp.where(finite, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
        source_weight = jnp.where(
            finite,
            source_weight,
            jnp.asarray(0.0, dtype=source_weight.dtype),
        )
        return score, ratio, source_weight

    def _source_sampled_action_ratios(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        data = batched_data.data
        score_eps = float(self.lit_config.nqs_sr_score_eps)
        action = jax.vmap(
            lambda one: self._call_action_ratio_apply(
                action_ratio_apply,
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                one,
                ground_energy=ground_energy,
                omega=omega,
            )[0],
            in_axes=(batched_data.vmap_axis,),
        )(data)
        source = jax.vmap(
            source_ratio_apply,
            in_axes=(batched_data.vmap_axis,),
        )(data)
        safe_source = jnp.where(
            jnp.abs(source) > score_eps,
            source,
            jnp.asarray(score_eps, dtype=jnp.real(source).dtype) + 0j,
        )
        sampled_source = jnp.maximum(
            jnp.abs(source),
            jnp.asarray(self.lit_config.nqs_source_floor, dtype=jnp.real(source).dtype),
        )
        source_weight = (jnp.abs(source) / jnp.maximum(sampled_source, score_eps)) ** 2
        ratio = action / safe_source
        finite = (
            jnp.isfinite(jnp.real(ratio))
            & jnp.isfinite(jnp.imag(ratio))
            & jnp.isfinite(source_weight)
        )
        ratio = jnp.where(finite, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
        source_weight = jnp.where(
            finite,
            source_weight,
            jnp.asarray(0.0, dtype=source_weight.dtype),
        )
        return ratio, source_weight

    def _reweighted_double_mc_sr_updates_from_scores(
        self,
        response_params,
        score,
        ratio,
        source_weight,
    ):
        components = _reweighted_double_mc_sr_components(
            score,
            ratio,
            source_weight,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        return self._double_mc_sr_updates_from_components(response_params, components)

    def _direct_double_mc_sr_updates_from_scores(
        self,
        response_params,
        source_ratio,
        source_weight,
        score,
        ratio,
    ):
        source_components = _reweighted_double_mc_sr_components(
            jnp.zeros((source_ratio.shape[0], 1), dtype=source_ratio.dtype),
            source_ratio,
            source_weight,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        components = _direct_double_mc_sr_components(
            score,
            ratio,
            source_components.normalization,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        return self._double_mc_sr_updates_from_components(response_params, components)

    def _double_mc_sr_updates_from_components(self, response_params, components):
        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(
            self.lit_config.nqs_sr_damping,
            dtype=components.gradient.dtype,
        )
        damping = jnp.maximum(
            damping,
            jnp.asarray(1e-12, dtype=components.gradient.dtype),
        )
        preconditioned = self._solve_sr_direction(
            components.score_aug,
            components.gradient,
            damping,
        )
        scale = jnp.asarray(
            self.lit_config.nqs_learning_rate,
            dtype=components.gradient.dtype,
        )
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=components.gradient.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm
                / (update_norm + jnp.asarray(1e-12, dtype=components.gradient.dtype)),
            )
        return unravel_fn(scale * preconditioned)

    def _nqs_direct_stats(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_batched_data,
        psi_batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        source_stats = self._nqs_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_batched_data,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )
        ratio, _ = self._source_sampled_action_ratios(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            psi_batched_data,
            source_ratio_apply=source_ratio_apply,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )
        components = _direct_double_mc_sr_components(
            jnp.zeros((ratio.shape[0], 1), dtype=ratio.dtype),
            ratio,
            source_stats.normalization,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        fidelity = components.fidelity
        action_norm = (
            jnp.asarray(source_norm, dtype=fidelity.dtype)
            * jnp.abs(source_stats.normalization) ** 2
            / jnp.maximum(fidelity, jnp.asarray(self.lit_config.nqs_sr_score_eps))
        )
        return source_stats._replace(
            loss=1.0 - fidelity,
            fidelity=fidelity,
            action_norm=jnp.real(action_norm),
            estimator_mode=jnp.asarray(1, dtype=jnp.int32),
        )

    def _nqs_direct_stats_from_ratio(self, source_stats, ratio, *, source_norm: float):
        components = _direct_double_mc_sr_components(
            jnp.zeros((ratio.shape[0], 1), dtype=ratio.dtype),
            ratio,
            source_stats.normalization,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        fidelity = components.fidelity
        action_norm = (
            jnp.asarray(source_norm, dtype=fidelity.dtype)
            * jnp.abs(source_stats.normalization) ** 2
            / jnp.maximum(fidelity, jnp.asarray(self.lit_config.nqs_sr_score_eps))
        )
        return source_stats._replace(
            loss=1.0 - fidelity,
            fidelity=fidelity,
            action_norm=jnp.real(action_norm),
            estimator_mode=jnp.asarray(1, dtype=jnp.int32),
        )

    def _nqs_loss(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        stats = nqs_lit_source_sampled_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
            local_action_fn=action_ratio_apply,
        )
        objective = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
        return objective, stats._replace(loss=objective)

    def _nqs_stats(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        return nqs_lit_source_sampled_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
            local_action_fn=action_ratio_apply,
        )

    def _nqs_stats_chunked(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        omega,
        chunk_size: int | None = None,
        local_device_chunks: bool = False,
        action_ratio_apply=None,
    ):
        if chunk_size is None:
            chunk_size = self._nqs_action_batch_size(batched_data.batch_size)
        chunk_size = max(1, min(int(chunk_size), int(batched_data.batch_size)))
        total_chunks = _chunk_count(batched_data.batch_size, chunk_size)
        progress_interval = max(1, total_chunks // 10)
        log_progress = self.lit_config.nqs_log_interval > 0 and total_chunks > 1
        if log_progress:
            logger.info(
                "nqs_stats omega=%.6f samples=%d chunk_size=%d chunks=%d",
                _host_scalar(omega),
                batched_data.batch_size,
                chunk_size,
                total_chunks,
            )

        @jax.jit
        def chunk_sums(local_params, chunk, local_omega):
            return nqs_lit_source_sampled_sums(
                response_apply,
                local_params,
                ground_logpsi,
                ground_params,
                chunk,
                source_ratio_apply=source_ratio_apply,
                ground_energy=ground_energy,
                omega=local_omega,
                eta=self.lit_config.eta,
                source_floor=self.lit_config.nqs_source_floor,
                local_action_fn=action_ratio_apply,
            )

        total_sums = None
        for chunk_index, chunk in enumerate(
            _batched_data_chunks(batched_data, chunk_size),
            start=1,
        ):
            if local_device_chunks:
                local_chunk = _local_device_batched_data(chunk)
            else:
                local_chunk = chunk
            sums = chunk_sums(response_params, local_chunk, omega)
            sums = _host_source_sums(sums)
            total_sums = (
                sums if total_sums is None else _add_source_sums(total_sums, sums)
            )
            if log_progress and (
                chunk_index == 1
                or chunk_index == total_chunks
                or chunk_index % progress_interval == 0
            ):
                logger.info(
                    "nqs_stats omega=%.6f chunk=%d/%d",
                    _host_scalar(omega),
                    chunk_index,
                    total_chunks,
                )
        if total_sums is None:
            msg = "Cannot evaluate NQS-LIT stats with an empty source pool."
            raise ValueError(msg)
        return nqs_lit_stats_from_source_sums(
            jax.tree.map(jnp.asarray, total_sums),
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )

    def _nqs_eval_stats(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_pool,
        *,
        source_ratio_apply: SourceRatioApply,
        source_norm: float,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        return self._nqs_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_pool,
            source_ratio_apply=source_ratio_apply,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )

    def _should_use_direct_psi(self, stats) -> bool:
        threshold = float(self.lit_config.nqs_reweight_ess_fraction_min)
        if threshold <= 0.0:
            return False
        return float(jax.device_get(stats.reweight_ess_fraction)) < threshold

    def _collect_direct_psi_pool(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        rng,
        *,
        ground_energy: float,
        omega,
        batches: int,
        action_ratio_apply=None,
    ):
        sample_plan = SamplePlan(
            self._make_action_log_amplitude(
                response_apply,
                ground_logpsi,
                ground_params,
                ground_energy=ground_energy,
                omega=omega,
                action_ratio_apply=action_ratio_apply,
            ),
            {"electrons": self.sampler},
        )
        rng, sample_rng = jax.random.split(rng)
        sampler_state = sample_plan.init(batched_data, sample_rng)
        sample_plan = _JittedSamplePlan(sample_plan, batched_data)
        batched_data, sampler_state, rng = _run_sample_steps(
            sample_plan,
            response_params,
            batched_data,
            sampler_state,
            rng,
            self.lit_config.nqs_direct_psi_burn_in,
        )
        pool, batched_data, _, rng = self._collect_sample_pool(
            sample_plan,
            response_params,
            batched_data,
            sampler_state,
            rng,
            batches=max(1, int(batches)),
            stride=max(1, int(self.lit_config.nqs_direct_psi_stride)),
        )
        return pool, batched_data, rng

    def _make_action_log_amplitude(
        self,
        response_apply,
        ground_logpsi,
        ground_params,
        *,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ):
        eps = float(self.lit_config.nqs_sr_score_eps)

        def log_amplitude(response_params, data):
            action, _, _ = self._call_action_ratio_apply(
                action_ratio_apply,
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                data,
                ground_energy=ground_energy,
                omega=omega,
            )
            return ground_logpsi(ground_params, data) + jnp.log(
                jnp.maximum(jnp.abs(action), eps)
            )

        return log_amplitude

    def _nqs_train_update_batch_size(self) -> int:
        configured = int(self.lit_config.nqs_train_update_batch_size)
        if configured > 0:
            return configured
        return max(1, int(self.config.batch_size))

    def _uses_projected_response_path(self) -> bool:
        return bool(self.lit_config.nqs_symmetry_projectors)

    def _nqs_sr_score_batch_size(self, update_batch_size: int) -> int:
        configured = int(self.lit_config.nqs_sr_score_batch_size)
        if configured <= 0:
            requested = max(1, int(update_batch_size))
        else:
            requested = min(max(1, configured), max(1, int(update_batch_size)))
        if self._uses_projected_response_path():
            return min(
                requested,
                int(self.lit_config.nqs_projected_sr_score_batch_cap),
            )
        return requested

    def _nqs_action_batch_size(self, batch_size: int) -> int:
        configured = int(self.lit_config.nqs_eval_batch_size)
        requested = max(1, int(batch_size))
        if configured > 0:
            requested = min(requested, configured)
        if self._uses_projected_response_path():
            requested = min(
                requested,
                int(self.lit_config.nqs_projected_action_batch_cap),
            )
        return requested

    def _use_nqs_sr_score_chunks(self, update_batch_size: int) -> bool:
        return self._nqs_sr_score_batch_size(update_batch_size) < max(
            1,
            int(update_batch_size),
        )

    def _nqs_sr_score_chunks(self, score_fn, response_params, batched_data, omega):
        chunk_size = self._nqs_sr_score_batch_size(batched_data.batch_size)
        total_chunks = _chunk_count(batched_data.batch_size, chunk_size)
        progress_interval = max(1, total_chunks // 10)
        log_progress = self.lit_config.nqs_log_interval <= 1 and total_chunks > 1
        if log_progress:
            logger.info(
                "nqs_sr_score omega=%.6f samples=%d chunk_size=%d chunks=%d",
                _host_scalar(omega),
                batched_data.batch_size,
                chunk_size,
                total_chunks,
            )
        score_parts = []
        ratio_parts = []
        weight_parts = []
        for chunk_index, chunk in enumerate(
            _batched_data_chunks(batched_data, chunk_size),
            start=1,
        ):
            local_chunk = _local_device_batched_data(chunk)
            score, ratio, source_weight = score_fn(
                response_params,
                local_chunk,
                omega,
            )
            score_parts.append(np.asarray(jax.device_get(score)))
            ratio_parts.append(np.asarray(jax.device_get(ratio)))
            weight_parts.append(np.asarray(jax.device_get(source_weight)))
            if log_progress and (
                chunk_index == 1
                or chunk_index == total_chunks
                or chunk_index % progress_interval == 0
            ):
                logger.info(
                    "nqs_sr_score omega=%.6f chunk=%d/%d",
                    _host_scalar(omega),
                    chunk_index,
                    total_chunks,
                )
        return (
            jnp.asarray(np.concatenate(score_parts, axis=0)),
            jnp.asarray(np.concatenate(ratio_parts, axis=0)),
            jnp.asarray(np.concatenate(weight_parts, axis=0)),
        )

    def _nqs_sr_source_ratio_chunks(
        self,
        source_ratio_fn,
        response_params,
        batched_data,
        omega,
    ):
        chunk_size = self._nqs_action_batch_size(batched_data.batch_size)
        total_chunks = _chunk_count(batched_data.batch_size, chunk_size)
        progress_interval = max(1, total_chunks // 10)
        log_progress = self.lit_config.nqs_log_interval <= 1 and total_chunks > 1
        if log_progress:
            logger.info(
                "nqs_source_ratio omega=%.6f samples=%d chunk_size=%d chunks=%d",
                _host_scalar(omega),
                batched_data.batch_size,
                chunk_size,
                total_chunks,
            )
        ratio_parts = []
        weight_parts = []
        for chunk_index, chunk in enumerate(
            _batched_data_chunks(batched_data, chunk_size),
            start=1,
        ):
            local_chunk = _local_device_batched_data(chunk)
            ratio, source_weight = source_ratio_fn(response_params, local_chunk, omega)
            ratio_parts.append(np.asarray(jax.device_get(ratio)))
            weight_parts.append(np.asarray(jax.device_get(source_weight)))
            if log_progress and (
                chunk_index == 1
                or chunk_index == total_chunks
                or chunk_index % progress_interval == 0
            ):
                logger.info(
                    "nqs_source_ratio omega=%.6f chunk=%d/%d",
                    _host_scalar(omega),
                    chunk_index,
                    total_chunks,
                )
        return (
            jnp.asarray(np.concatenate(ratio_parts, axis=0)),
            jnp.asarray(np.concatenate(weight_parts, axis=0)),
        )

    def _nqs_eval_batch_size(self) -> int:
        configured = int(self.lit_config.nqs_eval_batch_size)
        if configured > 0:
            return configured
        return max(1, int(self.config.batch_size))

    def _maybe_projection_leakage(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        eval_pool,
        projector: SymmetryProjector,
        *,
        ground_energy: float,
        omega,
        batch_index: int,
        action_ratio_apply=None,
    ) -> tuple[float, float]:
        sample_count = int(self.lit_config.nqs_leakage_diagnostic_samples)
        if sample_count <= 0:
            return 0.0, 0.0
        leakage_pool = _cyclic_batched_data_chunk(eval_pool, sample_count, batch_index)
        return self._projection_leakage(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            leakage_pool,
            projector,
            ground_energy=ground_energy,
            omega=omega,
            action_ratio_apply=action_ratio_apply,
        )

    def _projection_leakage(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        projector: SymmetryProjector,
        *,
        ground_energy: float,
        omega,
        action_ratio_apply=None,
    ) -> tuple[float, float]:
        if projector.is_identity:
            return 0.0, 0.0

        def one_response_leakage(data):
            def response_value(local_data):
                return jnp.exp(response_apply(response_params, local_data))

            value = response_value(data)
            projected = project_value(
                response_value,
                data,
                projector,
                chunk_size=_projection_chunk_size_from_config(self.lit_config),
            )
            denominator = jnp.maximum(jnp.abs(value) ** 2, 1e-24)
            return jnp.abs(value - projected) ** 2 / denominator

        def one_carrier_leakage(data):
            def carrier_value(local_data):
                action_ratio, _, _ = self._call_action_ratio_apply(
                    action_ratio_apply,
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    local_data,
                    ground_energy=ground_energy,
                    omega=omega,
                )
                return action_ratio * jnp.exp(ground_logpsi(ground_params, local_data))

            value = carrier_value(data)
            projected = project_value(
                carrier_value,
                data,
                projector,
                chunk_size=_projection_chunk_size_from_config(self.lit_config),
            )
            denominator = jnp.maximum(jnp.abs(value) ** 2, 1e-24)
            return jnp.abs(value - projected) ** 2 / denominator

        data = batched_data.data
        response = jax.vmap(
            one_response_leakage,
            in_axes=(batched_data.vmap_axis,),
        )(data)
        carrier = jax.vmap(
            one_carrier_leakage,
            in_axes=(batched_data.vmap_axis,),
        )(data)
        return (
            float(jnp.mean(jnp.real(response))),
            float(jnp.mean(jnp.real(carrier))),
        )

    def _log_nqs_summary(self, output_path: str, peaks, fidelity: np.ndarray) -> None:
        logger.info("Wrote NQS-LIT spectrum to %s", output_path)
        logger.info(
            "NQS-LIT fidelity range: min=%.6f max=%.6f",
            float(np.min(fidelity)),
            float(np.max(fidelity)),
        )
        for peak in peaks[: self.lit_config.preview_peaks]:
            logger.info(
                "peak omega=%.8f Ha broadened_intensity=%.6e",
                float(peak.energy),
                float(peak.intensity),
            )

    def _should_run_parallel_scan(self) -> bool:
        if self.lit_config.scan_parallel_worker:
            return False
        mode = self.lit_config.scan_parallel.lower()
        if mode in ("off", "false", "none", "0"):
            return False
        if len(_omega_grid_from_config(self.lit_config)) < 2:
            return False
        device_ids = _visible_cuda_devices()
        if len(device_ids) < 2:
            return False
        return self._parallel_worker_count(device_ids) > 1

    def _parallel_worker_count(self, device_ids: tuple[str, ...]) -> int:
        available_slots = len(device_ids) * int(
            self.lit_config.scan_parallel_procs_per_device
        )
        worker_limit = (
            int(self.lit_config.scan_parallel_workers)
            if self.lit_config.scan_parallel_workers > 0
            else available_slots
        )
        points_limit = max(
            1,
            int(
                np.ceil(
                    len(_omega_grid_from_config(self.lit_config))
                    / self.lit_config.scan_parallel_min_points_per_worker
                )
            ),
        )
        return max(1, min(available_slots, worker_limit, points_limit))

    def _run_parallel_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = _omega_grid_from_config(self.lit_config)
        device_ids = _visible_cuda_devices()
        worker_count = self._parallel_worker_count(device_ids)
        blocks = _split_omega_blocks(len(omega), worker_count)
        if len(blocks) <= 1:
            logger.info("Parallel LIT scan requested but only one block is needed.")
            self._run_serial_scan()
            return
        worker_devices = _parallel_worker_device_ids(
            device_ids,
            len(blocks),
            int(self.lit_config.scan_parallel_procs_per_device),
        )

        parallel_root = self.save_path / "parallel_scan"
        parallel_root.mkdir(parents=True, exist_ok=True)
        shared_source_pool_dir = _parallel_shared_source_pool_dir(
            self.lit_config,
            parallel_root,
        )
        if shared_source_pool_dir is not None:
            shared_source_pool_dir.mkdir(parents=True, exist_ok=True)
        base_config_path = parallel_root / "base_config.yaml"
        base_config_path.write_text(self.cfg.to_yaml())
        run_seed = (
            int(self.config.seed) if self.config.seed is not None else int(time.time())
        )

        logger.info(
            "Starting local-device LIT scan: workers=%d devices=%s "
            "procs_per_device=%d blocks=%s",
            len(blocks),
            ",".join(worker_devices),
            int(self.lit_config.scan_parallel_procs_per_device),
            ",".join(f"{block[0]}:{block[-1] + 1}" for block in blocks),
        )
        failures: list[_ParallelWorker] = []
        with ExitStack() as stack:
            workers: list[_ParallelWorker] = []
            for worker_index, block in enumerate(blocks):
                part_dir = parallel_root / f"block_{worker_index:03d}"
                part_dir.mkdir(parents=True, exist_ok=True)
                source_pool_dir = (
                    shared_source_pool_dir
                    if shared_source_pool_dir is not None
                    else part_dir / "source_pools"
                )
                source_pool_dir.mkdir(parents=True, exist_ok=True)
                log_path = part_dir / "worker.log"
                command = self._parallel_worker_command(
                    base_config_path,
                    part_dir,
                    omega[block],
                    run_seed=run_seed,
                    source_pool_dir=source_pool_dir,
                )
                device = worker_devices[worker_index]
                env = _parallel_worker_env(device)
                log_file = stack.enter_context(log_path.open("w", encoding="utf8"))
                process = subprocess.Popen(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                workers.append(
                    _ParallelWorker(
                        index=worker_index,
                        block=block,
                        path=part_dir / self.lit_config.output_filename,
                        log_path=log_path,
                        process=process,
                        device=device,
                    )
                )
            for worker in workers:
                process = worker.process
                returncode = process.wait()
                if returncode != 0:
                    failures.append(worker)
        if failures:
            details = "\n".join(
                f"worker {worker.index} device={worker.device} "
                f"returncode={worker.process.returncode}\n"
                f"{_tail_text(worker.log_path)}"
                for worker in failures
            )
            msg = f"Parallel LIT scan failed:\n{details}"
            raise RuntimeError(msg)

        logger.info("All parallel LIT workers finished; merging partial spectra.")
        self._merge_parallel_outputs(
            [worker.path for worker in workers],
            devices=[worker.device for worker in workers],
            blocks=[worker.block for worker in workers],
            axes=axes,
        )

    def _parallel_worker_command(
        self,
        base_config_path: UPath,
        part_dir: UPath,
        block_omega: np.ndarray,
        *,
        run_seed: int,
        source_pool_dir: UPath,
    ) -> list[str]:
        return [
            sys.executable,
            "-c",
            "from jaqmc.app.cli import cli; cli()",
            "molecule",
            "lit",
            "--yml",
            str(base_config_path),
            f"workflow.seed={run_seed}",
            f"workflow.save_path={part_dir}",
            f"workflow.restore_path={self.restore_path}",
            "lit.scan_parallel=off",
            "lit.scan_parallel_worker=true",
            f"lit.omega_min={float(block_omega[0])}",
            f"lit.omega_max={float(block_omega[-1])}",
            f"lit.omega_points={int(block_omega.size)}",
            "lit.omega_values="
            + ",".join(f"{float(value):.12g}" for value in block_omega),
            f"lit.nqs_source_pool_dir={source_pool_dir}",
        ]

    def _merge_parallel_outputs(
        self,
        paths: list[UPath],
        *,
        devices: list[str],
        blocks: list[np.ndarray],
        axes: tuple[int, ...],
    ) -> None:
        loaded = []
        for path in paths:
            if not path.exists():
                msg = f"Parallel LIT worker did not produce {path}"
                raise FileNotFoundError(msg)
            with path.open("rb") as f_in, np.load(f_in) as data:
                loaded.append({key: data[key] for key in data.files})

        omega = np.concatenate([part["omega"] for part in loaded])
        order = np.argsort(omega)
        omega = omega[order]
        concat_axis_fields = (
            "lit",
            "broadened",
            "fidelity",
            "residual_norm",
            "action_norm",
            "source_norm",
            "error_bound_monitor",
            "error_d",
            "reweight_ess",
            "reweight_ess_fraction",
            "estimator_mode",
            "normalization",
            "correction_overlap",
        )
        combined = {}
        for field_name in concat_axis_fields:
            arr = np.concatenate([part[field_name] for part in loaded], axis=1)
            combined[field_name] = arr[:, order]

        concat_sector_fields = (
            "sector_lit",
            "sector_broadened",
            "sector_fidelity",
            "sector_residual_norm",
            "sector_action_norm",
            "sector_error_bound_monitor",
            "sector_error_d",
            "sector_reweight_ess",
            "sector_reweight_ess_fraction",
            "sector_estimator_mode",
            "sector_normalization",
            "sector_correction_overlap",
            "sector_response_leakage",
            "sector_carrier_leakage",
        )
        for field_name in concat_sector_fields:
            if field_name not in loaded[0]:
                continue
            arr = np.concatenate([part[field_name] for part in loaded], axis=1)
            combined[field_name] = arr[:, order]

        axis_average_factor = float(
            loaded[0].get(
                "axis_average_factor",
                np.asarray(self._axis_average_factor(axes)),
            )
        )
        total_broadened = np.sum(combined["broadened"], axis=0) / axis_average_factor
        peaks = find_spectrum_peaks(
            omega,
            total_broadened,
            min_height_fraction=self.lit_config.peak_min_height_fraction,
        )
        source_centers_blocks = np.asarray(
            [part["source_centers"] for part in loaded],
            dtype=np.float64,
        )
        axis_source_norm_blocks = np.asarray(
            [part["axis_source_norm"] for part in loaded],
            dtype=np.float64,
        )
        ground_energies = np.asarray(
            [float(part["ground_energy"]) for part in loaded],
            dtype=np.float64,
        )
        output_path = self.save_path / self.lit_config.output_filename
        _save_npz(
            output_path,
            backend="nqs_lit",
            omega=omega,
            eta=self.lit_config.eta,
            axes=self.lit_config.axes,
            axis_indices=np.asarray(axes, dtype=np.int64),
            total_broadened=total_broadened,
            ground_energy=float(np.mean(ground_energies)),
            ground_energy_blocks=ground_energies,
            ground_checkpoint_step=loaded[0]["ground_checkpoint_step"],
            nqs_train_pool_batches=self.lit_config.nqs_train_pool_batches,
            nqs_eval_pool_batches=self.lit_config.nqs_eval_pool_batches,
            nqs_pool_stride=self.lit_config.nqs_pool_stride,
            nqs_parallel_shared_source_pool=bool(
                self.lit_config.nqs_parallel_shared_source_pool
            ),
            nqs_sr_estimator=np.asarray(
                "double_mc_centered_jacobian_reweighted_or_direct",
                dtype=str,
            ),
            nqs_reweight_ess_fraction_min=(
                self.lit_config.nqs_reweight_ess_fraction_min
            ),
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            nqs_ground_spatial_irrep=np.asarray(
                _optional_projector_label(self.lit_config.nqs_ground_spatial_irrep)
                or "",
                dtype=str,
            ),
            nqs_response_spatial_irreps=np.asarray(
                _projector_label_list(self.lit_config.nqs_response_spatial_irreps),
                dtype=str,
            ),
            nqs_so3_quadrature_order=self.lit_config.nqs_so3_quadrature_order,
            nqs_so2_quadrature_order=self.lit_config.nqs_so2_quadrature_order,
            nqs_leakage_diagnostic_samples=(
                self.lit_config.nqs_leakage_diagnostic_samples
            ),
            source_centers=np.mean(source_centers_blocks, axis=0),
            axis_source_norm=np.mean(axis_source_norm_blocks, axis=0),
            source_centers_blocks=source_centers_blocks,
            axis_source_norm_blocks=axis_source_norm_blocks,
            ground_symmetry_projector=loaded[0].get(
                "ground_symmetry_projector",
                np.asarray("unknown", dtype=str),
            ),
            response_symmetry_projectors=loaded[0].get(
                "response_symmetry_projectors",
                np.asarray([], dtype=str),
            ),
            sector_labels=loaded[0].get("sector_labels", np.asarray([], dtype=str)),
            sector_axes=loaded[0].get("sector_axes", np.asarray([], dtype=np.int64)),
            sector_source_norm=loaded[0].get(
                "sector_source_norm",
                np.asarray([], dtype=np.float64),
            ),
            sector_source_elastic_overlap=loaded[0].get(
                "sector_source_elastic_overlap",
                np.asarray([], dtype=np.complex128),
            ),
            axis_average_factor=axis_average_factor,
            peak_energies=np.asarray([peak.energy for peak in peaks]),
            peak_intensities=np.asarray([peak.intensity for peak in peaks]),
            peak_indices=np.asarray([peak.index for peak in peaks]),
            parallel_scan_enabled=True,
            parallel_scan_devices=np.asarray(devices, dtype=str),
            parallel_scan_blocks=np.asarray(
                [[int(block[0]), int(block[-1] + 1)] for block in blocks],
                dtype=np.int64,
            ),
            parallel_scan_part_paths=np.asarray([str(path) for path in paths]),
            **combined,
        )
        self._log_nqs_summary(str(output_path), peaks, combined["fidelity"])


_AXIS_NAMES = ("x", "y", "z")
_CACHE_LOCK_POLL_SECONDS = 1.0
_CACHE_LOCK_STALE_SECONDS = 6 * 60 * 60


def _chunk_count(batch_size: int, chunk_size: int) -> int:
    batch_size = max(0, int(batch_size))
    chunk_size = max(1, int(chunk_size))
    return max(1, (batch_size + chunk_size - 1) // chunk_size)


def _host_scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def _omega_grid_from_config(lit_config: MolecularLITConfig) -> np.ndarray:
    values = str(getattr(lit_config, "omega_values", "") or "").strip()
    if not values:
        return np.linspace(
            float(lit_config.omega_min),
            float(lit_config.omega_max),
            int(lit_config.omega_points),
        )
    omega = np.asarray(
        [float(value) for value in re.split(r"[\s,]+", values) if value],
        dtype=np.float64,
    )
    if omega.size == 0:
        msg = "lit.omega_values must contain at least one value when set."
        raise ValueError(msg)
    if not np.all(np.isfinite(omega)):
        msg = "lit.omega_values must be finite."
        raise ValueError(msg)
    if omega.size > 1 and not np.all(np.diff(omega) > 0.0):
        msg = "lit.omega_values must be strictly increasing."
        raise ValueError(msg)
    return omega


def _parallel_shared_source_pool_dir(
    lit_config: MolecularLITConfig,
    parallel_root: UPath,
) -> UPath | None:
    if lit_config.nqs_source_pool_dir:
        return UPath(lit_config.nqs_source_pool_dir)
    if lit_config.nqs_parallel_shared_source_pool:
        return parallel_root / "source_pools"
    return None


def _visible_cuda_devices() -> tuple[str, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = tuple(
            item.strip()
            for item in visible.split(",")
            if item.strip() and item.strip() != "-1"
        )
        return devices
    try:
        return tuple(str(idx) for idx, _ in enumerate(jax.devices("gpu")))
    except RuntimeError:
        return ()


def _split_omega_blocks(omega_points: int, worker_count: int) -> list[np.ndarray]:
    indices = np.arange(int(omega_points), dtype=np.int64)
    return [
        block
        for block in np.array_split(indices, max(1, int(worker_count)))
        if block.size > 0
    ]


def _parallel_worker_device_ids(
    device_ids: tuple[str, ...],
    worker_count: int,
    procs_per_device: int,
) -> tuple[str, ...]:
    if not device_ids:
        msg = "At least one CUDA device id is required."
        raise ValueError(msg)
    if procs_per_device < 1:
        msg = "procs_per_device must be positive."
        raise ValueError(msg)
    max_workers = len(device_ids) * int(procs_per_device)
    if worker_count > max_workers:
        msg = (
            f"worker_count={worker_count} exceeds available parallel slots "
            f"{max_workers}."
        )
        raise ValueError(msg)
    return tuple(device_ids[index % len(device_ids)] for index in range(worker_count))


def _parallel_worker_env(device_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _optional_projector_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = str(value).strip()
    if not label or label.lower() in {"auto", "none", "null"}:
        return None
    return label


def _projection_chunk_size_from_config(lit_config) -> int:
    return int(getattr(lit_config, "nqs_projection_chunk_size", 4))


def _projector_label_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text or text.lower() in {"auto", "none", "null"}:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _stack_sector_arrays(
    values: list[np.ndarray],
    omega_count: int,
    dtype,
) -> np.ndarray:
    if not values:
        return np.zeros((0, int(omega_count)), dtype=dtype)
    return np.stack([np.asarray(value, dtype=dtype) for value in values], axis=0)


def _safe_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", label)
    return slug.strip("_") or "sector"


def _stable_label_hash(label: str) -> int:
    value = 0
    for char in label:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def _spatial_projector_vector_component_norm(projector, axis: int) -> float:
    component = np.zeros(3, dtype=np.complex128)
    for matrix, coefficient in zip(
        projector.matrices,
        projector.coefficients,
        strict=True,
    ):
        component += (
            complex(coefficient)
            * np.asarray(matrix, dtype=np.float64)[
                :,
                int(axis),
            ]
        )
    return float(np.linalg.norm(component))


def _two_spin_tuple(values) -> tuple[int, int]:
    nspins = tuple(int(value) for value in values)
    if len(nspins) != 2:
        msg = f"Expected two spin populations, got {nspins}."
        raise ValueError(msg)
    return nspins


def _minimum_compatible_spin(nspins: tuple[int, int]) -> float:
    return abs(int(nspins[0]) - int(nspins[1])) * 0.5


def _same_symmetry_sector(left: SymmetryProjector, right: SymmetryProjector) -> bool:
    return left.spatial == right.spatial and left.spin == right.spin


def _reweighted_double_mc_sr_components(
    score: jnp.ndarray,
    ratio: jnp.ndarray,
    source_weight: jnp.ndarray,
    *,
    eps: float,
) -> _ReweightedDoubleMCSRComponents:
    """Return the article double-MC gradient and QFI from pi_Phi samples.

    ``ratio`` is ``Psi/Phi`` and ``score`` is the centered-target candidate
    ``nabla_theta log Psi`` before centering.  The formal double-MC estimator
    uses expectations over ``pi_Psi``; here those expectations are evaluated by
    importance reweighting from the reusable projected-source pool ``pi_Phi``.
    """
    real_dtype = jnp.real(ratio).dtype
    eps_value = jnp.asarray(eps, dtype=real_dtype)
    safe_source_weight_sum = jnp.maximum(jnp.sum(source_weight), eps_value)
    phi_weight = source_weight / safe_source_weight_sum
    amplitude = jnp.sum(phi_weight * ratio)
    ratio_abs2 = jnp.abs(ratio) ** 2
    ratio_norm = jnp.sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps_value)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm

    score_mean_psi = jnp.sum(psi_weight[:, None] * score, axis=0, keepdims=True)
    centered_score = score - score_mean_psi

    # This is E_{pi_Psi}[Delta J H_loc^*] written without dividing by ratio:
    # H_loc = (Phi/Psi) E_{pi_Phi}[Psi/Phi].
    score_covariance = jnp.sum(
        phi_weight[:, None] * ratio[:, None] * centered_score,
        axis=0,
    )
    gradient = 2.0 * jnp.real(jnp.conj(amplitude) * score_covariance / safe_ratio_norm)

    weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
    score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)

    fidelity = (jnp.abs(amplitude) ** 2) / safe_ratio_norm
    fidelity = jnp.clip(jnp.real(fidelity), 0.0, 1.0)
    reweight_ess = 1.0 / jnp.maximum(jnp.sum(psi_weight**2), eps_value)
    valid_sample_count = jnp.maximum(
        jnp.sum(source_weight > jnp.asarray(0.0, dtype=source_weight.dtype)),
        jnp.asarray(1, dtype=real_dtype),
    )
    return _ReweightedDoubleMCSRComponents(
        gradient=gradient,
        score_aug=score_aug,
        normalization=amplitude,
        fidelity=fidelity,
        reweight_ess=jnp.real(reweight_ess),
        reweight_ess_fraction=jnp.real(reweight_ess / valid_sample_count),
    )


def _direct_double_mc_sr_components(
    score: jnp.ndarray,
    ratio: jnp.ndarray,
    normalization: jnp.ndarray,
    *,
    eps: float,
) -> _ReweightedDoubleMCSRComponents:
    """Return direct-pi_Psi double-MC gradient and QFI components."""
    real_dtype = jnp.real(ratio).dtype
    eps_value = jnp.asarray(eps, dtype=real_dtype)
    safe_ratio = jnp.where(
        jnp.abs(ratio) > eps_value,
        ratio,
        eps_value + 0j,
    )
    hloc_conj = jnp.conj(normalization / safe_ratio)
    finite = (
        jnp.isfinite(jnp.real(hloc_conj))
        & jnp.isfinite(jnp.imag(hloc_conj))
        & jnp.all(jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)), axis=1)
    )
    score = jnp.where(finite[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
    hloc_conj = jnp.where(finite, hloc_conj, jnp.asarray(0.0, dtype=hloc_conj.dtype))
    valid_count = jnp.maximum(
        jnp.sum(finite),
        jnp.asarray(1, dtype=real_dtype),
    )
    sample_weight = finite.astype(real_dtype) / valid_count
    score_mean = jnp.sum(sample_weight[:, None] * score, axis=0, keepdims=True)
    centered_score = score - score_mean
    gradient = 2.0 * jnp.real(
        jnp.sum(sample_weight[:, None] * centered_score * hloc_conj[:, None], axis=0)
    )
    weighted_score = jnp.sqrt(sample_weight)[:, None] * centered_score
    score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)
    fidelity = jnp.clip(
        jnp.real(jnp.sum(sample_weight * jnp.conj(hloc_conj))),
        0.0,
        1.0,
    )
    return _ReweightedDoubleMCSRComponents(
        gradient=gradient,
        score_aug=score_aug,
        normalization=normalization,
        fidelity=fidelity,
        reweight_ess=valid_count,
        reweight_ess_fraction=jnp.asarray(1.0, dtype=real_dtype),
    )


def _save_npz(path: UPath, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f_out:
        np.savez(f_out, **payload)  # type: ignore[arg-type]


def _cache_lock_path(path: UPath) -> UPath:
    return path.parent / f"{path.name}.lock"


@contextmanager
def _file_lock(lock_path: UPath):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    while not acquired:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - float(lock_path.stat().st_mtime)
            except FileNotFoundError:
                continue
            if age > _CACHE_LOCK_STALE_SECONDS:
                logger.warning("Removing stale source cache lock %s", lock_path)
                with suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            time.sleep(_CACHE_LOCK_POLL_SECONDS)
            continue
        with os.fdopen(fd, "w", encoding="utf8") as f_out:
            f_out.write(f"pid={os.getpid()} time={time.time():.6f}\n")
        acquired = True
    try:
        yield
    finally:
        if acquired:
            with suppress(FileNotFoundError):
                lock_path.unlink()


def _save_scalar_cache(
    path: UPath,
    *,
    metadata: dict[str, float],
    **values: float | complex,
) -> None:
    payload: dict[str, object] = {}
    for key, value in values.items():
        dtype = np.complex128 if np.iscomplexobj(value) else np.float64
        payload[key] = np.asarray(value, dtype=dtype)
    for key, value in metadata.items():
        payload[f"metadata_{key}"] = np.asarray(value, dtype=np.float64)
    _save_npz(path, **payload)


def _load_scalar_cache(
    path: UPath,
    key: str,
    *,
    metadata: dict[str, float],
) -> float | complex:
    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        _validate_pool_metadata(npf, metadata)
        if key not in npf:
            msg = f"source scalar cache is missing field {key!r}"
            raise KeyError(msg)
        return npf[key].item()


def _tail_text(path: UPath, lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf8")
    except OSError as exc:
        return f"<failed to read {path}: {exc}>"
    return "\n".join(text.splitlines()[-lines:])


def _axis_indices(axes: str) -> tuple[int, ...]:
    lookup = {name: idx for idx, name in enumerate(_AXIS_NAMES)}
    result = []
    for raw in axes.lower():
        if raw not in lookup:
            msg = f"Unknown dipole axis {raw!r}; expected characters from 'xyz'."
            raise ValueError(msg)
        result.append(lookup[raw])
    if not result:
        msg = "At least one dipole axis is required."
        raise ValueError(msg)
    return tuple(result)


def _flatten_batched_tree(tree, batch_size: int) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        msg = "Cannot build SR score matrix from an empty parameter tree."
        raise ValueError(msg)
    return jnp.concatenate(
        [jnp.reshape(leaf, (batch_size, -1)) for leaf in leaves],
        axis=1,
    )


def _concat_batched_data(pool):
    if not pool:
        msg = "At least one sampled batch is required to build a source pool."
        raise ValueError(msg)
    first = pool[0]
    updates = {}
    for field_name in first.fields_with_batch:
        updates[field_name] = jnp.concatenate(
            [getattr(batch.data, field_name) for batch in pool],
            axis=0,
        )
    return first.__class__(
        data=first.data.merge(updates),
        fields_with_batch=first.fields_with_batch,
    )


def _slice_batched_data(pool: BatchedData, start: int, size: int) -> BatchedData:
    if size < 1:
        msg = "BatchedData chunk size must be positive."
        raise ValueError(msg)
    if start < 0 or start + size > pool.batch_size:
        msg = (
            f"Invalid BatchedData slice start={start} size={size} "
            f"for batch_size={pool.batch_size}."
        )
        raise ValueError(msg)
    updates = {
        field_name: jax.tree.map(
            operator.itemgetter(slice(start, start + size)),
            getattr(pool.data, field_name),
        )
        for field_name in pool.fields_with_batch
    }
    return pool.__class__(
        data=pool.data.merge(updates),
        fields_with_batch=pool.fields_with_batch,
    )


def _cyclic_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    batch_index: int,
) -> BatchedData:
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    if chunk_size >= pool.batch_size:
        return pool
    chunk_count = max(1, pool.batch_size // chunk_size)
    start = (int(batch_index) % chunk_count) * chunk_size
    return _slice_batched_data(pool, start, chunk_size)


def _batched_data_chunks(pool: BatchedData, requested_size: int):
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    start = 0
    while start < pool.batch_size:
        size = min(chunk_size, pool.batch_size - start)
        yield _slice_batched_data(pool, start, size)
        start += size


def _add_source_sums(left, right):
    return jax.tree.map(operator.add, left, right)


def _host_source_sums(sums):
    return jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), sums)


def _save_batched_pool(
    path: UPath,
    pool: BatchedData,
    *,
    metadata: dict[str, float] | None = None,
) -> None:
    payload: dict[str, object] = {
        "fields_with_batch": np.asarray(list(pool.fields_with_batch), dtype=str),
    }
    if metadata is not None:
        for key, value in metadata.items():
            payload[f"metadata_{key}"] = np.asarray(value, dtype=np.float64)
    for field_name in pool.fields_with_batch:
        payload[field_name] = np.asarray(jax.device_get(getattr(pool.data, field_name)))
    _save_npz(path, **payload)


def _load_batched_pool(
    path: UPath,
    reference: BatchedData,
    *,
    metadata: dict[str, float] | None = None,
) -> BatchedData:
    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        if metadata is not None:
            _validate_pool_metadata(npf, metadata)
        fields = tuple(str(field) for field in npf["fields_with_batch"].tolist())
        if fields != tuple(reference.fields_with_batch):
            msg = (
                "source pool batched fields do not match current data: "
                f"{fields} != {tuple(reference.fields_with_batch)}"
            )
            raise ValueError(msg)
        updates = {}
        for field_name in fields:
            if field_name not in npf:
                msg = f"source pool is missing field {field_name!r}"
                raise KeyError(msg)
            value = np.asarray(npf[field_name])
            reference_value = getattr(reference.data, field_name)
            if value.shape[1:] != reference_value.shape[1:]:
                msg = (
                    f"source pool field {field_name!r} has incompatible shape "
                    f"{value.shape}; expected trailing {reference_value.shape[1:]}"
                )
                raise ValueError(msg)
            updates[field_name] = value
    return reference.__class__(
        data=reference.data.merge(updates),
        fields_with_batch=fields,
    )


def _validate_pool_metadata(npf, metadata: dict[str, float]) -> None:
    for key, expected in metadata.items():
        npz_key = f"metadata_{key}"
        if npz_key not in npf:
            msg = f"source pool is missing metadata {key!r}"
            raise ValueError(msg)
        actual = float(npf[npz_key])
        if not np.isclose(actual, float(expected), rtol=1e-8, atol=1e-10):
            msg = (
                f"source pool metadata {key!r} mismatch: {actual} != {float(expected)}"
            )
            raise ValueError(msg)


def _copy_matching_parameters(target, source):
    if not jax.tree_util.tree_leaves(source):
        return target
    target_mut = unfreeze(target)
    source_mut = unfreeze(source)
    return freeze(_copy_matching_mapping(target_mut, source_mut))


def _copy_matching_mapping(target, source):
    if isinstance(target, dict) and isinstance(source, dict):
        return {
            key: _copy_matching_mapping(value, source[key]) if key in source else value
            for key, value in target.items()
        }
    if (
        hasattr(target, "shape")
        and hasattr(source, "shape")
        and target.shape == source.shape
    ):
        return jnp.asarray(source, dtype=target.dtype)
    return target


def _apply_updates(params, updates):
    return jax.tree.map(operator.add, params, updates)


def _fidelity_loss(fidelity, eps: float):
    return -jnp.log(jnp.maximum(fidelity, jnp.asarray(eps, dtype=fidelity.dtype)))


def _phase_angle(phase, dtype) -> jnp.ndarray:
    return jnp.asarray(jnp.angle(phase), dtype=dtype)


def _lit_error_monitor(
    *,
    fidelity: float,
    source_norm: float,
    normalization: complex,
    eta: float,
    error_d: float,
) -> float:
    """Return the NQS-LIT fidelity error monitor with sample-estimated ``D``."""
    clipped_fidelity = float(np.clip(fidelity, 1e-12, 1.0))
    phi_norm = float(np.sqrt(max(source_norm, 0.0)))
    normalization_abs = max(abs(normalization), 1e-12)
    return lit_error_bound(
        clipped_fidelity,
        phi_norm=phi_norm,
        normalization_abs=normalization_abs,
        eta=eta,
        d_factor=max(float(error_d), 0.0),
    )
