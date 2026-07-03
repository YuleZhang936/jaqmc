# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole NQS-LIT workflow."""

from __future__ import annotations

import logging
import operator
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

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
    NQSLITSourceSums,
    ground_local_energy,
    local_action_ratio,
    molecular_electronic_dipole,
    nqs_lit_double_sampled_stats,
    nqs_lit_source_sampled_stats,
    nqs_lit_source_sampled_sums,
    nqs_lit_stats_from_source_sums,
    restore_params_from_checkpoint,
)
from jaqmc.response.spectrum import find_spectrum_peaks
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler, MCMCState
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
    axes: str = "xyz"
    peak_min_height_fraction: float = 0.05
    output_filename: str = "lit_spectrum.npz"
    preview_roots: int = 5
    scan_parallel: str = "auto"
    scan_parallel_workers: int = 0
    scan_parallel_procs_per_device: int = 1
    scan_parallel_min_points_per_worker: int = 2
    scan_parallel_worker: bool = False
    scan_parallel_worker_index: int = 0
    nqs_checkpoint_path: str = ""
    nqs_allow_untrained_ground: bool = False
    nqs_ground_energy: float | None = None
    nqs_source_center_steps: int = 4
    nqs_source_center_override: float | None = None
    nqs_source_norm_override: float | None = None
    nqs_source_burn_in: int = 20
    nqs_source_floor: float = 1e-4
    nqs_train_pool_batches: int = 32
    nqs_eval_pool_batches: int = 8
    nqs_pool_stride: int = 1
    nqs_train_update_batch_size: int = 0
    nqs_eval_batch_size: int = 0
    nqs_source_pool_dir: str = ""
    nqs_reuse_source_pool: bool = True
    nqs_save_source_pool: bool = True
    nqs_reweight_ess_fraction_min: float = 0.05
    nqs_direct_psi_train: bool = False
    nqs_direct_psi_burn_in: int = 5
    nqs_direct_psi_batches: int = 1
    nqs_direct_psi_train_batches: int | None = None
    nqs_direct_psi_eval_batches: int | None = None
    nqs_direct_psi_stride: int = 1
    nqs_direct_psi_precompile: bool = True
    nqs_direct_psi_persistent_sampler: bool = True
    nqs_energy_steps: int = 2
    nqs_burn_in: int = 20
    nqs_iterations: int = 200
    nqs_learning_rate: float = 1e-3
    nqs_sr_damping: float = 1e-3
    nqs_sr_max_norm: float | None = 0.1
    nqs_sr_score_eps: float = 1e-10
    nqs_warm_start_omega: float | None = -3.674932217565499
    nqs_warm_start_iterations: int = 100
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
    nqs_log_interval: int = 50


@dataclass(frozen=True)
class _ParallelWorker:
    index: int
    block: np.ndarray
    path: UPath
    log_path: UPath
    process: subprocess.Popen[bytes]
    device: str


@dataclass(frozen=True)
class _ParallelSharedSource:
    ground_energy: float
    source_center: float
    source_norm: float
    source_pool_dir: UPath


@dataclass
class _DirectPsiState:
    batched_data: BatchedData
    sampler_state: dict[str, Any]
    rng: jax.Array


class _DirectPsiCarry(NamedTuple):
    batched_data: BatchedData
    sampler_state: dict[str, Any]
    rng: jax.Array
    initialized: jax.Array


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
        omega = np.linspace(
            self.lit_config.omega_min,
            self.lit_config.omega_max,
            self.lit_config.omega_points,
        )
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
        for _ in range(self.lit_config.nqs_burn_in):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = ground_sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
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

        for axis_pos, axis in enumerate(axes):
            (
                source_center,
                axis_phi_norm,
                batched_data,
                sampler_state,
                rng,
            ) = self._estimate_source_stats(
                ground_params,
                batched_data,
                sampler_state,
                ground_sample_plan,
                rng,
                axis=axis,
            )
            source_centers[axis_pos] = source_center
            axis_source_norm[axis_pos] = axis_phi_norm
            logger.info(
                "axis=%s source_center=%.8e source_norm=%.8e",
                _AXIS_NAMES[axis],
                source_center,
                axis_phi_norm,
            )

            rng, response_rng = jax.random.split(rng)
            response_apply, response_params = self._make_response_ansatz(
                example,
                response_rng,
                ground_params,
                axis=axis,
                source_center=source_center,
            )
            loaded_pools = self._try_load_source_pools(
                batched_data,
                axis=axis,
                source_center=source_center,
            )
            if loaded_pools is None:
                source_sample_plan, source_state, axis_batched_data, rng = (
                    self._prepare_source_sampler(
                        self.sampler,
                        batched_data,
                        ground_params,
                        ground_logpsi,
                        rng,
                        axis=axis,
                        source_center=source_center,
                    )
                )
                train_pool, axis_batched_data, source_state, rng = (
                    self._load_or_collect_source_pool(
                        source_sample_plan,
                        ground_params,
                        axis_batched_data,
                        source_state,
                        rng,
                        axis=axis,
                        source_center=source_center,
                        split="train",
                        batches=self.lit_config.nqs_train_pool_batches,
                    )
                )
                eval_pool, axis_batched_data, source_state, rng = (
                    self._load_or_collect_source_pool(
                        source_sample_plan,
                        ground_params,
                        axis_batched_data,
                        source_state,
                        rng,
                        axis=axis,
                        source_center=source_center,
                        split="eval",
                        batches=self.lit_config.nqs_eval_pool_batches,
                    )
                )
            else:
                train_pool, eval_pool = loaded_pools
                axis_batched_data = batched_data
            logger.info(
                "axis=%s source_pool train=%d eval=%d",
                _AXIS_NAMES[axis],
                train_pool.batch_size,
                eval_pool.batch_size,
            )

            update_step = self._make_nqs_update_step(
                response_apply,
                ground_params,
                ground_logpsi,
                ground_energy,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
            )
            precompile_omega = (
                float(self.lit_config.nqs_warm_start_omega)
                if self.lit_config.nqs_warm_start_omega is not None
                else float(omega[0])
            )
            rng = self._coordinate_parallel_direct_precompile(
                update_step,
                response_params,
                train_pool,
                axis_batched_data,
                rng,
                axis=axis,
                omega=jnp.asarray(precompile_omega),
            )
            axis_direct_carry = update_step.init_direct_carry(axis_batched_data, rng)
            response_params, axis_direct_carry = self._warm_start_axis(
                update_step,
                response_params,
                train_pool,
                axis_direct_carry,
                axis=axis,
            )
            axis_start_response_params = response_params
            axis_start_batched_data = axis_direct_carry.batched_data
            rng = axis_direct_carry.rng
            logger.info(
                "axis=%s omega_points independent_response_start=true",
                _AXIS_NAMES[axis],
            )
            for omega_pos, omega_value in enumerate(omega):
                point_response_params = axis_start_response_params
                point_direct_carry = update_step.init_direct_carry(
                    axis_start_batched_data,
                    rng,
                )
                stats = None
                for iteration in range(self.lit_config.nqs_iterations):
                    point_response_params, stats, point_direct_carry = update_step(
                        point_response_params,
                        train_pool,
                        jnp.asarray(float(omega_value)),
                        point_direct_carry,
                        iteration,
                    )
                    if (
                        self.lit_config.nqs_log_interval > 0
                        and (iteration + 1) % self.lit_config.nqs_log_interval == 0
                    ):
                        logger.info(
                            "axis=%s omega=%.6f iter=%d loss=%.6e "
                            "fidelity=%.6f lit=%.6e ess=%.3f mode=%d",
                            _AXIS_NAMES[axis],
                            float(omega_value),
                            iteration + 1,
                            float(stats.loss),
                            float(stats.fidelity),
                            float(stats.lit),
                            float(stats.reweight_ess_fraction),
                            int(stats.estimator_mode),
                        )
                if stats is None:
                    stats = self._nqs_stats_chunked(
                        response_apply,
                        point_response_params,
                        ground_logpsi,
                        ground_params,
                        train_pool,
                        axis=axis,
                        source_center=source_center,
                        source_norm=axis_phi_norm,
                        ground_energy=ground_energy,
                        omega=float(omega_value),
                    )
                stats, _, rng = self._nqs_eval_stats_with_fallback(
                    response_apply,
                    point_response_params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    point_direct_carry.batched_data,
                    point_direct_carry.rng,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=jnp.asarray(float(omega_value)),
                )
                host_stats = jax.device_get(stats)
                lit[axis_pos, omega_pos] = float(host_stats.lit)
                broadened[axis_pos, omega_pos] = float(host_stats.broadened)
                fidelity[axis_pos, omega_pos] = float(host_stats.fidelity)
                residual_norm[axis_pos, omega_pos] = float(host_stats.residual_norm)
                action_norm[axis_pos, omega_pos] = float(host_stats.action_norm)
                source_norm[axis_pos, omega_pos] = float(host_stats.source_norm)
                error_bound_monitor[axis_pos, omega_pos] = _lit_error_monitor(
                    fidelity=float(host_stats.fidelity),
                    source_norm=float(host_stats.source_norm),
                    normalization=complex(host_stats.normalization),
                    eta=float(self.lit_config.eta),
                    error_d=float(host_stats.error_d),
                )
                error_d[axis_pos, omega_pos] = float(host_stats.error_d)
                reweight_ess[axis_pos, omega_pos] = float(host_stats.reweight_ess)
                reweight_ess_fraction[axis_pos, omega_pos] = float(
                    host_stats.reweight_ess_fraction
                )
                estimator_mode[axis_pos, omega_pos] = int(host_stats.estimator_mode)
                normalization[axis_pos, omega_pos] = complex(host_stats.normalization)
                correction_overlap[axis_pos, omega_pos] = complex(
                    host_stats.correction_overlap
                )

        total_broadened = np.sum(broadened, axis=0)
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
            nqs_reweight_ess_fraction_min=(
                self.lit_config.nqs_reweight_ess_fraction_min
            ),
            nqs_direct_psi_train=bool(self.lit_config.nqs_direct_psi_train),
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_train_batches=self._nqs_direct_psi_train_batches(),
            nqs_direct_psi_eval_batches=self._nqs_direct_psi_eval_batches(),
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_direct_psi_precompile=bool(self.lit_config.nqs_direct_psi_precompile),
            nqs_direct_psi_persistent_sampler=bool(
                self.lit_config.nqs_direct_psi_persistent_sampler
            ),
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
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
        if self.lit_config.scan_parallel_workers < 0:
            msg = "lit.scan_parallel_workers must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.scan_parallel_procs_per_device < 1:
            msg = "lit.scan_parallel_procs_per_device must be positive."
            raise ValueError(msg)
        if self.lit_config.scan_parallel_min_points_per_worker < 1:
            msg = "lit.scan_parallel_min_points_per_worker must be positive."
            raise ValueError(msg)
        if self.lit_config.scan_parallel_worker_index < 0:
            msg = "lit.scan_parallel_worker_index must be nonnegative."
            raise ValueError(msg)
        self._validate_chunk_config()
        if not 0.0 <= self.lit_config.nqs_reweight_ess_fraction_min <= 1.0:
            msg = (
                "lit.nqs_reweight_ess_fraction_min must be between 0 and 1, got "
                f"{self.lit_config.nqs_reweight_ess_fraction_min}."
            )
            raise ValueError(msg)
        self._validate_direct_psi_config()

    def _validate_direct_psi_config(self) -> None:
        if self.lit_config.nqs_direct_psi_burn_in < 0:
            msg = "lit.nqs_direct_psi_burn_in must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_direct_psi_batches < 1:
            msg = "lit.nqs_direct_psi_batches must be positive."
            raise ValueError(msg)
        if (
            self.lit_config.nqs_direct_psi_train_batches is not None
            and self.lit_config.nqs_direct_psi_train_batches < 1
        ):
            msg = "lit.nqs_direct_psi_train_batches must be positive."
            raise ValueError(msg)
        if (
            self.lit_config.nqs_direct_psi_eval_batches is not None
            and self.lit_config.nqs_direct_psi_eval_batches < 1
        ):
            msg = "lit.nqs_direct_psi_eval_batches must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_direct_psi_stride < 1:
            msg = "lit.nqs_direct_psi_stride must be positive."
            raise ValueError(msg)

    def _validate_chunk_config(self) -> None:
        if self.lit_config.nqs_train_update_batch_size < 0:
            msg = "lit.nqs_train_update_batch_size must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_eval_batch_size < 0:
            msg = "lit.nqs_eval_batch_size must be nonnegative."
            raise ValueError(msg)

    def _make_response_ansatz(
        self,
        example,
        response_rng,
        ground_params,
        *,
        axis: int,
        source_center: float,
    ):
        del axis, source_center
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
        return response.apply, response_params

    def _prepare_source_sampler(
        self,
        sampler,
        batched_data,
        ground_params,
        ground_logpsi,
        rng,
        *,
        axis: int,
        source_center: float,
    ):
        source_plan = SamplePlan(
            self._make_source_log_amplitude(axis, source_center, ground_logpsi),
            {"electrons": sampler},
        )
        rng, source_rng = jax.random.split(rng)
        source_state = source_plan.init(batched_data, source_rng)
        for _ in range(self.lit_config.nqs_source_burn_in):
            rng, source_rng = jax.random.split(rng)
            batched_data, _, source_state = source_plan.step(
                ground_params,
                batched_data,
                source_state,
                source_rng,
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
            for _ in range(stride):
                rng, sample_rng = jax.random.split(rng)
                batched_data, _, sampler_state = sample_plan.step(
                    params,
                    batched_data,
                    sampler_state,
                    sample_rng,
                )
            pool.append(batched_data)
        return _concat_batched_data(pool), batched_data, sampler_state, rng

    def _load_or_collect_source_pool(
        self,
        sample_plan: SamplePlan,
        params,
        batched_data,
        sampler_state,
        rng,
        *,
        axis: int,
        source_center: float,
        split: str,
        batches: int,
        pool_root: UPath | None = None,
    ):
        pool_path = self._source_pool_path(axis, split, root=pool_root)
        metadata = self._source_pool_metadata(axis, source_center)
        if self.lit_config.nqs_reuse_source_pool and pool_path.exists():
            try:
                pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
                logger.info(
                    "Loaded %s source pool for axis=%s from %s",
                    split,
                    _AXIS_NAMES[axis],
                    pool_path,
                )
                return pool, batched_data, sampler_state, rng
            except (KeyError, ValueError, OSError) as exc:
                logger.warning(
                    "Ignoring incompatible %s source pool %s: %s",
                    split,
                    pool_path,
                    exc,
                )

        pool, batched_data, sampler_state, rng = self._collect_sample_pool(
            sample_plan,
            params,
            batched_data,
            sampler_state,
            rng,
            batches=batches,
        )
        if self.lit_config.nqs_save_source_pool:
            _save_batched_pool(pool_path, pool, metadata=metadata)
            logger.info(
                "Saved %s source pool for axis=%s to %s",
                split,
                _AXIS_NAMES[axis],
                pool_path,
            )
        return pool, batched_data, sampler_state, rng

    def _try_load_source_pools(
        self,
        batched_data,
        *,
        axis: int,
        source_center: float,
        pool_root: UPath | None = None,
    ):
        if not self.lit_config.nqs_reuse_source_pool:
            return None
        metadata = self._source_pool_metadata(axis, source_center)
        loaded = []
        for split in ("train", "eval"):
            pool_path = self._source_pool_path(axis, split, root=pool_root)
            if not pool_path.exists():
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
            loaded.append(pool)
        return tuple(loaded)

    def _source_pool_path(
        self, axis: int, split: str, *, root: UPath | None = None
    ) -> UPath:
        if root is None:
            root = (
                UPath(self.lit_config.nqs_source_pool_dir)
                if self.lit_config.nqs_source_pool_dir
                else self.save_path / "source_pools"
            )
        return root / f"axis_{_AXIS_NAMES[axis]}_{split}.npz"

    def _source_pool_metadata(
        self,
        axis: int,
        source_center: float,
    ) -> dict[str, float]:
        return {
            "axis": float(axis),
            "source_center": float(source_center),
            "source_floor": float(self.lit_config.nqs_source_floor),
        }

    def _coordinate_parallel_direct_precompile(
        self,
        update_step,
        response_params,
        train_pool,
        fallback_data,
        rng,
        *,
        axis: int,
        omega,
    ):
        if not (
            self.lit_config.nqs_direct_psi_train
            and self._needs_direct_psi_estimator()
            and self.lit_config.nqs_direct_psi_precompile
        ):
            return rng
        precompile = getattr(update_step, "precompile_direct", None)
        if precompile is None:
            return rng
        return precompile(response_params, train_pool, fallback_data, rng, omega)

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        direct_carry,
        *,
        axis: int,
    ):
        if (
            self.lit_config.nqs_warm_start_omega is None
            or self.lit_config.nqs_warm_start_iterations <= 0
        ):
            return response_params, direct_carry
        stats = None
        for iteration in range(self.lit_config.nqs_warm_start_iterations):
            response_params, stats, direct_carry = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(self.lit_config.nqs_warm_start_omega)),
                direct_carry,
                iteration,
            )
        if stats is not None:
            logger.info(
                "axis=%s warm_start omega=%.6f iterations=%d "
                "fidelity=%.6f lit=%.6e ess=%.3f mode=%d",
                _AXIS_NAMES[axis],
                float(self.lit_config.nqs_warm_start_omega),
                self.lit_config.nqs_warm_start_iterations,
                float(stats.fidelity),
                float(stats.lit),
                float(stats.reweight_ess_fraction),
                int(stats.estimator_mode),
            )
        return response_params, direct_carry

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
        return checkpoint_step, ground_params, self._ground_complex_logpsi

    def _ground_complex_logpsi(self, params, data) -> jnp.ndarray:
        phase, log_abs = self.wf.phase_logpsi(params, data)
        return log_abs + 1j * _phase_angle(phase, log_abs.dtype)

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

    def _estimate_source_stats(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
    ):
        if (
            self.lit_config.nqs_source_center_override is not None
            and self.lit_config.nqs_source_norm_override is not None
        ):
            return (
                float(self.lit_config.nqs_source_center_override),
                float(self.lit_config.nqs_source_norm_override),
                batched_data,
                sampler_state,
                rng,
            )
        mean_values = []
        mean_square_values = []
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
            mean_square_values.append(float(jnp.mean(dipole**2)))
        mean = float(np.mean(mean_values))
        center = mean
        if self.lit_config.nqs_source_center_override is not None:
            center = float(self.lit_config.nqs_source_center_override)
        variance = float(np.mean(mean_square_values)) - 2.0 * center * mean + center**2
        norm = float(max(variance, 1e-12))
        if self.lit_config.nqs_source_norm_override is not None:
            norm = float(self.lit_config.nqs_source_norm_override)
        return center, norm, batched_data, sampler_state, rng

    def _make_source_log_amplitude(
        self,
        axis: int,
        source_center: float,
        ground_logpsi,
    ):
        floor = float(self.lit_config.nqs_source_floor)

        def log_amplitude(params, data):
            source = molecular_electronic_dipole(data, axis) - source_center
            return ground_logpsi(params, data) + jnp.log(
                jnp.maximum(jnp.abs(source), floor)
            )

        return log_amplitude

    def _make_nqs_update_step(  # noqa: C901
        self,
        response_apply,
        ground_params,
        ground_logpsi,
        ground_energy: float,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
    ):
        @jax.jit
        def source_update(response_params, batched_data, omega):
            stats, updates = self._source_sr_stats_and_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
            )
            response_params = _apply_updates(response_params, updates)
            loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
            return response_params, stats._replace(loss=loss)

        direct_train_collect_initial = self._make_direct_psi_pool_collector(
            response_apply,
            ground_logpsi,
            ground_params,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            batches=self._nqs_direct_psi_train_batches(),
            burn_in=max(0, int(self.lit_config.nqs_direct_psi_burn_in)),
        )
        direct_train_collect_continue = self._make_direct_psi_pool_collector(
            response_apply,
            ground_logpsi,
            ground_params,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            batches=self._nqs_direct_psi_train_batches(),
            burn_in=0,
        )

        @jax.jit
        def fallback_update(
            response_params,
            source_batched_data,
            direct_batched_data,
            direct_sampler_state,
            direct_rng,
            direct_initialized,
            omega,
        ):
            source_stats, source_updates, source_sums = (
                self._source_sr_stats_updates_and_sums(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    source_batched_data,
                    axis=axis,
                    source_center=source_center,
                    source_norm=source_norm,
                    ground_energy=ground_energy,
                    omega=omega,
                )
            )
            source_loss = _fidelity_loss(
                source_stats.fidelity,
                self.lit_config.nqs_sr_score_eps,
            )
            source_stats = source_stats._replace(loss=source_loss)
            threshold = jnp.asarray(
                self.lit_config.nqs_reweight_ess_fraction_min,
                dtype=source_stats.reweight_ess_fraction.dtype,
            )
            use_direct = source_stats.reweight_ess_fraction < threshold

            def source_branch(_):
                source_response_params = _apply_updates(response_params, source_updates)
                return (
                    source_response_params,
                    source_stats,
                    direct_batched_data,
                    direct_sampler_state,
                    direct_rng,
                    jnp.asarray(False),
                )

            def direct_branch(_):
                def collect_initial(_):
                    return direct_train_collect_initial(
                        response_params,
                        direct_batched_data,
                        direct_sampler_state,
                        direct_rng,
                        omega,
                    )

                def collect_continue(_):
                    return direct_train_collect_continue(
                        response_params,
                        direct_batched_data,
                        direct_sampler_state,
                        direct_rng,
                        omega,
                    )

                psi_pool, next_batched_data, next_sampler_state, next_rng = (
                    jax.lax.cond(
                        direct_initialized,
                        collect_continue,
                        collect_initial,
                        operand=None,
                    )
                )
                direct_stats, direct_updates = (
                    self._direct_sr_stats_and_updates_from_source_sums(
                        response_apply,
                        response_params,
                        ground_logpsi,
                        ground_params,
                        source_sums,
                        psi_pool,
                        axis=axis,
                        source_center=source_center,
                        source_norm=source_norm,
                        ground_energy=ground_energy,
                        omega=omega,
                    )
                )
                direct_response_params = _apply_updates(
                    response_params,
                    direct_updates,
                )
                direct_loss = _fidelity_loss(
                    direct_stats.fidelity,
                    self.lit_config.nqs_sr_score_eps,
                )
                return (
                    direct_response_params,
                    direct_stats._replace(loss=direct_loss),
                    next_batched_data,
                    next_sampler_state,
                    next_rng,
                    jnp.asarray(True),
                )

            return jax.lax.cond(
                use_direct,
                direct_branch,
                source_branch,
                operand=None,
            )

        direct_enabled = (
            self.lit_config.nqs_direct_psi_train and self._needs_direct_psi_estimator()
        )

        def update(
            response_params,
            batched_data,
            omega,
            direct_carry,
            batch_index: int = 0,
        ):
            if int(batch_index) == 0:
                direct_carry = direct_carry._replace(initialized=jnp.asarray(False))
            update_batch = _cyclic_batched_data_chunk(
                batched_data,
                self._nqs_train_update_batch_size(),
                batch_index,
            )
            if not direct_enabled:
                response_params, stats = source_update(
                    response_params,
                    update_batch,
                    omega,
                )
                return response_params, stats, direct_carry
            (
                response_params,
                stats,
                direct_batched_data,
                direct_sampler_state,
                direct_rng,
                direct_initialized,
            ) = fallback_update(
                response_params,
                update_batch,
                direct_carry.batched_data,
                direct_carry.sampler_state,
                direct_carry.rng,
                direct_carry.initialized,
                omega,
            )
            return (
                response_params,
                stats,
                _DirectPsiCarry(
                    batched_data=direct_batched_data,
                    sampler_state=direct_sampler_state,
                    rng=direct_rng,
                    initialized=direct_initialized,
                ),
            )

        def precompile_direct(response_params, batched_data, fallback_data, rng, omega):
            return self._precompile_direct_fallback_kernels(
                response_params,
                batched_data,
                fallback_data,
                rng,
                omega,
                axis=axis,
                fallback_update=fallback_update,
            )

        update.init_direct_carry = self._init_direct_psi_carry  # type: ignore[attr-defined]
        update.precompile_direct = precompile_direct  # type: ignore[attr-defined]
        return update

    def _precompile_direct_fallback_kernels(
        self,
        response_params,
        batched_data,
        fallback_data,
        rng,
        omega,
        *,
        axis: int,
        fallback_update,
    ):
        if not (
            self.lit_config.nqs_direct_psi_train
            and self._needs_direct_psi_estimator()
            and self.lit_config.nqs_direct_psi_precompile
        ):
            return rng
        update_batch = _cyclic_batched_data_chunk(
            batched_data,
            self._nqs_train_update_batch_size(),
            0,
        )
        direct_carry = self._init_direct_psi_carry(fallback_data, rng)
        fallback_update.lower(
            response_params,
            update_batch,
            direct_carry.batched_data,
            direct_carry.sampler_state,
            direct_carry.rng,
            direct_carry.initialized,
            omega,
        ).compile()
        logger.info(
            "Precompiled fused direct pi_Psi fallback step for axis=%s",
            _AXIS_NAMES[axis],
        )
        return rng

    def _source_sr_stats_updates_and_sums(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        score, ratio, source_weight, source_sums = (
            self._source_sampled_action_scores_and_sums(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
        )
        stats = nqs_lit_stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        updates = self._weighted_sr_updates_from_scores(
            response_params,
            score,
            ratio,
            source_weight,
        )
        return stats, updates, source_sums

    def _source_sr_stats_and_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        score, ratio, source_weight, source_sums = (
            self._source_sampled_action_scores_and_sums(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
        )
        stats = nqs_lit_stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        updates = self._weighted_sr_updates_from_scores(
            response_params,
            score,
            ratio,
            source_weight,
        )
        return stats, updates

    def _weighted_sr_updates_from_scores(
        self,
        response_params,
        score,
        ratio,
        source_weight,
    ):
        eps = jnp.asarray(self.lit_config.nqs_sr_score_eps, dtype=ratio.real.dtype)
        phi_weight = source_weight / jnp.maximum(jnp.sum(source_weight), eps)
        amplitude = jnp.sum(phi_weight * ratio)
        ratio_norm = jnp.sum(phi_weight * jnp.abs(ratio) ** 2)
        psi_weight = phi_weight * jnp.abs(ratio) ** 2 / jnp.maximum(ratio_norm, eps)
        score_mean_psi = jnp.sum(psi_weight[:, None] * score, axis=0, keepdims=True)
        centered_score = score - score_mean_psi
        score_covariance = jnp.sum(
            phi_weight[:, None] * ratio[:, None] * centered_score,
            axis=0,
        )
        grad_flat = 2.0 * jnp.real(
            jnp.conj(amplitude) * score_covariance / jnp.maximum(ratio_norm, eps)
        )
        weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
        score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)

        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(self.lit_config.nqs_sr_damping, dtype=grad_flat.dtype)
        damping = jnp.maximum(damping, jnp.asarray(1e-12, dtype=grad_flat.dtype))
        preconditioned = self._solve_sr_direction(score_aug, grad_flat, damping)
        scale = jnp.asarray(self.lit_config.nqs_learning_rate, dtype=grad_flat.dtype)
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=grad_flat.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm / (update_norm + jnp.asarray(1e-12, dtype=grad_flat.dtype)),
            )
        return unravel_fn(scale * preconditioned)

    def _weighted_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        score, ratio, source_weight = self._source_sampled_action_scores(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
        )
        return self._weighted_sr_updates_from_scores(
            response_params,
            score,
            ratio,
            source_weight,
        )

    def _direct_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_batched_data,
        psi_batched_data,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        source_ratio, source_weight = self._source_sampled_action_ratios(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_batched_data,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
        )
        eps = jnp.asarray(
            self.lit_config.nqs_sr_score_eps,
            dtype=source_ratio.real.dtype,
        )
        phi_weight = source_weight / jnp.maximum(jnp.sum(source_weight), eps)
        amplitude = jnp.sum(phi_weight * source_ratio)

        score, ratio, _ = self._source_sampled_action_scores(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            psi_batched_data,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
        )
        safe_ratio = jnp.where(
            jnp.abs(ratio) > eps,
            ratio,
            jnp.asarray(eps, dtype=ratio.real.dtype) + 0j,
        )
        hloc_conj = jnp.conj(amplitude / safe_ratio)
        finite = jnp.isfinite(jnp.real(hloc_conj)) & jnp.isfinite(jnp.imag(hloc_conj))
        hloc_conj = jnp.where(finite, hloc_conj, jnp.asarray(0.0, dtype=ratio.dtype))
        score = jnp.where(finite[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
        sample_count = jnp.maximum(score.shape[0], 1)
        score_mean = jnp.mean(score, axis=0, keepdims=True)
        centered_score = score - score_mean
        grad_flat = 2.0 * jnp.real(
            jnp.mean(centered_score * hloc_conj[:, None], axis=0)
        )
        score_scale = jnp.sqrt(jnp.asarray(sample_count, dtype=grad_flat.dtype))
        weighted_score = centered_score / score_scale
        score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)

        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(self.lit_config.nqs_sr_damping, dtype=grad_flat.dtype)
        damping = jnp.maximum(damping, jnp.asarray(1e-12, dtype=grad_flat.dtype))
        preconditioned = self._solve_sr_direction(score_aug, grad_flat, damping)
        scale = jnp.asarray(self.lit_config.nqs_learning_rate, dtype=grad_flat.dtype)
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=grad_flat.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm / (update_norm + jnp.asarray(1e-12, dtype=grad_flat.dtype)),
            )
        return unravel_fn(scale * preconditioned)

    def _direct_sr_stats_and_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_batched_data,
        psi_batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        source_sums = nqs_lit_source_sampled_sums(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_batched_data,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
        )
        return self._direct_sr_stats_and_updates_from_source_sums(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_sums,
            psi_batched_data,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
        )

    def _direct_sr_stats_and_updates_from_source_sums(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_sums,
        psi_batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        source_stats = nqs_lit_stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        score, ratio, _ = self._source_sampled_action_scores(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            psi_batched_data,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
        )
        eps = jnp.asarray(self.lit_config.nqs_sr_score_eps, dtype=ratio.real.dtype)
        safe_ratio = jnp.where(
            jnp.abs(ratio) > eps,
            ratio,
            jnp.asarray(eps, dtype=ratio.real.dtype) + 0j,
        )
        normalization = source_stats.normalization
        hloc = normalization / safe_ratio
        finite = jnp.isfinite(jnp.real(hloc)) & jnp.isfinite(jnp.imag(hloc))
        hloc = jnp.where(finite, hloc, jnp.asarray(0.0, dtype=hloc.dtype))
        fidelity = jnp.clip(jnp.real(jnp.mean(hloc)), 0.0, 1.0)
        action_norm = (
            jnp.asarray(source_norm, dtype=fidelity.dtype)
            * jnp.abs(normalization) ** 2
            / jnp.maximum(fidelity, eps)
        )

        hloc_conj = jnp.conj(hloc)
        score = jnp.where(finite[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
        sample_count = jnp.maximum(score.shape[0], 1)
        score_mean = jnp.mean(score, axis=0, keepdims=True)
        centered_score = score - score_mean
        grad_flat = 2.0 * jnp.real(
            jnp.mean(centered_score * hloc_conj[:, None], axis=0)
        )
        score_scale = jnp.sqrt(jnp.asarray(sample_count, dtype=grad_flat.dtype))
        weighted_score = centered_score / score_scale
        score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)

        _, unravel_fn = ravel_pytree(response_params)
        damping = jnp.asarray(self.lit_config.nqs_sr_damping, dtype=grad_flat.dtype)
        damping = jnp.maximum(damping, jnp.asarray(1e-12, dtype=grad_flat.dtype))
        preconditioned = self._solve_sr_direction(score_aug, grad_flat, damping)
        scale = jnp.asarray(self.lit_config.nqs_learning_rate, dtype=grad_flat.dtype)
        if self.lit_config.nqs_sr_max_norm is not None:
            update_norm = jnp.linalg.norm(preconditioned)
            max_norm = jnp.asarray(
                self.lit_config.nqs_sr_max_norm,
                dtype=grad_flat.dtype,
            )
            scale = jnp.minimum(
                scale,
                max_norm / (update_norm + jnp.asarray(1e-12, dtype=grad_flat.dtype)),
            )
        stats = source_stats._replace(
            loss=1.0 - fidelity,
            fidelity=fidelity,
            action_norm=jnp.real(action_norm),
            estimator_mode=jnp.asarray(1, dtype=jnp.int32),
        )
        return stats, unravel_fn(scale * preconditioned)

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

    def _source_sampled_action_scores_and_sums(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        data = batched_data.data
        score_eps = float(self.lit_config.nqs_sr_score_eps)

        def action_score_and_aux(params, one):
            def split_log_action(local_params):
                local_action, local_response_ratio, local_eloc_response = (
                    local_action_ratio(
                        response_apply,
                        local_params,
                        ground_logpsi,
                        ground_params,
                        one,
                        ground_energy=ground_energy,
                        omega=omega,
                        eta=self.lit_config.eta,
                    )
                )
                safe_action = jnp.where(
                    jnp.abs(local_action) > score_eps,
                    local_action,
                    jnp.asarray(score_eps, dtype=local_action.real.dtype) + 0j,
                )
                value = jnp.log(safe_action)
                return jnp.stack([jnp.real(value), jnp.imag(value)]), (
                    local_action,
                    local_response_ratio,
                    local_eloc_response,
                )

            jac, aux = jax.jacrev(split_log_action, has_aux=True)(params)
            score_tree = jax.tree.map(lambda leaf: leaf[0] + 1j * leaf[1], jac)
            action, response_ratio, eloc_response = aux
            return action, response_ratio, eloc_response, score_tree

        action, response_ratio, eloc_response, score_tree = jax.vmap(
            lambda one: action_score_and_aux(response_params, one),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        score = _flatten_batched_tree(score_tree, action.shape[0])
        dipole = jax.vmap(
            lambda one: molecular_electronic_dipole(one, axis),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
        floor = jnp.asarray(self.lit_config.nqs_source_floor, dtype=dipole.dtype)

        stats_eps = jnp.asarray(1e-12, dtype=dipole.dtype)
        stats_sampled_source = jnp.maximum(jnp.abs(source), floor)
        stats_source_weight = (
            jnp.abs(source) / jnp.maximum(stats_sampled_source, stats_eps)
        ) ** 2
        finite_sums = (
            jnp.isfinite(jnp.real(action))
            & jnp.isfinite(jnp.imag(action))
            & jnp.isfinite(jnp.real(response_ratio))
            & jnp.isfinite(jnp.imag(response_ratio))
            & jnp.isfinite(source)
            & jnp.isfinite(stats_source_weight)
        )
        stats_action = jnp.where(
            finite_sums,
            action,
            jnp.asarray(0.0, dtype=action.dtype),
        )
        stats_response_ratio = jnp.where(
            finite_sums,
            response_ratio,
            jnp.asarray(0.0, dtype=response_ratio.dtype),
        )
        stats_source_weight = jnp.where(
            finite_sums,
            stats_source_weight,
            jnp.asarray(0.0, dtype=stats_source_weight.dtype),
        )
        safe_source_stats = jnp.where(
            jnp.abs(source) > stats_eps,
            source,
            stats_eps * jnp.where(source < 0, -1.0, 1.0),
        )
        stats_ratio = stats_action / safe_source_stats
        psi_weight_unnormalized = stats_source_weight * jnp.abs(stats_ratio) ** 2
        shift = jnp.asarray(omega, dtype=stats_response_ratio.real.dtype) + 1j * (
            jnp.asarray(self.lit_config.eta, dtype=stats_response_ratio.real.dtype)
        )
        hbar_response_ratio = stats_action + shift * stats_response_ratio
        response_over_source = stats_response_ratio / safe_source_stats
        hbar_over_source = hbar_response_ratio / safe_source_stats
        eloc_finite = jnp.isfinite(jnp.real(eloc_response))
        eloc_response = jnp.where(
            eloc_finite,
            eloc_response,
            jnp.asarray(0.0, dtype=eloc_response.dtype),
        )
        sample_count = jnp.asarray(action.shape[0], dtype=stats_source_weight.dtype)
        source_sums = NQSLITSourceSums(
            sample_count=sample_count,
            weight_sum=jnp.sum(stats_source_weight),
            valid_sample_count=jnp.sum(
                stats_source_weight > jnp.asarray(0.0, dtype=stats_source_weight.dtype)
            ),
            ratio_sum=jnp.sum(stats_source_weight * stats_ratio),
            ratio_abs2_sum=jnp.sum(stats_source_weight * jnp.abs(stats_ratio) ** 2),
            psi_weight_sum=jnp.sum(psi_weight_unnormalized),
            psi_weight_sq_sum=jnp.sum(psi_weight_unnormalized**2),
            response_conj_over_source_sum=jnp.sum(
                stats_source_weight * jnp.conj(stats_response_ratio) / safe_source_stats
            ),
            response_over_source_abs2_sum=jnp.sum(
                stats_source_weight * jnp.abs(response_over_source) ** 2
            ),
            hbar_over_source_sum=jnp.sum(stats_source_weight * hbar_over_source),
            hbar_over_source_abs2_sum=jnp.sum(
                stats_source_weight * jnp.abs(hbar_over_source) ** 2
            ),
            ground_energy_sum=jnp.real(jnp.sum(eloc_response)),
        )

        safe_source_score = jnp.where(
            jnp.abs(source) > score_eps,
            source,
            jnp.asarray(score_eps, dtype=source.dtype)
            * jnp.where(source < 0, -1.0, 1.0),
        )
        score_sampled_source = jnp.maximum(jnp.abs(source), floor)
        source_weight = (
            jnp.abs(source) / jnp.maximum(score_sampled_source, score_eps)
        ) ** 2
        ratio = action / safe_source_score
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
        return score, ratio, source_weight, source_sums

    def _source_sampled_action_scores(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        data = batched_data.data
        score_eps = float(self.lit_config.nqs_sr_score_eps)

        def action_and_score(params, one):
            action, _, _ = local_action_ratio(
                response_apply,
                params,
                ground_logpsi,
                ground_params,
                one,
                ground_energy=ground_energy,
                omega=omega,
                eta=self.lit_config.eta,
            )

            def split_log_action(local_params):
                local_action, _, _ = local_action_ratio(
                    response_apply,
                    local_params,
                    ground_logpsi,
                    ground_params,
                    one,
                    ground_energy=ground_energy,
                    omega=omega,
                    eta=self.lit_config.eta,
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
        dipole = jax.vmap(
            lambda one: molecular_electronic_dipole(one, axis),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
        safe_source = jnp.where(
            jnp.abs(source) > score_eps,
            source,
            jnp.asarray(score_eps, dtype=source.dtype)
            * jnp.where(source < 0, -1.0, 1.0),
        )
        sampled_source = jnp.maximum(
            jnp.abs(source),
            jnp.asarray(self.lit_config.nqs_source_floor, dtype=dipole.dtype),
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
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        data = batched_data.data
        eps = float(self.lit_config.nqs_sr_score_eps)
        action = jax.vmap(
            lambda one: local_action_ratio(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                one,
                ground_energy=ground_energy,
                omega=omega,
                eta=self.lit_config.eta,
            )[0],
            in_axes=(batched_data.vmap_axis,),
        )(data)
        dipole = jax.vmap(
            lambda one: molecular_electronic_dipole(one, axis),
            in_axes=(batched_data.vmap_axis,),
        )(data)
        source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
        safe_source = jnp.where(
            jnp.abs(source) > eps,
            source,
            jnp.asarray(eps, dtype=source.dtype) * jnp.where(source < 0, -1.0, 1.0),
        )
        sampled_source = jnp.maximum(
            jnp.abs(source),
            jnp.asarray(self.lit_config.nqs_source_floor, dtype=dipole.dtype),
        )
        source_weight = (jnp.abs(source) / jnp.maximum(sampled_source, eps)) ** 2
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

    def _nqs_loss(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        stats = nqs_lit_source_sampled_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
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
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        return nqs_lit_source_sampled_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            batched_data,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
        )

    def _nqs_stats_chunked(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        chunk_size = self._nqs_eval_batch_size()

        @jax.jit
        def chunk_sums(local_params, chunk, local_omega):
            return nqs_lit_source_sampled_sums(
                response_apply,
                local_params,
                ground_logpsi,
                ground_params,
                chunk,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=local_omega,
                eta=self.lit_config.eta,
                source_floor=self.lit_config.nqs_source_floor,
            )

        total_sums = None
        for chunk in _batched_data_chunks(batched_data, chunk_size):
            sums = chunk_sums(response_params, chunk, omega)
            total_sums = (
                sums if total_sums is None else _add_source_sums(total_sums, sums)
            )
        if total_sums is None:
            msg = "Cannot evaluate NQS-LIT stats with an empty source pool."
            raise ValueError(msg)
        return nqs_lit_stats_from_source_sums(
            total_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )

    def _nqs_double_stats(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_batched_data,
        psi_batched_data,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        return nqs_lit_double_sampled_stats(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_batched_data,
            psi_batched_data,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
            eta=self.lit_config.eta,
            source_floor=self.lit_config.nqs_source_floor,
        )

    def _nqs_eval_stats_with_fallback(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        source_pool,
        fallback_data,
        rng,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega,
    ):
        stats = self._nqs_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            source_pool,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=omega,
        )
        if not self._should_use_direct_psi(stats):
            return stats, fallback_data, rng
        psi_pool, fallback_data, rng = self._collect_direct_psi_pool(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            fallback_data,
            rng,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            omega=omega,
            batches=self._nqs_direct_psi_eval_batches(),
        )
        return (
            self._nqs_double_stats(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                source_pool,
                psi_pool,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
            ),
            fallback_data,
            rng,
        )

    def _should_use_direct_psi(self, stats) -> bool:
        threshold = float(self.lit_config.nqs_reweight_ess_fraction_min)
        if threshold <= 0.0:
            return False
        return float(jax.device_get(stats.reweight_ess_fraction)) < threshold

    def _needs_direct_psi_estimator(self) -> bool:
        return float(self.lit_config.nqs_reweight_ess_fraction_min) > 0.0

    def _nqs_direct_psi_train_batches(self) -> int:
        configured = self.lit_config.nqs_direct_psi_train_batches
        if configured is not None:
            return int(configured)
        return int(self.lit_config.nqs_direct_psi_batches)

    def _nqs_direct_psi_eval_batches(self) -> int:
        configured = self.lit_config.nqs_direct_psi_eval_batches
        if configured is not None:
            return int(configured)
        return int(self.lit_config.nqs_direct_psi_batches)

    def _nqs_train_update_batch_size(self) -> int:
        configured = int(self.lit_config.nqs_train_update_batch_size)
        if configured > 0:
            return configured
        return max(1, int(self.config.batch_size))

    def _nqs_eval_batch_size(self) -> int:
        configured = int(self.lit_config.nqs_eval_batch_size)
        if configured > 0:
            return configured
        return max(1, int(self.config.batch_size))

    def _init_direct_psi_state(self, batched_data, rng) -> _DirectPsiState:
        rng, sample_rng = jax.random.split(rng)
        sampler_state = {
            "electrons": self.sampler.init(
                batched_data.data.subset(("electrons",)),
                sample_rng,
            )
        }
        return _DirectPsiState(
            batched_data=batched_data,
            sampler_state=sampler_state,
            rng=rng,
        )

    def _init_direct_psi_carry(self, batched_data, rng) -> _DirectPsiCarry:
        state = self._init_direct_psi_state(batched_data, rng)
        return _DirectPsiCarry(
            batched_data=state.batched_data,
            sampler_state=state.sampler_state,
            rng=state.rng,
            initialized=jnp.asarray(False),
        )

    def _collect_direct_psi_pool_from_state(
        self,
        collector,
        response_params,
        direct_state: _DirectPsiState,
        *,
        omega,
    ) -> tuple[BatchedData, _DirectPsiState]:
        pool, batched_data, sampler_state, rng = collector(
            response_params,
            direct_state.batched_data,
            direct_state.sampler_state,
            direct_state.rng,
            omega,
        )
        return pool, _DirectPsiState(
            batched_data=batched_data,
            sampler_state=sampler_state,
            rng=rng,
        )

    def _single_device_mcmc_step(self, batch_log_prob, data, state, rng):
        logprob = batch_log_prob(data).real
        if logprob.ndim != 1:
            raise ValueError(
                f"log_amplitude should return a scalar, got shape {logprob.shape[1:]}."
            )
        num_accepts = jnp.array(0.0)
        data, _, _, num_accepts = jax.lax.fori_loop(
            0,
            int(self.sampler.steps),
            lambda _, x: self.sampler._mh_update(
                batch_log_prob,
                *x,
                stddev=state.stddev,
            ),
            (data, rng, logprob, num_accepts),
        )
        pmove = jnp.sum(num_accepts) / (int(self.sampler.steps) * logprob.shape[0])

        stddev, pmoves, counter = state
        counter += 1
        t_since_mcmc_update = counter % int(self.sampler.adapt_frequency)
        pmoves = pmoves.at[t_since_mcmc_update].set(pmove)
        stddev = jnp.where(
            t_since_mcmc_update == 0,
            jnp.where(
                jnp.mean(pmoves) > self.sampler.pmove_range[1],
                stddev * 1.1,
                jnp.where(
                    jnp.mean(pmoves) < self.sampler.pmove_range[0],
                    stddev / 1.1,
                    stddev,
                ),
            ),
            stddev,
        )
        new_state = MCMCState(counter=counter, pmoves=pmoves, stddev=stddev)
        return data, {"pmove": pmove}, new_state

    def _make_direct_psi_pool_collector(
        self,
        response_apply,
        ground_logpsi,
        ground_params,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        batches: int,
        burn_in: int,
    ):
        del axis, source_center
        batches = max(1, int(batches))
        burn_in = max(0, int(burn_in))
        stride = max(1, int(self.lit_config.nqs_direct_psi_stride))
        eps = float(self.lit_config.nqs_sr_score_eps)

        def action_log_amplitude(response_params, data, omega):
            action, _, _ = local_action_ratio(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                data,
                ground_energy=ground_energy,
                omega=omega,
                eta=self.lit_config.eta,
            )
            return ground_logpsi(ground_params, data) + jnp.log(
                jnp.maximum(jnp.abs(action), eps)
            )

        @jax.jit
        def collect(response_params, batched_data, sampler_state, rng, omega):
            def batch_log_prob(data_part, base_data):
                merged = base_data.merge(data_part)
                values = jax.vmap(
                    lambda one: action_log_amplitude(response_params, one, omega),
                    in_axes=(batched_data.vmap_axis,),
                )(merged)
                return 2.0 * values

            def one_sampler_step(carry, _):
                local_batched_data, local_sampler_state, local_rng = carry
                local_rng, sample_rng = jax.random.split(local_rng)
                base_data = local_batched_data.data
                data_part, _, electron_state = self._single_device_mcmc_step(
                    lambda x: batch_log_prob(x, base_data),
                    base_data.subset(("electrons",)),
                    local_sampler_state["electrons"],
                    sample_rng,
                )
                next_batched_data = replace(
                    local_batched_data,
                    data=base_data.merge(data_part),
                )
                next_sampler_state = {"electrons": electron_state}
                return (next_batched_data, next_sampler_state, local_rng), None

            carry = (batched_data, sampler_state, rng)
            if burn_in > 0:
                carry, _ = jax.lax.scan(
                    one_sampler_step,
                    carry,
                    None,
                    length=burn_in,
                )

            def collect_one_batch(carry, _):
                carry, _ = jax.lax.scan(
                    one_sampler_step,
                    carry,
                    None,
                    length=stride,
                )
                return carry, carry[0]

            carry, scanned_pool = jax.lax.scan(
                collect_one_batch,
                carry,
                None,
                length=batches,
            )
            batched_data, sampler_state, rng = carry
            pool = _flatten_scanned_batched_pool(scanned_pool, batched_data)
            pool = jax.tree.map(jax.lax.stop_gradient, pool)
            batched_data = jax.tree.map(jax.lax.stop_gradient, batched_data)
            return pool, batched_data, sampler_state, rng

        return collect

    def _collect_direct_psi_pool(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        rng,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
        batches: int,
    ):
        collector = self._make_direct_psi_pool_collector(
            response_apply,
            ground_logpsi,
            ground_params,
            axis=axis,
            source_center=source_center,
            ground_energy=ground_energy,
            batches=batches,
            burn_in=max(0, int(self.lit_config.nqs_direct_psi_burn_in)),
        )
        direct_state = self._init_direct_psi_state(batched_data, rng)
        pool, direct_state = self._collect_direct_psi_pool_from_state(
            collector,
            response_params,
            direct_state,
            omega=omega,
        )
        return pool, direct_state.batched_data, direct_state.rng

    def _make_action_log_amplitude(
        self,
        response_apply,
        ground_logpsi,
        ground_params,
        *,
        axis: int,
        source_center: float,
        ground_energy: float,
        omega,
    ):
        del axis, source_center
        eps = float(self.lit_config.nqs_sr_score_eps)

        def log_amplitude(response_params, data):
            action, _, _ = local_action_ratio(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                data,
                ground_energy=ground_energy,
                omega=omega,
                eta=self.lit_config.eta,
            )
            return ground_logpsi(ground_params, data) + jnp.log(
                jnp.maximum(jnp.abs(action), eps)
            )

        return log_amplitude

    def _log_nqs_summary(self, output_path: str, peaks, fidelity: np.ndarray) -> None:
        logger.info("Wrote NQS-LIT spectrum to %s", output_path)
        logger.info(
            "NQS-LIT fidelity range: min=%.6f max=%.6f",
            float(np.min(fidelity)),
            float(np.max(fidelity)),
        )
        for peak in peaks[: self.lit_config.preview_roots]:
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
        if self.lit_config.omega_points < 2:
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
                    self.lit_config.omega_points
                    / self.lit_config.scan_parallel_min_points_per_worker
                )
            ),
        )
        return max(1, min(available_slots, worker_limit, points_limit))

    def _run_parallel_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = np.linspace(
            self.lit_config.omega_min,
            self.lit_config.omega_max,
            self.lit_config.omega_points,
        )
        device_ids = _visible_cuda_devices()
        worker_count = self._parallel_worker_count(device_ids)
        blocks = _split_omega_blocks(self.lit_config.omega_points, worker_count)
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
        base_config_path = parallel_root / "base_config.yaml"
        base_config_path.write_text(self.cfg.to_yaml())
        compile_cache_dir = parallel_root / "jax_compile_cache"
        run_seed = (
            int(self.config.seed) if self.config.seed is not None else int(time.time())
        )
        shared_pool_root = (
            UPath(self.lit_config.nqs_source_pool_dir)
            if self.lit_config.nqs_source_pool_dir
            else parallel_root / "source_pools"
        )
        shared_source = self._prepare_parallel_shared_source(
            axes,
            run_seed=run_seed,
            source_pool_dir=shared_pool_root,
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
                log_path = part_dir / "worker.log"
                command = self._parallel_worker_command(
                    base_config_path,
                    part_dir,
                    omega[block],
                    run_seed=run_seed,
                    shared_source=shared_source,
                    worker_index=worker_index,
                )
                device = worker_devices[worker_index]
                env = _parallel_worker_env(
                    device,
                    cache_dir=compile_cache_dir / f"worker_{worker_index:03d}",
                    autotune_cache_dir=parallel_root
                    / "xla_autotune_cache"
                    / f"worker_{worker_index:03d}",
                )
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

    def _prepare_parallel_shared_source(
        self,
        axes: tuple[int, ...],
        *,
        run_seed: int,
        source_pool_dir: UPath,
    ) -> _ParallelSharedSource | None:
        if len(axes) != 1:
            logger.info(
                "Parallel shared source pools are only used for single-axis "
                "scans; falling back to per-worker source preparation."
            )
            return None

        axis = axes[0]
        fast_source = self._parallel_shared_source_from_overrides(
            axis,
            run_seed=run_seed,
            source_pool_dir=source_pool_dir,
        )
        if fast_source is not None:
            return fast_source

        logger.info(
            "Preparing shared source pool for parallel LIT workers on axis=%s.",
            _AXIS_NAMES[axis],
        )
        rng = jax.random.PRNGKey(run_seed)
        rng, data_rng, ground_rng, _, sample_rng = jax.random.split(rng, 5)
        batched_data = data_init(self.system_config, self.config.batch_size, data_rng)
        example = batched_data.unbatched_example()
        checkpoint_step, ground_params, ground_logpsi = self._resolve_nqs_ground_state(
            example, ground_rng
        )
        ground_sample_plan = SamplePlan(ground_logpsi, {"electrons": self.sampler})
        sampler_state = ground_sample_plan.init(batched_data, sample_rng)
        for _ in range(self.lit_config.nqs_burn_in):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = ground_sample_plan.step(
                ground_params,
                batched_data,
                sampler_state,
                sample_rng,
            )

        ground_energy, batched_data, sampler_state, rng = self._resolve_ground_energy(
            ground_logpsi,
            ground_params,
            batched_data,
            sampler_state,
            ground_sample_plan,
            rng,
        )
        source_center, source_norm, batched_data, sampler_state, rng = (
            self._estimate_source_stats(
                ground_params,
                batched_data,
                sampler_state,
                ground_sample_plan,
                rng,
                axis=axis,
            )
        )

        loaded_pools = self._try_load_source_pools(
            batched_data,
            axis=axis,
            source_center=source_center,
            pool_root=source_pool_dir,
        )
        if loaded_pools is None:
            source_sample_plan, source_state, axis_batched_data, rng = (
                self._prepare_source_sampler(
                    self.sampler,
                    batched_data,
                    ground_params,
                    ground_logpsi,
                    rng,
                    axis=axis,
                    source_center=source_center,
                )
            )
            train_pool, axis_batched_data, source_state, rng = (
                self._load_or_collect_source_pool(
                    source_sample_plan,
                    ground_params,
                    axis_batched_data,
                    source_state,
                    rng,
                    axis=axis,
                    source_center=source_center,
                    split="train",
                    batches=self.lit_config.nqs_train_pool_batches,
                    pool_root=source_pool_dir,
                )
            )
            eval_pool, _, _, _ = self._load_or_collect_source_pool(
                source_sample_plan,
                ground_params,
                axis_batched_data,
                source_state,
                rng,
                axis=axis,
                source_center=source_center,
                split="eval",
                batches=self.lit_config.nqs_eval_pool_batches,
                pool_root=source_pool_dir,
            )
        else:
            train_pool, eval_pool = loaded_pools

        logger.info(
            "Prepared shared source axis=%s checkpoint=%d energy=%.10f "
            "source_center=%.8e source_norm=%.8e train=%d eval=%d dir=%s",
            _AXIS_NAMES[axis],
            checkpoint_step,
            ground_energy,
            source_center,
            source_norm,
            train_pool.batch_size,
            eval_pool.batch_size,
            source_pool_dir,
        )
        return _ParallelSharedSource(
            ground_energy=float(ground_energy),
            source_center=float(source_center),
            source_norm=float(source_norm),
            source_pool_dir=source_pool_dir,
        )

    def _parallel_shared_source_from_overrides(
        self,
        axis: int,
        *,
        run_seed: int,
        source_pool_dir: UPath,
    ) -> _ParallelSharedSource | None:
        if (
            self.lit_config.nqs_ground_energy is None
            or self.lit_config.nqs_source_center_override is None
            or self.lit_config.nqs_source_norm_override is None
        ):
            return None

        rng = jax.random.PRNGKey(run_seed)
        _, data_rng, _, _, _ = jax.random.split(rng, 5)
        batched_data = data_init(self.system_config, self.config.batch_size, data_rng)
        source_center = float(self.lit_config.nqs_source_center_override)
        source_norm = float(self.lit_config.nqs_source_norm_override)
        loaded_pools = self._try_load_source_pools(
            batched_data,
            axis=axis,
            source_center=source_center,
            pool_root=source_pool_dir,
        )
        if loaded_pools is None:
            return None

        train_pool, eval_pool = loaded_pools
        ground_energy = float(self.lit_config.nqs_ground_energy)
        logger.info(
            "Using configured shared source axis=%s energy=%.10f "
            "source_center=%.8e source_norm=%.8e train=%d eval=%d dir=%s",
            _AXIS_NAMES[axis],
            ground_energy,
            source_center,
            source_norm,
            train_pool.batch_size,
            eval_pool.batch_size,
            source_pool_dir,
        )
        return _ParallelSharedSource(
            ground_energy=ground_energy,
            source_center=source_center,
            source_norm=source_norm,
            source_pool_dir=source_pool_dir,
        )

    def _parallel_worker_command(
        self,
        base_config_path: UPath,
        part_dir: UPath,
        block_omega: np.ndarray,
        *,
        run_seed: int,
        shared_source: _ParallelSharedSource | None = None,
        worker_index: int = 0,
    ) -> list[str]:
        source_pool_dir = (
            shared_source.source_pool_dir
            if shared_source is not None
            else part_dir / "source_pools"
        )
        command = [
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
            f"lit.nqs_source_pool_dir={source_pool_dir}",
            f"lit.scan_parallel_worker_index={int(worker_index)}",
        ]
        if shared_source is not None:
            command.extend(
                [
                    f"lit.nqs_ground_energy={float(shared_source.ground_energy)!r}",
                    "lit.nqs_source_center_steps=0",
                    "lit.nqs_source_center_override="
                    f"{float(shared_source.source_center)!r}",
                    "lit.nqs_source_norm_override="
                    f"{float(shared_source.source_norm)!r}",
                ]
            )
        return command

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

        total_broadened = np.sum(combined["broadened"], axis=0)
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
            nqs_reweight_ess_fraction_min=(
                self.lit_config.nqs_reweight_ess_fraction_min
            ),
            nqs_direct_psi_train=bool(self.lit_config.nqs_direct_psi_train),
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_train_batches=self._nqs_direct_psi_train_batches(),
            nqs_direct_psi_eval_batches=self._nqs_direct_psi_eval_batches(),
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_direct_psi_precompile=bool(self.lit_config.nqs_direct_psi_precompile),
            nqs_direct_psi_persistent_sampler=bool(
                self.lit_config.nqs_direct_psi_persistent_sampler
            ),
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            source_centers=np.mean(source_centers_blocks, axis=0),
            axis_source_norm=np.mean(axis_source_norm_blocks, axis=0),
            source_centers_blocks=source_centers_blocks,
            axis_source_norm_blocks=axis_source_norm_blocks,
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


def _parallel_worker_env(
    device_id: str,
    *,
    cache_dir: UPath | None = None,
    autotune_cache_dir: UPath | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
        env.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
        env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    if autotune_cache_dir is not None:
        autotune_cache_dir.mkdir(parents=True, exist_ok=True)
        (autotune_cache_dir / "tmp").mkdir(parents=True, exist_ok=True)
        flag = f"--xla_gpu_per_fusion_autotune_cache_dir={autotune_cache_dir}"
        xla_flags = env.get("XLA_FLAGS", "")
        if "xla_gpu_per_fusion_autotune_cache_dir" not in xla_flags:
            env["XLA_FLAGS"] = f"{xla_flags} {flag}".strip()
    return env


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _two_spin_tuple(values) -> tuple[int, int]:
    nspins = tuple(int(value) for value in values)
    if len(nspins) != 2:
        msg = f"Expected two spin populations, got {nspins}."
        raise ValueError(msg)
    return nspins


def _save_npz(path: UPath, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f_out:
        np.savez(f_out, **payload)  # type: ignore[arg-type]


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


def _flatten_scanned_batched_pool(scanned_pool: BatchedData, reference: BatchedData):
    updates = {}
    for field_name in reference.fields_with_batch:
        updates[field_name] = jax.tree.map(
            lambda value: jnp.reshape(
                value,
                (value.shape[0] * value.shape[1], *value.shape[2:]),
            ),
            getattr(scanned_pool.data, field_name),
        )
    return reference.__class__(
        data=reference.data.merge(updates),
        fields_with_batch=reference.fields_with_batch,
    )


def _tile_batched_data(pool: BatchedData, repeats: int) -> BatchedData:
    repeats = max(1, int(repeats))
    if repeats == 1:
        return pool
    updates = {}
    for field_name in pool.fields_with_batch:
        updates[field_name] = jax.tree.map(
            lambda value: jnp.concatenate([value] * repeats, axis=0),
            getattr(pool.data, field_name),
        )
    return pool.__class__(
        data=pool.data.merge(updates),
        fields_with_batch=pool.fields_with_batch,
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
            value = jnp.asarray(npf[field_name])
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
