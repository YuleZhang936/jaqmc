# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole NQS-LIT workflow."""

from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import numpy as np
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from jax import scipy as jsp
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

try:
    from jax import enable_x64 as _enable_x64
except ImportError:
    from jax.experimental import enable_x64 as _enable_x64  # type: ignore[no-redef]

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
    omega_values: tuple[float, ...] = field(default_factory=tuple)
    axes: str = "xyz"
    peak_min_height_fraction: float = 0.05
    output_filename: str = "lit_spectrum.npz"
    preview_roots: int = 5
    # Kept for one release so old configuration files still deserialize.  Any
    # non-serial value is rejected because frequency continuation is a single
    # predecessor chain.
    scan_parallel: str = "off"
    scan_parallel_workers: int = 0
    scan_parallel_procs_per_device: int = 1
    scan_parallel_min_points_per_worker: int = 2
    scan_parallel_remote_hosts: tuple[str, ...] = field(default_factory=tuple)
    scan_parallel_remote_root: str = ""
    scan_parallel_remote_python: str = ""
    scan_parallel_ssh_options: tuple[str, ...] = field(
        default_factory=lambda: (
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
        )
    )
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
    nqs_reweight_ess_fraction_min: float = 0.0
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
    nqs_reverse_kl_weight: float = 1.0
    nqs_spring_epsilon: float = 1e-3
    nqs_spring_decay: float = 0.99
    nqs_spring_damping_floor: float = 1e-12
    nqs_sr_damping: float | None = None
    nqs_sr_max_norm: float | None = 0.1
    nqs_sr_score_eps: float = 1e-10
    nqs_warm_start_omega: float | None = -3.674932217565499
    nqs_warm_start_iterations: int = 100
    nqs_continuation_iterations: int = 100
    nqs_continuation_step_fraction: float = 0.2
    nqs_continuation_fidelity_retention: float = 0.95
    nqs_continuation_min_step: float | None = None
    nqs_continuation_max_points: int = 256
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
    nqs_selection_interval: int = 50
    nqs_log_interval: int = 50


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
    use_direct: jax.Array


class _SpringState(NamedTuple):
    """Unscaled SPRING direction associated with the current frequency."""

    previous_direction: jax.Array


class _NQSUpdateCarry(NamedTuple):
    direct: _DirectPsiCarry
    spring: _SpringState


class _ContinuationRecord(NamedTuple):
    omega: float
    optimized: bool
    selected_iteration: int
    stats: Any
    inherited_fidelity: float
    step: float
    bisections: int
    probe_accepted: bool
    min_step_override: bool


class MoleculeLITWorkflow(Workflow):
    """Compute a molecular dipole response spectrum with NQS-LIT."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        self.lit_config = cfg.get("lit", MolecularLITConfig)
        self.system_config, self.wf = configure_system(cfg)
        self.sampler = cfg.get("sampler", MCMCSampler)
        self._validate_config()

    def run(self) -> None:
        self._run_serial_scan()

    def _run_serial_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = self._omega_grid()
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
        signed_lit = np.zeros_like(lit)
        broadened = np.zeros_like(lit)
        fidelity = np.zeros_like(lit)
        reverse_kl = np.zeros_like(lit)
        residual_norm = np.zeros_like(lit)
        equation_relative_residual = np.zeros_like(lit)
        action_norm = np.zeros_like(lit)
        source_norm = np.zeros_like(lit)
        error_bound_monitor = np.zeros_like(lit)
        error_d = np.zeros_like(lit)
        reweight_ess = np.zeros_like(lit)
        reweight_ess_fraction = np.zeros_like(lit)
        invalid_sample_fraction = np.zeros_like(lit)
        direct_hloc_rmse = np.full_like(lit, np.nan)
        direct_hloc_std = np.full_like(lit, np.nan)
        direct_hloc_sem = np.full_like(lit, np.nan)
        estimator_mode = np.zeros_like(lit, dtype=np.int64)
        selected_iteration = np.zeros_like(lit, dtype=np.int64)
        normalization = np.zeros((len(axes), len(omega)), dtype=np.complex128)
        correction_overlap = np.zeros_like(normalization)
        source_centers = np.zeros(len(axes), dtype=np.float64)
        axis_source_norm = np.zeros(len(axes), dtype=np.float64)
        warm_start_selected_iteration = np.zeros(len(axes), dtype=np.int64)
        continuation_axis: list[int] = []
        continuation_omega: list[float] = []
        continuation_optimized: list[bool] = []
        continuation_selected_iteration: list[int] = []
        continuation_fidelity: list[float] = []
        continuation_reverse_kl: list[float] = []
        continuation_invalid_sample_fraction: list[float] = []
        continuation_inherited_fidelity: list[float] = []
        continuation_step: list[float] = []
        continuation_bisections: list[int] = []
        continuation_probe_accepted: list[bool] = []
        continuation_min_step_override: list[bool] = []

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
            rng = self._precompile_direct_estimator(
                update_step,
                response_params,
                train_pool,
                axis_batched_data,
                rng,
                axis=axis,
                omega=jnp.asarray(precompile_omega),
            )
            (
                response_params,
                warm_start_stats,
                warm_start_selected_iteration[axis_pos],
                rng,
            ) = self._warm_start_axis(
                update_step,
                response_params,
                train_pool,
                eval_pool,
                axis_batched_data,
                rng,
                response_apply=response_apply,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
            )
            response_params, _, bridge_records, rng = self._continue_nqs_to_spectrum(
                update_step,
                response_params,
                warm_start_stats,
                train_pool,
                eval_pool,
                axis_batched_data,
                rng,
                response_apply=response_apply,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                target_omega=float(omega[0]),
                spectrum_omega=omega,
            )
            for record in bridge_records:
                host_bridge_stats = jax.device_get(record.stats)
                continuation_axis.append(axis)
                continuation_omega.append(record.omega)
                continuation_optimized.append(record.optimized)
                continuation_selected_iteration.append(record.selected_iteration)
                continuation_fidelity.append(float(host_bridge_stats.fidelity))
                continuation_reverse_kl.append(float(host_bridge_stats.reverse_kl))
                continuation_invalid_sample_fraction.append(
                    float(host_bridge_stats.invalid_sample_fraction)
                )
                continuation_inherited_fidelity.append(record.inherited_fidelity)
                continuation_step.append(record.step)
                continuation_bisections.append(record.bisections)
                continuation_probe_accepted.append(record.probe_accepted)
                continuation_min_step_override.append(record.min_step_override)
            logger.info(
                "axis=%s frequency_continuation=serial bridge_points=%d "
                "spectrum_points=%d",
                _AXIS_NAMES[axis],
                sum(record.optimized for record in bridge_records),
                len(omega),
            )
            for omega_pos, omega_value in enumerate(omega):
                (
                    response_params,
                    _,
                    selected_iteration[axis_pos, omega_pos],
                    rng,
                ) = self._optimize_nqs_frequency(
                    update_step,
                    response_params,
                    train_pool,
                    eval_pool,
                    axis_batched_data,
                    rng,
                    response_apply=response_apply,
                    ground_logpsi=ground_logpsi,
                    ground_params=ground_params,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=float(omega_value),
                    iterations=self.lit_config.nqs_iterations,
                    stage="spectrum",
                )
                stats, _, rng = self._nqs_eval_stats_with_fallback(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    axis_batched_data,
                    rng,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=jnp.asarray(float(omega_value)),
                )
                host_stats = jax.device_get(stats)
                signed_lit[axis_pos, omega_pos] = float(host_stats.signed_lit)
                lit[axis_pos, omega_pos] = float(host_stats.lit)
                broadened[axis_pos, omega_pos] = float(host_stats.broadened)
                fidelity[axis_pos, omega_pos] = float(host_stats.fidelity)
                reverse_kl[axis_pos, omega_pos] = float(host_stats.reverse_kl)
                residual_norm[axis_pos, omega_pos] = float(host_stats.residual_norm)
                equation_relative_residual[axis_pos, omega_pos] = float(
                    host_stats.equation_relative_residual
                )
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
                invalid_sample_fraction[axis_pos, omega_pos] = float(
                    host_stats.invalid_sample_fraction
                )
                direct_hloc_rmse[axis_pos, omega_pos] = float(
                    host_stats.direct_hloc_rmse
                )
                direct_hloc_std[axis_pos, omega_pos] = float(host_stats.direct_hloc_std)
                direct_hloc_sem[axis_pos, omega_pos] = float(host_stats.direct_hloc_sem)
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
            signed_lit=signed_lit,
            broadened=broadened,
            total_broadened=total_broadened,
            fidelity=fidelity,
            reverse_kl=reverse_kl,
            residual_norm=residual_norm,
            equation_relative_residual=equation_relative_residual,
            action_norm=action_norm,
            source_norm=source_norm,
            error_bound_monitor=error_bound_monitor,
            error_d=error_d,
            reweight_ess=reweight_ess,
            reweight_ess_fraction=reweight_ess_fraction,
            invalid_sample_fraction=invalid_sample_fraction,
            direct_hloc_rmse=direct_hloc_rmse,
            direct_hloc_std=direct_hloc_std,
            direct_hloc_sem=direct_hloc_sem,
            estimator_mode=estimator_mode,
            selected_iteration=selected_iteration,
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
            nqs_reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
            nqs_spring_epsilon=self.lit_config.nqs_spring_epsilon,
            nqs_spring_decay=self.lit_config.nqs_spring_decay,
            nqs_spring_damping_floor=self.lit_config.nqs_spring_damping_floor,
            nqs_selection_interval=self.lit_config.nqs_selection_interval,
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
            warm_start_selected_iteration=warm_start_selected_iteration,
            continuation_axis=np.asarray(continuation_axis, dtype=np.int64),
            continuation_omega=np.asarray(continuation_omega, dtype=np.float64),
            continuation_optimized=np.asarray(
                continuation_optimized,
                dtype=np.bool_,
            ),
            continuation_selected_iteration=np.asarray(
                continuation_selected_iteration,
                dtype=np.int64,
            ),
            continuation_fidelity=np.asarray(
                continuation_fidelity,
                dtype=np.float64,
            ),
            continuation_reverse_kl=np.asarray(
                continuation_reverse_kl,
                dtype=np.float64,
            ),
            continuation_invalid_sample_fraction=np.asarray(
                continuation_invalid_sample_fraction,
                dtype=np.float64,
            ),
            continuation_inherited_fidelity=np.asarray(
                continuation_inherited_fidelity,
                dtype=np.float64,
            ),
            continuation_step=np.asarray(continuation_step, dtype=np.float64),
            continuation_bisections=np.asarray(
                continuation_bisections,
                dtype=np.int64,
            ),
            continuation_probe_accepted=np.asarray(
                continuation_probe_accepted,
                dtype=np.bool_,
            ),
            continuation_min_step_override=np.asarray(
                continuation_min_step_override,
                dtype=np.bool_,
            ),
            nqs_continuation_iterations=self.lit_config.nqs_continuation_iterations,
            nqs_continuation_step_fraction=(
                self.lit_config.nqs_continuation_step_fraction
            ),
            nqs_continuation_fidelity_retention=(
                self.lit_config.nqs_continuation_fidelity_retention
            ),
            nqs_continuation_min_step=_optional_float(
                self.lit_config.nqs_continuation_min_step
            ),
            nqs_continuation_max_points=self.lit_config.nqs_continuation_max_points,
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
            peak_energies=np.asarray([peak.energy for peak in peaks]),
            peak_intensities=np.asarray([peak.intensity for peak in peaks]),
            peak_indices=np.asarray([peak.index for peak in peaks]),
        )
        self._log_nqs_summary(str(output_path), peaks, fidelity)

    def _omega_grid(self) -> np.ndarray:
        return _lit_omega_grid(self.lit_config)

    def _validate_config(self) -> None:
        omega = _lit_omega_grid(self.lit_config)
        if not np.isfinite(self.lit_config.eta) or self.lit_config.eta <= 0.0:
            msg = "lit.eta must be finite and positive."
            raise ValueError(msg)
        self._validate_serial_scan_config(omega)
        self._validate_chunk_config()
        if not 0.0 <= self.lit_config.nqs_reweight_ess_fraction_min <= 1.0:
            msg = (
                "lit.nqs_reweight_ess_fraction_min must be between 0 and 1, got "
                f"{self.lit_config.nqs_reweight_ess_fraction_min}."
            )
            raise ValueError(msg)
        self._validate_direct_psi_config()
        self._validate_nqs_stabilizer_config()
        self._validate_nqs_iteration_config()
        self._validate_continuation_config()

    def _validate_serial_scan_config(self, omega: np.ndarray) -> None:
        scan_parallel = str(self.lit_config.scan_parallel).lower()
        serial_modes = ("off", "false", "none", "0", "serial")
        if scan_parallel not in serial_modes:
            msg = (
                "Frequency-block parallel scans are incompatible with the "
                "published serial continuation chain; set "
                f"lit.scan_parallel=off, got {self.lit_config.scan_parallel!r}."
            )
            raise ValueError(msg)
        legacy_parallel_requested = (
            self.lit_config.scan_parallel_worker
            or self.lit_config.scan_parallel_worker_index != 0
            or self.lit_config.scan_parallel_workers != 0
            or self.lit_config.scan_parallel_procs_per_device != 1
            or self.lit_config.scan_parallel_min_points_per_worker != 2
            or bool(self.lit_config.scan_parallel_remote_hosts)
            or bool(self.lit_config.scan_parallel_remote_root)
            or bool(self.lit_config.scan_parallel_remote_python)
            or tuple(self.lit_config.scan_parallel_ssh_options)
            != (
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
            )
        )
        if legacy_parallel_requested:
            msg = (
                "Legacy lit.scan_parallel_* worker settings cannot be used "
                "with serial frequency continuation."
            )
            raise ValueError(msg)
        warm_omega = self.lit_config.nqs_warm_start_omega
        if warm_omega is not None:
            if not np.isfinite(warm_omega):
                msg = "lit.nqs_warm_start_omega must be finite or null."
                raise ValueError(msg)
            if warm_omega >= float(omega[0]):
                msg = (
                    "lit.nqs_warm_start_omega must be below the first spectrum "
                    "frequency for increasing serial continuation."
                )
                raise ValueError(msg)

    def _validate_nqs_stabilizer_config(self) -> None:
        if (
            not np.isfinite(self.lit_config.nqs_reverse_kl_weight)
            or self.lit_config.nqs_reverse_kl_weight < 0.0
        ):
            msg = "lit.nqs_reverse_kl_weight must be nonnegative."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.nqs_spring_epsilon)
            or self.lit_config.nqs_spring_epsilon <= 0.0
        ):
            msg = "lit.nqs_spring_epsilon must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.lit_config.nqs_spring_decay < 1.0:
            msg = "lit.nqs_spring_decay must satisfy 0 <= value < 1."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.nqs_spring_damping_floor)
            or self.lit_config.nqs_spring_damping_floor <= 0.0
        ):
            msg = "lit.nqs_spring_damping_floor must be positive."
            raise ValueError(msg)
        if (
            not np.isfinite(self.lit_config.nqs_learning_rate)
            or self.lit_config.nqs_learning_rate <= 0.0
        ):
            msg = "lit.nqs_learning_rate must be finite and positive."
            raise ValueError(msg)
        max_norm = self.lit_config.nqs_sr_max_norm
        if max_norm is not None and (
            not np.isfinite(max_norm) or float(max_norm) <= 0.0
        ):
            msg = "lit.nqs_sr_max_norm must be positive or null."
            raise ValueError(msg)
        if self.lit_config.nqs_sr_damping is not None:
            msg = (
                "lit.nqs_sr_damping is an obsolete absolute damping; use "
                "lit.nqs_spring_epsilon for scale-invariant SPRING damping."
            )
            raise ValueError(msg)

    def _validate_nqs_iteration_config(self) -> None:
        if self.lit_config.nqs_selection_interval < 1:
            msg = "lit.nqs_selection_interval must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_warm_start_iterations < 0:
            msg = "lit.nqs_warm_start_iterations must be nonnegative."
            raise ValueError(msg)
        if self.lit_config.nqs_iterations < 1:
            msg = "lit.nqs_iterations must be positive."
            raise ValueError(msg)

    def _validate_continuation_config(self) -> None:
        if self.lit_config.nqs_continuation_iterations < 1:
            msg = "lit.nqs_continuation_iterations must be positive."
            raise ValueError(msg)
        step_fraction = self.lit_config.nqs_continuation_step_fraction
        if not np.isfinite(step_fraction) or step_fraction <= 0.0:
            msg = "lit.nqs_continuation_step_fraction must be positive."
            raise ValueError(msg)
        retention = self.lit_config.nqs_continuation_fidelity_retention
        if not 0.0 < retention <= 1.0:
            msg = "lit.nqs_continuation_fidelity_retention must satisfy 0 < value <= 1."
            raise ValueError(msg)
        min_step = self.lit_config.nqs_continuation_min_step
        if min_step is not None and (
            not np.isfinite(min_step) or float(min_step) <= 0.0
        ):
            msg = "lit.nqs_continuation_min_step must be positive or null."
            raise ValueError(msg)
        if self.lit_config.nqs_continuation_max_points < 1:
            msg = "lit.nqs_continuation_max_points must be positive."
            raise ValueError(msg)

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

    def _precompile_direct_estimator(
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

    def _evaluate_nqs_checkpoint(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        eval_pool,
        *,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega: float,
    ):
        stats = self._nqs_stats_chunked(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            eval_pool,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
            omega=jnp.asarray(float(omega)),
        )
        return stats._replace(
            loss=_regularized_loss(
                stats,
                self.lit_config.nqs_reverse_kl_weight,
            )
        )

    def _optimize_nqs_frequency(
        self,
        update_step,
        initial_params,
        train_pool,
        eval_pool,
        fallback_data,
        rng,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        omega: float,
        iterations: int,
        stage: str,
    ):
        """Optimize one frequency and select parameters on a fixed eval pool.

        Returns:
            Best parameters, their held-out statistics, selected iteration,
            and the next random key.
        """

        def evaluate(params):
            return self._evaluate_nqs_checkpoint(
                response_apply,
                params,
                ground_logpsi,
                ground_params,
                eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=float(omega),
            )

        response_params = initial_params
        update_carry = update_step.init_carry(fallback_data, rng, response_params)
        best_params = response_params
        best_stats = evaluate(response_params)
        best_iteration = 0
        last_train_stats = None
        for iteration in range(max(0, int(iterations))):
            response_params, last_train_stats, update_carry = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(omega)),
                update_carry,
                iteration,
            )
            completed = iteration + 1
            should_select = (
                completed % self.lit_config.nqs_selection_interval == 0
                or completed == int(iterations)
            )
            if should_select:
                candidate_stats = evaluate(response_params)
                if _is_better_nqs_checkpoint(candidate_stats, best_stats):
                    best_params = response_params
                    best_stats = candidate_stats
                    best_iteration = completed
            if (
                self.lit_config.nqs_log_interval > 0
                and completed % self.lit_config.nqs_log_interval == 0
            ):
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d train_loss=%.6e "
                    "train_fidelity=%.6f train_reverse_kl=%.6e "
                    "best_iter=%d best_fidelity=%.6f best_reverse_kl=%.6e",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(last_train_stats.loss),
                    float(last_train_stats.fidelity),
                    float(last_train_stats.reverse_kl),
                    best_iteration,
                    float(best_stats.fidelity),
                    float(best_stats.reverse_kl),
                )
        rng = update_carry.direct.rng
        logger.info(
            "axis=%s stage=%s omega=%.6f selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e ess=%.3f",
            _AXIS_NAMES[axis],
            stage,
            float(omega),
            best_iteration,
            max(0, int(iterations)),
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(best_stats.reweight_ess_fraction),
        )
        return best_params, best_stats, best_iteration, rng

    def _continue_nqs_to_spectrum(
        self,
        update_step,
        response_params,
        current_stats,
        train_pool,
        eval_pool,
        fallback_data,
        rng,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        target_omega: float,
        spectrum_omega: np.ndarray,
    ):
        """Adaptively bridge the warm start to the first reported frequency.

        If ``psi_omega`` solves ``A(omega) psi = Phi``, reusing it at
        ``omega + delta`` gives a relative residual of order
        ``|delta| sqrt(L(omega) / ||Phi||^2)``.  We bound that quantity by the
        configured step fraction and verify every proposed step on the fixed
        held-out pool, bisecting when fidelity degrades too much.

        Returns:
            Parameters prepared for the target, the latest optimized held-out
            statistics, continuation records (including the target probe), and
            the next random key.

        Raises:
            RuntimeError: If a probe is invalid, an optimized bridge checkpoint
                is invalid, or the configured bridge-point cap is exhausted.
        """
        start_omega = self.lit_config.nqs_warm_start_omega
        if start_omega is None or float(start_omega) >= float(target_omega):
            return response_params, current_stats, [], rng

        common = dict(
            response_apply=response_apply,
            ground_logpsi=ground_logpsi,
            ground_params=ground_params,
            axis=axis,
            source_center=source_center,
            source_norm=source_norm,
            ground_energy=ground_energy,
        )
        current_omega = float(start_omega)
        if current_stats is None:
            current_stats = self._evaluate_nqs_checkpoint(
                response_params=response_params,
                eval_pool=eval_pool,
                omega=current_omega,
                **common,
            )
        min_step = _continuation_min_step(self.lit_config, spectrum_omega)
        records = []
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(target_omega)) * 8.0

        while target_omega - current_omega > tolerance:
            gap = float(target_omega - current_omega)
            step = _physics_continuation_step(
                current_stats,
                gap=gap,
                fraction=self.lit_config.nqs_continuation_step_fraction,
                min_step=min_step,
            )
            candidate_omega = current_omega + step
            bisections = 0
            while True:
                probe_stats = self._evaluate_nqs_checkpoint(
                    response_params=response_params,
                    eval_pool=eval_pool,
                    omega=candidate_omega,
                    **common,
                )
                probe_ok = _continuation_probe_is_acceptable(
                    current_stats,
                    probe_stats,
                    retention=self.lit_config.nqs_continuation_fidelity_retention,
                )
                candidate_gap = candidate_omega - current_omega
                if probe_ok or candidate_gap <= min_step * (1.0 + 1e-12):
                    break
                candidate_gap = max(min_step, 0.5 * candidate_gap)
                candidate_omega = min(target_omega, current_omega + candidate_gap)
                bisections += 1

            if not _finite_valid_nqs_stats(probe_stats):
                msg = (
                    "Frequency continuation produced non-finite/invalid held-out "
                    f"statistics at omega={candidate_omega:.8g}; refusing to "
                    "propagate a corrupted checkpoint."
                )
                raise RuntimeError(msg)
            actual_step = float(candidate_omega - current_omega)
            min_step_override = not probe_ok and actual_step <= min_step * (1.0 + 1e-12)
            if target_omega - candidate_omega <= tolerance:
                records.append(
                    _ContinuationRecord(
                        omega=float(candidate_omega),
                        optimized=False,
                        selected_iteration=-1,
                        stats=probe_stats,
                        inherited_fidelity=float(probe_stats.fidelity),
                        step=actual_step,
                        bisections=bisections,
                        probe_accepted=probe_ok,
                        min_step_override=min_step_override,
                    )
                )
                logger.info(
                    "axis=%s continuation_probe target=%.6f inherited_fidelity=%.6f "
                    "step=%.6e bisections=%d accepted=%s min_step_override=%s",
                    _AXIS_NAMES[axis],
                    target_omega,
                    float(probe_stats.fidelity),
                    actual_step,
                    bisections,
                    probe_ok,
                    min_step_override,
                )
                break

            optimized_count = sum(record.optimized for record in records)
            if optimized_count >= self.lit_config.nqs_continuation_max_points:
                msg = (
                    "Adaptive frequency continuation exceeded "
                    f"{self.lit_config.nqs_continuation_max_points} bridge points "
                    f"before omega={target_omega:.8g}."
                )
                raise RuntimeError(msg)
            inherited_fidelity = float(probe_stats.fidelity)
            response_params, current_stats, selected_iteration, rng = (
                self._optimize_nqs_frequency(
                    update_step,
                    response_params,
                    train_pool,
                    eval_pool,
                    fallback_data,
                    rng,
                    omega=candidate_omega,
                    iterations=self.lit_config.nqs_continuation_iterations,
                    stage="continuation",
                    **common,
                )
            )
            if not _finite_valid_nqs_stats(current_stats):
                msg = (
                    "Frequency continuation failed to obtain a finite held-out "
                    f"checkpoint at omega={candidate_omega:.8g}."
                )
                raise RuntimeError(msg)
            records.append(
                _ContinuationRecord(
                    omega=float(candidate_omega),
                    optimized=True,
                    selected_iteration=selected_iteration,
                    stats=current_stats,
                    inherited_fidelity=inherited_fidelity,
                    step=actual_step,
                    bisections=bisections,
                    probe_accepted=probe_ok,
                    min_step_override=min_step_override,
                )
            )
            logger.info(
                "axis=%s continuation_step omega=%.6f inherited_fidelity=%.6f "
                "selected_fidelity=%.6f step=%.6e bisections=%d accepted=%s "
                "min_step_override=%s",
                _AXIS_NAMES[axis],
                candidate_omega,
                inherited_fidelity,
                float(current_stats.fidelity),
                actual_step,
                bisections,
                probe_ok,
                min_step_override,
            )
            current_omega = candidate_omega

        return response_params, current_stats, records, rng

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        eval_pool,
        fallback_data,
        rng,
        **kwargs,
    ):
        if (
            self.lit_config.nqs_warm_start_omega is None
            or self.lit_config.nqs_warm_start_iterations <= 0
        ):
            return response_params, None, 0, rng
        result = self._optimize_nqs_frequency(
            update_step,
            response_params,
            train_pool,
            eval_pool,
            fallback_data,
            rng,
            omega=float(self.lit_config.nqs_warm_start_omega),
            iterations=self.lit_config.nqs_warm_start_iterations,
            stage="warm_start",
            **kwargs,
        )
        if result[1] is not None:
            logger.info(
                "axis=%s warm_start omega=%.6f iterations=%d "
                "selected_iter=%d fidelity=%.6f reverse_kl=%.6e",
                _AXIS_NAMES[kwargs["axis"]],
                float(self.lit_config.nqs_warm_start_omega),
                self.lit_config.nqs_warm_start_iterations,
                result[2],
                float(result[1].fidelity),
                float(result[1].reverse_kl),
            )
        return result

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
        def source_update(response_params, batched_data, spring_previous, omega):
            stats, updates, spring_state, damping = self._source_sr_stats_and_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                spring_state=_SpringState(spring_previous),
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
            )
            response_params = _apply_updates(response_params, updates)
            loss = _regularized_loss(
                stats,
                self.lit_config.nqs_reverse_kl_weight,
            )
            return (
                response_params,
                stats._replace(loss=loss),
                spring_state.previous_direction,
                damping,
            )

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
            direct_active,
            spring_previous,
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
            source_stats = nqs_lit_stats_from_source_sums(
                source_sums,
                source_norm=source_norm,
                omega=omega,
                eta=self.lit_config.eta,
            )
            source_loss = _regularized_loss(
                source_stats,
                self.lit_config.nqs_reverse_kl_weight,
            )
            source_stats = source_stats._replace(loss=source_loss)
            threshold = jnp.asarray(
                self.lit_config.nqs_reweight_ess_fraction_min,
                dtype=source_stats.reweight_ess_fraction.dtype,
            )
            use_direct = (
                direct_active
                | ~jnp.isfinite(source_stats.reweight_ess_fraction)
                | ~jnp.isfinite(source_stats.loss)
                | (source_stats.invalid_sample_fraction > 0.0)
                | (source_stats.reweight_ess_fraction < threshold)
            )

            def source_branch(_):
                source_updates, next_spring_state, spring_damping = (
                    self._weighted_sr_updates(
                        response_apply,
                        response_params,
                        ground_logpsi,
                        ground_params,
                        source_batched_data,
                        spring_state=_SpringState(spring_previous),
                        axis=axis,
                        source_center=source_center,
                        ground_energy=ground_energy,
                        omega=omega,
                    )
                )
                source_response_params = _apply_updates(response_params, source_updates)
                return (
                    source_response_params,
                    source_stats,
                    direct_batched_data,
                    direct_sampler_state,
                    direct_rng,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    next_spring_state.previous_direction,
                    spring_damping,
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
                direct_stats, direct_updates, next_spring_state, spring_damping = (
                    self._direct_sr_stats_and_updates_from_source_sums(
                        response_apply,
                        response_params,
                        ground_logpsi,
                        ground_params,
                        source_sums,
                        psi_pool,
                        spring_state=_SpringState(spring_previous),
                        axis=axis,
                        source_center=source_center,
                        source_norm=source_norm,
                        ground_energy=ground_energy,
                        omega=omega,
                    )
                )
                direct_loss = _regularized_loss(
                    direct_stats,
                    self.lit_config.nqs_reverse_kl_weight,
                )
                direct_response_params = _apply_updates(
                    response_params,
                    direct_updates,
                )
                return (
                    direct_response_params,
                    direct_stats._replace(loss=direct_loss),
                    next_batched_data,
                    next_sampler_state,
                    next_rng,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    next_spring_state.previous_direction,
                    spring_damping,
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
            update_carry,
            batch_index: int = 0,
        ):
            if int(batch_index) == 0:
                update_carry = update_carry._replace(
                    direct=update_carry.direct._replace(
                        initialized=jnp.asarray(False),
                        use_direct=jnp.asarray(False),
                    )
                )
            update_batch = _cyclic_batched_data_chunk(
                batched_data,
                self._nqs_train_update_batch_size(),
                batch_index,
            )
            if not direct_enabled:
                response_params, stats, spring_previous, _ = source_update(
                    response_params,
                    update_batch,
                    update_carry.spring.previous_direction,
                    omega,
                )
                return (
                    response_params,
                    stats,
                    update_carry._replace(spring=_SpringState(spring_previous)),
                )
            (
                response_params,
                stats,
                direct_batched_data,
                direct_sampler_state,
                direct_rng,
                direct_initialized,
                direct_active,
                spring_previous,
                _,
            ) = fallback_update(
                response_params,
                update_batch,
                update_carry.direct.batched_data,
                update_carry.direct.sampler_state,
                update_carry.direct.rng,
                update_carry.direct.initialized,
                update_carry.direct.use_direct,
                update_carry.spring.previous_direction,
                omega,
            )
            return (
                response_params,
                stats,
                _NQSUpdateCarry(
                    direct=_DirectPsiCarry(
                        batched_data=direct_batched_data,
                        sampler_state=direct_sampler_state,
                        rng=direct_rng,
                        initialized=direct_initialized,
                        use_direct=direct_active,
                    ),
                    spring=_SpringState(spring_previous),
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

        update.init_carry = self._init_nqs_update_carry  # type: ignore[attr-defined]
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
        update_carry = self._init_nqs_update_carry(
            fallback_data,
            rng,
            response_params,
        )
        fallback_update.lower(
            response_params,
            update_batch,
            update_carry.direct.batched_data,
            update_carry.direct.sampler_state,
            update_carry.direct.rng,
            update_carry.direct.initialized,
            update_carry.direct.use_direct,
            update_carry.spring.previous_direction,
            omega,
        ).compile()
        logger.info(
            "Precompiled fused direct pi_Psi fallback step for axis=%s",
            _AXIS_NAMES[axis],
        )
        return rng

    def _source_sr_stats_and_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        spring_state: _SpringState,
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
        updates, spring_state, damping = self._weighted_sr_updates_from_scores(
            response_params,
            score,
            ratio,
            source_weight,
            spring_state,
        )
        return stats, updates, spring_state, damping

    def _weighted_sr_updates_from_scores(
        self,
        response_params,
        score,
        ratio,
        source_weight,
        spring_state: _SpringState,
    ):
        (
            grad_flat,
            _,
            _,
            psi_weight,
            centered_score,
            _,
            _,
        ) = _regularized_action_gradient(
            score,
            ratio,
            source_weight,
            reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
            eps=self.lit_config.nqs_sr_score_eps,
        )
        weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
        score_aug = jnp.concatenate([weighted_score.real, weighted_score.imag], axis=0)
        centering_null = jnp.sqrt(psi_weight)
        zero_null = jnp.zeros_like(centering_null)
        kernel_null_vectors = jnp.stack(
            [
                jnp.concatenate([centering_null, zero_null]),
                jnp.concatenate([zero_null, centering_null]),
            ]
        )
        direction, spring_state, damping = _spring_direction_chunked(
            (score_aug.shape[0],),
            lambda _: score_aug,
            grad_flat,
            spring_state,
            epsilon_scale=self.lit_config.nqs_spring_epsilon,
            damping_floor=self.lit_config.nqs_spring_damping_floor,
            decay=self.lit_config.nqs_spring_decay,
            kernel_null_vectors=kernel_null_vectors,
        )
        updates = _scaled_direction_updates(
            response_params,
            direction,
            learning_rate=self.lit_config.nqs_learning_rate,
            max_norm=self.lit_config.nqs_sr_max_norm,
        )
        return updates, spring_state, damping

    def _weighted_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        spring_state: _SpringState,
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
            spring_state,
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
        spring_state: _SpringState,
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
        normalization = source_stats.normalization
        chunks = tuple(
            _batched_data_chunks(psi_batched_data, self._nqs_eval_batch_size())
        )
        if not chunks:
            msg = "Cannot compute direct SR update with an empty psi pool."
            raise ValueError(msg)

        def score_hloc_and_log_ratio(chunk):
            score, ratio, _ = self._source_sampled_action_scores(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                chunk,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
            eps = jnp.asarray(
                self.lit_config.nqs_sr_score_eps,
                dtype=ratio.real.dtype,
            )
            safe_ratio = jnp.where(
                jnp.abs(ratio) > eps,
                ratio,
                jnp.asarray(eps, dtype=ratio.real.dtype) + 0j,
            )
            hloc = normalization / safe_ratio
            finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
            finite_score = jnp.all(
                jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
                axis=1,
            )
            valid = (
                (jnp.abs(ratio) > eps)
                & finite_ratio
                & jnp.isfinite(jnp.real(hloc))
                & jnp.isfinite(jnp.imag(hloc))
                & finite_score
            )
            hloc = jnp.where(valid, hloc, jnp.asarray(0.0, dtype=hloc.dtype))
            score = jnp.where(
                valid[:, None],
                score,
                jnp.asarray(0.0, dtype=score.dtype),
            )
            log_ratio_abs2 = 2.0 * jnp.log(
                jnp.maximum(
                    jnp.where(finite_ratio, jnp.abs(ratio), 0.0),
                    eps,
                )
            )
            log_ratio_abs2 = jnp.where(valid, log_ratio_abs2, 0.0)
            return score, hloc, log_ratio_abs2, valid

        score_sum = None
        score_hloc_conj_sum = None
        hloc_sum = None
        hloc_conj_sum = None
        hloc_abs2_sum = None
        hloc_error_abs2_sum = None
        score_log_ratio_sum = None
        log_ratio_sum = None
        score_abs2_sum = None
        valid_count = None
        for chunk in chunks:
            score, hloc, log_ratio_abs2, valid = score_hloc_and_log_ratio(chunk)
            local_score_sum = jnp.sum(score, axis=0)
            local_score_hloc_conj_sum = jnp.sum(
                score * jnp.conj(hloc)[:, None],
                axis=0,
            )
            local_hloc_sum = jnp.sum(hloc)
            local_hloc_conj_sum = jnp.sum(jnp.conj(hloc))
            local_hloc_abs2_sum = jnp.sum(jnp.abs(hloc) ** 2)
            local_hloc_error_abs2_sum = jnp.sum(
                jnp.where(valid, jnp.abs(hloc - 1.0) ** 2, 0.0)
            )
            local_score_log_ratio_sum = jnp.sum(
                score * log_ratio_abs2[:, None],
                axis=0,
            )
            local_log_ratio_sum = jnp.sum(log_ratio_abs2)
            local_score_abs2_sum = jnp.sum(jnp.abs(score) ** 2)
            local_valid_count = jnp.sum(valid)
            score_sum = (
                local_score_sum if score_sum is None else score_sum + local_score_sum
            )
            score_hloc_conj_sum = (
                local_score_hloc_conj_sum
                if score_hloc_conj_sum is None
                else score_hloc_conj_sum + local_score_hloc_conj_sum
            )
            hloc_sum = local_hloc_sum if hloc_sum is None else hloc_sum + local_hloc_sum
            hloc_conj_sum = (
                local_hloc_conj_sum
                if hloc_conj_sum is None
                else hloc_conj_sum + local_hloc_conj_sum
            )
            hloc_abs2_sum = (
                local_hloc_abs2_sum
                if hloc_abs2_sum is None
                else hloc_abs2_sum + local_hloc_abs2_sum
            )
            hloc_error_abs2_sum = (
                local_hloc_error_abs2_sum
                if hloc_error_abs2_sum is None
                else hloc_error_abs2_sum + local_hloc_error_abs2_sum
            )
            score_log_ratio_sum = (
                local_score_log_ratio_sum
                if score_log_ratio_sum is None
                else score_log_ratio_sum + local_score_log_ratio_sum
            )
            log_ratio_sum = (
                local_log_ratio_sum
                if log_ratio_sum is None
                else log_ratio_sum + local_log_ratio_sum
            )
            score_abs2_sum = (
                local_score_abs2_sum
                if score_abs2_sum is None
                else score_abs2_sum + local_score_abs2_sum
            )
            valid_count = (
                local_valid_count
                if valid_count is None
                else valid_count + local_valid_count
            )
        if (
            score_sum is None
            or score_hloc_conj_sum is None
            or hloc_sum is None
            or hloc_conj_sum is None
            or hloc_abs2_sum is None
            or hloc_error_abs2_sum is None
            or score_log_ratio_sum is None
            or log_ratio_sum is None
            or score_abs2_sum is None
            or valid_count is None
        ):
            msg = "Cannot compute direct SR update with an empty psi pool."
            raise ValueError(msg)

        sample_count_int = sum(chunk.batch_size for chunk in chunks)
        sample_count = jnp.asarray(sample_count_int, dtype=score_sum.real.dtype)
        safe_sample_count = jnp.maximum(
            valid_count,
            jnp.asarray(1, dtype=sample_count.dtype),
        )
        score_mean = score_sum / safe_sample_count
        hloc_mean = hloc_sum / safe_sample_count
        hloc_conj_mean = hloc_conj_sum / safe_sample_count
        hloc_abs2_mean = hloc_abs2_sum / safe_sample_count
        hloc_error_abs2_mean = hloc_error_abs2_sum / safe_sample_count
        direct_hloc_rmse = jnp.sqrt(jnp.maximum(jnp.real(hloc_error_abs2_mean), 0.0))
        direct_hloc_std = jnp.sqrt(
            jnp.maximum(jnp.real(hloc_abs2_mean - jnp.abs(hloc_mean) ** 2), 0.0)
        )
        direct_hloc_sem = direct_hloc_std / jnp.sqrt(safe_sample_count)
        fidelity = jnp.clip(jnp.real(hloc_mean), 0.0, 1.0)
        eps = jnp.asarray(self.lit_config.nqs_sr_score_eps, dtype=fidelity.dtype)
        action_norm = (
            jnp.asarray(source_norm, dtype=fidelity.dtype)
            * jnp.abs(normalization) ** 2
            / jnp.maximum(fidelity, eps)
        )

        score_hloc_conj_mean = score_hloc_conj_sum / safe_sample_count
        fidelity_gradient = 2.0 * jnp.real(
            score_hloc_conj_mean - score_mean * hloc_conj_mean
        )
        log_ratio_mean = log_ratio_sum / safe_sample_count
        reverse_kl_gradient = 2.0 * jnp.real(
            score_log_ratio_sum / safe_sample_count - score_mean * log_ratio_mean
        )
        grad_flat = (
            fidelity_gradient
            - jnp.asarray(
                self.lit_config.nqs_reverse_kl_weight,
                dtype=fidelity_gradient.dtype,
            )
            * reverse_kl_gradient
        )
        score_scale = jnp.sqrt(safe_sample_count)

        def score_aug_chunk(index: int):
            score, _, _, valid = score_hloc_and_log_ratio(chunks[index])
            centered_score = jnp.where(
                valid[:, None],
                score - score_mean,
                jnp.asarray(0.0, dtype=score.dtype),
            )
            weighted_score = centered_score / score_scale
            return jnp.concatenate(
                [weighted_score.real, weighted_score.imag],
                axis=0,
            )

        qfi_trace = jnp.maximum(
            score_abs2_sum / safe_sample_count - jnp.sum(jnp.abs(score_mean) ** 2),
            jnp.asarray(0.0, dtype=grad_flat.dtype),
        )
        real_null_blocks = []
        imag_null_blocks = []
        for chunk in chunks:
            _, _, _, valid = score_hloc_and_log_ratio(chunk)
            valid_float = valid.astype(grad_flat.dtype)
            zero_block = jnp.zeros_like(valid_float)
            real_null_blocks.append(jnp.concatenate([valid_float, zero_block]))
            imag_null_blocks.append(jnp.concatenate([zero_block, valid_float]))
        kernel_null_vectors = jnp.stack(
            [jnp.concatenate(real_null_blocks), jnp.concatenate(imag_null_blocks)]
        )
        direction, spring_state, damping = _spring_direction_chunked(
            tuple(2 * chunk.batch_size for chunk in chunks),
            score_aug_chunk,
            grad_flat,
            spring_state,
            epsilon_scale=self.lit_config.nqs_spring_epsilon,
            damping_floor=self.lit_config.nqs_spring_damping_floor,
            decay=self.lit_config.nqs_spring_decay,
            qfi_trace=qfi_trace,
            kernel_null_vectors=kernel_null_vectors,
        )
        updates = _scaled_direction_updates(
            response_params,
            direction,
            learning_rate=self.lit_config.nqs_learning_rate,
            max_norm=self.lit_config.nqs_sr_max_norm,
        )
        reverse_kl = jnp.where(
            valid_count > 0,
            jnp.maximum(
                log_ratio_mean - source_stats.log_ratio_norm,
                jnp.asarray(0.0, dtype=fidelity.dtype),
            ),
            jnp.asarray(0.0, dtype=fidelity.dtype),
        )
        equation_relative_residual = jnp.sqrt(
            jnp.maximum(1.0 / jnp.maximum(fidelity, eps) - 1.0, 0.0)
        )
        stats = source_stats._replace(
            loss=1.0 - fidelity + self.lit_config.nqs_reverse_kl_weight * reverse_kl,
            fidelity=fidelity,
            reverse_kl=reverse_kl,
            residual_norm=jnp.asarray(source_norm, dtype=fidelity.dtype)
            * equation_relative_residual**2,
            equation_relative_residual=equation_relative_residual,
            action_norm=jnp.real(action_norm),
            invalid_sample_fraction=1.0 - valid_count / jnp.maximum(sample_count, 1.0),
            estimator_mode=jnp.asarray(1, dtype=jnp.int32),
            direct_hloc_rmse=jnp.real(direct_hloc_rmse),
            direct_hloc_std=jnp.real(direct_hloc_std),
            direct_hloc_sem=jnp.real(direct_hloc_sem),
        )
        return stats, updates, spring_state, damping

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
        base_finite_sums = (
            jnp.isfinite(jnp.real(action))
            & jnp.isfinite(jnp.imag(action))
            & jnp.isfinite(jnp.real(response_ratio))
            & jnp.isfinite(jnp.imag(response_ratio))
            & jnp.isfinite(source)
            & jnp.isfinite(stats_source_weight)
        )
        safe_source_stats = jnp.where(
            jnp.abs(source) > stats_eps,
            source,
            stats_eps * jnp.where(source < 0, -1.0, 1.0),
        )
        raw_stats_ratio = action / safe_source_stats
        finite_stats_ratio = jnp.isfinite(jnp.real(raw_stats_ratio)) & jnp.isfinite(
            jnp.imag(raw_stats_ratio)
        )
        finite_sums = base_finite_sums & finite_stats_ratio
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
        stats_ratio = jnp.where(
            finite_sums,
            raw_stats_ratio,
            jnp.asarray(0.0, dtype=raw_stats_ratio.dtype),
        )
        stats_ratio_abs = jnp.where(
            stats_source_weight > 0.0,
            jnp.abs(stats_ratio),
            0.0,
        )
        max_stats_ratio_abs = jnp.max(stats_ratio_abs)
        stats_ratio_scale = jnp.where(
            max_stats_ratio_abs > 0.0,
            max_stats_ratio_abs,
            jnp.asarray(1.0, dtype=stats_ratio_abs.dtype),
        )
        scaled_stats_ratio = stats_ratio / jax.lax.stop_gradient(stats_ratio_scale)
        scaled_stats_ratio_abs2 = jnp.abs(scaled_stats_ratio) ** 2
        psi_weight_unnormalized = stats_source_weight * scaled_stats_ratio_abs2
        log_ratio_abs2 = 2.0 * jnp.log(
            jnp.maximum(jnp.abs(scaled_stats_ratio), stats_eps)
        )
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
            valid_sample_count=jnp.sum(finite_sums),
            ratio_scale=stats_ratio_scale,
            ratio_sum=jnp.sum(stats_source_weight * scaled_stats_ratio),
            ratio_abs2_sum=jnp.sum(stats_source_weight * scaled_stats_ratio_abs2),
            psi_weight_sum=jnp.sum(psi_weight_unnormalized),
            psi_weight_sq_sum=jnp.sum(psi_weight_unnormalized**2),
            psi_log_ratio_abs2_sum=jnp.sum(psi_weight_unnormalized * log_ratio_abs2),
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
        source_stats = self._nqs_stats_chunked(
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

        @jax.jit
        def direct_hloc_sum(local_params, chunk, normalization, local_omega):
            data = chunk.data
            action = jax.vmap(
                lambda one: local_action_ratio(
                    response_apply,
                    local_params,
                    ground_logpsi,
                    ground_params,
                    one,
                    ground_energy=ground_energy,
                    omega=local_omega,
                    eta=self.lit_config.eta,
                )[0],
                in_axes=(chunk.vmap_axis,),
            )(data)
            dipole = jax.vmap(
                lambda one: molecular_electronic_dipole(one, axis),
                in_axes=(chunk.vmap_axis,),
            )(data)
            eps = jnp.asarray(self.lit_config.nqs_sr_score_eps, dtype=dipole.dtype)
            source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
            safe_source = jnp.where(
                jnp.abs(source) > eps,
                source,
                eps * jnp.where(source < 0, -1.0, 1.0),
            )
            ratio = action / safe_source
            safe_ratio = jnp.where(
                jnp.abs(ratio) > eps,
                ratio,
                jnp.asarray(eps, dtype=ratio.real.dtype) + 0j,
            )
            hloc = normalization / safe_ratio
            finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
            finite = (
                (jnp.abs(ratio) > eps)
                & finite_ratio
                & jnp.isfinite(jnp.real(hloc))
                & jnp.isfinite(jnp.imag(hloc))
            )
            hloc = jnp.where(finite, hloc, jnp.asarray(0.0, dtype=hloc.dtype))
            log_ratio_abs2 = 2.0 * jnp.log(
                jnp.maximum(
                    jnp.where(finite_ratio, jnp.abs(ratio), 0.0),
                    eps,
                )
            )
            sample_count = jnp.asarray(action.shape[0], dtype=hloc.real.dtype)
            return (
                jnp.sum(hloc),
                jnp.sum(jnp.abs(hloc) ** 2),
                jnp.sum(jnp.where(finite, jnp.abs(hloc - 1.0) ** 2, 0.0)),
                jnp.sum(jnp.where(finite, log_ratio_abs2, 0.0)),
                jnp.sum(finite),
                sample_count,
            )

        total_hloc = None
        total_hloc_abs2 = None
        total_hloc_error_abs2 = None
        total_log_ratio_abs2 = None
        total_valid_count = None
        total_count = None
        for chunk in _batched_data_chunks(
            psi_batched_data, self._nqs_eval_batch_size()
        ):
            (
                hloc_sum,
                hloc_abs2_sum,
                hloc_error_abs2_sum,
                log_ratio_abs2_sum,
                valid_count,
                sample_count,
            ) = direct_hloc_sum(
                response_params,
                chunk,
                source_stats.normalization,
                omega,
            )
            total_hloc = hloc_sum if total_hloc is None else total_hloc + hloc_sum
            total_hloc_abs2 = (
                hloc_abs2_sum
                if total_hloc_abs2 is None
                else total_hloc_abs2 + hloc_abs2_sum
            )
            total_hloc_error_abs2 = (
                hloc_error_abs2_sum
                if total_hloc_error_abs2 is None
                else total_hloc_error_abs2 + hloc_error_abs2_sum
            )
            total_log_ratio_abs2 = (
                log_ratio_abs2_sum
                if total_log_ratio_abs2 is None
                else total_log_ratio_abs2 + log_ratio_abs2_sum
            )
            total_valid_count = (
                valid_count
                if total_valid_count is None
                else total_valid_count + valid_count
            )
            total_count = (
                sample_count if total_count is None else total_count + sample_count
            )
        if (
            total_hloc is None
            or total_hloc_abs2 is None
            or total_hloc_error_abs2 is None
            or total_log_ratio_abs2 is None
            or total_valid_count is None
            or total_count is None
        ):
            msg = "Cannot evaluate direct NQS-LIT stats with an empty psi pool."
            raise ValueError(msg)

        eps = jnp.asarray(self.lit_config.nqs_sr_score_eps, dtype=total_count.dtype)
        safe_count = jnp.maximum(total_valid_count, eps)
        hloc_mean = total_hloc / safe_count
        hloc_abs2_mean = total_hloc_abs2 / safe_count
        hloc_error_abs2_mean = total_hloc_error_abs2 / safe_count
        direct_hloc_rmse = jnp.sqrt(jnp.maximum(jnp.real(hloc_error_abs2_mean), 0.0))
        direct_hloc_std = jnp.sqrt(
            jnp.maximum(jnp.real(hloc_abs2_mean - jnp.abs(hloc_mean) ** 2), 0.0)
        )
        direct_hloc_sem = direct_hloc_std / jnp.sqrt(safe_count)
        fidelity = jnp.clip(jnp.real(hloc_mean), 0.0, 1.0)
        direct_log_ratio_mean = total_log_ratio_abs2 / safe_count
        reverse_kl = jnp.where(
            total_valid_count > 0,
            jnp.maximum(
                direct_log_ratio_mean - source_stats.log_ratio_norm,
                jnp.asarray(0.0, dtype=fidelity.dtype),
            ),
            jnp.asarray(0.0, dtype=fidelity.dtype),
        )
        equation_relative_residual = jnp.sqrt(
            jnp.maximum(1.0 / jnp.maximum(fidelity, eps) - 1.0, 0.0)
        )
        action_norm = (
            jnp.asarray(source_norm, dtype=fidelity.dtype)
            * jnp.abs(source_stats.normalization) ** 2
            / jnp.maximum(fidelity, eps)
        )
        return source_stats._replace(
            loss=1.0 - fidelity + self.lit_config.nqs_reverse_kl_weight * reverse_kl,
            fidelity=fidelity,
            reverse_kl=jnp.real(reverse_kl),
            residual_norm=jnp.asarray(source_norm, dtype=fidelity.dtype)
            * equation_relative_residual**2,
            equation_relative_residual=equation_relative_residual,
            action_norm=jnp.real(action_norm),
            invalid_sample_fraction=1.0
            - total_valid_count / jnp.maximum(total_count, 1.0),
            estimator_mode=jnp.asarray(1, dtype=jnp.int32),
            direct_hloc_rmse=jnp.real(direct_hloc_rmse),
            direct_hloc_std=jnp.real(direct_hloc_std),
            direct_hloc_sem=jnp.real(direct_hloc_sem),
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
        force_direct: bool = False,
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
        if not force_direct and not self._should_use_direct_psi(stats):
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
        ess_fraction = float(jax.device_get(stats.reweight_ess_fraction))
        invalid_fraction = float(jax.device_get(stats.invalid_sample_fraction))
        return (
            not np.isfinite(ess_fraction)
            or not np.isfinite(invalid_fraction)
            or invalid_fraction > 0.0
            or ess_fraction < threshold
        )

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
            use_direct=jnp.asarray(False),
        )

    def _init_nqs_update_carry(
        self,
        batched_data,
        rng,
        response_params,
    ) -> _NQSUpdateCarry:
        flat_params, _ = ravel_pytree(response_params)
        return _NQSUpdateCarry(
            direct=self._init_direct_psi_carry(batched_data, rng),
            spring=_SpringState(previous_direction=jnp.zeros_like(flat_params)),
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


_AXIS_NAMES = ("x", "y", "z")


def _lit_omega_grid(config: MolecularLITConfig) -> np.ndarray:
    if config.omega_values:
        omega = np.asarray(tuple(float(value) for value in config.omega_values))
        if omega.ndim != 1 or omega.size == 0:
            msg = "lit.omega_values must be a non-empty one-dimensional sequence."
            raise ValueError(msg)
        if not np.all(np.isfinite(omega)):
            msg = "lit.omega_values must contain only finite values."
            raise ValueError(msg)
        if np.any(np.diff(omega) <= 0.0):
            msg = "lit.omega_values must be strictly increasing."
            raise ValueError(msg)
        return omega
    if config.omega_points < 1:
        msg = "lit.omega_points must be positive."
        raise ValueError(msg)
    if not np.isfinite(config.omega_min) or not np.isfinite(config.omega_max):
        msg = "lit.omega_min and lit.omega_max must be finite."
        raise ValueError(msg)
    if config.omega_points > 1 and config.omega_max <= config.omega_min:
        msg = "lit.omega_max must exceed lit.omega_min for a serial scan."
        raise ValueError(msg)
    return np.linspace(
        config.omega_min,
        config.omega_max,
        config.omega_points,
    )


def _continuation_min_step(
    config: MolecularLITConfig,
    spectrum_omega: np.ndarray,
) -> float:
    configured = config.nqs_continuation_min_step
    if configured is not None:
        return float(configured)
    candidates = [float(config.nqs_continuation_step_fraction) * float(config.eta)]
    spacings = np.diff(np.asarray(spectrum_omega, dtype=np.float64))
    if spacings.size:
        candidates.append(float(np.min(spacings)))
    return max(np.finfo(np.float64).eps, min(candidates))


def _physics_continuation_step(stats, *, gap: float, fraction: float, min_step: float):
    """Choose a homotopy step from the inherited LIT residual estimate.

    Returns:
        A positive step no larger than the remaining target gap.
    """
    lit = float(jax.device_get(stats.lit))
    source_norm = float(jax.device_get(stats.source_norm))
    if (
        np.isfinite(lit)
        and np.isfinite(source_norm)
        and lit > 0.0
        and source_norm > 0.0
    ):
        proposed = float(fraction) * np.sqrt(source_norm / lit)
    else:
        proposed = float(min_step)
    return min(float(gap), max(float(min_step), proposed))


def _finite_valid_nqs_stats(stats) -> bool:
    values = (
        float(jax.device_get(stats.loss)),
        float(jax.device_get(stats.fidelity)),
        float(jax.device_get(stats.reverse_kl)),
        float(jax.device_get(stats.invalid_sample_fraction)),
    )
    return bool(np.all(np.isfinite(values))) and values[-1] <= 0.0


def _continuation_probe_is_acceptable(current, candidate, *, retention: float) -> bool:
    if not _finite_valid_nqs_stats(candidate):
        return False
    current_fidelity = float(jax.device_get(current.fidelity))
    candidate_fidelity = float(jax.device_get(candidate.fidelity))
    if not np.isfinite(current_fidelity):
        return False
    required = max(0.0, float(retention) * current_fidelity)
    return candidate_fidelity >= required


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
    """Merge independently scaled source moments without overflowing.

    Returns:
        The combined source moments in units of the larger input scale.
    """
    scale = jnp.maximum(left.ratio_scale, right.ratio_scale)
    tiny = jnp.asarray(jnp.finfo(scale.dtype).tiny, dtype=scale.dtype)
    scale = jnp.maximum(scale, tiny)
    left_factor = left.ratio_scale / scale
    right_factor = right.ratio_scale / scale

    def log_moment(moment, weight, factor):
        factor_sq = factor**2
        safe_factor = jnp.maximum(factor, tiny)
        return factor_sq * (moment + 2.0 * jnp.log(safe_factor) * weight)

    merged = jax.tree.map(operator.add, left, right)
    return merged._replace(
        ratio_scale=scale,
        ratio_sum=left_factor * left.ratio_sum + right_factor * right.ratio_sum,
        ratio_abs2_sum=(
            left_factor**2 * left.ratio_abs2_sum
            + right_factor**2 * right.ratio_abs2_sum
        ),
        psi_weight_sum=(
            left_factor**2 * left.psi_weight_sum
            + right_factor**2 * right.psi_weight_sum
        ),
        psi_weight_sq_sum=(
            left_factor**4 * left.psi_weight_sq_sum
            + right_factor**4 * right.psi_weight_sq_sum
        ),
        psi_log_ratio_abs2_sum=(
            log_moment(
                left.psi_log_ratio_abs2_sum,
                left.psi_weight_sum,
                left_factor,
            )
            + log_moment(
                right.psi_log_ratio_abs2_sum,
                right.psi_weight_sum,
                right_factor,
            )
        ),
    )


def _solve_sr_direction_chunked(
    chunk_rows: tuple[int, ...],
    score_aug_chunk,
    grad_flat,
    damping,
    *,
    kernel_null_vectors=None,
    kernel_projector_scale=1.0,
):
    """Solve the SR system from score chunks without materializing all scores.

    Args:
        chunk_rows: Row count for each real-augmented score chunk.
        score_aug_chunk: Callable returning one real-augmented score chunk.
        grad_flat: Flattened objective gradient.
        damping: Positive SR damping added to the metric.
        kernel_null_vectors: Known left-null vectors of the centered score
            matrix, used to lift the Gram matrix null space.
        kernel_projector_scale: Positive eigenvalue assigned to those lifted
            null-space directions.

    Returns:
        Flattened preconditioned SR direction.

    Raises:
        ValueError: If no score chunks are provided.
    """
    if not chunk_rows:
        msg = "At least one SR score chunk is required."
        raise ValueError(msg)
    parameter_count = grad_flat.shape[0]
    sample_count = sum(int(rows) for rows in chunk_rows)
    original_dtype = grad_flat.dtype
    # Local x64 is deliberately enabled even when the rest of the workflow is
    # float32.  The Gram solve is small compared with score construction, and
    # this is the numerically sensitive part of SPRING.
    with _enable_x64(True):
        solve_dtype = jnp.float64
        grad_solve = grad_flat.astype(solve_dtype)
        damping_solve = jnp.asarray(damping, dtype=solve_dtype)
        score_chunks = tuple(
            score_aug_chunk(index).astype(solve_dtype)
            for index in range(len(chunk_rows))
        )
        if parameter_count <= sample_count:
            metric = jnp.zeros(
                (parameter_count, parameter_count),
                dtype=solve_dtype,
            )
            for score_aug in score_chunks:
                metric = metric + score_aug.T @ score_aug
            metric = (metric + metric.T) / 2.0
            metric = metric + damping_solve * jnp.eye(
                parameter_count,
                dtype=solve_dtype,
            )
            chol = jsp.linalg.cho_factor(metric, lower=True)
            direction = jsp.linalg.cho_solve(chol, grad_solve)
        else:
            row_blocks = []
            for row_score_aug in score_chunks:
                column_blocks = [
                    row_score_aug @ column_score_aug.T
                    for column_score_aug in score_chunks
                ]
                row_blocks.append(jnp.concatenate(column_blocks, axis=1))
            kernel = jnp.concatenate(row_blocks, axis=0)
            kernel = (kernel + kernel.T) / 2.0
            if kernel_null_vectors is not None:
                null_vectors = jnp.asarray(
                    kernel_null_vectors,
                    dtype=solve_dtype,
                )
                if null_vectors.ndim == 1:
                    null_vectors = null_vectors[None, :]
                null_norm = jnp.linalg.norm(null_vectors, axis=1, keepdims=True)
                normalized_null = jnp.where(
                    null_norm > 0.0,
                    null_vectors
                    / jnp.maximum(
                        null_norm,
                        jnp.asarray(jnp.finfo(solve_dtype).tiny, dtype=solve_dtype),
                    ),
                    jnp.asarray(0.0, dtype=solve_dtype),
                )
                kernel = kernel + jnp.asarray(
                    kernel_projector_scale,
                    dtype=solve_dtype,
                ) * (normalized_null.T @ normalized_null)
            kernel = (kernel + kernel.T) / 2.0
            kernel = kernel + damping_solve * jnp.eye(
                sample_count,
                dtype=solve_dtype,
            )
            rhs = jnp.concatenate(
                [score_aug @ grad_solve for score_aug in score_chunks],
                axis=0,
            )
            chol = jsp.linalg.cho_factor(kernel, lower=True)
            alpha = jsp.linalg.cho_solve(chol, rhs)
            projected = jnp.zeros_like(grad_solve)
            start = 0
            for score_aug, rows in zip(score_chunks, chunk_rows, strict=True):
                stop = start + int(rows)
                projected = projected + score_aug.T @ alpha[start:stop]
                start = stop
            direction = (grad_solve - projected) / damping_solve
        direction = direction.astype(original_dtype)
    return jnp.where(
        jnp.all(jnp.isfinite(direction)),
        direction,
        jnp.zeros_like(direction),
    )


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


def _regularized_action_gradient(
    score,
    ratio,
    source_weight,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    eps: float | jnp.ndarray,
):
    """Return the PRL fidelity-minus-reverse-KL action-state gradient.

    ``ratio`` is ``barPsi / Phi`` and ``score`` is
    ``d log(barPsi) / d theta``.  Rescaling all ratios by their largest
    magnitude prevents overflow without changing the fidelity, reverse KL, or
    either gradient.
    """
    real_dtype = ratio.real.dtype
    eps_array = jnp.asarray(eps, dtype=real_dtype)
    finite_score = jnp.all(
        jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
        axis=1,
    )
    finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
    finite_weight = jnp.isfinite(source_weight) & (source_weight >= 0.0)
    valid = finite_score & finite_ratio & finite_weight
    score = jnp.where(valid[:, None], score, jnp.asarray(0.0, dtype=score.dtype))
    ratio = jnp.where(valid, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
    source_weight = jnp.where(
        valid,
        source_weight,
        jnp.asarray(0.0, dtype=source_weight.dtype),
    )
    safe_weight_sum = jnp.maximum(jnp.sum(source_weight), eps_array)
    phi_weight = source_weight / safe_weight_sum
    max_ratio_abs = jnp.max(jnp.where(phi_weight > 0.0, jnp.abs(ratio), 0.0))
    ratio_scale = jnp.where(
        max_ratio_abs > 0.0,
        max_ratio_abs,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    scaled_ratio = ratio / jax.lax.stop_gradient(ratio_scale)
    ratio_abs2 = jnp.abs(scaled_ratio) ** 2
    ratio_norm = jnp.sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps_array)
    has_action_mass = jnp.isfinite(ratio_norm) & (ratio_norm > 0.0)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm

    score_mean = jnp.sum(psi_weight[:, None] * score, axis=0, keepdims=True)
    centered_score = score - score_mean
    amplitude = jnp.sum(phi_weight * scaled_ratio)
    score_covariance = jnp.sum(
        phi_weight[:, None] * scaled_ratio[:, None] * centered_score,
        axis=0,
    )
    fidelity_gradient = 2.0 * jnp.real(
        jnp.conj(amplitude) * score_covariance / safe_ratio_norm
    )

    log_ratio_abs2 = 2.0 * jnp.log(jnp.maximum(jnp.abs(scaled_ratio), eps_array))
    log_ratio_mean = jnp.sum(psi_weight * log_ratio_abs2)
    centered_log_ratio = log_ratio_abs2 - log_ratio_mean
    reverse_kl_gradient = 2.0 * jnp.real(
        jnp.sum(
            psi_weight[:, None] * centered_score * centered_log_ratio[:, None],
            axis=0,
        )
    )
    combined_gradient = (
        fidelity_gradient
        - jnp.asarray(
            reverse_kl_weight,
            dtype=real_dtype,
        )
        * reverse_kl_gradient
    )
    reverse_kl = jnp.where(
        has_action_mass,
        jnp.maximum(
            log_ratio_mean - jnp.log(safe_ratio_norm),
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    fidelity = jnp.where(
        has_action_mass,
        jnp.clip(jnp.abs(amplitude) ** 2 / safe_ratio_norm, 0.0, 1.0),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    return (
        combined_gradient,
        fidelity_gradient,
        reverse_kl_gradient,
        psi_weight,
        centered_score,
        fidelity,
        reverse_kl,
    )


def _spring_direction_chunked(
    chunk_rows: tuple[int, ...],
    score_aug_chunk,
    grad_flat,
    state: _SpringState,
    *,
    epsilon_scale: float | jnp.ndarray,
    damping_floor: float | jnp.ndarray,
    decay: float | jnp.ndarray,
    qfi_trace=None,
    kernel_null_vectors=None,
):
    """Solve the scale-invariant SPRING system and retain unscaled history.

    Returns:
        Unscaled direction, updated SPRING state, and absolute damping.
    """
    if qfi_trace is None:
        qfi_trace = jnp.asarray(0.0, dtype=grad_flat.dtype)
        for index in range(len(chunk_rows)):
            score_aug = score_aug_chunk(index)
            qfi_trace = qfi_trace + jnp.sum(score_aug**2)
    parameter_count = jnp.asarray(max(int(grad_flat.shape[0]), 1), grad_flat.dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.maximum(
        jnp.asarray(epsilon_scale, dtype=grad_flat.dtype) * mean_metric_diagonal,
        jnp.asarray(damping_floor, dtype=grad_flat.dtype),
    )
    rhs = (
        grad_flat
        + damping
        * jnp.asarray(
            decay,
            dtype=grad_flat.dtype,
        )
        * state.previous_direction
    )
    direction = _solve_sr_direction_chunked(
        chunk_rows,
        score_aug_chunk,
        rhs,
        damping,
        kernel_null_vectors=kernel_null_vectors,
    )
    valid_system = (
        jnp.isfinite(qfi_trace)
        & (qfi_trace > 0.0)
        & jnp.isfinite(damping)
        & jnp.all(jnp.isfinite(grad_flat))
        & jnp.all(jnp.isfinite(state.previous_direction))
        & jnp.all(jnp.isfinite(direction))
    )
    direction = jnp.where(valid_system, direction, jnp.zeros_like(direction))
    return (
        direction,
        _SpringState(previous_direction=jax.lax.stop_gradient(direction)),
        damping,
    )


def _scaled_direction_updates(
    params,
    direction,
    *,
    learning_rate: float,
    max_norm: float | None,
):
    _, unravel_fn = ravel_pytree(params)
    scale = jnp.asarray(learning_rate, dtype=direction.dtype)
    if max_norm is not None:
        update_norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(
            scale,
            jnp.asarray(max_norm, dtype=direction.dtype)
            / (update_norm + jnp.asarray(1e-12, dtype=direction.dtype)),
        )
    return unravel_fn(scale * direction)


def _regularized_loss(stats, reverse_kl_weight: float):
    return (
        1.0
        - stats.fidelity
        + jnp.asarray(
            reverse_kl_weight,
            dtype=stats.fidelity.dtype,
        )
        * stats.reverse_kl
    )


def _is_better_nqs_checkpoint(candidate, incumbent) -> bool:
    """Compare held-out checkpoints, rejecting non-finite/invalid estimates.

    Returns:
        Whether the candidate should replace the incumbent.
    """

    def score(stats):
        loss = float(jax.device_get(stats.loss))
        fidelity = float(jax.device_get(stats.fidelity))
        reverse_kl = float(jax.device_get(stats.reverse_kl))
        invalid = float(jax.device_get(stats.invalid_sample_fraction))
        finite = bool(np.all(np.isfinite((loss, fidelity, reverse_kl, invalid))))
        valid = finite and invalid <= 0.0
        return valid, -loss, fidelity, -reverse_kl

    candidate_score = score(candidate)
    incumbent_score = score(incumbent)
    if not candidate_score[0]:
        return False
    if not incumbent_score[0]:
        return True
    return candidate_score[1:] > incumbent_score[1:]


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
