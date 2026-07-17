# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole NQS-LIT workflow."""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Mapping
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
    parity_log_amplitude_loss,
    parity_project_log_amplitude,
    restore_params_from_checkpoint,
    source_aligned_vector_logpsi,
)
from jaqmc.response.source_sector import (
    SourceSector,
    discover_source_sector,
    source_sector_covariance_loss,
    transform_molecule_data,
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

_ATOM_PARITY_PENDING_SECTOR_LABEL = "atom_parity_pending"
_ATOM_HARD_ODD_SECTOR_LABEL = "atom_odd_hard"
_ATOM_HARD_EVEN_SECTOR_LABEL = "atom_even_hard"


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
    nqs_source_center_override: float | tuple[float, float, float] | None = None
    nqs_source_norm_override: float | tuple[float, float, float] | None = None
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
    nqs_stage_fidelity_min: float = 0.0
    nqs_stage_reweight_ess_fraction_min: float = 0.0
    nqs_stage_fidelity_gain_min: float = 0.0
    nqs_continuation_allow_min_step_override: bool = True
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
    nqs_source_aligned: bool = False
    nqs_source_aligned_residual_scale: float = 1e-3
    # Runtime support is deliberately restricted to an automatically selected
    # hard atomic parity (opposite the diagnosed ground-state parity), or a
    # symmetry-free multi-center C1 response.  Legacy values are still accepted
    # so older YAML files deserialize, but no longer enable vector-head soft
    # covariance training.
    nqs_source_symmetry_mode: str = "atom_c1"
    nqs_source_symmetry_weight: float = 0.0
    nqs_source_symmetry_learning_rate: float = 1e-3
    nqs_source_symmetry_max_norm: float | None = 1e-2
    nqs_source_symmetry_eval_batch_size: int = 256
    nqs_source_symmetry_tolerance: float = 1e-5
    nqs_source_symmetry_max_operations: int = 16
    # Hard guard on the worst batch-mean loss among the active non-identity
    # operations.  Set null explicitly to disable the guard.
    nqs_source_symmetry_max_covariance: float | None = 1e-3
    # Reject an atomic checkpoint unless one inversion parity has a held-out
    # residual below this threshold.  The response uses the opposite parity.
    nqs_atomic_ground_parity_max_loss: float = 1e-3
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


class _SpringOptimizerDiagnostics(NamedTuple):
    """Low-cost scalar diagnostics for one source-sampled SPRING update."""

    available: jax.Array
    combined_gradient_norm: jax.Array
    fidelity_gradient_norm: jax.Array
    weighted_reverse_kl_gradient_norm: jax.Array
    fidelity_kl_cosine: jax.Array
    gradient_cancellation_ratio: jax.Array
    direction_norm: jax.Array
    update_norm: jax.Array
    clip_factor: jax.Array
    damping: jax.Array
    mean_metric_diagonal: jax.Array
    history_gradient_ratio: jax.Array
    parameter_group_gradient_rms: jax.Array
    parameter_group_update_norm: jax.Array


_SPRING_PARAMETER_GROUPS = (
    "raw",
    "source_coefficient",
    "residual_log_scale",
)


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


class _SourceCovarianceMetrics(NamedTuple):
    """Held-out covariance summary across active symmetry operations.

    Each operation loss is already averaged over the evaluation batch.  The
    mean remains the soft objective, while the maximum is the hard admission
    criterion for checkpoints and continuation.
    """

    mean_loss: jnp.ndarray
    max_loss: jnp.ndarray
    worst_operation_index: jnp.ndarray


class _AtomicParityResolution(NamedTuple):
    """Host-side atomic ground/response parity admission result."""

    ground_parity: int
    response_parity: int
    even_loss: float
    odd_loss: float
    selected_loss: float


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
        # Resolve the deliberately narrow atom/C1 policy from physical fixed
        # fields before checkpoint loading, sampling, or compilation.  The
        # shape-only example replaces fixed values and must never classify the
        # geometry.
        source_sector = self._configured_source_sector(batched_data.data)
        shape_example = batched_data.unbatched_example()

        checkpoint_step, ground_params, ground_logpsi = self._resolve_nqs_ground_state(
            shape_example, ground_rng
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
        parity_resolution = self._resolve_atomic_parity(
            ground_logpsi,
            ground_params,
            batched_data,
            source_sector,
        )
        source_sector = _resolve_atomic_parity_sector(
            source_sector,
            parity_resolution.response_parity,
        )

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
        source_covariance_loss = np.zeros_like(lit)
        source_covariance_max_loss = np.zeros_like(lit)
        direct_hloc_rmse = np.full_like(lit, np.nan)
        direct_hloc_std = np.full_like(lit, np.nan)
        direct_hloc_sem = np.full_like(lit, np.nan)
        estimator_mode = np.zeros_like(lit, dtype=np.int64)
        selected_iteration = np.zeros_like(lit, dtype=np.int64)
        normalization = np.zeros((len(axes), len(omega)), dtype=np.complex128)
        correction_overlap = np.zeros_like(normalization)
        source_centers = np.zeros(len(axes), dtype=np.float64)
        axis_source_norm = np.zeros(len(axes), dtype=np.float64)
        pure_source_covariance_loss = np.zeros(len(axes), dtype=np.float64)
        pure_source_covariance_max_loss = np.zeros(len(axes), dtype=np.float64)
        pure_source_covariance_worst_operation = np.full(
            len(axes),
            -1,
            dtype=np.int64,
        )
        warm_start_selected_iteration = np.zeros(len(axes), dtype=np.int64)
        continuation_axis: list[int] = []
        continuation_omega: list[float] = []
        continuation_optimized: list[bool] = []
        continuation_selected_iteration: list[int] = []
        continuation_fidelity: list[float] = []
        continuation_reverse_kl: list[float] = []
        continuation_invalid_sample_fraction: list[float] = []
        continuation_source_covariance_loss: list[float] = []
        continuation_source_covariance_max_loss: list[float] = []
        continuation_inherited_fidelity: list[float] = []
        continuation_step: list[float] = []
        continuation_bisections: list[int] = []
        continuation_probe_accepted: list[bool] = []
        continuation_min_step_override: list[bool] = []

        source_guard_operations = self._source_guard_operations(source_sector)
        logger.info(
            "NQS-LIT response_policy=%s source_sector=%s order=%d "
            "soft_operations=%d source_guard_operations=%d",
            _response_symmetry_policy(
                source_sector,
                parity_resolution.response_parity,
            ),
            source_sector.label,
            source_sector.order,
            len(self._active_source_sector_operations(source_sector)),
            len(source_guard_operations),
        )
        source_sector_active_operations = np.asarray(
            self._active_source_sector_operations(source_sector),
            dtype=np.float64,
        ).reshape((-1, 3, 3))
        source_guard_operations_array = np.asarray(
            source_guard_operations,
            dtype=np.float64,
        ).reshape((-1, 3, 3))

        # Estimate all three dipole centers on one ground-state chain.  The
        # atomic inversion sector is retained only to project the affine,
        # origin-dependent center before norms are finalized; both supported
        # response policies themselves use one scalar axis at a time.
        (
            vector_source_centers,
            vector_source_norms,
            batched_data,
            sampler_state,
            rng,
        ) = self._estimate_vector_source_stats(
            ground_params,
            batched_data,
            sampler_state,
            ground_sample_plan,
            rng,
            source_sector=source_sector,
        )

        for axis_pos, axis in enumerate(axes):
            source_center = float(vector_source_centers[axis])
            axis_phi_norm = float(vector_source_norms[axis])
            source_centers[axis_pos] = source_center
            axis_source_norm[axis_pos] = axis_phi_norm
            logger.info(
                "axis=%s source_center=%.8e source_norm=%.8e",
                _AXIS_NAMES[axis],
                source_center,
                axis_phi_norm,
            )

            rng, response_rng = jax.random.split(rng)
            response_apply, response_vector_apply, response_params = (
                self._make_response_ansatz(
                    shape_example,
                    response_rng,
                    ground_params,
                    axis=axis,
                    source_center=source_center,
                    source_centers=vector_source_centers,
                    source_sector=source_sector,
                    response_parity=parity_resolution.response_parity,
                    ground_logpsi=ground_logpsi,
                    initialization_data=_cyclic_batched_data_chunk(
                        batched_data,
                        min(
                            batched_data.batch_size,
                            int(self.lit_config.nqs_source_symmetry_eval_batch_size),
                        ),
                        0,
                    ),
                    initial_omega=(
                        float(self.lit_config.nqs_warm_start_omega)
                        if self.lit_config.nqs_warm_start_omega is not None
                        else float(omega[0])
                    ),
                )
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
            pure_source_metrics = self._validate_pure_source_covariance(
                ground_logpsi,
                ground_params,
                eval_pool,
                source_sector,
                vector_source_centers,
                axis=axis,
                response_parity=parity_resolution.response_parity,
            )
            pure_source_covariance_loss[axis_pos] = float(pure_source_metrics.mean_loss)
            pure_source_covariance_max_loss[axis_pos] = float(
                pure_source_metrics.max_loss
            )
            pure_source_covariance_worst_operation[axis_pos] = int(
                pure_source_metrics.worst_operation_index
            )

            update_step = self._make_nqs_update_step(
                response_apply,
                ground_params,
                ground_logpsi,
                ground_energy,
                response_vector_apply=response_vector_apply,
                source_sector=source_sector,
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
                continuation_source_covariance_loss.append(
                    float(host_bridge_stats.source_covariance_loss)
                )
                continuation_source_covariance_max_loss.append(
                    float(host_bridge_stats.source_covariance_max_loss)
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
                covariance_metrics = _coerce_source_covariance_metrics(
                    getattr(
                        update_step,
                        "source_covariance_metrics",
                        update_step.source_covariance_loss,
                    )(
                        response_params,
                        eval_pool,
                    )
                )
                stats = stats._replace(
                    loss=_regularized_loss(
                        stats,
                        self.lit_config.nqs_reverse_kl_weight,
                    )
                    + jnp.asarray(
                        self.lit_config.nqs_source_symmetry_weight,
                        dtype=stats.loss.dtype,
                    )
                    * covariance_metrics.mean_loss,
                    source_covariance_loss=covariance_metrics.mean_loss,
                    source_covariance_max_loss=covariance_metrics.max_loss,
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
                source_covariance_loss[axis_pos, omega_pos] = float(
                    host_stats.source_covariance_loss
                )
                source_covariance_max_loss[axis_pos, omega_pos] = float(
                    host_stats.source_covariance_max_loss
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
            source_covariance_loss=source_covariance_loss,
            source_covariance_max_loss=source_covariance_max_loss,
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
            nqs_source_aligned=bool(self.lit_config.nqs_source_aligned),
            nqs_source_aligned_residual_scale=(
                self.lit_config.nqs_source_aligned_residual_scale
            ),
            nqs_source_symmetry_mode=self.lit_config.nqs_source_symmetry_mode,
            nqs_source_symmetry_weight=self.lit_config.nqs_source_symmetry_weight,
            nqs_source_symmetry_learning_rate=(
                self.lit_config.nqs_source_symmetry_learning_rate
            ),
            nqs_source_symmetry_max_norm=_optional_float(
                self.lit_config.nqs_source_symmetry_max_norm
            ),
            nqs_source_symmetry_eval_batch_size=(
                self.lit_config.nqs_source_symmetry_eval_batch_size
            ),
            nqs_source_symmetry_tolerance=(
                self.lit_config.nqs_source_symmetry_tolerance
            ),
            nqs_source_symmetry_max_operations=(
                self.lit_config.nqs_source_symmetry_max_operations
            ),
            nqs_source_symmetry_max_covariance=_optional_float(
                self.lit_config.nqs_source_symmetry_max_covariance
            ),
            nqs_atomic_ground_parity_max_loss=(
                self.lit_config.nqs_atomic_ground_parity_max_loss
            ),
            source_sector_label=source_sector.label,
            source_sector_order=source_sector.order,
            source_sector_active_operations=source_sector_active_operations,
            source_guard_operations=source_guard_operations_array,
            response_symmetry_policy=_response_symmetry_policy(
                source_sector,
                parity_resolution.response_parity,
            ),
            response_hard_parity=bool(_is_atom_hard_parity_sector(source_sector)),
            atomic_ground_parity=parity_resolution.ground_parity,
            response_parity=parity_resolution.response_parity,
            atomic_ground_even_parity_loss=parity_resolution.even_loss,
            atomic_ground_odd_parity_loss=parity_resolution.odd_loss,
            atomic_ground_selected_parity_loss=parity_resolution.selected_loss,
            response_symmetry_center=np.asarray(
                source_sector.center,
                dtype=np.float64,
            ),
            response_soft_symmetry_enabled=False,
            nqs_source_symmetry_legacy_training_ignored=True,
            vector_source_centers=vector_source_centers,
            vector_source_norms=vector_source_norms,
            pure_source_covariance_loss=pure_source_covariance_loss,
            pure_source_covariance_max_loss=pure_source_covariance_max_loss,
            pure_source_covariance_worst_active_operation=(
                pure_source_covariance_worst_operation
            ),
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
            continuation_source_covariance_loss=np.asarray(
                continuation_source_covariance_loss,
                dtype=np.float64,
            ),
            continuation_source_covariance_max_loss=np.asarray(
                continuation_source_covariance_max_loss,
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
            nqs_stage_fidelity_min=self.lit_config.nqs_stage_fidelity_min,
            nqs_stage_reweight_ess_fraction_min=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            nqs_stage_fidelity_gain_min=self.lit_config.nqs_stage_fidelity_gain_min,
            nqs_continuation_allow_min_step_override=bool(
                self.lit_config.nqs_continuation_allow_min_step_override
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
        self._validate_source_sector_config()
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

    def _validate_source_sector_config(self) -> None:
        _three_component_override(
            self.lit_config.nqs_source_center_override,
            name="lit.nqs_source_center_override",
        )
        _three_component_override(
            self.lit_config.nqs_source_norm_override,
            name="lit.nqs_source_norm_override",
            positive=True,
        )
        residual_scale = self.lit_config.nqs_source_aligned_residual_scale
        if not np.isfinite(residual_scale) or residual_scale <= 0.0:
            msg = "lit.nqs_source_aligned_residual_scale must be positive."
            raise ValueError(msg)
        mode = str(self.lit_config.nqs_source_symmetry_mode).lower()
        valid_modes = {
            "atom_c1",
            "off",
            "none",
            "identity",
            "c1",
            "auto",
            "on",
            "general",
            "inversion",
        }
        if mode not in valid_modes:
            msg = (
                "lit.nqs_source_symmetry_mode must be one of "
                f"{sorted(valid_modes)}, got "
                f"{self.lit_config.nqs_source_symmetry_mode!r}."
            )
            raise ValueError(msg)
        weight = self.lit_config.nqs_source_symmetry_weight
        if not np.isfinite(weight) or weight < 0.0:
            msg = "lit.nqs_source_symmetry_weight must be finite and nonnegative."
            raise ValueError(msg)
        learning_rate = self.lit_config.nqs_source_symmetry_learning_rate
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            msg = "lit.nqs_source_symmetry_learning_rate must be positive."
            raise ValueError(msg)
        max_norm = self.lit_config.nqs_source_symmetry_max_norm
        if max_norm is not None and (
            not np.isfinite(max_norm) or float(max_norm) <= 0.0
        ):
            msg = "lit.nqs_source_symmetry_max_norm must be positive or null."
            raise ValueError(msg)
        if self.lit_config.nqs_source_symmetry_eval_batch_size < 1:
            msg = "lit.nqs_source_symmetry_eval_batch_size must be positive."
            raise ValueError(msg)
        tolerance = self.lit_config.nqs_source_symmetry_tolerance
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            msg = "lit.nqs_source_symmetry_tolerance must be positive."
            raise ValueError(msg)
        if self.lit_config.nqs_source_symmetry_max_operations < 1:
            msg = "lit.nqs_source_symmetry_max_operations must be positive."
            raise ValueError(msg)
        max_covariance = self.lit_config.nqs_source_symmetry_max_covariance
        if max_covariance is not None and (
            not np.isfinite(max_covariance) or float(max_covariance) <= 0.0
        ):
            msg = "lit.nqs_source_symmetry_max_covariance must be positive or null."
            raise ValueError(msg)
        self._validate_atomic_parity_config()

    def _validate_atomic_parity_config(self) -> None:
        """Validate the mandatory atomic checkpoint parity admission guard.

        Raises:
            ValueError: If the threshold is non-finite or outside ``(0, 1)``.
        """
        parity_maximum = self.lit_config.nqs_atomic_ground_parity_max_loss
        if not np.isfinite(parity_maximum) or not 0.0 < parity_maximum < 1.0:
            msg = (
                "lit.nqs_atomic_ground_parity_max_loss must be finite and "
                "strictly between 0 and 1."
            )
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
        stage_thresholds = (
            ("nqs_stage_fidelity_min", self.lit_config.nqs_stage_fidelity_min),
            (
                "nqs_stage_reweight_ess_fraction_min",
                self.lit_config.nqs_stage_reweight_ess_fraction_min,
            ),
            (
                "nqs_stage_fidelity_gain_min",
                self.lit_config.nqs_stage_fidelity_gain_min,
            ),
        )
        for name, value in stage_thresholds:
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                msg = f"lit.{name} must be finite and between 0 and 1."
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

    def _configured_source_sector(self, geometry_data) -> SourceSector:
        """Resolve the restricted atomic-hard-parity or molecular-C1 policy.

        Returns:
            ``atom_parity_pending`` with identity and inversion for exactly one
            nucleus, or the identity-only ``C1`` sector for a multi-nuclear
            geometry with no discovered nontrivial operation.  The atomic
            ground checkpoint later selects the opposite response parity.

        Raises:
            NotImplementedError: If a multi-nuclear geometry is not C1.
            RuntimeError: If atomic geometry discovery omits inversion.
        """
        mode = str(self.lit_config.nqs_source_symmetry_mode).lower()
        sector = discover_source_sector(
            geometry_data.atoms,
            geometry_data.charges,
            tolerance=float(self.lit_config.nqs_source_symmetry_tolerance),
        )
        atom_count = int(np.asarray(geometry_data.atoms).shape[0])
        if atom_count == 1:
            identity = next(
                operation
                for operation in sector.operations
                if _is_identity_operation(operation)
            )
            inversion = next(
                (
                    operation
                    for operation in sector.operations
                    if np.allclose(
                        np.asarray(operation),
                        -np.eye(3),
                        rtol=0.0,
                        atol=self.lit_config.nqs_source_symmetry_tolerance,
                    )
                ),
                None,
            )
            if inversion is None:
                msg = "Atomic source-sector discovery did not contain inversion."
                raise RuntimeError(msg)
            resolved = replace(
                sector,
                operations=(identity, inversion),
                label=_ATOM_PARITY_PENDING_SECTOR_LABEL,
            )
        elif sector.is_trivial:
            resolved = replace(sector, label="C1")
        else:
            msg = (
                "NQS-LIT currently supports only one-center atoms with an "
                "automatically selected hard response parity and multi-center "
                "C1 molecules without "
                f"spatial symmetry; discovered n_atoms={atom_count}, "
                f"sector={sector.label!r}, order={sector.order}."
            )
            raise NotImplementedError(msg)

        if mode != "atom_c1" or self.lit_config.nqs_source_symmetry_weight > 0.0:
            logger.warning(
                "Legacy lit.nqs_source_symmetry_mode and soft-training weight "
                "are ignored by the restricted atom/C1 runtime (mode=%s, "
                "weight=%.6g); source-guard thresholds remain active; resolved "
                "policy=%s.",
                mode,
                self.lit_config.nqs_source_symmetry_weight,
                _response_symmetry_policy(resolved, 0),
            )
        return resolved

    def _resolve_atomic_parity(
        self,
        ground_logpsi,
        ground_params,
        batched_data: BatchedData,
        source_sector: SourceSector,
    ) -> _AtomicParityResolution:
        """Diagnose atomic ground parity and select the opposite response parity.

        C1 molecules have no hard spatial projector and return zero parity
        labels with non-applicable losses.  An atom is accepted only when the
        checkpoint is a clean inversion eigenstate on the held-out batch.

        Returns:
            The diagnosed ground parity, opposite response parity, and both
            held-out parity losses.  C1 returns zero characters and NaN losses.

        Raises:
            RuntimeError: If neither atomic inversion parity passes the hard
                admission threshold, the losses are invalid, or the result is
                ambiguous because both amplitudes vanish.
        """
        if not _is_atom_parity_sector(source_sector):
            return _AtomicParityResolution(
                0, 0, float("nan"), float("nan"), float("nan")
            )

        evaluation_batch = _cyclic_batched_data_chunk(
            batched_data,
            min(
                batched_data.batch_size,
                int(self.lit_config.nqs_source_symmetry_eval_batch_size),
            ),
            0,
        )
        inversion = jnp.asarray(
            -np.eye(3),
            dtype=evaluation_batch.data.electrons.dtype,
        )
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=evaluation_batch.data.electrons.dtype,
        )

        @jax.jit
        def evaluate(local_ground_params, local_batch):
            def paired_logs(data):
                return (
                    ground_logpsi(local_ground_params, data),
                    ground_logpsi(
                        local_ground_params,
                        transform_molecule_data(data, inversion, symmetry_center),
                    ),
                )

            log_amplitudes, inverted_log_amplitudes = jax.vmap(
                paired_logs,
                in_axes=(local_batch.vmap_axis,),
            )(local_batch.data)
            return (
                parity_log_amplitude_loss(
                    log_amplitudes,
                    inverted_log_amplitudes,
                    1,
                ),
                parity_log_amplitude_loss(
                    log_amplitudes,
                    inverted_log_amplitudes,
                    -1,
                ),
            )

        even_loss_array, odd_loss_array = jax.device_get(
            evaluate(ground_params, evaluation_batch)
        )
        even_loss = float(even_loss_array)
        odd_loss = float(odd_loss_array)
        ground_parity = 1 if even_loss <= odd_loss else -1
        selected_loss = even_loss if ground_parity == 1 else odd_loss
        opposite_loss = odd_loss if ground_parity == 1 else even_loss
        maximum = float(self.lit_config.nqs_atomic_ground_parity_max_loss)
        logger.info(
            "Atomic ground parity diagnosis even_loss=%.6e odd_loss=%.6e "
            "selected=%s response=%s maximum=%.6e",
            even_loss,
            odd_loss,
            "even" if ground_parity == 1 else "odd",
            "odd" if ground_parity == 1 else "even",
            maximum,
        )
        if (
            not np.isfinite(even_loss)
            or not np.isfinite(odd_loss)
            or selected_loss > maximum
            or opposite_loss < 2.0 - maximum
        ):
            msg = (
                "Atomic ground checkpoint is not a clean inversion-parity "
                f"eigenstate: even_loss={even_loss:.6e}, "
                f"odd_loss={odd_loss:.6e}, required selected_loss <= "
                f"lit.nqs_atomic_ground_parity_max_loss={maximum:.6e} and "
                "opposite_loss >= 2 - maximum. Retrain or explicitly project "
                "the ground state before computing its dipole response."
            )
            raise RuntimeError(msg)
        return _AtomicParityResolution(
            ground_parity,
            -ground_parity,
            even_loss,
            odd_loss,
            selected_loss,
        )

    def _active_source_sector_operations(
        self,
        sector: SourceSector,
    ) -> tuple[jnp.ndarray, ...]:
        if _is_atom_hard_parity_sector(sector):
            return ()
        if self.lit_config.nqs_source_symmetry_weight <= 0.0:
            return ()
        return tuple(
            jnp.asarray(operation)
            for operation in sector.operations
            if not _is_identity_operation(operation)
        )

    def _source_guard_operations(
        self,
        sector: SourceSector,
    ) -> tuple[jnp.ndarray, ...]:
        """Return diagnostic source operations, independent of soft training."""
        return tuple(
            jnp.asarray(operation)
            for operation in sector.operations
            if not _is_identity_operation(operation)
        )

    def _validate_pure_source_covariance(
        self,
        ground_logpsi,
        ground_params,
        eval_pool: BatchedData,
        source_sector: SourceSector,
        source_centers,
        *,
        axis: int,
        response_parity: int = 0,
    ) -> _SourceCovarianceMetrics:
        """Check that the sampled dipole source belongs to its target sector.

        A response regularizer cannot repair covariance already violated by
        ``(D-D0) Psi0``.  Evaluate that pure source once on the fixed held-out
        pool before optimizing a response, using every configured operation.

        Returns:
            Mean, maximum, and worst-operation index for the held-out source
            covariance over active non-identity operations.

        Raises:
            ValueError: If ``source_centers`` is not a Cartesian vector.
            RuntimeError: If the covariance is non-finite or exceeds the
                configured maximum.
        """
        active_operations = self._source_guard_operations(source_sector)
        maximum = self.lit_config.nqs_source_symmetry_max_covariance
        if not active_operations or maximum is None:
            return _SourceCovarianceMetrics(
                mean_loss=jnp.asarray(0.0),
                max_loss=jnp.asarray(0.0),
                worst_operation_index=jnp.asarray(-1, dtype=jnp.int32),
            )

        evaluation_batch = _cyclic_batched_data_chunk(
            eval_pool,
            min(
                eval_pool.batch_size,
                int(self.lit_config.nqs_source_symmetry_eval_batch_size),
            ),
            0,
        )
        centers = jnp.asarray(
            source_centers,
            dtype=evaluation_batch.data.electrons.dtype,
        )
        if centers.shape != (3,):
            msg = f"source_centers must have shape (3,), got {centers.shape}."
            raise ValueError(msg)

        if _is_atom_hard_parity_sector(source_sector):
            return self._validate_atomic_source_parity(
                ground_logpsi,
                ground_params,
                evaluation_batch,
                source_sector,
                centers,
                active_operations,
                axis=axis,
                response_parity=response_parity,
                maximum=float(maximum),
            )

        operations = jnp.stack(active_operations)

        def pure_source_vector_apply(local_ground_params, data):
            ground_log_amplitude = ground_logpsi(local_ground_params, data)
            complex_dtype = jnp.result_type(ground_log_amplitude, jnp.complex64)
            centered_dipole = -jnp.sum(data.electrons, axis=0) - centers
            source_log_amplitude = jnp.log(jnp.abs(centered_dipole)) + 1j * jnp.where(
                centered_dipole < 0.0,
                jnp.asarray(jnp.pi, dtype=centered_dipole.dtype),
                jnp.asarray(0.0, dtype=centered_dipole.dtype),
            )
            return jnp.asarray(
                ground_log_amplitude,
                dtype=complex_dtype,
            ) + source_log_amplitude.astype(complex_dtype)

        @jax.jit
        def evaluate_covariance(local_ground_params, local_evaluation_batch):
            losses = jax.lax.map(
                lambda operation: _vector_covariance_penalty_loss(
                    pure_source_vector_apply,
                    local_ground_params,
                    local_evaluation_batch,
                    source_sector,
                    operation,
                ),
                operations,
            )
            return _summarize_source_covariance_losses(losses)

        metrics = jax.device_get(evaluate_covariance(ground_params, evaluation_batch))
        mean_loss = float(metrics.mean_loss)
        max_loss = float(metrics.max_loss)
        worst_operation_index = int(metrics.worst_operation_index)
        logger.info(
            "axis=%s pure_source_heldout_covariance_mean=%.6e "
            "pure_source_heldout_covariance_max=%.6e "
            "worst_active_operation=%d maximum=%s",
            _AXIS_NAMES[axis],
            mean_loss,
            max_loss,
            worst_operation_index,
            f"{float(maximum):.6e}",
        )
        if (
            not np.isfinite(mean_loss)
            or not np.isfinite(max_loss)
            or max_loss > float(maximum)
        ):
            msg = (
                f"axis={_AXIS_NAMES[axis]} pure-source held-out worst-operation "
                f"covariance {max_loss:.6e} (operation "
                f"{worst_operation_index}, mean {mean_loss:.6e}) exceeds "
                f"lit.nqs_source_symmetry_max_covariance={maximum!r}. "
                "The sampled (D-D0)Psi0 source is outside the configured "
                "sector; check ground-state symmetry, source centers, and "
                "source-pool equilibration before response optimization."
            )
            raise RuntimeError(msg)
        return metrics

    def _validate_atomic_source_parity(
        self,
        ground_logpsi,
        ground_params,
        evaluation_batch: BatchedData,
        source_sector: SourceSector,
        centers,
        active_operations: tuple[jnp.ndarray, ...],
        *,
        axis: int,
        response_parity: int,
        maximum: float,
    ) -> _SourceCovarianceMetrics:
        """Validate one atomic dipole source against its diagnosed parity.

        Returns:
            The scalar parity loss encoded as the mean and maximum covariance
            metric, with the inversion operation recorded as the worst index.

        Raises:
            ValueError: If the requested parity disagrees with the resolved
                atomic sector.
            RuntimeError: If the sector lacks a unique inversion operation or
                the held-out source parity is non-finite or above ``maximum``.
        """
        expected_parity = _response_parity_character(source_sector)
        if response_parity != expected_parity:
            msg = (
                "Atomic source validation requires resolved response parity "
                f"{expected_parity:+d}, got {response_parity!r}."
            )
            raise ValueError(msg)
        if len(active_operations) != 1 or not np.allclose(
            np.asarray(jax.device_get(active_operations[0])),
            -np.eye(3),
            rtol=0.0,
            atol=1e-10,
        ):
            msg = (
                "A resolved atomic source sector must contain exactly one "
                "non-identity inversion operation."
            )
            raise RuntimeError(msg)

        inversion = jnp.asarray(
            active_operations[0],
            dtype=evaluation_batch.data.electrons.dtype,
        )
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=evaluation_batch.data.electrons.dtype,
        )

        def pure_source_scalar_apply(local_ground_params, data):
            ground_log_amplitude = ground_logpsi(local_ground_params, data)
            complex_dtype = jnp.result_type(ground_log_amplitude, jnp.complex64)
            source_factor = jnp.asarray(
                molecular_electronic_dipole(data, axis) - centers[axis],
                dtype=complex_dtype,
            )
            return jnp.asarray(
                ground_log_amplitude,
                dtype=complex_dtype,
            ) + jnp.log(source_factor)

        @jax.jit
        def evaluate_parity(local_ground_params, local_evaluation_batch):
            def paired_logs(data):
                return (
                    pure_source_scalar_apply(local_ground_params, data),
                    pure_source_scalar_apply(
                        local_ground_params,
                        transform_molecule_data(data, inversion, symmetry_center),
                    ),
                )

            source_logs, inverted_source_logs = jax.vmap(
                paired_logs,
                in_axes=(local_evaluation_batch.vmap_axis,),
            )(local_evaluation_batch.data)
            return parity_log_amplitude_loss(
                source_logs,
                inverted_source_logs,
                response_parity,
            )

        loss_array = jax.device_get(evaluate_parity(ground_params, evaluation_batch))
        loss = float(loss_array)
        logger.info(
            "axis=%s pure_source_heldout_parity=%+.0f loss=%.6e maximum=%.6e",
            _AXIS_NAMES[axis],
            float(response_parity),
            loss,
            maximum,
        )
        if not np.isfinite(loss) or loss > maximum:
            msg = (
                f"axis={_AXIS_NAMES[axis]} pure-source held-out parity loss "
                f"{loss:.6e} exceeds "
                "lit.nqs_source_symmetry_max_covariance="
                f"{maximum:.6e} for response parity {response_parity:+d}. "
                "The sampled (D-D0)Psi0 source is outside the diagnosed "
                "atomic response sector; check the ground checkpoint, source "
                "center, and source-pool equilibration."
            )
            raise RuntimeError(msg)
        return _SourceCovarianceMetrics(
            mean_loss=jnp.asarray(loss_array),
            max_loss=jnp.asarray(loss_array),
            worst_operation_index=jnp.asarray(0, dtype=jnp.int32),
        )

    def _make_response_ansatz(  # noqa: C901
        self,
        example,
        response_rng,
        ground_params,
        *,
        axis: int,
        source_center: float,
        source_centers=None,
        source_sector: SourceSector | None = None,
        response_parity: int = 0,
        ground_logpsi=None,
        initialization_data: BatchedData | None = None,
        initial_omega: float | None = None,
    ):
        if source_sector is None:
            msg = (
                "A response symmetry policy resolved from the physical fixed "
                "geometry is required."
            )
            raise ValueError(msg)
        del source_centers
        hard_atomic_parity = _is_atom_hard_parity_sector(source_sector)
        if _is_atom_parity_sector(source_sector) and not hard_atomic_parity:
            msg = "Atomic response parity must be diagnosed before ansatz creation."
            raise ValueError(msg)
        if hard_atomic_parity:
            expected_parity = _response_parity_character(source_sector)
            if response_parity != expected_parity:
                msg = (
                    "Atomic response ansatz requires resolved parity "
                    f"{expected_parity:+d}, got {response_parity!r}."
                )
                raise ValueError(msg)
        elif response_parity != 0:
            msg = "A C1 response must not receive a hard parity character."
            raise ValueError(msg)
        response = MolecularResponseFermiNet(
            nspins=_two_spin_tuple(self.system_config.electron_spins),
            ndets=int(self.lit_config.nqs_response_ndets),
            hidden_dims_single=tuple(self.lit_config.nqs_response_hidden_dims_single),
            hidden_dims_double=tuple(self.lit_config.nqs_response_hidden_dims_double),
            use_last_layer=bool(self.lit_config.nqs_response_use_last_layer),
            envelope=self.lit_config.nqs_response_envelope,
            orbitals_spin_split=bool(self.lit_config.nqs_response_orbitals_spin_split),
        )
        raw_params = response.init(response_rng, example)
        raw_params = _copy_matching_parameters(raw_params, ground_params)

        inversion = jnp.asarray(-np.eye(3), dtype=example.electrons.dtype)
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=example.electrons.dtype,
        )

        def inverted_data(data):
            return transform_molecule_data(data, inversion, symmetry_center)

        def raw_apply(params, data):
            return response.apply(params, data)

        def projected_raw_apply(params, data):
            raw_logpsi = raw_apply(params, data)
            if not hard_atomic_parity:
                return raw_logpsi
            return parity_project_log_amplitude(
                raw_logpsi,
                raw_apply(params, inverted_data(data)),
                response_parity,
            )

        if not self.lit_config.nqs_source_aligned:
            return projected_raw_apply, None, raw_params

        if ground_logpsi is None:
            msg = "A ground log wavefunction is required for source-aligned response."
            raise ValueError(msg)
        center = jnp.asarray(source_center, dtype=example.electrons.dtype)
        if initial_omega is None:
            initial_omega = self.lit_config.nqs_warm_start_omega
        if initial_omega is None:
            initial_omega = float(self.lit_config.omega_min)
        coefficient = 1.0 / complex(-float(initial_omega), -self.lit_config.eta)
        real_dtype = example.electrons.dtype

        def source_logpsi(data):
            ground_log_amplitude = ground_logpsi(ground_params, data)
            complex_dtype = jnp.result_type(
                ground_log_amplitude,
                jnp.complex64,
            )
            source_factor = jnp.asarray(coefficient, dtype=complex_dtype) * (
                molecular_electronic_dipole(data, axis) - center
            )
            return jnp.asarray(
                ground_log_amplitude,
                dtype=complex_dtype,
            ) + jnp.log(source_factor)

        def projected_source_logpsi(data):
            source_log_amplitude = source_logpsi(data)
            if not hard_atomic_parity:
                return source_log_amplitude
            return parity_project_log_amplitude(
                source_log_amplitude,
                source_logpsi(inverted_data(data)),
                response_parity,
            )

        if initialization_data is None:
            raw_initial_logpsi = projected_raw_apply(raw_params, example)
            ground_initial_logpsi = ground_logpsi(ground_params, example)
            initial_dipole = molecular_electronic_dipole(example, axis)
            projected_source_initial_logpsi = projected_source_logpsi(example)
        else:
            raw_initial_logpsi = jax.vmap(
                lambda one: projected_raw_apply(raw_params, one),
                in_axes=(initialization_data.vmap_axis,),
            )(initialization_data.data)
            ground_initial_logpsi = jax.vmap(
                lambda one: ground_logpsi(ground_params, one),
                in_axes=(initialization_data.vmap_axis,),
            )(initialization_data.data)
            initial_dipole = jax.vmap(
                lambda one: molecular_electronic_dipole(one, axis),
                in_axes=(initialization_data.vmap_axis,),
            )(initialization_data.data)
            projected_source_initial_logpsi = jax.vmap(
                projected_source_logpsi,
                in_axes=(initialization_data.vmap_axis,),
            )(initialization_data.data)
        if hard_atomic_parity:
            residual_log_scale = _calibrated_residual_log_scale_from_logs(
                jnp.asarray(raw_initial_logpsi)[..., None],
                jnp.asarray(projected_source_initial_logpsi)[..., None],
                target_ratio=self.lit_config.nqs_source_aligned_residual_scale,
            )
        else:
            residual_log_scale = _calibrated_residual_log_scale(
                jnp.asarray(raw_initial_logpsi)[..., None],
                ground_initial_logpsi,
                initial_dipole[..., None],
                center[None],
                coefficient,
                target_ratio=self.lit_config.nqs_source_aligned_residual_scale,
            )
        logger.info(
            "Initialized source-aligned residual at relative scale %.3e "
            "(log_scale=%.6f)",
            self.lit_config.nqs_source_aligned_residual_scale,
            residual_log_scale,
        )
        response_params = freeze(
            {
                "raw": raw_params,
                "source_coefficient": jnp.asarray(
                    [coefficient.real, coefficient.imag],
                    dtype=real_dtype,
                ),
                "residual_log_scale": jnp.asarray(
                    residual_log_scale,
                    dtype=real_dtype,
                ),
            }
        )

        def unprojected_aligned_apply(params, data):
            coefficient_parts = params["source_coefficient"]
            source_coefficient = coefficient_parts[0] + 1j * coefficient_parts[1]
            raw_logpsi = raw_apply(params["raw"], data)
            dipole = molecular_electronic_dipole(data, axis)
            return source_aligned_vector_logpsi(
                raw_logpsi,
                ground_logpsi(ground_params, data),
                dipole,
                center,
                source_coefficient,
                params["residual_log_scale"],
            )

        def aligned_apply(params, data):
            aligned_logpsi = unprojected_aligned_apply(params, data)
            if not hard_atomic_parity:
                return aligned_logpsi
            return parity_project_log_amplitude(
                aligned_logpsi,
                unprojected_aligned_apply(params, inverted_data(data)),
                response_parity,
            )

        return aligned_apply, None, response_params

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
        source_covariance_evaluator=None,
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
        physical_loss = _regularized_loss(
            stats,
            self.lit_config.nqs_reverse_kl_weight,
        )
        if source_covariance_evaluator is None:
            return stats._replace(loss=physical_loss)
        covariance_metrics = _coerce_source_covariance_metrics(
            source_covariance_evaluator(response_params, eval_pool)
        )
        return stats._replace(
            loss=physical_loss
            + jnp.asarray(
                self.lit_config.nqs_source_symmetry_weight,
                dtype=physical_loss.dtype,
            )
            * covariance_metrics.mean_loss,
            source_covariance_loss=covariance_metrics.mean_loss,
            source_covariance_max_loss=covariance_metrics.max_loss,
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

        Raises:
            RuntimeError: If no held-out checkpoint satisfies the configured
                numerical, source-covariance, and stage-quality requirements.
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
                source_covariance_evaluator=getattr(
                    update_step,
                    "source_covariance_metrics",
                    getattr(update_step, "source_covariance_loss", None),
                ),
            )

        response_params = initial_params
        update_carry = update_step.init_carry(fallback_data, rng, response_params)
        maximum_covariance = self.lit_config.nqs_source_symmetry_max_covariance
        best_params = response_params
        best_stats = evaluate(response_params)
        initial_fidelity = float(jax.device_get(best_stats.fidelity))
        required_fidelity = _nqs_stage_required_fidelity(
            initial_fidelity,
            floor=self.lit_config.nqs_stage_fidelity_min,
            gain=self.lit_config.nqs_stage_fidelity_gain_min,
        )
        best_iteration = 0
        selected_params = None
        selected_stats = None
        selected_iteration = 0
        if (
            _is_eligible_nqs_checkpoint(
                best_stats,
                max_source_covariance=maximum_covariance,
            )
            and _nqs_stage_quality_failure(
                best_stats,
                min_fidelity=required_fidelity,
                min_reweight_ess_fraction=(
                    self.lit_config.nqs_stage_reweight_ess_fraction_min
                ),
            )
            is None
        ):
            selected_params = best_params
            selected_stats = best_stats
        last_train_stats = None
        last_optimizer_diagnostics = None
        for iteration in range(max(0, int(iterations))):
            response_params, last_train_stats, update_carry = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(omega)),
                update_carry,
                iteration,
            )
            last_optimizer_diagnostics = getattr(
                update_step,
                "last_spring_optimizer_diagnostics",
                None,
            )
            completed = iteration + 1
            should_select = (
                completed % self.lit_config.nqs_selection_interval == 0
                or completed == int(iterations)
            )
            if should_select:
                candidate_stats = evaluate(response_params)
                if _is_better_nqs_checkpoint(
                    candidate_stats,
                    best_stats,
                    max_source_covariance=(
                        self.lit_config.nqs_source_symmetry_max_covariance
                    ),
                ):
                    best_params = response_params
                    best_stats = candidate_stats
                    best_iteration = completed
                if (
                    _is_eligible_nqs_checkpoint(
                        candidate_stats,
                        max_source_covariance=maximum_covariance,
                    )
                    and _nqs_stage_quality_failure(
                        candidate_stats,
                        min_fidelity=required_fidelity,
                        min_reweight_ess_fraction=(
                            self.lit_config.nqs_stage_reweight_ess_fraction_min
                        ),
                    )
                    is None
                    and (
                        selected_stats is None
                        or _is_better_nqs_checkpoint(
                            candidate_stats,
                            selected_stats,
                            max_source_covariance=maximum_covariance,
                        )
                    )
                ):
                    selected_params = response_params
                    selected_stats = candidate_stats
                    selected_iteration = completed
            if (
                self.lit_config.nqs_log_interval > 0
                and completed % self.lit_config.nqs_log_interval == 0
            ):
                reported_stats = (
                    selected_stats if selected_stats is not None else best_stats
                )
                reported_iteration = (
                    selected_iteration if selected_stats is not None else best_iteration
                )
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d train_loss=%.6e "
                    "train_fidelity=%.6f train_reverse_kl=%.6e "
                    "train_covariance_operation=%.6e best_iter=%d "
                    "best_fidelity=%.6f best_reverse_kl=%.6e "
                    "best_covariance_mean=%.6e best_covariance_max=%.6e",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(last_train_stats.loss),
                    float(last_train_stats.fidelity),
                    float(last_train_stats.reverse_kl),
                    float(getattr(last_train_stats, "source_covariance_loss", 0.0)),
                    reported_iteration,
                    float(reported_stats.fidelity),
                    float(reported_stats.reverse_kl),
                    float(
                        getattr(
                            reported_stats,
                            "source_covariance_loss",
                            0.0,
                        )
                    ),
                    float(
                        getattr(
                            reported_stats,
                            "source_covariance_max_loss",
                            getattr(
                                reported_stats,
                                "source_covariance_loss",
                                0.0,
                            ),
                        )
                    ),
                )
                _log_spring_optimizer_diagnostics(
                    last_optimizer_diagnostics,
                    axis=axis,
                    stage=stage,
                    omega=float(omega),
                    iteration=completed,
                )
        best_covariance_mean, best_covariance_max = _source_covariance_host_values(
            best_stats
        )
        if not _is_eligible_nqs_checkpoint(
            best_stats,
            max_source_covariance=maximum_covariance,
        ):
            msg = (
                f"axis={_AXIS_NAMES[axis]} stage={stage} omega={float(omega):.6f} "
                "produced no eligible held-out checkpoint; best covariance "
                f"mean={best_covariance_mean:.6e}, max={best_covariance_max:.6e}, "
                f"maximum={maximum_covariance!r}."
            )
            raise RuntimeError(msg)
        if selected_stats is None or selected_params is None:
            _require_nqs_stage_quality(
                best_stats,
                min_fidelity=required_fidelity,
                min_reweight_ess_fraction=(
                    self.lit_config.nqs_stage_reweight_ess_fraction_min
                ),
                context=(
                    f"axis={_AXIS_NAMES[axis]} stage={stage} "
                    f"omega={float(omega):.6f} failed its held-out quality gate "
                    f"(initial fidelity={initial_fidelity:.6f}, "
                    "configured floor="
                    f"{self.lit_config.nqs_stage_fidelity_min:.6f}, "
                    "required gain="
                    f"{self.lit_config.nqs_stage_fidelity_gain_min:.6f})"
                ),
            )
            msg = (
                f"axis={_AXIS_NAMES[axis]} stage={stage} "
                f"omega={float(omega):.6f} produced no selectable checkpoint."
            )
            raise RuntimeError(msg)
        best_params = selected_params
        best_stats = selected_stats
        best_iteration = selected_iteration
        _require_nqs_stage_quality(
            best_stats,
            min_fidelity=required_fidelity,
            min_reweight_ess_fraction=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            context=(
                f"axis={_AXIS_NAMES[axis]} stage={stage} "
                f"omega={float(omega):.6f} failed its held-out quality gate "
                f"(initial fidelity={initial_fidelity:.6f}, "
                f"configured floor={self.lit_config.nqs_stage_fidelity_min:.6f}, "
                f"required gain={self.lit_config.nqs_stage_fidelity_gain_min:.6f})"
            ),
        )
        rng = update_carry.direct.rng
        logger.info(
            "axis=%s stage=%s omega=%.6f selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e "
            "covariance_mean=%.6e covariance_max=%.6e ess=%.3f "
            "required_fidelity=%.6f required_ess=%.3f",
            _AXIS_NAMES[axis],
            stage,
            float(omega),
            best_iteration,
            max(0, int(iterations)),
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(getattr(best_stats, "source_covariance_loss", 0.0)),
            float(
                getattr(
                    best_stats,
                    "source_covariance_max_loss",
                    getattr(best_stats, "source_covariance_loss", 0.0),
                )
            ),
            float(best_stats.reweight_ess_fraction),
            required_fidelity,
            self.lit_config.nqs_stage_reweight_ess_fraction_min,
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
        source_covariance_evaluator = getattr(
            update_step,
            "source_covariance_metrics",
            getattr(update_step, "source_covariance_loss", None),
        )
        current_omega = float(start_omega)
        if current_stats is None:
            current_stats = self._evaluate_nqs_checkpoint(
                response_params=response_params,
                eval_pool=eval_pool,
                omega=current_omega,
                source_covariance_evaluator=source_covariance_evaluator,
                **common,
            )
        maximum_covariance = self.lit_config.nqs_source_symmetry_max_covariance
        _require_eligible_nqs_checkpoint(
            current_stats,
            max_source_covariance=maximum_covariance,
            context=(
                "Frequency continuation received an ineligible starting "
                f"checkpoint at omega={current_omega:.8g}"
            ),
        )
        _require_nqs_stage_quality(
            current_stats,
            min_fidelity=self.lit_config.nqs_stage_fidelity_min,
            min_reweight_ess_fraction=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            context=(
                "Frequency continuation received a starting checkpoint below "
                f"the absolute quality floor at omega={current_omega:.8g}"
            ),
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
                    source_covariance_evaluator=source_covariance_evaluator,
                    **common,
                )
                probe_ok = _continuation_probe_is_acceptable(
                    current_stats,
                    probe_stats,
                    retention=self.lit_config.nqs_continuation_fidelity_retention,
                    max_source_covariance=maximum_covariance,
                    min_fidelity=self.lit_config.nqs_stage_fidelity_min,
                    min_reweight_ess_fraction=(
                        self.lit_config.nqs_stage_reweight_ess_fraction_min
                    ),
                )
                candidate_gap = candidate_omega - current_omega
                if probe_ok or candidate_gap <= min_step * (1.0 + 1e-12):
                    break
                candidate_gap = max(min_step, 0.5 * candidate_gap)
                candidate_omega = min(target_omega, current_omega + candidate_gap)
                bisections += 1

            _require_eligible_nqs_checkpoint(
                probe_stats,
                max_source_covariance=maximum_covariance,
                context=(
                    "Frequency continuation produced non-finite/invalid held-out "
                    f"statistics at omega={candidate_omega:.8g}; refusing to "
                    "propagate a corrupted checkpoint"
                ),
            )
            actual_step = float(candidate_omega - current_omega)
            min_step_override = not probe_ok and actual_step <= min_step * (1.0 + 1e-12)
            if (
                min_step_override
                and not self.lit_config.nqs_continuation_allow_min_step_override
            ):
                current_fidelity = float(jax.device_get(current_stats.fidelity))
                candidate_fidelity = float(jax.device_get(probe_stats.fidelity))
                candidate_ess = float(jax.device_get(probe_stats.reweight_ess_fraction))
                required_probe_fidelity = max(
                    self.lit_config.nqs_stage_fidelity_min,
                    self.lit_config.nqs_continuation_fidelity_retention
                    * current_fidelity,
                )
                msg = (
                    "Frequency continuation reached its minimum step without "
                    "an acceptable inherited checkpoint; refusing the legacy "
                    f"override at omega={candidate_omega:.8g}: fidelity="
                    f"{candidate_fidelity:.6f}, required="
                    f"{required_probe_fidelity:.6f}, ESS fraction="
                    f"{candidate_ess:.6f}, required ESS="
                    f"{self.lit_config.nqs_stage_reweight_ess_fraction_min:.6f}."
                )
                raise RuntimeError(msg)
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
                    "covariance_mean=%.6e covariance_max=%.6e step=%.6e "
                    "bisections=%d accepted=%s min_step_override=%s",
                    _AXIS_NAMES[axis],
                    target_omega,
                    float(probe_stats.fidelity),
                    float(getattr(probe_stats, "source_covariance_loss", 0.0)),
                    float(
                        getattr(
                            probe_stats,
                            "source_covariance_max_loss",
                            getattr(probe_stats, "source_covariance_loss", 0.0),
                        )
                    ),
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
            _require_eligible_nqs_checkpoint(
                current_stats,
                max_source_covariance=maximum_covariance,
                context=(
                    "Frequency continuation failed to obtain a finite held-out "
                    f"checkpoint at omega={candidate_omega:.8g}"
                ),
            )
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
                "selected_fidelity=%.6f covariance_mean=%.6e "
                "covariance_max=%.6e step=%.6e bisections=%d accepted=%s "
                "min_step_override=%s",
                _AXIS_NAMES[axis],
                candidate_omega,
                inherited_fidelity,
                float(current_stats.fidelity),
                float(getattr(current_stats, "source_covariance_loss", 0.0)),
                float(
                    getattr(
                        current_stats,
                        "source_covariance_max_loss",
                        getattr(current_stats, "source_covariance_loss", 0.0),
                    )
                ),
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
                "selected_iter=%d fidelity=%.6f reverse_kl=%.6e "
                "covariance_mean=%.6e covariance_max=%.6e",
                _AXIS_NAMES[kwargs["axis"]],
                float(self.lit_config.nqs_warm_start_omega),
                self.lit_config.nqs_warm_start_iterations,
                result[2],
                float(result[1].fidelity),
                float(result[1].reverse_kl),
                float(getattr(result[1], "source_covariance_loss", 0.0)),
                float(
                    getattr(
                        result[1],
                        "source_covariance_max_loss",
                        getattr(result[1], "source_covariance_loss", 0.0),
                    )
                ),
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

    def _estimate_vector_source_stats(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        source_sector: SourceSector | None = None,
    ):
        center_override = _three_component_override(
            self.lit_config.nqs_source_center_override,
            name="lit.nqs_source_center_override",
        )
        norm_override = _three_component_override(
            self.lit_config.nqs_source_norm_override,
            name="lit.nqs_source_norm_override",
            positive=True,
        )
        electron_count = int(batched_data.data.electrons.shape[-2])
        if center_override is not None and norm_override is not None:
            center = _project_source_center_to_invariant_subspace(
                center_override,
                source_sector,
                electron_count=electron_count,
                tolerance=float(self.lit_config.nqs_source_symmetry_tolerance),
            )
            _log_source_center_projection(
                center_override,
                center,
                source_sector,
            )
            return (
                center,
                norm_override,
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
                lambda one: -jnp.sum(one.electrons, axis=0),
                in_axes=(batched_data.vmap_axis,),
            )(batched_data.data)
            mean_values.append(np.asarray(jnp.mean(dipole, axis=0)))
            mean_square_values.append(np.asarray(jnp.mean(dipole**2, axis=0)))
        mean = np.mean(mean_values, axis=0, dtype=np.float64)
        center = np.array(mean, copy=True)
        if center_override is not None:
            center = center_override
        unprojected_center = np.array(center, copy=True)
        center = _project_source_center_to_invariant_subspace(
            center,
            source_sector,
            electron_count=electron_count,
            tolerance=float(self.lit_config.nqs_source_symmetry_tolerance),
        )
        _log_source_center_projection(
            unprojected_center,
            center,
            source_sector,
        )
        variance = (
            np.mean(mean_square_values, axis=0, dtype=np.float64)
            - 2.0 * center * mean
            + center**2
        )
        norm = np.maximum(variance, 1e-12)
        if norm_override is not None:
            norm = norm_override
        return center, norm, batched_data, sampler_state, rng

    def _estimate_source_stats(
        self,
        ground_params,
        batched_data,
        sampler_state,
        sample_plan: SamplePlan,
        rng,
        *,
        axis: int,
        source_sector: SourceSector | None = None,
    ):
        """Return one component while preserving the previous private API.

        Returns:
            The selected center and norm, updated sampler data/state, and RNG.
        """
        centers, norms, batched_data, sampler_state, rng = (
            self._estimate_vector_source_stats(
                ground_params,
                batched_data,
                sampler_state,
                sample_plan,
                rng,
                source_sector=source_sector,
            )
        )
        return (
            float(centers[int(axis)]),
            float(norms[int(axis)]),
            batched_data,
            sampler_state,
            rng,
        )

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
        response_vector_apply=None,
        source_sector: SourceSector | None = None,
        axis: int,
        source_center: float,
        source_norm: float,
    ):
        active_operations = (
            self._active_source_sector_operations(source_sector)
            if source_sector is not None and response_vector_apply is not None
            else ()
        )
        identity_operation = jnp.eye(3)

        if active_operations:
            evaluation_operations = jnp.stack(active_operations)

            @jax.jit
            def evaluate_covariance_batch(response_params, evaluation_batch):
                losses = jax.lax.map(
                    lambda operation: _vector_covariance_penalty_loss(
                        response_vector_apply,
                        response_params,
                        evaluation_batch,
                        source_sector,
                        operation,
                    ),
                    evaluation_operations,
                )
                return _summarize_source_covariance_losses(losses)

            def evaluate_source_covariance_metrics(response_params, evaluation_pool):
                evaluation_batch = _cyclic_batched_data_chunk(
                    evaluation_pool,
                    min(
                        evaluation_pool.batch_size,
                        int(self.lit_config.nqs_source_symmetry_eval_batch_size),
                    ),
                    0,
                )
                return evaluate_covariance_batch(response_params, evaluation_batch)

        else:

            def evaluate_source_covariance_metrics(response_params, evaluation_pool):
                del evaluation_pool
                first_leaf = jax.tree_util.tree_leaves(response_params)[0]
                zero = jnp.asarray(0.0, dtype=first_leaf.dtype)
                return _SourceCovarianceMetrics(
                    mean_loss=zero,
                    max_loss=zero,
                    worst_operation_index=jnp.asarray(-1, dtype=jnp.int32),
                )

        def evaluate_source_covariance(response_params, evaluation_pool):
            return evaluate_source_covariance_metrics(
                response_params,
                evaluation_pool,
            ).mean_loss

        @jax.jit
        def source_update(
            response_params,
            batched_data,
            spring_previous,
            omega,
            source_operation,
        ):
            (
                stats,
                updates,
                spring_state,
                _,
                covariance_loss,
                optimizer_diagnostics,
            ) = self._source_sr_stats_and_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                batched_data,
                response_vector_apply=response_vector_apply,
                source_sector=source_sector,
                source_operation=(source_operation if active_operations else None),
                spring_state=_SpringState(spring_previous),
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=omega,
            )
            response_params = _apply_updates(response_params, updates)
            loss = (
                _regularized_loss(
                    stats,
                    self.lit_config.nqs_reverse_kl_weight,
                )
                + jnp.asarray(
                    self.lit_config.nqs_source_symmetry_weight,
                    dtype=stats.loss.dtype,
                )
                * covariance_loss
            )
            return (
                response_params,
                stats._replace(
                    loss=loss,
                    source_covariance_loss=covariance_loss,
                ),
                spring_state.previous_direction,
                optimizer_diagnostics,
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
            source_operation,
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
                (
                    source_updates,
                    next_spring_state,
                    _,
                    covariance_loss,
                    optimizer_diagnostics,
                ) = self._weighted_sr_updates(
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
                    response_vector_apply=response_vector_apply,
                    source_sector=source_sector,
                    source_operation=(source_operation if active_operations else None),
                )
                source_response_params = _apply_updates(response_params, source_updates)
                regularized_source_stats = source_stats._replace(
                    loss=source_stats.loss
                    + jnp.asarray(
                        self.lit_config.nqs_source_symmetry_weight,
                        dtype=source_stats.loss.dtype,
                    )
                    * covariance_loss,
                    source_covariance_loss=covariance_loss,
                )
                return (
                    source_response_params,
                    regularized_source_stats,
                    direct_batched_data,
                    direct_sampler_state,
                    direct_rng,
                    jnp.asarray(False),
                    jnp.asarray(False),
                    next_spring_state.previous_direction,
                    optimizer_diagnostics,
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
                direct_stats, direct_updates, next_spring_state, _ = (
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
                covariance_loss, covariance_updates = (
                    self._source_sector_penalty_updates(
                        response_vector_apply,
                        response_params,
                        source_batched_data,
                        source_sector,
                        source_operation if active_operations else None,
                    )
                )
                if covariance_updates is not None:
                    direct_updates = jax.tree.map(
                        operator.add,
                        direct_updates,
                        covariance_updates,
                    )
                direct_loss = (
                    _regularized_loss(
                        direct_stats,
                        self.lit_config.nqs_reverse_kl_weight,
                    )
                    + jnp.asarray(
                        self.lit_config.nqs_source_symmetry_weight,
                        dtype=direct_stats.loss.dtype,
                    )
                    * covariance_loss
                )
                direct_response_params = _apply_updates(
                    response_params,
                    direct_updates,
                )
                return (
                    direct_response_params,
                    direct_stats._replace(
                        loss=direct_loss,
                        source_covariance_loss=covariance_loss,
                    ),
                    next_batched_data,
                    next_sampler_state,
                    next_rng,
                    jnp.asarray(True),
                    jnp.asarray(True),
                    next_spring_state.previous_direction,
                    _empty_spring_optimizer_diagnostics(response_params),
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
                source_operation = (
                    active_operations[int(batch_index) % len(active_operations)]
                    if active_operations
                    else identity_operation
                )
                (
                    response_params,
                    stats,
                    spring_previous,
                    optimizer_diagnostics,
                ) = source_update(
                    response_params,
                    update_batch,
                    update_carry.spring.previous_direction,
                    omega,
                    source_operation,
                )
                update.last_spring_optimizer_diagnostics = (  # type: ignore[attr-defined]
                    optimizer_diagnostics
                )
                return (
                    response_params,
                    stats,
                    update_carry._replace(spring=_SpringState(spring_previous)),
                )
            source_operation = (
                active_operations[int(batch_index) % len(active_operations)]
                if active_operations
                else identity_operation
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
                optimizer_diagnostics,
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
                source_operation,
            )
            update.last_spring_optimizer_diagnostics = (  # type: ignore[attr-defined]
                optimizer_diagnostics
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
                source_operation=(
                    active_operations[0] if active_operations else identity_operation
                ),
            )

        update.init_carry = self._init_nqs_update_carry  # type: ignore[attr-defined]
        update.last_spring_optimizer_diagnostics = None  # type: ignore[attr-defined]
        update.precompile_direct = precompile_direct  # type: ignore[attr-defined]
        update.source_covariance_loss = (  # type: ignore[attr-defined]
            evaluate_source_covariance
        )
        update.source_covariance_metrics = (  # type: ignore[attr-defined]
            evaluate_source_covariance_metrics
        )
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
        source_operation,
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
            source_operation,
        ).compile()
        logger.info(
            "Precompiled fused direct pi_Psi fallback step for axis=%s",
            _AXIS_NAMES[axis],
        )
        return rng

    def _source_sector_penalty_updates(
        self,
        response_vector_apply,
        response_params,
        batched_data,
        source_sector: SourceSector | None,
        source_operation,
    ):
        """Return a covariance loss and an independently clipped SGD update.

        The physical SPRING metric is built from only the selected Cartesian
        response component.  The other vector heads therefore lie in its exact
        null space.  Applying their covariance gradients through that metric
        would amplify them by the tiny damping and consume the global SR norm
        clip, so symmetry uses its own Euclidean step.
        """
        if (
            response_vector_apply is None
            or source_sector is None
            or source_operation is None
            or self.lit_config.nqs_source_symmetry_weight <= 0.0
        ):
            first_leaf = jax.tree_util.tree_leaves(response_params)[0]
            return jnp.asarray(0.0, dtype=first_leaf.dtype), None
        loss, flat_gradient = _vector_covariance_penalty_gradient(
            response_vector_apply,
            response_params,
            batched_data,
            source_sector,
            source_operation,
        )
        finite_penalty = jnp.isfinite(loss) & jnp.all(jnp.isfinite(flat_gradient))
        flat_gradient = jnp.where(
            finite_penalty,
            flat_gradient,
            jnp.zeros_like(flat_gradient),
        )
        updates = _symmetry_gradient_updates(
            response_params,
            flat_gradient,
            weight=self.lit_config.nqs_source_symmetry_weight,
            learning_rate=self.lit_config.nqs_source_symmetry_learning_rate,
            max_norm=self.lit_config.nqs_source_symmetry_max_norm,
        )
        return loss, updates

    def _source_sr_stats_and_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        response_vector_apply=None,
        source_sector: SourceSector | None = None,
        source_operation=None,
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
        covariance_loss, covariance_updates = self._source_sector_penalty_updates(
            response_vector_apply,
            response_params,
            batched_data,
            source_sector,
            source_operation,
        )
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
            )
        )
        if covariance_updates is not None:
            updates = jax.tree.map(operator.add, updates, covariance_updates)
        return (
            stats,
            updates,
            spring_state,
            damping,
            covariance_loss,
            optimizer_diagnostics,
        )

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
            fidelity_gradient,
            reverse_kl_gradient,
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
        qfi_trace = jnp.sum(score_aug**2)
        centering_null = jnp.sqrt(psi_weight)
        zero_null = jnp.zeros_like(centering_null)
        kernel_null_vectors = jnp.stack(
            [
                jnp.concatenate([centering_null, zero_null]),
                jnp.concatenate([zero_null, centering_null]),
            ]
        )
        previous_direction = spring_state.previous_direction
        direction, spring_state, damping = _spring_direction_chunked(
            (score_aug.shape[0],),
            lambda _: score_aug,
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
        optimizer_diagnostics = _spring_optimizer_diagnostics(
            response_params,
            grad_flat,
            fidelity_gradient,
            reverse_kl_gradient,
            direction,
            updates,
            previous_direction,
            reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
            learning_rate=self.lit_config.nqs_learning_rate,
            max_norm=self.lit_config.nqs_sr_max_norm,
            damping=damping,
            decay=self.lit_config.nqs_spring_decay,
            qfi_trace=qfi_trace,
        )
        return updates, spring_state, damping, optimizer_diagnostics

    def _weighted_sr_updates(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        response_vector_apply=None,
        source_sector: SourceSector | None = None,
        source_operation=None,
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
        covariance_loss, covariance_updates = self._source_sector_penalty_updates(
            response_vector_apply,
            response_params,
            batched_data,
            source_sector,
            source_operation,
        )
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
            )
        )
        if covariance_updates is not None:
            updates = jax.tree.map(operator.add, updates, covariance_updates)
        return updates, spring_state, damping, covariance_loss, optimizer_diagnostics

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


def _log_spring_optimizer_diagnostics(
    diagnostics: _SpringOptimizerDiagnostics | None,
    *,
    axis: int,
    stage: str,
    omega: float,
    iteration: int,
) -> None:
    """Log the most recent source-sampled SPRING scalar diagnostics."""
    if diagnostics is None:
        return
    host = jax.device_get(diagnostics)
    if not bool(host.available):
        return
    logger.info(
        "axis=%s stage=%s omega=%.6f iter=%d spring_grad=%.6e "
        "spring_grad_fidelity=%.6e spring_grad_kl_weighted=%.6e "
        "spring_fidelity_kl_cosine=%.6f spring_gradient_cancellation=%.6e "
        "spring_direction=%.6e spring_update=%.6e spring_clip_factor=%.6e "
        "spring_clipped=%d spring_damping=%.6e spring_qfi_mean_diagonal=%.6e "
        "spring_history_gradient_ratio=%.6e raw_grad_rms=%.6e "
        "raw_update=%.6e source_coefficient_grad_rms=%.6e "
        "source_coefficient_update=%.6e residual_log_scale_grad_rms=%.6e "
        "residual_log_scale_update=%.6e",
        _AXIS_NAMES[axis],
        stage,
        omega,
        iteration,
        float(host.combined_gradient_norm),
        float(host.fidelity_gradient_norm),
        float(host.weighted_reverse_kl_gradient_norm),
        float(host.fidelity_kl_cosine),
        float(host.gradient_cancellation_ratio),
        float(host.direction_norm),
        float(host.update_norm),
        float(host.clip_factor),
        int(host.clip_factor < 1.0),
        float(host.damping),
        float(host.mean_metric_diagonal),
        float(host.history_gradient_ratio),
        float(host.parameter_group_gradient_rms[0]),
        float(host.parameter_group_update_norm[0]),
        float(host.parameter_group_gradient_rms[1]),
        float(host.parameter_group_update_norm[1]),
        float(host.parameter_group_gradient_rms[2]),
        float(host.parameter_group_update_norm[2]),
    )


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


def _finite_valid_nqs_stats(
    stats,
    *,
    max_source_covariance: float | None = None,
) -> bool:
    return _is_eligible_nqs_checkpoint(
        stats,
        max_source_covariance=max_source_covariance,
    )


def _nqs_stage_required_fidelity(
    initial_fidelity: float,
    *,
    floor: float,
    gain: float,
) -> float:
    """Return the absolute fidelity required at the end of one NQS stage.

    A checkpoint that already meets the configured floor may be propagated
    unchanged.  Otherwise, the stage must both reach the floor and make the
    requested minimum recovery from its initial held-out fidelity.
    """
    floor = float(floor)
    gain = float(gain)
    initial_fidelity = float(initial_fidelity)
    if not np.isfinite(initial_fidelity):
        return floor
    if initial_fidelity >= floor:
        return floor
    return max(floor, initial_fidelity + gain)


def _nqs_stage_quality_failure(
    stats,
    *,
    min_fidelity: float,
    min_reweight_ess_fraction: float,
) -> str | None:
    """Describe an absolute held-out quality-gate failure, if any.

    Returns:
        A human-readable failure description, or ``None`` when both active
        thresholds pass.
    """
    fidelity = float(jax.device_get(stats.fidelity))
    ess_fraction = float(jax.device_get(stats.reweight_ess_fraction))
    failures = []
    if min_fidelity > 0.0 and (
        not np.isfinite(fidelity) or fidelity < float(min_fidelity)
    ):
        failures.append(f"fidelity={fidelity:.6f} < required={float(min_fidelity):.6f}")
    if min_reweight_ess_fraction > 0.0 and (
        not np.isfinite(ess_fraction) or ess_fraction < float(min_reweight_ess_fraction)
    ):
        failures.append(
            "ESS fraction="
            f"{ess_fraction:.6f} < required="
            f"{float(min_reweight_ess_fraction):.6f}"
        )
    return "; ".join(failures) if failures else None


def _require_nqs_stage_quality(
    stats,
    *,
    min_fidelity: float,
    min_reweight_ess_fraction: float,
    context: str,
) -> None:
    """Raise when a finite checkpoint misses an absolute science gate.

    Raises:
        RuntimeError: If an active fidelity or ESS threshold is missed.
    """
    failure = _nqs_stage_quality_failure(
        stats,
        min_fidelity=min_fidelity,
        min_reweight_ess_fraction=min_reweight_ess_fraction,
    )
    if failure is not None:
        raise RuntimeError(f"{context}; {failure}.")


def _continuation_probe_is_acceptable(
    current,
    candidate,
    *,
    retention: float,
    max_source_covariance: float | None = None,
    min_fidelity: float = 0.0,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    if not _finite_valid_nqs_stats(
        candidate,
        max_source_covariance=max_source_covariance,
    ):
        return False
    if (
        _nqs_stage_quality_failure(
            candidate,
            min_fidelity=min_fidelity,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        is not None
    ):
        return False
    current_fidelity = float(jax.device_get(current.fidelity))
    candidate_fidelity = float(jax.device_get(candidate.fidelity))
    if not np.isfinite(current_fidelity):
        return False
    required = max(0.0, float(retention) * current_fidelity)
    return candidate_fidelity >= required


def _three_component_override(
    value: float | tuple[float, float, float] | None,
    *,
    name: str,
    positive: bool = False,
) -> np.ndarray | None:
    """Normalize a scalar or Cartesian override to a length-three host array.

    Returns:
        ``None`` when unset, otherwise a finite float64 array of shape ``(3,)``.

    Raises:
        ValueError: If the override has the wrong shape or invalid values.
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(3, float(array), dtype=np.float64)
    elif array.shape == (3,):
        array = np.array(array, dtype=np.float64, copy=True)
    else:
        msg = f"{name} must be a scalar or length-three Cartesian vector."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    if positive and np.any(array <= 0.0):
        msg = f"{name} must contain only positive values."
        raise ValueError(msg)
    return array


def _calibrated_residual_log_scale_from_logs(
    raw_logpsi,
    source_logpsi,
    *,
    target_ratio: float,
) -> float:
    """Calibrate a raw residual from componentwise source/raw log amplitudes.

    FermiNet determinant heads have an arbitrary absolute scale, so multiplying
    the raw response by a nominal ``1e-4`` does not imply a ``1e-4`` residual.
    The median log-norm ratio on a small ground-state batch fixes that gauge.

    Returns:
        The additive raw-response log scale that realizes ``target_ratio``.

    Raises:
        ValueError: If no finite, nonzero source/raw norm pair is available.
    """
    raw_logs = np.asarray(jax.device_get(raw_logpsi))
    source_logs = np.asarray(jax.device_get(source_logpsi))
    if raw_logs.shape != source_logs.shape or raw_logs.ndim < 1:
        msg = (
            "raw_logpsi and source_logpsi must have the same componentwise "
            f"shape, got {raw_logs.shape} and {source_logs.shape}."
        )
        raise ValueError(msg)

    def vector_log_norm(component_log_magnitudes: np.ndarray) -> np.ndarray:
        finite = np.isfinite(component_log_magnitudes)
        invalid = ~(finite | np.isneginf(component_log_magnitudes))
        masked = np.where(finite, component_log_magnitudes, -np.inf)
        maximum = np.max(masked, axis=-1)
        safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
        scaled_norm_sq = np.sum(
            np.where(
                finite,
                np.exp(2.0 * (masked - safe_maximum[..., None])),
                0.0,
            ),
            axis=-1,
        )
        result = safe_maximum + 0.5 * np.log(scaled_norm_sq)
        return np.where(np.any(invalid, axis=-1), np.nan, result)

    with np.errstate(divide="ignore", invalid="ignore"):
        source_log_norm = vector_log_norm(np.real(source_logs))
        raw_log_norm = vector_log_norm(np.real(raw_logs))
        offsets = source_log_norm - raw_log_norm
    finite_offsets = offsets[np.isfinite(offsets)]
    if finite_offsets.size == 0:
        msg = (
            "Cannot calibrate the source-aligned residual because no finite, "
            "nonzero source/raw vector norm pair was found."
        )
        raise ValueError(msg)
    return float(np.log(float(target_ratio)) + np.median(finite_offsets))


def _calibrated_residual_log_scale(
    raw_logpsi,
    ground_logpsi,
    dipole,
    source_center,
    source_coefficient,
    *,
    target_ratio: float,
) -> float:
    """Calibrate a raw residual against an unprojected dipole source.

    Returns:
        The additive raw-response log scale that realizes ``target_ratio``.
    """
    ground_logs = np.asarray(jax.device_get(ground_logpsi))
    dipoles = np.asarray(jax.device_get(dipole))
    centers = np.asarray(jax.device_get(source_center))
    coefficient = complex(np.asarray(jax.device_get(source_coefficient)))
    source_factor = coefficient * (dipoles - centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        source_logs = np.asarray(ground_logs)[..., None] + np.log(source_factor)
    return _calibrated_residual_log_scale_from_logs(
        raw_logpsi,
        source_logs,
        target_ratio=target_ratio,
    )


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


def _is_identity_operation(operation, *, tolerance: float = 1e-10) -> bool:
    return bool(
        np.allclose(
            np.asarray(operation),
            np.eye(3),
            rtol=0.0,
            atol=tolerance,
        )
    )


def _is_atom_parity_sector(sector: SourceSector) -> bool:
    """Return whether a sector denotes pending or resolved atomic parity."""
    if (
        sector.label
        not in {
            _ATOM_PARITY_PENDING_SECTOR_LABEL,
            _ATOM_HARD_ODD_SECTOR_LABEL,
            _ATOM_HARD_EVEN_SECTOR_LABEL,
        }
        or sector.order != 2
    ):
        return False
    return any(
        np.allclose(
            np.asarray(operation),
            -np.eye(3),
            rtol=0.0,
            atol=1e-10,
        )
        for operation in sector.operations
    )


def _is_atom_hard_parity_sector(sector: SourceSector) -> bool:
    """Return whether an atomic sector has a diagnosed response parity."""
    return _is_atom_parity_sector(sector) and sector.label in {
        _ATOM_HARD_ODD_SECTOR_LABEL,
        _ATOM_HARD_EVEN_SECTOR_LABEL,
    }


def _response_parity_character(sector: SourceSector) -> int:
    """Return +1 for hard even, -1 for hard odd, or zero if unresolved/C1."""
    if sector.label == _ATOM_HARD_EVEN_SECTOR_LABEL:
        return 1
    if sector.label == _ATOM_HARD_ODD_SECTOR_LABEL:
        return -1
    return 0


def _resolve_atomic_parity_sector(
    sector: SourceSector,
    response_parity: int,
) -> SourceSector:
    """Attach one diagnosed response parity to a pending atomic sector.

    Returns:
        The resolved hard-parity atomic sector, or the unchanged C1 sector.

    Raises:
        ValueError: If the supplied character is invalid for the sector.
    """
    if sector.label == _ATOM_PARITY_PENDING_SECTOR_LABEL:
        if response_parity not in (-1, 1):
            msg = (
                "A pending atomic sector requires response parity +1 or -1, "
                f"got {response_parity!r}."
            )
            raise ValueError(msg)
        label = (
            _ATOM_HARD_EVEN_SECTOR_LABEL
            if response_parity == 1
            else _ATOM_HARD_ODD_SECTOR_LABEL
        )
        return replace(sector, label=label)
    if _is_atom_hard_parity_sector(sector):
        expected = _response_parity_character(sector)
        if response_parity != expected:
            msg = (
                f"Resolved atomic sector requires parity {expected:+d}, got "
                f"{response_parity!r}."
            )
            raise ValueError(msg)
        return sector
    if response_parity != 0:
        msg = f"C1 sector cannot resolve atomic parity {response_parity!r}."
        raise ValueError(msg)
    return sector


def _response_symmetry_policy(sector: SourceSector, response_parity: int) -> str:
    """Return the persisted runtime policy name for one resolved geometry.

    Raises:
        ValueError: If the supplied character is invalid for the sector.
    """
    if sector.label == _ATOM_PARITY_PENDING_SECTOR_LABEL:
        if response_parity != 0:
            msg = "Pending atomic parity cannot have a resolved character."
            raise ValueError(msg)
        return "atom_hard_auto"
    if _is_atom_hard_parity_sector(sector):
        expected = _response_parity_character(sector)
        if response_parity != expected:
            msg = (
                f"Atomic response policy requires parity {expected:+d}, got "
                f"{response_parity!r}."
            )
            raise ValueError(msg)
        return "atom_hard_even" if response_parity == 1 else "atom_hard_odd"
    if response_parity != 0:
        msg = f"C1 response policy cannot use parity {response_parity!r}."
        raise ValueError(msg)
    return "molecule_c1"


def _project_source_center_to_invariant_subspace(
    source_center,
    source_sector: SourceSector | None,
    *,
    electron_count: int,
    tolerance: float = 1e-10,
) -> np.ndarray:
    r"""Project an affine dipole center into the sector's invariant subspace.

    For a spatial operation about ``c``, the electronic dipole transforms as
    ``D(gX) = g (D(X) + N_e c) - N_e c``.  Consequently ``D-D0`` is a
    Cartesian vector precisely when ``q = D0 + N_e c`` is fixed by every
    configured operation.  The joint nullspace of the stacked ``g-I``
    constraints gives the closest symmetry-consistent center.

    Returns:
        A float64 Cartesian center.  A missing or trivial sector is returned
        unchanged so callers of the previous private estimator API retain C1
        behavior.

    Raises:
        ValueError: If ``source_center`` is not a Cartesian vector.
    """
    center = np.asarray(source_center, dtype=np.float64)
    if center.shape != (3,):
        msg = f"source_center must have shape (3,), got {center.shape}."
        raise ValueError(msg)
    if source_sector is None or source_sector.is_trivial:
        return np.array(center, copy=True)

    symmetry_center = np.asarray(source_sector.center, dtype=np.float64)
    q = center + int(electron_count) * symmetry_center
    constraints = np.concatenate(
        [
            np.asarray(operation, dtype=np.float64) - np.eye(3)
            for operation in source_sector.operations
            if not _is_identity_operation(operation)
        ],
        axis=0,
    )
    _, singular_values, right_vectors = np.linalg.svd(
        constraints,
        full_matrices=True,
    )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    numerical_cutoff = (
        64.0 * max(constraints.shape) * np.finfo(np.float64).eps * max(1.0, largest)
    )
    cutoff = max(numerical_cutoff, float(tolerance) * max(1.0, largest))
    rank = int(np.sum(singular_values > cutoff))
    null_basis = right_vectors[rank:].T
    if null_basis.shape[1] == 0:
        projected_q = np.zeros(3, dtype=np.float64)
    else:
        projected_q = null_basis @ (null_basis.T @ q)
    return projected_q - int(electron_count) * symmetry_center


def _log_source_center_projection(
    unprojected_center,
    projected_center,
    source_sector: SourceSector | None,
) -> None:
    """Log the Cartesian correction made by a nontrivial source sector."""
    if source_sector is None or source_sector.is_trivial:
        return
    correction = np.asarray(projected_center) - np.asarray(unprojected_center)
    logger.info(
        "NQS-LIT source-center projection sector=%s correction="
        "(%.8e, %.8e, %.8e) correction_norm=%.8e",
        source_sector.label,
        float(correction[0]),
        float(correction[1]),
        float(correction[2]),
        float(np.linalg.norm(correction)),
    )


def _vector_covariance_penalty_gradient(
    response_vector_apply,
    response_params,
    batched_data: BatchedData,
    source_sector: SourceSector,
    source_operation,
):
    """Differentiate one source-sector covariance operation on one MC batch.

    Returns:
        The scalar covariance loss and its flattened real-parameter gradient.
    """
    loss, gradient = jax.value_and_grad(
        lambda local_params: _vector_covariance_penalty_loss(
            response_vector_apply,
            local_params,
            batched_data,
            source_sector,
            source_operation,
        )
    )(response_params)
    flat_gradient, _ = ravel_pytree(gradient)
    return loss, flat_gradient


def _vector_covariance_penalty_loss(
    response_vector_apply,
    response_params,
    batched_data: BatchedData,
    source_sector: SourceSector,
    source_operation,
):
    """Evaluate one source-sector covariance operation on one MC batch.

    Returns:
        Mean scale-invariant vector covariance residual.
    """
    per_sample = jax.vmap(
        lambda one: source_sector_covariance_loss(
            lambda transformed: response_vector_apply(
                response_params,
                transformed,
            ),
            one,
            source_sector,
            source_operation,
        ),
        in_axes=(batched_data.vmap_axis,),
    )(batched_data.data)
    return jnp.mean(per_sample)


def _summarize_source_covariance_losses(losses) -> _SourceCovarianceMetrics:
    """Summarize batch-mean losses across active symmetry operations.

    A non-finite operation is mapped to an infinite hard-guard value so a
    single broken operation cannot be hidden by the operation mean.

    Returns:
        The mean, guarded maximum, and worst active-operation index.
    """
    losses = jnp.asarray(losses)
    if losses.shape[0] == 0:
        return _SourceCovarianceMetrics(
            mean_loss=jnp.asarray(0.0, dtype=losses.dtype),
            max_loss=jnp.asarray(0.0, dtype=losses.dtype),
            worst_operation_index=jnp.asarray(-1, dtype=jnp.int32),
        )
    guarded_losses = jnp.where(jnp.isfinite(losses), losses, jnp.inf)
    return _SourceCovarianceMetrics(
        mean_loss=jnp.mean(losses),
        max_loss=jnp.max(guarded_losses),
        worst_operation_index=jnp.argmax(guarded_losses).astype(jnp.int32),
    )


def _coerce_source_covariance_metrics(value) -> _SourceCovarianceMetrics:
    """Accept the legacy scalar covariance-evaluator result as one operation.

    Returns:
        A full covariance metric tuple, with a scalar interpreted as both the
        mean and maximum of a one-operation evaluator.
    """
    if isinstance(value, _SourceCovarianceMetrics):
        return value
    scalar = jnp.asarray(value)
    return _SourceCovarianceMetrics(
        mean_loss=scalar,
        max_loss=scalar,
        worst_operation_index=jnp.asarray(0, dtype=jnp.int32),
    )


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


def _direction_update_scale(
    direction,
    *,
    learning_rate: float,
    max_norm: float | None,
):
    """Return the scalar applied to an unscaled SPRING direction."""
    scale = jnp.asarray(learning_rate, dtype=direction.dtype)
    if max_norm is not None:
        direction_norm = jnp.linalg.norm(direction)
        scale = jnp.minimum(
            scale,
            jnp.asarray(max_norm, dtype=direction.dtype)
            / (direction_norm + jnp.asarray(1e-12, dtype=direction.dtype)),
        )
    return scale


def _top_level_group_rms_and_norm(tree, group_name: str, *, dtype):
    """Return per-coordinate RMS and L2 norm for one top-level pytree group."""
    missing = jnp.asarray(jnp.nan, dtype=dtype)
    if not isinstance(tree, Mapping) or group_name not in tree:
        return missing, missing
    leaves = jax.tree_util.tree_leaves(tree[group_name])
    element_count = sum(int(leaf.size) for leaf in leaves)
    if element_count == 0:
        return missing, missing
    squared_norm = jnp.asarray(0.0, dtype=dtype)
    for leaf in leaves:
        leaf_array = jnp.asarray(leaf)
        squared_norm = squared_norm + jnp.sum(jnp.abs(leaf_array) ** 2)
    norm = jnp.sqrt(jnp.maximum(jnp.real(squared_norm), 0.0))
    rms = norm / jnp.sqrt(jnp.asarray(element_count, dtype=dtype))
    return rms, norm


def _spring_optimizer_diagnostics(
    params,
    combined_gradient,
    fidelity_gradient,
    reverse_kl_gradient,
    direction,
    updates,
    previous_direction,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    learning_rate: float,
    max_norm: float | None,
    damping,
    decay: float | jnp.ndarray,
    qfi_trace,
) -> _SpringOptimizerDiagnostics:
    """Summarize existing source-sampled SPRING tensors as scalar diagnostics.

    Returns:
        Scalar optimizer and top-level parameter-group diagnostics.
    """
    dtype = combined_gradient.dtype
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    combined_gradient_norm = jnp.linalg.norm(combined_gradient)
    fidelity_gradient_norm = jnp.linalg.norm(fidelity_gradient)
    weighted_reverse_kl_gradient = (
        jnp.asarray(reverse_kl_weight, dtype=dtype) * reverse_kl_gradient
    )
    weighted_reverse_kl_gradient_norm = jnp.linalg.norm(weighted_reverse_kl_gradient)
    gradient_product = fidelity_gradient_norm * weighted_reverse_kl_gradient_norm
    fidelity_kl_cosine = jnp.where(
        gradient_product > tiny,
        jnp.vdot(fidelity_gradient, weighted_reverse_kl_gradient).real
        / jnp.maximum(gradient_product, tiny),
        jnp.asarray(0.0, dtype=dtype),
    )
    component_gradient_norm = fidelity_gradient_norm + weighted_reverse_kl_gradient_norm
    gradient_cancellation_ratio = jnp.where(
        component_gradient_norm > tiny,
        combined_gradient_norm / jnp.maximum(component_gradient_norm, tiny),
        jnp.asarray(0.0, dtype=dtype),
    )

    direction_norm = jnp.linalg.norm(direction)
    learning_rate_array = jnp.asarray(learning_rate, dtype=dtype)
    update_scale = _direction_update_scale(
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )
    update_norm = jnp.abs(update_scale) * direction_norm
    clip_factor = update_scale / learning_rate_array

    parameter_count = jnp.asarray(max(int(combined_gradient.size), 1), dtype=dtype)
    qfi_trace = jnp.asarray(qfi_trace, dtype=dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.asarray(damping, dtype=dtype)
    history_rhs = (
        damping
        * jnp.asarray(decay, dtype=dtype)
        * jnp.asarray(previous_direction, dtype=dtype)
    )
    history_gradient_ratio = jnp.linalg.norm(history_rhs) / jnp.maximum(
        combined_gradient_norm,
        tiny,
    )

    _, unravel_fn = ravel_pytree(params)
    gradient_tree = unravel_fn(combined_gradient)
    group_gradient_rms = []
    group_update_norm = []
    for group_name in _SPRING_PARAMETER_GROUPS:
        gradient_rms, _ = _top_level_group_rms_and_norm(
            gradient_tree,
            group_name,
            dtype=dtype,
        )
        _, update_group_norm = _top_level_group_rms_and_norm(
            updates,
            group_name,
            dtype=dtype,
        )
        group_gradient_rms.append(gradient_rms)
        group_update_norm.append(update_group_norm)
    return _SpringOptimizerDiagnostics(
        available=jnp.asarray(True),
        combined_gradient_norm=combined_gradient_norm,
        fidelity_gradient_norm=fidelity_gradient_norm,
        weighted_reverse_kl_gradient_norm=weighted_reverse_kl_gradient_norm,
        fidelity_kl_cosine=fidelity_kl_cosine,
        gradient_cancellation_ratio=gradient_cancellation_ratio,
        direction_norm=direction_norm,
        update_norm=update_norm,
        clip_factor=clip_factor,
        damping=damping,
        mean_metric_diagonal=mean_metric_diagonal,
        history_gradient_ratio=history_gradient_ratio,
        parameter_group_gradient_rms=jnp.stack(group_gradient_rms),
        parameter_group_update_norm=jnp.stack(group_update_norm),
    )


def _empty_spring_optimizer_diagnostics(params) -> _SpringOptimizerDiagnostics:
    """Return an unavailable diagnostic record for a non-source SPRING branch."""
    first_leaf = jax.tree_util.tree_leaves(params)[0]
    dtype = jnp.real(first_leaf).dtype
    missing = jnp.asarray(jnp.nan, dtype=dtype)
    return _SpringOptimizerDiagnostics(
        available=jnp.asarray(False),
        combined_gradient_norm=missing,
        fidelity_gradient_norm=missing,
        weighted_reverse_kl_gradient_norm=missing,
        fidelity_kl_cosine=missing,
        gradient_cancellation_ratio=missing,
        direction_norm=missing,
        update_norm=missing,
        clip_factor=missing,
        damping=missing,
        mean_metric_diagonal=missing,
        history_gradient_ratio=missing,
        parameter_group_gradient_rms=jnp.full(3, missing, dtype=dtype),
        parameter_group_update_norm=jnp.full(3, missing, dtype=dtype),
    )


def _scaled_direction_updates(
    params,
    direction,
    *,
    learning_rate: float,
    max_norm: float | None,
):
    _, unravel_fn = ravel_pytree(params)
    scale = _direction_update_scale(
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )
    return unravel_fn(scale * direction)


def _symmetry_gradient_updates(
    params,
    flat_gradient,
    *,
    weight: float | jnp.ndarray,
    learning_rate: float,
    max_norm: float | None,
):
    """Take a separately scaled descent step for source covariance.

    This deliberately does not use the selected-axis SPRING metric; parameters
    exclusive to the two auxiliary vector heads have zero score in that metric.

    Returns:
        A parameter-shaped tree of independently clipped covariance updates.
    """
    direction = -jnp.asarray(weight, dtype=flat_gradient.dtype) * flat_gradient
    return _scaled_direction_updates(
        params,
        direction,
        learning_rate=learning_rate,
        max_norm=max_norm,
    )


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


def _source_covariance_host_values(stats) -> tuple[float, float]:
    """Read mean and worst-operation covariance diagnostics on the host.

    Returns:
        The operation-mean and worst-operation covariance losses.  Legacy
        statistics without the maximum field use their scalar mean for both.
    """
    mean_loss = float(jax.device_get(getattr(stats, "source_covariance_loss", 0.0)))
    max_loss = float(
        jax.device_get(getattr(stats, "source_covariance_max_loss", mean_loss))
    )
    return mean_loss, max_loss


def _require_eligible_nqs_checkpoint(
    stats,
    *,
    max_source_covariance: float | None,
    context: str,
) -> None:
    """Raise when a checkpoint is invalid or violates its worst-op guard.

    Raises:
        RuntimeError: If the checkpoint is not eligible for propagation.
    """
    if _is_eligible_nqs_checkpoint(
        stats,
        max_source_covariance=max_source_covariance,
    ):
        return
    covariance_mean, covariance_max = _source_covariance_host_values(stats)
    msg = (
        f"{context}; covariance mean={covariance_mean:.6e}, "
        f"max={covariance_max:.6e}, maximum={max_source_covariance!r}."
    )
    raise RuntimeError(msg)


def _is_eligible_nqs_checkpoint(
    stats,
    *,
    max_source_covariance: float | None = None,
) -> bool:
    """Return whether one held-out checkpoint is numerically admissible."""
    loss = float(jax.device_get(stats.loss))
    fidelity = float(jax.device_get(stats.fidelity))
    reverse_kl = float(jax.device_get(stats.reverse_kl))
    invalid = float(jax.device_get(stats.invalid_sample_fraction))
    covariance_mean, covariance_max = _source_covariance_host_values(stats)
    finite = bool(
        np.all(
            np.isfinite(
                (
                    loss,
                    fidelity,
                    reverse_kl,
                    invalid,
                    covariance_mean,
                    covariance_max,
                )
            )
        )
    )
    within_covariance = max_source_covariance is None or covariance_max <= float(
        max_source_covariance
    )
    return finite and invalid <= 0.0 and within_covariance


def _is_better_nqs_checkpoint(
    candidate,
    incumbent,
    *,
    max_source_covariance: float | None = None,
) -> bool:
    """Compare held-out checkpoints, rejecting non-finite/invalid estimates.

    Returns:
        Whether the candidate should replace the incumbent.
    """

    def score(stats):
        loss = float(jax.device_get(stats.loss))
        fidelity = float(jax.device_get(stats.fidelity))
        reverse_kl = float(jax.device_get(stats.reverse_kl))
        valid = _is_eligible_nqs_checkpoint(
            stats,
            max_source_covariance=max_source_covariance,
        )
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
