# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular dipole NQS-LIT workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import operator
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from functools import partial
from typing import Any, NamedTuple
from zipfile import BadZipFile

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
from jaqmc.response.inversion import lit_block_statistics
from jaqmc.response.inversion_io import (
    LITInversionSettings,
    invert_lit_npz,
    lit_inversion_npz_payload,
)
from jaqmc.response.lit import lit_error_bound
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    NQSLITSourceSums,
    NQSLITStats,
    ground_local_energy,
    local_action_ratio,
    merge_nqs_lit_source_sums,
    merge_nqs_lit_source_sums_across_devices,
    molecular_electronic_dipole,
    nqs_lit_source_sampled_sums,
    nqs_lit_stats_from_source_sums,
    parity_log_amplitude_loss,
    parity_project_log_amplitude,
    restore_params_from_checkpoint,
    weighted_complex_moments,
)
from jaqmc.response.source_sector import (
    SourceSector,
    discover_source_sector,
    transform_molecule_data,
)
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils import parallel_jax
from jaqmc.utils.checkpoint import NumPyCheckpointManager
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
_CONTINUATION_CHECKPOINT_SCHEMA_VERSION = 3
_CONTINUATION_CHECKPOINT_PREFIX = "continuation"


@configurable_dataclass
class MolecularLITConfig:
    """Configuration for molecular dipole NQS-LIT spectra."""

    eta: float = 0.02
    omega_min: float = 0.0
    omega_max: float = 1.0
    omega_points: int = 501
    omega_values: tuple[float, ...] = field(default_factory=tuple)
    axes: str = "xyz"
    output_filename: str = "lit_spectrum.npz"
    # A formal inversion runs after the raw workflow NPZ has been written.
    # The current NPZ is always included; additional paths support a matched,
    # correlated multi-eta fit without weakening the serial frequency chain.
    inversion_enabled: bool = False
    inversion_output_filename: str = "lit_inversion.npz"
    inversion_additional_input_paths: tuple[str, ...] = field(default_factory=tuple)
    inversion_assume_independent: bool = False
    inversion_threshold: float | None = None
    inversion_pole_energies: tuple[float, ...] = field(default_factory=tuple)
    inversion_continuum_grid: tuple[float, ...] = field(default_factory=tuple)
    inversion_continuum_regularization: float = 0.0
    inversion_fit_pole_energies: bool = False
    inversion_pole_energy_bounds: tuple[tuple[float, float], ...] = field(
        default_factory=tuple
    )
    inversion_covariance_relative_tolerance: float = 1e-10
    inversion_max_fitted_poles: int = 8
    inversion_pole_fit_tolerance: float = 1e-7
    inversion_pole_fit_max_iterations: int = 200
    inversion_solver_tolerance: float = 1e-10
    inversion_solver_max_iterations: int | None = None
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
    # Per-device alternatives keep one configuration useful across different
    # local GPU counts.  They are mutually exclusive with the corresponding
    # global sizes above; in serial mode the multiplier is one.
    nqs_train_update_batch_size_per_device: int = 0
    nqs_eval_batch_size_per_device: int = 0
    # Frequencies always remain one serial continuation chain.  This optional
    # execution mode shards the Monte Carlo batch for one frequency across all
    # process-local devices while keeping parameters, statistics, and SPRING
    # state replicated.
    nqs_data_parallel: str = "off"
    nqs_source_pool_dir: str = ""
    nqs_reuse_source_pool: bool = True
    nqs_save_source_pool: bool = True
    # Before applying the resolvent action, fit the independent response NQS
    # directly to the dipole source on the fixed pi_Phi pools.  This is needed
    # in particular for hard atomic parity: copying an opposite-parity ground
    # state and immediately projecting it can create the exact zero function.
    nqs_source_distillation_iterations: int = 1000
    nqs_energy_steps: int = 2
    nqs_burn_in: int = 20
    nqs_iterations: int = 200
    nqs_learning_rate: float = 1e-3
    nqs_reverse_kl_weight: float = 1.0
    nqs_spring_epsilon: float = 1e-3
    nqs_spring_decay: float = 0.99
    nqs_spring_damping_floor: float = 1e-12
    nqs_sr_max_norm: float | None = 0.1
    nqs_sr_score_eps: float = 1e-10
    nqs_warm_start_omega: float | None = -3.674932217565499
    nqs_warm_start_iterations: int = 100
    # Stop an NQS optimization only after its best held-out fidelity has
    # plateaued.  A zero patience disables early stopping.  The baseline is
    # established no earlier than ``start_iteration``; subsequent cumulative
    # improvements must exceed ``min_delta`` to reset the patience clock.
    nqs_fidelity_plateau_start_iteration: int = 0
    nqs_fidelity_plateau_patience_iterations: int = 0
    nqs_fidelity_plateau_min_delta: float = 1e-5
    # Maximum optimization budget for each adaptive continuation bridge.
    nqs_continuation_iterations: int = 100
    nqs_continuation_step_fraction: float = 0.2
    # Cap the next proposal by the latest accepted bridge step.  A clean
    # bridge may grow that cap by this factor; any bisected/recovered bridge
    # holds its actual successful step for the next proposal.
    nqs_continuation_step_growth_factor: float = 1.25
    nqs_continuation_fidelity_retention: float = 0.95
    nqs_stage_reweight_ess_fraction_min: float = 0.0
    nqs_continuation_allow_min_step_override: bool = True
    nqs_continuation_min_step: float | None = None
    nqs_continuation_max_points: int = 256
    # Optional previous run, continuation-checkpoint root, axis directory, or
    # checkpoint file.  New checkpoints are always written below save_path.
    nqs_continuation_restore_path: str = ""
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
    # Runtime support is deliberately restricted to an automatically selected
    # hard atomic parity (opposite the diagnosed ground-state parity), or a
    # symmetry-free multi-center C1 response.
    nqs_parity_eval_batch_size: int = 256
    nqs_sector_tolerance: float = 1e-5
    nqs_atomic_source_parity_max_loss: float = 1e-3
    # Reject an atomic checkpoint unless one inversion parity has a held-out
    # residual below this threshold.  The response uses the opposite parity.
    nqs_atomic_ground_parity_max_loss: float = 1e-3
    nqs_selection_interval: int = 50
    nqs_log_interval: int = 50


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


_SPRING_PARAMETER_GROUPS = ("response",)


class _NQSUpdateCarry(NamedTuple):
    spring: _SpringState
    rng: jax.Array


class _SourceDistillationStats(NamedTuple):
    """Held-out source-overlap diagnostics for response initialization."""

    loss: jax.Array
    fidelity: jax.Array
    reverse_kl: jax.Array
    reweight_ess_fraction: jax.Array
    invalid_sample_fraction: jax.Array


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


class _ContinuationCapacityDiagnostics(NamedTuple):
    """Point-budget forecast for one continuation proposal."""

    remaining_gap: float
    remaining_bridge_slots: int
    required_mean_step: float
    capacity_ratio: float


@dataclass
class _FidelityPlateauTracker:
    """Host-side early stopping on cumulative best held-out fidelity."""

    start_iteration: int
    patience_iterations: int
    min_delta: float
    reference_fidelity: float | None = None
    last_significant_iteration: int | None = None

    @property
    def enabled(self) -> bool:
        return self.patience_iterations > 0

    def observe(self, iteration: int, best_fidelity: float) -> bool:
        """Record the best fidelity and report whether it has plateaued.

        Returns:
            Whether the patience window has elapsed without a significant
            cumulative improvement.
        """
        if not self.enabled or iteration < self.start_iteration:
            return False
        if not np.isfinite(best_fidelity):
            return False
        if self.reference_fidelity is None:
            self.reference_fidelity = float(best_fidelity)
            self.last_significant_iteration = int(iteration)
            return False
        if best_fidelity > self.reference_fidelity + self.min_delta:
            self.reference_fidelity = float(best_fidelity)
            self.last_significant_iteration = int(iteration)
            return False
        if self.last_significant_iteration is None:
            return False
        return iteration - self.last_significant_iteration >= self.patience_iterations

    def defer(self, iteration: int) -> None:
        """Restart patience after an unhealthy held-out observation."""
        if (
            self.enabled
            and self.reference_fidelity is not None
            and iteration >= self.start_iteration
        ):
            self.last_significant_iteration = int(iteration)


class _ContinuationCheckpoint(NamedTuple):
    """Serializable latest-good state for one response-axis bridge chain."""

    schema_version: int
    state_fingerprint: str
    full_config_digest: str
    axis: int
    target_omega: float
    current_omega: float
    accepted_points: int
    ground_checkpoint_step: int
    ground_energy: float
    source_center: float
    source_norm: float
    response_parity: int
    response_params: Any
    rng: jax.Array
    current_stats: NQSLITStats
    history_json: str
    warm_start_selected_iteration: int


class _ContinuationResumeState(NamedTuple):
    """Validated in-memory continuation state restored from one checkpoint."""

    response_params: Any
    rng: jax.Array
    current_stats: NQSLITStats
    current_omega: float
    records: tuple[_ContinuationRecord, ...]
    warm_start_selected_iteration: int


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
        self._nqs_chunk_sums_kernel_cache: dict[
            tuple[int, int, int],
            tuple[Any, Any, Any],
        ] = {}
        self._source_distillation_eval_kernel_cache: dict[tuple[Any, ...], Any] = {}
        self._validate_config()

    def run(self) -> None:
        self._run_serial_scan()

    def _run_serial_scan(self) -> None:
        axes = _axis_indices(self.lit_config.axes)
        omega = self._omega_grid()
        seed = self.config.seed if self.config.seed is not None else int(time.time())
        self._run_seed = int(seed)
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
        source_pool_target_sha256 = _source_pool_target_digest(
            ground_params,
            batched_data.data,
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

        signed_lit = np.zeros((len(axes), len(omega)), dtype=np.float64)
        broadened = np.zeros_like(signed_lit)
        fidelity = np.zeros_like(signed_lit)
        reverse_kl = np.zeros_like(signed_lit)
        residual_norm = np.zeros_like(signed_lit)
        equation_relative_residual = np.zeros_like(signed_lit)
        action_norm = np.zeros_like(signed_lit)
        source_norm = np.zeros_like(signed_lit)
        error_bound_monitor = np.zeros_like(signed_lit)
        error_d = np.zeros_like(signed_lit)
        error_d_correction = np.zeros_like(signed_lit)
        error_d_shifted = np.zeros_like(signed_lit)
        error_d_valid = np.zeros_like(signed_lit, dtype=np.bool_)
        reweight_ess = np.zeros_like(signed_lit)
        reweight_ess_fraction = np.zeros_like(signed_lit)
        reweight_max_fraction = np.zeros_like(signed_lit)
        invalid_sample_fraction = np.zeros_like(signed_lit)
        selected_iteration = np.zeros_like(signed_lit, dtype=np.int64)
        normalization = np.zeros((len(axes), len(omega)), dtype=np.complex128)
        correction_overlap = np.zeros_like(normalization)
        source_centers = np.zeros(len(axes), dtype=np.float64)
        axis_source_norm = np.zeros(len(axes), dtype=np.float64)
        atomic_source_parity_loss = np.full(len(axes), np.nan, dtype=np.float64)
        eval_pool_sha256 = np.empty(len(axes), dtype="<U64")
        axis_jackknife_blocks: list[np.ndarray] = []
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

        logger.info(
            "NQS-LIT response_policy=%s source_sector=%s order=%d",
            _response_symmetry_policy(
                source_sector,
                parity_resolution.response_parity,
            ),
            source_sector.label,
            source_sector.order,
        )

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
            response_apply, response_params = self._make_response_ansatz(
                shape_example,
                response_rng,
                ground_params,
                source_sector=source_sector,
                response_parity=parity_resolution.response_parity,
            )
            loaded_pools = self._try_load_source_pools(
                batched_data,
                axis=axis,
                source_center=source_center,
                target_sha256=source_pool_target_sha256,
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
                        target_sha256=source_pool_target_sha256,
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
                        target_sha256=source_pool_target_sha256,
                        split="eval",
                        batches=self.lit_config.nqs_eval_pool_batches,
                    )
                )
            else:
                train_pool, eval_pool = loaded_pools
                axis_batched_data = batched_data
            self._validate_source_pool_chunks(train_pool, eval_pool)
            eval_pool_sha256[axis_pos] = _tree_content_digest(eval_pool)
            self._log_response_pool_capacity(
                response_params,
                train_pool,
                eval_pool,
                axis=axis,
            )
            atomic_source_parity_loss[axis_pos] = self._validate_atomic_source_parity(
                ground_logpsi,
                ground_params,
                eval_pool,
                source_sector,
                vector_source_centers,
                axis=axis,
                response_parity=parity_resolution.response_parity,
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
            state_fingerprint, full_config_digest = _continuation_checkpoint_digests(
                self.lit_config,
                response_params=response_params,
                ground_params=ground_params,
                train_pool=train_pool,
                eval_pool=eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                ground_checkpoint_step=checkpoint_step,
                response_parity=parity_resolution.response_parity,
                target_omega=float(omega[0]),
                spectrum_omega=omega,
            )
            resume_state = self._restore_nqs_continuation_checkpoint(
                response_params,
                rng,
                eval_pool,
                response_apply=response_apply,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                axis=axis,
                source_center=source_center,
                source_norm=axis_phi_norm,
                ground_energy=ground_energy,
                ground_checkpoint_step=checkpoint_step,
                response_parity=parity_resolution.response_parity,
                target_omega=float(omega[0]),
                state_fingerprint=state_fingerprint,
                full_config_digest=full_config_digest,
            )
            if resume_state is None:
                response_params, rng = self._distill_response_from_source(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    train_pool,
                    eval_pool,
                    rng,
                    axis=axis,
                    source_center=source_center,
                )
                (
                    response_params,
                    continuation_start_stats,
                    warm_start_selected_iteration[axis_pos],
                    rng,
                ) = self._warm_start_axis(
                    update_step,
                    response_params,
                    train_pool,
                    eval_pool,
                    rng,
                    response_apply=response_apply,
                    ground_logpsi=ground_logpsi,
                    ground_params=ground_params,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                )
                resume_omega = None
                existing_records: tuple[_ContinuationRecord, ...] = ()
            else:
                response_params = resume_state.response_params
                rng = resume_state.rng
                continuation_start_stats = resume_state.current_stats
                warm_start_selected_iteration[axis_pos] = (
                    resume_state.warm_start_selected_iteration
                )
                resume_omega = resume_state.current_omega
                existing_records = resume_state.records
            checkpoint_callback = partial(
                self._save_nqs_continuation_checkpoint,
                axis=axis,
                target_omega=float(omega[0]),
                ground_checkpoint_step=checkpoint_step,
                ground_energy=ground_energy,
                source_center=source_center,
                source_norm=axis_phi_norm,
                response_parity=parity_resolution.response_parity,
                state_fingerprint=state_fingerprint,
                full_config_digest=full_config_digest,
                warm_start_selected_iteration=int(
                    warm_start_selected_iteration[axis_pos]
                ),
            )
            response_params, _, bridge_records, rng = self._continue_nqs_to_spectrum(
                update_step,
                response_params,
                continuation_start_stats,
                train_pool,
                eval_pool,
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
                resume_omega=resume_omega,
                existing_records=existing_records,
                checkpoint_callback=checkpoint_callback,
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
            omega_jackknife_blocks: list[np.ndarray] = []
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
                stats, block_sums = self._nqs_stats_chunked(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    axis=axis,
                    source_center=source_center,
                    source_norm=axis_phi_norm,
                    ground_energy=ground_energy,
                    omega=jnp.asarray(float(omega_value)),
                    return_block_sums=True,
                )
                stats = stats._replace(
                    loss=_regularized_loss(
                        stats,
                        self.lit_config.nqs_reverse_kl_weight,
                    )
                )
                host_stats = jax.device_get(stats)
                omega_jackknife_blocks.append(
                    _signed_lit_jackknife_pseudovalues(
                        stats,
                        block_sums,
                        source_norm=axis_phi_norm,
                        omega=float(omega_value),
                        eta=float(self.lit_config.eta),
                    )
                )
                signed_lit[axis_pos, omega_pos] = float(host_stats.signed_lit)
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
                    error_d_valid=bool(host_stats.error_d_valid),
                )
                error_d[axis_pos, omega_pos] = float(host_stats.error_d)
                error_d_correction[axis_pos, omega_pos] = float(
                    host_stats.error_d_correction
                )
                error_d_shifted[axis_pos, omega_pos] = float(host_stats.error_d_shifted)
                error_d_valid[axis_pos, omega_pos] = bool(host_stats.error_d_valid)
                reweight_ess[axis_pos, omega_pos] = float(host_stats.reweight_ess)
                reweight_ess_fraction[axis_pos, omega_pos] = float(
                    host_stats.reweight_ess_fraction
                )
                reweight_max_fraction[axis_pos, omega_pos] = float(
                    host_stats.reweight_max_fraction
                )
                invalid_sample_fraction[axis_pos, omega_pos] = float(
                    host_stats.invalid_sample_fraction
                )
                normalization[axis_pos, omega_pos] = complex(host_stats.normalization)
                correction_overlap[axis_pos, omega_pos] = complex(
                    host_stats.correction_overlap
                )
            axis_jackknife_blocks.append(np.stack(omega_jackknife_blocks, axis=0))

        signed_lit_jackknife_blocks = np.stack(axis_jackknife_blocks, axis=0)
        uncertainty_output: dict[str, Any] = {}
        if signed_lit_jackknife_blocks.shape[-1] >= 2:
            block_statistics = lit_block_statistics(signed_lit_jackknife_blocks)
            uncertainty_output = {
                "signed_lit_jackknife_blocks": signed_lit_jackknife_blocks,
                "signed_lit_jackknife_block_count": np.asarray(
                    block_statistics.block_count,
                    dtype=np.int64,
                ),
                "signed_lit_covariance": block_statistics.covariance,
                "signed_lit_standard_error": block_statistics.standard_error,
            }
        total_broadened = np.sum(broadened, axis=0)
        output_path = self.save_path / self.lit_config.output_filename
        _save_npz(
            output_path,
            backend="nqs_lit",
            omega=omega,
            eta=self.lit_config.eta,
            axes=self.lit_config.axes,
            axis_indices=np.asarray(axes, dtype=np.int64),
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
            error_d_correction=error_d_correction,
            error_d_shifted=error_d_shifted,
            error_d_valid=error_d_valid,
            reweight_ess=reweight_ess,
            reweight_ess_fraction=reweight_ess_fraction,
            reweight_max_fraction=reweight_max_fraction,
            invalid_sample_fraction=invalid_sample_fraction,
            selected_iteration=selected_iteration,
            normalization=normalization,
            correction_overlap=correction_overlap,
            ground_energy=ground_energy,
            ground_checkpoint_step=checkpoint_step,
            nqs_train_pool_batches=self.lit_config.nqs_train_pool_batches,
            nqs_eval_pool_batches=self.lit_config.nqs_eval_pool_batches,
            nqs_pool_stride=self.lit_config.nqs_pool_stride,
            nqs_reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
            nqs_spring_epsilon=self.lit_config.nqs_spring_epsilon,
            nqs_spring_decay=self.lit_config.nqs_spring_decay,
            nqs_spring_damping_floor=self.lit_config.nqs_spring_damping_floor,
            nqs_source_distillation_iterations=(
                self.lit_config.nqs_source_distillation_iterations
            ),
            nqs_parity_eval_batch_size=self.lit_config.nqs_parity_eval_batch_size,
            nqs_sector_tolerance=self.lit_config.nqs_sector_tolerance,
            nqs_atomic_source_parity_max_loss=(
                self.lit_config.nqs_atomic_source_parity_max_loss
            ),
            nqs_atomic_ground_parity_max_loss=(
                self.lit_config.nqs_atomic_ground_parity_max_loss
            ),
            source_sector_label=source_sector.label,
            source_sector_order=source_sector.order,
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
            vector_source_centers=vector_source_centers,
            vector_source_norms=vector_source_norms,
            atomic_source_parity_loss=atomic_source_parity_loss,
            nqs_selection_interval=self.lit_config.nqs_selection_interval,
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            warm_start_selected_iteration=warm_start_selected_iteration,
            nqs_fidelity_plateau_start_iteration=(
                self.lit_config.nqs_fidelity_plateau_start_iteration
            ),
            nqs_fidelity_plateau_patience_iterations=(
                self.lit_config.nqs_fidelity_plateau_patience_iterations
            ),
            nqs_fidelity_plateau_min_delta=(
                self.lit_config.nqs_fidelity_plateau_min_delta
            ),
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
            nqs_continuation_step_growth_factor=(
                self.lit_config.nqs_continuation_step_growth_factor
            ),
            nqs_continuation_fidelity_retention=(
                self.lit_config.nqs_continuation_fidelity_retention
            ),
            nqs_stage_reweight_ess_fraction_min=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            nqs_continuation_allow_min_step_override=bool(
                self.lit_config.nqs_continuation_allow_min_step_override
            ),
            nqs_continuation_min_step=_optional_float(
                self.lit_config.nqs_continuation_min_step
            ),
            nqs_continuation_max_points=self.lit_config.nqs_continuation_max_points,
            nqs_continuation_restore_path=(
                self.lit_config.nqs_continuation_restore_path
            ),
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
            eval_pool_sha256=eval_pool_sha256,
            **uncertainty_output,
        )
        self._log_nqs_summary(str(output_path), fidelity)
        if self.lit_config.inversion_enabled:
            self._run_formal_inversion(output_path)

    def _inversion_settings(self) -> LITInversionSettings:
        threshold = self.lit_config.inversion_threshold
        if threshold is None:
            msg = "lit.inversion_threshold is required when inversion is enabled."
            raise ValueError(msg)
        bounds = self.lit_config.inversion_pole_energy_bounds
        return LITInversionSettings(
            threshold=threshold,
            pole_energies=self.lit_config.inversion_pole_energies,
            continuum_grid=self.lit_config.inversion_continuum_grid,
            continuum_regularization=(
                self.lit_config.inversion_continuum_regularization
            ),
            fit_pole_energies=self.lit_config.inversion_fit_pole_energies,
            pole_energy_bounds=bounds or None,
            covariance_relative_tolerance=(
                self.lit_config.inversion_covariance_relative_tolerance
            ),
            max_fitted_poles=self.lit_config.inversion_max_fitted_poles,
            pole_fit_tolerance=self.lit_config.inversion_pole_fit_tolerance,
            pole_fit_max_iterations=(self.lit_config.inversion_pole_fit_max_iterations),
            solver_tolerance=self.lit_config.inversion_solver_tolerance,
            solver_max_iterations=(self.lit_config.inversion_solver_max_iterations),
        )

    def _run_formal_inversion(self, output_path: UPath) -> None:
        additional = self.lit_config.inversion_additional_input_paths
        source_paths = (str(output_path), *additional)
        if len(set(source_paths)) != len(source_paths):
            msg = "lit inversion input paths must be unique."
            raise ValueError(msg)
        inversion = invert_lit_npz(
            source_paths,
            self._inversion_settings(),
            assume_independent=self.lit_config.inversion_assume_independent,
        )
        inversion_path = self.save_path / self.lit_config.inversion_output_filename
        _save_npz_compressed(
            inversion_path,
            **lit_inversion_npz_payload(inversion),
        )
        logger.info(
            "Wrote formal LIT inversion to %s (poles=%s, eta_count=%d, "
            "underdetermined=%s)",
            inversion_path,
            np.array2string(inversion.result.pole_energies, precision=10),
            inversion.result.diagnostics.unique_eta_count,
            inversion.result.diagnostics.underdetermined,
        )

    def _omega_grid(self) -> np.ndarray:
        return _lit_omega_grid(self.lit_config)

    def _validate_config(self) -> None:
        omega = _lit_omega_grid(self.lit_config)
        if not np.isfinite(self.lit_config.eta) or self.lit_config.eta <= 0.0:
            msg = "lit.eta must be finite and positive."
            raise ValueError(msg)
        self._validate_serial_scan_config(omega)
        self._validate_chunk_config()
        self._validate_data_parallel_config()
        self._validate_nqs_stabilizer_config()
        self._validate_source_sector_config()
        self._validate_nqs_iteration_config()
        self._validate_continuation_config()
        self._validate_inversion_config()

    def _validate_inversion_config(self) -> None:
        if not self.lit_config.inversion_enabled:
            return
        output_filename = self.lit_config.inversion_output_filename
        if not output_filename or output_filename == self.lit_config.output_filename:
            msg = (
                "lit.inversion_output_filename must be nonempty and differ "
                "from lit.output_filename."
            )
            raise ValueError(msg)
        if not output_filename.endswith(".npz"):
            msg = "lit.inversion_output_filename must end in '.npz'."
            raise ValueError(msg)
        additional = self.lit_config.inversion_additional_input_paths
        if any(not isinstance(path, str) or not path for path in additional):
            msg = "lit.inversion_additional_input_paths must contain nonempty paths."
            raise ValueError(msg)
        if len(set(additional)) != len(additional):
            msg = "lit.inversion_additional_input_paths must be unique."
            raise ValueError(msg)
        self._inversion_settings()

    def _validate_serial_scan_config(self, omega: np.ndarray) -> None:
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
        distillation_iterations = self.lit_config.nqs_source_distillation_iterations
        if (
            isinstance(distillation_iterations, (bool, np.bool_))
            or not isinstance(distillation_iterations, (int, np.integer))
            or int(distillation_iterations) < 1
        ):
            msg = "lit.nqs_source_distillation_iterations must be a positive integer."
            raise ValueError(msg)
        plateau_integers = (
            (
                "nqs_fidelity_plateau_start_iteration",
                self.lit_config.nqs_fidelity_plateau_start_iteration,
            ),
            (
                "nqs_fidelity_plateau_patience_iterations",
                self.lit_config.nqs_fidelity_plateau_patience_iterations,
            ),
        )
        for name, value in plateau_integers:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"lit.{name} must be a nonnegative integer.")
        plateau_delta = self.lit_config.nqs_fidelity_plateau_min_delta
        if not np.isfinite(plateau_delta) or not 0.0 <= float(plateau_delta) <= 1.0:
            raise ValueError(
                "lit.nqs_fidelity_plateau_min_delta must be finite and between 0 and 1."
            )

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
        if self.lit_config.nqs_parity_eval_batch_size < 1:
            msg = "lit.nqs_parity_eval_batch_size must be positive."
            raise ValueError(msg)
        tolerance = self.lit_config.nqs_sector_tolerance
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            msg = "lit.nqs_sector_tolerance must be positive."
            raise ValueError(msg)
        source_parity_maximum = self.lit_config.nqs_atomic_source_parity_max_loss
        if (
            not np.isfinite(source_parity_maximum)
            or not 0.0 < source_parity_maximum < 1.0
        ):
            msg = (
                "lit.nqs_atomic_source_parity_max_loss must be finite and "
                "strictly between 0 and 1."
            )
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
        continuation_iterations = self.lit_config.nqs_continuation_iterations
        if (
            isinstance(continuation_iterations, (bool, np.bool_))
            or not isinstance(continuation_iterations, (int, np.integer))
            or int(continuation_iterations) < 1
        ):
            msg = "lit.nqs_continuation_iterations must be a positive integer."
            raise ValueError(msg)
        step_fraction = self.lit_config.nqs_continuation_step_fraction
        if not np.isfinite(step_fraction) or step_fraction <= 0.0:
            msg = "lit.nqs_continuation_step_fraction must be positive."
            raise ValueError(msg)
        growth = self.lit_config.nqs_continuation_step_growth_factor
        if not np.isfinite(growth) or not 1.0 <= float(growth) <= 2.0:
            msg = (
                "lit.nqs_continuation_step_growth_factor must be finite and "
                "satisfy 1 <= value <= 2."
            )
            raise ValueError(msg)
        retention = self.lit_config.nqs_continuation_fidelity_retention
        if not 0.0 < retention <= 1.0:
            msg = "lit.nqs_continuation_fidelity_retention must satisfy 0 < value <= 1."
            raise ValueError(msg)
        stage_ess = self.lit_config.nqs_stage_reweight_ess_fraction_min
        if not np.isfinite(stage_ess) or not 0.0 <= float(stage_ess) <= 1.0:
            msg = (
                "lit.nqs_stage_reweight_ess_fraction_min must be finite and "
                "between 0 and 1."
            )
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

    def _validate_chunk_config(self) -> None:
        pairs = (
            (
                "nqs_train_update_batch_size",
                self.lit_config.nqs_train_update_batch_size,
                "nqs_train_update_batch_size_per_device",
                self.lit_config.nqs_train_update_batch_size_per_device,
            ),
            (
                "nqs_eval_batch_size",
                self.lit_config.nqs_eval_batch_size,
                "nqs_eval_batch_size_per_device",
                self.lit_config.nqs_eval_batch_size_per_device,
            ),
        )
        for global_name, global_value, local_name, local_value in pairs:
            if int(global_value) < 0:
                raise ValueError(f"lit.{global_name} must be nonnegative.")
            if int(local_value) < 0:
                raise ValueError(f"lit.{local_name} must be nonnegative.")
            if int(global_value) > 0 and int(local_value) > 0:
                raise ValueError(
                    f"lit.{global_name} and lit.{local_name} are mutually exclusive."
                )
        for name in ("nqs_train_pool_batches", "nqs_eval_pool_batches"):
            value = getattr(self.lit_config, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"lit.{name} must be a positive integer.")

    def _validate_data_parallel_config(self) -> None:
        mode = self._nqs_data_parallel_mode()
        if mode not in {"off", "local_devices"}:
            msg = (
                "lit.nqs_data_parallel must be 'off' or 'local_devices', got "
                f"{self.lit_config.nqs_data_parallel!r}."
            )
            raise ValueError(msg)
        if mode == "off":
            return
        if jax.process_count() != 1:
            msg = (
                "lit.nqs_data_parallel=local_devices currently requires one JAX "
                "process; use one process controlling all GPUs on the worker."
            )
            raise ValueError(msg)
        # A fully constructed workflow always has the base workflow config.
        # Keep mode-only validation usable by lightweight object.__new__ unit
        # fixtures that deliberately provide only ``lit_config``.
        if not hasattr(self, "config"):
            return
        self._validate_data_parallel_batch_size(
            self._nqs_train_update_batch_size(),
            purpose="training",
        )
        self._validate_data_parallel_batch_size(
            self._nqs_eval_batch_size(),
            purpose="evaluation",
        )

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
        sector = discover_source_sector(
            geometry_data.atoms,
            geometry_data.charges,
            tolerance=float(self.lit_config.nqs_sector_tolerance),
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
                        atol=self.lit_config.nqs_sector_tolerance,
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
                int(self.lit_config.nqs_parity_eval_batch_size),
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

    def _validate_atomic_source_parity(
        self,
        ground_logpsi,
        ground_params,
        eval_pool: BatchedData,
        source_sector: SourceSector,
        source_centers,
        *,
        axis: int,
        response_parity: int = 0,
    ) -> float:
        """Validate an atomic dipole source against its diagnosed hard parity.

        C1 molecules have no spatial-sector preflight and return ``NaN``.

        Returns:
            The held-out atomic source parity loss, or ``NaN`` for C1.

        Raises:
            ValueError: If ``source_centers`` is not a Cartesian vector.
            RuntimeError: If the atomic parity loss is non-finite or exceeds
                the configured maximum.
        """
        if not _is_atom_hard_parity_sector(source_sector):
            return float("nan")

        evaluation_batch = _cyclic_batched_data_chunk(
            eval_pool,
            min(
                eval_pool.batch_size,
                int(self.lit_config.nqs_parity_eval_batch_size),
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
        expected_parity = _response_parity_character(source_sector)
        if response_parity != expected_parity:
            msg = (
                "Atomic source validation requires resolved response parity "
                f"{expected_parity:+d}, got {response_parity!r}."
            )
            raise ValueError(msg)
        active_operations = tuple(
            operation
            for operation in source_sector.operations
            if not _is_identity_operation(operation)
        )
        if len(active_operations) != 1 or not np.allclose(
            np.asarray(active_operations[0]),
            -np.eye(3),
            rtol=0.0,
            atol=self.lit_config.nqs_sector_tolerance,
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
        maximum = float(self.lit_config.nqs_atomic_source_parity_max_loss)
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
                "lit.nqs_atomic_source_parity_max_loss="
                f"{maximum:.6e} for response parity {response_parity:+d}. "
                "The sampled (D-D0)Psi0 source is outside the diagnosed "
                "atomic response sector; check the ground checkpoint, source "
                "center, and source-pool equilibration."
            )
            raise RuntimeError(msg)
        return loss

    def _make_response_ansatz(  # noqa: C901
        self,
        example,
        response_rng,
        ground_params,
        *,
        source_sector: SourceSector | None = None,
        response_parity: int = 0,
    ):
        """Create the independent PRL-style response NQS.

        The production parameter tree is exactly the raw response-network
        parameter tree.  In a hard atomic sector the raw network is initialized
        independently rather than copied from the ground state: copying a
        nearly pure ground-state parity and projecting onto the opposite parity
        produces a numerically singular zero state.  The subsequent fixed
        ``pi_Phi`` distillation supplies the physically useful initialization.
        For a symmetry-free C1 molecule, matching ground-state parameters are
        still a useful nonsingular starting point.

        Returns:
            Scalar response apply function and the direct response-network
            parameter tree.

        Raises:
            ValueError: If the resolved atom/C1 symmetry policy is inconsistent.
        """
        if source_sector is None:
            msg = (
                "A response symmetry policy resolved from the physical fixed "
                "geometry is required."
            )
            raise ValueError(msg)
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
        if not hard_atomic_parity:
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

        return projected_raw_apply, raw_params

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
        target_sha256: str,
        split: str,
        batches: int,
        pool_root: UPath | None = None,
    ):
        pool_path = self._source_pool_path(axis, split, root=pool_root)
        expected_walkers = self._expected_source_pool_walkers(batches)
        metadata = self._source_pool_metadata(
            axis,
            source_center,
            target_sha256=target_sha256,
            expected_walkers=expected_walkers,
        )
        if self.lit_config.nqs_reuse_source_pool and pool_path.exists():
            try:
                pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
                _require_pool_walker_count(
                    pool,
                    expected_walkers=expected_walkers,
                    split=split,
                )
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
        _require_pool_walker_count(
            pool,
            expected_walkers=expected_walkers,
            split=split,
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
        target_sha256: str,
        pool_root: UPath | None = None,
    ):
        if not self.lit_config.nqs_reuse_source_pool:
            return None
        loaded = []
        split_batches = (
            ("train", self.lit_config.nqs_train_pool_batches),
            ("eval", self.lit_config.nqs_eval_pool_batches),
        )
        for split, batches in split_batches:
            expected_walkers = self._expected_source_pool_walkers(batches)
            metadata = self._source_pool_metadata(
                axis,
                source_center,
                target_sha256=target_sha256,
                expected_walkers=expected_walkers,
            )
            pool_path = self._source_pool_path(axis, split, root=pool_root)
            if not pool_path.exists():
                return None
            try:
                pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
                _require_pool_walker_count(
                    pool,
                    expected_walkers=expected_walkers,
                    split=split,
                )
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
        *,
        target_sha256: str,
        expected_walkers: int,
    ) -> dict[str, object]:
        return {
            "axis": float(axis),
            "source_center": float(source_center),
            "source_floor": float(self.lit_config.nqs_source_floor),
            "walker_count": float(expected_walkers),
            # A pi_Phi pool is distributed according to the exact ground
            # checkpoint and Hamiltonian geometry.  Binding both here avoids
            # silently reusing statistically incompatible walkers after a
            # checkpoint or system change that leaves the pool shape intact.
            "target_sha256": str(target_sha256),
        }

    def _expected_source_pool_walkers(self, batches: int) -> int:
        return int(self.config.batch_size) * int(batches)

    def _source_distillation_log_ratios(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
    ):
        """Return ``log(Psi_response/Phi)`` and exact-pi_Phi weights."""
        score_eps = jnp.asarray(
            self.lit_config.nqs_sr_score_eps,
            dtype=batched_data.data.electrons.dtype,
        )
        source_floor = jnp.asarray(
            self.lit_config.nqs_source_floor,
            dtype=batched_data.data.electrons.dtype,
        )

        def one(data):
            response_log = response_apply(response_params, data)
            ground_log = ground_logpsi(ground_params, data)
            dipole = molecular_electronic_dipole(data, axis)
            source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
            safe_abs_source = jnp.maximum(jnp.abs(source), score_eps)
            source_phase = jnp.where(
                source < 0.0,
                jnp.asarray(jnp.pi, dtype=source.dtype),
                jnp.asarray(0.0, dtype=source.dtype),
            )
            complex_dtype = jnp.result_type(response_log, ground_log, jnp.complex64)
            source_log = (
                jnp.asarray(ground_log, dtype=complex_dtype)
                + jnp.log(safe_abs_source).astype(complex_dtype)
                + 1j * source_phase.astype(complex_dtype)
            )
            sampled_abs_source = jnp.maximum(jnp.abs(source), source_floor)
            source_weight = (
                jnp.abs(source) / jnp.maximum(sampled_abs_source, score_eps)
            ) ** 2
            return (
                jnp.asarray(response_log, dtype=complex_dtype) - source_log,
                source_weight,
            )

        return jax.vmap(one, in_axes=(batched_data.vmap_axis,))(batched_data.data)

    def _source_distillation_scores(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        axis_name: str | None = None,
    ):
        """Return response log-scores and stable response/source ratios."""
        score_eps = float(self.lit_config.nqs_sr_score_eps)

        def one(params, data):
            def split_log_ratio(local_params):
                response_log = response_apply(local_params, data)
                ground_log = ground_logpsi(ground_params, data)
                dipole = molecular_electronic_dipole(data, axis)
                source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
                safe_abs_source = jnp.maximum(jnp.abs(source), score_eps)
                source_phase = jnp.where(source < 0.0, jnp.pi, 0.0)
                complex_dtype = jnp.result_type(
                    response_log,
                    ground_log,
                    jnp.complex64,
                )
                log_ratio = (
                    jnp.asarray(response_log, dtype=complex_dtype)
                    - jnp.asarray(ground_log, dtype=complex_dtype)
                    - jnp.log(safe_abs_source).astype(complex_dtype)
                    - 1j * jnp.asarray(source_phase, dtype=complex_dtype)
                )
                return jnp.stack((jnp.real(log_ratio), jnp.imag(log_ratio))), (
                    log_ratio,
                    source,
                )

            jacobian, (log_ratio, source) = jax.jacrev(
                split_log_ratio,
                has_aux=True,
            )(params)
            score_tree = jax.tree.map(
                lambda leaf: leaf[0] + 1j * leaf[1],
                jacobian,
            )
            return log_ratio, source, score_tree

        log_ratio, source, score_tree = jax.vmap(
            lambda data: one(response_params, data),
            in_axes=(batched_data.vmap_axis,),
        )(batched_data.data)
        score = _flatten_batched_tree(score_tree, log_ratio.shape[0])
        source_floor = jnp.asarray(
            self.lit_config.nqs_source_floor,
            dtype=source.dtype,
        )
        eps = jnp.asarray(score_eps, dtype=source.dtype)
        sampled_abs_source = jnp.maximum(jnp.abs(source), source_floor)
        source_weight = (jnp.abs(source) / jnp.maximum(sampled_abs_source, eps)) ** 2
        finite = (
            jnp.isfinite(jnp.real(log_ratio))
            & jnp.isfinite(jnp.imag(log_ratio))
            & jnp.isfinite(source_weight)
            & jnp.all(
                jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
                axis=1,
            )
        )
        safe_log_real = jnp.where(finite, jnp.real(log_ratio), -jnp.inf)
        log_scale = jnp.max(safe_log_real)
        if axis_name is not None:
            log_scale = jax.lax.pmax(log_scale, axis_name=axis_name)
        log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
        log_scale = jax.lax.stop_gradient(log_scale)
        ratio = jnp.where(
            finite,
            jnp.exp(log_ratio - log_scale),
            jnp.asarray(0.0, dtype=log_ratio.dtype),
        )
        source_weight = jnp.where(finite, source_weight, 0.0)
        score = jnp.where(finite[:, None], score, 0.0)
        return score, ratio, source_weight, log_ratio

    def _evaluate_source_distillation(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        eval_pool,
        *,
        axis: int,
        source_center: float,
    ) -> _SourceDistillationStats:
        """Evaluate initialization fidelity on the independent held-out pool.

        Returns:
            Held-out normalized-overlap, reverse-KL, ESS, and health statistics.

        Raises:
            ValueError: If the evaluation pool is empty or cannot be sharded.
        """
        cache = getattr(self, "_source_distillation_eval_kernel_cache", None)
        if cache is None:
            cache = {}
            self._source_distillation_eval_kernel_cache = cache
        chunk_size = self._nqs_eval_batch_size()
        cache_key = (
            self._nqs_data_parallel_mode(),
            id(response_apply),
            id(ground_logpsi),
            int(axis),
            float(source_center),
            min(int(eval_pool.batch_size), int(chunk_size)),
        )
        data_parallel = self._nqs_data_parallel_enabled()
        kernel = cache.get(cache_key)
        if kernel is None:

            def evaluate(local_params, local_ground_params, local_pool):
                return self._source_distillation_log_ratios(
                    response_apply,
                    (
                        parallel_jax.pvary(local_params)
                        if data_parallel
                        else local_params
                    ),
                    ground_logpsi,
                    local_ground_params,
                    local_pool,
                    axis=axis,
                    source_center=source_center,
                )

            if data_parallel:
                kernel = parallel_jax.jit_sharded(
                    evaluate,
                    in_specs=(
                        parallel_jax.SHARE_PARTITION,
                        parallel_jax.SHARE_PARTITION,
                        eval_pool.partition_spec,
                    ),
                    out_specs=(
                        parallel_jax.DATA_PARTITION,
                        parallel_jax.DATA_PARTITION,
                    ),
                    check_vma=True,
                )
            else:
                kernel = jax.jit(evaluate)
            cache[cache_key] = kernel

        kernel_params = response_params
        kernel_ground_params = ground_params
        if data_parallel:
            kernel_params = _replicate_across_local_devices(response_params)
            kernel_ground_params = _replicate_across_local_devices(ground_params)

        log_ratios = []
        source_weights = []
        for chunk in _batched_data_chunks(eval_pool, chunk_size):
            kernel_chunk = chunk
            if data_parallel:
                self._validate_data_parallel_batch(
                    chunk,
                    purpose="distillation evaluation chunk",
                )
                kernel_chunk = _shard_batched_data_across_local_devices(chunk)
            local_log_ratio, local_source_weight = kernel(
                kernel_params,
                kernel_ground_params,
                kernel_chunk,
            )
            log_ratios.append(local_log_ratio)
            source_weights.append(local_source_weight)
        if not log_ratios:
            raise ValueError("Cannot evaluate source distillation on an empty pool.")
        return _source_distillation_stats_from_log_ratios(
            jnp.concatenate(log_ratios),
            jnp.concatenate(source_weights),
            reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
        )

    def _distill_response_from_source(  # noqa: C901
        self,
        response_apply,
        initial_params,
        ground_logpsi,
        ground_params,
        train_pool,
        eval_pool,
        rng,
        *,
        axis: int,
        source_center: float,
    ):
        """Fit the direct response NQS to ``Phi`` before action optimization.

        Returns:
            Best independently selected response parameters and the unchanged
            workflow random key.

        Raises:
            RuntimeError: If the initial held-out estimator is invalid.
        """
        iterations = int(self.lit_config.nqs_source_distillation_iterations)
        data_parallel = self._nqs_data_parallel_enabled()
        device_count = jax.local_device_count() if data_parallel else 1

        def update_impl(params, local_ground_params, batch, spring_previous):
            score, ratio, source_weight, log_ratio = self._source_distillation_scores(
                response_apply,
                parallel_jax.pvary(params) if data_parallel else params,
                ground_logpsi,
                local_ground_params,
                batch,
                axis=axis,
                source_center=source_center,
                axis_name=(parallel_jax.BATCH_AXIS_NAME if data_parallel else None),
            )
            spring_state = _SpringState(spring_previous)
            if data_parallel:
                updates, spring_state, _, diagnostics = (
                    self._weighted_sr_updates_from_scores_data_parallel(
                        params,
                        score,
                        ratio,
                        source_weight,
                        spring_state,
                        device_count=device_count,
                    )
                )
            else:
                updates, spring_state, _, diagnostics = (
                    self._weighted_sr_updates_from_scores(
                        params,
                        score,
                        ratio,
                        source_weight,
                        spring_state,
                    )
                )
            stats = _source_distillation_stats_from_log_ratios(
                log_ratio,
                source_weight,
                reverse_kl_weight=self.lit_config.nqs_reverse_kl_weight,
                axis_name=(parallel_jax.BATCH_AXIS_NAME if data_parallel else None),
            )
            return (
                _apply_updates(params, updates),
                stats,
                spring_state.previous_direction,
                diagnostics,
            )

        if data_parallel:
            sample_batch = _indexed_batched_data_chunk(
                train_pool,
                self._nqs_train_update_batch_size(),
                0,
            )
            self._validate_data_parallel_batch(
                sample_batch,
                purpose="distillation training",
            )
            update_kernel = parallel_jax.jit_sharded(
                update_impl,
                in_specs=(
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    sample_batch.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                ),
                out_specs=parallel_jax.SHARE_PARTITION,
                check_vma=True,
            )
        else:
            update_kernel = jax.jit(update_impl)

        def evaluate(params):
            return jax.device_get(
                self._evaluate_source_distillation(
                    response_apply,
                    params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    axis=axis,
                    source_center=source_center,
                )
            )

        params = initial_params
        flat_params, _ = ravel_pytree(params)
        spring_previous = jnp.zeros_like(flat_params)
        initial_stats = evaluate(params)
        if not _finite_source_distillation_stats(initial_stats):
            raise RuntimeError(
                f"axis={_AXIS_NAMES[axis]} source distillation has invalid "
                "initial held-out statistics."
            )
        best_params = params
        best_stats = initial_stats
        best_iteration = 0
        plateau = _FidelityPlateauTracker(
            start_iteration=self.lit_config.nqs_fidelity_plateau_start_iteration,
            patience_iterations=(
                self.lit_config.nqs_fidelity_plateau_patience_iterations
            ),
            min_delta=self.lit_config.nqs_fidelity_plateau_min_delta,
        )
        if plateau.start_iteration == 0:
            plateau.observe(0, float(best_stats.fidelity))
        executed_iterations = iterations
        stop_reason = "max_budget"
        selection_interval = int(self.lit_config.nqs_selection_interval)
        forced_evaluations = {iterations}
        if plateau.enabled and 0 < plateau.start_iteration <= iterations:
            forced_evaluations.add(plateau.start_iteration)
        shuffle_seed = self._training_shuffle_seed(
            axis=axis,
            stage="source_distillation",
        )

        for iteration in range(iterations):
            update_batch = _shuffled_batched_data_chunk(
                train_pool,
                self._nqs_train_update_batch_size(),
                iteration,
                seed=shuffle_seed,
            )
            if data_parallel:
                kernel_params = _replicate_across_local_devices(params)
                kernel_ground_params = _replicate_across_local_devices(ground_params)
                kernel_batch = _shard_batched_data_across_local_devices(update_batch)
                kernel_spring = _replicate_across_local_devices(spring_previous)
            else:
                kernel_params = params
                kernel_ground_params = ground_params
                kernel_batch = update_batch
                kernel_spring = spring_previous
            params, train_stats, spring_previous, diagnostics = update_kernel(
                kernel_params,
                kernel_ground_params,
                kernel_batch,
                kernel_spring,
            )
            completed = iteration + 1
            candidate_stats = None
            if completed % selection_interval == 0 or completed in forced_evaluations:
                candidate_stats = evaluate(params)
                if _finite_source_distillation_stats(candidate_stats) and float(
                    candidate_stats.loss
                ) < float(best_stats.loss):
                    best_params = params
                    best_stats = candidate_stats
                    best_iteration = completed
            if (
                self.lit_config.nqs_log_interval > 0
                and completed % self.lit_config.nqs_log_interval == 0
            ):
                host_train, host_diagnostics = jax.device_get(
                    (train_stats, diagnostics)
                )
                logger.info(
                    "axis=%s stage=source_distillation iter=%d "
                    "train_loss=%.6e train_fidelity=%.6f "
                    "train_reverse_kl=%.6e train_ess=%.3f best_iter=%d "
                    "best_loss=%.6e best_fidelity=%.6f best_reverse_kl=%.6e "
                    "best_ess=%.3f",
                    _AXIS_NAMES[axis],
                    completed,
                    float(host_train.loss),
                    float(host_train.fidelity),
                    float(host_train.reverse_kl),
                    float(host_train.reweight_ess_fraction),
                    best_iteration,
                    float(best_stats.loss),
                    float(best_stats.fidelity),
                    float(best_stats.reverse_kl),
                    float(best_stats.reweight_ess_fraction),
                )
                _log_spring_optimizer_diagnostics(
                    host_diagnostics,
                    axis=axis,
                    stage="source_distillation",
                    omega=float("nan"),
                    iteration=completed,
                )
            if (
                candidate_stats is not None
                and _finite_source_distillation_stats(candidate_stats)
                and plateau.observe(completed, float(best_stats.fidelity))
            ):
                executed_iterations = completed
                stop_reason = "fidelity_plateau"
                break

        logger.info(
            "axis=%s stage=source_distillation selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e ess=%.3f "
            "invalid=%.3e initial_fidelity=%.6f fidelity_gain=%+.6e "
            "stop_reason=%s",
            _AXIS_NAMES[axis],
            best_iteration,
            executed_iterations,
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(best_stats.reweight_ess_fraction),
            float(best_stats.invalid_sample_fraction),
            float(initial_stats.fidelity),
            float(best_stats.fidelity) - float(initial_stats.fidelity),
            stop_reason,
        )
        return best_params, rng

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
        physical_loss = _regularized_loss(
            stats,
            self.lit_config.nqs_reverse_kl_weight,
        )
        return stats._replace(loss=physical_loss)

    def _optimize_nqs_frequency(  # noqa: C901
        self,
        update_step,
        initial_params,
        train_pool,
        eval_pool,
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
        """Optimize one frequency until its best held-out fidelity plateaus.

        Model selection and early stopping use one fixed evaluation pool.  The
        plateau clock starts only after the configured baseline iteration and
        resets when the cumulative best fidelity improves by more than the
        configured tolerance.  An unhealthy observation cannot be selected
        and restarts the patience clock.  The best numerically healthy,
        ESS-qualified checkpoint is returned even when the maximum budget is
        exhausted.

        Returns:
            Best parameters, their held-out statistics, selected iteration,
            and the next random key.

        Raises:
            ValueError: If the iteration budget is not positive.
            RuntimeError: If no held-out checkpoint satisfies numerical,
                source-covariance, and estimator-health requirements.
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

        maximum_iterations = int(iterations)
        if maximum_iterations < 1:
            raise ValueError("NQS optimizer iterations must be positive.")
        plateau = _FidelityPlateauTracker(
            start_iteration=(self.lit_config.nqs_fidelity_plateau_start_iteration),
            patience_iterations=(
                self.lit_config.nqs_fidelity_plateau_patience_iterations
            ),
            min_delta=self.lit_config.nqs_fidelity_plateau_min_delta,
        )
        forced_evaluations = {maximum_iterations}
        if plateau.enabled and 0 < plateau.start_iteration <= maximum_iterations:
            forced_evaluations.add(plateau.start_iteration)
        executed_iterations = maximum_iterations
        stop_reason = "max_budget"

        response_params = initial_params
        update_carry = update_step.init_carry(rng, response_params)
        minimum_ess = self.lit_config.nqs_stage_reweight_ess_fraction_min
        initial_stats = jax.device_get(evaluate(response_params))
        latest_stats = initial_stats
        initial_fidelity = float(jax.device_get(initial_stats.fidelity))
        best_params = None
        best_stats = None
        best_iteration = -1
        if _is_selectable_nqs_checkpoint(
            initial_stats,
            min_reweight_ess_fraction=minimum_ess,
        ):
            best_params = response_params
            best_stats = initial_stats
            best_iteration = 0
            if plateau.start_iteration == 0:
                plateau.observe(0, initial_fidelity)
        last_train_stats = None
        last_optimizer_diagnostics = None
        shuffle_seed = self._training_shuffle_seed(
            axis=axis,
            stage=stage,
            omega=float(omega),
        )
        for iteration in range(maximum_iterations):
            chunk_index = iteration
            if train_pool is not None:
                chunk_index = _shuffled_batched_data_chunk_index(
                    train_pool.batch_size,
                    self._nqs_train_update_batch_size(),
                    iteration,
                    seed=shuffle_seed,
                )
            response_params, last_train_stats, update_carry = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(omega)),
                update_carry,
                chunk_index,
            )
            last_optimizer_diagnostics = getattr(
                update_step,
                "last_spring_optimizer_diagnostics",
                None,
            )
            completed = iteration + 1
            should_select = (
                completed % self.lit_config.nqs_selection_interval == 0
                or completed in forced_evaluations
            )
            candidate_stats = None
            if should_select:
                candidate_stats = jax.device_get(evaluate(response_params))
                latest_stats = candidate_stats
                candidate_selectable = _is_selectable_nqs_checkpoint(
                    candidate_stats,
                    min_reweight_ess_fraction=minimum_ess,
                )
                if not candidate_selectable:
                    plateau.defer(completed)
                if candidate_selectable and (
                    best_stats is None
                    or _is_better_nqs_checkpoint(
                        candidate_stats,
                        best_stats,
                        min_reweight_ess_fraction=minimum_ess,
                    )
                ):
                    best_params = response_params
                    best_stats = candidate_stats
                    best_iteration = completed
            if (
                self.lit_config.nqs_log_interval > 0
                and completed % self.lit_config.nqs_log_interval == 0
            ):
                reported_stats = best_stats if best_stats is not None else latest_stats
                host_train_stats, host_optimizer_diagnostics = jax.device_get(
                    (last_train_stats, last_optimizer_diagnostics)
                )
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d train_loss=%.6e "
                    "train_fidelity=%.6f train_reverse_kl=%.6e train_ess=%.3f "
                    "best_iter=%d best_fidelity=%.6f best_reverse_kl=%.6e "
                    "best_ess=%.3f",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(host_train_stats.loss),
                    float(host_train_stats.fidelity),
                    float(host_train_stats.reverse_kl),
                    float(host_train_stats.reweight_ess_fraction),
                    best_iteration,
                    float(reported_stats.fidelity),
                    float(reported_stats.reverse_kl),
                    float(reported_stats.reweight_ess_fraction),
                )
                _log_spring_optimizer_diagnostics(
                    host_optimizer_diagnostics,
                    axis=axis,
                    stage=stage,
                    omega=float(omega),
                    iteration=completed,
                )
            if (
                candidate_stats is not None
                and best_stats is not None
                and _is_selectable_nqs_checkpoint(
                    candidate_stats,
                    min_reweight_ess_fraction=minimum_ess,
                )
                and plateau.observe(
                    completed,
                    float(jax.device_get(best_stats.fidelity)),
                )
            ):
                executed_iterations = completed
                stop_reason = "fidelity_plateau"
                plateau_reference_fidelity = plateau.reference_fidelity
                plateau_last_significant_iteration = plateau.last_significant_iteration
                if (
                    plateau_reference_fidelity is None
                    or plateau_last_significant_iteration is None
                ):
                    msg = "plateau stop requires an initialized fidelity reference"
                    raise RuntimeError(msg)
                logger.info(
                    "axis=%s stage=%s omega=%.6f iter=%d action=plateau_stop "
                    "best_fidelity=%.6f reference_fidelity=%.6f "
                    "last_significant_iter=%d patience=%d min_delta=%.3e",
                    _AXIS_NAMES[axis],
                    stage,
                    float(omega),
                    completed,
                    float(best_stats.fidelity),
                    plateau_reference_fidelity,
                    plateau_last_significant_iteration,
                    plateau.patience_iterations,
                    plateau.min_delta,
                )
                break
        if best_stats is None or best_params is None:
            msg = (
                f"axis={_AXIS_NAMES[axis]} stage={stage} omega={float(omega):.6f} "
                "produced no healthy held-out checkpoint; ESS="
                f"{float(latest_stats.reweight_ess_fraction):.6f}, required ESS="
                f"{minimum_ess:.6f}."
            )
            raise RuntimeError(msg)
        _require_nqs_stage_health(
            best_stats,
            min_reweight_ess_fraction=minimum_ess,
            context=(
                f"axis={_AXIS_NAMES[axis]} stage={stage} "
                f"omega={float(omega):.6f} failed held-out estimator health"
            ),
        )
        rng = update_carry.rng
        logger.info(
            "axis=%s stage=%s omega=%.6f selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e "
            "ess=%.3f initial_fidelity=%.6f fidelity_gain=%+.6e "
            "stop_reason=%s required_ess=%.3f",
            _AXIS_NAMES[axis],
            stage,
            float(omega),
            best_iteration,
            executed_iterations,
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(best_stats.reweight_ess_fraction),
            initial_fidelity,
            float(best_stats.fidelity) - initial_fidelity,
            stop_reason,
            minimum_ess,
        )
        return best_params, best_stats, best_iteration, rng

    def _continuation_checkpoint_save_dir(self, axis: int) -> UPath | None:
        save_path = getattr(self, "save_path", None)
        if save_path is None:
            return None
        return (
            UPath(save_path)
            / "continuation_checkpoints"
            / (f"axis_{_AXIS_NAMES[axis]}")
        )

    def _continuation_checkpoint_restore_path(
        self,
        axis: int,
    ) -> tuple[UPath | None, bool]:
        """Resolve one optional explicit or same-run continuation restore path.

        Returns:
            Resolved path and whether that exact axis checkpoint is required.
            A configured run/checkpoint root is optional per axis so axes not
            reached before an interruption can start fresh.
        """
        configured = str(self.lit_config.nqs_continuation_restore_path).strip()
        explicit = bool(configured)
        if not explicit:
            return self._continuation_checkpoint_save_dir(axis), False

        root = UPath(configured)
        if root.suffix == ".npz":
            return root, True
        axis_name = f"axis_{_AXIS_NAMES[axis]}"
        if root.name == axis_name:
            return root, True
        if not root.exists():
            return root / axis_name, True
        nested_root = root / "continuation_checkpoints"
        if nested_root.exists():
            root = nested_root
        return root / axis_name, False

    def _load_nqs_continuation_checkpoint(
        self,
        template: _ContinuationCheckpoint,
        *,
        axis: int,
        state_fingerprint: str,
        full_config_digest: str,
    ) -> tuple[UPath, _ContinuationCheckpoint] | None:
        """Load one structurally compatible latest checkpoint, if present.

        Returns:
            Checkpoint path and restored bundle, or ``None`` for a fresh run.

        Raises:
            RuntimeError: If an explicit checkpoint is absent or a discovered
                checkpoint has incompatible metadata or tree structure.
        """
        restore_path, required = self._continuation_checkpoint_restore_path(axis)
        if restore_path is None:
            return None
        checkpoint_path = _latest_continuation_checkpoint_path(restore_path)
        if checkpoint_path is None:
            if required:
                msg = (
                    "No readable continuation checkpoint found for axis="
                    f"{_AXIS_NAMES[axis]} at explicit restore path {restore_path}."
                )
                raise RuntimeError(msg)
            return None

        metadata = _read_continuation_checkpoint_metadata(checkpoint_path)
        if metadata["schema_version"] != _CONTINUATION_CHECKPOINT_SCHEMA_VERSION:
            msg = (
                f"Continuation checkpoint {checkpoint_path} uses schema "
                f"{metadata['schema_version']}, expected "
                f"{_CONTINUATION_CHECKPOINT_SCHEMA_VERSION}."
            )
            raise RuntimeError(msg)
        if metadata["state_fingerprint"] != state_fingerprint:
            msg = (
                f"Continuation checkpoint {checkpoint_path} is incompatible "
                "with the current physical/ansatz state fingerprint "
                f"({metadata['state_fingerprint']} != {state_fingerprint})."
            )
            raise RuntimeError(msg)
        if metadata["full_config_digest"] != full_config_digest:
            logger.warning(
                "Restoring continuation checkpoint %s with a changed non-state "
                "configuration; current gates will be re-applied (%s != %s)",
                checkpoint_path,
                metadata["full_config_digest"],
                full_config_digest,
            )

        save_dir = self._continuation_checkpoint_save_dir(axis)
        if save_dir is None:
            save_dir = checkpoint_path.parent
        manager = NumPyCheckpointManager(
            save_dir,
            checkpoint_path,
            prefix=_CONTINUATION_CHECKPOINT_PREFIX,
        )
        try:
            initial_step, restored = manager.restore(template)
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"Continuation checkpoint {checkpoint_path} has an incompatible tree."
            raise RuntimeError(msg) from exc
        if initial_step != int(restored.accepted_points):
            msg = (
                f"Continuation checkpoint {checkpoint_path} has inconsistent "
                f"step/count metadata ({initial_step} != "
                f"{int(restored.accepted_points)})."
            )
            raise RuntimeError(msg)
        return checkpoint_path, restored

    def _restore_nqs_continuation_checkpoint(
        self,
        response_params_template,
        rng_template,
        eval_pool,
        *,
        response_apply,
        ground_logpsi,
        ground_params,
        axis: int,
        source_center: float,
        source_norm: float,
        ground_energy: float,
        ground_checkpoint_step: int,
        response_parity: int,
        target_omega: float,
        state_fingerprint: str,
        full_config_digest: str,
    ) -> _ContinuationResumeState | None:
        """Restore and revalidate the latest complete bridge checkpoint.

        Returns:
            Validated resume state, or ``None`` when an implicit same-run
            checkpoint does not exist.

        Raises:
            RuntimeError: If an explicit checkpoint is absent or a discovered
                checkpoint is incompatible, malformed, or fails current gates.
        """
        template = _ContinuationCheckpoint(
            schema_version=_CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
            axis=int(axis),
            target_omega=float(target_omega),
            current_omega=float(self.lit_config.nqs_warm_start_omega or target_omega),
            accepted_points=0,
            ground_checkpoint_step=int(ground_checkpoint_step),
            ground_energy=float(ground_energy),
            source_center=float(source_center),
            source_norm=float(source_norm),
            response_parity=int(response_parity),
            response_params=response_params_template,
            rng=rng_template,
            current_stats=_empty_nqs_lit_stats(),
            history_json="[]",
            warm_start_selected_iteration=0,
        )
        loaded = self._load_nqs_continuation_checkpoint(
            template,
            axis=axis,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
        )
        if loaded is None:
            return None
        checkpoint_path, restored = loaded
        if int(restored.axis) != int(axis):
            msg = (
                f"Continuation checkpoint {checkpoint_path} is for axis="
                f"{int(restored.axis)}, expected {axis}."
            )
            raise RuntimeError(msg)

        current_omega = float(restored.current_omega)
        start_omega = self.lit_config.nqs_warm_start_omega
        if (
            start_omega is None
            or current_omega < float(start_omega)
            or current_omega >= float(target_omega)
        ):
            msg = (
                f"Continuation checkpoint {checkpoint_path} has current omega "
                f"{current_omega:.8g} outside the resumable interval "
                f"[{start_omega}, {target_omega})."
            )
            raise RuntimeError(msg)

        records = _continuation_records_from_json(
            str(restored.history_json),
            stats_template=restored.current_stats,
        )
        optimized_count = sum(record.optimized for record in records)
        if optimized_count != int(restored.accepted_points):
            msg = (
                f"Continuation checkpoint {checkpoint_path} history contains "
                f"{optimized_count} optimized points, expected "
                f"{int(restored.accepted_points)}."
            )
            raise RuntimeError(msg)
        if not records or not records[-1].optimized:
            msg = (
                f"Continuation checkpoint {checkpoint_path} has no latest-good record."
            )
            raise RuntimeError(msg)
        if not np.isclose(records[-1].omega, current_omega, rtol=0.0, atol=1e-12):
            msg = (
                f"Continuation checkpoint {checkpoint_path} history ends at "
                f"{records[-1].omega:.8g}, not current omega {current_omega:.8g}."
            )
            raise RuntimeError(msg)

        revalidated_stats = jax.device_get(
            self._evaluate_nqs_checkpoint(
                response_apply=response_apply,
                response_params=restored.response_params,
                ground_logpsi=ground_logpsi,
                ground_params=ground_params,
                eval_pool=eval_pool,
                axis=axis,
                source_center=source_center,
                source_norm=source_norm,
                ground_energy=ground_energy,
                omega=current_omega,
            )
        )
        _require_eligible_nqs_checkpoint(
            revalidated_stats,
            context=(
                f"Restored continuation checkpoint at omega={current_omega:.8g} "
                "failed current numerical validation"
            ),
        )
        _require_nqs_stage_health(
            revalidated_stats,
            min_reweight_ess_fraction=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            context=(
                f"Restored continuation checkpoint at omega={current_omega:.8g} "
                "failed current estimator-health validation"
            ),
        )
        records[-1] = records[-1]._replace(stats=revalidated_stats)
        logger.info(
            "Restored axis=%s continuation checkpoint %s omega=%.6f "
            "bridge_points=%d stored_fidelity=%.6f revalidated_fidelity=%.6f "
            "revalidated_ess=%.3f",
            _AXIS_NAMES[axis],
            checkpoint_path,
            current_omega,
            optimized_count,
            float(restored.current_stats.fidelity),
            float(revalidated_stats.fidelity),
            float(revalidated_stats.reweight_ess_fraction),
        )
        return _ContinuationResumeState(
            response_params=restored.response_params,
            rng=restored.rng,
            current_stats=revalidated_stats,
            current_omega=current_omega,
            records=tuple(records),
            warm_start_selected_iteration=int(restored.warm_start_selected_iteration),
        )

    def _save_nqs_continuation_checkpoint(
        self,
        response_params,
        current_stats,
        rng,
        current_omega: float,
        records: list[_ContinuationRecord],
        *,
        axis: int,
        target_omega: float,
        ground_checkpoint_step: int,
        ground_energy: float,
        source_center: float,
        source_norm: float,
        response_parity: int,
        state_fingerprint: str,
        full_config_digest: str,
        warm_start_selected_iteration: int,
    ) -> None:
        """Atomically persist one optimized, numerically healthy bridge."""
        if jax.process_index() != 0:
            return
        save_dir = self._continuation_checkpoint_save_dir(axis)
        if save_dir is None:
            return
        accepted_points = sum(record.optimized for record in records)
        if accepted_points <= 0 or not records or not records[-1].optimized:
            return
        checkpoint = _ContinuationCheckpoint(
            schema_version=_CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
            state_fingerprint=state_fingerprint,
            full_config_digest=full_config_digest,
            axis=int(axis),
            target_omega=float(target_omega),
            current_omega=float(current_omega),
            accepted_points=int(accepted_points),
            ground_checkpoint_step=int(ground_checkpoint_step),
            ground_energy=float(ground_energy),
            source_center=float(source_center),
            source_norm=float(source_norm),
            response_parity=int(response_parity),
            response_params=response_params,
            rng=rng,
            current_stats=current_stats,
            history_json=_continuation_records_to_json(records),
            warm_start_selected_iteration=int(warm_start_selected_iteration),
        )
        manager = NumPyCheckpointManager(
            save_dir,
            prefix=_CONTINUATION_CHECKPOINT_PREFIX,
        )
        checkpoint_path = manager.save(accepted_points - 1, jax.device_get(checkpoint))
        logger.info(
            "Saved axis=%s continuation checkpoint %s omega=%.6f "
            "bridge_points=%d fidelity=%.6f ess=%.3f",
            _AXIS_NAMES[axis],
            checkpoint_path,
            current_omega,
            accepted_points,
            float(current_stats.fidelity),
            float(current_stats.reweight_ess_fraction),
        )

    def _require_continuation_probe_recovery_allowed(
        self,
        current_stats,
        probe_stats,
        *,
        candidate_omega: float,
        min_step_override: bool,
    ) -> None:
        """Reject an unsafe or explicitly disabled minimum-step recovery.

        Raises:
            RuntimeError: If held-out ESS is too low or minimum-step recovery
                was explicitly disabled.
        """
        if not min_step_override:
            return
        probe_ess_failure = _nqs_stage_ess_failure(
            probe_stats,
            min_reweight_ess_fraction=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
        )
        if probe_ess_failure is not None:
            msg = (
                "Frequency continuation reached its minimum step with "
                "insufficient held-out importance-sampling ESS; refusing to "
                "optimize from an unreliable estimator at "
                f"omega={candidate_omega:.8g}: "
                f"{probe_ess_failure}."
            )
            raise RuntimeError(msg)
        if self.lit_config.nqs_continuation_allow_min_step_override:
            return
        current_fidelity = float(jax.device_get(current_stats.fidelity))
        candidate_fidelity = float(jax.device_get(probe_stats.fidelity))
        candidate_ess = float(jax.device_get(probe_stats.reweight_ess_fraction))
        required_probe_fidelity = max(
            self.lit_config.nqs_continuation_fidelity_retention * current_fidelity,
            0.0,
        )
        msg = (
            "Frequency continuation reached its minimum step without an "
            "acceptable inherited checkpoint and recovery is disabled at "
            f"omega={candidate_omega:.8g}: fidelity={candidate_fidelity:.6f}, "
            f"required relative fidelity={required_probe_fidelity:.6f}, ESS "
            f"fraction={candidate_ess:.6f}, required ESS="
            f"{self.lit_config.nqs_stage_reweight_ess_fraction_min:.6f}."
        )
        raise RuntimeError(msg)

    def _continue_nqs_to_spectrum(
        self,
        update_step,
        response_params,
        current_stats,
        train_pool,
        eval_pool,
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
        resume_omega: float | None = None,
        existing_records: tuple[_ContinuationRecord, ...] = (),
        checkpoint_callback: Callable[
            [Any, Any, Any, float, list[_ContinuationRecord]],
            None,
        ]
        | None = None,
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
        start_omega = (
            float(resume_omega)
            if resume_omega is not None
            else self.lit_config.nqs_warm_start_omega
        )
        if start_omega is None or float(start_omega) >= float(target_omega):
            return response_params, current_stats, list(existing_records), rng

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
            current_stats = jax.device_get(
                self._evaluate_nqs_checkpoint(
                    response_params=response_params,
                    eval_pool=eval_pool,
                    omega=current_omega,
                    **common,
                )
            )
        _require_eligible_nqs_checkpoint(
            current_stats,
            context=(
                "Frequency continuation received an ineligible starting "
                f"checkpoint at omega={current_omega:.8g}"
            ),
        )
        _require_nqs_stage_health(
            current_stats,
            min_reweight_ess_fraction=(
                self.lit_config.nqs_stage_reweight_ess_fraction_min
            ),
            context=(
                "Frequency continuation received an estimator-unhealthy "
                f"starting checkpoint at omega={current_omega:.8g}"
            ),
        )
        min_step = _continuation_min_step(self.lit_config, spectrum_omega)
        records = list(existing_records)
        existing_optimized_count = sum(record.optimized for record in records)
        if existing_optimized_count > self.lit_config.nqs_continuation_max_points:
            msg = (
                "Restored frequency continuation already contains "
                f"{existing_optimized_count} bridge points, exceeding the current "
                f"limit {self.lit_config.nqs_continuation_max_points}."
            )
            raise RuntimeError(msg)
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(target_omega)) * 8.0

        def evaluate_probe(omega: float):
            return jax.device_get(
                self._evaluate_nqs_checkpoint(
                    response_params=response_params,
                    eval_pool=eval_pool,
                    omega=omega,
                    **common,
                )
            )

        while target_omega - current_omega > tolerance:
            gap = float(target_omega - current_omega)
            physics_step = _physics_continuation_step(
                current_stats,
                gap=gap,
                fraction=self.lit_config.nqs_continuation_step_fraction,
                min_step=min_step,
            )
            history_cap = _continuation_history_step_cap(
                records,
                growth_factor=(self.lit_config.nqs_continuation_step_growth_factor),
                min_step=min_step,
            )
            step = min(
                physics_step,
                gap if history_cap is None else history_cap,
            )
            optimized_count = sum(record.optimized for record in records)
            capacity = _continuation_capacity_diagnostics(
                remaining_gap=gap,
                optimized_count=optimized_count,
                maximum=self.lit_config.nqs_continuation_max_points,
                chosen_step=step,
            )
            logger.info(
                "axis=%s continuation_proposal current_omega=%.6f "
                "physics_step=%.6e history_cap=%.6e chosen_step=%.6e "
                "remaining_gap=%.6e optimized_points=%d "
                "remaining_bridge_slots=%d required_mean_step=%.6e "
                "capacity_ratio=%.3f",
                _AXIS_NAMES[axis],
                current_omega,
                physics_step,
                float("nan") if history_cap is None else history_cap,
                step,
                capacity.remaining_gap,
                optimized_count,
                capacity.remaining_bridge_slots,
                capacity.required_mean_step,
                capacity.capacity_ratio,
            )
            candidate_omega = current_omega + step
            probe_stats, candidate_omega, probe_ok, bisections = (
                _bisect_continuation_probe(
                    evaluate_probe,
                    current_stats,
                    current_omega=current_omega,
                    candidate_omega=candidate_omega,
                    target_omega=target_omega,
                    min_step=min_step,
                    retention=(self.lit_config.nqs_continuation_fidelity_retention),
                    min_reweight_ess_fraction=(
                        self.lit_config.nqs_stage_reweight_ess_fraction_min
                    ),
                )
            )

            _require_eligible_nqs_checkpoint(
                probe_stats,
                context=(
                    "Frequency continuation produced non-finite/invalid "
                    "held-out statistics at "
                    f"omega={candidate_omega:.8g}; refusing to propagate "
                    "a corrupted checkpoint"
                ),
            )
            actual_step = float(candidate_omega - current_omega)
            min_step_override = not probe_ok and actual_step <= min_step * (1.0 + 1e-12)
            self._require_continuation_probe_recovery_allowed(
                current_stats,
                probe_stats,
                candidate_omega=candidate_omega,
                min_step_override=min_step_override,
            )
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
                    "axis=%s continuation_probe target=%.6f "
                    "inherited_fidelity=%.6f step=%.6e bisections=%d "
                    "accepted=%s min_step_override=%s",
                    _AXIS_NAMES[axis],
                    target_omega,
                    float(probe_stats.fidelity),
                    actual_step,
                    bisections,
                    probe_ok,
                    min_step_override,
                )
                return response_params, current_stats, records, rng

            _require_continuation_point_capacity(
                optimized_count,
                maximum=self.lit_config.nqs_continuation_max_points,
                target_omega=target_omega,
            )
            inherited_fidelity = float(probe_stats.fidelity)
            (
                response_params,
                current_stats,
                selected_iteration,
                rng,
            ) = self._optimize_nqs_frequency(
                update_step,
                response_params,
                train_pool,
                eval_pool,
                rng,
                omega=candidate_omega,
                iterations=self.lit_config.nqs_continuation_iterations,
                stage="continuation",
                **common,
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
            if checkpoint_callback is not None:
                checkpoint_callback(
                    response_params,
                    current_stats,
                    rng,
                    current_omega,
                    records,
                )

        return response_params, current_stats, records, rng

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        eval_pool,
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
                tolerance=float(self.lit_config.nqs_sector_tolerance),
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
            tolerance=float(self.lit_config.nqs_sector_tolerance),
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

    def _make_nqs_update_step(
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
        data_parallel = self._nqs_data_parallel_enabled()
        data_parallel_device_count = jax.local_device_count()
        data_parallel_ground_params = (
            _replicate_across_local_devices(ground_params)
            if data_parallel
            else ground_params
        )

        def source_update_impl(
            response_params,
            local_ground_params,
            batched_data,
            spring_previous,
            omega,
        ):
            if data_parallel:
                (
                    stats,
                    updates,
                    spring_state,
                    _,
                    optimizer_diagnostics,
                ) = self._source_sr_stats_and_updates_data_parallel(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    local_ground_params,
                    batched_data,
                    spring_state=_SpringState(spring_previous),
                    axis=axis,
                    source_center=source_center,
                    source_norm=source_norm,
                    ground_energy=ground_energy,
                    omega=omega,
                    device_count=data_parallel_device_count,
                )
            else:
                (
                    stats,
                    updates,
                    spring_state,
                    _,
                    optimizer_diagnostics,
                ) = self._source_sr_stats_and_updates(
                    response_apply,
                    response_params,
                    ground_logpsi,
                    local_ground_params,
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
                optimizer_diagnostics,
            )

        source_update_kernel = None if data_parallel else jax.jit(source_update_impl)

        def update(
            response_params,
            batched_data,
            omega,
            update_carry,
            batch_index: int = 0,
        ):
            nonlocal source_update_kernel
            update_batch = _indexed_batched_data_chunk(
                batched_data,
                self._nqs_train_update_batch_size(),
                batch_index,
            )
            if source_update_kernel is None:
                source_update_kernel = self._data_parallel_source_update_kernel(
                    source_update_impl,
                    update_batch,
                )
            kernel_response_params = response_params
            kernel_ground_params = data_parallel_ground_params
            kernel_batch = update_batch
            kernel_spring_previous = update_carry.spring.previous_direction
            kernel_omega = omega
            if data_parallel:
                kernel_response_params = _replicate_across_local_devices(
                    kernel_response_params
                )
                kernel_batch = _shard_batched_data_across_local_devices(kernel_batch)
                kernel_spring_previous = _replicate_across_local_devices(
                    kernel_spring_previous
                )
                kernel_omega = _replicate_across_local_devices(kernel_omega)
            (
                response_params,
                stats,
                spring_previous,
                optimizer_diagnostics,
            ) = source_update_kernel(
                kernel_response_params,
                kernel_ground_params,
                kernel_batch,
                kernel_spring_previous,
                kernel_omega,
            )
            update.last_spring_optimizer_diagnostics = (  # type: ignore[attr-defined]
                optimizer_diagnostics
            )
            return (
                response_params,
                stats,
                update_carry._replace(spring=_SpringState(spring_previous)),
            )

        update.init_carry = self._init_nqs_update_carry  # type: ignore[attr-defined]
        update.last_spring_optimizer_diagnostics = None  # type: ignore[attr-defined]
        return update

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
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
            )
        )
        return (
            stats,
            updates,
            spring_state,
            damping,
            optimizer_diagnostics,
        )

    def _source_sr_stats_and_updates_data_parallel(
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
        device_count: int,
    ):
        """Evaluate one exact global source update from local batch shards.

        Returns:
            Global statistics, replicated updates and SPRING state, damping,
            and optimizer diagnostics.
        """
        score, ratio, source_weight, local_source_sums = (
            self._source_sampled_action_scores_and_sums(
                response_apply,
                parallel_jax.pvary(response_params),
                ground_logpsi,
                ground_params,
                batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
        )
        source_sums = _merge_source_sums_across_devices(local_source_sums)
        stats = nqs_lit_stats_from_source_sums(
            source_sums,
            source_norm=source_norm,
            omega=omega,
            eta=self.lit_config.eta,
        )
        updates, spring_state, damping, optimizer_diagnostics = (
            self._weighted_sr_updates_from_scores_data_parallel(
                response_params,
                score,
                ratio,
                source_weight,
                spring_state,
                device_count=device_count,
            )
        )
        return (
            stats,
            updates,
            spring_state,
            damping,
            optimizer_diagnostics,
        )

    def _weighted_sr_updates_from_scores_data_parallel(
        self,
        response_params,
        score,
        ratio,
        source_weight,
        spring_state: _SpringState,
        *,
        device_count: int,
    ):
        """Return a replicated SPRING update from row-sharded action scores."""
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
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )
        weighted_score = jnp.sqrt(psi_weight)[:, None] * centered_score
        local_score_aug = jnp.concatenate(
            [weighted_score.real, weighted_score.imag],
            axis=0,
        )
        qfi_trace = jax.lax.psum(
            jnp.sum(local_score_aug**2),
            axis_name=parallel_jax.BATCH_AXIS_NAME,
        )
        centering_null = jnp.sqrt(psi_weight)
        zero_null = jnp.zeros_like(centering_null)
        local_kernel_null_vectors = jnp.stack(
            [
                jnp.concatenate([centering_null, zero_null]),
                jnp.concatenate([zero_null, centering_null]),
            ]
        )
        previous_direction = spring_state.previous_direction
        direction, spring_state, damping = _spring_direction_data_parallel(
            local_score_aug,
            grad_flat,
            spring_state,
            epsilon_scale=self.lit_config.nqs_spring_epsilon,
            damping_floor=self.lit_config.nqs_spring_damping_floor,
            decay=self.lit_config.nqs_spring_decay,
            device_count=device_count,
            qfi_trace=qfi_trace,
            local_kernel_null_vectors=local_kernel_null_vectors,
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
        response_over_source_moments = weighted_complex_moments(
            response_over_source,
            stats_source_weight,
        )
        hbar_over_source_moments = weighted_complex_moments(
            hbar_over_source,
            stats_source_weight,
        )
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
            ground_energy_sum=jnp.real(jnp.sum(eloc_response)),
            response_over_source_moments=response_over_source_moments,
            hbar_over_source_moments=hbar_over_source_moments,
            psi_weight_max=jnp.max(psi_weight_unnormalized),
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

    def _nqs_chunk_sums_kernel(
        self,
        response_apply,
        ground_logpsi,
        batched_data,
        *,
        axis: int,
    ):
        """Return a persistent compiled kernel for held-out source sums.

        JAX caches a compiled executable by jitted callable identity.  Creating
        the callable inside :meth:`_nqs_stats_chunked` therefore retraced and
        recompiled the same expensive local-action kernel at every checkpoint
        selection.  Cache one callable per response/ground closure and static
        dipole axis; all numerical values remain dynamic arguments so the same
        executable is reused across checkpoints and frequencies.

        Returns:
            A jitted callable that evaluates additive NQS-LIT source sums for
            one fixed-shape chunk.
        """
        cache = getattr(self, "_nqs_chunk_sums_kernel_cache", None)
        if cache is None:
            cache = {}
            self._nqs_chunk_sums_kernel_cache = cache
        data_parallel = self._nqs_data_parallel_enabled()
        device_count = jax.local_device_count() if data_parallel else 1
        key = (
            id(response_apply),
            id(ground_logpsi),
            int(axis),
            data_parallel,
            device_count,
        )
        cached = cache.get(key)
        if (
            cached is not None
            and cached[0] is response_apply
            and cached[1] is ground_logpsi
        ):
            return cached[2]

        def chunk_sums_impl(
            local_params,
            local_ground_params,
            chunk,
            local_source_center,
            local_ground_energy,
            local_omega,
            local_eta,
            local_source_floor,
        ):
            sums = nqs_lit_source_sampled_sums(
                response_apply,
                local_params,
                ground_logpsi,
                local_ground_params,
                chunk,
                axis=axis,
                source_center=local_source_center,
                ground_energy=local_ground_energy,
                omega=local_omega,
                eta=local_eta,
                source_floor=local_source_floor,
            )
            if data_parallel:
                sums = _merge_source_sums_across_devices(sums)
            return sums

        if data_parallel:
            chunk_sums = parallel_jax.jit_sharded(
                chunk_sums_impl,
                in_specs=(
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    batched_data.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                ),
                out_specs=parallel_jax.SHARE_PARTITION,
                check_vma=True,
            )
            logger.info(
                "Configured held-out NQS-LIT data parallelism devices=%d",
                device_count,
            )
        else:
            chunk_sums = jax.jit(chunk_sums_impl)

        cache[key] = (response_apply, ground_logpsi, chunk_sums)
        return chunk_sums

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
        return_block_sums: bool = False,
    ):
        chunk_size = self._nqs_eval_batch_size()
        self._validate_data_parallel_batch(
            batched_data,
            purpose="held-out evaluation pool",
        )
        chunk_sums = self._nqs_chunk_sums_kernel(
            response_apply,
            ground_logpsi,
            batched_data,
            axis=axis,
        )
        source_center_array = jnp.asarray(source_center)
        ground_energy_array = jnp.asarray(ground_energy)
        omega_array = jnp.asarray(omega)
        eta_array = jnp.asarray(self.lit_config.eta)
        source_floor_array = jnp.asarray(self.lit_config.nqs_source_floor)
        kernel_response_params = response_params
        kernel_ground_params = ground_params
        kernel_source_center = source_center_array
        kernel_ground_energy = ground_energy_array
        kernel_omega = omega_array
        kernel_eta = eta_array
        kernel_source_floor = source_floor_array
        if self._nqs_data_parallel_enabled():
            kernel_response_params = _replicate_across_local_devices(response_params)
            kernel_ground_params = _replicate_across_local_devices(ground_params)
            kernel_source_center = _replicate_across_local_devices(source_center_array)
            kernel_ground_energy = _replicate_across_local_devices(ground_energy_array)
            kernel_omega = _replicate_across_local_devices(omega_array)
            kernel_eta = _replicate_across_local_devices(eta_array)
            kernel_source_floor = _replicate_across_local_devices(source_floor_array)

        total_sums = None
        block_sums = []
        for chunk in _batched_data_chunks(batched_data, chunk_size):
            self._validate_data_parallel_batch(chunk, purpose="evaluation")
            kernel_chunk = (
                _shard_batched_data_across_local_devices(chunk)
                if self._nqs_data_parallel_enabled()
                else chunk
            )
            sums = chunk_sums(
                kernel_response_params,
                kernel_ground_params,
                kernel_chunk,
                kernel_source_center,
                kernel_ground_energy,
                kernel_omega,
                kernel_eta,
                kernel_source_floor,
            )
            if return_block_sums:
                block_sums.append(sums)
            total_sums = (
                sums if total_sums is None else _add_source_sums(total_sums, sums)
            )
        if total_sums is None:
            msg = "Cannot evaluate NQS-LIT stats with an empty source pool."
            raise ValueError(msg)
        stats = _nqs_stats_from_source_sums(
            total_sums,
            jnp.asarray(source_norm),
            omega_array,
            eta_array,
        )
        if return_block_sums:
            return stats, tuple(block_sums)
        return stats

    def _log_response_pool_capacity(
        self,
        response_params,
        train_pool: BatchedData,
        eval_pool: BatchedData,
        *,
        axis: int,
    ) -> None:
        parameter_count = int(ravel_pytree(response_params)[0].size)
        train_walkers = int(train_pool.batch_size)
        eval_walkers = int(eval_pool.batch_size)
        train_batch = self._nqs_train_update_batch_size()
        eval_batch = self._nqs_eval_batch_size()
        device_count = (
            jax.local_device_count() if self._nqs_data_parallel_enabled() else 1
        )
        denominator = max(parameter_count, 1)
        logger.info(
            "axis=%s response_parameter_count=%d raw_train_walkers=%d "
            "raw_eval_walkers=%d train_walkers_per_parameter=%.3f "
            "eval_walkers_per_parameter=%.3f global_train_batch=%d "
            "per_device_train_batch=%d global_eval_batch=%d "
            "per_device_eval_batch=%d devices=%d",
            _AXIS_NAMES[axis],
            parameter_count,
            train_walkers,
            eval_walkers,
            train_walkers / denominator,
            eval_walkers / denominator,
            train_batch,
            train_batch // device_count,
            eval_batch,
            eval_batch // device_count,
            device_count,
        )
        if train_walkers < parameter_count:
            logger.warning(
                "axis=%s fixed response train pool has fewer raw walkers (%d) "
                "than response parameters (%d); held-out overfitting risk is high",
                _AXIS_NAMES[axis],
                train_walkers,
                parameter_count,
            )

    def _nqs_train_update_batch_size(self) -> int:
        return self._nqs_effective_batch_size(
            global_size=self.lit_config.nqs_train_update_batch_size,
            per_device_size=(self.lit_config.nqs_train_update_batch_size_per_device),
        )

    def _nqs_eval_batch_size(self) -> int:
        return self._nqs_effective_batch_size(
            global_size=self.lit_config.nqs_eval_batch_size,
            per_device_size=self.lit_config.nqs_eval_batch_size_per_device,
        )

    def _nqs_effective_batch_size(
        self,
        *,
        global_size: int,
        per_device_size: int,
    ) -> int:
        """Resolve a configured global or per-device walker count.

        Returns:
            The global batch size used by the fixed-pool kernels.

        Raises:
            ValueError: If both global and per-device sizes are positive.
        """
        configured_global = int(global_size)
        configured_per_device = int(per_device_size)
        if configured_global > 0 and configured_per_device > 0:
            raise ValueError("Global and per-device NQS batch sizes are exclusive.")
        if configured_global > 0:
            return configured_global
        if configured_per_device > 0:
            device_count = (
                jax.local_device_count() if self._nqs_data_parallel_enabled() else 1
            )
            return configured_per_device * device_count
        return max(1, int(self.config.batch_size))

    def _training_shuffle_seed(
        self,
        *,
        axis: int,
        stage: str,
        omega: float | None = None,
    ) -> int:
        configured_seed = getattr(getattr(self, "config", None), "seed", None)
        base_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int(getattr(self, "_run_seed", 0))
        )
        frequency = "none" if omega is None else float(omega).hex()
        payload = f"{base_seed}:{int(axis)}:{stage}:{frequency}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def _nqs_data_parallel_enabled(self) -> bool:
        return self._nqs_data_parallel_mode() == "local_devices"

    def _nqs_data_parallel_mode(self) -> str:
        # YAML 1.1 parsers commonly decode an unquoted ``off`` as ``False``.
        # Preserve that established spelling without accepting ``True`` as a
        # second, underspecified parallel mode.
        configured = self.lit_config.nqs_data_parallel
        if configured is False:
            return "off"
        return str(configured).lower()

    def _validate_data_parallel_batch_size(
        self,
        batch_size: int,
        *,
        purpose: str,
    ) -> None:
        if not self._nqs_data_parallel_enabled():
            return
        device_count = jax.local_device_count()
        if batch_size < device_count or batch_size % device_count != 0:
            msg = (
                f"Data-parallel NQS-LIT {purpose} batch size {batch_size} must "
                f"be a positive multiple of the {device_count} local devices."
            )
            raise ValueError(msg)

    def _validate_data_parallel_batch(self, batched_data, *, purpose: str) -> None:
        self._validate_data_parallel_batch_size(
            int(batched_data.batch_size),
            purpose=purpose,
        )

    def _validate_source_pool_chunks(
        self,
        train_pool,
        eval_pool,
    ) -> None:
        """Validate complete, fixed-size train/evaluation chunk partitions.

        Raises:
            ValueError: If a pool is empty, has a partial chunk, or cannot shard.
        """
        pool_specs = (
            ("training", train_pool, self._nqs_train_update_batch_size()),
            ("held-out evaluation", eval_pool, self._nqs_eval_batch_size()),
        )
        for purpose, pool, requested_size in pool_specs:
            pool_size = int(pool.batch_size)
            chunk_size = min(int(requested_size), pool_size)
            if chunk_size < 1:
                raise ValueError(f"NQS-LIT {purpose} source pool is empty.")
            if chunk_size < pool_size and pool_size % chunk_size != 0:
                raise ValueError(
                    f"NQS-LIT {purpose} source pool size {pool_size} must be "
                    f"divisible by its effective chunk size {chunk_size}."
                )
            self._validate_data_parallel_batch_size(
                chunk_size,
                purpose=f"{purpose} chunk",
            )

    def _data_parallel_source_update_kernel(self, source_update, update_batch):
        self._validate_data_parallel_batch(update_batch, purpose="training")
        device_count = jax.local_device_count()
        logger.info(
            "Compiling single-frequency NQS-LIT data parallelism devices=%d "
            "global_train_batch=%d local_train_batch=%d",
            device_count,
            int(update_batch.batch_size),
            int(update_batch.batch_size) // device_count,
        )
        return parallel_jax.jit_sharded(
            source_update,
            in_specs=(
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
                update_batch.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
            ),
            out_specs=parallel_jax.SHARE_PARTITION,
            check_vma=True,
        )

    def _init_nqs_update_carry(
        self,
        rng,
        response_params,
    ) -> _NQSUpdateCarry:
        flat_params, _ = ravel_pytree(response_params)
        return _NQSUpdateCarry(
            spring=_SpringState(previous_direction=jnp.zeros_like(flat_params)),
            rng=rng,
        )

    def _log_nqs_summary(self, output_path: str, fidelity: np.ndarray) -> None:
        logger.info("Wrote NQS-LIT spectrum to %s", output_path)
        logger.info(
            "NQS-LIT fidelity range: min=%.6f max=%.6f",
            float(np.min(fidelity)),
            float(np.max(fidelity)),
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
        "spring_history_gradient_ratio=%.6e response_grad_rms=%.6e "
        "response_update=%.6e",
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


def _bisect_continuation_probe(
    evaluate: Callable[[float], Any],
    current_stats,
    *,
    current_omega: float,
    candidate_omega: float,
    target_omega: float,
    min_step: float,
    retention: float,
    min_reweight_ess_fraction: float,
):
    """Bisect an inherited-parameter probe until it is safe or minimal.

    Returns:
        Probe statistics, accepted frequency, acceptance flag, and bisections.
    """
    bisections = 0
    while True:
        probe_stats = evaluate(candidate_omega)
        probe_ok = _continuation_probe_is_acceptable(
            current_stats,
            probe_stats,
            retention=retention,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        candidate_gap = candidate_omega - current_omega
        if probe_ok or candidate_gap <= min_step * (1.0 + 1e-12):
            return probe_stats, candidate_omega, probe_ok, bisections
        candidate_gap = max(min_step, 0.5 * candidate_gap)
        candidate_omega = min(target_omega, current_omega + candidate_gap)
        bisections += 1


def _require_continuation_point_capacity(
    optimized_count: int,
    *,
    maximum: int,
    target_omega: float,
) -> None:
    """Reject an additional optimized bridge after the configured cap.

    Raises:
        RuntimeError: If no additional bridge point may be optimized.
    """
    if optimized_count >= maximum:
        msg = (
            "Adaptive frequency continuation exceeded "
            f"{maximum} bridge points before omega={target_omega:.8g}."
        )
        raise RuntimeError(msg)


def _continuation_capacity_diagnostics(
    *,
    remaining_gap: float,
    optimized_count: int,
    maximum: int,
    chosen_step: float,
) -> _ContinuationCapacityDiagnostics:
    """Forecast whether one proposal pace fits the remaining point budget.

    The forecast is diagnostic only.  It assumes ``chosen_step`` is held for
    every remaining optimized bridge plus the final unoptimized target probe;
    later growth or bisection will change the realized trajectory.

    Returns:
        The remaining budget and proposal pace relative to its required mean.

    Raises:
        ValueError: If the gap, step, or point counts are invalid.
    """
    remaining_gap = float(remaining_gap)
    chosen_step = float(chosen_step)
    optimized_count = int(optimized_count)
    maximum = int(maximum)
    if not np.isfinite(remaining_gap) or remaining_gap <= 0.0:
        msg = f"remaining_gap must be finite and positive, got {remaining_gap!r}."
        raise ValueError(msg)
    if not np.isfinite(chosen_step) or chosen_step <= 0.0:
        msg = f"chosen_step must be finite and positive, got {chosen_step!r}."
        raise ValueError(msg)
    if optimized_count < 0 or maximum < 0 or optimized_count > maximum:
        msg = (
            "Continuation point counts must satisfy "
            f"0 <= optimized_count <= maximum, got {optimized_count} and {maximum}."
        )
        raise ValueError(msg)
    remaining_bridge_slots = maximum - optimized_count
    required_mean_step = remaining_gap / (remaining_bridge_slots + 1)
    return _ContinuationCapacityDiagnostics(
        remaining_gap=remaining_gap,
        remaining_bridge_slots=remaining_bridge_slots,
        required_mean_step=required_mean_step,
        capacity_ratio=chosen_step / required_mean_step,
    )


def _continuation_history_step_cap(
    records: list[_ContinuationRecord],
    *,
    growth_factor: float,
    min_step: float,
) -> float | None:
    """Recover a conservative next-step cap from accepted bridge history.

    A bridge that needed any bisection, failed its inherited probe, or used a
    minimum-step override holds its actual accepted step.  Only a completely
    clean bridge may grow the cap.  Because the accepted record is already in
    the durable checkpoint history, interrupted and uninterrupted runs make
    the same next proposal without adding controller state to the schema.

    Returns:
        The next history-derived step cap, or ``None`` without an accepted
        bridge.

    Raises:
        RuntimeError: If the latest accepted record contains an invalid step.
    """
    latest = next((record for record in reversed(records) if record.optimized), None)
    if latest is None:
        return None
    accepted_step = float(latest.step)
    if not np.isfinite(accepted_step) or accepted_step <= 0.0:
        msg = (
            "Latest optimized continuation record has an invalid accepted "
            f"step {accepted_step!r} at omega={float(latest.omega):.8g}."
        )
        raise RuntimeError(msg)
    clean_success = (
        int(latest.bisections) == 0
        and bool(latest.probe_accepted)
        and not bool(latest.min_step_override)
    )
    multiplier = float(growth_factor) if clean_success else 1.0
    return max(float(min_step), multiplier * accepted_step)


def _physics_continuation_step(stats, *, gap: float, fraction: float, min_step: float):
    """Choose a homotopy step from the inherited LIT residual estimate.

    Returns:
        A positive step no larger than the remaining target gap.
    """
    signed_lit = float(jax.device_get(stats.signed_lit))
    source_norm = float(jax.device_get(stats.source_norm))
    if (
        np.isfinite(signed_lit)
        and np.isfinite(source_norm)
        and signed_lit > 0.0
        and source_norm > 0.0
    ):
        proposed = float(fraction) * np.sqrt(source_norm / signed_lit)
    else:
        proposed = float(min_step)
    return min(float(gap), max(float(min_step), proposed))


def _finite_valid_nqs_stats(stats) -> bool:
    return _is_eligible_nqs_checkpoint(stats)


def _nqs_stage_ess_failure(
    stats,
    *,
    min_reweight_ess_fraction: float,
) -> str | None:
    """Describe a held-out importance-sampling ESS failure, if any.

    Returns:
        A human-readable failure description, or ``None`` when the active
        estimator-health threshold passes.
    """
    ess_fraction = float(jax.device_get(stats.reweight_ess_fraction))
    if min_reweight_ess_fraction > 0.0 and (
        not np.isfinite(ess_fraction) or ess_fraction < float(min_reweight_ess_fraction)
    ):
        return (
            "ESS fraction="
            f"{ess_fraction:.6f} < required="
            f"{float(min_reweight_ess_fraction):.6f}"
        )
    return None


def _require_nqs_stage_health(
    stats,
    *,
    min_reweight_ess_fraction: float,
    context: str,
) -> None:
    """Raise when a finite checkpoint misses the held-out ESS guard.

    Raises:
        RuntimeError: If the active ESS threshold is missed.
    """
    failure = _nqs_stage_ess_failure(
        stats,
        min_reweight_ess_fraction=min_reweight_ess_fraction,
    )
    if failure is not None:
        raise RuntimeError(f"{context}; {failure}.")


def _continuation_probe_is_acceptable(
    current,
    candidate,
    *,
    retention: float,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Return whether an inherited checkpoint is safe enough to optimize.

    This relative check protects initialization quality without imposing an
    arbitrary absolute fidelity threshold on the optimized result.
    """
    if not _finite_valid_nqs_stats(candidate):
        return False
    if (
        _nqs_stage_ess_failure(
            candidate,
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


def _save_npz_compressed(path: UPath, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f_out:
        np.savez_compressed(f_out, **payload)  # type: ignore[arg-type]


_CONTINUATION_HISTORY_STAT_FIELDS = (
    "fidelity",
    "reverse_kl",
    "invalid_sample_fraction",
)

_CONTINUATION_STATE_CONFIG_EXCLUSIONS = frozenset(
    {
        "output_filename",
        "inversion_enabled",
        "inversion_output_filename",
        "inversion_additional_input_paths",
        "inversion_assume_independent",
        "inversion_threshold",
        "inversion_pole_energies",
        "inversion_continuum_grid",
        "inversion_continuum_regularization",
        "inversion_fit_pole_energies",
        "inversion_pole_energy_bounds",
        "inversion_covariance_relative_tolerance",
        "inversion_max_fitted_poles",
        "inversion_pole_fit_tolerance",
        "inversion_pole_fit_max_iterations",
        "inversion_solver_tolerance",
        "inversion_solver_max_iterations",
        "nqs_checkpoint_path",
        "nqs_source_pool_dir",
        "nqs_reuse_source_pool",
        "nqs_save_source_pool",
        "nqs_data_parallel",
        "nqs_train_update_batch_size",
        "nqs_eval_batch_size",
        "nqs_train_update_batch_size_per_device",
        "nqs_eval_batch_size_per_device",
        "nqs_train_pool_batches",
        "nqs_eval_pool_batches",
        "nqs_pool_stride",
        "nqs_source_center_steps",
        "nqs_source_burn_in",
        "nqs_energy_steps",
        "nqs_burn_in",
        "nqs_source_distillation_iterations",
        "nqs_learning_rate",
        "nqs_reverse_kl_weight",
        "nqs_spring_epsilon",
        "nqs_spring_decay",
        "nqs_spring_damping_floor",
        "nqs_sr_max_norm",
        "nqs_sr_score_eps",
        "nqs_warm_start_iterations",
        "nqs_iterations",
        "nqs_fidelity_plateau_start_iteration",
        "nqs_fidelity_plateau_patience_iterations",
        "nqs_fidelity_plateau_min_delta",
        "nqs_continuation_iterations",
        "nqs_continuation_step_fraction",
        "nqs_continuation_step_growth_factor",
        "nqs_continuation_fidelity_retention",
        "nqs_stage_reweight_ess_fraction_min",
        "nqs_continuation_allow_min_step_override",
        "nqs_continuation_min_step",
        "nqs_continuation_max_points",
        "nqs_continuation_restore_path",
        "nqs_selection_interval",
        "nqs_log_interval",
        "nqs_parity_eval_batch_size",
        "nqs_atomic_source_parity_max_loss",
        "nqs_atomic_ground_parity_max_loss",
    }
)


def _empty_nqs_lit_stats() -> NQSLITStats:
    return NQSLITStats(
        **{
            name: jnp.asarray(False if name == "error_d_valid" else 0.0)
            for name in NQSLITStats._fields
        }
    )


def _checkpoint_json_value(value):
    if isinstance(value, Enum):
        return _checkpoint_json_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _checkpoint_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_checkpoint_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _checkpoint_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _checkpoint_json_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _tree_shape_dtype_signature(tree) -> dict[str, object]:
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaves = []
    for key_path, leaf in leaves_with_path:
        shape = tuple(int(size) for size in getattr(leaf, "shape", ()))
        dtype = getattr(leaf, "dtype", type(leaf).__name__)
        leaves.append(
            {
                "path": str(key_path),
                "shape": list(shape),
                "dtype": str(dtype),
            }
        )
    return {"treedef": str(treedef), "leaves": leaves}


def _tree_content_digest(tree) -> str:
    """Return a deterministic SHA-256 digest of one concrete PyTree.

    Checkpoint compatibility must distinguish equal-shaped ground states and
    source pools belonging to different systems or runs.  Fresh response
    parameters intentionally use only their shape/dtype signature because
    they are a restore template and are replaced by the saved parameters.

    Raises:
        TypeError: If a leaf uses an object dtype without stable byte content.
    """
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(tree)
    digest = hashlib.sha256(str(treedef).encode("utf-8"))
    for key_path, leaf in leaves_with_path:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            msg = (
                "Continuation checkpoint fingerprints cannot hash object-dtype "
                f"leaf {key_path}."
            )
            raise TypeError(msg)
        metadata = json.dumps(
            {
                "path": str(key_path),
                "shape": list(array.shape),
                "dtype": array.dtype.str,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _source_pool_target_digest(ground_params, molecule_data) -> str:
    """Bind a source pool to the density and Hamiltonian that define pi_Phi.

    Returns:
        A stable hexadecimal digest of the ground parameters and static
        molecular geometry.
    """
    return _canonical_sha256(
        {
            "schema_version": 1,
            "ground_params_sha256": _tree_content_digest(ground_params),
            "atoms_sha256": _tree_content_digest(molecule_data.atoms),
            "charges_sha256": _tree_content_digest(molecule_data.charges),
        }
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        _checkpoint_json_value(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _continuation_checkpoint_digests(
    config: MolecularLITConfig,
    *,
    response_params,
    ground_params,
    train_pool,
    eval_pool,
    axis: int,
    source_center: float,
    source_norm: float,
    ground_energy: float,
    ground_checkpoint_step: int,
    response_parity: int,
    target_omega: float,
    spectrum_omega: np.ndarray,
) -> tuple[str, str]:
    """Return compatibility and full-audit digests for continuation state."""
    config_payload = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
        if config_field.name != "nqs_continuation_restore_path"
    }
    state_config = {
        name: value
        for name, value in config_payload.items()
        if name not in _CONTINUATION_STATE_CONFIG_EXCLUSIONS
    }
    dynamic_payload = {
        "schema_version": _CONTINUATION_CHECKPOINT_SCHEMA_VERSION,
        "axis": int(axis),
        "source_center": float(source_center),
        "source_norm": float(source_norm),
        "ground_energy": float(ground_energy),
        "ground_checkpoint_step": int(ground_checkpoint_step),
        "response_parity": int(response_parity),
        "target_omega": float(target_omega),
        "spectrum_omega": np.asarray(spectrum_omega, dtype=np.float64),
        "response_params": _tree_shape_dtype_signature(response_params),
        "ground_params": {
            "signature": _tree_shape_dtype_signature(ground_params),
            "content_sha256": _tree_content_digest(ground_params),
        },
        # Pool contents include the sampled electron configurations and the
        # static molecular data, so they bind a resume to both its held-out
        # population and its Hamiltonian/system identity.
        "train_pool_sha256": _tree_content_digest(train_pool),
        "eval_pool_sha256": _tree_content_digest(eval_pool),
    }
    state_fingerprint = _canonical_sha256(
        {"config": state_config, "dynamic": dynamic_payload}
    )
    full_config_digest = _canonical_sha256(
        {"config": config_payload, "dynamic": dynamic_payload}
    )
    return state_fingerprint, full_config_digest


def _continuation_records_to_json(records: list[_ContinuationRecord]) -> str:
    payload = []
    for record in records:
        stats_payload = {
            name: float(jax.device_get(getattr(record.stats, name, 0.0)))
            for name in _CONTINUATION_HISTORY_STAT_FIELDS
        }
        payload.append(
            {
                "omega": float(record.omega),
                "optimized": bool(record.optimized),
                "selected_iteration": int(record.selected_iteration),
                "stats": stats_payload,
                "inherited_fidelity": float(record.inherited_fidelity),
                "step": float(record.step),
                "bisections": int(record.bisections),
                "probe_accepted": bool(record.probe_accepted),
                "min_step_override": bool(record.min_step_override),
            }
        )
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _continuation_records_from_json(
    encoded: str,
    *,
    stats_template: NQSLITStats,
) -> list[_ContinuationRecord]:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Continuation checkpoint history is not valid JSON."
        ) from exc
    if not isinstance(payload, list):
        raise RuntimeError("Continuation checkpoint history must be a JSON list.")
    records = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("stats"), dict):
            raise RuntimeError("Continuation checkpoint history entry is malformed.")
        replacements = {
            name: jnp.asarray(float(item["stats"][name]))
            for name in _CONTINUATION_HISTORY_STAT_FIELDS
        }
        records.append(
            _ContinuationRecord(
                omega=float(item["omega"]),
                optimized=bool(item["optimized"]),
                selected_iteration=int(item["selected_iteration"]),
                stats=stats_template._replace(**replacements),
                inherited_fidelity=float(item["inherited_fidelity"]),
                step=float(item["step"]),
                bisections=int(item["bisections"]),
                probe_accepted=bool(item["probe_accepted"]),
                min_step_override=bool(item["min_step_override"]),
            )
        )
    return records


def _read_continuation_checkpoint_metadata(path: UPath) -> dict[str, object]:
    with path.open("rb") as f_in, np.load(f_in, allow_pickle=False) as npf:
        return {
            "schema_version": int(npf["schema_version"].item()),
            "state_fingerprint": str(npf["state_fingerprint"].item()),
            "full_config_digest": str(npf["full_config_digest"].item()),
        }


def _latest_continuation_checkpoint_path(path: UPath) -> UPath | None:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            path.glob(f"{_CONTINUATION_CHECKPOINT_PREFIX}_ckpt_*.npz"),
            reverse=True,
        )
    else:
        return None
    for candidate in candidates:
        try:
            _read_continuation_checkpoint_metadata(candidate)
        except (OSError, EOFError, BadZipFile, KeyError, ValueError):
            logger.warning("Ignoring unreadable continuation checkpoint %s", candidate)
            continue
        return candidate
    return None


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
    if len(result) != len(set(result)):
        msg = f"Duplicate dipole axes are not allowed: {axes!r}."
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


def _replicate_across_local_devices(value):
    """Place a pytree with replicated sharding on the local device mesh.

    Returns:
        The same logical value with ``PartitionSpec()`` sharding.
    """
    return jax.device_put(
        value,
        parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION),
    )


def _shard_batched_data_across_local_devices(pool: BatchedData) -> BatchedData:
    """Place batch fields across the local mesh and replicate shared fields.

    Returns:
        The unchanged global batch with its declared partitioning applied.
    """
    return jax.device_put(
        pool,
        parallel_jax.make_sharding(pool.partition_spec),
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


def _indexed_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    chunk_index: int,
) -> BatchedData:
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    if chunk_size >= pool.batch_size:
        return pool
    if pool.batch_size % chunk_size != 0:
        raise ValueError(
            f"Source pool size {pool.batch_size} must be divisible by training "
            f"chunk size {chunk_size}."
        )
    chunk_count = pool.batch_size // chunk_size
    if not 0 <= int(chunk_index) < chunk_count:
        raise ValueError(
            f"Training chunk index {chunk_index} is outside [0, {chunk_count})."
        )
    return _slice_batched_data(pool, int(chunk_index) * chunk_size, chunk_size)


def _shuffled_batched_data_chunk_index(
    pool_size: int,
    requested_size: int,
    iteration: int,
    *,
    seed: int,
) -> int:
    """Map an iteration to a deterministic, epoch-wise random chunk index.

    Every epoch visits each fixed-pool chunk exactly once.  Exact divisibility
    is required so no walkers disappear behind a partial tail.

    Returns:
        The contiguous chunk index selected for ``iteration``.

    Raises:
        ValueError: If the iteration is negative or the pool has a partial tail.
    """
    if int(iteration) < 0:
        raise ValueError("Training iteration must be nonnegative.")
    chunk_size = min(max(1, int(requested_size)), max(1, int(pool_size)))
    if chunk_size >= int(pool_size):
        return 0
    if int(pool_size) % chunk_size != 0:
        raise ValueError(
            f"Source pool size {pool_size} must be divisible by training "
            f"chunk size {chunk_size}."
        )
    chunk_count = int(pool_size) // chunk_size
    epoch, position = divmod(int(iteration), chunk_count)
    seed_sequence = np.random.SeedSequence(
        (
            int(seed) & 0xFFFFFFFF,
            (int(seed) >> 32) & 0xFFFFFFFF,
            epoch & 0xFFFFFFFF,
            (epoch >> 32) & 0xFFFFFFFF,
        )
    )
    order = np.random.default_rng(seed_sequence).permutation(chunk_count)
    return int(order[position])


def _shuffled_batched_data_chunk(
    pool: BatchedData,
    requested_size: int,
    iteration: int,
    *,
    seed: int,
) -> BatchedData:
    chunk_index = _shuffled_batched_data_chunk_index(
        pool.batch_size,
        requested_size,
        iteration,
        seed=seed,
    )
    return _indexed_batched_data_chunk(pool, requested_size, chunk_index)


def _batched_data_chunks(pool: BatchedData, requested_size: int):
    chunk_size = min(max(1, int(requested_size)), max(1, pool.batch_size))
    start = 0
    while start < pool.batch_size:
        size = min(chunk_size, pool.batch_size - start)
        yield _slice_batched_data(pool, start, size)
        start += size


@jax.jit
def _nqs_stats_from_source_sums(
    sums: NQSLITSourceSums,
    source_norm,
    omega,
    eta,
):
    """Convert held-out sums with one persistent fused executable.

    Returns:
        Standard NQS-LIT statistics for the accumulated source moments.
    """
    return nqs_lit_stats_from_source_sums(
        sums,
        source_norm=source_norm,
        omega=omega,
        eta=eta,
    )


@jax.jit
def _add_source_sums(left, right):
    """Compatibility wrapper for the stable Chan/source-scale merge.

    Returns:
        The merged source-sampled accumulator.
    """
    return merge_nqs_lit_source_sums(left, right)


def _signed_lit_jackknife_pseudovalues(
    full_stats: NQSLITStats,
    block_sums: tuple[NQSLITSourceSums, ...],
    *,
    source_norm: float,
    omega: float,
    eta: float,
) -> np.ndarray:
    """Return delete-one-block pseudovalues for the nonlinear LIT estimator.

    The signed transform is a ratio of accumulated moments, so treating each
    evaluation chunk as an independent transform and averaging those values is
    biased.  We instead merge all chunks except one, recompute the estimator,
    and form standard delete-one-block jackknife pseudovalues.  Prefix/suffix
    Chan merges keep this linear in the number of chunks.

    Raises:
        RuntimeError: If a defensive leave-one-out merge is unexpectedly empty.
    """
    block_count = len(block_sums)
    full_value = float(jax.device_get(full_stats.signed_lit))
    if block_count < 2:
        return np.asarray([full_value], dtype=np.float64)

    prefix: list[NQSLITSourceSums | None] = [None]
    for block in block_sums:
        previous = prefix[-1]
        prefix.append(block if previous is None else _add_source_sums(previous, block))

    suffix: list[NQSLITSourceSums | None] = [None] * (block_count + 1)
    for block_index in range(block_count - 1, -1, -1):
        following = suffix[block_index + 1]
        block = block_sums[block_index]
        suffix[block_index] = (
            block if following is None else _add_source_sums(block, following)
        )

    leave_one_out = np.empty(block_count, dtype=np.float64)
    for block_index in range(block_count):
        left = prefix[block_index]
        right = suffix[block_index + 1]
        if left is None:
            excluded = right
        elif right is None:
            excluded = left
        else:
            excluded = _add_source_sums(left, right)
        if excluded is None:  # Defensive; block_count >= 2 makes this unreachable.
            msg = "Jackknife exclusion produced an empty evaluation pool."
            raise RuntimeError(msg)
        excluded_stats = _nqs_stats_from_source_sums(
            excluded,
            jnp.asarray(source_norm),
            jnp.asarray(omega),
            jnp.asarray(eta),
        )
        leave_one_out[block_index] = float(jax.device_get(excluded_stats.signed_lit))

    return block_count * full_value - (block_count - 1) * leave_one_out


def _merge_source_sums_across_devices(
    local_sums: NQSLITSourceSums,
    *,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
) -> NQSLITSourceSums:
    """Compatibility wrapper for the stable data-parallel Chan merge.

    Returns:
        The globally merged source-sampled accumulator on every device.
    """
    return merge_nqs_lit_source_sums_across_devices(
        local_sums,
        axis_name=axis_name,
    )


def _solve_sr_direction_data_parallel(
    local_score_aug,
    grad_flat,
    damping,
    *,
    device_count: int,
    local_kernel_null_vectors=None,
    kernel_projector_scale=1.0,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
):
    """Solve the exact global SR system without gathering the full score.

    The dual branch redistributes parameter columns with ``all_to_all``.  Each
    device therefore owns all global sample rows for only a fraction of the
    parameters, constructs one Gram contribution, and replicates only the
    ``O(batch**2)`` kernel and Cholesky factor.

    Returns:
        The replicated flattened SR direction.

    Raises:
        ValueError: If ``device_count`` is not positive.
    """
    if device_count < 1:
        msg = "device_count must be positive."
        raise ValueError(msg)
    parameter_count = int(grad_flat.shape[0])
    local_sample_count = int(local_score_aug.shape[0])
    sample_count = local_sample_count * int(device_count)
    original_dtype = grad_flat.dtype
    with _enable_x64(True):
        solve_dtype = jnp.float64
        score_solve = local_score_aug.astype(solve_dtype)
        grad_solve = grad_flat.astype(solve_dtype)
        damping_solve = jnp.asarray(damping, dtype=solve_dtype)
        if parameter_count <= sample_count:
            metric = jax.lax.psum(
                score_solve.T @ score_solve,
                axis_name=axis_name,
            )
            metric = (metric + metric.T) / 2.0
            metric = metric + damping_solve * jnp.eye(
                parameter_count,
                dtype=solve_dtype,
            )
            chol = jsp.linalg.cho_factor(metric, lower=True)
            direction = jsp.linalg.cho_solve(chol, grad_solve)
        else:
            padded_parameter_count = (
                (parameter_count + device_count - 1) // device_count
            ) * device_count
            score_padded = jnp.pad(
                score_solve,
                ((0, 0), (0, padded_parameter_count - parameter_count)),
            )
            score_by_parameter = jax.lax.all_to_all(
                score_padded,
                axis_name=axis_name,
                split_axis=1,
                concat_axis=0,
                tiled=True,
            )
            kernel = jax.lax.psum(
                score_by_parameter @ score_by_parameter.T,
                axis_name=axis_name,
            )
            kernel = (kernel + kernel.T) / 2.0
            if local_kernel_null_vectors is not None:
                null_vectors = jax.lax.all_gather(
                    jnp.asarray(local_kernel_null_vectors, dtype=solve_dtype),
                    axis_name=axis_name,
                    axis=1,
                    tiled=True,
                )
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
            local_rhs = score_solve @ grad_solve
            rhs = jax.lax.all_gather(
                local_rhs,
                axis_name=axis_name,
                axis=0,
                tiled=True,
            )
            chol = jsp.linalg.cho_factor(kernel, lower=True)
            alpha = jsp.linalg.cho_solve(chol, rhs)
            alpha_start = jax.lax.axis_index(axis_name) * local_sample_count
            local_alpha = jax.lax.dynamic_slice_in_dim(
                alpha,
                alpha_start,
                local_sample_count,
                axis=0,
            )
            projected = jax.lax.psum(
                score_solve.T @ local_alpha,
                axis_name=axis_name,
            )
            direction = (grad_solve - projected) / damping_solve
        direction = direction.astype(original_dtype)
    return jnp.where(
        jnp.all(jnp.isfinite(direction)),
        direction,
        jnp.zeros_like(direction),
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
    metadata: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "fields_with_batch": np.asarray(list(pool.fields_with_batch), dtype=str),
    }
    if metadata is not None:
        for key, value in metadata.items():
            encoded = np.asarray(value)
            if encoded.ndim != 0 or encoded.dtype.hasobject:
                msg = f"source pool metadata {key!r} must be a scalar value"
                raise TypeError(msg)
            payload[f"metadata_{key}"] = encoded
    for field_name in pool.fields_with_batch:
        payload[field_name] = np.asarray(jax.device_get(getattr(pool.data, field_name)))
    _save_npz(path, **payload)


def _load_batched_pool(
    path: UPath,
    reference: BatchedData,
    *,
    metadata: Mapping[str, object] | None = None,
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


def _validate_pool_metadata(npf, metadata: Mapping[str, object]) -> None:
    for key, expected in metadata.items():
        npz_key = f"metadata_{key}"
        if npz_key not in npf:
            msg = f"source pool is missing metadata {key!r}"
            raise ValueError(msg)
        encoded = np.asarray(npf[npz_key])
        if encoded.ndim != 0:
            msg = f"source pool metadata {key!r} must be scalar"
            raise ValueError(msg)
        actual: Any = encoded.item()
        matches: bool
        if isinstance(expected, str):
            matches = isinstance(actual, str) and actual == expected
        else:
            numeric_expected: Any = expected
            try:
                matches = bool(
                    np.isclose(
                        float(actual),
                        float(numeric_expected),
                        rtol=1e-8,
                        atol=1e-10,
                    )
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            msg = f"source pool metadata {key!r} mismatch: {actual!r} != {expected!r}"
            raise ValueError(msg)


def _require_pool_walker_count(
    pool: BatchedData,
    *,
    expected_walkers: int,
    split: str,
) -> None:
    actual_walkers = int(pool.batch_size)
    if actual_walkers != int(expected_walkers):
        msg = (
            f"{split} source pool has {actual_walkers} walkers; expected exactly "
            f"{int(expected_walkers)} from workflow.batch_size * configured batches"
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
    axis_name: str | None = None,
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

    def global_sum(value, *, axis=None, keepdims=False):
        reduced = jnp.sum(value, axis=axis, keepdims=keepdims)
        if axis_name is not None:
            reduced = jax.lax.psum(reduced, axis_name=axis_name)
        return reduced

    def global_max(value):
        reduced = jnp.max(value)
        if axis_name is not None:
            reduced = jax.lax.pmax(reduced, axis_name=axis_name)
        return reduced

    safe_weight_sum = jnp.maximum(global_sum(source_weight), eps_array)
    phi_weight = source_weight / safe_weight_sum
    max_ratio_abs = global_max(jnp.where(phi_weight > 0.0, jnp.abs(ratio), 0.0))
    ratio_scale = jnp.where(
        max_ratio_abs > 0.0,
        max_ratio_abs,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    scaled_ratio = ratio / jax.lax.stop_gradient(ratio_scale)
    ratio_abs2 = jnp.abs(scaled_ratio) ** 2
    ratio_norm = global_sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps_array)
    has_action_mass = jnp.isfinite(ratio_norm) & (ratio_norm > 0.0)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm

    score_mean = global_sum(
        psi_weight[:, None] * score,
        axis=0,
        keepdims=True,
    )
    centered_score = score - score_mean
    amplitude = global_sum(phi_weight * scaled_ratio)
    score_covariance = global_sum(
        phi_weight[:, None] * scaled_ratio[:, None] * centered_score,
        axis=0,
    )
    fidelity_gradient = 2.0 * jnp.real(
        jnp.conj(amplitude) * score_covariance / safe_ratio_norm
    )

    log_ratio_abs2 = 2.0 * jnp.log(jnp.maximum(jnp.abs(scaled_ratio), eps_array))
    log_ratio_mean = global_sum(psi_weight * log_ratio_abs2)
    centered_log_ratio = log_ratio_abs2 - log_ratio_mean
    reverse_kl_gradient = 2.0 * jnp.real(
        global_sum(
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


def _source_distillation_stats_from_log_ratios(
    log_ratio,
    source_weight,
    *,
    reverse_kl_weight: float | jnp.ndarray,
    axis_name: str | None = None,
) -> _SourceDistillationStats:
    """Evaluate normalized response/source overlap on ``pi_Phi`` samples.

    Returns:
        Scale-invariant fidelity, reverse-KL, ESS, and sample-health statistics.
    """
    real_dtype = jnp.real(log_ratio).dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    finite = (
        jnp.isfinite(jnp.real(log_ratio))
        & jnp.isfinite(jnp.imag(log_ratio))
        & jnp.isfinite(source_weight)
        & (source_weight >= 0.0)
    )
    local_count = jnp.asarray(log_ratio.size, dtype=real_dtype)
    valid_count = jnp.sum(finite.astype(real_dtype))
    safe_real = jnp.where(finite & (source_weight > 0.0), jnp.real(log_ratio), -jnp.inf)
    log_scale = jnp.max(safe_real)
    if axis_name is not None:
        local_count = jax.lax.psum(local_count, axis_name=axis_name)
        valid_count = jax.lax.psum(valid_count, axis_name=axis_name)
        log_scale = jax.lax.pmax(log_scale, axis_name=axis_name)
    log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
    log_scale = jax.lax.stop_gradient(log_scale)
    ratio = jnp.where(
        finite,
        jnp.exp(log_ratio - log_scale),
        jnp.asarray(0.0, dtype=log_ratio.dtype),
    )
    weight = jnp.where(finite, source_weight, 0.0)

    def global_sum(value):
        total = jnp.sum(value)
        if axis_name is not None:
            total = jax.lax.psum(total, axis_name=axis_name)
        return total

    weight_sum = global_sum(weight)
    phi_weight = weight / jnp.maximum(weight_sum, eps)
    ratio_abs2 = jnp.abs(ratio) ** 2
    ratio_norm = global_sum(phi_weight * ratio_abs2)
    safe_ratio_norm = jnp.maximum(ratio_norm, eps)
    amplitude = global_sum(phi_weight * ratio)
    fidelity = jnp.clip(jnp.abs(amplitude) ** 2 / safe_ratio_norm, 0.0, 1.0)
    psi_weight = phi_weight * ratio_abs2 / safe_ratio_norm
    log_ratio_abs2 = 2.0 * jnp.log(jnp.maximum(jnp.abs(ratio), eps))
    reverse_kl = jnp.maximum(
        global_sum(psi_weight * log_ratio_abs2) - jnp.log(safe_ratio_norm),
        0.0,
    )
    psi_weight_sq_sum = global_sum(psi_weight**2)
    ess = 1.0 / jnp.maximum(psi_weight_sq_sum, eps)
    ess_fraction = ess / jnp.maximum(local_count, 1.0)
    invalid_fraction = 1.0 - valid_count / jnp.maximum(local_count, 1.0)
    has_mass = (weight_sum > 0.0) & (ratio_norm > 0.0)
    fidelity = jnp.where(has_mass, fidelity, 0.0)
    reverse_kl = jnp.where(has_mass, reverse_kl, jnp.inf)
    ess_fraction = jnp.where(has_mass, ess_fraction, 0.0)
    loss = (
        1.0 - fidelity + jnp.asarray(reverse_kl_weight, dtype=real_dtype) * reverse_kl
    )
    return _SourceDistillationStats(
        loss=loss,
        fidelity=fidelity,
        reverse_kl=reverse_kl,
        reweight_ess_fraction=ess_fraction,
        invalid_sample_fraction=invalid_fraction,
    )


def _finite_source_distillation_stats(stats: _SourceDistillationStats) -> bool:
    return all(
        np.isfinite(float(value))
        for value in (
            stats.loss,
            stats.fidelity,
            stats.reverse_kl,
            stats.reweight_ess_fraction,
            stats.invalid_sample_fraction,
        )
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


def _spring_direction_data_parallel(
    local_score_aug,
    grad_flat,
    state: _SpringState,
    *,
    epsilon_scale: float | jnp.ndarray,
    damping_floor: float | jnp.ndarray,
    decay: float | jnp.ndarray,
    device_count: int,
    qfi_trace,
    local_kernel_null_vectors=None,
    axis_name: str = parallel_jax.BATCH_AXIS_NAME,
):
    """Apply SPRING to a row-sharded score matrix.

    Returns:
        Replicated direction, replicated next history state, and damping.
    """
    parameter_count = jnp.asarray(max(int(grad_flat.shape[0]), 1), grad_flat.dtype)
    mean_metric_diagonal = qfi_trace / parameter_count
    damping = jnp.maximum(
        jnp.asarray(epsilon_scale, dtype=grad_flat.dtype) * mean_metric_diagonal,
        jnp.asarray(damping_floor, dtype=grad_flat.dtype),
    )
    rhs = (
        grad_flat
        + damping * jnp.asarray(decay, dtype=grad_flat.dtype) * state.previous_direction
    )
    direction = _solve_sr_direction_data_parallel(
        local_score_aug,
        rhs,
        damping,
        device_count=device_count,
        local_kernel_null_vectors=local_kernel_null_vectors,
        axis_name=axis_name,
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
    if group_name == "response":
        group = tree
    elif isinstance(tree, Mapping) and group_name in tree:
        group = tree[group_name]
    else:
        return missing, missing
    leaves = jax.tree_util.tree_leaves(group)
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


def _require_eligible_nqs_checkpoint(
    stats,
    *,
    context: str,
) -> None:
    """Raise when a checkpoint is numerically invalid.

    Raises:
        RuntimeError: If the checkpoint is not eligible for propagation.
    """
    if _is_eligible_nqs_checkpoint(stats):
        return
    raise RuntimeError(f"{context}; held-out statistics are non-finite or invalid.")


def _is_eligible_nqs_checkpoint(stats) -> bool:
    """Return whether one held-out checkpoint is numerically admissible."""
    loss = float(jax.device_get(stats.loss))
    fidelity = float(jax.device_get(stats.fidelity))
    reverse_kl = float(jax.device_get(stats.reverse_kl))
    invalid = float(jax.device_get(stats.invalid_sample_fraction))
    finite = bool(
        np.all(
            np.isfinite(
                (
                    loss,
                    fidelity,
                    reverse_kl,
                    invalid,
                )
            )
        )
    )
    return finite and invalid <= 0.0


def _is_selectable_nqs_checkpoint(
    stats,
    *,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Return whether a checkpoint is safe for selection and propagation."""
    return (
        _is_eligible_nqs_checkpoint(stats)
        and _nqs_stage_ess_failure(
            stats,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        is None
    )


def _is_better_nqs_checkpoint(
    candidate,
    incumbent,
    *,
    min_reweight_ess_fraction: float = 0.0,
) -> bool:
    """Prefer the highest-fidelity healthy held-out checkpoint.

    Returns:
        Whether the candidate should replace the incumbent.
    """

    def score(stats):
        loss = float(jax.device_get(stats.loss))
        fidelity = float(jax.device_get(stats.fidelity))
        reverse_kl = float(jax.device_get(stats.reverse_kl))
        valid = _is_selectable_nqs_checkpoint(
            stats,
            min_reweight_ess_fraction=min_reweight_ess_fraction,
        )
        return valid, fidelity, -loss, -reverse_kl

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
    error_d_valid: bool,
) -> float:
    """Apply the paper's leading-order error monitor, dividing by ``|N|`` once.

    Supplement Eq. (19) drops higher orders in ``1 - fidelity``.  The result is
    therefore useful as a convergence/systematic-error monitor near unit
    fidelity, but is not advertised as a rigorous upper bound at moderate
    fidelity.

    Returns:
        The finite bound monitor, or ``NaN`` when an input is invalid.
    """
    fidelity = float(fidelity)
    source_norm = float(source_norm)
    eta = float(eta)
    error_d = float(error_d)
    normalization_abs = float(abs(normalization))
    if (
        not bool(error_d_valid)
        or not np.isfinite(error_d)
        or error_d < 0.0
        or not np.isfinite(normalization_abs)
        or normalization_abs <= 0.0
        or not np.isfinite(fidelity)
        or not 0.0 < fidelity <= 1.0
        or not np.isfinite(source_norm)
        or source_norm < 0.0
        or not np.isfinite(eta)
        or eta <= 0.0
    ):
        return float("nan")
    phi_norm = float(np.sqrt(source_norm))
    return lit_error_bound(
        fidelity,
        phi_norm=phi_norm,
        normalization_abs=normalization_abs,
        eta=eta,
        d_factor=error_d,
    )
