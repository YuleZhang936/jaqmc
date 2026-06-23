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
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from upath import UPath

from jaqmc.app.molecule.data import data_init
from jaqmc.app.molecule.workflow import configure_system
from jaqmc.data import BatchedData
from jaqmc.response.inversion import fit_lit_basis_expansion
from jaqmc.response.lit import lit_error_bound
from jaqmc.response.nqs_lit import (
    MolecularResponseFermiNet,
    ground_local_energy,
    local_action_ratio,
    molecular_electronic_dipole,
    nqs_lit_double_sampled_stats,
    nqs_lit_source_sampled_stats,
    restore_params_from_checkpoint,
)
from jaqmc.response.spectrum import find_spectrum_peaks
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
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
    scan_parallel_min_points_per_worker: int = 2
    scan_parallel_worker: bool = False
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
    nqs_source_pool_dir: str = ""
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
    inversion_enabled: bool = True
    inversion_threshold: float = 0.0
    inversion_response_max: float | None = None
    inversion_response_points: int = 1000
    inversion_basis_count: int = 8
    inversion_alpha1_grid: tuple[float, ...] = field(
        default_factory=lambda: (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    )
    inversion_alpha2_grid: tuple[float, ...] = field(default_factory=tuple)
    inversion_l2_grid: tuple[float, ...] = field(
        default_factory=lambda: (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    )


@dataclass(frozen=True)
class _ParallelWorker:
    index: int
    block: np.ndarray
    path: UPath
    log_path: UPath
    process: subprocess.Popen[bytes]
    device: str


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
            response_params, axis_batched_data, rng = self._warm_start_axis(
                update_step,
                response_params,
                train_pool,
                axis_batched_data,
                rng,
                axis=axis,
            )
            for omega_pos, omega_value in enumerate(omega):
                stats = None
                for iteration in range(self.lit_config.nqs_iterations):
                    response_params, stats, axis_batched_data, rng = update_step(
                        response_params,
                        train_pool,
                        jnp.asarray(float(omega_value)),
                        axis_batched_data,
                        rng,
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
                    stats = self._nqs_stats(
                        response_apply,
                        response_params,
                        ground_logpsi,
                        ground_params,
                        train_pool,
                        axis=axis,
                        source_center=source_center,
                        source_norm=axis_phi_norm,
                        ground_energy=ground_energy,
                        omega=float(omega_value),
                    )
                stats, axis_batched_data, rng = self._nqs_eval_stats_with_fallback(
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
        inversion_output, inversion_peaks = self._invert_lit_spectrum(omega, lit)
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
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            source_centers=source_centers,
            axis_source_norm=axis_source_norm,
            peak_energies=np.asarray([peak.energy for peak in peaks]),
            peak_intensities=np.asarray([peak.intensity for peak in peaks]),
            peak_indices=np.asarray([peak.index for peak in peaks]),
            inversion_peak_energies=np.asarray(
                [peak.energy for peak in inversion_peaks]
            ),
            inversion_peak_intensities=np.asarray(
                [peak.intensity for peak in inversion_peaks]
            ),
            inversion_peak_indices=np.asarray([peak.index for peak in inversion_peaks]),
            **inversion_output,
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
        if self.lit_config.scan_parallel_min_points_per_worker < 1:
            msg = "lit.scan_parallel_min_points_per_worker must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.lit_config.nqs_reweight_ess_fraction_min <= 1.0:
            msg = (
                "lit.nqs_reweight_ess_fraction_min must be between 0 and 1, got "
                f"{self.lit_config.nqs_reweight_ess_fraction_min}."
            )
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
    ):
        pool_path = self._source_pool_path(axis, split)
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

    def _source_pool_path(self, axis: int, split: str) -> UPath:
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

    def _warm_start_axis(
        self,
        update_step,
        response_params,
        train_pool,
        fallback_data,
        rng,
        *,
        axis: int,
    ):
        if (
            self.lit_config.nqs_warm_start_omega is None
            or self.lit_config.nqs_warm_start_iterations <= 0
        ):
            return response_params, fallback_data, rng
        stats = None
        for _ in range(self.lit_config.nqs_warm_start_iterations):
            response_params, stats, fallback_data, rng = update_step(
                response_params,
                train_pool,
                jnp.asarray(float(self.lit_config.nqs_warm_start_omega)),
                fallback_data,
                rng,
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
        @jax.jit
        def reweighted_update(response_params, batched_data, omega):
            _, stats = self._nqs_loss(
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
            updates = self._weighted_sr_updates(
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
            response_params = _apply_updates(response_params, updates)
            loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
            return response_params, stats._replace(loss=loss)

        @jax.jit
        def direct_update(
            response_params,
            source_batched_data,
            psi_batched_data,
            omega,
        ):
            stats = self._nqs_double_stats(
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
            )
            updates = self._direct_sr_updates(
                response_apply,
                response_params,
                ground_logpsi,
                ground_params,
                source_batched_data,
                psi_batched_data,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            )
            response_params = _apply_updates(response_params, updates)
            loss = _fidelity_loss(stats.fidelity, self.lit_config.nqs_sr_score_eps)
            return response_params, stats._replace(loss=loss)

        def update(response_params, batched_data, omega, fallback_data, rng):
            stats = self._nqs_stats(
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
            if self._should_use_direct_psi(stats):
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
                    batches=self.lit_config.nqs_direct_psi_batches,
                )
                response_params, stats = direct_update(
                    response_params,
                    batched_data,
                    psi_pool,
                    omega,
                )
                return response_params, stats, fallback_data, rng
            response_params, stats = reweighted_update(
                response_params,
                batched_data,
                omega,
            )
            return response_params, stats, fallback_data, rng

        return update

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
        stats = self._nqs_stats(
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
            batches=max(1, min(self.lit_config.nqs_direct_psi_batches, 2)),
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
        sample_plan = SamplePlan(
            self._make_action_log_amplitude(
                response_apply,
                ground_logpsi,
                ground_params,
                axis=axis,
                source_center=source_center,
                ground_energy=ground_energy,
                omega=omega,
            ),
            {"electrons": self.sampler},
        )
        rng, sample_rng = jax.random.split(rng)
        sampler_state = sample_plan.init(batched_data, sample_rng)
        for _ in range(max(0, int(self.lit_config.nqs_direct_psi_burn_in))):
            rng, sample_rng = jax.random.split(rng)
            batched_data, _, sampler_state = sample_plan.step(
                response_params,
                batched_data,
                sampler_state,
                sample_rng,
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

    def _invert_lit_spectrum(self, omega: np.ndarray, lit: np.ndarray):
        if not self.lit_config.inversion_enabled:
            return {"inversion_enabled": False}, []
        if omega.size < 2:
            logger.warning(
                "Skipping LIT inversion because at least two omega points are "
                "required; got %d.",
                omega.size,
            )
            return {
                "inversion_enabled": False,
                "inversion_skip_reason": "omega_points_lt_2",
            }, []

        response_omega = _inversion_response_grid(
            omega,
            eta=float(self.lit_config.eta),
            threshold=float(self.lit_config.inversion_threshold),
            response_max=self.lit_config.inversion_response_max,
            response_points=int(self.lit_config.inversion_response_points),
        )
        axis_response = []
        axis_fit_lit = []
        coefficients = []
        alpha1 = []
        alpha2 = []
        l2_regularization = []
        chi2 = []
        objective = []
        for axis_lit in lit:
            result = fit_lit_basis_expansion(
                omega,
                axis_lit,
                float(self.lit_config.eta),
                threshold=float(self.lit_config.inversion_threshold),
                response_omega=response_omega,
                basis_count=int(self.lit_config.inversion_basis_count),
                alpha1_grid=self.lit_config.inversion_alpha1_grid,
                alpha2_grid=self.lit_config.inversion_alpha2_grid,
                l2_grid=self.lit_config.inversion_l2_grid,
            )
            axis_response.append(result.response)
            axis_fit_lit.append(result.fit_lit)
            coefficients.append(result.coefficients)
            alpha1.append(result.alpha1)
            alpha2.append(result.alpha2)
            l2_regularization.append(result.l2_regularization)
            chi2.append(result.chi2)
            objective.append(result.objective)

        axis_response_arr = np.asarray(axis_response, dtype=np.float64)
        total_response = np.sum(axis_response_arr, axis=0)
        inversion_peaks = find_spectrum_peaks(
            response_omega,
            total_response,
            min_height_fraction=self.lit_config.peak_min_height_fraction,
        )
        return (
            {
                "inversion_enabled": True,
                "inversion_method": "regularized_basis_expansion",
                "inversion_response_omega": response_omega,
                "inversion_response": axis_response_arr,
                "total_inversion_response": total_response,
                "inversion_fit_lit": np.asarray(axis_fit_lit, dtype=np.float64),
                "inversion_coefficients": np.asarray(coefficients, dtype=np.float64),
                "inversion_alpha1": np.asarray(alpha1, dtype=np.float64),
                "inversion_alpha2": np.asarray(alpha2, dtype=np.float64),
                "inversion_l2_regularization": np.asarray(
                    l2_regularization,
                    dtype=np.float64,
                ),
                "inversion_chi2": np.asarray(chi2, dtype=np.float64),
                "inversion_objective": np.asarray(objective, dtype=np.float64),
                "inversion_threshold": float(self.lit_config.inversion_threshold),
                "inversion_basis_count": int(self.lit_config.inversion_basis_count),
            },
            inversion_peaks,
        )

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
        worker_limit = (
            int(self.lit_config.scan_parallel_workers)
            if self.lit_config.scan_parallel_workers > 0
            else len(device_ids)
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
        return max(1, min(len(device_ids), worker_limit, points_limit))

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

        parallel_root = self.save_path / "parallel_scan"
        parallel_root.mkdir(parents=True, exist_ok=True)
        base_config_path = parallel_root / "base_config.yaml"
        base_config_path.write_text(self.cfg.to_yaml())
        run_seed = (
            int(self.config.seed) if self.config.seed is not None else int(time.time())
        )

        logger.info(
            "Starting local-device LIT scan: workers=%d devices=%s blocks=%s",
            len(blocks),
            ",".join(device_ids[: len(blocks)]),
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
                )
                env = _parallel_worker_env(device_ids[worker_index])
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
                        device=device_ids[worker_index],
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
    ) -> list[str]:
        source_pool_dir = part_dir / "source_pools"
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
            "lit.inversion_enabled=false",
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

        total_broadened = np.sum(combined["broadened"], axis=0)
        peaks = find_spectrum_peaks(
            omega,
            total_broadened,
            min_height_fraction=self.lit_config.peak_min_height_fraction,
        )
        inversion_output, inversion_peaks = self._invert_lit_spectrum(
            omega,
            combined["lit"],
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
            nqs_direct_psi_burn_in=self.lit_config.nqs_direct_psi_burn_in,
            nqs_direct_psi_batches=self.lit_config.nqs_direct_psi_batches,
            nqs_direct_psi_stride=self.lit_config.nqs_direct_psi_stride,
            nqs_warm_start_omega=_optional_float(self.lit_config.nqs_warm_start_omega),
            nqs_warm_start_iterations=self.lit_config.nqs_warm_start_iterations,
            source_centers=np.mean(source_centers_blocks, axis=0),
            axis_source_norm=np.mean(axis_source_norm_blocks, axis=0),
            source_centers_blocks=source_centers_blocks,
            axis_source_norm_blocks=axis_source_norm_blocks,
            peak_energies=np.asarray([peak.energy for peak in peaks]),
            peak_intensities=np.asarray([peak.intensity for peak in peaks]),
            peak_indices=np.asarray([peak.index for peak in peaks]),
            inversion_peak_energies=np.asarray(
                [peak.energy for peak in inversion_peaks]
            ),
            inversion_peak_intensities=np.asarray(
                [peak.intensity for peak in inversion_peaks]
            ),
            inversion_peak_indices=np.asarray([peak.index for peak in inversion_peaks]),
            parallel_scan_enabled=True,
            parallel_scan_devices=np.asarray(devices, dtype=str),
            parallel_scan_blocks=np.asarray(
                [[int(block[0]), int(block[-1] + 1)] for block in blocks],
                dtype=np.int64,
            ),
            parallel_scan_part_paths=np.asarray([str(path) for path in paths]),
            **combined,
            **inversion_output,
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


def _parallel_worker_env(device_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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


def _inversion_response_grid(
    omega: np.ndarray,
    *,
    eta: float,
    threshold: float,
    response_max: float | None,
    response_points: int,
) -> np.ndarray:
    if response_points < 2:
        msg = f"lit.inversion_response_points must be at least 2, got {response_points}"
        raise ValueError(msg)
    upper = (
        float(response_max)
        if response_max is not None
        else max(
            float(np.max(omega) + 8.0 * eta),
            float(threshold + 1.25 * max(np.max(omega) - threshold, eta)),
        )
    )
    if upper <= threshold:
        msg = (
            "lit.inversion_response_max must exceed lit.inversion_threshold, got "
            f"{upper} <= {threshold}"
        )
        raise ValueError(msg)
    return np.linspace(float(threshold), upper, int(response_points))


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
