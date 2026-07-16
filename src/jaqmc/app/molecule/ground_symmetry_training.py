# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: DOC201,DOC501

r"""Low-cost spatial-symmetry training for molecular ground states.

The ordinary VMC energy update remains untouched.  On a configurable sparse
schedule, :class:`GroundSymmetryVMCWorkStage` adds a stateless, clipped
Euclidean update which minimizes the phase-aware covariance residual

.. math::

    \frac{|\Psi(gX)-\chi(g)\Psi(X)|^2}
         {|\Psi(gX)|^2+|\chi(g)\Psi(X)|^2}.

Only one operation and a small walker prefix are used by an update.  Full-bank
evaluation is deliberately infrequent.  Consequently the training cost does
not scale with the point-group order on every VMC iteration, while checkpoints
retain the exact :class:`~jaqmc.workflow.stage.vmc.VMCState` layout used by
ordinary training.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from operator import itemgetter, sub
from typing import Any

import jax
import numpy as np
import optax
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.ground_symmetry import scalar_log_covariance_residual
from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.response.source_sector import (
    discover_source_sector,
    transform_molecule_data,
)
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.workflow.stage.base import StageAbort
from jaqmc.workflow.stage.vmc import VMCState, VMCWorkStage

type PhaseLogPsi = Callable[[Params, MoleculeData], tuple[jnp.ndarray, jnp.ndarray]]


@configurable_dataclass
class GroundSymmetryConfig:
    """Configuration for symmetry-aware molecular ground-state training.

    ``irrep='a1'`` is appropriate only for a known nondegenerate totally
    symmetric state.  A nontrivial one-dimensional representation can be
    supplied with ``irrep='explicit'`` and one ``(real, imag)`` character per
    operation in the discovered finite group, including the identity.  A
    multidimensional or unknown representation is intentionally unsupported:
    silently forcing it into A1 would change the physics.
    """

    enabled: bool = False
    updates_enabled: bool = True
    global_mcmc_enabled: bool = True
    irrep: str = "a1"
    characters: tuple[tuple[float, float], ...] = ()
    geometry_tolerance: float = 1.0e-5
    linear_axial_order: int = 16
    linear_random_operation_pairs: int = 8
    atom_random_rotation_quartets: int = 8
    operation_seed: int = 1729

    pretrain_enabled: bool = True
    pretrain_update_interval: int = 4
    pretrain_learning_rate: float = 1.0e-3
    pretrain_source_weight_beta: float = 0.05
    pretrain_source_weight_warmup_steps: int = 500

    train_update_interval: int = 4
    train_learning_rate: float = 1.0e-3
    train_source_weight_beta: float = 0.25
    train_source_weight_warmup_steps: int = 5_000

    update_batch_size: int = 64
    source_weight_min: float = 1.0e-3
    source_weight_max: float = 10.0
    log_amplitude_huber_delta: float = 1.0
    covariance_training_weight: float = 0.1
    max_update_norm: float = 1.0e-3
    energy_update_trust_ratio: float = 0.05
    evaluation_interval: int = 500
    maximum_covariance: float = 1.0e-3
    mcmc_global_step_interval: int = 5

    def validate(self) -> None:  # noqa: C901
        mode = self.irrep.lower()
        if mode not in {"a1", "explicit"}:
            raise ValueError("ground_symmetry.irrep must be 'a1' or 'explicit'.")
        if mode == "a1" and self.characters:
            raise ValueError(
                "ground_symmetry.characters must be empty when irrep='a1'."
            )
        if mode == "explicit" and not self.characters:
            raise ValueError(
                "ground_symmetry.characters is required when irrep='explicit'."
            )
        if not np.isfinite(self.geometry_tolerance) or self.geometry_tolerance <= 0:
            raise ValueError("ground_symmetry.geometry_tolerance must be positive.")
        if self.linear_axial_order < 2:
            raise ValueError("ground_symmetry.linear_axial_order must be at least 2.")
        if self.linear_random_operation_pairs < 0:
            raise ValueError(
                "ground_symmetry.linear_random_operation_pairs must be nonnegative."
            )
        if self.atom_random_rotation_quartets < 0:
            raise ValueError(
                "ground_symmetry.atom_random_rotation_quartets must be nonnegative."
            )
        for name in ("pretrain_update_interval", "train_update_interval"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"ground_symmetry.{name} must be positive.")
        for name in (
            "pretrain_learning_rate",
            "train_learning_rate",
            "source_weight_min",
            "source_weight_max",
            "log_amplitude_huber_delta",
            "max_update_norm",
            "energy_update_trust_ratio",
            "maximum_covariance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"ground_symmetry.{name} must be positive.")
        if (
            not np.isfinite(self.covariance_training_weight)
            or self.covariance_training_weight < 0.0
        ):
            raise ValueError(
                "ground_symmetry.covariance_training_weight must be nonnegative."
            )
        for name in ("pretrain_source_weight_beta", "train_source_weight_beta"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"ground_symmetry.{name} must lie in [0, 1].")
        for name in (
            "pretrain_source_weight_warmup_steps",
            "train_source_weight_warmup_steps",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ground_symmetry.{name} must be nonnegative.")
        if self.update_batch_size < 1:
            raise ValueError("ground_symmetry.update_batch_size must be positive.")
        if self.source_weight_max < self.source_weight_min:
            raise ValueError(
                "ground_symmetry.source_weight_max must be >= source_weight_min."
            )
        if self.evaluation_interval < 1:
            raise ValueError("ground_symmetry.evaluation_interval must be positive.")
        if self.mcmc_global_step_interval < 1:
            raise ValueError(
                "ground_symmetry.mcmc_global_step_interval must be positive."
            )


@dataclass(frozen=True)
class GroundSymmetrySpecification:
    """Host-side symmetry data shared by training and the MCMC sampler."""

    label: str
    center: tuple[float, float, float]
    operations: tuple[tuple[tuple[float, float, float], ...], ...]
    characters: tuple[complex, ...]
    finite_group_operations: tuple[tuple[tuple[float, float, float], ...], ...]
    invariant_vector_projector: tuple[tuple[float, float, float], ...]

    @property
    def is_trivial(self) -> bool:
        return len(self.operations) == 0


@dataclass(frozen=True)
class GroundSymmetryStageSettings:
    """Stage-specific sparse-update schedule."""

    update_interval: int
    learning_rate: float
    source_weight_beta: float
    source_weight_warmup_steps: int


@dataclass(frozen=True)
class GroundSymmetryRuntime:
    """JAX numerical closure for ground-state covariance updates."""

    phase_logpsi: PhaseLogPsi
    specification: GroundSymmetrySpecification
    config: GroundSymmetryConfig

    def __post_init__(self) -> None:
        if self.specification.is_trivial:
            raise ValueError("GroundSymmetryRuntime requires a nontrivial operation.")

    @property
    def operation_count(self) -> int:
        return len(self.specification.operations)

    def loss_and_grad(
        self,
        params: Params,
        batched_data: BatchedData[MoleculeData],
        operation_index: jnp.ndarray,
        source_weight_beta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Params, dict[str, jnp.ndarray]]:
        """Evaluate one operation on a walker prefix and differentiate it."""
        batch = _slice_batched_prefix(batched_data, self.config.update_batch_size)
        operations = jnp.asarray(
            self.specification.operations,
            dtype=batch.data.electrons.dtype,
        )
        characters = jnp.asarray(self.specification.characters)
        operation = operations[operation_index]
        character = characters[operation_index]

        def objective(local_params):
            original = _batch_complex_logpsi(self.phase_logpsi, local_params, batch)
            transformed_data = transform_molecule_data(
                batch.data,
                operation,
                self.specification.center,
            )
            transformed = _batch_complex_logpsi(
                self.phase_logpsi,
                local_params,
                dataclasses.replace(batch, data=transformed_data),
            )
            residual = scalar_log_covariance_residual(
                original,
                transformed,
                character=character,
            )
            # The bounded covariance residual is the right physical diagnostic,
            # but it saturates near one and has an exponentially small gradient
            # when one symmetry image is many log units smaller than the other.
            # A robust log-amplitude term supplies a non-vanishing training
            # signal in precisely that tail-failure regime.  The bounded term
            # remains in the objective at a smaller weight to retain phase/sign
            # sensitivity and the correct local metric near covariance.
            log_amplitude_error = jnp.real(transformed - original)
            amplitude_residual = _huber_loss(
                log_amplitude_error,
                self.config.log_amplitude_huber_delta,
            )
            training_residual = (
                amplitude_residual + self.config.covariance_training_weight * residual
            )
            weights, ess_fraction, reference_dipole = self._source_weights(
                batch,
                source_weight_beta,
            )
            numerator = parallel_jax.pmean(jnp.mean(weights * training_residual))
            denominator = parallel_jax.pmean(jnp.mean(weights))
            weighted_loss = numerator / denominator
            covariance_loss = (
                parallel_jax.pmean(jnp.mean(weights * residual)) / denominator
            )
            log_amplitude_loss = (
                parallel_jax.pmean(jnp.mean(weights * amplitude_residual)) / denominator
            )
            unweighted_covariance_loss = parallel_jax.pmean(jnp.mean(residual))
            return weighted_loss, (
                covariance_loss,
                log_amplitude_loss,
                unweighted_covariance_loss,
                ess_fraction,
                reference_dipole,
            )

        (loss, auxiliary), grads = jax.value_and_grad(objective, has_aux=True)(
            parallel_jax.pvary(params)
        )
        grads = parallel_jax.pmean(grads)
        (
            covariance_loss,
            log_amplitude_loss,
            unweighted_covariance_loss,
            ess_fraction,
            reference_dipole,
        ) = auxiliary
        return (
            loss,
            grads,
            {
                "ground_symmetry_covariance_loss": covariance_loss,
                "ground_symmetry_log_amplitude_loss": log_amplitude_loss,
                "ground_symmetry_unweighted_covariance_loss": (
                    unweighted_covariance_loss
                ),
                "ground_symmetry_source_ess_fraction": ess_fraction,
                "ground_symmetry_reference_dipole_x": reference_dipole[0],
                "ground_symmetry_reference_dipole_y": reference_dipole[1],
                "ground_symmetry_reference_dipole_z": reference_dipole[2],
            },
        )

    def diagnose(
        self,
        params: Params,
        batched_data: BatchedData[MoleculeData],
        source_weight_beta: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Evaluate the complete configured operation bank without gradients."""
        batch = _slice_batched_prefix(batched_data, self.config.update_batch_size)
        original = _batch_complex_logpsi(self.phase_logpsi, params, batch)
        weights, ess_fraction, _ = self._source_weights(batch, source_weight_beta)
        operations = jnp.asarray(
            self.specification.operations,
            dtype=batch.data.electrons.dtype,
        )
        characters = jnp.asarray(self.specification.characters)

        def operation_losses(inputs):
            operation, character = inputs
            transformed_data = transform_molecule_data(
                batch.data,
                operation,
                self.specification.center,
            )
            transformed = _batch_complex_logpsi(
                self.phase_logpsi,
                params,
                dataclasses.replace(batch, data=transformed_data),
            )
            residual = scalar_log_covariance_residual(
                original,
                transformed,
                character=character,
            )
            weighted = parallel_jax.pmean(jnp.mean(weights * residual)) / (
                parallel_jax.pmean(jnp.mean(weights))
            )
            unweighted = parallel_jax.pmean(jnp.mean(residual))
            return weighted, unweighted

        weighted, unweighted = jax.lax.map(
            operation_losses,
            (operations, characters),
        )
        worst = jnp.argmax(weighted)
        return {
            "ground_symmetry_eval_mean": jnp.mean(weighted),
            "ground_symmetry_eval_max": weighted[worst],
            "ground_symmetry_eval_worst_operation": worst,
            "ground_symmetry_eval_unweighted_mean": jnp.mean(unweighted),
            "ground_symmetry_eval_unweighted_max": jnp.max(unweighted),
            "ground_symmetry_eval_source_ess_fraction": ess_fraction,
            "ground_symmetry_eval_pass": (
                weighted[worst] <= self.config.maximum_covariance
            ).astype(jnp.float32),
            "ground_symmetry_eval_active": jnp.asarray(1.0, dtype=weighted.dtype),
        }

    def _source_weights(
        self,
        batch: BatchedData[MoleculeData],
        beta: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dipole = -jnp.sum(batch.data.electrons, axis=-2)
        mean_dipole = parallel_jax.pmean(jnp.mean(dipole, axis=0))
        projector = jnp.asarray(
            self.specification.invariant_vector_projector,
            dtype=dipole.dtype,
        )
        reference = jnp.einsum(
            "ij,j->i",
            projector,
            mean_dipole,
            precision=jax.lax.Precision.HIGHEST,
        )
        relevance = jnp.sum((dipole - reference) ** 2, axis=-1)
        mean_relevance = parallel_jax.pmean(jnp.mean(relevance))
        normalized = relevance / jnp.where(mean_relevance > 0.0, mean_relevance, 1.0)
        normalized = jnp.where(
            mean_relevance > 0.0,
            normalized,
            jnp.ones_like(normalized),
        )
        weights = (1.0 - beta) + beta * normalized
        weights = jnp.clip(
            weights,
            self.config.source_weight_min,
            self.config.source_weight_max,
        )
        mean_weight = parallel_jax.pmean(jnp.mean(weights))
        mean_weight_square = parallel_jax.pmean(jnp.mean(weights**2))
        ess_fraction = mean_weight**2 / mean_weight_square
        return jax.lax.stop_gradient(weights), ess_fraction, reference


@dataclass(kw_only=True, eq=False)
class GroundSymmetryVMCWorkStage(VMCWorkStage):
    """VMC stage with a sparse, stateless post-optimizer symmetry update."""

    ground_symmetry: GroundSymmetryRuntime
    ground_symmetry_settings: GroundSymmetryStageSettings

    @classmethod
    def from_stage(
        cls,
        stage: VMCWorkStage,
        runtime: GroundSymmetryRuntime,
        settings: GroundSymmetryStageSettings,
    ) -> GroundSymmetryVMCWorkStage:
        return cls(
            config=stage.config,
            name=stage.name,
            wavefunction=stage.wavefunction,
            sample_plan=stage.sample_plan,
            estimators=stage.estimators,
            writers=stage.writers,
            optimizer=stage.optimizer,
            ground_symmetry=runtime,
            ground_symmetry_settings=settings,
        )

    def compute_step_with_symmetry(
        self,
        state: VMCState,
        rngs: PRNGKey,
        operation_index: jnp.ndarray,
        source_weight_beta: jnp.ndarray,
    ) -> tuple[dict[str, Any], VMCState]:
        """Apply the normal optimizer step followed by one clipped update."""
        stats, energy_state = super().compute_step(state, rngs)
        energy_updates = jax.tree.map(sub, energy_state.params, state.params)
        energy_update_norm = optax.tree.norm(energy_updates)
        loss, grads, symmetry_stats = self.ground_symmetry.loss_and_grad(
            energy_state.params,
            energy_state.batched_data,
            operation_index,
            source_weight_beta,
        )
        gradient_finite = jnp.all(
            jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads)])
        )
        grads = jax.tree.map(
            lambda grad: jnp.where(jnp.isfinite(grad), grad, jnp.zeros_like(grad)),
            grads,
        )
        grad_norm = optax.tree.norm(grads)
        requested_norm = self.ground_symmetry_settings.learning_rate * grad_norm
        allowed_norm = jnp.minimum(
            self.ground_symmetry.config.max_update_norm,
            self.ground_symmetry.config.energy_update_trust_ratio * energy_update_norm,
        )
        effective_norm = jnp.minimum(requested_norm, allowed_norm)
        scale = effective_norm / jnp.maximum(grad_norm, jnp.finfo(loss.dtype).tiny)
        valid = jnp.isfinite(loss) & gradient_finite & jnp.isfinite(grad_norm)
        scale = jnp.where(valid, scale, 0.0)
        symmetry_updates = jax.tree.map(lambda grad: -scale * grad, grads)
        params = optax.apply_updates(energy_state.params, symmetry_updates)
        stats.update(symmetry_stats)
        stats.update(
            {
                "ground_symmetry_active": jnp.asarray(1.0, dtype=loss.dtype),
                "ground_symmetry_loss": loss,
                "ground_symmetry_operation": operation_index,
                "ground_symmetry_grad_norm": grad_norm,
                "ground_symmetry_update_norm": optax.tree.norm(symmetry_updates),
                "ground_symmetry_energy_update_norm": energy_update_norm,
                "ground_symmetry_valid": valid.astype(jnp.float32),
            }
        )
        return stats, dataclasses.replace(energy_state, params=params)

    def loop(self, state: VMCState, initial_step: int, rngs):
        """Run ordinary steps and sparse symmetry/diagnostic kernels.

        Yields:
            ``(step, state)`` after every completed optimization iteration.
        """
        check_vma = self.config.check_vma
        if isinstance(self.optimizer, KFACOptimizer) and check_vma:
            self.logger.warning("Disabling check_vma (incompatible with KFAC).")
            check_vma = False

        partition = state.partition()
        split_rngs = parallel_jax.jit_sharded(
            lambda r: tuple(jax.random.split(r)),
            in_specs=parallel_jax.DATA_PARTITION,
            out_specs=parallel_jax.DATA_PARTITION,
        )
        compute = parallel_jax.jit_sharded(
            self.compute_step,
            in_specs=(partition, parallel_jax.DATA_PARTITION),
            out_specs=(parallel_jax.SHARE_PARTITION, partition),
            check_vma=check_vma,
            donate_argnums=0,
        )
        compute_with_symmetry = parallel_jax.jit_sharded(
            self.compute_step_with_symmetry,
            in_specs=(
                partition,
                parallel_jax.DATA_PARTITION,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
            ),
            out_specs=(parallel_jax.SHARE_PARTITION, partition),
            check_vma=check_vma,
            donate_argnums=0,
        )
        diagnose = parallel_jax.jit_sharded(
            self.ground_symmetry.diagnose,
            in_specs=(
                parallel_jax.SHARE_PARTITION,
                state.batched_data.partition_spec,
                parallel_jax.SHARE_PARTITION,
            ),
            out_specs=parallel_jax.SHARE_PARTITION,
            check_vma=check_vma,
        )

        if initial_step == 0:
            state, rngs = self.burn_in(state, rngs)

        interval = self.ground_symmetry_settings.update_interval
        for step in range(initial_step, self.config.iterations):
            rngs, sub_rngs = split_rngs(rngs)
            ramp_steps = self.ground_symmetry_settings.source_weight_warmup_steps
            ramp = 1.0 if ramp_steps == 0 else min(1.0, (step + 1) / ramp_steps)
            beta = jnp.asarray(
                ramp * self.ground_symmetry_settings.source_weight_beta,
                dtype=state.batched_data.data.electrons.dtype,
            )
            run_symmetry = (
                self.ground_symmetry.config.updates_enabled and step % interval == 0
            )
            if run_symmetry:
                operation_index = jnp.asarray(
                    (step // interval) % self.ground_symmetry.operation_count,
                    dtype=jnp.int32,
                )
                stats, state = compute_with_symmetry(
                    state,
                    sub_rngs,
                    operation_index,
                    beta,
                )
            else:
                stats, state = compute(state, sub_rngs)
                _fill_symmetry_step_defaults(stats)

            if step == 0 or (
                (step + 1) % self.ground_symmetry.config.evaluation_interval == 0
            ):
                stats.update(diagnose(state.params, state.batched_data, beta))
            else:
                _fill_symmetry_eval_defaults(stats)

            self.writers.write(step, stats)
            if run_symmetry and not bool(
                np.asarray(jax.device_get(stats["ground_symmetry_valid"]))
            ):
                raise StageAbort(step, state)
            if self._has_nan(stats):
                raise StageAbort(step, state)
            yield step, state


def build_ground_symmetry_specification(
    atoms: np.ndarray,
    charges: np.ndarray,
    config: GroundSymmetryConfig,
) -> GroundSymmetrySpecification:
    """Discover exact geometry operations and build a training operation bank."""
    config.validate()
    finite_sector = discover_source_sector(
        atoms,
        charges,
        tolerance=config.geometry_tolerance,
        axial_order=config.linear_axial_order,
    )
    finite_operations = np.asarray(finite_sector.operations, dtype=np.float64)
    finite_characters = _resolve_finite_characters(finite_operations, config)

    operations = finite_operations
    characters = finite_characters
    label = finite_sector.label
    if (
        finite_sector.label == "atom_Oh"
        and config.atom_random_rotation_quartets > 0
        and config.irrep.lower() == "a1"
    ):
        random_operations = _atomic_o3_training_bank(
            config.atom_random_rotation_quartets,
            seed=config.operation_seed,
        )
        operations = _append_unique_operations(
            finite_operations,
            random_operations,
            tolerance=config.geometry_tolerance,
        )
        characters = np.ones(operations.shape[0], dtype=np.complex128)
        label = f"atom_Oh+O3bank_{operations.shape[0]}"
    elif (
        finite_sector.label.startswith("linear_")
        and config.linear_random_operation_pairs > 0
        and config.irrep.lower() == "a1"
    ):
        random_operations = _linear_training_bank(
            np.asarray(atoms, dtype=np.float64),
            np.asarray(finite_sector.center, dtype=np.float64),
            pairs=config.linear_random_operation_pairs,
            allow_axis_reversal=finite_sector.label.startswith("linear_D"),
            seed=config.operation_seed,
        )
        operations = _append_unique_operations(
            finite_operations,
            random_operations,
            tolerance=config.geometry_tolerance,
        )
        characters = np.ones(operations.shape[0], dtype=np.complex128)
        label = f"{finite_sector.label}+O2bank_{operations.shape[0]}"

    identity_mask = np.all(
        np.isclose(
            operations,
            np.eye(3),
            rtol=0.0,
            atol=config.geometry_tolerance,
        ),
        axis=(1, 2),
    )
    active_operations = operations[~identity_mask]
    active_characters = characters[~identity_mask]
    projector = _invariant_vector_projector(operations, config.geometry_tolerance)
    return GroundSymmetrySpecification(
        label=label,
        center=_vector_tuple(finite_sector.center),
        operations=_operation_tuple(active_operations),
        characters=tuple(complex(value) for value in active_characters),
        finite_group_operations=_operation_tuple(finite_operations),
        invariant_vector_projector=_matrix_tuple(projector),
    )


def stage_settings(
    config: GroundSymmetryConfig,
    *,
    pretrain: bool,
) -> GroundSymmetryStageSettings:
    if pretrain:
        return GroundSymmetryStageSettings(
            update_interval=config.pretrain_update_interval,
            learning_rate=config.pretrain_learning_rate,
            source_weight_beta=config.pretrain_source_weight_beta,
            source_weight_warmup_steps=config.pretrain_source_weight_warmup_steps,
        )
    return GroundSymmetryStageSettings(
        update_interval=config.train_update_interval,
        learning_rate=config.train_learning_rate,
        source_weight_beta=config.train_source_weight_beta,
        source_weight_warmup_steps=config.train_source_weight_warmup_steps,
    )


def validate_ground_symmetry_batching(
    config: GroundSymmetryConfig,
    workflow_batch_size: int,
) -> None:
    """Validate the configured global auxiliary batch against device sharding."""
    device_count = jax.device_count()
    if config.update_batch_size > workflow_batch_size:
        raise ValueError(
            "ground_symmetry.update_batch_size cannot exceed workflow.batch_size "
            f"({config.update_batch_size} > {workflow_batch_size})."
        )
    if config.update_batch_size % device_count != 0:
        raise ValueError(
            "ground_symmetry.update_batch_size must be divisible by the global "
            f"JAX device count ({device_count})."
        )


def _batch_complex_logpsi(
    phase_logpsi: PhaseLogPsi,
    params: Params,
    batch: BatchedData[MoleculeData],
) -> jnp.ndarray:
    phase, log_abs = jax.vmap(
        phase_logpsi,
        in_axes=(None, batch.vmap_axis),
    )(params, batch.data)
    complex_dtype = jnp.result_type(phase, log_abs, jnp.complex64)
    phase = jnp.asarray(phase, dtype=complex_dtype)
    phase_angle = jnp.angle(phase)
    return jnp.asarray(log_abs, dtype=complex_dtype) + 1j * phase_angle


def _huber_loss(error: jnp.ndarray, delta: float) -> jnp.ndarray:
    absolute = jnp.abs(error)
    delta_array = jnp.asarray(delta, dtype=absolute.dtype)
    return jnp.where(
        absolute <= delta_array,
        0.5 * error**2,
        delta_array * (absolute - 0.5 * delta_array),
    )


def _slice_batched_prefix(
    batch: BatchedData[MoleculeData], requested_size: int
) -> BatchedData[MoleculeData]:
    local_requested_size = int(requested_size) // jax.device_count()
    size = min(batch.batch_size, local_requested_size)
    data = dataclasses.replace(
        batch.data,
        **{
            field: jax.tree.map(itemgetter(slice(0, size)), batch.data[field])
            for field in batch.fields_with_batch
        },
    )
    return dataclasses.replace(batch, data=data)


def _resolve_finite_characters(
    operations: np.ndarray,
    config: GroundSymmetryConfig,
) -> np.ndarray:
    if config.irrep.lower() == "a1":
        return np.ones(operations.shape[0], dtype=np.complex128)
    character_parts = np.asarray(config.characters, dtype=np.float64)
    characters = character_parts[:, 0] + 1j * character_parts[:, 1]
    if characters.shape != (operations.shape[0],):
        raise ValueError(
            "ground_symmetry.characters must have one entry per discovered "
            f"finite-group operation ({operations.shape[0]}), got {characters.shape}."
        )
    tolerance = config.geometry_tolerance * 10.0
    if not np.all(np.isfinite(characters)) or not np.allclose(
        np.abs(characters), 1.0, rtol=0.0, atol=tolerance
    ):
        raise ValueError("ground_symmetry.characters must be finite unit phases.")
    identity = _find_operation(operations, np.eye(3), config.geometry_tolerance)
    if not np.isclose(characters[identity], 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("The identity operation must have character 1.")
    for left_index, left in enumerate(operations):
        for right_index, right in enumerate(operations):
            product_index = _find_operation(
                operations,
                left @ right,
                config.geometry_tolerance,
            )
            if not np.isclose(
                characters[left_index] * characters[right_index],
                characters[product_index],
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    "ground_symmetry.characters do not form a one-dimensional "
                    "representation of the discovered group."
                )
    return characters


def _atomic_o3_training_bank(quartets: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    operations: list[np.ndarray] = []
    for _ in range(quartets):
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        w, x, y, z = quaternion
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        operations.extend((rotation, rotation.T, -rotation, -rotation.T))
    return np.asarray(operations)


def _linear_training_bank(
    atoms: np.ndarray,
    center: np.ndarray,
    *,
    pairs: int,
    allow_axis_reversal: bool,
    seed: int,
) -> np.ndarray:
    centered = atoms - center
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axis = right[0]
    reference = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    basis = np.stack((first, second, axis), axis=1)
    rng = np.random.default_rng(seed)
    operations: list[np.ndarray] = []
    for _ in range(pairs):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        if rng.integers(0, 2):
            perpendicular = np.asarray([[cosine, sine], [sine, -cosine]])
        else:
            perpendicular = np.asarray([[cosine, -sine], [sine, cosine]])
        axis_sign = -1.0 if allow_axis_reversal and rng.integers(0, 2) else 1.0
        local = np.zeros((3, 3), dtype=np.float64)
        local[:2, :2] = perpendicular
        local[2, 2] = axis_sign
        operation = basis @ local @ basis.T
        operations.extend((operation, operation.T))
    return np.asarray(operations)


def _append_unique_operations(
    base: np.ndarray,
    candidates: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    result = [operation for operation in base]
    for candidate in candidates:
        if not any(
            np.allclose(candidate, existing, rtol=0.0, atol=tolerance)
            for existing in result
        ):
            result.append(candidate)
    return np.asarray(result)


def _invariant_vector_projector(
    operations: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    constraints = np.concatenate(
        [operation - np.eye(3) for operation in operations],
        axis=0,
    )
    _, singular_values, right = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular_values > tolerance * max(1.0, singular_values[0])))
    nullspace = right[rank:].T
    if nullspace.size == 0:
        return np.zeros((3, 3), dtype=np.float64)
    return nullspace @ nullspace.T


def _find_operation(
    operations: np.ndarray,
    candidate: np.ndarray,
    tolerance: float,
) -> int:
    matches = np.all(
        np.isclose(operations, candidate, rtol=0.0, atol=tolerance),
        axis=(1, 2),
    )
    indices = np.flatnonzero(matches)
    if indices.size != 1:
        raise ValueError("Discovered operations are not a unique closed finite group.")
    return int(indices[0])


def _operation_tuple(
    operations: np.ndarray,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    return tuple(_matrix_tuple(operation) for operation in operations)


def _matrix_tuple(
    matrix: np.ndarray,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        _vector_tuple(matrix[0]),
        _vector_tuple(matrix[1]),
        _vector_tuple(matrix[2]),
    )


def _vector_tuple(values: Any) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _fill_symmetry_step_defaults(stats: dict[str, Any]) -> None:
    defaults = {
        "ground_symmetry_active": 0.0,
        "ground_symmetry_loss": 0.0,
        "ground_symmetry_covariance_loss": 0.0,
        "ground_symmetry_log_amplitude_loss": 0.0,
        "ground_symmetry_unweighted_covariance_loss": 0.0,
        "ground_symmetry_source_ess_fraction": 0.0,
        "ground_symmetry_reference_dipole_x": 0.0,
        "ground_symmetry_reference_dipole_y": 0.0,
        "ground_symmetry_reference_dipole_z": 0.0,
        "ground_symmetry_operation": -1,
        "ground_symmetry_grad_norm": 0.0,
        "ground_symmetry_update_norm": 0.0,
        "ground_symmetry_energy_update_norm": 0.0,
        "ground_symmetry_valid": 1.0,
    }
    dtype = _stats_dtype(stats)
    stats.update(
        {key: jnp.asarray(value, dtype=dtype) for key, value in defaults.items()}
    )


def _fill_symmetry_eval_defaults(stats: dict[str, Any]) -> None:
    defaults = {
        "ground_symmetry_eval_mean": 0.0,
        "ground_symmetry_eval_max": 0.0,
        "ground_symmetry_eval_worst_operation": -1,
        "ground_symmetry_eval_unweighted_mean": 0.0,
        "ground_symmetry_eval_unweighted_max": 0.0,
        "ground_symmetry_eval_source_ess_fraction": 0.0,
        "ground_symmetry_eval_pass": 0.0,
        "ground_symmetry_eval_active": 0.0,
    }
    dtype = _stats_dtype(stats)
    stats.update(
        {key: jnp.asarray(value, dtype=dtype) for key, value in defaults.items()}
    )


def _stats_dtype(stats: dict[str, Any]) -> jnp.dtype:
    for leaf in jax.tree.leaves(stats):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            return dtype
    return jnp.dtype(jnp.float32)
