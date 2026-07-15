# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Direct NQS Lorentz integral transform for molecular dipole response.

This module follows the correction-vector formulation used in
arXiv:2504.20195: a full neural response wavefunction ``Psi_L`` is optimized so
that ``(H - E0 - omega - i eta) Psi_L`` is parallel to the source
``Phi = (D - <D>) Psi_0``.  Estimators are written for samples drawn from the
source density ``pi_Phi``; the optimizer can then reuse one source pool across
all response energies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple
from zipfile import BadZipFile

import h5py
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from upath import UPath

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.geometry import obc
from jaqmc.utils.checkpoint import _pytree_key_path, from_npz
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.logdet import LogDet
from jaqmc.wavefunction.output.orbital import SplitChannelDense

type ResponseApply = Callable[[Params, MoleculeData], jnp.ndarray]
type GroundLogPsi = Callable[[Params, MoleculeData], jnp.ndarray]


class NQSLITStats(NamedTuple):
    """Diagnostics for one source-sampled NQS-LIT batch."""

    loss: jnp.ndarray
    fidelity: jnp.ndarray
    reverse_kl: jnp.ndarray
    signed_lit: jnp.ndarray
    lit: jnp.ndarray
    broadened: jnp.ndarray
    source_norm: jnp.ndarray
    action_norm: jnp.ndarray
    log_ratio_norm: jnp.ndarray
    correction_overlap: jnp.ndarray
    normalization: jnp.ndarray
    residual_norm: jnp.ndarray
    equation_relative_residual: jnp.ndarray
    ground_energy_mean: jnp.ndarray
    correction_norm: jnp.ndarray
    shifted_hamiltonian_norm: jnp.ndarray
    error_d: jnp.ndarray
    reweight_ess: jnp.ndarray
    reweight_ess_fraction: jnp.ndarray
    invalid_sample_fraction: jnp.ndarray
    estimator_mode: jnp.ndarray
    direct_hloc_rmse: jnp.ndarray
    direct_hloc_std: jnp.ndarray
    direct_hloc_sem: jnp.ndarray


class NQSLITSourceSums(NamedTuple):
    """Scale-aware source-sampled sums for chunked NQS-LIT evaluation.

    The five ratio-moment fields are expressed in units of ``ratio_scale``.
    This keeps the fidelity, reverse-KL, and ESS estimators finite even when
    the unscaled action/source ratio has a very large dynamic range.
    """

    sample_count: jnp.ndarray
    weight_sum: jnp.ndarray
    valid_sample_count: jnp.ndarray
    ratio_scale: jnp.ndarray
    ratio_sum: jnp.ndarray
    ratio_abs2_sum: jnp.ndarray
    psi_weight_sum: jnp.ndarray
    psi_weight_sq_sum: jnp.ndarray
    psi_log_ratio_abs2_sum: jnp.ndarray
    response_conj_over_source_sum: jnp.ndarray
    response_over_source_abs2_sum: jnp.ndarray
    hbar_over_source_sum: jnp.ndarray
    hbar_over_source_abs2_sum: jnp.ndarray
    ground_energy_sum: jnp.ndarray


class MolecularResponseFermiNet(nn.Module):
    """Complex FermiNet-style response wavefunction ``Psi_L``."""

    nspins: tuple[int, int]
    ndets: int = 16
    hidden_dims_single: tuple[int, ...] = (256, 256, 256, 256)
    hidden_dims_double: tuple[int, ...] = (32, 32, 32, 32)
    use_last_layer: bool = False
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True

    def setup(self) -> None:
        self.feature_layer = MoleculeFeatures()
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        self.backbone_layer = FermiLayers(
            self.nspins,
            hidden_dims,
            use_last_layer=self.use_last_layer,
        )
        self.orbital_layer = _ComplexOrbitalProjection(
            nspins=self.nspins,
            ndets=self.ndets,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=self.ndets,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self.logdet_layer = LogDet()

    def __call__(self, data: MoleculeData) -> jnp.ndarray:
        embedding = self.feature_layer(data.electrons, data.atoms)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"],
            embedding["ee_features"],
        )
        orbitals = self.orbital_layer(h_one)
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"],
            embedding["r_ae"],
        )
        return self.logdet_layer(orbitals)["logpsi"]


class _ComplexOrbitalProjection(nn.Module):
    """Project FermiNet electron features to complex orbital matrices."""

    nspins: tuple[int, int]
    ndets: int
    orbitals_spin_split: bool = True
    use_bias: bool = False

    @nn.compact
    def __call__(self, h_one: jnp.ndarray) -> jnp.ndarray:
        n_electrons = sum(self.nspins)
        features = [self.ndets, n_electrons, 2]
        active_spins = [spin for spin in self.nspins if spin > 0]
        if self.orbitals_spin_split and len(active_spins) > 1:
            orbitals = SplitChannelDense(self.nspins, features, self.use_bias)(h_one)
        else:
            orbitals = nn.DenseGeneral(features, use_bias=self.use_bias)(h_one)
        orbitals = jnp.transpose(orbitals, (1, 0, 2, 3))
        return orbitals[..., 0] + 1j * orbitals[..., 1]


def molecular_electronic_dipole(data: MoleculeData, axis: int) -> jnp.ndarray:
    """Electronic dipole component for a fixed-nuclei transition source."""
    return -jnp.sum(data.electrons[:, int(axis)])


def molecular_potential_energy(data: MoleculeData) -> jnp.ndarray:
    """Molecular Coulomb potential energy for one walker."""
    nelec = data.electrons.shape[0]
    natom = data.atoms.shape[0]
    r_ae = obc.pair_displacements_between(data.electrons, data.atoms)[1]
    r_ee = obc.pair_displacements_within(data.electrons)[1] + jnp.eye(nelec)
    r_aa = obc.pair_displacements_within(data.atoms)[1] + jnp.eye(natom)
    return (
        jnp.sum(-jnp.ones(nelec)[:, None] * data.charges / r_ae)
        + jnp.sum(jnp.triu(1 / r_ee, k=1))
        + jnp.sum(jnp.triu(data.charges * data.charges[:, None] / r_aa, k=1))
    )


def _flat_electrons(data: MoleculeData) -> tuple[jnp.ndarray, tuple[int, ...]]:
    shape = data.electrons.shape
    return jnp.ravel(data.electrons), shape


def _with_flat_electrons(
    data: MoleculeData, flat_electrons: jnp.ndarray, shape: tuple[int, ...]
) -> MoleculeData:
    return data.merge({"electrons": jnp.reshape(flat_electrons, shape)})


def ground_local_energy(
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    data: MoleculeData,
) -> jnp.ndarray:
    """Evaluate the ground-state local energy by coordinate derivatives."""
    flat, shape = _flat_electrons(data)

    def logpsi_flat(x):
        value = ground_logpsi(ground_params, _with_flat_electrons(data, x, shape))
        return jnp.real(value)

    grad_log = jax.grad(logpsi_flat)(flat)
    hess_log = jax.hessian(logpsi_flat)(flat)
    kinetic = -0.5 * (jnp.trace(hess_log) + jnp.dot(grad_log, grad_log))
    return kinetic + molecular_potential_energy(data)


def local_action_ratio(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    data: MoleculeData,
    *,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``((H-z) Psi_L / Psi_0, Psi_L / Psi_0, E_L[Psi_L])``."""
    flat, shape = _flat_electrons(data)

    def ground_log_flat(x):
        return ground_logpsi(ground_params, _with_flat_electrons(data, x, shape))

    def response_log_flat(x):
        return response_apply(response_params, _with_flat_electrons(data, x, shape))

    response_log = response_log_flat(flat)
    log_ratio = response_log - ground_log_flat(flat)
    response_ratio = jnp.exp(log_ratio)
    grad_re = jax.grad(lambda x: jnp.real(response_log_flat(x)))(flat)
    grad_im = jax.grad(lambda x: jnp.imag(response_log_flat(x)))(flat)
    grad_log_response = grad_re + 1j * grad_im
    hess_re = jax.hessian(lambda x: jnp.real(response_log_flat(x)))(flat)
    hess_im = jax.hessian(lambda x: jnp.imag(response_log_flat(x)))(flat)
    lap_log_response = jnp.trace(hess_re) + 1j * jnp.trace(hess_im)
    local_energy = -0.5 * (
        lap_log_response + jnp.dot(grad_log_response, grad_log_response)
    ) + molecular_potential_energy(data)
    shift = jnp.asarray(omega, dtype=response_ratio.real.dtype) + 1j * jnp.asarray(
        eta,
        dtype=response_ratio.real.dtype,
    )
    action = response_ratio * (
        local_energy - jnp.asarray(ground_energy, dtype=response_ratio.dtype) - shift
    )
    return action, response_ratio, local_energy


def nqs_lit_source_sampled_stats(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    batched_data: BatchedData[MoleculeData],
    *,
    axis: int,
    source_center: float | jnp.ndarray,
    source_norm: float | jnp.ndarray,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    source_floor: float | jnp.ndarray = 0.0,
    eps: float = 1e-12,
) -> NQSLITStats:
    """Compute fidelity and LIT observables from ``pi_Phi`` samples."""
    sums = nqs_lit_source_sampled_sums(
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        axis=axis,
        source_center=source_center,
        ground_energy=ground_energy,
        omega=omega,
        eta=eta,
        source_floor=source_floor,
        eps=eps,
    )
    return nqs_lit_stats_from_source_sums(
        sums,
        source_norm=source_norm,
        omega=omega,
        eta=eta,
        eps=eps,
    )


def nqs_lit_source_sampled_sums(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    batched_data: BatchedData[MoleculeData],
    *,
    axis: int,
    source_center: float | jnp.ndarray,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    source_floor: float | jnp.ndarray = 0.0,
    eps: float = 1e-12,
) -> NQSLITSourceSums:
    """Return additive raw sums for source-sampled NQS-LIT observables."""
    data = batched_data.data
    action, response_ratio, eloc_response = jax.vmap(
        lambda one: local_action_ratio(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            one,
            ground_energy=ground_energy,
            omega=omega,
            eta=eta,
        ),
        in_axes=(batched_data.vmap_axis,),
    )(data)
    dipole = jax.vmap(
        lambda one: molecular_electronic_dipole(one, axis),
        in_axes=(batched_data.vmap_axis,),
    )(data)
    source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
    floor = jnp.asarray(source_floor, dtype=dipole.dtype)
    sampled_source_abs = jnp.maximum(jnp.abs(source), floor)
    source_weight = (jnp.abs(source) / jnp.maximum(sampled_source_abs, eps)) ** 2
    base_finite = (
        jnp.isfinite(jnp.real(action))
        & jnp.isfinite(jnp.imag(action))
        & jnp.isfinite(jnp.real(response_ratio))
        & jnp.isfinite(jnp.imag(response_ratio))
        & jnp.isfinite(source)
        & jnp.isfinite(source_weight)
    )
    safe_source = jnp.where(
        jnp.abs(source) > eps,
        source,
        jnp.asarray(eps, dtype=source.dtype) * jnp.where(source < 0, -1.0, 1.0),
    )
    ratio = action / safe_source
    finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
    finite = base_finite & finite_ratio
    action = jnp.where(finite, action, jnp.asarray(0.0, dtype=action.dtype))
    response_ratio = jnp.where(
        finite,
        response_ratio,
        jnp.asarray(0.0, dtype=response_ratio.dtype),
    )
    source_weight = jnp.where(
        finite,
        source_weight,
        jnp.asarray(0.0, dtype=source_weight.dtype),
    )
    ratio = jnp.where(finite, ratio, jnp.asarray(0.0, dtype=ratio.dtype))
    ratio_abs = jnp.where(source_weight > 0.0, jnp.abs(ratio), 0.0)
    max_ratio_abs = jnp.max(ratio_abs)
    ratio_scale = jnp.where(
        max_ratio_abs > 0.0,
        max_ratio_abs,
        jnp.asarray(1.0, dtype=ratio_abs.dtype),
    )
    scaled_ratio = ratio / jax.lax.stop_gradient(ratio_scale)
    scaled_ratio_abs2 = jnp.abs(scaled_ratio) ** 2
    psi_weight_unnormalized = source_weight * scaled_ratio_abs2
    log_ratio_abs2 = 2.0 * jnp.log(
        jnp.maximum(jnp.abs(scaled_ratio), jnp.asarray(eps, dtype=ratio_abs.dtype))
    )
    shift = jnp.asarray(omega, dtype=response_ratio.real.dtype) + 1j * jnp.asarray(
        eta,
        dtype=response_ratio.real.dtype,
    )
    hbar_response_ratio = action + shift * response_ratio
    response_over_source = response_ratio / safe_source
    hbar_over_source = hbar_response_ratio / safe_source
    eloc_finite = jnp.isfinite(jnp.real(eloc_response))
    eloc_response = jnp.where(
        eloc_finite,
        eloc_response,
        jnp.asarray(0.0, dtype=eloc_response.dtype),
    )
    sample_count = jnp.asarray(action.shape[0], dtype=source_weight.real.dtype)
    return NQSLITSourceSums(
        sample_count=sample_count,
        weight_sum=jnp.sum(source_weight),
        valid_sample_count=jnp.sum(finite),
        ratio_scale=ratio_scale,
        ratio_sum=jnp.sum(source_weight * scaled_ratio),
        ratio_abs2_sum=jnp.sum(source_weight * scaled_ratio_abs2),
        psi_weight_sum=jnp.sum(psi_weight_unnormalized),
        psi_weight_sq_sum=jnp.sum(psi_weight_unnormalized**2),
        psi_log_ratio_abs2_sum=jnp.sum(psi_weight_unnormalized * log_ratio_abs2),
        response_conj_over_source_sum=jnp.sum(
            source_weight * jnp.conj(response_ratio) / safe_source
        ),
        response_over_source_abs2_sum=jnp.sum(
            source_weight * jnp.abs(response_over_source) ** 2
        ),
        hbar_over_source_sum=jnp.sum(source_weight * hbar_over_source),
        hbar_over_source_abs2_sum=jnp.sum(
            source_weight * jnp.abs(hbar_over_source) ** 2
        ),
        ground_energy_sum=jnp.real(jnp.sum(eloc_response)),
    )


def nqs_lit_stats_from_source_sums(
    sums: NQSLITSourceSums,
    *,
    source_norm: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    eps: float = 1e-12,
) -> NQSLITStats:
    """Convert additive source-sampled sums into standard diagnostics."""
    real_dtype = sums.weight_sum.dtype
    safe_weight_sum = jnp.maximum(
        sums.weight_sum,
        jnp.asarray(eps, dtype=real_dtype),
    )
    ratio_scale = jnp.maximum(
        sums.ratio_scale,
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    scaled_normalization = sums.ratio_sum / safe_weight_sum
    scaled_ratio_norm = sums.ratio_abs2_sum / safe_weight_sum
    has_action_mass = (
        jnp.isfinite(scaled_ratio_norm)
        & jnp.isfinite(ratio_scale)
        & (sums.psi_weight_sum > jnp.asarray(0.0, dtype=real_dtype))
    )
    normalization = ratio_scale * scaled_normalization
    ratio_norm = ratio_scale**2 * scaled_ratio_norm
    log_ratio_norm = 2.0 * jnp.log(ratio_scale) + jnp.log(
        jnp.maximum(
            scaled_ratio_norm,
            jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
        )
    )
    reweight_ess = sums.psi_weight_sum**2 / jnp.maximum(
        sums.psi_weight_sq_sum,
        jnp.asarray(eps, dtype=real_dtype),
    )
    valid_sample_count = jnp.maximum(
        sums.valid_sample_count,
        jnp.asarray(1, dtype=real_dtype),
    )
    reweight_ess_fraction = reweight_ess / valid_sample_count
    fidelity = (jnp.abs(scaled_normalization) ** 2) / jnp.maximum(
        scaled_ratio_norm,
        jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
    )
    fidelity = jnp.clip(jnp.real(fidelity), 0.0, 1.0)
    reverse_kl = sums.psi_log_ratio_abs2_sum / jnp.maximum(
        sums.psi_weight_sum, jnp.asarray(eps, dtype=real_dtype)
    ) - jnp.log(
        jnp.maximum(
            scaled_ratio_norm,
            jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
        )
    )
    reverse_kl = jnp.where(
        has_action_mass,
        jnp.maximum(
            jnp.real(reverse_kl),
            jnp.asarray(0.0, dtype=real_dtype),
        ),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    loss = 1.0 - fidelity

    phi_norm = jnp.asarray(source_norm, dtype=real_dtype)
    action_norm = phi_norm * ratio_norm
    correction_overlap = phi_norm * sums.response_conj_over_source_sum / safe_weight_sum
    safe_normalization = normalization + jnp.asarray(eps, dtype=normalization.dtype)
    normalized_overlap = correction_overlap / jnp.conj(safe_normalization)
    signed_lit = -jnp.imag(normalized_overlap) / jnp.asarray(eta)
    lit = jnp.maximum(signed_lit, 0.0)
    broadened = jnp.asarray(eta) * lit / jnp.pi
    safe_scaled_normalization = scaled_normalization + jnp.asarray(
        eps,
        dtype=scaled_normalization.dtype,
    )
    residual_mean = (
        scaled_ratio_norm
        / jnp.maximum(
            jnp.abs(safe_scaled_normalization) ** 2,
            jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype),
        )
        - 2.0 * jnp.real(scaled_normalization / safe_scaled_normalization)
        + 1.0
    )
    residual_mean = jnp.maximum(
        jnp.real(residual_mean),
        jnp.asarray(0.0, dtype=real_dtype),
    )
    residual_norm = phi_norm * residual_mean
    equation_relative_residual = jnp.sqrt(residual_mean)
    correction_norm, shifted_hamiltonian_norm, error_d = _source_sampled_error_d_sums(
        sums,
        phi_norm,
        safe_normalization,
        normalized_overlap,
        omega=omega,
        eta=eta,
        eps=eps,
    )
    safe_sample_count = jnp.maximum(
        sums.sample_count,
        jnp.asarray(1, dtype=real_dtype),
    )
    invalid_sample_fraction = jnp.maximum(
        1.0 - sums.valid_sample_count / safe_sample_count,
        jnp.asarray(0.0, dtype=real_dtype),
    )
    invalid_sample_fraction = jnp.where(
        has_action_mass,
        invalid_sample_fraction,
        jnp.asarray(1.0, dtype=real_dtype),
    )
    nan_real = jnp.asarray(jnp.nan, dtype=real_dtype)
    return NQSLITStats(
        loss=jnp.real(loss),
        fidelity=fidelity,
        reverse_kl=reverse_kl,
        signed_lit=jnp.real(signed_lit),
        lit=jnp.real(lit),
        broadened=jnp.real(broadened),
        source_norm=jnp.real(phi_norm),
        action_norm=jnp.real(action_norm),
        log_ratio_norm=jnp.real(log_ratio_norm),
        correction_overlap=correction_overlap,
        normalization=normalization,
        residual_norm=jnp.real(residual_norm),
        equation_relative_residual=jnp.real(equation_relative_residual),
        ground_energy_mean=jnp.real(sums.ground_energy_sum / safe_sample_count),
        correction_norm=jnp.real(correction_norm),
        shifted_hamiltonian_norm=jnp.real(shifted_hamiltonian_norm),
        error_d=jnp.real(error_d),
        reweight_ess=jnp.real(reweight_ess),
        reweight_ess_fraction=jnp.real(reweight_ess_fraction),
        invalid_sample_fraction=jnp.real(invalid_sample_fraction),
        estimator_mode=jnp.asarray(0, dtype=jnp.int32),
        direct_hloc_rmse=nan_real,
        direct_hloc_std=nan_real,
        direct_hloc_sem=nan_real,
    )


def _source_sampled_error_d_sums(
    sums: NQSLITSourceSums,
    phi_norm: jnp.ndarray,
    safe_normalization: jnp.ndarray,
    normalized_overlap: jnp.ndarray,
    *,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    safe_weight_sum = jnp.maximum(
        sums.weight_sum,
        jnp.asarray(eps, dtype=sums.weight_sum.dtype),
    )
    correction_norm = (
        phi_norm
        * sums.response_over_source_abs2_sum
        / safe_weight_sum
        / jnp.maximum(jnp.abs(safe_normalization) ** 2, eps)
    )
    phi_norm_safe = jnp.maximum(phi_norm, jnp.asarray(eps, dtype=phi_norm.dtype))
    correction_projection = jnp.abs(normalized_overlap) ** 2 / phi_norm_safe
    correction_perp = jnp.sqrt(
        jnp.maximum(
            jnp.real(correction_norm - correction_projection),
            jnp.asarray(0.0, dtype=phi_norm.dtype),
        )
    )
    shift_norm = jnp.sqrt(
        jnp.asarray(omega, dtype=phi_norm.dtype) ** 2
        + jnp.asarray(eta, dtype=phi_norm.dtype) ** 2
    )
    shift_norm = jnp.maximum(shift_norm, jnp.asarray(eps, dtype=shift_norm.dtype))
    shifted_hamiltonian_norm = (
        phi_norm
        * sums.hbar_over_source_abs2_sum
        / safe_weight_sum
        / jnp.maximum(jnp.abs(safe_normalization) ** 2, eps)
        / shift_norm**2
    )
    shifted_overlap = (
        phi_norm
        * sums.hbar_over_source_sum
        / safe_weight_sum
        / safe_normalization
        / shift_norm
    )
    shifted_projection = jnp.abs(shifted_overlap) ** 2 / phi_norm_safe
    shifted_perp = jnp.sqrt(
        jnp.maximum(
            jnp.real(shifted_hamiltonian_norm - shifted_projection),
            jnp.asarray(0.0, dtype=phi_norm.dtype),
        )
    )
    return (
        correction_norm,
        shifted_hamiltonian_norm,
        jnp.minimum(correction_perp, shifted_perp),
    )


def nqs_lit_double_sampled_stats(
    response_apply: ResponseApply,
    response_params: Params,
    ground_logpsi: GroundLogPsi,
    ground_params: Params,
    source_batched_data: BatchedData[MoleculeData],
    psi_batched_data: BatchedData[MoleculeData],
    *,
    axis: int,
    source_center: float | jnp.ndarray,
    source_norm: float | jnp.ndarray,
    ground_energy: float | jnp.ndarray,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    source_floor: float | jnp.ndarray = 0.0,
    eps: float = 1e-12,
) -> NQSLITStats:
    """Compute NQS-LIT diagnostics with direct ``pi_Psi`` samples.

    The source pool supplies the complex normalization
    ``N=<Phi|Psi>/<Phi|Phi>`` and stable overlap estimator.  The direct
    ``pi_Psi`` pool supplies the double-Monte-Carlo fidelity estimator,
    avoiding source-pool reweighting when its effective sample size collapses.
    """
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
        eta=eta,
        source_floor=source_floor,
        eps=eps,
    )
    source_stats = nqs_lit_stats_from_source_sums(
        source_sums,
        source_norm=source_norm,
        omega=omega,
        eta=eta,
        eps=eps,
    )
    data = psi_batched_data.data
    action = jax.vmap(
        lambda one: local_action_ratio(
            response_apply,
            response_params,
            ground_logpsi,
            ground_params,
            one,
            ground_energy=ground_energy,
            omega=omega,
            eta=eta,
        )[0],
        in_axes=(psi_batched_data.vmap_axis,),
    )(data)
    dipole = jax.vmap(
        lambda one: molecular_electronic_dipole(one, axis),
        in_axes=(psi_batched_data.vmap_axis,),
    )(data)
    source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
    safe_source = jnp.where(
        jnp.abs(source) > eps,
        source,
        jnp.asarray(eps, dtype=source.dtype) * jnp.where(source < 0, -1.0, 1.0),
    )
    ratio = action / safe_source
    safe_ratio = jnp.where(
        jnp.abs(ratio) > eps,
        ratio,
        jnp.asarray(eps, dtype=ratio.real.dtype) + 0j,
    )
    raw_hloc = source_stats.normalization / safe_ratio
    finite_ratio = jnp.isfinite(jnp.real(ratio)) & jnp.isfinite(jnp.imag(ratio))
    finite = (
        (jnp.abs(ratio) > eps)
        & finite_ratio
        & jnp.isfinite(jnp.real(raw_hloc))
        & jnp.isfinite(jnp.imag(raw_hloc))
    )
    valid_count = jnp.sum(finite)
    safe_valid_count = jnp.maximum(valid_count, 1)
    hloc = jnp.where(finite, raw_hloc, jnp.asarray(0.0, dtype=raw_hloc.dtype))
    hloc_mean = jnp.sum(hloc) / safe_valid_count
    direct_hloc_rmse = jnp.sqrt(
        jnp.real(jnp.sum(jnp.where(finite, jnp.abs(raw_hloc - 1.0) ** 2, 0.0)))
        / safe_valid_count
    )
    direct_hloc_std = jnp.sqrt(
        jnp.real(jnp.sum(jnp.where(finite, jnp.abs(raw_hloc - hloc_mean) ** 2, 0.0)))
        / safe_valid_count
    )
    sample_count = jnp.asarray(hloc.shape[0], dtype=direct_hloc_std.dtype)
    direct_hloc_sem = direct_hloc_std / jnp.sqrt(safe_valid_count)
    fidelity = jnp.clip(jnp.real(hloc_mean), 0.0, 1.0)
    log_ratio_abs2 = 2.0 * jnp.log(
        jnp.maximum(
            jnp.where(finite_ratio, jnp.abs(ratio), 0.0),
            jnp.asarray(eps, dtype=ratio.real.dtype),
        )
    )
    direct_log_ratio_mean = (
        jnp.sum(jnp.where(finite, log_ratio_abs2, 0.0)) / safe_valid_count
    )
    reverse_kl = jnp.where(
        valid_count > 0,
        jnp.maximum(
            direct_log_ratio_mean - source_stats.log_ratio_norm,
            jnp.asarray(0.0, dtype=fidelity.dtype),
        ),
        jnp.asarray(0.0, dtype=fidelity.dtype),
    )
    equation_relative_residual = jnp.sqrt(
        jnp.maximum(1.0 / jnp.maximum(fidelity, eps) - 1.0, 0.0)
    )
    invalid_sample_fraction = 1.0 - valid_count / jnp.maximum(sample_count, 1.0)
    action_norm = (
        jnp.asarray(source_norm, dtype=fidelity.dtype)
        * jnp.abs(source_stats.normalization) ** 2
        / jnp.maximum(fidelity, jnp.asarray(eps, dtype=fidelity.dtype))
    )
    return source_stats._replace(
        loss=1.0 - fidelity,
        fidelity=fidelity,
        reverse_kl=jnp.real(reverse_kl),
        residual_norm=jnp.asarray(source_norm, dtype=fidelity.dtype)
        * equation_relative_residual**2,
        equation_relative_residual=equation_relative_residual,
        action_norm=jnp.real(action_norm),
        estimator_mode=jnp.asarray(1, dtype=jnp.int32),
        direct_hloc_rmse=jnp.real(direct_hloc_rmse),
        direct_hloc_std=jnp.real(direct_hloc_std),
        direct_hloc_sem=jnp.real(direct_hloc_sem),
        invalid_sample_fraction=jnp.real(invalid_sample_fraction),
    )


def _source_sampled_error_d(
    action: jnp.ndarray,
    response_ratio: jnp.ndarray,
    source_weighted_mean,
    safe_source: jnp.ndarray,
    phi_norm: jnp.ndarray,
    safe_normalization: jnp.ndarray,
    normalized_overlap: jnp.ndarray,
    *,
    omega: float | jnp.ndarray,
    eta: float | jnp.ndarray,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    normalized_response_over_source = response_ratio / safe_normalization / safe_source
    correction_norm = phi_norm * source_weighted_mean(
        jnp.abs(normalized_response_over_source) ** 2
    )
    phi_norm_safe = jnp.maximum(phi_norm, jnp.asarray(eps, dtype=phi_norm.dtype))
    correction_projection = jnp.abs(normalized_overlap) ** 2 / phi_norm_safe
    correction_perp = jnp.sqrt(
        jnp.maximum(
            jnp.real(correction_norm - correction_projection),
            jnp.asarray(0.0, dtype=phi_norm.dtype),
        )
    )

    shift = jnp.asarray(omega, dtype=response_ratio.real.dtype) + 1j * jnp.asarray(
        eta,
        dtype=response_ratio.real.dtype,
    )
    hbar_response_ratio = action + shift * response_ratio
    shift_norm = jnp.sqrt(
        jnp.asarray(omega, dtype=response_ratio.real.dtype) ** 2
        + jnp.asarray(eta, dtype=response_ratio.real.dtype) ** 2
    )
    shift_norm = jnp.maximum(shift_norm, jnp.asarray(eps, dtype=shift_norm.dtype))
    shifted_over_source = (
        hbar_response_ratio / safe_normalization / safe_source / shift_norm
    )
    shifted_hamiltonian_norm = phi_norm * source_weighted_mean(
        jnp.abs(shifted_over_source) ** 2
    )
    shifted_overlap = phi_norm * source_weighted_mean(shifted_over_source)
    shifted_projection = jnp.abs(shifted_overlap) ** 2 / phi_norm_safe
    shifted_perp = jnp.sqrt(
        jnp.maximum(
            jnp.real(shifted_hamiltonian_norm - shifted_projection),
            jnp.asarray(0.0, dtype=phi_norm.dtype),
        )
    )
    return (
        correction_norm,
        shifted_hamiltonian_norm,
        jnp.minimum(
            correction_perp,
            shifted_perp,
        ),
    )


def restore_params_from_checkpoint(
    checkpoint_path: str | Path | UPath,
    fallback_params: Params,
    *,
    state_field: str = "params",
) -> tuple[int, Params]:
    """Restore a parameter subtree from a JaQMC stage checkpoint."""
    path = UPath(checkpoint_path)
    if path.is_dir():
        ckpt_files = sorted(path.glob("*ckpt_*.npz"), reverse=True)
        if not ckpt_files:
            msg = f"No checkpoint files found in {path}"
            raise FileNotFoundError(msg)
        path = ckpt_files[0]
    if not path.is_file():
        msg = f"Checkpoint path does not exist: {path}"
        raise FileNotFoundError(msg)
    with path.open("rb") as f:
        try:
            with np.load(f) as npf:
                step = int(npf["step"].item()) if "step" in npf else -1
                params = _restore_prefixed_tree(npf, fallback_params, state_field)
        except (OSError, EOFError, BadZipFile) as exc:
            msg = f"Failed to restore checkpoint {path}"
            raise ValueError(msg) from exc
    return step, params


def _restore_prefixed_tree(
    npf: Mapping[str, np.ndarray | h5py.Group],
    fallback: Params,
    state_field: str,
) -> Params:
    ref_vals_with_path, treedef = jax.tree_util.tree_flatten_with_path(fallback)
    restored = []
    for key_path, ref_val in ref_vals_with_path:
        name = _pytree_key_path(key_path)
        candidates = (f"{state_field}/{name}", name)
        for candidate in candidates:
            if candidate in npf:
                restored.append(from_npz(candidate, npf, ref_val))
                break
        else:
            msg = (
                f"Checkpoint is missing parameter leaf {candidates[0]!r} "
                f"(or fallback key {name!r})"
            )
            raise KeyError(msg)
    return jax.tree.unflatten(treedef, restored)
