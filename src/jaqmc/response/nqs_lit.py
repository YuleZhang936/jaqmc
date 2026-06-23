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
    lit: jnp.ndarray
    broadened: jnp.ndarray
    source_norm: jnp.ndarray
    action_norm: jnp.ndarray
    correction_overlap: jnp.ndarray
    normalization: jnp.ndarray
    residual_norm: jnp.ndarray
    ground_energy_mean: jnp.ndarray
    correction_norm: jnp.ndarray
    shifted_hamiltonian_norm: jnp.ndarray
    error_d: jnp.ndarray
    reweight_ess: jnp.ndarray
    reweight_ess_fraction: jnp.ndarray
    estimator_mode: jnp.ndarray


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
    finite = (
        jnp.isfinite(jnp.real(action))
        & jnp.isfinite(jnp.imag(action))
        & jnp.isfinite(jnp.real(response_ratio))
        & jnp.isfinite(jnp.imag(response_ratio))
        & jnp.isfinite(source)
    )
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
    weight_norm = jnp.maximum(jnp.mean(source_weight), eps)

    def source_weighted_mean(value):
        return jnp.mean(source_weight * value) / weight_norm

    safe_source = jnp.where(
        jnp.abs(source) > eps,
        source,
        jnp.asarray(eps, dtype=source.dtype) * jnp.where(source < 0, -1.0, 1.0),
    )
    ratio = action / safe_source
    normalization = source_weighted_mean(ratio)
    ratio_norm = source_weighted_mean(jnp.abs(ratio) ** 2)
    psi_weight_unnormalized = source_weight * jnp.abs(ratio) ** 2
    psi_weight_sum = jnp.sum(psi_weight_unnormalized)
    reweight_ess = psi_weight_sum**2 / jnp.maximum(
        jnp.sum(psi_weight_unnormalized**2),
        eps,
    )
    valid_sample_count = jnp.maximum(
        jnp.sum(source_weight > jnp.asarray(0.0, dtype=source_weight.dtype)),
        jnp.asarray(1, dtype=reweight_ess.dtype),
    )
    reweight_ess_fraction = reweight_ess / valid_sample_count
    fidelity = (jnp.abs(normalization) ** 2) / jnp.maximum(ratio_norm, eps)
    fidelity = jnp.clip(jnp.real(fidelity), 0.0, 1.0)
    loss = 1.0 - fidelity

    phi_norm = jnp.asarray(source_norm, dtype=dipole.dtype)
    action_norm = phi_norm * ratio_norm
    correction_overlap = phi_norm * source_weighted_mean(
        jnp.conj(response_ratio) / safe_source
    )
    safe_normalization = normalization + jnp.asarray(eps, dtype=normalization.dtype)
    normalized_overlap = correction_overlap / jnp.conj(safe_normalization)
    lit = jnp.maximum(-jnp.imag(normalized_overlap) / jnp.asarray(eta), 0.0)
    broadened = jnp.asarray(eta) * lit / jnp.pi
    residual = ratio / safe_normalization
    residual_norm = phi_norm * source_weighted_mean(jnp.abs(residual - 1.0) ** 2)
    correction_norm, shifted_hamiltonian_norm, error_d = _source_sampled_error_d(
        action,
        response_ratio,
        source_weighted_mean,
        safe_source,
        phi_norm,
        safe_normalization,
        normalized_overlap,
        omega=omega,
        eta=eta,
        eps=eps,
    )
    return NQSLITStats(
        loss=jnp.real(loss),
        fidelity=fidelity,
        lit=jnp.real(lit),
        broadened=jnp.real(broadened),
        source_norm=jnp.real(phi_norm),
        action_norm=jnp.real(action_norm),
        correction_overlap=correction_overlap,
        normalization=normalization,
        residual_norm=jnp.real(residual_norm),
        ground_energy_mean=jnp.real(jnp.mean(eloc_response)),
        correction_norm=jnp.real(correction_norm),
        shifted_hamiltonian_norm=jnp.real(shifted_hamiltonian_norm),
        error_d=jnp.real(error_d),
        reweight_ess=jnp.real(reweight_ess),
        reweight_ess_fraction=jnp.real(reweight_ess_fraction),
        estimator_mode=jnp.asarray(0, dtype=jnp.int32),
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
    source_stats = nqs_lit_source_sampled_stats(
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
        eta=eta,
        source_floor=source_floor,
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
    hloc = source_stats.normalization / safe_ratio
    finite = jnp.isfinite(jnp.real(hloc)) & jnp.isfinite(jnp.imag(hloc))
    hloc = jnp.where(finite, hloc, jnp.asarray(0.0, dtype=hloc.dtype))
    fidelity = jnp.clip(jnp.real(jnp.mean(hloc)), 0.0, 1.0)
    action_norm = (
        jnp.asarray(source_norm, dtype=fidelity.dtype)
        * jnp.abs(source_stats.normalization) ** 2
        / jnp.maximum(fidelity, jnp.asarray(eps, dtype=fidelity.dtype))
    )
    return source_stats._replace(
        loss=1.0 - fidelity,
        fidelity=fidelity,
        action_norm=jnp.real(action_norm),
        estimator_mode=jnp.asarray(1, dtype=jnp.int32),
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
