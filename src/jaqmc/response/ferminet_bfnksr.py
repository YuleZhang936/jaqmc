# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001, RUF100

"""BF-NKSR response using a restored JaQMC FermiNet ground state."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import optax
import yaml
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from scipy import special
from scipy.stats import qmc

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow
from jaqmc.response.qc_warmstart import (
    build_casscf_krylov_teacher_model,
    build_casscf_seed_model,
    casscf_krylov_teacher_values,
    evaluate_casscf_krylov_teacher_targets,
    evaluate_casscf_ratio_carriers,
)
from jaqmc.response.monte_carlo import systematic_resample
from jaqmc.response.spectrum import (
    Peak,
    ProjectedSpectrum,
    find_spectrum_peaks,
    lorentzian_spectrum,
    projected_spectrum,
)
from jaqmc.utils.checkpoint import tree_from_npz, tree_to_npz
from jaqmc.utils.config import ConfigManager
from jaqmc.wavefunction.backbone.ferminet import FermiLayers
from jaqmc.wavefunction.input.atomic import MoleculeFeatures
from jaqmc.wavefunction.output.envelope import Envelope, EnvelopeType
from jaqmc.wavefunction.output.orbital import OrbitalProjection

jax.config.update("jax_enable_x64", True)

Params = dict[str, Any]
HARTREE_TO_EV = 27.211386245988
_RESPONSE_HEAD_TRANSFORM_KEY = "__response_head_transform"
_RESPONSE_BLOCKS_KEY = "__response_blocks"
_RESPONSE_SOURCE_LIFT_KEY = "__response_source_lift"
OFFICIAL_RESPONSE_FLOW = "official_cas_dressed_teacher_projected_resolvent"
OFFICIAL_ENRICHMENT_TRAINING_OBJECTIVE = "gauge-action"
OFFICIAL_ENRICHMENT_SELECTION_OBJECTIVE = "action-oracle"
OFFICIAL_ENRICHMENT_SAMPLING = "mixture"
OFFICIAL_FINAL_SAMPLING = "cas-dressed-teacher-production-qmc-resampling"
OFFICIAL_SOURCE_ENVELOPE_CORE_DECAY = 2.0
OFFICIAL_SOURCE_ENVELOPE_DIFFUSE_DECAY = 0.5
OFFICIAL_STRONG_ORACLE_SAMPLES = 256
OFFICIAL_DIFFUSE_PTAIL_KAPPAS = (0.25, 0.35, 0.5, 0.75, 1.0)
OFFICIAL_PARTIAL_WAVE_CLOSURE_POWERS = OFFICIAL_DIFFUSE_PTAIL_KAPPAS
OFFICIAL_PARTIAL_WAVE_CLOSURE_SCALE = 1.0
OFFICIAL_CORRELATED_DIPOLE_EE_SCALES = (0.35, 0.7, 1.4, 2.8)
OFFICIAL_SOURCE_LIFT_RADIAL_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
OFFICIAL_SOURCE_LIFT_PAIR_SCALES = OFFICIAL_CORRELATED_DIPOLE_EE_SCALES
OFFICIAL_SOURCE_LIFT_SEED_SAMPLES = 512
OFFICIAL_SOURCE_LIFT_SEED_RANK_RTOL = 1e-10
OFFICIAL_SOURCE_LIFT_SEED_RANK_ATOL = 1e-14
OFFICIAL_SOURCE_LIFT_SEED_CAPTURE_FLOOR = 1e-12
OFFICIAL_RESPONSE_RADIAL_POWERS = 2
OFFICIAL_RESPONSE_RADIAL_SCALE = 1.0
OFFICIAL_DRESSING_RADIAL_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
OFFICIAL_DRESSING_PAIR_SCALES = OFFICIAL_CORRELATED_DIPOLE_EE_SCALES
OFFICIAL_PRODUCTION_SAMPLER = "bright-influence"
OFFICIAL_PRODUCTION_SAMPLERS = ("bright-influence",)
MATRIX_LEVERAGE_DENOMINATOR_FLOOR = 1e-150


def _envelope_component_arrays(
    decays: jax.Array | np.ndarray | None,
    weights: jax.Array | np.ndarray | None,
    *,
    component_name: str,
    required: bool,
) -> tuple[np.ndarray, np.ndarray]:
    decays_np = (
        np.asarray([], dtype=np.float64)
        if decays is None
        else np.asarray(decays, dtype=np.float64).reshape(-1)
    )
    if decays_np.size == 0:
        if required:
            msg = f"one-electron {component_name} envelope mixture requires decays"
            raise ValueError(msg)
        return decays_np, np.asarray([], dtype=np.float64)
    if np.any(decays_np <= 0) or not np.all(np.isfinite(decays_np)):
        msg = f"one-electron {component_name} envelope decays must be positive"
        raise ValueError(msg)
    if weights is None:
        return decays_np, np.ones_like(decays_np)
    weights_np = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights_np.shape != decays_np.shape:
        msg = f"one-electron {component_name} envelope weights must match decays"
        raise ValueError(msg)
    if np.any(weights_np < 0) or not np.all(np.isfinite(weights_np)):
        msg = (
            f"one-electron {component_name} envelope weights must be finite "
            "and nonnegative"
        )
        raise ValueError(msg)
    return decays_np, weights_np


def make_batches(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield slice(start, min(start + batch_size, n_items))


def sample_envelope(
    key: jax.Array,
    n_samples: int,
    decay: float,
    electron_shape: tuple[int, int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    kr, kmu, kphi = jax.random.split(key, 3)
    sample_shape = (n_samples, *electron_shape[:-1])
    radius = jax.random.gamma(kr, 3.0, sample_shape) / (2 * decay)
    mu = jax.random.uniform(kmu, sample_shape, minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(kphi, sample_shape, minval=0.0, maxval=2 * jnp.pi)
    rho = jnp.sqrt(jnp.maximum(0.0, 1.0 - mu**2))
    points = jnp.stack(
        [radius * rho * jnp.cos(phi), radius * rho * jnp.sin(phi), radius * mu],
        axis=-1,
    )
    log_density = jnp.sum(
        3 * jnp.log(decay) - jnp.log(jnp.pi) - 2 * decay * radius,
        axis=tuple(range(1, radius.ndim)),
    )
    density = jnp.exp(log_density)
    return points, density, radius


def sample_envelope_sobol(
    *,
    n_samples: int,
    decay: float,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the envelope distribution with a scrambled Sobol sequence.

    Returns:
        Cartesian points, normalized envelope density, and electron radii.
    """
    electron_count = int(np.prod(electron_shape[:-1]))
    engine = qmc.Sobol(d=3 * electron_count, scramble=scramble, seed=seed)
    if not scramble:
        uniforms = engine.random(n_samples + 1)[1:]
    elif n_samples > 0 and n_samples & (n_samples - 1) == 0:
        uniforms = engine.random_base2(int(np.log2(n_samples)))
    else:
        uniforms = engine.random(n_samples)
    uniforms = np.clip(uniforms, np.finfo(np.float64).eps, 1 - np.finfo(float).eps)
    uniforms = uniforms.reshape(n_samples, *electron_shape[:-1], 3)
    radius = special.gammaincinv(3.0, uniforms[..., 0]) / (2 * decay)
    mu = 2 * uniforms[..., 1] - 1
    phi = 2 * np.pi * uniforms[..., 2]
    rho = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    points = np.stack(
        [radius * rho * np.cos(phi), radius * rho * np.sin(phi), radius * mu],
        axis=-1,
    )
    log_density = np.sum(
        3 * np.log(decay) - np.log(np.pi) - 2 * decay * radius,
        axis=tuple(range(1, radius.ndim)),
    )
    density = np.exp(log_density)
    return points, density, radius


def sample_source_envelope_sobol(
    *,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a two-electron core/diffuse source-envelope proposal.

    Returns:
        Cartesian points, normalized mixture proposal density, and electron
        radii. Currently supports two-electron response heads initialized with
        one core and one diffuse envelope.

    Raises:
        ValueError: If the electron shape is not a two-electron coordinate set.
    """
    if electron_shape != (2, 3):
        msg = (
            "source-envelope-sobol final sampling currently requires "
            f"electron_shape=(2, 3), got {electron_shape}"
        )
        raise ValueError(msg)
    engine = qmc.Sobol(d=6, scramble=scramble, seed=seed)
    if not scramble:
        uniforms = engine.random(n_samples + 1)[1:]
    elif n_samples > 0 and n_samples & (n_samples - 1) == 0:
        uniforms = engine.random_base2(int(np.log2(n_samples)))
    else:
        uniforms = engine.random(n_samples)
    uniforms = np.clip(uniforms, np.finfo(np.float64).eps, 1 - np.finfo(float).eps)
    uniforms = uniforms.reshape(n_samples, 2, 3)
    assignments = np.zeros((n_samples, 2), dtype=np.float64)
    assignments[:, 0] = core_decay
    assignments[:, 1] = diffuse_decay
    assignments[1::2, 0] = diffuse_decay
    assignments[1::2, 1] = core_decay
    radius = special.gammaincinv(3.0, uniforms[..., 0]) / (2 * assignments)
    mu = 2 * uniforms[..., 1] - 1
    phi = 2 * np.pi * uniforms[..., 2]
    rho = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    points = np.stack(
        [radius * rho * np.cos(phi), radius * rho * np.sin(phi), radius * mu],
        axis=-1,
    )
    radii = np.linalg.norm(points, axis=2)
    core_log_density = 3 * np.log(core_decay) - np.log(np.pi) - 2 * core_decay * radii
    diffuse_log_density = (
        3 * np.log(diffuse_decay) - np.log(np.pi) - 2 * diffuse_decay * radii
    )
    component_cd = core_log_density[:, 0] + diffuse_log_density[:, 1]
    component_dc = diffuse_log_density[:, 0] + core_log_density[:, 1]
    density = 0.5 * (np.exp(component_cd) + np.exp(component_dc))
    return points, density, radii


def sample_source_envelope_pz_sobol(
    *,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the official source-envelope ``p_z`` proposal.

    For one-electron systems this is the normalized ``p_z^2`` envelope with
    the diffuse decay.  For two-electron systems, one electron is sampled from
    ``s^2 ~ exp(-2 a r)`` while the other is sampled from
    ``p_z^2 ~ z^2 exp(-2 a r)``.  The two-electron branch matches the dominant
    1s2p dipole channel more closely than a spherical diffuse envelope and
    reduces final-matrix variance for He-like audits.

    Returns:
        Cartesian points, normalized mixture proposal density, and electron
        radii.

    Raises:
        ValueError: If the electron shape is not a two-electron coordinate set.
    """
    if electron_shape == (1, 3):
        return sample_one_electron_pz_envelope_mixture_sobol(
            n_samples=n_samples,
            decays=np.asarray([diffuse_decay], dtype=np.float64),
            weights=np.asarray([1.0], dtype=np.float64),
            electron_shape=electron_shape,
            seed=seed,
            scramble=scramble,
        )
    if electron_shape != (2, 3):
        msg = (
            "source-envelope-pz-sobol final sampling requires "
            f"electron_shape=(1, 3) or (2, 3), got {electron_shape}"
        )
        raise ValueError(msg)
    engine = qmc.Sobol(d=6, scramble=scramble, seed=seed)
    if not scramble:
        uniforms = engine.random(n_samples + 1)[1:]
    elif n_samples > 0 and n_samples & (n_samples - 1) == 0:
        uniforms = engine.random_base2(int(np.log2(n_samples)))
    else:
        uniforms = engine.random(n_samples)
    uniforms = np.clip(uniforms, np.finfo(np.float64).eps, 1 - np.finfo(float).eps)
    points, radii = _source_envelope_pz_points_from_uniforms(
        uniforms.reshape(n_samples, 2, 3),
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
    )
    density = source_envelope_pz_density(
        points,
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
    )
    return points, density, radii


def one_electron_pz_envelope_mixture_density(
    points: np.ndarray,
    *,
    decays: jax.Array | np.ndarray,
    weights: jax.Array | np.ndarray | None = None,
    spherical_decays: jax.Array | np.ndarray | None = None,
    spherical_weights: jax.Array | np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a one-electron mixture of normalized ``s`` and ``p_z`` envelopes.

    Each p component is
    ``q_alpha(r) = alpha^5 / pi * z^2 * exp(-2 alpha r)``.  Optional spherical
    floor components use ``q_alpha(r) = alpha^3 / pi * exp(-2 alpha r)`` and
    keep weak-form gradient terms covered at the ``p_z`` node.

    Returns:
        One proposal density value per sample.

    Raises:
        ValueError: If the coordinate shape or decay list is invalid.
    """
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.ndim != 3 or points_np.shape[1:] != (1, 3):
        msg = (
            "one-electron pz envelope density requires points with shape "
            f"(n_samples, 1, 3), got {points_np.shape}"
        )
        raise ValueError(msg)
    decays_np, weights_np = _envelope_component_arrays(
        decays,
        weights,
        component_name="pz",
        required=True,
    )
    spherical_decays_np, spherical_weights_np = _envelope_component_arrays(
        spherical_decays,
        spherical_weights,
        component_name="spherical",
        required=False,
    )
    weight_sum = float(np.sum(weights_np) + np.sum(spherical_weights_np))
    if weight_sum <= 0:
        msg = "one-electron envelope mixture requires positive total weight"
        raise ValueError(msg)
    weights_np = weights_np / weight_sum
    spherical_weights_np = spherical_weights_np / weight_sum
    radius = np.linalg.norm(points_np[:, 0, :], axis=1)
    z2 = np.maximum(points_np[:, 0, 2] ** 2, np.finfo(np.float64).tiny)
    pz_log_components = (
        5 * np.log(decays_np[None, :])
        - np.log(np.pi)
        + np.log(z2[:, None])
        - 2 * radius[:, None] * decays_np[None, :]
    )
    density = np.sum(weights_np[None, :] * np.exp(pz_log_components), axis=1)
    if spherical_decays_np.size:
        s_log_components = (
            3 * np.log(spherical_decays_np[None, :])
            - np.log(np.pi)
            - 2 * radius[:, None] * spherical_decays_np[None, :]
        )
        density += np.sum(
            spherical_weights_np[None, :] * np.exp(s_log_components),
            axis=1,
        )
    return density


def sample_one_electron_pz_envelope_mixture_sobol(
    *,
    n_samples: int,
    decays: jax.Array | np.ndarray,
    weights: jax.Array | np.ndarray | None = None,
    spherical_decays: jax.Array | np.ndarray | None = None,
    spherical_weights: jax.Array | np.ndarray | None = None,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a one-electron response-envelope ``p_z`` mixture.

    The radial law for each component is ``r^4 exp(-2 alpha r)`` and the
    angular law has density proportional to ``mu^2``.  This proposal matches
    the direct one-electron ``z exp(-alpha r)`` response-envelope channel used
    by the formal neural response heads more closely than a spherical envelope.

    Returns:
        Cartesian points, normalized mixture proposal density, and radii.

    Raises:
        ValueError: If the electron shape or decay list is invalid.
    """
    if electron_shape != (1, 3):
        msg = (
            "one-electron-pz-envelope-mixture-sobol final sampling requires "
            f"electron_shape=(1, 3), got {electron_shape}"
        )
        raise ValueError(msg)
    decays_np, weights_np = _envelope_component_arrays(
        decays,
        weights,
        component_name="pz",
        required=True,
    )
    spherical_decays_np, spherical_weights_np = _envelope_component_arrays(
        spherical_decays,
        spherical_weights,
        component_name="spherical",
        required=False,
    )
    component_decays = np.concatenate([decays_np, spherical_decays_np])
    component_weights = np.concatenate([weights_np, spherical_weights_np])
    component_is_pz = np.concatenate(
        [
            np.ones(decays_np.shape, dtype=bool),
            np.zeros(spherical_decays_np.shape, dtype=bool),
        ]
    )
    weight_sum = float(np.sum(component_weights))
    if weight_sum <= 0:
        msg = "one-electron envelope mixture requires positive total weight"
        raise ValueError(msg)
    component_prob = component_weights / weight_sum
    component_cdf = np.cumsum(component_prob)
    engine = qmc.Sobol(d=4, scramble=scramble, seed=seed)
    if not scramble:
        uniforms = engine.random(n_samples + 1)[1:]
    elif n_samples > 0 and n_samples & (n_samples - 1) == 0:
        uniforms = engine.random_base2(int(np.log2(n_samples)))
    else:
        uniforms = engine.random(n_samples)
    uniforms = np.clip(uniforms, np.finfo(np.float64).eps, 1 - np.finfo(float).eps)
    component_index = np.searchsorted(
        component_cdf,
        uniforms[:, 0],
        side="right",
    )
    component_index = np.minimum(component_index, component_decays.size - 1)
    assigned_decays = component_decays[component_index]
    assigned_is_pz = component_is_pz[component_index]
    gamma_shape = np.where(assigned_is_pz, 5.0, 3.0)
    radius = special.gammaincinv(gamma_shape, uniforms[:, 1]) / (2 * assigned_decays)
    mu_pz = np.cbrt(2 * uniforms[:, 2] - 1)
    mu_s = 2 * uniforms[:, 2] - 1
    mu = np.where(assigned_is_pz, mu_pz, mu_s)
    phi = 2 * np.pi * uniforms[:, 3]
    rho = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    points = np.stack(
        [radius * rho * np.cos(phi), radius * rho * np.sin(phi), radius * mu],
        axis=-1,
    )[:, None, :]
    radii = np.linalg.norm(points, axis=2)
    density = one_electron_pz_envelope_mixture_density(
        points,
        decays=decays_np,
        weights=weights_np,
        spherical_decays=spherical_decays_np,
        spherical_weights=spherical_weights_np,
    )
    return points, density, radii


def sample_one_electron_pz_envelope_mixture_sobol_antithetic(
    *,
    n_samples: int,
    decays: jax.Array | np.ndarray,
    weights: jax.Array | np.ndarray | None = None,
    spherical_decays: jax.Array | np.ndarray | None = None,
    spherical_weights: jax.Array | np.ndarray | None = None,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the one-electron ``p_z`` mixture with inversion pairs.

    Returns:
        Cartesian points, normalized mixture proposal density, and radii.
    """
    base_samples = (int(n_samples) + 1) // 2
    points, density, radii = sample_one_electron_pz_envelope_mixture_sobol(
        n_samples=base_samples,
        decays=decays,
        weights=weights,
        spherical_decays=spherical_decays,
        spherical_weights=spherical_weights,
        electron_shape=electron_shape,
        seed=seed,
        scramble=scramble,
    )
    paired_points = np.concatenate([points, -points], axis=0)[:n_samples]
    paired_density = np.concatenate([density, density], axis=0)[:n_samples]
    paired_radii = np.concatenate([radii, radii], axis=0)[:n_samples]
    return paired_points, paired_density, paired_radii


def sample_source_envelope_pz_sobol_antithetic(
    *,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    electron_shape: tuple[int, int],
    seed: int,
    scramble: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the core/``p_z`` proposal with global inversion pairs.

    The proposal density is invariant under ``R -> -R``.  Pairing every Sobol
    point with its inverted partner cancels odd-parity Monte Carlo noise in
    source projections for atom-centered dipole spectra without changing the
    importance density.

    Returns:
        Cartesian points, normalized mixture proposal density, and electron
        radii.
    """
    base_samples = (int(n_samples) + 1) // 2
    points, density, radii = sample_source_envelope_pz_sobol(
        n_samples=base_samples,
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
        electron_shape=electron_shape,
        seed=seed,
        scramble=scramble,
    )
    paired_points = np.concatenate([points, -points], axis=0)[:n_samples]
    paired_density = np.concatenate([density, density], axis=0)[:n_samples]
    paired_radii = np.concatenate([radii, radii], axis=0)[:n_samples]
    return paired_points, paired_density, paired_radii


def _source_envelope_pz_points_from_uniforms(
    uniforms: np.ndarray,
    *,
    core_decay: float,
    diffuse_decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform ``(n, 2, 3)`` uniforms into stratified core/pz points.

    Returns:
        Cartesian points and electron radii.
    """
    n_samples = uniforms.shape[0]
    assignments = np.zeros((n_samples, 2), dtype=np.float64)
    assignments[:, 0] = core_decay
    assignments[:, 1] = diffuse_decay
    assignments[1::2, 0] = diffuse_decay
    assignments[1::2, 1] = core_decay
    pz_mask = np.zeros((n_samples, 2), dtype=bool)
    pz_mask[:, 1] = True
    pz_mask[1::2, 0] = True
    pz_mask[1::2, 1] = False

    gamma_shape = np.where(pz_mask, 5.0, 3.0)
    radius = special.gammaincinv(gamma_shape, uniforms[..., 0]) / (2 * assignments)
    mu_s = 2 * uniforms[..., 1] - 1
    mu_pz = np.cbrt(2 * uniforms[..., 1] - 1)
    mu = np.where(pz_mask, mu_pz, mu_s)
    phi = 2 * np.pi * uniforms[..., 2]
    rho = np.sqrt(np.maximum(0.0, 1.0 - mu**2))
    points = np.stack(
        [radius * rho * np.cos(phi), radius * rho * np.sin(phi), radius * mu],
        axis=-1,
    )
    radii = np.linalg.norm(points, axis=2)
    return points, radii


def source_envelope_pz_density(
    points: np.ndarray,
    *,
    core_decay: float,
    diffuse_decay: float,
) -> np.ndarray:
    """Evaluate the normalized two-electron core/pz mixture density.

    Returns:
        One base proposal density value per sample.
    """
    radii = np.linalg.norm(points, axis=2)
    z2 = np.maximum(points[..., 2] ** 2, np.finfo(np.float64).tiny)
    core_log_density = 3 * np.log(core_decay) - np.log(np.pi) - 2 * core_decay * radii
    diffuse_log_density = (
        5 * np.log(diffuse_decay)
        - np.log(np.pi)
        + np.log(z2)
        - 2 * diffuse_decay * radii
    )
    component_cd = core_log_density[:, 0] + diffuse_log_density[:, 1]
    component_dc = diffuse_log_density[:, 0] + core_log_density[:, 1]
    return 0.5 * (np.exp(component_cd) + np.exp(component_dc))


def spherical_envelope_product_density(
    points: np.ndarray,
    *,
    decay: float,
) -> np.ndarray:
    """Evaluate a product of normalized spherical Slater envelopes.

    Returns:
        One normalized proposal-density value per sample.
    """
    points_np = np.asarray(points, dtype=np.float64)
    radii = np.linalg.norm(points_np, axis=2)
    log_density = np.sum(
        3 * np.log(float(decay)) - np.log(np.pi) - 2 * float(decay) * radii,
        axis=1,
    )
    return np.exp(log_density)


def warm_start_auxiliary_proposal_density(
    points: np.ndarray,
    *,
    core_decay: float,
    diffuse_decay: float,
) -> np.ndarray:
    """Evaluate the normalized auxiliary proposal used by teacher pretraining.

    Returns:
        One normalized auxiliary-density value per sample.

    Raises:
        ValueError: If the point array does not have shape ``(n,e,3)``.
    """
    points_np = np.asarray(points, dtype=np.float64)
    if points_np.ndim != 3:
        msg = "warm-start auxiliary proposal points must have shape (n,e,3)"
        raise ValueError(msg)
    if points_np.shape[1:] == (1, 3):
        return one_electron_pz_envelope_mixture_density(
            points_np,
            decays=np.asarray([diffuse_decay], dtype=np.float64),
            weights=np.asarray([1.0], dtype=np.float64),
            spherical_decays=np.asarray([core_decay], dtype=np.float64),
            spherical_weights=np.asarray([0.25], dtype=np.float64),
        )
    if points_np.shape[1:] == (2, 3):
        return source_envelope_pz_density(
            points_np,
            core_decay=core_decay,
            diffuse_decay=diffuse_decay,
        )
    return spherical_envelope_product_density(points_np, decay=diffuse_decay)


def sample_warm_start_auxiliary_proposal(
    *,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    electron_shape: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the auxiliary pretraining proposal and its normalized density.

    Returns:
        Points and one normalized density value per point.
    """
    if electron_shape == (1, 3):
        points, density, _ = sample_one_electron_pz_envelope_mixture_sobol_antithetic(
            n_samples=n_samples,
            decays=np.asarray([diffuse_decay], dtype=np.float64),
            weights=np.asarray([1.0], dtype=np.float64),
            spherical_decays=np.asarray([core_decay], dtype=np.float64),
            spherical_weights=np.asarray([0.25], dtype=np.float64),
            electron_shape=electron_shape,
            seed=seed,
        )
        return points, density
    if electron_shape == (2, 3):
        points, density, _ = sample_source_envelope_pz_sobol_antithetic(
            n_samples=n_samples,
            core_decay=core_decay,
            diffuse_decay=diffuse_decay,
            electron_shape=electron_shape,
            seed=seed,
        )
        return points, density
    points, density, _ = sample_envelope_sobol(
        n_samples=n_samples,
        decay=diffuse_decay,
        electron_shape=electron_shape,
        seed=seed,
    )
    return points, density


def _ground_density_np(
    ground: FermiNetGround,
    points: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    pieces = []
    for chunk in make_batches(points.shape[0], max(1, int(batch_size))):
        logpsi = np.asarray(ground_logpsi_batch(ground, jnp.asarray(points[chunk])))
        pieces.append(np.exp(np.clip(2.0 * logpsi, -745.0, 700.0)))
    return np.concatenate(pieces, axis=0)


def _teacher_pretrain_normalizers(
    ground: FermiNetGround,
    teacher_model: Any,
    proposal_points: np.ndarray,
    proposal_density: np.ndarray,
    *,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    proposal_density = np.asarray(proposal_density, dtype=np.float64)
    proposal_density = np.maximum(proposal_density, np.finfo(np.float64).tiny)
    ground_raw = _ground_density_np(ground, proposal_points, batch_size=batch_size)
    teacher_values = casscf_krylov_teacher_values(teacher_model, proposal_points)
    ground_norm = float(np.mean(ground_raw / proposal_density))
    teacher_norms = np.mean(teacher_values**2 / proposal_density[:, None], axis=0)
    tiny = np.finfo(np.float64).tiny
    ground_norm = max(ground_norm, tiny)
    teacher_norms = np.maximum(teacher_norms, tiny)
    return ground_norm, np.asarray(teacher_norms, dtype=np.float64)


def krylov_teacher_pretrain_density(
    ground: FermiNetGround,
    teacher_model: Any,
    points: np.ndarray,
    *,
    core_decay: float,
    diffuse_decay: float,
    ground_weight: float,
    teacher_weight: float,
    aux_weight: float,
    ground_norm: float,
    teacher_norms: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Evaluate the normalized-component pretraining mixture density.

    Returns:
        One positive, unnormalized mixture-density value per sample.
    """
    points_np = np.asarray(points, dtype=np.float64)
    density = np.zeros((points_np.shape[0],), dtype=np.float64)
    if ground_weight > 0.0:
        ground_raw = _ground_density_np(ground, points_np, batch_size=batch_size)
        density += float(ground_weight) * ground_raw / float(ground_norm)
    if teacher_weight > 0.0:
        teacher_values = casscf_krylov_teacher_values(teacher_model, points_np)
        norms = np.asarray(teacher_norms, dtype=np.float64).reshape(1, -1)
        teacher_density = np.mean(teacher_values**2 / norms, axis=1)
        density += float(teacher_weight) * teacher_density
    if aux_weight > 0.0:
        density += float(aux_weight) * warm_start_auxiliary_proposal_density(
            points_np,
            core_decay=core_decay,
            diffuse_decay=diffuse_decay,
        )
    return np.maximum(density, np.finfo(np.float64).tiny)


def sample_krylov_teacher_pretrain_distribution(  # noqa: C901
    ground: FermiNetGround,
    teacher_model: Any,
    *,
    n_samples: int,
    walkers: int,
    burn_in: int,
    steps_between: int,
    width: float,
    core_decay: float,
    diffuse_decay: float,
    ground_weight: float,
    teacher_weight: float,
    aux_weight: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample ``q_pre = q0 + q_CAS-teacher + q_aux`` for Sobolev pretraining.

    Returns:
        Sampled points, positive density values, and sampling diagnostics.

    Raises:
        ValueError: If sampling parameters or component weights are invalid.
    """
    if int(n_samples) < 1:
        msg = "Krylov teacher pretraining requires positive n_samples"
        raise ValueError(msg)
    if int(walkers) < 1:
        msg = "Krylov teacher pretraining requires at least one walker"
        raise ValueError(msg)
    if int(burn_in) < 0 or int(steps_between) < 1:
        msg = "Krylov teacher pretraining MCMC requires burn_in>=0 and steps_between>=1"
        raise ValueError(msg)
    if not (np.isfinite(width) and float(width) > 0.0):
        msg = "Krylov teacher pretraining MCMC width must be positive"
        raise ValueError(msg)
    weights = np.asarray(
        [ground_weight, teacher_weight, aux_weight],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        msg = "Krylov teacher pretraining mixture weights must be finite nonnegative"
        raise ValueError(msg)
    if not np.any(weights > 0.0):
        msg = "Krylov teacher pretraining mixture needs a positive component weight"
        raise ValueError(msg)
    walkers = min(int(walkers), int(n_samples))
    norm_samples = max(walkers, min(max(1024, walkers), int(n_samples)))
    proposal_points, proposal_density = sample_warm_start_auxiliary_proposal(
        n_samples=norm_samples,
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
        electron_shape=ground.electron_shape,
        seed=seed + 17,
    )
    ground_norm, teacher_norms = _teacher_pretrain_normalizers(
        ground,
        teacher_model,
        proposal_points,
        proposal_density,
        batch_size=batch_size,
    )
    points = np.asarray(proposal_points[:walkers], dtype=np.float64)
    logq = np.log(
        krylov_teacher_pretrain_density(
            ground,
            teacher_model,
            points,
            core_decay=core_decay,
            diffuse_decay=diffuse_decay,
            ground_weight=ground_weight,
            teacher_weight=teacher_weight,
            aux_weight=aux_weight,
            ground_norm=ground_norm,
            teacher_norms=teacher_norms,
            batch_size=batch_size,
        )
    )
    rng = np.random.default_rng(int(seed))
    accept_rates: list[float] = []

    def step(local_points: np.ndarray, local_logq: np.ndarray):
        proposal = local_points + float(width) * rng.normal(size=local_points.shape)
        proposal_logq = np.log(
            krylov_teacher_pretrain_density(
                ground,
                teacher_model,
                proposal,
                core_decay=core_decay,
                diffuse_decay=diffuse_decay,
                ground_weight=ground_weight,
                teacher_weight=teacher_weight,
                aux_weight=aux_weight,
                ground_norm=ground_norm,
                teacher_norms=teacher_norms,
                batch_size=batch_size,
            )
        )
        accept = np.log(rng.random(local_points.shape[0])) < proposal_logq - local_logq
        local_points = np.where(
            accept.reshape((-1,) + (1,) * (local_points.ndim - 1)),
            proposal,
            local_points,
        )
        local_logq = np.where(accept, proposal_logq, local_logq)
        accept_rates.append(float(np.mean(accept)))
        return local_points, local_logq

    for _ in range(int(burn_in)):
        points, logq = step(points, logq)
    collected = []
    while sum(piece.shape[0] for piece in collected) < int(n_samples):
        for _ in range(int(steps_between)):
            points, logq = step(points, logq)
        collected.append(points.copy())
    samples = np.concatenate(collected, axis=0)[: int(n_samples)]
    final_logq = np.log(
        krylov_teacher_pretrain_density(
            ground,
            teacher_model,
            samples,
            core_decay=core_decay,
            diffuse_decay=diffuse_decay,
            ground_weight=ground_weight,
            teacher_weight=teacher_weight,
            aux_weight=aux_weight,
            ground_norm=ground_norm,
            teacher_norms=teacher_norms,
            batch_size=batch_size,
        )
    )
    density = np.exp(final_logq - float(np.max(final_logq)))
    density = np.maximum(density, np.finfo(np.float64).tiny)
    stats = {
        "pmove": float(np.mean(accept_rates)) if accept_rates else float("nan"),
        "ground_norm": float(ground_norm),
        "teacher_norms": np.asarray(teacher_norms, dtype=np.float64),
        "ground_weight": float(ground_weight),
        "teacher_weight": float(teacher_weight),
        "aux_weight": float(aux_weight),
        "walkers": int(walkers),
        "burn_in": int(burn_in),
        "steps_between": int(steps_between),
        "width": float(width),
    }
    return samples, density, stats


def _dressed_teacher_values_np(
    params: Params,
    ground: FermiNetGround,
    teacher_model: Any,
    points: np.ndarray,
    *,
    head_count: int,
    batch_size: int,
) -> np.ndarray:
    pieces = []
    radial_scales = jnp.asarray(OFFICIAL_DRESSING_RADIAL_SCALES, dtype=jnp.float64)
    pair_scales = jnp.asarray(OFFICIAL_DRESSING_PAIR_SCALES, dtype=jnp.float64)
    for chunk in make_batches(points.shape[0], max(1, int(batch_size))):
        chunk_points_np = np.asarray(points[chunk], dtype=np.float64)
        teacher_values = casscf_krylov_teacher_values(teacher_model, chunk_points_np)[
            :, : int(head_count)
        ]
        dressed_values = cas_dressed_teacher_values_from_arrays(
            params,
            ground,
            jnp.asarray(chunk_points_np),
            jnp.asarray(teacher_values),
            radial_scales=radial_scales,
            pair_scales=pair_scales,
        )
        pieces.append(np.asarray(dressed_values, dtype=np.float64))
    return np.concatenate(pieces, axis=0)


def _production_component_normalizers(
    params: Params,
    ground: FermiNetGround,
    teacher_model: Any,
    proposal_points: np.ndarray,
    proposal_density: np.ndarray,
    *,
    head_count: int,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    proposal_density = np.maximum(
        np.asarray(proposal_density, dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    ground_raw = _ground_density_np(ground, proposal_points, batch_size=batch_size)
    source_raw = (
        np.asarray(
            source_values(ground, jnp.asarray(proposal_points)),
            dtype=np.float64,
        )
        ** 2
    )
    teacher_values = casscf_krylov_teacher_values(teacher_model, proposal_points)[
        :, : int(head_count)
    ]
    dressed_values = _dressed_teacher_values_np(
        params,
        ground,
        teacher_model,
        proposal_points,
        head_count=head_count,
        batch_size=batch_size,
    )
    tiny = np.finfo(np.float64).tiny
    return {
        "ground_norm": max(float(np.mean(ground_raw / proposal_density)), tiny),
        "source_norm": max(float(np.mean(source_raw / proposal_density)), tiny),
        "teacher_norms": np.maximum(
            np.mean(teacher_values**2 / proposal_density[:, None], axis=0),
            tiny,
        ),
        "dressed_norms": np.maximum(
            np.mean(dressed_values**2 / proposal_density[:, None], axis=0),
            tiny,
        ),
    }


def _normalized_production_component_densities(
    params: Params,
    ground: FermiNetGround,
    teacher_model: Any,
    points: np.ndarray,
    *,
    head_count: int,
    normalizers: Mapping[str, Any],
    batch_size: int,
    auxiliary_density: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the normalized production proposal components at ``points``.

    Returns:
        Array with columns for ground, source, teacher, dressed, and optionally
        auxiliary components.
    """
    points_np = np.asarray(points, dtype=np.float64)
    ground_raw = _ground_density_np(ground, points_np, batch_size=batch_size)
    source_raw = (
        np.asarray(
            source_values(ground, jnp.asarray(points_np)),
            dtype=np.float64,
        )
        ** 2
    )
    teacher_values = casscf_krylov_teacher_values(teacher_model, points_np)[
        :, : int(head_count)
    ]
    dressed_values = _dressed_teacher_values_np(
        params,
        ground,
        teacher_model,
        points_np,
        head_count=head_count,
        batch_size=batch_size,
    )
    teacher_norms = np.asarray(normalizers["teacher_norms"], dtype=np.float64)
    dressed_norms = np.asarray(normalizers["dressed_norms"], dtype=np.float64)
    component_list = [
        ground_raw / float(normalizers["ground_norm"]),
        source_raw / float(normalizers["source_norm"]),
        np.mean(teacher_values**2 / teacher_norms[None, :], axis=1),
        np.mean(dressed_values**2 / dressed_norms[None, :], axis=1),
    ]
    if auxiliary_density is not None:
        component_list.append(np.asarray(auxiliary_density, dtype=np.float64))
    components = np.column_stack(component_list)
    return np.maximum(components, np.finfo(np.float64).tiny)


def _line_search_simplex_vertex_step(
    denominator: np.ndarray,
    vertex_delta: np.ndarray,
    weights_squared: np.ndarray,
) -> float:
    """Return the exact convex line-search step to a simplex vertex."""
    tiny = MATRIX_LEVERAGE_DENOMINATOR_FLOOR

    def derivative(step: float) -> float:
        trial = np.maximum(denominator + step * vertex_delta, tiny)
        return -float(np.sum(weights_squared * vertex_delta / trial**2))

    if derivative(0.0) >= 0.0:
        return 0.0
    if derivative(1.0) <= 0.0:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if derivative(mid) <= 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _optimize_matrix_leverage_mixture_weights(
    components: np.ndarray,
    leverage: np.ndarray,
    *,
    iterations: int = 128,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Solve ``min_pi sum l_i^2 / (H_i pi)`` over the probability simplex.

    Returns:
        Optimized simplex weights and final objective value.

    Raises:
        ValueError: If the component matrix and leverage vector are incompatible.
        RuntimeError: If the optimizer produces invalid weights.
    """
    matrix = np.maximum(
        np.asarray(components, dtype=np.float64),
        MATRIX_LEVERAGE_DENOMINATOR_FLOOR,
    )
    scores = np.maximum(
        np.asarray(leverage, dtype=np.float64),
        MATRIX_LEVERAGE_DENOMINATOR_FLOOR,
    )
    if matrix.ndim != 2 or matrix.shape[0] != scores.shape[0]:
        msg = "bright-influence mixture optimization received incompatible shapes"
        raise ValueError(msg)
    component_count = int(matrix.shape[1])
    if component_count < 1:
        msg = "bright-influence mixture optimization needs at least one component"
        raise ValueError(msg)
    pi = np.full((component_count,), 1.0 / float(component_count), dtype=np.float64)
    weights_squared = scores**2
    objective = float("inf")
    for _ in range(int(iterations)):
        denominator = np.maximum(matrix @ pi, MATRIX_LEVERAGE_DENOMINATOR_FLOOR)
        objective = float(np.sum(weights_squared / denominator))
        gradient = -(matrix.T @ (weights_squared / denominator**2))
        vertex = int(np.argmin(gradient))
        vertex_value = float(gradient[vertex])
        dual_gap = float(pi @ gradient - vertex_value)
        if dual_gap <= float(tolerance) * max(1.0, abs(objective)):
            break
        vertex_density = matrix[:, vertex]
        step = _line_search_simplex_vertex_step(
            denominator,
            vertex_density - denominator,
            weights_squared,
        )
        if step <= 0.0:
            break
        pi *= 1.0 - step
        pi[vertex] += step
    pi = np.maximum(pi, 0.0)
    pi_sum = float(np.sum(pi))
    if not np.isfinite(pi_sum) or pi_sum <= 0.0:
        msg = "bright-influence mixture optimization produced invalid weights"
        raise RuntimeError(msg)
    pi /= pi_sum
    final_denominator = np.maximum(
        matrix @ pi,
        MATRIX_LEVERAGE_DENOMINATOR_FLOOR,
    )
    final_objective = float(np.sum(weights_squared / final_denominator))
    return pi, final_objective


def _project_precomputed_values_against_ground_np(
    ground: FermiNetGround,
    points: np.ndarray,
    density: np.ndarray,
    values: np.ndarray,
    gradients: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same sample-estimated ``Q0`` projection used by final matrices.

    Returns:
        Projected values, projected gradients, and potential shift values.
    """
    points_np = np.asarray(points, dtype=np.float64)
    density_np = np.maximum(
        np.asarray(density, dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    ground_value_pieces = []
    ground_gradient_pieces = []
    potential_pieces = []
    for chunk in make_batches(points_np.shape[0], int(batch_size)):
        chunk_points = jnp.asarray(points_np[chunk])
        ground_values, ground_gradients = ground_values_and_gradients(
            ground,
            chunk_points,
        )
        ground_value_pieces.append(np.asarray(ground_values, dtype=np.float64))
        ground_gradient_pieces.append(np.asarray(ground_gradients, dtype=np.float64))
        potential_pieces.append(
            np.asarray(potential_shift(ground, chunk_points), dtype=np.float64)
        )
    ground_values_np = np.concatenate(ground_value_pieces, axis=0)
    ground_gradients_np = np.concatenate(ground_gradient_pieces, axis=0)
    potential_shift_np = np.concatenate(potential_pieces, axis=0)
    weights = 1.0 / density_np / float(points_np.shape[0])
    denominator = float(
        np.einsum("n,n,n->", weights, ground_values_np, ground_values_np)
    )
    denominator = max(denominator, np.finfo(np.float64).tiny)
    coeff = np.einsum("n,n,ni->i", weights, ground_values_np, values) / denominator
    projected_values = values - ground_values_np[:, None] * coeff[None, :]
    projected_gradients = (
        gradients - ground_gradients_np[:, None, :, :] * coeff[None, :, None, None]
    )
    return projected_values, projected_gradients, potential_shift_np


def _head_weak_matrices_from_projected_np(
    head_values: np.ndarray,
    head_gradients: np.ndarray,
    source_projected: np.ndarray,
    potential_shift_np: np.ndarray,
    density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble head-only weak matrices from already projected pilot values.

    Returns:
        Head-only overlap, weak Hamiltonian, and projected source vector.
    """
    sample_count = int(head_values.shape[0])
    weights = 1.0 / np.maximum(density, np.finfo(np.float64).tiny) / sample_count
    overlap = np.einsum("n,ni,nj->ij", weights, head_values, head_values)
    flat_gradients = head_gradients.reshape(sample_count, head_values.shape[1], -1)
    kinetic = 0.5 * np.einsum("nid,njd->nij", flat_gradients, flat_gradients)
    potential = potential_shift_np[:, None, None] * (
        head_values[:, :, None] * head_values[:, None, :]
    )
    hamiltonian = np.einsum("n,nij->ij", weights, kinetic + potential)
    source = np.einsum("n,ni,n->i", weights, head_values, source_projected)[:, None]
    return overlap, hamiltonian, source


def sample_cas_dressed_teacher_bright_influence_distribution(  # noqa: C901
    params: Params,
    ground: FermiNetGround,
    teacher_model: Any,
    *,
    head_count: int,
    n_samples: int,
    core_decay: float,
    diffuse_decay: float,
    batch_size: int,
    seed: int,
    basis: str,
    finite_difference_step: float,
    candidate_factor: int = 2,
    max_candidate_samples: int = 32768,
    gradient_weight: float = 1.0,
    potential_weight: float = 1.0,
    source_weight: float = 1.0,
    winsor_quantile: float = 0.995,
    floor_fraction: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample the bright-influence proposal for production weak matrices.

    The pilot pool is drawn from the auxiliary Sobol proposal.  Each pilot point
    is projected against the ground state, assigned the tracked bright-pole
    influence score ``|k_b(R)-Omega_b s_b(R)|`` plus the source-strength
    influence, and then used to optimize the existing production proposal
    mixture.  Production samples are drawn from that optimized mixture.

    Returns:
        Sampled points, positive normalized production-mixture density values up
        to a common log shift, and diagnostics.

    Raises:
        ValueError: If sampling or leverage parameters are invalid.
    """
    if int(n_samples) < 1:
        msg = "CAS-dressed bright-influence sampling requires positive n_samples"
        raise ValueError(msg)
    if int(head_count) < 1:
        msg = "CAS-dressed bright-influence sampling requires a positive head_count"
        raise ValueError(msg)
    if int(batch_size) < 1:
        msg = "CAS-dressed bright-influence batch_size must be positive"
        raise ValueError(msg)
    if int(candidate_factor) < 1:
        msg = "CAS-dressed bright-influence candidate_factor must be positive"
        raise ValueError(msg)
    if int(max_candidate_samples) < 0:
        msg = "CAS-dressed bright-influence max_candidate_samples must be nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(gradient_weight)
        and np.isfinite(potential_weight)
        and np.isfinite(source_weight)
        and gradient_weight >= 0.0
        and potential_weight >= 0.0
        and source_weight >= 0.0
    ):
        msg = "CAS-dressed bright-influence weights must be finite nonnegative"
        raise ValueError(msg)
    if not (np.isfinite(winsor_quantile) and 0.0 < winsor_quantile <= 1.0):
        msg = "CAS-dressed bright-influence winsor_quantile must be in (0, 1]"
        raise ValueError(msg)
    if not (np.isfinite(floor_fraction) and floor_fraction >= 0.0):
        msg = "CAS-dressed bright-influence floor_fraction must be nonnegative"
        raise ValueError(msg)

    candidate_samples = max(int(n_samples), int(n_samples) * int(candidate_factor))
    if int(max_candidate_samples) > 0:
        candidate_samples = max(
            int(n_samples),
            min(candidate_samples, int(max_candidate_samples)),
        )
    proposal_points, proposal_density = sample_warm_start_auxiliary_proposal(
        n_samples=candidate_samples,
        core_decay=core_decay,
        diffuse_decay=diffuse_decay,
        electron_shape=ground.electron_shape,
        seed=seed + 17,
    )
    proposal_density = np.maximum(
        np.asarray(proposal_density, dtype=np.float64),
        np.finfo(np.float64).tiny,
    )
    normalizers = _production_component_normalizers(
        params,
        ground,
        teacher_model,
        proposal_points,
        proposal_density,
        head_count=head_count,
        batch_size=batch_size,
    )
    value_blocks, gradient_blocks = cas_dressed_teacher_basis_value_gradient_blocks(
        params,
        ground,
        teacher_model,
        [proposal_points],
        head_count=head_count,
        basis=basis,
        finite_difference_step=finite_difference_step,
        batch_size=batch_size,
    )
    values_raw = np.asarray(value_blocks[0], dtype=np.float64)
    gradients_raw = np.asarray(gradient_blocks[0], dtype=np.float64)
    projected_values, projected_gradients, potential_shift_np = (
        _project_precomputed_values_against_ground_np(
            ground,
            proposal_points,
            proposal_density,
            values_raw,
            gradients_raw,
            batch_size=batch_size,
        )
    )
    source_projected = projected_values[:, 0]
    head_values = projected_values[:, 1:]
    head_gradients = projected_gradients[:, 1:]
    overlap, hamiltonian, source_vector = _head_weak_matrices_from_projected_np(
        head_values,
        head_gradients,
        source_projected,
        potential_shift_np,
        proposal_density,
    )
    (
        _,
        roots_jax,
        source_weights_jax,
        _,
        ritz_coefficients_jax,
    ) = _projected_source_spectrum_core(
        jnp.asarray(overlap),
        jnp.asarray(hamiltonian),
        jnp.asarray(source_vector),
    )
    roots_np = np.asarray(roots_jax, dtype=np.float64)
    source_weights_np = np.maximum(
        np.asarray(source_weights_jax, dtype=np.float64),
        0.0,
    )
    ritz_coefficients_np = np.asarray(ritz_coefficients_jax, dtype=np.float64)
    source_weight_max = (
        float(np.max(source_weights_np)) if source_weights_np.size else 0.0
    )
    bright_cutoff = 0.05 * max(source_weight_max, 0.0)
    bright_indices_np = np.flatnonzero(source_weights_np >= bright_cutoff).astype(
        np.int64
    )
    if bright_indices_np.size == 0:
        bright_indices_np = np.asarray(
            [int(np.argmax(source_weights_np)) if source_weights_np.size else 0],
            dtype=np.int64,
        )
    bright_roots = roots_np[bright_indices_np]
    bright_source_weights = source_weights_np[bright_indices_np]
    if not np.any(bright_source_weights > 0.0):
        bright_rho = np.ones((bright_indices_np.size,), dtype=np.float64)
    else:
        bright_rho = bright_source_weights
    bright_rho = bright_rho / np.sum(bright_rho)
    bright_coefficients = ritz_coefficients_np[:, bright_indices_np]
    bright_values = head_values @ bright_coefficients
    bright_gradients = np.einsum(
        "naei,ab->nbei",
        head_gradients,
        bright_coefficients,
    )
    bright_flat_gradients = bright_gradients.reshape(
        head_values.shape[0],
        bright_indices_np.size,
        -1,
    )
    bright_overlap_integrand = bright_values**2
    bright_kinetic_integrand = 0.5 * np.sum(bright_flat_gradients**2, axis=2)
    bright_potential_integrand = potential_shift_np[:, None] * bright_overlap_integrand
    bright_hamiltonian_integrand = (
        float(gradient_weight) * bright_kinetic_integrand
        + float(potential_weight) * bright_potential_integrand
    )
    bright_energy_influence = (
        bright_hamiltonian_integrand - bright_roots[None, :] * bright_overlap_integrand
    )
    influence_second_moment = np.sum(
        bright_rho[None, :] * bright_energy_influence**2,
        axis=1,
    )
    source_amplitudes = source_vector[:, 0] @ bright_coefficients
    source_strength_influence = (
        2.0 * source_amplitudes[None, :] * source_projected[:, None] * bright_values
    )
    influence_second_moment += float(source_weight) * np.sum(
        bright_rho[None, :] * source_strength_influence**2,
        axis=1,
    )
    leverage = np.sqrt(np.maximum(influence_second_moment, 0.0))
    leverage = np.maximum(leverage, 0.0)
    finite_positive = leverage[np.isfinite(leverage) & (leverage > 0.0)]
    if finite_positive.size == 0:
        msg = "CAS-dressed bright-influence scores are all zero or nonfinite"
        raise ValueError(msg)
    winsor_limit = float(np.quantile(finite_positive, float(winsor_quantile)))
    leverage = np.where(np.isfinite(leverage), leverage, winsor_limit)
    leverage = np.minimum(leverage, winsor_limit)
    floor = float(floor_fraction) * max(float(np.mean(finite_positive)), 1e-300)
    leverage = np.maximum(leverage, max(floor, np.finfo(np.float64).tiny))

    component_densities = _normalized_production_component_densities(
        params,
        ground,
        teacher_model,
        proposal_points,
        head_count=head_count,
        normalizers=normalizers,
        batch_size=batch_size,
        auxiliary_density=proposal_density,
    )
    component_weights, leverage_objective = _optimize_matrix_leverage_mixture_weights(
        component_densities, leverage
    )
    mixture_density = np.maximum(
        component_densities @ component_weights,
        np.finfo(np.float64).tiny,
    )
    log_weights = np.log(mixture_density) - np.log(proposal_density)
    log_weights_shift = float(np.max(log_weights))
    relative_weights = np.exp(log_weights - log_weights_shift)
    weight_sum = float(np.sum(relative_weights))
    probabilities = relative_weights / weight_sum
    mixture_normalizer = float(
        np.exp(log_weights_shift) * weight_sum / float(candidate_samples)
    )
    mixture_normalizer = max(mixture_normalizer, np.finfo(np.float64).tiny)
    proposal_ess = float(weight_sum**2 / np.sum(relative_weights**2))
    indices = systematic_resample(probabilities, int(n_samples), seed=seed + 31)
    samples = np.asarray(proposal_points[indices], dtype=np.float64)
    selected_density = np.asarray(mixture_density[indices], dtype=np.float64)
    selected_density = selected_density / mixture_normalizer
    final_log_density = np.log(np.maximum(selected_density, np.finfo(np.float64).tiny))
    density_log_shift = float(np.max(final_log_density))
    density = np.exp(final_log_density - density_log_shift)
    density = np.maximum(density, np.finfo(np.float64).tiny)
    unique_fraction = float(np.unique(indices).size / int(n_samples))
    stats = {
        "sampler": "bright_influence_mixture_sobol_resampling",
        "pmove": 1.0,
        "density_log_shift": density_log_shift,
        "proposal_samples": int(candidate_samples),
        "proposal_ess": proposal_ess,
        "proposal_ess_fraction": proposal_ess / float(candidate_samples),
        "proposal_max_weight_fraction": float(np.max(probabilities)),
        "resampling_unique_fraction": unique_fraction,
        "leverage_normalizer": mixture_normalizer,
        "leverage_mean": float(np.mean(leverage)),
        "leverage_max": float(np.max(leverage)),
        "leverage_winsor_limit": winsor_limit,
        "leverage_floor": max(floor, np.finfo(np.float64).tiny),
        "leverage_gradient_weight": float(gradient_weight),
        "leverage_potential_weight": float(potential_weight),
        "leverage_source_weight": float(source_weight),
        "leverage_winsor_quantile": float(winsor_quantile),
        "leverage_floor_fraction": float(floor_fraction),
        "leverage_mixture_objective": leverage_objective,
        "leverage_component_weights": component_weights,
        "leverage_bright_indices": bright_indices_np,
        "leverage_bright_roots": bright_roots,
        "leverage_bright_source_weights": bright_source_weights,
        "leverage_bright_rho": bright_rho,
        "log_weight_shift": log_weights_shift,
        "proposal_component_count": 0,
        "ground_norm": float(normalizers["ground_norm"]),
        "source_norm": float(normalizers["source_norm"]),
        "teacher_norms": np.asarray(normalizers["teacher_norms"], dtype=np.float64),
        "dressed_norms": np.asarray(normalizers["dressed_norms"], dtype=np.float64),
        "ground_weight": float(component_weights[0]),
        "source_weight": float(component_weights[1]),
        "teacher_weight": float(component_weights[2]),
        "dressed_weight": float(component_weights[3]),
        "aux_weight": float(component_weights[4]),
        "walkers": 0,
        "burn_in": 0,
        "steps_between": 1,
        "width": float("nan"),
        "resampling_factor": int(candidate_factor),
        "max_candidate_samples": int(max_candidate_samples),
    }
    return samples, density, stats


def rescale_density_pieces_to_common_log_shift(
    density_pieces: list[np.ndarray],
    sampling_stats: list[Mapping[str, Any]],
) -> tuple[list[np.ndarray], float, np.ndarray]:
    """Put per-replica production densities on one common log-density scale.

    The production sampler returns ``exp(log q - c_b)`` for each replica/block
    to avoid overflow.  A constant cancels inside a single generalized
    eigenproblem, but not when multiple blocks are averaged or bootstrapped.
    This helper converts all blocks to ``exp(log q - max_b c_b)``.

    Returns:
        Rescaled density pieces, the global log-density shift, and the
        original per-piece shifts.

    Raises:
        ValueError: If piece/stat counts mismatch or shifts are nonfinite.
    """
    if len(density_pieces) != len(sampling_stats):
        msg = "density pieces and sampling stats must have matching lengths"
        raise ValueError(msg)
    if not density_pieces:
        return [], float("nan"), np.asarray([], dtype=np.float64)
    shifts = np.asarray(
        [float(stats["density_log_shift"]) for stats in sampling_stats],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(shifts)):
        msg = "production density log shifts must be finite"
        raise ValueError(msg)
    global_shift = float(np.max(shifts))
    rescaled = [
        np.maximum(
            np.asarray(density, dtype=np.float64) * np.exp(shift - global_shift),
            np.finfo(np.float64).tiny,
        )
        for density, shift in zip(density_pieces, shifts, strict=True)
    ]
    return rescaled, global_shift, shifts


class FermiNetResponseHeads(nn.Module):
    """Direct antisymmetric FermiNet-style response heads.

    Each head is a signed sum of determinant wavefunctions produced by the
    standard JaQMC molecular FermiNet feature, two-electron backbone, orbital,
    and envelope modules.  The output is ``chi_theta_i(R)`` itself, not a ratio
    or product with the restored ground-state wavefunction.
    """

    nspins: tuple[int, int]
    n_heads: int
    determinants_per_head: int = 1
    hidden_dims_single: tuple[int, ...] = (16, 16)
    hidden_dims_double: tuple[int, ...] = (4, 4)
    envelope: EnvelopeType = EnvelopeType.abs_isotropic
    orbitals_spin_split: bool = True
    use_last_layer: bool = False
    orbital_radial_powers: int = 0
    orbital_radial_scale: float = 1.0
    spatial_parity: str = "none"

    def setup(self) -> None:
        hidden_dims = list(zip(self.hidden_dims_single, self.hidden_dims_double))
        total_determinants = self.n_heads * self.determinants_per_head
        self.feature_layer = MoleculeFeatures()
        self.backbone_layer = FermiLayers(
            self.nspins,
            hidden_dims,
            use_last_layer=self.use_last_layer,
        )
        self.orbital_layer = OrbitalProjection(
            nspins=self.nspins,
            ndets=total_determinants,
            orbitals_spin_split=self.orbitals_spin_split,
            use_bias=False,
        )
        self.envelope_layer = Envelope(
            envelope_type=self.envelope,
            ndets=total_determinants,
            nspins=self.nspins,
            orbitals_spin_split=self.orbitals_spin_split,
        )
        self._orbital_radial_coeff = (
            self.param(
                "orbital_radial_coeff",
                nn.initializers.zeros,
                (total_determinants, self.orbital_radial_powers),
            )
            if self.orbital_radial_powers > 0
            else None
        )

    def _raw_values(self, data: MoleculeData) -> jax.Array:
        embedding = self.feature_layer(data.electrons, data.atoms)
        h_one, _ = self.backbone_layer(
            embedding["ae_features"], embedding["ee_features"]
        )
        orbitals = self.orbital_layer(h_one)
        orbitals = orbitals * self.envelope_layer(
            embedding["ae_vec"], embedding["r_ae"]
        )
        if self.orbital_radial_powers > 0:
            charge_center = jnp.sum(data.atoms * data.charges[:, None], axis=0)
            charge_center = charge_center / jnp.sum(data.charges)
            column_mask = jnp.zeros((orbitals.shape[-1],), dtype=orbitals.dtype)
            column_mask = column_mask.at[-1].set(1.0)
            multipliers = jnp.ones_like(orbitals)
            shifted = data.electrons - charge_center[None, :]
            radius = jnp.sqrt(jnp.sum(shifted**2, axis=1) + 1e-12)
            radial_feature = radius / (float(self.orbital_radial_scale) + radius)
            powers = radial_feature[:, None] ** jnp.arange(
                1, self.orbital_radial_powers + 1, dtype=orbitals.dtype
            )
            coeff = self._orbital_radial_coeff
            radial_poly = 1.0 + jnp.einsum("np,dp->dn", powers, coeff)
            multipliers = multipliers * (
                1.0 + column_mask[None, None, :] * (radial_poly[:, :, None] - 1.0)
            )
            orbitals = orbitals * multipliers
        signs, logdets = jnp.linalg.slogdet(orbitals)
        signs = jnp.reshape(signs, (self.n_heads, self.determinants_per_head))
        logdets = jnp.reshape(logdets, (self.n_heads, self.determinants_per_head))
        logmax = jnp.max(logdets, axis=1)
        signed_sum = jnp.sum(signs * jnp.exp(logdets - logmax[:, None]), axis=1)
        values = signed_sum * jnp.exp(logmax)
        return values

    def __call__(self, data: MoleculeData) -> jax.Array:
        values = self._raw_values(data)
        if self.spatial_parity == "none":
            return values
        charge_center = jnp.sum(data.atoms * data.charges[:, None], axis=0)
        charge_center = charge_center / jnp.sum(data.charges)
        inverted = MoleculeData(
            electrons=2.0 * charge_center[None, :] - data.electrons,
            atoms=data.atoms,
            charges=data.charges,
        )
        inverted_values = self._raw_values(inverted)
        if self.spatial_parity == "odd":
            return 0.5 * (values - inverted_values)
        if self.spatial_parity == "even":
            return 0.5 * (values + inverted_values)
        msg = (
            "spatial_parity must be one of 'none', 'odd', or 'even', "
            f"got {self.spatial_parity!r}"
        )
        raise ValueError(msg)


ResponseModel = FermiNetResponseHeads


def make_response_model(
    *,
    nspins: tuple[int, int],
    n_heads: int,
    hidden: int,
    hidden_double: int,
    layers: int,
    determinants_per_head: int,
    orbital_radial_powers: int = 0,
    orbital_radial_scale: float = 1.0,
    spatial_parity: str = "none",
) -> FermiNetResponseHeads:
    hidden_dims_single = tuple([hidden] * layers)
    hidden_dims_double = tuple([hidden_double] * layers)
    return FermiNetResponseHeads(
        nspins=nspins,
        n_heads=n_heads,
        determinants_per_head=determinants_per_head,
        hidden_dims_single=hidden_dims_single,
        hidden_dims_double=hidden_dims_double,
        orbital_radial_powers=orbital_radial_powers,
        orbital_radial_scale=orbital_radial_scale,
        spatial_parity=spatial_parity,
    )


def init_response_params(
    key: jax.Array,
    *,
    ground: FermiNetGround,
    initial_decay_min: float,
    initial_decay_max: float,
) -> Params:
    if ground.response_model is None:
        msg = "ground.response_model must be configured before init_response_params"
        raise ValueError(msg)
    params = ground.response_model.init(
        key,
        _ground_data(ground, jnp.zeros(ground.electron_shape)),
    )
    return initialize_response_envelope_decay(
        params,
        n_heads=ground.response_model.n_heads,
        determinants_per_head=ground.response_model.determinants_per_head,
        initial_decay_min=initial_decay_min,
        initial_decay_max=initial_decay_max,
    )


def _response_envelope_determinant_decays(
    *,
    n_heads: int,
    determinants_per_head: int,
    initial_decay_min: float,
    initial_decay_max: float,
    decay_values: jax.Array | np.ndarray | None,
) -> jax.Array:
    total_determinants = n_heads * determinants_per_head
    if decay_values is None:
        return jnp.repeat(
            jnp.linspace(initial_decay_max, initial_decay_min, n_heads),
            repeats=determinants_per_head,
        )
    decays_np = np.asarray(decay_values, dtype=np.float64).reshape(-1)
    if np.any(decays_np <= 0) or not np.all(np.isfinite(decays_np)):
        msg = "response envelope decays must be finite and positive"
        raise ValueError(msg)
    if decays_np.size == n_heads:
        return jnp.repeat(jnp.asarray(decays_np), repeats=determinants_per_head)
    if decays_np.size == total_determinants:
        return jnp.asarray(decays_np)
    msg = (
        "response envelope decay initializer length must be n_heads "
        "or n_heads*determinants_per_head"
    )
    raise ValueError(msg)


def initialize_response_envelope_decay(
    params: Params,
    *,
    n_heads: int,
    determinants_per_head: int,
    initial_decay_min: float,
    initial_decay_max: float,
    decay_values: jax.Array | np.ndarray | None = None,
) -> Params:
    """Set response-envelope ``sigma`` values by head for stable initialization.

    Returns:
        A Flax parameter tree with envelope decays spread from
        ``initial_decay_max`` to ``initial_decay_min`` over response heads, or
        with explicit ``decay_values`` if supplied.
    """
    total_determinants = n_heads * determinants_per_head
    determinant_decays = _response_envelope_determinant_decays(
        n_heads=n_heads,
        determinants_per_head=determinants_per_head,
        initial_decay_min=initial_decay_min,
        initial_decay_max=initial_decay_max,
        decay_values=decay_values,
    )
    mutable = unfreeze(params)

    def update_sigma(
        tree: dict[str, Any],
        head_decays: jax.Array | None = None,
    ) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                if key.startswith("head_"):
                    head_idx = int(key.removeprefix("head_"))
                    start = head_idx * determinants_per_head
                    stop = start + determinants_per_head
                    update_sigma(value, determinant_decays[start:stop])
                else:
                    update_sigma(value, head_decays)
            elif key == "sigma" and value.shape[-1] == total_determinants:
                shape = (1,) * (value.ndim - 1) + (total_determinants,)
                tree[key] = jnp.ones_like(value) * determinant_decays.reshape(shape)
            elif (
                head_decays is not None
                and key == "sigma"
                and value.shape[-1] == determinants_per_head
            ):
                shape = (1,) * (value.ndim - 1) + (determinants_per_head,)
                tree[key] = jnp.ones_like(value) * head_decays.reshape(shape)

    update_sigma(mutable)
    return freeze(mutable)


def project_response_envelope_sigma_floor(
    params: Params,
    *,
    decay_floor: float,
) -> Params:
    """Project trainable response envelope decay rates above a floor.

    Returns:
        Parameter tree with ``envelope_layer`` sigma leaves clipped in
        absolute value when ``decay_floor`` is positive.
    """
    if decay_floor <= 0.0:
        return params
    floor = jnp.asarray(float(decay_floor), dtype=jnp.float64)
    mutable = unfreeze(params)

    def update(tree: dict[str, Any], path: tuple[str, ...] = ()) -> None:
        for key, value in tree.items():
            next_path = (*path, str(key))
            if isinstance(value, dict):
                update(value, next_path)
            elif "envelope_layer" in next_path and key == "sigma":
                sign = jnp.where(value < 0, -1.0, 1.0).astype(value.dtype)
                tree[key] = sign * jnp.maximum(
                    jnp.abs(value),
                    floor.astype(value.dtype),
                )

    update(mutable)
    return freeze(mutable)


def response_warm_start_train_mask(params: Params):
    """Return an optax mask for value warm-start training.

    The orbital-basis targets are finite-region value data.  Keeping response
    envelope decay parameters fixed preserves the initialized asymptotic tail
    so the warm-start cannot fit QC values by creating uncontrolled tail
    growth.  Envelope amplitudes remain trainable.

    Returns:
        A PyTree mask that trains all response parameters except envelope
        decay leaves and fixed linear head transforms.
    """

    def walk(tree: Any, path: tuple[str, ...] = ()):
        if isinstance(tree, Mapping):
            return {key: walk(value, (*path, str(key))) for key, value in tree.items()}
        freezes_decay = "envelope_layer" in path and path[-1:] == ("sigma",)
        return not freezes_decay and _RESPONSE_HEAD_TRANSFORM_KEY not in path

    return freeze(walk(unfreeze(params)))


def make_warm_start_optimizer(
    learning_rate: float,
    params: Params,
    *,
    freeze_envelope_decay: bool,
) -> optax.GradientTransformation:
    """Build the Adam optimizer used by bounded correction training.

    Returns:
        Adam optimizer, optionally masked to freeze envelope decay leaves.
    """
    tx = optax.adam(float(learning_rate))
    if not freeze_envelope_decay:
        return tx
    return optax.masked(tx, response_warm_start_train_mask(params))


def num_response_heads(ground: FermiNetGround) -> int:
    if ground.response_model is None:
        msg = "ground.response_model must be configured before response evaluation"
        raise ValueError(msg)
    return ground.response_model.n_heads


def response_value_single(
    params: Params, ground: FermiNetGround, coords: jax.Array
) -> jax.Array:
    """Evaluate direct antisymmetric neural response heads.

    The heads are FermiNet-style continuous-coordinate wavefunctions and are
    not parameterized as ``Psi0 * f``.

    Returns:
        One scalar direct response value per neural head.

    Raises:
        ValueError: If no response model is attached to the restored ground.
    """
    if is_response_block_dictionary(params):
        pieces = [
            response_value_single(params[_RESPONSE_BLOCKS_KEY][name], ground, coords)
            for name in response_block_names(params)
        ]
        if not pieces:
            return jnp.zeros((0,), dtype=coords.dtype)
        return jnp.concatenate(pieces, axis=0)
    if ground.response_model is not None:
        raw_values = ground.response_model.apply(params, _ground_data(ground, coords))
    else:
        msg = "ground.response_model must be configured before response evaluation"
        raise ValueError(msg)
    transform = response_head_transform(params)
    values = (
        raw_values
        if transform is None
        else raw_values
        @ jnp.asarray(
            transform,
            dtype=raw_values.dtype,
        )
    )
    source_lift_coeff = response_source_lift_coefficients(params)
    if source_lift_coeff is not None:
        source_lift = source_lift_features_single(ground, coords)
        values = values + source_lift @ jnp.asarray(
            source_lift_coeff,
            dtype=values.dtype,
        )
    return values


def cusp_neutral_radius(distance: jax.Array, radius: float) -> jax.Array:
    """Map distances to a cusp-neutral radial coordinate.

    Returns:
        ``sqrt(r^2+r0^2)-r0``, whose derivative at the origin is zero.
    """
    radius_value = jnp.asarray(float(radius), dtype=distance.dtype)
    return jnp.sqrt(distance**2 + radius_value**2) - radius_value


def source_cusp_lift_tau(distance: jax.Array, radius: jax.Array | float) -> jax.Array:
    """Local radial coordinate with unit origin slope and compact far effect.

    Returns:
        A smooth tau used in p-wave cusp-lift factors.
    """
    radius_value = jnp.asarray(radius, dtype=distance.dtype)
    return radius_value * (-jnp.expm1(-distance / radius_value))


def source_cusp_support_radii_from_radius(
    ground: FermiNetGround,
    radius: float,
) -> jax.Array:
    """Return non-overlapping per-nucleus support radii for a requested radius."""
    requested = jnp.asarray(float(radius), dtype=ground.atoms.dtype)
    atom_count = int(ground.atoms.shape[0])
    if atom_count <= 1:
        return jnp.full((atom_count,), requested, dtype=ground.atoms.dtype)
    atom_delta = ground.atoms[:, None, :] - ground.atoms[None, :, :]
    atom_distance = jnp.sqrt(jnp.sum(atom_delta**2, axis=2) + 1e-24)
    atom_distance = jnp.where(
        jnp.eye(atom_count, dtype=bool),
        jnp.asarray(jnp.inf, dtype=ground.atoms.dtype),
        atom_distance,
    )
    nearest = jnp.min(atom_distance, axis=1)
    nonoverlap = 0.45 * nearest
    return jnp.maximum(
        jnp.minimum(requested, nonoverlap),
        jnp.asarray(1e-6, dtype=ground.atoms.dtype),
    )


def response_head_transform(params: Params) -> jax.Array | None:
    """Return the optional raw-head to response-basis transform."""
    param_tree = params.get("params", {})
    if _RESPONSE_HEAD_TRANSFORM_KEY not in param_tree:
        return None
    return jnp.asarray(param_tree[_RESPONSE_HEAD_TRANSFORM_KEY])


def response_source_lift_coefficients(params: Params) -> jax.Array | None:
    """Return trainable source-sector seed coefficients, if present."""
    param_tree = params.get("params", {})
    if _RESPONSE_SOURCE_LIFT_KEY not in param_tree:
        return None
    return jnp.asarray(param_tree[_RESPONSE_SOURCE_LIFT_KEY])


def source_lift_feature_count() -> int:
    """Return the fixed source-lift seed feature count."""
    return (
        1
        + 3 * len(OFFICIAL_SOURCE_LIFT_RADIAL_SCALES)
        + 2 * len(OFFICIAL_SOURCE_LIFT_PAIR_SCALES)
    )


def source_lift_carrier_features_single(
    ground: FermiNetGround,
    point: jax.Array,
) -> jax.Array:
    """Return source-sector carrier-bank values ``d_z(R) g_k(R)``.

    Returns:
        One real carrier per fixed source-lift bank feature.
    """
    dipole = source_carrier_value_single(ground, point)
    charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
    charge_center = charge_center / jnp.sum(ground.charges)
    shifted = point - charge_center[None, :]
    radii = jnp.sqrt(jnp.sum(shifted**2, axis=1) + 1e-12)
    mean_radius = jnp.mean(radii)
    radial_scales = jnp.asarray(
        OFFICIAL_SOURCE_LIFT_RADIAL_SCALES,
        dtype=point.dtype,
    )
    radial_features = mean_radius / (radial_scales + mean_radius)
    radial_exp_features = jnp.exp(-mean_radius / radial_scales)
    radial_diffuse_features = (
        mean_radius / radial_scales * jnp.exp(-mean_radius / radial_scales)
    )
    pair_scales = jnp.asarray(OFFICIAL_SOURCE_LIFT_PAIR_SCALES, dtype=point.dtype)
    pair_features = electron_pair_features(
        point,
        pair_scales,
    )
    pair_exp_features = electron_pair_exp_features(
        point,
        pair_scales,
    )
    invariant_features = jnp.concatenate(
        [
            jnp.ones((1,), dtype=point.dtype),
            radial_features,
            radial_exp_features,
            radial_diffuse_features,
            pair_features,
            pair_exp_features,
        ],
        axis=0,
    )
    return dipole * invariant_features


def source_lift_features_single(
    ground: FermiNetGround,
    point: jax.Array,
) -> jax.Array:
    """Return nonzero source-sector seed values ``Psi0 * d_z * g_k``.

    The invariant factors are system-independent radial and electron-pair
    scalars, so multiplying them by the physical source carrier keeps the
    dipole response sector while avoiding a raw zero candidate block.

    Returns:
        One wavefunction value per fixed source-lift bank feature.
    """
    phase, logpsi = ground_phase_logpsi_single(ground, point)
    return source_lift_carrier_features_single(ground, point) * phase * jnp.exp(logpsi)


def set_response_source_lift_coefficients(
    params: Params,
    coefficients: np.ndarray | jax.Array,
) -> Params:
    """Attach source-sector seed coefficients to selected response heads.

    Returns:
        Updated parameter tree.

    Raises:
        ValueError: If the coefficient matrix has an invalid shape or values.
    """
    coeff_np = np.asarray(coefficients, dtype=np.float64)
    if coeff_np.ndim != 2:
        msg = "source-lift coefficients must be a matrix"
        raise ValueError(msg)
    if coeff_np.shape[0] != source_lift_feature_count() or coeff_np.shape[1] < 1:
        msg = (
            "source-lift coefficients must have shape "
            f"({source_lift_feature_count()}, n_heads)"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(coeff_np)):
        msg = "source-lift coefficients must be finite"
        raise ValueError(msg)
    mutable = unfreeze(params)
    mutable.setdefault("params", {})[_RESPONSE_SOURCE_LIFT_KEY] = jnp.asarray(
        coeff_np,
    )
    return freeze(mutable)


def initialize_response_source_lift(
    params: Params,
    *,
    head_count: int,
    key: jax.Array,
    scale: float = 1.0,
) -> Params:
    """Initialize nonzero source-sector seed coefficients for a candidate block.

    Returns:
        Updated parameter tree.

    Raises:
        ValueError: If ``head_count`` is not positive.
    """
    head_count = int(head_count)
    if head_count < 1:
        msg = "head_count must be positive"
        raise ValueError(msg)
    feature_count = source_lift_feature_count()
    coeff = 0.05 * np.asarray(
        jax.random.normal(key, (feature_count, head_count)),
        dtype=np.float64,
    )
    seed_features = max(1, feature_count - 1)
    for head in range(head_count):
        coeff[1 + (head % seed_features), head] += float(scale)
    return set_response_source_lift_coefficients(params, coeff)


def is_response_block_dictionary(params: Params) -> bool:
    """Return whether ``params`` stores an appended response-basis dictionary."""
    return _RESPONSE_BLOCKS_KEY in params


def response_block_names(params: Params) -> tuple[str, ...]:
    """Return deterministic response-basis block names."""
    if not is_response_block_dictionary(params):
        return ()
    return tuple(sorted(params[_RESPONSE_BLOCKS_KEY]))


def empty_response_basis_params() -> Params:
    """Return an empty response-basis dictionary."""
    return freeze({_RESPONSE_BLOCKS_KEY: {}})


def response_block_head_count(params: Params, ground: FermiNetGround) -> int:
    """Return the number of selected response heads in one block."""
    transform = response_head_transform(params)
    if transform is not None:
        return int(transform.shape[1])
    return num_response_heads(ground)


def response_basis_head_count(params: Params, ground: FermiNetGround) -> int:
    """Return the total number of selected response heads in ``params``."""
    if not is_response_block_dictionary(params):
        return response_block_head_count(params, ground)
    return sum(
        response_block_head_count(params[_RESPONSE_BLOCKS_KEY][name], ground)
        for name in response_block_names(params)
    )


def select_response_heads(
    params: Params,
    *,
    raw_head_count: int,
    head_count: int,
) -> Params:
    """Attach a transform selecting the first ``head_count`` raw heads.

    Returns:
        Parameter tree with a fixed raw-to-selected head transform.

    Raises:
        ValueError: If ``head_count`` is outside the raw-head range.
    """
    raw_head_count = int(raw_head_count)
    head_count = int(head_count)
    if head_count < 1 or head_count > raw_head_count:
        msg = "head_count must be in [1, raw_head_count]"
        raise ValueError(msg)
    transform = np.eye(raw_head_count, dtype=np.float64)[:, :head_count]
    return set_response_head_transform(params, transform)


def select_scaled_response_heads(
    params: Params,
    *,
    raw_head_count: int,
    head_count: int,
    scale: float,
) -> Params:
    """Attach a scaled first-head selector for small correction initialization.

    Returns:
        Parameter tree with ``scale * I`` stored as the raw-head transform.

    Raises:
        ValueError: If scale is nonfinite or head dimensions are invalid.
    """
    if not np.isfinite(float(scale)):
        msg = "scaled response head selector requires finite scale"
        raise ValueError(msg)
    raw_head_count = int(raw_head_count)
    head_count = int(head_count)
    if head_count < 1 or head_count > raw_head_count:
        msg = "head_count must be in [1, raw_head_count]"
        raise ValueError(msg)
    transform = float(scale) * np.eye(raw_head_count, dtype=np.float64)[:, :head_count]
    return set_response_head_transform(params, transform)


def append_response_block_params(params: Params, block_params: Params) -> Params:
    """Append one independently parameterized response block to a dictionary.

    Returns:
        A response-basis dictionary containing all previous blocks plus the new
        block.
    """
    mutable = unfreeze(params) if is_response_block_dictionary(params) else None
    if mutable is None:
        mutable = unfreeze(empty_response_basis_params())
    blocks = mutable.setdefault(_RESPONSE_BLOCKS_KEY, {})
    next_idx = len(blocks)
    block_name = f"block_{next_idx:03d}"
    while block_name in blocks:
        next_idx += 1
        block_name = f"block_{next_idx:03d}"
    blocks[block_name] = block_params
    return freeze(mutable)


def replace_response_block_params(
    params: Params,
    block_name: str,
    block_params: Params,
) -> Params:
    """Return ``params`` with one response block replaced.

    Raises:
        ValueError: If ``params`` is not a block dictionary or the block is
            unknown.
    """
    if not is_response_block_dictionary(params):
        msg = "response block replacement requires a block dictionary"
        raise ValueError(msg)
    mutable = unfreeze(params)
    if block_name not in mutable[_RESPONSE_BLOCKS_KEY]:
        msg = f"unknown response block {block_name!r}"
        raise ValueError(msg)
    mutable[_RESPONSE_BLOCKS_KEY][block_name] = block_params
    return freeze(mutable)


def last_response_block_name(params: Params) -> str:
    """Return the name of the last appended response block.

    Raises:
        ValueError: If the response-basis dictionary is empty.
    """
    names = response_block_names(params)
    if not names:
        msg = "response basis has no blocks"
        raise ValueError(msg)
    return names[-1]


def set_response_head_transform(
    params: Params, transform: np.ndarray | jax.Array
) -> Params:
    """Attach a fixed linear response-head transform to the parameter tree.

    Returns:
        Parameter tree with the transform stored under the response params.

    Raises:
        ValueError: If the transform is not a finite nonempty matrix.
    """
    transform_np = np.asarray(transform, dtype=np.float64)
    if transform_np.ndim != 2:
        msg = "response head transform must be a matrix"
        raise ValueError(msg)
    if transform_np.shape[0] < 1 or transform_np.shape[1] < 1:
        msg = "response head transform must have at least one row and column"
        raise ValueError(msg)
    if not np.all(np.isfinite(transform_np)):
        msg = "response head transform must be finite"
        raise ValueError(msg)
    mutable = unfreeze(params)
    mutable.setdefault("params", {})[_RESPONSE_HEAD_TRANSFORM_KEY] = jnp.asarray(
        transform_np
    )
    return freeze(mutable)


def right_transform_response_heads(
    params: Params,
    right_transform: np.ndarray | jax.Array,
    *,
    raw_head_count: int,
) -> Params:
    """Apply a right transform to selected direct and source-lift head outputs.

    Returns:
        Updated parameter tree.

    Raises:
        ValueError: If the right transform is invalid or incompatible.
    """
    right_np = np.asarray(right_transform, dtype=np.float64)
    if right_np.ndim != 2 or right_np.shape[0] < 1 or right_np.shape[1] < 1:
        msg = "right_transform must be a nonempty matrix"
        raise ValueError(msg)
    if not np.all(np.isfinite(right_np)):
        msg = "right_transform must be finite"
        raise ValueError(msg)
    existing = response_head_transform(params)
    if existing is None:
        left = np.eye(int(raw_head_count), dtype=np.float64)
    else:
        left = np.asarray(existing, dtype=np.float64)
    if left.shape[1] != right_np.shape[0]:
        msg = "right_transform input dimension must match selected head count"
        raise ValueError(msg)
    updated = set_response_head_transform(params, left @ right_np)
    source_lift = response_source_lift_coefficients(params)
    if source_lift is None:
        return updated
    source_lift_np = np.asarray(source_lift, dtype=np.float64)
    updated_source_lift = source_lift_np @ right_np
    return set_response_source_lift_coefficients(updated, updated_source_lift)


def response_values(
    params: Params, ground: FermiNetGround, points: jax.Array
) -> jax.Array:
    return jax.vmap(response_value_single, (None, None, 0))(params, ground, points)


def response_subspace_pretrain_loss(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    target_values: jax.Array,
    density_or_q: jax.Array,
    *,
    head_count: int,
    ridge: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Value-only warm-start loss from the paper's subspace objective.

    Returns:
        Normalized loss and diagnostics for the batch least-squares solve.
    """
    head_values = response_values(params, ground, points)[:, : int(head_count)]
    targets = jnp.asarray(target_values, dtype=head_values.dtype)
    weights = 1 / density_or_q / points.shape[0]
    gram = jnp.einsum("n,ni,nj->ij", weights, head_values, head_values)
    rhs = jnp.einsum("n,ni,nt->it", weights, head_values, targets)
    eye = jnp.eye(head_values.shape[1], dtype=head_values.dtype)
    coeff = jnp.linalg.solve(gram + float(ridge) * eye, rhs)
    fitted = head_values @ coeff
    residual = fitted - targets
    numerator = jnp.einsum("n,nt,nt->", weights, residual, residual)
    denominator = jnp.maximum(
        jnp.einsum("n,nt,nt->", weights, targets, targets),
        jnp.asarray(1e-30, dtype=head_values.dtype),
    )
    loss = numerator / denominator
    return loss, {
        "loss": loss,
        "target_norm": jnp.sqrt(denominator),
        "residual_norm": jnp.sqrt(jnp.maximum(numerator, 0.0)),
        "coeff_norm": jnp.linalg.norm(coeff),
        "condition": jnp.linalg.cond(gram + float(ridge) * eye),
    }


def init_cas_dressing_params(
    key: jax.Array,
    *,
    teacher_count: int,
    atom_count: int,
    radial_scale_count: int,
    pair_scale_count: int,
    hidden: int = 0,
    layers: int = 0,
    init_scale: float = 0.0,
) -> Params:
    """Initialize the CAS-explicit neural dressing at the identity map.

    Returns:
        Parameter tree for the matrix dressing.

    Raises:
        ValueError: If dimensions or scale are invalid.
    """
    teacher_count = int(teacher_count)
    if teacher_count < 1:
        msg = "CAS dressing requires at least one teacher"
        raise ValueError(msg)
    if int(atom_count) < 1:
        msg = "CAS dressing requires at least one atom"
        raise ValueError(msg)
    scale = float(init_scale)
    if not (np.isfinite(scale) and scale >= 0.0):
        msg = "CAS dressing init_scale must be finite and nonnegative"
        raise ValueError(msg)
    shape = (teacher_count, teacher_count)
    params: Params = {
        "bias": jnp.zeros(shape, dtype=jnp.float64),
        "radial_coeff": scale
        * jnp.zeros(
            (int(atom_count), int(radial_scale_count), *shape),
            dtype=jnp.float64,
        ),
        "pair_coeff": scale
        * jnp.zeros((int(pair_scale_count), *shape), dtype=jnp.float64),
    }
    hidden = int(hidden)
    layers = int(layers)
    if hidden > 0 and layers > 0:
        feature_dim = int(atom_count) * int(radial_scale_count) + int(pair_scale_count)
        keys = jax.random.split(key, layers + 1)
        weights = []
        biases = []
        in_dim = feature_dim
        for layer_idx in range(layers):
            weight = jax.random.normal(keys[layer_idx], (in_dim, hidden)) / np.sqrt(
                max(1, in_dim)
            )
            weights.append(weight.astype(jnp.float64))
            biases.append(jnp.zeros((hidden,), dtype=jnp.float64))
            in_dim = hidden
        params["mlp_weights"] = tuple(weights)
        params["mlp_biases"] = tuple(biases)
        params["mlp_out_weight"] = jnp.zeros((in_dim, teacher_count * teacher_count))
        params["mlp_out_bias"] = jnp.zeros((teacher_count * teacher_count,))
    return params


def cas_dressing_feature_vector_single(
    ground: FermiNetGround,
    point: jax.Array,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
) -> jax.Array:
    """Evaluate permutation-symmetric scalar features for neural dressing.

    Returns:
        One flattened invariant feature vector.
    """
    pieces = []
    if radial_scales.size:
        electron_atom = point[:, None, :] - ground.atoms[None, :, :]
        distances = jnp.sqrt(jnp.sum(electron_atom**2, axis=-1) + 1e-12)
        radial_features = jnp.mean(
            jnp.exp(-distances[:, :, None] / radial_scales[None, None, :]),
            axis=0,
        )
        pieces.append(jnp.ravel(radial_features))
    if pair_scales.size:
        pieces.append(electron_pair_exp_features(point, pair_scales))
    if not pieces:
        return jnp.zeros((0,), dtype=point.dtype)
    return jnp.concatenate(pieces, axis=0)


def cas_dressing_matrix_single(
    params: Params,
    ground: FermiNetGround,
    point: jax.Array,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
) -> jax.Array:
    """Evaluate the permutation-symmetric scalar matrix ``A(R;theta)``.

    Returns:
        Dressing matrix with shape ``(n_teachers, n_teachers)``.
    """
    bias = jnp.asarray(params["bias"], dtype=point.dtype)
    radial_coeff = jnp.asarray(params["radial_coeff"], dtype=point.dtype)
    pair_coeff = jnp.asarray(params["pair_coeff"], dtype=point.dtype)
    teacher_count = bias.shape[0]
    matrix = jnp.eye(teacher_count, dtype=point.dtype) + bias
    features = cas_dressing_feature_vector_single(
        ground,
        point,
        radial_scales,
        pair_scales,
    )
    if radial_scales.size:
        radial_size = ground.atoms.shape[0] * radial_scales.size
        radial_features = jnp.reshape(
            features[:radial_size],
            (ground.atoms.shape[0], radial_scales.size),
        )
        matrix = matrix + jnp.einsum(
            "as,asij->ij",
            radial_features,
            radial_coeff,
        )
    if pair_scales.size:
        pair_features = features[-pair_scales.size :]
        matrix = matrix + jnp.einsum("s,sij->ij", pair_features, pair_coeff)
    if "mlp_weights" in params:
        hidden = features
        for weight, layer_bias in zip(
            params["mlp_weights"],
            params["mlp_biases"],
            strict=True,
        ):
            hidden = jnp.tanh(
                hidden @ jnp.asarray(weight, dtype=point.dtype)
                + jnp.asarray(layer_bias, dtype=point.dtype)
            )
        mlp_out = hidden @ jnp.asarray(
            params["mlp_out_weight"], dtype=point.dtype
        ) + jnp.asarray(params["mlp_out_bias"], dtype=point.dtype)
        matrix = matrix + jnp.reshape(mlp_out, (teacher_count, teacher_count))
    return matrix


def cas_dressing_matrices_and_gradients(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    *,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate ``A`` and coordinate gradients of ``A`` for a batch.

    Returns:
        Dressing matrices and their coordinate gradients.
    """

    def matrix_fn(point: jax.Array) -> jax.Array:
        return cas_dressing_matrix_single(
            params,
            ground,
            point,
            radial_scales,
            pair_scales,
        )

    matrices = jax.vmap(matrix_fn)(points)
    gradients = jax.vmap(jax.jacrev(matrix_fn))(points)
    return matrices, gradients


def cas_dressed_teacher_values_from_arrays(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    teacher_values: jax.Array,
    *,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
) -> jax.Array:
    """Apply the CAS-explicit dressing to precomputed teacher values.

    Returns:
        Dressed response values with one column per dressed teacher.
    """
    matrices = jax.vmap(
        lambda point: cas_dressing_matrix_single(
            params,
            ground,
            point,
            radial_scales,
            pair_scales,
        )
    )(points)
    return jnp.einsum("nb,nba->na", teacher_values, matrices)


def cas_dressed_teacher_values_and_gradients_from_arrays(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    teacher_values: jax.Array,
    teacher_gradients: jax.Array,
    *,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return dressed teacher values and product-rule gradients."""
    matrices, matrix_gradients = cas_dressing_matrices_and_gradients(
        params,
        ground,
        points,
        radial_scales=radial_scales,
        pair_scales=pair_scales,
    )
    values = jnp.einsum("nb,nba->na", teacher_values, matrices)
    gradients = jnp.einsum("nbei,nba->naei", teacher_gradients, matrices)
    gradients = gradients + jnp.einsum(
        "nb,nbaei->naei",
        teacher_values,
        matrix_gradients,
    )
    return values, gradients


def cas_dressed_teacher_projected_stats(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    teacher_values: jax.Array,
    teacher_gradients: jax.Array,
    *,
    head_count: int,
    radial_scales: jax.Array,
    pair_scales: jax.Array,
    max_condition: float,
) -> dict[str, jax.Array]:
    """Assemble the dressed head-only weak matrix and source projection.

    Returns:
        Ritz roots, source weights, matrices, and conditioning diagnostics.
    """
    head_count = int(head_count)
    dressed_values, dressed_gradients = (
        cas_dressed_teacher_values_and_gradients_from_arrays(
            params,
            ground,
            points,
            teacher_values[:, :head_count],
            teacher_gradients[:, :head_count],
            radial_scales=radial_scales,
            pair_scales=pair_scales,
        )
    )
    source_values_block, source_gradients_block = source_values_and_gradients(
        ground,
        points,
    )
    projection_values = jnp.concatenate(
        [source_values_block[:, None], dressed_values],
        axis=1,
    )
    projection_gradients = jnp.concatenate(
        [source_gradients_block[:, None, :, :], dressed_gradients],
        axis=1,
    )
    ground_values, ground_gradients = ground_values_and_gradients(ground, points)
    projection_values, projection_gradients, _ = project_values_against_ground(
        projection_values,
        projection_gradients,
        ground_values,
        ground_gradients,
        density,
    )
    source_projected = projection_values[:, 0]
    dressed_values = projection_values[:, 1:]
    dressed_gradients = projection_gradients[:, 1:]
    overlap, hamiltonian = weak_matrices(
        dressed_values,
        dressed_gradients,
        potential_shift(ground, points),
        density,
    )
    weights = 1 / density / points.shape[0]
    source_vector = jnp.einsum(
        "n,ni,n->i",
        weights,
        dressed_values,
        source_projected,
    )[:, None]
    overlap_norm, roots, source_weights, ritz_vectors, ritz_coefficients = (
        _projected_source_spectrum_core(
            overlap,
            hamiltonian,
            source_vector,
        )
    )
    overlap_evals = jnp.linalg.eigvalsh((overlap_norm + overlap_norm.T) / 2)
    min_eval = jnp.maximum(jnp.min(overlap_evals), 1e-12)
    max_eval = jnp.maximum(jnp.max(overlap_evals), min_eval)
    condition = max_eval / min_eval
    condition_penalty = (
        jnp.maximum(
            jnp.log(condition) - jnp.log(float(max_condition)),
            0.0,
        )
        ** 2
    )
    matrices, matrix_gradients = cas_dressing_matrices_and_gradients(
        params,
        ground,
        points,
        radial_scales=radial_scales,
        pair_scales=pair_scales,
    )
    identity = jnp.eye(head_count, dtype=matrices.dtype)
    dressing_norm = jnp.mean(
        jnp.sum((matrices - identity[None, :, :]) ** 2, axis=(1, 2))
    )
    dressing_grad_norm = jnp.mean(jnp.sum(matrix_gradients**2, axis=(1, 2, 3, 4)))
    return {
        "overlap": overlap,
        "hamiltonian": hamiltonian,
        "source": source_vector,
        "overlap_norm": overlap_norm,
        "roots": roots,
        "source_weights": source_weights,
        "ritz_vectors": ritz_vectors,
        "ritz_coefficients": ritz_coefficients,
        "condition": condition,
        "condition_penalty": condition_penalty,
        "dressing_norm": dressing_norm,
        "dressing_grad_norm": dressing_grad_norm,
    }


def cas_dressed_teacher_fixed_bright_loss(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    teacher_values: jax.Array,
    teacher_gradients: jax.Array,
    baseline_coefficients: jax.Array,
    bright_indices: jax.Array,
    root_weights: jax.Array,
    *,
    head_count: int,
    energy_weight: float,
    visibility_weight: float,
    condition_weight: float,
    overlap_weight: float,
    regularizer_weight: float,
    gradient_regularizer_length: float,
    max_condition: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Optimize fixed baseline bright vectors through Rayleigh quotients.

    Returns:
        Scalar loss and diagnostics for the fixed bright response vectors.
    """
    del overlap_weight
    head_count = int(head_count)
    dressed_values, dressed_gradients = (
        cas_dressed_teacher_values_and_gradients_from_arrays(
            params,
            ground,
            points,
            teacher_values[:, :head_count],
            teacher_gradients[:, :head_count],
            radial_scales=jnp.asarray(
                OFFICIAL_DRESSING_RADIAL_SCALES,
                dtype=points.dtype,
            ),
            pair_scales=jnp.asarray(
                OFFICIAL_DRESSING_PAIR_SCALES,
                dtype=points.dtype,
            ),
        )
    )
    source_values_block, source_gradients_block = source_values_and_gradients(
        ground,
        points,
    )
    projection_values = jnp.concatenate(
        [source_values_block[:, None], dressed_values],
        axis=1,
    )
    projection_gradients = jnp.concatenate(
        [source_gradients_block[:, None, :, :], dressed_gradients],
        axis=1,
    )
    ground_values, ground_gradients = ground_values_and_gradients(ground, points)
    projection_values, projection_gradients, _ = project_values_against_ground(
        projection_values,
        projection_gradients,
        ground_values,
        ground_gradients,
        density,
    )
    source_projected = projection_values[:, 0]
    dressed_values = projection_values[:, 1:]
    dressed_gradients = projection_gradients[:, 1:]
    pot_shift = potential_shift(ground, points)
    overlap, hamiltonian = weak_matrices(
        dressed_values,
        dressed_gradients,
        pot_shift,
        density,
    )
    weights = 1 / density / points.shape[0]
    source_vector = jnp.einsum(
        "n,ni,n->i",
        weights,
        dressed_values,
        source_projected,
    )[:, None]
    overlap_norm, roots, source_weights, _, _ = _projected_source_spectrum_core(
        overlap,
        hamiltonian,
        source_vector,
    )
    overlap_evals = jnp.linalg.eigvalsh((overlap_norm + overlap_norm.T) / 2)
    min_eval = jnp.maximum(jnp.min(overlap_evals), 1e-12)
    max_eval = jnp.maximum(jnp.max(overlap_evals), min_eval)
    condition = max_eval / min_eval
    condition_penalty = (
        jnp.maximum(
            jnp.log(condition) - jnp.log(float(max_condition)),
            0.0,
        )
        ** 2
    )
    reference_coefficients = baseline_coefficients[:, bright_indices]
    fixed_values = dressed_values @ reference_coefficients
    fixed_gradients = jnp.einsum(
        "naei,ab->nbei",
        dressed_gradients,
        reference_coefficients,
    )
    fixed_flat_gradients = jnp.reshape(
        fixed_gradients,
        (*fixed_gradients.shape[:2], -1),
    )
    fixed_overlap = jnp.einsum("n,nb,nb->b", weights, fixed_values, fixed_values)
    fixed_hamiltonian_integrand = 0.5 * jnp.sum(fixed_flat_gradients**2, axis=2)
    fixed_hamiltonian_integrand = (
        fixed_hamiltonian_integrand + pot_shift[:, None] * fixed_values**2
    )
    fixed_hamiltonian = jnp.einsum("n,nb->b", weights, fixed_hamiltonian_integrand)
    tracked_roots = fixed_hamiltonian / jnp.maximum(
        fixed_overlap,
        jnp.asarray(1e-30, dtype=points.dtype),
    )
    fixed_source_amplitudes = jnp.einsum(
        "n,nb,n->b",
        weights,
        fixed_values,
        source_projected,
    )
    tracked_source_weights = fixed_source_amplitudes**2 / jnp.maximum(
        fixed_overlap,
        jnp.asarray(1e-30, dtype=points.dtype),
    )
    matrices, matrix_gradients = cas_dressing_matrices_and_gradients(
        params,
        ground,
        points,
        radial_scales=jnp.asarray(OFFICIAL_DRESSING_RADIAL_SCALES, dtype=points.dtype),
        pair_scales=jnp.asarray(OFFICIAL_DRESSING_PAIR_SCALES, dtype=points.dtype),
    )
    identity = jnp.eye(head_count, dtype=matrices.dtype)
    dressing_norm = jnp.mean(
        jnp.sum((matrices - identity[None, :, :]) ** 2, axis=(1, 2))
    )
    dressing_grad_norm = jnp.mean(jnp.sum(matrix_gradients**2, axis=(1, 2, 3, 4)))
    rho = jnp.asarray(root_weights, dtype=points.dtype)
    rho = rho / jnp.maximum(jnp.sum(rho), jnp.asarray(1e-30, dtype=points.dtype))
    energy_loss = jnp.sum(rho * tracked_roots)
    visibility_loss = -jnp.sum(
        rho * jnp.log(tracked_source_weights + jnp.asarray(1e-30, dtype=points.dtype))
    )
    tracking_loss = jnp.asarray(0.0, dtype=points.dtype)
    regularizer = dressing_norm + (float(gradient_regularizer_length) ** 2) * (
        dressing_grad_norm
    )
    source_weight_sum = jnp.sum(tracked_source_weights)
    source_weighted_root = jnp.sum(
        tracked_roots * tracked_source_weights
    ) / jnp.maximum(
        source_weight_sum,
        jnp.asarray(1e-30, dtype=points.dtype),
    )
    loss = (
        float(energy_weight) * energy_loss
        + float(visibility_weight) * visibility_loss
        + float(condition_weight) * condition_penalty
        + float(regularizer_weight) * regularizer
    )
    return loss, {
        "loss": loss,
        "energy_loss": energy_loss,
        "visibility_loss": visibility_loss,
        "tracking_loss": tracking_loss,
        "regularizer": regularizer,
        "dressing_norm": dressing_norm,
        "dressing_grad_norm": dressing_grad_norm,
        "condition_penalty": condition_penalty,
        "overlap_penalty": tracking_loss,
        "root0": tracked_roots[0],
        "source_weighted_root": source_weighted_root,
        "source_weight_sum": source_weight_sum,
        "source_weight0": tracked_source_weights[0],
        "condition": condition,
        "matched_overlap0": jnp.asarray(1.0, dtype=points.dtype),
        "matched_index0": bright_indices[0],
        "roots": roots,
        "source_weights": source_weights,
        "tracked_roots": tracked_roots,
        "tracked_source_weights": tracked_source_weights,
        "matched_indices": bright_indices,
        "matched_overlaps": jnp.ones_like(tracked_roots),
    }


def _scalar_stats(stats: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in stats.items():
        arr = np.asarray(value)
        if arr.shape == ():
            out[key] = float(arr)
    return out


def _cas_dressing_projected_objective_stats(
    stats: Mapping[str, Any],
    *,
    root_weights_np: np.ndarray,
    root_count: int,
) -> dict[str, float]:
    """Return scalar source-visible objective diagnostics for projected stats."""
    roots = np.asarray(stats["roots"], dtype=np.float64)
    source_weights = np.maximum(
        np.asarray(stats["source_weights"], dtype=np.float64),
        0.0,
    )
    count = max(1, min(int(root_count), roots.shape[0], source_weights.shape[0]))
    weights = np.asarray(root_weights_np, dtype=np.float64)[:count]
    if weights.size != count or not np.any(weights > 0.0):
        weights = np.ones((count,), dtype=np.float64) / float(count)
    else:
        weights = weights / np.sum(weights)
    visible_weights = source_weights[:count]
    source_weight_sum = float(np.sum(visible_weights))
    source_weighted_root = float(
        np.sum(roots[:count] * visible_weights) / max(source_weight_sum, 1e-30)
    )
    return {
        "root0": float(roots[0]),
        "weighted_root": float(np.dot(roots[:count], weights)),
        "source_weighted_root": source_weighted_root,
        "source_weight_sum": source_weight_sum,
        "source_weight0": float(source_weights[0]),
        "condition": float(np.asarray(stats["condition"])),
    }


def _cas_dressing_tracked_objective_stats(
    stats: Mapping[str, Any],
    *,
    baseline_vectors_np: np.ndarray,
    bright_indices_np: np.ndarray,
    root_weights_np: np.ndarray,
) -> dict[str, float]:
    """Return scalar diagnostics for baseline-tracked bright roots."""
    roots = np.asarray(stats["roots"], dtype=np.float64)
    source_weights = np.maximum(
        np.asarray(stats["source_weights"], dtype=np.float64),
        0.0,
    )
    vectors = np.asarray(stats["ritz_vectors"], dtype=np.float64)
    bright_indices = np.asarray(bright_indices_np, dtype=np.int64)
    if bright_indices.size == 0:
        bright_indices = np.asarray([int(np.argmax(source_weights))], dtype=np.int64)
    reference = np.asarray(baseline_vectors_np, dtype=np.float64)[:, bright_indices]
    overlaps = (reference.T @ vectors) ** 2
    matched_indices = np.argmax(overlaps, axis=1)
    matched_overlaps = overlaps[np.arange(bright_indices.size), matched_indices]
    tracked_roots = roots[matched_indices]
    tracked_source_weights = source_weights[matched_indices]
    weights = np.asarray(root_weights_np, dtype=np.float64)
    if weights.size != bright_indices.size or not np.any(weights > 0.0):
        weights = np.ones((bright_indices.size,), dtype=np.float64)
    weights = weights / np.sum(weights)
    source_weight_sum = float(np.sum(tracked_source_weights))
    source_weighted_root = float(
        np.sum(tracked_roots * tracked_source_weights) / max(source_weight_sum, 1e-30)
    )
    return {
        "root0": float(tracked_roots[0]),
        "weighted_root": float(np.dot(tracked_roots, weights)),
        "source_weighted_root": source_weighted_root,
        "source_weight_sum": source_weight_sum,
        "source_weight0": float(tracked_source_weights[0]),
        "condition": float(np.asarray(stats["condition"])),
        "matched_overlap0": float(matched_overlaps[0]),
        "matched_index0": float(matched_indices[0]),
    }


def fine_tune_cas_dressed_teacher_block(  # noqa: C901
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    teacher_values_np: np.ndarray,
    teacher_gradients_np: np.ndarray,
    *,
    head_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_roots: int,
    energy_weight: float,
    visibility_weight: float,
    condition_weight: float,
    overlap_weight: float,
    regularizer_weight: float,
    gradient_regularizer_length: float,
    max_condition: float,
    validation_fraction: float,
    bright_threshold: float,
    validation_blocks: int,
    acceptance_sigma: float,
    seed: int,
    log_every: int,
) -> tuple[Params, dict[str, Any]]:
    """Fine tune the CAS-explicit matrix dressing ``A(R; theta)``.

    Returns:
        Updated dressing parameters and scalar diagnostics.

    Raises:
        ValueError: If samples, density, or teacher arrays are invalid.
    """
    points_np = np.asarray(points_np)
    density_np = np.asarray(density_np, dtype=np.float64)
    teacher_values_np = np.asarray(teacher_values_np, dtype=np.float64)
    teacher_gradients_np = np.asarray(teacher_gradients_np, dtype=np.float64)
    head_count = int(head_count)
    if teacher_values_np.shape != (points_np.shape[0], head_count):
        msg = "CAS-dressed teacher values must have shape (samples, heads)"
        raise ValueError(msg)
    expected_grad_shape = (points_np.shape[0], head_count, *points_np.shape[1:])
    if teacher_gradients_np.shape != expected_grad_shape:
        msg = (
            "CAS-dressed teacher gradients must have shape "
            "(samples, heads, electrons, 3)"
        )
        raise ValueError(msg)
    if density_np.shape != (points_np.shape[0],):
        msg = "CAS-dressed teacher density must have shape (samples,)"
        raise ValueError(msg)
    if (
        np.any(~np.isfinite(teacher_values_np))
        or np.any(~np.isfinite(teacher_gradients_np))
        or np.any(~np.isfinite(density_np))
    ):
        msg = "CAS-dressed teacher inputs must be finite"
        raise ValueError(msg)
    if np.any(density_np <= 0):
        msg = "CAS-dressed teacher density must be positive"
        raise ValueError(msg)
    if not (
        np.isfinite(validation_fraction) and 0.0 <= float(validation_fraction) < 1.0
    ):
        msg = "CAS-dressed teacher validation_fraction must be in [0, 1)"
        raise ValueError(msg)
    if not (np.isfinite(bright_threshold) and 0.0 <= float(bright_threshold) <= 1.0):
        msg = "CAS-dressed teacher bright_threshold must be in [0, 1]"
        raise ValueError(msg)
    if int(validation_blocks) < 1:
        msg = "CAS-dressed teacher validation_blocks must be positive"
        raise ValueError(msg)
    if not (np.isfinite(acceptance_sigma) and float(acceptance_sigma) >= 0.0):
        msg = "CAS-dressed teacher acceptance_sigma must be nonnegative"
        raise ValueError(msg)

    points = jnp.asarray(points_np)
    density = jnp.asarray(density_np)
    teacher_values = jnp.asarray(teacher_values_np)
    teacher_gradients = jnp.asarray(teacher_gradients_np)
    root_count = max(1, min(int(n_roots), head_count))

    @jax.jit
    def spectrum_step(eval_params: Params):
        return cas_dressed_teacher_projected_stats(
            eval_params,
            ground,
            points,
            density,
            teacher_values,
            teacher_gradients,
            head_count=head_count,
            radial_scales=jnp.asarray(
                OFFICIAL_DRESSING_RADIAL_SCALES,
                dtype=points.dtype,
            ),
            pair_scales=jnp.asarray(
                OFFICIAL_DRESSING_PAIR_SCALES,
                dtype=points.dtype,
            ),
            max_condition=max_condition,
        )

    baseline_stats_jax = spectrum_step(params)
    baseline_source_weights = np.maximum(
        np.asarray(baseline_stats_jax["source_weights"], dtype=np.float64),
        0.0,
    )
    baseline_vectors_np = np.asarray(
        baseline_stats_jax["ritz_vectors"],
        dtype=np.float64,
    )
    baseline_coefficients_np = np.asarray(
        baseline_stats_jax["ritz_coefficients"],
        dtype=np.float64,
    )
    bright_cutoff = float(bright_threshold) * max(
        float(np.max(baseline_source_weights[:root_count])),
        0.0,
    )
    bright_indices_np = np.flatnonzero(
        baseline_source_weights[:root_count] >= bright_cutoff
    ).astype(np.int64)
    if bright_indices_np.size == 0:
        bright_indices_np = np.asarray(
            [int(np.argmax(baseline_source_weights[:root_count]))],
            dtype=np.int64,
        )
    root_weights_np = baseline_source_weights[bright_indices_np]
    if not np.any(root_weights_np > 0.0):
        root_weights_np = np.ones((bright_indices_np.size,), dtype=np.float64)
    root_weights_np = root_weights_np / np.sum(root_weights_np)
    baseline_tracked_stats = _cas_dressing_tracked_objective_stats(
        baseline_stats_jax,
        baseline_vectors_np=baseline_vectors_np,
        bright_indices_np=bright_indices_np,
        root_weights_np=root_weights_np,
    )
    baseline_source_weight_sum = baseline_tracked_stats["source_weight_sum"]
    baseline_objective = baseline_tracked_stats["source_weighted_root"]
    root_weights = jnp.asarray(root_weights_np)
    baseline_coefficients = jnp.asarray(baseline_coefficients_np)
    bright_indices = jnp.asarray(bright_indices_np)

    split_rng = np.random.default_rng(int(seed) + 17)
    split_order = split_rng.permutation(points_np.shape[0])
    validation_count = round(float(validation_fraction) * points_np.shape[0])
    if points_np.shape[0] > 1 and validation_fraction > 0.0:
        validation_count = min(max(1, validation_count), points_np.shape[0] - 1)
    else:
        validation_count = 0
    val_indices = split_order[:validation_count]
    train_indices = split_order[validation_count:]
    if train_indices.size == 0:
        train_indices = split_order
        val_indices = split_order[:0]
    if val_indices.size == 0:
        val_indices = train_indices
    train_points_np = points_np[train_indices]
    train_density_np = density_np[train_indices]
    train_teacher_values_np = teacher_values_np[train_indices]
    train_teacher_gradients_np = teacher_gradients_np[train_indices]
    val_points = jnp.asarray(points_np[val_indices])
    val_density = jnp.asarray(density_np[val_indices])
    val_teacher_values = jnp.asarray(teacher_values_np[val_indices])
    val_teacher_gradients = jnp.asarray(teacher_gradients_np[val_indices])

    @jax.jit
    def validation_spectrum_step(eval_params: Params):
        return cas_dressed_teacher_projected_stats(
            eval_params,
            ground,
            val_points,
            val_density,
            val_teacher_values,
            val_teacher_gradients,
            head_count=head_count,
            radial_scales=jnp.asarray(
                OFFICIAL_DRESSING_RADIAL_SCALES,
                dtype=points.dtype,
            ),
            pair_scales=jnp.asarray(
                OFFICIAL_DRESSING_PAIR_SCALES,
                dtype=points.dtype,
            ),
            max_condition=max_condition,
        )

    baseline_validation_raw_stats = validation_spectrum_step(params)
    baseline_validation_vectors_np = np.asarray(
        baseline_validation_raw_stats["ritz_vectors"],
        dtype=np.float64,
    )
    baseline_validation_stats = _cas_dressing_tracked_objective_stats(
        baseline_validation_raw_stats,
        baseline_vectors_np=baseline_validation_vectors_np,
        bright_indices_np=bright_indices_np,
        root_weights_np=root_weights_np,
    )
    baseline_params_for_validation = jax.tree_util.tree_map(lambda x: x.copy(), params)
    validation_local_blocks = [
        block
        for block in np.array_split(
            np.arange(val_indices.size, dtype=np.int64),
            min(int(validation_blocks), max(1, val_indices.size)),
        )
        if block.size
    ]

    baseline_block_objectives = []
    baseline_block_source_sums = []
    baseline_block_vectors = []
    for block in validation_local_blocks:
        baseline_block_stats = cas_dressed_teacher_projected_stats(
            baseline_params_for_validation,
            ground,
            jnp.asarray(points_np[val_indices][block]),
            jnp.asarray(density_np[val_indices][block]),
            jnp.asarray(teacher_values_np[val_indices][block]),
            jnp.asarray(teacher_gradients_np[val_indices][block]),
            head_count=head_count,
            radial_scales=jnp.asarray(
                OFFICIAL_DRESSING_RADIAL_SCALES,
                dtype=points.dtype,
            ),
            pair_scales=jnp.asarray(
                OFFICIAL_DRESSING_PAIR_SCALES,
                dtype=points.dtype,
            ),
            max_condition=max_condition,
        )
        block_baseline_vectors = np.asarray(
            baseline_block_stats["ritz_vectors"],
            dtype=np.float64,
        )
        baseline_block_vectors.append(block_baseline_vectors)
        block_baseline_objective_stats = _cas_dressing_tracked_objective_stats(
            baseline_block_stats,
            baseline_vectors_np=block_baseline_vectors,
            bright_indices_np=bright_indices_np,
            root_weights_np=root_weights_np,
        )
        baseline_block_objectives.append(
            block_baseline_objective_stats["source_weighted_root"]
        )
        baseline_block_source_sums.append(
            block_baseline_objective_stats["source_weight_sum"]
        )
    baseline_validation_block_objectives = np.asarray(
        baseline_block_objectives,
        dtype=np.float64,
    )
    baseline_validation_block_source_sums = np.asarray(
        baseline_block_source_sums,
        dtype=np.float64,
    )

    def validation_block_objectives(
        eval_params: Params,
    ) -> tuple[np.ndarray, np.ndarray]:
        objectives = []
        source_sums = []
        for block_idx, block in enumerate(validation_local_blocks):
            block_stats = cas_dressed_teacher_projected_stats(
                eval_params,
                ground,
                jnp.asarray(points_np[val_indices][block]),
                jnp.asarray(density_np[val_indices][block]),
                jnp.asarray(teacher_values_np[val_indices][block]),
                jnp.asarray(teacher_gradients_np[val_indices][block]),
                head_count=head_count,
                radial_scales=jnp.asarray(
                    OFFICIAL_DRESSING_RADIAL_SCALES,
                    dtype=points.dtype,
                ),
                pair_scales=jnp.asarray(
                    OFFICIAL_DRESSING_PAIR_SCALES,
                    dtype=points.dtype,
                ),
                max_condition=max_condition,
            )
            block_objective_stats = _cas_dressing_tracked_objective_stats(
                block_stats,
                baseline_vectors_np=baseline_block_vectors[block_idx],
                bright_indices_np=bright_indices_np,
                root_weights_np=root_weights_np,
            )
            objectives.append(block_objective_stats["source_weighted_root"])
            source_sums.append(block_objective_stats["source_weight_sum"])
        return (
            np.asarray(objectives, dtype=np.float64),
            np.asarray(source_sums, dtype=np.float64),
        )

    def validation_matrix_payload(eval_params: Params) -> dict[str, Any]:
        raw_stats = validation_spectrum_step(eval_params)
        block_overlaps = []
        block_hamiltonians = []
        block_sources = []
        point_blocks = []
        density_blocks = []
        for block in validation_local_blocks:
            block_points = points_np[val_indices][block]
            block_density = density_np[val_indices][block]
            block_stats = cas_dressed_teacher_projected_stats(
                eval_params,
                ground,
                jnp.asarray(block_points),
                jnp.asarray(block_density),
                jnp.asarray(teacher_values_np[val_indices][block]),
                jnp.asarray(teacher_gradients_np[val_indices][block]),
                head_count=head_count,
                radial_scales=jnp.asarray(
                    OFFICIAL_DRESSING_RADIAL_SCALES,
                    dtype=points.dtype,
                ),
                pair_scales=jnp.asarray(
                    OFFICIAL_DRESSING_PAIR_SCALES,
                    dtype=points.dtype,
                ),
                max_condition=max_condition,
            )
            block_overlaps.append(np.asarray(block_stats["overlap"], dtype=np.float64))
            block_hamiltonians.append(
                np.asarray(block_stats["hamiltonian"], dtype=np.float64)
            )
            block_sources.append(np.asarray(block_stats["source"], dtype=np.float64))
            point_blocks.append(np.asarray(block_points))
            density_blocks.append(np.asarray(block_density, dtype=np.float64))
        return {
            "certified_overlap": np.asarray(raw_stats["overlap"], dtype=np.float64),
            "certified_hamiltonian": np.asarray(
                raw_stats["hamiltonian"],
                dtype=np.float64,
            ),
            "certified_source": np.asarray(raw_stats["source"], dtype=np.float64),
            "certified_block_overlaps": np.asarray(block_overlaps, dtype=np.float64),
            "certified_block_hamiltonians": np.asarray(
                block_hamiltonians,
                dtype=np.float64,
            ),
            "certified_block_sources": np.asarray(block_sources, dtype=np.float64),
            "certified_block_counts": np.asarray(
                [block.shape[0] for block in point_blocks],
                dtype=np.float64,
            ),
            "certified_point_blocks": point_blocks,
            "certified_density_blocks": density_blocks,
            "certified_points": np.asarray(points_np[val_indices]),
            "certified_density": np.asarray(density_np[val_indices], dtype=np.float64),
        }

    baseline_validation_se = (
        float(
            np.std(baseline_validation_block_objectives, ddof=1)
            / np.sqrt(baseline_validation_block_objectives.size)
        )
        if baseline_validation_block_objectives.size > 1
        else 0.0
    )

    @jax.jit
    def eval_step(eval_params: Params):
        return cas_dressed_teacher_fixed_bright_loss(
            eval_params,
            ground,
            points,
            density,
            teacher_values,
            teacher_gradients,
            baseline_coefficients,
            bright_indices,
            root_weights,
            head_count=head_count,
            energy_weight=energy_weight,
            visibility_weight=visibility_weight,
            condition_weight=condition_weight,
            overlap_weight=overlap_weight,
            regularizer_weight=regularizer_weight,
            gradient_regularizer_length=gradient_regularizer_length,
            max_condition=max_condition,
        )

    optimizer = optax.adam(float(learning_rate))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(
        current_params: Params,
        current_state: optax.OptState,
        batch_points: jax.Array,
        batch_density: jax.Array,
        batch_teacher_values: jax.Array,
        batch_teacher_gradients: jax.Array,
    ) -> tuple[Params, optax.OptState, dict[str, jax.Array]]:
        def loss_fn(local_params: Params):
            return cas_dressed_teacher_fixed_bright_loss(
                local_params,
                ground,
                batch_points,
                batch_density,
                batch_teacher_values,
                batch_teacher_gradients,
                baseline_coefficients,
                bright_indices,
                root_weights,
                head_count=head_count,
                energy_weight=energy_weight,
                visibility_weight=visibility_weight,
                condition_weight=condition_weight,
                overlap_weight=overlap_weight,
                regularizer_weight=regularizer_weight,
                gradient_regularizer_length=gradient_regularizer_length,
                max_condition=max_condition,
            )

        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
        updates, next_state = optimizer.update(grads, current_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_state, stats | {"loss": loss}

    _, initial_stats_jax = eval_step(params)
    final_stats = _scalar_stats(initial_stats_jax) | {
        "baseline_root0": baseline_tracked_stats["root0"],
        "baseline_source_weighted_root": baseline_objective,
        "baseline_source_weight_sum": baseline_source_weight_sum,
        "validation_root0": baseline_validation_stats["root0"],
        "validation_source_weighted_root": (
            baseline_validation_stats["source_weighted_root"]
        ),
        "validation_source_weight_sum": (
            baseline_validation_stats["source_weight_sum"]
        ),
        "validation_condition": baseline_validation_stats["condition"],
        "baseline_validation_root0": baseline_validation_stats["root0"],
        "baseline_validation_source_weighted_root": (
            baseline_validation_stats["source_weighted_root"]
        ),
        "baseline_validation_source_weight_sum": (
            baseline_validation_stats["source_weight_sum"]
        ),
        "baseline_validation_condition": baseline_validation_stats["condition"],
        "validation_se": baseline_validation_se,
        "validation_candidate_se": baseline_validation_se,
        "baseline_validation_se": baseline_validation_se,
        "validation_pair_delta": 0.0,
        "validation_pair_se": 0.0,
        "validation_source_pair_delta": 0.0,
        "validation_source_pair_se": 0.0,
        "last_validation_pair_delta": 0.0,
        "last_validation_pair_se": 0.0,
        "last_validation_source_pair_delta": 0.0,
        "last_validation_source_pair_se": 0.0,
        "accepted": 0.0,
        "accepted_epoch": float("nan"),
        "anchor_loss": float("nan"),
        "anchor_coeff_norm": float("nan"),
    }
    if int(epochs) < 1:
        print(
            "response_cas_dressed_teacher_ready "
            "neural_training=skipped "
            f"root0={final_stats['root0']:.10f} "
            f"source_objective={final_stats['source_weighted_root']:.10f} "
            f"condition={final_stats['condition']:.3e}"
        )
        return params, final_stats

    rng = np.random.default_rng(int(seed))
    initial_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
    best_params = initial_params
    best_validation_stats = baseline_validation_stats
    best_validation_objective = float(baseline_validation_stats["source_weighted_root"])
    best_validation_se = baseline_validation_se
    best_validation_candidate_se = baseline_validation_se
    best_validation_pair_delta = 0.0
    best_validation_source_pair_delta = 0.0
    best_validation_source_pair_se = 0.0
    last_validation_pair_delta = 0.0
    last_validation_pair_se = 0.0
    last_validation_source_pair_delta = 0.0
    last_validation_source_pair_se = 0.0
    baseline_validation_source_sum = float(
        baseline_validation_stats["source_weight_sum"]
    )
    accepted_epoch = float("nan")
    for epoch in range(int(epochs)):
        order = rng.permutation(train_points_np.shape[0])
        for batch in make_batches(train_points_np.shape[0], int(batch_size)):
            batch_indices = order[batch]
            params, opt_state, stats = train_step(
                params,
                opt_state,
                jnp.asarray(train_points_np[batch_indices]),
                jnp.asarray(train_density_np[batch_indices]),
                jnp.asarray(train_teacher_values_np[batch_indices]),
                jnp.asarray(train_teacher_gradients_np[batch_indices]),
            )
            final_stats = _scalar_stats(stats) | {
                "baseline_root0": baseline_tracked_stats["root0"],
                "baseline_source_weighted_root": baseline_objective,
                "baseline_source_weight_sum": baseline_source_weight_sum,
                "validation_root0": best_validation_stats["root0"],
                "validation_source_weighted_root": (
                    best_validation_stats["source_weighted_root"]
                ),
                "validation_source_weight_sum": (
                    best_validation_stats["source_weight_sum"]
                ),
                "validation_condition": best_validation_stats["condition"],
                "baseline_validation_root0": baseline_validation_stats["root0"],
                "baseline_validation_source_weighted_root": (
                    baseline_validation_stats["source_weighted_root"]
                ),
                "baseline_validation_source_weight_sum": (
                    baseline_validation_stats["source_weight_sum"]
                ),
                "baseline_validation_condition": (
                    baseline_validation_stats["condition"]
                ),
                "validation_se": best_validation_se,
                "validation_candidate_se": best_validation_candidate_se,
                "baseline_validation_se": baseline_validation_se,
                "validation_pair_delta": best_validation_pair_delta,
                "validation_pair_se": best_validation_se,
                "validation_source_pair_delta": best_validation_source_pair_delta,
                "validation_source_pair_se": best_validation_source_pair_se,
                "last_validation_pair_delta": last_validation_pair_delta,
                "last_validation_pair_se": last_validation_pair_se,
                "last_validation_source_pair_delta": last_validation_source_pair_delta,
                "last_validation_source_pair_se": last_validation_source_pair_se,
                "accepted": float(np.isfinite(accepted_epoch)),
                "accepted_epoch": accepted_epoch,
                "anchor_loss": float("nan"),
                "anchor_coeff_norm": float("nan"),
            }
        epoch_validation_stats = _cas_dressing_tracked_objective_stats(
            validation_spectrum_step(params),
            baseline_vectors_np=baseline_validation_vectors_np,
            bright_indices_np=bright_indices_np,
            root_weights_np=root_weights_np,
        )
        validation_objective = float(epoch_validation_stats["source_weighted_root"])
        validation_source_sum = float(epoch_validation_stats["source_weight_sum"])
        validation_condition = float(epoch_validation_stats["condition"])
        validation_block_values, validation_block_source_sums = (
            validation_block_objectives(params)
        )
        validation_block_deltas = (
            validation_block_values - baseline_validation_block_objectives
        )
        validation_pair_delta = (
            float(np.mean(validation_block_deltas))
            if validation_block_deltas.size
            else validation_objective
            - float(baseline_validation_stats["source_weighted_root"])
        )
        validation_se = (
            float(
                np.std(validation_block_deltas, ddof=1)
                / np.sqrt(validation_block_deltas.size)
            )
            if validation_block_deltas.size > 1
            else 0.0
        )
        validation_candidate_se = (
            float(
                np.std(validation_block_values, ddof=1)
                / np.sqrt(validation_block_values.size)
            )
            if validation_block_values.size > 1
            else 0.0
        )
        validation_source_block_deltas = (
            validation_block_source_sums - baseline_validation_block_source_sums
        )
        validation_source_pair_delta = (
            float(np.mean(validation_source_block_deltas))
            if validation_source_block_deltas.size
            else validation_source_sum - baseline_validation_source_sum
        )
        validation_source_pair_se = (
            float(
                np.std(validation_source_block_deltas, ddof=1)
                / np.sqrt(validation_source_block_deltas.size)
            )
            if validation_source_block_deltas.size > 1
            else 0.0
        )
        last_validation_pair_delta = validation_pair_delta
        last_validation_pair_se = validation_se
        last_validation_source_pair_delta = validation_source_pair_delta
        last_validation_source_pair_se = validation_source_pair_se
        validation_improvement = validation_pair_delta
        validation_significance = float(acceptance_sigma) * validation_se
        source_preservation_significance = (
            float(acceptance_sigma) * validation_source_pair_se
        )
        validation_passed = (
            np.isfinite(validation_objective)
            and validation_objective < best_validation_objective
            and validation_improvement < -validation_significance
            and validation_candidate_se <= baseline_validation_se + validation_se
            and validation_source_sum >= 0.5 * baseline_validation_source_sum
            and validation_source_pair_delta >= -source_preservation_significance
            and validation_condition <= max_condition
        )
        if validation_passed:
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), params)
            best_validation_stats = epoch_validation_stats
            best_validation_objective = validation_objective
            best_validation_se = validation_se
            best_validation_candidate_se = validation_candidate_se
            best_validation_pair_delta = validation_pair_delta
            best_validation_source_pair_delta = validation_source_pair_delta
            best_validation_source_pair_se = validation_source_pair_se
            accepted_epoch = float(epoch)
        if log_every > 0 and (epoch % int(log_every) == 0 or epoch == epochs - 1):
            print(
                "response_cas_dressed_teacher_finetune "
                f"epoch={epoch:05d} "
                f"loss={final_stats['loss']:.8e} "
                f"root0={final_stats['root0']:.10f} "
                f"source_objective={final_stats['source_weighted_root']:.10f} "
                f"validation_objective={validation_objective:.10f} "
                f"validation_pair_delta={validation_pair_delta:.3e} "
                f"validation_se={validation_se:.3e} "
                f"candidate_se={validation_candidate_se:.3e} "
                f"best_validation_objective={best_validation_objective:.10f} "
                f"baseline_root0={baseline_tracked_stats['root0']:.10f} "
                f"baseline_source_objective={baseline_objective:.10f} "
                f"visibility={final_stats['visibility_loss']:.3e} "
                f"regularizer={final_stats['regularizer']:.3e} "
                f"condition={final_stats['condition']:.3e}"
            )
    accepted = np.isfinite(accepted_epoch)
    selected_params = best_params if accepted else initial_params
    _, final_eval_stats_jax = eval_step(selected_params)
    certified_payload = validation_matrix_payload(selected_params)
    final_stats = (
        _scalar_stats(final_eval_stats_jax)
        | {
            "baseline_root0": baseline_tracked_stats["root0"],
            "baseline_source_weighted_root": baseline_objective,
            "baseline_source_weight_sum": baseline_source_weight_sum,
            "validation_root0": best_validation_stats["root0"],
            "validation_source_weighted_root": (
                best_validation_stats["source_weighted_root"]
            ),
            "validation_source_weight_sum": best_validation_stats["source_weight_sum"],
            "validation_condition": best_validation_stats["condition"],
            "baseline_validation_root0": baseline_validation_stats["root0"],
            "baseline_validation_source_weighted_root": (
                baseline_validation_stats["source_weighted_root"]
            ),
            "baseline_validation_source_weight_sum": (
                baseline_validation_stats["source_weight_sum"]
            ),
            "baseline_validation_condition": baseline_validation_stats["condition"],
            "validation_se": best_validation_se,
            "validation_candidate_se": best_validation_candidate_se,
            "baseline_validation_se": baseline_validation_se,
            "validation_pair_delta": best_validation_pair_delta,
            "validation_pair_se": best_validation_se,
            "validation_source_pair_delta": best_validation_source_pair_delta,
            "validation_source_pair_se": best_validation_source_pair_se,
            "last_validation_pair_delta": last_validation_pair_delta,
            "last_validation_pair_se": last_validation_pair_se,
            "last_validation_source_pair_delta": last_validation_source_pair_delta,
            "last_validation_source_pair_se": last_validation_source_pair_se,
            "accepted": float(accepted),
            "accepted_epoch": accepted_epoch,
            "anchor_loss": float("nan"),
            "anchor_coeff_norm": float("nan"),
        }
        | certified_payload
    )
    print(
        "response_cas_dressed_teacher_select "
        f"accepted={bool(accepted)} "
        f"root0={final_stats['root0']:.10f} "
        f"source_objective={final_stats['source_weighted_root']:.10f} "
        f"validation_objective={final_stats['validation_source_weighted_root']:.10f} "
        f"baseline_root0={baseline_tracked_stats['root0']:.10f} "
        f"baseline_source_objective={baseline_objective:.10f} "
        "baseline_validation_objective="
        f"{baseline_validation_stats['source_weighted_root']:.10f}"
    )
    return selected_params, final_stats


def response_values_and_gradients(
    params: Params, ground: FermiNetGround, points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    values = response_values(params, ground, points)
    gradients = jax.vmap(jax.jacrev(response_value_single, argnums=2), (None, None, 0))(
        params, ground, points
    )
    return values, gradients


def normalized_subspace(
    overlap: jax.Array, hamiltonian: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    diag = jnp.sqrt(jnp.maximum(jnp.diag(overlap), 1e-12))
    overlap_norm = overlap / diag[:, None] / diag[None, :]
    hamiltonian_norm = hamiltonian / diag[:, None] / diag[None, :]
    evals, evecs = jnp.linalg.eigh((overlap_norm + overlap_norm.T) / 2)
    evals = jnp.maximum(evals, 1e-8)
    whitening = evecs / jnp.sqrt(evals)
    hamiltonian_tilde = whitening.T @ ((hamiltonian_norm + hamiltonian_norm.T) / 2)
    hamiltonian_tilde = hamiltonian_tilde @ whitening
    roots = jnp.linalg.eigvalsh((hamiltonian_tilde + hamiltonian_tilde.T) / 2)
    return overlap_norm, hamiltonian_norm, roots


def normalized_source_spectrum(
    overlap: jax.Array, hamiltonian: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Whiten a response subspace and return source-channel Ritz weights.

    Returns:
        Normalized overlap, Ritz roots, and weights for the first basis vector
        treated as the source vector.
    """
    diag = jnp.sqrt(jnp.maximum(jnp.diag(overlap), 1e-12))
    overlap_norm = overlap / diag[:, None] / diag[None, :]
    hamiltonian_norm = hamiltonian / diag[:, None] / diag[None, :]
    overlap_sym = (overlap_norm + overlap_norm.T) / 2
    hamiltonian_sym = (hamiltonian_norm + hamiltonian_norm.T) / 2
    evals, evecs = jnp.linalg.eigh(overlap_sym)
    evals = jnp.maximum(evals, 1e-8)
    whitening = evecs / jnp.sqrt(evals)
    hamiltonian_tilde = whitening.T @ hamiltonian_sym @ whitening
    roots, vectors = jnp.linalg.eigh((hamiltonian_tilde + hamiltonian_tilde.T) / 2)
    source_norm = overlap[:, 0] / diag
    source_tilde = whitening.T @ source_norm
    amplitudes = vectors.T @ source_tilde
    weights = amplitudes**2
    return overlap_norm, roots, weights


def _projected_source_spectrum_core(
    overlap: jax.Array,
    hamiltonian: jax.Array,
    source: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Whiten a head-only response subspace and return Ritz data.

    Returns:
        Normalized overlap, Ritz roots, source weights, whitened Ritz vectors,
        and generalized eigenvector coefficients in the original head basis.
    """
    diag = jnp.sqrt(jnp.maximum(jnp.diag(overlap), 1e-12))
    overlap_norm = overlap / diag[:, None] / diag[None, :]
    hamiltonian_norm = hamiltonian / diag[:, None] / diag[None, :]
    overlap_sym = (overlap_norm + overlap_norm.T) / 2
    hamiltonian_sym = (hamiltonian_norm + hamiltonian_norm.T) / 2
    evals, evecs = jnp.linalg.eigh(overlap_sym)
    evals = jnp.maximum(evals, 1e-8)
    whitening = evecs / jnp.sqrt(evals)
    hamiltonian_tilde = whitening.T @ hamiltonian_sym @ whitening
    roots, vectors = jnp.linalg.eigh((hamiltonian_tilde + hamiltonian_tilde.T) / 2)
    source_norm = source[:, 0] / diag
    source_tilde = whitening.T @ source_norm
    amplitudes = vectors.T @ source_tilde
    weights = amplitudes**2
    coefficients = (whitening @ vectors) / diag[:, None]
    return overlap_norm, roots, weights, vectors, coefficients


def normalized_projected_source_spectrum(
    overlap: jax.Array,
    hamiltonian: jax.Array,
    source: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Whiten a head-only response subspace and return source Ritz weights.

    Returns:
        Normalized overlap, Ritz roots, source weights, and whitened Ritz
        vectors computed from the separate projected source vector.
    """
    overlap_norm, roots, weights, vectors, _ = _projected_source_spectrum_core(
        overlap,
        hamiltonian,
        source,
    )
    return overlap_norm, roots, weights, vectors


def moment_diagnostics(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    spectrum: ProjectedSpectrum,
    *,
    overlap_cutoff: float,
    source_in_basis: bool = True,
) -> MomentDiagnostics:
    """Check source-indexed zeroth/first spectral moments.

    When the projected source is the first basis function, the spectral
    measure can be compared directly to the source-basis matrix element.
    In head-only retained mode, only the projected spectral moments are
    available from ``S,K,p``.

    Returns:
        Moment errors and retained-overlap conditioning diagnostics.
    """
    weights = np.asarray(spectrum.weights)
    if weights.ndim == 3:
        weights = weights[:, 0, 0]
    weights = np.asarray(weights.real, dtype=np.float64)
    poles = np.asarray(spectrum.excitation_energies, dtype=np.float64)
    spectral_norm = float(np.sum(weights))
    spectral_first = float(np.sum(poles * weights))
    if source_in_basis:
        source_arr = np.asarray(source, dtype=np.complex128)
        if source_arr.ndim == 2:
            source_norm = float(source_arr[0, 0].real)
        else:
            source_norm = float(source_arr[0].real)
        source_first = float(np.asarray(hamiltonian, dtype=np.complex128)[0, 0].real)
    else:
        source_norm = spectral_norm
        source_first = spectral_first
    norm_rel = abs(spectral_norm - source_norm) / max(abs(source_norm), 1e-12)
    first_rel = abs(spectral_first - source_first) / max(abs(source_first), 1e-12)
    overlap_evals = np.linalg.eigvalsh(
        (np.asarray(overlap, dtype=np.complex128) + np.asarray(overlap).conj().T) / 2
    ).real
    positive = overlap_evals[
        overlap_evals > max(float(overlap_cutoff) * float(np.max(overlap_evals)), 1e-14)
    ]
    if positive.size:
        overlap_condition = float(np.max(positive) / np.min(positive))
    else:
        overlap_condition = float("inf")
    return MomentDiagnostics(
        source_norm=source_norm,
        spectral_norm=spectral_norm,
        source_first_moment=source_first,
        spectral_first_moment=spectral_first,
        norm_rel_error=float(norm_rel),
        first_moment_rel_error=float(first_rel),
        min_weight=float(np.min(weights)) if weights.size else float("nan"),
        overlap_condition=overlap_condition,
    )


def should_accept_enrichment(
    *,
    initial_capture: float,
    final_capture: float,
    initial_objective: float,
    final_objective: float,
    moments: MomentDiagnostics,
    candidate_heads: int,
    min_relative_improvement: float,
    min_capture: float,
    min_objective_improvement: float,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    require_training_improvement: bool = True,
    holdout_capture_ratio_min: float | None = None,
    holdout_objective_delta_min: float | None = None,
    holdout_pass_fraction: float | None = None,
    holdout_pass_count: int | None = None,
    holdout_count: int | None = None,
    min_holdout_pass_fraction: float = 1.0,
    strong_residual_epsilon_max: float | None = None,
    strong_residual_epsilon_old_max: float | None = None,
    strong_residual_epsilon_over_eta_max: float | None = None,
    strong_residual_epsilon_over_eta_old_max: float | None = None,
    strong_residual_nonfinite_count: int = 0,
    strong_residual_node_fraction_max: float | None = None,
    strong_residual_node_fraction_old_max: float | None = None,
    max_strong_residual_epsilon_over_eta: float = float("inf"),
    max_strong_residual_provisional_ratio: float = 0.7,
    max_strong_residual_node_fraction: float = 1e-2,
    max_strong_residual_node_fraction_growth: float = 10.0,
    source_bright_passed: bool = True,
    source_bright_shift: float = float("nan"),
    active_heads: int = 0,
    attempt: int = 0,
) -> EnrichmentDiagnostics:
    capture_ratio = final_capture / max(initial_capture, min_capture, 1e-30)
    objective_delta = final_objective - initial_objective
    residual_capture_ratio = (
        capture_ratio
        if holdout_capture_ratio_min is None
        else float(holdout_capture_ratio_min)
    )
    residual_objective_delta = (
        objective_delta
        if holdout_objective_delta_min is None
        else float(holdout_objective_delta_min)
    )
    residual_pass_fraction = (
        1.0 if holdout_pass_fraction is None else float(holdout_pass_fraction)
    )
    residual_pass_count = 1 if holdout_pass_count is None else int(holdout_pass_count)
    residual_holdout_count = 1 if holdout_count is None else int(holdout_count)
    if require_training_improvement:
        residual_gate_passed = (
            final_capture >= min_capture
            and residual_capture_ratio >= 1 + min_relative_improvement
            and residual_objective_delta >= min_objective_improvement
            and residual_pass_fraction >= min_holdout_pass_fraction
        )
    else:
        residual_gate_passed = (
            final_capture >= min_capture
            and final_objective >= min_objective_improvement
            and residual_pass_fraction >= min_holdout_pass_fraction
        )
    strong_residual_global_passed = (
        True
        if strong_residual_epsilon_over_eta_max is None
        else (
            np.isfinite(strong_residual_epsilon_over_eta_max)
            and strong_residual_epsilon_over_eta_max
            <= max_strong_residual_epsilon_over_eta
        )
    )
    strong_residual_ratio = float("nan")
    if (
        strong_residual_epsilon_over_eta_max is not None
        and strong_residual_epsilon_over_eta_old_max is not None
        and np.isfinite(strong_residual_epsilon_over_eta_max)
        and np.isfinite(strong_residual_epsilon_over_eta_old_max)
        and strong_residual_epsilon_over_eta_old_max > 0.0
    ):
        strong_residual_ratio = (
            strong_residual_epsilon_over_eta_max
            / strong_residual_epsilon_over_eta_old_max
        )
    strong_residual_node_fraction_ratio = float("nan")
    if (
        strong_residual_node_fraction_max is not None
        and strong_residual_node_fraction_old_max is not None
        and np.isfinite(strong_residual_node_fraction_max)
        and np.isfinite(strong_residual_node_fraction_old_max)
        and strong_residual_node_fraction_old_max >= 0.0
    ):
        strong_residual_node_fraction_ratio = strong_residual_node_fraction_max / max(
            strong_residual_node_fraction_old_max, 1e-12
        )
    strong_residual_node_fraction_passed = True
    if strong_residual_node_fraction_max is not None:
        strong_residual_node_fraction_passed = np.isfinite(
            strong_residual_node_fraction_max
        )
        if strong_residual_node_fraction_passed:
            node_fraction_material = (
                strong_residual_node_fraction_max > max_strong_residual_node_fraction
            )
            node_fraction_grew = (
                np.isfinite(strong_residual_node_fraction_ratio)
                and strong_residual_node_fraction_ratio
                > max_strong_residual_node_fraction_growth
            )
            strong_residual_node_fraction_passed = not (
                node_fraction_material and node_fraction_grew
            )
    strong_residual_hard_passed = int(strong_residual_nonfinite_count) <= 0 and bool(
        strong_residual_node_fraction_passed
    )
    strong_residual_improved = (
        True
        if strong_residual_epsilon_over_eta_max is None
        else (
            strong_residual_hard_passed
            and np.isfinite(strong_residual_ratio)
            and strong_residual_ratio <= max_strong_residual_provisional_ratio
        )
    )
    strong_residual_passed = (
        strong_residual_hard_passed and strong_residual_global_passed
    )
    strong_residual_basis_passed = strong_residual_passed or strong_residual_improved
    accepted = (
        candidate_heads > 0
        and np.isfinite(capture_ratio)
        and np.isfinite(objective_delta)
        and residual_gate_passed
        and moments.norm_rel_error <= max_moment_rel_error
        and moments.first_moment_rel_error <= max_moment_rel_error
        and moments.overlap_condition <= max_overlap_condition
        and strong_residual_basis_passed
        and source_bright_passed
    )
    production_ready = bool(accepted and strong_residual_passed)
    accepted_reason = "residual" if accepted else "rejected"
    if accepted and not strong_residual_passed and strong_residual_improved:
        accepted_reason = "provisional_strong_residual"
    if not accepted and not source_bright_passed:
        accepted_reason = "source_bright_rejected"
    elif not accepted and not residual_gate_passed:
        accepted_reason = "residual_rejected"
    elif not accepted and not strong_residual_basis_passed:
        accepted_reason = "strong_residual_rejected"
    return EnrichmentDiagnostics(
        accepted=bool(accepted),
        active_heads_before=int(active_heads),
        candidate_heads=int(candidate_heads),
        accepted_heads=int(
            active_heads + candidate_heads if accepted else active_heads
        ),
        attempt=int(attempt),
        initial_capture=float(initial_capture),
        final_capture=float(final_capture),
        capture_ratio=float(capture_ratio),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_delta=float(objective_delta),
        moment_norm_rel_error=moments.norm_rel_error,
        moment_first_rel_error=moments.first_moment_rel_error,
        overlap_condition=moments.overlap_condition,
        holdout_capture_ratio_min=float(residual_capture_ratio),
        holdout_objective_delta_min=float(residual_objective_delta),
        holdout_pass_fraction=float(residual_pass_fraction),
        holdout_pass_count=int(residual_pass_count),
        holdout_count=int(residual_holdout_count),
        strong_residual_epsilon_max=(
            float("nan")
            if strong_residual_epsilon_max is None
            else float(strong_residual_epsilon_max)
        ),
        strong_residual_epsilon_old_max=(
            float("nan")
            if strong_residual_epsilon_old_max is None
            else float(strong_residual_epsilon_old_max)
        ),
        strong_residual_epsilon_over_eta_max=(
            float("nan")
            if strong_residual_epsilon_over_eta_max is None
            else float(strong_residual_epsilon_over_eta_max)
        ),
        strong_residual_epsilon_over_eta_old_max=(
            float("nan")
            if strong_residual_epsilon_over_eta_old_max is None
            else float(strong_residual_epsilon_over_eta_old_max)
        ),
        strong_residual_epsilon_over_eta_ratio=float(strong_residual_ratio),
        strong_residual_nonfinite_count=int(strong_residual_nonfinite_count),
        strong_residual_node_fraction_max=(
            float("nan")
            if strong_residual_node_fraction_max is None
            else float(strong_residual_node_fraction_max)
        ),
        strong_residual_node_fraction_old_max=(
            float("nan")
            if strong_residual_node_fraction_old_max is None
            else float(strong_residual_node_fraction_old_max)
        ),
        strong_residual_node_fraction_ratio=float(strong_residual_node_fraction_ratio),
        strong_residual_node_fraction_passed=bool(strong_residual_node_fraction_passed),
        strong_residual_hard_passed=bool(strong_residual_hard_passed),
        strong_residual_improved=bool(strong_residual_improved),
        strong_residual_passed=bool(strong_residual_passed),
        production_ready=bool(production_ready),
        source_bright_passed=bool(source_bright_passed),
        source_bright_shift=float(source_bright_shift),
        accepted_reason=accepted_reason,
    )


def failed_moment_diagnostics() -> MomentDiagnostics:
    """Return moment diagnostics that fail every validation gate."""
    return MomentDiagnostics(
        source_norm=float("nan"),
        spectral_norm=float("nan"),
        source_first_moment=float("nan"),
        spectral_first_moment=float("nan"),
        norm_rel_error=float("inf"),
        first_moment_rel_error=float("inf"),
        min_weight=float("nan"),
        overlap_condition=float("inf"),
    )


def summarize_pole_validations(
    poles: list[float] | np.ndarray,
    moments: list[MomentDiagnostics],
    *,
    max_moment_rel_error: float,
    max_overlap_condition: float,
) -> PoleValidationDiagnostics:
    """Summarize independent held-out pole/moment validation audits.

    Returns:
        Median/spread bright-pole statistics over the held-out audits that
        pass moment and overlap-conditioning gates.
    """
    poles_arr = np.asarray(poles, dtype=np.float64)
    norm_errors = np.asarray(
        [item.norm_rel_error for item in moments], dtype=np.float64
    )
    first_errors = np.asarray(
        [item.first_moment_rel_error for item in moments], dtype=np.float64
    )
    conditions = np.asarray(
        [item.overlap_condition for item in moments], dtype=np.float64
    )
    total = int(poles_arr.size)
    pass_mask = (
        np.isfinite(poles_arr)
        & (norm_errors <= max_moment_rel_error)
        & (first_errors <= max_moment_rel_error)
        & (conditions <= max_overlap_condition)
    )
    passed = poles_arr[pass_mask]
    if passed.size:
        pole_median = float(np.median(passed))
        pole_mean = float(np.mean(passed))
        pole_std = float(np.std(passed))
        pole_min = float(np.min(passed))
        pole_max = float(np.max(passed))
        pole_spread = pole_max - pole_min
    else:
        pole_median = float("inf")
        pole_mean = float("inf")
        pole_std = float("inf")
        pole_min = float("inf")
        pole_max = float("inf")
        pole_spread = float("inf")
    finite_or_inf_norm = norm_errors if norm_errors.size else np.asarray([float("inf")])
    finite_or_inf_first = (
        first_errors if first_errors.size else np.asarray([float("inf")])
    )
    finite_or_inf_conditions = (
        conditions if conditions.size else np.asarray([float("inf")])
    )
    return PoleValidationDiagnostics(
        pole_median=pole_median,
        pole_mean=pole_mean,
        pole_std=pole_std,
        pole_spread=float(pole_spread),
        pole_min=pole_min,
        pole_max=pole_max,
        pass_count=int(np.count_nonzero(pass_mask)),
        total_count=total,
        moment_norm_rel_error_max=float(np.max(finite_or_inf_norm)),
        moment_first_rel_error_max=float(np.max(finite_or_inf_first)),
        overlap_condition_max=float(np.max(finite_or_inf_conditions)),
    )


def pole_validation_passed(
    validation: PoleValidationDiagnostics,
    *,
    max_spread: float,
    min_pass_fraction: float,
) -> bool:
    """Return whether independent held-out pole audits are stable enough."""
    if validation.total_count <= 0:
        return False
    pass_fraction = validation.pass_count / validation.total_count
    return (
        np.isfinite(validation.pole_median)
        and np.isfinite(validation.pole_spread)
        and validation.pole_spread <= max_spread
        and pass_fraction >= min_pass_fraction
    )


def source_bright_gate_passed(
    *,
    active_validation: PoleValidationDiagnostics,
    candidate_validation: PoleValidationDiagnostics,
    active_heads: int,
    max_spread: float,
    min_pass_fraction: float,
    max_regression: float,
) -> tuple[bool, float]:
    """Return whether a candidate preserves the source-bright spectrum.

    This is an optional stability audit.  The formal paper-aligned acceptance
    path uses held-out residual improvement plus source moments/conditioning;
    callers can additionally enable this gate to reject candidates that move
    an already-visible bright pole upward by more than the allowed regression.
    """
    if not pole_validation_passed(
        candidate_validation,
        max_spread=max_spread,
        min_pass_fraction=min_pass_fraction,
    ):
        return False, float("nan")
    if active_heads <= 0:
        return True, float("nan")
    if not pole_validation_passed(
        active_validation,
        max_spread=max_spread,
        min_pass_fraction=min_pass_fraction,
    ):
        return True, float("nan")
    shift = candidate_validation.pole_median - active_validation.pole_median
    return bool(shift <= max_regression), float(shift)


@dataclass(frozen=True)
class FermiNetGround:
    """Restored FermiNet ground state for the molecular BF-NKSR flow."""

    wf: Any
    params: Params
    atoms: jax.Array
    charges: jax.Array
    electron_shape: tuple[int, int]
    nspins: tuple[int, int]
    energy: float
    checkpoint_step: int
    response_model: ResponseModel | None = None


@dataclass(frozen=True)
class MomentDiagnostics:
    """Moment and conditioning checks for a source-indexed response basis."""

    source_norm: float
    spectral_norm: float
    source_first_moment: float
    spectral_first_moment: float
    norm_rel_error: float
    first_moment_rel_error: float
    min_weight: float
    overlap_condition: float


@dataclass(frozen=True)
class PoleValidationDiagnostics:
    """Independent held-out bright-pole stability diagnostics."""

    pole_median: float
    pole_mean: float
    pole_std: float
    pole_spread: float
    pole_min: float
    pole_max: float
    pass_count: int
    total_count: int
    moment_norm_rel_error_max: float
    moment_first_rel_error_max: float
    overlap_condition_max: float


@dataclass(frozen=True)
class EnrichmentDiagnostics:
    """Held-out residual-enrichment acceptance diagnostics."""

    accepted: bool
    active_heads_before: int
    candidate_heads: int
    accepted_heads: int
    attempt: int
    initial_capture: float
    final_capture: float
    capture_ratio: float
    initial_objective: float
    final_objective: float
    objective_delta: float
    moment_norm_rel_error: float
    moment_first_rel_error: float
    overlap_condition: float
    holdout_capture_ratio_min: float = float("nan")
    holdout_objective_delta_min: float = float("nan")
    holdout_pass_fraction: float = float("nan")
    holdout_pass_count: int = 0
    holdout_count: int = 0
    selected_pole: float = float("nan")
    pole_improvement: float = float("nan")
    pole_spread: float = float("nan")
    pole_validation_pass_count: int = 0
    pole_validation_count: int = 0
    strong_residual_epsilon_max: float = float("nan")
    strong_residual_epsilon_old_max: float = float("nan")
    strong_residual_epsilon_over_eta_max: float = float("nan")
    strong_residual_epsilon_over_eta_old_max: float = float("nan")
    strong_residual_epsilon_over_eta_ratio: float = float("nan")
    strong_residual_nonfinite_count: int = 0
    strong_residual_node_fraction_max: float = float("nan")
    strong_residual_node_fraction_old_max: float = float("nan")
    strong_residual_node_fraction_ratio: float = float("nan")
    strong_residual_node_fraction_passed: bool = True
    strong_residual_hard_passed: bool = True
    strong_residual_improved: bool = True
    strong_residual_passed: bool = True
    production_ready: bool = True
    strong_oracle_train_epsilon_old_max: float = float("nan")
    strong_oracle_train_epsilon_oracle_max: float = float("nan")
    strong_oracle_validation_epsilon_old_max: float = float("nan")
    strong_oracle_validation_epsilon_oracle_max: float = float("nan")
    strong_oracle_validation_ratio_max: float = float("nan")
    strong_oracle_validation_ratio_winsor99_max: float = float("nan")
    strong_oracle_validation_ratio_p95_max: float = float("nan")
    strong_oracle_validation_ratio_p99_max: float = float("nan")
    strong_oracle_validation_ratio_pointwise_max_max: float = float("nan")
    strong_oracle_validation_improvement_min: float = float("nan")
    strong_oracle_train_epsilon2_improvement_min: float = float("nan")
    strong_oracle_validation_epsilon2_improvement_min: float = float("nan")
    strong_oracle_validation_relative_epsilon2_improvement_min: float = float("nan")
    strong_oracle_validation_winsor99_epsilon2_improvement_min: float = float("nan")
    strong_oracle_validation_winsor99_relative_epsilon2_improvement_min: float = float(
        "nan"
    )
    strong_oracle_action_consistency_l2_max: float = float("nan")
    strong_oracle_action_consistency_p95_max: float = float("nan")
    strong_oracle_action_consistency_p99_max: float = float("nan")
    strong_oracle_action_consistency_pointwise_max_max: float = float("nan")
    strong_oracle_candidate_value_norm_min: float = float("nan")
    strong_oracle_candidate_value_norm_max: float = float("nan")
    strong_oracle_candidate_action_norm_min: float = float("nan")
    strong_oracle_candidate_action_norm_max: float = float("nan")
    strong_oracle_candidate_action_condition_max: float = float("nan")
    strong_oracle_passed: bool = True
    source_bright_passed: bool = True
    source_bright_shift: float = float("nan")
    accepted_reason: str = "residual"


@dataclass(frozen=True)
class ExternalCASBasisBlock:
    """Accepted external CASSCF/FermiNet response basis block.

    The block stores coefficients in the raw CASSCF carrier bank.  It is not a
    response-head parameter tree: production matrices evaluate it directly as
    ``Psi0 * rho_CAS`` columns.
    """

    model: Any
    coefficients: np.ndarray
    target_mode: str
    correction_omegas: np.ndarray
    correction_eta: float
    tau_rel: float
    tau_abs: float
    ratio_clip: float
    finite_difference_step: float
    method: str
    basis: str
    ncas: int
    n_roots: int
    tau: float


def external_cas_basis_count(
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None,
) -> int:
    """Return the number of direct external CAS/FN basis columns."""
    if not external_blocks:
        return 0
    return int(
        sum(np.asarray(block.coefficients).shape[1] for block in external_blocks)
    )


def strong_oracle_diagnostics_from_result(
    oracle: Mapping[str, np.ndarray | float | int],
) -> dict[str, float]:
    """Extract scalar enrichment diagnostics from a strong-oracle result.

    Returns:
        Scalar diagnostics used by the enrichment acceptance gates.
    """
    train_old = np.asarray(oracle["strong_oracle_train_epsilon_old"], dtype=np.float64)
    train_new = np.asarray(
        oracle["strong_oracle_train_epsilon_oracle"], dtype=np.float64
    )
    val_old = np.asarray(
        oracle["strong_oracle_validation_epsilon_old"], dtype=np.float64
    )
    val_new = np.asarray(
        oracle["strong_oracle_validation_epsilon_oracle"], dtype=np.float64
    )
    val_ratio = np.asarray(oracle["strong_oracle_validation_ratio"], dtype=np.float64)
    val_ratio_w99 = np.asarray(
        oracle["strong_oracle_validation_ratio_winsor99"], dtype=np.float64
    )
    val_ratio_p99 = np.asarray(
        oracle["strong_oracle_validation_ratio_p99"], dtype=np.float64
    )
    val_ratio_point_max = np.asarray(
        oracle["strong_oracle_validation_ratio_pointwise_max"],
        dtype=np.float64,
    )
    diagnostics = {
        "strong_oracle_train_epsilon_old_max": float(np.max(train_old)),
        "strong_oracle_train_epsilon_oracle_max": float(np.max(train_new)),
        "strong_oracle_validation_epsilon_old_max": float(np.max(val_old)),
        "strong_oracle_validation_epsilon_oracle_max": float(np.max(val_new)),
        "strong_oracle_validation_ratio_max": float(np.max(val_ratio)),
        "strong_oracle_validation_ratio_winsor99_max": float(np.max(val_ratio_w99)),
        "strong_oracle_validation_ratio_p95_max": float(
            oracle["strong_oracle_validation_ratio_p95_max"]
        ),
        "strong_oracle_validation_ratio_p99_max": float(np.max(val_ratio_p99)),
        "strong_oracle_validation_ratio_pointwise_max_max": float(
            np.max(val_ratio_point_max)
        ),
        "strong_oracle_validation_improvement_min": float(
            oracle["strong_oracle_validation_improvement_min"]
        ),
        "strong_oracle_train_epsilon2_improvement_min": float(
            oracle["strong_oracle_train_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_epsilon2_improvement_min": float(
            oracle["strong_oracle_validation_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_relative_epsilon2_improvement_min": float(
            oracle["strong_oracle_validation_relative_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_winsor99_epsilon2_improvement_min": float(
            oracle["strong_oracle_validation_winsor99_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min": (
            float(
                oracle[
                    "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min"
                ]
            )
        ),
        "strong_oracle_action_consistency_l2_max": float(
            oracle["strong_oracle_action_consistency_l2_max"]
        ),
        "strong_oracle_action_consistency_p95_max": float(
            oracle["strong_oracle_action_consistency_p95_max"]
        ),
        "strong_oracle_action_consistency_p99_max": float(
            oracle["strong_oracle_action_consistency_p99_max"]
        ),
        "strong_oracle_action_consistency_pointwise_max_max": float(
            oracle["strong_oracle_action_consistency_pointwise_max_max"]
        ),
        "strong_oracle_candidate_value_norm_min": float(
            oracle["strong_oracle_candidate_value_norm_min"]
        ),
        "strong_oracle_candidate_value_norm_max": float(
            oracle["strong_oracle_candidate_value_norm_max"]
        ),
        "strong_oracle_candidate_action_norm_min": float(
            oracle["strong_oracle_candidate_action_norm_min"]
        ),
        "strong_oracle_candidate_action_norm_max": float(
            oracle["strong_oracle_candidate_action_norm_max"]
        ),
        "strong_oracle_candidate_action_condition_max": float(
            oracle["strong_oracle_candidate_action_condition_max"]
        ),
    }
    return diagnostics


def log_strong_oracle_result(
    oracle: Mapping[str, np.ndarray | float | int],
    *,
    attempt: int,
    samples: int,
    action_schur: bool,
    ridge: float,
    schur_ridge: float,
    label: str = "raw",
) -> None:
    """Print the compact strong-oracle diagnostic line."""
    diagnostics = strong_oracle_diagnostics_from_result(oracle)
    train_old = np.asarray(oracle["strong_oracle_train_epsilon_old"], dtype=np.float64)
    train_new = np.asarray(
        oracle["strong_oracle_train_epsilon_oracle"], dtype=np.float64
    )
    val_old = np.asarray(
        oracle["strong_oracle_validation_epsilon_old"], dtype=np.float64
    )
    val_new = np.asarray(
        oracle["strong_oracle_validation_epsilon_oracle"], dtype=np.float64
    )
    val_ratio = np.asarray(oracle["strong_oracle_validation_ratio"], dtype=np.float64)
    val_ratio_w99 = np.asarray(
        oracle["strong_oracle_validation_ratio_winsor99"], dtype=np.float64
    )
    val_ratio_p99 = np.asarray(
        oracle["strong_oracle_validation_ratio_p99"], dtype=np.float64
    )
    val_ratio_point_max = np.asarray(
        oracle["strong_oracle_validation_ratio_pointwise_max"],
        dtype=np.float64,
    )
    raw_w99 = float(oracle["strong_oracle_raw_validation_ratio_winsor99_max"])
    raw_action_rel_improvement = float(
        oracle["strong_oracle_raw_validation_relative_epsilon2_improvement_min"]
    )
    val_action_rel_improvement = diagnostics[
        "strong_oracle_validation_relative_epsilon2_improvement_min"
    ]
    val_w99_action_rel_improvement = diagnostics[
        "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min"
    ]
    print(
        "response_enrichment_strong_oracle "
        f"attempt={attempt:02d} label={label} samples={samples} "
        "action_mode=ad "
        f"action_schur={action_schur} "
        f"ridge={ridge:.3e} "
        f"schur_ridge={schur_ridge:.3e} "
        f"train_old_max={float(np.max(train_old)):.3e} "
        f"train_oracle_max={float(np.max(train_new)):.3e} "
        f"val_old_max={float(np.max(val_old)):.3e} "
        f"val_oracle_max={float(np.max(val_new)):.3e} "
        f"val_ratio_max={float(np.max(val_ratio)):.3e} "
        f"val_ratio_w99_max={float(np.max(val_ratio_w99)):.3e} "
        f"val_ratio_p99_max={float(np.max(val_ratio_p99)):.3e} "
        f"val_ratio_pointmax_max={float(np.max(val_ratio_point_max)):.3e} "
        f"val_action_rel_improve="
        f"{val_action_rel_improvement:.3e} "
        f"val_w99_action_rel_improve="
        f"{val_w99_action_rel_improvement:.3e} "
        f"raw_val_w99={raw_w99:.3e} "
        f"raw_val_action_rel_improve="
        f"{raw_action_rel_improvement:.3e} "
        f"cand_value_norm_min="
        f"{diagnostics['strong_oracle_candidate_value_norm_min']:.3e} "
        f"cand_action_norm_min="
        f"{diagnostics['strong_oracle_candidate_action_norm_min']:.3e} "
        f"cand_action_cond_max="
        f"{diagnostics['strong_oracle_candidate_action_condition_max']:.3e}"
    )


def strong_oracle_gate_passed(
    diagnostics: EnrichmentDiagnostics,
    *,
    max_validation_ratio_winsor99: float,
    max_validation_ratio_p99: float,
    max_validation_ratio_pointwise: float,
    min_validation_relative_epsilon2_improvement: float,
    min_validation_winsor99_relative_epsilon2_improvement: float,
    min_candidate_value_norm: float,
    min_candidate_action_norm: float,
    max_candidate_action_condition: float,
) -> bool:
    """Return whether optional strong-oracle candidate gates pass."""

    def upper_passed(value: float, threshold: float) -> bool:
        if not np.isfinite(threshold):
            return True
        return np.isfinite(value) and value <= threshold

    def signed_lower_passed(value: float, threshold: float) -> bool:
        if np.isneginf(threshold):
            return True
        return np.isfinite(value) and value >= threshold

    def nonnegative_floor_passed(value: float, threshold: float) -> bool:
        if threshold <= 0.0:
            return True
        return np.isfinite(value) and value >= threshold

    return (
        upper_passed(
            diagnostics.strong_oracle_validation_ratio_winsor99_max,
            max_validation_ratio_winsor99,
        )
        and upper_passed(
            diagnostics.strong_oracle_validation_ratio_p99_max,
            max_validation_ratio_p99,
        )
        and upper_passed(
            diagnostics.strong_oracle_validation_ratio_pointwise_max_max,
            max_validation_ratio_pointwise,
        )
        and signed_lower_passed(
            diagnostics.strong_oracle_validation_relative_epsilon2_improvement_min,
            min_validation_relative_epsilon2_improvement,
        )
        and signed_lower_passed(
            diagnostics.strong_oracle_validation_winsor99_relative_epsilon2_improvement_min,
            min_validation_winsor99_relative_epsilon2_improvement,
        )
        and nonnegative_floor_passed(
            diagnostics.strong_oracle_candidate_value_norm_min,
            min_candidate_value_norm,
        )
        and nonnegative_floor_passed(
            diagnostics.strong_oracle_candidate_action_norm_min,
            min_candidate_action_norm,
        )
        and upper_passed(
            diagnostics.strong_oracle_candidate_action_condition_max,
            max_candidate_action_condition,
        )
    )


def apply_strong_oracle_gate(
    diagnostics: EnrichmentDiagnostics,
    *,
    max_validation_ratio_winsor99: float,
    max_validation_ratio_p99: float,
    max_validation_ratio_pointwise: float,
    min_validation_relative_epsilon2_improvement: float,
    min_validation_winsor99_relative_epsilon2_improvement: float,
    min_candidate_value_norm: float,
    min_candidate_action_norm: float,
    max_candidate_action_condition: float,
) -> EnrichmentDiagnostics:
    """Apply optional strong-oracle gates to an accepted candidate.

    Returns:
        Diagnostics with ``strong_oracle_passed`` set and acceptance downgraded
        when an otherwise accepted candidate fails the oracle gate.
    """
    passed = strong_oracle_gate_passed(
        diagnostics,
        max_validation_ratio_winsor99=max_validation_ratio_winsor99,
        max_validation_ratio_p99=max_validation_ratio_p99,
        max_validation_ratio_pointwise=max_validation_ratio_pointwise,
        min_validation_relative_epsilon2_improvement=(
            min_validation_relative_epsilon2_improvement
        ),
        min_validation_winsor99_relative_epsilon2_improvement=(
            min_validation_winsor99_relative_epsilon2_improvement
        ),
        min_candidate_value_norm=min_candidate_value_norm,
        min_candidate_action_norm=min_candidate_action_norm,
        max_candidate_action_condition=max_candidate_action_condition,
    )
    if diagnostics.accepted and not passed:
        return replace(
            diagnostics,
            accepted=False,
            accepted_heads=diagnostics.active_heads_before,
            strong_oracle_passed=False,
            production_ready=False,
            accepted_reason="strong_oracle_rejected",
        )
    return replace(diagnostics, strong_oracle_passed=passed)


def action_oracle_acceptance_thresholds(
    *,
    selection_objective: str,
    min_validation_relative_epsilon2_improvement: float,
    min_validation_winsor99_relative_epsilon2_improvement: float,
) -> tuple[float, float]:
    """Return effective lower bounds for action-oracle candidate selection."""
    if selection_objective != "action-oracle":
        return (
            min_validation_relative_epsilon2_improvement,
            min_validation_winsor99_relative_epsilon2_improvement,
        )
    return (
        0.0
        if np.isneginf(min_validation_relative_epsilon2_improvement)
        else min_validation_relative_epsilon2_improvement,
        0.0
        if np.isneginf(min_validation_winsor99_relative_epsilon2_improvement)
        else min_validation_winsor99_relative_epsilon2_improvement,
    )


def promote_action_oracle_acceptance(
    diagnostics: EnrichmentDiagnostics,
    *,
    selection_objective: str,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    max_validation_ratio_winsor99: float,
    max_validation_ratio_p99: float,
    max_validation_ratio_pointwise: float,
    min_validation_relative_epsilon2_improvement: float,
    min_validation_winsor99_relative_epsilon2_improvement: float,
    min_candidate_value_norm: float,
    min_candidate_action_norm: float,
    max_candidate_action_condition: float,
) -> EnrichmentDiagnostics:
    """Accept a candidate when held-out action-space oracle selection passes.

    This is the no-action-backprop version of residual enrichment: candidate
    heads are trained by the regular FermiNet flow, then ranked/accepted by
    whether their action columns reduce the old strong residual on held-out
    strong samples.

    Returns:
        Possibly promoted diagnostics.
    """
    if selection_objective != "action-oracle" or diagnostics.accepted:
        return diagnostics
    strong_oracle_passed = strong_oracle_gate_passed(
        diagnostics,
        max_validation_ratio_winsor99=max_validation_ratio_winsor99,
        max_validation_ratio_p99=max_validation_ratio_p99,
        max_validation_ratio_pointwise=max_validation_ratio_pointwise,
        min_validation_relative_epsilon2_improvement=(
            min_validation_relative_epsilon2_improvement
        ),
        min_validation_winsor99_relative_epsilon2_improvement=(
            min_validation_winsor99_relative_epsilon2_improvement
        ),
        min_candidate_value_norm=min_candidate_value_norm,
        min_candidate_action_norm=min_candidate_action_norm,
        max_candidate_action_condition=max_candidate_action_condition,
    )
    structural_passed = (
        diagnostics.candidate_heads > 0
        and diagnostics.source_bright_passed
        and diagnostics.strong_residual_passed
        and diagnostics.moment_norm_rel_error <= max_moment_rel_error
        and diagnostics.moment_first_rel_error <= max_moment_rel_error
        and diagnostics.overlap_condition <= max_overlap_condition
    )
    if not (strong_oracle_passed and structural_passed):
        return diagnostics
    return replace(
        diagnostics,
        accepted=True,
        accepted_heads=(diagnostics.active_heads_before + diagnostics.candidate_heads),
        strong_oracle_passed=True,
        production_ready=diagnostics.strong_residual_passed,
        accepted_reason="action_oracle",
    )


def is_better_enrichment_candidate(
    candidate: EnrichmentDiagnostics,
    current_best: EnrichmentDiagnostics | None,
) -> bool:
    """Compare accepted candidates using robust held-out residual diagnostics.

    Source-bright pole stability is a validation gate, not the selection
    objective for the formal residual-enrichment workflow.  Among accepted
    candidates, prefer the one with the strongest worst-case held-out residual
    improvement before falling back to median residual objective statistics.

    Returns:
        Whether ``candidate`` should replace ``current_best``.
    """
    if not candidate.accepted:
        return False
    if current_best is None:
        return True
    selection_keys = (
        (
            "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min",
            "max",
        ),
        (
            "strong_oracle_validation_relative_epsilon2_improvement_min",
            "max",
        ),
        ("strong_oracle_validation_ratio_winsor99_max", "min"),
        ("strong_oracle_validation_ratio_p99_max", "min"),
        ("holdout_objective_delta_min", "max"),
        ("holdout_capture_ratio_min", "max"),
        ("final_objective", "max"),
        ("final_capture", "max"),
    )
    for key, direction in selection_keys:
        candidate_value = float(getattr(candidate, key))
        current_value = float(getattr(current_best, key))
        candidate_finite = np.isfinite(candidate_value)
        current_finite = np.isfinite(current_value)
        if candidate_finite != current_finite:
            return bool(candidate_finite)
        if candidate_finite and candidate_value != current_value:
            if direction == "min":
                return candidate_value < current_value
            return candidate_value > current_value
    return candidate.attempt < current_best.attempt


def residual_holdout_acceptance_summary(
    initial_captures: np.ndarray,
    final_captures: np.ndarray,
    initial_objectives: np.ndarray,
    final_objectives: np.ndarray,
    *,
    min_relative_improvement: float,
    min_capture: float,
    min_objective_improvement: float,
    require_training_improvement: bool,
) -> dict[str, float | int]:
    """Summarize paper-style held-out residual validation.

    Candidate selection still ranks by the held-out residual objective, but
    acceptance is conservative: the stored ratio/delta are the worst finite
    held-out values and the pass fraction records how many independent
    validation sets satisfy the residual gate.

    Returns:
        Median held-out residual statistics plus worst-case acceptance
        ratio/delta and pass-count diagnostics.

    Raises:
        ValueError: If the held-out arrays are empty or have different shapes.
    """
    initial_captures = np.asarray(initial_captures, dtype=np.float64)
    final_captures = np.asarray(final_captures, dtype=np.float64)
    initial_objectives = np.asarray(initial_objectives, dtype=np.float64)
    final_objectives = np.asarray(final_objectives, dtype=np.float64)
    if (
        initial_captures.shape != final_captures.shape
        or initial_captures.shape != initial_objectives.shape
        or initial_captures.shape != final_objectives.shape
    ):
        msg = "held-out residual arrays must have matching shapes"
        raise ValueError(msg)
    if initial_captures.size == 0:
        msg = "at least one held-out residual validation set is required"
        raise ValueError(msg)

    ratios = final_captures / np.maximum(
        np.maximum(initial_captures, min_capture), 1e-30
    )
    objective_deltas = final_objectives - initial_objectives
    finite_mask = (
        np.isfinite(initial_captures)
        & np.isfinite(final_captures)
        & np.isfinite(initial_objectives)
        & np.isfinite(final_objectives)
        & np.isfinite(ratios)
        & np.isfinite(objective_deltas)
    )
    if require_training_improvement:
        pass_mask = (
            finite_mask
            & (final_captures >= min_capture)
            & (ratios >= 1 + min_relative_improvement)
            & (objective_deltas >= min_objective_improvement)
        )
    else:
        pass_mask = (
            finite_mask
            & (final_captures >= min_capture)
            & (final_objectives >= min_objective_improvement)
        )

    def finite_median(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")

    def finite_min(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.min(finite)) if finite.size else float("-inf")

    pass_count = int(np.count_nonzero(pass_mask))
    total_count = int(initial_captures.size)
    return {
        "initial_capture": finite_median(initial_captures),
        "final_capture": finite_median(final_captures),
        "initial_objective": finite_median(initial_objectives),
        "final_objective": finite_median(final_objectives),
        "capture_ratio_min": finite_min(ratios),
        "objective_delta_min": finite_min(objective_deltas),
        "pass_count": pass_count,
        "count": total_count,
        "pass_fraction": float(pass_count / total_count),
    }


def heldout_objective_selection_summary(
    captures: np.ndarray,
    objectives: np.ndarray,
) -> dict[str, float]:
    """Summarize held-out objectives for robust within-attempt early stopping.

    Returns:
        Worst finite objective plus median objective/capture tie-breakers.

    Raises:
        ValueError: If arrays are empty or have different shapes.
    """
    captures = np.asarray(captures, dtype=np.float64)
    objectives = np.asarray(objectives, dtype=np.float64)
    if captures.shape != objectives.shape:
        msg = "held-out captures and objectives must have matching shapes"
        raise ValueError(msg)
    if captures.size == 0:
        msg = "at least one held-out objective is required"
        raise ValueError(msg)
    finite_objectives = objectives[np.isfinite(objectives)]
    finite_captures = captures[np.isfinite(captures)]
    return {
        "objective_min": (
            float(np.min(finite_objectives))
            if finite_objectives.size
            else float("-inf")
        ),
        "objective": (
            float(np.median(finite_objectives))
            if finite_objectives.size
            else float("nan")
        ),
        "capture": (
            float(np.median(finite_captures)) if finite_captures.size else float("nan")
        ),
    }


def is_better_heldout_objective_summary(
    candidate: dict[str, float],
    current_best: dict[str, float],
) -> bool:
    """Return whether a held-out summary is a better early-stop snapshot."""
    for key in ("objective_min", "objective", "capture"):
        candidate_value = float(candidate[key])
        current_value = float(current_best[key])
        candidate_finite = np.isfinite(candidate_value)
        current_finite = np.isfinite(current_value)
        if candidate_finite != current_finite:
            return bool(candidate_finite)
        if candidate_finite and candidate_value != current_value:
            return candidate_value > current_value
    return False


def enrichment_history_arrays(
    history: list[EnrichmentDiagnostics],
) -> dict[str, np.ndarray]:
    """Return compact arrays for saving candidate-generation diagnostics."""

    def float_history(field: str) -> np.ndarray:
        return np.asarray([getattr(item, field) for item in history], dtype=np.float64)

    return {
        "enrichment_history_accepted": np.asarray(
            [item.accepted for item in history], dtype=bool
        ),
        "enrichment_history_active_heads_before": np.asarray(
            [item.active_heads_before for item in history], dtype=np.int64
        ),
        "enrichment_history_candidate_heads": np.asarray(
            [item.candidate_heads for item in history], dtype=np.int64
        ),
        "enrichment_history_accepted_heads": np.asarray(
            [item.accepted_heads for item in history], dtype=np.int64
        ),
        "enrichment_history_attempt": np.asarray(
            [item.attempt for item in history], dtype=np.int64
        ),
        "enrichment_history_initial_capture": np.asarray(
            [item.initial_capture for item in history], dtype=np.float64
        ),
        "enrichment_history_final_capture": np.asarray(
            [item.final_capture for item in history], dtype=np.float64
        ),
        "enrichment_history_capture_ratio": np.asarray(
            [item.capture_ratio for item in history], dtype=np.float64
        ),
        "enrichment_history_initial_objective": np.asarray(
            [item.initial_objective for item in history], dtype=np.float64
        ),
        "enrichment_history_final_objective": np.asarray(
            [item.final_objective for item in history], dtype=np.float64
        ),
        "enrichment_history_objective_delta": np.asarray(
            [item.objective_delta for item in history], dtype=np.float64
        ),
        "enrichment_history_moment_norm_rel_error": np.asarray(
            [item.moment_norm_rel_error for item in history], dtype=np.float64
        ),
        "enrichment_history_moment_first_rel_error": np.asarray(
            [item.moment_first_rel_error for item in history], dtype=np.float64
        ),
        "enrichment_history_overlap_condition": np.asarray(
            [item.overlap_condition for item in history], dtype=np.float64
        ),
        "enrichment_history_holdout_capture_ratio_min": np.asarray(
            [item.holdout_capture_ratio_min for item in history], dtype=np.float64
        ),
        "enrichment_history_holdout_objective_delta_min": np.asarray(
            [item.holdout_objective_delta_min for item in history], dtype=np.float64
        ),
        "enrichment_history_holdout_pass_fraction": np.asarray(
            [item.holdout_pass_fraction for item in history], dtype=np.float64
        ),
        "enrichment_history_holdout_pass_count": np.asarray(
            [item.holdout_pass_count for item in history], dtype=np.int64
        ),
        "enrichment_history_holdout_count": np.asarray(
            [item.holdout_count for item in history], dtype=np.int64
        ),
        "enrichment_history_selected_pole": np.asarray(
            [item.selected_pole for item in history], dtype=np.float64
        ),
        "enrichment_history_pole_improvement": np.asarray(
            [item.pole_improvement for item in history], dtype=np.float64
        ),
        "enrichment_history_pole_spread": np.asarray(
            [item.pole_spread for item in history], dtype=np.float64
        ),
        "enrichment_history_pole_validation_pass_count": np.asarray(
            [item.pole_validation_pass_count for item in history], dtype=np.int64
        ),
        "enrichment_history_pole_validation_count": np.asarray(
            [item.pole_validation_count for item in history], dtype=np.int64
        ),
        "enrichment_history_strong_residual_epsilon_max": np.asarray(
            [item.strong_residual_epsilon_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_epsilon_old_max": np.asarray(
            [item.strong_residual_epsilon_old_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_epsilon_over_eta_max": np.asarray(
            [item.strong_residual_epsilon_over_eta_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_epsilon_over_eta_old_max": np.asarray(
            [item.strong_residual_epsilon_over_eta_old_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_epsilon_over_eta_ratio": np.asarray(
            [item.strong_residual_epsilon_over_eta_ratio for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_nonfinite_count": np.asarray(
            [item.strong_residual_nonfinite_count for item in history],
            dtype=np.int64,
        ),
        "enrichment_history_strong_residual_node_fraction_max": np.asarray(
            [item.strong_residual_node_fraction_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_node_fraction_old_max": np.asarray(
            [item.strong_residual_node_fraction_old_max for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_node_fraction_ratio": np.asarray(
            [item.strong_residual_node_fraction_ratio for item in history],
            dtype=np.float64,
        ),
        "enrichment_history_strong_residual_node_fraction_passed": np.asarray(
            [item.strong_residual_node_fraction_passed for item in history],
            dtype=bool,
        ),
        "enrichment_history_strong_residual_hard_passed": np.asarray(
            [item.strong_residual_hard_passed for item in history],
            dtype=bool,
        ),
        "enrichment_history_strong_residual_improved": np.asarray(
            [item.strong_residual_improved for item in history], dtype=bool
        ),
        "enrichment_history_strong_residual_passed": np.asarray(
            [item.strong_residual_passed for item in history], dtype=bool
        ),
        "enrichment_history_production_ready": np.asarray(
            [item.production_ready for item in history], dtype=bool
        ),
        "enrichment_history_strong_oracle_train_epsilon_old_max": float_history(
            "strong_oracle_train_epsilon_old_max"
        ),
        "enrichment_history_strong_oracle_train_epsilon_oracle_max": (
            float_history("strong_oracle_train_epsilon_oracle_max")
        ),
        "enrichment_history_strong_oracle_validation_epsilon_old_max": (
            float_history("strong_oracle_validation_epsilon_old_max")
        ),
        "enrichment_history_strong_oracle_validation_epsilon_oracle_max": (
            float_history("strong_oracle_validation_epsilon_oracle_max")
        ),
        "enrichment_history_strong_oracle_validation_ratio_max": float_history(
            "strong_oracle_validation_ratio_max"
        ),
        "enrichment_history_strong_oracle_validation_ratio_winsor99_max": (
            float_history("strong_oracle_validation_ratio_winsor99_max")
        ),
        "enrichment_history_strong_oracle_validation_ratio_p95_max": (
            float_history("strong_oracle_validation_ratio_p95_max")
        ),
        "enrichment_history_strong_oracle_validation_ratio_p99_max": (
            float_history("strong_oracle_validation_ratio_p99_max")
        ),
        "enrichment_history_strong_oracle_validation_ratio_pointwise_max_max": (
            float_history("strong_oracle_validation_ratio_pointwise_max_max")
        ),
        "enrichment_history_strong_oracle_validation_improvement_min": (
            float_history("strong_oracle_validation_improvement_min")
        ),
        "enrichment_history_strong_oracle_train_epsilon2_improvement_min": (
            float_history("strong_oracle_train_epsilon2_improvement_min")
        ),
        "enrichment_history_strong_oracle_validation_epsilon2_improvement_min": (
            float_history("strong_oracle_validation_epsilon2_improvement_min")
        ),
        (
            "enrichment_history_strong_oracle_validation_"
            "relative_epsilon2_improvement_min"
        ): (
            float_history("strong_oracle_validation_relative_epsilon2_improvement_min")
        ),
        (
            "enrichment_history_strong_oracle_validation_"
            "winsor99_epsilon2_improvement_min"
        ): (
            float_history("strong_oracle_validation_winsor99_epsilon2_improvement_min")
        ),
        (
            "enrichment_history_strong_oracle_validation_"
            "winsor99_relative_epsilon2_improvement_min"
        ): (
            float_history(
                "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min"
            )
        ),
        "enrichment_history_strong_oracle_action_consistency_l2_max": (
            float_history("strong_oracle_action_consistency_l2_max")
        ),
        "enrichment_history_strong_oracle_action_consistency_p95_max": (
            float_history("strong_oracle_action_consistency_p95_max")
        ),
        "enrichment_history_strong_oracle_action_consistency_p99_max": (
            float_history("strong_oracle_action_consistency_p99_max")
        ),
        "enrichment_history_strong_oracle_action_consistency_pointwise_max_max": (
            float_history("strong_oracle_action_consistency_pointwise_max_max")
        ),
        "enrichment_history_strong_oracle_candidate_value_norm_min": (
            float_history("strong_oracle_candidate_value_norm_min")
        ),
        "enrichment_history_strong_oracle_candidate_value_norm_max": (
            float_history("strong_oracle_candidate_value_norm_max")
        ),
        "enrichment_history_strong_oracle_candidate_action_norm_min": (
            float_history("strong_oracle_candidate_action_norm_min")
        ),
        "enrichment_history_strong_oracle_candidate_action_norm_max": (
            float_history("strong_oracle_candidate_action_norm_max")
        ),
        "enrichment_history_strong_oracle_candidate_action_condition_max": (
            float_history("strong_oracle_candidate_action_condition_max")
        ),
        "enrichment_history_strong_oracle_passed": np.asarray(
            [item.strong_oracle_passed for item in history], dtype=bool
        ),
        "enrichment_history_source_bright_passed": np.asarray(
            [item.source_bright_passed for item in history], dtype=bool
        ),
        "enrichment_history_source_bright_shift": np.asarray(
            [item.source_bright_shift for item in history], dtype=np.float64
        ),
        "enrichment_history_accepted_reason": np.asarray(
            [item.accepted_reason for item in history], dtype="<U32"
        ),
    }


class _PrefixedNPZ(Mapping[str, np.ndarray]):
    def __init__(self, npz, prefix: str) -> None:
        self.npz = npz
        self.prefix = prefix

    def __getitem__(self, key: str) -> np.ndarray:
        return self.npz[self.prefix + key]

    def __iter__(self):
        return iter(self.npz)

    def __len__(self) -> int:
        return len(self.npz.files)


def _latest_checkpoint(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path
    ckpts = sorted(run_path.glob("train_ckpt_*.npz"))
    if not ckpts:
        msg = f"no train_ckpt_*.npz found in {run_path}"
        raise FileNotFoundError(msg)
    return ckpts[-1]


def load_ferminet_ground(
    run_path: Path, *, ground_energy: float | None
) -> FermiNetGround:
    """Restore the FermiNet molecule workflow checkpoint.

    Returns:
        Restored ground-state wavefunction, parameters, atom data, and energy.

    Raises:
        FileNotFoundError: If the checkpoint or adjacent config is missing.
    """
    checkpoint = _latest_checkpoint(run_path)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        msg = f"missing config.yaml next to checkpoint: {config_path}"
        raise FileNotFoundError(msg)
    with config_path.open() as f:
        cfg = ConfigManager(yaml.safe_load(f))
    workflow = MoleculeTrainWorkflow(cfg)
    batched_data = workflow.data_init(1, jax.random.PRNGKey(0))
    single_data = batched_data.unbatched_example()
    template_params = workflow.train_stage.wavefunction.init_params(
        single_data, jax.random.PRNGKey(1)
    )
    local_data = batched_data.data
    with np.load(checkpoint) as data:
        step = int(data["step"])
        params = tree_from_npz(_PrefixedNPZ(data, "params/"), template_params)
    return FermiNetGround(
        wf=workflow.train_stage.wavefunction,
        params=params,
        atoms=jnp.asarray(local_data.atoms),
        charges=jnp.asarray(local_data.charges),
        electron_shape=tuple(int(dim) for dim in single_data.electrons.shape),
        nspins=tuple(int(dim) for dim in workflow.train_stage.wavefunction.nspins),
        energy=float("nan") if ground_energy is None else float(ground_energy),
        checkpoint_step=step,
    )


def _ground_data(ground: FermiNetGround, point: jax.Array) -> MoleculeData:
    return MoleculeData(electrons=point, atoms=ground.atoms, charges=ground.charges)


def ground_phase_logpsi_single(ground: FermiNetGround, point: jax.Array):
    return ground.wf.phase_logpsi(ground.params, _ground_data(ground, point))


def ground_value_single(ground: FermiNetGround, point: jax.Array) -> jax.Array:
    phase, logpsi = ground_phase_logpsi_single(ground, point)
    return phase * jnp.exp(logpsi)


def ground_values_and_gradients(
    ground: FermiNetGround, points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    gradients = jax.vmap(jax.grad(ground_value_single, argnums=1), (None, 0))(
        ground, points
    )
    return values, gradients


def ground_value_laplacian_single(
    ground: FermiNetGround, point: jax.Array
) -> jax.Array:
    hessian = jax.hessian(ground_value_single, argnums=1)(ground, point)
    coord_size = point.size
    return jnp.trace(jnp.reshape(hessian, (coord_size, coord_size)))


def ground_values_and_laplacians(
    ground: FermiNetGround, points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    laplacians = jax.vmap(ground_value_laplacian_single, (None, 0))(ground, points)
    return values, laplacians


def ground_logpsi_batch(ground: FermiNetGround, points: jax.Array) -> jax.Array:
    _, logpsi = jax.vmap(ground_phase_logpsi_single, (None, 0))(ground, points)
    return logpsi


def ground_logpsi_single(ground: FermiNetGround, point: jax.Array) -> jax.Array:
    _, logpsi = ground_phase_logpsi_single(ground, point)
    return logpsi


def molecular_potential(ground: FermiNetGround, coords: jax.Array) -> jax.Array:
    electron_atom_radius = jnp.linalg.norm(
        coords[:, None, :] - ground.atoms[None, :, :], axis=2
    )
    electron_nuclear = -jnp.sum(ground.charges[None, :] / electron_atom_radius)
    electron_electron_radius = jnp.linalg.norm(
        coords[:, None, :] - coords[None, :, :], axis=2
    )
    electron_electron_radius = electron_electron_radius + jnp.eye(coords.shape[0])
    electron_repulsion = jnp.sum(jnp.triu(1 / electron_electron_radius, k=1))
    atom_atom_radius = jnp.linalg.norm(
        ground.atoms[:, None, :] - ground.atoms[None, :, :], axis=2
    )
    atom_atom_radius = atom_atom_radius + jnp.eye(ground.atoms.shape[0])
    charge_products = ground.charges[:, None] * ground.charges[None, :]
    nuclear_repulsion = jnp.sum(jnp.triu(charge_products / atom_atom_radius, k=1))
    return electron_nuclear + electron_repulsion + nuclear_repulsion


def local_energy_single(ground: FermiNetGround, point: jax.Array) -> jax.Array:
    grad_logpsi = jax.grad(ground_logpsi_single, argnums=1)(ground, point)
    hessian = jax.hessian(ground_logpsi_single, argnums=1)(ground, point)
    coord_size = point.size
    laplacian = jnp.trace(jnp.reshape(hessian, (coord_size, coord_size)))
    kinetic = -0.5 * (laplacian + jnp.sum(grad_logpsi**2))
    potential = molecular_potential(ground, point)
    return kinetic + potential


def local_energy_batch(ground: FermiNetGround, points: jax.Array) -> jax.Array:
    return jax.vmap(local_energy_single, (None, 0))(ground, points)


def ground_hbar_values(
    ground: FermiNetGround,
    points: jax.Array,
    ground_values: jax.Array | None = None,
) -> jax.Array:
    """Evaluate ``(H-E0) Psi0`` from the unfactored wavefunction action.

    Returns:
        One shifted ground-action value per sampled point.
    """
    values = (
        jax.vmap(ground_value_single, (None, 0))(ground, points)
        if ground_values is None
        else ground_values
    )
    laplacians = jax.vmap(ground_value_laplacian_single, (None, 0))(
        ground,
        points,
    )
    return -0.5 * laplacians + potential_shift(ground, points) * values


def source_carrier_value_single(ground: FermiNetGround, point: jax.Array) -> jax.Array:
    """Return the scalar dipole carrier ``f`` in ``source = Psi0 * f``."""
    charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
    charge_center = charge_center / jnp.sum(ground.charges)
    return jnp.sum((point - charge_center[None, :])[:, 2])


def carrier_product_rule_hbar_single(
    ground: FermiNetGround,
    point: jax.Array,
    carrier_fn: Callable[[FermiNetGround, jax.Array], jax.Array],
) -> jax.Array:
    """Evaluate ``(H-E0)(Psi0 f)`` by the analytic carrier product rule.

    Returns:
        The shifted Hamiltonian action for one carrier-product column.
    """
    psi0 = ground_value_single(ground, point)
    carrier_value = carrier_fn(ground, point)
    ground_gradient = jax.grad(ground_value_single, argnums=1)(ground, point)
    ground_laplacian = ground_value_laplacian_single(ground, point)
    ground_hbar = -0.5 * ground_laplacian + molecular_potential(ground, point) * psi0
    ground_hbar = ground_hbar - ground.energy * psi0
    grad_carrier = jax.grad(carrier_fn, argnums=1)(ground, point)
    carrier_hessian = jax.hessian(carrier_fn, argnums=1)(ground, point)
    coord_size = point.size
    carrier_laplacian = jnp.trace(
        jnp.reshape(carrier_hessian, (coord_size, coord_size))
    )
    return (
        carrier_value * ground_hbar
        - 0.5 * psi0 * carrier_laplacian
        - jnp.sum(ground_gradient * grad_carrier)
    )


def carrier_product_rule_hbar_batch(
    ground: FermiNetGround,
    points: jax.Array,
    carrier_fn: Callable[[FermiNetGround, jax.Array], jax.Array],
) -> jax.Array:
    """Batched ``(H-E0)(Psi0 f)`` carrier-action evaluator.

    Returns:
        One shifted Hamiltonian action value per sampled point.
    """
    return jax.vmap(
        lambda point: carrier_product_rule_hbar_single(ground, point, carrier_fn)
    )(points)


def sample_ground_mcmc(
    ground: FermiNetGround,
    *,
    key: jax.Array,
    n_samples: int,
    walkers: int,
    burn_in: int,
    steps_between: int,
    width: float,
    envelope_decay: float,
) -> tuple[np.ndarray, float]:
    """Sample from the restored FermiNet ``|Psi_0|^2`` density.

    Returns:
        Cartesian samples and the mean Metropolis acceptance probability.
    """
    points, _, _ = sample_envelope(key, walkers, envelope_decay, ground.electron_shape)
    logp = 2 * ground_logpsi_batch(ground, points)

    @jax.jit
    def mh_step(points: jax.Array, logp: jax.Array, key: jax.Array):
        kp, ka, kn = jax.random.split(key, 3)
        proposal = points + width * jax.random.normal(kp, points.shape)
        proposal_logp = 2 * ground_logpsi_batch(ground, proposal)
        accept = jnp.log(jax.random.uniform(ka, (walkers,))) < proposal_logp - logp
        accept_shape = (walkers,) + (1,) * (points.ndim - 1)
        points = jnp.where(jnp.reshape(accept, accept_shape), proposal, points)
        logp = jnp.where(accept, proposal_logp, logp)
        return points, logp, kn, jnp.mean(accept)

    accept_rates = []
    run_key = jax.random.fold_in(key, 19)
    for _ in range(burn_in):
        points, logp, run_key, pmove = mh_step(points, logp, run_key)
        accept_rates.append(float(pmove))
    collected = []
    while sum(batch.shape[0] for batch in collected) < n_samples:
        for _ in range(steps_between):
            points, logp, run_key, pmove = mh_step(points, logp, run_key)
            accept_rates.append(float(pmove))
        collected.append(np.asarray(points))
    return np.concatenate(collected, axis=0)[:n_samples], float(np.mean(accept_rates))


def estimate_ground_energy(
    ground: FermiNetGround,
    *,
    key: jax.Array,
    n_samples: int,
    walkers: int,
    burn_in: int,
    steps_between: int,
    width: float,
    batch_size: int,
    envelope_decay: float,
) -> tuple[float, float, float]:
    """Estimate ``E0`` from fixed-ground-state local energies.

    Returns:
        Mean energy, standard error estimated over local-energy samples, and
        MCMC acceptance probability.
    """
    samples, pmove = sample_ground_mcmc(
        ground,
        key=key,
        n_samples=n_samples,
        walkers=walkers,
        burn_in=burn_in,
        steps_between=steps_between,
        width=width,
        envelope_decay=envelope_decay,
    )
    pieces = []

    @jax.jit
    def chunk_energy(points: jax.Array):
        return local_energy_batch(ground, points)

    for chunk in make_batches(n_samples, batch_size):
        pieces.append(np.asarray(chunk_energy(jnp.asarray(samples[chunk]))))
    energies = np.concatenate(pieces)
    mean = float(np.mean(energies))
    stderr = float(np.std(energies, ddof=1) / np.sqrt(energies.size))
    return mean, stderr, pmove


def source_value_single(ground: FermiNetGround, point: jax.Array) -> jax.Array:
    phase, logpsi = ground_phase_logpsi_single(ground, point)
    charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
    charge_center = charge_center / jnp.sum(ground.charges)
    return jnp.sum((point - charge_center[None, :])[:, 2]) * phase * jnp.exp(logpsi)


def source_values_and_gradients(
    ground: FermiNetGround, points: jax.Array
) -> tuple[jax.Array, jax.Array]:
    values = jax.vmap(source_value_single, (None, 0))(ground, points)
    gradients = jax.vmap(jax.grad(source_value_single, argnums=1), (None, 0))(
        ground, points
    )
    return values, gradients


def source_values(ground: FermiNetGround, points: jax.Array) -> jax.Array:
    return jax.vmap(source_value_single, (None, 0))(ground, points)


def electron_pair_features(point: jax.Array, scales: jax.Array) -> jax.Array:
    """Return bounded symmetric electron-pair features for each scale.

    Returns:
        Mean ``r_ij / (scale + r_ij)`` over electron pairs, or zeros for a
        one-electron system.
    """
    if scales.size == 0:
        return jnp.zeros((0,), dtype=point.dtype)
    electron_count = point.shape[0]
    if electron_count < 2:
        return jnp.zeros((scales.size,), dtype=point.dtype)
    diff = point[:, None, :] - point[None, :, :]
    pair_r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    pair_mask = jnp.triu(
        jnp.ones((electron_count, electron_count), dtype=point.dtype), 1
    )
    pair_count = electron_count * (electron_count - 1) / 2
    feature = pair_r[:, :, None] / (scales[None, None, :] + pair_r[:, :, None])
    return jnp.sum(pair_mask[:, :, None] * feature, axis=(0, 1)) / pair_count


def electron_pair_exp_features(point: jax.Array, scales: jax.Array) -> jax.Array:
    """Return diffuse symmetric electron-pair exponential features.

    Returns:
        Mean ``exp(-r_ij / scale)`` over electron pairs, or zeros for a
        one-electron system.
    """
    if scales.size == 0:
        return jnp.zeros((0,), dtype=point.dtype)
    electron_count = point.shape[0]
    if electron_count < 2:
        return jnp.zeros((scales.size,), dtype=point.dtype)
    diff = point[:, None, :] - point[None, :, :]
    pair_r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    pair_mask = jnp.triu(
        jnp.ones((electron_count, electron_count), dtype=point.dtype), 1
    )
    pair_count = electron_count * (electron_count - 1) / 2
    feature = jnp.exp(-pair_r[:, :, None] / scales[None, None, :])
    return jnp.sum(pair_mask[:, :, None] * feature, axis=(0, 1)) / pair_count


def two_center_pair_features(
    point: jax.Array,
    atoms: jax.Array,
    scales: jax.Array,
) -> jax.Array:
    """Return bounded covalent pair features for a diatomic geometry.

    For each scale this averages
    ``r_ij/(s+r_ij) * (w_A(i)w_B(j) + w_B(i)w_A(j))`` over electron pairs,
    where ``w_A`` and ``w_B`` are smooth two-center assignment weights.  The
    feature is even under atom exchange and gives the two-center odd source
    probes an electron-nuclear correlation channel beyond a global pair
    distance.

    Returns:
        One bounded pair feature per scale.
    """
    if scales.size == 0:
        return jnp.zeros((0,), dtype=point.dtype)
    electron_count = point.shape[0]
    if electron_count < 2:
        return jnp.zeros((scales.size,), dtype=point.dtype)
    r_a = jnp.linalg.norm(point - atoms[0][None, :], axis=1)
    r_b = jnp.linalg.norm(point - atoms[1][None, :], axis=1)
    diff = point[:, None, :] - point[None, :, :]
    pair_r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    pair_mask = jnp.triu(
        jnp.ones((electron_count, electron_count), dtype=point.dtype), 1
    )
    pair_count = electron_count * (electron_count - 1) / 2
    scaled_a = jnp.exp(-r_a[:, None] / scales[None, :])
    scaled_b = jnp.exp(-r_b[:, None] / scales[None, :])
    norm = scaled_a + scaled_b + 1e-12
    weight_a = scaled_a / norm
    weight_b = scaled_b / norm
    covalent = (
        weight_a[:, None, :] * weight_b[None, :, :]
        + weight_b[:, None, :] * weight_a[None, :, :]
    )
    pair_distance = pair_r[:, :, None] / (scales[None, None, :] + pair_r[:, :, None])
    return jnp.sum(pair_mask[:, :, None] * covalent * pair_distance, axis=(0, 1)) / (
        pair_count
    )


def cusp_safe_auxiliary_pwave_values(
    ground: FermiNetGround,
    point: jax.Array,
    radial_powers: jax.Array,
    radial_scale: float,
) -> jax.Array:
    """Return diffuse atom-centered p-tail carrier probes.

    The public argument name is kept for compatibility with existing hidden
    CLI plumbing, but the official flow interprets the entries as tail
    exponents ``kappa``.  The carrier has far-field behavior
    ``z_iA exp((Z_A-kappa) r_iA)`` so that multiplying by a one-electron
    hydrogenic ground state can represent ``z exp(-kappa r)`` tails.
    """
    if radial_powers.size == 0:
        return jnp.zeros((0,), dtype=point.dtype)
    tail_kappas = radial_powers
    electron_atom = point[:, None, :] - ground.atoms[None, :, :]
    r_ea = jnp.sqrt(jnp.sum(electron_atom**2, axis=2) + 1e-24)
    support_radii = source_cusp_support_radii_from_radius(
        ground,
        float(radial_scale),
    )
    tau0 = source_cusp_lift_tau(r_ea, support_radii[None, :])
    tau_inf = cusp_neutral_radius(r_ea, float(radial_scale))
    charges = ground.charges[None, :, None]
    exponent = (charges / 2.0) * tau0[:, :, None] + (
        charges - tail_kappas[None, None, :]
    ) * tau_inf[:, :, None]
    lift = jnp.exp(jnp.clip(exponent, -80.0, 80.0))
    return jnp.sum(
        electron_atom[:, :, 2, None] * lift,
        axis=(0, 1),
    )


def auxiliary_source_carrier_value_single(
    ground: FermiNetGround,
    point: jax.Array,
    exponents: jax.Array,
    atom_odd_exponents: jax.Array | None = None,
    atom_odd_slater_decays: jax.Array | None = None,
    bond_odd_slater_decays: jax.Array | None = None,
    dipole_ee_scales: jax.Array | None = None,
    bond_odd_ee_slater_decays: jax.Array | None = None,
    bond_odd_ee_scales: jax.Array | None = None,
    dipole_radial_powers: jax.Array | None = None,
    dipole_radial_scale: float = 1.0,
) -> jax.Array:
    """Evaluate fixed auxiliary carrier probes before multiplying by Psi0.

    The first BF-NKSR source remains the physical dipole source.  These
    auxiliary source probes are optional fixed basis functions from the paper's
    source-design family and can improve radial resolution without changing
    the physical spectrum channel being read.  ``exponents`` gives centered
    ``p_z exp(-alpha r^2)`` probes. ``dipole_radial_powers`` is the historical
    CLI name for official diffuse p-tail exponents ``kappa`` and gives
    atom-centered carriers with far-field form
    ``z_iA exp((Z_A-kappa) r_iA)``. ``atom_odd_exponents`` gives, for two-atom
    systems, atom-centered odd Gaussian differences
    ``exp(-alpha |r-R_1|^2) - exp(-alpha |r-R_2|^2)``.  ``atom_odd_slater_decays``
    gives the corresponding Slater-tail differences
    ``exp(-zeta |r-R_1|) - exp(-zeta |r-R_2|)``. ``bond_odd_slater_decays``
    gives two-center odd orbital-ratio probes
    ``(exp(-zeta r_A) - exp(-zeta r_B)) / (exp(-r_A) + exp(-r_B))``.
    ``dipole_ee_scales`` gives correlated dipole probes
    ``sum_i z_i * mean_{i<j} r_ij / (s + r_ij)``.  The
    ``bond_odd_ee_*`` arrays give correlated two-center odd probes using the
    cross product of Slater decays and smooth covalent pair scales.

    Returns:
        One scalar carrier value per auxiliary probe.

    Raises:
        ValueError: If atom-odd probes are requested for a non-diatomic system.
    """
    dipole_radial_powers = (
        jnp.asarray([], dtype=point.dtype)
        if dipole_radial_powers is None
        else dipole_radial_powers
    )
    atom_odd_exponents = (
        jnp.asarray([], dtype=point.dtype)
        if atom_odd_exponents is None
        else atom_odd_exponents
    )
    atom_odd_slater_decays = (
        jnp.asarray([], dtype=point.dtype)
        if atom_odd_slater_decays is None
        else atom_odd_slater_decays
    )
    bond_odd_slater_decays = (
        jnp.asarray([], dtype=point.dtype)
        if bond_odd_slater_decays is None
        else bond_odd_slater_decays
    )
    dipole_ee_scales = (
        jnp.asarray([], dtype=point.dtype)
        if dipole_ee_scales is None
        else dipole_ee_scales
    )
    bond_odd_ee_slater_decays = (
        jnp.asarray([], dtype=point.dtype)
        if bond_odd_ee_slater_decays is None
        else bond_odd_ee_slater_decays
    )
    bond_odd_ee_scales = (
        jnp.asarray([], dtype=point.dtype)
        if bond_odd_ee_scales is None
        else bond_odd_ee_scales
    )
    if (
        exponents.size == 0
        and dipole_radial_powers.size == 0
        and atom_odd_exponents.size == 0
        and atom_odd_slater_decays.size == 0
        and bond_odd_slater_decays.size == 0
        and dipole_ee_scales.size == 0
        and bond_odd_ee_slater_decays.size == 0
    ):
        return jnp.zeros((0,), dtype=point.dtype)
    charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
    charge_center = charge_center / jnp.sum(ground.charges)
    shifted = point - charge_center[None, :]
    radius2 = jnp.sum(shifted**2, axis=1)
    pieces = []
    if exponents.size:
        pieces.append(
            jnp.sum(
                shifted[:, 2, None] * jnp.exp(-exponents[None, :] * radius2[:, None]),
                axis=0,
            )
        )
    pieces.append(
        cusp_safe_auxiliary_pwave_values(
            ground,
            point,
            dipole_radial_powers,
            dipole_radial_scale,
        )
    )
    if dipole_ee_scales.size:
        pair_features = electron_pair_features(point, dipole_ee_scales)
        pieces.append(jnp.sum(shifted[:, 2]) * pair_features)
    if (
        atom_odd_exponents.size
        or atom_odd_slater_decays.size
        or bond_odd_slater_decays.size
        or bond_odd_ee_slater_decays.size
    ):
        if ground.atoms.shape[0] != 2:
            msg = "two-center auxiliary sources require exactly two atoms"
            raise ValueError(msg)
        r2_a = jnp.sum((point - ground.atoms[0][None, :]) ** 2, axis=1)
        r2_b = jnp.sum((point - ground.atoms[1][None, :]) ** 2, axis=1)
    if atom_odd_exponents.size:
        pieces.append(
            jnp.sum(
                jnp.exp(-atom_odd_exponents[None, :] * r2_a[:, None])
                - jnp.exp(-atom_odd_exponents[None, :] * r2_b[:, None]),
                axis=0,
            )
        )
    if atom_odd_slater_decays.size:
        r_a = jnp.sqrt(r2_a + 1e-12)
        r_b = jnp.sqrt(r2_b + 1e-12)
        pieces.append(
            jnp.sum(
                jnp.exp(-atom_odd_slater_decays[None, :] * r_a[:, None])
                - jnp.exp(-atom_odd_slater_decays[None, :] * r_b[:, None]),
                axis=0,
            )
        )
    if bond_odd_slater_decays.size:
        r_a = jnp.sqrt(r2_a + 1e-12)
        r_b = jnp.sqrt(r2_b + 1e-12)
        even_bonding = jnp.exp(-r_a) + jnp.exp(-r_b) + 1e-8
        pieces.append(
            jnp.sum(
                (
                    jnp.exp(-bond_odd_slater_decays[None, :] * r_a[:, None])
                    - jnp.exp(-bond_odd_slater_decays[None, :] * r_b[:, None])
                )
                / even_bonding[:, None],
                axis=0,
            )
        )
    if bond_odd_ee_slater_decays.size:
        r_a = jnp.sqrt(r2_a + 1e-12)
        r_b = jnp.sqrt(r2_b + 1e-12)
        even_bonding = jnp.exp(-r_a) + jnp.exp(-r_b) + 1e-8
        bond_values = jnp.sum(
            (
                jnp.exp(-bond_odd_ee_slater_decays[None, :] * r_a[:, None])
                - jnp.exp(-bond_odd_ee_slater_decays[None, :] * r_b[:, None])
            )
            / even_bonding[:, None],
            axis=0,
        )
        pair_features = two_center_pair_features(
            point,
            ground.atoms,
            bond_odd_ee_scales,
        )
        pieces.append(jnp.ravel(bond_values[:, None] * pair_features[None, :]))
    return jnp.concatenate(pieces, axis=0)


def auxiliary_source_value_single(
    ground: FermiNetGround,
    point: jax.Array,
    exponents: jax.Array,
    atom_odd_exponents: jax.Array | None = None,
    atom_odd_slater_decays: jax.Array | None = None,
    bond_odd_slater_decays: jax.Array | None = None,
    dipole_ee_scales: jax.Array | None = None,
    bond_odd_ee_slater_decays: jax.Array | None = None,
    bond_odd_ee_scales: jax.Array | None = None,
    dipole_radial_powers: jax.Array | None = None,
    dipole_radial_scale: float = 1.0,
) -> jax.Array:
    """Evaluate fixed auxiliary source columns ``Psi0 * f``.

    Returns:
        One scalar source value per auxiliary probe.
    """
    carriers = auxiliary_source_carrier_value_single(
        ground,
        point,
        exponents,
        atom_odd_exponents,
        atom_odd_slater_decays,
        bond_odd_slater_decays,
        dipole_ee_scales,
        bond_odd_ee_slater_decays,
        bond_odd_ee_scales,
        dipole_radial_powers,
        dipole_radial_scale,
    )
    phase, logpsi = ground_phase_logpsi_single(ground, point)
    return carriers * phase * jnp.exp(logpsi)


def auxiliary_source_values_and_gradients(
    ground: FermiNetGround,
    points: jax.Array,
    exponents: jax.Array,
    atom_odd_exponents: jax.Array | None = None,
    atom_odd_slater_decays: jax.Array | None = None,
    bond_odd_slater_decays: jax.Array | None = None,
    dipole_ee_scales: jax.Array | None = None,
    bond_odd_ee_slater_decays: jax.Array | None = None,
    bond_odd_ee_scales: jax.Array | None = None,
    dipole_radial_powers: jax.Array | None = None,
    dipole_radial_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate auxiliary source probes and coordinate gradients.

    Returns:
        Batched auxiliary source values and their coordinate gradients.
    """
    dipole_radial_powers = (
        jnp.asarray([], dtype=points.dtype)
        if dipole_radial_powers is None
        else dipole_radial_powers
    )
    atom_odd_exponents = (
        jnp.asarray([], dtype=points.dtype)
        if atom_odd_exponents is None
        else atom_odd_exponents
    )
    atom_odd_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if atom_odd_slater_decays is None
        else atom_odd_slater_decays
    )
    bond_odd_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_slater_decays is None
        else bond_odd_slater_decays
    )
    dipole_ee_scales = (
        jnp.asarray([], dtype=points.dtype)
        if dipole_ee_scales is None
        else dipole_ee_scales
    )
    bond_odd_ee_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_ee_slater_decays is None
        else bond_odd_ee_slater_decays
    )
    bond_odd_ee_scales = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_ee_scales is None
        else bond_odd_ee_scales
    )
    if (
        exponents.size == 0
        and dipole_radial_powers.size == 0
        and atom_odd_exponents.size == 0
        and atom_odd_slater_decays.size == 0
        and bond_odd_slater_decays.size == 0
        and dipole_ee_scales.size == 0
        and bond_odd_ee_slater_decays.size == 0
    ):
        empty_values = jnp.zeros((points.shape[0], 0), dtype=points.dtype)
        empty_grads = jnp.zeros(
            (points.shape[0], 0, *ground.electron_shape), dtype=points.dtype
        )
        return empty_values, empty_grads
    values = jax.vmap(
        auxiliary_source_value_single,
        (None, 0, None, None, None, None, None, None, None, None, None),
    )(
        ground,
        points,
        exponents,
        atom_odd_exponents,
        atom_odd_slater_decays,
        bond_odd_slater_decays,
        dipole_ee_scales,
        bond_odd_ee_slater_decays,
        bond_odd_ee_scales,
        dipole_radial_powers,
        dipole_radial_scale,
    )
    gradients = jax.vmap(
        jax.jacrev(auxiliary_source_value_single, argnums=1),
        (None, 0, None, None, None, None, None, None, None, None, None),
    )(
        ground,
        points,
        exponents,
        atom_odd_exponents,
        atom_odd_slater_decays,
        bond_odd_slater_decays,
        dipole_ee_scales,
        bond_odd_ee_slater_decays,
        bond_odd_ee_scales,
        dipole_radial_powers,
        dipole_radial_scale,
    )
    return values, gradients


def auxiliary_source_values(
    ground: FermiNetGround,
    points: jax.Array,
    exponents: jax.Array,
    atom_odd_exponents: jax.Array | None = None,
    atom_odd_slater_decays: jax.Array | None = None,
    bond_odd_slater_decays: jax.Array | None = None,
    dipole_ee_scales: jax.Array | None = None,
    bond_odd_ee_slater_decays: jax.Array | None = None,
    bond_odd_ee_scales: jax.Array | None = None,
    dipole_radial_powers: jax.Array | None = None,
    dipole_radial_scale: float = 1.0,
) -> jax.Array:
    """Evaluate auxiliary source probes without coordinate gradients.

    Returns:
        Batched auxiliary source values.
    """
    dipole_radial_powers = (
        jnp.asarray([], dtype=points.dtype)
        if dipole_radial_powers is None
        else dipole_radial_powers
    )
    atom_odd_exponents = (
        jnp.asarray([], dtype=points.dtype)
        if atom_odd_exponents is None
        else atom_odd_exponents
    )
    atom_odd_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if atom_odd_slater_decays is None
        else atom_odd_slater_decays
    )
    bond_odd_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_slater_decays is None
        else bond_odd_slater_decays
    )
    dipole_ee_scales = (
        jnp.asarray([], dtype=points.dtype)
        if dipole_ee_scales is None
        else dipole_ee_scales
    )
    bond_odd_ee_slater_decays = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_ee_slater_decays is None
        else bond_odd_ee_slater_decays
    )
    bond_odd_ee_scales = (
        jnp.asarray([], dtype=points.dtype)
        if bond_odd_ee_scales is None
        else bond_odd_ee_scales
    )
    if (
        exponents.size == 0
        and dipole_radial_powers.size == 0
        and atom_odd_exponents.size == 0
        and atom_odd_slater_decays.size == 0
        and bond_odd_slater_decays.size == 0
        and dipole_ee_scales.size == 0
        and bond_odd_ee_slater_decays.size == 0
    ):
        return jnp.zeros((points.shape[0], 0), dtype=points.dtype)
    return jax.vmap(
        auxiliary_source_value_single,
        (None, 0, None, None, None, None, None, None, None, None, None),
    )(
        ground,
        points,
        exponents,
        atom_odd_exponents,
        atom_odd_slater_decays,
        bond_odd_slater_decays,
        dipole_ee_scales,
        bond_odd_ee_slater_decays,
        bond_odd_ee_scales,
        dipole_radial_powers,
        dipole_radial_scale,
    )


def _auxiliary_source_count(
    exponents: jax.Array | np.ndarray | None,
    atom_odd_exponents: jax.Array | np.ndarray | None = None,
    atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    dipole_ee_scales: jax.Array | np.ndarray | None = None,
    bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    dipole_radial_powers: jax.Array | np.ndarray | None = None,
) -> int:
    count = 0 if exponents is None else int(np.asarray(exponents).size)
    count += (
        0 if atom_odd_exponents is None else int(np.asarray(atom_odd_exponents).size)
    )
    count += (
        0
        if atom_odd_slater_decays is None
        else int(np.asarray(atom_odd_slater_decays).size)
    )
    count += (
        0
        if bond_odd_slater_decays is None
        else int(np.asarray(bond_odd_slater_decays).size)
    )
    count += 0 if dipole_ee_scales is None else int(np.asarray(dipole_ee_scales).size)
    bond_ee_decay_count = (
        0
        if bond_odd_ee_slater_decays is None
        else int(np.asarray(bond_odd_ee_slater_decays).size)
    )
    bond_ee_scale_count = (
        0 if bond_odd_ee_scales is None else int(np.asarray(bond_odd_ee_scales).size)
    )
    count += bond_ee_decay_count * bond_ee_scale_count
    count += (
        0
        if dipole_radial_powers is None
        else int(np.asarray(dipole_radial_powers).size)
    )
    return count


def ground_projection_coefficients(
    values: jax.Array,
    ground_values: jax.Array,
    density_or_q: jax.Array,
) -> jax.Array:
    """Estimate global coefficients for removing ground-state components.

    Returns:
        One projection coefficient per basis function.
    """
    weights = 1 / density_or_q / values.shape[0]
    ground_norm = jnp.maximum(
        jnp.einsum("n,n,n->", weights, ground_values, ground_values), 1e-14
    )
    return jnp.einsum("n,n,ni->i", weights, ground_values, values) / ground_norm


def apply_ground_projection_coefficients(
    values: jax.Array,
    gradients: jax.Array,
    ground_values: jax.Array,
    ground_gradients: jax.Array,
    coeff: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply fixed ``Q0`` projection coefficients to values and gradients.

    Returns:
        Projected values and projected coordinate gradients.
    """
    values = values - ground_values[:, None] * coeff[None, :]
    gradients = gradients - ground_gradients[:, None, :, :] * coeff[None, :, None, None]
    return values, gradients


def project_values_against_ground(
    values: jax.Array,
    gradients: jax.Array,
    ground_values: jax.Array,
    ground_gradients: jax.Array,
    density_or_q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the sample-estimated ``Q0`` projection to basis functions.

    Returns:
        Projected values, projected coordinate gradients, and the removed
        ground-state coefficients for each basis function.
    """
    coeff = ground_projection_coefficients(values, ground_values, density_or_q)
    values, gradients = apply_ground_projection_coefficients(
        values, gradients, ground_values, ground_gradients, coeff
    )
    return values, gradients, coeff


def potential_shift(ground: FermiNetGround, points: jax.Array) -> jax.Array:
    potential = jax.vmap(molecular_potential, (None, 0))(ground, points)
    return potential - ground.energy


def weak_matrices(
    values: jax.Array,
    gradients: jax.Array,
    pot_shift: jax.Array,
    density_or_q: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    overlap, _, hamiltonian = weak_matrices_with_kinetic(
        values, gradients, pot_shift, density_or_q
    )
    return overlap, hamiltonian


def weak_matrices_with_kinetic(
    values: jax.Array,
    gradients: jax.Array,
    pot_shift: jax.Array,
    density_or_q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    weights = 1 / density_or_q / values.shape[0]
    overlap = jnp.einsum("n,ni,nj->ij", weights, values, values)
    flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
    kinetic = 0.5 * jnp.einsum("nid,njd->nij", flat_gradients, flat_gradients)
    kinetic_matrix = jnp.einsum("n,nij->ij", weights, kinetic)
    potential = pot_shift[:, None, None] * values[:, :, None] * values[:, None, :]
    potential_matrix = jnp.einsum("n,nij->ij", weights, potential)
    return overlap, kinetic_matrix, kinetic_matrix + potential_matrix


def project_values_laplacians_against_ground(
    values: jax.Array,
    laplacians: jax.Array,
    ground_values: jax.Array,
    ground_laplacians: jax.Array,
    density_or_q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the sample-estimated ``Q0`` projection to Laplacian columns.

    Returns:
        Projected values, projected Laplacians, and removed ground coefficients.
    """
    coeff = ground_projection_coefficients(values, ground_values, density_or_q)
    values = values - ground_values[:, None] * coeff[None, :]
    laplacians = laplacians - ground_laplacians[:, None] * coeff[None, :]
    return values, laplacians, coeff


def project_values_hbar_against_ground(
    values: jax.Array,
    hbar_values: jax.Array,
    ground_values: jax.Array,
    ground_hbar: jax.Array,
    density_or_q: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply the sample-estimated ``Q0`` projection to action columns.

    Returns:
        Projected values, projected action columns, and removed coefficients.
    """
    coeff = ground_projection_coefficients(values, ground_values, density_or_q)
    values = values - ground_values[:, None] * coeff[None, :]
    hbar_values = hbar_values - ground_hbar[:, None] * coeff[None, :]
    return values, hbar_values, coeff


def strong_action_metric_matrices_from_hbar(
    values: jax.Array,
    hbar_values: jax.Array,
    density_or_q: jax.Array,
    *,
    omegas: jax.Array,
    eta: float,
    clip: float = float("inf"),
) -> jax.Array:
    """Estimate ``<A_z chi_i, A_z chi_j>`` for candidate action control.

    ``A_z`` uses the same sign convention as the strong-residual audit:
    ``A_z chi = z chi - (H - E0) chi``.  The optional clip is a per-sample
    robust weight on the largest action-column magnitude and is disabled when
    non-finite or non-positive.

    Returns:
        One Hermitian action Gram matrix per requested frequency.
    """
    z_values = jnp.asarray(omegas, dtype=values.dtype) + 1j * float(eta)
    action = z_values[:, None, None] * values[None, :, :] - hbar_values[None, :, :]
    weights = 1 / density_or_q / values.shape[0]
    if np.isfinite(float(clip)) and float(clip) > 0.0:
        clip_sq = jnp.asarray(float(clip) ** 2, dtype=values.dtype)
        action_power = jnp.max(jnp.abs(action) ** 2, axis=(0, 2))
        robust_weight = jnp.minimum(1.0, clip_sq / jnp.maximum(action_power, clip_sq))
        weights = weights * robust_weight
    metric = jnp.einsum("n,lni,lnj->lij", weights, jnp.conj(action), action)
    return (metric + jnp.conj(jnp.swapaxes(metric, 1, 2))) / 2


def strong_action_metric_matrices(
    values: jax.Array,
    laplacians: jax.Array,
    pot_shift: jax.Array,
    density_or_q: jax.Array,
    *,
    omegas: jax.Array,
    eta: float,
    clip: float = float("inf"),
) -> jax.Array:
    """Estimate ``<A_z chi_i, A_z chi_j>`` from explicit Laplacians.

    Returns:
        One Hermitian action Gram matrix per requested frequency.
    """
    hbar_values = -0.5 * laplacians + pot_shift[:, None] * values
    return strong_action_metric_matrices_from_hbar(
        values,
        hbar_values,
        density_or_q,
        omegas=omegas,
        eta=eta,
        clip=clip,
    )


def _normalized_source_weights(
    source_count: int,
    source_weights: jax.Array | np.ndarray | None,
    dtype: jnp.dtype,
) -> jax.Array:
    if source_weights is None:
        return jnp.ones((source_count,), dtype=dtype) / source_count
    weights = jnp.asarray(source_weights, dtype=dtype)
    return weights / jnp.sum(weights)


def _action_region_masks(
    ground: FermiNetGround,
    points: jax.Array,
    ground_values: jax.Array,
    action_values: jax.Array,
) -> jax.Array:
    """Return all/node/tail/cusp/high-action masks for action-risk training."""
    sample_count = points.shape[0]
    all_mask = jnp.ones((sample_count,), dtype=points.dtype)
    ground_abs = jnp.abs(ground_values)
    node_threshold = jnp.quantile(ground_abs, 0.20)
    node_mask = ground_abs <= node_threshold
    charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
    charge_center = charge_center / jnp.sum(ground.charges)
    shifted = points - charge_center[None, None, :]
    tail_radius = jnp.sqrt(jnp.mean(jnp.sum(shifted**2, axis=-1), axis=1) + 1e-12)
    tail_threshold = jnp.quantile(tail_radius, 0.80)
    tail_mask = tail_radius >= tail_threshold
    electron_atom = points[:, :, None, :] - ground.atoms[None, None, :, :]
    min_en = jnp.min(jnp.sqrt(jnp.sum(electron_atom**2, axis=-1) + 1e-24), axis=(1, 2))
    en_cusp_mask = min_en <= jnp.asarray(0.25, dtype=points.dtype)
    if points.shape[1] < 2:
        ee_cusp_mask = jnp.zeros((sample_count,), dtype=bool)
    else:
        ee_vec = points[:, :, None, :] - points[:, None, :, :]
        ee_dist = jnp.sqrt(jnp.sum(ee_vec**2, axis=-1) + 1e-24)
        ee_dist = (
            ee_dist + jnp.eye(points.shape[1], dtype=points.dtype)[None, :, :] * 1e6
        )
        min_ee = jnp.min(ee_dist, axis=(1, 2))
        ee_cusp_mask = min_ee <= jnp.asarray(0.25, dtype=points.dtype)
    action_power = jnp.max(jnp.abs(action_values) ** 2, axis=(0, 2))
    action_threshold = jnp.quantile(action_power, 0.80)
    high_action_mask = action_power >= action_threshold
    masks = jnp.stack(
        [
            all_mask,
            node_mask.astype(points.dtype),
            tail_mask.astype(points.dtype),
            en_cusp_mask.astype(points.dtype),
            ee_cusp_mask.astype(points.dtype),
            high_action_mask.astype(points.dtype),
        ],
        axis=0,
    )
    return masks


def _weighted_action_epsilon2(
    action_columns: jax.Array,
    source_values: jax.Array,
    region_weights: jax.Array,
    *,
    source_weights: jax.Array,
    delta: float,
) -> jax.Array:
    """Solve weighted action least squares and return epsilon^2 per region/frequency.

    Returns:
        Matrix of relative residual variances with shape ``(regions, omegas)``.
    """
    basis_count = int(action_columns.shape[2])
    eye = jnp.eye(basis_count, dtype=action_columns.dtype)
    weights = region_weights / jnp.maximum(
        jnp.sum(region_weights, axis=1, keepdims=True),
        float(delta),
    )

    def one_region(region_weight: jax.Array) -> jax.Array:
        gram = jnp.einsum(
            "n,lni,lnj->lij",
            region_weight,
            jnp.conj(action_columns),
            action_columns,
        )
        gram = (gram + jnp.conj(jnp.swapaxes(gram, 1, 2))) / 2
        rhs = jnp.einsum(
            "n,lni,na->lia",
            region_weight,
            jnp.conj(action_columns),
            source_values,
        )
        coeffs = jax.vmap(jnp.linalg.solve)(
            gram + float(delta) * eye[None, :, :],
            rhs,
        )
        fit = jnp.einsum("lni,lia->lna", action_columns, coeffs)
        residual = source_values[None, :, :] - fit
        residual2 = jnp.real(
            jnp.einsum(
                "n,lna,lna->la",
                region_weight,
                jnp.conj(residual),
                residual,
            )
        )
        source_norms = jnp.maximum(
            jnp.real(
                jnp.einsum(
                    "n,na,na->a",
                    region_weight,
                    jnp.conj(source_values),
                    source_values,
                )
            ),
            float(delta),
        )
        return jnp.sum(
            source_weights[None, :] * residual2 / source_norms[None, :],
            axis=1,
        )

    return jax.vmap(one_region)(weights)


def residual_aligned_seed_coefficients_from_columns(
    old_values: jax.Array,
    old_hbar: jax.Array,
    bank_values: jax.Array,
    bank_hbar: jax.Array,
    density: jax.Array,
    *,
    source_count: int,
    active_heads: int,
    candidate_heads: int,
    omegas: jax.Array,
    omega_weights: jax.Array,
    source_weights: jax.Array | np.ndarray | None,
    eta: float,
    delta: float,
    rank_rtol: float = OFFICIAL_SOURCE_LIFT_SEED_RANK_RTOL,
    rank_atol: float = OFFICIAL_SOURCE_LIFT_SEED_RANK_ATOL,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select residual-aligned seed directions from precomputed action columns.

    Returns:
        Bank coefficient matrix and action-novelty diagnostics.
    """
    old_end = int(source_count) + int(active_heads)
    values_np = np.asarray(jax.device_get(old_values[:, :old_end]))
    hbar_np = np.asarray(jax.device_get(old_hbar[:, :old_end]))
    source_np = values_np[:, :source_count]
    bank_values_np = np.asarray(jax.device_get(bank_values))
    bank_hbar_np = np.asarray(jax.device_get(bank_hbar))
    density_np = np.asarray(jax.device_get(density), dtype=np.float64)
    weights = 1.0 / density_np / max(1, int(density_np.size))
    omegas_np = np.asarray(jax.device_get(omegas), dtype=np.float64)
    omega_weights_np = np.asarray(jax.device_get(omega_weights), dtype=np.float64)
    omega_weights_np = omega_weights_np / np.sum(omega_weights_np)
    source_weights_np = np.asarray(
        jax.device_get(
            _normalized_source_weights(
                source_count,
                source_weights,
                old_values.dtype,
            )
        ),
        dtype=np.float64,
    )
    feature_count = int(bank_values_np.shape[1])
    rank_empty = {
        "seed_action_rank": 0.0,
        "seed_top_capture": 0.0,
        "seed_trace": 0.0,
        "seed_selected": 0.0,
        "seed_coeff_norm_min": 0.0,
        "seed_coeff_norm_max": 0.0,
        "seed_invalid": 1.0,
    }
    if feature_count < 1:
        return np.zeros((0, int(candidate_heads))), rank_empty
    action_metric = np.zeros((feature_count, feature_count), dtype=np.float64)
    capture_metric = np.zeros((feature_count, feature_count), dtype=np.float64)
    eye_old = np.eye(old_end, dtype=np.complex128)
    for omega, omega_weight in zip(omegas_np, omega_weights_np, strict=True):
        z_value = omega + 1j * float(eta)
        old_action = z_value * values_np - hbar_np
        bank_action = z_value * bank_values_np - bank_hbar_np
        weighted_old = weights[:, None] * old_action
        old_gram = old_action.conj().T @ weighted_old
        old_gram = (old_gram + old_gram.conj().T) / 2
        regularized = old_gram + float(delta) * eye_old
        old_source_rhs = old_action.conj().T @ (weights[:, None] * source_np)
        source_coeff = np.linalg.solve(regularized, old_source_rhs)
        residual = source_np - old_action @ source_coeff
        bank_rhs = old_action.conj().T @ (weights[:, None] * bank_action)
        bank_coeff = np.linalg.solve(regularized, bank_rhs)
        bank_perp = bank_action - old_action @ bank_coeff
        weighted_bank = weights[:, None] * bank_perp
        action_metric += float(omega_weight) * np.real(
            bank_perp.conj().T @ weighted_bank
        )
        residual_rhs = bank_perp.conj().T @ (weights[:, None] * residual)
        residual_norm = np.real(
            np.sum(
                weights[:, None] * np.conj(residual) * residual,
                axis=0,
            )
        )
        residual_norm = np.maximum(residual_norm, float(delta))
        for source_idx in range(source_count):
            rhs = residual_rhs[:, source_idx]
            capture_metric += (
                float(omega_weight)
                * float(source_weights_np[source_idx])
                * np.real(np.outer(rhs, np.conj(rhs)) / residual_norm[source_idx])
            )
    action_metric = (action_metric + action_metric.T) / 2
    capture_metric = (capture_metric + capture_metric.T) / 2
    evals, evecs = np.linalg.eigh(action_metric)
    if evals.size == 0:
        return np.zeros((feature_count, int(candidate_heads))), rank_empty
    rank_floor = max(float(rank_atol), float(rank_rtol) * float(np.max(evals)))
    keep = evals > rank_floor
    action_rank = int(np.sum(keep))
    action_trace = float(np.sum(np.maximum(evals, 0.0)))
    if action_rank < 1:
        return np.zeros((feature_count, int(candidate_heads))), {
            **rank_empty,
            "seed_trace": action_trace,
        }
    basis = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    capture_hat = basis.T @ capture_metric @ basis
    capture_hat = (capture_hat + capture_hat.T) / 2
    capture_evals, capture_evecs = np.linalg.eigh(capture_hat)
    order = np.argsort(capture_evals)[::-1]
    selected = min(int(candidate_heads), int(order.size))
    coeff = np.zeros((feature_count, int(candidate_heads)), dtype=np.float64)
    if selected > 0:
        coeff[:, :selected] = basis @ capture_evecs[:, order[:selected]]
    coeff_norms = np.linalg.norm(coeff[:, : max(1, selected)], axis=0)
    top_capture = float(max(capture_evals[order[0]], 0.0)) if int(order.size) else 0.0
    invalid = bool(
        action_rank < 1
        or selected < 1
        or top_capture < OFFICIAL_SOURCE_LIFT_SEED_CAPTURE_FLOOR
    )
    return coeff, {
        "seed_action_rank": float(action_rank),
        "seed_top_capture": top_capture,
        "seed_trace": action_trace,
        "seed_selected": float(selected),
        "seed_coeff_norm_min": float(np.min(coeff_norms)) if coeff_norms.size else 0.0,
        "seed_coeff_norm_max": float(np.max(coeff_norms)) if coeff_norms.size else 0.0,
        "seed_invalid": float(invalid),
    }


def residual_aligned_source_lift_seed_coefficients(
    fixed_basis_params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    *,
    active_heads: int,
    candidate_heads: int,
    omegas: jax.Array,
    omega_weights: jax.Array,
    source_weights: jax.Array | np.ndarray | None,
    eta: float,
    delta: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    rank_rtol: float = OFFICIAL_SOURCE_LIFT_SEED_RANK_RTOL,
    rank_atol: float = OFFICIAL_SOURCE_LIFT_SEED_RANK_ATOL,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select source-lift directions by held-in action-residual alignment.

    Returns:
        Coefficient matrix with shape ``(source_lift_features, candidate_heads)``
        and seed novelty diagnostics.

    Raises:
        ValueError: If the source-lift bank shape is inconsistent.
    """
    old_values, old_hbar = source_aux_and_head_values_and_hbar(
        fixed_basis_params,
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=active_heads,
    )
    bank_values, bank_hbar = source_lift_bank_values_and_hbar(ground, points)
    ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    ground_hbar = ground_hbar_values(ground, points, ground_values)
    old_values, old_hbar, _ = project_values_hbar_against_ground(
        old_values,
        old_hbar,
        ground_values,
        ground_hbar,
        density,
    )
    bank_values, bank_hbar, _ = project_values_hbar_against_ground(
        bank_values,
        bank_hbar,
        ground_values,
        ground_hbar,
        density,
    )
    aux_source_count = _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    source_count = 1 + aux_source_count
    old_end = source_count + int(active_heads)
    values_np = np.asarray(jax.device_get(old_values[:, :old_end]))
    hbar_np = np.asarray(jax.device_get(old_hbar[:, :old_end]))
    source_np = values_np[:, :source_count]
    bank_values_np = np.asarray(jax.device_get(bank_values))
    bank_hbar_np = np.asarray(jax.device_get(bank_hbar))
    density_np = np.asarray(jax.device_get(density), dtype=np.float64)
    weights = 1.0 / density_np / max(1, int(density_np.size))
    omegas_np = np.asarray(jax.device_get(omegas), dtype=np.float64)
    omega_weights_np = np.asarray(jax.device_get(omega_weights), dtype=np.float64)
    omega_weights_np = omega_weights_np / np.sum(omega_weights_np)
    source_weights_np = np.asarray(
        jax.device_get(
            _normalized_source_weights(
                source_count,
                source_weights,
                old_values.dtype,
            )
        ),
        dtype=np.float64,
    )
    feature_count = source_lift_feature_count()
    rank_empty = {
        "seed_action_rank": 0.0,
        "seed_top_capture": 0.0,
        "seed_trace": 0.0,
        "seed_selected": 0.0,
        "seed_coeff_norm_min": 0.0,
        "seed_coeff_norm_max": 0.0,
        "seed_invalid": 1.0,
    }
    if bank_values_np.shape[1] != feature_count:
        msg = "source-lift bank feature count mismatch"
        raise ValueError(msg)
    action_metric = np.zeros((feature_count, feature_count), dtype=np.float64)
    capture_metric = np.zeros((feature_count, feature_count), dtype=np.float64)
    eye_old = np.eye(old_end, dtype=np.complex128)
    for omega, omega_weight in zip(omegas_np, omega_weights_np, strict=True):
        z_value = omega + 1j * float(eta)
        old_action = z_value * values_np - hbar_np
        bank_action = z_value * bank_values_np - bank_hbar_np
        weighted_old = weights[:, None] * old_action
        old_gram = old_action.conj().T @ weighted_old
        old_gram = (old_gram + old_gram.conj().T) / 2
        regularized = old_gram + float(delta) * eye_old
        old_source_rhs = old_action.conj().T @ (weights[:, None] * source_np)
        source_coeff = np.linalg.solve(regularized, old_source_rhs)
        residual = source_np - old_action @ source_coeff
        bank_rhs = old_action.conj().T @ (weights[:, None] * bank_action)
        bank_coeff = np.linalg.solve(regularized, bank_rhs)
        bank_perp = bank_action - old_action @ bank_coeff
        weighted_bank = weights[:, None] * bank_perp
        action_metric += float(omega_weight) * np.real(
            bank_perp.conj().T @ weighted_bank
        )
        residual_rhs = bank_perp.conj().T @ (weights[:, None] * residual)
        residual_norm = np.real(
            np.sum(
                weights[:, None] * np.conj(residual) * residual,
                axis=0,
            )
        )
        residual_norm = np.maximum(residual_norm, float(delta))
        for source_idx in range(source_count):
            rhs = residual_rhs[:, source_idx]
            capture_metric += (
                float(omega_weight)
                * float(source_weights_np[source_idx])
                * np.real(np.outer(rhs, np.conj(rhs)) / residual_norm[source_idx])
            )
    action_metric = (action_metric + action_metric.T) / 2
    capture_metric = (capture_metric + capture_metric.T) / 2
    evals, evecs = np.linalg.eigh(action_metric)
    if evals.size == 0:
        return np.zeros((feature_count, int(candidate_heads))), rank_empty
    rank_floor = max(float(rank_atol), float(rank_rtol) * float(np.max(evals)))
    keep = evals > rank_floor
    action_rank = int(np.sum(keep))
    action_trace = float(np.sum(np.maximum(evals, 0.0)))
    if action_rank < 1:
        return np.zeros((feature_count, int(candidate_heads))), {
            **rank_empty,
            "seed_trace": action_trace,
        }
    basis = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    capture_hat = basis.T @ capture_metric @ basis
    capture_hat = (capture_hat + capture_hat.T) / 2
    capture_evals, capture_evecs = np.linalg.eigh(capture_hat)
    order = np.argsort(capture_evals)[::-1]
    selected = min(int(candidate_heads), int(order.size))
    coeff = np.zeros((feature_count, int(candidate_heads)), dtype=np.float64)
    if selected > 0:
        coeff[:, :selected] = basis @ capture_evecs[:, order[:selected]]
    coeff_norms = np.linalg.norm(coeff[:, : max(1, selected)], axis=0)
    top_capture = float(max(capture_evals[order[0]], 0.0)) if int(order.size) else 0.0
    invalid = bool(
        action_rank < 1
        or selected < 1
        or top_capture < OFFICIAL_SOURCE_LIFT_SEED_CAPTURE_FLOOR
    )
    diagnostics = {
        "seed_action_rank": float(action_rank),
        "seed_top_capture": top_capture,
        "seed_trace": action_trace,
        "seed_selected": float(selected),
        "seed_coeff_norm_min": float(np.min(coeff_norms)) if coeff_norms.size else 0.0,
        "seed_coeff_norm_max": float(np.max(coeff_norms)) if coeff_norms.size else 0.0,
        "seed_invalid": float(invalid),
    }
    return coeff, diagnostics


def gauge_stabilized_enrichment_objective_from_columns(
    values: jax.Array,
    action_values: jax.Array,
    density: jax.Array,
    region_masks: jax.Array,
    *,
    active_heads: int,
    candidate_heads: int,
    source_count: int,
    omega_weights: jax.Array,
    source_weights: jax.Array | np.ndarray | None = None,
    lambda_rough: float,
    delta: float,
    norm_floor: float = 1e-8,
    softmax_kappa: float = 8.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Gauge-fixed region-balanced action residual risk for candidate blocks.

    Returns:
        Scalar loss and action-risk diagnostics.
    """
    old_end = int(source_count) + int(active_heads)
    candidate_start = old_end
    candidate_end = candidate_start + int(candidate_heads)
    weights = 1 / density / values.shape[0]
    source_values = values[:, :source_count]
    old_values = values[:, :old_end]
    candidate_values = values[:, candidate_start:candidate_end]
    old_actions = action_values[:, :, :old_end]
    candidate_actions = action_values[:, :, candidate_start:candidate_end]
    omega_weights = jnp.asarray(omega_weights, dtype=weights.dtype)
    normalized_omega_weights = omega_weights / jnp.sum(omega_weights)
    normalized_source_weights = _normalized_source_weights(
        source_count,
        source_weights,
        weights.dtype,
    )
    value_bb = jnp.einsum("n,ni,nj->ij", weights, old_values, old_values)
    value_bc = jnp.einsum("n,ni,nc->ic", weights, old_values, candidate_values)
    value_cc = jnp.einsum("n,nc,nd->cd", weights, candidate_values, candidate_values)
    action_bb = jnp.real(
        jnp.einsum(
            "l,n,lni,lnj->ij",
            normalized_omega_weights,
            weights,
            jnp.conj(old_actions),
            old_actions,
        )
    )
    action_bc = jnp.real(
        jnp.einsum(
            "l,n,lni,lnc->ic",
            normalized_omega_weights,
            weights,
            jnp.conj(old_actions),
            candidate_actions,
        )
    )
    action_cc = jnp.real(
        jnp.einsum(
            "l,n,lnc,lnd->cd",
            normalized_omega_weights,
            weights,
            jnp.conj(candidate_actions),
            candidate_actions,
        )
    )
    old_metric = (value_bb + action_bb + (value_bb + action_bb).T) / 2
    old_candidate_metric = value_bc + action_bc
    candidate_metric_raw = (value_cc + action_cc + (value_cc + action_cc).T) / 2
    old_eye = jnp.eye(old_end, dtype=values.dtype)
    candidate_eye = jnp.eye(candidate_heads, dtype=values.dtype)
    old_coeff = jax.lax.stop_gradient(
        jnp.linalg.solve(
            old_metric + float(delta) * old_eye,
            old_candidate_metric,
        )
    )
    candidate_values_perp = candidate_values - old_values @ old_coeff
    candidate_actions_perp = candidate_actions - jnp.einsum(
        "lni,ic->lnc",
        old_actions,
        old_coeff,
    )
    candidate_metric = jnp.real(
        jnp.einsum("n,nc,nd->cd", weights, candidate_values_perp, candidate_values_perp)
        + jnp.einsum(
            "l,n,lnc,lnd->cd",
            normalized_omega_weights,
            weights,
            jnp.conj(candidate_actions_perp),
            candidate_actions_perp,
        )
    )
    candidate_metric = (candidate_metric + candidate_metric.T) / 2
    metric_eigvals, metric_eigvecs = jnp.linalg.eigh(
        candidate_metric + float(delta) * candidate_eye
    )
    metric_eigvals = jnp.maximum(metric_eigvals, float(norm_floor))
    invsqrt = jax.lax.stop_gradient(
        (metric_eigvecs / jnp.sqrt(metric_eigvals)[None, :]) @ metric_eigvecs.T
    )
    candidate_values_hat = candidate_values_perp @ invsqrt
    candidate_actions_hat = jnp.einsum("lnc,cd->lnd", candidate_actions_perp, invsqrt)
    new_actions = jnp.concatenate([old_actions, candidate_actions_hat], axis=2)
    old_epsilon2 = _weighted_action_epsilon2(
        old_actions,
        source_values,
        weights[None, :] * region_masks,
        source_weights=normalized_source_weights,
        delta=delta,
    )
    new_epsilon2 = _weighted_action_epsilon2(
        new_actions,
        source_values,
        weights[None, :] * region_masks,
        source_weights=normalized_source_weights,
        delta=delta,
    )
    log_relative = jnp.log(new_epsilon2 + float(delta)) - jnp.log(
        old_epsilon2 + float(delta)
    )
    weighted_log_relative = log_relative + jnp.log(normalized_omega_weights)[None, :]
    softmax_kappa = float(softmax_kappa)
    action_risk = (
        jax.nn.logsumexp(softmax_kappa * weighted_log_relative) / softmax_kappa
    )
    action_trace = jnp.maximum(
        jnp.real(jnp.trace(candidate_metric_raw) / max(1, int(candidate_heads))),
        0.0,
    )
    roughness_penalty = jnp.log1p(action_trace)
    candidate_hat_gram = jnp.einsum(
        "n,nc,nd->cd",
        weights,
        candidate_values_hat,
        candidate_values_hat,
    )
    candidate_diag = jnp.maximum(jnp.real(jnp.diag(candidate_hat_gram)), float(delta))
    normalized_candidate_overlap = candidate_hat_gram / jnp.sqrt(
        candidate_diag[:, None] * candidate_diag[None, :]
    )
    block_penalty = jnp.real(
        jnp.mean((normalized_candidate_overlap - candidate_eye) ** 2)
    )
    trace_metric = jnp.maximum(
        jnp.real(jnp.trace(candidate_metric) / max(1, int(candidate_heads))),
        float(delta),
    )
    norm_floor_penalty = (
        jnp.maximum(
            0.0,
            jnp.log(float(norm_floor)) - jnp.log(trace_metric),
        )
        ** 2
    )
    loss = (
        action_risk
        + float(lambda_rough) * roughness_penalty
        + 0.05 * block_penalty
        + norm_floor_penalty
    )
    all_old = old_epsilon2[0]
    all_new = new_epsilon2[0]
    weighted_old = jnp.sum(normalized_omega_weights * all_old)
    weighted_new = jnp.sum(normalized_omega_weights * all_new)
    capture = weighted_old - weighted_new
    relative_epsilon2_improvement = capture / jnp.maximum(weighted_old, float(delta))
    redundancy = jnp.real(
        jnp.trace(candidate_metric_raw - candidate_metric)
        / jnp.maximum(jnp.trace(candidate_metric_raw), float(delta))
    )
    objective = -action_risk
    return loss, {
        "loss": loss,
        "objective": objective,
        "capture": capture,
        "redundancy": redundancy,
        "roughness": action_trace,
        "block_penalty": block_penalty,
        "sobolev_metric_weight": jnp.asarray(0.0, dtype=weights.dtype),
        "graph_metric_eta": jnp.asarray(0.0, dtype=weights.dtype),
        "local_action_metric_weight": jnp.asarray(1.0, dtype=weights.dtype),
        "local_action_trace": action_trace,
        "action_old_epsilon": jnp.sqrt(weighted_old),
        "action_new_epsilon": jnp.sqrt(weighted_new),
        "action_relative_epsilon2_improvement": relative_epsilon2_improvement,
        "gauge_metric_trace": trace_metric,
        "gauge_norm_floor_penalty": norm_floor_penalty,
        "region_action_risk": action_risk,
    }


def gauge_stabilized_enrichment_training_loss(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    *,
    active_heads: int,
    candidate_heads: int,
    omegas: jax.Array,
    omega_weights: jax.Array,
    source_weights: jax.Array | np.ndarray | None = None,
    eta: float,
    lambda_rough: float,
    delta: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Gauge-stabilized source-sector action-risk enrichment loss.

    Returns:
        Scalar loss and diagnostics for the candidate block.
    """
    values, hbar_values = source_aux_and_head_values_and_hbar(
        params,
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=active_heads + candidate_heads,
    )
    ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    ground_hbar = ground_hbar_values(ground, points, ground_values)
    values, hbar_values, _ = project_values_hbar_against_ground(
        values,
        hbar_values,
        ground_values,
        ground_hbar,
        density,
    )
    z_values = jnp.asarray(omegas, dtype=values.dtype) + 1j * float(eta)
    action_values = (
        z_values[:, None, None] * values[None, :, :] - hbar_values[None, :, :]
    )
    region_masks = _action_region_masks(ground, points, ground_values, action_values)
    aux_source_count = _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    return gauge_stabilized_enrichment_objective_from_columns(
        values,
        action_values,
        density,
        region_masks,
        active_heads=active_heads,
        candidate_heads=candidate_heads,
        source_count=1 + aux_source_count,
        omega_weights=omega_weights,
        source_weights=source_weights,
        lambda_rough=lambda_rough,
        delta=delta,
    )


def gauge_normalize_candidate_block(
    fixed_basis_params: Params,
    candidate_params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    *,
    active_heads: int,
    candidate_heads: int,
    omegas: jax.Array,
    eta: float,
    delta: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> Params:
    """Return candidate params with the gauge whitening stored as a transform."""
    params = append_response_block_params(fixed_basis_params, candidate_params)
    values, hbar_values = source_aux_and_head_values_and_hbar(
        params,
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=active_heads + candidate_heads,
    )
    ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    ground_hbar = ground_hbar_values(ground, points, ground_values)
    values, hbar_values, _ = project_values_hbar_against_ground(
        values,
        hbar_values,
        ground_values,
        ground_hbar,
        density,
    )
    z_values = jnp.asarray(omegas, dtype=values.dtype) + 1j * float(eta)
    action_values = (
        z_values[:, None, None] * values[None, :, :] - hbar_values[None, :, :]
    )
    aux_source_count = _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    old_end = 1 + aux_source_count + int(active_heads)
    candidate_start = old_end
    candidate_end = candidate_start + int(candidate_heads)
    weights = 1 / density / values.shape[0]
    normalized_omega_weights = jnp.ones_like(jnp.asarray(omegas)) / len(omegas)
    old_values = values[:, :old_end]
    candidate_values = values[:, candidate_start:candidate_end]
    old_actions = action_values[:, :, :old_end]
    candidate_actions = action_values[:, :, candidate_start:candidate_end]
    value_bb = jnp.einsum("n,ni,nj->ij", weights, old_values, old_values)
    value_bc = jnp.einsum("n,ni,nc->ic", weights, old_values, candidate_values)
    value_cc = jnp.einsum("n,nc,nd->cd", weights, candidate_values, candidate_values)
    action_bb = jnp.real(
        jnp.einsum(
            "l,n,lni,lnj->ij",
            normalized_omega_weights,
            weights,
            jnp.conj(old_actions),
            old_actions,
        )
    )
    action_bc = jnp.real(
        jnp.einsum(
            "l,n,lni,lnc->ic",
            normalized_omega_weights,
            weights,
            jnp.conj(old_actions),
            candidate_actions,
        )
    )
    action_cc = jnp.real(
        jnp.einsum(
            "l,n,lnc,lnd->cd",
            normalized_omega_weights,
            weights,
            jnp.conj(candidate_actions),
            candidate_actions,
        )
    )
    old_metric = (value_bb + action_bb + (value_bb + action_bb).T) / 2
    old_candidate_metric = value_bc + action_bc
    old_eye = jnp.eye(old_end, dtype=values.dtype)
    candidate_eye = jnp.eye(candidate_heads, dtype=values.dtype)
    old_coeff = jnp.linalg.solve(
        old_metric + float(delta) * old_eye,
        old_candidate_metric,
    )
    raw_metric = (value_cc + action_cc + (value_cc + action_cc).T) / 2
    old_component = old_candidate_metric.T @ old_coeff
    candidate_metric = raw_metric - old_component
    candidate_metric = (candidate_metric + candidate_metric.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(candidate_metric + float(delta) * candidate_eye)
    eigvals = jnp.maximum(eigvals, 1e-8)
    right_transform = (eigvecs / jnp.sqrt(eigvals)[None, :]) @ eigvecs.T
    return right_transform_response_heads(
        candidate_params,
        np.asarray(right_transform, dtype=np.float64),
        raw_head_count=num_response_heads(ground),
    )


def source_aux_and_head_values_and_gradients(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    head_count: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    source, source_grad = source_values_and_gradients(ground, points)
    aux_exponents = jnp.asarray(
        [] if aux_source_exponents is None else aux_source_exponents,
        dtype=points.dtype,
    )
    aux_dipole_radial_powers = jnp.asarray(
        []
        if aux_source_dipole_radial_powers is None
        else aux_source_dipole_radial_powers,
        dtype=points.dtype,
    )
    aux_atom_odd_exponents = jnp.asarray(
        [] if aux_source_atom_odd_exponents is None else aux_source_atom_odd_exponents,
        dtype=points.dtype,
    )
    aux_atom_odd_slater_decays = jnp.asarray(
        []
        if aux_source_atom_odd_slater_decays is None
        else aux_source_atom_odd_slater_decays,
        dtype=points.dtype,
    )
    aux_bond_odd_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_slater_decays is None
        else aux_source_bond_odd_slater_decays,
        dtype=points.dtype,
    )
    aux_dipole_ee_scales = jnp.asarray(
        [] if aux_source_dipole_ee_scales is None else aux_source_dipole_ee_scales,
        dtype=points.dtype,
    )
    aux_bond_odd_ee_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_ee_slater_decays is None
        else aux_source_bond_odd_ee_slater_decays,
        dtype=points.dtype,
    )
    aux_bond_odd_ee_scales = jnp.asarray(
        [] if aux_source_bond_odd_ee_scales is None else aux_source_bond_odd_ee_scales,
        dtype=points.dtype,
    )
    aux_values, aux_grads = auxiliary_source_values_and_gradients(
        ground,
        points,
        aux_exponents,
        aux_atom_odd_exponents,
        aux_atom_odd_slater_decays,
        aux_bond_odd_slater_decays,
        aux_dipole_ee_scales,
        aux_bond_odd_ee_slater_decays,
        aux_bond_odd_ee_scales,
        aux_dipole_radial_powers,
        aux_source_dipole_radial_scale,
    )
    if head_count == 0:
        heads = jnp.zeros((points.shape[0], 0), dtype=points.dtype)
        head_grads = jnp.zeros(
            (points.shape[0], 0, *ground.electron_shape), dtype=points.dtype
        )
    else:
        heads, head_grads = response_values_and_gradients(params, ground, points)
    if head_count is not None and head_count > 0:
        heads = heads[:, :head_count]
        head_grads = head_grads[:, :head_count]
    values = jnp.concatenate([source[:, None], aux_values, heads], axis=1)
    gradients = jnp.concatenate(
        [source_grad[:, None, :], aux_grads, head_grads], axis=1
    )
    return values, gradients, source


def source_aux_and_head_value_single(
    params: Params,
    ground: FermiNetGround,
    point: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    head_count: int | None = None,
) -> jax.Array:
    """Evaluate one unprojected BF-NKSR source/head basis value vector.

    Returns:
        One value per physical source, auxiliary source, and retained head.
    """
    source = source_value_single(ground, point)
    aux_exponents = jnp.asarray(
        [] if aux_source_exponents is None else aux_source_exponents,
        dtype=point.dtype,
    )
    aux_dipole_radial_powers = jnp.asarray(
        []
        if aux_source_dipole_radial_powers is None
        else aux_source_dipole_radial_powers,
        dtype=point.dtype,
    )
    aux_atom_odd_exponents = jnp.asarray(
        [] if aux_source_atom_odd_exponents is None else aux_source_atom_odd_exponents,
        dtype=point.dtype,
    )
    aux_atom_odd_slater_decays = jnp.asarray(
        []
        if aux_source_atom_odd_slater_decays is None
        else aux_source_atom_odd_slater_decays,
        dtype=point.dtype,
    )
    aux_bond_odd_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_slater_decays is None
        else aux_source_bond_odd_slater_decays,
        dtype=point.dtype,
    )
    aux_dipole_ee_scales = jnp.asarray(
        [] if aux_source_dipole_ee_scales is None else aux_source_dipole_ee_scales,
        dtype=point.dtype,
    )
    aux_bond_odd_ee_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_ee_slater_decays is None
        else aux_source_bond_odd_ee_slater_decays,
        dtype=point.dtype,
    )
    aux_bond_odd_ee_scales = jnp.asarray(
        [] if aux_source_bond_odd_ee_scales is None else aux_source_bond_odd_ee_scales,
        dtype=point.dtype,
    )
    aux_values = auxiliary_source_value_single(
        ground,
        point,
        aux_exponents,
        aux_atom_odd_exponents,
        aux_atom_odd_slater_decays,
        aux_bond_odd_slater_decays,
        aux_dipole_ee_scales,
        aux_bond_odd_ee_slater_decays,
        aux_bond_odd_ee_scales,
        aux_dipole_radial_powers,
        aux_source_dipole_radial_scale,
    )
    if head_count == 0:
        heads = jnp.zeros((0,), dtype=point.dtype)
    else:
        heads = response_value_single(params, ground, point)
        if head_count is not None:
            heads = heads[:head_count]
    return jnp.concatenate([source[None], aux_values, heads], axis=0)


def source_aux_carrier_value_single(
    ground: FermiNetGround,
    point: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> jax.Array:
    """Evaluate source and auxiliary carriers ``f`` in columns ``Psi0 * f``.

    Returns:
        One carrier value per physical and auxiliary source column.
    """
    source = source_carrier_value_single(ground, point)
    aux_exponents = jnp.asarray(
        [] if aux_source_exponents is None else aux_source_exponents,
        dtype=point.dtype,
    )
    aux_values = auxiliary_source_carrier_value_single(
        ground,
        point,
        aux_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale,
    )
    return jnp.concatenate([source[None], aux_values], axis=0)


def source_aux_carrier_values_and_hbar(
    ground: FermiNetGround,
    points: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate source/aux values and ``(H-E0)`` action by product rule.

    Returns:
        Batched source/aux column values and shifted Hamiltonian actions.
    """

    def carrier_fn(point: jax.Array) -> jax.Array:
        return source_aux_carrier_value_single(
            ground,
            point,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        )

    carrier_values = jax.vmap(carrier_fn)(points)
    carrier_gradients = jax.vmap(jax.jacrev(carrier_fn))(points)
    carrier_hessians = jax.vmap(jax.hessian(carrier_fn))(points)
    coord_size = int(np.prod(points.shape[1:]))
    carrier_hessians = jnp.reshape(
        carrier_hessians,
        (
            points.shape[0],
            carrier_values.shape[1],
            coord_size,
            coord_size,
        ),
    )
    carrier_laplacians = jnp.trace(carrier_hessians, axis1=2, axis2=3)
    psi0, ground_gradients = ground_values_and_gradients(ground, points)
    ground_hbar = ground_hbar_values(ground, points, psi0)
    grad_dot = jnp.sum(
        ground_gradients[:, None, :, :] * carrier_gradients,
        axis=(2, 3),
    )
    values = psi0[:, None] * carrier_values
    hbar_values = (
        carrier_values * ground_hbar[:, None]
        - 0.5 * psi0[:, None] * carrier_laplacians
        - grad_dot
    )
    return values, hbar_values


def source_lift_bank_values_and_hbar(
    ground: FermiNetGround,
    points: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate source-lift bank values and ``(H-E0)`` actions.

    Returns:
        Batched ``Psi0 * d_z * g_k`` values and shifted Hamiltonian actions.
    """

    def carrier_fn(point: jax.Array) -> jax.Array:
        return source_lift_carrier_features_single(ground, point)

    carrier_values = jax.vmap(carrier_fn)(points)
    carrier_gradients = jax.vmap(jax.jacrev(carrier_fn))(points)
    carrier_hessians = jax.vmap(jax.hessian(carrier_fn))(points)
    coord_size = int(np.prod(points.shape[1:]))
    carrier_hessians = jnp.reshape(
        carrier_hessians,
        (
            points.shape[0],
            carrier_values.shape[1],
            coord_size,
            coord_size,
        ),
    )
    carrier_laplacians = jnp.trace(carrier_hessians, axis1=2, axis2=3)
    psi0, ground_gradients = ground_values_and_gradients(ground, points)
    ground_hbar = ground_hbar_values(ground, points, psi0)
    grad_dot = jnp.sum(
        ground_gradients[:, None, :, :] * carrier_gradients,
        axis=(2, 3),
    )
    values = psi0[:, None] * carrier_values
    hbar_values = (
        carrier_values * ground_hbar[:, None]
        - 0.5 * psi0[:, None] * carrier_laplacians
        - grad_dot
    )
    return values, hbar_values


def carrier_bank_values_and_hbar_from_arrays(
    ground: FermiNetGround,
    points: jax.Array,
    carrier_values: jax.Array | np.ndarray,
    carrier_gradients: jax.Array | np.ndarray,
    carrier_laplacians: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Convert carrier value/derivative arrays into ``Psi0*f`` action columns.

    Returns:
        Batched wavefunction values and shifted Hamiltonian actions.
    """
    carriers = jnp.asarray(carrier_values, dtype=points.dtype)
    gradients = jnp.asarray(carrier_gradients, dtype=points.dtype)
    laplacians = jnp.asarray(carrier_laplacians, dtype=points.dtype)
    psi0, ground_gradients = ground_values_and_gradients(ground, points)
    ground_hbar = ground_hbar_values(ground, points, psi0)
    grad_dot = jnp.sum(
        ground_gradients[:, None, :, :] * gradients,
        axis=(2, 3),
    )
    values = psi0[:, None] * carriers
    hbar_values = (
        carriers * ground_hbar[:, None] - 0.5 * psi0[:, None] * laplacians - grad_dot
    )
    return values, hbar_values


def carrier_bank_values_gradients_and_hbar_from_arrays(
    ground: FermiNetGround,
    points: jax.Array,
    carrier_values: jax.Array | np.ndarray,
    carrier_gradients: jax.Array | np.ndarray,
    carrier_laplacians: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert carrier arrays into ``Psi0*f`` values, gradients, and actions.

    Returns:
        Batched wavefunction values, gradients, and shifted Hamiltonian actions.
    """
    carriers = jnp.asarray(carrier_values, dtype=points.dtype)
    carrier_grads = jnp.asarray(carrier_gradients, dtype=points.dtype)
    psi0, ground_gradients = ground_values_and_gradients(ground, points)
    values, hbar_values = carrier_bank_values_and_hbar_from_arrays(
        ground,
        points,
        carriers,
        carrier_grads,
        carrier_laplacians,
    )
    gradients = (
        ground_gradients[:, None, :, :] * carriers[:, :, None, None]
        + psi0[:, None, None, None] * carrier_grads
    )
    return values, gradients, hbar_values


def evaluate_external_cas_carriers(
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None,
    points: np.ndarray,
    *,
    derivatives: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Evaluate selected external CAS carrier columns on electron configurations.

    Returns:
        Carrier values and optional derivative arrays for the accepted CAS block.

    Raises:
        RuntimeError: If derivative arrays are requested but unavailable.
    """
    if not external_blocks:
        empty_values = np.zeros((points.shape[0], 0), dtype=np.float64)
        return empty_values, None, None
    values_list = []
    gradients_list = []
    laplacians_list = []
    for block in external_blocks:
        bank = evaluate_casscf_ratio_carriers(
            block.model,
            points,
            target_mode=block.target_mode,
            correction_omegas=block.correction_omegas,
            correction_eta=block.correction_eta,
            tau_rel=block.tau_rel,
            tau_abs=block.tau_abs,
            ratio_clip=block.ratio_clip,
            derivatives=derivatives,
            finite_difference_step=block.finite_difference_step,
        )
        coeff = np.asarray(block.coefficients, dtype=np.float64)
        values_list.append(np.asarray(bank.values @ coeff, dtype=np.float64))
        if derivatives:
            if bank.gradients is None or bank.laplacians is None:
                msg = "external CAS carrier derivatives were not evaluated"
                raise RuntimeError(msg)
            gradients_list.append(
                np.asarray(
                    np.einsum("nfea,fc->ncea", bank.gradients, coeff),
                    dtype=np.float64,
                )
            )
            laplacians_list.append(
                np.asarray(bank.laplacians @ coeff, dtype=np.float64)
            )
    values = np.concatenate(values_list, axis=1)
    if not derivatives:
        return values, None, None
    return (
        values,
        np.concatenate(gradients_list, axis=1),
        np.concatenate(laplacians_list, axis=1),
    )


def external_cas_values_gradients_and_hbar(
    ground: FermiNetGround,
    points: jax.Array,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate accepted external CAS/FN basis values, gradients, and actions.

    Returns:
        Wavefunction values, gradients, and shifted Hamiltonian actions.
    """
    carrier_values, carrier_gradients, carrier_laplacians = (
        evaluate_external_cas_carriers(
            external_blocks,
            np.asarray(points),
            derivatives=True,
        )
    )
    if carrier_values.shape[1] == 0:
        empty_values = jnp.zeros((points.shape[0], 0), dtype=points.dtype)
        empty_gradients = jnp.zeros(
            (points.shape[0], 0, *points.shape[1:]), dtype=points.dtype
        )
        return empty_values, empty_gradients, empty_values
    return carrier_bank_values_gradients_and_hbar_from_arrays(
        ground,
        points,
        carrier_values,
        carrier_gradients,
        carrier_laplacians,
    )


def append_external_basis_values_gradients(
    values: jax.Array,
    gradients: jax.Array,
    ground: FermiNetGround,
    points: jax.Array,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None,
) -> tuple[jax.Array, jax.Array]:
    """Append direct external CAS/FN columns to an existing basis evaluation.

    Returns:
        Basis values and gradients with accepted external columns appended.
    """
    if external_cas_basis_count(external_blocks) == 0:
        return values, gradients
    ext_values, ext_gradients, _ = external_cas_values_gradients_and_hbar(
        ground,
        points,
        external_blocks,
    )
    return (
        jnp.concatenate([values, ext_values], axis=1),
        jnp.concatenate([gradients, ext_gradients], axis=1),
    )


def append_external_basis_values_hbar(
    values: jax.Array,
    hbar_values: jax.Array,
    ground: FermiNetGround,
    points: jax.Array,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None,
) -> tuple[jax.Array, jax.Array]:
    """Append direct external CAS/FN action columns to an existing basis.

    Returns:
        Basis values and shifted Hamiltonian actions with external columns appended.
    """
    if external_cas_basis_count(external_blocks) == 0:
        return values, hbar_values
    ext_values, _, ext_hbar = external_cas_values_gradients_and_hbar(
        ground,
        points,
        external_blocks,
    )
    return (
        jnp.concatenate([values, ext_values], axis=1),
        jnp.concatenate([hbar_values, ext_hbar], axis=1),
    )


def source_aux_and_head_values_and_laplacians(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    head_count: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate unprojected basis values and coordinate Laplacians.

    Returns:
        Batched basis values and their coordinate Laplacians.
    """

    def value_fn(point: jax.Array) -> jax.Array:
        return source_aux_and_head_value_single(
            params,
            ground,
            point,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )

    values = jax.vmap(value_fn)(points)
    hessians = jax.vmap(jax.hessian(value_fn))(points)
    coord_size = int(np.prod(points.shape[1:]))
    hessians = jnp.reshape(
        hessians,
        (points.shape[0], values.shape[1], coord_size, coord_size),
    )
    laplacians = jnp.trace(hessians, axis1=2, axis2=3)
    return values, laplacians


def source_aux_and_head_values_and_hbar(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    *,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    head_count: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate basis values and ``(H-E0)`` action columns.

    Source and auxiliary columns are represented as ``Psi0 * f`` and use the
    carrier product rule.  Neural response heads keep the direct coordinate
    Hessian action because they are not fixed carrier columns.

    Returns:
        Batched basis values and shifted Hamiltonian actions.
    """
    source_values, source_hbar = source_aux_carrier_values_and_hbar(
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    if head_count == 0:
        head_values = jnp.zeros((points.shape[0], 0), dtype=points.dtype)
        head_hbar = jnp.zeros((points.shape[0], 0), dtype=points.dtype)
    else:

        def head_value_fn(point: jax.Array) -> jax.Array:
            values = response_value_single(params, ground, point)
            return values if head_count is None else values[:head_count]

        head_values = jax.vmap(head_value_fn)(points)
        head_hessians = jax.vmap(jax.hessian(head_value_fn))(points)
        coord_size = int(np.prod(points.shape[1:]))
        head_hessians = jnp.reshape(
            head_hessians,
            (points.shape[0], head_values.shape[1], coord_size, coord_size),
        )
        head_laplacians = jnp.trace(head_hessians, axis1=2, axis2=3)
        head_hbar = (
            -0.5 * head_laplacians
            + potential_shift(ground, points)[:, None] * head_values
        )
    return (
        jnp.concatenate([source_values, head_values], axis=1),
        jnp.concatenate([source_hbar, head_hbar], axis=1),
    )


def source_plus_head_matrices(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    *,
    head_count: int | None = None,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    values, gradients, _ = source_aux_and_head_values_and_gradients(
        params,
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=head_count,
    )
    ground_values, ground_gradients = ground_values_and_gradients(ground, points)
    values, gradients, _ = project_values_against_ground(
        values, gradients, ground_values, ground_gradients, density
    )
    overlap, hamiltonian = weak_matrices(
        values, gradients, potential_shift(ground, points), density
    )
    return overlap, hamiltonian, values[:, 0]


def strong_residual_polish_loss(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    projection_coeff: jax.Array,
    response_coeffs: jax.Array,
    z_values: jax.Array,
    *,
    head_count: int,
    source_index: int,
    clip: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Soft-clipped pointwise strong-residual polish objective.

    ``projection_coeff`` and ``response_coeffs`` are fixed from the current
    weak-form matrices. The final strong-residual audit recomputes them
    independently; this loss only gives the candidate heads a short local
    polish toward the same pointwise residual definition used by the audit.

    Returns:
        Scalar polish loss and pointwise residual diagnostics.
    """
    values, hbar_values = source_aux_and_head_values_and_hbar(
        params,
        ground,
        points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=head_count,
    )
    ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
    ground_hbar = ground_hbar_values(ground, points, ground_values)
    values = values - ground_values[:, None] * projection_coeff[None, :]
    hbar_values = hbar_values - ground_hbar[:, None] * projection_coeff[None, :]
    x_values = jnp.einsum("ni,li->nl", values, response_coeffs)
    hbar_x_values = jnp.einsum("ni,li->nl", hbar_values, response_coeffs)
    phi = values[:, source_index]
    residual = phi[:, None] - z_values[None, :] * x_values + hbar_x_values
    weights = 1 / density
    source_norm = jnp.maximum(
        jnp.mean(weights * jnp.abs(phi) ** 2),
        jnp.asarray(1e-14, dtype=points.dtype),
    )
    residual_density = weights[:, None] * jnp.abs(residual) ** 2 / source_norm
    clip_sq = jnp.asarray(float(clip) ** 2, dtype=residual_density.dtype)
    clipped = clip_sq * jnp.log1p(residual_density / clip_sq)
    loss = jnp.mean(clipped)
    epsilon = jnp.sqrt(jnp.mean(residual_density, axis=0))
    return loss, {
        "loss": loss,
        "epsilon_max": jnp.max(epsilon),
        "residual_density_mean": jnp.mean(residual_density),
        "residual_density_max": jnp.max(residual_density),
    }


def strong_residual_polish_coefficients_from_density(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    *,
    head_count: int,
    omegas: np.ndarray,
    eta: float,
    source_index: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    batch_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute fixed correction-vector coefficients for strong polish.

    Returns:
        Ground-projection coefficients, response coefficients, and complex
        shifted frequencies.
    """
    projection_coeff = final_projection_coefficients_from_density(
        params,
        ground,
        points_np,
        density_np,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
    )
    overlap, hamiltonian, _ = final_weak_matrices_from_density_with_projection(
        params,
        ground,
        points_np,
        density_np,
        projection_coeff,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
    )
    z_values = np.asarray(omegas, dtype=np.float64) + 1j * float(eta)
    rhs = np.asarray(overlap[:, source_index], dtype=np.complex128)
    systems = (
        z_values[:, None, None] * np.asarray(overlap)[None, :, :]
        - np.asarray(hamiltonian)[None, :, :]
    )
    coeffs = np.linalg.solve(systems, rhs[None, :, None])[:, :, 0]
    return projection_coeff, jnp.asarray(coeffs), jnp.asarray(z_values)


def oracle_residual_profiles(
    residual: np.ndarray, phi: np.ndarray, density: np.ndarray
) -> dict[str, np.ndarray]:
    """Summarize strong residual distributions beyond their L2 norm.

    Returns:
        Per-frequency L2, winsorized-L2, quantile, and max residual scales.
    """
    weights = 1 / np.asarray(density, dtype=np.float64)
    source_norm = np.maximum(np.sum(weights * np.abs(phi) ** 2), 1e-14)
    contributions = weights[:, None] * np.abs(residual) ** 2 / source_norm
    q95 = np.quantile(contributions, 0.95, axis=0)
    q99 = np.quantile(contributions, 0.99, axis=0)
    return {
        "l2": np.sqrt(np.sum(contributions, axis=0)),
        "winsor99": np.sqrt(np.sum(np.minimum(contributions, q99[None, :]), axis=0)),
        "p95": np.sqrt(q95),
        "p99": np.sqrt(q99),
        "max": np.sqrt(np.max(contributions, axis=0)),
    }


def oracle_action_consistency_profiles(
    ad_columns: np.ndarray, fd_columns: np.ndarray, density: np.ndarray
) -> dict[str, np.ndarray]:
    """Compare AD action columns with explicit finite-difference actions.

    Returns:
        Per-frequency relative L2 and pointwise quantile/max mismatch scales.
    """
    weights = 1 / np.asarray(density, dtype=np.float64)
    diff_power = np.sum(np.abs(ad_columns - fd_columns) ** 2, axis=2)
    fd_power = np.sum(np.abs(fd_columns) ** 2, axis=2)
    weighted_diff = np.sum(weights[:, None] * diff_power, axis=0)
    weighted_fd = np.maximum(np.sum(weights[:, None] * fd_power, axis=0), 1e-30)
    sample_rel = np.sqrt(diff_power / np.maximum(fd_power, 1e-30))
    return {
        "l2": np.sqrt(weighted_diff / weighted_fd),
        "p95": np.quantile(sample_rel, 0.95, axis=0),
        "p99": np.quantile(sample_rel, 0.99, axis=0),
        "max": np.max(sample_rel, axis=0),
    }


def oracle_candidate_prefilter_profiles(
    candidate_values: np.ndarray,
    action_columns: np.ndarray,
    phi: np.ndarray,
    density: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Summarize candidate projected-value and action-column norms.

    Returns:
        Per-candidate value/action norms and compact min/max/condition metrics.
    """
    weights = 1 / np.asarray(density, dtype=np.float64)
    source_norm = np.maximum(np.sum(weights * np.abs(phi) ** 2), 1e-14)
    value_power = np.sum(
        weights[:, None] * np.abs(candidate_values) ** 2,
        axis=0,
    )
    action_power = np.sum(
        weights[:, None, None] * np.abs(action_columns) ** 2,
        axis=0,
    )
    value_norm = np.sqrt(value_power / source_norm)
    action_norm = np.sqrt(action_power / source_norm)
    action_floor = np.maximum(np.min(action_norm, axis=1), 1e-30)
    action_condition = np.max(action_norm, axis=1) / action_floor
    return {
        "value_norm": value_norm,
        "value_norm_min": float(np.min(value_norm)),
        "value_norm_max": float(np.max(value_norm)),
        "action_norm": action_norm,
        "action_norm_min": float(np.min(action_norm)),
        "action_norm_max": float(np.max(action_norm)),
        "action_condition": action_condition,
        "action_condition_max": float(np.max(action_condition)),
    }


def action_space_schur_complement(
    residual: np.ndarray,
    candidate_columns: np.ndarray,
    old_columns: np.ndarray,
    density: np.ndarray,
    *,
    split: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project residual and candidates away from the old action-column span.

    Returns:
        Residualized residuals and candidate action columns.

    Raises:
        ValueError: If input sample/omega axes or train split are inconsistent.
    """
    residual = np.asarray(residual, dtype=np.complex128)
    candidate_columns = np.asarray(candidate_columns, dtype=np.complex128)
    old_columns = np.asarray(old_columns, dtype=np.complex128)
    density = np.asarray(density, dtype=np.float64)
    if old_columns.ndim != 3 or candidate_columns.ndim != 3:
        msg = "action-space Schur columns must have shape (samples, omegas, heads)"
        raise ValueError(msg)
    if residual.shape != candidate_columns.shape[:2]:
        msg = "action-space Schur residual shape must match column sample/omega axes"
        raise ValueError(msg)
    if old_columns.shape[:2] != candidate_columns.shape[:2]:
        msg = "old and candidate action columns must share sample/omega axes"
        raise ValueError(msg)
    if old_columns.shape[2] == 0:
        return residual, candidate_columns
    split = int(split)
    if split < 1 or split >= residual.shape[0]:
        msg = "action-space Schur split must leave train and validation samples"
        raise ValueError(msg)
    if density.shape[0] != residual.shape[0]:
        msg = "action-space Schur density must match sample count"
        raise ValueError(msg)
    old_count = old_columns.shape[2]
    schur_residual = np.empty_like(residual)
    schur_columns = np.empty_like(candidate_columns)
    weights = 1 / density[:split]
    sqrt_w = np.sqrt(weights)
    ridge_matrix = float(ridge) * np.eye(old_count, dtype=np.complex128)
    for omega_idx in range(residual.shape[1]):
        old_train = old_columns[:split, omega_idx, :] * sqrt_w[:, None]
        gram = old_train.conj().T @ old_train + ridge_matrix
        residual_train = residual[:split, omega_idx] * sqrt_w
        residual_rhs = old_train.conj().T @ residual_train
        residual_coeff = np.linalg.solve(gram, residual_rhs)
        candidate_train = candidate_columns[:split, omega_idx, :] * sqrt_w[:, None]
        candidate_rhs = old_train.conj().T @ candidate_train
        candidate_coeff = np.linalg.solve(gram, candidate_rhs)
        old_all = old_columns[:, omega_idx, :]
        schur_residual[:, omega_idx] = residual[:, omega_idx] - old_all @ residual_coeff
        schur_columns[:, omega_idx, :] = (
            candidate_columns[:, omega_idx, :] - old_all @ candidate_coeff
        )
    return schur_residual, schur_columns


def oracle_linear_combination_metrics(
    residual: np.ndarray,
    columns: np.ndarray,
    phi: np.ndarray,
    density: np.ndarray,
    *,
    ridge: float,
    split: int,
) -> dict[str, np.ndarray | float]:
    """Solve a train-split ridge oracle and summarize validation residuals.

    Returns:
        Strong residual profiles, ratios, and epsilon^2 improvements.

    Raises:
        ValueError: If input sample/omega axes or train split are inconsistent.
    """
    residual = np.asarray(residual, dtype=np.complex128)
    columns = np.asarray(columns, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)
    density = np.asarray(density, dtype=np.float64)
    if columns.ndim != 3:
        msg = "oracle columns must have shape (samples, omegas, candidates)"
        raise ValueError(msg)
    if residual.shape != columns.shape[:2]:
        msg = "oracle residual shape must match column sample/omega axes"
        raise ValueError(msg)
    if phi.shape[0] != residual.shape[0] or density.shape[0] != residual.shape[0]:
        msg = "oracle phi/density must match residual sample count"
        raise ValueError(msg)
    split = int(split)
    if split < 1 or split >= residual.shape[0]:
        msg = "oracle split must leave train and validation samples"
        raise ValueError(msg)
    candidate_heads = columns.shape[2]
    if candidate_heads < 1:
        msg = "oracle requires at least one candidate column"
        raise ValueError(msg)

    def weighted_epsilon(local_residual, local_phi, local_density):
        weights = 1 / np.asarray(local_density, dtype=np.float64)
        source_norm = np.maximum(np.sum(weights * np.abs(local_phi) ** 2), 1e-14)
        return np.sqrt(
            np.sum(weights[:, None] * np.abs(local_residual) ** 2, axis=0) / source_norm
        )

    def solve_oracle_for_omega(omega_idx: int):
        weights = 1 / density[:split]
        sqrt_w = np.sqrt(weights)
        y = columns[:split, omega_idx, :] * sqrt_w[:, None]
        r = residual[:split, omega_idx] * sqrt_w
        gram = y.conj().T @ y + float(ridge) * np.eye(candidate_heads)
        rhs_oracle = y.conj().T @ r
        return np.linalg.solve(gram, rhs_oracle)

    oracle_coeffs = np.stack(
        [solve_oracle_for_omega(idx) for idx in range(residual.shape[1])],
        axis=0,
    )
    oracle_residual = residual - np.einsum("nlc,lc->nl", columns, oracle_coeffs)
    train_old = weighted_epsilon(residual[:split], phi[:split], density[:split])
    train_oracle = weighted_epsilon(
        oracle_residual[:split], phi[:split], density[:split]
    )
    val_old = weighted_epsilon(residual[split:], phi[split:], density[split:])
    val_oracle = weighted_epsilon(oracle_residual[split:], phi[split:], density[split:])
    val_ratio = val_oracle / np.maximum(val_old, 1e-14)
    train_old_profiles = oracle_residual_profiles(
        residual[:split], phi[:split], density[:split]
    )
    train_oracle_profiles = oracle_residual_profiles(
        oracle_residual[:split], phi[:split], density[:split]
    )
    val_old_profiles = oracle_residual_profiles(
        residual[split:], phi[split:], density[split:]
    )
    val_oracle_profiles = oracle_residual_profiles(
        oracle_residual[split:], phi[split:], density[split:]
    )
    val_ratio_winsor99 = val_oracle_profiles["winsor99"] / np.maximum(
        val_old_profiles["winsor99"], 1e-14
    )
    val_ratio_p95 = val_oracle_profiles["p95"] / np.maximum(
        val_old_profiles["p95"], 1e-14
    )
    val_ratio_p99 = val_oracle_profiles["p99"] / np.maximum(
        val_old_profiles["p99"], 1e-14
    )
    val_ratio_pointwise_max = val_oracle_profiles["max"] / np.maximum(
        val_old_profiles["max"], 1e-14
    )
    train_epsilon2_improvement = train_old**2 - train_oracle**2
    val_epsilon2_improvement = val_old**2 - val_oracle**2
    val_relative_epsilon2_improvement = val_epsilon2_improvement / np.maximum(
        val_old**2,
        1e-14,
    )
    val_winsor99_epsilon2_improvement = (
        val_old_profiles["winsor99"] ** 2 - val_oracle_profiles["winsor99"] ** 2
    )
    val_winsor99_relative_epsilon2_improvement = (
        val_winsor99_epsilon2_improvement
        / np.maximum(val_old_profiles["winsor99"] ** 2, 1e-14)
    )
    return {
        "oracle_coeffs": oracle_coeffs,
        "oracle_residual": oracle_residual,
        "train_epsilon_old": train_old,
        "train_epsilon_oracle": train_oracle,
        "validation_epsilon_old": val_old,
        "validation_epsilon_oracle": val_oracle,
        "validation_ratio": val_ratio,
        "validation_ratio_max": float(np.max(val_ratio)),
        "validation_ratio_winsor99": val_ratio_winsor99,
        "validation_ratio_winsor99_max": float(np.max(val_ratio_winsor99)),
        "validation_ratio_p95": val_ratio_p95,
        "validation_ratio_p95_max": float(np.max(val_ratio_p95)),
        "validation_ratio_p99": val_ratio_p99,
        "validation_ratio_p99_max": float(np.max(val_ratio_p99)),
        "validation_ratio_pointwise_max": val_ratio_pointwise_max,
        "validation_ratio_pointwise_max_max": float(np.max(val_ratio_pointwise_max)),
        "validation_improvement_min": float(1 - np.max(val_ratio)),
        "train_winsor99_old": train_old_profiles["winsor99"],
        "train_winsor99_oracle": train_oracle_profiles["winsor99"],
        "validation_winsor99_old": val_old_profiles["winsor99"],
        "validation_winsor99_oracle": val_oracle_profiles["winsor99"],
        "train_epsilon2_improvement": train_epsilon2_improvement,
        "train_epsilon2_improvement_min": float(np.min(train_epsilon2_improvement)),
        "validation_epsilon2_improvement": val_epsilon2_improvement,
        "validation_epsilon2_improvement_min": float(np.min(val_epsilon2_improvement)),
        "validation_relative_epsilon2_improvement": (val_relative_epsilon2_improvement),
        "validation_relative_epsilon2_improvement_min": float(
            np.min(val_relative_epsilon2_improvement)
        ),
        "validation_winsor99_epsilon2_improvement": (val_winsor99_epsilon2_improvement),
        "validation_winsor99_epsilon2_improvement_min": float(
            np.min(val_winsor99_epsilon2_improvement)
        ),
        "validation_winsor99_relative_epsilon2_improvement": (
            val_winsor99_relative_epsilon2_improvement
        ),
        "validation_winsor99_relative_epsilon2_improvement_min": float(
            np.min(val_winsor99_relative_epsilon2_improvement)
        ),
    }


def strong_oracle_linear_combination_diagnostic(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    *,
    active_heads: int,
    candidate_heads: int,
    omegas: np.ndarray,
    eta: float,
    source_index: int,
    ridge: float,
    action_schur: bool = False,
    action_schur_ridge: float | None = None,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    batch_size: int,
) -> dict[str, np.ndarray | float | int]:
    """Oracle strong-form linear-combination diagnostic for candidate heads.

    The diagnostic freezes the trained candidate block, solves the old
    correction vector on weak matrices, then asks whether any ridge-regularized
    linear combination of candidate ``(z - Hbar) chi_j`` columns can reduce the
    old strong residual on held-out strong samples.

    Returns:
        Train/validation strong residual epsilons before and after the oracle
        candidate combination, plus validation improvement ratios.

    Raises:
        ValueError: If the diagnostic sample or source/candidate layout is
        invalid.
    """
    points_np = np.asarray(points_np)
    density_np = np.asarray(density_np)
    omegas = np.asarray(omegas, dtype=np.float64)
    if candidate_heads < 1:
        msg = "strong oracle diagnostic requires at least one candidate head"
        raise ValueError(msg)
    if points_np.shape[0] < 2:
        msg = "strong oracle diagnostic requires at least two samples"
        raise ValueError(msg)
    if density_np.shape[0] != points_np.shape[0]:
        msg = "strong oracle points and density must have same length"
        raise ValueError(msg)
    if omegas.size == 0:
        msg = "strong oracle diagnostic requires at least one omega"
        raise ValueError(msg)
    full_head_count = int(active_heads) + int(candidate_heads)
    source_count = 1 + _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    if source_index < 0 or source_index >= source_count:
        msg = "strong oracle source_index must select a source channel"
        raise ValueError(msg)
    old_end = source_count + int(active_heads)
    candidate_start = old_end
    candidate_end = candidate_start + int(candidate_heads)
    projection_coeff = final_projection_coefficients_from_density(
        params,
        ground,
        points_np,
        density_np,
        head_count=full_head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
    )
    overlap, hamiltonian, _ = final_weak_matrices_from_density_with_projection(
        params,
        ground,
        points_np,
        density_np,
        projection_coeff,
        head_count=full_head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
    )
    z_values = np.asarray(omegas, dtype=np.float64) + 1j * float(eta)
    s_old = np.asarray(overlap[:old_end, :old_end])
    k_old = np.asarray(hamiltonian[:old_end, :old_end])
    rhs = np.asarray(overlap[:old_end, source_index], dtype=np.complex128)
    systems = z_values[:, None, None] * s_old[None, :, :] - k_old[None, :, :]
    old_coeffs = np.linalg.solve(systems, rhs[None, :, None])[:, :, 0]
    old_coeffs_jax = jnp.asarray(old_coeffs)
    z_jax = jnp.asarray(z_values)
    schur_ridge = float(ridge if action_schur_ridge is None else action_schur_ridge)

    @jax.jit
    def chunk_oracle_columns(
        points: jax.Array,
        density: jax.Array,
        project_coeff: jax.Array,
        coeffs: jax.Array,
        z: jax.Array,
    ):
        values, hbar_values = source_aux_and_head_values_and_hbar(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=full_head_count,
        )
        ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
        ground_hbar = ground_hbar_values(ground, points, ground_values)
        values = values - ground_values[:, None] * project_coeff[None, :]
        hbar_values = hbar_values - ground_hbar[:, None] * project_coeff[None, :]
        old_values = values[:, :old_end]
        old_hbar = hbar_values[:, :old_end]
        old_action_columns = (
            z[None, :, None] * old_values[:, None, :] - old_hbar[:, None, :]
        )
        x_old = jnp.einsum("ni,li->nl", old_values, coeffs)
        hbar_x_old = jnp.einsum("ni,li->nl", old_hbar, coeffs)
        phi = values[:, source_index]
        old_residual = phi[:, None] - z[None, :] * x_old + hbar_x_old
        cand_values = values[:, candidate_start:candidate_end]
        cand_hbar = hbar_values[:, candidate_start:candidate_end]
        oracle_columns = (
            z[None, :, None] * cand_values[:, None, :] - cand_hbar[:, None, :]
        )
        return (
            old_residual,
            oracle_columns,
            old_action_columns,
            cand_values,
            phi,
            density,
        )

    residual_pieces = []
    column_pieces = []
    old_action_column_pieces = []
    candidate_value_pieces = []
    phi_pieces = []
    density_pieces = []
    for chunk in make_batches(points_np.shape[0], batch_size):
        (
            residual,
            columns,
            old_action_columns,
            candidate_values,
            phi,
            density,
        ) = chunk_oracle_columns(
            jnp.asarray(points_np[chunk]),
            jnp.asarray(density_np[chunk]),
            projection_coeff,
            old_coeffs_jax,
            z_jax,
        )
        residual_pieces.append(np.asarray(residual, dtype=np.complex128))
        column_pieces.append(np.asarray(columns, dtype=np.complex128))
        old_action_column_pieces.append(
            np.asarray(old_action_columns, dtype=np.complex128)
        )
        candidate_value_pieces.append(np.asarray(candidate_values, dtype=np.complex128))
        phi_pieces.append(np.asarray(phi, dtype=np.complex128))
        density_pieces.append(np.asarray(density, dtype=np.float64))
    residual_all = np.concatenate(residual_pieces, axis=0)
    columns_all = np.concatenate(column_pieces, axis=0)
    old_action_columns_all = np.concatenate(old_action_column_pieces, axis=0)
    candidate_values_all = np.concatenate(candidate_value_pieces, axis=0)
    phi_all = np.concatenate(phi_pieces, axis=0)
    density_all = np.concatenate(density_pieces, axis=0)
    split = max(1, points_np.shape[0] // 2)
    if split >= points_np.shape[0]:
        split = points_np.shape[0] - 1

    raw_metrics = oracle_linear_combination_metrics(
        residual_all,
        columns_all,
        phi_all,
        density_all,
        ridge=ridge,
        split=split,
    )
    oracle_residual_input = residual_all
    oracle_column_input = columns_all
    if action_schur:
        oracle_residual_input, oracle_column_input = action_space_schur_complement(
            residual_all,
            columns_all,
            old_action_columns_all,
            density_all,
            split=split,
            ridge=schur_ridge,
        )
    selected_metrics = (
        oracle_linear_combination_metrics(
            oracle_residual_input,
            oracle_column_input,
            phi_all,
            density_all,
            ridge=ridge,
            split=split,
        )
        if action_schur
        else raw_metrics
    )
    action_consistency = oracle_action_consistency_profiles(
        columns_all, columns_all, density_all
    )
    candidate_prefilter = oracle_candidate_prefilter_profiles(
        candidate_values_all,
        oracle_column_input,
        phi_all,
        density_all,
    )
    metrics = selected_metrics
    raw = raw_metrics
    return {
        "strong_oracle_omegas": omegas,
        "strong_oracle_samples": int(points_np.shape[0]),
        "strong_oracle_train_samples": int(split),
        "strong_oracle_validation_samples": int(points_np.shape[0] - split),
        "strong_oracle_ridge": float(ridge),
        "strong_oracle_action_mode": "ad",
        "strong_oracle_action_schur": bool(action_schur),
        "strong_oracle_action_schur_ridge": float(schur_ridge),
        "strong_oracle_raw_validation_ratio_max": float(raw["validation_ratio_max"]),
        "strong_oracle_raw_validation_ratio_winsor99_max": float(
            raw["validation_ratio_winsor99_max"]
        ),
        "strong_oracle_raw_validation_ratio_p99_max": float(
            raw["validation_ratio_p99_max"]
        ),
        "strong_oracle_raw_validation_ratio_pointwise_max_max": float(
            raw["validation_ratio_pointwise_max_max"]
        ),
        "strong_oracle_raw_validation_relative_epsilon2_improvement_min": (
            float(raw["validation_relative_epsilon2_improvement_min"])
        ),
        "strong_oracle_raw_validation_winsor99_relative_epsilon2_improvement_min": (
            float(raw["validation_winsor99_relative_epsilon2_improvement_min"])
        ),
        "strong_oracle_coeffs": metrics["oracle_coeffs"],
        "strong_oracle_raw_coeffs": raw["oracle_coeffs"],
        "strong_oracle_train_epsilon_old": metrics["train_epsilon_old"],
        "strong_oracle_train_epsilon_oracle": metrics["train_epsilon_oracle"],
        "strong_oracle_validation_epsilon_old": metrics["validation_epsilon_old"],
        "strong_oracle_validation_epsilon_oracle": (
            metrics["validation_epsilon_oracle"]
        ),
        "strong_oracle_validation_ratio": metrics["validation_ratio"],
        "strong_oracle_validation_ratio_max": float(metrics["validation_ratio_max"]),
        "strong_oracle_validation_ratio_winsor99": metrics["validation_ratio_winsor99"],
        "strong_oracle_validation_ratio_winsor99_max": float(
            metrics["validation_ratio_winsor99_max"]
        ),
        "strong_oracle_validation_ratio_p95": metrics["validation_ratio_p95"],
        "strong_oracle_validation_ratio_p95_max": float(
            metrics["validation_ratio_p95_max"]
        ),
        "strong_oracle_validation_ratio_p99": metrics["validation_ratio_p99"],
        "strong_oracle_validation_ratio_p99_max": float(
            metrics["validation_ratio_p99_max"]
        ),
        "strong_oracle_validation_ratio_pointwise_max": metrics[
            "validation_ratio_pointwise_max"
        ],
        "strong_oracle_validation_ratio_pointwise_max_max": float(
            metrics["validation_ratio_pointwise_max_max"]
        ),
        "strong_oracle_validation_improvement_min": float(
            metrics["validation_improvement_min"]
        ),
        "strong_oracle_train_winsor99_old": metrics["train_winsor99_old"],
        "strong_oracle_train_winsor99_oracle": metrics["train_winsor99_oracle"],
        "strong_oracle_validation_winsor99_old": metrics["validation_winsor99_old"],
        "strong_oracle_validation_winsor99_oracle": metrics[
            "validation_winsor99_oracle"
        ],
        "strong_oracle_train_epsilon2_improvement": (
            metrics["train_epsilon2_improvement"]
        ),
        "strong_oracle_train_epsilon2_improvement_min": float(
            metrics["train_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_epsilon2_improvement": (
            metrics["validation_epsilon2_improvement"]
        ),
        "strong_oracle_validation_epsilon2_improvement_min": float(
            metrics["validation_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_relative_epsilon2_improvement": (
            metrics["validation_relative_epsilon2_improvement"]
        ),
        "strong_oracle_validation_relative_epsilon2_improvement_min": float(
            metrics["validation_relative_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_winsor99_epsilon2_improvement": (
            metrics["validation_winsor99_epsilon2_improvement"]
        ),
        "strong_oracle_validation_winsor99_epsilon2_improvement_min": float(
            metrics["validation_winsor99_epsilon2_improvement_min"]
        ),
        "strong_oracle_validation_winsor99_relative_epsilon2_improvement": (
            metrics["validation_winsor99_relative_epsilon2_improvement"]
        ),
        "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min": (
            float(metrics["validation_winsor99_relative_epsilon2_improvement_min"])
        ),
        "strong_oracle_action_consistency_l2": action_consistency["l2"],
        "strong_oracle_action_consistency_l2_max": float(
            np.max(action_consistency["l2"])
        ),
        "strong_oracle_action_consistency_p95": action_consistency["p95"],
        "strong_oracle_action_consistency_p95_max": float(
            np.max(action_consistency["p95"])
        ),
        "strong_oracle_action_consistency_p99": action_consistency["p99"],
        "strong_oracle_action_consistency_p99_max": float(
            np.max(action_consistency["p99"])
        ),
        "strong_oracle_action_consistency_pointwise_max": (action_consistency["max"]),
        "strong_oracle_action_consistency_pointwise_max_max": float(
            np.max(action_consistency["max"])
        ),
        "strong_oracle_candidate_value_norm_min": (
            candidate_prefilter["value_norm_min"]
        ),
        "strong_oracle_candidate_value_norm_max": (
            candidate_prefilter["value_norm_max"]
        ),
        "strong_oracle_candidate_action_norm_min": (
            candidate_prefilter["action_norm_min"]
        ),
        "strong_oracle_candidate_action_norm_max": (
            candidate_prefilter["action_norm_max"]
        ),
        "strong_oracle_candidate_action_condition_max": (
            candidate_prefilter["action_condition_max"]
        ),
    }


def region_balanced_holdout_indices(
    ground: FermiNetGround,
    points_np: np.ndarray,
    *,
    n_select: int,
    seed: int,
    node_quantile: float,
    tail_quantile: float,
    en_cusp_radius: float,
    ee_cusp_radius: float,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    """Select a holdout cache with explicit node/cusp/tail/bulk coverage.

    Returns:
        Selected sample indices and diagnostics describing available/selected
        region counts.
    """
    total = int(points_np.shape[0])
    n_select = min(max(0, int(n_select)), total)
    if n_select == 0:
        return np.asarray([], dtype=np.int64), {
            "labels": np.asarray([], dtype="<U1"),
            "available_counts": np.asarray([], dtype=np.int64),
            "drawn_counts": np.asarray([], dtype=np.int64),
            "selected_counts": np.asarray([], dtype=np.int64),
            "node_threshold": float("nan"),
            "tail_threshold": float("nan"),
            "en_cusp_radius": float(en_cusp_radius),
            "ee_cusp_radius": float(ee_cusp_radius),
        }

    ground_abs_chunks = []
    for chunk in make_batches(total, max(1, int(batch_size))):
        ground_values = jax.vmap(ground_value_single, (None, 0))(
            ground, jnp.asarray(points_np[chunk])
        )
        ground_abs_chunks.append(np.abs(np.asarray(ground_values, dtype=np.float64)))
    ground_abs = np.concatenate(ground_abs_chunks, axis=0)
    atoms_np = np.asarray(ground.atoms, dtype=np.float64)
    charges_np = np.asarray(ground.charges, dtype=np.float64)
    charge_center = np.sum(atoms_np * charges_np[:, None], axis=0)
    charge_center = charge_center / np.sum(charges_np)
    shifted = points_np - charge_center[None, None, :]
    tail_radius = np.sqrt(np.mean(np.sum(shifted**2, axis=-1), axis=1))
    electron_atom = points_np[:, :, None, :] - atoms_np[None, None, :, :]
    min_en = np.min(np.sqrt(np.sum(electron_atom**2, axis=-1) + 1e-24), axis=(1, 2))
    nelec = points_np.shape[1]
    if nelec < 2:
        min_ee = np.full((total,), np.inf, dtype=np.float64)
    else:
        ee_vec = points_np[:, :, None, :] - points_np[:, None, :, :]
        ee_dist = np.sqrt(np.sum(ee_vec**2, axis=-1) + 1e-24)
        ee_dist = ee_dist + np.eye(nelec, dtype=np.float64)[None, :, :] * 1e6
        min_ee = np.min(ee_dist, axis=(1, 2))

    node_threshold = float(np.quantile(ground_abs, float(node_quantile)))
    tail_threshold = float(np.quantile(tail_radius, float(tail_quantile)))
    node_mask = ground_abs <= node_threshold
    en_cusp_mask = min_en <= float(en_cusp_radius)
    ee_cusp_mask = min_ee <= float(ee_cusp_radius)
    tail_mask = tail_radius >= tail_threshold
    bulk_mask = ~(node_mask | en_cusp_mask | ee_cusp_mask | tail_mask)
    labels = np.asarray(["node_tube", "en_cusp", "ee_cusp", "tail", "bulk"])
    masks = [node_mask, en_cusp_mask, ee_cusp_mask, tail_mask, bulk_mask]
    available_counts = np.asarray(
        [int(np.count_nonzero(mask)) for mask in masks], dtype=np.int64
    )
    selected: list[int] = []
    selected_mask = np.zeros(total, dtype=bool)
    draw_counts = np.zeros(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    base_quota = n_select // len(masks)
    extra = n_select % len(masks)
    for idx, mask in enumerate(masks):
        quota = base_quota + (1 if idx < extra else 0)
        candidates = np.flatnonzero(mask & ~selected_mask)
        rng.shuffle(candidates)
        chosen = candidates[:quota]
        if chosen.size:
            selected.extend(int(item) for item in chosen)
            selected_mask[chosen] = True
            draw_counts[idx] += int(chosen.size)
    if len(selected) < n_select:
        remaining = np.flatnonzero(~selected_mask)
        rng.shuffle(remaining)
        chosen = remaining[: n_select - len(selected)]
        selected.extend(int(item) for item in chosen)
    indices = np.asarray(selected, dtype=np.int64)
    indices.sort()
    selected_counts = np.asarray(
        [int(np.count_nonzero(mask[indices])) for mask in masks], dtype=np.int64
    )
    return indices, {
        "labels": labels,
        "available_counts": available_counts,
        "drawn_counts": draw_counts,
        "selected_counts": selected_counts,
        "node_threshold": node_threshold,
        "tail_threshold": tail_threshold,
        "en_cusp_radius": float(en_cusp_radius),
        "ee_cusp_radius": float(ee_cusp_radius),
    }


def region_balanced_summary_text(
    diagnostics: dict[str, np.ndarray | float | int],
) -> str:
    """Format region-balanced cache diagnostics for one log line.

    Returns:
        Human-readable counts and thresholds.
    """
    labels = np.asarray(diagnostics["labels"]).astype(str)
    available = np.asarray(diagnostics["available_counts"], dtype=np.int64)
    drawn = np.asarray(diagnostics["drawn_counts"], dtype=np.int64)
    selected = np.asarray(diagnostics["selected_counts"], dtype=np.int64)
    label_text = ",".join(labels.tolist())
    available_text = ",".join(str(int(item)) for item in available)
    drawn_text = ",".join(str(int(item)) for item in drawn)
    selected_text = ",".join(str(int(item)) for item in selected)
    return (
        f"labels={label_text} "
        f"available={available_text} "
        f"drawn={drawn_text} "
        f"selected_membership={selected_text} "
        f"node_threshold={float(diagnostics['node_threshold']):.3e} "
        f"tail_threshold={float(diagnostics['tail_threshold']):.3e} "
        f"en_cusp_radius={float(diagnostics['en_cusp_radius']):.3e} "
        f"ee_cusp_radius={float(diagnostics['ee_cusp_radius']):.3e}"
    )


def _train_residual_enrichment_attempt(  # noqa: C901
    params: Params,
    ground: FermiNetGround,
    *,
    key: jax.Array,
    active_heads: int,
    candidate_heads: int,
    attempt: int,
    train_samples: int,
    holdout_samples: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    envelope_decay: float,
    ritz_warmup_epochs: int,
    ritz_warmup_learning_rate_scale: float,
    ritz_warmup_roots: int,
    ritz_accept: bool,
    ritz_accept_root_floor: float,
    ritz_accept_min_weight: float,
    ritz_accept_min_pole_improvement: float,
    validation_holdouts: int,
    holdout_min_pass_fraction: float,
    ritz_validation_max_spread: float,
    ritz_validation_min_pass_fraction: float,
    enrichment_sampling: str,
    enrichment_sampling_walkers: int,
    enrichment_sampling_burn_in: int,
    enrichment_sampling_steps_between: int,
    enrichment_sampling_width: float,
    enrichment_sampling_density_batch_size: int,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float,
    q_node_ground_power: float,
    q_tail_weight: float,
    q_tail_envelope_decay: float,
    q_kinetic_weight: float,
    source_envelope_core_decay: float,
    source_envelope_diffuse_decay: float,
    residual_omegas: np.ndarray,
    residual_omega_weights: np.ndarray,
    residual_source_weights: np.ndarray,
    residual_eta: float,
    enrichment_training_objective: str,
    enrichment_selection_objective: str,
    lambda_rough: float,
    delta: float,
    min_relative_improvement: float,
    min_capture: float,
    min_objective_improvement: float,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    require_training_improvement: bool,
    source_bright_gate: bool,
    source_bright_max_regression: float,
    strong_residual_samples: int,
    strong_residual_omegas: np.ndarray,
    strong_residual_source_index: int,
    strong_residual_batch_size: int,
    max_strong_residual_epsilon_over_eta: float,
    max_strong_residual_provisional_ratio: float,
    strong_polish_epochs: int,
    strong_polish_samples: int,
    strong_polish_batch_size: int,
    strong_polish_learning_rate_scale: float,
    strong_polish_clip: float,
    strong_oracle_samples: int,
    strong_oracle_ridge: float,
    strong_oracle_action_schur: bool,
    strong_oracle_action_schur_ridge: float,
    strong_oracle_max_validation_ratio_winsor99: float,
    strong_oracle_max_validation_ratio_p99: float,
    strong_oracle_max_validation_ratio_pointwise: float,
    strong_oracle_min_validation_relative_epsilon2_improvement: float,
    strong_oracle_min_validation_winsor99_relative_epsilon2_improvement: float,
    strong_oracle_min_candidate_value_norm: float,
    strong_oracle_min_candidate_action_norm: float,
    strong_oracle_max_candidate_action_condition: float,
    region_balanced_cache: bool,
    region_node_quantile: float,
    region_tail_quantile: float,
    region_en_cusp_radius: float,
    region_ee_cusp_radius: float,
    overlap_cutoff: float,
    aux_source_exponents: np.ndarray,
    aux_source_dipole_radial_powers: np.ndarray,
    aux_source_dipole_radial_scale: float,
    aux_source_atom_odd_exponents: np.ndarray,
    aux_source_atom_odd_slater_decays: np.ndarray,
    aux_source_bond_odd_slater_decays: np.ndarray,
    aux_source_dipole_ee_scales: np.ndarray,
    aux_source_bond_odd_ee_slater_decays: np.ndarray,
    aux_source_bond_odd_ee_scales: np.ndarray,
    validation_every: int,
    log_every: int,
) -> tuple[Params, EnrichmentDiagnostics]:
    """Train and accept/reject one BF-NKSR residual-enrichment candidate.

    Returns:
        The accepted parameter tree, or the initial tree if rejected, plus
        held-out acceptance diagnostics.

    Raises:
        ValueError: If the requested candidate block is not available.
    """
    available_heads = num_response_heads(ground)
    if candidate_heads < 1 or candidate_heads > available_heads:
        msg = (
            "candidate_heads must be between 1 and the raw candidate-block "
            f"head count ({available_heads}), got {candidate_heads}"
        )
        raise ValueError(msg)

    fixed_basis_params = (
        params
        if is_response_block_dictionary(params)
        else empty_response_basis_params()
    )
    candidate_init_key = jax.random.fold_in(key, 70_001 + int(attempt))
    candidate_params = init_response_params(
        candidate_init_key,
        ground=ground,
        initial_decay_min=0.2,
        initial_decay_max=2.0,
    )
    candidate_params = initialize_response_envelope_decay(
        candidate_params,
        n_heads=num_response_heads(ground),
        determinants_per_head=ground.response_model.determinants_per_head,
        initial_decay_min=0.2,
        initial_decay_max=2.0,
        decay_values=None,
    )
    candidate_params = select_response_heads(
        candidate_params,
        raw_head_count=num_response_heads(ground),
        head_count=candidate_heads,
    )
    candidate_params = initialize_response_source_lift(
        candidate_params,
        head_count=candidate_heads,
        key=jax.random.fold_in(candidate_init_key, 9_173),
        scale=1.0,
    )

    def with_candidate(local_candidate_params: Params) -> Params:
        return append_response_block_params(fixed_basis_params, local_candidate_params)

    initial_candidate_params = candidate_params
    initial_params = with_candidate(initial_candidate_params)
    validation_holdout_count = max(1, int(validation_holdouts))
    split_keys = jax.random.split(key, validation_holdout_count + 1)
    train_key = split_keys[0]
    holdout_keys = split_keys[1:]
    sampling_head_count = active_heads + candidate_heads
    points, density, train_pmove = sample_enrichment_distribution(
        initial_params,
        ground,
        key=train_key,
        n_samples=train_samples,
        sampling=enrichment_sampling,
        envelope_decay=envelope_decay,
        head_count=sampling_head_count,
        walkers=enrichment_sampling_walkers,
        burn_in=enrichment_sampling_burn_in,
        steps_between=enrichment_sampling_steps_between,
        width=enrichment_sampling_width,
        density_batch_size=enrichment_sampling_density_batch_size,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
        source_envelope_core_decay=source_envelope_core_decay,
        source_envelope_diffuse_decay=source_envelope_diffuse_decay,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    holdouts: list[tuple[jax.Array, jax.Array, float]] = []
    for holdout_key in holdout_keys:
        holdouts.append(
            sample_enrichment_distribution(
                initial_params,
                ground,
                key=holdout_key,
                n_samples=holdout_samples,
                sampling=enrichment_sampling,
                envelope_decay=envelope_decay,
                head_count=sampling_head_count,
                walkers=enrichment_sampling_walkers,
                burn_in=enrichment_sampling_burn_in,
                steps_between=enrichment_sampling_steps_between,
                width=enrichment_sampling_width,
                density_batch_size=enrichment_sampling_density_batch_size,
                eps_env=eps_env,
                ground_weight=ground_weight,
                source_weight=source_weight,
                head_weight=head_weight,
                q_node_weight=q_node_weight,
                q_node_ground_power=q_node_ground_power,
                q_tail_weight=q_tail_weight,
                q_tail_envelope_decay=q_tail_envelope_decay,
                q_kinetic_weight=q_kinetic_weight,
                source_envelope_core_decay=source_envelope_core_decay,
                source_envelope_diffuse_decay=source_envelope_diffuse_decay,
                aux_source_exponents=aux_source_exponents,
                aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
                aux_source_bond_odd_slater_decays=(aux_source_bond_odd_slater_decays),
                aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                aux_source_bond_odd_ee_slater_decays=(
                    aux_source_bond_odd_ee_slater_decays
                ),
                aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            )
        )
    holdout_points, holdout_density, _ = holdouts[0]
    finite_holdout_pmoves = [
        item[2] for item in holdouts if np.isfinite(float(item[2]))
    ]
    holdout_pmove = (
        float(np.mean(finite_holdout_pmoves)) if finite_holdout_pmoves else float("nan")
    )
    if np.isfinite(train_pmove) or np.isfinite(holdout_pmove):
        print(
            "response_enrichment_sampling "
            f"attempt={attempt:02d} mode={enrichment_sampling} "
            f"head_count={sampling_head_count} "
            f"validation_holdouts={validation_holdout_count} "
            f"train_pmove~{train_pmove:.3f} "
            f"holdout_pmove~{holdout_pmove:.3f}"
        )
    holdout_points_np = np.asarray(holdout_points)
    holdout_density_np = np.asarray(holdout_density)
    region_cache_by_count: dict[int, np.ndarray] = {}

    def holdout_cache_indices(requested: int, purpose: str) -> np.ndarray:
        sample_count = min(int(requested), int(holdout_points_np.shape[0]))
        if not region_balanced_cache:
            return np.arange(sample_count, dtype=np.int64)
        if sample_count not in region_cache_by_count:
            indices, region_diagnostics = region_balanced_holdout_indices(
                ground,
                holdout_points_np,
                n_select=sample_count,
                seed=104_729 + 1009 * int(attempt) + int(sample_count),
                node_quantile=region_node_quantile,
                tail_quantile=region_tail_quantile,
                en_cusp_radius=region_en_cusp_radius,
                ee_cusp_radius=region_ee_cusp_radius,
                batch_size=strong_residual_batch_size,
            )
            region_cache_by_count[sample_count] = indices
            print(
                "response_enrichment_region_balanced_cache "
                f"attempt={attempt:02d} purpose={purpose} "
                f"samples={sample_count} "
                f"{region_balanced_summary_text(region_diagnostics)}"
            )
        return region_cache_by_count[sample_count]

    omegas = jnp.asarray(residual_omegas)
    omega_weights = jnp.asarray(residual_omega_weights)
    source_weights = jnp.asarray(residual_source_weights)

    def make_candidate_optimizer(step_size: float):
        return optax.adam(step_size)

    optimizer = make_candidate_optimizer(learning_rate)
    opt_state = optimizer.init(candidate_params)

    def loss_fn(
        local_candidate_params: Params,
        batch_points: jax.Array,
        batch_density: jax.Array,
    ):
        local_params = with_candidate(local_candidate_params)
        return gauge_stabilized_enrichment_training_loss(
            local_params,
            ground,
            batch_points,
            batch_density,
            active_heads=active_heads,
            candidate_heads=candidate_heads,
            omegas=omegas,
            omega_weights=omega_weights,
            source_weights=source_weights,
            eta=residual_eta,
            lambda_rough=lambda_rough,
            delta=delta,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        )

    @jax.jit
    def step(candidate_params: Params, opt_state: optax.OptState, idx: jax.Array):
        (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            candidate_params, points[idx], density[idx]
        )
        updates, opt_state = optimizer.update(grads, opt_state, candidate_params)
        candidate_params = optax.apply_updates(candidate_params, updates)
        return candidate_params, opt_state, loss, stats

    @jax.jit
    def evaluate_on(
        candidate_params: Params, eval_points: jax.Array, eval_density: jax.Array
    ):
        return loss_fn(candidate_params, eval_points, eval_density)

    def evaluate(candidate_params: Params):
        return evaluate_on(candidate_params, holdout_points, holdout_density)

    def collect_holdout_stats(local_candidate_params: Params) -> dict[str, np.ndarray]:
        captures = []
        objectives = []
        for eval_points, eval_density, _ in holdouts:
            _, stats = evaluate_on(local_candidate_params, eval_points, eval_density)
            captures.append(float(stats["capture"]))
            objectives.append(float(stats["objective"]))
        return {
            "captures": np.asarray(captures, dtype=np.float64),
            "objectives": np.asarray(objectives, dtype=np.float64),
        }

    def aggregate_holdout_stats(local_candidate_params: Params) -> dict[str, float]:
        stats = collect_holdout_stats(local_candidate_params)
        return heldout_objective_selection_summary(
            stats["captures"],
            stats["objectives"],
        )

    warmup_roots = max(1, min(int(ritz_warmup_roots), int(candidate_heads)))
    warmup_optimizer = make_candidate_optimizer(
        learning_rate * float(ritz_warmup_learning_rate_scale)
    )

    @jax.jit
    def warmup_step(
        candidate_params: Params,
        opt_state: optax.OptState,
        idx: jax.Array,
    ):
        (loss, stats), grads = jax.value_and_grad(
            lambda local_candidate_params: ritz_training_loss(
                with_candidate(local_candidate_params),
                ground,
                points[idx],
                density[idx],
                warmup_roots,
                head_start=active_heads,
                head_count=candidate_heads,
            ),
            has_aux=True,
        )(
            candidate_params,
        )
        updates, opt_state = warmup_optimizer.update(
            grads,
            opt_state,
            candidate_params,
        )
        candidate_params = optax.apply_updates(candidate_params, updates)
        return candidate_params, opt_state, loss, stats

    @jax.jit
    def warmup_evaluate_on(
        candidate_params: Params, eval_points: jax.Array, eval_density: jax.Array
    ):
        return ritz_training_loss(
            with_candidate(candidate_params),
            ground,
            eval_points,
            eval_density,
            warmup_roots,
            head_start=active_heads,
            head_count=candidate_heads,
        )

    def warmup_evaluate(candidate_params: Params):
        return warmup_evaluate_on(candidate_params, holdout_points, holdout_density)

    def stats_are_finite(loss: jax.Array, stats: dict[str, jax.Array]) -> bool:
        return np.isfinite(float(loss)) and all(
            np.isfinite(float(value)) for value in stats.values()
        )

    def rejected_diagnostics(
        *,
        reason: str,
        initial_stats: dict[str, jax.Array],
        final_stats: dict[str, jax.Array],
    ) -> tuple[Params, EnrichmentDiagnostics]:
        diagnostics = EnrichmentDiagnostics(
            accepted=False,
            active_heads_before=int(active_heads),
            candidate_heads=int(candidate_heads),
            accepted_heads=int(active_heads),
            attempt=int(attempt),
            initial_capture=float(initial_stats["capture"]),
            final_capture=float(final_stats["capture"]),
            capture_ratio=float("nan"),
            initial_objective=float(initial_stats["objective"]),
            final_objective=float(final_stats["objective"]),
            objective_delta=(
                float(final_stats["objective"]) - float(initial_stats["objective"])
            ),
            moment_norm_rel_error=float("inf"),
            moment_first_rel_error=float("inf"),
            overlap_condition=float("inf"),
            holdout_pass_fraction=0.0,
            holdout_pass_count=0,
            holdout_count=validation_holdout_count,
            source_bright_passed=False,
            accepted_reason="rejected",
        )
        print(
            "response_enrichment_validate "
            f"attempt={attempt:02d} accepted=False reason={reason} "
            f"accepted_heads={diagnostics.accepted_heads} "
            f"initial_capture={diagnostics.initial_capture:.8e} "
            f"final_capture={diagnostics.final_capture:.8e} "
            f"objective_delta={diagnostics.objective_delta:.8e}"
        )
        return fixed_basis_params, diagnostics

    initial_loss, initial_first_holdout_stats = evaluate(initial_candidate_params)
    initial_stats = aggregate_holdout_stats(initial_candidate_params)
    if not stats_are_finite(
        initial_loss, initial_first_holdout_stats
    ) or not np.isfinite(float(initial_stats["objective_min"])):
        return rejected_diagnostics(
            reason="initial_nonfinite",
            initial_stats=initial_stats,
            final_stats=initial_stats,
        )
    active_pole = float("inf")
    active_pole_validation = PoleValidationDiagnostics(
        pole_median=float("inf"),
        pole_mean=float("inf"),
        pole_std=float("inf"),
        pole_spread=float("inf"),
        pole_min=float("inf"),
        pole_max=float("inf"),
        pass_count=0,
        total_count=validation_holdout_count,
        moment_norm_rel_error_max=float("inf"),
        moment_first_rel_error_max=float("inf"),
        overlap_condition_max=float("inf"),
    )
    if ritz_accept or source_bright_gate:
        active_pole_validation = validate_bright_pole_on_holdouts(
            fixed_basis_params,
            ground,
            holdouts,
            head_count=active_heads,
            overlap_cutoff=overlap_cutoff,
            root_floor=ritz_accept_root_floor,
            min_weight=ritz_accept_min_weight,
            max_moment_rel_error=max_moment_rel_error,
            max_overlap_condition=max_overlap_condition,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        )
        active_pole = active_pole_validation.pole_median
    best_candidate_params = initial_candidate_params
    best_params = initial_params
    best_stats = initial_stats
    best_epoch = -1
    best_ritz_pole = float("inf")
    print(
        "response_enrichment_initial "
        f"attempt={attempt:02d} "
        f"active_heads={active_heads} "
        f"candidate_heads={candidate_heads} "
        f"loss={float(initial_loss):.8f} "
        f"objective={float(initial_stats['objective']):.8e} "
        f"capture={float(initial_stats['capture']):.8e}"
    )
    rng = np.random.default_rng(37)
    validation_every = max(1, validation_every)
    warmup_epochs = max(0, int(ritz_warmup_epochs))
    if warmup_epochs:
        warmup_opt_state = warmup_optimizer.init(candidate_params)
        for epoch in range(warmup_epochs):
            idx = jnp.asarray(rng.choice(train_samples, size=batch_size, replace=False))
            (
                candidate_params,
                warmup_opt_state,
                warmup_loss,
                warmup_stats,
            ) = warmup_step(
                candidate_params,
                warmup_opt_state,
                idx,
            )
            if not stats_are_finite(warmup_loss, warmup_stats):
                print(
                    "response_enrichment_ritz_warmup_nonfinite "
                    f"attempt={attempt:02d} epoch={epoch:05d}"
                )
                break
            if epoch % validation_every == 0 or epoch == warmup_epochs - 1:
                holdout_stats = aggregate_holdout_stats(candidate_params)
                if is_better_heldout_objective_summary(holdout_stats, best_stats):
                    best_candidate_params = candidate_params
                    best_params = with_candidate(candidate_params)
                    best_stats = holdout_stats
                    best_epoch = -(epoch + 1)
                if ritz_accept:
                    pole_validation = validate_bright_pole_on_holdouts(
                        with_candidate(candidate_params),
                        ground,
                        holdouts,
                        head_count=active_heads + candidate_heads,
                        overlap_cutoff=overlap_cutoff,
                        root_floor=ritz_accept_root_floor,
                        min_weight=ritz_accept_min_weight,
                        max_moment_rel_error=max_moment_rel_error,
                        max_overlap_condition=max_overlap_condition,
                        aux_source_exponents=aux_source_exponents,
                        aux_source_dipole_radial_powers=(
                            aux_source_dipole_radial_powers
                        ),
                        aux_source_dipole_radial_scale=(aux_source_dipole_radial_scale),
                        aux_source_atom_odd_exponents=(aux_source_atom_odd_exponents),
                        aux_source_atom_odd_slater_decays=(
                            aux_source_atom_odd_slater_decays
                        ),
                        aux_source_bond_odd_slater_decays=(
                            aux_source_bond_odd_slater_decays
                        ),
                        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                        aux_source_bond_odd_ee_slater_decays=(
                            aux_source_bond_odd_ee_slater_decays
                        ),
                        aux_source_bond_odd_ee_scales=(aux_source_bond_odd_ee_scales),
                    )
                    pole = pole_validation.pole_median
                    pole_improvement = active_pole - pole
                    if (
                        ritz_accept
                        and pole_validation_passed(
                            pole_validation,
                            max_spread=ritz_validation_max_spread,
                            min_pass_fraction=ritz_validation_min_pass_fraction,
                        )
                        and pole < best_ritz_pole
                        and pole_improvement >= ritz_accept_min_pole_improvement
                    ):
                        best_ritz_pole = pole
            if epoch % log_every == 0 or epoch == warmup_epochs - 1:
                _, warmup_holdout_stats = warmup_evaluate(candidate_params)
                pole_text = ""
                if ritz_accept:
                    pole_text = (
                        f" active_pole={active_pole:.8f}"
                        f" best_bright_pole={best_ritz_pole:.8f}"
                        f" active_pass={active_pole_validation.pass_count}"
                        f"/{active_pole_validation.total_count}"
                    )
                print(
                    "response_enrichment_ritz_warmup "
                    f"attempt={attempt:02d} "
                    f"epoch={epoch:05d} loss={float(warmup_loss):.8f} "
                    f"root0={float(warmup_stats['root0']):.8f} "
                    f"holdout_root0={float(warmup_holdout_stats['root0']):.8f}"
                    f"{pole_text}"
                )
        candidate_params = best_candidate_params
        opt_state = optimizer.init(candidate_params)

    for epoch in range(epochs):
        idx = jnp.asarray(rng.choice(train_samples, size=batch_size, replace=False))
        candidate_params, opt_state, loss, stats = step(
            candidate_params,
            opt_state,
            idx,
        )
        if not stats_are_finite(loss, stats):
            print(
                "response_enrichment_train_nonfinite "
                f"attempt={attempt:02d} epoch={epoch:05d}"
            )
            break
        if epoch % validation_every == 0 or epoch == epochs - 1:
            holdout_stats = aggregate_holdout_stats(candidate_params)
            if is_better_heldout_objective_summary(holdout_stats, best_stats):
                best_candidate_params = candidate_params
                best_params = with_candidate(candidate_params)
                best_stats = holdout_stats
                best_epoch = epoch
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(
                "response_enrichment_train "
                f"attempt={attempt:02d} "
                f"epoch={epoch:05d} loss={float(loss):.8f} "
                f"objective={float(stats['objective']):.8e} "
                f"capture={float(stats['capture']):.8e} "
                f"redundancy={float(stats['redundancy']):.3e} "
                f"roughness={float(stats['roughness']):.3e} "
                f"local_action_trace={float(stats['local_action_trace']):.3e}"
            )

    polish_epochs = max(0, int(strong_polish_epochs))
    if polish_epochs:
        polish_count = min(max(1, int(strong_polish_samples)), int(train_samples))
        polish_batch = min(max(1, int(strong_polish_batch_size)), polish_count)
        polish_indices = rng.choice(train_samples, size=polish_count, replace=False)
        polish_points_np = np.asarray(points)[polish_indices]
        polish_density_np = np.asarray(density)[polish_indices]
        polish_head_count = active_heads + candidate_heads
        try:
            (
                polish_projection_coeff,
                polish_response_coeffs,
                polish_z_values,
            ) = strong_residual_polish_coefficients_from_density(
                best_params,
                ground,
                polish_points_np,
                polish_density_np,
                head_count=polish_head_count,
                omegas=strong_residual_omegas,
                eta=residual_eta,
                source_index=strong_residual_source_index,
                aux_source_exponents=aux_source_exponents,
                aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
                aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
                aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                aux_source_bond_odd_ee_slater_decays=(
                    aux_source_bond_odd_ee_slater_decays
                ),
                aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
                batch_size=polish_batch,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(
                "response_enrichment_strong_polish_skip "
                f"attempt={attempt:02d} reason={type(exc).__name__}"
            )
        else:
            polish_points = jnp.asarray(polish_points_np)
            polish_density = jnp.asarray(polish_density_np)
            polish_optimizer = make_candidate_optimizer(
                learning_rate * float(strong_polish_learning_rate_scale)
            )
            polish_opt_state = polish_optimizer.init(best_candidate_params)
            polish_candidate_params = best_candidate_params

            @jax.jit
            def polish_step(
                local_candidate_params: Params,
                local_opt_state: optax.OptState,
                idx: jax.Array,
            ):
                (polish_loss, polish_stats), polish_grads = jax.value_and_grad(
                    lambda candidate: strong_residual_polish_loss(
                        with_candidate(candidate),
                        ground,
                        polish_points[idx],
                        polish_density[idx],
                        polish_projection_coeff,
                        polish_response_coeffs,
                        polish_z_values,
                        head_count=polish_head_count,
                        source_index=strong_residual_source_index,
                        clip=strong_polish_clip,
                        aux_source_exponents=aux_source_exponents,
                        aux_source_dipole_radial_powers=(
                            aux_source_dipole_radial_powers
                        ),
                        aux_source_dipole_radial_scale=(aux_source_dipole_radial_scale),
                        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                        aux_source_atom_odd_slater_decays=(
                            aux_source_atom_odd_slater_decays
                        ),
                        aux_source_bond_odd_slater_decays=(
                            aux_source_bond_odd_slater_decays
                        ),
                        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                        aux_source_bond_odd_ee_slater_decays=(
                            aux_source_bond_odd_ee_slater_decays
                        ),
                        aux_source_bond_odd_ee_scales=(aux_source_bond_odd_ee_scales),
                    ),
                    has_aux=True,
                )(
                    local_candidate_params,
                )
                polish_updates, local_opt_state = polish_optimizer.update(
                    polish_grads, local_opt_state, local_candidate_params
                )
                local_candidate_params = optax.apply_updates(
                    local_candidate_params,
                    polish_updates,
                )
                return (
                    local_candidate_params,
                    local_opt_state,
                    polish_loss,
                    polish_stats,
                )

            for polish_epoch in range(polish_epochs):
                polish_idx = jnp.asarray(
                    rng.choice(polish_count, size=polish_batch, replace=False)
                )
                (
                    polish_candidate_params,
                    polish_opt_state,
                    polish_loss,
                    polish_stats,
                ) = polish_step(
                    polish_candidate_params,
                    polish_opt_state,
                    polish_idx,
                )
                if not stats_are_finite(polish_loss, polish_stats):
                    print(
                        "response_enrichment_strong_polish_nonfinite "
                        f"attempt={attempt:02d} epoch={polish_epoch:05d}"
                    )
                    break
                if polish_epoch % log_every == 0 or polish_epoch == polish_epochs - 1:
                    polish_holdout_stats = aggregate_holdout_stats(
                        polish_candidate_params
                    )
                    print(
                        "response_enrichment_strong_polish "
                        f"attempt={attempt:02d} "
                        f"epoch={polish_epoch:05d} "
                        f"loss={float(polish_loss):.8e} "
                        f"epsilon_max={float(polish_stats['epsilon_max']):.3e} "
                        f"holdout_objective="
                        f"{float(polish_holdout_stats['objective']):.8e} "
                        f"holdout_capture="
                        f"{float(polish_holdout_stats['capture']):.8e}"
                    )
            final_polish_stats = aggregate_holdout_stats(polish_candidate_params)
            initial_polish_raw = collect_holdout_stats(initial_candidate_params)
            final_polish_raw = collect_holdout_stats(polish_candidate_params)
            polish_acceptance = residual_holdout_acceptance_summary(
                initial_polish_raw["captures"],
                final_polish_raw["captures"],
                initial_polish_raw["objectives"],
                final_polish_raw["objectives"],
                min_relative_improvement=min_relative_improvement,
                min_capture=min_capture,
                min_objective_improvement=min_objective_improvement,
                require_training_improvement=require_training_improvement,
            )
            polish_pass_fraction = float(polish_acceptance["pass_fraction"])
            polish_passed = polish_pass_fraction >= float(holdout_min_pass_fraction)
            if polish_passed and is_better_heldout_objective_summary(
                final_polish_stats, best_stats
            ):
                best_candidate_params = polish_candidate_params
                best_params = with_candidate(polish_candidate_params)
                best_stats = final_polish_stats
                best_epoch = epochs + polish_epochs
                print(
                    "response_enrichment_strong_polish_accept "
                    f"attempt={attempt:02d} "
                    f"holdout_pass={int(polish_acceptance['pass_count'])}"
                    f"/{int(polish_acceptance['count'])} "
                    f"holdout_objective={float(best_stats['objective']):.8e} "
                    f"holdout_capture={float(best_stats['capture']):.8e}"
                )
            else:
                print(
                    "response_enrichment_strong_polish_reject "
                    f"attempt={attempt:02d} "
                    f"holdout_pass={int(polish_acceptance['pass_count'])}"
                    f"/{int(polish_acceptance['count'])} "
                    f"holdout_objective="
                    f"{float(final_polish_stats['objective']):.8e} "
                    f"holdout_capture={float(final_polish_stats['capture']):.8e}"
                )

    gauge_count = min(512, int(holdout_points_np.shape[0]))
    if gauge_count > 0:
        gauge_indices = holdout_cache_indices(gauge_count, "gauge")
        try:
            normalized_candidate_params = gauge_normalize_candidate_block(
                fixed_basis_params,
                best_candidate_params,
                ground,
                jnp.asarray(holdout_points_np[gauge_indices]),
                jnp.asarray(holdout_density_np[gauge_indices]),
                active_heads=active_heads,
                candidate_heads=candidate_heads,
                omegas=omegas,
                eta=residual_eta,
                delta=delta,
                aux_source_exponents=aux_source_exponents,
                aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
                aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
                aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                aux_source_bond_odd_ee_slater_decays=(
                    aux_source_bond_odd_ee_slater_decays
                ),
                aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(
                "response_enrichment_gauge_normalize_skip "
                f"attempt={attempt:02d} reason={type(exc).__name__}"
            )
        else:
            normalized_stats = aggregate_holdout_stats(normalized_candidate_params)
            if np.isfinite(float(normalized_stats["objective"])):
                best_candidate_params = normalized_candidate_params
                best_params = with_candidate(best_candidate_params)
                best_stats = normalized_stats
                print(
                    "response_enrichment_gauge_normalize "
                    f"attempt={attempt:02d} samples={gauge_count} "
                    f"objective={float(best_stats['objective']):.8e} "
                    f"capture={float(best_stats['capture']):.8e}"
                )

    validation_params = best_params
    validation_candidate_heads = int(candidate_heads)
    accepted_head_count = active_heads + validation_candidate_heads
    strong_oracle_diagnostics = {
        "strong_oracle_train_epsilon_old_max": float("nan"),
        "strong_oracle_train_epsilon_oracle_max": float("nan"),
        "strong_oracle_validation_epsilon_old_max": float("nan"),
        "strong_oracle_validation_epsilon_oracle_max": float("nan"),
        "strong_oracle_validation_ratio_max": float("nan"),
        "strong_oracle_validation_ratio_winsor99_max": float("nan"),
        "strong_oracle_validation_ratio_p95_max": float("nan"),
        "strong_oracle_validation_ratio_p99_max": float("nan"),
        "strong_oracle_validation_ratio_pointwise_max_max": float("nan"),
        "strong_oracle_validation_improvement_min": float("nan"),
        "strong_oracle_train_epsilon2_improvement_min": float("nan"),
        "strong_oracle_validation_epsilon2_improvement_min": float("nan"),
        "strong_oracle_validation_relative_epsilon2_improvement_min": float("nan"),
        "strong_oracle_validation_winsor99_epsilon2_improvement_min": float("nan"),
        "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min": (
            float("nan")
        ),
        "strong_oracle_action_consistency_l2_max": float("nan"),
        "strong_oracle_action_consistency_p95_max": float("nan"),
        "strong_oracle_action_consistency_p99_max": float("nan"),
        "strong_oracle_action_consistency_pointwise_max_max": float("nan"),
        "strong_oracle_candidate_value_norm_min": float("nan"),
        "strong_oracle_candidate_value_norm_max": float("nan"),
        "strong_oracle_candidate_action_norm_min": float("nan"),
        "strong_oracle_candidate_action_norm_max": float("nan"),
        "strong_oracle_candidate_action_condition_max": float("nan"),
    }
    if strong_oracle_samples > 0:
        oracle_samples = min(int(strong_oracle_samples), int(holdout_points.shape[0]))
        if oracle_samples >= 2:
            oracle_indices = holdout_cache_indices(oracle_samples, "strong_oracle")
            oracle_samples = int(oracle_indices.shape[0])
            oracle = None
            try:
                oracle = strong_oracle_linear_combination_diagnostic(
                    best_params,
                    ground,
                    holdout_points_np[oracle_indices],
                    holdout_density_np[oracle_indices],
                    active_heads=active_heads,
                    candidate_heads=candidate_heads,
                    omegas=strong_residual_omegas,
                    eta=residual_eta,
                    source_index=strong_residual_source_index,
                    ridge=strong_oracle_ridge,
                    action_schur=strong_oracle_action_schur,
                    action_schur_ridge=strong_oracle_action_schur_ridge,
                    aux_source_exponents=aux_source_exponents,
                    aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                    aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                    aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                    aux_source_atom_odd_slater_decays=(
                        aux_source_atom_odd_slater_decays
                    ),
                    aux_source_bond_odd_slater_decays=(
                        aux_source_bond_odd_slater_decays
                    ),
                    aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                    aux_source_bond_odd_ee_slater_decays=(
                        aux_source_bond_odd_ee_slater_decays
                    ),
                    aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
                    batch_size=strong_residual_batch_size,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                print(
                    "response_enrichment_strong_oracle_skip "
                    f"attempt={attempt:02d} reason={type(exc).__name__}"
                )
            if oracle is not None:
                train_old = np.asarray(
                    oracle["strong_oracle_train_epsilon_old"], dtype=np.float64
                )
                train_new = np.asarray(
                    oracle["strong_oracle_train_epsilon_oracle"], dtype=np.float64
                )
                val_old = np.asarray(
                    oracle["strong_oracle_validation_epsilon_old"], dtype=np.float64
                )
                val_new = np.asarray(
                    oracle["strong_oracle_validation_epsilon_oracle"],
                    dtype=np.float64,
                )
                val_ratio = np.asarray(
                    oracle["strong_oracle_validation_ratio"], dtype=np.float64
                )
                val_ratio_w99 = np.asarray(
                    oracle["strong_oracle_validation_ratio_winsor99"],
                    dtype=np.float64,
                )
                val_ratio_p99 = np.asarray(
                    oracle["strong_oracle_validation_ratio_p99"], dtype=np.float64
                )
                val_ratio_point_max = np.asarray(
                    oracle["strong_oracle_validation_ratio_pointwise_max"],
                    dtype=np.float64,
                )
                strong_oracle_diagnostics.update(
                    {
                        "strong_oracle_train_epsilon_old_max": float(np.max(train_old)),
                        "strong_oracle_train_epsilon_oracle_max": float(
                            np.max(train_new)
                        ),
                        "strong_oracle_validation_epsilon_old_max": float(
                            np.max(val_old)
                        ),
                        "strong_oracle_validation_epsilon_oracle_max": float(
                            np.max(val_new)
                        ),
                        "strong_oracle_validation_ratio_max": float(np.max(val_ratio)),
                        "strong_oracle_validation_ratio_winsor99_max": float(
                            np.max(val_ratio_w99)
                        ),
                        "strong_oracle_validation_ratio_p95_max": float(
                            oracle["strong_oracle_validation_ratio_p95_max"]
                        ),
                        "strong_oracle_validation_ratio_p99_max": float(
                            np.max(val_ratio_p99)
                        ),
                        "strong_oracle_validation_ratio_pointwise_max_max": (
                            float(np.max(val_ratio_point_max))
                        ),
                        "strong_oracle_validation_improvement_min": float(
                            oracle["strong_oracle_validation_improvement_min"]
                        ),
                        "strong_oracle_train_epsilon2_improvement_min": float(
                            oracle["strong_oracle_train_epsilon2_improvement_min"]
                        ),
                        "strong_oracle_validation_epsilon2_improvement_min": (
                            float(
                                oracle[
                                    "strong_oracle_validation_epsilon2_improvement_min"
                                ]
                            )
                        ),
                        "strong_oracle_validation_relative_epsilon2_improvement_min": (
                            float(
                                oracle[
                                    "strong_oracle_validation_relative_epsilon2_improvement_min"
                                ]
                            )
                        ),
                        "strong_oracle_validation_winsor99_epsilon2_improvement_min": (
                            float(
                                oracle[
                                    "strong_oracle_validation_winsor99_epsilon2_improvement_min"
                                ]
                            )
                        ),
                        (
                            "strong_oracle_validation_winsor99_"
                            "relative_epsilon2_improvement_min"
                        ): (
                            float(
                                oracle[
                                    "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min"
                                ]
                            )
                        ),
                        "strong_oracle_action_consistency_l2_max": float(
                            oracle["strong_oracle_action_consistency_l2_max"]
                        ),
                        "strong_oracle_action_consistency_p95_max": float(
                            oracle["strong_oracle_action_consistency_p95_max"]
                        ),
                        "strong_oracle_action_consistency_p99_max": float(
                            oracle["strong_oracle_action_consistency_p99_max"]
                        ),
                        "strong_oracle_action_consistency_pointwise_max_max": (
                            float(
                                oracle[
                                    "strong_oracle_action_consistency_pointwise_max_max"
                                ]
                            )
                        ),
                        "strong_oracle_candidate_value_norm_min": float(
                            oracle["strong_oracle_candidate_value_norm_min"]
                        ),
                        "strong_oracle_candidate_value_norm_max": float(
                            oracle["strong_oracle_candidate_value_norm_max"]
                        ),
                        "strong_oracle_candidate_action_norm_min": float(
                            oracle["strong_oracle_candidate_action_norm_min"]
                        ),
                        "strong_oracle_candidate_action_norm_max": float(
                            oracle["strong_oracle_candidate_action_norm_max"]
                        ),
                        "strong_oracle_candidate_action_condition_max": float(
                            oracle["strong_oracle_candidate_action_condition_max"]
                        ),
                    }
                )
                strong_oracle_diagnostics.update(
                    strong_oracle_diagnostics_from_result(oracle)
                )
                candidate_value_norm_min = strong_oracle_diagnostics[
                    "strong_oracle_candidate_value_norm_min"
                ]
                candidate_action_norm_min = strong_oracle_diagnostics[
                    "strong_oracle_candidate_action_norm_min"
                ]
                candidate_action_condition_max = strong_oracle_diagnostics[
                    "strong_oracle_candidate_action_condition_max"
                ]
                validation_action_rel_improvement = strong_oracle_diagnostics[
                    "strong_oracle_validation_relative_epsilon2_improvement_min"
                ]
                validation_w99_action_rel_improvement = strong_oracle_diagnostics[
                    "strong_oracle_validation_winsor99_relative_epsilon2_improvement_min"
                ]
                raw_w99 = float(
                    oracle["strong_oracle_raw_validation_ratio_winsor99_max"]
                )
                raw_action_rel_improvement = float(
                    oracle[
                        "strong_oracle_raw_validation_relative_epsilon2_improvement_min"
                    ]
                )
                print(
                    "response_enrichment_strong_oracle "
                    f"attempt={attempt:02d} label=raw-pool "
                    f"samples={oracle_samples} "
                    "action_mode=ad "
                    f"action_schur={strong_oracle_action_schur} "
                    f"ridge={strong_oracle_ridge:.3e} "
                    f"schur_ridge={strong_oracle_action_schur_ridge:.3e} "
                    f"train_old_max={float(np.max(train_old)):.3e} "
                    f"train_oracle_max={float(np.max(train_new)):.3e} "
                    f"val_old_max={float(np.max(val_old)):.3e} "
                    f"val_oracle_max={float(np.max(val_new)):.3e} "
                    f"val_ratio_max={float(np.max(val_ratio)):.3e} "
                    f"val_ratio_w99_max={float(np.max(val_ratio_w99)):.3e} "
                    f"val_ratio_p99_max={float(np.max(val_ratio_p99)):.3e} "
                    f"val_ratio_pointmax_max="
                    f"{float(np.max(val_ratio_point_max)):.3e} "
                    f"val_action_rel_improve="
                    f"{validation_action_rel_improvement:.3e} "
                    f"val_w99_action_rel_improve="
                    f"{validation_w99_action_rel_improvement:.3e} "
                    f"raw_val_w99={raw_w99:.3e} "
                    f"raw_val_action_rel_improve="
                    f"{raw_action_rel_improvement:.3e} "
                    f"cand_value_norm_min={candidate_value_norm_min:.3e} "
                    f"cand_action_norm_min={candidate_action_norm_min:.3e} "
                    f"cand_action_cond_max={candidate_action_condition_max:.3e}"
                )
    overlap, hamiltonian, _ = source_plus_head_matrices(
        validation_params,
        ground,
        holdout_points,
        holdout_density,
        head_count=accepted_head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    source_vector = np.asarray(overlap)[:, 0:1]
    try:
        spectrum = projected_spectrum(
            np.asarray(overlap),
            np.asarray(hamiltonian),
            source_vector,
            overlap_cutoff=overlap_cutoff,
        )
        moments = moment_diagnostics(
            np.asarray(overlap),
            np.asarray(hamiltonian),
            source_vector,
            spectrum,
            overlap_cutoff=overlap_cutoff,
        )
    except (np.linalg.LinAlgError, ValueError):
        return rejected_diagnostics(
            reason="validation_matrix_invalid",
            initial_stats=initial_stats,
            final_stats=best_stats,
        )
    initial_accept_raw = collect_holdout_stats(initial_candidate_params)
    final_accept_raw = collect_holdout_stats(best_candidate_params)
    holdout_acceptance = residual_holdout_acceptance_summary(
        initial_accept_raw["captures"],
        final_accept_raw["captures"],
        initial_accept_raw["objectives"],
        final_accept_raw["objectives"],
        min_relative_improvement=min_relative_improvement,
        min_capture=min_capture,
        min_objective_improvement=min_objective_improvement,
        require_training_improvement=require_training_improvement,
    )
    candidate_pole_validation = validate_bright_pole_on_holdouts(
        validation_params,
        ground,
        holdouts,
        head_count=accepted_head_count,
        overlap_cutoff=overlap_cutoff,
        root_floor=ritz_accept_root_floor,
        min_weight=ritz_accept_min_weight,
        max_moment_rel_error=max_moment_rel_error,
        max_overlap_condition=max_overlap_condition,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    selected_validation = candidate_pole_validation
    selected_pole = candidate_pole_validation.pole_median
    pole_improvement = active_pole - selected_pole
    pole_requirement_passed = True
    if ritz_accept:
        pole_requirement_passed = (
            pole_validation_passed(
                candidate_pole_validation,
                max_spread=ritz_validation_max_spread,
                min_pass_fraction=ritz_validation_min_pass_fraction,
            )
            and pole_improvement >= ritz_accept_min_pole_improvement
        )
    source_bright_passed = True
    source_bright_shift = float("nan")
    if source_bright_gate:
        source_bright_passed, source_bright_shift = source_bright_gate_passed(
            active_validation=active_pole_validation,
            candidate_validation=candidate_pole_validation,
            active_heads=active_heads,
            max_spread=ritz_validation_max_spread,
            min_pass_fraction=ritz_validation_min_pass_fraction,
            max_regression=source_bright_max_regression,
        )
    validation_passed = source_bright_passed and pole_requirement_passed
    strong_residual_epsilon_max = float("nan")
    strong_residual_epsilon_old_max = float("nan")
    strong_residual_epsilon_over_eta_max = float("nan")
    strong_residual_epsilon_over_eta_old_max = float("nan")
    strong_residual_epsilon_over_eta_ratio = float("nan")
    strong_residual_nonfinite_count = 0
    strong_residual_node_fraction_max = float("nan")
    strong_residual_node_fraction_old_max = float("nan")
    if strong_residual_samples > 0:
        audit_samples = min(int(strong_residual_samples), int(holdout_points.shape[0]))
        audit_indices = holdout_cache_indices(audit_samples, "strong_residual")
        audit_samples = int(audit_indices.shape[0])
        strong_residual_old = strong_residual_audit_from_density(
            fixed_basis_params,
            ground,
            holdout_points_np[audit_indices],
            holdout_density_np[audit_indices],
            head_count=active_heads,
            omegas=strong_residual_omegas,
            eta=residual_eta,
            source_index=strong_residual_source_index,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            batch_size=strong_residual_batch_size,
        )
        strong_residual = strong_residual_audit_from_density(
            validation_params,
            ground,
            holdout_points_np[audit_indices],
            holdout_density_np[audit_indices],
            head_count=accepted_head_count,
            omegas=strong_residual_omegas,
            eta=residual_eta,
            source_index=strong_residual_source_index,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            batch_size=strong_residual_batch_size,
        )
        strong_residual_epsilon_max = float(
            strong_residual["strong_residual_max_epsilon"]
        )
        strong_residual_epsilon_old_max = float(
            strong_residual_old["strong_residual_max_epsilon"]
        )
        strong_residual_epsilon_over_eta_max = float(
            strong_residual["strong_residual_max_epsilon_over_eta"]
        )
        strong_residual_epsilon_over_eta_old_max = float(
            strong_residual_old["strong_residual_max_epsilon_over_eta"]
        )
        strong_residual_nonfinite_count = strong_residual_hard_nonfinite_count(
            strong_residual
        )
        strong_residual_node_fraction_max = strong_residual_region_fraction_max(
            strong_residual,
            "node_tube",
        )
        strong_residual_node_fraction_old_max = strong_residual_region_fraction_max(
            strong_residual_old,
            "node_tube",
        )
        if (
            np.isfinite(strong_residual_epsilon_over_eta_max)
            and np.isfinite(strong_residual_epsilon_over_eta_old_max)
            and strong_residual_epsilon_over_eta_old_max > 0.0
        ):
            strong_residual_epsilon_over_eta_ratio = (
                strong_residual_epsilon_over_eta_max
                / strong_residual_epsilon_over_eta_old_max
            )
        print(
            "response_enrichment_strong_residual "
            f"attempt={attempt:02d} samples={audit_samples} "
            f"source_index={strong_residual_source_index} "
            f"old_max_epsilon={strong_residual_epsilon_old_max:.3e} "
            f"max_epsilon={strong_residual_epsilon_max:.3e} "
            f"old_max_epsilon_over_eta="
            f"{strong_residual_epsilon_over_eta_old_max:.3e} "
            f"max_epsilon_over_eta="
            f"{strong_residual_epsilon_over_eta_max:.3e} "
            f"new_over_old="
            f"{strong_residual_epsilon_over_eta_ratio:.3e} "
            f"nonfinite_count={strong_residual_nonfinite_count} "
            f"node_fraction={strong_residual_node_fraction_max:.3e} "
            f"old_node_fraction={strong_residual_node_fraction_old_max:.3e}"
        )
        region_summary = strong_residual_region_summary_text(strong_residual)
        if region_summary:
            print(
                "response_enrichment_strong_residual_regions "
                f"attempt={attempt:02d} {region_summary}"
            )
        nonfinite_summary = strong_residual_nonfinite_summary_text(strong_residual)
        if nonfinite_summary:
            print(
                "response_enrichment_strong_residual_nonfinite "
                f"attempt={attempt:02d} {nonfinite_summary}"
            )
    acceptance_moments = replace(
        moments,
        norm_rel_error=selected_validation.moment_norm_rel_error_max,
        first_moment_rel_error=selected_validation.moment_first_rel_error_max,
        overlap_condition=selected_validation.overlap_condition_max,
    )
    diagnostics = should_accept_enrichment(
        initial_capture=float(holdout_acceptance["initial_capture"]),
        final_capture=float(holdout_acceptance["final_capture"]),
        initial_objective=float(holdout_acceptance["initial_objective"]),
        final_objective=float(holdout_acceptance["final_objective"]),
        moments=acceptance_moments,
        active_heads=active_heads,
        candidate_heads=validation_candidate_heads,
        attempt=attempt,
        min_relative_improvement=min_relative_improvement,
        min_capture=min_capture,
        min_objective_improvement=min_objective_improvement,
        max_moment_rel_error=max_moment_rel_error,
        max_overlap_condition=max_overlap_condition,
        require_training_improvement=require_training_improvement,
        holdout_capture_ratio_min=float(holdout_acceptance["capture_ratio_min"]),
        holdout_objective_delta_min=float(holdout_acceptance["objective_delta_min"]),
        holdout_pass_fraction=float(holdout_acceptance["pass_fraction"]),
        holdout_pass_count=int(holdout_acceptance["pass_count"]),
        holdout_count=int(holdout_acceptance["count"]),
        min_holdout_pass_fraction=holdout_min_pass_fraction,
        strong_residual_epsilon_max=(
            None if strong_residual_samples <= 0 else strong_residual_epsilon_max
        ),
        strong_residual_epsilon_old_max=(
            None if strong_residual_samples <= 0 else strong_residual_epsilon_old_max
        ),
        strong_residual_epsilon_over_eta_max=(
            None
            if strong_residual_samples <= 0
            else strong_residual_epsilon_over_eta_max
        ),
        strong_residual_epsilon_over_eta_old_max=(
            None
            if strong_residual_samples <= 0
            else strong_residual_epsilon_over_eta_old_max
        ),
        strong_residual_nonfinite_count=(
            0 if strong_residual_samples <= 0 else strong_residual_nonfinite_count
        ),
        strong_residual_node_fraction_max=(
            None if strong_residual_samples <= 0 else strong_residual_node_fraction_max
        ),
        strong_residual_node_fraction_old_max=(
            None
            if strong_residual_samples <= 0
            else strong_residual_node_fraction_old_max
        ),
        max_strong_residual_epsilon_over_eta=(max_strong_residual_epsilon_over_eta),
        max_strong_residual_provisional_ratio=(max_strong_residual_provisional_ratio),
        source_bright_passed=validation_passed,
        source_bright_shift=source_bright_shift,
    )
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    source_norm = float(source_vector[0, 0].real)
    final_pole = first_bright_pole(
        spectrum.excitation_energies,
        weights,
        source_norm=source_norm,
        root_floor=ritz_accept_root_floor,
        min_weight=ritz_accept_min_weight,
    )
    if not np.isfinite(selected_pole):
        selected_pole = final_pole
    if not np.isfinite(pole_improvement):
        pole_improvement = active_pole - selected_pole
    accepted_reason = diagnostics.accepted_reason
    if not diagnostics.accepted and not pole_requirement_passed:
        accepted_reason = "pole_validation_rejected"
    elif not diagnostics.accepted and not source_bright_passed:
        accepted_reason = "source_bright_rejected"
    diagnostics = replace(
        diagnostics,
        moment_norm_rel_error=selected_validation.moment_norm_rel_error_max,
        moment_first_rel_error=selected_validation.moment_first_rel_error_max,
        overlap_condition=selected_validation.overlap_condition_max,
        selected_pole=float(selected_pole),
        pole_improvement=float(pole_improvement),
        pole_spread=selected_validation.pole_spread,
        pole_validation_pass_count=selected_validation.pass_count,
        pole_validation_count=selected_validation.total_count,
        strong_residual_epsilon_max=diagnostics.strong_residual_epsilon_max,
        strong_residual_epsilon_over_eta_max=(
            diagnostics.strong_residual_epsilon_over_eta_max
        ),
        strong_residual_passed=diagnostics.strong_residual_passed,
        **strong_oracle_diagnostics,
        source_bright_passed=source_bright_passed,
        source_bright_shift=source_bright_shift,
        accepted_reason=accepted_reason,
    )
    (
        effective_min_oracle_rel_improvement,
        effective_min_oracle_w99_rel_improvement,
    ) = action_oracle_acceptance_thresholds(
        selection_objective=enrichment_selection_objective,
        min_validation_relative_epsilon2_improvement=(
            strong_oracle_min_validation_relative_epsilon2_improvement
        ),
        min_validation_winsor99_relative_epsilon2_improvement=(
            strong_oracle_min_validation_winsor99_relative_epsilon2_improvement
        ),
    )
    diagnostics = promote_action_oracle_acceptance(
        diagnostics,
        selection_objective=enrichment_selection_objective,
        max_moment_rel_error=max_moment_rel_error,
        max_overlap_condition=max_overlap_condition,
        max_validation_ratio_winsor99=(strong_oracle_max_validation_ratio_winsor99),
        max_validation_ratio_p99=strong_oracle_max_validation_ratio_p99,
        max_validation_ratio_pointwise=(strong_oracle_max_validation_ratio_pointwise),
        min_validation_relative_epsilon2_improvement=(
            effective_min_oracle_rel_improvement
        ),
        min_validation_winsor99_relative_epsilon2_improvement=(
            effective_min_oracle_w99_rel_improvement
        ),
        min_candidate_value_norm=strong_oracle_min_candidate_value_norm,
        min_candidate_action_norm=strong_oracle_min_candidate_action_norm,
        max_candidate_action_condition=(strong_oracle_max_candidate_action_condition),
    )
    diagnostics = apply_strong_oracle_gate(
        diagnostics,
        max_validation_ratio_winsor99=(strong_oracle_max_validation_ratio_winsor99),
        max_validation_ratio_p99=strong_oracle_max_validation_ratio_p99,
        max_validation_ratio_pointwise=(strong_oracle_max_validation_ratio_pointwise),
        min_validation_relative_epsilon2_improvement=(
            effective_min_oracle_rel_improvement
        ),
        min_validation_winsor99_relative_epsilon2_improvement=(
            effective_min_oracle_w99_rel_improvement
        ),
        min_candidate_value_norm=strong_oracle_min_candidate_value_norm,
        min_candidate_action_norm=strong_oracle_min_candidate_action_norm,
        max_candidate_action_condition=(strong_oracle_max_candidate_action_condition),
    )
    print(
        "response_enrichment_validate "
        f"attempt={attempt:02d} "
        f"accepted={diagnostics.accepted} "
        f"reason={diagnostics.accepted_reason} "
        f"accepted_heads={diagnostics.accepted_heads} "
        f"best_epoch={best_epoch:05d} "
        f"initial_capture={diagnostics.initial_capture:.8e} "
        f"final_capture={diagnostics.final_capture:.8e} "
        f"capture_ratio={diagnostics.capture_ratio:.6f} "
        f"holdout_capture_ratio_min={diagnostics.holdout_capture_ratio_min:.6f} "
        f"objective_delta={diagnostics.objective_delta:.8e} "
        f"holdout_objective_delta_min="
        f"{diagnostics.holdout_objective_delta_min:.8e} "
        f"holdout_pass={diagnostics.holdout_pass_count}"
        f"/{diagnostics.holdout_count} "
        f"selected_pole={diagnostics.selected_pole:.10f} "
        f"pole_improvement={diagnostics.pole_improvement:.3e} "
        f"pole_spread={diagnostics.pole_spread:.3e} "
        f"pole_pass={diagnostics.pole_validation_pass_count}"
        f"/{diagnostics.pole_validation_count} "
        f"production_ready={diagnostics.production_ready} "
        f"strong_residual_passed={diagnostics.strong_residual_passed} "
        f"strong_residual_improved={diagnostics.strong_residual_improved} "
        f"strong_residual_old_epsilon_over_eta="
        f"{diagnostics.strong_residual_epsilon_over_eta_old_max:.3e} "
        f"strong_residual_epsilon_over_eta="
        f"{diagnostics.strong_residual_epsilon_over_eta_max:.3e} "
        f"strong_residual_new_over_old="
        f"{diagnostics.strong_residual_epsilon_over_eta_ratio:.3e} "
        f"strong_oracle_passed={diagnostics.strong_oracle_passed} "
        f"strong_oracle_w99="
        f"{diagnostics.strong_oracle_validation_ratio_winsor99_max:.3e} "
        f"source_bright_passed={diagnostics.source_bright_passed} "
        f"source_bright_shift={diagnostics.source_bright_shift:.3e} "
        f"m0_rel={diagnostics.moment_norm_rel_error:.3e} "
        f"m1_rel={diagnostics.moment_first_rel_error:.3e} "
        f"overlap_condition={diagnostics.overlap_condition:.3e}"
    )
    if diagnostics.accepted:
        return validation_params, diagnostics
    return fixed_basis_params, diagnostics


def _train_casscf_external_enrichment_attempt(  # noqa: C901
    params: Params,
    ground: FermiNetGround,
    *,
    key: jax.Array,
    active_heads: int,
    candidate_heads: int,
    attempt: int,
    train_samples: int,
    holdout_samples: int,
    envelope_decay: float,
    validation_holdouts: int,
    enrichment_sampling: str,
    enrichment_sampling_walkers: int,
    enrichment_sampling_burn_in: int,
    enrichment_sampling_steps_between: int,
    enrichment_sampling_width: float,
    enrichment_sampling_density_batch_size: int,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float,
    q_node_ground_power: float,
    q_tail_weight: float,
    q_tail_envelope_decay: float,
    q_kinetic_weight: float,
    source_envelope_core_decay: float,
    source_envelope_diffuse_decay: float,
    residual_omegas: np.ndarray,
    residual_omega_weights: np.ndarray,
    residual_source_weights: np.ndarray,
    residual_eta: float,
    casscf_seed_basis: str,
    casscf_seed_ncas: int,
    casscf_seed_n_roots: int,
    casscf_seed_target_mode: str,
    casscf_seed_samples: int,
    casscf_seed_tau_rel: float,
    casscf_seed_tau_abs: float,
    casscf_seed_ratio_clip: float,
    casscf_seed_finite_difference_step: float,
    casscf_seed_state_average: bool,
    delta: float,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    strong_residual_samples: int,
    strong_residual_omegas: np.ndarray,
    strong_residual_source_index: int,
    strong_residual_batch_size: int,
    max_strong_residual_epsilon_over_eta: float,
    max_strong_residual_provisional_ratio: float,
    overlap_cutoff: float,
    aux_source_exponents: np.ndarray,
    aux_source_dipole_radial_powers: np.ndarray,
    aux_source_dipole_radial_scale: float,
    aux_source_atom_odd_exponents: np.ndarray,
    aux_source_atom_odd_slater_decays: np.ndarray,
    aux_source_bond_odd_slater_decays: np.ndarray,
    aux_source_dipole_ee_scales: np.ndarray,
    aux_source_bond_odd_ee_slater_decays: np.ndarray,
    aux_source_bond_odd_ee_scales: np.ndarray,
) -> tuple[Params, EnrichmentDiagnostics, ExternalCASBasisBlock | None]:
    """Select and validate a direct external CASSCF/FermiNet basis block.

    Returns:
        Unchanged neural parameters, acceptance diagnostics, and an accepted
        external CAS block when the held-out checks pass.

    Raises:
        RuntimeError: If CASSCF derivative data expected by the block is absent.
        ValueError: If the requested seed sample count is invalid.
    """
    fixed_basis_params = (
        params
        if is_response_block_dictionary(params)
        else empty_response_basis_params()
    )
    validation_holdout_count = max(1, int(validation_holdouts))
    train_key, holdout_key = jax.random.split(key, 2)
    points, density, train_pmove = sample_enrichment_distribution(
        fixed_basis_params,
        ground,
        key=train_key,
        n_samples=train_samples,
        sampling=enrichment_sampling,
        envelope_decay=envelope_decay,
        head_count=active_heads,
        walkers=enrichment_sampling_walkers,
        burn_in=enrichment_sampling_burn_in,
        steps_between=enrichment_sampling_steps_between,
        width=enrichment_sampling_width,
        density_batch_size=enrichment_sampling_density_batch_size,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
        source_envelope_core_decay=source_envelope_core_decay,
        source_envelope_diffuse_decay=source_envelope_diffuse_decay,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    holdout_points, holdout_density, holdout_pmove = sample_enrichment_distribution(
        fixed_basis_params,
        ground,
        key=holdout_key,
        n_samples=holdout_samples,
        sampling=enrichment_sampling,
        envelope_decay=envelope_decay,
        head_count=active_heads,
        walkers=enrichment_sampling_walkers,
        burn_in=enrichment_sampling_burn_in,
        steps_between=enrichment_sampling_steps_between,
        width=enrichment_sampling_width,
        density_batch_size=enrichment_sampling_density_batch_size,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
        source_envelope_core_decay=source_envelope_core_decay,
        source_envelope_diffuse_decay=source_envelope_diffuse_decay,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    if np.isfinite(train_pmove) or np.isfinite(holdout_pmove):
        print(
            "response_enrichment_sampling "
            f"attempt={attempt:02d} mode={enrichment_sampling} "
            f"head_count={active_heads} validation_holdouts=1 "
            f"train_pmove~{train_pmove:.3f} holdout_pmove~{holdout_pmove:.3f}"
        )
    seed_sample_count = min(int(casscf_seed_samples), int(points.shape[0]))
    if seed_sample_count < 1:
        msg = "--casscf-seed-samples must select at least one sample"
        raise ValueError(msg)
    seed_points = points[:seed_sample_count]
    seed_density = density[:seed_sample_count]
    casscf_model = build_casscf_seed_model(
        ground,
        basis=casscf_seed_basis,
        ncas=casscf_seed_ncas,
        n_roots=casscf_seed_n_roots,
        source_axis=2,
        state_average=casscf_seed_state_average,
    )
    seed_bank = evaluate_casscf_ratio_carriers(
        casscf_model,
        np.asarray(seed_points),
        target_mode=casscf_seed_target_mode,
        correction_omegas=np.asarray(residual_omegas, dtype=np.float64),
        correction_eta=residual_eta,
        tau_rel=casscf_seed_tau_rel,
        tau_abs=casscf_seed_tau_abs,
        ratio_clip=casscf_seed_ratio_clip,
        derivatives=True,
        finite_difference_step=casscf_seed_finite_difference_step,
    )
    if seed_bank.gradients is None or seed_bank.laplacians is None:
        msg = "CASSCF seed derivatives were not evaluated"
        raise RuntimeError(msg)
    old_values, old_hbar = source_aux_and_head_values_and_hbar(
        fixed_basis_params,
        ground,
        seed_points,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        head_count=active_heads,
    )
    bank_values, bank_hbar = carrier_bank_values_and_hbar_from_arrays(
        ground,
        seed_points,
        seed_bank.values,
        seed_bank.gradients,
        seed_bank.laplacians,
    )
    ground_values = jax.vmap(ground_value_single, (None, 0))(ground, seed_points)
    ground_hbar = ground_hbar_values(ground, seed_points, ground_values)
    old_values, old_hbar, _ = project_values_hbar_against_ground(
        old_values,
        old_hbar,
        ground_values,
        ground_hbar,
        seed_density,
    )
    bank_values, bank_hbar, _ = project_values_hbar_against_ground(
        bank_values,
        bank_hbar,
        ground_values,
        ground_hbar,
        seed_density,
    )
    aux_source_count = _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    seed_coefficients, seed_diagnostics = (
        residual_aligned_seed_coefficients_from_columns(
            old_values,
            old_hbar,
            bank_values,
            bank_hbar,
            seed_density,
            source_count=1 + aux_source_count,
            active_heads=active_heads,
            candidate_heads=candidate_heads,
            omegas=jnp.asarray(residual_omegas),
            omega_weights=jnp.asarray(residual_omega_weights),
            source_weights=jnp.asarray(residual_source_weights),
            eta=residual_eta,
            delta=delta,
        )
    )
    selected = int(seed_diagnostics["seed_selected"])
    selected_coefficients = np.zeros((seed_bank.values.shape[1], 0), dtype=np.float64)
    if selected > 0:
        raw_selected = np.asarray(seed_coefficients[:, :selected], dtype=np.float64)
        norms = np.linalg.norm(raw_selected, axis=0)
        selected_coefficients = raw_selected[:, norms > 0]
    print(
        "response_enrichment_casscf_seed_action "
        f"attempt={attempt:02d} samples={seed_sample_count} "
        f"method={seed_bank.method} basis={seed_bank.basis} "
        f"ncas={casscf_model.ncas} roots={len(casscf_model.ci_vectors)} "
        f"bank_features={seed_bank.values.shape[1]} tau={seed_bank.tau:.3e} "
        f"rank={seed_diagnostics['seed_action_rank']:.0f} "
        f"selected={selected_coefficients.shape[1]} "
        f"top_capture={seed_diagnostics['seed_top_capture']:.3e} "
        f"trace={seed_diagnostics['seed_trace']:.3e} "
        f"coeff_norm_min={seed_diagnostics['seed_coeff_norm_min']:.3e} "
        f"coeff_norm_max={seed_diagnostics['seed_coeff_norm_max']:.3e} "
        f"invalid={bool(seed_diagnostics['seed_invalid'])}"
    )

    def diagnostics_for_rejection(reason: str) -> EnrichmentDiagnostics:
        print(
            "response_enrichment_casscf_external_validate "
            f"attempt={attempt:02d} accepted=False reason={reason} "
            f"external_basis=0 active_heads={active_heads}"
        )
        return EnrichmentDiagnostics(
            accepted=False,
            active_heads_before=active_heads,
            candidate_heads=candidate_heads,
            accepted_heads=active_heads,
            attempt=attempt,
            initial_capture=0.0,
            final_capture=0.0,
            capture_ratio=0.0,
            initial_objective=0.0,
            final_objective=0.0,
            objective_delta=0.0,
            moment_norm_rel_error=float("inf"),
            moment_first_rel_error=float("inf"),
            overlap_condition=float("inf"),
            holdout_pass_fraction=0.0,
            holdout_pass_count=0,
            holdout_count=validation_holdout_count,
            production_ready=False,
            strong_residual_improved=False,
            strong_residual_passed=False,
            strong_oracle_passed=False,
            accepted_reason=reason,
        )

    if bool(seed_diagnostics["seed_invalid"]) or selected_coefficients.shape[1] == 0:
        return (
            fixed_basis_params,
            diagnostics_for_rejection("casscf_seed_no_action_novelty"),
            None,
        )
    external_block = ExternalCASBasisBlock(
        model=casscf_model,
        coefficients=selected_coefficients,
        target_mode=casscf_seed_target_mode,
        correction_omegas=np.asarray(residual_omegas, dtype=np.float64),
        correction_eta=float(residual_eta),
        tau_rel=float(casscf_seed_tau_rel),
        tau_abs=float(casscf_seed_tau_abs),
        ratio_clip=float(casscf_seed_ratio_clip),
        finite_difference_step=float(casscf_seed_finite_difference_step),
        method=seed_bank.method,
        basis=seed_bank.basis,
        ncas=int(casscf_model.ncas),
        n_roots=len(casscf_model.ci_vectors),
        tau=float(seed_bank.tau),
    )
    if strong_residual_samples <= 0:
        return (
            fixed_basis_params,
            diagnostics_for_rejection("external_seed_missing_strong_residual_audit"),
            None,
        )
    audit_samples = min(int(strong_residual_samples), int(holdout_points.shape[0]))
    audit_points = np.asarray(holdout_points)[:audit_samples]
    audit_density = np.asarray(holdout_density)[:audit_samples]
    strong_residual_old = strong_residual_audit_from_density(
        fixed_basis_params,
        ground,
        audit_points,
        audit_density,
        head_count=active_heads,
        omegas=strong_residual_omegas,
        eta=residual_eta,
        source_index=strong_residual_source_index,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=strong_residual_batch_size,
    )
    strong_residual_new = strong_residual_audit_from_density(
        fixed_basis_params,
        ground,
        audit_points,
        audit_density,
        head_count=active_heads,
        omegas=strong_residual_omegas,
        eta=residual_eta,
        source_index=strong_residual_source_index,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        external_blocks=(external_block,),
        batch_size=strong_residual_batch_size,
    )
    old_over_eta = float(strong_residual_old["strong_residual_max_epsilon_over_eta"])
    new_over_eta = float(strong_residual_new["strong_residual_max_epsilon_over_eta"])
    old_epsilon = float(strong_residual_old["strong_residual_max_epsilon"])
    new_epsilon = float(strong_residual_new["strong_residual_max_epsilon"])
    ratio = (
        new_over_eta / old_over_eta
        if np.isfinite(old_over_eta) and old_over_eta > 0
        else float("inf")
    )
    nonfinite_count = strong_residual_hard_nonfinite_count(strong_residual_new)
    strong_improved = bool(np.isfinite(ratio) and ratio < 1.0)
    strong_hard_passed = bool(
        np.isfinite(new_over_eta)
        and new_over_eta <= float(max_strong_residual_epsilon_over_eta)
        and nonfinite_count == 0
    )
    strong_provisional_passed = bool(
        strong_improved
        and ratio <= float(max_strong_residual_provisional_ratio)
        and nonfinite_count == 0
    )
    print(
        "response_enrichment_casscf_external_strong_residual "
        f"attempt={attempt:02d} samples={audit_samples} "
        f"old_max_epsilon={old_epsilon:.3e} max_epsilon={new_epsilon:.3e} "
        f"old_max_epsilon_over_eta={old_over_eta:.3e} "
        f"max_epsilon_over_eta={new_over_eta:.3e} "
        f"new_over_old={ratio:.3e} nonfinite_count={nonfinite_count}"
    )
    if not (strong_hard_passed or strong_provisional_passed):
        reason = (
            "external_seed_strong_residual_not_improved"
            if not strong_improved
            else "external_seed_strong_residual_rejected"
        )
        objective_delta = old_over_eta - new_over_eta
        final_capture = max(0.0, old_epsilon**2 - new_epsilon**2)
        reciprocal_ratio = 1.0 / ratio if np.isfinite(ratio) and ratio > 0 else 0.0
        diagnostics = EnrichmentDiagnostics(
            accepted=False,
            active_heads_before=active_heads,
            candidate_heads=external_block.coefficients.shape[1],
            accepted_heads=active_heads,
            attempt=attempt,
            initial_capture=0.0,
            final_capture=final_capture,
            capture_ratio=reciprocal_ratio,
            initial_objective=-old_over_eta,
            final_objective=-new_over_eta,
            objective_delta=objective_delta,
            moment_norm_rel_error=float("inf"),
            moment_first_rel_error=float("inf"),
            overlap_condition=float("inf"),
            holdout_capture_ratio_min=reciprocal_ratio,
            holdout_objective_delta_min=objective_delta,
            holdout_pass_fraction=0.0,
            holdout_pass_count=0,
            holdout_count=1,
            strong_residual_epsilon_max=new_epsilon,
            strong_residual_epsilon_old_max=old_epsilon,
            strong_residual_epsilon_over_eta_max=new_over_eta,
            strong_residual_epsilon_over_eta_old_max=old_over_eta,
            strong_residual_epsilon_over_eta_ratio=ratio,
            strong_residual_nonfinite_count=nonfinite_count,
            strong_residual_hard_passed=strong_hard_passed,
            strong_residual_improved=strong_improved,
            strong_residual_passed=False,
            production_ready=False,
            strong_oracle_passed=True,
            accepted_reason=reason,
        )
        print(
            "response_enrichment_casscf_external_validate "
            f"attempt={attempt:02d} accepted=False "
            f"reason={reason} active_heads={active_heads} "
            f"external_basis={external_block.coefficients.shape[1]} "
            "strong_residual_passed=False "
            f"strong_residual_new_over_old={ratio:.3e} "
            "m0_rel=inf m1_rel=inf overlap_condition=inf"
        )
        return fixed_basis_params, diagnostics, None
    try:
        overlap, hamiltonian, source, *_ = final_weak_matrix_blocks_from_density(
            fixed_basis_params,
            ground,
            [np.asarray(holdout_points)],
            [np.asarray(holdout_density)],
            head_count=active_heads,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            external_blocks=(external_block,),
            batch_size=strong_residual_batch_size,
        )
        spectrum = projected_spectrum(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=overlap_cutoff,
        )
        moments = moment_diagnostics(
            overlap,
            hamiltonian,
            source,
            spectrum,
            overlap_cutoff=overlap_cutoff,
        )
    except (np.linalg.LinAlgError, ValueError) as exc:
        print(
            "response_enrichment_casscf_external_matrix_reject "
            f"attempt={attempt:02d} reason={type(exc).__name__}"
        )
        return (
            fixed_basis_params,
            diagnostics_for_rejection("external_seed_matrix_invalid"),
            None,
        )
    moments_passed = bool(
        np.isfinite(moments.norm_rel_error)
        and np.isfinite(moments.first_moment_rel_error)
        and np.isfinite(moments.overlap_condition)
        and moments.norm_rel_error <= float(max_moment_rel_error)
        and moments.first_moment_rel_error <= float(max_moment_rel_error)
        and moments.overlap_condition <= float(max_overlap_condition)
    )
    accepted = bool(
        (strong_hard_passed or strong_provisional_passed) and moments_passed
    )
    reason = "external_seed"
    if not accepted:
        if not strong_improved:
            reason = "external_seed_strong_residual_not_improved"
        elif not (strong_hard_passed or strong_provisional_passed):
            reason = "external_seed_strong_residual_rejected"
        elif not moments_passed:
            reason = "external_seed_moment_rejected"
    objective_delta = old_over_eta - new_over_eta
    final_capture = max(0.0, old_epsilon**2 - new_epsilon**2)
    reciprocal_ratio = 1.0 / ratio if np.isfinite(ratio) and ratio > 0 else 0.0
    diagnostics = EnrichmentDiagnostics(
        accepted=accepted,
        active_heads_before=active_heads,
        candidate_heads=external_block.coefficients.shape[1],
        accepted_heads=active_heads,
        attempt=attempt,
        initial_capture=0.0,
        final_capture=final_capture,
        capture_ratio=reciprocal_ratio,
        initial_objective=-old_over_eta,
        final_objective=-new_over_eta,
        objective_delta=objective_delta,
        moment_norm_rel_error=moments.norm_rel_error,
        moment_first_rel_error=moments.first_moment_rel_error,
        overlap_condition=moments.overlap_condition,
        holdout_capture_ratio_min=reciprocal_ratio,
        holdout_objective_delta_min=objective_delta,
        holdout_pass_fraction=1.0 if accepted else 0.0,
        holdout_pass_count=1 if accepted else 0,
        holdout_count=1,
        strong_residual_epsilon_max=new_epsilon,
        strong_residual_epsilon_old_max=old_epsilon,
        strong_residual_epsilon_over_eta_max=new_over_eta,
        strong_residual_epsilon_over_eta_old_max=old_over_eta,
        strong_residual_epsilon_over_eta_ratio=ratio,
        strong_residual_nonfinite_count=nonfinite_count,
        strong_residual_hard_passed=strong_hard_passed,
        strong_residual_improved=strong_improved,
        strong_residual_passed=bool(strong_hard_passed or strong_provisional_passed),
        production_ready=bool(strong_hard_passed and moments_passed),
        strong_oracle_passed=True,
        accepted_reason=reason,
    )
    print(
        "response_enrichment_casscf_external_validate "
        f"attempt={attempt:02d} accepted={diagnostics.accepted} "
        f"reason={diagnostics.accepted_reason} active_heads={active_heads} "
        f"external_basis={external_block.coefficients.shape[1]} "
        f"strong_residual_passed={diagnostics.strong_residual_passed} "
        f"strong_residual_new_over_old={ratio:.3e} "
        f"m0_rel={moments.norm_rel_error:.3e} "
        f"m1_rel={moments.first_moment_rel_error:.3e} "
        f"overlap_condition={moments.overlap_condition:.3e}"
    )
    return fixed_basis_params, diagnostics, external_block if accepted else None


def train_residual_enrichment_block(
    params: Params,
    ground: FermiNetGround,
    *,
    key: jax.Array,
    active_heads: int,
    candidate_heads: int,
    candidate_attempts: int,
    attempt_lr_decay: float,
    train_samples: int,
    holdout_samples: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    envelope_decay: float,
    ritz_warmup_epochs: int,
    ritz_warmup_learning_rate_scale: float,
    ritz_warmup_roots: int,
    ritz_accept: bool,
    ritz_accept_root_floor: float,
    ritz_accept_min_weight: float,
    ritz_accept_min_pole_improvement: float,
    validation_holdouts: int,
    holdout_min_pass_fraction: float,
    ritz_validation_max_spread: float,
    ritz_validation_min_pass_fraction: float,
    enrichment_sampling: str,
    enrichment_sampling_walkers: int,
    enrichment_sampling_burn_in: int,
    enrichment_sampling_steps_between: int,
    enrichment_sampling_width: float,
    enrichment_sampling_density_batch_size: int,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float,
    q_node_ground_power: float,
    q_tail_weight: float,
    q_tail_envelope_decay: float,
    q_kinetic_weight: float,
    source_envelope_core_decay: float,
    source_envelope_diffuse_decay: float,
    residual_omegas: np.ndarray,
    residual_omega_weights: np.ndarray,
    residual_source_weights: np.ndarray,
    residual_eta: float,
    casscf_seed: bool,
    casscf_seed_basis: str,
    casscf_seed_ncas: int,
    casscf_seed_n_roots: int,
    casscf_seed_target_mode: str,
    casscf_seed_samples: int,
    casscf_seed_tau_rel: float,
    casscf_seed_tau_abs: float,
    casscf_seed_ratio_clip: float,
    casscf_seed_finite_difference_step: float,
    casscf_seed_state_average: bool,
    enrichment_training_objective: str,
    enrichment_selection_objective: str,
    lambda_rough: float,
    delta: float,
    min_relative_improvement: float,
    min_capture: float,
    min_objective_improvement: float,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    require_training_improvement: bool,
    source_bright_gate: bool,
    source_bright_max_regression: float,
    strong_residual_samples: int,
    strong_residual_omegas: np.ndarray,
    strong_residual_source_index: int,
    strong_residual_batch_size: int,
    max_strong_residual_epsilon_over_eta: float,
    max_strong_residual_provisional_ratio: float,
    strong_polish_epochs: int,
    strong_polish_samples: int,
    strong_polish_batch_size: int,
    strong_polish_learning_rate_scale: float,
    strong_polish_clip: float,
    strong_oracle_samples: int,
    strong_oracle_ridge: float,
    strong_oracle_action_schur: bool,
    strong_oracle_action_schur_ridge: float,
    strong_oracle_max_validation_ratio_winsor99: float,
    strong_oracle_max_validation_ratio_p99: float,
    strong_oracle_max_validation_ratio_pointwise: float,
    strong_oracle_min_validation_relative_epsilon2_improvement: float,
    strong_oracle_min_validation_winsor99_relative_epsilon2_improvement: float,
    strong_oracle_min_candidate_value_norm: float,
    strong_oracle_min_candidate_action_norm: float,
    strong_oracle_max_candidate_action_condition: float,
    region_balanced_cache: bool,
    region_node_quantile: float,
    region_tail_quantile: float,
    region_en_cusp_radius: float,
    region_ee_cusp_radius: float,
    overlap_cutoff: float,
    aux_source_exponents: np.ndarray,
    aux_source_dipole_radial_powers: np.ndarray,
    aux_source_dipole_radial_scale: float,
    aux_source_atom_odd_exponents: np.ndarray,
    aux_source_atom_odd_slater_decays: np.ndarray,
    aux_source_bond_odd_slater_decays: np.ndarray,
    aux_source_dipole_ee_scales: np.ndarray,
    aux_source_bond_odd_ee_slater_decays: np.ndarray,
    aux_source_bond_odd_ee_scales: np.ndarray,
    validation_every: int,
    log_every: int,
) -> tuple[
    Params,
    EnrichmentDiagnostics,
    list[EnrichmentDiagnostics],
    ExternalCASBasisBlock | None,
]:
    """Generate one or more candidates and accept the best validated block.

    Returns:
        Accepted parameters (or the input tree if all attempts fail), selected
        diagnostics, all attempt diagnostics for saving, and an accepted
        external CAS block when the direct-CAS route succeeds.

    Raises:
        RuntimeError: If no diagnostics can be produced for the selected route.
    """
    attempts = max(1, int(candidate_attempts))
    lr_decay = max(0.0, float(attempt_lr_decay))
    diagnostics_history: list[EnrichmentDiagnostics] = []
    best_params = params
    best_diagnostics: EnrichmentDiagnostics | None = None
    best_rejected: EnrichmentDiagnostics | None = None
    best_external: ExternalCASBasisBlock | None = None

    if casscf_seed:
        for attempt in range(attempts):
            candidate_params, diagnostics, external_block = (
                _train_casscf_external_enrichment_attempt(
                    params,
                    ground,
                    key=jax.random.fold_in(key, 10_000 + attempt),
                    active_heads=active_heads,
                    candidate_heads=candidate_heads,
                    attempt=attempt,
                    train_samples=train_samples,
                    holdout_samples=holdout_samples,
                    envelope_decay=envelope_decay,
                    validation_holdouts=validation_holdouts,
                    enrichment_sampling=enrichment_sampling,
                    enrichment_sampling_walkers=enrichment_sampling_walkers,
                    enrichment_sampling_burn_in=enrichment_sampling_burn_in,
                    enrichment_sampling_steps_between=enrichment_sampling_steps_between,
                    enrichment_sampling_width=enrichment_sampling_width,
                    enrichment_sampling_density_batch_size=(
                        enrichment_sampling_density_batch_size
                    ),
                    eps_env=eps_env,
                    ground_weight=ground_weight,
                    source_weight=source_weight,
                    head_weight=head_weight,
                    q_node_weight=q_node_weight,
                    q_node_ground_power=q_node_ground_power,
                    q_tail_weight=q_tail_weight,
                    q_tail_envelope_decay=q_tail_envelope_decay,
                    q_kinetic_weight=q_kinetic_weight,
                    source_envelope_core_decay=source_envelope_core_decay,
                    source_envelope_diffuse_decay=source_envelope_diffuse_decay,
                    residual_omegas=residual_omegas,
                    residual_omega_weights=residual_omega_weights,
                    residual_source_weights=residual_source_weights,
                    residual_eta=residual_eta,
                    casscf_seed_basis=casscf_seed_basis,
                    casscf_seed_ncas=casscf_seed_ncas,
                    casscf_seed_n_roots=casscf_seed_n_roots,
                    casscf_seed_target_mode=casscf_seed_target_mode,
                    casscf_seed_samples=casscf_seed_samples,
                    casscf_seed_tau_rel=casscf_seed_tau_rel,
                    casscf_seed_tau_abs=casscf_seed_tau_abs,
                    casscf_seed_ratio_clip=casscf_seed_ratio_clip,
                    casscf_seed_finite_difference_step=(
                        casscf_seed_finite_difference_step
                    ),
                    casscf_seed_state_average=casscf_seed_state_average,
                    delta=delta,
                    max_moment_rel_error=max_moment_rel_error,
                    max_overlap_condition=max_overlap_condition,
                    strong_residual_samples=strong_residual_samples,
                    strong_residual_omegas=strong_residual_omegas,
                    strong_residual_source_index=strong_residual_source_index,
                    strong_residual_batch_size=strong_residual_batch_size,
                    max_strong_residual_epsilon_over_eta=(
                        max_strong_residual_epsilon_over_eta
                    ),
                    max_strong_residual_provisional_ratio=(
                        max_strong_residual_provisional_ratio
                    ),
                    overlap_cutoff=overlap_cutoff,
                    aux_source_exponents=aux_source_exponents,
                    aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                    aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                    aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                    aux_source_atom_odd_slater_decays=(
                        aux_source_atom_odd_slater_decays
                    ),
                    aux_source_bond_odd_slater_decays=(
                        aux_source_bond_odd_slater_decays
                    ),
                    aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                    aux_source_bond_odd_ee_slater_decays=(
                        aux_source_bond_odd_ee_slater_decays
                    ),
                    aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
                )
            )
            diagnostics_history.append(diagnostics)
            if diagnostics.accepted:
                return (
                    candidate_params,
                    diagnostics,
                    diagnostics_history,
                    external_block,
                )
            if best_rejected is None or diagnostics.objective_delta > (
                best_rejected.objective_delta
            ):
                best_rejected = diagnostics
        if best_rejected is None:
            msg = "CASSCF external enrichment did not produce diagnostics"
            raise RuntimeError(msg)
        return best_params, best_rejected, diagnostics_history, best_external

    for attempt in range(attempts):
        attempt_lr = learning_rate * (lr_decay**attempt if lr_decay else 1.0)
        candidate_params, diagnostics = _train_residual_enrichment_attempt(
            params,
            ground,
            key=jax.random.fold_in(key, 10_000 + attempt),
            active_heads=active_heads,
            candidate_heads=candidate_heads,
            attempt=attempt,
            train_samples=train_samples,
            holdout_samples=holdout_samples,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=attempt_lr,
            envelope_decay=envelope_decay,
            ritz_warmup_epochs=ritz_warmup_epochs,
            ritz_warmup_learning_rate_scale=ritz_warmup_learning_rate_scale,
            ritz_warmup_roots=ritz_warmup_roots,
            ritz_accept=ritz_accept,
            ritz_accept_root_floor=ritz_accept_root_floor,
            ritz_accept_min_weight=ritz_accept_min_weight,
            ritz_accept_min_pole_improvement=ritz_accept_min_pole_improvement,
            validation_holdouts=validation_holdouts,
            holdout_min_pass_fraction=holdout_min_pass_fraction,
            ritz_validation_max_spread=ritz_validation_max_spread,
            ritz_validation_min_pass_fraction=ritz_validation_min_pass_fraction,
            enrichment_sampling=enrichment_sampling,
            enrichment_sampling_walkers=enrichment_sampling_walkers,
            enrichment_sampling_burn_in=enrichment_sampling_burn_in,
            enrichment_sampling_steps_between=enrichment_sampling_steps_between,
            enrichment_sampling_width=enrichment_sampling_width,
            enrichment_sampling_density_batch_size=(
                enrichment_sampling_density_batch_size
            ),
            eps_env=eps_env,
            ground_weight=ground_weight,
            source_weight=source_weight,
            head_weight=head_weight,
            q_node_weight=q_node_weight,
            q_node_ground_power=q_node_ground_power,
            q_tail_weight=q_tail_weight,
            q_tail_envelope_decay=q_tail_envelope_decay,
            q_kinetic_weight=q_kinetic_weight,
            source_envelope_core_decay=source_envelope_core_decay,
            source_envelope_diffuse_decay=source_envelope_diffuse_decay,
            residual_omegas=residual_omegas,
            residual_omega_weights=residual_omega_weights,
            residual_source_weights=residual_source_weights,
            residual_eta=residual_eta,
            enrichment_training_objective=enrichment_training_objective,
            enrichment_selection_objective=enrichment_selection_objective,
            lambda_rough=lambda_rough,
            delta=delta,
            min_relative_improvement=min_relative_improvement,
            min_capture=min_capture,
            min_objective_improvement=min_objective_improvement,
            max_moment_rel_error=max_moment_rel_error,
            max_overlap_condition=max_overlap_condition,
            require_training_improvement=require_training_improvement,
            source_bright_gate=source_bright_gate,
            source_bright_max_regression=source_bright_max_regression,
            strong_residual_samples=strong_residual_samples,
            strong_residual_omegas=strong_residual_omegas,
            strong_residual_source_index=strong_residual_source_index,
            strong_residual_batch_size=strong_residual_batch_size,
            max_strong_residual_epsilon_over_eta=(max_strong_residual_epsilon_over_eta),
            max_strong_residual_provisional_ratio=(
                max_strong_residual_provisional_ratio
            ),
            strong_polish_epochs=strong_polish_epochs,
            strong_polish_samples=strong_polish_samples,
            strong_polish_batch_size=strong_polish_batch_size,
            strong_polish_learning_rate_scale=strong_polish_learning_rate_scale,
            strong_polish_clip=strong_polish_clip,
            strong_oracle_samples=strong_oracle_samples,
            strong_oracle_ridge=strong_oracle_ridge,
            strong_oracle_action_schur=strong_oracle_action_schur,
            strong_oracle_action_schur_ridge=strong_oracle_action_schur_ridge,
            strong_oracle_max_validation_ratio_winsor99=(
                strong_oracle_max_validation_ratio_winsor99
            ),
            strong_oracle_max_validation_ratio_p99=(
                strong_oracle_max_validation_ratio_p99
            ),
            strong_oracle_max_validation_ratio_pointwise=(
                strong_oracle_max_validation_ratio_pointwise
            ),
            strong_oracle_min_validation_relative_epsilon2_improvement=(
                strong_oracle_min_validation_relative_epsilon2_improvement
            ),
            strong_oracle_min_validation_winsor99_relative_epsilon2_improvement=(
                strong_oracle_min_validation_winsor99_relative_epsilon2_improvement
            ),
            strong_oracle_min_candidate_value_norm=(
                strong_oracle_min_candidate_value_norm
            ),
            strong_oracle_min_candidate_action_norm=(
                strong_oracle_min_candidate_action_norm
            ),
            strong_oracle_max_candidate_action_condition=(
                strong_oracle_max_candidate_action_condition
            ),
            region_balanced_cache=region_balanced_cache,
            region_node_quantile=region_node_quantile,
            region_tail_quantile=region_tail_quantile,
            region_en_cusp_radius=region_en_cusp_radius,
            region_ee_cusp_radius=region_ee_cusp_radius,
            overlap_cutoff=overlap_cutoff,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            validation_every=validation_every,
            log_every=log_every,
        )
        diagnostics_history.append(diagnostics)
        if is_better_enrichment_candidate(diagnostics, best_diagnostics):
            best_params = candidate_params
            best_diagnostics = diagnostics
        if not diagnostics.accepted and (
            best_rejected is None
            or diagnostics.final_objective > best_rejected.final_objective
        ):
            best_rejected = diagnostics

    if best_diagnostics is not None:
        print(
            "response_enrichment_select "
            f"accepted=True attempt={best_diagnostics.attempt:02d} "
            f"reason={best_diagnostics.accepted_reason} "
            f"accepted_heads={best_diagnostics.accepted_heads} "
            f"selected_pole={best_diagnostics.selected_pole:.10f} "
            f"final_objective={best_diagnostics.final_objective:.8e}"
        )
        return best_params, best_diagnostics, diagnostics_history, None

    fallback = best_rejected if best_rejected is not None else diagnostics_history[-1]
    print(
        "response_enrichment_select "
        f"accepted=False accepted_heads={fallback.accepted_heads} "
        f"best_rejected_attempt={fallback.attempt:02d} "
        f"final_objective={fallback.final_objective:.8e}"
    )
    return params, fallback, diagnostics_history, None


def ritz_training_loss(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    n_roots: int,
    *,
    head_start: int = 0,
    head_count: int | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    values, gradients = response_values_and_gradients(params, ground, points)
    if head_count is not None:
        head_end = head_start + head_count
        values = values[:, head_start:head_end]
        gradients = gradients[:, head_start:head_end]
    ground_values, ground_gradients = ground_values_and_gradients(ground, points)
    values, gradients, _ = project_values_against_ground(
        values, gradients, ground_values, ground_gradients, density
    )
    overlap, hamiltonian = weak_matrices(
        values, gradients, potential_shift(ground, points), density
    )
    overlap_norm, _, roots = normalized_subspace(overlap, hamiltonian)
    offdiag = overlap_norm - jnp.eye(overlap_norm.shape[0])
    norm_penalty = jnp.mean(jnp.log(jnp.maximum(jnp.diag(overlap), 1e-12)) ** 2)
    overlap_penalty = jnp.mean(offdiag**2)
    energy_loss = jnp.sum(roots[:n_roots])
    loss = energy_loss + 0.05 * overlap_penalty + 0.001 * norm_penalty
    return loss, {"loss": loss, "energy_loss": energy_loss, "root0": roots[0]}


def first_bright_pole(
    poles: np.ndarray,
    weights: np.ndarray,
    *,
    source_norm: float,
    root_floor: float,
    min_weight: float,
) -> float:
    normalized_weights = np.asarray(weights, dtype=np.float64) / max(
        float(source_norm), 1e-12
    )
    poles = np.asarray(poles, dtype=np.float64)
    keep = (poles > root_floor) & (normalized_weights >= min_weight)
    if not np.any(keep):
        return float("inf")
    return float(poles[np.argmax(keep)])


def bright_poles(
    poles: np.ndarray,
    weights: np.ndarray,
    *,
    source_norm: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
) -> np.ndarray:
    """Return the first source-bright poles, padded with NaN.

    The paper diagnostics are spectral, not only first-pole diagnostics.  This
    helper applies the same normalized source-weight gate as
    ``first_bright_pole`` but preserves several visible roots for block,
    leave-one-out, and bootstrap audits of the full projected map.
    """
    root_count = max(0, int(max_roots))
    output = np.full(root_count, np.nan, dtype=np.float64)
    if root_count == 0:
        return output
    normalized_weights = np.asarray(weights, dtype=np.float64) / max(
        float(source_norm), 1e-12
    )
    poles = np.asarray(poles, dtype=np.float64)
    keep = (poles > root_floor) & (normalized_weights >= min_weight)
    selected = poles[keep][:root_count]
    output[: selected.size] = selected
    return output


def heldout_bright_pole_and_moments(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    density: jax.Array,
    *,
    head_count: int,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[float, MomentDiagnostics]:
    """Evaluate the first bright pole for a held-out source-indexed basis.

    Returns:
        First bright pole and source moment diagnostics.
    """
    overlap, hamiltonian, _ = source_plus_head_matrices(
        params,
        ground,
        points,
        density,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
    )
    overlap_np = np.asarray(overlap)
    hamiltonian_np = np.asarray(hamiltonian)
    source_vector = overlap_np[:, 0:1]
    spectrum = projected_spectrum(
        overlap_np,
        hamiltonian_np,
        source_vector,
        overlap_cutoff=overlap_cutoff,
    )
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    source_norm = float(source_vector[0, 0].real)
    pole = first_bright_pole(
        spectrum.excitation_energies,
        weights,
        source_norm=source_norm,
        root_floor=root_floor,
        min_weight=min_weight,
    )
    moments = moment_diagnostics(
        overlap_np,
        hamiltonian_np,
        source_vector,
        spectrum,
        overlap_cutoff=overlap_cutoff,
    )
    return pole, moments


def validate_bright_pole_on_holdouts(
    params: Params,
    ground: FermiNetGround,
    holdouts: list[tuple[jax.Array, jax.Array, float]],
    *,
    head_count: int,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_moment_rel_error: float,
    max_overlap_condition: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> PoleValidationDiagnostics:
    """Validate a candidate pole on independent held-out sample sets.

    Returns:
        Aggregated pole stability and pass/fail diagnostics across holdouts.
    """
    poles: list[float] = []
    moments_list: list[MomentDiagnostics] = []
    for points, density, _ in holdouts:
        try:
            pole, moments = heldout_bright_pole_and_moments(
                params,
                ground,
                points,
                density,
                head_count=head_count,
                overlap_cutoff=overlap_cutoff,
                root_floor=root_floor,
                min_weight=min_weight,
                aux_source_exponents=aux_source_exponents,
                aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
                aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
                aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                aux_source_bond_odd_ee_slater_decays=(
                    aux_source_bond_odd_ee_slater_decays
                ),
                aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            )
        except (np.linalg.LinAlgError, ValueError):
            pole = float("inf")
            moments = failed_moment_diagnostics()
        poles.append(float(pole))
        moments_list.append(moments)
    return summarize_pole_validations(
        poles,
        moments_list,
        max_moment_rel_error=max_moment_rel_error,
        max_overlap_condition=max_overlap_condition,
    )


def log_mixture_q(
    params: Params,
    ground: FermiNetGround,
    points: jax.Array,
    *,
    head_count: int | None,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    envelope_decay: float,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float = 0.0,
    q_node_ground_power: float = 0.5,
    q_tail_weight: float = 0.0,
    q_tail_envelope_decay: float = 0.0,
    q_kinetic_weight: float = 0.0,
) -> jax.Array:
    radius = jnp.linalg.norm(points, axis=2)
    source = source_values(ground, points)
    aux_exponents = jnp.asarray(
        [] if aux_source_exponents is None else aux_source_exponents,
        dtype=points.dtype,
    )
    aux_dipole_radial_powers = jnp.asarray(
        []
        if aux_source_dipole_radial_powers is None
        else aux_source_dipole_radial_powers,
        dtype=points.dtype,
    )
    aux_atom_odd_exponents = jnp.asarray(
        [] if aux_source_atom_odd_exponents is None else aux_source_atom_odd_exponents,
        dtype=points.dtype,
    )
    aux_atom_odd_slater_decays = jnp.asarray(
        []
        if aux_source_atom_odd_slater_decays is None
        else aux_source_atom_odd_slater_decays,
        dtype=points.dtype,
    )
    aux_bond_odd_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_slater_decays is None
        else aux_source_bond_odd_slater_decays,
        dtype=points.dtype,
    )
    aux_dipole_ee_scales = jnp.asarray(
        [] if aux_source_dipole_ee_scales is None else aux_source_dipole_ee_scales,
        dtype=points.dtype,
    )
    aux_bond_odd_ee_slater_decays = jnp.asarray(
        []
        if aux_source_bond_odd_ee_slater_decays is None
        else aux_source_bond_odd_ee_slater_decays,
        dtype=points.dtype,
    )
    aux_bond_odd_ee_scales = jnp.asarray(
        [] if aux_source_bond_odd_ee_scales is None else aux_source_bond_odd_ee_scales,
        dtype=points.dtype,
    )
    aux_sources = auxiliary_source_values(
        ground,
        points,
        aux_exponents,
        aux_atom_odd_exponents,
        aux_atom_odd_slater_decays,
        aux_bond_odd_slater_decays,
        aux_dipole_ee_scales,
        aux_bond_odd_ee_slater_decays,
        aux_bond_odd_ee_scales,
        aux_dipole_radial_powers,
        aux_source_dipole_radial_scale,
    )
    heads = response_values(params, ground, points)
    if head_count is not None:
        heads = heads[:, :head_count]
    log_env_density = jnp.sum(
        3 * jnp.log(envelope_decay) - jnp.log(jnp.pi) - 2 * envelope_decay * radius,
        axis=1,
    )
    log_env = jnp.log(eps_env) + log_env_density
    ground_logpsi = ground_logpsi_batch(ground, points)
    log_ground = jnp.log(ground_weight) + 2 * ground_logpsi
    node_power = float(q_node_ground_power)
    log_node = jnp.log(q_node_weight) + (
        node_power * 2 * ground_logpsi + (1.0 - node_power) * log_env_density
    )
    tail_decay = (
        0.5 * float(envelope_decay)
        if float(q_tail_envelope_decay) <= 0.0
        else float(q_tail_envelope_decay)
    )
    log_tail_density = jnp.sum(
        3 * jnp.log(tail_decay) - jnp.log(jnp.pi) - 2 * tail_decay * radius,
        axis=1,
    )
    log_tail = jnp.log(q_tail_weight) + log_tail_density
    source_norm = source**2
    if aux_sources.shape[1]:
        source_norm = source_norm + jnp.sum(aux_sources**2, axis=1)
    log_source = jnp.log(source_weight) + jnp.log(source_norm + 1e-300)
    head_norm = (
        jnp.sum(heads**2, axis=1) if heads.shape[1] else jnp.zeros(points.shape[0])
    )
    log_heads = jnp.log(head_weight) + jnp.log(head_norm + 1e-300)
    components = [log_env, log_ground, log_source, log_heads, log_node, log_tail]
    if float(q_kinetic_weight) > 0.0:
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
        kinetic_norm = (
            jnp.mean(jnp.sum(flat_gradients**2, axis=2), axis=1)
            if values.shape[1]
            else jnp.zeros(points.shape[0], dtype=points.dtype)
        )
        components.append(jnp.log(q_kinetic_weight) + jnp.log(kinetic_norm + 1e-300))
    return jax.scipy.special.logsumexp(jnp.stack(components, axis=0), axis=0)


def sample_mixture_mcmc(
    params: Params,
    ground: FermiNetGround,
    *,
    key: jax.Array,
    n_samples: int,
    walkers: int,
    burn_in: int,
    steps_between: int,
    width: float,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    envelope_decay: float,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float = 0.0,
    q_node_ground_power: float = 0.5,
    q_tail_weight: float = 0.0,
    q_tail_envelope_decay: float = 0.0,
    q_kinetic_weight: float = 0.0,
) -> tuple[np.ndarray, float]:
    points, _, _ = sample_envelope(key, walkers, envelope_decay, ground.electron_shape)
    logq = log_mixture_q(
        params,
        ground,
        points,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        envelope_decay=envelope_decay,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
    )

    @jax.jit
    def mh_step(points: jax.Array, logq: jax.Array, key: jax.Array):
        kp, ka, kn = jax.random.split(key, 3)
        proposal = points + width * jax.random.normal(kp, points.shape)
        proposal_logq = log_mixture_q(
            params,
            ground,
            proposal,
            head_count=head_count,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            envelope_decay=envelope_decay,
            eps_env=eps_env,
            ground_weight=ground_weight,
            source_weight=source_weight,
            head_weight=head_weight,
            q_node_weight=q_node_weight,
            q_node_ground_power=q_node_ground_power,
            q_tail_weight=q_tail_weight,
            q_tail_envelope_decay=q_tail_envelope_decay,
            q_kinetic_weight=q_kinetic_weight,
        )
        accept = jnp.log(jax.random.uniform(ka, (walkers,))) < proposal_logq - logq
        accept_shape = (walkers,) + (1,) * (points.ndim - 1)
        points = jnp.where(jnp.reshape(accept, accept_shape), proposal, points)
        logq = jnp.where(accept, proposal_logq, logq)
        return points, logq, kn, jnp.mean(accept)

    accept_rates = []
    run_key = jax.random.fold_in(key, 1)
    for _ in range(burn_in):
        points, logq, run_key, pmove = mh_step(points, logq, run_key)
        accept_rates.append(float(pmove))
    collected = []
    while sum(batch.shape[0] for batch in collected) < n_samples:
        for _ in range(steps_between):
            points, logq, run_key, pmove = mh_step(points, logq, run_key)
            accept_rates.append(float(pmove))
        collected.append(np.asarray(points))
    return np.concatenate(collected, axis=0)[:n_samples], float(np.mean(accept_rates))


def mixture_q_density(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    *,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    envelope_decay: float,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    batch_size: int,
    q_node_weight: float = 0.0,
    q_node_ground_power: float = 0.5,
    q_tail_weight: float = 0.0,
    q_tail_envelope_decay: float = 0.0,
    q_kinetic_weight: float = 0.0,
) -> np.ndarray:
    """Evaluate the unnormalized adaptive mixture Q on fixed samples.

    Returns:
        One unnormalized Q value per sample.
    """

    @jax.jit
    def chunk_density(points: jax.Array):
        return jnp.exp(
            log_mixture_q(
                params,
                ground,
                points,
                head_count=head_count,
                aux_source_exponents=aux_source_exponents,
                aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
                aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
                aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
                aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
                aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
                aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
                aux_source_bond_odd_ee_slater_decays=(
                    aux_source_bond_odd_ee_slater_decays
                ),
                aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
                envelope_decay=envelope_decay,
                eps_env=eps_env,
                ground_weight=ground_weight,
                source_weight=source_weight,
                head_weight=head_weight,
                q_node_weight=q_node_weight,
                q_node_ground_power=q_node_ground_power,
                q_tail_weight=q_tail_weight,
                q_tail_envelope_decay=q_tail_envelope_decay,
                q_kinetic_weight=q_kinetic_weight,
            )
        )

    pieces = []
    for chunk in make_batches(points_np.shape[0], batch_size):
        pieces.append(np.asarray(chunk_density(jnp.asarray(points_np[chunk]))))
    return np.concatenate(pieces, axis=0)


def sample_enrichment_distribution(
    params: Params,
    ground: FermiNetGround,
    *,
    key: jax.Array,
    n_samples: int,
    sampling: str,
    envelope_decay: float,
    head_count: int,
    walkers: int,
    burn_in: int,
    steps_between: int,
    width: float,
    density_batch_size: int,
    eps_env: float,
    ground_weight: float,
    source_weight: float,
    head_weight: float,
    q_node_weight: float,
    q_node_ground_power: float,
    q_tail_weight: float,
    q_tail_envelope_decay: float,
    q_kinetic_weight: float,
    source_envelope_core_decay: float,
    source_envelope_diffuse_decay: float,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
) -> tuple[jax.Array, jax.Array, float]:
    """Draw fixed train/holdout samples for residual enrichment.

    ``mixture`` follows the paper's adaptive positive sampling density Q.  The
    returned density is the unnormalized Q used in the importance weights, so
    matrix scales are common and eigenvalues/residual ratios are unchanged.

    Returns:
        Points, unnormalized sampling density values, and the MCMC acceptance
        probability when applicable.

    Raises:
        ValueError: If ``sampling`` is not a supported mode.
    """
    if sampling == "envelope":
        points, density, _ = sample_envelope(
            key, n_samples, envelope_decay, ground.electron_shape
        )
        return points, density, float("nan")
    if sampling in {"source-envelope-sobol", "source-envelope-pz-sobol"}:
        if source_envelope_core_decay <= 0 or source_envelope_diffuse_decay <= 0:
            msg = (
                f"enrichment sampling {sampling!r} requires positive "
                "source-envelope core and diffuse decays"
            )
            raise ValueError(msg)
        seed = int(
            jax.random.randint(
                key,
                (),
                minval=0,
                maxval=np.iinfo(np.int32).max,
                dtype=jnp.int32,
            )
        )
        if sampling == "source-envelope-pz-sobol":
            points_np, density_np, _ = sample_source_envelope_pz_sobol(
                n_samples=n_samples,
                core_decay=source_envelope_core_decay,
                diffuse_decay=source_envelope_diffuse_decay,
                electron_shape=ground.electron_shape,
                seed=seed,
            )
        else:
            points_np, density_np, _ = sample_source_envelope_sobol(
                n_samples=n_samples,
                core_decay=source_envelope_core_decay,
                diffuse_decay=source_envelope_diffuse_decay,
                electron_shape=ground.electron_shape,
                seed=seed,
            )
        return jnp.asarray(points_np), jnp.asarray(density_np), float("nan")
    if sampling != "mixture":
        msg = f"unknown enrichment sampling mode {sampling!r}"
        raise ValueError(msg)

    samples, pmove = sample_mixture_mcmc(
        params,
        ground,
        key=key,
        n_samples=n_samples,
        walkers=walkers,
        burn_in=burn_in,
        steps_between=steps_between,
        width=width,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        envelope_decay=envelope_decay,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
    )
    density = mixture_q_density(
        params,
        ground,
        samples,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        envelope_decay=envelope_decay,
        eps_env=eps_env,
        ground_weight=ground_weight,
        source_weight=source_weight,
        head_weight=head_weight,
        q_node_weight=q_node_weight,
        q_node_ground_power=q_node_ground_power,
        q_tail_weight=q_tail_weight,
        q_tail_envelope_decay=q_tail_envelope_decay,
        q_kinetic_weight=q_kinetic_weight,
        batch_size=density_batch_size,
    )
    return jnp.asarray(samples), jnp.asarray(density), pmove


def final_weak_matrices_from_density(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    *,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate final weak matrices from independent importance samples.

    Returns:
        Overlap matrix, weak Hamiltonian matrix, and projected source vector.
    """
    n_basis = (
        head_count
        + 1
        + _auxiliary_source_count(
            aux_source_exponents,
            aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales,
            aux_source_dipole_radial_powers,
        )
    )
    projection_coeff = final_projection_coefficients_from_density(
        params,
        ground,
        points_np,
        density_np,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
    )
    return final_weak_matrices_from_density_with_projection(
        params,
        ground,
        points_np,
        density_np,
        projection_coeff,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        batch_size=batch_size,
        n_basis=n_basis,
    )


def final_projection_coefficients_from_density(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    *,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None = None,
    batch_size: int,
) -> jax.Array:
    """Estimate final ``Q0`` projection coefficients from fixed samples.

    Returns:
        One global projection coefficient per source/head basis function.
    """
    external_count = external_cas_basis_count(external_blocks)
    n_basis = (
        head_count
        + 1
        + _auxiliary_source_count(
            aux_source_exponents,
            aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales,
            aux_source_dipole_radial_powers,
        )
        + external_count
    )
    n_samples = points_np.shape[0]

    @jax.jit
    def chunk_projection_contrib(points: jax.Array, density: jax.Array):
        values, _, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        ground_values, _ = ground_values_and_gradients(ground, points)
        weights = 1 / density
        ground_norm = jnp.einsum("n,n,n->", weights, ground_values, ground_values)
        numerator = jnp.einsum("n,n,ni->i", weights, ground_values, values)
        return ground_norm, numerator

    def chunk_projection_contrib_with_external(
        points: jax.Array,
        density: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        values, _ = append_external_basis_values_gradients(
            values,
            gradients,
            ground,
            points,
            external_blocks,
        )
        ground_values, _ = ground_values_and_gradients(ground, points)
        weights = 1 / density
        ground_norm = jnp.einsum("n,n,n->", weights, ground_values, ground_values)
        numerator = jnp.einsum("n,n,ni->i", weights, ground_values, values)
        return ground_norm, numerator

    projection_norm = 0.0
    projection_numerator = np.zeros(n_basis, dtype=np.float64)
    for chunk in make_batches(n_samples, batch_size):
        chunk_points = jnp.asarray(points_np[chunk])
        chunk_density = jnp.asarray(density_np[chunk])
        if external_count:
            ground_norm, numerator = chunk_projection_contrib_with_external(
                chunk_points,
                chunk_density,
            )
        else:
            ground_norm, numerator = chunk_projection_contrib(
                chunk_points,
                chunk_density,
            )
        projection_norm += float(ground_norm)
        projection_numerator += np.asarray(numerator)
    return jnp.asarray(projection_numerator / max(projection_norm, 1e-14))


def final_weak_matrices_from_density_with_projection(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    projection_coeff: jax.Array,
    *,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None = None,
    batch_size: int,
    n_basis: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate final weak matrices using fixed ``Q0`` coefficients.

    Returns:
        Overlap matrix, weak Hamiltonian matrix, and projected source vector.
    """
    external_count = external_cas_basis_count(external_blocks)
    n_basis = (
        head_count
        + 1
        + _auxiliary_source_count(
            aux_source_exponents,
            aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales,
            aux_source_dipole_radial_powers,
        )
        + external_count
        if n_basis is None
        else n_basis
    )
    overlap_total = np.zeros((n_basis, n_basis), dtype=np.float64)
    hamiltonian_total = np.zeros((n_basis, n_basis), dtype=np.float64)
    source_total = np.zeros((n_basis, 1), dtype=np.float64)
    n_samples = points_np.shape[0]

    @jax.jit
    def chunk_contrib(points: jax.Array, density: jax.Array, coeff: jax.Array):
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        ground_values, ground_gradients = ground_values_and_gradients(ground, points)
        values, gradients = apply_ground_projection_coefficients(
            values, gradients, ground_values, ground_gradients, coeff
        )
        source = values[:, 0]
        weights = 1 / density
        overlap = jnp.einsum("n,ni,nj->ij", weights, values, values)
        source_vec = jnp.einsum("n,ni,n->i", weights, values, source)
        flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
        kinetic = 0.5 * jnp.einsum("nid,njd->nij", flat_gradients, flat_gradients)
        potential = (
            potential_shift(ground, points)[:, None, None]
            * values[:, :, None]
            * values[:, None, :]
        )
        hamiltonian = jnp.einsum("n,nij->ij", weights, kinetic + potential)
        return overlap, hamiltonian, source_vec[:, None]

    def chunk_contrib_with_external(
        points: jax.Array,
        density: jax.Array,
        coeff: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        values, gradients = append_external_basis_values_gradients(
            values,
            gradients,
            ground,
            points,
            external_blocks,
        )
        ground_values, ground_gradients = ground_values_and_gradients(ground, points)
        values, gradients = apply_ground_projection_coefficients(
            values,
            gradients,
            ground_values,
            ground_gradients,
            coeff,
        )
        source = values[:, 0]
        weights = 1 / density
        overlap = jnp.einsum("n,ni,nj->ij", weights, values, values)
        source_vec = jnp.einsum("n,ni,n->i", weights, values, source)
        flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
        kinetic = 0.5 * jnp.einsum("nid,njd->nij", flat_gradients, flat_gradients)
        potential = (
            potential_shift(ground, points)[:, None, None]
            * values[:, :, None]
            * values[:, None, :]
        )
        hamiltonian = jnp.einsum("n,nij->ij", weights, kinetic + potential)
        return overlap, hamiltonian, source_vec[:, None]

    for chunk in make_batches(n_samples, batch_size):
        chunk_points = jnp.asarray(points_np[chunk])
        chunk_density = jnp.asarray(density_np[chunk])
        if external_count:
            overlap, hamiltonian, source = chunk_contrib_with_external(
                chunk_points,
                chunk_density,
                projection_coeff,
            )
        else:
            overlap, hamiltonian, source = chunk_contrib(
                chunk_points,
                chunk_density,
                projection_coeff,
            )
        overlap_total += np.asarray(overlap)
        hamiltonian_total += np.asarray(hamiltonian)
        source_total += np.asarray(source)
    return (
        overlap_total / n_samples,
        hamiltonian_total / n_samples,
        source_total / n_samples,
    )


def _project_raw_weak_matrices(
    raw_overlap: np.ndarray,
    raw_hamiltonian: np.ndarray,
    projection_numerator: np.ndarray,
    projection_norm: float,
    ground_hamiltonian: np.ndarray,
    ground_hamiltonian_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the ``Q0`` ground projection to unprojected weak matrices.

    Returns:
        Projected overlap, Hamiltonian, and physical source vector.  Keeping
        this algebra explicit lets block/LOO/bootstrap diagnostics recompute
        the ground projection for each resampled estimator instead of
        reusing the all-sample projection.
    """
    norm = max(float(projection_norm), 1e-14)
    numerator = np.asarray(projection_numerator, dtype=np.float64)
    coeff = numerator / norm
    raw_overlap = np.asarray(raw_overlap, dtype=np.float64)
    raw_hamiltonian = np.asarray(raw_hamiltonian, dtype=np.float64)
    ground_hamiltonian = np.asarray(ground_hamiltonian, dtype=np.float64)
    overlap = (
        raw_overlap
        - np.outer(numerator, coeff)
        - np.outer(coeff, numerator)
        + float(projection_norm) * np.outer(coeff, coeff)
    )
    hamiltonian = (
        raw_hamiltonian
        - np.outer(ground_hamiltonian, coeff)
        - np.outer(coeff, ground_hamiltonian)
        + float(ground_hamiltonian_norm) * np.outer(coeff, coeff)
    )
    return overlap, hamiltonian, overlap[:, 0:1]


def _limit_matrix_blocks(
    point_blocks: list[np.ndarray],
    density_blocks: list[np.ndarray],
    sample_limit: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if int(sample_limit) <= 0:
        return [], []
    limited_points = []
    limited_density = []
    remaining = int(sample_limit)
    for points, density in zip(point_blocks, density_blocks, strict=True):
        if remaining <= 0:
            break
        take = min(remaining, int(points.shape[0]))
        limited_points.append(np.asarray(points[:take]))
        limited_density.append(np.asarray(density[:take]))
        remaining -= take
    return limited_points, limited_density


def weak_matrix_blocks_from_precomputed_values(
    ground: FermiNetGround,
    point_blocks: list[np.ndarray],
    density_blocks: list[np.ndarray],
    value_blocks: list[np.ndarray],
    gradient_blocks: list[np.ndarray],
    *,
    batch_size: int,
    return_block_statistics: bool = False,
) -> tuple[np.ndarray, ...]:
    """Estimate projected weak matrices for precomputed basis values.

    ``value_blocks`` and ``gradient_blocks`` must include the physical source as
    column 0 so the returned source vector is directly comparable with the
    neural BF-NKSR matrices.

    Returns:
        Projected overlap, shifted Hamiltonian, and source vector.  If
        ``return_block_statistics`` is true, also returns the projected
        per-block matrices and the raw moments needed by final bootstrap
        diagnostics, matching ``final_weak_matrix_blocks_from_density``.

    Raises:
        ValueError: If block lists are empty or have incompatible shapes.
    """
    if not (
        len(point_blocks)
        == len(density_blocks)
        == len(value_blocks)
        == len(gradient_blocks)
    ):
        msg = "precomputed weak-matrix block lists must have matching lengths"
        raise ValueError(msg)
    if not point_blocks:
        msg = "at least one precomputed matrix block is required"
        raise ValueError(msg)
    raw_block_overlaps = []
    raw_block_hamiltonians = []
    block_projection_numerators = []
    block_projection_norms = []
    block_ground_hamiltonians = []
    block_ground_hamiltonian_norms = []
    block_counts = []
    for points_block, density_block, values_block, gradients_block in zip(
        point_blocks,
        density_blocks,
        value_blocks,
        gradient_blocks,
        strict=True,
    ):
        points = np.asarray(points_block)
        density = np.asarray(density_block, dtype=np.float64)
        values = np.asarray(values_block, dtype=np.float64)
        gradients = np.asarray(gradients_block, dtype=np.float64)
        if values.shape[:1] != points.shape[:1] or gradients.shape[:2] != values.shape:
            msg = "precomputed values/gradients are incompatible with points"
            raise ValueError(msg)
        n_basis = int(values.shape[1])
        raw_overlap = np.zeros((n_basis, n_basis), dtype=np.float64)
        raw_hamiltonian = np.zeros((n_basis, n_basis), dtype=np.float64)
        projection_numerator = np.zeros(n_basis, dtype=np.float64)
        projection_norm = 0.0
        ground_hamiltonian = np.zeros(n_basis, dtype=np.float64)
        ground_hamiltonian_norm = 0.0
        block_samples = int(points.shape[0])
        for chunk in make_batches(block_samples, batch_size):
            chunk_points = jnp.asarray(points[chunk])
            chunk_density = density[chunk]
            chunk_values = values[chunk]
            chunk_gradients = gradients[chunk]
            weights = 1.0 / chunk_density
            raw_overlap += np.einsum(
                "n,ni,nj->ij",
                weights,
                chunk_values,
                chunk_values,
                optimize=True,
            )
            flat_gradients = chunk_gradients.reshape(
                chunk_gradients.shape[0], chunk_gradients.shape[1], -1
            )
            kinetic = 0.5 * np.einsum(
                "nid,njd->nij",
                flat_gradients,
                flat_gradients,
                optimize=True,
            )
            pot_shift = np.asarray(potential_shift(ground, chunk_points))
            potential_matrix = (
                pot_shift[:, None, None]
                * chunk_values[:, :, None]
                * chunk_values[:, None, :]
            )
            raw_hamiltonian += np.einsum(
                "n,nij->ij",
                weights,
                kinetic + potential_matrix,
                optimize=True,
            )
            ground_values, ground_gradients = ground_values_and_gradients(
                ground, chunk_points
            )
            ground_values_np = np.asarray(ground_values)
            ground_gradients_np = np.asarray(ground_gradients).reshape(
                chunk_values.shape[0], -1
            )
            projection_norm += float(
                np.einsum("n,n,n->", weights, ground_values_np, ground_values_np)
            )
            projection_numerator += np.einsum(
                "n,n,ni->i",
                weights,
                ground_values_np,
                chunk_values,
                optimize=True,
            )
            ground_kinetic_basis = 0.5 * np.einsum(
                "nd,nid->ni",
                ground_gradients_np,
                flat_gradients,
                optimize=True,
            )
            ground_potential_basis = (
                pot_shift[:, None] * ground_values_np[:, None] * chunk_values
            )
            ground_hamiltonian += np.einsum(
                "n,ni->i",
                weights,
                ground_kinetic_basis + ground_potential_basis,
                optimize=True,
            )
            ground_kinetic_norm = 0.5 * np.einsum(
                "nd,nd->n",
                ground_gradients_np,
                ground_gradients_np,
                optimize=True,
            )
            ground_potential_norm = pot_shift * ground_values_np * ground_values_np
            ground_hamiltonian_norm += float(
                np.einsum(
                    "n,n->",
                    weights,
                    ground_kinetic_norm + ground_potential_norm,
                    optimize=True,
                )
            )
        raw_block_overlaps.append(raw_overlap / block_samples)
        raw_block_hamiltonians.append(raw_hamiltonian / block_samples)
        block_projection_numerators.append(projection_numerator / block_samples)
        block_projection_norms.append(projection_norm / block_samples)
        block_ground_hamiltonians.append(ground_hamiltonian / block_samples)
        block_ground_hamiltonian_norms.append(ground_hamiltonian_norm / block_samples)
        block_counts.append(block_samples)
    counts = np.asarray(block_counts, dtype=np.float64)
    weights = counts / np.sum(counts)
    raw_overlap = np.tensordot(weights, np.asarray(raw_block_overlaps), axes=(0, 0))
    raw_hamiltonian = np.tensordot(
        weights, np.asarray(raw_block_hamiltonians), axes=(0, 0)
    )
    projection_numerator = np.tensordot(
        weights, np.asarray(block_projection_numerators), axes=(0, 0)
    )
    projection_norm = float(np.dot(weights, np.asarray(block_projection_norms)))
    ground_hamiltonian = np.tensordot(
        weights, np.asarray(block_ground_hamiltonians), axes=(0, 0)
    )
    ground_hamiltonian_norm = float(
        np.dot(weights, np.asarray(block_ground_hamiltonian_norms))
    )
    overlap, hamiltonian, source = _project_raw_weak_matrices(
        raw_overlap,
        raw_hamiltonian,
        projection_numerator,
        projection_norm,
        ground_hamiltonian,
        ground_hamiltonian_norm,
    )
    if not return_block_statistics:
        return overlap, hamiltonian, source

    raw_block_overlaps_arr = np.asarray(raw_block_overlaps)
    raw_block_hamiltonians_arr = np.asarray(raw_block_hamiltonians)
    block_projection_numerators_arr = np.asarray(block_projection_numerators)
    block_projection_norms_arr = np.asarray(block_projection_norms)
    block_ground_hamiltonians_arr = np.asarray(block_ground_hamiltonians)
    block_ground_hamiltonian_norms_arr = np.asarray(block_ground_hamiltonian_norms)
    block_overlaps = []
    block_hamiltonians = []
    block_sources = []
    for (
        raw_overlap_block,
        raw_hamiltonian_block,
        projection_numerator_block,
        projection_norm_block,
        ground_hamiltonian_block,
        ground_hamiltonian_norm_block,
    ) in zip(
        raw_block_overlaps_arr,
        raw_block_hamiltonians_arr,
        block_projection_numerators_arr,
        block_projection_norms_arr,
        block_ground_hamiltonians_arr,
        block_ground_hamiltonian_norms_arr,
        strict=True,
    ):
        block_overlap, block_hamiltonian, block_source = _project_raw_weak_matrices(
            raw_overlap_block,
            raw_hamiltonian_block,
            projection_numerator_block,
            float(projection_norm_block),
            ground_hamiltonian_block,
            float(ground_hamiltonian_norm_block),
        )
        block_overlaps.append(block_overlap)
        block_hamiltonians.append(block_hamiltonian)
        block_sources.append(block_source)
    return (
        overlap,
        hamiltonian,
        source,
        np.asarray(block_overlaps),
        np.asarray(block_hamiltonians),
        np.asarray(block_sources),
        counts,
        raw_block_overlaps_arr,
        raw_block_hamiltonians_arr,
        block_projection_numerators_arr,
        block_projection_norms_arr,
        block_ground_hamiltonians_arr,
        block_ground_hamiltonian_norms_arr,
    )


def retained_weak_matrix_blocks_from_precomputed_values(
    ground: FermiNetGround,
    point_blocks: list[np.ndarray],
    density_blocks: list[np.ndarray],
    value_blocks: list[np.ndarray],
    gradient_blocks: list[np.ndarray],
    *,
    batch_size: int,
    return_block_statistics: bool = False,
) -> tuple[np.ndarray, ...]:
    """Estimate head-only retained matrices with a separate physical source.

    ``value_blocks`` and ``gradient_blocks`` contain the physical source in
    column 0 followed by retained heads.  The returned basis excludes column 0:
    ``S`` and ``K`` are head-head matrices and ``p`` is the projected
    head-source overlap vector.

    Returns:
        Projected retained overlap, Hamiltonian, and source vector.  If
        ``return_block_statistics`` is true, returns projected per-block
        retained matrices plus placeholders for raw projection-resampling
        statistics.

    Raises:
        ValueError: If the precomputed blocks do not contain source plus heads.
    """
    payload = weak_matrix_blocks_from_precomputed_values(
        ground,
        point_blocks,
        density_blocks,
        value_blocks,
        gradient_blocks,
        batch_size=batch_size,
        return_block_statistics=return_block_statistics,
    )
    overlap_all = np.asarray(payload[0], dtype=np.float64)
    hamiltonian_all = np.asarray(payload[1], dtype=np.float64)
    if overlap_all.shape[0] < 2:
        msg = "retained precomputed blocks require source plus at least one head"
        raise ValueError(msg)
    overlap = overlap_all[1:, 1:]
    hamiltonian = hamiltonian_all[1:, 1:]
    source = overlap_all[1:, 0:1]
    if not return_block_statistics:
        return overlap, hamiltonian, source
    block_overlaps_all = np.asarray(payload[3], dtype=np.float64)
    block_hamiltonians_all = np.asarray(payload[4], dtype=np.float64)
    block_overlaps = block_overlaps_all[:, 1:, 1:]
    block_hamiltonians = block_hamiltonians_all[:, 1:, 1:]
    block_sources = block_overlaps_all[:, 1:, 0:1]
    counts = payload[6]
    return (
        overlap,
        hamiltonian,
        source,
        block_overlaps,
        block_hamiltonians,
        block_sources,
        counts,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def cas_dressed_teacher_basis_value_gradient_blocks(
    params: Params,
    ground: FermiNetGround,
    teacher_model: Any,
    point_blocks: list[np.ndarray],
    *,
    head_count: int,
    basis: str,
    finite_difference_step: float,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Evaluate source and CAS-dressed teacher heads on final samples.

    Returns:
        Value and gradient blocks with source in column 0 and dressed teachers
        in the remaining columns.  The source column is used only to assemble
        the projected source vector ``p``.

    Raises:
        ValueError: If dimensions or batch parameters are invalid.
        RuntimeError: If teacher gradients are unavailable.
    """
    head_count = int(head_count)
    if head_count < 1:
        msg = "CAS-dressed teacher basis requires at least one teacher head"
        raise ValueError(msg)
    if int(batch_size) < 1:
        msg = "CAS-dressed teacher basis batch size must be positive"
        raise ValueError(msg)
    radial_scales = jnp.asarray(OFFICIAL_DRESSING_RADIAL_SCALES, dtype=jnp.float64)
    pair_scales = jnp.asarray(OFFICIAL_DRESSING_PAIR_SCALES, dtype=jnp.float64)
    value_blocks: list[np.ndarray] = []
    gradient_blocks: list[np.ndarray] = []
    for points_block in point_blocks:
        points_np = np.asarray(points_block, dtype=np.float64)
        block_values = []
        block_gradients = []
        for chunk in make_batches(points_np.shape[0], batch_size):
            chunk_points_np = points_np[chunk]
            chunk_points = jnp.asarray(chunk_points_np)
            source_values_block, source_gradients_block = source_values_and_gradients(
                ground,
                chunk_points,
            )
            qc_targets = evaluate_casscf_krylov_teacher_targets(
                teacher_model,
                chunk_points_np,
                basis=basis,
                gradients=True,
                finite_difference_step=finite_difference_step,
            )
            if qc_targets.gradients is None:
                msg = "CAS-dressed teacher basis requires teacher gradients"
                raise RuntimeError(msg)
            dressed_values, dressed_gradients = (
                cas_dressed_teacher_values_and_gradients_from_arrays(
                    params,
                    ground,
                    chunk_points,
                    jnp.asarray(qc_targets.values[:, :head_count]),
                    jnp.asarray(qc_targets.gradients[:, :head_count]),
                    radial_scales=radial_scales,
                    pair_scales=pair_scales,
                )
            )
            block_values.append(
                np.concatenate(
                    [
                        np.asarray(source_values_block)[:, None],
                        np.asarray(dressed_values),
                    ],
                    axis=1,
                )
            )
            block_gradients.append(
                np.concatenate(
                    [
                        np.asarray(source_gradients_block)[:, None, :, :],
                        np.asarray(dressed_gradients),
                    ],
                    axis=1,
                )
            )
        value_blocks.append(np.concatenate(block_values, axis=0))
        gradient_blocks.append(np.concatenate(block_gradients, axis=0))
    return value_blocks, gradient_blocks


def krylov_teacher_model_metadata(
    teacher_model: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return energy/source metadata without evaluating teacher values."""
    coefficients = np.asarray(teacher_model.coefficients, dtype=np.float64)
    excitation_energies = np.asarray(
        teacher_model.excitation_energies,
        dtype=np.float64,
    )
    root_source_overlaps = np.asarray(
        teacher_model.root_source_overlaps,
        dtype=np.float64,
    )
    teacher_energies = np.einsum(
        "ia,i,ia->a",
        coefficients,
        excitation_energies,
        coefficients,
        optimize=True,
    )
    teacher_source_overlaps = coefficients.T @ root_source_overlaps
    return (
        teacher_energies,
        teacher_source_overlaps,
        excitation_energies,
        root_source_overlaps,
        np.asarray(teacher_model.singular_values, dtype=np.float64),
        coefficients,
    )


def retained_krylov_source_moments(
    teacher_model: Any,
    *,
    max_order: int = 2,
) -> np.ndarray:
    """Return source moments inside the retained Krylov teacher subspace."""
    coefficients = np.asarray(teacher_model.coefficients, dtype=np.float64)
    excitation_energies = np.asarray(
        teacher_model.excitation_energies,
        dtype=np.float64,
    )
    root_source = np.asarray(teacher_model.root_source_overlaps, dtype=np.float64)
    if root_source.ndim == 1:
        root_source = root_source[:, None]
    if coefficients.size == 0 or root_source.size == 0:
        return np.asarray([], dtype=np.float64)
    retained_hamiltonian = coefficients.T @ (
        excitation_energies[:, None] * coefficients
    )
    retained_source = coefficients.T @ root_source
    moments = []
    h_power = np.eye(retained_hamiltonian.shape[0], dtype=np.float64)
    for order in range(int(max_order) + 1):
        if order:
            h_power = h_power @ retained_hamiltonian
        moments.append(retained_source.T @ h_power @ retained_source)
    return np.asarray(moments, dtype=np.float64)


def _source_weighted_projected_objective(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
) -> dict[str, float]:
    """Compute the source-weighted retained-Ritz objective used for selection.

    Returns:
        Objective, first bright pole, source weight, retained dimension, and
        overlap condition diagnostics.
    """
    try:
        spectrum = projected_spectrum(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=float(overlap_cutoff),
        )
    except (np.linalg.LinAlgError, ValueError):
        return {
            "objective": float("inf"),
            "bright_root0": float("inf"),
            "bright_weight0": float("nan"),
            "source_weight_sum": float("nan"),
            "retained": 0.0,
            "condition": float("inf"),
        }
    roots = np.asarray(spectrum.excitation_energies, dtype=np.float64)
    if roots.size == 0:
        return {
            "objective": float("inf"),
            "bright_root0": float("inf"),
            "bright_weight0": float("nan"),
            "source_weight_sum": 0.0,
            "retained": 0.0,
            "condition": float("inf"),
        }
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    finite = np.isfinite(roots) & (roots >= float(root_floor))
    selected = np.flatnonzero(finite)[: max(1, int(max_roots))]
    selected_weight_sum = float(np.sum(weights[selected])) if selected.size else 0.0
    if selected.size and selected_weight_sum > 0.0:
        objective = float(
            np.dot(weights[selected], roots[selected]) / selected_weight_sum
        )
    else:
        objective = float("inf")
    source_norm = float(np.sum(weights))
    normalized_weights = weights / max(source_norm, 1e-30)
    bright = np.flatnonzero(finite & (normalized_weights >= float(min_weight)))
    bright_root0 = float(roots[bright[0]]) if bright.size else float("inf")
    bright_weight0 = (
        float(normalized_weights[bright[0]]) if bright.size else float("nan")
    )
    retained_evals = np.asarray(spectrum.retained_overlap_eigenvalues, dtype=np.float64)
    if retained_evals.size:
        condition = float(np.max(retained_evals) / max(np.min(retained_evals), 1e-300))
    else:
        condition = float("inf")
    return {
        "objective": objective,
        "bright_root0": bright_root0,
        "bright_weight0": bright_weight0,
        "source_weight_sum": selected_weight_sum,
        "retained": float(retained_evals.size),
        "condition": condition,
    }


def _project_cross_overlap(
    raw_cross: np.ndarray,
    left_projection: np.ndarray,
    right_projection: np.ndarray,
    projection_norm: float,
) -> np.ndarray:
    norm = max(float(projection_norm), 1e-14)
    left_coeff = np.asarray(left_projection, dtype=np.float64) / norm
    right_coeff = np.asarray(right_projection, dtype=np.float64) / norm
    return (
        np.asarray(raw_cross, dtype=np.float64)
        - np.outer(left_projection, right_coeff)
        - np.outer(left_coeff, right_projection)
        + norm * np.outer(left_coeff, right_coeff)
    )


def _overlap_whitener(
    overlap: np.ndarray,
    *,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    overlap_sym = (np.asarray(overlap, dtype=np.float64) + np.asarray(overlap).T) / 2
    evals, evecs = np.linalg.eigh(overlap_sym)
    if evals.size == 0:
        return np.empty((overlap_sym.shape[0], 0), dtype=np.float64), evals
    threshold = max(float(cutoff) * max(float(np.max(evals)), 1.0), 1e-14)
    keep = evals > threshold
    if not np.any(keep):
        return np.empty((overlap_sym.shape[0], 0), dtype=np.float64), evals
    return evecs[:, keep] / np.sqrt(evals[keep])[None, :], evals


def retained_head_validation_from_blocks(  # noqa: C901
    ground: FermiNetGround,
    point_blocks: list[np.ndarray],
    density_blocks: list[np.ndarray],
    reference_value_blocks: list[np.ndarray],
    trial_value_blocks: list[np.ndarray],
    *,
    batch_size: int,
    overlap_cutoff: float,
) -> dict[str, float]:
    """Validate final retained heads against the lambda=0 teacher subspace.

    Returns:
        Value trust ratio, subspace fidelity, overlap conditions, and retained
        validation rank.

    Raises:
        ValueError: If block lists or head dimensions are inconsistent.
    """
    if not (
        len(point_blocks)
        == len(density_blocks)
        == len(reference_value_blocks)
        == len(trial_value_blocks)
    ):
        msg = "retained-head validation block lists must have matching lengths"
        raise ValueError(msg)
    if not point_blocks:
        return {
            "value_trust_ratio": float("nan"),
            "subspace_fidelity": float("nan"),
            "reference_condition": float("inf"),
            "trial_condition": float("inf"),
            "retained_rank": 0.0,
        }
    head_count = int(np.asarray(reference_value_blocks[0]).shape[1] - 1)
    if head_count < 1:
        return {
            "value_trust_ratio": float("nan"),
            "subspace_fidelity": float("nan"),
            "reference_condition": float("inf"),
            "trial_condition": float("inf"),
            "retained_rank": 0.0,
        }
    n_total = int(sum(np.asarray(block).shape[0] for block in point_blocks))
    raw_reference = np.zeros((head_count, head_count), dtype=np.float64)
    raw_trial = np.zeros_like(raw_reference)
    raw_cross = np.zeros_like(raw_reference)
    reference_projection = np.zeros(head_count, dtype=np.float64)
    trial_projection = np.zeros(head_count, dtype=np.float64)
    projection_norm = 0.0
    for points_block, density_block, reference_block, trial_block in zip(
        point_blocks,
        density_blocks,
        reference_value_blocks,
        trial_value_blocks,
        strict=True,
    ):
        points = np.asarray(points_block)
        density = np.asarray(density_block, dtype=np.float64)
        reference_values = np.asarray(reference_block, dtype=np.float64)[:, 1:]
        trial_values = np.asarray(trial_block, dtype=np.float64)[:, 1:]
        if reference_values.shape != trial_values.shape:
            msg = "retained-head validation reference/trial shapes must match"
            raise ValueError(msg)
        if reference_values.shape[1] != head_count:
            msg = "retained-head validation head count changed across blocks"
            raise ValueError(msg)
        for chunk in make_batches(points.shape[0], int(batch_size)):
            chunk_points = jnp.asarray(points[chunk])
            weights = 1.0 / density[chunk] / max(n_total, 1)
            ground_values = np.asarray(
                jax.vmap(ground_value_single, (None, 0))(ground, chunk_points),
                dtype=np.float64,
            )
            ref = reference_values[chunk]
            trial = trial_values[chunk]
            raw_reference += np.einsum("n,ni,nj->ij", weights, ref, ref)
            raw_trial += np.einsum("n,ni,nj->ij", weights, trial, trial)
            raw_cross += np.einsum("n,ni,nj->ij", weights, ref, trial)
            reference_projection += np.einsum("n,n,ni->i", weights, ground_values, ref)
            trial_projection += np.einsum("n,n,ni->i", weights, ground_values, trial)
            projection_norm += float(
                np.einsum("n,n,n->", weights, ground_values, ground_values)
            )
    reference_overlap = _project_cross_overlap(
        raw_reference,
        reference_projection,
        reference_projection,
        projection_norm,
    )
    trial_overlap = _project_cross_overlap(
        raw_trial,
        trial_projection,
        trial_projection,
        projection_norm,
    )
    cross_overlap = _project_cross_overlap(
        raw_cross,
        reference_projection,
        trial_projection,
        projection_norm,
    )
    reference_trace = max(float(np.trace(reference_overlap)), 1e-30)
    correction_trace = float(
        np.trace(reference_overlap)
        + np.trace(trial_overlap)
        - 2.0 * np.trace(cross_overlap)
    )
    value_trust_ratio = max(correction_trace, 0.0) / reference_trace
    reference_whitener, reference_evals = _overlap_whitener(
        reference_overlap,
        cutoff=overlap_cutoff,
    )
    trial_whitener, trial_evals = _overlap_whitener(
        trial_overlap,
        cutoff=overlap_cutoff,
    )
    rank = min(reference_whitener.shape[1], trial_whitener.shape[1])
    if rank:
        overlap_map = reference_whitener.T @ cross_overlap @ trial_whitener
        subspace_fidelity = float(np.sum(overlap_map**2) / rank)
        subspace_fidelity = float(np.clip(subspace_fidelity, 0.0, 1.0))
    else:
        subspace_fidelity = float("nan")

    def condition_from_evals(evals: np.ndarray) -> float:
        positive = np.asarray(evals, dtype=np.float64)
        positive = positive[positive > 0.0]
        if positive.size == 0:
            return float("inf")
        return float(np.max(positive) / max(np.min(positive), 1e-300))

    return {
        "value_trust_ratio": value_trust_ratio,
        "subspace_fidelity": subspace_fidelity,
        "reference_condition": condition_from_evals(reference_evals),
        "trial_condition": condition_from_evals(trial_evals),
        "retained_rank": float(rank),
    }


def _matrix_summary_for_cutoffs(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    cutoffs: np.ndarray,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> dict[str, np.ndarray]:
    return cutoff_sensitivity_diagnostics(
        overlap,
        hamiltonian,
        source,
        cutoffs=np.asarray(cutoffs, dtype=np.float64),
        root_floor=0.0,
        min_weight=float(min_weight),
        max_roots=int(max_roots),
        source_in_basis=source_in_basis,
    )


def _print_matrix_debug_summary(
    label: str,
    diagnostics: dict[str, np.ndarray],
) -> None:
    cutoffs = np.asarray(diagnostics["cutoffs"], dtype=np.float64)
    roots = np.asarray(diagnostics["bright_roots"], dtype=np.float64)
    retained = np.asarray(diagnostics["retained"], dtype=np.int64)
    condition = np.asarray(diagnostics["condition"], dtype=np.float64)
    root0 = roots[:, 0] if roots.ndim == 2 and roots.shape[1] else np.asarray([])
    root_text = ",".join(f"{float(value):.10f}" for value in root0)
    cutoff_text = ",".join(f"{float(value):.1e}" for value in cutoffs)
    retained_text = ",".join(str(int(value)) for value in retained)
    condition_text = ",".join(f"{float(value):.3e}" for value in condition)
    print(
        "matrix_subspace_debug "
        f"label={label} "
        f"cutoffs={cutoff_text} "
        f"root0_ha={root_text} "
        f"retained={retained_text} "
        f"condition={condition_text}"
    )


def strong_residual_density_diagnostics(
    density_np: np.ndarray,
) -> dict[str, float | int]:
    """Summarize proposal-density health for strong residual audits.

    Returns:
        Counts of nonfinite/nonpositive entries and finite positive min/max.
    """
    density_finite = np.isfinite(density_np)
    density_positive = density_finite & (density_np > 0)
    density_positive_values = density_np[density_positive]
    return {
        "nonfinite_count": int(density_np.size - np.count_nonzero(density_finite)),
        "nonpositive_count": int(density_np.size - np.count_nonzero(density_positive)),
        "min_positive": (
            float(np.min(density_positive_values))
            if density_positive_values.size
            else float("nan")
        ),
        "max_finite": (
            float(np.max(density_np[density_finite]))
            if np.any(density_finite)
            else float("nan")
        ),
    }


def linear_system_condition_numbers(systems: np.ndarray) -> np.ndarray:
    """Compute condition numbers without failing on invalid systems.

    Returns:
        One condition number per leading system slice, with inf on failure.
    """
    conditions = np.full(systems.shape[0], np.inf, dtype=np.float64)
    for idx, system in enumerate(systems):
        if np.all(np.isfinite(system)):
            try:
                conditions[idx] = float(np.linalg.cond(system))
            except np.linalg.LinAlgError:
                conditions[idx] = float("inf")
    return conditions


def strong_residual_audit_from_density(
    params: Params,
    ground: FermiNetGround,
    points_np: np.ndarray,
    density_np: np.ndarray,
    *,
    head_count: int,
    omegas: np.ndarray,
    eta: float,
    source_index: int = 0,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None = None,
    batch_size: int,
) -> dict[str, np.ndarray | float | int]:
    """Evaluate the paper's pointwise strong residual diagnostic.

    The correction-vector coefficients are obtained from the weak-form
    projected matrices.  Only this audit evaluates Laplacians:
    ``r_a(R; z) = Phi_a(R) - [z - (H - E0)] X_a^B(R; z)``.

    Returns:
        Arrays ready to save in the output NPZ, including
        ``strong_residual_epsilon = ||r_a(z)|| / ||Phi_a||``.

    Raises:
        ValueError: If no samples/frequencies are supplied or source_index is
        outside the source block.
    """
    points_np = np.asarray(points_np)
    density_np = np.asarray(density_np, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    if points_np.shape[0] == 0:
        msg = "strong residual audit requires at least one sample"
        raise ValueError(msg)
    if density_np.shape[0] != points_np.shape[0]:
        msg = "strong residual audit points and density must have same length"
        raise ValueError(msg)
    if omegas.size == 0:
        msg = "strong residual audit requires at least one omega"
        raise ValueError(msg)
    source_count = 1 + _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    if source_index < 0 or source_index >= source_count:
        msg = (
            "strong residual source_index must select a source channel, "
            f"got {source_index} for {source_count} sources"
        )
        raise ValueError(msg)

    density_diagnostics = strong_residual_density_diagnostics(density_np)

    projection_coeff = final_projection_coefficients_from_density(
        params,
        ground,
        points_np,
        density_np,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        external_blocks=external_blocks,
        batch_size=batch_size,
    )
    overlap, hamiltonian, _ = final_weak_matrices_from_density_with_projection(
        params,
        ground,
        points_np,
        density_np,
        projection_coeff,
        head_count=head_count,
        aux_source_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
        aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        external_blocks=external_blocks,
        batch_size=batch_size,
    )
    z_values = omegas + 1j * float(eta)
    rhs = np.asarray(overlap[:, source_index], dtype=np.complex128)
    systems = (
        z_values[:, None, None] * np.asarray(overlap)[None, :, :]
        - np.asarray(hamiltonian)[None, :, :]
    )
    system_conditions = linear_system_condition_numbers(systems)
    coeffs = np.linalg.solve(systems, rhs[None, :, None])[:, :, 0]
    coeffs_jax = jnp.asarray(coeffs)
    z_jax = jnp.asarray(z_values)

    def chunk_residual_contrib(
        points: jax.Array,
        density: jax.Array,
        project_coeff: jax.Array,
        response_coeffs: jax.Array,
        z: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        values, hbar_values = source_aux_and_head_values_and_hbar(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        values, hbar_values = append_external_basis_values_hbar(
            values,
            hbar_values,
            ground,
            points,
            external_blocks,
        )
        ground_values = jax.vmap(ground_value_single, (None, 0))(ground, points)
        ground_hbar = ground_hbar_values(ground, points, ground_values)
        values = values - ground_values[:, None] * project_coeff[None, :]
        hbar_values = hbar_values - ground_hbar[:, None] * project_coeff[None, :]
        x_values = jnp.einsum("ni,li->nl", values, response_coeffs)
        hbar_x_values = jnp.einsum("ni,li->nl", hbar_values, response_coeffs)
        phi = values[:, source_index]
        residual = phi[:, None] - z[None, :] * x_values + hbar_x_values
        weights = 1 / density
        residual_sq = weights[:, None] * jnp.abs(residual) ** 2
        source_sq = weights * jnp.abs(phi) ** 2
        charge_center = jnp.sum(ground.atoms * ground.charges[:, None], axis=0)
        charge_center = charge_center / jnp.sum(ground.charges)
        shifted = points - charge_center[None, None, :]
        tail_radius = jnp.sqrt(jnp.mean(jnp.sum(shifted**2, axis=-1), axis=1))
        ae_vec = points[:, :, None, :] - ground.atoms[None, None, :, :]
        min_en = jnp.min(jnp.sqrt(jnp.sum(ae_vec**2, axis=-1) + 1e-24), axis=(1, 2))
        nelec = points.shape[1]
        if nelec < 2:
            min_ee = jnp.full((points.shape[0],), jnp.inf, dtype=points.dtype)
        else:
            ee_vec = points[:, :, None, :] - points[:, None, :, :]
            ee_dist = jnp.sqrt(jnp.sum(ee_vec**2, axis=-1) + 1e-24)
            ee_dist = ee_dist + jnp.eye(nelec, dtype=points.dtype)[None, :, :] * 1e6
            min_ee = jnp.min(ee_dist, axis=(1, 2))
        high_action = jnp.max(jnp.abs(hbar_x_values), axis=1)
        return (
            residual_sq,
            source_sq,
            jnp.abs(ground_values),
            min_en,
            min_ee,
            tail_radius,
            high_action,
            jnp.sum(~jnp.isfinite(values)),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.sum(~jnp.isfinite(hbar_values)),
            jnp.sum(~jnp.isfinite(residual)),
            jnp.sum(~jnp.isfinite(residual_sq)),
            jnp.sum(~jnp.isfinite(source_sq)),
        )

    residual_sq_total = np.zeros(omegas.shape, dtype=np.float64)
    source_sq_total = 0.0
    residual_sq_samples: list[np.ndarray] = []
    source_sq_samples: list[np.ndarray] = []
    ground_abs_samples: list[np.ndarray] = []
    min_en_samples: list[np.ndarray] = []
    min_ee_samples: list[np.ndarray] = []
    tail_radius_samples: list[np.ndarray] = []
    high_action_samples: list[np.ndarray] = []
    value_nonfinite_count = 0
    laplacian_nonfinite_count = 0
    hbar_nonfinite_count = 0
    residual_nonfinite_count = 0
    residual_contrib_nonfinite_count = 0
    source_contrib_nonfinite_count = 0
    n_samples = points_np.shape[0]
    for chunk in make_batches(n_samples, batch_size):
        (
            residual_sq,
            source_sq,
            ground_abs,
            min_en,
            min_ee,
            tail_radius,
            high_action,
            value_nonfinite,
            laplacian_nonfinite,
            hbar_nonfinite,
            residual_nonfinite,
            residual_sq_nonfinite,
            source_sq_nonfinite,
        ) = chunk_residual_contrib(
            jnp.asarray(points_np[chunk]),
            jnp.asarray(density_np[chunk]),
            projection_coeff,
            coeffs_jax,
            z_jax,
        )
        residual_sq_np = np.asarray(residual_sq, dtype=np.float64)
        source_sq_np = np.asarray(source_sq, dtype=np.float64)
        value_nonfinite_count += int(value_nonfinite)
        laplacian_nonfinite_count += int(laplacian_nonfinite)
        hbar_nonfinite_count += int(hbar_nonfinite)
        residual_nonfinite_count += int(residual_nonfinite)
        residual_contrib_nonfinite_count += int(residual_sq_nonfinite)
        source_contrib_nonfinite_count += int(source_sq_nonfinite)
        residual_sq_np = np.where(np.isfinite(residual_sq_np), residual_sq_np, np.inf)
        source_sq_np = np.where(np.isfinite(source_sq_np), source_sq_np, np.inf)
        residual_sq_total += np.sum(residual_sq_np, axis=0)
        source_sq_total += float(np.sum(source_sq_np))
        residual_sq_samples.append(residual_sq_np)
        source_sq_samples.append(source_sq_np)
        ground_abs_samples.append(np.asarray(ground_abs, dtype=np.float64))
        min_en_samples.append(np.asarray(min_en, dtype=np.float64))
        min_ee_samples.append(np.asarray(min_ee, dtype=np.float64))
        tail_radius_samples.append(np.asarray(tail_radius, dtype=np.float64))
        high_action_samples.append(np.asarray(high_action, dtype=np.float64))

    residual_norms = np.sqrt(np.maximum(residual_sq_total / n_samples, 0.0))
    source_sq_mean = source_sq_total / n_samples
    source_norm = float(np.sqrt(max(source_sq_mean, 0.0)))
    source_invalid = (
        source_contrib_nonfinite_count > 0
        or not np.isfinite(source_norm)
        or source_norm <= 0
    )
    source_denominator = max(source_norm, 1e-14) if not source_invalid else 1e-14
    epsilon = residual_norms / source_denominator
    epsilon = np.where(np.isfinite(epsilon), epsilon, np.inf)
    if source_invalid:
        epsilon = np.full_like(epsilon, np.inf)
    epsilon_over_eta = epsilon / max(float(eta), 1e-14)
    residual_by_sample = np.concatenate(residual_sq_samples, axis=0)
    source_by_sample = np.concatenate(source_sq_samples, axis=0)
    ground_abs_all = np.concatenate(ground_abs_samples, axis=0)
    min_en_all = np.concatenate(min_en_samples, axis=0)
    min_ee_all = np.concatenate(min_ee_samples, axis=0)
    tail_radius_all = np.concatenate(tail_radius_samples, axis=0)
    high_action_all = np.concatenate(high_action_samples, axis=0)

    def finite_quantile(values: np.ndarray, q: float, default: float) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(np.quantile(finite, q)) if finite.size else float(default)

    node_threshold = finite_quantile(ground_abs_all, 0.10, 0.0)
    en_threshold = 0.25
    ee_threshold = 0.25
    tail_threshold = finite_quantile(tail_radius_all, 0.90, float("inf"))
    action_threshold = finite_quantile(high_action_all, 0.90, float("inf"))
    node_mask = ground_abs_all <= node_threshold
    en_cusp_mask = min_en_all <= en_threshold
    ee_cusp_mask = min_ee_all <= ee_threshold
    tail_mask = tail_radius_all >= tail_threshold
    high_action_mask = high_action_all >= action_threshold
    en_shell_width = 0.10 * en_threshold
    en_shell_distance = np.abs(min_en_all - en_threshold)
    en_cutoff_shell_mask = en_shell_distance <= en_shell_width
    high_action_en_cusp_mask = high_action_mask & en_cusp_mask
    high_action_ee_cusp_mask = high_action_mask & ee_cusp_mask
    high_action_tail_mask = high_action_mask & tail_mask
    high_action_shell_mask = high_action_mask & en_cutoff_shell_mask
    high_action_bulk_mask = high_action_mask & ~(
        high_action_en_cusp_mask
        | high_action_ee_cusp_mask
        | high_action_tail_mask
        | high_action_shell_mask
    )
    union_mask = node_mask | en_cusp_mask | ee_cusp_mask | tail_mask | high_action_mask
    bulk_mask = ~union_mask
    region_labels = np.asarray(
        [
            "all",
            "node_tube",
            "en_cusp",
            "ee_cusp",
            "tail",
            "high_action",
            "high_action_en_cusp",
            "high_action_ee_cusp",
            "high_action_tail",
            "high_action_shell",
            "high_action_bulk",
            "bulk",
        ]
    )
    region_masks = [
        np.ones(n_samples, dtype=bool),
        node_mask,
        en_cusp_mask,
        ee_cusp_mask,
        tail_mask,
        high_action_mask,
        high_action_en_cusp_mask,
        high_action_ee_cusp_mask,
        high_action_tail_mask,
        high_action_shell_mask,
        high_action_bulk_mask,
        bulk_mask,
    ]
    region_counts = np.asarray(
        [int(np.count_nonzero(mask)) for mask in region_masks], dtype=np.int64
    )
    region_residual_sq = np.asarray(
        [np.sum(residual_by_sample[mask], axis=0) for mask in region_masks],
        dtype=np.float64,
    )
    region_source_sq = np.asarray(
        [np.sum(source_by_sample[mask]) for mask in region_masks],
        dtype=np.float64,
    )
    region_residual_norms = np.sqrt(np.maximum(region_residual_sq / n_samples, 0.0))
    region_source_norms = np.sqrt(np.maximum(region_source_sq / n_samples, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        region_epsilon = region_residual_norms / np.maximum(
            region_source_norms[:, None], 1e-14
        )
    region_epsilon = np.where(np.isfinite(region_epsilon), region_epsilon, np.inf)
    region_epsilon_over_eta = region_epsilon / max(float(eta), 1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):
        region_residual_fraction = region_residual_sq / np.maximum(
            residual_sq_total[None, :], 1e-300
        )
    region_residual_fraction = np.where(
        np.isfinite(region_residual_fraction),
        region_residual_fraction,
        np.inf,
    )
    return {
        "strong_residual_omegas": omegas,
        "strong_residual_epsilon": epsilon,
        "strong_residual_epsilon_over_eta": epsilon_over_eta,
        "strong_residual_norm": residual_norms,
        "strong_residual_source_norm": float(source_norm),
        "strong_residual_eta": float(eta),
        "strong_residual_source_index": int(source_index),
        "strong_residual_samples": int(n_samples),
        "strong_residual_max_epsilon": float(np.max(epsilon)),
        "strong_residual_max_epsilon_over_eta": float(np.max(epsilon_over_eta)),
        "strong_residual_density_nonfinite_count": int(
            density_diagnostics["nonfinite_count"]
        ),
        "strong_residual_density_nonpositive_count": int(
            density_diagnostics["nonpositive_count"]
        ),
        "strong_residual_density_min_positive": float(
            density_diagnostics["min_positive"]
        ),
        "strong_residual_density_max_finite": float(density_diagnostics["max_finite"]),
        "strong_residual_overlap_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(overlap))
        ),
        "strong_residual_hamiltonian_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(hamiltonian))
        ),
        "strong_residual_coeff_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(coeffs))
        ),
        "strong_residual_system_condition": system_conditions,
        "strong_residual_system_condition_max": float(np.max(system_conditions)),
        "strong_residual_value_nonfinite_count": int(value_nonfinite_count),
        "strong_residual_laplacian_nonfinite_count": int(laplacian_nonfinite_count),
        "strong_residual_hbar_nonfinite_count": int(hbar_nonfinite_count),
        "strong_residual_point_residual_nonfinite_count": int(residual_nonfinite_count),
        "strong_residual_residual_contrib_nonfinite_count": int(
            residual_contrib_nonfinite_count
        ),
        "strong_residual_source_contrib_nonfinite_count": int(
            source_contrib_nonfinite_count
        ),
        "strong_residual_region_labels": region_labels,
        "strong_residual_region_counts": region_counts,
        "strong_residual_region_source_norm": region_source_norms,
        "strong_residual_region_residual_norm": region_residual_norms,
        "strong_residual_region_epsilon": region_epsilon,
        "strong_residual_region_epsilon_over_eta": region_epsilon_over_eta,
        "strong_residual_region_residual_fraction": region_residual_fraction,
        "strong_residual_region_node_abs_threshold": node_threshold,
        "strong_residual_region_en_cusp_threshold": en_threshold,
        "strong_residual_region_ee_cusp_threshold": ee_threshold,
        "strong_residual_region_tail_radius_threshold": tail_threshold,
        "strong_residual_region_high_action_threshold": action_threshold,
        "strong_residual_region_en_cutoff_shell_width": en_shell_width,
        "strong_residual_region_en_cutoff_shell_min_distance": (
            finite_quantile(en_shell_distance, 0.0, float("inf"))
        ),
    }


def final_weak_matrix_blocks_from_density(
    params: Params,
    ground: FermiNetGround,
    point_blocks: list[np.ndarray],
    density_blocks: list[np.ndarray],
    *,
    head_count: int,
    aux_source_exponents: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_powers: jax.Array | np.ndarray | None = None,
    aux_source_dipole_radial_scale: float = 1.0,
    aux_source_atom_odd_exponents: jax.Array | np.ndarray | None = None,
    aux_source_atom_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_dipole_ee_scales: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_slater_decays: jax.Array | np.ndarray | None = None,
    aux_source_bond_odd_ee_scales: jax.Array | np.ndarray | None = None,
    external_blocks: tuple[ExternalCASBasisBlock, ...] | None = None,
    batch_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Estimate final matrices and per-block contributions.

    Per-block unprojected weak matrices and ground-projection moments are kept
    so final leave-one-out/bootstrap diagnostics can recompute the whole
    ``Q0``-projected ``(S,K,p)`` estimator for each resample.

    Returns:
        Aggregate overlap, Hamiltonian, source, per-block overlaps,
        Hamiltonians, sources, block sample counts, per-block projection
        numerators, projection norms, ground-Hamiltonian vectors, and
        ground-Hamiltonian scalars.

    Raises:
        ValueError: If block lists are inconsistent or empty.
    """
    if len(point_blocks) != len(density_blocks):
        msg = "point_blocks and density_blocks must have the same length"
        raise ValueError(msg)
    if not point_blocks:
        msg = "at least one final matrix block is required"
        raise ValueError(msg)
    external_count = external_cas_basis_count(external_blocks)
    n_basis = (
        head_count
        + 1
        + _auxiliary_source_count(
            aux_source_exponents,
            aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales,
            aux_source_dipole_radial_powers,
        )
        + external_count
    )

    @jax.jit
    def chunk_raw_contrib(points: jax.Array, density: jax.Array):
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=(aux_source_bond_odd_ee_slater_decays),
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        ground_values, ground_gradients = ground_values_and_gradients(ground, points)
        weights = 1 / density
        overlap = jnp.einsum("n,ni,nj->ij", weights, values, values)
        projection_norm = jnp.einsum("n,n,n->", weights, ground_values, ground_values)
        projection_numerator = jnp.einsum("n,n,ni->i", weights, ground_values, values)
        flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
        flat_ground_gradients = jnp.reshape(ground_gradients, (points.shape[0], -1))
        potential = potential_shift(ground, points)
        kinetic = 0.5 * jnp.einsum("nid,njd->nij", flat_gradients, flat_gradients)
        potential_matrix = (
            potential[:, None, None] * values[:, :, None] * values[:, None, :]
        )
        hamiltonian = jnp.einsum("n,nij->ij", weights, kinetic + potential_matrix)
        ground_kinetic_basis = 0.5 * jnp.einsum(
            "nd,nid->ni", flat_ground_gradients, flat_gradients
        )
        ground_potential_basis = potential[:, None] * ground_values[:, None] * values
        ground_hamiltonian = jnp.einsum(
            "n,ni->i", weights, ground_kinetic_basis + ground_potential_basis
        )
        ground_kinetic_norm = 0.5 * jnp.einsum(
            "nd,nd->n", flat_ground_gradients, flat_ground_gradients
        )
        ground_potential_norm = potential * ground_values * ground_values
        ground_hamiltonian_norm = jnp.einsum(
            "n,n->", weights, ground_kinetic_norm + ground_potential_norm
        )
        return (
            overlap,
            hamiltonian,
            projection_numerator,
            projection_norm,
            ground_hamiltonian,
            ground_hamiltonian_norm,
        )

    def chunk_raw_contrib_with_external(points: jax.Array, density: jax.Array):
        values, gradients, _ = source_aux_and_head_values_and_gradients(
            params,
            ground,
            points,
            aux_source_exponents=aux_source_exponents,
            aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
            aux_source_dipole_radial_scale=aux_source_dipole_radial_scale,
            aux_source_atom_odd_exponents=aux_source_atom_odd_exponents,
            aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
            aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
            aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
            aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
            aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
            head_count=head_count,
        )
        values, gradients = append_external_basis_values_gradients(
            values,
            gradients,
            ground,
            points,
            external_blocks,
        )
        ground_values, ground_gradients = ground_values_and_gradients(ground, points)
        weights = 1 / density
        overlap = jnp.einsum("n,ni,nj->ij", weights, values, values)
        projection_norm = jnp.einsum("n,n,n->", weights, ground_values, ground_values)
        projection_numerator = jnp.einsum("n,n,ni->i", weights, ground_values, values)
        flat_gradients = jnp.reshape(gradients, (*gradients.shape[:2], -1))
        flat_ground_gradients = jnp.reshape(ground_gradients, (points.shape[0], -1))
        potential = potential_shift(ground, points)
        kinetic = 0.5 * jnp.einsum("nid,njd->nij", flat_gradients, flat_gradients)
        potential_matrix = (
            potential[:, None, None] * values[:, :, None] * values[:, None, :]
        )
        hamiltonian = jnp.einsum("n,nij->ij", weights, kinetic + potential_matrix)
        ground_kinetic_basis = 0.5 * jnp.einsum(
            "nd,nid->ni", flat_ground_gradients, flat_gradients
        )
        ground_potential_basis = potential[:, None] * ground_values[:, None] * values
        ground_hamiltonian = jnp.einsum(
            "n,ni->i", weights, ground_kinetic_basis + ground_potential_basis
        )
        ground_kinetic_norm = 0.5 * jnp.einsum(
            "nd,nd->n", flat_ground_gradients, flat_ground_gradients
        )
        ground_potential_norm = potential * ground_values * ground_values
        ground_hamiltonian_norm = jnp.einsum(
            "n,n->", weights, ground_kinetic_norm + ground_potential_norm
        )
        return (
            overlap,
            hamiltonian,
            projection_numerator,
            projection_norm,
            ground_hamiltonian,
            ground_hamiltonian_norm,
        )

    raw_block_overlaps = []
    raw_block_hamiltonians = []
    block_projection_numerators = []
    block_projection_norms = []
    block_ground_hamiltonians = []
    block_ground_hamiltonian_norms = []
    block_counts = []
    for points_block, density_block in zip(point_blocks, density_blocks, strict=True):
        raw_overlap = np.zeros((n_basis, n_basis), dtype=np.float64)
        raw_hamiltonian = np.zeros((n_basis, n_basis), dtype=np.float64)
        projection_numerator = np.zeros(n_basis, dtype=np.float64)
        projection_norm = 0.0
        ground_hamiltonian = np.zeros(n_basis, dtype=np.float64)
        ground_hamiltonian_norm = 0.0
        block_samples = int(points_block.shape[0])
        for chunk in make_batches(block_samples, batch_size):
            chunk_points = jnp.asarray(points_block[chunk])
            chunk_density = jnp.asarray(density_block[chunk])
            if external_count:
                (
                    overlap_chunk,
                    hamiltonian_chunk,
                    numerator_chunk,
                    norm_chunk,
                    ground_hamiltonian_chunk,
                    ground_hamiltonian_norm_chunk,
                ) = chunk_raw_contrib_with_external(chunk_points, chunk_density)
            else:
                (
                    overlap_chunk,
                    hamiltonian_chunk,
                    numerator_chunk,
                    norm_chunk,
                    ground_hamiltonian_chunk,
                    ground_hamiltonian_norm_chunk,
                ) = chunk_raw_contrib(chunk_points, chunk_density)
            raw_overlap += np.asarray(overlap_chunk)
            raw_hamiltonian += np.asarray(hamiltonian_chunk)
            projection_numerator += np.asarray(numerator_chunk)
            projection_norm += float(norm_chunk)
            ground_hamiltonian += np.asarray(ground_hamiltonian_chunk)
            ground_hamiltonian_norm += float(ground_hamiltonian_norm_chunk)
        raw_block_overlaps.append(raw_overlap / block_samples)
        raw_block_hamiltonians.append(raw_hamiltonian / block_samples)
        block_projection_numerators.append(projection_numerator / block_samples)
        block_projection_norms.append(projection_norm / block_samples)
        block_ground_hamiltonians.append(ground_hamiltonian / block_samples)
        block_ground_hamiltonian_norms.append(ground_hamiltonian_norm / block_samples)
        block_counts.append(block_samples)
    counts = np.asarray(block_counts, dtype=np.float64)
    weights = counts / np.sum(counts)
    raw_block_overlaps_arr = np.asarray(raw_block_overlaps)
    raw_block_hamiltonians_arr = np.asarray(raw_block_hamiltonians)
    block_projection_numerators_arr = np.asarray(block_projection_numerators)
    block_projection_norms_arr = np.asarray(block_projection_norms)
    block_ground_hamiltonians_arr = np.asarray(block_ground_hamiltonians)
    block_ground_hamiltonian_norms_arr = np.asarray(block_ground_hamiltonian_norms)
    raw_overlap = np.tensordot(weights, raw_block_overlaps_arr, axes=(0, 0))
    raw_hamiltonian = np.tensordot(weights, raw_block_hamiltonians_arr, axes=(0, 0))
    projection_numerator = np.tensordot(
        weights, block_projection_numerators_arr, axes=(0, 0)
    )
    projection_norm = float(np.dot(weights, block_projection_norms_arr))
    ground_hamiltonian = np.tensordot(
        weights, block_ground_hamiltonians_arr, axes=(0, 0)
    )
    ground_hamiltonian_norm = float(np.dot(weights, block_ground_hamiltonian_norms_arr))
    overlap, hamiltonian, source = _project_raw_weak_matrices(
        raw_overlap,
        raw_hamiltonian,
        projection_numerator,
        projection_norm,
        ground_hamiltonian,
        ground_hamiltonian_norm,
    )
    block_overlaps = []
    block_hamiltonians = []
    block_sources = []
    for (
        raw_overlap_block,
        raw_hamiltonian_block,
        projection_numerator_block,
        projection_norm_block,
        ground_hamiltonian_block,
        ground_hamiltonian_norm_block,
    ) in zip(
        raw_block_overlaps_arr,
        raw_block_hamiltonians_arr,
        block_projection_numerators_arr,
        block_projection_norms_arr,
        block_ground_hamiltonians_arr,
        block_ground_hamiltonian_norms_arr,
        strict=True,
    ):
        block_overlap, block_hamiltonian, block_source = _project_raw_weak_matrices(
            raw_overlap_block,
            raw_hamiltonian_block,
            projection_numerator_block,
            float(projection_norm_block),
            ground_hamiltonian_block,
            float(ground_hamiltonian_norm_block),
        )
        block_overlaps.append(block_overlap)
        block_hamiltonians.append(block_hamiltonian)
        block_sources.append(block_source)
    return (
        overlap,
        hamiltonian,
        source,
        np.asarray(block_overlaps),
        np.asarray(block_hamiltonians),
        np.asarray(block_sources),
        counts,
        raw_block_overlaps_arr,
        raw_block_hamiltonians_arr,
        block_projection_numerators_arr,
        block_projection_norms_arr,
        block_ground_hamiltonians_arr,
        block_ground_hamiltonian_norms_arr,
    )


def print_cutoff_sensitivity_report(
    cutoff_diagnostics: dict[str, np.ndarray] | None,
) -> None:
    if cutoff_diagnostics is None:
        return
    cutoffs = np.asarray(cutoff_diagnostics["cutoffs"], dtype=np.float64)
    roots = np.asarray(cutoff_diagnostics["bright_roots"], dtype=np.float64)
    spread = np.asarray(cutoff_diagnostics["bright_root_spread_ev"], dtype=np.float64)
    if not (cutoffs.size and roots.ndim == 2 and roots.shape[1] > 0):
        return
    cutoff_text = ",".join(f"{float(value):.1e}" for value in cutoffs)
    root0_text = ",".join(f"{float(value):.10f}" for value in roots[:, 0])
    spread_text = ",".join(f"{float(value):.3e}" for value in spread)
    print(
        "cutoff_sensitivity_diagnostics "
        f"cutoffs={cutoff_text} "
        f"root0_ha={root0_text} "
        f"root_spread_ev={spread_text}"
    )


def strong_residual_region_summary_text(
    strong_residual: dict[str, np.ndarray | float | int],
) -> str:
    labels = np.asarray(
        strong_residual.get("strong_residual_region_labels", np.asarray([]))
    )
    eps_over_eta = np.asarray(
        strong_residual.get("strong_residual_region_epsilon_over_eta", np.asarray([])),
        dtype=np.float64,
    )
    fractions = np.asarray(
        strong_residual.get("strong_residual_region_residual_fraction", np.asarray([])),
        dtype=np.float64,
    )
    counts = np.asarray(
        strong_residual.get("strong_residual_region_counts", np.asarray([])),
        dtype=np.int64,
    )
    if labels.size == 0 or eps_over_eta.ndim != 2 or counts.size != labels.size:
        return ""
    label_text = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in labels
    ]
    with np.errstate(all="ignore"):
        max_eps = np.nanmax(eps_over_eta, axis=1)
        max_fraction = (
            np.nanmax(fractions, axis=1)
            if fractions.ndim == 2 and fractions.shape[0] == labels.size
            else np.full(labels.size, np.nan)
        )
    max_eps = np.where(np.isnan(max_eps), np.where(counts > 0, np.inf, 0.0), max_eps)
    max_fraction = np.where(
        np.isnan(max_fraction),
        np.where(counts > 0, np.inf, 0.0),
        max_fraction,
    )
    search = max_eps[1:] if max_eps.size > 1 else max_eps
    offset = 1 if max_eps.size > 1 else 0
    worst_idx = int(np.argmax(search) + offset) if search.size else 0
    labels_joined = ",".join(label_text)
    eps_joined = ",".join(f"{float(value):.3e}" for value in max_eps)
    frac_joined = ",".join(f"{float(value):.3e}" for value in max_fraction)
    count_joined = ",".join(str(int(value)) for value in counts)
    return (
        f"labels={labels_joined} "
        f"counts={count_joined} "
        f"max_epsilon_over_eta={eps_joined} "
        f"max_residual_fraction={frac_joined} "
        f"worst_region={label_text[worst_idx]} "
        f"worst_region_epsilon_over_eta={float(max_eps[worst_idx]):.3e}"
    )


STRONG_RESIDUAL_HARD_COUNTER_KEYS = (
    "strong_residual_density_nonfinite_count",
    "strong_residual_density_nonpositive_count",
    "strong_residual_overlap_nonfinite_count",
    "strong_residual_hamiltonian_nonfinite_count",
    "strong_residual_coeff_nonfinite_count",
    "strong_residual_value_nonfinite_count",
    "strong_residual_laplacian_nonfinite_count",
    "strong_residual_hbar_nonfinite_count",
    "strong_residual_point_residual_nonfinite_count",
    "strong_residual_residual_contrib_nonfinite_count",
    "strong_residual_source_contrib_nonfinite_count",
)


def strong_residual_hard_nonfinite_count(
    strong_residual: dict[str, np.ndarray | float | int],
) -> int:
    """Return hard-failure counter total for strong residual acceptance."""
    total = sum(
        max(int(strong_residual.get(key, 0)), 0)
        for key in STRONG_RESIDUAL_HARD_COUNTER_KEYS
    )
    system_condition_max = float(
        strong_residual.get("strong_residual_system_condition_max", np.nan)
    )
    if not np.isfinite(system_condition_max):
        total += 1
    return int(total)


def strong_residual_region_fraction_max(
    strong_residual: dict[str, np.ndarray | float | int],
    label: str,
) -> float:
    """Return the maximum residual contribution fraction for one region."""
    labels = np.asarray(
        strong_residual.get("strong_residual_region_labels", np.asarray([]))
    )
    fractions = np.asarray(
        strong_residual.get("strong_residual_region_residual_fraction", np.asarray([])),
        dtype=np.float64,
    )
    if labels.size == 0 or fractions.ndim != 2 or fractions.shape[0] != labels.size:
        return float("nan")
    label_text = np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in labels
        ]
    )
    matches = np.nonzero(label_text == label)[0]
    if matches.size == 0:
        return float("nan")
    values = fractions[int(matches[0])]
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def strong_residual_nonfinite_summary_text(
    strong_residual: dict[str, np.ndarray | float | int],
) -> str:
    """Summarize nonfinite strong-residual audit counters for logging.

    Returns:
        A compact log string, or an empty string when all counters are clean.
    """
    fields = [
        ("density_nonfinite", "strong_residual_density_nonfinite_count"),
        ("density_nonpositive", "strong_residual_density_nonpositive_count"),
        ("overlap_nonfinite", "strong_residual_overlap_nonfinite_count"),
        ("hamiltonian_nonfinite", "strong_residual_hamiltonian_nonfinite_count"),
        ("coeff_nonfinite", "strong_residual_coeff_nonfinite_count"),
        ("value_nonfinite", "strong_residual_value_nonfinite_count"),
        ("laplacian_nonfinite", "strong_residual_laplacian_nonfinite_count"),
        ("hbar_nonfinite", "strong_residual_hbar_nonfinite_count"),
        (
            "point_residual_nonfinite",
            "strong_residual_point_residual_nonfinite_count",
        ),
        (
            "residual_contrib_nonfinite",
            "strong_residual_residual_contrib_nonfinite_count",
        ),
        (
            "source_contrib_nonfinite",
            "strong_residual_source_contrib_nonfinite_count",
        ),
    ]
    parts = []
    total = 0
    for label, key in fields:
        value = int(strong_residual.get(key, 0))
        total += max(value, 0)
        parts.append(f"{label}={value}")
    system_condition_max = float(
        strong_residual.get("strong_residual_system_condition_max", np.nan)
    )
    density_min_positive = float(
        strong_residual.get("strong_residual_density_min_positive", np.nan)
    )
    density_max_finite = float(
        strong_residual.get("strong_residual_density_max_finite", np.nan)
    )
    if total == 0 and np.isfinite(system_condition_max):
        return ""
    parts.extend(
        [
            f"system_cond_max={system_condition_max:.3e}",
            f"density_min_positive={density_min_positive:.3e}",
            f"density_max_finite={density_max_finite:.3e}",
        ]
    )
    return " ".join(parts)


def print_strong_residual_report(
    strong_residual: dict[str, np.ndarray | float | int] | None,
) -> None:
    if strong_residual is None or int(strong_residual["strong_residual_samples"]) <= 0:
        return
    eps = np.asarray(strong_residual["strong_residual_epsilon"])
    omega_text = ",".join(
        f"{float(value):.6f}"
        for value in np.asarray(strong_residual["strong_residual_omegas"])
    )
    eps_text = ",".join(f"{float(value):.3e}" for value in eps)
    print(
        "strong_residual_diagnostics "
        f"samples={int(strong_residual['strong_residual_samples'])} "
        f"source_index={int(strong_residual['strong_residual_source_index'])} "
        f"eta={float(strong_residual['strong_residual_eta']):.3e} "
        f"max_epsilon="
        f"{float(strong_residual['strong_residual_max_epsilon']):.3e} "
        f"max_epsilon_over_eta="
        f"{float(strong_residual['strong_residual_max_epsilon_over_eta']):.3e} "
        f"omegas={omega_text} "
        f"epsilons={eps_text}"
    )
    region_summary = strong_residual_region_summary_text(strong_residual)
    if region_summary:
        print(f"strong_residual_region_diagnostics {region_summary}")
    nonfinite_summary = strong_residual_nonfinite_summary_text(strong_residual)
    if nonfinite_summary:
        print(f"strong_residual_nonfinite_diagnostics {nonfinite_summary}")


def print_validation_report(
    args: argparse.Namespace,
    ground: FermiNetGround,
    spectrum: ProjectedSpectrum,
    peaks: list[Peak],
    pmove: float,
    energy_stderr: float,
    *,
    active_heads: int,
    external_basis_count: int,
    enrichment: EnrichmentDiagnostics | None,
    moments: MomentDiagnostics,
    cutoff_diagnostics: dict[str, np.ndarray] | None = None,
    final_replica: dict[str, np.ndarray | float | int] | None = None,
    strong_residual: dict[str, np.ndarray | float | int] | None = None,
) -> None:
    print("JaQMC FermiNet BF-NKSR response")
    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_step={ground.checkpoint_step}")
    print(f"ground_energy_ha={ground.energy:.10f}")
    if not np.isnan(energy_stderr):
        print(f"ground_energy_stderr_ha={energy_stderr:.3e}")
    print(f"final_samples={args.final_samples} mcmc_pmove~{pmove:.3f}")
    print(f"response_flow={response_flow_name(args)}")
    print(f"active_response_heads={active_heads}")
    print(f"external_cas_basis_count={external_basis_count}")
    aux_source_count = _auxiliary_source_count(
        args.aux_source_gaussian_exponents,
        args.aux_source_atom_odd_gaussian_exponents,
        args.aux_source_atom_odd_slater_decays,
        args.aux_source_bond_odd_slater_decays,
        args.aux_source_dipole_ee_scales,
        args.aux_source_bond_odd_ee_slater_decays,
        args.aux_source_bond_odd_ee_scales,
        args.aux_source_dipole_radial_powers,
    )
    print(f"aux_source_count={aux_source_count}")
    print(f"output={args.output}")
    if enrichment is not None:
        print(
            "enrichment_diagnostics "
            f"accepted={enrichment.accepted} "
            f"reason={enrichment.accepted_reason} "
            f"attempt={enrichment.attempt:02d} "
            f"active_heads_before={enrichment.active_heads_before} "
            f"candidate_heads={enrichment.candidate_heads} "
            f"accepted_heads={enrichment.accepted_heads} "
            f"capture_ratio={enrichment.capture_ratio:.6f} "
            f"initial_capture={enrichment.initial_capture:.8e} "
            f"final_capture={enrichment.final_capture:.8e} "
            f"objective_delta={enrichment.objective_delta:.8e} "
            f"holdout_ratio_min={enrichment.holdout_capture_ratio_min:.6f} "
            f"holdout_delta_min={enrichment.holdout_objective_delta_min:.3e} "
            f"holdout_pass={enrichment.holdout_pass_count}"
            f"/{enrichment.holdout_count} "
            f"selected_pole={enrichment.selected_pole:.10f} "
            f"pole_improvement={enrichment.pole_improvement:.3e} "
            f"pole_spread={enrichment.pole_spread:.3e} "
            f"pole_pass={enrichment.pole_validation_pass_count}"
            f"/{enrichment.pole_validation_count} "
            f"production_ready={enrichment.production_ready} "
            f"strong_residual_hard_passed="
            f"{enrichment.strong_residual_hard_passed} "
            f"strong_residual_passed={enrichment.strong_residual_passed} "
            f"strong_residual_improved={enrichment.strong_residual_improved} "
            f"strong_residual_nonfinite_count="
            f"{enrichment.strong_residual_nonfinite_count} "
            f"strong_residual_old_epsilon_over_eta="
            f"{enrichment.strong_residual_epsilon_over_eta_old_max:.3e} "
            f"strong_residual_epsilon_over_eta="
            f"{enrichment.strong_residual_epsilon_over_eta_max:.3e} "
            f"strong_residual_new_over_old="
            f"{enrichment.strong_residual_epsilon_over_eta_ratio:.3e} "
            f"strong_residual_node_fraction="
            f"{enrichment.strong_residual_node_fraction_max:.3e} "
            f"strong_residual_node_fraction_new_over_old="
            f"{enrichment.strong_residual_node_fraction_ratio:.3e}"
        )
    print(
        "moment_diagnostics "
        f"m0_rel={moments.norm_rel_error:.3e} "
        f"m1_rel={moments.first_moment_rel_error:.3e} "
        f"min_weight={moments.min_weight:.3e} "
        f"overlap_condition={moments.overlap_condition:.3e}"
    )
    print_cutoff_sensitivity_report(cutoff_diagnostics)
    if final_replica is not None and int(final_replica["loo_count"]) > 0:
        print(
            "final_replica_diagnostics "
            f"loo_count={int(final_replica['loo_count'])} "
            f"projection_resampling="
            f"{bool(final_replica['projection_resampling'])} "
            f"loo_mean_ha={float(final_replica['loo_mean']):.10f} "
            f"loo_std_ha={float(final_replica['loo_std']):.3e} "
            f"loo_jackknife_se_ha={float(final_replica['loo_jackknife_se']):.3e} "
            f"loo_jackknife_se_ev="
            f"{float(final_replica['loo_jackknife_se_ev']):.3e} "
            f"loo_min_ha={float(final_replica['loo_min']):.10f} "
            f"loo_max_ha={float(final_replica['loo_max']):.10f}"
        )
        if int(final_replica["bootstrap_count"]) > 0:
            print(
                "final_bootstrap_diagnostics "
                f"replicates={int(final_replica['bootstrap_count'])} "
                f"mean_ha={float(final_replica['bootstrap_mean']):.10f} "
                f"se_ha={float(final_replica['bootstrap_se']):.3e} "
                f"se_ev={float(final_replica['bootstrap_se_ev']):.3e} "
                f"min_ha={float(final_replica['bootstrap_min']):.10f} "
                f"max_ha={float(final_replica['bootstrap_max']):.10f}"
            )
        root_mean = np.asarray(final_replica["loo_root_mean"], dtype=np.float64)
        root_se_ev = np.asarray(
            final_replica["loo_root_jackknife_se_ev"], dtype=np.float64
        )
        if root_mean.size:
            roots_text = ",".join(f"{float(value):.10f}" for value in root_mean)
            se_text = ",".join(f"{float(value):.3e}" for value in root_se_ev)
            print(
                "final_replica_root_diagnostics "
                f"loo_root_mean_ha={roots_text} "
                f"loo_jackknife_se_ev={se_text}"
            )
        bootstrap_root_mean = np.asarray(
            final_replica["bootstrap_root_mean"], dtype=np.float64
        )
        bootstrap_root_se_ev = np.asarray(
            final_replica["bootstrap_root_se_ev"], dtype=np.float64
        )
        if bootstrap_root_mean.size:
            roots_text = ",".join(
                f"{float(value):.10f}" for value in bootstrap_root_mean
            )
            se_text = ",".join(f"{float(value):.3e}" for value in bootstrap_root_se_ev)
            print(
                "final_bootstrap_root_diagnostics "
                f"root_mean_ha={roots_text} "
                f"se_ev={se_text}"
            )
    print_strong_residual_report(strong_residual)
    print("projected poles and weights")
    for idx, (pole, weight) in enumerate(
        zip(spectrum.excitation_energies, spectrum.weights[:, 0, 0], strict=False)
    ):
        print(f"root={idx:02d} pole_ha={pole:.10f} weight={float(weight.real):.10e}")
    print("peaks read from broadened neural spectrum")
    for peak in peaks:
        print(f"peak_ha={peak.energy:.10f} intensity={peak.intensity:.10e}")


def projected_spectrum_with_head_fallback(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    active_heads: int,
    aux_source_count: int = 0,
    external_basis_count: int = 0,
    source_in_basis: bool = True,
    overlap_cutoff: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, ProjectedSpectrum]:
    """Compute the final spectrum, discarding invalid tail heads if needed.

    Returns:
        Retained response-head count plus sliced matrices and spectrum.

    Raises:
        ValueError: If no final response basis is available.
    """
    source_basis_count = 1 if bool(source_in_basis) else 0
    if int(external_basis_count) > 0:
        basis_count = (
            int(active_heads)
            + source_basis_count
            + int(aux_source_count)
            + int(external_basis_count)
        )
        overlap_slice = np.asarray(overlap)[:basis_count, :basis_count]
        hamiltonian_slice = np.asarray(hamiltonian)[:basis_count, :basis_count]
        source_slice = np.asarray(source)[:basis_count]
        spectrum = projected_spectrum(
            overlap_slice,
            hamiltonian_slice,
            source_slice,
            overlap_cutoff=overlap_cutoff,
        )
        return active_heads, overlap_slice, hamiltonian_slice, source_slice, spectrum
    last_error: Exception | None = None
    for head_count in range(int(active_heads), -1, -1):
        basis_count = head_count + source_basis_count + int(aux_source_count)
        if basis_count < 1:
            continue
        overlap_slice = np.asarray(overlap)[:basis_count, :basis_count]
        hamiltonian_slice = np.asarray(hamiltonian)[:basis_count, :basis_count]
        source_slice = np.asarray(source)[:basis_count]
        try:
            spectrum = projected_spectrum(
                overlap_slice,
                hamiltonian_slice,
                source_slice,
                overlap_cutoff=overlap_cutoff,
            )
        except (np.linalg.LinAlgError, ValueError) as exc:
            last_error = exc
            print(
                "response_final_matrix_reject "
                f"head_count={head_count} reason={type(exc).__name__}: {exc}"
            )
            continue
        if head_count != active_heads:
            print(
                "response_final_matrix_fallback "
                f"requested_heads={active_heads} retained_heads={head_count}"
            )
        return head_count, overlap_slice, hamiltonian_slice, source_slice, spectrum
    if last_error is None:
        msg = "no final response basis was available"
        raise ValueError(msg)
    raise last_error


def _source_weight_norm(
    source: np.ndarray,
    weights: np.ndarray,
    *,
    source_in_basis: bool,
) -> float:
    if source_in_basis:
        return float(np.asarray(source)[0, 0].real)
    positive_weights = np.asarray(weights, dtype=np.float64)
    positive_weights = positive_weights[np.isfinite(positive_weights)]
    positive_weights = np.maximum(positive_weights, 0.0)
    return float(np.sum(positive_weights))


def _matrix_first_bright_pole(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    source_in_basis: bool = True,
) -> float:
    spectrum = projected_spectrum(
        overlap,
        hamiltonian,
        source,
        overlap_cutoff=overlap_cutoff,
    )
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    source_norm = _source_weight_norm(
        source,
        weights,
        source_in_basis=source_in_basis,
    )
    return first_bright_pole(
        spectrum.excitation_energies,
        weights,
        source_norm=source_norm,
        root_floor=root_floor,
        min_weight=min_weight,
    )


def _safe_matrix_first_bright_pole(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    source_in_basis: bool = True,
) -> float:
    try:
        return _matrix_first_bright_pole(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            source_in_basis=source_in_basis,
        )
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def _matrix_bright_poles(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    spectrum = projected_spectrum(
        overlap,
        hamiltonian,
        source,
        overlap_cutoff=overlap_cutoff,
    )
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    source_norm = _source_weight_norm(
        source,
        weights,
        source_in_basis=source_in_basis,
    )
    return bright_poles(
        spectrum.excitation_energies,
        weights,
        source_norm=source_norm,
        root_floor=root_floor,
        min_weight=min_weight,
        max_roots=max_roots,
    )


def _safe_matrix_bright_poles(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    try:
        return _matrix_bright_poles(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            source_in_basis=source_in_basis,
        )
    except (np.linalg.LinAlgError, ValueError):
        return np.full(max(0, int(max_roots)), np.nan, dtype=np.float64)


def _empty_final_replica_pole_diagnostics() -> dict[str, np.ndarray | float | int]:
    empty = np.asarray([], dtype=np.float64)
    empty_matrix = np.empty((0, 0), dtype=np.float64)
    empty_int = np.asarray([], dtype=np.int64)
    return {
        "counts": empty,
        "projection_resampling": False,
        "block_poles": empty,
        "block_bright_poles": empty_matrix,
        "loo_poles": empty,
        "loo_bright_poles": empty_matrix,
        "loo_count": 0,
        "loo_mean": float("nan"),
        "loo_std": float("nan"),
        "loo_jackknife_se": float("nan"),
        "loo_jackknife_se_ev": float("nan"),
        "loo_min": float("nan"),
        "loo_max": float("nan"),
        "loo_root_count": empty_int,
        "loo_root_mean": empty,
        "loo_root_std": empty,
        "loo_root_jackknife_se": empty,
        "loo_root_jackknife_se_ev": empty,
        "loo_root_min": empty,
        "loo_root_max": empty,
        "bootstrap_poles": empty,
        "bootstrap_bright_poles": empty_matrix,
        "bootstrap_count": 0,
        "bootstrap_mean": float("nan"),
        "bootstrap_std": float("nan"),
        "bootstrap_se": float("nan"),
        "bootstrap_se_ev": float("nan"),
        "bootstrap_min": float("nan"),
        "bootstrap_max": float("nan"),
        "bootstrap_root_count": empty_int,
        "bootstrap_root_mean": empty,
        "bootstrap_root_std": empty,
        "bootstrap_root_se": empty,
        "bootstrap_root_se_ev": empty,
        "bootstrap_root_min": empty,
        "bootstrap_root_max": empty,
    }


def _pole_sample_summary(poles: np.ndarray) -> tuple[int, float, float, float, float]:
    finite = poles[np.isfinite(poles)]
    if not finite.size:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return (
        int(finite.size),
        float(np.mean(finite)),
        std,
        float(np.min(finite)),
        float(np.max(finite)),
    )


def _pole_matrix_summary(
    poles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poles = np.asarray(poles, dtype=np.float64)
    if poles.ndim != 2 or poles.shape[1] == 0:
        empty = np.asarray([], dtype=np.float64)
        return np.asarray([], dtype=np.int64), empty, empty, empty, empty
    finite = np.isfinite(poles)
    counts = np.sum(finite, axis=0).astype(np.int64)
    means = np.full(poles.shape[1], np.nan, dtype=np.float64)
    stds = np.full(poles.shape[1], np.nan, dtype=np.float64)
    mins = np.full(poles.shape[1], np.nan, dtype=np.float64)
    maxs = np.full(poles.shape[1], np.nan, dtype=np.float64)
    for root_idx in range(poles.shape[1]):
        values = poles[finite[:, root_idx], root_idx]
        if values.size:
            means[root_idx] = float(np.mean(values))
            stds[root_idx] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            mins[root_idx] = float(np.min(values))
            maxs[root_idx] = float(np.max(values))
    return counts, means, stds, mins, maxs


def _block_first_bright_poles(
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    source_in_basis: bool = True,
) -> np.ndarray:
    values = [
        _safe_matrix_first_bright_pole(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            source_in_basis=source_in_basis,
        )
        for overlap, hamiltonian, source in zip(
            overlaps, hamiltonians, sources, strict=True
        )
    ]
    return np.asarray(values, dtype=np.float64)


def _block_bright_poles(
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    values = [
        _safe_matrix_bright_poles(
            overlap,
            hamiltonian,
            source,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            source_in_basis=source_in_basis,
        )
        for overlap, hamiltonian, source in zip(
            overlaps, hamiltonians, sources, strict=True
        )
    ]
    if not values:
        return np.empty((0, max(0, int(max_roots))), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _leave_one_out_first_bright_poles(
    counts: np.ndarray,
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    source_in_basis: bool = True,
) -> np.ndarray:
    total_count = float(np.sum(counts))
    if counts.size < 2 or total_count <= 0:
        return np.asarray([], dtype=np.float64)
    weights = counts / total_count
    total_overlap = np.tensordot(weights, overlaps, axes=(0, 0))
    total_hamiltonian = np.tensordot(weights, hamiltonians, axes=(0, 0))
    total_source = np.tensordot(weights, sources, axes=(0, 0))
    loo_values = []
    for count, overlap, hamiltonian, source in zip(
        counts, overlaps, hamiltonians, sources, strict=True
    ):
        remaining = total_count - float(count)
        if remaining <= 0:
            loo_values.append(float("nan"))
            continue
        loo_values.append(
            _safe_matrix_first_bright_pole(
                (total_overlap * total_count - overlap * count) / remaining,
                (total_hamiltonian * total_count - hamiltonian * count) / remaining,
                (total_source * total_count - source * count) / remaining,
                overlap_cutoff=overlap_cutoff,
                root_floor=root_floor,
                min_weight=min_weight,
                source_in_basis=source_in_basis,
            )
        )
    return np.asarray(loo_values, dtype=np.float64)


def _leave_one_out_bright_poles(
    counts: np.ndarray,
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    root_count = max(0, int(max_roots))
    total_count = float(np.sum(counts))
    if counts.size < 2 or total_count <= 0:
        return np.empty((0, root_count), dtype=np.float64)
    weights = counts / total_count
    total_overlap = np.tensordot(weights, overlaps, axes=(0, 0))
    total_hamiltonian = np.tensordot(weights, hamiltonians, axes=(0, 0))
    total_source = np.tensordot(weights, sources, axes=(0, 0))
    loo_values = []
    for count, overlap, hamiltonian, source in zip(
        counts, overlaps, hamiltonians, sources, strict=True
    ):
        remaining = total_count - float(count)
        if remaining <= 0:
            loo_values.append(np.full(root_count, np.nan, dtype=np.float64))
            continue
        loo_values.append(
            _safe_matrix_bright_poles(
                (total_overlap * total_count - overlap * count) / remaining,
                (total_hamiltonian * total_count - hamiltonian * count) / remaining,
                (total_source * total_count - source * count) / remaining,
                overlap_cutoff=overlap_cutoff,
                root_floor=root_floor,
                min_weight=min_weight,
                max_roots=root_count,
                source_in_basis=source_in_basis,
            )
        )
    return np.asarray(loo_values, dtype=np.float64)


def _bootstrap_first_bright_poles(
    counts: np.ndarray,
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    total_count = float(np.sum(counts))
    if counts.size < 2 or total_count <= 0 or bootstrap_replicates <= 0:
        return np.asarray([], dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_values = []
    for _ in range(int(bootstrap_replicates)):
        indices = rng.integers(0, counts.size, size=counts.size)
        selected_counts = counts[indices]
        selected_total = float(np.sum(selected_counts))
        if selected_total <= 0:
            bootstrap_values.append(float("nan"))
            continue
        weights = selected_counts / selected_total
        bootstrap_values.append(
            _safe_matrix_first_bright_pole(
                np.tensordot(weights, overlaps[indices], axes=(0, 0)),
                np.tensordot(weights, hamiltonians[indices], axes=(0, 0)),
                np.tensordot(weights, sources[indices], axes=(0, 0)),
                overlap_cutoff=overlap_cutoff,
                root_floor=root_floor,
                min_weight=min_weight,
                source_in_basis=source_in_basis,
            )
        )
    return np.asarray(bootstrap_values, dtype=np.float64)


def _bootstrap_bright_poles(
    counts: np.ndarray,
    overlaps: np.ndarray,
    hamiltonians: np.ndarray,
    sources: np.ndarray,
    *,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    source_in_basis: bool = True,
) -> np.ndarray:
    root_count = max(0, int(max_roots))
    total_count = float(np.sum(counts))
    if counts.size < 2 or total_count <= 0 or bootstrap_replicates <= 0:
        return np.empty((0, root_count), dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_values = []
    for _ in range(int(bootstrap_replicates)):
        indices = rng.integers(0, counts.size, size=counts.size)
        selected_counts = counts[indices]
        selected_total = float(np.sum(selected_counts))
        if selected_total <= 0:
            bootstrap_values.append(np.full(root_count, np.nan, dtype=np.float64))
            continue
        weights = selected_counts / selected_total
        bootstrap_values.append(
            _safe_matrix_bright_poles(
                np.tensordot(weights, overlaps[indices], axes=(0, 0)),
                np.tensordot(weights, hamiltonians[indices], axes=(0, 0)),
                np.tensordot(weights, sources[indices], axes=(0, 0)),
                overlap_cutoff=overlap_cutoff,
                root_floor=root_floor,
                min_weight=min_weight,
                max_roots=root_count,
                source_in_basis=source_in_basis,
            )
        )
    return np.asarray(bootstrap_values, dtype=np.float64)


def _aggregate_projected_raw_blocks(
    weights: np.ndarray,
    raw_overlaps: np.ndarray,
    raw_hamiltonians: np.ndarray,
    projection_numerators: np.ndarray,
    projection_norms: np.ndarray,
    ground_hamiltonians: np.ndarray,
    ground_hamiltonian_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_overlap = np.tensordot(weights, raw_overlaps, axes=(0, 0))
    raw_hamiltonian = np.tensordot(weights, raw_hamiltonians, axes=(0, 0))
    projection_numerator = np.tensordot(weights, projection_numerators, axes=(0, 0))
    projection_norm = float(np.dot(weights, projection_norms))
    ground_hamiltonian = np.tensordot(weights, ground_hamiltonians, axes=(0, 0))
    ground_hamiltonian_norm = float(np.dot(weights, ground_hamiltonian_norms))
    return _project_raw_weak_matrices(
        raw_overlap,
        raw_hamiltonian,
        projection_numerator,
        projection_norm,
        ground_hamiltonian,
        ground_hamiltonian_norm,
    )


def _raw_projection_blocks_are_available(
    *arrays: np.ndarray | None,
) -> bool:
    return all(array is not None for array in arrays)


def _projected_raw_block_matrices(
    counts: np.ndarray,
    raw_overlaps: np.ndarray,
    raw_hamiltonians: np.ndarray,
    projection_numerators: np.ndarray,
    projection_norms: np.ndarray,
    ground_hamiltonians: np.ndarray,
    ground_hamiltonian_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlaps = []
    hamiltonians = []
    sources = []
    for idx in range(counts.size):
        overlap, hamiltonian, source = _aggregate_projected_raw_blocks(
            np.asarray([1.0], dtype=np.float64),
            raw_overlaps[idx : idx + 1],
            raw_hamiltonians[idx : idx + 1],
            projection_numerators[idx : idx + 1],
            projection_norms[idx : idx + 1],
            ground_hamiltonians[idx : idx + 1],
            ground_hamiltonian_norms[idx : idx + 1],
        )
        overlaps.append(overlap)
        hamiltonians.append(hamiltonian)
        sources.append(source)
    return np.asarray(overlaps), np.asarray(hamiltonians), np.asarray(sources)


def _leave_one_out_projected_raw_matrices(
    counts: np.ndarray,
    raw_overlaps: np.ndarray,
    raw_hamiltonians: np.ndarray,
    projection_numerators: np.ndarray,
    projection_norms: np.ndarray,
    ground_hamiltonians: np.ndarray,
    ground_hamiltonian_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_count = float(np.sum(counts))
    if counts.size < 2 or total_count <= 0:
        basis = raw_overlaps.shape[1]
        return (
            np.empty((0, basis, basis), dtype=np.float64),
            np.empty((0, basis, basis), dtype=np.float64),
            np.empty((0, basis, 1), dtype=np.float64),
        )
    overlaps = []
    hamiltonians = []
    sources = []
    for idx, count in enumerate(counts):
        remaining = total_count - float(count)
        if remaining <= 0:
            continue
        mask = np.ones(counts.size, dtype=bool)
        mask[idx] = False
        weights = counts[mask] / remaining
        overlap, hamiltonian, source = _aggregate_projected_raw_blocks(
            weights,
            raw_overlaps[mask],
            raw_hamiltonians[mask],
            projection_numerators[mask],
            projection_norms[mask],
            ground_hamiltonians[mask],
            ground_hamiltonian_norms[mask],
        )
        overlaps.append(overlap)
        hamiltonians.append(hamiltonian)
        sources.append(source)
    return np.asarray(overlaps), np.asarray(hamiltonians), np.asarray(sources)


def _bootstrap_projected_raw_matrices(
    counts: np.ndarray,
    raw_overlaps: np.ndarray,
    raw_hamiltonians: np.ndarray,
    projection_numerators: np.ndarray,
    projection_norms: np.ndarray,
    ground_hamiltonians: np.ndarray,
    ground_hamiltonian_norms: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if counts.size < 2 or float(np.sum(counts)) <= 0 or bootstrap_replicates <= 0:
        basis = raw_overlaps.shape[1]
        return (
            np.empty((0, basis, basis), dtype=np.float64),
            np.empty((0, basis, basis), dtype=np.float64),
            np.empty((0, basis, 1), dtype=np.float64),
        )
    rng = np.random.default_rng(int(bootstrap_seed))
    overlaps = []
    hamiltonians = []
    sources = []
    for _ in range(int(bootstrap_replicates)):
        indices = rng.integers(0, counts.size, size=counts.size)
        selected_counts = counts[indices]
        selected_total = float(np.sum(selected_counts))
        if selected_total <= 0:
            continue
        weights = selected_counts / selected_total
        overlap, hamiltonian, source = _aggregate_projected_raw_blocks(
            weights,
            raw_overlaps[indices],
            raw_hamiltonians[indices],
            projection_numerators[indices],
            projection_norms[indices],
            ground_hamiltonians[indices],
            ground_hamiltonian_norms[indices],
        )
        overlaps.append(overlap)
        hamiltonians.append(hamiltonian)
        sources.append(source)
    return np.asarray(overlaps), np.asarray(hamiltonians), np.asarray(sources)


def final_replica_pole_diagnostics(
    block_overlaps: np.ndarray | None,
    block_hamiltonians: np.ndarray | None,
    block_sources: np.ndarray | None,
    block_counts: np.ndarray | None,
    *,
    block_raw_overlaps: np.ndarray | None = None,
    block_raw_hamiltonians: np.ndarray | None = None,
    block_projection_numerators: np.ndarray | None = None,
    block_projection_norms: np.ndarray | None = None,
    block_ground_hamiltonians: np.ndarray | None = None,
    block_ground_hamiltonian_norms: np.ndarray | None = None,
    retained_heads: int,
    aux_source_count: int = 0,
    external_basis_count: int = 0,
    source_in_basis: bool = True,
    overlap_cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int = 5,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, np.ndarray | float | int]:
    """Compute block, jackknife, and bootstrap final pole diagnostics.

    Returns:
        Arrays and summary statistics for final-matrix replica sensitivity.
    """
    if (
        block_overlaps is None
        or block_hamiltonians is None
        or block_sources is None
        or block_counts is None
    ):
        return _empty_final_replica_pole_diagnostics()
    basis_count = (
        int(retained_heads)
        + (1 if bool(source_in_basis) else 0)
        + int(aux_source_count)
        + int(external_basis_count)
    )
    counts = np.asarray(block_counts, dtype=np.float64)
    overlaps = np.asarray(block_overlaps)[:, :basis_count, :basis_count]
    hamiltonians = np.asarray(block_hamiltonians)[:, :basis_count, :basis_count]
    sources = np.asarray(block_sources)[:, :basis_count, :]
    max_roots = max(0, int(max_roots))
    projection_resampling = _raw_projection_blocks_are_available(
        block_raw_overlaps,
        block_raw_hamiltonians,
        block_projection_numerators,
        block_projection_norms,
        block_ground_hamiltonians,
        block_ground_hamiltonian_norms,
    )
    if projection_resampling:
        raw_overlaps = np.asarray(block_raw_overlaps, dtype=np.float64)[
            :, :basis_count, :basis_count
        ]
        raw_hamiltonians = np.asarray(block_raw_hamiltonians, dtype=np.float64)[
            :, :basis_count, :basis_count
        ]
        projection_numerators = np.asarray(
            block_projection_numerators, dtype=np.float64
        )[:, :basis_count]
        projection_norms = np.asarray(block_projection_norms, dtype=np.float64)
        ground_hamiltonians = np.asarray(block_ground_hamiltonians, dtype=np.float64)[
            :, :basis_count
        ]
        ground_hamiltonian_norms = np.asarray(
            block_ground_hamiltonian_norms, dtype=np.float64
        )
        overlaps, hamiltonians, sources = _projected_raw_block_matrices(
            counts,
            raw_overlaps,
            raw_hamiltonians,
            projection_numerators,
            projection_norms,
            ground_hamiltonians,
            ground_hamiltonian_norms,
        )
    block_poles_arr = _block_first_bright_poles(
        overlaps,
        hamiltonians,
        sources,
        overlap_cutoff=overlap_cutoff,
        root_floor=root_floor,
        min_weight=min_weight,
        source_in_basis=source_in_basis,
    )
    block_bright_poles = _block_bright_poles(
        overlaps,
        hamiltonians,
        sources,
        overlap_cutoff=overlap_cutoff,
        root_floor=root_floor,
        min_weight=min_weight,
        max_roots=max_roots,
        source_in_basis=source_in_basis,
    )
    if projection_resampling:
        loo_overlaps, loo_hamiltonians, loo_sources = (
            _leave_one_out_projected_raw_matrices(
                counts,
                raw_overlaps,
                raw_hamiltonians,
                projection_numerators,
                projection_norms,
                ground_hamiltonians,
                ground_hamiltonian_norms,
            )
        )
        loo_poles = _block_first_bright_poles(
            loo_overlaps,
            loo_hamiltonians,
            loo_sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            source_in_basis=source_in_basis,
        )
        loo_bright_poles = _block_bright_poles(
            loo_overlaps,
            loo_hamiltonians,
            loo_sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            source_in_basis=source_in_basis,
        )
    else:
        loo_poles = _leave_one_out_first_bright_poles(
            counts,
            overlaps,
            hamiltonians,
            sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            source_in_basis=source_in_basis,
        )
        loo_bright_poles = _leave_one_out_bright_poles(
            counts,
            overlaps,
            hamiltonians,
            sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            source_in_basis=source_in_basis,
        )
    loo_count, loo_mean, loo_std, loo_min, loo_max = _pole_sample_summary(loo_poles)
    loo_jackknife_se = (
        float((loo_count - 1) / np.sqrt(loo_count) * loo_std)
        if loo_count > 1
        else (0.0 if loo_count == 1 else float("nan"))
    )
    (
        loo_root_count,
        loo_root_mean,
        loo_root_std,
        loo_root_min,
        loo_root_max,
    ) = _pole_matrix_summary(loo_bright_poles)
    loo_root_jackknife_se = np.full_like(loo_root_std, np.nan)
    positive_loo = loo_root_count > 0
    loo_root_jackknife_se[positive_loo] = 0.0
    multi_loo = loo_root_count > 1
    loo_root_jackknife_se[multi_loo] = (
        (loo_root_count[multi_loo] - 1) / np.sqrt(loo_root_count[multi_loo])
    ) * loo_root_std[multi_loo]
    if projection_resampling:
        bootstrap_overlaps, bootstrap_hamiltonians, bootstrap_sources = (
            _bootstrap_projected_raw_matrices(
                counts,
                raw_overlaps,
                raw_hamiltonians,
                projection_numerators,
                projection_norms,
                ground_hamiltonians,
                ground_hamiltonian_norms,
                bootstrap_replicates=max(0, int(bootstrap_replicates)),
                bootstrap_seed=bootstrap_seed,
            )
        )
        bootstrap_poles = _block_first_bright_poles(
            bootstrap_overlaps,
            bootstrap_hamiltonians,
            bootstrap_sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            source_in_basis=source_in_basis,
        )
        bootstrap_bright_poles = _block_bright_poles(
            bootstrap_overlaps,
            bootstrap_hamiltonians,
            bootstrap_sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            source_in_basis=source_in_basis,
        )
    else:
        bootstrap_poles = _bootstrap_first_bright_poles(
            counts,
            overlaps,
            hamiltonians,
            sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            bootstrap_replicates=max(0, int(bootstrap_replicates)),
            bootstrap_seed=bootstrap_seed,
            source_in_basis=source_in_basis,
        )
        bootstrap_bright_poles = _bootstrap_bright_poles(
            counts,
            overlaps,
            hamiltonians,
            sources,
            overlap_cutoff=overlap_cutoff,
            root_floor=root_floor,
            min_weight=min_weight,
            max_roots=max_roots,
            bootstrap_replicates=max(0, int(bootstrap_replicates)),
            bootstrap_seed=bootstrap_seed,
            source_in_basis=source_in_basis,
        )
    (
        bootstrap_count,
        bootstrap_mean,
        bootstrap_std,
        bootstrap_min,
        bootstrap_max,
    ) = _pole_sample_summary(bootstrap_poles)
    (
        bootstrap_root_count,
        bootstrap_root_mean,
        bootstrap_root_std,
        bootstrap_root_min,
        bootstrap_root_max,
    ) = _pole_matrix_summary(bootstrap_bright_poles)
    return {
        "counts": counts,
        "projection_resampling": bool(projection_resampling),
        "block_poles": block_poles_arr,
        "block_bright_poles": block_bright_poles,
        "loo_poles": loo_poles,
        "loo_bright_poles": loo_bright_poles,
        "loo_count": loo_count,
        "loo_mean": loo_mean,
        "loo_std": loo_std,
        "loo_jackknife_se": loo_jackknife_se,
        "loo_jackknife_se_ev": loo_jackknife_se * HARTREE_TO_EV,
        "loo_min": loo_min,
        "loo_max": loo_max,
        "loo_root_count": loo_root_count,
        "loo_root_mean": loo_root_mean,
        "loo_root_std": loo_root_std,
        "loo_root_jackknife_se": loo_root_jackknife_se,
        "loo_root_jackknife_se_ev": loo_root_jackknife_se * HARTREE_TO_EV,
        "loo_root_min": loo_root_min,
        "loo_root_max": loo_root_max,
        "bootstrap_poles": bootstrap_poles,
        "bootstrap_bright_poles": bootstrap_bright_poles,
        "bootstrap_count": bootstrap_count,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_std": bootstrap_std,
        "bootstrap_se": bootstrap_std,
        "bootstrap_se_ev": bootstrap_std * HARTREE_TO_EV,
        "bootstrap_min": bootstrap_min,
        "bootstrap_max": bootstrap_max,
        "bootstrap_root_count": bootstrap_root_count,
        "bootstrap_root_mean": bootstrap_root_mean,
        "bootstrap_root_std": bootstrap_root_std,
        "bootstrap_root_se": bootstrap_root_std,
        "bootstrap_root_se_ev": bootstrap_root_std * HARTREE_TO_EV,
        "bootstrap_root_min": bootstrap_root_min,
        "bootstrap_root_max": bootstrap_root_max,
    }


def _cutoff_sensitivity_one(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    cutoff: float,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    spectrum = projected_spectrum(
        overlap,
        hamiltonian,
        source,
        overlap_cutoff=float(cutoff),
    )
    weights = np.asarray(spectrum.weights[:, 0, 0].real, dtype=np.float64)
    source_norm = _source_weight_norm(
        source,
        weights,
        source_in_basis=source_in_basis,
    )
    normalized = weights / max(source_norm, 1e-12)
    poles = np.asarray(spectrum.excitation_energies, dtype=np.float64)
    keep = (poles > root_floor) & (normalized >= min_weight)
    roots = np.full(max_roots, np.nan, dtype=np.float64)
    norm_weights = np.full(max_roots, np.nan, dtype=np.float64)
    selected_roots = poles[keep][:max_roots]
    selected_weights = normalized[keep][:max_roots]
    roots[: selected_roots.size] = selected_roots
    norm_weights[: selected_weights.size] = selected_weights
    moments = moment_diagnostics(
        overlap,
        hamiltonian,
        source,
        spectrum,
        overlap_cutoff=float(cutoff),
        source_in_basis=source_in_basis,
    )
    return (
        roots,
        norm_weights,
        int(spectrum.excitation_energies.size),
        moments.overlap_condition,
        moments.norm_rel_error,
        moments.first_moment_rel_error,
    )


def cutoff_sensitivity_diagnostics(
    overlap: np.ndarray,
    hamiltonian: np.ndarray,
    source: np.ndarray,
    *,
    cutoffs: np.ndarray,
    root_floor: float,
    min_weight: float,
    max_roots: int,
    source_in_basis: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate final spectrum diagnostics across whitening cutoffs.

    Returns:
        Arrays for saved cutoff-sensitivity audits.  Each cutoff reruns the
        full whitening/eigensolve map on the final matrices and records the
        visible source-bright roots, normalized source weights, source moments,
        retained dimension, and overlap condition.
    """
    cutoffs = np.asarray(cutoffs, dtype=np.float64)
    root_count = max(0, int(max_roots))
    roots = np.full((cutoffs.size, root_count), np.nan, dtype=np.float64)
    norm_weights = np.full_like(roots, np.nan)
    retained = np.zeros(cutoffs.size, dtype=np.int64)
    condition = np.full(cutoffs.size, np.nan, dtype=np.float64)
    moment_norm_rel = np.full(cutoffs.size, np.nan, dtype=np.float64)
    moment_first_rel = np.full(cutoffs.size, np.nan, dtype=np.float64)
    success = np.zeros(cutoffs.size, dtype=bool)
    for idx, cutoff in enumerate(cutoffs):
        try:
            (
                roots[idx],
                norm_weights[idx],
                retained[idx],
                condition[idx],
                moment_norm_rel[idx],
                moment_first_rel[idx],
            ) = _cutoff_sensitivity_one(
                overlap,
                hamiltonian,
                source,
                cutoff=float(cutoff),
                root_floor=root_floor,
                min_weight=min_weight,
                max_roots=root_count,
                source_in_basis=source_in_basis,
            )
            success[idx] = True
        except (np.linalg.LinAlgError, ValueError):
            continue
    root_spread_ev = np.full(root_count, np.nan, dtype=np.float64)
    for root_idx in range(root_count):
        values = roots[:, root_idx]
        finite = values[np.isfinite(values)]
        if finite.size:
            root_spread_ev[root_idx] = (
                float(np.max(finite) - np.min(finite)) * HARTREE_TO_EV
            )
    return {
        "cutoffs": cutoffs,
        "success": success,
        "retained": retained,
        "condition": condition,
        "moment_norm_rel_error": moment_norm_rel,
        "moment_first_rel_error": moment_first_rel,
        "bright_roots": roots,
        "bright_norm_weights": norm_weights,
        "bright_root_spread_ev": root_spread_ev,
    }


def response_flow_name(args: argparse.Namespace) -> str:
    del args
    return OFFICIAL_RESPONSE_FLOW


def _namespace_sequence_has_items(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _namespace_positive_finite(value: Any) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(scalar) and scalar > 0.0)


def _namespace_nonzero_or_nonfinite(value: Any) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return True
    return bool((not np.isfinite(scalar)) or abs(scalar) > 0.0)


def validate_official_response_profile(args: argparse.Namespace) -> None:
    """Validate the single CAS-dressed-teacher projected-resolvent workflow.

    Raises:
        ValueError: If an argument selects a route outside the official path.
    """
    errors: list[str] = []

    def require_equal(name: str, expected: Any, flag: str) -> None:
        actual = getattr(args, name, expected)
        if actual != expected:
            errors.append(f"{flag}={actual!r} must be {expected!r}")

    def reject(condition: bool, flag: str, reason: str) -> None:
        if condition:
            errors.append(f"{flag}: {reason}")

    require_equal("training_flow", "cas-dressed-teacher", "--training-flow")
    require_equal("warm_start", "casscf", "--warm-start")
    require_equal("final_sampling", OFFICIAL_FINAL_SAMPLING, "--final-sampling")
    reject(
        int(getattr(args, "warm_start_heads", 0)) < 1,
        "--warm-start-heads",
        "Krylov-CASSCF dressed-teacher workflow needs at least one teacher",
    )
    reject(
        not _namespace_positive_finite(
            getattr(
                args,
                "response_source_envelope_core_decay",
                OFFICIAL_SOURCE_ENVELOPE_CORE_DECAY,
            )
        ),
        "--response-source-envelope-core-decay",
        "fixed-CAS final sampling requires a positive core decay",
    )
    reject(
        not _namespace_positive_finite(
            getattr(
                args,
                "response_source_envelope_diffuse_decay",
                OFFICIAL_SOURCE_ENVELOPE_DIFFUSE_DECAY,
            )
        ),
        "--response-source-envelope-diffuse-decay",
        "fixed-CAS final sampling requires a positive diffuse decay",
    )
    reject(
        int(getattr(args, "final_sobol_replicas", 1)) < 1,
        "--final-sobol-replicas",
        "fixed-CAS final sampling needs at least one Sobol block",
    )

    if errors:
        msg = (
            "official BF-NKSR response has one supported CAS-dressed teacher path "
            f"({OFFICIAL_RESPONSE_FLOW}); unsupported options: " + "; ".join(errors)
        )
        raise ValueError(msg)


def apply_official_partial_wave_closure(
    args: argparse.Namespace,
    *,
    electron_count: int | None = None,
) -> None:
    """Install the internal diagnostic closure basis for official workflow."""
    args.aux_source_dipole_radial_powers = list(OFFICIAL_PARTIAL_WAVE_CLOSURE_POWERS)
    args.aux_source_dipole_radial_scale = OFFICIAL_PARTIAL_WAVE_CLOSURE_SCALE
    args.aux_source_dipole_ee_scales = (
        list(OFFICIAL_CORRELATED_DIPOLE_EE_SCALES)
        if electron_count is not None and int(electron_count) >= 2
        else []
    )
    args.residual_aux_source_weight = 0.0


def residual_source_weights_from_inputs(
    *,
    source_count: int,
    explicit_weights: list[float] | np.ndarray | None,
    physical_source_weight: float,
    aux_source_weight: float,
) -> np.ndarray:
    """Build paper-style source-channel weights for residual enrichment.

    Returns:
        Unnormalized non-negative source-channel weights.

    Raises:
        ValueError: If the source count or requested weights are invalid.
    """
    if source_count < 1:
        msg = "source_count must be positive"
        raise ValueError(msg)
    if explicit_weights is not None and len(explicit_weights):
        weights = np.asarray(explicit_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (source_count,):
            msg = (
                "--residual-source-weights must have length equal to "
                f"1 + aux_source_count ({source_count})"
            )
            raise ValueError(msg)
    else:
        weights = np.full(source_count, float(aux_source_weight), dtype=np.float64)
        weights[0] = float(physical_source_weight)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        msg = "residual source weights must be finite and non-negative"
        raise ValueError(msg)
    if not np.any(weights > 0):
        msg = "residual source weights must contain a positive weight"
        raise ValueError(msg)
    return weights


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "Run the Krylov-CASSCF-teacher neural BF-NKSR projected-resolvent "
            "workflow: a source-Hamiltonian CASSCF/CASCI Krylov SVD block "
            "supervises direct neural response heads, then final matrices and "
            "spectra are evaluated in real space with the NN-VMC ground state."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ground-energy", type=float, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("ferminet_bfnksr_response.npz")
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument(
        "--warm-start",
        choices=["casscf"],
        default="casscf",
        help=(
            "Official fixed-block initialization. CASSCF/CASCI roots are "
            "converted to source-Hamiltonian Krylov teachers for direct "
            "neural response heads; final matrices and spectra discard the "
            "QC evaluator."
        ),
    )
    parser.add_argument("--warm-start-heads", type=int, default=None)
    parser.add_argument("--warm-start-samples", type=int, default=32768)
    parser.add_argument("--warm-start-epochs", type=int, default=200)
    parser.add_argument("--warm-start-batch-size", type=int, default=4096)
    parser.add_argument("--warm-start-learning-rate", type=float, default=0.001)
    parser.add_argument("--warm-start-ridge", type=float, default=1e-8)
    parser.add_argument(
        "--production-sampler",
        choices=OFFICIAL_PRODUCTION_SAMPLERS,
        default=OFFICIAL_PRODUCTION_SAMPLER,
        help="Production sampling strategy for dressed-teacher matrices.",
    )
    parser.add_argument("--production-leverage-candidate-factor", type=int, default=2)
    parser.add_argument(
        "--production-leverage-max-candidates",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--production-leverage-gradient-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--production-leverage-potential-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--production-leverage-source-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--production-leverage-winsor-quantile",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--production-leverage-floor-fraction",
        type=float,
        default=1e-8,
    )
    parser.add_argument("--warm-start-ground-weight", type=float, default=1.0)
    parser.add_argument("--warm-start-teacher-weight", type=float, default=1.0)
    parser.add_argument("--warm-start-aux-weight", type=float, default=0.25)
    parser.add_argument("--warm-start-basis", type=str, default="aug-cc-pvdz")
    parser.add_argument("--warm-start-n-roots", type=int, default=8)
    parser.add_argument(
        "--krylov-teacher-svd-rtol",
        type=float,
        default=1e-4,
        help=(
            "Relative singular-value cutoff for the CASSCF source-Hamiltonian "
            "Krylov teacher snapshot."
        ),
    )
    parser.add_argument(
        "--krylov-teacher-svd-atol",
        type=float,
        default=1e-14,
        help=(
            "Absolute singular-value cutoff for the CASSCF source-Hamiltonian "
            "Krylov teacher snapshot."
        ),
    )
    parser.add_argument("--fixed-cas-ncas", type=int, default=0)
    parser.add_argument("--fixed-cas-gradient-weight", type=float, default=0.1)
    parser.add_argument("--fixed-cas-fd-step", type=float, default=1e-3)
    parser.add_argument("--fixed-cas-finetune-epochs", type=int, default=0)
    parser.add_argument("--fixed-cas-finetune-batch-size", type=int, default=1024)
    parser.add_argument(
        "--fixed-cas-finetune-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--fixed-cas-finetune-roots", type=int, default=3)
    parser.add_argument(
        "--fixed-cas-finetune-energy-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--fixed-cas-finetune-condition-weight",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--fixed-cas-finetune-overlap-weight",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--fixed-cas-finetune-max-condition",
        type=float,
        default=1e6,
    )
    parser.add_argument(
        "--fixed-cas-finetune-root-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum first Ritz root allowed when accepting fixed-block "
            "fine-tuning. This prevents accepting variational collapses."
        ),
    )
    parser.add_argument(
        "--fixed-cas-finetune-validation-fraction",
        type=float,
        default=0.25,
        help=(
            "Held-out fraction of CAS-dressed fine-tuning samples used for "
            "validation-matrix acceptance."
        ),
    )
    parser.add_argument(
        "--fixed-cas-finetune-bright-threshold",
        type=float,
        default=0.05,
        help=(
            "Relative baseline source-weight threshold for selecting tracked "
            "bright roots during CAS-dressed fine tuning."
        ),
    )
    parser.add_argument(
        "--fixed-cas-finetune-validation-blocks",
        type=int,
        default=4,
        help="Held-out validation blocks used to estimate acceptance SE.",
    )
    parser.add_argument(
        "--fixed-cas-finetune-acceptance-sigma",
        type=float,
        default=1.0,
        help="Required held-out improvement in combined block-SE units.",
    )
    parser.add_argument(
        "--fixed-cas-state-average",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use PySCF state-averaged CASSCF targets. Disabled by default "
            "because the robust production path uses spin-penalized CASCI roots."
        ),
    )
    parser.add_argument(
        "--cas-dressing-init-scale",
        type=float,
        default=0.0,
        help="Initial coefficient scale for the CAS dressing; 0 is identity.",
    )
    parser.add_argument(
        "--cas-dressing-visibility-weight",
        type=float,
        default=1e-3,
        help="Weight for the source-visibility log term in dressed fine-tuning.",
    )
    parser.add_argument(
        "--cas-dressing-regularizer-weight",
        type=float,
        default=1e-4,
        help="Weight for ||A-I||^2 + ell_A^2 ||grad A||^2.",
    )
    parser.add_argument(
        "--cas-dressing-gradient-regularizer-length",
        type=float,
        default=1.0,
        help="Length scale ell_A used by the dressing-gradient regularizer.",
    )
    parser.add_argument("--warm-start-min-source-overlap", type=float, default=1e-8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--hidden-double", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--determinants-per-head", type=int, default=1)
    parser.add_argument(
        "--response-orbital-radial-powers",
        type=int,
        default=OFFICIAL_RESPONSE_RADIAL_POWERS,
    )
    parser.add_argument(
        "--response-orbital-radial-scale",
        type=float,
        default=OFFICIAL_RESPONSE_RADIAL_SCALE,
    )
    parser.add_argument(
        "--response-source-envelope-core-decay",
        type=float,
        default=OFFICIAL_SOURCE_ENVELOPE_CORE_DECAY,
    )
    parser.add_argument(
        "--response-source-envelope-diffuse-decay",
        type=float,
        default=OFFICIAL_SOURCE_ENVELOPE_DIFFUSE_DECAY,
    )
    parser.add_argument(
        "--response-spatial-parity",
        choices=["none", "odd", "even"],
        default="none",
    )
    parser.add_argument(
        "--training-flow",
        choices=[
            "cas-dressed-teacher",
        ],
        default="cas-dressed-teacher",
    )
    parser.add_argument(
        "--residual-omegas",
        type=float,
        nargs="+",
        default=[0.34, 0.39, 0.44, 0.49, 0.56],
        help=(
            "Frequencies used only by the optional strong-residual audit when "
            "--strong-residual-audit-omegas is omitted."
        ),
    )
    parser.add_argument(
        "--residual-eta",
        type=float,
        default=0.02,
        help=("Lorentzian eta used only by the optional strong-residual audit."),
    )
    parser.add_argument(
        "--strong-residual-audit-samples",
        type=int,
        default=0,
        help=(
            "Run an expensive pointwise strong-residual audit on this many "
            "final validation samples. Disabled by default."
        ),
    )
    parser.add_argument(
        "--strong-residual-audit-omegas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Frequencies for the strong-residual audit. If omitted, reuse "
            "--residual-omegas."
        ),
    )
    parser.add_argument(
        "--strong-residual-audit-source-index",
        type=int,
        default=0,
        help="Source-block index audited by the pointwise strong residual.",
    )
    parser.add_argument(
        "--strong-residual-audit-batch-size",
        type=int,
        default=64,
        help="Batch size for Laplacian-based strong-residual diagnostics.",
    )
    parser.add_argument(
        "--aux-source-gaussian-exponents",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary fixed p_z exp(-alpha r^2) source probes included in "
            "the BF-NKSR basis while reading the physical dipole channel."
        ),
    )
    parser.add_argument(
        "--aux-source-dipole-radial-powers",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary local p-wave cusp-safe radial probes included in the "
            "BF-NKSR basis. Power p=0 is the leading cusp-lifted profile."
        ),
    )
    parser.add_argument(
        "--aux-source-dipole-radial-scale",
        type=float,
        default=1.0,
        help="Positive scale used by --aux-source-dipole-radial-powers.",
    )
    parser.add_argument(
        "--aux-source-atom-odd-gaussian-exponents",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary fixed two-center odd Gaussian source probes included "
            "in the BF-NKSR basis while reading the physical dipole channel."
        ),
    )
    parser.add_argument(
        "--aux-source-atom-odd-slater-decays",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary fixed two-center odd Slater source probes included "
            "in the BF-NKSR basis while reading the physical dipole channel."
        ),
    )
    parser.add_argument(
        "--aux-source-bond-odd-slater-decays",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary fixed two-center odd Slater orbital-ratio probes included "
            "in the BF-NKSR basis while reading the physical dipole channel."
        ),
    )
    parser.add_argument(
        "--aux-source-dipole-ee-scales",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Auxiliary correlated dipole probes source * mean r_ij/(s+r_ij) "
            "included in the BF-NKSR basis."
        ),
    )
    parser.add_argument(
        "--aux-source-bond-odd-ee-slater-decays",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Slater decays for correlated two-center odd probes; combined "
            "with --aux-source-bond-odd-ee-scales."
        ),
    )
    parser.add_argument(
        "--aux-source-bond-odd-ee-scales",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Electron-pair scales for correlated two-center odd probes; "
            "combined with --aux-source-bond-odd-ee-slater-decays."
        ),
    )
    parser.add_argument("--energy-samples", type=int, default=200000)
    parser.add_argument("--energy-walkers", type=int, default=4096)
    parser.add_argument("--energy-burn-in", type=int, default=500)
    parser.add_argument("--energy-steps-between", type=int, default=8)
    parser.add_argument("--energy-mcmc-width", type=float, default=1.0)
    parser.add_argument("--energy-batch-size", type=int, default=4096)
    parser.add_argument("--bright-min-weight", type=float, default=1e-4)
    parser.add_argument(
        "--final-sampling",
        choices=[
            OFFICIAL_FINAL_SAMPLING,
        ],
        default=OFFICIAL_FINAL_SAMPLING,
    )
    parser.add_argument("--final-samples", type=int, default=500000)
    parser.add_argument("--final-sobol-replicas", type=int, default=1)
    parser.add_argument("--final-bootstrap-replicates", type=int, default=200)
    parser.add_argument(
        "--final-diagnostic-roots",
        type=int,
        default=5,
        help=(
            "Number of source-bright roots to keep in final block, "
            "leave-one-out, and bootstrap diagnostics."
        ),
    )
    parser.add_argument(
        "--cas-raw-diagnostic-samples",
        type=int,
        default=0,
        help=(
            "If positive, assemble source+CAS-Krylov-teacher and source+NN-head "
            "weak matrices on this many final samples for subspace debugging."
        ),
    )
    parser.add_argument("--matrix-batch-size", type=int, default=8192)
    parser.add_argument("--eta", type=float, default=0.003)
    parser.add_argument("--grid-size", type=int, default=6001)
    parser.add_argument("--omega-min", type=float, default=0.25)
    parser.add_argument("--omega-max", type=float, default=0.85)
    parser.add_argument("--peak-min-height-fraction", type=float, default=0.01)
    parser.add_argument("--envelope-decay", type=float, default=0.22)
    parser.add_argument(
        "--response-envelope-decays",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Explicit response-head envelope decays. Provide n_heads values "
            "or n_heads*determinants_per_head values; overrides the linear "
            "initial-decay profile."
        ),
    )
    parser.add_argument("--initial-decay-min", type=float, default=0.16)
    parser.add_argument("--initial-decay-max", type=float, default=0.55)
    parser.add_argument("--eps-env", type=float, default=0.05)
    parser.add_argument("--ground-weight", type=float, default=1.0)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--head-weight", type=float, default=0.5)
    parser.add_argument("--overlap-cutoff", type=float, default=1e-8)
    parser.add_argument(
        "--cutoff-diagnostic-values",
        type=float,
        nargs="*",
        default=[
            1e-10,
            3e-10,
            1e-9,
            3e-9,
            1e-8,
            3e-8,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
        ],
        help=(
            "Whitening cutoffs audited on the final matrices. The active "
            "--overlap-cutoff is always included."
        ),
    )
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()
    if args.warm_start_heads is None:
        args.warm_start_heads = args.n_heads
    if not (0 <= args.warm_start_heads <= args.n_heads):
        msg = "--warm-start-heads must be between 0 and --n-heads"
        raise ValueError(msg)
    if args.warm_start_heads < 1:
        msg = "--warm-start requires at least one --warm-start-heads"
        raise ValueError(msg)
    if args.warm_start_samples < 1:
        msg = "--warm-start-samples must be positive"
        raise ValueError(msg)
    if args.warm_start_epochs < 0:
        msg = "--warm-start-epochs must be nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.cas_dressing_init_scale)
        and args.cas_dressing_init_scale >= 0.0
    ):
        msg = "--cas-dressing-init-scale must be finite and nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.cas_dressing_visibility_weight)
        and args.cas_dressing_visibility_weight >= 0.0
    ):
        msg = "--cas-dressing-visibility-weight must be finite and nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.cas_dressing_regularizer_weight)
        and args.cas_dressing_regularizer_weight >= 0.0
    ):
        msg = "--cas-dressing-regularizer-weight must be finite and nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.cas_dressing_gradient_regularizer_length)
        and args.cas_dressing_gradient_regularizer_length >= 0.0
    ):
        msg = (
            "--cas-dressing-gradient-regularizer-length must be finite and nonnegative"
        )
        raise ValueError(msg)
    if args.warm_start_batch_size < 1:
        msg = "--warm-start-batch-size must be positive"
        raise ValueError(msg)
    if not (
        np.isfinite(args.warm_start_learning_rate) and args.warm_start_learning_rate > 0
    ):
        msg = "--warm-start-learning-rate must be positive"
        raise ValueError(msg)
    if not (np.isfinite(args.warm_start_ridge) and args.warm_start_ridge >= 0):
        msg = "--warm-start-ridge must be finite and nonnegative"
        raise ValueError(msg)
    if args.production_leverage_candidate_factor < 1:
        msg = "--production-leverage-candidate-factor must be positive"
        raise ValueError(msg)
    if args.production_leverage_max_candidates < 0:
        msg = "--production-leverage-max-candidates must be nonnegative"
        raise ValueError(msg)
    leverage_weights = np.asarray(
        [
            args.production_leverage_gradient_weight,
            args.production_leverage_potential_weight,
            args.production_leverage_source_weight,
        ],
        dtype=np.float64,
    )
    if np.any(leverage_weights < 0.0) or not np.all(np.isfinite(leverage_weights)):
        msg = "--production-leverage-*weight values must be finite nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.production_leverage_winsor_quantile)
        and 0.0 < args.production_leverage_winsor_quantile <= 1.0
    ):
        msg = "--production-leverage-winsor-quantile must be in (0, 1]"
        raise ValueError(msg)
    if not (
        np.isfinite(args.production_leverage_floor_fraction)
        and args.production_leverage_floor_fraction >= 0.0
    ):
        msg = "--production-leverage-floor-fraction must be finite nonnegative"
        raise ValueError(msg)
    warm_start_component_weights = np.asarray(
        [
            args.warm_start_ground_weight,
            args.warm_start_teacher_weight,
            args.warm_start_aux_weight,
        ],
        dtype=np.float64,
    )
    if np.any(warm_start_component_weights < 0) or not np.all(
        np.isfinite(warm_start_component_weights)
    ):
        msg = "--warm-start-*weight values must be finite and nonnegative"
        raise ValueError(msg)
    if not np.any(warm_start_component_weights > 0):
        msg = "at least one warm-start sampling component weight must be positive"
        raise ValueError(msg)
    if args.warm_start_n_roots < 1:
        msg = "--warm-start-n-roots must be positive"
        raise ValueError(msg)
    if not (
        np.isfinite(args.krylov_teacher_svd_rtol) and args.krylov_teacher_svd_rtol >= 0
    ):
        msg = "--krylov-teacher-svd-rtol must be finite and nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.krylov_teacher_svd_atol) and args.krylov_teacher_svd_atol >= 0
    ):
        msg = "--krylov-teacher-svd-atol must be finite and nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.warm_start_min_source_overlap)
        and args.warm_start_min_source_overlap >= 0
    ):
        msg = "--warm-start-min-source-overlap must be finite and nonnegative"
        raise ValueError(msg)
    if args.fixed_cas_ncas < 0:
        msg = "--fixed-cas-ncas must be nonnegative"
        raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_gradient_weight)
        and args.fixed_cas_gradient_weight >= 0
    ):
        msg = "--fixed-cas-gradient-weight must be finite and nonnegative"
        raise ValueError(msg)
    if not (np.isfinite(args.fixed_cas_fd_step) and args.fixed_cas_fd_step > 0):
        msg = "--fixed-cas-fd-step must be positive"
        raise ValueError(msg)
    if args.fixed_cas_finetune_epochs < 0:
        msg = "--fixed-cas-finetune-epochs must be nonnegative"
        raise ValueError(msg)
    if args.fixed_cas_finetune_batch_size < 1:
        msg = "--fixed-cas-finetune-batch-size must be positive"
        raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_finetune_learning_rate)
        and args.fixed_cas_finetune_learning_rate > 0
    ):
        msg = "--fixed-cas-finetune-learning-rate must be positive"
        raise ValueError(msg)
    if args.fixed_cas_finetune_roots < 1:
        msg = "--fixed-cas-finetune-roots must be positive"
        raise ValueError(msg)
    for name in (
        "fixed_cas_finetune_energy_weight",
        "fixed_cas_finetune_condition_weight",
        "fixed_cas_finetune_overlap_weight",
    ):
        value = float(getattr(args, name))
        if not (np.isfinite(value) and value >= 0):
            msg = f"--{name.replace('_', '-')} must be finite and nonnegative"
            raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_finetune_max_condition)
        and args.fixed_cas_finetune_max_condition > 1
    ):
        msg = "--fixed-cas-finetune-max-condition must be finite and > 1"
        raise ValueError(msg)
    if not np.isfinite(args.fixed_cas_finetune_root_floor):
        msg = "--fixed-cas-finetune-root-floor must be finite"
        raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_finetune_validation_fraction)
        and 0.0 <= args.fixed_cas_finetune_validation_fraction < 1.0
    ):
        msg = "--fixed-cas-finetune-validation-fraction must be in [0, 1)"
        raise ValueError(msg)
    if (
        args.fixed_cas_finetune_epochs > 0
        and args.fixed_cas_finetune_validation_fraction <= 0.0
    ):
        msg = (
            "--fixed-cas-finetune-validation-fraction must be positive when "
            "--fixed-cas-finetune-epochs is positive"
        )
        raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_finetune_bright_threshold)
        and 0.0 <= args.fixed_cas_finetune_bright_threshold <= 1.0
    ):
        msg = "--fixed-cas-finetune-bright-threshold must be in [0, 1]"
        raise ValueError(msg)
    if args.fixed_cas_finetune_validation_blocks < 1:
        msg = "--fixed-cas-finetune-validation-blocks must be positive"
        raise ValueError(msg)
    if not (
        np.isfinite(args.fixed_cas_finetune_acceptance_sigma)
        and args.fixed_cas_finetune_acceptance_sigma >= 0.0
    ):
        msg = "--fixed-cas-finetune-acceptance-sigma must be nonnegative"
        raise ValueError(msg)
    validate_official_response_profile(args)
    response_ansatz_name = "cas_explicit_krylov_teacher_matrix_dressing"
    print(
        "response_official_workflow "
        f"name={OFFICIAL_RESPONSE_FLOW} "
        f"ansatz={response_ansatz_name} "
        f"cas_dressing_init_scale={args.cas_dressing_init_scale:.6g} "
        f"cas_dressing_visibility_weight="
        f"{args.cas_dressing_visibility_weight:.6g} "
        f"cas_dressing_regularizer_weight="
        f"{args.cas_dressing_regularizer_weight:.6g} "
        f"warm_start={args.warm_start} "
        f"warm_start_heads={args.warm_start_heads} "
        f"fixed_cas_basis={args.warm_start_basis} "
        f"fixed_cas_ncas={args.fixed_cas_ncas} "
        f"fixed_cas_roots={args.warm_start_n_roots} "
        f"krylov_svd_rtol={args.krylov_teacher_svd_rtol:.6g} "
        f"krylov_svd_atol={args.krylov_teacher_svd_atol:.6g} "
        "warm_start_sampling=cas_dressed_teacher_bright_influence_mixture "
        f"fixed_cas_gradient_weight={args.fixed_cas_gradient_weight:.6g} "
        f"fixed_cas_finetune_epochs={args.fixed_cas_finetune_epochs} "
        f"final_sampling={args.final_sampling} "
        f"source_envelope_core_decay="
        f"{args.response_source_envelope_core_decay:.6g} "
        f"source_envelope_diffuse_decay="
        f"{args.response_source_envelope_diffuse_decay:.6g} "
        f"dressing_radial_scales={list(OFFICIAL_DRESSING_RADIAL_SCALES)} "
        f"dressing_pair_scales={list(OFFICIAL_DRESSING_PAIR_SCALES)} "
        "closure=none"
    )
    if args.strong_residual_audit_samples < 0:
        msg = "--strong-residual-audit-samples must be nonnegative"
        raise ValueError(msg)
    if args.strong_residual_audit_source_index < 0:
        msg = "--strong-residual-audit-source-index must be nonnegative"
        raise ValueError(msg)
    if args.strong_residual_audit_batch_size < 1:
        msg = "--strong-residual-audit-batch-size must be positive"
        raise ValueError(msg)
    residual_omegas = np.asarray(args.residual_omegas, dtype=np.float64)
    if residual_omegas.size == 0 or not np.all(np.isfinite(residual_omegas)):
        msg = "--residual-omegas must contain finite diagnostic frequencies"
        raise ValueError(msg)
    aux_source_exponents = np.asarray(
        args.aux_source_gaussian_exponents, dtype=np.float64
    )
    aux_source_dipole_radial_powers = np.asarray(
        args.aux_source_dipole_radial_powers, dtype=np.float64
    )
    aux_source_atom_odd_exponents = np.asarray(
        args.aux_source_atom_odd_gaussian_exponents, dtype=np.float64
    )
    aux_source_atom_odd_slater_decays = np.asarray(
        args.aux_source_atom_odd_slater_decays, dtype=np.float64
    )
    aux_source_bond_odd_slater_decays = np.asarray(
        args.aux_source_bond_odd_slater_decays, dtype=np.float64
    )
    aux_source_dipole_ee_scales = np.asarray(
        args.aux_source_dipole_ee_scales, dtype=np.float64
    )
    aux_source_bond_odd_ee_slater_decays = np.asarray(
        args.aux_source_bond_odd_ee_slater_decays, dtype=np.float64
    )
    aux_source_bond_odd_ee_scales = np.asarray(
        args.aux_source_bond_odd_ee_scales, dtype=np.float64
    )
    if np.any(aux_source_exponents < 0) or not np.all(
        np.isfinite(aux_source_exponents)
    ):
        msg = "--aux-source-gaussian-exponents must be finite and non-negative"
        raise ValueError(msg)
    if np.any(aux_source_dipole_radial_powers < 0) or not np.all(
        np.isfinite(aux_source_dipole_radial_powers)
    ):
        msg = "--aux-source-dipole-radial-powers must be finite and non-negative"
        raise ValueError(msg)
    if aux_source_dipole_radial_powers.size and (
        not np.isfinite(args.aux_source_dipole_radial_scale)
        or args.aux_source_dipole_radial_scale <= 0
    ):
        msg = "--aux-source-dipole-radial-scale must be finite and positive"
        raise ValueError(msg)
    if np.any(aux_source_atom_odd_exponents < 0) or not np.all(
        np.isfinite(aux_source_atom_odd_exponents)
    ):
        msg = "--aux-source-atom-odd-gaussian-exponents must be finite and non-negative"
        raise ValueError(msg)
    if np.any(aux_source_atom_odd_slater_decays < 0) or not np.all(
        np.isfinite(aux_source_atom_odd_slater_decays)
    ):
        msg = "--aux-source-atom-odd-slater-decays must be finite and non-negative"
        raise ValueError(msg)
    if np.any(aux_source_bond_odd_slater_decays < 0) or not np.all(
        np.isfinite(aux_source_bond_odd_slater_decays)
    ):
        msg = "--aux-source-bond-odd-slater-decays must be finite and non-negative"
        raise ValueError(msg)
    if np.any(aux_source_dipole_ee_scales <= 0) or not np.all(
        np.isfinite(aux_source_dipole_ee_scales)
    ):
        msg = "--aux-source-dipole-ee-scales must be finite and positive"
        raise ValueError(msg)
    if np.any(aux_source_bond_odd_ee_slater_decays < 0) or not np.all(
        np.isfinite(aux_source_bond_odd_ee_slater_decays)
    ):
        msg = "--aux-source-bond-odd-ee-slater-decays must be finite and non-negative"
        raise ValueError(msg)
    if np.any(aux_source_bond_odd_ee_scales <= 0) or not np.all(
        np.isfinite(aux_source_bond_odd_ee_scales)
    ):
        msg = "--aux-source-bond-odd-ee-scales must be finite and positive"
        raise ValueError(msg)
    if bool(aux_source_bond_odd_ee_slater_decays.size) != bool(
        aux_source_bond_odd_ee_scales.size
    ):
        msg = (
            "--aux-source-bond-odd-ee-slater-decays and "
            "--aux-source-bond-odd-ee-scales must be provided together"
        )
        raise ValueError(msg)
    if args.response_orbital_radial_powers < 0:
        msg = "--response-orbital-radial-powers must be non-negative"
        raise ValueError(msg)
    if args.response_orbital_radial_powers and (
        not np.isfinite(args.response_orbital_radial_scale)
        or args.response_orbital_radial_scale <= 0
    ):
        msg = "--response-orbital-radial-scale must be positive when powers are used"
        raise ValueError(msg)
    response_envelope_decays = np.asarray(
        args.response_envelope_decays, dtype=np.float64
    )
    if response_envelope_decays.size and (
        np.any(response_envelope_decays <= 0)
        or not np.all(np.isfinite(response_envelope_decays))
    ):
        msg = "--response-envelope-decays must be finite and positive"
        raise ValueError(msg)
    cutoff_diagnostic_values = np.asarray(
        args.cutoff_diagnostic_values, dtype=np.float64
    )
    if cutoff_diagnostic_values.size and (
        np.any(cutoff_diagnostic_values <= 0)
        or not np.all(np.isfinite(cutoff_diagnostic_values))
    ):
        msg = "--cutoff-diagnostic-values must be finite and positive"
        raise ValueError(msg)
    cutoff_diagnostic_values = np.unique(
        np.concatenate(
            [
                cutoff_diagnostic_values.reshape(-1),
                np.asarray([args.overlap_cutoff], dtype=np.float64),
            ]
        )
    )
    if args.final_diagnostic_roots < 0:
        msg = "--final-diagnostic-roots must be nonnegative"
        raise ValueError(msg)
    if args.cas_raw_diagnostic_samples < 0:
        msg = "--cas-raw-diagnostic-samples must be nonnegative"
        raise ValueError(msg)
    ground = load_ferminet_ground(args.checkpoint, ground_energy=args.ground_energy)
    if (
        aux_source_atom_odd_exponents.size
        or aux_source_atom_odd_slater_decays.size
        or aux_source_bond_odd_slater_decays.size
        or aux_source_bond_odd_ee_slater_decays.size
    ) and ground.atoms.shape[0] != 2:
        msg = "two-center auxiliary sources require exactly two atoms"
        raise ValueError(msg)
    if (
        aux_source_dipole_ee_scales.size or aux_source_bond_odd_ee_slater_decays.size
    ) and ground.electron_shape[0] < 2:
        msg = "electron-pair auxiliary sources require at least two electrons"
        raise ValueError(msg)
    key = jax.random.PRNGKey(args.seed)
    init_key, energy_key = jax.random.split(key, 2)
    energy_stderr = np.nan
    energy_pmove = np.nan
    if np.isnan(ground.energy):
        energy, energy_stderr, energy_pmove = estimate_ground_energy(
            ground,
            key=energy_key,
            n_samples=args.energy_samples,
            walkers=args.energy_walkers,
            burn_in=args.energy_burn_in,
            steps_between=args.energy_steps_between,
            width=args.energy_mcmc_width,
            batch_size=args.energy_batch_size,
            envelope_decay=args.envelope_decay,
        )
        ground = replace(ground, energy=energy)
        print(
            "ground_energy_evaluation "
            f"samples={args.energy_samples} energy_ha={energy:.10f} "
            f"stderr_ha={energy_stderr:.3e} pmove~{energy_pmove:.3f}"
        )
    params: Params | None = None
    warm_start_heads = 0
    warm_start_loss = float("nan")
    warm_start_target_count = 0
    warm_start_backend = ""
    warm_start_qc_energies = np.asarray([], dtype=np.float64)
    warm_start_qc_source_overlaps = np.asarray([], dtype=np.float64)
    warm_start_root_energies = np.asarray([], dtype=np.float64)
    warm_start_root_source_overlaps = np.asarray([], dtype=np.float64)
    warm_start_krylov_singular_values = np.asarray([], dtype=np.float64)
    warm_start_krylov_coefficients = np.asarray([], dtype=np.float64)
    warm_start_krylov_source_moments = np.asarray([], dtype=np.float64)
    warm_start_sampling_stats: dict[str, Any] = {
        "sampler": "not-run",
        "pmove": float("nan"),
        "density_log_shift": 0.0,
        "proposal_samples": 0,
        "proposal_ess": float("nan"),
        "proposal_ess_fraction": float("nan"),
        "proposal_max_weight_fraction": float("nan"),
        "resampling_unique_fraction": float("nan"),
        "proposal_component_count": 0,
        "ground_norm": float("nan"),
        "source_norm": float("nan"),
        "teacher_norms": np.asarray([], dtype=np.float64),
        "dressed_norms": np.asarray([], dtype=np.float64),
        "ground_weight": float(args.warm_start_ground_weight),
        "source_weight": float("nan"),
        "teacher_weight": float(args.warm_start_teacher_weight),
        "dressed_weight": float("nan"),
        "aux_weight": float(args.warm_start_aux_weight),
        "walkers": 0,
        "burn_in": 0,
        "steps_between": 1,
        "width": float("nan"),
    }
    fine_tune_loss = float("nan")
    fine_tune_stats: dict[str, float] = {
        "loss": float("nan"),
        "energy_loss": float("nan"),
        "anchor_loss": float("nan"),
        "visibility_loss": float("nan"),
        "regularizer": float("nan"),
        "dressing_norm": float("nan"),
        "dressing_grad_norm": float("nan"),
        "condition_penalty": float("nan"),
        "overlap_penalty": float("nan"),
        "root0": float("nan"),
        "source_weighted_root": float("nan"),
        "source_weight_sum": float("nan"),
        "source_weight0": float("nan"),
        "baseline_root0": float("nan"),
        "baseline_source_weighted_root": float("nan"),
        "baseline_source_weight_sum": float("nan"),
        "condition": float("nan"),
        "validation_root0": float("nan"),
        "validation_source_weighted_root": float("nan"),
        "validation_source_weight_sum": float("nan"),
        "validation_condition": float("nan"),
        "validation_se": float("nan"),
        "validation_candidate_se": float("nan"),
        "validation_pair_delta": float("nan"),
        "validation_pair_se": float("nan"),
        "validation_source_pair_delta": float("nan"),
        "validation_source_pair_se": float("nan"),
        "last_validation_pair_delta": float("nan"),
        "last_validation_pair_se": float("nan"),
        "last_validation_source_pair_delta": float("nan"),
        "last_validation_source_pair_se": float("nan"),
        "baseline_validation_root0": float("nan"),
        "baseline_validation_source_weighted_root": float("nan"),
        "baseline_validation_source_weight_sum": float("nan"),
        "baseline_validation_condition": float("nan"),
        "baseline_validation_se": float("nan"),
        "anchor_coeff_norm": float("nan"),
        "accepted": 0.0,
        "accepted_epoch": float("nan"),
    }
    cas_dressing_training = bool(args.fixed_cas_finetune_epochs > 0)
    teacher_model = build_casscf_krylov_teacher_model(
        ground,
        basis=args.warm_start_basis,
        n_targets=args.warm_start_heads,
        n_roots=args.warm_start_n_roots,
        ncas=args.fixed_cas_ncas,
        source_axis=2,
        state_average=args.fixed_cas_state_average,
        svd_rtol=args.krylov_teacher_svd_rtol,
        svd_atol=args.krylov_teacher_svd_atol,
    )
    (
        warm_start_qc_energies,
        warm_start_qc_source_overlaps,
        warm_start_root_energies,
        warm_start_root_source_overlaps,
        warm_start_krylov_singular_values,
        warm_start_krylov_coefficients,
    ) = krylov_teacher_model_metadata(teacher_model)
    warm_start_krylov_source_moments = retained_krylov_source_moments(
        teacher_model,
        max_order=2,
    )
    warm_start_backend = teacher_model.seed_model.method
    warm_start_target_count = int(warm_start_krylov_coefficients.shape[1])
    params = init_cas_dressing_params(
        init_key,
        teacher_count=warm_start_target_count,
        atom_count=int(ground.atoms.shape[0]),
        radial_scale_count=len(OFFICIAL_DRESSING_RADIAL_SCALES),
        pair_scale_count=len(OFFICIAL_DRESSING_PAIR_SCALES),
        hidden=args.hidden,
        layers=args.layers,
        init_scale=args.cas_dressing_init_scale,
    )
    if not cas_dressing_training:
        warm_start_loss = 0.0
        print(
            "response_cas_dressed_teacher_ready "
            f"backend={warm_start_backend} "
            "neural_training=skipped "
            f"requested_heads={args.warm_start_heads} "
            f"targets={warm_start_target_count} "
            f"basis={args.warm_start_basis} "
            f"ncas={args.fixed_cas_ncas} "
            f"root_window={args.warm_start_n_roots} "
            f"krylov_rank={warm_start_target_count} "
            f"krylov_svd_rtol={args.krylov_teacher_svd_rtol:.3e} "
            f"krylov_svd_atol={args.krylov_teacher_svd_atol:.3e} "
            f"krylov_singular_values={list(warm_start_krylov_singular_values)}"
        )
    else:
        warm_points, warm_density, warm_start_sampling_stats = (
            sample_cas_dressed_teacher_bright_influence_distribution(
                params,
                ground,
                teacher_model,
                head_count=warm_start_target_count,
                n_samples=args.warm_start_samples,
                core_decay=args.response_source_envelope_core_decay,
                diffuse_decay=args.response_source_envelope_diffuse_decay,
                batch_size=args.warm_start_batch_size,
                seed=args.seed + 424_243,
                basis=args.warm_start_basis,
                finite_difference_step=args.fixed_cas_fd_step,
                candidate_factor=args.production_leverage_candidate_factor,
                max_candidate_samples=args.production_leverage_max_candidates,
                gradient_weight=args.production_leverage_gradient_weight,
                potential_weight=args.production_leverage_potential_weight,
                source_weight=args.production_leverage_source_weight,
                winsor_quantile=args.production_leverage_winsor_quantile,
                floor_fraction=args.production_leverage_floor_fraction,
            )
        )
        qc_targets = evaluate_casscf_krylov_teacher_targets(
            teacher_model,
            warm_points,
            basis=args.warm_start_basis,
            gradients=True,
            finite_difference_step=args.fixed_cas_fd_step,
        )
        if qc_targets.gradients is None:
            msg = "Krylov-CASSCF neural warm start requires target gradients"
            raise RuntimeError(msg)
        warm_start_backend = qc_targets.backend
        warm_start_qc_energies = qc_targets.excitation_energies
        warm_start_qc_source_overlaps = qc_targets.source_overlaps
        warm_start_root_energies = (
            np.asarray([], dtype=np.float64)
            if qc_targets.root_energies is None
            else qc_targets.root_energies
        )
        warm_start_root_source_overlaps = (
            np.asarray([], dtype=np.float64)
            if qc_targets.root_source_overlaps is None
            else qc_targets.root_source_overlaps
        )
        warm_start_krylov_singular_values = (
            np.asarray([], dtype=np.float64)
            if qc_targets.krylov_singular_values is None
            else qc_targets.krylov_singular_values
        )
        warm_start_krylov_coefficients = (
            np.asarray([], dtype=np.float64)
            if qc_targets.krylov_coefficients is None
            else qc_targets.krylov_coefficients
        )
        target_values = qc_targets.values
        target_gradients = qc_targets.gradients
        warm_start_target_count = int(target_values.shape[1])
        print(
            "response_cas_dressed_teacher_start "
            f"backend={warm_start_backend} "
            f"samples={warm_points.shape[0]} "
            "sampling=cas_dressed_teacher_bright_influence_mixture "
            f"sampling_pmove={warm_start_sampling_stats['pmove']:.3f} "
            "sampling_ess_frac="
            f"{warm_start_sampling_stats['proposal_ess_fraction']:.3f} "
            "sampling_unique_frac="
            f"{warm_start_sampling_stats['resampling_unique_fraction']:.3f} "
            f"sampling_walkers={int(warm_start_sampling_stats['walkers'])} "
            f"requested_heads={args.warm_start_heads} "
            f"targets={warm_start_target_count} "
            f"epochs={args.fixed_cas_finetune_epochs} "
            f"basis={args.warm_start_basis} "
            f"ncas={args.fixed_cas_ncas} "
            f"root_window={args.warm_start_n_roots} "
            f"krylov_rank={warm_start_target_count} "
            f"krylov_svd_rtol={args.krylov_teacher_svd_rtol:.3e} "
            f"krylov_svd_atol={args.krylov_teacher_svd_atol:.3e} "
            f"gradient_weight={args.fixed_cas_gradient_weight:.3e} "
            f"fd_step={args.fixed_cas_fd_step:.3e} "
            f"ground_norm={float(warm_start_sampling_stats['ground_norm']):.3e} "
            f"krylov_singular_values={list(warm_start_krylov_singular_values)}"
        )
        warm_start_loss = 0.0
        params, fine_tune_stats = fine_tune_cas_dressed_teacher_block(
            params,
            ground,
            warm_points,
            warm_density,
            target_values,
            target_gradients,
            head_count=warm_start_target_count,
            epochs=args.fixed_cas_finetune_epochs,
            batch_size=args.fixed_cas_finetune_batch_size,
            learning_rate=args.fixed_cas_finetune_learning_rate,
            n_roots=args.fixed_cas_finetune_roots,
            energy_weight=args.fixed_cas_finetune_energy_weight,
            visibility_weight=args.cas_dressing_visibility_weight,
            condition_weight=args.fixed_cas_finetune_condition_weight,
            overlap_weight=args.fixed_cas_finetune_overlap_weight,
            regularizer_weight=args.cas_dressing_regularizer_weight,
            gradient_regularizer_length=(args.cas_dressing_gradient_regularizer_length),
            max_condition=args.fixed_cas_finetune_max_condition,
            validation_fraction=args.fixed_cas_finetune_validation_fraction,
            bright_threshold=args.fixed_cas_finetune_bright_threshold,
            validation_blocks=args.fixed_cas_finetune_validation_blocks,
            acceptance_sigma=args.fixed_cas_finetune_acceptance_sigma,
            seed=args.seed + 29_791,
            log_every=args.log_every,
        )
        fine_tune_loss = float(fine_tune_stats["loss"])
    warm_start_heads = int(warm_start_target_count)
    active_heads = int(warm_start_heads)
    external_basis_blocks: tuple[ExternalCASBasisBlock, ...] = ()
    print(f"response_cas_dressed_teacher_block_ready active_heads={active_heads}")
    final_block_overlaps = None
    final_block_hamiltonians = None
    final_block_sources = None
    final_block_counts = None
    final_block_raw_overlaps = None
    final_block_raw_hamiltonians = None
    final_block_projection_numerators = None
    final_block_projection_norms = None
    final_block_ground_hamiltonians = None
    final_block_ground_hamiltonian_norms = None
    final_density = None
    production_certified_final = (
        bool(cas_dressing_training)
        and "certified_overlap" in fine_tune_stats
        and "certified_hamiltonian" in fine_tune_stats
        and "certified_source" in fine_tune_stats
    )
    if (
        args.response_source_envelope_core_decay <= 0
        or args.response_source_envelope_diffuse_decay <= 0
    ):
        msg = (
            "official final sampling requires positive "
            "--response-source-envelope-core-decay and "
            "--response-source-envelope-diffuse-decay"
        )
        raise ValueError(msg)
    replicas = max(1, int(args.final_sobol_replicas))
    sample_counts = [args.final_samples // replicas] * replicas
    for idx in range(args.final_samples % replicas):
        sample_counts[idx] += 1
    sample_pieces = []
    density_pieces = []
    final_sampling_stats: list[dict[str, Any]] = []
    if params is None:
        msg = "CAS-dressed teacher parameters were not initialized"
        raise RuntimeError(msg)
    if production_certified_final:
        sample_pieces = [
            np.asarray(block)
            for block in fine_tune_stats["certified_point_blocks"]
            if np.asarray(block).shape[0] > 0
        ]
        density_pieces = [
            np.asarray(block, dtype=np.float64)
            for block in fine_tune_stats["certified_density_blocks"]
            if np.asarray(block).shape[0] > 0
        ]
        final_sampling_stats.append(
            {
                **warm_start_sampling_stats,
                "sampler": ("production_certified_bright_influence_validation_pool"),
            }
        )
    else:
        for replica_idx, replica_samples in enumerate(sample_counts):
            if replica_samples <= 0:
                continue
            replica_points, replica_density, replica_stats = (
                sample_cas_dressed_teacher_bright_influence_distribution(
                    params,
                    ground,
                    teacher_model,
                    head_count=active_heads,
                    n_samples=replica_samples,
                    core_decay=args.response_source_envelope_core_decay,
                    diffuse_decay=args.response_source_envelope_diffuse_decay,
                    batch_size=args.matrix_batch_size,
                    seed=args.seed + 7919 + 104729 * replica_idx,
                    basis=args.warm_start_basis,
                    finite_difference_step=args.fixed_cas_fd_step,
                    candidate_factor=args.production_leverage_candidate_factor,
                    max_candidate_samples=args.production_leverage_max_candidates,
                    gradient_weight=args.production_leverage_gradient_weight,
                    potential_weight=args.production_leverage_potential_weight,
                    source_weight=args.production_leverage_source_weight,
                    winsor_quantile=args.production_leverage_winsor_quantile,
                    floor_fraction=args.production_leverage_floor_fraction,
                )
            )
            sample_pieces.append(replica_points)
            density_pieces.append(np.asarray(replica_density))
            final_sampling_stats.append(replica_stats)
    if not sample_pieces:
        msg = "--final-samples must be positive for official production sampling"
        raise ValueError(msg)
    if production_certified_final:
        final_density_log_shift = float(
            warm_start_sampling_stats.get("density_log_shift", 0.0)
        )
        final_density_log_shifts = np.full(
            (len(density_pieces),),
            final_density_log_shift,
            dtype=np.float64,
        )
    else:
        density_pieces, final_density_log_shift, final_density_log_shifts = (
            rescale_density_pieces_to_common_log_shift(
                density_pieces,
                final_sampling_stats,
            )
        )
    samples = np.concatenate(sample_pieces, axis=0)
    final_density = np.concatenate(density_pieces, axis=0)
    final_aux_source_count = _auxiliary_source_count(
        aux_source_exponents,
        aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales,
        aux_source_dipole_radial_powers,
    )
    if final_aux_source_count:
        msg = "CAS-dressed final basis does not support auxiliary sources"
        raise ValueError(msg)
    if production_certified_final:
        overlap = np.asarray(fine_tune_stats["certified_overlap"], dtype=np.float64)
        hamiltonian = np.asarray(
            fine_tune_stats["certified_hamiltonian"],
            dtype=np.float64,
        )
        source = np.asarray(fine_tune_stats["certified_source"], dtype=np.float64)
        final_block_overlaps = np.asarray(
            fine_tune_stats["certified_block_overlaps"],
            dtype=np.float64,
        )
        final_block_hamiltonians = np.asarray(
            fine_tune_stats["certified_block_hamiltonians"],
            dtype=np.float64,
        )
        final_block_sources = np.asarray(
            fine_tune_stats["certified_block_sources"],
            dtype=np.float64,
        )
        final_block_counts = np.asarray(
            fine_tune_stats["certified_block_counts"],
            dtype=np.float64,
        )
    else:
        value_blocks, gradient_blocks = cas_dressed_teacher_basis_value_gradient_blocks(
            params,
            ground,
            teacher_model,
            sample_pieces,
            head_count=active_heads,
            basis=args.warm_start_basis,
            finite_difference_step=args.fixed_cas_fd_step,
            batch_size=args.matrix_batch_size,
        )
        (
            overlap,
            hamiltonian,
            source,
            final_block_overlaps,
            final_block_hamiltonians,
            final_block_sources,
            final_block_counts,
            final_block_raw_overlaps,
            final_block_raw_hamiltonians,
            final_block_projection_numerators,
            final_block_projection_norms,
            final_block_ground_hamiltonians,
            final_block_ground_hamiltonian_norms,
        ) = retained_weak_matrix_blocks_from_precomputed_values(
            ground,
            sample_pieces,
            density_pieces,
            value_blocks,
            gradient_blocks,
            batch_size=args.matrix_batch_size,
            return_block_statistics=True,
        )
    final_selected_metrics = _source_weighted_projected_objective(
        overlap,
        hamiltonian,
        source,
        overlap_cutoff=args.overlap_cutoff,
        root_floor=args.fixed_cas_finetune_root_floor,
        min_weight=args.bright_min_weight,
        max_roots=args.fixed_cas_finetune_roots,
    )
    pmove = (
        float(np.mean([stats["pmove"] for stats in final_sampling_stats]))
        if final_sampling_stats
        else np.nan
    )
    final_sampling_ess_fraction = (
        float(
            np.mean([stats["proposal_ess_fraction"] for stats in final_sampling_stats])
        )
        if final_sampling_stats
        else np.nan
    )
    final_sampling_unique_fraction = (
        float(
            np.mean(
                [stats["resampling_unique_fraction"] for stats in final_sampling_stats]
            )
        )
        if final_sampling_stats
        else np.nan
    )
    final_sampling_max_weight_fraction = (
        float(
            np.max(
                [
                    stats["proposal_max_weight_fraction"]
                    for stats in final_sampling_stats
                ]
            )
        )
        if final_sampling_stats
        else np.nan
    )
    final_sampling_sampler = (
        ",".join(sorted({str(stats["sampler"]) for stats in final_sampling_stats}))
        if final_sampling_stats
        else ""
    )
    final_sampling_component_weights = (
        np.mean(
            np.stack(
                [
                    np.asarray(stats["leverage_component_weights"], dtype=np.float64)
                    for stats in final_sampling_stats
                    if "leverage_component_weights" in stats
                ],
                axis=0,
            ),
            axis=0,
        )
        if any("leverage_component_weights" in stats for stats in final_sampling_stats)
        else np.full((5,), np.nan, dtype=np.float64)
    )
    final_sampling_leverage_normalizer = (
        float(
            np.mean(
                [
                    stats["leverage_normalizer"]
                    for stats in final_sampling_stats
                    if "leverage_normalizer" in stats
                ]
            )
        )
        if any("leverage_normalizer" in stats for stats in final_sampling_stats)
        else np.nan
    )
    final_sampling_leverage_mean = (
        float(
            np.mean(
                [
                    stats["leverage_mean"]
                    for stats in final_sampling_stats
                    if "leverage_mean" in stats
                ]
            )
        )
        if any("leverage_mean" in stats for stats in final_sampling_stats)
        else np.nan
    )
    final_sampling_leverage_max = (
        float(
            np.max(
                [
                    stats["leverage_max"]
                    for stats in final_sampling_stats
                    if "leverage_max" in stats
                ]
            )
        )
        if any("leverage_max" in stats for stats in final_sampling_stats)
        else np.nan
    )
    final_sampling_leverage_winsor_limit = (
        float(
            np.mean(
                [
                    stats["leverage_winsor_limit"]
                    for stats in final_sampling_stats
                    if "leverage_winsor_limit" in stats
                ]
            )
        )
        if any("leverage_winsor_limit" in stats for stats in final_sampling_stats)
        else np.nan
    )
    print(
        "response_cas_dressed_teacher_final_basis "
        f"production_certified={production_certified_final} "
        f"active_heads={active_heads} "
        f"source_objective={final_selected_metrics['objective']:.10f} "
        f"bright_root0={final_selected_metrics['bright_root0']:.10f} "
        f"bright_weight0={final_selected_metrics['bright_weight0']:.3e} "
        f"condition={final_selected_metrics['condition']:.3e} "
        f"final_samples={samples.shape[0]} "
        f"sampling_ess_frac={final_sampling_ess_fraction:.3f} "
        f"sampling_unique_frac={final_sampling_unique_fraction:.3f}"
    )
    trained_active_heads = active_heads
    external_basis_dim = external_cas_basis_count(external_basis_blocks)
    active_heads, overlap, hamiltonian, source, spectrum = (
        projected_spectrum_with_head_fallback(
            overlap,
            hamiltonian,
            source,
            active_heads=active_heads,
            aux_source_count=final_aux_source_count,
            external_basis_count=external_basis_dim,
            source_in_basis=False,
            overlap_cutoff=args.overlap_cutoff,
        )
    )
    final_moments = moment_diagnostics(
        overlap,
        hamiltonian,
        source,
        spectrum,
        overlap_cutoff=args.overlap_cutoff,
        source_in_basis=False,
    )
    cutoff_diagnostics = cutoff_sensitivity_diagnostics(
        overlap,
        hamiltonian,
        source,
        cutoffs=cutoff_diagnostic_values,
        root_floor=0.0,
        min_weight=args.bright_min_weight,
        max_roots=args.final_diagnostic_roots,
        source_in_basis=False,
    )
    matrix_debug_cutoffs = np.asarray([], dtype=np.float64)
    cas_raw_debug_overlap = np.asarray([], dtype=np.float64)
    cas_raw_debug_hamiltonian = np.asarray([], dtype=np.float64)
    cas_raw_debug_source = np.asarray([], dtype=np.float64)
    cas_raw_debug_qc_energies = np.asarray([], dtype=np.float64)
    cas_raw_debug_qc_source_overlaps = np.asarray([], dtype=np.float64)
    cas_raw_debug_retained = np.asarray([], dtype=np.int64)
    cas_raw_debug_condition = np.asarray([], dtype=np.float64)
    cas_raw_debug_bright_roots = np.asarray([], dtype=np.float64)
    cas_raw_debug_bright_norm_weights = np.asarray([], dtype=np.float64)
    cas_dressed_debug_overlap = np.asarray([], dtype=np.float64)
    cas_dressed_debug_hamiltonian = np.asarray([], dtype=np.float64)
    cas_dressed_debug_source = np.asarray([], dtype=np.float64)
    cas_dressed_debug_retained = np.asarray([], dtype=np.int64)
    cas_dressed_debug_condition = np.asarray([], dtype=np.float64)
    cas_dressed_debug_bright_roots = np.asarray([], dtype=np.float64)
    cas_dressed_debug_bright_norm_weights = np.asarray([], dtype=np.float64)
    cas_raw_debug_sample_count = 0
    if args.cas_raw_diagnostic_samples:
        debug_point_blocks, debug_density_blocks = _limit_matrix_blocks(
            sample_pieces,
            density_pieces,
            int(args.cas_raw_diagnostic_samples),
        )
        cas_raw_debug_sample_count = int(
            sum(block.shape[0] for block in debug_point_blocks)
        )
        if cas_raw_debug_sample_count:
            debug_points = np.concatenate(debug_point_blocks, axis=0)
            debug_qc_targets = evaluate_casscf_krylov_teacher_targets(
                teacher_model,
                debug_points,
                basis=args.warm_start_basis,
                gradients=True,
                finite_difference_step=args.fixed_cas_fd_step,
            )
            if debug_qc_targets.gradients is None:
                msg = "CAS Krylov-teacher matrix diagnostic requires target gradients"
                raise RuntimeError(msg)
            debug_cas_values = debug_qc_targets.values[:, :active_heads]
            debug_cas_gradients = debug_qc_targets.gradients[:, :active_heads]
            debug_cas_source_overlaps = debug_qc_targets.source_overlaps[:active_heads]
            cas_value_blocks = []
            cas_gradient_blocks = []
            offset = 0
            for block in debug_point_blocks:
                count = int(block.shape[0])
                source_values_block, source_gradients_block = (
                    source_values_and_gradients(
                        ground,
                        jnp.asarray(block),
                    )
                )
                cas_values_block = debug_cas_values[offset : offset + count]
                cas_gradients_block = debug_cas_gradients[offset : offset + count]
                cas_value_blocks.append(
                    np.concatenate(
                        [
                            np.asarray(source_values_block)[:, None],
                            np.asarray(cas_values_block),
                        ],
                        axis=1,
                    )
                )
                cas_gradient_blocks.append(
                    np.concatenate(
                        [
                            np.asarray(source_gradients_block)[:, None, :, :],
                            np.asarray(cas_gradients_block),
                        ],
                        axis=1,
                    )
                )
                offset += count
            (
                cas_raw_debug_overlap,
                cas_raw_debug_hamiltonian,
                cas_raw_debug_source,
            ) = retained_weak_matrix_blocks_from_precomputed_values(
                ground,
                debug_point_blocks,
                debug_density_blocks,
                cas_value_blocks,
                cas_gradient_blocks,
                batch_size=args.matrix_batch_size,
            )
            dressed_debug_values, dressed_debug_gradients = (
                cas_dressed_teacher_basis_value_gradient_blocks(
                    params,
                    ground,
                    teacher_model,
                    debug_point_blocks,
                    head_count=active_heads,
                    basis=args.warm_start_basis,
                    finite_difference_step=args.fixed_cas_fd_step,
                    batch_size=args.matrix_batch_size,
                )
            )
            (
                cas_dressed_debug_overlap,
                cas_dressed_debug_hamiltonian,
                cas_dressed_debug_source,
            ) = retained_weak_matrix_blocks_from_precomputed_values(
                ground,
                debug_point_blocks,
                debug_density_blocks,
                dressed_debug_values,
                dressed_debug_gradients,
                batch_size=args.matrix_batch_size,
            )
            matrix_debug_cutoffs = np.unique(
                np.concatenate(
                    [
                        cutoff_diagnostic_values.reshape(-1),
                        np.asarray([1e-6, 1e-5], dtype=np.float64),
                    ]
                )
            )
            cas_raw_summary = _matrix_summary_for_cutoffs(
                cas_raw_debug_overlap,
                cas_raw_debug_hamiltonian,
                cas_raw_debug_source,
                cutoffs=matrix_debug_cutoffs,
                min_weight=args.bright_min_weight,
                max_roots=args.final_diagnostic_roots,
                source_in_basis=False,
            )
            cas_dressed_summary = _matrix_summary_for_cutoffs(
                cas_dressed_debug_overlap,
                cas_dressed_debug_hamiltonian,
                cas_dressed_debug_source,
                cutoffs=matrix_debug_cutoffs,
                min_weight=args.bright_min_weight,
                max_roots=args.final_diagnostic_roots,
                source_in_basis=False,
            )
            _print_matrix_debug_summary("cas_krylov_teacher", cas_raw_summary)
            _print_matrix_debug_summary("cas_dressed_teacher", cas_dressed_summary)
            cas_raw_debug_qc_energies = debug_qc_targets.excitation_energies
            cas_raw_debug_qc_source_overlaps = debug_cas_source_overlaps
            cas_raw_debug_retained = cas_raw_summary["retained"]
            cas_raw_debug_condition = cas_raw_summary["condition"]
            cas_raw_debug_bright_roots = cas_raw_summary["bright_roots"]
            cas_raw_debug_bright_norm_weights = cas_raw_summary["bright_norm_weights"]
            cas_dressed_debug_retained = cas_dressed_summary["retained"]
            cas_dressed_debug_condition = cas_dressed_summary["condition"]
            cas_dressed_debug_bright_roots = cas_dressed_summary["bright_roots"]
            cas_dressed_debug_bright_norm_weights = cas_dressed_summary[
                "bright_norm_weights"
            ]
    final_replica = final_replica_pole_diagnostics(
        final_block_overlaps,
        final_block_hamiltonians,
        final_block_sources,
        final_block_counts,
        block_raw_overlaps=final_block_raw_overlaps,
        block_raw_hamiltonians=final_block_raw_hamiltonians,
        block_projection_numerators=final_block_projection_numerators,
        block_projection_norms=final_block_projection_norms,
        block_ground_hamiltonians=final_block_ground_hamiltonians,
        block_ground_hamiltonian_norms=final_block_ground_hamiltonian_norms,
        retained_heads=active_heads,
        aux_source_count=final_aux_source_count,
        external_basis_count=external_basis_dim,
        source_in_basis=False,
        overlap_cutoff=args.overlap_cutoff,
        root_floor=0.0,
        min_weight=args.bright_min_weight,
        max_roots=args.final_diagnostic_roots,
        bootstrap_replicates=args.final_bootstrap_replicates,
        bootstrap_seed=args.seed + 1_000_003,
    )
    omega = np.linspace(args.omega_min, args.omega_max, args.grid_size)
    intensity = lorentzian_spectrum(
        omega, spectrum.excitation_energies, spectrum.weights[:, 0, 0], args.eta
    )
    peaks = find_spectrum_peaks(
        omega,
        intensity,
        min_height_fraction=args.peak_min_height_fraction,
        max_peaks=active_heads + 1 + final_aux_source_count + external_basis_dim,
    )
    strong_residual = {
        "strong_residual_omegas": np.asarray([], dtype=np.float64),
        "strong_residual_epsilon": np.asarray([], dtype=np.float64),
        "strong_residual_epsilon_over_eta": np.asarray([], dtype=np.float64),
        "strong_residual_norm": np.asarray([], dtype=np.float64),
        "strong_residual_source_norm": np.nan,
        "strong_residual_eta": np.nan,
        "strong_residual_source_index": int(args.strong_residual_audit_source_index),
        "strong_residual_samples": 0,
        "strong_residual_max_epsilon": np.nan,
        "strong_residual_max_epsilon_over_eta": np.nan,
        "strong_residual_density_nonfinite_count": 0,
        "strong_residual_density_nonpositive_count": 0,
        "strong_residual_density_min_positive": np.nan,
        "strong_residual_density_max_finite": np.nan,
        "strong_residual_overlap_nonfinite_count": 0,
        "strong_residual_hamiltonian_nonfinite_count": 0,
        "strong_residual_coeff_nonfinite_count": 0,
        "strong_residual_system_condition": np.asarray([], dtype=np.float64),
        "strong_residual_system_condition_max": np.nan,
        "strong_residual_value_nonfinite_count": 0,
        "strong_residual_laplacian_nonfinite_count": 0,
        "strong_residual_hbar_nonfinite_count": 0,
        "strong_residual_point_residual_nonfinite_count": 0,
        "strong_residual_residual_contrib_nonfinite_count": 0,
        "strong_residual_source_contrib_nonfinite_count": 0,
        "strong_residual_region_labels": np.asarray([], dtype="<U1"),
        "strong_residual_region_counts": np.asarray([], dtype=np.int64),
        "strong_residual_region_source_norm": np.asarray([], dtype=np.float64),
        "strong_residual_region_residual_norm": np.asarray([], dtype=np.float64),
        "strong_residual_region_epsilon": np.asarray([], dtype=np.float64),
        "strong_residual_region_epsilon_over_eta": np.asarray([], dtype=np.float64),
        "strong_residual_region_residual_fraction": np.asarray([], dtype=np.float64),
        "strong_residual_region_node_abs_threshold": np.nan,
        "strong_residual_region_en_cusp_threshold": np.nan,
        "strong_residual_region_ee_cusp_threshold": np.nan,
        "strong_residual_region_tail_radius_threshold": np.nan,
        "strong_residual_region_high_action_threshold": np.nan,
        "strong_residual_region_en_cutoff_shell_width": np.nan,
        "strong_residual_region_en_cutoff_shell_min_distance": np.nan,
    }
    if args.strong_residual_audit_samples:
        msg = (
            "strong residual audit is not implemented for the CAS-dressed "
            "teacher ansatz because the external CAS teacher has no available "
            "pointwise second-derivative action"
        )
        raise ValueError(msg)
    response_param_arrays = {
        f"response_params/{key}": value for key, value in tree_to_npz(params).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        ground_energy=ground.energy,
        overlap=overlap,
        hamiltonian=hamiltonian,
        source=source,
        poles=spectrum.excitation_energies,
        weights=spectrum.weights,
        omega=omega,
        intensity=intensity,
        samples=samples,
        sample_density=final_density,
        energy_stderr=energy_stderr,
        energy_pmove=energy_pmove,
        nspins=np.asarray(ground.nspins),
        response_official_workflow=True,
        response_official_workflow_name=OFFICIAL_RESPONSE_FLOW,
        response_official_final_sampling=OFFICIAL_FINAL_SAMPLING,
        response_official_partial_wave_closure=False,
        response_official_closure_kind="none",
        response_warm_start=args.warm_start,
        response_warm_start_heads=warm_start_heads,
        response_warm_start_requested_heads=args.warm_start_heads,
        response_warm_start_samples=args.warm_start_samples,
        response_warm_start_epochs=args.warm_start_epochs,
        response_warm_start_loss=warm_start_loss,
        response_warm_start_freeze_envelope_decay=False,
        response_warm_start_envelope_decay_floor=args.initial_decay_min,
        response_warm_start_target_count=warm_start_target_count,
        response_warm_start_basis=args.warm_start_basis,
        response_warm_start_min_source_overlap=args.warm_start_min_source_overlap,
        response_warm_start_backend=warm_start_backend,
        response_warm_start_teacher_mode="casscf-krylov-teachers",
        response_warm_start_sampling="cas_dressed_teacher_bright_influence_mixture",
        response_production_sampler=args.production_sampler,
        response_warm_start_sampling_sampler=warm_start_sampling_stats["sampler"],
        response_warm_start_sampling_pmove=warm_start_sampling_stats["pmove"],
        response_warm_start_sampling_walkers=warm_start_sampling_stats["walkers"],
        response_warm_start_sampling_burn_in=warm_start_sampling_stats["burn_in"],
        response_warm_start_sampling_steps_between=(
            warm_start_sampling_stats["steps_between"]
        ),
        response_warm_start_sampling_width=warm_start_sampling_stats["width"],
        response_warm_start_sampling_density_log_shift=(
            warm_start_sampling_stats["density_log_shift"]
        ),
        response_warm_start_sampling_proposal_samples=(
            warm_start_sampling_stats["proposal_samples"]
        ),
        response_warm_start_sampling_proposal_ess=(
            warm_start_sampling_stats["proposal_ess"]
        ),
        response_warm_start_sampling_proposal_ess_fraction=(
            warm_start_sampling_stats["proposal_ess_fraction"]
        ),
        response_warm_start_sampling_proposal_max_weight_fraction=(
            warm_start_sampling_stats["proposal_max_weight_fraction"]
        ),
        response_warm_start_sampling_resampling_unique_fraction=(
            warm_start_sampling_stats["resampling_unique_fraction"]
        ),
        response_warm_start_sampling_proposal_component_count=(
            warm_start_sampling_stats["proposal_component_count"]
        ),
        response_warm_start_sampling_leverage_normalizer=(
            warm_start_sampling_stats.get("leverage_normalizer", np.nan)
        ),
        response_warm_start_sampling_leverage_mean=(
            warm_start_sampling_stats.get("leverage_mean", np.nan)
        ),
        response_warm_start_sampling_leverage_max=(
            warm_start_sampling_stats.get("leverage_max", np.nan)
        ),
        response_warm_start_sampling_leverage_winsor_limit=(
            warm_start_sampling_stats.get("leverage_winsor_limit", np.nan)
        ),
        response_warm_start_sampling_ground_weight=(
            warm_start_sampling_stats["ground_weight"]
        ),
        response_warm_start_sampling_source_weight=(
            warm_start_sampling_stats["source_weight"]
        ),
        response_warm_start_sampling_teacher_weight=(
            warm_start_sampling_stats["teacher_weight"]
        ),
        response_warm_start_sampling_dressed_weight=(
            warm_start_sampling_stats["dressed_weight"]
        ),
        response_warm_start_sampling_aux_weight=(
            warm_start_sampling_stats["aux_weight"]
        ),
        response_warm_start_sampling_ground_norm=(
            warm_start_sampling_stats["ground_norm"]
        ),
        response_warm_start_sampling_source_norm=(
            warm_start_sampling_stats["source_norm"]
        ),
        response_warm_start_sampling_teacher_norms=(
            warm_start_sampling_stats["teacher_norms"]
        ),
        response_warm_start_sampling_dressed_norms=(
            warm_start_sampling_stats["dressed_norms"]
        ),
        response_warm_start_qc_energies=warm_start_qc_energies,
        response_warm_start_qc_source_overlaps=warm_start_qc_source_overlaps,
        response_warm_start_root_energies=warm_start_root_energies,
        response_warm_start_root_source_overlaps=warm_start_root_source_overlaps,
        response_warm_start_krylov_singular_values=(warm_start_krylov_singular_values),
        response_warm_start_krylov_coefficients=(warm_start_krylov_coefficients),
        response_warm_start_krylov_source_moments=(warm_start_krylov_source_moments),
        response_warm_start_krylov_svd_rtol=args.krylov_teacher_svd_rtol,
        response_warm_start_krylov_svd_atol=args.krylov_teacher_svd_atol,
        response_fixed_cas_ncas=args.fixed_cas_ncas,
        response_fixed_cas_n_roots=args.warm_start_n_roots,
        response_fixed_cas_gradient_weight=args.fixed_cas_gradient_weight,
        response_fixed_cas_fd_step=args.fixed_cas_fd_step,
        response_fixed_cas_state_average=args.fixed_cas_state_average,
        response_fixed_cas_finetune_epochs=args.fixed_cas_finetune_epochs,
        response_fixed_cas_finetune_batch_size=(args.fixed_cas_finetune_batch_size),
        response_fixed_cas_finetune_learning_rate=(
            args.fixed_cas_finetune_learning_rate
        ),
        response_fixed_cas_finetune_roots=args.fixed_cas_finetune_roots,
        response_fixed_cas_finetune_energy_weight=(
            args.fixed_cas_finetune_energy_weight
        ),
        response_fixed_cas_finetune_condition_weight=(
            args.fixed_cas_finetune_condition_weight
        ),
        response_fixed_cas_finetune_overlap_weight=(
            args.fixed_cas_finetune_overlap_weight
        ),
        response_fixed_cas_finetune_max_condition=(
            args.fixed_cas_finetune_max_condition
        ),
        response_fixed_cas_finetune_root_floor=(args.fixed_cas_finetune_root_floor),
        response_fixed_cas_finetune_validation_fraction=(
            args.fixed_cas_finetune_validation_fraction
        ),
        response_fixed_cas_finetune_bright_threshold=(
            args.fixed_cas_finetune_bright_threshold
        ),
        response_fixed_cas_finetune_validation_blocks=(
            args.fixed_cas_finetune_validation_blocks
        ),
        response_fixed_cas_finetune_acceptance_sigma=(
            args.fixed_cas_finetune_acceptance_sigma
        ),
        response_cas_dressing_finetune_loss=fine_tune_loss,
        response_cas_dressing_finetune_energy_loss=(fine_tune_stats["energy_loss"]),
        response_cas_dressing_finetune_visibility_loss=(
            fine_tune_stats["visibility_loss"]
        ),
        response_cas_dressing_finetune_regularizer=(fine_tune_stats["regularizer"]),
        response_cas_dressing_finetune_dressing_norm=(fine_tune_stats["dressing_norm"]),
        response_cas_dressing_finetune_dressing_grad_norm=(
            fine_tune_stats["dressing_grad_norm"]
        ),
        response_cas_dressing_finetune_condition=(fine_tune_stats["condition"]),
        response_cas_dressing_finetune_condition_penalty=(
            fine_tune_stats["condition_penalty"]
        ),
        response_cas_dressing_finetune_root0=fine_tune_stats["root0"],
        response_cas_dressing_finetune_source_weighted_root=(
            fine_tune_stats["source_weighted_root"]
        ),
        response_cas_dressing_finetune_source_weight_sum=(
            fine_tune_stats["source_weight_sum"]
        ),
        response_cas_dressing_finetune_source_weight0=(
            fine_tune_stats["source_weight0"]
        ),
        response_cas_dressing_finetune_accepted=fine_tune_stats["accepted"],
        response_cas_dressing_finetune_accepted_epoch=(
            fine_tune_stats["accepted_epoch"]
        ),
        response_cas_dressing_finetune_validation_root0=(
            fine_tune_stats["validation_root0"]
        ),
        response_cas_dressing_finetune_validation_source_weighted_root=(
            fine_tune_stats["validation_source_weighted_root"]
        ),
        response_cas_dressing_finetune_validation_source_weight_sum=(
            fine_tune_stats["validation_source_weight_sum"]
        ),
        response_cas_dressing_finetune_validation_condition=(
            fine_tune_stats["validation_condition"]
        ),
        response_cas_dressing_finetune_validation_se=(fine_tune_stats["validation_se"]),
        response_cas_dressing_finetune_validation_candidate_se=(
            fine_tune_stats["validation_candidate_se"]
        ),
        response_cas_dressing_finetune_validation_pair_delta=(
            fine_tune_stats["validation_pair_delta"]
        ),
        response_cas_dressing_finetune_validation_pair_se=(
            fine_tune_stats["validation_pair_se"]
        ),
        response_cas_dressing_finetune_validation_source_pair_delta=(
            fine_tune_stats["validation_source_pair_delta"]
        ),
        response_cas_dressing_finetune_validation_source_pair_se=(
            fine_tune_stats["validation_source_pair_se"]
        ),
        response_cas_dressing_finetune_last_validation_pair_delta=(
            fine_tune_stats["last_validation_pair_delta"]
        ),
        response_cas_dressing_finetune_last_validation_pair_se=(
            fine_tune_stats["last_validation_pair_se"]
        ),
        response_cas_dressing_finetune_last_validation_source_pair_delta=(
            fine_tune_stats["last_validation_source_pair_delta"]
        ),
        response_cas_dressing_finetune_last_validation_source_pair_se=(
            fine_tune_stats["last_validation_source_pair_se"]
        ),
        response_cas_dressing_finetune_baseline_validation_root0=(
            fine_tune_stats["baseline_validation_root0"]
        ),
        response_cas_dressing_finetune_baseline_validation_source_weighted_root=(
            fine_tune_stats["baseline_validation_source_weighted_root"]
        ),
        response_cas_dressing_finetune_baseline_validation_source_weight_sum=(
            fine_tune_stats["baseline_validation_source_weight_sum"]
        ),
        response_cas_dressing_finetune_baseline_validation_condition=(
            fine_tune_stats["baseline_validation_condition"]
        ),
        response_cas_dressing_finetune_baseline_validation_se=(
            fine_tune_stats["baseline_validation_se"]
        ),
        response_cas_dressing_init_scale=args.cas_dressing_init_scale,
        response_cas_dressing_visibility_weight=(args.cas_dressing_visibility_weight),
        response_cas_dressing_regularizer_weight=(args.cas_dressing_regularizer_weight),
        response_cas_dressing_gradient_regularizer_length=(
            args.cas_dressing_gradient_regularizer_length
        ),
        response_cas_dressed_final_selected_objective=(
            final_selected_metrics["objective"]
        ),
        response_cas_dressed_final_selected_bright_root0=(
            final_selected_metrics["bright_root0"]
        ),
        response_cas_dressed_final_selected_bright_weight0=(
            final_selected_metrics["bright_weight0"]
        ),
        response_cas_dressed_final_selected_condition=(
            final_selected_metrics["condition"]
        ),
        response_ansatz="cas_explicit_krylov_teacher_matrix_dressing",
        response_determinants_per_head=args.determinants_per_head,
        response_orbital_radial_powers=args.response_orbital_radial_powers,
        response_orbital_radial_scale=args.response_orbital_radial_scale,
        response_envelope_decays=response_envelope_decays,
        response_spatial_parity=args.response_spatial_parity,
        aux_source_gaussian_exponents=aux_source_exponents,
        aux_source_dipole_radial_powers=aux_source_dipole_radial_powers,
        aux_source_dipole_radial_scale=args.aux_source_dipole_radial_scale,
        aux_source_atom_odd_gaussian_exponents=aux_source_atom_odd_exponents,
        aux_source_atom_odd_slater_decays=aux_source_atom_odd_slater_decays,
        aux_source_bond_odd_slater_decays=aux_source_bond_odd_slater_decays,
        aux_source_dipole_ee_scales=aux_source_dipole_ee_scales,
        aux_source_bond_odd_ee_slater_decays=aux_source_bond_odd_ee_slater_decays,
        aux_source_bond_odd_ee_scales=aux_source_bond_odd_ee_scales,
        aux_source_count=final_aux_source_count,
        response_source_envelope_core_decay=(
            np.nan
            if args.response_source_envelope_core_decay <= 0
            else args.response_source_envelope_core_decay
        ),
        response_source_envelope_diffuse_decay=(
            np.nan
            if args.response_source_envelope_diffuse_decay <= 0
            else args.response_source_envelope_diffuse_decay
        ),
        response_flow=response_flow_name(args),
        trained_response_heads=trained_active_heads,
        active_response_heads=active_heads,
        final_matrix_retained_heads=active_heads,
        final_matrix_external_basis_count=external_basis_dim,
        final_sampling=args.final_sampling,
        final_production_sampler=args.production_sampler,
        final_production_certified_estimator=production_certified_final,
        final_production_certified_samples=samples.shape[0],
        final_production_certified_blocks=(
            0 if final_block_counts is None else np.asarray(final_block_counts).shape[0]
        ),
        final_sampling_sampler=final_sampling_sampler,
        final_sobol_replicas=args.final_sobol_replicas,
        final_sampling_pmove=pmove,
        final_sampling_proposal_ess_fraction=final_sampling_ess_fraction,
        final_sampling_resampling_unique_fraction=(final_sampling_unique_fraction),
        final_sampling_proposal_max_weight_fraction=(
            final_sampling_max_weight_fraction
        ),
        final_sampling_leverage_candidate_factor=(
            args.production_leverage_candidate_factor
        ),
        final_sampling_leverage_max_candidates=(
            args.production_leverage_max_candidates
        ),
        final_sampling_leverage_gradient_weight=(
            args.production_leverage_gradient_weight
        ),
        final_sampling_leverage_potential_weight=(
            args.production_leverage_potential_weight
        ),
        final_sampling_leverage_source_weight=(args.production_leverage_source_weight),
        final_sampling_leverage_component_weights=(final_sampling_component_weights),
        final_sampling_leverage_normalizer=final_sampling_leverage_normalizer,
        final_sampling_leverage_mean=final_sampling_leverage_mean,
        final_sampling_leverage_max=final_sampling_leverage_max,
        final_sampling_leverage_winsor_limit=(final_sampling_leverage_winsor_limit),
        final_sampling_density_log_shift=final_density_log_shift,
        final_sampling_density_log_shifts=final_density_log_shifts,
        final_diagnostic_roots=args.final_diagnostic_roots,
        cutoff_diagnostic_values=cutoff_diagnostic_values,
        cutoff_diagnostic_success=cutoff_diagnostics["success"],
        cutoff_diagnostic_retained=cutoff_diagnostics["retained"],
        cutoff_diagnostic_condition=cutoff_diagnostics["condition"],
        cutoff_diagnostic_moment_norm_rel_error=(
            cutoff_diagnostics["moment_norm_rel_error"]
        ),
        cutoff_diagnostic_moment_first_rel_error=(
            cutoff_diagnostics["moment_first_rel_error"]
        ),
        cutoff_diagnostic_bright_roots=cutoff_diagnostics["bright_roots"],
        cutoff_diagnostic_bright_norm_weights=(
            cutoff_diagnostics["bright_norm_weights"]
        ),
        cutoff_diagnostic_bright_root_spread_ev=(
            cutoff_diagnostics["bright_root_spread_ev"]
        ),
        matrix_debug_cutoffs=matrix_debug_cutoffs,
        cas_raw_debug_sample_count=cas_raw_debug_sample_count,
        cas_raw_debug_overlap=cas_raw_debug_overlap,
        cas_raw_debug_hamiltonian=cas_raw_debug_hamiltonian,
        cas_raw_debug_source=cas_raw_debug_source,
        cas_raw_debug_qc_energies=cas_raw_debug_qc_energies,
        cas_raw_debug_qc_source_overlaps=cas_raw_debug_qc_source_overlaps,
        cas_raw_debug_retained=cas_raw_debug_retained,
        cas_raw_debug_condition=cas_raw_debug_condition,
        cas_raw_debug_bright_roots=cas_raw_debug_bright_roots,
        cas_raw_debug_bright_norm_weights=cas_raw_debug_bright_norm_weights,
        cas_dressed_debug_overlap=cas_dressed_debug_overlap,
        cas_dressed_debug_hamiltonian=cas_dressed_debug_hamiltonian,
        cas_dressed_debug_source=cas_dressed_debug_source,
        cas_dressed_debug_retained=cas_dressed_debug_retained,
        cas_dressed_debug_condition=cas_dressed_debug_condition,
        cas_dressed_debug_bright_roots=cas_dressed_debug_bright_roots,
        cas_dressed_debug_bright_norm_weights=cas_dressed_debug_bright_norm_weights,
        source_envelope_antithetic=False,
        cas_dressed_production_sampling=True,
        final_bootstrap_replicates=args.final_bootstrap_replicates,
        final_replica_counts=final_replica["counts"],
        final_replica_projection_resampling=final_replica["projection_resampling"],
        final_replica_raw_overlaps=(
            np.asarray([])
            if final_block_raw_overlaps is None
            else final_block_raw_overlaps
        ),
        final_replica_raw_hamiltonians=(
            np.asarray([])
            if final_block_raw_hamiltonians is None
            else final_block_raw_hamiltonians
        ),
        final_replica_projection_numerators=(
            np.asarray([])
            if final_block_projection_numerators is None
            else final_block_projection_numerators
        ),
        final_replica_projection_norms=(
            np.asarray([])
            if final_block_projection_norms is None
            else final_block_projection_norms
        ),
        final_replica_ground_hamiltonians=(
            np.asarray([])
            if final_block_ground_hamiltonians is None
            else final_block_ground_hamiltonians
        ),
        final_replica_ground_hamiltonian_norms=(
            np.asarray([])
            if final_block_ground_hamiltonian_norms is None
            else final_block_ground_hamiltonian_norms
        ),
        final_replica_block_poles=final_replica["block_poles"],
        final_replica_block_bright_poles=final_replica["block_bright_poles"],
        final_replica_loo_poles=final_replica["loo_poles"],
        final_replica_loo_bright_poles=final_replica["loo_bright_poles"],
        final_replica_loo_count=final_replica["loo_count"],
        final_replica_loo_mean=final_replica["loo_mean"],
        final_replica_loo_std=final_replica["loo_std"],
        final_replica_loo_jackknife_se=final_replica["loo_jackknife_se"],
        final_replica_loo_jackknife_se_ev=final_replica["loo_jackknife_se_ev"],
        final_replica_loo_min=final_replica["loo_min"],
        final_replica_loo_max=final_replica["loo_max"],
        final_replica_loo_root_count=final_replica["loo_root_count"],
        final_replica_loo_root_mean=final_replica["loo_root_mean"],
        final_replica_loo_root_std=final_replica["loo_root_std"],
        final_replica_loo_root_jackknife_se=(final_replica["loo_root_jackknife_se"]),
        final_replica_loo_root_jackknife_se_ev=(
            final_replica["loo_root_jackknife_se_ev"]
        ),
        final_replica_loo_root_min=final_replica["loo_root_min"],
        final_replica_loo_root_max=final_replica["loo_root_max"],
        final_bootstrap_poles=final_replica["bootstrap_poles"],
        final_bootstrap_bright_poles=final_replica["bootstrap_bright_poles"],
        final_bootstrap_count=final_replica["bootstrap_count"],
        final_bootstrap_mean=final_replica["bootstrap_mean"],
        final_bootstrap_std=final_replica["bootstrap_std"],
        final_bootstrap_se=final_replica["bootstrap_se"],
        final_bootstrap_se_ev=final_replica["bootstrap_se_ev"],
        final_bootstrap_min=final_replica["bootstrap_min"],
        final_bootstrap_max=final_replica["bootstrap_max"],
        final_bootstrap_root_count=final_replica["bootstrap_root_count"],
        final_bootstrap_root_mean=final_replica["bootstrap_root_mean"],
        final_bootstrap_root_std=final_replica["bootstrap_root_std"],
        final_bootstrap_root_se=final_replica["bootstrap_root_se"],
        final_bootstrap_root_se_ev=final_replica["bootstrap_root_se_ev"],
        final_bootstrap_root_min=final_replica["bootstrap_root_min"],
        final_bootstrap_root_max=final_replica["bootstrap_root_max"],
        residual_omegas=residual_omegas,
        moment_source_norm=final_moments.source_norm,
        moment_spectral_norm=final_moments.spectral_norm,
        moment_source_first=final_moments.source_first_moment,
        moment_spectral_first=final_moments.spectral_first_moment,
        moment_norm_rel_error=final_moments.norm_rel_error,
        moment_first_rel_error=final_moments.first_moment_rel_error,
        moment_min_weight=final_moments.min_weight,
        moment_overlap_condition=final_moments.overlap_condition,
        **strong_residual,
        **response_param_arrays,
    )
    print_validation_report(
        args,
        ground,
        spectrum,
        peaks,
        pmove,
        energy_stderr,
        active_heads=active_heads,
        external_basis_count=external_basis_dim,
        enrichment=None,
        moments=final_moments,
        cutoff_diagnostics=cutoff_diagnostics,
        final_replica=final_replica,
        strong_residual=strong_residual,
    )


if __name__ == "__main__":
    main()
