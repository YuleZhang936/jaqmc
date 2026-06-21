# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Orbital-basis warm-start targets for neural projected response."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

_SYMBOL_BY_Z = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
}


@dataclass(frozen=True)
class OrbitalWarmStartTargets:
    """Real-space target values from an orbital-basis teacher calculation."""

    values: np.ndarray
    excitation_energies: np.ndarray
    source_overlaps: np.ndarray
    backend: str
    basis: str
    target_mode: str
    gradients: np.ndarray | None = None
    root_energies: np.ndarray | None = None
    root_source_overlaps: np.ndarray | None = None
    krylov_singular_values: np.ndarray | None = None
    krylov_coefficients: np.ndarray | None = None


CISWarmStartTargets = OrbitalWarmStartTargets


@dataclass(frozen=True)
class CASSCFSeedModel:
    """State-averaged CASSCF/CASCI data used only for seed carriers."""

    mol: Any
    mo_coeff: np.ndarray
    ncore: int
    ncas: int
    nelecas: tuple[int, int]
    ci_vectors: tuple[np.ndarray, ...]
    energies: np.ndarray
    source_overlaps: np.ndarray
    backend: str
    basis: str
    method: str


@dataclass(frozen=True)
class CASSCFKrylovTeacherModel:
    """Canonical source-Hamiltonian Krylov CASSCF teacher block."""

    seed_model: CASSCFSeedModel
    excited_roots: np.ndarray
    coefficients: np.ndarray
    singular_values: np.ndarray
    excitation_energies: np.ndarray
    root_source_overlaps: np.ndarray


@dataclass(frozen=True)
class CASSCFCarrierBank:
    """Real-space CAS-ratio carrier values and optional derivatives."""

    values: np.ndarray
    gradients: np.ndarray | None
    laplacians: np.ndarray | None
    excitation_energies: np.ndarray
    source_overlaps: np.ndarray
    basis: str
    method: str
    tau: float


def element_symbol(charge: float) -> str:
    """Return a chemical symbol for a nuclear charge.

    Raises:
        ValueError: If the charge cannot be mapped to a supported element.
    """
    z_value = round(float(charge))
    if z_value not in _SYMBOL_BY_Z or not np.isclose(float(charge), z_value):
        msg = f"unsupported nuclear charge for QC warm start: {charge}"
        raise ValueError(msg)
    return _SYMBOL_BY_Z[z_value]


def _import_pyscf(backend: str):
    if backend not in {"auto", "pyscf", "gpu4pyscf"}:
        msg = f"unknown QC backend: {backend}"
        raise ValueError(msg)
    if backend in {"auto", "gpu4pyscf"}:
        try:
            from gpu4pyscf import scf as gpu_scf  # type: ignore
            from pyscf import gto, tdscf  # type: ignore

            return gto, gpu_scf, tdscf, "gpu4pyscf"
        except Exception:
            if backend == "gpu4pyscf":
                raise
    from pyscf import gto, scf, tdscf  # type: ignore

    return gto, scf, tdscf, "pyscf"


def _make_mol(ground: Any, *, basis: str, backend: str):
    gto, scf_mod, tdscf, used_backend = _import_pyscf(backend)
    atoms = [
        (element_symbol(charge), tuple(np.asarray(coord, dtype=float)))
        for charge, coord in zip(ground.charges, ground.atoms, strict=True)
    ]
    nelec = int(ground.electron_shape[0])
    nuclear_charge = round(float(np.sum(np.asarray(ground.charges))))
    charge = nuclear_charge - nelec
    spin = int(ground.nspins[0] - ground.nspins[1])
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    return mol, scf_mod, tdscf, used_backend


def _make_casscf_mol(ground: Any, *, basis: str):
    from pyscf import gto, scf  # type: ignore

    atoms = [
        (element_symbol(charge), tuple(np.asarray(coord, dtype=float)))
        for charge, coord in zip(ground.charges, ground.atoms, strict=True)
    ]
    nelec = int(ground.electron_shape[0])
    nuclear_charge = round(float(np.sum(np.asarray(ground.charges))))
    charge = nuclear_charge - nelec
    spin = int(ground.nspins[0] - ground.nspins[1])
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.kernel()
    if not bool(getattr(mf, "converged", True)):
        msg = "SCF did not converge for CASSCF seed"
        raise RuntimeError(msg)
    return mol, mf


def _as_nelecas(nelecas: int | tuple[int, int], nspins: tuple[int, int]):
    if isinstance(nelecas, tuple):
        return (int(nelecas[0]), int(nelecas[1]))
    total = int(nelecas)
    spin = int(nspins[0] - nspins[1])
    nalpha = (total + spin) // 2
    nbeta = total - nalpha
    return (int(nalpha), int(nbeta))


def _string_occupations(norb: int, nelec: int) -> list[np.ndarray]:
    from pyscf.fci import cistring  # type: ignore

    strings = cistring.gen_strings4orblist(range(int(norb)), int(nelec))
    occupations = []
    for string in strings:
        occupations.append(
            np.asarray(
                [orb for orb in range(int(norb)) if int(string) & (1 << orb)],
                dtype=np.int64,
            )
        )
    return occupations


def _fast_one_two_electron_cas_values(
    mo_values: np.ndarray,
    ci: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nelecas: tuple[int, int],
) -> np.ndarray | None:
    if int(ncore) != 0:
        return None
    n_samples, nelec, _ = mo_values.shape
    del n_samples
    ncas = int(ncas)
    if nelecas == (1, 0) and nelec == 1 and ci.shape == (ncas, 1):
        return np.asarray(mo_values[:, 0, :ncas] @ ci[:, 0])
    if nelecas == (0, 1) and nelec == 1 and ci.shape == (1, ncas):
        return np.asarray(mo_values[:, 0, :ncas] @ ci[0, :])
    if nelecas == (1, 1) and nelec == 2 and ci.shape == (ncas, ncas):
        return np.asarray(
            np.einsum(
                "si,ij,sj->s",
                mo_values[:, 0, :ncas],
                ci,
                mo_values[:, 1, :ncas],
                optimize=True,
            )
        )
    return None


def _cas_wavefunction_values(
    mo_values: np.ndarray,
    ci_vector: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    nelecas: tuple[int, int],
) -> np.ndarray:
    n_samples, nelec, _ = mo_values.shape
    nalpha = int(ncore) + int(nelecas[0])
    nbeta = int(ncore) + int(nelecas[1])
    if nalpha + nbeta != nelec:
        msg = "CASSCF seed electron count mismatches sample shape"
        raise ValueError(msg)
    ci = np.asarray(ci_vector, dtype=np.float64)
    fast_values = _fast_one_two_electron_cas_values(
        mo_values,
        ci,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
    )
    if fast_values is not None:
        return fast_values
    core_occ = np.arange(int(ncore), dtype=np.int64)
    alpha_occ = _string_occupations(int(ncas), int(nelecas[0]))
    beta_occ = _string_occupations(int(ncas), int(nelecas[1]))
    if ci.shape != (len(alpha_occ), len(beta_occ)):
        msg = "CASSCF CI vector shape is incompatible with active strings"
        raise ValueError(msg)
    values = np.zeros((n_samples,), dtype=np.float64)
    for sample_idx, mo in enumerate(mo_values):
        alpha_mo = mo[:nalpha]
        beta_mo = mo[nalpha:]
        total = 0.0
        for ia, active_alpha in enumerate(alpha_occ):
            occ_alpha = np.concatenate([core_occ, int(ncore) + active_alpha])
            det_alpha = _determinant(alpha_mo, occ_alpha)
            for ib, active_beta in enumerate(beta_occ):
                coeff = ci[ia, ib]
                occ_beta = np.concatenate([core_occ, int(ncore) + active_beta])
                total += coeff * det_alpha * _determinant(beta_mo, occ_beta)
        values[sample_idx] = total
    return values


def _eval_mo_values(model: CASSCFSeedModel, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    ao = model.mol.eval_gto("GTOval_sph", points.reshape(-1, 3))
    mo = np.asarray(ao @ model.mo_coeff, dtype=np.float64)
    return mo.reshape(points.shape[0], points.shape[1], -1)


def _state_values(
    model: CASSCFSeedModel,
    points: np.ndarray,
    *,
    roots: Sequence[int] | None = None,
) -> np.ndarray:
    mo_values = _eval_mo_values(model, points)
    ci_vectors = (
        model.ci_vectors
        if roots is None
        else tuple(model.ci_vectors[int(root)] for root in roots)
    )
    columns = [
        _cas_wavefunction_values(
            mo_values,
            ci,
            ncore=model.ncore,
            ncas=model.ncas,
            nelecas=model.nelecas,
        )
        for ci in ci_vectors
    ]
    return np.stack(columns, axis=1)


def _cas_ratio_columns(
    state_values: np.ndarray,
    *,
    energies: np.ndarray,
    source_overlaps: np.ndarray,
    target_mode: str,
    correction_omegas: np.ndarray | None,
    correction_eta: float,
    tau: float,
    ratio_clip: float,
) -> np.ndarray:
    ground_values = state_values[:, 0]
    denom = ground_values**2 + float(tau) ** 2
    columns = []
    if target_mode in {"states", "states+correction"} and state_values.shape[1] > 1:
        columns.append(state_values[:, 1:] * ground_values[:, None] / denom[:, None])
    if target_mode in {"correction", "states+correction"} and state_values.shape[1] > 1:
        excitation_energies = energies[1:] - energies[0]
        if correction_omegas is None:
            correction_omegas = excitation_energies
        for omega in np.asarray(correction_omegas, dtype=np.float64):
            coeff = source_overlaps[1:] / (
                complex(float(omega), float(correction_eta)) - excitation_energies
            )
            correction = state_values[:, 1:] @ coeff
            ratio = correction * ground_values / denom
            columns.extend([np.real(ratio)[:, None], np.imag(ratio)[:, None]])
    if not columns:
        return np.zeros((state_values.shape[0], 0), dtype=np.float64)
    values = np.concatenate(columns, axis=1)
    if np.isfinite(float(ratio_clip)) and float(ratio_clip) > 0:
        values = values / np.sqrt(1.0 + (values / float(ratio_clip)) ** 2)
    return np.asarray(values, dtype=np.float64)


def _transition_source_overlaps(
    model: CASSCFSeedModel,
    *,
    source_axis: int,
) -> np.ndarray:
    from pyscf.fci import direct_spin1  # type: ignore

    dipole_ao = model.mol.intor("int1e_r")[int(source_axis)]
    active_mo = model.mo_coeff[:, model.ncore : model.ncore + model.ncas]
    dipole_active = active_mo.T @ dipole_ao @ active_mo
    overlaps = np.zeros((len(model.ci_vectors),), dtype=np.float64)
    ground_ci = model.ci_vectors[0]
    for root, ci in enumerate(model.ci_vectors):
        try:
            dm1 = direct_spin1.trans_rdm1(
                ci,
                ground_ci,
                model.ncas,
                model.nelecas,
            )
            overlaps[root] = float(np.einsum("pq,qp->", dipole_active, dm1).real)
        except Exception:
            overlaps[root] = 0.0
    return overlaps


def build_casscf_seed_model(  # noqa: C901
    ground: Any,
    *,
    basis: str,
    ncas: int,
    nelecas: int | tuple[int, int] | None = None,
    n_roots: int = 8,
    source_axis: int = 2,
    state_average: bool = True,
) -> CASSCFSeedModel:
    """Build a PySCF CASSCF/CASCI model for response seed carriers.

    Returns:
        CASSCF seed model containing active-space roots and metadata.

    Raises:
        RuntimeError: If SCF or CASSCF/CASCI fails to converge.
        ValueError: If active-space inputs are inconsistent.
    """
    from pyscf import lib, mcscf  # type: ignore

    if getattr(lib.param, "TMPDIR", None) is None:
        lib.param.TMPDIR = os.environ.get("TMPDIR", "/tmp")

    mol, mf = _make_casscf_mol(ground, basis=basis)
    nmo = int(np.asarray(mf.mo_coeff).shape[1])
    ncas = nmo if int(ncas) <= 0 else min(int(ncas), nmo)
    if nelecas is None:
        nelecas_tuple = tuple(int(x) for x in ground.nspins)
    else:
        nelecas_tuple = _as_nelecas(nelecas, tuple(int(x) for x in ground.nspins))
    if sum(nelecas_tuple) > int(mol.nelectron):
        msg = "CASSCF active electrons exceed total electron count"
        raise ValueError(msg)
    method = "sa-casscf"
    closed_shell = ground.nspins[0] == ground.nspins[1]
    use_state_average = bool(state_average and closed_shell)
    mc = (
        mcscf.CASSCF(mf, ncas, nelecas_tuple)
        if use_state_average
        else mcscf.CASCI(mf, ncas, nelecas_tuple)
    )
    if closed_shell and not use_state_average:
        mc.fix_spin_(ss=0.0, shift=0.5)
    mc.fcisolver.nroots = max(1, int(n_roots))
    if use_state_average and int(n_roots) > 1:
        mc = mc.state_average_([1.0 / int(n_roots)] * int(n_roots))
    else:
        method = "casci"
    mc.kernel()
    if not bool(getattr(mc, "converged", True)):
        msg = "CASSCF/CASCI seed calculation did not converge"
        raise RuntimeError(msg)
    ci_raw = mc.ci if isinstance(mc.ci, list) else [mc.ci]
    ci_vectors = tuple(np.asarray(ci, dtype=np.float64) for ci in ci_raw)
    if len(ci_vectors) == 0:
        msg = "CASSCF/CASCI seed calculation produced no roots"
        raise ValueError(msg)
    if hasattr(mc, "e_states") and getattr(mc, "e_states") is not None:
        energies = np.asarray(mc.e_states, dtype=np.float64)[: len(ci_vectors)]
    elif isinstance(getattr(mc, "e_tot", None), (list, tuple, np.ndarray)):
        energies = np.asarray(mc.e_tot, dtype=np.float64)[: len(ci_vectors)]
    else:
        energies = np.asarray([float(mc.e_tot)] * len(ci_vectors), dtype=np.float64)
    if energies.size == len(ci_vectors):
        order = np.argsort(energies)
        energies = energies[order]
        ci_vectors = tuple(ci_vectors[int(idx)] for idx in order)
    model = CASSCFSeedModel(
        mol=mol,
        mo_coeff=np.asarray(mc.mo_coeff, dtype=np.float64),
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
        nelecas=tuple(int(x) for x in mc.nelecas),
        ci_vectors=ci_vectors,
        energies=energies,
        source_overlaps=np.zeros((len(ci_vectors),), dtype=np.float64),
        backend="pyscf",
        basis=str(basis),
        method=method,
    )
    source_overlaps = _transition_source_overlaps(model, source_axis=source_axis)
    return replace(model, source_overlaps=source_overlaps)


def evaluate_casscf_ratio_carriers(
    model: CASSCFSeedModel,
    points: np.ndarray,
    *,
    target_mode: str = "states+correction",
    correction_omegas: np.ndarray | None = None,
    correction_eta: float = 0.05,
    tau_rel: float = 1e-6,
    tau_abs: float = 1e-12,
    ratio_clip: float = float("inf"),
    derivatives: bool = False,
    finite_difference_step: float = 1e-3,
) -> CASSCFCarrierBank:
    """Evaluate CASSCF ratio carriers and optional finite-difference derivatives.

    Returns:
        CAS-ratio carrier bank on the requested electron configurations.

    Raises:
        ValueError: If target mode or finite-difference options are invalid.
    """
    if target_mode not in {"states", "correction", "states+correction"}:
        msg = f"unknown CASSCF seed target_mode: {target_mode}"
        raise ValueError(msg)
    points = np.asarray(points, dtype=np.float64)
    state_values = _state_values(model, points)
    tau = max(
        float(tau_abs),
        float(tau_rel) * float(np.sqrt(np.mean(state_values[:, 0] ** 2) + 1e-300)),
    )

    def ratio_at(local_points: np.ndarray) -> np.ndarray:
        local_state_values = _state_values(model, local_points)
        return _cas_ratio_columns(
            local_state_values,
            energies=model.energies,
            source_overlaps=model.source_overlaps,
            target_mode=target_mode,
            correction_omegas=correction_omegas,
            correction_eta=correction_eta,
            tau=tau,
            ratio_clip=ratio_clip,
        )

    values = ratio_at(points)
    gradients = None
    laplacians = None
    if derivatives:
        step = float(finite_difference_step)
        if not (np.isfinite(step) and step > 0):
            msg = "finite_difference_step must be positive"
            raise ValueError(msg)
        gradients = np.zeros(
            (points.shape[0], values.shape[1], *points.shape[1:]),
            dtype=np.float64,
        )
        laplacians = np.zeros_like(values)
        for electron in range(points.shape[1]):
            for axis in range(3):
                plus = points.copy()
                minus = points.copy()
                plus[:, electron, axis] += step
                minus[:, electron, axis] -= step
                plus_values = ratio_at(plus)
                minus_values = ratio_at(minus)
                gradients[:, :, electron, axis] = (plus_values - minus_values) / (
                    2.0 * step
                )
                laplacians += (plus_values + minus_values - 2.0 * values) / (step**2)
    excitation_energies = (
        model.energies[1:] - model.energies[0]
        if model.energies.size > 1
        else np.asarray([], dtype=np.float64)
    )
    return CASSCFCarrierBank(
        values=np.asarray(values, dtype=np.float64),
        gradients=gradients,
        laplacians=laplacians,
        excitation_energies=np.asarray(excitation_energies, dtype=np.float64),
        source_overlaps=np.asarray(model.source_overlaps[1:], dtype=np.float64),
        basis=model.basis,
        method=model.method,
        tau=float(tau),
    )


def _select_source_visible_cas_roots(
    model: CASSCFSeedModel,
    *,
    n_targets: int,
    min_source_overlap: float,
) -> np.ndarray:
    excited = np.arange(1, len(model.ci_vectors), dtype=np.int64)
    if excited.size == 0:
        msg = "CASSCF neural targets require at least one excited/root target"
        raise ValueError(msg)
    overlaps = np.abs(np.asarray(model.source_overlaps, dtype=np.float64)[excited])
    order = excited[np.argsort(-overlaps)]
    if min_source_overlap > 0.0:
        bright = order[
            np.abs(np.asarray(model.source_overlaps, dtype=np.float64)[order])
            >= float(min_source_overlap)
        ]
        if bright.size >= int(n_targets):
            order = bright
    selected = order[: int(n_targets)]
    if selected.size < int(n_targets):
        msg = (
            "CASSCF neural target selection found fewer excited roots than "
            f"requested ({selected.size} < {int(n_targets)}); increase "
            "--warm-start-n-roots or lower --warm-start-min-source-overlap"
        )
        raise ValueError(msg)
    return selected


def _as_source_block(source_overlaps: np.ndarray) -> np.ndarray:
    source_block = np.asarray(source_overlaps, dtype=np.float64)
    if source_block.ndim == 1:
        source_block = source_block[:, None]
    if source_block.ndim != 2:
        msg = "CASSCF source overlap block must be a vector or matrix"
        raise ValueError(msg)
    return source_block


def _scaled_excitation_window(excitation_energies: np.ndarray) -> np.ndarray:
    energies = np.asarray(excitation_energies, dtype=np.float64).reshape(-1)
    if energies.size == 0:
        msg = "Krylov CASSCF teachers require at least one excited root"
        raise ValueError(msg)
    if not np.all(np.isfinite(energies)):
        msg = "Krylov CASSCF excitation energies must be finite"
        raise ValueError(msg)
    midpoint = 0.5 * (float(np.max(energies)) + float(np.min(energies)))
    half_width = 0.5 * (float(np.max(energies)) - float(np.min(energies)))
    span = half_width + 10.0 * np.finfo(np.float64).eps
    return (energies - midpoint) / span


def casscf_krylov_teacher_coefficients(  # noqa: C901
    excitation_energies: np.ndarray,
    source_overlaps: np.ndarray,
    *,
    max_teachers: int | None = None,
    svd_rtol: float = 1e-4,
    svd_atol: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Build canonical source-Hamiltonian Krylov teacher coefficients.

    The returned coefficient matrix has one row per excited CASSCF/CASCI root
    and one column per retained canonical teacher.  Columns are the left
    singular vectors of the block Krylov snapshot matrix
    ``[B, Hhat B, ..., Hhat^(M-1) B]``.

    Returns:
        ``(coefficients, singular_values)`` for retained teachers.

    Raises:
        ValueError: If the source block has zero numerical rank or inputs are
        inconsistent.
    """
    energies = np.asarray(excitation_energies, dtype=np.float64).reshape(-1)
    source_block = _as_source_block(source_overlaps)
    if source_block.shape[0] != energies.size:
        msg = "source overlap block and excitation energies disagree in root count"
        raise ValueError(msg)
    if energies.size == 0:
        msg = "Krylov CASSCF teachers require at least one excited root"
        raise ValueError(msg)
    if max_teachers is not None and int(max_teachers) < 1:
        msg = "max_teachers must be positive when provided"
        raise ValueError(msg)
    if not (
        np.isfinite(float(svd_rtol))
        and np.isfinite(float(svd_atol))
        and float(svd_rtol) >= 0.0
        and float(svd_atol) >= 0.0
    ):
        msg = "Krylov SVD tolerances must be finite and nonnegative"
        raise ValueError(msg)
    if not np.all(np.isfinite(source_block)):
        msg = "Krylov CASSCF source overlaps must be finite"
        raise ValueError(msg)
    scaled = _scaled_excitation_window(energies)
    snapshots = [
        (scaled[:, None] ** power) * source_block for power in range(energies.size)
    ]
    snapshot = np.concatenate(snapshots, axis=1)
    if not np.any(np.abs(snapshot) > 0.0):
        msg = "CASSCF source-Hamiltonian Krylov snapshot has zero source norm"
        raise ValueError(msg)
    left, singular_values, _ = np.linalg.svd(snapshot, full_matrices=False)
    if singular_values.size == 0:
        msg = "CASSCF source-Hamiltonian Krylov SVD returned no singular values"
        raise ValueError(msg)
    cutoff = max(float(svd_atol), float(svd_rtol) * float(singular_values[0]))
    retained = np.nonzero(singular_values > cutoff)[0]
    if retained.size == 0:
        msg = (
            "CASSCF source-Hamiltonian Krylov snapshot has no singular values "
            f"above cutoff {cutoff:.3e}"
        )
        raise ValueError(msg)
    if max_teachers is not None:
        retained = retained[: int(max_teachers)]
    coefficients = np.asarray(left[:, retained], dtype=np.float64)
    for col in range(coefficients.shape[1]):
        pivot = int(np.argmax(np.abs(coefficients[:, col])))
        if coefficients[pivot, col] < 0.0:
            coefficients[:, col] *= -1.0
    return coefficients, np.asarray(singular_values[retained], dtype=np.float64)


def _finite_difference_gradients(
    values_fn: Any,
    points: np.ndarray,
    values: np.ndarray,
    *,
    finite_difference_step: float,
) -> np.ndarray:
    step = float(finite_difference_step)
    if not (np.isfinite(step) and step > 0):
        msg = "finite_difference_step must be positive"
        raise ValueError(msg)
    gradients = np.zeros(
        (points.shape[0], values.shape[1], *points.shape[1:]),
        dtype=np.float64,
    )
    for electron in range(points.shape[1]):
        for axis in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[:, electron, axis] += step
            minus[:, electron, axis] -= step
            gradients[:, :, electron, axis] = (values_fn(plus) - values_fn(minus)) / (
                2.0 * step
            )
    return gradients


def build_casscf_krylov_teacher_model(
    ground: Any,
    *,
    basis: str,
    n_targets: int,
    n_roots: int,
    ncas: int,
    source_axis: int = 2,
    state_average: bool = True,
    svd_rtol: float = 1e-4,
    svd_atol: float = 1e-14,
) -> CASSCFKrylovTeacherModel:
    """Build the canonical source-Hamiltonian Krylov CASSCF teacher model.

    A single CASSCF/CASCI root pool is built, excited-root source overlaps are
    combined with the diagonal CAS Hamiltonian into a block Krylov snapshot,
    and retained SVD columns define the teacher functions.  Raw root labels are
    not used as target labels.

    Returns:
        Reusable CASSCF Krylov teacher model.

    Raises:
        ValueError: If no source-controllable teacher direction is retained.
    """
    if int(n_targets) < 1:
        msg = "n_targets must be positive for CASSCF Krylov teachers"
        raise ValueError(msg)
    if int(n_roots) < 2:
        n_roots = 2
    model = build_casscf_seed_model(
        ground,
        basis=basis,
        ncas=ncas,
        n_roots=int(n_roots),
        source_axis=source_axis,
        state_average=state_average,
    )
    excited_roots = np.arange(1, len(model.ci_vectors), dtype=np.int64)
    excitation_energies = model.energies[excited_roots] - model.energies[0]
    root_source_overlaps = model.source_overlaps[excited_roots]
    coefficients, singular_values = casscf_krylov_teacher_coefficients(
        excitation_energies,
        root_source_overlaps,
        max_teachers=int(n_targets),
        svd_rtol=float(svd_rtol),
        svd_atol=float(svd_atol),
    )
    return CASSCFKrylovTeacherModel(
        seed_model=model,
        excited_roots=np.asarray(excited_roots, dtype=np.int64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        excitation_energies=np.asarray(excitation_energies, dtype=np.float64),
        root_source_overlaps=np.asarray(root_source_overlaps, dtype=np.float64),
    )


def casscf_krylov_teacher_values(
    teacher_model: CASSCFKrylovTeacherModel,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate canonical Krylov teacher values on electron configurations.

    Returns:
        Matrix with shape ``(n_samples, n_teachers)``.
    """
    raw_values = _state_values(
        teacher_model.seed_model,
        np.asarray(points, dtype=np.float64),
        roots=teacher_model.excited_roots,
    )
    return np.asarray(raw_values @ teacher_model.coefficients, dtype=np.float64)


def evaluate_casscf_krylov_teacher_targets(
    teacher_model: CASSCFKrylovTeacherModel,
    points: np.ndarray,
    *,
    basis: str | None = None,
    gradients: bool = False,
    finite_difference_step: float = 1e-3,
) -> OrbitalWarmStartTargets:
    """Evaluate canonical Krylov CASSCF teacher targets on fixed points.

    Returns:
        Teacher values, optional finite-difference gradients, and metadata.
    """
    points = np.asarray(points, dtype=np.float64)
    values = casscf_krylov_teacher_values(teacher_model, points)
    target_gradients = None
    if gradients:
        target_gradients = _finite_difference_gradients(
            lambda local_points: casscf_krylov_teacher_values(
                teacher_model,
                local_points,
            ),
            points,
            values,
            finite_difference_step=finite_difference_step,
        )
    teacher_source_overlaps = (
        teacher_model.coefficients.T @ teacher_model.root_source_overlaps
    )
    teacher_energies = np.einsum(
        "ia,i,ia->a",
        teacher_model.coefficients,
        teacher_model.excitation_energies,
        teacher_model.coefficients,
        optimize=True,
    )
    seed_model = teacher_model.seed_model
    return OrbitalWarmStartTargets(
        values=np.asarray(values, dtype=np.float64),
        gradients=target_gradients,
        excitation_energies=np.asarray(teacher_energies, dtype=np.float64),
        source_overlaps=np.asarray(teacher_source_overlaps, dtype=np.float64),
        backend=seed_model.method,
        basis=seed_model.basis if basis is None else str(basis),
        target_mode="casscf-krylov-teachers",
        root_energies=np.asarray(
            teacher_model.excitation_energies,
            dtype=np.float64,
        ),
        root_source_overlaps=np.asarray(
            teacher_model.root_source_overlaps,
            dtype=np.float64,
        ),
        krylov_singular_values=np.asarray(
            teacher_model.singular_values,
            dtype=np.float64,
        ),
        krylov_coefficients=np.asarray(
            teacher_model.coefficients,
            dtype=np.float64,
        ),
    )


def build_casscf_krylov_teacher_targets(
    ground: Any,
    points: np.ndarray,
    *,
    basis: str,
    n_targets: int,
    n_roots: int,
    ncas: int,
    source_axis: int = 2,
    state_average: bool = True,
    gradients: bool = False,
    finite_difference_step: float = 1e-3,
    svd_rtol: float = 1e-4,
    svd_atol: float = 1e-14,
) -> OrbitalWarmStartTargets:
    """Build and evaluate canonical Krylov CASSCF teacher targets.

    Returns:
        Teacher values, optional finite-difference gradients, and metadata.
    """
    teacher_model = build_casscf_krylov_teacher_model(
        ground,
        basis=basis,
        n_targets=n_targets,
        n_roots=n_roots,
        ncas=ncas,
        source_axis=source_axis,
        state_average=state_average,
        svd_rtol=svd_rtol,
        svd_atol=svd_atol,
    )
    return evaluate_casscf_krylov_teacher_targets(
        teacher_model,
        points,
        basis=basis,
        gradients=gradients,
        finite_difference_step=finite_difference_step,
    )


def build_casscf_neural_response_targets(
    ground: Any,
    points: np.ndarray,
    *,
    basis: str,
    n_targets: int,
    n_roots: int,
    ncas: int,
    source_axis: int = 2,
    state_average: bool = True,
    min_source_overlap: float = 0.0,
    gradients: bool = False,
    finite_difference_step: float = 1e-3,
) -> OrbitalWarmStartTargets:
    """Build direct CASSCF/CASCI root targets for fixed neural response heads.

    Unlike the external-ratio carrier path, this returns the CASSCF/CASCI
    functions themselves.  Root 0 is treated as the active-space ground state
    and is excluded from the supervised response block.

    Returns:
        CASSCF/CASCI root values and optional finite-difference gradients.

    Raises:
        ValueError: If root selection or finite-difference options are invalid.
    """
    if int(n_targets) < 1:
        msg = "n_targets must be positive for CASSCF neural warm start"
        raise ValueError(msg)
    if int(n_roots) < int(n_targets) + 1:
        n_roots = int(n_targets) + 1
    if not np.isfinite(min_source_overlap) or min_source_overlap < 0:
        msg = "min_source_overlap must be finite and nonnegative"
        raise ValueError(msg)
    points = np.asarray(points, dtype=np.float64)
    model = build_casscf_seed_model(
        ground,
        basis=basis,
        ncas=ncas,
        n_roots=int(n_roots),
        source_axis=source_axis,
        state_average=state_average,
    )
    selected_roots = _select_source_visible_cas_roots(
        model,
        n_targets=int(n_targets),
        min_source_overlap=float(min_source_overlap),
    )

    def selected_values(local_points: np.ndarray) -> np.ndarray:
        return np.asarray(
            _state_values(model, local_points, roots=selected_roots),
            dtype=np.float64,
        )

    values = selected_values(points)
    target_gradients = None
    if gradients:
        step = float(finite_difference_step)
        if not (np.isfinite(step) and step > 0):
            msg = "finite_difference_step must be positive"
            raise ValueError(msg)
        target_gradients = np.zeros(
            (points.shape[0], values.shape[1], *points.shape[1:]),
            dtype=np.float64,
        )
        for electron in range(points.shape[1]):
            for axis in range(3):
                plus = points.copy()
                minus = points.copy()
                plus[:, electron, axis] += step
                minus[:, electron, axis] -= step
                target_gradients[:, :, electron, axis] = (
                    selected_values(plus) - selected_values(minus)
                ) / (2.0 * step)
    excitation_energies = model.energies[selected_roots] - model.energies[0]
    source_overlaps = model.source_overlaps[selected_roots]
    return OrbitalWarmStartTargets(
        values=np.asarray(values, dtype=np.float64),
        gradients=target_gradients,
        excitation_energies=np.asarray(excitation_energies, dtype=np.float64),
        source_overlaps=np.asarray(source_overlaps, dtype=np.float64),
        backend=model.method,
        basis=str(basis),
        target_mode="casscf-roots",
    )


def _determinant(mo_values: np.ndarray, occ: np.ndarray) -> float:
    if occ.size == 0:
        return 1.0
    return float(np.linalg.det(mo_values[:, occ]))


def _closed_shell_cis_values(
    mol: Any,
    mo_coeff: np.ndarray,
    amplitudes: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    n_samples, nelec, _ = points.shape
    nocc, nvir, n_roots = amplitudes.shape
    nalpha = nelec // 2
    nbeta = nelec - nalpha
    if nalpha != nbeta or nalpha != nocc:
        msg = "closed-shell CIS warm start requires nalpha=nbeta=nocc"
        raise ValueError(msg)
    occ = np.arange(nocc, dtype=np.int64)
    values = np.zeros((n_samples, n_roots), dtype=np.float64)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    ao = mol.eval_gto("GTOval_sph", np.asarray(points, dtype=np.float64).reshape(-1, 3))
    mo_all = np.asarray(ao @ mo_coeff, dtype=np.float64).reshape(n_samples, nelec, -1)
    for sample_idx, mo in enumerate(mo_all):
        mo_up = mo[:nalpha]
        mo_down = mo[nalpha:]
        det_up = _determinant(mo_up, occ)
        det_down = _determinant(mo_down, occ)
        for i in range(nocc):
            for a in range(nvir):
                coeffs = amplitudes[i, a]
                if not np.any(coeffs):
                    continue
                virt = nocc + a
                occ_exc = occ.copy()
                occ_exc[i] = virt
                up_exc = _determinant(mo_up, occ_exc) * det_down
                down_exc = det_up * _determinant(mo_down, occ_exc)
                values[sample_idx] += inv_sqrt2 * coeffs * (up_exc + down_exc)
    return values


def _unrestricted_cis_values(
    mol: Any,
    mo_coeff_alpha: np.ndarray,
    mo_coeff_beta: np.ndarray,
    alpha_amplitudes: np.ndarray,
    beta_amplitudes: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    n_samples, nelec, _ = points.shape
    nocc_alpha, nvir_alpha, n_roots = alpha_amplitudes.shape
    nocc_beta, nvir_beta, beta_roots = beta_amplitudes.shape
    if beta_roots != n_roots:
        msg = "unrestricted CIS alpha/beta root counts disagree"
        raise ValueError(msg)
    if nocc_alpha + nocc_beta != nelec:
        msg = "unrestricted CIS warm start occupation count mismatches walkers"
        raise ValueError(msg)

    occ_alpha = np.arange(nocc_alpha, dtype=np.int64)
    occ_beta = np.arange(nocc_beta, dtype=np.int64)
    values = np.zeros((n_samples, n_roots), dtype=np.float64)
    ao = mol.eval_gto("GTOval_sph", np.asarray(points, dtype=np.float64).reshape(-1, 3))
    mo_alpha_all = np.asarray(ao @ mo_coeff_alpha, dtype=np.float64).reshape(
        n_samples,
        nelec,
        -1,
    )
    mo_beta_all = np.asarray(ao @ mo_coeff_beta, dtype=np.float64).reshape(
        n_samples,
        nelec,
        -1,
    )
    for sample_idx, (mo_alpha, mo_beta) in enumerate(
        zip(mo_alpha_all, mo_beta_all, strict=True)
    ):
        alpha_block = mo_alpha[:nocc_alpha]
        beta_block = mo_beta[nocc_alpha:]
        det_alpha = _determinant(alpha_block, occ_alpha)
        det_beta = _determinant(beta_block, occ_beta)
        for i in range(nocc_alpha):
            for a in range(nvir_alpha):
                coeffs = alpha_amplitudes[i, a]
                if not np.any(coeffs):
                    continue
                occ_exc = occ_alpha.copy()
                occ_exc[i] = nocc_alpha + a
                values[sample_idx] += (
                    coeffs * _determinant(alpha_block, occ_exc) * det_beta
                )
        for i in range(nocc_beta):
            for a in range(nvir_beta):
                coeffs = beta_amplitudes[i, a]
                if not np.any(coeffs):
                    continue
                occ_exc = occ_beta.copy()
                occ_exc[i] = nocc_beta + a
                values[sample_idx] += (
                    coeffs * det_alpha * _determinant(beta_block, occ_exc)
                )
    return values


def _tda_amplitudes(td: Any, nocc: int, nvir: int, n_roots: int) -> np.ndarray:
    roots = min(int(n_roots), len(td.e))
    amplitudes = np.zeros((nocc, nvir, roots), dtype=np.float64)
    for root in range(roots):
        xy = td.xy[root]
        x_block = xy[0] if isinstance(xy, tuple) else xy
        amplitudes[:, :, root] = np.asarray(x_block, dtype=np.float64).reshape(
            nocc,
            nvir,
        )
    return amplitudes


def _unrestricted_tda_amplitudes(
    td: Any,
    nocc_alpha: int,
    nvir_alpha: int,
    nocc_beta: int,
    nvir_beta: int,
    n_roots: int,
) -> tuple[np.ndarray, np.ndarray]:
    roots = min(int(n_roots), len(td.e))
    alpha = np.zeros((nocc_alpha, nvir_alpha, roots), dtype=np.float64)
    beta = np.zeros((nocc_beta, nvir_beta, roots), dtype=np.float64)
    for root in range(roots):
        xy = td.xy[root]
        x_block = xy[0] if isinstance(xy, tuple) else xy
        x_alpha, x_beta = x_block
        alpha[:, :, root] = np.asarray(x_alpha, dtype=np.float64).reshape(
            nocc_alpha,
            nvir_alpha,
        )
        beta[:, :, root] = np.asarray(x_beta, dtype=np.float64).reshape(
            nocc_beta,
            nvir_beta,
        )
    return alpha, beta


def _transition_dipoles(td: Any, n_roots: int, source_axis: int) -> np.ndarray:
    try:
        dipoles = np.asarray(td.transition_dipole(), dtype=np.float64)
        return dipoles[:n_roots, int(source_axis)]
    except Exception:
        return np.ones((n_roots,), dtype=np.float64)


def _target_columns(
    *,
    state_values: np.ndarray,
    excitation_energies: np.ndarray,
    source_overlaps: np.ndarray,
    target_mode: str,
    correction_omegas: np.ndarray | None,
    correction_eta: float,
    min_source_overlap: float,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    if target_mode in {"states", "states+correction"}:
        order = np.argsort(-np.abs(source_overlaps))
        if min_source_overlap > 0:
            order = order[np.abs(source_overlaps[order]) >= min_source_overlap]
            if order.size == 0:
                msg = (
                    "source-bright CIS/TDA state filtering removed all state "
                    "targets; increase n_roots or lower min_source_overlap"
                )
                raise ValueError(msg)
        columns.append(state_values[:, order])
    if target_mode in {"correction", "states+correction"}:
        if correction_omegas is None:
            correction_omegas = excitation_energies
        correction_omegas = np.asarray(correction_omegas, dtype=np.float64)
        for omega in correction_omegas:
            denom = complex(float(omega), float(correction_eta)) - excitation_energies
            coeff = source_overlaps / denom
            correction = state_values @ coeff
            columns.extend(
                [
                    np.real(correction)[:, None],
                    np.imag(correction)[:, None],
                ]
            )
    target_values = np.concatenate(columns, axis=1) if columns else state_values
    finite_mask = np.all(np.isfinite(target_values), axis=0)
    return np.asarray(target_values[:, finite_mask], dtype=np.float64)


def build_orbital_basis_response_targets(
    ground: Any,
    points: np.ndarray,
    *,
    basis: str,
    n_roots: int,
    source_axis: int = 2,
    target_mode: str = "states+correction",
    correction_omegas: np.ndarray | None = None,
    correction_eta: float = 0.05,
    min_source_overlap: float = 0.0,
    backend: str = "auto",
) -> OrbitalWarmStartTargets:
    """Build value-only real-space CIS/TDA targets for warm-start pretraining.

    The returned targets are only supervised initialization data.  They are not
    meant to enter final BF-NKSR matrices.

    Returns:
        Real target columns and QC metadata.

    Raises:
        RuntimeError: If the reference SCF calculation does not converge.
        ValueError: If the reference or target options are unsupported.
    """
    if target_mode not in {"states", "correction", "states+correction"}:
        msg = f"unknown target_mode: {target_mode}"
        raise ValueError(msg)
    if not np.isfinite(min_source_overlap) or min_source_overlap < 0:
        msg = "min_source_overlap must be finite and nonnegative"
        raise ValueError(msg)
    points = np.asarray(points, dtype=np.float64)
    mol, scf_mod, tdscf, used_backend = _make_mol(
        ground,
        basis=basis,
        backend=backend,
    )
    closed_shell = ground.nspins[0] == ground.nspins[1]
    mf = scf_mod.RHF(mol) if closed_shell else scf_mod.UHF(mol)
    mf.kernel()
    if not bool(getattr(mf, "converged", True)):
        msg = "SCF did not converge for orbital-basis warm start"
        raise RuntimeError(msg)
    mf_td = mf.to_cpu() if hasattr(mf, "to_cpu") else mf
    if closed_shell:
        nocc = int(mol.nelectron // 2)
        nvir = int(np.asarray(mf_td.mo_coeff).shape[1] - nocc)
        max_roots = nocc * nvir
    else:
        mo_coeff_alpha, mo_coeff_beta = mf_td.mo_coeff
        mo_occ_alpha, mo_occ_beta = mf_td.mo_occ
        nocc_alpha = int(np.count_nonzero(np.asarray(mo_occ_alpha) > 0))
        nocc_beta = int(np.count_nonzero(np.asarray(mo_occ_beta) > 0))
        nvir_alpha = int(np.asarray(mo_coeff_alpha).shape[1] - nocc_alpha)
        nvir_beta = int(np.asarray(mo_coeff_beta).shape[1] - nocc_beta)
        max_roots = nocc_alpha * nvir_alpha + nocc_beta * nvir_beta
    if max_roots < 1:
        msg = "orbital-basis warm start requires at least one virtual orbital"
        raise ValueError(msg)
    td = tdscf.TDA(mf_td)
    td.nstates = min(int(n_roots), max_roots)
    td.kernel()
    excitation_energies = np.asarray(td.e, dtype=np.float64)
    if closed_shell:
        amplitudes = _tda_amplitudes(td, nocc, nvir, int(n_roots))
        n_found = amplitudes.shape[2]
    else:
        alpha_amplitudes, beta_amplitudes = _unrestricted_tda_amplitudes(
            td,
            nocc_alpha,
            nvir_alpha,
            nocc_beta,
            nvir_beta,
            int(n_roots),
        )
        n_found = alpha_amplitudes.shape[2]
    excitation_energies = excitation_energies[:n_found]
    source_overlaps = _transition_dipoles(
        td,
        n_found,
        int(source_axis),
    )
    if closed_shell:
        state_values = _closed_shell_cis_values(
            mol,
            np.asarray(mf_td.mo_coeff, dtype=np.float64),
            amplitudes,
            points,
        )
    else:
        state_values = _unrestricted_cis_values(
            mol,
            np.asarray(mo_coeff_alpha, dtype=np.float64),
            np.asarray(mo_coeff_beta, dtype=np.float64),
            alpha_amplitudes,
            beta_amplitudes,
            points,
        )
    target_values = _target_columns(
        state_values=state_values,
        excitation_energies=excitation_energies,
        source_overlaps=source_overlaps,
        target_mode=target_mode,
        correction_omegas=correction_omegas,
        correction_eta=correction_eta,
        min_source_overlap=float(min_source_overlap),
    )
    return OrbitalWarmStartTargets(
        values=np.asarray(target_values, dtype=np.float64),
        excitation_energies=excitation_energies,
        source_overlaps=source_overlaps,
        backend=used_backend,
        basis=str(basis),
        target_mode=str(target_mode),
    )


def build_closed_shell_cis_targets(
    ground: Any,
    points: np.ndarray,
    *,
    basis: str,
    n_roots: int,
    source_axis: int = 2,
    target_mode: str = "states+correction",
    correction_omegas: np.ndarray | None = None,
    correction_eta: float = 0.05,
    min_source_overlap: float = 0.0,
    backend: str = "auto",
) -> OrbitalWarmStartTargets:
    """Compatibility wrapper for the closed-shell orbital warm-start path.

    Returns:
        Orbital-basis warm-start values and QC metadata.

    Raises:
        ValueError: If the supplied ground state is not closed shell.
    """
    if ground.nspins[0] != ground.nspins[1]:
        msg = "closed-shell CIS warm start requires nalpha=nbeta"
        raise ValueError(msg)
    return build_orbital_basis_response_targets(
        ground,
        points,
        basis=basis,
        n_roots=n_roots,
        source_axis=source_axis,
        target_mode=target_mode,
        correction_omegas=correction_omegas,
        correction_eta=correction_eta,
        min_source_overlap=min_source_overlap,
        backend=backend,
    )
