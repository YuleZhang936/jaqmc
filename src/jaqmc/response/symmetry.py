# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Symmetry projectors for source-projected molecular NQS-LIT."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations, product

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import MoleculeData
from jaqmc.array_types import Params
from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME

type MoleculeValueFn = Callable[[MoleculeData], jnp.ndarray]
type MoleculeLogApply = Callable[[Params, MoleculeData], jnp.ndarray]

_DEFAULT_SO3_QUADRATURE_ORDER = 4
_DEFAULT_SO2_QUADRATURE_ORDER = 24
_DEFAULT_PROJECTOR_CHUNK_SIZE = 4


@dataclass(frozen=True)
class SpatialProjector:
    """Spatial projector represented by a finite group or quadrature sum."""

    label: str
    matrices: tuple[tuple[tuple[float, float, float], ...], ...]
    coefficients: tuple[complex, ...]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimension: int = 1
    group_label: str = "C1"

    @property
    def is_identity(self) -> bool:
        return len(self.matrices) == 1 and self.label in {"identity", "c1"}


@dataclass(frozen=True)
class SpinProjector:
    """Löwdin projector onto a total-spin sector in fixed ``M_S`` space."""

    nspins: tuple[int, int]
    target_s: float | None
    label: str = "identity"

    @property
    def is_identity(self) -> bool:
        return self.target_s is None or len(allowed_total_spins(self.nspins)) <= 1


@dataclass(frozen=True)
class SymmetryProjector:
    """Combined molecular sector projector ``P_lambda = P_Gamma P_S``."""

    spatial: SpatialProjector
    spin: SpinProjector
    label: str

    @property
    def is_identity(self) -> bool:
        return self.spatial.is_identity and self.spin.is_identity


def identity_spatial_projector(label: str = "identity") -> SpatialProjector:
    """Return the identity spatial projector."""
    return SpatialProjector(
        label=label,
        matrices=(_matrix_tuple(np.eye(3)),),
        coefficients=(1.0 + 0.0j,),
    )


def parity_spatial_projector(
    parity: str,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> SpatialProjector:
    """Return the even or odd inversion projector."""
    normalized = parity.lower()
    if normalized not in {"even", "odd"}:
        msg = f"parity must be 'even' or 'odd', got {parity!r}."
        raise ValueError(msg)
    sign = 1.0 if normalized == "even" else -1.0
    return SpatialProjector(
        label=f"parity_{normalized}",
        matrices=(_matrix_tuple(np.eye(3)), _matrix_tuple(-np.eye(3))),
        coefficients=(0.5 + 0.0j, 0.5 * sign + 0.0j),
        origin=tuple(float(x) for x in origin),
        group_label="Ci",
    )


def atomic_angular_spatial_projector(
    angular_momentum: int,
    *,
    parity: str | None = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    quadrature_order: int = _DEFAULT_SO3_QUADRATURE_ORDER,
) -> SpatialProjector:
    """Return a numerical Haar projector onto an atomic ``L`` sector."""
    l_value = int(angular_momentum)
    if l_value < 0:
        msg = f"angular_momentum must be nonnegative, got {angular_momentum!r}."
        raise ValueError(msg)
    order = _positive_quadrature_order(quadrature_order, "SO(3)")
    matrices, coefficients = _so3_character_projector_terms(l_value, order)
    projector = SpatialProjector(
        label=f"L{l_value}",
        matrices=matrices,
        coefficients=coefficients,
        origin=tuple(float(x) for x in origin),
        dimension=2 * l_value + 1,
        group_label=f"SO3_q{order}",
    )
    if parity is None:
        return projector
    normalized = parity.lower()
    if normalized in {"g", "gerade", "even"}:
        return _with_parity(
            projector,
            "even",
            suffix="even",
            group_label=f"O3_q{order}",
        )
    if normalized in {"u", "ungerade", "odd"}:
        return _with_parity(projector, "odd", suffix="odd", group_label=f"O3_q{order}")
    msg = f"Unknown atomic parity {parity!r}; expected even/odd."
    raise ValueError(msg)


def linear_spatial_projector(
    lambda_abs: int,
    axis: Sequence[float],
    *,
    parity: str | None = None,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    quadrature_order: int = _DEFAULT_SO2_QUADRATURE_ORDER,
) -> SpatialProjector:
    """Return a numerical axial projector onto a linear-molecule ``|Lambda|``."""
    lambda_value = int(lambda_abs)
    if lambda_value < 0:
        msg = f"lambda_abs must be nonnegative, got {lambda_abs!r}."
        raise ValueError(msg)
    order = _positive_quadrature_order(quadrature_order, "SO(2)")
    axis_arr = _unit_vector(np.asarray(axis, dtype=np.float64))
    matrices = []
    coefficients = []
    for point in range(order):
        angle = 2.0 * np.pi * float(point) / float(order)
        matrices.append(_matrix_tuple(_axis_rotation_matrix(axis_arr, angle)))
        if lambda_value == 0:
            coefficients.append(1.0 / float(order))
        else:
            coefficients.append(2.0 * np.cos(lambda_value * angle) / float(order))
    projector = SpatialProjector(
        label=f"Lambda{lambda_value}",
        matrices=tuple(matrices),
        coefficients=tuple(complex(coefficient) for coefficient in coefficients),
        origin=tuple(float(x) for x in origin),
        dimension=1 if lambda_value == 0 else 2,
        group_label=f"SO2_q{order}",
    )
    if parity is None:
        return projector
    normalized = parity.lower()
    if normalized in {"g", "gerade", "even"}:
        return _with_parity(
            projector,
            "even",
            suffix="g",
            group_label=f"Dinfh_q{order}",
        )
    if normalized in {"u", "ungerade", "odd"}:
        return _with_parity(projector, "odd", suffix="u", group_label=f"Dinfh_q{order}")
    msg = f"Unknown linear-molecule parity {parity!r}; expected g/u or even/odd."
    raise ValueError(msg)


def finite_spatial_irrep_projectors(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    origin: Sequence[float] | None = None,
    tolerance: float = 1e-5,
) -> tuple[SpatialProjector, ...]:
    """Construct central projectors for the finite nuclear point group.

    The finite group operations are discovered directly from the clamped nuclear
    geometry.  Irreducible central projectors are then obtained from the class
    algebra, which avoids hard-coded character tables and keeps the response
    workflow applicable to any finite molecular point group represented by the
    discovered operations.
    """
    origin_arr = _symmetry_origin(atoms) if origin is None else np.asarray(origin)
    operations = _discover_point_group_operations(
        atoms,
        charges,
        origin_arr,
        tolerance,
    )
    if len(operations) <= 1:
        return (identity_spatial_projector("A1"),)

    multiplication = _multiplication_table(operations, tolerance)
    classes = _conjugacy_classes(multiplication)
    class_characters = _class_irreducible_characters(multiplication, classes)
    group_label = f"PG{len(operations)}"
    projectors = []
    for irrep_index, (dimension, characters) in enumerate(class_characters):
        coefficients = [0j] * len(operations)
        for class_index, class_members in enumerate(classes):
            coefficient = (
                float(dimension)
                * np.conj(characters[class_index])
                / float(len(operations))
            )
            for operation_index in class_members:
                coefficients[operation_index] = complex(coefficient)
        label = _irrep_label(
            irrep_index,
            dimension,
            characters,
            classes,
            operations,
        )
        projectors.append(
            SpatialProjector(
                label=label,
                matrices=tuple(_matrix_tuple(operation) for operation in operations),
                coefficients=tuple(coefficients),
                origin=tuple(float(x) for x in origin_arr),
                dimension=int(dimension),
                group_label=group_label,
            )
        )
    return tuple(sorted(projectors, key=_projector_sort_key))


def select_spatial_projector(
    projectors: Sequence[SpatialProjector],
    label: str,
) -> SpatialProjector:
    """Select one spatial projector by label.

    Labels are matched case-insensitively.  The ``irrep:`` prefix is accepted so
    configuration files can distinguish an irrep label from a projector mode.
    Numeric labels select by zero-based position in the supplied projector list.
    """
    if not projectors:
        msg = "No spatial projectors are available for selection."
        raise ValueError(msg)
    normalized = _normalize_projector_label(label)
    if normalized.isdigit():
        index = int(normalized)
        if 0 <= index < len(projectors):
            return projectors[index]
    for projector in projectors:
        if _normalize_projector_label(projector.label) == normalized:
            return projector
    available = ", ".join(projector.label for projector in projectors)
    msg = f"Unknown spatial irrep {label!r}; available projectors: {available}."
    raise ValueError(msg)


def select_spatial_projectors(
    projectors: Sequence[SpatialProjector],
    labels: Sequence[str],
) -> tuple[SpatialProjector, ...]:
    """Select spatial projectors by label while preserving user order."""
    selected = []
    seen = set()
    for label in labels:
        projector = select_spatial_projector(projectors, label)
        key = _normalize_projector_label(projector.label)
        if key in seen:
            continue
        selected.append(projector)
        seen.add(key)
    return tuple(selected)


def make_dipole_spatial_projectors(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    mode: str,
    axis: int,
    irrep_labels: Sequence[str] | None = None,
    origin: Sequence[float] | None = None,
    tolerance: float = 1e-5,
    so3_quadrature_order: int = _DEFAULT_SO3_QUADRATURE_ORDER,
    so2_quadrature_order: int = _DEFAULT_SO2_QUADRATURE_ORDER,
) -> tuple[SpatialProjector, ...]:
    """Build all candidate spatial sectors for a dipole source component."""
    normalized = mode.lower()
    origin_arr = _symmetry_origin(atoms) if origin is None else np.asarray(origin)
    rank = _geometry_rank(atoms, origin_arr, tolerance)
    if irrep_labels:
        return select_spatial_projectors(
            _spatial_projector_selection_candidates(
                atoms,
                charges,
                mode=mode,
                axis=axis,
                origin=origin_arr,
                tolerance=tolerance,
                so3_quadrature_order=so3_quadrature_order,
                so2_quadrature_order=so2_quadrature_order,
            ),
            irrep_labels,
        )
    if normalized in {"auto", "continuous", "angular", "so3", "so2"}:
        if rank == 0:
            return (
                atomic_angular_spatial_projector(
                    1,
                    parity="odd",
                    origin=origin_arr,
                    quadrature_order=so3_quadrature_order,
                ),
            )
        if rank == 1:
            return _linear_dipole_spatial_projectors(
                atoms,
                charges,
                axis=axis,
                origin=origin_arr,
                tolerance=tolerance,
                quadrature_order=so2_quadrature_order,
            )
    if normalized == "finite" and rank <= 1:
        if _operation_maps_atoms(-np.eye(3), atoms, charges, origin_arr, tolerance):
            return (parity_spatial_projector("odd", origin=origin_arr),)
        return (
            _signed_permutation_axis_projector(
                atoms,
                charges,
                axis=axis,
                origin=origin_arr,
                tolerance=tolerance,
            ),
        )
    if normalized in {"auto", "irreps", "all", "point_group"}:
        return finite_spatial_irrep_projectors(
            atoms,
            charges,
            origin=origin_arr,
            tolerance=tolerance,
        )
    return (
        make_dipole_spatial_projector(
            atoms,
            charges,
            mode=mode,
            axis=axis,
            origin=origin,
            tolerance=tolerance,
            so3_quadrature_order=so3_quadrature_order,
            so2_quadrature_order=so2_quadrature_order,
        ),
    )


def make_dipole_spatial_projector(  # noqa: C901
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    mode: str,
    axis: int,
    origin: Sequence[float] | None = None,
    tolerance: float = 1e-5,
    so3_quadrature_order: int = _DEFAULT_SO3_QUADRATURE_ORDER,
    so2_quadrature_order: int = _DEFAULT_SO2_QUADRATURE_ORDER,
) -> SpatialProjector:
    """Build the spatial projector used for a dipole source component.

    This helper returns a single explicit projector.  Use
    :func:`make_dipole_spatial_projectors` in ``auto`` mode to enumerate all
    finite point-group irreducible sectors and let the source norm choose the
    nonzero ones.
    """
    normalized = mode.lower()
    origin_arr = _symmetry_origin(atoms) if origin is None else np.asarray(origin)
    rank = _geometry_rank(atoms, origin_arr, tolerance)
    if normalized in {"off", "none", "identity", "c1"}:
        return identity_spatial_projector("c1")
    if rank == 0 and normalized in {"auto", "continuous", "angular", "so3"}:
        return atomic_angular_spatial_projector(
            1,
            parity="odd",
            origin=origin_arr,
            quadrature_order=so3_quadrature_order,
        )
    if rank == 1 and normalized in {"auto", "continuous", "angular", "so2"}:
        return _linear_dipole_spatial_projectors(
            atoms,
            charges,
            axis=axis,
            origin=origin_arr,
            tolerance=tolerance,
            quadrature_order=so2_quadrature_order,
        )[0]
    if normalized == "auto":
        projectors = finite_spatial_irrep_projectors(
            atoms,
            charges,
            origin=origin_arr,
            tolerance=tolerance,
        )
        if len(projectors) == 1:
            return projectors[0]
        msg = (
            "lit.nqs_spatial_projector='auto' enumerates multiple sectors; "
            "call make_dipole_spatial_projectors instead."
        )
        raise ValueError(msg)
    if normalized in {"parity_odd", "odd", "ungerade", "u"}:
        _require_operation_symmetry(-np.eye(3), atoms, charges, origin_arr, tolerance)
        return parity_spatial_projector("odd", origin=origin_arr)
    if normalized in {"parity_even", "even", "gerade", "g"}:
        _require_operation_symmetry(-np.eye(3), atoms, charges, origin_arr, tolerance)
        return parity_spatial_projector("even", origin=origin_arr)
    if normalized == "signed_permutation_axis":
        return _signed_permutation_axis_projector(
            atoms,
            charges,
            axis=axis,
            origin=origin_arr,
            tolerance=tolerance,
        )
    for projector in _spatial_projector_selection_candidates(
        atoms,
        charges,
        mode=mode,
        axis=axis,
        origin=origin_arr,
        tolerance=tolerance,
        so3_quadrature_order=so3_quadrature_order,
        so2_quadrature_order=so2_quadrature_order,
    ):
        if _normalize_projector_label(projector.label) == _normalize_projector_label(
            mode
        ):
            return projector
    msg = (
        "Unknown lit.nqs_spatial_projector mode "
        f"{mode!r}; expected auto, continuous, identity, parity_odd, "
        "parity_even, signed_permutation_axis, or an available irrep label."
    )
    raise ValueError(msg)


def make_ground_spatial_projector(  # noqa: C901
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    mode: str,
    irrep_label: str | None = None,
    origin: Sequence[float] | None = None,
    tolerance: float = 1e-5,
    so3_quadrature_order: int = _DEFAULT_SO3_QUADRATURE_ORDER,
    so2_quadrature_order: int = _DEFAULT_SO2_QUADRATURE_ORDER,
) -> SpatialProjector:
    """Build the spatial projector for the ground-state sector."""
    normalized = mode.lower()
    origin_arr = _symmetry_origin(atoms) if origin is None else np.asarray(origin)
    rank = _geometry_rank(atoms, origin_arr, tolerance)
    if irrep_label:
        return select_spatial_projector(
            _ground_spatial_projector_candidates(
                atoms,
                charges,
                mode=mode,
                origin=origin_arr,
                tolerance=tolerance,
                so3_quadrature_order=so3_quadrature_order,
                so2_quadrature_order=so2_quadrature_order,
            ),
            irrep_label,
        )
    if normalized in {"off", "none", "identity", "c1"}:
        return identity_spatial_projector("c1")
    if normalized in {"auto", "continuous", "angular", "so3", "so2"}:
        if rank == 0:
            return atomic_angular_spatial_projector(
                0,
                parity="even",
                origin=origin_arr,
                quadrature_order=so3_quadrature_order,
            )
        if rank == 1:
            parity = (
                "g"
                if _operation_maps_atoms(
                    -np.eye(3),
                    atoms,
                    charges,
                    origin_arr,
                    tolerance,
                )
                else None
            )
            return linear_spatial_projector(
                0,
                _linear_geometry_axis(atoms, origin_arr, tolerance),
                parity=parity,
                origin=origin_arr,
                quadrature_order=so2_quadrature_order,
            )
    if normalized == "auto":
        if rank <= 1:
            if _operation_maps_atoms(-np.eye(3), atoms, charges, origin_arr, tolerance):
                return parity_spatial_projector("even", origin=origin_arr)
            return identity_spatial_projector("c1")
        return _totally_symmetric_projector(
            finite_spatial_irrep_projectors(
                atoms,
                charges,
                origin=origin_arr,
                tolerance=tolerance,
            )
        )
    if normalized in {"parity_even", "even", "gerade", "g"}:
        _require_operation_symmetry(-np.eye(3), atoms, charges, origin_arr, tolerance)
        return parity_spatial_projector("even", origin=origin_arr)
    if normalized in {"parity_odd", "odd", "ungerade", "u"}:
        _require_operation_symmetry(-np.eye(3), atoms, charges, origin_arr, tolerance)
        return parity_spatial_projector("odd", origin=origin_arr)
    for projector in _ground_spatial_projector_candidates(
        atoms,
        charges,
        mode=mode,
        origin=origin_arr,
        tolerance=tolerance,
        so3_quadrature_order=so3_quadrature_order,
        so2_quadrature_order=so2_quadrature_order,
    ):
        if _normalize_projector_label(projector.label) == _normalize_projector_label(
            mode
        ):
            return projector
    msg = (
        "Unknown lit.nqs_ground_spatial_projector mode "
        f"{mode!r}; expected auto, continuous, identity, parity_even, "
        "parity_odd, or an available irrep label."
    )
    raise ValueError(msg)


def make_spin_projector(
    nspins: tuple[int, int],
    *,
    target_s: float | None,
    enabled: bool = True,
    label: str | None = None,
) -> SpinProjector:
    """Return a Löwdin spin projector, or identity when disabled."""
    if not enabled or target_s is None:
        return SpinProjector(nspins=nspins, target_s=None, label="identity")
    return SpinProjector(
        nspins=nspins,
        target_s=float(target_s),
        label=label or f"S={float(target_s):g}",
    )


def allowed_total_spins(nspins: tuple[int, int]) -> tuple[float, ...]:
    """Return all total-spin values compatible with fixed ``M_S``."""
    n_up, n_down = (int(nspins[0]), int(nspins[1]))
    n_electrons = n_up + n_down
    min_s = abs(n_up - n_down) * 0.5
    max_s = n_electrons * 0.5
    count = round(max_s - min_s) + 1
    return tuple(float(min_s + index) for index in range(max(1, count)))


def spin_project_value(
    value_fn: MoleculeValueFn,
    data: MoleculeData,
    projector: SpinProjector,
) -> jnp.ndarray:
    """Apply the Löwdin total-spin projector to ``value_fn`` at ``data``."""
    if projector.is_identity:
        return value_fn(data)
    allowed = allowed_total_spins(projector.nspins)
    target_s = float(projector.target_s)
    if not any(np.isclose(spin, target_s) for spin in allowed):
        msg = (
            f"target_s={target_s:g} is incompatible with nspins={projector.nspins}; "
            f"allowed values are {allowed}."
        )
        raise ValueError(msg)

    projected_fn = value_fn
    target_eigenvalue = target_s * (target_s + 1.0)
    for other_s in allowed:
        if np.isclose(other_s, target_s):
            continue
        other_eigenvalue = other_s * (other_s + 1.0)
        denominator = target_eigenvalue - other_eigenvalue
        previous_fn = projected_fn

        def projected_fn(  # type: ignore[no-redef]
            local_data: MoleculeData,
            *,
            previous_fn=previous_fn,
            other_eigenvalue=other_eigenvalue,
            denominator=denominator,
        ) -> jnp.ndarray:
            return (
                spin_squared_value(previous_fn, local_data, projector.nspins)
                - other_eigenvalue * previous_fn(local_data)
            ) / denominator

    return projected_fn(data)


def spin_squared_value(
    value_fn: MoleculeValueFn,
    data: MoleculeData,
    nspins: tuple[int, int],
) -> jnp.ndarray:
    """Apply ``S^2`` to a spin-blocked spatial wavefunction value."""
    n_up, n_down = (int(nspins[0]), int(nspins[1]))
    n_electrons = n_up + n_down
    if n_electrons == 0:
        return value_fn(data)
    if n_up > n_down:
        majority = tuple(range(n_up))
        minority = tuple(range(n_up, n_electrons))
    else:
        majority = tuple(range(n_up, n_electrons))
        minority = tuple(range(n_up))
    sz_abs = abs(n_up - n_down) * 0.5
    base = (sz_abs * (sz_abs + 1.0) + len(minority)) * value_fn(data)
    pairs = tuple((min_idx, maj_idx) for min_idx in minority for maj_idx in majority)
    if not pairs:
        return base

    all_indices = jnp.arange(n_electrons, dtype=jnp.int32)
    electrons = data.electrons

    swapped_sum = 0.0
    for min_idx, maj_idx in pairs:
        perm = all_indices.at[min_idx].set(maj_idx)
        perm = perm.at[maj_idx].set(min_idx)
        swapped_sum = swapped_sum + value_fn(data.merge({"electrons": electrons[perm]}))
    return base - swapped_sum


def spatial_project_value(
    value_fn: MoleculeValueFn,
    data: MoleculeData,
    projector: SpatialProjector,
    *,
    chunk_size: int | None = _DEFAULT_PROJECTOR_CHUNK_SIZE,
) -> jnp.ndarray:
    """Apply a finite spatial projector to ``value_fn`` at ``data``."""
    if projector.is_identity:
        return value_fn(data)
    matrices = jnp.asarray(projector.matrices, dtype=data.electrons.dtype)
    coefficients = jnp.asarray(
        projector.coefficients,
        dtype=jnp.result_type(data.electrons.dtype, 1j),
    )
    term_count = len(projector.matrices)
    chunk = _projector_chunk_size(chunk_size, term_count)
    if term_count <= chunk:
        return _spatial_project_value_chunk(
            value_fn,
            data,
            matrices,
            coefficients,
            projector.origin,
        )

    padded_count = ((term_count + chunk - 1) // chunk) * chunk
    pad_count = padded_count - term_count
    if pad_count:
        identity = jnp.eye(3, dtype=matrices.dtype)
        matrices = jnp.concatenate(
            [matrices, jnp.broadcast_to(identity, (pad_count, 3, 3))],
            axis=0,
        )
        coefficients = jnp.pad(coefficients, (0, pad_count))

    def body(total, chunk_index):
        start = chunk_index * chunk
        matrix_chunk = jax.lax.dynamic_slice(matrices, (start, 0, 0), (chunk, 3, 3))
        coefficient_chunk = jax.lax.dynamic_slice(coefficients, (start,), (chunk,))
        contribution = _spatial_project_value_chunk(
            value_fn,
            data,
            matrix_chunk,
            coefficient_chunk,
            projector.origin,
        )
        return total + contribution, None

    total = _varying_scan_carry(jnp.asarray(0.0, dtype=coefficients.dtype))
    total, _ = jax.lax.scan(
        body,
        total,
        jnp.arange(padded_count // chunk, dtype=jnp.int32),
    )
    return total


def project_value(
    value_fn: MoleculeValueFn,
    data: MoleculeData,
    projector: SymmetryProjector,
    *,
    chunk_size: int | None = _DEFAULT_PROJECTOR_CHUNK_SIZE,
) -> jnp.ndarray:
    """Apply ``P_Gamma P_S`` to a value function."""

    def spin_projected(local_data: MoleculeData) -> jnp.ndarray:
        return spin_project_value(value_fn, local_data, projector.spin)

    return spatial_project_value(
        spin_projected,
        data,
        projector.spatial,
        chunk_size=chunk_size,
    )


def projected_log_apply(
    raw_log_apply: MoleculeLogApply,
    projector: SymmetryProjector,
    *,
    eps: float = 1e-12,
    chunk_size: int | None = _DEFAULT_PROJECTOR_CHUNK_SIZE,
) -> MoleculeLogApply:
    """Wrap a log-amplitude function with hard symmetry projection."""
    if projector.is_identity:
        return raw_log_apply

    def apply(params: Params, data: MoleculeData) -> jnp.ndarray:
        def value_fn(local_data: MoleculeData) -> jnp.ndarray:
            return jnp.exp(raw_log_apply(params, local_data))

        return safe_complex_log(
            project_value(value_fn, data, projector, chunk_size=chunk_size),
            eps=eps,
        )

    return apply


def safe_complex_log(value: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    """Return a numerically guarded complex logarithm."""
    real_dtype = jnp.real(value).dtype
    floor = jnp.asarray(eps, dtype=real_dtype)
    magnitude = jnp.maximum(jnp.abs(value), floor)
    return jnp.log(magnitude) + 1j * jnp.asarray(jnp.angle(value), dtype=real_dtype)


def transform_electrons(
    data: MoleculeData,
    matrix: Sequence[Sequence[float]],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> MoleculeData:
    """Evaluate coordinates at ``Q^{-1} R`` for a row-vector convention."""
    transform = jnp.asarray(matrix, dtype=data.electrons.dtype)
    center = jnp.asarray(origin, dtype=data.electrons.dtype)
    electrons = (data.electrons - center) @ transform + center
    return data.merge({"electrons": electrons})


def transform_electrons_many(
    data: MoleculeData,
    matrices: jnp.ndarray,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> MoleculeData:
    """Evaluate coordinates for a batch of spatial operations."""
    transforms = jnp.asarray(matrices, dtype=data.electrons.dtype)
    center = jnp.asarray(origin, dtype=data.electrons.dtype)
    electrons = jnp.einsum("...d,tdf->t...f", data.electrons - center, transforms)
    return data.merge({"electrons": electrons + center})


def _spatial_project_value_chunk(
    value_fn: MoleculeValueFn,
    data: MoleculeData,
    matrices: jnp.ndarray,
    coefficients: jnp.ndarray,
    origin: Sequence[float],
) -> jnp.ndarray:
    transformed = transform_electrons_many(data, matrices, origin)
    values = jax.vmap(
        value_fn,
        in_axes=(MoleculeData(electrons=0, atoms=None, charges=None),),
    )(transformed)
    return jnp.sum(coefficients * values, axis=0)


def _projector_chunk_size(chunk_size: int | None, term_count: int) -> int:
    if chunk_size is None or int(chunk_size) <= 0:
        return max(1, int(term_count))
    return min(max(1, int(chunk_size)), max(1, int(term_count)))


def _varying_scan_carry(value: jnp.ndarray) -> jnp.ndarray:
    if hasattr(jax.lax, "pcast"):
        try:
            return jax.lax.pcast(value, BATCH_AXIS_NAME, to="varying")
        except (NameError, ValueError):
            return value
    if hasattr(jax.lax, "pvary"):
        try:
            return jax.lax.pvary(value, BATCH_AXIS_NAME)
        except (NameError, ValueError):
            return value
    return value


def _spatial_projector_selection_candidates(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    mode: str,
    axis: int,
    origin: np.ndarray,
    tolerance: float,
    so3_quadrature_order: int,
    so2_quadrature_order: int,
) -> tuple[SpatialProjector, ...]:
    rank = _geometry_rank(atoms, origin, tolerance)
    if rank == 0:
        return (
            atomic_angular_spatial_projector(
                0,
                parity="even",
                origin=origin,
                quadrature_order=so3_quadrature_order,
            ),
            atomic_angular_spatial_projector(
                1,
                parity="odd",
                origin=origin,
                quadrature_order=so3_quadrature_order,
            ),
        )
    if rank == 1:
        ground = _ground_spatial_projector_candidates(
            atoms,
            charges,
            mode=mode,
            origin=origin,
            tolerance=tolerance,
            so3_quadrature_order=so3_quadrature_order,
            so2_quadrature_order=so2_quadrature_order,
        )
        response = _linear_dipole_spatial_projectors(
            atoms,
            charges,
            axis=axis,
            origin=origin,
            tolerance=tolerance,
            quadrature_order=so2_quadrature_order,
        )
        return _unique_spatial_projectors((*ground, *response))
    return finite_spatial_irrep_projectors(
        atoms,
        charges,
        origin=origin,
        tolerance=tolerance,
    )


def _ground_spatial_projector_candidates(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    mode: str,
    origin: np.ndarray,
    tolerance: float,
    so3_quadrature_order: int,
    so2_quadrature_order: int,
) -> tuple[SpatialProjector, ...]:
    del mode
    rank = _geometry_rank(atoms, origin, tolerance)
    if rank == 0:
        return (
            atomic_angular_spatial_projector(
                0,
                parity="even",
                origin=origin,
                quadrature_order=so3_quadrature_order,
            ),
        )
    if rank == 1:
        parity = (
            "g"
            if _operation_maps_atoms(-np.eye(3), atoms, charges, origin, tolerance)
            else None
        )
        return (
            linear_spatial_projector(
                0,
                _linear_geometry_axis(atoms, origin, tolerance),
                parity=parity,
                origin=origin,
                quadrature_order=so2_quadrature_order,
            ),
        )
    return finite_spatial_irrep_projectors(
        atoms,
        charges,
        origin=origin,
        tolerance=tolerance,
    )


def _linear_dipole_spatial_projectors(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    axis: int,
    origin: np.ndarray,
    tolerance: float,
    quadrature_order: int,
) -> tuple[SpatialProjector, ...]:
    molecular_axis = _linear_geometry_axis(atoms, origin, tolerance)
    cart_axis = np.eye(3)[int(axis)]
    signed_parallel = float(np.dot(molecular_axis, cart_axis))
    parallel_weight = abs(signed_parallel)
    perpendicular_weight = float(
        np.linalg.norm(cart_axis - signed_parallel * molecular_axis)
    )
    parity = (
        "u"
        if _operation_maps_atoms(-np.eye(3), atoms, charges, origin, tolerance)
        else None
    )
    projectors = []
    if parallel_weight > tolerance:
        projectors.append(
            linear_spatial_projector(
                0,
                molecular_axis,
                parity=parity,
                origin=origin,
                quadrature_order=quadrature_order,
            )
        )
    if perpendicular_weight > tolerance:
        projectors.append(
            linear_spatial_projector(
                1,
                molecular_axis,
                parity=parity,
                origin=origin,
                quadrature_order=quadrature_order,
            )
        )
    return _unique_spatial_projectors(tuple(projectors))


def _unique_spatial_projectors(
    projectors: tuple[SpatialProjector, ...],
) -> tuple[SpatialProjector, ...]:
    unique = []
    seen = set()
    for projector in projectors:
        key = _normalize_projector_label(projector.label)
        if key in seen:
            continue
        unique.append(projector)
        seen.add(key)
    return tuple(unique)


def _with_parity(
    projector: SpatialProjector,
    parity: str,
    *,
    suffix: str,
    group_label: str,
) -> SpatialProjector:
    sign = 1.0 if parity == "even" else -1.0
    matrices = []
    coefficients = []
    for matrix, coefficient in zip(
        projector.matrices,
        projector.coefficients,
        strict=True,
    ):
        arr = np.asarray(matrix, dtype=np.float64)
        matrices.extend((_matrix_tuple(arr), _matrix_tuple(-arr)))
        coefficients.extend(
            (
                0.5 * complex(coefficient),
                0.5 * sign * complex(coefficient),
            )
        )
    return SpatialProjector(
        label=f"{projector.label}_{suffix}",
        matrices=tuple(matrices),
        coefficients=tuple(coefficients),
        origin=projector.origin,
        dimension=projector.dimension,
        group_label=group_label,
    )


def _so3_character_projector_terms(
    angular_momentum: int,
    quadrature_order: int,
) -> tuple[
    tuple[tuple[tuple[float, float, float], ...], ...],
    tuple[complex, ...],
]:
    l_value = int(angular_momentum)
    order = _positive_quadrature_order(quadrature_order, "SO(3)")
    n_alpha = max(order, 2 * l_value + 3)
    n_beta = max(order, l_value + 2)
    n_gamma = n_alpha
    beta_nodes, beta_weights = np.polynomial.legendre.leggauss(n_beta)
    alpha_weight = 2.0 * np.pi / float(n_alpha)
    gamma_weight = 2.0 * np.pi / float(n_gamma)
    prefactor = float(2 * l_value + 1) / (8.0 * np.pi**2)
    matrices = []
    coefficients = []
    for alpha_index in range(n_alpha):
        alpha = 2.0 * np.pi * float(alpha_index) / float(n_alpha)
        for beta_x, beta_weight in zip(beta_nodes, beta_weights, strict=True):
            beta = float(np.arccos(float(beta_x)))
            coefficient_beta = prefactor * alpha_weight * gamma_weight
            coefficient_beta *= float(beta_weight)
            for gamma_index in range(n_gamma):
                gamma = 2.0 * np.pi * float(gamma_index) / float(n_gamma)
                matrix = _euler_zyz_matrix(alpha, beta, gamma)
                character = _so3_character(l_value, _rotation_angle(matrix))
                matrices.append(_matrix_tuple(matrix))
                coefficients.append(complex(coefficient_beta * character))
    return tuple(matrices), tuple(coefficients)


def _so3_character(angular_momentum: int, angle: float) -> float:
    denominator = np.sin(0.5 * angle)
    if abs(float(denominator)) < 1e-12:
        return float(2 * int(angular_momentum) + 1)
    numerator = np.sin((float(angular_momentum) + 0.5) * angle)
    return float(numerator / denominator)


def _euler_zyz_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    return _rotation_z(alpha) @ _rotation_y(beta) @ _rotation_z(gamma)


def _rotation_angle(matrix: np.ndarray) -> float:
    cosine = 0.5 * (float(np.trace(matrix)) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def _axis_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = _unit_vector(axis)
    x_value, y_value, z_value = unit
    cosine = np.cos(angle)
    sine = np.sin(angle)
    cross = np.asarray(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=np.float64,
    )
    outer = np.outer(unit, unit)
    return cosine * np.eye(3) + sine * cross + (1.0 - cosine) * outer


def _linear_geometry_axis(
    atoms: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    centered = np.asarray(atoms, dtype=np.float64) - origin
    norms = np.linalg.norm(centered, axis=1)
    if norms.size == 0 or float(np.max(norms)) <= tolerance:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return _unit_vector(centered[int(np.argmax(norms))])


def _positive_quadrature_order(order: int, group_label: str) -> int:
    value = int(order)
    if value < 1:
        msg = f"{group_label} quadrature order must be positive, got {order!r}."
        raise ValueError(msg)
    return value


def _signed_permutation_axis_projector(
    atoms: np.ndarray,
    charges: np.ndarray,
    *,
    axis: int,
    origin: np.ndarray,
    tolerance: float,
) -> SpatialProjector:
    matrices = []
    coefficients = []
    for matrix in _signed_permutation_matrices():
        if not _operation_maps_atoms(matrix, atoms, charges, origin, tolerance):
            continue
        image = matrix[:, int(axis)]
        nonzero = np.flatnonzero(np.abs(image) > 0.5)
        if nonzero.size != 1 or int(nonzero[0]) != int(axis):
            continue
        matrices.append(_matrix_tuple(matrix))
        coefficients.append(complex(float(image[int(axis)])) / 1.0)
    if not matrices:
        return identity_spatial_projector("c1")
    scale = 1.0 / float(len(matrices))
    return SpatialProjector(
        label=f"signed_permutation_axis_{int(axis)}",
        matrices=tuple(matrices),
        coefficients=tuple(scale * coefficient for coefficient in coefficients),
        origin=tuple(float(x) for x in origin),
        group_label=f"signed_perm_axis_{int(axis)}",
    )


def _totally_symmetric_projector(
    projectors: tuple[SpatialProjector, ...],
) -> SpatialProjector:
    for projector in projectors:
        coefficients = np.asarray(projector.coefficients, dtype=np.complex128)
        if projector.dimension == 1 and np.allclose(
            coefficients,
            np.full_like(coefficients, 1.0 / len(coefficients)),
            atol=5e-8,
        ):
            return projector
    return projectors[0]


def _projector_sort_key(projector: SpatialProjector) -> tuple[int, int, str]:
    coefficients = np.asarray(projector.coefficients, dtype=np.complex128)
    totally_symmetric = int(
        not (
            projector.dimension == 1
            and np.allclose(
                coefficients,
                np.full_like(coefficients, 1.0 / len(coefficients)),
                atol=5e-8,
            )
        )
    )
    return (totally_symmetric, projector.dimension, projector.label)


def _irrep_label(
    irrep_index: int,
    dimension: int,
    characters: np.ndarray,
    classes: tuple[tuple[int, ...], ...],
    operations: tuple[np.ndarray, ...],
) -> str:
    normalized = characters / max(float(dimension), 1.0)
    if dimension == 1 and np.allclose(normalized, 1.0, atol=5e-7):
        base = "A1"
    else:
        base = f"irrep_{irrep_index:02d}_d{dimension}"
    inversion_class = _operation_class_index(-np.eye(3), operations, classes)
    if inversion_class is not None:
        parity = np.real_if_close(normalized[inversion_class])
        if np.isclose(parity, 1.0, atol=5e-7):
            base = f"{base}_g"
        elif np.isclose(parity, -1.0, atol=5e-7):
            base = f"{base}_u"
    return base


def _normalize_projector_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.startswith("irrep:"):
        normalized = normalized.split(":", 1)[1].strip()
    return normalized.replace("-", "_")


def _operation_class_index(
    matrix: np.ndarray,
    operations: tuple[np.ndarray, ...],
    classes: tuple[tuple[int, ...], ...],
) -> int | None:
    for operation_index, operation in enumerate(operations):
        if np.allclose(operation, matrix, atol=5e-7):
            for class_index, class_members in enumerate(classes):
                if operation_index in class_members:
                    return class_index
    return None


def _discover_point_group_operations(
    atoms: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    atoms = np.asarray(atoms, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    candidates: list[np.ndarray] = [np.eye(3)]
    centered = atoms - origin
    rank = int(np.linalg.matrix_rank(centered, tol=tolerance))
    if rank == 0:
        candidates.extend(_signed_permutation_matrices())
    elif rank == 1:
        candidates.extend(_linear_group_candidates(centered, atoms, charges, origin))
    else:
        candidates.extend(_principal_axis_candidates(centered))
        candidates.extend(_frame_mapping_candidates(centered, charges, rank, tolerance))

    operations: list[np.ndarray] = []
    for candidate in candidates:
        orthogonal_candidate = _orthogonalized(candidate)
        if orthogonal_candidate is None:
            continue
        if not _operation_maps_atoms(
            orthogonal_candidate,
            atoms,
            charges,
            origin,
            tolerance,
        ):
            continue
        _append_unique_operation(operations, orthogonal_candidate, tolerance)
    return _close_group(tuple(operations), atoms, charges, origin, tolerance)


def _principal_axis_candidates(centered: np.ndarray) -> tuple[np.ndarray, ...]:
    covariance = centered.T @ centered
    _, _, vh = np.linalg.svd(covariance)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1.0
    return tuple(basis @ matrix @ basis.T for matrix in _signed_permutation_matrices())


def _linear_group_candidates(
    centered: np.ndarray,
    atoms: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
) -> tuple[np.ndarray, ...]:
    del atoms, charges, origin
    norms = np.linalg.norm(centered, axis=1)
    axis = centered[int(np.argmax(norms))]
    axis = axis / np.linalg.norm(axis)
    basis = _basis_from_axis(axis)
    candidates = []
    for matrix in _signed_permutation_matrices():
        image = matrix @ np.asarray([0.0, 0.0, 1.0])
        if np.count_nonzero(np.abs(image[:2]) > 0.5):
            continue
        candidates.append(basis @ matrix @ basis.T)
    return tuple(candidates)


def _frame_mapping_candidates(  # noqa: C901
    centered: np.ndarray,
    charges: np.ndarray,
    rank: int,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    vectors = [
        (index, vector, float(charges[index]))
        for index, vector in enumerate(centered)
        if np.linalg.norm(vector) > tolerance
    ]
    if rank >= 3:
        reference = _first_independent_tuple(vectors, 3, tolerance)
        if reference is None:
            return ()
        ref_indices, ref_vectors, ref_charges = reference
        ref_frame = np.column_stack(ref_vectors)
        candidates = []
        for target in permutations(vectors, 3):
            target_indices = tuple(item[0] for item in target)
            if len(set(target_indices)) < 3:
                continue
            target_vectors = tuple(item[1] for item in target)
            target_charges = tuple(item[2] for item in target)
            if not _frame_signature_matches(
                ref_vectors,
                ref_charges,
                target_vectors,
                target_charges,
                tolerance,
            ):
                continue
            target_frame = np.column_stack(target_vectors)
            candidates.append(target_frame @ np.linalg.inv(ref_frame))
        del ref_indices
        return tuple(candidates)

    reference = _first_independent_tuple(vectors, 2, tolerance)
    if reference is None:
        return ()
    _, ref_vectors, ref_charges = reference
    ref_normal = _unit_vector(np.cross(ref_vectors[0], ref_vectors[1]))
    ref_frame = np.column_stack([ref_vectors[0], ref_vectors[1], ref_normal])
    candidates = []
    for target in permutations(vectors, 2):
        if target[0][0] == target[1][0]:
            continue
        target_vectors = (target[0][1], target[1][1])
        target_charges = (target[0][2], target[1][2])
        if not _frame_signature_matches(
            ref_vectors,
            ref_charges,
            target_vectors,
            target_charges,
            tolerance,
        ):
            continue
        normal = _unit_vector(np.cross(target_vectors[0], target_vectors[1]))
        for sign in (-1.0, 1.0):
            target_frame = np.column_stack(
                [target_vectors[0], target_vectors[1], sign * normal]
            )
            candidates.append(target_frame @ np.linalg.inv(ref_frame))
    return tuple(candidates)


def _first_independent_tuple(
    vectors: list[tuple[int, np.ndarray, float]],
    size: int,
    tolerance: float,
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], tuple[float, ...]] | None:
    for combo in combinations(vectors, size):
        matrix = np.column_stack([item[1] for item in combo])
        if np.linalg.matrix_rank(matrix, tol=tolerance) < size:
            continue
        return (
            tuple(item[0] for item in combo),
            tuple(item[1] for item in combo),
            tuple(item[2] for item in combo),
        )
    return None


def _frame_signature_matches(
    ref_vectors: tuple[np.ndarray, ...],
    ref_charges: tuple[float, ...],
    target_vectors: tuple[np.ndarray, ...],
    target_charges: tuple[float, ...],
    tolerance: float,
) -> bool:
    if any(
        not np.isclose(ref, target, atol=tolerance, rtol=0.0)
        for ref, target in zip(ref_charges, target_charges, strict=True)
    ):
        return False
    ref_gram = np.asarray(ref_vectors) @ np.asarray(ref_vectors).T
    target_gram = np.asarray(target_vectors) @ np.asarray(target_vectors).T
    return bool(np.allclose(ref_gram, target_gram, atol=tolerance, rtol=0.0))


def _basis_from_axis(axis: np.ndarray) -> np.ndarray:
    z_axis = _unit_vector(axis)
    trial = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, z_axis))) > 0.8:
        trial = np.asarray([0.0, 1.0, 0.0])
    x_axis = _unit_vector(trial - np.dot(trial, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        msg = "Cannot normalize a zero vector."
        raise ValueError(msg)
    return np.asarray(vector, dtype=np.float64) / norm


def _orthogonalized(matrix: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        return None
    u, _, vh = np.linalg.svd(matrix)
    candidate = u @ vh
    if np.linalg.det(candidate) * np.linalg.det(matrix) < 0:
        u[:, -1] *= -1.0
        candidate = u @ vh
    if not np.allclose(candidate.T @ candidate, np.eye(3), atol=1e-7):
        return None
    return candidate


def _append_unique_operation(
    operations: list[np.ndarray],
    candidate: np.ndarray,
    tolerance: float,
) -> None:
    for operation in operations:
        if np.allclose(operation, candidate, atol=max(tolerance, 1e-7)):
            return
    operations.append(candidate)


def _close_group(
    initial_operations: tuple[np.ndarray, ...],
    atoms: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, ...]:
    operations = list(initial_operations)
    changed = True
    while changed:
        changed = False
        current = tuple(operations)
        for left in current:
            for right in current:
                candidate = _orthogonalized(left @ right)
                if candidate is None:
                    continue
                if not _operation_maps_atoms(
                    candidate,
                    atoms,
                    charges,
                    origin,
                    tolerance,
                ):
                    continue
                before = len(operations)
                _append_unique_operation(operations, candidate, tolerance)
                changed = changed or len(operations) > before
        if len(operations) > 192:
            msg = (
                "Discovered an unexpectedly large finite point-group operation "
                "set; check the molecular geometry or loosen symmetry projection."
            )
            raise ValueError(msg)
    identity_index = next(
        (
            index
            for index, operation in enumerate(operations)
            if np.allclose(operation, np.eye(3), atol=max(tolerance, 1e-7))
        ),
        None,
    )
    if identity_index is None:
        operations.insert(0, np.eye(3))
    elif identity_index != 0:
        operations.insert(0, operations.pop(identity_index))
    return tuple(operations)


def _multiplication_table(
    operations: tuple[np.ndarray, ...],
    tolerance: float,
) -> np.ndarray:
    order = len(operations)
    table = np.zeros((order, order), dtype=np.int64)
    for left_index, left in enumerate(operations):
        for right_index, right in enumerate(operations):
            product_matrix = left @ right
            match = _find_operation_index(product_matrix, operations, tolerance)
            if match is None:
                msg = "Point-group operation set is not closed under multiplication."
                raise ValueError(msg)
            table[left_index, right_index] = match
    return table


def _find_operation_index(
    matrix: np.ndarray,
    operations: tuple[np.ndarray, ...],
    tolerance: float,
) -> int | None:
    for index, operation in enumerate(operations):
        if np.allclose(operation, matrix, atol=max(tolerance, 1e-7)):
            return index
    return None


def _conjugacy_classes(multiplication: np.ndarray) -> tuple[tuple[int, ...], ...]:
    order = multiplication.shape[0]
    inverses = _inverse_indices(multiplication)
    unseen = set(range(order))
    classes = []
    while unseen:
        element = min(unseen)
        class_members = {
            int(multiplication[int(multiplication[g, element]), inverses[g]])
            for g in range(order)
        }
        class_tuple = tuple(sorted(class_members))
        classes.append(class_tuple)
        unseen.difference_update(class_tuple)
    return tuple(classes)


def _inverse_indices(multiplication: np.ndarray) -> np.ndarray:
    order = multiplication.shape[0]
    inverses = np.zeros(order, dtype=np.int64)
    for index in range(order):
        matches = np.flatnonzero(
            (multiplication[index] == 0) & (multiplication[:, index] == 0)
        )
        if matches.size != 1:
            msg = "Could not identify a unique inverse in the point group."
            raise ValueError(msg)
        inverses[index] = int(matches[0])
    return inverses


def _class_irreducible_characters(  # noqa: C901
    multiplication: np.ndarray,
    classes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, np.ndarray], ...]:
    class_count = len(classes)
    class_sizes = np.asarray([len(class_members) for class_members in classes])
    class_of = np.zeros(multiplication.shape[0], dtype=np.int64)
    for class_index, class_members in enumerate(classes):
        for member in class_members:
            class_of[member] = class_index

    class_multiplication = []
    for left_class in classes:
        matrix = np.zeros((class_count, class_count), dtype=np.float64)
        for col, right_class in enumerate(classes):
            counts = np.zeros(class_count, dtype=np.float64)
            for left in left_class:
                for right in right_class:
                    counts[class_of[multiplication[left, right]]] += 1.0
            matrix[:, col] = counts / class_sizes
        class_multiplication.append(matrix)

    weights = np.sqrt(np.arange(1, class_count + 1, dtype=np.float64))
    combined = sum(
        weight * matrix for weight, matrix in zip(weights, class_multiplication)
    )
    _, eigenvectors = np.linalg.eig(combined)
    order = int(multiplication.shape[0])
    irreps = []
    for column in range(class_count):
        vector = eigenvectors[:, column]
        if np.linalg.norm(vector) <= 0.0:
            continue
        lambdas = np.asarray(
            [_class_eigenvalue(matrix, vector) for matrix in class_multiplication],
            dtype=np.complex128,
        )
        dimension = np.sqrt(
            float(order) / float(np.real(np.sum((np.abs(lambdas) ** 2) / class_sizes)))
        )
        dimension_int = max(1, round(float(np.real_if_close(dimension))))
        characters = dimension_int * lambdas / class_sizes
        characters = np.real_if_close(characters, tol=1000)
        irreps.append((dimension_int, np.asarray(characters, dtype=np.complex128)))

    unique_irreps = []
    for dimension, characters in irreps:
        normalized = np.concatenate(
            [
                np.asarray([dimension], dtype=np.complex128),
                np.round(characters, decimals=10),
            ]
        )
        if any(
            other_dimension == dimension
            and np.allclose(other_characters, characters, atol=5e-7)
            for other_dimension, other_characters in unique_irreps
        ):
            continue
        unique_irreps.append((dimension, characters))
        del normalized
    if sum(dimension * dimension for dimension, _ in unique_irreps) != order:
        msg = "Failed to decompose point-group class algebra into irreps."
        raise ValueError(msg)
    return tuple(unique_irreps)


def _class_eigenvalue(matrix: np.ndarray, vector: np.ndarray) -> complex:
    image = matrix @ vector
    index = int(np.argmax(np.abs(vector)))
    if abs(vector[index]) < 1e-12:
        return complex(np.vdot(vector, image) / np.vdot(vector, vector))
    return complex(image[index] / vector[index])


def _signed_permutation_matrices() -> tuple[np.ndarray, ...]:
    matrices = []
    for perm in permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            base[row, col] = 1.0
        for signs in product((-1.0, 1.0), repeat=3):
            matrices.append(np.diag(signs) @ base)
    return tuple(matrices)


def _symmetry_origin(atoms: np.ndarray) -> np.ndarray:
    if atoms.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.asarray(atoms, dtype=np.float64), axis=0)


def _geometry_rank(
    atoms: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> int:
    atoms = np.asarray(atoms, dtype=np.float64)
    if atoms.size == 0:
        return 0
    return int(np.linalg.matrix_rank(atoms - origin, tol=tolerance))


def _require_operation_symmetry(
    matrix: np.ndarray,
    atoms: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> None:
    if not _operation_maps_atoms(matrix, atoms, charges, origin, tolerance):
        msg = "Requested spatial projector is not a symmetry of the nuclear geometry."
        raise ValueError(msg)


def _operation_maps_atoms(
    matrix: np.ndarray,
    atoms: np.ndarray,
    charges: np.ndarray,
    origin: np.ndarray,
    tolerance: float,
) -> bool:
    atoms = np.asarray(atoms, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    transformed = (atoms - origin) @ np.asarray(matrix, dtype=np.float64).T + origin
    unmatched = list(range(atoms.shape[0]))
    for position, charge in zip(transformed, charges, strict=True):
        match_index = None
        for candidate in unmatched:
            if not np.isclose(charge, charges[candidate], atol=tolerance, rtol=0.0):
                continue
            if np.linalg.norm(position - atoms[candidate]) <= tolerance:
                match_index = candidate
                break
        if match_index is None:
            return False
        unmatched.remove(match_index)
    return True


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    arr = np.asarray(matrix, dtype=np.float64)
    return tuple(tuple(float(x) for x in row) for row in arr)
