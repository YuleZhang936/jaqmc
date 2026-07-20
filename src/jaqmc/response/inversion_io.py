# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""Load and combine signed-LIT workflow outputs for formal inversion.

The workflow writes one NPZ file per broadening-width calculation.  Formal
inversion should fit all widths simultaneously and retain correlations caused
by evaluating every frequency on the same held-out Monte Carlo pool.  This
module performs the deliberately strict validation needed before those files
are treated as one statistical data set.

``signed_lit`` is always taken from the full-pool ratio-of-sums estimator in
each NPZ.  Matched jackknife pseudo-values are used only to estimate its
statistical covariance; their arithmetic mean never replaces the full-pool
result.  Following the PRL likelihood, the fidelity-bound systematic error is
then added pointwise in quadrature.  This diagonal systematic model is the
paper's explicit approximation, not a claim that optimization errors at
different frequencies are physically independent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from jaqmc.response.inversion import (
    LITInversionResult,
    _covariances,
    _one_dimensional_finite_array,
    _pole_bounds,
    _validate_model_components,
    _validate_solver_options,
    invert_signed_lit,
    lit_block_statistics,
)


@dataclass(frozen=True)
class LITNPZMetadata:
    """Audit metadata for one file in an aggregated LIT data set."""

    path: str
    axes: str
    axis_indices: tuple[int, ...]
    observation_start: int
    observation_stop: int
    eta_values: tuple[float, ...]
    eval_pool_sha256: tuple[str, ...] | None
    block_count: int | None
    has_covariance: bool
    has_systematic_error: bool


@dataclass(frozen=True)
class AggregatedLITNPZ:
    """Validated, flattened input for :func:`invert_signed_lit`.

    ``signed_lit`` has shape ``(n_axes, n_observations)``.  ``covariance`` is
    the paper-style total covariance: correlated ``statistical_covariance``
    plus ``diag(systematic_error**2)``.  When matched pseudo-values were
    available, ``block_estimates`` has shape
    ``(n_axes, n_observations, n_blocks)``.  It is ``None`` for the explicit
    independent-file fallback.
    """

    omega: NDArray[np.float64]
    eta: NDArray[np.float64]
    signed_lit: NDArray[np.float64]
    covariance: NDArray[np.float64]
    statistical_covariance: NDArray[np.float64]
    systematic_error: NDArray[np.float64]
    block_estimates: NDArray[np.float64] | None
    axes: str
    axis_indices: NDArray[np.int64]
    covariance_mode: Literal["matched_blocks", "independent_files"]
    metadata: tuple[LITNPZMetadata, ...]


@dataclass(frozen=True)
class LITInversionSettings:
    """Validated, reproducible settings for a formal NPZ inversion."""

    threshold: float
    pole_energies: ArrayLike = ()
    continuum_grid: ArrayLike = ()
    continuum_regularization: float = 0.0
    fit_pole_energies: bool = False
    pole_energy_bounds: ArrayLike | None = None
    covariance_relative_tolerance: float = 1e-10
    max_fitted_poles: int = 8
    pole_fit_tolerance: float = 1e-7
    pole_fit_max_iterations: int = 200
    solver_tolerance: float = 1e-10
    solver_max_iterations: int | None = None

    def __post_init__(self) -> None:
        energies = _one_dimensional_finite_array(
            self.pole_energies,
            "pole_energies",
        ).copy()
        grid = _one_dimensional_finite_array(
            self.continuum_grid,
            "continuum_grid",
        ).copy()
        _validate_model_components(float(self.threshold), energies, grid)
        _validate_solver_options(
            int(self.max_fitted_poles),
            float(self.solver_tolerance),
            float(self.pole_fit_tolerance),
            float(self.covariance_relative_tolerance),
        )
        regularization = float(self.continuum_regularization)
        if not np.isfinite(regularization) or regularization < 0.0:
            msg = (
                "continuum_regularization must be finite and nonnegative, got "
                f"{self.continuum_regularization}"
            )
            raise ValueError(msg)
        if (
            isinstance(self.pole_fit_max_iterations, (bool, np.bool_))
            or int(self.pole_fit_max_iterations) < 1
        ):
            msg = "pole_fit_max_iterations must be a positive integer"
            raise ValueError(msg)
        solver_iterations = self.solver_max_iterations
        if solver_iterations is not None and (
            isinstance(solver_iterations, (bool, np.bool_))
            or int(solver_iterations) < 1
        ):
            msg = "solver_max_iterations must be a positive integer or None"
            raise ValueError(msg)

        bounds: NDArray[np.float64]
        if self.pole_energy_bounds is None:
            bounds = np.empty((0, 2), dtype=np.float64)
        else:
            bounds = np.asarray(self.pole_energy_bounds, dtype=np.float64)
            if bounds.size == 0:
                bounds = np.empty((0, 2), dtype=np.float64)
        if bool(self.fit_pole_energies):
            bounds = _pole_bounds(
                None if bounds.size == 0 else bounds,
                energies,
                float(self.threshold),
            ).copy()
        elif bounds.size:
            msg = "pole_energy_bounds require fit_pole_energies=true"
            raise ValueError(msg)

        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "pole_energies", energies)
        object.__setattr__(self, "continuum_grid", grid)
        object.__setattr__(self, "continuum_regularization", regularization)
        object.__setattr__(self, "fit_pole_energies", bool(self.fit_pole_energies))
        object.__setattr__(self, "pole_energy_bounds", bounds)
        object.__setattr__(
            self,
            "covariance_relative_tolerance",
            float(self.covariance_relative_tolerance),
        )
        object.__setattr__(self, "max_fitted_poles", int(self.max_fitted_poles))
        object.__setattr__(
            self,
            "pole_fit_tolerance",
            float(self.pole_fit_tolerance),
        )
        object.__setattr__(
            self,
            "pole_fit_max_iterations",
            int(self.pole_fit_max_iterations),
        )
        object.__setattr__(self, "solver_tolerance", float(self.solver_tolerance))
        if solver_iterations is not None:
            object.__setattr__(self, "solver_max_iterations", int(solver_iterations))


@dataclass(frozen=True)
class LITInversionJackknife:
    """Delete-one-block uncertainty of nonlinear inversion outputs."""

    block_count: int
    leave_one_out_pole_energies: NDArray[np.float64]
    pole_energy_standard_error: NDArray[np.float64]
    pole_energy_bias: NDArray[np.float64]
    bias_corrected_pole_energies: NDArray[np.float64]
    leave_one_out_pole_strengths: NDArray[np.float64]
    pole_strength_standard_error: NDArray[np.float64]
    pole_strength_bias: NDArray[np.float64]
    bias_corrected_pole_strengths: NDArray[np.float64]
    leave_one_out_continuum_density: NDArray[np.float64]
    continuum_density_standard_error: NDArray[np.float64]
    continuum_density_bias: NDArray[np.float64]
    bias_corrected_continuum_density: NDArray[np.float64]


@dataclass(frozen=True)
class LITNPZInversion:
    """Formal inversion together with its validated inputs and uncertainty."""

    data: AggregatedLITNPZ
    settings: LITInversionSettings
    result: LITInversionResult
    jackknife: LITInversionJackknife | None


@dataclass(frozen=True)
class _LoadedLITNPZ:
    path: str
    omega: NDArray[np.float64]
    eta: NDArray[np.float64]
    signed_lit: NDArray[np.float64]
    axes: str
    axis_indices: NDArray[np.int64]
    blocks: NDArray[np.float64] | None
    eval_pool_sha256: tuple[str, ...] | None
    covariance: NDArray[np.float64] | None
    systematic_error: NDArray[np.float64]


def _normalize_paths(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
) -> tuple[str | PathLike[str], ...]:
    normalized: tuple[str | PathLike[str], ...]
    if isinstance(paths, (str, PathLike)):
        normalized = (paths,)
    elif isinstance(paths, Iterable):
        normalized = tuple(paths)
    else:
        msg = "paths must be an NPZ path or a sequence of NPZ paths"
        raise TypeError(msg)
    if not normalized:
        msg = "at least one workflow NPZ path is required"
        raise ValueError(msg)
    if not all(isinstance(path, (str, PathLike)) for path in normalized):
        msg = "every workflow NPZ path must be a string or os.PathLike"
        raise TypeError(msg)
    return normalized


def _required(archive: np.lib.npyio.NpzFile, name: str, path: str) -> np.ndarray:
    if name not in archive.files:
        msg = f"{path}: workflow NPZ is missing required field {name!r}"
        raise ValueError(msg)
    return np.asarray(archive[name])


def _scalar_text(value: np.ndarray, name: str, path: str) -> str:
    if value.ndim != 0:
        msg = f"{path}: {name} must be a scalar string, got shape {value.shape}"
        raise ValueError(msg)
    scalar = value.item()
    if isinstance(scalar, bytes):
        try:
            scalar = scalar.decode("utf-8")
        except UnicodeDecodeError as error:
            msg = f"{path}: {name} is not valid UTF-8"
            raise ValueError(msg) from error
    if not isinstance(scalar, str) or not scalar:
        msg = f"{path}: {name} must be a nonempty scalar string"
        raise ValueError(msg)
    return scalar


def _load_axes(
    archive: np.lib.npyio.NpzFile,
    path: str,
) -> tuple[str, NDArray[np.int64]]:
    axes = _scalar_text(_required(archive, "axes", path), "axes", path).lower()
    raw_indices = _required(archive, "axis_indices", path)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        msg = f"{path}: axis_indices must be a nonempty vector"
        raise ValueError(msg)
    if not np.issubdtype(raw_indices.dtype, np.integer):
        msg = f"{path}: axis_indices must contain integers"
        raise ValueError(msg)
    indices = np.asarray(raw_indices, dtype=np.int64)
    if np.any((indices < 0) | (indices > 2)):
        msg = f"{path}: axis_indices must contain only Cartesian indices 0, 1, 2"
        raise ValueError(msg)
    if np.unique(indices).size != indices.size:
        msg = f"{path}: axis_indices must not contain duplicate axes"
        raise ValueError(msg)
    expected_axes = "".join("xyz"[index] for index in indices)
    if axes != expected_axes:
        msg = (
            f"{path}: axes={axes!r} is inconsistent with "
            f"axis_indices={indices.tolist()} (expected {expected_axes!r})"
        )
        raise ValueError(msg)
    return axes, indices


def _load_observations(
    archive: np.lib.npyio.NpzFile,
    path: str,
    n_axes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    omega = np.asarray(_required(archive, "omega", path), dtype=np.float64)
    if omega.ndim != 1 or omega.size == 0:
        msg = f"{path}: omega must be a nonempty one-dimensional array"
        raise ValueError(msg)
    if not np.all(np.isfinite(omega)):
        msg = f"{path}: omega must contain only finite values"
        raise ValueError(msg)

    raw_eta = np.asarray(_required(archive, "eta", path), dtype=np.float64)
    if raw_eta.ndim == 0 or raw_eta.size == 1:
        eta = np.full(omega.size, float(raw_eta.reshape(-1)[0]), dtype=np.float64)
    elif raw_eta.ndim == 1 and raw_eta.shape == omega.shape:
        eta = np.array(raw_eta, dtype=np.float64, copy=True)
    else:
        msg = (
            f"{path}: eta must be scalar or have the same shape as omega; "
            f"got {raw_eta.shape} and {omega.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0.0):
        msg = f"{path}: eta must contain only finite, positive values"
        raise ValueError(msg)

    signed_lit = np.asarray(
        _required(archive, "signed_lit", path),
        dtype=np.float64,
    )
    expected_shape = (n_axes, omega.size)
    if signed_lit.shape != expected_shape:
        msg = (
            f"{path}: signed_lit must have shape {expected_shape}, "
            f"got {signed_lit.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(signed_lit)):
        msg = f"{path}: signed_lit must contain only finite values"
        raise ValueError(msg)
    return omega, eta, signed_lit


def _optional_block_count(
    archive: np.lib.npyio.NpzFile,
    path: str,
) -> int | None:
    field = "signed_lit_jackknife_block_count"
    if field not in archive.files:
        return None
    raw = np.asarray(archive[field])
    if raw.ndim != 0 or not np.issubdtype(raw.dtype, np.integer):
        msg = f"{path}: {field} must be a scalar integer"
        raise ValueError(msg)
    return int(raw)


def _load_blocks(
    archive: np.lib.npyio.NpzFile,
    path: str,
    expected_prefix: tuple[int, int],
) -> NDArray[np.float64] | None:
    legacy_fields = (
        "signed_lit_blocks",
        "signed_lit_block_count",
        "block_count",
    )
    present_legacy = [name for name in legacy_fields if name in archive.files]
    if present_legacy:
        msg = (
            f"{path}: unsupported legacy raw-block fields {present_legacy}; "
            "regenerate matched signed_lit_jackknife_blocks"
        )
        raise ValueError(msg)
    field = "signed_lit_jackknife_blocks"
    has_blocks = field in archive.files
    declared_count = _optional_block_count(archive, path)
    if not has_blocks:
        if declared_count is not None:
            msg = f"{path}: block-count metadata is present but block values are absent"
            raise ValueError(msg)
        return None
    blocks = np.asarray(archive[field], dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[:2] != expected_prefix:
        msg = (
            f"{path}: {field} must have shape "
            f"{(*expected_prefix, 'n_blocks')!r}, got {blocks.shape}"
        )
        raise ValueError(msg)
    if blocks.shape[-1] < 2:
        msg = f"{path}: {field} requires at least two blocks"
        raise ValueError(msg)
    if declared_count is not None and declared_count != blocks.shape[-1]:
        msg = (
            f"{path}: declared block count {declared_count} does not match "
            f"{field} shape {blocks.shape[-1]}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(blocks)):
        msg = f"{path}: {field} must contain only finite values"
        raise ValueError(msg)
    return blocks


def _load_systematic_error(
    archive: np.lib.npyio.NpzFile,
    path: str,
    expected_shape: tuple[int, int],
) -> NDArray[np.float64]:
    error = np.asarray(
        _required(archive, "error_bound_monitor", path),
        dtype=np.float64,
    )
    if error.shape != expected_shape:
        msg = (
            f"{path}: error_bound_monitor must have shape {expected_shape}, "
            f"got {error.shape}"
        )
        raise ValueError(msg)
    valid = _required(archive, "error_d_valid", path)
    if valid.shape != expected_shape or not np.issubdtype(valid.dtype, np.bool_):
        msg = f"{path}: error_d_valid must be a boolean array of shape {expected_shape}"
        raise ValueError(msg)
    if not np.all(valid):
        invalid_count = int(np.count_nonzero(~valid))
        msg = f"{path}: {invalid_count} LIT points have invalid fidelity/D error bounds"
        raise ValueError(msg)
    if not np.all(np.isfinite(error)) or np.any(error < 0.0):
        msg = (
            f"{path}: error_bound_monitor must contain only finite, "
            "nonnegative raw-LIT errors"
        )
        raise ValueError(msg)
    return error


def _load_pool_digests(
    archive: np.lib.npyio.NpzFile,
    path: str,
    n_axes: int,
) -> tuple[str, ...] | None:
    if "eval_pool_sha256" not in archive.files:
        return None
    raw = np.asarray(archive["eval_pool_sha256"])
    if raw.ndim == 0:
        raw_values = [raw.item()] * n_axes
    elif raw.ndim == 1 and raw.size == n_axes:
        raw_values = list(raw)
    else:
        msg = (
            f"{path}: eval_pool_sha256 must be scalar or have one value per "
            f"axis ({n_axes}), got {raw.shape}"
        )
        raise ValueError(msg)
    digests: list[str] = []
    for raw_value in raw_values:
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, bytes):
            try:
                value = value.decode("ascii")
            except UnicodeDecodeError as error:
                msg = f"{path}: eval_pool_sha256 must contain ASCII strings"
                raise ValueError(msg) from error
        if not isinstance(value, str) or not value:
            msg = f"{path}: eval_pool_sha256 must contain nonempty strings"
            raise ValueError(msg)
        digests.append(value)
    return tuple(digests)


def _load_covariance(
    archive: np.lib.npyio.NpzFile,
    path: str,
    expected_shape: tuple[int, int, int],
) -> NDArray[np.float64] | None:
    if "signed_lit_covariance" not in archive.files:
        return None
    covariance = np.asarray(archive["signed_lit_covariance"], dtype=np.float64)
    if covariance.shape != expected_shape:
        msg = (
            f"{path}: signed_lit_covariance must have shape {expected_shape}, "
            f"got {covariance.shape}"
        )
        raise ValueError(msg)
    try:
        return _covariances(
            covariance,
            expected_shape[0],
            expected_shape[1],
            relative_tolerance=1e-10,
        )
    except ValueError as error:
        msg = f"{path}: invalid signed_lit_covariance: {error}"
        raise ValueError(msg) from error


def _load_npz(path_like: str | PathLike[str]) -> _LoadedLITNPZ:
    path = str(Path(path_like))
    try:
        with np.load(path, allow_pickle=False) as archive:
            axes, axis_indices = _load_axes(archive, path)
            omega, eta, signed_lit = _load_observations(
                archive,
                path,
                axis_indices.size,
            )
            blocks = _load_blocks(
                archive,
                path,
                signed_lit.shape,
            )
            eval_pool_sha256 = _load_pool_digests(
                archive,
                path,
                axis_indices.size,
            )
            covariance = _load_covariance(
                archive,
                path,
                (axis_indices.size, omega.size, omega.size),
            )
            systematic_error = _load_systematic_error(
                archive,
                path,
                signed_lit.shape,
            )
    except (OSError, EOFError) as error:
        msg = f"could not read workflow NPZ {path}: {error}"
        raise ValueError(msg) from error
    return _LoadedLITNPZ(
        path=path,
        omega=omega,
        eta=eta,
        signed_lit=signed_lit,
        axes=axes,
        axis_indices=axis_indices,
        blocks=blocks,
        eval_pool_sha256=eval_pool_sha256,
        covariance=covariance,
        systematic_error=systematic_error,
    )


def _validate_common_axes(files: tuple[_LoadedLITNPZ, ...]) -> None:
    reference = files[0]
    for current in files[1:]:
        if current.axes != reference.axes or not np.array_equal(
            current.axis_indices,
            reference.axis_indices,
        ):
            msg = (
                "workflow NPZ files use incompatible response axes: "
                f"{reference.path} has {reference.axes!r}/"
                f"{reference.axis_indices.tolist()}, while {current.path} has "
                f"{current.axes!r}/{current.axis_indices.tolist()}"
            )
            raise ValueError(msg)


def _matched_blocks_available(files: tuple[_LoadedLITNPZ, ...]) -> bool:
    if any(file.blocks is None or file.eval_pool_sha256 is None for file in files):
        return False
    block_counts = {file.blocks.shape[-1] for file in files if file.blocks is not None}
    pool_digests = {file.eval_pool_sha256 for file in files}
    return len(block_counts) == 1 and len(pool_digests) == 1


def _independent_covariance(files: tuple[_LoadedLITNPZ, ...]) -> NDArray[np.float64]:
    missing = [file.path for file in files if file.covariance is None]
    if missing:
        msg = (
            "assume_independent=True requires signed_lit_covariance in every "
            f"workflow NPZ; missing from {missing}"
        )
        raise ValueError(msg)
    n_axes = files[0].axis_indices.size
    total_observations = sum(file.omega.size for file in files)
    result = np.zeros(
        (n_axes, total_observations, total_observations),
        dtype=np.float64,
    )
    start = 0
    for file in files:
        stop = start + file.omega.size
        covariance = file.covariance
        if covariance is None:  # Defensive; rejected before allocation above.
            raise RuntimeError("missing covariance after validation")
        result[:, start:stop, start:stop] = covariance
        start = stop
    return result


def aggregate_lit_npz(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    assume_independent: bool = False,
) -> AggregatedLITNPZ:
    """Load workflow NPZs as one correlated, multi-``eta`` inversion input.

    The preferred path requires every file to contain matched
    ``signed_lit_jackknife_blocks``, the same block count, and identical
    ``eval_pool_sha256`` values for every response axis.  Concatenating matched
    pseudo-values before calling
    :func:`lit_block_statistics` preserves correlations across frequencies and
    broadening widths.  Legacy raw-block fields are rejected because they
    cannot be combined consistently with jackknife pseudo-values.

    Every file must also contain finite ``error_bound_monitor`` values with
    ``error_d_valid=true``.  Following PRL Eq. (A9), their squares are added to
    the statistical covariance diagonal.  Cross-frequency statistical
    covariance is retained; the systematic diagonal is the paper's explicit
    likelihood approximation.

    If matching cannot be proven, this function raises by default.  Passing
    ``assume_independent=True`` is an explicit statistical assumption and is
    accepted only when every file contains its own ``signed_lit_covariance``;
    those matrices are then combined block-diagonally.
    """
    normalized_paths = _normalize_paths(paths)
    files = tuple(_load_npz(path) for path in normalized_paths)
    _validate_common_axes(files)

    omega = np.concatenate([file.omega for file in files])
    eta = np.concatenate([file.eta for file in files])
    signed_lit = np.concatenate([file.signed_lit for file in files], axis=1)
    systematic_error = np.concatenate(
        [file.systematic_error for file in files],
        axis=1,
    )

    if _matched_blocks_available(files):
        block_estimates = np.concatenate(
            [file.blocks for file in files if file.blocks is not None],
            axis=1,
        )
        statistical_covariance = lit_block_statistics(block_estimates).covariance
        covariance_mode: Literal["matched_blocks", "independent_files"] = (
            "matched_blocks"
        )
    elif assume_independent:
        block_estimates = None
        statistical_covariance = _independent_covariance(files)
        covariance_mode = "independent_files"
    else:
        descriptions = []
        if any(file.blocks is None for file in files):
            descriptions.append("matched block pseudo-values are missing")
        block_counts = {
            file.blocks.shape[-1] for file in files if file.blocks is not None
        }
        if len(block_counts) > 1:
            descriptions.append(f"block counts differ ({sorted(block_counts)})")
        if any(file.eval_pool_sha256 is None for file in files):
            descriptions.append("eval_pool_sha256 is missing")
        pool_digests = {
            file.eval_pool_sha256 for file in files if file.eval_pool_sha256 is not None
        }
        if len(pool_digests) > 1:
            descriptions.append("eval_pool_sha256 differs between files")
        detail = "; ".join(descriptions) or "matching could not be proven"
        msg = (
            "cannot preserve cross-file LIT covariance: "
            f"{detail}. Regenerate files with matched held-out blocks, or pass "
            "assume_independent=True only when independence is justified and "
            "every file stores signed_lit_covariance."
        )
        raise ValueError(msg)

    covariance = np.array(statistical_covariance, copy=True)
    diagonal = np.arange(covariance.shape[-1])
    covariance[:, diagonal, diagonal] += systematic_error**2
    covariance = _covariances(
        covariance,
        covariance.shape[0],
        covariance.shape[1],
        relative_tolerance=1e-10,
    )

    metadata: list[LITNPZMetadata] = []
    start = 0
    for file in files:
        stop = start + file.omega.size
        metadata.append(
            LITNPZMetadata(
                path=file.path,
                axes=file.axes,
                axis_indices=tuple(int(value) for value in file.axis_indices),
                observation_start=start,
                observation_stop=stop,
                eta_values=tuple(float(value) for value in np.unique(file.eta)),
                eval_pool_sha256=file.eval_pool_sha256,
                block_count=None if file.blocks is None else file.blocks.shape[-1],
                has_covariance=file.covariance is not None,
                has_systematic_error=True,
            )
        )
        start = stop

    reference = files[0]
    return AggregatedLITNPZ(
        omega=omega,
        eta=eta,
        signed_lit=signed_lit,
        covariance=covariance,
        statistical_covariance=statistical_covariance,
        systematic_error=systematic_error,
        block_estimates=block_estimates,
        axes=reference.axes,
        axis_indices=reference.axis_indices.copy(),
        covariance_mode=covariance_mode,
        metadata=tuple(metadata),
    )


def _invert_aggregated_lit(
    data: AggregatedLITNPZ,
    settings: LITInversionSettings,
    signed_lit: NDArray[np.float64],
) -> LITInversionResult:
    bounds = np.asarray(settings.pole_energy_bounds, dtype=np.float64)
    result = invert_signed_lit(
        data.omega,
        data.eta,
        signed_lit,
        threshold=settings.threshold,
        pole_energies=settings.pole_energies,
        continuum_grid=settings.continuum_grid,
        covariance=data.covariance,
        covariance_relative_tolerance=settings.covariance_relative_tolerance,
        continuum_regularization=settings.continuum_regularization,
        fit_pole_energies=settings.fit_pole_energies,
        pole_energy_bounds=None if bounds.size == 0 else bounds,
        max_fitted_poles=settings.max_fitted_poles,
        pole_fit_tolerance=settings.pole_fit_tolerance,
        pole_fit_max_iterations=settings.pole_fit_max_iterations,
        solver_tolerance=settings.solver_tolerance,
        solver_max_iterations=settings.solver_max_iterations,
    )
    if not all(result.diagnostics.solver_success):
        failed = [
            message
            for success, message in zip(
                result.diagnostics.solver_success,
                result.diagnostics.solver_messages,
                strict=True,
            )
            if not success
        ]
        msg = f"formal LIT inversion NNLS failed: {failed}"
        raise RuntimeError(msg)
    if result.diagnostics.pole_fit_success is False:
        msg = (
            f"formal LIT pole-energy fit failed: {result.diagnostics.pole_fit_message}"
        )
        raise RuntimeError(msg)
    return result


def _jackknife_standard_error(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    block_count = values.shape[0]
    centered = values - np.mean(values, axis=0)
    return np.sqrt((block_count - 1) / block_count * np.sum(centered**2, axis=0))


def _jackknife_bias_and_correction(
    full: NDArray[np.float64],
    leave_one_out: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    block_count = leave_one_out.shape[0]
    mean = np.mean(leave_one_out, axis=0)
    bias = (block_count - 1) * (mean - full)
    return bias, full - bias


def _invert_jackknife_blocks(
    data: AggregatedLITNPZ,
    settings: LITInversionSettings,
    full: LITInversionResult,
) -> LITInversionJackknife | None:
    if data.block_estimates is None:
        return None
    block_count = int(data.block_estimates.shape[-1])
    leave_one_out_lit = (
        block_count * data.signed_lit[..., np.newaxis] - data.block_estimates
    ) / (block_count - 1)
    inversions = tuple(
        _invert_aggregated_lit(data, settings, leave_one_out_lit[..., index])
        for index in range(block_count)
    )
    energies = np.stack([value.pole_energies for value in inversions], axis=0)
    strengths = np.stack([value.pole_strengths for value in inversions], axis=0)
    continuum = np.stack(
        [value.continuum_density for value in inversions],
        axis=0,
    )
    energy_bias, corrected_energies = _jackknife_bias_and_correction(
        full.pole_energies,
        energies,
    )
    strength_bias, corrected_strengths = _jackknife_bias_and_correction(
        full.pole_strengths,
        strengths,
    )
    continuum_bias, corrected_continuum = _jackknife_bias_and_correction(
        full.continuum_density,
        continuum,
    )
    return LITInversionJackknife(
        block_count=block_count,
        leave_one_out_pole_energies=energies,
        pole_energy_standard_error=_jackknife_standard_error(energies),
        pole_energy_bias=energy_bias,
        bias_corrected_pole_energies=corrected_energies,
        leave_one_out_pole_strengths=strengths,
        pole_strength_standard_error=_jackknife_standard_error(strengths),
        pole_strength_bias=strength_bias,
        bias_corrected_pole_strengths=corrected_strengths,
        leave_one_out_continuum_density=continuum,
        continuum_density_standard_error=_jackknife_standard_error(continuum),
        continuum_density_bias=continuum_bias,
        bias_corrected_continuum_density=corrected_continuum,
    )


def invert_lit_npz(
    paths: str | PathLike[str] | Sequence[str | PathLike[str]],
    settings: LITInversionSettings,
    *,
    assume_independent: bool = False,
) -> LITNPZInversion:
    """Load workflow NPZs, perform formal inversion, and propagate blocks."""
    data = aggregate_lit_npz(paths, assume_independent=assume_independent)
    result = _invert_aggregated_lit(data, settings, data.signed_lit)
    jackknife = _invert_jackknife_blocks(data, settings, result)
    return LITNPZInversion(
        data=data,
        settings=settings,
        result=result,
        jackknife=jackknife,
    )


def _metadata_json(metadata: tuple[LITNPZMetadata, ...]) -> str:
    records = [
        {
            "path": value.path,
            "axes": value.axes,
            "axis_indices": value.axis_indices,
            "observation_start": value.observation_start,
            "observation_stop": value.observation_stop,
            "eta_values": value.eta_values,
            "eval_pool_sha256": value.eval_pool_sha256,
            "block_count": value.block_count,
            "has_covariance": value.has_covariance,
            "has_systematic_error": value.has_systematic_error,
        }
        for value in metadata
    ]
    return json.dumps(records, sort_keys=True)


def lit_inversion_npz_payload(inversion: LITNPZInversion) -> dict[str, object]:
    """Build a pickle-free, self-describing archive payload."""
    data = inversion.data
    settings = inversion.settings
    result = inversion.result
    diagnostics = result.diagnostics
    bounds = np.asarray(settings.pole_energy_bounds, dtype=np.float64)
    payload: dict[str, object] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "method": np.asarray("jaqmc.response.inversion_io.invert_lit_npz"),
        "source_lit_paths": np.asarray([value.path for value in data.metadata]),
        "source_metadata_json": np.asarray(_metadata_json(data.metadata)),
        "omega": data.omega,
        "eta": data.eta,
        "signed_lit": data.signed_lit,
        "covariance": data.covariance,
        "statistical_covariance": data.statistical_covariance,
        "systematic_error": data.systematic_error,
        "axes": np.asarray(data.axes),
        "axis_indices": data.axis_indices,
        "covariance_mode": np.asarray(data.covariance_mode),
        "threshold": np.asarray(settings.threshold, dtype=np.float64),
        "initial_pole_energies": np.asarray(settings.pole_energies),
        "pole_energy_bounds": bounds,
        "input_continuum_grid": np.asarray(settings.continuum_grid),
        "continuum_regularization": np.asarray(
            settings.continuum_regularization,
            dtype=np.float64,
        ),
        "fit_pole_energies": np.asarray(settings.fit_pole_energies),
        "covariance_relative_tolerance": np.asarray(
            settings.covariance_relative_tolerance,
            dtype=np.float64,
        ),
        "max_fitted_poles": np.asarray(settings.max_fitted_poles, dtype=np.int64),
        "pole_fit_tolerance": np.asarray(
            settings.pole_fit_tolerance,
            dtype=np.float64,
        ),
        "pole_fit_max_iterations": np.asarray(
            settings.pole_fit_max_iterations,
            dtype=np.int64,
        ),
        "solver_tolerance": np.asarray(
            settings.solver_tolerance,
            dtype=np.float64,
        ),
        "solver_max_iterations": np.asarray(
            -1
            if settings.solver_max_iterations is None
            else settings.solver_max_iterations,
            dtype=np.int64,
        ),
        "pole_energies": result.pole_energies,
        "pole_energies_ev": result.pole_energies * 27.211386245981,
        "pole_strengths": result.pole_strengths,
        "continuum_grid": result.continuum_grid,
        "continuum_density": result.continuum_density,
        "fitted_lit": result.fitted_lit,
        "residual": result.residual,
        "residual_norms": diagnostics.residual_norms,
        "weighted_residual_norms": diagnostics.weighted_residual_norms,
        "regularization_norms": diagnostics.regularization_norms,
        "reduced_chi_squared": diagnostics.reduced_chi_squared,
        "condition_numbers": diagnostics.condition_numbers,
        "effective_ranks": diagnostics.effective_ranks,
        "active_coefficients": diagnostics.active_coefficients,
        "solver_success": np.asarray(diagnostics.solver_success),
        "solver_status": np.asarray(diagnostics.solver_status, dtype=np.int64),
        "solver_messages": np.asarray(diagnostics.solver_messages),
        "solver_optimality": diagnostics.solver_optimality,
        "pole_fit_attempted": np.asarray(
            diagnostics.pole_fit_success is not None,
        ),
        "pole_fit_success": np.asarray(bool(diagnostics.pole_fit_success)),
        "pole_fit_message": np.asarray(diagnostics.pole_fit_message or ""),
        "pole_fit_iterations": np.asarray(
            -1
            if diagnostics.pole_fit_iterations is None
            else diagnostics.pole_fit_iterations,
            dtype=np.int64,
        ),
        "objective": np.asarray(diagnostics.objective, dtype=np.float64),
        "statistically_weighted": np.asarray(diagnostics.statistically_weighted),
        "covariance_effective_ranks": (
            np.empty(0, dtype=np.int64)
            if diagnostics.covariance_effective_ranks is None
            else diagnostics.covariance_effective_ranks
        ),
        "covariance_truncated": (
            np.empty(0, dtype=np.bool_)
            if diagnostics.covariance_truncated is None
            else np.asarray(diagnostics.covariance_truncated)
        ),
        "unique_eta_count": np.asarray(
            diagnostics.unique_eta_count,
            dtype=np.int64,
        ),
        "underdetermined": np.asarray(diagnostics.underdetermined),
        "underdetermined_reasons": np.asarray(diagnostics.underdetermined_reasons),
    }
    if data.block_estimates is not None:
        payload["signed_lit_jackknife_blocks"] = data.block_estimates

    jackknife = inversion.jackknife
    payload["jackknife_available"] = np.asarray(jackknife is not None)
    if jackknife is None:
        payload.update(
            {
                "jackknife_block_count": np.asarray(0, dtype=np.int64),
                "jackknife_leave_one_out_pole_energies": np.empty(
                    (0, result.pole_energies.size),
                ),
                "jackknife_pole_energy_standard_error": np.full_like(
                    result.pole_energies,
                    np.nan,
                ),
                "jackknife_pole_energy_bias": np.full_like(
                    result.pole_energies,
                    np.nan,
                ),
                "jackknife_bias_corrected_pole_energies": np.full_like(
                    result.pole_energies,
                    np.nan,
                ),
                "jackknife_leave_one_out_pole_strengths": np.empty(
                    (0, *result.pole_strengths.shape),
                ),
                "jackknife_pole_strength_standard_error": np.full_like(
                    result.pole_strengths,
                    np.nan,
                ),
                "jackknife_pole_strength_bias": np.full_like(
                    result.pole_strengths,
                    np.nan,
                ),
                "jackknife_bias_corrected_pole_strengths": np.full_like(
                    result.pole_strengths,
                    np.nan,
                ),
                "jackknife_leave_one_out_continuum_density": np.empty(
                    (0, *result.continuum_density.shape),
                ),
                "jackknife_continuum_density_standard_error": np.full_like(
                    result.continuum_density,
                    np.nan,
                ),
                "jackknife_continuum_density_bias": np.full_like(
                    result.continuum_density,
                    np.nan,
                ),
                "jackknife_bias_corrected_continuum_density": np.full_like(
                    result.continuum_density,
                    np.nan,
                ),
            }
        )
        return payload

    payload.update(
        {
            "jackknife_block_count": np.asarray(
                jackknife.block_count,
                dtype=np.int64,
            ),
            "jackknife_leave_one_out_pole_energies": (
                jackknife.leave_one_out_pole_energies
            ),
            "jackknife_pole_energy_standard_error": (
                jackknife.pole_energy_standard_error
            ),
            "jackknife_pole_energy_bias": jackknife.pole_energy_bias,
            "jackknife_bias_corrected_pole_energies": (
                jackknife.bias_corrected_pole_energies
            ),
            "jackknife_leave_one_out_pole_strengths": (
                jackknife.leave_one_out_pole_strengths
            ),
            "jackknife_pole_strength_standard_error": (
                jackknife.pole_strength_standard_error
            ),
            "jackknife_pole_strength_bias": jackknife.pole_strength_bias,
            "jackknife_bias_corrected_pole_strengths": (
                jackknife.bias_corrected_pole_strengths
            ),
            "jackknife_leave_one_out_continuum_density": (
                jackknife.leave_one_out_continuum_density
            ),
            "jackknife_continuum_density_standard_error": (
                jackknife.continuum_density_standard_error
            ),
            "jackknife_continuum_density_bias": jackknife.continuum_density_bias,
            "jackknife_bias_corrected_continuum_density": (
                jackknife.bias_corrected_continuum_density
            ),
        }
    )
    return payload


__all__ = [
    "AggregatedLITNPZ",
    "LITInversionJackknife",
    "LITInversionSettings",
    "LITNPZInversion",
    "LITNPZMetadata",
    "aggregate_lit_npz",
    "invert_lit_npz",
    "lit_inversion_npz_payload",
]
