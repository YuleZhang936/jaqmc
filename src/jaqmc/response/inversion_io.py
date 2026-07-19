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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from jaqmc.response.inversion import _covariances, lit_block_statistics


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


__all__ = [
    "AggregatedLITNPZ",
    "LITNPZMetadata",
    "aggregate_lit_npz",
]
