# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

r"""Nonnegative inversion of signed Lorentz integral transforms.

The response is represented as a small set of discrete poles below a known
continuum threshold and a nonnegative piecewise-linear continuum above it,

.. math::

   R_a(E) = \sum_k S_{ak}\,\delta(E-E_k)
            + \Theta(E-I) C_a(E).

Pole energies are shared between response axes; pole strengths and continuum
densities are fitted independently for each axis.  The input to this module is
the *raw signed LIT*, not ``eta * L / pi`` and not a clipped spectrum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LITInversionDiagnostics:
    """Numerical diagnostics for each fitted response axis.

    Array-valued fields have one entry per response axis.  Condition numbers
    and ranks refer to the whitened, regularized design matrix actually passed
    to the bounded least-squares solver.  ``reduced_chi_squared`` uses the
    number of positive coefficients as an approximate parameter count.
    Covariance rank and truncation fields are ``None`` when covariance
    weighting was not requested.
    """

    residual_norms: NDArray[np.float64]
    weighted_residual_norms: NDArray[np.float64]
    regularization_norms: NDArray[np.float64]
    reduced_chi_squared: NDArray[np.float64]
    condition_numbers: NDArray[np.float64]
    effective_ranks: NDArray[np.int64]
    active_coefficients: NDArray[np.int64]
    solver_success: tuple[bool, ...]
    solver_status: tuple[int, ...]
    solver_messages: tuple[str, ...]
    solver_optimality: NDArray[np.float64]
    pole_fit_success: bool | None
    pole_fit_message: str | None
    pole_fit_iterations: int | None
    objective: float
    statistically_weighted: bool
    covariance_effective_ranks: NDArray[np.int64] | None
    covariance_truncated: tuple[bool, ...] | None
    unique_eta_count: int
    underdetermined: bool
    underdetermined_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LITInversionResult:
    """Result of a mixed discrete-pole and continuum LIT inversion.

    All response-valued arrays carry a leading response-axis dimension, even
    when the input contained only one axis.  Observation dimensions in
    ``fitted_lit`` and ``residual`` match the broadcast shape of ``omega`` and
    ``eta``.
    """

    pole_energies: NDArray[np.float64]
    pole_strengths: NDArray[np.float64]
    continuum_grid: NDArray[np.float64]
    continuum_density: NDArray[np.float64]
    fitted_lit: NDArray[np.float64]
    residual: NDArray[np.float64]
    diagnostics: LITInversionDiagnostics


@dataclass(frozen=True)
class LITBlockStatistics:
    """Mean and correlated Monte Carlo uncertainty from matched blocks."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    block_count: int


def lit_block_statistics(block_estimates: ArrayLike) -> LITBlockStatistics:
    """Estimate the covariance of a signed-LIT mean from matched blocks.

    The final two dimensions must be ``(n_observations, n_blocks)``; optional
    leading dimensions usually index response axes.  Keeping the same block
    index across all frequencies and broadening widths preserves their Monte
    Carlo correlations.  The returned covariance is that of the *mean*, not
    the covariance of individual block values, and is therefore divided by
    ``n_blocks * (n_blocks - 1)``.
    """
    values = np.asarray(block_estimates, dtype=np.float64)
    if values.ndim < 2:
        msg = (
            "block_estimates must have at least observation and block "
            f"dimensions, got {values.shape}"
        )
        raise ValueError(msg)
    block_count = int(values.shape[-1])
    if block_count < 2:
        msg = "at least two matched blocks are required for a covariance estimate"
        raise ValueError(msg)
    if values.shape[-2] < 1:
        msg = "block_estimates must contain at least one observation"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "block_estimates must contain only finite values"
        raise ValueError(msg)

    mean = np.mean(values, axis=-1)
    centered = values - mean[..., np.newaxis]
    covariance = np.einsum(
        "...ib,...jb->...ij",
        centered,
        centered,
        optimize=True,
    ) / (block_count * (block_count - 1))
    standard_error = np.sqrt(
        np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0.0)
    )
    return LITBlockStatistics(
        mean=mean,
        covariance=covariance,
        standard_error=standard_error,
        block_count=block_count,
    )


@dataclass(frozen=True)
class _Whitening:
    """Axis-specific whitening operation."""

    standard_deviation: NDArray[np.float64] | None = None
    covariance_whitener: NDArray[np.float64] | None = None
    covariance_effective_rank: int | None = None
    covariance_was_truncated: bool | None = None

    def apply_vector(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.standard_deviation is not None:
            return value / self.standard_deviation
        if self.covariance_whitener is not None:
            return self.covariance_whitener @ value
        return value

    def apply_matrix(self, value: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.standard_deviation is not None:
            return value / self.standard_deviation[:, np.newaxis]
        if self.covariance_whitener is not None:
            return self.covariance_whitener @ value
        return value


@dataclass(frozen=True)
class _AxisSolve:
    """Internal result from one nonnegative least-squares solve."""

    coefficients: NDArray[np.float64]
    fitted: NDArray[np.float64]
    residual: NDArray[np.float64]
    weighted_residual: NDArray[np.float64]
    regularization_norm: float
    condition_number: float
    effective_rank: int
    active_coefficients: int
    success: bool
    status: int
    message: str
    optimality: float
    augmented_objective: float


def _observation_arrays(
    omega: ArrayLike,
    eta: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...]]:
    omega_array = np.asarray(omega, dtype=np.float64)
    eta_array = np.asarray(eta, dtype=np.float64)
    try:
        omega_array, eta_array = np.broadcast_arrays(omega_array, eta_array)
    except ValueError as error:
        msg = (
            "omega and eta must be broadcast-compatible, got "
            f"{omega_array.shape} and {eta_array.shape}"
        )
        raise ValueError(msg) from error
    if omega_array.ndim == 0:
        omega_array = omega_array.reshape(1)
        eta_array = eta_array.reshape(1)
    if omega_array.size == 0:
        msg = "omega and eta must contain at least one observation"
        raise ValueError(msg)
    if not np.all(np.isfinite(omega_array)):
        msg = "omega must contain only finite values"
        raise ValueError(msg)
    if not np.all(np.isfinite(eta_array)) or np.any(eta_array <= 0):
        msg = "eta must contain only finite, positive values"
        raise ValueError(msg)
    return omega_array.ravel(), eta_array.ravel(), omega_array.shape


def _one_dimensional_finite_array(
    value: ArrayLike | None,
    name: str,
) -> NDArray[np.float64]:
    if value is None:
        return np.empty(0, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional, got {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def lit_pole_kernel(
    omega: ArrayLike,
    eta: ArrayLike,
    pole_energies: ArrayLike,
) -> NDArray[np.float64]:
    """Build the Lorentz kernel for discrete poles.

    The returned shape is ``broadcast(omega, eta).shape + (n_poles,)``.
    """
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    kernel = 1.0 / (
        (energies[np.newaxis, :] - omega_flat[:, np.newaxis]) ** 2
        + eta_flat[:, np.newaxis] ** 2
    )
    return kernel.reshape((*observation_shape, energies.size))


def lit_linear_continuum_kernel(
    omega: ArrayLike,
    eta: ArrayLike,
    continuum_grid: ArrayLike,
) -> NDArray[np.float64]:
    """Build the LIT kernel for nonnegative piecewise-linear continuum values.

    ``continuum_grid`` contains the energy locations of the nodal continuum
    densities.  The density is taken to be zero outside the first and last
    grid points.  Each linear-basis/Lorentzian integral is evaluated exactly
    with its arctangent and logarithmic primitive.  This remains accurate when
    ``eta`` is much narrower than a continuum interval, where fixed-order
    quadrature can miss nearly the entire peak.
    """
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    grid = _one_dimensional_finite_array(continuum_grid, "continuum_grid")
    if grid.size < 2:
        msg = "continuum_grid must contain at least two points"
        raise ValueError(msg)
    widths = np.diff(grid)
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        msg = "continuum_grid must be strictly increasing"
        raise ValueError(msg)

    lower = grid[:-1][np.newaxis, :]
    upper = grid[1:][np.newaxis, :]
    widths_2d = widths[np.newaxis, :]
    omega_2d = omega_flat[:, np.newaxis]
    eta_2d = eta_flat[:, np.newaxis]
    x_lower = lower - omega_2d
    x_upper = upper - omega_2d

    # atan(x_upper / eta) - atan(x_lower / eta), expressed as one
    # branch-aware atan2.  Scaling both arguments avoids overflow for finite
    # but very large energies.
    scale = np.maximum.reduce(
        (np.abs(x_lower), np.abs(x_upper), np.broadcast_to(eta_2d, x_lower.shape))
    )
    scaled_eta = eta_2d / scale
    angle = np.arctan2(
        scaled_eta * (widths_2d / scale),
        scaled_eta**2 + (x_lower / scale) * (x_upper / scale),
    )
    inverse_lorentz_integral = angle / eta_2d
    half_log_ratio = np.log(np.hypot(x_upper, eta_2d)) - np.log(
        np.hypot(x_lower, eta_2d)
    )
    left_integral = (
        (upper - omega_2d) * inverse_lorentz_integral - half_log_ratio
    ) / widths_2d
    right_integral = (
        (omega_2d - lower) * inverse_lorentz_integral + half_log_ratio
    ) / widths_2d

    # Exact basis integrals are nonnegative.  Remove only roundoff-level
    # negative zeros from primitive cancellation; a materially negative value
    # indicates invalid arithmetic and is rejected.
    roundoff = (
        64.0
        * np.finfo(np.float64).eps
        * np.maximum(
            inverse_lorentz_integral,
            1.0,
        )
    )
    if np.any(left_integral < -roundoff) or np.any(right_integral < -roundoff):
        msg = "analytic continuum kernel lost nonnegativity"
        raise FloatingPointError(msg)
    left_integral = np.maximum(left_integral, 0.0)
    right_integral = np.maximum(right_integral, 0.0)

    kernel = np.zeros((omega_flat.size, grid.size), dtype=np.float64)
    kernel[:, :-1] += left_integral
    kernel[:, 1:] += right_integral
    return kernel.reshape((*observation_shape, grid.size))


def _coefficient_matrix(
    value: ArrayLike | None,
    expected_columns: int,
    name: str,
) -> tuple[NDArray[np.float64] | None, bool]:
    if value is None:
        return None, False
    array = np.asarray(value, dtype=np.float64)
    was_one_dimensional = array.ndim == 1
    if was_one_dimensional:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[1] != expected_columns:
        msg = (
            f"{name} must have shape ({expected_columns},) or "
            f"(n_axes, {expected_columns}), got {array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array, was_one_dimensional


def forward_lit(
    omega: ArrayLike,
    eta: ArrayLike,
    *,
    pole_energies: ArrayLike | None = None,
    pole_strengths: ArrayLike | None = None,
    continuum_grid: ArrayLike | None = None,
    continuum_density: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Evaluate a mixed pole-plus-piecewise-linear continuum signed LIT.

    One-dimensional coefficient inputs produce an array with the broadcast
    shape of ``omega`` and ``eta``.  Two-dimensional coefficients produce a
    leading response-axis dimension.  If both terms are supplied, their axis
    counts must agree (or one term may contain a single broadcastable axis).
    """
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    grid = _one_dimensional_finite_array(continuum_grid, "continuum_grid")
    strengths, strengths_were_1d = _coefficient_matrix(
        pole_strengths,
        energies.size,
        "pole_strengths",
    )
    density, density_was_1d = _coefficient_matrix(
        continuum_density,
        grid.size,
        "continuum_density",
    )
    if strengths is None and density is None:
        msg = "at least one of pole_strengths or continuum_density is required"
        raise ValueError(msg)
    if strengths is not None and energies.size == 0:
        msg = "pole_energies are required with pole_strengths"
        raise ValueError(msg)
    if density is not None and grid.size < 2:
        msg = "at least two continuum_grid points are required with continuum_density"
        raise ValueError(msg)

    axis_counts = [
        coefficient.shape[0]
        for coefficient in (strengths, density)
        if coefficient is not None
    ]
    n_axes = max(axis_counts)
    if any(count not in (1, n_axes) for count in axis_counts):
        msg = f"coefficient axis counts must agree or equal one, got {axis_counts}"
        raise ValueError(msg)
    output = np.zeros((n_axes, omega_flat.size), dtype=np.float64)
    if strengths is not None:
        pole_kernel = lit_pole_kernel(omega_flat, eta_flat, energies).reshape(
            omega_flat.size,
            energies.size,
        )
        output += np.broadcast_to(strengths, (n_axes, energies.size)) @ pole_kernel.T
    if density is not None:
        continuum_kernel = lit_linear_continuum_kernel(
            omega_flat,
            eta_flat,
            grid,
        ).reshape(omega_flat.size, grid.size)
        output += np.broadcast_to(density, (n_axes, grid.size)) @ continuum_kernel.T

    output = output.reshape((n_axes, *observation_shape))
    all_inputs_were_1d = (strengths is None or strengths_were_1d) and (
        density is None or density_was_1d
    )
    return output[0] if n_axes == 1 and all_inputs_were_1d else output


def _response_axes(
    signed_lit: ArrayLike,
    observation_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    values = np.asarray(signed_lit, dtype=np.float64)
    n_observations = int(np.prod(observation_shape))
    if values.shape == observation_shape:
        values = values.reshape(1, n_observations)
    elif values.ndim >= 2 and values.shape[1:] == observation_shape:
        values = values.reshape(values.shape[0], n_observations)
    elif values.ndim == 2 and values.shape[1] == n_observations:
        values = values.copy()
    else:
        msg = (
            "signed_lit must have the observation shape or a leading response-axis "
            f"dimension; expected {observation_shape}, got {values.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "signed_lit must contain only finite values"
        raise ValueError(msg)
    return values


def _standard_deviations(
    value: ArrayLike,
    n_axes: int,
    observation_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    n_observations = int(np.prod(observation_shape))
    if array.ndim == 0:
        array = np.full((n_axes, n_observations), float(array))
    elif array.shape == observation_shape:
        array = np.broadcast_to(
            array.reshape(1, n_observations), (n_axes, n_observations)
        )
    elif array.ndim >= 2 and array.shape == (n_axes, *observation_shape):
        array = array.reshape(n_axes, n_observations)
    elif array.shape == (n_observations,):
        array = np.broadcast_to(array, (n_axes, n_observations))
    elif array.shape != (n_axes, n_observations):
        msg = (
            "standard_deviation must be scalar, observation-shaped, or include a "
            f"leading response-axis dimension; got {array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        msg = "standard_deviation must contain only finite, positive values"
        raise ValueError(msg)
    return np.asarray(array, dtype=np.float64)


def _covariances(
    value: ArrayLike,
    n_axes: int,
    n_observations: int,
    *,
    relative_tolerance: float,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (n_observations, n_observations):
        array = np.broadcast_to(
            array,
            (n_axes, n_observations, n_observations),
        )
    elif array.shape != (n_axes, n_observations, n_observations):
        msg = (
            "covariance must have shape (n_observations, n_observations) or "
            f"(n_axes, n_observations, n_observations), got {array.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "covariance must contain only finite values"
        raise ValueError(msg)
    validated = np.empty_like(array, dtype=np.float64)
    for axis, covariance_axis in enumerate(array):
        covariance_scale = float(np.max(np.abs(covariance_axis), initial=0.0))
        symmetry_tolerance = (
            max(relative_tolerance, 100.0 * np.finfo(np.float64).eps) * covariance_scale
        )
        if np.max(np.abs(covariance_axis - covariance_axis.T), initial=0.0) > (
            symmetry_tolerance
        ):
            msg = f"covariance for axis {axis} must be symmetric"
            raise ValueError(msg)
        symmetric_covariance = (covariance_axis + covariance_axis.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(symmetric_covariance)
        spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
        eigenvalue_tolerance = relative_tolerance * spectral_scale
        if eigenvalues[0] < -eigenvalue_tolerance:
            msg = f"covariance for axis {axis} must be positive semidefinite"
            raise ValueError(msg)
        validated[axis] = symmetric_covariance
    return validated


def _whitenings(
    *,
    n_axes: int,
    observation_shape: tuple[int, ...],
    standard_deviation: ArrayLike | None,
    covariance: ArrayLike | None,
    covariance_relative_tolerance: float,
) -> tuple[list[_Whitening], bool]:
    if standard_deviation is not None and covariance is not None:
        msg = "provide standard_deviation or covariance, not both"
        raise ValueError(msg)
    n_observations = int(np.prod(observation_shape))
    if standard_deviation is not None:
        deviations = _standard_deviations(
            standard_deviation,
            n_axes,
            observation_shape,
        )
        return [
            _Whitening(standard_deviation=deviations[axis]) for axis in range(n_axes)
        ], True
    if covariance is None:
        return [_Whitening() for _ in range(n_axes)], False

    covariances = _covariances(
        covariance,
        n_axes,
        n_observations,
        relative_tolerance=covariance_relative_tolerance,
    )
    whitenings: list[_Whitening] = []
    for axis, covariance_axis in enumerate(covariances):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_axis)
        spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
        if not spectral_scale > 0.0:
            msg = f"covariance for axis {axis} has no positive-variance subspace"
            raise ValueError(msg)
        eigenvalue_tolerance = covariance_relative_tolerance * spectral_scale
        retained = eigenvalues > eigenvalue_tolerance
        effective_rank = int(np.count_nonzero(retained))
        if effective_rank == 0:
            msg = f"covariance for axis {axis} has no positive-variance subspace"
            raise ValueError(msg)
        retained_vectors = eigenvectors[:, retained]
        whitener = (retained_vectors / np.sqrt(eigenvalues[retained])[np.newaxis, :]).T
        whitenings.append(
            _Whitening(
                covariance_whitener=np.asarray(whitener, dtype=np.float64),
                covariance_effective_rank=effective_rank,
                covariance_was_truncated=effective_rank < n_observations,
            )
        )
    return whitenings, True


def _continuum_curvature_matrix(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a grid-aware finite-element curvature penalty."""
    matrix = np.zeros((max(grid.size - 2, 0), grid.size), dtype=np.float64)
    for index in range(1, grid.size - 1):
        left_width = grid[index] - grid[index - 1]
        right_width = grid[index + 1] - grid[index]
        scale = np.sqrt(2.0 / (left_width + right_width))
        matrix[index - 1, index - 1] = scale / left_width
        matrix[index - 1, index] = -scale * (1.0 / left_width + 1.0 / right_width)
        matrix[index - 1, index + 1] = scale / right_width
    return matrix


def _matrix_condition(
    matrix: NDArray[np.float64],
) -> tuple[float, int]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.inf, 0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == min(matrix.shape) and singular_values[-1] > 0
        else np.inf
    )
    return condition, rank


@dataclass(frozen=True)
class _NNLSResult:
    """Result of the Lawson-Hanson nonnegative least-squares algorithm."""

    x: NDArray[np.float64]
    success: bool
    status: int
    message: str
    optimality: float
    iterations: int


def _kkt_optimality(
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    coefficients: NDArray[np.float64],
) -> float:
    """Return a dimensionless, column-scale-invariant KKT violation."""
    gradient = matrix.T @ (matrix @ coefficients - target)
    column_norms = np.linalg.norm(matrix, axis=0)
    target_norm = float(np.linalg.norm(target))
    gradient_scale = column_norms * max(target_norm, np.finfo(np.float64).tiny)
    normalized_gradient = np.divide(
        gradient,
        gradient_scale,
        out=np.zeros_like(gradient),
        where=gradient_scale > 0.0,
    )
    positive = coefficients > 0.0
    positive_violation = (
        float(np.max(np.abs(normalized_gradient[positive]), initial=0.0))
        if np.any(positive)
        else 0.0
    )
    bound_violation = float(np.max(-normalized_gradient[~positive], initial=0.0))
    return max(positive_violation, bound_violation)


def _nonnegative_least_squares(  # noqa: C901
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int | None,
) -> _NNLSResult:
    """Solve NNLS with the finite Lawson-Hanson active-set algorithm.

    This implementation deliberately uses only NumPy.  Every unconstrained
    passive-set problem is solved by SVD-backed ``numpy.linalg.lstsq``, which
    also makes rank-deficient LIT designs safe.  Positive column scaling does
    not change the NNLS feasible set, so columns are normalized internally and
    coefficients are mapped back afterward.  This prevents pole and continuum
    columns with different physical scales from changing active-set decisions.
    """
    original_matrix = matrix
    original_target = target
    n_coefficients = matrix.shape[1]
    if n_coefficients == 0:
        return _NNLSResult(
            x=np.empty(0, dtype=np.float64),
            success=True,
            status=1,
            message="empty coefficient vector",
            optimality=0.0,
            iterations=0,
        )
    column_scales = np.linalg.norm(matrix, axis=0)
    safe_column_scales = np.where(column_scales > 0.0, column_scales, 1.0)
    matrix = matrix / safe_column_scales[np.newaxis, :]
    target_scale = float(np.linalg.norm(target))
    safe_target_scale = target_scale if target_scale > 0.0 else 1.0
    target = target / safe_target_scale
    iteration_limit = (
        max_iterations if max_iterations is not None else max(30 * n_coefficients, 1)
    )
    if iteration_limit < 1:
        msg = f"solver_max_iterations must be positive, got {iteration_limit}"
        raise ValueError(msg)

    coefficients = np.zeros(n_coefficients, dtype=np.float64)
    passive = np.zeros(n_coefficients, dtype=bool)
    column_norms = np.linalg.norm(matrix, axis=0)
    target_norm = float(np.linalg.norm(target))
    dual_scale = column_norms * max(target_norm, np.finfo(np.float64).tiny)
    dual_tolerance = tolerance * dual_scale
    dual = matrix.T @ target
    iterations = 0
    success = True
    message = "KKT conditions satisfied"

    while np.any((~passive) & (dual > dual_tolerance)):
        if iterations >= iteration_limit:
            success = False
            message = "maximum NNLS iterations reached"
            break
        normalized_dual = np.divide(
            dual,
            dual_scale,
            out=np.full_like(dual, -np.inf),
            where=dual_scale > 0.0,
        )
        candidate_dual = np.where(~passive, normalized_dual, -np.inf)
        entering = int(np.argmax(candidate_dual))
        passive[entering] = True

        while True:
            trial = np.zeros_like(coefficients)
            trial[passive] = np.linalg.lstsq(
                matrix[:, passive],
                target,
                rcond=None,
            )[0]
            coefficient_tolerance = tolerance * np.maximum(
                np.abs(trial),
                np.abs(coefficients),
            )
            nonpositive = passive & (trial <= 0.0)
            if not np.any(nonpositive):
                coefficients = trial
                break
            denominators = coefficients[nonpositive] - trial[nonpositive]
            valid = denominators > 0
            if np.any(valid):
                step = float(
                    np.min(coefficients[nonpositive][valid] / denominators[valid])
                )
            else:
                step = 0.0
            coefficients += step * (trial - coefficients)
            to_remove = passive & (coefficients <= coefficient_tolerance)
            coefficients[to_remove] = 0.0
            passive[to_remove] = False
            iterations += 1
            if iterations >= iteration_limit:
                success = False
                message = "maximum NNLS iterations reached"
                break
        if not success:
            break
        residual = target - matrix @ coefficients
        dual = matrix.T @ residual
        iterations += 1

    coefficients = (
        np.maximum(coefficients, 0.0) * safe_target_scale / safe_column_scales
    )
    optimality = _kkt_optimality(
        original_matrix,
        original_target,
        coefficients,
    )
    optimality_limit = max(
        10.0 * tolerance,
        100.0 * np.finfo(np.float64).eps * max(original_matrix.shape, default=1),
    )
    if success and (not np.isfinite(optimality) or optimality > optimality_limit):
        success = False
        message = (
            "NNLS active set terminated with dimensionless KKT violation "
            f"{optimality:.3e} above {optimality_limit:.3e}"
        )
    return _NNLSResult(
        x=coefficients,
        success=success,
        status=1 if success else 0,
        message=message,
        optimality=optimality,
        iterations=iterations,
    )


def _solve_axis(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    whitening: _Whitening,
    regularizer: NDArray[np.float64],
    *,
    solver_tolerance: float,
    solver_max_iterations: int | None,
) -> _AxisSolve:
    weighted_design = whitening.apply_matrix(design)
    weighted_target = whitening.apply_vector(target)
    if regularizer.shape[0]:
        augmented_design = np.vstack((weighted_design, regularizer))
        augmented_target = np.concatenate(
            (weighted_target, np.zeros(regularizer.shape[0], dtype=np.float64))
        )
    else:
        augmented_design = weighted_design
        augmented_target = weighted_target
    solution = _nonnegative_least_squares(
        augmented_design,
        augmented_target,
        tolerance=solver_tolerance,
        max_iterations=solver_max_iterations,
    )
    coefficients = np.maximum(np.asarray(solution.x, dtype=np.float64), 0.0)
    fitted = design @ coefficients
    residual = fitted - target
    weighted_residual = whitening.apply_vector(residual)
    regularization_residual = regularizer @ coefficients
    column_norms = np.linalg.norm(augmented_design, axis=0)
    normalized_design = (
        augmented_design
        / np.where(
            column_norms > 0.0,
            column_norms,
            1.0,
        )[np.newaxis, :]
    )
    condition_number, effective_rank = _matrix_condition(normalized_design)
    coefficient_scale = max(
        float(np.max(coefficients, initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    active_coefficients = int(
        np.count_nonzero(coefficients > 1e-10 * coefficient_scale)
    )
    return _AxisSolve(
        coefficients=coefficients,
        fitted=fitted,
        residual=residual,
        weighted_residual=weighted_residual,
        regularization_norm=float(np.linalg.norm(regularization_residual)),
        condition_number=condition_number,
        effective_rank=effective_rank,
        active_coefficients=active_coefficients,
        success=bool(solution.success),
        status=int(solution.status),
        message=str(solution.message),
        optimality=float(solution.optimality),
        augmented_objective=float(
            weighted_residual @ weighted_residual
            + regularization_residual @ regularization_residual
        ),
    )


def _regularizer(
    n_poles: int,
    continuum_grid: NDArray[np.float64],
    regularization: float,
) -> NDArray[np.float64]:
    if regularization < 0 or not np.isfinite(regularization):
        msg = (
            "continuum_regularization must be finite and nonnegative, got "
            f"{regularization}"
        )
        raise ValueError(msg)
    curvature = _continuum_curvature_matrix(continuum_grid)
    matrix = np.zeros((curvature.shape[0], n_poles + continuum_grid.size))
    if curvature.size:
        matrix[:, n_poles:] = np.sqrt(regularization) * curvature
    return matrix


def _design_matrix(
    omega: NDArray[np.float64],
    eta: NDArray[np.float64],
    pole_energies: NDArray[np.float64],
    continuum_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    pieces: list[NDArray[np.float64]] = []
    if pole_energies.size:
        pieces.append(
            lit_pole_kernel(omega, eta, pole_energies).reshape(
                omega.size,
                pole_energies.size,
            )
        )
    if continuum_grid.size:
        pieces.append(
            lit_linear_continuum_kernel(
                omega,
                eta,
                continuum_grid,
            ).reshape(omega.size, continuum_grid.size)
        )
    if not pieces:
        msg = "at least one discrete pole or continuum component is required"
        raise ValueError(msg)
    return np.hstack(pieces)


def _fit_coefficients(
    omega: NDArray[np.float64],
    eta: NDArray[np.float64],
    response: NDArray[np.float64],
    whitenings: list[_Whitening],
    pole_energies: NDArray[np.float64],
    continuum_grid: NDArray[np.float64],
    continuum_regularization: float,
    solver_tolerance: float,
    solver_max_iterations: int | None,
) -> tuple[NDArray[np.float64], list[_AxisSolve]]:
    design = _design_matrix(
        omega,
        eta,
        pole_energies,
        continuum_grid,
    )
    regularizer = _regularizer(
        pole_energies.size,
        continuum_grid,
        continuum_regularization,
    )
    solves = [
        _solve_axis(
            design,
            response[axis],
            whitenings[axis],
            regularizer,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )
        for axis in range(response.shape[0])
    ]
    return design, solves


def _pole_bounds(
    bounds: ArrayLike | None,
    pole_energies: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    if bounds is None:
        msg = "pole_energy_bounds are required when fit_pole_energies is true"
        raise ValueError(msg)
    array = np.asarray(bounds, dtype=np.float64)
    if array.shape != (pole_energies.size, 2):
        msg = (
            "pole_energy_bounds must have shape (n_poles, 2), got "
            f"{array.shape} for {pole_energies.size} poles"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or np.any(array[:, 0] >= array[:, 1]):
        msg = "each pole energy bound must be finite and strictly increasing"
        raise ValueError(msg)
    if np.any(array[:, 1] >= threshold):
        msg = "all pole energy bounds must lie strictly below threshold"
        raise ValueError(msg)
    if array.shape[0] > 1 and np.any(array[:-1, 1] >= array[1:, 0]):
        msg = "pole energy bounds must be ordered and non-overlapping"
        raise ValueError(msg)
    if np.any(pole_energies < array[:, 0]) or np.any(pole_energies > array[:, 1]):
        msg = "initial pole_energies must lie within pole_energy_bounds"
        raise ValueError(msg)
    return array


@dataclass(frozen=True)
class _PoleOptimization:
    """Internal result of bounded shared-pole coordinate minimization."""

    x: NDArray[np.float64]
    objective: float
    success: bool
    message: str
    iterations: int


def _golden_coordinate_search(
    objective: Callable[[NDArray[np.float64]], float],
    current: NDArray[np.float64],
    current_objective: float,
    index: int,
    lower: float,
    upper: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], float]:
    """Minimize one bounded coordinate after a coarse global scan."""

    def evaluate(value: float) -> tuple[NDArray[np.float64], float]:
        candidate = current.copy()
        candidate[index] = value
        return candidate, float(objective(candidate))

    scan = np.linspace(lower, upper, 9)
    scan = np.unique(np.concatenate((scan, [current[index]])))
    values = np.empty(scan.size, dtype=np.float64)
    candidates: list[NDArray[np.float64]] = []
    for scan_index, value in enumerate(scan):
        candidate, candidate_objective = evaluate(float(value))
        candidates.append(candidate)
        values[scan_index] = candidate_objective
    best_index = int(np.argmin(values))
    best_candidate = candidates[best_index]
    best_objective = float(values[best_index])
    if current_objective <= best_objective:
        best_candidate = current.copy()
        best_objective = current_objective

    left_index = max(best_index - 1, 0)
    right_index = min(best_index + 1, scan.size - 1)
    left = float(scan[left_index])
    right = float(scan[right_index])
    if left == right:
        return best_candidate, best_objective

    inverse_phi = (np.sqrt(5.0) - 1.0) / 2.0
    point_left = right - inverse_phi * (right - left)
    point_right = left + inverse_phi * (right - left)
    candidate_left, objective_left = evaluate(point_left)
    candidate_right, objective_right = evaluate(point_right)
    coordinate_tolerance = tolerance * max(1.0, abs(lower), abs(upper))
    for _ in range(80):
        if right - left <= coordinate_tolerance:
            break
        if objective_left <= objective_right:
            right = point_right
            point_right = point_left
            candidate_right = candidate_left
            objective_right = objective_left
            point_left = right - inverse_phi * (right - left)
            candidate_left, objective_left = evaluate(point_left)
        else:
            left = point_left
            point_left = point_right
            candidate_left = candidate_right
            objective_left = objective_right
            point_right = left + inverse_phi * (right - left)
            candidate_right, objective_right = evaluate(point_right)
    for candidate, candidate_objective in (
        (candidate_left, objective_left),
        (candidate_right, objective_right),
    ):
        if candidate_objective < best_objective:
            best_candidate = candidate
            best_objective = candidate_objective
    return best_candidate, best_objective


def _fit_shared_pole_energies(
    objective: Callable[[NDArray[np.float64]], float],
    initial: NDArray[np.float64],
    bounds: NDArray[np.float64],
    *,
    tolerance: float,
    max_iterations: int,
) -> _PoleOptimization:
    """Pure-NumPy bounded coordinate fit for a few shared pole energies."""
    if max_iterations < 1:
        msg = f"pole_fit_max_iterations must be positive, got {max_iterations}"
        raise ValueError(msg)
    current = initial.copy()
    current_objective = float(objective(current))
    for iteration in range(1, max_iterations + 1):
        previous = current.copy()
        previous_objective = current_objective
        for index in range(current.size):
            current, current_objective = _golden_coordinate_search(
                objective,
                current,
                current_objective,
                index,
                float(bounds[index, 0]),
                float(bounds[index, 1]),
                tolerance,
            )
        movement = float(np.max(np.abs(current - previous), initial=0.0))
        energy_scale = max(float(np.max(np.abs(current), initial=0.0)), 1.0)
        objective_change = abs(previous_objective - current_objective)
        objective_scale = max(abs(previous_objective), abs(current_objective), 1.0)
        if movement <= tolerance * energy_scale and (
            objective_change <= tolerance * objective_scale
        ):
            return _PoleOptimization(
                x=current,
                objective=current_objective,
                success=True,
                message="bounded coordinate fit converged",
                iterations=iteration,
            )
    return _PoleOptimization(
        x=current,
        objective=current_objective,
        success=False,
        message="maximum pole-fit iterations reached",
        iterations=max_iterations,
    )


def _unique_eta_count(eta: NDArray[np.float64]) -> int:
    """Count eta values after merging roundoff-equivalent widths."""
    ordered = np.sort(eta)
    count = 1
    reference = float(ordered[0])
    for value in ordered[1:]:
        if not np.isclose(value, reference, rtol=1e-12, atol=1e-15):
            count += 1
            reference = float(value)
    return count


def _validate_model_components(
    threshold: float,
    energies: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> None:
    if not np.isfinite(threshold):
        msg = f"threshold must be finite, got {threshold}"
        raise ValueError(msg)
    if energies.size and (
        np.any(np.diff(energies) <= 0) or np.any(energies >= threshold)
    ):
        msg = "pole_energies must be strictly increasing and below threshold"
        raise ValueError(msg)
    if grid.size and (grid.size < 2 or np.any(np.diff(grid) <= 0)):
        msg = "continuum_grid must have at least two strictly increasing points"
        raise ValueError(msg)
    if grid.size:
        threshold_tolerance = 1e-12 * max(1.0, abs(threshold))
        if not np.isclose(grid[0], threshold, rtol=0.0, atol=threshold_tolerance):
            msg = "the first continuum_grid point must equal threshold"
            raise ValueError(msg)
    if not energies.size and not grid.size:
        msg = "at least one discrete pole or continuum component is required"
        raise ValueError(msg)


def _validate_solver_options(
    max_fitted_poles: int,
    solver_tolerance: float,
    pole_fit_tolerance: float,
    covariance_relative_tolerance: float,
) -> None:
    if max_fitted_poles < 1:
        msg = f"max_fitted_poles must be positive, got {max_fitted_poles}"
        raise ValueError(msg)
    if solver_tolerance <= 0 or not np.isfinite(solver_tolerance):
        msg = f"solver_tolerance must be finite and positive, got {solver_tolerance}"
        raise ValueError(msg)
    if pole_fit_tolerance <= 0 or not np.isfinite(pole_fit_tolerance):
        msg = (
            f"pole_fit_tolerance must be finite and positive, got {pole_fit_tolerance}"
        )
        raise ValueError(msg)
    if not 0 < covariance_relative_tolerance < 1:
        msg = (
            "covariance_relative_tolerance must lie strictly between zero and one, "
            f"got {covariance_relative_tolerance}"
        )
        raise ValueError(msg)


def _refine_pole_energies(
    omega: NDArray[np.float64],
    eta: NDArray[np.float64],
    response: NDArray[np.float64],
    whitenings: list[_Whitening],
    energies: NDArray[np.float64],
    grid: NDArray[np.float64],
    *,
    threshold: float,
    continuum_regularization: float,
    solver_tolerance: float,
    solver_max_iterations: int | None,
    fit_pole_energies: bool,
    pole_energy_bounds: ArrayLike | None,
    max_fitted_poles: int,
    pole_fit_tolerance: float,
    pole_fit_max_iterations: int,
) -> tuple[NDArray[np.float64], _PoleOptimization | None]:
    if not fit_pole_energies:
        if pole_energy_bounds is not None:
            msg = "pole_energy_bounds require fit_pole_energies=true"
            raise ValueError(msg)
        return energies, None
    if energies.size == 0:
        msg = "at least one initial pole energy is required for pole fitting"
        raise ValueError(msg)
    if energies.size > max_fitted_poles:
        msg = (
            f"refusing to fit {energies.size} pole energies; "
            f"max_fitted_poles is {max_fitted_poles}"
        )
        raise ValueError(msg)
    bounds = _pole_bounds(pole_energy_bounds, energies, threshold)

    def objective(candidate: NDArray[np.float64]) -> float:
        _, candidate_solves = _fit_coefficients(
            omega,
            eta,
            response,
            whitenings,
            candidate,
            grid,
            continuum_regularization,
            solver_tolerance,
            solver_max_iterations,
        )
        return float(sum(solve.augmented_objective for solve in candidate_solves))

    optimization = _fit_shared_pole_energies(
        objective,
        energies,
        bounds,
        tolerance=pole_fit_tolerance,
        max_iterations=pole_fit_max_iterations,
    )
    return optimization.x, optimization


def _inversion_diagnostics(
    design: NDArray[np.float64],
    solves: list[_AxisSolve],
    whitenings: list[_Whitening],
    eta: NDArray[np.float64],
    *,
    statistically_weighted: bool,
    pole_optimization: _PoleOptimization | None,
) -> LITInversionDiagnostics:
    active_coefficients = np.asarray(
        [solve.active_coefficients for solve in solves],
        dtype=np.int64,
    )
    weighted_residual_norms = np.asarray(
        [np.linalg.norm(solve.weighted_residual) for solve in solves],
        dtype=np.float64,
    )
    weighted_observation_counts = np.asarray(
        [solve.weighted_residual.size for solve in solves],
        dtype=np.int64,
    )
    degrees_of_freedom = np.maximum(
        weighted_observation_counts - active_coefficients,
        1,
    )
    covariance_effective_ranks: NDArray[np.int64] | None = None
    covariance_truncated: tuple[bool, ...] | None = None
    if whitenings[0].covariance_effective_rank is not None:
        covariance_effective_ranks = np.asarray(
            [whitening.covariance_effective_rank for whitening in whitenings],
            dtype=np.int64,
        )
        covariance_truncated = tuple(
            bool(whitening.covariance_was_truncated) for whitening in whitenings
        )
    unique_eta_count = _unique_eta_count(eta)
    underdetermined_reasons: list[str] = []
    if unique_eta_count == 1:
        underdetermined_reasons.append(
            "only one eta is present; cross-width inversion stability is untested"
        )
    data_rank = int(np.linalg.matrix_rank(design))
    if data_rank < design.shape[1]:
        underdetermined_reasons.append(
            f"unregularized design rank {data_rank} is below "
            f"{design.shape[1]} coefficients"
        )
    if design.shape[0] <= design.shape[1]:
        underdetermined_reasons.append(
            "the observation count does not exceed the coefficient count"
        )
    if (
        covariance_effective_ranks is not None
        and np.min(covariance_effective_ranks) <= design.shape[1]
    ):
        underdetermined_reasons.append(
            "the covariance effective rank does not exceed the coefficient count"
        )
    return LITInversionDiagnostics(
        residual_norms=np.asarray(
            [np.linalg.norm(solve.residual) for solve in solves],
            dtype=np.float64,
        ),
        weighted_residual_norms=weighted_residual_norms,
        regularization_norms=np.asarray(
            [solve.regularization_norm for solve in solves],
            dtype=np.float64,
        ),
        reduced_chi_squared=weighted_residual_norms**2 / degrees_of_freedom,
        condition_numbers=np.asarray(
            [solve.condition_number for solve in solves],
            dtype=np.float64,
        ),
        effective_ranks=np.asarray(
            [solve.effective_rank for solve in solves],
            dtype=np.int64,
        ),
        active_coefficients=active_coefficients,
        solver_success=tuple(solve.success for solve in solves),
        solver_status=tuple(solve.status for solve in solves),
        solver_messages=tuple(solve.message for solve in solves),
        solver_optimality=np.asarray(
            [solve.optimality for solve in solves],
            dtype=np.float64,
        ),
        pole_fit_success=(
            None if pole_optimization is None else pole_optimization.success
        ),
        pole_fit_message=(
            None if pole_optimization is None else pole_optimization.message
        ),
        pole_fit_iterations=(
            None if pole_optimization is None else pole_optimization.iterations
        ),
        objective=float(sum(solve.augmented_objective for solve in solves)),
        statistically_weighted=statistically_weighted,
        covariance_effective_ranks=covariance_effective_ranks,
        covariance_truncated=covariance_truncated,
        unique_eta_count=unique_eta_count,
        underdetermined=bool(underdetermined_reasons),
        underdetermined_reasons=tuple(underdetermined_reasons),
    )


def invert_signed_lit(
    omega: ArrayLike,
    eta: ArrayLike,
    signed_lit: ArrayLike,
    *,
    threshold: float,
    pole_energies: ArrayLike | None = None,
    continuum_grid: ArrayLike | None = None,
    standard_deviation: ArrayLike | None = None,
    covariance: ArrayLike | None = None,
    covariance_relative_tolerance: float = 1e-10,
    continuum_regularization: float = 0.0,
    fit_pole_energies: bool = False,
    pole_energy_bounds: ArrayLike | None = None,
    max_fitted_poles: int = 8,
    pole_fit_tolerance: float = 1e-7,
    pole_fit_max_iterations: int = 200,
    solver_tolerance: float = 1e-10,
    solver_max_iterations: int | None = None,
) -> LITInversionResult:
    """Invert a raw signed LIT into nonnegative poles and continuum density.

    ``omega`` and ``eta`` may be multidimensional broadcast-compatible grids;
    this is convenient for fitting several broadening widths at once.  A
    single-axis ``signed_lit`` has that broadcast shape.  Multi-axis data add a
    leading axis dimension.  Independently saved NPZ blocks can instead be
    flattened and concatenated in matching ``omega``, ``eta``, and last-axis
    ``signed_lit`` order.  Covariances use that same flattened order.
    Positive-semidefinite empirical covariances are supported: eigenmodes below
    ``covariance_relative_tolerance`` times the spectral radius are discarded,
    and diagnostics report the retained rank.

    The continuum is represented only on ``continuum_grid`` and is zero beyond
    its last point.  The first continuum node must coincide with ``threshold``.
    Pole refinement is a variable-projection fit: every trial set of shared
    pole energies performs independent nonnegative linear solves for all axes.
    Explicit, ordered, non-overlapping bounds are required to avoid permutation
    ambiguity.
    """
    omega_flat, eta_flat, observation_shape = _observation_arrays(omega, eta)
    response = _response_axes(signed_lit, observation_shape)
    energies = _one_dimensional_finite_array(pole_energies, "pole_energies")
    grid = _one_dimensional_finite_array(continuum_grid, "continuum_grid")
    _validate_model_components(threshold, energies, grid)
    _validate_solver_options(
        max_fitted_poles,
        solver_tolerance,
        pole_fit_tolerance,
        covariance_relative_tolerance,
    )

    whitenings, statistically_weighted = _whitenings(
        n_axes=response.shape[0],
        observation_shape=observation_shape,
        standard_deviation=standard_deviation,
        covariance=covariance,
        covariance_relative_tolerance=covariance_relative_tolerance,
    )

    energies, pole_optimization = _refine_pole_energies(
        omega_flat,
        eta_flat,
        response,
        whitenings,
        energies,
        grid,
        threshold=threshold,
        continuum_regularization=continuum_regularization,
        solver_tolerance=solver_tolerance,
        solver_max_iterations=solver_max_iterations,
        fit_pole_energies=fit_pole_energies,
        pole_energy_bounds=pole_energy_bounds,
        max_fitted_poles=max_fitted_poles,
        pole_fit_tolerance=pole_fit_tolerance,
        pole_fit_max_iterations=pole_fit_max_iterations,
    )

    design, solves = _fit_coefficients(
        omega_flat,
        eta_flat,
        response,
        whitenings,
        energies,
        grid,
        continuum_regularization,
        solver_tolerance,
        solver_max_iterations,
    )
    coefficients = np.stack([solve.coefficients for solve in solves])
    fitted = np.stack([solve.fitted for solve in solves])
    residual = np.stack([solve.residual for solve in solves])
    diagnostics = _inversion_diagnostics(
        design,
        solves,
        whitenings,
        eta_flat,
        statistically_weighted=statistically_weighted,
        pole_optimization=pole_optimization,
    )
    return LITInversionResult(
        pole_energies=energies.copy(),
        pole_strengths=coefficients[:, : energies.size],
        continuum_grid=grid.copy(),
        continuum_density=coefficients[:, energies.size :],
        fitted_lit=fitted.reshape((response.shape[0], *observation_shape)),
        residual=residual.reshape((response.shape[0], *observation_shape)),
        diagnostics=diagnostics,
    )


# A shorter alias for interactive use.
invert_lit = invert_signed_lit


__all__ = [
    "LITBlockStatistics",
    "LITInversionDiagnostics",
    "LITInversionResult",
    "forward_lit",
    "invert_lit",
    "invert_signed_lit",
    "lit_block_statistics",
    "lit_linear_continuum_kernel",
    "lit_pole_kernel",
]
