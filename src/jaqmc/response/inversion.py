# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: DOC201,DOC501

"""LIT inversion by the regularized basis expansion of arXiv:2504.20195."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BasisLITInversionResult:
    """Result of a regularized basis-expansion LIT inversion."""

    response_omega: NDArray[np.float64]
    response: NDArray[np.float64]
    fit_lit: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    alpha1: float
    alpha2: float
    l2_regularization: float
    chi2: float
    objective: float


def lit_response_basis(
    response_omega: ArrayLike,
    *,
    threshold: float,
    basis_count: int,
    alpha1: float,
    alpha2: float,
) -> NDArray[np.float64]:
    """Evaluate the response basis ``omega'^alpha1 exp(-alpha2 omega'/n)``."""
    if basis_count < 1:
        msg = f"basis_count must be positive, got {basis_count}"
        raise ValueError(msg)
    if alpha1 <= -1.0:
        msg = f"alpha1 must be greater than -1, got {alpha1}"
        raise ValueError(msg)
    if alpha2 <= 0.0:
        msg = f"alpha2 must be positive, got {alpha2}"
        raise ValueError(msg)
    omega = np.asarray(response_omega, dtype=np.float64)
    omega_prime = omega - float(threshold)
    active = omega_prime >= 0.0
    x = np.where(active, omega_prime, 0.0)
    n = np.arange(1, int(basis_count) + 1, dtype=np.float64)
    basis = x[:, None] ** float(alpha1) * np.exp(-float(alpha2) * x[:, None] / n)
    return np.where(active[:, None], basis, 0.0)


def lit_basis_transform(
    omega0: ArrayLike,
    response_omega: ArrayLike,
    basis: ArrayLike,
    eta: float,
) -> NDArray[np.float64]:
    """Numerically transform response basis functions into LIT basis values."""
    if eta <= 0.0:
        msg = f"eta must be positive, got {eta}"
        raise ValueError(msg)
    omega0_arr = np.asarray(omega0, dtype=np.float64)
    response_grid = np.asarray(response_omega, dtype=np.float64)
    basis_arr = np.asarray(basis, dtype=np.float64)
    if omega0_arr.ndim != 1:
        msg = "omega0 must be one-dimensional"
        raise ValueError(msg)
    if response_grid.ndim != 1:
        msg = "response_omega must be one-dimensional"
        raise ValueError(msg)
    if basis_arr.ndim != 2 or basis_arr.shape[0] != response_grid.size:
        msg = (
            "basis must have shape (response_omega.size, basis_count), got "
            f"{basis_arr.shape}"
        )
        raise ValueError(msg)
    if response_grid.size < 2:
        msg = "response_omega must contain at least two grid points"
        raise ValueError(msg)
    if np.any(np.diff(response_grid) <= 0):
        msg = "response_omega must be strictly increasing"
        raise ValueError(msg)

    kernel = 1.0 / (
        (response_grid[:, None] - omega0_arr[None, :]) ** 2 + float(eta) ** 2
    )
    transformed = np.trapezoid(
        basis_arr[:, :, None] * kernel[:, None, :],
        response_grid,
        axis=0,
    )
    return np.asarray(transformed.T, dtype=np.float64)


def fit_lit_basis_expansion(
    omega0: ArrayLike,
    lit: ArrayLike,
    eta: float,
    *,
    threshold: float = 0.0,
    response_omega: ArrayLike | None = None,
    response_points: int = 1000,
    response_max: float | None = None,
    basis_count: int = 8,
    alpha1_grid: ArrayLike | None = None,
    alpha2_grid: ArrayLike | None = None,
    l2_grid: ArrayLike | None = None,
    errors: ArrayLike | None = None,
) -> BasisLITInversionResult:
    """Invert LIT samples with the paper's regularized basis expansion."""
    omega0_arr, lit_arr, sigma = _prepare_fit_inputs(omega0, lit, errors)
    response_grid = _response_grid(
        omega0_arr,
        eta,
        threshold=float(threshold),
        response_omega=response_omega,
        response_points=response_points,
        response_max=response_max,
    )
    alpha1_values = _as_grid(
        alpha1_grid,
        default=np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float64),
        name="alpha1_grid",
    )
    alpha2_values = _as_grid(
        alpha2_grid,
        default=_auto_alpha2_grid(response_grid, float(threshold)),
        name="alpha2_grid",
    )
    l2_values = _as_grid(
        l2_grid,
        default=np.asarray([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2], dtype=np.float64),
        name="l2_grid",
    )

    best: BasisLITInversionResult | None = None
    for alpha1 in alpha1_values:
        for alpha2 in alpha2_values:
            basis = lit_response_basis(
                response_grid,
                threshold=float(threshold),
                basis_count=int(basis_count),
                alpha1=float(alpha1),
                alpha2=float(alpha2),
            )
            transform = lit_basis_transform(omega0_arr, response_grid, basis, eta)
            for l2_regularization in l2_values:
                candidate = _fit_linear_coefficients(
                    transform,
                    basis,
                    response_grid,
                    lit_arr,
                    sigma,
                    alpha1=float(alpha1),
                    alpha2=float(alpha2),
                    l2_regularization=float(l2_regularization),
                )
                if best is None or candidate.objective < best.objective:
                    best = candidate
    if best is None:
        msg = "failed to fit any LIT inversion candidate"
        raise RuntimeError(msg)
    return best


def _prepare_fit_inputs(
    omega0: ArrayLike,
    lit: ArrayLike,
    errors: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    omega0_arr = np.asarray(omega0, dtype=np.float64)
    lit_arr = np.asarray(lit, dtype=np.float64)
    if omega0_arr.ndim != 1 or lit_arr.ndim != 1:
        msg = "omega0 and lit must be one-dimensional"
        raise ValueError(msg)
    if omega0_arr.shape != lit_arr.shape:
        msg = f"omega0 and lit shapes differ: {omega0_arr.shape}, {lit_arr.shape}"
        raise ValueError(msg)
    if omega0_arr.size < 2:
        msg = "at least two LIT grid points are required for inversion"
        raise ValueError(msg)
    if np.any(np.diff(omega0_arr) < 0):
        msg = "omega0 grid must be sorted in ascending order"
        raise ValueError(msg)
    if not np.all(np.isfinite(lit_arr)):
        msg = "lit contains non-finite values"
        raise ValueError(msg)
    sigma = (
        np.ones_like(lit_arr)
        if errors is None
        else np.asarray(errors, dtype=np.float64)
    )
    if sigma.shape != lit_arr.shape:
        msg = f"errors shape {sigma.shape} does not match lit {lit_arr.shape}"
        raise ValueError(msg)
    return omega0_arr, lit_arr, np.where(sigma > 0.0, sigma, 1.0)


def _fit_linear_coefficients(
    transform: NDArray[np.float64],
    basis: NDArray[np.float64],
    response_grid: NDArray[np.float64],
    lit: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    alpha1: float,
    alpha2: float,
    l2_regularization: float,
) -> BasisLITInversionResult:
    weighted_transform = transform / sigma[:, None]
    weighted_lit = lit / sigma
    lhs = weighted_transform.T @ weighted_transform
    lhs = lhs + float(l2_regularization) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = weighted_transform.T @ weighted_lit
    coefficients = np.asarray(np.linalg.solve(lhs, rhs), dtype=np.float64)
    fit_lit = np.asarray(transform @ coefficients, dtype=np.float64)
    residual = (fit_lit - lit) / sigma
    chi2 = float(np.sum(residual**2))
    objective = float(chi2 + float(l2_regularization) * np.sum(coefficients**2))
    response = np.asarray(basis @ coefficients, dtype=np.float64)
    return BasisLITInversionResult(
        response_omega=response_grid,
        response=response,
        fit_lit=fit_lit,
        coefficients=coefficients,
        alpha1=float(alpha1),
        alpha2=float(alpha2),
        l2_regularization=float(l2_regularization),
        chi2=chi2,
        objective=objective,
    )


def _response_grid(
    omega0: NDArray[np.float64],
    eta: float,
    *,
    threshold: float,
    response_omega: ArrayLike | None,
    response_points: int,
    response_max: float | None,
) -> NDArray[np.float64]:
    if response_omega is not None:
        grid = np.asarray(response_omega, dtype=np.float64)
    else:
        if response_points < 2:
            msg = f"response_points must be at least 2, got {response_points}"
            raise ValueError(msg)
        upper = (
            float(response_max)
            if response_max is not None
            else _auto_response_max(omega0, eta, threshold)
        )
        if upper <= threshold:
            msg = (
                "response_max must be larger than threshold, got "
                f"{upper} <= {threshold}"
            )
            raise ValueError(msg)
        grid = np.linspace(float(threshold), upper, int(response_points))
    if grid.ndim != 1 or grid.size < 2:
        msg = "response_omega must be a one-dimensional grid with at least two points"
        raise ValueError(msg)
    if np.any(np.diff(grid) <= 0):
        msg = "response_omega must be strictly increasing"
        raise ValueError(msg)
    return grid


def _auto_response_max(
    omega0: NDArray[np.float64],
    eta: float,
    threshold: float,
) -> float:
    span = max(float(np.max(omega0) - threshold), float(eta))
    return max(float(np.max(omega0) + 8.0 * eta), float(threshold + 1.25 * span))


def _auto_alpha2_grid(
    response_grid: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    span = max(float(response_grid[-1] - threshold), np.finfo(np.float64).eps)
    return np.geomspace(0.2 / span, 20.0 / span, 12)


def _as_grid(
    values: ArrayLike | None,
    *,
    default: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    if values is None:
        result = default
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.size == 0:
            result = default
    result = np.ravel(result).astype(np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        msg = f"{name} must contain finite values"
        raise ValueError(msg)
    return result
