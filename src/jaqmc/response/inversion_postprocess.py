# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit, CPU-only postprocessing of saved molecular LIT archives."""

from __future__ import annotations

import logging
from dataclasses import field
from pathlib import Path
from typing import SupportsInt, cast

import numpy as np
from numpy.typing import ArrayLike
from upath import UPath

from jaqmc.response.inversion import initialize_lit_poles
from jaqmc.response.inversion_io import (
    LITInversionSettings,
    aggregate_lit_npz,
    invert_lit_npz,
    lit_inversion_npz_payload,
)
from jaqmc.utils.config import ConfigManager, configurable_dataclass

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


@configurable_dataclass
class LITInversionPostprocessConfig:
    """Configuration for an explicitly requested saved-LIT inversion.

    This configuration is intentionally separate from ``molecule lit``.  The
    expensive NQS calculation always stops after writing its raw NPZ; users may
    invoke this postprocessor repeatedly with different model orders.
    """

    input_paths: tuple[str, ...] = field(default_factory=tuple)
    output_path: str = "lit_inversion.npz"
    assume_independent: bool = False
    threshold: float | None = None
    ionized_energy: float | None = None
    require_determined: bool = True

    # A positive model order finds initial pole locations from the saved LIT.
    # It is a manual hypothesis and is never inferred from experimental peaks.
    pole_count: int = 0
    pole_search_energy_min: float | None = None
    pole_search_energy_max: float | None = None
    pole_search_grid_points: int = 0
    pole_minimum_separation: float | None = None

    pole_energies: tuple[float, ...] = field(default_factory=tuple)
    fit_pole_energies: bool = False
    pole_energy_bounds: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    continuum_grid: tuple[float, ...] = field(default_factory=tuple)
    continuum_regularization: float = 0.0
    covariance_relative_tolerance: float = 1e-10
    max_fitted_poles: int = 8
    pole_fit_tolerance: float = 1e-7
    pole_fit_max_iterations: int = 200
    solver_tolerance: float = 1e-10
    solver_max_iterations: int | None = None


class LITInversionPostprocessor:
    """Load raw LIT NPZs and write one explicitly requested inversion NPZ."""

    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.config = cfg.get("inversion", LITInversionPostprocessConfig)
        self._validate_config()

    def __call__(self, dry_run: bool = False) -> None:
        self.cfg.finalize()
        if not dry_run:
            self.run()

    def _validate_config(self) -> None:  # noqa: C901
        config = self.config
        paths = config.input_paths
        if not paths or any(not isinstance(path, str) or not path for path in paths):
            msg = "inversion.input_paths must contain at least one nonempty path."
            raise ValueError(msg)
        normalized_inputs = tuple(Path(path).expanduser().resolve() for path in paths)
        if len(set(normalized_inputs)) != len(normalized_inputs):
            msg = "inversion.input_paths must be unique."
            raise ValueError(msg)
        if not config.output_path or not config.output_path.endswith(".npz"):
            msg = "inversion.output_path must be a nonempty '.npz' path."
            raise ValueError(msg)
        normalized_output = Path(config.output_path).expanduser().resolve()
        if normalized_output in normalized_inputs:
            msg = "inversion.output_path must not overwrite a raw LIT input."
            raise ValueError(msg)
        if (config.threshold is None) == (config.ionized_energy is None):
            msg = "Set exactly one of inversion.threshold and inversion.ionized_energy."
            raise ValueError(msg)
        for name in ("threshold", "ionized_energy"):
            value = getattr(config, name)
            if value is not None and not np.isfinite(value):
                msg = f"inversion.{name} must be finite or null."
                raise ValueError(msg)
        if config.threshold is not None and config.threshold <= 0.0:
            msg = "inversion.threshold must be positive."
            raise ValueError(msg)

        pole_count = _nonnegative_integer(config.pole_count, "pole_count")
        grid_points = _nonnegative_integer(
            config.pole_search_grid_points,
            "pole_search_grid_points",
        )
        if pole_count and config.pole_energies:
            msg = "inversion.pole_count and inversion.pole_energies are exclusive."
            raise ValueError(msg)
        if pole_count and config.pole_energy_bounds:
            msg = "data-driven pole initialization generates its own bounds."
            raise ValueError(msg)
        if pole_count and not config.fit_pole_energies:
            msg = "inversion.fit_pole_energies must be true with pole_count."
            raise ValueError(msg)
        if pole_count > config.max_fitted_poles:
            msg = "inversion.pole_count cannot exceed max_fitted_poles."
            raise ValueError(msg)
        if pole_count and 0 < grid_points < max(4 * pole_count + 1, 17):
            msg = "inversion.pole_search_grid_points is too small for pole_count."
            raise ValueError(msg)
        for name in (
            "pole_search_energy_min",
            "pole_search_energy_max",
            "pole_minimum_separation",
        ):
            value = getattr(config, name)
            if value is not None and not np.isfinite(value):
                msg = f"inversion.{name} must be finite or null."
                raise ValueError(msg)
        if (
            config.pole_minimum_separation is not None
            and config.pole_minimum_separation <= 0.0
        ):
            msg = "inversion.pole_minimum_separation must be positive or null."
            raise ValueError(msg)
        if not pole_count and not config.pole_energies and not config.continuum_grid:
            msg = (
                "The manual inversion model is empty; set inversion.pole_count, "
                "pole_energies, or continuum_grid."
            )
            raise ValueError(msg)

    def run(self) -> None:
        config = self.config
        paths = tuple(config.input_paths)
        data = aggregate_lit_npz(
            paths,
            assume_independent=config.assume_independent,
        )
        threshold, threshold_payload = _resolve_threshold(config, paths)

        pole_initialization = None
        pole_energies: ArrayLike
        pole_bounds: ArrayLike
        if config.pole_count > 0:
            pole_initialization = initialize_lit_poles(
                data.omega,
                data.eta,
                data.signed_lit,
                threshold=threshold,
                pole_count=config.pole_count,
                energy_min=config.pole_search_energy_min,
                energy_max=config.pole_search_energy_max,
                candidate_grid_points=(config.pole_search_grid_points or None),
                minimum_separation=config.pole_minimum_separation,
                continuum_grid=config.continuum_grid,
                covariance=data.covariance,
                covariance_relative_tolerance=(config.covariance_relative_tolerance),
                continuum_regularization=config.continuum_regularization,
                solver_tolerance=config.solver_tolerance,
                solver_max_iterations=config.solver_max_iterations,
            )
            pole_energies = pole_initialization.pole_energies
            pole_bounds = pole_initialization.pole_energy_bounds
            logger.info(
                "Initialized manual K=%d hypothesis from saved LIT data: "
                "energies=%s objective=%.10e",
                config.pole_count,
                np.array2string(pole_energies, precision=10),
                pole_initialization.objective,
            )
        else:
            pole_energies = config.pole_energies
            pole_bounds = config.pole_energy_bounds

        settings = LITInversionSettings(
            threshold=threshold,
            pole_energies=pole_energies,
            continuum_grid=config.continuum_grid,
            continuum_regularization=config.continuum_regularization,
            fit_pole_energies=config.fit_pole_energies,
            pole_energy_bounds=(
                None
                if np.asarray(pole_bounds, dtype=np.float64).size == 0
                else pole_bounds
            ),
            covariance_relative_tolerance=config.covariance_relative_tolerance,
            max_fitted_poles=config.max_fitted_poles,
            pole_fit_tolerance=config.pole_fit_tolerance,
            pole_fit_max_iterations=config.pole_fit_max_iterations,
            solver_tolerance=config.solver_tolerance,
            solver_max_iterations=config.solver_max_iterations,
        )
        inversion = invert_lit_npz(
            paths,
            settings,
            assume_independent=config.assume_independent,
        )
        diagnostics = inversion.result.diagnostics
        if config.require_determined and diagnostics.underdetermined:
            reasons = "; ".join(diagnostics.underdetermined_reasons)
            msg = f"Manual LIT inversion is underdetermined: {reasons}"
            raise RuntimeError(msg)

        payload = lit_inversion_npz_payload(inversion)
        payload.update(threshold_payload)
        payload["manual_postprocess"] = np.asarray(True)
        payload["requested_pole_count"] = np.asarray(
            config.pole_count,
            dtype=np.int64,
        )
        payload["pole_initialization_method"] = np.asarray(
            "data_greedy_nnls" if pole_initialization is not None else "configured"
        )
        if pole_initialization is not None:
            payload["pole_initialization_objective"] = np.asarray(
                pole_initialization.objective,
                dtype=np.float64,
            )
            payload["pole_initialization_candidate_grid_points"] = np.asarray(
                pole_initialization.candidate_grid_points,
                dtype=np.int64,
            )
            payload["pole_initialization_minimum_separation"] = np.asarray(
                pole_initialization.minimum_separation,
                dtype=np.float64,
            )

        output_path = UPath(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f_out:
            np.savez_compressed(f_out, **payload)  # type: ignore[arg-type]
        logger.info(
            "Wrote manual LIT inversion to %s (poles=%s, eta_count=%d, "
            "cross_width_validated=%s, underdetermined=%s)",
            output_path,
            np.array2string(inversion.result.pole_energies, precision=10),
            diagnostics.unique_eta_count,
            diagnostics.cross_width_validated,
            diagnostics.underdetermined,
        )


def _nonnegative_integer(value: object, name: str) -> int:
    try:
        converted = int(cast(SupportsInt, value))
    except (TypeError, ValueError, OverflowError) as error:
        msg = f"inversion.{name} must be a nonnegative integer."
        raise ValueError(msg) from error
    if isinstance(value, (bool, np.bool_)) or converted != value or converted < 0:
        msg = f"inversion.{name} must be a nonnegative integer."
        raise ValueError(msg)
    return converted


def _resolve_threshold(
    config: LITInversionPostprocessConfig,
    paths: tuple[str, ...],
) -> tuple[float, dict[str, object]]:
    if config.threshold is not None:
        return float(config.threshold), {
            "threshold_source": np.asarray("configured"),
        }
    assert config.ionized_energy is not None
    ground_energies = _load_ground_energies(paths)
    reference = float(ground_energies[0])
    if not np.allclose(ground_energies, reference, rtol=1e-10, atol=1e-10):
        msg = (
            "input LIT files use inconsistent ground energies; a joint inversion "
            "requires one common threshold"
        )
        raise ValueError(msg)
    threshold = float(config.ionized_energy) - reference
    if not np.isfinite(threshold) or threshold <= 0.0:
        msg = f"The derived inversion threshold must be positive; got {threshold}."
        raise ValueError(msg)
    return threshold, {
        "threshold_source": np.asarray("ionized_energy_minus_ground_energy"),
        "ionized_energy": np.asarray(config.ionized_energy, dtype=np.float64),
        "source_ground_energies": ground_energies,
    }


def _load_ground_energies(paths: tuple[str, ...]) -> np.ndarray:
    values = []
    for raw_path in paths:
        path = str(Path(raw_path).expanduser())
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "ground_energy" not in archive.files:
                    msg = f"{path}: ground_energy is required with ionized_energy."
                    raise ValueError(msg)
                raw = np.asarray(archive["ground_energy"], dtype=np.float64)
        except (OSError, EOFError) as error:
            msg = f"could not read workflow NPZ {path}: {error}"
            raise ValueError(msg) from error
        if raw.ndim != 0 or not np.isfinite(raw.item()):
            msg = f"{path}: ground_energy must be a finite scalar."
            raise ValueError(msg)
        values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


__all__ = [
    "LITInversionPostprocessConfig",
    "LITInversionPostprocessor",
]
