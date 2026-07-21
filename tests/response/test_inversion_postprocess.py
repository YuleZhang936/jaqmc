# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from jaqmc.app.molecule.lit_workflow import MolecularLITConfig
from jaqmc.response.inversion import forward_lit
from jaqmc.response.inversion_postprocess import LITInversionPostprocessor
from jaqmc.utils.config import ConfigManager


def _write_single_pole_lit(path: Path, *, ground_energy: float = -2.903) -> None:
    omega = np.linspace(0.75, 0.90, 81)
    eta = 0.003
    signed_lit = forward_lit(
        omega,
        eta,
        pole_energies=np.asarray([0.78]),
        pole_strengths=np.asarray([1.25]),
    )[np.newaxis, :]
    phase = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    profile = np.linspace(-1.0, 1.0, omega.size)
    blocks = (
        signed_lit[..., np.newaxis]
        + (1e-5 * profile[:, np.newaxis] * np.sin(phase)[np.newaxis, :])[
            np.newaxis, ...
        ]
    )
    np.savez(
        path,
        omega=omega,
        eta=eta,
        axes="x",
        axis_indices=np.asarray([0]),
        signed_lit=signed_lit,
        signed_lit_jackknife_blocks=blocks,
        signed_lit_jackknife_block_count=blocks.shape[-1],
        eval_pool_sha256=np.asarray(["pool-x"]),
        error_bound_monitor=np.full_like(signed_lit, 1e-6),
        error_d_valid=np.ones_like(signed_lit, dtype=np.bool_),
        ground_energy=np.asarray(ground_energy),
    )


def test_molecule_lit_config_has_no_inversion_controls():
    config = MolecularLITConfig()

    assert not hasattr(config, "inversion_enabled")
    assert not hasattr(config, "inversion_pole_energies")


def test_manual_postprocessor_infers_threshold_and_writes_separate_output(
    tmp_path: Path,
):
    input_path = tmp_path / "lit_spectrum.npz"
    output_path = tmp_path / "inversion_k1.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "output_path": str(output_path),
                "ionized_energy": -2.0,
                "pole_count": 1,
                "fit_pole_energies": True,
                "max_fitted_poles": 1,
                "pole_fit_tolerance": 1e-9,
            }
        }
    )

    LITInversionPostprocessor(cfg)()

    assert input_path.exists()
    assert output_path.exists()
    with np.load(output_path, allow_pickle=False) as result:
        assert result["manual_postprocess"].item()
        assert result["requested_pole_count"].item() == 1
        assert result["pole_initialization_method"].item() == "data_greedy_nnls"
        assert result["threshold_source"].item() == (
            "ionized_energy_minus_ground_energy"
        )
        assert result["threshold"].item() == pytest.approx(0.903)
        np.testing.assert_allclose(result["pole_energies"], [0.78], atol=2e-7)


def test_manual_postprocessor_requires_an_explicit_model_hypothesis(tmp_path: Path):
    input_path = tmp_path / "lit_spectrum.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "threshold": 0.903,
            }
        }
    )

    with pytest.raises(ValueError, match="manual inversion model is empty"):
        LITInversionPostprocessor(cfg)


def test_manual_postprocessor_never_overwrites_raw_input(tmp_path: Path):
    input_path = tmp_path / "lit_spectrum.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "output_path": str(input_path),
                "threshold": 0.903,
                "pole_energies": [0.78],
            }
        }
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        LITInversionPostprocessor(cfg)


def test_manual_postprocessor_rejects_peak_count_mixed_with_peak_locations(
    tmp_path: Path,
):
    input_path = tmp_path / "lit_spectrum.npz"
    _write_single_pole_lit(input_path)
    cfg = ConfigManager(
        {
            "inversion": {
                "input_paths": [str(input_path)],
                "threshold": 0.903,
                "pole_count": 1,
                "pole_energies": [0.78],
                "fit_pole_energies": True,
            }
        }
    )

    with pytest.raises(ValueError, match="exclusive"):
        LITInversionPostprocessor(cfg)
