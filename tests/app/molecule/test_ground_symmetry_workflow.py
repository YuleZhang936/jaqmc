# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import h5py
import numpy as np

from jaqmc.app.molecule import MoleculeTrainWorkflow
from jaqmc.utils.config import ConfigManager


def test_ground_symmetry_workflow_runs_stateless_post_kfac_update(tmp_path):
    config = ConfigManager(
        {
            "workflow": {"seed": 7, "save_path": str(tmp_path), "batch_size": 8},
            "system": {"module": "atom", "symbol": "H"},
            "wf": {
                "ndets": 2,
                "hidden_dims_single": [8, 8],
                "hidden_dims_double": [4, 4],
            },
            "sampler": {"steps": 1},
            "ground_symmetry": {
                "enabled": True,
                "atom_random_rotation_quartets": 0,
                "pretrain_enabled": False,
                "train_update_interval": 1,
                "train_source_weight_warmup_steps": 1,
                "update_batch_size": 4,
                "evaluation_interval": 1,
                "mcmc_global_step_interval": 1,
            },
            "pretrain": {"run": {"iterations": 0}},
            "train": {
                "run": {
                    "burn_in": 0,
                    "iterations": 1,
                    "save_step_interval": 1,
                    "save_time_interval": 0,
                }
            },
        }
    )

    MoleculeTrainWorkflow(config)()

    with h5py.File(tmp_path / "train_stats.h5") as stats:
        required = {
            "ground_symmetry_loss",
            "ground_symmetry_active",
            "ground_symmetry_covariance_loss",
            "ground_symmetry_log_amplitude_loss",
            "ground_symmetry_update_norm",
            "ground_symmetry_energy_update_norm",
            "ground_symmetry_eval_max",
            "ground_symmetry_eval_active",
            "ground_symmetry_valid",
            "symmetry_pmove",
            "symmetry_move_active",
        }
        assert required <= set(stats)
        values = {name: float(stats[name][-1]) for name in required}

    assert all(np.isfinite(value) for value in values.values())
    np.testing.assert_allclose(values["ground_symmetry_valid"], 1.0)
    np.testing.assert_allclose(values["ground_symmetry_active"], 1.0)
    np.testing.assert_allclose(values["ground_symmetry_eval_active"], 1.0)
    np.testing.assert_allclose(values["symmetry_move_active"], 1.0)
    assert values["ground_symmetry_update_norm"] <= 1.0e-3 + 1.0e-7
    assert values["ground_symmetry_update_norm"] <= (
        0.05 * values["ground_symmetry_energy_update_norm"] + 1.0e-7
    )


def test_diagnostic_only_mode_preserves_ordinary_vmc_state(tmp_path):
    ordinary_path = tmp_path / "ordinary"
    diagnostic_path = tmp_path / "diagnostic"

    def make_config(save_path, ground_symmetry):
        return ConfigManager(
            {
                "workflow": {
                    "seed": 19,
                    "save_path": str(save_path),
                    "batch_size": 4,
                },
                "system": {"module": "atom", "symbol": "H"},
                "wf": {
                    "ndets": 1,
                    "hidden_dims_single": [4],
                    "hidden_dims_double": [2],
                },
                "sampler": {"steps": 1},
                "ground_symmetry": ground_symmetry,
                "pretrain": {
                    "run": {
                        "burn_in": 0,
                        "iterations": 1,
                        "save_step_interval": 1,
                        "save_time_interval": 0,
                    }
                },
                "train": {
                    "run": {
                        "burn_in": 0,
                        "iterations": 1,
                        "save_step_interval": 1,
                        "save_time_interval": 0,
                    }
                },
            }
        )

    MoleculeTrainWorkflow(make_config(ordinary_path, {"enabled": False}))()
    MoleculeTrainWorkflow(
        make_config(
            diagnostic_path,
            {
                "enabled": True,
                "updates_enabled": False,
                "global_mcmc_enabled": False,
                "atom_random_rotation_quartets": 0,
                "update_batch_size": 4,
                "evaluation_interval": 1,
            },
        )
    )()

    for stage in ("pretrain", "train"):
        with (
            np.load(ordinary_path / f"{stage}_ckpt_000000.npz") as ordinary,
            np.load(diagnostic_path / f"{stage}_ckpt_000000.npz") as diagnostic,
        ):
            assert ordinary.files == diagnostic.files
            for name in ordinary.files:
                np.testing.assert_array_equal(ordinary[name], diagnostic[name])

    for stage in ("pretrain", "train"):
        with h5py.File(diagnostic_path / f"{stage}_stats.h5") as stats:
            np.testing.assert_allclose(stats["ground_symmetry_active"][-1], 0.0)
            np.testing.assert_allclose(stats["ground_symmetry_update_norm"][-1], 0.0)
            np.testing.assert_allclose(stats["ground_symmetry_eval_active"][-1], 1.0)
            assert np.isfinite(stats["ground_symmetry_eval_max"][-1])
            assert "symmetry_pmove" not in stats
