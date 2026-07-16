# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Summarize a from-scratch ground-symmetry diagnostic/treatment A/B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _scalar_summary(values: np.ndarray) -> dict[str, float] | None:
    values = _finite(values)
    if values.size == 0:
        return None
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _active_values(stats: h5py.File, field: str, active_field: str) -> np.ndarray:
    if field not in stats or active_field not in stats:
        return np.asarray([], dtype=np.float64)
    active = np.asarray(stats[active_field]) > 0.5
    values = np.asarray(stats[field])
    return values[active]


def _summarize_stage(path: Path) -> dict[str, object]:
    with h5py.File(path) as stats:
        result: dict[str, object] = {"steps": len(next(iter(stats.values())))}
        for field in (
            "ground_symmetry_eval_max",
            "ground_symmetry_eval_mean",
            "ground_symmetry_eval_unweighted_max",
            "ground_symmetry_eval_source_ess_fraction",
            "ground_symmetry_eval_pass",
        ):
            result[field] = _scalar_summary(
                _active_values(stats, field, "ground_symmetry_eval_active")
            )

        updates = _active_values(
            stats,
            "ground_symmetry_update_norm",
            "ground_symmetry_active",
        )
        energy_updates = _active_values(
            stats,
            "ground_symmetry_energy_update_norm",
            "ground_symmetry_active",
        )
        result["symmetry_updates"] = int(updates.size)
        result["ground_symmetry_update_norm"] = _scalar_summary(updates)
        valid_energy = energy_updates > 0.0
        result["symmetry_to_energy_update_ratio"] = _scalar_summary(
            updates[valid_energy] / energy_updates[valid_energy]
        )

        if "total_energy" in stats:
            energy = _finite(np.asarray(stats["total_energy"]))
            tail = energy[-min(100, energy.size) :]
            result["energy_last_100"] = (
                {
                    "count": int(tail.size),
                    "mean": float(np.mean(tail)),
                    "std": float(np.std(tail)),
                }
                if tail.size
                else None
            )
        if "total_energy_var" in stats:
            result["energy_variance_last_100"] = _scalar_summary(
                np.asarray(stats["total_energy_var"])[-100:]
            )

        symmetry_acceptance = _active_values(
            stats,
            "symmetry_pmove",
            "symmetry_move_active",
        )
        result["global_symmetry_acceptance"] = _scalar_summary(symmetry_acceptance)
        return result


def summarize(root: Path) -> dict[str, object]:
    result: dict[str, object] = {"root": str(root), "cases": {}}
    cases: dict[str, object] = result["cases"]  # type: ignore[assignment]
    for case in ("baseline", "treatment"):
        case_result: dict[str, object] = {}
        for stage in ("pretrain", "train"):
            path = root / case / f"{stage}_stats.h5"
            case_result[stage] = _summarize_stage(path) if path.exists() else None
        cases[case] = case_result

    try:
        baseline = cases["baseline"]["train"]  # type: ignore[index]
        treatment = cases["treatment"]["train"]  # type: ignore[index]
        baseline_final = baseline["ground_symmetry_eval_max"]["last"]
        treatment_final = treatment["ground_symmetry_eval_max"]["last"]
        result["train_eval_max_treatment_to_baseline_ratio"] = float(
            treatment_final / baseline_final
        )
        result["train_eval_max_baseline_to_treatment_ratio"] = float(
            baseline_final / treatment_final
        )
    except (KeyError, TypeError, ZeroDivisionError):
        result["train_eval_max_treatment_to_baseline_ratio"] = None
        result["train_eval_max_baseline_to_treatment_ratio"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
