# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Summarize short warm-start NQS-LIT residual-scale/KL diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_ITERATION_RE = re.compile(
    r"stage=warm_start\s+omega=(?P<omega>\S+)\s+iter=(?P<iteration>\d+).*?"
    r"train_loss=(?P<train_loss>\S+)\s+"
    r"train_fidelity=(?P<train_fidelity>\S+)\s+"
    r"train_reverse_kl=(?P<train_reverse_kl>\S+).*?"
    r"best_iter=(?P<best_iteration>\d+)\s+"
    r"best_fidelity=(?P<best_fidelity>\S+)\s+"
    r"best_reverse_kl=(?P<best_reverse_kl>\S+).*?"
    r"best_covariance_mean=(?P<covariance_mean>\S+)\s+"
    r"best_covariance_max=(?P<covariance_max>\S+)"
)
_SELECTED_RE = re.compile(
    r"stage=warm_start\s+omega=(?P<omega>\S+)\s+"
    r"selected_iter=(?P<selected_iteration>\d+)/(?P<iterations>\d+)\s+"
    r"heldout_loss=(?P<loss>\S+)\s+"
    r"fidelity=(?P<fidelity>\S+)\s+"
    r"reverse_kl=(?P<reverse_kl>\S+)\s+"
    r"covariance_mean=(?P<covariance_mean>\S+)\s+"
    r"covariance_max=(?P<covariance_max>\S+)\s+ess=(?P<ess>\S+)"
)
_SPRING_RE = re.compile(
    r"stage=warm_start\s+omega=(?P<omega>\S+)\s+iter=(?P<iteration>\d+)\s+"
    r"spring_grad=(?P<combined_gradient_norm>\S+)\s+"
    r"spring_grad_fidelity=(?P<fidelity_gradient_norm>\S+)\s+"
    r"spring_grad_kl_weighted=(?P<weighted_reverse_kl_gradient_norm>\S+)\s+"
    r"spring_fidelity_kl_cosine=(?P<fidelity_kl_cosine>\S+)\s+"
    r"spring_gradient_cancellation=(?P<gradient_cancellation_ratio>\S+)\s+"
    r"spring_direction=(?P<direction_norm>\S+)\s+"
    r"spring_update=(?P<update_norm>\S+)\s+"
    r"spring_clip_factor=(?P<clip_factor>\S+)\s+"
    r"spring_clipped=(?P<clipped>\d+)\s+"
    r"spring_damping=(?P<damping>\S+)\s+"
    r"spring_qfi_mean_diagonal=(?P<qfi_mean_diagonal>\S+)\s+"
    r"spring_history_gradient_ratio=(?P<history_gradient_ratio>\S+)\s+"
    r"raw_grad_rms=(?P<raw_gradient_rms>\S+)\s+"
    r"raw_update=(?P<raw_update_norm>\S+)\s+"
    r"source_coefficient_grad_rms=(?P<source_coefficient_gradient_rms>\S+)\s+"
    r"source_coefficient_update=(?P<source_coefficient_update_norm>\S+)\s+"
    r"residual_log_scale_grad_rms=(?P<residual_log_scale_gradient_rms>\S+)\s+"
    r"residual_log_scale_update=(?P<residual_log_scale_update_norm>\S+)"
)
_COMPLETE_RE = re.compile(
    r"RESPONSE_AB_CASE_COMPLETE,name=(?P<name>[^,]+),"
    r"status=(?P<status>\d+),elapsed_seconds=(?P<elapsed_seconds>\d+)"
)


def _read_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _typed_match(match: re.Match[str]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for key, value in match.groupdict().items():
        if key in {
            "iteration",
            "best_iteration",
            "selected_iteration",
            "iterations",
            "clipped",
        }:
            result[key] = int(value)
        else:
            result[key] = _number(value)
    return result


def _last_spring_value(case: dict[str, object], key: str) -> float | int | None:
    record = case.get("last_spring_record")
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    return value if isinstance(value, (float, int)) else None


def _selected_value(case: dict[str, object], key: str) -> float | int | None:
    record = case.get("selected")
    if not isinstance(record, dict):
        return None
    value = record.get(key)
    return value if isinstance(value, (float, int)) else None


def _summarize_case(case_dir: Path) -> dict[str, object]:
    metadata = _read_metadata(case_dir / "case.env")
    log_path = case_dir / "run.log"
    result: dict[str, object] = {
        "name": metadata.get("name", case_dir.name),
        "path": str(case_dir),
        "residual_scale": _number(metadata.get("residual_scale", "")),
        "kl_weight": _number(metadata.get("kl_weight", "")),
        "warm_iterations_requested": int(metadata["warm_iterations"])
        if metadata.get("warm_iterations", "").isdigit()
        else None,
        "seed": int(metadata["seed"]) if metadata.get("seed", "").isdigit() else None,
        "completed": False,
        "status": None,
        "elapsed_seconds": None,
        "warm_records": 0,
        "first_record": None,
        "last_record": None,
        "selected": None,
        "spring_records": 0,
        "first_spring_record": None,
        "last_spring_record": None,
        "heldout_fidelity_gain_from_first_record": None,
        "heldout_regularized_loss_gain_from_first_record": None,
    }
    if not log_path.exists():
        return result

    text = log_path.read_text(errors="replace")
    iterations = [_typed_match(match) for match in _ITERATION_RE.finditer(text)]
    selected_matches = list(_SELECTED_RE.finditer(text))
    spring_records = [_typed_match(match) for match in _SPRING_RE.finditer(text)]
    complete_matches = list(_COMPLETE_RE.finditer(text))
    result["warm_records"] = len(iterations)
    if iterations:
        result["first_record"] = iterations[0]
        result["last_record"] = iterations[-1]
    result["spring_records"] = len(spring_records)
    if spring_records:
        result["first_spring_record"] = spring_records[0]
        result["last_spring_record"] = spring_records[-1]
    if selected_matches:
        selected = _typed_match(selected_matches[-1])
        result["selected"] = selected
        if iterations:
            first = iterations[0]
            fidelity = selected.get("fidelity")
            first_fidelity = first.get("best_fidelity")
            if isinstance(fidelity, float) and isinstance(first_fidelity, float):
                result["heldout_fidelity_gain_from_first_record"] = (
                    fidelity - first_fidelity
                )
            loss = selected.get("loss")
            first_best_fidelity = first.get("best_fidelity")
            first_best_kl = first.get("best_reverse_kl")
            kl_weight = result["kl_weight"]
            if (
                isinstance(loss, float)
                and isinstance(first_best_fidelity, float)
                and isinstance(first_best_kl, float)
                and isinstance(kl_weight, float)
            ):
                first_loss = 1.0 - first_best_fidelity + kl_weight * first_best_kl
                result["heldout_regularized_loss_gain_from_first_record"] = (
                    first_loss - loss
                )
    if complete_matches:
        complete = complete_matches[-1].groupdict()
        result["completed"] = True
        result["status"] = int(complete["status"])
        result["elapsed_seconds"] = int(complete["elapsed_seconds"])
    return result


def summarize(root: Path) -> dict[str, object]:
    cases = [
        _summarize_case(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "case.env").exists()
    ]

    def ranking_key(case: dict[str, object]) -> tuple[bool, float]:
        selected = case.get("selected")
        fidelity = selected.get("fidelity") if isinstance(selected, dict) else None
        return (not isinstance(fidelity, float), -(fidelity or 0.0))

    ranking = [
        {
            "name": case["name"],
            "residual_scale": case["residual_scale"],
            "kl_weight": case["kl_weight"],
            "selected_fidelity": _selected_value(case, "fidelity"),
            "selected_reverse_kl": _selected_value(case, "reverse_kl"),
            "selected_loss": _selected_value(case, "loss"),
            "fidelity_gain_from_first_record": case[
                "heldout_fidelity_gain_from_first_record"
            ],
            "last_spring_combined_gradient_norm": _last_spring_value(
                case, "combined_gradient_norm"
            ),
            "last_spring_fidelity_gradient_norm": _last_spring_value(
                case, "fidelity_gradient_norm"
            ),
            "last_spring_weighted_reverse_kl_gradient_norm": _last_spring_value(
                case, "weighted_reverse_kl_gradient_norm"
            ),
            "last_spring_fidelity_kl_cosine": _last_spring_value(
                case, "fidelity_kl_cosine"
            ),
            "last_spring_gradient_cancellation_ratio": _last_spring_value(
                case, "gradient_cancellation_ratio"
            ),
            "last_spring_direction_norm": _last_spring_value(case, "direction_norm"),
            "last_spring_update_norm": _last_spring_value(case, "update_norm"),
            "last_spring_clip_factor": _last_spring_value(case, "clip_factor"),
            "last_spring_clipped": _last_spring_value(case, "clipped"),
            "last_spring_damping": _last_spring_value(case, "damping"),
            "last_spring_qfi_mean_diagonal": _last_spring_value(
                case, "qfi_mean_diagonal"
            ),
            "last_spring_history_gradient_ratio": _last_spring_value(
                case, "history_gradient_ratio"
            ),
            "last_spring_raw_gradient_rms": _last_spring_value(
                case, "raw_gradient_rms"
            ),
            "last_spring_raw_update_norm": _last_spring_value(case, "raw_update_norm"),
            "last_spring_source_coefficient_gradient_rms": _last_spring_value(
                case, "source_coefficient_gradient_rms"
            ),
            "last_spring_source_coefficient_update_norm": _last_spring_value(
                case, "source_coefficient_update_norm"
            ),
            "last_spring_residual_log_scale_gradient_rms": _last_spring_value(
                case, "residual_log_scale_gradient_rms"
            ),
            "last_spring_residual_log_scale_update_norm": _last_spring_value(
                case, "residual_log_scale_update_norm"
            ),
        }
        for case in sorted(cases, key=ranking_key)
        if isinstance(case.get("selected"), dict)
    ]
    return {
        "root": str(root),
        "n_cases": len(cases),
        "n_completed_successfully": sum(
            case.get("completed") is True and case.get("status") == 0 for case in cases
        ),
        "ranking_by_selected_fidelity": ranking,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
