#!/usr/bin/env bash

# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -u -o pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=$1
config="$repo_root/runs/he_atom_lit_formal/ground_symmetry_pilot.yml"

mkdir -p "$output_root"

run_case() {
  local name=$1
  shift
  local output_path="$output_root/$name"
  if [[ -e "$output_path" ]]; then
    echo "AB_CASE_REFUSED,name=$name,path=$output_path,reason=already_exists"
    return 2
  fi

  echo "AB_CASE_START,name=$name,path=$output_path"
  local started=$SECONDS
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$repo_root/.venv-gpu/bin/jaqmc" molecule train \
    --yml "$config" \
    "workflow.save_path=$output_path" \
    "workflow.restore_path=$output_path" \
    "$@"
  local status=$?
  echo "AB_CASE_COMPLETE,name=$name,status=$status,elapsed_seconds=$((SECONDS - started))"
  return "$status"
}

echo "AB_START,output_root=$output_root"
run_case \
  baseline \
  ground_symmetry.enabled=true \
  ground_symmetry.updates_enabled=false \
  ground_symmetry.global_mcmc_enabled=false
baseline_status=$?

run_case \
  treatment \
  ground_symmetry.enabled=true \
  ground_symmetry.updates_enabled=true \
  ground_symmetry.global_mcmc_enabled=true
treatment_status=$?

summary_status=1
if "$repo_root/.venv-gpu/bin/python" \
  "$repo_root/scripts/summarize_ground_symmetry_ab.py" \
  "$output_root" >"$output_root/summary.json"; then
  summary_status=0
fi

success=0
if [[ $baseline_status -eq 0 ]]; then
  success=$((success + 1))
fi
if [[ $treatment_status -eq 0 ]]; then
  success=$((success + 1))
fi
failed=$((2 - success))
echo "AB_COMPLETE,n_success=$success,n_failed=$failed,summary_status=$summary_status"

if [[ $failed -ne 0 || $summary_status -ne 0 ]]; then
  exit 1
fi
