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
training_path="$output_root/training"
config="$repo_root/runs/he_atom_lit_formal/ground_symmetry.yml"

mkdir -p "$output_root"
if [[ -e "$training_path" ]]; then
  echo "GROUND_REFUSED,path=$training_path,reason=already_exists"
  exit 2
fi

echo "GROUND_START,path=$training_path"
started=$SECONDS
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$repo_root/.venv-gpu/bin/jaqmc" molecule train \
  --yml "$config" \
  "workflow.save_path=$training_path" \
  "workflow.restore_path=$training_path"
status=$?
echo "GROUND_COMPLETE,status=$status,elapsed_seconds=$((SECONDS - started))"
exit "$status"
