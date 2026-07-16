#!/usr/bin/env bash

# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Run paired, warm-start-only NQS-LIT optimizer diagnostics.  Every case uses
# the same ground checkpoint, fixed source pools, and RNG seed; only the
# source-aligned residual scale and reverse-KL weight change.

set -u -o pipefail

usage() {
  cat <<'EOF'
usage: run_response_optimizer_ab.sh \
  --ground-checkpoint PATH \
  --ground-energy HARTREE \
  --source-pool-dir PATH \
  --output-root PATH [options]

Required:
  --ground-checkpoint PATH   Ground-state checkpoint file.
  --ground-energy FLOAT      Fixed ground energy in Hartree.
  --source-pool-dir PATH     Directory containing axis_x_{train,eval}.npz.
  --output-root PATH         New parent directory for the six case outputs.

Options:
  --config PATH              LIT YAML (default: runs/he_atom_lit_formal/lit.yml).
  --residual-scales CSV      Default: 0.01,0.1.
  --kl-weights CSV           Default: 1,0.1,0.
  --warm-iterations N        Default: 300.
  --selection-interval N     Default: 50.
  --log-interval N           Default: 50.
  --warm-omega FLOAT         Default: -3.674932217565499.
  --probe-delta FLOAT        Tiny positive target offset (default: 1e-6).
  --source-norm FLOAT        Reuse a measured He source norm (default: unset).
  --seed N                   Identical seed for every case (default: 6789).
  --device N                 CUDA device index (default: 0).
  --jaqmc PATH               jaqmc executable (default: .venv-gpu/bin/jaqmc).
  --python PATH              Python executable (default: .venv-gpu/bin/python).
  -h, --help                 Show this help.

The reported spectrum contains one intentionally negligible one-step probe at
warm_omega + probe_delta.  No optimized continuation bridge is run; the data
of interest are the stage=warm_start records in each run.log.
EOF
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="$repo_root/runs/he_atom_lit_formal/lit.yml"
jaqmc="$repo_root/.venv-gpu/bin/jaqmc"
python="$repo_root/.venv-gpu/bin/python"
ground_checkpoint=""
ground_energy=""
source_pool_dir=""
output_root=""
source_norm=""
residual_scales_csv="0.01,0.1"
kl_weights_csv="1,0.1,0"
warm_iterations=300
selection_interval=50
log_interval=50
warm_omega=-3.674932217565499
probe_delta=1e-6
seed=6789
device=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ground-checkpoint)
      ground_checkpoint=${2-}
      shift 2
      ;;
    --ground-energy)
      ground_energy=${2-}
      shift 2
      ;;
    --source-pool-dir)
      source_pool_dir=${2-}
      shift 2
      ;;
    --output-root)
      output_root=${2-}
      shift 2
      ;;
    --source-norm)
      source_norm=${2-}
      shift 2
      ;;
    --config)
      config=${2-}
      shift 2
      ;;
    --residual-scales)
      residual_scales_csv=${2-}
      shift 2
      ;;
    --kl-weights)
      kl_weights_csv=${2-}
      shift 2
      ;;
    --warm-iterations)
      warm_iterations=${2-}
      shift 2
      ;;
    --selection-interval)
      selection_interval=${2-}
      shift 2
      ;;
    --log-interval)
      log_interval=${2-}
      shift 2
      ;;
    --warm-omega)
      warm_omega=${2-}
      shift 2
      ;;
    --probe-delta)
      probe_delta=${2-}
      shift 2
      ;;
    --seed)
      seed=${2-}
      shift 2
      ;;
    --device)
      device=${2-}
      shift 2
      ;;
    --jaqmc)
      jaqmc=${2-}
      shift 2
      ;;
    --python)
      python=${2-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "RESPONSE_AB_REFUSED,reason=unknown_argument,value=$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ground_checkpoint" || -z "$ground_energy" || \
      -z "$source_pool_dir" || -z "$output_root" ]]; then
  echo "RESPONSE_AB_REFUSED,reason=missing_required_argument" >&2
  usage >&2
  exit 2
fi

for path in "$ground_checkpoint" "$config" \
  "$source_pool_dir/axis_x_train.npz" "$source_pool_dir/axis_x_eval.npz"; do
  if [[ ! -f "$path" ]]; then
    echo "RESPONSE_AB_REFUSED,reason=missing_file,path=$path" >&2
    exit 2
  fi
done
for executable in "$jaqmc" "$python"; do
  if [[ ! -x "$executable" ]]; then
    echo "RESPONSE_AB_REFUSED,reason=missing_executable,path=$executable" >&2
    exit 2
  fi
done

if [[ ! "$warm_iterations" =~ ^[1-9][0-9]*$ || \
      ! "$selection_interval" =~ ^[1-9][0-9]*$ || \
      ! "$log_interval" =~ ^[1-9][0-9]*$ || \
      ! "$seed" =~ ^[0-9]+$ || ! "$device" =~ ^[0-9]+$ ]]; then
  echo "RESPONSE_AB_REFUSED,reason=invalid_integer_option" >&2
  exit 2
fi

probe_omega=$(
  "$python" -c \
    'import math, sys
warm = float(sys.argv[1])
delta = float(sys.argv[2])
if not (math.isfinite(warm) and math.isfinite(delta) and delta > 0.0):
    raise SystemExit("warm omega must be finite and probe delta positive")
print(format(warm + delta, ".17g"))' \
    "$warm_omega" "$probe_delta"
) || {
  echo "RESPONSE_AB_REFUSED,reason=invalid_frequency_option" >&2
  exit 2
}

IFS=',' read -r -a residual_scales <<<"$residual_scales_csv"
IFS=',' read -r -a kl_weights <<<"$kl_weights_csv"
if [[ ${#residual_scales[@]} -eq 0 || ${#kl_weights[@]} -eq 0 ]]; then
  echo "RESPONSE_AB_REFUSED,reason=empty_case_grid" >&2
  exit 2
fi

slug() {
  local value=$1
  value=${value// /}
  value=${value//-/m}
  value=${value//+/p}
  value=${value//./p}
  value=${value//E/e}
  printf '%s' "$value"
}

mkdir -p "$output_root"
echo "RESPONSE_AB_START,output_root=$output_root,warm_omega=$warm_omega,probe_omega=$probe_omega,warm_iterations=$warm_iterations"

diagnostic_override_args=(
  lit.nqs_stage_fidelity_min=0
  lit.nqs_stage_reweight_ess_fraction_min=0
  lit.nqs_stage_fidelity_gain_min=0
  lit.nqs_continuation_allow_min_step_override=true
)
if [[ -n "$source_norm" ]]; then
  diagnostic_override_args+=("lit.nqs_source_norm_override=$source_norm")
fi

n_success=0
n_failed=0
for residual_scale_raw in "${residual_scales[@]}"; do
  residual_scale=${residual_scale_raw// /}
  for kl_weight_raw in "${kl_weights[@]}"; do
    kl_weight=${kl_weight_raw// /}
    name="residual_$(slug "$residual_scale")_kl_$(slug "$kl_weight")"
    output_path="$output_root/$name"
    log_path="$output_path/run.log"
    if [[ -e "$output_path" ]]; then
      echo "RESPONSE_AB_CASE_REFUSED,name=$name,path=$output_path,reason=already_exists"
      n_failed=$((n_failed + 1))
      continue
    fi
    mkdir -p "$output_path"
    {
      printf 'name=%s\n' "$name"
      printf 'residual_scale=%s\n' "$residual_scale"
      printf 'kl_weight=%s\n' "$kl_weight"
      printf 'warm_omega=%s\n' "$warm_omega"
      printf 'probe_omega=%s\n' "$probe_omega"
      printf 'warm_iterations=%s\n' "$warm_iterations"
      printf 'seed=%s\n' "$seed"
    } >"$output_path/case.env"

    echo "RESPONSE_AB_CASE_START,name=$name,path=$output_path,residual_scale=$residual_scale,kl_weight=$kl_weight" | tee "$log_path"
    started=$SECONDS
    CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 \
      "$jaqmc" molecule lit \
      --yml "$config" \
      "workflow.seed=$seed" \
      "workflow.save_path=$output_path" \
      "workflow.restore_path=$ground_checkpoint" \
      "lit.nqs_checkpoint_path=$ground_checkpoint" \
      "lit.nqs_ground_energy=$ground_energy" \
      "lit.nqs_source_pool_dir=$source_pool_dir" \
      "${diagnostic_override_args[@]}" \
      lit.nqs_reuse_source_pool=true \
      lit.nqs_save_source_pool=false \
      lit.nqs_direct_psi_train=false \
      "lit.nqs_source_aligned_residual_scale=$residual_scale" \
      "lit.nqs_reverse_kl_weight=$kl_weight" \
      "lit.nqs_warm_start_omega=$warm_omega" \
      "lit.nqs_warm_start_iterations=$warm_iterations" \
      "lit.nqs_selection_interval=$selection_interval" \
      "lit.nqs_log_interval=$log_interval" \
      "lit.omega_values=[$probe_omega]" \
      lit.nqs_continuation_iterations=1 \
      lit.nqs_iterations=1 \
      2>&1 | tee -a "$log_path"
    status=${PIPESTATUS[0]}
    elapsed=$((SECONDS - started))
    echo "RESPONSE_AB_CASE_COMPLETE,name=$name,status=$status,elapsed_seconds=$elapsed" | tee -a "$log_path"
    if [[ $status -eq 0 ]]; then
      n_success=$((n_success + 1))
    else
      n_failed=$((n_failed + 1))
    fi
  done
done

summary_status=1
if "$python" "$repo_root/scripts/summarize_response_optimizer_ab.py" \
  "$output_root" >"$output_root/summary.json"; then
  summary_status=0
fi

echo "RESPONSE_AB_COMPLETE,n_success=$n_success,n_failed=$n_failed,summary_status=$summary_status,summary=$output_root/summary.json"
if [[ $n_failed -ne 0 || $summary_status -ne 0 ]]; then
  exit 1
fi
