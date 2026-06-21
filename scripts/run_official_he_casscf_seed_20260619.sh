#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/opt/tiger/jaqmc}
PY="$BASE/.venv-gpu/bin/python"
RUN_TAG=${RUN_TAG:-20260619_fixedcas_he}

WARM_SAMPLES=${WARM_SAMPLES:-16384}
HE_WARM_EPOCHS=${HE_WARM_EPOCHS:-160}
WARM_BATCH_SIZE=${WARM_BATCH_SIZE:-1024}
PRODUCTION_SAMPLER=${PRODUCTION_SAMPLER:-bright-influence}
PRODUCTION_LEVERAGE_CANDIDATE_FACTOR=${PRODUCTION_LEVERAGE_CANDIDATE_FACTOR:-2}
PRODUCTION_LEVERAGE_MAX_CANDIDATES=${PRODUCTION_LEVERAGE_MAX_CANDIDATES:-32768}
PRODUCTION_LEVERAGE_SOURCE_WEIGHT=${PRODUCTION_LEVERAGE_SOURCE_WEIGHT:-1.0}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-20}
FINETUNE_BATCH_SIZE=${FINETUNE_BATCH_SIZE:-1024}
FINETUNE_LEARNING_RATE=${FINETUNE_LEARNING_RATE:-0.0001}
FINETUNE_ROOTS=${FINETUNE_ROOTS:-3}
FINETUNE_CONDITION_WEIGHT=${FINETUNE_CONDITION_WEIGHT:-0.01}
FINETUNE_OVERLAP_WEIGHT=${FINETUNE_OVERLAP_WEIGHT:-0.01}
FINETUNE_MAX_CONDITION=${FINETUNE_MAX_CONDITION:-1000000}
FINETUNE_ROOT_FLOOR=${FINETUNE_ROOT_FLOOR:-0.0}
FINETUNE_VALIDATION_FRACTION=${FINETUNE_VALIDATION_FRACTION:-0.25}
FINETUNE_BRIGHT_THRESHOLD=${FINETUNE_BRIGHT_THRESHOLD:-0.05}
FINETUNE_VALIDATION_BLOCKS=${FINETUNE_VALIDATION_BLOCKS:-4}
FINETUNE_ACCEPTANCE_SIGMA=${FINETUNE_ACCEPTANCE_SIGMA:-1.0}
FINAL_SAMPLES=${FINAL_SAMPLES:-131072}
FINAL_REPLICAS=${FINAL_REPLICAS:-4}
FINAL_BOOTSTRAP=${FINAL_BOOTSTRAP:-80}
MATRIX_BATCH_SIZE=${MATRIX_BATCH_SIZE:-1024}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export TMPDIR=${TMPDIR:-/tmp}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32

mkdir -p "$BASE/runs/helium_ferminet_formal"
mkdir -p "$BASE/runs/official_20260619_logs"

echo "official_he_fixed_cas_start $(date -Is)"
"$PY" "$BASE/scripts/ferminet_bfnksr_response.py" \
  --checkpoint "$BASE/runs/helium_ferminet_formal/train_ckpt_000399.npz" \
  --ground-energy -2.9038051867 \
  --output "$BASE/runs/helium_ferminet_formal/he_official_fixedcas_${RUN_TAG}.npz" \
  --seed 101 \
  --training-flow cas-dressed-teacher \
  --n-heads 8 \
  --warm-start casscf \
  --warm-start-heads 8 \
  --warm-start-samples "$WARM_SAMPLES" \
  --warm-start-epochs "$HE_WARM_EPOCHS" \
  --warm-start-batch-size "$WARM_BATCH_SIZE" \
  --warm-start-learning-rate 0.001 \
  --production-sampler "$PRODUCTION_SAMPLER" \
  --production-leverage-candidate-factor "$PRODUCTION_LEVERAGE_CANDIDATE_FACTOR" \
  --production-leverage-max-candidates "$PRODUCTION_LEVERAGE_MAX_CANDIDATES" \
  --production-leverage-source-weight "$PRODUCTION_LEVERAGE_SOURCE_WEIGHT" \
  --warm-start-basis aug-cc-pvdz \
  --warm-start-n-roots 8 \
  --fixed-cas-ncas 6 \
  --no-fixed-cas-state-average \
  --fixed-cas-gradient-weight 0.1 \
  --fixed-cas-fd-step 0.001 \
  --fixed-cas-finetune-epochs "$FINETUNE_EPOCHS" \
  --fixed-cas-finetune-batch-size "$FINETUNE_BATCH_SIZE" \
  --fixed-cas-finetune-learning-rate "$FINETUNE_LEARNING_RATE" \
  --fixed-cas-finetune-roots "$FINETUNE_ROOTS" \
  --fixed-cas-finetune-condition-weight "$FINETUNE_CONDITION_WEIGHT" \
  --fixed-cas-finetune-overlap-weight "$FINETUNE_OVERLAP_WEIGHT" \
  --fixed-cas-finetune-max-condition "$FINETUNE_MAX_CONDITION" \
  --fixed-cas-finetune-root-floor "$FINETUNE_ROOT_FLOOR" \
  --fixed-cas-finetune-validation-fraction "$FINETUNE_VALIDATION_FRACTION" \
  --fixed-cas-finetune-bright-threshold "$FINETUNE_BRIGHT_THRESHOLD" \
  --fixed-cas-finetune-validation-blocks "$FINETUNE_VALIDATION_BLOCKS" \
  --fixed-cas-finetune-acceptance-sigma "$FINETUNE_ACCEPTANCE_SIGMA" \
  --hidden 20 \
  --hidden-double 6 \
  --layers 2 \
  --determinants-per-head 1 \
  --response-spatial-parity odd \
  --final-sampling cas-dressed-teacher-production-qmc-resampling \
  --final-samples "$FINAL_SAMPLES" \
  --final-sobol-replicas "$FINAL_REPLICAS" \
  --final-bootstrap-replicates "$FINAL_BOOTSTRAP" \
  --matrix-batch-size "$MATRIX_BATCH_SIZE" \
  --envelope-decay 0.8 \
  --initial-decay-min 0.20 \
  --initial-decay-max 2.00 \
  --eta 0.005 \
  --omega-min 0.4 \
  --omega-max 2.0 \
  --grid-size 4001 \
  --overlap-cutoff 1e-8 \
  --log-every 10
echo "official_he_fixed_cas_done $(date -Is)"
