#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/opt/tiger/jaqmc}
PY="$BASE/.venv-gpu/bin/python"
RUN_TAG=${RUN_TAG:-20260619_pv5z_krylov_heads24}

WARM_SAMPLES=${WARM_SAMPLES:-16384}
H_WARM_EPOCHS=${H_WARM_EPOCHS:-120}
HE_WARM_EPOCHS=${HE_WARM_EPOCHS:-160}
WARM_BATCH_SIZE=${WARM_BATCH_SIZE:-1024}
PRODUCTION_SAMPLER=${PRODUCTION_SAMPLER:-bright-influence}
PRODUCTION_LEVERAGE_CANDIDATE_FACTOR=${PRODUCTION_LEVERAGE_CANDIDATE_FACTOR:-2}
PRODUCTION_LEVERAGE_MAX_CANDIDATES=${PRODUCTION_LEVERAGE_MAX_CANDIDATES:-32768}
PRODUCTION_LEVERAGE_SOURCE_WEIGHT=${PRODUCTION_LEVERAGE_SOURCE_WEIGHT:-1.0}
KRYLOV_SVD_RTOL=${KRYLOV_SVD_RTOL:-1e-4}
KRYLOV_SVD_ATOL=${KRYLOV_SVD_ATOL:-1e-14}
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
CAS_DRESSING_VISIBILITY_WEIGHT=${CAS_DRESSING_VISIBILITY_WEIGHT:-0.001}
CAS_DRESSING_REGULARIZER_WEIGHT=${CAS_DRESSING_REGULARIZER_WEIGHT:-0.0001}
CAS_DRESSING_GRAD_LENGTH=${CAS_DRESSING_GRAD_LENGTH:-1.0}
FINAL_SAMPLES=${FINAL_SAMPLES:-131072}
FINAL_REPLICAS=${FINAL_REPLICAS:-4}
FINAL_BOOTSTRAP=${FINAL_BOOTSTRAP:-80}
MATRIX_BATCH_SIZE=${MATRIX_BATCH_SIZE:-1024}
CAS_BASIS=${CAS_BASIS:-aug-cc-pv5z}
CAS_TARGETS=${CAS_TARGETS:-24}
RESPONSE_HEADS=${RESPONSE_HEADS:-$CAS_TARGETS}
CAS_ROOTS=${CAS_ROOTS:-$((CAS_TARGETS + 1))}
H_CAS_NCAS=${H_CAS_NCAS:-60}
HE_CAS_NCAS=${HE_CAS_NCAS:-20}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export TMPDIR=${TMPDIR:-/tmp}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

mkdir -p "$BASE/runs/hydrogen_ferminet_formal"
mkdir -p "$BASE/runs/helium_ferminet_formal"
mkdir -p "$BASE/runs/official_20260619_logs"

echo "official_h_krylov_cas_start $(date -Is)"
"$PY" "$BASE/scripts/ferminet_bfnksr_response.py" \
  --checkpoint "$BASE/runs/hydrogen_ferminet_formal/train_ckpt_000799.npz" \
  --ground-energy -0.4999999 \
  --output "$BASE/runs/hydrogen_ferminet_formal/h_official_krylovcas_${RUN_TAG}.npz" \
  --seed 181 \
  --training-flow cas-dressed-teacher \
  --n-heads "$RESPONSE_HEADS" \
  --warm-start casscf \
  --warm-start-heads "$CAS_TARGETS" \
  --warm-start-samples "$WARM_SAMPLES" \
  --warm-start-epochs "$H_WARM_EPOCHS" \
  --warm-start-batch-size "$WARM_BATCH_SIZE" \
  --warm-start-learning-rate 0.001 \
  --production-sampler "$PRODUCTION_SAMPLER" \
  --production-leverage-candidate-factor "$PRODUCTION_LEVERAGE_CANDIDATE_FACTOR" \
  --production-leverage-max-candidates "$PRODUCTION_LEVERAGE_MAX_CANDIDATES" \
  --production-leverage-source-weight "$PRODUCTION_LEVERAGE_SOURCE_WEIGHT" \
  --warm-start-basis "$CAS_BASIS" \
  --warm-start-n-roots "$CAS_ROOTS" \
  --krylov-teacher-svd-rtol "$KRYLOV_SVD_RTOL" \
  --krylov-teacher-svd-atol "$KRYLOV_SVD_ATOL" \
  --fixed-cas-ncas "$H_CAS_NCAS" \
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
  --cas-dressing-visibility-weight "$CAS_DRESSING_VISIBILITY_WEIGHT" \
  --cas-dressing-regularizer-weight "$CAS_DRESSING_REGULARIZER_WEIGHT" \
  --cas-dressing-gradient-regularizer-length "$CAS_DRESSING_GRAD_LENGTH" \
  --hidden 20 \
  --hidden-double 4 \
  --layers 2 \
  --determinants-per-head 1 \
  --response-spatial-parity odd \
  --final-sampling cas-dressed-teacher-production-qmc-resampling \
  --final-samples "$FINAL_SAMPLES" \
  --final-sobol-replicas "$FINAL_REPLICAS" \
  --final-bootstrap-replicates "$FINAL_BOOTSTRAP" \
  --matrix-batch-size "$MATRIX_BATCH_SIZE" \
  --envelope-decay 0.32 \
  --initial-decay-min 0.5 \
  --initial-decay-max 0.5 \
  --eta 0.004 \
  --omega-min 0.25 \
  --omega-max 0.65 \
  --grid-size 4001 \
  --overlap-cutoff 1e-8 \
  --log-every 10
echo "official_h_krylov_cas_done $(date -Is)"

echo "official_he_krylov_cas_start $(date -Is)"
"$PY" "$BASE/scripts/ferminet_bfnksr_response.py" \
  --checkpoint "$BASE/runs/helium_ferminet_formal/train_ckpt_000399.npz" \
  --ground-energy -2.9038051867 \
  --output "$BASE/runs/helium_ferminet_formal/he_official_krylovcas_${RUN_TAG}.npz" \
  --seed 101 \
  --training-flow cas-dressed-teacher \
  --n-heads "$RESPONSE_HEADS" \
  --warm-start casscf \
  --warm-start-heads "$CAS_TARGETS" \
  --warm-start-samples "$WARM_SAMPLES" \
  --warm-start-epochs "$HE_WARM_EPOCHS" \
  --warm-start-batch-size "$WARM_BATCH_SIZE" \
  --warm-start-learning-rate 0.001 \
  --production-sampler "$PRODUCTION_SAMPLER" \
  --production-leverage-candidate-factor "$PRODUCTION_LEVERAGE_CANDIDATE_FACTOR" \
  --production-leverage-max-candidates "$PRODUCTION_LEVERAGE_MAX_CANDIDATES" \
  --production-leverage-source-weight "$PRODUCTION_LEVERAGE_SOURCE_WEIGHT" \
  --warm-start-basis "$CAS_BASIS" \
  --warm-start-n-roots "$CAS_ROOTS" \
  --krylov-teacher-svd-rtol "$KRYLOV_SVD_RTOL" \
  --krylov-teacher-svd-atol "$KRYLOV_SVD_ATOL" \
  --fixed-cas-ncas "$HE_CAS_NCAS" \
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
  --cas-dressing-visibility-weight "$CAS_DRESSING_VISIBILITY_WEIGHT" \
  --cas-dressing-regularizer-weight "$CAS_DRESSING_REGULARIZER_WEIGHT" \
  --cas-dressing-gradient-regularizer-length "$CAS_DRESSING_GRAD_LENGTH" \
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
echo "official_he_krylov_cas_done $(date -Is)"
