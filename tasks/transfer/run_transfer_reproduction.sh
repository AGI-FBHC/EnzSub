#!/usr/bin/env bash
# Reproduce the three representation-transfer experiments with their formal
# shared-seed protocols. Precomputed embeddings are treated as immutable input.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-6}"
RUN_LOG_DIR="${REPO_ROOT}/artifacts/transfer/reproduction_logs"

mkdir -p "${RUN_LOG_DIR}"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONDONTWRITEBYTECODE=1

echo "[1/3] UniKP controlled representation comparisons"
PYTHON_BIN="${PYTHON_BIN}" DEVICE="cuda:0" \
  bash tasks/transfer/unikp/run_kcat_km_comparison.sh \
  2>&1 | tee "${RUN_LOG_DIR}/unikp.log"

echo "[2/3] Seq2Topt five-seed, five-fold comparisons"
for seed in 0 1 2 3 4; do
  PYTHON="${PYTHON_BIN}" DEVICE="cuda:0" SEED="${seed}" \
    bash tasks/transfer/Seq2Topt/run_seq2topt_cv_comparison.sh \
    2>&1 | tee "${RUN_LOG_DIR}/seq2topt_seed_${seed}.log"
done

echo "[3/3] ReactZyme four-representation, ten-seed comparisons"
PATH="$(dirname "${PYTHON_BIN}"):${PATH}" "${PYTHON_BIN}" \
  tasks/transfer/ReactZyme/enzsub_reactzyme/run_pipeline.py \
  --config tasks/transfer/ReactZyme/enzsub_reactzyme/config.reproduction.json \
  --stage audit \
  2>&1 | tee "${RUN_LOG_DIR}/reactzyme_audit.log"

PATH="$(dirname "${PYTHON_BIN}"):${PATH}" "${PYTHON_BIN}" \
  tasks/transfer/ReactZyme/enzsub_reactzyme/run_pipeline.py \
  --config tasks/transfer/ReactZyme/enzsub_reactzyme/config.reproduction.json \
  --stage train --resume \
  2>&1 | tee "${RUN_LOG_DIR}/reactzyme_train.log"

echo "All transfer reproductions finished."
