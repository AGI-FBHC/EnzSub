#!/usr/bin/env bash
set -euo pipefail

cd artifacts/downstream/ec/hcft

PYTHON=${PYTHON:-python}

LOG_DIR=artifacts/downstream/ec/hcft/run_logs
mkdir -p "${LOG_DIR}"

run_one () {
  local cfg="$1"
  local name
  name=$(basename "${cfg}" .yaml)

  echo "============================================================"
  echo "[START] ${name}"
  echo "Config: ${cfg}"
  echo "Time: $(date)"
  echo "============================================================"

  "${PYTHON}" hcft_eval.py --config "${cfg}" 2>&1 | tee "${LOG_DIR}/${name}.log"

  echo "============================================================"
  echo "[DONE] ${name}"
  echo "Time: $(date)"
  echo "============================================================"
  echo
}

run_one artifacts/downstream/ec/hcft/hcft_06B_general.yaml
run_one artifacts/downstream/ec/hcft/hcft_06B_strict.yaml
run_one artifacts/downstream/ec/hcft/hcft_06B_hard_ec3.yaml

run_one artifacts/downstream/ec/hcft/hcft_3B_general.yaml
run_one artifacts/downstream/ec/hcft/hcft_3B_strict.yaml
run_one artifacts/downstream/ec/hcft/hcft_3B_hard_ec3.yaml

run_one artifacts/downstream/ec/hcft/hcft_protbert_general.yaml
run_one artifacts/downstream/ec/hcft/hcft_protbert_strict.yaml
run_one artifacts/downstream/ec/hcft/hcft_protbert_hard_ec3.yaml