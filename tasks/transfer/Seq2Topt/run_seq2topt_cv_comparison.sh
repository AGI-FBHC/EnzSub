#!/usr/bin/env bash
# Five-fold Seq2Topt comparison: original ESM-2 8M versus routed EnzSub CPT.
# The 291-sample published holdout remains untouched by fold construction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/artifacts/transfer/seq2topt/data/Topt/train_os.csv}"
TEST_CSV="${TEST_CSV:-${REPO_ROOT}/artifacts/transfer/seq2topt/data/Topt/test.csv}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/artifacts/transfer/seq2topt/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/transfer/seq2topt/results/cv_comparison}"
ESM8_CACHE="${ESM8_CACHE:-${CACHE_ROOT}/esm2_t6_8M_UR50D/published_base}"
BACKBONES="${BACKBONES:-esm2_t33_650M esm2_t36_3B protbert_bfd}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
USE_AMP="${USE_AMP:-1}"
FORCE_CACHE="${FORCE_CACHE:-0}"

mkdir -p "${OUTPUT_ROOT}"

cache_args=()
if [[ "${FORCE_CACHE}" == "1" ]]; then
  cache_args+=(--overwrite)
fi
if [[ "${USE_AMP}" == "1" ]]; then
  cache_args+=(--amp)
fi

echo "Preparing original Seq2Topt ESM-2 t6 8M cache"
"${PYTHON}" "${CODE_DIR}/precompute_esm8_features.py" \
  --inputs "${TRAIN_CSV}" "${TEST_CSV}" \
  --cache-dir "${ESM8_CACHE}" \
  --batch-size 8 \
  --device "${DEVICE}" \
  "${cache_args[@]}"

run_cv() {
  local name="$1"
  local cache_dir="$2"
  local output_dir="${OUTPUT_ROOT}/${name}/seed_${SEED}"
  mkdir -p "${output_dir}"
  [[ -f "${cache_dir}/manifest.json" ]] || {
    echo "Missing cache manifest: ${cache_dir}/manifest.json" >&2
    exit 2
  }
  local args=(
    "${PYTHON}" "${CODE_DIR}/run_cv_enzsub.py"
    --train-csv "${TRAIN_CSV}"
    --test-csv "${TEST_CSV}"
    --cache-dir "${cache_dir}"
    --output-dir "${output_dir}"
    --folds "${FOLDS}"
    --seed "${SEED}"
    --epochs "${EPOCHS}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --effective-batch-size "${EFFECTIVE_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
  )
  if [[ "${USE_AMP}" == "1" ]]; then
    args+=(--amp)
  fi
  echo "Running ${name}"
  "${args[@]}" 2>&1 | tee "${output_dir}.console.log"
}

run_cv "published_esm2_t6_8M" "${ESM8_CACHE}"
for encoder in ${BACKBONES}; do
  run_cv "${encoder}_cpt" "${CACHE_ROOT}/${encoder}/cpt"
done

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/seed_*/metrics.json")):
    result = json.loads(path.read_text())
    rows.append({
        "method": path.parent.parent.name,
        "encoder_type": result["encoder_type"],
        "representation": result["representation"],
        "oof_rmse": result["oof_metrics"]["rmse"],
        "oof_mae": result["oof_metrics"]["mae"],
        "oof_r2": result["oof_metrics"]["r2"],
        "test_ensemble_rmse": result["test_ensemble_metrics"]["rmse"],
        "test_ensemble_mae": result["test_ensemble_metrics"]["mae"],
        "test_ensemble_r2": result["test_ensemble_metrics"]["r2"],
        "source": str(path.relative_to(root)),
    })
fields = list(rows[0]) if rows else ["method", "encoder_type", "representation"]
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {root / 'summary.csv'} with {len(rows)} rows")
PY
