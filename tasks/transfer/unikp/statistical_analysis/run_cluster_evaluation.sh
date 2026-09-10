#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
ROOT="${ROOT:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${ROOT}/datasets}"
FOLD_ROOT="${FOLD_ROOT:?set FOLD_ROOT to the cluster fold directory}"
FEATURE_ROOT="${FEATURE_ROOT:-${ROOT}/embedding_comparison_outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/statistical_analysis/homology_40id/cluster_eval}"
N_JOBS="${N_JOBS:-16}"

BACKBONES=(
  esm2_650m_cpt_sub_epoch15
  protbert_bfd_cpt_sub_epoch15
  esm2_3b_cpt_sub_epoch5
)
TASKS=(kcat km kcat_km)

for task in "${TASKS[@]}"; do
  fold_manifest="${FOLD_ROOT}/${task}/sample_folds.csv"
  [[ -s "${fold_manifest}" ]] || { echo "[error] missing ${fold_manifest}" >&2; exit 2; }
  for backbone in "${BACKBONES[@]}"; do
    base="${FEATURE_ROOT}/${task}/${backbone}"
    prott5="${base}/${task}_features_unikp_prott5.pkl"
    enzsub="${base}/${task}_features_enzsub_cpt_sub.pkl"
    [[ -s "${prott5}" && -s "${enzsub}" ]] || { echo "[skip] ${task}/${backbone}: feature cache missing"; continue; }
    out="${OUTPUT_ROOT}/${task}/${backbone}"
    if [[ -s "${out}/${task}_cluster_metrics.csv" && -s "${out}/${task}_cluster_predictions.csv" ]]; then
      echo "[exists] ${task}/${backbone}: cluster results already complete"
      continue
    fi
    echo "[cluster-eval] task=${task} backbone=${backbone}"
    "${PYTHON}" "${ROOT}/statistical_analysis/run_cluster_unikp.py" \
      --unikp-root "${ROOT}" --data-dir "${DATA_DIR}" --task "${task}" \
      --fold-manifest "${fold_manifest}" --prott5-feature-cache "${prott5}" \
      --enzsub-feature-cache "${enzsub}" --output-dir "${out}" --n-jobs "${N_JOBS}"
  done
done
