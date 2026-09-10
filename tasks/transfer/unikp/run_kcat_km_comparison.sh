#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
TASKS="${TASKS:-kcat km kcat_km}"
ENZSUB_CODE_DIR="${REPO_ROOT}/src"
PROT_T5_PATH="${PROT_T5_PATH:-Rostlab/prot_t5_xl_uniref50}"
PROT_T5_CACHE_DIR="${PROT_T5_CACHE_DIR:-}"
PROT_T5_LOCAL_FILES_ONLY="${PROT_T5_LOCAL_FILES_ONLY:-0}"
ESM2_650M_CHECKPOINT="${ESM2_650M_CHECKPOINT:-${REPO_ROOT}/checkpoints/sub/esm2_650M_cpt_sub.pt}"
PROTBERT_CHECKPOINT="${PROTBERT_CHECKPOINT:-${REPO_ROOT}/checkpoints/sub/protbert_cpt_sub.pt}"
ESM2_3B_CHECKPOINT="${ESM2_3B_CHECKPOINT:-${REPO_ROOT}/checkpoints/sub/esm2_3B_cpt_sub.pt}"
LOG_DIR="${REPO_ROOT}/artifacts/transfer/unikp/logs"
SHARED_CACHE_DIR="${REPO_ROOT}/artifacts/transfer/unikp/cache"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}" "${SHARED_CACHE_DIR}" "${REPO_ROOT}/artifacts/transfer/unikp/results"

PROT_T5_ARGS=(--prot-t5-path "${PROT_T5_PATH}")
if [[ -n "${PROT_T5_CACHE_DIR}" ]]; then
  PROT_T5_ARGS+=(--prot-t5-cache-dir "${PROT_T5_CACHE_DIR}")
fi
if [[ "${PROT_T5_LOCAL_FILES_ONLY}" == "1" ]]; then
  PROT_T5_ARGS+=(--prot-t5-local-files-only)
fi

run_comparison() {
  local task="$1"
  local run_name="$2"
  local encoder_type="$3"
  local model_mode="$4"
  local checkpoint="$5"
  local lora_rank="$6"
  local embedding_batch_size="$7"
  local feature_cache="${8:-}"
  local prott5_feature_cache="${9:-}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[error] Checkpoint not found: ${checkpoint}" >&2
    return 1
  fi

  echo "[run] Starting ${task} comparison: ${run_name}"
  local feature_cache_args=()
  if [[ -n "${feature_cache}" ]]; then
    if [[ ! -f "${feature_cache}" ]]; then
      echo "[cache] EnzSub feature cache not found; it will be built: ${feature_cache}"
      feature_cache=""
    fi
    if [[ -n "${feature_cache}" ]]; then
      feature_cache_args+=(--enzsub-feature-cache "${feature_cache}")
    fi
  fi
  if [[ -n "${prott5_feature_cache}" ]]; then
    if [[ ! -f "${prott5_feature_cache}" ]]; then
      echo "[cache] ProtT5 feature cache not found; it will be built: ${prott5_feature_cache}"
      prott5_feature_cache=""
    fi
    if [[ -n "${prott5_feature_cache}" ]]; then
      feature_cache_args+=(--prott5-feature-cache "${prott5_feature_cache}")
    fi
  fi
  "${PYTHON_BIN}" compare_prott5_enzsub_unikp.py \
    --task "${task}" \
    "${PROT_T5_ARGS[@]}" \
    --encoder-type "${encoder_type}" \
    --model-mode "${model_mode}" \
    --checkpoint "${checkpoint}" \
    --device "${DEVICE}" \
    --lora-rank "${lora_rank}" \
    --enzsub-batch-size "${embedding_batch_size}" \
    --enzsub-code-dir "${ENZSUB_CODE_DIR}" \
    --smiles-cache "${SHARED_CACHE_DIR}/${task}_unikp_smiles.pkl" \
    --prott5-cache "${SHARED_CACHE_DIR}/${task}_prott5_seq.pkl" \
    --enzsub-cache "${SHARED_CACHE_DIR}/${task}_${run_name}_seq.pkl" \
    --output-dir "${REPO_ROOT}/artifacts/transfer/unikp/results/${task}/${run_name}" \
    "${feature_cache_args[@]}" \
    --holdout-runs 5 \
    --kfold-runs 1 \
    --n-jobs 16 \
    --save-features 2>&1 | tee "${LOG_DIR}/${task}_${run_name}_${STAMP}.log"
}

for task in ${TASKS}; do
  run_comparison \
    "${task}" \
    "esm2_650m_cpt_sub_epoch15" \
    "esm2_650m" \
    "cpt_sub" \
    "${ESM2_650M_CHECKPOINT}" \
    16 \
    8 \
    "embedding_comparison_outputs_650m/${task}/${task}_features_enzsub_cpt_sub.pkl" \
    "embedding_comparison_outputs_650m/${task}/${task}_features_unikp_prott5.pkl"

  run_comparison \
    "${task}" \
    "protbert_bfd_cpt_sub_epoch15" \
    "protbert_bfd" \
    "cpt_sub" \
    "${PROTBERT_CHECKPOINT}" \
    4 \
    8 \
    "embedding_comparison_outputs_protbert/${task}/protbert_bfd_cpt_sub/${task}_features_enzsub_cpt_sub.pkl" \
    "embedding_comparison_outputs_protbert/${task}/protbert_bfd_cpt_sub/${task}_features_unikp_prott5.pkl"

  run_comparison \
    "${task}" \
    "esm2_3b_cpt_sub_epoch5" \
    "esm2_t36_3B" \
    "cpt_sub" \
    "${ESM2_3B_CHECKPOINT}" \
    16 \
    4 \
    "" \
    "embedding_comparison_outputs/${task}/${task}_features_unikp_prott5.pkl"
done

echo "[done] All comparisons finished"
