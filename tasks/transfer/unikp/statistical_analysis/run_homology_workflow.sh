#!/usr/bin/env bash
set -euo pipefail

# Additive workflow. It preserves embedding_comparison_outputs/ and creates
# all new artifacts below statistical_analysis/.

ROOT="${ROOT:-$(pwd)}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${ROOT}/datasets}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${ROOT}/statistical_analysis}"
PREP_ROOT="${ANALYSIS_ROOT}/homology_inputs"
CLUSTER_ROOT="${ANALYSIS_ROOT}/homology_40id/clusters"
PROFILE_INPUT_ROOT="${ANALYSIS_ROOT}/random_split_inputs"
PROFILE_ROOT="${ANALYSIS_ROOT}/homology_profile/search"
PROFILE_SUMMARY_ROOT="${ANALYSIS_ROOT}/homology_profile/max_identity"
MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
RUN_PROFILES="${RUN_PROFILES:-1}"
RUN_CLUSTER_EVAL="${RUN_CLUSTER_EVAL:-1}"

mkdir -p "${ANALYSIS_ROOT}"

"${PYTHON}" "${ROOT}/statistical_analysis/analyze_random_stats.py" \
  --root "${ROOT}/embedding_comparison_outputs" --data-dir "${DATA_DIR}" \
  --output-dir "${ANALYSIS_ROOT}/random_split" \
  --bootstrap-reps "${BOOTSTRAP_REPS:-2000}" --permutation-reps "${PERMUTATION_REPS:-9999}" \
  --seed "${SEED:-20260905}"

"${PYTHON}" "${ROOT}/statistical_analysis/prepare_sequences.py" \
  --unikp-root "${ROOT}" --data-dir "${DATA_DIR}" --output-dir "${PREP_ROOT}"

"${PYTHON}" "${ROOT}/statistical_analysis/make_random_split_fastas.py" \
  --unikp-root "${ROOT}" --data-dir "${DATA_DIR}" \
  --prediction-root "${ROOT}/embedding_comparison_outputs" \
  --output-dir "${PROFILE_INPUT_ROOT}"

MMSEQS_BIN="${MMSEQS_BIN}" INPUT_ROOT="${PREP_ROOT}" \
  OUTPUT_ROOT="${CLUSTER_ROOT}/mmseqs" TMP_ROOT="${CLUSTER_ROOT}/tmp" \
  THREADS="${MMSEQS_THREADS:-32}" \
  bash "${ROOT}/statistical_analysis/run_mmseqs_homology.sh"

for task in kcat km kcat_km; do
  cluster_tsv="${CLUSTER_ROOT}/mmseqs/${task}/cluster_40id_cov80_cluster.tsv"
  [[ -s "${cluster_tsv}" ]] || { echo "[error] missing ${cluster_tsv}" >&2; exit 2; }
  "${PYTHON}" "${ROOT}/statistical_analysis/make_cluster_folds.py" \
    --cluster-tsv "${cluster_tsv}" \
    --sequence-manifest "${PREP_ROOT}/${task}/sequence_manifest.csv" \
    --sample-manifest "${PREP_ROOT}/${task}/sample_manifest.csv" \
    --output-dir "${CLUSTER_ROOT}/${task}" --n-folds "${N_FOLDS:-5}" --seed 42
done

if [[ "${RUN_PROFILES}" == "1" ]]; then
  MMSEQS_BIN="${MMSEQS_BIN}" INPUT_ROOT="${PROFILE_INPUT_ROOT}" \
    OUTPUT_ROOT="${PROFILE_ROOT}" TMP_ROOT="${ANALYSIS_ROOT}/homology_profile/tmp" \
    THREADS="${MMSEQS_THREADS:-32}" \
    bash "${ROOT}/statistical_analysis/run_mmseqs_profiles.sh"
  "${PYTHON}" "${ROOT}/statistical_analysis/summarize_mmseqs_profiles.py" \
    --profile-root "${PROFILE_ROOT}" --output-root "${PROFILE_SUMMARY_ROOT}"
fi

if [[ "${RUN_CLUSTER_EVAL}" == "1" ]]; then
  ROOT="${ROOT}" DATA_DIR="${DATA_DIR}" FOLD_ROOT="${CLUSTER_ROOT}" \
    FEATURE_ROOT="${ROOT}/embedding_comparison_outputs" \
    OUTPUT_ROOT="${ANALYSIS_ROOT}/homology_40id/cluster_eval" \
    N_JOBS="${N_JOBS:-16}" bash "${ROOT}/statistical_analysis/run_cluster_evaluation.sh"
fi

echo "[done] UniKP statistical and homology workflow"
