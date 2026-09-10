#!/usr/bin/env bash
# Run formal paired HCFT statistics across:
#   3 backbones × 3 HCFT settings
#
# Comparisons:
#   1. Base -> CPT
#   2. Base -> Base-SUB
#   3. CPT -> CPT-SUB
#   4. Base -> CPT-SUB
#
# Statistics are computed by analyze_hcft_paired_statistics.py at the
# unique-anchor level, with bootstrap confidence intervals, Wilcoxon tests,
# sign-flip permutation tests, and Holm correction across all experiments.
#
# Usage:
#   bash run_hcft_formal_paired_statistics.sh
#
# Optional overrides:
#   PYTHON_BIN=/path/to/python \
#   STAT_SCRIPT=/path/to/analyze_hcft_paired_statistics.py \
#   N_BOOT=5000 N_PERM=9999 \
#   INCLUDE_CPT_ONLY=1 INCLUDE_FINAL_MODEL=1 \
#   bash run_hcft_formal_paired_statistics.sh

set -Eeuo pipefail

# ==============================================================================
# Limit native numerical libraries to one thread.
# This avoids OMP "Resource temporarily unavailable" errors.
# ==============================================================================
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ==============================================================================
# Paths and statistical settings
# ==============================================================================
HCFT_ROOT="${HCFT_ROOT:-artifacts/downstream/ec/hcft}"
EXP_ROOT="${EXP_ROOT:-${HCFT_ROOT}/experiments}"
STAT_SCRIPT="${STAT_SCRIPT:-${HCFT_ROOT}/analyze_hcft_paired_statistics.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# With 9 experiments and up to 4 comparisons per experiment, these settings
# provide stable CIs and a minimum Monte Carlo p value of 1/(9999+1)=1e-4.
N_BOOT="${N_BOOT:-5000}"
N_PERM="${N_PERM:-9999}"
CONFIDENCE="${CONFIDENCE:-0.95}"
RANDOM_SEED="${RANDOM_SEED:-20260717}"

# Set to 0 to omit a comparison.
INCLUDE_CPT_ONLY="${INCLUDE_CPT_ONLY:-1}"     # Base -> CPT
INCLUDE_FINAL_MODEL="${INCLUDE_FINAL_MODEL:-1}" # Base -> CPT-SUB

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${HCFT_ROOT}/statistics/formal_all_backbones/${RUN_TAG}}"
LOG_FILE="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"

on_error() {
    local exit_code=$?
    echo
    echo "[ERROR] Command failed with exit code ${exit_code}."
    echo "[ERROR] See log: ${LOG_FILE}"
    exit "${exit_code}"
}
trap on_error ERR

echo "==============================================================================" | tee "${LOG_FILE}"
echo "Formal paired HCFT statistics" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Python       : ${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "Stat script  : ${STAT_SCRIPT}" | tee -a "${LOG_FILE}"
echo "Experiments  : ${EXP_ROOT}" | tee -a "${LOG_FILE}"
echo "Output       : ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "Bootstrap    : ${N_BOOT}" | tee -a "${LOG_FILE}"
echo "Permutations : ${N_PERM}" | tee -a "${LOG_FILE}"
echo "OMP threads  : ${OMP_NUM_THREADS}" | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${STAT_SCRIPT}" ]]; then
    echo "Statistics script not found: ${STAT_SCRIPT}" >&2
    exit 1
fi

# ==============================================================================
# Experiment files
# ==============================================================================
declare -A EXP_FILES=(
    [3B_general]="${EXP_ROOT}/3B_general/hcft_anchor_summary.csv"
    [3B_hard_ec3]="${EXP_ROOT}/3B_hard_ec3/hcft_anchor_summary.csv"
    [3B_strict]="${EXP_ROOT}/3B_strict/hcft_anchor_summary.csv"

    [06B_general]="${EXP_ROOT}/06B_general/hcft_anchor_summary.csv"
    [06B_hard_ec3]="${EXP_ROOT}/06B_hard_ec3/hcft_anchor_summary.csv"
    [06B_strict]="${EXP_ROOT}/06B_strict/hcft_anchor_summary.csv"

    [protbert_general]="${EXP_ROOT}/protbert_general/hcft_anchor_summary.csv"
    [protbert_hard_ec3]="${EXP_ROOT}/protbert_hard_ec3/hcft_anchor_summary.csv"
    [protbert_strict]="${EXP_ROOT}/protbert_strict/hcft_anchor_summary.csv"
)

# Expected embedding names in each backbone.
declare -A BASE_EMB=(
    [3B]="esm2_3b_base"
    [06B]="esm2_base"
    [protbert]="protbert_base"
)
declare -A CPT_EMB=(
    [3B]="esm2_3b_cpt"
    [06B]="esm2_cpt"
    [protbert]="protbert_cpt"
)
declare -A BASE_SUB_EMB=(
    [3B]="esm2_3b_base_sub"
    [06B]="esm2_base_sub"
    [protbert]="protbert_base_sub"
)
declare -A CPT_SUB_EMB=(
    [3B]="esm2_3b_cpt_sub"
    [06B]="esm2_cpt_sub"
    [protbert]="protbert_cpt_sub"
)

experiment_order=(
    3B_general
    3B_hard_ec3
    3B_strict
    06B_general
    06B_hard_ec3
    06B_strict
    protbert_general
    protbert_hard_ec3
    protbert_strict
)

backbone_of() {
    local experiment="$1"
    case "${experiment}" in
        3B_*) echo "3B" ;;
        06B_*) echo "06B" ;;
        protbert_*) echo "protbert" ;;
        *)
            echo "Unknown experiment prefix: ${experiment}" >&2
            return 1
            ;;
    esac
}

# ==============================================================================
# Preflight: verify files, columns, and expected embedding names.
# ==============================================================================
echo "[1/3] Preflight validation" | tee -a "${LOG_FILE}"

for experiment in "${experiment_order[@]}"; do
    file="${EXP_FILES[${experiment}]}"
    if [[ ! -f "${file}" ]]; then
        echo "Missing experiment file: ${file}" >&2
        exit 1
    fi
done

export EXP_ROOT

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["EXP_ROOT"])

specs = {
    "3B_general": (
        root / "3B_general/hcft_anchor_summary.csv",
        ["esm2_3b_base", "esm2_3b_cpt",
         "esm2_3b_base_sub", "esm2_3b_cpt_sub"],
    ),
    "3B_hard_ec3": (
        root / "3B_hard_ec3/hcft_anchor_summary.csv",
        ["esm2_3b_base", "esm2_3b_cpt",
         "esm2_3b_base_sub", "esm2_3b_cpt_sub"],
    ),
    "3B_strict": (
        root / "3B_strict/hcft_anchor_summary.csv",
        ["esm2_3b_base", "esm2_3b_cpt",
         "esm2_3b_base_sub", "esm2_3b_cpt_sub"],
    ),
    "06B_general": (
        root / "06B_general/hcft_anchor_summary.csv",
        ["esm2_base", "esm2_cpt", "esm2_base_sub", "esm2_cpt_sub"],
    ),
    "06B_hard_ec3": (
        root / "06B_hard_ec3/hcft_anchor_summary.csv",
        ["esm2_base", "esm2_cpt", "esm2_base_sub", "esm2_cpt_sub"],
    ),
    "06B_strict": (
        root / "06B_strict/hcft_anchor_summary.csv",
        ["esm2_base", "esm2_cpt", "esm2_base_sub", "esm2_cpt_sub"],
    ),
    "protbert_general": (
        root / "protbert_general/hcft_anchor_summary.csv",
        ["protbert_base", "protbert_cpt",
         "protbert_base_sub", "protbert_cpt_sub"],
    ),
    "protbert_hard_ec3": (
        root / "protbert_hard_ec3/hcft_anchor_summary.csv",
        ["protbert_base", "protbert_cpt",
         "protbert_base_sub", "protbert_cpt_sub"],
    ),
    "protbert_strict": (
        root / "protbert_strict/hcft_anchor_summary.csv",
        ["protbert_base", "protbert_cpt",
         "protbert_base_sub", "protbert_cpt_sub"],
    ),
}

required_columns = {
    "embedding", "anchor", "hcft_acc", "hcft_margin_mean"
}
errors = []

for name, (path, expected_embeddings) in specs.items():
    if not path.exists():
        errors.append(f"{name}: missing file {path}")
        continue

    header = pd.read_csv(path, nrows=0)
    missing_cols = required_columns - set(header.columns)
    if missing_cols:
        errors.append(
            f"{name}: missing columns {sorted(missing_cols)}"
        )
        continue

    emb = (
        pd.read_csv(path, usecols=["embedding"])["embedding"]
        .astype(str)
        .value_counts()
    )
    available = set(emb.index)
    missing_emb = [x for x in expected_embeddings if x not in available]

    print(f"\n[{name}] {path}")
    print(emb.to_string())

    if missing_emb:
        errors.append(
            f"{name}: missing embeddings {missing_emb}; "
            f"available={sorted(available)}"
        )

if errors:
    print("\nPreflight errors:")
    for error in errors:
        print(f"  - {error}")
    raise SystemExit(1)

print("\nPreflight passed: all files and embeddings are available.")
PY

# ==============================================================================
# Build one combined invocation.
# Running all experiments together lets Holm correction cover every formal
# experiment within each metric rather than correcting each directory separately.
# ==============================================================================
echo | tee -a "${LOG_FILE}"
echo "[2/3] Build formal analysis command" | tee -a "${LOG_FILE}"

cmd=(
    "${PYTHON_BIN}"
    "${STAT_SCRIPT}"
)

for experiment in "${experiment_order[@]}"; do
    cmd+=(--experiment "${experiment}:${EXP_FILES[${experiment}]}")
done

for experiment in "${experiment_order[@]}"; do
    backbone="$(backbone_of "${experiment}")"
    base="${BASE_EMB[${backbone}]}"
    cpt="${CPT_EMB[${backbone}]}"
    base_sub="${BASE_SUB_EMB[${backbone}]}"
    cpt_sub="${CPT_SUB_EMB[${backbone}]}"

    # Direct substrate-aware effect on the original representation.
    cmd+=(--pair "${experiment}@base_to_base_sub:${base}:${base_sub}")

    # Direct substrate-aware effect after CPT.
    cmd+=(--pair "${experiment}@cpt_to_cpt_sub:${cpt}:${cpt_sub}")

    # CPT-only control.
    if [[ "${INCLUDE_CPT_ONLY}" == "1" ]]; then
        cmd+=(--pair "${experiment}@base_to_cpt:${base}:${cpt}")
    fi

    # Final complete model versus the original backbone.
    if [[ "${INCLUDE_FINAL_MODEL}" == "1" ]]; then
        cmd+=(--pair "${experiment}@base_to_cpt_sub:${base}:${cpt_sub}")
    fi
done

cmd+=(
    --bootstrap "${N_BOOT}"
    --permutations "${N_PERM}"
    --confidence "${CONFIDENCE}"
    --seed "${RANDOM_SEED}"
    --output-dir "${OUTPUT_DIR}"
)

printf 'Command:\n  ' | tee -a "${LOG_FILE}"
printf '%q ' "${cmd[@]}" | tee -a "${LOG_FILE}"
printf '\n\n' | tee -a "${LOG_FILE}"

# ==============================================================================
# Run
# ==============================================================================
echo "[3/3] Run paired statistics" | tee -a "${LOG_FILE}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Completed" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Main statistics : ${OUTPUT_DIR}/paired_statistics.csv" | tee -a "${LOG_FILE}"
echo "Anchor deltas   : ${OUTPUT_DIR}/paired_anchor_deltas.csv" | tee -a "${LOG_FILE}"
echo "Diagnostics     : ${OUTPUT_DIR}/pairing_diagnostics.csv" | tee -a "${LOG_FILE}"
echo "Log             : ${LOG_FILE}" | tee -a "${LOG_FILE}"