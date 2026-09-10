#!/usr/bin/env bash
set -euo pipefail

# Use the same conventions as the EnzSub analyses:
# MMseqs2, 80% coverage, cov-mode 0, alignment-mode 3.
# This script never installs MMseqs2. Set MMSEQS_BIN explicitly if it is not
# on PATH in the server environment.

MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
IDENTITY="${IDENTITY:-0.40}"
COVERAGE="${COVERAGE:-0.80}"
THREADS="${THREADS:-32}"
INPUT_ROOT="${INPUT_ROOT:?set INPUT_ROOT to the prepared sequence directory}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT for MMseqs2 outputs}"
TMP_ROOT="${TMP_ROOT:-${OUTPUT_ROOT}/tmp}"

if ! command -v "${MMSEQS_BIN}" >/dev/null 2>&1 && [[ ! -x "${MMSEQS_BIN}" ]]; then
  echo "[error] MMseqs2 executable not found: ${MMSEQS_BIN}" >&2
  echo "[hint] locate the existing installation and rerun with MMSEQS_BIN=/absolute/path/mmseqs" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"

for task_dir in "${INPUT_ROOT}"/*; do
  [[ -d "${task_dir}" ]] || continue
  task="$(basename "${task_dir}")"
  fasta="${task_dir}/sequences.fasta"
  [[ -s "${fasta}" ]] || { echo "[skip] ${task}: missing ${fasta}"; continue; }
  prefix="${OUTPUT_ROOT}/${task}/cluster_40id_cov80"
  mkdir -p "${OUTPUT_ROOT}/${task}" "${TMP_ROOT}/${task}"
  if [[ -s "${prefix}_cluster.tsv" ]]; then
    echo "[exists] ${prefix}_cluster.tsv"
    continue
  fi
  echo "[cluster] ${task} identity=${IDENTITY} coverage=${COVERAGE} threads=${THREADS}"
  "${MMSEQS_BIN}" easy-cluster "${fasta}" "${prefix}" "${TMP_ROOT}/${task}" \
    --min-seq-id "${IDENTITY}" -c "${COVERAGE}" \
    --cov-mode 0 --alignment-mode 3 --threads "${THREADS}"
done
