#!/usr/bin/env bash
set -euo pipefail

MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
THREADS="${THREADS:-32}"
MAX_SEQS="${MAX_SEQS:-1000}"
INPUT_ROOT="${INPUT_ROOT:?set INPUT_ROOT to random split FASTA inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT for MMseqs2 profile outputs}"
TMP_ROOT="${TMP_ROOT:-${OUTPUT_ROOT}/tmp}"

if ! command -v "${MMSEQS_BIN}" >/dev/null 2>&1 && [[ ! -x "${MMSEQS_BIN}" ]]; then
  echo "[error] MMseqs2 executable not found: ${MMSEQS_BIN}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"
FORMAT="query,target,fident,alnlen,qcov,tcov,evalue,bits"

while IFS= read -r test_fasta; do
  rel="${test_fasta#${INPUT_ROOT}/}"
  test_dir="$(dirname "${rel}")"
  train_fasta="$(dirname "${test_fasta}")/train.fasta"
  [[ -s "${train_fasta}" ]] || { echo "[skip] no train FASTA for ${test_fasta}"; continue; }
  [[ -s "${test_fasta}" ]] || continue
  out_dir="${OUTPUT_ROOT}/${test_dir}"
  result="${out_dir}/search.tsv"
  mkdir -p "${out_dir}" "${TMP_ROOT}/${test_dir}"
  if [[ -s "${result}" ]]; then
    echo "[exists] ${result}"
    continue
  fi
  echo "[search] test=${test_fasta} train=${train_fasta}"
  "${MMSEQS_BIN}" easy-search "${test_fasta}" "${train_fasta}" "${result}" "${TMP_ROOT}/${test_dir}" \
    --format-output "${FORMAT}" --max-seqs "${MAX_SEQS}" --threads "${THREADS}" \
    --cov-mode 0 --alignment-mode 3
done < <(find "${INPUT_ROOT}" -type f -name test.fasta | sort)
