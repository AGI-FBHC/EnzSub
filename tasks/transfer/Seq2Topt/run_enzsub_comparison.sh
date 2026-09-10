#!/usr/bin/env bash
# Run the frozen-representation Seq2Topt comparison in one command.
#
# Default experiments:
#   1. base:    original ESM-2 650M backbone
#   2. cpt:     CPT backbone from the supplied checkpoint, without SUB LoRA
#   3. cpt_sub: CPT backbone plus trained SUB LoRA
#
# All modes use the same data, validation policy, task head, and seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"

ENZSUB_ROOT="${ENZSUB_ROOT:-${REPO_ROOT}/src}"
ENZSUB_CHECKPOINT="${ENZSUB_CHECKPOINT:-${REPO_ROOT}/checkpoints/sub/esm2_650M_cpt_sub.pt}"
ENCODER_TYPE="${ENCODER_TYPE:-esm2_t33_650M}"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/artifacts/transfer/seq2topt/data/Topt/train_os.csv}"
TEST_CSV="${TEST_CSV:-${REPO_ROOT}/artifacts/transfer/seq2topt/data/Topt/test.csv}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/artifacts/transfer/seq2topt/cache/esm2_650M}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/transfer/seq2topt/results/esm2_650M}"

# Space-separated lists make it easy to run repeated seeds, for example:
#   SEEDS="0 1 2" bash run_enzsub_comparison.sh
MODES="${MODES:-base cpt cpt_sub}"
SEEDS="${SEEDS:-0 1 2}"

# Paper-aligned task-head hyperparameters.
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
HEAD_DIM="${HEAD_DIM:-320}"
WINDOW="${WINDOW:-3}"
N_HEAD="${N_HEAD:-4}"
N_RD="${N_RD:-4}"
TARGET_SCALE="${TARGET_SCALE:-120}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
SPLIT_UNIT="${SPLIT_UNIT:-sequence}"

# Resource controls. Batch size 1 is the conservative default for the 650M
# encoder and long proteins; raise it to 2 or 4 when GPU memory permits.
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float16}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"
USE_AMP="${USE_AMP:-1}"

# Existing compatible feature caches are reused. Existing completed training
# runs are skipped unless FORCE_TRAIN=1. Set FORCE_CACHE=1 to rebuild features.
FORCE_CACHE="${FORCE_CACHE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

read -r -a MODE_ARRAY <<< "${MODES}"
read -r -a SEED_ARRAY <<< "${SEEDS}"

mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
  printf '  +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

contains_non_base_mode=0
for mode in "${MODE_ARRAY[@]}"; do
  case "${mode}" in
    base) ;;
    cpt|cpt_sub) contains_non_base_mode=1 ;;
    *)
      echo "Unsupported mode: ${mode}. Choose from: base cpt cpt_sub" >&2
      exit 2
      ;;
  esac
done

[[ -f "${ENZSUB_ROOT}/sub/model.py" ]] || {
  echo "EnzSub root must contain sub/model.py: ${ENZSUB_ROOT}" >&2
  exit 2
}
if [[ "${contains_non_base_mode}" == "1" && ! -f "${ENZSUB_CHECKPOINT}" ]]; then
  echo "EnzSub checkpoint not found: ${ENZSUB_CHECKPOINT}" >&2
  exit 2
fi
[[ -f "${TRAIN_CSV}" ]] || { echo "Training CSV not found: ${TRAIN_CSV}" >&2; exit 2; }
[[ -f "${TEST_CSV}" ]] || { echo "Test CSV not found: ${TEST_CSV}" >&2; exit 2; }

log "Environment check"
run_cmd "${PYTHON}" -c \
  'import torch, numpy, pandas, esm; print("torch", torch.__version__); print("cuda", torch.cuda.is_available()); print("numpy", numpy.__version__); print("pandas", pandas.__version__); print("fair-esm", esm.__version__)'
if [[ "${DEVICE}" == cuda* ]]; then
  run_cmd "${PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA device requested but CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
fi

log "Run configuration"
printf '  project:              %s\n' "${SCRIPT_DIR}"
printf '  EnzSub root:          %s\n' "${ENZSUB_ROOT}"
printf '  EnzSub checkpoint:    %s\n' "${ENZSUB_CHECKPOINT}"
printf '  encoder:              %s\n' "${ENCODER_TYPE}"
printf '  modes:                %s\n' "${MODES}"
printf '  seeds:                %s\n' "${SEEDS}"
printf '  split unit:           %s\n' "${SPLIT_UNIT}"
printf '  cache root:           %s\n' "${CACHE_ROOT}"
printf '  output root:          %s\n' "${OUTPUT_ROOT}"
df -h "${CACHE_ROOT}" | sed 's/^/  /'

for mode in "${MODE_ARRAY[@]}"; do
  cache_dir="${CACHE_ROOT}/${ENCODER_TYPE}/${mode}"
  mkdir -p "${cache_dir}"

  precompute_cmd=(
    "${PYTHON}" "${CODE_DIR}/precompute_enzsub_features.py"
    --enzsub-root "${ENZSUB_ROOT}"
  )
  if [[ "${mode}" != "base" ]]; then
    precompute_cmd+=(--enzsub-checkpoint "${ENZSUB_CHECKPOINT}")
  fi
  precompute_cmd+=(
    --encoder-type "${ENCODER_TYPE}"
    --model-mode "${mode}"
    --inputs "${TRAIN_CSV}" "${TEST_CSV}"
    --cache-dir "${cache_dir}"
    --batch-size "${CACHE_BATCH_SIZE}"
    --device "${DEVICE}"
    --storage-dtype "${STORAGE_DTYPE}"
  )
  if [[ -n "${MAX_SEQ_LENGTH}" ]]; then
    precompute_cmd+=(--max-seq-length "${MAX_SEQ_LENGTH}")
  fi
  if [[ "${USE_AMP}" == "1" ]]; then
    precompute_cmd+=(--amp)
  fi
  if [[ "${FORCE_CACHE}" == "1" ]]; then
    precompute_cmd+=(--overwrite)
  fi

  log "Precompute/reuse EnzSub features: mode=${mode}"
  run_cmd "${precompute_cmd[@]}"

  for seed in "${SEED_ARRAY[@]}"; do
    run_dir="${OUTPUT_ROOT}/${ENCODER_TYPE}/${mode}/seed_${seed}"
    mkdir -p "${run_dir}"
    if [[ -f "${run_dir}/metrics.json" && "${FORCE_TRAIN}" != "1" ]]; then
      log "Skip completed training: mode=${mode}, seed=${seed}"
      continue
    fi

    log "Train Seq2Topt head: mode=${mode}, seed=${seed}"
    train_cmd=(
      "${PYTHON}" "${CODE_DIR}/run_train_enzsub.py"
      --train-csv "${TRAIN_CSV}" \
      --test-csv "${TEST_CSV}" \
      --cache-dir "${cache_dir}" \
      --output-dir "${run_dir}" \
      --seed "${seed}" \
      --split-unit "${SPLIT_UNIT}" \
      --validation-ratio "${VALIDATION_RATIO}" \
      --target-scale "${TARGET_SCALE}" \
      --head-dim "${HEAD_DIM}" \
      --window "${WINDOW}" \
      --n-head "${N_HEAD}" \
      --n-rd "${N_RD}" \
      --epochs "${EPOCHS}" \
      --learning-rate "${LEARNING_RATE}" \
      --batch-size "${TRAIN_BATCH_SIZE}" \
      --effective-batch-size "${EFFECTIVE_BATCH_SIZE}" \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}"
    )
    if [[ "${USE_AMP}" == "1" ]]; then
      train_cmd+=(--amp)
    fi
    run_cmd "${train_cmd[@]}" 2>&1 | tee "${run_dir}/console.log"
  done
done

log "Summarize experiments"
run_cmd "${PYTHON}" - "${OUTPUT_ROOT}" "${ENCODER_TYPE}" "${MODES}" "${SEEDS}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
encoder_type = sys.argv[2]
modes = sys.argv[3].split()
seeds = sys.argv[4].split()
rows = []

for mode in modes:
    for seed in seeds:
        metrics_path = output_root / encoder_type / mode / f"seed_{seed}" / "metrics.json"
        if not metrics_path.is_file():
            print(f"WARNING: missing {metrics_path}", file=sys.stderr)
            continue
        result = json.loads(metrics_path.read_text())
        val = result["best_validation_metrics"]
        test = result["test_metrics"]
        rows.append({
            "encoder_type": encoder_type,
            "mode": mode,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "validation_rmse": val["rmse"],
            "validation_mae": val["mae"],
            "validation_r2": val["r2"],
            "test_rmse": test["rmse"],
            "test_mae": test["mae"],
            "test_r2": test["r2"],
        })

fields = [
    "encoder_type", "mode", "seed", "best_epoch",
    "validation_rmse", "validation_mae", "validation_r2",
    "test_rmse", "test_mae", "test_r2",
]
summary_path = output_root / "summary.csv"
with summary_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

aggregate_rows = []
for mode in modes:
    selected = [row for row in rows if row["mode"] == mode]
    if not selected:
        continue
    aggregate = {"encoder_type": encoder_type, "mode": mode, "n_seeds": len(selected)}
    for metric in ("validation_rmse", "validation_mae", "validation_r2", "test_rmse", "test_mae", "test_r2"):
        values = [float(row[metric]) for row in selected]
        aggregate[f"{metric}_mean"] = statistics.mean(values)
        aggregate[f"{metric}_std"] = statistics.pstdev(values)
    aggregate_rows.append(aggregate)

aggregate_path = output_root / "summary_by_mode.csv"
if aggregate_rows:
    with aggregate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

paper_reference = {
    "model": "published Seq2Topt ESM-2 8M release reference",
    "test_n": 291,
    "test_rmse": 12.256410477984348,
    "test_mae": 8.886408722605491,
    "test_r2": 0.5669624162932285,
}
(output_root / "published_reference.json").write_text(
    json.dumps(paper_reference, indent=2, sort_keys=True) + "\n"
)

print(f"Wrote {summary_path}")
if aggregate_rows:
    print(f"Wrote {aggregate_path}")
print(f"Wrote {output_root / 'published_reference.json'}")
PY

log "All requested experiments completed"
printf '  Per-run results: %s/%s/<mode>/seed_<seed>/\n' "${OUTPUT_ROOT}" "${ENCODER_TYPE}"
printf '  Run summary:     %s/summary.csv\n' "${OUTPUT_ROOT}"
printf '  Mode summary:    %s/summary_by_mode.csv\n' "${OUTPUT_ROOT}"
printf '  Paper reference: %s/published_reference.json\n' "${OUTPUT_ROOT}"
