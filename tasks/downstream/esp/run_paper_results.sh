#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

python tasks/downstream/esp/sweep.py --config configs/downstream/esp/esm2_650M.yaml
python tasks/downstream/esp/sweep.py --config configs/downstream/esp/esm2_3B.yaml
python tasks/downstream/esp/sweep.py --config configs/downstream/esp/protbert.yaml
