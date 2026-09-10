# Enzyme-substrate specificity

This directory contains the ESP evaluation used to compare frozen protein
representations under a common enzyme-substrate prediction protocol. Released
processed tables, molecular features, protein embeddings, and result tables are
external artifacts under `artifacts/downstream/esp/`.

Paper configurations are under `configs/downstream/esp/`. From the repository
root, inspect a run without launching model inference with:

```bash
python tasks/downstream/esp/sweep.py \
  --config configs/downstream/esp/esm2_650M.yaml \
  --dry-run
```

Remove `--dry-run` after installing the external artifacts and confirming the
device. The workflow generates protein representations, reuses the released
substrate features, and evaluates each configured readout with the same split
and seed contract.

The substrate-neighborhood coherence analysis is under
`analysis/substrate_coherence/`. It measures substrate chemistry after
enzyme-only neighbor retrieval and is distinct from Random Neighbor Score under
`tasks/analysis/rns/`.
