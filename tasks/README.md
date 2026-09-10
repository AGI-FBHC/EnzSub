# Additional downstream tasks

This directory contains the non-CLEAN downstream workflows migrated from the
research workspace. Task code is kept here, while public run configurations are
centralized under `configs/`. Embeddings, checkpoints, model caches, logs, and
generated results are external artifacts.

Migrated groups:

- `tasks/downstream/active_site/`: active-site prediction.
- `tasks/downstream/esp/`: enzyme-substrate specificity and substrate-neighborhood
  analyses.
- `tasks/transfer/ReactZyme/`: reaction-level retrieval and comparison workflows.
- `tasks/transfer/Seq2Topt/`: temperature-optimum prediction and EnzSub adapter
  experiments.
- `tasks/transfer/unikp/`: UniKP kinetic-parameter baselines and EnzSub
  comparisons.

The `CLEAN` workflow is intentionally excluded.
