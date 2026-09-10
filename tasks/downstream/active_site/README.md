# Active-site prediction

This task predicts residue-level active-site labels from frozen protein
representations. The migrated workflow contains the embedding exporter,
training/evaluation code, and paper-result configurations. Per-residue
embeddings and EnzSub checkpoints are external artifacts.

The data files are under `data/active_site/`. Public configurations use
repository-relative paths and checkpoint placeholders.

Run from the repository root:

```bash
PYTHONPATH=src python tasks/downstream/active_site/scripts/sweep.py \
  --config configs/downstream/active_site/esm2_650M.yaml
```
