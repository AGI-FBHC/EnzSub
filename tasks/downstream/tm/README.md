# Tm downstream task

The main Tm workflow generates mean-pooled protein embeddings and evaluates
them with XGBoost and k-nearest-neighbour regression. The train, validation,
and test splits are provided under `data/tm/`; embeddings and checkpoints are
external artifacts.

Example:

```bash
PYTHONPATH=src python tasks/downstream/tm/sweep.py \
  --config configs/downstream/tm/esm2_650M.yaml
```
