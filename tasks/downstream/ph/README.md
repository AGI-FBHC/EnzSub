# pH downstream task

The main pH workflow generates mean-pooled protein embeddings for the selected
representation states and trains an XGBoost regressor on the training split.
Validation data are used for model selection and the test split is evaluated
only after the model is fixed.

The public data files are under `data/ph/`. Embeddings and checkpoints are
external artifacts under the paths named in the public configurations.

Example:

```bash
PYTHONPATH=src python tasks/downstream/ph/sweep.py \
  --config configs/downstream/ph/esm2_650M.yaml
```
