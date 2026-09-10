# EC downstream task

The public EC workflow uses frozen protein representations and a deterministic
k-nearest-neighbour multi-label readout. It does not train an EC classifier.

The default protocol uses `split100` as the reference gallery and evaluates
queries from `NEW`, `PRICE`, and `split10`. The `split10` query uses the same
`split100` gallery and excludes an identical query entry when present.

Run from the repository root:

```bash
PYTHONPATH=src python tasks/downstream/ec/sweep/sweep.py \
  --config configs/downstream/ec/esm2_650M.yaml
```

The command expects the Zenodo data, embeddings/checkpoints, and the paths in
the selected configuration. It writes only derived outputs under `artifacts/`.

The embedding exporter supports the four representation states `base`, `cpt`,
`base_sub`, and `cpt_sub`. `knn_eval.py` can also evaluate previously released
embedding directories without loading a protein encoder.
