# EC-number classification

The released EC evaluation is a deterministic, training-free cosine k-nearest
neighbour readout over frozen protein embeddings. The paper protocol fixes
`k=1` and uses `split100` as the reference gallery.

From the repository root, run:

```bash
python knn_ec.py --config config_enzsub06B.yaml
```

The configuration evaluates the four ESM2-650M representation states (Base,
CPT, Base-SUB, and CPT-SUB) on NEW-392 and PRICE-149. It reads labels from
`data/ec/`, embeddings from `embeddings/ec/06B/`, and writes derived results to
`artifacts/downstream/ec/results/enzsub06B/`.

The script verifies input coverage before evaluation and exits with an error if
any reproduced MCC differs from the archived paper result.
