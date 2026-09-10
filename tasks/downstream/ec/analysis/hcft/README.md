# HCFT analysis

HCFT (Homology-Conflicting Functional Triplets) is an EC representation
diagnostic. It constructs triplets in which the positive shares the selected
EC label with the anchor but has lower sequence identity, while the negative
has higher sequence identity but a different EC label.

The analysis reports anchor-level HCFT accuracy and margin. Identity pairs,
EC labels, embeddings, and generated triplets are external artifacts. The
formal configurations are under `configs/analysis/hcft/`.

Run from the repository root with the released artifacts available:

```bash
PYTHONPATH=src python tasks/downstream/ec/analysis/hcft/hcft_eval.py \
  --config configs/analysis/hcft/esm2_650M_general.yaml
```

The code keeps the unit of statistical analysis at the anchor enzyme rather
than treating all sampled triplets as independent observations.
