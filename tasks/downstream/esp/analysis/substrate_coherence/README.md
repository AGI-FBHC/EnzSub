# Substrate-neighborhood coherence analysis

This workflow is distinct from the separate Random Neighbor Score (RNS) analysis under tasks/analysis/rns/.
This analysis evaluates substrate-neighborhood consistency
for the ESP task. It retrieves neighbors using only protein embeddings, then
measures the chemical similarity of their substrate sets with an independent
Morgan fingerprint/Tanimoto procedure.

The code is under `subcoh/`; the main entry point is
`3_analyze_substrate_neighborhood_coherence.py`. Homology filtering, random
neighbor baselines, paired query-level statistics, and audit outputs are kept
in the same workflow.

Run from the repository root:

```bash
PYTHONPATH=src python \
  tasks/downstream/esp/analysis/substrate_coherence/3_analyze_substrate_neighborhood_coherence.py \
  --config configs/analysis/substrate_coherence/esm2_3B.yaml
```

Embedding tables, processed ESP data, homology tables, and generated results
are external artifacts. The analysis does not use ChemBERTa vectors to define
the neighbor relation.
