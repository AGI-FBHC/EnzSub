# Random Neighbor Score (RNS)

RNS is a representation-level uncertainty analysis that compares a fixed
backbone representation with one comparison representation. For each real
enzyme, composition-matched random sequences are added to the candidate pool;
the score is the fraction of random sequences among the top-k neighbors.
Lower RNS indicates fewer random sequences in the neighborhood.

This workflow is independent of the ESP substrate-neighborhood coherence
analysis. The latter measures substrate chemistry of enzyme neighbors, whereas
RNS measures neighborhood sensitivity to sequence randomization.

The public code includes:

- `build_random_fasta.py`: construct composition-matched random sequences.
- `prepare_rns_fasta.py`: combine real and random FASTA files and write metadata.
- `run_rns_enzsub.py`: extract base/comparison representations and compute RNS.
- `plot_rns_backbone_boxplot.py`: aggregate released RNS tables and generate plots.

The released result tables, embeddings, and checkpoints are external artifacts.
The small FASTA/metadata inputs are under `data/rns/`.

Run from the repository root with the public analysis configuration:

```bash
PYTHONPATH=src python tasks/analysis/rns/run_rns_enzsub.py \
  --config configs/analysis/rns/esm2_650M_cpt.yaml
```
