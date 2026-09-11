# Additional downstream baselines

This directory contains the non-CLEAN downstream baselines used alongside
EnzSub: ReactZyme, Seq2Topt, and UniKP. Their large pretrained models,
embeddings, caches, and generated results are external artifacts.

`CLEAN` is not part of this public repository.

After downloading the matching data and embedding bundle into `artifacts/`
and `embeddings/`, reproduce all three controlled comparisons with:

```bash
PHYSICAL_GPU=6 bash tasks/transfer/run_transfer_reproduction.sh
```

The launcher preserves the formal repeated-evaluation protocols: UniKP uses
five holdout runs, Seq2Topt uses five shared seeds with five folds per seed,
and ReactZyme uses ten shared seeds for every representation. The experiments
reuse released embeddings and do not regenerate them.
