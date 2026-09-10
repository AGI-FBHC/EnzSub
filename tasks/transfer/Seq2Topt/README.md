# Seq2Topt transfer benchmark

This directory compares frozen EnzSub residue representations in the Seq2Topt
optimal-temperature prediction architecture. The supplied paper train/test split,
task head, target scaling, and evaluation metrics remain fixed across protein
representations.

The EnzSub workflow caches frozen residue features, trains only the dimensional
adapter and Seq2Topt prediction head, selects a checkpoint on a deterministic
validation partition, and evaluates the held-out test set after selection. Cache
manifests include the EnzSub checkpoint hash so incompatible features are not
silently reused.

After installing the released Seq2Topt data and EnzSub checkpoints, run a
backbone-specific comparison from the repository root:

```bash
bash tasks/transfer/Seq2Topt/run_enzsub_comparison.sh
bash tasks/transfer/Seq2Topt/run_enzsub_comparison_3b.sh
bash tasks/transfer/Seq2Topt/run_enzsub_comparison_protbert.sh
```

The scripts compare Base, CPT, and CPT-SUB under the same task protocol. Device,
seeds, modes, batch sizes, input tables, and checkpoint paths can be overridden
with environment variables. Inputs, feature caches, task checkpoints, and result
tables are stored under `artifacts/transfer/seq2topt/`.

The five-fold comparison is available through
`run_seq2topt_cv_comparison.sh`. Tests under `tests/` cover cache metadata,
masking, checkpoint compatibility, and split behavior without running the full
protein backbone.

The baseline code is adapted from
[SizheQiu/Seq2Topt](https://github.com/SizheQiu/Seq2Topt), licensed under
GPL-3.0. Cite the original Seq2Topt paper when using this benchmark.
