# UniKP statistical and homology analysis

This directory is an additive analysis layer for the completed UniKP
representation comparisons. It does not overwrite the original prediction,
metric, embedding, or cache files.

The workflow has two separate evidence classes:

1. `analyze_random_stats.py` performs sequence-level paired bootstrap and
   sign-flip tests on the existing prediction CSVs.
2. `prepare_sequences.py`, `make_random_split_fastas.py` and
   `run_mmseqs_homology.sh` prepare the same task records for MMseqs2
   homology profiling and 40% identity clustering.
3. `make_cluster_folds.py` assigns complete MMseqs2 clusters to deterministic
   folds using seed 42.
4. `run_cluster_unikp.py` evaluates the existing ProtT5 and EnzSub feature
   caches under that fixed cluster split. It never regenerates embeddings.
5. `analyze_cluster_stats.py` performs paired cluster bootstrap confidence
   intervals and complete-cluster representation-swap permutation tests on
   the saved out-of-fold predictions. It reports Holm-adjusted p-values and
   writes per-sample audit details without retraining.
6. `analyze_random_fold_stats.py` reports each held-out random 5-fold CV fold
   separately, performs within-fold sequence-level paired tests, and records
   fold-direction consistency as a supplementary sensitivity analysis.

The intended MMseqs2 settings are version 16.747c6, 80% coverage,
`--cov-mode 0`, and `--alignment-mode 3`; the main kinetic isolation uses
`--min-seq-id 0.40`. The shell script exits if an existing MMseqs2 binary is
not available instead of installing or modifying the environment.

All results should be saved under a new analysis directory, separately from
`embedding_comparison_outputs/`.

The cluster-statistics outputs are written under
`statistical_analysis/homology_40id/cluster_stats/`, including
`paired_statistics_cluster.csv`, `cluster_bootstrap_deltas.csv`,
`analysis_manifest.json`, and paired error-detail files.
