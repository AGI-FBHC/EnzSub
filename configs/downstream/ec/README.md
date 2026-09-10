# EC downstream evaluation

This directory contains the public EC k-nearest-neighbour evaluation
configurations. The evaluation is deterministic: `split100` is the reference
gallery, and `split10`, `NEW`, and `PRICE` are query sets. `split10` uses the
gallery exclusion rule when it overlaps the reference set.

The embedding files and model checkpoints are external artifacts. Download and
verify them using the repository data manifest before running a configuration.

The ESM2-650M, ESM2-3B, and ProtBERT configurations correspond to the four
representation states `base`, `cpt`, `base_sub`, and `cpt_sub`. Checkpoint file
names are public-artifact placeholders until the Zenodo release manifest is
finalized.
