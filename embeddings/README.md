# EnzSub embedding archive

This directory stages the precomputed representations used by the EnzSub
downstream and analysis workflows. The files are copied from the original
experiment directories; the sources remain unchanged.

## Layout

- `ec/`: EC classification embeddings for ProtBERT-BFD, ESM-2 650M, and
  ESM-2 3B. Each backbone includes `base`, `cpt`, `base_sub`, and `cpt_sub`.
- `esp/`: enzyme embeddings used by the enzyme-substrate specificity task.
- `as/`: per-residue embeddings used by active-site prediction.
- `pH/` and `tm/`: sequence-level embeddings used by the two regression tasks.
- `transfer/ReactZyme/`: the four models in the main ReactZyme configuration,
  together with the MAT reaction embeddings required by that workflow.
- `transfer/Seq2Topt/`: precomputed per-sequence feature caches.
- `transfer/UniKP/`: EnzSub/UniKP feature matrices and sequence caches for the
  kcat, Km, and joint kcat/Km comparisons.
- `rns/`: base and CPT representations plus metadata for Random Neighbor Score
  analysis.

HCFT reuses the EC embeddings, and substrate-neighborhood coherence reuses the
ESP embeddings; these representations are not duplicated. CLEAN artifacts are
not included.

The pre-existing `as/0.4B/protbert_bfd_*` checkpoint-selection directories are
legacy selection artifacts, not part of the four-state main comparison. They
have been retained because this organization pass was copy-only.

No project-level data license is granted by this staging directory. Add the
chosen data license and repository record before public release.
