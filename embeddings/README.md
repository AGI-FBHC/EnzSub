# EnzSub embedding archive

Precomputed representations are hosted at
[chaohua06/EnzSub-Embeddings](https://huggingface.co/datasets/chaohua06/EnzSub-Embeddings).
Download them from the repository root so the archive contents are placed
directly in this directory:

```bash
hf download chaohua06/EnzSub-Embeddings \
  --repo-type dataset \
  --local-dir embeddings
```

## Layout

- `ec/`: EC classification embeddings for ProtBERT-BFD, ESM-2 650M, and ESM-2
  3B, with Base, CPT, Base-SUB, and CPT-SUB states.
- `esp/`: enzyme-substrate tables and enzyme embeddings used by ESP.
- `as/`: per-residue embeddings used by active-site prediction.
- `pH/` and `tm/`: sequence-level embeddings used by the two regression tasks.
- `transfer/ReactZyme/`: protein and MAT reaction features used by ReactZyme.
- `transfer/Seq2Topt/`: precomputed per-sequence feature caches.
- `transfer/UniKP/`: EnzSub/UniKP feature matrices and sequence caches.
- `rns/`: base and comparison representations used by Random Neighbor Score.

HCFT reuses EC embeddings, and substrate-neighborhood coherence reuses ESP
embeddings; those files are not duplicated. CLEAN artifacts are not included.

The complete archive requires approximately 90 GiB of disk space. Hugging Face
supports selective downloads; for example, only the ESM-2 650M EC subset can be
installed with:

```bash
hf download chaohua06/EnzSub-Embeddings \
  --repo-type dataset \
  --include "ec/06B/**" \
  --local-dir embeddings
```

No EnzSub data license has yet been granted. The archive is publicly readable,
but reuse and redistribution permissions will be defined in a later release.
