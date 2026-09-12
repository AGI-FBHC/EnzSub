# Data and external artifacts

Small redistribution-ready evaluation inputs are versioned in this directory.
The larger author-assembled or author-processed training inputs are hosted at
[chaohua06/EnzSub-dataset](https://huggingface.co/datasets/chaohua06/EnzSub-dataset).

Install them from the repository root with:

```bash
hf download chaohua06/EnzSub-dataset \
  --repo-type dataset \
  --local-dir data
```

The download populates:

```text
data/cpt/   ENZ30 train and validation FASTA files
data/sub/   processed OED pair and enzyme-statistics tables
data/tm/    fixed Tm train, validation, and test splits
```

Precomputed embeddings and final CPT-SUB checkpoints are distributed through
the separate repositories recorded in `manifest.yaml`. Internal raw-data
construction notebooks, exploratory intermediates, and third-party benchmark
data are intentionally outside this release.

The Hugging Face dataset includes a SHA-256 manifest. A DOI and formal data
license have not yet been assigned; use the exact repository revision when
recording provenance.
