# SUB configurations

This directory contains one Base-SUB and one CPT-SUB paper configuration for
each backbone:

- ESM2-650M uses ECFP4 (`radius=2`, `n_bits=2048`).
- ESM2-3B uses frozen ChemBERTa.
- ProtBERT-BFD uses frozen ChemBERTa.

Base-SUB starts from the original backbone. CPT-SUB starts from the matching CPT
checkpoint under `checkpoints/cpt/`. Both modes train LoRA parameters and the
SUB task heads while keeping the backbone frozen as declared in each YAML file.

Run any configuration from the repository root:

```bash
python -m enzsub.sub.train --config configs/sub/<backbone>/<config>.yaml
```

The processed OED table and enzyme statistics are external release artifacts
installed under `data/sub/`.
