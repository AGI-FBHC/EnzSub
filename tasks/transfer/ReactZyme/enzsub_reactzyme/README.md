# EnzSub-ReactZyme comparison

This workflow evaluates frozen enzyme representations with one shared
ReactZyme reaction encoder, matching head, data split, and retrieval metric
contract. It supports EnzSub, the released ReactZyme ESM encoder, fair-ESM
baselines, and explicitly configured external protein language models.

## Required assets

The configured split directory must contain:

```text
positive_train_val_seq_smi.pt
negative_train_val_seq_smi.pt
positive_test_seq_smi.pt
negative_test_seq_smi.pt
```

The MAT checkpoint, split files, model checkpoints, and released results are
external artifacts. Their public paths are represented in `config.example.json`
and `data/manifest.yaml`.

## Protocol

Reaction representations are generated once and shared across protein
representations. For each protein representation, the matching head is trained
from scratch with the same five seeds and selected using a validation partition
inside the released training pool. The held-out test partition is evaluated only
after model selection. The primary retrieval candidates are the unique positive
enzymes and reactions in the test partition.

The public ReactZyme release does not provide all negative files required by the
comparison. When regenerated, this workflow uses the documented reaction-side
hard-negative procedure in `generate_negatives.py` and records the seed,
candidate pool, exclusions, and input hashes in the protocol manifest.

## Run

Copy the example configuration, edit only public artifact locations and the
available device, then launch the pipeline from the repository root:

```bash
cp tasks/transfer/ReactZyme/enzsub_reactzyme/config.example.json config.local.json
python tasks/transfer/ReactZyme/enzsub_reactzyme/run_pipeline.py \
  --config config.local.json
```

Use `--stage` to run a single stage and `--resume` to reuse complete outputs.
`audit_protocol.py` checks split coverage, pair counts, overlaps, hashes, and
embedding metadata before training. `collect_comparison.py` aggregates seed-level
metrics only after all requested runs are present.

This code adapts [WillHua127/ReactZyme](https://github.com/WillHua127/ReactZyme),
released under CC0-1.0. Cite the original ReactZyme work when using this
benchmark.
