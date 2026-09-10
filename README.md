# EnzSub

EnzSub adapts pretrained protein language models to enzyme sequences and
substrate-aware objectives. This repository contains the code and formal
configurations used for continued pretraining (CPT), substrate-aware training
(SUB), downstream evaluation, transfer benchmarks, and representation analyses.

The release is organized for paper reproduction rather than as a copy of the
internal research workspace. Large checkpoints, embeddings, processed benchmark
tables, and generated results are distributed separately; their expected paths
are recorded in `data/manifest.yaml`.

## Release status

- Training and evaluation code: included.
- Paper-main CPT and SUB configurations: included for ESM2-650M, ESM2-3B, and
  ProtBERT-BFD.
- Small pH, Tm, active-site, and RNS inputs: included.
- CPT/SUB training data, EC/ESP benchmark artifacts, checkpoints, and released
  result tables: pending the public data archive.

Until the archive record is added to `data/manifest.yaml`, the repository can be
validated and inspected but cannot reproduce every reported number from a fresh
checkout.

## Repository layout

```text
configs/                 training, downstream, and analysis configurations
data/                    small versioned inputs and the external-artifact manifest
docs/reproduction.md     staged reproduction guide and evidence boundaries
scripts/                 repository validation utilities
src/enzsub/              CPT and SUB implementation
tasks/downstream/        EC, ESP, active-site, pH, and Tm evaluations
tasks/transfer/          ReactZyme, Seq2Topt, and UniKP comparisons
tasks/analysis/rns/      Random Neighbor Score analysis
artifacts/               generated outputs, excluded from version control
```

## Installation

The main experiments were developed for Linux with CUDA. Create the environment
and install the local package from the repository root:

```bash
conda env create -f environment.yml
conda activate enzsub
python -m pip install -e .
python scripts/validate_repository.py
```

PyTorch/CUDA compatibility depends on the local driver. If the pinned CUDA build
cannot be installed on a target machine, install a compatible PyTorch build first
and then install the remaining dependencies.

## Training

Run ESM2 CPT with one or more GPUs:

```bash
python -m enzsub.cpt.esm2.train_random_cpt_ddp \
  --config configs/cpt/config_esm2_650M_random_cpt.yaml \
  --gpus 0,1
```

ProtBERT-BFD uses the corresponding entry point:

```bash
python -m enzsub.cpt.protbert.train_protbert_random_cpt_ddp \
  --config configs/cpt/config_protbert_random_cpt.yaml \
  --gpus 0,1
```

Run SUB after placing the released inputs and checkpoints at the paths declared
by the selected configuration:

```bash
python -m enzsub.sub.train \
  --config configs/sub/esm2_650M/cpt_sub_r16_reg015_type005_con08_ep15.yaml
```

The same entry point accepts every Base-SUB and CPT-SUB configuration under
`configs/sub/`.

## Evaluation and analyses

Task-specific commands and artifact requirements are documented under
`tasks/downstream/`, `tasks/transfer/`, and `tasks/analysis/`. The downstream
collection covers EC prediction, enzyme-substrate specificity, active-site
prediction, optimum pH, and melting temperature. HCFT and substrate-neighborhood
coherence remain task-specific analyses; Random Neighbor Score is a separate
representation-level analysis.

See `docs/reproduction.md` for the recommended stage order and the distinction
between static validation, completed inference, and reproduced paper results.

## Authors and attribution

EnzSub code is maintained by **JkBai** and the **AGI&FBHC Laboratory**.
Transfer evaluations include adapted code from ReactZyme, Seq2Topt, and UniKP;
their upstream repositories and licenses are listed in
`THIRD_PARTY_NOTICES.md`.

The EnzSub citation, public archive DOI, and project license must be added before
the GitHub release is made public.
