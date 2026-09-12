# EnzSub

EnzSub adapts pretrained protein language models to enzyme sequences through
continued pretraining (CPT) and substrate-aware training (SUB). This repository
contains the paper-oriented training code, formal configurations, downstream
evaluation code, transfer benchmarks, and representation analyses.

The release is a curated reproduction repository rather than a copy of the
internal research workspace. Precomputed embeddings and final CPT-SUB
checkpoints are hosted separately on Hugging Face.

## What is included

- CPT and SUB training code for ESM-2 650M, ESM-2 3B, and ProtBERT-BFD.
- Main Base-SUB and CPT-SUB configurations for all three backbones.
- Reproduction entry points for EC, enzyme-substrate specificity (ESP),
  active-site prediction, optimum pH, and melting temperature.
- ReactZyme, Seq2Topt, and UniKP transfer comparisons.
- HCFT, substrate-neighborhood coherence, and Random Neighbor Score analyses.
- Small evaluation inputs for EC, pH, Tm, active-site prediction, and RNS.

Large embeddings and checkpoints are not tracked by Git. The currently released
artifacts are:

- [EnzSub embeddings](https://huggingface.co/datasets/chaohua06/EnzSub-Embeddings)
- [EnzSub checkpoints](https://huggingface.co/chaohua06/EnzSub)

The CPT and SUB training datasets will be released separately. Training from a
fresh base model therefore remains unavailable until those files are published;
the released embeddings are sufficient for the streamlined downstream
reproduction commands below.

## Repository layout

```text
configs/                 formal training, downstream, and analysis configurations
data/                    versioned inputs and external-artifact manifest
docs/reproduction.md     complete reproduction guide and evidence boundaries
scripts/                 repository validation utilities
src/enzsub/              CPT and SUB implementation
tasks/downstream/        EC, ESP, active-site, pH, and Tm evaluations
tasks/transfer/          ReactZyme, Seq2Topt, and UniKP comparisons
tasks/analysis/rns/      Random Neighbor Score analysis
embeddings/              download target for released representations
checkpoints/             download target for released model weights
artifacts/               generated outputs; excluded from version control
```

## Environment

The experiments were run on Linux with Python 3.8.19, PyTorch 2.1.2, and CUDA
12.1. Create the public environment from the repository root:

```bash
conda env create -f environment.yml
conda activate enzsub
python scripts/validate_repository.py
```

The environment file is a minimal reproducible specification derived from the
original `esm_env`; it intentionally excludes unrelated packages accumulated in
that research environment. A CUDA-capable Linux system is required for ESP,
active-site prediction, and full CPT/SUB training. EC, pH, and Tm can be run on
CPU from released embeddings.

## Download released artifacts

Install the Hugging Face CLI if it is not already available, then download the
embeddings directly into the path expected by the public configurations:

```bash
hf download chaohua06/EnzSub-Embeddings \
  --repo-type dataset \
  --local-dir embeddings
```

The complete embedding collection requires approximately 90 GiB of disk space.
To download only the ESM-2 650M EC files for the quickest smoke reproduction:

```bash
hf download chaohua06/EnzSub-Embeddings \
  --repo-type dataset \
  --include "ec/06B/**" \
  --local-dir embeddings
```

The model repository contains the final CPT-SUB checkpoints for ProtBERT-BFD
(`04B`), ESM-2 650M (`06B`), and ESM-2 3B (`3B`):

```bash
hf download chaohua06/EnzSub --local-dir checkpoints/released
```

These final checkpoints are not required by the embedding-only commands below.
See [the reproduction guide](docs/reproduction.md) for the artifact boundary and
expected directory layout.

## Reproduce the main ESM-2 650M downstream results

Run all commands from the repository root. EC is deterministic and uses the
paper setting `k=1` to evaluate Base, CPT, Base-SUB, and CPT-SUB on NEW-392 and
PRICE-149:

```bash
python knn_ec.py --config config_enzsub06B.yaml
```

The other streamlined entry points reuse the same four released representation
states and the paper seeds:

```bash
python ph.py --config config_ph_enzsub06B.yaml
python tm.py --config config_tm_enzsub06B.yaml
CUDA_VISIBLE_DEVICES=0 python esp.py --config config_esp_enzsub06B.yaml
CUDA_VISIBLE_DEVICES=0 python active_site.py --config config_active_site_enzsub06B.yaml
```

Generated tables and task checkpoints are written under `artifacts/`. The pH
and Tm entry points train fixed XGBoost readouts on CPU. ESP and active-site
prediction train fixed MLP readouts on the selected GPU. None of these commands
regenerates protein embeddings.

## Training

After the CPT/SUB training data are released under `data/cpt/` and `data/sub/`,
ESM-2 CPT can be launched with one or more GPUs:

```bash
python -m enzsub.cpt.esm2.train_random_cpt_ddp \
  --config configs/cpt/config_esm2_650M_random_cpt.yaml \
  --gpus 0,1
```

ProtBERT-BFD uses its corresponding entry point:

```bash
python -m enzsub.cpt.protbert.train_protbert_random_cpt_ddp \
  --config configs/cpt/config_protbert_random_cpt.yaml \
  --gpus 0,1
```

Run SUB with the matching CPT checkpoint and formal configuration:

```bash
python -m enzsub.sub.train \
  --config configs/sub/esm2_650M/cpt_sub_r16_reg015_type005_con08_ep15.yaml
```

The same entry point accepts the Base-SUB and CPT-SUB configurations under
`configs/sub/`. Hardware-dependent batch size or GPU indices may be changed;
optimization, data split, seed, and metric settings must remain unchanged for a
paper-equivalent reproduction.

## Transfer benchmarks and analyses

Task-specific commands and required external assets are documented under
`tasks/transfer/`, `tasks/downstream/`, and `tasks/analysis/`. The unified
transfer launcher preserves the repeated-evaluation protocols used for
ReactZyme, Seq2Topt, and UniKP:

```bash
PHYSICAL_GPU=0 bash tasks/transfer/run_transfer_reproduction.sh
```

HCFT reuses EC representations, substrate-neighborhood coherence reuses ESP
representations, and RNS uses its dedicated released representation set. CLEAN
is intentionally excluded from this repository.

## Reproducibility boundary

`python scripts/validate_repository.py` performs static source, YAML, shell, and
path checks. It does not run model inference. A paper result is reproduced only
when the released inputs, model state, split, seed, metric definition, and final
metric table agree. See [docs/reproduction.md](docs/reproduction.md) for the
recommended verification order.

## Authors, citation, and license

EnzSub is maintained by **JkBai** and the **AGI&FBHC Laboratory**. Adapted
third-party transfer code and upstream licenses are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The manuscript citation and DOI will be added after publication. No EnzSub
software or data license has yet been granted; source availability alone does
not grant permission to reuse, modify, or redistribute the project. A license
file will be added after the authors complete the licensing review.
