# Reproducing EnzSub experiments

## Scope

The paper-main representation line consists of random-mask continued pretraining
on ENZ30 followed by Base-SUB and CPT-SUB training for ESM-2 650M, ESM-2 3B,
and ProtBERT-BFD. The formal SUB configurations use ECFP4 for ESM-2 650M and
frozen ChemBERTa for ESM-2 3B and ProtBERT-BFD.

The public release supports two distinct workflows:

1. downstream and analysis reproduction from released embeddings;
2. CPT/SUB training from the formal configurations once the training datasets
   are released.

The first workflow is currently available. The second remains data-blocked; the
code and configurations are included, but `data/cpt/` and `data/sub/` have not
yet been published.

## Reference environment

The original experiments used Linux, Python 3.8.19, PyTorch 2.1.2, and CUDA
12.1. The public `environment.yml` retains the relevant package versions from
the original `esm_env` while excluding unrelated research packages.

```bash
conda env create -f environment.yml
conda activate enzsub
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"
python scripts/validate_repository.py
```

Expected major versions are Python 3.8, PyTorch 2.1.2, and CUDA 12.1. The
repository validator confirms release structure and syntax only; it is not a
GPU or numerical reproduction test.

## Artifact installation

Download all precomputed representations to the repository-relative path used
by the public configurations:

```bash
hf download chaohua06/EnzSub-Embeddings \
  --repo-type dataset \
  --local-dir embeddings
```

The archive contains `ec/`, `esp/`, `as/`, `pH/`, `tm/`, `transfer/`, and
`rns/`. Do not add an extra directory level: after download, EC files must be
reachable as `embeddings/ec/06B/...`, not
`embeddings/EnzSub-Embeddings/ec/06B/...`.

Final CPT-SUB checkpoints are available separately:

```bash
hf download chaohua06/EnzSub --local-dir checkpoints/released
```

The resulting model files are:

```text
checkpoints/released/04B/cpt_sub_04B.pt
checkpoints/released/06B/cpt_sub_06B.pt
checkpoints/released/3B/cpt_sub_3B.pt
```

They correspond to ProtBERT-BFD, ESM-2 650M, and ESM-2 3B, respectively. These
are final CPT-SUB checkpoints. Intermediate CPT checkpoints referenced by the
CPT-SUB training configurations are not part of the current model release.

## Fast verification: EC classification

EC classification is the smallest end-to-end check because it is deterministic,
training-free, and CPU-compatible:

```bash
python knn_ec.py --config config_enzsub06B.yaml
```

The command evaluates Base, CPT, Base-SUB, and CPT-SUB on NEW-392 and PRICE-149
with `k=1`. It validates input coverage, writes results under
`artifacts/downstream/ec/results/enzsub06B/`, and fails if an MCC value differs
from the archived reference by more than `1e-6`.

## ESM-2 650M downstream reproduction

```bash
python ph.py --config config_ph_enzsub06B.yaml
python tm.py --config config_tm_enzsub06B.yaml
CUDA_VISIBLE_DEVICES=0 python esp.py --config config_esp_enzsub06B.yaml
CUDA_VISIBLE_DEVICES=0 python active_site.py --config config_active_site_enzsub06B.yaml
```

The public wrappers never regenerate embeddings. pH and Tm train fixed XGBoost
readouts on CPU. ESP and active-site prediction train MLP readouts on the
visible GPU. Their public configurations fix the representation states,
downstream hyperparameters, and paper seeds.

## Transfer comparisons

After downloading the corresponding `embeddings/transfer/` artifacts, run:

```bash
PHYSICAL_GPU=0 bash tasks/transfer/run_transfer_reproduction.sh
```

The launcher preserves the comparison protocols: ReactZyme uses ten shared
seeds, Seq2Topt uses five shared seeds with five folds per seed, and UniKP uses
five holdout runs. Individual commands and additional baseline assets are
documented in each task directory.

## CPT and SUB training

Training requires the unreleased processed inputs under `data/cpt/` and
`data/sub/`. Once available, the recommended sequence is:

```text
1. Run CPT for the selected backbone, or install the matching CPT checkpoint.
2. Run Base-SUB from the pretrained backbone.
3. Run CPT-SUB from the matching CPT checkpoint.
4. Preserve the resolved configuration and checkpoint hash with every run.
5. Generate downstream embeddings or compare against the released embeddings.
```

Example commands are provided in the root README. GPU indices and batch sizes
may be adjusted for memory capacity. Changes to the data split, seed,
optimization schedule, objective weights, checkpoint, or metric definition are
not paper-equivalent reproductions.

## Evidence levels

- **Static validation:** source and configuration checks complete; no inference.
- **Completed run:** the command exits successfully and writes its expected
  outputs.
- **Reproduced task:** released inputs, checkpoint identity, split, seeds, and
  metric definitions match, and generated task metrics are available.
- **Reproduced paper result:** generated values also agree with the archived
  paper result within the task's stated numerical tolerance.

The repository intentionally excludes raw internal data-construction notebooks,
exploratory runs, caches, and superseded configurations. Their absence does not
change the released evaluation protocols, but training cannot be claimed as
independently reproduced until the processed CPT and SUB inputs are published.
