# Reproducing EnzSub experiments

## Scope

The paper-main representation line consists of random-mask continued pretraining
on ENZ30 followed by Base-SUB and CPT-SUB training for ESM2-650M, ESM2-3B, and
ProtBERT-BFD. Downstream evaluations cover EC, enzyme-substrate specificity,
active sites, optimum pH, and melting temperature. ReactZyme, Seq2Topt, and
UniKP are maintained as transfer benchmarks.

The formal SUB configurations use ECFP4 for ESM2-650M and frozen ChemBERTa for
ESM2-3B and ProtBERT-BFD. These choices are fixed in `configs/sub/`.

## Reproduction levels

1. `python scripts/validate_repository.py` verifies source syntax, YAML syntax,
   public paths, duplicate README names, and forbidden generated files. This is
   a static repository check, not model inference.
2. CPT or SUB training is complete only when the configured run finishes and
   writes its checkpoint and configuration snapshot.
3. A downstream task is reproduced only when released inputs and checkpoints
   have been installed, inference has completed, and the task's evaluation
   output has been generated.
4. A paper result is reproduced only when the metric definition, split, model
   checkpoint, random seeds, and released result table all match the paper.

## External artifacts

Download the release archive identified in `data/manifest.yaml` and extract it
into the repository root. The archive is expected to populate `data/cpt/`,
`data/sub/`, `data/ec/`, `artifacts/`, and `checkpoints/` without requiring
absolute path edits. Do not rename checkpoint files after extraction because the
paper-main YAML files use stable public names.

The repository intentionally does not publish internal data-construction
notebooks, exploratory runs, caches, or raw intermediate files. Released
processed inputs are sufficient for the public training and evaluation entry
points.

## Recommended order

```text
1. Install the environment and the local package.
2. Run scripts/validate_repository.py.
3. Install the external artifact archive.
4. Run CPT, or use the released CPT checkpoints.
5. Run Base-SUB and CPT-SUB, or use the released SUB checkpoints.
6. Run each downstream or transfer task from its own README.
7. Compare generated outputs with the released result tables.
```

Every released run should preserve its configuration, random seed, base-model
identifier, checkpoint identifier, software environment, and output manifest.
GPU index and batch size may be adjusted for hardware capacity; changes that
alter optimization, model weights, data splits, or metric definitions are not
equivalent reproductions.
