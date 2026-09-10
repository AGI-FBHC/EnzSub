# UniKP transfer benchmark

This directory evaluates EnzSub representations in the UniKP kinetic-parameter
protocol for kcat, Km, and kcat/Km. The comparison keeps the substrate encoder,
downstream ExtraTrees regressor, labels, split seeds, and metrics fixed while
replacing only the protein representation.

Run the three-backbone comparison from the repository root after installing the
released UniKP data, molecular-model assets, and EnzSub checkpoints:

```bash
bash tasks/transfer/unikp/run_kcat_km_comparison.sh
```

`DEVICE`, checkpoint paths, task selection, ProtT5 location, and cache locations
can be overridden through environment variables documented at the top of the
script. Generated caches and result tables belong under `artifacts/transfer/`
and are not committed to Git.

The statistical-analysis directory contains the random-split and
homology-isolated evaluation utilities used to summarize repeated runs. A static
code check does not establish that these evaluations have completed.

The baseline code is adapted from
[Luo-SynBioLab/UniKP](https://github.com/Luo-SynBioLab/UniKP), licensed under
GPL-3.0. Cite the original UniKP paper when using this benchmark.
