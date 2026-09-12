# Model checkpoints

The released final CPT-SUB checkpoints are hosted at
[chaohua06/EnzSub](https://huggingface.co/chaohua06/EnzSub). Download them from
the repository root with:

```bash
hf download chaohua06/EnzSub --local-dir checkpoints/released
```

The current release contains:

```text
released/04B/cpt_sub_04B.pt   ProtBERT-BFD CPT-SUB
released/06B/cpt_sub_06B.pt   ESM-2 650M CPT-SUB
released/3B/cpt_sub_3B.pt     ESM-2 3B CPT-SUB
```

These files are final CPT-SUB checkpoints. Intermediate CPT checkpoints used to
initialize CPT-SUB training are expected under `checkpoints/cpt/` and have not
yet been released. The embedding-only downstream entry points do not require
model checkpoints.
