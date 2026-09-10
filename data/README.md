# Data and external artifacts

Small redistribution-ready inputs are versioned in this directory. Large
training sets, benchmark tables, embeddings, model checkpoints, and generated
results are released separately and are described in `manifest.yaml`.

After extracting the public archive into the repository root, the paths used by
the formal YAML configurations should resolve without editing machine-specific
directories. Internal raw-data construction scripts and exploratory
intermediates are intentionally outside the public release.

The archive DOI, file checksums, and final release status must be filled in before
publication. A missing DOI or checksum means that the associated experiment is
not yet independently reproducible from a fresh checkout.
