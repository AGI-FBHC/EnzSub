# Configuration policy

Configurations are part of the reproducibility record and should be committed
to Git after removing private absolute paths. Each formal configuration must
declare its data manifest, backbone, checkpoint, split, seed, optimizer,
precision, and output naming convention.

The directory is organized by workflow:

- `cpt/` and `sub/` contain the formal training configurations.
- `downstream/` contains EC, ESP, active-site, pH, and Tm configurations.
- `analysis/` contains HCFT, substrate-coherence, and RNS configurations.

Paper ablations are kept in explicitly named `ablations/` subdirectories.
Exploratory and checkpoint-selection configurations are not included.
