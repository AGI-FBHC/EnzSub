#!/usr/bin/env python3
"""Run the ESM2-650M Tm evaluation from released embeddings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_tm_enzsub06B.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    embedding_root = root / config["emb_root"]
    missing = [
        str(embedding_root / model / f"{split}.pt")
        for model in config["models"]
        for split in ("train", "val", "test")
        if not (embedding_root / model / f"{split}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing released embeddings:\n" + "\n".join(missing))

    command = [
        sys.executable,
        str(root / "tasks/downstream/tm/sweep.py"),
        "--config",
        str(config_path),
        "--skip-embedding",
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
