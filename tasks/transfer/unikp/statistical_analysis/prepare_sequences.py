#!/usr/bin/env python3
"""Export task records and unique protein FASTA files for homology analysis."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

TASKS = ("kcat", "km", "kcat_km")

def norm_seq(value: object) -> str:
    return str(value).upper().replace(" ", "").replace("\n", "")

def seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()[:16]

def load_task(root: Path, data_dir: Path, task: str):
    sys.path.insert(0, str(root))
    from evaluate_enzsub_in_unikp import load_task as loader

    args = SimpleNamespace(task=task, data_dir=str(data_dir))
    return loader(args)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unikp-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        sequences, smiles, labels, metadata = load_task(args.unikp_root, args.data_dir, task)
        records = pd.DataFrame(
            {
                "sample_index": range(len(labels)),
                "sequence": [norm_seq(x) for x in sequences],
                "smiles": list(smiles),
                "label_log10": labels,
            }
        )
        for column in metadata.columns:
            if column not in records.columns:
                records[column] = metadata[column].tolist()
        records["sequence_hash"] = records["sequence"].map(seq_hash)
        records["sequence_id"] = "seq_" + records["sequence_hash"]
        records["sequence_length"] = records["sequence"].str.len()

        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        records.to_csv(task_dir / "sample_manifest.csv", index=False)
        unique = (
            records.groupby(["sequence_id", "sequence_hash", "sequence"], as_index=False)
            .agg(n_records=("sample_index", "size"))
            .sort_values("sequence_id")
        )
        unique.to_csv(task_dir / "sequence_manifest.csv", index=False)
        with open(task_dir / "sequences.fasta", "w") as handle:
            for row in unique.itertuples(index=False):
                handle.write(f">{row.sequence_id}\n{row.sequence}\n")
        print(
            f"[{task}] records={len(records)} unique_sequences={len(unique)} "
            f"fasta={task_dir / 'sequences.fasta'}"
        )

if __name__ == "__main__":
    main()
