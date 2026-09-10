#!/usr/bin/env python3
"""Create train/test FASTA pairs from the already completed random predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

TASKS = ("kcat", "km", "kcat_km")
BACKBONES = (
    "esm2_650m_cpt_sub_epoch15",
    "protbert_bfd_cpt_sub_epoch15",
    "esm2_3b_cpt_sub_epoch5",
)

def norm_seq(value: object) -> str:
    return str(value).upper().replace(" ", "").replace("\n", "")

def load_sequences(root: Path, data_dir: Path, task: str):
    sys.path.insert(0, str(root))
    from evaluate_enzsub_in_unikp import load_task

    args = SimpleNamespace(task=task, data_dir=str(data_dir))
    sequences, _, labels, _ = load_task(args)
    return [norm_seq(x) for x in sequences], labels

def choose_prediction(root: Path, task: str) -> Path:
    for backbone in BACKBONES:
        path = root / task / backbone / f"{task}_comparison_predictions.csv"
        if path.exists():
            return path
    raise FileNotFoundError(f"no completed prediction file found for {task}")

def write_fasta(path: Path, sequences, indices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(path, "w") as handle:
        for index in sorted(set(int(x) for x in indices)):
            sequence = sequences[index]
            if sequence in seen:
                continue
            seen.add(sequence)
            handle.write(f">sample_{index}\n{sequence}\n")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unikp-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    args = parser.parse_args()

    for task in args.tasks:
        sequences, _ = load_sequences(args.unikp_root, args.data_dir, task)
        prediction_path = choose_prediction(args.prediction_root, task)
        predictions = pd.read_csv(prediction_path)
        predictions["sample_index"] = pd.to_numeric(predictions["sample_index"], errors="raise").astype(int)
        predictions["run"] = pd.to_numeric(predictions["run"], errors="raise").astype(int)
        output = args.output_dir / task
        manifest_rows = []

        for split, split_df in predictions.groupby("split", sort=False):
            if split == "5-fold CV":
                for fold in sorted(split_df["fold"].dropna().unique()):
                    test = split_df[split_df["fold"] == fold]["sample_index"].unique()
                    all_indices = pd.Index(range(len(sequences)))
                    train = all_indices[~all_indices.isin(test)].to_numpy()
                    run_dir = output / "5fold_cv" / f"fold_{int(fold):02d}"
                    write_fasta(run_dir / "train.fasta", sequences, train)
                    write_fasta(run_dir / "test.fasta", sequences, test)
                    manifest_rows.extend(
                        {"task": task, "split": split, "run": 1, "fold": int(fold), "role": "test", "sample_index": int(i)}
                        for i in test
                    )
            else:
                for run, run_df in split_df.groupby("run", sort=True):
                    run_labels = run_df["sample_split"].astype(str).str.lower()
                    test = run_df.loc[run_labels == "test", "sample_index"].unique()
                    train = run_df.loc[run_labels == "train", "sample_index"].unique()
                    if len(test) == 0 or len(train) == 0:
                        raise ValueError(f"{task} {split} run={run}: missing train/test labels")
                    run_dir = output / "holdout" / f"run_{int(run):02d}"
                    write_fasta(run_dir / "train.fasta", sequences, train)
                    write_fasta(run_dir / "test.fasta", sequences, test)
                    manifest_rows.extend(
                        {"task": task, "split": split, "run": int(run), "fold": -1, "role": "test", "sample_index": int(i)}
                        for i in test
                    )

        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows).to_csv(output / "random_split_manifest.csv", index=False)
        print(f"[{task}] source={prediction_path} output={output}")

if __name__ == "__main__":
    main()
