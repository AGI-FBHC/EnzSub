#!/usr/bin/env python3
"""Assign complete MMseqs2 clusters to balanced, deterministic folds."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

def read_clusters(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", header=None, names=["representative", "member"], dtype=str)
    if raw.empty:
        raise ValueError(f"empty MMseqs2 cluster table: {path}")
    raw["cluster_id"] = raw["representative"]
    return raw

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-tsv", type=Path, required=True)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    clusters = read_clusters(args.cluster_tsv)
    sequence_manifest = pd.read_csv(args.sequence_manifest, dtype=str)
    sample_manifest = pd.read_csv(args.sample_manifest)
    required = {"sequence_id", "sequence_hash", "n_records"}
    if not required.issubset(sequence_manifest.columns):
        raise ValueError(f"sequence manifest missing {sorted(required - set(sequence_manifest.columns))}")
    if "sequence_id" not in sample_manifest.columns:
        raise ValueError("sample manifest must contain sequence_id")
    missing_members = set(clusters["member"]) - set(sequence_manifest["sequence_id"])
    if missing_members:
        raise ValueError(f"MMseqs2 cluster table contains unknown members: {len(missing_members)}")

    cluster_sizes = dict(zip(sequence_manifest["sequence_id"], sequence_manifest["n_records"].astype(int)))
    cluster_counts = clusters.groupby("cluster_id")["member"].apply(lambda x: sum(cluster_sizes[m] for m in x)).to_dict()
    rng = random.Random(args.seed)
    ordered = list(cluster_counts)
    rng.shuffle(ordered)
    ordered.sort(key=lambda cluster: cluster_counts[cluster], reverse=True)

    fold_counts = [0] * args.n_folds
    cluster_to_fold = {}
    for cluster in ordered:
        fold = min(range(args.n_folds), key=lambda i: fold_counts[i])
        cluster_to_fold[cluster] = fold
        fold_counts[fold] += cluster_counts[cluster]

    clusters["fold"] = clusters["cluster_id"].map(cluster_to_fold).astype(int)
    if clusters.groupby("cluster_id")["fold"].nunique().max() != 1:
        raise AssertionError("a cluster crosses folds")
    sequence_to_cluster = dict(zip(clusters["member"], clusters["cluster_id"]))
    sequence_manifest["cluster_id"] = sequence_manifest["sequence_id"].map(sequence_to_cluster)
    sequence_manifest["fold"] = sequence_manifest["cluster_id"].map(cluster_to_fold).astype(int)
    sample_manifest["fold"] = sample_manifest["sequence_id"].map(dict(zip(sequence_manifest["sequence_id"], sequence_manifest["fold"]))).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.output_dir / "cluster_membership_with_folds.tsv", sep="\t", index=False)
    sequence_manifest.to_csv(args.output_dir / "sequence_folds.csv", index=False)
    sample_manifest.to_csv(args.output_dir / "sample_folds.csv", index=False)
    pd.DataFrame({"fold": range(args.n_folds), "n_records": fold_counts}).to_csv(
        args.output_dir / "fold_counts.csv", index=False
    )
    print(f"saved {args.output_dir}; fold sample counts={fold_counts}")

if __name__ == "__main__":
    main()
