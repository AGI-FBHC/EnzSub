#!/usr/bin/env python3
"""Summarize maximum-identity hits from MMseqs2 profile TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]

def summarize(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, sep="\t", header=None, names=COLS)
    except pd.errors.EmptyDataError:
        raw = pd.DataFrame(columns=COLS)
    if raw.empty:
        return pd.DataFrame(columns=["query", "best_target", "max_identity", "qcov", "tcov", "identity_bin"])
    for column in ["fident", "qcov", "tcov", "bits"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["query", "fident"])
    raw = raw.sort_values(["query", "fident", "qcov", "tcov", "bits"], ascending=[True, False, False, False, False])
    best = raw.drop_duplicates("query", keep="first").copy()
    best = best.rename(columns={"target": "best_target", "fident": "max_identity"})
    best["identity_bin"] = pd.cut(
        best["max_identity"],
        bins=[-0.000001, 0.30, 0.40, 0.60, 0.80, 1.000001],
        labels=["[0,0.30)", "[0.30,0.40)", "[0.40,0.60)", "[0.60,0.80)", "[0.80,1.00]"],
        right=False,
    )
    return best[["query", "best_target", "max_identity", "qcov", "tcov", "identity_bin"]]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in sorted(args.profile_root.rglob("search.tsv")):
        relative = path.relative_to(args.profile_root)
        out = args.output_root / relative.parent / "max_identity.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = summarize(path)
        result.to_csv(out, index=False)
        print(f"[saved] {out} rows={len(result)}")

if __name__ == "__main__":
    main()
