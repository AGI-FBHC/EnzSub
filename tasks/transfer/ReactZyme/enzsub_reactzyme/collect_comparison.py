#!/usr/bin/env python3
"""Collect per-model ReactZyme summaries into JSON and CSV comparison tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    rows = []
    for model_name in args.models:
        summary = json.loads((args.results_dir / model_name / "summary.json").read_text())
        for direction, metrics in summary["aggregate_mean_sd"].items():
            for metric, value in metrics.items():
                rows.append({"model": model_name, "direction": direction, "metric": metric, "mean": value["mean"], "sd": value["sd"], "n_seeds": len(summary["runs"])})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.results_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "direction", "metric", "mean", "sd", "n_seeds"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved comparison for {len(args.models)} models to {args.results_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
