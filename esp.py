#!/usr/bin/env python3
"""Reproduce ESP classification from released enzyme-substrate tables."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the EnzSub ESP results.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve(base, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError("Missing required file: {}".format(path))


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base = config_path.parent
    embedding_root = resolve(base, config["embedding_root"])
    output_dir = resolve(base, config["output_dir"])
    evaluator = resolve(base, "tasks/downstream/esp/2_train_evaluate.py")
    train_name = config["files"]["train"]
    test_name = config["files"]["test"]
    params = config["mlp"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for state, directory in config["states"].items():
        state_dir = embedding_root / directory
        train_file = state_dir / train_name
        test_file = state_dir / test_name
        require_file(train_file)
        require_file(test_file)
        for seed in config["seeds"]:
            run_dir = output_dir / state / "seed_{}".format(seed)
            command = [
                sys.executable,
                str(evaluator),
                "--train-file",
                str(train_file),
                "--test-files",
                str(test_file),
                "--test-names",
                "ID_Test",
                "--output-dir",
                str(run_dir),
                "--mol-feature",
                "ecfp",
                "--model-type",
                "mlp",
                "--skip-cv",
                "--seed",
                str(seed),
                "--hidden-dims",
                *[str(value) for value in params["hidden_dims"]],
                "--dropout",
                str(params["dropout"]),
                "--lr",
                str(params["lr"]),
                "--weight-decay",
                str(params["weight_decay"]),
                "--batch-size",
                str(params["batch_size"]),
                "--max-epochs",
                str(params["max_epochs"]),
                "--patience",
                str(params["patience"]),
                "--clip-grad",
                str(params["clip_grad"]),
                "--torch-device",
                str(config["device"]),
                "--mlp-pos-weight",
                "auto",
            ]
            subprocess.run(command, check=True)
            with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)["external_test_results"]["ID_Test"]["overall"]
            rows.append({"state": state, "seed": int(seed), **metrics})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output_dir / "results_per_seed.csv", index=False)
    metric_columns = ("accuracy", "roc_auc", "mcc", "f1", "precision", "recall")
    summary = []
    for state, group in per_seed.groupby("state", sort=False):
        row = {"state": state, "n_seeds": int(len(group))}
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=0))
        summary.append(row)
    pd.DataFrame(summary).to_csv(output_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
