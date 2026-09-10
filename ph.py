#!/usr/bin/env python3
"""Reproduce optimum-pH regression from released protein embeddings."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the EnzSub pH results.")
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
    evaluator = resolve(base, "tasks/downstream/ph/xgboost_eval.py")
    data = {name: resolve(base, value) for name, value in config["data"].items()}
    for path in data.values():
        require_file(path)

    embedding_args = []
    for state, directory in config["states"].items():
        state_dir = embedding_root / directory
        for split in ("train", "val", "test"):
            require_file(state_dir / (split + "_all.pt"))
        embedding_args.append("{}:{}".format(state, state_dir))

    params = config["xgboost"]
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in config["seeds"]:
        run_name = "seed_{}".format(seed)
        command = [
            sys.executable,
            str(evaluator),
            "--embedding-dirs",
            *embedding_args,
            "--train-csv",
            str(data["train"]),
            "--val-csv",
            str(data["val"]),
            "--test-csv",
            str(data["test"]),
            "--output-dir",
            str(output_dir),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--n-estimators",
            str(params["n_estimators"]),
            "--max-depth",
            str(params["max_depth"]),
            "--learning-rate",
            str(params["learning_rate"]),
            "--subsample",
            str(params["subsample"]),
            "--colsample-bytree",
            str(params["colsample_bytree"]),
            "--min-child-weight",
            str(params["min_child_weight"]),
            "--gamma",
            str(params["gamma"]),
            "--reg-alpha",
            str(params["reg_alpha"]),
            "--reg-lambda",
            str(params["reg_lambda"]),
            "--colsample-bylevel",
            str(params["colsample_bylevel"]),
            "--colsample-bynode",
            str(params["colsample_bynode"]),
            "--tree-method",
            str(params["tree_method"]),
            "--n-jobs",
            str(params["n_jobs"]),
            "--early-stopping-rounds",
            "0",
            "--cv-folds",
            "0",
            "--no-scaler",
        ]
        subprocess.run(command, check=True)
        frame = pd.read_csv(output_dir / run_name / "results.csv")
        frame["seed"] = int(seed)
        per_seed.append(frame)

    all_runs = pd.concat(per_seed, ignore_index=True)
    all_runs.to_csv(output_dir / "results_per_seed.csv", index=False)
    metric_columns = [name for name in all_runs.columns if name.startswith("test_")]
    rows = []
    for state, group in all_runs.groupby("embedding", sort=False):
        row = {"state": state, "n_seeds": int(len(group))}
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
