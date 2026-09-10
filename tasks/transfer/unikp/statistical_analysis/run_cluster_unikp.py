#!/usr/bin/env python3
"""Run fixed MMseqs2-cluster folds using already saved UniKP feature caches."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_labels(root: Path, data_dir: Path, task: str) -> np.ndarray:
    sys.path.insert(0, str(root))
    from evaluate_enzsub_in_unikp import load_task

    args = SimpleNamespace(task=task, data_dir=str(data_dir))
    _, _, labels, _ = load_task(args)
    return np.asarray(labels, dtype=float)

def load_features(path: Path, expected_rows: int) -> np.ndarray:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    features = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] != expected_rows:
        raise ValueError(f"{path}: {features.shape[0]} rows, expected {expected_rows}")
    return features

def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y_true
    return {
        "r2": float(r2_score(y_true, pred)),
        "pearson": float(np.corrcoef(y_true, pred)[0, 1]) if np.std(pred) > 0 and np.std(y_true) > 0 else 0.0,
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unikp-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["kcat", "km", "kcat_km"], required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--prott5-feature-cache", type=Path, required=True)
    parser.add_argument("--enzsub-feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    labels = load_labels(args.unikp_root, args.data_dir, args.task)
    folds = pd.read_csv(args.fold_manifest)
    folds["sample_index"] = pd.to_numeric(folds["sample_index"], errors="raise").astype(int)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    if len(folds) != len(labels) or set(folds["sample_index"]) != set(range(len(labels))):
        raise ValueError("fold manifest must contain exactly one row for every sample index")
    if folds["sample_index"].duplicated().any():
        raise ValueError("fold manifest has duplicated sample_index")
    folds = folds.sort_values("sample_index")
    fold_ids = folds["fold"].to_numpy()
    prott5 = load_features(args.prott5_feature_cache, len(labels))
    enzsub = load_features(args.enzsub_feature_cache, len(labels))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    metric_rows = []
    unique_folds = sorted(np.unique(fold_ids).tolist())
    for representation, features in (("ProtT5", prott5), ("EnzSub", enzsub)):
        pred = np.full(len(labels), np.nan, dtype=float)
        for fold in unique_folds:
            test = np.flatnonzero(fold_ids == fold)
            train = np.flatnonzero(fold_ids != fold)
            kwargs = {"n_jobs": args.n_jobs, "random_state": args.seed + int(fold)}
            if args.n_estimators is not None:
                kwargs["n_estimators"] = args.n_estimators
            model = ExtraTreesRegressor(**kwargs)
            model.fit(features[train], labels[train])
            pred[test] = model.predict(features[test])
        if np.isnan(pred).any():
            raise AssertionError(f"missing predictions for {representation}")
        m = metrics(labels, pred)
        metric_rows.append({"task": args.task, "representation": representation, "split": "cluster_40id_5fold", "n_folds": len(unique_folds), **m})
        pred_rows.append(pd.DataFrame({
            "task": args.task,
            "representation": representation,
            "split": "cluster_40id_5fold",
            "sample_index": np.arange(len(labels)),
            "fold": fold_ids,
            "label_log10": labels,
            "pred_log10": pred,
        }))

    pd.DataFrame(metric_rows).to_csv(args.output_dir / f"{args.task}_cluster_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(args.output_dir / f"{args.task}_cluster_predictions.csv", index=False)
    with open(args.output_dir / f"{args.task}_cluster_run_config.json", "w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(pd.DataFrame(metric_rows).to_string(index=False))

if __name__ == "__main__":
    main()
