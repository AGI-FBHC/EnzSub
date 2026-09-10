#!/usr/bin/env python3
"""Paired statistics for the completed UniKP representation comparisons.

This script deliberately reads the existing prediction CSV files. It does not
retrain a model, regenerate embeddings, or change the original outputs.
The statistical unit is the unique protein sequence: repeated measurements
for one sequence are averaged before the paired permutation/bootstrap step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TASKS = ("kcat", "km", "kcat_km")
BACKBONES = (
    "esm2_650m_cpt_sub_epoch15",
    "protbert_bfd_cpt_sub_epoch15",
    "esm2_3b_cpt_sub_epoch5",
)
REQUIRED_COLUMNS = {
    "split",
    "protein_embedding",
    "run",
    "sample_index",
    "fold",
    "label_log10",
    "pred_log10",
}

def normalize_sequence(value: object) -> str:
    return str(value).upper().replace(" ", "").replace("\n", "")

def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]

def load_sequences(task: str, data_dir: Path) -> List[str]:
    # Reuse the exact loader that produced the completed prediction files.
    import sys

    # The script is stored in statistical_analysis/, while the original
    # UniKP loader remains in the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluate_enzsub_in_unikp import load_task

    args = SimpleNamespace(task=task, data_dir=str(data_dir))
    sequences, _, labels, _ = load_task(args)
    if len(sequences) != len(labels):
        raise ValueError(f"{task}: sequence/label length mismatch")
    return [normalize_sequence(x) for x in sequences]

def discover_prediction_files(root: Path, task: str) -> List[Tuple[str, Path]]:
    found = []
    for backbone in BACKBONES:
        path = root / task / backbone / f"{task}_comparison_predictions.csv"
        if path.exists():
            found.append((backbone, path))
    return found

def _metric_values(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    err = preds - labels
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    if np.std(labels) <= 1e-12 or np.std(preds) <= 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(labels, preds)[0, 1])
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "pearson": pearson}

def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    estimates = []
    for start in range(0, reps, 100):
        n = min(100, reps - start)
        indices = rng.integers(0, values.size, size=(n, values.size))
        estimates.append(np.mean(values[indices], axis=1))
    boot = np.concatenate(estimates)
    return tuple(np.quantile(boot, [0.025, 0.975]).tolist())

def _sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, reps: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    observed = abs(float(np.mean(values)))
    exceed = 0
    done = 0
    for start in range(0, reps, 100):
        n = min(100, reps - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, values.size))
        null = np.abs(np.mean(signs * values[None, :], axis=1))
        exceed += int(np.sum(null >= observed))
        done += n
    return float((exceed + 1) / (done + 1))

def _holm_adjust(pvalues: Iterable[float]) -> List[float]:
    values = np.asarray(list(pvalues), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    order = np.argsort(values[finite])
    ordered = values[finite][order]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered)
    adjusted = np.minimum(adjusted, 1.0)
    target = np.flatnonzero(finite)[order]
    result[target] = adjusted
    return result.tolist()

def analyse_pair(
    task: str,
    backbone: str,
    prediction_path: Path,
    sequences: List[str],
    rng: np.random.Generator,
    bootstrap_reps: int,
    permutation_reps: int,
    details_dir: Path,
) -> List[Dict[str, object]]:
    df = pd.read_csv(prediction_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{prediction_path}: missing columns {sorted(missing)}")

    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="raise").astype(int)
    if df["sample_index"].min() < 0 or df["sample_index"].max() >= len(sequences):
        raise ValueError(f"{prediction_path}: sample_index outside task data")
    df["fold"] = df["fold"].fillna(-1).astype(str)
    df["run"] = pd.to_numeric(df["run"], errors="raise").astype(int)
    df["sequence_hash"] = [sequence_hash(sequences[i]) for i in df["sample_index"]]

    rows: List[Dict[str, object]] = []
    for split, split_df in df.groupby("split", sort=False):
        # Holdout prediction files contain predictions for both the training
        # and test rows. Only the held-out test rows are evaluation evidence.
        if split != "5-fold CV":
            if "sample_split" not in split_df.columns:
                raise ValueError(f"{prediction_path}: holdout split lacks sample_split")
            split_df = split_df[split_df["sample_split"].astype(str).str.lower() == "test"].copy()
            if split_df.empty:
                raise ValueError(f"{prediction_path}: no held-out test rows in {split}")
        base = split_df[split_df["protein_embedding"] == "ProtT5"].copy()
        enz = split_df[split_df["protein_embedding"] == "EnzSub"].copy()
        keys = ["run", "sample_index", "fold"]
        if base.duplicated(keys).any() or enz.duplicated(keys).any():
            raise ValueError(f"{prediction_path}: duplicated paired prediction key in {split}")
        base = base[keys + ["label_log10", "pred_log10", "sequence_hash"]]
        enz = enz[keys + ["label_log10", "pred_log10"]]
        paired = base.merge(enz, on=keys, suffixes=("_base", "_enz"), how="outer", indicator=True)
        if not (paired["_merge"] == "both").all():
            raise ValueError(f"{prediction_path}: unpaired rows in {split}")
        if not np.allclose(paired["label_log10_base"], paired["label_log10_enz"], equal_nan=False):
            raise ValueError(f"{prediction_path}: labels differ between representations in {split}")

        paired["abs_err_base"] = np.abs(paired["pred_log10_base"] - paired["label_log10_base"])
        paired["abs_err_enz"] = np.abs(paired["pred_log10_enz"] - paired["label_log10_enz"])
        paired["sq_err_base"] = (paired["pred_log10_base"] - paired["label_log10_base"]) ** 2
        paired["sq_err_enz"] = (paired["pred_log10_enz"] - paired["label_log10_enz"]) ** 2
        paired["mae_improvement"] = paired["abs_err_base"] - paired["abs_err_enz"]
        paired["mse_improvement"] = paired["sq_err_base"] - paired["sq_err_enz"]
        paired["task"] = task
        paired["backbone"] = backbone
        paired["split"] = split

        details_path = details_dir / task / f"{backbone}__{str(split).replace('/', '_')}.csv"
        details_path.parent.mkdir(parents=True, exist_ok=True)
        paired.to_csv(details_path, index=False)

        sequence_level = (
            paired.groupby("sequence_hash", as_index=False)
            .agg(
                abs_err_base=("abs_err_base", "mean"),
                abs_err_enz=("abs_err_enz", "mean"),
                sq_err_base=("sq_err_base", "mean"),
                sq_err_enz=("sq_err_enz", "mean"),
                n_records=("sample_index", "size"),
            )
        )
        sequence_level["mae_improvement"] = sequence_level["abs_err_base"] - sequence_level["abs_err_enz"]
        sequence_level["mse_improvement"] = sequence_level["sq_err_base"] - sequence_level["sq_err_enz"]

        labels = paired["label_log10_base"].to_numpy(float)
        base_pred = paired["pred_log10_base"].to_numpy(float)
        enz_pred = paired["pred_log10_enz"].to_numpy(float)
        base_metrics = _metric_values(labels, base_pred)
        enz_metrics = _metric_values(labels, enz_pred)
        mae_ci = _bootstrap_ci(sequence_level["mae_improvement"].to_numpy(float), rng, bootstrap_reps)
        mse_ci = _bootstrap_ci(sequence_level["mse_improvement"].to_numpy(float), rng, bootstrap_reps)
        rows.append(
            {
                "task": task,
                "backbone": backbone,
                "prediction_file": str(prediction_path),
                "split": split,
                "n_records": int(len(paired)),
                "n_unique_sequences": int(len(sequence_level)),
                "mean_records_per_sequence": float(sequence_level["n_records"].mean()),
                "base_mae": base_metrics["mae"],
                "enzsub_mae": enz_metrics["mae"],
                "delta_mae_enzsub_minus_base": enz_metrics["mae"] - base_metrics["mae"],
                "base_rmse": base_metrics["rmse"],
                "enzsub_rmse": enz_metrics["rmse"],
                "delta_rmse_enzsub_minus_base": enz_metrics["rmse"] - base_metrics["rmse"],
                "base_r2": base_metrics["r2"],
                "enzsub_r2": enz_metrics["r2"],
                "delta_r2_enzsub_minus_base": enz_metrics["r2"] - base_metrics["r2"],
                "base_pearson": base_metrics["pearson"],
                "enzsub_pearson": enz_metrics["pearson"],
                "delta_pearson_enzsub_minus_base": enz_metrics["pearson"] - base_metrics["pearson"],
                "mae_improvement_base_minus_enzsub": float(sequence_level["mae_improvement"].mean()),
                "mae_improvement_ci95_low": mae_ci[0],
                "mae_improvement_ci95_high": mae_ci[1],
                "mae_p_two_sided_cluster_signflip": _sign_flip_pvalue(
                    sequence_level["mae_improvement"].to_numpy(float), rng, permutation_reps
                ),
                "mse_improvement_base_minus_enzsub": float(sequence_level["mse_improvement"].mean()),
                "mse_improvement_ci95_low": mse_ci[0],
                "mse_improvement_ci95_high": mse_ci[1],
                "mse_p_two_sided_cluster_signflip": _sign_flip_pvalue(
                    sequence_level["mse_improvement"].to_numpy(float), rng, permutation_reps
                ),
                "bootstrap_reps": bootstrap_reps,
                "permutation_reps": permutation_reps,
                "statistical_unit": "unique protein sequence",
            }
        )
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = args.output_dir / "paired_error_details"
    rng = np.random.default_rng(args.seed)
    all_rows: List[Dict[str, object]] = []
    audit: Dict[str, object] = {"root": str(args.root), "data_dir": str(args.data_dir), "tasks": {}}

    for task in TASKS:
        sequences = load_sequences(task, args.data_dir)
        files = discover_prediction_files(args.root, task)
        task_audit = {"n_task_records": len(sequences), "prediction_files": [str(p) for _, p in files]}
        audit["tasks"][task] = task_audit
        if not files:
            raise FileNotFoundError(f"No prediction files found for task {task}")
        for backbone, path in files:
            all_rows.extend(
                analyse_pair(
                    task,
                    backbone,
                    path,
                    sequences,
                    rng,
                    args.bootstrap_reps,
                    args.permutation_reps,
                    details_dir,
                )
            )

    result = pd.DataFrame(all_rows)
    result["mae_p_holm"] = _holm_adjust(result["mae_p_two_sided_cluster_signflip"])
    result["mse_p_holm"] = _holm_adjust(result["mse_p_two_sided_cluster_signflip"])
    result_path = args.output_dir / "paired_statistics_random.csv"
    result.to_csv(result_path, index=False)
    with open(args.output_dir / "analysis_manifest.json", "w") as handle:
        json.dump(
            {
                "analysis": "paired random-split UniKP statistics",
                "metric_space": "log10 target space",
                "primary_metric": "MAE",
                "positive_improvement": "base error minus EnzSub error",
                "sequence_grouping": "exact normalized protein sequence",
                "bootstrap_reps": args.bootstrap_reps,
                "permutation_reps": args.permutation_reps,
                "seed": args.seed,
                "source_prediction_files": audit,
                "note": "Five holdout repeats are repeated partitions, not independent biological replicates.",
            },
            handle,
            indent=2,
        )
    print(result.to_string(index=False))
    print(f"[saved] {result_path}")

if __name__ == "__main__":
    main()
