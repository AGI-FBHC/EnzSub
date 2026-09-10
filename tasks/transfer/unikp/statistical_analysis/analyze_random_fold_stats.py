#!/usr/bin/env python3
"""Fold-level consistency and paired tests for the existing 5-fold CV results.

This is a supplementary analysis to ``analyze_random_stats.py``. It uses the
same saved prediction files and metric space, but reports each held-out fold
separately. Within a fold, repeated records from one protein are averaged
before the paired sign-flip/bootstrap analysis. The five folds are not treated
as independent biological replicates.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from analyze_random_stats import (
    BACKBONES,
    TASKS,
    _bootstrap_ci,
    _holm_adjust,
    _metric_values,
    _sign_flip_pvalue,
    discover_prediction_files,
    load_sequences,
    sequence_hash,
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

def _exact_fold_signflip_pvalue(values: Iterable[float]) -> float:
    """Exact two-sided sign-flip sensitivity test across at most five folds."""

    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-15)]
    if values.size == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        null.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed))

def _pair_fold(
    task: str,
    backbone: str,
    prediction_path: Path,
    sequences: List[str],
    rng: np.random.Generator,
    bootstrap_reps: int,
    permutation_reps: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    df = pd.read_csv(prediction_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{prediction_path}: missing columns {sorted(missing)}")
    cv = df[df["split"].astype(str) == "5-fold CV"].copy()
    if cv.empty:
        raise ValueError(f"{prediction_path}: no 5-fold CV rows")
    cv["sample_index"] = pd.to_numeric(cv["sample_index"], errors="raise").astype(int)
    cv["run"] = pd.to_numeric(cv["run"], errors="raise").astype(int)
    cv["fold"] = pd.to_numeric(cv["fold"], errors="raise").astype(int)
    cv["label_log10"] = pd.to_numeric(cv["label_log10"], errors="raise")
    cv["pred_log10"] = pd.to_numeric(cv["pred_log10"], errors="raise")
    cv["sequence_hash"] = [sequence_hash(sequences[i]) for i in cv["sample_index"]]

    fold_rows: List[Dict[str, object]] = []
    sequence_rows: List[Dict[str, object]] = []
    for (run, fold), fold_df in cv.groupby(["run", "fold"], sort=True):
        base = fold_df[fold_df["protein_embedding"] == "ProtT5"].copy()
        enz = fold_df[fold_df["protein_embedding"] == "EnzSub"].copy()
        keys = ["run", "sample_index", "fold"]
        if base.duplicated(keys).any() or enz.duplicated(keys).any():
            raise ValueError(f"{prediction_path}: duplicated paired key in run={run}, fold={fold}")
        base = base[keys + ["label_log10", "pred_log10", "sequence_hash"]]
        enz = enz[keys + ["label_log10", "pred_log10"]]
        paired = base.merge(enz, on=keys, how="outer", suffixes=("_base", "_enz"), indicator=True)
        if not (paired["_merge"] == "both").all():
            raise ValueError(f"{prediction_path}: unpaired rows in run={run}, fold={fold}")
        if not np.allclose(paired["label_log10_base"], paired["label_log10_enz"], equal_nan=False):
            raise ValueError(f"{prediction_path}: labels differ in run={run}, fold={fold}")

        labels = paired["label_log10_base"].to_numpy(float)
        base_pred = paired["pred_log10_base"].to_numpy(float)
        enz_pred = paired["pred_log10_enz"].to_numpy(float)
        base_metrics = _metric_values(labels, base_pred)
        enz_metrics = _metric_values(labels, enz_pred)
        sequence_level = (
            paired.assign(
                abs_err_base=np.abs(base_pred - labels),
                abs_err_enz=np.abs(enz_pred - labels),
                sq_err_base=(base_pred - labels) ** 2,
                sq_err_enz=(enz_pred - labels) ** 2,
            )
            .groupby("sequence_hash", as_index=False)
            .agg(
                abs_err_base=("abs_err_base", "mean"),
                abs_err_enz=("abs_err_enz", "mean"),
                sq_err_base=("sq_err_base", "mean"),
                sq_err_enz=("sq_err_enz", "mean"),
                n_records=("sample_index", "size"),
            )
        )
        sequence_level["mae_improvement_base_minus_enzsub"] = sequence_level["abs_err_base"] - sequence_level["abs_err_enz"]
        sequence_level["mse_improvement_base_minus_enzsub"] = sequence_level["sq_err_base"] - sequence_level["sq_err_enz"]
        sequence_level.insert(0, "fold", int(fold))
        sequence_level.insert(0, "run", int(run))
        sequence_level.insert(0, "backbone", backbone)
        sequence_level.insert(0, "task", task)
        sequence_rows.extend(sequence_level.to_dict("records"))

        mae_values = sequence_level["mae_improvement_base_minus_enzsub"].to_numpy(float)
        mse_values = sequence_level["mse_improvement_base_minus_enzsub"].to_numpy(float)
        mae_ci = _bootstrap_ci(mae_values, rng, bootstrap_reps)
        mse_ci = _bootstrap_ci(mse_values, rng, bootstrap_reps)
        row: Dict[str, object] = {
            "task": task,
            "backbone": backbone,
            "run": int(run),
            "fold": int(fold),
            "split": "5-fold CV",
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
            "mae_improvement_base_minus_enzsub": float(mae_values.mean()),
            "mae_improvement_ci95_low": mae_ci[0],
            "mae_improvement_ci95_high": mae_ci[1],
            "mae_p_two_sided_sequence_signflip": _sign_flip_pvalue(mae_values, rng, permutation_reps),
            "mse_improvement_base_minus_enzsub": float(mse_values.mean()),
            "mse_improvement_ci95_low": mse_ci[0],
            "mse_improvement_ci95_high": mse_ci[1],
            "mse_p_two_sided_sequence_signflip": _sign_flip_pvalue(mse_values, rng, permutation_reps),
            "statistical_unit_within_fold": "unique protein sequence",
        }
        fold_rows.append(row)
    return fold_rows, sequence_rows, []

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.bootstrap_reps < 100 or args.permutation_reps < 100:
        raise ValueError("use at least 100 bootstrap and permutation repetitions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fold_rows: List[Dict[str, object]] = []
    sequence_rows: List[Dict[str, object]] = []
    audit: Dict[str, object] = {"analysis": "supplementary fold-level random 5-fold CV statistics", "tasks": {}}
    for task in TASKS:
        sequences = load_sequences(task, args.data_dir)
        files = discover_prediction_files(args.root, task)
        audit["tasks"][task] = {"n_task_records": len(sequences), "prediction_files": [str(path) for _, path in files]}
        for backbone, prediction_path in files:
            rows, seq_rows, _ = _pair_fold(
                task,
                backbone,
                prediction_path,
                sequences,
                rng,
                args.bootstrap_reps,
                args.permutation_reps,
            )
            fold_rows.extend(rows)
            sequence_rows.extend(seq_rows)

    fold_df = pd.DataFrame(fold_rows)
    fold_df["mae_p_holm_across_all_folds"] = _holm_adjust(fold_df["mae_p_two_sided_sequence_signflip"])
    fold_df["mse_p_holm_across_all_folds"] = _holm_adjust(fold_df["mse_p_two_sided_sequence_signflip"])
    fold_df.to_csv(args.output_dir / "fold_statistics_random_cv.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(args.output_dir / "fold_sequence_improvements.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    for (task, backbone, run), group in fold_df.groupby(["task", "backbone", "run"], sort=True):
        mae_delta = group["delta_mae_enzsub_minus_base"].to_numpy(float)
        rmse_delta = group["delta_rmse_enzsub_minus_base"].to_numpy(float)
        r2_delta = group["delta_r2_enzsub_minus_base"].to_numpy(float)
        pearson_delta = group["delta_pearson_enzsub_minus_base"].to_numpy(float)
        summary_rows.append(
            {
                "task": task,
                "backbone": backbone,
                "run": int(run),
                "n_folds": int(len(group)),
                "n_improved_mae_folds": int(np.sum(mae_delta < 0)),
                "n_improved_rmse_folds": int(np.sum(rmse_delta < 0)),
                "n_improved_r2_folds": int(np.sum(r2_delta > 0)),
                "n_improved_pearson_folds": int(np.sum(pearson_delta > 0)),
                "mean_delta_mae": float(np.mean(mae_delta)),
                "median_delta_mae": float(np.median(mae_delta)),
                "min_delta_mae": float(np.min(mae_delta)),
                "max_delta_mae": float(np.max(mae_delta)),
                "fold_signflip_p_mae_sensitivity": _exact_fold_signflip_pvalue(mae_delta),
                "mean_delta_r2": float(np.mean(r2_delta)),
                "median_delta_r2": float(np.median(r2_delta)),
                "fold_signflip_p_r2_sensitivity": _exact_fold_signflip_pvalue(r2_delta),
                "note": "five folds overlap in training data; this is a consistency sensitivity analysis, not five independent biological replicates",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.output_dir / "fold_consistency_summary.csv", index=False)

    with open(args.output_dir / "fold_analysis_manifest.json", "w") as handle:
        json.dump(
            {
                **audit,
                "metric_space": "log10 target space",
                "primary_metric": "MAE",
                "within_fold_statistical_unit": "unique protein sequence",
                "fold_summary_unit": "held-out fold",
                "bootstrap_reps": args.bootstrap_reps,
                "permutation_reps": args.permutation_reps,
                "seed": args.seed,
                "note": "The five CV folds share training data. Pooled sequence-level statistics remain the primary inference; fold-level p-values and exact fold sign-flip values are supplementary robustness diagnostics.",
            },
            handle,
            indent=2,
        )
    print(fold_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"[saved] {args.output_dir}")

if __name__ == "__main__":
    main()
