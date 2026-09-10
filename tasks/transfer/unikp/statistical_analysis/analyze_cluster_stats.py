#!/usr/bin/env python3
"""Paired uncertainty analysis for the fixed MMseqs2 cluster evaluation.

The script reads the already generated cluster-isolated prediction files. It
does not retrain models or regenerate embeddings. The statistical unit is an
MMseqs2 cluster: bootstrap resamples clusters with replacement, and the paired
permutation test swaps ProtT5 and EnzSub labels for complete clusters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TASKS = ("kcat", "km", "kcat_km")
BACKBONES = (
    "esm2_650m_cpt_sub_epoch15",
    "protbert_bfd_cpt_sub_epoch15",
    "esm2_3b_cpt_sub_epoch5",
)
REQUIRED_PREDICTION_COLUMNS = {
    "task",
    "representation",
    "split",
    "sample_index",
    "fold",
    "label_log10",
    "pred_log10",
}
METRICS = ("mae", "rmse", "r2", "pearson")

def _holm_adjust(pvalues: Iterable[float]) -> List[float]:
    values = np.asarray(list(pvalues), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result.tolist()
    order = np.argsort(values[finite])
    ordered = values[finite][order]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered)
    adjusted = np.minimum(adjusted, 1.0)
    target = np.flatnonzero(finite)[order]
    result[target] = adjusted
    return result.tolist()

def _metric_values(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)
    err = preds - labels
    abs_err = np.abs(err)
    sq_err = err ** 2
    ss_tot = float(np.sum((labels - np.mean(labels)) ** 2))
    if np.std(labels) <= 1e-12 or np.std(preds) <= 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(labels, preds)[0, 1])
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "r2": float(1.0 - np.sum(sq_err) / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": pearson,
    }

def _metric_values_from_summaries(
    n: np.ndarray,
    y_sum: np.ndarray,
    y_sq_sum: np.ndarray,
    pred_sum: np.ndarray,
    pred_sq_sum: np.ndarray,
    y_pred_sum: np.ndarray,
    abs_err_sum: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Calculate metrics from sufficient statistics, one row per resample."""

    n = np.asarray(n, dtype=float)
    y_sum = np.asarray(y_sum, dtype=float)
    y_sq_sum = np.asarray(y_sq_sum, dtype=float)
    pred_sum = np.asarray(pred_sum, dtype=float)
    pred_sq_sum = np.asarray(pred_sq_sum, dtype=float)
    y_pred_sum = np.asarray(y_pred_sum, dtype=float)
    abs_err_sum = np.asarray(abs_err_sum, dtype=float)
    mse = (y_sq_sum - 2.0 * y_pred_sum + pred_sq_sum) / n
    y_mean = y_sum / n
    pred_mean = pred_sum / n
    ss_tot = y_sq_sum - (y_sum * y_sum) / n
    covariance = y_pred_sum / n - y_mean * pred_mean
    y_variance = y_sq_sum / n - y_mean * y_mean
    pred_variance = pred_sq_sum / n - pred_mean * pred_mean
    denominator = np.sqrt(np.maximum(y_variance, 0.0) * np.maximum(pred_variance, 0.0))
    pearson = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
    sse = y_sq_sum - 2.0 * y_pred_sum + pred_sq_sum
    r2 = np.full_like(ss_tot, np.nan)
    valid_ss_tot = ss_tot > 1e-12
    r2[valid_ss_tot] = 1.0 - sse[valid_ss_tot] / ss_tot[valid_ss_tot]
    return {
        "mae": abs_err_sum / n,
        "rmse": np.sqrt(np.maximum(mse, 0.0)),
        "r2": r2,
        "pearson": pearson,
    }

def _summary_arrays(labels: np.ndarray, preds: np.ndarray, cluster_ids: np.ndarray) -> Dict[str, np.ndarray]:
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    counts = np.bincount(inverse, minlength=n_clusters).astype(float)
    y_sum = np.bincount(inverse, weights=labels, minlength=n_clusters)
    y_sq_sum = np.bincount(inverse, weights=labels ** 2, minlength=n_clusters)
    pred_sum = np.bincount(inverse, weights=preds, minlength=n_clusters)
    pred_sq_sum = np.bincount(inverse, weights=preds ** 2, minlength=n_clusters)
    y_pred_sum = np.bincount(inverse, weights=labels * preds, minlength=n_clusters)
    abs_err_sum = np.bincount(inverse, weights=np.abs(preds - labels), minlength=n_clusters)
    return {
        "cluster_ids": unique_clusters.astype(str),
        "n": counts,
        "y_sum": y_sum,
        "y_sq_sum": y_sq_sum,
        "pred_sum": pred_sum,
        "pred_sq_sum": pred_sq_sum,
        "y_pred_sum": y_pred_sum,
        "abs_err_sum": abs_err_sum,
    }

def _resampled_metrics(summary: Dict[str, np.ndarray], draws: np.ndarray) -> Dict[str, np.ndarray]:
    arrays = {key: summary[key][draws].sum(axis=1) for key in ("n", "y_sum", "y_sq_sum", "pred_sum", "pred_sq_sum", "y_pred_sum", "abs_err_sum")}
    return _metric_values_from_summaries(
        arrays["n"],
        arrays["y_sum"],
        arrays["y_sq_sum"],
        arrays["pred_sum"],
        arrays["pred_sq_sum"],
        arrays["y_pred_sum"],
        arrays["abs_err_sum"],
    )

def _bootstrap_deltas(
    base_summary: Dict[str, np.ndarray],
    enz_summary: Dict[str, np.ndarray],
    rng: np.random.Generator,
    reps: int,
    batch_size: int = 100,
) -> Dict[str, np.ndarray]:
    n_clusters = len(base_summary["cluster_ids"])
    output = {metric: np.empty(reps, dtype=float) for metric in METRICS}
    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        base = _resampled_metrics(base_summary, draws)
        enz = _resampled_metrics(enz_summary, draws)
        for metric in METRICS:
            output[metric][start:stop] = enz[metric] - base[metric]
    return output

def _permutation_pvalues(
    base_summary: Dict[str, np.ndarray],
    enz_summary: Dict[str, np.ndarray],
    observed_delta: Dict[str, float],
    rng: np.random.Generator,
    reps: int,
    batch_size: int = 100,
) -> Dict[str, float]:
    n_clusters = len(base_summary["cluster_ids"])
    exceed = {metric: 0 for metric in METRICS}
    done = 0
    for start in range(0, reps, batch_size):
        stop = min(start + batch_size, reps)
        swap = rng.integers(0, 2, size=(stop - start, n_clusters)).astype(bool)
        base = {}
        enz = {}
        for key in ("n", "y_sum", "y_sq_sum"):
            base[key] = np.broadcast_to(base_summary[key], swap.shape)
            enz[key] = np.broadcast_to(enz_summary[key], swap.shape)
        for key in ("pred_sum", "pred_sq_sum", "y_pred_sum", "abs_err_sum"):
            b = np.broadcast_to(base_summary[key], swap.shape)
            e = np.broadcast_to(enz_summary[key], swap.shape)
            base[key] = np.where(swap, e, b).sum(axis=1)
            enz[key] = np.where(swap, b, e).sum(axis=1)
        # Shared label summaries are not representation-dependent.
        y_n = base_summary["n"]
        y_sum = base_summary["y_sum"]
        y_sq_sum = base_summary["y_sq_sum"]
        n = y_n.sum()
        y_total = y_sum.sum()
        y_sq_total = y_sq_sum.sum()
        base_metrics = _metric_values_from_summaries(
            np.full(stop - start, n),
            np.full(stop - start, y_total),
            np.full(stop - start, y_sq_total),
            base["pred_sum"],
            base["pred_sq_sum"],
            base["y_pred_sum"],
            base["abs_err_sum"],
        )
        enz_metrics = _metric_values_from_summaries(
            np.full(stop - start, n),
            np.full(stop - start, y_total),
            np.full(stop - start, y_sq_total),
            enz["pred_sum"],
            enz["pred_sq_sum"],
            enz["y_pred_sum"],
            enz["abs_err_sum"],
        )
        for metric in METRICS:
            null_delta = enz_metrics[metric] - base_metrics[metric]
            finite = np.isfinite(null_delta)
            exceed[metric] += int(np.sum(np.abs(null_delta[finite]) >= abs(observed_delta[metric])))
        done += stop - start
    return {metric: float((exceed[metric] + 1) / (done + 1)) for metric in METRICS}

def _load_pair(
    task: str,
    backbone: str,
    prediction_path: Path,
    cluster_dir: Path,
    details_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    predictions = pd.read_csv(prediction_path)
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"{prediction_path}: missing columns {sorted(missing)}")
    if set(predictions["representation"].astype(str)) != {"ProtT5", "EnzSub"}:
        raise ValueError(f"{prediction_path}: expected ProtT5 and EnzSub representations")
    if predictions["split"].astype(str).nunique() != 1 or predictions["split"].iloc[0] != "cluster_40id_5fold":
        raise ValueError(f"{prediction_path}: unexpected split labels")
    predictions["sample_index"] = pd.to_numeric(predictions["sample_index"], errors="raise").astype(int)
    predictions["fold"] = pd.to_numeric(predictions["fold"], errors="raise").astype(int)
    predictions["label_log10"] = pd.to_numeric(predictions["label_log10"], errors="raise")
    predictions["pred_log10"] = pd.to_numeric(predictions["pred_log10"], errors="raise")
    if predictions.duplicated(["representation", "sample_index"]).any():
        raise ValueError(f"{prediction_path}: duplicated representation/sample_index pair")

    sample_folds = pd.read_csv(cluster_dir / "sample_folds.csv")
    sequence_folds = pd.read_csv(cluster_dir / "sequence_folds.csv")
    required_sample = {"sample_index", "sequence_id", "fold"}
    required_sequence = {"sequence_id", "cluster_id", "fold"}
    if not required_sample.issubset(sample_folds.columns):
        raise ValueError(f"{cluster_dir}/sample_folds.csv: missing {sorted(required_sample - set(sample_folds.columns))}")
    if not required_sequence.issubset(sequence_folds.columns):
        raise ValueError(f"{cluster_dir}/sequence_folds.csv: missing {sorted(required_sequence - set(sequence_folds.columns))}")
    if sample_folds["sample_index"].duplicated().any() or sequence_folds["sequence_id"].duplicated().any():
        raise ValueError(f"{cluster_dir}: duplicated sample or sequence key")
    if sequence_folds.groupby("cluster_id")["fold"].nunique().max() != 1:
        raise ValueError(f"{cluster_dir}: a cluster is assigned to more than one fold")

    sample_map = sample_folds[["sample_index", "sequence_id", "fold"]].rename(columns={"fold": "manifest_fold"})
    sequence_map = sequence_folds[["sequence_id", "cluster_id", "fold"]].rename(columns={"fold": "sequence_fold"})
    sample_map = sample_map.merge(sequence_map, on="sequence_id", how="left", validate="many_to_one")
    if sample_map["cluster_id"].isna().any():
        raise ValueError(f"{cluster_dir}: sample without cluster assignment")
    if not np.array_equal(sample_map["manifest_fold"].to_numpy(), sample_map["sequence_fold"].to_numpy()):
        raise ValueError(f"{cluster_dir}: sample and sequence fold assignments disagree")

    base = predictions[predictions["representation"] == "ProtT5"].copy()
    enz = predictions[predictions["representation"] == "EnzSub"].copy()
    keys = ["sample_index"]
    paired = base.merge(enz, on=keys, how="outer", suffixes=("_base", "_enz"), indicator=True)
    if not (paired["_merge"] == "both").all():
        raise ValueError(f"{prediction_path}: ProtT5 and EnzSub rows are not paired")
    if not np.allclose(paired["label_log10_base"], paired["label_log10_enz"], equal_nan=False):
        raise ValueError(f"{prediction_path}: paired labels differ")
    paired = paired[["sample_index", "fold_base", "label_log10_base", "pred_log10_base", "pred_log10_enz"]]
    paired = paired.rename(columns={"fold_base": "fold", "label_log10_base": "label_log10"})
    paired = paired.merge(sample_map, on="sample_index", how="left", validate="one_to_one")
    if paired["cluster_id"].isna().any() or not np.array_equal(paired["fold"].to_numpy(), paired["manifest_fold"].to_numpy()):
        raise ValueError(f"{prediction_path}: prediction rows disagree with cluster fold manifest")
    paired["base_abs_err"] = np.abs(paired["pred_log10_base"] - paired["label_log10"])
    paired["enzsub_abs_err"] = np.abs(paired["pred_log10_enz"] - paired["label_log10"])
    paired["base_sq_err"] = (paired["pred_log10_base"] - paired["label_log10"]) ** 2
    paired["enzsub_sq_err"] = (paired["pred_log10_enz"] - paired["label_log10"]) ** 2
    paired["task"] = task
    paired["backbone"] = backbone
    details_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(details_path, index=False)

    audit = {
        "task": task,
        "backbone": backbone,
        "prediction_file": str(prediction_path),
        "n_records": int(len(paired)),
        "n_unique_sequences": int(paired["sequence_id"].nunique()),
        "n_clusters": int(paired["cluster_id"].nunique()),
        "n_folds": int(paired["fold"].nunique()),
        "cluster_record_count_min": int(paired.groupby("cluster_id").size().min()),
        "cluster_record_count_max": int(paired.groupby("cluster_id").size().max()),
    }
    return paired, audit

def analyse_pair(
    task: str,
    backbone: str,
    prediction_path: Path,
    cluster_dir: Path,
    output_dir: Path,
    rng: np.random.Generator,
    bootstrap_reps: int,
    permutation_reps: int,
) -> Tuple[Dict[str, object], Dict[str, object], pd.DataFrame, pd.DataFrame]:
    paired, audit = _load_pair(
        task,
        backbone,
        prediction_path,
        cluster_dir,
        output_dir / "paired_error_details" / task / f"{backbone}.csv",
    )
    labels = paired["label_log10"].to_numpy(float)
    base_pred = paired["pred_log10_base"].to_numpy(float)
    enz_pred = paired["pred_log10_enz"].to_numpy(float)
    base_metrics = _metric_values(labels, base_pred)
    enz_metrics = _metric_values(labels, enz_pred)
    observed_delta = {metric: enz_metrics[metric] - base_metrics[metric] for metric in METRICS}
    base_summary = _summary_arrays(labels, base_pred, paired["cluster_id"].to_numpy(str))
    enz_summary = _summary_arrays(labels, enz_pred, paired["cluster_id"].to_numpy(str))
    if not np.array_equal(base_summary["cluster_ids"], enz_summary["cluster_ids"]):
        raise ValueError(f"{prediction_path}: cluster summaries are not aligned")

    bootstrap = _bootstrap_deltas(base_summary, enz_summary, rng, bootstrap_reps)
    permutation_p = _permutation_pvalues(base_summary, enz_summary, observed_delta, rng, permutation_reps)
    row: Dict[str, object] = {
        **audit,
        "split": "cluster_40id_5fold",
        "metric_space": "log10 target space",
        "statistical_unit": "MMseqs2 cluster",
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps": permutation_reps,
    }
    for metric in METRICS:
        row[f"base_{metric}"] = base_metrics[metric]
        row[f"enzsub_{metric}"] = enz_metrics[metric]
        row[f"delta_{metric}_enzsub_minus_base"] = observed_delta[metric]
        row[f"delta_{metric}_ci95_low"] = float(np.nanquantile(bootstrap[metric], 0.025))
        row[f"delta_{metric}_ci95_high"] = float(np.nanquantile(bootstrap[metric], 0.975))
        row[f"{metric}_p_two_sided_cluster_swap"] = permutation_p[metric]

    bootstrap_rows = pd.DataFrame({"bootstrap_rep": np.arange(bootstrap_reps), **bootstrap})
    bootstrap_rows.insert(0, "backbone", backbone)
    bootstrap_rows.insert(0, "task", task)
    return row, audit, bootstrap_rows, pd.DataFrame([row])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--permutation-reps", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.bootstrap_reps < 100 or args.permutation_reps < 100:
        raise ValueError("use at least 100 bootstrap and permutation repetitions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    result_rows: List[Dict[str, object]] = []
    audits: List[Dict[str, object]] = []
    bootstrap_rows: List[pd.DataFrame] = []
    for task in TASKS:
        for backbone in BACKBONES:
            prediction_path = args.prediction_root / task / backbone / f"{task}_cluster_predictions.csv"
            cluster_dir = args.cluster_root / task
            if not prediction_path.exists():
                raise FileNotFoundError(prediction_path)
            row, audit, bootstrap, _ = analyse_pair(
                task,
                backbone,
                prediction_path,
                cluster_dir,
                args.output_dir,
                rng,
                args.bootstrap_reps,
                args.permutation_reps,
            )
            result_rows.append(row)
            audits.append(audit)
            bootstrap_rows.append(bootstrap)

    result = pd.DataFrame(result_rows)
    for metric in METRICS:
        p_column = f"{metric}_p_two_sided_cluster_swap"
        result[f"{metric}_p_holm"] = _holm_adjust(result[p_column].tolist())
    result_path = args.output_dir / "paired_statistics_cluster.csv"
    result.to_csv(result_path, index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(args.output_dir / "cluster_bootstrap_deltas.csv", index=False)
    with open(args.output_dir / "analysis_manifest.json", "w") as handle:
        json.dump(
            {
                "analysis": "paired MMseqs2-cluster-isolated UniKP statistics",
                "metric_space": "log10 target space",
                "primary_metric": "MAE",
                "statistical_unit": "MMseqs2 cluster at 40% identity and 80% coverage",
                "bootstrap_method": "resample complete clusters with replacement",
                "permutation_method": "swap ProtT5 and EnzSub labels for complete clusters",
                "bootstrap_reps": args.bootstrap_reps,
                "permutation_reps": args.permutation_reps,
                "seed": args.seed,
                "source_prediction_root": str(args.prediction_root),
                "source_cluster_root": str(args.cluster_root),
                "audits": audits,
                "note": "Predictions are out-of-fold under fixed cluster folds; p-values quantify paired representation differences conditional on these predictions and do not represent independent retraining replicates.",
            },
            handle,
            indent=2,
        )
    print(result.to_string(index=False))
    print(f"[saved] {result_path}")

if __name__ == "__main__":
    main()
