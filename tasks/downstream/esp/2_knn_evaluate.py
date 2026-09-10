#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_knn_evaluate.py  —  k-NN Enzyme-Substrate Binding Prediction

Strategy:
  1. Concatenate enzyme ESM embedding (1280d) + substrate ChemBERTa embedding (768d)
     → joint representation (2048d) for each (enzyme, substrate) pair
  2. L2-normalize the joint vector
  3. Find k nearest neighbors in training set (cosine similarity in joint space)
  4. Predict binding probability = fraction of neighbors with Binding=1

No CV — single train→test inference. Runs multiple k values in one pass.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score,
    f1_score, precision_score, recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos

    out = {
        "n_samples": n, "n_positive": n_pos, "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if n_pos == 0 or n_neg == 0:
        out.update({"roc_auc": float("nan"), "mcc": float("nan"),
                     "f1": float("nan"), "precision": float("nan"),
                     "recall": float("nan"), "skipped": True})
        return out

    out.update({
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "skipped": False,
    })
    return out

# =============================================================================
# Feature extraction
# =============================================================================
def build_joint_features(df: pd.DataFrame):
    """
    Concatenate ESM (1280d) + ChemBERTa (768d) → joint (2048d).
    Returns X_joint, y.
    """
    for col in ["enzyme_vector", "ChemBERTa_vector", "Binding"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: '{col}'")

    X_enz = np.stack(df["enzyme_vector"].values).astype(np.float32)
    X_sub = np.stack(df["ChemBERTa_vector"].values).astype(np.float32)
    X_joint = np.concatenate([X_enz, X_sub], axis=1)
    y = df["Binding"].to_numpy().astype(np.float32)

    return X_joint, y

def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

# =============================================================================
# Core k-NN: cosine similarity in joint space, vote by label ratio
# =============================================================================
def knn_predict_multi_k(
    query: np.ndarray,      # (N_q, D)
    train: np.ndarray,      # (N_tr, D)
    train_y: np.ndarray,    # (N_tr,)
    k_values: List[int],
    batch_size: int = 256,
) -> Dict[int, np.ndarray]:
    """
    For each query, find k nearest neighbors in train (cosine similarity),
    predict P(binding) = mean(neighbor labels).
    Runs all k values in one pass by precomputing top-k_max neighbors.
    """
    N_q = query.shape[0]
    k_max = max(k_values)

    query_norm = normalize_rows(query)
    train_norm = normalize_rows(train)

    all_scores = {k: np.zeros(N_q, dtype=np.float32) for k in k_values}

    for start in range(0, N_q, batch_size):
        end = min(start + batch_size, N_q)
        B = end - start

        # Cosine similarity in joint space: (B, N_tr)
        sim = query_norm[start:end] @ train_norm.T

        # Top-k_max indices, sorted descending
        if train.shape[0] > k_max * 4:
            top_idx = np.argpartition(-sim, k_max, axis=1)[:, :k_max]
            for b in range(B):
                order = np.argsort(-sim[b, top_idx[b]])
                top_idx[b] = top_idx[b][order]
        else:
            top_idx = np.argsort(-sim, axis=1)[:, :k_max]

        for b in range(B):
            nbr_labels = train_y[top_idx[b]]  # (k_max,)
            for k in k_values:
                all_scores[k][start + b] = nbr_labels[:k].mean()

        if (end % 2000 == 0) or (end == N_q):
            logging.info(f"  Processed {end}/{N_q} queries")

    return all_scores

# =============================================================================
# Evaluation with optional similarity bins
# =============================================================================
def eval_overall(y_true, y_prob):
    return {"overall": compute_metrics(y_true, y_prob)}

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="k-NN ESP binding prediction (joint space, multi-k, no CV)")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--test-name", type=str, default="ID_Test")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    logging.info(f"Loading train: {args.train_file}")
    df_train = pd.read_pickle(args.train_file).dropna(subset=["enzyme_vector", "Binding"])
    X_tr, y_tr = build_joint_features(df_train)
    logging.info(f"  Train: n={len(y_tr)}, pos={int(y_tr.sum())}, neg={int(len(y_tr)-y_tr.sum())}")
    logging.info(f"  Joint feature dim: {X_tr.shape[1]} (ESM 1280 + ChemBERTa 768)")

    logging.info(f"Loading test: {args.test_file}")
    df_test = pd.read_pickle(args.test_file).dropna(subset=["enzyme_vector", "Binding"])
    X_te, y_te = build_joint_features(df_test)
    logging.info(f"  Test: n={len(y_te)}, pos={int(y_te.sum())}, neg={int(len(y_te)-y_te.sum())}")

    # ---- Run k-NN for all k values ----
    logging.info(f"\nRunning k-NN (joint space) for k={args.k_values} ...")
    scores_by_k = knn_predict_multi_k(
        query=X_te,
        train=X_tr,
        train_y=y_tr,
        k_values=args.k_values,
        batch_size=args.batch_size,
    )

    # ---- Evaluate each k ----
    results = {"config": {
        "method": "knn_joint_space",
        "feature": "enzyme + ChemBERTa_768d (concat)",
        "similarity": "cosine",
        "voting": "label_ratio",
        "train_file": args.train_file,
        "test_file": args.test_file,
        "k_values": args.k_values,
    }}

    all_k_results = {}
    for k in args.k_values:
        probs = scores_by_k[k]
        res = eval_overall(y_te, probs)
        all_k_results[k] = res

    results["results_by_k"] = {str(k): v for k, v in all_k_results.items()}

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print(f"k-NN ESP (joint space)  |  test={args.test_name}")
    print(f"{'='*80}")
    print(f"{'k':>5} {'ACC':>8} {'AUC':>8} {'MCC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'n':>8}")
    print("-" * 72)
    for k in args.k_values:
        o = all_k_results[k]["overall"]
        print(f"{k:>5} {o['accuracy']:>8.4f} {o['roc_auc']:>8.4f} {o['mcc']:>8.4f} "
              f"{o['f1']:>8.4f} {o['precision']:>8.4f} {o['recall']:>8.4f} {o['n_samples']:>8d}")
    print("=" * 80)

    # ---- Save ----
    def convert_nan(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_nan(v) for v in obj]
        return obj

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(convert_nan(results), f, indent=2)
    logging.info(f"Saved: {metrics_path}")

if __name__ == "__main__":
    main()