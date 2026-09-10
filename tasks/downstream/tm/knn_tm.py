#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN 回归预测熔解温度 (单模型版)

用法:
  python knn_tm.py \\
      --emb-dir embeddings/protbert_cpt \\
      --train-csv data/train.csv \\
      --test-csv data/test.csv \\
      --model-name protbert_cpt \\
      --output-dir results/knn/protbert_cpt \\
      --ks 1 3 5 10 20 50
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["OMP_NUM_THREADS"] = "4"

# ============================================================================
# 数据加载
# ============================================================================

def load_embeddings_and_labels(emb_dir: str, split: str, csv_path: str):
    emb_path = Path(emb_dir) / f"{split}.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    embeddings_dict = torch.load(emb_path, map_location="cpu")
    df = pd.read_csv(csv_path)

    emb_mapping = {}
    for emb_key, emb_value in embeddings_dict.items():
        simple_id = emb_key.split("|")[0]
        emb_mapping[simple_id] = emb_value

    X_list, y_list, seq_ids = [], [], []
    for _, row in df.iterrows():
        sid = row["seq_id"]
        if sid in emb_mapping:
            X_list.append(emb_mapping[sid].numpy())
            y_list.append(row["meltPoint"])
            seq_ids.append(sid)

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  {split}: {len(X)} samples, embed_dim={X.shape[1]}")
    return X, y, seq_ids

# ============================================================================
# k-NN 回归
# ============================================================================

def run_knn(X_train, y_train, X_test, y_test, k, metric="cosine"):
    knn = KNeighborsRegressor(
        n_neighbors=k, metric=metric,
        weights="distance", n_jobs=8, algorithm="brute",
    )
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    pearson_r, _ = pearsonr(y_test, preds)
    spearman_r, _ = spearmanr(y_test, preds)

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'Pearson_r': float(pearson_r),
        'Spearman_r': float(spearman_r),
    }, preds

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="k-NN Tm Regression")
    parser.add_argument("--emb-dir", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs='+', default=[1])
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查已完成
    results_file = output_dir / "knn_results.json"
    if results_file.exists():
        print(f"Already done: {results_file}")
        with open(results_file) as f:
            r = json.load(f)
        b = r.get('best_test_metrics', {})
        print(f"  Best k={r.get('best_k')}: MAE={b.get('MAE', 0):.4f}")
        return

    print("=" * 70)
    print(f"k-NN Tm Regression: {args.model_name}")
    print("=" * 70)
    print(f"k values: {args.ks}")
    print(f"Metric:   {args.metric}")

    # 加载数据
    print("\nLoading data...")
    X_train, y_train, train_ids = load_embeddings_and_labels(
        args.emb_dir, "train", args.train_csv)
    X_test, y_test, test_ids = load_embeddings_and_labels(
        args.emb_dir, "test", args.test_csv)

    # 逐 k 评估
    all_k_results = {}
    best_mae = float('inf')
    best_k = args.ks[0]
    best_preds = None

    for k in args.ks:
        metrics, preds = run_knn(X_train, y_train, X_test, y_test, k, args.metric)
        print(f"  k={k:3d}: MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"R²={metrics['R2']:.4f}  Spearman={metrics['Spearman_r']:.4f}")

        all_k_results[f"k={k}"] = {'k': k, 'test_metrics': metrics}

        if metrics['MAE'] < best_mae:
            best_mae = metrics['MAE']
            best_k = k
            best_preds = preds

    print(f"\n  Best: k={best_k}, MAE={best_mae:.4f}°C")

    # 保存预测结果
    pd.DataFrame({
        'seq_id': test_ids,
        'true_tm': y_test,
        'pred_tm': best_preds,
        'error': best_preds - y_test,
    }).to_csv(output_dir / f"predictions_k{best_k}.csv", index=False)

    # 保存汇总
    output = {
        'model': args.model_name,
        'metric': args.metric,
        'best_k': best_k,
        'best_test_metrics': all_k_results[f"k={best_k}"]['test_metrics'],
        'all_k_results': all_k_results,
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved: {results_file}")

if __name__ == "__main__":
    main()