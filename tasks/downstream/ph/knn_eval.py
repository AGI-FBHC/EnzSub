#!/usr/bin/env python3
"""
Step 2b: k-NN pH 回归评估

用法:
  python knn_eval.py \
      --embedding-dirs esm2_base:/path/to/emb1 esm2_cpt:/path/to/emb2 \
      --train-csv data/processed/train.csv \
      --test-csv data/processed/test.csv \
      --k 1 3 5 10 20 \
      --metric cosine
"""

import argparse
import json
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from utils import EmbeddingStore

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_r": float(spearmanr(y_true, y_pred)[0]),
    }

def parse_kv_pairs(pairs: List[str]) -> Dict[str, str]:
    result = {}
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Expected name:path, got: {pair}")
        name, path = pair.split(":", 1)
        result[name.strip()] = path.strip()
    return result

def main():
    parser = argparse.ArgumentParser(description="k-NN pH regression")
    parser.add_argument("--embedding-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=["cosine", "euclidean", "manhattan"])
    parser.add_argument("--weights", type=str, default="distance",
                        choices=["distance", "uniform"])
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="experiments/knn")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    emb_dirs = parse_kv_pairs(args.embedding_dirs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(output_dir, "knn.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("k-NN pH Regression")
    log.info("=" * 60)
    log.info(f"Models: {list(emb_dirs.keys())}")
    log.info(f"k: {args.k}, metric: {args.metric}, "
             f"weights: {args.weights}, normalize: {args.normalize}")

    all_results = []

    for emb_name, emb_dir in emb_dirs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {emb_name}")
        log.info(f"  dir: {emb_dir}")

        store = EmbeddingStore(emb_dir)

        # 精确加载: train.csv → train_all.pt, test.csv → test_all.pt
        X_train, y_train, _ = store.load_with_labels(
            args.train_csv, normalize=args.normalize)
        X_test, y_test, test_ids = store.load_with_labels(
            args.test_csv, normalize=args.normalize)

        if X_train is None or X_test is None:
            log.warning(f"  Missing data, skipping")
            continue

        log.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

        best_mae = float("inf")
        best_k = args.k[0]
        best_preds = None

        for k in args.k:
            knn = KNeighborsRegressor(
                n_neighbors=k, metric=args.metric,
                weights=args.weights, n_jobs=8, algorithm="brute",
            )
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            m = compute_metrics(y_test, y_pred)

            log.info(f"  k={k:3d}: MAE={m['mae']:.4f}  "
                     f"R²={m['r2']:.4f}  Spearman={m['spearman_r']:.4f}")

            all_results.append({"embedding": emb_name, "k": k, **m})

            if m["mae"] < best_mae:
                best_mae = m["mae"]
                best_k = k
                best_preds = y_pred

        if args.save_predictions and best_preds is not None:
            pred_dir = os.path.join(output_dir, "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            pd.DataFrame({
                "protein_id": test_ids,
                "true_pH": y_test,
                "pred_pH": best_preds,
                "error": best_preds - y_test,
            }).to_csv(os.path.join(pred_dir, f"{emb_name}_k{best_k}.csv"),
                      index=False)

        log.info(f"  Best: k={best_k}, MAE={best_mae:.4f}")

    if not all_results:
        log.error("No results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    log.info(f"\n{'='*60}")
    log.info("Summary (best k per model)")
    log.info(f"{'='*60}")
    for emb_name in emb_dirs:
        sub = df[df["embedding"] == emb_name]
        if sub.empty:
            continue
        best = sub.loc[sub["mae"].idxmin()]
        log.info(f"  {emb_name:30s}  k={int(best['k']):3d}  "
                 f"MAE={best['mae']:.4f}  R²={best['r2']:.4f}  "
                 f"Spearman={best['spearman_r']:.4f}")

    json.dump({"embeddings": emb_dirs, "k_values": args.k,
               "metric": args.metric, "weights": args.weights,
               "normalize": args.normalize, "timestamp": timestamp},
              open(os.path.join(output_dir, "config.json"), "w"), indent=2)

    log.info(f"\nResults: {output_dir}")

if __name__ == "__main__":
    main()