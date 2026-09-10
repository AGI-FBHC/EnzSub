#!/usr/bin/env python3
"""
Step 2a: XGBoost pH 回归 (训练 + 评估 + 保存模型)

用法:
  python xgboost_eval.py \
      --embedding-dirs esm2_base:/path/to/emb1 esm2_cpt:/path/to/emb2 \
      --train-csv data/processed/train.csv \
      --val-csv data/processed/val.csv \
      --test-csv data/processed/test.csv \
      --output-dir experiments/xgboost
"""

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from utils import EmbeddingStore

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

def fit_xgb_model(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  xgb_params: dict):
    """保持原有 XGBoost 训练逻辑，只封装为函数。"""
    from xgboost import XGBRegressor

    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

def summarize_cv_metrics(fold_metrics: List[dict]) -> dict:
    summary = {}
    metric_keys = ["mae", "rmse", "r2", "pearson_r", "spearman_r"]
    for key in metric_keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary

def run_train_cv(X_train: np.ndarray, y_train: np.ndarray,
                 xgb_params: dict, use_scaler: bool,
                 seed: int, n_splits: int = 5) -> Tuple[List[dict], dict]:
    """
    只在训练集内部做 K-fold CV。

    固定的 val/test split 不参与 CV；它们仍然只用于主流程的 early stopping 与最终评估。
    """
    if n_splits <= 1:
        return [], {}
    if n_splits > len(X_train):
        raise ValueError(f"cv_folds={n_splits} is larger than train size={len(X_train)}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        scaler = None
        if use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        model = fit_xgb_model(X_tr, y_tr, X_va, y_va, xgb_params)
        pred_va = model.predict(X_va)
        metrics = compute_metrics(y_va, pred_va)
        metrics["fold"] = fold
        metrics["n_train"] = int(len(tr_idx))
        metrics["n_val"] = int(len(va_idx))
        metrics["best_iteration"] = int(getattr(model, "best_iteration", -1))
        fold_metrics.append(metrics)

        log.info(f"    CV fold {fold}/{n_splits}: "
                 f"MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}  "
                 f"SCC={metrics['spearman_r']:.4f}")

    return fold_metrics, summarize_cv_metrics(fold_metrics)

def main():
    parser = argparse.ArgumentParser(description="XGBoost pH regression")
    parser.add_argument("--embedding-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)

    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--colsample-bylevel", type=float, default=0.8)
    parser.add_argument("--colsample-bynode", type=float, default=0.8)
    parser.add_argument("--tree-method", type=str, default="hist",
                        choices=("auto", "exact", "approx", "hist"))
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="K-fold CV within train split only. Use 5 for train-only 5-fold CV; 0 disables it.")
    parser.add_argument("--use-scaler", action="store_true", default=True)
    parser.add_argument("--no-scaler", dest="use_scaler", action="store_false")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-dir", type=str, default="experiments/xgboost")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional stable run directory name under output-dir")
    args = parser.parse_args()
    if args.early_stopping_rounds < 0:
        parser.error("--early-stopping-rounds must be >= 0; use 0 to disable it")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be >= 1")

    emb_dirs = parse_kv_pairs(args.embedding_dirs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(output_dir, "xgboost.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "colsample_bylevel": args.colsample_bylevel,
        "colsample_bynode": args.colsample_bynode,
        "tree_method": args.tree_method,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }
    if args.early_stopping_rounds > 0:
        xgb_params["early_stopping_rounds"] = args.early_stopping_rounds

    log.info("=" * 60)
    log.info("XGBoost pH Regression")
    log.info("=" * 60)
    log.info(f"Models: {list(emb_dirs.keys())}")
    log.info(f"Params: {json.dumps(xgb_params, sort_keys=True)}")
    log.info(
        "Early stopping: %s",
        args.early_stopping_rounds if args.early_stopping_rounds > 0 else "disabled",
    )
    log.info(f"Train CV folds: {args.cv_folds if args.cv_folds and args.cv_folds > 1 else 0}")

    all_results = []

    for emb_name, emb_dir in emb_dirs.items():
        log.info(f"\n{'='*60}")
        log.info(f"Model: {emb_name}")
        log.info(f"  dir: {emb_dir}")

        store = EmbeddingStore(emb_dir)

        # 加载数据: 通过 CSV 文件名自动推断 split
        X_train, y_train, _ = store.load_with_labels(args.train_csv)
        X_val, y_val, _ = store.load_with_labels(args.val_csv)
        X_test, y_test, test_ids = store.load_with_labels(args.test_csv)

        if X_train is None or X_test is None:
            log.error(f"  Missing data, skipping {emb_name}")
            continue

        log.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        model_dir = os.path.join(output_dir, "models", emb_name)
        os.makedirs(model_dir, exist_ok=True)

        # 训练集内部 K-fold CV；不改变固定 val/test 主流程
        cv_folds, cv_summary = [], {}
        if args.cv_folds and args.cv_folds > 1:
            log.info(f"  Train-only {args.cv_folds}-fold CV...")
            cv_folds, cv_summary = run_train_cv(
                X_train, y_train,
                xgb_params=xgb_params,
                use_scaler=args.use_scaler,
                seed=args.seed,
                n_splits=args.cv_folds,
            )
            log.info(f"  CV MAE={cv_summary['mae_mean']:.4f}+/-{cv_summary['mae_std']:.4f}  "
                     f"R²={cv_summary['r2_mean']:.4f}+/-{cv_summary['r2_std']:.4f}")
            with open(os.path.join(model_dir, "cv_train_5fold.json"), "w") as f:
                json.dump({
                    "cv_folds": args.cv_folds,
                    "seed": args.seed,
                    "folds": cv_folds,
                    "summary": cv_summary,
                }, f, indent=2)

        # Scaler
        scaler = None
        X_tr, X_va, X_te = X_train, X_val, X_test
        if args.use_scaler:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_va = scaler.transform(X_val)
            X_te = scaler.transform(X_test)

        # 训练
        t0 = time.time()
        model = fit_xgb_model(X_tr, y_train, X_va, y_val, xgb_params)
        train_time = time.time() - t0

        # 评估
        train_m = compute_metrics(y_train, model.predict(X_tr))
        val_m = compute_metrics(y_val, model.predict(X_va))
        test_m = compute_metrics(y_test, model.predict(X_te))

        log.info(f"  Train MAE={train_m['mae']:.4f}  "
                 f"Val MAE={val_m['mae']:.4f}  "
                 f"Test MAE={test_m['mae']:.4f}")
        log.info(f"  Test R²={test_m['r2']:.4f}  "
                 f"PCC={test_m['pearson_r']:.4f}  "
                 f"SCC={test_m['spearman_r']:.4f}  "
                 f"({train_time:.1f}s)")

        # 保存模型
        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump({"xgb": model, "scaler": scaler, "params": xgb_params}, f)

        # 保存预测
        pred_dir = os.path.join(output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        y_pred = model.predict(X_te)
        pd.DataFrame({
            "protein_id": test_ids,
            "true_pH": y_test,
            "pred_pH": y_pred,
            "error": y_pred - y_test,
        }).to_csv(os.path.join(pred_dir, f"{emb_name}.csv"), index=False)

        result_row = {
            "embedding": emb_name,
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "test_r2": test_m["r2"],
            "test_pearson_r": test_m["pearson_r"],
            "test_spearman_r": test_m["spearman_r"],
            "train_mae": train_m["mae"],
            "val_mae": val_m["mae"],
            "training_time": train_time,
        }
        if cv_summary:
            result_row.update({
                "train_cv_mae_mean": cv_summary["mae_mean"],
                "train_cv_mae_std": cv_summary["mae_std"],
                "train_cv_rmse_mean": cv_summary["rmse_mean"],
                "train_cv_rmse_std": cv_summary["rmse_std"],
                "train_cv_r2_mean": cv_summary["r2_mean"],
                "train_cv_r2_std": cv_summary["r2_std"],
                "train_cv_pearson_r_mean": cv_summary["pearson_r_mean"],
                "train_cv_pearson_r_std": cv_summary["pearson_r_std"],
                "train_cv_spearman_r_mean": cv_summary["spearman_r_mean"],
                "train_cv_spearman_r_std": cv_summary["spearman_r_std"],
            })
        all_results.append(result_row)

    if not all_results:
        log.error("No results")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    log.info(f"\n{'='*60}")
    log.info("Summary")
    log.info(f"{'='*60}")
    for _, r in df.sort_values("test_mae").iterrows():
        cv_text = ""
        if "train_cv_mae_mean" in r and not pd.isna(r["train_cv_mae_mean"]):
            cv_text = f"  CV={r['train_cv_mae_mean']:.4f}+/-{r['train_cv_mae_std']:.4f}"
        log.info(f"  {r['embedding']:30s}  MAE={r['test_mae']:.4f}{cv_text}  "
                 f"R²={r['test_r2']:.4f}  SCC={r['test_spearman_r']:.4f}")

    json.dump({"embeddings": emb_dirs, "xgb_params": xgb_params,
               "use_scaler": args.use_scaler, "seed": args.seed,
               "cv_folds": args.cv_folds,
               "timestamp": timestamp, "run_name": run_name},
              open(os.path.join(output_dir, "config.json"), "w"), indent=2)

    log.info(f"\nResults: {output_dir}")

if __name__ == "__main__":
    main()
