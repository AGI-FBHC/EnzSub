#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 熔解温度回归 (解耦版，修正版)

不依赖 tm_experiment_config.py, 所有路径通过命令行传入。

主要修正:
  1. 修复 XGBRegressor 中 objective / eval_metric 重复传参导致的报错。
  2. 将 early_stopping_rounds 从 500 调整为默认 100，并支持命令行修改。
  3. 增强 embedding / label 加载的鲁棒性，检查空匹配、缺失列、缺失标签和异常维度。
  4. CV 输出文件名和 metrics key 根据 --cv-folds 动态命名，不再固定写死 5fold。
  5. 保存完整训练参数，包括 early_stopping_rounds。

用法示例:
  python train_xgboost_tm_fixed.py \
      --emb-dir embeddings/protbert_cpt \
      --train-csv data/train.csv \
      --val-csv data/val.csv \
      --test-csv data/test.csv \
      --model-name protbert_cpt \
      --output-dir results/protbert_cpt/seed_2025 \
      --seed 2025 \
      --cv-folds 5

关闭 CV:
  python train_xgboost_tm_fixed.py \
      --emb-dir embeddings/protbert_cpt \
      --train-csv data/train.csv \
      --val-csv data/val.csv \
      --test-csv data/test.csv \
      --model-name protbert_cpt \
      --output-dir results/protbert_cpt/seed_2025 \
      --seed 2025 \
      --cv-folds 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ============================================================================
# 模型参数
# ============================================================================

FIXED_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1,
    "gamma": 0.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_jobs": 8,
    "tree_method": "hist",
}

# ============================================================================
# 数据加载
# ============================================================================

def _to_1d_numpy_embedding(emb_value, emb_key: str):
    """将单条 embedding 转成 1D np.ndarray。"""
    if isinstance(emb_value, torch.Tensor):
        emb = emb_value.detach().cpu().numpy()
    else:
        emb = np.asarray(emb_value)

    emb = np.asarray(emb).squeeze()

    if emb.ndim != 1:
        raise ValueError(
            f"Embedding for key={emb_key!r} is not 1D after squeeze: shape={emb.shape}. "
            "This script expects mean-pooled protein embeddings."
        )

    return emb.astype(np.float32)

def load_embeddings_and_labels(emb_dir: str, split: str, csv_path: str):
    """
    加载 mean-pooled embeddings + Tm 标签。

    Args:
        emb_dir: 包含 train.pt / val.pt / test.pt 的目录。
        split: "train" / "val" / "test"。
        csv_path: 对应 CSV 文件，要求至少包含 seq_id 和 meltPoint 列。

    Returns:
        X: np.ndarray, shape=(n, D)
        y: np.ndarray, shape=(n,)
        seq_ids: list[str]
    """
    emb_path = Path(emb_dir) / f"{split}.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"seq_id", "meltPoint"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns {missing_cols}: {csv_path}")

    embeddings_dict = torch.load(emb_path, map_location="cpu")
    if not isinstance(embeddings_dict, dict):
        raise TypeError(f"Expected dict in {emb_path}, got {type(embeddings_dict)}")

    # ID 映射：embedding key 可能含 "|" 描述，只取第一个字段作为 seq_id。
    emb_mapping = {}
    duplicated_simple_ids = 0
    for emb_key, emb_value in embeddings_dict.items():
        simple_id = str(emb_key).split("|")[0]
        if simple_id in emb_mapping:
            duplicated_simple_ids += 1
        emb_mapping[simple_id] = _to_1d_numpy_embedding(emb_value, str(emb_key))

    X_list, y_list, seq_ids = [], [], []
    missing_embedding = 0
    missing_label = 0

    for _, row in df.iterrows():
        sid = str(row["seq_id"])
        tm = row["meltPoint"]

        if pd.isna(tm):
            missing_label += 1
            continue

        if sid not in emb_mapping:
            missing_embedding += 1
            continue

        X_list.append(emb_mapping[sid])
        y_list.append(float(tm))
        seq_ids.append(sid)

    if len(X_list) == 0:
        raise ValueError(
            f"No samples matched for split={split}. "
            f"Check seq_id in {csv_path} and embedding keys in {emb_path}."
        )

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(
        f"  {split}: {len(X)} samples matched, "
        f"missing_embedding={missing_embedding}, missing_label={missing_label}, "
        f"duplicated_simple_ids={duplicated_simple_ids}, embed_dim={X.shape[1]}"
    )

    return X, y, seq_ids

# ============================================================================
# 训练与评估
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, seed=2025, early_stopping_rounds=100):
    """训练 XGBoost，并使用验证集 early stopping。"""
    model = xgb.XGBRegressor(
        **FIXED_PARAMS,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds,
    )

    # XGBoost sklearn API 使用 eval_set 中最后一个数据集做 early stopping。
    # 因此这里保留 train 便于观察，但真正 early stopping 依据是 X_val / y_val。
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    if best_iteration is not None and best_score is not None:
        print(f"  Best iteration: {best_iteration}, best val MAE: {float(best_score):.4f}")

    return model

def _safe_corr(y_true, y_pred, corr_fn):
    """避免常数预测或异常输入导致相关系数计算直接中断。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2 or len(y_pred) < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    value, _ = corr_fn(y_true, y_pred)
    return float(value)

def evaluate_model(model, X, y, split_name="test"):
    """评估模型。"""
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    pearson_r = _safe_corr(y, preds, pearsonr)
    spearman_r = _safe_corr(y, preds, spearmanr)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(pearson_r),
        "Spearman_r": float(spearman_r),
    }

    print(
        f"  {split_name}: MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"R²={r2:.4f}  Pearson={pearson_r:.4f}  Spearman={spearman_r:.4f}"
    )

    return metrics, preds

def summarize_cv_metrics(fold_metrics):
    """汇总 K-fold CV 指标: mean / std。"""
    summary = {}
    metric_keys = ["MAE", "RMSE", "R2", "Pearson_r", "Spearman_r"]

    for key in metric_keys:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_std"] = float(np.nanstd(vals))

    return summary

def cross_validate_train(X_train, y_train, seed=2025, n_splits=5, early_stopping_rounds=100):
    """
    仅在训练集内部做 K-fold CV。

    说明:
      - 不使用固定 val/test。
      - 每折内部用 fold-val 做 early stopping。
      - 固定 test 仍只用于最终模型评估。
    """
    if n_splits <= 1:
        return [], {}
    if n_splits > len(X_train):
        raise ValueError(f"cv_folds={n_splits} is larger than train samples={len(X_train)}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    print(f"\nTraining-set {n_splits}-fold CV...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        print(f"\n  CV fold {fold}/{n_splits}")
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = train_model(
            X_tr,
            y_tr,
            X_va,
            y_va,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        metrics, _ = evaluate_model(model, X_va, y_va, split_name=f"cv_fold_{fold}")
        metrics["fold"] = fold
        metrics["n_train"] = int(len(tr_idx))
        metrics["n_val"] = int(len(va_idx))
        metrics["best_iteration"] = int(getattr(model, "best_iteration", -1))
        fold_metrics.append(metrics)

    cv_summary = summarize_cv_metrics(fold_metrics)
    print(
        f"\n  CV summary: MAE={cv_summary['MAE_mean']:.4f}±{cv_summary['MAE_std']:.4f}  "
        f"R²={cv_summary['R2_mean']:.4f}±{cv_summary['R2_std']:.4f}  "
        f"Spearman={cv_summary['Spearman_r_mean']:.4f}±{cv_summary['Spearman_r_std']:.4f}"
    )

    return fold_metrics, cv_summary

def save_predictions(output_dir, split, ids, true_vals, preds):
    """保存单个 split 的预测结果。"""
    pd.DataFrame(
        {
            "seq_id": ids,
            "true_tm": true_vals,
            "pred_tm": preds,
            "error": preds - true_vals,
            "abs_error": np.abs(preds - true_vals),
        }
    ).to_csv(output_dir / f"predictions_{split}.csv", index=False)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="XGBoost Tm Regression")
    parser.add_argument(
        "--emb-dir",
        type=str,
        required=True,
        help="Embedding directory, containing train.pt, val.pt, test.pt",
    )
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True, help="Model name for labeling")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="K-fold CV within training set only. Set 0/1 to disable.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="Early stopping patience. Recommended: 50-100 for this Tm setting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun even if metrics.json already exists.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"
    cv_required = args.cv_folds and args.cv_folds > 1
    cv_key = f"train_cv_{args.cv_folds}fold"

    # 检查已完成：如果需要 CV，则 metrics.json 必须包含对应 CV key 才算完成。
    if metrics_file.exists() and not args.overwrite:
        with open(metrics_file, "r", encoding="utf-8") as f:
            old_metrics = json.load(f)

        cv_done = (not cv_required) or (cv_key in old_metrics)
        if cv_done:
            print(f"Already done: {metrics_file}")
            t = old_metrics.get("test", {})
            print(f"  Test MAE={t.get('MAE', 0):.4f}  R²={t.get('R2', 0):.4f}")
            if cv_required:
                c = old_metrics.get(cv_key, {})
                print(f"  Train CV MAE={c.get('MAE_mean', 0):.4f}±{c.get('MAE_std', 0):.4f}")
            print("  Use --overwrite to rerun.")
            return

        print(f"Existing metrics found but required CV is missing, rerunning: {metrics_file}")

    print("=" * 70)
    print(f"XGBoost Tm Regression: {args.model_name}")
    print("=" * 70)
    print(f"Seed: {args.seed}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Early stopping rounds: {args.early_stopping_rounds}")

    # 加载数据
    print("\nLoading data...")
    X_train, y_train, train_ids = load_embeddings_and_labels(args.emb_dir, "train", args.train_csv)
    X_val, y_val, val_ids = load_embeddings_and_labels(args.emb_dir, "val", args.val_csv)
    X_test, y_test, test_ids = load_embeddings_and_labels(args.emb_dir, "test", args.test_csv)

    # Shuffle 训练数据。XGBoost 本身不依赖顺序，但这样可确保 CV / 保存顺序与 seed 绑定。
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    train_ids = [train_ids[i] for i in shuffle_idx]

    # 训练集内部 K-fold CV，不影响最终 train/val/test 评估逻辑。
    cv_fold_metrics, cv_summary = cross_validate_train(
        X_train,
        y_train,
        seed=args.seed,
        n_splits=args.cv_folds,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # 最终模型：用固定 train 训练，用固定 val early stopping。
    print("\nTraining final model...")
    model = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # 评估
    print("\nEvaluating final model...")
    train_metrics, train_preds = evaluate_model(model, X_train, y_train, "train")
    val_metrics, val_preds = evaluate_model(model, X_val, y_val, "val")
    test_metrics, test_preds = evaluate_model(model, X_test, y_test, "test")

    # 保存
    print("\nSaving...")

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    run_params = {
        **FIXED_PARAMS,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "early_stopping_rounds": args.early_stopping_rounds,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
    }
    with open(output_dir / "fixed_params.json", "w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    save_predictions(output_dir, "train", train_ids, y_train, train_preds)
    save_predictions(output_dir, "val", val_ids, y_val, val_preds)
    save_predictions(output_dir, "test", test_ids, y_test, test_preds)

    if cv_required:
        with open(output_dir / f"cv_train_{args.cv_folds}fold.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_splits": args.cv_folds,
                    "folds": cv_fold_metrics,
                    "summary": cv_summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    all_metrics = {
        "model": args.model_name,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
        "early_stopping_rounds": args.early_stopping_rounds,
        "params": FIXED_PARAMS,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_score": float(getattr(model, "best_score", np.nan)),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    if cv_required:
        all_metrics[cv_key] = cv_summary
        all_metrics[f"{cv_key}_folds"] = cv_fold_metrics

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\nDone: Test MAE={test_metrics['MAE']:.4f}°C")
    if cv_required:
        print(f"   Train CV MAE={cv_summary['MAE_mean']:.4f}±{cv_summary['MAE_std']:.4f}°C")
    print(f"   Best iteration={int(getattr(model, 'best_iteration', -1))}")
    print(f"   Results: {output_dir}")

if __name__ == "__main__":
    main()
