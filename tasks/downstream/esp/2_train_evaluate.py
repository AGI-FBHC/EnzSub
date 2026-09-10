#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2_train_evaluate.py  (sweep-only version)

GBDT / MLP 训练 + 评估，专为 sweep.py 调用设计。

主要改动:
- enzyme_vector 列名 (替代 ESM_vector)
- CV 索引固定走 StratifiedKFold (删除外部索引加载)
- 删除 similarity_bin 分箱评估 (只保留 overall)
- argparse 精简到 sweep.py 实际会传的参数

本次改动 (类别不平衡处理):
- MLP 改用 BCEWithLogitsLoss + pos_weight (替代裸 BCELoss)
- BindingMLP.forward 输出 logits (移除内部 sigmoid)，数值更稳定
- 新增 mlp_predict_proba 时显式 sigmoid
- 新增 --mlp-pos-weight 参数 ('auto' / 'none' / 浮点值)
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score,
    f1_score, precision_score, recall_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# Utils
# =============================================================================
def ensure_writable_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return path
    except PermissionError:
        fallback = os.path.join(".", "results", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(fallback, exist_ok=True)
        logging.warning(f"Permission denied for output_dir='{path}'. Falling back to '{fallback}'")
        return fallback

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_pos_weight(y: np.ndarray, spec: str) -> Optional[float]:
    """
    Resolve the pos_weight value for BCEWithLogitsLoss.

    spec:
      - 'none'  -> None (no weighting, equivalent to plain BCE)
      - 'auto'  -> n_negative / n_positive computed from y
      - float   -> use the provided value directly
    """
    spec = str(spec).strip().lower()
    if spec == "none":
        return None
    if spec == "auto":
        y_int = y.astype(int)
        n_pos = int(np.sum(y_int == 1))
        n_neg = int(np.sum(y_int == 0))
        if n_pos == 0:
            logging.warning("pos_weight='auto' but no positive samples found; disabling pos_weight.")
            return None
        pw = float(n_neg) / float(n_pos)
        return pw
    # try parse as float
    try:
        return float(spec)
    except ValueError:
        raise ValueError(f"--mlp-pos-weight must be 'auto', 'none', or a float; got '{spec}'")

# =============================================================================
# Feature building
# =============================================================================
def create_model_input(df: pd.DataFrame, mol_feature: str) -> Tuple[np.ndarray, np.ndarray, int]:
    if "enzyme_vector" not in df.columns:
        raise ValueError("DataFrame must contain column 'enzyme_vector'")
    if "Binding" not in df.columns:
        raise ValueError("DataFrame must contain column 'Binding'")

    X_enz = np.stack(df["enzyme_vector"].values).astype(np.float32)

    if mol_feature == "ecfp":
        if "ECFP" not in df.columns:
            raise ValueError("mol_feature=ecfp but 'ECFP' not found")
        ecfp_strs = df["ECFP"].astype(str).values
        X_mol = np.stack([(np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48) for s in ecfp_strs]).astype(np.float32)
        mol_dim = int(X_mol.shape[1])
    elif mol_feature == "gnn":
        if "GNN_vector" not in df.columns:
            raise ValueError("mol_feature=gnn but 'GNN_vector' not found")
        X_mol = np.stack(df["GNN_vector"].values).astype(np.float32)
        mol_dim = int(X_mol.shape[1])
    elif mol_feature == "chemberta":
        if "ChemBERTa_vector" not in df.columns:
            raise ValueError("mol_feature=chemberta but 'ChemBERTa_vector' not found")
        X_mol = np.stack(df["ChemBERTa_vector"].values).astype(np.float32)
        mol_dim = int(X_mol.shape[1])
    else:
        raise ValueError("mol_feature must be one of {'ecfp','gnn','chemberta'}")

    # ESP-like fusion order: [mol, enzyme]
    X = np.concatenate([X_mol, X_enz], axis=1).astype(np.float32)
    y = df["Binding"].to_numpy().astype(np.float32)
    return X, y, mol_dim

def make_sample_weight(y: np.ndarray, neg_weight: float) -> np.ndarray:
    return np.where(y == 0, float(neg_weight), 1.0).astype(np.float32)

# =============================================================================
# CV folds (always StratifiedKFold)
# =============================================================================
def make_stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    tr_folds, te_folds = [], []
    for tr, te in skf.split(idx, y.astype(int)):
        tr_folds.append(tr.astype(np.int64))
        te_folds.append(te.astype(np.int64))
    return tr_folds, te_folds

# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    n = int(len(y_true))
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos

    out: Dict[str, Any] = {
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if n_pos == 0 or n_neg == 0:
        out.update({
            "roc_auc": float("nan"),
            "mcc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "skipped": True
        })
        return out

    out.update({
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "skipped": False
    })
    return out

def aggregate_metrics(ms: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    def agg(key: str) -> Dict[str, float]:
        vals = np.array([m[key] for m in ms], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    return {
        "accuracy": agg("accuracy"),
        "roc_auc": agg("roc_auc"),
        "mcc": agg("mcc"),
        "f1": agg("f1"),
        "precision": agg("precision"),
        "recall": agg("recall"),
    }

# =============================================================================
# MLP
# =============================================================================
class BindingMLP(nn.Module):
    """
    NOTE: forward now returns raw LOGITS (no sigmoid), so it can be used with
    BCEWithLogitsLoss (numerically stable + supports pos_weight).
    Apply sigmoid externally when you need probabilities.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            self.norms.append(nn.Identity())
            self.dropouts.append(nn.Dropout(dropout))
            prev = h

        self.output = nn.Linear(prev, 1)

    def forward(self, x):
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            x_in = x
            x = layer(x)
            x = norm(x)
            x = torch.relu(x)
            x = dropout(x)
            if self.use_residual and x.shape == x_in.shape:
                x = x + x_in
        x = self.output(x)
        return x.squeeze(-1)  # logits

class BindingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return int(len(self.y))

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def mlp_predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    ds = BindingDataset(X, np.zeros((X.shape[0],), dtype=np.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for bx, _ in dl:
            bx = bx.to(device)
            logits = model(bx)
            p = torch.sigmoid(logits).detach().cpu().numpy()  # logits -> prob
            probs.append(p)
    return np.concatenate(probs, axis=0)

def _make_bce_criterion(pos_weight: Optional[float], device: torch.device) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)

def train_mlp_one_fold(
    X_tr, y_tr, X_va, y_va,
    input_dim, hidden_dims, dropout, use_residual,
    lr, weight_decay, batch_size, max_epochs, patience, clip_grad,
    pos_weight,
    device,
):
    model = BindingMLP(input_dim, hidden_dims, dropout, use_residual).to(device)
    criterion = _make_bce_criterion(pos_weight, device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_dl = DataLoader(BindingDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(BindingDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    patience_cnt = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for bx, by in tr_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for bx, by in va_dl:
                bx = bx.to(device)
                logits = model(bx)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                all_p.append(p)
                all_y.append(by.numpy())
        all_p = np.concatenate(all_p, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        m = compute_metrics(all_y, all_p)
        val_auc = m["roc_auc"]

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}

def cv_evaluate_mlp(X, y, train_folds, test_folds, mlp_args, device):
    fold_summaries = []
    fold_metrics = []
    best_epochs = []

    for fold_id, (tr_idx, te_idx) in enumerate(zip(train_folds, test_folds), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[te_idx], y[te_idx]

        # Resolve pos_weight per-fold from the TRAIN split only (avoids leaking val distribution)
        fold_args = dict(mlp_args)
        pw_spec = fold_args.pop("pos_weight_spec")
        fold_args["pos_weight"] = resolve_pos_weight(y_tr, pw_spec)

        model, info = train_mlp_one_fold(
            X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
            device=device, **fold_args
        )
        probs = mlp_predict_proba(model, X_va, device=device, batch_size=fold_args["batch_size"])
        m = compute_metrics(y_va, probs)
        fold_metrics.append(m)
        best_epochs.append(info["best_epoch"])

        fold_summaries.append({
            "fold": fold_id,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(te_idx)),
            "best_epoch": int(info["best_epoch"]),
            "pos_weight": fold_args["pos_weight"],
            "metrics": m
        })
        logging.info(
            f"[MLP Fold {fold_id}] ACC={m['accuracy']:.4f} AUC={m['roc_auc']:.4f} "
            f"MCC={m['mcc']:.4f} F1={m['f1']:.4f} REC={m['recall']:.4f} "
            f"(pos_weight={fold_args['pos_weight']}, n_val={m['n_samples']})"
        )

    agg = aggregate_metrics(fold_metrics)
    med_epoch = int(np.median(best_epochs)) if len(best_epochs) > 0 else int(mlp_args["max_epochs"])

    logging.info(
        f"[MLP CV Mean] ACC={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f} | "
        f"AUC={agg['roc_auc']['mean']:.4f}±{agg['roc_auc']['std']:.4f} | "
        f"MCC={agg['mcc']['mean']:.4f}±{agg['mcc']['std']:.4f} | "
        f"F1={agg['f1']['mean']:.4f}±{agg['f1']['std']:.4f} | "
        f"median_best_epoch={med_epoch}"
    )
    return {"folds": fold_summaries, "aggregate": agg, "median_best_epoch": med_epoch}

def train_final_mlp_full(
    X, y, input_dim, hidden_dims, dropout, use_residual,
    lr, weight_decay, batch_size, train_epochs, clip_grad, pos_weight, device,
):
    model = BindingMLP(input_dim, hidden_dims, dropout, use_residual).to(device)
    criterion = _make_bce_criterion(pos_weight, device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dl = DataLoader(BindingDataset(X, y), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(int(train_epochs)):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
    return model

# =============================================================================
# GBDT
# =============================================================================
def cv_evaluate_gbdt(X, y, train_folds, test_folds, xgb_params, num_round, neg_weight,
                     early_stopping_rounds=None):
    import xgboost as xgb

    fold_summaries = []
    fold_metrics = []
    best_iters = []

    for fold_id, (tr_idx, te_idx) in enumerate(zip(train_folds, test_folds), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        w_tr = make_sample_weight(y_tr, neg_weight=neg_weight)
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dtest = xgb.DMatrix(X_te, label=y_te)

        train_kwargs = dict(
            params=xgb_params, dtrain=dtrain, num_boost_round=int(num_round),
            evals=[(dtrain, "train"), (dtest, "val")], verbose_eval=False,
        )
        if early_stopping_rounds is not None and int(early_stopping_rounds) > 0:
            train_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

        booster = xgb.train(**train_kwargs)

        # Determine the iteration count this fold settled on.
        # With early stopping, best_iteration is the 0-indexed optimum on the val eval_metric.
        if early_stopping_rounds is not None and int(early_stopping_rounds) > 0 \
                and getattr(booster, "best_iteration", None) is not None:
            best_it = int(booster.best_iteration) + 1  # convert to a round count
            # Predict using only the trees up to the best iteration.
            prob = booster.predict(dtest, iteration_range=(0, best_it))
        else:
            best_it = int(num_round)
            prob = booster.predict(dtest)

        best_iters.append(best_it)

        m = compute_metrics(y_te, prob)
        fold_metrics.append(m)

        fold_summaries.append({
            "fold": fold_id,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(te_idx)),
            "best_iteration": best_it,
            "metrics": m
        })
        logging.info(
            f"[GBDT Fold {fold_id}] ACC={m['accuracy']:.4f} AUC={m['roc_auc']:.4f} "
            f"MCC={m['mcc']:.4f} (best_iter={best_it}, n_val={m['n_samples']})"
        )

    agg = aggregate_metrics(fold_metrics)
    med_iter = int(np.median(best_iters)) if len(best_iters) > 0 else int(num_round)
    logging.info(
        f"[GBDT CV Mean] ACC={agg['accuracy']['mean']:.4f}±{agg['accuracy']['std']:.4f} | "
        f"AUC={agg['roc_auc']['mean']:.4f}±{agg['roc_auc']['std']:.4f} | "
        f"MCC={agg['mcc']['mean']:.4f}±{agg['mcc']['std']:.4f} | "
        f"median_best_iter={med_iter}"
    )
    return {"folds": fold_summaries, "aggregate": agg, "median_best_iteration": med_iter}

def train_final_gbdt_full(X, y, xgb_params, num_round, neg_weight):
    import xgboost as xgb
    w = make_sample_weight(y, neg_weight=neg_weight)
    dtrain = xgb.DMatrix(X, label=y, weight=w)
    # No early stopping here: num_round is already the CV-chosen optimum (or the user value
    # when CV is skipped). There is no held-out val set at the final-fit stage by design.
    booster = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=int(num_round),
                        evals=[(dtrain, "train")], verbose_eval=False)
    return booster

# =============================================================================
# External evaluation (overall only, no similarity bins)
# =============================================================================
def eval_external_overall(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Dict[str, Any]]:
    return {"overall": compute_metrics(y_true, y_prob)}

def save_external_predictions(output_dir, test_name, df, y_true, y_prob,
                              id_col="Uniprot ID"):
    """
    保存逐样本预测, 供下游 OOD / ROC / bootstrap 分析。

    重要: df 必须是 create_model_input 实际使用的那个 df (已 dropna),
    其行顺序与 y_true / y_prob 严格一致。
    """
    import pandas as pd
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    ids = (df[id_col].astype(str).tolist()
           if id_col in df.columns
           else [f"row{i}" for i in range(len(y_true))])

    out = pd.DataFrame({
        "row_idx": range(len(y_true)),   # 位置键: 与 OOD 相似度表对齐 (Uniprot ID 不唯一)
        "sample_id": ids,
        "y_true": y_true.astype(int),
        "y_prob": y_prob.astype(float),
    })
    out.to_csv(os.path.join(pred_dir, f"{test_name}.csv"), index=False)
# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="GBDT/MLP train+evaluate (sweep-only)")

    # data
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-files", type=str, nargs="+", required=True)
    parser.add_argument("--test-names", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--mol-feature", type=str, default="gnn", choices=["ecfp", "gnn", "chemberta"])

    # model + CV
    parser.add_argument("--model-type", type=str, required=True, choices=["gbdt", "mlp"])
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)

    # MLP args
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--use-residual", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--torch-device", type=str, default="cuda:0")
    parser.add_argument("--mlp-pos-weight", type=str, default="auto",
                        help="pos_weight for BCEWithLogitsLoss: 'auto' (n_neg/n_pos), 'none', or a float")

    # GBDT args
    parser.add_argument("--num-round", type=int, default=361)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-child-weight", type=float, default=4.38583)
    parser.add_argument("--max-delta-step", type=float, default=1.65010)
    parser.add_argument("--eta", type=float, default=0.0920737)
    parser.add_argument("--alpha", type=float, default=2.81396)
    parser.add_argument("--lambda_", type=float, default=1.15217)
    parser.add_argument("--neg-weight", type=float, default=0.14162338902155536)
    parser.add_argument("--xgb-device", type=str, default="cpu")
    parser.add_argument("--tree-method", type=str, default="hist", choices=["hist", "gpu_hist", "auto"])
    parser.add_argument("--sampling-method", type=str, default="uniform", choices=["uniform", "gradient_based"])
    parser.add_argument("--max-bin", type=int, default=128)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=30,
                        help="GBDT early stopping patience during CV; set 0 to disable. "
                             "The final model is trained on the CV-chosen median best_iteration.")

    args = parser.parse_args()

    if len(args.test_files) != len(args.test_names):
        raise ValueError("test-files 和 test-names 数量必须相同")

    set_seed(args.seed)
    args.output_dir = ensure_writable_dir(args.output_dir)

    # --- Load train ---
    logging.info("Loading train data...")
    df_train = pd.read_pickle(args.train_file).dropna(subset=["enzyme_vector", "Binding"])
    X, y, mol_dim = create_model_input(df_train, args.mol_feature)
    n = int(X.shape[0])
    input_dim = int(X.shape[1])
    logging.info(f"Train: n={n} | X={X.shape} (mol_dim={mol_dim}, enzyme_dim={input_dim - mol_dim})")

    # --- Build folds ---
    cv_summary = None
    if not args.skip_cv:
        tr_folds, te_folds = make_stratified_folds(y, n_splits=args.cv_splits, seed=args.seed)
        logging.info(f"StratifiedKFold: {len(tr_folds)} folds, seed={args.seed}")
    else:
        logging.info("Skipping cross-validation (--skip-cv)")

    # --- Load test sets ---
    test_dfs: Dict[str, pd.DataFrame] = {}
    for f, name in zip(args.test_files, args.test_names):
        if not os.path.exists(f):
            logging.warning(f"Missing test file, skipped: {f}")
            continue
        df = pd.read_pickle(f).dropna(subset=["enzyme_vector", "Binding"])
        if args.mol_feature == "gnn" and "GNN_vector" not in df.columns:
            logging.warning(f"{name}: no GNN_vector, skipped"); continue
        if args.mol_feature == "ecfp" and "ECFP" not in df.columns:
            logging.warning(f"{name}: no ECFP, skipped"); continue
        if args.mol_feature == "chemberta" and "ChemBERTa_vector" not in df.columns:
            logging.warning(f"{name}: no ChemBERTa_vector, skipped"); continue
        test_dfs[name] = df
        logging.info(f"Test {name}: n={len(df)}")

    results: Dict[str, Any] = {
        "config": {
            "model_type": args.model_type,
            "train_file": args.train_file,
            "mol_feature": args.mol_feature,
            "mol_dim": mol_dim,
            "enzyme_dim": input_dim - mol_dim,
            "skip_cv": args.skip_cv,
            "cv_splits": args.cv_splits,
            "seed": args.seed,
        }
    }

    # =========================================================================
    # MLP branch
    # =========================================================================
    if args.model_type == "mlp":
        device = torch.device(args.torch_device if torch.cuda.is_available() and args.torch_device.startswith("cuda") else "cpu")
        logging.info(f"MLP device: {device}")

        # Report what pos_weight 'auto' would resolve to on the full train set (for logging only)
        full_pw = resolve_pos_weight(y, args.mlp_pos_weight)
        logging.info(f"MLP pos_weight spec='{args.mlp_pos_weight}' -> full-train value={full_pw}")

        mlp_args = {
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "use_residual": args.use_residual,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "clip_grad": args.clip_grad,
            "pos_weight_spec": args.mlp_pos_weight,  # resolved per-fold inside cv_evaluate_mlp
        }
        results["config"]["mlp"] = {k: v for k, v in mlp_args.items()}
        results["config"]["mlp"]["pos_weight_full_train"] = full_pw

        if not args.skip_cv:
            logging.info("Running MLP 5-fold CV...")
            cv_summary = cv_evaluate_mlp(X, y, tr_folds, te_folds, mlp_args, device)
            results["cv"] = cv_summary
            final_epochs = int(cv_summary["median_best_epoch"])
        else:
            final_epochs = args.max_epochs

        logging.info(f"Training FINAL MLP on full train for epochs={final_epochs} (pos_weight={full_pw}) ...")
        final_model = train_final_mlp_full(
            X=X, y=y, input_dim=input_dim,
            hidden_dims=args.hidden_dims, dropout=args.dropout, use_residual=args.use_residual,
            lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
            train_epochs=final_epochs, clip_grad=args.clip_grad, pos_weight=full_pw, device=device
        )

        external = {}
        for name, df in test_dfs.items():
            Xt, yt, _ = create_model_input(df, args.mol_feature)
            prob = mlp_predict_proba(final_model, Xt, device=device, batch_size=args.batch_size)
            external[name] = eval_external_overall(yt, prob)
            save_external_predictions(args.output_dir, name, df, yt, prob)
            o = external[name]["overall"]
            logging.info(f"[{name}] ACC={o['accuracy']:.4f} AUC={o['roc_auc']:.4f} MCC={o['mcc']:.4f} (n={o['n_samples']})")
        results["external_test_results"] = external

        torch.save({
            "state_dict": final_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "use_residual": args.use_residual,
            "train_epochs": final_epochs,
            "pos_weight": full_pw,
        }, os.path.join(args.output_dir, "final_mlp.pth"))

    # =========================================================================
    # GBDT branch
    # =========================================================================
    else:
        import xgboost as xgb

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": float(args.eta),
            "max_depth": int(args.max_depth),
            "min_child_weight": float(args.min_child_weight),
            "max_delta_step": float(args.max_delta_step),
            "alpha": float(args.alpha),
            "lambda": float(args.lambda_),
            "max_bin": int(args.max_bin),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "seed": int(args.seed),
        }

        dev = str(args.xgb_device).lower()
        if dev.startswith("cuda"):
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = dev
            xgb_params["sampling_method"] = args.sampling_method
        else:
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cpu"
            if args.sampling_method == "gradient_based":
                logging.warning("CPU hist only supports sampling_method=uniform. Switching to uniform.")
            xgb_params["sampling_method"] = "uniform"

        results["config"]["gbdt"] = {
            "num_round": args.num_round,
            "neg_weight": args.neg_weight,
            "early_stopping_rounds": args.early_stopping_rounds,
            "xgb_params": xgb_params
        }

        if not args.skip_cv:
            logging.info("Running GBDT 5-fold CV...")
            cv_summary = cv_evaluate_gbdt(X, y, tr_folds, te_folds, xgb_params,
                                          int(args.num_round), float(args.neg_weight),
                                          early_stopping_rounds=int(args.early_stopping_rounds))
            results["cv"] = cv_summary
            # Use the CV-chosen optimum so the final model isn't over-/under-trained.
            final_num_round = int(cv_summary["median_best_iteration"])
            logging.info(f"GBDT final num_round set from CV median best_iteration = {final_num_round}")
        else:
            final_num_round = int(args.num_round)
            logging.info(f"CV skipped; GBDT final num_round = {final_num_round} (from --num-round)")

        results["config"]["gbdt"]["final_num_round"] = final_num_round

        logging.info(f"Training FINAL GBDT on full train (num_round={final_num_round}) ...")
        final_booster = train_final_gbdt_full(X, y, xgb_params,
                                              final_num_round, float(args.neg_weight))

        external = {}
        for name, df in test_dfs.items():
            Xt, yt, _ = create_model_input(df, args.mol_feature)
            dtest = xgb.DMatrix(Xt, label=yt)
            prob = final_booster.predict(dtest)
            external[name] = eval_external_overall(yt, prob)
            save_external_predictions(args.output_dir, name, df, yt, prob)
            o = external[name]["overall"]
            logging.info(f"[{name}] ACC={o['accuracy']:.4f} AUC={o['roc_auc']:.4f} MCC={o['mcc']:.4f} (n={o['n_samples']})")
        results["external_test_results"] = external

        final_booster.save_model(os.path.join(args.output_dir, "final_gbdt.json"))

    # --- save metrics ---
    def convert_nan(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: convert_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_nan(v) for v in obj]
        return obj

    if "external_test_results" in results:
        results["test_results"] = results["external_test_results"]
    if "cv" in results and "aggregate" in results["cv"]:
        results["cv_metrics"] = results["cv"]["aggregate"]

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(convert_nan(results), f, indent=2)
    logging.info(f"Saved metrics: {metrics_path}")

    # --- short summary ---
    if "external_test_results" in results:
        print("\n" + "=" * 60)
        print(f"Summary - model={args.model_type.upper()}")
        if args.skip_cv:
            print("(CV skipped)")
        print("=" * 60)
        print(f"{'Test':<18} {'ACC':>8} {'AUC':>8} {'MCC':>8} {'n':>8}")
        print("-" * 60)
        for name, res in results["external_test_results"].items():
            o = res["overall"]
            print(f"{name:<18} {o['accuracy']:>8.4f} {o['roc_auc']:>8.4f} {o['mcc']:>8.4f} {o['n_samples']:>8d}")
        print("=" * 60)

if __name__ == "__main__":
    main()