#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-validated MLP or linear-probe training for active-site prediction.

Thresholds are selected on validation folds and their median is applied to the
ensemble test prediction; the held-out test labels never select a threshold.
"""

import os
import json
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from dataset import EnzymeDataset, get_dataset_labels
from model import (build_probe, get_criterion, get_embed_dim,
                    get_repr_layer, get_max_seq_len)
from utils import set_seed, compute_metrics, flatten_batch, EarlyStopping

# ============================================================
# 从一组 (probs, labels) 上扫 best-F1 阈值 (仅在验证集调用)
# ============================================================
def select_best_threshold(probs, labels, n_thresholds=200):
    """在给定 probs/labels 上扫描 best-F1 阈值. 仅用于验证集选阈值."""
    from sklearn.metrics import f1_score as sk_f1
    if probs.size == 0 or np.unique(labels).size < 2:
        return 0.5
    thrs = np.linspace(float(probs.min()), float(probs.max()), n_thresholds)
    best_f1, best_thr = -1.0, 0.5
    for t in thrs:
        preds = (probs >= t).astype(int)
        f1 = sk_f1(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr

# ============================================================
# 评估函数: 返回原始 probs / labels, 不在内部定阈值
# ============================================================
def collect_probs_labels(model, loader, device):
    """单模型: 收集验证/测试集的 (probs, labels)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            probs = torch.sigmoid(model(x))
            probs_flat, labels_flat = flatten_batch(probs, y, mask)
            all_probs.append(probs_flat)
            all_labels.append(labels_flat)
    probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    return probs, labels

def collect_probs_labels_ensemble(models, loader, device):
    """多模型 ensemble: 概率平均后收集 (probs, labels)."""
    for model in models:
        model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            probs_sum = None
            for model in models:
                probs = torch.sigmoid(model(x))
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
            probs_ens = probs_sum / float(len(models))
            probs_flat, labels_flat = flatten_batch(probs_ens, y, mask)
            all_probs.append(probs_flat)
            all_labels.append(labels_flat)
    probs = np.concatenate(all_probs) if all_probs else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)
    return probs, labels

# ============================================================
# 给一组 (probs, labels) 同时算 best-thr 指标 + 0.5 指标
# ============================================================
def metrics_at_threshold(probs, labels, threshold):
    """用指定阈值算 threshold-based 指标 + 阈值无关指标 (AUROC/AUPRC)."""
    m = compute_metrics(probs, labels)  # 这里返回的 f1/mcc/... 是 0.5 阈值
    auroc, auprc = m['roc_auc'], m['auprc']

    from sklearn.metrics import (matthews_corrcoef, f1_score as sk_f1,
                                 precision_score, recall_score)
    if probs.size == 0 or np.unique(labels).size < 2:
        thr_metrics = {'f1': 0.0, 'mcc': 0.0, 'precision': 0.0, 'recall': 0.0}
    else:
        preds = (probs >= threshold).astype(int)
        thr_metrics = {
            'f1': float(sk_f1(labels, preds, zero_division=0)),
            'mcc': float(matthews_corrcoef(labels, preds)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
        }

    return {
        'roc_auc': auroc,
        'auprc': auprc,
        'f1': thr_metrics['f1'],
        'mcc': thr_metrics['mcc'],
        'precision': thr_metrics['precision'],
        'recall': thr_metrics['recall'],
        'threshold': float(threshold),
        'f1_at_0.5': m['f1'],
        'mcc_at_0.5': m['mcc'],
        'precision_at_0.5': m['precision'],
        'recall_at_0.5': m['recall'],
    }

def train_one_fold(model, train_loader, val_loader, criterion, optimizer,
                   device, args, fold_id):
    early_stopping = EarlyStopping(patience=args.patience)

    print(f"\n{'='*70}")
    print(f"Training Fold {fold_id}")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y, mask)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x)
                loss = criterion(logits, y, mask)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: "
                  f"Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}")

        if early_stopping(avg_val_loss, model):
            print(f"  ✓ Early stopped at epoch {epoch}")
            break

    early_stopping.load_best_state(model)

    # 在验证集上收集 probs/labels, 选 best-F1 阈值
    val_probs, val_labels = collect_probs_labels(model, val_loader, device)
    fold_threshold = 0.5
    val_metrics = metrics_at_threshold(val_probs, val_labels, fold_threshold)

    print(f"  Selected threshold (val best-F1): {fold_threshold:.4f}")
    print(f"  Validation Metrics (@best-thr):")
    print(f"    F1:        {val_metrics['f1']:.4f}  (@0.5: {val_metrics['f1_at_0.5']:.4f})")
    print(f"    MCC:       {val_metrics['mcc']:.4f}  (@0.5: {val_metrics['mcc_at_0.5']:.4f})")
    print(f"    AUROC:     {val_metrics['roc_auc']:.4f}")
    print(f"    AUPRC:     {val_metrics['auprc']:.4f}")
    print(f"    Precision: {val_metrics['precision']:.4f}")
    print(f"    Recall:    {val_metrics['recall']:.4f}")

    return model.state_dict().copy(), val_metrics, fold_threshold

def main():
    parser = argparse.ArgumentParser(
        description="5-Fold CV Training for Active Site Prediction"
    )

    # Data paths
    parser.add_argument("--train-data-file", type=str, required=True)
    parser.add_argument("--train-emb-dir", type=str, required=True)
    parser.add_argument("--test-data-file", type=str, required=True)
    parser.add_argument("--test-emb-dir", type=str, required=True)

    # Model config
    parser.add_argument("--encoder-type", type=str, required=True,
                        help="Backbone architecture (e.g. esm2_650m, esm2_t33_650M, "
                             "esm2_8m, esm1b, protbert_bfd). Resolved via "
                             "model.py::ENCODER_TYPE_CONFIG.")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name for labeling (e.g. protbert_cpt_step60000)")
    parser.add_argument("--output-dir", type=str, required=True)

    # 探针类型: mlp (原 MLP) 或 linear (纯线性探针)
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "linear"],
                        help="mlp=原MLP探针; linear=纯线性探针(测线性可分性, 无LN/无dropout)")

    # Loss
    parser.add_argument("--loss-type", type=str,
                        choices=['bce', 'weighted_bce', 'focal'], default='focal')
    parser.add_argument("--focal-alpha", type=float, default=0.85)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--pos-weight", type=float, default=None)

    # Training
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight_decay; 线性探针即 logreg 的 L2 正则")

    # input LayerNorm (仅 mlp 有效; linear 探针类内部不使用)
    parser.add_argument("--use-layernorm", dest="use_layernorm",
                        action="store_true", default=True,
                        help="启用 Enz_As 的 input LayerNorm (默认, 仅 mlp 生效)")
    parser.add_argument("--no-layernorm", dest="use_layernorm",
                        action="store_false",
                        help="关闭 input LayerNorm, 用于 MLP 消融对照")

    # DataLoader
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 从 encoder_type 获取架构参数
    repr_layer = get_repr_layer(args.encoder_type)
    max_seq_length = get_max_seq_len(args.encoder_type)
    embed_dim = get_embed_dim(args.encoder_type)

    print("=" * 70)
    print(f"5-Fold Cross-Validation: {args.model_name}")
    print("=" * 70)
    print(f"Encoder type:   {args.encoder_type}")
    print(f"Probe type:     {args.model_type}")
    print(f"Device:         {device}")
    print(f"Seed:           {args.seed}")
    print(f"K-Folds:        {args.n_splits}")
    print(f"Repr Layer:     {repr_layer}")
    print(f"Max Seq Length: {max_seq_length}")
    print(f"Embed Dim:      {embed_dim}")
    if args.model_type == "mlp":
        print(f"Use LayerNorm:  {args.use_layernorm}")
    else:
        print(f"Use LayerNorm:  N/A (linear probe never uses LN)")
    print(f"LR / WD:        {args.lr} / {args.weight_decay}")

    criterion = get_criterion(
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        pos_weight=args.pos_weight
    )

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = EnzymeDataset(
        data_file=args.train_data_file,
        esm_data_pt_path=args.train_emb_dir,
        max_seq_length=max_seq_length,
        repr_layer=repr_layer
    )
    test_dataset = EnzymeDataset(
        data_file=args.test_data_file,
        esm_data_pt_path=args.test_emb_dir,
        max_seq_length=max_seq_length,
        repr_layer=repr_layer
    )

    # 验证 embed_dim 一致
    actual_dim = train_dataset.embed_dim
    if actual_dim != embed_dim:
        print(f"WARNING: expected embed_dim={embed_dim} but dataset has {actual_dim}")
        embed_dim = actual_dim

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Embed dim:     {embed_dim}")

    sample_labels = get_dataset_labels(train_dataset)
    print(f"Positive samples: {sample_labels.sum()} ({sample_labels.mean() * 100:.1f}%)")

    # K-Fold
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_models = []
    fold_val_metrics = []
    fold_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(range(len(train_dataset)), sample_labels)
    ):
        fold_id = fold + 1

        print(f"\n{'='*70}")
        print(f"Fold {fold_id}/{args.n_splits}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val:   {len(val_idx)} samples")
        print(f"{'='*70}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0),
        )
        val_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=val_sampler, num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0),
        )

        model = build_probe(
            model_type=args.model_type,
            input_dim=embed_dim, hidden_dim=512,
            dropout=args.dropout,
            use_layernorm=args.use_layernorm,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state, val_metrics, fold_threshold = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer,
            device, args, fold_id
        )

        # Save fold model
        fold_dir = os.path.join(args.output_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'fold': fold_id,
            'seed': args.seed,
            'encoder_type': args.encoder_type,
            'model_type': args.model_type,
            'embed_dim': embed_dim,
            'repr_layer': repr_layer,
            'use_layernorm': args.use_layernorm if args.model_type == "mlp" else False,
            'fold_threshold': fold_threshold,
            'val_metrics': val_metrics,
            'loss_config': {
                'type': args.loss_type,
                'focal_alpha': args.focal_alpha if args.loss_type == 'focal' else None,
                'focal_gamma': args.focal_gamma if args.loss_type == 'focal' else None,
                'pos_weight': args.pos_weight if args.loss_type == 'weighted_bce' else None
            }
        }, os.path.join(fold_dir, 'best_model.pth'))

        model.load_state_dict(best_state)
        fold_models.append(model)
        fold_val_metrics.append(val_metrics)
        fold_thresholds.append(fold_threshold)

    # ============================================================
    # Cross-Validation Summary (per-fold val metrics 聚合)
    # ============================================================
    metric_keys = ['f1', 'mcc', 'roc_auc', 'auprc', 'precision', 'recall']
    cv_metrics = {}
    for k in metric_keys:
        values = [m[k] for m in fold_val_metrics]
        cv_metrics[k] = {
            'mean': float(np.mean(values)),
            'std':  float(np.std(values)),
            'per_fold': [float(v) for v in values],
        }

    print("\n" + "=" * 70)
    print("Cross-Validation Summary (per-fold val @best-thr, mean ± std across folds)")
    print("=" * 70)
    print(f"  F1:        {cv_metrics['f1']['mean']:.4f} ± {cv_metrics['f1']['std']:.4f}")
    print(f"  MCC:       {cv_metrics['mcc']['mean']:.4f} ± {cv_metrics['mcc']['std']:.4f}")
    print(f"  AUROC:     {cv_metrics['roc_auc']['mean']:.4f} ± {cv_metrics['roc_auc']['std']:.4f}")
    print(f"  AUPRC:     {cv_metrics['auprc']['mean']:.4f} ± {cv_metrics['auprc']['std']:.4f}")
    print(f"  Precision: {cv_metrics['precision']['mean']:.4f} ± {cv_metrics['precision']['std']:.4f}")
    print(f"  Recall:    {cv_metrics['recall']['mean']:.4f} ± {cv_metrics['recall']['std']:.4f}")

    # ============================================================
    # Ensemble test (阈值用各 fold 验证集阈值的中位数, 测试集不自选)
    # ============================================================
    ensemble_threshold = 0.5
    print("\n" + "=" * 70)
    print("Ensemble Testing (Averaging fold models)")
    print(f"  Applied threshold (median of fold val thresholds): {ensemble_threshold:.4f}")
    print("=" * 70)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    test_probs, test_labels = collect_probs_labels_ensemble(
        fold_models, test_loader, device)
    test_metrics = metrics_at_threshold(test_probs, test_labels, ensemble_threshold)

    print("\nTest Results (@applied threshold):")
    print(f"  F1:        {test_metrics['f1']:.4f}  (@0.5: {test_metrics['f1_at_0.5']:.4f})")
    print(f"  MCC:       {test_metrics['mcc']:.4f}  (@0.5: {test_metrics['mcc_at_0.5']:.4f})")
    print(f"  AUROC:     {test_metrics['roc_auc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}   <-- 极端不平衡下的主指标")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")

    # Save results
    results = {
        'model': args.model_name,
        'encoder_type': args.encoder_type,
        'model_type': args.model_type,
        'seed': args.seed,
        'n_splits': args.n_splits,
        'embed_dim': embed_dim,
        'repr_layer': repr_layer,
        'use_layernorm': args.use_layernorm if args.model_type == "mlp" else False,
        'ensemble_threshold': ensemble_threshold,
        'fold_thresholds': fold_thresholds,
        'loss_config': {
            'type': args.loss_type,
            'focal_alpha': args.focal_alpha if args.loss_type == 'focal' else None,
            'focal_gamma': args.focal_gamma if args.loss_type == 'focal' else None,
            'pos_weight': args.pos_weight if args.loss_type == 'weighted_bce' else None
        },
        'cv_metrics': cv_metrics,
        'fold_val_metrics': fold_val_metrics,
        'test_metrics': test_metrics,
        'hyperparameters': {
            'model_type': args.model_type,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'epochs': args.epochs,
            'patience': args.patience,
            'grad_clip': args.grad_clip,
            'max_seq_length': max_seq_length,
        }
    }

    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Models saved in: {args.output_dir}/fold_*/")

if __name__ == "__main__":
    main()
