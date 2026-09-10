#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions
"""

import torch
import numpy as np
import random
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,   # NEW
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score
)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(probs, labels, mask=None, threshold=0.5, return_best_threshold=False):
    """
    threshold: 用于 F1/MCC/precision/recall 的判定阈值 (默认 0.5)
    return_best_threshold: 若为 True, 额外在当前 probs/labels 上扫出 best-F1 阈值
                           ⚠️ 仅用于在【验证集】上选阈值, 切勿在测试集上用它选阈值
    """
    if mask is not None:
        valid_idx = mask > 0
        probs = probs[valid_idx]
        labels = labels[valid_idx]

    if probs.size == 0 or labels.size == 0:
        out = {'roc_auc': float('nan'), 'auprc': float('nan'),
               'mcc': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        if return_best_threshold:
            out['best_threshold'] = 0.5
        return out

    if np.unique(labels).size < 2:
        roc_auc = float('nan'); auprc = float('nan')
    else:
        roc_auc = float(roc_auc_score(labels, probs))
        auprc   = float(average_precision_score(labels, probs))

    # 可选: 在当前数据上扫 best-F1 阈值 (仅验证集用)
    if return_best_threshold:
        thrs = np.linspace(probs.min(), probs.max(), 200)
        best_f1, best_thr = -1.0, 0.5
        for t in thrs:
            p = (probs >= t).astype(int)
            f = f1_score(labels, p, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(t)
        threshold = best_thr

    preds = (probs >= threshold).astype(int)
    metrics = {
        'roc_auc': roc_auc,
        'auprc': auprc,
        'mcc': float(matthews_corrcoef(labels, preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'threshold': float(threshold),
    }
    if return_best_threshold:
        metrics['best_threshold'] = float(threshold)
    return metrics

def flatten_batch(probs, labels, masks):
    """
    Flatten batch tensors and apply mask

    Args:
        probs: torch.Tensor (batch, seq_len)
        labels: torch.Tensor (batch, seq_len)
        masks: torch.Tensor (batch, seq_len)

    Returns:
        probs_flat: np.array (n_valid_residues,)
        labels_flat: np.array (n_valid_residues,)
    """
    probs = probs.cpu().numpy().ravel()
    labels = labels.cpu().numpy().ravel()
    masks = masks.cpu().numpy().ravel()

    valid_idx = masks > 0
    return probs[valid_idx], labels[valid_idx]

class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def load_best_state(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)