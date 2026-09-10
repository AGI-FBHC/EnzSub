#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLP and linear probes for residue-level active-site prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Encoder 配置 (唯一来源)
# ============================================================================

ENCODER_TYPE_CONFIG = {
    # ESM-1b
    "esm1b":          {"embed_dim": 1280, "repr_layer": 33, "max_seq_len": 1022},

    # ESM-2 全尺寸 (两套别名都加, 与 sub/encoders/esm.py 的 _ESM_REGISTRY 对齐)
    "esm2_t6_8M":     {"embed_dim": 320,  "repr_layer": 6,  "max_seq_len": 1022},
    "esm2_8m":        {"embed_dim": 320,  "repr_layer": 6,  "max_seq_len": 1022},

    "esm2_t12_35M":   {"embed_dim": 480,  "repr_layer": 12, "max_seq_len": 1022},
    "esm2_35m":       {"embed_dim": 480,  "repr_layer": 12, "max_seq_len": 1022},

    "esm2_t30_150M":  {"embed_dim": 640,  "repr_layer": 30, "max_seq_len": 1022},
    "esm2_150m":      {"embed_dim": 640,  "repr_layer": 30, "max_seq_len": 1022},

    "esm2_t33_650M":  {"embed_dim": 1280, "repr_layer": 33, "max_seq_len": 1022},
    "esm2_650m":      {"embed_dim": 1280, "repr_layer": 33, "max_seq_len": 1022},

    "esm2_t36_3B":    {"embed_dim": 2560, "repr_layer": 36, "max_seq_len": 1022},
    "esm2_3b":        {"embed_dim": 2560, "repr_layer": 36, "max_seq_len": 1022},

    "esm2_t48_15B":   {"embed_dim": 5120, "repr_layer": 48, "max_seq_len": 1022},
    "esm2_15b":       {"embed_dim": 5120, "repr_layer": 48, "max_seq_len": 1022},

    # ProtBERT
    "protbert_bfd":   {"embed_dim": 1024, "repr_layer": 30, "max_seq_len": 510},

    # External PLM baselines.  repr_layer=-1 denotes the final residue-level
    # hidden state written by generate_plm_per_residue_embeddings.py.
    "ankh3_large":    {"embed_dim": 1536, "repr_layer": -1, "max_seq_len": 1022},
    "prott5_xl":      {"embed_dim": 1024, "repr_layer": -1, "max_seq_len": 1022},
    "esm3_small":     {"embed_dim": 1536, "repr_layer": -1, "max_seq_len": 1022},
}

def get_embed_dim(encoder_type: str) -> int:
    return ENCODER_TYPE_CONFIG[encoder_type]["embed_dim"]

def get_repr_layer(encoder_type: str) -> int:
    return ENCODER_TYPE_CONFIG[encoder_type]["repr_layer"]

def get_max_seq_len(encoder_type: str) -> int:
    return ENCODER_TYPE_CONFIG[encoder_type]["max_seq_len"]

# ============================================================================
# 模型定义
# ============================================================================

class Enz_As(nn.Module):
    """
    MLP classifier for residue-level active site prediction
    forward returns logits (B, L). Apply sigmoid outside if needed.
    """

    def __init__(self, input_dim=1280, hidden_dim=512, dropout=0.4, use_layernorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim) if use_layernorm else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        logits = self.classifier(x).squeeze(-1)
        return logits

class Enz_As_Linear(nn.Module):
    """
    纯线性探针 (linear probe): frozen embedding → 单个 Linear → logits.

    用于测量表示的【线性可分性】, 对表示几何差异比 MLP 更敏感。

    刻意【不加 LayerNorm / 不加 dropout】:
      - LayerNorm 带可学习仿射 (weight/bias), 会逐维重缩放, 抹掉 CPT 可能
        改善的尺度/模长信息, 使探针不再纯粹反映原始表示的线性可分性。
      - dropout 对单层线性意义不大, 默认 0。
    若确实想要轻微正则, 可传 dropout>0 (一般不建议; L2 正则交给 optimizer 的
    weight_decay 更标准)。
    """

    def __init__(self, input_dim=1280, dropout=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        logits = self.linear(x).squeeze(-1)
        return logits

def build_probe(model_type="mlp", input_dim=1280, hidden_dim=512,
                dropout=0.4, use_layernorm=True):
    """
    统一探针工厂。

      - model_type="mlp":    Enz_As (原 MLP, 含可选 LayerNorm)
      - model_type="linear": Enz_As_Linear (纯线性探针, 无 LN / 无 dropout)

    Returns:
        nn.Module
    """
    model_type = model_type.lower()
    if model_type == "mlp":
        return Enz_As(input_dim=input_dim, hidden_dim=hidden_dim,
                      dropout=dropout, use_layernorm=use_layernorm)
    elif model_type == "linear":
        # 线性探针刻意忽略 hidden_dim / use_layernorm; dropout 默认 0
        return Enz_As_Linear(input_dim=input_dim, dropout=0.0)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r} (expected 'mlp' or 'linear')")

# ============================================================================
# Loss 函数
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be one of ['mean','sum','none']")
        self.reduction = reduction

    def forward(self, logits, targets, mask=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight_value = None if pos_weight is None else float(pos_weight)

    def forward(self, logits, targets, mask=None):
        pos_weight = self.pos_weight_value
        if pos_weight is None and mask is not None:
            valid_targets = targets[mask > 0]
            n_pos = float((valid_targets > 0.5).sum().item())
            n_neg = float((valid_targets <= 0.5).sum().item())
            pos_weight = n_neg / max(n_pos, 1.0)

        pos_weight_tensor = None
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(
                [pos_weight], device=logits.device, dtype=logits.dtype)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight_tensor, reduction='none')

        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

class MaskedBCEWithLogitsLoss(nn.Module):
    def forward(self, logits, targets, mask=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()

def get_criterion(loss_type='focal', **kwargs):
    loss_type = loss_type.lower()

    if loss_type == 'bce':
        print("Using BCEWithLogitsLoss (masked)")
        return MaskedBCEWithLogitsLoss()

    elif loss_type == 'weighted_bce':
        pos_weight = kwargs.get('pos_weight', None)
        if pos_weight is not None:
            print(f"Using Weighted BCEWithLogitsLoss (pos_weight={float(pos_weight):.4f})")
        else:
            print("Using Weighted BCEWithLogitsLoss (pos_weight=auto)")
        return WeightedBCELoss(pos_weight=pos_weight)

    elif loss_type == 'focal':
        alpha = kwargs.get('focal_alpha', 0.85)
        gamma = kwargs.get('focal_gamma', 2.0)
        reduction = kwargs.get('reduction', 'mean')
        print(f"Using Focal Loss (alpha={alpha}, gamma={gamma})")
        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
