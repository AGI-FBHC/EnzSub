#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.stats — 配对统计

统计单位始终是 query enzyme (不是 enzyme-substrate pair)。
bootstrap 以 query 为重采样单位。
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.stats import wilcoxon
    _SCIPY = True
except Exception:  # pragma: no cover
    _SCIPY = False

def bootstrap_ci_mean(
    deltas: np.ndarray, n_boot: int = 2000, seed: int = 2025, alpha: float = 0.05
) -> Tuple[float, float]:
    """对 mean(delta) 的 percentile bootstrap CI，重采样单位为 query。"""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = deltas.size
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = deltas[idx].mean()
    lo = float(np.percentile(means, 100 * (alpha / 2)))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi

def wilcoxon_signed_rank(base: np.ndarray, sub: np.ndarray) -> Tuple[float, float]:
    """对 (sub - base) 的 Wilcoxon signed-rank 检验。返回 (statistic, p)。"""
    base = np.asarray(base, dtype=np.float64)
    sub = np.asarray(sub, dtype=np.float64)
    d = sub - base
    if not _SCIPY:
        return float("nan"), float("nan")
    nz = d[d != 0]
    if nz.size == 0:
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(sub, base, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception as exc:  # pragma: no cover
        logger.warning("Wilcoxon failed: %s", exc)
        return float("nan"), float("nan")

def sign_flip_permutation_p(
    deltas: np.ndarray, n_perm: int = 10000, seed: int = 2025
) -> float:
    """
    paired sign-flip permutation test (two-sided)，原假设 mean(delta)=0。
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return float("nan")
    obs = abs(deltas.mean())
    rng = np.random.default_rng(seed)
    n = deltas.size
    count = 0
    for _ in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=n)
        if abs((signs * deltas).mean()) >= obs - 1e-15:
            count += 1
    return float((count + 1) / (n_perm + 1))

def paired_summary(
    base_vals: np.ndarray,
    sub_vals: np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 10000,
    seed: int = 2025,
) -> Dict[str, float]:
    """对一组配对 query 计算完整配对统计 (base vs sub)。"""
    base_vals = np.asarray(base_vals, dtype=np.float64)
    sub_vals = np.asarray(sub_vals, dtype=np.float64)
    assert base_vals.shape == sub_vals.shape
    deltas = sub_vals - base_vals
    n = deltas.size

    out: Dict[str, float] = {
        "n_queries": int(n),
        "base_mean": float(base_vals.mean()) if n else float("nan"),
        "sub_mean": float(sub_vals.mean()) if n else float("nan"),
        "mean_delta": float(deltas.mean()) if n else float("nan"),
        "median_delta": float(np.median(deltas)) if n else float("nan"),
        "fraction_improved": float(np.mean(deltas > 0)) if n else float("nan"),
    }
    lo, hi = bootstrap_ci_mean(deltas, n_boot=n_boot, seed=seed)
    out["bootstrap_ci_low"] = lo
    out["bootstrap_ci_high"] = hi
    stat, p = wilcoxon_signed_rank(base_vals, sub_vals)
    out["wilcoxon_statistic"] = stat
    out["wilcoxon_p"] = p
    out["permutation_p"] = sign_flip_permutation_p(deltas, n_perm=n_perm, seed=seed)
    return out