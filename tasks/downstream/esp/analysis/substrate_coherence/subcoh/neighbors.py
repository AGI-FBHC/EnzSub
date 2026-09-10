#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.neighbors — enzyme-only 近邻检索 + 随机近邻基线

设计要点:
  * eligible candidate pool 与「随机抽样索引」均与模型无关 (只取决于 cohort +
    同源过滤 + self 排除)，因此 base 与 base_sub 严格共享 → 配对公平。
  * 实际 top-k 近邻随模型 embedding 变化 (这正是分析对象)。
  * 内存安全: cosine 相似度按 query 分 batch 计算。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def l2_normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (X / norms).astype(np.float32)

def build_eligible_pools(
    n: int,
    exclusion_by_idx: Optional[List[Set[int]]],
) -> List[np.ndarray]:
    """
    为每个 query 构建 eligible candidate index 数组 (排除 self 与同源)。
    与模型无关。exclusion_by_idx[i] = 需排除的候选 index 集合 (不含 self)。
    """
    pools: List[np.ndarray] = []
    all_idx = np.arange(n)
    for i in range(n):
        excl = {i}
        if exclusion_by_idx is not None:
            excl |= exclusion_by_idx[i]
        mask = np.ones(n, dtype=bool)
        mask[list(excl)] = False
        pools.append(all_idx[mask])
    return pools

def topk_neighbors(
    emb_norm: np.ndarray,
    eligible_pools: Sequence[np.ndarray],
    k_max: int,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个 query，在其 eligible pool 内按 cosine 相似度取 top-k_max 近邻。

    返回:
      nbr_idx : (N, k_max) int64, 不足 k_max 处填 -1
      nbr_sim : (N, k_max) float32, 不足处填 NaN
    (具体每个 k/有效性的筛选在上层根据 pool 大小处理。)
    """
    n = emb_norm.shape[0]
    nbr_idx = np.full((n, k_max), -1, dtype=np.int64)
    nbr_sim = np.full((n, k_max), np.nan, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = emb_norm[start:end] @ emb_norm.T  # (B, N)
        for bi, qi in enumerate(range(start, end)):
            pool = eligible_pools[qi]
            if pool.size == 0:
                continue
            row = sims[bi, pool]
            kk = min(k_max, pool.size)
            # 取 top-kk
            if pool.size > kk * 4:
                part = np.argpartition(-row, kk - 1)[:kk]
                order = part[np.argsort(-row[part])]
            else:
                order = np.argsort(-row)[:kk]
            sel = pool[order]
            nbr_idx[qi, :kk] = sel
            nbr_sim[qi, :kk] = row[order]
    return nbr_idx, nbr_sim

def sample_random_neighbors(
    eligible_pools: Sequence[np.ndarray],
    k: int,
    n_repeats: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    """
    为每个有效 query (pool>=k) 抽样随机近邻索引 (不放回，重复 n_repeats 次)。
    与模型无关 → base / sub 共享。返回 {query_idx -> (n_repeats, k) int 索引}。

    使用 per-query 派生的确定性 seed，保证可复现且跨模型一致。
    """
    out: Dict[int, np.ndarray] = {}
    for qi, pool in enumerate(eligible_pools):
        if pool.size < k:
            continue
        rng = np.random.default_rng(seed + qi * 1_000_003 + k * 7919)
        draws = np.empty((n_repeats, k), dtype=np.int64)
        for r in range(n_repeats):
            draws[r] = rng.choice(pool, size=k, replace=False)
        out[qi] = draws
    return out