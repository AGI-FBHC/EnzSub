#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.coherence — query 邻域底物化学一致性 (SubChem) 与 enrichment

C(S_i, S_j) 仅依赖底物集合 (与模型无关)，因此在一个 cohort+metric+filter 下
对 base / sub / random 共用同一套 coherence 缓存，既高效又保证三者严格可比。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chem import SubstrateChemistry
from .neighbors import topk_neighbors

logger = logging.getLogger(__name__)

class CoherenceComputer:
    """memoized C(i, j)；substrate set 与模型无关。"""

    def __init__(
        self,
        substrates_by_idx: Sequence[List[str]],
        chem: SubstrateChemistry,
        metric: str,
    ) -> None:
        self.substrates = list(substrates_by_idx)
        self.chem = chem
        self.metric = metric
        self._cache: Dict[Tuple[int, int], float] = {}

    def coh(self, i: int, j: int) -> float:
        key = (i, j) if i <= j else (j, i)
        v = self._cache.get(key)
        if v is None:
            v = self.chem.set_similarity(self.substrates[i], self.substrates[j], self.metric)
            self._cache[key] = v
        return v

@dataclass
class ModelConditionResult:
    # query_idx -> 指标 (仅对 pool>=k 的有效 query)
    coherence: Dict[int, float]
    enrichment: Dict[int, float]
    random_mean: Dict[int, float]
    random_std: Dict[int, float]
    n_eligible: Dict[int, int]
    nbr_idx: np.ndarray      # (N, k) 实际近邻 index，无效填 -1
    nbr_sim: np.ndarray      # (N, k) cosine 相似度

def random_baseline(
    computer: CoherenceComputer,
    random_draws: Dict[int, np.ndarray],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """对每个 query 计算随机近邻 coherence 的 mean/std (模型无关，base/sub 共享)。"""
    rmean: Dict[int, float] = {}
    rstd: Dict[int, float] = {}
    for qi, draws in random_draws.items():
        # draws: (n_repeats, k)
        per_repeat = np.empty(draws.shape[0], dtype=np.float64)
        for r in range(draws.shape[0]):
            per_repeat[r] = np.mean([computer.coh(qi, int(c)) for c in draws[r]])
        rmean[qi] = float(per_repeat.mean())
        rstd[qi] = float(per_repeat.std())
    return rmean, rstd

def compute_model_condition(
    emb_norm: np.ndarray,
    eligible_pools: Sequence[np.ndarray],
    k: int,
    computer: CoherenceComputer,
    random_draws: Dict[int, np.ndarray],
    rmean: Dict[int, float],
    rstd: Dict[int, float],
    batch_size: int = 256,
) -> ModelConditionResult:
    """
    对单个模型，在给定 homology 条件 (eligible_pools)、给定 k 与 metric 下，
    计算每个有效 query 的 SubChem 与 enrichment。

    rmean/rstd 由 random_baseline() 预先算好 (模型无关) 传入。
    """
    n = emb_norm.shape[0]
    nbr_idx, nbr_sim = topk_neighbors(emb_norm, eligible_pools, k_max=k, batch_size=batch_size)

    coherence: Dict[int, float] = {}
    enrichment: Dict[int, float] = {}
    n_eligible: Dict[int, int] = {}

    for qi in range(n):
        pool = eligible_pools[qi]
        n_eligible[qi] = int(pool.size)
        if pool.size < k:
            continue  # invalid for this k
        nbrs = nbr_idx[qi, :k]
        if np.any(nbrs < 0):
            continue
        c = float(np.mean([computer.coh(qi, int(j)) for j in nbrs]))
        coherence[qi] = c
        if qi in rmean:
            enrichment[qi] = c - rmean[qi]

    return ModelConditionResult(
        coherence=coherence, enrichment=enrichment,
        random_mean={q: rmean.get(q, float("nan")) for q in coherence},
        random_std={q: rstd.get(q, float("nan")) for q in coherence},
        n_eligible=n_eligible, nbr_idx=nbr_idx, nbr_sim=nbr_sim,
    )