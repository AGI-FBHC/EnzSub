#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.chem — 底物化学表示与集合相似度

独立于 SUB 的底物编码器 (ChemBERTa)。底物相似度全部基于 RDKit Morgan
fingerprint (默认 radius=2, nBits=2048, 即 ECFP4-like) + Tanimoto。

提供:
  - canonicalize_smiles: RDKit canonicalization, 非法返回 None
  - SubstrateChemistry: FP 缓存 + set-vs-set 相似度 (C_sym / C_max 等)，
    所有相似度位于 [0, 1]。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    _RDKIT_AVAILABLE = False

# 允许的 set-vs-set 相似度指标
SET_METRICS = ("c_sym", "c_max", "mean_pairwise", "median_pairwise", "jaccard")
# 两个必需指标
REQUIRED_SET_METRICS = ("c_sym", "c_max")

class ChemError(RuntimeError):
    pass

def require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ChemError(
            "RDKit is required for substrate chemistry. Install via "
            "`pip install rdkit` (or rdkit-pypi)."
        )

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES; 非法 / 空 / 无法解析返回 None。"""
    require_rdkit()
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

class SubstrateChemistry:
    """
    Morgan fingerprint 缓存 + set-vs-set 相似度。

    所有 SMILES 必须是已 canonicalize 过的 (调用方负责)。FP / set-pair 相似度
    全部带缓存，避免重复计算。
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        require_rdkit()
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self._fp: Dict[str, object] = {}
        # set-pair 相似度缓存: key = (sig_a, sig_b, metric)  (sig_a <= sig_b)
        self._pair_cache: Dict[Tuple[str, str, str], float] = {}
        # 每个 substrate-set 的稳定签名缓存
        self._sig_cache: Dict[Tuple[str, ...], str] = {}

    # ---- fingerprint ----
    def fp(self, canonical_smiles: str):
        fp = self._fp.get(canonical_smiles)
        if fp is None:
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is None:
                raise ChemError(f"Un-parseable canonical SMILES in fp(): {canonical_smiles!r}")
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            self._fp[canonical_smiles] = fp
        return fp

    def precompute(self, smiles_iterable: Sequence[str]) -> None:
        for s in smiles_iterable:
            self.fp(s)
        logger.info("Pre-computed %d Morgan fingerprints (r=%d, bits=%d)",
                    len(self._fp), self.radius, self.n_bits)

    # ---- substrate-set signature (用于缓存) ----
    @staticmethod
    def set_signature(substrate_set: Sequence[str]) -> str:
        """对一个 (去重后) substrate set 给出稳定签名。"""
        import hashlib

        key = tuple(sorted(set(substrate_set)))
        h = hashlib.sha1("\n".join(key).encode("utf-8")).hexdigest()[:16]
        return h

    # ---- pairwise Tanimoto matrix helpers ----
    def _row_max_col_max(
        self, set_i: Sequence[str], set_j: Sequence[str]
    ) -> Tuple[List[float], List[float], float, float, float]:
        """
        返回 (row_max, col_max, global_max, mean_all, median_all)。
        row_max[a] = max_t T(s_a, t); col_max[b] = max_s T(s, t_b)。
        """
        fi = [self.fp(s) for s in set_i]
        fj = [self.fp(t) for t in set_j]

        row_max: List[float] = []
        all_sims: List[float] = []
        # 逐行: T(s_a, all j) —— BulkTanimoto 高效
        col_max_arr = np.zeros(len(fj), dtype=np.float64) if fj else np.zeros(0)
        for fa in fi:
            sims = DataStructs.BulkTanimotoSimilarity(fa, fj) if fj else []
            if sims:
                row_max.append(float(max(sims)))
                all_sims.extend(sims)
                col_max_arr = np.maximum(col_max_arr, np.asarray(sims, dtype=np.float64))
            else:
                row_max.append(0.0)
        col_max = [float(x) for x in col_max_arr] if len(col_max_arr) else []
        global_max = float(max(all_sims)) if all_sims else 0.0
        mean_all = float(np.mean(all_sims)) if all_sims else 0.0
        median_all = float(np.median(all_sims)) if all_sims else 0.0
        return row_max, col_max, global_max, mean_all, median_all

    # ---- public: set-vs-set similarity ----
    def set_similarity(
        self, set_i: Sequence[str], set_j: Sequence[str], metric: str
    ) -> float:
        """
        计算两个底物集合之间的相似度。结果 ∈ [0, 1]。

        metric:
          - 'c_sym'          : 对称最佳匹配平均 (主指标)
          - 'c_max'          : 任意一对的最大 Tanimoto (敏感性)
          - 'mean_pairwise'  : 所有 pair 的平均 (附加)
          - 'median_pairwise': 所有 pair 的中位 (附加)
          - 'jaccard'        : 精确 canonical 底物集合的 Jaccard overlap (附加)
        """
        if metric not in SET_METRICS:
            raise ValueError(f"Unknown set metric: {metric}; valid={SET_METRICS}")

        si = sorted(set(set_i))
        sj = sorted(set(set_j))
        if not si or not sj:
            return 0.0

        if metric == "jaccard":
            a, b = set(si), set(sj)
            inter = len(a & b)
            union = len(a | b)
            return float(inter / union) if union else 0.0

        sig_i = self.set_signature(si)
        sig_j = self.set_signature(sj)
        ckey = (min(sig_i, sig_j), max(sig_i, sig_j), metric)
        if ckey in self._pair_cache:
            return self._pair_cache[ckey]

        row_max, col_max, gmax, mean_all, median_all = self._row_max_col_max(si, sj)

        if metric == "c_sym":
            val = 0.5 * (float(np.mean(row_max)) + float(np.mean(col_max)))
        elif metric == "c_max":
            val = gmax
        elif metric == "mean_pairwise":
            val = mean_all
        elif metric == "median_pairwise":
            val = median_all
        else:  # pragma: no cover
            raise ValueError(metric)

        val = float(min(1.0, max(0.0, val)))
        self._pair_cache[ckey] = val
        return val