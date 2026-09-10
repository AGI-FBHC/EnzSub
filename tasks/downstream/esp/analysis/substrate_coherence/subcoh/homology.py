#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.homology — 序列同源控制

读取标准化的 MMseqs2 all-vs-all 结果表 (由 prepare_esp_homology_table.py 生成)，
为每个 query 构建「需要从近邻候选中排除的同源酶」集合。

标准化同源表必须包含列:
    query_id, target_id, pident, query_coverage, target_coverage
其中:
    - pident          : 序列一致性, 归一化到 [0, 1]
    - query_coverage  : 比对覆盖 query 的比例 [0, 1]
    - target_coverage : 比对覆盖 target 的比例 [0, 1]
query_id / target_id 必须是 seq_hash (与 enzyme cohort 对齐)。
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STD_COLUMNS = ("query_id", "target_id", "pident", "query_coverage", "target_coverage")

class HomologyTable:
    """缓存标准化同源表，按 cutoff 提供 per-query 排除集合。"""

    def __init__(self, df: pd.DataFrame) -> None:
        missing = [c for c in STD_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Homology table missing columns: {missing}; need {STD_COLUMNS}")
        self.df = df
        # 自动检测 pident 是否为百分比 (>1)，归一化到 [0,1]
        pmax = float(df["pident"].max()) if len(df) else 0.0
        if pmax > 1.0001:
            logger.info("pident appears to be in percent (max=%.2f); dividing by 100", pmax)
            self.df = self.df.copy()
            self.df["pident"] = self.df["pident"] / 100.0
        # cutoff -> {query_id -> set(target_id)}
        self._cache: Dict[tuple, Dict[str, Set[str]]] = {}

    @classmethod
    def from_tsv(cls, path: str) -> "HomologyTable":
        if not os.path.exists(path):
            raise FileNotFoundError(f"homology-table not found: {path}")
        df = pd.read_csv(path, sep="\t")
        logger.info("Loaded homology table: %s (%d hits)", path, len(df))
        return cls(df)

    def exclusion_sets(
        self, identity_cutoff: float, min_coverage: float = 0.0
    ) -> Dict[str, Set[str]]:
        """
        返回 {query_id -> set(target_id)}：所有 pident >= cutoff 且覆盖 >= min_coverage
        的同源对 (对称化：q->t 与 t->q 都排除)。
        """
        key = (round(float(identity_cutoff), 6), round(float(min_coverage), 6))
        if key in self._cache:
            return self._cache[key]

        d = self.df
        mask = (d["pident"] >= identity_cutoff)
        if min_coverage > 0:
            mask &= (d["query_coverage"] >= min_coverage) & (d["target_coverage"] >= min_coverage)
        hits = d.loc[mask, ["query_id", "target_id"]]

        excl: Dict[str, Set[str]] = {}
        for q, t in zip(hits["query_id"].values, hits["target_id"].values):
            if q == t:
                continue
            excl.setdefault(q, set()).add(t)
            excl.setdefault(t, set()).add(q)  # 对称
        self._cache[key] = excl
        logger.info("Homology exclusion @ identity>=%.2f, cov>=%.2f: %d queries have neighbours to exclude",
                    identity_cutoff, min_coverage, len(excl))
        return excl

    def pident_lookup(self) -> Dict[tuple, float]:
        """返回 {(q,t): pident}，用于审计近邻的 sequence identity (对称化, 取最大)。"""
        out: Dict[tuple, float] = {}
        for q, t, p in zip(self.df["query_id"].values,
                           self.df["target_id"].values,
                           self.df["pident"].values):
            if q == t:
                continue
            out[(q, t)] = max(out.get((q, t), 0.0), float(p))
            out[(t, q)] = max(out.get((t, q), 0.0), float(p))
        return out