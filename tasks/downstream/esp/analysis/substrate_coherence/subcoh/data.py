#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.data — ESP pkl 读取、清洗、按酶聚合、cohort 交集、底物过滤

关键不变量:
  * 按完整 sequence 聚合 (一个 sequence -> 一个 enzyme record)；
  * 同一 sequence 在同一模型文件内的多行 enzyme_vector 必须数值一致 (否则报错)；
  * 不同模型比较时，query cohort 与 substrate annotation 必须完全一致 (本模块强制)；
  * substrate annotation 与模型无关 (来自同一 ESP 表)，只有 enzyme_vector 随模型变化。
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .chem import SubstrateChemistry, canonicalize_smiles

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("sequence", "SMILES", "Binding", "enzyme_vector")
EMBEDDING_CONSISTENCY_ATOL = 1e-4

def seq_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()

@dataclass
class ModelData:
    """单个模型文件清洗+聚合后的结果 (substrate set 仍随模型独立保存以便交叉校验)。"""

    name: str
    path: str
    backbone: str
    mode: str
    # seq_hash -> embedding (float32, 1-D)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    # seq_hash -> 去重后的 canonical SMILES 列表 (排序)
    substrates: Dict[str, List[str]] = field(default_factory=dict)
    # seq_hash -> 完整 sequence
    sequences: Dict[str, str] = field(default_factory=dict)
    emb_dim: int = 0
    stats: Dict[str, object] = field(default_factory=dict)

def load_and_aggregate(
    name: str,
    path: str,
    backbone: str,
    mode: str,
    chem: SubstrateChemistry,
) -> ModelData:
    """读取一个模型的 ESP pkl，过滤 Binding==1，清洗，按 sequence 聚合。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{name}] model file not found: {path}")

    df = pd.read_pickle(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] pkl missing required columns: {missing}")

    n_raw = int(len(df))

    # 1) Binding == 1
    df = df[df["Binding"] == 1].copy()
    n_pos = int(len(df))

    # 2/3/4) 去除缺失 sequence / SMILES / enzyme_vector
    def _valid_vec(v) -> bool:
        if v is None:
            return False
        try:
            arr = np.asarray(v, dtype=np.float32)
        except Exception:
            return False
        return arr.ndim == 1 and arr.size > 0 and np.all(np.isfinite(arr))

    df = df[df["sequence"].apply(lambda s: isinstance(s, str) and len(s) > 0)]
    df = df[df["SMILES"].apply(lambda s: isinstance(s, str) and len(s) > 0)]
    df = df[df["enzyme_vector"].apply(_valid_vec)]

    # 5/6) canonicalize SMILES, 记录非法
    canon = df["SMILES"].apply(canonicalize_smiles)
    invalid_mask = canon.isna()
    n_invalid_smiles = int(invalid_mask.sum())
    if n_invalid_smiles:
        bad = df.loc[invalid_mask, "SMILES"].unique().tolist()[:10]
        logger.warning("[%s] %d rows with invalid SMILES dropped (e.g. %s)",
                       name, n_invalid_smiles, bad)
    df = df.loc[~invalid_mask].copy()
    df["canonical_smiles"] = canon.loc[~invalid_mask].values

    # 重复 enzyme-substrate pair 统计 (按 sequence + canonical_smiles)
    pair_dupes = int(df.duplicated(subset=["sequence", "canonical_smiles"]).sum())

    embeddings: Dict[str, np.ndarray] = {}
    substrates: Dict[str, List[str]] = {}
    sequences: Dict[str, str] = {}
    emb_dims = set()
    inconsistent_emb: List[str] = []

    for sequence, grp in df.groupby("sequence", sort=False):
        h = seq_hash(sequence)
        vecs = np.stack([np.asarray(v, dtype=np.float32) for v in grp["enzyme_vector"].values])
        # 9/10) 同一 sequence 多行 embedding 必须一致
        if vecs.shape[0] > 1:
            spread = float(np.max(np.abs(vecs - vecs[0:1])))
            if spread > EMBEDDING_CONSISTENCY_ATOL:
                inconsistent_emb.append(f"{h[:10]} (max|Δ|={spread:.2e})")
        emb = vecs[0].astype(np.float32)
        emb_dims.add(int(emb.shape[0]))
        embeddings[h] = emb
        sequences[h] = sequence
        # 7) 酶内 canonical SMILES 去重
        substrates[h] = sorted(set(grp["canonical_smiles"].tolist()))

    if len(emb_dims) != 1:
        raise ValueError(f"[{name}] inconsistent embedding dims within file: {sorted(emb_dims)}")
    emb_dim = emb_dims.pop()

    if inconsistent_emb:
        raise ValueError(
            f"[{name}] {len(inconsistent_emb)} sequence(s) have DIFFERENT enzyme_vector "
            f"across rows (atol={EMBEDDING_CONSISTENCY_ATOL}). This must not be silently "
            f"averaged. Examples: {inconsistent_emb[:5]}"
        )

    n_unique_enz = len(embeddings)
    n_unique_sub = len({s for subs in substrates.values() for s in subs})
    sizes = np.array([len(v) for v in substrates.values()])
    stats = {
        "n_raw_rows": n_raw,
        "n_positive_rows": n_pos,
        "n_unique_enzymes": n_unique_enz,
        "n_unique_canonical_substrates": n_unique_sub,
        "n_invalid_smiles_rows": n_invalid_smiles,
        "n_duplicate_pairs": pair_dupes,
        "substrate_count_min": int(sizes.min()) if sizes.size else 0,
        "substrate_count_median": float(np.median(sizes)) if sizes.size else 0.0,
        "substrate_count_max": int(sizes.max()) if sizes.size else 0,
        "n_single_substrate_enzymes": int(np.sum(sizes == 1)),
        "n_multi_substrate_enzymes": int(np.sum(sizes >= 2)),
        "emb_dim": emb_dim,
    }
    logger.info(
        "[%s] raw=%d pos=%d uniq_enz=%d uniq_sub=%d invalid_smiles=%d dim=%d",
        name, n_raw, n_pos, n_unique_enz, n_unique_sub, n_invalid_smiles, emb_dim,
    )

    return ModelData(
        name=name, path=path, backbone=backbone, mode=mode,
        embeddings=embeddings, substrates=substrates, sequences=sequences,
        emb_dim=emb_dim, stats=stats,
    )

def build_global_id_map(models: Sequence[ModelData]) -> Dict[str, str]:
    """
    从所有模型的 sequence 并集构建全局 enzyme ID (按 seq_hash 排序)。
    保证 query ID 在所有模型 / 所有比较中一致。
    """
    all_hashes = sorted({h for m in models for h in m.embeddings})
    id_map = {h: f"E{idx:06d}" for idx, h in enumerate(all_hashes)}
    logger.info("Global enzyme ID map: %d unique sequences", len(id_map))
    return id_map

@dataclass
class Cohort:
    """一个 (modelA vs modelB) 比较使用的共享 cohort。"""

    backbone: str
    model_a: str
    model_b: str
    seq_hashes: List[str]                       # 排序后的共有 enzyme (顺序固定)
    enzyme_ids: List[str]                       # 与 seq_hashes 对齐
    substrates: Dict[str, List[str]]            # seq_hash -> canonical SMILES (共享, 已校验一致)
    emb_a: np.ndarray                           # (N, D) 对齐 seq_hashes
    emb_b: np.ndarray
    emb_dim: int

def build_cohort(
    model_a: ModelData,
    model_b: ModelData,
    id_map: Dict[str, str],
) -> Cohort:
    """构建两个模型共有 enzyme 的对齐 cohort，并校验 substrate annotation 完全一致。"""
    if model_a.backbone != model_b.backbone:
        raise ValueError(
            f"Backbone mismatch in comparison {model_a.name} vs {model_b.name}: "
            f"{model_a.backbone} != {model_b.backbone}. Refusing to mix embedding spaces."
        )
    if model_a.emb_dim != model_b.emb_dim:
        raise ValueError(
            f"Embedding dim mismatch {model_a.name}({model_a.emb_dim}) vs "
            f"{model_b.name}({model_b.emb_dim})."
        )

    common = sorted(set(model_a.embeddings) & set(model_b.embeddings))
    if not common:
        raise ValueError(f"No common enzymes between {model_a.name} and {model_b.name}.")

    # 校验 substrate annotation 一致 (来自同一 ESP 表，应完全相同)
    mismatched = []
    substrates: Dict[str, List[str]] = {}
    for h in common:
        sa = tuple(model_a.substrates[h])
        sb = tuple(model_b.substrates[h])
        if sa != sb:
            mismatched.append(h[:10])
        substrates[h] = list(sa)
    if mismatched:
        raise ValueError(
            f"Substrate annotations differ between {model_a.name} and {model_b.name} "
            f"for {len(mismatched)} shared enzymes (e.g. {mismatched[:5]}). "
            f"Paired comparison requires identical substrate sets."
        )

    enzyme_ids = [id_map[h] for h in common]
    emb_a = np.stack([model_a.embeddings[h] for h in common]).astype(np.float32)
    emb_b = np.stack([model_b.embeddings[h] for h in common]).astype(np.float32)

    logger.info(
        "Cohort %s vs %s [%s]: %d shared enzymes (A had %d, B had %d)",
        model_a.name, model_b.name, model_a.backbone, len(common),
        len(model_a.embeddings), len(model_b.embeddings),
    )
    return Cohort(
        backbone=model_a.backbone, model_a=model_a.name, model_b=model_b.name,
        seq_hashes=common, enzyme_ids=enzyme_ids, substrates=substrates,
        emb_a=emb_a, emb_b=emb_b, emb_dim=model_a.emb_dim,
    )

# =============================================================================
# 高频底物 / currency metabolite 过滤
# =============================================================================
def load_currency_file(path: Optional[str]) -> Dict[str, Optional[str]]:
    """读取 currency metabolite 文件 (每行一个 SMILES)，返回 {原始: canonical}。"""
    out: Dict[str, Optional[str]] = {}
    if not path:
        return out
    if not os.path.exists(path):
        raise FileNotFoundError(f"currency-metabolite-file not found: {path}")
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out[s] = canonicalize_smiles(s)
    logger.info("Loaded %d currency metabolites from %s", len(out), path)
    return out

def compute_excluded_substrates(
    substrates: Dict[str, List[str]],
    currency_canon: Dict[str, Optional[str]],
    max_enzyme_fraction: Optional[float],
) -> Tuple[set, pd.DataFrame]:
    """
    返回 (excluded_canonical_set, excluded_table)。
    频率定义: enzyme_fraction = #含该底物的酶 / #酶 (cohort 内, 未过滤)。
    """
    n_enz = max(1, len(substrates))
    # 统计每个 canonical substrate 出现在多少个酶中
    enz_count: Dict[str, int] = {}
    for subs in substrates.values():
        for s in set(subs):
            enz_count[s] = enz_count.get(s, 0) + 1

    currency_set = {c for c in currency_canon.values() if c is not None}

    rows = []
    excluded = set()
    for canon, cnt in sorted(enz_count.items(), key=lambda kv: -kv[1]):
        frac = cnt / n_enz
        reasons = []
        if canon in currency_set:
            reasons.append("currency_file")
        if max_enzyme_fraction is not None and frac > max_enzyme_fraction:
            reasons.append(f"frequency>{max_enzyme_fraction}")
        if reasons:
            excluded.add(canon)
            rows.append({
                "canonical_smiles": canon,
                "number_of_enzymes": cnt,
                "enzyme_fraction": round(frac, 6),
                "exclusion_reason": "+".join(reasons),
            })

    # currency 文件里给的原始 SMILES (即便不在 cohort) 也记录便于审计
    cohort_canon = set(enz_count)
    for orig, canon in currency_canon.items():
        if canon is None:
            rows.append({
                "original_smiles": orig, "canonical_smiles": None,
                "number_of_enzymes": 0, "enzyme_fraction": 0.0,
                "exclusion_reason": "currency_file_unparseable",
            })
        elif canon not in cohort_canon:
            rows.append({
                "original_smiles": orig, "canonical_smiles": canon,
                "number_of_enzymes": 0, "enzyme_fraction": 0.0,
                "exclusion_reason": "currency_file_not_in_cohort",
            })

    table = pd.DataFrame(rows)
    if "original_smiles" not in table.columns:
        table["original_smiles"] = None
    logger.info("Metabolite filtering: %d substrates excluded (of %d in cohort)",
                len(excluded), len(cohort_canon))
    return excluded, table

def apply_substrate_filter(
    substrates: Dict[str, List[str]],
    excluded: set,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    返回 (filtered_substrates, dropped_seq_hashes)。
    底物集合被清空的酶从 cohort 中剔除 (无法定义化学一致性)。
    """
    filtered: Dict[str, List[str]] = {}
    dropped: List[str] = []
    for h, subs in substrates.items():
        kept = [s for s in subs if s not in excluded]
        if kept:
            filtered[h] = kept
        else:
            dropped.append(h)
    if dropped:
        logger.info("Metabolite filter dropped %d enzymes with empty substrate set", len(dropped))
    return filtered, dropped