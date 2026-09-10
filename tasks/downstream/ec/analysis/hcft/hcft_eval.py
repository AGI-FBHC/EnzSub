#!/usr/bin/env python3
"""
hcft_eval.py — Homology-Conflicting Functional Triplet analysis (HCFT)

HCFT 是 HFS 之后的更严格几何诊断。它不再只比较同一个 identity bin 内
same-EC 与 different-EC pair 的平均相似度，而是主动构造“同源性与功能标签
冲突”的三元组：

    anchor enzyme a
    positive enzyme p: 与 a 共享 EC4，但序列一致性更低
    negative enzyme n: 与 a 不共享 EC4，但序列一致性更高

若 embedding space 真正具有 EC-aware functional ordering，则应满足：

    sim(z_a, z_p) > sim(z_a, z_n)

即：即使 negative 在序列上更像 anchor，模型仍应把同 EC4 的 positive 排得更近。

核心指标：
  HCFT-Acc    = mean[sim(a,p) > sim(a,n)]，先按 anchor 求均值，再对 anchor 平均
  HCFT-Margin = mean[sim(a,p) - sim(a,n)]，先按 anchor 求均值，再对 anchor 平均

输入：
  --identity-pairs   build_identity_pairs.py 产出的 pair_identity.csv
  --embedding-dirs   name:path，每个模型一个 embedding 目录，复用 sweep/knn_eval.py 的 EmbeddingStore
  --label-csv        EC 标签 CSV；只有在需要重算 functional_match 或 pair 文件没有 labels1/labels2 时使用

注意：
  1. 本脚本默认使用已有 MMseqs pair_identity.csv。因此 HCFT 是在 MMseqs 可观测的
     pairwise identity graph 上构造 triplet，不是全量 all-vs-all 序列空间。
  2. 统计单位是 anchor enzyme。不要直接把所有 triplet 混在一起做 p value；本脚本
     默认输出 anchor-level summary，再汇总到 model-level。
  3. 主分析建议 ec_level=4, match_mode=any_overlap。

输出：
  hcft_anchor_summary.csv   每个 model/seed/anchor 的 HCFT
  hcft_seed_summary.csv     每个 model/seed 的 anchor-level 平均结果
  hcft_summary.csv          每个 model 跨 seed 的均值与标准差
  delta_hcft_summary.csv    相对 baseline 的 HCFT 变化
  hcft_eval.log / config.json
  可选：--save-triplets 输出 sampled_triplets_seed*.csv 和 scored_triplets.csv
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from hfs_common import (
    setup_sweep_path,
    build_labelsets,
    functional_match,
    VALID_MATCH_MODES,
)

setup_sweep_path()
from knn_eval import EmbeddingStore, parse_kv_pairs  # noqa: E402
from utils import load_ec_labels  # noqa: E402

# ======================================================================
# 默认参数
# ======================================================================
_DEFAULTS = dict(
    ec_level=4,
    match_mode="any_overlap",
    negative_mode="general",          # general | hard_ec1 | hard_ec2 | hard_ec3
    positive_max_identity=40.0,        # positive: same EC4, identity <= this
    negative_min_identity=40.0,        # negative: different EC4, identity >= this
    min_identity_margin=10.0,          # identity(a,n) - identity(a,p) >= this
    n_neg_per_pos=5,
    max_pos_per_anchor=200,
    max_neg_per_anchor=500,
    max_triplets_per_anchor=1000,
    min_triplets_per_anchor=10,
    n_seeds=5,
    baseline_name=None,
    recompute_match=False,
    chunk_size=100_000,
    save_triplets=False,
    max_anchors=None,
)

_VALID_NEGATIVE_MODES = ("general", "hard_ec1", "hard_ec2", "hard_ec3")

# ======================================================================
# YAML config 支持
# ======================================================================
def load_config(path: str) -> dict:
    """读 YAML config，返回扁平化的 hcft_eval 参数 dict。

    支持：
      common:      ec_level / match_mode 等共享项
      hcft_eval:   本脚本参数，会覆盖 common

    若 YAML 没有 common/hcft_eval，则把顶层当作参数 dict。
    """
    try:
        import yaml
    except ImportError:
        raise SystemExit("读取 --config 需要 PyYAML。请先安装: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"config 文件顶层必须是 mapping: {path}")

    common = raw.get("common") or {}
    section = raw.get("hcft_eval") or {}
    if not common and not section:
        section = {k: v for k, v in raw.items()
                   if k not in ("build_identity_pairs", "hfs_eval", "plot_hfs")}
    merged = {}
    merged.update(common)
    merged.update(section)
    return merged

def resolve(key, cli_val, cfg: dict, default):
    """参数优先级：命令行显式值 > config 文件 > 内置默认。"""
    if cli_val is not None:
        return cli_val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default

# ======================================================================
# EC 标签与 negative 难度
# ======================================================================
def parse_label_field(x) -> Set[str]:
    """把 pair_identity.csv 中的 labels1/labels2 字段解析成 set[str]。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {p.strip() for p in s.split(";") if p.strip()}

def labels_to_str(labels: Optional[Set[str]]) -> str:
    if not labels:
        return ""
    return ";".join(sorted(labels))

def ec_prefixes(labels: Set[str], level: int) -> Set[str]:
    """取 EC 前缀集合。例如 level=3: 1.1.1.1 -> 1.1.1。"""
    out = set()
    for ec in labels:
        parts = ec.split(".")
        if len(parts) >= level:
            out.add(".".join(parts[:level]))
    return out

def negative_passes_mode(
    anchor_labels: Set[str],
    cand_labels: Set[str],
    negative_mode: str,
) -> bool:
    """判断一个 different-EC negative 是否满足指定 hard negative 条件。

    general: 只要求不共享完整 EC 标签。
    hard_ec1/2/3: 不共享完整 EC 标签，但共享指定 level 的 EC 前缀。
    """
    if not anchor_labels or not cand_labels:
        return False
    if anchor_labels & cand_labels:
        return False
    if negative_mode == "general":
        return True
    if negative_mode.startswith("hard_ec"):
        level = int(negative_mode.replace("hard_ec", ""))
        return bool(ec_prefixes(anchor_labels, level) & ec_prefixes(cand_labels, level))
    raise ValueError(f"未知 negative_mode: {negative_mode!r}")

# ======================================================================
# pair 文件加载
# ======================================================================
def load_pairs_for_hcft(
    identity_pairs_csv: str,
    ec_level: int,
    match_mode: str,
    label_csv: Optional[str],
    recompute_match: bool,
    negative_mode: str,
    log,
) -> pd.DataFrame:
    """读取 pair_identity.csv，并确保有 id1/id2/identity/functional_match/labels1/labels2。"""
    if not os.path.exists(identity_pairs_csv):
        raise FileNotFoundError(f"identity-pairs 文件不存在: {identity_pairs_csv}")

    df = pd.read_csv(identity_pairs_csv, dtype={"id1": str, "id2": str})
    if df.empty:
        raise ValueError(f"identity-pairs 文件为空: {identity_pairs_csv}")
    for col in ("id1", "id2", "identity"):
        if col not in df.columns:
            raise ValueError(f"identity-pairs 缺少必需列 {col!r}; 现有列: {list(df.columns)}")

    df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
    n_bad = int(df["identity"].isna().sum())
    if n_bad:
        log(f"  identity 解析失败丢弃 pair: {n_bad}")
    df = df[~df["identity"].isna()].copy()
    log(f"  读取有效 pair 行数: {len(df)}")

    # 检查 pair 文件口径
    file_ec_level = None
    file_match_mode = None
    if "ec_level" in df.columns and len(df):
        try:
            file_ec_level = int(pd.to_numeric(df["ec_level"].iloc[0]))
        except Exception:
            file_ec_level = None
    if "match_mode" in df.columns and len(df):
        file_match_mode = str(df["match_mode"].iloc[0])

    has_labels = "labels1" in df.columns and "labels2" in df.columns
    consistent = (
        not recompute_match
        and "functional_match" in df.columns
        and file_ec_level == ec_level
        and file_match_mode == match_mode
        and has_labels
    )

    if consistent:
        log(f"  functional_match/labels: 复用 pair 文件列 "
            f"(file ec_level={file_ec_level}, match_mode={file_match_mode})")
        df = df[df["functional_match"].isin(["positive", "negative"])].copy()
    else:
        reason = "强制重算" if recompute_match else (
            f"pair文件口径/ec标签列不足 "
            f"(file ec_level={file_ec_level}, match_mode={file_match_mode}, has_labels={has_labels})"
        )
        log(f"  functional_match/labels: 用 --label-csv 现场重算 ({reason})")
        if not label_csv or not os.path.exists(label_csv):
            raise FileNotFoundError(
                "需要重算 functional_match/labels，但 --label-csv 不存在或未提供。"
            )
        id_to_raw, _ = load_ec_labels(label_csv)
        id_to_labels = build_labelsets(id_to_raw, ec_level)
        log(f"  标签重算: EC{ec_level} 下有效标签序列 {len(id_to_labels)}")

        matches, labels1, labels2 = [], [], []
        for a, b in zip(df["id1"].values, df["id2"].values):
            la = id_to_labels.get(a, set())
            lb = id_to_labels.get(b, set())
            matches.append(functional_match(la, lb, match_mode))
            labels1.append(labels_to_str(la))
            labels2.append(labels_to_str(lb))
        df["functional_match"] = matches
        df["labels1"] = labels1
        df["labels2"] = labels2
        n_skip = int((df["functional_match"] == "skip").sum())
        if n_skip:
            log(f"  重算后被判为 skip 丢弃 pair: {n_skip}")
        df = df[df["functional_match"].isin(["positive", "negative"])].copy()

    # hard negative 需要 labels1/labels2
    if negative_mode != "general" and not ("labels1" in df.columns and "labels2" in df.columns):
        raise ValueError("negative_mode != general 时必须有 labels1/labels2，或提供 --label-csv 重算。")

    n_pos = int((df["functional_match"] == "positive").sum())
    n_neg = int((df["functional_match"] == "negative").sum())
    log(f"  HCFT 可用 pair: {len(df)} [pos={n_pos}, neg={n_neg}]")
    return df.reset_index(drop=True)

# ======================================================================
# 构建 directed adjacency 与 triplets
# ======================================================================
def make_directed_edges(df: pd.DataFrame) -> pd.DataFrame:
    """把 undirected pair 展开成 anchor -> candidate 的 directed edges。"""
    left = pd.DataFrame({
        "anchor": df["id1"].values,
        "candidate": df["id2"].values,
        "identity": df["identity"].values,
        "functional_match": df["functional_match"].values,
        "anchor_labels": df["labels1"].values,
        "candidate_labels": df["labels2"].values,
    })
    right = pd.DataFrame({
        "anchor": df["id2"].values,
        "candidate": df["id1"].values,
        "identity": df["identity"].values,
        "functional_match": df["functional_match"].values,
        "anchor_labels": df["labels2"].values,
        "candidate_labels": df["labels1"].values,
    })
    return pd.concat([left, right], ignore_index=True)

def _sample_without_replacement(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)

def sample_triplets_for_seed(
    directed: pd.DataFrame,
    seed: int,
    negative_mode: str,
    positive_max_identity: float,
    negative_min_identity: float,
    min_identity_margin: float,
    n_neg_per_pos: int,
    max_pos_per_anchor: int,
    max_neg_per_anchor: int,
    max_triplets_per_anchor: int,
    min_triplets_per_anchor: int,
    max_anchors: Optional[int],
    log,
) -> pd.DataFrame:
    """按 anchor 构造并采样 HCFT triplets。"""
    rng = np.random.default_rng(seed)

    anchors = directed["anchor"].drop_duplicates().to_numpy()
    if max_anchors is not None and len(anchors) > max_anchors:
        anchors = anchors[rng.choice(len(anchors), size=max_anchors, replace=False)]
        directed = directed[directed["anchor"].isin(set(anchors))].copy()
        log(f"  seed={seed}: 使用 max_anchors={max_anchors} 后 anchors={len(anchors)}")

    rows = []
    n_anchor_seen = 0
    n_anchor_with_pos = 0
    n_anchor_with_neg = 0
    n_anchor_with_triplet = 0

    for anchor, g in directed.groupby("anchor", sort=False):
        n_anchor_seen += 1

        pos = g[(g["functional_match"] == "positive") &
                (g["identity"] <= positive_max_identity)]
        if pos.empty:
            continue
        n_anchor_with_pos += 1

        neg = g[(g["functional_match"] == "negative") &
                (g["identity"] >= negative_min_identity)]
        if neg.empty:
            continue

        if negative_mode != "general":
            # 对 hard negative 逐行判断是否共享指定 EC 前缀但不共享完整 EC。
            anchor_labels_cache = {}
            keep = []
            for r in neg.itertuples(index=False):
                alabel_s = r.anchor_labels
                clabel_s = r.candidate_labels
                if alabel_s not in anchor_labels_cache:
                    anchor_labels_cache[alabel_s] = parse_label_field(alabel_s)
                la = anchor_labels_cache[alabel_s]
                lc = parse_label_field(clabel_s)
                keep.append(negative_passes_mode(la, lc, negative_mode))
            neg = neg[np.array(keep, dtype=bool)]
            if neg.empty:
                continue

        n_anchor_with_neg += 1

        pos = pos.reset_index(drop=True)
        neg = neg.reset_index(drop=True)
        p_idx = _sample_without_replacement(rng, len(pos), max_pos_per_anchor)
        n_idx = _sample_without_replacement(rng, len(neg), max_neg_per_anchor)
        pos = pos.iloc[p_idx].reset_index(drop=True)
        neg = neg.iloc[n_idx].reset_index(drop=True)

        # 为每个 positive 采若干满足 identity conflict 的 negatives。
        anchor_rows = []
        neg_identities = neg["identity"].to_numpy(dtype=float)
        for pr in pos.itertuples(index=False):
            valid_neg_idx = np.where(neg_identities - float(pr.identity) >= min_identity_margin)[0]
            if len(valid_neg_idx) == 0:
                continue
            take = min(n_neg_per_pos, len(valid_neg_idx))
            chosen = rng.choice(valid_neg_idx, size=take, replace=False)
            for ni in chosen:
                nr = neg.iloc[int(ni)]
                anchor_rows.append({
                    "seed": seed,
                    "anchor": anchor,
                    "positive": pr.candidate,
                    "negative": nr["candidate"],
                    "id_anchor_positive": float(pr.identity),
                    "id_anchor_negative": float(nr["identity"]),
                    "delta_identity": float(nr["identity"]) - float(pr.identity),
                    "negative_mode": negative_mode,
                })
                if len(anchor_rows) >= max_triplets_per_anchor:
                    break
            if len(anchor_rows) >= max_triplets_per_anchor:
                break

        if len(anchor_rows) >= min_triplets_per_anchor:
            rows.extend(anchor_rows)
            n_anchor_with_triplet += 1

    out = pd.DataFrame(rows)
    log(f"  seed={seed}: anchors_seen={n_anchor_seen}, with_pos={n_anchor_with_pos}, "
        f"with_neg={n_anchor_with_neg}, with_triplets={n_anchor_with_triplet}, "
        f"triplets={len(out)}")
    return out

# ======================================================================
# embedding cosine + HCFT 计算
# ======================================================================
def pair_cosines(X: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray,
                 chunk_size: int) -> np.ndarray:
    n = len(idx_i)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        vi = X[idx_i[s:e]]
        vj = X[idx_j[s:e]]
        out[s:e] = np.einsum("ij,ij->i", vi, vj)
    return out

def eval_one_model_on_triplets(
    emb_name: str,
    emb_dir: str,
    triplets: pd.DataFrame,
    seed: int,
    chunk_size: int,
    min_triplets_per_anchor: int,
    save_scored: bool,
    scored_dir: Optional[str],
    log,
) -> Tuple[List[dict], List[dict], Optional[pd.DataFrame]]:
    """对一个模型和一个 seed 的 triplets 计算 HCFT。"""
    if triplets.empty:
        return [], [], None
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"embedding 目录不存在: {emb_dir}")

    store = EmbeddingStore(emb_dir)
    available = store.get_available_ids()
    if not available:
        raise ValueError(f"模型 {emb_name} 的目录里没有可用 embedding: {emb_dir}")

    needed = sorted(set(triplets["anchor"]) |
                    set(triplets["positive"]) |
                    set(triplets["negative"]))
    have = [sid for sid in needed if sid in available]
    log(f"    {emb_name} seed={seed}: triplet unique IDs={len(needed)}, have embedding={len(have)}")
    if not have:
        return [], [], None

    X, valid_ids = store.load(have, normalize=True)
    id2row = {sid: r for r, sid in enumerate(valid_ids)}

    mask_have = (triplets["anchor"].map(id2row.__contains__) &
                 triplets["positive"].map(id2row.__contains__) &
                 triplets["negative"].map(id2row.__contains__))
    n_drop = int((~mask_have).sum())
    if n_drop:
        log(f"    {emb_name} seed={seed}: 因缺 embedding 丢弃 triplets={n_drop}")
    sub = triplets[mask_have].copy()
    if sub.empty:
        return [], [], None

    ai = sub["anchor"].map(id2row).to_numpy(dtype=np.int64)
    pi = sub["positive"].map(id2row).to_numpy(dtype=np.int64)
    ni = sub["negative"].map(id2row).to_numpy(dtype=np.int64)

    sim_pos = pair_cosines(X, ai, pi, chunk_size)
    sim_neg = pair_cosines(X, ai, ni, chunk_size)
    sub["embedding"] = emb_name
    sub["sim_anchor_positive"] = sim_pos
    sub["sim_anchor_negative"] = sim_neg
    sub["hcft_margin"] = sim_pos - sim_neg
    sub["hcft_correct"] = (sim_pos > sim_neg).astype(int)

    anchor_rows = []
    for anchor, g in sub.groupby("anchor", sort=False):
        if len(g) < min_triplets_per_anchor:
            continue
        anchor_rows.append({
            "embedding": emb_name,
            "seed": seed,
            "anchor": anchor,
            "negative_mode": str(g["negative_mode"].iloc[0]),
            "n_triplets": int(len(g)),
            "hcft_acc": float(g["hcft_correct"].mean()),
            "hcft_margin_mean": float(g["hcft_margin"].mean()),
            "hcft_margin_median": float(g["hcft_margin"].median()),
            "mean_id_pos": float(g["id_anchor_positive"].mean()),
            "mean_id_neg": float(g["id_anchor_negative"].mean()),
            "mean_delta_identity": float(g["delta_identity"].mean()),
        })

    if not anchor_rows:
        return [], [], sub if save_scored else None

    adf = pd.DataFrame(anchor_rows)
    seed_row = {
        "embedding": emb_name,
        "seed": seed,
        "negative_mode": str(adf["negative_mode"].iloc[0]),
        "n_anchors": int(len(adf)),
        "n_triplets_total": int(adf["n_triplets"].sum()),
        # anchor-level average: 每个 anchor 权重相同
        "hcft_acc_anchor_mean": float(adf["hcft_acc"].mean()),
        "hcft_acc_anchor_std": float(adf["hcft_acc"].std(ddof=0)),
        "hcft_margin_anchor_mean": float(adf["hcft_margin_mean"].mean()),
        "hcft_margin_anchor_std": float(adf["hcft_margin_mean"].std(ddof=0)),
        "mean_delta_identity_anchor_mean": float(adf["mean_delta_identity"].mean()),
        # triplet-level reference: 仅作参考，不作为主统计口径
        "hcft_acc_triplet_mean_ref": float(sub["hcft_correct"].mean()),
        "hcft_margin_triplet_mean_ref": float(sub["hcft_margin"].mean()),
    }

    scored = None
    if save_scored:
        scored = sub
        if scored_dir:
            os.makedirs(scored_dir, exist_ok=True)
            scored_path = os.path.join(scored_dir, f"scored_triplets__{emb_name}__seed{seed}.csv")
            scored.to_csv(scored_path, index=False)

    return anchor_rows, [seed_row], scored

# ======================================================================
# delta / summary
# ======================================================================
def pick_baseline(emb_names: Sequence[str], explicit: Optional[str]) -> Optional[str]:
    if explicit:
        if explicit not in emb_names:
            raise ValueError(f"--baseline-name {explicit!r} 不在 embedding 列表 {list(emb_names)} 中")
        return explicit
    cands = [n for n in emb_names
             if "base" in n.lower() and "sub" not in n.lower() and "cpt" not in n.lower()]
    return cands[0] if cands else None

def summarize_across_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame()
    rows = []
    for emb, g in seed_df.groupby("embedding", sort=False):
        rows.append({
            "embedding": emb,
            "negative_mode": str(g["negative_mode"].iloc[0]),
            "n_seeds_ok": int(len(g)),
            "n_anchors_mean": float(g["n_anchors"].mean()),
            "n_triplets_total_mean": float(g["n_triplets_total"].mean()),
            "hcft_acc_mean": float(g["hcft_acc_anchor_mean"].mean()),
            "hcft_acc_std": float(g["hcft_acc_anchor_mean"].std(ddof=0)),
            "hcft_margin_mean": float(g["hcft_margin_anchor_mean"].mean()),
            "hcft_margin_std": float(g["hcft_margin_anchor_mean"].std(ddof=0)),
            "mean_delta_identity": float(g["mean_delta_identity_anchor_mean"].mean()),
            "hcft_acc_triplet_mean_ref": float(g["hcft_acc_triplet_mean_ref"].mean()),
            "hcft_margin_triplet_mean_ref": float(g["hcft_margin_triplet_mean_ref"].mean()),
        })
    return pd.DataFrame(rows)

def compute_delta(seed_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if seed_df.empty or baseline is None:
        return pd.DataFrame()
    base = seed_df[seed_df["embedding"] == baseline]
    base_map = {int(r.seed): r for r in base.itertuples(index=False)}
    rows = []
    for r in seed_df.itertuples(index=False):
        if r.embedding == baseline:
            continue
        b = base_map.get(int(r.seed))
        if b is None:
            continue
        rows.append({
            "embedding": r.embedding,
            "baseline": baseline,
            "seed": int(r.seed),
            "negative_mode": r.negative_mode,
            "hcft_acc": r.hcft_acc_anchor_mean,
            "baseline_hcft_acc": b.hcft_acc_anchor_mean,
            "delta_hcft_acc": r.hcft_acc_anchor_mean - b.hcft_acc_anchor_mean,
            "hcft_margin": r.hcft_margin_anchor_mean,
            "baseline_hcft_margin": b.hcft_margin_anchor_mean,
            "delta_hcft_margin": r.hcft_margin_anchor_mean - b.hcft_margin_anchor_mean,
        })
    return pd.DataFrame(rows)

# ======================================================================
# main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Homology-Conflicting Functional Triplet analysis (HCFT)")
    ap.add_argument("--config", default=None, help="YAML config 路径")
    ap.add_argument("--embedding-dirs", nargs="+", default=None, help="name:path pairs")
    ap.add_argument("--label-csv", default=None, help="EC 标签 CSV；重算 match 或 labels 时需要")
    ap.add_argument("--identity-pairs", default=None, help="build_identity_pairs.py 产出的 pair_identity.csv")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--ec-level", type=int, default=None, choices=[1, 2, 3, 4])
    ap.add_argument("--match-mode", default=None, choices=list(VALID_MATCH_MODES))
    ap.add_argument("--negative-mode", default=None, choices=list(_VALID_NEGATIVE_MODES))
    ap.add_argument("--positive-max-identity", type=float, default=None)
    ap.add_argument("--negative-min-identity", type=float, default=None)
    ap.add_argument("--min-identity-margin", type=float, default=None)
    ap.add_argument("--n-neg-per-pos", type=int, default=None)
    ap.add_argument("--max-pos-per-anchor", type=int, default=None)
    ap.add_argument("--max-neg-per-anchor", type=int, default=None)
    ap.add_argument("--max-triplets-per-anchor", type=int, default=None)
    ap.add_argument("--min-triplets-per-anchor", type=int, default=None)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--baseline-name", default=None)
    ap.add_argument("--recompute-match", action="store_const", const=True, default=None)
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--save-triplets", action="store_const", const=True, default=None,
                    help="保存采样 triplets 和每个模型的 scored triplets，文件可能较大")
    ap.add_argument("--max-anchors", type=int, default=None,
                    help="调试用：最多采样多少个 anchor。正式结果建议不设。")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}

    if args.embedding_dirs is not None:
        emb_dirs = parse_kv_pairs(args.embedding_dirs)
    elif cfg.get("embedding_dirs"):
        ed = cfg["embedding_dirs"]
        emb_dirs = dict(ed) if isinstance(ed, dict) else parse_kv_pairs(list(ed))
    else:
        ap.error("必须提供 embedding-dirs (命令行 --embedding-dirs 或 config)")

    label_csv = resolve("label_csv", args.label_csv, cfg, None)
    identity_pairs = resolve("identity_pairs", args.identity_pairs, cfg, None)
    output_dir = resolve("output_dir", args.output_dir, cfg, None)
    missing = [n for n, v in [("identity-pairs", identity_pairs), ("output-dir", output_dir)] if not v]
    if missing:
        ap.error("缺少必填参数: " + ", ".join(missing))

    ec_level = int(resolve("ec_level", args.ec_level, cfg, _DEFAULTS["ec_level"]))
    match_mode = resolve("match_mode", args.match_mode, cfg, _DEFAULTS["match_mode"])
    negative_mode = resolve("negative_mode", args.negative_mode, cfg, _DEFAULTS["negative_mode"])
    positive_max_identity = float(resolve("positive_max_identity", args.positive_max_identity, cfg, _DEFAULTS["positive_max_identity"]))
    negative_min_identity = float(resolve("negative_min_identity", args.negative_min_identity, cfg, _DEFAULTS["negative_min_identity"]))
    min_identity_margin = float(resolve("min_identity_margin", args.min_identity_margin, cfg, _DEFAULTS["min_identity_margin"]))
    n_neg_per_pos = int(resolve("n_neg_per_pos", args.n_neg_per_pos, cfg, _DEFAULTS["n_neg_per_pos"]))
    max_pos_per_anchor = int(resolve("max_pos_per_anchor", args.max_pos_per_anchor, cfg, _DEFAULTS["max_pos_per_anchor"]))
    max_neg_per_anchor = int(resolve("max_neg_per_anchor", args.max_neg_per_anchor, cfg, _DEFAULTS["max_neg_per_anchor"]))
    max_triplets_per_anchor = int(resolve("max_triplets_per_anchor", args.max_triplets_per_anchor, cfg, _DEFAULTS["max_triplets_per_anchor"]))
    min_triplets_per_anchor = int(resolve("min_triplets_per_anchor", args.min_triplets_per_anchor, cfg, _DEFAULTS["min_triplets_per_anchor"]))
    n_seeds = int(resolve("n_seeds", args.n_seeds, cfg, _DEFAULTS["n_seeds"]))
    baseline_name = resolve("baseline_name", args.baseline_name, cfg, _DEFAULTS["baseline_name"])
    recompute_match = bool(resolve("recompute_match", args.recompute_match, cfg, _DEFAULTS["recompute_match"]))
    chunk_size = int(resolve("chunk_size", args.chunk_size, cfg, _DEFAULTS["chunk_size"]))
    save_triplets = bool(resolve("save_triplets", args.save_triplets, cfg, _DEFAULTS["save_triplets"]))
    max_anchors = resolve("max_anchors", args.max_anchors, cfg, _DEFAULTS["max_anchors"])
    max_anchors = None if max_anchors is None else int(max_anchors)

    if ec_level not in (1, 2, 3, 4):
        ap.error(f"非法 ec_level: {ec_level}")
    if match_mode not in VALID_MATCH_MODES:
        ap.error(f"非法 match_mode: {match_mode}")
    if negative_mode not in _VALID_NEGATIVE_MODES:
        ap.error(f"非法 negative_mode: {negative_mode}")
    if negative_mode != "general" and ec_level != 4:
        ap.error("hard_ec1/2/3 negative_mode 建议且当前要求 ec_level=4，因为需要定义完整 EC4 不共享。")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "hcft_eval.log")
    log_fh = open(log_path, "w")

    def log(msg=""):
        line = str(msg)
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    emb_names = list(emb_dirs.keys())
    seeds = list(range(n_seeds))

    try:
        log("=" * 70)
        log(f" hcft_eval  |  {datetime.now()}")
        log("=" * 70)
        if args.config:
            log(f" config                 : {args.config}")
        log(f" embeddings             : {emb_names}")
        log(f" identity_pairs         : {identity_pairs}")
        log(f" label_csv              : {label_csv}")
        log(f" output_dir             : {output_dir}")
        log(f" ec_level               : {ec_level}")
        log(f" match_mode             : {match_mode}")
        log(f" negative_mode          : {negative_mode}")
        log(f" positive_max_identity  : {positive_max_identity}")
        log(f" negative_min_identity  : {negative_min_identity}")
        log(f" min_identity_margin    : {min_identity_margin}")
        log(f" n_neg_per_pos          : {n_neg_per_pos}")
        log(f" max_pos_per_anchor     : {max_pos_per_anchor}")
        log(f" max_neg_per_anchor     : {max_neg_per_anchor}")
        log(f" max_triplets_per_anchor: {max_triplets_per_anchor}")
        log(f" min_triplets_per_anchor: {min_triplets_per_anchor}")
        log(f" n_seeds                : {n_seeds} (seeds={seeds})")
        log(f" recompute_match        : {recompute_match}")
        log(f" save_triplets          : {save_triplets}")
        if max_anchors is not None:
            log(f" max_anchors            : {max_anchors}")

        # ---- 载入 pair，展开 directed graph ----
        log("\n[pairs] 加载 pair_identity.csv")
        pair_df = load_pairs_for_hcft(
            identity_pairs, ec_level, match_mode, label_csv,
            recompute_match, negative_mode, log,
        )

        log("\n[graph] 展开 undirected pairs 为 directed anchor-candidate edges")
        directed = make_directed_edges(pair_df)
        log(f"  directed edges: {len(directed)}; anchors: {directed['anchor'].nunique()}")

        triplet_dir = os.path.join(output_dir, "triplets") if save_triplets else None
        scored_dir = os.path.join(output_dir, "scored_triplets") if save_triplets else None
        if save_triplets:
            os.makedirs(triplet_dir, exist_ok=True)
            os.makedirs(scored_dir, exist_ok=True)

        all_anchor_rows, all_seed_rows = [], []

        # ---- 每个 seed 先采样 triplets，再对所有模型复用 ----
        for sd in seeds:
            log(f"\n[triplets] Sampling seed={sd}")
            triplets = sample_triplets_for_seed(
                directed=directed,
                seed=sd,
                negative_mode=negative_mode,
                positive_max_identity=positive_max_identity,
                negative_min_identity=negative_min_identity,
                min_identity_margin=min_identity_margin,
                n_neg_per_pos=n_neg_per_pos,
                max_pos_per_anchor=max_pos_per_anchor,
                max_neg_per_anchor=max_neg_per_anchor,
                max_triplets_per_anchor=max_triplets_per_anchor,
                min_triplets_per_anchor=min_triplets_per_anchor,
                max_anchors=max_anchors,
                log=log,
            )

            if triplets.empty:
                log(f"  seed={sd}: 没有构造出有效 triplets，跳过该 seed。")
                continue
            if save_triplets:
                triplets.to_csv(os.path.join(triplet_dir, f"sampled_triplets_seed{sd}.csv"), index=False)

            log(f"\n[models] Evaluate seed={sd}")
            for name, emb_dir in emb_dirs.items():
                anchor_rows, seed_rows, _ = eval_one_model_on_triplets(
                    emb_name=name,
                    emb_dir=emb_dir,
                    triplets=triplets,
                    seed=sd,
                    chunk_size=chunk_size,
                    min_triplets_per_anchor=min_triplets_per_anchor,
                    save_scored=save_triplets,
                    scored_dir=scored_dir,
                    log=log,
                )
                all_anchor_rows.extend(anchor_rows)
                all_seed_rows.extend(seed_rows)
                if seed_rows:
                    r = seed_rows[0]
                    log(f"    {name}: HCFT-Acc={r['hcft_acc_anchor_mean']:.4f}, "
                        f"Margin={r['hcft_margin_anchor_mean']:+.4f}, "
                        f"anchors={r['n_anchors']}, triplets={r['n_triplets_total']}")
                else:
                    log(f"    {name}: no valid anchors after embedding filtering")

        # ---- 输出 ----
        anchor_df = pd.DataFrame(all_anchor_rows)
        seed_df = pd.DataFrame(all_seed_rows)
        summary_df = summarize_across_seeds(seed_df)

        anchor_csv = os.path.join(output_dir, "hcft_anchor_summary.csv")
        seed_csv = os.path.join(output_dir, "hcft_seed_summary.csv")
        summary_csv = os.path.join(output_dir, "hcft_summary.csv")
        anchor_df.to_csv(anchor_csv, index=False)
        seed_df.to_csv(seed_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)

        baseline = pick_baseline(emb_names, baseline_name)
        delta_df = compute_delta(seed_df, baseline) if baseline else pd.DataFrame()
        delta_csv = os.path.join(output_dir, "delta_hcft_summary.csv")
        delta_df.to_csv(delta_csv, index=False)

        log("\n" + "=" * 70)
        log("HCFT summary across seeds:")
        if summary_df.empty:
            log("  No valid HCFT results.")
        else:
            for r in summary_df.itertuples(index=False):
                log(f"  {r.embedding}: Acc={r.hcft_acc_mean:.4f} ± {r.hcft_acc_std:.4f}; "
                    f"Margin={r.hcft_margin_mean:+.4f} ± {r.hcft_margin_std:.4f}; "
                    f"anchors≈{r.n_anchors_mean:.1f}; triplets≈{r.n_triplets_total_mean:.1f}")

        if baseline:
            log(f"\n[delta] baseline = {baseline}")
        else:
            log("\n[delta] 未找到 baseline，跳过 delta 解读。")

        config = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config_file": args.config,
            "embedding_dirs": emb_dirs,
            "identity_pairs": identity_pairs,
            "label_csv": label_csv,
            "output_dir": output_dir,
            "ec_level": ec_level,
            "match_mode": match_mode,
            "negative_mode": negative_mode,
            "positive_max_identity": positive_max_identity,
            "negative_min_identity": negative_min_identity,
            "min_identity_margin": min_identity_margin,
            "n_neg_per_pos": n_neg_per_pos,
            "max_pos_per_anchor": max_pos_per_anchor,
            "max_neg_per_anchor": max_neg_per_anchor,
            "max_triplets_per_anchor": max_triplets_per_anchor,
            "min_triplets_per_anchor": min_triplets_per_anchor,
            "n_seeds": n_seeds,
            "seeds": seeds,
            "baseline": baseline,
            "recompute_match": recompute_match,
            "chunk_size": chunk_size,
            "save_triplets": save_triplets,
            "max_anchors": max_anchors,
            "outputs": {
                "hcft_anchor_summary": anchor_csv,
                "hcft_seed_summary": seed_csv,
                "hcft_summary": summary_csv,
                "delta_hcft_summary": delta_csv,
            },
        }
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        log(f"\n[done] → {output_dir}")
        log(f"  {anchor_csv}")
        log(f"  {seed_csv}")
        log(f"  {summary_csv}")
        log(f"  {delta_csv}")

    finally:
        log_fh.close()

if __name__ == "__main__":
    main()
