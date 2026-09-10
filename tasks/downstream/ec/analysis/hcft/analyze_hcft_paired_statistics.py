#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_hcft_paired_statistics.py

对 hcft_anchor_summary.csv 中同一 anchor 的两种 embedding 进行配对统计。

主要输出
--------
1. paired_anchor_deltas.csv
   每个 experiment / embedding pair / anchor 的配对差值。

2. paired_statistics.csv
   全局 anchor-level 配对统计：
   - mean paired delta
   - 95% anchor bootstrap CI
   - median paired delta
   - fraction of anchors improved
   - Wilcoxon signed-rank test
   - paired sign-flip permutation test
   - Holm-adjusted p values

3. pairing_diagnostics.csv
   左右模型的 anchor 数量、未配对数量、triplet 支持是否一致等诊断。

4. family_paired_statistics.csv（提供 --family-map-csv 时生成）
   对每个 EC3 family 计算同样的配对统计。支持按 Base 预先筛出的
   family 集合进行限制，避免使用 CPT-SUB 重新选择家族。

统计单位
--------
- 先在 anchor × seed 层面对齐。
- 如果存在多个 seed，再对同一 anchor 的 delta 取均值。
- Bootstrap、Wilcoxon 和 permutation test 均以 unique anchor 为独立单位。
- 不把同一 anchor 下的 triplet 当作独立样本。

示例
----
先查看文件中的 embedding 名称：

python analyze_hcft_paired_statistics.py \
  --experiment 3B_general:/path/to/hcft_anchor_summary.csv \
  --list-embeddings

运行配对分析：

python analyze_hcft_paired_statistics.py \
  --experiment 3B_general:/path/to/hcft_anchor_summary.csv \
  --pair base_to_base_sub:esm2_base:esm2_base_sub \
  --pair cpt_to_cpt_sub:esm2_cpt:esm2_cpt_sub \
  --output-dir /path/to/output

若不同 experiment 使用不同 embedding 名称，可限定 pair 的适用 experiment：

python analyze_hcft_paired_statistics.py \
  --experiment 3B_general:/path/to/file.csv \
  --pair 3B_general@base_to_base_sub:esm2_3b_base:esm2_3b_base_sub
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REQUIRED_COLUMNS = {
    "embedding",
    "anchor",
    "hcft_acc",
    "hcft_margin_mean",
}

@dataclass(frozen=True)
class PairSpec:
    label: str
    left: str
    right: str
    experiment: Optional[str] = None

def parse_named_path(item: str) -> Tuple[str, str]:
    if ":" not in item:
        raise ValueError(
            f"--experiment 格式错误: {item!r}; 应为 NAME:/path/to/file_or_dir"
        )
    name, path = item.split(":", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"--experiment 格式错误: {item!r}")
    return name, path

def resolve_anchor_csv(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        path = os.path.join(path, "hcft_anchor_summary.csv")
    return path

def parse_pair_spec(item: str) -> PairSpec:
    """
    支持:
      LABEL:LEFT:RIGHT
      EXPERIMENT@LABEL:LEFT:RIGHT
    """
    parts = item.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--pair 格式错误: {item!r}; "
            "应为 LABEL:LEFT_EMBEDDING:RIGHT_EMBEDDING，"
            "或 EXPERIMENT@LABEL:LEFT:RIGHT"
        )

    scope_label, left, right = [x.strip() for x in parts]
    if not scope_label or not left or not right:
        raise ValueError(f"--pair 格式错误: {item!r}")

    experiment = None
    label = scope_label
    if "@" in scope_label:
        experiment, label = scope_label.split("@", 1)
        experiment = experiment.strip()
        label = label.strip()
        if not experiment or not label:
            raise ValueError(f"--pair experiment scope 格式错误: {item!r}")

    return PairSpec(
        label=label,
        left=left,
        right=right,
        experiment=experiment,
    )

def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p values，保留 NaN。"""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return out

    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p[idx]
        running = max(running, adjusted)
        out[idx] = min(running, 1.0)
    return out

def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> Tuple[float, float]:
    """
    Nonparametric anchor bootstrap CI for the mean.

    为控制内存，以 batch 方式生成重采样索引。
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)

    if n == 0:
        return np.nan, np.nan
    if n == 1 or n_boot <= 0:
        return float(x[0]), float(x[0])

    boot_means = np.empty(n_boot, dtype=np.float64)
    written = 0

    while written < n_boot:
        b = min(batch_size, n_boot - written)
        idx = rng.integers(0, n, size=(b, n), endpoint=False)
        boot_means[written:written + b] = x[idx].mean(axis=1)
        written += b

    alpha = 1.0 - confidence
    low, high = np.quantile(
        boot_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(low), float(high)

def sign_flip_permutation_pvalue(
    values: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> float:
    """
    Two-sided paired sign-flip permutation test on the mean delta.

    H0: delta 的符号可交换。
    返回 Monte Carlo p value，使用 +1 correction。
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x != 0.0]
    n = len(x)

    if n == 0:
        return 1.0

    observed = abs(float(x.mean()))

    # 小样本时执行精确枚举。
    if n <= 20:
        total = 1 << n
        exceed = 0
        for mask in range(total):
            signs = np.ones(n, dtype=np.float64)
            for i in range(n):
                if (mask >> i) & 1:
                    signs[i] = -1.0
            stat = abs(float(np.mean(signs * x)))
            if stat >= observed - 1e-15:
                exceed += 1
        return float(exceed / total)

    if n_perm <= 0:
        return np.nan

    exceed = 0
    done = 0
    while done < n_perm:
        b = min(batch_size, n_perm - done)
        signs = rng.integers(0, 2, size=(b, n), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        perm_means = np.abs((signs @ x) / n)
        exceed += int(np.sum(perm_means >= observed - 1e-15))
        done += b

    return float((exceed + 1) / (n_perm + 1))

def safe_wilcoxon(values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.all(x == 0):
        return 0.0, 1.0

    try:
        result = wilcoxon(
            x,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan

def load_experiment(name: str, path: str) -> pd.DataFrame:
    csv_path = resolve_anchor_csv(path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"experiment={name!r} 找不到文件: {csv_path}"
        )

    df = pd.read_csv(csv_path, dtype={"anchor": str})
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} 缺少必要列: {sorted(missing)}"
        )

    df["anchor"] = df["anchor"].astype(str)
    df["embedding"] = df["embedding"].astype(str)

    for col in [
        "seed",
        "hcft_acc",
        "hcft_margin_mean",
        "n_triplets",
        "mean_id_pos",
        "mean_id_neg",
        "mean_delta_identity",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "seed" not in df.columns:
        df["seed"] = 0
    df["seed"] = df["seed"].fillna(0).astype(int)

    df["experiment"] = name
    df["source_csv"] = csv_path
    return df

def pair_embeddings(
    df: pd.DataFrame,
    experiment: str,
    pair: PairSpec,
) -> Tuple[pd.DataFrame, dict]:
    left = df[df["embedding"] == pair.left].copy()
    right = df[df["embedding"] == pair.right].copy()

    if left.empty or right.empty:
        available = sorted(df["embedding"].dropna().unique().tolist())
        raise ValueError(
            f"[{experiment}] pair={pair.label!r} 缺少 embedding。"
            f"left={pair.left!r} rows={len(left)}, "
            f"right={pair.right!r} rows={len(right)}。"
            f"可用 embedding={available}"
        )

    # 一个实验目录通常固定 negative_mode；若文件内包含多个 mode，
    # 将其纳入配对键。
    keys = ["anchor", "seed"]
    for col in ["negative_mode"]:
        if col in left.columns and col in right.columns:
            if left[col].nunique(dropna=False) > 1 or right[col].nunique(dropna=False) > 1:
                keys.append(col)

    left_dup = int(left.duplicated(keys).sum())
    right_dup = int(right.duplicated(keys).sum())
    if left_dup or right_dup:
        raise ValueError(
            f"[{experiment}] pair={pair.label}: 配对键 {keys} 非唯一。"
            f"left duplicates={left_dup}, right duplicates={right_dup}。"
            "说明一个文件中可能混入了未纳入配对键的多组配置。"
        )

    metric_cols = ["hcft_acc", "hcft_margin_mean"]
    optional_cols = [
        "n_triplets",
        "mean_id_pos",
        "mean_id_neg",
        "mean_delta_identity",
    ]

    keep_left = keys + metric_cols + [
        c for c in optional_cols if c in left.columns
    ]
    keep_right = keys + metric_cols + [
        c for c in optional_cols if c in right.columns
    ]

    left_small = left[keep_left].rename(
        columns={c: f"{c}_left" for c in keep_left if c not in keys}
    )
    right_small = right[keep_right].rename(
        columns={c: f"{c}_right" for c in keep_right if c not in keys}
    )

    outer = left_small.merge(
        right_small,
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    paired = outer[outer["_merge"] == "both"].copy()
    paired = paired.drop(columns="_merge")

    paired["experiment"] = experiment
    paired["pair"] = pair.label
    paired["left_embedding"] = pair.left
    paired["right_embedding"] = pair.right

    paired["delta_hcft_acc"] = (
        paired["hcft_acc_right"] - paired["hcft_acc_left"]
    )
    paired["delta_hcft_margin_mean"] = (
        paired["hcft_margin_mean_right"]
        - paired["hcft_margin_mean_left"]
    )

    diag = {
        "experiment": experiment,
        "pair": pair.label,
        "left_embedding": pair.left,
        "right_embedding": pair.right,
        "n_left_rows": int(len(left_small)),
        "n_right_rows": int(len(right_small)),
        "n_paired_rows": int(len(paired)),
        "n_left_only": int((outer["_merge"] == "left_only").sum()),
        "n_right_only": int((outer["_merge"] == "right_only").sum()),
        "n_unique_paired_anchors": int(paired["anchor"].nunique()),
    }

    for col in optional_cols:
        lcol = f"{col}_left"
        rcol = f"{col}_right"
        if lcol in paired.columns and rcol in paired.columns:
            if col == "n_triplets":
                mismatch = paired[lcol].fillna(-1) != paired[rcol].fillna(-1)
            else:
                mismatch = ~np.isclose(
                    paired[lcol].to_numpy(dtype=float),
                    paired[rcol].to_numpy(dtype=float),
                    equal_nan=True,
                    rtol=1e-10,
                    atol=1e-12,
                )
            diag[f"n_{col}_mismatch"] = int(np.sum(mismatch))

    return paired, diag

def aggregate_to_anchor(paired_rows: pd.DataFrame) -> pd.DataFrame:
    """
    多 seed 时，先对同一 anchor 的左右值和 delta 取均值，
    使统计推断以 unique anchor 为单位。
    """
    agg_cols = {
        "hcft_acc_left": "mean",
        "hcft_acc_right": "mean",
        "delta_hcft_acc": "mean",
        "hcft_margin_mean_left": "mean",
        "hcft_margin_mean_right": "mean",
        "delta_hcft_margin_mean": "mean",
        "seed": "nunique",
    }

    for col in [
        "n_triplets_left",
        "n_triplets_right",
        "mean_id_pos_left",
        "mean_id_pos_right",
        "mean_id_neg_left",
        "mean_id_neg_right",
        "mean_delta_identity_left",
        "mean_delta_identity_right",
    ]:
        if col in paired_rows.columns:
            agg_cols[col] = "mean"

    out = (
        paired_rows.groupby(
            [
                "experiment",
                "pair",
                "left_embedding",
                "right_embedding",
                "anchor",
            ],
            as_index=False,
            sort=False,
        )
        .agg(agg_cols)
        .rename(columns={"seed": "n_seeds_paired"})
    )
    return out

def summarize_metric(
    anchor_df: pd.DataFrame,
    metric: str,
    n_boot: int,
    n_perm: int,
    confidence: float,
    seed: int,
) -> dict:
    left_col = f"{metric}_left"
    right_col = f"{metric}_right"
    delta_col = f"delta_{metric}"

    clean = anchor_df[
        [left_col, right_col, delta_col]
    ].replace([np.inf, -np.inf], np.nan).dropna()

    left = clean[left_col].to_numpy(dtype=float)
    right = clean[right_col].to_numpy(dtype=float)
    delta = clean[delta_col].to_numpy(dtype=float)

    rng_boot = np.random.default_rng(seed)
    rng_perm = np.random.default_rng(seed + 104729)

    ci_low, ci_high = bootstrap_mean_ci(
        delta,
        n_boot=n_boot,
        confidence=confidence,
        rng=rng_boot,
    )
    w_stat, w_p = safe_wilcoxon(delta)
    perm_p = sign_flip_permutation_pvalue(
        delta,
        n_perm=n_perm,
        rng=rng_perm,
    )

    return {
        "metric": metric,
        "n_anchors": int(len(delta)),
        "mean_left": float(np.mean(left)) if len(left) else np.nan,
        "mean_right": float(np.mean(right)) if len(right) else np.nan,
        "mean_delta": float(np.mean(delta)) if len(delta) else np.nan,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_confidence": confidence,
        "median_delta": float(np.median(delta)) if len(delta) else np.nan,
        "fraction_improved": float(np.mean(delta > 0)) if len(delta) else np.nan,
        "fraction_tied": float(np.mean(delta == 0)) if len(delta) else np.nan,
        "fraction_worsened": float(np.mean(delta < 0)) if len(delta) else np.nan,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p": w_p,
        "permutation_p": perm_p,
    }

def add_holm_global(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    out["wilcoxon_p_holm"] = np.nan
    out["permutation_p_holm"] = np.nan

    # 每个 metric × test family 内，对所有 experiment × pair 比较校正。
    for metric, idx in out.groupby("metric").groups.items():
        idx = list(idx)
        out.loc[idx, "wilcoxon_p_holm"] = holm_adjust(
            out.loc[idx, "wilcoxon_p"].to_numpy(dtype=float)
        )
        out.loc[idx, "permutation_p_holm"] = holm_adjust(
            out.loc[idx, "permutation_p"].to_numpy(dtype=float)
        )
    return out

def load_family_map(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"anchor": str})
    required = {"anchor", "ec3_family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"family map 缺少列: {sorted(missing)}"
        )

    keep = ["anchor", "ec3_family"]
    if "anchor_ec4" in df.columns:
        keep.append("anchor_ec4")

    out = df[keep].dropna(subset=["anchor", "ec3_family"]).copy()
    out["anchor"] = out["anchor"].astype(str)
    out["ec3_family"] = out["ec3_family"].astype(str)

    # 同一 anchor 应只对应一个 family。
    conflicts = (
        out.groupby("anchor")["ec3_family"].nunique()
        .loc[lambda s: s > 1]
    )
    if len(conflicts):
        raise ValueError(
            f"family map 中有 {len(conflicts)} 个 anchor 对应多个 EC3 family。"
        )

    return out.drop_duplicates("anchor")

def load_selected_families(path: str) -> set:
    df = pd.read_csv(path)
    if "ec3_family" not in df.columns:
        raise ValueError(
            f"{path} 缺少 ec3_family 列。"
        )
    return set(df["ec3_family"].dropna().astype(str))

def summarize_families(
    anchor_df: pd.DataFrame,
    n_boot: int,
    n_perm: int,
    confidence: float,
    seed: int,
    min_anchors: int,
    min_ec4: int,
    min_triplets: int,
    selected_families: Optional[set],
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "experiment",
        "pair",
        "left_embedding",
        "right_embedding",
        "ec3_family",
    ]

    for gi, (keys, g) in enumerate(anchor_df.groupby(group_cols, sort=False)):
        experiment, pair, left_emb, right_emb, family = keys

        if selected_families is not None and family not in selected_families:
            continue

        n_anchors = int(g["anchor"].nunique())
        n_ec4 = (
            int(g["anchor_ec4"].nunique())
            if "anchor_ec4" in g.columns
            else np.nan
        )
        triplet_total = (
            float(g["n_triplets_left"].sum())
            if "n_triplets_left" in g.columns
            else np.nan
        )

        passes = n_anchors >= min_anchors
        if np.isfinite(n_ec4):
            passes = passes and n_ec4 >= min_ec4
        if np.isfinite(triplet_total):
            passes = passes and triplet_total >= min_triplets

        if not passes:
            continue

        for mi, metric in enumerate(["hcft_acc", "hcft_margin_mean"]):
            stat = summarize_metric(
                g,
                metric=metric,
                n_boot=n_boot,
                n_perm=n_perm,
                confidence=confidence,
                seed=seed + gi * 1009 + mi * 7919,
            )
            stat.update({
                "experiment": experiment,
                "pair": pair,
                "left_embedding": left_emb,
                "right_embedding": right_emb,
                "ec3_family": family,
                "n_anchor_ec4": n_ec4,
                "n_triplets_left_total": triplet_total,
            })
            rows.append(stat)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["wilcoxon_p_holm"] = np.nan
    out["permutation_p_holm"] = np.nan

    # 对每个 experiment × pair × metric 内的 family tests 做 Holm 校正。
    group = ["experiment", "pair", "metric"]
    for _, idx in out.groupby(group).groups.items():
        idx = list(idx)
        out.loc[idx, "wilcoxon_p_holm"] = holm_adjust(
            out.loc[idx, "wilcoxon_p"].to_numpy(dtype=float)
        )
        out.loc[idx, "permutation_p_holm"] = holm_adjust(
            out.loc[idx, "permutation_p"].to_numpy(dtype=float)
        )

    return out

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Paired anchor-level statistics for HCFT embedding comparisons."
        )
    )
    ap.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Repeatable NAME:/path/to/hcft_anchor_summary.csv or experiment dir.",
    )
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "Repeatable LABEL:LEFT:RIGHT, or "
            "EXPERIMENT@LABEL:LEFT:RIGHT."
        ),
    )
    ap.add_argument(
        "--list-embeddings",
        action="store_true",
        help="Only list available embeddings in each experiment and exit.",
    )
    ap.add_argument("--output-dir", default="./hcft_paired_statistics")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260717)

    ap.add_argument(
        "--family-map-csv",
        default=None,
        help=(
            "Optional CSV with anchor, ec3_family and optionally anchor_ec4. "
            "可直接使用 summarize_hcft_by_ec_family.py 输出的 "
            "anchor_with_family.csv。"
        ),
    )
    ap.add_argument(
        "--selected-families-csv",
        default=None,
        help=(
            "Optional Base-defined family list with ec3_family column, "
            "例如 Base 的 family_candidates.csv。"
        ),
    )
    ap.add_argument("--min-family-anchors", type=int, default=30)
    ap.add_argument("--min-family-ec4", type=int, default=3)
    ap.add_argument("--min-family-triplets", type=int, default=500)
    ap.add_argument("--family-bootstrap", type=int, default=2000)
    ap.add_argument("--family-permutations", type=int, default=5000)

    args = ap.parse_args()

    experiments: Dict[str, str] = {}
    for item in args.experiment:
        name, path = parse_named_path(item)
        if name in experiments:
            raise SystemExit(f"重复 experiment 名称: {name}")
        experiments[name] = path

    loaded = {
        name: load_experiment(name, path)
        for name, path in experiments.items()
    }

    if args.list_embeddings:
        for name, df in loaded.items():
            print(f"\n[{name}] {df['source_csv'].iloc[0]}")
            counts = df["embedding"].value_counts()
            print(counts.to_string())
        return

    if not args.pair:
        raise SystemExit(
            "未提供 --pair。请先使用 --list-embeddings 查看 embedding 名称。"
        )

    pair_specs = [parse_pair_spec(x) for x in args.pair]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paired_rows = []
    diagnostics = []

    for experiment, df in loaded.items():
        applicable = [
            p for p in pair_specs
            if p.experiment is None or p.experiment == experiment
        ]
        if not applicable:
            print(f"[skip] {experiment}: no applicable --pair")
            continue

        for pair in applicable:
            paired, diag = pair_embeddings(
                df=df,
                experiment=experiment,
                pair=pair,
            )
            all_paired_rows.append(paired)
            diagnostics.append(diag)

            print(
                f"[pair] {experiment} | {pair.label}: "
                f"rows={len(paired):,}, "
                f"anchors={paired['anchor'].nunique():,}"
            )

    if not all_paired_rows:
        raise SystemExit("没有产生任何配对结果。")

    paired_rows = pd.concat(all_paired_rows, ignore_index=True)
    paired_rows.to_csv(
        output_dir / "paired_anchor_seed_rows.csv",
        index=False,
    )

    anchor_df = aggregate_to_anchor(paired_rows)

    family_map = None
    if args.family_map_csv:
        family_map = load_family_map(args.family_map_csv)
        anchor_df = anchor_df.merge(
            family_map,
            on="anchor",
            how="left",
            validate="many_to_one",
        )

    anchor_df.to_csv(
        output_dir / "paired_anchor_deltas.csv",
        index=False,
    )

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(
        output_dir / "pairing_diagnostics.csv",
        index=False,
    )

    stats_rows = []
    for gi, (keys, g) in enumerate(
        anchor_df.groupby(
            ["experiment", "pair", "left_embedding", "right_embedding"],
            sort=False,
        )
    ):
        experiment, pair, left_emb, right_emb = keys

        for mi, metric in enumerate(["hcft_acc", "hcft_margin_mean"]):
            stat = summarize_metric(
                g,
                metric=metric,
                n_boot=args.bootstrap,
                n_perm=args.permutations,
                confidence=args.confidence,
                seed=args.seed + gi * 1009 + mi * 7919,
            )
            stat.update({
                "experiment": experiment,
                "pair": pair,
                "left_embedding": left_emb,
                "right_embedding": right_emb,
                "n_paired_seed_rows": int(
                    paired_rows[
                        (paired_rows["experiment"] == experiment)
                        & (paired_rows["pair"] == pair)
                    ].shape[0]
                ),
            })
            stats_rows.append(stat)

    stats_df = add_holm_global(pd.DataFrame(stats_rows))
    stats_df.to_csv(
        output_dir / "paired_statistics.csv",
        index=False,
    )

    if family_map is not None:
        selected_families = (
            load_selected_families(args.selected_families_csv)
            if args.selected_families_csv
            else None
        )

        family_anchor = anchor_df.dropna(subset=["ec3_family"]).copy()
        family_stats = summarize_families(
            anchor_df=family_anchor,
            n_boot=args.family_bootstrap,
            n_perm=args.family_permutations,
            confidence=args.confidence,
            seed=args.seed + 99991,
            min_anchors=args.min_family_anchors,
            min_ec4=args.min_family_ec4,
            min_triplets=args.min_family_triplets,
            selected_families=selected_families,
        )
        family_stats.to_csv(
            output_dir / "family_paired_statistics.csv",
            index=False,
        )

    print("\n[outputs]")
    print(output_dir / "paired_anchor_seed_rows.csv")
    print(output_dir / "paired_anchor_deltas.csv")
    print(output_dir / "paired_statistics.csv")
    print(output_dir / "pairing_diagnostics.csv")
    if family_map is not None:
        print(output_dir / "family_paired_statistics.csv")

    print("\n[global paired statistics]")
    show_cols = [
        "experiment",
        "pair",
        "metric",
        "n_anchors",
        "mean_left",
        "mean_right",
        "mean_delta",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "median_delta",
        "fraction_improved",
        "wilcoxon_p",
        "wilcoxon_p_holm",
        "permutation_p",
        "permutation_p_holm",
    ]
    print(stats_df[show_cols].to_string(index=False))

if __name__ == "__main__":
    main()