#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_substrate_coherence.py
为 3_analyze_substrate_neighborhood_coherence.py 的输出生成诊断图。

设计原则 (按 spec §15)
-----------------------
- 绘图逻辑与核心计算逻辑完全分离：本脚本只读取主分析写出的两个 CSV
  (per_query_coherence.csv, paired_comparison_summary.csv)，不做任何近邻检索或
  统计推断，只把已算好的数值画出来。
- 当前阶段以「可审核」为先，不追求论文级排版；每张图同时导出 PNG 和 PDF。
- 所有聚合统计单位是 query enzyme：mean coherence 用 per-query 行求均值，
  误差棒用 query 自助法 (bootstrap) 95% CI，与主分析口径一致。

生成的图 (在 --plot-dir 下)
---------------------------
1. mean_coherence_by_model        各模型 mean coherence/enrichment + 95% bootstrap CI
2. paired_delta_distribution      Base-SUB 减 Base 的 per-query delta 分布 (主比较)
3. sensitivity_over_k             不同 k 下 mean delta 走势
4. sensitivity_over_identity      不同 identity cutoff 下 mean delta 走势
5. filtered_vs_unfiltered         代谢物 filtered / unfiltered 对比
6. csym_vs_cmax                   C_sym / C_max 主指标 vs 敏感性指标对比
7. coherence_vs_neighbor_identity coherence 与近邻序列同源性的关系 (同源混杂诊断)
8. substrate_set_size_groups      不同底物集合大小分组的结果

用法
----
python plot_substrate_coherence.py \
    --analysis-dir /path/to/substrate_coherence/esm2_650m \
    --plot-dir     /path/to/substrate_coherence/esm2_650m/plots
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_subcoh")

# 默认聚焦条件，保证每张图维度可控；可用 CLI 覆盖
DEFAULT_METRIC = "c_sym"
DEFAULT_FILTER = "unfiltered"

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def save_fig(fig, plot_dir: str, name: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(plot_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s.{png,pdf}", name)

def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, seed: int = 2025
                      ) -> Tuple[float, float, float]:
    """以样本 (query) 为单位的均值 95% 自助置信区间。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def split_comparison(comparison: str) -> Tuple[str, str]:
    """主分析把 comparison 标签写为 'name_a__vs__name_b'。"""
    if "__vs__" in comparison:
        a, b = comparison.split("__vs__", 1)
    elif ":" in comparison:
        a, b = comparison.split(":", 1)
    else:
        raise ValueError(f"cannot split comparison label: {comparison}")
    return a, b

def primary_comparison(df: pd.DataFrame) -> Optional[str]:
    """优先选 comparison_type == primary 的比较；否则取第一个。"""
    if "comparison_type" in df.columns:
        prim = df.loc[df["comparison_type"] == "primary", "comparison"].unique()
        if len(prim):
            return prim[0]
    comps = df["comparison"].unique()
    return comps[0] if len(comps) else None

# ---------------------------------------------------------------------------
# 各图
# ---------------------------------------------------------------------------
def plot_mean_coherence_by_model(pq: pd.DataFrame, plot_dir: str, k: int,
                                 metric: str, met_filter: str, cond: str) -> None:
    sub = pq[(pq["k"] == k) & (pq["set_similarity_metric"] == metric)
             & (pq["metabolite_filter"] == met_filter)
             & (pq["homology_condition"] == cond)]
    if sub.empty:
        logger.warning("mean_coherence_by_model: no rows for k=%s metric=%s", k, metric); return
    models = sorted(sub["model"].unique())
    coh_m, coh_lo, coh_hi, enr_m = [], [], [], []
    for m in models:
        v = sub[sub["model"] == m]
        mc, lo, hi = bootstrap_ci_mean(v["coherence"].values)
        coh_m.append(mc); coh_lo.append(mc - lo); coh_hi.append(hi - mc)
        enr_m.append(np.nanmean(v["enrichment"].values) if "enrichment" in v else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(7, len(models) * 1.4), 4.2))
    x = np.arange(len(models))
    ax1.bar(x, coh_m, yerr=[coh_lo, coh_hi], capsize=4, color="#4C72B0")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax1.set_ylabel("mean coherence (95% CI)"); ax1.set_title(f"Coherence  k={k}  {metric}")
    ax2.bar(x, enr_m, color="#55A868")
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("mean enrichment"); ax2.set_title("Enrichment vs random")
    ax2.axhline(0, color="k", lw=0.8)
    fig.suptitle(f"filter={met_filter}  homology={cond}", fontsize=9)
    save_fig(fig, plot_dir, "mean_coherence_by_model")

def plot_paired_delta_distribution(pq: pd.DataFrame, plot_dir: str, comparison: str,
                                   k: int, metric: str, met_filter: str, cond: str) -> None:
    sub = pq[(pq["comparison"] == comparison) & (pq["k"] == k)
             & (pq["set_similarity_metric"] == metric)
             & (pq["metabolite_filter"] == met_filter)
             & (pq["homology_condition"] == cond)]
    if sub.empty:
        logger.warning("paired_delta: no rows"); return
    models = sub["model"].unique()
    if len(models) != 2:
        logger.warning("paired_delta: expected 2 models, got %s", list(models)); return
    a, b = split_comparison(comparison)
    pa = sub[sub["model"] == a].set_index("query_id")["coherence"]
    pb = sub[sub["model"] == b].set_index("query_id")["coherence"]
    common = pa.index.intersection(pb.index)
    delta = (pb.loc[common] - pa.loc[common]).values
    if delta.size == 0:
        logger.warning("paired_delta: no common queries"); return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delta, bins=40, color="#C44E52", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.mean(delta), color="navy", ls="--", lw=1.2,
               label=f"mean={np.mean(delta):.4f}")
    frac = float((delta > 0).mean())
    ax.set_xlabel(f"per-query delta  ({b} - {a})")
    ax.set_ylabel("number of query enzymes")
    ax.set_title(f"Paired delta  k={k}  {metric}\nfrac improved={frac:.2f}, n={delta.size}")
    ax.legend(fontsize=8)
    save_fig(fig, plot_dir, "paired_delta_distribution")

def _delta_line(ax, summ: pd.DataFrame, xcol: str, label_prefix: str = "") -> None:
    summ = summ.sort_values(xcol)
    ax.errorbar(summ[xcol], summ["mean_delta"],
                yerr=[summ["mean_delta"] - summ["bootstrap_ci_low"],
                      summ["bootstrap_ci_high"] - summ["mean_delta"]],
                marker="o", capsize=4, label=label_prefix or None)
    ax.axhline(0, color="k", lw=0.8)

def plot_sensitivity_over_k(summ: pd.DataFrame, plot_dir: str, comparison: str,
                            metric: str, met_filter: str, cond: str) -> None:
    sub = summ[(summ["comparison"] == comparison) & (summ["set_similarity_metric"] == metric)
               & (summ["metabolite_filter"] == met_filter)
               & (summ["homology_condition"] == cond)]
    if sub.empty:
        logger.warning("sensitivity_over_k: no rows"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    _delta_line(ax, sub, "k")
    ax.set_xlabel("k (neighbours)"); ax.set_ylabel("mean delta (95% CI)")
    ax.set_title(f"Sensitivity over k  {metric}\n{comparison}")
    save_fig(fig, plot_dir, "sensitivity_over_k")

def plot_sensitivity_over_identity(summ: pd.DataFrame, plot_dir: str, comparison: str,
                                   k: int, metric: str, met_filter: str) -> None:
    sub = summ[(summ["comparison"] == comparison) & (summ["k"] == k)
               & (summ["set_similarity_metric"] == metric)
               & (summ["metabolite_filter"] == met_filter)]
    if sub.empty:
        logger.warning("sensitivity_over_identity: no rows"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    # unrestricted 没有 cutoff 数值，单独画一条参考线
    restr = sub[sub["homology_condition"] != "unrestricted"]
    if not restr.empty:
        _delta_line(ax, restr, "identity_cutoff")
    unr = sub[sub["homology_condition"] == "unrestricted"]
    if not unr.empty:
        ax.axhline(float(unr["mean_delta"].iloc[0]), color="gray", ls=":",
                   label="unrestricted")
    ax.set_xlabel("identity cutoff (exclude neighbours >= cutoff)")
    ax.set_ylabel("mean delta (95% CI)")
    ax.set_title(f"Sensitivity over homology cutoff  k={k}  {metric}")
    ax.legend(fontsize=8)
    save_fig(fig, plot_dir, "sensitivity_over_identity")

def plot_filtered_vs_unfiltered(summ: pd.DataFrame, plot_dir: str, comparison: str,
                                k: int, metric: str, cond: str) -> None:
    sub = summ[(summ["comparison"] == comparison) & (summ["k"] == k)
               & (summ["set_similarity_metric"] == metric)
               & (summ["homology_condition"] == cond)]
    if sub.empty or sub["metabolite_filter"].nunique() < 2:
        logger.warning("filtered_vs_unfiltered: need both filters"); return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    order = ["unfiltered", "filtered"]
    present = [f for f in order if f in sub["metabolite_filter"].values]
    vals = [float(sub[sub["metabolite_filter"] == f]["mean_delta"].iloc[0]) for f in present]
    los = [float(sub[sub["metabolite_filter"] == f]["bootstrap_ci_low"].iloc[0]) for f in present]
    his = [float(sub[sub["metabolite_filter"] == f]["bootstrap_ci_high"].iloc[0]) for f in present]
    x = np.arange(len(present))
    ax.bar(x, vals, yerr=[np.array(vals) - los, np.array(his) - vals],
           capsize=5, color=["#8172B3", "#CCB974"][:len(present)])
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean delta (95% CI)")
    ax.set_title(f"Metabolite filter effect  k={k}  {metric}")
    save_fig(fig, plot_dir, "filtered_vs_unfiltered")

def plot_csym_vs_cmax(summ: pd.DataFrame, plot_dir: str, comparison: str,
                      k: int, met_filter: str, cond: str) -> None:
    sub = summ[(summ["comparison"] == comparison) & (summ["k"] == k)
               & (summ["metabolite_filter"] == met_filter)
               & (summ["homology_condition"] == cond)]
    metrics = [m for m in ("c_sym", "c_max") if m in sub["set_similarity_metric"].values]
    if len(metrics) < 2:
        logger.warning("csym_vs_cmax: need both metrics"); return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vals = [float(sub[sub["set_similarity_metric"] == m]["mean_delta"].iloc[0]) for m in metrics]
    los = [float(sub[sub["set_similarity_metric"] == m]["bootstrap_ci_low"].iloc[0]) for m in metrics]
    his = [float(sub[sub["set_similarity_metric"] == m]["bootstrap_ci_high"].iloc[0]) for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x, vals, yerr=[np.array(vals) - los, np.array(his) - vals],
           capsize=5, color=["#4C72B0", "#DD8452"])
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean delta (95% CI)")
    ax.set_title(f"Primary vs sensitivity metric  k={k}")
    save_fig(fig, plot_dir, "csym_vs_cmax")

def plot_coherence_vs_neighbor_identity(pq: pd.DataFrame, plot_dir: str, comparison: str,
                                        k: int, metric: str, met_filter: str) -> None:
    # 用 unrestricted 条件，这样近邻同源性有完整范围
    sub = pq[(pq["comparison"] == comparison) & (pq["k"] == k)
             & (pq["set_similarity_metric"] == metric)
             & (pq["metabolite_filter"] == met_filter)
             & (pq["homology_condition"] == "unrestricted")]
    sub = sub.dropna(subset=["mean_neighbor_sequence_identity"])
    if sub.empty:
        logger.warning("coherence_vs_neighbor_identity: no rows (need homology table)"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, color in zip(sub["model"].unique(), ["#4C72B0", "#C44E52", "#55A868"]):
        v = sub[sub["model"] == m]
        ax.scatter(v["mean_neighbor_sequence_identity"], v["coherence"],
                   s=10, alpha=0.5, color=color, label=m)
    ax.set_xlabel("mean neighbour sequence identity")
    ax.set_ylabel("per-query coherence")
    ax.set_title(f"Coherence vs neighbour homology  k={k}  {metric}")
    ax.legend(fontsize=7)
    save_fig(fig, plot_dir, "coherence_vs_neighbor_identity")

def plot_substrate_set_size_groups(pq: pd.DataFrame, plot_dir: str, comparison: str,
                                   k: int, metric: str, met_filter: str, cond: str) -> None:
    sub = pq[(pq["comparison"] == comparison) & (pq["k"] == k)
             & (pq["set_similarity_metric"] == metric)
             & (pq["metabolite_filter"] == met_filter)
             & (pq["homology_condition"] == cond)]
    if sub.empty:
        logger.warning("substrate_set_size_groups: no rows"); return
    a, b = split_comparison(comparison)
    groups = ["1", "2-4", ">=5"]
    present = [g for g in groups if g in sub["substrate_count_group"].unique()]
    present += [g for g in sub["substrate_count_group"].unique() if g not in present]
    means, los, his = [], [], []
    for g in present:
        gg = sub[sub["substrate_count_group"] == g]
        pa = gg[gg["model"] == a].set_index("query_id")["coherence"]
        pb = gg[gg["model"] == b].set_index("query_id")["coherence"]
        common = pa.index.intersection(pb.index)
        d = (pb.loc[common] - pa.loc[common]).values
        mc, lo, hi = bootstrap_ci_mean(d)
        means.append(mc); los.append(mc - lo); his.append(hi - mc)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(present))
    ax.bar(x, means, yerr=[los, his], capsize=5, color="#937860")
    ax.set_xticks(x); ax.set_xticklabels(present)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("substrate-set size group")
    ax.set_ylabel("mean delta (95% CI)")
    ax.set_title(f"Delta by substrate-set size  k={k}  {metric}")
    save_fig(fig, plot_dir, "substrate_set_size_groups")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnostic plots for substrate neighbourhood coherence")
    p.add_argument("--analysis-dir", type=str, required=True,
                   help="主分析 --output-dir，需含 per_query_coherence.csv 和 paired_comparison_summary.csv")
    p.add_argument("--plot-dir", type=str, default=None,
                   help="图输出目录 (默认 analysis-dir/plots)")
    p.add_argument("--comparison", type=str, default=None,
                   help="要画的 comparison (默认自动选 primary)")
    p.add_argument("--k", type=int, default=10, help="主图使用的 k (默认 10)")
    p.add_argument("--metric", type=str, default=DEFAULT_METRIC)
    p.add_argument("--metabolite-filter", type=str, default=DEFAULT_FILTER,
                   choices=["unfiltered", "filtered"])
    p.add_argument("--homology-condition", type=str, default="unrestricted")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    pq_path = os.path.join(args.analysis_dir, "per_query_coherence.csv")
    sm_path = os.path.join(args.analysis_dir, "paired_comparison_summary.csv")
    for p in (pq_path, sm_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing analysis output: {p}")
    pq = pd.read_csv(pq_path)
    summ = pd.read_csv(sm_path)
    plot_dir = args.plot_dir or os.path.join(args.analysis_dir, "plots")

    comparison = args.comparison or primary_comparison(summ)
    if comparison is None:
        raise ValueError("no comparison found in summary")
    logger.info("plotting comparison=%s k=%s metric=%s filter=%s homology=%s",
                comparison, args.k, args.metric, args.metabolite_filter, args.homology_condition)

    k, metric, mf, cond = args.k, args.metric, args.metabolite_filter, args.homology_condition
    plot_mean_coherence_by_model(pq, plot_dir, k, metric, mf, cond)
    plot_paired_delta_distribution(pq, plot_dir, comparison, k, metric, mf, cond)
    plot_sensitivity_over_k(summ, plot_dir, comparison, metric, mf, cond)
    plot_sensitivity_over_identity(summ, plot_dir, comparison, k, metric, mf)
    plot_filtered_vs_unfiltered(summ, plot_dir, comparison, k, metric, cond)
    plot_csym_vs_cmax(summ, plot_dir, comparison, k, mf, cond)
    plot_coherence_vs_neighbor_identity(pq, plot_dir, comparison, k, metric, mf)
    plot_substrate_set_size_groups(pq, plot_dir, comparison, k, metric, mf, cond)
    logger.info("All plots written to %s", plot_dir)

if __name__ == "__main__":
    main()