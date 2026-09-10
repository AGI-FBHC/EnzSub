#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_substrate_coherence_paper_main.py

Paper-ready main panels for substrate-neighbourhood coherence analysis.

Input files from 3_analyze_substrate_neighborhood_coherence.py:
  1. per_query_coherence.csv
  2. paired_comparison_summary.csv

Main outputs:
  1. paper_mean_delta_by_backbone.pdf/png/svg
     Point estimate + query-level bootstrap 95% CI for mean Δcoherence.

  2. paper_per_query_delta_distribution.pdf/png/svg
     Violin + boxplot distribution of per-query Δcoherence.

  3. paper_fraction_improved_by_backbone.pdf/png/svg
     Fraction of query enzymes with positive Δcoherence.

  4. paper_main_summary.csv
     Numeric values used for plotting.

  5. paper_per_query_delta.csv
     Per-query paired deltas.

Example:
python plot_substrate_coherence_paper_main.py \
  --analysis-dirs \
    ProtBERT-BFD:/path/to/protbert_bfd/analysis \
    ESM2-650M:/path/to/esm2_650m/analysis \
    ESM2-3B:/path/to/esm2_3b/analysis \
  --plot-dir /path/to/paper_plots \
  --k 10 \
  --metric c_sym \
  --metabolite-filter unfiltered \
  --homology-condition unrestricted
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# User-adjustable style configuration
# =============================================================================

BACKBONE_ORDER = [
    "ProtBERT-BFD",
    "ESM2-650M",
    "ESM2-3B",
]

BACKBONE_COLORS = {
    "ProtBERT-BFD": "#8A8A8A",
    "ESM2-650M": "#6A8CAF",
    "ESM2-3B": "#9A7BB5",
}

BACKBONE_DISPLAY_NAMES = {
    "ProtBERT-BFD": "ProtBERT-BFD",
    "ESM2-650M": "ESM2-650M",
    "ESM2-3B": "ESM2-3B",
}

STYLE = {
    "font.family": "Arial",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

FIGSIZE_MEAN_DELTA = (3.0, 2.35)
FIGSIZE_DISTRIBUTION = (3.3, 2.55)
FIGSIZE_FRACTION = (3.0, 2.35)

OUTPUT_EXTENSIONS = ("pdf", "png", "svg")

DEFAULT_METRIC = "c_sym"
DEFAULT_METABOLITE_FILTER = "unfiltered"
DEFAULT_HOMOLOGY_CONDITION = "unrestricted"

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("paper_subcoh")

# =============================================================================
# Utilities
# =============================================================================

def apply_style() -> None:
    for k, v in STYLE.items():
        plt.rcParams[k] = v

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_figure(fig: plt.Figure, plot_dir: str, name: str) -> None:
    ensure_dir(plot_dir)

    for ext in OUTPUT_EXTENSIONS:
        out = os.path.join(plot_dir, f"{name}.{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")

    plt.close(fig)
    logger.info("saved %s.{%s}", name, ",".join(OUTPUT_EXTENSIONS))

def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

def parse_analysis_dirs(items: List[str]) -> Dict[str, str]:
    """
    Parse:
      BackboneLabel:/path/to/analysis_dir
    """
    out = {}

    for item in items:
        if ":" not in item:
            raise ValueError(
                f"Invalid --analysis-dirs item: {item}\n"
                "Expected format: BackboneLabel:/path/to/analysis_dir"
            )

        label, path = item.split(":", 1)
        label = label.strip()
        path = path.strip()

        if not label:
            raise ValueError(f"Empty backbone label in item: {item}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Analysis directory does not exist: {path}")

        out[label] = path

    return out

def split_comparison(comparison: str) -> Tuple[str, str]:
    """
    Upstream comparison labels usually look like:
      model_a__vs__model_b

    Fallback:
      model_a:model_b
    """
    if "__vs__" in comparison:
        a, b = comparison.split("__vs__", 1)
    elif ":" in comparison:
        a, b = comparison.split(":", 1)
    else:
        raise ValueError(
            f"Cannot split comparison label: {comparison}\n"
            "Expected format like model_a__vs__model_b"
        )

    return a, b

def primary_comparison(summary_df: pd.DataFrame) -> Optional[str]:
    """
    Prefer comparison_type == primary.
    Otherwise use the first comparison in the summary file.
    """
    if "comparison" not in summary_df.columns:
        return None

    if "comparison_type" in summary_df.columns:
        x = summary_df.loc[
            summary_df["comparison_type"].astype(str) == "primary",
            "comparison",
        ].dropna().unique()

        if len(x) > 0:
            return str(x[0])

    x = summary_df["comparison"].dropna().unique()
    return str(x[0]) if len(x) > 0 else None

def bootstrap_ci_mean(
    x: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2025,
) -> Tuple[float, float, float]:
    """
    Query-level bootstrap CI for mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)

    return (
        float(np.mean(x)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )

def bootstrap_ci_fraction_positive(
    x: np.ndarray,
    n_boot: int = 5000,
    seed: int = 2025,
) -> Tuple[float, float, float]:
    """
    Query-level bootstrap CI for fraction(delta > 0).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        return np.nan, np.nan, np.nan
    if x.size == 1:
        val = float(x[0] > 0)
        return val, val, val

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    props = (x[idx] > 0).mean(axis=1)

    return (
        float((x > 0).mean()),
        float(np.percentile(props, 2.5)),
        float(np.percentile(props, 97.5)),
    )

def load_one_backbone(backbone: str, analysis_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pq_path = os.path.join(analysis_dir, "per_query_coherence.csv")
    sm_path = os.path.join(analysis_dir, "paired_comparison_summary.csv")

    if not os.path.exists(pq_path):
        raise FileNotFoundError(f"Missing file: {pq_path}")
    if not os.path.exists(sm_path):
        raise FileNotFoundError(f"Missing file: {sm_path}")

    pq = pd.read_csv(pq_path)
    summ = pd.read_csv(sm_path)

    pq["backbone"] = backbone
    summ["backbone"] = backbone

    return pq, summ

def load_all_backbones(backbone_dirs: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_pq = []
    all_summ = []

    for backbone, analysis_dir in backbone_dirs.items():
        logger.info("loading %s from %s", backbone, analysis_dir)
        pq, summ = load_one_backbone(backbone, analysis_dir)
        all_pq.append(pq)
        all_summ.append(summ)

    return (
        pd.concat(all_pq, ignore_index=True),
        pd.concat(all_summ, ignore_index=True),
    )

def filter_summary(
    summ: pd.DataFrame,
    comparison: str,
    k: int,
    metric: str,
    metabolite_filter: str,
    homology_condition: str,
) -> pd.DataFrame:
    sub = summ[
        (summ["comparison"].astype(str) == comparison)
        & (summ["k"].astype(int) == int(k))
        & (summ["set_similarity_metric"].astype(str) == metric)
        & (summ["metabolite_filter"].astype(str) == metabolite_filter)
        & (summ["homology_condition"].astype(str) == homology_condition)
    ].copy()

    return sub

def filter_per_query(
    pq: pd.DataFrame,
    comparison: str,
    k: int,
    metric: str,
    metabolite_filter: str,
    homology_condition: str,
) -> pd.DataFrame:
    sub = pq[
        (pq["comparison"].astype(str) == comparison)
        & (pq["k"].astype(int) == int(k))
        & (pq["set_similarity_metric"].astype(str) == metric)
        & (pq["metabolite_filter"].astype(str) == metabolite_filter)
        & (pq["homology_condition"].astype(str) == homology_condition)
    ].copy()

    return sub

def compute_per_query_delta(
    pq_b: pd.DataFrame,
    comparison: str,
    k: int,
    metric: str,
    metabolite_filter: str,
    homology_condition: str,
    backbone: str,
) -> pd.DataFrame:
    """
    Compute paired per-query delta:
      model_b - model_a

    model_a and model_b are parsed from comparison.
    """
    model_a, model_b = split_comparison(comparison)

    sub = filter_per_query(
        pq_b,
        comparison=comparison,
        k=k,
        metric=metric,
        metabolite_filter=metabolite_filter,
        homology_condition=homology_condition,
    )

    if sub.empty:
        raise ValueError(
            f"No per-query rows found for backbone={backbone}, comparison={comparison}, "
            f"k={k}, metric={metric}, metabolite_filter={metabolite_filter}, "
            f"homology_condition={homology_condition}"
        )

    available_models = sorted(sub["model"].astype(str).unique())

    if model_a not in available_models or model_b not in available_models:
        raise ValueError(
            f"Model names parsed from comparison do not match the 'model' column.\n"
            f"backbone={backbone}\n"
            f"comparison={comparison}\n"
            f"model_a={model_a}\n"
            f"model_b={model_b}\n"
            f"available_models={available_models}"
        )

    pa = (
        sub[sub["model"].astype(str) == model_a]
        .groupby("query_id", as_index=True)["coherence"]
        .mean()
    )
    pb = (
        sub[sub["model"].astype(str) == model_b]
        .groupby("query_id", as_index=True)["coherence"]
        .mean()
    )

    common = pa.index.intersection(pb.index)

    if len(common) == 0:
        raise ValueError(
            f"No common query_id between paired models for backbone={backbone}, "
            f"comparison={comparison}"
        )

    delta = pb.loc[common].values - pa.loc[common].values

    out = pd.DataFrame({
        "backbone": backbone,
        "query_id": common,
        "comparison": comparison,
        "model_a": model_a,
        "model_b": model_b,
        "delta": delta,
        "coherence_a": pa.loc[common].values,
        "coherence_b": pb.loc[common].values,
    })

    return out

def collect_main_data(
    pq: pd.DataFrame,
    summ: pd.DataFrame,
    backbone_order: List[str],
    comparison_arg: Optional[str],
    k: int,
    metric: str,
    metabolite_filter: str,
    homology_condition: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
      summary_plot_df:
        one row per backbone
      delta_df:
        one row per paired query enzyme
    """
    summary_records = []
    delta_frames = []

    for backbone in backbone_order:
        pq_b = pq[pq["backbone"] == backbone].copy()
        summ_b = summ[summ["backbone"] == backbone].copy()

        if pq_b.empty or summ_b.empty:
            logger.warning("skip backbone with no data: %s", backbone)
            continue

        comparison = comparison_arg or primary_comparison(summ_b)

        if comparison is None:
            raise ValueError(f"No comparison found for backbone={backbone}")

        logger.info("using comparison for %s: %s", backbone, comparison)

        summ_sub = filter_summary(
            summ_b,
            comparison=comparison,
            k=k,
            metric=metric,
            metabolite_filter=metabolite_filter,
            homology_condition=homology_condition,
        )

        if summ_sub.empty:
            raise ValueError(
                f"No summary rows found for backbone={backbone}, comparison={comparison}, "
                f"k={k}, metric={metric}, metabolite_filter={metabolite_filter}, "
                f"homology_condition={homology_condition}"
            )

        if len(summ_sub) > 1:
            logger.warning(
                "multiple summary rows found for backbone=%s; using the first row",
                backbone,
            )

        srow = summ_sub.iloc[0]

        delta_b = compute_per_query_delta(
            pq_b=pq_b,
            comparison=comparison,
            k=k,
            metric=metric,
            metabolite_filter=metabolite_filter,
            homology_condition=homology_condition,
            backbone=backbone,
        )

        d = delta_b["delta"].values

        per_query_mean, per_query_lo, per_query_hi = bootstrap_ci_mean(d)
        frac, frac_lo, frac_hi = bootstrap_ci_fraction_positive(d)

        summary_records.append({
            "backbone": backbone,
            "comparison": comparison,
            "k": k,
            "set_similarity_metric": metric,
            "metabolite_filter": metabolite_filter,
            "homology_condition": homology_condition,
            "n_query": int(delta_b.shape[0]),

            # Values from paired_comparison_summary.csv
            "summary_mean_delta": float(srow["mean_delta"]),
            "summary_bootstrap_ci_low": float(srow["bootstrap_ci_low"]),
            "summary_bootstrap_ci_high": float(srow["bootstrap_ci_high"]),

            # Values recomputed from per-query deltas
            "per_query_mean_delta": per_query_mean,
            "per_query_bootstrap_ci_low": per_query_lo,
            "per_query_bootstrap_ci_high": per_query_hi,
            "median_delta": float(np.median(d)),
            "fraction_improved": frac,
            "fraction_improved_ci_low": frac_lo,
            "fraction_improved_ci_high": frac_hi,
        })

        delta_frames.append(delta_b)

    if not summary_records:
        raise ValueError("No valid backbone data collected.")

    summary_plot_df = pd.DataFrame(summary_records)
    delta_df = pd.concat(delta_frames, ignore_index=True)

    return summary_plot_df, delta_df

def ordered_backbones(existing: List[str]) -> List[str]:
    existing_set = set(existing)
    ordered = [b for b in BACKBONE_ORDER if b in existing_set]
    ordered += [b for b in existing if b not in ordered]
    return ordered

def display_name(backbone: str) -> str:
    return BACKBONE_DISPLAY_NAMES.get(backbone, backbone)

def color_for(backbone: str) -> str:
    return BACKBONE_COLORS.get(backbone, "#777777")

# =============================================================================
# Plotting functions
# =============================================================================

def plot_mean_delta_by_backbone(
    summary_df: pd.DataFrame,
    plot_dir: str,
    use_summary_ci: bool = True,
    show_title: bool = False,
) -> None:
    """
    Paper panel:
      point estimate + 95% CI for mean paired Δcoherence.

    This is preferred over a bar plot because Δcoherence is a paired effect size.
    """
    backbones = ordered_backbones(summary_df["backbone"].tolist())
    df = summary_df.set_index("backbone").loc[backbones].reset_index()

    if use_summary_ci:
        mean_col = "summary_mean_delta"
        lo_col = "summary_bootstrap_ci_low"
        hi_col = "summary_bootstrap_ci_high"
    else:
        mean_col = "per_query_mean_delta"
        lo_col = "per_query_bootstrap_ci_low"
        hi_col = "per_query_bootstrap_ci_high"

    y = df[mean_col].values.astype(float)
    lo = df[lo_col].values.astype(float)
    hi = df[hi_col].values.astype(float)

    x = np.arange(len(df))
    colors = [color_for(b) for b in df["backbone"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_MEAN_DELTA)

    ax.axhline(0, color="black", linewidth=0.75, zorder=1)

    for i, (xi, yi, li, hi_i, color) in enumerate(zip(x, y, lo, hi, colors)):
        ax.vlines(
            xi,
            li,
            hi_i,
            color="black",
            linewidth=0.8,
            zorder=2,
        )
        ax.hlines(
            [li, hi_i],
            xi - 0.055,
            xi + 0.055,
            color="black",
            linewidth=0.8,
            zorder=2,
        )
        ax.scatter(
            xi,
            yi,
            s=34,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )

    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(b) for b in df["backbone"]], rotation=30, ha="right")

    ax.set_ylabel(r"Mean $\Delta$coherence")

    if show_title:
        ax.set_title("Mean coherence change")

    clean_axis(ax)

    ymin = min(np.nanmin(lo), 0)
    ymax = max(np.nanmax(hi), 0)
    pad = max((ymax - ymin) * 0.22, 0.005)
    ax.set_ylim(ymin - pad, ymax + pad)

    save_figure(fig, plot_dir, "paper_mean_delta_by_backbone")

def plot_per_query_delta_distribution(
    delta_df: pd.DataFrame,
    plot_dir: str,
    show_title: bool = False,
) -> None:
    """
    Paper panel:
      distribution of per-query paired Δcoherence across backbones.
    """
    backbones = ordered_backbones(delta_df["backbone"].unique().tolist())

    data = []
    labels = []
    colors = []

    for b in backbones:
        vals = delta_df.loc[delta_df["backbone"] == b, "delta"].dropna().values.astype(float)
        if vals.size == 0:
            continue

        data.append(vals)
        labels.append(display_name(b))
        colors.append(color_for(b))

    if not data:
        raise ValueError("No delta data available for distribution plot.")

    x = np.arange(1, len(data) + 1)

    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)

    parts = ax.violinplot(
        data,
        positions=x,
        widths=0.70,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.45)
        body.set_alpha(0.30)

    ax.boxplot(
        data,
        positions=x,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.85),
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.65),
        whiskerprops=dict(color="black", linewidth=0.65),
        capprops=dict(color="black", linewidth=0.65),
    )

    means = [np.mean(v) for v in data]
    ax.scatter(
        x,
        means,
        s=22,
        color="black",
        zorder=4,
    )

    ax.axhline(0, color="black", linewidth=0.75, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"Per-query $\Delta$coherence")

    if show_title:
        ax.set_title("Per-query coherence change")

    clean_axis(ax)

    all_vals = np.concatenate(data)

    ymin = min(np.nanpercentile(all_vals, 1), 0)
    ymax = max(np.nanpercentile(all_vals, 99), 0)
    pad = max((ymax - ymin) * 0.16, 0.01)

    ax.set_ylim(ymin - pad, ymax + pad)

    save_figure(fig, plot_dir, "paper_per_query_delta_distribution")

def plot_fraction_improved_by_backbone(
    summary_df: pd.DataFrame,
    plot_dir: str,
    show_title: bool = False,
) -> None:
    """
    Paper or supplementary panel:
      fraction of query enzymes with Δcoherence > 0.

    This remains a bar plot because the variable is a percentage.
    """
    backbones = ordered_backbones(summary_df["backbone"].tolist())
    df = summary_df.set_index("backbone").loc[backbones].reset_index()

    y = df["fraction_improved"].values.astype(float) * 100.0
    lo = df["fraction_improved_ci_low"].values.astype(float) * 100.0
    hi = df["fraction_improved_ci_high"].values.astype(float) * 100.0
    yerr = np.vstack([y - lo, hi - y])

    x = np.arange(len(df))
    colors = [color_for(b) for b in df["backbone"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_FRACTION)

    ax.bar(
        x,
        y,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        zorder=2,
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=0.75,
        capsize=2.4,
        capthick=0.75,
        zorder=3,
    )

    ax.axhline(50, color="black", linewidth=0.75, linestyle="--", zorder=1)

    ax.text(
        len(df) - 0.45,
        51.5,
        "50%",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([display_name(b) for b in df["backbone"]], rotation=30, ha="right")

    ax.set_ylabel(r"Queries with $\Delta$coherence > 0 (%)")
    ax.set_ylim(0, 100)

    if show_title:
        ax.set_title("Positive-query fraction")

    clean_axis(ax)

    save_figure(fig, plot_dir, "paper_fraction_improved_by_backbone")

# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paper-ready main plots for substrate-neighbourhood coherence."
    )

    p.add_argument(
        "--analysis-dirs",
        nargs="+",
        required=True,
        help=(
            "Backbone-labeled analysis directories. Format: "
            "BackboneLabel:/path/to/analysis_dir"
        ),
    )

    p.add_argument(
        "--plot-dir",
        type=str,
        required=True,
        help="Output directory for paper-ready plots.",
    )

    p.add_argument(
        "--comparison",
        type=str,
        default=None,
        help=(
            "Exact comparison label. If omitted, use comparison_type == primary "
            "separately for each backbone."
        ),
    )

    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Neighbour number used for the main panels.",
    )

    p.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help="Substrate-set similarity metric, e.g. c_sym.",
    )

    p.add_argument(
        "--metabolite-filter",
        type=str,
        default=DEFAULT_METABOLITE_FILTER,
        choices=["unfiltered", "filtered"],
    )

    p.add_argument(
        "--homology-condition",
        type=str,
        default=DEFAULT_HOMOLOGY_CONDITION,
    )

    p.add_argument(
        "--show-title",
        action="store_true",
        help="Show simple panel titles. Usually keep this off for paper assembly.",
    )

    p.add_argument(
        "--use-per-query-ci-for-mean",
        action="store_true",
        help=(
            "Use CI recomputed from per_query_coherence.csv for the mean-delta panel. "
            "By default, use the CI in paired_comparison_summary.csv."
        ),
    )

    return p

def main() -> None:
    args = build_argparser().parse_args()

    apply_style()
    ensure_dir(args.plot_dir)

    backbone_dirs = parse_analysis_dirs(args.analysis_dirs)

    input_backbones = list(backbone_dirs.keys())
    backbone_order = ordered_backbones(input_backbones)

    pq, summ = load_all_backbones(backbone_dirs)

    summary_df, delta_df = collect_main_data(
        pq=pq,
        summ=summ,
        backbone_order=backbone_order,
        comparison_arg=args.comparison,
        k=args.k,
        metric=args.metric,
        metabolite_filter=args.metabolite_filter,
        homology_condition=args.homology_condition,
    )

    summary_csv = os.path.join(args.plot_dir, "paper_main_summary.csv")
    delta_csv = os.path.join(args.plot_dir, "paper_per_query_delta.csv")

    summary_df.to_csv(summary_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)

    logger.info("wrote %s", summary_csv)
    logger.info("wrote %s", delta_csv)

    plot_mean_delta_by_backbone(
        summary_df=summary_df,
        plot_dir=args.plot_dir,
        use_summary_ci=not args.use_per_query_ci_for_mean,
        show_title=args.show_title,
    )

    plot_per_query_delta_distribution(
        delta_df=delta_df,
        plot_dir=args.plot_dir,
        show_title=args.show_title,
    )

    plot_fraction_improved_by_backbone(
        summary_df=summary_df,
        plot_dir=args.plot_dir,
        show_title=args.show_title,
    )

    logger.info("all paper-ready panels written to %s", args.plot_dir)

if __name__ == "__main__":
    main()