#!/usr/bin/env python3

"""
plot_rns_backbone_boxplot.py

Load RNS comparison results from three backbone result directories and draw:

1) Main / paper figure:
   rns_delta_boxplot_k500
   Boxplots of per-sequence ΔRNS (= CPT - Base) at representative k.

2) Supplementary figure:
   rns_delta_boxplots_by_k
   Boxplots of per-sequence ΔRNS across all k values.

3) Supplementary figure:
   rns_mean_delta_line
   Mean ΔRNS line plot across k.

4) Supplementary / optional figure:
   rns_fraction_non_worsened_bar
   Fraction of sequences with decreased or unchanged RNS.
   This avoids the misleading interpretation that all non-improved sequences are worsened.

Expected files inside each result directory:
- rns_per_sequence.csv
- rns_base_vs_cmp.csv

Definition:
- delta_rns = RNS_CPT - RNS_Base
- lower RNS is better
- delta_rns < 0 means RNS decreases after CPT, i.e. improved
- delta_rns = 0 means unchanged
- delta_rns > 0 means worsened
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    "result_dirs": {
        "ProtBERT-BFD": "artifacts/analysis/rns/results_protbert_bfd_cpt_bf16",
        "ESM2-650M": "artifacts/analysis/rns/results_esm2_650m_cpt_bf16",
        "ESM2-3B": "artifacts/analysis/rns/results_esm2_3b_cpt_bf16",
    },

    "output_dir": "artifacts/analysis/rns/figures_backbone_rns",

    "backbone_order": ["ProtBERT-BFD", "ESM2-650M", "ESM2-3B"],

    "colors": {
        "ProtBERT-BFD": "#8A8A8A",
        "ESM2-650M": "#6A8CAF",
        "ESM2-3B": "#9A7BB5",
    },

    "light_colors": {
        "ProtBERT-BFD": "#D8D8D8",
        "ESM2-650M": "#D6E2EF",
        "ESM2-3B": "#E3D8EC",
    },

    "fig_dpi": 600,
    "box_alpha": 1.00,
    "box_width": 0.52,
    "showfliers": False,

    "show_mean_marker": True,
    "mean_marker": "o",
    "mean_marker_size": 4.8,
    "mean_marker_color": "black",

    "unchanged_tol": 1e-12,

    "font_family": "Arial",
    "fontsize_title": 8,
    "fontsize_label": 8,
    "fontsize_tick": 7,
    "fontsize_legend": 7,
    "fontsize_annot": 6.5,

    "boxplot_panel_width": 2.55,
    "boxplot_panel_height": 2.65,
    "single_boxplot_width": 3.1,
    "single_boxplot_height": 2.55,
    "lineplot_width": 3.4,
    "lineplot_height": 2.55,
    "barplot_width": 3.25,
    "barplot_height": 2.55,

    "representative_k": 500,

    "zero_line_color": "black",
    "zero_line_style": "-",
    "zero_line_width": 0.75,

    "chance_line_color": "black",
    "chance_line_style": "--",
    "chance_line_width": 0.75,

    "edge_color": "black",
    "edge_linewidth": 0.65,

    "save_png": True,
    "save_pdf": True,
    "save_svg": True,

    "show_titles": False,
}

def apply_style(cfg):
    plt.rcParams.update({
        "font.family": cfg["font_family"],
        "font.size": cfg["fontsize_tick"],
        "axes.labelsize": cfg["fontsize_label"],
        "axes.titlesize": cfg["fontsize_title"],
        "xtick.labelsize": cfg["fontsize_tick"],
        "ytick.labelsize": cfg["fontsize_tick"],
        "legend.fontsize": cfg["fontsize_legend"],
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_figure(fig, out_dir, stem, dpi=300, save_png=True, save_pdf=True, save_svg=True):
    out_dir = Path(out_dir)

    if save_png:
        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_pdf:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    if save_svg:
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)

def prettify_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

def get_present_backbones(df, cfg):
    order = cfg["backbone_order"]
    present = [b for b in order if b in set(df["backbone"].astype(str))]
    return present

def classify_delta(vals, tol):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]

    improved = vals < -tol
    unchanged = np.abs(vals) <= tol
    worsened = vals > tol
    non_worsened = improved | unchanged

    return {
        "n": int(vals.size),
        "n_improved": int(improved.sum()),
        "n_unchanged": int(unchanged.sum()),
        "n_worsened": int(worsened.sum()),
        "n_non_worsened": int(non_worsened.sum()),
        "fraction_improved": float(improved.mean()) if vals.size else np.nan,
        "fraction_unchanged": float(unchanged.mean()) if vals.size else np.nan,
        "fraction_worsened": float(worsened.mean()) if vals.size else np.nan,
        "fraction_non_worsened": float(non_worsened.mean()) if vals.size else np.nan,
    }

def bootstrap_ci_fraction(mask, n_boot=5000, seed=2025):
    mask = np.asarray(mask, dtype=float)
    mask = mask[~np.isnan(mask)]

    if mask.size == 0:
        return np.nan, np.nan, np.nan
    if mask.size == 1:
        v = float(mask[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mask.size, size=(n_boot, mask.size))
    props = mask[idx].mean(axis=1)

    return (
        float(mask.mean()),
        float(np.percentile(props, 2.5)),
        float(np.percentile(props, 97.5)),
    )

def load_data(cfg):
    per_seq_rows = []
    summary_rows = []

    for backbone, result_dir in cfg["result_dirs"].items():
        result_dir = Path(result_dir)

        per_seq_path = result_dir / "rns_per_sequence.csv"
        summary_path = result_dir / "rns_base_vs_cmp.csv"

        if not per_seq_path.exists():
            raise FileNotFoundError(f"Missing file: {per_seq_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing file: {summary_path}")

        per_seq_df = pd.read_csv(per_seq_path)
        summary_df = pd.read_csv(summary_path)

        if "delta_rns" not in per_seq_df.columns:
            raise ValueError(f"'delta_rns' column not found in {per_seq_path}")
        if "k" not in per_seq_df.columns:
            raise ValueError(f"'k' column not found in {per_seq_path}")

        per_seq_df["backbone"] = backbone
        summary_df["backbone"] = backbone

        per_seq_rows.append(per_seq_df)
        summary_rows.append(summary_df)

    per_seq_all = pd.concat(per_seq_rows, ignore_index=True)
    summary_all = pd.concat(summary_rows, ignore_index=True)

    backbone_order = cfg["backbone_order"]

    per_seq_all["backbone"] = pd.Categorical(
        per_seq_all["backbone"],
        categories=backbone_order,
        ordered=True,
    )
    summary_all["backbone"] = pd.Categorical(
        summary_all["backbone"],
        categories=backbone_order,
        ordered=True,
    )

    per_seq_all = per_seq_all.sort_values(["k", "backbone"]).reset_index(drop=True)
    summary_all = summary_all.sort_values(["k", "backbone"]).reset_index(drop=True)

    return per_seq_all, summary_all

def plot_boxplots_by_k(per_seq_df, cfg, out_dir):
    ks = sorted(per_seq_df["k"].unique())
    backbones = get_present_backbones(per_seq_df, cfg)
    colors = cfg["light_colors"]

    n_panels = len(ks)
    fig_width = cfg["boxplot_panel_width"] * n_panels
    fig_height = cfg["boxplot_panel_height"]

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_height),
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    global_ymin = np.nanmin(per_seq_df["delta_rns"].values)
    global_ymax = np.nanmax(per_seq_df["delta_rns"].values)
    pad = 0.08 * (global_ymax - global_ymin + 1e-12)
    ymin = global_ymin - pad
    ymax = global_ymax + pad

    for ax, k in zip(axes, ks):
        sub = per_seq_df[per_seq_df["k"] == k]

        data = []
        for backbone in backbones:
            vals = sub.loc[sub["backbone"] == backbone, "delta_rns"].dropna().values
            data.append(vals)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=cfg["box_width"],
            showfliers=cfg["showfliers"],
            medianprops={"color": "black", "linewidth": 0.85},
            boxprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
            whiskerprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
            capprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
        )

        for patch, backbone in zip(bp["boxes"], backbones):
            patch.set_facecolor(colors[backbone])
            patch.set_alpha(cfg["box_alpha"])
            patch.set_rasterized(False)

        if cfg["show_mean_marker"]:
            for i, backbone in enumerate(backbones, start=1):
                vals = sub.loc[sub["backbone"] == backbone, "delta_rns"].dropna().values
                if len(vals) > 0:
                    mean_val = np.mean(vals)
                    ax.plot(
                        i,
                        mean_val,
                        marker=cfg["mean_marker"],
                        markersize=cfg["mean_marker_size"],
                        color=cfg["mean_marker_color"],
                        linestyle="None",
                        zorder=4,
                    )

        ax.axhline(
            0,
            color=cfg["zero_line_color"],
            linestyle=cfg["zero_line_style"],
            linewidth=cfg["zero_line_width"],
        )

        if cfg["show_titles"]:
            ax.set_title(f"k = {k}")

        ax.set_xticks(range(1, len(backbones) + 1))
        ax.set_xticklabels(backbones, rotation=30, ha="right")
        ax.tick_params(axis="y")
        ax.set_ylim(ymin, ymax)
        prettify_axis(ax)

    axes[0].set_ylabel(r"$\Delta$RNS")
    fig.text(
        0.5,
        -0.02,
        r"$\Delta$RNS = RNS$_{CPT}$ - RNS$_{Base}$; lower values indicate reduced RNS after CPT.",
        ha="center",
        fontsize=cfg["fontsize_annot"],
    )

    fig.tight_layout()

    save_figure(
        fig,
        out_dir,
        "rns_delta_boxplots_by_k",
        dpi=cfg["fig_dpi"],
        save_png=cfg["save_png"],
        save_pdf=cfg["save_pdf"],
        save_svg=cfg["save_svg"],
    )

def plot_boxplot_representative_k(per_seq_df, cfg, out_dir):
    ks = sorted(per_seq_df["k"].unique())
    backbones = get_present_backbones(per_seq_df, cfg)
    colors = cfg["light_colors"]

    rep_k = cfg["representative_k"]
    if rep_k is None:
        rep_k = ks[len(ks) // 2]

    if rep_k not in ks:
        raise ValueError(f"representative_k={rep_k} not found. Available k values: {ks}")

    sub = per_seq_df[per_seq_df["k"] == rep_k]

    fig, ax = plt.subplots(figsize=(cfg["single_boxplot_width"], cfg["single_boxplot_height"]))

    data = []
    for backbone in backbones:
        vals = sub.loc[sub["backbone"] == backbone, "delta_rns"].dropna().values
        data.append(vals)

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=cfg["box_width"],
        showfliers=cfg["showfliers"],
        medianprops={"color": "black", "linewidth": 0.85},
        boxprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
        whiskerprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
        capprops={"linewidth": cfg["edge_linewidth"], "color": cfg["edge_color"]},
    )

    for patch, backbone in zip(bp["boxes"], backbones):
        patch.set_facecolor(colors[backbone])
        patch.set_alpha(cfg["box_alpha"])
        patch.set_rasterized(False)

    if cfg["show_mean_marker"]:
        for i, backbone in enumerate(backbones, start=1):
            vals = sub.loc[sub["backbone"] == backbone, "delta_rns"].dropna().values
            if len(vals) > 0:
                mean_val = np.mean(vals)
                ax.plot(
                    i,
                    mean_val,
                    marker=cfg["mean_marker"],
                    markersize=cfg["mean_marker_size"],
                    color=cfg["mean_marker_color"],
                    linestyle="None",
                    zorder=4,
                )

    ax.axhline(
        0,
        color=cfg["zero_line_color"],
        linestyle=cfg["zero_line_style"],
        linewidth=cfg["zero_line_width"],
    )

    ax.set_xticks(range(1, len(backbones) + 1))
    ax.set_xticklabels(backbones, rotation=30, ha="right")

    ax.set_ylabel(r"$\Delta$RNS")
    if cfg["show_titles"]:
        ax.set_title(f"Per-sequence ΔRNS at k = {rep_k}")

    prettify_axis(ax)

    vals_all = sub["delta_rns"].dropna().values
    ymin = min(np.nanpercentile(vals_all, 1), 0)
    ymax = max(np.nanpercentile(vals_all, 99), 0)
    pad = max((ymax - ymin) * 0.16, 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()

    save_figure(
        fig,
        out_dir,
        f"rns_delta_boxplot_k{rep_k}",
        dpi=cfg["fig_dpi"],
        save_png=cfg["save_png"],
        save_pdf=cfg["save_pdf"],
        save_svg=cfg["save_svg"],
    )

def plot_mean_delta_line(summary_df, per_seq_df, cfg, out_dir):
    backbones = get_present_backbones(per_seq_df, cfg)
    colors = cfg["colors"]

    fig, ax = plt.subplots(figsize=(cfg["lineplot_width"], cfg["lineplot_height"]))

    mean_df = (
        per_seq_df
        .groupby(["backbone", "k"], observed=False)["delta_rns"]
        .mean()
        .reset_index(name="mean_delta_rns")
    )

    for backbone in backbones:
        sub = mean_df[mean_df["backbone"] == backbone].sort_values("k")
        ax.plot(
            sub["k"].values,
            sub["mean_delta_rns"].values,
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            color=colors[backbone],
            label=backbone,
        )

    ax.axhline(
        0,
        color=cfg["zero_line_color"],
        linestyle=cfg["zero_line_style"],
        linewidth=cfg["zero_line_width"],
    )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Mean $\Delta$RNS")

    if cfg["show_titles"]:
        ax.set_title("Mean ΔRNS across k")

    ax.legend(frameon=False)
    prettify_axis(ax)

    fig.tight_layout()

    save_figure(
        fig,
        out_dir,
        "rns_mean_delta_line",
        dpi=cfg["fig_dpi"],
        save_png=cfg["save_png"],
        save_pdf=cfg["save_pdf"],
        save_svg=cfg["save_svg"],
    )

def build_fraction_table(per_seq_df, cfg):
    tol = cfg["unchanged_tol"]
    rows = []

    for (k, backbone), sub in per_seq_df.groupby(["k", "backbone"], observed=False):
        vals = sub["delta_rns"].dropna().values
        stats = classify_delta(vals, tol=tol)

        improved_mask = vals < -tol
        unchanged_mask = np.abs(vals) <= tol
        non_worsened_mask = improved_mask | unchanged_mask

        frac, lo, hi = bootstrap_ci_fraction(non_worsened_mask.astype(float))

        rows.append({
            "k": k,
            "backbone": backbone,
            **stats,
            "fraction_non_worsened_ci_low": lo,
            "fraction_non_worsened_ci_high": hi,
        })

    out = pd.DataFrame(rows)
    return out

def plot_fraction_non_worsened(fraction_df, cfg, out_dir):
    backbones = [b for b in cfg["backbone_order"] if b in set(fraction_df["backbone"].astype(str))]
    colors = cfg["colors"]
    ks = sorted(fraction_df["k"].unique())

    x = np.arange(len(backbones))
    width = 0.22 if len(ks) <= 3 else 0.16

    fig, ax = plt.subplots(figsize=(cfg["barplot_width"], cfg["barplot_height"]))

    for i, k in enumerate(ks):
        sub = (
            fraction_df[fraction_df["k"] == k]
            .set_index("backbone")
            .loc[backbones]
            .reset_index()
        )

        y = sub["fraction_non_worsened"].values * 100.0
        lo = sub["fraction_non_worsened_ci_low"].values * 100.0
        hi = sub["fraction_non_worsened_ci_high"].values * 100.0

        yerr = np.vstack([y - lo, hi - y])

        offset = (i - (len(ks) - 1) / 2) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            label=f"k={k}",
            edgecolor="black",
            linewidth=0.55,
            alpha=0.95,
        )

        ax.errorbar(
            x + offset,
            y,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=0.65,
            capsize=2.2,
            capthick=0.65,
            zorder=3,
        )

    ax.axhline(
        50,
        color=cfg["chance_line_color"],
        linestyle=cfg["chance_line_style"],
        linewidth=cfg["chance_line_width"],
    )

    ax.text(
        len(backbones) - 0.45,
        51.5,
        "50%",
        ha="right",
        va="bottom",
        fontsize=cfg["fontsize_annot"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(backbones, rotation=30, ha="right")
    ax.set_ylabel("Sequences with decreased or unchanged RNS (%)")

    if cfg["show_titles"]:
        ax.set_title("Fraction of non-worsened sequences")

    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    prettify_axis(ax)

    fig.tight_layout()

    save_figure(
        fig,
        out_dir,
        "rns_fraction_non_worsened_bar",
        dpi=cfg["fig_dpi"],
        save_png=cfg["save_png"],
        save_pdf=cfg["save_pdf"],
        save_svg=cfg["save_svg"],
    )

def save_tables(per_seq_df, summary_df, fraction_df, out_dir):
    out_dir = Path(out_dir)

    per_seq_df.to_csv(out_dir / "merged_rns_per_sequence.csv", index=False)
    summary_df.to_csv(out_dir / "merged_rns_base_vs_cmp.csv", index=False)
    fraction_df.to_csv(out_dir / "merged_rns_fraction_non_worsened.csv", index=False)

    stats_rows = []
    tol = CONFIG["unchanged_tol"]

    for (k, backbone), sub in per_seq_df.groupby(["k", "backbone"], observed=False):
        vals = sub["delta_rns"].dropna().values
        cls = classify_delta(vals, tol=tol)

        if len(vals) == 0:
            continue

        stats_rows.append({
            "k": k,
            "backbone": backbone,
            "n": len(vals),
            "mean_delta_rns": np.mean(vals),
            "median_delta_rns": np.median(vals),
            "std_delta_rns": np.std(vals),
            "q1": np.percentile(vals, 25),
            "q3": np.percentile(vals, 75),
            "min": np.min(vals),
            "max": np.max(vals),
            **cls,
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out_dir / "merged_rns_delta_stats.csv", index=False)

def main():
    cfg = CONFIG
    apply_style(cfg)

    out_dir = ensure_dir(cfg["output_dir"])

    per_seq_df, summary_df = load_data(cfg)
    fraction_df = build_fraction_table(per_seq_df, cfg)

    save_tables(per_seq_df, summary_df, fraction_df, out_dir)

    plot_boxplots_by_k(per_seq_df, cfg, out_dir)
    plot_boxplot_representative_k(per_seq_df, cfg, out_dir)
    plot_mean_delta_line(summary_df, per_seq_df, cfg, out_dir)
    plot_fraction_non_worsened(fraction_df, cfg, out_dir)

    rep_k = cfg["representative_k"]

    print("Done.")
    print(f"Figures saved to: {out_dir}")
    print("Generated figure files:")
    print(" - rns_delta_boxplots_by_k.(png/pdf/svg)")
    print(f" - rns_delta_boxplot_k{rep_k}.(png/pdf/svg)")
    print(" - rns_mean_delta_line.(png/pdf/svg)")
    print(" - rns_fraction_non_worsened_bar.(png/pdf/svg)")
    print("Generated tables:")
    print(" - merged_rns_per_sequence.csv")
    print(" - merged_rns_base_vs_cmp.csv")
    print(" - merged_rns_delta_stats.csv")
    print(" - merged_rns_fraction_non_worsened.csv")

if __name__ == "__main__":
    main()