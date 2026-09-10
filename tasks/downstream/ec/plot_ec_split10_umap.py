#!/usr/bin/env python3
"""Visualize high-frequency full EC4 labels with two automatic strategies.

The official EC protocol uses ``split10.csv`` as a query subset of
``split100.csv``.  Its embeddings therefore live in each model directory's
``split100_all.pt`` rather than in a separate ``split10_all.pt`` file.

Only single-label enzymes contribute to EC4 frequencies.  One run produces a
global top-frequency EC4 figure and a second figure selecting the most frequent
complete numeric EC4 within every EC1 class.  Lexical order breaks exact
frequency ties.  Multi-label enzymes, incomplete EC labels, and unselected EC4
labels are grey background samples in both strategies.

The default t-SNE projection L2-normalizes the sampled embeddings without PCA,
fits CPT with random initialization, and then fits CPT-SUB using the CPT 2D
coordinates as initialization.  Silhouette is computed in the original
normalized high-dimensional embedding space using coloured samples only.

Example (run from the server EC task directory):
    python plot_ec_split10_umap.py \
      --split10-csv data/split10.csv \
      --cpt-emb-root embedding/esm2_t33_650M/paper/esm2_t33_650M_cpt \
      --cpt-sub-emb-root embedding/esm2_t33_650M/paper/cpt_sub_r16_ecfp \
      --output-dir paper/results/umap_ec_split10_cpt_vs_cpt_sub

Dependencies: numpy, pandas, torch, scikit-learn, matplotlib.  UMAP mode also
requires umap-learn.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import torch
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
except ImportError as exc:  # pragma: no cover - server environment dependent
    raise SystemExit(
        "Missing plotting dependency. Install: "
        "pip install scikit-learn matplotlib pandas numpy torch"
    ) from exc

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 5.4,
        "axes.labelsize": 5.1,
        "axes.titlesize": 6.2,
        "legend.fontsize": 4.5,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

STATE_COLORS = {"CPT": "#62636C", "CPT-SUB": "#3F7773"}
AXIS_COLOR = "#555555"
TEXT_COLOR = "#242424"
OTHER_COLOR = "#D4D5D6"
# Two server-rendered 89 mm panels plus a 5 mm central assembly gap = 183 mm.
FIGURE_SIZE_MM = (89.0, 58.0)
FIGURE_SIZE = tuple(value / 25.4 for value in FIGURE_SIZE_MM)
OUTPUT_DPI = 600
RANDOM_STATE = 42
TSNE_PERPLEXITY = 30.0
TSNE_ITERATIONS = 1000
DISTANCE_METRIC = "cosine"
MAX_GRAY_LIMIT = 50000
PROJECTION_BASIS = "direct L2-normalized embeddings without PCA"
TSNE_ALIGNMENT = "CPT coordinates used to initialize the CPT-SUB t-SNE"
CLASS_COLORS = (
    "#4E79A7",
    "#E08B3E",
    "#59A14F",
    "#D7656B",
    "#8F72B2",
    "#58A5A6",
    "#C59B32",
    "#A56A43",
    "#6F8FB3",
    "#B06A92",
)
ID_CANDIDATES = ("Entry", "entry", "id", "ID", "seq_id", "uniprot_id")
EC_CANDIDATES = ("EC number", "ec_number", "EC", "ec", "label")
COMPLETE_EC4_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")

@dataclass(frozen=True)
class RunConfig:
    variant_slug: str
    selection_strategy: str
    global_top_n: int | None
    split10_csv: str
    cpt_embedding_file: str
    cpt_sub_embedding_file: str
    id_column: str
    ec_column: str
    ec_level: int
    ec_selection_rule: str
    selected_ec_classes: list[str]
    selected_ec4_by_ec1: dict[str, list[str]]
    selected_ec_raw_frequencies: dict[str, int]
    selected_ec_prebalance_counts: dict[str, int]
    selected_ec_plotted_counts_per_panel: dict[str, int]
    assignment_rule: str
    n_split10_rows: int
    n_singlelabel_rows: int
    n_multilabel_rows: int
    n_singlelabel_complete_ec4_rows: int
    n_singlelabel_incomplete_ec_rows: int
    n_unique_singlelabel_complete_ec4: int
    n_neighbors: int
    min_dist: float
    metric: str
    random_state: int
    method: str
    perplexity: float
    tsne_iterations: int
    cpt_tsne_initialization: str
    cpt_sub_tsne_initialization: str
    tsne_alignment: str
    projection_basis: str
    silhouette_space: str
    projection_population_rule: str
    figure_size_inches: list[float]
    figure_size_mm: list[float]
    output_dpi: int
    balance_round_to: int
    max_gray: int
    balanced_colored_per_group: int
    n_gray_before_sampling: int
    n_gray_plotted_per_panel: int
    n_visualization_samples: int
    sampling_seeds: dict[str, int]
    sampling_algorithm: str
    silhouette_scores: dict[str, float | None]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned CPT/CPT-SUB t-SNE/UMAP of split10 EC classes."
    )
    parser.add_argument("--split10-csv", type=Path, required=True)
    parser.add_argument("--cpt-emb-root", type=Path, required=True)
    parser.add_argument("--cpt-sub-emb-root", type=Path, required=True)
    parser.add_argument(
        "--embedding-file",
        default="split100_all.pt",
        help="Merged embedding filename inside both roots (default: split100_all.pt).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--ec-column", default=None)
    parser.add_argument(
        "--global-top-n",
        type=int,
        default=None,
        help=(
            "Number of globally most frequent single-label EC4 classes in the "
            "companion figure (default: match the number of represented EC1 classes)."
        ),
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.10)
    parser.add_argument(
        "--method", choices=("tsne", "umap"), default="tsne",
        help="Dimensionality-reduction method (default: tsne).",
    )
    parser.add_argument("--balance-round-to", type=int, default=100)
    parser.add_argument("--max-gray", type=int, default=MAX_GRAY_LIMIT)
    parser.set_defaults(
        metric=DISTANCE_METRIC,
        random_state=RANDOM_STATE,
        perplexity=TSNE_PERPLEXITY,
        tsne_iterations=TSNE_ITERATIONS,
    )
    return parser.parse_args()

def detect_delimiter(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"split10 label table not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline()
    if "\t" in header:
        return "\t"
    if "," in header:
        return ","
    raise ValueError(f"Could not detect tab/comma delimiter from: {path}")

def select_column(
    columns: Iterable[str], requested: str | None, candidates: Sequence[str], role: str
) -> str:
    available = list(columns)
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested {role} column {requested!r} is absent; found {available}"
            )
        return requested
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"Could not identify {role} column; found {available}")

def canonical_id(raw_id: object) -> str:
    """Match the EC FASTA reader: retain the first whitespace-delimited token."""
    return str(raw_id).strip().split()[0] if str(raw_id).strip() else ""

def split_ec_labels(value: object) -> tuple[str, ...]:
    """Mirror the formal EC evaluator: semicolon split, strip, deduplicate."""
    labels: list[str] = []
    seen: set[str] = set()
    for raw_label in str(value).split(";"):
        label = raw_label.strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return tuple(labels)

def is_complete_ec4(label: str) -> bool:
    """Return whether a label is a complete four-level numeric EC number."""
    return COMPLETE_EC4_PATTERN.fullmatch(label) is not None

def ec1_from_ec4(label: str) -> str:
    if not is_complete_ec4(label):
        raise ValueError(f"Not a complete EC4 label: {label!r}")
    return label.split(".", 1)[0]

def ec1_sort_key(ec1: str) -> tuple[int, int | str]:
    return (0, int(ec1)) if ec1.isdigit() else (1, ec1)

def load_split10(
    path: Path,
    requested_id_column: str | None,
    requested_ec_column: str | None,
) -> tuple[pd.DataFrame, str, str]:
    table = pd.read_csv(
        path,
        sep=detect_delimiter(path),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    id_column = select_column(table.columns, requested_id_column, ID_CANDIDATES, "ID")
    ec_column = select_column(table.columns, requested_ec_column, EC_CANDIDATES, "EC")
    table = table[[id_column, ec_column]].copy()
    table[id_column] = table[id_column].map(canonical_id)
    if (table[id_column] == "").any():
        raise ValueError("split10 contains an empty sequence ID")
    duplicates = table.loc[table[id_column].duplicated(keep=False), id_column].unique()
    if len(duplicates):
        raise ValueError(
            f"split10 contains duplicate IDs; examples: {duplicates[:5].tolist()}"
        )

    table["ec_labels_full"] = table[ec_column].map(split_ec_labels)
    if (~table["ec_labels_full"].map(bool)).any():
        bad = table.loc[~table["ec_labels_full"].map(bool), id_column].head(5).tolist()
        raise ValueError(f"split10 rows without EC labels; examples: {bad}")
    table["single_label_ec4"] = table["ec_labels_full"].map(
        lambda labels: labels[0]
        if len(labels) == 1 and is_complete_ec4(labels[0])
        else ""
    )
    return table, id_column, ec_column

def resolve_embedding_file(root_or_file: Path, filename: str) -> Path:
    path = root_or_file if root_or_file.is_file() else root_or_file / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Merged embedding file not found: {path}. "
            "For split10, use the model's split100_all.pt file."
        )
    return path

def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch
        return torch.load(path, map_location="cpu")

def load_embeddings(path: Path, ordered_ids: Sequence[str]) -> np.ndarray:
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}")
    raw_embeddings = payload.get("embeddings", payload)
    if not isinstance(raw_embeddings, dict):
        raise TypeError(f"Expected an embedding dictionary in {path}")

    embeddings: dict[str, Any] = {}
    for raw_id, tensor in raw_embeddings.items():
        seq_id = canonical_id(raw_id)
        if seq_id in embeddings:
            raise ValueError(f"Duplicate canonical embedding ID {seq_id!r} in {path}")
        embeddings[seq_id] = tensor

    missing = [seq_id for seq_id in ordered_ids if seq_id not in embeddings]
    if missing:
        raise ValueError(
            f"{len(missing)} split10 IDs are absent from {path}; examples: {missing[:5]}"
        )

    rows: list[np.ndarray] = []
    expected_dim: int | None = None
    for seq_id in ordered_ids:
        value = embeddings[seq_id]
        if isinstance(value, dict) and "mean_representations" in value:
            value = value["mean_representations"]
        vector = torch.as_tensor(value).detach().cpu().float().numpy().reshape(-1)
        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        elif vector.shape[0] != expected_dim:
            raise ValueError(
                f"Inconsistent embedding dimension for {seq_id!r} in {path}: "
                f"{vector.shape[0]} vs {expected_dim}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite embedding values for {seq_id!r} in {path}")
        rows.append(vector)
    return np.stack(rows).astype(np.float32, copy=False)

def count_single_label_complete_ec4(table: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter(
        label for label in table["single_label_ec4"] if label
    )
    if not counts:
        raise ValueError("No complete single-label EC4 values are available")
    return counts

def choose_per_ec1_high_frequency_classes(
    counts: Counter[str],
) -> tuple[list[str], dict[str, int]]:
    """Select exactly one most frequent single-label EC4 within every EC1.

    The selection is never ranked or truncated across EC1 classes.  An exact
    within-EC1 frequency tie is resolved lexically to make the rule stable.
    """
    candidates_by_ec1: dict[str, list[str]] = {}
    for label in counts:
        candidates_by_ec1.setdefault(ec1_from_ec4(label), []).append(label)
    if len(candidates_by_ec1) > len(CLASS_COLORS):
        raise ValueError(
            f"Found {len(candidates_by_ec1)} EC1 classes but only "
            f"{len(CLASS_COLORS)} reference colours"
        )

    selected: list[str] = []
    tie_sizes_by_label: dict[str, int] = {}
    for ec1 in sorted(candidates_by_ec1, key=ec1_sort_key):
        ranked = sorted(
            candidates_by_ec1[ec1], key=lambda label: (-counts[label], label)
        )
        winner = ranked[0]
        selected.append(winner)
        tie_sizes_by_label[winner] = sum(
            counts[label] == counts[winner] for label in ranked
        )
    return selected, tie_sizes_by_label

def choose_global_high_frequency_classes(
    counts: Counter[str], top_n: int
) -> tuple[list[str], dict[str, int]]:
    """Select the global top-N single-label complete EC4 classes."""
    if top_n < 1 or top_n > len(CLASS_COLORS):
        raise ValueError(
            f"--global-top-n must be between 1 and {len(CLASS_COLORS)}"
        )
    ranked = sorted(counts, key=lambda label: (-counts[label], label))
    selected = ranked[:top_n]
    if len(selected) < top_n:
        raise ValueError(
            f"Requested global top-{top_n}, but only {len(selected)} EC4 classes exist"
        )
    tie_sizes_by_label = {
        label: sum(counts[other] == counts[label] for other in counts)
        for label in selected
    }
    return selected, tie_sizes_by_label

def assign_highlight_class(
    labels: tuple[str, ...], selected: set[str]
) -> tuple[str, int]:
    if len(labels) == 1 and is_complete_ec4(labels[0]) and labels[0] in selected:
        return labels[0], 1
    return "Other", 0

def fit_aligned_umap(
    cpt_matrix: np.ndarray,
    cpt_sub_matrix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit CPT first and use its coordinates to initialize CPT-SUB UMAP."""
    try:
        import umap.umap_ as umap
    except ImportError as exc:  # pragma: no cover - server environment dependent
        raise SystemExit("UMAP mode requires: pip install umap-learn") from exc
    if cpt_matrix.shape != cpt_sub_matrix.shape:
        raise ValueError(
            f"Paired embedding matrices differ: {cpt_matrix.shape} vs {cpt_sub_matrix.shape}"
        )
    if cpt_matrix.shape[0] < 4:
        raise ValueError("UMAP needs at least four paired embedding rows")

    cpt_normalized = normalize(cpt_matrix, norm="l2", axis=1, copy=True)
    cpt_sub_normalized = normalize(cpt_sub_matrix, norm="l2", axis=1, copy=True)
    common = dict(
        n_components=2,
        n_neighbors=min(args.n_neighbors, cpt_normalized.shape[0] - 1),
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    cpt_coordinates = umap.UMAP(init="spectral", **common).fit_transform(cpt_normalized)
    cpt_sub_coordinates = umap.UMAP(
        init=cpt_coordinates.copy(), **common
    ).fit_transform(cpt_sub_normalized)
    return cpt_coordinates, cpt_sub_coordinates, 0

def run_tsne(
    matrix: np.ndarray,
    args: argparse.Namespace,
    init_coordinates: np.ndarray | None = None,
) -> np.ndarray:
    perplexity = float(args.perplexity)
    if matrix.shape[0] <= perplexity:
        raise ValueError(
            f"t-SNE requires n_samples > perplexity; got {matrix.shape[0]} "
            f"samples and perplexity={perplexity:g}"
        )
    common = dict(
        n_components=2,
        perplexity=perplexity,
        metric=args.metric,
        init=init_coordinates if init_coordinates is not None else "random",
        learning_rate="auto",
        random_state=args.random_state,
        method="barnes_hut",
        angle=0.5,
    )
    try:
        return TSNE(max_iter=args.tsne_iterations, **common).fit_transform(matrix)
    except TypeError:  # scikit-learn < 1.5 uses n_iter.
        return TSNE(n_iter=args.tsne_iterations, **common).fit_transform(matrix)

def fit_aligned_tsne(
    cpt_matrix: np.ndarray,
    cpt_sub_matrix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Use the CPT map to initialize the paired CPT-SUB t-SNE."""
    if cpt_matrix.shape != cpt_sub_matrix.shape:
        raise ValueError(
            f"Paired embedding matrices differ: {cpt_matrix.shape} vs {cpt_sub_matrix.shape}"
        )
    cpt_normalized = normalize(cpt_matrix, norm="l2", axis=1, copy=True)
    cpt_sub_normalized = normalize(cpt_sub_matrix, norm="l2", axis=1, copy=True)
    cpt_coordinates = run_tsne(cpt_normalized, args)
    cpt_sub_coordinates = run_tsne(
        cpt_sub_normalized, args, cpt_coordinates.copy()
    )
    return cpt_coordinates, cpt_sub_coordinates, 0

def fit_projection(
    cpt_matrix: np.ndarray,
    cpt_sub_matrix: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    if args.method == "tsne":
        return fit_aligned_tsne(cpt_matrix, cpt_sub_matrix, args)
    return fit_aligned_umap(cpt_matrix, cpt_sub_matrix, args)

def select_visualization_indices(
    groups: np.ndarray,
    colored_groups: Sequence[str],
    balance_round_to: int,
    max_gray: int,
    seed: int,
) -> tuple[np.ndarray, int, dict[str, int], dict[str, int], int, int]:
    """Balance coloured EC groups and optionally subsample grey samples."""
    rng = np.random.default_rng(seed)
    group_indices = {label: np.flatnonzero(groups == label) for label in colored_groups}
    if any(len(indices) == 0 for indices in group_indices.values()):
        missing = [label for label, indices in group_indices.items() if len(indices) == 0]
        raise ValueError(f"No samples available for highlighted EC groups: {missing}")
    min_count = min(len(indices) for indices in group_indices.values())
    if balance_round_to > 0 and min_count >= balance_round_to:
        target = (min_count // balance_round_to) * balance_round_to
    else:
        target = min_count
    colored = [
        np.sort(rng.choice(group_indices[label], size=target, replace=False))
        for label in colored_groups
    ]
    grey = np.flatnonzero(~np.isin(groups, colored_groups))
    n_gray_before = len(grey)
    if len(grey) > max_gray:
        grey = np.sort(rng.choice(grey, size=max_gray, replace=False))
    before_counts = {label: len(group_indices[label]) for label in colored_groups}
    plotted_counts = {label: target for label in colored_groups}
    return (
        np.concatenate([*colored, grey]).astype(int),
        target,
        before_counts,
        plotted_counts,
        n_gray_before,
        len(grey),
    )

def coloured_silhouette(
    matrix: np.ndarray, groups: np.ndarray, colored_groups: Sequence[str]
) -> float | None:
    mask = np.isin(groups, colored_groups)
    labels = groups[mask]
    if len(np.unique(labels)) < 2 or len(labels) <= len(np.unique(labels)):
        return None
    normalized = normalize(matrix[mask], norm="l2", axis=1, copy=True)
    return float(silhouette_score(normalized, labels, metric="cosine"))

def build_state_records(
    labels: pd.DataFrame,
    id_column: str,
    state: str,
    coordinates: np.ndarray,
    class_to_color: dict[str, str],
    sampling_seed: int,
) -> pd.DataFrame:
    records = pd.DataFrame(
        {
            "seq_id": labels[id_column].tolist(),
            "state": state,
            "ec_labels_full": labels["ec_labels_full"].map(";".join),
            "n_ec_labels_full": labels["ec_labels_full"].map(len),
            "is_single_label": labels["ec_labels_full"].map(len).eq(1),
            "single_label_complete_ec4": labels["single_label_ec4"].tolist(),
            "highlight_ec": labels["highlight_ec"].tolist(),
            "is_colored": labels["highlight_ec"].ne("Other").tolist(),
            "plot_color": labels["highlight_ec"].map(
                lambda label: class_to_color.get(label, OTHER_COLOR)
            ),
            "n_highlight_ec_matches": labels["n_highlight_ec_matches"].tolist(),
            "highlight_ec_split10_count": labels["highlight_ec_count"].tolist(),
            "sampling_seed": sampling_seed,
            "dim_1": coordinates[:, 0],
            "dim_2": coordinates[:, 1],
        }
    )
    return records

def draw_figure(
    records: pd.DataFrame,
    selected_classes: Sequence[str],
    output_dir: Path,
    output_stem: str,
    method: str,
    silhouette_scores: dict[str, float | None],
) -> None:
    class_to_color = {
        label: CLASS_COLORS[index] for index, label in enumerate(selected_classes)
    }
    figure = plt.figure(figsize=FIGURE_SIZE, constrained_layout=False)
    axes = [
        figure.add_axes((0.055, 0.275, 0.415, 0.56)),
        figure.add_axes((0.530, 0.275, 0.415, 0.56)),
    ]
    method_label = "UMAP" if method == "umap" else "t-SNE"

    # Aligned projections must also share one display window.  Independent
    # limits make both states fill their axes and can visually suppress genuine
    # contraction, expansion, or displacement.
    all_coordinates = records[["dim_1", "dim_2"]].to_numpy(dtype=float)
    lower = all_coordinates.min(axis=0)
    upper = all_coordinates.max(axis=0)
    center = (lower + upper) / 2.0
    span = max(float((upper - lower).max()), 1.0)
    half = span * 0.575

    # Choose one low-density corner from the combined aligned map and reuse it
    # in both panels, keeping the score labels aligned without covering the
    # dominant point clouds.
    normalized = (all_coordinates - (center - half)) / (2.0 * half)
    corner_candidates = (
        (0.035, 0.965, "left", "top", (normalized[:, 0] < 0.30) & (normalized[:, 1] > 0.80)),
        (0.965, 0.965, "right", "top", (normalized[:, 0] > 0.70) & (normalized[:, 1] > 0.80)),
        (0.035, 0.035, "left", "bottom", (normalized[:, 0] < 0.30) & (normalized[:, 1] < 0.20)),
        (0.965, 0.035, "right", "bottom", (normalized[:, 0] > 0.70) & (normalized[:, 1] < 0.20)),
    )
    sil_x, sil_y, sil_ha, sil_va, _ = min(
        corner_candidates, key=lambda candidate: int(candidate[4].sum())
    )

    for panel_index, (axis, state) in enumerate(zip(axes, ("CPT", "CPT-SUB"))):
        subset = records.loc[records["state"] == state]
        other = subset.loc[subset["highlight_ec"] == "Other"]
        axis.scatter(
            other["dim_1"],
            other["dim_2"],
            s=3.0,
            c=OTHER_COLOR,
            alpha=0.42,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
        for label in selected_classes:
            highlighted = subset.loc[subset["highlight_ec"] == label]
            axis.scatter(
                highlighted["dim_1"],
                highlighted["dim_2"],
                s=9.5,
                c=class_to_color[label],
                alpha=0.86,
                edgecolors="white",
                linewidths=0.18,
                rasterized=False,
                zorder=2,
            )
        axis.set_title(
            state,
            color=STATE_COLORS[state],
            fontsize=6.2,
            fontweight="bold",
            pad=2.4,
        )
        axis.set_xlabel(
            f"{method_label} 1", fontsize=5.1, labelpad=1.7, color=AXIS_COLOR
        )
        axis.set_ylabel(
            f"{method_label} 2" if panel_index == 0 else "",
            fontsize=5.1,
            labelpad=1.7,
            color=AXIS_COLOR,
        )
        axis.set_xlim(float(center[0] - half), float(center[0] + half))
        axis.set_ylim(float(center[1] - half), float(center[1] + half))
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS_COLOR)
            axis.spines[side].set_linewidth(0.60)
        score = silhouette_scores.get(state)
        score_text = (
            "N/A"
            if score is None or not np.isfinite(score)
            else f"{score:.3f}"
        )
        axis.text(
            sil_x,
            sil_y,
            f"Sil={score_text}",
            transform=axis.transAxes,
            ha=sil_ha,
            va=sil_va,
            fontsize=4.7,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#A0A0A0",
                "linewidth": 0.35,
                "alpha": 0.88,
            },
            zorder=5,
        )

    figure.text(
        0.5,
        0.965,
        "EC function space",
        ha="center",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=TEXT_COLOR,
    )

    legend_handles = [
        Patch(
            facecolor=class_to_color[label], edgecolor="white", linewidth=0.2,
            label=label,
        )
        for label in selected_classes
    ]
    legend_handles.append(
        Patch(
            facecolor=OTHER_COLOR, edgecolor="none", linewidth=0.0,
            label="Other",
        )
    )
    legend_axis = figure.add_axes((0.025, 0.015, 0.95, 0.205))
    legend_axis.axis("off")
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(4, len(legend_handles)),
        title="Complete EC4 label",
        frameon=False,
        fontsize=4.5,
        title_fontsize=4.9,
        borderpad=0.0,
        labelspacing=0.30,
        columnspacing=0.85,
        handlelength=0.75,
        handleheight=0.75,
        handletextpad=0.35,
    )
    for suffix in ("svg", "pdf", "png", "tiff"):
        figure.savefig(
            output_dir / f"{output_stem}.{suffix}",
            facecolor="none",
            dpi=OUTPUT_DPI,
            transparent=True,
        )
    plt.close(figure)

def selected_ec4_by_ec1(selected_classes: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for label in selected_classes:
        grouped.setdefault(ec1_from_ec4(label), []).append(label)
    return grouped

def run_variant(
    *,
    args: argparse.Namespace,
    base_labels: pd.DataFrame,
    id_column: str,
    ec_column: str,
    counts: Counter[str],
    cpt_file: Path,
    cpt_sub_file: Path,
    all_cpt_matrix: np.ndarray,
    all_cpt_sub_matrix: np.ndarray,
    variant_slug: str,
    selection_strategy: str,
    selection_rule: str,
    tie_break_rule: str,
    selected_classes: Sequence[str],
    tie_sizes_by_label: dict[str, int],
    global_top_n: int | None,
) -> dict[str, Any]:
    labels = base_labels.copy()
    selected_set = set(selected_classes)
    assignments = pd.Series(
        [
            assign_highlight_class(full_labels, selected_set)
            for full_labels in labels["ec_labels_full"]
        ],
        index=labels.index,
    )
    labels["highlight_ec"] = assignments.map(lambda value: value[0])
    labels["n_highlight_ec_matches"] = assignments.map(lambda value: value[1])
    labels["highlight_ec_count"] = labels["highlight_ec"].map(
        lambda label: counts[label] if label != "Other" else 0
    )
    all_groups = labels["highlight_ec"].to_numpy(dtype=object)
    (
        selected_indices,
        balance_target,
        prebalance_counts,
        plotted_counts,
        n_gray_before,
        n_gray_plotted,
    ) = select_visualization_indices(
        all_groups,
        selected_classes,
        args.balance_round_to,
        args.max_gray,
        args.random_state,
    )
    labels = labels.iloc[selected_indices].reset_index(drop=True)
    groups = all_groups[selected_indices]
    cpt_matrix = all_cpt_matrix[selected_indices]
    cpt_sub_matrix = all_cpt_sub_matrix[selected_indices]

    silhouette_scores = {
        "CPT": coloured_silhouette(cpt_matrix, groups, selected_classes),
        "CPT-SUB": coloured_silhouette(cpt_sub_matrix, groups, selected_classes),
    }
    cpt_coordinates, cpt_sub_coordinates, _ = fit_projection(
        cpt_matrix, cpt_sub_matrix, args
    )
    class_to_color = {
        label: CLASS_COLORS[index] for index, label in enumerate(selected_classes)
    }
    cpt_records = build_state_records(
        labels,
        id_column,
        "CPT",
        cpt_coordinates,
        class_to_color,
        args.random_state,
    )
    cpt_sub_records = build_state_records(
        labels,
        id_column,
        "CPT-SUB",
        cpt_sub_coordinates,
        class_to_color,
        args.random_state,
    )
    records = pd.concat((cpt_records, cpt_sub_records), ignore_index=True)
    records.insert(0, "selection_strategy", selection_strategy)

    output_stem = (
        f"ec_split10_{variant_slug}_cpt_cpt_sub_aligned_{args.method}"
    )
    records.to_csv(
        args.output_dir / f"{output_stem}_source_data.csv", index=False
    )
    selection_rows = [
        {
            "selection_strategy": selection_strategy,
            "selection_rank": rank,
            "ec1_class": f"EC {ec1_from_ec4(label)}",
            "ec1_number": int(ec1_from_ec4(label)),
            "selected_ec4": label,
            "original_frequency_single_label": counts[label],
            "n_before_balance": prebalance_counts[label],
            "n_plotted_per_panel": plotted_counts[label],
            "n_plotted_across_two_panels": 2 * plotted_counts[label],
            "balance_target": balance_target,
            "frequency_tie_size": tie_sizes_by_label[label],
            "tie_break_rule": tie_break_rule,
            "color_hex": class_to_color[label],
        }
        for rank, label in enumerate(selected_classes, start=1)
    ]
    statistics_name = f"{output_stem}_selection_statistics.csv"
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / statistics_name, index=False
    )

    n_samples = len(base_labels)
    n_singlelabel_rows = int(base_labels["ec_labels_full"].map(len).eq(1).sum())
    n_multilabel_rows = int(base_labels["ec_labels_full"].map(len).gt(1).sum())
    n_complete_singlelabel_rows = int(
        base_labels["single_label_ec4"].ne("").sum()
    )
    run_config = RunConfig(
        variant_slug=variant_slug,
        selection_strategy=selection_strategy,
        global_top_n=global_top_n,
        split10_csv=str(args.split10_csv),
        cpt_embedding_file=str(cpt_file),
        cpt_sub_embedding_file=str(cpt_sub_file),
        id_column=id_column,
        ec_column=ec_column,
        ec_level=4,
        ec_selection_rule=selection_rule,
        selected_ec_classes=list(selected_classes),
        selected_ec4_by_ec1=selected_ec4_by_ec1(selected_classes),
        selected_ec_raw_frequencies={
            label: counts[label] for label in selected_classes
        },
        selected_ec_prebalance_counts=prebalance_counts,
        selected_ec_plotted_counts_per_panel=plotted_counts,
        assignment_rule=(
            "Colour only single-label rows matching the selected EC4. Multi-label, "
            "incomplete, and unselected rows are grey. Excess selected rows removed "
            "by class balancing are not reclassified as grey."
        ),
        n_split10_rows=n_samples,
        n_singlelabel_rows=n_singlelabel_rows,
        n_multilabel_rows=n_multilabel_rows,
        n_singlelabel_complete_ec4_rows=n_complete_singlelabel_rows,
        n_singlelabel_incomplete_ec_rows=(
            n_singlelabel_rows - n_complete_singlelabel_rows
        ),
        n_unique_singlelabel_complete_ec4=len(counts),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        method=args.method,
        perplexity=args.perplexity,
        tsne_iterations=args.tsne_iterations,
        cpt_tsne_initialization="random",
        cpt_sub_tsne_initialization="CPT 2D coordinates",
        tsne_alignment=TSNE_ALIGNMENT,
        projection_basis=PROJECTION_BASIS,
        silhouette_space=(
            "original L2-normalized high-dimensional embeddings; coloured EC4 "
            "samples only; cosine distance"
        ),
        projection_population_rule=(
            "Each selection variant is balanced, sampled, and projected independently. "
            "Compare CPT with CPT-SUB within a variant; do not compare coordinates or "
            "Silhouette values across variants as if they used one sample population."
        ),
        figure_size_inches=list(FIGURE_SIZE),
        figure_size_mm=list(FIGURE_SIZE_MM),
        output_dpi=OUTPUT_DPI,
        balance_round_to=args.balance_round_to,
        max_gray=args.max_gray,
        balanced_colored_per_group=balance_target,
        n_gray_before_sampling=n_gray_before,
        n_gray_plotted_per_panel=n_gray_plotted,
        n_visualization_samples=len(labels),
        sampling_seeds={
            "colored_and_gray_master_numpy_seed": args.random_state,
        },
        sampling_algorithm=(
            "numpy.random.default_rng(seed), selected classes in recorded legend "
            "order, then grey background"
        ),
        silhouette_scores=silhouette_scores,
    )
    metadata_name = f"{output_stem}_metadata.json"
    with (args.output_dir / metadata_name).open("w", encoding="utf-8") as handle:
        json.dump(
            asdict(run_config), handle, indent=2, ensure_ascii=False, allow_nan=False
        )

    draw_figure(
        records,
        selected_classes,
        args.output_dir,
        output_stem,
        args.method,
        silhouette_scores,
    )
    print(
        f"[{selection_strategy}] samples per state: {len(labels)}; "
        f"highlighted: {', '.join(selected_classes)}"
    )
    return {
        "variant_slug": variant_slug,
        "selection_strategy": selection_strategy,
        "selected_ec_classes": list(selected_classes),
        "balance_target": balance_target,
        "n_visualization_samples_per_state": len(labels),
        "silhouette_scores": silhouette_scores,
        "files": {
            "svg": f"{output_stem}.svg",
            "pdf": f"{output_stem}.pdf",
            "png": f"{output_stem}.png",
            "source_data": f"{output_stem}_source_data.csv",
            "selection_statistics": statistics_name,
            "metadata": metadata_name,
        },
    }

def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if args.balance_round_to < 0:
        raise ValueError("--balance-round-to cannot be negative")
    if args.max_gray < 1 or args.max_gray > MAX_GRAY_LIMIT:
        raise ValueError(f"--max-gray must be between 1 and {MAX_GRAY_LIMIT}")
    if args.global_top_n is not None and (
        args.global_top_n < 1 or args.global_top_n > len(CLASS_COLORS)
    ):
        raise ValueError(
            f"--global-top-n must be between 1 and {len(CLASS_COLORS)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels, id_column, ec_column = load_split10(
        args.split10_csv, args.id_column, args.ec_column
    )
    counts = count_single_label_complete_ec4(labels)
    per_ec1_classes, per_ec1_ties = choose_per_ec1_high_frequency_classes(counts)
    global_top_n = args.global_top_n or len(per_ec1_classes)
    global_classes, global_ties = choose_global_high_frequency_classes(
        counts, global_top_n
    )

    cpt_file = resolve_embedding_file(args.cpt_emb_root, args.embedding_file)
    cpt_sub_file = resolve_embedding_file(args.cpt_sub_emb_root, args.embedding_file)
    ordered_ids = labels[id_column].tolist()
    all_cpt_matrix = load_embeddings(cpt_file, ordered_ids)
    all_cpt_sub_matrix = load_embeddings(cpt_sub_file, ordered_ids)
    if all_cpt_matrix.shape != all_cpt_sub_matrix.shape:
        raise ValueError(
            "CPT/CPT-SUB matrices differ: "
            f"{all_cpt_matrix.shape} vs {all_cpt_sub_matrix.shape}"
        )

    manifests = [
        run_variant(
            args=args,
            base_labels=labels,
            id_column=id_column,
            ec_column=ec_column,
            counts=counts,
            cpt_file=cpt_file,
            cpt_sub_file=cpt_sub_file,
            all_cpt_matrix=all_cpt_matrix,
            all_cpt_sub_matrix=all_cpt_sub_matrix,
            variant_slug=f"global_top{global_top_n}",
            selection_strategy="global_top_n",
            selection_rule=(
                f"Select the globally most frequent {global_top_n} complete "
                "numeric EC4 labels among single-label rows only; resolve exact "
                "ties by lexical EC4 order."
            ),
            tie_break_rule="global frequency descending, then lexical EC4 order",
            selected_classes=global_classes,
            tie_sizes_by_label=global_ties,
            global_top_n=global_top_n,
        ),
        run_variant(
            args=args,
            base_labels=labels,
            id_column=id_column,
            ec_column=ec_column,
            counts=counts,
            cpt_file=cpt_file,
            cpt_sub_file=cpt_sub_file,
            all_cpt_matrix=all_cpt_matrix,
            all_cpt_sub_matrix=all_cpt_sub_matrix,
            variant_slug="per_ec1_top1",
            selection_strategy="per_ec1_top1",
            selection_rule=(
                "For every EC1 class, select the most frequent complete numeric "
                "EC4 among single-label rows only; resolve exact within-EC1 ties "
                "by lexical EC4 order."
            ),
            tie_break_rule=(
                "within-EC1 frequency descending, then lexical EC4 order"
            ),
            selected_classes=per_ec1_classes,
            tie_sizes_by_label=per_ec1_ties,
            global_top_n=None,
        ),
    ]
    manifest = {
        "script": Path(__file__).name,
        "method": args.method,
        "random_state": args.random_state,
        "global_top_n": global_top_n,
        "global_top_n_source": (
            "command line" if args.global_top_n is not None
            else "number of represented EC1 classes"
        ),
        "figure_size_mm": list(FIGURE_SIZE_MM),
        "variants": manifests,
    }
    manifest_path = args.output_dir / "ec_split10_all_variants_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Wrote both EC variants and manifest to: {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
