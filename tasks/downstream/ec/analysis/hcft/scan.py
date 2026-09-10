#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select HCFT case anchors from the official anchor-level summary.

Edit CONFIG, then run:

    python scan_hcft_case_anchors.py

CPT-SUB is used here only as a screening signal: the selected anchor must fail
under Base and improve clearly under the final model. The exact sampled
triplets are inspected only to require a diverse positive/negative pool; no
embedding neighbourhood is constructed and no figure data are produced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIG: edit only this section
# =============================================================================
CONFIG = {
    "anchor_summary": (
        "artifacts/downstream/ec/hcft/"
        "experiments/06B_strict/hcft_anchor_summary.csv"
    ),
    "triplets_csv": (
        "artifacts/downstream/ec/hcft/"
        "experiments/06B_strict_triplets/triplets/"
        "sampled_triplets_seed0.csv"
    ),
    "base_name": "esm2_base",
    "final_name": "esm2_cpt_sub",
    "seed": 0,

    # A moderate triplet count makes the anchor statistically credible while
    # keeping the all-triplet case panel readable.
    "min_triplets": 80,
    "max_triplets": 300,

    # Base must show an anchor-level HCFT failure.
    "max_base_acc": 0.40,
    "max_base_margin": 0.0,

    # The final model must show a clear repair. These conditions are used only
    # to select the anchor; final-model values need not appear in figure d/e.
    "min_final_acc": 0.80,
    "min_final_margin": 0.03,
    "min_delta_accuracy": 0.40,
    "min_delta_margin": 0.05,

    # Triplet-pool diversity. These constraints exclude anchors whose apparent
    # repeated failure is generated mainly by one negative protein.
    "min_unique_positives": 10,
    "min_unique_negatives": 5,
    "min_unique_negative_ec4": 2,
    "max_single_negative_fraction": 0.35,

    # EC annotation is required for negative-EC4 diversity.
    "label_csv": "data/ec/split100.csv",
    "require_single_complete_ec4": True,

    "output_candidates": (
        "artifacts/downstream/ec/hcft/case_selection/"
        "06B_strict/case_anchor_candidates.csv"
    ),
    "output_all": (
        "artifacts/downstream/ec/hcft/case_selection/"
        "06B_strict/case_anchor_screen_all.csv"
    ),
}
# =============================================================================

def as_path(value: str | Path, name: str, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"CONFIG[{name!r}] is not a file: {path}")
    return path

def validate_config() -> None:
    if int(CONFIG["min_triplets"]) < 1:
        raise ValueError("CONFIG['min_triplets'] must be positive")
    if int(CONFIG["max_triplets"]) < int(CONFIG["min_triplets"]):
        raise ValueError("CONFIG['max_triplets'] must be >= min_triplets")
    for key in (
        "min_unique_positives",
        "min_unique_negatives",
        "min_unique_negative_ec4",
    ):
        if int(CONFIG[key]) < 1:
            raise ValueError(f"CONFIG[{key!r}] must be positive")
    for key in (
        "max_base_acc",
        "min_final_acc",
        "min_delta_accuracy",
        "max_single_negative_fraction",
    ):
        value = float(CONFIG[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"CONFIG[{key!r}] must be between 0 and 1")

def ec_at_level(labels: list[str], level: int) -> set[str]:
    result: set[str] = set()
    for label in labels:
        parts = label.split(".")
        if (
            len(parts) >= level
            and all(part and part != "-" for part in parts[:level])
        ):
            result.add(".".join(parts[:level]))
    return result

def read_label_table(path: Path) -> dict[str, list[str]]:
    for separator in ("\t", ","):
        frame = pd.read_csv(path, sep=separator, dtype=str, na_filter=False)
        if len(frame.columns) >= 2:
            break
    else:  # pragma: no cover
        raise ValueError(f"Could not parse a label table from {path}")

    id_column = next(
        (
            column
            for column in ("Entry", "entry", "id", "ID", "seq_id", "uniprot_id")
            if column in frame
        ),
        frame.columns[0],
    )
    ec_column = next(
        (
            column
            for column in ("EC number", "ec_number", "EC", "ec", "label")
            if column in frame
        ),
        frame.columns[1],
    )
    return {
        str(identifier).strip(): [
            item.strip()
            for item in str(raw_labels).split(";")
            if item.strip()
        ]
        for identifier, raw_labels in zip(frame[id_column], frame[ec_column])
        if str(identifier).strip()
    }

def load_paired_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path, dtype={"anchor": str})
    required = {
        "embedding",
        "seed",
        "anchor",
        "n_triplets",
        "hcft_acc",
        "hcft_margin_mean",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for column in ("seed", "n_triplets", "hcft_acc", "hcft_margin_mean"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    summary = summary.dropna(subset=list(required)).copy()
    summary["seed"] = summary["seed"].astype(int)
    summary = summary.loc[summary["seed"].eq(int(CONFIG["seed"]))]

    base = summary.loc[
        summary["embedding"].astype(str).eq(str(CONFIG["base_name"]))
    ].set_index("anchor")
    final = summary.loc[
        summary["embedding"].astype(str).eq(str(CONFIG["final_name"]))
    ].set_index("anchor")
    if base.empty or final.empty:
        available = sorted(summary["embedding"].astype(str).unique())
        raise ValueError(
            "Configured embedding names were not both found. "
            f"Available at seed {CONFIG['seed']}: {available}"
        )
    if base.index.has_duplicates or final.index.has_duplicates:
        raise ValueError("Expected one row per anchor/embedding/seed")

    paired = base[["n_triplets", "hcft_acc", "hcft_margin_mean"]].join(
        final[["n_triplets", "hcft_acc", "hcft_margin_mean"]],
        how="inner",
        lsuffix="_base",
        rsuffix="_final",
        validate="one_to_one",
    )
    paired["triplet_count_consistent"] = paired["n_triplets_base"].eq(
        paired["n_triplets_final"]
    )
    paired["delta_accuracy"] = (
        paired["hcft_acc_final"] - paired["hcft_acc_base"]
    )
    paired["delta_margin"] = (
        paired["hcft_margin_mean_final"]
        - paired["hcft_margin_mean_base"]
    )
    return paired

def load_triplet_diversity(
    path: Path,
    labels: dict[str, list[str]],
) -> pd.DataFrame:
    """Summarize independent positive/negative support for each anchor."""
    triplets = pd.read_csv(
        path,
        dtype={"anchor": str, "positive": str, "negative": str},
    )
    required = {"seed", "anchor", "positive", "negative"}
    missing = required - set(triplets.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    triplets["seed"] = pd.to_numeric(triplets["seed"], errors="coerce")
    triplets = triplets.dropna(subset=list(required)).copy()
    triplets["seed"] = triplets["seed"].astype(int)
    triplets = triplets.loc[
        triplets["seed"].eq(int(CONFIG["seed"]))
    ].copy()
    if triplets.empty:
        raise ValueError(f"No triplets for seed={CONFIG['seed']} in {path}")
    if triplets.duplicated(["anchor", "positive", "negative"]).any():
        raise ValueError("Exact triplet file contains duplicate triplets")

    records: list[dict[str, object]] = []
    for anchor, group in triplets.groupby("anchor", sort=False):
        negative_counts = group["negative"].value_counts()
        negative_ec4: set[str] = set()
        negatives_with_complete_ec4 = 0
        for negative in negative_counts.index:
            ec4 = ec_at_level(labels.get(str(negative), []), 4)
            negative_ec4.update(ec4)
            negatives_with_complete_ec4 += bool(ec4)
        records.append({
            "anchor": anchor,
            "n_triplets_exact": len(group),
            "n_unique_positives": group["positive"].nunique(),
            "n_unique_negatives": group["negative"].nunique(),
            "n_unique_negative_ec4": len(negative_ec4),
            "n_negatives_with_complete_ec4": negatives_with_complete_ec4,
            "top_negative": str(negative_counts.index[0]),
            "top_negative_triplets": int(negative_counts.iloc[0]),
            "top_negative_fraction": float(
                negative_counts.iloc[0] / len(group)
            ),
        })
    return pd.DataFrame(records).set_index("anchor")

def main() -> None:
    validate_config()
    summary_path = as_path(CONFIG["anchor_summary"], "anchor_summary")
    paired = load_paired_summary(summary_path)

    label_path_value = CONFIG.get("label_csv")
    if not label_path_value:
        raise ValueError("CONFIG['label_csv'] is required")
    labels = read_label_table(as_path(label_path_value, "label_csv"))
    paired["anchor_ec4"] = [
        ";".join(sorted(ec_at_level(labels.get(anchor, []), 4)))
        for anchor in paired.index
    ]
    paired["n_complete_ec4"] = [
        len(ec_at_level(labels.get(anchor, []), 4))
        for anchor in paired.index
    ]
    diversity = load_triplet_diversity(
        as_path(CONFIG["triplets_csv"], "triplets_csv"),
        labels,
    )
    paired = paired.join(diversity, how="left", validate="one_to_one")
    paired["triplet_file_count_matches"] = paired[
        "n_triplets_base"
    ].eq(paired["n_triplets_exact"])

    passes = (
        paired["triplet_count_consistent"]
        & paired["triplet_file_count_matches"]
        & paired["n_triplets_base"].between(
            int(CONFIG["min_triplets"]),
            int(CONFIG["max_triplets"]),
            inclusive="both",
        )
        & paired["hcft_acc_base"].le(float(CONFIG["max_base_acc"]))
        & paired["hcft_margin_mean_base"].le(float(CONFIG["max_base_margin"]))
        & paired["hcft_acc_final"].ge(float(CONFIG["min_final_acc"]))
        & paired["hcft_margin_mean_final"].ge(float(CONFIG["min_final_margin"]))
        & paired["delta_accuracy"].ge(float(CONFIG["min_delta_accuracy"]))
        & paired["delta_margin"].ge(float(CONFIG["min_delta_margin"]))
        & paired["n_unique_positives"].ge(
            int(CONFIG["min_unique_positives"])
        )
        & paired["n_unique_negatives"].ge(
            int(CONFIG["min_unique_negatives"])
        )
        & paired["n_unique_negative_ec4"].ge(
            int(CONFIG["min_unique_negative_ec4"])
        )
        & paired["top_negative_fraction"].le(
            float(CONFIG["max_single_negative_fraction"])
        )
    )
    if bool(CONFIG["require_single_complete_ec4"]):
        passes &= paired["n_complete_ec4"].eq(1)

    paired["passes_case_filter"] = passes
    paired.index.name = "anchor"
    paired = paired.reset_index().sort_values(
        [
            "passes_case_filter",
            "delta_accuracy",
            "hcft_acc_base",
            "delta_margin",
            "n_triplets_base",
        ],
        ascending=[False, False, True, False, True],
    )
    candidates = paired.loc[paired["passes_case_filter"]].copy()
    candidates.insert(0, "candidate_rank", range(1, len(candidates) + 1))

    output_all = as_path(CONFIG["output_all"], "output_all", must_exist=False)
    output_candidates = as_path(
        CONFIG["output_candidates"],
        "output_candidates",
        must_exist=False,
    )
    output_all.parent.mkdir(parents=True, exist_ok=True)
    output_candidates.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_all, index=False)
    candidates.to_csv(output_candidates, index=False)

    print(f"Paired anchors screened: {len(paired)}")
    print(f"Case candidates: {len(candidates)}")
    print(f"wrote {output_candidates}")
    print(f"wrote {output_all}")
    if not candidates.empty:
        columns = [
            "candidate_rank",
            "anchor",
            "anchor_ec4",
            "n_triplets_base",
            "hcft_acc_base",
            "hcft_acc_final",
            "delta_accuracy",
            "hcft_margin_mean_base",
            "hcft_margin_mean_final",
            "delta_margin",
            "n_unique_positives",
            "n_unique_negatives",
            "n_unique_negative_ec4",
            "top_negative_fraction",
        ]
        print(candidates[columns].to_string(index=False))

if __name__ == "__main__":
    main()
