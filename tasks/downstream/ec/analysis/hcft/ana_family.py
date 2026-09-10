#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate anchor-balanced HCFT accuracy and margin for every EC3 family.

Usage
-----
    python ana_family.py --config test.yaml

Input
-----
1. ``hcft_anchor_summary.csv`` produced by ``hcft_eval.py``.
2. An EC label table such as ``split100.csv``.

Family definition
-----------------
Only anchors with exactly one complete EC4 label are retained. The first three
EC levels define the EC3 family:

    1.14.14.80 -> 1.14.14

Statistical unit
----------------
For each embedding, seed and EC3 family, HCFT accuracy and margin are averaged
across unique anchors. Every anchor has equal weight; triplets are not treated
as independent observations. Results are then averaged across seeds.

Outputs
-------
``ec3_family_scores_by_seed.csv``
    Per embedding / seed / EC3 family anchor-balanced scores.

``ec3_family_scores.csv``
    Final per embedding / EC3 family scores across seeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REQUIRED_SUMMARY_COLUMNS = {
    "embedding",
    "seed",
    "anchor",
    "n_triplets",
    "hcft_acc",
    "hcft_margin_mean",
}

ID_COLUMN_CANDIDATES = (
    "Entry",
    "entry",
    "id",
    "ID",
    "seq_id",
    "uniprot_id",
)

EC_COLUMN_CANDIDATES = (
    "EC number",
    "EC_number",
    "ec_number",
    "EC",
    "ec",
    "label",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HCFT accuracy and margin for every EC3 family."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file.",
    )
    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required. Install it with: pip install pyyaml"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config

def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()

def require_config(config: dict, key: str):
    value = config.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required config key: {key}")
    return value

def resolve_summary_csv(path: Path) -> Path:
    if path.is_dir():
        path = path / "hcft_anchor_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"HCFT anchor summary does not exist: {path}")
    return path

def sniff_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","

def resolve_column(
    columns: Iterable[str],
    explicit: Optional[str],
    candidates: Iterable[str],
    role: str,
) -> str:
    columns = list(columns)
    if explicit:
        if explicit not in columns:
            raise ValueError(
                f"Configured {role} column {explicit!r} was not found. "
                f"Available columns: {columns}"
            )
        return explicit

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not infer the {role} column. Available columns: {columns}. "
        f"Set the column name explicitly in the YAML config."
    )

def normalize_ec4(raw_value: object) -> Optional[str]:
    if raw_value is None or pd.isna(raw_value):
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    lower = value.lower()
    if lower.startswith("ec "):
        value = value[3:].strip()
    elif lower.startswith("ec:"):
        value = value[3:].strip()

    parts = [part.strip() for part in value.split(".")]
    if len(parts) < 4:
        return None
    if any(not part or "-" in part for part in parts[:4]):
        return None
    return ".".join(parts[:4])

def split_ec_field(raw_value: object) -> list[str]:
    if raw_value is None or pd.isna(raw_value):
        return []
    return [
        item.strip()
        for item in str(raw_value).split(";")
        if item.strip()
    ]

def load_single_ec4_family_map(
    label_csv: Path,
    id_column: Optional[str],
    ec_column: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    if not label_csv.is_file():
        raise FileNotFoundError(f"EC label table does not exist: {label_csv}")

    separator = sniff_separator(label_csv)
    labels = pd.read_csv(
        label_csv,
        sep=separator,
        dtype=str,
        keep_default_na=False,
    )

    id_column = resolve_column(
        labels.columns,
        id_column,
        ID_COLUMN_CANDIDATES,
        "ID",
    )
    ec_column = resolve_column(
        labels.columns,
        ec_column,
        EC_COLUMN_CANDIDATES,
        "EC",
    )

    anchor_to_ec4: dict[str, set[str]] = {}
    for anchor, raw_ecs in labels[[id_column, ec_column]].itertuples(
        index=False,
        name=None,
    ):
        anchor = str(anchor).strip()
        if not anchor:
            continue

        valid_ec4 = {
            ec4
            for raw_ec in split_ec_field(raw_ecs)
            if (ec4 := normalize_ec4(raw_ec)) is not None
        }
        if valid_ec4:
            anchor_to_ec4.setdefault(anchor, set()).update(valid_ec4)

    rows = []
    n_multilabel_excluded = 0
    for anchor, ec4_labels in anchor_to_ec4.items():
        if len(ec4_labels) != 1:
            n_multilabel_excluded += 1
            continue
        ec4 = next(iter(ec4_labels))
        rows.append(
            {
                "anchor": anchor,
                "anchor_ec4": ec4,
                "ec3_family": ".".join(ec4.split(".")[:3]),
            }
        )

    family_map = pd.DataFrame(rows)
    if family_map.empty:
        raise ValueError(
            "No anchors with exactly one complete EC4 label were found."
        )
    if family_map["anchor"].duplicated().any():
        raise ValueError("The EC label table produced duplicate anchor mappings.")

    diagnostics = {
        "id_column": id_column,
        "ec_column": ec_column,
        "n_labeled_anchors": len(anchor_to_ec4),
        "n_single_ec4_anchors": len(family_map),
        "n_multilabel_anchors_excluded": n_multilabel_excluded,
        "n_ec3_families_in_label_table": family_map["ec3_family"].nunique(),
    }
    return family_map, diagnostics

def load_anchor_summary(
    summary_csv: Path,
    embeddings_config,
) -> tuple[pd.DataFrame, list[str]]:
    summary = pd.read_csv(summary_csv, dtype={"anchor": str})
    missing = REQUIRED_SUMMARY_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(
            f"{summary_csv} is missing required columns: {sorted(missing)}"
        )

    for column in [
        "seed",
        "n_triplets",
        "hcft_acc",
        "hcft_margin_mean",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    summary["anchor"] = summary["anchor"].astype(str)
    summary["embedding"] = summary["embedding"].astype(str)
    summary = summary.dropna(
        subset=[
            "anchor",
            "embedding",
            "seed",
            "n_triplets",
            "hcft_acc",
            "hcft_margin_mean",
        ]
    ).copy()
    summary["seed"] = summary["seed"].astype(int)
    summary["n_triplets"] = summary["n_triplets"].astype(int)

    available = sorted(summary["embedding"].unique().tolist())
    if embeddings_config == "all":
        selected = available
    elif isinstance(embeddings_config, list) and embeddings_config:
        selected = [str(item) for item in embeddings_config]
    else:
        raise ValueError(
            "Config key 'embeddings' must be 'all' or a non-empty list."
        )

    missing_embeddings = [
        embedding for embedding in selected if embedding not in available
    ]
    if missing_embeddings:
        raise ValueError(
            f"Embeddings not found in {summary_csv}: {missing_embeddings}. "
            f"Available embeddings: {available}"
        )

    summary = summary[summary["embedding"].isin(selected)].copy()

    duplicate_keys = ["embedding", "seed", "anchor"]
    duplicates = int(summary.duplicated(duplicate_keys).sum())
    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate rows for keys {duplicate_keys}. "
            "The input may contain multiple HCFT configurations."
        )

    return summary, selected

def summarize_by_seed(mapped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["embedding", "seed", "ec3_family"]

    for (embedding, seed, family), group in mapped.groupby(
        group_columns,
        sort=False,
    ):
        accuracy = group["hcft_acc"].to_numpy(dtype=float)
        margin = group["hcft_margin_mean"].to_numpy(dtype=float)

        rows.append(
            {
                "embedding": embedding,
                "seed": int(seed),
                "ec3_family": family,
                "n_anchors": int(group["anchor"].nunique()),
                "n_ec4": int(group["anchor_ec4"].nunique()),
                "n_triplets_total": int(group["n_triplets"].sum()),
                "hcft_acc": float(np.mean(accuracy)),
                "hcft_acc_anchor_std": float(np.std(accuracy, ddof=0)),
                "hcft_margin": float(np.mean(margin)),
                "hcft_margin_anchor_std": float(np.std(margin, ddof=0)),
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No EC3 family scores could be computed.")
    return output.sort_values(
        ["embedding", "seed", "hcft_acc", "ec3_family"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

def summarize_across_seeds(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (embedding, family), group in by_seed.groupby(
        ["embedding", "ec3_family"],
        sort=False,
    ):
        rows.append(
            {
                "embedding": embedding,
                "ec3_family": family,
                "n_seeds": int(group["seed"].nunique()),
                "n_anchors_mean": float(group["n_anchors"].mean()),
                "n_anchors_min": int(group["n_anchors"].min()),
                "n_anchors_max": int(group["n_anchors"].max()),
                "n_ec4_mean": float(group["n_ec4"].mean()),
                "n_triplets_total_mean": float(
                    group["n_triplets_total"].mean()
                ),
                "hcft_acc_mean": float(group["hcft_acc"].mean()),
                "hcft_acc_seed_std": float(
                    group["hcft_acc"].std(ddof=0)
                ),
                "hcft_margin_mean": float(group["hcft_margin"].mean()),
                "hcft_margin_seed_std": float(
                    group["hcft_margin"].std(ddof=0)
                ),
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No across-seed EC3 family scores could be computed.")
    return output.sort_values(
        ["embedding", "hcft_acc_mean", "ec3_family"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    config_dir = config_path.parent

    summary_csv = resolve_summary_csv(
        resolve_path(
            require_config(config, "anchor_summary_csv"),
            config_dir,
        )
    )
    label_csv = resolve_path(
        require_config(config, "label_csv"),
        config_dir,
    )
    output_dir = resolve_path(
        require_config(config, "output_dir"),
        config_dir,
    )
    embeddings_config = require_config(config, "embeddings")
    id_column = config.get("id_column")
    ec_column = config.get("ec_column")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EC3 family HCFT evaluation")
    print("=" * 78)
    print(f"anchor summary : {summary_csv}")
    print(f"label table    : {label_csv}")
    print(f"output dir     : {output_dir}")

    family_map, label_diagnostics = load_single_ec4_family_map(
        label_csv=label_csv,
        id_column=id_column,
        ec_column=ec_column,
    )
    summary, selected_embeddings = load_anchor_summary(
        summary_csv=summary_csv,
        embeddings_config=embeddings_config,
    )

    mapped = summary.merge(
        family_map,
        on="anchor",
        how="inner",
        validate="many_to_one",
    )
    if mapped.empty:
        raise ValueError(
            "No anchors overlap between the HCFT summary and EC family map."
        )

    by_seed = summarize_by_seed(mapped)
    final_scores = summarize_across_seeds(by_seed)

    by_seed_path = output_dir / "ec3_family_scores_by_seed.csv"
    final_path = output_dir / "ec3_family_scores.csv"
    by_seed.to_csv(by_seed_path, index=False)
    final_scores.to_csv(final_path, index=False)

    summary_anchors = set(summary["anchor"])
    mapped_anchors = set(mapped["anchor"])

    print("\n[label mapping]")
    for key, value in label_diagnostics.items():
        print(f"  {key:>30s}: {value}")
    print(f"  {'HCFT anchors selected':>30s}: {len(summary_anchors)}")
    print(f"  {'HCFT anchors mapped':>30s}: {len(mapped_anchors)}")
    print(
        f"  {'HCFT anchors excluded':>30s}: "
        f"{len(summary_anchors - mapped_anchors)}"
    )

    print("\n[embeddings]")
    for embedding in selected_embeddings:
        n_families = final_scores.loc[
            final_scores["embedding"] == embedding,
            "ec3_family",
        ].nunique()
        print(f"  {embedding}: {n_families} EC3 families")

    print("\n[outputs]")
    print(f"  {by_seed_path}")
    print(f"  {final_path}")
    print("\n[done]")

if __name__ == "__main__":
    main()