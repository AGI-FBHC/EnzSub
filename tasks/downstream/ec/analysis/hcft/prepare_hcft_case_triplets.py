#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare Base-only exact HCFT triplets for case figures d/e.

Edit CONFIG, then run:

    python prepare_hcft_case_triplets.py

The input is the exact ``sampled_triplets_seed*.csv`` saved by hcft_eval.py.
No triplets are reconstructed and no sequence identities are recomputed.
Only Base embedding similarities are calculated. The final model is deliberately
absent because these outputs support the Base-failure section of the results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =============================================================================
# CONFIG: edit only this section
# =============================================================================
CONFIG = {
    # Replace/add anchors after inspecting case_anchor_candidates.csv.
    "selected_anchors": ["A4VLZ2"],

    # Exact triplets from the strict/general30 HCFT run.
    "triplets_csv": (
        "artifacts/downstream/ec/hcft/"
        "experiments/06B_strict_triplets/triplets/"
        "sampled_triplets_seed0.csv"
    ),
    "anchor_summary": (
        "artifacts/downstream/ec/hcft/"
        "experiments/06B_strict_triplets/hcft_anchor_summary.csv"
    ),
    "base_embedding_dir": (
        "artifacts/downstream/ec/embeddings/"
        "esm2_t33_650M/paper/esm2_t33_650M_base"
    ),
    "label_csv": "data/ec/split100.csv",
    "base_name": "esm2_base",
    "seed": 0,

    # Validation guards for 06B_strict (= general30).
    "expected_negative_mode": "general",
    "positive_max_identity": 30.0,
    "negative_min_identity": 40.0,
    "min_identity_margin": 10.0,
    "summary_tolerance": 1e-5,
    "require_summary_consistency": True,
    "require_label_consistency": True,

    # A compact table for choosing reactions to draw in panel d. This does not
    # affect panel e, which uses every row in output_all_triplets.
    "representative_failures_per_anchor": 20,
    "prefer_same_ec3_negative": True,
    "max_representatives_per_negative": 1,

    # Selected anchors must also have diverse Base-failing negatives. This is a
    # second guard after scan_hcft_case_anchors.py.
    "min_unique_failed_negatives": 5,
    "min_unique_failed_negative_ec4": 2,
    "max_single_failed_negative_fraction": 0.50,

    "output_all_triplets": (
        "artifacts/downstream/ec/hcft/case_selection/"
        "06B_strict/base_case_triplets_all.csv"
    ),
    "output_representative_failures": (
        "artifacts/downstream/ec/hcft/case_selection/"
        "06B_strict/base_case_representative_failures.csv"
    ),
    "output_validation": (
        "artifacts/downstream/ec/hcft/case_selection/"
        "06B_strict/base_case_anchor_validation.csv"
    ),
}
# =============================================================================

torch = None
functional = None

def require_torch() -> None:
    global torch, functional
    if torch is not None:
        return
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required. Run this script in the HCFT environment."
        ) from error
    torch = torch_module
    functional = functional_module

def as_path(value: str | Path, name: str, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"CONFIG[{name!r}] does not exist: {path}")
    return path

def validate_config() -> list[str]:
    anchors = list(dict.fromkeys(
        str(value).strip()
        for value in CONFIG["selected_anchors"]
        if str(value).strip()
    ))
    if not anchors:
        raise ValueError("CONFIG['selected_anchors'] cannot be empty")
    if int(CONFIG["representative_failures_per_anchor"]) < 1:
        raise ValueError(
            "CONFIG['representative_failures_per_anchor'] must be positive"
        )
    for key in (
        "max_representatives_per_negative",
        "min_unique_failed_negatives",
        "min_unique_failed_negative_ec4",
    ):
        if int(CONFIG[key]) < 1:
            raise ValueError(f"CONFIG[{key!r}] must be positive")
    fraction = float(CONFIG["max_single_failed_negative_fraction"])
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "CONFIG['max_single_failed_negative_fraction'] must be "
            "between 0 and 1"
        )
    for key in (
        "positive_max_identity",
        "negative_min_identity",
        "min_identity_margin",
    ):
        value = float(CONFIG[key])
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"CONFIG[{key!r}] must be between 0 and 100")
    return anchors

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

def ec_at_level(labels: list[str], level: int) -> set[str]:
    output: set[str] = set()
    for label in labels:
        parts = label.split(".")
        if (
            len(parts) >= level
            and all(part and part != "-" for part in parts[:level])
        ):
            output.add(".".join(parts[:level]))
    return output

def labels_as_text(labels: dict[str, list[str]], identifier: str, level: int) -> str:
    return ";".join(sorted(ec_at_level(labels.get(identifier, []), level)))

def load_embeddings(directory: Path) -> dict[str, object]:
    require_torch()
    files = sorted(directory.glob("*_all.pt"))
    if not files:
        raise FileNotFoundError(f"No merged *_all.pt files found in {directory}")
    embeddings: dict[str, object] = {}
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a dictionary")
        values = payload.get("embeddings", {})
        if not isinstance(values, dict):
            raise ValueError(f"{path} has no embeddings mapping")
        embeddings.update({
            str(identifier): vector.detach().float().cpu()
            for identifier, vector in values.items()
        })
    return embeddings

def load_exact_triplets(path: Path, anchors: list[str]) -> pd.DataFrame:
    triplets = pd.read_csv(
        path,
        dtype={"anchor": str, "positive": str, "negative": str},
    )
    required = {
        "seed",
        "anchor",
        "positive",
        "negative",
        "id_anchor_positive",
        "id_anchor_negative",
        "delta_identity",
        "negative_mode",
    }
    missing = required - set(triplets.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for column in (
        "seed",
        "id_anchor_positive",
        "id_anchor_negative",
        "delta_identity",
    ):
        triplets[column] = pd.to_numeric(triplets[column], errors="coerce")
    triplets = triplets.dropna(subset=list(required)).copy()
    triplets["seed"] = triplets["seed"].astype(int)
    triplets = triplets.loc[
        triplets["seed"].eq(int(CONFIG["seed"]))
        & triplets["anchor"].isin(anchors)
    ].copy()
    missing_anchors = sorted(set(anchors) - set(triplets["anchor"]))
    if missing_anchors:
        raise KeyError(f"No exact triplets found for anchors: {missing_anchors}")

    observed_modes = set(triplets["negative_mode"].astype(str).str.strip())
    expected_mode = str(CONFIG["expected_negative_mode"])
    if observed_modes != {expected_mode}:
        raise ValueError(
            f"Observed negative_mode={sorted(observed_modes)}, "
            f"expected only {expected_mode!r}"
        )
    tolerance = 1e-9
    invalid = (
        triplets["id_anchor_positive"].gt(
            float(CONFIG["positive_max_identity"]) + tolerance
        )
        | triplets["id_anchor_negative"].lt(
            float(CONFIG["negative_min_identity"]) - tolerance
        )
        | triplets["delta_identity"].lt(
            float(CONFIG["min_identity_margin"]) - tolerance
        )
    )
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} rows violate the configured strict-HCFT "
            "identity constraints"
        )
    if triplets.duplicated(["anchor", "positive", "negative"]).any():
        raise ValueError("Exact triplet file contains duplicate triplets")
    return triplets.reset_index(drop=True)

def score_base_triplets(
    triplets: pd.DataFrame,
    embeddings: dict[str, object],
) -> pd.DataFrame:
    require_torch()
    required_ids = (
        set(triplets["anchor"])
        | set(triplets["positive"])
        | set(triplets["negative"])
    )
    missing = sorted(required_ids - set(embeddings))
    if missing:
        raise KeyError(
            f"Base embeddings missing for {len(missing)} proteins, "
            f"e.g. {missing[:10]}"
        )

    anchor_matrix = functional.normalize(
        torch.stack([embeddings[value] for value in triplets["anchor"]]),
        dim=1,
    )
    positive_matrix = functional.normalize(
        torch.stack([embeddings[value] for value in triplets["positive"]]),
        dim=1,
    )
    negative_matrix = functional.normalize(
        torch.stack([embeddings[value] for value in triplets["negative"]]),
        dim=1,
    )
    scored = triplets.copy()
    scored["base_positive_similarity"] = (
        (anchor_matrix * positive_matrix).sum(dim=1).tolist()
    )
    scored["base_negative_similarity"] = (
        (anchor_matrix * negative_matrix).sum(dim=1).tolist()
    )
    scored["base_triplet_margin"] = (
        scored["base_positive_similarity"]
        - scored["base_negative_similarity"]
    )
    scored["base_correct"] = scored["base_triplet_margin"].gt(0)
    scored["base_failed"] = ~scored["base_correct"]
    return scored

def annotate_ec(scored: pd.DataFrame, labels: dict[str, list[str]]) -> pd.DataFrame:
    output = scored.copy()
    for role in ("anchor", "positive", "negative"):
        output[f"{role}_ec4"] = [
            labels_as_text(labels, identifier, 4)
            for identifier in output[role]
        ]
        output[f"{role}_ec3"] = [
            labels_as_text(labels, identifier, 3)
            for identifier in output[role]
        ]
    output["positive_shares_anchor_ec4"] = [
        bool(
            ec_at_level(labels.get(anchor, []), 4)
            & ec_at_level(labels.get(positive, []), 4)
        )
        for anchor, positive in zip(output["anchor"], output["positive"])
    ]
    output["negative_differs_from_anchor_ec4"] = [
        bool(
            ec_at_level(labels.get(anchor, []), 4)
            and ec_at_level(labels.get(negative, []), 4)
            and not (
                ec_at_level(labels.get(anchor, []), 4)
                & ec_at_level(labels.get(negative, []), 4)
            )
        )
        for anchor, negative in zip(output["anchor"], output["negative"])
    ]
    output["negative_shares_anchor_ec3"] = [
        bool(
            ec_at_level(labels.get(anchor, []), 3)
            & ec_at_level(labels.get(negative, []), 3)
        )
        for anchor, negative in zip(output["anchor"], output["negative"])
    ]
    return output

def validate_against_summary(
    scored: pd.DataFrame,
    summary_path: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path, dtype={"anchor": str})
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
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")
    official = summary.loc[
        summary["embedding"].astype(str).eq(str(CONFIG["base_name"]))
        & pd.to_numeric(summary["seed"], errors="coerce").eq(int(CONFIG["seed"]))
        & summary["anchor"].isin(scored["anchor"])
    ].copy()
    if official["anchor"].duplicated().any():
        raise ValueError("Official summary has duplicate Base rows")

    records: list[dict[str, object]] = []
    for anchor, group in scored.groupby("anchor", sort=False):
        failures = group.loc[group["base_failed"]]
        failed_negative_counts = failures["negative"].value_counts()
        failed_negative_ec4 = {
            ec4
            for value in failures["negative_ec4"]
            for ec4 in str(value).split(";")
            if ec4
        }
        records.append({
            "anchor": anchor,
            "recomputed_n_triplets": len(group),
            "recomputed_base_acc": group["base_correct"].mean(),
            "recomputed_base_margin": group["base_triplet_margin"].mean(),
            "n_base_failures": len(failures),
            "n_unique_positives": group["positive"].nunique(),
            "n_unique_negatives": group["negative"].nunique(),
            "n_unique_failed_negatives": failures["negative"].nunique(),
            "n_unique_failed_negative_ec4": len(failed_negative_ec4),
            "top_failed_negative": (
                str(failed_negative_counts.index[0])
                if not failed_negative_counts.empty
                else ""
            ),
            "top_failed_negative_count": (
                int(failed_negative_counts.iloc[0])
                if not failed_negative_counts.empty
                else 0
            ),
            "top_failed_negative_fraction": (
                float(failed_negative_counts.iloc[0] / len(failures))
                if len(failures)
                else 0.0
            ),
        })
    recomputed = pd.DataFrame(records)
    official = official[
        ["anchor", "n_triplets", "hcft_acc", "hcft_margin_mean"]
    ].rename(columns={
        "n_triplets": "official_n_triplets",
        "hcft_acc": "official_base_acc",
        "hcft_margin_mean": "official_base_margin",
    })
    validation = recomputed.merge(
        official,
        on="anchor",
        how="left",
        validate="one_to_one",
    )
    tolerance = float(CONFIG["summary_tolerance"])
    validation["n_triplets_match"] = validation[
        "recomputed_n_triplets"
    ].eq(validation["official_n_triplets"])
    validation["acc_abs_error"] = (
        validation["recomputed_base_acc"]
        - validation["official_base_acc"]
    ).abs()
    validation["margin_abs_error"] = (
        validation["recomputed_base_margin"]
        - validation["official_base_margin"]
    ).abs()
    validation["summary_consistent"] = (
        validation["n_triplets_match"]
        & validation["acc_abs_error"].le(tolerance)
        & validation["margin_abs_error"].le(tolerance)
    )
    validation["failure_diversity_passes"] = (
        validation["n_unique_failed_negatives"].ge(
            int(CONFIG["min_unique_failed_negatives"])
        )
        & validation["n_unique_failed_negative_ec4"].ge(
            int(CONFIG["min_unique_failed_negative_ec4"])
        )
        & validation["top_failed_negative_fraction"].le(
            float(CONFIG["max_single_failed_negative_fraction"])
        )
    )
    if bool(CONFIG["require_summary_consistency"]) and not validation[
        "summary_consistent"
    ].all():
        failed = validation.loc[
            ~validation["summary_consistent"],
            ["anchor", "n_triplets_match", "acc_abs_error", "margin_abs_error"],
        ]
        raise RuntimeError(
            "Recomputed Base values do not match the official summary:\n"
            + failed.to_string(index=False)
        )
    if not validation["failure_diversity_passes"].all():
        failed = validation.loc[
            ~validation["failure_diversity_passes"],
            [
                "anchor",
                "n_base_failures",
                "n_unique_failed_negatives",
                "n_unique_failed_negative_ec4",
                "top_failed_negative",
                "top_failed_negative_fraction",
            ],
        ]
        raise RuntimeError(
            "Selected anchor does not have sufficiently diverse Base-failing "
            "negatives. Choose another candidate or revise the explicit "
            "diversity thresholds:\n"
            + failed.to_string(index=False)
        )
    return validation

def choose_representative_failures(scored: pd.DataFrame) -> pd.DataFrame:
    failures = scored.loc[scored["base_failed"]].copy()
    if failures.empty:
        return failures
    failures["ec3_priority"] = (
        failures["negative_shares_anchor_ec3"].astype(int)
        if bool(CONFIG["prefer_same_ec3_negative"])
        else 0
    )
    failures = failures.sort_values(
        ["anchor", "ec3_priority", "base_triplet_margin", "delta_identity"],
        ascending=[True, False, True, False],
    )
    failures["rank_within_negative"] = (
        failures.groupby(["anchor", "negative"]).cumcount() + 1
    )
    failures = failures.loc[
        failures["rank_within_negative"].le(
            int(CONFIG["max_representatives_per_negative"])
        )
    ].copy()
    failures["representative_rank_within_anchor"] = (
        failures.groupby("anchor").cumcount() + 1
    )
    return failures.loc[
        failures["representative_rank_within_anchor"].le(
            int(CONFIG["representative_failures_per_anchor"])
        )
    ].drop(columns=["ec3_priority"])

def main() -> None:
    anchors = validate_config()
    triplet_path = as_path(CONFIG["triplets_csv"], "triplets_csv")
    summary_path = as_path(CONFIG["anchor_summary"], "anchor_summary")
    embedding_dir = as_path(CONFIG["base_embedding_dir"], "base_embedding_dir")
    label_path = as_path(CONFIG["label_csv"], "label_csv")

    triplets = load_exact_triplets(triplet_path, anchors)
    embeddings = load_embeddings(embedding_dir)
    labels = read_label_table(label_path)
    scored = annotate_ec(score_base_triplets(triplets, embeddings), labels)
    if bool(CONFIG["require_label_consistency"]):
        invalid_labels = ~(
            scored["positive_shares_anchor_ec4"]
            & scored["negative_differs_from_anchor_ec4"]
        )
        if invalid_labels.any():
            columns = [
                "anchor",
                "anchor_ec4",
                "positive",
                "positive_ec4",
                "negative",
                "negative_ec4",
            ]
            raise RuntimeError(
                f"{int(invalid_labels.sum())} triplets disagree with the "
                "expected EC4 positive/negative relation:\n"
                + scored.loc[invalid_labels, columns].head(20).to_string(
                    index=False
                )
            )
    validation = validate_against_summary(scored, summary_path)
    representatives = choose_representative_failures(scored)

    preferred_order = [
        "seed", "anchor", "anchor_ec4", "positive", "positive_ec4",
        "negative", "negative_ec4", "id_anchor_positive",
        "id_anchor_negative", "delta_identity", "negative_mode",
        "base_positive_similarity", "base_negative_similarity",
        "base_triplet_margin", "base_correct", "base_failed",
        "negative_shares_anchor_ec3", "positive_shares_anchor_ec4",
        "negative_differs_from_anchor_ec4",
    ]
    remaining = [column for column in scored if column not in preferred_order]
    scored = scored[preferred_order + remaining]

    output_all = as_path(
        CONFIG["output_all_triplets"],
        "output_all_triplets",
        must_exist=False,
    )
    output_representatives = as_path(
        CONFIG["output_representative_failures"],
        "output_representative_failures",
        must_exist=False,
    )
    output_validation = as_path(
        CONFIG["output_validation"],
        "output_validation",
        must_exist=False,
    )
    for output in (output_all, output_representatives, output_validation):
        output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_all, index=False)
    representatives.to_csv(output_representatives, index=False)
    validation.to_csv(output_validation, index=False)

    print(f"Anchors: {', '.join(anchors)}")
    print(f"Exact Base triplets: {len(scored)}")
    print(f"Base failures: {int(scored['base_failed'].sum())}")
    print(f"Representative failures: {len(representatives)}")
    print(f"wrote {output_all}")
    print(f"wrote {output_representatives}")
    print(f"wrote {output_validation}")
    print(validation.to_string(index=False))

if __name__ == "__main__":
    main()