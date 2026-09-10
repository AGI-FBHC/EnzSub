#!/usr/bin/env python3
"""
4_scaffold_subgroup_analysis.py — Scaffold Holdout Subgroup Analysis

纯后处理脚本，不修改任何现有训练/评估代码。

流程:
  1. 加载已保存的 GBDT 模型 + 带 embedding 的 test pkl
  2. 对每个 test sample 计算 scaffold 并标注 eval_group
  3. 重新预测并按 eval_group 分组计算指标
  4. 计算 holdout_drop / retention / delta_vs_baseline
  5. 输出 annotated CSV + summary tables

Usage:
    python 4_scaffold_subgroup_analysis.py \
        --results-dir results/esp_scaffold_holdout_YYYYMMDD \
        --step1-dir ./chemical_ood_prep_step1 \
        --step4-dir ./chemical_ood_prep_step4 \
        --output-dir results/esp_scaffold_holdout_YYYYMMDD/_scaffold_analysis
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

RDLogger.DisableLog("rdApp.*")

# ===========================================================================
# Scaffold computation (identical to Step 1)
# ===========================================================================

def compute_generic_scaffold(smiles: str) -> str:
    if not smiles or pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return ""
    try:
        scaff = GetScaffoldForMol(mol)
        generic = MakeScaffoldGeneric(scaff)
        smi = Chem.MolToSmiles(generic, canonical=True)
        return smi if smi else ""
    except Exception:
        return ""

# ===========================================================================
# Metrics (copied from 2_train_evaluate.py to ensure identical computation)
# ===========================================================================

from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score,
    f1_score, precision_score, recall_score,
)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    n_neg = n - n_pos

    out = {
        "n_samples": n, "n_positive": n_pos, "n_negative": n_neg,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if n_pos == 0 or n_neg == 0:
        out.update({"roc_auc": None, "mcc": None, "f1": None,
                    "precision": None, "recall": None, "skipped": True,
                    "skip_reason": "single_class"})
        return out

    out.update({
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "skipped": False,
    })
    return out

# ===========================================================================
# Feature construction (copied from 2_train_evaluate.py)
# ===========================================================================

def create_features(df: pd.DataFrame, mol_feature: str = "gnn") -> tuple:
    X_esm = np.stack(df["ESM_vector"].values).astype(np.float32)
    if mol_feature == "gnn":
        X_mol = np.stack(df["GNN_vector"].values).astype(np.float32)
    elif mol_feature == "ecfp":
        ecfp_strs = df["ECFP"].astype(str).values
        X_mol = np.stack([
            (np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48)
            for s in ecfp_strs
        ]).astype(np.float32)
    else:
        raise ValueError(f"Unknown mol_feature: {mol_feature}")
    X = np.concatenate([X_mol, X_esm], axis=1).astype(np.float32)
    y = df["Binding"].to_numpy().astype(np.float32)
    return X, y

# ===========================================================================
# Safe formatter for metrics that may be None
# ===========================================================================

def fmt(val, spec=".4f"):
    """Safe format: return 'N/A' for None/missing, formatted number otherwise."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        return f"{val:{spec}}"
    except (ValueError, TypeError):
        return str(val)

# ===========================================================================
# Experiment / model mapping
# ===========================================================================

# Which models belong to which experiment, and what's the baseline
EXPERIMENT_MODEL_MAP = {
    "Exp1_single": {
        "pairs": [
            {"baseline": "EnzSub",  "holdout": "exp1_cpt_sub", "backbone": "CPT"},
            {"baseline": "ESM_SUB", "holdout": "exp1_esm",     "backbone": "ESM-2"},
        ],
    },
    "Exp2_single": {
        "pairs": [
            {"baseline": "EnzSub",  "holdout": "exp2_cpt_sub", "backbone": "CPT"},
            {"baseline": "ESM_SUB", "holdout": "exp2_esm",     "backbone": "ESM-2"},
        ],
    },
}

# ===========================================================================
# Core analysis
# ===========================================================================

def load_scaffold_config(step1_dir: str, step4_dir: str) -> dict:
    """Load scaffold definitions from previous steps."""
    config = {}

    # OED scaffold set (from Step 1 pair table)
    pair_path = os.path.join(step1_dir, "pair_with_scaffold_table.csv")
    pair_df = pd.read_csv(pair_path, keep_default_na=False)
    config["oed_scaffold_set"] = set(
        pair_df[pair_df["generic_murcko_scaffold"] != ""]["generic_murcko_scaffold"].unique()
    )

    # Always-seen scaffolds (from Step 4)
    as_path = os.path.join(step4_dir, "always_seen_background_scaffolds.csv")
    as_df = pd.read_csv(as_path, keep_default_na=False)
    config["always_seen_set"] = set(as_df["generic_murcko_scaffold"].tolist())

    # Experiment holdout scaffolds (from Step 4)
    exp_path = os.path.join(step4_dir, "targeted_holdout_experiments.csv")
    exp_df = pd.read_csv(exp_path, keep_default_na=False)
    config["experiments"] = {}
    for _, row in exp_df.iterrows():
        config["experiments"][row["experiment_id"]] = {
            "holdout_scaffolds": set(row["holdout_scaffold_list"].split("|")),
        }

    print(f"  OED scaffolds: {len(config['oed_scaffold_set'])}")
    print(f"  Always-seen: {len(config['always_seen_set'])}")
    for eid, econf in config["experiments"].items():
        print(f"  {eid} holdout: {econf['holdout_scaffolds']}")

    return config

def annotate_eval_group(
    df: pd.DataFrame,
    holdout_set: set,
    always_seen_set: set,
    oed_scaffold_set: set,
) -> pd.DataFrame:
    """
    Annotate each test sample with scaffold info and eval_group.
    Works on ALL samples (positive + negative), not just positives.
    """
    df = df.copy()

    # Compute scaffold from SMILES
    if "generic_murcko_scaffold" not in df.columns:
        print("    Computing scaffolds from SMILES...")
        df["generic_murcko_scaffold"] = df["SMILES"].apply(compute_generic_scaffold)

    scaff = df["generic_murcko_scaffold"]

    df["is_empty_scaffold"] = (scaff == "") | scaff.isna()
    df["is_holdout_scaffold"] = scaff.isin(holdout_set)
    df["is_always_seen_background"] = scaff.isin(always_seen_set)
    df["is_oed_seen_scaffold"] = scaff.isin(oed_scaffold_set) | df["is_empty_scaffold"]
    df["is_novel_to_oed"] = ~df["is_oed_seen_scaffold"] & ~df["is_empty_scaffold"]

    # Assign eval_group (mutually exclusive, priority order)
    conditions = [
        df["is_holdout_scaffold"],
        df["is_always_seen_background"],
        df["is_novel_to_oed"],
        df["is_empty_scaffold"],
    ]
    choices = ["HOLDOUT", "ALWAYS_SEEN_BACKGROUND", "NOVEL_TO_OED", "EMPTY_ACYCLIC"]
    df["eval_group"] = np.select(conditions, choices, default="SEEN_NONHOLDOUT")

    return df

def predict_with_saved_gbdt(
    model_path: str, X: np.ndarray
) -> np.ndarray:
    """Load saved GBDT model and predict."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_path)
    dtest = xgb.DMatrix(X)
    return booster.predict(dtest)

def compute_subgroup_metrics(
    df_annotated: pd.DataFrame,
    y_prob: np.ndarray,
) -> dict:
    """Compute metrics per eval_group."""
    df = df_annotated.copy()
    df["_prob"] = y_prob
    y_true = df["Binding"].to_numpy().astype(np.float32)

    results = {"overall": compute_metrics(y_true, y_prob)}

    groups = ["HOLDOUT", "SEEN_NONHOLDOUT", "ALWAYS_SEEN_BACKGROUND",
              "NOVEL_TO_OED", "EMPTY_ACYCLIC"]

    for group in groups:
        sub = df[df["eval_group"] == group]
        if len(sub) == 0:
            results[group] = {"skipped": True, "n_samples": 0, "n_positive": 0, "n_negative": 0}
        else:
            results[group] = compute_metrics(
                sub["Binding"].to_numpy(), sub["_prob"].to_numpy()
            )

    return results

def compute_derived_metrics(
    holdout_metrics: dict,
    seen_metrics: dict,
    baseline_holdout_metrics: dict,
    primary_metric: str = "roc_auc",
) -> dict:
    """Compute holdout_drop, retention, delta_vs_baseline."""
    derived = {}

    ho_val = holdout_metrics.get(primary_metric)
    seen_val = seen_metrics.get(primary_metric)
    bl_ho_val = baseline_holdout_metrics.get(primary_metric)

    # holdout_drop = seen - holdout (positive means holdout is worse)
    if ho_val is not None and seen_val is not None:
        derived["holdout_drop"] = round(seen_val - ho_val, 6)
    else:
        derived["holdout_drop"] = None

    # holdout_retention = holdout / seen (1.0 = no drop)
    if ho_val is not None and seen_val is not None and seen_val > 0:
        derived["holdout_retention"] = round(ho_val / seen_val, 6)
    else:
        derived["holdout_retention"] = None

    # delta_vs_baseline = holdout_model_on_holdout - baseline_on_holdout
    if ho_val is not None and bl_ho_val is not None:
        derived["delta_vs_baseline_on_holdout"] = round(ho_val - bl_ho_val, 6)
    else:
        derived["delta_vs_baseline_on_holdout"] = None

    derived["primary_metric"] = primary_metric
    derived["holdout_value"] = ho_val
    derived["seen_value"] = seen_val
    derived["baseline_holdout_value"] = bl_ho_val

    return derived

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scaffold subgroup analysis (post-processing, zero code changes to pipeline)"
    )
    parser.add_argument("--results-dir", type=str, required=True,
                        help="ESP results directory (with _embeddings/ and model dirs)")
    parser.add_argument("--step1-dir", type=str, required=True,
                        help="chemical_ood_prep_step1 directory")
    parser.add_argument("--step4-dir", type=str, required=True,
                        help="chemical_ood_prep_step4 directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results-dir/_scaffold_analysis)")
    parser.add_argument("--mol-feature", type=str, default="gnn", choices=["gnn", "ecfp"])
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "_scaffold_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Scaffold Subgroup Analysis (Post-Processing)")
    print("=" * 70)

    # --- 1. Load scaffold config ---
    print("\n[1] Loading scaffold configuration...")
    scaff_config = load_scaffold_config(args.step1_dir, args.step4_dir)

    # --- 2. Find available models ---
    print("\n[2] Scanning available models...")
    embedding_dir = os.path.join(args.results_dir, "_embeddings")
    all_models = {}

    for model_name in os.listdir(embedding_dir):
        model_emb_dir = os.path.join(embedding_dir, model_name)
        if not os.path.isdir(model_emb_dir):
            continue

        # Find test pkl (with embeddings)
        test_pkls = [f for f in os.listdir(model_emb_dir)
                     if f.startswith("processed_test") and f.endswith(".pkl")]
        if not test_pkls:
            continue

        # Find saved GBDT model
        gbdt_path = os.path.join(args.results_dir, model_name, "final_gbdt.json")
        if not os.path.exists(gbdt_path):
            print(f"  ⚠ {model_name}: no GBDT model found, skipping")
            continue

        test_pkl_path = os.path.join(model_emb_dir, test_pkls[0])
        all_models[model_name] = {
            "test_pkl": test_pkl_path,
            "gbdt_path": gbdt_path,
        }
        print(f"  ✓ {model_name}")

    if not all_models:
        print("  ERROR: No models found with saved GBDT + test embeddings")
        sys.exit(1)

    # --- 3. Process each experiment ---
    all_experiment_results = {}

    for exp_id, exp_mapping in EXPERIMENT_MODEL_MAP.items():
        if exp_id not in scaff_config["experiments"]:
            print(f"\n  ⚠ {exp_id} not in scaffold config, skipping")
            continue

        holdout_set = scaff_config["experiments"][exp_id]["holdout_scaffolds"]
        always_seen_set = scaff_config["always_seen_set"]
        oed_set = scaff_config["oed_scaffold_set"]

        print(f"\n{'='*60}")
        print(f"  Experiment: {exp_id}")
        print(f"  Holdout scaffolds: {holdout_set}")
        print(f"{'='*60}")

        exp_results = {"pairs": []}

        # --- Pre-compute scaffold annotations ONCE per experiment ---
        # Find any model whose test pkl has SMILES and correct test-set size
        scaffold_annotations = None
        reference_n_rows = None
        scaffold_cols = [
            "SMILES", "generic_murcko_scaffold", "eval_group",
            "is_holdout_scaffold", "is_always_seen_background",
            "is_oed_seen_scaffold", "is_novel_to_oed", "is_empty_scaffold",
        ]

        for mn in set(
            m for p in exp_mapping["pairs"] for m in [p["baseline"], p["holdout"]]
        ):
            if mn not in all_models:
                continue
            df_probe = pd.read_pickle(all_models[mn]["test_pkl"])
            df_probe = df_probe.dropna(subset=["ESM_vector", "Binding"])
            if "SMILES" in df_probe.columns:
                print(f"\n  Computing scaffold annotations from {mn} "
                      f"(has SMILES, {len(df_probe)} rows)...")
                scaffold_annotations = annotate_eval_group(
                    df_probe, holdout_set, always_seen_set, oed_set
                )
                scaffold_annotations = scaffold_annotations[scaffold_cols].copy()
                reference_n_rows = len(scaffold_annotations)
                n_ho = (scaffold_annotations["eval_group"] == "HOLDOUT").sum()
                n_seen = (scaffold_annotations["eval_group"] == "SEEN_NONHOLDOUT").sum()
                print(f"    HOLDOUT: {n_ho} samples, SEEN_NONHOLDOUT: {n_seen} samples")
                print(f"    Reference row count: {reference_n_rows}")
                break

        if scaffold_annotations is None:
            print(f"  ⚠ No model has SMILES for {exp_id}, skipping experiment")
            continue

        for pair in exp_mapping["pairs"]:
            baseline_name = pair["baseline"]
            holdout_name = pair["holdout"]
            backbone = pair["backbone"]

            if baseline_name not in all_models:
                print(f"  ⚠ Baseline {baseline_name} not available, skipping pair")
                continue
            if holdout_name not in all_models:
                print(f"  ⚠ Holdout {holdout_name} not available, skipping pair")
                continue

            print(f"\n  --- {backbone} backbone ---")
            print(f"      Baseline: {baseline_name}")
            print(f"      Holdout:  {holdout_name}")

            pair_result = {
                "backbone": backbone,
                "baseline_model": baseline_name,
                "holdout_model": holdout_name,
            }

            # Process both models
            for role, model_name in [("baseline", baseline_name), ("holdout", holdout_name)]:
                info = all_models[model_name]

                # Load test data
                df_test = pd.read_pickle(info["test_pkl"])
                df_test = df_test.dropna(subset=["ESM_vector", "Binding"])
                n_rows = len(df_test)

                # --- Decide how to attach scaffold annotations ---
                if "SMILES" in df_test.columns:
                    # This pkl has SMILES — annotate directly
                    df_annotated = annotate_eval_group(df_test, holdout_set, always_seen_set, oed_set)
                elif n_rows == reference_n_rows:
                    # Same row count as reference — safe to reuse by row position
                    df_annotated = df_test.reset_index(drop=True).copy()
                    cached = scaffold_annotations.reset_index(drop=True)
                    for col in scaffold_cols:
                        df_annotated[col] = cached[col].values
                    print(f"      {model_name}: scaffold annotations reused by row position "
                          f"({n_rows} rows, matches reference)")
                else:
                    # Row count mismatch — this model's test pkl is from a different run
                    print(f"      ⚠ {model_name}: SKIPPED — no SMILES column and row count "
                          f"mismatch ({n_rows} vs reference {reference_n_rows}).")
                    print(f"        This likely means {model_name}'s embeddings were generated "
                          f"from a different data file (e.g., old run without SMILES).")
                    print(f"        Fix: re-run embedding generation for {model_name} using "
                          f"test.pkl")
                    continue

                # Build features & predict
                X, y = create_features(df_annotated, args.mol_feature)
                y_prob = predict_with_saved_gbdt(info["gbdt_path"], X)

                # Compute per-group metrics
                group_metrics = compute_subgroup_metrics(df_annotated, y_prob)

                pair_result[f"{role}_metrics"] = group_metrics

                # Print key results
                overall = group_metrics["overall"]
                holdout_m = group_metrics.get("HOLDOUT", {})
                seen_m = group_metrics.get("SEEN_NONHOLDOUT", {})

                print(f"      {model_name} ({role}):")
                print(f"        Overall:         AUC={fmt(overall.get('roc_auc'))}  "
                      f"MCC={fmt(overall.get('mcc'))}  "
                      f"(n={overall['n_samples']})")

                if not holdout_m.get("skipped", True):
                    print(f"        HOLDOUT:         AUC={holdout_m['roc_auc']:.4f}  "
                          f"MCC={holdout_m['mcc']:.4f}  "
                          f"(n={holdout_m['n_samples']}, pos={holdout_m['n_positive']}, neg={holdout_m['n_negative']})")
                else:
                    print(f"        HOLDOUT:         skipped (n={holdout_m.get('n_samples',0)})")

                if not seen_m.get("skipped", True):
                    print(f"        SEEN_NONHOLDOUT: AUC={seen_m['roc_auc']:.4f}  "
                          f"MCC={seen_m['mcc']:.4f}  "
                          f"(n={seen_m['n_samples']}, pos={seen_m['n_positive']}, neg={seen_m['n_negative']})")

                # Save annotated test CSV
                ann_path = os.path.join(
                    args.output_dir,
                    f"{exp_id}_{model_name}_test_annotated.csv"
                )
                df_out = df_annotated[["Uniprot ID" if "Uniprot ID" in df_annotated.columns else "enzyme_id",
                                        "SMILES", "Binding", "generic_murcko_scaffold",
                                        "eval_group", "is_holdout_scaffold",
                                        "is_always_seen_background", "is_oed_seen_scaffold",
                                        "is_novel_to_oed", "is_empty_scaffold"]].copy()
                df_out.columns = [c.replace(" ", "_") for c in df_out.columns]
                df_out["prediction"] = y_prob
                df_out["model_name"] = model_name
                df_out["experiment_id"] = exp_id
                df_out.to_csv(ann_path, index=False)

            # Compute derived metrics
            bl_metrics = pair_result.get("baseline_metrics", {})
            ho_metrics = pair_result.get("holdout_metrics", {})

            bl_holdout = bl_metrics.get("HOLDOUT", {})
            ho_holdout = ho_metrics.get("HOLDOUT", {})
            ho_seen = ho_metrics.get("SEEN_NONHOLDOUT", {})

            for metric_name in ["roc_auc", "mcc"]:
                derived = compute_derived_metrics(
                    holdout_metrics=ho_holdout,
                    seen_metrics=ho_seen,
                    baseline_holdout_metrics=bl_holdout,
                    primary_metric=metric_name,
                )
                pair_result[f"derived_{metric_name}"] = derived

            # Print derived summary
            d_auc = pair_result.get("derived_roc_auc", {})
            print(f"\n      [{backbone}] Derived (AUC):")
            print(f"        holdout_drop:     {d_auc.get('holdout_drop', 'N/A')}")
            print(f"        holdout_retention: {d_auc.get('holdout_retention', 'N/A')}")
            print(f"        delta_vs_baseline: {d_auc.get('delta_vs_baseline_on_holdout', 'N/A')}")

            exp_results["pairs"].append(pair_result)

        all_experiment_results[exp_id] = exp_results

    # --- 4. Build summary table ---
    print(f"\n{'='*60}")
    print(f"  Building summary tables")
    print(f"{'='*60}")

    summary_rows = []
    for exp_id, exp_res in all_experiment_results.items():
        for pair in exp_res["pairs"]:
            for role in ["baseline", "holdout"]:
                model_name = pair[f"{role}_model"]
                metrics = pair.get(f"{role}_metrics", {})

                for group in ["overall", "HOLDOUT", "SEEN_NONHOLDOUT",
                              "ALWAYS_SEEN_BACKGROUND", "NOVEL_TO_OED", "EMPTY_ACYCLIC"]:
                    m = metrics.get(group, {})
                    if m.get("skipped", True) and group != "overall":
                        continue
                    summary_rows.append({
                        "experiment": exp_id,
                        "backbone": pair["backbone"],
                        "role": role,
                        "model": model_name,
                        "eval_group": group,
                        "n_samples": m.get("n_samples", 0),
                        "n_positive": m.get("n_positive", 0),
                        "n_negative": m.get("n_negative", 0),
                        "roc_auc": m.get("roc_auc"),
                        "mcc": m.get("mcc"),
                        "f1": m.get("f1"),
                        "accuracy": m.get("accuracy"),
                        "precision": m.get("precision"),
                        "recall": m.get("recall"),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "scaffold_subgroup_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Written: scaffold_subgroup_metrics.csv ({len(summary_df)} rows)")

    # Derived metrics table
    derived_rows = []
    for exp_id, exp_res in all_experiment_results.items():
        for pair in exp_res["pairs"]:
            for metric_name in ["roc_auc", "mcc"]:
                d = pair.get(f"derived_{metric_name}", {})
                derived_rows.append({
                    "experiment": exp_id,
                    "backbone": pair["backbone"],
                    "baseline_model": pair["baseline_model"],
                    "holdout_model": pair["holdout_model"],
                    "metric": metric_name,
                    "holdout_value": d.get("holdout_value"),
                    "seen_value": d.get("seen_value"),
                    "baseline_holdout_value": d.get("baseline_holdout_value"),
                    "holdout_drop": d.get("holdout_drop"),
                    "holdout_retention": d.get("holdout_retention"),
                    "delta_vs_baseline": d.get("delta_vs_baseline_on_holdout"),
                })

    derived_df = pd.DataFrame(derived_rows)
    derived_path = os.path.join(args.output_dir, "scaffold_derived_metrics.csv")
    derived_df.to_csv(derived_path, index=False)
    print(f"  Written: scaffold_derived_metrics.csv ({len(derived_df)} rows)")

    # Full report JSON
    def to_native(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, set): return list(obj)
        return obj

    report = {
        "timestamp": datetime.now().isoformat(),
        "design_note": (
            "SUB-stage scaffold holdout ONLY. ESP train is unchanged. "
            "The question: when SUB training never sees certain scaffold substrates, "
            "does the final model still generalize on ESP test samples with those scaffolds?"
        ),
        "experiments": {
            eid: {
                "holdout_scaffolds": list(scaff_config["experiments"].get(eid, {}).get("holdout_scaffolds", [])),
                "pairs": eres["pairs"],
            }
            for eid, eres in all_experiment_results.items()
        },
    }

    report_path = os.path.join(args.output_dir, "scaffold_analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=to_native)
    print(f"  Written: scaffold_analysis_report.json")

    # --- 5. Print final summary ---
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")

    for exp_id, exp_res in all_experiment_results.items():
        print(f"\n  ━━━ {exp_id} ━━━")
        for pair in exp_res["pairs"]:
            backbone = pair["backbone"]
            bl = pair["baseline_model"]
            ho = pair["holdout_model"]
            d = pair.get("derived_roc_auc", {})

            print(f"  [{backbone}] {bl} (baseline) vs {ho} (holdout)")

            # Overall
            bl_overall = pair.get("baseline_metrics", {}).get("overall", {})
            ho_overall = pair.get("holdout_metrics", {}).get("overall", {})
            print(f"    Overall:    baseline AUC={fmt(bl_overall.get('roc_auc'))}  |  "
                  f"holdout AUC={fmt(ho_overall.get('roc_auc'))}")

            # HOLDOUT group
            bl_ho = pair.get("baseline_metrics", {}).get("HOLDOUT", {})
            ho_ho = pair.get("holdout_metrics", {}).get("HOLDOUT", {})
            bl_ok = not bl_ho.get("skipped", True)
            ho_ok = not ho_ho.get("skipped", True)
            if bl_ok and ho_ok:
                print(f"    HOLDOUT:    baseline AUC={fmt(bl_ho.get('roc_auc'))}  |  "
                      f"holdout AUC={fmt(ho_ho.get('roc_auc'))}  "
                      f"(n={ho_ho['n_samples']}, pos={ho_ho['n_positive']})")
            elif ho_ok:
                print(f"    HOLDOUT:    baseline=N/A  |  "
                      f"holdout AUC={fmt(ho_ho.get('roc_auc'))}  "
                      f"(n={ho_ho['n_samples']}, pos={ho_ho['n_positive']})")
            else:
                print(f"    HOLDOUT:    insufficient data")

            # SEEN group
            ho_seen = pair.get("holdout_metrics", {}).get("SEEN_NONHOLDOUT", {})
            if not ho_seen.get("skipped", True):
                print(f"    SEEN:       holdout model AUC={fmt(ho_seen.get('roc_auc'))}  "
                      f"(n={ho_seen['n_samples']}, pos={ho_seen['n_positive']})")

            # Derived
            print(f"    Drop(AUC):  {fmt(d.get('holdout_drop'))}")
            print(f"    Retention:  {fmt(d.get('holdout_retention'))}")
            print(f"    Δ vs base:  {fmt(d.get('delta_vs_baseline_on_holdout'))}")

            # Warning for small groups
            ho_ho = pair.get("holdout_metrics", {}).get("HOLDOUT", {})
            if ho_ho.get("n_positive", 0) < 20:
                print(f"    ⚠ WARNING: HOLDOUT group has only {ho_ho.get('n_positive',0)} "
                      f"positive samples — interpret with caution")

    print(f"\n  Output: {args.output_dir}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()