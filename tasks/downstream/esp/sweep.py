#!/usr/bin/env python3
"""
ESP checkpoint sweep pipeline (cleaned).

Pipeline:
  Step 0.5: ChemBERTa substrate embeddings (one-time)
  Step 1:   Protein/enzyme embeddings per model
  Step 2:   Downstream GBDT/MLP/kNN evaluation per model x seed x method

Step scripts reused:
  1b_generate_chemberta_embeddings.py
  1_generate_embeddings.py
  1_generate_plm_embeddings.py
  2_train_evaluate.py
  2_knn_evaluate.py
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")
CHECKPOINT_MODES = ("cpt", "base_sub", "cpt_sub")
VALID_SOURCES = ("enzsub", "external_plm")

# =============================================================================
# Config loading
# =============================================================================
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_seeds(cfg):
    seeds = cfg.get("eval", {}).get("seeds", [2025])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("eval.seeds must be a non-empty list")
    seeds = [int(s) for s in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds

def get_methods(cfg):
    methods = cfg.get("eval", {}).get("methods", ["gbdt"])
    if isinstance(methods, str):
        methods = [methods]
    if not isinstance(methods, list) or not methods:
        raise ValueError("eval.methods must be a non-empty list")
    methods = [str(m).lower() for m in methods]
    valid = {"gbdt", "mlp", "knn"}
    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise ValueError(f"Unknown eval.methods entries: {unknown}; valid={sorted(valid)}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate methods are not allowed: {methods}")
    return methods

def parse_model_entry(name, entry):
    if entry is None:
        entry = {}
    source = entry.get("source", "enzsub")
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source for {name}: {source}; valid={VALID_SOURCES}")
    mode = entry.get("mode", "base")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode for {name}: {mode}")
    if source == "external_plm" and mode != "base":
        raise ValueError(f"External PLM {name} currently supports mode=base only")

    model = {
        "name": name,
        "source": source,
        "encoder_type": entry.get("encoder_type", "esm1b"),
        "mode": mode,
        "checkpoint": entry.get("checkpoint"),
        "lora_rank": entry.get("lora_rank", 8),
        "lora_alpha": entry.get("lora_alpha", 16),
    }
    if source == "external_plm":
        for required in ("backend", "model_id"):
            if not entry.get(required):
                raise ValueError(f"External PLM {name} requires '{required}'")
        model["plm"] = {
            key: entry[key]
            for key in (
                "backend", "model_id", "local_path", "revision",
                "model_class", "tokenizer_class", "input_mode", "prefix",
                "pretrained_name", "dtype", "trust_remote_code",
                "validate_token_count", "batch_size", "token_budget",
                "max_batch_size", "max_len",
            )
            if key in entry
        }
    return model

def script_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

# =============================================================================
# Subprocess helper
# =============================================================================
def run_command(cmd, log_file, env=None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAILED ({proc.returncode}), see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-12:]:
                print(f"    {line.rstrip()}")
        return False
    print(f"  Done in {elapsed:.0f}s")
    return True

def pkl_has_column(path, column):
    import pandas as pd
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_pickle(path)
    except Exception:
        return False
    return column in df.columns and df[column].notna().sum() > 0

# =============================================================================
# Step 0.5: ChemBERTa
# =============================================================================
def run_chemberta(cfg, data_paths, main_dir, log_dir, args):
    chem_cfg = cfg.get("chemberta", {})
    if args.skip_chemberta or not chem_cfg.get("enabled", True):
        return data_paths

    input_pkls = [data_paths["train_pkl"], *data_paths["test_sets"].values()]
    if all(pkl_has_column(p, "ChemBERTa_vector") for p in input_pkls):
        print("\n[Step 0.5] ChemBERTa already present in all pkls")
        return data_paths

    output_dir = os.path.join(main_dir, "_chemberta")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        sys.executable, script_path("1b_generate_chemberta_embeddings.py"),
        "--input-pkls", *input_pkls,
        "--output-dir", output_dir,
        "--batch-size", str(chem_cfg.get("batch_size", 64)),
        "--max-length", str(chem_cfg.get("max_length", 128)),
        "--device", cfg.get("device", "cuda:0"),
    ]

    print("\n[Step 0.5] ChemBERTa embeddings")
    ok = run_command(cmd, os.path.join(log_dir, "chemberta.log"))
    if not ok:
        raise RuntimeError("ChemBERTa step failed")

    updated = {
        "train_pkl": os.path.join(output_dir, os.path.basename(data_paths["train_pkl"])),
        "test_sets": {},
    }
    for test_name, test_path in data_paths["test_sets"].items():
        updated["test_sets"][test_name] = os.path.join(output_dir, os.path.basename(test_path))
    return updated

# =============================================================================
# Step 1: protein embeddings
# =============================================================================
def check_embedding_outputs(model_dir, input_pkls):
    for pkl in input_pkls:
        output_path = os.path.join(model_dir, os.path.basename(pkl))
        if not pkl_has_column(output_path, "enzyme_vector"):
            return False
    return True

def run_embedding(model, cfg, data_paths, main_dir, log_dir, args):
    model_dir = os.path.join(main_dir, "_embeddings", model["name"])
    input_pkls = [data_paths["train_pkl"], *data_paths["test_sets"].values()]

    if check_embedding_outputs(model_dir, input_pkls):
        print(f"  {model['name']}: embeddings exist, skipping")
        return model_dir

    if model["source"] == "enzsub" and model["mode"] in CHECKPOINT_MODES and not model["checkpoint"]:
        raise ValueError(f"{model['name']} mode={model['mode']} requires checkpoint")
    if model["source"] == "enzsub" and model["checkpoint"] and not os.path.exists(model["checkpoint"]):
        raise FileNotFoundError(f"Checkpoint not found: {model['checkpoint']}")

    if model["source"] == "external_plm":
        plm = model["plm"]
        cmd = [
            sys.executable, script_path("1_generate_plm_embeddings.py"),
            "--model-name", model["name"],
            "--backend", plm["backend"],
            "--model-id", plm["model_id"],
            "--input-pkls", *input_pkls,
            "--output-dir", model_dir,
            "--batch-size", str(plm.get("batch_size", cfg.get("batch_size", 4))),
            "--token-budget", str(plm.get("token_budget", cfg.get("token_budget", 0))),
            "--max-batch-size", str(plm.get("max_batch_size", cfg.get("max_batch_size", 16))),
            "--max-len", str(plm.get("max_len", cfg.get("max_len", 1022))),
            "--dtype", str(plm.get("dtype", "auto")),
            "--device", cfg.get("device", "cuda:0"),
        ]
        option_flags = {
            "local_path": "--local-path",
            "revision": "--revision",
            "model_class": "--model-class",
            "tokenizer_class": "--tokenizer-class",
            "input_mode": "--input-mode",
            "prefix": "--prefix",
            "pretrained_name": "--pretrained-name",
        }
        for key, flag in option_flags.items():
            if key in plm and plm[key] is not None:
                cmd += [flag, str(plm[key])]
        if plm.get("trust_remote_code", False):
            cmd.append("--trust-remote-code")
        if not plm.get("validate_token_count", True):
            cmd.append("--no-validate-token-count")
    else:
        cmd = [
            sys.executable, script_path("1_generate_embeddings.py"),
            "--encoder-type", model["encoder_type"],
            "--model-mode", model["mode"],
            "--input-pkls", *input_pkls,
            "--output-dir", model_dir,
            "--batch-size", str(cfg.get("batch_size", 16)),
            "--max-len", str(cfg.get("max_len", 1022)),
            "--device", cfg.get("device", "cuda:0"),
        ]
        if model["checkpoint"]:
            cmd += ["--checkpoint", model["checkpoint"]]
        if model["mode"] in ("base_sub", "cpt_sub"):
            cmd += ["--lora-rank", str(model["lora_rank"])]
            cmd += ["--lora-alpha", str(model["lora_alpha"])]

    log_file = os.path.join(log_dir, f"emb_{model['name']}.log")
    print(f"  {model['name']}: generating enzyme embeddings ({model['source']})")
    ok = run_command(cmd, log_file)
    if not ok:
        raise RuntimeError(f"Embedding failed for {model['name']}")
    return model_dir

def embedding_file(model_dir, pkl_path):
    return os.path.join(model_dir, os.path.basename(pkl_path))

# =============================================================================
# Step 2: downstream evaluation
# =============================================================================
def _flatten_kv(prefix, mapping):
    out = []
    for k, v in mapping.items():
        # 保留末尾的下划线（用于 lambda_ 这类避开 Python 保留字的参数名）
        if k.endswith('_'):
            flag = f"--{k[:-1].replace('_', '-')}_"
        else:
            flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, (list, tuple)):
            out.append(flag)
            out.extend(str(x) for x in v)
        else:
            out += [flag, str(v)]
    return out

def run_train_eval(model, model_dir, cfg, data_paths, seed, method, main_dir, log_dir):
    """Dispatch to 2_train_evaluate.py with method-specific params from yaml."""
    eval_cfg = cfg.get("eval", {})
    method_cfg = eval_cfg.get(method, {}) or {}

    output_dir = os.path.join(main_dir, method, model["name"], f"seed_{seed}")
    metrics_file = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_file):
        print(f"    {method} seed={seed}: existing metrics, skipping")
        return output_dir

    train_file = embedding_file(model_dir, data_paths["train_pkl"])
    test_files = [embedding_file(model_dir, p) for p in data_paths["test_sets"].values()]
    test_names = list(data_paths["test_sets"].keys())

    cmd = [
        sys.executable, script_path("2_train_evaluate.py"),
        "--train-file", train_file,
        "--test-files", *test_files,
        "--test-names", *test_names,
        "--output-dir", output_dir,
        "--mol-feature", eval_cfg.get("mol_feature", "gnn"),
        "--model-type", method,
        "--cv-splits", str(eval_cfg.get("cv_splits", 5)),
        "--seed", str(seed),
    ]
    if not eval_cfg.get("cv", True):
        cmd.append("--skip-cv")

    # Method-specific params from yaml (eval.gbdt.* or eval.mlp.*)
    cmd += _flatten_kv(method, method_cfg)

    print(f"    {method} seed={seed}: train/evaluate")
    ok = run_command(cmd, os.path.join(log_dir, f"eval_{method}_{model['name']}_seed{seed}.log"))
    if not ok:
        raise RuntimeError(f"Evaluation failed for {model['name']} method={method} seed={seed}")
    return output_dir

def run_knn_eval(model, model_dir, cfg, data_paths, main_dir, log_dir):
    output_dir = os.path.join(main_dir, "knn", model["name"])
    metrics_file = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_file):
        print(f"    k-NN: existing metrics, skipping")
        return output_dir

    knn_cfg = cfg.get("knn", {})
    train_file = embedding_file(model_dir, data_paths["train_pkl"])
    if len(data_paths["test_sets"]) != 1:
        raise ValueError("2_knn_evaluate.py currently supports one test set per run")
    test_name, test_source = next(iter(data_paths["test_sets"].items()))
    test_file = embedding_file(model_dir, test_source)

    cmd = [
        sys.executable, script_path("2_knn_evaluate.py"),
        "--train-file", train_file,
        "--test-file", test_file,
        "--test-name", test_name,
        "--output-dir", output_dir,
        "--k-values", *[str(k) for k in knn_cfg.get("k_values", [1, 4, 8, 16, 32, 64])],
        "--batch-size", str(knn_cfg.get("batch_size", 256)),
    ]
    print("    k-NN: train/evaluate")
    ok = run_command(cmd, os.path.join(log_dir, f"knn_{model['name']}.log"))
    if not ok:
        raise RuntimeError(f"k-NN evaluation failed for {model['name']}")
    return output_dir

# =============================================================================
# Result aggregation -> CSV
# =============================================================================
def _metric_value(item, key):
    value = item.get(key)
    return None if value is None else float(value)

def _mean_std(values):
    import numpy as np
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())

def collect_train_results(models, seeds, methods, main_dir):
    metric_keys = ["accuracy", "roc_auc", "mcc", "f1", "precision", "recall"]
    per_seed_rows = []

    for model in models:
        for method in [m for m in methods if m in ("gbdt", "mlp")]:
            for seed in seeds:
                metrics_file = os.path.join(main_dir, method, model["name"], f"seed_{seed}", "metrics.json")
                if not os.path.exists(metrics_file):
                    continue
                with open(metrics_file) as f:
                    metrics = json.load(f)

                row = {
                    "model": model["name"],
                    "mode": model["mode"],
                    "method": method,
                    "seed": seed,
                }
                cv = metrics.get("cv", {}).get("aggregate", {})
                for key in metric_keys:
                    if key in cv:
                        row[f"cv_{key}_mean"] = cv[key].get("mean")
                        row[f"cv_{key}_std"] = cv[key].get("std")

                external = metrics.get("external_test_results", metrics.get("test_results", {}))
                for test_name, test_data in external.items():
                    overall = test_data.get("overall", {})
                    for key in metric_keys:
                        row[f"{test_name}_{key}"] = _metric_value(overall, key)
                    row[f"{test_name}_n_samples"] = overall.get("n_samples")
                per_seed_rows.append(row)

    return per_seed_rows

def collect_cv_fold_results(models, seeds, methods, main_dir):
    """Collect per-fold CV metrics from each seed's metrics.json.

    2_train_evaluate.py already stores fold-level CV details under:
        metrics["cv"]["folds"]

    This function only exposes those existing fold results as a separate CSV.
    It does not change training, validation, final fitting, or external testing.
    """
    metric_keys = ["accuracy", "roc_auc", "mcc", "f1", "precision", "recall"]
    rows = []

    for model in models:
        for method in [m for m in methods if m in ("gbdt", "mlp")]:
            for seed in seeds:
                metrics_file = os.path.join(main_dir, method, model["name"], f"seed_{seed}", "metrics.json")
                if not os.path.exists(metrics_file):
                    continue

                with open(metrics_file) as f:
                    metrics = json.load(f)

                folds = metrics.get("cv", {}).get("folds", [])
                if not folds:
                    continue

                for fold in folds:
                    m = fold.get("metrics", {})
                    row = {
                        "model": model["name"],
                        "mode": model["mode"],
                        "method": method,
                        "seed": seed,
                        "fold": fold.get("fold"),
                        "n_train": fold.get("n_train"),
                        "n_val": fold.get("n_val"),
                        # MLP has best_epoch; GBDT has best_iteration.
                        # The non-applicable column is left empty in the CSV.
                        "best_epoch": fold.get("best_epoch"),
                        "best_iteration": fold.get("best_iteration"),
                        # Only MLP fold summaries contain pos_weight.
                        "pos_weight": fold.get("pos_weight"),
                        "n_samples": m.get("n_samples"),
                        "n_positive": m.get("n_positive"),
                        "n_negative": m.get("n_negative"),
                    }

                    for key in metric_keys:
                        row[key] = _metric_value(m, key)

                    rows.append(row)

    return rows

def collect_knn_results(models, main_dir, data_paths):
    metric_keys = ["accuracy", "roc_auc", "mcc", "f1", "precision", "recall"]
    rows = []
    test_name = next(iter(data_paths["test_sets"].keys()), "test")

    for model in models:
        metrics_file = os.path.join(main_dir, "knn", model["name"], "metrics.json")
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)

        for k, result in metrics.get("results_by_k", {}).items():
            overall = result.get("overall", {})
            row = {
                "model": model["name"],
                "mode": model["mode"],
                "method": "knn",
                "seed": None,
                "k": int(k),
            }
            for key in metric_keys:
                row[f"{test_name}_{key}"] = _metric_value(overall, key)
            row[f"{test_name}_n_samples"] = overall.get("n_samples")
            rows.append(row)
    return rows

def aggregate_rows(per_seed_rows):
    metric_cols = [
        key for row in per_seed_rows for key in row
        if key not in ("model", "mode", "method", "seed", "k") and not key.endswith("_n_samples")
    ]
    metric_cols = sorted(set(metric_cols))

    grouped = {}
    for row in per_seed_rows:
        grouped.setdefault(
            (row["model"], row["mode"], row.get("method", "gbdt"), row.get("k")),
            [],
        ).append(row)

    agg_rows = []
    for (model, mode, method, k), rows in grouped.items():
        seeds = sorted({r.get("seed") for r in rows if r.get("seed") is not None})
        agg = {
            "model": model,
            "mode": mode,
            "method": method,
            "n_runs": len(rows),
            "n_seeds": len(seeds),
        }
        if k is not None:
            agg["k"] = k
        for col in metric_cols:
            mean, std = _mean_std([r.get(col) for r in rows])
            agg[f"{col}_mean"] = mean
            agg[f"{col}_std"] = std
        agg_rows.append(agg)
    return agg_rows

def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ESP checkpoint sweep")
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-chemberta", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)
    seeds = get_seeds(cfg)
    methods = get_methods(cfg)
    models = [parse_model_entry(name, entry) for name, entry in cfg["models"].items()]

    exp_cfg = cfg.get("experiment", {})
    exp_name = args.experiment_name or exp_cfg.get("name", "esp_sweep")
    main_dir = os.path.join(exp_cfg.get("results_root", "results"), exp_name)
    log_dir = os.path.join(main_dir, "_logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)

    data_paths = {
        "train_pkl": cfg["data"]["train_pkl"],
        "test_sets": cfg["data"]["test_sets"],
    }

    print("=" * 70)
    print(f"ESP Sweep | {datetime.now()}")
    print(f"Config:  {config_path}")
    print(f"Output:  {main_dir}")
    print(f"Methods: {methods}")
    print(f"Seeds:   {seeds}")
    print("=" * 70)
    for model in models:
        ckpt = os.path.basename(model["checkpoint"]) if model["checkpoint"] else "-"
        encoder = (
            model.get("plm", {}).get("model_id", "external_plm")
            if model["source"] == "external_plm"
            else model["encoder_type"]
        )
        print(f"  {model['name']:28s} {encoder:36s} {model['mode']:8s} ckpt={ckpt}")

    if args.dry_run:
        print("\n[DRY RUN] Done.")
        return

    # Step 0.5: ChemBERTa (one-time, shared across models)
    data_paths = run_chemberta(cfg, data_paths, main_dir, log_dir, args)

    # Step 1: per-model enzyme embeddings
    model_embedding_dirs = {}
    if not args.skip_embedding:
        print("\n[Step 1] Enzyme embeddings")
        for model in models:
            model_embedding_dirs[model["name"]] = run_embedding(
                model, cfg, data_paths, main_dir, log_dir, args
            )
    else:
        for model in models:
            model_embedding_dirs[model["name"]] = os.path.join(main_dir, "_embeddings", model["name"])

    # Step 2: downstream evaluation
    if not args.skip_eval:
        print("\n[Step 2] Downstream evaluation")
        for model in models:
            model_dir = model_embedding_dirs[model["name"]]
            for method in [m for m in methods if m in ("gbdt", "mlp")]:
                for seed in seeds:
                    run_train_eval(model, model_dir, cfg, data_paths, seed, method, main_dir, log_dir)
            if "knn" in methods:
                run_knn_eval(model, model_dir, cfg, data_paths, main_dir, log_dir)

    # Aggregate -> CSV (no markdown table)
    per_seed_rows = collect_train_results(models, seeds, methods, main_dir)
    cv_fold_rows = collect_cv_fold_results(models, seeds, methods, main_dir)

    if "knn" in methods:
        per_seed_rows.extend(collect_knn_results(models, main_dir, data_paths))

    agg_rows = aggregate_rows(per_seed_rows)

    comparison_dir = os.path.join(main_dir, "_comparison")
    summary_csv = os.path.join(comparison_dir, "sweep_summary.csv")
    per_seed_csv = os.path.join(comparison_dir, "sweep_summary_per_seed.csv")
    cv_folds_csv = os.path.join(comparison_dir, "sweep_cv_folds.csv")

    write_csv(per_seed_csv, per_seed_rows)
    write_csv(summary_csv, agg_rows)
    write_csv(cv_folds_csv, cv_fold_rows)

    print("\n" + "=" * 70)
    print(f"Done: {datetime.now()}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Per-seed CSV: {per_seed_csv}")
    if cv_fold_rows:
        print(f"CV folds CSV: {cv_folds_csv}")
    else:
        print("CV folds CSV: not generated (no CV fold rows found)")
    print(f"Logs: {log_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
