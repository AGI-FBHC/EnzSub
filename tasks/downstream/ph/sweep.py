#!/usr/bin/env python3
"""Run the optimum-pH embedding and regression workflow."""

import argparse
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
HF_POOLING = "last_hidden_state_masked_mean_no_special_tokens"
ESM3_POOLING = "esm3_final_layernorm_residue_mean_no_special_tokens"
MODE_DESC = {"base": "pretrained", "cpt": "CPT",
             "base_sub": "base+SUB", "cpt_sub": "CPT+SUB"}

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def parse_model_entry(name: str, entry: dict, default_encoder_type: str) -> dict:
    if entry is None:
        entry = {}
    source = entry.get("source", "enzsub")
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Invalid source '{source}' for '{name}'; expected one of {VALID_SOURCES}"
        )
    mode = entry.get("mode", "base")
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}' for '{name}'")

    if source == "external_plm":
        if mode != "base":
            raise ValueError(f"External PLM '{name}' only supports mode=base")
        if not entry.get("backend") or not entry.get("model_id"):
            raise ValueError(
                f"External PLM '{name}' requires both backend and model_id"
            )
        return {
            "name": name,
            "source": source,
            "encoder_type": entry["model_id"],
            "mode": mode,
            "checkpoint": None,
            "lora_rank": None,
            "lora_alpha": None,
            "backend": entry["backend"],
        }

    encoder_type = entry.get("encoder_type", default_encoder_type)
    if not encoder_type:
        raise ValueError(
            f"Model '{name}' has no encoder_type and no global encoder_type is set in config"
        )
    return {
        "name": name,
        "source": source,
        "encoder_type": encoder_type,
        "mode": mode,
        "checkpoint": entry.get("checkpoint"),
        "lora_rank": entry.get("lora_rank", 8),
        "lora_alpha": entry.get("lora_alpha", 16),
    }

def get_eval_seeds(cfg: dict) -> list:
    if "eval" in cfg and "seeds" in cfg["eval"]:
        seeds = cfg["eval"]["seeds"]
    elif "xgboost" in cfg and "seeds" in cfg["xgboost"]:
        seeds = cfg["xgboost"]["seeds"]
    elif "xgboost" in cfg and "seed" in cfg["xgboost"]:
        seeds = [cfg["xgboost"]["seed"]]
    elif "eval" in cfg and "seed" in cfg["eval"]:
        seeds = [cfg["eval"]["seed"]]
    else:
        seeds = [2025]

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list, e.g. eval.seeds: [2025, 2026, 2027]")

    seeds = [int(s) for s in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    return seeds

class StatusTracker:
    def __init__(self, log_dir: str):
        self.path = os.path.join(log_dir, "status.json")
        self.status = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.status = json.load(f)

    def get(self, name: str) -> str:
        return self.status.get(name, "pending")

    def set(self, name: str, state: str):
        self.status[name] = state
        with open(self.path, "w") as f:
            json.dump(self.status, f, indent=2)

    def reset(self):
        self.status = {}
        if os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

def _get_csv_paths(cfg: dict) -> list:
    """返回 train.csv, val.csv, test.csv 的路径列表"""
    processed_dir = cfg["data"]["processed_dir"]
    splits = cfg["data"].get("splits", ["train", "val", "test"])
    paths = []
    for s in splits:
        p = os.path.join(processed_dir, f"{s}.csv")
        if os.path.exists(p):
            paths.append(p)
    return paths

def validate_external_artifacts(model_info: dict, cfg: dict) -> tuple:
    """Reject stale, partial, or semantically incompatible external embeddings."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is unavailable for artifact validation"

    emb_dir = os.path.join(cfg["emb_root"], model_info["name"])
    expected_pooling = (
        ESM3_POOLING if model_info["backend"] == "esm3" else HF_POOLING
    )
    expected_files = [
        f"{split}_all.pt"
        for split in cfg["data"].get("splits", ["train", "val", "test"])
    ]
    for filename in expected_files:
        path = os.path.join(emb_dir, filename)
        if not os.path.exists(path):
            return False, f"missing {filename}"
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            return False, f"cannot load {filename}: {exc}"
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, dict):
            return False, f"{filename} has no embeddings mapping"
        checks = {
            "model_id": model_info["encoder_type"],
            "backend": model_info["backend"],
            "pooling": expected_pooling,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                return False, (
                    f"{filename} has {key}={payload.get(key)!r}, expected {expected!r}"
                )
        completed = payload.get("completed_records")
        input_rows = payload.get("input_rows")
        duplicate_rows = payload.get("duplicate_rows", 0)
        if completed != len(embeddings):
            return False, f"{filename} completed_records does not match stored embeddings"
        if input_rows is None or completed != input_rows - duplicate_rows:
            return False, f"{filename} is partial; regenerate without --limit"
    return True, "ok"

# ======== Phase 0 ========

def run_parse_fasta(cfg: dict, log_dir: str) -> bool:
    data_cfg = cfg["data"]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "parse_fasta.py")
    splits = data_cfg.get("splits", ["train", "val", "test"])

    cmd = [
        sys.executable, script,
        "--input-dir", data_cfg["raw_dir"],
        "--output-dir", data_cfg["processed_dir"],
        "--ph-min", str(data_cfg.get("ph_min", 2.0)),
        "--ph-max", str(data_cfg.get("ph_max", 12.0)),
        "--max-seq-length", str(data_cfg.get("max_seq_length", 4000)),
        "--splits", *splits,
    ]

    print(f"  CMD: {' '.join(cmd)}")
    log_file = os.path.join(log_dir, "parse_fasta.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"  ✗ FAILED, see {log_file}")
        return False
    print(f"  ✓ Done in {time.time() - t0:.0f}s")
    return True

# ======== Phase 1 ========

def run_embedding(
    model_info: dict,
    cfg: dict,
    config_path: str,
    log_dir: str,
    overwrite_external: bool = False,
) -> bool:
    name = model_info["name"]
    encoder_type = model_info["encoder_type"]
    emb_dir = os.path.join(cfg["emb_root"], name)
    env = os.environ.copy()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_info["source"] == "external_plm":
        script = os.path.join(script_dir, "generate_plm_embeddings.py")
        cmd = [
            sys.executable, script,
            "--config", config_path,
            "--models", name,
            "--device", cfg["device"],
        ]
        if overwrite_external:
            cmd.append("--overwrite")
        print(
            f"  CMD: [external_plm/{model_info['backend']}] "
            f"{encoder_type}"
        )
    else:
        script = os.path.join(script_dir, "generate_embeddings.py")
        csv_files = _get_csv_paths(cfg)
        if not csv_files:
            print("  ✗ No CSV files found")
            return False
        cmd = [
            sys.executable, script,
            "--encoder-type", encoder_type,
            "--model-mode", model_info["mode"],
            "--csv-files", *csv_files,
            "--output-dir", emb_dir,
            "--batch-size", str(cfg.get("batch_size", 16)),
            "--max-len", str(cfg.get("max_len", 1022)),
            "--max-tokens-per-batch", str(cfg.get("max_tokens_per_batch", 8000)),
            "--device", cfg["device"],
        ]
        if model_info["checkpoint"]:
            cmd += ["--checkpoint", model_info["checkpoint"]]
        if model_info["mode"] in ("base_sub", "cpt_sub"):
            cmd += ["--lora-rank", str(model_info["lora_rank"])]
            cmd += ["--lora-alpha", str(model_info["lora_alpha"])]
        print(f"  CMD: [{encoder_type}/{model_info['mode']}] {' '.join(cmd[:8])} ...")

        sub_parent = cfg.get("sub_parent_dir", "")
        if sub_parent:
            env["PYTHONPATH"] = sub_parent + os.pathsep + env.get("PYTHONPATH", "")

    log_file = os.path.join(log_dir, f"emb_{name}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

    if proc.returncode != 0:
        print(f"  ✗ FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-10:]:
                print(f"    {line.rstrip()}")
        return False

    expected = [f"{split}_all.pt" for split in cfg["data"].get(
        "splits", ["train", "val", "test"]
    )]
    missing = [filename for filename in expected
               if not os.path.exists(os.path.join(emb_dir, filename))]
    if missing:
        print(f"  ✗ Embedding command finished but files are missing: {missing}")
        return False
    print(f"  ✓ {len(expected)} files, {time.time() - t0:.0f}s")
    return True

# ======== Phase 2a: XGBoost ========

def _xgb_results_have_required_cv(results_csv: str, cv_folds: int) -> bool:
    """
    判断已有 XGBoost results.csv 是否满足当前 CV 配置。

    cv_folds<=1 时，原有 results.csv 即可复用；
    cv_folds>1 时，必须包含 train_cv_mae_mean，否则说明是旧结果，需要补跑。
    """
    if not os.path.exists(results_csv):
        return False
    if not cv_folds or cv_folds <= 1:
        return True
    try:
        import pandas as pd
        df = pd.read_csv(results_csv, nrows=1)
        return "train_cv_mae_mean" in df.columns
    except Exception:
        return False

def _expected_xgb_params(xgb_cfg: dict, seed: int) -> dict:
    params = {
        "n_estimators": int(xgb_cfg.get("n_estimators", 500)),
        "max_depth": int(xgb_cfg.get("max_depth", 6)),
        "learning_rate": float(xgb_cfg.get("learning_rate", 0.05)),
        "subsample": float(xgb_cfg.get("subsample", 0.8)),
        "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
        "min_child_weight": float(xgb_cfg.get("min_child_weight", 5.0)),
        "gamma": float(xgb_cfg.get("gamma", 0.1)),
        "reg_alpha": float(xgb_cfg.get("reg_alpha", 0.05)),
        "reg_lambda": float(xgb_cfg.get("reg_lambda", 5.0)),
        "random_state": int(seed),
        "n_jobs": int(xgb_cfg.get("n_jobs", 16)),
        "colsample_bylevel": float(xgb_cfg.get("colsample_bylevel", 0.8)),
        "colsample_bynode": float(xgb_cfg.get("colsample_bynode", 0.8)),
        "tree_method": str(xgb_cfg.get("tree_method", "hist")),
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }
    early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 50))
    if early_stopping_rounds > 0:
        params["early_stopping_rounds"] = early_stopping_rounds
    return params

def _xgb_results_match_current_config(
    output_dir: str,
    results_csv: str,
    xgb_cfg: dict,
    seed: int,
    cv_folds: int,
) -> bool:
    if not _xgb_results_have_required_cv(results_csv, cv_folds):
        return False
    config_json = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_json):
        return False
    try:
        with open(config_json) as f:
            saved = json.load(f)
        return (
            saved.get("xgb_params") == _expected_xgb_params(xgb_cfg, seed)
            and bool(saved.get("use_scaler")) == bool(xgb_cfg.get("use_scaler", True))
            and int(saved.get("seed")) == int(seed)
            and int(saved.get("cv_folds", 0) or 0) == int(cv_folds)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

def run_xgboost(emb_names: list, cfg: dict, log_dir: str, seed: int) -> str:
    script = os.path.join(os.path.dirname(__file__), "xgboost_eval.py")
    xgb_cfg = cfg["xgboost"]
    cv_folds = int(xgb_cfg.get("cv_folds", 0) or 0)
    processed_dir = cfg["data"]["processed_dir"]
    output_base = xgb_cfg.get("output_dir", "experiments/xgboost")
    run_name = f"seed_{seed}"
    output_dir = os.path.join(output_base, run_name)

    emb_args = [f"{n}:{os.path.join(cfg['emb_root'], n)}" for n in emb_names]

    results_csv = os.path.join(output_dir, "results.csv")
    if os.path.exists(results_csv):
        if _xgb_results_match_current_config(
            output_dir, results_csv, xgb_cfg, seed, cv_folds
        ):
            print(f"  seed={seed}: matching existing results found, skipping -> {output_dir}")
            return output_dir
        print(f"  seed={seed}: existing results do not match current XGBoost config, rerunning -> {output_dir}")

    cmd = [
        sys.executable, script,
        "--embedding-dirs", *emb_args,
        "--train-csv", os.path.join(processed_dir, "train.csv"),
        "--val-csv", os.path.join(processed_dir, "val.csv"),
        "--test-csv", os.path.join(processed_dir, "test.csv"),
        "--n-estimators", str(xgb_cfg.get("n_estimators", 500)),
        "--max-depth", str(xgb_cfg.get("max_depth", 6)),
        "--learning-rate", str(xgb_cfg.get("learning_rate", 0.05)),
        "--subsample", str(xgb_cfg.get("subsample", 0.8)),
        "--colsample-bytree", str(xgb_cfg.get("colsample_bytree", 0.8)),
        "--min-child-weight", str(xgb_cfg.get("min_child_weight", 5.0)),
        "--gamma", str(xgb_cfg.get("gamma", 0.1)),
        "--reg-alpha", str(xgb_cfg.get("reg_alpha", 0.05)),
        "--reg-lambda", str(xgb_cfg.get("reg_lambda", 5.0)),
        "--colsample-bylevel", str(xgb_cfg.get("colsample_bylevel", 0.8)),
        "--colsample-bynode", str(xgb_cfg.get("colsample_bynode", 0.8)),
        "--tree-method", str(xgb_cfg.get("tree_method", "hist")),
        "--n-jobs", str(xgb_cfg.get("n_jobs", 16)),
        "--early-stopping-rounds", str(xgb_cfg.get("early_stopping_rounds", 50)),
        "--seed", str(seed),
        "--output-dir", output_base,
        "--run-name", run_name,
    ]
    if cv_folds and cv_folds > 1:
        cmd += ["--cv-folds", str(cv_folds)]
    if not xgb_cfg.get("use_scaler", True):
        cmd.append("--no-scaler")

    print(f"  CMD: ... ({len(emb_names)} models, seed={seed})")
    log_file = os.path.join(log_dir, f"xgboost_seed{seed}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"  ✗ FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-15:]:
                print(f"    {line.rstrip()}")
        return None

    out = output_dir
    print(f"  ✓ Done in {time.time() - t0:.0f}s → {out}")
    return out

# ======== Phase 2b: k-NN ========

def run_knn(emb_names: list, cfg: dict, log_dir: str) -> str:
    script = os.path.join(os.path.dirname(__file__), "knn_eval.py")
    knn_cfg = cfg["knn"]
    processed_dir = cfg["data"]["processed_dir"]

    emb_args = [f"{n}:{os.path.join(cfg['emb_root'], n)}" for n in emb_names]
    k_args = [str(k) for k in knn_cfg["k_values"]]

    cmd = [
        sys.executable, script,
        "--embedding-dirs", *emb_args,
        "--train-csv", os.path.join(processed_dir, "train.csv"),
        "--test-csv", os.path.join(processed_dir, "test.csv"),
        "--k", *k_args,
        "--metric", knn_cfg.get("metric", "cosine"),
        "--weights", knn_cfg.get("weights", "distance"),
        "--output-dir", knn_cfg.get("output_dir", "experiments/knn"),
        "--save-predictions",
    ]
    if knn_cfg.get("normalize", False):
        cmd.append("--normalize")

    print(f"  CMD: ... ({len(emb_names)} models)")
    log_file = os.path.join(log_dir, "knn.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(f"  ✗ FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-15:]:
                print(f"    {line.rstrip()}")
        return None

    base = knn_cfg.get("output_dir", "experiments/knn")
    runs = sorted(Path(base).glob("run_*"))
    out = str(runs[-1]) if runs else base
    print(f"  ✓ Done in {time.time() - t0:.0f}s → {out}")
    return out

# ======== Summary ========

def print_summary_legacy(xgb_dir: str, knn_dir: str):
    import pandas as pd

    print("\n## pH Regression Results\n")

    if xgb_dir:
        csv = os.path.join(xgb_dir, "results.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            print("### XGBoost\n")
            print("| Model | MAE | RMSE | R² | PCC | SCC |")
            print("|:------|----:|-----:|---:|----:|----:|")
            for _, r in df.sort_values("test_mae").iterrows():
                print(f"| {r['embedding']} | {r['test_mae']:.4f} | "
                      f"{r['test_rmse']:.4f} | {r['test_r2']:.4f} | "
                      f"{r['test_pearson_r']:.4f} | {r['test_spearman_r']:.4f} |")
            print()

    if knn_dir:
        csv = os.path.join(knn_dir, "results.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            print("### k-NN (best k)\n")
            print("| Model | k | MAE | R² | SCC |")
            print("|:------|--:|----:|---:|----:|")
            for name in df["embedding"].unique():
                sub = df[df["embedding"] == name]
                b = sub.loc[sub["mae"].idxmin()]
                print(f"| {name} | {int(b['k'])} | {b['mae']:.4f} | "
                      f"{b['r2']:.4f} | {b['spearman_r']:.4f} |")
            print()

def _fmt_mean_std(mean, std, decimals=4):
    if mean is None:
        return "-"
    if std is None or abs(std) < 1e-12:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"

def _collect_xgb_results(xgb_dirs):
    import pandas as pd

    if isinstance(xgb_dirs, str):
        xgb_dirs = [xgb_dirs]

    frames = []
    for xgb_dir in xgb_dirs or []:
        if not xgb_dir:
            continue
        csv_path = os.path.join(xgb_dir, "results.csv")
        if not os.path.exists(csv_path):
            continue

        seed = None
        config_path = os.path.join(xgb_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                seed = json.load(f).get("seed")
        if seed is None:
            base = os.path.basename(os.path.normpath(xgb_dir))
            if base.startswith("seed_"):
                seed = base.replace("seed_", "")

        df = pd.read_csv(csv_path)
        df["seed"] = int(seed) if seed is not None else -1
        df["run_dir"] = xgb_dir
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _write_xgb_summary(df, output_base):
    import pandas as pd

    metric_cols = [
        "train_cv_mae_mean",
        "train_cv_mae_std",
        "train_mae",
        "val_mae",
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_pearson_r",
        "test_spearman_r",
    ]
    rows = []
    for embedding, sub in df.groupby("embedding"):
        row = {"embedding": embedding, "n_seeds": int(sub["seed"].nunique())}
        for col in metric_cols:
            if col in sub:
                row[f"{col}_mean"] = float(sub[col].mean())
                row[f"{col}_std"] = float(sub[col].std(ddof=0))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("test_mae_mean")
    os.makedirs(output_base, exist_ok=True)
    summary.to_csv(os.path.join(output_base, "sweep_summary.csv"), index=False)
    df.to_csv(os.path.join(output_base, "sweep_summary_per_seed.csv"), index=False)
    return summary

def _xgb_output_base(xgb_dirs):
    if isinstance(xgb_dirs, str):
        xgb_dirs = [xgb_dirs]
    first_dir = next((d for d in xgb_dirs or [] if d), None)
    if not first_dir:
        return "experiments/xgboost"
    return os.path.dirname(os.path.normpath(first_dir))

def print_summary(xgb_dirs, knn_dir: str):
    import pandas as pd

    print("\n## pH Regression Results\n")

    xgb_df = _collect_xgb_results(xgb_dirs)
    if xgb_df is not None:
        xgb_base = _xgb_output_base(xgb_dirs)
        xgb_summary = _write_xgb_summary(xgb_df, xgb_base)
        print("### XGBoost\n")
        print("| Model | Seeds | Train CV MAE | Val MAE | Test MAE | Test RMSE | Test R2 | Test SCC |")
        print("|:------|------:|-------------:|--------:|---------:|----------:|--------:|---------:|")
        for _, r in xgb_summary.iterrows():
            print(f"| {r['embedding']} | {int(r['n_seeds'])} | "
                  f"{_fmt_mean_std(r.get('train_cv_mae_mean_mean'), r.get('train_cv_mae_mean_std'))} | "
                  f"{_fmt_mean_std(r.get('val_mae_mean'), r.get('val_mae_std'))} | "
                  f"{_fmt_mean_std(r.get('test_mae_mean'), r.get('test_mae_std'))} | "
                  f"{_fmt_mean_std(r.get('test_rmse_mean'), r.get('test_rmse_std'))} | "
                  f"{_fmt_mean_std(r.get('test_r2_mean'), r.get('test_r2_std'))} | "
                  f"{_fmt_mean_std(r.get('test_spearman_r_mean'), r.get('test_spearman_r_std'))} |")
        print(f"\nXGBoost summary CSV: {os.path.join(xgb_base, 'sweep_summary.csv')}")
        print(f"XGBoost per-seed CSV: {os.path.join(xgb_base, 'sweep_summary_per_seed.csv')}")
        print()

    if knn_dir:
        csv = os.path.join(knn_dir, "results.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            print("### k-NN (deterministic best k)\n")
            print("| Model | k | MAE | R2 | SCC |")
            print("|:------|--:|----:|---:|----:|")
            for name in df["embedding"].unique():
                sub = df[df["embedding"] == name]
                b = sub.loc[sub["mae"].idxmin()]
                print(f"| {name} | {int(b['k'])} | {b['mae']:.4f} | "
                      f"{b['r2']:.4f} | {b['spearman_r']:.4f} |")
            print()

# ======== Main ========

def main():
    parser = argparse.ArgumentParser(description="pH Regression Sweep")
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-parsing", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    parser.add_argument(
        "--overwrite-embedding",
        action="store_true",
        help="Regenerate external-PLM artifacts, including stale ESM3 pre-norm files",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)
    seeds = get_eval_seeds(cfg)
    # 解析 models
    global_encoder_type = cfg.get("encoder_type")  # 不再硬编码默认值，由 parse_model_entry 校验
    models = [parse_model_entry(n, e, global_encoder_type) for n, e in cfg["models"].items()]

    log_dir = os.path.join("logs", f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    tracker = StatusTracker(log_dir)
    if args.reset:
        tracker.reset()

    # 打印计划
    print("=" * 70)
    print(f" pH Regression Sweep  |  {datetime.now()}")
    print(f" Config: {config_path}")
    print(f" Seeds:  {seeds}")
    if "xgboost" in cfg:
        print(f" XGB CV: {int(cfg['xgboost'].get('cv_folds', 0) or 0)} folds")
    print("=" * 70)

    print(f"\nModels ({len(models)}):")
    for m in models:
        ckpt = os.path.basename(m["checkpoint"]) if m["checkpoint"] else "—"
        label = MODE_DESC.get(m["mode"], m["mode"])
        print(f"  {m['name']:35s}  {m['source']:12s}  "
              f"{m['encoder_type']:32s}  {m['mode']:8s} ({label})  ckpt={ckpt}")

    if args.dry_run:
        print("\n[DRY RUN] Done.")
        return

    # Phase 0
    if not args.skip_parsing:
        print(f"\n{'='*70}\n Phase 0: Parse FASTA\n{'='*70}")
        if not run_parse_fasta(cfg, log_dir):
            return
    else:
        print("\n⏭  Skip parsing")

    # Phase 1
    successful = []
    if not args.skip_embedding:
        print(f"\n{'='*70}\n Phase 1: Embedding\n{'='*70}")
        for i, m in enumerate(models, 1):
            name = m["name"]
            print(f"\n[{i}/{len(models)}] {name}")

            if tracker.get(name) == "embedding_done":
                print("  → Already done")
                successful.append(name)
                continue
            if (m["source"] == "enzsub" and m["checkpoint"]
                    and not os.path.exists(m["checkpoint"])):
                print(f"  ✗ Checkpoint missing: {m['checkpoint']}")
                tracker.set(name, "ckpt_missing")
                continue
            if (m["source"] == "enzsub" and m["mode"] in CHECKPOINT_MODES
                    and not m["checkpoint"]):
                print(f"  ✗ mode={m['mode']} needs checkpoint")
                tracker.set(name, "config_error")
                continue

            if run_embedding(
                m,
                cfg,
                config_path,
                log_dir,
                overwrite_external=args.overwrite_embedding,
            ):
                tracker.set(name, "embedding_done")
                successful.append(name)
            else:
                tracker.set(name, "embedding_failed")

        print(f"\nEmbedding: {len(successful)}/{len(models)} ok")
    else:
        print("\n⏭  Skip embedding")
        expected_files = [
            f"{split}_all.pt"
            for split in cfg["data"].get("splits", ["train", "val", "test"])
        ]
        for m in models:
            d = os.path.join(cfg["emb_root"], m["name"])
            files_exist = os.path.isdir(d) and all(
                os.path.exists(os.path.join(d, filename))
                for filename in expected_files
            )
            if not files_exist:
                continue
            if m["source"] == "external_plm":
                ready, reason = validate_external_artifacts(m, cfg)
                if not ready:
                    print(f"  ✗ {m['name']}: {reason}")
                    continue
            successful.append(m["name"])
        print(f"Found {len(successful)} existing")

    if not successful:
        print("No embeddings, aborting")
        return

    # Phase 2a
    xgb_out = []
    if not args.skip_xgboost and "xgboost" in cfg:
        print(f"\n{'='*70}\n Phase 2a: XGBoost ({len(successful)} models x {len(seeds)} seeds)\n{'='*70}")
        for seed in seeds:
            print(f"\n[seed={seed}]")
            out = run_xgboost(successful, cfg, log_dir, seed)
            if out:
                xgb_out.append(out)

    # Phase 2b
    knn_out = None
    if not args.skip_knn and "knn" in cfg:
        print(f"\n{'='*70}\n Phase 2b: k-NN ({len(successful)} models)\n{'='*70}")
        knn_out = run_knn(successful, cfg, log_dir)

    # Summary
    if xgb_out or knn_out:
        print_summary(xgb_out, knn_out)

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(xgb_out, knn_out)
        with open(os.path.join(log_dir, "summary.md"), "w") as f:
            f.write(buf.getvalue())

    print(f"\n{'='*70}")
    print(f" Done: {datetime.now()}")
    print(f" Logs: {log_dir}")
    if xgb_out:
        print(f" XGBoost: {', '.join(xgb_out)}")
    if knn_out:
        print(f" k-NN: {knn_out}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
