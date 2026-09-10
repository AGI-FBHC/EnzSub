#!/usr/bin/env python3
"""
Tm 预测 Checkpoint Sweep 编排脚本

功能:
  1. 读取 sweep_config.yaml
  2. 逐模型生成 mean-pooled embedding (train/val/test)
  3. 运行 XGBoost 和/或 k-NN 评估
  4. 输出 Markdown 汇总表

用法:
  python sweep.py                          # 完整运行
  python sweep.py --config sweep_config.yaml
  python sweep.py --dry-run
  python sweep.py --skip-embedding
  python sweep.py --skip-eval
  python sweep.py --reset
"""

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

MODE_DESC = {
    "base": "pretrained", "cpt": "CPT",
    "base_sub": "base+SUB", "cpt_sub": "CPT+SUB",
}

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def parse_model_entry(name, entry, default_encoder_type=None):
    """
    Per-model encoder_type:
      - dict entry: use 'encoder_type' key, fall back to global default
      - str/None entry: use global default
    """
    if entry is None:
        return {'name': name, 'source': 'enzsub', 'mode': 'base', 'checkpoint': None,
                'lora_rank': 8, 'lora_alpha': 16,
                'encoder_type': default_encoder_type}
    if isinstance(entry, str):
        return {'name': name, 'source': 'enzsub', 'mode': 'cpt', 'checkpoint': entry,
                'lora_rank': 8, 'lora_alpha': 16,
                'encoder_type': default_encoder_type}
    if isinstance(entry, dict):
        source = entry.get('source', 'enzsub')
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source={source!r} for {name}; expected one of {VALID_SOURCES}"
            )
        mode = entry.get('mode', 'base')
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode={mode!r} for {name}")
        if source == 'external_plm':
            if mode != 'base':
                raise ValueError(f"External PLM {name} only supports mode=base")
            if not entry.get('backend') or not entry.get('model_id'):
                raise ValueError(
                    f"External PLM {name} requires backend and model_id"
                )
            return {
                'name': name,
                'source': source,
                'mode': mode,
                'checkpoint': None,
                'lora_rank': None,
                'lora_alpha': None,
                'encoder_type': entry['model_id'],
                'backend': entry['backend'],
            }
        return {
            'name': name, 'source': source, 'mode': mode,
            'checkpoint': entry.get('checkpoint'),
            'lora_rank': entry.get('lora_rank', 8),
            'lora_alpha': entry.get('lora_alpha', 16),
            'encoder_type': entry.get('encoder_type', default_encoder_type),
        }
    raise ValueError(f"Invalid model entry for {name}: {entry}")

def get_eval_seeds(cfg):
    eval_cfg = cfg.get("eval", {})
    if "seeds" in eval_cfg:
        seeds = eval_cfg["seeds"]
    elif "seed" in eval_cfg:
        seeds = [eval_cfg["seed"]]
    else:
        seeds = [2025]

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("eval.seeds must be a non-empty list, e.g. seeds: [2025, 2026, 2027]")

    seeds = [int(s) for s in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"eval.seeds contains duplicate values: {seeds}")
    return seeds

def get_xgb_cv_folds(cfg):
    """读取 XGBoost 训练集内部 CV 折数。0/1 表示关闭。"""
    eval_cfg = cfg.get("eval", {})
    xgb_cfg = eval_cfg.get("xgboost", {}) or {}
    return int(xgb_cfg.get("cv_folds", 0))

def _xgb_metrics_complete(metrics_file, cv_folds=0):
    """判断已有 XGBoost metrics 是否满足当前配置。"""
    if not os.path.exists(metrics_file):
        return False
    if not cv_folds or cv_folds <= 1:
        return True
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
    except Exception:
        return False
    return f"train_cv_{cv_folds}fold" in metrics

def _metrics_are_current(metrics_file, emb_dir):
    """Return False when embeddings/metadata are newer than saved evaluation."""
    if not os.path.exists(metrics_file):
        return False
    inputs = []
    for split in ("train", "val", "test"):
        for filename in (f"{split}.pt", f"{split}.metadata.json"):
            path = os.path.join(emb_dir, filename)
            if os.path.exists(path):
                inputs.append(path)
    if not inputs:
        return False
    return os.path.getmtime(metrics_file) >= max(os.path.getmtime(path) for path in inputs)

class StatusTracker:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "status.json")
        self.status = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.status = json.load(f)

    def get(self, name):
        return self.status.get(name, "pending")

    def set(self, name, state):
        self.status[name] = state
        with open(self.path, "w") as f:
            json.dump(self.status, f, indent=2)

    def reset(self):
        self.status = {}
        with open(self.path, "w") as f:
            json.dump({}, f)

# ======== Phase 1: Embedding ========

def check_tm_embeddings(emb_dir):
    """检查 train/val/test.pt 是否都存在"""
    for s in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(emb_dir, f"{s}.pt")):
            return False
    return True

def validate_external_artifacts(model_info, cfg):
    """Validate complete external-PLM artifacts without changing legacy loaders."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is unavailable for artifact validation"

    emb_dir = os.path.join(cfg['emb_root'], model_info['name'])
    expected_pooling = (
        ESM3_POOLING if model_info['backend'] == 'esm3' else HF_POOLING
    )
    dimensions = set()
    for split in ('train', 'val', 'test'):
        emb_path = os.path.join(emb_dir, f"{split}.pt")
        metadata_path = os.path.join(emb_dir, f"{split}.metadata.json")
        if not os.path.exists(emb_path):
            return False, f"missing {split}.pt"
        if not os.path.exists(metadata_path):
            return False, f"missing {split}.metadata.json"
        try:
            with open(metadata_path, encoding='utf-8') as handle:
                metadata = json.load(handle)
            embeddings = torch.load(emb_path, map_location='cpu')
        except Exception as exc:
            return False, f"cannot load {split} artifact: {exc}"
        if not isinstance(embeddings, dict):
            return False, f"{split}.pt is not an embedding dictionary"
        checks = {
            'model_name': model_info['name'],
            'model_id': model_info['encoder_type'],
            'backend': model_info['backend'],
            'pooling': expected_pooling,
        }
        for key, expected in checks.items():
            if metadata.get(key) != expected:
                return False, (
                    f"{split}.metadata.json has {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if metadata.get('complete') is not True or metadata.get('limited') is not False:
            return False, f"{split} artifact is partial or smoke-test limited"
        hashes = metadata.get('sequence_sha256')
        if not isinstance(hashes, dict) or set(hashes) != set(embeddings):
            return False, f"{split} sequence hashes do not match embedding IDs"
        if metadata.get('completed_records') != len(embeddings):
            return False, f"{split} completed_records does not match embeddings"
        if metadata.get('unique_records') != len(embeddings):
            return False, f"{split} artifact does not cover every unique FASTA record"
        split_dims = set()
        for seq_id, value in embeddings.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                return False, f"{split} embedding {seq_id!r} is not a 1D Tensor"
            if not torch.isfinite(value).all():
                return False, f"{split} embedding {seq_id!r} contains NaN/Inf"
            split_dims.add(int(value.numel()))
        if len(split_dims) != 1:
            return False, f"{split} has inconsistent embedding dimensions"
        dimension = next(iter(split_dims))
        if metadata.get('embedding_dim') != dimension:
            return False, f"{split} embedding_dim metadata mismatch"
        dimensions.add(dimension)
    if len(dimensions) != 1:
        return False, f"embedding dimension differs across splits: {sorted(dimensions)}"
    return True, "ok"

def check_model_embeddings(model_info, cfg):
    emb_dir = os.path.join(cfg['emb_root'], model_info['name'])
    if model_info.get('source', 'enzsub') == 'external_plm':
        return validate_external_artifacts(model_info, cfg)
    if check_tm_embeddings(emb_dir):
        return True, "ok"
    return False, "missing train.pt, val.pt, or test.pt"

def run_embedding(model_info, cfg, config_path, log_dir, overwrite_external=False):
    name = model_info['name']
    mode = model_info['mode']
    encoder_type = model_info['encoder_type']
    data = cfg['data']
    emb_dir = os.path.join(cfg['emb_root'], name)

    ready, _ = check_model_embeddings(model_info, cfg)
    if ready:
        print(f"    ✓ Complete embedding artifacts exist, skipping")
        return True

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if model_info.get('source', 'enzsub') == 'external_plm':
        script = os.path.join(script_dir, "generate_plm_embeddings.py")
        cmd = [
            sys.executable, script,
            "--config", config_path,
            "--models", name,
            "--device", cfg["device"],
        ]
        if overwrite_external:
            cmd.append("--overwrite")
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("RAYON_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        log_file = os.path.join(log_dir, f"emb_{name}.log")
        print(
            f"    CMD: [external_plm/{model_info['backend']}] "
            f"{model_info['encoder_type']}"
        )
        t0 = time.time()
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"    ✗ FAILED (code {proc.returncode}), see {log_file}")
            with open(log_file) as lf:
                for line in lf.readlines()[-8:]:
                    print(f"      {line.rstrip()}")
            return False
        print(f"    ✓ Done in {elapsed:.0f}s")
        return True

    script = os.path.join(script_dir, "generate_tm_embeddings.py")

    # 构建 fasta 参数: train:path val:path test:path
    fasta_args = []
    for split in ["train", "val", "test"]:
        fasta_key = f"{split}_fasta"
        if fasta_key in data:
            fasta_args.append(f"{split}:{data[fasta_key]}")

    cmd = [
        sys.executable, script,
        "--encoder-type", encoder_type,
        "--model-mode", mode,
        "--batch-size", str(cfg.get("batch_size", 32)),
        "--device", cfg["device"],
        "--fasta", *fasta_args,
        "--output-dir", emb_dir,
    ]

    if model_info['checkpoint']:
        cmd += ["--checkpoint", model_info['checkpoint']]
    if mode in ("base_sub", "cpt_sub"):
        cmd += ["--lora-rank", str(model_info['lora_rank'])]
        cmd += ["--lora-alpha", str(model_info['lora_alpha'])]

    env = os.environ.copy()
    sub_parent_dir = cfg.get("sub_parent_dir")
    if sub_parent_dir:
        env["PYTHONPATH"] = sub_parent_dir + os.pathsep + env.get("PYTHONPATH", "")

    log_file = os.path.join(log_dir, f"emb_{name}.log")
    print(f"    CMD: {' '.join(cmd[:8])} ...")

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ FAILED (code {proc.returncode}), see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-5:]:
                print(f"      {line.rstrip()}")
        return False

    print(f"    ✓ Done in {elapsed:.0f}s")
    return True

# ======== Phase 2: XGBoost ========

def run_xgboost(model_info, cfg, log_dir, seed):
    name = model_info['name']
    data = cfg['data']
    cv_folds = get_xgb_cv_folds(cfg)

    emb_dir = os.path.join(cfg['emb_root'], name)
    output_dir = os.path.join(cfg['results_root'], "xgboost", name, f"seed_{seed}")

    # 检查已完成。若当前配置要求 CV，而旧 metrics 中没有 CV，则补跑。
    metrics_file = os.path.join(output_dir, "metrics.json")
    metrics_complete = _xgb_metrics_complete(metrics_file, cv_folds)
    metrics_current = _metrics_are_current(metrics_file, emb_dir)
    if metrics_complete and metrics_current:
        _print_xgb_results(metrics_file)
        return True
    stale_metrics = os.path.exists(metrics_file) and not metrics_current
    if stale_metrics:
        print(f"    Existing XGB metrics are older than embeddings, rerunning")
    elif os.path.exists(metrics_file) and cv_folds and cv_folds > 1:
        print(f"    Existing XGB metrics found but CV is missing, rerunning")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "train_xgboost_tm.py")

    cmd = [
        sys.executable, script,
        "--emb-dir", emb_dir,
        "--train-csv", data["train_csv"],
        "--val-csv", data["val_csv"],
        "--test-csv", data["test_csv"],
        "--model-name", name,
        "--output-dir", output_dir,
        "--seed", str(seed),
    ]
    xgb_cfg = cfg.get("eval", {}).get("xgboost", {}) or {}
    if "early_stopping_rounds" in xgb_cfg:
        cmd += ["--early-stopping-rounds", str(int(xgb_cfg["early_stopping_rounds"]))]
    if cv_folds and cv_folds > 1:
        cmd += ["--cv-folds", str(cv_folds)]
    if stale_metrics:
        cmd.append("--overwrite")

    log_file = os.path.join(log_dir, f"xgb_{name}_seed{seed}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ XGBoost FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-8:]:
                print(f"      {line.rstrip()}")
        return False

    print(f"    ✓ XGBoost done in {elapsed:.0f}s")
    _print_xgb_results(metrics_file)
    return True

def _print_xgb_results(metrics_file):
    try:
        with open(metrics_file) as f:
            r = json.load(f)
        t = r.get('test', {})
        print(f"    XGB Test: MAE={t.get('MAE', 0):.4f}  "
              f"R²={t.get('R2', 0):.4f}  Spearman={t.get('Spearman_r', 0):.4f}")
        c = r.get('train_cv_5fold')
        if c:
            print(f"    XGB Train-CV: MAE={c.get('MAE_mean', 0):.4f}+/-{c.get('MAE_std', 0):.4f}  "
                  f"R²={c.get('R2_mean', 0):.4f}+/-{c.get('R2_std', 0):.4f}")
    except Exception:
        pass

# ======== Phase 2: k-NN ========

def run_knn(model_info, cfg, log_dir):
    name = model_info['name']
    data = cfg['data']
    knn_cfg = cfg['eval'].get('knn', {})

    emb_dir = os.path.join(cfg['emb_root'], name)
    output_dir = os.path.join(cfg['results_root'], "knn", name)

    results_file = os.path.join(output_dir, "knn_results.json")
    if os.path.exists(results_file):
        _print_knn_results(results_file)
        return True

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "knn_tm.py")

    ks = knn_cfg.get("ks", [1])

    cmd = [
        sys.executable, script,
        "--emb-dir", emb_dir,
        "--train-csv", data["train_csv"],
        "--test-csv", data["test_csv"],
        "--model-name", name,
        "--output-dir", output_dir,
        "--ks", *[str(k) for k in ks],
        "--metric", knn_cfg.get("metric", "cosine"),
    ]

    log_file = os.path.join(log_dir, f"knn_{name}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ k-NN FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-8:]:
                print(f"      {line.rstrip()}")
        return False

    print(f"    ✓ k-NN done in {elapsed:.0f}s")
    _print_knn_results(results_file)
    return True

def _print_knn_results(results_file):
    try:
        with open(results_file) as f:
            r = json.load(f)
        t = r.get('best_test_metrics', {})
        print(f"    k-NN (k={r.get('best_k')}): MAE={t.get('MAE', 0):.4f}  "
              f"R²={t.get('R2', 0):.4f}  Spearman={t.get('Spearman_r', 0):.4f}")
    except Exception:
        pass

# ======== Phase 3: 汇总 ========

def print_summary_legacy(cfg, models):
    import csv

    seed = cfg['eval']['seed']
    methods = cfg['eval'].get('methods', ['xgboost'])
    results_root = cfg['results_root']

    print(f"\n{'='*70}")
    print(f" Results Summary")
    print(f"{'='*70}")

    rows = []
    for m in models:
        name = m['name']
        row = {"model": name, "mode": m['mode'], "encoder": m['encoder_type']}

        if 'xgboost' in methods:
            f = os.path.join(results_root, "xgboost", name, f"seed_{seed}", "metrics.json")
            if os.path.exists(f):
                with open(f) as fp:
                    r = json.load(fp)
                t = r.get('test', {})
                row["xgb_MAE"] = t.get("MAE", 0)
                row["xgb_RMSE"] = t.get("RMSE", 0)
                row["xgb_R2"] = t.get("R2", 0)
                row["xgb_Spearman"] = t.get("Spearman_r", 0)

        if 'knn' in methods:
            f = os.path.join(results_root, "knn", name, "knn_results.json")
            if os.path.exists(f):
                with open(f) as fp:
                    r = json.load(fp)
                t = r.get('best_test_metrics', {})
                row["knn_k"] = r.get("best_k", "?")
                row["knn_MAE"] = t.get("MAE", 0)
                row["knn_R2"] = t.get("R2", 0)
                row["knn_Spearman"] = t.get("Spearman_r", 0)

        rows.append(row)

    if not rows:
        print("  No results found.")
        return

    # XGBoost 表
    if 'xgboost' in methods:
        xgb_rows = [r for r in rows if "xgb_MAE" in r]
        if xgb_rows:
            xgb_rows.sort(key=lambda x: x.get("xgb_MAE", 999))
            print(f"\n### XGBoost Regression (sorted by MAE ↑)\n")
            print(f"| {'Model':<35s} | {'Mode':<8s} | {'MAE':>7s} | "
                  f"{'RMSE':>7s} | {'R²':>7s} | {'Spearman':>8s} |")
            print(f"|{'-'*37}|{'-'*10}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*10}|")
            for r in xgb_rows:
                print(f"| {r['model']:<35s} | {r['mode']:<8s} | "
                      f"{r.get('xgb_MAE',0):7.4f} | {r.get('xgb_RMSE',0):7.4f} | "
                      f"{r.get('xgb_R2',0):7.4f} | {r.get('xgb_Spearman',0):8.4f} |")
            best = xgb_rows[0]
            print(f"\n🏆 Best XGB: {best['model']}  MAE={best.get('xgb_MAE',0):.4f}°C")

    # k-NN 表
    if 'knn' in methods:
        knn_rows = [r for r in rows if "knn_MAE" in r]
        if knn_rows:
            knn_rows.sort(key=lambda x: x.get("knn_MAE", 999))
            print(f"\n### k-NN Regression (sorted by MAE ↑)\n")
            print(f"| {'Model':<35s} | {'Mode':<8s} | {'Best k':>6s} | "
                  f"{'MAE':>7s} | {'R²':>7s} | {'Spearman':>8s} |")
            print(f"|{'-'*37}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*10}|")
            for r in knn_rows:
                print(f"| {r['model']:<35s} | {r['mode']:<8s} | "
                      f"{r.get('knn_k','?'):>6} | {r.get('knn_MAE',0):7.4f} | "
                      f"{r.get('knn_R2',0):7.4f} | {r.get('knn_Spearman',0):8.4f} |")
            best = knn_rows[0]
            print(f"\n🏆 Best k-NN: {best['model']}  MAE={best.get('knn_MAE',0):.4f}°C  "
                  f"k={best.get('knn_k','?')}")

    # CSV
    csv_path = os.path.join(results_root, "sweep_summary.csv")
    if rows:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSummary CSV: {csv_path}")

def _mean_std(values):
    import numpy as np

    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())

def _fmt_mean_std(mean, std, decimals=4):
    if mean is None:
        return "-"
    if std is None or abs(std) < 1e-12:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"

def _write_csv(csv_path, rows):
    import csv

    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _collect_xgb_seed_results(cfg, model_name, seeds):
    records = []
    for seed in seeds:
        metrics_file = os.path.join(
            cfg["results_root"], "xgboost", model_name, f"seed_{seed}", "metrics.json"
        )
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        records.append({"seed": seed, "metrics": metrics})
    return records

def _add_split_aggregates(row, records, prefix, split, metrics):
    for metric in metrics:
        values = [r["metrics"].get(split, {}).get(metric) for r in records]
        mean, std = _mean_std(values)
        row[f"{prefix}_{split}_{metric}_mean"] = mean
        row[f"{prefix}_{split}_{metric}_std"] = std

def _add_train_cv_aggregates(row, records, prefix, metrics):
    """汇总训练集内部 K-fold CV 的 mean/std。"""
    for metric in metrics:
        cv_means = [
            r["metrics"].get("train_cv_5fold", {}).get(f"{metric}_mean")
            for r in records
        ]
        cv_stds = [
            r["metrics"].get("train_cv_5fold", {}).get(f"{metric}_std")
            for r in records
        ]
        mean_of_means, seed_std = _mean_std(cv_means)
        fold_std_mean, _ = _mean_std(cv_stds)
        row[f"{prefix}_train_cv_{metric}_mean"] = mean_of_means
        row[f"{prefix}_train_cv_{metric}_std"] = fold_std_mean if fold_std_mean is not None else seed_std

def _build_summary_rows(cfg, models, seeds):
    methods = cfg['eval'].get('methods', ['xgboost'])
    results_root = cfg['results_root']
    agg_rows = []
    per_seed_rows = []

    for m in models:
        name = m['name']
        row = {"model": name, "mode": m['mode'], "encoder": m['encoder_type']}

        if 'xgboost' in methods:
            records = _collect_xgb_seed_results(cfg, name, seeds)
            row["xgb_n_seeds"] = len(records)
            for record in records:
                metrics = record["metrics"]
                seed_row = {
                    "model": name,
                    "mode": m["mode"],
                    "encoder": m["encoder_type"],
                    "method": "xgboost",
                    "seed": record["seed"],
                }
                for split in ("train", "val", "test"):
                    for metric in ("MAE", "RMSE", "R2", "Pearson_r", "Spearman_r"):
                        seed_row[f"{split}_{metric}"] = metrics.get(split, {}).get(metric)
                cv = metrics.get("train_cv_5fold", {})
                for metric in ("MAE", "RMSE", "R2", "Pearson_r", "Spearman_r"):
                    seed_row[f"train_cv_{metric}_mean"] = cv.get(f"{metric}_mean")
                    seed_row[f"train_cv_{metric}_std"] = cv.get(f"{metric}_std")
                per_seed_rows.append(seed_row)

            _add_train_cv_aggregates(
                row, records, "xgb",
                ("MAE", "RMSE", "R2", "Pearson_r", "Spearman_r")
            )
            for split in ("val", "test"):
                _add_split_aggregates(
                    row, records, "xgb", split,
                    ("MAE", "RMSE", "R2", "Pearson_r", "Spearman_r")
                )

        if 'knn' in methods:
            f = os.path.join(results_root, "knn", name, "knn_results.json")
            if os.path.exists(f):
                with open(f) as fp:
                    r = json.load(fp)
                t = r.get('best_test_metrics', {})
                row["knn_k"] = r.get("best_k", "?")
                row["knn_test_MAE"] = t.get("MAE")
                row["knn_test_RMSE"] = t.get("RMSE")
                row["knn_test_R2"] = t.get("R2")
                row["knn_test_Pearson_r"] = t.get("Pearson_r")
                row["knn_test_Spearman_r"] = t.get("Spearman_r")

        agg_rows.append(row)

    return agg_rows, per_seed_rows

def print_summary(cfg, models):
    seeds = get_eval_seeds(cfg)
    methods = cfg['eval'].get('methods', ['xgboost'])
    results_root = cfg['results_root']

    print(f"\n{'='*70}")
    print(" Results Summary")
    print(f"{'='*70}")

    rows, per_seed_rows = _build_summary_rows(cfg, models, seeds)
    if not rows:
        print("  No results found.")
        return

    if 'xgboost' in methods:
        xgb_rows = [r for r in rows if r.get("xgb_n_seeds", 0) > 0]
        if xgb_rows:
            xgb_rows.sort(
                key=lambda x: x.get("xgb_test_MAE_mean")
                if x.get("xgb_test_MAE_mean") is not None else 999
            )
            print(f"\n### XGBoost Regression (n={len(seeds)} seed config, sorted by test MAE)\n")
            print(f"| {'Model':<35s} | {'Encoder':<14s} | {'Mode':<8s} | {'Seeds':>5s} | "
                  f"{'Train CV MAE':>17s} | {'Val MAE':>17s} | {'Test MAE':>17s} | "
                  f"{'Test R2':>17s} | {'Test Spearman':>17s} |")
            print(f"|{'-'*37}|{'-'*16}|{'-'*10}|{'-'*7}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*19}|")
            for r in xgb_rows:
                print(f"| {r['model']:<35s} | {str(r.get('encoder','-')):<14s} | "
                      f"{r['mode']:<8s} | {r.get('xgb_n_seeds',0):5d} | "
                      f"{_fmt_mean_std(r.get('xgb_train_cv_MAE_mean'), r.get('xgb_train_cv_MAE_std')):>17s} | "
                      f"{_fmt_mean_std(r.get('xgb_val_MAE_mean'), r.get('xgb_val_MAE_std')):>17s} | "
                      f"{_fmt_mean_std(r.get('xgb_test_MAE_mean'), r.get('xgb_test_MAE_std')):>17s} | "
                      f"{_fmt_mean_std(r.get('xgb_test_R2_mean'), r.get('xgb_test_R2_std')):>17s} | "
                      f"{_fmt_mean_std(r.get('xgb_test_Spearman_r_mean'), r.get('xgb_test_Spearman_r_std')):>17s} |")
            best = xgb_rows[0]
            print(f"\nBest XGB: {best['model']}  "
                  f"Test MAE={_fmt_mean_std(best.get('xgb_test_MAE_mean'), best.get('xgb_test_MAE_std'))}")

    if 'knn' in methods:
        knn_rows = [r for r in rows if "knn_test_MAE" in r]
        if knn_rows:
            knn_rows.sort(key=lambda x: x.get("knn_test_MAE") if x.get("knn_test_MAE") is not None else 999)
            print("\n### k-NN Regression (deterministic, sorted by MAE)\n")
            print(f"| {'Model':<35s} | {'Encoder':<14s} | {'Mode':<8s} | {'Best k':>6s} | "
                  f"{'MAE':>7s} | {'R2':>7s} | {'Spearman':>8s} |")
            print(f"|{'-'*37}|{'-'*16}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*9}|{'-'*10}|")
            for r in knn_rows:
                print(f"| {r['model']:<35s} | {str(r.get('encoder','-')):<14s} | "
                      f"{r['mode']:<8s} | "
                      f"{r.get('knn_k','?'):>6} | {r.get('knn_test_MAE',0):7.4f} | "
                      f"{r.get('knn_test_R2',0):7.4f} | {r.get('knn_test_Spearman_r',0):8.4f} |")
            best = knn_rows[0]
            print(f"\nBest k-NN: {best['model']}  MAE={best.get('knn_test_MAE',0):.4f}  "
                  f"k={best.get('knn_k','?')}")

    summary_csv = os.path.join(results_root, "sweep_summary.csv")
    per_seed_csv = os.path.join(results_root, "sweep_summary_per_seed.csv")
    _write_csv(summary_csv, rows)
    _write_csv(per_seed_csv, per_seed_rows)
    print(f"\nSummary CSV: {summary_csv}")
    if per_seed_rows:
        print(f"Per-seed CSV: {per_seed_csv}")

# ======== Main ========

def main():
    parser = argparse.ArgumentParser(
        description="Tm Prediction Checkpoint Sweep Pipeline"
    )
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--overwrite-embedding",
        action="store_true",
        help="Regenerate external-PLM artifacts, including stale ESM3 files",
    )
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)

    # 顶层 encoder_type 作为默认值, 每个模型可覆盖
    default_encoder_type = cfg.get("encoder_type")
    models = [parse_model_entry(n, e, default_encoder_type)
              for n, e in cfg["models"].items()]

    # 校验: 每个模型必须有 encoder_type
    missing = [m['name'] for m in models if not m['encoder_type']]
    if missing:
        raise ValueError(
            f"The following models have no encoder_type "
            f"(neither per-model nor global default):\n  " +
            "\n  ".join(missing) +
            f"\nFix: add `encoder_type:` to each model entry, "
            f"or set top-level `encoder_type:` as default."
        )

    log_dir = os.path.join("logs", f"tm_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    tracker = StatusTracker(log_dir)
    if args.reset:
        tracker.reset()

    eval_methods = cfg['eval'].get('methods', ['xgboost'])
    seeds = get_eval_seeds(cfg)

    # 统计 encoder 分布
    enc_counts = {}
    for m in models:
        enc_counts[m['encoder_type']] = enc_counts.get(m['encoder_type'], 0) + 1
    enc_summary = ", ".join(f"{k}:{v}" for k, v in sorted(enc_counts.items()))

    # ---- 打印计划 ----
    print("=" * 70)
    print(f" Tm Prediction Sweep  |  {datetime.now()}")
    print(f" Config:    {config_path}")
    print(f" Encoders:  {enc_summary}")
    print(f" Default:   {default_encoder_type or '(none)'}")
    print(f" Eval:      {', '.join(eval_methods)}")
    print(f" Seeds:     {seeds}")
    print(f" Logs:      {log_dir}")
    print("=" * 70)

    print(f"\nModels ({len(models)}):")
    for m in models:
        emb_dir = os.path.join(cfg['emb_root'], m['name'])
        ready, reason = check_model_embeddings(m, cfg)
        status = "✓ ready" if ready else "  new"
        ckpt = os.path.basename(m['checkpoint']) if m['checkpoint'] else "—"
        label = MODE_DESC.get(m['mode'], m['mode'])
        print(f"  {m['name']:40s}  [{m['source']:<12s}]  "
              f"[{m['encoder_type']:<32s}]  {m['mode']:8s} "
              f"({label:10s})  [{status}]  ckpt={ckpt}")
        if not ready and reason not in ("missing train.pt, val.pt, or test.pt", "missing train.pt"):
            print(f"    artifact status: {reason}")

    if args.dry_run:
        print(f"\n[DRY RUN] Done.")
        return

    # ---- Phase 1: Embedding ----
    successful = []

    if not args.skip_embedding:
        print(f"\n{'='*70}")
        print(f" Phase 1: Mean-Pooled Embedding Generation")
        print(f"{'='*70}")

        for i, m in enumerate(models, 1):
            name = m['name']
            print(f"\n[{i}/{len(models)}] {name} "
                  f"(encoder={m['encoder_type']}, mode={m['mode']})")

            if tracker.get(name) == "embedding_done":
                ready, reason = check_model_embeddings(m, cfg)
                if ready:
                    print(f"  → Already done, skipping")
                    successful.append(name)
                    continue
                print(f"  → Recorded as done but artifacts are not reusable: {reason}")

            if (m['source'] == 'enzsub' and m['checkpoint']
                    and not os.path.exists(m['checkpoint'])):
                print(f"  ✗ Checkpoint not found: {m['checkpoint']}")
                tracker.set(name, "ckpt_missing")
                continue

            if (m['source'] == 'enzsub' and m['mode'] in CHECKPOINT_MODES
                    and not m['checkpoint']):
                print(f"  ✗ mode={m['mode']} requires checkpoint")
                tracker.set(name, "config_error")
                continue

            ok = run_embedding(
                m,
                cfg,
                config_path,
                log_dir,
                overwrite_external=args.overwrite_embedding,
            )
            if ok:
                ready, reason = check_model_embeddings(m, cfg)
                if ready:
                    tracker.set(name, "embedding_done")
                    successful.append(name)
                else:
                    print(f"  ✗ Generated artifacts failed validation: {reason}")
                    tracker.set(name, "embedding_invalid")
            else:
                tracker.set(name, "embedding_failed")

        print(f"\nEmbedding: {len(successful)}/{len(models)} succeeded")
    else:
        for m in models:
            ready, reason = check_model_embeddings(m, cfg)
            if ready:
                successful.append(m['name'])
            else:
                print(f"  ⚠️  Embeddings not reusable for {m['name']}: {reason}")
        print(f"Found {len(successful)} existing embedding sets")

    # ---- Phase 2: 评估 ----
    if not args.skip_eval and successful:
        successful_models = [m for m in models if m['name'] in successful]

        if 'xgboost' in eval_methods:
            cv_folds = get_xgb_cv_folds(cfg)
            cv_msg = f", train-CV={cv_folds}-fold" if cv_folds and cv_folds > 1 else ""
            print(f"\n{'='*70}")
            print(f" Phase 2a: XGBoost ({len(successful_models)} models x {len(seeds)} seeds{cv_msg})")
            print(f"{'='*70}")
            for i, m in enumerate(successful_models, 1):
                print(f"\n[{i}/{len(successful_models)}] {m['name']}")
                for seed in seeds:
                    print(f"  seed={seed}")
                    run_xgboost(m, cfg, log_dir, seed)

        if 'knn' in eval_methods:
            print(f"\n{'='*70}")
            print(f" Phase 2b: k-NN ({len(successful_models)} models)")
            print(f"{'='*70}")
            for i, m in enumerate(successful_models, 1):
                print(f"\n[{i}/{len(successful_models)}] {m['name']}")
                run_knn(m, cfg, log_dir)

    # ---- Phase 3: 汇总 ----
    print_summary(cfg, models)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(cfg, models)
    md_path = os.path.join(log_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(buf.getvalue())

    print(f"\n{'='*70}")
    print(f" Sweep Complete: {datetime.now()}")
    print(f" Logs:    {log_dir}")
    print(f" Summary: {md_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
