#!/usr/bin/env python3
"""Run multi-seed active-site embedding and probe evaluation."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")
CHECKPOINT_MODES = ("cpt", "base_sub", "cpt_sub")
VALID_PROBE_TYPES = ("mlp", "linear")

MODE_DESC = {
    "base":     "pretrained",
    "cpt":      "CPT",
    "base_sub": "base+SUB",
    "cpt_sub":  "CPT+SUB",
}

# 汇总用指标 (新增 auprc)
METRIC_KEYS = ['f1', 'mcc', 'roc_auc', 'auprc', 'precision', 'recall']

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

# ======== 配置校验 ========

def validate_seeds(cfg: dict) -> list:
    """从 cfg['eval'] 读取 seeds, 做校验并返回 list."""
    eval_cfg = cfg.get('eval', {})

    if 'seed' in eval_cfg and 'seeds' not in eval_cfg:
        raise ValueError(
            "配置里发现老字段 `eval.seed` (标量). 本版本只支持 `eval.seeds` (列表). "
            f"请改为:  seeds: [{eval_cfg['seed']}]"
        )

    if 'seeds' not in eval_cfg:
        raise ValueError("配置缺少 `eval.seeds` 字段, 请提供一个种子列表, 例如 seeds: [2025]")

    seeds = eval_cfg['seeds']
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError(f"`eval.seeds` 必须是非空列表, 当前: {seeds!r}")

    seeds = [int(s) for s in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"`eval.seeds` 有重复种子: {seeds}")

    return seeds

def get_probe_type(cfg: dict) -> str:
    """从 cfg['eval']['mlp']['model_type'] 读探针类型, 默认 mlp."""
    mlp_cfg = cfg.get('eval', {}).get('mlp', {})
    pt = str(mlp_cfg.get('model_type', 'mlp')).lower()
    if pt not in VALID_PROBE_TYPES:
        raise ValueError(
            f"eval.mlp.model_type 必须是 {VALID_PROBE_TYPES} 之一, 当前: {pt!r}")
    return pt

# ======== 模型条目解析 ========

def parse_model_entry(name: str, entry, default_encoder_type: str) -> dict:
    if entry is None:
        return {'name': name, 'mode': 'base', 'checkpoint': None,
                'lora_rank': 8, 'lora_alpha': 16,
                'encoder_type': default_encoder_type}

    if isinstance(entry, str):
        return {'name': name, 'mode': 'cpt', 'checkpoint': entry,
                'lora_rank': 8, 'lora_alpha': 16,
                'encoder_type': default_encoder_type}

    if isinstance(entry, dict):
        mode = entry.get('mode', 'base')
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}' for model '{name}'. "
                             f"Expected: {' / '.join(VALID_MODES)}")
        return {
            'name': name,
            'mode': mode,
            'checkpoint': entry.get('checkpoint'),
            'lora_rank': entry.get('lora_rank', 8),
            'lora_alpha': entry.get('lora_alpha', 16),
            'encoder_type': entry.get('encoder_type', default_encoder_type),
        }

    raise ValueError(f"Invalid model entry for {name}: {entry}")

# ======== 结果路径辅助 (按 probe_type 分目录) ========

def mlp_output_dir(cfg, probe_type, name, seed):
    return os.path.join(cfg['results_root'], probe_type, name, f"seed_{seed}")

def knn_output_dir(cfg, probe_type, name, seed):
    # k-NN 本身与探针类型无关, 但为目录整齐也放在 probe_type 下
    return os.path.join(cfg['results_root'], probe_type, "knn", name, f"seed_{seed}")

# ======== 状态追踪 (只管 Phase 1 embedding) ========

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
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.status, f, indent=2)

    def reset(self):
        self.status = {}
        self._save()

# ======== Phase 1: Embedding 生成 (种子/探针类型无关) ========

def count_pt_files(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith('.pt')])

def run_embedding(model_info: dict, cfg: dict, log_dir: str) -> bool:
    name = model_info['name']
    mode = model_info['mode']
    data = cfg['data']
    emb_root = cfg['emb_root']

    train_emb_dir = os.path.join(emb_root, name, "train")
    test_emb_dir = os.path.join(emb_root, name, "test")

    sub_parent = cfg["sub_parent_dir"]
    script = cfg["scripts"]["generate_embedding"]

    cmd_base = [
        sys.executable, script,
        "--encoder-type", model_info['encoder_type'],
        "--model-mode", mode,
        "--batch-size", str(cfg.get("batch_size", 32)),
        "--device", cfg["device"],
    ]

    if model_info['checkpoint']:
        cmd_base += ["--checkpoint", model_info['checkpoint']]

    if mode in ("base_sub", "cpt_sub"):
        cmd_base += ["--lora-rank", str(model_info['lora_rank'])]
        cmd_base += ["--lora-alpha", str(model_info['lora_alpha'])]

    env = os.environ.copy()
    env["PYTHONPATH"] = sub_parent + os.pathsep + env.get("PYTHONPATH", "")

    success = True

    for split, fasta_key, emb_dir in [
        ("train", "train_fasta", train_emb_dir),
        ("test",  "test_fasta",  test_emb_dir),
    ]:
        existing = count_pt_files(emb_dir)
        if existing > 0:
            print(f"    {split}: {existing} files exist, skipping")
            continue

        fasta_path = data[fasta_key]
        cmd = cmd_base + ["--fasta", fasta_path, "--output-dir", emb_dir]

        log_file = os.path.join(log_dir, f"emb_{name}_{split}.log")
        print(f"    {split}: generating...")

        t0 = time.time()
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"    ✗ {split} FAILED (code {proc.returncode}), see {log_file}")
            with open(log_file) as lf:
                for line in lf.readlines()[-5:]:
                    print(f"      {line.rstrip()}")
            success = False
            break

        n = count_pt_files(emb_dir)
        print(f"    ✓ {split}: {n} files, {elapsed:.0f}s")

    return success

# ======== Phase 2: 探针评估 (单种子, mlp 或 linear) ========

def run_probe_eval(model_info: dict, cfg: dict, seed: int, probe_type: str,
                   log_dir: str) -> bool:
    """运行探针 5-fold CV 评估 (单种子). probe_type ∈ {mlp, linear}."""
    name = model_info['name']
    data = cfg['data']
    eval_cfg = cfg['eval']['mlp']     # 探针超参仍读 eval.mlp 段
    encoder_type = model_info['encoder_type']

    train_emb_dir = os.path.join(cfg['emb_root'], name, "train")
    test_emb_dir = os.path.join(cfg['emb_root'], name, "test")
    output_dir = mlp_output_dir(cfg, probe_type, name, seed)

    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        _print_existing_results(results_file, f"{probe_type}[seed={seed}]")
        return True

    script = cfg["scripts"]["train_kfold"]

    # linear 探针默认更激进的 lr / 更长 epoch (若 config 未显式给出)
    default_lr = 5e-3 if probe_type == "linear" else 1e-3
    default_epochs = 150 if probe_type == "linear" else 100
    default_patience = 25 if probe_type == "linear" else 15

    cmd = [
        sys.executable, script,
        "--train-data-file", data["train_labels"],
        "--train-emb-dir", train_emb_dir,
        "--test-data-file", data["test_labels"],
        "--test-emb-dir", test_emb_dir,
        "--model-name", name,
        "--encoder-type", encoder_type,
        "--output-dir", output_dir,
        "--model-type", probe_type,
        "--loss-type", eval_cfg.get("loss_type", "focal"),
        "--focal-alpha", str(eval_cfg.get("focal_alpha", 0.65)),
        "--focal-gamma", str(eval_cfg.get("focal_gamma", 2.0)),
        "--seed", str(seed),
        "--n-splits", str(eval_cfg.get("n_splits", 5)),
        "--lr", str(eval_cfg.get("lr", default_lr)),
        "--weight-decay", str(eval_cfg.get("weight_decay", 1e-4)),
        "--batch-size", str(eval_cfg.get("batch_size", 16)),
        "--dropout", str(eval_cfg.get("dropout", 0.4)),
        "--epochs", str(eval_cfg.get("epochs", default_epochs)),
        "--patience", str(eval_cfg.get("patience", default_patience)),
        "--device", cfg["device"],
        "--pin-memory",
    ]

    log_file = os.path.join(log_dir, f"{probe_type}_{name}_seed{seed}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ {probe_type}[seed={seed}] FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-8:]:
                print(f"      {line.rstrip()}")
        return False

    print(f"    ✓ {probe_type}[seed={seed}] done in {elapsed:.0f}s")
    _print_existing_results(results_file, f"{probe_type}[seed={seed}]")
    return True

# ======== Phase 2: k-NN 评估 (单种子) ========

def run_knn_eval(model_info: dict, cfg: dict, seed: int, probe_type: str,
                 log_dir: str) -> bool:
    name = model_info['name']
    data = cfg['data']
    knn_cfg = cfg['eval']['knn']
    encoder_type = model_info['encoder_type']

    train_emb_dir = os.path.join(cfg['emb_root'], name, "train")
    test_emb_dir = os.path.join(cfg['emb_root'], name, "test")
    output_dir = knn_output_dir(cfg, probe_type, name, seed)

    results_file = os.path.join(output_dir, "knn_results.json")
    if os.path.exists(results_file):
        _print_existing_knn_results(results_file, seed)
        return True

    script = cfg["scripts"]["knn_active_site"]
    ks = knn_cfg.get("ks", [1, 3, 5, 10, 20])

    cmd = [
        sys.executable, script,
        "--train-data-file", data["train_labels"],
        "--train-emb-dir", train_emb_dir,
        "--test-data-file", data["test_labels"],
        "--test-emb-dir", test_emb_dir,
        "--model-name", name,
        "--encoder-type", encoder_type,
        "--output-dir", output_dir,
        "--ks", *[str(k) for k in ks],
        "--seed", str(seed),
        "--device", cfg["device"],
    ]

    log_file = os.path.join(log_dir, f"knn_{name}_seed{seed}.log")
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    ✗ k-NN[seed={seed}] FAILED, see {log_file}")
        with open(log_file) as lf:
            for line in lf.readlines()[-8:]:
                print(f"      {line.rstrip()}")
        return False

    print(f"    ✓ k-NN[seed={seed}] done in {elapsed:.0f}s")
    _print_existing_knn_results(results_file, seed)
    return True

# ======== 单种子结果打印 ========

def _print_existing_results(results_file, tag=""):
    try:
        with open(results_file) as f:
            r = json.load(f)
        t = r.get("test_metrics", {})
        cv = r.get("cv_metrics", {})
        if cv:
            cv_f1 = cv.get('f1', {})
            cv_auprc = cv.get('auprc', {})
            print(f"    {tag} CV:   F1={cv_f1.get('mean',0):.4f}±{cv_f1.get('std',0):.4f}  "
                  f"AUPRC={cv_auprc.get('mean',0):.4f}±{cv_auprc.get('std',0):.4f}")
        print(f"    {tag} Test: F1={t.get('f1', 0):.4f}  "
              f"AUPRC={t.get('auprc', 0):.4f}  "
              f"AUROC={t.get('roc_auc', 0):.4f}")
    except Exception:
        pass

def _print_existing_knn_results(results_file, seed):
    try:
        with open(results_file) as f:
            r = json.load(f)
        t = r.get("best_test_metrics", {})
        best_k = r.get("best_k", "?")
        print(f"    k-NN[seed={seed}] (per-seed best_k={best_k}) "
              f"F1={t.get('f1', 0):.4f}  "
              f"AUPRC={t.get('auprc', 0):.4f}  "
              f"AUROC={t.get('roc_auc', 0):.4f}")
    except Exception:
        pass

# ======== Phase 3: 跨种子聚合 ========

def _agg(values):
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())

def collect_probe_results(cfg, probe_type, model_name, seeds):
    results_root = cfg['results_root']

    per_seed = []
    for seed in seeds:
        rf = os.path.join(results_root, probe_type, model_name,
                          f"seed_{seed}", "results.json")
        if not os.path.exists(rf):
            continue
        with open(rf) as f:
            r = json.load(f)

        entry = {'seed': seed, 'test': {}, 'cv_mean': {}, 'cv_std_within_seed': {}}

        t = r.get('test_metrics', {})
        for k in METRIC_KEYS:
            if k in t:
                entry['test'][k] = float(t[k])

        cv = r.get('cv_metrics', {})
        for k in METRIC_KEYS:
            if k in cv:
                entry['cv_mean'][k] = float(cv[k].get('mean', 0))
                entry['cv_std_within_seed'][k] = float(cv[k].get('std', 0))

        per_seed.append(entry)

    agg = {'test': {}, 'cv': {}}
    for k in METRIC_KEYS:
        test_vals = [e['test'][k] for e in per_seed if k in e['test']]
        cv_vals = [e['cv_mean'][k] for e in per_seed if k in e['cv_mean']]
        agg['test'][k] = _agg(test_vals)
        agg['cv'][k] = _agg(cv_vals)

    return {'per_seed': per_seed, 'aggregated': agg}

def collect_knn_results(cfg, probe_type, model_name, seeds, ks):
    results_root = cfg['results_root']

    per_seed = []
    for seed in seeds:
        rf = os.path.join(results_root, probe_type, "knn", model_name,
                          f"seed_{seed}", "knn_results.json")
        if not os.path.exists(rf):
            continue
        with open(rf) as f:
            r = json.load(f)

        entry = {
            'seed': seed,
            'per_seed_best_k': r.get('best_k'),
            'per_seed_best_test': r.get('best_test_metrics', {}),
            'all_k_test': {},
        }

        all_k = r.get('all_k_results', {})
        for _, kv in all_k.items():
            k_val = int(kv.get('k'))
            tm = kv.get('test_metrics', {})
            entry['all_k_test'][k_val] = {mk: float(tm.get(mk, 0)) for mk in METRIC_KEYS}

        per_seed.append(entry)

    if not per_seed:
        return {'per_seed': [], 'aggregated': None}

    k_sets = [set(e['all_k_test'].keys()) for e in per_seed]
    common_ks = sorted(set.intersection(*k_sets)) if k_sets else []
    if not common_ks:
        return {'per_seed': per_seed, 'aggregated': None}

    mean_f1_by_k = {}
    for k_val in common_ks:
        f1s = [e['all_k_test'][k_val]['f1'] for e in per_seed]
        mean_f1_by_k[k_val] = float(np.mean(f1s))

    best_k = max(common_ks, key=lambda k: mean_f1_by_k[k])

    agg_metrics = {}
    for mk in METRIC_KEYS:
        vals = [e['all_k_test'][best_k][mk] for e in per_seed]
        agg_metrics[mk] = _agg(vals)

    return {
        'per_seed': per_seed,
        'aggregated': {
            'cross_seed_best_k': int(best_k),
            'metrics': agg_metrics,
            'mean_f1_by_k': mean_f1_by_k,
        }
    }

# ======== Phase 3: 打印 & CSV ========

def _fmt_meanstd(pair, decimals=4):
    if pair is None or pair[0] is None:
        return "—"
    mean, std = pair
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"

def print_summary(cfg: dict, models: list, seeds: list, probe_type: str):
    methods = cfg['eval'].get('methods', ['mlp'])
    results_root = cfg['results_root']
    knn_ks = cfg['eval'].get('knn', {}).get('ks', [1, 3, 5, 10, 20])

    n_seeds = len(seeds)
    print(f"\n{'='*70}")
    print(f" Results Summary  |  probe={probe_type}  |  {n_seeds} seed(s): {seeds}")
    print(f"{'='*70}")

    model_results = []
    for m in models:
        item = {'name': m['name'], 'mode': m['mode'],
                'encoder': m['encoder_type']}
        if 'mlp' in methods:
            item['probe'] = collect_probe_results(cfg, probe_type, m['name'], seeds)
        if 'knn' in methods:
            item['knn'] = collect_knn_results(cfg, probe_type, m['name'], seeds, knn_ks)
        model_results.append(item)

    # ---- 探针表 (跨种子 mean ± std, 含 AUPRC) ----
    if 'mlp' in methods:
        probe_rows = [r for r in model_results
                      if 'probe' in r and r['probe']['per_seed']]
        if probe_rows:
            def _sort_key(r):
                cv_auprc = r['probe']['aggregated']['cv'].get('auprc')
                if cv_auprc and cv_auprc[0] is not None:
                    return -cv_auprc[0]
                test_auprc = r['probe']['aggregated']['test'].get('auprc')
                return -(test_auprc[0] if test_auprc and test_auprc[0] is not None else 0)
            probe_rows.sort(key=_sort_key)

            print(f"\n### {probe_type.upper()} probe 5-Fold CV  "
                  f"(n={n_seeds} seeds, sorted by cross-seed CV AUPRC)\n")
            print(f"| {'Model':<38s} | {'Enc':<12s} | {'Mode':<8s} | "
                  f"{'CV AUPRC':>16s} | {'CV F1':>16s} | "
                  f"{'Test AUPRC':>16s} | {'Test F1':>16s} | {'Test AUROC':>16s} |")
            print(f"|{'-'*40}|{'-'*14}|{'-'*10}|"
                  f"{'-'*18}|{'-'*18}|{'-'*18}|{'-'*18}|{'-'*18}|")

            for r in probe_rows:
                agg = r['probe']['aggregated']
                n_found = len(r['probe']['per_seed'])
                name_col = r['name']
                if n_found < n_seeds:
                    name_col = f"{r['name']} ({n_found}/{n_seeds})"
                print(f"| {name_col:<38s} | {r['encoder']:<12s} | {r['mode']:<8s} | "
                      f"{_fmt_meanstd(agg['cv'].get('auprc')):>16s} | "
                      f"{_fmt_meanstd(agg['cv'].get('f1')):>16s} | "
                      f"{_fmt_meanstd(agg['test'].get('auprc')):>16s} | "
                      f"{_fmt_meanstd(agg['test'].get('f1')):>16s} | "
                      f"{_fmt_meanstd(agg['test'].get('roc_auc')):>16s} |")

            best = probe_rows[0]
            bagg = best['probe']['aggregated']
            print(f"\n🏆 Best {probe_type} (by cross-seed CV AUPRC): {best['name']}")
            print(f"   CV   AUPRC={_fmt_meanstd(bagg['cv'].get('auprc'))}   "
                  f"F1={_fmt_meanstd(bagg['cv'].get('f1'))}")
            print(f"   Test AUPRC={_fmt_meanstd(bagg['test'].get('auprc'))}   "
                  f"F1={_fmt_meanstd(bagg['test'].get('f1'))}   "
                  f"AUROC={_fmt_meanstd(bagg['test'].get('roc_auc'))}")

    # ---- k-NN 表 ----
    if 'knn' in methods:
        knn_rows = [r for r in model_results
                    if 'knn' in r and r['knn']['aggregated'] is not None]
        if knn_rows:
            def _knn_key(r):
                ap = r['knn']['aggregated']['metrics'].get('auprc')
                return -(ap[0] if ap and ap[0] is not None else 0)
            knn_rows.sort(key=_knn_key)

            print(f"\n### k-NN  (n={n_seeds} seeds, best k by cross-seed mean F1)\n")
            print(f"| {'Model':<38s} | {'Enc':<12s} | {'Mode':<8s} | {'k':>4s} | "
                  f"{'AUPRC':>16s} | {'F1':>16s} | {'AUROC':>16s} |")
            print(f"|{'-'*40}|{'-'*14}|{'-'*10}|{'-'*6}|{'-'*18}|{'-'*18}|{'-'*18}|")

            for r in knn_rows:
                agg = r['knn']['aggregated']
                n_found = len(r['knn']['per_seed'])
                name_col = r['name']
                if n_found < n_seeds:
                    name_col = f"{r['name']} ({n_found}/{n_seeds})"
                print(f"| {name_col:<38s} | {r['encoder']:<12s} | {r['mode']:<8s} | "
                      f"{agg['cross_seed_best_k']:>4d} | "
                      f"{_fmt_meanstd(agg['metrics'].get('auprc')):>16s} | "
                      f"{_fmt_meanstd(agg['metrics'].get('f1')):>16s} | "
                      f"{_fmt_meanstd(agg['metrics'].get('roc_auc')):>16s} |")

            best = knn_rows[0]
            bagg = best['knn']['aggregated']
            print(f"\n🏆 Best k-NN: {best['name']}  k={bagg['cross_seed_best_k']}  "
                  f"AUPRC={_fmt_meanstd(bagg['metrics'].get('auprc'))}")

    # ---- CSV ----
    csv_path = os.path.join(results_root, f"sweep_summary_{probe_type}.csv")
    _write_per_seed_csv(csv_path, model_results, methods, knn_ks)
    print(f"\nPer-seed CSV: {csv_path}")

    agg_csv_path = os.path.join(results_root, f"sweep_summary_{probe_type}_aggregated.csv")
    _write_aggregated_csv(agg_csv_path, model_results, methods, n_seeds)
    print(f"Aggregated CSV: {agg_csv_path}")

def _write_per_seed_csv(csv_path, model_results, methods, knn_ks):
    import csv

    rows = []
    all_seeds = []
    for r in model_results:
        for src in ('probe', 'knn'):
            if src in r:
                for e in r[src]['per_seed']:
                    if e['seed'] not in all_seeds:
                        all_seeds.append(e['seed'])

    for r in model_results:
        probe_by_seed = {e['seed']: e for e in r.get('probe', {}).get('per_seed', [])}
        knn_by_seed = {e['seed']: e for e in r.get('knn', {}).get('per_seed', [])}

        for seed in all_seeds:
            if seed not in probe_by_seed and seed not in knn_by_seed:
                continue
            row = {'model': r['name'], 'encoder': r['encoder'],
                   'mode': r['mode'], 'seed': seed}

            if 'mlp' in methods and seed in probe_by_seed:
                e = probe_by_seed[seed]
                for mk in METRIC_KEYS:
                    if mk in e['test']:
                        row[f'probe_test_{mk}'] = e['test'][mk]
                    if mk in e['cv_mean']:
                        row[f'probe_cv_{mk}_mean'] = e['cv_mean'][mk]
                        row[f'probe_cv_{mk}_std'] = e['cv_std_within_seed'].get(mk, '')

            if 'knn' in methods and seed in knn_by_seed:
                e = knn_by_seed[seed]
                row['knn_per_seed_best_k'] = e.get('per_seed_best_k', '')
                pb = e.get('per_seed_best_test', {})
                for mk in METRIC_KEYS:
                    if mk in pb:
                        row[f'knn_best_{mk}'] = pb[mk]
                for k_val in sorted(e.get('all_k_test', {}).keys()):
                    for mk in METRIC_KEYS:
                        row[f'knn_k{k_val}_{mk}'] = e['all_k_test'][k_val].get(mk, '')

            rows.append(row)

    if not rows:
        return

    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _write_aggregated_csv(csv_path, model_results, methods, n_seeds):
    import csv

    rows = []
    for r in model_results:
        row = {'model': r['name'], 'encoder': r['encoder'],
               'mode': r['mode'], 'n_seeds': n_seeds}
        if 'mlp' in methods and r.get('probe', {}).get('per_seed'):
            row['probe_n_found'] = len(r['probe']['per_seed'])
            agg = r['probe']['aggregated']
            for mk in METRIC_KEYS:
                if agg['test'].get(mk) and agg['test'][mk][0] is not None:
                    row[f'probe_test_{mk}_mean'] = agg['test'][mk][0]
                    row[f'probe_test_{mk}_std'] = agg['test'][mk][1]
                if agg['cv'].get(mk) and agg['cv'][mk][0] is not None:
                    row[f'probe_cv_{mk}_mean'] = agg['cv'][mk][0]
                    row[f'probe_cv_{mk}_std'] = agg['cv'][mk][1]

        if 'knn' in methods and r.get('knn', {}).get('aggregated'):
            row['knn_n_found'] = len(r['knn']['per_seed'])
            kagg = r['knn']['aggregated']
            row['knn_cross_seed_best_k'] = kagg['cross_seed_best_k']
            for mk in METRIC_KEYS:
                pair = kagg['metrics'].get(mk)
                if pair and pair[0] is not None:
                    row[f'knn_{mk}_mean'] = pair[0]
                    row[f'knn_{mk}_std'] = pair[1]

        rows.append(row)

    if not rows:
        return

    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

# ======== 主函数 ========

def main():
    parser = argparse.ArgumentParser(
        description="Active Site Prediction Sweep Pipeline (multi-seed, mlp/linear)"
    )
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)
    seeds = validate_seeds(cfg)
    probe_type = get_probe_type(cfg)

    default_encoder = cfg.get('encoder_type', 'esm2_650m')
    models = []
    for name, entry in cfg["models"].items():
        models.append(parse_model_entry(name, entry, default_encoder))

    log_dir = os.path.join("logs", f"sweep_{probe_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)

    tracker = StatusTracker(log_dir)
    if args.reset:
        tracker.reset()

    eval_methods = cfg['eval'].get('methods', ['mlp'])

    print("=" * 70)
    print(f" Active Site Prediction Sweep  |  {datetime.now()}")
    print(f" Config:      {config_path}")
    print(f" Encoder:     {cfg.get('encoder_type', '?')}")
    print(f" Probe type:  {probe_type}")
    print(f" Eval:        {', '.join(eval_methods)}")
    print(f" Seeds:       {seeds}  (n={len(seeds)})")
    print(f" Logs:        {log_dir}")
    print("=" * 70)

    modes_present = {m['mode'] for m in models}
    print(f"\nAblation coverage: {' / '.join(sorted(modes_present))}")
    missing = set(VALID_MODES) - modes_present
    if missing:
        print(f"  (not covered: {' / '.join(sorted(missing))})")

    n_eval_tasks = len(models) * len(seeds) * len(eval_methods)
    print(f"\nPhase 2 total runs: {len(models)} models × {len(seeds)} seeds "
          f"× {len(eval_methods)} method(s) = {n_eval_tasks}")

    print(f"\nModels ({len(models)}):")
    for m in models:
        train_n = count_pt_files(os.path.join(cfg['emb_root'], m['name'], "train"))
        test_n = count_pt_files(os.path.join(cfg['emb_root'], m['name'], "test"))
        emb_status = f"train={train_n} test={test_n}" if train_n or test_n else "new"
        state = tracker.get(m['name'])
        ckpt_info = os.path.basename(m['checkpoint']) if m['checkpoint'] else "—"
        mode_label = MODE_DESC.get(m['mode'], m['mode'])
        print(f"  {m['name']:40s}  enc={m['encoder_type']:14s}  "
              f"{m['mode']:8s} ({mode_label:10s})  "
              f"[{emb_status}] ({state})  ckpt={ckpt_info}")

    if args.dry_run:
        print(f"\n[DRY RUN] Done.")
        return

    # ---- Phase 1: Embedding ----
    successful = []

    if not args.skip_embedding:
        print(f"\n{'='*70}")
        print(f" Phase 1: Per-Residue Embedding Generation (seed/probe-independent)")
        print(f"{'='*70}")

        for i, m in enumerate(models, 1):
            name = m['name']
            print(f"\n[{i}/{len(models)}] {name} (mode={m['mode']})")

            if tracker.get(name) == "embedding_done":
                print(f"  → Already done, skipping")
                successful.append(name)
                continue

            if m['checkpoint'] and not os.path.exists(m['checkpoint']):
                print(f"  ✗ Checkpoint not found: {m['checkpoint']}")
                tracker.set(name, "ckpt_missing")
                continue

            if m['mode'] in CHECKPOINT_MODES and not m['checkpoint']:
                print(f"  ✗ mode={m['mode']} requires checkpoint")
                tracker.set(name, "config_error")
                continue

            ok = run_embedding(m, cfg, log_dir)
            if ok:
                tracker.set(name, "embedding_done")
                successful.append(name)
            else:
                tracker.set(name, "embedding_failed")

        print(f"\nEmbedding: {len(successful)}/{len(models)} succeeded")
    else:
        for m in models:
            emb_dir = os.path.join(cfg['emb_root'], m['name'])
            train_n = count_pt_files(os.path.join(emb_dir, "train"))
            test_n = count_pt_files(os.path.join(emb_dir, "test"))
            if train_n > 0 and test_n > 0:
                successful.append(m['name'])
            else:
                print(f"  ⚠️  Missing embeddings for {m['name']}, skipping")
        print(f"Found {len(successful)} existing embedding sets")

    # ---- Phase 2: 评估 ----
    if not args.skip_eval and successful:
        successful_models = [m for m in models if m['name'] in successful]
        total_models = len(successful_models)

        if 'mlp' in eval_methods:
            print(f"\n{'='*70}")
            print(f" Phase 2a: {probe_type.upper()} probe 5-Fold CV  "
                  f"({total_models} models × {len(seeds)} seeds = "
                  f"{total_models * len(seeds)} runs)")
            print(f"{'='*70}")

            for i, m in enumerate(successful_models, 1):
                print(f"\n[Model {i}/{total_models}] {m['name']}")
                for j, seed in enumerate(seeds, 1):
                    print(f"  [seed {j}/{len(seeds)}] seed={seed}")
                    run_probe_eval(m, cfg, seed, probe_type, log_dir)

        if 'knn' in eval_methods:
            print(f"\n{'='*70}")
            print(f" Phase 2b: k-NN Evaluation  "
                  f"({total_models} models × {len(seeds)} seeds = "
                  f"{total_models * len(seeds)} runs)")
            print(f"{'='*70}")

            for i, m in enumerate(successful_models, 1):
                print(f"\n[Model {i}/{total_models}] {m['name']}")
                for j, seed in enumerate(seeds, 1):
                    print(f"  [seed {j}/{len(seeds)}] seed={seed}")
                    run_knn_eval(m, cfg, seed, probe_type, log_dir)

    # ---- Phase 3: 汇总 ----
    print_summary(cfg, models, seeds, probe_type)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(cfg, models, seeds, probe_type)
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
