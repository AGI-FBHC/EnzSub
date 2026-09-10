#!/usr/bin/env python3
"""Run the EC embedding and deterministic k-nearest-neighbour workflow."""

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

MODE_DESC = {
    "base":     "pretrained",
    "cpt":      "CPT",
    "base_sub": "base+SUB",
    "cpt_sub":  "CPT+SUB",
}

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

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

# ======== Test-set specification ========
def normalize_test_sets(raw_test_sets: dict, default_gallery_csv: str) -> dict:
    """
    把 YAML 里两种写法统一成:
        {name: {"query": csv, "gallery": csv, "exclude_self": bool}}
    """
    specs = {}
    for name, entry in raw_test_sets.items():
        if isinstance(entry, str):
            # 老式: name: path
            specs[name] = {
                "query": entry,
                "gallery": default_gallery_csv,
                "exclude_self": False,
            }
        elif isinstance(entry, dict):
            if "query" not in entry:
                raise ValueError(f"test_set '{name}' 缺少 'query' 字段")
            specs[name] = {
                "query": entry["query"],
                "gallery": entry.get("gallery", default_gallery_csv),
                "exclude_self": bool(entry.get("exclude_self", False)),
            }
        else:
            raise ValueError(f"test_set '{name}' 配置非法: {entry}")
    return specs

# ======== Embedding 生成 ========

def run_embedding(model_info: dict, cfg: dict) -> bool:
    name = model_info['name']
    mode = model_info['mode']
    emb_dir = os.path.join(cfg["emb_root"], name)

    sub_parent = cfg["sub_parent_dir"]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "generate_embeddings.py")

    cmd = [
        sys.executable, script,
        "--encoder-type", model_info['encoder_type'],
        "--model-mode", mode,
        "--fasta", *cfg["fasta_files"],
        "--output-dir", emb_dir,
        "--batch-size", str(cfg.get("batch_size", 8)),
        "--max-len", str(cfg.get("max_len", 1022)),
        "--device", cfg["device"],
    ]

    if model_info['checkpoint']:
        cmd += ["--checkpoint", model_info['checkpoint']]

    if mode in ("base_sub", "cpt_sub"):
        cmd += ["--lora-rank", str(model_info['lora_rank'])]
        cmd += ["--lora-alpha", str(model_info['lora_alpha'])]

    print(f"  CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = sub_parent + os.pathsep + env.get("PYTHONPATH", "")

    t0 = time.time()
    # stdout/stderr 继承父进程 → tqdm 进度条实时显示在控制台
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ✗ FAILED (code {proc.returncode})")
        return False

    n_files = len(list(Path(emb_dir).glob("*.pt")))
    print(f"  ✓ {n_files} embeddings, {elapsed:.0f}s")
    return True

# ======== k-NN 评估 ========

def run_knn(emb_names: list, cfg: dict) -> str:
    script = os.path.join(os.path.dirname(__file__), "knn_eval.py")
    knn_cfg = cfg["knn"]

    emb_dir_args = []
    for name in emb_names:
        emb_dir_args.append(f"{name}:{os.path.join(cfg['emb_root'], name)}")

    # Normalize test-set specifications before passing them to the evaluator.
    test_specs = normalize_test_sets(knn_cfg["test_sets"], knn_cfg["train_csv"])
    test_sets_json = json.dumps(test_specs)

    k_args = [str(k) for k in knn_cfg["k_values"]]

    # output_dir 直接用 config 值; knn_eval 内部会自动追加 run_<timestamp> 一层
    output_dir = knn_cfg.get("output_dir", "experiments/knn")

    cmd = [
        sys.executable, script,
        "--embedding-dirs", *emb_dir_args,
        "--train-csv", knn_cfg["train_csv"],
        "--test-sets-json", test_sets_json,
        "--k", *k_args,
        "--tau", str(knn_cfg.get("tau", 0.5)),
        "--output-dir", output_dir,
        "--save-predictions",
    ]

    print(f"  CMD: {' '.join(cmd[:6])} ... ({len(emb_dir_args)} models)")
    # 打印 test set 计划, 让自检配置一目了然
    for tn, sp in test_specs.items():
        flag = " [SELF-CHECK, exclude_self]" if sp["exclude_self"] else ""
        print(f"        test='{tn}'  gallery={os.path.basename(sp['gallery'])}{flag}")

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  ✗ k-NN FAILED (code {proc.returncode})")
        return None

    # knn_eval 在 output_dir 下新建了 run_<timestamp>/; 取最新的一个返回
    run_dirs = sorted(Path(output_dir).glob("run_*"),
                      key=lambda p: p.stat().st_mtime)
    actual_dir = str(run_dirs[-1]) if run_dirs else output_dir

    print(f"  ✓ k-NN done in {elapsed:.0f}s → {actual_dir}")
    return actual_dir

# ======== Markdown 汇总 ========

def print_markdown_summary(results_csv: str):
    import pandas as pd

    df = pd.read_csv(results_csv)
    print("\n## EC Classification Sweep Results (4-cell ablation)\n")

    for tname in df["test_set"].unique():
        sub = df[df["test_set"] == tname].copy()
        k1 = sub[sub["k"] == 1].sort_values("f1", ascending=False)

        # Record whether self-neighbours were excluded.
        is_self = bool(sub["exclude_self"].iloc[0]) if "exclude_self" in sub else False
        gal = sub["gallery"].iloc[0] if "gallery" in sub else "?"
        tag = "  _(leave-one-out self-check)_" if is_self else ""
        print(f"### {tname.upper()} (k=1, gallery={gal}){tag}\n")

        print(f"| Model | F1 | Precision | Recall | MCC | AUC | AUPRC |")
        print(f"|:------|---:|----------:|-------:|----:|----:|------:|")
        for _, row in k1.iterrows():
            print(f"| {row['embedding']} | "
                  f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                  f"{row['mcc']:.4f} | {row['auc']:.4f} | {row['auprc']:.4f} |")
        print()

        best = sub.loc[sub["f1"].idxmax()]
        print(f"**Best**: {best['embedding']} (k={int(best['k'])}) → "
              f"F1={best['f1']:.4f}\n")

# ======== 主函数 ========

def main():
    parser = argparse.ArgumentParser(
        description="EC Classification Checkpoint Sweep Pipeline"
    )
    parser.add_argument("--config", type=str, default="sweep_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-knn", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, config_path)
        if os.path.exists(candidate):
            config_path = candidate

    cfg = load_config(config_path)

    default_encoder = cfg.get('encoder_type', 'protbert_bfd')
    raw_models = cfg["models"]
    models = []
    for name, entry in raw_models.items():
        models.append(parse_model_entry(name, entry, default_encoder))

    print("=" * 70)
    print(f" EC Classification Ablation Sweep  |  {datetime.now()}")
    print(f" Config:  {config_path}")
    print(f" Encoder: {cfg.get('encoder_type', 'protbert_bfd')}")
    print("=" * 70)

    modes_present = {m['mode'] for m in models}
    print(f"\nAblation matrix coverage: {' / '.join(sorted(modes_present))}")
    missing = set(VALID_MODES) - modes_present
    if missing:
        print(f"  (not covered: {' / '.join(sorted(missing))})")

    print(f"\nModels ({len(models)}):")
    for m in models:
        emb_dir = os.path.join(cfg["emb_root"], m['name'])
        exists = "exists" if os.path.isdir(emb_dir) else "new"
        ckpt_info = os.path.basename(m['checkpoint']) if m['checkpoint'] else "—"
        mode_label = MODE_DESC.get(m['mode'], m['mode'])
        print(f"  {m['name']:40s}  enc={m['encoder_type']:14s}  "
              f"mode={m['mode']:8s} ({mode_label:10s})  "
              f"[{exists}]  ckpt={ckpt_info}")

    # Preview the resolved test-set plan.
    test_specs_preview = normalize_test_sets(cfg["knn"]["test_sets"],
                                             cfg["knn"]["train_csv"])
    print(f"\nDevice: {cfg['device']}, Batch: {cfg.get('batch_size', 8)}")
    print(f"k-NN: k={cfg['knn']['k_values']}, tau={cfg['knn'].get('tau', 0.5)}")
    print("Test sets:")
    for tn, sp in test_specs_preview.items():
        flag = "  [SELF-CHECK]" if sp["exclude_self"] else ""
        print(f"  {tn:20s} query={os.path.basename(sp['query']):20s} "
              f"gallery={os.path.basename(sp['gallery'])}{flag}")

    if args.dry_run:
        print("\n[DRY RUN] Done.")
        return

    # ---- Phase 1: Embedding ----
    successful = []

    if not args.skip_embedding:
        print(f"\n{'='*70}")
        print(" Phase 1: Embedding Generation")
        print(f"{'='*70}")

        for i, m in enumerate(models, 1):
            name = m['name']
            print(f"\n[{i}/{len(models)}] {name} (mode={m['mode']})")

            if m['checkpoint'] and not os.path.exists(m['checkpoint']):
                print(f"  ✗ Checkpoint not found: {m['checkpoint']}")
                continue

            if m['mode'] in CHECKPOINT_MODES and not m['checkpoint']:
                print(f"  ✗ mode={m['mode']} requires checkpoint")
                continue

            if run_embedding(m, cfg):
                successful.append(name)

        print(f"\nEmbedding: {len(successful)}/{len(models)} succeeded")
    else:
        for m in models:
            emb_dir = os.path.join(cfg["emb_root"], m['name'])
            if os.path.isdir(emb_dir) and len(list(Path(emb_dir).glob("*.pt"))) > 0:
                successful.append(m['name'])
            else:
                print(f"  WARNING: {m['name']} has no embeddings, skipping")
        print(f"Found {len(successful)} existing embedding sets")

    # ---- Phase 2: k-NN ----
    knn_output = None
    if not args.skip_knn and successful:
        print(f"\n{'='*70}")
        print(f" Phase 2: k-NN EC Classification ({len(successful)} models)")
        print(f"{'='*70}")

        knn_output = run_knn(successful, cfg)

        if knn_output:
            results_csv = os.path.join(knn_output, "results.csv")
            if os.path.exists(results_csv):
                print_markdown_summary(results_csv)

    print(f"\n{'='*70}")
    print(f" Pipeline Complete: {datetime.now()}")
    if knn_output:
        print(f" Results: {knn_output}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
