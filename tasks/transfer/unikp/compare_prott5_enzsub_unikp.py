#!/usr/bin/env python3
"""
Compare UniKP ProtT5 embeddings with EnzSub embeddings under identical settings.

The four comparisons are:
  1. UniKP baseline, 5-fold CV, ProtT5 protein embedding
  2. EnzSub-UniKP, 5-fold CV, EnzSub protein embedding
  3. UniKP baseline, holdout repeated N times, ProtT5 protein embedding
  4. EnzSub-UniKP, holdout repeated N times, EnzSub protein embedding

Everything other than the protein embedding is shared:
  - UniKP SMILES Transformer substrate embedding
  - ExtraTreesRegressor downstream model
  - labels, sample order, split seeds, and evaluation metrics
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluate_enzsub_in_unikp import (
    DEFAULT_ENZSUB_CODE_DIR,
    ROOT,
    enzsub_sequence_to_vec,
    evaluate_holdout,
    evaluate_kfold,
    load_task,
    smiles_to_vec,
)

def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _as_path(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None

def _cache_load(
    cache_path: Path | None,
    *,
    expected_items: Sequence[str],
    item_key: str,
    label: str,
    rebuild: bool,
) -> np.ndarray | None:
    if cache_path is None or rebuild or not cache_path.exists():
        return None

    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        cached_items = payload.get(item_key)
        cached_embeddings = payload.get("embeddings")
        if cached_items == list(expected_items) and cached_embeddings is not None:
            print(f"[cache] Loaded {label}: {cache_path}")
            return np.asarray(cached_embeddings, dtype=np.float32)

    print(f"[cache] Ignoring {label} cache because it does not match this dataset.")
    return None

def _cache_save(
    cache_path: Path | None,
    *,
    items: Sequence[str],
    item_key: str,
    embeddings: np.ndarray,
    label: str,
) -> None:
    if cache_path is None:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({item_key: list(items), "embeddings": embeddings}, f)
    print(f"[cache] Saved {label}: {cache_path}")

def load_or_build_smiles_embeddings(
    smiles: Sequence[str],
    *,
    cache_path: Path | None,
    rebuild: bool,
) -> np.ndarray:
    cached = _cache_load(
        cache_path,
        expected_items=smiles,
        item_key="smiles",
        label="UniKP SMILES embeddings",
        rebuild=rebuild,
    )
    if cached is not None:
        return cached

    print("[feature] Building UniKP SMILES embeddings")
    emb = np.asarray(smiles_to_vec(smiles), dtype=np.float32)
    print(f"[feature] UniKP SMILES shape: {emb.shape}")
    _cache_save(
        cache_path,
        items=smiles,
        item_key="smiles",
        embeddings=emb,
        label="UniKP SMILES embeddings",
    )
    return emb

def _preprocess_prott5_sequence(seq: str) -> str:
    seq = str(seq).upper().replace(" ", "").replace("\n", "")
    if len(seq) > 1000:
        seq = seq[:500] + seq[-500:]
    return seq

@torch.no_grad()
def prott5_sequence_to_vec(
    sequences: Sequence[str],
    *,
    model_path: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """UniKP-style ProtT5 sequence embedding with mean residue pooling."""
    from transformers import T5EncoderModel, T5Tokenizer

    processed = [_preprocess_prott5_sequence(seq) for seq in sequences]
    unique_sequences = list(dict.fromkeys(processed))
    total_batches = math.ceil(len(unique_sequences) / batch_size)

    print(
        f"[feature] ProtT5 unique sequences: {len(unique_sequences)}; "
        f"batch_size={batch_size}; batches={total_batches}"
    )
    print(
        f"[feature] Loading ProtT5 from {model_path!r}; "
        f"cache_dir={cache_dir or 'transformers_default'}; "
        f"local_files_only={local_files_only}"
    )
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            do_lower_case=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = T5EncoderModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        raise OSError(
            "Could not load ProtT5. Use --prot-t5-path with either a local "
            "model directory or the Hugging Face id 'Rostlab/prot_t5_xl_uniref50'. "
            "If the model is already in the HF cache and the server is offline, "
            "also pass --prot-t5-local-files-only and optionally --prot-t5-cache-dir."
        ) from exc
    model = model.to(device).eval()

    seq_to_vec: dict[str, np.ndarray] = {}
    for start in tqdm(
        range(0, len(unique_sequences), batch_size),
        total=total_batches,
        desc="[feature] ProtT5 embedding",
    ):
        chunk = unique_sequences[start:start + batch_size]
        spaced = [" ".join(seq) for seq in chunk]
        spaced = [re.sub(r"[UZOB]", "X", seq) for seq in spaced]

        ids = tokenizer.batch_encode_plus(
            spaced,
            add_special_tokens=True,
            padding=True,
        )
        input_ids = torch.tensor(ids["input_ids"], device=device)
        attention_mask = torch.tensor(ids["attention_mask"], device=device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state.detach().cpu().numpy()
        mask = attention_mask.detach().cpu().numpy()

        for seq, token_embeddings, token_mask in zip(chunk, hidden, mask):
            seq_len = int(token_mask.sum())
            residue_embeddings = token_embeddings[:seq_len - 1]
            seq_to_vec[seq] = residue_embeddings.mean(axis=0).astype(np.float32)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.stack([seq_to_vec[seq] for seq in processed]).astype(np.float32)

def load_or_build_prott5_embeddings(
    sequences: Sequence[str],
    *,
    cache_path: Path | None,
    rebuild: bool,
    model_path: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    batch_size: int,
) -> np.ndarray:
    cached = _cache_load(
        cache_path,
        expected_items=sequences,
        item_key="sequences",
        label="ProtT5 sequence embeddings",
        rebuild=rebuild,
    )
    if cached is not None:
        return cached

    print("[feature] Building ProtT5 protein embeddings")
    emb = prott5_sequence_to_vec(
        sequences,
        model_path=model_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        batch_size=batch_size,
    )
    print(f"[feature] ProtT5 sequence shape: {emb.shape}")
    _cache_save(
        cache_path,
        items=sequences,
        item_key="sequences",
        embeddings=emb,
        label="ProtT5 sequence embeddings",
    )
    return emb

def load_or_build_enzsub_embeddings(
    sequences: Sequence[str],
    args: argparse.Namespace,
    *,
    cache_path: Path | None,
) -> np.ndarray:
    cached = _cache_load(
        cache_path,
        expected_items=sequences,
        item_key="sequences",
        label="EnzSub sequence embeddings",
        rebuild=args.rebuild_cache,
    )
    if cached is not None:
        return cached

    if args.model_mode != "base" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --model-mode is not 'base'.")

    print("[feature] Building EnzSub protein embeddings")
    emb = enzsub_sequence_to_vec(
        sequences,
        enzsub_code_dir=Path(args.enzsub_code_dir),
        encoder_type=args.encoder_type,
        model_mode=args.model_mode,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.enzsub_batch_size,
        max_seq_len=args.max_seq_len,
        truncate_mode=args.truncate_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    print(f"[feature] EnzSub sequence shape: {emb.shape}")
    _cache_save(
        cache_path,
        items=sequences,
        item_key="sequences",
        embeddings=emb,
        label="EnzSub sequence embeddings",
    )
    return emb

def load_feature_matrix_cache(
    path: Path,
    expected_rows: int,
    *,
    label: str = "full feature matrix",
) -> np.ndarray:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "features" in payload:
        features = payload["features"]
    else:
        features = payload
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {features.shape[0]} rows, but this dataset has "
            f"{expected_rows} rows."
        )
    print(f"[cache] Loaded {label}: {path}")
    return features

def _split_label(train_ratio: float, n_runs: int) -> str:
    train_pct = int(round(train_ratio * 100))
    test_pct = int(round((1.0 - train_ratio) * 100))
    return f"{train_pct}/{test_pct} x{n_runs}"

def default_holdout_train_ratio(task: str) -> float:
    if task == "km":
        return 0.8
    return 0.9

def _add_eval_columns(
    df: pd.DataFrame,
    *,
    task: str,
    setting: str,
    split_name: str,
    protein_embedding: str,
    feature_dim: int,
) -> pd.DataFrame:
    out = df.copy()
    if "split" in out.columns:
        out = out.rename(columns={"split": "sample_split"})
    out.insert(0, "task", task)
    out.insert(1, "setting", setting)
    out.insert(2, "split", split_name)
    out.insert(3, "protein_embedding", protein_embedding)
    out.insert(4, "feature_dim", feature_dim)
    return out

def evaluate_feature_set(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    task: str,
    setting: str,
    protein_embedding: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(
        f"\n[eval] {setting} | {protein_embedding} | 5-fold CV "
        f"(feature_dim={features.shape[1]})"
    )
    cv_metrics, cv_preds = evaluate_kfold(
        features,
        labels,
        n_splits=args.kfold_splits,
        n_runs=args.kfold_runs,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    cv_metrics = _add_eval_columns(
        cv_metrics,
        task=task,
        setting=setting,
        split_name=f"{args.kfold_splits}-fold CV",
        protein_embedding=protein_embedding,
        feature_dim=features.shape[1],
    )
    cv_preds = _add_eval_columns(
        cv_preds,
        task=task,
        setting=setting,
        split_name=f"{args.kfold_splits}-fold CV",
        protein_embedding=protein_embedding,
        feature_dim=features.shape[1],
    )

    holdout_name = _split_label(args.holdout_train_ratio, args.holdout_runs)
    print(
        f"\n[eval] {setting} | {protein_embedding} | {holdout_name} "
        f"(feature_dim={features.shape[1]})"
    )
    holdout_metrics, holdout_preds = evaluate_holdout(
        features,
        labels,
        train_ratio=args.holdout_train_ratio,
        n_runs=args.holdout_runs,
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    holdout_metrics = _add_eval_columns(
        holdout_metrics,
        task=task,
        setting=setting,
        split_name=holdout_name,
        protein_embedding=protein_embedding,
        feature_dim=features.shape[1],
    )
    holdout_preds = _add_eval_columns(
        holdout_preds,
        task=task,
        setting=setting,
        split_name=holdout_name,
        protein_embedding=protein_embedding,
        feature_dim=features.shape[1],
    )

    metrics = pd.concat([cv_metrics, holdout_metrics], ignore_index=True)
    preds = pd.concat([cv_preds, holdout_preds], ignore_index=True)
    return metrics, preds

def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["r2", "pearson", "rmse", "mae"]
    group_cols = ["task", "setting", "split", "protein_embedding", "feature_dim"]
    summary = (
        metrics.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    counts = (
        metrics.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_eval_runs")
    )
    summary = summary.merge(counts, on=group_cols, how="left")
    return summary

def _format_metric(mean_value: float, std_value: float, n_runs: int) -> str:
    if pd.isna(std_value) or n_runs <= 1:
        return f"{mean_value:.6f}"
    return f"{mean_value:.6f} +/- {std_value:.6f}"

def make_paper_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_split = {"5-fold CV": 0}
    sort_setting = {"UniKP baseline": 0, "EnzSub-UniKP": 1}

    table = summary.copy()
    table["_split_order"] = table["split"].map(sort_split).fillna(1)
    table["_setting_order"] = table["setting"].map(sort_setting).fillna(99)
    table = table.sort_values(["_split_order", "_setting_order", "split"]).reset_index(drop=True)

    display_rows = []
    numeric_rows = []
    for _, row in table.iterrows():
        n_runs = int(row["n_eval_runs"])
        display_rows.append({
            "Setting": row["setting"],
            "Split": row["split"],
            "Protein embedding": row["protein_embedding"],
            "R2": _format_metric(row["r2_mean"], row["r2_std"], n_runs),
            "PCC": _format_metric(row["pearson_mean"], row["pearson_std"], n_runs),
            "RMSE": _format_metric(row["rmse_mean"], row["rmse_std"], n_runs),
            "MAE": _format_metric(row["mae_mean"], row["mae_std"], n_runs),
        })
        numeric_rows.append({
            "Task": row["task"],
            "Setting": row["setting"],
            "Split": row["split"],
            "Protein embedding": row["protein_embedding"],
            "R2_mean": row["r2_mean"],
            "R2_std": row["r2_std"],
            "PCC_mean": row["pearson_mean"],
            "PCC_std": row["pearson_std"],
            "RMSE_mean": row["rmse_mean"],
            "RMSE_std": row["rmse_std"],
            "MAE_mean": row["mae_mean"],
            "MAE_std": row["mae_std"],
            "n_eval_runs": n_runs,
        })

    return pd.DataFrame(display_rows), pd.DataFrame(numeric_rows)

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    values = [[str(value) for value in row] for row in df.to_numpy()]
    headers = [str(col) for col in df.columns]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in values)) if values else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in values)
    return "\n".join(lines) + "\n"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four-way UniKP ProtT5 vs EnzSub embedding comparisons."
    )
    parser.add_argument("--task", choices=["kcat", "km", "kcat_km"], default="kcat")
    parser.add_argument("--data-dir", default=str(ROOT / "datasets"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "embedding_comparison_outputs"),
    )

    parser.add_argument(
        "--prot-t5-path",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Local path or Hugging Face id for the ProtT5 model.",
    )
    parser.add_argument(
        "--prot-t5-cache-dir",
        default=None,
        help="Optional Hugging Face/Transformers cache directory for ProtT5.",
    )
    parser.add_argument(
        "--prot-t5-local-files-only",
        action="store_true",
        help="Load ProtT5 only from local files/cache; useful on offline servers.",
    )
    parser.add_argument("--prot-t5-batch-size", type=int, default=1)
    parser.add_argument(
        "--prott5-feature-cache",
        default=None,
        help="Optional full UniKP feature matrix cache, i.e. SMILES+ProtT5.",
    )
    parser.add_argument(
        "--enzsub-feature-cache",
        default=None,
        help=(
            "Optional full EnzSub-UniKP feature matrix cache, i.e. "
            "SMILES+EnzSub. When supplied, no EnzSub embedding is built."
        ),
    )

    parser.add_argument("--enzsub-code-dir", default=str(DEFAULT_ENZSUB_CODE_DIR))
    parser.add_argument("--encoder-type", default="esm2_3b")
    parser.add_argument(
        "--model-mode",
        choices=["base", "cpt", "base_sub", "cpt_sub"],
        default="cpt_sub",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--enzsub-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--truncate-mode", choices=["head", "head_tail"], default="head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    parser.add_argument("--smiles-cache", default=None)
    parser.add_argument("--prott5-cache", default=None)
    parser.add_argument("--enzsub-cache", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--save-features", action="store_true")

    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--kfold-splits", type=int, default=5)
    parser.add_argument("--kfold-runs", type=int, default=1)
    parser.add_argument(
        "--holdout-train-ratio",
        type=float,
        default=None,
        help="Default: 0.9 for kcat/kcat_km, 0.8 for km.",
    )
    parser.add_argument("--holdout-runs", type=int, default=5)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Default: do not set this parameter, matching ExtraTreesRegressor().",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.holdout_train_ratio is None:
        args.holdout_train_ratio = default_holdout_train_ratio(args.task)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prott5_feature_cache = _as_path(args.prott5_feature_cache)
    enzsub_feature_cache = _as_path(args.enzsub_feature_cache)
    prott5_cache = _as_path(args.prott5_cache) or out_dir / f"{args.task}_prott5_seq.pkl"
    enzsub_cache = (
        _as_path(args.enzsub_cache)
        or out_dir / f"{args.task}_enzsub_{args.model_mode}_seq.pkl"
    )

    print(f"[task] Loading {args.task} data")
    sequences, smiles, labels, metadata = load_task(args)
    labels = np.asarray(labels, dtype=float)
    print(f"[task] Samples: {len(labels)}")
    print(f"[task] Output dir: {out_dir}")
    print(f"[task] Device: {args.device}")

    if prott5_feature_cache is not None and enzsub_feature_cache is not None:
        smiles_vec = None
    else:
        smiles_cache = _as_path(args.smiles_cache) or out_dir / f"{args.task}_unikp_smiles.pkl"
        smiles_vec = load_or_build_smiles_embeddings(
            smiles,
            cache_path=smiles_cache,
            rebuild=args.rebuild_cache,
        )

    if prott5_feature_cache is not None:
        prott5_features = load_feature_matrix_cache(
            prott5_feature_cache,
            expected_rows=len(labels),
        )
    else:
        prott5_seq_vec = load_or_build_prott5_embeddings(
            sequences,
            cache_path=prott5_cache,
            rebuild=args.rebuild_cache,
            model_path=args.prot_t5_path,
            cache_dir=args.prot_t5_cache_dir,
            local_files_only=args.prot_t5_local_files_only,
            device=args.device,
            batch_size=args.prot_t5_batch_size,
        )
        prott5_features = np.concatenate((smiles_vec, prott5_seq_vec), axis=1)

    if enzsub_feature_cache is not None:
        enzsub_features = load_feature_matrix_cache(
            enzsub_feature_cache,
            expected_rows=len(labels),
            label="EnzSub-UniKP feature matrix",
        )
    else:
        enzsub_seq_vec = load_or_build_enzsub_embeddings(
            sequences,
            args,
            cache_path=enzsub_cache,
        )
        enzsub_features = np.concatenate((smiles_vec, enzsub_seq_vec), axis=1)

    print(f"[feature] ProtT5-UniKP fused shape: {prott5_features.shape}")
    print(f"[feature] EnzSub-UniKP fused shape: {enzsub_features.shape}")

    if args.save_features:
        prott5_feature_path = out_dir / f"{args.task}_features_unikp_prott5.pkl"
        enzsub_feature_path = out_dir / f"{args.task}_features_enzsub_{args.model_mode}.pkl"
        with open(prott5_feature_path, "wb") as f:
            pickle.dump(
                {
                    "features": prott5_features,
                    "labels": labels,
                    "metadata": metadata,
                    "args": vars(args),
                },
                f,
            )
        with open(enzsub_feature_path, "wb") as f:
            pickle.dump(
                {
                    "features": enzsub_features,
                    "labels": labels,
                    "metadata": metadata,
                    "args": vars(args),
                },
                f,
            )
        print(f"[feature] Saved: {prott5_feature_path}")
        print(f"[feature] Saved: {enzsub_feature_path}")

    prott5_metrics, prott5_preds = evaluate_feature_set(
        prott5_features,
        labels,
        task=args.task,
        setting="UniKP baseline",
        protein_embedding="ProtT5",
        args=args,
    )
    enzsub_metrics, enzsub_preds = evaluate_feature_set(
        enzsub_features,
        labels,
        task=args.task,
        setting="EnzSub-UniKP",
        protein_embedding="EnzSub",
        args=args,
    )

    metrics = pd.concat([prott5_metrics, enzsub_metrics], ignore_index=True)
    preds = pd.concat([prott5_preds, enzsub_preds], ignore_index=True)
    summary = summarize_metrics(metrics)
    paper_table, paper_table_numeric = make_paper_tables(summary)

    metrics_path = out_dir / f"{args.task}_comparison_metrics_all.csv"
    summary_path = out_dir / f"{args.task}_comparison_metrics_summary.csv"
    preds_path = out_dir / f"{args.task}_comparison_predictions.csv"
    paper_table_path = out_dir / f"{args.task}_paper_table.csv"
    paper_table_numeric_path = out_dir / f"{args.task}_paper_table_numeric.csv"
    paper_table_md_path = out_dir / f"{args.task}_paper_table.md"
    config_path = out_dir / f"{args.task}_comparison_run_config.json"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    preds.to_csv(preds_path, index=False)
    paper_table.to_csv(paper_table_path, index=False)
    paper_table_numeric.to_csv(paper_table_numeric_path, index=False)
    paper_table_md_path.write_text(dataframe_to_markdown(paper_table))
    with open(config_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "paper_alignment": {
                    "kcat_5fold": "UniKP Fig.2 reports 5-fold CV; this script uses KFold with fixed seed for reproducibility.",
                    "kcat_holdout": "UniKP Fig.3 uses 90/10 random split repeated 5 times; this script uses the same ratio and number of repeats by default.",
                    "km_holdout": "UniKP Km code uses an 80/20 train/test split; this script defaults to 80/20 for --task km.",
                    "random_seeds": "UniKP paper/code do not disclose fixed seeds; this script fixes seeds so ProtT5 and EnzSub use identical splits.",
                    "downstream_model": "ExtraTreesRegressor is used for both embeddings; n_estimators is left at sklearn default unless explicitly provided.",
                },
            },
            f,
            indent=2,
        )

    display_cols = [
        "setting",
        "split",
        "protein_embedding",
        "r2_mean",
        "pearson_mean",
        "rmse_mean",
        "mae_mean",
        "r2_std",
        "pearson_std",
        "rmse_std",
        "mae_std",
        "n_eval_runs",
    ]
    print("\n[summary]")
    print(summary[display_cols].to_string(index=False))
    print("\n[paper table]")
    print(paper_table.to_string(index=False))
    print(f"\n[saved] {metrics_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {preds_path}")
    print(f"[saved] {paper_table_path}")
    print(f"[saved] {paper_table_numeric_path}")
    print(f"[saved] {paper_table_md_path}")
    print(f"[saved] {config_path}")

if __name__ == "__main__":
    main()
