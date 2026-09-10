#!/usr/bin/env python3
"""
Evaluate EnzSub enzyme embeddings inside the UniKP feature pipeline.

This script intentionally leaves the original UniKP scripts untouched. It keeps:
  - UniKP SMILES Transformer substrate embeddings
  - UniKP-style ExtraTreesRegressor downstream model
  - UniKP label transforms and common train/eval splits

Only the protein embedding is replaced with an EnzSub downstream encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from build_vocab import TorchVocab, Vocab, WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split

ROOT = Path(__file__).resolve().parent

def _default_enzsub_code_dir() -> Path:
    return ROOT.parents[2] / "src"

DEFAULT_ENZSUB_CODE_DIR = _default_enzsub_code_dir()

def _register_vocab_pickle_aliases() -> None:
    """
    UniKP's vocab.pkl may have been created by running build_vocab.py as a
    script, which stores classes as __main__.WordVocab. When this file is
    imported by another entrypoint, __main__ is that entrypoint instead.
    """
    import __main__

    for name, cls in {
        "TorchVocab": TorchVocab,
        "Vocab": Vocab,
        "WordVocab": WordVocab,
    }.items():
        if not hasattr(__main__, name):
            setattr(__main__, name, cls)

def smiles_to_vec(smiles: Sequence[str]) -> np.ndarray:
    """UniKP SMILES Transformer embedding, copied without changing the method."""
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3

    _register_vocab_pickle_aliases()
    vocab = WordVocab.load_vocab(str(ROOT / "vocab.pkl"))

    def get_inputs(sm: str) -> Tuple[List[int], List[int]]:
        seq_len = 220
        tokens = sm.split()
        if len(tokens) > 218:
            tokens = tokens[:109] + tokens[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in tokens]
        ids = [sos_index] + ids + [eos_index]
        seg = [1] * len(ids)
        padding = [pad_index] * (seq_len - len(ids))
        ids.extend(padding)
        seg.extend(padding)
        return ids, seg

    def get_array(split_smiles: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_id, x_seg = [], []
        for sm in split_smiles:
            ids, seg = get_inputs(sm)
            x_id.append(ids)
            x_seg.append(seg)
        return torch.tensor(x_id), torch.tensor(x_seg)

    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state = torch.load(str(ROOT / "trfm_12_23000.pkl"), map_location="cpu")
    trfm.load_state_dict(state)
    trfm.eval()

    x_split = [split(str(sm)) for sm in smiles]
    x_id, _ = get_array(x_split)
    return trfm.encode(torch.t(x_id))

def _preprocess_sequence(seq: str, max_len: int | None, truncate_mode: str) -> str:
    seq = str(seq).upper().replace(" ", "").replace("\n", "")
    if max_len is None or len(seq) <= max_len:
        return seq
    if truncate_mode == "head":
        return seq[:max_len]
    if truncate_mode == "head_tail":
        left = max_len // 2
        right = max_len - left
        return seq[:left] + seq[-right:]
    raise ValueError(f"Unknown truncate_mode: {truncate_mode}")

@torch.no_grad()
def enzsub_sequence_to_vec(
    sequences: Sequence[str],
    *,
    enzsub_code_dir: Path,
    encoder_type: str,
    model_mode: str,
    checkpoint: str | None,
    device: str,
    batch_size: int,
    max_seq_len: int | None,
    truncate_mode: str,
    lora_rank: int,
    lora_alpha: int,
) -> np.ndarray:
    """Generate EnzSub enzyme embeddings, deduplicating repeated sequences."""
    sys.path.insert(0, str(enzsub_code_dir.resolve()))
    from enzsub.sub.model import EnzSubModelForDownstream

    if model_mode != "base" and not checkpoint:
        raise ValueError(f"--checkpoint is required when --model-mode={model_mode!r}")

    encoder = EnzSubModelForDownstream(
        encoder_type=encoder_type,
        model_mode=model_mode,
        checkpoint_path=None if model_mode == "base" else checkpoint,
        freeze_backbone=True,
        device=device,
        strict_lora_load=True,
        allow_lora_reverse_detect=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    effective_max = max_seq_len
    if effective_max is None and hasattr(encoder, "max_seq_len"):
        effective_max = encoder.max_seq_len

    processed = [
        _preprocess_sequence(seq, effective_max, truncate_mode)
        for seq in sequences
    ]

    unique_sequences = list(dict.fromkeys(processed))
    seq_to_vec = {}
    total_batches = math.ceil(len(unique_sequences) / batch_size)
    print(
        f"[feature] EnzSub unique sequences: {len(unique_sequences)}; "
        f"batch_size={batch_size}; batches={total_batches}"
    )
    for start in tqdm(
        range(0, len(unique_sequences), batch_size),
        total=total_batches,
        desc="[feature] EnzSub embedding",
    ):
        chunk = unique_sequences[start:start + batch_size]
        batch = [(f"seq_{start + i}", seq) for i, seq in enumerate(chunk)]
        emb = encoder.get_embedding(batch).cpu().numpy()
        for seq, vec in zip(chunk, emb):
            seq_to_vec[seq] = vec.astype(np.float32, copy=False)

    return np.stack([seq_to_vec[seq] for seq in processed]).astype(np.float32)

def load_or_build_sequence_embeddings(
    sequences: Sequence[str],
    args: argparse.Namespace,
) -> np.ndarray:
    cache_path = Path(args.embedding_cache) if args.embedding_cache else None
    if cache_path and cache_path.exists() and not args.rebuild_cache:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        cached_sequences = payload.get("sequences")
        cached_embeddings = payload.get("embeddings")
        if cached_sequences == list(sequences) and cached_embeddings is not None:
            print(f"[cache] Loaded EnzSub sequence embeddings: {cache_path}")
            return np.asarray(cached_embeddings, dtype=np.float32)
        print("[cache] Existing embedding cache does not match current sequences; rebuilding.")

    emb = enzsub_sequence_to_vec(
        sequences,
        enzsub_code_dir=Path(args.enzsub_code_dir),
        encoder_type=args.encoder_type,
        model_mode=args.model_mode,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.embedding_batch_size,
        max_seq_len=args.max_seq_len,
        truncate_mode=args.truncate_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"sequences": list(sequences), "embeddings": emb}, f)
        print(f"[cache] Saved EnzSub sequence embeddings: {cache_path}")

    return emb

def load_kcat_dataset(path: Path) -> Tuple[List[str], List[str], np.ndarray, pd.DataFrame]:
    with open(path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        value = float(item["Value"])
        smiles = str(item["Smiles"])
        if value == 0 or "." in smiles:
            continue
        rows.append({
            "sequence": item["Sequence"],
            "smiles": smiles,
            "label": math.log(value, 10),
            "ECNumber": item.get("ECNumber", ""),
            "Organism": item.get("Organism", ""),
            "Substrate": item.get("Substrate", ""),
            "Type": item.get("Type", ""),
        })

    df = pd.DataFrame(rows)
    return df["sequence"].tolist(), df["smiles"].tolist(), df["label"].to_numpy(float), df

def load_km_dataset(path: Path) -> Tuple[List[str], List[str], np.ndarray, pd.DataFrame]:
    df = pd.read_pickle(path)
    out = pd.DataFrame({
        "sequence": df["Sequence"],
        "smiles": df["smiles"],
        "label": df["log10_KM"].astype(float),
    }).dropna()
    return out["sequence"].tolist(), out["smiles"].tolist(), out["label"].to_numpy(float), out

def load_kcat_km_dataset(path: Path) -> Tuple[List[str], List[str], np.ndarray, pd.DataFrame]:
    df = pd.read_excel(path)
    value_col = "kcat/KM Value [1/mMs-1]"
    out = pd.DataFrame({
        "sequence": df["Seqs"],
        "smiles": df["smiles"],
        "label": df[value_col].astype(float).map(lambda x: math.log(x, 10)),
        "Substrate": df.get("Substrate ", ""),
        "Primary Accession No.": df.get("Primary Accession No.", ""),
    }).dropna()
    return out["sequence"].tolist(), out["smiles"].tolist(), out["label"].to_numpy(float), out

def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

def evaluate_holdout(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_ratio: float,
    n_runs: int,
    seed: int,
    n_estimators: int | None,
    n_jobs: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    pred_frames = []

    all_indices = np.arange(len(labels))
    for run in tqdm(range(n_runs), desc="[eval] ExtraTrees holdout runs"):
        run_seed = seed + run
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=train_ratio,
            random_state=run_seed,
            shuffle=True,
        )
        print(
            f"[eval] run={run + 1}/{n_runs}: fitting ExtraTrees "
            f"(train={len(train_idx)}, test={len(test_idx)}, "
            f"n_estimators={n_estimators or 'sklearn_default'}, n_jobs={n_jobs})",
            flush=True,
        )
        model_kwargs = {"n_jobs": n_jobs, "random_state": run_seed}
        if n_estimators is not None:
            model_kwargs["n_estimators"] = n_estimators
        model = ExtraTreesRegressor(**model_kwargs)
        model.fit(features[train_idx], labels[train_idx])

        pred_test = model.predict(features[test_idx])
        metrics = regression_metrics(labels[test_idx], pred_test)
        metrics.update({"run": run + 1, "train_size": len(train_idx), "test_size": len(test_idx)})
        metrics_rows.append(metrics)

        pred_all = model.predict(features)
        frame = pd.DataFrame({
            "run": run + 1,
            "sample_index": all_indices,
            "split": np.where(np.isin(all_indices, train_idx), "train", "test"),
            "label_log10": labels,
            "pred_log10": pred_all,
            "label_raw": np.power(10.0, labels),
            "pred_raw": np.power(10.0, pred_all),
        })
        pred_frames.append(frame)

    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True)

def evaluate_kfold(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    n_runs: int,
    seed: int,
    n_estimators: int | None,
    n_jobs: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    pred_frames = []
    indices = np.arange(len(labels))

    for run in tqdm(range(n_runs), desc="[eval] ExtraTrees kfold runs"):
        run_seed = seed + run
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=run_seed)
        pred = np.zeros_like(labels, dtype=float)
        fold_ids = np.zeros_like(labels, dtype=int)

        for fold, (train_idx, test_idx) in enumerate(
            tqdm(
                kf.split(features, labels),
                total=n_splits,
                desc=f"[eval] run {run + 1} folds",
                leave=False,
            ),
            start=1,
        ):
            print(
                f"[eval] run={run + 1}/{n_runs}, fold={fold}/{n_splits}: "
                f"fitting ExtraTrees (train={len(train_idx)}, test={len(test_idx)}, "
                f"n_estimators={n_estimators or 'sklearn_default'}, n_jobs={n_jobs})",
                flush=True,
            )
            model_kwargs = {"n_jobs": n_jobs, "random_state": run_seed}
            if n_estimators is not None:
                model_kwargs["n_estimators"] = n_estimators
            model = ExtraTreesRegressor(**model_kwargs)
            model.fit(features[train_idx], labels[train_idx])
            pred[test_idx] = model.predict(features[test_idx])
            fold_ids[test_idx] = fold

        metrics = regression_metrics(labels, pred)
        metrics.update({"run": run + 1, "n_splits": n_splits, "n_samples": len(labels)})
        metrics_rows.append(metrics)

        pred_frames.append(pd.DataFrame({
            "run": run + 1,
            "sample_index": indices,
            "fold": fold_ids,
            "label_log10": labels,
            "pred_log10": pred,
            "label_raw": np.power(10.0, labels),
            "pred_raw": np.power(10.0, pred),
        }))

    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True)

def load_task(args: argparse.Namespace) -> Tuple[List[str], List[str], np.ndarray, pd.DataFrame]:
    data_dir = Path(args.data_dir)
    if args.task == "kcat":
        return load_kcat_dataset(data_dir / "Kcat_combination_0918_wildtype_mutant.json")
    if args.task == "km":
        return load_km_dataset(data_dir / "Km_test_11722.pkl")
    if args.task == "kcat_km":
        return load_kcat_km_dataset(data_dir / "kcat_km_samples.xlsx")
    raise ValueError(f"Unknown task: {args.task}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace UniKP protein embeddings with EnzSub embeddings and evaluate ExtraTrees."
    )
    parser.add_argument("--task", choices=["kcat", "km", "kcat_km"], default="kcat")
    parser.add_argument("--data-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--output-dir", default=str(ROOT / "enzsub_unikp_outputs"))

    parser.add_argument("--enzsub-code-dir", default=str(DEFAULT_ENZSUB_CODE_DIR))
    parser.add_argument("--encoder-type", default="esm2_3b")
    parser.add_argument("--model-mode", choices=["base", "cpt", "base_sub", "cpt_sub"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--truncate-mode", choices=["head", "head_tail"], default="head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--kfold-splits", type=int, default=5)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Default: do not set this parameter, matching ExtraTreesRegressor().",
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--save-features", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[task] Loading {args.task} data")
    sequences, smiles, labels, metadata = load_task(args)
    print(f"[task] Samples: {len(labels)}")

    print("[feature] Building UniKP SMILES embeddings")
    smiles_vec = smiles_to_vec(smiles)
    print(f"[feature] SMILES shape: {smiles_vec.shape}")

    print("[feature] Building EnzSub enzyme embeddings")
    seq_vec = load_or_build_sequence_embeddings(sequences, args)
    print(f"[feature] EnzSub sequence shape: {seq_vec.shape}")

    features = np.concatenate((smiles_vec, seq_vec), axis=1)
    print(f"[feature] Fused shape: {features.shape}")

    if args.save_features:
        feature_path = out_dir / f"{args.task}_features_enzsub_{args.model_mode}.pkl"
        with open(feature_path, "wb") as f:
            pickle.dump({
                "features": features,
                "smiles_vec": smiles_vec,
                "seq_vec": seq_vec,
                "labels": labels,
                "metadata": metadata,
                "args": vars(args),
            }, f)
        print(f"[feature] Saved: {feature_path}")

    if args.task in {"kcat", "km"}:
        train_ratio = 0.9 if args.task == "kcat" else 0.8
        metrics, preds = evaluate_holdout(
            features,
            labels,
            train_ratio=train_ratio,
            n_runs=args.n_runs,
            seed=args.seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )
    else:
        metrics, preds = evaluate_kfold(
            features,
            labels,
            n_splits=args.kfold_splits,
            n_runs=args.n_runs,
            seed=args.seed,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
        )

    metrics_path = out_dir / f"{args.task}_metrics_enzsub_{args.model_mode}.csv"
    preds_path = out_dir / f"{args.task}_predictions_enzsub_{args.model_mode}.csv"
    metrics.to_csv(metrics_path, index=False)
    preds.to_csv(preds_path, index=False)

    print("\n[metrics]")
    print(metrics.to_string(index=False))
    print(f"\n[saved] {metrics_path}")
    print(f"[saved] {preds_path}")

if __name__ == "__main__":
    main()
