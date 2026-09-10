#!/usr/bin/env python3
"""
Predict kcat or Km for new enzyme--substrate pairs with an EnzSub representation.

The downstream contract follows the controlled UniKP comparison:
the SMILES representation and ExtraTrees head are retained, while the protein
representation is supplied by EnzSub.  The ExtraTrees head is fitted on a
previously generated full feature cache, so the labelled comparison dataset is
not embedded again.  Only the query sequences and SMILES are embedded.

This is an EnzSub-UniKP fitted-head predictor, not the official pretrained
UniKP predictor.  Predictions are returned both in log10 space and on the
original scale.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor

from evaluate_enzsub_in_unikp import ROOT, enzsub_sequence_to_vec, smiles_to_vec

def load_training_feature_cache(
    path: Path,
    *,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a comparison feature cache containing fused features and labels."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict) or "features" not in payload or "labels" not in payload:
        raise ValueError(
            f"{path} must be a full feature cache with 'features' and 'labels'."
        )

    cache_args = payload.get("args") or {}
    cached_task = cache_args.get("task")
    if cached_task and cached_task != task:
        raise ValueError(
            f"Training cache task is {cached_task!r}, but --task is {task!r}."
        )

    features = np.asarray(payload["features"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.float32).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"{path} features must be a 2-D matrix, got {features.shape}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{path} has {features.shape[0]} feature rows but {labels.shape[0]} labels."
        )
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError(f"{path} contains non-finite features or labels.")

    return features, labels, cache_args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict kcat or Km with an EnzSub protein representation."
    )
    parser.add_argument("--task", choices=["kcat", "km"], required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file with required columns: sequence and smiles.",
    )
    parser.add_argument(
        "--training-feature-cache",
        required=True,
        help="Full EnzSub-UniKP feature cache produced by the comparison pipeline.",
    )
    parser.add_argument("--output", required=True, help="Output CSV path.")

    parser.add_argument("--enzsub-code-dir", default=None)
    parser.add_argument("--encoder-type", default="esm2_650m")
    parser.add_argument("--model-mode", choices=["base", "cpt", "base_sub", "cpt_sub"], default="cpt_sub")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--truncate-mode", choices=["head", "head_tail"], default="head")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    cache_path = Path(args.training_feature_cache).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    query = pd.read_csv(input_path)
    required_columns = {"sequence", "smiles"}
    missing = sorted(required_columns.difference(query.columns))
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {', '.join(missing)}"
        )
    if query.empty:
        raise ValueError(f"{input_path} contains no query rows.")
    query = query.copy()
    query["sequence"] = query["sequence"].astype(str)
    query["smiles"] = query["smiles"].astype(str)
    if query[["sequence", "smiles"]].isin(["", "nan", "None"]).any().any():
        raise ValueError(f"{input_path} contains an empty sequence or SMILES value.")

    train_features, labels, cache_args = load_training_feature_cache(
        cache_path,
        task=args.task,
    )

    print(f"[train] Loading cached features: {cache_path}")
    print(f"[train] Samples: {len(labels)}; feature_dim={train_features.shape[1]}")
    print(f"[query] Samples: {len(query)}")

    smiles_features = np.asarray(smiles_to_vec(query["smiles"].tolist()), dtype=np.float32)

    enzsub_code_dir = (
        Path(args.enzsub_code_dir).expanduser().resolve()
        if args.enzsub_code_dir
        else None
    )
    if enzsub_code_dir is None:
        from evaluate_enzsub_in_unikp import DEFAULT_ENZSUB_CODE_DIR

        enzsub_code_dir = DEFAULT_ENZSUB_CODE_DIR

    sequence_features = enzsub_sequence_to_vec(
        query["sequence"].tolist(),
        enzsub_code_dir=enzsub_code_dir,
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
    query_features = np.concatenate((smiles_features, sequence_features), axis=1)
    if query_features.shape[1] != train_features.shape[1]:
        raise ValueError(
            "Query feature dimension does not match the training cache: "
            f"query={query_features.shape[1]}, train={train_features.shape[1]}. "
            "Check the SMILES encoder and EnzSub checkpoint/backbone."
        )

    model_kwargs = {"n_jobs": args.n_jobs, "random_state": args.seed}
    if args.n_estimators is not None:
        model_kwargs["n_estimators"] = args.n_estimators
    model = ExtraTreesRegressor(**model_kwargs)
    model.fit(train_features, labels)

    pred_log10 = model.predict(query_features)
    output = query.copy()
    output["prediction_log10"] = pred_log10
    output["prediction_original_scale"] = np.power(10.0, pred_log10)
    output["task"] = args.task
    output["encoder_type"] = args.encoder_type
    output["model_mode"] = args.model_mode
    output["training_feature_cache"] = str(cache_path)
    output["training_samples"] = len(labels)
    output["training_feature_dim"] = train_features.shape[1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[saved] {output_path}")
    cache_encoder = cache_args.get("encoder_type")
    cache_mode = cache_args.get("model_mode")
    if cache_encoder and cache_encoder != args.encoder_type:
        print(
            "[warning] Training cache encoder_type differs from the query encoder_type; "
            "verify that the cached EnzSub features match the requested model."
        )
    if cache_mode and cache_mode != args.model_mode:
        print(
            "[warning] Training cache model_mode differs from the query model_mode; "
            "verify that the cached EnzSub features match the requested model."
        )

if __name__ == "__main__":
    main()
