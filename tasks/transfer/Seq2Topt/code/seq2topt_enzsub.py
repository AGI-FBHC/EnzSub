#!/usr/bin/env python3
"""Predict Topt with a frozen EnzSub encoder and a trained Seq2Topt task head."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import List

import pandas as pd
import torch

from enzsub_seq2topt import EnzSubSeq2Topt, autocast_context

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Topt with EnzSub + Seq2Topt.")
    parser.add_argument("--input", required=True, help="CSV containing protein sequences")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-checkpoint", required=True)
    parser.add_argument("--enzsub-root", required=True)
    parser.add_argument("--enzsub-checkpoint")
    parser.add_argument("--id-column", default="uniprot_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow a different EnzSub checkpoint than the one used for cached training features.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device)
    task_checkpoint_path = Path(args.task_checkpoint).expanduser().resolve()
    task_checkpoint = torch_load(task_checkpoint_path)
    model_config = task_checkpoint["model_config"]
    cache_identity = task_checkpoint["cache_identity"]

    encoder_type = cache_identity["encoder_type"]
    model_mode = cache_identity["model_mode"]
    expected_hash = cache_identity.get("enzsub_checkpoint_sha256")
    enzsub_checkpoint = None
    if args.enzsub_checkpoint:
        enzsub_checkpoint = Path(args.enzsub_checkpoint).expanduser().resolve()
        if not enzsub_checkpoint.is_file():
            raise FileNotFoundError(enzsub_checkpoint)
        actual_hash = sha256_file(enzsub_checkpoint)
        if expected_hash and actual_hash != expected_hash and not args.allow_checkpoint_mismatch:
            raise RuntimeError(
                "The EnzSub checkpoint differs from the encoder used to build the training cache. "
                "Pass --allow-checkpoint-mismatch only for an intentional ablation."
            )
    elif model_mode != "base":
        raise ValueError(f"model_mode={model_mode!r} requires --enzsub-checkpoint")

    model = EnzSubSeq2Topt(
        enzsub_root=args.enzsub_root,
        encoder_type=encoder_type,
        model_mode=model_mode,
        enzsub_checkpoint=enzsub_checkpoint,
        head_dim=int(model_config["head_dim"]),
        window=int(model_config["window"]),
        n_head=int(model_config["n_head"]),
        n_RD=int(model_config["n_RD"]),
        max_seq_length=int(cache_identity["max_seq_length"]),
        device=device,
    )
    if model.encoder_dim != int(model_config["encoder_dim"]):
        raise RuntimeError(
            f"Encoder dimension mismatch: model={model.encoder_dim}, "
            f"checkpoint={model_config['encoder_dim']}"
        )
    model.load_task_state_dict(task_checkpoint["task_state_dict"])
    model.to(device)
    model.eval()

    input_path = Path(args.input).expanduser().resolve()
    table = pd.read_csv(input_path)
    if args.sequence_column not in table.columns:
        raise KeyError(f"Input CSV does not contain {args.sequence_column!r}")
    if args.id_column not in table.columns:
        table[args.id_column] = [str(index) for index in range(len(table))]

    ids = table[args.id_column].astype(str).tolist()
    sequences = table[args.sequence_column].astype(str).tolist()
    predictions: List[float] = []
    target_scale = float(task_checkpoint.get("target_scale", 120.0))

    with torch.no_grad():
        for start in range(0, len(table), args.batch_size):
            batch_ids = ids[start : start + args.batch_size]
            batch_sequences = sequences[start : start + args.batch_size]
            with autocast_context(device, enabled=args.amp and device.type == "cuda"):
                output = model(batch_ids, batch_sequences).squeeze(-1)
            predictions.extend((output.float().cpu().numpy() * target_scale).tolist())

    result = table.copy()
    result["predicted_topt"] = predictions
    result["enzsub_truncated"] = [
        len("".join(sequence.split())) > model.max_seq_length for sequence in sequences
    ]
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} predictions to {output_path}")

if __name__ == "__main__":
    main()
