#!/usr/bin/env python3
"""Precompute frozen EnzSub per-residue features for Seq2Topt experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch

from enzsub_seq2topt import (
    _ensure_enzsub_importable,
    _move_tokens,
    autocast_context,
    normalize_sequence,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

def load_unique_sequences(
    paths: Sequence[Path], sequence_column: str, max_seq_length: int
) -> Tuple[Dict[str, str], Dict[str, Dict[str, object]]]:
    unique: Dict[str, str] = {}
    sources: Dict[str, Dict[str, object]] = {}
    for path in paths:
        table = pd.read_csv(path)
        if sequence_column not in table.columns:
            raise KeyError(f"{path} does not contain column {sequence_column!r}")
        for raw_sequence in table[sequence_column].tolist():
            sequence = normalize_sequence(raw_sequence, max_seq_length)
            unique.setdefault(sequence_key(sequence), sequence)
        sources[str(path.resolve())] = {
            "rows": int(len(table)),
            "sha256": sha256_file(path),
        }
    return unique, sources

def chunked(items: Sequence[Tuple[str, str]], size: int) -> Iterable[List[Tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen EnzSub per-residue features for Seq2Topt."
    )
    parser.add_argument("--enzsub-root", required=True)
    parser.add_argument("--enzsub-checkpoint")
    parser.add_argument("--encoder-type", default="esm2_650m")
    parser.add_argument(
        "--model-mode",
        choices=["base", "cpt", "base_sub", "cpt_sub"],
        default="cpt_sub",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(PROJECT_DIR / "data" / "Topt" / "train_os.csv"),
            str(PROJECT_DIR / "data" / "Topt" / "test.csv"),
        ],
    )
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.model_mode != "base" and not args.enzsub_checkpoint:
        raise ValueError(f"model_mode={args.model_mode!r} requires --enzsub-checkpoint")

    device = resolve_device(args.device)
    enzsub_root = _ensure_enzsub_importable(args.enzsub_root)
    from enzsub.sub.model import EnzSubModelForDownstream

    checkpoint = None
    checkpoint_sha256 = None
    if args.enzsub_checkpoint:
        checkpoint = Path(args.enzsub_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)

    encoder = EnzSubModelForDownstream(
        encoder_type=args.encoder_type,
        model_mode=args.model_mode,
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        freeze_backbone=True,
        device=str(device),
        strict_lora_load=True,
        allow_lora_reverse_detect=True,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    encoder_limit = int(encoder.max_seq_len)
    requested_limit = encoder_limit if args.max_seq_length is None else args.max_seq_length
    max_seq_length = min(int(requested_limit), encoder_limit)
    if max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive")

    input_paths = [Path(value).expanduser().resolve() for value in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    sequences, source_metadata = load_unique_sequences(
        input_paths, args.sequence_column, max_seq_length
    )
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    feature_dir = cache_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "encoder_type": args.encoder_type,
        "model_mode": args.model_mode,
        "encoder_dim": int(encoder.hidden_dim),
        "max_seq_length": max_seq_length,
        "storage_dtype": args.storage_dtype,
        "enzsub_root": str(enzsub_root),
        "enzsub_checkpoint": None if checkpoint is None else str(checkpoint),
        "enzsub_checkpoint_sha256": checkpoint_sha256,
        "sources": source_metadata,
        "unique_sequences": len(sequences),
    }
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        existing = json.loads(manifest_path.read_text())
        identity_keys = (
            "encoder_type",
            "model_mode",
            "encoder_dim",
            "max_seq_length",
            "storage_dtype",
            "enzsub_checkpoint_sha256",
        )
        mismatches = {
            key: (existing.get(key), manifest.get(key))
            for key in identity_keys
            if existing.get(key) != manifest.get(key)
        }
        if mismatches:
            raise RuntimeError(
                f"Cache manifest is incompatible: {mismatches}. "
                "Use a new --cache-dir or pass --overwrite."
            )

    pending = [
        (key, sequence)
        for key, sequence in sequences.items()
        if args.overwrite or not (feature_dir / f"{key}.pt").is_file()
    ]
    # Similar lengths share a batch to minimize dynamic-padding memory.
    pending.sort(key=lambda item: len(item[1]))
    print(
        f"Caching {len(pending)}/{len(sequences)} unique sequences on {device}; "
        f"encoder_dim={encoder.hidden_dim}, max_seq_length={max_seq_length}"
    )

    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    completed = 0
    for batch in chunked(pending, args.batch_size):
        samples = [(key, sequence) for key, sequence in batch]
        tokens = _move_tokens(encoder.tokenize(samples), device)
        with torch.no_grad(), autocast_context(
            device, enabled=args.amp and device.type == "cuda"
        ):
            outputs = encoder(tokens, return_per_residue=True)
        features = outputs["per_residue"]
        masks = outputs["residue_mask"].bool()

        for row, (key, sequence) in enumerate(batch):
            residue_features = features[row][masks[row]].detach().cpu().to(storage_dtype)
            if residue_features.shape[0] != len(sequence):
                raise RuntimeError(
                    f"Residue alignment mismatch for {key}: "
                    f"sequence={len(sequence)}, features={residue_features.shape[0]}"
                )
            destination = feature_dir / f"{key}.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(
                {
                    "features": residue_features,
                    "sequence_length": len(sequence),
                    "sequence_sha256": key,
                },
                temporary,
            )
            os.replace(temporary, destination)
        completed += len(batch)
        print(f"  {completed}/{len(pending)}", end="\r", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nFeature cache ready: {cache_dir}")

if __name__ == "__main__":
    main()
