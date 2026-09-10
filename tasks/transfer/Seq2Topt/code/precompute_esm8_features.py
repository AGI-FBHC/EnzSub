#!/usr/bin/env python3
"""Cache the original Seq2Topt ESM-2 t6 8M residue representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch

from enzsub_seq2topt import autocast_context, normalize_sequence

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
        raise RuntimeError("CUDA was requested but CUDA is not available.")
    return device

def chunked(items: Sequence[Tuple[str, str]], size: int) -> Iterable[List[Tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache original Seq2Topt ESM-2 t6 8M residue features."
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1022)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_seq_length <= 0:
        raise ValueError("batch size and max sequence length must be positive")
    device = resolve_device(args.device)
    input_paths = [Path(value).expanduser().resolve() for value in args.inputs]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    sequences, source_metadata = load_unique_sequences(
        input_paths, args.sequence_column, args.max_seq_length
    )
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    feature_dir = cache_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "encoder_type": "esm2_t6_8M_UR50D",
        "model_mode": "published_base",
        "encoder_dim": 320,
        "max_seq_length": args.max_seq_length,
        "storage_dtype": args.storage_dtype,
        "representation_layer": 6,
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
            "representation_layer",
        )
        mismatches = {
            key: (existing.get(key), manifest.get(key))
            for key in identity_keys
            if existing.get(key) != manifest.get(key)
        }
        if mismatches:
            raise RuntimeError(f"Cache manifest is incompatible: {mismatches}")

    import esm

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    batch_converter = alphabet.get_batch_converter()
    pending = [
        (key, sequence)
        for key, sequence in sequences.items()
        if args.overwrite or not (feature_dir / f"{key}.pt").is_file()
    ]
    pending.sort(key=lambda item: len(item[1]))
    print(
        f"Caching {len(pending)}/{len(sequences)} unique ESM-2 t6 8M sequences "
        f"on {device}; layer=6, dim=320"
    )
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    completed = 0
    for batch in chunked(pending, args.batch_size):
        samples = [(key, sequence) for key, sequence in batch]
        _, _, tokens = batch_converter(samples)
        tokens = tokens.to(device, non_blocking=True)
        with torch.no_grad(), autocast_context(
            device, enabled=args.amp and device.type == "cuda"
        ):
            outputs = model(tokens, repr_layers=[6], return_contacts=False)
        representations = outputs["representations"][6]
        for row, (key, sequence) in enumerate(batch):
            residue_features = representations[row, 1 : len(sequence) + 1].detach().cpu().to(storage_dtype)
            if residue_features.shape != (len(sequence), 320):
                raise RuntimeError(
                    f"Residue alignment mismatch for {key}: "
                    f"sequence={len(sequence)}, features={tuple(residue_features.shape)}"
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
