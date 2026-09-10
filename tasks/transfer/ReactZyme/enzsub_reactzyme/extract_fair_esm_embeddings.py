#!/usr/bin/env python3
"""Extract residue-only frozen fair-ESM representations for ReactZyme."""
from __future__ import annotations

import argparse
from pathlib import Path

import esm
import torch

from common import load_pair_file, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(str(args.checkpoint))
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    paths = (
        args.split_dir / f"positive_train_val_{args.split_name}.pt",
        args.split_dir / f"positive_test_{args.split_name}.pt",
    )
    sequences = sorted({sequence for path in paths for _, sequence, _ in load_pair_file(path, 1.0)})
    embeddings = {}
    truncated = 0
    with torch.inference_mode():
        for index, original_sequence in enumerate(sequences, start=1):
            sequence = original_sequence[:args.max_length]
            truncated += int(len(sequence) != len(original_sequence))
            _, _, tokens = batch_converter([(str(index), sequence)])
            tokens = tokens.to(device)
            representation = model(tokens, repr_layers=[model.num_layers], return_contacts=False)["representations"][model.num_layers][0]
            start = int(getattr(alphabet, "prepend_bos", False))
            end = start + len(sequence)
            if representation.shape[0] < end:
                raise RuntimeError(
                    f"fair-ESM token count too short for sequence {index}: "
                    f"tokens={representation.shape[0]}, residues={len(sequence)}"
                )
            vector = representation[start:end].mean(dim=0).float().cpu()
            if vector.numel() == 0 or not torch.isfinite(vector).all():
                raise ValueError(f"Invalid fair-ESM embedding for sequence {index}")
            embeddings[original_sequence] = vector
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index % 1000 == 0 or index == len(sequences):
                print(f"Encoded {index}/{len(sequences)} sequences")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact": "reactzyme_fair_esm_enzyme_embeddings",
        "checkpoint": str(args.checkpoint),
        "split_name": args.split_name,
        "sequence_count": len(embeddings),
        "embedding_dim": int(next(iter(embeddings.values())).numel()),
        "max_length": args.max_length,
        "truncated_sequence_count": truncated,
        "pooling": "fair_esm_final_layer_residue_mean_excluding_bos_eos",
    }
    torch.save({"embeddings": embeddings, "metadata": metadata}, args.output)
    write_json(args.output.with_suffix(".json"), metadata)
    print(f"Saved {len(embeddings)} fair-ESM embeddings to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
