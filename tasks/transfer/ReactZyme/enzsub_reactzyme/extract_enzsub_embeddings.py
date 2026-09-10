#!/usr/bin/env python3
"""Extract frozen EnzSub sequence representations for a ReactZyme split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from common import load_pair_file, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enzsub-code-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-mode", choices=["base", "cpt", "base_sub", "cpt_sub"], default="cpt_sub")
    parser.add_argument("--encoder-type", default="esm2_t33_650M")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if not args.enzsub_code_root.is_dir():
        raise FileNotFoundError(args.enzsub_code_root)
    if args.model_mode != "base" and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sys.path.insert(0, str(args.enzsub_code_root))
    try:
        # Server layout: repository root/train/sub/model.py
        from enzsub.sub.model import EnzSubModelForDownstream
    except ModuleNotFoundError as error:
        if error.name != "sub":
            raise
        # Legacy/local layout retained for portability.
        from model.Sub.model import EnzSubModelForDownstream

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EnzSubModelForDownstream(
        encoder_type=args.encoder_type,
        model_mode=args.model_mode,
        checkpoint_path=None if args.model_mode == "base" else str(args.checkpoint),
        freeze_backbone=True,
        device=str(device),
    )
    paths = [args.split_dir / f"positive_train_val_{args.split_name}.pt", args.split_dir / f"positive_test_{args.split_name}.pt"]
    sequences = sorted({seq for path in paths for _, seq, _ in load_pair_file(path, 1.0)})
    embeddings = {}
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        vectors = model.get_embedding([(str(index), sequence) for index, sequence in enumerate(batch)])
        for sequence, vector in zip(batch, vectors):
            embeddings[sequence] = vector.detach().cpu().float()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    write_json(args.output.with_suffix(".json"), {
        "artifact": "enzsub_reactzyme_enzyme_embeddings",
        "checkpoint": str(args.checkpoint), "model_mode": args.model_mode,
        "encoder_type": args.encoder_type, "split_name": args.split_name,
        "sequence_count": len(embeddings),
        "embedding_dim": int(next(iter(embeddings.values())).numel()),
        "pooling": "EnzSub mean-pooled final-layer enzyme representation h",
    })
    print(f"Saved {len(embeddings)} EnzSub embeddings to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
