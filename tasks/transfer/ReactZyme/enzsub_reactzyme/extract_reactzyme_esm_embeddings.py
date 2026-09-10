#!/usr/bin/env python3
"""Extract the original ReactZyme-style ESM enzyme vectors for one split."""
from __future__ import annotations

import argparse
from pathlib import Path

import esm
import torch

from common import load_pair_file, write_json

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet(str(args.checkpoint))
    model = model.to(device).eval()
    layer = model.num_layers
    paths = [args.split_dir / f"positive_train_val_{args.split_name}.pt", args.split_dir / f"positive_test_{args.split_name}.pt"]
    sequences = sorted({sequence for path in paths for _, sequence, _ in load_pair_file(path, 1.0)})
    embeddings = {}
    with torch.no_grad():
        for index, sequence in enumerate(sequences, start=1):
            tokens = torch.tensor(alphabet.encode(sequence[:args.max_length])).view(1, -1).to(device)
            output = model(tokens, repr_layers=[layer], return_contacts=False)
            # Mirrors ReactZyme process_esm.py: mean across all emitted tokens.
            embeddings[sequence] = output["representations"][layer].squeeze(0).mean(0).detach().cpu().float()
            if device.type == "cuda": torch.cuda.empty_cache()
            if index % 1000 == 0: print(f"Encoded {index}/{len(sequences)} sequences")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": {"artifact": "reactzyme_esm_enzyme_embeddings", "checkpoint": str(args.checkpoint), "split_name": args.split_name, "embedding_dim": int(next(iter(embeddings.values())).numel()), "pooling": "ReactZyme process_esm token mean including special tokens", "max_length": args.max_length}}, args.output)
    write_json(args.output.with_suffix(".json"), {"artifact": "reactzyme_esm_enzyme_embeddings", "sequence_count": len(embeddings), "checkpoint": str(args.checkpoint), "split_name": args.split_name, "max_length": args.max_length})
    print(f"Saved {len(embeddings)} ReactZyme-style ESM embeddings to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
