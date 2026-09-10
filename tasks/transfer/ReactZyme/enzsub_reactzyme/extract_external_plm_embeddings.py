#!/usr/bin/env python3
"""Extract external frozen-PLM enzyme representations for ReactZyme.

The actual model adapters live in the established EC external-PLM generator.
This thin bridge deliberately reuses those adapters and their pooling rules,
while deriving the enzyme universe directly from ReactZyme positive pairs and
writing the dictionary format consumed by ``train_and_evaluate.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from common import load_pair_file

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-source", type=Path, required=True,
                        help="Existing generate_plm_embeddings.py with the validated adapters")
    parser.add_argument("--backend", choices=("hf_t5", "hf_encoder", "esm3"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--local-path")
    parser.add_argument("--pretrained-name")
    parser.add_argument("--model-class", default="auto")
    parser.add_argument("--tokenizer-class", default="auto")
    parser.add_argument("--input-mode", default="raw")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Atomically save after this many newly encoded sequences")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Require already cached model files")
    parser.add_argument("--resume", action="store_true", help="Reuse only a matching complete output")
    parser.add_argument("--validate-batch-equivalence", action="store_true")
    parser.add_argument("--validate-min-cosine", type=float, default=0.999)
    return parser.parse_args()

def load_adapter_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("enzsub_external_plm_adapters", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load external PLM adapter source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("build_adapter", "make_batches", "preprocess_sequence", "sequence_sha256")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Adapter source is missing required symbols: {missing}")
    return module

def enzyme_sequences(split_dir: Path, split_name: str):
    paths = (
        split_dir / f"positive_train_val_{split_name}.pt",
        split_dir / f"positive_test_{split_name}.pt",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted({sequence for path in paths for _, sequence, _ in load_pair_file(path, 1.0)})

def expected_metadata(args: argparse.Namespace, adapter_info: Any, sequence_hashes: Mapping[str, str]) -> Dict[str, Any]:
    info = asdict(adapter_info) if hasattr(adapter_info, "__dataclass_fields__") else dict(adapter_info)
    return {
        "artifact": "reactzyme_external_plm_enzyme_embeddings",
        "model_name": args.model_name,
        "model_id": args.model_id,
        "backend": args.backend,
        "adapter": info,
        "split_name": args.split_name,
        "max_length": args.max_length,
        "sequence_count": len(sequence_hashes),
        "sequence_sha256": dict(sequence_hashes),
    }

def load_resumable_output(path: Path, expected: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("embeddings"), Mapping):
        raise ValueError(f"Cannot resume malformed output: {path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Cannot resume output without metadata: {path}")
    checks = ("model_name", "model_id", "backend", "adapter", "split_name", "max_length", "sequence_sha256")
    mismatched = [key for key in checks if metadata.get(key) != expected.get(key)]
    if mismatched:
        raise ValueError(f"Refusing to resume incompatible output {path}: {mismatched}")
    if set(payload["embeddings"]) != set(expected["sequence_sha256"]):
        unknown = set(payload["embeddings"]) - set(expected["sequence_sha256"])
        if unknown:
            raise ValueError(f"Refusing to resume output containing unknown sequences: {path}")
    result: Dict[str, torch.Tensor] = {}
    for key, value in payload["embeddings"].items():
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten().cpu()
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"Refusing to resume non-finite embedding for {key!r}")
        result[str(key)] = tensor
    dimensions = {value.numel() for value in result.values()}
    if len(dimensions) > 1:
        raise ValueError(f"Refusing to resume inconsistent embedding dimensions: {path}")
    return result

def atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)

def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.max_length <= 0 or args.batch_size <= 0 or args.max_batch_size <= 0 or args.save_every <= 0:
        raise ValueError("max-length, batch-size, max-batch-size, and save-every must be positive")

    adapter_module = load_adapter_module(args.adapter_source)
    sequences = enzyme_sequences(args.split_dir, args.split_name)
    sequence_hashes = {sequence: adapter_module.sequence_sha256(sequence) for sequence in sequences}
    spec: Dict[str, Any] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "model_class": args.model_class,
        "tokenizer_class": args.tokenizer_class,
        "input_mode": args.input_mode,
        "prefix": args.prefix,
        "dtype": args.dtype,
        "validate_token_count": True,
        "validate_batch_equivalence": args.validate_batch_equivalence,
        "validate_min_cosine": args.validate_min_cosine,
    }
    if args.local_path:
        spec["local_path"] = args.local_path
    if args.pretrained_name:
        spec["pretrained_name"] = args.pretrained_name
    adapter = adapter_module.build_adapter(spec, args.device)
    metadata = expected_metadata(args, adapter.info, sequence_hashes)
    embeddings = load_resumable_output(args.output, metadata) if args.resume else {}
    if len(embeddings) == len(sequences):
        print(f"Resuming: complete matching artifact already exists at {args.output}")
        return 0

    records = []
    truncated, replaced = 0, 0
    for sequence in sequences:
        cleaned, original_length, replaced_count = adapter_module.preprocess_sequence(sequence, args.max_length)
        truncated += int(original_length > args.max_length)
        replaced += replaced_count
        if sequence not in embeddings:
            records.append((sequence, cleaned))
    batches = adapter_module.make_batches(
        records, args.batch_size, args.token_budget, args.max_batch_size
    )
    def save_progress(status: str) -> None:
        progress = dict(metadata)
        progress.update({"status": status, "completed_sequence_count": len(embeddings)})
        if embeddings:
            progress["embedding_dim"] = next(iter(embeddings.values())).numel()
        atomic_save(args.output, {"embeddings": embeddings, "metadata": progress})

    newly_encoded = 0
    for index, batch in enumerate(batches, start=1):
        vectors = adapter.embed(batch)
        if vectors.shape[0] != len(batch):
            raise RuntimeError(f"Adapter returned {vectors.shape[0]} vectors for batch of {len(batch)}")
        for (sequence, _), vector in zip(batch, vectors):
            vector = torch.as_tensor(vector, dtype=torch.float32).flatten().cpu()
            if vector.numel() == 0 or not torch.isfinite(vector).all():
                raise ValueError(f"Invalid embedding generated for sequence {sequence!r}")
            embeddings[sequence] = vector
            newly_encoded += 1
        if newly_encoded >= args.save_every:
            save_progress("in_progress")
            newly_encoded = 0
        if index % 25 == 0 or index == len(batches):
            print(f"Encoded {len(embeddings)}/{len(sequences)} sequences ({index}/{len(batches)} batches)")
    dimensions = {vector.numel() for vector in embeddings.values()}
    if len(embeddings) != len(sequences) or len(dimensions) != 1:
        raise RuntimeError("Embedding output has incomplete coverage or inconsistent dimensions")
    metadata.update({
        "status": "complete",
        "completed_sequence_count": len(embeddings),
        "embedding_dim": next(iter(dimensions)),
        "truncated_sequence_count": truncated,
        "nonstandard_residue_replacement_count": replaced,
        "pooling": metadata["adapter"].get("pooling"),
    })
    atomic_save(args.output, {"embeddings": embeddings, "metadata": metadata})
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Saved {len(embeddings)} {args.model_name} embeddings to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
