#!/usr/bin/env python3
"""Write a provenance and coverage manifest for one ReactZyme split."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from common import normalise_reaction

Pair = Tuple[str, str]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_pairs(path: Path) -> List[Pair]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}")
    return [(normalise_reaction(values[0]), str(values[1])) for values in payload.values()]

def pair_summary(path: Path, label: str) -> Dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "label": label}
    pairs = load_pairs(path)
    unique_pairs = set(pairs)
    return {
        "path": str(path),
        "exists": True,
        "label": label,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "record_count": len(pairs),
        "unique_pair_count": len(unique_pairs),
        "duplicate_record_count": len(pairs) - len(unique_pairs),
        "unique_reaction_count": len({reaction for reaction, _ in pairs}),
        "unique_enzyme_count": len({enzyme for _, enzyme in pairs}),
    }

def overlap_summary(train: Sequence[Pair], test: Sequence[Pair]) -> Dict:
    train_pairs, test_pairs = set(train), set(test)
    train_reactions = {reaction for reaction, _ in train}
    test_reactions = {reaction for reaction, _ in test}
    train_enzymes = {enzyme for _, enzyme in train}
    test_enzymes = {enzyme for _, enzyme in test}
    return {
        "train_test_pair_overlap": len(train_pairs & test_pairs),
        "train_test_reaction_overlap": len(train_reactions & test_reactions),
        "train_test_enzyme_overlap": len(train_enzymes & test_enzymes),
        "positive_pair_count_union": len(train_pairs | test_pairs),
        "positive_reaction_count_union": len(train_reactions | test_reactions),
        "positive_enzyme_count_union": len(train_enzymes | test_enzymes),
    }

def negative_collision_summary(positive: Iterable[Pair], negative: Iterable[Pair]) -> Dict:
    positives = set(positive)
    negatives = list(negative)
    return {
        "negative_record_count": len(negatives),
        "negative_unique_pair_count": len(set(negatives)),
        "negative_duplicate_record_count": len(negatives) - len(set(negatives)),
        "negative_positive_collision_count": len(set(negatives) & positives),
    }

def file_record(path: Path) -> Dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}

def metadata_record(path: Path) -> Dict:
    record = file_record(path)
    if record["exists"]:
        try:
            record["metadata"] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            record["metadata_error"] = str(exc)
    return record

def selected_models(configured: Sequence[Mapping], names: Sequence[str] | None) -> List[Mapping]:
    if not names:
        return list(configured)
    by_name = {item["name"]: item for item in configured}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown model names: {missing}")
    return [by_name[name] for name in names]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Optional subset of configured models")
    parser.add_argument("--output", type=Path, help="Manifest output path")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    runtime = config.get("runtime", {})
    split_name = runtime.get("split_name", "seq_smi")
    split_dir = Path(paths["split_dir"])
    artifacts_dir = Path(paths["artifacts_dir"])
    output = args.output or artifacts_dir / f"protocol_manifest_{split_name}.json"

    positive_train_path = split_dir / f"positive_train_val_{split_name}.pt"
    positive_test_path = split_dir / f"positive_test_{split_name}.pt"
    negative_train_path = split_dir / f"negative_train_val_{split_name}.pt"
    negative_test_path = split_dir / f"negative_test_{split_name}.pt"
    positive_train = load_pairs(positive_train_path)
    positive_test = load_pairs(positive_test_path)
    negative_train = load_pairs(negative_train_path) if negative_train_path.exists() else []
    negative_test = load_pairs(negative_test_path) if negative_test_path.exists() else []

    positive_union = positive_train + positive_test
    test_reactions = sorted({reaction for reaction, _ in positive_test})
    test_enzymes = sorted({enzyme for _, enzyme in positive_test})
    all_reactions = sorted({reaction for reaction, _ in positive_union})
    all_enzymes = sorted({enzyme for _, enzyme in positive_union})

    split_files = {
        "positive_train_val": pair_summary(positive_train_path, "positive"),
        "positive_test": pair_summary(positive_test_path, "positive"),
        "negative_train_val": pair_summary(negative_train_path, "negative"),
        "negative_test": pair_summary(negative_test_path, "negative"),
    }
    negative_report_path = split_dir / f"negative_sampling_{split_name}.json"
    source_zip = split_dir / "downloads" / "enzyme_smi_split.zip"
    model_records = []
    for model in selected_models(config.get("models", []), args.models):
        name = model["name"]
        model_dir = artifacts_dir / "models" / name
        model_records.append({
            "name": name,
            "kind": model.get("kind"),
            "embedding": file_record(model_dir / "enzyme_embeddings.pt"),
            "metadata": metadata_record(model_dir / "enzyme_embeddings.json"),
        })

    evaluation = config.get("evaluation", {
        "candidate_scope": "test_positive_unique",
        "label_scope": "test_positive_pairs",
        "top_k_accuracy": "hit_rate_at_k",
        "map": "mean_average_precision_over_query_rows_with_at_least_one_positive",
        "mrr": "mean_reciprocal_rank_of_first_positive",
        "ndcg": "binary_relevance_with_rowwise_ideal_dcg",
    })
    manifest = {
        "schema_version": 1,
        "artifact": "reactzyme_protocol_manifest",
        "config": file_record(args.config),
        "split_name": split_name,
        "paths": {
            "split_dir": str(split_dir),
            "artifacts_dir": str(artifacts_dir),
            "results_dir": str(paths.get("results_dir", "")),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "configured_python_bin": runtime.get("python_bin"),
            "configured_device": runtime.get("device"),
        },
        "source_files": {
            "enzyme_smi_split_zip": file_record(source_zip),
            "mat_checkpoint": file_record(Path(paths["mat_checkpoint"])),
        },
        "split_files": split_files,
        "positive_overlap": overlap_summary(positive_train, positive_test),
        "negative_validation": {
            "train_val": negative_collision_summary(positive_union, negative_train),
            "test": negative_collision_summary(positive_union, negative_test),
            "sampling_report": metadata_record(negative_report_path),
        },
        "candidate_sets": {
            "test_positive_unique": {
                "reaction_count": len(test_reactions),
                "enzyme_count": len(test_enzymes),
                "label_pair_count": len(set(positive_test)),
            },
            "full_positive_union": {
                "reaction_count": len(all_reactions),
                "enzyme_count": len(all_enzymes),
                "label_pair_count": len(set(positive_union)),
                "status": "diagnostic_scope_not_used_by_current_training",
            },
        },
        "evaluation_contract": evaluation,
        "models": model_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote protocol manifest to {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
