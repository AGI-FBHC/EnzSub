from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

Pair = Tuple[str, str, float]

def normalise_reaction(reaction: str) -> str:
    return str(reaction).replace("*", "C")

def load_pair_file(path: str | Path, label: float) -> List[Pair]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a dictionary in {path}, got {type(payload).__name__}")
    pairs: List[Pair] = []
    for key, values in payload.items():
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError(f"Malformed ReactZyme record {key!r} in {path}")
        pairs.append((normalise_reaction(values[0]), str(values[1]), float(label)))
    return pairs

def load_split(split_dir: str | Path, split_name: str) -> Tuple[List[Pair], List[Pair]]:
    root = Path(split_dir)
    train = load_pair_file(root / f"positive_train_val_{split_name}.pt", 1.0)
    train += load_pair_file(root / f"negative_train_val_{split_name}.pt", 0.0)
    test = load_pair_file(root / f"positive_test_{split_name}.pt", 1.0)
    test += load_pair_file(root / f"negative_test_{split_name}.pt", 0.0)
    return train, test

def load_embedding_dict(path: str | Path, kind: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu")
    if isinstance(payload, Mapping) and "embeddings" in payload:
        payload = payload["embeddings"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{kind} embeddings at {path} must be a non-empty dictionary")
    result: Dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten().cpu()
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid {kind} embedding for key {key!r}")
        result[str(key)] = tensor
    dimensions = {value.numel() for value in result.values()}
    if len(dimensions) != 1:
        raise ValueError(f"{kind} embeddings have inconsistent dimensions: {sorted(dimensions)}")
    return result

def require_coverage(pairs: Iterable[Pair], molecules: Mapping[str, torch.Tensor], enzymes: Mapping[str, torch.Tensor]) -> None:
    missing_molecules = sorted({mol for mol, _, _ in pairs if mol not in molecules})
    missing_enzymes = sorted({seq for _, seq, _ in pairs if seq not in enzymes})
    if missing_molecules or missing_enzymes:
        items = []
        if missing_molecules:
            items.append(f"missing molecular embeddings={len(missing_molecules)}")
        if missing_enzymes:
            items.append(f"missing enzyme embeddings={len(missing_enzymes)}")
        raise KeyError("; ".join(items))

def stratified_split(pairs: Sequence[Pair], validation_fraction: float, seed: int) -> Tuple[List[Pair], List[Pair]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = random.Random(seed)
    train, validation = [], []
    for label in (0.0, 1.0):
        group = [pair for pair in pairs if pair[2] == label]
        rng.shuffle(group)
        cut = max(1, int(round(len(group) * validation_fraction)))
        validation.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def write_json(path: str | Path, payload: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
