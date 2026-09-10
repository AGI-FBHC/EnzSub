#!/usr/bin/env python3
"""Deterministic training/evaluation for cached EnzSub + Seq2Topt features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from enzsub_seq2topt import Seq2ToptFeatureModel, autocast_context, normalize_sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Cached feature files are immutable for a given representation.  Keeping the
# decoded tensors in-process avoids reopening one .pt file for every sample on
# every epoch and fold.  A CV run handles one representation at a time, so the
# cache is released when that process exits.
_FEATURE_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_cached_feature(feature_dir: Path, key: str) -> torch.Tensor:
    cache_key = (str(feature_dir), key)
    features = _FEATURE_CACHE.get(cache_key)
    if features is None:
        payload = torch_load(feature_dir / f"{key}.pt")
        features = payload["features"].float().contiguous()
        _FEATURE_CACHE[cache_key] = features
    return features

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass

def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)

def metrics(targets: Sequence[float], predictions: Sequence[float]) -> Dict[str, float]:
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    residual = target - prediction
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if denominator == 0 else float(1.0 - np.sum(residual ** 2) / denominator)
    return {"rmse": rmse, "mae": mae, "r2": r2}

class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        feature_dir: Path,
        max_seq_length: int,
        id_column: str,
        sequence_column: str,
        target_column: str,
        target_scale: float,
    ):
        self.table = table.reset_index(drop=True).copy()
        self.feature_dir = feature_dir
        self.max_seq_length = int(max_seq_length)
        self.id_column = id_column
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.target_scale = float(target_scale)
        required = {id_column, sequence_column, target_column}
        missing = required.difference(self.table.columns)
        if missing:
            raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

        self.sequences = [
            normalize_sequence(value, self.max_seq_length)
            for value in self.table[self.sequence_column].tolist()
        ]
        self.keys = [sequence_key(sequence) for sequence in self.sequences]
        self.lengths = [len(sequence) for sequence in self.sequences]
        missing_features = [
            key for key in set(self.keys) if not (self.feature_dir / f"{key}.pt").is_file()
        ]
        if missing_features:
            raise FileNotFoundError(
                f"Feature cache is missing {len(missing_features)} sequences. "
                "Run precompute_enzsub_features.py with the same data and cache directory."
            )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.table.iloc[index]
        features = load_cached_feature(self.feature_dir, self.keys[index])
        if features.ndim != 2 or features.shape[0] != self.lengths[index]:
            raise RuntimeError(
                f"Invalid cached feature shape for {self.keys[index]}: {tuple(features.shape)}"
            )
        return {
            "id": str(row[self.id_column]),
            "sequence": self.sequences[index],
            "features": features,
            "target": float(row[self.target_column]),
            "target_normalized": float(row[self.target_column]) / self.target_scale,
        }

def collate_cached(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    features = [item["features"] for item in batch]
    lengths = torch.tensor([value.shape[0] for value in features], dtype=torch.long)
    padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return {
        "ids": [item["id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "features": padded,
        "mask": mask,
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
        "targets_normalized": torch.tensor(
            [item["target_normalized"] for item in batch], dtype=torch.float32
        ),
    }

def split_train_validation(
    table: pd.DataFrame,
    validation_ratio: float,
    seed: int,
    split_unit: str,
    sequence_column: str,
    max_seq_length: int,
):
    rng = np.random.RandomState(seed)
    if split_unit == "row":
        indices = np.arange(len(table))
        rng.shuffle(indices)
        validation_size = int(len(indices) * validation_ratio)
        validation_indices = indices[:validation_size]
        train_indices = indices[validation_size:]
    elif split_unit == "sequence":
        keys = np.array(
            [
                sequence_key(normalize_sequence(value, max_seq_length))
                for value in table[sequence_column].tolist()
            ]
        )
        unique_keys = np.array(sorted(set(keys.tolist())))
        rng.shuffle(unique_keys)
        validation_size = max(1, int(len(unique_keys) * validation_ratio))
        validation_keys = set(unique_keys[:validation_size].tolist())
        validation_indices = np.flatnonzero(np.isin(keys, list(validation_keys)))
        train_indices = np.flatnonzero(~np.isin(keys, list(validation_keys)))
    else:
        raise ValueError(f"Unknown split_unit={split_unit!r}")
    return (
        table.iloc[train_indices].reset_index(drop=True),
        table.iloc[validation_indices].reset_index(drop=True),
    )

def sortish_batches(
    lengths: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    bucket_multiplier: int = 20,
) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(len(lengths)))
    rng = random.Random(seed)
    if not shuffle:
        return [
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        ]
    rng.shuffle(indices)
    bucket_size = max(batch_size, batch_size * bucket_multiplier)
    ordered: List[int] = []
    for start in range(0, len(indices), bucket_size):
        bucket = indices[start : start + bucket_size]
        bucket.sort(key=lambda index: lengths[index])
        ordered.extend(bucket)
    batches = [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]
    rng.shuffle(batches)
    return batches

def make_loader(
    dataset: CachedFeatureDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    batches = sortish_batches(dataset.lengths, batch_size, shuffle, seed)
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collate_cached,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

def evaluate(
    model: Seq2ToptFeatureModel,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    amp: bool,
):
    model.eval()
    all_ids: List[str] = []
    all_sequences: List[str] = []
    all_targets: List[float] = []
    all_predictions: List[float] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with autocast_context(device, enabled=amp and device.type == "cuda"):
                prediction = model(features, mask).squeeze(-1)
            prediction = prediction.float().cpu().numpy() * target_scale
            all_ids.extend(batch["ids"])
            all_sequences.extend(batch["sequences"])
            all_targets.extend(batch["targets"].numpy().tolist())
            all_predictions.extend(prediction.tolist())
    return (
        metrics(all_targets, all_predictions),
        all_ids,
        all_sequences,
        all_targets,
        all_predictions,
    )

def atomic_torch_save(payload: Dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Seq2Topt head on cached frozen EnzSub residue features."
    )
    parser.add_argument(
        "--train-csv", default=str(PROJECT_DIR / "data" / "Topt" / "train_os.csv")
    )
    parser.add_argument(
        "--test-csv", default=str(PROJECT_DIR / "data" / "Topt" / "test.csv")
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="uniprot_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--target-column", default="topt")
    parser.add_argument("--target-scale", type=float, default=120.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-unit", choices=["row", "sequence"], default="sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=320)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-rd", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--decay-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.target_scale <= 0:
        raise ValueError("--target-scale must be positive")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError("--effective-batch-size must be divisible by --batch-size")
    accumulation_steps = args.effective_batch_size // args.batch_size

    set_deterministic_seed(args.seed)
    device = resolve_device(args.device)
    train_csv = Path(args.train_csv).expanduser().resolve()
    test_csv = Path(args.test_csv).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing feature cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    feature_dir = cache_dir / "features"

    full_train = pd.read_csv(train_csv)
    test_table = pd.read_csv(test_csv)
    train_table, validation_table = split_train_validation(
        full_train,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        split_unit=args.split_unit,
        sequence_column=args.sequence_column,
        max_seq_length=int(manifest["max_seq_length"]),
    )

    dataset_kwargs = {
        "feature_dir": feature_dir,
        "max_seq_length": int(manifest["max_seq_length"]),
        "id_column": args.id_column,
        "sequence_column": args.sequence_column,
        "target_column": args.target_column,
        "target_scale": args.target_scale,
    }
    train_dataset = CachedFeatureDataset(train_table, **dataset_kwargs)
    validation_dataset = CachedFeatureDataset(validation_table, **dataset_kwargs)
    test_dataset = CachedFeatureDataset(test_table, **dataset_kwargs)

    model = Seq2ToptFeatureModel(
        encoder_dim=int(manifest["encoder_dim"]),
        head_dim=args.head_dim,
        window=args.window,
        n_head=args.n_head,
        n_RD=args.n_rd,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_interval, gamma=args.lr_decay
    )
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")

    run_config = {
        "arguments": vars(args),
        "model": model.task_config,
        "cache_manifest": manifest,
        "train_csv": {"path": str(train_csv), "sha256": sha256_file(train_csv)},
        "test_csv": {"path": str(test_csv), "sha256": sha256_file(test_csv)},
        "split_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "device": str(device),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    best_path = output_dir / "best_task_model.pt"
    history: List[Dict[str, float]] = []
    best_validation_rmse = math.inf
    print(
        f"device={device}, train={len(train_dataset)}, val={len(validation_dataset)}, "
        f"test={len(test_dataset)}, accumulation_steps={accumulation_steps}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            num_workers=args.num_workers,
        )
        optimizer.zero_grad(set_to_none=True)
        train_targets: List[float] = []
        train_predictions: List[float] = []
        total_batches = len(train_loader)

        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target_normalized = batch["targets_normalized"].to(device, non_blocking=True)
            group_start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, total_batches - group_start)

            with autocast_context(device, enabled=args.amp and device.type == "cuda"):
                prediction = model(features, mask).squeeze(-1)
                loss = F.mse_loss(prediction.float(), target_normalized.float())
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()

            should_step = (step + 1) % accumulation_steps == 0 or step + 1 == total_batches
            if should_step:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_targets.extend(batch["targets"].numpy().tolist())
            train_predictions.extend(
                (prediction.detach().float().cpu().numpy() * args.target_scale).tolist()
            )

        validation_loader = make_loader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
        )
        train_metrics = metrics(train_targets, train_predictions)
        validation_metrics, *_ = evaluate(
            model, validation_loader, device, args.target_scale, args.amp
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)

        if validation_metrics["rmse"] < best_validation_rmse:
            best_validation_rmse = validation_metrics["rmse"]
            atomic_torch_save(
                {
                    "format_version": 1,
                    "task_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "model_config": model.task_config,
                    "target_scale": args.target_scale,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "cache_identity": {
                        key: manifest.get(key)
                        for key in (
                            "encoder_type",
                            "model_mode",
                            "encoder_dim",
                            "max_seq_length",
                            "enzsub_checkpoint_sha256",
                        )
                    },
                },
                best_path,
            )

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(
            f"epoch={epoch:02d} "
            f"train_rmse={train_metrics['rmse']:.4f} "
            f"val_rmse={validation_metrics['rmse']:.4f} "
            f"val_r2={validation_metrics['r2']:.4f}"
        )
        scheduler.step()

    best_checkpoint = torch_load(best_path)
    model.load_state_dict(best_checkpoint["task_state_dict"], strict=True)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    test_metrics, ids, sequences, targets, predictions = evaluate(
        model, test_loader, device, args.target_scale, args.amp
    )
    result = {
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_metrics": best_checkpoint["validation_metrics"],
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(
        {
            args.id_column: ids,
            args.sequence_column: sequences,
            f"experimental_{args.target_column}": targets,
            f"predicted_{args.target_column}": predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)
    print(
        f"best_epoch={result['best_epoch']}, test_rmse={test_metrics['rmse']:.4f}, "
        f"test_mae={test_metrics['mae']:.4f}, test_r2={test_metrics['r2']:.4f}"
    )

if __name__ == "__main__":
    main()
