#!/usr/bin/env python3
"""Grouped five-fold OOF evaluation for cached Seq2Topt representations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from enzsub_seq2topt import Seq2ToptFeatureModel, autocast_context, normalize_sequence
from run_train_enzsub import (
    CachedFeatureDataset,
    atomic_torch_save,
    evaluate,
    make_grad_scaler,
    make_loader,
    metrics,
    resolve_device,
    sequence_key,
    set_deterministic_seed,
    sha256_file,
    torch_load,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

def sequence_groups(table: pd.DataFrame, sequence_column: str, max_seq_length: int) -> np.ndarray:
    return np.asarray(
        [sequence_key(normalize_sequence(value, max_seq_length)) for value in table[sequence_column]],
        dtype=object,
    )

def build_sequence_folds(
    table: pd.DataFrame,
    n_folds: int,
    seed: int,
    sequence_column: str,
    max_seq_length: int,
) -> np.ndarray:
    """Assign rows to balanced folds without splitting duplicate sequences."""
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    groups = sequence_groups(table, sequence_column, max_seq_length)
    counts = pd.Series(groups).value_counts().to_dict()
    unique_groups = np.asarray(sorted(counts), dtype=object)
    np.random.RandomState(seed).shuffle(unique_groups)
    loads = np.zeros(n_folds, dtype=np.int64)
    group_to_fold: Dict[str, int] = {}
    for group in sorted(unique_groups.tolist(), key=lambda value: -counts[value]):
        fold = int(np.argmin(loads))
        group_to_fold[str(group)] = fold
        loads[fold] += int(counts[group])
    fold_ids = np.asarray([group_to_fold[str(group)] for group in groups], dtype=np.int64)
    if set(fold_ids.tolist()) != set(range(n_folds)):
        raise RuntimeError(f"Invalid fold assignment: {sorted(set(fold_ids.tolist()))}")
    return fold_ids

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped five-fold CV on cached Seq2Topt features.")
    parser.add_argument("--train-csv", default=str(PROJECT_DIR / "data" / "Topt" / "train_os.csv"))
    parser.add_argument("--test-csv", default=str(PROJECT_DIR / "data" / "Topt" / "test.csv"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id-column", default="uniprot_id")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--target-column", default="topt")
    parser.add_argument("--target-scale", type=float, default=120.0)
    parser.add_argument("--folds", type=int, default=5)
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

def train_fold(args, fold, train_table, validation_table, test_table, manifest, feature_dir, output_dir, device):
    set_deterministic_seed(args.seed + fold)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_interval, args.lr_decay)
    scaler = make_grad_scaler(enabled=args.amp and device.type == "cuda")
    accumulation_steps = args.effective_batch_size // args.batch_size
    fold_dir = output_dir / f"fold_{fold + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_path = fold_dir / "best_task_model.pt"
    best_rmse = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loader = make_loader(train_dataset, args.batch_size, True, args.seed + fold * 1000 + epoch, args.num_workers)
        optimizer.zero_grad(set_to_none=True)
        train_targets, train_predictions = [], []
        for step, batch in enumerate(loader):
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets_normalized"].to(device, non_blocking=True)
            start = (step // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(loader) - start)
            with autocast_context(device, enabled=args.amp and device.type == "cuda"):
                prediction = model(features, mask).squeeze(-1)
                loss = F.mse_loss(prediction.float(), targets.float()) / group_size
            scaler.scale(loss).backward()
            should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_targets.extend(batch["targets"].numpy().tolist())
            train_predictions.extend((prediction.detach().float().cpu().numpy() * args.target_scale).tolist())

        validation_loader = make_loader(validation_dataset, args.eval_batch_size, False, args.seed, args.num_workers)
        train_metrics = metrics(train_targets, train_predictions)
        validation_metrics, *_ = evaluate(model, validation_loader, device, args.target_scale, args.amp)
        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
        if validation_metrics["rmse"] < best_rmse:
            best_rmse = validation_metrics["rmse"]
            atomic_torch_save({
                "format_version": 1,
                "task_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": model.task_config,
                "target_scale": args.target_scale,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "cache_identity": {key: manifest.get(key) for key in (
                    "encoder_type", "model_mode", "encoder_dim", "max_seq_length", "enzsub_checkpoint_sha256"
                )},
            }, best_path)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        print(f"fold={fold + 1}/{args.folds} epoch={epoch:02d} val_rmse={validation_metrics['rmse']:.4f}")
        scheduler.step()

    checkpoint = torch_load(best_path)
    model.load_state_dict(checkpoint["task_state_dict"], strict=True)
    validation_metrics, val_ids, val_sequences, val_targets, val_predictions = evaluate(
        model, validation_loader, device, args.target_scale, args.amp
    )
    test_loader = make_loader(test_dataset, args.eval_batch_size, False, args.seed, args.num_workers)
    test_metrics, test_ids, test_sequences, test_targets, test_predictions = evaluate(
        model, test_loader, device, args.target_scale, args.amp
    )
    target_name = f"experimental_{args.target_column}"
    prediction_name = f"predicted_{args.target_column}"
    pd.DataFrame({
        "fold": fold + 1, args.id_column: val_ids, args.sequence_column: val_sequences,
        target_name: val_targets, prediction_name: val_predictions,
    }).to_csv(fold_dir / "validation_predictions.csv", index=False)
    pd.DataFrame({
        "fold": fold + 1, args.id_column: test_ids, args.sequence_column: test_sequences,
        target_name: test_targets, prediction_name: test_predictions,
    }).to_csv(fold_dir / "test_predictions.csv", index=False)
    result = {
        "fold": fold + 1,
        "n_train": len(train_dataset), "n_validation": len(validation_dataset), "n_test": len(test_dataset),
        "best_epoch": int(checkpoint["epoch"]),
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
    }
    (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    val_frame = pd.DataFrame({
        "row_index": validation_table["__row_index"].to_numpy(), "fold": fold + 1,
        args.id_column: val_ids, args.sequence_column: val_sequences,
        target_name: val_targets, prediction_name: val_predictions,
    })
    test_frame = pd.DataFrame({
        "fold": fold + 1, args.id_column: test_ids, args.sequence_column: test_sequences,
        target_name: test_targets, prediction_name: test_predictions,
    })
    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, val_frame, test_frame

def main() -> None:
    args = parse_args()
    if args.target_scale <= 0 or args.folds < 2:
        raise ValueError("target scale must be positive and folds must be at least 2")
    if args.effective_batch_size % args.batch_size != 0:
        raise ValueError("--effective-batch-size must be divisible by --batch-size")
    set_deterministic_seed(args.seed)
    device = resolve_device(args.device)
    train_csv = Path(args.train_csv).expanduser().resolve()
    test_csv = Path(args.test_csv).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("model_mode") not in {"published_base", "base", "cpt"}:
        raise ValueError(f"Unsupported cache model_mode={manifest.get('model_mode')!r}")
    feature_dir = cache_dir / "features"
    train_table = pd.read_csv(train_csv).reset_index(drop=True)
    train_table["__row_index"] = np.arange(len(train_table), dtype=np.int64)
    test_table = pd.read_csv(test_csv).reset_index(drop=True)
    fold_ids = build_sequence_folds(train_table, args.folds, args.seed, args.sequence_column, int(manifest["max_seq_length"]))
    groups = sequence_groups(train_table, args.sequence_column, int(manifest["max_seq_length"]))
    if any(len(set(fold_ids[groups == group].tolist())) != 1 for group in set(groups)):
        raise RuntimeError("A normalized sequence was assigned to multiple folds")
    (output_dir / "run_config.json").write_text(json.dumps({
        "arguments": vars(args), "cache_manifest": manifest,
        "train_csv": {"path": str(train_csv), "sha256": sha256_file(train_csv)},
        "test_csv": {"path": str(test_csv), "sha256": sha256_file(test_csv)},
        "device": str(device), "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "split": {
            "type": "grouped_sequence_kfold", "folds": args.folds, "seed": args.seed,
            "train_rows": len(train_table), "test_rows": len(test_table),
            "unique_train_sequences": int(len(set(groups.tolist()))),
            "fold_row_counts": [int(np.sum(fold_ids == fold)) for fold in range(args.folds)],
        },
    }, indent=2, sort_keys=True) + "\n")

    fold_results, validation_frames, test_frames = [], [], []
    for fold in range(args.folds):
        validation_mask = fold_ids == fold
        result, val_frame, test_frame = train_fold(
            args, fold, train_table.loc[~validation_mask].reset_index(drop=True),
            train_table.loc[validation_mask].reset_index(drop=True), test_table,
            manifest, feature_dir, output_dir, device,
        )
        fold_results.append(result)
        validation_frames.append(val_frame)
        test_frames.append(test_frame)

    target_name = f"experimental_{args.target_column}"
    prediction_name = f"predicted_{args.target_column}"
    oof = pd.concat(validation_frames, ignore_index=True).sort_values("row_index")
    test_by_fold = pd.concat(test_frames, ignore_index=True)
    test_ensemble = test_by_fold.groupby([args.id_column, args.sequence_column, target_name], as_index=False)[prediction_name].mean()
    oof_metrics = metrics(oof[target_name], oof[prediction_name])
    test_ensemble_metrics = metrics(test_ensemble[target_name], test_ensemble[prediction_name])
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_by_fold.to_csv(output_dir / "test_predictions_by_fold.csv", index=False)
    test_ensemble.to_csv(output_dir / "test_predictions_ensemble.csv", index=False)
    pd.DataFrame(fold_results).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps({
        "encoder_type": manifest["encoder_type"], "representation": manifest["model_mode"],
        "fold_metrics": fold_results, "oof_metrics": oof_metrics,
        "test_ensemble_metrics": test_ensemble_metrics,
    }, indent=2, sort_keys=True) + "\n")
    print(f"representation={manifest['model_mode']} encoder={manifest['encoder_type']} "
          f"OOF rmse={oof_metrics['rmse']:.4f} mae={oof_metrics['mae']:.4f} r2={oof_metrics['r2']:.4f}")
    print(f"test_ensemble rmse={test_ensemble_metrics['rmse']:.4f} "
          f"mae={test_ensemble_metrics['mae']:.4f} r2={test_ensemble_metrics['r2']:.4f}")

if __name__ == "__main__":
    main()
