#!/usr/bin/env python3
"""Reproduce EC-number classification from released protein embeddings."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the EnzSub EC k=1 retrieval results."
    )
    parser.add_argument("--config", required=True, help="Path to the EC YAML config.")
    return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    if config.get("evaluation", {}).get("k") != 1:
        raise ValueError("EC reproduction uses the fixed paper setting k=1.")
    return config


def load_ec_labels(csv_path: Path) -> Tuple[Dict[str, List[str]], List[str]]:
    if not csv_path.is_file():
        raise FileNotFoundError("Missing EC label file: {}".format(csv_path))

    frame = pd.read_csv(csv_path, sep="\t", dtype=str, na_filter=False)
    if len(frame.columns) < 2:
        frame = pd.read_csv(csv_path, sep=",", dtype=str, na_filter=False)

    id_col = next(
        (name for name in ("Entry", "entry", "id", "ID", "seq_id", "uniprot_id")
         if name in frame.columns),
        frame.columns[0],
    )
    ec_col = next(
        (name for name in ("EC number", "ec_number", "EC", "ec", "label")
         if name in frame.columns),
        frame.columns[1],
    )

    id_to_ecs: Dict[str, List[str]] = {}
    ordered_ids: List[str] = []
    for seq_id, raw_ec in zip(frame[id_col], frame[ec_col]):
        labels = [item.strip() for item in raw_ec.split(";") if item.strip()]
        if labels:
            seq_id = str(seq_id)
            id_to_ecs[seq_id] = labels
            ordered_ids.append(seq_id)
    return id_to_ecs, ordered_ids


def torch_load(path: Path) -> object:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_embedding_set(path: Path, expected_ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    if not path.is_file():
        raise FileNotFoundError("Missing embedding file: {}".format(path))
    payload = torch_load(path)
    if not isinstance(payload, Mapping) or "embeddings" not in payload:
        raise ValueError("Invalid embedding file: {}".format(path))
    embeddings = payload["embeddings"]
    missing = [seq_id for seq_id in expected_ids if seq_id not in embeddings]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "{} is missing {}/{} expected IDs (for example: {}).".format(
                path, len(missing), len(expected_ids), preview
            )
        )

    tensors = [torch.as_tensor(embeddings[seq_id]).float() for seq_id in expected_ids]
    matrix = F.normalize(torch.stack(tensors), dim=1).numpy()
    return matrix, list(expected_ids)


def build_label_matrix(
    ids: Sequence[str], id_to_ecs: Mapping[str, Sequence[str]], ec_to_idx: Mapping[str, int]
) -> np.ndarray:
    matrix = np.zeros((len(ids), len(ec_to_idx)), dtype=np.int32)
    for row, seq_id in enumerate(ids):
        for ec in id_to_ecs.get(seq_id, []):
            matrix[row, ec_to_idx[ec]] = 1
    return matrix


def predict_k1(
    query: np.ndarray, gallery: np.ndarray, gallery_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros((query.shape[0], gallery_labels.shape[1]), dtype=np.int32)
    scores = np.zeros_like(predictions, dtype=np.float32)
    gallery_tensor = torch.from_numpy(gallery)
    chunk_size = 4096
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        similarities = torch.from_numpy(query[start:end]) @ gallery_tensor.T
        neighbors = similarities.argmax(dim=1).numpy()
        predictions[start:end] = gallery_labels[neighbors]
        scores[start:end] = gallery_labels[neighbors]
    return predictions, scores


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    metrics = {
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    sample_mcc = []
    for truth, prediction in zip(y_true, y_pred):
        if truth.sum() > 0 or prediction.sum() > 0:
            sample_mcc.append(matthews_corrcoef(truth, prediction))
    metrics["mcc"] = float(np.mean(sample_mcc)) if sample_mcc else 0.0

    positives = y_true.sum(axis=0)
    valid = (positives > 0) & (positives < y_true.shape[0])
    if valid.any():
        metrics["weighted_auc"] = float(
            roc_auc_score(y_true[:, valid], y_score[:, valid], average="weighted")
        )
        metrics["weighted_auprc"] = float(
            average_precision_score(y_true[:, valid], y_score[:, valid], average="weighted")
        )
    else:
        metrics["weighted_auc"] = 0.0
        metrics["weighted_auprc"] = 0.0
    return metrics


def compare_with_reference(
    results: pd.DataFrame, expected: Mapping[str, Mapping[str, float]], tolerance: float
) -> None:
    failures = []
    for state, datasets in expected.items():
        for dataset, expected_mcc in datasets.items():
            match = results[(results["state"] == state) & (results["dataset"] == dataset)]
            if len(match) != 1:
                failures.append("{} / {} is absent from results".format(state, dataset))
                continue
            observed = float(match.iloc[0]["mcc"])
            delta = observed - float(expected_mcc)
            status = "PASS" if abs(delta) <= tolerance else "FAIL"
            print(
                "  {:8s} {:9s} expected={:.6f} observed={:.6f} delta={:+.2e} {}".format(
                    state, dataset, expected_mcc, observed, delta, status
                )
            )
            if status == "FAIL":
                failures.append("{} / {} differs by {:.6g}".format(state, dataset, delta))
    if failures:
        raise RuntimeError("Reference check failed: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    base_dir = config_path.parent

    embedding_root = resolve(base_dir, config["embedding_root"])
    output_dir = resolve(base_dir, config["output_dir"])
    gallery_csv = resolve(base_dir, config["data"]["gallery"])
    query_paths = {
        name: resolve(base_dir, value) for name, value in config["data"]["queries"].items()
    }

    gallery_labels_by_id, gallery_ids = load_ec_labels(gallery_csv)
    gallery_ids = sorted(seq_id for seq_id in gallery_ids if "_" not in seq_id)
    query_ids: Dict[str, List[str]] = {}
    all_labels_by_id = dict(gallery_labels_by_id)
    for name, path in query_paths.items():
        labels, ids = load_ec_labels(path)
        query_ids[name] = ids
        all_labels_by_id.update(labels)

    ec_names = sorted({ec for labels in all_labels_by_id.values() for ec in labels})
    ec_to_idx = {ec: index for index, ec in enumerate(ec_names)}
    gallery_y = build_label_matrix(gallery_ids, all_labels_by_id, ec_to_idx)

    rows = []
    print("EC-number classification: fixed cosine k-NN (k=1)")
    for state, directory in config["states"].items():
        state_dir = embedding_root / directory
        gallery_x, loaded_gallery_ids = load_embedding_set(
            state_dir / "split100_all.pt", gallery_ids
        )
        if loaded_gallery_ids != gallery_ids:
            raise RuntimeError("Gallery order changed while loading {}.".format(state))
        for dataset, ids in query_ids.items():
            query_x, loaded_query_ids = load_embedding_set(
                state_dir / config["embedding_files"][dataset], ids
            )
            query_y = build_label_matrix(loaded_query_ids, all_labels_by_id, ec_to_idx)
            prediction, score = predict_k1(query_x, gallery_x, gallery_y)
            metrics = compute_metrics(query_y, prediction, score)
            rows.append(
                {
                    "backbone": config["backbone"],
                    "state": state,
                    "dataset": dataset,
                    "k": 1,
                    "gallery_size": len(gallery_ids),
                    "query_size": len(loaded_query_ids),
                    **metrics,
                }
            )
            print(
                "  {:8s} {:9s} n={:<4d} MCC={:.6f} weighted-F1={:.6f}".format(
                    state,
                    dataset,
                    len(loaded_query_ids),
                    metrics["mcc"],
                    metrics["weighted_f1"],
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results.csv", index=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    expected = config.get("expected_mcc", {})
    if expected:
        print("\nReference check")
        compare_with_reference(
            results, expected, float(config.get("reference_tolerance", 1e-6))
        )
    print("\nResults written to {}".format(output_dir / "results.csv"))


if __name__ == "__main__":
    main()
