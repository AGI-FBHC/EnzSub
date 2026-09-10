#!/usr/bin/env python3
"""Audit, replay, and statistically compare completed ReactZyme runs.

This module is deliberately post-hoc only: it never trains a model and never
modifies an existing checkpoint, metric JSON, split, or embedding artifact.
All new files live under ``<results-dir>/posthoc_analysis``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from common import load_embedding_dict, load_split, require_coverage
from train_and_evaluate import EnzGFMReactZymeHead

QUERY_METRICS = (
    "ap",
    "reciprocal_rank",
    "top_1_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "top_10_accuracy",
    "top_20_accuracy",
    "top_50_accuracy",
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_20",
    "ndcg_at_50",
)
SUMMARY_TO_QUERY_METRIC = {"map": "ap", "mrr_first_hit": "reciprocal_rank"}
SUMMARY_TO_QUERY_METRIC.update({f"top_{k}_accuracy": f"top_{k}_accuracy" for k in (1, 3, 5, 10, 20, 50)})
SUMMARY_TO_QUERY_METRIC.update({f"ndcg_at_{k}": f"ndcg_at_{k}" for k in (1, 3, 5, 10, 20, 50)})

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def file_record(path: Path, with_hash: bool = True) -> Dict[str, object]:
    result: Dict[str, object] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        result["bytes"] = path.stat().st_size
        if with_hash:
            result["sha256"] = sha256_file(path)
    return result

def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def select_models(config: Mapping, names: Sequence[str] | None) -> List[Mapping]:
    configured = {entry["name"]: entry for entry in config["models"]}
    selected = list(names or config.get("comparison_models", configured.keys()))
    missing = sorted(set(selected) - set(configured))
    if missing:
        raise ValueError(f"Unknown configured models: {missing}")
    return [configured[name] for name in selected]

def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)

def command_audit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    expected_seeds = [int(seed) for seed in config["training"]["seeds"]]
    artifact_dir = Path(config["paths"]["artifacts_dir"])
    output = args.output or results_dir / "posthoc_analysis" / "protocol_and_results_manifest.json"
    models = select_models(config, args.models)

    model_records = []
    issues = []
    for model in models:
        name = model["name"]
        model_dir = results_dir / name
        summary_path = model_dir / "summary.json"
        record: Dict[str, object] = {
            "name": name,
            "kind": model.get("kind"),
            "embedding": file_record(artifact_dir / "models" / name / "enzyme_embeddings.pt"),
            "summary": file_record(summary_path),
            "runs": [],
        }
        if not summary_path.exists():
            issues.append(f"{name}: missing summary.json")
            model_records.append(record)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = summary.get("runs", [])
        observed_seeds = [int(run.get("seed", -1)) for run in runs]
        record["summary_seed_order"] = observed_seeds
        record["expected_seeds"] = expected_seeds
        record["summary_seed_set_matches_config"] = sorted(observed_seeds) == sorted(expected_seeds)
        if not record["summary_seed_set_matches_config"]:
            issues.append(f"{name}: summary seed set {observed_seeds} does not match {expected_seeds}")
        for seed in expected_seeds:
            run_dir = model_dir / f"seed_{seed}"
            metrics_path, checkpoint_path = run_dir / "metrics.json", run_dir / "model.pt"
            run_record: Dict[str, object] = {
                "seed": seed,
                "metrics": file_record(metrics_path),
                "checkpoint": file_record(checkpoint_path),
            }
            if not metrics_path.exists() or not checkpoint_path.exists():
                issues.append(f"{name}/seed_{seed}: missing metrics or checkpoint")
            else:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                run_record["metrics_seed_matches_directory"] = int(payload.get("seed", -1)) == seed
                run_record["directions"] = sorted(key for key in ("enzyme_to_reaction", "reaction_to_enzyme") if key in payload)
                run_record["candidate_counts"] = payload.get("candidate_counts")
                if not run_record["metrics_seed_matches_directory"]:
                    issues.append(f"{name}/seed_{seed}: metrics seed mismatch")
                if len(run_record["directions"]) != 2:
                    issues.append(f"{name}/seed_{seed}: missing retrieval direction")
            record["runs"].append(run_record)
        model_records.append(record)

    manifest = {
        "schema_version": 1,
        "artifact": "reactzyme_posthoc_protocol_and_results_manifest",
        "analysis_boundary": "audit only; no model, split, embedding, or core metric file was modified",
        "config": file_record(args.config),
        "protocol_manifest": file_record(artifact_dir / f"protocol_manifest_{config['runtime'].get('split_name', 'seq_smi')}.json"),
        "expected_seeds": expected_seeds,
        "models": model_records,
        "issues": issues,
        "status": "ready_for_posthoc_replay" if not issues else "audit_findings_require_review",
    }
    write_json(output, manifest)
    print(f"Wrote {output}; issues={len(issues)}")
    return 0

def build_test_matrices(split_dir: Path, split_name: str, artifacts_dir: Path, model_name: str):
    _, test_pairs = load_split(split_dir, split_name)
    positives = [(mol, enzyme, label) for mol, enzyme, label in test_pairs if label == 1.0]
    molecules = load_embedding_dict(artifacts_dir / "mat_reaction_embeddings.pt", "MAT reaction")
    enzymes = load_embedding_dict(artifacts_dir / "models" / model_name / "enzyme_embeddings.pt", f"{model_name} enzyme")
    require_coverage(positives, molecules, enzymes)
    molecule_ids = sorted({molecule for molecule, _, _ in positives})
    enzyme_ids = sorted({enzyme for _, enzyme, _ in positives})
    molecule_index = {value: index for index, value in enumerate(molecule_ids)}
    enzyme_index = {value: index for index, value in enumerate(enzyme_ids)}
    labels = torch.zeros((len(enzyme_ids), len(molecule_ids)), dtype=torch.bool)
    for molecule, enzyme, _ in positives:
        labels[enzyme_index[enzyme], molecule_index[molecule]] = True
    return molecules, enzymes, molecule_ids, enzyme_ids, labels

@torch.no_grad()
def score_matrix(
    model: torch.nn.Module,
    molecules: Mapping[str, torch.Tensor],
    enzymes: Mapping[str, torch.Tensor],
    molecule_ids: Sequence[str],
    enzyme_ids: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    molecule_tensor = torch.stack([molecules[key] for key in molecule_ids]).to(device)
    enzyme_tensor = torch.stack([enzymes[key] for key in enzyme_ids]).to(device)
    rows = []
    model.eval()
    for index, enzyme in enumerate(enzyme_tensor, start=1):
        chunks = []
        for start in range(0, len(molecule_tensor), batch_size):
            current = molecule_tensor[start : start + batch_size]
            chunks.append(model(current, enzyme.expand(current.size(0), -1)).detach().cpu())
        rows.append(torch.cat(chunks))
        if index % 250 == 0 or index == len(enzyme_tensor):
            print(f"Scored {index}/{len(enzyme_tensor)} enzyme queries", flush=True)
    return torch.stack(rows)

def metric_row(scores: torch.Tensor, labels: torch.Tensor, direction: str, query_id: str, seed: int, model_name: str) -> Dict[str, object]:
    order = torch.argsort(scores, descending=True, stable=True)
    relevance = labels[order].to(torch.float32)
    positives = int(relevance.sum().item())
    if positives == 0:
        raise ValueError("Post-hoc replay encountered a query with no positive label")
    ranks = torch.arange(1, relevance.numel() + 1, dtype=torch.float32)
    precision = torch.cumsum(relevance, dim=0) / ranks
    first = int(torch.where(relevance > 0)[0][0].item())
    row: Dict[str, object] = {
        "model": model_name,
        "seed": seed,
        "direction": direction,
        "query_sha256": sha256_text(query_id),
        "positive_count": positives,
        "candidate_count": int(relevance.numel()),
        "ap": float((precision * relevance).sum().item() / positives),
        "reciprocal_rank": 1.0 / (first + 1),
    }
    for k in (1, 3, 5, 10, 20, 50):
        limit = min(k, relevance.numel())
        top = relevance[:limit]
        row[f"top_{k}_accuracy"] = float(bool(top.any()))
        discount = torch.log2(torch.arange(2, limit + 2, dtype=torch.float32))
        dcg = float((top / discount).sum().item())
        ideal = min(positives, limit)
        ideal_discount = torch.log2(torch.arange(2, ideal + 2, dtype=torch.float32))
        idcg = float((torch.ones(ideal, dtype=torch.float32) / ideal_discount).sum().item())
        row[f"ndcg_at_{k}"] = dcg / idcg
    return row

def command_replay(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    artifacts_dir, split_dir = Path(config["paths"]["artifacts_dir"]), Path(config["paths"]["split_dir"])
    split_name = config["runtime"].get("split_name", "seq_smi")
    device = torch.device(args.device if args.device and torch.cuda.is_available() else "cpu")
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    models = select_models(config, args.models)
    base_output = results_dir / "posthoc_analysis" / "per_query"
    for spec in models:
        name = spec["name"]
        molecules, enzymes, molecule_ids, enzyme_ids, labels = build_test_matrices(split_dir, split_name, artifacts_dir, name)
        molecule_dim = next(iter(molecules.values())).numel()
        enzyme_dim = next(iter(enzymes.values())).numel()
        for seed in seeds:
            output = base_output / name / f"seed_{seed}" / "per_query.csv"
            metadata = output.with_suffix(".json")
            if args.resume and output.exists() and metadata.exists():
                print(f"Reusing completed post-hoc replay: {name}/seed_{seed}")
                continue
            checkpoint_path = results_dir / name / f"seed_{seed}" / "model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model = EnzGFMReactZymeHead(molecule_dim, enzyme_dim).to(device)
            model.load_state_dict(state_dict, strict=True)
            print(f"Replaying {name}/seed_{seed} on {device}", flush=True)
            scores = score_matrix(model, molecules, enzymes, molecule_ids, enzyme_ids, device, args.batch_size)
            rows = []
            for index, enzyme in enumerate(enzyme_ids):
                rows.append(metric_row(scores[index], labels[index], "enzyme_to_reaction", enzyme, seed, name))
            for index, molecule in enumerate(molecule_ids):
                rows.append(metric_row(scores[:, index], labels[:, index], "reaction_to_enzyme", molecule, seed, name))
            fields = ("model", "seed", "direction", "query_sha256", "positive_count", "candidate_count", *QUERY_METRICS)
            write_csv(output, rows, fields)
            write_json(metadata, {
                "schema_version": 1,
                "artifact": "reactzyme_posthoc_per_query_metrics",
                "model": name,
                "seed": seed,
                "checkpoint": file_record(checkpoint_path),
                "candidate_counts": {"enzymes": len(enzyme_ids), "reactions": len(molecule_ids)},
                "row_counts": {"enzyme_to_reaction": len(enzyme_ids), "reaction_to_enzyme": len(molecule_ids)},
                "split_name": split_name,
                "analysis_boundary": "replayed saved checkpoint only; no training was performed",
            })
            del model, scores
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0

def t_distribution_summary(deltas: Sequence[float]) -> Dict[str, float | None]:
    values = np.asarray(deltas, dtype=float)
    if values.size < 2:
        return {"mean_delta": float(values.mean()), "sd_delta": 0.0, "ci95_low": None, "ci95_high": None, "paired_t": None, "paired_t_p_two_sided": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(values.size)
    try:
        from scipy import stats
        t_value = mean / se if se else (math.inf if mean else 0.0)
        p_value = float(2 * stats.t.sf(abs(t_value), df=values.size - 1)) if math.isfinite(t_value) else 0.0
        critical = float(stats.t.ppf(0.975, df=values.size - 1))
        low, high = mean - critical * se, mean + critical * se
    except ImportError:
        t_value, p_value, low, high = None, None, None, None
    return {"mean_delta": mean, "sd_delta": sd, "ci95_low": low, "ci95_high": high, "paired_t": t_value, "paired_t_p_two_sided": p_value}

def exact_sign_flip_pvalue(deltas: Sequence[float]) -> float:
    values = np.asarray(deltas, dtype=float)
    observed = abs(float(values.mean()))
    null_means = [abs(float(np.mean(np.asarray(signs) * values))) for signs in itertools.product((-1.0, 1.0), repeat=values.size)]
    return float(sum(value >= observed - 1e-15 for value in null_means) / len(null_means))

def holm_adjust(records: List[Dict[str, object]], key: str) -> None:
    valid = [(index, float(record[key])) for index, record in enumerate(records) if record.get(key) is not None]
    valid.sort(key=lambda item: item[1])
    count, running = len(valid), 0.0
    for rank, (index, pvalue) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * pvalue))
        records[index][f"{key}_holm"] = running

def read_seed_metrics(results_dir: Path, model: str) -> Dict[int, Dict]:
    summary_path = results_dir / model / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    result = {}
    for run in payload["runs"]:
        result[int(run["seed"])] = run
    return result

def query_bootstrap(
    baseline_dir: Path,
    model_dir: Path,
    seeds: Sequence[int],
    direction: str,
    metric: str,
    reps: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    per_seed = []
    for seed in seeds:
        def load_rows(root: Path) -> Dict[str, float]:
            path = root / f"seed_{seed}" / "per_query.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                return {row["query_sha256"]: float(row[metric]) for row in csv.DictReader(handle) if row["direction"] == direction}
        left, right = load_rows(baseline_dir), load_rows(model_dir)
        if set(left) != set(right):
            raise ValueError(f"Query mismatch for seed={seed}, direction={direction}, metric={metric}")
        per_seed.append(np.asarray([right[key] - left[key] for key in sorted(left)], dtype=float))
    effects = np.empty(reps, dtype=float)
    for index in range(reps):
        selected = rng.integers(0, len(per_seed), size=len(per_seed))
        effects[index] = float(np.mean([np.mean(per_seed[item][rng.integers(0, len(per_seed[item]), size=len(per_seed[item]))]) for item in selected]))
    observed = float(np.mean([values.mean() for values in per_seed]))
    return {
        "mean_delta": observed,
        "ci95_low": float(np.quantile(effects, 0.025)),
        "ci95_high": float(np.quantile(effects, 0.975)),
        "n_seeds": len(per_seed),
        "query_count_per_seed": len(per_seed[0]),
        "bootstrap_reps": reps,
    }

def command_statistics(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    results_dir = Path(config["paths"]["results_dir"])
    output_dir = args.output_dir or results_dir / "posthoc_analysis" / "statistics"
    selected = [spec["name"] for spec in select_models(config, args.models)]
    baseline = args.baseline
    if baseline not in selected:
        raise ValueError("Baseline must be included in --models (or comparison_models)")
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    base_runs = read_seed_metrics(results_dir, baseline)
    comparisons = [name for name in selected if name != baseline]
    rows: List[Dict[str, object]] = []
    for model in comparisons:
        model_runs = read_seed_metrics(results_dir, model)
        if sorted(model_runs) != sorted(base_runs) or sorted(base_runs) != sorted(seeds):
            raise ValueError(f"Seeds do not match for comparison {model} vs {baseline}")
        for direction in ("enzyme_to_reaction", "reaction_to_enzyme"):
            metric_names = sorted(base_runs[seeds[0]][direction])
            for metric in metric_names:
                baseline_values = [float(base_runs[seed][direction][metric]) for seed in seeds]
                model_values = [float(model_runs[seed][direction][metric]) for seed in seeds]
                values = [right - left for left, right in zip(baseline_values, model_values)]
                row: Dict[str, object] = {
                    "comparison": f"{model}_minus_{baseline}", "model": model, "baseline": baseline,
                    "direction": direction, "metric": metric, "n_seeds": len(seeds),
                    "seeds": ";".join(map(str, seeds)), "baseline_mean": statistics.mean(baseline_values),
                    "model_mean": statistics.mean(model_values), "exact_signflip_p_two_sided": exact_sign_flip_pvalue(values),
                    "is_primary": metric == "map",
                }
                row.update(t_distribution_summary(values))
                rows.append(row)
    primary = [row for row in rows if row["is_primary"]]
    holm_adjust(primary, "paired_t_p_two_sided")
    for row in rows:
        row.setdefault("paired_t_p_two_sided_holm", None)
    fields = ("comparison", "model", "baseline", "direction", "metric", "n_seeds", "seeds", "baseline_mean", "model_mean", "mean_delta", "sd_delta", "ci95_low", "ci95_high", "paired_t", "paired_t_p_two_sided", "paired_t_p_two_sided_holm", "exact_signflip_p_two_sided", "is_primary")
    write_csv(output_dir / "seed_level_statistics.csv", rows, fields)

    query_rows = []
    per_query_root = results_dir / "posthoc_analysis" / "per_query"
    rng = np.random.default_rng(args.bootstrap_seed)
    if args.query_bootstrap:
        for model in comparisons:
            for direction in ("enzyme_to_reaction", "reaction_to_enzyme"):
                for summary_metric in ("map",):
                    query_metric = SUMMARY_TO_QUERY_METRIC[summary_metric]
                    try:
                        result = query_bootstrap(per_query_root / baseline, per_query_root / model, seeds, direction, query_metric, args.bootstrap_reps, rng)
                        query_rows.append({"comparison": f"{model}_minus_{baseline}", "model": model, "baseline": baseline, "direction": direction, "summary_metric": summary_metric, "query_metric": query_metric, **result})
                    except FileNotFoundError:
                        print(f"Query bootstrap deferred: per-query replay missing for {model} or {baseline}", flush=True)
                        break
    if query_rows:
        write_csv(output_dir / "query_level_hierarchical_bootstrap.csv", query_rows, ("comparison", "model", "baseline", "direction", "summary_metric", "query_metric", "mean_delta", "ci95_low", "ci95_high", "n_seeds", "query_count_per_seed", "bootstrap_reps"))
    write_json(output_dir / "statistical_manifest.json", {
        "schema_version": 1,
        "analysis_unit": {
            "seed_level": "matched random seed across representations",
            "query_level": "matched retrieval query nested within matched random seed",
        },
        "delta_definition": "model metric minus baseline metric; positive favours model",
        "primary_family": "MAP separately for enzyme-to-reaction and reaction-to-enzyme; Holm correction across all configured model contrasts and directions",
        "secondary_metrics": "Top-k, NDCG and MRR are descriptive unless a separate multiplicity family is declared",
        "seed_level_tests": "two-sided paired t test and exact two-sided sign-flip test; matched seed count is reported in each result row",
        "query_level_interval": "two-stage bootstrap resampling matched seeds then matched queries; reports interval rather than a population-wide p value",
        "bootstrap_reps": args.bootstrap_reps if args.query_bootstrap else None,
        "bootstrap_seed": args.bootstrap_seed if args.query_bootstrap else None,
    })
    print(f"Wrote seed-level statistics to {output_dir}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Hash and validate completed result inputs without altering them")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--models", nargs="+")
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=command_audit)

    replay = commands.add_parser("replay", help="Replay saved checkpoints and write per-query retrieval metrics")
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--models", nargs="+", required=True)
    replay.add_argument("--device", default="cuda")
    replay.add_argument("--batch-size", type=int, default=512)
    replay.add_argument("--resume", action="store_true")
    replay.set_defaults(func=command_replay)

    stats = commands.add_parser("statistics", help="Run seed-level and optional hierarchical query-level paired analyses")
    stats.add_argument("--config", type=Path, required=True)
    stats.add_argument("--models", nargs="+")
    stats.add_argument("--baseline", default="reactzyme_esm2_650m")
    stats.add_argument("--output-dir", type=Path)
    stats.add_argument("--query-bootstrap", action="store_true")
    stats.add_argument("--bootstrap-reps", type=int, default=10000)
    stats.add_argument("--bootstrap-seed", type=int, default=20260907)
    stats.set_defaults(func=command_statistics)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
