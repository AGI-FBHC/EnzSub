#!/usr/bin/env python3
"""Audit SUB-train homology and re-evaluate cached ReactZyme E2R queries.

This is a post-hoc Stage-B analysis.  It never trains a model, changes a
checkpoint, regenerates embeddings, or edits a ReactZyme split.  New outputs
are written below ``<results_dir>/posthoc_analysis/sub_train_ood_stage_b``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from common import load_pair_file
from posthoc_analysis import (
    exact_sign_flip_pvalue,
    holm_adjust,
    sha256_file,
    sha256_text,
    t_distribution_summary,
    write_csv,
    write_json,
)

E2R_METRICS = ("ap", "top_1_accuracy", "top_10_accuracy", "ndcg_at_10")
SUMMARY_NAMES = {"ap": "map", "top_1_accuracy": "top_1_accuracy", "top_10_accuracy": "top_10_accuracy", "ndcg_at_10": "ndcg_at_10"}

def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def stage_spec(config: Mapping) -> Mapping:
    spec = config.get("sub_train_ood_stage_b")
    if not isinstance(spec, Mapping):
        raise ValueError("Config requires a sub_train_ood_stage_b object")
    return spec

def output_root(config: Mapping) -> Path:
    return Path(config["paths"]["results_dir"]) / "posthoc_analysis" / "sub_train_ood_stage_b"

def normalise_sequence(sequence: object) -> str:
    return "".join(str(sequence).upper().split())

def write_fasta(records: Sequence[Mapping[str, object]], path: Path, prefix: str) -> Dict[str, Mapping[str, object]]:
    mapping: Dict[str, Mapping[str, object]] = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            proxy = f"{prefix}{index}"
            if proxy in mapping:
                raise RuntimeError("Proxy-ID collision")
            mapping[proxy] = record
            handle.write(f">{proxy}\n{record['aligned_sequence']}\n")
    return mapping

def build_sub_train_reference(spec: Mapping, root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    source = Path(spec["source_csv"])
    frame = pd.read_csv(source)
    needed = (spec["id_column"], spec["sequence_column"], spec["smiles_column"])
    missing = sorted(set(needed).difference(frame.columns))
    if missing:
        raise ValueError(f"SUB source lacks columns: {missing}")
    rows = []
    for protein_id, group in frame.groupby(spec["id_column"], sort=True):
        raw = group[spec["sequence_column"]].iloc[0]
        sequence = normalise_sequence(raw)
        smiles = group[spec["smiles_column"]].dropna().astype(str).str.strip()
        if not sequence or smiles[smiles.ne("")].empty:
            continue
        rows.append({"protein_id": str(protein_id), "aligned_sequence": sequence[: int(spec["max_sequence_length"])], "raw_length": len(sequence)})
    if not rows:
        raise RuntimeError("No eligible OED enzymes")
    rng = np.random.RandomState(int(spec["split_seed"]))
    indices = np.arange(len(rows)); rng.shuffle(indices)
    train_count = int(len(rows) * float(spec["train_ratio"]))
    train = [rows[index] for index in indices[:train_count]]
    train.sort(key=lambda row: str(row["protein_id"]))
    reference = root / "reference" / "sub_train_enzymes.csv"
    write_csv(reference, train, ("protein_id", "raw_length", "aligned_sequence"))
    manifest = {
        "source_csv": str(source), "source_sha256": sha256_file(source), "source_rows": int(len(frame)),
        "eligible_unique_enzymes": len(rows), "train_unique_enzymes": len(train),
        "max_sequence_length": int(spec["max_sequence_length"]), "split_method": "enzyme_level_random",
        "split_seed": int(spec["split_seed"]), "train_ratio": float(spec["train_ratio"]),
        "reference_csv": str(reference),
    }
    write_json(root / "reference" / "sub_train_reference_manifest.json", manifest)
    return train, manifest

def build_test_records(config: Mapping, root: Path) -> List[Dict[str, object]]:
    split_dir = Path(config["paths"]["split_dir"])
    split_name = config["runtime"].get("split_name", "seq_smi")
    pairs = load_pair_file(split_dir / f"positive_test_{split_name}.pt", 1.0)
    unique = sorted({sequence for _, sequence, _ in pairs})
    limit = int(stage_spec(config)["sub_reference"]["max_sequence_length"])
    records = [{"query_sha256": sha256_text(sequence), "raw_sequence": sequence,
                "aligned_sequence": normalise_sequence(sequence)[:limit], "raw_length": len(normalise_sequence(sequence))}
               for sequence in unique]
    if len({row["query_sha256"] for row in records}) != len(records):
        raise RuntimeError("ReactZyme test query hash collision")
    write_csv(
        root / "homology" / "test_query_index.csv",
        ({key: record[key] for key in ("query_sha256", "raw_length", "aligned_sequence")} for record in records),
        ("query_sha256", "raw_length", "aligned_sequence"),
    )
    return records

def run(command: Sequence[str], log) -> None:
    log.write("$ " + " ".join(map(str, command)) + "\n"); log.flush()
    subprocess.run(list(map(str, command)), check=True, stdout=log, stderr=subprocess.STDOUT)

def remove_mmseqs_prefix(path: Path) -> None:
    for candidate in path.parent.glob(path.name + "*"):
        if candidate.is_dir() and not candidate.is_symlink(): shutil.rmtree(candidate)
        else: candidate.unlink()

def identity_bin(identity: float, edges: Sequence[float]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if identity < upper or upper == edges[-1]:
            closing = "]" if upper == edges[-1] else ")"
            return f"[{lower:.1f},{upper:.1f}{closing}"
    raise ValueError(f"Identity outside configured bins: {identity}")

def command_homology(args: argparse.Namespace) -> int:
    config, spec = load_config(args.config), None
    spec = stage_spec(config); root = output_root(config); homology = root / "homology"
    table_path = homology / "test_to_sub_train_identity.csv"
    if table_path.exists() and not args.force:
        print(f"Existing homology table retained: {table_path}"); return 0
    if args.force and root.exists(): shutil.rmtree(homology, ignore_errors=True)
    test = build_test_records(config, root)
    reference, reference_manifest = build_sub_train_reference(spec["sub_reference"], root)
    test_fasta, reference_fasta = homology / "test.fasta", homology / "sub_train.fasta"
    test_map, reference_map = write_fasta(test, test_fasta, "q"), write_fasta(reference, reference_fasta, "s")
    mm = spec["mmseqs"]; db = homology / "db"; db.mkdir(parents=True, exist_ok=True)
    qdb, rdb, result, best = (db / name for name in ("testDB", "subTrainDB", "resultDB", "resultDB_best"))
    remove_mmseqs_prefix(result); remove_mmseqs_prefix(best)
    tmp = Path(mm["tmp_dir"]); tmp.mkdir(parents=True, exist_ok=True)
    alis = homology / "test_vs_sub_train.m8"
    if alis.exists(): alis.unlink()
    maximum = max(int(mm["max_seqs"]), len(reference_map) + 10)
    log_path = homology / "mmseqs.log"
    with log_path.open("w", encoding="utf-8") as log:
        run((mm["bin"], "createdb", test_fasta, qdb), log)
        run((mm["bin"], "createdb", reference_fasta, rdb), log)
        run((mm["bin"], "search", qdb, rdb, result, tmp, "--threads", mm["threads"], "--prefilter-mode", mm["prefilter_mode"], "-e", mm["evalue"], "--max-seqs", maximum, "--min-seq-id", mm["min_seq_id"]), log)
        run((mm["bin"], "filterdb", result, best, "--extract-lines", "1", "--threads", mm["threads"]), log)
        run((mm["bin"], "convertalis", qdb, rdb, best, alis, "--threads", mm["threads"], "--format-output", "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits"), log)
    hits = {}
    with alis.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 12: raise ValueError("Unexpected MMseqs2 output schema")
            query, target = fields[:2]; alnlen = int(fields[3]); qlen, tlen = int(fields[6]), int(fields[9])
            if query in hits: raise RuntimeError("More than one MMseqs best hit for query")
            hits[query] = {"best_hit_uniprot": reference_map[target]["protein_id"], "best_hit_identity": float(fields[2]),
                           "alignment_length": alnlen, "query_coverage": alnlen / qlen, "reference_coverage": alnlen / tlen,
                           "evalue": float(fields[10]), "bits": float(fields[11])}
    edges = [float(value) for value in spec["identity_bin_edges"]]
    rows = []
    for proxy, record in test_map.items():
        hit = hits.get(proxy, {"best_hit_uniprot": None, "best_hit_identity": 0.0, "alignment_length": 0, "query_coverage": 0.0, "reference_coverage": 0.0, "evalue": None, "bits": None})
        identity = float(hit["best_hit_identity"])
        row = {"query_sha256": record["query_sha256"], "raw_length": record["raw_length"], "aligned_length": len(record["aligned_sequence"]),
               **hit, "has_hit": proxy in hits, "identity_bin": identity_bin(identity, edges),
               "is_exact_sequence": identity >= 0.999999 and hit["query_coverage"] >= 0.999999 and hit["reference_coverage"] >= 0.999999,
               "high_identity_full_coverage": identity >= float(spec["primary_identity_cutoff"]) and min(hit["query_coverage"], hit["reference_coverage"]) >= float(spec["coverage_sensitivity_cutoff"])}
        rows.append(row)
    rows.sort(key=lambda row: row["query_sha256"])
    fields = ("query_sha256", "raw_length", "aligned_length", "best_hit_uniprot", "best_hit_identity", "alignment_length", "query_coverage", "reference_coverage", "evalue", "bits", "has_hit", "identity_bin", "is_exact_sequence", "high_identity_full_coverage")
    write_csv(table_path, rows, fields)
    summary = []
    for name, mask in (("all", rows), ("exact_sequence", [row for row in rows if row["is_exact_sequence"]]), ("high_identity_full_coverage", [row for row in rows if row["high_identity_full_coverage"]])):
        summary.append({"group": name, "test_enzymes": len(mask), "fraction": len(mask) / len(rows)})
    for label in sorted({row["identity_bin"] for row in rows}):
        group = [row for row in rows if row["identity_bin"] == label]
        summary.append({"group": label, "test_enzymes": len(group), "fraction": len(group) / len(rows)})
    write_csv(homology / "identity_bin_coverage_summary.csv", summary, ("group", "test_enzymes", "fraction"))
    write_json(homology / "homology_manifest.json", {"schema_version": 1, "analysis_boundary": "test-to-actual-SUB-train sequence search only; no model was trained or altered", "config": str(args.config), "reference": reference_manifest, "test_unique_enzymes": len(test), "mmseqs": dict(mm), "identity_bin_edges": edges, "primary_identity_cutoff": spec["primary_identity_cutoff"], "coverage_sensitivity_cutoff": spec["coverage_sensitivity_cutoff"], "identity_table": str(table_path)})
    print(f"Wrote {table_path} for {len(rows)} ReactZyme test enzymes")
    return 0

def load_e2r(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return {row["query_sha256"]: {metric: float(row[metric]) for metric in E2R_METRICS} for row in csv.DictReader(handle) if row["direction"] == "enzyme_to_reaction"}

def command_statistics(args: argparse.Namespace) -> int:
    config, spec = load_config(args.config), None
    spec = stage_spec(config); root = output_root(config); identity_path = root / "homology" / "test_to_sub_train_identity.csv"
    if not identity_path.exists(): raise FileNotFoundError(f"Run homology first: {identity_path}")
    with identity_path.open(encoding="utf-8") as handle: identity = {row["query_sha256"]: row for row in csv.DictReader(handle)}
    models = list(args.models or config["comparison_models"]); baseline = args.baseline
    if baseline not in models: raise ValueError("Baseline must be selected")
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    per_query = Path(config["paths"]["results_dir"]) / "posthoc_analysis" / "per_query"
    subsets = {"all": set(identity), "identity_lt_0_40": {key for key, row in identity.items() if float(row["best_hit_identity"]) < float(spec["primary_identity_cutoff"])},
               "no_high_identity_full_coverage": {key for key, row in identity.items() if row["high_identity_full_coverage"].lower() != "true"}}
    for label in sorted({row["identity_bin"] for row in identity.values()}): subsets[f"identity_bin_{label}"] = {key for key, row in identity.items() if row["identity_bin"] == label}
    rows = []
    cache: Dict[tuple[str, int], Dict[str, Dict[str, float]]] = {}
    for model in models:
        for seed in seeds:
            path = per_query / model / f"seed_{seed}" / "per_query.csv"
            cache[(model, seed)] = load_e2r(path)
            if set(cache[(model, seed)]) != set(identity): raise ValueError(f"E2R query mismatch: {model}/seed_{seed}")
    for subset_name, keys in subsets.items():
        if not keys: continue
        for model in models:
            if model == baseline: continue
            for metric in E2R_METRICS:
                left = [float(np.mean([cache[(baseline, seed)][key][metric] for key in keys])) for seed in seeds]
                right = [float(np.mean([cache[(model, seed)][key][metric] for key in keys])) for seed in seeds]
                delta = [b - a for a, b in zip(left, right)]
                row = {"subset": subset_name, "query_count": len(keys), "comparison": f"{model}_minus_{baseline}", "model": model, "baseline": baseline, "direction": "enzyme_to_reaction", "query_metric": metric, "summary_metric": SUMMARY_NAMES[metric], "n_seeds": len(seeds), "seeds": ";".join(map(str, seeds)), "baseline_mean": float(np.mean(left)), "model_mean": float(np.mean(right)), "exact_signflip_p_two_sided": exact_sign_flip_pvalue(delta), "is_primary": subset_name == "identity_lt_0_40" and metric == "ap"}
                row.update(t_distribution_summary(delta)); rows.append(row)
    primary = [row for row in rows if row["is_primary"]]; holm_adjust(primary, "paired_t_p_two_sided")
    for row in rows: row.setdefault("paired_t_p_two_sided_holm", None)
    fields = ("subset", "query_count", "comparison", "model", "baseline", "direction", "query_metric", "summary_metric", "n_seeds", "seeds", "baseline_mean", "model_mean", "mean_delta", "sd_delta", "ci95_low", "ci95_high", "paired_t", "paired_t_p_two_sided", "paired_t_p_two_sided_holm", "exact_signflip_p_two_sided", "is_primary")
    write_csv(root / "stage_b" / "e2r_per_seed_by_identity_bin.csv", rows, fields)
    write_json(root / "stage_b" / "analysis_manifest.json", {"schema_version": 1, "analysis_boundary": "post-hoc E2R re-aggregation of cached checkpoint replay; no training or checkpoint modification", "baseline": baseline, "models": models, "seeds": seeds, "primary_endpoint": "E2R MAP/AP on identity_lt_0_40; Holm across three EnzSub contrasts", "identity_table": str(identity_path), "subsets": {name: len(keys) for name, keys in subsets.items()}, "coverage_sensitivity": "no_high_identity_full_coverage excludes only hits with identity >= 0.40 and both coverages >= 0.80"})
    print(f"Wrote Stage-B statistics to {root / 'stage_b'}")
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    homology = commands.add_parser("homology", help="Build actual SUB-train reference and run exhaustive MMseqs2 search")
    homology.add_argument("--config", type=Path, required=True); homology.add_argument("--force", action="store_true"); homology.set_defaults(func=command_homology)
    stats = commands.add_parser("statistics", help="Re-aggregate cached E2R queries by SUB-train identity")
    stats.add_argument("--config", type=Path, required=True); stats.add_argument("--models", nargs="+"); stats.add_argument("--baseline", default="reactzyme_esm2_650m"); stats.set_defaults(func=command_statistics)
    return parser.parse_args()

if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(arguments.func(arguments))
