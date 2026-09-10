#!/usr/bin/env python3
"""Profile test-to-train protein homology for an existing ReactZyme split.

The script does not alter a split. It reports maximum pairwise identity against
the positive training enzymes and applies an explicit 80% coverage filter to
both query and target sequences after MMseqs2 search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch

from common import load_pair_file

def sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()

def write_fasta(path: Path, sequences: Iterable[str]) -> Dict[str, str]:
    mapping = {sequence_id(sequence): sequence for sequence in sorted(set(sequences))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for identifier, sequence in mapping.items():
            handle.write(f">{identifier}\n{sequence}\n")
    return mapping

def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 executable path or command")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Temporary FASTA/search directory; default is output-dir/work")
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--max-seqs", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8, help="Bound MMseqs2 CPU use while other experiments run")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    executable = shutil.which(args.mmseqs) if "/" not in args.mmseqs else args.mmseqs
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(f"MMseqs2 executable not found: {args.mmseqs}")
    version = subprocess.run([executable, "version"], check=True, capture_output=True, text=True).stdout.strip()
    output_dir = args.output_dir
    work_dir = args.work_dir or output_dir / "work"
    train_path = args.split_dir / f"positive_train_val_{args.split_name}.pt"
    test_path = args.split_dir / f"positive_test_{args.split_name}.pt"
    train_pairs = load_pair_file(train_path, 1.0)
    test_pairs = load_pair_file(test_path, 1.0)
    train_sequences = [sequence for _, sequence, _ in train_pairs]
    test_sequences = [sequence for _, sequence, _ in test_pairs]
    train_fasta, test_fasta = work_dir / "train_positive_enzymes.fasta", work_dir / "test_positive_enzymes.fasta"
    train_map, test_map = write_fasta(train_fasta, train_sequences), write_fasta(test_fasta, test_sequences)
    result_tsv, temporary = work_dir / "test_to_train.tsv", work_dir / "mmseqs_tmp"
    if not (args.resume and result_tsv.exists()):
        command = [
            executable, "easy-search", str(test_fasta), str(train_fasta), str(result_tsv), str(temporary),
            "--alignment-mode", "3", "--cov-mode", "0", "-c", str(args.coverage), "--max-seqs", str(args.max_seqs),
            "--threads", str(args.threads),
            "--format-output", "query,target,pident,alnlen,qlen,tlen,evalue,bits",
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    best_any: Dict[str, Dict[str, object]] = {}
    best_bidirectional: Dict[str, Dict[str, object]] = {}
    with result_tsv.open(encoding="utf-8") as handle:
        for line in handle:
            query, target, pident, alnlen, qlen, tlen, evalue, bits = line.rstrip("\n").split("\t")
            identity = float(pident)
            if identity <= 1.0:
                identity *= 100.0
            alignment, query_length, target_length = int(alnlen), int(qlen), int(tlen)
            query_coverage, target_coverage = alignment / query_length, alignment / target_length
            bidirectional = query_coverage >= args.coverage and target_coverage >= args.coverage
            candidate = {"target_sha256": target, "identity_pct": identity, "alignment_length": alignment, "query_coverage": query_coverage, "target_coverage": target_coverage, "bidirectional_coverage": bidirectional, "evalue": float(evalue), "bits": float(bits)}
            old = best_any.get(query)
            if old is None or identity > float(old["identity_pct"]):
                best_any[query] = candidate
            if bidirectional:
                old = best_bidirectional.get(query)
                if old is None or identity > float(old["identity_pct"]):
                    best_bidirectional[query] = candidate
    rows = []
    for identifier, sequence in sorted(test_map.items()):
        hit_any = best_any.get(identifier)
        hit = best_bidirectional.get(identifier)
        rows.append({
            "test_sequence_sha256": identifier,
            "sequence_length": len(sequence),
            "has_mmseqs_hit": hit_any is not None,
            "max_identity_any_pct": hit_any["identity_pct"] if hit_any else None,
            "max_identity_pct": hit["identity_pct"] if hit else None,
            "best_train_sequence_sha256": hit["target_sha256"] if hit else None,
            "alignment_length": hit["alignment_length"] if hit else None,
            "query_coverage": hit["query_coverage"] if hit else None,
            "target_coverage": hit["target_coverage"] if hit else None,
            "bidirectional_coverage": hit["bidirectional_coverage"] if hit else False,
            "evalue": hit["evalue"] if hit else None,
            "bits": hit["bits"] if hit else None,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "test_to_train_max_identity.csv", rows, list(rows[0]))
    thresholds = (10, 30, 40, 50, 60)
    eligible = [row for row in rows if row["max_identity_pct"] is not None and row["bidirectional_coverage"]]
    summary = {
        "schema_version": 1,
        "artifact": "reactzyme_test_to_train_homology_profile",
        "analysis_boundary": "post-hoc profile of the existing split; this is not a cluster-held-out retraining result",
        "mmseqs_executable": executable,
        "mmseqs_version": version,
        "search_contract": {"alignment_mode": 3, "coverage_mode": 0, "minimum_coverage": args.coverage, "postfilter": "alignment/query length and alignment/target length must both be >= minimum coverage", "max_seqs": args.max_seqs, "threads": args.threads},
        "train_unique_sequences": len(train_map),
        "test_unique_sequences": len(test_map),
        "test_sequences_with_bidirectional_hit": len(eligible),
        "test_sequences_without_bidirectional_hit": len(rows) - len(eligible),
        "counts_with_max_identity_at_or_below_threshold": {str(value): sum(1 for row in eligible if float(row["max_identity_pct"]) <= value) for value in thresholds},
        "files": {"train_positive": str(train_path), "test_positive": str(test_path), "per_test_identity": str(output_dir / "test_to_train_max_identity.csv")},
    }
    (output_dir / "homology_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote homology profile to {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
