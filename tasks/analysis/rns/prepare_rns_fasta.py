#!/usr/bin/env python3
"""
prepare_rns_fasta.py

Step 1 of the RNS analysis pipeline.

Merge a real-enzyme FASTA and a composition-matched random FASTA into a single
combined FASTA, and emit a CSV metadata table describing every entry. The
combined FASTA can then be fed to a PLM embedding script, and the metadata
table is used downstream to compute:

        RNS(P_i) = #random_neighbors_in_top_k / k

Metadata columns:
    seq_id, is_random, source_id, length

For real sequences      : is_random=0, source_id == seq_id
For random shuffles     : is_random=1, source_id is parsed from the
                          "random_N|<original_header>" prefix.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterator, Tuple

def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (header_without_'>', sequence) pairs from a FASTA file."""
    header = None
    chunks = []
    with path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)

def parse_real_header(header: str) -> str:
    """Real-enzyme seq_id = everything up to the first whitespace.

    e.g. 'tr|A0A6G4MVN3|A0A6G4MVN3_9ENTR Protein-disulfide reductase...'
         -> 'tr|A0A6G4MVN3|A0A6G4MVN3_9ENTR'
    """
    return header.split(None, 1)[0]

def parse_random_header(header: str) -> Tuple[str, str]:
    """Parse a random-shuffle header.

    Input header looks like:
        'random_1|tr|A0A6G4MVN3|A0A6G4MVN3_9ENTR'
    or, if the upstream script preserved the full original description:
        'random_1|tr|A0A6G4MVN3|A0A6G4MVN3_9ENTR description text...'

    We:
      * take only up to the first whitespace as the seq_id stem
      * split once on '|' to separate the 'random_N' tag from the source id
      * the source_id is then normalised so it matches what parse_real_header
        would produce for the corresponding real entry (i.e. no trailing
        description)

    Returns (seq_id, source_id).
    """
    stem = header.split(None, 1)[0]  # drop any trailing description

    if not stem.startswith("random_") or "|" not in stem:
        raise ValueError(
            f"Random-sequence header does not match 'random_N|<source_id>' "
            f"pattern: {header!r}"
        )

    tag, source_id = stem.split("|", 1)
    if not source_id:
        raise ValueError(f"Empty source_id parsed from random header: {header!r}")

    seq_id = f"{tag}|{source_id}"
    return seq_id, source_id

def write_fasta_record(fh, seq_id: str, sequence: str, width: int = 60) -> None:
    fh.write(f">{seq_id}\n")
    for i in range(0, len(sequence), width):
        fh.write(sequence[i:i + width])
        fh.write("\n")

def main():
    p = argparse.ArgumentParser(
        description="Merge real + random FASTAs and emit RNS metadata CSV."
    )
    p.add_argument("--real", required=True, type=Path,
                   help="Real-enzyme FASTA (e.g. ENZ30_val.fasta).")
    p.add_argument("--random", required=True, type=Path,
                   help="Random-shuffle FASTA (e.g. ENZ30_val_random.fasta).")
    p.add_argument("--out-fasta", required=True, type=Path,
                   help="Combined output FASTA.")
    p.add_argument("--out-meta", required=True, type=Path,
                   help="Output metadata CSV.")
    p.add_argument("--width", type=int, default=60,
                   help="FASTA sequence line width (default: 60).")
    p.add_argument("--strict-source-check", action="store_true",
                   help="Fail if any random sequence's source_id is not "
                        "present among the real sequence IDs.")
    args = p.parse_args()

    for path in (args.real, args.random):
        if not path.is_file():
            sys.exit(f"ERROR: input file not found: {path}")
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    meta_rows: list[tuple[str, int, str, int]] = []  # (seq_id, is_random, source_id, length)

    n_real = 0
    n_random = 0
    real_ids: set[str] = set()       # for the optional source-id cross-check
    unknown_source_ids: list[str] = []

    total_len = 0
    min_len = None
    max_len = 0

    def _track_id(seq_id: str) -> None:
        if seq_id in seen_ids:
            duplicate_ids.append(seq_id)
        else:
            seen_ids.add(seq_id)

    with args.out_fasta.open("w") as out_fh:
        for header, seq in iter_fasta(args.real):
            seq_id = parse_real_header(header)
            length = len(seq)

            _track_id(seq_id)
            real_ids.add(seq_id)

            write_fasta_record(out_fh, seq_id, seq, width=args.width)
            meta_rows.append((seq_id, 0, seq_id, length))

            n_real += 1
            total_len += length
            min_len = length if min_len is None else min(min_len, length)
            max_len = max(max_len, length)

        for header, seq in iter_fasta(args.random):
            seq_id, source_id = parse_random_header(header)
            length = len(seq)

            _track_id(seq_id)

            if source_id not in real_ids:
                unknown_source_ids.append(source_id)

            write_fasta_record(out_fh, seq_id, seq, width=args.width)
            meta_rows.append((seq_id, 1, source_id, length))

            n_random += 1
            total_len += length
            min_len = length if min_len is None else min(min_len, length)
            max_len = max(max_len, length)

    with args.out_meta.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["seq_id", "is_random", "source_id", "length"])
        writer.writerows(meta_rows)

    total = n_real + n_random
    mean_len = (total_len / total) if total else 0.0
    n_dup = len(duplicate_ids)
    n_unknown_src = len(unknown_source_ids)

    print("=== RNS prepare summary ===")
    print(f"Real FASTA               : {args.real}")
    print(f"Random FASTA             : {args.random}")
    print(f"Combined FASTA out       : {args.out_fasta}")
    print(f"Metadata CSV out         : {args.out_meta}")
    print("-" * 40)
    print(f"Real sequences           : {n_real}")
    print(f"Random sequences         : {n_random}")
    print(f"Total written            : {total}")
    print(f"  mean length            : {mean_len:.2f}")
    print(f"  min  length            : {min_len if min_len is not None else 0}")
    print(f"  max  length            : {max_len}")
    print("-" * 40)
    print(f"Duplicate seq_ids        : {n_dup}")
    if n_dup:
        preview = duplicate_ids[:5]
        print(f"  examples              : {preview}"
              + (" ..." if n_dup > len(preview) else ""))

    print(f"Random→unknown source_id : {n_unknown_src}")
    if n_unknown_src:
        preview = unknown_source_ids[:5]
        print(f"  examples              : {preview}"
              + (" ..." if n_unknown_src > len(preview) else ""))
        if args.strict_source_check:
            sys.exit(
                "ERROR: --strict-source-check set and some random sequences "
                "reference source_ids not present in the real FASTA."
            )

if __name__ == "__main__":
    main()
