#!/usr/bin/env python3
"""
build_random_fasta.py

Build a composition-matched random sequence FASTA for protein embedding
Random Neighbor Score (RNS) analysis.

Core idea:
    For each real enzyme sequence, generate N random sequences by full-sequence
    random permutation. Length and amino acid composition are preserved; all
    biological order information (motifs, local context, domain structure) is
    destroyed.

These random sequences are intended to be embedded by a PLM together with the
real enzyme set, forming a joint latent space in which:

        RNS(P_i) = #random_neighbors_in_top_k / k

is computed per real protein P_i.
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

from tqdm import tqdm

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (header_without_leading_'>', sequence) pairs from a FASTA file.

    Keeps things minimal: no BioPython dependency. Tolerates blank lines and
    sequences wrapped across multiple lines.
    """
    header: str | None = None
    chunks: List[str] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]  # drop '>'
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)

def clean_sequence(seq: str) -> Tuple[str, int]:
    """Uppercase, strip whitespace, drop non-standard residues.

    Returns (cleaned_sequence, num_illegal_chars_removed).
    """
    seq = seq.upper()
    cleaned_chars = []
    illegal = 0
    for ch in seq:
        if ch.isspace():
            continue
        if ch in STANDARD_AA:
            cleaned_chars.append(ch)
        else:
            illegal += 1
    return "".join(cleaned_chars), illegal

def shuffle_sequence(seq: str, rng: random.Random) -> str:
    """Return a full-sequence random permutation of `seq`."""
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)

def generate_shuffles(
    seq: str,
    n: int,
    rng: random.Random,
    max_attempts_per_shuffle: int = 20,
) -> Tuple[List[str], int]:
    """Generate `n` distinct shuffles of `seq`.

    Constraints:
      * each shuffle != original sequence
      * shuffles should ideally be pairwise distinct (best-effort)

    For very short / low-complexity sequences (e.g. all same residue) it can be
    impossible to satisfy these constraints; in that case we fall back to
    returning whatever we managed to produce and report the failures.

    Returns (list_of_shuffles, num_failed_slots).
    """
    if len(set(seq)) <= 1 or len(seq) < 2:
        return [], n

    produced: List[str] = []
    seen: set[str] = {seq}  # don't allow equal-to-original
    failed = 0

    for _ in range(n):
        ok = False
        for _attempt in range(max_attempts_per_shuffle):
            cand = shuffle_sequence(seq, rng)
            if cand not in seen:
                produced.append(cand)
                seen.add(cand)
                ok = True
                break
        if not ok:
            failed += 1

    return produced, failed

def make_random_header(original_header: str, idx: int) -> str:
    """Prefix the original header with `random_{idx}|` for traceability.

    The original header is kept intact (whitespace and description included)
    so downstream code can map a random sequence back to its source.
    """
    return f"random_{idx}|{original_header}"

def main():
    parser = argparse.ArgumentParser(
        description="Build composition-matched random shuffle FASTA for PLM RNS analysis."
    )
    parser.add_argument(
        "--input_fasta",
        type=str,
        default="data/rns/ENZ30_val.fasta",
        help="Path to input real-enzyme FASTA.",
    )
    parser.add_argument(
        "--output_fasta",
        type=str,
        default="data/rns/ENZ30_val_random.fasta",
        help="Path to output random FASTA.",
    )
    parser.add_argument(
        "--n_shuffles",
        type=int,
        default=5,
        help="Number of random shuffles to generate per real sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=2,
        help="Minimum sequence length after cleaning (shorter sequences are skipped).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, drop any sequence that contains illegal characters "
             "instead of just stripping them.",
    )
    args = parser.parse_args()

    in_path = Path(args.input_fasta)
    out_path = Path(args.output_fasta)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    n_input = 0
    n_kept = 0                     # real sequences that produced at least 1 shuffle
    n_illegal_seqs = 0             # sequences containing >=1 illegal residue
    n_dropped_strict = 0           # dropped due to --strict
    n_dropped_too_short = 0        # length < min_len after cleaning
    n_dropped_low_complexity = 0   # single-residue sequences -> cannot shuffle
    n_random_written = 0
    n_failed_slots = 0             # individual shuffle attempts that gave up
    lengths_written: List[int] = []

    with out_path.open("w") as out_fh:
        for header, raw_seq in tqdm(iter_fasta(in_path), desc="shuffling", unit="seq"):
            n_input += 1
            cleaned, n_illegal_chars = clean_sequence(raw_seq)

            if n_illegal_chars > 0:
                n_illegal_seqs += 1
                if args.strict:
                    n_dropped_strict += 1
                    continue

            if len(cleaned) < args.min_len:
                n_dropped_too_short += 1
                continue

            if len(set(cleaned)) <= 1:
                n_dropped_low_complexity += 1
                continue

            shuffles, failed = generate_shuffles(cleaned, args.n_shuffles, rng)
            n_failed_slots += failed

            if not shuffles:
                continue

            n_kept += 1
            for i, sh in enumerate(shuffles, start=1):
                out_fh.write(f">{make_random_header(header, i)}\n")
                out_fh.write(sh + "\n")
                n_random_written += 1
                lengths_written.append(len(sh))

    print("\n=== Random FASTA build summary ===")
    print(f"Input FASTA              : {in_path}")
    print(f"Output FASTA             : {out_path}")
    print(f"Shuffles per real seq    : {args.n_shuffles}")
    print(f"Random seed              : {args.seed}")
    print(f"Strict illegal handling  : {args.strict}")
    print("-" * 40)
    print(f"Original sequences read  : {n_input}")
    print(f"  with illegal chars     : {n_illegal_seqs}")
    print(f"  dropped (strict mode)  : {n_dropped_strict}")
    print(f"  dropped (too short)    : {n_dropped_too_short}  (min_len={args.min_len})")
    print(f"  dropped (low complex.) : {n_dropped_low_complexity}")
    print(f"Real seqs used           : {n_kept}")
    print(f"Random sequences written : {n_random_written}")
    print(f"Shuffle failures (slots) : {n_failed_slots}")

    if lengths_written:
        print("-" * 40)
        print(f"Length stats of written random sequences:")
        print(f"  mean   : {statistics.mean(lengths_written):.2f}")
        print(f"  median : {statistics.median(lengths_written):.1f}")
        print(f"  min    : {min(lengths_written)}")
        print(f"  max    : {max(lengths_written)}")
        overall = Counter()
    else:
        print("WARNING: no random sequences were written.")

if __name__ == "__main__":
    main()
