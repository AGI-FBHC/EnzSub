#!/usr/bin/env python3
"""Create ReactZyme-code-faithful reaction-similarity hard negatives.

The released ReactZyme repository provides an executable example for building
nearest-neighbour reaction dictionaries with Levenshtein distance. It does not
release a completed negative-pair file or an executable enzyme-side negative
sampler. This script therefore implements the released reaction side
explicitly and records the resulting protocol in JSON.
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import torch
from Levenshtein import distance

from common import load_pair_file, write_json

def normalise_reaction(reaction):
    return reaction.replace("*", "C")

def load_positive_pairs(split_dir, split_name):
    train = load_pair_file(split_dir / f"positive_train_val_{split_name}.pt", 1.0)
    test = load_pair_file(split_dir / f"positive_test_{split_name}.pt", 1.0)
    return train, test

def nearest_reactions(target, reactions, candidate_pool):
    """Match the public prepare_negative.py ranking rule deterministically."""
    ranked = [(candidate, distance(target, candidate)) for candidate in reactions if candidate != target]
    ranked.sort(key=lambda item: (item[1], item[0]))
    return ranked[:candidate_pool]

def generate_partition(positives, reaction_candidates, all_positive_pairs,
                       nearest_cache, candidate_pool, per_positive, partition,
                       split_name, split_dir, partition_seed):
    rng = random.Random(partition_seed)
    negatives = []
    trace = []
    for anchor_index, (reaction, sequence, _) in enumerate(positives):
        reaction = normalise_reaction(reaction)
        ranked = nearest_cache.get(reaction)
        if ranked is None:
            ranked = nearest_reactions(reaction, reaction_candidates, candidate_pool)
            nearest_cache[reaction] = ranked
        eligible = []
        for rank, (candidate, candidate_distance) in enumerate(ranked, 1):
            if (candidate, sequence) in all_positive_pairs:
                continue
            eligible.append((candidate, candidate_distance, rank))
        if len(eligible) < per_positive:
            raise RuntimeError(
                "No sufficient eligible reaction-similarity negatives in the "
                f"top-{candidate_pool} pool for {partition} anchor "
                f"{anchor_index} ({reaction!r}, {sequence[:24]!r})"
            )
        for candidate, candidate_distance, rank in rng.sample(eligible, per_positive):
            negatives.append((candidate, sequence))
            trace.append({
                "anchor_index": anchor_index,
                "candidate_rank": rank,
                "candidate_distance": candidate_distance,
            })

    payload = {
        f"negative_{index:09d}": [reaction, sequence]
        for index, (reaction, sequence) in enumerate(negatives)
    }
    output_path = split_dir / f"negative_{partition}_{split_name}.pt"
    torch.save(payload, output_path)

    pair_list = list(payload.values())
    ranks = [item["candidate_rank"] for item in trace]
    distances = [item["candidate_distance"] for item in trace]
    unique_pairs = len({tuple(pair) for pair in pair_list})
    return {
        "partition": partition,
        "positive_count": len(positives),
        "negative_count": len(pair_list),
        "negative_unique_pair_count": unique_pairs,
        "negative_duplicate_pair_count": len(pair_list) - unique_pairs,
        "candidate_reaction_count": len(reaction_candidates),
        "candidate_pool": candidate_pool,
        "per_positive": per_positive,
        "partition_seed": partition_seed,
        "selection": "uniform_eligible_from_top_pool",
        "candidate_rank_min": min(ranks) if ranks else None,
        "candidate_rank_max": max(ranks) if ranks else None,
        "candidate_rank_mean": statistics.mean(ranks) if ranks else None,
        "candidate_distance_min": min(distances) if distances else None,
        "candidate_distance_max": max(distances) if distances else None,
        "candidate_distance_mean": statistics.mean(distances) if distances else None,
        "output": str(output_path),
    }

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Base seed; test sampling uses seed + 1.")
    parser.add_argument("--per-positive", type=int, default=1)
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument(
        "--selection",
        choices=["uniform_eligible_from_top_pool"],
        default="uniform_eligible_from_top_pool",
        help="Select without replacement from eligible reactions in the top-k pool.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.per_positive < 1:
        raise ValueError("--per-positive must be at least 1")
    if args.candidate_pool < args.per_positive:
        raise ValueError("--candidate-pool must be >= --per-positive")

    train, test = load_positive_pairs(args.split_dir, args.split_name)
    all_positive_pairs = {
        (normalise_reaction(reaction), sequence)
        for reaction, sequence, _ in train + test
    }
    reaction_candidates = sorted({reaction for reaction, _ in all_positive_pairs})
    nearest_cache = {}
    partitions = [
        generate_partition(
            train, reaction_candidates, all_positive_pairs, nearest_cache,
            args.candidate_pool, args.per_positive, "train_val", args.split_name,
            args.split_dir, args.seed,
        ),
        generate_partition(
            test, reaction_candidates, all_positive_pairs, nearest_cache,
            args.candidate_pool, args.per_positive, "test", args.split_name,
            args.split_dir, args.seed + 1,
        ),
    ]
    report = {
        "artifact": "reactzyme_reaction_similarity_negatives",
        "source": "ReactZyme official prepare_negative.py pattern",
        "method": "Levenshtein-ranked reaction replacement; uniformly sample eligible top-k candidates; all known positive pairs excluded",
        "enzyme_side": "not included: no executable official implementation or released negative file",
        "split_name": args.split_name,
        "seed": args.seed,
        "selection": args.selection,
        "per_positive": args.per_positive,
        "candidate_pool": args.candidate_pool,
        "candidate_scope": "all_unique_positive_reactions_from_train_and_test",
        "exclusion_scope": "all_positive_pairs_from_train_and_test",
        "partitions": partitions,
    }
    write_json(args.split_dir / f"negative_sampling_{args.split_name}.json", report)
    print(f"Saved ReactZyme similarity negatives under {args.split_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
