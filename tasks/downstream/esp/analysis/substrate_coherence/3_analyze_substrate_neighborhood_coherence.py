#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_analyze_substrate_neighborhood_coherence.py

enzyme-only embedding 近邻底物化学一致性分析 (EnzSub 论文 §2.3 表示层证据)。

回答的问题:
    在不使用 ChemBERTa / cross-modal head / 任何 SUB 监督头的情况下，仅用酶序列
    embedding 检索邻近酶，SUB 模型的邻近酶是否具有更高的底物化学相似性？

这是 enzyme-only 几何分析，与已有 pair-level ESP 分类 (2_knn_evaluate.py /
2_train_evaluate.py) 完全不同；本脚本不修改任何现有 ESP 流程。

Use the public configurations under ``configs/analysis/substrate_coherence``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subcoh.chem import REQUIRED_SET_METRICS, SubstrateChemistry
from subcoh.coherence import CoherenceComputer, compute_model_condition, random_baseline
from subcoh.data import (
    apply_substrate_filter, build_cohort, build_global_id_map,
    compute_excluded_substrates, load_and_aggregate, load_currency_file,
)
from subcoh.homology import HomologyTable
from subcoh.neighbors import build_eligible_pools, l2_normalize, sample_random_neighbors
from subcoh.stats import paired_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("subcoh.main")

PRIMARY_MODES = frozenset({"base", "base_sub"})

def substrate_count_group(n: int) -> str:
    if n <= 1:
        return "single"
    if n <= 4:
        return "2-4"
    return ">=5"

# =============================================================================
# Config resolution (YAML 与 CLI 合并；CLI 优先)
# =============================================================================
def parse_kv_list(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--model-files entry must be name=path, got: {it}")
        name, path = it.split("=", 1)
        out[name.strip()] = path.strip()
    return out

def load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}

def resolve_models(args, cfg) -> Dict[str, dict]:
    """
    合并模型元数据。返回 {name: {backbone, mode, pkl}}。
    来源优先级: --model-files (pkl) > metadata yaml > config yaml。
    backbone/mode 不允许靠文件名猜测，必须显式提供。
    """
    models: Dict[str, dict] = {}
    # 1) config / metadata yaml
    for src in (cfg.get("models", {}), load_yaml(args.model_metadata).get("models", {})):
        for name, ent in (src or {}).items():
            ent = ent or {}
            m = models.setdefault(name, {})
            for key in ("backbone", "mode", "pkl"):
                if ent.get(key) is not None:
                    m[key] = ent[key]
    # 2) --model-files 覆盖 pkl
    for name, path in parse_kv_list(args.model_files).items():
        models.setdefault(name, {})["pkl"] = path

    # 校验
    for name, m in models.items():
        for key in ("backbone", "mode", "pkl"):
            if not m.get(key):
                raise ValueError(
                    f"Model '{name}' missing '{key}'. backbone/mode must be specified "
                    f"explicitly (metadata yaml or config), pkl via --model-files or config."
                )
    return models

def resolve_comparisons(args, cfg, models) -> List[Tuple[str, str]]:
    raw = args.comparisons or cfg.get("comparisons") or []
    comps: List[Tuple[str, str]] = []
    for c in raw:
        if isinstance(c, str):
            a, b = c.split(":", 1)
        else:  # yaml list of [a, b] or {a: , b:}
            a, b = c[0], c[1]
        a, b = a.strip(), b.strip()
        for nm in (a, b):
            if nm not in models:
                raise ValueError(f"Comparison references unknown model '{nm}'")
        comps.append((a, b))
    if not comps:
        raise ValueError("No comparisons specified (--comparisons or config.comparisons).")
    return comps

# =============================================================================
# Per-comparison execution
# =============================================================================
def neighbor_identity_stats(
    query_hash: str, nbr_hashes: List[str], pident_lookup: Optional[dict]
) -> Tuple[float, float]:
    if not pident_lookup or not nbr_hashes:
        return float("nan"), float("nan")
    vals = [pident_lookup.get((query_hash, h), 0.0) for h in nbr_hashes]
    return float(np.mean(vals)), float(np.median(vals))

def run_comparison(
    name_a: str, name_b: str, model_a, model_b, id_map,
    chem: SubstrateChemistry,
    homology: Optional[HomologyTable],
    pident_lookup: Optional[dict],
    args,
    per_query_rows: List[dict],
    summary_rows: List[dict],
    neighbor_rows: List[dict],
    excluded_tables: List[pd.DataFrame],
    sanity: dict,
):
    cohort = build_cohort(model_a, model_b, id_map)
    comparison = f"{name_a}__vs__{name_b}"
    is_primary = frozenset({model_a.mode, model_b.mode}) == PRIMARY_MODES
    comp_type = "primary" if is_primary else "secondary"
    backbone = cohort.backbone
    hash_to_seqlen = {h: len(model_a.sequences[h]) for h in cohort.seq_hashes}

    # metabolite filters
    filters = ["unfiltered"]
    do_filter = bool(args.currency_metabolite_file) or (args.max_substrate_enzyme_fraction is not None)
    if do_filter:
        filters.append("filtered")
    currency = load_currency_file(args.currency_metabolite_file)

    # homology conditions
    homology_conditions: List[Tuple[str, Optional[float]]] = [("unrestricted", None)]
    if homology is not None:
        for cut in args.identity_cutoffs:
            homology_conditions.append((f"identity<{cut:g}", float(cut)))
    elif args.identity_cutoffs:
        logger.warning("identity-cutoffs given but no --homology-table; running unrestricted only "
                       "(NOT labelled as low-homology).")

    metrics = list(REQUIRED_SET_METRICS) + [m for m in args.extra_set_metrics if m not in REQUIRED_SET_METRICS]

    for met_filter in filters:
        # 构建该 filter 下的有效 cohort
        if met_filter == "filtered":
            excluded, table = compute_excluded_substrates(
                cohort.substrates, currency, args.max_substrate_enzyme_fraction)
            table.insert(0, "comparison", comparison)
            excluded_tables.append(table)
            filt_subs, dropped = apply_substrate_filter(cohort.substrates, excluded)
        else:
            filt_subs, dropped = dict(cohort.substrates), []

        eff_hashes = [h for h in cohort.seq_hashes if h not in set(dropped)]
        if len(eff_hashes) < max(args.k_values) + 1:
            logger.warning("[%s/%s] too few enzymes (%d) after filter; skipping",
                           comparison, met_filter, len(eff_hashes))
            continue
        hash_to_idx = {h: i for i, h in enumerate(eff_hashes)}
        sub_by_idx = [filt_subs[h] for h in eff_hashes]
        nsub_by_idx = [len(s) for s in sub_by_idx]

        sel = np.array([cohort.seq_hashes.index(h) for h in eff_hashes])
        emb_a = l2_normalize(cohort.emb_a[sel])
        emb_b = l2_normalize(cohort.emb_b[sel])
        n = len(eff_hashes)

        for homology_cond, cutoff in homology_conditions:
            # homology -> idx 排除集
            if cutoff is None:
                exclusion_by_idx = None
            else:
                excl_sets = homology.exclusion_sets(cutoff, args.min_alignment_coverage)
                exclusion_by_idx = []
                for i, h in enumerate(eff_hashes):
                    tgt = excl_sets.get(h, set())
                    exclusion_by_idx.append({hash_to_idx[t] for t in tgt if t in hash_to_idx})
            pools = build_eligible_pools(n, exclusion_by_idx)

            for metric in metrics:
                computer = CoherenceComputer(sub_by_idx, chem, metric)
                for k in args.k_values:
                    draws = sample_random_neighbors(pools, k, args.random_repeats, args.seed)
                    rmean, rstd = random_baseline(computer, draws)

                    res_a = compute_model_condition(emb_a, pools, k, computer, draws, rmean, rstd,
                                                    batch_size=args.batch_size)
                    res_b = compute_model_condition(emb_b, pools, k, computer, draws, rmean, rstd,
                                                    batch_size=args.batch_size)

                    valid = sorted(set(res_a.coherence) & set(res_b.coherence))
                    n_excluded = n - len(valid)

                    # ---- sanity: 近邻不含自身 ----
                    for res in (res_a, res_b):
                        for qi in valid:
                            assert qi not in set(res.nbr_idx[qi, :k].tolist()), "query in its own neighbours"

                    # ---- per-query rows (两个模型各一行) ----
                    for (mname, mode, res) in ((name_a, model_a.mode, res_a),
                                               (name_b, model_b.mode, res_b)):
                        for qi in valid:
                            qh = eff_hashes[qi]
                            nbr = res.nbr_idx[qi, :k]
                            nbr_h = [eff_hashes[int(j)] for j in nbr if j >= 0]
                            mid, medid = neighbor_identity_stats(qh, nbr_h, pident_lookup)
                            # low-homology sanity: 近邻不得落入该 query 的同源排除集。
                            # 注意排除条件 = (pident >= cutoff) 且 (query/target 覆盖 >= min_cov)；
                            # 因此只凭 pident 判断会误报 —— 高 identity 但低覆盖的局部相似
                            # 不算可靠的全长同源, 本就不该被排除, 也允许留在近邻里。
                            # 这里直接对照实际排除集 (按 idx), 与过滤逻辑严格一致。
                            if exclusion_by_idx is not None:
                                excl_i = exclusion_by_idx[qi]
                                for j in nbr:
                                    if int(j) >= 0:
                                        assert int(j) not in excl_i, \
                                            "homology-excluded neighbour leaked into top-k"
                            per_query_rows.append({
                                "backbone": backbone, "model": mname, "mode": mode,
                                "comparison": comparison, "comparison_type": comp_type,
                                "query_id": id_map[qh], "query_sequence_hash": qh,
                                "n_query_substrates": nsub_by_idx[qi],
                                "k": k,
                                "identity_cutoff": cutoff,
                                "homology_condition": homology_cond,
                                "metabolite_filter": met_filter,
                                "set_similarity_metric": metric,
                                "coherence": res.coherence[qi],
                                "random_coherence_mean": res.random_mean.get(qi),
                                "random_coherence_std": res.random_std.get(qi),
                                "enrichment": res.enrichment.get(qi),
                                "n_eligible_candidates": res.n_eligible[qi],
                                "mean_neighbor_sequence_identity": mid,
                                "median_neighbor_sequence_identity": medid,
                                "substrate_count_group": substrate_count_group(nsub_by_idx[qi]),
                            })

                    # ---- paired summary (coherence) ----
                    base_c = np.array([res_a.coherence[q] for q in valid])
                    sub_c = np.array([res_b.coherence[q] for q in valid])
                    psum = paired_summary(base_c, sub_c, n_boot=args.n_boot,
                                          n_perm=args.n_perm, seed=args.seed)
                    base_e = np.array([res_a.enrichment.get(q, np.nan) for q in valid])
                    sub_e = np.array([res_b.enrichment.get(q, np.nan) for q in valid])
                    # neighbour identity confounder check
                    bid = np.nanmean([res_a_row_identity(res_a, q, k, eff_hashes, pident_lookup)
                                      for q in valid]) if pident_lookup else float("nan")
                    sid = np.nanmean([res_a_row_identity(res_b, q, k, eff_hashes, pident_lookup)
                                      for q in valid]) if pident_lookup else float("nan")

                    summary_rows.append({
                        "backbone": backbone, "comparison": comparison,
                        "comparison_type": comp_type,
                        "base_model": name_a, "sub_model": name_b,
                        "k": k, "identity_cutoff": cutoff,
                        "homology_condition": homology_cond,
                        "metabolite_filter": met_filter,
                        "set_similarity_metric": metric,
                        "n_queries": psum["n_queries"], "n_excluded_queries": n_excluded,
                        "base_mean": psum["base_mean"], "sub_mean": psum["sub_mean"],
                        "mean_delta": psum["mean_delta"], "median_delta": psum["median_delta"],
                        "bootstrap_ci_low": psum["bootstrap_ci_low"],
                        "bootstrap_ci_high": psum["bootstrap_ci_high"],
                        "fraction_improved": psum["fraction_improved"],
                        "wilcoxon_statistic": psum["wilcoxon_statistic"],
                        "wilcoxon_p": psum["wilcoxon_p"],
                        "permutation_p": psum["permutation_p"],
                        "base_mean_enrichment": float(np.nanmean(base_e)) if base_e.size else float("nan"),
                        "sub_mean_enrichment": float(np.nanmean(sub_e)) if sub_e.size else float("nan"),
                        "mean_enrichment_delta": float(np.nanmean(sub_e - base_e)) if base_e.size else float("nan"),
                        "base_mean_neighbor_identity": bid,
                        "sub_mean_neighbor_identity": sid,
                    })

                    # ---- neighbor audit (限定条件以控制文件大小) ----
                    if (met_filter == "unfiltered" and metric == "c_sym"
                            and k == args.audit_k):
                        for (mname, res) in ((name_a, res_a), (name_b, res_b)):
                            for qi in valid:
                                qh = eff_hashes[qi]
                                for rank, j in enumerate(res.nbr_idx[qi, :k], start=1):
                                    if j < 0:
                                        continue
                                    j = int(j)
                                    jh = eff_hashes[j]
                                    neighbor_rows.append({
                                        "backbone": backbone, "model": mname,
                                        "comparison": comparison,
                                        "homology_condition": homology_cond,
                                        "query_id": id_map[qh], "neighbor_rank": rank,
                                        "neighbor_id": id_map[jh],
                                        "enzyme_cosine_similarity": float(res.nbr_sim[qi, rank - 1]),
                                        "sequence_identity": (pident_lookup.get((qh, jh), 0.0)
                                                              if pident_lookup else None),
                                        "query_substrate_count": nsub_by_idx[qi],
                                        "neighbor_substrate_count": nsub_by_idx[j],
                                        "substrate_set_similarity": computer.coh(qi, j),
                                        "query_substrates": ";".join(sub_by_idx[qi]),
                                        "neighbor_substrates": ";".join(sub_by_idx[j]),
                                    })

    # ---- sanity: self-vs-self delta ≈ 0 (用 base embedding 与自身比较一格) ----
    sanity[comparison] = _self_delta_check(cohort, chem, args)

def res_a_row_identity(res, qi, k, eff_hashes, pident_lookup) -> float:
    nbr = res.nbr_idx[qi, :k]
    nbr_h = [eff_hashes[int(j)] for j in nbr if j >= 0]
    if not pident_lookup or not nbr_h:
        return float("nan")
    return float(np.mean([pident_lookup.get((eff_hashes[qi], h), 0.0) for h in nbr_h]))

def _self_delta_check(cohort, chem, args) -> dict:
    """同一 embedding 与自身比较，paired delta 必须为 0 (sanity check #10)。"""
    emb = l2_normalize(cohort.emb_a)
    n = len(cohort.seq_hashes)
    sub_by_idx = [cohort.substrates[h] for h in cohort.seq_hashes]
    pools = build_eligible_pools(n, None)
    k = min(args.k_values)
    computer = CoherenceComputer(sub_by_idx, chem, "c_sym")
    draws = sample_random_neighbors(pools, k, 10, args.seed)
    rmean, rstd = random_baseline(computer, draws)
    r1 = compute_model_condition(emb, pools, k, computer, draws, rmean, rstd, args.batch_size)
    r2 = compute_model_condition(emb, pools, k, computer, draws, rmean, rstd, args.batch_size)
    valid = sorted(set(r1.coherence) & set(r2.coherence))
    delta = np.array([r2.coherence[q] - r1.coherence[q] for q in valid])
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    return {"self_vs_self_max_abs_delta": max_abs, "n_queries": len(valid)}

# =============================================================================
# Main
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="enzyme-only substrate neighbourhood coherence analysis")
    p.add_argument("--config", type=str, default=None, help="optional YAML with all settings")
    p.add_argument("--model-files", type=str, nargs="+", default=None,
                   help="name=path entries; path = per-model ESP pkl with enzyme_vector")
    p.add_argument("--model-metadata", type=str, default=None,
                   help="YAML mapping model name -> {backbone, mode, pkl}")
    p.add_argument("--comparisons", type=str, nargs="+", default=None,
                   help="modelA:modelB pairs (primary = base:base_sub)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--identity-cutoffs", type=float, nargs="+", default=[0.30, 0.40, 0.50])
    p.add_argument("--min-alignment-coverage", type=float, default=0.80)
    p.add_argument("--homology-table", type=str, default=None)
    p.add_argument("--random-repeats", type=int, default=200)
    p.add_argument("--fingerprint-radius", type=int, default=2)
    p.add_argument("--fingerprint-bits", type=int, default=2048)
    p.add_argument("--currency-metabolite-file", type=str, default=None)
    p.add_argument("--max-substrate-enzyme-fraction", type=float, default=None)
    p.add_argument("--extra-set-metrics", type=str, nargs="*", default=[],
                   help="optional: mean_pairwise median_pairwise jaccard")
    p.add_argument("--audit-k", type=int, default=10, help="k used for neighbor_pairs.csv audit")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--save-full-sequence", action="store_true",
                   help="include full protein sequence in enzyme_id_map.csv")
    p.add_argument("--self-test", action="store_true", help="run synthetic correctness test and exit")
    return p

def merge_scalar_config(args, cfg):
    """用 config yaml 填补未在 CLI 指定的标量参数。"""
    a = cfg.get("analysis", {})
    mapping = {
        "output_dir": "output_dir", "k_values": "k_values",
        "identity_cutoffs": "identity_cutoffs", "min_alignment_coverage": "min_alignment_coverage",
        "homology_table": "homology_table", "random_repeats": "random_repeats",
        "fingerprint_radius": "fingerprint_radius", "fingerprint_bits": "fingerprint_bits",
        "currency_metabolite_file": "currency_metabolite_file",
        "max_substrate_enzyme_fraction": "max_substrate_enzyme_fraction",
        "extra_set_metrics": "extra_set_metrics", "audit_k": "audit_k",
        "n_boot": "n_boot", "n_perm": "n_perm", "batch_size": "batch_size", "seed": "seed",
    }
    defaults = build_argparser().parse_args([])
    for attr, key in mapping.items():
        if key in a and getattr(args, attr) == getattr(defaults, attr):
            setattr(args, attr, a[key])
    return args

def main():
    args = build_argparser().parse_args()

    if args.self_test:
        from subcoh.selftest import run_self_test
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    cfg = load_yaml(args.config)
    args = merge_scalar_config(args, cfg)
    if not args.output_dir:
        raise ValueError("--output-dir is required (or analysis.output_dir in config)")
    os.makedirs(args.output_dir, exist_ok=True)

    models_meta = resolve_models(args, cfg)
    comparisons = resolve_comparisons(args, cfg, models_meta)

    chem = SubstrateChemistry(radius=args.fingerprint_radius, n_bits=args.fingerprint_bits)

    # 仅加载比较中实际用到的模型
    used = sorted({m for c in comparisons for m in c})
    model_data = {}
    data_summary = {}
    for name in used:
        m = models_meta[name]
        md = load_and_aggregate(name, m["pkl"], m["backbone"], m["mode"], chem)
        model_data[name] = md
        data_summary[name] = {"backbone": m["backbone"], "mode": m["mode"],
                              "pkl": m["pkl"], **md.stats}

    id_map = build_global_id_map(list(model_data.values()))

    homology = HomologyTable.from_tsv(args.homology_table) if args.homology_table else None
    pident_lookup = homology.pident_lookup() if homology is not None else None

    per_query_rows: List[dict] = []
    summary_rows: List[dict] = []
    neighbor_rows: List[dict] = []
    excluded_tables: List[pd.DataFrame] = []
    sanity: dict = {}

    for a, b in comparisons:
        logger.info("=== comparison: %s vs %s ===", a, b)
        run_comparison(a, b, model_data[a], model_data[b], id_map, chem,
                       homology, pident_lookup, args,
                       per_query_rows, summary_rows, neighbor_rows,
                       excluded_tables, sanity)

    # ---- 写出 ----
    od = args.output_dir
    pd.DataFrame(per_query_rows).to_csv(os.path.join(od, "per_query_coherence.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(od, "paired_comparison_summary.csv"), index=False)
    if neighbor_rows:
        pd.DataFrame(neighbor_rows).to_csv(os.path.join(od, "neighbor_pairs.csv"), index=False)
    if excluded_tables:
        pd.concat(excluded_tables, ignore_index=True).to_csv(
            os.path.join(od, "excluded_substrates.csv"), index=False)

    # enzyme_id_map
    id_rows = []
    seq_lookup = {}
    for md in model_data.values():
        seq_lookup.update(md.sequences)
    for h, eid in sorted(id_map.items(), key=lambda kv: kv[1]):
        row = {"enzyme_id": eid, "sequence_hash": h}
        if args.save_full_sequence:
            row["sequence"] = seq_lookup.get(h, "")
        id_rows.append(row)
    pd.DataFrame(id_rows).to_csv(os.path.join(od, "enzyme_id_map.csv"), index=False)

    with open(os.path.join(od, "data_summary.json"), "w") as f:
        json.dump(data_summary, f, indent=2)
    with open(os.path.join(od, "analysis_config.json"), "w") as f:
        json.dump({**vars(args), "timestamp": datetime.now().isoformat(),
                   "comparisons": comparisons}, f, indent=2)
    with open(os.path.join(od, "sanity_report.json"), "w") as f:
        json.dump(sanity, f, indent=2)

    logger.info("Done. Outputs in %s", od)
    logger.info("Self-vs-self max|delta| per comparison: %s",
                {c: round(v["self_vs_self_max_abs_delta"], 9) for c, v in sanity.items()})

if __name__ == "__main__":
    main()
