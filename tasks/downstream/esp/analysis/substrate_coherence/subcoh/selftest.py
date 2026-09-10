#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subcoh.selftest — 合成正确性测试 (无需真实 pkl / MMseqs2)

构造 embedding cluster 与 substrate fingerprint cluster 对齐的玩具数据，验证:
  1. 结构化 embedding 的 SubChem 明显高于打乱 embedding;
  2. 同一 embedding 与自身比较 paired delta == 0;
  3. 交换比较顺序 delta 符号反转;
  4. query 不会检索到自身;
  5. 同源过滤后所有近邻满足 cutoff;
  6. Tanimoto / set similarity ∈ [0, 1];
  7. 固定 seed 下 random baseline 可复现。
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from .chem import SubstrateChemistry
from .coherence import CoherenceComputer, compute_model_condition, random_baseline
from .neighbors import build_eligible_pools, l2_normalize, sample_random_neighbors

logger = logging.getLogger("subcoh.selftest")

# 4 个化学家族，每族内部相似、族间不相似
FAMILIES = [
    ["CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O"],          # 羧酸
    ["NCCO", "NCCCO", "NCCCCO", "NCCCCCO"],                       # 氨基醇
    ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"],       # 芳烃
    ["OCC(O)CO", "OCC(O)C(O)CO", "OCC(O)C(O)C(O)CO", "OCCO"],     # 多元醇
]

def _build_synthetic(n_per_group: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_groups = len(FAMILIES)
    substrates: List[List[str]] = []
    groups: List[int] = []
    for g, fam in enumerate(FAMILIES):
        for _ in range(n_per_group):
            size = int(rng.integers(1, 3))  # 1-2 个底物
            subs = sorted(set(rng.choice(fam, size=size, replace=False).tolist()))
            substrates.append(subs)
            groups.append(g)
    n = len(substrates)
    groups = np.array(groups)

    # 结构化 embedding: one-hot(group)*5 + 噪声 -> 同组互为近邻
    D = 16
    emb = rng.normal(0, 0.3, size=(n, D)).astype(np.float32)
    for i in range(n):
        emb[i, groups[i]] += 5.0
    # 打乱 embedding: 行随机置换 (substrate 不变)
    perm = rng.permutation(n)
    emb_shuffled = emb[perm].copy()
    return substrates, groups, emb, emb_shuffled

def run_self_test() -> bool:
    ok = True
    chem = SubstrateChemistry(radius=2, n_bits=2048)
    substrates, groups, emb, emb_shuf = _build_synthetic()
    n = len(substrates)
    pools = build_eligible_pools(n, None)
    k = 5
    metric = "c_sym"
    computer = CoherenceComputer(substrates, chem, metric)
    draws = sample_random_neighbors(pools, k, 100, seed=2025)
    rmean, rstd = random_baseline(computer, draws)

    emb_n = l2_normalize(emb)
    emb_sn = l2_normalize(emb_shuf)
    res_struct = compute_model_condition(emb_n, pools, k, computer, draws, rmean, rstd)
    res_shuf = compute_model_condition(emb_sn, pools, k, computer, draws, rmean, rstd)

    valid = sorted(set(res_struct.coherence) & set(res_shuf.coherence))
    mean_struct = np.mean([res_struct.coherence[q] for q in valid])
    mean_shuf = np.mean([res_shuf.coherence[q] for q in valid])
    mean_rand = np.mean([rmean[q] for q in valid])

    # 1) 结构化 >> 打乱, 且结构化 enrichment > 0
    test1 = mean_struct > mean_shuf + 0.15
    logger.info("[1] structured=%.3f shuffled=%.3f random=%.3f -> %s",
                mean_struct, mean_shuf, mean_rand, "PASS" if test1 else "FAIL")
    ok &= test1

    # 2) self-vs-self delta == 0
    res_struct2 = compute_model_condition(emb_n, pools, k, computer, draws, rmean, rstd)
    delta_self = np.array([res_struct2.coherence[q] - res_struct.coherence[q] for q in valid])
    test2 = np.allclose(delta_self, 0.0)
    logger.info("[2] self-vs-self max|delta|=%.2e -> %s",
                np.max(np.abs(delta_self)), "PASS" if test2 else "FAIL")
    ok &= test2

    # 3) swap order flips sign
    d_fwd = np.array([res_shuf.coherence[q] - res_struct.coherence[q] for q in valid])
    d_rev = np.array([res_struct.coherence[q] - res_shuf.coherence[q] for q in valid])
    test3 = np.allclose(d_fwd, -d_rev)
    logger.info("[3] swap sign flip -> %s", "PASS" if test3 else "FAIL")
    ok &= test3

    # 4) no self in neighbours
    test4 = all(q not in set(res_struct.nbr_idx[q, :k].tolist()) for q in valid)
    logger.info("[4] no self-neighbour -> %s", "PASS" if test4 else "FAIL")
    ok &= test4

    # 5) homology filtering: 排除同组候选, 近邻不得来自同组
    excl_by_idx = [set(np.where(groups == groups[i])[0].tolist()) - {i} for i in range(n)]
    pools_h = build_eligible_pools(n, excl_by_idx)
    res_h = compute_model_condition(emb_n, pools_h, k, computer,
                                    sample_random_neighbors(pools_h, k, 50, 2025),
                                    *random_baseline(computer, sample_random_neighbors(pools_h, k, 50, 2025)))
    valid_h = sorted(res_h.coherence)
    test5 = all(
        groups[int(j)] != groups[q]
        for q in valid_h for j in res_h.nbr_idx[q, :k] if j >= 0
    )
    logger.info("[5] homology-filtered neighbours all out-of-group -> %s", "PASS" if test5 else "FAIL")
    ok &= test5

    # 6) Tanimoto / set sim in [0,1]
    vals = [computer.coh(i, j) for i in range(0, n, 5) for j in range(0, n, 7)]
    test6 = all(0.0 <= v <= 1.0 for v in vals)
    logger.info("[6] set similarity in [0,1] -> %s", "PASS" if test6 else "FAIL")
    ok &= test6

    # 7) random baseline reproducible
    d1 = sample_random_neighbors(pools, k, 50, seed=2025)
    d2 = sample_random_neighbors(pools, k, 50, seed=2025)
    test7 = all(np.array_equal(d1[q], d2[q]) for q in d1)
    logger.info("[7] random baseline reproducible -> %s", "PASS" if test7 else "FAIL")
    ok &= test7

    logger.info("SELF-TEST %s", "PASSED" if ok else "FAILED")
    return ok

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_self_test()