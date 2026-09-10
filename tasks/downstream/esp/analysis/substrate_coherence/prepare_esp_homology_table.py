#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_esp_homology_table.py
为 enzyme-only 底物化学一致性分析 (3_analyze_substrate_neighborhood_coherence.py)
准备序列同源关系表。

为什么需要这个脚本
-------------------
序列同源是本分析最重要的混杂因素：如果 SUB 模型只是把序列更相似的酶检索为近邻，
那么底物化学相似度升高并不能证明它获得了额外的底物化学组织能力。因此主分析需要
在「低同源近邻」条件下复核结论。本脚本生成一张所有模型共享的标准化同源表，供
主分析按 identity cutoff 过滤近邻候选。

由于所有比较模型 (base / sub / cpt ...) 共享同一份 ESP 测试集，cohort 完全相同，
所以 MMseqs2 只需运行一次，结果对所有模型复用。不要为每个模型重复跑。

流程
----
1. 从一个或多个 pkl 读取 `sequence`，按完整序列去重；
2. 用与主分析一致的 sha1(seq) 作为 seq_hash，写出 FASTA (header = seq_hash)；
3. 调用 MMseqs2 `easy-search` 做 all-vs-all 比对；
4. 把 MMseqs2 raw 输出转换成标准化 TSV：
       query_id  target_id  pident  query_coverage  target_coverage
   其中 query_id / target_id 即 seq_hash，pident 归一化到 [0, 1]。

注意
----
- seq_hash 与 subcoh.data.seq_hash 完全一致 (sha1, utf-8)，保证主分析能对齐。
- MMseqs2 是独立二进制 (不是 pip 包)，需预先安装并在 PATH 中，或用 --mmseqs 指定路径。
- 若你的环境无法直接跑 MMseqs2，可用 --fasta-only 只导出 FASTA，手动比对后再用
  --raw-m8 + --convert-only 把已有 m8 结果转换为标准表。

用法
----
# 全流程 (需要 mmseqs)
python prepare_esp_homology_table.py \
    --input-pkls /path/to/base/ID_Test.pkl \
    --output-table /path/to/mmseqs_hits.tsv \
    --tmp-dir /path/to/tmp \
    --min-seq-id 0.25 --coverage 0.0 --threads 16

# 只导出 FASTA (在别处比对)
python prepare_esp_homology_table.py \
    --input-pkls /path/to/base/ID_Test.pkl \
    --fasta-only --output-fasta enzymes.fasta

# 把已有 m8 结果转换为标准表
python prepare_esp_homology_table.py \
    --input-pkls /path/to/base/ID_Test.pkl \
    --convert-only --raw-m8 hits.m8 --output-table mmseqs_hits.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_homology")

# 必须与 subcoh.data.seq_hash 保持一致
def seq_hash(sequence: str) -> str:
    """sha1(utf-8) — 与主分析 subcoh.data.seq_hash 完全一致。"""
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()

# MMseqs2 easy-search 默认 BLAST tab (m8) 列含义，覆盖列取自 --format-output。
# 我们显式请求一组列以便稳定解析。
FORMAT_OUTPUT = "query,target,pident,qcov,tcov,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
FORMAT_COLS = FORMAT_OUTPUT.split(",")

def collect_unique_sequences(pkls: List[str]) -> Dict[str, str]:
    """从所有 pkl 读取 sequence，去重，返回 {seq_hash: sequence}。"""
    seen: Dict[str, str] = {}
    for p in pkls:
        if not os.path.exists(p):
            raise FileNotFoundError(f"input pkl not found: {p}")
        df = pd.read_pickle(p)
        if "sequence" not in df.columns:
            raise ValueError(f"{p} missing 'sequence' column")
        for s in df["sequence"].dropna().unique():
            if isinstance(s, str) and len(s) > 0:
                seen[seq_hash(s)] = s
    logger.info("Collected %d unique sequences from %d pkl(s)", len(seen), len(pkls))
    return seen

def write_fasta(seqs: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        for h, s in seqs.items():
            f.write(f">{h}\n{s}\n")
    logger.info("Wrote FASTA (%d records): %s", len(seqs), path)

def run_mmseqs_easy_search(
    fasta: str, out_m8: str, tmp_dir: str, mmseqs_bin: str,
    min_seq_id: float, coverage: float, sensitivity: float, threads: int,
    max_seqs: int,
) -> None:
    """all-vs-all：query=target=同一 FASTA。"""
    if shutil.which(mmseqs_bin) is None and not os.path.exists(mmseqs_bin):
        raise FileNotFoundError(
            f"MMseqs2 binary '{mmseqs_bin}' not found. Install MMseqs2 and put it on PATH, "
            f"or pass --mmseqs /path/to/mmseqs, or use --fasta-only / --convert-only."
        )
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [
        mmseqs_bin, "easy-search", fasta, fasta, out_m8, tmp_dir,
        "--format-output", FORMAT_OUTPUT,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", "0",          # 覆盖按 query 和 target 两侧
        "-s", str(sensitivity),
        "--max-seqs", str(max_seqs),
        "--threads", str(threads),
        "-a", "1",                  # 输出 backtrace，保证 pident/cov 可计算
    ]
    logger.info("Running MMseqs2: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout or "")
        raise RuntimeError(f"MMseqs2 failed (returncode={proc.returncode})")
    logger.info("MMseqs2 done -> %s", out_m8)

def convert_m8_to_standard(raw_m8: str, out_table: str) -> None:
    """把 MMseqs2 m8 输出转换为标准化同源表。"""
    if not os.path.exists(raw_m8):
        raise FileNotFoundError(f"raw m8 not found: {raw_m8}")
    df = pd.read_csv(raw_m8, sep="\t", header=None, names=FORMAT_COLS)
    logger.info("Loaded raw m8: %s (%d rows)", raw_m8, len(df))

    # pident 归一化到 [0,1] (MMseqs2 的 pident 通常已是 0-1，但有的版本是 0-100)
    pmax = float(df["pident"].max()) if len(df) else 0.0
    if pmax > 1.0001:
        logger.info("pident appears to be percent (max=%.2f); dividing by 100", pmax)
        df["pident"] = df["pident"] / 100.0

    std = pd.DataFrame({
        "query_id": df["query"].astype(str),
        "target_id": df["target"].astype(str),
        "pident": df["pident"].astype(float),
        "query_coverage": df["qcov"].astype(float),
        "target_coverage": df["tcov"].astype(float),
    })
    # 去掉自比对行 (q==t)；主分析也会排除，但提前去掉让表更干净更小
    before = len(std)
    std = std[std["query_id"] != std["target_id"]].reset_index(drop=True)
    logger.info("Dropped %d self-hits (q==t)", before - len(std))

    os.makedirs(os.path.dirname(os.path.abspath(out_table)) or ".", exist_ok=True)
    std.to_csv(out_table, sep="\t", index=False)
    logger.info("Wrote standardized homology table (%d hits): %s", len(std), out_table)
    # 简单分布报告
    for cut in (0.30, 0.40, 0.50):
        n = int((std["pident"] >= cut).sum())
        logger.info("  hits with pident >= %.2f : %d", cut, n)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare standardized MMseqs2 homology table for subcoh analysis")
    p.add_argument("--input-pkls", type=str, nargs="+", required=True,
                   help="一个或多个 ESP pkl (任意模型即可，cohort 相同)；只读 sequence 列")
    p.add_argument("--output-table", type=str, default=None,
                   help="标准化同源表输出路径 (TSV)")
    p.add_argument("--output-fasta", type=str, default=None,
                   help="FASTA 输出路径 (默认在 output-table 同目录 enzymes.fasta)")
    p.add_argument("--tmp-dir", type=str, default=None, help="MMseqs2 临时目录")
    p.add_argument("--mmseqs", type=str, default="mmseqs", help="MMseqs2 二进制路径")
    # 比对敏感度参数
    p.add_argument("--min-seq-id", type=float, default=0.25,
                   help="MMseqs2 预过滤的最低 identity；应低于最严的分析 cutoff (0.30) 以免漏掉边界对")
    p.add_argument("--coverage", type=float, default=0.0,
                   help="MMseqs2 -c 覆盖阈值；设 0 让主分析自己按 --min-alignment-coverage 过滤")
    p.add_argument("--sensitivity", type=float, default=7.5, help="MMseqs2 -s 灵敏度")
    p.add_argument("--max-seqs", type=int, default=10000, help="每个 query 保留的最大 target 数")
    p.add_argument("--threads", type=int, default=8)
    # 模式开关
    p.add_argument("--fasta-only", action="store_true", help="只导出 FASTA，不跑 MMseqs2")
    p.add_argument("--convert-only", action="store_true", help="只把已有 m8 转换为标准表")
    p.add_argument("--raw-m8", type=str, default=None, help="convert-only 模式下的输入 m8 路径")
    p.add_argument("--keep-m8", action="store_true", help="保留 MMseqs2 raw m8 中间文件")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    seqs = collect_unique_sequences(args.input_pkls)
    if not seqs:
        raise ValueError("No sequences collected; nothing to do.")

    # FASTA 路径
    if args.output_fasta:
        fasta_path = args.output_fasta
    elif args.output_table:
        fasta_path = os.path.join(os.path.dirname(os.path.abspath(args.output_table)) or ".", "enzymes.fasta")
    else:
        fasta_path = "enzymes.fasta"

    # 模式 1：只转换已有 m8
    if args.convert_only:
        if not args.raw_m8 or not args.output_table:
            raise ValueError("--convert-only requires --raw-m8 and --output-table")
        # 仍写一份 FASTA 方便对照
        write_fasta(seqs, fasta_path)
        convert_m8_to_standard(args.raw_m8, args.output_table)
        return

    # 模式 2：只导出 FASTA
    write_fasta(seqs, fasta_path)
    if args.fasta_only:
        logger.info("FASTA-only mode; run MMseqs2 externally then re-run with --convert-only.")
        return

    # 模式 3：全流程
    if not args.output_table:
        raise ValueError("--output-table is required (unless --fasta-only)")
    tmp_dir = args.tmp_dir or os.path.join(os.path.dirname(os.path.abspath(args.output_table)) or ".", "mmseqs_tmp")
    raw_m8 = os.path.join(os.path.dirname(os.path.abspath(args.output_table)) or ".", "mmseqs_raw_hits.m8")

    run_mmseqs_easy_search(
        fasta=fasta_path, out_m8=raw_m8, tmp_dir=tmp_dir, mmseqs_bin=args.mmseqs,
        min_seq_id=args.min_seq_id, coverage=args.coverage,
        sensitivity=args.sensitivity, threads=args.threads, max_seqs=args.max_seqs,
    )
    convert_m8_to_standard(raw_m8, args.output_table)

    if not args.keep_m8 and os.path.exists(raw_m8):
        os.remove(raw_m8)
        logger.info("Removed intermediate m8 (use --keep-m8 to keep).")

if __name__ == "__main__":
    main()