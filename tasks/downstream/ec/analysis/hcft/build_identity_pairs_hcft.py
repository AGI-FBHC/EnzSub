#!/usr/bin/env python3
"""
build_identity_pairs_hcft.py — 为 HCFT 生成 pairwise sequence identity 文件

流程:
  1. 读 GraphEC gallery FASTA + EC label CSV
  2. 在指定 --ec-level 下标准化 EC 标签, 只保留有有效标签的序列
  3. 写出 filtered_gallery.fasta (只含有有效标签的序列)
  4. 用 MMseqs2 对 filtered gallery 做 self-search → 得到可比对命中 pair 的 identity
     (或用 --precomputed-pairs 直接读已有结果, 跳过 MMseqs)
  5. 解析 / 流式处理 MMseqs 输出: 去 self pair、undirected 去重 (取 max identity)
  6. 按顶部配置标注 identity_bin、functional_match 和 EC-prefix relation
  7. 写 HCFT-ready pair_identity.csv 及若干诊断文件

重要事实:
  * MMseqs self-search 只输出"能比对上的"命中 pair, 不是全量 all-vs-all dense matrix。
    低 identity 区间 (尤其 0–30) 的 pair 数可能很少甚至为 0, 因为远缘同源/无同源的
    pair 根本不会被 MMseqs 召回。这不是 bug, 是 identity 来源的固有性质。
    identity_distribution.csv 给出每个 bin 的真实 pos/neg 数, 据此再调 bins。
  * 不构建 Python 内存中的 dense all-vs-all matrix; 全程基于 MMseqs 命中流式处理。

使用方式:
  直接修改本文件顶部 USER_CONFIG, 然后运行:
       python build_identity_pairs_hcft.py

  默认不读取命令行参数，避免在服务器上运行时忘记传参或传错参数。
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ---- 复用 ../sweep 的工具, 并导入本目录公共逻辑 ----
from hfs_common import (
    setup_sweep_path,
    build_labelsets,
    functional_match,
    make_bins,
    assign_bin,
    main_class_of,
    scale_identity_to_percent,
    VALID_MATCH_MODES,
)

setup_sweep_path()
from utils import read_fasta, load_ec_labels  # noqa: E402  (from ../sweep)

# ======================================================================
# 顶部参数配置
# ======================================================================
# 默认使用这里的参数。这样可以直接运行:
#   python build_identity_pairs_hcft.py
#
# 默认忽略命令行参数；如确实需要临时覆盖，可把 ALLOW_CLI_OVERRIDE 改成 True。
USE_TOP_CONFIG = True
ALLOW_CLI_OVERRIDE = False

USER_CONFIG = {
    # ---- 输入输出 ----
    "gallery_fasta": "data/ec/split100.fasta",
    "gallery_csv": "data/ec/split100.csv",
    "output_dir": "artifacts/downstream/ec/hcft/identity_pairs/split100",

    # ---- HCFT 标签定义 ----
    "ec_level": 4,
    "match_mode": "any_overlap",

    # 默认 identity bins: 5% 一个区间，便于后续 HCFT 按 identity gap 采样
    "bins": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],

    # ---- MMseqs 参数 ----
    "threads": 32,
    "sensitivity": 8.5,
    "evalue": 1e-1,
    "max_seqs": 100000,
    "min_seq_id": 0.0,

    # ---- 如果已经有 pairwise identity 文件，就填路径；否则保持 None，自动跑 MMseqs ----
    "precomputed_pairs": None,

    # ---- 文件解析与中间文件 ----
    "chunksize": 500_000,
    "keep_mmseqs_tmp": False,
}

# ======================================================================
# 参数解析
# ======================================================================
def build_argparser():
    ap = argparse.ArgumentParser(
        description="为 HCFT 生成 pairwise sequence identity 文件 (MMseqs2 self-search)"
    )

    # ---- 输入输出 ----
    ap.add_argument(
        "--gallery-fasta",
        default=USER_CONFIG["gallery_fasta"],
        help="GraphEC gallery FASTA"
    )
    ap.add_argument(
        "--gallery-csv",
        default=USER_CONFIG["gallery_csv"],
        help="GraphEC gallery EC label CSV/TSV"
    )
    ap.add_argument(
        "--output-dir",
        default=USER_CONFIG["output_dir"],
        help="输出目录"
    )

    # ---- EC / match mode ----
    ap.add_argument(
        "--ec-level",
        type=int,
        default=USER_CONFIG["ec_level"],
        choices=[1, 2, 3, 4],
        help="EC 层级: 1/2/3/4，默认 4"
    )
    ap.add_argument(
        "--match-mode",
        default=USER_CONFIG["match_mode"],
        choices=list(VALID_MATCH_MODES),
        help="多标签 pair 判定模式"
    )
    ap.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default=USER_CONFIG["bins"],
        help="identity bin 边界，例如 0 30 50 70 100；默认由 USER_CONFIG 控制"
    )

    # ---- MMseqs 参数 ----
    ap.add_argument(
        "--threads",
        type=int,
        default=USER_CONFIG["threads"],
        help="MMseqs 使用线程数"
    )
    ap.add_argument(
        "--sensitivity",
        type=float,
        default=USER_CONFIG["sensitivity"],
        help="mmseqs -s; 越大越敏感，召回更多远缘 pair"
    )
    ap.add_argument(
        "--evalue",
        type=float,
        default=USER_CONFIG["evalue"],
        help="mmseqs -e; 放宽可召回更多低 identity pair"
    )
    ap.add_argument(
        "--max-seqs",
        type=int,
        default=USER_CONFIG["max_seqs"],
        help="mmseqs --max-seqs; 每条 query 保留的最大命中数"
    )
    ap.add_argument(
        "--min-seq-id",
        type=float,
        default=USER_CONFIG["min_seq_id"],
        help="mmseqs --min-seq-id; HCFT 需要低 identity positive 和高 identity negative，一般保持 0"
    )

    # ---- 跳过 MMseqs ----
    ap.add_argument(
        "--precomputed-pairs",
        default=USER_CONFIG["precomputed_pairs"],
        help="已有 pairwise identity TSV/CSV; 给出则跳过 MMseqs"
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=USER_CONFIG["chunksize"],
        help="解析 pairwise 文件时的分块行数"
    )

    # 布尔参数：既支持 --keep-mmseqs-tmp，也支持 --no-keep-mmseqs-tmp
    ap.add_argument(
        "--keep-mmseqs-tmp",
        dest="keep_mmseqs_tmp",
        action="store_true",
        default=USER_CONFIG["keep_mmseqs_tmp"],
        help="保留 MMseqs 中间 DB"
    )
    ap.add_argument(
        "--no-keep-mmseqs-tmp",
        dest="keep_mmseqs_tmp",
        action="store_false",
        help="不保留 MMseqs 中间 DB"
    )

    return ap

def get_args():
    ap = build_argparser()

    if USE_TOP_CONFIG:
        if ALLOW_CLI_OVERRIDE:
            # 默认读取 USER_CONFIG；如果命令行传入参数，则覆盖对应项
            args = ap.parse_args()
        else:
            # 完全忽略命令行，只使用 USER_CONFIG
            args = ap.parse_args([])
    else:
        # 传统命令行模式
        args = ap.parse_args()

    # 必需参数检查
    required = ["gallery_csv", "output_dir"]
    if args.precomputed_pairs is None:
        required.append("gallery_fasta")

    for name in required:
        value = getattr(args, name)
        if value is None or str(value).strip() == "":
            ap.error(f"缺少必要参数: {name}")

    return args

# ======================================================================
# MMseqs2
# ======================================================================
def check_mmseqs() -> str:
    """确认 mmseqs 可执行; 找不到则明确报错 (不实现 Python fallback)。"""
    path = shutil.which("mmseqs")
    if path is None:
        raise RuntimeError(
            "未找到可执行的 'mmseqs'。请确认 MMseqs2 已安装且在 PATH 中。\n"
            "本脚本不提供无 MMseqs 的 Python fallback; "
            "如已有 pairwise identity 文件, 请改用 --precomputed-pairs。"
        )
    return path

def run_mmseqs_self_search(
    filtered_fasta: str,
    out_tsv: str,
    work_dir: str,
    threads: int,
    sensitivity: float,
    evalue: float,
    max_seqs: int,
    min_seq_id: float,
    log,
) -> list:
    """对 filtered_fasta 跑 mmseqs self-search, 输出 convertalis TSV 到 out_tsv。

    返回执行过的命令列表 (供写入 config / log)。
    """
    query_db = os.path.join(work_dir, "queryDB")
    result_db = os.path.join(work_dir, "resultDB")
    tmp_dir = os.path.join(work_dir, "mmseqs_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    fmt = (
        "query,target,pident,alnlen,mismatch,gapopen,"
        "qstart,qend,tstart,tend,evalue,bits"
    )

    cmds = [
        ["mmseqs", "createdb", filtered_fasta, query_db],
        [
            "mmseqs", "search", query_db, query_db, result_db, tmp_dir,
            "--threads", str(threads),
            "-s", str(sensitivity),
            "-e", str(evalue),
            "--max-seqs", str(max_seqs),
            "--min-seq-id", str(min_seq_id),
        ],
        [
            "mmseqs", "convertalis", query_db, query_db, result_db, out_tsv,
            "--threads", str(threads),
            "--format-output", fmt,
        ],
    ]

    executed = []
    for cmd in cmds:
        cmd_str = " ".join(cmd)
        log(f"  MMseqs CMD: {cmd_str}")
        executed.append(cmd_str)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # 记录 mmseqs 自身输出。只记录末尾，避免 log 爆炸。
        if proc.stdout:
            tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
            log(f"    [mmseqs stdout tail]\n{tail}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"MMseqs 命令失败 (code {proc.returncode}): {cmd_str}\n"
                f"完整输出见上方 log。"
            )

    return executed

# ======================================================================
# 解析 pairwise identity (MMseqs TSV 或 precomputed)
# ======================================================================
# convertalis 指定的列顺序
_MMSEQS_COLS = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
]

# precomputed 文件的列名兼容
_ID1_ALIASES = ["id1", "seq_id1", "query", "qseqid", "protein1"]
_ID2_ALIASES = ["id2", "seq_id2", "target", "sseqid", "protein2"]
_PID_ALIASES = [
    "identity", "seq_identity", "pident", "pid", "sequence_identity"
]

def _detect_columns(df_cols, log) -> tuple:
    """从已有 df 列名里识别 (id1_col, id2_col, identity_col)。"""
    cols_lower = {c.lower(): c for c in df_cols}

    def pick(aliases, what):
        for a in aliases:
            if a.lower() in cols_lower:
                return cols_lower[a.lower()]
        raise ValueError(
            f"无法在 precomputed 文件中识别 {what} 列。\n"
            f"  已有列: {list(df_cols)}\n"
            f"  支持的别名: {aliases}"
        )

    id1 = pick(_ID1_ALIASES, "id1")
    id2 = pick(_ID2_ALIASES, "id2")
    pid = pick(_PID_ALIASES, "identity")
    log(f"  列识别: id1={id1!r}  id2={id2!r}  identity={pid!r}")
    return id1, id2, pid

def _sniff_sep(path: str) -> str:
    """根据首行猜测分隔符 (\\t 或 ,)。"""
    with open(path) as f:
        first = f.readline()
    return "\t" if first.count("\t") >= first.count(",") else ","

def stream_dedup_pairs(
    pairs_path: str,
    is_mmseqs_raw: bool,
    chunksize: int,
    log,
) -> pd.DataFrame:
    """流式读取 pairwise 文件, 去 self pair、undirected 折叠取 max identity。

    返回 DataFrame[id1, id2, identity]  (id1<id2, identity 已是 0–100, 去重后)。

    - is_mmseqs_raw=True: 文件是本脚本生成的 convertalis TSV, 无表头, 列顺序固定。
    - is_mmseqs_raw=False: precomputed 文件, 有表头, 列名自动识别。

    注意:
      这里不构建 dense all-vs-all matrix。
      但如果 MMseqs 输出特别大，最终 unique hit pair 仍然需要进入内存。
    """
    accum = []
    n_raw = 0
    n_self = 0
    n_bad_identity = 0

    if is_mmseqs_raw:
        reader = pd.read_csv(
            pairs_path,
            sep="\t",
            header=None,
            names=_MMSEQS_COLS,
            usecols=["query", "target", "pident"],
            dtype={"query": str, "target": str, "pident": float},
            chunksize=chunksize,
        )
        id1c, id2c, pidc = "query", "target", "pident"
    else:
        sep = _sniff_sep(pairs_path)
        head = pd.read_csv(pairs_path, sep=sep, nrows=0)
        id1c, id2c, pidc = _detect_columns(head.columns, log)
        reader = pd.read_csv(
            pairs_path,
            sep=sep,
            usecols=[id1c, id2c, pidc],
            dtype={id1c: str, id2c: str},
            chunksize=chunksize,
        )

    for chunk in reader:
        n_raw += len(chunk)

        a = chunk[id1c].astype(str).values
        b = chunk[id2c].astype(str).values
        pid = pd.to_numeric(chunk[pidc], errors="coerce").values

        # 去 self
        keep = a != b
        n_self += int((~keep).sum())
        a, b, pid = a[keep], b[keep], pid[keep]

        # 去掉 identity 解析失败
        good = ~np.isnan(pid)
        n_bad_identity += int((~good).sum())
        a, b, pid = a[good], b[good], pid[good]

        if len(a) == 0:
            continue

        # undirected 规范排序
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        small = pd.DataFrame({"id1": lo, "id2": hi, "identity": pid})

        # 块内先折叠一次，减少累积体积
        small = (
            small.groupby(["id1", "id2"], as_index=False, sort=False)["identity"]
            .max()
        )
        accum.append(small)

    if not accum:
        raise ValueError(
            f"pairwise 文件 {pairs_path} 解析后没有任何有效 pair "
            f"(原始行数={n_raw}, self pair={n_self}, bad identity={n_bad_identity})。"
        )

    merged = pd.concat(accum, ignore_index=True)

    # 跨块最终折叠取 max identity
    merged = (
        merged.groupby(["id1", "id2"], as_index=False, sort=False)["identity"]
        .max()
    )

    # 标度归一: 0–1 → 0–100
    merged["identity"] = scale_identity_to_percent(merged["identity"], log)

    log(
        f"  原始行数: {n_raw}; self pair 丢弃: {n_self}; "
        f"bad identity 丢弃: {n_bad_identity}; "
        f"undirected 去重后 unique pair: {len(merged)}"
    )

    return merged

# ======================================================================
# HCFT 辅助: EC prefix relation
# ======================================================================
def ec_prefix_level(ec1: str, ec2: str) -> int:
    """返回两个标准化 EC 标签共享的最长前缀层级。"""
    a = str(ec1).split(".")
    b = str(ec2).split(".")
    n = min(len(a), len(b))
    level = 0
    for i in range(n):
        if a[i] == b[i]:
            level += 1
        else:
            break
    return level

def max_common_ec_prefix(labels1, labels2) -> int:
    """多标签 pair 的最大共享 EC prefix level。"""
    if not labels1 or not labels2:
        return 0
    return max(ec_prefix_level(a, b) for a in labels1 for b in labels2)

def relation_group(match: str, common_prefix: int) -> str:
    """给 HCFT 后续 hard-negative 分层使用。"""
    if match == "positive":
        return "positive_same_ec"
    return f"negative_common_prefix_{common_prefix}"

# ======================================================================
# 主流程
# ======================================================================
def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- logging (同时写文件与控制台) ----
    log_path = os.path.join(args.output_dir, "build_identity_pairs.log")
    log_fh = open(log_path, "w")

    def log(msg=""):
        line = str(msg)
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    try:
        log("=" * 70)
        log(f" build_identity_pairs  |  {datetime.now()}")
        log("=" * 70)
        log(f" gallery_fasta : {args.gallery_fasta}")
        log(f" gallery_csv   : {args.gallery_csv}")
        log(f" output_dir    : {args.output_dir}")
        log(f" ec_level      : {args.ec_level}")
        log(f" match_mode    : {args.match_mode}")
        log(f" bins          : {args.bins}")
        log(f" precomputed   : {args.precomputed_pairs or '(no, will run MMseqs)'}")
        log(f" threads       : {args.threads}")
        log(f" sensitivity   : {args.sensitivity}")
        log(f" evalue        : {args.evalue}")
        log(f" max_seqs      : {args.max_seqs}")
        log(f" min_seq_id    : {args.min_seq_id}")
        log(f" chunksize     : {args.chunksize}")
        log(f" keep_tmp      : {args.keep_mmseqs_tmp}")

        bins = make_bins(args.bins)
        log(f" bin labels    : {[b[2] for b in bins]}")

        # ---- 输入存在性检查 ----
        if not os.path.exists(args.gallery_csv):
            raise FileNotFoundError(f"gallery CSV 不存在: {args.gallery_csv}")

        if args.precomputed_pairs is None and not os.path.exists(args.gallery_fasta):
            raise FileNotFoundError(f"gallery FASTA 不存在: {args.gallery_fasta}")

        # ---- 1) 读标签, 标准化, 取有有效标签的酶 ----
        id_to_raw_ecs, _ = load_ec_labels(args.gallery_csv)
        log(f"\n[labels] CSV 中带非空 EC 字段的序列: {len(id_to_raw_ecs)}")

        id_to_labels = build_labelsets(id_to_raw_ecs, args.ec_level)
        log(
            f"[labels] 在 EC{args.ec_level} 下有 >=1 个有效标签的序列: "
            f"{len(id_to_labels)}"
        )

        if not id_to_labels:
            raise ValueError(
                f"在 EC{args.ec_level} 下没有任何序列有有效标签 — "
                f"检查 ec_level 或标签列。"
            )

        # ---- 2) 写 filtered_gallery.fasta (MMseqs 路径才需要) ----
        filtered_fasta = os.path.join(args.output_dir, "filtered_gallery.fasta")
        n_in_fasta_with_label = 0

        if args.precomputed_pairs is None:
            seq_ids, seqs = read_fasta(args.gallery_fasta)
            id2seq = dict(zip(seq_ids, seqs))

            with open(filtered_fasta, "w") as fo:
                for sid in id_to_labels:
                    if sid in id2seq:
                        fo.write(f">{sid}\n{id2seq[sid]}\n")
                        n_in_fasta_with_label += 1

            log(
                f"[fasta] FASTA 序列数: {len(seq_ids)}; "
                f"写入 filtered_gallery 的 (有标签且有序列): {n_in_fasta_with_label}"
            )

            missing_seq = len(id_to_labels) - n_in_fasta_with_label
            if missing_seq > 0:
                log(
                    f"[fasta] 注意: {missing_seq} 个有标签的 ID 在 FASTA 中缺序列, "
                    f"已忽略"
                )

            if n_in_fasta_with_label == 0:
                raise ValueError(
                    "filtered_gallery 为空: 标签 ID 与 FASTA ID 没有交集?"
                )
        else:
            log("[fasta] 使用 --precomputed-pairs, 跳过 filtered_gallery / MMseqs")

        # ---- 3) 得到 pairwise identity TSV ----
        mmseqs_tsv = os.path.join(args.output_dir, "mmseqs_result.tsv")
        mmseqs_cmds = []

        if args.precomputed_pairs is None:
            check_mmseqs()

            if args.keep_mmseqs_tmp:
                work_dir = os.path.join(args.output_dir, "mmseqs_work")
                os.makedirs(work_dir, exist_ok=True)

                mmseqs_cmds = run_mmseqs_self_search(
                    filtered_fasta=filtered_fasta,
                    out_tsv=mmseqs_tsv,
                    work_dir=work_dir,
                    threads=args.threads,
                    sensitivity=args.sensitivity,
                    evalue=args.evalue,
                    max_seqs=args.max_seqs,
                    min_seq_id=args.min_seq_id,
                    log=log,
                )
            else:
                with tempfile.TemporaryDirectory(prefix="mmseqs_") as work_dir:
                    mmseqs_cmds = run_mmseqs_self_search(
                        filtered_fasta=filtered_fasta,
                        out_tsv=mmseqs_tsv,
                        work_dir=work_dir,
                        threads=args.threads,
                        sensitivity=args.sensitivity,
                        evalue=args.evalue,
                        max_seqs=args.max_seqs,
                        min_seq_id=args.min_seq_id,
                        log=log,
                    )

            pairs_source = mmseqs_tsv
            is_mmseqs_raw = True
        else:
            if not os.path.exists(args.precomputed_pairs):
                raise FileNotFoundError(
                    f"precomputed pairs 不存在: {args.precomputed_pairs}"
                )
            pairs_source = args.precomputed_pairs
            is_mmseqs_raw = False

        # ---- 4) 流式去重 ----
        log(f"\n[parse] 解析 pairwise identity: {pairs_source}")
        pair_df = stream_dedup_pairs(
            pairs_path=pairs_source,
            is_mmseqs_raw=is_mmseqs_raw,
            chunksize=args.chunksize,
            log=log,
        )

        # ---- 5) 标注 bin / functional_match / EC-prefix relation, 写 pair_identity.csv ----
        pair_csv = os.path.join(args.output_dir, "pair_identity.csv")
        log(f"\n[label] 标注 bin / functional_match / EC-prefix relation → {pair_csv}")

        bin_labels = [b[2] for b in bins]
        cnt_pos = defaultdict(int)
        cnt_neg = defaultdict(int)
        cnt_by_class = defaultdict(int)

        n_skip_no_label = 0
        n_skip_ambiguous = 0
        n_skip_out_of_bin = 0
        n_written = 0

        header = [
            "id1", "id2", "identity", "identity_bin", "functional_match",
            "common_ec_prefix_level", "relation_group",
            "ec_level", "match_mode", "shared_labels", "labels1", "labels2",
        ]

        with open(pair_csv, "w", newline="") as fo:
            writer = csv.writer(fo)
            writer.writerow(header)

            for row in pair_df.itertuples(index=False):
                a = row.id1
                b = row.id2
                ident = float(row.identity)

                la = id_to_labels.get(a)
                lb = id_to_labels.get(b)

                if not la or not lb:
                    n_skip_no_label += 1
                    continue

                match = functional_match(la, lb, args.match_mode)

                if match == "skip":
                    n_skip_ambiguous += 1
                    continue

                bin_label = assign_bin(ident, bins)
                if bin_label is None:
                    n_skip_out_of_bin += 1
                    continue

                shared = sorted(la & lb)
                common_prefix = max_common_ec_prefix(la, lb)
                rel_group = relation_group(match, common_prefix)

                writer.writerow([
                    a,
                    b,
                    f"{ident:.4f}",
                    bin_label,
                    match,
                    common_prefix,
                    rel_group,
                    args.ec_level,
                    args.match_mode,
                    ";".join(shared),
                    ";".join(sorted(la)),
                    ";".join(sorted(lb)),
                ])

                n_written += 1

                if match == "positive":
                    cnt_pos[bin_label] += 1

                    # 主类拆分: 共享标签所属的每个 EC main class 各计一次
                    for cls in sorted({main_class_of(ec) for ec in shared}):
                        cnt_by_class[(bin_label, f"pos:class_{cls}")] += 1
                else:
                    cnt_neg[bin_label] += 1
                    cnt_by_class[(bin_label, "negative")] += 1
                    cnt_by_class[(bin_label, rel_group)] += 1

        log(f"[label] 写入 pair: {n_written}")
        log(f"[label] 跳过 - 无有效标签: {n_skip_no_label}")
        log(f"[label] 跳过 - 部分重叠歧义 (no_partial_overlap): {n_skip_ambiguous}")
        log(f"[label] 跳过 - 落在 bins 之外: {n_skip_out_of_bin}")

        # ---- 6) identity_distribution.csv ----
        dist_rows = []
        for left, right, label in bins:
            p = cnt_pos.get(label, 0)
            n = cnt_neg.get(label, 0)
            dist_rows.append({
                "identity_bin": label,
                "bin_left": left,
                "bin_right": right,
                "n_positive": p,
                "n_negative": n,
                "n_total": p + n,
            })

        dist_df = pd.DataFrame(dist_rows)
        dist_csv = os.path.join(args.output_dir, "identity_distribution.csv")
        dist_df.to_csv(dist_csv, index=False)

        log("\n[dist] 每个 identity bin 的 pos/neg pair 数:")
        for r in dist_rows:
            flag = ""
            if r["n_positive"] == 0 or r["n_negative"] == 0:
                flag = "  <-- 注意: 该 bin 某一类为 0, 该 bin 不适合做正负 pair 对比"

            log(
                f"  {r['identity_bin']:>8s}  "
                f"pos={r['n_positive']:>9d}  "
                f"neg={r['n_negative']:>9d}{flag}"
            )

        # ---- 7) pair_counts_by_bin_and_class.csv ----
        byclass_rows = []
        for (label, cat), c in sorted(cnt_by_class.items()):
            left, right = next((l, r) for l, r, lab in bins if lab == label)
            byclass_rows.append({
                "identity_bin": label,
                "bin_left": left,
                "bin_right": right,
                "category": cat,
                "n_pairs": c,
            })

        if byclass_rows:
            byclass_df = pd.DataFrame(byclass_rows)
        else:
            byclass_df = pd.DataFrame(
                columns=[
                    "identity_bin", "bin_left", "bin_right",
                    "category", "n_pairs",
                ]
            )

        byclass_csv = os.path.join(args.output_dir, "pair_counts_by_bin_and_class.csv")
        byclass_df.to_csv(byclass_csv, index=False)

        # ---- 8) config.json ----
        config = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "gallery_fasta": args.gallery_fasta,
            "gallery_csv": args.gallery_csv,
            "output_dir": args.output_dir,
            "ec_level": args.ec_level,
            "match_mode": args.match_mode,
            "bins": args.bins,
            "bin_labels": bin_labels,
            "used_precomputed": args.precomputed_pairs is not None,
            "precomputed_pairs": args.precomputed_pairs,
            "mmseqs": None if args.precomputed_pairs else {
                "threads": args.threads,
                "sensitivity": args.sensitivity,
                "evalue": args.evalue,
                "max_seqs": args.max_seqs,
                "min_seq_id": args.min_seq_id,
                "commands": mmseqs_cmds,
            },
            "n_seqs_with_nonempty_ec_field": len(id_to_raw_ecs),
            "n_seqs_valid_label_at_level": len(id_to_labels),
            "n_seqs_in_filtered_fasta": n_in_fasta_with_label,
            "n_pairs_written": n_written,
            "n_skip_no_label": n_skip_no_label,
            "n_skip_ambiguous": n_skip_ambiguous,
            "n_skip_out_of_bin": n_skip_out_of_bin,
            "outputs": {
                "pair_identity": pair_csv,
                "identity_distribution": dist_csv,
                "pair_counts_by_bin_and_class": byclass_csv,
                "mmseqs_result_tsv": None if args.precomputed_pairs else mmseqs_tsv,
                "filtered_gallery_fasta": None if args.precomputed_pairs else filtered_fasta,
            },
        }

        cfg_path = os.path.join(args.output_dir, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        log(f"\n[done] 输出目录: {args.output_dir}")
        log(f"  pair_identity.csv              : {n_written} pairs")
        log(f"  identity_distribution.csv      : 每 bin pos/neg 计数")
        log(f"  pair_counts_by_bin_and_class.csv")
        log(f"  config.json / build_identity_pairs.log")

        if any(r["n_positive"] == 0 or r["n_negative"] == 0 for r in dist_rows):
            log("\n[!] 有 bin 的 pos 或 neg 为 0 (常见于低 identity 区间)。")
            log("    MMseqs self-search 不召回远缘/无同源 pair 是预期行为, 不是 bug。")
            log("    可据 identity_distribution.csv 合并低 bin, 或放宽 --evalue / 调大 --max-seqs 重跑。")

    finally:
        log_fh.close()

if __name__ == "__main__":
    main()