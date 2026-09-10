#!/usr/bin/env python3
"""
共享工具函数: FASTA 读写、序列预处理、EC 标签加载
"""

import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# 构建 str.translate 映射表 (比逐字符拼接快 ~5x)
_TRANS_TABLE = str.maketrans(
    {ch: "X" for ch in set(map(chr, range(256))) - STANDARD_AAS - {" "}}
)

def preprocess_sequence(seq: str, max_len: int = 1022) -> str:
    """标准化氨基酸序列 → 空格分隔的 ProtBERT 输入格式"""
    seq = seq.upper()[:max_len]
    seq = seq.translate(_TRANS_TABLE)
    return " ".join(seq)

def read_fasta(fasta_path: str) -> Tuple[List[str], List[str]]:
    """读取 FASTA 文件, 返回 (seq_ids, sequences)"""
    seq_ids, sequences = [], []
    current_id, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seq_ids.append(current_id)
                    sequences.append("".join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seq_ids.append(current_id)
            sequences.append("".join(current_seq))

    return seq_ids, sequences

def load_ec_labels(csv_path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    加载 EC 标签映射 (兼容 TSV / CSV)
    """
    if not os.path.exists(csv_path):
        return {}, []

    df = pd.read_csv(csv_path, sep="\t", dtype=str, na_filter=False, engine="c")

    if len(df.columns) < 2:
        df = pd.read_csv(csv_path, sep=",", dtype=str, na_filter=False, engine="c")

    id_col = None
    for candidate in ["Entry", "entry", "id", "ID", "seq_id", "uniprot_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = df.columns[0]

    ec_col = None
    for candidate in ["EC number", "ec_number", "EC", "ec", "label"]:
        if candidate in df.columns:
            ec_col = candidate
            break
    if ec_col is None:
        ec_col = df.columns[1]

    ecs_series = df[ec_col].str.split(";").apply(
        lambda lst: [t.strip() for t in lst if t.strip()]
    )
    mask = ecs_series.map(len) > 0
    df = df.loc[mask].reset_index(drop=True)
    ecs_series = ecs_series.loc[mask].reset_index(drop=True)

    id_to_ecs = dict(zip(df[id_col].tolist(), ecs_series.tolist()))
    return id_to_ecs, list(id_to_ecs.keys())

def filter_ids_with_embeddings(ids: List[str], emb_dir: str) -> List[str]:
    """过滤出在 emb_dir 中有对应 .pt 文件的 ID"""
    existing = {f[:-3] for f in os.listdir(emb_dir) if f.endswith(".pt")}
    valid = [sid for sid in ids if sid in existing]
    return valid

def build_label_matrix(
    ids: List[str],
    id_to_ecs: Dict[str, List[str]],
    ec_to_idx: Dict[str, int],
) -> np.ndarray:
    """构建多标签矩阵 (n_samples, n_classes)"""
    n = len(ids)
    n_classes = len(ec_to_idx)
    mat = np.zeros((n, n_classes), dtype=np.float32)
    for i, sid in enumerate(ids):
        for ec in id_to_ecs.get(sid, []):
            if ec in ec_to_idx:
                mat[i, ec_to_idx[ec]] = 1.0
    return mat