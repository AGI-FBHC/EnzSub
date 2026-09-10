#!/usr/bin/env python3
"""
共享工具函数

- FASTA 解析与数据清洗
- EmbeddingStore: 按 split 名精确加载合并格式 embedding
- load_with_labels: embedding + pH 标签对齐
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# ======== FASTA 解析 ========

def parse_fasta_header(header: str) -> dict:
    """
    解析 EpHod FASTA header
    格式: >protein_id | organism | EC_number | pH_value | other_info
    """
    parts = header[1:].strip().split('|')
    result = {'protein_id': None, 'organism': None, 'EC': None, 'pH': None}
    if len(parts) >= 4:
        result['protein_id'] = parts[0].strip()
        result['organism'] = parts[1].strip()
        result['EC'] = parts[2].strip()
        try:
            result['pH'] = float(parts[3].strip())
        except ValueError:
            result['pH'] = None
    return result

def parse_fasta_to_dataframe(fasta_path: str) -> pd.DataFrame:
    """解析 EpHod FASTA 为 DataFrame"""
    records = []
    current_header, current_seq = None, []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    meta = parse_fasta_header(current_header)
                    meta['sequence'] = ''.join(current_seq)
                    records.append(meta)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            meta = parse_fasta_header(current_header)
            meta['sequence'] = ''.join(current_seq)
            records.append(meta)

    return pd.DataFrame(records)

def filter_ph_dataframe(
    df: pd.DataFrame,
    ph_min: float = 2.0,
    ph_max: float = 12.0,
    max_seq_length: int = 4000,
) -> Tuple[pd.DataFrame, dict]:
    """过滤 pH DataFrame, 返回 (filtered_df, stats)"""
    initial = len(df)
    stats = {'initial': initial}

    df = df.dropna(subset=['pH'])
    stats['removed_missing_pH'] = initial - len(df)

    before = len(df)
    df = df[(df['pH'] >= ph_min) & (df['pH'] <= ph_max)]
    stats['removed_pH_range'] = before - len(df)

    before = len(df)
    valid_mask = []
    for seq in df['sequence']:
        seq_upper = seq.upper()
        invalid_aa = set(seq_upper) - VALID_AMINO_ACIDS
        is_valid = 0 < len(seq_upper) <= max_seq_length and not invalid_aa
        valid_mask.append(is_valid)
    df = df[valid_mask].reset_index(drop=True)
    stats['removed_invalid_seq'] = before - len(df)
    stats['final'] = len(df)

    return df, stats

# ======== Embedding 存储 (合并格式, 按 split 精确加载) ========

class EmbeddingStore:
    """
    合并格式 embedding 加载器。

    目录结构:
        emb_dir/
            train_all.pt   → {"embeddings": {protein_id: Tensor(D,), ...}, ...}
            val_all.pt
            test_all.pt

    重要: 只加载明确请求的 split 文件, 不会扫描目录下所有 *_all.pt。
    这避免了旧文件 (如 phopt_training_all.pt) 干扰数据加载。
    """

    def __init__(self, emb_dir: str):
        self.emb_dir = emb_dir
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}  # split -> {id: tensor}

    def _load_split(self, split: str) -> Dict[str, torch.Tensor]:
        """加载单个 split 的 embedding 文件, 带缓存"""
        if split in self._cache:
            return self._cache[split]

        pt_path = os.path.join(self.emb_dir, f"{split}_all.pt")
        if not os.path.exists(pt_path):
            log.warning(f"Embedding file not found: {pt_path}")
            return {}

        data = torch.load(pt_path, map_location="cpu")
        embs = data.get("embeddings", {})
        self._cache[split] = embs
        log.info(f"  Loaded {len(embs)} embeddings from {split}_all.pt")
        return embs

    def load_with_labels(
        self,
        csv_path: str,
        normalize: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """
        加载 embedding 并与 CSV 标签对齐。

        根据 CSV 文件名自动推断 split:
            train.csv → train_all.pt
            val.csv   → val_all.pt
            test.csv  → test_all.pt

        Args:
            csv_path: CSV 路径 (必须有 protein_id, pH 列)
            normalize: 是否 L2 归一化

        Returns:
            (X, y, protein_ids) 或 (None, None, []) 如果无数据
        """
        # 从 CSV 文件名推断 split
        csv_stem = Path(csv_path).stem  # "train" / "val" / "test"
        embs = self._load_split(csv_stem)

        if not embs:
            log.warning(f"No embeddings for split '{csv_stem}'")
            return None, None, []

        # 读取 CSV 并对齐
        df = pd.read_csv(csv_path)
        embeddings, phs, ids = [], [], []

        for _, row in df.iterrows():
            pid = row['protein_id']
            if pid in embs:
                t = embs[pid]
                if t.dim() > 1:
                    t = t.mean(dim=0)
                embeddings.append(t.float())
                phs.append(row['pH'])
                ids.append(pid)

        if not embeddings:
            log.warning(f"No matching embeddings for {csv_path}")
            return None, None, []

        X = torch.stack(embeddings)
        if normalize:
            X = F.normalize(X, dim=1)
        X = X.numpy()
        y = np.array(phs)

        missing = len(df) - len(ids)
        if missing > 0:
            log.warning(f"Missing embeddings for {missing}/{len(df)} proteins")

        log.info(f"  {csv_stem}: {len(ids)} samples, dim={X.shape[1]}, "
                 f"pH=[{y.min():.1f}, {y.max():.1f}]")

        return X, y, ids