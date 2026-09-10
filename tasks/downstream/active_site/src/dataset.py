#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset for enzyme active site prediction (GraphEC style)

支持 ESM-2 (repr_layer=33, dim=1280) 和 ProtBERT-BFD (repr_layer=30, dim=1024)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import ast

class EnzymeDataset(Dataset):
    """
    Dataset for enzyme active site prediction.

    Features:
    - Loads precomputed ESM/ProtBERT embeddings
    - Pads/truncates to max_seq_length
    - Returns (embeddings, labels, mask) where mask indicates valid residues
    - 自动检测 embedding 维度 (1280 for ESM-2, 1024 for ProtBERT)
    """

    def __init__(
        self,
        data_file: str,
        esm_data_pt_path: str,
        max_seq_length: int = 1022,
        repr_layer: int = 33,
        verbose_skip: bool = True,
        max_skip_examples: int = 5,
    ):
        """
        Args:
            ...
            verbose_skip: 是否打印跳过样本的详细原因
            max_skip_examples: 每类跳过原因最多打印多少个具体样例
        """
        self.data_file = data_file
        self.esm_data_pt_path = esm_data_pt_path
        self.max_seq_length = max_seq_length
        self.repr_layer = repr_layer
        self.verbose_skip = verbose_skip
        self.max_skip_examples = max_skip_examples
        self.embed_dim = None
        self.samples = self._load_data()

        print(f"✓ Loaded {len(self.samples)} samples from {data_file}")
        print(f"  repr_layer={repr_layer}, embed_dim={self.embed_dim}, max_seq_length={max_seq_length}")

    def _load_data(self):
        """Load and preprocess all samples"""
        samples = []
        # 按原因分类记录，存 (line_num, protein_id, detail)
        skip_reasons = {
            'malformed_line':        [],  # 行字段不足3个
            'invalid_label_format':  [],  # 标签不是 [...] 格式
            'label_parse_error':     [],  # ast.literal_eval 失败
            'embedding_not_found':   [],  # .pt 文件不存在
            'repr_layer_missing':    [],  # .pt 里没有指定 repr_layer
        }

        with open(self.data_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split(',', 2)
                if len(parts) < 3:
                    skip_reasons['malformed_line'].append(
                        (line_num, '<unknown>', f"only {len(parts)} field(s): {line.strip()[:60]!r}")
                    )
                    continue

                protein_id = parts[0].strip()

                # Parse labels
                label_str = parts[2].rstrip(',').strip()
                if not (label_str.startswith('[') and label_str.endswith(']')):
                    skip_reasons['invalid_label_format'].append(
                        (line_num, protein_id, f"label_str starts/ends wrong: {label_str[:60]!r}")
                    )
                    continue

                try:
                    labels = np.array(ast.literal_eval(label_str), dtype=np.float32)
                except (ValueError, SyntaxError) as e:
                    skip_reasons['label_parse_error'].append(
                        (line_num, protein_id, f"{type(e).__name__}: {str(e)[:80]}")
                    )
                    continue

                # Load embedding
                emb_path = os.path.join(self.esm_data_pt_path, f"{protein_id}.pt")
                if not os.path.exists(emb_path):
                    skip_reasons['embedding_not_found'].append(
                        (line_num, protein_id, emb_path)
                    )
                    continue

                emb_dict = torch.load(emb_path, map_location='cpu')

                # Support both nested and flat dict formats
                if 'mean_representations' in emb_dict and isinstance(emb_dict['mean_representations'], dict):
                    token_embeddings = emb_dict['mean_representations'].get(self.repr_layer)
                    available_keys = list(emb_dict['mean_representations'].keys())
                    fmt = 'nested(mean_representations)'
                else:
                    token_embeddings = emb_dict.get(self.repr_layer)
                    available_keys = [k for k in emb_dict.keys() if isinstance(k, int)]
                    fmt = 'flat'

                if token_embeddings is None:
                    skip_reasons['repr_layer_missing'].append(
                        (line_num, protein_id,
                        f"need layer={self.repr_layer}, format={fmt}, available={available_keys}")
                    )
                    continue

                # 自动检测 embedding 维度 (首次)
                if self.embed_dim is None:
                    self.embed_dim = token_embeddings.shape[-1]

                # Pad/truncate
                L, D = token_embeddings.shape
                length = min(L, self.max_seq_length)

                padded_x = torch.zeros((self.max_seq_length, D), dtype=torch.float32)
                padded_y = torch.zeros(self.max_seq_length, dtype=torch.float32)
                padded_mask = torch.zeros(self.max_seq_length, dtype=torch.float32)

                padded_x[:length] = token_embeddings[:length]
                label_len = min(len(labels), length)
                padded_y[:label_len] = torch.from_numpy(labels[:label_len])
                padded_mask[:length] = 1.0

                samples.append((padded_x, padded_y, padded_mask))

        # ---- 打印跳过报告 ----
        total_skipped = sum(len(v) for v in skip_reasons.values())
        if total_skipped > 0:
            print(f"  ⚠️  Skipped {total_skipped} samples. Breakdown:")
            for reason, entries in skip_reasons.items():
                if not entries:
                    continue
                print(f"    [{reason}] {len(entries)} samples")
                if self.verbose_skip:
                    for line_num, pid, detail in entries[:self.max_skip_examples]:
                        print(f"      - line {line_num} | id={pid} | {detail}")
                    if len(entries) > self.max_skip_examples:
                        print(f"      ... and {len(entries) - self.max_skip_examples} more")

        if self.embed_dim is None:
            print(f"  ⚠️  No samples loaded — embed_dim fallback to 1280 (likely wrong, check skip reasons above)")
            self.embed_dim = 1280

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            x: (max_seq_length, embed_dim) embeddings
            y: (max_seq_length,) binary labels
            mask: (max_seq_length,) 1 for valid positions, 0 for padding
        """
        return self.samples[idx]

    def get_sample_label(self, idx):
        """
        Get binary sample-level label (for stratified split)
        Returns 1 if protein has any active site, 0 otherwise
        """
        _, y, mask = self.samples[idx]
        valid_labels = y[mask.bool()]
        return int((valid_labels > 0).any().item())

def get_dataset_labels(dataset):
    """
    Extract sample-level labels for stratified K-fold

    Args:
        dataset: EnzymeDataset

    Returns:
        labels: np.array of shape (n_samples,)
    """
    labels = []
    for i in range(len(dataset)):
        label = dataset.get_sample_label(i)
        labels.append(label)
    return np.array(labels)