#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tm 预测 Mean-Pooled Embedding 生成脚本 (统一版)

四格消融矩阵:
  - base / cpt / base_sub / cpt_sub

支持的 encoder 由 sub.EnzSubModelForDownstream 决定 (常见:
  esm2_t6_8M / esm2_t12_35M / esm2_t30_150M / esm2_650m / esm2_t36_3B / esm2_t48_15B
  / esm1b / protbert_bfd
)。本脚本不再硬编码白名单, 未知 encoder_type 会由 EnzSubModelForDownstream
自身抛错。

输出格式: {emb_root}/{model_name}/{split}.pt
  每个 .pt: {seq_id: Tensor(D,)}

用法:
  python generate_tm_embeddings.py \\
      --encoder-type protbert_bfd --model-mode base \\
      --fasta train:data/train.fasta val:data/val.fasta test:data/test.fasta \\
      --output-dir embeddings/protbert_base

  python generate_tm_embeddings.py \\
      --encoder-type esm2_t30_150M --model-mode cpt \\
      --checkpoint /path/to/cpt.pt \\
      --fasta train:data/train.fasta val:data/val.fasta test:data/test.fasta \\
      --output-dir embeddings/esm2_150m_cpt

[加速说明 - 长度分桶]
    每个 split 的序列在送入模型前已按「截断后长度」升序排序, 长度接近的序列落入同一
    batch, 动态 padding 时浪费的计算最少。由于 ESM / ProtBERT encoder 的池化是
    masked mean (padding token 已被 mask 排除), 分桶不改变任何序列的 embedding 数值
    (除 bf16/autocast 级别的微小浮点漂移外), 与原始顺序分 batch 在功能上等价。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# 确保 sub 包可导入
_sub_parent = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
if _sub_parent and os.path.isdir(os.path.join(_sub_parent, "sub")):
    sys.path.insert(0, _sub_parent)
else:
    _guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(_guess, "sub")):
        sys.path.insert(0, _guess)

# ======== 四格消融矩阵 ========
VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")

MODE_DESC = {
    "base":     "original pretrained backbone",
    "cpt":      "CPT backbone, no LoRA",
    "base_sub": "original backbone + SUB LoRA",
    "cpt_sub":  "CPT backbone + SUB LoRA",
}

# ======== FASTA 读取 ========

def read_fasta(fasta_path: str):
    ids, seqs = [], []
    current_id, current_seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append(''.join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        ids.append(current_id)
        seqs.append(''.join(current_seq))
    return ids, seqs

def preprocess_sequence(seq: str, max_len: int) -> str:
    seq = seq.upper().replace(' ', '')
    return seq[:max_len]

# ======== Encoder 构建 ========
def build_encoder(args):
    """
    构建 EnzSubModelForDownstream。

    新逻辑:
      - model_mode 是最高优先级
      - checkpoint_path 直接传原始 checkpoint 路径
      - 不在 Tm 脚本内解析 checkpoint 格式
      - 不在 Tm 脚本内拆 backbone / LoRA
      - base / cpt / base_sub / cpt_sub 的行为全部由 EnzSubModelForDownstream 决定
    """
    from enzsub.sub.model import EnzSubModelForDownstream

    mode = args.model_mode

    if mode == "base":
        if args.checkpoint:
            print("[build_encoder] WARNING: mode=base, checkpoint will be ignored.")
        checkpoint_path = None
    else:
        if not args.checkpoint:
            raise ValueError(f"mode={mode} requires --checkpoint")
        checkpoint_path = args.checkpoint

    print(f"[build_encoder] encoder={args.encoder_type}  mode={mode} ({MODE_DESC[mode]})")
    if checkpoint_path is not None:
        print(f"  checkpoint: {checkpoint_path}")

    encoder = EnzSubModelForDownstream(
        encoder_type=args.encoder_type,
        model_mode=mode,
        checkpoint_path=checkpoint_path,
        freeze_backbone=True,
        device=args.device,
        strict_lora_load=True,
        allow_lora_reverse_detect=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    return encoder

# ======== 长度分桶 ========

def _make_length_sorted_batches(seq_ids, sequences, batch_size, max_len):
    """
    按预处理后的真实长度升序排序, 再切固定大小的 batch。
    长度接近的序列落入同一 batch → 动态 padding 时浪费的计算最少。

    Args:
        seq_ids:   序列 id 列表
        sequences: 原始序列列表 (未预处理)
        batch_size: 每个 batch 的固定条数
        max_len:    序列截断长度 (排序前先预处理, 排序长度=模型实际看到的长度)

    Returns:
        List[List[(seq_id, processed_seq)]]
    """
    processed = [
        (sid, preprocess_sequence(seq, max_len))
        for sid, seq in zip(seq_ids, sequences)
    ]
    processed.sort(key=lambda x: len(x[1]))   # 截断后长度 = 模型实际看到的长度
    return [processed[i:i + batch_size]
            for i in range(0, len(processed), batch_size)]

# ======== Embedding 生成 ========

def generate_split_embeddings(
    encoder,
    fasta_file: str,
    output_path: str,
    batch_size: int = 32,
    max_len: int = 1022,
):
    """
    对单个 split 的 FASTA 生成 mean-pooled embedding

    输出: {seq_id: Tensor(D,)}

    Note:
        序列按「截断后长度」排序后分 batch (长度分桶), 减少动态 padding 浪费。
        输出 dict 以 seq_id 为 key, 与处理顺序无关。
    """
    if os.path.exists(output_path):
        print(f"  ✓ Already exists, skipping: {output_path}")
        return

    seq_ids, sequences = read_fasta(fasta_file)
    print(f"  {Path(fasta_file).stem}: {len(seq_ids)} sequences")

    # 用 encoder 自身的 max_seq_len 兜底 (ProtBERT=510, ESM=1022),
    # 避免外部传入的 max_len 超出 backbone 可接受范围。
    effective_max_len = min(max_len, encoder.max_seq_len)
    batches = _make_length_sorted_batches(
        seq_ids, sequences, batch_size, effective_max_len
    )

    embeddings_dict = {}

    for batch in tqdm(batches, desc=f"  Embedding {Path(fasta_file).stem}"):
        batch_ids = [sid for sid, _ in batch]
        # batch 已是 [(sid, processed_seq), ...], 直接送入
        embs = encoder.get_embedding(batch)  # [B, D]
        embs_cpu = embs.cpu()

        for j, sid in enumerate(batch_ids):
            embeddings_dict[sid] = embs_cpu[j]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_dict, output_path)
    print(f"  Saved {len(embeddings_dict)} embeddings → {output_path}")

# ======== CLI ========

def parse_fasta_pairs(pairs: list) -> dict:
    """解析 split:path 格式的 FASTA 列表"""
    result = {}
    for pair in pairs:
        if ':' in pair:
            split_name, path = pair.split(':', 1)
            result[split_name.strip()] = path.strip()
        else:
            stem = Path(pair).stem
            result[stem] = pair
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Tm Mean-Pooled Embedding (any encoder supported by EnzSubModelForDownstream)"
    )
    # 不再用 choices=[...] 限制, encoder 合法性由 sub 包自身验证
    parser.add_argument("--encoder-type", type=str, required=True,
                        help="Encoder name accepted by sub.EnzSubModelForDownstream "
                             "(e.g. esm2_t6_8M, esm2_t12_35M, esm2_t30_150M, "
                             "esm2_650m, esm2_t36_3B, esm1b, protbert_bfd)")
    parser.add_argument("--model-mode", type=str, required=True,
                        choices=list(VALID_MODES))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fasta", type=str, nargs="+", required=True,
                        help="split:path pairs, e.g. train:data/train.fasta")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=1022,
                        help="序列截断长度; 实际会与 encoder.max_seq_len 取 min")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = build_encoder(args)
    print(f"Encoder ready: {args.encoder_type}, mode={args.model_mode}, "
          f"dim={encoder.hidden_dim}")

    fasta_map = parse_fasta_pairs(args.fasta)

    for split_name, fasta_path in fasta_map.items():
        if not os.path.exists(fasta_path):
            print(f"WARNING: not found, skip: {fasta_path}")
            continue
        output_path = os.path.join(args.output_dir, f"{split_name}.pt")
        generate_split_embeddings(encoder, fasta_path, output_path,
                                  args.batch_size, args.max_len)

    print(f"\n✅ ALL DONE")

if __name__ == "__main__":
    main()