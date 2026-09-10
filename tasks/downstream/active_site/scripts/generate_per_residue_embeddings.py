#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active Site Per-Residue Embedding 生成脚本 (统一版)

四格消融矩阵:
  - base:     原始预训练 backbone
  - cpt:      CPT backbone (不加载 LoRA)
  - base_sub: 原始 backbone + SUB LoRA
  - cpt_sub:  CPT backbone + SUB LoRA

支持 encoder: ESM-2 650M / ESM-1b / ProtBERT-BFD

输出格式 (与现有 EnzymeDataset 完全兼容):
  每个序列一个 .pt:  {repr_layer: Tensor(L, D)}
  D = 1280 (ESM-2/ESM-1b), 1024 (ProtBERT)

[LoRA 自动检测]
  对 sub 模式 (base_sub / cpt_sub), LoRA 结构 (rank/alpha/target_modules)
  按以下优先级解析:
    1. checkpoint 内的 'lora_config' (新版 save_checkpoint 自动写入) → 全自动, 含 alpha
    2. 从 'lora_state_dict' 权重反推 rank 与 target_modules (alpha 无法反推)
    3. 兜底默认值: rank=16, target_modules=[q/k/v/out_proj] (仅当 lora_state 为空时)
  alpha 永远无法从权重反推, 反推路径下回退命令行 --lora-alpha (默认 32)。

用法:
  # ProtBERT base
  python generate_per_residue_embeddings.py \\
      --encoder-type protbert_bfd --model-mode base \\
      --fasta data/train.fasta --output-dir embeddings/protbert/base/train

  # ESM-2 CPT
  python generate_per_residue_embeddings.py \\
      --encoder-type esm2_650m --model-mode cpt \\
      --checkpoint /path/to/cpt_checkpoint.pt \\
      --fasta data/train.fasta --output-dir embeddings/esm2/cpt/train

  # ProtBERT CPT+SUB
  python generate_per_residue_embeddings.py \\
      --encoder-type protbert_bfd --model-mode cpt_sub \\
      --checkpoint /path/to/sub_checkpoint.pt \\
      --fasta data/train.fasta --output-dir embeddings/protbert/cpt_sub/train
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
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

# ======== Encoder 构建 (与 EC 脚本逻辑一致, 含 LoRA 自动检测) ========
def build_encoder(args):
    """
    构建 EnzSubModelForDownstream。

    新逻辑:
      - model_mode 是最高优先级
      - checkpoint_path 直接传原始 checkpoint 路径
      - 不在活性位点脚本内解析 checkpoint 格式
      - 不在活性位点脚本内拆 backbone / LoRA
      - base/cpt/base_sub/cpt_sub 的行为由 EnzSubModelForDownstream 统一决定
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

    print(f"[build_encoder] mode={mode} ({MODE_DESC[mode]})")
    if checkpoint_path is not None:
        print(f"[build_encoder] checkpoint: {checkpoint_path}")

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

# ======== Embedding 生成 ========

def generate_embeddings(
    encoder,
    fasta_file: str,
    output_dir: str,
    batch_size: int = 16,
):
    """
    对单个 FASTA 文件生成 per-residue embedding

    输出: 每条序列一个 .pt, 格式 {repr_layer: Tensor(L, D)}
    与现有 EnzymeDataset 完全兼容
    """
    os.makedirs(output_dir, exist_ok=True)
    seq_ids, sequences = read_fasta(fasta_file)
    repr_layer = encoder.repr_layer

    print(f"\n  FASTA: {fasta_file}")
    print(f"  Total sequences: {len(seq_ids)}")
    print(f"  repr_layer: {repr_layer}")

    # 断点续跑: 跳过已有的
    existing = set()
    for f_name in os.listdir(output_dir):
        if f_name.endswith('.pt'):
            existing.add(f_name[:-3])

    todo_indices = [i for i, sid in enumerate(seq_ids) if sid not in existing]

    if len(existing) > 0:
        print(f"  Skipping {len(existing)} already generated, "
              f"{len(todo_indices)} remaining")

    if not todo_indices:
        print(f"  All embeddings already exist, skipping")
        return

    # 分 batch 处理
    for batch_start in tqdm(range(0, len(todo_indices), batch_size),
                            desc="Generating per-residue embeddings"):
        batch_idx = todo_indices[batch_start:batch_start + batch_size]
        batch_ids = [seq_ids[i] for i in batch_idx]
        batch_seqs = [sequences[i] for i in batch_idx]

        # EnzSubModelForDownstream.get_per_residue_embedding
        # 输入: [(id, seq), ...]  输出: list of Tensor(L_i, D)
        sequences_input = list(zip(batch_ids, batch_seqs))
        per_residue = encoder.get_per_residue_embedding(sequences_input)

        for j, sid in enumerate(batch_ids):
            out_path = os.path.join(output_dir, f"{sid}.pt")
            torch.save({repr_layer: per_residue[j]}, out_path)

    total_saved = len([f for f in os.listdir(output_dir) if f.endswith('.pt')])
    print(f"  Done: {total_saved} embeddings in {output_dir}")

# ======== CLI ========

def main():
    parser = argparse.ArgumentParser(
        description="Active Site Per-Residue Embedding (ESM-2/ESM-1b/ProtBERT × 4-mode)"
    )
    parser.add_argument("--encoder-type", type=str, required=True,
                        help="Backbone encoder type (e.g. esm2_650m, esm2_t33_650M, "
                             "esm2_8m, esm1b, protbert_bfd; see sub/encoders/esm.py "
                             "and model.py::ENCODER_TYPE_CONFIG for full list)")
    parser.add_argument("--model-mode", type=str, required=True,
                        choices=list(VALID_MODES))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fasta", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="仅当 checkpoint 无 lora_config 且无法反推时的兜底 rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="alpha 无法从权重反推; 无 lora_config 时回退此值 "
                             "(务必与训练配置一致, 否则 scaling 错误)")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = build_encoder(args)
    print(f"Encoder ready: {args.encoder_type}, mode={args.model_mode}, "
          f"dim={encoder.hidden_dim}, repr_layer={encoder.repr_layer}")

    for fasta in args.fasta:
        if not os.path.exists(fasta):
            print(f"WARNING: not found, skip: {fasta}")
            continue
        generate_embeddings(encoder, fasta, args.output_dir, args.batch_size)

    print(f"\n✅ ALL DONE")

if __name__ == "__main__":
    main()