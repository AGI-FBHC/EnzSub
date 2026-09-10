#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1_generate_embeddings.py
ESP Enzyme-Substrate Binding 嵌入生成脚本

使用统一的 EnzSubModelForDownstream 接口，支持:
  - 多 encoder: ESM-2 / ESM-1b / ProtBERT-BFD
  - 四格消融矩阵: base / cpt / base_sub / cpt_sub

用法:
  # ESM-2 base (原始预训练)
  python 1_generate_embeddings.py \
      --encoder-type esm2_650m --model-mode base \
      --input-pkls train.pkl test.pkl \
      --output-dir embeddings/esm2_base/ \
      --device cuda:0

  # ESM-2 cpt_sub (CPT backbone + SUB LoRA)
  python 1_generate_embeddings.py \
      --encoder-type esm2_650m --model-mode cpt_sub \
      --checkpoint /path/to/cpt_sub_checkpoint.pt \
      --input-pkls train.pkl test.pkl \
      --output-dir embeddings/esm2_cpt_sub/ \
      --device cuda:0

  # ProtBERT base
  python 1_generate_embeddings.py \
      --encoder-type protbert_bfd --model-mode base \
      --input-pkls train.pkl test.pkl \
      --output-dir embeddings/protbert_base/ \
      --device cuda:0

[加速说明 - 长度分桶]
    唯一序列在送入模型前已按「截断后长度」升序排序, 长度接近的序列落入同一 batch,
    动态 padding 时浪费的计算最少。由于 ESM / ProtBERT encoder 的池化是 masked mean
    (padding token 已被 mask 排除), 分桶不改变任何序列的 embedding 数值
    (除 bf16/autocast 级别的微小浮点漂移外), 与随机分 batch 在功能上等价。
"""

import os
import sys
import argparse
import logging

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 确保 sub 包可导入
sys.path.append("src")
from enzsub.sub.model import EnzSubModelForDownstream

# ======== 四格消融矩阵 ========
VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")
CHECKPOINT_MODES = ("cpt", "base_sub", "cpt_sub")

MODE_DESC = {
    "base":     "original pretrained backbone",
    "cpt":      "CPT backbone, no LoRA",
    "base_sub": "original backbone + SUB LoRA",
    "cpt_sub":  "CPT backbone + SUB LoRA",
}

# ======== Encoder 构建 ========

def build_encoder(args):
    """
    构建 EnzSubModelForDownstream。

    新逻辑:
      - model_mode 是最高优先级
      - checkpoint_path 直接传原始 checkpoint 路径
      - 不在 ESP 脚本内解析 checkpoint 格式
      - 不在 ESP 脚本内拆 backbone / LoRA
      - base/cpt/base_sub/cpt_sub 的行为由 EnzSubModelForDownstream 统一决定
    """
    mode = args.model_mode

    if mode == "base":
        if args.checkpoint:
            logging.warning("mode=base, checkpoint will be ignored.")
        checkpoint_path = None
    else:
        if not args.checkpoint:
            raise ValueError(f"mode={mode} requires --checkpoint")
        checkpoint_path = args.checkpoint

    logging.info(f"[build_encoder] mode={mode} ({MODE_DESC[mode]}), encoder={args.encoder_type}")
    if checkpoint_path is not None:
        logging.info(f"[build_encoder] checkpoint: {checkpoint_path}")

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

def generate_embeddings_for_sequences(
    encoder,
    sequences: list,
    batch_size: int = 16,
    max_len: int = 1022,
) -> dict:
    """
    为序列列表生成嵌入

    Args:
        encoder: EnzSubModelForDownstream 实例
        sequences: 序列列表
        batch_size: 批次大小
        max_len: 最大序列长度

    Returns:
        dict: {sequence: embedding_numpy}

    Note:
        唯一序列按「截断后长度」升序排序后分 batch (长度分桶), 减少动态 padding 浪费。
        返回 dict 以原始序列字符串为 key, 与处理顺序无关。
    """
    # 用 encoder 自身的 max_seq_len 兜底 (ProtBERT=510, ESM=1022),
    # 避免外部传入的 max_len 超出 backbone 可接受范围。
    effective_max_len = min(max_len, encoder.max_seq_len)

    # 去重 + 过滤空序列
    unique_sequences = list(
        set(s for s in sequences if isinstance(s, str) and len(s) > 0)
    )
    # 按「截断后长度」排序: 截断后才是模型实际看到的长度, 分桶更准。
    # 保留 (原始序列, 截断序列) 配对, 截断序列送模型, 原始序列做 dict key。
    seq_pairs = [(s, s[:effective_max_len]) for s in unique_sequences]
    seq_pairs.sort(key=lambda p: len(p[1]))

    logging.info(f"共 {len(seq_pairs)} 条唯一序列需要处理 "
                 f"(截断后排序分桶, effective_max_len={effective_max_len})")

    all_embeddings = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(seq_pairs), batch_size), desc="Generating embeddings"):
            batch_pairs = seq_pairs[i:i + batch_size]
            batch_orig = [p[0] for p in batch_pairs]
            batch_trunc = [p[1] for p in batch_pairs]

            # 构造输入: [(id, truncated_seq), ...]
            data = [(str(j), seq) for j, seq in enumerate(batch_trunc)]

            # get_embedding → [B, D]
            embeddings = encoder.get_embedding(data)
            embs_cpu = embeddings.cpu().numpy()

            for j, original_seq in enumerate(batch_orig):
                all_embeddings[original_seq] = embs_cpu[j]

    return all_embeddings

def process_pkl_file(
    encoder,
    pkl_path: str,
    output_dir: str,
    batch_size: int = 16,
    max_len: int = 1022,
):
    """处理单个 pkl 文件，生成嵌入并保存"""
    output_path = os.path.join(output_dir, os.path.basename(pkl_path))
    if os.path.exists(output_path):
        try:
            df_check = pd.read_pickle(output_path)
            has_enz = "enzyme_vector" in df_check.columns and df_check["enzyme_vector"].notna().sum() > 0
            has_chem = "ChemBERTa_vector" in df_check.columns and df_check["ChemBERTa_vector"].notna().sum() > 0
            if has_enz and has_chem:
                logging.info(f"--- 跳过已存在 (enzyme_vector + ChemBERTa_vector 完整): {output_path} ---")
                return output_path
            else:
                missing = []
                if not has_enz: missing.append("enzyme_vector")
                if not has_chem: missing.append("ChemBERTa_vector")
        except Exception:
            logging.info(f"--- 已有文件损坏，重新生成: {output_path} ---")

    logging.info(f"--- 正在处理文件: {os.path.basename(pkl_path)} ---")

    df = pd.read_pickle(pkl_path)
    logging.info(f"加载了 {len(df)} 条记录")

    sequences_to_process = df['sequence'].dropna().unique().tolist()
    logging.info(f"找到 {len(sequences_to_process)} 条不重复的序列")

    embedding_map = generate_embeddings_for_sequences(
        encoder, sequences_to_process, batch_size, max_len,
    )

    df['enzyme_vector'] = df['sequence'].map(embedding_map)

    missing_count = df['enzyme_vector'].isnull().sum()
    if missing_count > 0:
        logging.warning(f"有 {missing_count} 行未能生成嵌入")

    output_path = os.path.join(output_dir, os.path.basename(pkl_path))
    df.to_pickle(output_path)
    logging.info(f"已保存到: {output_path}")

    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="ESP embedding generation (ESM-2/ESM-1b/ProtBERT, 4-mode ablation)"
    )

    # 新接口参数
    parser.add_argument("--encoder-type", type=str, default="esm2_650m",
                        choices=[
                            "esm1b",
                            "esm2_8m", "esm2_t6_8M",
                            "esm2_35m", "esm2_t12_35M",
                            "esm2_150m", "esm2_t30_150M",
                            "esm2_650m", "esm2_t33_650M",
                            "esm2_3b", "esm2_t36_3B",
                            "esm2_15b", "esm2_t48_15B",
                            "protbert_bfd",
                        ],
                        help="Encoder backbone type")
    parser.add_argument("--model-mode", type=str, required=True,
                        choices=list(VALID_MODES),
                        help="Ablation mode: base / cpt / base_sub / cpt_sub")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (required for cpt/base_sub/cpt_sub)")

    # LoRA 参数
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # 数据参数
    parser.add_argument("--input-pkls", type=str, nargs='+', required=True,
                        help="输入 pkl 文件路径列表")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")

    # 推理参数
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1022,
                        help="序列截断长度; 实际会与 encoder.max_seq_len 取 min")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 校验: cpt/base_sub/cpt_sub 必须有 checkpoint
    if args.model_mode in CHECKPOINT_MODES and not args.checkpoint:
        parser.error(f"--checkpoint is required for model-mode={args.model_mode}")

    logging.info(f"Encoder: {args.encoder_type}")
    logging.info(f"Mode: {args.model_mode} ({MODE_DESC[args.model_mode]})")
    logging.info(f"Device: {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 构建 encoder
    encoder = build_encoder(args)
    logging.info(f"Encoder ready, hidden_dim={encoder.hidden_dim}")

    # 处理每个 pkl
    for pkl_path in args.input_pkls:
        if not os.path.exists(pkl_path):
            logging.warning(f"文件不存在，跳过: {pkl_path}")
            continue

        process_pkl_file(
            encoder=encoder,
            pkl_path=pkl_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_len=args.max_len,
        )

    logging.info("✅ 所有文件处理完成！")

if __name__ == "__main__":
    main()