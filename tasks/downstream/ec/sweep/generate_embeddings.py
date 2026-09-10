#!/usr/bin/env python3
"""
EC 分类专用 Embedding 导出脚本

四格消融矩阵:
  - base:     原始预训练 backbone
  - cpt:      CPT backbone (不加载 LoRA)
  - base_sub: 原始 backbone + SUB LoRA
  - cpt_sub:  CPT backbone + SUB LoRA

支持 encoder: ESM-2 / ESM-1b / ProtBERT-BFD

输出格式: {output_dir}/{fasta_stem}_all.pt → {
    "embeddings": {seq_id: Tensor(D,), ...},
    "model_mode": str,
    "encoder_type": str,
}
  D = 1280 (ESM-2), 1280 (ESM-1b), 1024 (ProtBERT)

兼容模式: 同时生成 embeddings.index.json 供 knn_eval 快速查找

[加速说明 - 长度分桶]
    embedding 生成采用「按序列长度排序后分 batch」策略:
    长度接近的序列落入同一 batch, 动态 padding 时浪费的计算最少。
    由于 ESM / ProtBERT encoder 的池化是 masked mean (padding token 已被 mask 排除),
    分桶不会改变任何序列的 embedding 数值 (除 bf16/autocast 级别的微小浮点漂移外),
    与随机分 batch 在功能上等价, 仅顺序不同。
    断点续跑按 seq_id 存储, 与处理顺序无关, 不受分桶影响。
"""

import argparse
import os
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
SUB_MODES = ("base_sub", "cpt_sub")
CHECKPOINT_MODES = ("cpt", "base_sub", "cpt_sub")

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

def preprocess_sequence(seq: str, max_len: int = 1022) -> str:
    seq = seq.upper().replace(' ', '')
    return seq[:max_len]

# ======== Encoder 构建 ========
def build_encoder(args):
    from enzsub.sub.model import EnzSubModelForDownstream

    mode = args.model_mode

    if mode == "base":
        checkpoint_path = None
    else:
        if not args.checkpoint:
            raise ValueError(f"mode={mode} requires --checkpoint")
        checkpoint_path = args.checkpoint

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

def _make_length_sorted_batches(todo_ids, todo_seqs, batch_size, max_len):
    """
    按预处理后的真实长度升序排序, 再切固定大小的 batch。
    长度接近的序列落入同一 batch → 动态 padding 时浪费的计算最少。

    Args:
        todo_ids:  序列 id 列表
        todo_seqs: 原始序列列表 (未预处理)
        batch_size: 每个 batch 的固定条数
        max_len:    序列截断长度 (排序前先预处理, 排序长度=模型实际看到的长度)

    Returns:
        List[List[(seq_id, processed_seq)]]
    """
    processed = [
        (sid, preprocess_sequence(seq, max_len))
        for sid, seq in zip(todo_ids, todo_seqs)
    ]
    # 按截断后长度升序; Python sort 稳定, 等长序列保持原相对顺序
    processed.sort(key=lambda x: len(x[1]))
    return [processed[i:i + batch_size]
            for i in range(0, len(processed), batch_size)]

def _make_token_budget_batches(todo_ids, todo_seqs, token_budget, max_len,
                               max_batch_size=64):
    """
    [可选, 进阶] token 预算分桶: 先按长度排序, 再贪心打包。
    每个 batch 满足:  len(batch) * (batch 内最长序列长度) <= token_budget
    短序列自动多装, 长序列自动缩小 batch, GPU 利用率更均匀。

    Args:
        token_budget: 每 batch 的 token 上限 (条数 × batch 内最长长度)。
                      保守起步值 ≈ 当前 batch_size × max_len。
        max_batch_size: 单 batch 条数硬上限, 防止短序列堆出超大 batch。

    Returns:
        List[List[(seq_id, processed_seq)]]
    """
    processed = [
        (sid, preprocess_sequence(seq, max_len))
        for sid, seq in zip(todo_ids, todo_seqs)
    ]
    processed.sort(key=lambda x: len(x[1]))

    batches, cur = [], []
    for sid, seq in processed:
        cand_max = max([len(s) for _, s in cur] + [len(seq)])
        if cur and (len(cur) + 1) * cand_max > token_budget:
            batches.append(cur)
            cur = [(sid, seq)]
        elif len(cur) >= max_batch_size:
            batches.append(cur)
            cur = [(sid, seq)]
        else:
            cur.append((sid, seq))
    if cur:
        batches.append(cur)
    return batches

# ======== Embedding 生成 (合并输出) ========
def generate_embeddings(
    encoder,
    fasta_file: str,
    output_dir: str,
    metadata: dict,
    batch_size: int = 8,
    max_len: int = 1022,
    token_budget: int = 0,
    max_batch_size: int = 64,
):
    """对单个 FASTA 文件生成 embedding, 输出为单个 .pt 文件。

    分桶策略:
        token_budget <= 0 (默认): 固定大小分桶 (batch_size 条/batch)
        token_budget >  0:        token 预算分桶 (短序列多装)
    两种都先按长度排序, 减少动态 padding 浪费。
    """
    os.makedirs(output_dir, exist_ok=True)
    seq_ids, sequences = read_fasta(fasta_file)
    fasta_stem = Path(fasta_file).stem

    out_path = os.path.join(output_dir, f"{fasta_stem}_all.pt")

    # 检查是否已完成
    if os.path.exists(out_path):
        existing = torch.load(out_path, map_location="cpu")
        existing_embs = existing.get("embeddings", {})
        if len(existing_embs) >= len(seq_ids):
            return
        # 部分完成 → 断点续跑
    else:
        existing_embs = {}

    # 过滤已完成的 (按 seq_id, 与顺序无关 → 分桶不影响 resume 正确性)
    pairs = [(sid, seq) for sid, seq in zip(seq_ids, sequences)
             if sid not in existing_embs]

    if not pairs:
        return

    todo_ids, todo_seqs = zip(*pairs)
    new_embs = {}
    effective_max_len = min(max_len, encoder.max_seq_len)

    # 选择分桶策略
    if token_budget > 0:
        batches = _make_token_budget_batches(
            todo_ids, todo_seqs, token_budget, effective_max_len, max_batch_size
        )
    else:
        batches = _make_length_sorted_batches(
            todo_ids, todo_seqs, batch_size, effective_max_len
        )

    for batch in tqdm(batches, desc=f"Embedding {fasta_stem}"):
        batch_ids = [sid for sid, _ in batch]
        # batch 已是 [(sid, processed_seq), ...], 直接送入
        embeddings = encoder.get_embedding(batch)  # [B, D]

        # 一次性 D2H 传输，避免在循环里逐个 .cpu()
        embs_cpu = embeddings.cpu()
        for j, sid in enumerate(batch_ids):
            new_embs[sid] = embs_cpu[j]

    # 合并旧 + 新
    all_embs = {**existing_embs, **new_embs}

    save_dict = {"embeddings": all_embs}
    save_dict.update(metadata)
    torch.save(save_dict, out_path)

def main():
    parser = argparse.ArgumentParser(
        description="EC classification embedding export (ESM-2/ESM-1b/ProtBERT)"
    )
    parser.add_argument("--encoder-type", type=str, default="protbert_bfd",
                        help="Backbone encoder type (e.g. esm2_650m, esm2_t33_650M, "
                             "esm2_8m, esm1b, protbert_bfd; see sub/encoders/esm.py "
                             "for full list)")
    parser.add_argument("--model-mode", type=str, required=True,
                        choices=list(VALID_MODES))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--fasta", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="固定分桶时每个 batch 的条数 (token-budget 模式下忽略)")
    parser.add_argument("--max-len", type=int, default=1022,
                        help="序列截断长度; 实际会与 encoder.max_seq_len 取 min")
    parser.add_argument("--token-budget", type=int, default=0,
                        help=">0 时启用 token 预算分桶 (条数×batch内最长长度<=budget), "
                             "短序列自动多装; 0 表示用固定大小分桶。"
                             "保守起步值 ≈ batch_size × max_len。")
    parser.add_argument("--max-batch-size", type=int, default=64,
                        help="token-budget 模式下单 batch 条数硬上限")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder = build_encoder(args)

    metadata = {
        "model_mode": args.model_mode,
        "encoder_type": args.encoder_type,
    }
    if args.checkpoint:
        metadata["checkpoint_path"] = args.checkpoint

    for fasta in args.fasta:
        if not os.path.exists(fasta):
            continue
        metadata["source_fasta"] = fasta
        generate_embeddings(encoder, fasta, args.output_dir,
                            metadata, args.batch_size, args.max_len,
                            token_budget=args.token_budget,
                            max_batch_size=args.max_batch_size)

if __name__ == "__main__":
    main()