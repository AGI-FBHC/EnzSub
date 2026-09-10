#!/usr/bin/env python3
"""
Step 1: 蛋白质 Embedding 生成 (合并格式输出)

对每个输入 CSV 生成一个 {stem}_all.pt 文件。
只处理明确传入的 CSV, 不会扫描目录。

用法:
  python generate_embeddings.py \
      --encoder-type esm2_650m --model-mode base \
      --csv-files train.csv val.csv test.csv \
      --output-dir embeddings/esm2_base/ \
      --device cuda:0
"""

import argparse
import os
import sys
import logging
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_num_threads(4)
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

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
      - 不在当前脚本内解析 checkpoint 格式
      - 不在当前脚本内拆 backbone / LoRA
      - base / cpt / base_sub / cpt_sub 的行为全部由 EnzSubModelForDownstream 决定
    """
    from enzsub.sub.model import EnzSubModelForDownstream

    mode = args.model_mode

    if mode == "base":
        if args.checkpoint:
            log.warning("mode=base, checkpoint will be ignored.")
        checkpoint_path = None
    else:
        if not args.checkpoint:
            raise ValueError(f"mode={mode} requires --checkpoint")
        checkpoint_path = args.checkpoint

    log.info(f"[build_encoder] mode={mode}, encoder={args.encoder_type}")
    if checkpoint_path is not None:
        log.info(f"[build_encoder] checkpoint: {checkpoint_path}")

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

def generate_for_csv(
    encoder,
    csv_path: str,
    output_dir: str,
    metadata: dict,
    batch_size: int = 16,
    max_len: int = 1022,
    max_tokens_per_batch: int = 8000,
):
    """对单个 CSV 生成 {stem}_all.pt"""
    os.makedirs(output_dir, exist_ok=True)
    csv_stem = Path(csv_path).stem
    out_path = os.path.join(output_dir, f"{csv_stem}_all.pt")

    df = pd.read_csv(csv_path)
    all_ids = df['protein_id'].tolist()
    all_seqs = df['sequence'].tolist()

    # 断点续跑
    existing_embs = {}
    if os.path.exists(out_path):
        try:
            existing = torch.load(out_path, map_location="cpu")
            existing_embs = existing.get("embeddings", {})
            if len(existing_embs) >= len(all_ids):
                log.info(f"  {csv_stem}: all {len(all_ids)} done, skipping")
                return
            log.info(f"  {csv_stem}: resuming {len(existing_embs)}/{len(all_ids)}")
        except Exception:
            log.info(f"  {csv_stem}: corrupt file, regenerating")

    todo = [(pid, seq) for pid, seq in zip(all_ids, all_seqs)
            if pid not in existing_embs]
    log.info(f"  {csv_stem}: {len(todo)} to process ({len(existing_embs)} cached)")

    if not todo:
        return

    todo.sort(key=lambda x: len(x[1]))
    new_embs = {}
    i = 0

    with torch.no_grad():
        pbar = tqdm(total=len(todo), desc=f"  {csv_stem}")
        while i < len(todo):
            batch = []
            token_count = 0
            while i < len(todo) and len(batch) < batch_size:
                seq_len = min(len(todo[i][1]), max_len) + 2
                if batch and token_count + seq_len > max_tokens_per_batch:
                    break
                batch.append(todo[i])
                token_count += seq_len
                i += 1

            batch_ids = [pid for pid, _ in batch]
            batch_seqs = [seq[:max_len] for _, seq in batch]
            data = list(zip(batch_ids, batch_seqs))

            embeddings = encoder.get_embedding(data)
            embs_cpu = embeddings.cpu()
            for j, pid in enumerate(batch_ids):
                new_embs[pid] = embs_cpu[j]

            pbar.update(len(batch))
        pbar.close()

    all_embs = {**existing_embs, **new_embs}
    save_dict = {"embeddings": all_embs}
    save_dict.update(metadata)
    torch.save(save_dict, out_path)
    log.info(f"  {csv_stem}: saved {len(all_embs)} embeddings → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings (merged output format)"
    )
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
                        help="Encoder backbone type (ESM-2 full series + ESM-1b + ProtBERT-BFD)")
    parser.add_argument("--model-mode", type=str, required=True,
                        choices=list(VALID_MODES))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--csv-files", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.model_mode in CHECKPOINT_MODES and not args.checkpoint:
        parser.error(f"--checkpoint required for mode={args.model_mode}")

    encoder = build_encoder(args)
    log.info(f"Encoder ready: {args.encoder_type}, mode={args.model_mode}, "
             f"dim={encoder.hidden_dim}")

    metadata = {
        "model_mode": args.model_mode,
        "encoder_type": args.encoder_type,
    }
    if args.checkpoint:
        metadata["checkpoint_path"] = args.checkpoint

    for csv_path in args.csv_files:
        if not os.path.exists(csv_path):
            log.warning(f"Not found: {csv_path}")
            continue
        generate_for_csv(
            encoder, csv_path, args.output_dir,
            metadata, args.batch_size, args.max_len,
            args.max_tokens_per_batch,
        )

    log.info("Done")

if __name__ == "__main__":
    main()