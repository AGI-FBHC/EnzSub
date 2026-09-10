#!/usr/bin/env python3
"""
1b_generate_chemberta_embeddings.py — 为 ESP pkl 生成 ChemBERTa 底物 embedding

使用与 SUB 阶段完全相同的 ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)，
冻结状态，CLS token 作为 768d 底物表征。

这个脚本独立于蛋白质 encoder，只处理底物侧。
每个 pkl 只需要跑一次（ChemBERTa 对所有模型都一样）。

如果输出 pkl 已存在且包含 ChemBERTa_vector 列，则自动跳过。

Usage:
    python 1b_generate_chemberta_embeddings.py \
        --input-pkls train.pkl test.pkl \
        --output-dir chemberta_embeddings/ \
        --batch-size 64 \
        --device cuda:0
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ChemBERTa model name — same as SUB stage (EnzSubConfig.substrate_encoder_name)
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

def load_chemberta(device: str):
    """Load frozen ChemBERTa, return (model, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    logging.info(f"Loading ChemBERTa: {CHEMBERTA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    logging.info(f"ChemBERTa loaded (frozen, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer

def generate_chemberta_embeddings(
    model, tokenizer, smiles_list, batch_size=64, max_length=128, device="cpu"
):
    """
    Generate 768d CLS-token embeddings for a list of SMILES.
    Returns dict: {smiles: np.ndarray(768,)}
    """
    unique_smiles = sorted(set(s for s in smiles_list if isinstance(s, str) and len(s) > 0))
    logging.info(f"  {len(unique_smiles)} unique SMILES to encode")

    embeddings = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(unique_smiles), batch_size), desc="  ChemBERTa"):
            batch = unique_smiles[i:i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS token (same as SubstrateEncoder.forward)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            for j, smi in enumerate(batch):
                embeddings[smi] = cls_emb[j].astype(np.float32)

    return embeddings

def process_pkl(model, tokenizer, pkl_path, output_dir, batch_size, device):
    """Process one pkl: add ChemBERTa_vector column, save to output_dir."""
    output_path = os.path.join(output_dir, os.path.basename(pkl_path))

    # 跳过已有 ChemBERTa_vector 的文件
    if os.path.exists(output_path):
        try:
            df_check = pd.read_pickle(output_path)
            if "ChemBERTa_vector" in df_check.columns and df_check["ChemBERTa_vector"].notna().sum() > 0:
                logging.info(f"--- 跳过已存在: {output_path} (已有 ChemBERTa_vector) ---")
                return
        except Exception:
            pass  # 文件损坏则重新生成

    logging.info(f"\n--- Processing: {os.path.basename(pkl_path)} ---")
    df = pd.read_pickle(pkl_path)
    logging.info(f"  Loaded {len(df)} rows")

    if "SMILES" not in df.columns:
        logging.error(f"  No SMILES column! Skipping.")
        return

    smiles_list = df["SMILES"].dropna().tolist()
    emb_map = generate_chemberta_embeddings(
        model, tokenizer, smiles_list,
        batch_size=batch_size, device=device,
    )

    df["ChemBERTa_vector"] = df["SMILES"].map(emb_map)

    n_missing = df["ChemBERTa_vector"].isna().sum()
    if n_missing > 0:
        logging.warning(f"  {n_missing} rows missing ChemBERTa embedding")

    # Verify dimension
    sample = df["ChemBERTa_vector"].dropna().iloc[0]
    logging.info(f"  ChemBERTa dim: {len(sample)}")

    df.to_pickle(output_path)
    logging.info(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate ChemBERTa 768d substrate embeddings for ESP pkls"
    )
    parser.add_argument("--input-pkls", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, tokenizer = load_chemberta(device)

    for pkl_path in args.input_pkls:
        if not os.path.exists(pkl_path):
            logging.warning(f"File not found: {pkl_path}")
            continue
        process_pkl(model, tokenizer, pkl_path, args.output_dir,
                    args.batch_size, device)

    logging.info("\n✅ All done!")

if __name__ == "__main__":
    main()