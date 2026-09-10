#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为活性位点任务生成 ESM-2 650M 原始模型的【多层】per-residue embedding。

与现有 EnzymeDataset 完全兼容: 每条序列存一个 .pt, 内容为 {layer:int -> Tensor(L,D)}。
EnzymeDataset(repr_layer=L) 即可取出第 L 层。

关键口径 (与 sub/encoders/esm.py 对齐, 仅取层这一维度扩展):
  - 官方 fair-esm 加载 esm2_t33_650M_UR50D
  - batch_converter(truncation_seq_length=1022)
  - 一次前向 repr_layers=[多层], 共享 backbone, 省显存/时间
  - per-residue: 去掉 BOS(位置0), 取 [1 : L+1], L=min(len(seq),1022)
  - X 不替换为 <mask> (--no-x-mask 默认; 与诊断口径一致, 保留真实残基表征)

用法:
  python gen_multilayer_embeddings.py \
      --fasta ../data/Enzyme_active_sites_train.fasta \
      --output-dir results/embeddings_multilayer/esm2_650m_base/train \
      --layers 6 12 18 24 27 30 33 \
      --batch-size 8 --device cuda:0
  # test 同理换 fasta / output-dir
"""

import argparse
import os
import torch
from tqdm import tqdm

def read_fasta(path):
    ids, seqs = [], []
    cid, cseq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if cid is not None:
                    ids.append(cid); seqs.append(''.join(cseq))
                cid = line[1:].split()[0]; cseq = []
            else:
                cseq.append(line)
    if cid is not None:
        ids.append(cid); seqs.append(''.join(cseq))
    return ids, seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[6, 12, 18, 24, 27, 30, 33])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-seq-len", type=int, default=1022)
    ap.add_argument("--x-mask", action="store_true",
                    help="把 X 替换成 <mask> (默认 False, 即保留 X 的真实表征)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from esm import pretrained
    esm_name = "esm2_t33_650M_UR50D"
    # 优先用专用加载函数 (若已注册), 否则退回通用入口 (与 sub/encoders/esm.py 的 fallback 同路径)
    load_fn = getattr(pretrained, f"load_model_and_alphabet_{esm_name}", None)
    if load_fn is not None:
        model, alphabet = load_fn()
    else:
        model, alphabet = pretrained.load_model_and_alphabet(esm_name)
    model = model.to(device).eval()
    n_layers = model.num_layers  # 33
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=args.max_seq_len)

    layers = sorted(set(args.layers))
    assert max(layers) <= n_layers, f"layer {max(layers)} > num_layers {n_layers}"
    print(f"[gen] esm2_t33_650M, num_layers={n_layers}, dim={model.embed_dim}")
    print(f"[gen] extracting layers: {layers}")
    print(f"[gen] x_mask={args.x_mask} (False=保留X真实表征)")

    x_idx = alphabet.tok_to_idx.get("X", alphabet.unk_idx)
    mask_idx = alphabet.mask_idx
    pad_idx = alphabet.padding_idx
    cls_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx

    ids, seqs = read_fasta(args.fasta)
    # 断点续跑
    existing = {f[:-3] for f in os.listdir(args.output_dir) if f.endswith('.pt')}
    todo = [(i, s) for i, s in zip(ids, seqs) if i not in existing]
    if existing:
        print(f"[gen] skip {len(existing)} existing, {len(todo)} remaining")

    for start in tqdm(range(0, len(todo), args.batch_size), desc="embedding"):
        chunk = todo[start:start + args.batch_size]
        data = [(i, s) for i, s in chunk]
        _, _, tokens = batch_converter(data)
        if args.x_mask:
            tokens[tokens == x_idx] = mask_idx
        tokens = tokens.to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(tokens, repr_layers=layers, return_contacts=False)
        reps = {L: out["representations"][L] for L in layers}  # each (B, Ltok, D)

        for bi, (sid, seq) in enumerate(chunk):
            seq_len = min(len(seq.upper().replace(' ', '')), args.max_seq_len)
            # 去 BOS(0), 取 [1:seq_len+1]; 与 esm.py 完全一致
            d = {L: reps[L][bi, 1:seq_len + 1].cpu().float() for L in layers}
            torch.save(d, os.path.join(args.output_dir, f"{sid}.pt"))

    n = len([f for f in os.listdir(args.output_dir) if f.endswith('.pt')])
    print(f"[gen] done: {n} .pt files in {args.output_dir}")

if __name__ == "__main__":
    main()