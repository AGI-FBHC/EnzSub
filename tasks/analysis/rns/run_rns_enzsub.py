#!/usr/bin/env python3

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

def _ensure_sub_importable(explicit_parent: Optional[str]) -> None:
    candidates = []
    if explicit_parent:
        candidates.append(explicit_parent)
    env_first = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if env_first:
        candidates.append(env_first)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.dirname(here))
    candidates.append(here)
    for c in candidates:
        if c and (os.path.isdir(os.path.join(c, "sub")) or os.path.isdir(os.path.join(c, "enzsub", "sub"))):
            if c not in sys.path:
                sys.path.insert(0, c)
            return

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rns")

VALID_COMPARE_MODES = ("cpt", "base_sub", "cpt_sub")

DEFAULT_CONFIG = Path("configs/analysis/rns/esm2_650M_cpt.yaml")

def load_config() -> Dict:
    parser = argparse.ArgumentParser(description="Run Random Neighbor Score analysis.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"RNS configuration must be a mapping: {args.config}")
    return config

def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    header, chunks = None, []
    with path.open("r") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)

def parse_real_header(header: str) -> str:
    return header.split(None, 1)[0]

def parse_random_header(header: str) -> Tuple[str, str]:
    stem = header.split(None, 1)[0]
    if not stem.startswith("random_") or "|" not in stem:
        raise ValueError(f"Bad random header: {header!r}")
    tag, source_id = stem.split("|", 1)
    return f"{tag}|{source_id}", source_id

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def load_sequences(real_fasta: Path, random_fasta: Path, max_len: int):
    records: List[Dict] = []
    n_skipped_long = 0
    n_nonstandard = 0

    for header, seq in iter_fasta(real_fasta):
        seq_id = parse_real_header(header)
        L = len(seq)
        if L > max_len:
            n_skipped_long += 1
            continue
        if any(c not in STANDARD_AA for c in seq):
            n_nonstandard += 1
        records.append({"seq_id": seq_id, "is_random": 0, "source_id": seq_id,
                        "length": L, "sequence": seq})

    for header, seq in iter_fasta(random_fasta):
        seq_id, source_id = parse_random_header(header)
        L = len(seq)
        if L > max_len:
            n_skipped_long += 1
            continue
        if any(c not in STANDARD_AA for c in seq):
            n_nonstandard += 1
        records.append({"seq_id": seq_id, "is_random": 1, "source_id": source_id,
                        "length": L, "sequence": seq})

    return records, n_skipped_long, n_nonstandard

def build_encoder(encoder_type, model_mode, checkpoint_path, device,
                  lora_rank, lora_alpha):
    """model_mode in {base, cpt, base_sub, cpt_sub}."""
    from enzsub.sub.model import EnzSubModelForDownstream

    needs_ckpt = model_mode in ("cpt", "base_sub", "cpt_sub")
    if needs_ckpt and not checkpoint_path:
        raise ValueError(f"model_mode={model_mode!r} requires compare_checkpoint.")

    log.info("Building encoder: type=%s mode=%s", encoder_type, model_mode)
    encoder = EnzSubModelForDownstream(
        encoder_type=encoder_type,
        model_mode=model_mode,
        checkpoint_path=(checkpoint_path if needs_ckpt else None),
        freeze_backbone=True,
        device=device,
        strict_lora_load=True,
        allow_lora_reverse_detect=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    encoder.eval()
    return encoder

def _backbone_and_kind(encoder):
    """Return (backbone_module, 'esm'|'protbert')."""
    if hasattr(encoder.encoder, "esm"):
        return encoder.encoder.esm, "esm"
    if hasattr(encoder.encoder, "bert"):
        return encoder.encoder.bert, "protbert"
    raise AttributeError("Cannot find backbone (no .esm or .bert on encoder.encoder)")

def _preprocess_sequence(seq: str, max_len: int) -> str:
    return seq.upper().replace(" ", "")[:max_len]

@torch.no_grad()
def _embed_batch(encoder, backbone, kind, pairs, device, amp_dtype):
    """Return (B, D) fp32 pooled embeddings for one batch."""
    tokens = encoder.tokenize(pairs)  # ESM: Tensor [B,L]; ProtBERT: dict

    if kind == "esm":
        tokens = tokens.to(device)
        repr_layer = encoder.repr_layer
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                results = backbone(tokens, repr_layers=[repr_layer])
            hidden = results["representations"][repr_layer].float()   # bf16 -> fp32
        else:
            results = backbone(tokens, repr_layers=[repr_layer])
            hidden = results["representations"][repr_layer].float()

        enc = encoder.encoder
        residue_mask = (
            (tokens != enc.padding_idx) &
            (tokens != enc.cls_idx) &
            (tokens != enc.eos_idx)
        )  # [B, L]

    else:  # protbert
        tokens = {k: v.to(device) for k, v in tokens.items()}
        repr_layer = encoder.repr_layer
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = backbone(output_hidden_states=True, **tokens)
            hidden = outputs.hidden_states[repr_layer].float()        # bf16 -> fp32
        else:
            outputs = backbone(output_hidden_states=True, **tokens)
            hidden = outputs.hidden_states[repr_layer].float()

        enc = encoder.encoder
        input_ids = tokens["input_ids"]
        attn = tokens["attention_mask"].bool()
        residue_mask = (
            attn &
            (input_ids != enc.pad_token_id) &
            (input_ids != enc.cls_token_id) &
            (input_ids != enc.sep_token_id)
        )  # [B, L]

    mask_f = residue_mask.unsqueeze(-1).float()                       # [B, L, 1]
    summed = (hidden * mask_f).sum(dim=1)                             # [B, D]
    counts = mask_f.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    h = (summed / counts).float()
    return h

@torch.no_grad()
def extract_embeddings(encoder, records, embedding_dim, batch_size,
                       max_len, use_fp16, device, desc):
    N = len(records)
    out = np.zeros((N, embedding_dim), dtype=np.float32)

    effective_max_len = min(max_len, encoder.max_seq_len)

    order = sorted(range(N), key=lambda i: -records[i]["length"])

    amp_dtype = None
    if use_fp16 and device.startswith("cuda"):
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)

    backbone, kind = _backbone_and_kind(encoder)
    log.info("%s: backbone kind=%s, amp_dtype=%s, max_len=%d",
             desc, kind, amp_dtype, effective_max_len)

    pbar = tqdm(total=N, desc=desc, unit="seq")
    i = 0
    while i < len(order):
        idxs = order[i:i + batch_size]
        i += batch_size
        pairs = [(str(j), _preprocess_sequence(records[j]["sequence"],
                                               effective_max_len))
                 for j in idxs]
        h = _embed_batch(encoder, backbone, kind, pairs, device, amp_dtype)
        h_cpu = h.cpu().numpy()
        for b, j in enumerate(idxs):
            out[j] = h_cpu[b]
        pbar.update(len(idxs))
    pbar.close()
    return out

def cache_paths(out_dir: Path, model_tag: str):
    sub = out_dir / model_tag
    sub.mkdir(parents=True, exist_ok=True)
    return sub / "embeddings.npy", sub / "metadata.csv"

def save_embeddings_and_meta(embeddings, records, model_tag, out_dir):
    emb_path, meta_path = cache_paths(out_dir, model_tag)
    np.save(emb_path, embeddings)
    df = pd.DataFrame([
        {"embedding_index": i, "seq_id": r["seq_id"], "is_random": r["is_random"],
         "source_id": r["source_id"], "length": r["length"], "model": model_tag}
        for i, r in enumerate(records)
    ])
    df.to_csv(meta_path, index=False)
    log.info("Saved %s: emb=%s, meta=%s", model_tag, emb_path, meta_path)

def load_cached(model_tag: str, out_dir: Path):
    emb_path, meta_path = cache_paths(out_dir, model_tag)
    if emb_path.exists() and meta_path.exists():
        emb = np.load(emb_path)
        meta = pd.read_csv(meta_path)
        if len(emb) != len(meta):
            log.warning("Cache row count mismatch for %s; ignoring cache.", model_tag)
            return None
        return emb, meta
    return None

def compute_rns(embeddings, is_random, k_list, balance_pool, n_iterations, seed):
    is_random = is_random.astype(bool)
    real_idx = np.where(~is_random)[0]
    rand_idx = np.where(is_random)[0]
    n_real, n_rand = len(real_idx), len(rand_idx)
    log.info("RNS: n_real=%d, n_rand=%d, k_list=%s", n_real, n_rand, k_list)

    if not balance_pool:
        return _compute_rns_single_pass(embeddings, real_idx, rand_idx, k_list)

    pool_rand_n = min(n_real, n_rand)
    if pool_rand_n == n_rand:
        log.info("RNS: |random| <= |real|; balancing has no effect, single pass.")
        return _compute_rns_single_pass(embeddings, real_idx, rand_idx, k_list)

    log.info("RNS: undersampling %d random seqs per iter over %d iters",
             pool_rand_n, n_iterations)
    rng = np.random.default_rng(seed)
    sums = {k: np.zeros(n_real, dtype=np.float64) for k in k_list}
    for _it in tqdm(range(n_iterations), desc="rns-iters", unit="iter"):
        sampled = rng.choice(rand_idx, size=pool_rand_n, replace=False)
        per_iter = _compute_rns_single_pass(embeddings, real_idx, sampled, k_list)
        for k in k_list:
            sums[k] += per_iter[k]
    return {k: sums[k] / n_iterations for k in k_list}

def _compute_rns_single_pass(embeddings, real_idx, rand_idx, k_list):
    pool_indices = np.concatenate([real_idx, rand_idx])
    pool_emb = embeddings[pool_indices]
    n_real = len(real_idx)
    n_pool = len(pool_indices)
    is_random_in_pool = np.zeros(n_pool, dtype=bool)
    is_random_in_pool[n_real:] = True

    max_k = max(k_list)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine",
                          algorithm="brute", n_jobs=-1)
    nn.fit(pool_emb)
    _dists, idxs = nn.kneighbors(pool_emb[:n_real], return_distance=True)

    self_col = np.arange(n_real)[:, None]
    self_mask = (idxs == self_col)

    out: Dict[int, np.ndarray] = {}
    for k in k_list:
        rns_vals = np.empty(n_real, dtype=np.float64)
        for i in range(n_real):
            row = idxs[i]
            keep = row[~self_mask[i]][:k]
            if len(keep) < k:
                rns_vals[i] = is_random_in_pool[keep].mean() if len(keep) else 0.0
            else:
                rns_vals[i] = is_random_in_pool[keep].mean()
        out[k] = rns_vals
    return out

def make_summary(rns_base, rns_cmp, cmp_name, real_meta, out_dir):
    rows = []
    for k in rns_base:
        for i, row in real_meta.reset_index(drop=True).iterrows():
            base = float(rns_base[k][i])
            cmp_v = float(rns_cmp[k][i])
            rows.append({"seq_id": row["seq_id"], "source_id": row["source_id"],
                         "length": row["length"], "k": k,
                         "base_rns": base, f"{cmp_name}_rns": cmp_v,
                         "delta_rns": cmp_v - base})
    per_seq = pd.DataFrame(rows)
    per_seq.to_csv(out_dir / "rns_per_sequence.csv", index=False)

    summary_rows = []
    for tag, rns_dict in [("base", rns_base), (cmp_name, rns_cmp)]:
        for k, arr in rns_dict.items():
            summary_rows.append({
                "model": tag, "k": k,
                "mean_rns": float(np.mean(arr)),
                "median_rns": float(np.median(arr)),
                "std_rns": float(np.std(arr)),
                "min_rns": float(np.min(arr)),
                "max_rns": float(np.max(arr)),
                "ratio_rns_gt_0": float((arr > 0).mean()),
                "ratio_rns_gt_0_01": float((arr > 0.01).mean()),
                "ratio_rns_gt_0_05": float((arr > 0.05).mean()),
                "ratio_rns_gt_0_1": float((arr > 0.1).mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "rns_summary.csv", index=False)

    cmp_rows = []
    for k in rns_base:
        b, c = rns_base[k], rns_cmp[k]
        delta = c - b
        cmp_rows.append({
            "k": k,
            "base_mean_rns": float(np.mean(b)),
            f"{cmp_name}_mean_rns": float(np.mean(c)),
            "mean_delta_rns": float(np.mean(delta)),
            "median_delta_rns": float(np.median(delta)),
            "fraction_improved": float((delta < 0).mean()),
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out_dir / "rns_base_vs_cmp.csv", index=False)
    return per_seq, summary, cmp

def _save_both(fig, fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

def make_plots(rns_base, rns_cmp, cmp_name, out_dir):
    fig_dir = out_dir / "figures"
    ks = sorted(rns_base.keys())

    fig, ax = plt.subplots(figsize=(2 + 1.5 * len(ks), 4))
    positions, labels, data = [], [], []
    for j, k in enumerate(ks):
        positions.extend([j * 3, j * 3 + 1])
        labels.extend([f"base\nk={k}", f"{cmp_name}\nk={k}"])
        data.extend([rns_base[k], rns_cmp[k]])
    bp = ax.boxplot(data, positions=positions, widths=0.7, showfliers=False,
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#DD8452"] * len(ks)):
        patch.set_facecolor(c)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RNS")
    ax.set_title(f"RNS by model and k (base vs {cmp_name})")
    _save_both(fig, fig_dir, "rns_boxplot_base_vs_cmp")

    for k in ks:
        b, c = rns_base[k], rns_cmp[k]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(b, c, s=4, alpha=0.4, edgecolors="none")
        lo, hi = min(b.min(), c.min()), max(b.max(), c.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("base RNS")
        ax.set_ylabel(f"{cmp_name} RNS")
        ax.set_title(f"base vs {cmp_name} (k={k})\nbelow diagonal: {cmp_name} improved")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        _save_both(fig, fig_dir, f"rns_scatter_k{k}")

    for k in ks:
        delta = rns_cmp[k] - rns_base[k]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(delta, bins=60, color="#4C72B0", edgecolor="white")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"delta_rns = {cmp_name} - base")
        ax.set_ylabel("count")
        ax.set_title(f"Delta RNS distribution (k={k})\nleft of 0 = {cmp_name} decreased RNS")
        _save_both(fig, fig_dir, f"rns_delta_hist_k{k}")

def main():
    cfg = load_config()
    _ensure_sub_importable(cfg.get("sub_package_parent"))

    cmp_mode = cfg["compare_mode"]
    if cmp_mode not in VALID_COMPARE_MODES:
        sys.exit(f"ERROR: compare_mode must be one of {VALID_COMPARE_MODES}, "
                 f"got {cmp_mode!r}")

    encoder_type = cfg["encoder_type"]
    out_dir = Path(cfg["output_dir"].format(encoder=encoder_type, cmp=cmp_mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config_used.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    t0 = time.time()
    real_p = Path(cfg["real_fasta"])
    rand_p = Path(cfg["random_fasta"])
    for p in (real_p, rand_p):
        if not p.is_file():
            sys.exit(f"ERROR: missing FASTA: {p}")

    records, n_skipped_long, n_nonstd = load_sequences(real_p, rand_p, cfg["max_len"])
    n_real = sum(1 for r in records if r["is_random"] == 0)
    n_rand = sum(1 for r in records if r["is_random"] == 1)
    log.info("Loaded %d sequences (real=%d, random=%d). Skipped %d long, %d non-standard.",
             len(records), n_real, n_rand, n_skipped_long, n_nonstd)
    if n_real == 0 or n_rand == 0:
        sys.exit("ERROR: need both real and random sequences to compute RNS.")

    device = cfg["device"]

    def _get_embeddings(tag, model_mode, checkpoint):
        if not cfg["force_recompute"]:
            cached = load_cached(tag, out_dir)
            if cached is not None:
                emb, meta = cached
                if len(emb) == len(records) and (
                    meta["seq_id"].tolist() == [r["seq_id"] for r in records]
                ):
                    log.info("Using cached embeddings for '%s' (%s)", tag, emb.shape)
                    return emb, meta
                log.warning("Cache for '%s' mismatch; recomputing.", tag)

        encoder = build_encoder(
            encoder_type=encoder_type, model_mode=model_mode,
            checkpoint_path=checkpoint, device=device,
            lora_rank=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        )
        embedding_dim = int(encoder.hidden_dim)
        log.info("[%s] mode=%s dim=%d use_fp16=%s",
                 tag, model_mode, embedding_dim, cfg["use_fp16"])

        emb = extract_embeddings(
            encoder=encoder, records=records, embedding_dim=embedding_dim,
            batch_size=cfg["batch_size"], max_len=cfg["max_len"],
            use_fp16=cfg["use_fp16"], device=device, desc=f"emb-{tag}",
        )
        save_embeddings_and_meta(emb, records, tag, out_dir)

        del encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        _, meta = load_cached(tag, out_dir)
        return emb, meta

    base_emb, base_meta = _get_embeddings("base", "base", None)
    cmp_emb, cmp_meta = _get_embeddings(
        cmp_mode, cmp_mode, cfg["compare_checkpoint"]
    )

    expected_ids = [r["seq_id"] for r in records]
    for tag, meta, emb in [("base", base_meta, base_emb), (cmp_mode, cmp_meta, cmp_emb)]:
        assert len(emb) == len(records), \
            f"{tag}: embedding rows ({len(emb)}) != records ({len(records)})"
        assert meta["seq_id"].tolist() == expected_ids, \
            f"{tag}: metadata seq_id order does not match current records"
    assert base_emb.shape[1] == cmp_emb.shape[1], \
        "base/comparison embedding dims differ — backbone mismatch?"
    log.info("base embedding shape: %s", base_emb.shape)
    log.info("%s embedding shape: %s", cmp_mode, cmp_emb.shape)

    is_random_arr = np.array([r["is_random"] for r in records], dtype=np.int8)

    log.info("--- Computing RNS for BASE ---")
    rns_base = compute_rns(base_emb, is_random_arr, k_list=cfg["k_list"],
                           balance_pool=cfg["rns_balance_pool"],
                           n_iterations=cfg["rns_n_iterations"], seed=cfg["rns_seed"])
    log.info("--- Computing RNS for %s ---", cmp_mode.upper())
    rns_cmp = compute_rns(cmp_emb, is_random_arr, k_list=cfg["k_list"],
                          balance_pool=cfg["rns_balance_pool"],
                          n_iterations=cfg["rns_n_iterations"], seed=cfg["rns_seed"])

    real_meta = base_meta[base_meta["is_random"] == 0].reset_index(drop=True)
    per_seq, summary, cmp = make_summary(rns_base, rns_cmp, cmp_mode, real_meta, out_dir)
    make_plots(rns_base, rns_cmp, cmp_mode, out_dir)

    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f" RNS: base {encoder_type}  vs  {cmp_mode}  "
          f"({'bf16->fp32 pool' if cfg['use_fp16'] else 'fp32'})")
    print("=" * 64)
    print(f"encoder             : {encoder_type}")
    print(f"comparison mode     : {cmp_mode}")
    print(f"real sequences      : {n_real}")
    print(f"random sequences    : {n_rand}")
    print(f"skipped (len>{cfg['max_len']:>4}) : {n_skipped_long}")
    print(f"non-standard residues: {n_nonstd}")
    print(f"base emb shape      : {base_emb.shape}")
    print(f"{cmp_mode:<4} emb shape      : {cmp_emb.shape}")
    print(f"pool balance        : {cfg['rns_balance_pool']} (iters={cfg['rns_n_iterations']})")
    print("-" * 64)
    hdr_cmp = f"{cmp_mode}_mean"
    print(f"{'k':>6} {'base_mean':>12} {hdr_cmp:>14} {'delta':>10} {'frac_impr':>12}")
    for row in cmp.itertuples(index=False):
        k = row.k
        bm = getattr(row, "base_mean_rns")
        cm = getattr(row, f"{cmp_mode}_mean_rns")
        dm = getattr(row, "mean_delta_rns")
        fi = getattr(row, "fraction_improved")
        print(f"{k:>6} {bm:>12.4f} {cm:>14.4f} {dm:>10.4f} {fi:>12.2%}")
    print("-" * 64)
    print(f"output dir : {out_dir}")
    print(f"elapsed    : {elapsed:.1f} s")
    print("=" * 64)

if __name__ == "__main__":
    main()
