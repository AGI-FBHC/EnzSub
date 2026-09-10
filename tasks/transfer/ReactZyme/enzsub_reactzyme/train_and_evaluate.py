#!/usr/bin/env python3
"""Train and evaluate the EnzGFM-style ReactZyme head with EnzSub embeddings."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from common import load_embedding_dict, load_split, require_coverage, set_seed, stratified_split, write_json

class PairDataset(Dataset):
    def __init__(self, pairs, molecules, enzymes):
        self.pairs, self.molecules, self.enzymes = pairs, molecules, enzymes
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        molecule, enzyme, label = self.pairs[index]
        return self.molecules[molecule], self.enzymes[enzyme], torch.tensor(label, dtype=torch.float32)

class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.scale = dim ** 0.5
    def forward(self, query, key_value):
        weights = torch.softmax(self.q(query) @ self.k(key_value).transpose(1, 2) / self.scale, dim=-1)
        return weights @ self.v(key_value)

class EnzGFMReactZymeHead(nn.Module):
    """Architecture matched to EnzGFM's released ReactZyme retrieval script."""
    def __init__(self, molecule_dim: int, enzyme_dim: int, hidden_dim: int = 128, layers: int = 4):
        super().__init__()
        self.molecule_encoder = nn.Sequential(nn.Linear(molecule_dim, 256, bias=False), nn.BatchNorm1d(256), nn.SiLU(), nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.SiLU(), nn.Linear(256, hidden_dim, bias=False))
        self.enzyme_encoder = nn.Sequential(nn.Linear(enzyme_dim, 512, bias=False), nn.BatchNorm1d(512), nn.SiLU(), nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.SiLU(), nn.Linear(256, hidden_dim, bias=False))
        self.molecule_attention, self.enzyme_attention = CrossAttention(hidden_dim), CrossAttention(hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=layers, num_decoder_layers=layers, dim_feedforward=hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 64, bias=False), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 16, bias=False), nn.Linear(16, 1, bias=False))
    def forward(self, molecule, enzyme):
        molecule = self.molecule_encoder(molecule).unsqueeze(1)
        enzyme = self.enzyme_encoder(enzyme).unsqueeze(1)
        tokens = torch.cat([self.molecule_attention(molecule, enzyme), self.enzyme_attention(enzyme, molecule)], dim=1)
        return self.classifier(self.transformer(tokens, tokens).sum(dim=1)).squeeze(-1)

def run_epoch(model, loader, optimizer, pos_weight, device, train, accumulation_steps=1):
    model.train(train)
    total_loss, total_n = 0.0, 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step, (molecule, enzyme, label) in enumerate(loader, start=1):
        molecule, enzyme, label = molecule.to(device), enzyme.to(device), label.to(device)
        with torch.set_grad_enabled(train):
            loss = F.binary_cross_entropy_with_logits(model(molecule, enzyme), label, pos_weight=pos_weight)
        if train:
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * label.numel(); total_n += label.numel()
    return total_loss / total_n

def rank_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5, 10, 20, 50)) -> Dict[str, float]:
    ordered = torch.argsort(scores, dim=1, descending=True, stable=True)
    results, ap, rr = {f"top_{k}_accuracy": [] for k in ks}, [], []
    for row, order in enumerate(ordered):
        relevance = labels[row][order].float(); positives = int(relevance.sum())
        if positives == 0: continue
        precision = torch.cumsum(relevance, 0) / torch.arange(1, relevance.numel() + 1, dtype=torch.float32)
        ap.append(float((precision * relevance).sum() / positives))
        first = int(torch.where(relevance > 0)[0][0]); rr.append(1.0 / (first + 1))
        for k in ks:
            limit = min(k, relevance.numel()); top = relevance[:limit]
            results[f"top_{k}_accuracy"].append(float(top.any()))
            discounts = torch.log2(torch.arange(2, limit + 2, dtype=torch.float32))
            dcg = float((top / discounts).sum()); ideal = min(positives, limit)
            idcg = float((torch.ones(ideal) / torch.log2(torch.arange(2, ideal + 2, dtype=torch.float32))).sum())
            results.setdefault(f"ndcg_at_{k}", []).append(dcg / idcg)
    summary = {key: float(sum(values) / len(values)) for key, values in results.items()}
    summary.update({"map": float(sum(ap) / len(ap)), "mrr_first_hit": float(sum(rr) / len(rr))})
    return summary

@torch.no_grad()
def retrieval_metrics(model, positives, molecules, enzymes, device, batch_size):
    unique_molecules = sorted({m for m, _, _ in positives}); unique_enzymes = sorted({e for _, e, _ in positives})
    mol_tensor = torch.stack([molecules[key] for key in unique_molecules]).to(device)
    enzyme_tensor = torch.stack([enzymes[key] for key in unique_enzymes]).to(device)
    labels = torch.zeros((len(unique_enzymes), len(unique_molecules)))
    mi, ei = {k: i for i, k in enumerate(unique_molecules)}, {k: i for i, k in enumerate(unique_enzymes)}
    for molecule, enzyme, _ in positives: labels[ei[enzyme], mi[molecule]] = 1.0
    rows = []; model.eval()
    for enzyme in enzyme_tensor:
        chunks = []
        for start in range(0, len(mol_tensor), batch_size):
            mol_batch = mol_tensor[start : start + batch_size]
            chunks.append(model(mol_batch, enzyme.expand(mol_batch.size(0), -1)).cpu())
        rows.append(torch.cat(chunks))
    scores = torch.stack(rows)
    return {"enzyme_to_reaction": rank_metrics(scores, labels), "reaction_to_enzyme": rank_metrics(scores.T, labels.T), "candidate_counts": {"enzymes": len(unique_enzymes), "reactions": len(unique_molecules)}}

def aggregate_results(runs):
    """Mean and sample SD across independent seeds, as reported by EnzGFM."""
    aggregate = {}
    for direction in ("enzyme_to_reaction", "reaction_to_enzyme"):
        metric_names = runs[0][direction].keys()
        aggregate[direction] = {}
        for name in metric_names:
            values = [run[direction][name] for run in runs]
            aggregate[direction][name] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return aggregate

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True); parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory produced by this workflow's reaction stage")
    parser.add_argument("--model-name", required=True, help="Configured enzyme-representation branch name")
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--seeds", nargs="+", type=int, default=[47, 48, 49, 50, 51])
    parser.add_argument("--resume", action="store_true", help="Reuse complete seed metrics already present in output-dir")
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--accumulation-steps", type=int, default=1); parser.add_argument("--max-epochs", type=int, default=50); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.10); parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def main() -> int:
    args = parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    molecules = load_embedding_dict(args.artifacts_dir / "mat_reaction_embeddings.pt", "MAT reaction")
    enzymes = load_embedding_dict(args.artifacts_dir / "models" / args.model_name / "enzyme_embeddings.pt", f"{args.model_name} enzyme")
    train_pairs, test_pairs = load_split(args.split_dir, args.split_name); require_coverage(train_pairs + test_pairs, molecules, enzymes)
    positives = [pair for pair in test_pairs if pair[2] == 1.0]; molecule_dim, enzyme_dim = next(iter(molecules.values())).numel(), next(iter(enzymes.values())).numel()
    all_results = []
    for seed in args.seeds:
        run_dir = args.output_dir / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            required = ("seed", "enzyme_to_reaction", "reaction_to_enzyme", "candidate_counts")
            if all(key in existing for key in required) and int(existing["seed"]) == seed:
                print(f"Resuming: reusing completed seed {seed}")
                all_results.append(existing)
                continue
        set_seed(seed); train, validation = stratified_split(train_pairs, args.validation_fraction, seed)
        train_loader = DataLoader(PairDataset(train, molecules, enzymes), batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(PairDataset(validation, molecules, enzymes), batch_size=args.batch_size)
        model = EnzGFMReactZymeHead(molecule_dim, enzyme_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        def lr_factor(epoch):
            if epoch < 5:
                return (epoch + 1) / 5
            return max(0.0, (args.max_epochs - epoch) / max(args.max_epochs - 5, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        pos_weight = torch.tensor(sum(label == 0.0 for _, _, label in train) / max(sum(label == 1.0 for _, _, label in train), 1), device=device)
        best_loss, stale, best_state = float("inf"), 0, None
        for epoch in range(args.max_epochs):
            train_loss = run_epoch(model, train_loader, optimizer, pos_weight, device, True, args.accumulation_steps); validation_loss = run_epoch(model, validation_loader, optimizer, pos_weight, device, False)
            scheduler.step()
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience: break
        model.load_state_dict(best_state); metrics = retrieval_metrics(model, positives, molecules, enzymes, device, args.batch_size)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "seed": seed, "best_validation_loss": best_loss}, run_dir / "model.pt")
        record = {"seed": seed, "best_validation_loss": best_loss, "epochs_ran": epoch + 1, "train_loss_last": train_loss, "train_pair_count": len(train), "validation_pair_count": len(validation), **metrics}; write_json(run_dir / "metrics.json", record); all_results.append(record)
    manifest_path = args.artifacts_dir / f"protocol_manifest_{args.split_name}.json"
    write_json(args.output_dir / "summary.json", {"protocol": "EnzGFM-style enzyme-similarity ReactZyme evaluation", "model_name": args.model_name, "split_name": args.split_name, "molecule_dim": molecule_dim, "enzyme_dim": enzyme_dim, "seeds": args.seeds, "protocol_manifest": str(manifest_path) if manifest_path.exists() else None, "runs": all_results, "aggregate_mean_sd": aggregate_results(all_results)})
    print(f"Completed {len(all_results)} seeds; results written to {args.output_dir}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
