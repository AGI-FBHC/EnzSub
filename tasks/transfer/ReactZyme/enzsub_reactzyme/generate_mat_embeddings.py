#!/usr/bin/env python3
"""Generate 1024-dimensional ReactZyme reaction vectors with the EnzGFM MAT contract."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

from common import load_pair_file, normalise_reaction, write_json

MODEL_PARAMS = {"d_atom": 28, "d_model": 1024, "N": 8, "h": 16, "N_dense": 1,
                "lambda_attention": 0.33, "lambda_distance": 0.33,
                "leaky_relu_slope": 0.1, "dense_output_nonlinearity": "relu",
                "distance_matrix_kernel": "exp", "dropout": 0.0, "aggregation_type": "mean"}

def one_hot(value, choices):
    return np.asarray([value == (value if value in choices else choices[-1]) for value in choices], dtype=np.float32)

def molecule_graph(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse molecule: {smiles}")
    AllChem.Compute2DCoords(molecule)
    features = np.asarray([np.concatenate((
        one_hot(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]),
        one_hot(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5]),
        one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]),
        one_hot(atom.GetFormalCharge(), [-1, 0, 1]),
        [atom.IsInRing(), atom.GetIsAromatic()],
    )) for atom in molecule.GetAtoms()], dtype=np.float32)
    adjacency = np.eye(molecule.GetNumAtoms(), dtype=np.float32)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left, right] = adjacency[right, left] = 1.0
    conformer = molecule.GetConformer()
    positions = np.asarray([[conformer.GetAtomPosition(index).x, conformer.GetAtomPosition(index).y, conformer.GetAtomPosition(index).z] for index in range(molecule.GetNumAtoms())])
    distance = pairwise_distances(positions).astype(np.float32)
    node_features = np.zeros((features.shape[0] + 1, features.shape[1] + 1), dtype=np.float32)
    node_features[1:, 1:], node_features[0, 0] = features, 1.0
    padded_adjacency = np.zeros((adjacency.shape[0] + 1, adjacency.shape[1] + 1), dtype=np.float32)
    padded_adjacency[1:, 1:] = adjacency
    padded_distance = np.full((distance.shape[0] + 1, distance.shape[1] + 1), 1e6, dtype=np.float32)
    padded_distance[1:, 1:] = distance
    return node_features, padded_adjacency, padded_distance

def load_mat(checkpoint: Path, device: torch.device):
    from mat import make_model
    model = make_model(**MODEL_PARAMS)
    source = torch.load(checkpoint, map_location="cpu")
    target = model.state_dict()
    for name, value in source.items():
        if "generator" not in name:
            target[name].copy_(value.detach() if isinstance(value, torch.nn.Parameter) else value)
    return model.to(device).eval()

def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

@torch.no_grad()
def encode_reaction(model, reaction: str, device: torch.device):
    component_vectors = []
    for component in reaction.split("."):
        features, adjacency, distance = molecule_graph(component)
        features = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=device).unsqueeze(0)
        distance = torch.as_tensor(distance, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.sum(torch.abs(features), dim=-1) != 0
        component_vectors.append(model.encode(features, mask, adjacency, distance, None).squeeze(0).mean(0))
    return torch.stack(component_vectors).mean(0).detach().cpu().float()

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactzyme-root", type=Path, required=True, help="Repository containing mat.py")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="seq_smi")
    parser.add_argument("--mat-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if not args.mat_checkpoint.is_file():
        raise FileNotFoundError(args.mat_checkpoint)
    sys.path.insert(0, str(args.reactzyme_root))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = [args.split_dir / f"positive_train_val_{args.split_name}.pt", args.split_dir / f"positive_test_{args.split_name}.pt"]
    reactions = sorted({reaction for path in paths for reaction, _, _ in load_pair_file(path, 1.0)})
    model, embeddings = load_mat(args.mat_checkpoint, device), {}
    for index, reaction in enumerate(reactions, start=1):
        embeddings[reaction] = encode_reaction(model, reaction, device)
        if device.type == "cuda": torch.cuda.empty_cache()
        if index % 1000 == 0: print(f"Encoded {index}/{len(reactions)} reactions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": {"artifact": "enzgfm_mat_reaction_embeddings", "dimension": 1024, "pooling": "MAT atom mean, then reaction-component mean", "mat_checkpoint": str(args.mat_checkpoint), "mat_checkpoint_sha256": sha256_file(args.mat_checkpoint), "split_name": args.split_name}}, args.output)
    write_json(args.output.with_suffix(".json"), {"artifact": "enzgfm_mat_reaction_embeddings", "reaction_count": len(embeddings), "dimension": 1024, "split_name": args.split_name, "mat_checkpoint": str(args.mat_checkpoint)})
    print(f"Saved {len(embeddings)} MAT reaction embeddings to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
