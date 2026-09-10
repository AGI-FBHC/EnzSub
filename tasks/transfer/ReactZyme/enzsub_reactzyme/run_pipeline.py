#!/usr/bin/env python3
"""Run all configured enzyme-representation branches on one ReactZyme protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "enzsub_reactzyme"

def command(python_bin, script, *args):
    return [python_bin, str(WORKFLOW / script), *map(str, args)]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "negatives", "reaction", "representations", "audit", "train", "collect"], default="all")
    parser.add_argument("--models", nargs="+", help="Optional subset of configured model names")
    parser.add_argument("--resume", action="store_true", help="Reuse completed seed outputs during training")
    return parser.parse_args()

def selected_models(configured, names):
    known = {item["name"]: item for item in configured}
    if len(known) != len(configured): raise ValueError("Every models[].name must be unique")
    if not names: return configured
    missing = set(names) - known.keys()
    if missing: raise ValueError(f"Unknown model names: {sorted(missing)}")
    return [known[name] for name in names]

def representation_command(python_bin, model, paths, split_dir, split_name, device, output):
    kind = model["kind"]
    if kind == "enzsub":
        return command(python_bin, "extract_enzsub_embeddings.py", "--enzsub-code-root", paths["enzsub_code_root"], "--split-dir", split_dir, "--split-name", split_name, "--checkpoint", model["checkpoint"], "--model-mode", model.get("model_mode", "cpt_sub"), "--encoder-type", model["encoder_type"], "--batch-size", model.get("batch_size", 4), "--device", device, "--output", output)
    if kind == "reactzyme_esm":
        return command(python_bin, "extract_reactzyme_esm_embeddings.py", "--checkpoint", model["checkpoint"], "--split-dir", split_dir, "--split-name", split_name, "--max-length", model.get("max_length", 5000), "--device", device, "--output", output)
    if kind == "fair_esm":
        return command(python_bin, "extract_fair_esm_embeddings.py", "--checkpoint", model["checkpoint"], "--split-dir", split_dir, "--split-name", split_name, "--max-length", model.get("max_length", 1022), "--device", device, "--output", output)
    if kind == "external_plm":
        adapter_source = model.get("adapter_source", paths.get("external_plm_adapter_source"))
        if not adapter_source:
            raise ValueError(f"external_plm model {model['name']} requires adapter_source or paths.external_plm_adapter_source")
        args = [
            "--adapter-source", adapter_source,
            "--backend", model["backend"],
            "--model-name", model["name"],
            "--model-id", model["model_id"],
            "--model-class", model.get("model_class", "auto"),
            "--tokenizer-class", model.get("tokenizer_class", "auto"),
            "--input-mode", model.get("input_mode", "raw"),
            "--prefix", model.get("prefix", ""),
            "--dtype", model.get("dtype", "auto"),
            "--max-length", model.get("max_length", 1022),
            "--batch-size", model.get("batch_size", 4),
            "--token-budget", model.get("token_budget", 4096),
            "--max-batch-size", model.get("max_batch_size", 16),
            "--save-every", model.get("save_every", 100),
            "--split-dir", split_dir, "--split-name", split_name,
            "--device", device, "--output", output,
        ]
        for key, flag in (("local_path", "--local-path"), ("pretrained_name", "--pretrained-name")):
            if model.get(key):
                args.extend((flag, model[key]))
        if model.get("offline", False):
            args.append("--offline")
        if model.get("validate_batch_equivalence", False):
            args.append("--validate-batch-equivalence")
        if model.get("validate_min_cosine") is not None:
            args.extend(("--validate-min-cosine", model["validate_min_cosine"]))
        return command(python_bin, "extract_external_plm_embeddings.py", *args)
    raise ValueError(f"Unsupported kind for {model['name']}: {kind}")

def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    paths, runtime, training = (config[key] for key in ("paths", "runtime", "training"))
    models = selected_models(config["models"], args.models)
    split_dir, artifacts, results_dir = Path(paths["split_dir"]), Path(paths["artifacts_dir"]), Path(paths["results_dir"])
    mat_output = artifacts / "mat_reaction_embeddings.pt"
    python_bin = runtime.get("python_bin", sys.executable)
    device, split_name = runtime.get("device", "cuda"), runtime.get("split_name", "seq_smi")
    stages = ("negatives", "reaction", "representations", "audit", "train") if args.stage == "all" else (args.stage,)
    if "negatives" in stages:
        sampling = config.get("negative_sampling", {})
        subprocess.run(command(python_bin, "generate_negatives.py", "--split-dir", split_dir, "--split-name", split_name, "--seed", sampling.get("seed", 2025), "--per-positive", sampling.get("per_positive", 1), "--candidate-pool", sampling.get("candidate_pool", 1000), "--selection", sampling.get("selection", "uniform_eligible_from_top_pool")), check=True)
    if "reaction" in stages:
        subprocess.run(command(python_bin, "generate_mat_embeddings.py", "--reactzyme-root", ROOT, "--split-dir", split_dir, "--split-name", split_name, "--mat-checkpoint", paths["mat_checkpoint"], "--device", device, "--output", mat_output), check=True)
    if "representations" in stages:
        for model in models:
            output = artifacts / "models" / model["name"] / "enzyme_embeddings.pt"
            representation_python = model.get("representation_python_bin", python_bin)
            representation_args = representation_command(representation_python, model, paths, split_dir, split_name, device, output)
            if args.resume and model["kind"] == "external_plm":
                representation_args.append("--resume")
            subprocess.run(representation_args, check=True)
    if "audit" in stages:
        audit_args = ["--config", args.config]
        if args.models:
            audit_args.extend(["--models", *args.models])
        subprocess.run(command(python_bin, "audit_protocol.py", *audit_args), check=True)
    if "train" in stages:
        for model in models:
            train_args = ["--split-dir", split_dir, "--split-name", split_name, "--artifacts-dir", artifacts, "--model-name", model["name"], "--output-dir", results_dir / model["name"], "--seeds", *training.get("seeds", [47, 48, 49, 50, 51]), "--batch-size", training.get("batch_size", 512), "--accumulation-steps", training.get("accumulation_steps", 1), "--max-epochs", training.get("max_epochs", 50), "--patience", training.get("patience", 5), "--learning-rate", training.get("learning_rate", 0.0005), "--weight-decay", training.get("weight_decay", 0.001), "--validation-fraction", training.get("validation_fraction", 0.10), "--device", device]
            if args.resume:
                train_args.append("--resume")
            subprocess.run(command(python_bin, "train_and_evaluate.py", *train_args), check=True)
        subprocess.run(command(python_bin, "collect_comparison.py", "--results-dir", results_dir, "--models", *(model["name"] for model in models)), check=True)
    if "collect" in stages:
        comparison_models = config.get("comparison_models", [model["name"] for model in models])
        subprocess.run(command(python_bin, "collect_comparison.py", "--results-dir", results_dir, "--models", *comparison_models), check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
