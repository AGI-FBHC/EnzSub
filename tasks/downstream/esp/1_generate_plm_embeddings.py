#!/usr/bin/env python3
"""Generate external-PLM enzyme embeddings for ESP pickle files.

This is a self-contained ESP implementation. It supports Hugging Face T5
encoders, standard Hugging Face encoders, and the public ESM3 API. It reads and
writes the existing ESP pickle format:

* read protein sequences from the ``sequence`` column;
* encode each unique normalized sequence once across all input pickle files;
* write the vectors back to the existing ``enzyme_vector`` column.

Consequently, the existing ESP MLP/GBDT/k-NN evaluation code does not need to
know which PLM produced the enzyme representation.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_BACKENDS = frozenset({"hf_t5", "hf_encoder", "esm3"})

def preprocess_sequence(sequence: str, max_len: int) -> Tuple[str, int, int]:
    """Normalize a protein sequence and truncate by biological residues."""
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    compact = re.sub(r"\s+", "", sequence).upper()
    original_len = len(compact)
    cleaned_chars = [aa if aa in STANDARD_AAS else "X" for aa in compact]
    replaced = sum(a != b for a, b in zip(compact, cleaned_chars))
    cleaned = "".join(cleaned_chars)[:max_len]
    if not cleaned:
        raise ValueError("Encountered an empty protein sequence after preprocessing")
    return cleaned, original_len, replaced

def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def make_batches(
    records: Sequence[Tuple[str, str]],
    batch_size: int,
    token_budget: int = 0,
    max_batch_size: int = 64,
) -> List[List[Tuple[str, str]]]:
    """Length-sort records, then form fixed-size or token-budget batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")

    ordered = sorted(records, key=lambda item: len(item[1]))
    if token_budget <= 0:
        return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]

    batches: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    current_max_len = 0
    for record in ordered:
        candidate_max = max(current_max_len, len(record[1]))
        exceeds_budget = bool(current) and (len(current) + 1) * candidate_max > token_budget
        reaches_size_cap = len(current) >= max_batch_size
        if exceeds_budget or reaches_size_cap:
            batches.append(current)
            current = [record]
            current_max_len = len(record[1])
        else:
            current.append(record)
            current_max_len = candidate_max
    if current:
        batches.append(current)
    return batches

def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for embedding generation. "
            "Run this script in the GPU environment."
        ) from exc
    return torch

def resolve_dtype(torch, name: str, device: str):
    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose auto/fp32/fp16/bf16")
    dtype = mapping[normalized]
    if device.startswith("cpu") and dtype in (torch.float16, torch.bfloat16):
        raise ValueError(f"dtype={name} is not supported by this script on CPU")
    return dtype

def autocast_context(torch, device: str, dtype):
    if not device.startswith("cuda") or dtype not in (torch.float16, torch.bfloat16):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)

def special_tokens_mask(
    tokenizer,
    input_ids,
    encoded: MutableMapping[str, Any],
    torch,
):
    mask = encoded.pop("special_tokens_mask", None)
    if mask is not None:
        return mask.bool()
    rows = [
        tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
        for row in input_ids.cpu()
    ]
    return torch.tensor(rows, dtype=torch.bool, device=input_ids.device)

def remove_prefix_from_mask(
    input_ids,
    residue_mask,
    prefix_ids: Sequence[int],
) -> None:
    """Exclude an Ankh3 task prefix from residue pooling."""
    if not prefix_ids:
        return
    prefix = list(prefix_ids)
    for row_index in range(input_ids.shape[0]):
        ids = input_ids[row_index].tolist()
        found_at: Optional[int] = None
        for start in range(0, min(3, max(1, len(ids) - len(prefix) + 1))):
            if ids[start:start + len(prefix)] == prefix:
                found_at = start
                break
        if found_at is None:
            raise RuntimeError(
                "Could not locate configured Ankh3 prefix tokens in tokenized input. "
                "Check tokenizer/model revision and prefix configuration."
            )
        residue_mask[row_index, found_at:found_at + len(prefix)] = False

def masked_mean(hidden_states, residue_mask):
    weights = residue_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return ((hidden_states * weights).sum(dim=1) / denominator).float()

@dataclass
class AdapterInfo:
    backend: str
    model_id: str
    source: str
    revision: Optional[str]
    dtype: str
    pooling: str = "last_hidden_state_masked_mean_no_special_tokens"

class HFEncoderAdapter:
    """Adapter for Ankh/Ankh3, ProtT5, and standard HF encoders."""

    def __init__(self, spec: Mapping[str, Any], device: str):
        torch = import_torch()
        try:
            from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers and sentencepiece are required: "
                "pip install transformers sentencepiece"
            ) from exc

        self.torch = torch
        self.device = device
        self.spec = dict(spec)
        self.model_id = str(spec["model_id"])
        self.source = str(spec.get("local_path") or self.model_id)
        self.revision = spec.get("revision")
        self.input_mode = str(spec.get("input_mode", "raw"))
        self.prefix = str(spec.get("prefix", ""))
        self.validate_token_count = bool(spec.get("validate_token_count", True))
        self.load_dtype_name = str(spec.get("dtype", "auto"))
        self.load_dtype = resolve_dtype(torch, self.load_dtype_name, device)

        common_kwargs: Dict[str, Any] = {}
        if self.revision:
            common_kwargs["revision"] = self.revision
        if bool(spec.get("trust_remote_code", False)):
            common_kwargs["trust_remote_code"] = True

        tokenizer_kind = str(spec.get("tokenizer_class", "auto"))
        if tokenizer_kind == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.source, **common_kwargs)
        elif tokenizer_kind == "auto":
            self.tokenizer = AutoTokenizer.from_pretrained(self.source, **common_kwargs)
        else:
            raise ValueError(f"Unknown tokenizer_class={tokenizer_kind!r}")

        model_kind = str(spec.get("model_class", "auto"))
        model_kwargs = dict(common_kwargs)
        if self.load_dtype is not None:
            model_kwargs["torch_dtype"] = self.load_dtype
        if model_kind == "t5_encoder":
            self.model = T5EncoderModel.from_pretrained(self.source, **model_kwargs)
        elif model_kind == "auto":
            self.model = AutoModel.from_pretrained(self.source, **model_kwargs)
        else:
            raise ValueError(f"Unknown model_class={model_kind!r}")

        self.model.to(device)
        self.model.eval()
        self.resolved_revision = (
            self.revision
            or getattr(getattr(self.model, "config", None), "_commit_hash", None)
        )
        try:
            self.actual_dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            self.actual_dtype = self.load_dtype_name

        self.prefix_ids: List[int] = []
        if self.prefix:
            self.prefix_ids = list(
                self.tokenizer(self.prefix, add_special_tokens=False)["input_ids"]
            )
            if not self.prefix_ids:
                raise RuntimeError(f"Tokenizer produced no IDs for prefix {self.prefix!r}")

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            backend=str(self.spec["backend"]),
            model_id=self.model_id,
            source=self.source,
            revision=self.resolved_revision,
            dtype=self.actual_dtype,
        )

    def _prepare_inputs(self, sequences: Sequence[str]):
        if self.input_mode == "split_chars":
            prepared: Any = [list(sequence) for sequence in sequences]
            split_words = True
        elif self.input_mode == "space_separated":
            prepared = [" ".join(sequence) for sequence in sequences]
            split_words = False
        elif self.input_mode == "raw":
            prepared = [self.prefix + sequence for sequence in sequences]
            split_words = False
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode!r}")

        kwargs: Dict[str, Any] = {
            "add_special_tokens": True,
            "padding": True,
            "truncation": False,
            "return_attention_mask": True,
            "return_special_tokens_mask": True,
            "return_tensors": "pt",
        }
        if split_words:
            kwargs["is_split_into_words"] = True
        return self.tokenizer(prepared, **kwargs)

    def embed(self, records: Sequence[Tuple[str, str]]):
        torch = self.torch
        sequences = [sequence for _, sequence in records]
        encoded = self._prepare_inputs(sequences)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device).bool()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        residue_mask = attention_mask & ~special_tokens_mask(
            self.tokenizer, input_ids, encoded, torch
        )
        remove_prefix_from_mask(input_ids, residue_mask, self.prefix_ids)

        if self.validate_token_count:
            observed = residue_mask.sum(dim=1).cpu().tolist()
            expected = [len(sequence) for sequence in sequences]
            if observed != expected:
                details = list(zip([sid for sid, _ in records], expected, observed))[:5]
                raise RuntimeError(
                    "Residue-token count mismatch after excluding special tokens. "
                    f"Examples (id, expected, observed): {details}"
                )

        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in set(getattr(self.tokenizer, "model_input_names", []))
        }
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask.long()

        with torch.inference_mode(), autocast_context(
            torch, self.device, self.load_dtype
        ):
            outputs = self.model(**model_inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
        if hidden is None:
            raise RuntimeError(
                f"Model {self.model_id} did not return last_hidden_state/outputs[0]"
            )
        return masked_mean(hidden, residue_mask)

class ESM3Adapter:
    """Adapter using the stable public ESM3 embedding API.

    ``LogitsOutput.embeddings`` is the final transformer residual stream before
    its terminal LayerNorm.  Apply the model's own terminal norm before pooling
    so this pathway matches the final hidden-state representation used for the
    Hugging Face encoder backends.
    """

    def __init__(self, spec: Mapping[str, Any], device: str):
        torch = import_torch()
        try:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise RuntimeError(
                "Biohub ESM is required for ESM3. Install the pinned Biohub esm package."
            ) from exc

        self.torch = torch
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = device
        self.spec = dict(spec)
        self.model_id = str(spec["model_id"])
        self.source = str(spec.get("pretrained_name") or self.model_id)
        self.revision = spec.get("revision")
        self.model = ESM3.from_pretrained(self.source).to(device)
        self.model.eval()
        self.final_norm = getattr(getattr(self.model, "transformer", None), "norm", None)
        if self.final_norm is None:
            raise RuntimeError(
                "The installed ESM3 model does not expose transformer.norm, which is "
                "required to convert LogitsOutput.embeddings from pre-norm residuals "
                "to final hidden states."
            )

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            backend="esm3",
            model_id=self.model_id,
            source=self.source,
            revision=self.revision,
            dtype="bf16_autocast_by_official_api",
            pooling="esm3_final_layernorm_residue_mean_no_special_tokens",
        )

    def embed(self, records: Sequence[Tuple[str, str]]):
        torch = self.torch
        pooled = []
        for seq_id, sequence in records:
            protein = self.ESMProtein(sequence=sequence)
            protein_tensor = self.model.encode(protein)
            if hasattr(protein_tensor, "to"):
                moved = protein_tensor.to(self.device)
                if moved is not None:
                    protein_tensor = moved
            with torch.inference_mode():
                output = self.model.logits(
                    protein_tensor,
                    self.LogitsConfig(sequence=False, return_embeddings=True),
                )
            embedding = output.embeddings
            if embedding is None:
                raise RuntimeError(f"ESM3 returned no embeddings for {seq_id}")
            if embedding.ndim == 3:
                if embedding.shape[0] != 1:
                    raise RuntimeError(f"Unexpected ESM3 embedding shape: {embedding.shape}")
                embedding = embedding[0]
            # ``LogitsOutput.embeddings`` is the pre-LayerNorm residual stream.
            # Reapply ESM3's own terminal norm to obtain its final hidden state.
            norm_parameter = next(self.final_norm.parameters(), None)
            norm_dtype = norm_parameter.dtype if norm_parameter is not None else embedding.dtype
            with torch.inference_mode():
                embedding = self.final_norm(embedding.to(dtype=norm_dtype)).float()
            if embedding.shape[0] < len(sequence) + 2:
                raise RuntimeError(
                    f"ESM3 token count too short for {seq_id}: "
                    f"tokens={embedding.shape[0]}, residues={len(sequence)}"
                )
            residue_embedding = embedding[1:len(sequence) + 1]
            pooled.append(residue_embedding.mean(dim=0).float().cpu())
        return torch.stack(pooled, dim=0)

def build_adapter(spec: Mapping[str, Any], device: str):
    backend = str(spec.get("backend", ""))
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend={backend!r}; expected {sorted(SUPPORTED_BACKENDS)}"
        )
    if backend in {"hf_t5", "hf_encoder"}:
        return HFEncoderAdapter(spec, device)
    return ESM3Adapter(spec, device)

def output_has_embeddings(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_pickle(path)
    except Exception:
        return False
    return "enzyme_vector" in df.columns and df["enzyme_vector"].notna().sum() > 0

def build_model_spec(args: argparse.Namespace) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "backend": args.backend,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "validate_token_count": args.validate_token_count,
    }
    optional = {
        "local_path": args.local_path,
        "revision": args.revision,
        "model_class": args.model_class,
        "tokenizer_class": args.tokenizer_class,
        "input_mode": args.input_mode,
        "prefix": args.prefix,
        "pretrained_name": args.pretrained_name,
    }
    spec.update({key: value for key, value in optional.items() if value is not None})
    if args.trust_remote_code:
        spec["trust_remote_code"] = True
    return spec

def normalize_sequence(sequence: Any, max_len: int) -> Optional[str]:
    if not isinstance(sequence, str) or not sequence.strip():
        return None
    normalized, _, _ = preprocess_sequence(sequence, max_len)
    return normalized

def collect_pending_frames(
    input_pkls: Sequence[str],
    output_dir: Path,
) -> List[Tuple[str, Path, pd.DataFrame]]:
    pending: List[Tuple[str, Path, pd.DataFrame]] = []
    for pkl_path in input_pkls:
        source = Path(pkl_path)
        if not source.exists():
            raise FileNotFoundError(f"Input pickle not found: {source}")

        output_path = output_dir / source.name
        if output_has_embeddings(output_path):
            logging.info("Skipping existing enzyme embeddings: %s", output_path)
            continue

        df = pd.read_pickle(source)
        if "sequence" not in df.columns:
            raise ValueError(f"Missing 'sequence' column: {source}")
        pending.append((str(source), output_path, df))
    return pending

def generate_embedding_map(
    adapter,
    normalized_sequences: Sequence[str],
    batch_size: int,
    token_budget: int,
    max_batch_size: int,
) -> Dict[str, np.ndarray]:
    records = [
        (sequence_sha256(sequence), sequence)
        for sequence in sorted(set(normalized_sequences), key=len)
    ]
    batches = make_batches(
        records,
        batch_size=batch_size,
        token_budget=token_budget,
        max_batch_size=max_batch_size,
    )

    logging.info(
        "Encoding %d unique sequences in %d batches",
        len(records),
        len(batches),
    )
    embeddings: Dict[str, np.ndarray] = {}
    for batch in tqdm(batches, desc="External PLM embeddings"):
        batch_embeddings = adapter.embed(batch).detach().cpu().float().numpy()
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != len(batch):
            raise RuntimeError(
                f"Unexpected embedding shape {batch_embeddings.shape} "
                f"for batch size {len(batch)}"
            )
        for row, (_, sequence) in enumerate(batch):
            embeddings[sequence] = batch_embeddings[row].astype(np.float32, copy=False)
    return embeddings

def save_frame_with_embeddings(
    df: pd.DataFrame,
    output_path: Path,
    embedding_map: Mapping[str, np.ndarray],
    max_len: int,
) -> Tuple[int, int]:
    normalized = df["sequence"].map(lambda value: normalize_sequence(value, max_len))
    out = df.copy()
    out["enzyme_vector"] = normalized.map(embedding_map)
    missing = int(out["enzyme_vector"].isna().sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    out.to_pickle(tmp_path)
    os.replace(tmp_path, output_path)
    logging.info("Saved %s (rows=%d, missing=%d)", output_path, len(out), missing)
    return len(out), missing

def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    adapter_info,
    files: Sequence[Mapping[str, Any]],
    unique_sequences: int,
) -> None:
    metadata = {
        "model_name": args.model_name,
        "model_id": adapter_info.model_id,
        "model_source": adapter_info.source,
        "model_revision": adapter_info.revision,
        "backend": adapter_info.backend,
        "pooling": adapter_info.pooling,
        "inference_dtype": adapter_info.dtype,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "max_batch_size": args.max_batch_size,
        "unique_sequences": unique_sequences,
        "files": list(files),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "format_version": 1,
    }
    with open(output_dir / "embedding_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate external PLM enzyme embeddings for ESP pickle files"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--backend",
        required=True,
        choices=sorted(SUPPORTED_BACKENDS),
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-class", default=None)
    parser.add_argument("--tokenizer-class", default=None)
    parser.add_argument("--input-mode", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--pretrained-name", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--no-validate-token-count",
        dest="validate_token_count",
        action="store_false",
    )
    parser.set_defaults(validate_token_count=True)

    parser.add_argument("--input-pkls", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.max_batch_size <= 0:
        raise ValueError("batch-size and max-batch-size must be positive")
    if args.token_budget < 0:
        raise ValueError("token-budget must be zero or positive")
    if args.max_len <= 0:
        raise ValueError("max-len must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = collect_pending_frames(args.input_pkls, output_dir)
    if not pending:
        logging.info("All requested ESP embedding files already exist")
        return 0

    normalized_sequences: List[str] = []
    for _, _, df in pending:
        normalized_sequences.extend(
            normalized
            for normalized in (
                normalize_sequence(value, args.max_len) for value in df["sequence"]
            )
            if normalized is not None
        )
    unique_sequences = sorted(set(normalized_sequences), key=len)
    if not unique_sequences:
        raise ValueError("No valid protein sequences found in pending ESP pickle files")

    adapter = build_adapter(build_model_spec(args), args.device)
    file_summaries: List[Dict[str, Any]] = []
    try:
        embedding_map = generate_embedding_map(
            adapter=adapter,
            normalized_sequences=unique_sequences,
            batch_size=args.batch_size,
            token_budget=args.token_budget,
            max_batch_size=args.max_batch_size,
        )
        for source, output_path, df in pending:
            rows, missing = save_frame_with_embeddings(
                df=df,
                output_path=output_path,
                embedding_map=embedding_map,
                max_len=args.max_len,
            )
            file_summaries.append(
                {
                    "source": source,
                    "output": str(output_path),
                    "rows": rows,
                    "missing_embeddings": missing,
                }
            )
        write_metadata(
            output_dir=output_dir,
            args=args,
            adapter_info=adapter.info,
            files=file_summaries,
            unique_sequences=len(unique_sequences),
        )
    finally:
        del adapter
        gc.collect()
        torch = import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
