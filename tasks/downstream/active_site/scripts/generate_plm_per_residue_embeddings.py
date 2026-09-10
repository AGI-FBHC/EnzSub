#!/usr/bin/env python3
"""Generate residue-level embeddings for external PLM baselines in AS.

The output contract is the existing AS contract: one file per protein,
``{repr_layer: Tensor[L, D]}``.  Metadata is stored under ``_meta`` and is
ignored by the existing EnzymeDataset.  This script is deliberately
self-contained and does not import the EC/ESP generators or model/Sub.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_BACKENDS = frozenset({"hf_t5", "hf_encoder", "esm3"})
HF_REPRESENTATION = "last_hidden_state_residue_tokens_no_special_tokens"
ESM3_REPRESENTATION = "esm3_final_layernorm_residue_tokens_no_special_tokens"
FORMAT_VERSION = 1

def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in the PLM inference environment") from exc
    return torch

def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    with open(path, encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return value

def resolve_path(value: str, config_dir: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    return str(path if path.is_absolute() else (config_dir / path).resolve())

def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    sequences: List[str] = []
    seen: Dict[str, str] = {}
    current_id: Optional[str] = None
    chunks: List[str] = []

    def append_record(seq_id: Optional[str], sequence_chunks: Sequence[str]) -> None:
        if seq_id is None:
            return
        sequence = "".join(sequence_chunks)
        if seq_id in seen:
            if seen[seq_id] != sequence:
                raise ValueError(f"Duplicate FASTA id {seq_id!r} has different sequences")
            return
        seen[seq_id] = sequence
        ids.append(seq_id)
        sequences.append(sequence)

    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record(current_id, chunks)
                current_id = line[1:].split()[0]
                if not current_id:
                    raise ValueError(f"Empty FASTA identifier in {path}")
                chunks = []
            else:
                if current_id is None:
                    raise ValueError(f"Sequence appears before the first header in {path}")
                chunks.append(line)
    append_record(current_id, chunks)
    if not ids:
        raise ValueError(f"No FASTA records found in {path}")
    return ids, sequences

def read_label_lengths(path: str) -> Dict[str, int]:
    """Read the existing AS label CSV-like file without changing its format."""
    lengths: Dict[str, int] = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                raise ValueError(f"Malformed label row {path}:{line_number}")
            seq_id = parts[0].strip()
            if not seq_id:
                raise ValueError(f"Empty protein id in {path}:{line_number}")
            label_text = parts[2].rstrip(",").strip()
            if (
                line_number == 1
                and seq_id.lower() in {"id", "name", "protein_id"}
                and label_text.lower() in {"label", "labels"}
            ):
                continue
            try:
                labels = ast.literal_eval(label_text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Invalid label list in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(labels, (list, tuple)):
                raise ValueError(f"Labels are not a list in {path}:{line_number}")
            if seq_id in lengths:
                raise ValueError(f"Duplicate protein id {seq_id!r} in {path}")
            lengths[seq_id] = len(labels)
    if not lengths:
        raise ValueError(f"No label rows found in {path}")
    return lengths

def validate_fasta_label_alignment(fasta_path: str, label_path: str) -> Dict[str, Any]:
    """Reject silent residue/label shifts before any expensive model loading."""
    ids, raw_sequences = read_fasta(fasta_path)
    label_lengths = read_label_lengths(label_path)
    fasta_ids = set(ids)
    label_ids = set(label_lengths)
    missing_labels = sorted(fasta_ids - label_ids)
    extra_labels = sorted(label_ids - fasta_ids)
    length_mismatches = []
    for seq_id, raw_sequence in zip(ids, raw_sequences):
        compact_length = len(re.sub(r"\s+", "", raw_sequence))
        if seq_id in label_lengths and label_lengths[seq_id] != compact_length:
            length_mismatches.append(
                {
                    "id": seq_id,
                    "sequence_length": compact_length,
                    "label_length": label_lengths[seq_id],
                }
            )
    if missing_labels or extra_labels or length_mismatches:
        raise ValueError(
            "FASTA/label alignment failed: "
            f"missing_labels={missing_labels[:5]}, extra_labels={extra_labels[:5]}, "
            f"length_mismatches={length_mismatches[:5]}"
        )
    return {
        "fasta": fasta_path,
        "labels": label_path,
        "records": len(ids),
        "aligned": True,
    }

def preprocess_sequence(sequence: str, max_len: int) -> Tuple[str, int, int]:
    compact = re.sub(r"\s+", "", sequence).upper()
    original_len = len(compact)
    cleaned_chars = [aa if aa in STANDARD_AAS else "X" for aa in compact]
    replacement_count = sum(a != b for a, b in zip(compact, cleaned_chars))
    cleaned = "".join(cleaned_chars)[:max_len]
    if not cleaned:
        raise ValueError("Encountered an empty sequence after preprocessing")
    return cleaned, original_len, replacement_count

def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()

def make_batches(
    records: Sequence[Tuple[str, str]],
    batch_size: int,
    token_budget: int,
    max_batch_size: int,
) -> List[List[Tuple[str, str]]]:
    if batch_size <= 0 or max_batch_size <= 0:
        raise ValueError("batch_size and max_batch_size must be positive")
    ordered = sorted(records, key=lambda item: len(item[1]))
    if token_budget <= 0:
        return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]

    batches: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    current_max_len = 0
    for record in ordered:
        candidate_max = max(current_max_len, len(record[1]))
        exceeds_budget = bool(current) and (len(current) + 1) * candidate_max > token_budget
        reaches_cap = len(current) >= max_batch_size
        if exceeds_budget or reaches_cap:
            batches.append(current)
            current = [record]
            current_max_len = len(record[1])
        else:
            current.append(record)
            current_max_len = candidate_max
    if current:
        batches.append(current)
    return batches

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
        raise ValueError(f"Unsupported dtype {name!r}")
    dtype = mapping[normalized]
    if device.startswith("cpu") and dtype in (torch.float16, torch.bfloat16):
        raise ValueError(f"dtype={name} is not supported by this script on CPU")
    return dtype

def autocast_context(torch, device: str, dtype):
    if not device.startswith("cuda") or dtype not in (torch.float16, torch.bfloat16):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)

def atomic_torch_save(torch, payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, output_path)

def atomic_json_save(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
    os.replace(temporary, output_path)

def special_tokens_mask(tokenizer, input_ids, encoded: MutableMapping[str, Any], torch):
    mask = encoded.pop("special_tokens_mask", None)
    if mask is not None:
        return mask.bool()
    rows = [
        tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
        for row in input_ids.cpu()
    ]
    return torch.tensor(rows, dtype=torch.bool, device=input_ids.device)

def remove_prefix_from_mask(input_ids, residue_mask, prefix_ids: Sequence[int]) -> None:
    if not prefix_ids:
        return
    prefix = list(prefix_ids)
    for row_index in range(input_ids.shape[0]):
        ids = input_ids[row_index].tolist()
        found_at: Optional[int] = None
        for start in range(0, min(3, max(1, len(ids) - len(prefix) + 1))):
            if ids[start : start + len(prefix)] == prefix:
                found_at = start
                break
        if found_at is None:
            raise RuntimeError("Could not locate the configured Ankh3 prefix tokens")
        residue_mask[row_index, found_at : found_at + len(prefix)] = False

@dataclass(frozen=True)
class AdapterInfo:
    backend: str
    model_id: str
    source: str
    revision: Optional[str]
    dtype: str
    representation: str

class HFEncoderAdapter:
    def __init__(self, spec: Mapping[str, Any], device: str):
        torch = import_torch()
        try:
            from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and sentencepiece are required") from exc

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

        model_kwargs = dict(common_kwargs)
        if self.load_dtype is not None:
            model_kwargs["torch_dtype"] = self.load_dtype
        model_kind = str(spec.get("model_class", "auto"))
        if model_kind == "t5_encoder":
            self.model = T5EncoderModel.from_pretrained(self.source, **model_kwargs)
        elif model_kind == "auto":
            self.model = AutoModel.from_pretrained(self.source, **model_kwargs)
        else:
            raise ValueError(f"Unknown model_class={model_kind!r}")
        self.model.to(device).eval()

        self.resolved_revision = self.revision or getattr(
            getattr(self.model, "config", None), "_commit_hash", None
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
            representation=HF_REPRESENTATION,
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

        observed = residue_mask.sum(dim=1).cpu().tolist()
        expected = [len(sequence) for sequence in sequences]
        if self.validate_token_count and observed != expected:
            details = list(zip([sid for sid, _ in records], expected, observed))[:5]
            raise RuntimeError(
                "Residue-token count mismatch after excluding special tokens: "
                f"{details}"
            )

        allowed_inputs = set(getattr(self.tokenizer, "model_input_names", []))
        model_inputs = {key: value for key, value in encoded.items() if key in allowed_inputs}
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
            raise RuntimeError(f"Model {self.model_id} returned no hidden states")
        return [
            hidden[row_index, residue_mask[row_index]].float().cpu()
            for row_index in range(len(records))
        ]

class ESM3Adapter:
    def __init__(self, spec: Mapping[str, Any], device: str):
        torch = import_torch()
        try:
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise RuntimeError("Biohub esm is required for the ESM3 baseline") from exc

        self.torch = torch
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
        self.device = device
        self.spec = dict(spec)
        self.model_id = str(spec["model_id"])
        self.source = str(spec.get("pretrained_name") or self.model_id)
        self.revision = spec.get("revision")
        self.model = ESM3.from_pretrained(self.source, device=torch.device(device))
        self.model.eval()
        self.final_norm = getattr(getattr(self.model, "transformer", None), "norm", None)
        if self.final_norm is None:
            raise RuntimeError("Installed ESM3 does not expose transformer.norm")

    @property
    def info(self) -> AdapterInfo:
        return AdapterInfo(
            backend="esm3",
            model_id=self.model_id,
            source=self.source,
            revision=self.revision,
            dtype="bf16_autocast_by_official_api",
            representation=ESM3_REPRESENTATION,
        )

    def embed(self, records: Sequence[Tuple[str, str]]):
        torch = self.torch
        residue_embeddings = []
        for seq_id, sequence in records:
            protein_tensor = self.model.encode(self.ESMProtein(sequence=sequence))
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
                    raise RuntimeError(f"Unexpected ESM3 shape: {tuple(embedding.shape)}")
                embedding = embedding[0]
            with torch.inference_mode():
                norm_weight = getattr(self.final_norm, "weight", None)
                norm_bias = getattr(self.final_norm, "bias", None)
                embedding = torch.nn.functional.layer_norm(
                    embedding.float(),
                    self.final_norm.normalized_shape,
                    weight=None if norm_weight is None else norm_weight.float(),
                    bias=None if norm_bias is None else norm_bias.float(),
                    eps=self.final_norm.eps,
                )
            expected_tokens = len(sequence) + 2
            if embedding.shape[0] != expected_tokens:
                raise RuntimeError(
                    f"ESM3 token count mismatch for {seq_id}: "
                    f"tokens={embedding.shape[0]}, expected={expected_tokens}"
                )
            residue_embeddings.append(embedding[1 : len(sequence) + 1].float().cpu())
        return residue_embeddings

def build_adapter(spec: Mapping[str, Any], device: str):
    backend = str(spec.get("backend", ""))
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend={backend!r}")
    if backend in {"hf_t5", "hf_encoder"}:
        return HFEncoderAdapter(spec, device)
    return ESM3Adapter(spec, device)

def representation_for_backend(backend: str) -> str:
    return ESM3_REPRESENTATION if backend == "esm3" else HF_REPRESENTATION

def expected_contract(spec: Mapping[str, Any], sequence: str) -> Dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "model_id": str(spec["model_id"]),
        "backend": str(spec["backend"]),
        "representation": representation_for_backend(str(spec["backend"])),
        "repr_layer": int(spec.get("repr_layer", -1)),
        "embed_dim": int(spec["embed_dim"]),
        "max_len": int(spec.get("max_len", 1022)),
        "sequence_sha256": sequence_sha256(sequence),
        "residue_count": len(sequence),
    }

def validate_artifact(
    path: Path,
    spec: Mapping[str, Any],
    sequence: str,
    torch,
) -> Tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return False, f"unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return False, "payload is not a dictionary"
    contract = expected_contract(spec, sequence)
    repr_layer = contract["repr_layer"]
    tensor = payload.get(repr_layer)
    if not isinstance(tensor, torch.Tensor):
        return False, f"missing tensor at repr_layer={repr_layer}"
    if tensor.ndim != 2:
        return False, f"expected rank 2, got shape={tuple(tensor.shape)}"
    expected_shape = (contract["residue_count"], contract["embed_dim"])
    if tuple(tensor.shape) != expected_shape:
        return False, f"shape={tuple(tensor.shape)}, expected={expected_shape}"
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        return False, f"expected CPU float32, got {tensor.device}/{tensor.dtype}"
    if not torch.isfinite(tensor).all():
        return False, "contains NaN or Inf"
    metadata = payload.get("_meta")
    if not isinstance(metadata, dict):
        return False, "missing _meta"
    for key, expected in contract.items():
        if metadata.get(key) != expected:
            return False, f"metadata {key}={metadata.get(key)!r}, expected={expected!r}"
    return True, "ok"

def artifact_payload(
    tensor,
    spec: Mapping[str, Any],
    sequence: str,
    adapter_info: AdapterInfo,
) -> Dict[Any, Any]:
    contract = expected_contract(spec, sequence)
    metadata = dict(contract)
    metadata.update(
        {
            "model_source": adapter_info.source,
            "model_revision": adapter_info.revision,
            "inference_dtype": adapter_info.dtype,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {contract["repr_layer"]: tensor.float().cpu(), "_meta": metadata}

def normalized_records(fasta_path: str, max_len: int):
    ids, raw_sequences = read_fasta(fasta_path)
    records: List[Tuple[str, str]] = []
    truncated = 0
    replacements = 0
    for seq_id, raw_sequence in zip(ids, raw_sequences):
        sequence, original_len, replaced = preprocess_sequence(raw_sequence, max_len)
        records.append((seq_id, sequence))
        truncated += int(original_len > max_len)
        replacements += replaced
    return records, truncated, replacements

def validate_split(
    model_name: str,
    spec: Mapping[str, Any],
    split: str,
    fasta_path: str,
    output_dir: Path,
    limit: int = 0,
) -> Dict[str, Any]:
    torch = import_torch()
    records, truncated, replacements = normalized_records(
        fasta_path, int(spec.get("max_len", 1022))
    )
    selected = records[:limit] if limit > 0 else records
    failures: List[Tuple[str, str]] = []
    for seq_id, sequence in selected:
        ok, reason = validate_artifact(output_dir / f"{seq_id}.pt", spec, sequence, torch)
        if not ok:
            failures.append((seq_id, reason))
    result = {
        "model": model_name,
        "split": split,
        "fasta": fasta_path,
        "output_dir": str(output_dir),
        "expected_records": len(records),
        "checked_records": len(selected),
        "valid_records": len(selected) - len(failures),
        "complete": limit == 0 and not failures,
        "selected_complete": not failures,
        "truncated_records": truncated,
        "nonstandard_residue_count": replacements,
        "failures": [{"id": seq_id, "reason": reason} for seq_id, reason in failures[:20]],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_save(result, output_dir / "_manifest.json")
    return result

def generate_split(
    adapter,
    model_name: str,
    spec: Mapping[str, Any],
    split: str,
    fasta_path: str,
    output_dir: Path,
    limit: int,
) -> Dict[str, Any]:
    torch = import_torch()
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required") from exc

    records, truncated, replacements = normalized_records(
        fasta_path, int(spec.get("max_len", 1022))
    )
    selected = records[:limit] if limit > 0 else records
    pending: List[Tuple[str, str]] = []
    stale_count = 0
    for seq_id, sequence in selected:
        ok, reason = validate_artifact(output_dir / f"{seq_id}.pt", spec, sequence, torch)
        if not ok:
            pending.append((seq_id, sequence))
            stale_count += int(reason != "missing")

    batches = make_batches(
        pending,
        int(spec.get("batch_size", 4)),
        int(spec.get("token_budget", 0)),
        int(spec.get("max_batch_size", 16)),
    )
    print(
        f"[{model_name}:{split}] total={len(records)} selected={len(selected)} "
        f"pending={len(pending)} stale={stale_count} batches={len(batches)} "
        f"truncated={truncated} replacements={replacements}"
    )

    for batch in tqdm(batches, desc=f"{model_name}:{split}"):
        tensors = adapter.embed(batch)
        if len(tensors) != len(batch):
            raise RuntimeError("Adapter returned a different number of embeddings")
        for tensor, (seq_id, sequence) in zip(tensors, batch):
            tensor = tensor.detach().cpu().float()
            expected_shape = (len(sequence), int(spec["embed_dim"]))
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected embedding shape for {seq_id}: "
                    f"{tuple(tensor.shape)}, expected={expected_shape}"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite embedding for {seq_id}")
            atomic_torch_save(
                torch,
                artifact_payload(tensor, spec, sequence, adapter.info),
                output_dir / f"{seq_id}.pt",
            )

    result = validate_split(
        model_name, spec, split, fasta_path, output_dir, limit=limit
    )
    if not result["selected_complete"]:
        raise RuntimeError(
            f"Artifact validation failed for {model_name}/{split}: {result['failures'][:5]}"
        )
    return result

def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config.get("models"), dict) or not config["models"]:
        raise ValueError("Config must define a non-empty models mapping")
    if not isinstance(config.get("data"), dict):
        raise ValueError("Config must define FASTA and label paths")
    for key in ("train_fasta", "test_fasta", "train_labels", "test_labels"):
        if key not in config["data"]:
            raise ValueError(f"Config is missing data.{key}")
    if "emb_root" not in config:
        raise ValueError("Config is missing emb_root")
    for name, spec in config["models"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"models.{name} must be a mapping")
        if spec.get("source") != "external_plm":
            continue
        for key in ("backend", "model_id", "embed_dim", "encoder_type"):
            if key not in spec:
                raise ValueError(f"models.{name} is missing {key}")
        if int(spec.get("repr_layer", -1)) != -1:
            raise ValueError(f"models.{name}.repr_layer must be -1")

def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(str(config_path))
    validate_config(config)
    config_dir = config_path.parent

    external_models = {
        name: dict(spec)
        for name, spec in config["models"].items()
        if isinstance(spec, dict) and spec.get("source") == "external_plm"
    }
    if args.list_models:
        for name, spec in external_models.items():
            print(f"{name}\t{spec['backend']}\t{spec['model_id']}")
        return 0

    selected = args.models or list(external_models)
    unknown = sorted(set(selected) - set(external_models))
    if unknown:
        raise ValueError(f"Unknown external PLM model keys: {unknown}")
    if not selected:
        raise ValueError("No external PLM models selected")

    data = config["data"]
    fasta_paths = {
        "train": resolve_path(data["train_fasta"], config_dir),
        "test": resolve_path(data["test_fasta"], config_dir),
    }
    label_paths = {
        "train": resolve_path(data["train_labels"], config_dir),
        "test": resolve_path(data["test_labels"], config_dir),
    }
    for split, path in fasta_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} FASTA not found: {path}")
        if not os.path.exists(label_paths[split]):
            raise FileNotFoundError(f"{split} labels not found: {label_paths[split]}")
    alignment = {
        split: validate_fasta_label_alignment(fasta_paths[split], label_paths[split])
        for split in ("train", "test")
    }
    emb_root = Path(resolve_path(config["emb_root"], config_dir))
    device = str(args.device or config.get("device", "cuda:0"))

    summary: Dict[str, Any] = {
        "config": str(config_path),
        "device": device,
        "limit": args.limit,
        "validate_only": args.validate_only,
        "data_alignment": alignment,
        "models": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    for model_name in selected:
        spec = external_models[model_name]
        spec.setdefault("max_len", int(config.get("max_len", 1022)))
        spec.setdefault("repr_layer", -1)
        model_results = []
        adapter = None if args.validate_only else build_adapter(spec, device)
        try:
            for split, fasta_path in fasta_paths.items():
                output_dir = emb_root / model_name / split
                if args.validate_only:
                    result = validate_split(
                        model_name, spec, split, fasta_path, output_dir, limit=args.limit
                    )
                else:
                    result = generate_split(
                        adapter,
                        model_name,
                        spec,
                        split,
                        fasta_path,
                        output_dir,
                        limit=args.limit,
                    )
                model_results.append(result)
        finally:
            if adapter is not None:
                del adapter
        summary["models"][model_name] = model_results

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["complete"] = all(
        split_result["complete"]
        for model_results in summary["models"].values()
        for split_result in model_results
    )
    summary["selected_complete"] = all(
        split_result["selected_complete"]
        for model_results in summary["models"].values()
        for split_result in model_results
    )
    atomic_json_save(summary, emb_root / "plm_embedding_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["selected_complete"]:
        return 1
    if args.validate_only and args.limit == 0 and not summary["complete"]:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
