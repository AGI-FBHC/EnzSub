"""EnzSub-backed Seq2Topt model components.

This module intentionally leaves the original Seq2Topt implementation intact.
It replaces the frozen ESM-2 feature extractor with EnzSub's downstream encoder,
projects per-residue representations to the original 320-dimensional head, and
adds masking for BOS/EOS/padding positions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RDBlock

TokenBatch = Union[torch.Tensor, Dict[str, torch.Tensor]]

def autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context across old and new PyTorch AMP APIs."""
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

def _ensure_enzsub_importable(enzsub_root: Union[str, Path]) -> Path:
    """Add the directory containing the ``sub`` package to ``sys.path``."""
    root = Path(enzsub_root).expanduser().resolve()
    if not ((root / "sub" / "model.py").is_file() or (root / "enzsub" / "sub" / "model.py").is_file()):
        raise FileNotFoundError(
            f"EnzSub root must contain sub/model.py or enzsub/sub/model.py, but got: {root}"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

def _move_tokens(tokens: TokenBatch, device: torch.device) -> TokenBatch:
    if isinstance(tokens, dict):
        return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
    return tokens.to(device, non_blocking=True)

def normalize_sequence(sequence: str, max_length: int) -> str:
    """Normalize a protein sequence using the same unknown-residue convention as EnzSub."""
    sequence = re.sub(r"\s+", "", str(sequence).upper())
    sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", sequence)
    sequence = sequence[:max_length]
    if not sequence:
        raise ValueError("Protein sequences must contain at least one residue.")
    return sequence

class FeatureAdapter(nn.Module):
    """Project EnzSub residue features into the original Seq2Topt head dimension."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.input_dim == self.output_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.adapter(features)

class MaskedMultiAttModel(nn.Module):
    """Seq2Topt's original prediction head with residue-mask support.

    Module names match ``MultiAttModel`` where possible, making the architecture
    easy to compare with the original implementation.
    """

    def __init__(self, dim: int, window: int, n_head: int, n_RD: int):
        super().__init__()
        self.dim = int(dim)
        self.window = int(window)
        self.n_RD = int(n_RD)
        self.n_head = int(n_head)
        kernel_size = 2 * self.window + 1
        self.cnn_v = nn.Conv1d(
            self.dim, self.dim, kernel_size=kernel_size, padding=self.window
        )
        self.W_cnns = nn.ModuleList(
            [
                nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size=kernel_size,
                    padding=self.window,
                )
                for _ in range(self.n_head)
            ]
        )
        output_dim = 2 * self.n_head * self.dim
        self.RDs = nn.ModuleList([RDBlock(output_dim) for _ in range(self.n_RD)])
        self.output = nn.Linear(output_dim, 1)

    def forward(
        self,
        embedding: torch.Tensor,
        residue_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        if embedding.ndim != 3:
            raise ValueError(
                f"embedding must have shape [B, D, L], got {tuple(embedding.shape)}"
            )
        if residue_mask.ndim != 2:
            raise ValueError(
                f"residue_mask must have shape [B, L], got {tuple(residue_mask.shape)}"
            )
        if embedding.shape[0] != residue_mask.shape[0] or embedding.shape[2] != residue_mask.shape[1]:
            raise ValueError(
                "embedding and residue_mask batch/length dimensions do not match: "
                f"{tuple(embedding.shape)} vs {tuple(residue_mask.shape)}"
            )

        residue_mask = residue_mask.bool()
        if not torch.all(residue_mask.any(dim=1)):
            raise ValueError("Every sample must contain at least one valid residue.")

        valid = residue_mask.unsqueeze(1)
        embedding = embedding.masked_fill(~valid, 0.0)
        values = self.cnn_v(embedding)

        sum_features: List[torch.Tensor] = []
        max_features: List[torch.Tensor] = []
        attention_heads: List[torch.Tensor] = []
        for weight_cnn in self.W_cnns:
            logits = weight_cnn(embedding)
            # Under autocast, logits may be float16 even when the input tensor
            # is float32.  Derive the finite mask value from the actual logits
            # dtype; using float32.min here overflows when it is cast to half.
            mask_fill_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~valid, mask_fill_value)
            weights = F.softmax(logits, dim=-1)
            weighted_values = values * weights

            sum_features.append(torch.sum(weighted_values, dim=-1))
            max_features.append(
                weighted_values.masked_fill(~valid, mask_fill_value).amax(dim=-1)
            )
            if return_attention:
                attention_heads.append(weights.mean(dim=1))

        features = torch.cat(sum_features + max_features, dim=1)
        for residual_dense in self.RDs:
            features = residual_dense(features)
        prediction = self.output(features)

        if not return_attention:
            return prediction

        attention = torch.stack(attention_heads, dim=1).mean(dim=1)
        attention = attention.masked_fill(~residue_mask, 0.0)
        return prediction, attention

class Seq2ToptFeatureModel(nn.Module):
    """Trainable adapter and masked Seq2Topt prediction head.

    Keeping this component independent from EnzSub allows frozen per-residue
    features to be cached once and reused across all 30 training epochs.
    """

    def __init__(
        self,
        encoder_dim: int,
        head_dim: int = 320,
        window: int = 3,
        n_head: int = 4,
        n_RD: int = 4,
    ):
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.head_dim = int(head_dim)
        self.adapter = FeatureAdapter(self.encoder_dim, self.head_dim)
        self.predictor = MaskedMultiAttModel(
            dim=self.head_dim,
            window=window,
            n_head=n_head,
            n_RD=n_RD,
        )

    @property
    def task_config(self) -> Dict[str, int]:
        return {
            "encoder_dim": self.encoder_dim,
            "head_dim": self.head_dim,
            "window": self.predictor.window,
            "n_head": self.predictor.n_head,
            "n_RD": self.predictor.n_RD,
        }

    def forward(
        self,
        residue_features: torch.Tensor,
        residue_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        adapted = self.adapter(residue_features)
        return self.predictor(
            adapted.transpose(1, 2),
            residue_mask,
            return_attention=return_attention,
        )

class EnzSubSeq2Topt(nn.Module):
    """Frozen EnzSub per-residue encoder plus a masked Seq2Topt head."""

    VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")

    def __init__(
        self,
        enzsub_root: Union[str, Path],
        encoder_type: str = "esm2_650m",
        model_mode: str = "cpt_sub",
        enzsub_checkpoint: Optional[Union[str, Path]] = None,
        head_dim: int = 320,
        window: int = 3,
        n_head: int = 4,
        n_RD: int = 4,
        max_seq_length: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.enzsub_root = _ensure_enzsub_importable(enzsub_root)
        if model_mode not in self.VALID_MODES:
            raise ValueError(f"model_mode must be one of {self.VALID_MODES}, got {model_mode!r}")
        if model_mode != "base" and enzsub_checkpoint is None:
            raise ValueError(f"model_mode={model_mode!r} requires --enzsub-checkpoint")

        from enzsub.sub.model import EnzSubModelForDownstream

        checkpoint_path = None
        if enzsub_checkpoint is not None:
            checkpoint_path = str(Path(enzsub_checkpoint).expanduser().resolve())

        self.protein_encoder = EnzSubModelForDownstream(
            encoder_type=encoder_type,
            model_mode=model_mode,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            device=str(device),
            strict_lora_load=True,
            allow_lora_reverse_detect=True,
        )

        # EnzSub's downstream helper intentionally leaves trained LoRA parameters
        # trainable. This experiment freezes every encoder parameter so that only
        # the representation source changes relative to Seq2Topt.
        for parameter in self.protein_encoder.parameters():
            parameter.requires_grad = False
        self.protein_encoder.eval()

        self.encoder_type = encoder_type
        self.model_mode = model_mode
        self.encoder_dim = int(self.protein_encoder.hidden_dim)
        self.head_dim = int(head_dim)
        encoder_limit = int(self.protein_encoder.max_seq_len)
        requested_limit = encoder_limit if max_seq_length is None else int(max_seq_length)
        if requested_limit <= 0:
            raise ValueError("max_seq_length must be positive")
        self.max_seq_length = min(requested_limit, encoder_limit)

        self.task_model = Seq2ToptFeatureModel(
            encoder_dim=self.encoder_dim,
            head_dim=self.head_dim,
            window=window,
            n_head=n_head,
            n_RD=n_RD,
        )
        self.task_model.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen encoder deterministic even when the task head trains.
        self.protein_encoder.eval()
        return self

    @property
    def task_config(self) -> Dict[str, object]:
        return {
            "encoder_type": self.encoder_type,
            "model_mode": self.model_mode,
            "encoder_dim": self.encoder_dim,
            "head_dim": self.head_dim,
            "window": self.task_model.predictor.window,
            "n_head": self.task_model.predictor.n_head,
            "n_RD": self.task_model.predictor.n_RD,
            "max_seq_length": self.max_seq_length,
        }

    def task_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return only trainable adapter/head weights, excluding the EnzSub checkpoint."""
        return {
            key: value.detach().cpu()
            for key, value in self.task_model.state_dict().items()
        }

    def load_task_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.task_model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Task checkpoint is incompatible with this model. "
                f"Missing task keys={missing}, unexpected keys={unexpected}"
            )

    def _prepare_inputs(
        self, ids: Sequence[object], sequences: Sequence[str]
    ) -> List[Tuple[str, str]]:
        if len(ids) != len(sequences):
            raise ValueError("ids and sequences must have the same length")
        return [
            (str(sample_id), normalize_sequence(sequence, self.max_seq_length))
            for sample_id, sequence in zip(ids, sequences)
        ]

    def encode_residues(
        self, ids: Sequence[object], sequences: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self._prepare_inputs(ids, sequences)
        tokens = self.protein_encoder.tokenize(samples)
        encoder_device = next(self.protein_encoder.parameters()).device
        tokens = _move_tokens(tokens, encoder_device)
        with torch.no_grad():
            outputs = self.protein_encoder(tokens, return_per_residue=True)
        residue_features = outputs.get("per_residue")
        residue_mask = outputs.get("residue_mask")
        if residue_features is None or residue_mask is None:
            raise RuntimeError(
                "EnzSub encoder did not return per_residue and residue_mask outputs."
            )
        return residue_features, residue_mask.bool()

    def forward(
        self,
        ids: Sequence[object],
        sequences: Sequence[str],
        return_attention: bool = False,
    ):
        residue_features, residue_mask = self.encode_residues(ids, sequences)
        return self.task_model(
            residue_features,
            residue_mask,
            return_attention=return_attention,
        )
