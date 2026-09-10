"""
ESM 系列编码器 (ESM-2 全尺寸 + ESM-1b)

共用同一个类，通过 model_name 区分。
所有变体通过 esm.pretrained 加载，batch_converter tokenize。

ESM-2 尺寸/维度参考 (官方):
    esm2_t6_8M    :  6 layers, dim=320
    esm2_t12_35M  : 12 layers, dim=480
    esm2_t30_150M : 30 layers, dim=640
    esm2_650m     : 33 layers, dim=1280   (alias: esm2_t33_650M)
    esm2_t36_3B   : 36 layers, dim=2560
    esm2_t48_15B  : 48 layers, dim=5120
    esm1b         : 33 layers, dim=1280
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from .base import BaseEnzymeEncoder

_ESM_REGISTRY = {

    "esm1b":               "esm1b_t33_650M_UR50S",

    "esm2_t6_8M":          "esm2_t6_8M_UR50D",
    "esm2_8m":             "esm2_t6_8M_UR50D",

    "esm2_t12_35M":        "esm2_t12_35M_UR50D",
    "esm2_35m":            "esm2_t12_35M_UR50D",

    "esm2_t30_150M":       "esm2_t30_150M_UR50D",
    "esm2_150m":           "esm2_t30_150M_UR50D",

    "esm2_650m":           "esm2_t33_650M_UR50D",
    "esm2_t33_650M":       "esm2_t33_650M_UR50D",

    "esm2_t36_3B":         "esm2_t36_3B_UR50D",
    "esm2_3b":             "esm2_t36_3B_UR50D",

    "esm2_t48_15B":        "esm2_t48_15B_UR50D",
    "esm2_15b":            "esm2_t48_15B_UR50D",
}

class ESMEncoder(BaseEnzymeEncoder):
    """
    ESM-2 / ESM-1b 编码器

    加载流程: pretrained → (可选) CPT checkpoint → freeze
    LoRA 注入在外部由 model.py 统一处理。
    """

    def __init__(self, model_type: str = "esm2_650m"):
        super().__init__()
        if model_type not in _ESM_REGISTRY:
            raise ValueError(
                f"Unknown ESM type: {model_type!r}. "
                f"Choose from {sorted(_ESM_REGISTRY.keys())}"
            )

        self.model_type = model_type
        esm_name = _ESM_REGISTRY[model_type]

        from esm import pretrained
        load_fn = getattr(pretrained, f"load_model_and_alphabet_{esm_name}", None)
        if load_fn is None:

            model, alphabet = pretrained.load_model_and_alphabet(esm_name)
        else:
            model, alphabet = load_fn()

        self.esm = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter(truncation_seq_length=1022)
        self._num_layers = model.num_layers

        if hasattr(model, 'embed_dim'):
            self._hidden_dim = model.embed_dim
        elif hasattr(model, 'args') and hasattr(model.args, 'embed_dim'):
            self._hidden_dim = model.args.embed_dim
        else:
            raise AttributeError(
                f"Cannot determine embed_dim for {type(model).__name__}. "
                f"Expected model.embed_dim (ESM-2) or model.args.embed_dim (ESM-1b)."
            )

        self.padding_idx = alphabet.padding_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.mask_idx = alphabet.mask_idx
        self.unk_idx = alphabet.unk_idx

        if model_type == "esm1b":
            self._disable_torch_fast_path()

        print(f"[ESMEncoder] Loaded {esm_name} (alias={model_type}): "
              f"{self._num_layers} layers, dim={self._hidden_dim}")

    def _disable_torch_fast_path(self):
        """
        递归找到所有 MultiheadAttention 模块, 把 enable_torch_version 设为 False。
        这会强制走 MultiheadAttention.forward 中的标准实现分支, 调用 q_proj(x)/k_proj(x)/v_proj(x),
        从而让 LoRA adapter 正常参与计算。
        """
        count = 0
        for module in self.esm.modules():
            if hasattr(module, 'enable_torch_version'):
                module.enable_torch_version = False
                count += 1
        print(f"[ESMEncoder] Disabled torch fast path in {count} attention modules "
              f"(required for LoRA compatibility)")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def tokenize(self, sequences: List[Tuple[str, str]]) -> torch.Tensor:
        """
        Args:
            sequences: [(id, seq), ...]
        Returns:
            tokens: [B, L]
        """
        _, _, tokens = self.batch_converter(sequences)

        x_idx = self.alphabet.tok_to_idx.get("X", self.unk_idx)
        if (tokens == x_idx).any():
            tokens[tokens == x_idx] = self.mask_idx
        return tokens

    def forward(self, tokens: torch.Tensor,
                return_per_residue: bool = False) -> Dict[str, torch.Tensor]:
        results = self.esm(
            tokens,
            repr_layers=[self._num_layers],
            return_contacts=False
        )
        hidden = results["representations"][self._num_layers]

        residue_mask = (
            (tokens != self.padding_idx) &
            (tokens != self.cls_idx) &
            (tokens != self.eos_idx)
        )
        mask_float = residue_mask.unsqueeze(-1).float()

        h = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)

        out = {'h': h}
        if return_per_residue:
            out['per_residue'] = hidden
            out['residue_mask'] = residue_mask
        return out