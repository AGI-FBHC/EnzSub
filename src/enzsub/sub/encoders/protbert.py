"""
ProtBERT-BFD 编码器

使用 HuggingFace Transformers 加载。
Tokenization: 氨基酸用空格分隔后 AutoTokenizer。
"""

import re
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

from .base import BaseEnzymeEncoder

_PROTBERT_MODEL_NAME = "Rostlab/prot_bert_bfd"

class ProtBERTEncoder(BaseEnzymeEncoder):
    """
    ProtBERT-BFD 编码器

    30 layers, 1024-dim.
    Tokenization 与 ESM 不同: 需要空格分隔每个氨基酸。
    """

    def __init__(self, model_name: str = _PROTBERT_MODEL_NAME,
                 max_length: int = 1024):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        config = self.bert.config
        self._hidden_dim = config.hidden_size
        self._num_layers = config.num_hidden_layers

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        print(f"[ProtBERTEncoder] Loaded {model_name}: "
              f"{self._num_layers} layers, dim={self._hidden_dim}")

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def tokenize(self, sequences: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequences: [(id, seq), ...]
        Returns:
            dict with 'input_ids', 'attention_mask', 'token_type_ids'
        """

        spaced = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for _, seq in sequences]

        encoded = self.tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded

    def _build_residue_mask(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask'].bool()

        residue_mask = (
            attention_mask &
            (input_ids != self.pad_token_id) &
            (input_ids != self.cls_token_id) &
            (input_ids != self.sep_token_id)
        )
        return residue_mask

    def forward(self, tokens,
                return_per_residue: bool = False) -> Dict[str, torch.Tensor]:
        if isinstance(tokens, torch.Tensor):
            raise TypeError(
                "ProtBERTEncoder expects a dict from tokenize(), not a raw Tensor."
            )

        outputs = self.bert(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            token_type_ids=tokens.get('token_type_ids', None),
        )

        hidden = outputs.last_hidden_state

        residue_mask = self._build_residue_mask(tokens)
        mask_float = residue_mask.unsqueeze(-1).float()

        h = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)

        out = {'h': h}
        if return_per_residue:
            out['per_residue'] = hidden
            out['residue_mask'] = residue_mask

        return out