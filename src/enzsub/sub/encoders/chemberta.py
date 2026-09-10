"""
ChemBERTa 底物编码器

SMILES → tokenize → CLS embedding (768-dim)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any

from .base import BaseSubstrateEncoder

class ChemBERTaEncoder(BaseSubstrateEncoder):
    """
    ChemBERTa 底物编码器 (冻结)

    输出: CLS token embedding, 768-dim
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                 freeze: bool = True, max_length: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self._raw_dim = self.encoder.config.hidden_size

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        print(f"[ChemBERTaEncoder] Loaded {model_name}, dim={self._raw_dim}, "
              f"freeze={freeze}")

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    def prepare_batch(self, smiles_list: List[str],
                      device: torch.device) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(self, batch_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if all(not p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=batch_input['input_ids'],
                    attention_mask=batch_input['attention_mask'],
                )
        else:
            outputs = self.encoder(
                    input_ids=batch_input['input_ids'],
                    attention_mask=batch_input['attention_mask'],
                )

        h = outputs.last_hidden_state[:, 0, :]
        return {'h': h}