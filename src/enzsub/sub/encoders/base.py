"""
编码器基类

定义酶编码器和底物编码器的统一接口。
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseEnzymeEncoder(ABC, nn.Module):
    """
    酶编码器基类

    所有子类需实现:
        - tokenize(sequences) -> Any
        - forward(tokens, return_per_residue) -> dict
        - hidden_dim (property)
        - num_layers (property)
    """

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """隐层维度"""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Transformer 层数"""
        ...

    @abstractmethod
    def tokenize(self, sequences: List[Tuple[str, str]]) -> Any:
        """
        Tokenize 序列。

        Args:
            sequences: [(id, seq), ...] 格式

        Returns:
            tokens: 模型可接受的输入格式
        """
        ...

    @abstractmethod
    def forward(self, tokens, return_per_residue: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Returns:
            {
                'h': [B, hidden_dim]            - mean-pooled embedding
                'per_residue': [B, L, hidden_dim] - 仅 return_per_residue=True 时
            }
        """
        ...

class BaseSubstrateEncoder(ABC, nn.Module):
    """
    底物编码器基类

    所有子类需实现:
        - raw_dim (property): 投影前的维度
        - prepare_batch(smiles_list) -> Any: 准备 batch 输入
        - forward(batch_input) -> dict
    """

    @property
    @abstractmethod
    def raw_dim(self) -> int:
        """投影前的原始维度"""
        ...

    @abstractmethod
    def prepare_batch(self, smiles_list: List[str], device: torch.device) -> Any:
        """
        将 SMILES 列表转为模型输入格式。

        Args:
            smiles_list: SMILES 字符串列表
            device: 目标设备

        Returns:
            模型可接受的 batch 输入
        """
        ...

    @abstractmethod
    def forward(self, tokens, return_per_residue: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Returns:
            {
                'h': [B, hidden_dim]              - mean-pooled embedding
                'per_residue': [B, L, hidden_dim] - RAW hidden states (未乘 mask),
                                                    仅 return_per_residue=True 时
                'residue_mask': [B, L] (bool)     - True=有效残基 (非 pad/CLS/EOS),
                                                    仅 return_per_residue=True 时
            }

        Note: per_residue 必须是未掩码的 raw hidden, 由调用方按 residue_mask 处理。
            这是为了让 token-level preserve loss 能正确计算 cosine。
        """
        ...