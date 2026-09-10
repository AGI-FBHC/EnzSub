"""
ECFP (Extended Connectivity Fingerprint) 底物编码器

无可学习参数, 直接将 SMILES 转为固定长度 bit vector。
作为底物表示消融的"无预训练"baseline。

References:
    Rogers & Hahn, "Extended-Connectivity Fingerprints",
    J. Chem. Inf. Model., 2010.
"""

import logging
import warnings
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

from .base import BaseSubstrateEncoder

logger = logging.getLogger(__name__)

class ECFPEncoder(BaseSubstrateEncoder):
    """
    ECFP / Morgan Fingerprint 底物编码器

    Args:
        radius: Morgan 半径. radius=2 即 ECFP4, radius=3 即 ECFP6.
        n_bits: bit vector 长度 (默认 2048)

    forward 直接返回 bit vector (float32), 由下游 substrate_projection 投到对比空间。

    使用 MorganGenerator (RDKit 2022.09+) 替代旧的 GetMorganFingerprintAsBitVect,
    避免训练时的大量 deprecation warning。

    无可学习参数, 但保留一个 dummy buffer 以兼容
    next(self.parameters()) / .to(device) 等调用约定。
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048):
        super().__init__()
        self.radius = radius

        RDLogger.logger().setLevel(RDLogger.CRITICAL)
        self.n_bits = n_bits
        self._raw_dim = n_bits

        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

        self.register_buffer('_dummy', torch.zeros(1))

        print(f"[ECFPEncoder] Morgan FP (MorganGenerator): "
              f"radius={radius}, n_bits={n_bits}, trainable_params=0")

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    def prepare_batch(self, smiles_list: List[str],
                      device: torch.device) -> torch.Tensor:
        """
        SMILES 列表 → [B, n_bits] float32 tensor

        无效 SMILES 用全零向量 fallback。
        """
        fps = np.zeros((len(smiles_list), self.n_bits), dtype=np.float32)
        invalid_indices = []

        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_indices.append(i)
                continue
            try:
                fp = self._fp_gen.GetFingerprintAsNumPy(mol)
                fps[i] = fp.astype(np.float32)
            except Exception:
                invalid_indices.append(i)

        if invalid_indices:
            ratio = len(invalid_indices) / len(smiles_list)
            msg = (f"[ECFPEncoder] {len(invalid_indices)}/{len(smiles_list)} "
                   f"invalid SMILES → zero vector ({ratio:.1%})")
            if ratio > 0.1:
                warnings.warn(msg, stacklevel=2)
            else:
                logger.info(msg)

        return torch.from_numpy(fps).to(device)

    def forward(self, batch_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch_input: [B, n_bits] float32 tensor (from prepare_batch)
        Returns:
            {'h': [B, n_bits]}
        """
        return {'h': batch_input}