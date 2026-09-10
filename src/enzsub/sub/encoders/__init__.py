"""
编码器工厂

build_enzyme_encoder() / build_substrate_encoder() 根据 config 构建对应编码器。
"""

from typing import Optional
from .base import BaseEnzymeEncoder, BaseSubstrateEncoder
from ..config import EnzymeEncoderConfig, SubstrateEncoderConfig

def _enzyme_backbone_family(type_str: str) -> str:
    """判定 encoder type 所属的 backbone 家族 (大小写不敏感)"""
    t = type_str.lower()
    if t.startswith("esm"):
        return "esm"
    if t == "protbert_bfd":
        return "protbert"
    return "unknown"

def build_enzyme_encoder(config: EnzymeEncoderConfig) -> BaseEnzymeEncoder:
    """
    根据 config.type 构建酶编码器。

    支持:
        - ESM 家族 (任何带 'esm' 前缀的 key, 由 ESMEncoder._ESM_REGISTRY 解析):
            esm1b, esm2_t6_8M, esm2_t12_35M, esm2_t30_150M,
            esm2_650m / esm2_t33_650M, esm2_t36_3B, esm2_t48_15B
            以及对应的简写别名 (esm2_8m, esm2_35m, ...).
        - ProtBERT: protbert_bfd
    """

    raw = config.type
    family = _enzyme_backbone_family(raw)

    if family == "esm":
        from .esm import ESMEncoder
        return ESMEncoder(model_type=raw)

    elif family == "protbert":
        from .protbert import ProtBERTEncoder
        return ProtBERTEncoder()

    else:
        raise ValueError(
            f"Unknown enzyme encoder type: {config.type!r}. "
            f"Expected ESM-family ('esm1b', 'esm2_*') or 'protbert_bfd'."
        )

def build_substrate_encoder(config: SubstrateEncoderConfig) -> Optional[BaseSubstrateEncoder]:
    """
    根据 config.type 构建底物编码器。

    支持: "chemberta", "molclr", "ecfp", "none"
    返回 None 表示不使用底物编码器。
    """
    t = config.type.lower()

    if t == "chemberta":
        from .chemberta import ChemBERTaEncoder
        return ChemBERTaEncoder(
            model_name=config.model_name,
            freeze=config.freeze,
        )

    elif t == "molclr":
        if not config.molclr_pretrained_path:
            raise ValueError(
                "MolCLR encoder requires `substrate.molclr_pretrained_path` to be set."
            )
        from .molclr import MolCLREncoder
        return MolCLREncoder(
            pretrained_path=config.molclr_pretrained_path,
            emb_dim=config.molclr_emb_dim,
            feat_dim=config.molclr_feat_dim,
            num_layers=config.molclr_num_layers,
            drop_ratio=config.molclr_drop_ratio,
            pool=config.molclr_pool,
            freeze=config.freeze,
        )

    elif t == "ecfp":
        from .ecfp import ECFPEncoder
        return ECFPEncoder(
            radius=config.ecfp_radius,
            n_bits=config.ecfp_n_bits,
        )

    elif t == "none":
        return None

    else:
        raise ValueError(f"Unknown substrate encoder type: {config.type}")