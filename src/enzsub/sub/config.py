"""
EnzSub 统一配置
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class EnzymeEncoderConfig:
    type: str = "esm2_650m"
    cpt_checkpoint: Optional[str] = None
    freeze_backbone: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

@dataclass
class SubstrateEncoderConfig:
    type: str = "chemberta"
    freeze: bool = True
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    molclr_pretrained_path: Optional[str] = None
    molclr_emb_dim: int = 300
    molclr_feat_dim: int = 512
    molclr_num_layers: int = 5
    molclr_drop_ratio: float = 0.0
    molclr_pool: str = "mean"
    ecfp_radius: int = 2
    ecfp_n_bits: int = 2048

@dataclass
class ModelConfig:
    enzyme: EnzymeEncoderConfig = field(default_factory=EnzymeEncoderConfig)
    substrate: SubstrateEncoderConfig = field(default_factory=SubstrateEncoderConfig)
    projection_dim: int = 256
    projection_dropout: float = 0.1
    substrate_projection_hidden_dim: Optional[int] = None

@dataclass
class PreserveConfig:
    """
    CPT-preservation loss 配置。

    三种模式:
      - enabled=False, monitor_only=False  : 完全关闭 (零开销, 老行为)
      - enabled=False, monitor_only=True   : 只观测 cosine, 不加入 loss
                                             (用于跑 baseline 衰减曲线)
      - enabled=True                        : 正常 preserve, 同时观测
                                             (monitor_only 此时被忽略)
    """
    enabled: bool = False
    monitor_only: bool = False

    pool_weight: float = 0.03
    token_weight: float = 0.0

    use_pool_preserve: bool = True
    use_token_preserve: bool = False

    loss_type: str = "cosine"

    teacher_mode: str = "cpt_without_lora"

    cache_dir: str = "./preserve_cache"
    force_rebuild: bool = False
    cache_batch_size: int = 8

@dataclass
class TaskConfig:
    """任务和损失配置"""
    weight_pref: float = 1.0
    weight_type: float = 0.5
    weight_contrast: float = 0.2

    reg_loss_type: str = "mse"
    huber_delta: float = 1.0

    temperature: float = 0.07
    pref_dim: int = 11
    num_substrate_types: int = 8

    num_pos_samples: int = 4
    num_neg_samples: int = 16
    neg_sampling: str = "mixed"

    preserve: PreserveConfig = field(default_factory=PreserveConfig)

@dataclass
class DataConfig:
    oed_data: str = ""
    enzyme_stats: str = ""
    max_seq_length: int = 1022
    pref_features: List[str] = field(default_factory=lambda: [
        'MolWt', 'LogP', 'TPSA', 'HBD', 'HBA',
        'NumRotatableBonds', 'NumRings', 'NumAromaticRings',
        'NumHeavyAtoms', 'NumHeteroatoms', 'FractionCSP3'
    ])
    normalize_pref: bool = True
    split_method: str = "cluster"
    cluster_file: Optional[str] = None
    pref_log_transform: bool = False
    seq_identity_threshold: float = 0.3
    train_ratio: float = 0.8
    sim_easy_max: float = -0.214
    sim_medium_min: float = -0.214
    sim_medium_max: float = 0.603
    sim_hard_min: float = 0.603
    sim_hard_max: float = 0.95
    neg_easy_ratio: float = 0.4
    neg_medium_ratio: float = 0.4
    neg_hard_ratio: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    accumulation_steps: int = 4
    fp16: bool = True
    device: str = "cuda:0"
    num_workers: int = 4
    output_dir: str = "./outputs"
    exp_name: Optional[str] = None
    save_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "EnzSub"
    seed: int = 2025

@dataclass
class EnzSubFullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(yaml_path: str) -> EnzSubFullConfig:
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    def _fill(dc_cls, d: dict):
        from dataclasses import fields as dc_fields
        kwargs = {}
        for fld in dc_fields(dc_cls):
            if fld.name not in d:
                continue
            val = d[fld.name]
            if hasattr(fld.type, '__dataclass_fields__') and isinstance(val, dict):
                kwargs[fld.name] = _fill(fld.type, val)
            else:
                kwargs[fld.name] = val
        return dc_cls(**kwargs)

    return _fill(EnzSubFullConfig, raw)