"""
EnzSub 模型

组装: 酶编码器 + 底物编码器 + 投影头 + 任务头 + 损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .config import ModelConfig, TaskConfig, EnzSubFullConfig
from .lora import apply_lora
from .encoders import build_enzyme_encoder, build_substrate_encoder
from .encoders.base import BaseEnzymeEncoder, BaseSubstrateEncoder

def _is_protbert_type(encoder_type: str) -> bool:
    return encoder_type.lower() == "protbert_bfd"

def _is_esm_type(encoder_type: str) -> bool:
    """ESM-1b 和所有 ESM-2 变体"""
    return encoder_type.lower().startswith("esm")

def _build_projection(
    in_dim: int,
    out_dim: int,
    dropout: float = 0.1,
    hidden_dim: Optional[int] = None,
) -> nn.Sequential:
    """
    """
    if in_dim <= 0:
        raise ValueError(
            f"in_dim must be positive, got {in_dim}"
        )

    if out_dim <= 0:
        raise ValueError(
            f"out_dim must be positive, got {out_dim}"
        )

    mid = in_dim // 2 if hidden_dim is None else int(hidden_dim)

    if mid <= 0:
        raise ValueError(
            f"Projection hidden dimension must be positive, got {mid}"
        )

    proj = nn.Sequential(
        nn.Linear(in_dim, mid),
        nn.LayerNorm(mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, out_dim),
    )

    for module in proj.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return proj

class SubstratePreferenceHead(nn.Module):
    """B3.1: 底物偏好回归"""

    def __init__(self, hidden_dim: int, pref_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pref_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

class SubstrateTypeHead(nn.Module):
    """B3.2: 底物类型分类 (multi-label)"""

    def __init__(self, hidden_dim: int, num_types: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_types),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

class EnzSubModel(nn.Module):
    """
    EnzSub 完整训练模型

    根据 config 自动组装:
        - 酶编码器 (ESM-2 / ESM-1b / ProtBERT) + LoRA
        - 底物编码器 (ChemBERTa / GNN / None) + 投影头
        - B3.1 偏好回归头
        - B3.2 类型分类头
        - B4 对比学习 (需要底物编码器)
    """

    def __init__(self, config: EnzSubFullConfig):
        super().__init__()
        self.config = config
        mc = config.model
        tc = config.task

        self.enzyme_encoder = build_enzyme_encoder(mc.enzyme)
        enz_dim = self.enzyme_encoder.hidden_dim

        if mc.enzyme.freeze_backbone:
            for param in self.enzyme_encoder.parameters():
                param.requires_grad = False

        if mc.enzyme.cpt_checkpoint:
            self._load_cpt(mc.enzyme.cpt_checkpoint)

        if mc.enzyme.lora.enabled:
            self._inject_lora(mc.enzyme)

        self.enzyme_projection = _build_projection(
            enz_dim, mc.projection_dim, mc.projection_dropout
        )

        self.substrate_encoder = build_substrate_encoder(mc.substrate)
        self.has_substrate_encoder = self.substrate_encoder is not None

        if self.has_substrate_encoder:
            sub_dim = self.substrate_encoder.raw_dim
            sub_projection_hidden_dim = getattr(
                mc,
                "substrate_projection_hidden_dim",
                None,
            )

            self.substrate_projection = _build_projection(
                in_dim=sub_dim,
                out_dim=mc.projection_dim,
                dropout=mc.projection_dropout,
                hidden_dim=sub_projection_hidden_dim,
            )

            actual_hidden_dim = (
                sub_dim // 2
                if sub_projection_hidden_dim is None
                else int(sub_projection_hidden_dim)
            )

            substrate_type = getattr(mc.substrate, "type", "unknown")

            print(
                "[SubstrateProjection] "
                f"type={substrate_type}, "
                f"architecture={sub_dim} -> "
                f"{actual_hidden_dim} -> "
                f"{mc.projection_dim}"
            )

        self.pref_head = SubstratePreferenceHead(
            enz_dim, tc.pref_dim, mc.projection_dropout
        )
        self.type_head = SubstrateTypeHead(
            enz_dim, tc.num_substrate_types, mc.projection_dropout
        )

        self._print_param_stats()

    def _get_backbone(self) -> nn.Module:
        """获取底层 backbone (esm / bert)，用于 CPT 加载和 LoRA 注入"""
        enc = self.enzyme_encoder
        if hasattr(enc, 'esm'):
            return enc.esm
        elif hasattr(enc, 'bert'):
            return enc.bert
        else:
            raise AttributeError("Cannot find backbone in enzyme encoder")

    def _load_cpt(self, path: str):
        """加载 CPT checkpoint 到 backbone"""
        from .lora import adapt_keys_for_lora

        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)

        backbone = self._get_backbone()
        current_keys = set(backbone.state_dict().keys())

        if hasattr(self.enzyme_encoder, 'bert'):
            stripped_state = {}
            for k, v in state.items():
                if k.startswith("bert."):
                    stripped_state[k[len("bert."):]] = v
                else:
                    stripped_state[k] = v
            state = stripped_state

        adapted = adapt_keys_for_lora(state, current_keys)

        missing, unexpected = backbone.load_state_dict(adapted, strict=False)

        matched = len([k for k in adapted.keys() if k in current_keys])
        print(f"[CPT] Loaded {path}")
        print(f"      matched={matched}, missing={len(missing)}, unexpected={len(unexpected)}")
        print(f"      missing keys: {missing}")
        print(f"      unexpected keys: {unexpected}")

    def _inject_lora(self, enzyme_cfg):
        """注入 LoRA"""
        backbone = self._get_backbone()
        lora_cfg = enzyme_cfg.lora
        apply_lora(
            backbone,
            target_modules=lora_cfg.target_modules,
            rank=lora_cfg.rank,
            alpha=lora_cfg.alpha,
            dropout=lora_cfg.dropout,
        )

    def _print_param_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[EnzSubModel] Total: {total:,}, Trainable: {trainable:,} "
              f"({trainable/total*100:.2f}%)")

    def encode_enzyme(self, tokens, return_per_residue: bool = False):
        """编码酶 → h, z (+ optional per_residue, residue_mask)"""
        out = self.enzyme_encoder(tokens, return_per_residue=return_per_residue)
        h = out['h']
        z = self.enzyme_projection(h)
        z = F.normalize(z, p=2, dim=-1)
        result = {'h': h, 'z': z}
        if return_per_residue:
            if 'per_residue' in out:
                result['per_residue'] = out['per_residue']
            if 'residue_mask' in out:
                result['residue_mask'] = out['residue_mask']
        return result

    def encode_substrate(self, batch_input):
        """编码底物 → h, z"""
        out = self.substrate_encoder(batch_input)
        h = out['h']
        z = self.substrate_projection(h)
        z = F.normalize(z, p=2, dim=-1)
        return {'h': h, 'z': z}

    def forward(
        self,
        enzyme_tokens,
        pos_substrate_batch=None,
        neg_substrate_batch=None,
        return_per_residue: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            enzyme_tokens: tokenize 后的酶输入
            pos_substrate_batch / neg_substrate_batch: 已 prepare_batch 的底物
            return_per_residue: 是否返回 per-residue hidden states + mask
                                (token-level preserve loss 需要)

        Returns:
            Dict with z_enzyme, z_substrate_pos, z_substrate_neg,
                    pred_pref, pred_type, h_enzyme
                    (+ per_residue_enzyme, residue_mask_enzyme if return_per_residue)
        """
        enz_out = self.encode_enzyme(enzyme_tokens, return_per_residue=return_per_residue)
        z_enz = enz_out['z']
        h_enz = enz_out['h']

        z_sub_pos = None
        z_sub_neg = None

        if self.has_substrate_encoder and pos_substrate_batch is not None:
            sub_pos_out = self.encode_substrate(pos_substrate_batch)
            z_sub_pos = sub_pos_out['z']

        if self.has_substrate_encoder and neg_substrate_batch is not None:
            sub_neg_out = self.encode_substrate(neg_substrate_batch)
            z_sub_neg = sub_neg_out['z']

        pred_pref = self.pref_head(h_enz)
        pred_type = self.type_head(h_enz)

        result = {
            'z_enzyme': z_enz,
            'z_substrate_pos': z_sub_pos,
            'z_substrate_neg': z_sub_neg,
            'pred_pref': pred_pref,
            'pred_type': pred_type,
            'h_enzyme': h_enz,
        }
        if return_per_residue:
            result['per_residue_enzyme'] = enz_out.get('per_residue')
            result['residue_mask_enzyme'] = enz_out.get('residue_mask')
        return result

class EnzSubLoss(nn.Module):
    """
    EnzSub 多任务损失

    - B3.1: MSE (底物偏好回归)
    - B3.2: BCE (底物类型分类)
    - B4:   InfoNCE (跨模态对比学习)

    权重为 0 时自动跳过对应损失项。

    [设计说明 - 难度采样]
        负样本难度 (hardness) 由 dataset 的分级采样 (_sample_negatives_mixed)
        在 *采样分布* 层面控制: 偏好相似的酶贡献更难的负样本。
        InfoNCE 本身对所有负样本 **等权** (标准 InfoNCE),不在 loss 内做难度加权。
        这样 `mixed` vs `random` 的消融只差「负样本来源分布」这一个变量,
        结论可干净归因,不与 loss 加权机制混杂。
    """

    def __init__(self, task_config: TaskConfig):
        super().__init__()
        self.tc = task_config

        reg_loss_type = getattr(task_config, 'reg_loss_type', 'mse')
        if reg_loss_type == 'huber':
            huber_delta = getattr(task_config, 'huber_delta', 1.0)
            self.reg_loss = nn.SmoothL1Loss(beta=huber_delta)
        elif reg_loss_type == 'mse':
            self.reg_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown reg_loss_type: {reg_loss_type}")

        self.bce_loss = nn.BCEWithLogitsLoss()

    def info_nce_loss(self, z_enz, z_sub_pos, z_sub_neg) -> torch.Tensor:
        """
        跨模态 InfoNCE (标准形式, 所有负样本等权)

        Args:
            z_enz: [B, D]
            z_sub_pos: [B*K, D]  (展平后)
            z_sub_neg: [B*N, D]

        Note:
            难度信号仅作用于 dataset 的负样本采样分布,不在此处加权。
            参见类 docstring「设计说明 - 难度采样」。
        """
        B = z_enz.shape[0]
        K = z_sub_pos.shape[0] // B
        N = z_sub_neg.shape[0] // B

        z_sub_pos = z_sub_pos.view(B, K, -1)
        z_sub_neg = z_sub_neg.view(B, N, -1)

        z_enz_exp = z_enz.unsqueeze(1)
        pos_sim = (z_enz_exp * z_sub_pos).sum(-1) / self.tc.temperature
        neg_sim = (z_enz_exp * z_sub_neg).sum(-1) / self.tc.temperature

        loss = 0.0
        for k in range(K):
            logits = torch.cat([pos_sim[:, k:k+1], neg_sim], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)
            loss += F.cross_entropy(logits, labels)

        return loss / K

    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        losses = {}
        device = outputs['h_enzyme'].device
        total = torch.tensor(0.0, device=device)

        if self.tc.weight_pref > 0:
            loss_pref = self.reg_loss(outputs['pred_pref'], targets['pref_labels'])
            losses['loss_pref'] = loss_pref
            total = total + self.tc.weight_pref * loss_pref
        else:
            losses['loss_pref'] = torch.tensor(0.0, device=device)

        if self.tc.weight_type > 0:
            loss_type = self.bce_loss(outputs['pred_type'], targets['type_labels'])
            losses['loss_type'] = loss_type
            total = total + self.tc.weight_type * loss_type
        else:
            losses['loss_type'] = torch.tensor(0.0, device=device)

        if (self.tc.weight_contrast > 0
                and outputs['z_substrate_pos'] is not None
                and outputs['z_substrate_neg'] is not None):
            loss_contrast = self.info_nce_loss(
                outputs['z_enzyme'],
                outputs['z_substrate_pos'],
                outputs['z_substrate_neg'],
            )
            losses['loss_contrast'] = loss_contrast
            total = total + self.tc.weight_contrast * loss_contrast
        else:
            losses['loss_contrast'] = torch.tensor(0.0, device=device)

        losses['total_loss'] = total
        return losses

class EnzSubModelForDownstream(nn.Module):
    """
    下游任务统一 embedding 提取器。

    支持四种模式:
        - base:     原始预训练 backbone
        - cpt:      加载 CPT/backbone checkpoint, 不加载 LoRA
        - base_sub: 加载 backbone + SUB LoRA
        - cpt_sub:  加载 backbone + SUB LoRA

    关键设计:
        1. model_mode 是最高优先级。
           checkpoint 中是否存在 lora_config / lora_state_dict 不决定是否加载 LoRA。

        2. 只有 model_mode in ["base_sub", "cpt_sub"] 时才:
           - 读取 lora_config
           - 注入 LoRA 结构
           - 加载 lora_state_dict

        3. model_mode == "cpt" 时:
           - 加载 backbone 权重
           - 明确忽略 checkpoint 中的 LoRA 配置和 LoRA 权重

        4. 兼容旧调用:
           如果不传 model_mode，但传了 load_lora，则按旧逻辑推断:
              checkpoint_path is None → base
              checkpoint_path exists and load_lora=False → cpt
              checkpoint_path exists and load_lora=True  → cpt_sub
    """

    VALID_MODES = ("base", "cpt", "base_sub", "cpt_sub")
    SUB_MODES = ("base_sub", "cpt_sub")
    CHECKPOINT_MODES = ("cpt", "base_sub", "cpt_sub")

    MODE_DESC = {
        "base": "original pretrained backbone",
        "cpt": "CPT/backbone checkpoint, no LoRA",
        "base_sub": "original/CPT backbone + SUB LoRA",
        "cpt_sub": "CPT backbone + SUB LoRA",
    }

    DEFAULT_LORA_RANK = 16
    DEFAULT_LORA_ALPHA = 32
    DEFAULT_ESM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]
    DEFAULT_PROTBERT_LORA_TARGETS = ["query", "value"]

    def __init__(
        self,
        encoder_type: str = "esm2_650m",
        model_mode: Optional[str] = None,
        checkpoint_path=None,
        load_lora: Optional[bool] = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        freeze_backbone: bool = True,
        device: str = "cpu",
        strict_lora_load: bool = True,
        allow_lora_reverse_detect: bool = True,
    ):
        super().__init__()
        from .config import EnzymeEncoderConfig

        self.encoder_type = encoder_type
        self.model_mode = self._resolve_model_mode(
            model_mode=model_mode,
            load_lora=load_lora,
            checkpoint_path=checkpoint_path,
        )
        self.use_lora = self.model_mode in self.SUB_MODES
        self.strict_lora_load = strict_lora_load

        self.loaded_cpt = False
        self.loaded_lora = False
        self.has_lora_structure = False
        self.lora_config_source = None
        self.checkpoint_format = None

        if self.model_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown model_mode={self.model_mode!r}. "
                f"Choose from {self.VALID_MODES}."
            )

        if self.model_mode in self.CHECKPOINT_MODES and checkpoint_path is None:
            raise ValueError(f"model_mode={self.model_mode!r} requires checkpoint_path.")

        if self.model_mode == "base" and checkpoint_path is not None:
            print(
                "[Downstream][WARN] model_mode='base': checkpoint_path is provided "
                "but will be ignored."
            )

        ckpt = None
        if self.model_mode != "base":
            ckpt = self._read_checkpoint(checkpoint_path)
            self.checkpoint_format = self._detect_checkpoint_format(ckpt)

            print(f"[Downstream] mode={self.model_mode} ({self.MODE_DESC[self.model_mode]})")
            print(f"[Downstream] checkpoint_format={self.checkpoint_format}")
            self._print_checkpoint_meta(ckpt)

            if (not self.use_lora) and isinstance(ckpt, dict) and ckpt.get("lora_state_dict"):
                print(
                    "[Downstream] model_mode does not request LoRA; "
                    "lora_config/lora_state_dict in checkpoint will be ignored."
                )

        else:
            print(f"[Downstream] mode=base ({self.MODE_DESC[self.model_mode]})")

        final_lora_rank = lora_rank
        final_lora_alpha = lora_alpha
        final_lora_dropout = lora_dropout
        final_lora_targets = lora_target_modules

        if self.use_lora:
            (
                final_lora_rank,
                final_lora_alpha,
                final_lora_dropout,
                final_lora_targets,
            ) = self._resolve_lora_config(
                ckpt=ckpt,
                encoder_type=encoder_type,
                fallback_rank=lora_rank,
                fallback_alpha=lora_alpha,
                fallback_dropout=lora_dropout,
                fallback_targets=lora_target_modules,
                allow_reverse_detect=allow_lora_reverse_detect,
            )

        enc_config = EnzymeEncoderConfig(type=encoder_type)
        self.encoder = build_enzyme_encoder(enc_config)

        if self.use_lora:
            backbone = self._get_backbone()
            apply_lora(
                backbone,
                target_modules=final_lora_targets,
                rank=final_lora_rank,
                alpha=final_lora_alpha,
                dropout=final_lora_dropout,
            )
            self.has_lora_structure = True

        if self.model_mode != "base":
            self._load_checkpoint_by_mode(ckpt)

        if freeze_backbone:
            self._freeze(keep_lora_trainable=True)

        self.to(device)
        self.eval()
        self._print_status()

    def _resolve_model_mode(
        self,
        model_mode: Optional[str],
        load_lora: Optional[bool],
        checkpoint_path,
    ) -> str:
        """
        新接口优先使用 model_mode。
        若 model_mode=None，则兼容旧接口 load_lora。
        """
        if model_mode is not None:
            return model_mode

        if checkpoint_path is None:
            return "base"

        if load_lora is True:
            print(
                "[Downstream][WARN] model_mode is None and load_lora=True. "
                "Interpreting as model_mode='cpt_sub' for legacy compatibility. "
                "New code should pass model_mode explicitly."
            )
            return "cpt_sub"

        print(
            "[Downstream][WARN] model_mode is None. "
            "Interpreting checkpoint_path + load_lora=False/None as model_mode='cpt'. "
            "New code should pass model_mode explicitly."
        )
        return "cpt"

    @staticmethod
    def _read_checkpoint(path_or_state):
        if isinstance(path_or_state, dict):
            return path_or_state
        return torch.load(path_or_state, map_location="cpu")

    @staticmethod
    def _print_checkpoint_meta(ckpt: dict):
        if not isinstance(ckpt, dict):
            return
        for key in ("epoch", "global_step", "val_loss", "val_perplexity", "best_val_loss"):
            if key in ckpt:
                print(f"[Downstream] {key}: {ckpt[key]}")

    @staticmethod
    def _detect_checkpoint_format(ckpt) -> str:
        """
        返回:
            - sub_fmt: EnzSub/SUB checkpoint, 有 esm_state_dict
            - mlm_fmt: ProtBERT MLM/CPT checkpoint, 常见 bert. / cls. 前缀
            - cpt_fmt: 标准 CPT checkpoint, 有 model_state_dict
            - raw_fmt: 纯 state_dict
        """
        if isinstance(ckpt, dict) and "esm_state_dict" in ckpt:
            return "sub_fmt"

        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            keys = list(state.keys())
            if any(k.startswith("bert.") or k.startswith("cls.") for k in keys):
                return "mlm_fmt"
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                return "cpt_fmt"

        return "raw_fmt"

    def _default_lora_targets(self, encoder_type: str) -> List[str]:
        if _is_protbert_type(encoder_type):
            return list(self.DEFAULT_PROTBERT_LORA_TARGETS)
        if _is_esm_type(encoder_type):
            return list(self.DEFAULT_ESM_LORA_TARGETS)
        raise ValueError(
            f"Cannot infer LoRA target modules for encoder_type={encoder_type!r}. "
            "Pass lora_target_modules explicitly."
        )

    def _resolve_lora_config(
        self,
        ckpt: dict,
        encoder_type: str,
        fallback_rank: int,
        fallback_alpha: int,
        fallback_dropout: float,
        fallback_targets: Optional[List[str]],
        allow_reverse_detect: bool = True,
    ):
        """
        只在 base_sub / cpt_sub 模式调用。

        优先级:
            1. checkpoint['lora_config']
            2. 从 lora_state_dict 反推 rank 和 target_modules，alpha 用 fallback_alpha
            3. fallback 参数 / 默认值

        注意:
            alpha 无法从 LoRA 权重反推。因此没有 lora_config 时，必须依赖 fallback_alpha。
        """
        if not isinstance(ckpt, dict):
            raise TypeError("SUB mode requires checkpoint to be a dict-like checkpoint.")

        lora_state = ckpt.get("lora_state_dict", {}) or {}
        lora_config = ckpt.get("lora_config", None)

        if not lora_state:
            msg = (
                f"model_mode={self.model_mode!r} requires LoRA weights, "
                "but checkpoint has no lora_state_dict."
            )
            if self.strict_lora_load:
                raise RuntimeError(msg)
            print(f"[Downstream][WARN] {msg}")

        if lora_config:
            rank = lora_config.get("rank", fallback_rank)
            alpha = lora_config.get("alpha", fallback_alpha)
            dropout = lora_config.get("dropout", fallback_dropout)
            targets = lora_config.get("target_modules", fallback_targets)

            if targets is None:
                targets = self._default_lora_targets(encoder_type)

            self.lora_config_source = "checkpoint:lora_config"
            print(
                "[Downstream] Using lora_config from checkpoint: "
                f"rank={rank}, alpha={alpha}, dropout={dropout}, targets={targets}"
            )
            return int(rank), int(alpha), float(dropout), list(targets)

        if lora_state and allow_reverse_detect:
            import re

            detected_rank = None
            detected_targets = set()

            for k, v in lora_state.items():

                m = re.search(r"\.([^\.]+)\.lora\.lora_A\.weight$", k)
                if m:
                    detected_targets.add(m.group(1))
                    detected_rank = int(v.shape[0])

            if detected_rank is not None and detected_targets:
                targets = sorted(detected_targets)
                alpha = fallback_alpha
                dropout = fallback_dropout

                self.lora_config_source = "reverse-detected"
                print(
                    "[Downstream][WARN] checkpoint has no lora_config. "
                    "Reverse-detected LoRA structure from weights: "
                    f"rank={detected_rank}, targets={targets}. "
                    f"alpha cannot be inferred; using fallback alpha={alpha}."
                )
                return int(detected_rank), int(alpha), float(dropout), list(targets)

        rank = fallback_rank if fallback_rank is not None else self.DEFAULT_LORA_RANK
        alpha = fallback_alpha if fallback_alpha is not None else self.DEFAULT_LORA_ALPHA
        dropout = fallback_dropout
        targets = fallback_targets if fallback_targets is not None else self._default_lora_targets(encoder_type)

        self.lora_config_source = "fallback"
        print(
            "[Downstream][WARN] Failed to get LoRA config from checkpoint. "
            f"Using fallback: rank={rank}, alpha={alpha}, dropout={dropout}, targets={targets}. "
            "If these differ from training, downstream embeddings will be wrong."
        )

        return int(rank), int(alpha), float(dropout), list(targets)

    def _get_backbone(self) -> nn.Module:
        if hasattr(self.encoder, "esm"):
            return self.encoder.esm
        if hasattr(self.encoder, "bert"):
            return self.encoder.bert
        raise AttributeError("Cannot find backbone")

    def _extract_backbone_state(self, ckpt) -> dict:
        """
        根据 checkpoint 格式提取 backbone 权重。

        支持:
            - sub_fmt: ckpt['esm_state_dict']
            - cpt_fmt: ckpt['model_state_dict']
            - mlm_fmt: ckpt['model_state_dict']，同时处理 ProtBERT 的 bert. / cls. 前缀
            - raw_fmt: ckpt 本身或 ckpt['model_state_dict']
        """
        fmt = self._detect_checkpoint_format(ckpt)

        if fmt == "sub_fmt":
            return ckpt.get("esm_state_dict", {}) or {}

        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise TypeError(f"Cannot extract backbone state from checkpoint format={fmt}.")

        cleaned = {}
        for k, v in state.items():
            new_k = k

            if new_k.startswith("module."):
                new_k = new_k[len("module."):]

            if fmt == "mlm_fmt":
                if new_k.startswith("bert."):
                    new_k = new_k[len("bert."):]
                elif new_k.startswith("cls."):
                    continue

            cleaned[new_k] = v

        return cleaned

    def _load_checkpoint_by_mode(self, ckpt):
        """
        按 model_mode 加载。

        base:
            不调用本函数。

        cpt:
            只加载 backbone，不加载 LoRA。

        base_sub / cpt_sub:
            加载 backbone + LoRA。
        """
        from .lora import adapt_keys_for_lora

        backbone = self._get_backbone()
        current_keys = set(backbone.state_dict().keys())

        backbone_state = self._extract_backbone_state(ckpt)
        if not backbone_state:
            raise RuntimeError(
                f"No backbone weights found in checkpoint for model_mode={self.model_mode!r}."
            )

        adapted = adapt_keys_for_lora(backbone_state, current_keys)
        missing, unexpected = backbone.load_state_dict(adapted, strict=False)

        matched = sum(1 for k in adapted.keys() if k in current_keys)
        self.loaded_cpt = True

        print(
            "[Downstream] Loaded backbone: "
            f"matched={matched}/{len(backbone_state)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

        missing_non_lora = [k for k in missing if "lora" not in k.lower()]
        unexpected_non_lora = [k for k in unexpected if "lora" not in k.lower()]

        if missing_non_lora:
            print(f"  [WARN] Non-LoRA missing keys: {missing_non_lora[:10]}")
        if unexpected_non_lora:
            print(f"  [WARN] Non-LoRA unexpected keys: {unexpected_non_lora[:10]}")

        if self.use_lora:
            if not isinstance(ckpt, dict):
                raise TypeError("SUB mode requires dict checkpoint with lora_state_dict.")

            lora_state = ckpt.get("lora_state_dict", {}) or {}
            self._load_lora_state_strict(backbone, lora_state)

    def _load_lora_state_strict(self, backbone: nn.Module, lora_state: Dict[str, torch.Tensor]):
        """
        严格加载 LoRA 权重。

        检查:
            - 当前模型是否已经注入 LoRA
            - checkpoint 中每个 LoRA key 是否能匹配当前模型参数
            - shape 是否一致
            - 是否完整加载
        """
        if not self.has_lora_structure:
            raise RuntimeError(
                "LoRA weights are requested, but current model has no LoRA structure. "
                "This should only happen if model_mode handling is wrong."
            )

        if not lora_state:
            raise RuntimeError(
                f"model_mode={self.model_mode!r} requires LoRA weights, "
                "but lora_state_dict is empty."
            )

        params = dict(backbone.named_parameters())

        loaded = 0
        skipped = []
        shape_mismatch = []
        failed = []

        for k, v in lora_state.items():
            if k not in params:
                skipped.append(k)
                continue

            if tuple(params[k].shape) != tuple(v.shape):
                shape_mismatch.append((k, tuple(params[k].shape), tuple(v.shape)))
                continue

            try:
                params[k].data.copy_(v.to(device=params[k].device, dtype=params[k].dtype))
                loaded += 1
            except Exception as e:
                failed.append((k, str(e)))

        total = len(lora_state)
        self.loaded_lora = (loaded == total)

        print(f"[Downstream] Loaded LoRA: {loaded}/{total} tensors")

        if skipped:
            print(
                f"  [WARN] {len(skipped)} LoRA keys not found in current model. "
                f"Examples: {skipped[:8]}"
            )
        if shape_mismatch:
            print(
                f"  [WARN] {len(shape_mismatch)} LoRA shape mismatches. "
                f"Examples: {shape_mismatch[:5]}"
            )
        if failed:
            print(
                f"  [WARN] {len(failed)} LoRA copy failures. "
                f"Examples: {failed[:5]}"
            )

        if self.strict_lora_load and loaded != total:
            raise RuntimeError(
                "LoRA loading incomplete: "
                f"loaded={loaded}, total={total}, "
                f"skipped={len(skipped)}, "
                f"shape_mismatch={len(shape_mismatch)}, failed={len(failed)}. "
                "Likely causes: wrong target_modules, wrong rank, wrong checkpoint, "
                "or old checkpoint without matching lora_config."
            )

    def _freeze(self, keep_lora_trainable: bool = True):
        backbone = self._get_backbone()
        for name, param in backbone.named_parameters():
            if keep_lora_trainable and "lora" in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _print_status(self):
        parts = [self.encoder.__class__.__name__, f"mode={self.model_mode}"]

        if self.loaded_cpt:
            parts.append("backbone-loaded")

        if self.has_lora_structure:
            if self.loaded_lora:
                lora_tag = "LoRA(trained)"
            else:
                lora_tag = "LoRA(random/unloaded)"
            if self.lora_config_source:
                lora_tag += f"[cfg={self.lora_config_source}]"
            parts.append(lora_tag)

        print(f"\n{'=' * 50}")
        print(f" Downstream: {' + '.join(parts)}")
        print(f"{'=' * 50}\n")

    @property
    def hidden_dim(self) -> int:
        return self.encoder.hidden_dim

    def tokenize(self, sequences: List[Tuple[str, str]]):
        return self.encoder.tokenize(sequences)

    def forward(self, tokens, return_per_residue: bool = False):
        return self.encoder(tokens, return_per_residue=return_per_residue)

    @torch.no_grad()
    def get_embedding(self, sequences: List[Tuple[str, str]]) -> torch.Tensor:
        self.eval()
        tokens = self.tokenize(sequences)

        device = next(self.parameters()).device
        if isinstance(tokens, dict):
            tokens = {k: v.to(device) for k, v in tokens.items()}
        else:
            tokens = tokens.to(device)

        with torch.cuda.amp.autocast():
            out = self.forward(tokens)

        return out["h"].float()
    @property
    def repr_layer(self) -> int:
        """返回用于提取表示的 transformer 层号"""
        backbone = self._get_backbone()
        if hasattr(backbone, "num_layers"):
            return backbone.num_layers
        if hasattr(backbone, "config"):
            return backbone.config.num_hidden_layers
        raise AttributeError("Cannot determine repr_layer from backbone")

    @property
    def max_seq_len(self) -> int:
        """最大可接受的氨基酸序列长度，不含特殊 token"""
        if hasattr(self.encoder, "esm"):
            return 1022
        return 510

    @torch.no_grad()
    def get_per_residue_embedding(
        self,
        sequences: List[Tuple[str, str]],
    ) -> List[torch.Tensor]:
        """
        提取 per-residue embedding。

        Args:
            sequences: [(id, seq), ...]，seq 为原始氨基酸序列。

        Returns:
            list of Tensor(L_i, D)，每个 Tensor 已在 CPU 上，dtype=float32。
        """
        self.eval()
        tokens = self.tokenize(sequences)
        device = next(self.parameters()).device
        backbone = self._get_backbone()
        is_esm = hasattr(self.encoder, "esm")

        if is_esm:
            tokens = tokens.to(device)
            repr_layer = self.repr_layer
            with torch.cuda.amp.autocast():
                results = backbone(tokens, repr_layers=[repr_layer])
            reps = results["representations"][repr_layer]
        else:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            repr_layer = self.repr_layer
            with torch.cuda.amp.autocast():
                outputs = backbone(output_hidden_states=True, **tokens)
            reps = outputs.hidden_states[repr_layer]

        max_len = self.max_seq_len
        per_residue = []

        for i, (_, seq) in enumerate(sequences):
            seq_len = min(len(seq.upper().replace(" ", "")), max_len)
            emb = reps[i, 1:seq_len + 1].cpu().float()
            per_residue.append(emb)

        return per_residue