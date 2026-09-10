"""
LoRA (Low-Rank Adaptation) 实现

通用 LoRA 模块，可应用到任意包含 nn.Linear 的模型。
"""

import math
import torch
import torch.nn as nn
from typing import List
from contextlib import contextmanager

class LoRALinear(nn.Module):
    """LoRA adapter: 输出低秩增量 ΔW·x"""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

class LoRALayer(nn.Module):
    """
    包装: 原始 Linear (冻结) + LoRA adapter (可训练)

    通过 self.lora_enabled 标志位 (或 disable_lora_context manager)
    临时关闭 LoRA delta, 用于 teacher pass。
    """

    def __init__(self, original_layer: nn.Linear,
                 rank: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALinear(
            original_layer.in_features, original_layer.out_features,
            rank=rank, alpha=alpha, dropout=dropout
        )

        self.lora_enabled = True
        for param in self.original_layer.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    @property
    def in_features(self):
        return self.original_layer.in_features

    @property
    def out_features(self):
        return self.original_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_layer(x)
        if self.lora_enabled:
            out = out + self.lora(x)
        return out

@contextmanager
def disable_lora_context(module: nn.Module):
    """
    临时关闭 module 中所有 LoRALayer 的 LoRA delta + 切到 eval 模式禁用 dropout。
    退出时恢复。用于 teacher pass:

        with disable_lora_context(enzyme_encoder), torch.no_grad():
            teacher_out = enzyme_encoder(tokens)
    """
    lora_layers = [m for m in module.modules() if isinstance(m, LoRALayer)]
    saved_states = [l.lora_enabled for l in lora_layers]
    was_training = module.training

    for l in lora_layers:
        l.lora_enabled = False
    module.eval()

    try:
        yield
    finally:
        for l, s in zip(lora_layers, saved_states):
            l.lora_enabled = s
        if was_training:
            module.train()

def apply_lora(model: nn.Module, target_modules: List[str],
               rank: int = 8, alpha: int = 16, dropout: float = 0.1) -> nn.Module:
    """注入 LoRA 到指定 Linear 层 (原地修改)。"""
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        parts = name.rsplit('.', 1)
        parent_name = parts[0] if len(parts) > 1 else ''
        attr_name = parts[-1]
        parent = dict(model.named_modules())[parent_name] if parent_name else model
        replacements.append((parent, attr_name, module))

    for parent, attr_name, module in replacements:
        lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_layer)

    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower())
    print(f"[LoRA] Injected {len(replacements)} layers, {lora_params:,} trainable params")
    return model

def strip_lora_keys(state_dict: dict) -> dict:
    """把包含 .original_layer. 的 key 清理回原始格式。"""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('.original_layer.', '.')] = v
    return cleaned

def adapt_keys_for_lora(state_dict: dict, current_keys: set) -> dict:
    """把原始格式的 key 适配到带 LoRA 结构的模型。"""
    adapted = {}
    for k, v in state_dict.items():
        if k in current_keys:
            adapted[k] = v
        else:
            for suffix in ['.weight', '.bias']:
                if k.endswith(suffix):
                    base = k[:-len(suffix)]
                    new_key = base + '.original_layer' + suffix
                    if new_key in current_keys:
                        adapted[new_key] = v
                        break
            else:
                adapted[k] = v
    return adapted