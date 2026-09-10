"""
Checkpoint 保存/加载工具

统一处理 LoRA key 映射，分离各组件权重。
"""

import torch
from typing import Dict, Optional, Tuple

from .lora import strip_lora_keys

def save_checkpoint(
    model,
    save_path: str,
    epoch: int = 0,
    global_step: int = 0,
    optimizer_state: Optional[Dict] = None,
    scheduler_state: Optional[Dict] = None,
    best_val_loss: float = float('inf'),
    config=None,
    extra_info: Optional[Dict] = None,
):
    """
    保存 EnzSub checkpoint，分离各组件。

    保存结构:
        - config
        - esm_state_dict: backbone 权重 (key 已还原为原始格式)
        - lora_state_dict: LoRA adapter 权重
        - enzyme_projection_state_dict
        - substrate_projection_state_dict (如有)
        - pref_head_state_dict
        - type_head_state_dict
        - epoch, global_step, best_val_loss
    """
    checkpoint = {
        'config': config or model.config,
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
    }

    backbone = model._get_backbone()

    esm_state = {}
    lora_state = {}

    for name, param in backbone.named_parameters():
        if 'lora' in name.lower():
            lora_state[name] = param.data.cpu()
        else:
            esm_state[name] = param.data.cpu()

    for name, buf in backbone.named_buffers():
        esm_state[name] = buf.cpu()

    esm_state = strip_lora_keys(esm_state)

    checkpoint['esm_state_dict'] = esm_state
    checkpoint['lora_state_dict'] = lora_state

    _lora_cfg = getattr(getattr(model.config.model, 'enzyme', None), 'lora', None)
    if _lora_cfg is not None and getattr(_lora_cfg, 'enabled', False):
        checkpoint['lora_config'] = {
            'rank': _lora_cfg.rank,
            'alpha': _lora_cfg.alpha,
            'target_modules': list(_lora_cfg.target_modules),
            'dropout': _lora_cfg.dropout,
        }

    checkpoint['enzyme_projection_state_dict'] = {
        k: v.cpu() for k, v in model.enzyme_projection.state_dict().items()
    }
    if model.has_substrate_encoder:
        checkpoint['substrate_projection_state_dict'] = {
            k: v.cpu() for k, v in model.substrate_projection.state_dict().items()
        }

    checkpoint['pref_head_state_dict'] = {
        k: v.cpu() for k, v in model.pref_head.state_dict().items()
    }
    checkpoint['type_head_state_dict'] = {
        k: v.cpu() for k, v in model.type_head.state_dict().items()
    }

    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    if scheduler_state is not None:
        checkpoint['scheduler_state_dict'] = scheduler_state
    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to {save_path}")
    print(f"  ESM keys: {len(esm_state)}, LoRA keys: {len(lora_state)}")

def load_checkpoint(
    checkpoint_path: str,
    model=None,
    load_lora: bool = True,
    load_projections: bool = True,
    load_heads: bool = True,
    device: str = 'cpu',
) -> Tuple:
    """
    加载 EnzSub checkpoint。

    Returns:
        (model, info_dict)
    """
    from .lora import adapt_keys_for_lora

    ckpt = torch.load(checkpoint_path, map_location=device)

    if model is None:
        raise ValueError("必须传入已构建的 model 实例")

    backbone = model._get_backbone()
    current_keys = set(backbone.state_dict().keys())

    esm_state = ckpt.get('esm_state_dict', {})
    if esm_state:
        adapted = adapt_keys_for_lora(esm_state, current_keys)
        missing, unexpected = backbone.load_state_dict(adapted, strict=False)
        print(f"[Load] Backbone: loaded (missing={len(missing)})")

    if load_lora:
        lora_state = ckpt.get('lora_state_dict', {})
        if lora_state:
            params = dict(backbone.named_parameters())
            loaded = 0
            for k, v in lora_state.items():
                if k in params:
                    params[k].data.copy_(v)
                    loaded += 1
            print(f"[Load] LoRA: {loaded}/{len(lora_state)} tensors")

    if load_projections:
        enz_proj = ckpt.get('enzyme_projection_state_dict', {})
        if enz_proj:
            model.enzyme_projection.load_state_dict(enz_proj)
        sub_proj = ckpt.get('substrate_projection_state_dict', {})
        if sub_proj and model.has_substrate_encoder:
            model.substrate_projection.load_state_dict(sub_proj)
        print(f"[Load] Projections loaded")

    if load_heads:
        pref = ckpt.get('pref_head_state_dict', {})
        if pref:
            model.pref_head.load_state_dict(pref)
        type_h = ckpt.get('type_head_state_dict', {})
        if type_h:
            model.type_head.load_state_dict(type_h)
        print(f"[Load] Task heads loaded")

    model = model.to(device)

    info = {
        'epoch': ckpt.get('epoch', 0),
        'global_step': ckpt.get('global_step', 0),
        'best_val_loss': ckpt.get('best_val_loss', float('inf')),
        'optimizer_state_dict': ckpt.get('optimizer_state_dict'),
        'scheduler_state_dict': ckpt.get('scheduler_state_dict'),
        'extra_info': ckpt.get('extra_info'),
    }

    return model, info