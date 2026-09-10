"""
CPT-Preservation Loss

约束 SUB 阶段 LoRA 更新不要过度破坏 CPT backbone 已学到的 enzyme-intrinsic representation。

实现策略:
- pool-level: 启动时预提取 teacher pool embeddings 缓存, 训练时按 enzyme_id 查表, 零开销
- token-level: on-the-fly, 通过 disable_lora_context 在同模型上跑一次 no_grad teacher pass

teacher = CPT backbone (no LoRA delta), student = CPT backbone + LoRA adapter
所有 cosine 计算强制 fp32, 避免 bf16 数值噪声。
"""

import os
import hashlib
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .lora import disable_lora_context

def compute_cache_signature(
    encoder_type: str,
    cpt_checkpoint: Optional[str],
    max_seq_length: int,
) -> str:
    """生成 teacher pool 缓存的签名。任何影响 teacher 输出的字段变化 → 签名变化 → 自动失效。"""
    h = hashlib.md5()
    h.update(encoder_type.encode())
    h.update(str(max_seq_length).encode())

    if cpt_checkpoint and os.path.exists(cpt_checkpoint):
        h.update(os.path.abspath(cpt_checkpoint).encode())
        h.update(str(os.path.getmtime(cpt_checkpoint)).encode())
        ckpt_tag = "cpt"
    else:
        ckpt_tag = "nockpt"

    digest = h.hexdigest()[:10]
    return f"{encoder_type}__{ckpt_tag}__L{max_seq_length}__{digest}"

def cache_path_for(cache_dir: str, signature: str) -> str:
    return os.path.join(cache_dir, f"teacher_pool__{signature}.pt")

@torch.no_grad()
def build_teacher_pool_cache(
    enzyme_encoder,
    enzyme_pairs: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    用 frozen CPT backbone (LoRA disabled) 对每条酶做一次 mean-pooled forward,
    返回 dict[enzyme_id] -> tensor[D] (CPU, fp32)。
    """
    enzyme_encoder = enzyme_encoder.to(device)
    cache: Dict[str, torch.Tensor] = {}

    pbar = tqdm(range(0, len(enzyme_pairs), batch_size),
                desc="  Building teacher pool cache")

    with disable_lora_context(enzyme_encoder):
        for start in pbar:
            chunk = enzyme_pairs[start:start + batch_size]
            ids = [p[0] for p in chunk]
            tokens = enzyme_encoder.tokenize(chunk)

            if isinstance(tokens, dict):
                tokens = {k: v.to(device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(device)

            out = enzyme_encoder(tokens, return_per_residue=False)
            h = out['h'].detach().cpu().float()

            for i, eid in enumerate(ids):
                cache[eid] = h[i].clone()

    return cache

def load_or_build_teacher_pool_cache(
    enzyme_encoder,
    enzyme_pairs: List[Tuple[str, str]],
    cache_dir: str,
    signature: str,
    force_rebuild: bool,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    """加载或构建 teacher pool 缓存。"""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path_for(cache_dir, signature)

    if not force_rebuild and os.path.exists(path):
        print(f"[Preserve] Loading teacher pool cache: {path}")
        cache = torch.load(path, map_location='cpu')
        missing = [eid for eid, _ in enzyme_pairs if eid not in cache]
        if not missing:
            print(f"[Preserve] Cache hit: {len(cache)} enzymes")
            return cache
        else:
            print(f"[Preserve] Cache missing {len(missing)} enzymes "
                  f"(have {len(cache)}, need {len(enzyme_pairs)}), rebuilding...")

    print(f"[Preserve] Building teacher pool cache "
          f"({len(enzyme_pairs)} enzymes, batch={batch_size}) ...")
    cache = build_teacher_pool_cache(
        enzyme_encoder, enzyme_pairs, device, batch_size
    )

    torch.save(cache, path)
    print(f"[Preserve] Saved cache to {path}")

    return cache

def pool_preserve_loss(
    h_student: torch.Tensor,
    h_teacher: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pool-level CPT-preservation loss (cosine).
        L_pool = mean_i [1 - cos(h_student_i, stopgrad(h_teacher_i))]

    Args:
        h_student: [B, D] - SUB model 的 pre-projection mean-pooled hidden
        h_teacher: [B, D] - frozen CPT teacher 的 mean-pooled hidden (来自缓存)

    Returns:
        loss: scalar tensor (可反传)
        mean_cosine: scalar tensor (detached, 仅用于日志)
    """
    s = h_student.float()
    t = h_teacher.float().detach()

    cos = F.cosine_similarity(s, t, dim=-1)
    loss = (1.0 - cos).mean()

    return loss, cos.mean().detach()

def token_preserve_loss(
    per_residue_student: torch.Tensor,
    per_residue_teacher: torch.Tensor,
    residue_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Token-level CPT-preservation loss (cosine).
        L_token = mean over valid tokens [1 - cos(z_s_it, stopgrad(z_t_it))]
    排除 pad / CLS / EOS 等特殊 token (residue_mask=False 处)。

    Args:
        per_residue_student: [B, L, D] raw hidden states
        per_residue_teacher: [B, L, D] raw hidden states
        residue_mask: [B, L] bool, True 表示有效残基

    Returns:
        loss, mean_cosine
    """
    s = per_residue_student.float()
    t = per_residue_teacher.float().detach()

    cos = F.cosine_similarity(s, t, dim=-1)

    mask = residue_mask.float()
    valid_count = mask.sum().clamp(min=1.0)

    loss = ((1.0 - cos) * mask).sum() / valid_count
    mean_cos = (cos * mask).sum() / valid_count

    return loss, mean_cos.detach()