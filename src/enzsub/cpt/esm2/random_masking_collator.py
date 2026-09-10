
"""
Random (BERT-style) Masking Data Collator for ESM-2 Enzyme CPT.

与 span_masking_collator.py 的差异:
- 不再按片段成块 mask, 每个 valid token 独立按 mask_prob 概率被选中。
- 80/10/10 (mask / random replace / keep) 与 BERT 一致, 与 span 版本一致。
- 其余基础设施 (dynamic padding + pad_to_multiple_of=8, 80/10/10 向量化, pad
  位置 label 污染断言, 兼容 (name, seq) / (name, seq, desc) 两种输入) 完全沿用
  span 版本的设计。

[Balanced Conservative CPT - descriptor 兼容]
- 当 examples 含 desc 时, 返回 dict 多一个 'desc' key, shape [B, desc_dim],
  dtype float32; 否则保持原始返回。本 collator 不实时计算 descriptor。
"""

import torch
import numpy as np
from typing import List, Tuple, Sequence, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)

ExampleType = Union[
    Tuple[str, str],
    Tuple[str, str, torch.Tensor],
]

class EnzymeRandomMaskingCollator:
    """
    ESM-2 BERT 风格 token-level random masking collator.

    Dynamic padding 安全性:
        pad_idx 在 special_token 集合中, 因此 pad 位置永远不会进入 valid_mask,
        也就不会被 mask、不会写入 label。
    """

    def __init__(
        self,
        alphabet,
        mask_prob: float = 0.15,
        ignore_index: int = -100,
        truncation_len: int = 512,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        keep_token_prob: float = 0.1,
        pad_to_multiple_of: int = 8,
        strict_pad_check: bool = True,
    ):
        """
        Args:
            alphabet: ESM Alphabet
            mask_prob: 每个 valid token 被选中的概率 (默认 15%)
            ignore_index: Label 中的忽略索引
            truncation_len: 序列截断长度 (不含 CLS/EOS)
            mask_token_prob: 选中 token 用 [MASK] 的概率 (默认 80%)
            random_token_prob: 选中 token 用随机 token 替换的概率 (默认 10%)
            keep_token_prob: 选中 token 保持原样的概率 (默认 10%)
            pad_to_multiple_of: pad 后长度对齐的倍数 (bf16 tensor core 友好)
            strict_pad_check: 是否在每个 batch 断言 pad 位置 label 未被污染。
        """
        self.alphabet = alphabet
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.keep_token_prob = keep_token_prob

        assert abs(mask_token_prob + random_token_prob + keep_token_prob - 1.0) < 1e-6, \
            "mask/random/keep probabilities must sum to 1.0"

        self.batch_converter = alphabet.get_batch_converter(
            truncation_seq_length=truncation_len
        )

        self.pad_to_multiple_of = pad_to_multiple_of
        self.strict_pad_check = strict_pad_check

        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.pad_idx = alphabet.padding_idx

        special_set = {
            self.mask_idx, self.cls_idx, self.eos_idx, self.pad_idx,
            getattr(alphabet, "unk_idx", None)
        }
        special_set.discard(None)
        self._special_tokens_tensor = torch.tensor(
            sorted(special_set), dtype=torch.long
        )

        self.replacement_tokens = torch.tensor(
            [idx for idx in range(len(alphabet)) if idx not in special_set],
            dtype=torch.long
        )

        self.total_calls = 0
        self.actual_mask_rate_stats = deque(maxlen=10000)

    def _get_valid_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns a [B, L] bool tensor: True at positions eligible for masking."""
        special = self._special_tokens_tensor
        return ~torch.isin(tokens, special)

    def _pad_to_multiple(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        if self.pad_to_multiple_of is None:
            return batch_tokens
        B, L = batch_tokens.shape
        m = self.pad_to_multiple_of
        if L % m == 0:
            return batch_tokens
        new_L = ((L + m - 1) // m) * m
        pad_amount = new_L - L
        pad_block = batch_tokens.new_full((B, pad_amount), self.pad_idx)
        return torch.cat([batch_tokens, pad_block], dim=1)

    @staticmethod
    def _split_examples(
        examples: Sequence[ExampleType]
    ) -> Tuple[List[Tuple[str, str]], Union[List[torch.Tensor], None]]:
        """检测 examples 是 (n, s) 还是 (n, s, desc); 整个 batch 风格必须一致。"""
        if len(examples) == 0:
            return [], None

        first_arity = len(examples[0])
        if first_arity == 2:
            return list(examples), None
        elif first_arity == 3:
            ns_pairs: List[Tuple[str, str]] = []
            desc_list: List[torch.Tensor] = []
            for e in examples:
                if len(e) != 3:
                    raise ValueError(
                        "Mixed example formats in a single batch: expected all "
                        "(name, seq, desc) triples but got an item of arity "
                        f"{len(e)}."
                    )
                n, s, d = e
                ns_pairs.append((n, s))
                if not isinstance(d, torch.Tensor):
                    d = torch.as_tensor(d, dtype=torch.float32)
                desc_list.append(d)
            return ns_pairs, desc_list
        else:
            raise ValueError(
                f"EnzymeRandomMaskingCollator expects (name, seq) or "
                f"(name, seq, desc) tuples; got arity={first_arity}."
            )

    def __call__(self, examples: Sequence[ExampleType]) -> dict:
        """
        Args:
            examples: list of (name, seq) or (name, seq, desc) tuples.

        Returns:
            {
                'tokens': masked_tokens [B, L],     # L pad 到 8 的倍数
                'labels': labels        [B, L],     # 未掩码=-100, pad=-100
                'desc'  : desc          [B, D],     # 仅当 examples 含 desc 时
            }
        """
        self.total_calls += 1

        ns_pairs, desc_list = self._split_examples(examples)
        names, sequences, batch_tokens = self.batch_converter(ns_pairs)

        batch_tokens = self._pad_to_multiple(batch_tokens)

        batch_size, seq_len = batch_tokens.shape

        labels = torch.full_like(batch_tokens, self.ignore_index)
        masked_tokens = batch_tokens.clone()

        valid_mask = self._get_valid_mask(batch_tokens)
        rand_select = torch.rand(batch_tokens.shape) < self.mask_prob
        mask_select = rand_select & valid_mask

        per_sample_count = mask_select.sum(dim=1)
        for i in range(batch_size):
            if per_sample_count[i].item() == 0:
                valid_positions_i = valid_mask[i].nonzero(as_tuple=True)[0]
                if len(valid_positions_i) > 0:
                    forced_idx = valid_positions_i[
                        torch.randint(0, len(valid_positions_i), (1,)).item()
                    ]
                    mask_select[i, forced_idx] = True

        probs = torch.rand(batch_tokens.shape)

        do_mask_token = mask_select & (probs < self.mask_token_prob)
        do_random_token = (
            mask_select
            & (probs >= self.mask_token_prob)
            & (probs < self.mask_token_prob + self.random_token_prob)
        )

        masked_tokens[do_mask_token] = self.mask_idx

        n_random = do_random_token.sum().item()
        if n_random > 0:
            rand_idx = torch.randint(0, len(self.replacement_tokens), (n_random,))
            masked_tokens[do_random_token] = self.replacement_tokens[rand_idx]

        labels[mask_select] = batch_tokens[mask_select]

        total_masked = mask_select.sum().item()
        total_valid = valid_mask.sum().item()

        if self.strict_pad_check:
            pad_positions = (batch_tokens == self.pad_idx)
            if pad_positions.any():
                assert (labels[pad_positions] == self.ignore_index).all(), (
                    "BUG: pad token positions have non-ignore labels. "
                    "Check special_token set contains pad_idx."
                )

        actual_mask_rate = total_masked / total_valid if total_valid > 0 else 0
        self.actual_mask_rate_stats.append(actual_mask_rate)

        if self.total_calls % 100 == 0:
            recent = list(self.actual_mask_rate_stats)[-100:]
            avg_rate = float(np.mean(recent)) if recent else 0.0
            pad_ratio = (batch_tokens == self.pad_idx).float().mean().item()

            logger.info(
                f"[ESM2 Random Masking Stats] Calls: {self.total_calls}, "
                f"Target mask rate: {self.mask_prob:.3f}, "
                f"Actual mask rate (recent avg): {avg_rate:.3f}, "
                f"Batch shape: {tuple(batch_tokens.shape)}, "
                f"Pad ratio: {pad_ratio:.2%}"
                + (f", Desc: yes ({len(desc_list)})" if desc_list is not None else "")
            )

        result = {
            'tokens': masked_tokens,
            'labels': labels,
        }

        if desc_list is not None:
            desc_tensor = torch.stack(desc_list, dim=0).float()
            assert desc_tensor.dim() == 2 and desc_tensor.shape[0] == batch_size, (
                f"Bad desc tensor shape: {desc_tensor.shape}, expected [{batch_size}, D]"
            )
            result['desc'] = desc_tensor

        return result