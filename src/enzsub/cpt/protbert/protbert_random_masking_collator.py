
"""
Random (BERT-style) Token Masking Data Collator for ProtBERT-BFD Enzyme CPT.

[与 ESM-2 EnzymeRandomMaskingCollator 完全对齐 2026-05]
- 完全向量化的 mask 选择 + 80/10/10 切分 (torch RNG, 非 Python random)。
- 防御: per-sample 至少 1 个被 mask (与 ESM-2 完全一致)。
- pad_to_multiple_of=8, strict_pad_check, actual_mask_rate_stats (deque)。
- 兼容 (name, seq) / (name, seq, desc) 两种输入; desc 分支与 ESM-2 一致。
- ProtBERT 必要的预处理 (UZOB->X, 空格分隔) 仍保留。

与 ESM-2 collator 的本质差异 (模型决定的):
- 返回 dict 多一个 'attention_mask' (BertForMaskedLM 需要)。
- 用 BertTokenizer 而非 ESM Alphabet batch_converter。
- 序列必须在 tokenize 前做 UZOB->X 与空格分隔。
"""

import torch
import numpy as np
import re
from typing import List, Tuple, Sequence, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)

ExampleType = Union[
    Tuple[str, str],
    Tuple[str, str, torch.Tensor],
]

class ProtBertRandomMaskingCollator:
    """
    ProtBERT-BFD BERT 风格 token-level random masking collator.

    Dynamic padding 安全性:
        pad_idx 在 special_token 集合中, 因此 pad 位置永远不会进入 valid_mask,
        也就不会被 mask、不会写入 label。
    """

    def __init__(
        self,
        tokenizer,
        mask_prob: float = 0.15,
        ignore_index: int = -100,
        max_length: int = 512,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        keep_token_prob: float = 0.1,
        pad_to_multiple_of: int = 8,
        strict_pad_check: bool = True,
    ):
        """
        Args:
            tokenizer: HuggingFace BertTokenizer (Rostlab/prot_bert_bfd)
            mask_prob: 每个 valid token 被选中的概率 (默认 15%)
            ignore_index: Label 中的忽略索引
            max_length: 序列最大长度 (含特殊 token)
            mask_token_prob: 选中 token 用 [MASK] 的概率 (默认 80%)
            random_token_prob: 选中 token 用随机 token 替换的概率 (默认 10%)
            keep_token_prob: 选中 token 保持原样的概率 (默认 10%)
            pad_to_multiple_of: pad 后长度对齐的倍数 (bf16 tensor core 友好)
            strict_pad_check: 是否在每个 batch 断言 pad 位置 label 未被污染。
        """
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

        if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            aligned = ((max_length + pad_to_multiple_of - 1)
                       // pad_to_multiple_of) * pad_to_multiple_of
            logger.warning(
                f"max_length ({max_length}) is not a multiple of "
                f"pad_to_multiple_of ({pad_to_multiple_of}). "
                f"Auto-aligning: {max_length} -> {aligned}."
            )
            max_length = aligned
        self.max_length = max_length

        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.keep_token_prob = keep_token_prob

        assert abs(mask_token_prob + random_token_prob + keep_token_prob - 1.0) < 1e-6, \
            "mask/random/keep probabilities must sum to 1.0"

        self.pad_to_multiple_of = pad_to_multiple_of
        self.strict_pad_check = strict_pad_check

        self.mask_idx = tokenizer.mask_token_id
        self.cls_idx = tokenizer.cls_token_id
        self.sep_idx = tokenizer.sep_token_id
        self.pad_idx = tokenizer.pad_token_id
        self.unk_idx = tokenizer.unk_token_id

        special_set = {
            self.mask_idx, self.cls_idx, self.sep_idx,
            self.pad_idx, self.unk_idx,
        }
        special_set.discard(None)
        self._special_tokens_tensor = torch.tensor(
            sorted(special_set), dtype=torch.long
        )

        self.replacement_tokens = torch.tensor(
            [idx for idx in range(len(tokenizer)) if idx not in special_set],
            dtype=torch.long,
        )

        self.total_calls = 0
        self.actual_mask_rate_stats = deque(maxlen=10000)

    @staticmethod
    def preprocess_sequence(seq: str) -> str:
        """ProtBERT 序列预处理: U/Z/O/B -> X, 空格分隔每个残基。"""
        seq = re.sub(r"[UZOB]", "X", seq.upper())
        return " ".join(list(seq))

    def _get_valid_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Returns a [B, L] bool tensor: True at positions eligible for masking."""
        special = self._special_tokens_tensor
        return ~torch.isin(tokens, special)

    @staticmethod
    def _split_examples(
        examples: Sequence[ExampleType]
    ) -> Tuple[List[Tuple[str, str]], Union[List[torch.Tensor], None]]:
        """检测 examples 是 (n, s) 还是 (n, s, desc); 整 batch 风格必须一致。"""
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
                f"ProtBertRandomMaskingCollator expects (name, seq) or "
                f"(name, seq, desc) tuples; got arity={first_arity}."
            )

    def __call__(self, examples: Sequence[ExampleType]) -> dict:
        """
        Args:
            examples: list of (name, seq) or (name, seq, desc) tuples.
                      seq 是原始氨基酸序列 (无空格), 内部自动预处理。

        Returns:
            {
                'input_ids':      masked tokens [B, L]  (L pad 到 8 的倍数)
                'attention_mask': attention mask [B, L]
                'labels':         labels [B, L]  (未掩码=-100, pad=-100)
                'desc':           desc [B, D]   (仅当 examples 含 desc 时)
            }
        """
        self.total_calls += 1

        ns_pairs, desc_list = self._split_examples(examples)

        raw_sequences = [s for (_, s) in ns_pairs]
        processed_sequences = [self.preprocess_sequence(s) for s in raw_sequences]

        encoded = self.tokenizer(
            processed_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        batch_size, seq_len = input_ids.shape

        labels = torch.full_like(input_ids, self.ignore_index)
        masked_input_ids = input_ids.clone()

        valid_mask = self._get_valid_mask(input_ids)
        rand_select = torch.rand(input_ids.shape) < self.mask_prob
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

        probs = torch.rand(input_ids.shape)
        do_mask_token = mask_select & (probs < self.mask_token_prob)
        do_random_token = (
            mask_select
            & (probs >= self.mask_token_prob)
            & (probs < self.mask_token_prob + self.random_token_prob)
        )

        masked_input_ids[do_mask_token] = self.mask_idx

        n_random = do_random_token.sum().item()
        if n_random > 0:
            rand_idx = torch.randint(0, len(self.replacement_tokens), (n_random,))
            masked_input_ids[do_random_token] = self.replacement_tokens[rand_idx]

        labels[mask_select] = input_ids[mask_select]

        total_masked = mask_select.sum().item()
        total_valid = valid_mask.sum().item()

        if self.strict_pad_check:
            pad_positions = (input_ids == self.pad_idx)
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
            pad_ratio = (input_ids == self.pad_idx).float().mean().item()

            logger.info(
                f"[ProtBERT Random Masking Stats] Calls: {self.total_calls}, "
                f"Target mask rate: {self.mask_prob:.3f}, "
                f"Actual mask rate (recent avg): {avg_rate:.3f}, "
                f"Batch shape: {tuple(input_ids.shape)}, "
                f"Pad ratio: {pad_ratio:.2%}"
                + (f", Desc: yes ({len(desc_list)})" if desc_list is not None else "")
            )

        result = {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        if desc_list is not None:
            desc_tensor = torch.stack(desc_list, dim=0).float()
            assert desc_tensor.dim() == 2 and desc_tensor.shape[0] == batch_size, (
                f"Bad desc tensor shape: {desc_tensor.shape}, expected [{batch_size}, D]"
            )
            result['desc'] = desc_tensor

        return result