
"""
ProtBERT-BFD masked-language-model continued pre-training.

Author: JkBai
Laboratory: AGI&FBHC
"""

import os
import yaml
import contextlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForMaskedLM, BertTokenizer
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import wandb

try:
    from .protbert_random_masking_collator import ProtBertRandomMaskingCollator
except ImportError:
    from protbert_random_masking_collator import ProtBertRandomMaskingCollator

def setup_logging(rank, output_dir):
    log_file = os.path.join(output_dir, f'train_rank{rank}.log')
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - Rank %(rank)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    return logging.getLogger(__name__)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

class ProteinSequenceDataset(torch.utils.data.Dataset):
    """蛋白质序列数据集

    chunk 后的样本是 (name, seq) 或 (name, seq, desc), 取决于是否传 descriptor_map。
    chunk 命名规则: 长度 <= chunk_size 时保留原 name; 否则 f"{name}_chunk_{i}"。
    """
    def __init__(self, sequences, chunk_size=510, descriptor_map=None):
        self.chunks = []
        for idx, item in enumerate(sequences):
            if isinstance(item, tuple):
                name, seq = item
            else:
                name, seq = f"seq_{idx}", item

            seq_len = len(seq)
            if seq_len <= chunk_size:
                self.chunks.append((name, seq))
            else:
                num_chunks = (seq_len + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, seq_len)
                    self.chunks.append((f"{name}_chunk_{i}", seq[start:end]))

        self.descriptor_map = descriptor_map
        if descriptor_map is not None:
            missing = []
            for cname, _ in self.chunks:
                if cname not in descriptor_map:
                    missing.append(cname)
                    if len(missing) >= 5:
                        break
            if missing:
                raise KeyError(
                    f"{len(missing)}+ chunk names missing from descriptor_map. "
                    f"First few: {missing}. "
                    f"This means the precomputed descriptors do not align with the "
                    f"training dataset chunking. Re-run "
                    f"precompute_sequence_descriptors.py with the SAME "
                    f"max_sequence_length as training (chunk_size={chunk_size})."
                )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        name, seq = self.chunks[idx]
        if self.descriptor_map is not None:
            desc = self.descriptor_map[name]
            return (name, seq, desc)
        return (name, seq)

    def get_lengths(self):
        """供 DistributedLengthGroupedSampler 使用, 返回 tokenized 长度 (含 CLS/SEP)。"""
        return [len(chunk_seq) + 2 for (_, chunk_seq) in self.chunks]

class DescriptorHead(nn.Module):
    """Tiny MLP head from pooled hidden state -> desc-dim regression target."""
    def __init__(self, hidden_size: int, desc_dim: int = 41, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, desc_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)

def mean_pool(hidden_states: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over sequence dim. hidden_states: [B, L, H], pool_mask: [B, L] bool."""
    mask = pool_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

def build_pool_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     pad_idx: int, cls_idx: int, sep_idx: int) -> torch.Tensor:
    """Pool mask = not pad AND not cls AND not sep.

    与 ESM-2 build_pool_mask 完全等价:
    - ESM-2 那边是 (tokens != pad) & (tokens != cls) & (tokens != eos)
    - ProtBERT 这边 sep 扮演 eos 的角色
    - 不排除 mask_idx, 因为 masked 位置仍是合法残基位置

    attention_mask 在 ProtBERT 里跟 (input_ids != pad_idx) 等价, 这里用它
    避免对 input_ids 再做一次 elementwise eq (小优化, 语义一致)。
    """
    not_pad = attention_mask.bool()
    not_cls = (input_ids != cls_idx)
    not_sep = (input_ids != sep_idx)
    return not_pad & not_cls & not_sep

class _IndexedTensorMap:
    """Dict-like view over a [N, D] tensor with name -> row lookup.

    设计目的: 让 DataLoader worker spawn 时 pickle 友好。
    - 持有 1 个完整 tensor (1 个 storage / 1 个 FD) 和 1 个 str->int dict (无 FD)。
    - __getitem__ 在访问时 *临时* 生成 row view, view 不持久化, 不累积 FD。
    - 与原先 `Dict[str, Tensor[D]]` 接口完全兼容。
    """
    __slots__ = ('_idx', '_t')

    def __init__(self, name_to_idx, tensor):
        self._idx = name_to_idx
        self._t = tensor

    def __contains__(self, name):
        return name in self._idx

    def __getitem__(self, name):
        return self._t[self._idx[name]]

    def __len__(self):
        return self._t.shape[0]

    def keys(self):
        return self._idx.keys()

def load_descriptor_map(desc_pt_path: str, expected_desc_dim: int = None):
    """Load precomputed descriptor file -> (name -> Tensor[D]) dict-like + meta."""
    if not os.path.exists(desc_pt_path):
        raise FileNotFoundError(
            f"Descriptor file not found: {desc_pt_path}. "
            f"Run precompute_sequence_descriptors.py first."
        )
    payload = torch.load(desc_pt_path, map_location='cpu')
    names = payload['names']
    desc_norm = payload['descriptors_norm']
    feature_names = payload.get('feature_names', None)
    desc_dim = int(payload['desc_dim'])

    if expected_desc_dim is not None and desc_dim != expected_desc_dim:
        raise ValueError(
            f"desc_dim mismatch in {desc_pt_path}: file has {desc_dim}, "
            f"config says {expected_desc_dim}."
        )

    if len(names) != desc_norm.shape[0]:
        raise ValueError(f"names/desc length mismatch in {desc_pt_path}.")

    name_to_idx = payload.get('id_to_index', None)
    if name_to_idx is None:
        name_to_idx = {n: i for i, n in enumerate(names)}

    desc_map = _IndexedTensorMap(name_to_idx=name_to_idx, tensor=desc_norm)

    meta = {
        'desc_dim': desc_dim,
        'num_chunks': len(names),
        'feature_names': feature_names,
        'max_sequence_length': payload.get('max_sequence_length', None),
    }
    return desc_map, meta

def _reduce_eval_metrics(total_loss, total_correct, total_tokens, n_batches, world_size):
    t = torch.tensor(
        [total_loss, total_correct, total_tokens, n_batches],
        dtype=torch.float64, device='cuda'
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, token_sum, batch_sum = t.tolist()
    avg_loss = loss_sum / batch_sum if batch_sum > 0 else 0
    accuracy = correct_sum / token_sum if token_sum > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity, accuracy

@torch.no_grad()
def evaluate_full(model, dataloader, device, world_size, amp_dtype, desc="Validation"):
    """完整 MLM 验证。batch 中 'desc' 字段 (如有) 会被忽略。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = (amp_dtype is not None)

    for batch in tqdm(dataloader, desc=desc, leave=False, disable=(device != 0)):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_fct(logits.permute(0, 2, 1), labels)
        pred = logits.argmax(dim=-1)
        mask = (labels != -100)
        total_correct += ((pred == labels) & mask).sum().item()
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        n_batches += 1

    avg_loss, perplexity, accuracy = _reduce_eval_metrics(
        total_loss, total_correct, total_tokens, n_batches, world_size
    )
    model.train()
    return avg_loss, perplexity, accuracy

@torch.no_grad()
def evaluate_quick(model, dataloader, device, world_size, amp_dtype, max_batches=100):
    """快速 MLM 验证。batch 中 'desc' 字段 (如有) 会被忽略。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    use_amp = (amp_dtype is not None)

    for batch in dataloader:
        if n_batches >= max_batches:
            break
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = loss_fct(logits.permute(0, 2, 1), labels)
        pred = logits.argmax(dim=-1)
        mask = (labels != -100)
        total_correct += ((pred == labels) & mask).sum().item()
        total_loss += loss.item()
        total_tokens += mask.sum().item()
        n_batches += 1

    avg_loss, perplexity, accuracy = _reduce_eval_metrics(
        total_loss, total_correct, total_tokens, n_batches, world_size
    )
    model.train()
    return avg_loss, perplexity, accuracy

def _untie_lm_head_decoder(model, rank=0, logger=None):
    """
    Untie ``cls.predictions.decoder.weight`` from the input embeddings.

    与 ESM-2 _untie_lm_head_weight 等价, 只是 tying 点位置不同:
    - ESM-2: model.lm_head.weight ↔ model.embed_tokens.weight
    - ProtBERT: model.cls.predictions.decoder.weight ↔
                model.bert.embeddings.word_embeddings.weight
    """
    decoder = model.cls.predictions.decoder
    word_emb = model.bert.embeddings.word_embeddings

    if decoder.weight.data_ptr() != word_emb.weight.data_ptr():
        if rank == 0 and logger:
            logger.info("  decoder.weight is already untied; skipping.")
        return False

    new_decoder_weight = nn.Parameter(
        decoder.weight.detach().clone(),
        requires_grad=decoder.weight.requires_grad,
    )
    decoder.weight = new_decoder_weight

    model.tie_weights = lambda: None

    if rank == 0 and logger:
        logger.info("  [Untie] cls.predictions.decoder.weight has been untied from "
                    "word_embeddings.weight.")
        logger.info("  [Untie] model.tie_weights() monkey-patched to no-op.")
    return True

def freeze_protbert_layers(model, unfreeze_last_n_layers, unfreeze_lm_head=False,
                             rank=0, logger=None):
    for param in model.parameters():
        param.requires_grad = False

    total_layers = len(model.bert.encoder.layer)
    start_layer = total_layers
    if unfreeze_last_n_layers > 0:
        start_layer = max(0, total_layers - unfreeze_last_n_layers)
        for i in range(start_layer, total_layers):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

    lm_head_trainable = 0
    untied = False
    if unfreeze_lm_head:
        if rank == 0 and logger:
            logger.info("Unfreezing LM head after weight untie...")

        untied = _untie_lm_head_decoder(model, rank=rank, logger=logger)

        for param in model.cls.parameters():
            param.requires_grad = True

        word_emb_w = model.bert.embeddings.word_embeddings.weight
        decoder_w = model.cls.predictions.decoder.weight
        if word_emb_w.data_ptr() == decoder_w.data_ptr():
            raise RuntimeError(
                "Untie failed: word_embeddings.weight and decoder.weight still "
                "share storage."
            )
        assert not word_emb_w.requires_grad, (
            "word_embeddings.weight should remain frozen after untie."
        )

        lm_head_trainable = sum(
            p.numel() for p in model.cls.parameters() if p.requires_grad
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if logger and rank == 0:
        logger.info("ProtBERT-BFD Layer Freezing:")
        logger.info(f"  Total encoder layers: {total_layers}")
        if unfreeze_last_n_layers > 0:
            logger.info(
                f"  Unfrozen encoder layers: {start_layer} ~ {total_layers - 1} "
                f"({unfreeze_last_n_layers} layers)"
            )
        else:
            logger.info("  Unfrozen encoder layers: 0")
        logger.info(f"  LM head (model.cls): {'UNFROZEN' if unfreeze_lm_head else 'FROZEN'}")
        if unfreeze_lm_head:
            logger.info(f"    LM head trainable params: {lm_head_trainable/1e6:.4f}M")
            logger.info(f"    Untie succeeded: {untied}")
            logger.info(
                f"    word_embeddings.weight.requires_grad = "
                f"{model.bert.embeddings.word_embeddings.weight.requires_grad}"
            )
        logger.info(
            f"  Trainable params: {trainable_params/1e6:.2f}M / "
            f"{total_params/1e6:.2f}M "
            f"({100*trainable_params/total_params:.1f}%)"
        )

    return trainable_params, total_params

def resolve_precision(train_cfg, rank=0, logger=None):
    if 'precision' in train_cfg:
        precision = train_cfg['precision']
    elif train_cfg.get('use_mixed_precision', True):
        precision = 'bf16'
        if rank == 0 and logger:
            logger.warning("legacy use_mixed_precision -> precision='bf16'")
    else:
        precision = 'fp32'

    if precision == 'bf16':
        return torch.bfloat16, False
    elif precision == 'fp16':
        return torch.float16, True
    elif precision == 'fp32':
        return None, False
    else:
        raise ValueError(f"Unknown precision: {precision}")

def read_fasta(fasta_path):
    sequences = []
    current_name = None
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None:
                    sequences.append((current_name, ''.join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        sequences.append((current_name, ''.join(current_seq)))
    return sequences

def train_protbert_random_cpt(rank, world_size, config_path):
    setup_ddp(rank, world_size)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths_cfg = config['paths']
    train_cfg = config['training']

    mask_cfg = train_cfg.get('random_masking', train_cfg.get('masking', {}))
    wandb_cfg = config.get('wandb', {})

    bcfg = train_cfg.get('balanced_cpt', {}) or {}
    balanced_enabled = bool(bcfg.get('enabled', False))
    desc_cfg = bcfg.get('descriptor', {}) or {}

    desc_enabled = balanced_enabled and bool(desc_cfg.get('enabled', True))
    desc_weight = float(desc_cfg.get('weight', 0.05))
    desc_dim = int(desc_cfg.get('dim', 41))
    desc_dropout = float(desc_cfg.get('dropout', 0.1))

    if bcfg.get('anchor', {}).get('enabled', False):
        raise ValueError(
            "balanced_cpt.anchor.enabled=true is set, but this trainer has the "
            "teacher/anchor mechanism removed. Either set anchor.enabled=false "
            "or use the span_cpt trainer."
        )

    output_dir = paths_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(rank, output_dir)

    resume_from = train_cfg.get('resume_from', None)
    if resume_from is None and train_cfg.get('auto_resume', True):
        checkpoint_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if (f.startswith('checkpoint_epoch_') or f.startswith('checkpoint_step_')) and f.endswith('.pt'):
                    checkpoint_files.append(os.path.join(output_dir, f))
        if checkpoint_files:
            resume_from = max(checkpoint_files, key=os.path.getctime)
            if rank == 0:
                logger.info(f"Found existing checkpoint: {resume_from}")

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    checkpoint = None

    if resume_from and os.path.exists(resume_from):
        if rank == 0:
            logger.info("=" * 60)
            logger.info(f"Resuming from checkpoint: {resume_from}")
            logger.info("=" * 60)
        checkpoint = torch.load(resume_from, map_location=f'cuda:{rank}')
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        best_val_ppl = checkpoint.get('best_val_ppl', checkpoint.get('val_perplexity', float('inf')))
        if rank == 0:
            logger.info(f"Resuming from:")
            logger.info(f"  Epoch: {start_epoch}")
            logger.info(f"  Global step: {global_step}")
            logger.info(f"  Best val loss: {best_val_loss:.4f}")
            logger.info(f"  Best val ppl: {best_val_ppl:.2f}")
    else:
        if rank == 0:
            logger.info("Starting training from scratch")

    if rank == 0 and wandb_cfg.get('enabled', True):
        run_name = wandb_cfg.get('run_name', 'protbert_random_cpt')
        wandb_id = checkpoint.get('wandb_id', None) if checkpoint else None

        if wandb_id:
            logger.info(f"Resuming WandB run: {wandb_id}")
            wandb.init(
                project=wandb_cfg.get('project', 'enzyme-cpt-protbert'),
                id=wandb_id, resume="must",
            )
        else:
            run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=wandb_cfg.get('project', 'enzyme-cpt-protbert'),
                name=run_name,
                config={
                    'model': paths_cfg.get('model_name',
                                            paths_cfg.get('model_checkpoint', 'unknown')),
                    'model_type': 'ProtBERT-BFD',
                    'masking': 'random_bert_style',
                    'world_size': world_size,
                    'batch_size_per_gpu': train_cfg['batch_size'],
                    'gradient_accumulation': train_cfg.get('gradient_accumulation_steps', 1),
                    'effective_batch_size': train_cfg['batch_size'] * world_size * train_cfg.get('gradient_accumulation_steps', 1),
                    'num_epochs': train_cfg['num_epochs'],
                    'learning_rate': train_cfg['optimizer']['lr'],
                    'random_masking': mask_cfg,
                    'unfreeze_layers': train_cfg.get('unfreeze_last_n_layers', 6),
                    'unfreeze_lm_head': train_cfg.get('unfreeze_lm_head', False),
                    'lm_head_untied': train_cfg.get('unfreeze_lm_head', False),
                    'eval_steps': train_cfg.get('eval_steps', 5000),
                    'quick_eval_batches': train_cfg.get('quick_eval_batches', 200),
                    'precision': train_cfg.get('precision', 'bf16'),
                    'length_bucketing': True,
                    'balanced_cpt_enabled': balanced_enabled,
                    'desc_enabled': desc_enabled,
                    'desc_weight': desc_weight,
                    'desc_dim': desc_dim,
                },
                tags=wandb_cfg.get('tags', ['protbert', 'random_masking', 'cpt']),
                notes=wandb_cfg.get('notes', 'ProtBERT-BFD Random Masking CPT with DDP'),
            )
        logger.info(f"✓ WandB initialized: {wandb.run.url}")

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ProtBERT-BFD Random Masking CPT Training with DDP + WandB")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Random mask config: {mask_cfg}")
        logger.info("Balanced Conservative CPT (anchor REMOVED in this trainer):")
        logger.info(f"  enabled        : {balanced_enabled}")
        logger.info(f"  desc.enabled   : {desc_enabled}  weight={desc_weight}  dim={desc_dim}")

    model_name = paths_cfg.get('model_name', paths_cfg.get('model_checkpoint'))
    if rank == 0:
        logger.info(f"Loading ProtBERT-BFD model: {model_name}")

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

    model = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
    model = model.cuda(rank)

    student_hidden_size = model.config.hidden_size
    student_num_layers = model.config.num_hidden_layers

    if rank == 0:
        logger.info(f"✓ Model loaded")
        logger.info(f"  Vocab size: {tokenizer.vocab_size}")
        logger.info(f"  Encoder layers: {student_num_layers}")
        logger.info(f"  Hidden size: {student_hidden_size}")
        logger.info(f"  Max position embeddings: {model.config.max_position_embeddings}")
        logger.info(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    max_seq_len = train_cfg.get('max_sequence_length', 512)
    max_length_with_special = max_seq_len + 2
    PAD_MULTIPLE = 8
    aligned_max_length = (
        (max_length_with_special + PAD_MULTIPLE - 1) // PAD_MULTIPLE
    ) * PAD_MULTIPLE
    max_positions = model.config.max_position_embeddings
    if aligned_max_length > max_positions:
        raise ValueError(
            f"aligned max_length ({aligned_max_length}) exceeds ProtBERT max positions "
            f"({max_positions}). Reduce max_sequence_length to "
            f"{max_positions - 2 - PAD_MULTIPLE + 1}."
        )

    if rank == 0:
        logger.info(f"Sequence length config:")
        logger.info(f"  max_sequence_length (aa): {max_seq_len}")
        logger.info(f"  max_length with CLS/SEP: {max_length_with_special}")
        logger.info(f"  aligned (×{PAD_MULTIPLE}): {aligned_max_length}")

    unfreeze_n = train_cfg.get('unfreeze_last_n_layers', 6)
    unfreeze_lm_head = train_cfg.get('unfreeze_lm_head', False)
    trainable_params, total_params = freeze_protbert_layers(
        model, unfreeze_n,
        unfreeze_lm_head=unfreeze_lm_head,
        rank=rank, logger=logger
    )

    if rank == 0 and wandb_cfg.get('enabled', True):
        wandb.config.update({
            'trainable_params_M': trainable_params / 1e6,
            'total_params_M': total_params / 1e6,
            'trainable_ratio': trainable_params / total_params,
            'num_hidden_layers': student_num_layers,
            'hidden_size': student_hidden_size,
        })

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if checkpoint is not None:
        if rank == 0:
            logger.info("Loading model weights from checkpoint...")
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if rank == 0:
            logger.info("✓ Model weights loaded (strict=False for untie compat)")

    desc_head = None
    if desc_enabled:
        desc_head = DescriptorHead(
            hidden_size=student_hidden_size,
            desc_dim=desc_dim,
            dropout=desc_dropout,
        ).cuda(rank)
        desc_head = DDP(desc_head, device_ids=[rank], find_unused_parameters=False)
        if rank == 0:
            n_dh = sum(p.numel() for p in desc_head.parameters() if p.requires_grad)
            logger.info(f"DescriptorHead created: {n_dh/1e6:.4f}M trainable params, "
                        f"hidden={student_hidden_size} -> desc_dim={desc_dim}")

        if checkpoint is not None and 'desc_head_state_dict' in checkpoint:
            try:
                desc_head.module.load_state_dict(checkpoint['desc_head_state_dict'])
                if rank == 0:
                    logger.info("✓ DescriptorHead weights loaded from checkpoint.")
            except (RuntimeError, KeyError) as e:
                if rank == 0:
                    logger.warning(
                        f"Failed to load desc_head state ({e}); "
                        f"keeping freshly initialized weights."
                    )
        elif checkpoint is not None:
            if rank == 0:
                logger.warning(
                    "Checkpoint has no desc_head_state_dict (likely a vanilla CPT "
                    "checkpoint). DescriptorHead is initialized from scratch."
                )

    data_collator = ProtBertRandomMaskingCollator(
        tokenizer,
        mask_prob=mask_cfg.get('mask_prob', 0.15),
        mask_token_prob=mask_cfg.get('mask_token_prob', 0.8),
        random_token_prob=mask_cfg.get('random_token_prob', 0.1),
        keep_token_prob=mask_cfg.get('keep_token_prob', 0.1),
        max_length=max_length_with_special,
        pad_to_multiple_of=PAD_MULTIPLE,
    )

    if rank == 0:
        logger.info(f"Random Masking Configuration:")
        logger.info(f"  - mask_prob: {data_collator.mask_prob}")
        logger.info(f"  - mask_token_prob: {data_collator.mask_token_prob}")
        logger.info(f"  - random_token_prob: {data_collator.random_token_prob}")
        logger.info(f"  - keep_token_prob: {data_collator.keep_token_prob}")
        logger.info(f"  - max_length (with special, aligned): {data_collator.max_length}")
        logger.info(f"  - pad_to_multiple_of: {PAD_MULTIPLE}")

    pad_idx = tokenizer.pad_token_id
    cls_idx = tokenizer.cls_token_id
    sep_idx = tokenizer.sep_token_id

    if rank == 0:
        logger.info("Loading training/validation FASTA...")

    train_data = read_fasta(paths_cfg['train_fasta_file'])
    val_data = read_fasta(paths_cfg['val_fasta_file'])

    if rank == 0:
        logger.info(f"Train sequences (raw): {len(train_data):,}")
        logger.info(f"Val sequences (raw):   {len(val_data):,}")

    train_desc_map = None
    val_desc_map = None
    if desc_enabled:
        train_desc_path = paths_cfg.get('train_descriptor_file', None)
        val_desc_path = paths_cfg.get('val_descriptor_file', None)
        if not train_desc_path or not val_desc_path:
            raise ValueError(
                "balanced_cpt.descriptor.enabled=true but "
                "paths.train_descriptor_file / paths.val_descriptor_file are missing."
            )
        if rank == 0:
            logger.info(f"Loading train descriptors: {train_desc_path}")
        train_desc_map, train_desc_meta = load_descriptor_map(train_desc_path, expected_desc_dim=desc_dim)
        if rank == 0:
            logger.info(f"  Train descriptor chunks : {train_desc_meta['num_chunks']:,}")
            logger.info(f"  Train descriptor dim    : {train_desc_meta['desc_dim']}")
            logger.info(f"  Loading val descriptors : {val_desc_path}")
        val_desc_map, val_desc_meta = load_descriptor_map(val_desc_path, expected_desc_dim=desc_dim)
        if rank == 0:
            logger.info(f"  Val descriptor chunks : {val_desc_meta['num_chunks']:,}")

    train_dataset = ProteinSequenceDataset(
        train_data, chunk_size=max_seq_len, descriptor_map=train_desc_map
    )
    val_dataset = ProteinSequenceDataset(
        val_data, chunk_size=max_seq_len, descriptor_map=val_desc_map
    )

    if rank == 0:
        logger.info(f"Train chunks: {len(train_dataset):,}")
        logger.info(f"Val chunks:   {len(val_dataset):,}")
        if desc_enabled:
            if len(train_dataset) != len(train_desc_map):
                raise ValueError(
                    f"Train chunk count ({len(train_dataset)}) != "
                    f"train descriptor count ({len(train_desc_map)}). "
                    f"Re-run precompute_sequence_descriptors.py with the same "
                    f"max_sequence_length={max_seq_len}."
                )
            if len(val_dataset) != len(val_desc_map):
                raise ValueError(
                    f"Val chunk count ({len(val_dataset)}) != "
                    f"val descriptor count ({len(val_desc_map)})."
                )
            logger.info("✓ Chunk count vs descriptor count: aligned.")

    if rank == 0:
        logger.info("Computing chunk lengths for bucketing...")

    train_lengths = train_dataset.get_lengths()

    if rank == 0:
        logger.info(
            f"Length stats: "
            f"min={min(train_lengths)}, max={max(train_lengths)}, "
            f"mean={np.mean(train_lengths):.1f}, "
            f"median={int(np.median(train_lengths))}, "
            f"p95={int(np.percentile(train_lengths, 95))}, "
            f"p99={int(np.percentile(train_lengths, 99))}"
        )

    train_sampler = DistributedLengthGroupedSampler(
        batch_size=train_cfg['batch_size'],
        dataset=train_dataset,
        lengths=train_lengths,
        model_input_name=None,
        num_replicas=world_size,
        rank=rank,
        seed=train_cfg.get('seed', 42),
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size, rank=rank, shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=train_cfg.get('num_workers', 8),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=train_cfg.get('prefetch_factor', 4),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get('eval_batch_size', train_cfg['batch_size']),
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=train_cfg.get('num_workers', 2),
        pin_memory=True,
    )

    optimizer_cfg = train_cfg['optimizer']
    optim_params = [p for p in model.parameters() if p.requires_grad]
    if desc_head is not None:
        optim_params += [p for p in desc_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=float(optimizer_cfg['lr']),
        betas=tuple(optimizer_cfg['betas']),
        eps=float(optimizer_cfg['eps']),
        weight_decay=float(optimizer_cfg['weight_decay'])
    )

    if rank == 0:
        n_optim = sum(p.numel() for p in optim_params)
        logger.info(f"Optimizer: AdamW with {n_optim/1e6:.2f}M trainable params total "
                    f"(student + desc_head)")

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        if rank == 0:
            logger.info("Loading optimizer state from checkpoint...")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if rank == 0:
                logger.info("✓ Optimizer state loaded")
        except (ValueError, KeyError) as e:
            if rank == 0:
                logger.warning(
                    f"Failed to load optimizer state ({e}); "
                    f"likely due to parameter list mismatch. "
                    f"Optimizer re-initialized from scratch (weights preserved)."
                )

    num_epochs = train_cfg['num_epochs']
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = int(num_training_steps * train_cfg.get('warmup_ratio', 0.05))

    if train_cfg.get('scheduler') == 'cosine':
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    if checkpoint is not None and 'scheduler_state_dict' in checkpoint:
        if rank == 0:
            logger.info("Loading scheduler state from checkpoint...")
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if rank == 0:
            logger.info("✓ Scheduler state loaded")

    amp_dtype, use_scaler = resolve_precision(train_cfg, rank=rank, logger=logger)
    use_amp = (amp_dtype is not None)

    if rank == 0:
        precision_name = (
            'bf16' if amp_dtype == torch.bfloat16
            else 'fp16' if amp_dtype == torch.float16
            else 'fp32'
        )
        logger.info(f"Precision: {precision_name} (use_scaler={use_scaler})")

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if use_scaler and checkpoint is not None and 'scaler_state_dict' in checkpoint:
        if rank == 0:
            logger.info("Loading scaler state from checkpoint...")
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if rank == 0:
            logger.info("✓ Scaler state loaded")

    mlm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    desc_loss_fct = torch.nn.SmoothL1Loss()

    eval_steps = train_cfg.get('eval_steps', 5000)
    quick_eval_batches = train_cfg.get('quick_eval_batches', 200)
    use_quick_eval = quick_eval_batches is not None

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Training Configuration:")
        logger.info(f"  Model: {model_name} ({student_num_layers}L, {student_hidden_size}H)")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size per GPU: {train_cfg['batch_size']}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {train_cfg['batch_size'] * world_size * gradient_accumulation_steps}")
        logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
        logger.info(f"  Total steps: {num_training_steps:,}")
        logger.info(f"  Warmup steps: {num_warmup_steps:,}")
        logger.info(f"  Learning rate: {optimizer_cfg['lr']}")
        logger.info(f"  Scheduler: {train_cfg.get('scheduler', 'linear')}")
        logger.info(f"  Gradient clip norm: {train_cfg.get('gradient_clip_norm', 'None')}")
        logger.info(f"  Length bucketing: ENABLED")
        logger.info(f"  Unfreeze LM head: {unfreeze_lm_head}")
        logger.info(f"  Balanced CPT enabled: {balanced_enabled}")
        if balanced_enabled:
            logger.info(f"    desc weight: {desc_weight}  (enabled={desc_enabled})")
        logger.info("")
        logger.info("Evaluation Configuration:")
        logger.info(f"  Eval every {eval_steps} steps")
        if use_quick_eval:
            logger.info(f"  Quick eval: {quick_eval_batches} batches")
        else:
            logger.info(f"  Full eval during training")
        logger.info(f"  Full eval at end of each epoch")
        logger.info("=" * 60)

    running_total = 0.0
    running_mlm = 0.0
    running_desc = 0.0
    running_acc = 0.0
    log_steps = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if desc_head is not None:
            desc_head.train()

        train_sampler.set_epoch(epoch)

        epoch_total_loss = 0.0
        epoch_mlm_loss = 0.0
        epoch_desc_loss = 0.0
        epoch_correct = 0
        epoch_tokens = 0

        if rank == 0:
            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
        else:
            pbar = enumerate(train_loader)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in pbar:
            input_ids = batch['input_ids'].cuda(rank, non_blocking=True)
            attention_mask = batch['attention_mask'].cuda(rank, non_blocking=True)
            labels = batch['labels'].cuda(rank, non_blocking=True)
            desc_target = None
            if desc_enabled:
                if 'desc' not in batch:
                    raise RuntimeError(
                        "desc_enabled=True but batch has no 'desc'. "
                        "Check Dataset / Collator wiring."
                    )
                desc_target = batch['desc'].cuda(rank, non_blocking=True)
                if step == 0 and rank == 0:
                    logger.info(f"  batch desc shape: {tuple(desc_target.shape)} "
                                f"(expected [B={input_ids.shape[0]}, D={desc_dim}])")
                    assert desc_target.shape == (input_ids.shape[0], desc_dim), \
                        f"bad desc shape {desc_target.shape}"

            is_accumulating = ((step + 1) % gradient_accumulation_steps != 0)
            sync_context = model.no_sync() if is_accumulating else contextlib.nullcontext()

            with sync_context:
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=desc_enabled,
                    )
                    logits = outputs.logits

                    mlm_loss = mlm_loss_fct(logits.permute(0, 2, 1), labels)

                    if desc_enabled:

                        hidden = outputs.hidden_states[-1]
                        pool_mask = build_pool_mask(
                            input_ids, attention_mask,
                            pad_idx, cls_idx, sep_idx
                        )
                        z_student = mean_pool(hidden, pool_mask)
                        desc_pred = desc_head(z_student)
                        desc_loss = desc_loss_fct(desc_pred.float(), desc_target.float())
                    else:
                        desc_loss = torch.zeros((), device=input_ids.device)

                    total_loss = mlm_loss + desc_weight * desc_loss

                loss = total_loss / gradient_accumulation_steps

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = (labels != -100)
                correct = ((pred == labels) & mask).sum().item()
                n_tokens = mask.sum().item()

                batch_total = total_loss.item()
                batch_mlm = mlm_loss.item()
                batch_desc = desc_loss.item() if desc_enabled else 0.0
                batch_acc = correct / n_tokens if n_tokens > 0 else 0

                epoch_total_loss += batch_total
                epoch_mlm_loss += batch_mlm
                epoch_desc_loss += batch_desc
                epoch_correct += correct
                epoch_tokens += n_tokens

                running_total += batch_total
                running_mlm += batch_mlm
                running_desc += batch_desc
                running_acc += batch_acc
                log_steps += 1

            if not is_accumulating:
                if use_scaler:
                    if train_cfg.get('gradient_clip_norm'):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in optim_params], train_cfg['gradient_clip_norm']
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if train_cfg.get('gradient_clip_norm'):
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in optim_params], train_cfg['gradient_clip_norm']
                        )
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if rank == 0:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'tot': f"{batch_total:.4f}",
                            'mlm': f"{batch_mlm:.4f}",
                            'desc': f"{batch_desc:.3f}" if desc_enabled else "off",
                            'acc': f"{batch_acc:.3f}",
                            'lr': f"{current_lr:.2e}",
                            'L': input_ids.shape[1],
                        })

                log_interval = train_cfg.get('log_interval', 50)
                if rank == 0 and global_step % log_interval == 0 and wandb_cfg.get('enabled', True):
                    avg_total = running_total / log_steps
                    avg_mlm = running_mlm / log_steps
                    avg_desc = running_desc / log_steps
                    avg_acc = running_acc / log_steps
                    avg_ppl_mlm = np.exp(avg_mlm) if avg_mlm < 20 else float('inf')

                    if len(data_collator.actual_mask_rate_stats) > 0:
                        recent_rates = list(data_collator.actual_mask_rate_stats)[-100:]
                        avg_actual_rate = float(np.mean(recent_rates))
                    else:
                        avg_actual_rate = 0.0

                    wandb.log({
                        'train/total_loss': avg_total,
                        'train/mlm_loss': avg_mlm,
                        'train/desc_loss': avg_desc,
                        'train/accuracy': avg_acc,
                        'train/perplexity_mlm': avg_ppl_mlm,
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch + (step + 1) / len(train_loader),
                        'train/global_step': global_step,
                        'train/batch_seq_len': input_ids.shape[1],
                        'mask/actual_rate': avg_actual_rate,
                        'mask/target_rate': data_collator.mask_prob,
                    }, step=global_step)

                    running_total = 0.0
                    running_mlm = 0.0
                    running_desc = 0.0
                    running_acc = 0.0
                    log_steps = 0

                save_every_n_steps = train_cfg.get('save_every_n_steps', None)
                if save_every_n_steps and global_step % save_every_n_steps == 0:
                    if rank == 0:
                        ckpt_path = os.path.join(output_dir, f'checkpoint_step_{global_step}.pt')
                        ckpt_data = {
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict(),
                            'best_val_loss': best_val_loss,
                            'best_val_ppl': best_val_ppl,
                            'config': config,
                            'lm_head_untied': unfreeze_lm_head,
                            'balanced_cpt': {
                                'enabled': balanced_enabled,
                                'desc_enabled': desc_enabled,
                                'desc_weight': desc_weight,
                                'desc_dim': desc_dim,
                            },
                        }
                        if desc_head is not None:
                            ckpt_data['desc_head_state_dict'] = desc_head.module.state_dict()
                        if use_scaler:
                            ckpt_data['scaler_state_dict'] = scaler.state_dict()
                        if wandb_cfg.get('enabled', True) and wandb.run is not None:
                            ckpt_data['wandb_id'] = wandb.run.id

                        torch.save(ckpt_data, ckpt_path)
                        logger.info(f"✓ Step checkpoint saved: {ckpt_path}")

                    if world_size > 1:
                        dist.barrier()

                if global_step % eval_steps == 0:
                    if rank == 0:
                        logger.info(f"\n{'='*60}")
                        logger.info(f"Periodic Evaluation at Step {global_step}")
                        logger.info(f"{'='*60}")

                    if use_quick_eval:
                        eval_loss, eval_ppl, eval_acc = evaluate_quick(
                            model.module, val_loader, rank, world_size, amp_dtype,
                            max_batches=quick_eval_batches
                        )
                        eval_type = "quick"
                    else:
                        eval_loss, eval_ppl, eval_acc = evaluate_full(
                            model.module, val_loader, rank, world_size, amp_dtype,
                            desc=f"Eval@Step{global_step}"
                        )
                        eval_type = "full"

                    if rank == 0:
                        logger.info(f"Step {global_step} Validation ({eval_type}, MLM-only):")
                        logger.info(f"  Loss (MLM): {eval_loss:.4f}")
                        logger.info(f"  Perplexity: {eval_ppl:.2f}")
                        logger.info(f"  Accuracy: {eval_acc:.4f}")
                        logger.info(f"{'='*60}\n")

                        if wandb_cfg.get('enabled', True):
                            wandb.log({
                                f'eval_{eval_type}/loss': eval_loss,
                                f'eval_{eval_type}/perplexity': eval_ppl,
                                f'eval_{eval_type}/accuracy': eval_acc,
                                f'eval_{eval_type}/step': global_step,
                            }, step=global_step)

                    if world_size > 1:
                        dist.barrier()

        if rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{num_epochs} Training Summary:")
            logger.info(f"{'='*60}")

        avg_total = epoch_total_loss / max(len(train_loader), 1)
        avg_mlm = epoch_mlm_loss / max(len(train_loader), 1)
        avg_desc = epoch_desc_loss / max(len(train_loader), 1)
        train_accuracy = epoch_correct / epoch_tokens if epoch_tokens > 0 else 0
        train_ppl = np.exp(avg_mlm) if avg_mlm < 20 else float('inf')

        if rank == 0:
            logger.info(f"  Total loss: {avg_total:.4f}")
            logger.info(f"  MLM loss  : {avg_mlm:.4f}  (ppl={train_ppl:.2f})")
            logger.info(f"  Desc loss : {avg_desc:.4f}  (weighted={desc_weight*avg_desc:.4f})")
            logger.info(f"  Accuracy  : {train_accuracy:.4f}")
            logger.info(f"  Total steps: {global_step}")
            logger.info(f"{'='*60}")
            logger.info("Running full validation...")

        val_loss, val_ppl, val_acc = evaluate_full(
            model.module, val_loader, rank, world_size, amp_dtype,
            desc=f"Epoch{epoch+1} Validation"
        )

        if rank == 0:
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} Full Validation (MLM-only):")
            logger.info(f"  Loss: {val_loss:.4f}")
            logger.info(f"  Perplexity: {val_ppl:.2f}")
            logger.info(f"  Accuracy: {val_acc:.4f}")

            if wandb_cfg.get('enabled', True):
                wandb.log({
                    'val_epoch/loss': val_loss,
                    'val_epoch/perplexity': val_ppl,
                    'val_epoch/accuracy': val_acc,
                    'val_epoch/epoch': epoch + 1,
                    'epoch': epoch + 1,
                }, step=global_step)

                wandb.log({
                    'epoch_summary/train_total_loss': avg_total,
                    'epoch_summary/train_mlm_loss': avg_mlm,
                    'epoch_summary/train_desc_loss': avg_desc,
                    'epoch_summary/train_ppl_mlm': train_ppl,
                    'epoch_summary/train_acc': train_accuracy,
                    'epoch_summary/val_loss': val_loss,
                    'epoch_summary/val_ppl': val_ppl,
                    'epoch_summary/val_acc': val_acc,
                }, step=epoch + 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl

                save_path = os.path.join(output_dir, 'best_model.pt')
                ckpt_data = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl,
                    'val_accuracy': val_acc,
                    'config': config,
                    'lm_head_untied': unfreeze_lm_head,
                    'balanced_cpt': {
                        'enabled': balanced_enabled,
                        'desc_enabled': desc_enabled,
                        'desc_weight': desc_weight,
                        'desc_dim': desc_dim,
                    },
                }
                if desc_head is not None:
                    ckpt_data['desc_head_state_dict'] = desc_head.module.state_dict()
                torch.save(ckpt_data, save_path)
                logger.info(f"✓ Saved best model to {save_path}")
                logger.info(f"  Best val loss: {best_val_loss:.4f}, ppl: {best_val_ppl:.2f}")

                hf_save_dir = os.path.join(output_dir, 'best_model_hf')
                model.module.save_pretrained(hf_save_dir)
                tokenizer.save_pretrained(hf_save_dir)
                logger.info(f"✓ Saved HuggingFace format to {hf_save_dir}")
                if unfreeze_lm_head:
                    logger.info(
                        f"  decoder.weight was untied during training; "
                        f"config.tie_word_embeddings may still be True in config.json "
                        f"— set it to False manually if you want downstream loaders "
                        f"to respect the untie."
                    )

            if (epoch + 1) % train_cfg.get('save_every_n_epochs', 1) == 0:
                ckpt_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                ckpt_data = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl,
                    'val_accuracy': val_acc,
                    'best_val_loss': best_val_loss,
                    'best_val_ppl': best_val_ppl,
                    'config': config,
                    'lm_head_untied': unfreeze_lm_head,
                    'balanced_cpt': {
                        'enabled': balanced_enabled,
                        'desc_enabled': desc_enabled,
                        'desc_weight': desc_weight,
                        'desc_dim': desc_dim,
                    },
                }
                if desc_head is not None:
                    ckpt_data['desc_head_state_dict'] = desc_head.module.state_dict()
                if use_scaler:
                    ckpt_data['scaler_state_dict'] = scaler.state_dict()
                if wandb_cfg.get('enabled', True) and wandb.run is not None:
                    ckpt_data['wandb_id'] = wandb.run.id

                torch.save(ckpt_data, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        logger.info("\n" + "=" * 60)
        logger.info("Training Completed!")
        logger.info("=" * 60)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best validation perplexity: {best_val_ppl:.2f}")
        logger.info(f"Total training steps: {global_step}")
        logger.info(f"Output directory: {output_dir}")

        final_path = os.path.join(output_dir, 'final_model.pt')
        torch.save(model.module.state_dict(), final_path)
        logger.info(f"✓ Final ProtBERT encoder saved to {final_path}")

        hf_final_dir = os.path.join(output_dir, 'final_model_hf')
        model.module.save_pretrained(hf_final_dir)
        tokenizer.save_pretrained(hf_final_dir)
        logger.info(f"✓ Final model (HF format) saved to {hf_final_dir}")

        if desc_head is not None:
            dh_path = os.path.join(output_dir, 'final_desc_head.pt')
            torch.save({
                'desc_head_state_dict': desc_head.module.state_dict(),
                'desc_dim': desc_dim,
                'hidden_size': student_hidden_size,
                'dropout': desc_dropout,
            }, dh_path)
            logger.info(f"✓ Final desc_head saved to {dh_path}")

        if wandb_cfg.get('enabled', True):
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_val_perplexity'] = best_val_ppl
            wandb.run.summary['total_steps'] = global_step
            wandb.run.summary['total_epochs'] = num_epochs
            wandb.finish()
            logger.info("✓ WandB run finished")

    cleanup_ddp()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ProtBERT-BFD Random Masking CPT with DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--world_size', type=int, default=2,
                        help='Number of GPUs (ignored if --gpus is provided)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma separated GPU ids, e.g. "0,1"')
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = [g.strip() for g in args.gpus.split(',') if g.strip()]
        if not gpu_ids:
            raise ValueError("No valid GPU ids provided via --gpus")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
        world_size = len(gpu_ids)
    else:
        world_size = args.world_size

    print("=" * 60)
    print("ProtBERT-BFD Random Masking CPT Training")
    print("with DDP + WandB + Length Bucketing + bf16 + Untied LM Head")
    print("+ Optional descriptor regression aux loss")
    print("=" * 60)
    print(f"Config: {args.config}")
    if args.gpus:
        print(f"Using GPUs (CUDA_VISIBLE_DEVICES): {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"World size: {world_size} GPU(s) for DDP")
    print("=" * 60)

    import socket
    with socket.socket() as s:
        s.bind(('', 0))
        os.environ['MASTER_PORT'] = str(s.getsockname()[1])
    print(f"MASTER_PORT (auto): {os.environ['MASTER_PORT']}")
    print("=" * 60)

    import torch.multiprocessing as mp
    mp.spawn(
        train_protbert_random_cpt,
        args=(world_size, args.config),
        nprocs=world_size,
        join=True
    )

    print("=" * 60)
    print("✓ All processes completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
