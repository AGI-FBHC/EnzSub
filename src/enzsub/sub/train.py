"""Train EnzSub from a YAML configuration."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import numpy as np
from datetime import datetime
from typing import Dict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import EnzSubFullConfig, load_config
from .model import EnzSubModel, EnzSubLoss
from .dataset import create_dataloaders
from .checkpoint import save_checkpoint
from .lora import disable_lora_context
from .preserve import (
    compute_cache_signature,
    load_or_build_teacher_pool_cache,
    pool_preserve_loss,
    token_preserve_loss,
)

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class Trainer:

    def __init__(self, cfg: EnzSubFullConfig):
        self.cfg = cfg
        self.tc = cfg.task
        self.preserve_cfg = cfg.task.preserve
        self.device = torch.device(cfg.training.device)

        set_seed(cfg.training.seed)

        if cfg.training.exp_name is None:
            etype = cfg.model.enzyme.type
            lora_str = f"lora{cfg.model.enzyme.lora.rank}" if cfg.model.enzyme.lora.enabled else "nolora"
            sub_str = cfg.model.substrate.type
            preserve_str = ""
            if self.preserve_cfg.enabled:
                preserve_str = f"_prsv{self.preserve_cfg.pool_weight}"
            elif self.preserve_cfg.monitor_only:
                preserve_str = "_prsvmon"
            cfg.training.exp_name = (
                f"enzsub_{etype}_{sub_str}_{lora_str}{preserve_str}_"
                f"{datetime.now().strftime('%m%d_%H%M')}"
            )

        self.output_dir = os.path.join(cfg.training.output_dir, cfg.training.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self._init_model()
        self._init_data()
        self._init_optimizer()
        self._init_preserve()

        self.scaler = GradScaler() if cfg.training.fp16 else None
        self.criterion = EnzSubLoss(cfg.task)

        if cfg.training.use_wandb:
            wandb.init(project=cfg.training.wandb_project,
                       name=cfg.training.exp_name)

        self.global_step = 0
        self.best_val_loss = float('inf')

    @property
    def _preserve_active(self) -> bool:
        """是否需要计算 preserve 相关量(loss 或 cosine 观测)。"""
        return self.preserve_cfg.enabled or self.preserve_cfg.monitor_only

    @property
    def _need_per_residue(self) -> bool:
        """是否需要让 model 返回 per-residue hidden(给 token preserve 用)。"""
        if not self._preserve_active:
            return False
        if not self.preserve_cfg.use_token_preserve:
            return False
        if self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled:
            return True
        return self.preserve_cfg.token_weight > 0

    def _init_model(self):
        self.model = EnzSubModel(self.cfg).to(self.device)

    def _init_data(self):
        cfg = self.cfg

        tokenize_enzyme_fn = self.model.enzyme_encoder.tokenize

        if self.model.has_substrate_encoder:
            sub_enc = self.model.substrate_encoder
            prepare_substrate_fn = sub_enc.prepare_batch
        else:
            prepare_substrate_fn = None

        self.train_loader, self.val_loader = create_dataloaders(
            data_config=cfg.data,
            task_config=cfg.task,
            tokenize_enzyme_fn=tokenize_enzyme_fn,
            prepare_substrate_fn=prepare_substrate_fn,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )
        print(f"[Data] Train: {len(self.train_loader)} batches, "
              f"Val: {len(self.val_loader)} batches")

    def _init_optimizer(self):
        cfg = self.cfg.training
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        micro_per_epoch = len(self.train_loader)
        updates_per_epoch = math.ceil(micro_per_epoch / cfg.accumulation_steps)
        warmup_steps = cfg.warmup_epochs * updates_per_epoch
        total_steps = cfg.epochs * updates_per_epoch

        warmup_steps = min(warmup_steps, max(1, total_steps - 1))

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps,
                                  eta_min=1e-7),
            ],
            milestones=[warmup_steps],
        )
        print(f"[Sched] micro/epoch={micro_per_epoch}, accum={cfg.accumulation_steps}, "
              f"updates/epoch={updates_per_epoch}, "
              f"warmup_steps={warmup_steps} (~{cfg.warmup_epochs}ep), "
              f"total_steps={total_steps}")

    def _init_preserve(self):
        """如果 preserve 启用或仅观测,构建/加载 teacher pool 缓存。"""
        self.teacher_pool_cache = None

        if not self._preserve_active:
            return

        if self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled:
            print("[Preserve] MONITOR-ONLY mode: teacher cache will be built, "
                  "cosine logged to wandb but NOT added to total loss.")
        elif self.preserve_cfg.enabled:
            print(f"[Preserve] ACTIVE mode: pool_w={self.preserve_cfg.pool_weight}, "
                  f"token_w={self.preserve_cfg.token_weight}")

        if self.preserve_cfg.loss_type != "cosine":
            raise NotImplementedError(
                f"preserve.loss_type={self.preserve_cfg.loss_type!r}, only 'cosine' supported."
            )
        if self.preserve_cfg.teacher_mode != "cpt_without_lora":
            raise NotImplementedError(
                f"preserve.teacher_mode={self.preserve_cfg.teacher_mode!r}, "
                f"only 'cpt_without_lora' supported."
            )

        if self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled:
            need_pool = self.preserve_cfg.use_pool_preserve
        else:
            need_pool = (self.preserve_cfg.use_pool_preserve
                         and self.preserve_cfg.pool_weight > 0)

        if not need_pool:
            print("[Preserve] Pool preserve off; skip cache build.")
            return

        train_ds = self.train_loader.dataset
        seen = set()
        all_pairs = []
        for eid in train_ds.enzyme_ids:
            if eid in seen or eid not in train_ds.enzyme_data:
                continue
            seen.add(eid)
            all_pairs.append((eid, train_ds.enzyme_data[eid]['sequence']))

        val_ds = self.val_loader.dataset
        for eid in val_ds.enzyme_ids:
            if eid in seen or eid not in val_ds.enzyme_data:
                continue
            seen.add(eid)
            all_pairs.append((eid, val_ds.enzyme_data[eid]['sequence']))

        print(f"[Preserve] Collected {len(all_pairs)} unique enzymes for teacher pool cache")

        signature = compute_cache_signature(
            encoder_type=self.cfg.model.enzyme.type,
            cpt_checkpoint=self.cfg.model.enzyme.cpt_checkpoint,
            max_seq_length=self.cfg.data.max_seq_length,
        )

        self.teacher_pool_cache = load_or_build_teacher_pool_cache(
            enzyme_encoder=self.model.enzyme_encoder,
            enzyme_pairs=all_pairs,
            cache_dir=self.preserve_cfg.cache_dir,
            signature=signature,
            force_rebuild=self.preserve_cfg.force_rebuild,
            device=self.device,
            batch_size=self.preserve_cfg.cache_batch_size,
        )
        sample_v = next(iter(self.teacher_pool_cache.values()))
        print(f"[Preserve] Cache ready: {len(self.teacher_pool_cache)} entries, "
              f"dim={tuple(sample_v.shape)}")

    def _prepare_substrate_on_device(self, raw_batch):
        if raw_batch is None or not self.model.has_substrate_encoder:
            return None
        return self.model.substrate_encoder.prepare_batch(
            raw_batch['smiles'], self.device
        )

    def _move_enzyme_tokens(self, tokens):
        if isinstance(tokens, dict):
            return {k: v.to(self.device) for k, v in tokens.items()}
        return tokens.to(self.device)

    def _compute_preserve_losses(self, outputs, batch, enzyme_tokens) -> Dict[str, torch.Tensor]:
        """计算 pool 和 token preserve loss + cosine 指标。
        preserve 完全关闭则返回 {}。monitor_only 模式下也会计算这些量。"""
        if not self._preserve_active:
            return {}

        result: Dict[str, torch.Tensor] = {}

        pool_triggered = (
            self.preserve_cfg.use_pool_preserve
            and self.teacher_pool_cache is not None
            and (
                (self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled)
                or self.preserve_cfg.pool_weight > 0
            )
        )

        if pool_triggered:
            enzyme_ids = batch['enzyme_ids']
            try:
                h_teacher_list = [self.teacher_pool_cache[eid] for eid in enzyme_ids]
            except KeyError as e:
                raise RuntimeError(
                    f"enzyme_id {e} not in teacher_pool_cache. "
                    f"Set preserve.force_rebuild=true or delete cache file to rebuild."
                )
            h_teacher = torch.stack(h_teacher_list).to(self.device, non_blocking=True)
            h_student = outputs['h_enzyme']

            loss_pool, cos_pool = pool_preserve_loss(h_student, h_teacher)
            result['loss_pool_preserve'] = loss_pool
            result['cos_pool'] = cos_pool

        token_triggered = (
            self.preserve_cfg.use_token_preserve
            and (
                (self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled)
                or self.preserve_cfg.token_weight > 0
            )
        )

        if token_triggered:
            per_residue_student = outputs.get('per_residue_enzyme')
            residue_mask = outputs.get('residue_mask_enzyme')

            if per_residue_student is None or residue_mask is None:
                raise RuntimeError(
                    "Token preserve enabled but per_residue / residue_mask absent. "
                    "Check return_per_residue propagation."
                )

            with disable_lora_context(self.model.enzyme_encoder), torch.no_grad():
                teacher_out = self.model.enzyme_encoder(
                    enzyme_tokens, return_per_residue=True
                )
                per_residue_teacher = teacher_out['per_residue']

            loss_token, cos_token = token_preserve_loss(
                per_residue_student, per_residue_teacher, residue_mask
            )
            result['loss_token_preserve'] = loss_token
            result['cos_token'] = cos_token

        return result

    def _add_preserve_to_total(self, base_total, preserve_out):
        """把 preserve loss 加到 total_loss 上。
        monitor_only 模式下直接返回 base_total, 不参与训练。"""
        if not self.preserve_cfg.enabled:

            return base_total

        total = base_total
        if 'loss_pool_preserve' in preserve_out:
            total = total + self.preserve_cfg.pool_weight * preserve_out['loss_pool_preserve']
        if 'loss_token_preserve' in preserve_out:
            total = total + self.preserve_cfg.token_weight * preserve_out['loss_token_preserve']
        return total

    def _optimizer_step(self):
        """执行一次真实的 optimizer 更新 (含 unscale/clip/step/scheduler/zero_grad)。"""
        if self.cfg.training.fp16:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg.training.grad_clip)
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if (
            self.model.has_substrate_encoder
            and hasattr(self.model.substrate_encoder, "gnn")
        ):
            assert not self.model.substrate_encoder.gnn.training, (
                "Frozen MolCLR GNN entered train mode."
            )
        sums = {'total': 0.0, 'pref': 0.0, 'type': 0.0, 'contrast': 0.0,
                'pool_preserve': 0.0, 'token_preserve': 0.0,
                'cos_pool': 0.0, 'cos_token': 0.0}
        cnts = {'pool': 0, 'token': 0}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        self.optimizer.zero_grad()
        need_pr = self._need_per_residue

        step = -1
        for step, batch in enumerate(pbar):
            enzyme_tokens = self._move_enzyme_tokens(batch['enzyme_tokens'])
            pos_sub = self._prepare_substrate_on_device(batch['pos_substrate_batch'])
            neg_sub = self._prepare_substrate_on_device(batch['neg_substrate_batch'])

            targets = {
                'pref_labels': batch['pref_labels'].to(self.device),
                'type_labels': batch['type_labels'].to(self.device),

            }

            if self.cfg.training.fp16:
                with autocast():
                    outputs = self.model(enzyme_tokens, pos_sub, neg_sub,
                                         return_per_residue=need_pr)
                    losses = self.criterion(outputs, targets)
                    preserve_out = self._compute_preserve_losses(
                        outputs, batch, enzyme_tokens
                    )
                    total = self._add_preserve_to_total(losses['total_loss'], preserve_out)
                    loss = total / self.cfg.training.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(enzyme_tokens, pos_sub, neg_sub,
                                     return_per_residue=need_pr)
                losses = self.criterion(outputs, targets)
                preserve_out = self._compute_preserve_losses(outputs, batch, enzyme_tokens)
                total = self._add_preserve_to_total(losses['total_loss'], preserve_out)
                loss = total / self.cfg.training.accumulation_steps
                loss.backward()

            if (step + 1) % self.cfg.training.accumulation_steps == 0:
                self._optimizer_step()

            sums['total'] += total.item()
            sums['pref'] += losses['loss_pref'].item()
            sums['type'] += losses['loss_type'].item()
            sums['contrast'] += losses['loss_contrast'].item()
            if 'loss_pool_preserve' in preserve_out:
                sums['pool_preserve'] += preserve_out['loss_pool_preserve'].item()
                sums['cos_pool'] += preserve_out['cos_pool'].item()
                cnts['pool'] += 1
            if 'loss_token_preserve' in preserve_out:
                sums['token_preserve'] += preserve_out['loss_token_preserve'].item()
                sums['cos_token'] += preserve_out['cos_token'].item()
                cnts['token'] += 1

            postfix = {'loss': f"{total.item():.4f}",
                       'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"}
            if 'cos_pool' in preserve_out:
                postfix['cos'] = f"{preserve_out['cos_pool'].item():.3f}"
            pbar.set_postfix(**postfix)

            if (self.cfg.training.use_wandb
                    and self.global_step % 10 == 0 and self.global_step > 0):
                log = {
                    'train/loss_total': total.item(),
                    'train/loss_reg': losses['loss_pref'].item(),
                    'train/loss_type': losses['loss_type'].item(),
                    'train/loss_con': losses['loss_contrast'].item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step,
                }
                if 'loss_pool_preserve' in preserve_out:
                    log['train/loss_pool_preserve'] = preserve_out['loss_pool_preserve'].item()
                    log['train/mean_pool_cosine_student_teacher'] = preserve_out['cos_pool'].item()
                if 'loss_token_preserve' in preserve_out:
                    log['train/loss_token_preserve'] = preserve_out['loss_token_preserve'].item()
                    log['train/mean_token_cosine_student_teacher'] = preserve_out['cos_token'].item()
                wandb.log(log)

        if step >= 0 and (step + 1) % self.cfg.training.accumulation_steps != 0:
            self._optimizer_step()

        n = len(self.train_loader)
        out = {k: v / n for k, v in sums.items()
               if k not in ('cos_pool', 'cos_token')}
        out['cos_pool'] = sums['cos_pool'] / cnts['pool'] if cnts['pool'] else 0.0
        out['cos_token'] = sums['cos_token'] / cnts['token'] if cnts['token'] else 0.0
        return out

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        sums = {'total': 0.0, 'pref': 0.0, 'type': 0.0, 'contrast': 0.0,
                'pool_preserve': 0.0, 'token_preserve': 0.0,
                'cos_pool': 0.0, 'cos_token': 0.0}
        cnts = {'pool': 0, 'token': 0}

        all_pred_pref, all_true_pref = [], []
        all_pred_type, all_true_type = [], []
        all_z_enz, all_z_sub_pos = [], []

        need_pr = self._need_per_residue

        for batch in tqdm(self.val_loader, desc='Validating'):
            enzyme_tokens = self._move_enzyme_tokens(batch['enzyme_tokens'])
            pos_sub = self._prepare_substrate_on_device(batch['pos_substrate_batch'])
            neg_sub = self._prepare_substrate_on_device(batch['neg_substrate_batch'])

            targets = {
                'pref_labels': batch['pref_labels'].to(self.device),
                'type_labels': batch['type_labels'].to(self.device),

            }

            if self.cfg.training.fp16:
                with autocast():
                    outputs = self.model(enzyme_tokens, pos_sub, neg_sub,
                                         return_per_residue=need_pr)
                    losses = self.criterion(outputs, targets)
                    preserve_out = self._compute_preserve_losses(outputs, batch, enzyme_tokens)
            else:
                outputs = self.model(enzyme_tokens, pos_sub, neg_sub,
                                     return_per_residue=need_pr)
                losses = self.criterion(outputs, targets)
                preserve_out = self._compute_preserve_losses(outputs, batch, enzyme_tokens)

            total_val = losses['total_loss'].item()
            if self.preserve_cfg.enabled:
                if 'loss_pool_preserve' in preserve_out:
                    total_val += self.preserve_cfg.pool_weight * preserve_out['loss_pool_preserve'].item()
                if 'loss_token_preserve' in preserve_out:
                    total_val += self.preserve_cfg.token_weight * preserve_out['loss_token_preserve'].item()

            if 'loss_pool_preserve' in preserve_out:
                sums['pool_preserve'] += preserve_out['loss_pool_preserve'].item()
                sums['cos_pool'] += preserve_out['cos_pool'].item()
                cnts['pool'] += 1
            if 'loss_token_preserve' in preserve_out:
                sums['token_preserve'] += preserve_out['loss_token_preserve'].item()
                sums['cos_token'] += preserve_out['cos_token'].item()
                cnts['token'] += 1

            sums['total'] += total_val
            sums['pref'] += losses['loss_pref'].item()
            sums['type'] += losses['loss_type'].item()
            sums['contrast'] += losses['loss_contrast'].item()

            all_pred_pref.append(outputs['pred_pref'].cpu().numpy())
            all_true_pref.append(targets['pref_labels'].cpu().numpy())
            all_pred_type.append(torch.sigmoid(outputs['pred_type']).cpu().numpy())
            all_true_type.append(targets['type_labels'].cpu().numpy())

            if outputs['z_enzyme'] is not None:
                all_z_enz.append(outputs['z_enzyme'].cpu().numpy())
            if outputs['z_substrate_pos'] is not None:
                all_z_sub_pos.append(outputs['z_substrate_pos'].cpu().numpy())

        n = len(self.val_loader)
        metrics = {
            'loss': sums['total'] / n,
            'loss_pref': sums['pref'] / n,
            'loss_type': sums['type'] / n,
            'loss_contrast': sums['contrast'] / n,
            'loss_pool_preserve': sums['pool_preserve'] / n,
            'loss_token_preserve': sums['token_preserve'] / n,
            'cos_pool': sums['cos_pool'] / cnts['pool'] if cnts['pool'] else 0.0,
            'cos_token': sums['cos_token'] / cnts['token'] if cnts['token'] else 0.0,
        }

        pred_pref = np.concatenate(all_pred_pref)
        true_pref = np.concatenate(all_true_pref)
        metrics.update(self._regression_metrics(pred_pref, true_pref))

        pred_type = np.concatenate(all_pred_type)
        true_type = np.concatenate(all_true_type)
        metrics.update(self._multilabel_metrics(pred_type, true_type))

        if all_z_enz and all_z_sub_pos:
            z_e = np.concatenate(all_z_enz)
            z_s = np.concatenate(all_z_sub_pos)
            K = self.tc.num_pos_samples
            z_s = z_s.reshape(-1, K, z_s.shape[-1])
            metrics.update(self._retrieval_metrics(z_e, z_s))

        return metrics

    @staticmethod
    def _regression_metrics(pred, true):
        rs, ps, ss = [], [], []
        for d in range(pred.shape[1]):
            p, t = pred[:, d], true[:, d]
            ss_res = np.sum((t - p) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            rs.append(1 - ss_res / (ss_tot + 1e-8))
            if np.std(p) > 1e-8 and np.std(t) > 1e-8:
                ps.append(pearsonr(p, t)[0])
                ss.append(spearmanr(p, t)[0])
            else:
                ps.append(0.0); ss.append(0.0)
        return {'pref_r2': np.mean(rs), 'pref_pearson': np.mean(ps),
                'pref_spearman': np.mean(ss)}

    @staticmethod
    def _multilabel_metrics(pred_prob, true, threshold=0.5):
        pred_bin = (pred_prob >= threshold).astype(int)
        true_bin = true.astype(int)
        return {
            'type_f1_macro': f1_score(true_bin, pred_bin, average='macro', zero_division=0),
            'type_f1_micro': f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            'type_precision': precision_score(true_bin, pred_bin, average='macro', zero_division=0),
            'type_recall': recall_score(true_bin, pred_bin, average='macro', zero_division=0),
        }

    @staticmethod
    def _retrieval_metrics(z_enz, z_sub, k_values=(1, 5, 10)):
        N, K = z_sub.shape[0], z_sub.shape[1]
        all_sub = z_sub.reshape(-1, z_sub.shape[-1])
        sim = np.dot(z_enz, all_sub.T)

        recalls = {k: [] for k in k_values}
        mrr_list = []
        for i in range(N):
            pos_idx = set(range(i * K, (i + 1) * K))
            ranked = np.argsort(-sim[i])
            for k in k_values:
                recalls[k].append(len(pos_idx & set(ranked[:k].tolist())) > 0)
            for rank, idx in enumerate(ranked, 1):
                if idx in pos_idx:
                    mrr_list.append(1.0 / rank)
                    break
            else:
                mrr_list.append(0.0)

        m = {f'contrast_recall@{k}': np.mean(recalls[k]) for k in k_values}
        m['contrast_mrr'] = np.mean(mrr_list)
        return m

    def save(self, epoch, is_best=False):
        path = os.path.join(self.output_dir, f'checkpoint_epoch{epoch}.pt')
        save_checkpoint(self.model, path, epoch=epoch,
                        global_step=self.global_step,
                        optimizer_state=self.optimizer.state_dict(),
                        scheduler_state=self.scheduler.state_dict(),
                        best_val_loss=self.best_val_loss,
                        config=self.cfg)
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            save_checkpoint(self.model, best_path, epoch=epoch,
                            global_step=self.global_step,
                            best_val_loss=self.best_val_loss,
                            config=self.cfg)
            print("  Best model saved")

    def train(self):
        cfg = self.cfg.training
        print(f"\n{'='*60}")
        print(f"Starting: {cfg.exp_name}")
        if self.preserve_cfg.enabled:
            print(f"Preserve [ACTIVE]: pool_w={self.preserve_cfg.pool_weight}, "
                  f"token_w={self.preserve_cfg.token_weight}")
        elif self.preserve_cfg.monitor_only:
            print(f"Preserve [MONITOR-ONLY]: cosine logged but not added to loss")
        print(f"{'='*60}")

        for epoch in range(1, cfg.epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m = self.validate()

            print(f"\nEpoch {epoch}/{cfg.epochs}")
            print(f"  Train - loss: {train_m['total']:.4f}")
            print(f"  Val   - loss: {val_m['loss']:.4f}")
            print(f"  B3.1  - R²: {val_m['pref_r2']:.4f}, "
                  f"Pearson: {val_m['pref_pearson']:.4f}")
            print(f"  B3.2  - F1(macro): {val_m['type_f1_macro']:.4f}, "
                  f"F1(micro): {val_m['type_f1_micro']:.4f}")
            if 'contrast_recall@1' in val_m:
                print(f"  B4    - R@1: {val_m['contrast_recall@1']:.4f}, "
                      f"R@5: {val_m['contrast_recall@5']:.4f}, "
                      f"MRR: {val_m['contrast_mrr']:.4f}")

            if self._preserve_active:
                mode_tag = "ACTIVE" if self.preserve_cfg.enabled else "MONITOR"
                lines = []
                pool_show = self.preserve_cfg.use_pool_preserve and (
                    (self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled)
                    or self.preserve_cfg.pool_weight > 0
                )
                if pool_show:
                    lines.append(
                        f"pool: L={val_m['loss_pool_preserve']:.4f} "
                        f"cos={val_m['cos_pool']:.4f}"
                    )
                token_show = self.preserve_cfg.use_token_preserve and (
                    (self.preserve_cfg.monitor_only and not self.preserve_cfg.enabled)
                    or self.preserve_cfg.token_weight > 0
                )
                if token_show:
                    lines.append(
                        f"token: L={val_m['loss_token_preserve']:.4f} "
                        f"cos={val_m['cos_token']:.4f}"
                    )
                if lines:
                    print(f"  Prsv [{mode_tag}] - {' | '.join(lines)}")

            if cfg.use_wandb:
                rename = {
                    'loss': 'loss_total',
                    'loss_pref': 'loss_reg',
                    'loss_contrast': 'loss_con',
                    'cos_pool': 'mean_pool_cosine_student_teacher',
                    'cos_token': 'mean_token_cosine_student_teacher',
                }
                wandb_log = {'epoch': epoch}
                for k, v in val_m.items():
                    name = rename.get(k, k)
                    wandb_log[f'val/{name}'] = v
                wandb.log(wandb_log)

            is_best = val_m['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['loss']
            if epoch % cfg.save_every == 0 or is_best:
                self.save(epoch, is_best)

        print(f"\nTraining done. Best val loss: {self.best_val_loss:.4f}")
        if cfg.use_wandb:
            wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
