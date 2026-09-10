#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN Active Site Prediction (Positive-Negative Contrast Library)

支持 ESM-2 (repr_layer=33, dim=1280), ESM-1b (33, 1280), ProtBERT-BFD (30, 1024)
通过 --encoder-type 统一指定架构

核心思路 (对比打分版):
  1. 从训练集收集活性位点残基作为【正库】, 从非活性残基随机采样作为【负库】
  2. 对每个残基, 打分 = mean_topk_cos(query, 正库) − mean_topk_cos(query, 负库)
     (旧版仅用正库, 忽略负样本分布, 在极端类别不平衡下打分方向错乱、AUROC<0.5;
      引入负库后打分变为相对量, 方向被掰正)
  3. 负库随机采样, 固定多种子重采、打分取平均以降低方差
  4. 在验证集上调最优阈值, 在测试集上评估

接口与输出结构与旧版完全兼容 (sweep.py 依赖 best_k / best_test_metrics /
all_k_results[k].test_metrics), 仅打分逻辑变化 + 新增 --neg-ratio / --neg-seeds。

用法:
  python knn_active_site.py \\
      --encoder-type protbert_bfd \\
      --train-data-file data/Enzyme_active_sites_train.txt \\
      --train-emb-dir results/embeddings/protbert_cpt/train \\
      --test-data-file data/Enzyme_active_sites_test.txt \\
      --test-emb-dir results/embeddings/protbert_cpt/test \\
      --model-name protbert_cpt \\
      --output-dir results/knn/protbert_cpt \\
      --ks 1 3 5 10 20 \\
      --neg-ratio 1 --neg-seeds 0 1 2 \\
      --device cuda:0
"""

import os
import json
import argparse
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score

from dataset import EnzymeDataset, get_dataset_labels
from model import get_repr_layer, get_max_seq_len
from utils import set_seed, compute_metrics

# ============================================================================
# 核心函数
# ============================================================================

def collect_residue_embeddings(dataset, indices=None):
    all_embs = []
    all_labels = []

    if indices is None:
        indices = range(len(dataset))

    for idx in indices:
        x, y, mask = dataset[idx]
        valid = mask.bool()
        all_embs.append(x[valid])
        all_labels.append(y[valid].numpy())

    embeddings = torch.cat(all_embs, dim=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels

def build_pos_neg_libraries(embeddings, labels, neg_ratio, rng):
    """
    正库 = 全部活性位点残基 (L2 归一化)
    负库 = 从非活性残基中随机采样 (正库大小 × neg_ratio) 个 (L2 归一化)

    负库需采样而非全量: 非活性残基达数十万, 全量会让 top-k 几乎总能找到
    高相似项, 把负向得分压平, 失去区分力。等量(或小倍数)采样最有区分度。
    """
    pos_mask = labels > 0.5
    pos = torch.nn.functional.normalize(embeddings[pos_mask], p=2, dim=1)
    n_pos = int(pos_mask.sum())

    neg_all_idx = np.where(~pos_mask)[0]
    n_neg = min(int(n_pos * neg_ratio), len(neg_all_idx))
    sel = rng.choice(neg_all_idx, size=n_neg, replace=False)
    neg = torch.nn.functional.normalize(embeddings[sel], p=2, dim=1)

    return pos, neg, n_pos, n_neg

def _topk_mean_sim(query_norm, library, k):
    sim = query_norm @ library.T
    actual_k = min(k, library.shape[0])
    topk_sim, _ = sim.topk(actual_k, dim=1)
    return topk_sim.mean(dim=1)

def knn_contrast_batch(query_embs, pos_lib, neg_lib, k=5):
    """对比打分: 离正库的近邻相似度 − 离负库的近邻相似度"""
    query_norm = torch.nn.functional.normalize(query_embs, p=2, dim=1)
    pos_score = _topk_mean_sim(query_norm, pos_lib, k)
    neg_score = _topk_mean_sim(query_norm, neg_lib, k)
    return pos_score - neg_score

def evaluate_knn(dataset, indices, pos_lib, neg_lib, k, device, batch_size=512):
    """单负库下的评估 (供多负库种子循环调用)"""
    all_scores = []
    all_labels = []

    for idx in indices:
        x, y, mask = dataset[idx]
        valid = mask.bool()
        embs = x[valid].to(device)
        labs = y[valid].numpy()

        scores_list = []
        for start in range(0, embs.shape[0], batch_size):
            end = min(start + batch_size, embs.shape[0])
            batch_scores = knn_contrast_batch(
                embs[start:end], pos_lib, neg_lib, k=k)
            scores_list.append(batch_scores.cpu().numpy())

        scores = np.concatenate(scores_list)
        all_scores.append(scores)
        all_labels.append(labs)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return all_scores, all_labels

def evaluate_knn_multineg(dataset, indices, build_embs, build_labs, k,
                          neg_ratio, neg_seeds, device, batch_size=512):
    """
    多负库种子评估: 每个种子建一次负库, 打分取平均, 再算指标。
    build_embs/build_labs 用于建库; indices 用于查询 (二者通过外部划分隔离)。
    """
    score_runs = []
    labels_ref = None
    for ns in neg_seeds:
        rng = np.random.default_rng(ns)
        pos_lib, neg_lib, _, _ = build_pos_neg_libraries(
            build_embs, build_labs, neg_ratio, rng)
        pos_lib = pos_lib.to(device)
        neg_lib = neg_lib.to(device)

        scores, labs = evaluate_knn(dataset, indices, pos_lib, neg_lib,
                                    k, device, batch_size)
        score_runs.append(scores)
        labels_ref = labs

        del pos_lib, neg_lib
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    mean_scores = np.mean(score_runs, axis=0)
    metrics = compute_metrics(mean_scores, labels_ref)
    return metrics, mean_scores, labels_ref

def find_optimal_threshold(scores, labels, n_thresholds=200, beta=1.0):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_score = -1.0
    best_thr = 0.0

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        score = fbeta_score(labels, preds, beta=beta, zero_division=0)

        if score > best_score:
            best_score = score
            best_thr = thr

    return float(best_thr), float(best_score)

# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="k-NN Active Site Prediction (Positive-Negative Contrast Library)"
    )

    parser.add_argument("--train-data-file", type=str, required=True)
    parser.add_argument("--train-emb-dir", type=str, required=True)
    parser.add_argument("--test-data-file", type=str, required=True)
    parser.add_argument("--test-emb-dir", type=str, required=True)

    parser.add_argument("--encoder-type", type=str, required=True,
                        help="Backbone architecture (e.g. esm2_650m, esm2_t33_650M, "
                             "esm2_8m, esm1b, protbert_bfd). Resolved via "
                             "model.py::ENCODER_TYPE_CONFIG.")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name for labeling")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ks", type=int, nargs='+', default=[1, 3, 5, 10, 20])

    # 新增: 负库采样参数
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="负库大小 = 正库 × neg_ratio (默认 1, 即等量)")
    parser.add_argument("--neg-seeds", type=int, nargs='+', default=[2025, 2026, 3047],
                        help="负库采样种子, 多个则打分取平均以降方差")

    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 从 encoder_type 获取架构参数
    repr_layer = get_repr_layer(args.encoder_type)
    max_seq_length = get_max_seq_len(args.encoder_type)

    print("=" * 70)
    print(f"k-NN Active Site Prediction (POS-NEG contrast): {args.model_name}")
    print("=" * 70)
    print(f"Encoder type:   {args.encoder_type}")
    print(f"Device:         {device}")
    print(f"k values:       {args.ks}")
    print(f"neg_ratio:      {args.neg_ratio}")
    print(f"neg_seeds:      {args.neg_seeds}")
    print(f"Seed:           {args.seed}")
    print(f"repr_layer:     {repr_layer}")
    print(f"max_seq_length: {max_seq_length}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = EnzymeDataset(
        data_file=args.train_data_file,
        esm_data_pt_path=args.train_emb_dir,
        max_seq_length=max_seq_length,
        repr_layer=repr_layer
    )
    test_dataset = EnzymeDataset(
        data_file=args.test_data_file,
        esm_data_pt_path=args.test_emb_dir,
        max_seq_length=max_seq_length,
        repr_layer=repr_layer
    )

    embed_dim = train_dataset.embed_dim
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    print(f"Embed dim:     {embed_dim}")

    sample_labels = get_dataset_labels(train_dataset)

    # 统计正样本
    train_embs, train_labs = collect_residue_embeddings(train_dataset)
    n_total = len(train_labs)
    n_pos = int((train_labs > 0.5).sum())
    print(f"\nTrain residues: {n_total:,}")
    print(f"  Positive (active site): {n_pos:,} ({n_pos/n_total*100:.1f}%)")
    print(f"  Pos library: {n_pos:,} x {embed_dim}d")
    print(f"  Neg library: ~{int(n_pos*args.neg_ratio):,} x {embed_dim}d "
          f"(sampled, {len(args.neg_seeds)} seeds averaged)")

    # 对每个 k 值做实验
    all_results = {}

    for k in args.ks:
        print(f"\n{'='*70}")
        print(f"k = {k}")
        print(f"{'='*70}")

        # 5-Fold CV 调阈值 (建库=训练折, 查询=验证折, 隔离)
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                              random_state=args.seed)
        fold_thresholds = []
        fold_val_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(train_dataset)), sample_labels)
        ):
            fold_id = fold + 1

            fold_embs, fold_labs = collect_residue_embeddings(
                train_dataset, train_idx)

            val_metrics, val_scores, val_labels = evaluate_knn_multineg(
                train_dataset, val_idx, fold_embs, fold_labs, k,
                args.neg_ratio, args.neg_seeds, device
            )

            best_thr, best_f1 = find_optimal_threshold(val_scores, val_labels)
            fold_thresholds.append(best_thr)

            val_preds = (val_scores >= best_thr).astype(int)
            from sklearn.metrics import matthews_corrcoef, f1_score as sk_f1
            val_metrics_thr = val_metrics.copy()
            val_metrics_thr['f1'] = float(sk_f1(val_labels, val_preds, zero_division=0))
            val_metrics_thr['mcc'] = float(matthews_corrcoef(val_labels, val_preds))
            val_metrics_thr['threshold'] = best_thr

            fold_val_metrics.append(val_metrics_thr)

            print(f"  Fold {fold_id}: threshold={best_thr:.4f}, "
                  f"F1={val_metrics_thr['f1']:.4f}, "
                  f"MCC={val_metrics_thr['mcc']:.4f}, "
                  f"AUROC={val_metrics['roc_auc']:.4f}, "
                  f"AUPRC={val_metrics['auprc']:.4f}")

        # 全量建库 + 测试 (建库=全训练集, 查询=测试集, 隔离)
        avg_threshold = float(np.median(fold_thresholds))
        print(f"\n  Median threshold from CV: {avg_threshold:.4f}")

        test_metrics, test_scores, test_labels = evaluate_knn_multineg(
            test_dataset, range(len(test_dataset)), train_embs, train_labs, k,
            args.neg_ratio, args.neg_seeds, device
        )

        test_preds = (test_scores >= avg_threshold).astype(int)
        from sklearn.metrics import (
            matthews_corrcoef, f1_score as sk_f1,
            precision_score, recall_score
        )
        test_metrics['f1'] = float(sk_f1(test_labels, test_preds, zero_division=0))
        test_metrics['mcc'] = float(matthews_corrcoef(test_labels, test_preds))
        test_metrics['precision'] = float(precision_score(
            test_labels, test_preds, zero_division=0))
        test_metrics['recall'] = float(recall_score(
            test_labels, test_preds, zero_division=0))
        test_metrics['threshold'] = avg_threshold

        print(f"\n  Test Results (k={k}):")
        print(f"    F1:        {test_metrics['f1']:.4f}")
        print(f"    MCC:       {test_metrics['mcc']:.4f}")
        print(f"    AUROC:     {test_metrics['roc_auc']:.4f}")
        print(f"    AUPRC:     {test_metrics['auprc']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall:    {test_metrics['recall']:.4f}")

        all_results[f"k={k}"] = {
            'k': k,
            'threshold': avg_threshold,
            'fold_val_metrics': fold_val_metrics,
            'test_metrics': test_metrics
        }

    # 找最佳 k
    best_k_key = max(all_results.keys(),
                     key=lambda x: all_results[x]['test_metrics']['f1'])
    best_result = all_results[best_k_key]

    print(f"\n{'='*70}")
    print(f"Best k: {best_result['k']} "
          f"(F1={best_result['test_metrics']['f1']:.4f}, "
          f"MCC={best_result['test_metrics']['mcc']:.4f})")
    print(f"{'='*70}")

    # 保存结果 (结构与旧版兼容, 新增 method/neg_ratio/neg_seeds 元信息)
    output = {
        'model': args.model_name,
        'encoder_type': args.encoder_type,
        'seed': args.seed,
        'method': 'knn_posneg_contrast',
        'distance_metric': 'cosine',
        'neg_ratio': args.neg_ratio,
        'neg_seeds': args.neg_seeds,
        'embed_dim': embed_dim,
        'repr_layer': repr_layer,
        'max_seq_length': max_seq_length,
        'library_size': n_pos,
        'total_train_residues': n_total,
        'best_k': best_result['k'],
        'best_test_metrics': best_result['test_metrics'],
        'all_k_results': all_results
    }

    results_file = os.path.join(args.output_dir, 'knn_results.json')
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()