#!/usr/bin/env python3
"""Deterministic, training-free k-NN evaluation for EC prediction.

Each query set may use the default reference gallery or declare its own. When
query and gallery overlap, ``exclude_self`` removes an identical sequence ID
before selecting the final neighbours.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

from utils import load_ec_labels, filter_ids_with_embeddings, build_label_matrix

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# ======== Embedding 缓存 (避免重复加载大文件) ========

class EmbeddingStore:
    """
    统一的 embedding 加载器, 支持合并格式和单文件格式。
    合并格式的 .pt 文件只加载一次, 缓存在内存中。
    """

    def __init__(self, emb_dir: str):
        self.emb_dir = emb_dir
        self._merged: Dict[str, torch.Tensor] = {}  # sid -> tensor
        self._loaded = False

    def _load_merged_files(self):
        """加载目录下所有 *_all.pt 合并文件"""
        if self._loaded:
            return
        for fname in os.listdir(self.emb_dir):
            if fname.endswith("_all.pt"):
                path = os.path.join(self.emb_dir, fname)
                data = torch.load(path, map_location="cpu")
                embs = data.get("embeddings", {})
                self._merged.update(embs)
                logging.info(f"  Loaded {len(embs)} embeddings from {fname}")
        self._loaded = True

    def get_available_ids(self) -> set:
        """返回所有可用的 seq_id"""
        self._load_merged_files()
        ids = set(self._merged.keys())
        for f in os.listdir(self.emb_dir):
            if f.endswith(".pt") and not f.endswith("_all.pt"):
                ids.add(f[:-3])
        return ids

    def load(
        self,
        ids: List[str],
        normalize: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """加载指定 ID 的 embedding, 优先从合并缓存读取"""
        self._load_merged_files()

        embeddings, valid_ids = [], []
        missing_from_merged = []

        for sid in ids:
            if sid in self._merged:
                embeddings.append(self._merged[sid].float())
                valid_ids.append(sid)
            else:
                missing_from_merged.append(sid)

        if missing_from_merged:
            for sid in missing_from_merged:
                path = os.path.join(self.emb_dir, f"{sid}.pt")
                if not os.path.exists(path):
                    continue
                data = torch.load(path, map_location="cpu")
                embeddings.append(data["mean_representations"].float())
                valid_ids.append(sid)

        if not embeddings:
            return None, []

        X = torch.stack(embeddings)
        if normalize:
            X = F.normalize(X, dim=1)
        return X.numpy(), valid_ids

def filter_ids_with_store(ids: List[str], store: EmbeddingStore) -> List[str]:
    """用 EmbeddingStore 过滤有 embedding 的 ID"""
    available = store.get_available_ids()
    return [sid for sid in ids if sid in available]

# ======== k-NN 推理 ========

def _search_topk(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    k: int,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回每个 query 的 top-k (相似度, gallery 索引)。
    FAISS 优先; 否则走 PyTorch 分块路径 (避免物化巨型相似度矩阵)。

    The PyTorch fallback processes queries in chunks to avoid materializing the
    complete query-by-gallery similarity matrix.
    """
    n_gallery = gallery_emb.shape[0]
    k_eff = min(k, n_gallery)

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(gallery_emb.shape[1])
        index.add(gallery_emb.astype(np.float32))
        sims, indices = index.search(query_emb.astype(np.float32), k_eff)
        return sims, indices

    gallery_t = torch.from_numpy(gallery_emb)
    n_queries = query_emb.shape[0]
    sims_out = np.zeros((n_queries, k_eff), dtype=np.float32)
    idx_out = np.zeros((n_queries, k_eff), dtype=np.int64)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        q_chunk = torch.from_numpy(query_emb[start:end])
        sims_chunk = q_chunk @ gallery_t.T          # (chunk, n_gallery)
        s_t, i_t = sims_chunk.topk(k_eff, dim=1)
        sims_out[start:end] = s_t.numpy()
        idx_out[start:end] = i_t.numpy()

    return sims_out, idx_out

def knn_predict(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gallery_labels: np.ndarray,
    k: int,
    tau: float = 0.5,
    exclude_self: bool = False,
    query_ids: Optional[List[str]] = None,
    gallery_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k-NN 多标签预测。

    With ``exclude_self=True``, retrieve one extra neighbour and remove a match
    whose gallery ID equals the query ID before retaining the final ``k``.
    """
    n_queries = query_emb.shape[0]
    n_classes = gallery_labels.shape[1]

    y_pred = np.zeros((n_queries, n_classes), dtype=np.int32)
    y_score = np.zeros((n_queries, n_classes), dtype=np.float32)

    if exclude_self:
        if query_ids is None or gallery_ids is None:
            raise ValueError("exclude_self=True 需要同时传入 query_ids 和 gallery_ids")
        # 多检索一个, 给踢自己留余量
        search_k = min(k + 1, gallery_emb.shape[0])
    else:
        search_k = k

    sims, indices = _search_topk(query_emb, gallery_emb, search_k)

    # 用于按 ID 踢自己
    gid_arr = np.array(gallery_ids) if exclude_self else None

    final_neighbors = np.zeros((n_queries, k), dtype=np.int64)

    for i in range(n_queries):
        row_idx = indices[i]
        row_sim = sims[i]

        if exclude_self:
            qid = query_ids[i]
            keep_mask = gid_arr[row_idx] != qid
            row_idx = row_idx[keep_mask]
            row_sim = row_sim[keep_mask]
            # 截到 k 个 (检索了 k+1, 踢掉自己后通常正好 ≥k)
            row_idx = row_idx[:k]
            row_sim = row_sim[:k]
            # 极端情况: 踢完不足 k 个 (gallery 太小), 用已有的
            if len(row_idx) == 0:
                final_neighbors[i, :] = -1
                continue

        # 记录用于落盘的最近邻 (可能不足 k, 右侧补 -1)
        m = len(row_idx)
        final_neighbors[i, :m] = row_idx
        if m < k:
            final_neighbors[i, m:] = -1

        weights = np.exp(row_sim)
        weights /= weights.sum()
        scores = (gallery_labels[row_idx] * weights[:, None]).sum(axis=0)
        y_score[i] = scores
        y_pred[i] = (scores >= tau).astype(np.int32)
        if y_pred[i].sum() == 0 and scores.max() > 0:
            y_pred[i, scores.argmax()] = 1

    return y_pred, y_score, final_neighbors

# ======== 评估指标 ========

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    metrics = {}

    # ------------------------------------------------------------
    # 1) CLEAN 论文主指标: EC 类别支持度加权的 P / R / F1
    #    与官方 CLEAN evaluate.py 中 average="weighted" 完全一致。
    # ------------------------------------------------------------
    metrics["precision"] = float(precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))
    metrics["recall"] = float(recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))
    metrics["f1"] = float(f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    ))

    # 显式别名，写入 results.csv 后不再需要猜测 f1 的 averaging 方式。
    metrics["weighted_precision"] = metrics["precision"]
    metrics["weighted_recall"] = metrics["recall"]
    metrics["weighted_f1"] = metrics["f1"]

    # ------------------------------------------------------------
    # 2) 旧代码指标: 对每个蛋白分别算 P/R/F1 后再平均。
    #    这是 sample-average，只保留为辅助指标，不再命名为主 f1。
    # ------------------------------------------------------------
    sample_f1, sample_prec, sample_rec = [], [], []
    for i in range(len(y_pred)):
        if y_true[i].sum() > 0:
            sample_f1.append(f1_score(y_true[i], y_pred[i], zero_division=0))
            sample_prec.append(precision_score(y_true[i], y_pred[i], zero_division=0))
            sample_rec.append(recall_score(y_true[i], y_pred[i], zero_division=0))

    metrics["sample_f1"] = float(np.mean(sample_f1)) if sample_f1 else 0.0
    metrics["sample_precision"] = float(np.mean(sample_prec)) if sample_prec else 0.0
    metrics["sample_recall"] = float(np.mean(sample_rec)) if sample_rec else 0.0

    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    mccs = []
    for i in range(len(y_pred)):
        if y_true[i].sum() > 0 or y_pred[i].sum() > 0:
            try:
                mccs.append(matthews_corrcoef(y_true[i], y_pred[i]))
            except Exception:
                pass
    metrics["mcc"] = float(np.mean(mccs)) if mccs else 0.0

    # CLEAN 论文使用 weighted ROC-AUC。只保留在当前测试集同时含正负
    # 样本的 EC 列，否则 sklearn 对全 0 / 全 1 列无法定义 ROC-AUC。
    positives = y_true.sum(axis=0)
    valid_cols = (positives > 0) & (positives < y_true.shape[0])
    if valid_cols.sum() > 0:
        try:
            metrics["auc"] = roc_auc_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="weighted"
            )
        except ValueError:
            metrics["auc"] = 0.0
        try:
            metrics["auprc"] = average_precision_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="weighted"
            )
        except ValueError:
            metrics["auprc"] = 0.0

        # micro 排名指标继续保留为辅助结果。
        try:
            metrics["micro_auc"] = roc_auc_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="micro"
            )
        except ValueError:
            metrics["micro_auc"] = 0.0
        try:
            metrics["micro_auprc"] = average_precision_score(
                y_true[:, valid_cols], y_score[:, valid_cols], average="micro"
            )
        except ValueError:
            metrics["micro_auprc"] = 0.0
    else:
        metrics["auc"] = 0.0
        metrics["auprc"] = 0.0
        metrics["micro_auc"] = 0.0
        metrics["micro_auprc"] = 0.0

    metrics["weighted_auc"] = metrics["auc"]
    metrics["weighted_auprc"] = metrics["auprc"]

    return metrics

# ======== 预测结果保存 ========

def save_predictions(
    output_dir: str,
    emb_name: str,
    test_name: str,
    k: int,
    query_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    neighbor_indices: np.ndarray,
    gallery_ids: List[str],
    idx_to_ec: Dict[int, str],
):
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    records = []
    for i, sid in enumerate(query_ids):
        true_ecs = sorted(idx_to_ec[j] for j in np.where(y_true[i] == 1)[0])
        pred_ecs = sorted(idx_to_ec[j] for j in np.where(y_pred[i] == 1)[0])
        top_idx = np.argsort(y_score[i])[::-1][:10]
        top_preds = [(idx_to_ec[j], float(y_score[i, j])) for j in top_idx if y_score[i, j] > 0]
        # Ignore padding indices introduced when too few neighbours remain.
        nn_ids = [gallery_ids[j] for j in neighbor_indices[i] if j >= 0]

        records.append({
            "id": sid,
            "true_ec": ";".join(true_ecs),
            "pred_ec": ";".join(pred_ecs),
            "top_predictions": ";".join(f"{ec}:{s:.4f}" for ec, s in top_preds),
            "nearest_neighbors": ";".join(nn_ids),
            "correct": set(true_ecs) == set(pred_ecs),
        })

    tag = f"{emb_name}_{test_name}_k{k}"
    pd.DataFrame(records).to_csv(os.path.join(pred_dir, f"{tag}.csv"), index=False)

    np.savez_compressed(
        os.path.join(pred_dir, f"{tag}_raw.npz"),
        query_ids=np.array(query_ids),
        y_true=y_true, y_pred=y_pred, y_score=y_score,
        neighbor_indices=neighbor_indices,
    )

# ======== 参数解析 ========

def parse_kv_pairs(pairs: List[str]) -> Dict[str, str]:
    result = {}
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Expected name:path format, got: {pair}")
        name, path = pair.split(":", 1)
        result[name.strip()] = path.strip()
    return result

def normalize_test_specs(
    test_sets_flat: Optional[Dict[str, str]],
    test_sets_json: Optional[str],
    default_gallery_csv: str,
) -> Dict[str, dict]:
    """
    Normalize both input forms to the same internal specification:
        {name: {"query": csv, "gallery": csv, "exclude_self": bool}}

    - 扁平 name:path  → gallery 默认 = default_gallery_csv, exclude_self=False
    - JSON 规格        → 可覆盖 gallery / exclude_self
    """
    specs: Dict[str, dict] = {}

    if test_sets_flat:
        for name, qpath in test_sets_flat.items():
            specs[name] = {
                "query": qpath,
                "gallery": default_gallery_csv,
                "exclude_self": False,
            }

    if test_sets_json:
        parsed = json.loads(test_sets_json)
        for name, spec in parsed.items():
            if isinstance(spec, str):
                spec = {"query": spec}
            specs[name] = {
                "query": spec["query"],
                "gallery": spec.get("gallery", default_gallery_csv),
                "exclude_self": bool(spec.get("exclude_self", False)),
            }

    return specs

# ======== 主函数 ========

def main():
    parser = argparse.ArgumentParser(description="Standalone k-NN EC classification")
    parser.add_argument(
        "--embedding-dirs", type=str, nargs="+", required=True,
        help="name:path pairs",
    )
    parser.add_argument("--train-csv", type=str, required=True,
                        help="默认 gallery 的 EC 标签 csv (各 test set 未指定 gallery 时用它)")
    parser.add_argument("--test-sets", type=str, nargs="*", default=None,
                        help="扁平格式 name:path (gallery 默认用 --train-csv, 不踢自己)")
    parser.add_argument("--test-sets-json", type=str, default=None,
                        help='JSON: {"name":{"query":..,"gallery":..,"exclude_self":..}}')
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="experiments/knn_eval")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    emb_dirs = parse_kv_pairs(args.embedding_dirs)

    test_sets_flat = parse_kv_pairs(args.test_sets) if args.test_sets else None
    test_specs = normalize_test_specs(
        test_sets_flat, args.test_sets_json, args.train_csv
    )
    if not test_specs:
        raise ValueError("必须提供 --test-sets 或 --test-sets-json")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 只写文件, 不挂 StreamHandler → 控制台静默, 全部进 knn.log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(output_dir, "knn.log"))],
        force=True,
    )
    log = logging.info

    log("=" * 60)
    log("k-NN EC Classification (standalone)")
    log("=" * 60)
    log(f"Embeddings: {list(emb_dirs.keys())}")
    log(f"Test sets:  {list(test_specs.keys())}")
    log(f"k: {args.k}, tau: {args.tau}, FAISS: {FAISS_AVAILABLE}")

    # ---- 收集所有需要的 gallery csv (去重) 并加载标签 ----
    # A query set may define a gallery different from the global default.
    gallery_csvs = sorted({spec["gallery"] for spec in test_specs.values()})
    log(f"Gallery sources: {gallery_csvs}")

    # 标签全集 (gallery + query 都纳入, 保证 ec_to_idx 覆盖完整)
    id_to_ecs: Dict[str, List[str]] = {}
    gallery_ids_by_csv: Dict[str, List[str]] = {}

    for gcsv in gallery_csvs:
        g_ecs, g_ids = load_ec_labels(gcsv)
        id_to_ecs.update(g_ecs)
        # 过滤掉含下划线的 id (与原逻辑一致), 排序保持确定性
        gallery_ids_by_csv[gcsv] = sorted(sid for sid in g_ids if "_" not in sid)

    for spec in test_specs.values():
        q_ecs, _ = load_ec_labels(spec["query"])
        id_to_ecs.update(q_ecs)

    # EC 类别空间必须覆盖 gallery + 所有 query 的真实 EC。
    #
    # 旧代码只使用 gallery EC，导致测试集中 gallery 从未出现的真实 EC
    # 根本没有进入 y_true；这些本应是 false negative，却被静默忽略，
    # 从而高估 F1/Recall。这里使用此前已合并的 id_to_ecs 全集。
    all_ecs = set()
    for ecs in id_to_ecs.values():
        all_ecs.update(ecs)

    ec_to_idx = {ec: i for i, ec in enumerate(sorted(all_ecs))}
    idx_to_ec = {i: ec for ec, i in ec_to_idx.items()}
    n_classes = len(ec_to_idx)
    log(f"EC classes: {n_classes}")

    json.dump(ec_to_idx, open(os.path.join(output_dir, "ec_mapping.json"), "w"), indent=2)

    # ---- 评估循环 ----
    all_results = []

    for emb_name, emb_dir in emb_dirs.items():
        log(f"\n{'='*60}")
        log(f"Evaluating: {emb_name}")
        log(f"  dir: {emb_dir}")

        store = EmbeddingStore(emb_dir)

        # 预加载本模型用到的所有 gallery (按 csv 缓存, 避免重复)
        gallery_cache: Dict[str, Tuple[np.ndarray, List[str], np.ndarray]] = {}
        for gcsv in gallery_csvs:
            gids = filter_ids_with_store(gallery_ids_by_csv[gcsv], store)
            gemb, gids = store.load(gids)
            if gemb is None:
                log(f"  [WARN] gallery {gcsv}: 无 embedding")
                continue
            glabels = build_label_matrix(gids, id_to_ecs, ec_to_idx)
            gallery_cache[gcsv] = (gemb, gids, glabels)
            log(f"  Gallery {os.path.basename(gcsv)}: {gemb.shape}")

        for tname, spec in test_specs.items():
            gcsv = spec["gallery"]
            exclude_self = spec["exclude_self"]

            if gcsv not in gallery_cache:
                log(f"\n--- {tname} --- (gallery 缺失, 跳过)")
                continue
            gallery_emb, gallery_ids, gallery_labels = gallery_cache[gcsv]

            log(f"\n--- {tname} ---  gallery={os.path.basename(gcsv)}  "
                f"exclude_self={exclude_self}")

            _, test_ids_all = load_ec_labels(spec["query"])
            test_ids = filter_ids_with_store(test_ids_all, store)
            query_emb, query_ids = store.load(test_ids)
            if query_emb is None:
                log(f"No query embeddings, skipping")
                continue
            query_labels = build_label_matrix(query_ids, id_to_ecs, ec_to_idx)
            log(f"  Query shape: {query_emb.shape}")

            for k in args.k:
                y_pred, y_score, nn_idx = knn_predict(
                    query_emb, gallery_emb, gallery_labels, k=k, tau=args.tau,
                    exclude_self=exclude_self,
                    query_ids=query_ids if exclude_self else None,
                    gallery_ids=gallery_ids if exclude_self else None,
                )
                metrics = compute_metrics(query_labels, y_pred, y_score)

                # 明确报告测试真值中 gallery 从未出现、因此 k-NN 不可能
                # 正确预测的 EC；这些列现在已保留在 y_true 中并计入 FN。
                gallery_has_label = gallery_labels.sum(axis=0) > 0
                query_has_label = query_labels.sum(axis=0) > 0
                unseen_truth_cols = query_has_label & (~gallery_has_label)
                metrics["unseen_truth_ec_count"] = int(unseen_truth_cols.sum())
                if unseen_truth_cols.any():
                    metrics["samples_with_unseen_truth_ec"] = int(
                        (query_labels[:, unseen_truth_cols].sum(axis=1) > 0).sum()
                    )
                else:
                    metrics["samples_with_unseen_truth_ec"] = 0

                log(
                    f"  k={k}: weighted-F1={metrics['weighted_f1']:.4f}  "
                    f"weighted-Prec={metrics['weighted_precision']:.4f}  "
                    f"weighted-Rec={metrics['weighted_recall']:.4f}  "
                    f"weighted-AUC={metrics['weighted_auc']:.4f}  "
                    f"sample-F1={metrics['sample_f1']:.4f}  "
                    f"MCC={metrics['mcc']:.4f}  "
                    f"unseen-EC={metrics['unseen_truth_ec_count']}"
                )

                if args.save_predictions:
                    save_predictions(
                        output_dir, emb_name, tname, k,
                        query_ids, query_labels, y_pred, y_score,
                        nn_idx, gallery_ids, idx_to_ec,
                    )

                all_results.append({
                    "embedding": emb_name, "test_set": tname, "k": k,
                    "gallery": os.path.basename(gcsv),
                    "exclude_self": exclude_self,
                    **metrics,
                })

    # ---- 汇总 ----
    if not all_results:
        log("No results produced.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)

    # 汇总表写入 log
    log("\n" + "=" * 60)
    log("Summary")
    log("=" * 60)
    for tname in test_specs:
        sub = df[df["test_set"] == tname].copy()
        if sub.empty:
            continue
        cols = [
            "embedding", "k",
            "weighted_f1", "weighted_precision", "weighted_recall",
            "weighted_auc", "weighted_auprc",
            "sample_f1", "micro_f1", "macro_f1", "mcc",
            "unseen_truth_ec_count", "samples_with_unseen_truth_ec",
        ]
        cols = [c for c in cols if c in sub.columns]
        disp = sub.copy()
        for c in cols[2:]:
            disp[c] = disp[c].map(lambda x: f"{x:.4f}")
        log(f"\n--- {tname.upper()} ---\n" + disp[cols].to_string(index=False))

    log("\n" + "=" * 60)
    log("Best CLEAN-paper weighted-F1 per test set:")
    for tname in test_specs:
        sub = df[df["test_set"] == tname]
        if sub.empty:
            continue
        best = sub.loc[sub["weighted_f1"].idxmax()]
        log(f"  {tname}: {best['embedding']} k={best['k']}  "
            f"weighted-F1={best['weighted_f1']:.4f}  "
            f"sample-F1={best['sample_f1']:.4f}  MCC={best['mcc']:.4f}")

    json.dump(
        {"embeddings": emb_dirs,
         "test_specs": test_specs,
         "k_values": args.k, "tau": args.tau, "timestamp": timestamp},
        open(os.path.join(output_dir, "config.json"), "w"),
        indent=2,
    )

    log(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
