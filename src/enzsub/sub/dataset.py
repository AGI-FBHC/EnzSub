"""
EnzSub Dataset

数据集逻辑:
1. 构建酶级别数据 (序列、底物列表、偏好向量、类型标签)
2. 负采样策略: mixed (分级) 或 random (纯随机)
3. Collator 通过回调函数与编码器解耦
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import DataConfig, TaskConfig

class SubstrateTypeClassifier:
    """底物类型分类器 (8 类 multi-label)"""

    TYPE_NAMES = [
        'aromatic', 'aliphatic', 'heterocyclic', 'sugar',
        'amino_acid', 'lipid', 'nucleotide', 'small_polar',
    ]

    def classify(self, smiles: str) -> List[str]:
        if pd.isna(smiles) or not smiles:
            return ['small_polar']
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return ['small_polar']
            types = []
            if rdMolDescriptors.CalcNumAromaticRings(mol) > 0:
                types.append('aromatic')
            if rdMolDescriptors.CalcNumHeterocycles(mol) > 0:
                types.append('heterocyclic')
            if rdMolDescriptors.CalcNumAromaticRings(mol) == 0:
                types.append('aliphatic')
            num_oh = smiles.count('O')
            if num_oh >= 4 and ('C1OC' in smiles or 'C1CO' in smiles):
                types.append('sugar')
            if 'N' in smiles and 'C(=O)O' in smiles:
                types.append('amino_acid')
            if 'CCCCCCCC' in smiles:
                types.append('lipid')
            if ('P' in smiles or 'p' in smiles) and 'N' in smiles:
                types.append('nucleotide')
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            if mw < 200 and tpsa > 20:
                types.append('small_polar')
            return types or ['aliphatic']
        except Exception:
            return ['small_polar']

    def encode(self, types: List[str]) -> np.ndarray:
        vec = np.zeros(len(self.TYPE_NAMES), dtype=np.float32)
        for t in types:
            if t in self.TYPE_NAMES:
                vec[self.TYPE_NAMES.index(t)] = 1.0
        return vec

def signed_log_transform(x: np.ndarray) -> np.ndarray:
    """
    符号保留的 log 压缩: sign(x) * log1p(|x|)

    对所有实数安全 (兼容 LogP_mean 的负值)。
    对大值 (MolWt~1000, TPSA~500) 压缩, 对接近 0 的值近似线性。
    必须在 StandardScaler.fit 之前施加。
    """
    return np.sign(x) * np.log1p(np.abs(x))

class EnzSubDataset(Dataset):
    """
    EnzSub 训练数据集

    __getitem__ 返回纯 Python 数据 (字符串 / numpy)，
    不做任何 tokenization，由 Collator 处理。
    """

    def __init__(
        self,
        data_config: DataConfig,
        task_config: TaskConfig,
        split: str = "train",
    ):
        super().__init__()
        self.dc = data_config
        self.tc = task_config
        self.split = split
        self.type_classifier = SubstrateTypeClassifier()

        print(f"[EnzSubDataset] Loading data for {split} split...")
        self._load_data()
        self._build_enzyme_data()

        if self.tc.neg_sampling == "mixed":
            self._compute_similarity_matrix()
        else:
            self.similarity_groups = None

        self._split_data()

        if self.dc.normalize_pref and split == "train":
            self._fit_scaler()

    def _load_data(self):
        self.oed_df = pd.read_csv(self.dc.oed_data)
        self.enzyme_stats_df = pd.read_csv(self.dc.enzyme_stats)
        print(f"  OED records: {len(self.oed_df)}, "
              f"Unique enzymes: {len(self.enzyme_stats_df)}")

    def _build_enzyme_data(self):
        print("  Building enzyme-level data...")
        self.enzyme_data = {}

        for enzyme_id, group in tqdm(self.oed_df.groupby('UNIPROT'),
                                      desc="  Processing enzymes"):
            seq = group['Sequence'].iloc[0]
            if pd.isna(seq) or len(seq) == 0:
                continue
            if len(seq) > self.dc.max_seq_length:
                seq = seq[:self.dc.max_seq_length]

            substrates = group['SMILES'].dropna().unique().tolist()
            if not substrates:
                continue

            stats_row = self.enzyme_stats_df[
                self.enzyme_stats_df['enzyme_id'] == enzyme_id
            ]
            if len(stats_row) == 0:
                continue

            pref_cols = [f"{f}_mean" for f in self.dc.pref_features]
            pref_vector = stats_row[pref_cols].values[0].astype(np.float32)
            pref_vector = np.nan_to_num(pref_vector, nan=0.0)

            if getattr(self.dc, 'pref_log_transform', False):
                pref_vector = signed_log_transform(pref_vector)

            all_types = set()
            for smi in substrates:
                all_types.update(self.type_classifier.classify(smi))
            type_label = self.type_classifier.encode(list(all_types))

            self.enzyme_data[enzyme_id] = {
                'sequence': seq,
                'substrates': substrates,
                'pref_vector': pref_vector,
                'type_label': type_label,
            }

        self.enzyme_ids = list(self.enzyme_data.keys())
        print(f"  Valid enzymes: {len(self.enzyme_ids)}")
        self._build_substrate_index()

    def _build_substrate_index(self):
        self.all_substrates = set()
        for data in self.enzyme_data.values():
            self.all_substrates.update(data['substrates'])
        self.all_substrates = list(self.all_substrates)

        self.substrate_to_enzymes = {}
        for enz_id, data in self.enzyme_data.items():
            for sub in data['substrates']:
                self.substrate_to_enzymes.setdefault(sub, []).append(enz_id)
        print(f"  Total unique substrates: {len(self.all_substrates)}")

    def _compute_similarity_matrix(self):
        print("  Computing preference similarity matrix...")
        pref_matrix = np.stack([
            self.enzyme_data[eid]['pref_vector'] for eid in self.enzyme_ids
        ])
        if not getattr(self.dc, 'pref_log_transform', False):
            pref_matrix = signed_log_transform(pref_matrix)
        pref_matrix = StandardScaler().fit_transform(pref_matrix)
        self.similarity_matrix = cosine_similarity(pref_matrix)
        self.enzyme_to_idx = {eid: i for i, eid in enumerate(self.enzyme_ids)}

        print("  Building similarity groups...")
        self.similarity_groups = {}
        for i, enz_id in enumerate(self.enzyme_ids):
            sims = self.similarity_matrix[i]
            easy = np.where(sims < self.dc.sim_easy_max)[0].tolist()
            medium = np.where(
                (sims >= self.dc.sim_medium_min) & (sims < self.dc.sim_medium_max)
            )[0].tolist()
            hard = np.where(
                (sims >= self.dc.sim_hard_min) & (sims < self.dc.sim_hard_max)
            )[0].tolist()
            for lst in [easy, medium, hard]:
                if i in lst:
                    lst.remove(i)
            self.similarity_groups[enz_id] = {
                'easy': easy, 'medium': medium, 'hard': hard
            }

    def _split_data(self):
        if self.dc.split_method == "cluster":
            self._cluster_based_split()
        else:
            self._random_split()

    def _random_split(self):
        n = len(self.enzyme_ids)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(n * self.dc.train_ratio)
        if self.split == "train":
            self.indices = indices[:split_idx].tolist()
        else:
            self.indices = indices[split_idx:].tolist()
        print(f"  {self.split} set: {len(self.indices)} (random)")

    def _cluster_based_split(self):
        print("  Using cluster-based split...")
        if self.dc.cluster_file and os.path.exists(self.dc.cluster_file):
            clusters = self._load_cluster_file(self.dc.cluster_file)
        else:
            clusters = self._simple_sequence_clustering()
        self._split_by_clusters(clusters)

    def _load_cluster_file(self, path: str) -> Dict[str, str]:
        clusters = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0] in self.enzyme_data:
                    clusters[parts[0]] = parts[1]
        print(f"  Loaded {len(clusters)} cluster mappings")
        return clusters

    def _simple_sequence_clustering(self) -> Dict[str, int]:
        print(f"  Computing k-mer clusters (threshold={self.dc.seq_identity_threshold})...")
        k = 3
        enzyme_kmers = {}
        for eid in self.enzyme_ids:
            seq = self.enzyme_data[eid]['sequence']
            enzyme_kmers[eid] = {seq[i:i+k] for i in range(len(seq) - k + 1)}

        clusters = {}
        centers = []
        cid = 0
        shuffled = self.enzyme_ids.copy()
        np.random.seed(42)
        np.random.shuffle(shuffled)

        for eid in shuffled:
            kmers = enzyme_kmers[eid]
            best_c, best_sim = None, 0
            for ci, (_, ck) in enumerate(centers):
                inter = len(kmers & ck)
                union = len(kmers | ck)
                sim = inter / union if union else 0
                if sim > best_sim:
                    best_sim, best_c = sim, ci
            if best_sim >= self.dc.seq_identity_threshold and best_c is not None:
                clusters[eid] = best_c
            else:
                clusters[eid] = cid
                centers.append((eid, kmers))
                cid += 1

        print(f"  {len(centers)} clusters from {len(self.enzyme_ids)} enzymes")
        return clusters

    def _split_by_clusters(self, clusters):
        cluster_to_enzymes = {}
        for eid, cid in clusters.items():
            cluster_to_enzymes.setdefault(cid, []).append(eid)

        cluster_ids = list(cluster_to_enzymes.keys())
        np.random.seed(42)
        np.random.shuffle(cluster_ids)

        train_enz, val_enz = [], []
        target = int(len(self.enzyme_ids) * self.dc.train_ratio)
        for cid in cluster_ids:
            if len(train_enz) < target:
                train_enz.extend(cluster_to_enzymes[cid])
            else:
                val_enz.extend(cluster_to_enzymes[cid])

        eid_to_idx = {eid: i for i, eid in enumerate(self.enzyme_ids)}
        if self.split == "train":
            self.indices = [eid_to_idx[e] for e in train_enz if e in eid_to_idx]
        else:
            self.indices = [eid_to_idx[e] for e in val_enz if e in eid_to_idx]
        print(f"  {self.split} set: {len(self.indices)} (cluster)")

    def _fit_scaler(self):
        vecs = np.stack([
            self.enzyme_data[self.enzyme_ids[i]]['pref_vector']
            for i in self.indices
        ])
        self.scaler = StandardScaler().fit(vecs)

    def set_scaler(self, scaler):
        self.scaler = scaler

    def get_scaler(self):
        return getattr(self, 'scaler', None)

    def _sample_positives(self, substrates, k):
        if len(substrates) <= k:
            return random.choices(substrates, k=k)
        return random.sample(substrates, k)

    def _sample_negatives_mixed(self, enzyme_id, true_subs, n):
        """分级负采样 (easy/medium/hard)"""
        groups = self.similarity_groups[enzyme_id]
        n_easy = int(n * self.dc.neg_easy_ratio)
        n_med = int(n * self.dc.neg_medium_ratio)
        n_hard = n - n_easy - n_med

        def _sample(indices, count, level):
            out, labels = [], []
            if not indices:
                return out, labels
            selected = random.choices(indices, k=min(count * 2, len(indices)))
            for idx in selected:
                if len(out) >= count:
                    break
                subs = self.enzyme_data[self.enzyme_ids[idx]]['substrates']
                s = random.choice(subs)
                if s not in true_subs:
                    out.append(s)
                    labels.append(level)
            return out, labels

        e_s, e_h = _sample(groups['easy'], n_easy, 0)
        m_s, m_h = _sample(groups['medium'], n_med, 1)
        h_s, h_h = _sample(groups['hard'], n_hard, 2)
        negs = e_s + m_s + h_s
        hardness = e_h + m_h + h_h

        while len(negs) < n:
            s = random.choice(self.all_substrates)
            if s not in true_subs and s not in negs:
                negs.append(s)
                hardness.append(0)
        return negs[:n], hardness[:n]

    def _sample_negatives_random(self, true_subs, n):
        """纯随机负采样"""
        negs = []
        while len(negs) < n:
            s = random.choice(self.all_substrates)
            if s not in true_subs and s not in negs:
                negs.append(s)
        return negs, [0] * n

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        enzyme_id = self.enzyme_ids[real_idx]
        data = self.enzyme_data[enzyme_id]

        true_subs = set(data['substrates'])
        pos = self._sample_positives(data['substrates'], self.tc.num_pos_samples)

        if self.tc.neg_sampling == "mixed" and self.similarity_groups is not None:
            neg, hardness = self._sample_negatives_mixed(
                enzyme_id, true_subs, self.tc.num_neg_samples)
        else:
            neg, hardness = self._sample_negatives_random(
                true_subs, self.tc.num_neg_samples)

        pref = data['pref_vector'].copy()
        if hasattr(self, 'scaler') and self.scaler is not None:
            pref = self.scaler.transform([pref])[0]

        return {
            'enzyme_id': enzyme_id,
            'sequence': data['sequence'],
            'pos_substrates': pos,
            'neg_substrates': neg,
            'hardness': np.array(hardness, dtype=np.int64),
            'pref_vector': pref.astype(np.float32),
            'type_label': data['type_label'].astype(np.float32),
        }

class EnzSubCollator:
    """
    Collator 通过回调函数进行 tokenization，不直接持有 encoder。

    Args:
        tokenize_enzyme_fn: (sequences: List[Tuple[str,str]]) → tokens
        prepare_substrate_fn: (smiles: List[str], device) → batch_input
            如果为 None, 不处理底物 (substrate_type="none")
        device: 目标设备 (底物 prepare 可能需要)
    """

    def __init__(
        self,
        tokenize_enzyme_fn: Callable,
        prepare_substrate_fn: Optional[Callable] = None,
        num_pos: int = 4,
        num_neg: int = 16,
    ):
        self.tokenize_enzyme = tokenize_enzyme_fn
        self.prepare_substrate = prepare_substrate_fn
        self.num_pos = num_pos
        self.num_neg = num_neg

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        B = len(batch)

        enzyme_seqs = [(s['enzyme_id'], s['sequence']) for s in batch]
        enzyme_tokens = self.tokenize_enzyme(enzyme_seqs)

        pos_substrate_batch = None
        neg_substrate_batch = None

        if self.prepare_substrate is not None:

            pos_smiles = []
            for s in batch:
                pos_smiles.extend(s['pos_substrates'])
            pos_substrate_batch = {
                'smiles': pos_smiles,
                'num_per_enzyme': self.num_pos,
            }

            neg_smiles = []
            for s in batch:
                neg_smiles.extend(s['neg_substrates'])
            neg_substrate_batch = {
                'smiles': neg_smiles,
                'num_per_enzyme': self.num_neg,
            }

        pref_labels = torch.from_numpy(np.stack([s['pref_vector'] for s in batch]))
        type_labels = torch.from_numpy(np.stack([s['type_label'] for s in batch]))
        hardness = torch.from_numpy(np.stack([s['hardness'] for s in batch]))

        return {
            'enzyme_tokens': enzyme_tokens,
            'pos_substrate_batch': pos_substrate_batch,
            'neg_substrate_batch': neg_substrate_batch,
            'pref_labels': pref_labels,
            'type_labels': type_labels,
            'hardness': hardness,
            'enzyme_ids': [s['enzyme_id'] for s in batch],
        }

def create_dataloaders(
    data_config: DataConfig,
    task_config: TaskConfig,
    tokenize_enzyme_fn: Callable,
    prepare_substrate_fn: Optional[Callable],
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader"""

    train_ds = EnzSubDataset(data_config, task_config, split="train")
    val_ds = EnzSubDataset(data_config, task_config, split="val")

    val_ds.set_scaler(train_ds.get_scaler())

    collator = EnzSubCollator(
        tokenize_enzyme_fn=tokenize_enzyme_fn,
        prepare_substrate_fn=prepare_substrate_fn,
        num_pos=task_config.num_pos_samples,
        num_neg=task_config.num_neg_samples,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=True,
    )
    return train_loader, val_loader