"""
MolCLR 预训练 GIN 底物编码器

直接复用 MolCLR 官方 GINet 架构 (Wang et al., 2022, Nat Mach Intell),
加载在 ~10M ZINC/PubChem 分子上对比预训练得到的权重。

Reference:
    Wang et al., "Molecular Contrastive Learning of Representations via
    Graph Neural Networks", Nature Machine Intelligence, 2022.
    https://github.com/yuyangw/MolCLR

接入说明:
    - GINet / GINEConv 类逐字复用 MolCLR 官方实现
    - SMILES → graph 转换照搬 MolCLR dataset.py 的逻辑
    - checkpoint 用 feat_dim=512 训练 (非源码默认的 256)
    - 下游使用时只取 feat_lin 后的 h (512-dim), 丢弃 out_lin (pretraining projection head)
"""

import logging
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (MessagePassing, global_add_pool,
                                 global_max_pool, global_mean_pool)
from torch_geometric.utils import add_self_loops

from .base import BaseSubstrateEncoder

logger = logging.getLogger(__name__)

ATOM_LIST = list(range(1, 119))

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]

BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3

_RDKIT_LOGGER = RDLogger.logger()

def _suppress_rdkit_logs():
    """关闭 RDKit 的 warning/error 输出 (调用一次即可全局生效)"""
    _RDKIT_LOGGER.setLevel(RDLogger.CRITICAL)

def _smiles_to_molclr_graph(smiles: str) -> Optional[Data]:
    """
    SMILES → PyG Data, 严格按照 MolCLR 训练时的特征定义

    Returns:
        Data(x, edge_index, edge_attr) or None if SMILES 无效
        x:          [N, 2] long  — [atom_type_idx, chirality_idx]
        edge_index: [2, 2M] long — 双向边
        edge_attr:  [2M, 2] long — [bond_type_idx, bond_dir_idx]

    无效 SMILES / 罕见原子 (>118) / 罕见手性 / 单原子无键: 返回 None,
    由调用方决定 fallback 策略。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    N = mol.GetNumAtoms()
    if N == 0:
        return None

    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in ATOM_LIST:
            return None
        type_idx.append(ATOM_LIST.index(atomic_num))

        chiral_tag = atom.GetChiralTag()
        if chiral_tag not in CHIRALITY_LIST[:NUM_CHIRALITY_TAG]:
            chirality_idx.append(0)
        else:
            chirality_idx.append(CHIRALITY_LIST.index(chiral_tag))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        bd = bond.GetBondDir()
        if bt not in BOND_LIST or bd not in BONDDIR_LIST:
            continue
        bt_idx = BOND_LIST.index(bt)
        bd_idx = BONDDIR_LIST.index(bd)
        row += [start, end]
        col += [end, start]
        edge_feat.append([bt_idx, bd_idx])
        edge_feat.append([bt_idx, bd_idx])

    if len(row) == 0:

        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_feat, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GINEConv(MessagePassing):
    """单层 GIN with Edge features (MolCLR 官方版)"""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = (self.edge_embedding1(edge_attr[:, 0])
                           + self.edge_embedding2(edge_attr[:, 1]))
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GINet(nn.Module):
    """
    MolCLR 官方 GINet 架构

    结构:
        x_embedding1 + x_embedding2  (atom_type + chirality, 各自 emb_dim)
        → 5 × (GINEConv → BatchNorm → ReLU/Dropout)
        → global_mean_pool → feat_lin → [h]
        → out_lin → [out]   (pretraining-only projection head)

    下游使用时只取 h (feat_lin 输出, 维度 = feat_dim).
    """

    def __init__(self, num_layer: int = 5, emb_dim: int = 300,
                 feat_dim: int = 512, drop_ratio: float = 0.0,
                 pool: str = 'mean'):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim)
                                          for _ in range(num_layer)])

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pool: {pool}")

        self.feat_lin = nn.Linear(emb_dim, feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim // 2),
        )

    def forward(self, data, return_all: bool = False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        node_h = (
            self.x_embedding1(x[:, 0])
            + self.x_embedding2(x[:, 1])
        )

        for layer in range(self.num_layer):
            node_h = self.gnns[layer](
                node_h,
                edge_index,
                edge_attr,
            )
            node_h = self.batch_norms[layer](node_h)

            if layer == self.num_layer - 1:
                node_h = F.dropout(
                    node_h,
                    self.drop_ratio,
                    training=self.training,
                )
            else:
                node_h = F.dropout(
                    F.relu(node_h),
                    self.drop_ratio,
                    training=self.training,
                )

        h_pool = self.pool(node_h, data.batch)

        h_feat = self.feat_lin(h_pool)

        h_proj = self.out_lin(h_feat)

        if return_all:
            return {
                "pool": h_pool,
                "feat": h_feat,
                "proj": h_proj,
            }

        return h_feat

class MolCLREncoder(BaseSubstrateEncoder):
    """
    MolCLR 预训练 GIN 底物编码器

    用法:
        encoder = MolCLREncoder(
            pretrained_path="/path/to/molclr_gin.pth",
            emb_dim=300, feat_dim=512, num_layers=5,
            freeze=True,
        )
        batch = encoder.prepare_batch(smiles_list, device)
        out = encoder(batch)   # {'h': [B, 512]}
    """

    def __init__(
        self,
        pretrained_path: str,
        emb_dim: int = 300,
        feat_dim: int = 512,
        num_layers: int = 5,
        drop_ratio: float = 0.0,
        pool: str = 'mean',
        freeze: bool = True,
    ):
        super().__init__()
        self._raw_dim = feat_dim
        self._frozen = bool(freeze)
        _suppress_rdkit_logs()

        self.gnn = GINet(
            num_layer=num_layers,
            emb_dim=emb_dim,
            feat_dim=feat_dim,
            drop_ratio=drop_ratio,
            pool=pool,
        )

        self._load_pretrained_strict(pretrained_path)

        if self._frozen:
            for param in self.gnn.parameters():
                param.requires_grad = False

            self.gnn.eval()

        print(f"[MolCLREncoder] Loaded MolCLR pretrained GIN: "
              f"{num_layers} layers, emb_dim={emb_dim}, feat_dim={feat_dim}, "
              f"freeze={freeze}")

    def _load_pretrained_strict(self, path: str):
        """
        严格加载 MolCLR checkpoint。

        允许:
            - out_lin.* 出现在 unexpected 中 (下游不用 projection head, 但 ckpt 包含)
        禁止:
            - 任何其他 missing 或 unexpected key
        """
        ckpt = torch.load(path, map_location='cpu')

        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        missing, unexpected = self.gnn.load_state_dict(ckpt, strict=False)

        ALLOWED_PREFIXES = ('out_lin.',)

        unexpected_real = [k for k in unexpected
                           if not any(k.startswith(p) for p in ALLOWED_PREFIXES)]
        missing_real = [k for k in missing
                        if not any(k.startswith(p) for p in ALLOWED_PREFIXES)]

        if missing_real or unexpected_real:
            raise RuntimeError(
                f"[MolCLREncoder] Strict load failed!\n"
                f"  Missing keys (in model, not in ckpt):\n    {missing_real}\n"
                f"  Unexpected keys (in ckpt, not in model):\n    {unexpected_real}\n"
                f"  Checkpoint path: {path}"
            )

        n_loaded = len(ckpt) - len(unexpected)
        print(f"[MolCLREncoder] Strict load OK: "
              f"{n_loaded}/{len(ckpt)} tensors loaded from {path}")

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    def prepare_batch(self, smiles_list: List[str],
                      device: torch.device) -> Batch:
        """
        SMILES 列表 → PyG Batch

        无效 SMILES (None / 罕见原子) 用 "C" (单碳分子) fallback,
        逻辑与原 GNNEncoder 一致。
        """
        fallback = _smiles_to_molclr_graph("C")
        assert fallback is not None, "Fallback SMILES 'C' must be valid"

        data_list = []
        invalid_indices = []
        for i, smi in enumerate(smiles_list):
            data = _smiles_to_molclr_graph(smi)
            if data is None:
                data = fallback.clone()
                invalid_indices.append(i)
            data_list.append(data)

        if invalid_indices:
            ratio = len(invalid_indices) / len(smiles_list)
            msg = (f"[MolCLREncoder] {len(invalid_indices)}/{len(smiles_list)} "
                   f"invalid SMILES replaced with fallback ({ratio:.1%})")
            if ratio > 0.1:
                warnings.warn(msg, stacklevel=2)
            else:
                logger.info(msg)

        batch = Batch.from_data_list(data_list)

        invalid_mask = torch.zeros(len(smiles_list), dtype=torch.bool)
        for idx in invalid_indices:
            invalid_mask[idx] = True
        batch.invalid_mask = invalid_mask

        return batch.to(device)

    def train(self, mode: bool = True):
        """
        当 MolCLR 被冻结时，始终保持 GNN 为 eval 模式，
        避免 BatchNorm 使用当前小批次统计量。
        """
        super().train(mode)

        if self._frozen:
            self.gnn.eval()

        return self

    def forward(self, batch_input: Batch) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {'h': [B, feat_dim]}
        """
        h = self.gnn(batch_input)
        return {'h': h}