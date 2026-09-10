#!/usr/bin/env python3
"""
hfs_common.py — HFS pipeline 共享工具

集中放置三件容易出错、必须在 build_identity_pairs.py 和 hfs_eval.py 之间
保持完全一致的逻辑:

  1. EC 标签标准化 (normalize_ec / labelsets_at_level)
     - 按 --ec-level 截断
     - EC4 主实验下丢弃含 '-' 或缺级的不完整 EC
     - 低 level 下, 只要前若干级完整即可保留 (1.1.1.- 在 EC3 下 → 1.1.1)

  2. functional_match: 在指定 match_mode 下判定一对酶是 positive / negative / skip

  3. identity bin 划分: 把 0–100 的 pairwise identity 落到用户给定的 bins 上

另外提供:
  - setup_sweep_path(): 把 ../sweep 插入 sys.path, 以便复用 utils / knn_eval
  - canonical_pair(): undirected pair 规范排序

注意: 这里所有 EC / match / bin 逻辑是 HFS 的科学定义核心, 两个脚本务必只从此处导入,
不要各自再实现一份, 否则 build 和 eval 可能口径不一致。
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# 路径: 让 tasks/ec/hfs 下的脚本能 import tasks/ec/sweep 里的模块
# ----------------------------------------------------------------------
def setup_sweep_path() -> str:
    """把 ../sweep 插到 sys.path 最前, 返回该路径。

    优先用环境变量 ENZSUB_SWEEP_DIR 覆盖 (方便非标准布局);
    否则按本文件位置推断 ``<this_dir>/../sweep``。
    不要求项目被 pip install 成 package。
    """
    env_dir = os.environ.get("ENZSUB_SWEEP_DIR")
    if env_dir and os.path.isdir(env_dir):
        sweep_dir = os.path.abspath(env_dir)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        sweep_dir = os.path.normpath(os.path.join(here, "..", "..", "sweep"))
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(
            f"找不到 sweep 目录: {sweep_dir}\n"
            f"请确认 hfs 与 sweep 同在 tasks/ec/ 下, "
            f"或设置环境变量 ENZSUB_SWEEP_DIR 指向 sweep 目录。"
        )
    if sweep_dir not in sys.path:
        sys.path.insert(0, sweep_dir)
    return sweep_dir

# ----------------------------------------------------------------------
# EC 标签标准化
# ----------------------------------------------------------------------
VALID_MATCH_MODES = ("any_overlap", "no_partial_overlap")

def _ec_part_complete(part: str) -> bool:
    """单个 EC 级是否"完整": 非空且不含 '-'。"""
    part = part.strip()
    return part != "" and "-" not in part

def normalize_ec(raw_ec: str, ec_level: int) -> Optional[str]:
    """把单个原始 EC 字符串 (如 '1.1.1.1' / '2.7.-.-' / 'EC 3.2.1.-') 标准化到 ec_level。

    返回截断后的 EC 字符串 (如 '1.1.1'), 若在该 level 下不完整则返回 None。

    规则:
      - 去掉可能的前缀 'EC'/'ec' 和空白
      - 至少要有 ec_level 个级, 且前 ec_level 级每一级都完整 (无 '-'、非空)
      - EC4: '1.1.1.-' → None (第 4 级是 '-')
      - EC3: '1.1.1.-' → '1.1.1';  '2.7.-.-' (EC2) → '2.7'
    """
    if raw_ec is None:
        return None
    s = raw_ec.strip()
    if not s:
        return None
    # 去掉可能的 "EC " 前缀
    low = s.lower()
    if low.startswith("ec "):
        s = s[3:].strip()
    elif low.startswith("ec:"):
        s = s[3:].strip()
    parts = [p.strip() for p in s.split(".")]
    if len(parts) < ec_level:
        return None
    prefix = parts[:ec_level]
    if not all(_ec_part_complete(p) for p in prefix):
        return None
    return ".".join(prefix)

def labelset_at_level(raw_ecs: List[str], ec_level: int) -> Set[str]:
    """把一个酶的原始 EC 列表标准化成指定 level 下的有效标签集合 (可能为空)。"""
    out: Set[str] = set()
    for ec in raw_ecs:
        norm = normalize_ec(ec, ec_level)
        if norm is not None:
            out.add(norm)
    return out

def build_labelsets(
    id_to_raw_ecs: Dict[str, List[str]],
    ec_level: int,
) -> Dict[str, Set[str]]:
    """对所有酶批量标准化; 只保留在该 level 下有 >=1 个有效标签的酶。"""
    result: Dict[str, Set[str]] = {}
    for sid, raw in id_to_raw_ecs.items():
        labels = labelset_at_level(raw, ec_level)
        if labels:
            result[sid] = labels
    return result

# ----------------------------------------------------------------------
# functional match
# ----------------------------------------------------------------------
def functional_match(
    labels_i: Optional[Set[str]],
    labels_j: Optional[Set[str]],
    match_mode: str,
) -> str:
    """判定一对酶的功能关系, 返回 'positive' / 'negative' / 'skip'。

    any_overlap (主实验):
        positive  : 交集非空
        negative  : 交集为空
        skip      : 任一方在该 level 下无有效标签

    no_partial_overlap (敏感性分析, 严格):
        positive  : 标签集完全相等
        negative  : 交集为空
        skip      : 部分重叠但不完全相等; 或任一方无有效标签
    """
    if not labels_i or not labels_j:
        return "skip"
    inter = labels_i & labels_j
    if match_mode == "any_overlap":
        return "positive" if inter else "negative"
    elif match_mode == "no_partial_overlap":
        if labels_i == labels_j:
            return "positive"
        elif not inter:
            return "negative"
        else:
            return "skip"
    else:
        raise ValueError(
            f"未知 match_mode: {match_mode!r}; 期望 {VALID_MATCH_MODES}"
        )

# ----------------------------------------------------------------------
# identity bins
# ----------------------------------------------------------------------
def make_bins(edges: List[float]) -> List[Tuple[float, float, str]]:
    """把递增的 bin 边界 [0,30,50,70,100] 转成 [(left,right,label), ...]。

    label 形如 '0-30'。整数边界打印成整数, 否则保留浮点。
    """
    if len(edges) < 2:
        raise ValueError(f"bins 至少要 2 个边界, 得到: {edges}")
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError(f"bins 边界必须严格递增: {edges}")
    bins = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        ll = int(left) if float(left).is_integer() else left
        rr = int(right) if float(right).is_integer() else right
        bins.append((float(left), float(right), f"{ll}-{rr}"))
    return bins

def assign_bin(identity: float, bins: List[Tuple[float, float, str]]) -> Optional[str]:
    """把一个 0–100 的 identity 落到某个 bin。

    区间约定: 除最后一个 bin 外为左闭右开 [left, right);
    最后一个 bin 为闭区间 [left, right] (这样 identity=100 也能落进去)。
    若落在所有 bin 之外 (bins 未覆盖全 0–100 时), 返回 None。
    """
    n = len(bins)
    for idx, (left, right, label) in enumerate(bins):
        if idx == n - 1:
            if left <= identity <= right:
                return label
        else:
            if left <= identity < right:
                return label
    return None

def main_class_of(ec_label: str) -> str:
    """取 EC 标签的主类 (第一级), 如 '1.1.1' → '1'。"""
    return ec_label.split(".")[0]

# ----------------------------------------------------------------------
# pair 规范化
# ----------------------------------------------------------------------
def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """undirected pair 统一排序 → (min, max), 保证 A-B 和 B-A 折叠成同一条。"""
    return (a, b) if a <= b else (b, a)

# ----------------------------------------------------------------------
# identity 标度归一: 若最大值 <= 1.0 视为 0–1 分数, 乘 100 转成百分比
# ----------------------------------------------------------------------
def scale_identity_to_percent(values, log=None) -> "object":
    """传入 numpy array / pandas Series, 若最大值 <= 1.0 则 ×100。返回同类型。"""
    try:
        vmax = float(values.max())
    except ValueError:
        return values  # 空
    if vmax <= 1.0:
        if log is not None:
            log(f"  identity 最大值={vmax:.4f} <= 1.0, 判定为 0–1 分数, 自动 ×100 转百分比")
        return values * 100.0
    return values