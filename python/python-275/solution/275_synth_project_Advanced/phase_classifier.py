"""
phase_classifier.py
===================
非厄米谱相的分类与特征提取.
融合种子项目:
  - 1113_pantelisantonoudiou_seizyml_manuscript_code: 特征提取与分类
  - 481_graph_adj: 图的连通性分析 (用于拓扑不变量)

物理背景:
  非厄米系统存在多种谱相:
    1. PT 对称相: 所有本征值为实数.
    2. PT 破缺相: 部分本征值成为复共轭对.
    3. 趋肤相: 本征态局域在边界 (非零 IPR).
    4. 拓扑相: 非零 winding number.
  分类器基于谱特征:
    - 实谱比例
    - 平均 IPR
    - winding number
    - 能隙大小
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
import spectral_solver as ss


# ---------------------------------------------------------------------------
# 特征提取 (源自 seizyml 特征提取)
# ---------------------------------------------------------------------------
def extract_spectral_features(H: np.ndarray) -> Dict:
    """从哈密顿量提取谱特征向量.

    特征:
      - frac_real: 实本征值比例.
      - gap: 实部最小间距 (能隙).
      - max_ipr: 最大 IPR (趋肤效应指标).
      - mean_ipr: 平均 IPR.
      - spectral_center: 谱中心位置.
      - spectral_spread: 谱展宽.
      - asymmetry: 实部-虚部不对称度.
      - condition_number: 条件数 (非正态度).

    Returns
    -------
    features : dict of scalar features.
    """
    E, psi_R = ss.compute_spectrum(H, sort_by="real")
    N = H.shape[0]
    frac_real = float(np.sum(np.abs(E.imag) < 1e-6) / N)
    E_real_sorted = np.sort(E.real)
    gaps = np.diff(E_real_sorted)
    gap = float(np.min(gaps)) if len(gaps) > 0 else 0.0
    ipr, x_c = ss.skin_mode_localization(psi_R)
    max_ipr = float(np.max(ipr))
    mean_ipr = float(np.mean(ipr))
    spectral_center = complex(np.mean(E))
    spectral_spread = float(np.std(E))
    asymmetry = float(np.mean(E.real) * np.mean(E.imag))
    cond = float(np.linalg.cond(H))
    return {
        "frac_real": frac_real,
        "gap": gap,
        "max_ipr": max_ipr,
        "mean_ipr": mean_ipr,
        "spectral_center_real": spectral_center.real,
        "spectral_center_imag": spectral_center.imag,
        "spectral_spread": spectral_spread,
        "asymmetry": asymmetry,
        "condition_number": cond,
        "n_states": N,
    }


def extract_feature_vector(H: np.ndarray) -> np.ndarray:
    """将特征字典转为数值向量, 用于分类.

    Returns
    -------
    vec : (9,) 特征向量.
    """
    feats = extract_spectral_features(H)
    return np.array([
        feats["frac_real"],
        feats["gap"],
        feats["max_ipr"],
        feats["mean_ipr"],
        feats["spectral_center_real"],
        feats["spectral_center_imag"],
        feats["spectral_spread"],
        feats["asymmetry"],
        np.log10(max(feats["condition_number"], 1.0)),
    ])


# ---------------------------------------------------------------------------
# 谱相分类器 (基于规则)
# ---------------------------------------------------------------------------
def classify_phase(features: Dict) -> str:
    """基于规则的谱相分类.

    规则:
      - frac_real > 0.95 → 'PT_SYMMETRIC'
      - frac_real < 0.5 且 mean_ipr > 0.5 → 'SKIN_PHASE'
      - frac_real < 0.5 且 mean_ipr < 0.3 → 'PT_BROKEN'
      - 其他 → 'TRANSITIONAL'

    Returns
    -------
    phase : str 相名称.
    """
    frac_real = features.get("frac_real", 0.0)
    mean_ipr = features.get("mean_ipr", 0.0)
    if frac_real > 0.95:
        return "PT_SYMMETRIC"
    elif frac_real < 0.5 and mean_ipr > 0.5:
        return "SKIN_PHASE"
    elif frac_real < 0.5:
        return "PT_BROKEN"
    else:
        return "TRANSITIONAL"


# ---------------------------------------------------------------------------
# 连通性分析 (源自 481_graph_adj)
# ---------------------------------------------------------------------------
def build_adjacency_from_hamiltonian(H: np.ndarray,
                                     threshold: float = 1e-10) -> np.ndarray:
    """从哈密顿量构造耦合邻接矩阵.

    A[i,j] = 1 if |H[i,j]| > threshold and i != j.
    用于分析耦合拓扑.
    """
    N = H.shape[0]
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and abs(H[i, j]) > threshold:
                A[i, j] = 1
    return A


def adjacency_degree(adj: np.ndarray) -> np.ndarray:
    """每个节点的度 (源自 graph_adj_degree)."""
    return np.sum(adj, axis=1) + np.sum(adj, axis=0) - np.diag(adj)


def is_connected(adj: np.ndarray) -> bool:
    """检验图是否连通 (BFS, 源自 graph_adj)."""
    N = adj.shape[0]
    if N == 0:
        return True
    visited = set()
    queue = [0]
    visited.add(0)
    while queue:
        node = queue.pop(0)
        for neighbor in range(N):
            if adj[node, neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == N


def phase_diagram_classification(H_func, param_grid: np.ndarray,
                                 param_name: str,
                                 fixed_params: Dict = None) -> List[str]:
    """扫描参数, 对每个点分类谱相.

    Returns
    -------
    phases : list of str, 每个参数点的相.
    """
    if fixed_params is None:
        fixed_params = {}
    phases = []
    for p in param_grid:
        params = {param_name: p}
        params.update(fixed_params)
        try:
            H = H_func(**params)
            feats = extract_spectral_features(H)
            phase = classify_phase(feats)
        except Exception:
            phase = "UNKNOWN"
        phases.append(phase)
    return phases
