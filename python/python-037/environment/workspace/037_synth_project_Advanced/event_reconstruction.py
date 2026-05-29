r"""
event_reconstruction.py
事件重建与背景甄别模块

本模块实现：
1. 事件特征向量构建（能量、位置、时间、波形参数）
2. 层次聚类背景甄别（参考 chain_letter_tree 的距离矩阵与聚类思想）
3. 多变量判别分析（Fisher 判别）
4. 事件分类与纯度估计

核心算法：

A. 距离矩阵构建：
    对 N 个事件，定义距离度量：
        d_{ij}^2 = w_E (E_i - E_j)^2 + w_z (z_i - z_j)^2
                   + w_t (t_i - t_j)^2 + w_r (r_i - r_j)^2

    其中 w_E, w_z, w_t, w_r 为各变量的权重。

B. 层次聚类（单链接 Single-Linkage）：
    1. 初始时每个事件为一个簇
    2. 计算所有簇对之间的距离：
       d(C_a, C_b) = \min_{i \in C_a, j \in C_b} d_{ij}
    3. 合并距离最近的两个簇
    4. 重复直到所有簇合并为一棵二叉树（dendrogram）

C. Fisher 线性判别：
    寻找投影方向 \vec{w} 使得类间散度与类内散度之比最大：
        J(\vec{w}) = \frac{\vec{w}^T S_B \vec{w}}{\vec{w}^T S_W \vec{w}}
    最优解：\vec{w} \propto S_W^{-1} (\vec{\mu}_1 - \vec{\mu}_2)

D. 背景抑制因子：
    在保持信号效率 ε_s = 90% 的条件下，
    背景抑制因子 R_b = N_b^{\rm raw} / N_b^{\rm cut}

参考文献：
- Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering.
- Fisher, R. A. (1936). Annals of Eugenics, 7, 179.
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================================
# 特征提取
# ============================================================================

def extract_event_features(events: List[Dict]) -> np.ndarray:
    """
    从事件字典列表中提取特征矩阵。

    特征向量（标准化后）：
        x = [E_obs / E_max, z / z_max, t / T, sqrt(x^2 + y^2) / R]

    参数：
        events: 事件列表

    返回：
        X: (N, 4) 特征矩阵
    """
    if not events:
        return np.zeros((0, 4))

    energies = np.array([ev.get("energy_obs", 0.0) for ev in events])
    zs = np.array([ev.get("z", 0.0) for ev in events])
    ts = np.array([ev.get("time_day", 0.0) for ev in events])
    rs = np.sqrt(np.array([ev.get("x", 0.0) ** 2 + ev.get("y", 0.0) ** 2 for ev in events]))

    # 标准化
    E_max = np.max(energies) if np.max(energies) > 0 else 1.0
    z_max = np.max(zs) if np.max(zs) > 0 else 1.0
    t_max = 365.25
    R_max = np.max(rs) if np.max(rs) > 0 else 1.0

    X = np.column_stack([
        energies / E_max,
        zs / z_max,
        ts / t_max,
        rs / R_max,
    ])
    return X


# ============================================================================
# 距离矩阵与层次聚类
# ============================================================================

def build_distance_matrix(X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    构建欧氏加权距离矩阵。

    公式：
        d_{ij} = \sqrt{ \sum_k w_k (X_{ik} - X_{jk})^2 }

    参数：
        X: (N, D) 特征矩阵
        weights: (D,) 权重向量，None 则等权

    返回：
        Dmat: (N, N) 对称距离矩阵
    """
    N = X.shape[0]
    if N < 2:
        return np.zeros((N, N))
    if weights is None:
        weights = np.ones(X.shape[1])
    weights = np.asarray(weights)

    Dmat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diff = X[i] - X[j]
            dist = np.sqrt(np.sum(weights * diff ** 2))
            Dmat[i, j] = dist
            Dmat[j, i] = dist
    return Dmat


def symmetrize_distance_matrix(Dmat: np.ndarray) -> np.ndarray:
    """
    对称化距离矩阵：
        D^{\rm sym}_{ij} = \frac{1}{2} (D_{ij} + D_{ji})
    """
    return 0.5 * (Dmat + Dmat.T)


def single_linkage_clustering(Dmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    单链接层次聚类。

    算法（SLINK 简化版）：
        初始：N 个簇，每个簇含一个事件
        迭代：
            找到距离最近的两个簇 (a, b)
            d(C_{\rm new}, C_k) = \min( d(C_a, C_k), d(C_b, C_k) )
            合并 C_a 和 C_b

    参数：
        Dmat: (N, N) 距离矩阵

    返回：
        linkage_matrix: (N-1, 3) 每行 [cluster_a, cluster_b, distance]
        cluster_labels: (N,) 最终聚类标签（用于切割 dendrogram）
    """
    N = Dmat.shape[0]
    if N < 2:
        return np.zeros((0, 3)), np.zeros(N, dtype=int)

    # 初始簇：每个点一个簇
    clusters = [{i} for i in range(N)]
    linkage = []
    active = list(range(N))

    # 簇间距离缓存
    cluster_dist = {}
    for i in range(N):
        for j in range(i + 1, N):
            cluster_dist[(i, j)] = Dmat[i, j]

    next_id = N
    while len(active) > 1:
        # 找最近的簇对
        min_dist = np.inf
        pair = (active[0], active[1])
        for i_idx in range(len(active)):
            for j_idx in range(i_idx + 1, len(active)):
                a = active[i_idx]
                b = active[j_idx]
                key = (min(a, b), max(a, b))
                d = cluster_dist.get(key, np.inf)
                if d < min_dist:
                    min_dist = d
                    pair = (a, b)

        a, b = pair
        linkage.append([a, b, min_dist])

        # 合并为新簇
        new_cluster = next_id
        next_id += 1

        # 更新距离（单链接：取最小）
        for c in active:
            if c == a or c == b:
                continue
            key_a = (min(a, c), max(a, c))
            key_b = (min(b, c), max(b, c))
            da = cluster_dist.get(key_a, np.inf)
            db = cluster_dist.get(key_b, np.inf)
            key_new = (min(new_cluster, c), max(new_cluster, c))
            cluster_dist[key_new] = min(da, db)

        active = [c for c in active if c != a and c != b]
        active.append(new_cluster)

    linkage_matrix = np.array(linkage)

    # 生成切割标签（按距离阈值切割 dendrogram）
    if len(linkage) > 0:
        threshold = np.percentile(linkage_matrix[:, 2], 50.0)
        labels = cut_dendrogram(linkage_matrix, N, threshold)
    else:
        labels = np.zeros(N, dtype=int)

    return linkage_matrix, labels


def cut_dendrogram(linkage: np.ndarray, n_leaves: int, threshold: float) -> np.ndarray:
    """
    按距离阈值切割 dendrogram，生成聚类标签。

    参数：
        linkage: (N-1, 3) 连接矩阵
        n_leaves: 叶子节点数（原始事件数）
        threshold: 切割阈值

    返回：
        labels: (n_leaves,) 聚类标签
    """
    labels = np.zeros(n_leaves, dtype=int)
    cluster_id = 0

    def assign(node: int, cid: int):
        if node < n_leaves:
            labels[node] = cid
        else:
            row = int(node - n_leaves)
            if row < len(linkage):
                left, right, dist = linkage[row]
                if dist > threshold:
                    nonlocal cluster_id
                    cluster_id += 1
                    assign(int(left), cluster_id)
                    cluster_id += 1
                    assign(int(right), cluster_id)
                else:
                    assign(int(left), cid)
                    assign(int(right), cid)
            else:
                labels[node % n_leaves] = cid

    assign(2 * n_leaves - 2, cluster_id)
    return labels


# ============================================================================
# Fisher 判别分析
# ============================================================================

def fisher_discriminant(
    X_signal: np.ndarray,
    X_background: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    计算 Fisher 线性判别方向。

    公式：
        \vec{w} = S_W^{-1} (\vec{\mu}_s - \vec{\mu}_b)
        S_W = \Sigma_s + \Sigma_b    (类内散度矩阵)

    参数：
        X_signal: (N_s, D) 信号样本
        X_background: (N_b, D) 背景样本

    返回：
        w: (D,) 判别方向（已归一化）
        threshold: 最优分类阈值
        separation: 类间分离度
    """
    mu_s = np.mean(X_signal, axis=0)
    mu_b = np.mean(X_background, axis=0)

    cov_s = np.cov(X_signal, rowvar=False, bias=True)
    cov_b = np.cov(X_background, rowvar=False, bias=True)

    if cov_s.ndim == 0:
        cov_s = np.atleast_2d(cov_s)
    if cov_b.ndim == 0:
        cov_b = np.atleast_2d(cov_b)

    # 确保二维
    if X_signal.shape[1] == 1:
        cov_s = cov_s.reshape(1, 1) if cov_s.size == 1 else cov_s
        cov_b = cov_b.reshape(1, 1) if cov_b.size == 1 else cov_b

    Sw = cov_s + cov_b
    # 正则化避免奇异
    Sw += 1.0e-6 * np.eye(Sw.shape[0])

    try:
        w = np.linalg.solve(Sw, mu_s - mu_b)
    except np.linalg.LinAlgError:
        w = mu_s - mu_b

    # 归一化
    norm = np.linalg.norm(w)
    if norm > 1.0e-15:
        w = w / norm

    # 投影
    proj_s = X_signal @ w
    proj_b = X_background @ w

    # 最优阈值（两类投影均值的中点）
    threshold = 0.5 * (np.mean(proj_s) + np.mean(proj_b))
    separation = abs(np.mean(proj_s) - np.mean(proj_b)) / np.sqrt(np.var(proj_s) + np.var(proj_b))

    return w, float(threshold), float(separation)


def apply_discriminant_cut(
    X: np.ndarray,
    w: np.ndarray,
    threshold: float,
    direction: str = ">",
) -> np.ndarray:
    """
    应用 Fisher 判别切割。

    参数：
        X: (N, D) 特征矩阵
        w: (D,) 判别方向
        threshold: 阈值
        direction: ">" 或 "<"

    返回：
        mask: (N,) bool 数组，True 表示通过切割
    """
    proj = X @ w
    if direction == ">":
        return proj > threshold
    else:
        return proj < threshold


# ============================================================================
# 背景抑制评估
# ============================================================================

def evaluate_background_rejection(
    signal_events: List[Dict],
    background_events: List[Dict],
    w: np.ndarray,
    threshold: float,
    target_efficiency: float = 0.9,
) -> Dict:
    """
    评估背景抑制性能。

    参数：
        signal_events: 信号事件列表
        background_events: 背景事件列表
        w: Fisher 判别方向
        threshold: 判别阈值
        target_efficiency: 目标信号效率

    返回：
        results: 包含 efficiency, rejection, purity 的字典
    """
    X_s = extract_event_features(signal_events)
    X_b = extract_event_features(background_events)

    mask_s = apply_discriminant_cut(X_s, w, threshold, ">")
    mask_b = apply_discriminant_cut(X_b, w, threshold, ">")

    n_s_pass = int(np.sum(mask_s))
    n_b_pass = int(np.sum(mask_b))
    n_s_total = len(signal_events)
    n_b_total = len(background_events)

    efficiency = n_s_pass / n_s_total if n_s_total > 0 else 0.0
    rejection = n_b_total / n_b_pass if n_b_pass > 0 else np.inf
    purity = n_s_pass / (n_s_pass + n_b_pass) if (n_s_pass + n_b_pass) > 0 else 0.0

    return {
        "signal_efficiency": float(efficiency),
        "background_rejection": float(rejection),
        "purity": float(purity),
        "n_signal_pass": n_s_pass,
        "n_background_pass": n_b_pass,
        "target_efficiency": target_efficiency,
    }


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 构造两类人工数据
    np.random.seed(42)
    X_s = np.random.randn(50, 4) + np.array([2.0, 0.0, 0.0, 0.0])
    X_b = np.random.randn(50, 4)

    # 测试距离矩阵
    D = build_distance_matrix(X_s[:5])
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T), "距离矩阵不对称"
    assert np.all(np.diag(D) == 0.0), "对角线应为零"

    # 测试聚类
    linkage, labels = single_linkage_clustering(D)
    assert len(linkage) == 4, "连接矩阵长度应为 N-1"

    # 测试 Fisher 判别
    w, thr, sep = fisher_discriminant(X_s, X_b)
    assert sep > 0.5, f"分离度过低: {sep}"

    mask = apply_discriminant_cut(np.vstack([X_s, X_b]), w, thr, ">")
    assert np.sum(mask[:50]) > np.sum(mask[50:]), "信号应主要落在阈值上方"

    print("event_reconstruction.py: 所有自测通过")
