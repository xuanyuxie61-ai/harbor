"""
latent_path.py
==============
基于种子项目 1363_tsp_brute 的隐空间路径优化模块。
在物理信息 GAN 的隐空间中，通过旅行商问题（TSP）的最优路径
实现生成样本间的平滑过渡，用于隐空间插值与生成序列化。

核心数学：
  1. 旅行商问题（TSP）：
       给定 n 个城市及距离矩阵 D，寻找排列 π 使得总路径长度最小：
         C(π) = Σ_{i=1}^{n} D_{π(i), π((i mod n)+1)}
       精确解需要枚举 (n-1)!/2 个排列（对称性约化）。

  2. Trotter 排列生成算法（Algorithm 115, CACM 1962）：
       使用相邻换位（adjacent transposition）依次生成所有排列，
       保证相邻排列之间仅交换两个相邻元素（Gray code 性质）。

  3. 隐空间插值：
       给定排序后的隐向量序列 z_1, z_2, ..., z_n，
       线性插值：z(t) = (1-t)·z_i + t·z_{i+1},  t ∈ [0,1]
       球面插值（Slerp）：用于单位球面约束隐空间。
         z(t) = sin((1-t)·θ)/sin(θ)·z_i + sin(t·θ)/sin(θ)·z_{i+1}
         其中 cos(θ) = z_i·z_{j+1}。
"""

import numpy as np


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    """
    计算 TSP 排列 p 的总路径成本。

    Parameters
    ----------
    n : int
        城市数。
    distance : np.ndarray, shape (n, n)
        距离矩阵。
    p : np.ndarray, shape (n,)
        排列（1-based 索引）。

    Returns
    -------
    cost : float
        总路径长度。
    """
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        idx1 = int(p[i1]) - 1
        idx2 = int(p[i2]) - 1
        cost += distance[idx1, idx2]
        i1 = i2
    return float(cost)


def perm1_next3(n: int, p: np.ndarray, more: bool, rank: int) -> tuple:
    """
    Trotter 相邻换位排列生成算法。

    Parameters
    ----------
    n : int
        排列长度。
    p : np.ndarray, shape (n,)
        当前排列（1-based）。
    more : bool
        是否还有更多排列。
    rank : int
        当前排列的秩。

    Returns
    -------
    p, more, rank : np.ndarray, bool, int
        下一个排列。
    """
    if not more:
        p = np.arange(1, n + 1)
        more = True
        rank = 1
        return p, more, rank

    n2 = n
    m2 = rank
    s = n

    while True:
        q = m2 % n2
        t = m2 % (2 * n2)
        if q != 0:
            break
        if t == 0:
            s -= 1
        m2 = m2 // n2
        n2 -= 1
        if n2 == 0:
            p = np.arange(1, n + 1)
            more = False
            rank = 1
            return p, more, rank

    if n2 != 0:
        if q == t:
            s -= q
        else:
            s = s + q - n2
        # Python 0-based 索引交换
        idx1 = s - 1
        idx2 = s
        tmp = p[idx1]
        p[idx1] = p[idx2]
        p[idx2] = tmp
        rank += 1

    return p, more, rank


def tsp_brute(distance: np.ndarray) -> tuple:
    """
    穷举法求解小规模对称 TSP。

    Parameters
    ----------
    distance : np.ndarray, shape (n, n)
        对称距离矩阵，对角线为 0。

    Returns
    -------
    p_min : np.ndarray
        最优排列（1-based）。
    total_min : float
        最小路径长度。
    total_ave : float
        平均路径长度。
    """
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    if n < 2:
        return np.array([1]), 0.0, 0.0

    total_max = -np.inf
    total_min = np.inf
    total_ave = 0.0
    paths = 0

    p = np.arange(1, n + 1)
    more = False
    rank = 0

    while True:
        p, more, rank = perm1_next3(n, p, more, rank)
        if not more:
            break
        paths += 1
        total = path_cost(n, distance, p)
        total_ave += total
        if total > total_max:
            total_max = total
        if total < total_min:
            total_min = total
            p_min = np.copy(p)

    if paths == 0:
        return np.array([1]), 0.0, 0.0
    total_ave /= paths
    return p_min, float(total_min), float(total_ave)


def latent_space_tsp_path(vectors: np.ndarray) -> tuple:
    """
    对隐向量集合求解 TSP 最优排序，实现最平滑的生成过渡。

    Parameters
    ----------
    vectors : np.ndarray, shape (n, dim)
        隐向量集合。

    Returns
    -------
    ordered_vectors : np.ndarray, shape (n, dim)
        按 TSP 最优路径排序后的隐向量。
    path_cost_val : float
        最优路径总长度。
    """
    n = vectors.shape[0]
    if n <= 2:
        return vectors, 0.0
    # 计算欧氏距离矩阵
    diff = vectors[:, None, :] - vectors[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # 对于 n > 8，穷举计算量过大，使用贪心最近邻近似
    if n > 8:
        return _greedy_tsp_path(vectors, dist)
    p_min, total_min, _ = tsp_brute(dist)
    ordered = vectors[p_min - 1]
    return ordered, float(total_min)


def _greedy_tsp_path(vectors: np.ndarray, dist: np.ndarray) -> tuple:
    """贪心最近邻 TSP 近似。"""
    n = vectors.shape[0]
    visited = [False] * n
    current = 0
    path = [current]
    visited[current] = True
    total = 0.0
    for _ in range(n - 1):
        # 寻找最近的未访问城市
        nearest = -1
        min_dist = np.inf
        for j in range(n):
            if not visited[j] and dist[current, j] < min_dist:
                min_dist = dist[current, j]
                nearest = j
        if nearest == -1:
            break
        total += min_dist
        visited[nearest] = True
        path.append(nearest)
        current = nearest
    # 回到起点
    total += dist[current, path[0]]
    ordered = vectors[path]
    return ordered, float(total)


def slerp(z1: np.ndarray, z2: np.ndarray, t: float) -> np.ndarray:
    """
    球面线性插值（Spherical Linear Interpolation）。

    Parameters
    ----------
    z1, z2 : np.ndarray, shape (dim,)
        单位向量。
    t : float
        插值参数，t ∈ [0, 1]。

    Returns
    -------
    z : np.ndarray
        插值结果。
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    z1 = z1 / (np.linalg.norm(z1) + 1e-15)
    z2 = z2 / (np.linalg.norm(z2) + 1e-15)
    dot = np.clip(np.dot(z1, z2), -1.0, 1.0)
    theta = np.arccos(abs(dot))
    if theta < 1e-10:
        return (1.0 - t) * z1 + t * z2
    # 处理钝角
    if dot < 0.0:
        z2 = -z2
        dot = -dot
        theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        return (1.0 - t) * z1 + t * z2
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * z1 + b * z2


def latent_interpolation_sequence(ordered_vectors: np.ndarray,
                                  steps: int = 10) -> np.ndarray:
    """
    在已排序的隐向量序列之间进行 Slerp 插值，生成平滑过渡序列。

    Parameters
    ----------
    ordered_vectors : np.ndarray, shape (n, dim)
        已排序的隐向量。
    steps : int
        每对向量之间的插值步数。

    Returns
    -------
    sequence : np.ndarray, shape ((n-1)*steps+1, dim)
        插值序列。
    """
    n = ordered_vectors.shape[0]
    if n < 2:
        return ordered_vectors
    seq = []
    for i in range(n - 1):
        for s in range(steps):
            t = s / steps
            seq.append(slerp(ordered_vectors[i], ordered_vectors[i + 1], t))
    seq.append(ordered_vectors[-1])
    return np.array(seq)
