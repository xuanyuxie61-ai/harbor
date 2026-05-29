"""
spherical_embedding.py
资产相关性的球面嵌入与几何风险分析模块。

融入的原项目核心算法：
- 307_distance_to_position_sphere: 球面距离到经纬度坐标的非线性最小二乘
- 181_circle_monte_carlo: 单位圆上的随机采样与单变量积分
- 306_distance_to_position: 欧氏距离到坐标的MDS嵌入

科学背景：
在现代投资组合理论中，资产相关性矩阵可视为高维球面 S^{d-1} 上
点集的内积结构。通过球面多维尺度分析（Spherical MDS），
可将资产映射到球面，利用球面几何度量资产间的"角度距离"，
进而构建基于几何分散度的风险指标。
"""

import numpy as np
from scipy.optimize import least_squares


def sphere_distance1(lat1: float, lon1: float, lat2: float, lon2: float,
                      r: float = 1.0) -> float:
    """
    计算球面上两点间的大圆距离（Haversine公式）：

        d = r * arccos( sin(φ_1) sin(φ_2) + cos(φ_1) cos(φ_2) cos(Δλ) )

    其中 φ 为纬度，λ 为经度，r 为球半径。
    """
    # 数值稳定性：使用 arctan2 形式避免 arccos 的定义域问题
    cos_term = (np.sin(lat1) * np.sin(lat2) +
                np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    cos_term = np.clip(cos_term, -1.0, 1.0)
    return r * np.arccos(cos_term)


def ll_to_xyz(r: float, ll: np.ndarray) -> np.ndarray:
    """
    将经纬度 (φ, λ) 转换为笛卡尔坐标 (x, y, z)：

        x = r cos(λ) cos(φ)
        y = r sin(λ) cos(φ)
        z = -r sin(φ)

    参数
    ----------
    r : float
        球半径。
    ll : np.ndarray, shape (2, n)
        经纬度数组（弧度）。

    返回
    -------
    np.ndarray, shape (3, n)
        笛卡尔坐标。
    """
    n = ll.shape[1]
    xyz = np.zeros((3, n))
    xyz[0, :] = r * np.cos(ll[1, :]) * np.cos(ll[0, :])
    xyz[1, :] = r * np.sin(ll[1, :]) * np.cos(ll[0, :])
    xyz[2, :] = -r * np.sin(ll[0, :])
    return xyz


def xyz_to_ll(xyz: np.ndarray, r: float = 1.0) -> np.ndarray:
    """
    将笛卡尔坐标转换回经纬度。
    """
    x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
    lat = np.arcsin(np.clip(-z / r, -1.0, 1.0))
    lon = np.arctan2(y, x)
    return np.vstack([lat, lon])


def map_spherical_residual(ll_vec: np.ndarray, r: float, city_num: int,
                           distance: np.ndarray) -> np.ndarray:
    """
    球面嵌入的残差函数，用于 least_squares 优化。

    为消除刚性变换自由度，固定：
    - 城市1在 (0, 0)
    - 城市2的经度为 0

    残差由两部分组成：
    1. 固定约束残差（N1 = 3）
    2. 距离差异残差（N2 = n(n-1)/2）
    """
    ll = ll_vec.reshape(2, city_num)
    n1 = 3
    n2 = (city_num * (city_num - 1)) // 2
    f = np.zeros(n1 + n2)
    k = 0
    # 城市1固定在原点
    f[k] = ll[0, 0]
    k += 1
    f[k] = ll[1, 0]
    k += 1
    # 城市2经度为0
    f[k] = ll[1, 1]
    k += 1
    # 距离差异
    for i in range(city_num):
        for j in range(i + 1, city_num):
            d_computed = sphere_distance1(ll[0, i], ll[1, i],
                                           ll[0, j], ll[1, j], r)
            f[k] = distance[i, j] - d_computed
            k += 1
    return f


def correlation_to_spherical_embedding(corr: np.ndarray, r: float = 1.0,
                                        random_seed: int = 42) -> np.ndarray:
    """
    将资产相关性矩阵通过球面嵌入映射到三维球面 S^2。

    数学模型：
    对相关性矩阵 C，定义球面距离
        d_{ij} = arccos( C_{ij} ) ∈ [0, π]。
    该距离对应于单位球面上两点间的中心角。
    我们通过非线性最小二乘求解经纬度坐标 LL，使得
        || d_{ij} - d_{sphere}(LL_i, LL_j) ||^2 → min。

    参数
    ----------
    corr : np.ndarray, shape (n, n)
        相关性矩阵，对角线为1，取值 [-1, 1]。
    r : float
        球半径。
    random_seed : int
        随机种子。

    返回
    -------
    np.ndarray, shape (3, n)
        球面嵌入的笛卡尔坐标。
    """
    rng = np.random.default_rng(random_seed)
    n = corr.shape[0]
    if n < 3:
        raise ValueError("correlation_to_spherical_embedding: 资产数至少为3。")
    # 对称化并截断
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    # 球面距离
    distance = np.arccos(np.clip(corr, -1.0, 1.0))
    # 初始猜测
    ll0 = rng.random(2 * n)
    result = least_squares(
        lambda vec: map_spherical_residual(vec, r, n, distance),
        ll0,
        method="lm",
        max_nfev=2000 * n,
        ftol=1e-10,
        xtol=1e-10,
    )
    ll = result.x.reshape(2, n)
    xyz = ll_to_xyz(r, ll)
    return xyz


def circle01_sample_random(n: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    在单位圆 S^1 上均匀随机采样 n 个点。

        θ ~ U[0, 1)，
        x = [cos(2πθ), sin(2πθ)]^T。

    返回
    -------
    np.ndarray, shape (2, n)
        采样点坐标。
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.random(n)
    x = np.zeros((2, n))
    x[0, :] = np.cos(2.0 * np.pi * theta)
    x[1, :] = np.sin(2.0 * np.pi * theta)
    return x


def spherical_diversity_index(xyz: np.ndarray) -> float:
    """
    计算资产在球面上的几何分散度指标。

    定义：对 n 个单位球面上的点 {p_i}，分散度为
        D = (1 / n^2) Σ_{i,j} || p_i - p_j ||^2
          = 2 - 2 || (1/n) Σ_i p_i ||^2。

    当所有点重合时 D = 0；当点均匀分布时 D → 2。
    """
    n = xyz.shape[1]
    if n == 0:
        return 0.0
    centroid = np.mean(xyz, axis=1)
    norm_c = np.linalg.norm(centroid)
    diversity = 2.0 - 2.0 * norm_c ** 2
    return float(max(diversity, 0.0))


def angular_distance_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    计算球面点集的角度距离矩阵。

        d_{ij} = arccos( <p_i, p_j> / (||p_i|| ||p_j||) )。
    """
    norms = np.linalg.norm(xyz, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("angular_distance_matrix: 存在零范数向量。")
    unit = xyz / norms
    inner = np.clip(unit.T @ unit, -1.0, 1.0)
    return np.arccos(inner)
