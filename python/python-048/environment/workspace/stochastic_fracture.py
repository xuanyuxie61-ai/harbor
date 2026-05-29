"""
stochastic_fracture.py
随机裂缝网络生成模块

原项目映射:
    1007_random_sorted -> 有序正态随机向量生成
    1039_rng_cliff     -> Cliff 伪随机数生成器
    672_lights_out     -> 离散网格上的模 2 矩阵（用于裂缝连通性判定）

在微地震监测中，压裂产生的裂缝网络具有强烈的随机性与尺度层次结构。
本模块实现:
1. 基于有序随机分布的裂缝尺寸、方位角抽样；
2. 基于 Cliff RNG 的替代性随机源；
3. 基于 Lights Out 型模 2 矩阵的离散网格连通性分析，
   判断裂缝网络是否形成贯通簇（percolation cluster）。

核心公式:
1. 有序正态抽样（Bentley-Saxe 算法）:
   生成 N 个独立 N(0,1) 变量后排序等价于:
   U_{(1)} < U_{(2)} < ... < U_{(N)} 为有序均匀变量，
   通过逆 CDF 变换 X_{(i)} = Φ^{-1}(U_{(i)}) 得到有序正态变量。
   逆误差函数近似:
   Φ^{-1}(p) ≈ t - (c0 + c1 t + c2 t^2) / (1 + d1 t + d2 t^2 + d3 t^3)
   其中 t = sqrt(-2 ln(1-p))。

2. Cliff 随机数生成器:
   x_{n+1} = mod(-100 * ln(x_n), 1.0)
   当 x_n ∈ (0,1) 时产生确定性混沌序列，
   可用于随机裂缝位置的替代性生成。

3. 裂缝尺寸截断幂律分布:
   PDF(a) = (D_f / a_min) (a / a_min)^{-(D_f+1)}, a >= a_min

4. 离散网格连通性（Lights Out 代数）:
   在 M×N 网格上定义邻接矩阵 A（模 2），
   A_{ij} = 1 当格子 i 与 j 相邻（四邻域）或同一格子。
   裂缝占据向量 b ∈ {0,1}^{MN}，
   连通性判定转化为求解 A x = b (mod 2) 的可解性分析。
"""

import numpy as np
from scipy.special import erfc
from typing import Tuple, List


def normal_01_cdf_inv(p: float) -> float:
    """
    标准正态逆 CDF 的近似（Peter J. Acklam 近似）。

    公式:
        t = sqrt(-2 * ln(1-p))
        x = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0

    # Acklam 近似系数
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # 一次 Newton 修正
    e = 0.5 * erfc(-x / np.sqrt(2.0)) - p
    u = e * np.sqrt(2.0 * np.pi) * np.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return float(x)


def r8vec_normal_01_sorted(n: int) -> np.ndarray:
    """
    生成 n 个升序排列的标准正态随机变量。
    """
    if n <= 0:
        return np.array([])
    # 均匀有序抽样
    u = np.sort(np.random.uniform(0.0, 1.0, size=n))
    return np.array([normal_01_cdf_inv(float(ui)) for ui in u])


def rng_cliff_next(x: float) -> float:
    """
    Cliff 伪随机数生成器的单步迭代。

    公式:
        x_{next} = mod(-100 * ln(x), 1.0),  x ∈ (0,1)
    """
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return np.mod(-100.0 * np.log(x), 1.0)


def cliff_sequence(n: int, seed: float = 0.314159265) -> np.ndarray:
    """
    生成长度为 n 的 Cliff 序列。
    """
    if not (0.0 < seed < 1.0):
        seed = 0.314159265
    seq = np.zeros(n)
    x = seed
    for i in range(n):
        x = rng_cliff_next(x)
        if np.isnan(x):
            x = 0.5
        seq[i] = x
    return seq


def lights_out_matrix(mrow: int, ncol: int) -> np.ndarray:
    """
    构造 Lights Out 型邻接矩阵（模 2 意义下）。

    在 mrow × ncol 网格上，每个格子受自身及四邻域影响。
    矩阵 A 满足: A_{c, k} = 1 当格子 c 与 k 相邻或同一格子。

    参数:
        mrow, ncol: 网格行列数。

    返回:
        A: (mrow*ncol, mrow*ncol) 的 0/1 整数矩阵。
    """
    N = mrow * ncol
    A = np.zeros((N, N), dtype=int)

    def idx(i: int, j: int) -> int:
        if 1 <= i <= mrow and 1 <= j <= ncol:
            return (i - 1) * ncol + (j - 1)
        return -1

    for i in range(1, mrow + 1):
        for j in range(1, ncol + 1):
            c = idx(i, j)
            A[c, c] = 1
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nbr = idx(i + di, j + dj)
                if nbr >= 0:
                    A[nbr, c] = 1
    return A


def connectivity_mod2(occupied: np.ndarray, mrow: int, ncol: int) -> bool:
    """
    判断离散网格上被占据的格子是否形成至少一个贯穿连通簇。

    使用 BFS（非模 2 意义下，直接连通性）判断是否存在从左侧到右侧的通路。
    这里 Lights Out 矩阵用于构建邻接关系，实际连通性通过 BFS 遍历。
    """
    if occupied.size != mrow * ncol:
        raise ValueError("occupied 长度必须等于 mrow*ncol")
    grid = occupied.reshape((mrow, ncol)).astype(bool)

    # BFS 从左侧所有被占据格子出发
    visited = np.zeros((mrow, ncol), dtype=bool)
    queue = []
    for i in range(mrow):
        if grid[i, 0]:
            queue.append((i, 0))
            visited[i, 0] = True

    while queue:
        i, j = queue.pop(0)
        if j == ncol - 1:
            return True
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < mrow and 0 <= nj < ncol:
                if grid[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
    return False


def generate_fracture_network_params(num_fractures: int,
                                     a_min: float = 0.5,
                                     D_f: float = 2.2) -> dict:
    """
    生成裂缝网络的几何参数集合。

    返回:
        dict 包含:
            lengths: 裂缝半长 (m)
            strikes: 走向角 (°)
            dips: 倾角 (°)
            positions: 中心位置 [x,y,z]
    """
    # 使用有序正态抽样赋予裂缝尺寸层次结构
    sorted_normals = r8vec_normal_01_sorted(num_fractures)
    # 映射为截断幂律型长度: a = a_min * exp(|Z|)
    lengths = a_min * np.exp(np.abs(sorted_normals))

    # 方位角均匀分布，用 Cliff RNG 提供替代性扰动
    cliff = cliff_sequence(num_fractures, seed=0.271828182)
    strikes = 360.0 * cliff
    dips = 90.0 * np.random.uniform(0.0, 1.0, size=num_fractures)

    # 位置：在储层范围内均匀但有聚类趋势
    positions = np.random.randn(num_fractures, 3)
    positions[:, 0] *= 200.0  # x: 走向展布
    positions[:, 1] *= 50.0   # y: 倾向展布
    positions[:, 2] *= 30.0   # z: 深度变化

    return {
        "lengths": lengths,
        "strikes": strikes,
        "dips": dips,
        "positions": positions,
    }
