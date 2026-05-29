"""
numeric_utils.py
数值工具模块

融合原项目:
- 1208_test_int_2d: 2D Legendre-Gauss 数值积分
- 910_prime: 素数生成与随机种子管理

功能:
1. 2D Legendre-Gauss 积分用于计算聚合物链的径向分布函数积分
2. 素数表生成用于高维随机数种子和哈希寻址
3. 边界安全函数与数值稳定性工具
"""

import numpy as np
from math import sqrt, isclose
from typing import Callable, Tuple

# =============================================================================
# 素数相关: 融合 910_prime
# =============================================================================

def prime_count(n: int) -> int:
    """
    计算不超过 n 的素数个数（素数计数函数 π(n)）。
    
    数学背景:
        素数定理给出渐近估计: π(n) ~ n / ln(n)
    
    参数:
        n: 上限整数，n >= 2
    
    返回:
        素数个数
    """
    if n < 2:
        return 0
    if n == 2:
        return 1
    total = 1  # 2 is prime
    for i in range(3, n + 1, 2):
        p = 1
        limit = int(sqrt(i)) + 1
        for j in range(3, limit, 2):
            if i % j == 0:
                p = 0
                break
        total += p
    return total


def generate_primes(n: int) -> np.ndarray:
    """
    生成前 n 个素数序列，用于高维随机数种子。
    
    参数:
        n: 需要的素数个数
    
    返回:
        长度为 n 的素数数组
    """
    if n <= 0:
        return np.array([], dtype=int)
    primes = []
    candidate = 2
    while len(primes) < n:
        is_p = True
        limit = int(sqrt(candidate)) + 1
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return np.array(primes, dtype=int)


def seeded_random(seed_prime_index: int, size: Tuple[int, ...]) -> np.ndarray:
    """
    基于素数索引生成确定性伪随机数。
    利用素数序列初始化 numpy 的 RandomState，保证可重复性。
    
    参数:
        seed_prime_index: 使用第 seed_prime_index 个素数作为种子
        size: 输出数组形状
    
    返回:
        [0,1) 区间均匀分布随机数组
    """
    if seed_prime_index < 0:
        seed_prime_index = 0
    primes = generate_primes(seed_prime_index + 1)
    seed_val = int(primes[-1])
    rng = np.random.RandomState(seed=seed_val)
    return rng.rand(*size)


# =============================================================================
# 2D Legendre-Gauss 积分: 融合 1208_test_int_2d
# =============================================================================

def legendre_gauss_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Legendre-Gauss 积分节点和权重。
    
    数学公式:
        在区间 [-1,1] 上，Legendre 多项式 P_n(x) 的零点 x_i 为积分节点，
        权重 w_i = 2 / [(1 - x_i^2) * (P_n'(x_i))^2]
    
    参数:
        n: 积分点数
    
    返回:
        (nodes, weights): 节点和权重数组，长度均为 n
    """
    if n < 1:
        raise ValueError("legendre_gauss_nodes: n 必须 >= 1")
    if n > 100:
        # 对于大 n 使用 numpy 的 polynomial 模块
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return nodes, weights
    
    # 使用 numpy 的通用实现
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def integrate_2d_gauss(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int = 16,
    ny: int = 16,
) -> float:
    """
    2D Legendre-Gauss 数值积分。
    
    数学公式:
        I = ∫_{xlim}∫_{ylim} f(x,y) dy dx
          ≈ Σ_i Σ_j w_i w_j f(x_i, y_j) * (bx-ax)/2 * (by-ay)/2
    
    其中通过变量替换:
        x = (b-a)/2 * t + (a+b)/2,  t ∈ [-1,1]
    
    参数:
        f: 被积函数 f(x,y)，接受二维数组返回数组
        xlim: x 积分区间 (a,b)
        ylim: y 积分区间 (c,d)
        nx: x 方向积分点数
        ny: y 方向积分点数
    
    返回:
        积分估计值
    """
    if nx < 1 or ny < 1:
        raise ValueError("integrate_2d_gauss: nx, ny 必须 >= 1")
    
    ax, bx = xlim
    ay, by = ylim
    
    if not (np.isfinite(ax) and np.isfinite(bx) and np.isfinite(ay) and np.isfinite(by)):
        raise ValueError("integrate_2d_gauss: 积分限必须为有限值")
    
    if isclose(bx, ax) or isclose(by, ay):
        return 0.0
    
    x_nodes, x_weights = legendre_gauss_nodes(nx)
    y_nodes, y_weights = legendre_gauss_nodes(ny)
    
    # 仿射变换到实际区间
    x = 0.5 * (bx - ax) * x_nodes + 0.5 * (bx + ax)
    y = 0.5 * (by - ay) * y_nodes + 0.5 * (by + ay)
    wx = 0.5 * (bx - ax) * x_weights
    wy = 0.5 * (by - ay) * y_weights
    
    # 构建网格
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 计算函数值，加入边界处理
    with np.errstate(invalid='ignore', divide='ignore'):
        F = f(X, Y)
    
    # 将 nan/inf 替换为 0（数值鲁棒性）
    if np.any(~np.isfinite(F)):
        F = np.where(np.isfinite(F), F, 0.0)
    
    # 计算二重积分
    integral = 0.0
    for i in range(nx):
        for j in range(ny):
            integral += wx[i] * wy[j] * F[i, j]
    
    return float(integral)


# =============================================================================
# 通用数值安全工具
# =============================================================================

def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    安全除法，避免除零。
    
    参数:
        a: 分子数组
        b: 分母数组
        fill_value: 除零时填充值
    
    返回:
        a/b 的安全结果
    """
    b = np.asarray(b)
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    return result


def soft_cutoff(r: np.ndarray, rc: float, delta: float = 0.05) -> np.ndarray:
    """
    平滑截断函数，用于分子动力学势能在截断半径处的连续过渡。
    
    公式:
        f(r) = 1,                         r < rc - delta
        f(r) = 0.5 + 0.5*cos(π*(r-rc+delta)/delta),  rc-delta <= r <= rc
        f(r) = 0,                         r > rc
    
    参数:
        r: 距离数组
        rc: 截断半径
        delta: 过渡区宽度
    
    返回:
        平滑截断权重
    """
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=float)
    mask_in = r < (rc - delta)
    mask_trans = (r >= (rc - delta)) & (r <= rc)
    result[mask_in] = 1.0
    if np.any(mask_trans):
        result[mask_trans] = 0.5 + 0.5 * np.cos(np.pi * (r[mask_trans] - rc + delta) / delta)
    return result


def distance_matrix_pbc(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    考虑周期性边界条件（PBC）的距离矩阵计算。
    
    公式:
        dr_ij = r_i - r_j
        dr_ij = dr_ij - box * round(dr_ij / box)  (最小像约定)
        d_ij = |dr_ij|
    
    参数:
        positions: (N, dim) 位置数组
        box: (dim,) 模拟盒子边长
    
    返回:
        (N, N) 距离矩阵
    """
    N = positions.shape[0]
    dim = positions.shape[1]
    if box.shape[0] != dim:
        raise ValueError("box 维度与 positions 不匹配")
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # 最小像约定
    for d in range(dim):
        if box[d] > 0:
            diff[:, :, d] -= box[d] * np.rint(diff[:, :, d] / box[d])
    
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def mean_squared_displacement(trajectory: np.ndarray) -> np.ndarray:
    """
    计算均方位移 (MSD)。
    
    公式:
        MSD(τ) = < |r(t+τ) - r(t)|^2 >_t
    
    参数:
        trajectory: (n_frames, n_particles, dim) 轨迹数组
    
    返回:
        MSD 数组，长度为 n_frames
    """
    n_frames = trajectory.shape[0]
    if n_frames < 2:
        return np.zeros(n_frames)
    
    msd = np.zeros(n_frames)
    counts = np.zeros(n_frames)
    
    for dt in range(1, n_frames):
        displacements = trajectory[dt:, :, :] - trajectory[:-dt, :, :]
        sq_disp = np.sum(displacements ** 2, axis=2)
        msd[dt] = np.mean(sq_disp)
        counts[dt] = sq_disp.size
    
    msd[0] = 0.0
    counts[0] = 1.0
    return msd
