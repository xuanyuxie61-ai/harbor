"""
disorder_config.py

基于 truncated_normal (1360_truncated_normal)、square_surface_distance (1150_square_surface_distance)、
hypersphere (562_hypersphere) 的无序与热涨落采样模块。

在凝聚态物理中，Hubbard 模型常需考虑:
1. 在位无序势: ε_i ~ 截断正态分布 (Anderson 型无序)
2. 热初始构型采样: 从高温态随机抽取自旋构型
3. 高维参数空间中的蒙特卡洛随机游走 (hypersphere 均匀采样)
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple


# ---------------------------------------------------------------------------
# truncated_normal: 截断正态采样
# ---------------------------------------------------------------------------

def truncated_normal_ab_sample(mu: float, sigma: float, a: float, b: float, size: int = 1) -> np.ndarray:
    """
    在区间 [a, b] 上采样截断正态分布。
    
    算法:
        α = (a - μ)/σ, β = (b - μ)/σ
        Φ_α = Φ(α), Φ_β = Φ(β)
        u ~ Uniform(0,1)
        ξ = Φ^{-1}(Φ_α + u(Φ_β - Φ_α))
        x = μ + σ ξ
    """
    if sigma <= 0:
        raise ValueError("sigma > 0 required")
    if a >= b:
        raise ValueError("a < b required")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_cdf = norm.cdf(alpha)
    beta_cdf = norm.cdf(beta)
    if alpha_cdf >= beta_cdf:
        return np.full(size, mu)
    u = np.random.rand(size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)
    # 数值稳定性: 截断极端值
    xi_cdf = np.clip(xi_cdf, 1e-10, 1 - 1e-10)
    xi = norm.ppf(xi_cdf)
    return mu + sigma * xi


def truncated_normal_a_sample(mu: float, sigma: float, a: float, size: int = 1) -> np.ndarray:
    """单侧截断 [a, ∞)。"""
    return truncated_normal_ab_sample(mu, sigma, a, mu + 10.0 * sigma, size)


def truncated_normal_b_sample(mu: float, sigma: float, b: float, size: int = 1) -> np.ndarray:
    """单侧截断 (-∞, b]。"""
    return truncated_normal_ab_sample(mu, sigma, mu - 10.0 * sigma, b, size)


def generate_anderson_disorder(nsites: int, W: float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """
    生成 Anderson 型在位无序势:
        ε_i ~ TruncatedNormal(μ, σ, -W/2, W/2)
    
    参数:
        nsites: 格点数
        W: 无序强度 (分布宽度)
        mu: 中心
        sigma: 标准差
    """
    if W <= 0:
        raise ValueError("W > 0 required")
    return truncated_normal_ab_sample(mu, sigma, -W / 2.0, W / 2.0, size=nsites)


# ---------------------------------------------------------------------------
# square_surface_distance: 方边界采样
# ---------------------------------------------------------------------------

def square_surface_sample(n: int) -> np.ndarray:
    """
    在单位方边界上均匀采样 n 个点。
    用于开放边界条件的边界格点选取。
    
    返回:
        p: 形状 (n, 2)
    """
    if n < 1:
        raise ValueError("n >= 1 required")
    p = np.random.rand(n, 2)
    # 随机选择 x 或 y 坐标固定在边界 (0 或 1)
    i = np.random.randint(0, 2, size=n)
    s = np.random.randint(0, 2, size=n)
    # 将选定的坐标轴设为 0 或 1
    for idx in range(n):
        p[idx, i[idx]] = float(s[idx])
    return p


def boundary_site_indices(nx: int, ny: int) -> np.ndarray:
    """返回 nx × ny 方格子的边界格点索引。"""
    indices = []
    for iy in range(ny):
        for ix in range(nx):
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                indices.append(ix + iy * nx)
    return np.array(indices, dtype=int)


# ---------------------------------------------------------------------------
# hypersphere: 高维球面热态采样
# ---------------------------------------------------------------------------

def thermal_spin_configuration(nsites: int, beta: float, J: float = 1.0) -> np.ndarray:
    """
    从高斯随机场生成经典自旋初始构型，
    再投影到单位球面 (Marsaglia 方法)。
    
    自旋关联长度 ξ ~ 1/sqrt(β J)。
    在高温 (β→0) 下自旋随机取向；低温下趋于铁磁有序。
    """
    if nsites < 1:
        raise ValueError("nsites >= 1")
    if beta < 0:
        raise ValueError("beta >= 0")
    # 高斯随机场
    spins = np.random.randn(nsites, 3)
    # 投影到单位球面
    norms = np.sqrt(np.sum(spins ** 2, axis=1))
    norms = np.where(norms > 0, norms, 1.0)
    spins = spins / norms[:, np.newaxis]
    # 低温下加有序偏置
    if beta * J > 1.0:
        bias = np.tanh(beta * J)
        spins[:, 2] = spins[:, 2] * (1.0 - bias) + bias
        norms = np.sqrt(np.sum(spins ** 2, axis=1))
        norms = np.where(norms > 0, norms, 1.0)
        spins = spins / norms[:, np.newaxis]
    return spins


def random_phase_vector(dim: int) -> np.ndarray:
    """
    在单位复圆上均匀采样相位，
    用于 Gutzwiller 投影变分波函数的初始相位。
    """
    if dim < 1:
        raise ValueError("dim >= 1")
    theta = np.random.uniform(0, 2.0 * np.pi, size=dim)
    return np.exp(1j * theta)


def disordered_hubbard_parameters(nsites: int, W: float, U_base: float, U_var: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成空间变化的 Hubbard 参数:
        ε_i ~ TruncatedNormal(0, W, -W, W)   (在位无序)
        U_i = U_base + δU_i, δU_i ~ TruncatedNormal(0, U_var, -U_base, U_base)
    """
    epsilon = generate_anderson_disorder(nsites, W, mu=0.0, sigma=W / 3.0)
    delta_U = truncated_normal_ab_sample(0.0, U_var, -U_base * 0.5, U_base * 0.5, size=nsites)
    U = U_base + delta_U
    U = np.clip(U, 0.1, None)  # U 必须为正
    return epsilon, U


if __name__ == "__main__":
    eps, U = disordered_hubbard_parameters(10, W=2.0, U_base=4.0, U_var=0.5)
    print(f"Disorder mean eps={np.mean(eps):.3f}, U mean={np.mean(U):.3f}")
    sp = thermal_spin_configuration(8, beta=1.0)
    print(f"Spin norms: {np.mean(np.sqrt(np.sum(sp**2, axis=1))):.6f}")
