"""
transverse_field_basis.py
================================================================================
横向场（transverse-field）下的量子力学基函数与隧穿振幅计算。

融合来源：523_hermite_product_display（厄米特多项式）

物理背景：
  量子退火中的驱动哈密顿量为

      H_D = -Γ(t) Σ_i σ_i^x

  在单粒子图像下，横向场使自旋在 |↑⟩ 与 |↓⟩ 之间隧穿。
  将 σ^x 对角化，其本征态对应于谐振子基 |n⟩，可用厄米特多项式
  H_n(x) 描述。在二次量子化框架下，隧穿算符的矩阵元为

      ⟨m| exp( -i θ σ^x ) |n⟩ = Σ_k  (iθ)^k / k!  *  (σ^x)^k_{mn}

  而 σ^x 可用升降算符表示：σ^x = a† + a。

核心公式——物理学家厄米特多项式：
    H_0(x) = 1
    H_1(x) = 2x
    H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)

正交归一化：
    ∫_{-∞}^{+∞} exp(-x^2) H_m(x) H_n(x) dx = sqrt(pi) 2^n n! δ_{mn}

概率学家厄米特多项式（更常用于量子光学）：
    He_0(x) = 1
    He_1(x) = x
    He_n(x) = x He_{n-1}(x) - (n-1) He_{n-2}(x)

正交归一化：
    ∫_{-∞}^{+∞} exp(-x^2/2) He_m(x) He_n(x) dx = sqrt(2π) n! δ_{mn}
"""

import numpy as np
from typing import Tuple


def physicist_hermite_polynomials(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    计算物理学家厄米特多项式 H_n(x)，n = 0, ..., max_degree。

    返回数组 shape 为 (len(x), max_degree+1)。
    """
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    x = np.asarray(x, dtype=float)
    m = x.size
    p = np.zeros((m, max_degree + 1))
    p[:, 0] = 1.0
    if max_degree == 0:
        return p
    p[:, 1] = 2.0 * x
    for j in range(2, max_degree + 1):
        p[:, j] = 2.0 * x * p[:, j - 1] - 2.0 * (j - 1) * p[:, j - 2]
    return p


def probabilist_hermite_polynomials(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    计算概率学家厄米特多项式 He_n(x)，n = 0, ..., max_degree。
    """
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    x = np.asarray(x, dtype=float)
    m = x.size
    p = np.zeros((m, max_degree + 1))
    p[:, 0] = 1.0
    if max_degree == 0:
        return p
    p[:, 1] = x
    for j in range(2, max_degree + 1):
        p[:, j] = x * p[:, j - 1] - (j - 1) * p[:, j - 2]
    return p


def normalized_probabilist_hermite(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    归一化概率学家厄米特多项式，满足：
        ∫ exp(-x^2/2) Hen_m(x) Hen_n(x) dx = δ_{mn}
    """
    p = probabilist_hermite_polynomials(x, max_degree)
    # 归一化因子 1/sqrt( sqrt(2π) n! )
    from math import factorial, sqrt, pi
    norms = np.array([1.0 / sqrt(sqrt(2.0 * pi) * factorial(n)) for n in range(max_degree + 1)])
    return p * norms[np.newaxis, :]


def hermite_function_basis(x: np.ndarray, max_degree: int) -> np.ndarray:
    """
    厄米特函数基（量子谐振子本征函数在位置表象）：

        ψ_n(x) = H_n(x) exp(-x^2/2) / sqrt( 2^n n! sqrt(pi) )

    满足正交归一化 ∫ ψ_m(x) ψ_n(x) dx = δ_{mn}。
    """
    p = physicist_hermite_polynomials(x, max_degree)
    from math import factorial, sqrt, pi
    m = x.size
    f = np.zeros((m, max_degree + 1))
    f[:, 0] = np.exp(-0.5 * x ** 2) / sqrt(sqrt(pi))
    if max_degree == 0:
        return f
    f[:, 1] = 2.0 * x * np.exp(-0.5 * x ** 2) / sqrt(2.0 * sqrt(pi))
    for j in range(2, max_degree + 1):
        f[:, j] = (np.sqrt(2.0) * x * f[:, j - 1] - np.sqrt(j - 1) * f[:, j - 2]) / np.sqrt(j)
    return f


def tunneling_amplitude_1d(x: float, gamma: float, n_basis: int = 8) -> float:
    """
    计算一维双势阱中横向场 Γ 导致的基态-第一激发态隧穿振幅：

        Δ = ⟨0| exp( -γ σ^x ) |1⟩ ≈ Σ_{n=0}^{n_basis} c_n H_n(γ)

    其中系数 c_n 由势阱曲率在鞍点展开确定。
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    # 在 γ 处采样厄米特多项式
    hvals = physicist_hermite_polynomials(np.array([gamma]), n_basis)[0, :]
    # 简化模型：隧穿振幅 ≈ γ * exp(-γ^2) * Σ_{k=0}^{n_basis/2} (-1)^k γ^{2k} / k!
    # 这里直接用级数展开估算
    import math
    amp = 0.0
    for k in range(n_basis + 1):
        amp += hvals[k] * ((-1.0) ** k) / float(math.factorial(max(k, 1)))
    # 归一化边界
    amp = float(np.clip(amp, -10.0, 10.0))
    return amp


def transverse_field_hamiltonian_dense(n_spins: int, gamma: float) -> np.ndarray:
    """
    构造稠密横向场哈密顿量矩阵 H_D = -Γ Σ_i σ_i^x 在计算基 |z⟩ 下的表示。

    维度 D = 2^n_spins，仅适用于 n_spins ≤ 12（内存限制）。

    矩阵元：
        ⟨s'| H_D |s⟩ = -Γ  if s' 与 s 仅相差 1 位（单自旋翻转）
                       0   otherwise
    """
    if n_spins > 12:
        raise ValueError("Dense H_D construction only for n_spins <= 12")
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        for i in range(n_spins):
            flipped = state ^ (1 << i)
            H[state, flipped] -= gamma
    return H


def basis_change_matrix(n_spins: int, max_local_dim: int = 4) -> np.ndarray:
    """
    构建从计算基到局域厄米特函数基的变换矩阵块。

    对每个自旋 i，定义局域基 |n_i⟩ (n_i = 0, ..., max_local_dim-1)。
    变换矩阵元为厄米特函数在离散点上的取值。
    """
    if max_local_dim < 1:
        raise ValueError("max_local_dim must be >= 1")
    # 离散点取 Chebyshev 节点在 [-3,3]
    pts = np.cos(np.pi * (2 * np.arange(max_local_dim) + 1) / (2 * max_local_dim)) * 3.0
    psi = hermite_function_basis(pts, max_local_dim - 1)
    # psi shape (max_local_dim, max_local_dim)
    return psi


class TunnelingKernel:
    """
    隧穿核：用于量子蒙特卡洛路径积分中的世界线更新权重。
    """

    def __init__(self, beta: float, gamma: float, n_slices: int, n_basis: int = 8):
        if beta <= 0 or gamma < 0 or n_slices <= 0:
            raise ValueError("Invalid physical parameters")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n_slices = int(n_slices)
        self.n_basis = n_basis
        self.dtau = beta / n_slices

    def kinetic_matrix_element(self, s_left: int, s_right: int) -> float:
        """
        计算虚时间片之间横向场导致的跃迁矩阵元（Trotter 分解）：

            ⟨s_{τ+Δτ} | exp( -Δτ H_D ) | s_τ⟩ ≈
                exp( Δτ Γ Σ_i s_i^{τ} s_i^{τ+Δτ} )   (对 small Δτ)

        严格表达式使用双曲函数：
            ⟨s'| exp( a σ^x ) |s⟩ = cosh(a) δ_{s,s'} + sinh(a) δ_{s,-s'}
        """
        a = self.dtau * self.gamma
        if s_left == s_right:
            return np.cosh(a)
        else:
            return np.sinh(a)

    def path_weight_ratio(self, config_old: np.ndarray, config_new: np.ndarray,
                          slice_idx: int) -> float:
        """
        单自旋翻转 (slice_idx) 时的路径权重变化比。

        对 Trotter 切片 m，涉及相邻切片 m-1, m, m+1 的矩阵元：

            W = Π_{m}  K( s_{m-1}, s_m ) * exp( -Δτ H_classical(s_m) )

        当仅改变 s_m 时，权重比为
            R = K(s_{m-1}, s_m') K(s_m', s_{m+1}) / [ K(s_{m-1}, s_m) K(s_m, s_{m+1}) ]
                * exp( -Δτ [ E(s_m') - E(s_m) ] )
        """
        if config_old.shape != config_new.shape:
            raise ValueError("config shapes must match")
        if not np.all(np.isin(config_old, [-1, 1])):
            raise ValueError("configs must be +/-1")

        s_old = int(config_old[slice_idx])
        s_new = int(config_new[slice_idx])
        if s_old == s_new:
            return 1.0

        n = config_old.size
        prev_idx = (slice_idx - 1) % n
        next_idx = (slice_idx + 1) % n

        a = self.dtau * self.gamma
        # 边界处理：假设周期边界
        ratio = (
            (np.cosh(a) if config_old[prev_idx] == s_new else np.sinh(a)) *
            (np.cosh(a) if s_new == config_old[next_idx] else np.sinh(a))
        ) / (
            (np.cosh(a) if config_old[prev_idx] == s_old else np.sinh(a)) *
            (np.cosh(a) if s_old == config_old[next_idx] else np.sinh(a))
        )
        return float(ratio)
