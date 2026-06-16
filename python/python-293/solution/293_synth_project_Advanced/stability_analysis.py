# -*- coding: utf-8 -*-
"""
stability_analysis.py
======================
数值稳定性分析与等离子体波色散关系

融合种子项目:
  1142_gitvicky_Loss_Landscape_PINNs — 损失景观分析 / PINNs 数值方法
  1362_truncated_normal_sparse_grid   — 稀疏网格积分 / 截断正态

物理背景
--------
1. von Neumann 稳定性分析:
   对差分格式做 Fourier 分析，计算放大因子 G(k):
     f_j^n = G^n exp(ikjΔx)
   稳定性条件: |G(k)| ≤ 1 + O(Δt) 对所有 k

2. 等离子体色散关系:
   Bohm-Gross:  ω² = ω_p² + 3k²v_th²
   Landau 阻尼: Im(ω) = -√(π/8) ω_p / (kλ_D)³ exp(-1/(2k²λ_D²) - 3/2)

3. Hankel 矩阵特征值分析:
   从电场时间序列提取阻尼率和频率 (Prony 方法)

4. Two-stream 不稳定性:
   双束分布函数 → 正增长率 → 指数增长
"""

import numpy as np
import math
from typing import Tuple, Dict, Optional


class NumericalStabilityAnalyzer:
    """
    von Neumann 稳定性分析器

    对各类差分格式计算放大因子 G(θ):
      上风格式:    G = 1 - ν(1 - e^{-iθ})
      中心差分:    G = 1 - iν sin(θ)
      Lax-Wendroff: G = 1 - iν sin(θ) - ν²(1 - cos(θ))
      Leapfrog:    G² - 2iν sin(θ) G - 1 = 0

    其中 ν = CFL = uΔt/Δx, θ = kΔx
    """

    def __init__(self, scheme: str = 'centered'):
        """
        Parameters
        ----------
        scheme : str
            'upwind', 'centered', 'lax_wendroff', 'leapfrog'
        """
        valid = ['upwind', 'centered', 'lax_wendroff', 'leapfrog']
        if scheme not in valid:
            raise ValueError(f"Unknown scheme '{scheme}', valid: {valid}")
        self.scheme = scheme

    def amplification_factor(self, nu: float,
                               theta: np.ndarray) -> np.ndarray:
        """
        计算放大因子 G(θ; ν)

        Parameters
        ----------
        nu : float
            CFL 数 ν = uΔt/Δx.
        theta : np.ndarray
            归一化波数 θ = kΔx ∈ [0, 2π].

        Returns
        -------
        np.ndarray (complex)
            放大因子 G.
        """
        theta = np.asarray(theta, dtype=np.float64)
        e_ith = np.exp(-1j * theta)

        if self.scheme == 'upwind':
            return 1.0 - nu * (1.0 - e_ith)
        elif self.scheme == 'centered':
            return 1.0 - 1j * nu * np.sin(theta)
        elif self.scheme == 'lax_wendroff':
            return (1.0 - 1j * nu * np.sin(theta)
                    - nu ** 2 * (1.0 - np.cos(theta)))
        elif self.scheme == 'leapfrog':
            # G² - 2iν sin(θ) G - 1 = 0
            disc = -4.0 * nu ** 2 * np.sin(theta) ** 2 + 4.0
            disc = np.maximum(disc, 0.0)
            G1 = 1j * nu * np.sin(theta) + np.sqrt(disc) / 2.0
            return G1
        return np.ones_like(theta, dtype=complex)

    def stability_boundary(self, n_theta: int = 1000
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        确定格式的稳定性边界 |G| ≤ 1

        Returns
        -------
        nu_range : np.ndarray
            CFL 数范围.
        max_G : np.ndarray
            每个 ν 对应的 max_θ |G(θ;ν)|.
        """
        theta = np.linspace(0, 2.0 * math.pi, n_theta)
        nu_range = np.linspace(0.01, 2.0, 200)
        max_G = np.zeros_like(nu_range)

        for idx, nu in enumerate(nu_range):
            G = self.amplification_factor(nu, theta)
            max_G[idx] = np.max(np.abs(G))

        return nu_range, max_G

    def is_stable(self, nu: float, tol: float = 1.001) -> bool:
        """判断给定 CFL 数下格式是否稳定."""
        theta = np.linspace(0, 2.0 * math.pi, 1000)
        G = self.amplification_factor(nu, theta)
        return bool(np.max(np.abs(G)) <= tol)


class PlasmaDispersionSolver:
    """
    等离子体波色散关系求解器

    色散关系:
      D(ω, k) = 1 + (1/(k²λ_D²)) [1 + ζ Z(ζ)] = 0
    其中 ζ = ω / (k v_th √2)

    Bohm-Gross 近似:
      ω² ≈ ω_p² + 3k²v_th²  (实部)
      γ ≈ -√(π/8) (ω_p/(kλ_D)³) exp(-1/(2k²λ_D²) - 3/2)  (虚部)
    """

    def __init__(self, omega_p: float = 1.0, v_th: float = 1.0):
        self.omega_p = omega_p
        self.v_th = v_th
        self.lambda_D = v_th / omega_p

    def bohmgross_frequency(self, k: float) -> float:
        """
        Bohm-Gross 频率 (实部近似):

        .. math:: \\omega^2 = \\omega_p^2 + 3 k^2 v_{th}^2
        """
        return math.sqrt(self.omega_p ** 2 + 3.0 * k ** 2 * self.v_th ** 2)

    def landau_damping_rate(self, k: float) -> float:
        """
        Landau 阻尼率 (弱阻尼近似):

        .. math::
            \\gamma = -\\sqrt{\\frac{\\pi}{8}}
                      \\frac{\\omega_p}{(k\\lambda_D)^3}
                      \\exp\\left(-\\frac{1}{2(k\\lambda_D)^2} - \\frac{3}{2}\\right)
        """
        if abs(k) < 1.0e-300:
            return 0.0
        kl = k * self.lambda_D
        if abs(kl) < 1.0e-10:
            return 0.0
        omega = self.bohmgross_frequency(k)
        zeta = omega / (abs(k) * self.v_th * math.sqrt(2.0))
        gamma = (-math.sqrt(math.pi / 8.0) * omega
                 / (abs(kl) ** 3 + 1e-300)
                 * math.exp(-zeta * zeta))
        return gamma

    def solve_dispersion(self, k: float) -> complex:
        """
        数值求解色散关系 D(ω,k) = 0

        使用 Newton 迭代，初始猜测为 Bohm-Gross 近似。

        Returns
        -------
        complex
            ω = ω_r + i γ
        """
        from grid_integer_lib import plasma_dispersion_z

        omega_r = self.bohmgross_frequency(k)
        gamma_0 = self.landau_damping_rate(k)
        omega = complex(omega_r, gamma_0)

        # Newton 迭代
        for _ in range(50):
            if abs(k) < 1e-300:
                break
            zeta = omega / (k * self.v_th * math.sqrt(2.0))
            Z = plasma_dispersion_z(zeta)
            D = 1.0 + 1.0 / (k * self.lambda_D) ** 2 * (1.0 + zeta * Z)

            # 数值导数
            deps = 1e-7 * abs(omega) + 1e-20
            zeta_p = (omega + deps) / (k * self.v_th * math.sqrt(2.0))
            Z_p = plasma_dispersion_z(zeta_p)
            D_p = 1.0 + 1.0 / (k * self.lambda_D) ** 2 * (1.0 + zeta_p * Z_p)
            dD = (D_p - D) / deps

            if abs(dD) < 1e-300:
                break
            delta = D / dD
            omega -= delta
            if abs(delta) < 1e-12 * abs(omega) + 1e-20:
                break

        return omega

    def penetration_depth(self, omega: complex,
                           nu_coll: float = 0.01) -> float:
        """
        Langmuir 波趋肤深度: δ = v_th / √(ν_c² + ω²)
        """
        return self.v_th / math.sqrt(
            nu_coll ** 2 + abs(omega) ** 2 + 1e-300)


class HankelModeDecomposition:
    """
    Hankel 矩阵特征模分解

    从电场时间序列 E(t) 提取阻尼率和频率:

    1. 构建 Hankel 矩阵: H_{ij} = E(t_{i+j})
    2. SVD 分解: H = U Σ V^T
    3. 从特征值提取:
         ω = Im(ln(λ)) / Δt     (频率)
         γ = ln|λ| / Δt         (阻尼率)

    此方法等价于 Prony 分析 / ESPRIT 算法。
    """

    def __init__(self, signal: np.ndarray, dt: float,
                 n_modes: int = 5):
        self.signal = np.asarray(signal, dtype=np.float64)
        self.dt = dt
        self.n_modes = min(n_modes, len(signal) // 3)

    def decompose(self) -> Dict:
        """执行 Hankel 分解."""
        N = len(self.signal)
        L = min(N // 2, max(3 * self.n_modes, 10))
        K = N - L + 1

        # 构建 Hankel 矩阵
        H = np.zeros((L, K), dtype=np.float64)
        for i in range(L):
            for j in range(K):
                if i + j < N:
                    H[i, j] = self.signal[i + j]

        # SVD
        U, s, Vt = np.linalg.svd(H, full_matrices=False)

        # 取前 n_modes 个奇异值
        n_eff = min(self.n_modes, len(s))

        # 从移位矩阵提取特征值
        U1 = U[:-1, :n_eff]
        U2 = U[1:, :n_eff]
        try:
            S_inv = np.diag(1.0 / np.maximum(s[:n_eff], 1e-300))
            A = U1.T @ U2 @ Vt[:n_eff, :].T @ Vt[:n_eff, :] @ S_inv
            # 简化: 使用 U1 的伪逆
            A = np.linalg.pinv(U1) @ U2
            eigenvalues = np.linalg.eigvals(A)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(n_eff, dtype=complex) * 0.5

        # 提取频率和阻尼率
        modes = []
        for lam in eigenvalues:
            abs_lam = max(abs(lam), 1e-300)
            gamma = math.log(abs_lam) / self.dt
            omega = math.atan2(lam.imag, lam.real) / self.dt
            modes.append({
                'frequency': omega,
                'damping_rate': gamma,
                'amplitude': abs_lam,
            })

        modes.sort(key=lambda m: -abs(m['damping_rate']))

        return {
            'modes': modes[:n_eff],
            'singular_values': s[:n_eff],
            'eigenvalues': eigenvalues[:n_eff],
        }


class TwoStreamInstabilityAnalyzer:
    """
    双流不稳定性分析

    双束分布函数:
      f₀(v) = n₁/(√(2π)v_th) exp(-(v-v_b)²/(2v_th²))
            + n₂/(√(2π)v_th) exp(-(v+v_b)²/(2v_th²))

    色散关系:
      1 = ω_p²/(ω-kv_b)² + ω_p²/(ω+kv_b)²  (冷束极限)

    最大增长率发生在:
      k_max ≈ ω_p / (√3 v_b)  (对于 v_b >> v_th)
    """

    def __init__(self, v_b: float = 2.0, v_th: float = 0.5,
                 omega_p: float = 1.0):
        self.v_b = v_b
        self.v_th = v_th
        self.omega_p = omega_p

    def growth_rate(self, k: float) -> float:
        """
        计算给定波数 k 的增长率 Im(ω)

        使用流体近似:
          ω² = ω_p² + k²v_th² ± k²v_b·ω_p / √(1 + k²λ_D²)
        """
        if abs(k) < 1.0e-300:
            return 0.0

        wp2 = self.omega_p ** 2
        kvt2 = (k * self.v_th) ** 2
        kvb = k * self.v_b

        disc = wp2 ** 2 + 4.0 * wp2 * kvb ** 2
        disc = max(disc, 0.0)

        omega2_plus = 0.5 * (2.0 * wp2 + kvt2 + math.sqrt(disc))
        omega2_minus = 0.5 * (2.0 * wp2 + kvt2 - math.sqrt(disc))

        if omega2_minus < 0:
            return math.sqrt(abs(omega2_minus))
        return 0.0

    def most_unstable_mode(self, n_k: int = 200
                            ) -> Tuple[float, float]:
        """
        寻找最大增长率对应的波数和增长率

        Returns
        -------
        k_max : float
        gamma_max : float
        """
        k_max_range = np.linspace(0.01, 5.0 / max(self.v_th, 0.01), n_k)
        gammas = np.array([self.growth_rate(k) for k in k_max_range])

        idx_max = np.argmax(gammas)
        return float(k_max_range[idx_max]), float(gammas[idx_max])


class SparseGridStability:
    """
    稀疏网格稳定性分析

    使用 Smolyak 稀疏网格求积来评估高维速度空间中的
    数值稳定性条件。

    稀疏网格相比全张量积网格, 节点数从 O(N^d) 降至 O(N log(N)^{d-1}),
    但需要仔细分析积分精度对稳定性的影响。
    """

    @staticmethod
    def gauss_hermite_nodes_weights(level: int
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Hermite 求积节点和权重 (阶数 = 2*level - 1)

        用于速度空间积分:
          ∫ f(v) exp(-v²) dv ≈ Σ w_i f(v_i)
        """
        if level < 1:
            return np.array([0.0]), np.array([math.sqrt(math.pi)])

        nodes, weights = np.polynomial.hermite.hermgauss(level)
        return nodes, weights

    @staticmethod
    def smolyak_combination(level: int, dim: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smolyak 稀疏网格组合公式

        A_q(f) = Σ_{q-d+1 ≤ |i| ≤ q} (-1)^{q-|i|} C(d-1, q-|i|)
                 (Q_{i1} ⊗ ... ⊗ Q_{id})(f)

        Parameters
        ----------
        level : int
            稀疏网格层级.
        dim : int
            维度.

        Returns
        -------
        nodes : np.ndarray, shape (N, dim)
        weights : np.ndarray, shape (N,)
        """
        if dim == 1:
            return SparseGridStability.gauss_hermite_nodes_weights(level)

        # 递归构建 (1D → 2D 张量积简化)
        n1d, w1d = SparseGridStability.gauss_hermite_nodes_weights(level)

        if dim == 2:
            nn, ww = np.meshgrid(n1d, n1d, indexing='ij')
            wn, wn_w = np.meshgrid(w1d, w1d, indexing='ij')
            nodes = np.column_stack([nn.ravel(), ww.ravel()])
            weights = wn.ravel() * wn_w.ravel()
            return nodes, weights

        # 高维: 简化为嵌套 1D
        n_total = len(n1d) ** dim
        nodes = np.zeros((n_total, dim))
        weights = np.ones(n_total)
        for d in range(dim):
            idx = np.tile(np.repeat(n1d, len(n1d) ** (dim - d - 1)),
                          len(n1d) ** d)
            nodes[:, d] = idx[:n_total]
            w_idx = np.tile(np.repeat(w1d, len(w1d) ** (dim - d - 1)),
                            len(w1d) ** d)
            weights *= w_idx[:n_total]

        return nodes, weights
