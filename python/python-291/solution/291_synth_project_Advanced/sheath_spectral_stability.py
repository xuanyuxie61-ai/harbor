"""
sheath_spectral_stability.py
============================
谱稳定性分析模块。

本模块实现等离子体鞘层的数值稳定性分析：
    1. 构造线性化稳定性矩阵
    2. 特征值谱计算
    3. von Neumann 稳定性分析
    4. 数值色散/耗散诊断
    5. 伪谱分析

核心公式：
    线性化扰动方程：
        ∂u'/∂t = L u'
    其中 L 为线性化算子矩阵，u' = (φ', n_i', u_i')^T

    特征值问题：
        L v = λ v
    λ = σ + iω
        σ > 0 → 不稳定
        σ < 0 → 稳定
        σ = 0 → 边际稳定

    von Neumann 稳定性条件：
        ρ(I + Δt L) ≤ 1
    其中 ρ 为谱半径
"""

import numpy as np
from scipy import linalg as la
from typing import Tuple, List, Optional
import math

from fd_highorder import compact_fd_matrix_4th, fd_second_deriv_matrix


class SpectralStabilityAnalyzer:
    """
    谱稳定性分析器

    对鞘层平衡态进行线性稳定性分析
    """

    def __init__(
        self,
        x: np.ndarray,
        phi_eq: np.ndarray,
        n_i_eq: np.ndarray,
        n_e_eq: np.ndarray,
        params=None,
    ):
        """
        参数：
            x: 网格坐标
            phi_eq: 平衡态电势
            n_i_eq: 平衡态离子密度
            n_e_eq: 平衡态电子密度
            params: PlasmaParams 对象
        """
        self.x = x
        self.N = len(x)
        self.phi_eq = phi_eq
        self.n_i_eq = n_i_eq
        self.n_e_eq = n_e_eq
        self.dx = (x[-1] - x[0]) / (self.N - 1) if self.N > 1 else 1.0

        if params is not None:
            self.chi = params.chi_see
            self.gamma_see = params.gamma_e
            self.mu = params.mass_ratio
            self.u_bohm = 1.0
        else:
            self.chi = 1.0
            self.gamma_see = 0.05
            self.mu = 1e-5
            self.u_bohm = 1.0

        # 构造微分矩阵
        self.D1 = compact_fd_matrix_4th(self.N, self.dx)
        self.D2 = fd_second_deriv_matrix(x, order=2)

    def build_stability_matrix(self) -> np.ndarray:
        """
        构造线性化稳定性矩阵

        线性化流体-泊松方程组：
            ∂n_i'/∂t + ∂(n_i0 u_i')/∂x + ∂(n_i' u_i0)/∂x = 0
            ∂u_i'/∂t + u_i0 ∂u_i'/∂x = -∂φ'/∂x
            ∂²φ'/∂x² = n_e'(φ') - n_i'

        状态向量：U = (φ', n_i', u_i')^T
        维度：3N × 3N

        线性化矩阵结构：
            L = [ L_φφ   L_φn   L_φu ]
                [ L_nφ   L_nn   L_nu ]
                [ L_uφ   L_un   L_uu ]
        """
        N = self.N
        L = np.zeros((3 * N, 3 * N))

        # 分块索引
        # φ: 0..N-1
        # n_i: N..2N-1
        # u_i: 2N..3N-1

        # === Poisson 方程行：D² φ' = n_e'(φ') - n_i' ===
        # L_φφ: D² + ∂n_e/∂φ
        dne_dphi = self._compute_dne_dphi()
        for i in range(N):
            L[i, i] += dne_dphi[i]
        L[:N, :N] += self.D2

        # L_φn: -I (离子密度扰动)
        L[:N, N:2*N] = -np.eye(N)

        # === 离子连续性方程行 ===
        # ∂n_i'/∂t = -∂(n_i0 u_i')/∂x - ∂(n_i' u_i0)/∂x
        # u_i0 ≈ u_bohm (常数漂移)
        L[N:2*N, N:2*N] = -self.u_bohm * self.D1  # -u_0 · ∂/∂x
        L[N:2*N, 2*N:3*N] = -np.diag(self.n_i_eq) @ self.D1  # -n_i0 · ∂/∂x

        # === 离子动量方程行 ===
        # ∂u_i'/∂t = -∂φ'/∂x - u_i0 ∂u_i'/∂x
        L[2*N:3*N, :N] = -self.D1  # -∂φ'/∂x
        L[2*N:3*N, 2*N:3*N] = -self.u_bohm * self.D1  # -u_0 · ∂/∂x

        # 边界条件 (Dirichlet)
        for block in range(3):
            offset = block * N
            L[offset, :] = 0.0
            L[offset, offset] = 1.0
            L[offset + N - 1, :] = 0.0
            L[offset + N - 1, offset + N - 1] = 1.0

        return L

    def _compute_dne_dphi(self) -> np.ndarray:
        """计算 ∂n_e/∂φ 在平衡态的值"""
        phi = np.clip(self.phi_eq, -50.0, 0.0)
        chi_safe = max(self.chi, 1e-6)
        gamma_safe = min(max(self.gamma_see, 0.0), 0.99)

        dne = (
            (1.0 - gamma_safe) / chi_safe * np.exp(phi / chi_safe)
            + gamma_safe * np.exp(phi)
        )
        return dne

    def compute_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算稳定性矩阵的特征值

        返回：
            eigenvalues: 复数特征值
            eigenvectors: 特征向量矩阵
        """
        L = self.build_stability_matrix()
        eigenvalues, eigenvectors = la.eig(L)

        # 按实部排序
        idx = np.argsort(-eigenvalues.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def stability_diagnosis(self) -> dict:
        """
        稳定性诊断

        返回：
            max_growth_rate: 最大增长率
            n_unstable: 不稳定模数
            spectral_radius: 谱半径
            cfl_limit: CFL 时间步长限制
            is_stable: 是否数值稳定
        """
        eigenvalues, _ = self.compute_eigenvalues()

        growth_rates = eigenvalues.real
        frequencies = eigenvalues.imag

        max_growth = np.max(growth_rates)
        n_unstable = np.sum(growth_rates > 1e-10)

        spectral_radius = np.max(np.abs(eigenvalues))

        # CFL 限制 (对于显式 Euler)
        cfl_dt = 1.0 / spectral_radius if spectral_radius > 1e-15 else float('inf')

        return {
            'max_growth_rate': float(max_growth),
            'n_unstable': int(n_unstable),
            'spectral_radius': float(spectral_radius),
            'cfl_dt_limit': float(cfl_dt),
            'is_stable': max_growth < 1e-10,
            'eigenvalues_real': growth_rates[:20].tolist(),
            'eigenvalues_imag': frequencies[:20].tolist(),
        }

    def von_neumann_analysis(
        self,
        dt: float,
        scheme: str = "euler",
    ) -> dict:
        """
        von Neumann 稳定性分析

        对于时间离散格式：
            Euler 显式：G = I + dt L
            Euler 隐式：G = (I - dt L)^{-1}
            Crank-Nicolson：G = (I - dt/2 L)^{-1} (I + dt/2 L)

        稳定条件：ρ(G) ≤ 1

        参数：
            dt: 时间步长
            scheme: "euler" | "implicit" | "cn"

        返回：
            诊断结果字典
        """
        L = self.build_stability_matrix()
        N_total = L.shape[0]

        if scheme == "euler":
            G = np.eye(N_total) + dt * L
        elif scheme == "implicit":
            G = la.solve(np.eye(N_total) - dt * L, np.eye(N_total))
        elif scheme == "cn":
            A = np.eye(N_total) - 0.5 * dt * L
            B = np.eye(N_total) + 0.5 * dt * L
            G = la.solve(A, B)
        else:
            raise ValueError(f"未知格式: {scheme}")

        eigenvalues_G = la.eigvals(G)
        amplification = np.abs(eigenvalues_G)
        rho_G = np.max(amplification)

        return {
            'scheme': scheme,
            'dt': dt,
            'spectral_radius_G': float(rho_G),
            'max_amplification': float(np.max(amplification)),
            'min_amplification': float(np.min(amplification)),
            'is_stable': rho_G <= 1.0 + 1e-10,
            'n_unstable_modes': int(np.sum(amplification > 1.0 + 1e-10)),
        }

    def pseudospectrum(
        self,
        epsilon: float = 1e-3,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ε-伪谱计算

        ε-伪谱定义：
            Λ_ε(L) = {z ∈ ℂ : ||(zI - L)^{-1}|| > 1/ε}

        等价条件：
            σ_min(zI - L) < ε

        参数：
            epsilon: 伪谱半径参数
            n_points: 网格点数

        返回：
            Z_real, Z_imag: 伪谱边界点
        """
        L = self.build_stability_matrix()
        eigenvalues = la.eigvals(L)

        # 计算范围
        re_min = np.min(eigenvalues.real) - 2
        re_max = np.max(eigenvalues.real) + 2
        im_min = np.min(eigenvalues.imag) - 2
        im_max = np.max(eigenvalues.imag) + 2

        re_grid = np.linspace(re_min, re_max, n_points)
        im_grid = np.linspace(im_min, im_max, n_points)

        # 计算最小奇异值
        sigma_min_grid = np.zeros((n_points, n_points))
        for i, re in enumerate(re_grid):
            for j, im in enumerate(im_grid):
                z = complex(re, im)
                M = z * np.eye(L.shape[0]) - L
                try:
                    s = la.svdvals(M)
                    sigma_min_grid[i, j] = np.min(s)
                except la.LinAlgError:
                    sigma_min_grid[i, j] = epsilon

        # 提取 ε-伪谱边界
        boundary_mask = np.abs(sigma_min_grid - epsilon) < epsilon * 0.5
        boundary_points = np.where(boundary_mask)

        if len(boundary_points[0]) > 0:
            Z_real = re_grid[boundary_points[0]]
            Z_imag = im_grid[boundary_points[1]]
        else:
            Z_real = np.array([])
            Z_imag = np.array([])

        return Z_real, Z_imag


def compute_dispersion_from_stability(
    analyzer: SpectralStabilityAnalyzer,
    k_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从稳定性矩阵提取数值色散关系

    参数：
        analyzer: 稳定性分析器
        k_values: 波数数组

    返回：
        omega_real: 频率实部
        gamma_imag: 增长率虚部
    """
    eigenvalues, _ = analyzer.compute_eigenvalues()

    omega_real = eigenvalues.imag
    gamma_imag = eigenvalues.real

    return omega_real, gamma_imag
