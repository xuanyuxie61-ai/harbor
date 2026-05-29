"""
majorana_wavefunction.py

基于种子项目 666_legendre_shifted_polynomial（移位Legendre多项式）
和 1173_string_pde（一维波动方程有限差分），实现马约拉纳零能模
波函数的谱展开与动力学演化。

物理模型：
    马约拉纳零能模的波函数ψ(x)满足BdG方程在E=0时的极限：
        [-μ(x) - t ∂_x^2] u(x) + Δ ∂_x v(x) = 0
        Δ ∂_x u(x) + [μ(x) + t ∂_x^2] v(x) = 0

    在均匀情况下（μ=const, Δ=const, t=const），解析解为：
        ψ(x) = ψ(0) * exp(-x/ξ),  ξ = 2t/|Δ|

    我们采用移位Legendre多项式作为基函数进行Galerkin展开：
        ψ(x) ≈ Σ_{n=0}^{N} a_n P̃_n(x)
    其中P̃_n(x) = P_n(2x/L - 1)为定义在[0,L]上的移位Legendre多项式。

    同时利用有限差分法模拟马约拉纳波包的时空演化（类比波动方程）。
"""

import numpy as np
from typing import Tuple, Optional


class ShiftedLegendreBasis:
    """
    移位Legendre多项式基函数。

    标准Legendre多项式P_n(y)定义在y∈[-1,1]，通过变换：
        y = 2x/L - 1,  x∈[0,L]
    得到移位Legendre多项式P̃_n(x) = P_n(2x/L - 1)。

    递推关系：
        P̃_0(x) = 1
        P̃_1(x) = 2x/L - 1
        (n+1) P̃_{n+1}(x) = (2n+1)(2x/L-1) P̃_n(x) - n P̃_{n-1}(x)

    正交性：
        ∫_0^L P̃_m(x) P̃_n(x) dx = L/(2n+1) δ_{mn}
    """

    def __init__(self, domain_length: float, max_degree: int):
        """
        初始化移位Legendre基。

        Args:
            domain_length: 空间域长度 L
            max_degree: 最高多项式阶数 N
        """
        if domain_length <= 0:
            raise ValueError("域长度必须为正")
        if max_degree < 0:
            raise ValueError("最高阶数必须非负")
        self.L = domain_length
        self.N = max_degree

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        在点x处计算0到N阶的移位Legendre多项式值。

        Args:
            x: 形状为(M,)的数组，x∈[0,L]

        Returns:
            v: 形状为(M, N+1)的数组，v[m,n] = P̃_n(x[m])
        """
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < -1e-12) or np.any(x > self.L + 1e-12):
            # 边界处理：截断到有效域
            x = np.clip(x, 0.0, self.L)

        m = len(x)
        n = self.N
        v = np.zeros((m, n + 1))

        v[:, 0] = 1.0
        if n >= 1:
            y = 2.0 * x / self.L - 1.0
            v[:, 1] = y
            for i in range(1, n):
                v[:, i + 1] = (
                    (2.0 * i + 1.0) * y * v[:, i]
                    - i * v[:, i - 1]
                ) / (i + 1.0)

        return v

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算移位Legendre多项式的一阶导数。

        利用关系式：
            d/dx P̃_n(x) = (2/L) * d/dy P_n(y)
            (2n+1) P_n(y) = d/dy P_{n+1}(y) - d/dy P_{n-1}(y)
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0.0, self.L)
        m = len(x)
        n = self.N

        dv = np.zeros((m, n + 1))
        if n >= 1:
            y = 2.0 * x / self.L - 1.0
            # d/dy P_0 = 0, d/dy P_1 = 1
            dp = np.zeros((m, n + 1))
            if n >= 1:
                dp[:, 1] = 1.0
            for i in range(1, n):
                dp[:, i + 1] = (
                    (2.0 * i + 1.0) * (y * dp[:, i] + v[:, i])
                    - i * dp[:, i - 1]
                ) / (i + 1.0)
            # 需要重新计算v
            v = self.evaluate(x)
            for i in range(1, n):
                dp[:, i + 1] = (
                    (2.0 * i + 1.0) * (y * dp[:, i] + v[:, i])
                    - i * dp[:, i - 1]
                ) / (i + 1.0)
            dv = (2.0 / self.L) * dp

        return dv

    def inner_product(self, f: np.ndarray, g: np.ndarray) -> float:
        """
        计算两个函数在[0,L]上的内积（Gauss-Legendre积分）。
        """
        # 使用高阶Gauss-Legendre求积
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(self.N + 5)
        # 变换到[0,L]
        x_gl = 0.5 * self.L * (nodes + 1.0)
        w_gl = 0.5 * self.L * weights
        return float(np.sum(f(x_gl) * g(x_gl) * w_gl))


class MajoranaWavefunctionSolver:
    """
    马约拉纳波函数的谱方法与有限差分求解器。
    """

    def __init__(self, length: float, n_sites: int,
                 mu: float, t: float, delta: float):
        """
        初始化求解器参数。

        Args:
            length: 纳米线长度 L (nm)
            n_sites: 空间离散点数
            mu: 化学势 (meV)
            t: 跃迁能 (meV)
            delta: 超导能隙 (meV)
        """
        self.L = length
        self.n = n_sites
        self.mu = mu
        self.t = t
        self.delta = delta
        self.dx = length / (n_sites - 1)

    def analytical_zero_mode_profile(self, x: np.ndarray) -> np.ndarray:
        """
        均匀Kitaev链中马约拉纳零能模的解析波函数。

        在开边界条件下，零能模局域在链的两端，满足：
            ψ(x) ∝ exp(-x/ξ),  ξ = 2t/|Δ|

        更精确地，离散格点上的解满足递推关系：
            u_{j+1} = (μ/2t) u_j - (Δ/2t) v_j
            v_{j+1} = -(Δ/2t) u_j + (μ/2t) v_j
        在E=0且|μ|<2t的拓扑区内，解呈指数衰减。
        """
        x = np.asarray(x, dtype=np.float64)
        xi = 2.0 * abs(self.t) / (abs(self.delta) + 1e-15)
        # 归一化波函数
        psi = np.exp(-x / xi)
        # 边界处理：在x=0处最大，在x=L处最小
        norm = np.sqrt(np.trapezoid(psi ** 2, x))
        if norm > 1e-15:
            psi /= norm
        return psi

    def spectral_expansion_coefficients(self, wavefunction: np.ndarray,
                                        max_degree: int = 20) -> np.ndarray:
        """
        将波函数投影到移位Legendre多项式基上。

        展开系数：
            a_n = (2n+1)/L ∫_0^L ψ(x) P̃_n(x) dx
        """
        basis = ShiftedLegendreBasis(self.L, max_degree)
        x = np.linspace(0.0, self.L, self.n)
        v = basis.evaluate(x)

        coeffs = np.zeros(max_degree + 1)
        for n in range(max_degree + 1):
            integrand = wavefunction * v[:, n]
            coeffs[n] = ((2.0 * n + 1.0) / self.L
                         * np.trapezoid(integrand, x))

        return coeffs

    def reconstruct_from_spectral(self, coeffs: np.ndarray,
                                   x: np.ndarray) -> np.ndarray:
        """
        由谱系数重构波函数。
        """
        max_degree = len(coeffs) - 1
        basis = ShiftedLegendreBasis(self.L, max_degree)
        v = basis.evaluate(x)
        return v @ coeffs

    def finite_difference_time_evolution(self,
                                          initial_wave: np.ndarray,
                                          num_steps: int,
                                          dt: float,
                                          alpha: float) -> np.ndarray:
        """
        基于种子项目1173_string_pde的有限差分算法，
        模拟马约拉纳波包的动力学演化。

        将波动方程类比为马约拉纳准粒子的传播：
            ∂_t^2 ψ = c^2 ∂_x^2 ψ - m^2 ψ
        其中有效质量m与化学势μ相关，波速c与跃迁t相关。

        离散化格式（显式中心差分）：
            ψ_j^{n+1} = 2(1-α)ψ_j^n + α(ψ_{j-1}^n + ψ_{j+1}^n) - ψ_j^{n-1}
            α = (c*dt/dx)^2

        稳定性条件（CFL条件）：
            α ≤ 1
        """
        if alpha < 0 or alpha > 1.0:
            raise ValueError("CFL参数alpha必须在[0,1]范围内")
        if len(initial_wave) != self.n:
            raise ValueError("初始波函数长度必须与格点数匹配")

        u = np.zeros((num_steps + 1, self.n))

        for j in range(num_steps + 1):
            if j == 0:
                u[0, 0] = 0.0
                u[0, 1:self.n - 1] = initial_wave[1:self.n - 1]
                u[0, self.n - 1] = 0.0
            elif j == 1:
                u[1, 0] = 0.0
                for i in range(1, self.n - 1):
                    # 初始时间导数假设为零（静止启动）
                    u[1, i] = (
                        0.5 * alpha * u[0, i - 1]
                        + (1.0 - alpha) * u[0, i]
                        + 0.5 * alpha * u[0, i + 1]
                    )
                u[1, self.n - 1] = 0.0
            else:
                u[j, 0] = 0.0
                for i in range(1, self.n - 1):
                    u[j, i] = (
                        alpha * u[j - 1, i - 1]
                        + 2.0 * (1.0 - alpha) * u[j - 1, i]
                        + alpha * u[j - 1, i + 1]
                        - u[j - 2, i]
                    )
                u[j, self.n - 1] = 0.0

        return u

    def compute_probability_current(self, wavefunction: np.ndarray) -> np.ndarray:
        """
        计算概率流密度（连续性方程）。

        在BdG形式下，概率流密度为：
            J(x) = (iħ/2m) [ψ* ∂_x ψ - ψ ∂_x ψ*]
                 = (ħ/m) Im[ψ* ∂_x ψ]

        对于实波函数（马约拉纳条件），J=0，体现粒子-空穴对称性。
        """
        psi = np.asarray(wavefunction, dtype=np.complex128)
        dpsi = np.zeros_like(psi)
        dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * self.dx)
        dpsi[0] = (psi[1] - psi[0]) / self.dx
        dpsi[-1] = (psi[-1] - psi[-2]) / self.dx

        j = np.imag(np.conj(psi) * dpsi)
        return j

    def overlap_integral(self, psi1: np.ndarray,
                         psi2: np.ndarray) -> complex:
        """
        计算两个波函数的重叠积分。

        <ψ1|ψ2> = ∫ ψ1*(x) ψ2(x) dx
        """
        if len(psi1) != len(psi2):
            raise ValueError("波函数长度必须相同")
        x = np.linspace(0.0, self.L, len(psi1))
        integrand = np.conj(psi1) * psi2
        return complex(np.trapezoid(integrand, x))


def demo():
    """演示马约拉纳波函数求解。"""
    solver = MajoranaWavefunctionSolver(
        length=100.0, n_sites=100, mu=0.5, t=1.0, delta=0.8
    )
    x = np.linspace(0.0, solver.L, solver.n)
    psi = solver.analytical_zero_mode_profile(x)
    print("Analytical MZM profile norm:", np.trapezoid(psi ** 2, x))

    coeffs = solver.spectral_expansion_coefficients(psi, max_degree=15)
    print("Spectral coefficients (first 5):", coeffs[:5])

    psi_recon = solver.reconstruct_from_spectral(coeffs, x)
    error = np.max(np.abs(psi - psi_recon))
    print("Spectral reconstruction max error:", error)

    # 有限差分演化
    u = solver.finite_difference_time_evolution(
        initial_wave=psi, num_steps=50, dt=0.01, alpha=0.25
    )
    print("FD evolution final wave energy:", np.trapezoid(u[-1] ** 2, x))


if __name__ == "__main__":
    demo()
