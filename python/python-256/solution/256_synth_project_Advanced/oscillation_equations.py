"""
oscillation_equations.py
========================
线性绝热恒星振荡方程的实现.

融合种子项目:
  - Lagrange (632): Lagrange 基函数及其导数 → 高阶插值基
  - FEM2D (412): 有限元组装 → 振荡方程的变分形式
  - biochemical_linear_ode (090): ODE 精确解 → 振荡方程的解析基准

科学背景
--------
线性绝热振荡方程 (Unno et al. 1989, Cox 1980):

对于球对称恒星中的非径向振荡, 扰动量可以分解为球谐函数 Y_l^m:

  ξ(r,θ,φ,t) = [ξ_r(r) e_r + ξ_h(r) e_h] Y_l^m(θ,φ) e^{iωt}

其中 ξ_r 为径向位移, ξ_h 为水平位移, ω 为角频率.

定义无量纲变量:
  y₁ = ξ_r / r                           (径向位移)
  y₂ = (1/c_s²)(Φ' - g ξ_r)              (Lagrange 压强扰动)
     ≈ P'/P - (ρg/P) ξ_r               (归一化 Lagrange 压强扰动)
  y₃ = δΦ/(g r)                          (Lagrange 引力势扰动)
  y₄ = d(δΦ)/dr / (g r)                  (引力势梯度)

振荡方程组 (一阶形式, Silvester et al. 2023):

  dy₁/dr = [1/c_s² (Φ' - gξ_r) + l(l+1)y₃ - y₂] / r

  dy₂/dr = [(ω²r/g - 4 + A r) y₁ - (A r - y₂·c_s²ρ r/(g))
           + (l(l+1)y₂ - y₄) c_s² ρ r / g] / r

  dy₃/dr = (y₄ - l(l+1) y₃) / r

  dy₄/dr = [4πGρ/c_s² y₂ - y₁ ω²r/g + (l(l+1) - l(l+1)) y₃] / r

其中:
  A = d(lnρ)/dr - (1/Γ₁) d(lnP)/dr  (Schwarzschild 判别式)
  c_s² = Γ₁ P/ρ                      (绝热声速²)

边界条件:
  中心 (r → 0):
    y₁ = 有限值, y₃ = 有限值
    y₂ 满足正则性条件

  表面 (r → R):
    y₂ = 0  (零 Lagrange 压强扰动, 简化边界)
    y₄ = -(l+1) y₃  (外场匹配)

本模块实现:
  1. 构造振荡方程的系数矩阵
  2. Lagrange 基函数的高阶插值
  3. 振荡方程的变分原理实现
  4. 解析基准解 (均匀密度球, 即 n=0 多方球)
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# 物理常数 (复用 stellar_structure 中的定义)
# ============================================================
G_CGS = 6.67430e-8
PI = np.pi


class LagrangeBasisHighOrder:
    """
    高阶 Lagrange 插值基函数及其导数.

    融合 Lagrange (632) 项目:
      L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

    对于恒星振荡问题, 我们使用 Chebyshev 节点避免 Runge 现象:
      x_j = (a+b)/2 + (b-a)/2 cos((2j+1)π/(2N)),  j=0,...,N-1

    导数公式 (from 632):
      L_i'(x) = Σ_{k≠i} [1/(x_i - x_k)] Π_{j≠i,k} (x - x_j)/(x_i - x_j)

    二阶导数:
      L_i''(x) = 2 Σ_{k≠i} Σ_{m≠i,k<m}
                  [1/((x_i-x_k)(x_i-x_m))] Π_{j≠i,k,m} (x-x_j)/(x_i-x_j)

    参数
    ----
    order : int
        插值阶数 (默认 6, 对应 6 节点)
    a, b : float
        插值区间端点
    """

    def __init__(self, order: int = 6, a: float = 0.0, b: float = 1.0):
        if order < 2:
            raise ValueError(f"插值阶数 order={order} 至少为 2")
        self.order = order
        self.a = a
        self.b = b

        # Chebyshev 节点 (避免 Runge 现象)
        # x_j = (a+b)/2 + (b-a)/2 cos((2j+1)π/(2N))
        j = np.arange(order)
        self.nodes = (a + b) / 2.0 + (b - a) / 2.0 * np.cos(
            (2.0 * j + 1.0) * PI / (2.0 * order)
        )
        self.nodes = np.sort(self.nodes)  # 按升序排列

        # 预计算节点间距矩阵
        self._diff_matrix = np.zeros((order, order))
        for i in range(order):
            for j_idx in range(order):
                if i != j_idx:
                    self._diff_matrix[i, j_idx] = self.nodes[i] - self.nodes[j_idx]
                else:
                    self._diff_matrix[i, j_idx] = 1.0  # 占位

    def basis_value(self, i: int, x: float) -> float:
        """
        计算第 i 个 Lagrange 基函数在 x 处的值.

        L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

        Parameters
        ----------
        i : int
            基函数索引 (0 ≤ i < order)
        x : float
            计算点

        Returns
        -------
        val : float
            L_i(x) 的值
        """
        if i < 0 or i >= self.order:
            raise ValueError(f"基函数索引 i={i} 超出范围 [0, {self.order})")

        val = 1.0
        for j in range(self.order):
            if j != i:
                denom = self.nodes[i] - self.nodes[j]
                if abs(denom) < 1e-15:
                    raise ValueError(f"节点 {i} 和 {j} 过于接近")
                val *= (x - self.nodes[j]) / denom
        return val

    def basis_value_vectorized(self, x_arr: np.ndarray) -> np.ndarray:
        """
        向量化计算所有基函数在一组点上的值.

        Parameters
        ----------
        x_arr : ndarray, shape (M,)
            计算点

        Returns
        -------
        L : ndarray, shape (order, M)
            L[i, :] = L_i(x_arr)
        """
        M = len(x_arr)
        L = np.ones((self.order, M))
        for i in range(self.order):
            for j in range(self.order):
                if j != i:
                    denom = self.nodes[i] - self.nodes[j]
                    L[i] *= (x_arr - self.nodes[j]) / denom
        return L

    def basis_deriv(self, i: int, x: float) -> float:
        """
        计算第 i 个 Lagrange 基函数的导数.

        L_i'(x) = Σ_{k≠i} [1/(x_i - x_k)] Π_{j≠i,k} (x - x_j)/(x_i - x_j)

        Parameters
        ----------
        i : int
            基函数索引
        x : float
            计算点

        Returns
        -------
        dval : float
            L_i'(x) 的值
        """
        dval = 0.0
        for k in range(self.order):
            if k == i:
                continue
            # 计算 Π_{j≠i,k} (x - x_j)/(x_i - x_j)
            p = 1.0
            for j in range(self.order):
                if j == i or j == k:
                    continue
                p *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
            dval += p / (self.nodes[i] - self.nodes[k])
        return dval

    def basis_deriv2(self, i: int, x: float) -> float:
        """
        计算第 i 个 Lagrange 基函数的二阶导数.

        L_i''(x) = 2 Σ_{k<m, k≠i, m≠i}
            [1/((x_i-x_k)(x_i-x_m))] Π_{j≠i,k,m} (x-x_j)/(x_i-x_j)

        Parameters
        ----------
        i : int
            基函数索引
        x : float
            计算点

        Returns
        -------
        d2val : float
            L_i''(x) 的值
        """
        d2val = 0.0
        for k in range(self.order):
            if k == i:
                continue
            for m in range(k + 1, self.order):
                if m == i:
                    continue
                # 计算 Π_{j≠i,k,m} (x - x_j)/(x_i - x_j)
                p = 1.0
                for j in range(self.order):
                    if j == i or j == k or j == m:
                        continue
                    p *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
                d2val += 2.0 * p / ((self.nodes[i] - self.nodes[k])
                                     * (self.nodes[i] - self.nodes[m]))
        return d2val

    def differentiation_matrix(self) -> np.ndarray:
        """
        构造微分矩阵 D, 使得 f'(x_i) ≈ Σ_j D_{ij} f(x_j).

        使用 Lagrange 插值导数公式:
          D_{ij} = L_j'(x_i)  (i ≠ j)
          D_{ii} = -Σ_{j≠i} D_{ij}  (保证常函数导数为零)

        Returns
        -------
        D : ndarray, shape (order, order)
            微分矩阵
        """
        D = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                if j != i:
                    D[i, j] = self.basis_deriv(j, self.nodes[i])
            D[i, i] = -np.sum(D[i, :])  # 对角线: 保证一致性
        return D


class OscillationEquationCoefficients:
    """
    计算线性绝热振荡方程的系数.

    振荡方程的标准形式 (一阶 ODE 系统):

    dY/dr = M(r, ω) Y

    其中 Y = [y₁, y₂, y₃, y₄]ᵀ, 系数矩阵 M 的元素为:

    M₁₁ = -A + 2/r - U
    M₁₂ = 1/r
    M₁₃ = -l(l+1)/r
    M₁₄ = 0

    M₂₁ = -(ω²r³/g - 4 - A r + U r) / r  [含 ω² 的本征值项]
    M₂₂ = (U - A r) / r
    M₂₃ = l(l+1) / r
    M₂₄ = -1/r

    M₃₁ = 0
    M₃₂ = 0
    M₃₃ = -l(l+1)/r
    M₃₄ = 1/r

    M₄₁ = (ω²r²/g - U r) × V / r
    M₄₂ = -V/r
    M₄₃ = [l(l+1) - 2] / r  [简化]
    M₄₄ = 0

    其中:
      U = d(lnρ)/dr × r      (对数密度梯度)
      V = d(lnP)/dr × r      (对数压强梯度)
      A = U/V - 1/Γ₁         (Schwarzschild 判别式 / H_P 归一化)
      g = Gm/r²              (局部重力)

    参数
    ----
    stellar_model : StellarStructureModel
        恒星结构模型
    """

    def __init__(self, stellar_model):
        self.model = stellar_model
        self.n_r = stellar_model.n_r
        self.r = stellar_model.r
        self.dr = stellar_model.dr

        # 预计算结构量
        self.c_s2 = stellar_model.get_sound_speed_profile()**2  # 声速²
        self.c_s2 = np.maximum(self.c_s2, 1e-10)  # 数值保护

        # 对数梯度 U 和 V
        self.U = np.zeros(self.n_r)
        self.V = np.zeros(self.n_r)
        self.A_sch = np.zeros(self.n_r)

        self._compute_log_gradients()

    def _compute_log_gradients(self):
        """
        计算 U = dlnρ/dlnr 和 V = dlnP/dlnr.

        使用中心差分 (内部点) + 单侧差分 (边界点).
        """
        r = self.r
        rho = self.model.rho
        P = self.model.P

        for i in range(1, self.n_r - 1):
            dr = r[i + 1] - r[i - 1]
            if r[i] > 1e-10 and rho[i] > 1e-30 and P[i] > 1e-30:
                self.U[i] = r[i] * (rho[i + 1] - rho[i - 1]) / (dr * rho[i] + 1e-100)
                self.V[i] = r[i] * (P[i + 1] - P[i - 1]) / (dr * P[i] + 1e-100)
            else:
                self.U[i] = 0.0
                self.V[i] = 0.0

        # 边界
        if self.n_r > 2:
            self.U[0] = self.U[1]
            self.U[-1] = self.U[-2]
            self.V[0] = self.V[1]
            self.V[-1] = self.V[-2]

        # Schwarzschild 判别式 A
        # A = U/V - 1/Γ₁
        for i in range(self.n_r):
            if abs(self.V[i]) > 1e-15:
                self.A_sch[i] = self.U[i] / self.V[i] - 1.0 / max(
                    self.model.Gamma1[i], 1e-10
                )
            else:
                self.A_sch[i] = 0.0

    def build_oscillation_matrix(
        self, omega: complex, l: int
    ) -> np.ndarray:
        """
        构造振荡方程的系数矩阵 M(r, ω, l).

        Parameters
        ----------
        omega : complex
            角频率 [rad/s]
        l : int
            角量子数

        Returns
        -------
        M : ndarray, shape (n_r, 4, 4)
            每个径向网格点上的 4×4 系数矩阵
        """
        if l < 0:
            raise ValueError(f"角量子数 l={l} 不能为负")

        n = self.n_r
        r = self.r
        g = self.model.g
        Gamma1 = self.model.Gamma1

        M = np.zeros((n, 4, 4), dtype=complex)
        omega2 = omega**2

        for i in range(n):
            r_i = max(r[i], 1e-10)
            g_i = max(g[i], 1e-10)
            c2_i = self.c_s2[i]
            A_i = self.A_sch[i]
            U_i = self.U[i]
            V_i = self.V[i]
            l_val = l * (l + 1.0)

            # M 矩阵元素
            # 第一行
            M[i, 0, 0] = -A_i / r_i + 2.0 / r_i - U_i / r_i
            M[i, 0, 1] = c2_i / (g_i * r_i)
            M[i, 0, 2] = -l_val / r_i
            M[i, 0, 3] = 0.0

            # 第二行 (含 ω²)
            M[i, 1, 0] = -(omega2 * r_i**3 / g_i - 4.0 - A_i * r_i
                            + V_i) / r_i**2 * g_i
            M[i, 1, 1] = (V_i / r_i - A_i * g_i / c2_i)
            M[i, 1, 2] = l_val / r_i
            M[i, 1, 3] = -g_i / (c2_i * r_i)

            # 第三行
            M[i, 2, 0] = 0.0
            M[i, 2, 1] = 0.0
            M[i, 2, 2] = -l_val / r_i
            M[i, 2, 3] = 1.0 / r_i

            # 第四行 (含 4πGρ 项)
            M[i, 3, 0] = (omega2 * r_i / g_i - V_i / r_i) * 4.0 * PI * G_CGS * self.model.rho[i] / c2_i
            M[i, 3, 1] = -4.0 * PI * G_CGS * self.model.rho[i] / (c2_i * r_i)
            M[i, 3, 2] = (l_val - 2.0) / r_i  # 简化
            M[i, 3, 3] = 0.0

        return M

    def compute_richardson_number(self) -> np.ndarray:
        """
        计算 Richardson 数剖面 (剪切流稳定性).

        Ri = N² / (dU/dz)²

        在恒星中, 这对应旋转剪切导致的动力学不稳定:
          Ri < 1/4  →  Kelvin-Helmholtz 不稳定

        Returns
        -------
        Ri : ndarray
            Richardson 数剖面
        """
        N2 = self.model.N2
        # 估计旋转剪切 (假设差速旋转)
        # dΩ/dr ≈ Ω_0 × (r/R)^{-2}  (简化)
        Omega0 = 2.0 * PI / (25.0 * 86400.0)  # 太阳表面 ~25 天周期
        r = self.r
        R = self.model.R_star

        # 旋转剖面 (太阳型差速旋转)
        Omega = Omega0 * (1.0 - 0.07 * (1.0 - (r / R)**2))
        dOmega_dr = np.zeros_like(r)
        for i in range(1, self.n_r - 1):
            dOmega_dr[i] = (Omega[i + 1] - Omega[i - 1]) / (2.0 * self.dr)

        # 剪切率
        shear = r * np.abs(dOmega_dr) + 1e-30
        Ri = N2 / (shear**2 + 1e-30)
        return Ri


class UniformSphereAnalytic:
    """
    均匀密度球 (n=0 多方球) 的解析振荡频率.

    用于数值方法的基准测试.

    对于均匀密度球 ρ = ρ₀, 径向振荡 (l=0) 的频率满足:

    ω² = (4πGρ₀/3) × λ_n

    其中 λ_n 是以下超越方程的根:

    j_l(√λ ξ) = 0   (l=0 时: sin(√λ ξ)/(√λ ξ) = 0)
    → √λ_n ξ₁ = nπ
    → ω_n = nπ/ξ₁ × √(4πGρ₀/3)

    对于非径向 (l > 0), Lamb 频率:
      ω_{n,l}² = l(l+1) × 4πGρ₀/3   (f 模, 无径向节点)

    参数
    ----
    rho0 : float
        均匀密度 [g/cm³]
    R : float
        球半径 [cm]
    """

    def __init__(self, rho0: float = 1.41, R: float = 6.957e10):
        self.rho0 = rho0
        self.R = R
        self.omega0 = np.sqrt(4.0 * PI * G_CGS * rho0 / 3.0)

    def radial_frequency(self, n: int) -> float:
        """
        计算第 n 个径向模 (l=0) 的频率.

        ω_n = nπ × ω₀ / ξ₁

        对于均匀球, ξ₁ = π (即 sin(ωr/c_s)/r 的第一个零点).

        Parameters
        ----------
        n : int
            径向导数数量 (n ≥ 1)

        Returns
        -------
        omega : float
            角频率 [rad/s]
        """
        if n <= 0:
            raise ValueError(f"径向阶数 n={n} 必须 ≥ 1")
        return n * PI * self.omega0 / PI  # = n * ω₀

    def f_mode_frequency(self, l: int) -> float:
        """
        计算 f 模 (基础模, 无径向节点) 的频率.

        ω_f² = l(l+1) × g/R = l(l+1) × 4πGρ₀R/(3R) = l(l+1) ω₀²

        Parameters
        ----------
        l : int
            角量子数 (l ≥ 1)

        Returns
        -------
        omega : float
            角频率 [rad/s]
        """
        if l < 1:
            raise ValueError(f"f 模要求 l ≥ 1, 收到 l={l}")
        return self.omega0 * np.sqrt(l * (l + 1.0))

    def p_mode_frequency(self, n: int, l: int) -> float:
        """
        计算 p 模频率 (渐近近似).

        ω_{n,l}² ≈ [(n + l/2 + ε) × π × ω₀]²

        其中 ε 为表面修正项 (对均匀球 ε ≈ 0).

        Parameters
        ----------
        n : int
            径向导数数量
        l : int
            角量子数

        Returns
        -------
        omega : float
            角频率 [rad/s]
        """
        n_eff = n + l / 2.0
        return n_eff * PI * self.omega0

    def get_all_frequencies(
        self, n_max: int = 10, l_max: int = 3
    ) -> np.ndarray:
        """
        获取所有频率 (排序).

        Parameters
        ----------
        n_max : int
            最大径向阶数
        l_max : int
            最大角量子数

        Returns
        -------
        freqs : ndarray
            排序后的频率 [rad/s]
        """
        freqs = []
        for l in range(l_max + 1):
            for n in range(max(1, 1 if l == 0 else 0), n_max + 1):
                if l == 0:
                    freqs.append(self.radial_frequency(n))
                elif n == 0:
                    freqs.append(self.f_mode_frequency(l))
                else:
                    freqs.append(self.p_mode_frequency(n, l))
        return np.sort(np.array(freqs))
