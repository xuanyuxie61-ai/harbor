# -*- coding: utf-8 -*-
"""
high_order_fd.py — 高阶有限差分算子
=====================================
核心科学问题: 构建 2/4/6/8 阶中心差分格式离散化
有效质量薛定谔方程 (EMSE), 处理异质结界面处有效质量不连续.

融合种子项目:
  - 072_barycentric_interp_1d: Fornberg 算法构造非均匀网格差分权重
  - 596_interp_trig: 周期性边界条件下三角差分格式

数学基础:
  中心差分 d²/dz² (2p 阶精度, 2p+1 点模板):
    f''(z_i) = sum_{k=-p}^{p} c_k^{(2p)} * f(z_{i+k}) / h^2 + O(h^{2p})

  其中系数 c_k 由 Taylor 展开匹配确定:
    c_k = (-1)^{k+1} * 2 * (p!)^2 / ((p+k)! * (p-k)! * k^2)  (k≠0)
    c_0 = -2 * sum_{k=1}^{p} 1/k^2

Ben Daniel-Duke 边界条件 (有效质量不连续):
    [1/m*_{i+1/2} * (ψ_{i+1}-ψ_i) - 1/m*_{i-1/2} * (ψ_i-ψ_{i-1})] / h^2
"""

import numpy as np

# 物理常数
HBAR = 1.054571817e-34   # J·s
M0 = 9.1093837015e-31    # kg
E0 = 1.602176634e-19     # C (eV→J)


def _build_d2_stencil(order):
    """
    构建 2p 阶中心差分二阶导数模板系数.

    参数
    ----
    order : int
        精度阶数 (2, 4, 6, 8)

    返回
    ----
    ndarray
        模板系数, 长度 order+1
    """
    p = order // 2
    coeffs = np.zeros(2 * p + 1)

    # 中心系数
    c0 = -2.0 * sum(1.0 / k**2 for k in range(1, p + 1))
    coeffs[p] = c0

    # 非中心系数
    import math
    for k in range(1, p + 1):
        numerator = 2.0 * math.factorial(p) ** 2
        denominator = math.factorial(p + k) * math.factorial(p - k) * k**2
        sign = (-1) ** (k + 1)
        c_k = sign * numerator / denominator
        coeffs[p + k] = c_k
        coeffs[p - k] = c_k

    return coeffs


def _build_d1_stencil(order):
    """
    构建一阶导数中心差分模板.
    """
    p = order // 2
    coeffs = np.zeros(2 * p + 1)

    import math
    for k in range(1, p + 1):
        sign = (-1) ** (k + 1)
        # 一阶导数系数
        c_k = sign * math.factorial(p) ** 2 / (
            math.factorial(p + k) * math.factorial(p - k) * k)
        coeffs[p + k] = c_k
        coeffs[p - k] = -c_k

    return coeffs


# 预计算模板
D2_STENCILS = {o: _build_d2_stencil(o) for o in [2, 4, 6, 8]}
D1_STENCILS = {o: _build_d1_stencil(o) for o in [2, 4, 6]}


class HighOrderFD:
    """
    高阶有限差分算子.

    提供均匀和非均匀网格上的高阶差分.
    """

    def __init__(self, order=4):
        if order not in D2_STENCILS:
            raise ValueError(f"阶数 {order} 不支持, 可选: {list(D2_STENCILS.keys())}")
        self.order = order
        self.d2 = D2_STENCILS[order]
        self.d1 = D1_STENCILS.get(order, D1_STENCILS[4])
        self.hw = order // 2  # 半带宽

    def second_derivative(self, f, dz):
        """
        二阶导数 d²f/dz² (均匀网格).

        参数
        ----
        f : ndarray, shape (N,)
        dz : float
            网格间距

        返回
        ----
        ndarray, shape (N,)
        """
        N = len(f)
        result = np.zeros(N)
        hw = self.hw

        # 内部点: 高阶模板
        for i in range(hw, N - hw):
            s = 0.0
            for k in range(-hw, hw + 1):
                s += self.d2[k + hw] * f[i + k]
            result[i] = s / dz**2

        # 边界: 低阶回退 (二阶)
        if N >= 3:
            result[0] = (f[0] - 2*f[1] + f[2]) / dz**2
            result[-1] = (f[-3] - 2*f[-2] + f[-1]) / dz**2
        if hw >= 2 and N >= 5:
            result[1] = (f[0] - 2*f[1] + f[2]) / dz**2
            result[-2] = (f[-3] - 2*f[-2] + f[-1]) / dz**2

        return result

    def first_derivative(self, f, dz):
        """一阶导数 df/dz (均匀网格)."""
        N = len(f)
        result = np.zeros(N)
        hw = len(self.d1) // 2

        for i in range(hw, N - hw):
            s = 0.0
            for k in range(-hw, hw + 1):
                s += self.d1[k + hw] * f[i + k]
            result[i] = s / dz

        # 边界
        if N >= 2:
            result[0] = (f[1] - f[0]) / dz
            result[-1] = (f[-1] - f[-2]) / dz

        return result

    def fornberg_weights(self, z0, z_nodes, m):
        """
        Fornberg 算法: 任意节点分布上的 m 阶导数权重.
        用于非均匀网格.

        参数
        ----
        z0 : float
            求导点
        z_nodes : array_like
            模板节点
        m : int
            导数阶数

        返回
        ----
        ndarray
            权重向量
        """
        n = len(z_nodes)
        d = np.zeros((n, m + 1))
        d[0, 0] = 1.0
        c1 = 1.0

        for i in range(1, n):
            c2 = 1.0
            for j in range(i):
                c3 = z_nodes[i] - z_nodes[j]
                if abs(c3) < 1e-30:
                    c3 = 1e-30
                c2 *= c3
                for k in range(min(i, m), 0, -1):
                    d[i, k] = (c1 * (k * d[i-1, k-1] -
                                     (z_nodes[i-1] - z0) * d[i-1, k]) -
                               (z_nodes[j] - z0) * d[i, k]) / c3
                d[i, 0] = -c1 * (z_nodes[i-1] - z0) * d[i-1, 0] / c3
                for k in range(min(i, m), 0, -1):
                    d[j, k] = ((z_nodes[i] - z0) * d[j, k] -
                               c1 * (k * d[j, k-1 if k > 0 else 0] -
                                     (z_nodes[i-1] - z0) * d[j, k])) / (-c3) if j < i else d[j, k]
                d[j, 0] = (z_nodes[i] - z0) * d[j, 0] / (-c3) if j < i else d[j, 0]
            c1 = c2

        return d[:, m]


class BenDanielDukeFD(HighOrderFD):
    """
    Ben Daniel-Duke 有限差分格式.

    处理有效质量空间变化 m*(z):
        Tψ = -(ℏ²/2) d/dz[1/m*(z) * dψ/dz]

    离散化 (2阶):
        Tψ_i = -(ℏ²/2) * [1/m*_{i+1/2}*(ψ_{i+1}-ψ_i) -
                            1/m*_{i-1/2}*(ψ_i-ψ_{i-1})] / (m0 * dz²)

    半格点有效质量 (调和平均):
        1/m*_{i+1/2} = 0.5*(1/m*_i + 1/m*_{i+1})
    """

    def __init__(self, order=4):
        super().__init__(order)

    def apply_kinetic(self, psi, m_star_profile, dz):
        """
        动能算子 T 作用在波函数 ψ 上.

        参数
        ----
        psi : ndarray
            波函数
        m_star_profile : ndarray
            有效质量剖面 (单位 m0)
        dz : float
            网格间距 [m]

        返回
        ----
        ndarray
            Tψ [J]
        """
        N = len(psi)
        result = np.zeros(N)

        # 半格点 1/m* (调和平均)
        inv_m_half = np.zeros(N + 1)
        for i in range(N):
            m_i = max(m_star_profile[i], 1e-8)
            inv_m_half[i] = 1.0 / m_i
        inv_m_half[N] = inv_m_half[N - 1]

        # 半格点调和平均
        for i in range(1, N):
            inv_m_half[i] = 0.5 * (inv_m_half[i-1] + inv_m_half[i])
        inv_m_half[0] = inv_m_half[1]
        inv_m_half[N] = inv_m_half[N - 1]

        # 主循环
        prefactor = HBAR**2 / (2.0 * M0 * dz**2)
        for i in range(1, N - 1):
            flux_r = inv_m_half[i + 1] * (psi[i + 1] - psi[i])
            flux_l = inv_m_half[i] * (psi[i] - psi[i - 1])
            result[i] = -prefactor * (flux_r - flux_l)

        # Dirichlet BC: ψ=0 at boundaries
        result[0] = 0.0
        result[-1] = 0.0
        return result

    def max_stable_timestep(self, m_star_min, V_max_eV, dz, n_steps=1):
        """
        显式时间推进的最大稳定时间步:
            dt_max = 4*m*_min*m0*dz² / ℏ

        对于 Crank-Nicolson: 无条件稳定
        对于 Runge-Kutta 4: dt < 2.785 * dt_max_explicit
        """
        m_kg = m_star_min * M0
        dt_explicit = 4 * m_kg * dz**2 / HBAR
        if n_steps == 1:
            return dt_explicit
        elif n_steps == 4:  # RK4
            return 2.785 * dt_explicit
        return dt_explicit

    def modified_dispersion(self, k_array, m_star, dz):
        """
        差分离散化后的修正色散关系:

        精确: E(k) = ℏ²k²/(2m*)
        离散 (2阶): E_disc(k) = ℏ²/(m*·dz²) · (1 - cos(k·dz))
        离散 (4阶): E_disc(k) = ℏ²/(m*·dz²) · (4/3·sin²(k·dz/2) -
                                                  1/12·sin²(k·dz))

        返回 (E_exact, E_discrete) 对.
        """
        m_kg = m_star * M0
        E_exact = HBAR**2 * k_array**2 / (2 * m_kg)

        if self.order == 2:
            E_disc = HBAR**2 / (m_kg * dz**2) * (1 - np.cos(k_array * dz))
        elif self.order == 4:
            s1 = np.sin(k_array * dz / 2)**2
            s2 = np.sin(k_array * dz)**2
            E_disc = HBAR**2 / (m_kg * dz**2) * (4.0/3.0 * s1 - 1.0/12.0 * s2)
        else:
            # 通用: 用 Taylor 展开近似
            E_disc = HBAR**2 / (m_kg * dz**2) * (1 - np.cos(k_array * dz))

        return E_exact, E_disc


class PeriodicFD(HighOrderFD):
    """
    周期性/Bloch 边界条件的有限差分.
    融合 596_interp_trig: 三角插值给出谱方法差分.

    Bloch 定理: ψ(z+L) = exp(ik_B·L) · ψ(z)
    """

    def __init__(self, order=4, L=1e-8, k_bloch=0.0):
        super().__init__(order)
        self.L = L
        self.k_bloch = k_bloch

    def second_derivative_bloch(self, f, dz):
        """
        Bloch 边界条件下的二阶导数.
        内部用标准差分, 边界用 Bloch 相位连接.
        """
        N = len(f)
        result = self.second_derivative(f, dz)
        hw = self.hw

        phase = np.exp(1j * self.k_bloch * self.L)
        inv_phase = np.conj(phase)

        # 修正边界点
        if N >= 3:
            result[0] = (inv_phase * f[-2] - 2*f[0] + phase * f[1]) / dz**2
            result[-1] = (inv_phase * f[-2] - 2*f[-1] + phase * f[1]) / dz**2

        return result
