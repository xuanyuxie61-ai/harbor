# -*- coding: utf-8 -*-
"""
stability_analysis.py — 数值稳定性分析
=========================================
核心科学问题: 对有限差分离散化进行 von Neumann 稳定性分析,
确定最大稳定时间步长, 并分析空间离散的色散误差.

融合种子项目:
  - 综合各项目的数值分析思想
"""

import numpy as np
from high_order_fd import HBAR, M0, E0, D2_STENCILS


class VonNeumannStability:
    """
    Von Neumann 稳定性分析.

    分析差分格式的色散关系和稳定性条件.
    """

    def __init__(self, order=4):
        self.order = order
        self.stencil = D2_STENCILS[order]

    def amplification_factor(self, k_dz, dt, m_star, V_max=0.0):
        """
        计算放大因子 G(k·dz).

        对于薛定谔方程 iℏ ∂ψ/∂t = Hψ,
        使用 Crank-Nicolson:
            G = (1 - i·dt·E_disc/(2ℏ)) / (1 + i·dt·E_disc/(2ℏ))
        |G| = 1 对所有 k (无条件稳定)

        对于显式 Euler:
            G = 1 - i·dt·E_disc/ℏ
        |G|² = 1 + (dt·E_disc/ℏ)² > 1 (不稳定!)

        对于 Leapfrog:
            G = -i·dt·E_disc/ℏ ± sqrt(1 - (dt·E_disc/ℏ)²)
        |G| ≤ 1 iff dt·E_disc/ℏ ≤ 1

        参数
        ----
        k_dz : ndarray
            无量纲波数 k·dz
        dt : float
            时间步长 [s]
        m_star : float
            有效质量 [m0]
        V_max : float
            最大势能 [eV]
        """
        m_kg = m_star * M0
        k_dz = np.atleast_1d(k_dz)

        # 离散色散 (用模板计算)
        E_disc = self._discrete_dispersion(k_dz, m_star)

        # 放大因子 (Crank-Nicolson)
        omega_dt = E_disc * dt / HBAR
        G_real = (1 - 0) / (1 + 0)  # CN: |G|=1

        # 对于 Leapfrog
        G_leapfrog = np.zeros(len(k_dz), dtype=complex)
        stable = np.abs(omega_dt) <= 1.0
        G_leapfrog[stable] = -1j * omega_dt[stable] + np.sqrt(
            np.maximum(1 - omega_dt[stable]**2, 0))

        # 放大因子模
        G_mag_cn = np.ones_like(k_dz)
        G_mag_lf = np.abs(G_leapfrog)

        return G_mag_cn, G_mag_lf, stable

    def _discrete_dispersion(self, k_dz, m_star):
        """
        离散色散关系 E_disc(k).

        对于模板 {c_k}, 二阶导数:
            D2(k) = sum_k c_k * exp(i·k·dz·k) / dz²

        E_disc = ℏ²/(2m*) * (-D2(k))
        """
        m_kg = m_star * M0
        hw = len(self.stencil) // 2

        D2 = np.zeros_like(k_dz, dtype=complex)
        for j, c in enumerate(self.stencil):
            k_idx = j - hw
            D2 += c * np.exp(1j * k_dz * k_idx)

        # E = -ℏ²/(2m*) * D2(k) / dz²
        # 注意: D2 已经包含了 1/dz²
        E_disc = -HBAR**2 / (2 * m_kg) * D2 / 1.0  # dz² 在外部处理

        return np.real(E_disc)

    def stability_limit_explicit(self, m_star, dz):
        """
        显式格式的最大稳定时间步:
            对于 iℏ dψ/dt = Hψ 的显式 Euler:
            dt_max = 2ℏ / E_max ≈ 4*m*·m0*dz² / ℏ²
        """
        m_kg = m_star * M0
        return 4 * m_kg * dz**2 / HBAR

    def stability_limit_leapfrog(self, m_star, dz, V_max_eV):
        """
        Leapfrog 格式的稳定时间步:
            dt_max = ℏ / max(E_disc)
        其中 max(E_disc) ≈ ℏ²/(m*·dz²) + V_max
        """
        m_kg = m_star * M0
        E_max = HBAR**2 / (m_kg * dz**2) + V_max_eV * E0
        return HBAR / E_max if E_max > 0 else float('inf')

    def cfl_condition(self, v_max, dz):
        """
        CFL 条件 (对流项):
            dt < dz / v_max
        """
        if v_max <= 0:
            return float('inf')
        return dz / v_max


class DispersionAnalysis:
    """
    色散误差分析.

    比较精确色散和离散色散, 量化数值色散.
    """

    def __init__(self, order=4):
        self.order = order
        self.stencil = D2_STENCILS[order]

    def numerical_dispersion(self, k_array, m_star, dz):
        """
        计算数值色散 E_num(k) 和精确色散 E_exact(k).
        """
        m_kg = m_star * M0
        E_exact = HBAR**2 * k_array**2 / (2 * m_kg)

        # 数值色散
        hw = len(self.stencil) // 2
        E_num = np.zeros_like(k_array)

        for idx, k in enumerate(k_array):
            D2_val = 0.0
            for j, c in enumerate(self.stencil):
                n = j - hw
                D2_val += c * np.cos(k * dz * n)
            D2_val /= dz**2
            E_num[idx] = -HBAR**2 / (2 * m_kg) * D2_val

        return E_exact, E_num

    def dispersion_error(self, k_array, m_star, dz):
        """
        色散误差 = |E_num - E_exact| / E_exact
        """
        E_exact, E_num = self.numerical_dispersion(k_array, m_star, dz)
        error = np.abs(E_num - E_exact) / np.maximum(np.abs(E_exact), 1e-30)
        return error

    def nyquist_limit(self, dz):
        """
        Nyquist 极限波数:
            k_max = π / dz
        """
        return np.pi / dz

    def resolution_requirement(self, wavelength_min, points_per_wavelength=10):
        """
        最小波长所需的网格间距:
            dz ≤ λ_min / N_ppw
        """
        return wavelength_min / points_per_wavelength


class ConvergenceStudy:
    """
    网格收敛性研究.
    """

    def __init__(self, solver_func, z_range, exact_energy=None):
        """
        参数
        ----
        solver_func : callable
            求解器函数: solver_func(N) → (energies, error)
        z_range : tuple
            计算域 (z_min, z_max)
        exact_energy : float or None
            精确能量 (用于误差计算)
        """
        self.solver = solver_func
        self.z_range = z_range
        self.E_exact = exact_energy

    def run_convergence(self, N_list):
        """
        运行网格收敛测试.

        返回 (N, error, rate) 列表.
        """
        results = []
        prev_error = None
        prev_N = None

        for N in N_list:
            energies, error = self.solver(N)

            if self.E_exact is not None and len(energies) > 0:
                error = abs(energies[0] - self.E_exact)
            elif error is None:
                error = 0.0

            # 收敛速率
            rate = None
            if prev_error is not None and prev_error > 0 and error > 0:
                rate = np.log(prev_error / error) / np.log(prev_N / N)

            results.append({
                'N': N,
                'error': error,
                'rate': rate,
                'energies': energies,
            })

            prev_error = error
            prev_N = N

        return results

    def richardson_extrapolation(self, results):
        """
        Richardson 外推:
            E_exact ≈ (r^p * E_fine - E_coarse) / (r^p - 1)

        其中 r 是网格比, p 是收敛阶数.
        """
        if len(results) < 3:
            return None

        e1 = results[-3]['energies']
        e2 = results[-2]['energies']
        e3 = results[-1]['energies']
        E1 = e1[0] if len(e1) > 0 else 0
        E2 = e2[0] if len(e2) > 0 else 0
        E3 = e3[0] if len(e3) > 0 else 0

        r1 = results[-2]['N'] / results[-3]['N']
        r2 = results[-1]['N'] / results[-2]['N']

        # 估计阶数
        if E1 != E2 and E2 != E3:
            p_est = np.log(abs((E1 - E2) / (E2 - E3))) / np.log(r1)
        else:
            p_est = 2.0

        # 外推
        r = results[-1]['N'] / results[-2]['N']
        E_extrap = (r**p_est * E3 - E2) / (r**p_est - 1) if abs(r**p_est - 1) > 1e-10 else E3

        return E_extrap, p_est
