"""
stability_analysis.py
===================================================================
高阶有限差分格式稳定性分析模块

映射种子项目:
  - 1374_unstable_ode: 不稳定ODE系统 → 径向方程数值不稳定性的精确诊断
  - 353_fd1d_advection_ftcs: FTCS 不稳定格式 → von Neumann 稳定性分析基础

核心分析理论:
  von Neumann 稳定性分析 (傅里叶模态分析):
    设 u^n_j = G^n * exp(i*k*j*h), 则增长因子 G 满足:
    |G(k)| <= 1 + O(dt) 对所有 k 成立 => 格式稳定

  矩阵稳定性分析:
    差分格式 u^{n+1} = A * u^n, 稳定条件:
    rho(A) = max|eigenvalue(A)| <= 1 + O(dt)

  Courant-Friedrichs-Lewy (CFL) 条件:
    dt <= C_max * h^2 / (hbar/(2m))  对扩散型方程
    dt <= C_max * h / v_max          对对流型方程

  条件数分析:
    kappa(A) = ||A|| * ||A^{-1}||
    条件数过大 => 数值不稳定

  Lax-Richtmyer 等价定理:
    一致性 + 稳定性 <=> 收敛性
===================================================================
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


class StabilityAnalyzer:
    """
    有限差分格式稳定性综合分析器

    分析方法:
    1. von Neumann 傅里叶模态分析
    2. 矩阵谱半径分析
    3. 条件数增长追踪
    4. 能量守恒检验
    """

    def __init__(
        self,
        n_spatial: int = 200,
        dr: float = 0.15,
        mass_reduced: float = 469.0,
    ):
        self.N = n_spatial
        self.dr = dr
        self.m_red = mass_reduced

    def von_neumann_ftcs(
        self,
        V_array: np.ndarray,
        dt: float,
        l_quantum: int = 0
    ) -> Dict:
        """
        FTCS 格式的 von Neumann 稳定性分析

        对含时薛定谔方程 i*hbar*dpsi/dt = H*psi,
        FTCS 格式的增长因子为:
            G = 1 - i*dt/hbar * [hbar^2/(m*h^2)*(1-cos(kh)) + V]

        模长:
            |G|^2 = 1 + (dt/hbar)^2 * [hbar^2/(m*h^2)*(1-cos(kh)) + V]^2

        由于 |G|^2 >= 1 对所有 kh 成立, FTCS 无条件不稳定.

        参数:
            V_array: 势能数组
            dt: 时间步长
            l_quantum: 角动量量子数

        返回:
            稳定性诊断字典
        """
        h = self.dr
        h2 = h * h

        # 波数网格 (傅里叶空间)
        k_fourier = np.linspace(0, np.pi / h, self.N // 2 + 1)

        # 平均势
        V_mean = np.mean(np.real(V_array))

        # 离心势贡献
        r_avg = self.N * h / 2.0
        cent = l_quantum * (l_quantum + 1) / r_avg ** 2 if r_avg > 0 else 0.0

        # 扩散系数
        alpha = HBAR_C / (2.0 * self.m_red * h2)

        # 增长因子 |G|^2 对每个傅里叶模态
        G_squared = np.zeros(len(k_fourier))
        for idx, kk in enumerate(k_fourier):
            # 色散关系
            omega = alpha * (1.0 - np.cos(kk * h)) + (V_mean +
                     HBAR_C ** 2 / (2.0 * self.m_red) * cent) / HBAR_C
            G = 1.0 - 1j * dt * omega / HBAR_C * HBAR_C
            G_squared[idx] = abs(G) ** 2

        # 最大增长因子
        G_max = np.sqrt(np.max(G_squared))

        # CFL 数
        cfl_diffusion = alpha * dt / h2
        cfl_convection = np.sqrt(2.0 * abs(V_mean) / self.m_red) * dt / h

        return {
            'scheme': 'FTCS',
            'stable': False,  # FTCS 对薛定谔方程无条件不稳定
            'G_max': G_max,
            'G_max_over_1': G_max - 1.0,
            'cfl_diffusion': cfl_diffusion,
            'cfl_convection': cfl_convection,
            'n_modes_analyzed': len(k_fourier),
            'max_growth_rate': (G_max - 1.0) / dt if dt > 0 else float('inf'),
            'instability_e_folding_time': (
                h2 * self.m_red / (HBAR_C * np.pi ** 2)
                if cfl_diffusion > 0 else float('inf')
            ),
        }

    def von_neumann_crank_nicolson(
        self,
        V_array: np.ndarray,
        dt: float,
        l_quantum: int = 0
    ) -> Dict:
        """
        Crank-Nicolson 格式的 von Neumann 稳定性分析

        CN 格式:
            (I + i*dt/(2*hbar) * H) * psi^{n+1} = (I - i*dt/(2*hbar) * H) * psi^n

        增长因子:
            G = (1 - i*dt*omega/(2*hbar)) / (1 + i*dt*omega/(2*hbar))

        => |G|^2 = 1 对所有 k, omega 成立 => 无条件稳定 (酉格式)

        返回:
            稳定性诊断字典
        """
        h = self.dr
        alpha = HBAR_C / (2.0 * self.m_red * h * h)
        V_mean = np.mean(np.real(V_array))

        k_fourier = np.linspace(0, np.pi / h, self.N // 2 + 1)

        G_max = 1.0  # CN 格式 |G| = 1 精确成立
        all_stable = True

        G_values = np.zeros(len(k_fourier), dtype=complex)
        for idx, kk in enumerate(k_fourier):
            omega_h = alpha * (1.0 - np.cos(kk * h))
            numer = 1.0 - 1j * dt * omega_h / (2.0 * HBAR_C)
            denom = 1.0 + 1j * dt * omega_h / (2.0 * HBAR_C)
            G_values[idx] = numer / denom if abs(denom) > 1e-30 else 1.0
            if abs(abs(G_values[idx]) - 1.0) > 1e-10:
                all_stable = False

        return {
            'scheme': 'Crank-Nicolson',
            'stable': all_stable,
            'G_max': G_max,
            'is_unitary': True,
            'max_G_deviation': float(np.max(np.abs(np.abs(G_values) - 1.0))),
            'n_modes_analyzed': len(k_fourier),
        }

    def matrix_spectral_radius(
        self,
        V_array: np.ndarray,
        dt: float,
        scheme: str = 'forward_euler'
    ) -> Dict:
        """
        差分算子矩阵的谱半径分析

        将差分格式写成矩阵形式: psi^{n+1} = A * psi^n
        稳定性条件: rho(A) = max|lambda_i(A)| <= 1 + C*dt

        参数:
            V_array: 势能数组
            dt: 时间步长
            scheme: 'forward_euler' | 'backward_euler' | 'crank_nicolson'

        返回:
            谱半径、特征值分布等信息
        """
        N = self.N
        h = self.dr
        h2 = h * h
        alpha = HBAR_C / (2.0 * self.m_red)

        # 构造哈密顿矩阵 (三对角)
        # H = -alpha * D2 + diag(V)
        H = np.zeros((N, N), dtype=complex)
        for i in range(N):
            H[i, i] = 2.0 * alpha / h2 + V_array[min(i, len(V_array) - 1)]
            if i > 0:
                H[i, i - 1] = -alpha / h2
            if i < N - 1:
                H[i, i + 1] = -alpha / h2

        # 根据格式构造传播矩阵 A
        I = np.eye(N, dtype=complex)
        if scheme == 'forward_euler':
            A = I - 1j * dt / HBAR_C * H
        elif scheme == 'backward_euler':
            A = np.linalg.solve(I + 1j * dt / HBAR_C * H, I)
        elif scheme == 'crank_nicolson':
            L = I + 1j * dt / (2.0 * HBAR_C) * H
            R = I - 1j * dt / (2.0 * HBAR_C) * H
            A = np.linalg.solve(L, R)
        else:
            raise ValueError(f"未知格式: {scheme}")

        # 计算特征值
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigenvalues))

        # 条件数
        cond_A = np.linalg.cond(A) if N < 500 else float('inf')

        return {
            'scheme': scheme,
            'spectral_radius': float(spectral_radius),
            'stable': spectral_radius <= 1.0 + 1e-10,
            'max_eigenvalue_abs': float(np.max(np.abs(eigenvalues))),
            'min_eigenvalue_abs': float(np.min(np.abs(eigenvalues))),
            'condition_number': float(cond_A),
            'matrix_size': N,
            'eigenvalue_spread': float(
                np.max(np.abs(eigenvalues)) - np.min(np.abs(eigenvalues))
            ),
        }

    def energy_conservation_test(
        self,
        psi_history: List[np.ndarray],
        V_array: np.ndarray,
        dr: float
    ) -> Dict:
        """
        时间演化中的能量守恒检验

        <E> = integral psi* H psi dr
            = integral [hbar^2/(2m) |dpsi/dr|^2 + V|psi|^2] dr

        对酉演化, <E> 应为常数

        参数:
            psi_history: 各时间步的波函数列表
            V_array: 势能数组
            dr: 空间步长

        返回:
            能量守恒诊断
        """
        alpha = HBAR_C / (2.0 * self.m_red)
        energies = []

        for psi in psi_history:
            n = len(psi)
            # 动能: hbar^2/(2m) * |dpsi/dr|^2
            dpsi = np.zeros(n, dtype=complex)
            for i in range(1, n - 1):
                dpsi[i] = (psi[i + 1] - psi[i - 1]) / (2.0 * dr)
            dpsi[0] = (psi[1] - psi[0]) / dr
            dpsi[-1] = (psi[-1] - psi[-2]) / dr

            T_density = alpha * np.abs(dpsi) ** 2
            V_density = np.real(V_array[:n]) * np.abs(psi) ** 2

            E_kin = np.sum(T_density) * dr
            E_pot = np.sum(V_density) * dr
            energies.append(E_kin + E_pot)

        energies = np.array(energies)
        E_mean = np.mean(energies)
        E_std = np.std(energies)
        E_rel_drift = abs(energies[-1] - energies[0]) / (abs(E_mean) + 1e-30)

        return {
            'E_mean': float(E_mean),
            'E_initial': float(energies[0]),
            'E_final': float(energies[-1]),
            'E_std': float(E_std),
            'relative_drift': float(E_rel_drift),
            'conserved': E_rel_drift < 1e-6,
            'n_steps': len(energies),
        }

    def numerov_stability_check(
        self,
        K2_array: np.ndarray,
        dr: float
    ) -> Dict:
        """
        Numerov 方法的稳定性分析

        Numerov 递推:
            (1 + h^2/12 * K_{n+1}^2) u_{n+1} = 2*(1 - 5h^2/12*K_n^2) u_n
                                                 - (1 + h^2/12*K_{n-1}^2) u_{n-1}

        稳定性条件 (常系数近似 K^2 = const):
            特征方程: (1+h^2K^2/12)*r^2 - 2*(1-5h^2K^2/12)*r + (1+h^2K^2/12) = 0

        判别式 Delta = 4*(1-5h^2K^2/12)^2 - 4*(1+h^2K^2/12)^2

        若 K^2 > 0 (振荡区): |r| = 1 (稳定)
        若 K^2 < 0 (指数区): 需要 h^2*|K^2| 不超过临界值

        临界步长:
            h_crit = sqrt(12 / |K^2|) (近似)

        返回:
            稳定性诊断
        """
        h2 = dr * dr
        h2_12 = h2 / 12.0

        # 最大 |K^2| 决定最严格稳定性限制
        K2_max = np.max(np.abs(K2_array))

        # 临界步长
        h_crit = np.sqrt(12.0 / K2_max) if K2_max > 0 else float('inf')

        # 稳定性参数
        stability_param = h2_12 * K2_max

        # 检查振荡/指数区域分布
        n_oscillatory = np.sum(K2_array > 0)
        n_exponential = np.sum(K2_array <= 0)

        # 有效 Courant 型参数
        effective_courant = dr * np.sqrt(K2_max) if K2_max > 0 else 0.0

        return {
            'stable': stability_param < 1.0,
            'stability_parameter': float(stability_param),
            'critical_step_size': float(h_crit),
            'actual_step_size': dr,
            'safety_factor': float(h_crit / dr) if dr > 0 else float('inf'),
            'n_oscillatory_points': int(n_oscillatory),
            'n_exponential_points': int(n_exponential),
            'effective_courant': float(effective_courant),
        }

    def unstable_ode_diagnostic(
        self,
        mu: float = 5.0,
        t_max: float = 2.0,
        n_steps: int = 100
    ) -> Dict:
        """
        不稳定 ODE 系统诊断 (映射自 1374_unstable_ode)

        测试系统:
            y' = A * y,  A = [[mu, 1/mu], [-1/mu, mu]]

        特征值: lambda = mu +/- i/mu
        实部 > 0 => 指数增长 (物理不稳定)

        精确解:
            y1(t) = exp(mu*t) * (cos(t/mu) - mu^2*sin(t/mu))

        用于验证数值求解器是否能正确捕捉物理不稳定性
        与数值不稳定性 (如 FTCS) 的区别

        返回:
            诊断结果
        """
        A = np.array([[mu, 1.0 / mu], [-1.0 / mu, mu]])
        eigenvalues = np.linalg.eigvals(A)

        dt = t_max / n_steps
        t = np.linspace(0, t_max, n_steps + 1)

        # 精确解
        y1_exact = np.exp(mu * t) * (np.cos(t / mu) - mu ** 2 * np.sin(t / mu))
        y2_exact = np.exp(mu * t) * (np.cos(t / mu) + np.sin(t / mu))

        # 前向 Euler 数值解
        y = np.array([1.0, 0.0])
        y_numerical = np.zeros((n_steps + 1, 2))
        y_numerical[0] = y

        for i in range(n_steps):
            y = y + dt * (A @ y)
            y_numerical[i + 1] = y

        # 误差
        error_y1 = np.abs(y_numerical[:, 0] - y1_exact)
        error_y2 = np.abs(y_numerical[:, 1] - y2_exact)

        # 增长因子比较
        growth_exact = abs(y1_exact[-1]) / (abs(y1_exact[0]) + 1e-30)
        growth_numerical = abs(y_numerical[-1, 0]) / (
            abs(y_numerical[0, 0]) + 1e-30)

        return {
            'eigenvalues': eigenvalues.tolist(),
            'real_part_positive': all(np.real(eigenvalues) > 0),
            'physical_instability': True,
            'growth_exact': float(growth_exact),
            'growth_numerical': float(growth_numerical),
            'max_error_y1': float(np.max(error_y1)),
            'max_error_y2': float(np.max(error_y2)),
            'dt': dt,
            'mu': mu,
        }
