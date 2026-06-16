#!/usr/bin/env python3
"""
stability_analysis.py
=====================
数值稳定性分析模块。

包含:
  - Von Neumann 稳定性分析 (傅里叶分析)
  - 矩阵稳定性分析 (特征值谱)
  - CFL 条件计算
  - 色散与耗散关系
  - 刚性 ODE 稳定性域分析

数学基础:
  Von Neumann 分析:
    对线性格式 u_j^{n+1} = Σ_k g_k u_{j+k}^n
    令 u_j^n = ξ^n e^{i k j h}
    得到放大因子 g(kh) = Σ_k g_k e^{i k kh}
    稳定性要求: |g(kh)| ≤ 1 + O(Δt) 对所有 kh

  CFL 条件:
    Δt ≤ C_CFL × h / max|λ|
    其中 λ 为特征速度, h 为最小网格间距。

  矩阵方法:
    将半离散系统 du/dt = A u 离散化后,
    时间积分稳定的条件是 Δt × spec(A) ⊂ 稳定性域。

色散关系 (对流方程 ∂u/∂t + c ∂u/∂x = 0):
  数值色散: ω_num(k) = Im(ln g(k)) / Δt
  数值耗散: |g(k)| = exp(-α_num Δt)
"""

import numpy as np
from scipy.linalg import eigvals, eigh


class VonNeumannAnalysis:
    """
    Von Neumann 稳定性分析。
    对给定的有限差分格式, 计算放大因子谱。
    """

    @staticmethod
    def amplification_factor_upwind1(c, dt, dx, n_theta=360):
        """
        一阶迎风格式的放大因子:
          u_j^{n+1} = u_j^n - ν (u_j^n - u_{j-1}^n)
        其中 ν = c Δt / Δx (Courant 数)

        g(θ) = 1 - ν (1 - e^{-iθ})
             = 1 - ν + ν cos θ - i ν sin θ
        |g|² = (1 - ν + ν cos θ)² + (ν sin θ)²
             = 1 - 2ν(1-ν)(1 - cos θ)

        稳定条件: 0 ≤ ν ≤ 1
        """
        theta = np.linspace(0, 2 * np.pi, n_theta)
        nu = c * dt / dx
        g = 1.0 - nu * (1.0 - np.exp(-1j * theta))
        return theta, g, nu

    @staticmethod
    def amplification_factor_lax_wendroff(c, dt, dx, n_theta=360):
        """
        Lax-Wendroff 格式的放大因子:
          u_j^{n+1} = u_j^n - (ν/2)(u_{j+1}^n - u_{j-1}^n) + (ν²/2)(u_{j+1}^n - 2u_j^n + u_{j-1}^n)
        g(θ) = 1 - i ν sin θ + ν² (cos θ - 1)
        稳定条件: |ν| ≤ 1
        """
        theta = np.linspace(0, 2 * np.pi, n_theta)
        nu = c * dt / dx
        g = 1.0 - 1j * nu * np.sin(theta) + nu ** 2 * (np.cos(theta) - 1.0)
        return theta, g, nu

    @staticmethod
    def amplification_factor_weno5_linear(c, dt, dx, n_theta=360):
        """
        WENO5 的线性化放大因子 (假设光滑区, 权重退化为理想权重)。
        用于分析 WENO5 在光滑区的色散和耗散特性。

        WENO5 在光滑区的修正波数:
          k' h = arctan(Im(g)/Re(g))
        其中 g 为线性化放大因子。
        """
        theta = np.linspace(0, 2 * np.pi, n_theta)
        nu = c * dt / dx

        # WENO5 的三阶迎风子格式 (理想权重 d₀=0.1, d₁=0.6, d₂=0.3)
        # 简化为 5 阶迎风-中心混合格式
        g = (1.0 - nu * (1.0 - np.exp(-1j * theta)) *
             (1.0 + nu / 6.0 * (1.0 - np.exp(-1j * theta)) *
              (2.0 - np.exp(-1j * theta))))

        return theta, g, nu

    @staticmethod
    def amplification_factor_compact4(c, dt, dx, n_theta=360):
        """
        4阶紧致差分 + RK3 时间推进的放大因子。
        需要求解三对角系统。
        """
        theta = np.linspace(0, 2 * np.pi, n_theta)
        nu = c * dt / dx

        # 紧致差分的修正波数
        # α k'_{i-1} + k'_i + α k'_{i+1} = a (sin θ)/h
        # k'h = a sin θ / (1 + 2α cos θ)
        alpha = 0.25
        a = 1.5
        k_prime_h = a * np.sin(theta) / (1.0 + 2.0 * alpha * np.cos(theta))

        # RK3 时间推进 (TVD-RK3, Shu-Osher)
        # 放大因子: g = 1 + z + z²/2 + z³/6 (对线性问题等价于 RK3)
        z = -1j * nu * k_prime_h
        g = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0

        return theta, g, nu

    @staticmethod
    def check_stability(g_array, tol=1.0e-10):
        """
        检查稳定性: max|g| ≤ 1 + tol
        """
        max_g = np.max(np.abs(g_array))
        is_stable = max_g <= 1.0 + tol
        return is_stable, max_g

    @staticmethod
    def dissipation_dispersion(g_array, theta):
        """
        从放大因子提取耗散和色散关系。

        数值耗散率:
          α_num = -ln|g| / Δt
        数值色散:
          ω_num = arg(g) / Δt
        修正波数:
          k' h = arg(g) / ν (对对流方程)
        """
        mag_g = np.abs(g_array)
        phase_g = np.angle(g_array)

        # 数值耗散 (对数衰减率)
        dissipation = -np.log(np.maximum(mag_g, 1.0e-30))

        # 数值色散
        dispersion = phase_g

        return dissipation, dispersion


class MatrixStabilityAnalysis:
    """
    矩阵方法稳定性分析。
    对半离散系统 du/dt = A u, 分析 A 的特征值谱。
    """

    @staticmethod
    def fd_matrix_1d(N, dx, order=2, bc_type='dirichlet'):
        """
        构建一阶导数的有限差分矩阵。
        用于分析空间离散的谱特性。

        参数:
            N: 网格点数
            dx: 网格间距
            order: 精度阶数 (2 或 4)
            bc_type: 边界条件类型
        返回:
            A: (N, N) 差分矩阵
        """
        if order == 2:
            # 二阶中心差分
            A = np.zeros((N, N))
            for i in range(1, N - 1):
                A[i, i - 1] = -0.5 / dx
                A[i, i + 1] = 0.5 / dx
        elif order == 4:
            A = np.zeros((N, N))
            for i in range(2, N - 2):
                A[i, i - 2] = 1.0 / (12 * dx)
                A[i, i - 1] = -2.0 / (3 * dx)
                A[i, i + 1] = 2.0 / (3 * dx)
                A[i, i + 2] = -1.0 / (12 * dx)
        else:
            raise ValueError(f"不支持的阶数: {order}")

        # 边界处理
        if bc_type == 'dirichlet':
            A[0, :] = 0.0
            A[-1, :] = 0.0
        elif bc_type == 'periodic':
            if order == 2:
                A[0, N - 1] = -0.5 / dx
                A[0, 1] = 0.5 / dx
                A[N - 1, N - 2] = -0.5 / dx
                A[N - 1, 0] = 0.5 / dx
            elif order == 4:
                A[0, N - 2] = 1.0 / (12 * dx)
                A[0, N - 1] = -2.0 / (3 * dx)
                A[0, 1] = 2.0 / (3 * dx)
                A[0, 2] = -1.0 / (12 * dx)
                A[N - 1, N - 3] = 1.0 / (12 * dx)
                A[N - 1, N - 2] = -2.0 / (3 * dx)
                A[N - 1, 0] = 2.0 / (3 * dx)
                A[N - 1, 1] = -1.0 / (12 * dx)

        return A

    @staticmethod
    def laplacian_matrix_1d(N, dx, bc_type='dirichlet'):
        """
        二阶导数 (Laplacian) 矩阵。
        对应 ∂²u/∂x² 的离散。
        """
        A = np.zeros((N, N))
        for i in range(1, N - 1):
            A[i, i - 1] = 1.0 / dx ** 2
            A[i, i] = -2.0 / dx ** 2
            A[i, i + 1] = 1.0 / dx ** 2

        if bc_type == 'periodic':
            A[0, N - 1] = 1.0 / dx ** 2
            A[N - 1, 0] = 1.0 / dx ** 2

        return A

    @staticmethod
    def eigenvalue_spectrum(A):
        """
        计算矩阵特征值谱。
        返回:
            eigvals: 特征值 (复数)
            spectral_radius: 谱半径
            stiffness_ratio: 刚性比 (|λ_max| / |λ_min_nonzero|)
        """
        eigs = eigvals(A)
        abs_eigs = np.abs(eigs)
        sorted_abs = np.sort(abs_eigs)

        spectral_radius = sorted_abs[-1] if len(sorted_abs) > 0 else 0.0

        # 刚性比
        nonzero_eigs = sorted_abs[sorted_abs > 1.0e-14]
        if len(nonzero_eigs) >= 2:
            stiffness = nonzero_eigs[-1] / nonzero_eigs[0]
        else:
            stiffness = float('inf')

        return eigs, spectral_radius, stiffness

    @staticmethod
    def max_stable_dt(A, method='rk4'):
        """
        根据特征值谱计算最大稳定时间步长。

        对方法 stability domain:
          RK1 (Euler): |1 + z| ≤ 1, z ∈ disk(-1, 1)
          RK2: 椭圆形域
          RK4: 近似 |z| ≤ 2√2

        参数:
            A: 半离散矩阵
            method: 时间积分方法 ('euler', 'rk2', 'rk4')
        返回:
            dt_max: 最大稳定时间步长
        """
        eigs, spec_rad, _ = MatrixStabilityAnalysis.eigenvalue_spectrum(A)

        if spec_rad < 1.0e-14:
            return float('inf')

        # 各方法的稳定性域半径
        stability_radius = {
            'euler': 1.0,
            'rk2': 2.0,
            'rk3': 2.51,
            'rk4': 2.83,
        }
        r = stability_radius.get(method, 2.83)
        dt_max = r / spec_rad
        return dt_max


class CFLCondition:
    """
    CFL 条件计算器。
    用于确定 ICF 内爆模拟的最大稳定时间步长。
    """

    @staticmethod
    def acoustic_cfl(sound_speed, dx, dy, cfl_number=0.5):
        """
        声学 CFL 条件:
          Δt ≤ C_CFL × min(Δx, Δy) / max(c_s)

        参数:
            sound_speed: 声速场, shape=(N_r, N_z) [cm/s]
            dx, dy: 网格间距 [cm]
            cfl_number: CFL 数 (通常 0.3~0.8)
        返回:
            dt_max: 最大稳定时间步长 [s]
        """
        dh = min(dx, dy)
        cs_max = np.max(np.abs(sound_speed))
        if cs_max < 1.0e-30:
            return float('inf')
        return cfl_number * dh / cs_max

    @staticmethod
    def advective_cfl(velocity, dx, dy, cfl_number=0.5):
        """
        对流 CFL 条件:
          Δt ≤ C_CFL × min(Δx, Δy) / max(|u|)
        """
        dh = min(dx, dy)
        u_max = np.max(np.abs(velocity))
        if u_max < 1.0e-30:
            return float('inf')
        return cfl_number * dh / u_max

    @staticmethod
    def diffusive_cfl(diffusivity, dx, dy, cfl_number=0.25):
        """
        扩散 CFL 条件 (显式格式):
          Δt ≤ C_CFL × min(Δx, Δy)² / max(ν)
        注意: 扩散过程的 CFL 数通常为 0.25 或更小。
        """
        dh = min(dx, dy)
        nu_max = np.max(np.abs(diffusivity))
        if nu_max < 1.0e-30:
            return float('inf')
        return cfl_number * dh ** 2 / nu_max

    @staticmethod
    def combined_cfl(sound_speed, velocity, diffusivity, dx, dy, cfl_number=0.4):
        """
        综合 CFL 条件 (声学 + 对流 + 扩散):
          1/Δt ≤ 1/Δt_acoustic + 1/Δt_advective + 1/Δt_diffusive
        """
        dt_a = CFLCondition.acoustic_cfl(sound_speed, dx, dy, cfl_number)
        dt_v = CFLCondition.advective_cfl(velocity, dx, dy, cfl_number)
        dt_d = CFLCondition.diffusive_cfl(diffusivity, dx, dy, cfl_number * 0.5)

        inv_dt = 0.0
        if dt_a < float('inf'):
            inv_dt += 1.0 / dt_a
        if dt_v < float('inf'):
            inv_dt += 1.0 / dt_v
        if dt_d < float('inf'):
            inv_dt += 1.0 / dt_d

        return 1.0 / inv_dt if inv_dt > 0 else float('inf')


class StiffnessAnalyzer:
    """
    刚性分析器。
    分析 ICF 内爆中多尺度物理过程的刚性特征。
    """

    @staticmethod
    def analyze_reaction_diffusion(D, k_react, L):
        """
        反应-扩散系统的刚性分析。
        Damköhler 数 Da = k_react × L² / D
        Da >> 1 时系统刚性。

        参数:
            D: 扩散系数
            k_react: 反应速率常数
            L: 特征长度
        返回:
            Da: Damköhler 数
            stiffness_estimate: 刚性估计
        """
        Da = k_react * L ** 2 / max(D, 1.0e-30)
        return Da, Da

    @staticmethod
    def stability_domain_rk4(n_points=200):
        """
        生成 RK4 方法的稳定性域边界。
        |g(z)| = |1 + z + z²/2 + z³/6 + z⁴/24| = 1
        """
        theta = np.linspace(0, 2 * np.pi, n_points)
        # 参数化边界 (近似)
        # RK4 稳定性域在实轴上的范围: [-2.83, 0]
        # 在虚轴上的范围: [-2√2, 2√2]
        x = np.linspace(-3.5, 0.5, n_points)
        y = np.linspace(-3.5, 3.5, n_points)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y
        g = 1.0 + Z + Z ** 2 / 2.0 + Z ** 3 / 6.0 + Z ** 4 / 24.0
        return X, Y, np.abs(g)


def analyze_icf_stability(N=64, dx=0.01, gamma=5.0 / 3.0, rho0=1.0, p0=1.0e12):
    """
    综合 ICF 稳定性分析。
    对典型 ICF 参数进行 von Neumann 和矩阵稳定性分析。
    """
    # 声速
    cs = np.sqrt(gamma * p0 / rho0)

    # CFL 条件
    cfl = CFLCondition()
    dt_acoustic = cfl.acoustic_cfl(cs, dx, dx, cfl_number=0.5)

    # 矩阵分析
    A = MatrixStabilityAnalysis.fd_matrix_1d(N, dx, order=4, bc_type='periodic')
    eigs, spec_rad, stiffness = MatrixStabilityAnalysis.eigenvalue_spectrum(A)
    dt_matrix = MatrixStabilityAnalysis.max_stable_dt(A, method='rk4')

    # Von Neumann 分析
    vna = VonNeumannAnalysis()
    theta, g_lw, nu_lw = vna.amplification_factor_lax_wendroff(cs, dt_acoustic * 0.5, dx)
    is_stable, max_g = vna.check_stability(g_lw)

    return {
        'sound_speed': cs,
        'dt_acoustic': dt_acoustic,
        'dt_matrix': dt_matrix,
        'spectral_radius': spec_rad,
        'stiffness': stiffness,
        'lax_wendroff_stable': is_stable,
        'max_amplification': max_g,
        'eigenvalues_real': np.real(eigs),
        'eigenvalues_imag': np.imag(eigs)
    }
