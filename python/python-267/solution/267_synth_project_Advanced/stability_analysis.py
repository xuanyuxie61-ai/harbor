"""
stability_analysis.py — von Neumann 稳定性分析与数值色散
=========================================================

本模块对 BHZ 哈密顿量的有限差分离散化进行系统的稳定性分析。
核心方法包括:

1. **von Neumann 稳定性分析**: 通过 Fourier 模式分解, 分析
   时间步进格式的放大因子 (amplification factor) 是否满足 |g| ≤ 1。

2. **数值色散关系**: 比较有限差分近似的等效波数 k* 与精确波数 k,
   量化色散误差对边界态能量精度的影响。

3. **CFL 条件**: 确定显式时间步进格式的最大允许时间步长。

4. **谱半径分析**: 计算空间离散算子的谱半径, 判断刚性 (stiffness)。

物理背景
--------
对于含时薛定谔方程 iℏ ∂ψ/∂t = H_fd ψ:
  - 空间离散引入修正波数 k*(k)
  - 时间离散引入放大因子 g(ωΔt)
  - 稳定性要求对所有模式 |g| ≤ 1 + O(Δt)

时间步进格式:
  - Forward Euler:     ψ^{n+1} = (I - iΔtH/ℏ) ψ^n
    → g = 1 - iωΔt, |g|² = 1 + (ωΔt)² > 1 → 无条件不稳定

  - Backward Euler:    (I + iΔtH/ℏ) ψ^{n+1} = ψ^n
    → g = 1/(1+iωΔt), |g|² = 1/(1+(ωΔt)²) < 1 → 无条件稳定 (但数值耗散)

  - Crank-Nicolson:    (I + iΔtH/2ℏ) ψ^{n+1} = (I - iΔtH/2ℏ) ψ^n
    → g = (1-iωΔt/2)/(1+iωΔt/2), |g| = 1 → 无条件稳定 (保幺正)

  - B1G3 隐式多步法:  三阶隐式格式, 稳定性区域更大

来源映射
--------
- 061_b1g3: 隐式多步 ODE 积分器的稳定性分析
- 1435_zoomin: 特征方程求根用于确定稳定性边界
"""

import numpy as np
from typing import Dict, Tuple, List
from fd_stencils import (
    first_derivative_coefficients,
    second_derivative_coefficients,
    modified_wavenumber_first,
    modified_wavenumber_second,
)


def amplification_factor_forward_euler(omega_dt: np.ndarray) -> np.ndarray:
    """
    Forward Euler 格式的放大因子。

    g_FE = 1 - iωΔt

    模: |g_FE|² = 1 + (ωΔt)²

    Parameters
    ----------
    omega_dt : ndarray
        无量纲频率 ωΔt

    Returns
    -------
    g : ndarray, complex
        放大因子
    """
    return 1.0 - 1j * omega_dt


def amplification_factor_backward_euler(omega_dt: np.ndarray) -> np.ndarray:
    """
    Backward Euler 格式的放大因子。

    g_BE = 1 / (1 + iωΔt)

    模: |g_BE|² = 1 / (1 + (ωΔt)²) < 1 → 数值耗散

    Parameters
    ----------
    omega_dt : ndarray

    Returns
    -------
    g : ndarray, complex
    """
    return 1.0 / (1.0 + 1j * omega_dt)


def amplification_factor_crank_nicolson(omega_dt: np.ndarray) -> np.ndarray:
    """
    Crank-Nicolson 格式的放大因子。

    g_CN = (1 - iωΔt/2) / (1 + iωΔt/2)

    模: |g_CN| = 1 → 严格保幺正 (unitary)

    Parameters
    ----------
    omega_dt : ndarray

    Returns
    -------
    g : ndarray, complex
    """
    return (1.0 - 0.5j * omega_dt) / (1.0 + 0.5j * omega_dt)


def amplification_factor_b1g3(omega_dt: np.ndarray) -> np.ndarray:
    """
    B1G3 隐式多步法的放大因子。

    B1G3 是一个三阶隐式方法, 其残差形式为:
        A₃y₃ + A₂y₂ + A₁y₁ = (2Δt/√3) f(t_m, y_m)

    其中系数:
        A₃ = 0.5 + 1/√3
        A₂ = -2/√3
        A₁ = -0.5 + 1/√3

    对测试方程 y' = λy (λ = -iω), 放大因子满足:
        (A₃ - z·B₃)g³ + (A₂ - z·B₂)g² + (A₁ - z·B₁)g = 0

    其中 z = λΔt = -iωΔt。

    对于简化的稳定性分析, 我们使用 B1G3 的有效放大因子近似:
        g_B1G3 ≈ exp(z·(A₃+A₂+A₁)/B_sum) 的低阶近似

    精确的稳定性区域需要通过求解多项式方程获得。

    Parameters
    ----------
    omega_dt : ndarray

    Returns
    -------
    g : ndarray, complex
        B1G3 的有效放大因子 (主模)
    """
    # B1G3 系数
    sqrt3 = np.sqrt(3.0)
    A3 = 0.5 + 1.0 / sqrt3
    A2 = -2.0 / sqrt3
    A1 = -0.5 + 1.0 / sqrt3
    B_coeff = 1.0 / sqrt3

    # 对 y' = -iωy, z = -iωΔt
    z = -1j * omega_dt

    # 简化的放大因子: g = (A₃ + A₂g⁻¹ + A₁g⁻²) / (z·B)
    # 对于单步等效: g ≈ 1 + z·B/(A₃+A₂+A₁) 的一阶近似
    # 更精确: 求解特征多项式

    # 稳定性多项式: A₃g³ + A₂g² + A₁g - zB(g²+...) = 0
    # 简化为有效单步: g_eff = (1 + z·B_eff) / (1 - z·B_eff_star)

    # 使用 B1G3 的 A-稳定性性质: 对 Re(z) ≤ 0, |g| ≤ 1
    # 对纯虚数 z = -iωΔt, B1G3 是无条件稳定的 (A-稳定)

    A_sum = A3 + A2 + A1  # = 0.5 + 1/√3 - 2/√3 - 0.5 + 1/√3 = 0
    # 注意 A_sum = 0, 这是 consistent 的条件

    # 有效放大因子: 通过求解 B1G3 的特征方程
    # 对于标量测试: A₃g³ + (A₂-zB₂)g² + (A₁-zB₁)g - zB₀ = 0
    # 其中 B₀, B₁, B₂ 是方法的具体系数

    # 简化: 使用 Crank-Nicolson-like 的等效形式
    # B1G3 的有效阶数 ≈ 3, 稳定性区域包含左半平面
    # g_B1G3 ≈ (1 - iωΔt/2 + (ωΔt)²/12) / (1 + iωΔt/2 + (ωΔt)²/12)

    omega_dt_sq = omega_dt ** 2
    numerator = 1.0 - 0.5j * omega_dt - omega_dt_sq / 12.0
    denominator = 1.0 + 0.5j * omega_dt - omega_dt_sq / 12.0

    # 避免除零
    denom_safe = np.where(np.abs(denominator) < 1e-15,
                          1e-15 + 0j, denominator)
    g = numerator / denom_safe

    return g


def stability_region_boundary(n_points: int = 1000,
                              scheme: str = 'forward_euler'
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算时间步进格式稳定性区域的边界。

    稳定性区域: {z ∈ ℂ : |g(z)| ≤ 1}

    对于标量测试方程 y' = λy, z = λΔt。

    边界由 |g(z)| = 1 确定。

    Parameters
    ----------
    n_points : int
        边界点数
    scheme : str
        时间格式名称

    Returns
    -------
    z_real, z_imag : ndarray
        稳定性区域边界的实部和虚部
    """
    # 参数化: g = exp(iθ), θ ∈ [0, 2π]
    theta = np.linspace(0, 2 * np.pi, n_points)
    g_values = np.exp(1j * theta)

    if scheme == 'forward_euler':
        # g = 1 + z → z = g - 1
        z = g_values - 1.0
    elif scheme == 'backward_euler':
        # g = 1/(1+z) → 1+z = 1/g → z = 1/g - 1
        z = 1.0 / g_values - 1.0
    elif scheme == 'crank_nicolson':
        # g = (1-z/2)/(1+z/2) → 解出 z
        # g(1+z/2) = 1-z/2 → g + gz/2 = 1-z/2 → z(g/2+1/2) = 1-g
        # z = 2(1-g)/(1+g)
        denom = 1.0 + g_values
        denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        z = 2.0 * (1.0 - g_values) / denom_safe
    else:
        raise ValueError(f"未知格式: {scheme}")

    return z.real, z.imag


def spectral_radius_fd(p: int, h: float, Nx: int,
                       operator_type: str = 'second') -> float:
    """
    计算有限差分算子的谱半径 (最大本征值绝对值)。

    对周期边界条件下的二阶导数算子:
        λ_max = (1/h²) × 2 Σ_{j=1}^{p} d_j (1 - cos(jπ))  (最高频率模式)
              = (1/h²) × 4 Σ_{j=1}^{p} d_j   (对奇数 j)

    这决定了时间步进的 CFL 条件:
        Δt ≤ C / ρ(D)

    其中 ρ(D) 是谱半径, C 是格式相关的常数。

    Parameters
    ----------
    p : int
        半带宽
    h : float
        网格间距
    Nx : int
        网格点数
    operator_type : str
        'first' 或 'second'

    Returns
    -------
    rho : float
        谱半径
    """
    if operator_type == 'second':
        d_coeffs, d_diag = second_derivative_coefficients(p)
        # 最高频率模式 k = π/h
        # λ_max = (1/h²) [d_diag + 2 Σ_j d_j cos(jπ)]
        # = (1/h²) [d_diag + 2 Σ_j d_j (-1)^j]
        lambda_max = abs(d_diag)
        for j_idx in range(p):
            j = j_idx + 1
            lambda_max += 2.0 * abs(d_coeffs[j_idx])  # 上界
        rho = lambda_max / (h * h)
    elif operator_type == 'first':
        c_coeffs = first_derivative_coefficients(p)
        # 最高频率模式
        rho = 0.0
        for j_idx in range(p):
            j = j_idx + 1
            rho += 2.0 * abs(c_coeffs[j_idx])
        rho /= h
    else:
        raise ValueError(f"未知算子类型: {operator_type}")

    return rho


def cfl_condition(p: int, h: float, scheme: str = 'forward_euler'
                  ) -> Dict[str, float]:
    """
    计算 CFL (Courant-Friedrichs-Lewy) 条件限制的最大时间步长。

    对于薛定谔方程 i∂ψ/∂t = Hψ, 空间离散后:
        ρ = 谱半径(H_fd)

    CFL 条件:
        Forward Euler:   Δt ≤ 2/ρ  (实际不稳定, 仅形式条件)
        Backward Euler:  无条件稳定
        Crank-Nicolson:  无条件稳定
        B1G3:            无条件稳定 (A-稳定)

    但对精度有要求: ω_max Δt < π/4 (每周期至少 8 步)

    Parameters
    ----------
    p : int
        半带宽
    h : float
        网格间距
    scheme : str
        时间格式

    Returns
    -------
    cfl : dict
    """
    # 使用二阶导数算子的谱半径作为特征能量尺度
    # (BHZ 哈密顿量的能量尺度正比于 1/h²)
    rho_2nd = spectral_radius_fd(p, h, Nx=100, operator_type='second')
    rho_1st = spectral_radius_fd(p, h, Nx=100, operator_type='first')

    # BHZ 模型的典型能量尺度 (使用 A 和 B 参数)
    # E_max ~ |A|/h × (FD修正) + |B|/h² × (FD修正)
    # 这里简化为 ρ_2nd 作为参考

    if scheme == 'forward_euler':
        dt_max_stability = 2.0 / rho_2nd if rho_2nd > 0 else np.inf
    elif scheme in ('backward_euler', 'crank_nicolson', 'b1g3'):
        dt_max_stability = np.inf  # 无条件稳定
    else:
        dt_max_stability = np.nan

    # 精度限制: 最高频率模式每周期至少 8 个时间步
    omega_max = rho_2nd  # 最高频率
    dt_max_accuracy = np.pi / (4.0 * omega_max) if omega_max > 0 else np.inf

    return {
        'spectral_radius_1st': rho_1st,
        'spectral_radius_2nd': rho_2nd,
        'max_energy_scale': rho_2nd,
        'dt_max_stability': dt_max_stability,
        'dt_max_accuracy': dt_max_accuracy,
        'recommended_dt': min(dt_max_accuracy, 0.01) if np.isfinite(dt_max_accuracy) else 0.01,
        'scheme': scheme,
        'unconditionally_stable': scheme != 'forward_euler'
    }


def numerical_dispersion_analysis(p_range: List[int],
                                  kh_max: float = np.pi,
                                  n_points: int = 500
                                  ) -> Dict:
    """
    系统分析不同阶数有限差分的数值色散特性。

    对每个半带宽 p, 计算:
    1. 一阶导数修正波数 k*₁(kh) 与 kh 的偏差
    2. 二阶导数修正波数 (k*₂h)² 与 (kh)² 的偏差
    3. BHZ 哈密顿量特征值的相对误差 (近似)

    对 BHZ 模型, 色散误差导致的能量误差:
        ΔE/E ~ 2(k*²-k²)/(k²) + O(A²/M²)

    Parameters
    ----------
    p_range : list of int
        要分析的半带宽列表
    kh_max : float
        最大无量纲波数
    n_points : int
        波数网格点数

    Returns
    -------
    results : dict
        包含各阶数色散误差的分析结果
    """
    kh = np.linspace(0.01, kh_max, n_points)
    results = {}

    for p in p_range:
        # 一阶导数色散
        k1_star = modified_wavenumber_first(kh, p)
        err_1st = np.abs(k1_star - kh) / np.abs(kh)

        # 二阶导数色散
        k2_star = modified_wavenumber_second(kh, p)
        err_2nd = np.abs(k2_star - kh**2) / kh**2

        # BHZ 能量误差估计
        # E(k) = ε(k) ± sqrt(M(k)² + A²k²)
        # 色散导致的误差主要来自 k² → k*² 的偏差
        delta_k2_rel = np.abs(k2_star - kh**2) / kh**2

        # 最大误差 (在 kh = π 附近)
        max_err_1st = float(np.max(err_1st))
        max_err_2nd = float(np.max(err_2nd))
        max_err_k2 = float(np.max(delta_k2_rel))

        # 收敛阶: 误差 ~ (kh)^{2p}
        # 通过拟合 log(error) vs log(kh) 确定
        if n_points > 10:
            log_kh = np.log(kh[:n_points//2])
            log_err = np.log(err_2nd[:n_points//2] + 1e-30)
            # 线性拟合
            coeffs = np.polyfit(log_kh, log_err, 1)
            convergence_order = coeffs[0]
        else:
            convergence_order = 2.0 * p

        results[p] = {
            'order_2p': 2 * p,
            'max_dispersion_error_1st': max_err_1st,
            'max_dispersion_error_2nd': max_err_2nd,
            'max_k2_error': max_err_k2,
            'convergence_order_measured': convergence_order,
            'expected_convergence_order': 2 * p,
            'kh_at_max_error': float(kh[np.argmax(err_2nd)])
        }

    return results


def von_neumann_analysis_report(p: int = 2, h: float = 0.5,
                                Nx: int = 20) -> str:
    """
    生成完整的 von Neumann 稳定性分析报告。

    Parameters
    ----------
    p : int
        半带宽
    h : float
        网格间距 (nm)
    Nx : int
        参考网格点数

    Returns
    -------
    report : str
        格式化的分析报告
    """
    lines = []
    lines.append("=" * 70)
    lines.append("von Neumann 稳定性分析报告")
    lines.append("=" * 70)
    lines.append(f"半带宽 p = {p} (精度 {2*p} 阶)")
    lines.append(f"网格间距 h = {h} nm")
    lines.append(f"参考网格点数 Nx = {Nx}")

    # 1. 谱半径
    lines.append("\n--- 谱半径分析 ---")
    rho_1 = spectral_radius_fd(p, h, Nx, 'first')
    rho_2 = spectral_radius_fd(p, h, Nx, 'second')
    lines.append(f"  一阶算子谱半径 ρ(D₁) = {rho_1:.6e} nm⁻¹")
    lines.append(f"  二阶算子谱半径 ρ(D₂) = {rho_2:.6e} nm⁻²")

    # 2. CFL 条件
    lines.append("\n--- CFL 条件 ---")
    for scheme in ['forward_euler', 'backward_euler',
                   'crank_nicolson', 'b1g3']:
        cfl = cfl_condition(p, h, scheme)
        lines.append(f"\n  格式: {scheme}")
        lines.append(f"    无条件稳定: {cfl['unconditionally_stable']}")
        lines.append(f"    Δt_max (稳定): {cfl['dt_max_stability']:.6e}")
        lines.append(f"    Δt_max (精度): {cfl['dt_max_accuracy']:.6e}")
        lines.append(f"    推荐 Δt: {cfl['recommended_dt']:.6e}")

    # 3. 色散分析
    lines.append("\n--- 数值色散分析 ---")
    disp = numerical_dispersion_analysis([p])
    for p_key, data in disp.items():
        lines.append(f"\n  p={p_key} ({2*p_key}阶精度):")
        lines.append(f"    一阶导数最大色散误差: {data['max_dispersion_error_1st']:.6e}")
        lines.append(f"    二阶导数最大色散误差: {data['max_dispersion_error_2nd']:.6e}")
        lines.append(f"    k² 最大相对误差: {data['max_k2_error']:.6e}")
        lines.append(f"    实测收敛阶: {data['convergence_order_measured']:.2f}")
        lines.append(f"    理论收敛阶: {data['expected_convergence_order']}")

    # 4. 稳定性区域
    lines.append("\n--- 稳定性区域特征 ---")
    lines.append("  Forward Euler:    圆 |1+z| ≤ 1, 中心 z=-1, 半径 1")
    lines.append("  Backward Euler:   |1+z| ≥ 1 的外部区域")
    lines.append("  Crank-Nicolson:   Re(z) ≤ 0 的左半平面")
    lines.append("  B1G3:             包含左半平面 (A-稳定)")

    lines.append("\n  对薛定谔方程 i∂ψ/∂t = Hψ:")
    lines.append("    特征值 z = -iEΔt/ℏ 在虚轴上")
    lines.append("    Forward Euler: 虚轴不在稳定区域内 → 不稳定")
    lines.append("    Crank-Nicolson: 虚轴在稳定区域边界 → 条件稳定 (|g|=1)")
    lines.append("    Backward Euler: 虚轴在稳定区域内 → 稳定但有数值耗散")
    lines.append("    B1G3: 虚轴在稳定区域内 → 稳定")

    return "\n".join(lines)
