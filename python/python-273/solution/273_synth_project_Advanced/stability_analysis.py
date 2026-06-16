"""
stability_analysis.py — von Neumann 稳定性分析与 CFL 条件
=========================================================

融合种子项目:
  - 368_fd2d_poisson: 差分格式的数值稳定性
  - 004_alpert_rule: 高阶精度格式的色散与耗散分析
  - 818_normal_ode: 指数衰减 ODE 的数值稳定性域

物理背景:
  晶格动力学时间积分的稳定性要求:
    dt < dt_max = 2 / omega_max

  其中 omega_max 是最高声子频率。
  这是 CFL (Courant-Friedrichs-Lewy) 条件的晶格动力学版本。

  对空间离散化, von Neumann 分析给出修正波数:
    (k*h)_eff^2 = -sum_j s_j * exp(i*j*k*h)

  数值色散: omega_eff(k) 与 omega_exact(k) 的偏差
  数值耗散: 虚部导致振幅衰减

  高阶差分模板的优势:
    2阶: (kh)_eff ≈ kh - (kh)^3/12 + ...
    4阶: (kh)_eff ≈ kh + (kh)^5/90 + ...
    6阶: (kh)_eff ≈ kh - (kh)^7/3780 + ...
"""

import numpy as np
from typing import Tuple, Dict
from fd_stencil import get_fd_stencil, spectral_analysis_fd


def von_neumann_stability_analysis(
    dynamical_matrix: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """
    von Neumann 稳定性分析。

    对运动方程 d^2u/dt^2 = -D*u, 离散化为:
      u^{n+1} = 2*u^n - u^{n-1} - dt^2 * D * u^n

    放大因子 g 满足:
      g^2 - 2*g + 1 + dt^2 * lambda_k = 0
      g = 1 - dt^2 * lambda_k / 2 ± sqrt((dt^2 * lambda_k / 2)^2 - dt^2 * lambda_k)

    稳定性条件: |g| <= 1 对所有 k
    等价于: dt^2 * lambda_max <= 4
    即: dt <= 2 / omega_max

    参数:
        dynamical_matrix: (N, N) 动力学矩阵
        dt: 时间步长

    返回:
        results: 稳定性分析结果字典
    """
    # 求解特征值
    eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
    eigenvalues = np.sort(eigenvalues)

    # 处理负特征值 (不稳定模式)
    omega_sq = eigenvalues
    omega = np.sqrt(np.maximum(omega_sq, 0.0))
    omega_max = omega[-1]

    # CFL 条件
    dt_max = 2.0 / max(omega_max, 1e-15)
    cfl_number = dt / dt_max if dt_max > 0 else np.inf

    # 放大因子谱
    stable = True
    for lam in omega_sq:
        discriminant = (dt ** 2 * lam / 2.0) ** 2 - dt ** 2 * lam
        if discriminant > 0:
            g1 = 1.0 - dt ** 2 * lam / 2.0 + np.sqrt(discriminant)
            g2 = 1.0 - dt ** 2 * lam / 2.0 - np.sqrt(discriminant)
            if max(abs(g1), abs(g2)) > 1.0 + 1e-10:
                stable = False
        else:
            g_mag_sq = (1.0 - dt ** 2 * lam / 2.0) ** 2 + abs(discriminant)
            if g_mag_sq > 1.0 + 1e-10:
                stable = False

    return {
        'omega_max': omega_max,
        'dt_max': dt_max,
        'cfl_number': cfl_number,
        'is_stable': stable and cfl_number <= 1.0,
        'omega_min': omega[0],
        'frequency_range': (omega[0], omega_max),
        'n_modes': len(omega),
        'n_unstable': int(np.sum(omega_sq < -1e-10)),
    }


def compute_cfl_condition(
    h: float,
    wave_speed: float,
    spatial_order: int = 2,
    n_dim: int = 3,
) -> Dict[str, float]:
    """
    CFL 条件计算。

    对波动方程 d^2u/dt^2 = c^2 * Laplacian(u):
      CFL = c * dt / h <= CFL_max

    CFL_max 依赖于:
      - 空间维度 n_dim
      - 差分精度阶数 spatial_order
      - 网格类型

    对 2阶中心差分, 3D:
      CFL_max = 1 / sqrt(3) ≈ 0.577

    对 4阶中心差分:
      CFL_max ≈ 0.539 (略小, 因为模板更宽)

    参数:
        h: 网格间距
        wave_speed: 波速 c
        spatial_order: 空间差分阶数
        n_dim: 空间维度

    返回:
        cfl_info: CFL 条件信息
    """
    # 经验 CFL 上限 (随阶数增加而减小)
    cfl_max_map = {
        (2, 1): 1.0,
        (2, 2): 1.0 / np.sqrt(2),
        (2, 3): 1.0 / np.sqrt(3),
        (4, 1): 0.95,
        (4, 2): 0.67,
        (4, 3): 0.54,
        (6, 1): 0.90,
        (6, 2): 0.63,
        (6, 3): 0.51,
        (8, 1): 0.87,
        (8, 2): 0.61,
        (8, 3): 0.49,
    }
    key = (spatial_order, n_dim)
    cfl_max = cfl_max_map.get(key, 1.0 / np.sqrt(n_dim))

    dt_max = cfl_max * h / max(wave_speed, 1e-15)

    return {
        'cfl_max': cfl_max,
        'dt_max': dt_max,
        'wave_speed': wave_speed,
        'grid_spacing': h,
        'spatial_order': spatial_order,
        'n_dim': n_dim,
    }


def numerical_dispersion_analysis(
    h: float,
    c: float,
    spatial_order: int = 4,
    n_k: int = 200,
) -> Dict[str, np.ndarray]:
    """
    数值色散分析: omega_numerical(k) vs omega_exact(k)。

    对波动方程, 精确色散: omega = c * k
    数值色散: omega_num = (c/h) * (kh)_eff

    其中 (kh)_eff 由差分模板的谱分析给出。

    返回:
        results: {k, omega_exact, omega_numerical, relative_error}
    """
    stencil = get_fd_stencil(spatial_order)
    kh, kh_eff = spectral_analysis_fd(stencil, h, n_k)

    k_values = kh / h
    omega_exact = c * k_values
    omega_numerical = c * kh_eff / h

    rel_error = np.zeros_like(omega_exact)
    nonzero = omega_exact > 1e-15
    rel_error[nonzero] = abs(omega_numerical[nonzero] - omega_exact[nonzero]) / omega_exact[nonzero]

    return {
        'k': k_values,
        'omega_exact': omega_exact,
        'omega_numerical': omega_numerical,
        'relative_error': rel_error,
        'kh': kh,
        'kh_eff': kh_eff,
    }


def numerical_dissipation_analysis(
    stencil: np.ndarray,
    h: float,
    n_k: int = 200,
) -> Dict[str, np.ndarray]:
    """
    数值耗散分析: 差分模板的虚部 (非物理衰减)。

    中心差分模板应无耗散 (纯实数符号色散),
    但非中心模板 (如 upwind) 会引入数值耗散。

    对声子计算, 确保使用中心差分以保持哈密顿结构。
    """
    kh = np.linspace(0, np.pi, n_k)
    half_w = len(stencil) // 2
    symbol = np.zeros(n_k, dtype=complex)
    for j, s in enumerate(stencil):
        m = j - half_w
        symbol += s * np.exp(1j * m * kh)

    dissipation = np.abs(np.imag(symbol))

    return {
        'kh': kh,
        'dissipation': dissipation,
        'max_dissipation': np.max(dissipation),
    }


def energy_conservation_check(
    total_energy: np.ndarray,
    times: np.ndarray,
) -> Dict[str, float]:
    """
    能量守恒检查 (融合 md_parfor 的 e_lost 计算)。

    relative_drift = |E(T) - E(0)| / |E(0)|
    rms_fluctuation = std(E) / |mean(E)|

    辛积分器应显示:
      - relative_drift ~ O(dt^2) (不随 T 增长)
      - 能量在均值附近振荡 (无长期漂移)
    """
    E0 = total_energy[0]
    if abs(E0) < 1e-30:
        E0 = 1e-30

    relative_drift = abs(total_energy[-1] - E0) / abs(E0)
    mean_E = np.mean(total_energy)
    rms_fluct = np.std(total_energy) / max(abs(mean_E), 1e-30)

    # 线性漂移拟合
    if len(times) > 2:
        coeffs = np.polyfit(times, total_energy, 1)
        linear_drift_rate = coeffs[0]  # dE/dt
    else:
        linear_drift_rate = 0.0

    return {
        'relative_drift': relative_drift,
        'rms_fluctuation': rms_fluct,
        'linear_drift_rate': linear_drift_rate,
        'E_initial': E0,
        'E_final': total_energy[-1],
    }


def adaptive_dt_control(
    omega_max: float,
    safety_factor: float = 0.9,
    dt_min: float = 1e-6,
    dt_max: float = 1.0,
) -> float:
    """
    自适应时间步控制 (融合 normal_ode 的稳定性约束)。

    dt = safety_factor * 2 / omega_max

    参数:
        omega_max: 最高声子频率
        safety_factor: 安全因子 (< 1)
        dt_min, dt_max: 步长上下界

    返回:
        dt: 推荐时间步长
    """
    if omega_max < 1e-15:
        return dt_max
    dt = safety_factor * 2.0 / omega_max
    return np.clip(dt, dt_min, dt_max)
