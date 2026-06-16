# -*- coding: utf-8 -*-
"""
stability_analysis.py
======================================================================
von Neumann 稳定性分析与 CFL 条件计算

物理背景:
    辐射传输方程经有限差分离散后, 形成半离散系统:
        du/dt = L u
    其中 L 为空间差分算子矩阵。
    显式时间推进格式稳定的必要条件是:
        max |eigenvalue(L)| * dt <= stability_bound           (1)

    对不同差分格式:
        - Forward Euler:  stability_bound = 2.0 (实部)
        - RK2 (midpoint): stability_bound = 1.0
        - RK3 (TVD):      stability_bound = 2.52
        - RK4:            stability_bound = 2.83

    CFL 条件:
        CFL = |a| * dt / dx <= CFL_max                       (2)
    对纯对流: CFL_max = 1 (Forward Euler), 对高阶格式更小。

    本模块计算:
    1. 空间算子 L 的特征值谱
    2. 最大允许时间步长
    3. 不同格式的稳定性域可视化 (不输出, 仅计算)
    4. 耗散与色散分析

数学公式:
    修正波数 (modified wavenumber) 对 6 阶差分:
        k' dx = (4/3) sin(k dx) - (1/15) sin(2 k dx)
                + (4/45) sin(3 k dx)                        (3)
    虚部为耗散, 实部为色散。

依赖: numpy, high_order_fdm
======================================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

try:
    from . import high_order_fdm as fd
except ImportError:
    import high_order_fdm as fd


class StabilityParameters:
    """稳定性分析参数。"""

    def __init__(
        self,
        n_points: int = 64,
        dx: float = 0.1,
        advection_speed: float = 1.0,
        diffusion_coeff: float = 0.01,
        hyperdiffusion_coeff: float = 1.0e-4,
    ):
        self.n = n_points
        self.dx = dx
        self.a = advection_speed
        self.nu = diffusion_coeff
        self.nu4 = hyperdiffusion_coeff

        if n_points < 16:
            raise ValueError(f"稳定性分析需要至少 16 个点: {n_points}")
        if dx <= 0.0:
            raise ValueError(f"网格间距必须为正: {dx}")


def build_operator_matrix(
    params: StabilityParameters,
    scheme: str = "compact_4th",
) -> np.ndarray:
    """
    构建半离散算子矩阵 L。

    方程: du/dt = -a Du + nu D2 u - nu4 D4 u
    其中 D, D2, D4 分别为一阶、二阶、四阶差分算子。

    scheme: 'compact_4th' 或 'central_6th'
    """
    n = params.n
    dx = params.dx
    L = np.zeros((n, n), dtype=np.float64)

    # 对流项 (一阶差分)
    if scheme == "central_6th":
        for i in range(3, n - 3):
            L[i, i + 3] = -params.a * (-1.0) / (60.0 * dx)
            L[i, i + 2] = -params.a * (9.0) / (60.0 * dx)
            L[i, i + 1] = -params.a * (-45.0) / (60.0 * dx)
            L[i, i - 1] = -params.a * (45.0) / (60.0 * dx)
            L[i, i - 2] = -params.a * (-9.0) / (60.0 * dx)
            L[i, i - 3] = -params.a * (1.0) / (60.0 * dx)
    else:
        alpha = 0.25
        # Pade 隐式: 近似为显式处理
        for i in range(1, n - 1):
            L[i, i + 1] = -params.a * 3.0 / (4.0 * dx)
            L[i, i - 1] = params.a * 3.0 / (4.0 * dx)

    # 扩散项 (二阶差分, 4 阶精度)
    for i in range(2, n - 2):
        L[i, i + 2] += params.nu * (-1.0) / (12.0 * dx * dx)
        L[i, i + 1] += params.nu * (16.0) / (12.0 * dx * dx)
        L[i, i] += params.nu * (-30.0) / (12.0 * dx * dx)
        L[i, i - 1] += params.nu * (16.0) / (12.0 * dx * dx)
        L[i, i - 2] += params.nu * (-1.0) / (12.0 * dx * dx)

    # 超扩散项 (四阶差分, 双调和)
    coeff_nu4 = params.nu4
    if coeff_nu4 > 0.0:
        D2 = np.zeros((n, n), dtype=np.float64)
        for i in range(2, n - 2):
            D2[i, i + 2] = -1.0 / (12.0 * dx * dx)
            D2[i, i + 1] = 16.0 / (12.0 * dx * dx)
            D2[i, i] = -30.0 / (12.0 * dx * dx)
            D2[i, i - 1] = 16.0 / (12.0 * dx * dx)
            D2[i, i - 2] = -1.0 / (12.0 * dx * dx)
        D4 = D2 @ D2
        L -= coeff_nu4 * D4

    # Neumann 边界 (镜像)
    L[0, :] = L[1, :]
    L[-1, :] = L[-2, :]

    return L


def compute_eigenvalue_spectrum(
    L: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    计算算子矩阵的特征值谱。

    返回:
        eigenvalues: 复数特征值数组
        eigenvectors: 特征向量矩阵
        max_real_part: 最大实部 (决定稳定性)
    """
    eigenvalues, eigenvectors = np.linalg.eig(L)
    max_real = np.max(np.real(eigenvalues))
    return eigenvalues, eigenvectors, max_real


def compute_cfl_limit(
    params: StabilityParameters,
    scheme: str = "forward_euler",
) -> float:
    """
    计算 CFL 限制 (最大稳定时间步长)。

    对不同格式:
        Forward Euler: dt_max = 2 / |lambda_max|
        RK2:           dt_max = sqrt(2) / |lambda_max|
        RK3 (TVD):     dt_max = 2.52 / |lambda_max|
        RK4:           dt_max = 2.83 / |lambda_max|
    """
    L = build_operator_matrix(params, scheme="central_6th")
    eigenvalues, _, _ = compute_eigenvalue_spectrum(L)

    spec_radius = np.max(np.abs(eigenvalues))
    if spec_radius < 1.0e-30:
        return 1.0e10

    stability_bounds = {
        "forward_euler": 2.0,
        "rk2": np.sqrt(2.0),
        "rk3_tvd": 2.52,
        "rk4": 2.83,
        "leapfrog": 1.0,
    }

    bound = stability_bounds.get(scheme, 2.0)
    return bound / spec_radius


def modified_wavenumber_6th(k_dx: np.ndarray) -> np.ndarray:
    """
    6 阶中心差分的修正波数。
    k' dx = (4/3) sin(k dx) - (1/15) sin(2 k dx)
            + (4/45) sin(3 k dx)
    """
    return (
        (4.0 / 3.0) * np.sin(k_dx)
        - (1.0 / 15.0) * np.sin(2.0 * k_dx)
        + (4.0 / 45.0) * np.sin(3.0 * k_dx)
    )


def modified_wavenumber_4th_compact(k_dx: np.ndarray) -> np.ndarray:
    """
    4 阶紧致格式的修正波数 (隐式求解)。
    k' dx = (3 sin(k dx)) / (2 (1 + alpha cos(k dx)))
    其中 alpha = 1/4。
    """
    alpha = 0.25
    return 3.0 * np.sin(k_dx) / (2.0 * (1.0 + alpha * np.cos(k_dx)))


def dispersion_analysis(
    params: StabilityParameters,
) -> Dict[str, np.ndarray]:
    """
    色散与耗散分析。

    计算不同格式的修正波数与精确波数的偏差。
    """
    k_dx = np.linspace(0.0, np.pi, 200)
    k_exact = k_dx

    k_6th = modified_wavenumber_6th(k_dx)
    k_compact = modified_wavenumber_4th_compact(k_dx)
    k_2nd = np.sin(k_dx)

    error_2nd = np.abs(k_2nd - k_exact)
    error_4th = np.abs(k_compact - k_exact)
    error_6th = np.abs(k_6th - k_exact)

    return {
        "k_dx": k_dx,
        "k_exact": k_exact,
        "k_2nd": k_2nd,
        "k_4th_compact": k_compact,
        "k_6th": k_6th,
        "error_2nd": error_2nd,
        "error_4th": error_4th,
        "error_6th": error_6th,
    }


def von_neumann_growth_rate(
    k_dx: np.ndarray,
    dt: float,
    params: StabilityParameters,
    scheme: str = "forward_euler",
) -> np.ndarray:
    """
    von Neumann 增长率分析: 计算 amplification factor |G(k)|。

    对 du/dt = -a u_x + nu u_xx:
        G = 1 - dt * (i a k' + nu k'^2)  (Forward Euler)
        稳定性要求: |G| <= 1
    """
    k_prime = modified_wavenumber_6th(k_dx)
    growth_rate = np.abs(
        1.0 - dt * (1j * params.a * k_prime / params.dx - params.nu * k_prime**2 / params.dx**2)
    )
    return growth_rate


def full_stability_report(params: StabilityParameters) -> Dict:
    """
    生成完整稳定性报告。
    """
    report = {}

    for scheme in ["forward_euler", "rk2", "rk3_tvd", "rk4"]:
        dt_max = compute_cfl_limit(params, scheme)
        report[scheme] = {"dt_max": dt_max, "cfl_number": dt_max * params.a / params.dx}

    L = build_operator_matrix(params, "central_6th")
    eigenvalues, _, max_real = compute_eigenvalue_spectrum(L)
    report["spectral_radius"] = float(np.max(np.abs(eigenvalues)))
    report["max_real_part"] = float(max_real)
    report["min_real_part"] = float(np.min(np.real(eigenvalues)))

    disp = dispersion_analysis(params)
    report["dispersion"] = disp

    return report
