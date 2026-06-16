"""
高阶有限差分模板模块

实现 2/4/6/8 阶中心差分模板，用于 PIC 中泊松方程的空间离散。
基于 John Burkardt 的 r8ncf 带状矩阵存储思想，提供矩阵-向量乘积操作。

核心公式：
    一阶导数 (中心差分):
        2阶: f'(x) ≈ (-f_{i+1} + f_{i-1}) / (2*dx)
        4阶: f'(x) ≈ (f_{i-2} - 8*f_{i-1} + 8*f_{i+1} - f_{i+2}) / (12*dx)
        6阶: f'(x) ≈ (-f_{i-3} + 9*f_{i-2} - 45*f_{i-1} + 45*f_{i+1} - 9*f_{i+2} + f_{i+3}) / (60*dx)

    二阶导数:
        2阶: f''(x) ≈ (f_{i-1} - 2*f_i + f_{i+1}) / dx^2
        4阶: f''(x) ≈ (-f_{i-2} + 16*f_{i-1} - 30*f_i + 16*f_{i+1} - f_{i+2}) / (12*dx^2)
        6阶: f''(x) ≈ (2*f_{i-3} - 27*f_{i-2} + 270*f_{i-1} - 490*f_i
                           + 270*f_{i+1} - 27*f_{i+2} + 2*f_{i+3}) / (180*dx^2)
"""

import numpy as np
from typing import Tuple, Dict


# ============================================================================
# 有限差分模板系数 (中心差分)
# ============================================================================
# 一阶导数系数, 键: 阶数, 值: (偏移列表, 系数列表, 除数)
FD_FIRST_DERIV: Dict[int, Tuple[list, list, float]] = {
    2: ([-1, 1],
        [-1.0, 1.0],
        2.0),
    4: ([-2, -1, 1, 2],
        [1.0, -8.0, 8.0, -1.0],
        12.0),
    6: ([-3, -2, -1, 1, 2, 3],
        [-1.0, 9.0, -45.0, 45.0, -9.0, 1.0],
        60.0),
    8: ([-4, -3, -2, -1, 1, 2, 3, 4],
        [1.0, -32.0/3.0, 168.0, -672.0, 672.0, -168.0, 32.0/3.0, -1.0],
        840.0 / 1.0),  # = 840
}

# 修正 8 阶: 使用标准系数
FD_FIRST_DERIV[8] = ([-4, -3, -2, -1, 1, 2, 3, 4],
                      [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0,
                       4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0],
                      1.0)  # 系数已含分母

# 二阶导数系数
FD_SECOND_DERIV: Dict[int, Tuple[list, list, float]] = {
    2: ([-1, 0, 1],
        [1.0, -2.0, 1.0],
        1.0),
    4: ([-2, -1, 0, 1, 2],
        [-1.0, 16.0, -30.0, 16.0, -1.0],
        12.0),
    6: ([-3, -2, -1, 0, 1, 2, 3],
        [2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0],
        180.0),
    8: ([-4, -3, -2, -1, 0, 1, 2, 3, 4],
        [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0,
         8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0],
        1.0),
}


def get_stencil(order: int, derivative: int = 2) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    获取指定阶数的有限差分模板

    Parameters:
        order      : 精度阶数 (2, 4, 6, 8)
        derivative : 导数阶数 (1 或 2)

    Returns:
        (offsets, coeffs, divisor) : 偏移、系数、除数
    """
    if derivative == 1:
        if order not in FD_FIRST_DERIV:
            raise ValueError(f"一阶导数不支持 {order} 阶精度, 可选: {list(FD_FIRST_DERIV.keys())}")
        offsets, coeffs, divisor = FD_FIRST_DERIV[order]
    elif derivative == 2:
        if order not in FD_SECOND_DERIV:
            raise ValueError(f"二阶导数不支持 {order} 阶精度, 可选: {list(FD_SECOND_DERIV.keys())}")
        offsets, coeffs, divisor = FD_SECOND_DERIV[order]
    else:
        raise ValueError(f"不支持 {derivative} 阶导数")
    return np.array(offsets), np.array(coeffs), divisor


def apply_fd_1d(f: np.ndarray, dx: float, order: int = 2,
                 derivative: int = 2, bc: str = 'periodic') -> np.ndarray:
    """
    对 1D 数组应用有限差分算子

    Parameters:
        f          : 输入数组 [N]
        dx         : 网格间距
        order      : 精度阶数 (2, 4, 6, 8)
        derivative : 导数阶数 (1 或 2)
        bc         : 边界条件 ('periodic', 'dirichlet', 'neumann')

    Returns:
        df : 导数数组 [N]
    """
    N = len(f)
    offsets, coeffs, divisor = get_stencil(order, derivative)
    half_width = max(abs(offsets))

    df = np.zeros_like(f)

    # 内部点
    for off, c in zip(offsets, coeffs):
        for i in range(half_width, N - half_width):
            df[i] += c * f[i + off]

    # 边界处理
    if bc == 'periodic':
        for off, c in zip(offsets, coeffs):
            for i in range(half_width):
                j = i + off
                if j < 0:
                    j += N
                elif j >= N:
                    j -= N
                df[i] += c * f[j]
            for i in range(N - half_width, N):
                j = i + off
                if j < 0:
                    j += N
                elif j >= N:
                    j -= N
                df[i] += c * f[j]
    elif bc == 'dirichlet':
        # 边界处降阶到一阶单侧差分或保持零
        df[0] = df[half_width]
        df[-1] = df[-half_width - 1]
    elif bc == 'neumann':
        df[0] = df[1]
        df[-1] = df[-2]
    else:
        raise ValueError(f"未知边界条件: {bc}")

    df /= (divisor * dx**derivative)
    return df


def build_band_matrix(N: int, offsets: np.ndarray, coeffs: np.ndarray,
                       divisor: float, dx: float, derivative: int,
                       bc: str = 'periodic') -> np.ndarray:
    """
    构造有限差分算子的带状矩阵表示 (基于 r8ncf 思想)

    返回 N x (2*half_width+1) 的带状矩阵, 中心列为对角元

    Parameters:
        N          : 网格点数
        offsets    : 偏移
        coeffs     : 系数
        divisor    : 除数
        dx         : 网格间距
        derivative : 导数阶数
        bc         : 边界条件
    """
    half_width = int(max(abs(offsets)))
    band_width = 2 * half_width + 1
    band = np.zeros((N, band_width))

    for off, c in zip(offsets, coeffs):
        col = half_width + int(off)
        factor = c / (divisor * dx**derivative)
        for i in range(N):
            band[i, col] = factor

    # 周期边界修正
    if bc == 'periodic':
        for off, c in zip(offsets, coeffs):
            factor = c / (divisor * dx**derivative)
            for i in range(half_width):
                j = i + int(off)
                if j < 0:
                    col = half_width + int(off)
                    band[i, half_width + (j - i)] = factor  # wrap
                elif j >= N:
                    pass  # wrap handled by periodicity

    return band


def band_matvec(band: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    带状矩阵-向量乘积 (John Burkardt r8ncf_mtv 思想)

    band : N x (2*w+1)
    x    : N
    """
    N = len(x)
    half_width = (band.shape[1] - 1) // 2
    result = np.zeros(N)
    for i in range(N):
        for j in range(-half_width, half_width + 1):
            col = half_width + j
            idx = i + j
            if 0 <= idx < N:
                result[i] += band[i, col] * x[idx]
            else:
                # 周期边界
                idx_p = idx % N
                result[i] += band[i, col] * x[idx_p]
    return result


def truncation_error_analysis(k: np.ndarray, dx: float, order: int,
                                derivative: int = 2) -> np.ndarray:
    """
    有限差分模板的截断误差分析

    对于 exp(ikx), 数值波数 k_num 与真实波数 k 的关系:
        k_num^d * dx^d = sum_j c_j * exp(i*j*k*dx) / divisor

    误差 = |k_num - k| / k
    """
    offsets, coeffs, divisor = get_stencil(order, derivative)

    if derivative == 1:
        # 数值波数
        k_num_complex = np.zeros_like(k, dtype=complex)
        for off, c in zip(offsets, coeffs):
            k_num_complex += c * np.exp(1j * off * k * dx)
        k_num_complex /= (divisor * dx)
        k_num = k_num_complex / (1j * k + 1e-300)
        return np.abs(k_num - 1.0)
    else:
        # 二阶导数
        factor_num = np.zeros_like(k, dtype=complex)
        for off, c in zip(offsets, coeffs):
            factor_num += c * np.exp(1j * off * k * dx)
        factor_num /= (divisor * dx**2)
        factor_exact = -(k**2) + 0j
        ratio = factor_num / (factor_exact + 1e-300)
        return np.abs(ratio - 1.0)


def modified_wavenumber(k: np.ndarray, dx: float, order: int,
                          derivative: int = 2) -> np.ndarray:
    """
    计算修正波数 (modified wavenumber)

    用于数值色散分析: 比较数值相速度与解析相速度
    """
    offsets, coeffs, divisor = get_stencil(order, derivative)

    if derivative == 1:
        k_mod = np.zeros_like(k, dtype=complex)
        for off, c in zip(offsets, coeffs):
            k_mod += c * np.exp(1j * off * k * dx)
        k_mod /= (divisor * dx * 1j)
        return k_mod
    else:
        k_mod_sq = np.zeros_like(k, dtype=complex)
        for off, c in zip(offsets, coeffs):
            k_mod_sq += c * np.exp(1j * off * k * dx)
        k_mod_sq /= (divisor * dx**2)
        return -k_mod_sq  # 返回 -k_mod^2 使为正


def spectral_content(kdx_max: float, order: int) -> float:
    """
    给定 k*dx 的最大值, 计算该阶数模板的谱含量
    即 |k_num/k| 在 kdx_max 处的值

    理想值 = 1.0, 偏离越大表示色散误差越大
    """
    k_test = np.array([kdx_max])
    dx_test = 1.0
    km = modified_wavenumber(k_test, dx_test, order, derivative=1)
    ratio = km / (k_test + 1e-300)
    return float(np.abs(ratio[0]))
