"""
带状矩阵泊松方程求解器

基于 John Burkardt 的 r8ncf (narrowband banded matrix) 存储思想,
实现 1D 泊松方程的高阶有限差分求解:
    d^2 phi / dx^2 = -rho / epsilon_0

采用 LU 分解或迭代法求解带状系统。
"""

import numpy as np
from typing import Optional, Tuple
try:
    from high_order_fd import get_stencil
except ImportError:
    from high_order_fd import get_stencil


def build_poisson_band_matrix(N: int, dx: float, fd_order: int = 2,
                                bc: str = 'periodic') -> np.ndarray:
    """
    构造泊松方程的带状矩阵 (负拉普拉斯算子):
        A * phi = -rho / epsilon_0

    对于 2阶中心差分:
        (-phi_{i-1} + 2*phi_i - phi_{i+1}) / dx^2 = -rho_i / epsilon_0

    Parameters:
        N        : 网格点数
        dx       : 网格间距
        fd_order : 有限差分阶数 (2, 4, 6, 8)
        bc       : 边界条件 ('periodic', 'dirichlet')

    Returns:
        A : N x N 矩阵 (稀疏但以稠密形式返回供小规模使用)
    """
    offsets, coeffs, divisor = get_stencil(fd_order, derivative=2)
    A = np.zeros((N, N))

    for i in range(N):
        for off, c in zip(offsets, coeffs):
            j = i + int(off)
            if bc == 'periodic':
                j = j % N
            elif bc == 'dirichlet':
                if j < 0 or j >= N:
                    continue  # Dirichlet: phi = 0 at boundary
            if 0 <= j < N:
                A[i, j] = c / (divisor * dx**2)

    return A


def build_poisson_band_narrow(N: int, dx: float, fd_order: int = 2) -> Tuple[np.ndarray, int]:
    """
    构造窄带存储格式的泊松矩阵 (r8ncf 风格)

    Returns:
        (band, half_width) : band 为 N x (2*half_width+1), 中心列为对角元
    """
    offsets, coeffs, divisor = get_stencil(fd_order, derivative=2)
    half_width = int(max(abs(offsets)))
    band_width = 2 * half_width + 1
    band = np.zeros((N, band_width))

    for i in range(N):
        for off, c in zip(offsets, coeffs):
            col = half_width + int(off)
            j = (i + int(off)) % N  # 周期边界
            band[i, col] = c / (divisor * dx**2)

    return band, half_width


def lu_solve_banded(band: np.ndarray, half_width: int, rhs: np.ndarray) -> np.ndarray:
    """
    带状矩阵 LU 分解求解 (简化版本)

    对于小规模系统, 使用稠密 LU 分解
    大规模时应使用迭代法或 FFT
    """
    N = len(rhs)
    # 还原稠密矩阵
    A = np.zeros((N, N))
    for i in range(N):
        for j_off in range(-half_width, half_width + 1):
            col = half_width + j_off
            j = (i + j_off) % N
            A[i, j] = band[i, col]
    return np.linalg.solve(A, rhs)


def solve_poisson_1d(rho: np.ndarray, dx: float, epsilon_0: float,
                      fd_order: int = 2, bc: str = 'periodic',
                      method: str = 'direct') -> np.ndarray:
    """
    求解 1D 泊松方程:
        d^2 phi / dx^2 = -rho / epsilon_0

    Parameters:
        rho       : 电荷密度 [N]
        dx        : 网格间距
        epsilon_0 : 真空介电常数
        fd_order  : 有限差分阶数
        bc        : 边界条件
        method    : 'direct' (LU) 或 'fft' (周期边界时)

    Returns:
        phi : 电势 [N]
    """
    N = len(rho)

    if method == 'fft' and bc == 'periodic':
        return solve_poisson_fft(rho, dx, epsilon_0, fd_order)

    # 直接法
    A = build_poisson_band_matrix(N, dx, fd_order, bc)

    # 处理零均值约束 (周期边界)
    rhs = -rho / epsilon_0
    if bc == 'periodic':
        rhs = rhs - np.mean(rhs)  # 保证可解性 (Fredholm alternative)
        # 添加零均值约束替换一行
        A[0, :] = 1.0 / N
        rhs[0] = 0.0

    phi = np.linalg.solve(A, rhs)
    return phi


def solve_poisson_fft(rho: np.ndarray, dx: float, epsilon_0: float,
                       fd_order: int = 2) -> np.ndarray:
    """
    使用 FFT 求解周期边界泊松方程 (谱方法)

    对于高阶有限差分, 使用修正波数:
        -k_mod^2 * phi_hat = -rho_hat / epsilon_0

    其中 k_mod 为有限差分的修正波数
    """
    from high_order_fd import modified_wavenumber

    N = len(rho)
    L = N * dx

    # 波数数组
    k_freq = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    k_mod_sq = modified_wavenumber(k_freq, dx, fd_order, derivative=2)

    # FFT of charge density
    rho_hat = np.fft.fft(rho)

    # 求解 (避免零频率奇点)
    phi_hat = np.zeros_like(rho_hat, dtype=complex)
    for i in range(N):
        if abs(k_mod_sq[i]) > 1e-14:
            phi_hat[i] = rho_hat[i] / (epsilon_0 * k_mod_sq[i])
        else:
            phi_hat[i] = 0.0  # 零均值约束

    # 逆 FFT
    phi = np.real(np.fft.ifft(phi_hat))
    return phi


def electric_field_from_potential(phi: np.ndarray, dx: float,
                                    fd_order: int = 2,
                                    bc: str = 'periodic') -> np.ndarray:
    """
    从电势计算电场:
        E = -d(phi)/dx

    使用负的一阶有限差分
    """
    from high_order_fd import apply_fd_1d
    E = -apply_fd_1d(phi, dx, order=fd_order, derivative=1, bc=bc)
    return E


def poisson_residual(phi: np.ndarray, rho: np.ndarray, dx: float,
                      epsilon_0: float, fd_order: int = 2,
                      bc: str = 'periodic') -> np.ndarray:
    """
    计算泊松方程残差:
        r = d^2 phi / dx^2 + rho / epsilon_0
    """
    from high_order_fd import apply_fd_1d
    laplacian_phi = apply_fd_1d(phi, dx, order=fd_order, derivative=2, bc=bc)
    return laplacian_phi + rho / epsilon_0


def energy_from_field(E: np.ndarray, dx: float, epsilon_0: float) -> float:
    """
    计算电场能量:
        W_E = (1/2) * epsilon_0 * integral(E^2 dx)
    """
    return 0.5 * epsilon_0 * np.sum(E**2) * dx


def jacobi_poisson(rhs: np.ndarray, dx: float, epsilon_0: float,
                     fd_order: int = 2, max_iter: int = 1000,
                     tol: float = 1e-10) -> np.ndarray:
    """
    Jacobi 迭代法求解泊松方程 (教学目的)

    对于高阶格式, 使用分裂迭代
    """
    N = len(rhs)
    phi = np.zeros(N)
    offsets, coeffs, divisor = get_stencil(fd_order, derivative=2)

    # 对角元
    diag_idx = np.where(offsets == 0)[0][0]
    diag_val = coeffs[diag_idx] / (divisor * dx**2)

    for iteration in range(max_iter):
        phi_old = phi.copy()
        for i in range(N):
            s = 0.0
            for off, c in zip(offsets, coeffs):
                if off == 0:
                    continue
                j = (i + int(off)) % N
                s += c * phi_old[j]
            phi[i] = (-rhs[i] / epsilon_0 - s) / diag_val / (divisor * dx**2)

        # 收敛检查
        res = np.max(np.abs(phi - phi_old))
        if res < tol:
            break

    return phi


def multigrid_v_cycle(rho: np.ndarray, dx: float, epsilon_0: float,
                        fd_order: int = 2, n_levels: int = 3,
                        n_smooth: int = 3) -> np.ndarray:
    """
    简化的 V-cycle 多重网格求解器 (2阶差分)

    仅用于演示多重网格思想, 高阶格式暂不支持
    """
    if fd_order != 2:
        # 高阶时退化为直接法
        return solve_poisson_1d(rho, dx, epsilon_0, fd_order=2, method='direct')

    N = len(rho)
    if N < 8 or n_levels <= 1:
        return solve_poisson_1d(rho, dx, epsilon_0, fd_order=2, method='direct')

    # 粗化
    rho_coarse = 0.5 * (rho[::2] + rho[1::2])
    dx_coarse = 2.0 * dx

    # 递归求解粗网格
    phi_coarse = multigrid_v_cycle(rho_coarse, dx_coarse, epsilon_0,
                                     fd_order, n_levels - 1, n_smooth)

    # 插值回细网格
    phi = np.zeros(N)
    for i in range(len(phi_coarse)):
        phi[2*i] = phi_coarse[i]
        phi[2*i + 1] = phi_coarse[i]

    # 光滑修正
    for _ in range(n_smooth):
        phi = jacobi_poisson(rho, dx, epsilon_0, fd_order=2, max_iter=1)

    return phi
