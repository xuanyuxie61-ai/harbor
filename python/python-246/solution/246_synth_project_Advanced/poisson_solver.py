"""
poisson_solver.py  —  宇宙学 Poisson 方程求解器
==============================================

科学来源种子:
  - 367_fd2d_heat_steady / fd2d_heat_steady.m, interior.m, boundary.m
    将其 2D 稳态热传导 FD 离散化思路扩展到 3D 周期 Poisson:
        ∇² Φ = S
    其中 interior.m 的 5 点模板被替换为高阶 FD (见 finite_difference.py),
    boundary.m 的边界注入被替换为周期 FFT 对角化。
  - 979_r8gb / r8gb_fa.m, r8gb_sl.m
    为 1D 验证路径提供带状矩阵直接求解; 并用于构造离散 Laplace
    算子以估计条件数。

物理背景:
  共动坐标 Poisson 方程:
      ∇² Φ(x,t) = 4π G ρ̄ a² δ(x,t)
                = (3/2) H0² Ω_m a^{-1} δ(x,t)
  周期边界条件 → 用 FFT 精确求逆:
      Φ(k) = - S(k) / |k|²    (k ≠ 0)
      Φ(k=0) = 0               (约定平均势为零)
  对 k=0 零模的处理来自质量守恒: ∫ δ d³x = 0 (周期域)。
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from finite_difference import laplacian_3d_periodic, r8gb_fa_python, r8gb_sl_python


# ---------------------------------------------------------------------------- #
#                         3D 周期 Poisson FFT 求解器
# ---------------------------------------------------------------------------- #
def poisson_fft_3d(source: NDArray, box_length: float) -> NDArray:
    """
    求解周期域 [0, L]³ 上的 Poisson 方程:
        ∇² Φ = source
    使用 FFT:
        Φ(k) = source(k) / ( -|k|² )
    k = (2π/L) (n_x, n_y, n_z),  n_i ∈ [-N/2, N/2-1]

    Parameters
    ----------
    source : (N, N, N) array
        右端项 (4πG ρ̄ a² δ)
    box_length : float
        盒子边长 L [Mpc/h]

    Returns
    -------
    phi : (N, N, N) array
        引力势,平均值为零
    """
    if source.ndim != 3:
        raise ValueError("source 必须为 3 维数组")
    n = source.shape[0]
    if source.shape[1] != n or source.shape[2] != n:
        raise ValueError("仅支持立方网格")
    # 波数:
    kfreq = np.fft.fftfreq(n, d=1.0 / n) * (2.0 * np.pi / box_length)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    # FFT:
    s_hat = np.fft.fftn(source)
    # 求逆 (处理 k=0):
    phi_hat = np.zeros_like(s_hat)
    nz = k2 > 1e-12
    phi_hat[nz] = -s_hat[nz] / k2[nz]
    phi_hat[~nz] = 0.0
    phi = np.real(np.fft.ifftn(phi_hat))
    # 减去平均值 (规范选择):
    phi -= phi.mean()
    return phi


# ---------------------------------------------------------------------------- #
#                   离散 Laplace 算子 (1D 带存储, 验证用)
# ---------------------------------------------------------------------------- #
def build_laplacian_1d_banded(n: int, h: float, p: int = 1) -> NDArray:
    """
    构造 1D 周期 2p 阶 Laplace 算子的带状存储 (r8gb 格式)。
    返回形状 (2*ml+mu+1, n) 的数组,ml = mu = p。
    内部调用 fd_stencil_2nd 获取模板。
    """
    from finite_difference import fd_stencil_2nd
    stencil = fd_stencil_2nd(p)
    ml = p
    mu = p
    m = 2 * ml + mu + 1
    a = np.zeros((m, n))
    for j in range(n):
        for diag in range(m):
            i_row = diag + j - ml - mu - 1
            # 模板索引: stencil[s],  s = i_row - j (相对位置)
            s = i_row - j
            if -p <= s <= p:
                a[diag, j] = stencil[s + p] / (h * h)
    return a


def condition_number_estimate_1d(n: int, h: float, p: int = 1) -> float:
    """
    估计 1D 周期 Laplace 算子的谱条件数 (忽略零模):
        κ = |λ_max| / |λ_min_nonzero|
    其中 λ_k = -4/h² sin²(π k / N) (2 阶); 对高阶类似但有修正。
    """
    k = np.arange(1, n // 2 + 1)
    # 2 阶解析 (基准):
    lam = -4.0 / (h * h) * np.sin(np.pi * k / n) ** 2
    if p > 1:
        from finite_difference import fd_stencil_2nd
        s = fd_stencil_2nd(p)
        # 符号:
        lam = np.zeros_like(k, dtype=float)
        for idx, ki in enumerate(k):
            val = 0.0 + 0j
            for m in range(-p, p + 1):
                val += s[m + p] * np.exp(1j * 2.0 * np.pi * m * ki / n)
            lam[idx] = np.real(val) / (h * h)
    lam_nz = np.abs(lam[lam != 0])
    if lam_nz.size == 0:
        return float("inf")
    return float(lam_nz.max() / lam_nz.min())


# ---------------------------------------------------------------------------- #
#                1D 带状直接求解 (验证路径)
# ---------------------------------------------------------------------------- #
def poisson_1d_banded(source: NDArray, h: float, p: int = 1) -> NDArray:
    """
    1D 非周期 Poisson 方程 (Dirichlet BC Φ=0):
        -Φ'' = -source    即  Φ'' = source
    用带状 LU 分解求解 (来自 r8gb_fa + r8gb_sl)。
    """
    n = source.size
    a = build_laplacian_1d_banded(n, h, p)
    # 添加 Dirichlet 边界 (第一行和最后一行):
    ml = p
    mu = p
    m = 2 * ml + mu + 1
    # 首行:  Φ_0 = 0 → 令 a[m-1, 0] = 1, 其他清零
    for diag in range(m):
        if diag != m - 1:
            a[diag, 0] = 0.0
    a[m - 1, 0] = 1.0
    # 末行:  Φ_{n-1} = 0
    for diag in range(m):
        if diag != m - 1:
            a[diag, n - 1] = 0.0
    a[m - 1, n - 1] = 1.0
    rhs = source.copy()
    rhs[0] = 0.0
    rhs[-1] = 0.0
    alu, pivot, info = r8gb_fa_python(n, ml, mu, a)
    if info != 0:
        raise RuntimeError(f"带状 LU 分解在第 {info} 步失败")
    phi = r8gb_sl_python(n, ml, mu, alu, pivot, rhs, job=0)
    return phi


# ---------------------------------------------------------------------------- #
#                 多网格粗糙校正 (可选, 提升精度)
# ---------------------------------------------------------------------------- #
def poisson_v_cycle_3d(source: NDArray, box_length: float,
                       n_levels: int = 2, n_smooth: int = 3) -> NDArray:
    """
    简化的 V-cycle 多网格 Poisson 求解器。
    在 3D 周期域上,使用 Gauss-Seidel 光滑与粗糙网格修正。
    用于验证 FFT 解并估计多网格收敛率。

    Parameters
    ----------
    source : (N, N, N) array
    box_length : float
    n_levels : int  V-cycle 层数
    n_smooth : int 每层光滑迭代次数

    Returns
    -------
    phi : (N, N, N) array  近似解
    """
    n = source.shape[0]
    h = box_length / n
    phi = np.zeros_like(source)
    # 仅做 1 个 V-cycle:
    for _ in range(1):
        phi = _v_cycle_recursive(source, phi, h, n_smooth, level=0,
                                 max_level=n_levels)
    return phi


def _v_cycle_recursive(source: NDArray, phi: NDArray, h: float,
                       n_smooth: int, level: int, max_level: int) -> NDArray:
    """递归 V-cycle。"""
    n = source.shape[0]
    # 光滑:
    for _ in range(n_smooth):
        phi = _gauss_seidel_step(phi, source, h)
    if level >= max_level or n <= 4:
        return phi
    # 残差:
    residual = source - laplacian_3d_periodic(phi, h, p=1)
    # 限制到粗糙网格 (全加权):
    n_c = n // 2
    r_c = np.zeros((n_c, n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            for k in range(n_c):
                r_c[i, j, k] = (
                    residual[2 * i, 2 * j, 2 * k]
                    + residual[2 * i + 1, 2 * j, 2 * k]
                    + residual[2 * i, 2 * j + 1, 2 * k]
                    + residual[2 * i + 1, 2 * j + 1, 2 * k]
                    + residual[2 * i, 2 * j, 2 * k + 1]
                    + residual[2 * i + 1, 2 * j, 2 * k + 1]
                    + residual[2 * i, 2 * j + 1, 2 * k + 1]
                    + residual[2 * i + 1, 2 * j + 1, 2 * k + 1]
                ) / 8.0
    h_c = h * 2.0
    e_c = np.zeros_like(r_c)
    e_c = _v_cycle_recursive(r_c, e_c, h_c, n_smooth, level + 1, max_level)
    # 延拓 (三线性):
    e_f = np.zeros_like(phi)
    for i in range(n_c):
        for j in range(n_c):
            for k in range(n_c):
                e_f[2 * i, 2 * j, 2 * k] += e_c[i, j, k]
                e_f[2 * i + 1, 2 * j, 2 * k] += e_c[i, j, k]
                e_f[2 * i, 2 * j + 1, 2 * k] += e_c[i, j, k]
                e_f[2 * i + 1, 2 * j + 1, 2 * k] += e_c[i, j, k]
                e_f[2 * i, 2 * j, 2 * k + 1] += e_c[i, j, k]
                e_f[2 * i + 1, 2 * j, 2 * k + 1] += e_c[i, j, k]
                e_f[2 * i, 2 * j + 1, 2 * k + 1] += e_c[i, j, k]
                e_f[2 * i + 1, 2 * j + 1, 2 * k + 1] += e_c[i, j, k]
    phi = phi + e_f
    # 再次光滑:
    for _ in range(n_smooth):
        phi = _gauss_seidel_step(phi, source, h)
    return phi


def _gauss_seidel_step(phi: NDArray, source: NDArray, h: float) -> NDArray:
    """红黑 Gauss-Seidel 一步。"""
    n = phi.shape[0]
    phi_new = phi.copy()
    inv_diag = -h * h / 6.0  # 3D 5 点 (实际 7 点) 对角元
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if (i + j + k) % 2 == 0:
                    neigh = (
                        phi[(i + 1) % n, j, k] + phi[(i - 1) % n, j, k]
                        + phi[i, (j + 1) % n, k] + phi[i, (j - 1) % n, k]
                        + phi[i, j, (k + 1) % n] + phi[i, j, (k - 1) % n]
                    )
                    phi_new[i, j, k] = (neigh - h * h * source[i, j, k]) / 6.0
    return phi_new
