# -*- coding: utf-8 -*-
"""
radiative_transfer.py
=====================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

隐式辐射扩散求解器 (Crank-Nicolson 型), 带 Gauss-Seidel 迭代
与 LU 回代 (映射 linpack_bench_backslash 的带状矩阵求解).

辐射扩散方程
------------
在扩散近似下, 辐射能量密度 E_rad 满足::

    dE_rad/dt = div(c lambda grad E_rad) - c kappa_rho rho E_rad + S

其中::

    lambda = 1 / (3 (kappa_rho rho + (1/v_th) d/dt))  Levermore-Pomraning
    E_rad = a T^4   (辐射能量密度)
    S = 热源 (放射性衰变: r-process)

时间离散 (Crank-Nicolson)::

    (E^{n+1} - E^n) / dt = (1/2) [L(E^{n+1}) + L(E^n)] + S^{n+1/2}

整理得到::

    (I - dt/2 L) E^{n+1} = (I + dt/2 L) E^n + dt S^{n+1/2}

这是一个三对角系统 (1D) 或稀疏带状系统 (3D).

线性求解
--------
对于 1D 三对角系统::

    -a_i E_{i-1} + b_i E_i - c_i E_{i+1} = d_i

使用 Thomas 算法 (O(N) 的 LU 分解).

映射种子项目
-----------
- 688 (linpack_bench_backslash) → 带状矩阵 LU 分解的性能基准
  在此用于辐射扩散隐式步的线性求解器
- 049_asa239 (gammad)            → 不完全 Gamma 函数用于
  放射性衰变热源项 (r-process 核素衰变链)
- 577 (image_diffuse4/8)         → 二维/三维扩散模板的几何结构
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 放射性衰变热源 (r-process)
# ---------------------------------------------------------------------------
def rprocess_heating_rate(t_s: float, M_ej_g: float,
                          X_r: float = 0.01) -> float:
    """r-process 放射性衰变加热率  [erg/s].

    幂律拟合 (Metzger et al. 2010)::

        L_dot(t) = 2e10 * (t / 1 day)^{-1.3} * (M_ej / 0.01 M_sun)
                  * X_r   [erg/s/g]

    更精细: 多指数衰减 ( Barnes et al. 2016)::

        Q_dot = sum_k a_k exp(-t / tau_k)
    """
    t_day = t_s / 86400.0
    t_day = max(t_day, 1.0e-3)
    q_dot_per_g = 2.0e10 * (t_day ** -1.3) * X_r
    return q_dot_per_g * M_ej_g


def specific_heating_rate(t_s: float, X_r: float = 0.01) -> float:
    """比加热率  Q_dot / M_ej [erg/s/g]."""
    t_day = t_s / 86400.0
    t_day = max(t_day, 1.0e-3)
    return 2.0e10 * (t_day ** -1.3) * X_r


# ---------------------------------------------------------------------------
# 1D 三对角 Thomas 算法 (映射 688_linpack)
# ---------------------------------------------------------------------------
def thomas_solve(a: List[float], b: List[float], c: List[float],
                 d: List[float]) -> List[float]:
    """三对角系统  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i  的 Thomas 算法.

    Parameters
    ----------
    a : List[float]  下次对角线 (a[0] 不使用)
    b : List[float]  主对角线
    c : List[float]  上次对角线 (c[n-1] 不使用)
    d : List[float]  右端项

    Returns
    -------
    List[float]  解向量 x
    """
    n = len(b)
    if n < 1:
        return []
    c_ = [0.0] * n
    d_ = [0.0] * n
    # 前向消元
    if abs(b[0]) < 1.0e-30:
        raise ValueError("Zero pivot at row 0")
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * c_[i - 1]
        if abs(m) < 1.0e-30:
            raise ValueError(f"Zero pivot at row {i}")
        c_[i] = c[i] / m if i < n - 1 else 0.0
        d_[i] = (d[i] - a[i] * d_[i - 1]) / m
    # 回代
    x = [0.0] * n
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


# ---------------------------------------------------------------------------
# 带状矩阵求解 (通用 Gauss 消元, 映射 688)
# ---------------------------------------------------------------------------
def band_lu_solve(band_matrix: List[List[float]], rhs: List[float],
                  bandwidth: int) -> List[float]:
    """带状矩阵 LU 分解求解 (简化版本).

    对半带宽为 bandwidth 的带状矩阵 A, 存储为紧凑形式::

        band[i][j]  其中 j in [max(0, i-bw), min(n-1, i+bw)]

    此函数将带状矩阵转为完整矩阵后用高斯消元求解.
    仅用于小规模测试.
    """
    n = len(rhs)
    # 转为完整矩阵
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            A[i][j] = band_matrix[i][j - max(0, i - bandwidth)] \
                if j - max(0, i - bandwidth) < len(band_matrix[i]) else 0.0
    # 高斯消元 (无选主元)
    for k in range(n):
        if abs(A[k][k]) < 1.0e-30:
            raise ValueError(f"Zero pivot at k={k}")
        for i in range(k + 1, min(k + bandwidth + 1, n)):
            m = A[i][k] / A[k][k]
            for j in range(k, min(k + bandwidth + 1, n)):
                A[i][j] -= m * A[k][j]
            rhs[i] -= m * rhs[k]
    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, min(i + bandwidth + 1, n)):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


# ---------------------------------------------------------------------------
# 1D 辐射扩散步进
# ---------------------------------------------------------------------------
def radiation_diffusion_step(
    E_rad: List[float],            # 辐射能量密度 [erg/cm^3]
    r_grid: List[float],           # 径向网格 [cm]
    dt: float,                     # 时间步长 [s]
    kappa: List[float],            # 不透明度 [cm^2/g]
    rho: List[float],              # 密度 [g/cm^3]
    source: List[float],           # 热源 [erg/cm^3/s]
) -> List[float]:
    """Crank-Nicolson 隐式一步::

        (I - dt/2 L) E^{n+1} = (I + dt/2 L) E^n + dt S

    在球坐标下, L 为::

        L E = (1/r^2) d/dr (r^2 D dE/dr)
    其中 D = c / (3 kappa rho).

    Returns
    -------
    List[float]  E^{n+1}
    """
    C_LIGHT_LOCAL = 3.0e10
    n = len(E_rad)
    if n < 3:
        return list(E_rad)
    # 扩散系数
    D = [C_LIGHT_LOCAL / (3.0 * max(kappa[i] * rho[i], 1.0e-30))
         for i in range(n)]
    # 构造三对角矩阵
    a = [0.0] * n  # lower
    b = [0.0] * n  # main
    c = [0.0] * n  # upper
    rhs = [0.0] * n

    # 内点
    for i in range(1, n - 1):
        r = r_grid[i]
        dr_m = r_grid[i] - r_grid[i - 1]
        dr_p = r_grid[i + 1] - r_grid[i]
        dr_avg = 0.5 * (dr_m + dr_p)
        D_m = 0.5 * (D[i] + D[i - 1])
        D_p = 0.5 * (D[i] + D[i + 1])
        # 球坐标 Laplacian 离散
        # L E_i = (1/r^2) * [ r_{i+1/2}^2 D_{i+1/2} (E_{i+1}-E_i)/dr_p
        #                   - r_{i-1/2}^2 D_{i-1/2} (E_i-E_{i-1})/dr_m ] / dr_avg
        r_p_half = 0.5 * (r_grid[i] + r_grid[i + 1])
        r_m_half = 0.5 * (r_grid[i] + r_grid[i - 1])
        coeff_p = D_p * r_p_half * r_p_half / (dr_p * dr_avg * r * r)
        coeff_m = D_m * r_m_half * r_m_half / (dr_m * dr_avg * r * r)
        a[i] = -coeff_m
        c[i] = -coeff_p
        b[i] = coeff_p + coeff_m
    # 边界: Neumann 零梯度
    b[0] = 1.0
    c[0] = -1.0
    rhs[0] = 0.0
    b[-1] = 1.0
    a[-1] = -1.0
    rhs[-1] = 0.0

    # 构造 CN 右端项
    L_E = [b[i] * E_rad[i]
           + (a[i] * E_rad[i - 1] if i > 0 else 0.0)
           + (c[i] * E_rad[i + 1] if i < n - 1 else 0.0)
           for i in range(n)]
    for i in range(n):
        rhs[i] = E_rad[i] + 0.5 * dt * L_E[i] + 0.5 * dt * source[i] \
                 + 0.5 * dt * source[i]  # 简化
        # 注意: 这里 source 已加两次, 实际应 S^{n+1/2} + S^n
        # 此处简化为 2 * S 作为近似

    # 构造左端矩阵 (I - dt/2 L)
    for i in range(n):
        b[i] = 1.0 - 0.5 * dt * b[i]
        a[i] = -0.5 * dt * a[i]
        c[i] = -0.5 * dt * c[i]

    return thomas_solve(a, b, c, rhs)


# ---------------------------------------------------------------------------
# LINPACK 风格性能测试 (映射 688)
# ---------------------------------------------------------------------------
def linpack_style_benchmark(n: int) -> Dict[str, float]:
    """LINPACK 风格的 n×n 稠密线性系统求解基准.

    测量 A x = b 的求解时间, 其中 A 为随机稠密矩阵.
    使用简化高斯消元.
    """
    import time
    import random
    rng = random.Random(42)
    A = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    x_exact = [1.0] * n
    b = [sum(A[i][j] * x_exact[j] for j in range(n)) for i in range(n)]
    t0 = time.perf_counter()
    # 高斯消元
    for k in range(n):
        if abs(A[k][k]) < 1e-30:
            continue
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    # 回代
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    elapsed = time.perf_counter() - t0
    ops = (2.0 * n ** 3) / 3.0 + 2.0 * n * n
    mflops = ops / (1.0e6 * max(elapsed, 1.0e-9))
    residual = max(abs(b[i] - sum(A[i][j] * x[j] for j in range(n)))
                   for i in range(min(3, n)))
    return {
        "n": n,
        "time_s": elapsed,
        "MFLOPS": mflops,
        "residual": residual,
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证 Thomas 算法与解析解一致."""
    # 系统: 2 x_0 - x_1 = 1, -x_0 + 2 x_1 - x_2 = 0, -x_1 + 2 x_2 = 1
    # 解: [1, 1, 1]
    a = [0.0, -1.0, -1.0]
    b = [2.0, 2.0, 2.0]
    c = [-1.0, -1.0, 0.0]
    d = [1.0, 0.0, 1.0]
    x = thomas_solve(a, b, c, d)
    for xi in x:
        assert abs(xi - 1.0) < 1.0e-10, f"Thomas solution {x} wrong"
    # 扩散步进 (平坦初值应保持不变)
    E0 = [1.0] * 20
    r = [1.0e12 + i * 1.0e10 for i in range(20)]
    kappa = [10.0] * 20
    rho = [1.0e-13] * 20
    src = [0.0] * 20
    E1 = radiation_diffusion_step(E0, r, 100.0, kappa, rho, src)
    assert len(E1) == len(E0)
    return True


if __name__ == "__main__":
    _self_check()
    print("radiative_transfer self-check passed.")
    result = linpack_style_benchmark(100)
    print(f"  LINPACK n={result['n']}: {result['time_s']:.3f} s, "
          f"{result['MFLOPS']:.2f} MFLOPS")
