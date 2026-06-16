"""
brillouin_quadrature.py — Brillouin 区积分与高精度求积
========================================================

本模块实现 Brillouin 区 (BZ) 上各种积分的高精度数值求积方案,
用于计算拓扑不变量 (Chern 数, Z₂), 态密度, 光学响应等。

核心方法:
1. **等距网格 + 周期修正**: 最简单但精度有限
2. **Gauss-Legendre 求积**: 高精度, 适合光滑被积函数
3. **Monkhorst-Pack 网格**: 凝聚态物理标准方法
4. **精确单项式积分**: 借鉴 disk/pyramid 积分的 Gamma 函数公式

数学公式
--------
**BZ 积分的一般形式:**
    I = (1/Ω_BZ) ∫_{BZ} f(k) d²k

其中 Ω_BZ = (2π)²/A 是 BZ 面积, A 是实空间原胞面积。

**Monkhorst-Pack 网格:**
    k_i = (2i - N - 1) / (2N) × G,  i = 1, ..., N
其中 G 是倒格矢。

**Gauss-Legendre 求积:**
    ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
对 2n-1 次多项式精确。

**二维 BZ 上多项式精确积分:**
借鉴 299_disk01_integrals 和 930_pyramid_exactness:
    ∫_{BZ} kx^a ky^b dkx dky

对正方 BZ [-π/a, π/a]²:
    = ∫_{-π/a}^{π/a} kx^a dkx × ∫_{-π/a}^{π/a} ky^b dky
    = [kx^{a+1}/(a+1)]_{-π/a}^{π/a} × [ky^{b+1}/(b+1)]_{-π/a}^{π/a}

当 a 或 b 为奇数时, 积分为零 (对称性)。

来源映射
--------
- 299_disk01_integrals: 精确单项式积分 (Gamma 函数)
- 930_pyramid_exactness: 多项式精确度检验框架
- 1345_triangulation_plot: BZ 三角剖分 (数据结构)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple, Dict, List, Callable, Optional


def monkhorst_pack_grid(N: int, shift: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    生成 Monkhorst-Pack k 点网格。

    k_i = (2i - N - 1 + 2×shift) / (2N) × 2π/a,  i = 1, ..., N

    对 shift = 0.5, 即 Γ-中心网格。
    对 shift = 0, 即标准 MP 网格 (不包含 Γ 点, 当 N 为偶数)。

    Parameters
    ----------
    N : int
        每维格点数
    shift : float
        偏移 (0 或 0.5)

    Returns
    -------
    kx, ky : ndarray, shape (N,)
    weight : float
        每个 k 点的权重 (归一化)
    """
    indices = np.arange(1, N + 1)
    k_1d = (2 * indices - N - 1 + 2 * shift) / (2 * N) * 2 * np.pi
    weight = 1.0 / (N * N)
    return k_1d, k_1d, weight


def gauss_legendre_2d(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D Gauss-Legendre 求积网格 (映射到 BZ [-π, π]²)。

    1D Gauss-Legendre: 在 [-1, 1] 上有 n 个点 x_i 和权重 w_i,
    对 2n-1 次多项式精确。

    映射到 [-π, π]: k = πx, dk = π dx
    2D: 权重 = π² w_i w_j

    Parameters
    ----------
    N : int
        每维 Gauss 点数

    Returns
    -------
    kx, ky : ndarray, shape (N,)
    weights : ndarray, shape (N, N)
    """
    x_1d, w_1d = leggauss(N)
    k_1d = np.pi * x_1d  # 映射到 [-π, π]
    w_mapped = np.pi * w_1d  # 权重乘以 π

    # 2D 外积
    weights_2d = np.outer(w_mapped, w_mapped)

    return k_1d, k_1d, weights_2d


def triangulate_bz(N: int) -> Dict:
    """
    将正方 BZ [-π, π]² 三角剖分。

    每个方格分为两个三角形, 用于自适应积分和 Berry 相位计算。

    借鉴 1345_triangulation_plot 的三角剖分数据结构。

    Parameters
    ----------
    N : int
        每维格点数

    Returns
    -------
    result : dict
        'nodes': shape ((N+1)², 2)
        'triangles': shape (2N², 3) — 三角形顶点索引
        'areas': shape (2N²,)
    """
    kx = np.linspace(-np.pi, np.pi, N + 1)
    ky = np.linspace(-np.pi, np.pi, N + 1)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')

    nodes = np.column_stack([kxx.ravel(), kyy.ravel()])

    triangles = []
    for i in range(N):
        for j in range(N):
            # 方格的四个角
            n00 = i * (N + 1) + j
            n10 = (i + 1) * (N + 1) + j
            n01 = i * (N + 1) + (j + 1)
            n11 = (i + 1) * (N + 1) + (j + 1)

            # 两个三角形
            triangles.append([n00, n10, n01])
            triangles.append([n10, n11, n01])

    triangles = np.array(triangles)

    # 计算三角形面积
    dk = 2 * np.pi / N
    areas = np.full(len(triangles), 0.5 * dk * dk)

    return {
        'nodes': nodes,
        'triangles': triangles,
        'areas': areas,
        'N': N,
        'n_nodes': len(nodes),
        'n_triangles': len(triangles)
    }


def exact_monomial_integral_2d(a: int, b: int,
                               Lx: float = np.pi,
                               Ly: float = np.pi) -> float:
    """
    计算 BZ 上单项式 kx^a × ky^b 的精确积分。

    ∫_{-Lx}^{Lx} ∫_{-Ly}^{Ly} kx^a ky^b dkx dky

    = [kx^{a+1}/(a+1)]_{-Lx}^{Lx} × [ky^{b+1}/(b+1)]_{-Ly}^{Ly}

    当 a 为奇数: ∫_{-L}^{L} kx^a dkx = 0 (对称性)
    当 a 为偶数: ∫_{-L}^{L} kx^a dkx = 2 L^{a+1}/(a+1)

    借鉴 299_disk01_integrals 的 Gamma 函数方法:
    对圆盘上的积分, 使用:
        ∫_{D} x^{2a} y^{2b} dxdy = 2 Γ(a+1/2) Γ(b+1/2) / Γ(a+b+1) × R^{2(a+b+1)} / (2(a+b+1))

    对 BZ (正方), 可以分解为两个 1D 积分的乘积。

    Parameters
    ----------
    a, b : int
        指数
    Lx, Ly : float
        半区间长度

    Returns
    -------
    I : float
    """
    def integral_1d(n: int, L: float) -> float:
        if n % 2 == 1:
            return 0.0
        return 2.0 * L ** (n + 1) / (n + 1)

    return integral_1d(a, Lx) * integral_1d(b, Ly)


def quadrature_exactness_test(N: int, max_degree: int = 6,
                              method: str = 'gauss_legendre'
                              ) -> Dict:
    """
    测试求积规则的多项式精确度。

    借鉴 930_pyramid_exactness 的精确度检验框架:
    对每个总次数 d = 0, ..., max_degree, 枚举所有单项式 kx^a ky^b (a+b=d),
    比较数值积分与精确积分。

    Gauss-Legendre N 点: 精确到 2N-1 次。
    Monkhorst-Pack N 点: 精确到 2N-1 次 (对周期函数指数精确)。

    Parameters
    ----------
    N : int
        每维格点数
    max_degree : int
        最大多项式次数
    method : str
        'gauss_legendre' 或 'monkhorst_pack'

    Returns
    -------
    result : dict
    """
    if method == 'gauss_legendre':
        kx, ky, weights = gauss_legendre_2d(N)
    elif method == 'monkhorst_pack':
        kx, ky, w = monkhorst_pack_grid(N)
        weights = np.full((N, N), w)
    else:
        raise ValueError(f"未知方法: {method}")

    results = {}
    exact_up_to = -1

    for total_deg in range(max_degree + 1):
        max_error = 0.0
        for a in range(total_deg + 1):
            b = total_deg - a
            # 精确值
            exact = exact_monomial_integral_2d(a, b)

            # 数值积分
            numerical = 0.0
            for i in range(len(kx)):
                for j in range(len(ky)):
                    f_val = kx[i] ** a * ky[j] ** b
                    numerical += weights[i, j] * f_val

            error = abs(numerical - exact)
            max_error = max(max_error, error)

        results[total_deg] = {
            'max_error': float(max_error),
            'exact': max_error < 1e-8
        }
        if max_error < 1e-8:
            exact_up_to = total_deg

    expected = 2 * N - 1 if method == 'gauss_legendre' else 2 * N - 1

    return {
        'method': method,
        'N': N,
        'expected_exact_up_to': expected,
        'actual_exact_up_to': exact_up_to,
        'degree_results': results,
        'passed': exact_up_to >= expected
    }


def integrate_bz(func: Callable, N: int = 30,
                 method: str = 'gauss_legendre') -> Tuple[float, float]:
    """
    在 BZ 上积分一般函数 f(kx, ky)。

    Parameters
    ----------
    func : callable
        接受 (kx, ky) 返回标量
    N : int
    method : str

    Returns
    -------
    integral : float
    error_estimate : float
    """
    if method == 'gauss_legendre':
        kx, ky, weights = gauss_legendre_2d(N)
    elif method == 'monkhorst_pack':
        kx, ky, w = monkhorst_pack_grid(N)
        weights = np.full((N, N), w)
    else:
        raise ValueError(f"未知方法: {method}")

    integral = 0.0
    for i in range(len(kx)):
        for j in range(len(ky)):
            integral += weights[i, j] * func(kx[i], ky[j])

    # 误差估计: 比较 N 和 N//2 的结果
    if N >= 10:
        N_half = N // 2
        if method == 'gauss_legendre':
            kx2, ky2, w2 = gauss_legendre_2d(N_half)
        else:
            kx2, ky2, w2 = monkhorst_pack_grid(N_half)
            w2 = np.full((N_half, N_half), w2)

        integral_half = 0.0
        for i in range(len(kx2)):
            for j in range(len(ky2)):
                integral_half += w2[i, j] * func(kx2[i], ky2[j])

        error_estimate = abs(integral - integral_half)
    else:
        error_estimate = np.nan

    return integral, error_estimate


def brillouin_integration_report(N_values: List[int] = None) -> str:
    """生成 BZ 积分精度测试报告。"""
    if N_values is None:
        N_values = [5, 10, 15, 20, 30]

    lines = []
    lines.append("=" * 60)
    lines.append("Brillouin 区求积精度测试报告")
    lines.append("=" * 60)

    for method in ['gauss_legendre', 'monkhorst_pack']:
        lines.append(f"\n--- 方法: {method} ---")
        for N in N_values:
            result = quadrature_exactness_test(N, max_degree=8, method=method)
            lines.append(f"  N={N:2d}: "
                         f"精确到 {result['actual_exact_up_to']} 次 "
                         f"(期望 {result['expected_exact_up_to']})"
                         f" {'✓' if result['passed'] else '✗'}")

    # 示例积分
    lines.append(f"\n--- 示例积分: ∫∫ (kx²+ky²) dkx dky ---")
    exact_val = exact_monomial_integral_2d(2, 0) + exact_monomial_integral_2d(0, 2)
    lines.append(f"  精确值: {exact_val:.10f}")

    for N in [10, 20, 30]:
        val_gl, err_gl = integrate_bz(lambda kx, ky: kx**2 + ky**2, N, 'gauss_legendre')
        val_mp, err_mp = integrate_bz(lambda kx, ky: kx**2 + ky**2, N, 'monkhorst_pack')
        lines.append(f"  N={N:2d}: GL={val_gl:.10f} (err={err_gl:.2e}), "
                     f"MP={val_mp:.10f} (err={err_mp:.2e})")

    return "\n".join(lines)
