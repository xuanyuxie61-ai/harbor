"""
fast_spectral_quadrature.py

基于 939_quad_fast_rule 核心算法的快速谱求积模块。

原项目提供了 Fejér 1 型快速积分和 Gauss-Legendre 快速积分，
利用 FFT 和对称三对角矩阵特征值分解实现高效数值积分。

在本气候归因框架中，该模块用于：
1. 大气垂直剖面上的水汽、温度积分
2. 辐射传输方程中的角度积分
3. 快速计算柱平均物理量

核心公式：
- Fejér 1 型规则（基于 DCT/FFT）：
    x_k = cos( π * (2k-1) / (2n) ),  k=1,...,n
    权重 w 通过 IFFT 快速构造：
        v_0 = [ 2*exp(iπk/n)/(1-4k^2), 0 ]_{k=0}^{m-1}
        v_1 = v_0[:-1] + conj(v_0[-1:0:-1])
        w = ifft(v_1)
    积分：Q = w^T f(x)

- Gauss-Legendre 规则（基于对称三对角特征值分解）：
    β_k = 0.5 / sqrt(1 - (2k)^{-2})
    T = tridiag(β, 0, β)
    [eigval, eigvec] = eig(T)
    x = diag(eigvec)（Legendre 点）
    w = 2 * eigval[0,:]^2（权重）
    积分：Q = w * f(x)

- 柱积分（大气总水汽含量）：
    TCW = ∫_0^{z_top} ρ(z) q(z) dz
    其中 ρ 为空气密度，q 为比湿。
"""

import numpy as np


def fejer1_integrate_fast(f, n):
    """
    Fejér 1 型快速积分（基于 939 的 fejer1_integrate_fast）。

    积分区间 [-1, 1]。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    N = np.arange(1, n, 2)
    L = len(N)
    m = n - L
    k = np.arange(m)
    temp1 = np.exp(1j * np.pi * k / n)
    temp2 = np.zeros(L + 1)
    v0 = np.concatenate([2.0 * temp1 / (1.0 - 4.0 * k ** 2), temp2])
    v1 = v0[:-1] + np.conj(v0[:0:-1])
    w = np.real(np.fft.ifft(v1))

    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
    fx = f(x)
    quad = float(np.dot(w, fx))
    return quad


def gauss_legendre_integrate_fast(f, n):
    """
    Gauss-Legendre 快速积分（基于 939 的 gauss_legendre_integrate_fast）。

    积分区间 [-1, 1]。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, n + 1)) ** (-2))
    # 构造对称三对角矩阵
    tridiag = np.diag(beta, 1) + np.diag(beta, -1)
    eigval, eigvec = np.linalg.eigh(tridiag)
    x = eigvec[0, :]
    idx = np.argsort(x)
    x = x[idx]
    w = 2.0 * eigval[0, idx] ** 2

    fx = f(x)
    quad = float(np.dot(w, fx))
    return quad


def clenshaw_curtis_rule_compute(n):
    """
    Clenshaw-Curtis 求积规则（基于 939 的相关实现）。

    节点：x_k = cos(kπ/n)，包含端点。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    x = np.cos(np.pi * np.arange(n + 1) / n)
    # 权重通过 DCT 计算
    c = np.ones(n + 1)
    c[0] = 2.0
    c[n] = 2.0
    k = np.arange(n + 1)
    # 使用显式公式
    w = np.zeros(n + 1)
    for j in range(n + 1):
        if j == 0 or j == n:
            w[j] = 1.0 / (n ** 2 - 1.0)
        else:
            s = 0.0
            for m in range(1, n // 2):
                s += np.cos(2.0 * m * j * np.pi / n) / (4.0 * m ** 2 - 1.0)
            if n % 2 == 0:
                s += 0.5 * np.cos(n * j * np.pi / n) / (n ** 2 - 1.0)
            w[j] = (1.0 - 2.0 * s) / n
    w[0] *= 0.5
    w[n] *= 0.5
    return x, w


def integrate_vertical_column(z_levels, values, method="gauss_legendre"):
    """
    对大气垂直柱进行数值积分。

    Parameters
    ----------
    z_levels : ndarray
        垂直层次坐标（归一化到 [-1, 1] 或 [0, 1]）。
    values : ndarray
        各层次上的物理量值。
    method : str
        积分方法。

    Returns
    -------
    integral : float
    """
    z_levels = np.asarray(z_levels)
    values = np.asarray(values)

    if method == "gauss_legendre":
        n = len(z_levels)
        # 将 [z_min, z_max] 映射到 [-1, 1]
        z_min, z_max = z_levels[0], z_levels[-1]
        scale = 0.5 * (z_max - z_min)
        shift = 0.5 * (z_max + z_min)

        beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, n + 1)) ** (-2))
        tridiag = np.diag(beta, 1) + np.diag(beta, -1)
        eigval, eigvec = np.linalg.eigh(tridiag)
        x = eigvec[0, :]
        idx = np.argsort(x)
        x = x[idx]
        w = 2.0 * eigval[idx] ** 2

        z_phys = scale * x + shift
        # 线性插值获取函数值
        f_vals = np.interp(z_phys, z_levels, values)
        integral = scale * np.dot(w, f_vals)
        return float(integral)
    else:
        # 梯形法则 fallback
        return float(np.trapezoid(values, z_levels))


def compute_total_column_water_vapor(z, q, rho):
    """
    计算大气总水汽含量（TCWV）。

    公式：
        TCWV = ∫_0^{z_top} ρ(z) * q(z) dz

    Parameters
    ----------
    z : ndarray
        高度坐标（m）。
    q : ndarray
        比湿（kg/kg）。
    rho : ndarray
        空气密度（kg/m³）。

    Returns
    -------
    tcwv : float
        总水汽含量（kg/m²）。
    """
    f_vals = rho * q
    return integrate_vertical_column(z, f_vals, method="gauss_legendre")


def test_fast_quad():
    # 测试：∫_{-1}^1 x^2 dx = 2/3
    def f(x):
        return x ** 2
    val = gauss_legendre_integrate_fast(f, 16)
    assert abs(val - 2.0 / 3.0) < 1e-10
    val2 = fejer1_integrate_fast(f, 64)
    assert abs(val2 - 2.0 / 3.0) < 1e-10
    print("fast_spectral_quadrature 自测试通过")


if __name__ == "__main__":
    test_fast_quad()
