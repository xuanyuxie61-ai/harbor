"""
elliptic_solutions.py
椭圆积分精确解 — 冰盖一维剖面验证基准

基于种子项目 335_elliptic_integral 的 Carlson 对称形式与
Jacobi 椭圆函数算法，提供冰盖力学中的精确解析解，
用于验证数值模型的精度与收敛性。

核心数学:
  1. Vialov 剖面 (等宽渠道, 冰面无积累):
       对于宽度为 L 的矩形截面冰川，表面高程满足:

       s(x) = H_0 \left[ 1 - \left(\frac{2|x|}{L}\right)^{n+1} \right]^{n/(2n+2)}

       其中 n 为 Glen 指数，H_0 为中线最大厚度。

  2. 椭圆积分关系:
       某些简化冰盖模型 (如完美塑性近似 n -> \infty) 的剖面
       涉及第一类/第二类完全椭圆积分:

       K(k) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - k^2 \sin^2\theta}}
       E(k) = \int_0^{\pi/2} \sqrt{1 - k^2 \sin^2\theta} \, d\theta

  3. Carlson 对称形式:
       R_F(x, y, z) = \frac{1}{2} \int_0^{\infty} \frac{dt}{\sqrt{(t+x)(t+y)(t+z)}}

       K(k) = R_F(0, 1-k^2, 1)
       E(k) = R_F(0, 1-k^2, 1) - \frac{k^2}{3} R_D(0, 1-k^2, 1)

  4. Jacobi 椭圆函数:
       sn(u|m), cn(u|m), dn(u|m) 通过 Landen 变换 (AGM) 计算

应用场景:
  - 验证 SIA 数值解与 Vialov 解析解的偏差
  - 计算冰盖剖面的精确体积与表面积
  - 提供网格收敛性分析 (Mesh Convergence Analysis) 的基准
"""

import numpy as np
from typing import Tuple


def carlson_rf(x: float, y: float, z: float, errtol: float = 1e-10) -> float:
    """
    Carlson 对称形式 R_F(x, y, z)。

    算法: 迭代使用 duplication theorem 直至参数几乎相等，
    然后做 5 阶 Taylor 展开。

    参数:
        x, y, z: >= 0, 至多一个为 0
        errtol: 收敛容差

    返回:
        R_F(x, y, z)
    """
    x0, y0, z0 = float(x), float(y), float(z)

    if x0 < 0 or y0 < 0 or z0 < 0:
        raise ValueError("Arguments to R_F must be non-negative.")

    if x0 + y0 + z0 < 1e-20:
        return 0.0

    x, y, z = x0, y0, z0
    s = 0.0
    power4 = 1.0

    for _ in range(100):
        lam = np.sqrt(x * y) + np.sqrt(y * z) + np.sqrt(z * x)
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        power4 *= 0.25

        avg = (x + y + z) / 3.0
        dx = 1.0 - x / avg
        dy = 1.0 - y / avg
        dz = 1.0 - z / avg

        if max(abs(dx), abs(dy), abs(dz)) < errtol:
            break

    # 5 阶 Taylor 展开
    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz

    rf = (1.0 - e2 / 10.0 + e3 / 14.0 + e2 ** 2 / 24.0
          - 3.0 * e2 * e3 / 44.0) / np.sqrt(avg)

    return float(rf)


def carlson_rd(x: float, y: float, z: float, errtol: float = 1e-10) -> float:
    """
    Carlson 对称形式 R_D(x, y, z) = R_J(x, y, z, z)。
    """
    x0, y0, z0 = float(x), float(y), float(z)

    if x0 < 0 or y0 < 0 or z0 <= 0:
        raise ValueError("Arguments to R_D must be non-negative with z > 0.")

    x, y, z = x0, y0, z0
    s = 0.0
    power4 = 1.0
    fac = 1.0

    for _ in range(100):
        lam = np.sqrt(x * y) + np.sqrt(y * z) + np.sqrt(z * x)
        s += fac / (np.sqrt(z) * (z + lam))
        fac *= 0.25
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)

        avg = (x + y + 3.0 * z) / 5.0
        dx = 1.0 - x / avg
        dy = 1.0 - y / avg
        dz = 1.0 - z / avg

        if max(abs(dx), abs(dy), abs(dz)) < errtol:
            break

    e2 = dx * dy + 2.0 * dz * (dx + dy) + 3.0 * dz ** 2
    e3 = dy * dz * (dx + dy) + 2.0 * dx * dy * dz + 4.0 * dz ** 3
    e4 = dx * dy * dz ** 2
    e5 = dx * dy ** 2 * dz ** 2

    rd = (1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 ** 2 / 88.0
          - 3.0 * e4 / 22.0 - 9.0 * e2 * e3 / 52.0 + 3.0 * e5 / 26.0)
    rd = 3.0 * s + power4 * rd / (avg * np.sqrt(avg))

    return float(rd)


def elliptic_k_complete(k: float) -> float:
    """
    第一类完全椭圆积分 K(k)。

        K(k) = R_F(0, 1-k^2, 1)
    """
    m = k ** 2
    if m < 0 or m > 1:
        raise ValueError("k^2 must be in [0, 1]")
    return carlson_rf(0.0, 1.0 - m, 1.0)


def elliptic_e_complete(k: float) -> float:
    """
    第二类完全椭圆积分 E(k)。

        E(k) = R_F(0, 1-k^2, 1) - \frac{k^2}{3} R_D(0, 1-k^2, 1)
    """
    m = k ** 2
    if m < 0 or m > 1:
        raise ValueError("k^2 must be in [0, 1]")
    return carlson_rf(0.0, 1.0 - m, 1.0) - (m / 3.0) * carlson_rd(0.0, 1.0 - m, 1.0)


def vialov_profile(x: np.ndarray, L: float, H0: float, n: float = 3.0) -> np.ndarray:
    """
    Vialov 解析冰盖剖面 (等宽渠道, 无积累)。

        H(x) = H_0 \left[ 1 - \left( \frac{2|x|}{L} \right)^{n+1} \right]^{n/(2n+2)}

    参数:
        x: 水平坐标 (m), 范围 [-L/2, L/2]
        L: 冰川总宽度 (m)
        H0: 中心最大厚度 (m)
        n: Glen 指数

    返回:
        H: 厚度剖面
    """
    x = np.asarray(x, dtype=np.float64)
    L = float(L)
    H0 = float(H0)

    if L <= 0 or H0 <= 0:
        raise ValueError("L and H0 must be positive.")

    xi = 2.0 * np.abs(x) / L
    xi = np.clip(xi, 0.0, 1.0)

    exponent_base = n + 1.0
    exponent_result = n / (2.0 * n + 2.0)

    # 避免负数开方
    base = 1.0 - xi ** exponent_base
    base = np.maximum(base, 0.0)

    H = H0 * (base ** exponent_result)
    return H


def vialov_volume_exact(L: float, H0: float, n: float = 3.0) -> float:
    """
    Vialov 剖面的精确体积 (单位长度冰川)。

        V = \int_{-L/2}^{L/2} H(x) dx
          = H_0 L \cdot B\left(\frac{1}{n+1}, \frac{3n+2}{2n+2}\right) / (n+1)

    其中 B 为 Beta 函数。
    """
    from math import gamma
    a = 1.0 / (n + 1.0)
    b = (3.0 * n + 2.0) / (2.0 * n + 2.0)
    beta = gamma(a) * gamma(b) / gamma(a + b)
    return float(H0 * L * beta / (n + 1.0))


def bueler_exact_radius(accumulation: float, A: float,
                        rho_g: float, n: float = 3.0,
                        H0: float = 1000.0) -> float:
    """
    Bueler (2003) 圆对称冰盖稳态半径解析公式。

    对于圆对称、恒定积累率 a_m 的稳态冰盖:

        R = \left[ \frac{2(n+1)}{n} \left( \frac{a_m}{2 A (\rho g)^n} \right)^{1/n} H_0^{(n+2)/n} \right]^{n/(2n+2)}

    参数:
        accumulation: 积累率 a_m (m/s)
        A: 率因子 (Pa^{-n} s^{-1})
        rho_g: \rho g (Pa m^{-1})
        n: Glen 指数
        H0: 中心厚度 (m)

    返回:
        R: 冰盖半径 (m)
    """
    if accumulation <= 0 or A <= 0:
        return 0.0

    prefactor = 2.0 * (n + 1.0) / n
    term = (accumulation / (2.0 * A * (rho_g ** n))) ** (1.0 / n)
    R = (prefactor * term * (H0 ** ((n + 2.0) / n))) ** (n / (2.0 * n + 2.0))
    return float(R)


def exact_surface_area_vialov(L: float, H0: float, n: float = 3.0) -> float:
    """
    近似计算 Vialov 剖面的表面积。

    对于缓坡冰川，表面积近似为:
        A_s \approx L + 2 \int_0^{L/2} \frac{1}{2} (\partial H/\partial x)^2 dx

    这里采用更精确的数值积分。
    """
    nx = 1000
    x = np.linspace(-L / 2.0, L / 2.0, nx)
    H = vialov_profile(x, L, H0, n)

    # 数值导数
    dHdx = np.zeros_like(H)
    dHdx[1:-1] = (H[2:] - H[:-2]) / (x[2] - x[0])

    # 弧长积分
    ds = np.sqrt(1.0 + dHdx ** 2)
    arc_length = np.trapezoid(ds, x)
    return float(arc_length)


def convergence_test_vialov(nx_list: list, L: float, H0: float, n: float = 3.0) -> dict:
    """
    网格收敛性测试: 比较数值解与 Vialov 解析解的 L2 误差。

    返回:
        {'nx': [...], 'error_l2': [...], 'order': float}
    """
    errors = []
    for nx in nx_list:
        x = np.linspace(-L / 2.0, L / 2.0, nx)
        H_exact = vialov_profile(x, L, H0, n)

        # 简化数值解: 直接采样解析解作为"数值解"
        # 实际应用中这里会调用 SIA 求解器
        H_num = H_exact.copy()
        # 加入人工离散化误差模拟
        dx = x[1] - x[0]
        H_num = H_num + 0.01 * dx * np.sin(2 * np.pi * x / L)

        err = np.sqrt(np.mean((H_exact - H_num) ** 2))
        errors.append(err)

    # 估计收敛阶
    order = None
    if len(errors) >= 2 and len(nx_list) >= 2:
        order = -np.log(errors[-1] / errors[-2]) / np.log(nx_list[-1] / nx_list[-2])

    return {
        'nx': nx_list,
        'error_l2': errors,
        'order': order,
    }
