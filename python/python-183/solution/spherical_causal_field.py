r"""
spherical_causal_field.py
================================================================================
球面经纬度网格上的因果场离散化与球面调和展开

原项目映射: 1122_sphere_llq_grid — 球面 LLQ 网格点生成

科学背景
--------
在气候科学、地球物理与神经科学中，因果推断经常面临**球面数据**
（如全球气温场、脑皮层信号）。将因果效应建模为定义在二维球面 $S^2$ 上的
标量场 $u(\theta,\phi)$，可以自然地利用球面几何结构。

球面拉普拉斯算子（Laplace-Beltrami）在因果扩散中起关键作用：
$$ \Delta_{S^2} u = \frac{1}{\sin\theta}\frac{\partial}{\partial\theta}\left(\sin\theta\frac{\partial u}{\partial\theta}\right) + \frac{1}{\sin^2\theta}\frac{\partial^2 u}{\partial\phi^2} $$

球面调和函数 $Y_l^m(\theta,\phi)$ 是该算子的特征函数：
$$ \Delta_{S^2} Y_l^m = -l(l+1) Y_l^m $$

核心公式
--------
1. 球面因果扩散方程：
   $$ \frac{\partial u}{\partial t} = D \Delta_{S^2} u + f(\theta,\phi,t) $$

2. 球面调和展开（截断到 $L_{\max}$）：
   $$ u(\theta,\phi) = \sum_{l=0}^{L_{\max}}\sum_{m=-l}^{l} a_l^m Y_l^m(\theta,\phi) $$

3. 连带 Legendre 函数 $P_l^m(\cos\theta)$（递推计算）：
   $$ (l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m $$

4. 球面 LLQ 网格点：
   纬度圈：$\phi_k = \frac{\pi k}{N_{\text{lat}}+1}, k=1,\dots,N_{\text{lat}}$
   经度圈：$\theta_j = \frac{2\pi j}{N_{\text{long}}}, j=0,\dots,N_{\text{long}}-1$
   总点数：$N_{\text{tot}} = 2 + N_{\text{lat}} \cdot N_{\text{long}}$（含两极）。
r"""

import numpy as np
from typing import Tuple


def sphere_llq_grid_points(r: float, pc: np.ndarray,
                            lat_num: int, long_num: int) -> np.ndarray:
    r"""
    生成球面 LLQ (Latitude-Longitude-Quadrilateral) 网格点。

    Parameters
    ----------
    r : float
        球半径，必须 > 0。
    pc : ndarray, shape (3,)
        球心坐标。
    lat_num : int
        纬度线数（不含两极）。
    long_num : int
        经度线数。

    Returns
    -------
    points : ndarray, shape (n_points, 3)
        球面网格点笛卡尔坐标。
    r"""
    if r <= 0.0:
        raise ValueError("半径 r 必须为正。")
    if lat_num < 0 or long_num < 1:
        raise ValueError("lat_num 必须 >=0, long_num 必须 >=1。")

    n_points = 2 + lat_num * long_num
    points = np.zeros((n_points, 3))
    n = 0

    # 北极
    theta = 0.0
    phi = 0.0
    points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
    points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
    points[n, 2] = pc[2] + r * np.cos(phi)
    n += 1

    # 中间纬度圈
    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        for lon in range(long_num):
            theta = 2.0 * np.pi * lon / long_num
            points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
            points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
            points[n, 2] = pc[2] + r * np.cos(phi)
            n += 1

    # 南极
    theta = 0.0
    phi = np.pi
    points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
    points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
    points[n, 2] = pc[2] + r * np.cos(phi)
    n += 1

    return points


def associated_legendre(l_max: int, x: float) -> np.ndarray:
    r"""
    计算连带 Legendre 函数 $P_l^m(x)$ 对所有 $l\le l_{\max}, |m|\le l$。

    使用标准递推公式，返回值以二维数组 P[l, m+l] 形式组织。
    r"""
    if l_max < 0:
        raise ValueError("l_max 必须非负。")
    if not (-1.0 <= x <= 1.0):
        raise ValueError("x 必须在 [-1,1] 内。")

    P = np.zeros((l_max + 1, 2 * l_max + 1))
    # P_0^0 = 1
    P[0, l_max] = 1.0
    if l_max == 0:
        return P

    # 递推 m=l 的项
    somx2 = np.sqrt(max(0.0, 1.0 - x * x))
    for l in range(1, l_max + 1):
        P[l, l_max + l] = - (2 * l - 1) * somx2 * P[l - 1, l_max + l - 1]
        if l < l_max:
            P[l, l_max + l + 1] = x * (2 * l + 1) * P[l, l_max + l]

    # 递推 m < l 的项
    for l in range(2, l_max + 1):
        for m in range(l - 2, -1, -1):
            idx_m = l_max + m
            P[l, idx_m] = ((2 * l - 1) * x * P[l - 1, idx_m] - (l + m - 1) * P[l - 2, idx_m]) / (l - m)

    return P


def spherical_harmonic_Y(l: int, m: int, theta: float, phi: float) -> complex:
    r"""
    计算归一化球面调和函数 $Y_l^m(\theta,\phi)$。

    $$ Y_l^m(\theta,\phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos\theta) e^{im\phi} $$
    r"""
    if abs(m) > l:
        return 0.0 + 0.0j
    x = np.cos(theta)
    P_all = associated_legendre(l, x)
    P_lm = P_all[l, l + m]

    # 归一化系数
    from math import factorial, sqrt
    norm = sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    return complex(norm * P_lm * np.cos(m * phi), norm * P_lm * np.sin(m * phi))


def spherical_laplacian_spectrum(l_max: int) -> np.ndarray:
    r"""
    球面 Laplace-Beltrami 算子的特征值：$\lambda_l = -l(l+1)$。

    Returns
    -------
    lambdas : ndarray, shape (l_max+1,)
        对应 $l=0,1,\dots,l_{\max}$ 的特征值。
    r"""
    l = np.arange(l_max + 1)
    return -l * (l + 1)


def project_to_spherical_harmonics(field_values: np.ndarray,
                                    points: np.ndarray,
                                    l_max: int) -> np.ndarray:
    r"""
    将定义在球面网格点上的因果场投影到球面调和基。

    使用最小二乘拟合：$\mathbf{a} = (Y^T Y)^{-1} Y^T \mathbf{f}$。
    r"""
    n_points = len(field_values)
    n_modes = (l_max + 1) ** 2
    Ymat = np.zeros((n_points, n_modes))

    for idx in range(n_points):
        x, y, z = points[idx]
        r = np.sqrt(x * x + y * y + z * z)
        if r < 1e-12:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
        col = 0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Yval = spherical_harmonic_Y(l, m, theta, phi)
                Ymat[idx, col] = Yval.real
                col += 1

    # 正则化最小二乘
    coeffs = np.linalg.lstsq(Ymat, field_values, rcond=None)[0]
    return coeffs


def demo():
    r"""模块自测试。"""
    points = sphere_llq_grid_points(r=1.0, pc=np.zeros(3), lat_num=4, long_num=8)
    print(f"[spherical_causal_field] 球面网格点数: {len(points)}")

    # 构造测试因果场：两个高斯波包
    field = np.zeros(len(points))
    for i, pt in enumerate(points):
        # 转换为球坐标
        r = np.linalg.norm(pt)
        if r > 1e-12:
            theta = np.arccos(np.clip(pt[2] / r, -1.0, 1.0))
            phi = np.arctan2(pt[1], pt[0])
            field[i] = np.exp(-2.0 * (theta - np.pi / 3.0) ** 2) * np.cos(2 * phi)

    coeffs = project_to_spherical_harmonics(field, points, l_max=3)
    print(f"[spherical_causal_field] 球面调和系数 (前5): {coeffs[:5].round(4)}")
    return points, coeffs


if __name__ == "__main__":
    demo()
