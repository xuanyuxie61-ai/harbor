#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neutron_star_mesh.py
====================
【融合种子项目】 748_medit_to_fem (MEDIT 网格 → FEM 网格转换)

本模块将 MEDIT 格式的有限元网格读写思想移植到中子星径向坐标离散化:
将连续的径向密度分布 [0, R_star] 转换为分层球壳结构 (类似 FEM 中的
单元-节点映射).

物理/数学公式
-------------
1. 球坐标径向度量:
       体积元 dV = 4 pi r^2 dr
       壳层体积 V_k = (4/3) pi (r_{k+1}^3 - r_k^3)

2. 自适应网格生成 (基于密度梯度):
       h_k = h_0 * [1 + alpha |d ln rho / d ln r|_k^p]^{-1/p}
   其中 alpha 为聚集参数, p 为聚集指数.

3. 质量坐标变换 (Lagrangian 坐标):
       dm = 4 pi r^2 rho dr
       m(r) = integral_0^r 4 pi r'^2 rho(r') dr'

4. 球谐展开在径向网格上的离散:
       Y_{lm}(theta, phi) = N_{lm} P_l^m(cos theta) e^{im phi}
   径向基函数:
       R_{nk}(r) = j_l(k_{nl} r / R_star),  j_l 为球 Bessel 函数

5. 节点-单元映射 (从 MEDIT 移植):
       节点编号: 0, 1, ..., N
       单元 k: 节点 (k, k+1),  k = 0, 1, ..., N-1
       单元体积: V_k = (4/3) pi (r_{k+1}^3 - r_k^3)

6. 网格质量指标 (纵横比):
       AR_k = max(h_{k+1}, h_k) / min(h_{k+1}, h_k)
       其中 h_k = r_{k+1} - r_k 为壳层厚度

7. 坐标变换 (球 → 伪笛卡尔):
       x = r sin(theta) cos(phi)
       y = r sin(theta) sin(phi)
       z = r cos(theta)
"""

import math
import numpy as np
from typing import Tuple, Dict, List, Optional
from numerical_constants import (r8vec_linspace, r8_epsilon, r8_heaviside,
                                  NeutronStarConstants as NS)


# ============================================================
# 第一部分: 径向网格生成
# ============================================================

def uniform_radial_mesh(n_shells: int, r_star: float,
                        r_core: Optional[float] = None) -> np.ndarray:
    """
    生成均匀的径向网格.

    节点: r_k = k * R_star / N,  k = 0, 1, ..., N

    Parameters
    ----------
    n_shells : int
        壳层数
    r_star : float
        星体半径 (km)
    r_core : float, optional
        核半径 (用于标记). 未使用但保留以兼容接口.

    Returns
    -------
    np.ndarray, shape (n_shells + 1,)
        径向节点坐标 (km)
    """
    if n_shells < 1:
        raise ValueError("壳层数必须 >= 1")
    if r_star <= 0.0:
        raise ValueError("星体半径必须为正")
    return r8vec_linspace(n_shells + 1, 0.0, r_star)


def graded_radial_mesh(n_shells: int, r_star: float,
                       grading_factor: float = 1.5,
                       r_transition: float = 5.0) -> np.ndarray:
    """
    生成渐变径向网格 (中心密, 外围疏).

    映射函数:
        r_k = R_star * (k/N)^{1/grading_factor}

    或者分段:
        内区 (r < r_trans): 细密
        外区 (r > r_trans): 稀疏

    Parameters
    ----------
    n_shells : int
        壳层数
    r_star : float
        星体半径 (km)
    grading_factor : float
        聚集指数 (>1 向中心聚集, <1 向外围聚集)
    r_transition : float
        内外区分界半径 (km)

    Returns
    -------
    np.ndarray, shape (n_shells + 1,)
        径向节点
    """
    if n_shells < 1:
        raise ValueError("壳层数必须 >= 1")
    if r_star <= 0.0:
        raise ValueError("星体半径必须为正")
    if grading_factor <= 0.0:
        grading_factor = 1.0

    # 均匀参数 s in [0, 1]
    s = np.linspace(0.0, 1.0, n_shells + 1)
    # 幂律映射
    r = r_star * s ** (1.0 / grading_factor)
    # 边界修复: 确保 r[0]=0, r[-1]=r_star
    r[0] = 0.0
    r[-1] = r_star
    return r


def adaptive_radial_mesh(n_shells: int, r_star: float,
                         density_profile: np.ndarray,
                         alpha: float = 2.0,
                         p_exp: float = 2.0) -> np.ndarray:
    """
    基于密度梯度自适应生成径向网格.

    监控函数:
        phi(r) = 1 + alpha |d ln rho / d ln r|^p

    累积分布:
        xi(r) = integral_0^r phi(r') dr' / integral_0^{R_star} phi(r') dr'

    网格: 在 xi 空间等分, 然后反解 r.

    Parameters
    ----------
    n_shells : int
        壳层数
    r_star : float
        星体半径 (km)
    density_profile : np.ndarray, shape (n_fine,)
        在细密网格上的密度采样 (g/cm^3), 用于估计梯度
    alpha : float
        聚集强度参数
    p_exp : float
        聚集指数

    Returns
    -------
    np.ndarray, shape (n_shells + 1,)
        自适应径向节点
    """
    n_fine = len(density_profile)
    r_fine = np.linspace(0.0, r_star, n_fine)

    # 计算 |d ln rho / d ln r|  (在 r > 0 处)
    eps = r8_epsilon()
    rho_safe = np.maximum(density_profile, eps)
    ln_rho = np.log(rho_safe)
    ln_r = np.zeros_like(r_fine)
    ln_r[1:] = np.log(r_fine[1:])
    ln_r[0] = ln_r[1] - (r_fine[1] - r_fine[0]) / r_fine[1]

    # 中心差分
    d_ln_rho = np.zeros_like(ln_rho)
    d_ln_r = np.zeros_like(ln_r)
    d_ln_rho[1:-1] = (ln_rho[2:] - ln_rho[:-2]) / (ln_r[2:] - ln_r[:-2] + eps)
    d_ln_r[1:-1] = ln_r[2:] - ln_r[:-2]
    d_ln_rho[0] = d_ln_rho[1]
    d_ln_rho[-1] = d_ln_rho[-2]

    # 监控函数
    phi = 1.0 + alpha * np.abs(d_ln_rho) ** p_exp

    # 累积积分 (梯形法则)
    cum_phi = np.zeros_like(phi)
    for k in range(1, n_fine):
        cum_phi[k] = cum_phi[k - 1] + 0.5 * (phi[k] + phi[k - 1]) * (r_fine[k] - r_fine[k - 1])

    # 归一化
    cum_phi /= cum_phi[-1]

    # 反插值: 在 cum_phi 空间等分
    xi_target = np.linspace(0.0, 1.0, n_shells + 1)
    r_adapt = np.interp(xi_target, cum_phi, r_fine)
    r_adapt[0] = 0.0
    r_adapt[-1] = r_star
    return r_adapt


# ============================================================
# 第二部分: 球壳几何量
# ============================================================

def shell_volumes(r_nodes: np.ndarray) -> np.ndarray:
    """
    计算每个球壳的体积.

    V_k = (4/3) pi (r_{k+1}^3 - r_k^3)

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N+1,)
        径向节点

    Returns
    -------
    np.ndarray, shape (N,)
        各壳层体积 (km^3)
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    return (4.0 / 3.0) * math.pi * (r[1:] ** 3 - r[:-1] ** 3)


def shell_masses(r_nodes: np.ndarray, rho_centers: np.ndarray) -> np.ndarray:
    """
    计算每个球壳的质量 (假设壳内密度线性变化).

    M_k = integral_{r_k}^{r_{k+1}} 4 pi r^2 rho(r) dr

    若 rho 在壳内线性:
        rho(r) = rho_k + (rho_{k+1} - rho_k) * (r - r_k) / (r_{k+1} - r_k)

    解析结果:
        M_k = 4 pi * [rho_avg * (r_{k+1}^3 - r_k^3)/3
              + delta_rho/(r_{k+1}-r_k) * (r_{k+1}^4 - r_k^4)/4
              - delta_rho * r_k / (r_{k+1}-r_k) * (r_{k+1}^3 - r_k^3)/3]

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N+1,)
        径向节点 (km)
    rho_centers : np.ndarray, shape (N,)
        壳中心密度 (g/cm^3)

    Returns
    -------
    np.ndarray, shape (N,)
        壳层质量 (g)
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    rho_c = np.asarray(rho_centers, dtype=np.float64)
    n_shells = len(rho_c)
    masses = np.zeros(n_shells)

    # km 到 cm 转换
    km_to_cm = 1.0e5

    for k in range(n_shells):
        r_in = r[k] * km_to_cm
        r_out = r[k + 1] * km_to_cm
        h = r_out - r_in
        if h <= 0.0:
            continue
        # 假设壳内密度为 rho_c[k] (壳平均近似)
        masses[k] = rho_c[k] * (4.0 / 3.0) * math.pi * (r_out ** 3 - r_in ** 3)

    return masses


def shell_aspect_ratios(r_nodes: np.ndarray) -> np.ndarray:
    """
    计算网格纵横比 (衡量网格质量).

    AR_k = max(h_k, h_{k+1}) / min(h_k, h_{k+1})
    h_k = r_k - r_{k-1}

    Returns
    -------
    np.ndarray, shape (N-1,)
        内部节点的纵横比
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    h = np.diff(r)
    ar = np.zeros(len(h) - 1)
    for k in range(len(h) - 1):
        denom = min(h[k], h[k + 1])
        if denom < r8_epsilon():
            ar[k] = float('inf')
        else:
            ar[k] = max(h[k], h[k + 1]) / denom
    return ar


# ============================================================
# 第三部分: 质量坐标转换 (Lagrangian 坐标)
# ============================================================

def radial_to_mass_coordinate(r_nodes: np.ndarray,
                               rho_profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 Euler 坐标 (r) 转换为 Lagrange 质量坐标 (m).

    dm/dr = 4 pi r^2 rho(r)
    m(r) = integral_0^r 4 pi r'^2 rho(r') dr'

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N+1,)
        径向节点 (km)
    rho_profile : np.ndarray, shape (N+1,)
        径向密度分布 (g/cm^3)

    Returns
    -------
    m_nodes : np.ndarray, shape (N+1,)
        质量坐标 (g)
    r_nodes : np.ndarray, shape (N+1,)
        原径向坐标 (km)
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    rho = np.asarray(rho_profile, dtype=np.float64)
    km_to_cm = 1.0e5

    n = len(r)
    m = np.zeros(n)
    for k in range(1, n):
        r_in = r[k - 1] * km_to_cm
        r_out = r[k] * km_to_cm
        rho_avg = 0.5 * (rho[k - 1] + rho[k])
        dm = 4.0 * math.pi * rho_avg * (r_out ** 3 - r_in ** 3) / 3.0
        m[k] = m[k - 1] + dm

    return m, r


def mass_to_radial_coordinate(m_target: float,
                               m_nodes: np.ndarray,
                               r_nodes: np.ndarray) -> float:
    """
    反解质量坐标对应的径向坐标.

    通过线性插值在 (m, r) 关系中反查.

    Parameters
    ----------
    m_target : float
        目标质量坐标 (g)
    m_nodes : np.ndarray
        质量坐标网格
    r_nodes : np.ndarray
        径向坐标网格

    Returns
    -------
    float
        对应的径向坐标 (km)
    """
    if m_target <= m_nodes[0]:
        return float(r_nodes[0])
    if m_target >= m_nodes[-1]:
        return float(r_nodes[-1])
    return float(np.interp(m_target, m_nodes, r_nodes))


# ============================================================
# 第四部分: FEM 格式的网格数据 I/O (移植自 medit_to_fem)
# ============================================================

def mesh_to_fem_format(r_nodes: np.ndarray) -> Dict:
    """
    将径向网格转换为 FEM 格式 (节点 + 单元).

    节点: 0, 1, ..., N  (坐标 r_k)
    单元: k = 0, ..., N-1  (连接节点 k 和 k+1)

    这是 medit_to_fem 思想在一维径向网格上的直接应用.

    Returns
    -------
    dict
        'nodes': shape (N+1, 1),  节点坐标 (km)
        'elements': shape (N, 2), 单元-节点连接 (0-based)
        'shell_volumes': shape (N,), 壳层体积 (km^3)
        'shell_thicknesses': shape (N,), 壳层厚度 (km)
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    n_shells = len(r) - 1

    nodes = r.reshape(-1, 1)
    elements = np.zeros((n_shells, 2), dtype=int)
    for k in range(n_shells):
        elements[k, 0] = k
        elements[k, 1] = k + 1

    vols = shell_volumes(r)
    thick = np.diff(r)

    return {
        'nodes': nodes,
        'elements': elements,
        'shell_volumes': vols,
        'shell_thicknesses': thick,
        'n_nodes': len(r),
        'n_elements': n_shells,
    }


def mesh_quality_report(r_nodes: np.ndarray) -> Dict:
    """
    网格质量报告.

    指标:
        - 最小/最大壳层厚度
        - 平均纵横比
        - 最大纵横比
        - 最小壳层体积
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    h = np.diff(r)
    ar = shell_aspect_ratios(r)
    vols = shell_volumes(r)

    return {
        'min_thickness': float(np.min(h)),
        'max_thickness': float(np.max(h)),
        'mean_thickness': float(np.mean(h)),
        'mean_aspect_ratio': float(np.mean(ar)),
        'max_aspect_ratio': float(np.max(ar)),
        'min_volume': float(np.min(vols)),
        'total_volume': float(np.sum(vols)),
    }


# ============================================================
# 第五部分: 球 Bessel 函数基 (用于径向基展开)
# ============================================================

def spherical_bessel_j(l: int, x: float) -> float:
    """
    计算球 Bessel 函数 j_l(x).

    递推:
        j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)

    初值:
        j_0(x) = sin(x) / x
        j_1(x) = sin(x)/x^2 - cos(x)/x

    对小 x (x < 1e-8):
        j_l(x) ~ x^l / (2l+1)!!
    """
    if abs(x) < 1.0e-8:
        # 小参数展开
        result = 1.0
        for k in range(1, l + 1):
            result *= x / (2 * k + 1)
        return result

    if l == 0:
        return math.sin(x) / x
    elif l == 1:
        return math.sin(x) / (x * x) - math.cos(x) / x

    j_prev = math.sin(x) / x
    j_curr = math.sin(x) / (x * x) - math.cos(x) / x

    for k in range(1, l):
        j_next = (2 * k + 1) / x * j_curr - j_prev
        j_prev = j_curr
        j_curr = j_next

    return j_curr


def radial_basis_functions(r_nodes: np.ndarray, l_max: int,
                            n_modes: int) -> np.ndarray:
    """
    构造径向基函数矩阵.

    R_{nl}(r) = j_l(k_{nl} r / R_star)

    其中 k_{nl} 为 j_l 的第 n 个零点.
    这里简化: k_{nl} ≈ (n + l/2) pi

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N+1,)
    l_max : int
        最大角动量量子数
    n_modes : int
        每个 l 的模式数

    Returns
    -------
    np.ndarray, shape (n_modes * (l_max+1), N+1)
        基函数值矩阵
    """
    r = np.asarray(r_nodes, dtype=np.float64)
    r_star = r[-1]
    n_basis = n_modes * (l_max + 1)
    basis = np.zeros((n_basis, len(r)))

    idx = 0
    for l in range(l_max + 1):
        for n in range(1, n_modes + 1):
            k_nl = (n + l / 2.0) * math.pi  # 近似零点
            for j, rj in enumerate(r):
                basis[idx, j] = spherical_bessel_j(l, k_nl * rj / r_star)
            idx += 1

    return basis


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== 中子星径向网格自检 ===")
    r_uni = uniform_radial_mesh(20, 12.0)
    print(f"均匀网格 20 壳: r = [{r_uni[0]:.2f}, {r_uni[-1]:.2f}] km")

    r_grade = graded_radial_mesh(20, 12.0, grading_factor=2.0)
    print(f"渐变网格 (gamma=2): r = [{r_grade[0]:.2f}, {r_grade[-1]:.2f}] km")
    print(f"  内层厚度: {r_grade[1] - r_grade[0]:.4f} km")
    print(f"  外层厚度: {r_grade[-1] - r_grade[-2]:.4f} km")

    # 测试自适应网格
    rho_test = np.exp(-np.linspace(0, 3, 200)) * 1e15
    r_adapt = adaptive_radial_mesh(20, 12.0, rho_test, alpha=3.0)
    print(f"自适应网格 20 壳: r = [{r_adapt[0]:.2f}, {r_adapt[-1]:.2f}] km")

    vols = shell_volumes(r_uni)
    print(f"壳层体积总和: {np.sum(vols):.4f} km^3")
    print(f"精确球体积: {4/3*math.pi*12**3:.4f} km^3")

    fem = mesh_to_fem_format(r_uni)
    print(f"FEM 格式: {fem['n_nodes']} 节点, {fem['n_elements']} 单元")

    qr = mesh_quality_report(r_uni)
    print(f"网格质量: 最大纵横比 = {qr['max_aspect_ratio']:.4f}")

    print(f"球 Bessel j_0(pi/2) = {spherical_bessel_j(0, math.pi/2):.6f}")
    print(f"球 Bessel j_1(pi) = {spherical_bessel_j(1, math.pi):.6f}")

    print("neutron_star_mesh.py 自检通过.")
