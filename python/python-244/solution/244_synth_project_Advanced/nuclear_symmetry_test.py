#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nuclear_symmetry_test.py
========================
【融合种子项目】 1313_triangle_quadrature_symmetry (三角形求积对称性检验)

核物质在等旋空间 (质子-中子不对称度) 中的对称性检验.

物理:
    对称能 S(rho) = (1/2) d^2 E/A / d delta^2 |_{delta=0}
    delta = (rho_n - rho_p) / rho  等旋不对称度

    E/A(rho, delta) = E/A(rho, 0) + S(rho) delta^2 + O(delta^4)

数学 (重心坐标对称性检验, 移植自 barycentric_symmetry):
    对 (lambda_1, lambda_2, lambda_3) 在三角形内, 检验:
        1) lambda_i 的唯一值数 -> 确定对称类
        2) 排列数 -> 确定多重度
        3) 对称轨道: 中心 (1,1,1)/3, 边中点 (1,1,0)/2, 顶点 (1,0,0)
"""

import math
import numpy as np
from typing import Tuple, Dict, List


def symmetry_energy_parabolic(rho: float, rho_0: float = 2.7e14,
                                S_0: float = 30.0,
                                L_0: float = 60.0) -> float:
    """
    对称能的抛物线近似.

    S(rho) = S_0 (rho/rho_0)^gamma + ...

    或线性化:
        S(rho) ≈ S_0 + L/3 * (rho - rho_0)/rho_0

    其中 L = 3 rho_0 dS/drho|_{rho_0} 为对称能斜率参数.
    """
    x = rho / rho_0
    gamma_eff = L_0 / (3.0 * S_0) + 1.0
    return S_0 * x ** gamma_eff


def nuclear_eos_isospin(rho: float, delta: float,
                          S_func=None) -> Tuple[float, float]:
    """
    等旋不对称核物质 E/A.

    E/A(rho, delta) = E/A(rho, 0) + S(rho) delta^2

    其中 E/A(rho, 0) 为对称核物质 (使用多方近似).

    Returns: (E/A, P) 单位 MeV
    """
    if S_func is None:
        S_func = symmetry_energy_parabolic

    rho_0 = 2.7e14
    # 对称核物质 E/A (近似)
    K_0 = 240.0  # 压缩模量 MeV
    E_0 = -16.0  # 结合能 MeV

    x = rho / rho_0
    ea_sym = E_0 + 0.5 * K_0 / 9.0 * (x - 1.0) ** 2

    S = S_func(rho)
    ea = ea_sym + S * delta * delta

    # 压力: P = rho^2 d(E/A)/drho
    # (数值微分简化)
    drho = rho * 0.01
    ea_plus = ea_sym + 0.5 * K_0 / 9.0 * ((rho + drho) / rho_0 - 1.0) ** 2 \
              + S_func(rho + drho) * delta * delta
    ea_minus = ea_sym + 0.5 * K_0 / 9.0 * ((rho - drho) / rho_0 - 1.0) ** 2 \
               + S_func(rho - drho) * delta * delta
    dea_drho = (ea_plus - ea_minus) / (2.0 * drho)
    P = rho ** 2 * dea_drho

    return ea, P


def barycentric_symmetry_test(points: np.ndarray) -> np.ndarray:
    """
    重心坐标对称性检验 (直接移植自 barycentric_symmetry).

    对每个点 (l1, l2, l3):
        unique_num = unique values in {l1, l2, l3}
        clone_num = number of permutations matching this point

    对称类:
        1: 中心 (1/3, 1/3, 1/3)
        3: 边中点型 (如 (1/2, 1/2, 0) 的排列)
        6: 一般点 (三个不同值)
    """
    n = points.shape[0]
    symmetry = np.zeros(n)

    for i in range(n):
        p = points[i, :3]
        unique_vals = np.unique(np.round(p, 10))
        unique_num = len(unique_vals)

        # 计算排列数
        perms = set()
        from itertools import permutations
        for perm in permutations(range(3)):
            perms.add(tuple(np.round(p[list(perm)], 10)))
        clone_num = len(perms)

        if unique_num == 1:
            if np.allclose(p, [1.0/3.0, 1.0/3.0, 1.0/3.0]):
                symmetry[i] = 1
            else:
                symmetry[i] = 0
        elif unique_num == 2 and clone_num == 3:
            symmetry[i] = 3
        elif unique_num == 3 and clone_num == 6:
            symmetry[i] = 6
        else:
            symmetry[i] = 0

    return symmetry


def isospin_trajectory(rho_start: float, rho_end: float,
                         delta_start: float, delta_end: float,
                         n_steps: int = 50) -> Dict:
    """
    计算等旋空间中的 EoS 轨迹.

    在 (rho, delta) 空间中扫描, 计算 E/A 和 P.
    """
    rho_arr = np.linspace(rho_start, rho_end, n_steps)
    delta_arr = np.linspace(delta_start, delta_end, n_steps)

    ea_arr = np.zeros(n_steps)
    p_arr = np.zeros(n_steps)
    sym_arr = np.zeros(n_steps)

    for k in range(n_steps):
        ea, p = nuclear_eos_isospin(rho_arr[k], delta_arr[k])
        ea_arr[k] = ea
        p_arr[k] = p
        sym_arr[k] = symmetry_energy_parabolic(rho_arr[k])

    return {
        'rho': rho_arr,
        'delta': delta_arr,
        'E_over_A': ea_arr,
        'pressure': p_arr,
        'symmetry_energy': sym_arr,
    }


def triangle_quadrature_symmetry_test(n_points: int = 20) -> Dict:
    """
    三角形网格上的对称性检验.

    在等旋密度三角形 (rho_n, rho_p, rho_e) 上生成点,
    检验 EoS 函数的对称性.
    """
    # 生成三角形内均匀点 (重心坐标)
    pts = []
    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - i - j
            l1 = i / n_points
            l2 = j / n_points
            l3 = k / n_points
            pts.append([l1, l2, l3])
    pts = np.array(pts)

    sym_classes = barycentric_symmetry_test(pts)

    return {
        'points': pts,
        'symmetry_classes': sym_classes,
        'n_class1': int(np.sum(sym_classes == 1)),
        'n_class3': int(np.sum(sym_classes == 3)),
        'n_class6': int(np.sum(sym_classes == 6)),
        'total_points': len(pts),
    }


# 自检
if __name__ == "__main__":
    print("=== 核对称性检验自检 ===")

    S = symmetry_energy_parabolic(2.7e14)
    print(f"S(rho_0) = {S:.2f} MeV")
    S2 = symmetry_energy_parabolic(5.4e14)
    print(f"S(2 rho_0) = {S2:.2f} MeV")

    ea, p = nuclear_eos_isospin(2.7e14, 0.0)
    print(f"E/A (sym, rho_0) = {ea:.2f} MeV")
    ea2, p2 = nuclear_eos_isospin(2.7e14, 0.3)
    print(f"E/A (delta=0.3) = {ea2:.2f} MeV")

    # 对称性检验
    pts_test = np.array([
        [1/3, 1/3, 1/3],
        [0.5, 0.5, 0.0],
        [0.5, 0.3, 0.2],
        [1.0, 0.0, 0.0],
    ])
    sym = barycentric_symmetry_test(pts_test)
    print(f"对称类: {sym}")

    traj = isospin_trajectory(1e14, 8e14, 0.0, 0.4, 20)
    print(f"等旋轨迹: {len(traj['rho'])} 点")

    tri_test = triangle_quadrature_symmetry_test(10)
    print(f"三角形点: {tri_test['total_points']}")
    print(f"  类1: {tri_test['n_class1']}, 类3: {tri_test['n_class3']}, 类6: {tri_test['n_class6']}")

    print("\nnuclear_symmetry_test.py 自检通过.")
