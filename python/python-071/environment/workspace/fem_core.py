# -*- coding: utf-8 -*-
"""
fem_core.py
有限元核心离散模块

融合来源:
- 413_fem2d_sample: 三角形 T3/T6 基函数、单元刚度矩阵组装
- 1409_wedge_integrals: 楔形区域上的单项式积分

功能:
- 定义线性三角形(T3)和二次三角形(T6)的形函数及其导数
- 组装局部刚度矩阵与质量矩阵
- 计算楔形区域积分（用于边界层积分计算）
"""

import numpy as np
import math


def basis_mn_t3(t, n, p):
    """
    计算 T3（线性三角形）单元上所有基函数及其导数。
    融合自 413_fem2d_sample 的 basis_mn_t3。

    参数:
      t: (2,3) 三角形顶点坐标，逆时针排列
      n: 求值点个数
      p: (2,n) 求值点坐标

    返回:
      phi: (3,n) 基函数值
      dphidx: (3,n) x 方向导数
      dphidy: (3,n) y 方向导数

    数学公式:
      设三角形面积为 A（实际是 2*面积），则
      phi_1 = [(x3-x2)(y-y2) - (y3-y2)(x-x2)] / A
      phi_2 = [(x1-x3)(y-y3) - (y1-y3)(x-x3)] / A
      phi_3 = [(x2-x1)(y-y1) - (y2-y1)(x-x1)] / A
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(2, 1)
    elif p.shape[0] != 2 and p.shape[1] == 2:
        p = p.T  # 转换为 (2, n) 格式

    area = t[0, 0] * (t[1, 1] - t[1, 2]) \
         + t[0, 1] * (t[1, 2] - t[1, 0]) \
         + t[0, 2] * (t[1, 0] - t[1, 1])

    # 边界处理：退化三角形
    if abs(area) < 1e-15:
        area = 1e-15 * np.sign(area) if area != 0 else 1e-15

    phi = np.zeros((3, n), dtype=float)
    dphidx = np.zeros((3, n), dtype=float)
    dphidy = np.zeros((3, n), dtype=float)

    phi[0, :] = (t[0, 2] - t[0, 1]) * (p[1, :] - t[1, 1]) \
              - (t[1, 2] - t[1, 1]) * (p[0, :] - t[0, 1])
    dphidx[0, :] = -(t[1, 2] - t[1, 1])
    dphidy[0, :] =  (t[0, 2] - t[0, 1])

    phi[1, :] = (t[0, 0] - t[0, 2]) * (p[1, :] - t[1, 2]) \
              - (t[1, 0] - t[1, 2]) * (p[0, :] - t[0, 2])
    dphidx[1, :] = -(t[1, 0] - t[1, 2])
    dphidy[1, :] =  (t[0, 0] - t[0, 2])

    phi[2, :] = (t[0, 1] - t[0, 0]) * (p[1, :] - t[1, 0]) \
              - (t[1, 1] - t[1, 0]) * (p[0, :] - t[0, 0])
    dphidx[2, :] = -(t[1, 1] - t[1, 0])
    dphidy[2, :] =  (t[0, 1] - t[0, 0])

    phi /= area
    dphidx /= area
    dphidy /= area

    return phi, dphidx, dphidy


def basis_mn_t6(t, n, p):
    """
    计算 T6（二次三角形）单元上所有基函数及其导数。
    融合自 413_fem2d_sample 的 basis_mn_t6。

    参数:
      t: (2,6) 节点坐标（顶点+边中点）
      n: 求值点个数
      p: (2,n) 求值点坐标

    返回:
      phi: (6,n) 基函数值
      dphidx, dphidy: (6,n) 导数
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(2, 1)

    phi = np.zeros((6, n), dtype=float)
    dphidx = np.zeros((6, n), dtype=float)
    dphidy = np.zeros((6, n), dtype=float)

    # 辅助函数
    def compute_basis(idx, p1, p2, p3, p4, p5, p6):
        gx = (p[0, :] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p[1, :] - p1[1])
        gn = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        hx = (p[0, :] - p4[0]) * (p6[1] - p4[1]) - (p6[0] - p4[0]) * (p[1, :] - p4[1])
        hn = (p5[0] - p4[0]) * (p6[1] - p4[1]) - (p6[0] - p4[0]) * (p5[1] - p4[1])

        gn = np.where(np.abs(gn) < 1e-15, 1e-15, gn)
        hn = np.where(np.abs(hn) < 1e-15, 1e-15, hn)

        ph = (gx * hx) / (gn * hn)
        dpx = ((p3[1] - p1[1]) * hx + gx * (p6[1] - p4[1])) / (gn * hn)
        dpy = -((p3[0] - p1[0]) * hx + gx * (p6[0] - p4[0])) / (gn * hn)
        return ph, dpx, dpy

    # Basis 1
    phi[0, :], dphidx[0, :], dphidy[0, :] = compute_basis(
        0, t[:, 1], t[:, 0], t[:, 2], t[:, 3], t[:, 0], t[:, 5])
    # Basis 2
    phi[1, :], dphidx[1, :], dphidy[1, :] = compute_basis(
        1, t[:, 0], t[:, 1], t[:, 2], t[:, 4], t[:, 1], t[:, 3])
    # Basis 3
    phi[2, :], dphidx[2, :], dphidy[2, :] = compute_basis(
        2, t[:, 1], t[:, 2], t[:, 0], t[:, 5], t[:, 2], t[:, 4])
    # Basis 4
    phi[3, :], dphidx[3, :], dphidy[3, :] = compute_basis(
        3, t[:, 2], t[:, 0], t[:, 1], t[:, 4], t[:, 2], t[:, 3])
    # Basis 5
    phi[4, :], dphidx[4, :], dphidy[4, :] = compute_basis(
        4, t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 1], t[:, 4])
    # Basis 6
    phi[5, :], dphidx[5, :], dphidy[5, :] = compute_basis(
        5, t[:, 1], t[:, 2], t[:, 0], t[:, 5], t[:, 2], t[:, 3])

    return phi, dphidx, dphidy


def local_stiffness_matrix_t3(vertices, nu=1.0):
    """
    计算 T3 单元的局部刚度矩阵（扩散项）。

    数学模型:
      对于扩散算子 -nu * Laplacian(u)，局部刚度矩阵为
      K_{ij} = nu * integral_T (dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy) dOmega

      对于线性三角形，导数为常数，因此
      K_{ij} = nu * |T| * (dphi_i/dx * dphi_j/dx + dphi_i/dy * dphi_j/dy)
      其中 |T| = |area| / 2 为三角形面积。
    """
    vertices = np.asarray(vertices, dtype=float)
    area2 = vertices[0, 0] * (vertices[1, 1] - vertices[1, 2]) \
          + vertices[0, 1] * (vertices[1, 2] - vertices[1, 0]) \
          + vertices[0, 2] * (vertices[1, 0] - vertices[1, 1])
    area = abs(area2) * 0.5
    if area < 1e-15:
        area = 1e-15

    # 计算形函数导数（常数）
    _, dphidx, dphidy = basis_mn_t3(vertices, 1, vertices[:, 0:1])

    K = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            K[i, j] = nu * area * (dphidx[i, 0] * dphidx[j, 0] + dphidy[i, 0] * dphidy[j, 0])
    return K


def local_mass_matrix_t3(vertices):
    """
    计算 T3 单元的集中质量矩阵（Lumped mass）。

    数学模型:
      一致质量矩阵: M_{ij} = integral_T phi_i * phi_j dOmega
      对于 T3 单元，一致质量矩阵为
        M = (|T|/12) * [[2,1,1],[1,2,1],[1,1,2]]
      集中质量矩阵为对角阵:
        M_lump = (|T|/3) * I_3
    """
    vertices = np.asarray(vertices, dtype=float)
    area2 = vertices[0, 0] * (vertices[1, 1] - vertices[1, 2]) \
          + vertices[0, 1] * (vertices[1, 2] - vertices[1, 0]) \
          + vertices[0, 2] * (vertices[1, 0] - vertices[1, 1])
    area = abs(area2) * 0.5
    if area < 1e-15:
        area = 1e-15
    return (area / 3.0) * np.eye(3, dtype=float)


def wedge01_monomial_integral(e):
    """
    计算单位楔形区域上单项式的积分。
    融合自 1409_wedge_integrals 的 wedge01_monomial_integral。

    积分区域:
      0 <= x, 0 <= y, x + y <= 1, -1 <= z <= 1

    数学公式:
      对 e = [e1, e2, e3]，积分为
      I = (1 / ((e1+e2+2)(e1+e2+3))) * product_{i=1}^{e2} (i / (e1+i))
      若 e3 为奇数则 I = 0；若 e3 为偶数则乘以 2/(e3+1)。

    边界处理:
      e3 == -1 为非法输入，返回 0.0。
    """
    e = np.asarray(e, dtype=int)
    if e[2] == -1:
        return 0.0

    value = 1.0
    k = e[0]
    for i in range(1, e[1] + 1):
        k = k + 1
        value = value * i / k

    k = k + 1
    value = value / k
    k = k + 1
    value = value / k

    if e[2] % 2 == 1:
        value = 0.0
    else:
        value = value * 2.0 / (e[2] + 1)

    return float(value)


def wedge_boundary_layer_integral(nu, delta, order=4):
    """
    利用楔形区域积分计算边界层动量厚度。

    物理模型:
      在湍流边界层中，动量厚度 theta 可通过对速度剖面 u(z) 的积分估计：
        theta = integral_0^delta (u/U_inf) * (1 - u/U_inf) dz
      这里使用楔形积分框架近似计算边界层内的能量耗散。

    参数:
      nu: 运动粘度
      delta: 边界层厚度
      order: 积分阶数

    返回:
      边界层动量厚度估计值
    """
    theta = 0.0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            e = [i, j, 0]
            coeff = ((-1) ** j) * math.comb(order, i) * math.comb(order - i, j)
            theta += coeff * wedge01_monomial_integral(e)
    theta *= delta * np.sqrt(nu)
    return theta
