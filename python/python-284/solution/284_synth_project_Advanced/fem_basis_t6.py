# -*- coding: utf-8 -*-
"""
fem_basis_t6.py — 二次三角形有限元基函数
============================================
核心科学问题: 使用 T6 二次三角形单元离散化
二维薛定谔方程, 提供高阶精度.

融合种子项目:
  - 375_fem_basis_t6_display: T6 二次三角形基函数

数学基础:
  T6 单元有 6 个节点 (3 顶点 + 3 边中点).
  面积坐标 (重心坐标):
    L_1, L_2, L_3,  L_1+L_2+L_3=1

  基函数:
    φ_1 = L_1(2L_1-1)  (顶点 1)
    φ_2 = L_2(2L_2-1)  (顶点 2)
    φ_3 = L_3(2L_3-1)  (顶点 3)
    φ_4 = 4L_1L_2       (边 1-2 中点)
    φ_5 = 4L_2L_3       (边 2-3 中点)
    φ_6 = 4L_3L_1       (边 3-1 中点)
"""

import numpy as np


class T6Element:
    """
    T6 二次三角形有限元.
    """

    def __init__(self, nodes):
        """
        参数
        ----
        nodes : ndarray, shape (6, 2)
            6 个节点坐标 [(x1,y1), ..., (x6,y6)]
            节点编号: 1,2,3 顶点; 4,5,6 边中点
            4 在 1-2 边, 5 在 2-3 边, 6 在 3-1 边
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        assert self.nodes.shape == (6, 2)

        # 顶点坐标
        self.v1 = self.nodes[0]
        self.v2 = self.nodes[1]
        self.v3 = self.nodes[2]

        # 计算面积
        self.area = self._triangle_area(self.v1, self.v2, self.v3)

    def _triangle_area(self, p1, p2, p3):
        """三角形面积 = 0.5 * |det([p2-p1, p3-p1])|."""
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) -
                          (p3[0] - p1[0]) * (p2[1] - p1[1]))

    def area_coordinates(self, x, y):
        """
        计算面积坐标 (L1, L2, L3).

        L_i = (面积_i) / (总面积)

        其中 面积_i 是点 (x,y) 对边的三角形面积.
        """
        A = self.area
        if A < 1e-30:
            return 1.0/3, 1.0/3, 1.0/3

        x1, y1 = self.v1
        x2, y2 = self.v2
        x3, y3 = self.v3

        L1 = ((x2 - x) * (y3 - y1) - (x3 - x) * (y2 - y1)) / (2 * A)
        L2 = ((x3 - x) * (y1 - y2) - (x1 - x) * (y3 - y2)) / (2 * A)
        L3 = 1.0 - L1 - L2

        # 截断到 [0, 1]
        L1 = np.clip(L1, 0, 1)
        L2 = np.clip(L2, 0, 1)
        L3 = np.clip(L3, 0, 1)
        total = L1 + L2 + L3
        if total > 0:
            L1 /= total
            L2 /= total
            L3 /= total

        return L1, L2, L3

    def basis_values(self, x, y):
        """
        计算 6 个基函数在 (x,y) 处的值.

        φ_1 = L_1(2L_1-1)
        φ_2 = L_2(2L_2-1)
        φ_3 = L_3(2L_3-1)
        φ_4 = 4L_1L_2
        φ_5 = 4L_2L_3
        φ_6 = 4L_3L_1
        """
        L1, L2, L3 = self.area_coordinates(x, y)

        phi = np.zeros(6)
        phi[0] = L1 * (2 * L1 - 1)
        phi[1] = L2 * (2 * L2 - 1)
        phi[2] = L3 * (2 * L3 - 1)
        phi[3] = 4 * L1 * L2
        phi[4] = 4 * L2 * L3
        phi[5] = 4 * L3 * L1

        return phi

    def basis_gradients(self, x, y):
        """
        计算基函数梯度 dφ/dx, dφ/dy.

        dL_i/dx = b_i / (2A), dL_i/dy = c_i / (2A)

        其中:
          b_1 = y_2 - y_3, c_1 = x_3 - x_2
          b_2 = y_3 - y_1, c_2 = x_1 - x_3
          b_3 = y_1 - y_2, c_3 = x_2 - x_1

        dφ_1/dx = (4L_1-1)*dL_1/dx
        dφ_4/dx = 4*(L_2*dL_1/dx + L_1*dL_2/dx)
        """
        L1, L2, L3 = self.area_coordinates(x, y)
        A = self.area
        if A < 1e-30:
            return np.zeros((6, 2))

        x1, y1 = self.v1
        x2, y2 = self.v2
        x3, y3 = self.v3

        # dL/dx, dL/dy
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])

        dLdx = b / (2 * A)
        dLdy = c / (2 * A)

        # 基函数梯度
        grad = np.zeros((6, 2))

        # 顶点
        grad[0, 0] = (4 * L1 - 1) * dLdx[0]
        grad[0, 1] = (4 * L1 - 1) * dLdy[0]
        grad[1, 0] = (4 * L2 - 1) * dLdx[1]
        grad[1, 1] = (4 * L2 - 1) * dLdy[1]
        grad[2, 0] = (4 * L3 - 1) * dLdx[2]
        grad[2, 1] = (4 * L3 - 1) * dLdy[2]

        # 边中点
        grad[3, 0] = 4 * (L2 * dLdx[0] + L1 * dLdx[1])
        grad[3, 1] = 4 * (L2 * dLdy[0] + L1 * dLdy[1])
        grad[4, 0] = 4 * (L3 * dLdx[1] + L2 * dLdx[2])
        grad[4, 1] = 4 * (L3 * dLdy[1] + L2 * dLdy[2])
        grad[5, 0] = 4 * (L1 * dLdx[2] + L3 * dLdx[0])
        grad[5, 1] = 4 * (L1 * dLdy[2] + L3 * dLdy[0])

        return grad

    def stiffness_matrix(self, m_star_inv=1.0):
        """
        单元刚度矩阵:
            K_{ij} = integral_Ω (ℏ²/(2m*)) * ∇φ_i · ∇φ_j dA

        用 7 点高斯求积.
        """
        K = np.zeros((6, 6))

        # 7 点高斯规则 (三角形)
        xi_gp, eta_gp, w_gp = self._gauss_points_triangle(7)

        for gp in range(len(w_gp)):
            xi, eta = xi_gp[gp], eta_gp[gp]
            # 物理坐标
            x = self.v1[0] + xi * (self.v2[0] - self.v1[0]) + eta * (self.v3[0] - self.v1[0])
            y = self.v1[1] + xi * (self.v2[1] - self.v1[1]) + eta * (self.v3[1] - self.v1[1])

            grad = self.basis_gradients(x, y)

            for i in range(6):
                for j in range(6):
                    K[i, j] += w_gp[gp] * m_star_inv * (
                        grad[i, 0] * grad[j, 0] + grad[i, 1] * grad[j, 1])

        K *= self.area
        return K

    def mass_matrix(self):
        """
        单元质量矩阵:
            M_{ij} = integral_Ω φ_i * φ_j dA
        """
        M = np.zeros((6, 6))

        xi_gp, eta_gp, w_gp = self._gauss_points_triangle(7)

        for gp in range(len(w_gp)):
            xi, eta = xi_gp[gp], eta_gp[gp]
            x = self.v1[0] + xi * (self.v2[0] - self.v1[0]) + eta * (self.v3[0] - self.v1[0])
            y = self.v1[1] + xi * (self.v2[1] - self.v1[1]) + eta * (self.v3[1] - self.v1[1])

            phi = self.basis_values(x, y)

            for i in range(6):
                for j in range(6):
                    M[i, j] += w_gp[gp] * phi[i] * phi[j]

        M *= self.area
        return M

    def _gauss_points_triangle(self, n_pts):
        """
        三角形高斯求积点和权重.
        """
        if n_pts == 1:
            return np.array([1.0/3]), np.array([1.0/3]), np.array([1.0])
        elif n_pts == 3:
            xi = np.array([1.0/6, 2.0/3, 1.0/6])
            eta = np.array([1.0/6, 1.0/6, 2.0/3])
            w = np.array([1.0/3, 1.0/3, 1.0/3])
            return xi, eta, w
        elif n_pts == 7:
            a1 = 0.059715871789770
            b1 = 0.470142064105115
            a2 = 0.797426985353087
            b2 = 0.101286507323456
            w0 = 0.225000000000000
            w1 = 0.132394152788506
            w2 = 0.125939180544827

            xi = np.array([1.0/3, a1, b1, a1, b1, b1, a1])
            eta = np.array([1.0/3, a1, a1, b1, b1, a1, b1])
            w = np.array([w0, w1, w1, w1, w1, w2, w2])
            return xi, eta, w
        else:
            # 默认 3 点
            return self._gauss_points_triangle(3)

    def contains_point(self, x, y):
        """检查点 (x,y) 是否在单元内."""
        L1, L2, L3 = self.area_coordinates(x, y)
        return L1 >= -1e-10 and L2 >= -1e-10 and L3 >= -1e-10
