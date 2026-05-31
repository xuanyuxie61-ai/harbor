# -*- coding: utf-8 -*-
"""
curvature_flow.py
=================
曲率驱动流（Mean Curvature Flow）的高精度数值实现，
以及基于球面 Lebedev 求积的曲率相关积分运算。

融合原始项目:
  - 1120_sphere_lebedev_rule: 球面 Lebedev 高斯求积规则

核心数学公式
------------
1. 平均曲率流 (Mean Curvature Flow):
   ∂X/∂t = κ n
   水平集形式:
   ∂φ/∂t = κ |∇φ|
   其中 κ = ∇·(∇φ/|∇φ|) 为平均曲率。

2. Willmore 流（高阶曲率流）:
   ∂X/∂t = -(Δ_s κ + ½ κ³) n
   其中 Δ_s 为曲面 Laplacian (Laplace-Beltrami 算子)。
   水平集形式:
   ∂φ/∂t = (Δ_s κ + ½ κ³) |∇φ|

3. 球面 Lebedev 求积规则:
   ∫_{S²} f(θ,φ) dΩ ≈ 4π Σ_{i=1}^{N} w_i f(θ_i, φ_i)
   其中 (θ_i, φ_i) 为 Lebedev 节点，w_i 为权重。
   该规则对球谐函数 Y_l^m 具有极高的代数精度。

4. 曲率在球坐标下的表达:
   对水平集函数 φ(r,θ,φ)，在球面 r=R 上的平均曲率:
   κ = (1/R²) Δ_{S²} φ / |∇_{S²} φ|
   其中 Δ_{S²} 为球面 Laplacian。

5. 球面 Laplacian (Laplace-Beltrami):
   Δ_{S²} f = (1/sinθ) ∂/∂θ(sinθ ∂f/∂θ) + (1/sin²θ) ∂²f/∂φ²

6. 面积元与曲率积分:
   对嵌入 R³ 中的曲面 Σ，Willmore 能量:
   W = ∫_Σ κ² dA
   在水平集框架下:
   W = ∫_{R³} κ² δ(φ) |∇φ| dx
"""

import numpy as np


# Lebedev 规则数据子集（6 点与 14 点规则，用于高精度曲率积分）
# 数据源自 1120_sphere_lebedev_rule 的 ld0006 与 ld0014
_LEBEDEV_6_X = np.array([ 0.0,  0.0,  0.0,  0.0,  1.0, -1.0], dtype=np.float64)
_LEBEDEV_6_Y = np.array([ 0.0,  0.0,  1.0, -1.0,  0.0,  0.0], dtype=np.float64)
_LEBEDEV_6_Z = np.array([ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float64)
_LEBEDEV_6_W = np.array([0.1666666666666667, 0.1666666666666667,
                         0.1666666666666667, 0.1666666666666667,
                         0.1666666666666667, 0.1666666666666667], dtype=np.float64)

_LEBEDEV_14_X = np.array([
    0.0, 0.0, 0.0, 0.0, 1.0, -1.0,
    0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258, -0.5773502691896258,
    0.5773502691896258, -0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_Y = np.array([
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0,
    0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_Z = np.array([
    1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    0.5773502691896258, -0.5773502691896258, -0.5773502691896258,
    0.5773502691896258,  0.5773502691896258, -0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_W = np.array([
    0.0666666666666667, 0.0666666666666667, 0.0666666666666667, 0.0666666666666667,
    0.0666666666666667, 0.0666666666666667,
    0.0750000000000000, 0.0750000000000000, 0.0750000000000000, 0.0750000000000000,
    0.0750000000000000, 0.0750000000000000, 0.0750000000000000, 0.0750000000000000
], dtype=np.float64)


def lebedev_by_order(order):
    """
    返回指定阶数的 Lebedev 球面求积节点与权重。
    参数:
        order : int, 求积阶数（支持 6 或 14）
    返回:
        x, y, z, w : ndarray
    """
    if order == 6:
        return _LEBEDEV_6_X.copy(), _LEBEDEV_6_Y.copy(), _LEBEDEV_6_Z.copy(), _LEBEDEV_6_W.copy()
    elif order == 14:
        return _LEBEDEV_14_X.copy(), _LEBEDEV_14_Y.copy(), _LEBEDEV_14_Z.copy(), _LEBEDEV_14_W.copy()
    else:
        raise ValueError(f"lebedev_by_order: unsupported order {order}, use 6 or 14")


class CurvatureFlow:
    """
    曲率驱动流求解器，支持平均曲率流与 Willmore 流。
    """

    def __init__(self, levelset):
        self.ls = levelset

    def compute_mean_curvature_flow_velocity(self):
        """
        计算平均曲率流速度场:
        V_n = κ
        返回 κ 的 ndarray。
        """
        return self.ls.compute_curvature()

    def compute_willmore_flow_rhs(self):
        """
        计算 Willmore 流右端项:
        L(φ) = (Δ_s κ + ½ κ³) |∇φ|
        其中 Δ_s κ 为曲率的曲面 Laplacian。

        数值实现:
        1. 先计算 κ
        2. 在零等值线附近带状区域计算 Δ_s κ
        3. 组合得到 Willmore 流速度
        """
        kappa = self.ls.compute_curvature()
        phi = self.ls.phi
        dx, dy = self.ls.dx, self.ls.dy
        nx, ny = self.ls.nx, self.ls.ny

        # 计算曲率的 Laplacian（近似曲面 Laplacian）
        lap_kappa = np.zeros_like(kappa)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                lap_kappa[i, j] = (kappa[i + 1, j] - 2.0 * kappa[i, j] + kappa[i - 1, j]) / (dx * dx) \
                                  + (kappa[i, j + 1] - 2.0 * kappa[i, j] + kappa[i, j - 1]) / (dy * dy)

        lap_kappa[0, :] = lap_kappa[1, :]
        lap_kappa[-1, :] = lap_kappa[-2, :]
        lap_kappa[:, 0] = lap_kappa[:, 1]
        lap_kappa[:, -1] = lap_kappa[:, -2]

        _, _, grad_norm = self.ls.compute_gradient_norm()
        rhs = (lap_kappa + 0.5 * kappa ** 3) * grad_norm
        # 限制幅值
        rhs = np.clip(rhs, -1e4, 1e4)
        return rhs

    @staticmethod
    def integrate_on_sphere_surface(f_vals, order=14):
        """
        使用 Lebedev 规则在球面上积分函数 f。
        ∫_{S²} f dΩ ≈ 4π Σ w_i f_i

        参数:
            f_vals : ndarray, 在 Lebedev 节点上的函数值
            order  : 求积阶数
        返回:
            integral : float
        """
        x, y, z, w = lebedev_by_order(order)
        if len(f_vals) != len(w):
            raise ValueError("integrate_on_sphere_surface: f_vals length must match quadrature nodes")
        integral = 4.0 * np.pi * np.sum(w * f_vals)
        return integral

    @staticmethod
    def sphere_distance(lat1, lon1, lat2, lon2, R=1.0):
        """
        球面上两点间的大圆距离（Haversine 公式）。
        融入 distance_to_position_sphere 的球面距离思想。

        d = R · arccos( sinφ₁ sinφ₂ + cosφ₁ cosφ₂ cos(Δλ) )
        """
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return R * c

    def compute_willmore_energy(self):
        """
        计算 Willmore 能量 W = ∫_Σ κ² dA。
        在水平集框架下使用平滑 Delta 函数近似:
        W ≈ ∫ κ² δ_ε(φ) |∇φ| dx dy
        """
        phi = self.ls.phi
        kappa = self.ls.compute_curvature()
        _, _, grad_norm = self.ls.compute_gradient_norm()
        dx, dy = self.ls.dx, self.ls.dy
        eps = 1.5 * max(dx, dy)

        delta = np.zeros_like(phi)
        mask = np.abs(phi) < eps
        delta[mask] = (1.0 / (2.0 * eps)) * (1.0 + np.cos(np.pi * phi[mask] / eps))

        W = np.sum(kappa ** 2 * delta * grad_norm) * dx * dy
        return W

    def compute_surface_area(self):
        """
        计算界面面积（长度）。
        A = ∫ δ(φ) |∇φ| dx dy
        """
        return self.ls.compute_interface_length()

    def compute_gauss_map_variance(self):
        """
        计算高斯映射的方差，衡量界面几何复杂性。
        对凸闭曲线，高斯映射将每点法向量映射到 S¹。
        方差小表示形状接近圆。
        """
        nx_vec, ny_vec = self.ls.compute_normal()
        # 法向量角度
        theta = np.arctan2(ny_vec, nx_vec)
        # 只在界面附近统计
        phi = self.ls.phi
        eps = 1.5 * max(self.ls.dx, self.ls.dy)
        mask = np.abs(phi) < eps
        if np.sum(mask) == 0:
            return 0.0
        theta_masked = theta[mask]
        var = np.var(theta_masked)
        return var
