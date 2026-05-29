"""
interface_tracking.py
=====================
界面追踪与几何分析模块

基于相场序参量 φ 的等值面提取界面位置，计算界面速度、
曲率、法向量等几何量。界面定义为 φ = 0 的等值线（二维）
或等值面（三维）。

核心公式：
    界面位置：Γ(t) = {x | φ(x, t) = 0}
    界面法向量：n = ∇φ / |∇φ|
    界面曲率：κ = ∇·n = ∇·(∇φ / |∇φ|)
    界面速度：V_n = -φ_t / |∇φ|

Level Set 方程：
    ∂φ/∂t + V_n |∇φ| = 0
"""

import numpy as np


class InterfaceTracker:
    """
    基于相场的界面追踪器。
    """

    def __init__(self, nx, ny, dx, dy):
        """
        初始化界面追踪器。

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

    def compute_gradient(self, phi):
        """
        计算 φ 的梯度 ∇φ，采用中心差分。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        tuple of ndarray
            (grad_x, grad_y) 梯度分量。
        """
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)

        grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * self.dx)
        grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * self.dy)

        # 边界：前向/后向差分
        grad_x[0, :] = (phi[1, :] - phi[0, :]) / self.dx
        grad_x[-1, :] = (phi[-1, :] - phi[-2, :]) / self.dx
        grad_y[:, 0] = (phi[:, 1] - phi[:, 0]) / self.dy
        grad_y[:, -1] = (phi[:, -1] - phi[:, -2]) / self.dy

        return grad_x, grad_y

    def compute_gradient_magnitude(self, phi):
        """
        计算梯度模长 |∇φ|。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            |∇φ|。
        """
        gx, gy = self.compute_gradient(phi)
        return np.sqrt(gx ** 2 + gy ** 2)

    def compute_normal(self, phi):
        """
        计算界面单位法向量 n = ∇φ / |∇φ|。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        tuple of ndarray
            (n_x, n_y) 法向量分量。
        """
        gx, gy = self.compute_gradient(phi)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = np.maximum(grad_mag, 1e-12)
        return gx / grad_mag, gy / grad_mag

    def compute_curvature(self, phi):
        """
        计算界面平均曲率 κ = ∇·(∇φ / |∇φ|)。

        采用守恒型格式：
            κ = [∂/∂x(gx/|∇φ|) + ∂/∂y(gy/|∇φ|)]

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        ndarray
            曲率场。
        """
        gx, gy = self.compute_gradient(phi)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mag = np.maximum(grad_mag, 1e-12)

        nx = gx / grad_mag
        ny = gy / grad_mag

        # 计算 div(n)
        dnx_dx = np.zeros_like(nx)
        dny_dy = np.zeros_like(ny)

        dnx_dx[1:-1, :] = (nx[2:, :] - nx[:-2, :]) / (2.0 * self.dx)
        dny_dy[:, 1:-1] = (ny[:, 2:] - ny[:, :-2]) / (2.0 * self.dy)

        curvature = dnx_dx + dny_dy

        # 仅在界面附近保留
        interface_mask = np.abs(phi) < 0.9
        return curvature * interface_mask

    def compute_interface_velocity(self, phi_old, phi_new, dt):
        """
        计算界面法向速度：
            V_n = -(φ_new - φ_old) / (dt * |∇φ|)

        Parameters
        ----------
        phi_old, phi_new : ndarray
            两个时刻的序参量场。
        dt : float
            时间步长。

        Returns
        -------
        ndarray
            界面法向速度场。
        """
        dphi_dt = (phi_new - phi_old) / dt
        grad_mag = self.compute_gradient_magnitude(phi_new)
        grad_mag = np.maximum(grad_mag, 1e-12)
        V_n = -dphi_dt / grad_mag

        # 仅在界面附近保留
        interface_mask = np.abs(phi_new) < 0.9
        return V_n * interface_mask

    def extract_interface_points(self, phi, threshold=0.0):
        """
        使用线性插值提取 φ = threshold 的等值线点。

        遍历每个网格边，若两端点 φ 值跨越 threshold，
        则线性插值求出交点位置。

        Parameters
        ----------
        phi : ndarray
            序参量场。
        threshold : float
            等值线阈值，默认为 0（界面）。

        Returns
        -------
        list of tuple
            等值线点列表 [(x1, y1), (x2, y2), ...]。
        """
        points = []
        x_coords = np.linspace(0, (self.nx - 1) * self.dx, self.nx)
        y_coords = np.linspace(0, (self.ny - 1) * self.dy, self.ny)

        # 水平边 (i, j) 到 (i+1, j)
        for j in range(self.ny):
            for i in range(self.nx - 1):
                val1 = phi[i, j] - threshold
                val2 = phi[i + 1, j] - threshold
                if val1 * val2 < 0:
                    # 线性插值
                    t = abs(val1) / (abs(val1) + abs(val2))
                    x = x_coords[i] + t * (x_coords[i + 1] - x_coords[i])
                    y = y_coords[j]
                    points.append((x, y))

        # 垂直边 (i, j) 到 (i, j+1)
        for i in range(self.nx):
            for j in range(self.ny - 1):
                val1 = phi[i, j] - threshold
                val2 = phi[i, j + 1] - threshold
                if val1 * val2 < 0:
                    t = abs(val1) / (abs(val1) + abs(val2))
                    x = x_coords[i]
                    y = y_coords[j] + t * (y_coords[j + 1] - y_coords[j])
                    points.append((x, y))

        return points

    def compute_interface_area(self, phi):
        """
        计算界面面积（二维为界面长度）：
            A = ∫ δ(φ) |∇φ| dΩ

        其中 δ(φ) 为 Dirac delta 函数的相场近似：
            δ(φ) = (1/(2ε)) (1 - tanh²(φ/ε))

        或使用更光滑的近似：
            δ(φ) = (15/(16ε)) (1 - (φ/ε)²)²  for |φ| < ε

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        float
            界面面积（长度）。
        """
        # 使用 tanh 型 delta 近似，宽度与相场界面宽度相关
        epsilon = 0.03  # delta 函数宽度
        delta_approx = (1.0 / (2.0 * epsilon)) * (1.0 - np.tanh(phi / epsilon) ** 2)
        grad_mag = self.compute_gradient_magnitude(phi)

        area = np.sum(delta_approx * grad_mag) * self.dx * self.dy
        return area

    def compute_morphology_number(self, phi):
        """
        计算形态学数（Morphology Number），表征界面复杂度：
            M = (界面长度)² / (4π × 固相面积)

        M = 1 为圆形，M > 1 为更复杂形态。

        Parameters
        ----------
        phi : ndarray
            序参量场。

        Returns
        -------
        float
            形态学数。
        """
        interface_length = self.compute_interface_area(phi)

        # 固相面积（φ > 0 的区域）
        solid_area = np.sum(phi > 0) * self.dx * self.dy
        if solid_area < 1e-12:
            return float('inf')

        morphology = (interface_length ** 2) / (4.0 * np.pi * solid_area)
        return morphology

    def tip_velocity_dendrite(self, phi_old, phi_new, dt):
        """
        对于枝晶生长，计算尖端速度。
        枝晶尖端定义为界面最突出的点（φ = 0 且曲率最大负值处）。

        Parameters
        ----------
        phi_old, phi_new : ndarray
            两个时刻的序参量场。
        dt : float
            时间步长。

        Returns
        -------
        float
            尖端速度（若找不到尖端返回 0）。
        """
        curvature = self.compute_curvature(phi_new)
        V_n = self.compute_interface_velocity(phi_old, phi_new, dt)

        # 在界面附近找曲率最大的负值（最尖锐处）
        interface_mask = np.abs(phi_new) < 0.5
        if not np.any(interface_mask):
            return 0.0

        # 在界面mask区域内找最小曲率
        curv_interface = np.where(interface_mask, curvature, 0.0)
        min_curv = np.min(curv_interface)

        # 在曲率接近最小值的区域取平均速度
        tip_mask = interface_mask & (curvature < min_curv * 0.8)
        if np.any(tip_mask):
            tip_velocity = np.mean(V_n[tip_mask])
            return tip_velocity
        return 0.0
