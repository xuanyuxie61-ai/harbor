# -*- coding: utf-8 -*-
"""
levelset_function.py
====================
水平集函数的构建、初始化、符号距离函数（SDF）转换与等值面提取。

融合原始项目:
  - 667_levels: 水平集/等高线概念（等值面提取思想）
  - 307_distance_to_position_sphere: 距离变换与球面几何（距离函数计算思想）

核心数学公式
------------
1. 水平集函数 φ(x,t): R² × [0,T] → R
   界面 Γ(t) = { x ∈ R² | φ(x,t) = 0 }
   内部 Ω⁻(t) = { x | φ(x,t) < 0 }
   外部 Ω⁺(t) = { x | φ(x,t) > 0 }

2. 符号距离函数 (Signed Distance Function):
   φ(x) =  dist(x, Γ)   if x ∈ Ω⁺
   φ(x) = -dist(x, Γ)   if x ∈ Ω⁻
   |∇φ| = 1 几乎处处成立

3. 球面距离变换（融入 distance_to_position_sphere 思想）:
   对球面 S² 上两点 p, q，大圆距离:
   d(p,q) = R · arccos( (p·q) / R² )
   此处将其推广到平面带权距离变换。

4. 法向量与曲率:
   n = ∇φ / |∇φ|
   κ = ∇ · n = ∇ · (∇φ / |∇φ|)
     = (φ_{xx} φ_y² - 2 φ_x φ_y φ_{xy} + φ_{yy} φ_x²) / (φ_x² + φ_y²)^{3/2}
"""

import numpy as np
from numerical_utils import central_diff_2nd, laplacian_2d


class LevelSetFunction:
    """
    二维水平集函数的封装类，支持笛卡尔网格上的初始化、
    符号距离函数构建与界面几何量计算。
    """

    def __init__(self, nx, ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
        """
        初始化水平集函数。
        参数:
            nx, ny : 网格点数
            xlim, ylim : 计算域边界
        """
        if nx < 5 or ny < 5:
            raise ValueError("LevelSetFunction: nx and ny must be at least 5")
        self.nx = nx
        self.ny = ny
        self.xlim = xlim
        self.ylim = ylim
        self.x = np.linspace(xlim[0], xlim[1], nx)
        self.y = np.linspace(ylim[0], ylim[1], ny)
        self.dx = (xlim[1] - xlim[0]) / (nx - 1)
        self.dy = (ylim[1] - ylim[0]) / (ny - 1)
        self.phi = np.zeros((nx, ny), dtype=np.float64)

    def init_circle(self, cx=0.0, cy=0.0, r=0.3):
        """
        初始化为圆的符号距离函数。
        φ(x,y) = √((x-cx)² + (y-cy)²) - r
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.phi = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r
        return self

    def init_ellipse(self, cx=0.0, cy=0.0, a=0.4, b=0.2, theta=0.0):
        """
        初始化为椭圆的符号距离函数（近似）。
        旋转后的椭圆方程:
        ((x-cx)cosθ + (y-cy)sinθ)²/a² + (-(x-cx)sinθ + (y-cy)cosθ)²/b² = 1
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        xr = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
        yr = -(X - cx) * np.sin(theta) + (Y - cy) * np.cos(theta)
        # 近似 SDF: 使用代数距离再修正
        d_alg = np.sqrt((xr / a) ** 2 + (yr / b) ** 2) - 1.0
        # 在边界附近用线性化近似
        self.phi = d_alg * np.minimum(a, b)
        return self

    def init_star_shape(self, cx=0.0, cy=0.0, r0=0.3, amp=0.05, n_peaks=5):
        """
        初始化为星形/花瓣形的符号距离函数。
        极坐标方程: r(θ) = r0 + amp · sin(n_peaks · θ)
        使用近似距离变换初始化。
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        theta = np.arctan2(Y - cy, X - cx)
        r_target = r0 + amp * np.sin(n_peaks * theta)
        r_actual = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # 近似 SDF
        self.phi = r_actual - r_target
        return self

    def init_two_circles(self, c1=(-0.3, 0.0), c2=(0.3, 0.0), r=0.25):
        """
        两个相交圆的并集（用于拓扑分裂测试）。
        φ = min(φ₁, φ₂)   (min 对应并集)
        其中 φ_i = √((x-cx_i)² + (y-cy_i)²) - r_i
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        phi1 = np.sqrt((X - c1[0]) ** 2 + (Y - c1[1]) ** 2) - r
        phi2 = np.sqrt((X - c2[0]) ** 2 + (Y - c2[1]) ** 2) - r
        self.phi = np.minimum(phi1, phi2)
        return self

    def init_rectangle(self, cx=0.0, cy=0.0, w=0.5, h=0.3):
        """
        矩形符号距离函数。
        到矩形边界的距离（考虑内部/外部）。
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        dx = np.maximum(np.abs(X - cx) - w / 2.0, 0.0)
        dy = np.maximum(np.abs(Y - cy) - h / 2.0, 0.0)
        # 外部距离
        dist_out = np.sqrt(dx ** 2 + dy ** 2)
        # 内部距离（取最大负值）
        dist_in = np.minimum(np.maximum(np.abs(X - cx) - w / 2.0, np.abs(Y - cy) - h / 2.0), 0.0)
        self.phi = dist_out + dist_in
        return self

    def compute_gradient_norm(self):
        """
        计算 |∇φ| 的二阶中心差分近似。
        |∇φ| = √(φ_x² + φ_y²)
        """
        phi = self.phi
        dx, dy = self.dx, self.dy
        phi_x = central_diff_2nd(phi, dx, axis=0)
        phi_y = central_diff_2nd(phi, dy, axis=1)
        grad_norm = np.sqrt(phi_x ** 2 + phi_y ** 2)
        # 避免除零
        grad_norm = np.maximum(grad_norm, 1e-12)
        return phi_x, phi_y, grad_norm

    def compute_curvature(self):
        """
        计算界面的平均曲率 κ。
        公式（二维）:
        κ = ∇·(∇φ/|∇φ|)
          = (φ_{xx} φ_y² - 2 φ_x φ_y φ_{xy} + φ_{yy} φ_x²) / (φ_x² + φ_y²)^{3/2}

        数值实现使用二阶中心差分，在边界处退化为内部值（Neumann）。
        """
        # HOLE_3: 实现二维水平集平均曲率计算
        # 需要计算一阶偏导 φ_x, φ_y 和二阶偏导 φ_xx, φ_yy, φ_xy
        # 最后按曲率公式组合并返回 κ
        raise NotImplementedError("HOLE_3: Curvature formula implementation missing")

    def compute_normal(self):
        """
        计算单位法向量 n = ∇φ / |∇φ|。
        """
        phi_x, phi_y, grad_norm = self.compute_gradient_norm()
        nx = phi_x / grad_norm
        ny = phi_y / grad_norm
        return nx, ny

    def signed_distance_redistancing_brute(self, max_iter=50, dtau=0.5):
        """
        基于 PDE 的粗粒度符号距离函数重初始化。
        求解:
            ∂φ/∂τ + sign(φ_0)(|∇φ| - 1) = 0
        采用显式迎风格式，dtau 为伪时间步长。

        融入 distance_to_position_sphere 的距离变换思想，
        将欧氏距离视为球面距离在半径趋于无穷时的极限。
        """
        phi0 = self.phi.copy()
        phi = phi0.copy()
        dx = self.dx
        ny_local = self.ny

        s0 = np.sign(phi0)
        s0 = np.where(s0 == 0, 1.0, s0)

        for _iter in range(max_iter):
            phi_new = phi.copy()
            for i in range(1, self.nx - 1):
                for j in range(1, ny_local - 1):
                    # 迎风格式导数
                    if s0[i, j] > 0:
                        dxp = phi[i + 1, j] - phi[i, j]
                        dxm = phi[i, j] - phi[i - 1, j]
                        dyp = phi[i, j + 1] - phi[i, j]
                        dym = phi[i, j] - phi[i, j - 1]
                    else:
                        dxp = phi[i, j] - phi[i - 1, j]
                        dxm = phi[i + 1, j] - phi[i, j]
                        dyp = phi[i, j] - phi[i, j - 1]
                        dym = phi[i, j + 1] - phi[i, j]

                    gx = np.maximum(dxp / dx, 0.0) ** 2 + np.minimum(dxm / dx, 0.0) ** 2
                    gy = np.maximum(dyp / dx, 0.0) ** 2 + np.minimum(dym / dx, 0.0) ** 2
                    grad_mag = np.sqrt(gx + gy)
                    phi_new[i, j] = phi[i, j] - dtau * s0[i, j] * (grad_mag - 1.0)
            phi = phi_new.copy()
            # 边界保持零阶外推
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]

        self.phi = phi
        return self

    def compute_volume(self):
        """
        计算界面内部区域的体积（面积）。
        V = ∫_{φ<0} dx dy ≈ Σ_{φ_{ij}<0} dx·dy
        """
        return np.sum(self.phi < 0) * self.dx * self.dy

    def compute_interface_length(self):
        """
        估计界面长度（周长）。
        使用零交叉点计数乘以网格尺寸作为一阶近似，
        更精确的做法可用:
        L = ∫ δ(φ) |∇φ| dx dy
        其中 δ(φ) ≈ 1/(2ε) · (1 + cos(πφ/ε)) for |φ|<ε
        """
        phi = self.phi
        dx, dy = self.dx, self.dy
        eps = 1.5 * max(dx, dy)
        delta = np.zeros_like(phi)
        mask = np.abs(phi) < eps
        delta[mask] = (1.0 / (2.0 * eps)) * (1.0 + np.cos(np.pi * phi[mask] / eps))
        _, _, grad_norm = self.compute_gradient_norm()
        L = np.sum(delta * grad_norm) * dx * dy
        return L

    def get_zero_levelset_points(self):
        """
        提取零等值线上的近似点（线性插值法）。
        对每条网格边，若两端 φ 值符号相反，则线性插值求零点。
        """
        points = []
        phi = self.phi
        nx, ny = self.nx, self.ny
        x = self.x
        y = self.y

        # 水平边
        for i in range(nx):
            for j in range(ny - 1):
                if phi[i, j] * phi[i, j + 1] < 0:
                    t = phi[i, j] / (phi[i, j] - phi[i, j + 1])
                    px = x[i]
                    py = y[j] + t * (y[j + 1] - y[j])
                    points.append((px, py))

        # 垂直边
        for i in range(nx - 1):
            for j in range(ny):
                if phi[i, j] * phi[i + 1, j] < 0:
                    t = phi[i, j] / (phi[i, j] - phi[i + 1, j])
                    px = x[i] + t * (x[i + 1] - x[i])
                    py = y[j]
                    points.append((px, py))

        return np.array(points) if points else np.zeros((0, 2))
