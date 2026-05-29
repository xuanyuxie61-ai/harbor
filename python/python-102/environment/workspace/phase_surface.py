"""
phase_surface.py
================
基于极小曲面（Minimal Surface）理论构造超构表面的平滑连续相位分布。

本模块源自项目 768_minimal_surface_exact（悬链面、螺旋面、Scherk 面
等极小曲面精确解）的核心思想，将平均曲率为零的极小曲面理论
应用于超构表面相位轮廓的平滑化与优化。

科学背景：
在超构表面设计中，离散的纳米柱阵列实现的相位分布往往是不连续的，
导致高阶衍射和散射损耗。为了获得高性能器件，需要一种“最平滑”的
连续相位过渡。

极小曲面是平均曲率 H = 0  everywhere 的曲面，满足 Plateau 问题的
Euler-Lagrange 方程：
    (1 + u_y²) u_xx - 2 u_x u_y u_xy + (1 + u_x²) u_yy = 0

这个方程恰好描述了“面积最小化”或“曲率最小化”的表面。
将其类比到相位分布 Φ(x,y)，我们寻求使总“曲率能量”最小的相位面：
    E[Φ] = ∫ ( |∇²Φ|² + λ |∇Φ - ∇Φ_target|² ) dA

其中第一项为弯曲能（对应极小曲面条件），第二项为 fidelity 项，
λ 为拉格朗日乘子。

本模块实现：
1. 极小曲面型相位平滑（基于平均曲率流）
2. 悬链面/螺旋面型相位轮廓生成
3. 相位面的曲率分析与质量评估
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


class PhaseSurface:
    """
    基于极小曲面理论的超构表面相位轮廓生成器。
    """

    def __init__(self, x_grid, y_grid):
        self.x = np.array(x_grid, dtype=np.float64)
        self.y = np.array(y_grid, dtype=np.float64)
        self.nx = len(x_grid)
        self.ny = len(y_grid)
        self.dx = x_grid[1] - x_grid[0] if self.nx > 1 else 1.0
        self.dy = y_grid[1] - y_grid[0] if self.ny > 1 else 1.0
        self.k0 = 2.0 * np.pi / 1.55e-6

    # ------------------------------------------------------------------
    # 极小曲面平滑（平均曲率流）
    # ------------------------------------------------------------------
    def minimal_surface_smooth(self, phi_init, lambda_fidelity=0.1,
                                max_iter=100, dt=0.1):
        """
        使用平均曲率流（Mean Curvature Flow）平滑初始相位分布。

        演化方程：
            ∂Φ/∂t = H + λ (Φ_target - Φ)
        其中 H 为平均曲率：
            H = [(1+Φ_y²) Φ_xx - 2 Φ_x Φ_y Φ_xy + (1+Φ_x²) Φ_yy]
                / [2 (1 + Φ_x² + Φ_y²)^{3/2}]

        离散化后使用隐式欧拉法求解。

        Parameters
        ----------
        phi_init : ndarray, shape (nx, ny)
            初始相位分布（如离散纳米柱的阶梯相位）
        lambda_fidelity : float
            fidelity 权重
        max_iter : int
            最大迭代步数
        dt : float
            伪时间步长

        Returns
        -------
        phi_smooth : ndarray, shape (nx, ny)
            平滑后的相位分布
        """
        phi = phi_init.copy()
        for it in range(max_iter):
            # 计算梯度
            dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
            dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
            dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
            dxy_phi = np.gradient(dx_phi, self.dy, axis=1)

            # 平均曲率（归一化前），梯度裁剪防止溢出
            grad_max = 1e6
            dx_phi = np.clip(dx_phi, -grad_max, grad_max)
            dy_phi = np.clip(dy_phi, -grad_max, grad_max)
            numerator = (1.0 + dy_phi ** 2) * dxx_phi \
                        - 2.0 * dx_phi * dy_phi * dxy_phi \
                        + (1.0 + dx_phi ** 2) * dyy_phi
            denom_sq = 1.0 + dx_phi ** 2 + dy_phi ** 2
            denom_sq = np.clip(denom_sq, 0.0, 1e12)
            denominator = 2.0 * denom_sq ** 1.5
            denominator = np.clip(denominator, 1e-15, 1e18)
            H = np.zeros_like(phi)
            mask = np.isfinite(denominator) & (np.abs(numerator) < 1e18)
            H[mask] = numerator[mask] / denominator[mask]

            # 显式更新（带 fidelity）
            phi_new = phi + dt * (H + lambda_fidelity * (phi_init - phi))

            # 边界保持（Dirichlet）
            phi_new[0, :] = phi_init[0, :]
            phi_new[-1, :] = phi_init[-1, :]
            phi_new[:, 0] = phi_init[:, 0]
            phi_new[:, -1] = phi_init[:, -1]

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new
            if diff < 1e-8:
                print(f"[phase_surface] 平均曲率流收敛于迭代 {it}")
                break

        return phi

    # ------------------------------------------------------------------
    # 悬链面型相位轮廓（源自 768_minimal_surface_exact）
    # ------------------------------------------------------------------
    def catenoid_phase_profile(self, a_param, center=(0.0, 0.0)):
        """
        生成悬链面（Catenoid）型相位轮廓。

        悬链面是极小曲面的经典解，参数方程为：
            X(u,v) = a cosh(v/a) cos(u)
            Y(u,v) = a cosh(v/a) sin(u)
            Z(u,v) = a v

        将其映射为相位分布：
            Φ(x,y) = Φ₀ * acosh( a * sqrt((x-cx)² + (y-cy)²) ) / a

        Parameters
        ----------
        a_param : float
            悬链面形状参数 [m]
        center : tuple
            中心位置
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        cx, cy = center
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # 防止 R < 1/a 导致 acosh 无定义
        R = np.maximum(R, 1.01 / a_param)
        phi = self.k0 * np.arccosh(a_param * R) / a_param
        return phi

    def helicoid_phase_profile(self, a_param, center=(0.0, 0.0)):
        """
        生成螺旋面（Helicoid）型相位轮廓。

        螺旋面参数方程：
            X(u,v) = v cos(u)
            Y(u,v) = v sin(u)
            Z(u,v) = a u

        映射为螺旋相位（光学涡旋）：
            Φ(x,y) = l * arctan2(y - cy, x - cx)
        其中 l 为拓扑荷（这里用 a_param 控制）。
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        cx, cy = center
        phi = a_param * np.arctan2(Y - cy, X - cx)
        return phi

    def scherk_phase_profile(self, a_param):
        """
        生成 Scherk 第一曲面型相位轮廓。

        Scherk 面方程：
            Z(x,y) = (1/a) ln( cos(ax) / cos(ay) )

        映射为相位：
            Φ(x,y) = Φ₀ * ln( cos(a x) / cos(a y) ) / a
        """
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        # 限制在 cos > 0 的区域
        cx = np.cos(a_param * X)
        cy_ = np.cos(a_param * Y)
        cx = np.maximum(cx, 1e-6)
        cy_ = np.maximum(cy_, 1e-6)
        phi = self.k0 * np.log(cx / cy_) / a_param
        return phi

    # ------------------------------------------------------------------
    # 曲率分析与质量评估
    # ------------------------------------------------------------------
    def compute_mean_curvature(self, phi):
        """
        计算相位分布作为曲面 z = Φ(x,y) 的平均曲率 H。
        """
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
        dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
        dxy_phi = np.gradient(dx_phi, self.dy, axis=1)

        dx_phi = np.clip(dx_phi, -1e6, 1e6)
        dy_phi = np.clip(dy_phi, -1e6, 1e6)
        numerator = (1.0 + dy_phi ** 2) * dxx_phi \
                    - 2.0 * dx_phi * dy_phi * dxy_phi \
                    + (1.0 + dx_phi ** 2) * dyy_phi
        denom_sq = np.clip(1.0 + dx_phi ** 2 + dy_phi ** 2, 0.0, 1e12)
        denominator = 2.0 * denom_sq ** 1.5
        denominator = np.clip(denominator, 1e-15, 1e18)
        H = np.zeros_like(phi)
        mask = np.isfinite(denominator) & (np.abs(numerator) < 1e18)
        H[mask] = numerator[mask] / denominator[mask]
        return H

    def compute_gaussian_curvature(self, phi):
        """
        计算高斯曲率 K。
        """
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dxx_phi = np.gradient(dx_phi, self.dx, axis=0)
        dyy_phi = np.gradient(dy_phi, self.dy, axis=1)
        dxy_phi = np.gradient(dx_phi, self.dy, axis=1)

        dx_phi = np.clip(dx_phi, -1e6, 1e6)
        dy_phi = np.clip(dy_phi, -1e6, 1e6)
        E = 1.0 + dx_phi ** 2
        F = dx_phi * dy_phi
        G = 1.0 + dy_phi ** 2
        denom_sqrt = np.sqrt(np.clip(1.0 + dx_phi ** 2 + dy_phi ** 2, 1e-15, 1e12))
        L = dxx_phi / denom_sqrt
        M = dxy_phi / denom_sqrt
        N = dyy_phi / denom_sqrt

        denom = E * G - F ** 2
        denom = np.clip(denom, 1e-15, 1e18)
        K = np.zeros_like(phi)
        mask = np.isfinite(denom)
        K[mask] = (L[mask] * N[mask] - M[mask] ** 2) / denom[mask]
        return K

    def surface_energy(self, phi):
        """
        计算相位面的 Willmore 能量（弯曲能）：
            W = ∫ H² dA
        """
        H = self.compute_mean_curvature(phi)
        dx_phi, dy_phi = np.gradient(phi, self.dx, self.dy)
        dA = np.sqrt(1.0 + dx_phi ** 2 + dy_phi ** 2) * self.dx * self.dy
        energy = np.sum(H ** 2 * dA)
        return energy

    def laplacian_smooth(self, phi_init, n_iter=50):
        """
        使用 Laplacian 平滑作为对比方法。
        迭代：Φ^{new} = Φ + α ΔΦ
        """
        phi = phi_init.copy()
        alpha = 0.1
        for _ in range(n_iter):
            lap = np.zeros_like(phi)
            lap[1:-1, 1:-1] = (
                phi[2:, 1:-1] + phi[:-2, 1:-1] +
                phi[1:-1, 2:] + phi[1:-1, :-2] - 4 * phi[1:-1, 1:-1]
            ) / (self.dx * self.dy)
            phi = phi + alpha * lap
            # 边界固定
            phi[0, :] = phi_init[0, :]
            phi[-1, :] = phi_init[-1, :]
            phi[:, 0] = phi_init[:, 0]
            phi[:, -1] = phi_init[:, -1]
        return phi


def demo():
    """演示：生成极小曲面型相位轮廓并分析曲率。"""
    nx, ny = 65, 65
    x = np.linspace(-3e-6, 3e-6, nx)
    y = np.linspace(-3e-6, 3e-6, ny)

    ps = PhaseSurface(x, y)

    # 1. 悬链面相位
    phi_cat = ps.catenoid_phase_profile(a_param=2.0e6, center=(0.0, 0.0))
    H_cat = ps.compute_mean_curvature(phi_cat)
    W_cat = ps.surface_energy(phi_cat)
    print(f"[phase_surface] 悬链面相位: max H = {np.max(np.abs(H_cat)):.3e}")
    print(f"[phase_surface] 悬链面 Willmore 能量: {W_cat:.4e}")

    # 2. 螺旋面相位（光学涡旋）
    phi_hel = ps.helicoid_phase_profile(a_param=2.0, center=(0.0, 0.0))
    H_hel = ps.compute_mean_curvature(phi_hel)
    W_hel = ps.surface_energy(phi_hel)
    print(f"[phase_surface] 螺旋面相位: max H = {np.max(np.abs(H_hel)):.3e}")
    print(f"[phase_surface] 螺旋面 Willmore 能量: {W_hel:.4e}")

    # 3. 离散相位平滑
    # 构造阶梯相位（模拟离散纳米柱）
    phi_disc = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            phi_disc[i, j] = np.floor((i + j) / 8) * (np.pi / 4)
    phi_smooth = ps.minimal_surface_smooth(phi_disc, lambda_fidelity=0.05,
                                            max_iter=80, dt=0.05)
    W_disc = ps.surface_energy(phi_disc)
    W_smooth = ps.surface_energy(phi_smooth)
    print(f"[phase_surface] 离散相位 Willmore 能量: {W_disc:.4e}")
    print(f"[phase_surface] 平滑后 Willmore 能量: {W_smooth:.4e}")
    print(f"[phase_surface] 能量降低比: {W_disc / W_smooth:.2f}x")

    return phi_cat, phi_hel, phi_smooth


if __name__ == "__main__":
    demo()
