"""
navier_stokes_solver.py
=======================
Navier-Stokes 方程求解器

求解不可压缩流体在相场界面存在时的运动方程：

连续性方程（不可压条件）：
    ∇·v = 0

动量方程：
    ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + F_σ

界面张力源项（基于相场模型）：
    F_σ = σ κ ∇φ δ(φ)

在相场模型中，界面张力可以表示为：
    F_σ = (3σ / (2√2 ε)) (∇·(|∇φ|² I - ∇φ⊗∇φ)) ∇φ

或更简洁地：
    F_σ = -σ (∂W/∂φ) ∇φ

其中 σ 为表面张力系数，ε 为界面宽度。
"""

import numpy as np


class NavierStokesSolver:
    """
    基于投影法（Projection Method）的二维不可压 Navier-Stokes 求解器。

    投影法将每时间步分为两步：
    1. 预测步：求解中间速度 v*（不考虑压力梯度）
    2. 投影步：求解压力泊松方程并修正速度使其满足不可压条件
    """

    def __init__(self, nx, ny, dx, dy, dt, rho=1.0, mu=0.01,
                 surface_tension=0.1, epsilon=0.01):
        """
        初始化 NS 求解器。

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        dt : float
            时间步长。
        rho : float
            流体密度。
        mu : float
            动力粘度。
        surface_tension : float
            表面张力系数 σ。
        epsilon : float
            相场界面宽度参数。
        """
        if nx < 3 or ny < 3:
            raise ValueError("网格维度必须至少为 3")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("步长参数必须为正")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.rho = rho
        self.mu = mu
        self.surface_tension = surface_tension
        self.epsilon = epsilon

        # 预计算系数
        self.nu = mu / rho  # 运动粘度
        self.inv_rho = 1.0 / rho

    def compute_surface_tension_force(self, phi):
        """
        基于相场模型计算界面张力体积力。

        采用连续表面力（CSF）模型：
            F_σ = σ κ ∇φ / |∇φ| * |∇φ|

        在相场模型中可等价表示为：
            F_σ = (3σ / (2√2 ε)) W'(φ) ∇φ

        其中 W'(φ) = φ³ - φ。

        Parameters
        ----------
        phi : ndarray, shape (nx, ny)
            序参量场。

        Returns
        -------
        tuple of ndarray
            (Fx, Fy) 界面张力在 x 和 y 方向的分量。
        """
        phi_clipped = np.clip(phi, -1.2, 1.2)

        # 计算 ∇φ
        grad_phi_x = np.zeros_like(phi)
        grad_phi_y = np.zeros_like(phi)

        grad_phi_x[1:-1, :] = (phi_clipped[2:, :] - phi_clipped[:-2, :]) / (2.0 * self.dx)
        grad_phi_y[:, 1:-1] = (phi_clipped[:, 2:] - phi_clipped[:, :-2]) / (2.0 * self.dy)

        # 双阱势导数 W'(φ) = φ³ - φ
        dwdphi = phi_clipped ** 3 - phi_clipped

        # 界面张力力系数
        coeff = (3.0 * self.surface_tension) / (2.0 * np.sqrt(2.0) * self.epsilon)

        Fx = coeff * dwdphi * grad_phi_x
        Fy = coeff * dwdphi * grad_phi_y

        return Fx, Fy

    def advection_diffusion_velocity(self, vx, vy):
        """
        计算速度的对流-扩散项右端项：
            -v·∇v + ν∇²v

        采用中心差分计算扩散项，迎风格式计算对流项。

        Parameters
        ----------
        vx, vy : ndarray
            当前速度分量。

        Returns
        -------
        tuple of ndarray
            (rhs_x, rhs_y) 速度方程右端项。
        """
        rhs_x = np.zeros_like(vx)
        rhs_y = np.zeros_like(vy)

        # 扩散项：ν∇²v
        lap_vx = np.zeros_like(vx)
        lap_vy = np.zeros_like(vy)

        lap_vx[1:-1, 1:-1] = (
            (vx[2:, 1:-1] - 2.0 * vx[1:-1, 1:-1] + vx[:-2, 1:-1]) / (self.dx ** 2) +
            (vx[1:-1, 2:] - 2.0 * vx[1:-1, 1:-1] + vx[1:-1, :-2]) / (self.dy ** 2)
        )
        lap_vy[1:-1, 1:-1] = (
            (vy[2:, 1:-1] - 2.0 * vy[1:-1, 1:-1] + vy[:-2, 1:-1]) / (self.dx ** 2) +
            (vy[1:-1, 2:] - 2.0 * vy[1:-1, 1:-1] + vy[1:-1, :-2]) / (self.dy ** 2)
        )

        rhs_x += self.nu * lap_vx
        rhs_y += self.nu * lap_vy

        # 对流项：-v·∇v（迎风格式）
        # x 方向速度的对流
        dvx_dx = np.zeros_like(vx)
        dvx_dy = np.zeros_like(vx)

        mask_pos = vx >= 0
        dvx_dx[1:-1, :][mask_pos[1:-1, :]] = (
            vx[1:-1, :][mask_pos[1:-1, :]] - vx[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        dvx_dx[1:-1, :][mask_neg[1:-1, :]] = (
            vx[2:, :][mask_neg[1:-1, :]] - vx[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx

        mask_pos = vy >= 0
        dvx_dy[:, 1:-1][mask_pos[:, 1:-1]] = (
            vx[:, 1:-1][mask_pos[:, 1:-1]] - vx[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        dvx_dy[:, 1:-1][mask_neg[:, 1:-1]] = (
            vx[:, 2:][mask_neg[:, 1:-1]] - vx[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        rhs_x -= vx * dvx_dx + vy * dvx_dy

        # y 方向速度的对流
        dvy_dx = np.zeros_like(vy)
        dvy_dy = np.zeros_like(vy)

        mask_pos = vx >= 0
        dvy_dx[1:-1, :][mask_pos[1:-1, :]] = (
            vy[1:-1, :][mask_pos[1:-1, :]] - vy[:-2, :][mask_pos[1:-1, :]]
        ) / self.dx
        mask_neg = vx < 0
        dvy_dx[1:-1, :][mask_neg[1:-1, :]] = (
            vy[2:, :][mask_neg[1:-1, :]] - vy[1:-1, :][mask_neg[1:-1, :]]
        ) / self.dx

        mask_pos = vy >= 0
        dvy_dy[:, 1:-1][mask_pos[:, 1:-1]] = (
            vy[:, 1:-1][mask_pos[:, 1:-1]] - vy[:, :-2][mask_pos[:, 1:-1]]
        ) / self.dy
        mask_neg = vy < 0
        dvy_dy[:, 1:-1][mask_neg[:, 1:-1]] = (
            vy[:, 2:][mask_neg[:, 1:-1]] - vy[:, 1:-1][mask_neg[:, 1:-1]]
        ) / self.dy

        rhs_y -= vx * dvy_dx + vy * dvy_dy

        return rhs_x, rhs_y

    def solve_pressure_poisson_gs(self, div_vstar, max_iter=1000, tol=1e-6):
        """
        使用 Gauss-Seidel 迭代求解压力泊松方程：
            ∇²p = (ρ/Δt) ∇·v*

        采用齐次 Neumann 边界条件（∂p/∂n = 0）。

        离散形式（五点差分）：
            (p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4p_{i,j}) / h² = rhs

        Parameters
        ----------
        div_vstar : ndarray
            中间速度的散度场。
        max_iter : int
            最大迭代次数。
        tol : float
            收敛容差。

        Returns
        -------
        ndarray
            压力场 p。
        """
        p = np.zeros_like(div_vstar)
        rhs = (self.rho / self.dt) * div_vstar

        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for it in range(max_iter):
            p_old = p.copy()

            # 内部点 Gauss-Seidel 更新
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    p[i, j] = (
                        (p[i + 1, j] + p[i - 1, j]) / dx2 +
                        (p[i, j + 1] + p[i, j - 1]) / dy2 -
                        rhs[i, j]
                    ) / denom

            # Neumann 边界条件（镜像值）
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

            # 固定参考压力（消除零模）
            p -= p.mean()

            # 检查收敛
            diff = np.max(np.abs(p - p_old))
            if diff < tol:
                break

        return p

    def projection_step(self, vx_star, vy_star):
        """
        投影法第二步：修正速度使其满足不可压条件。

        计算压力梯度并修正速度：
            v^{n+1} = v* - (Δt/ρ) ∇p

        Parameters
        ----------
        vx_star, vy_star : ndarray
            预测中间速度。

        Returns
        -------
        tuple of ndarray
            (vx_new, vy_new, p) 修正后的速度和压力。
        """
        # 计算中间速度的散度
        div_vstar = np.zeros_like(vx_star)
        div_vstar[1:-1, 1:-1] = (
            (vx_star[2:, 1:-1] - vx_star[:-2, 1:-1]) / (2.0 * self.dx) +
            (vy_star[1:-1, 2:] - vy_star[1:-1, :-2]) / (2.0 * self.dy)
        )

        # 求解压力泊松方程
        p = self.solve_pressure_poisson_gs(div_vstar)

        # 计算压力梯度
        grad_p_x = np.zeros_like(p)
        grad_p_y = np.zeros_like(p)

        grad_p_x[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2.0 * self.dx)
        grad_p_y[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * self.dy)

        # 修正速度
        vx_new = vx_star - (self.dt / self.rho) * grad_p_x
        vy_new = vy_star - (self.dt / self.rho) * grad_p_y

        return vx_new, vy_new, p

    def time_step(self, vx, vy, phi):
        """
        执行一个完整的 NS 时间步（投影法）。

        1. 计算预测速度 v* = v^n + Δt * (-v·∇v + ν∇²v + F_σ/ρ)
        2. 投影修正满足 ∇·v^{n+1} = 0

        Parameters
        ----------
        vx, vy : ndarray
            当前速度场。
        phi : ndarray
            当前相场（用于计算界面张力）。

        Returns
        -------
        tuple of ndarray
            (vx_new, vy_new, p) 新时刻的速度和压力。
        """
        # 计算对流通量 + 扩散 + 界面张力
        rhs_x, rhs_y = self.advection_diffusion_velocity(vx, vy)

        Fx, Fy = self.compute_surface_tension_force(phi)
        rhs_x += self.inv_rho * Fx
        rhs_y += self.inv_rho * Fy

        # 预测步
        vx_star = vx + self.dt * rhs_x
        vy_star = vy + self.dt * rhs_y

        # 应用无滑移边界条件
        vx_star[0, :] = 0.0
        vx_star[-1, :] = 0.0
        vx_star[:, 0] = 0.0
        vx_star[:, -1] = 0.0
        vy_star[0, :] = 0.0
        vy_star[-1, :] = 0.0
        vy_star[:, 0] = 0.0
        vy_star[:, -1] = 0.0

        # 投影步
        vx_new, vy_new, p = self.projection_step(vx_star, vy_star)

        return vx_new, vy_new, p
