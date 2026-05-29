"""
density_field.py
================
密度场数值输运与演化模块

基于一维对流方程的 Lax-Wendroff 有限差分格式（融入 fd1d_advection_lax_wendroff 核心算法），
为宇宙学密度场提供数值演化与守恒性检验工具。

在宇宙大尺度结构模拟中，密度场 ρ(x,t) 的演化满足连续性方程:
    ∂ρ/∂t + ∇·(ρ v) = 0

对于一维情况，若速度场 u 为常数:
    ∂ρ/∂t + u ∂ρ/∂x = 0

核心公式
--------
Lax-Wendroff 格式（二阶精度、显式、条件稳定）:

    对 ∂u/∂t = -c ∂u/∂x，离散格式为:
        u_j^{n+1} = u_j^n
                    - (c Δt / 2Δx) (u_{j+1}^n - u_{j-1}^n)
                    + (c² Δt² / 2Δx²) (u_{j+1}^n - 2u_j^n + u_{j-1}^n)

    稳定性条件（CFL）:
        |c| Δt / Δx ≤ 1

    截断误差:
        O(Δt², Δx²)

三维推广（分量分裂）:
    对多维问题采用 Strang 分裂，交替沿 x, y, z 方向应用一维 Lax-Wendroff:
        ρ^{n+1} = L_x(Δt/2) L_y(Δt/2) L_z(Δt) L_y(Δt/2) L_x(Δt/2) ρ^n

质量守恒检验:
    M^n = Σ_j ρ_j^n Δx
    理想情况下 M^{n+1} = M^n
"""

import numpy as np
from typing import Tuple


class AdvectionSolver:
    """
    一维/三维对流方程的 Lax-Wendroff 求解器。
    """

    def __init__(self, nx: int, dx: float, c: float = 1.0):
        """
        Parameters
        ----------
        nx : int
            网格数
        dx : float
            网格间距
        c : float
            对流速度
        """
        self.nx = nx
        self.dx = dx
        self.c = c

    def lax_wendroff_step_1d(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        单步 Lax-Wendroff 更新（融入 fd1d_advection_lax_wendroff 核心算法）。

        Parameters
        ----------
        u : np.ndarray, shape (nx,)
            当前场值
        dt : float
            时间步长

        Returns
        -------
        u_new : np.ndarray
            更新后的场值
        """
        nx = self.nx
        if len(u) != nx:
            raise ValueError("u 长度与 nx 不符")
        # CFL 条件检查
        courant = abs(self.c) * dt / self.dx
        if courant > 1.0:
            raise ValueError(f"CFL 条件破坏: {courant:.3f} > 1")

        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2

        # 周期性边界索引
        im1 = np.roll(np.arange(nx), 1)
        ip1 = np.roll(np.arange(nx), -1)

        u_new = (
            u
            - c1 * (u[ip1] - u[im1])
            + c2 * (u[ip1] - 2.0 * u + u[im1])
        )
        return u_new

    def evolve_1d(
        self, u0: np.ndarray, t_final: float, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一维对流方程演化。

        Returns
        -------
        t_arr : np.ndarray
            时间序列
        u_arr : np.ndarray
            场值历史
        """
        dt = t_final / n_steps
        t_arr = np.linspace(0.0, t_final, n_steps + 1)
        u_arr = np.zeros((n_steps + 1, self.nx))
        u_arr[0] = u0
        u = u0.copy()
        for i in range(n_steps):
            u = self.lax_wendroff_step_1d(u, dt)
            u_arr[i + 1] = u
        return t_arr, u_arr

    def lax_wendroff_step_3d_x(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        沿 x 方向的 Lax-Wendroff 步。
        """
        nx = u.shape[0]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for i in range(nx):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            u_new[i, :, :] = (
                u[i, :, :]
                - c1 * (u[ip1, :, :] - u[im1, :, :])
                + c2 * (u[ip1, :, :] - 2.0 * u[i, :, :] + u[im1, :, :])
            )
        return u_new

    def lax_wendroff_step_3d_y(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """沿 y 方向。"""
        ny = u.shape[1]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for j in range(ny):
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            u_new[:, j, :] = (
                u[:, j, :]
                - c1 * (u[:, jp1, :] - u[:, jm1, :])
                + c2 * (u[:, jp1, :] - 2.0 * u[:, j, :] + u[:, jm1, :])
            )
        return u_new

    def lax_wendroff_step_3d_z(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """沿 z 方向。"""
        nz = u.shape[2]
        c1 = 0.5 * self.c * dt / self.dx
        c2 = 0.5 * (self.c * dt / self.dx) ** 2
        u_new = u.copy()
        for k in range(nz):
            km1 = (k - 1) % nz
            kp1 = (k + 1) % nz
            u_new[:, :, k] = (
                u[:, :, k]
                - c1 * (u[:, :, kp1] - u[:, :, km1])
                + c2 * (u[:, :, kp1] - 2.0 * u[:, :, k] + u[:, :, km1])
            )
        return u_new

    def strang_split_3d(
        self, u: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Strang 分裂三维 Lax-Wendroff 步:
            u^{n+1} = L_x(dt/2) L_y(dt/2) L_z(dt) L_y(dt/2) L_x(dt/2) u^n
        """
        dt2 = dt * 0.5
        u = self.lax_wendroff_step_3d_x(u, dt2)
        u = self.lax_wendroff_step_3d_y(u, dt2)
        u = self.lax_wendroff_step_3d_z(u, dt)
        u = self.lax_wendroff_step_3d_y(u, dt2)
        u = self.lax_wendroff_step_3d_x(u, dt2)
        return u

    def evolve_3d_density_field(
        self,
        rho0: np.ndarray,
        t_final: float,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        三维密度场对流演化（用于测试密度场的平流守恒性）。

        Returns
        -------
        t_arr : np.ndarray
            时间序列
        rho_arr : np.ndarray
            密度场历史
        """
        dt = t_final / n_steps
        t_arr = np.linspace(0.0, t_final, n_steps + 1)
        rho_arr = np.zeros((n_steps + 1,) + rho0.shape)
        rho_arr[0] = rho0
        rho = rho0.copy()
        for i in range(n_steps):
            rho = self.strang_split_3d(rho, dt)
            rho_arr[i + 1] = rho
        return t_arr, rho_arr


def test_mass_conservation() -> float:
    """
    质量守恒性测试。

    初始高斯包在一维周期域中传播，检验总质量是否守恒。
    """
    nx = 101
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    u0 = np.exp(-((x - 0.5) ** 2) / (2 * 0.02 ** 2))
    solver = AdvectionSolver(nx, dx, c=1.0)
    t_arr, u_arr = solver.evolve_1d(u0, t_final=1.0, n_steps=1000)
    mass0 = np.sum(u0) * dx
    mass_final = np.sum(u_arr[-1]) * dx
    return abs(mass_final - mass0) / mass0


if __name__ == "__main__":
    err = test_mass_conservation()
    print(f"一维质量守恒相对误差: {err:.4e}")

    # 三维测试
    nx = 16
    dx = 1.0
    rho0 = np.random.rand(nx, nx, nx)
    solver3d = AdvectionSolver(nx, dx, c=0.5)
    t_arr, rho_arr = solver3d.evolve_3d_density_field(rho0, t_final=0.1, n_steps=50)
    mass0 = rho0.sum()
    massf = rho_arr[-1].sum()
    print(f"三维质量守恒相对误差: {abs(massf - mass0) / mass0:.4e}")
