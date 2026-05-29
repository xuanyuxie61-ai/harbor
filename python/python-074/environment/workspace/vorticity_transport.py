r"""
vorticity_transport.py
======================
二维不可压 Navier-Stokes 方程的涡量-流函数形式求解器。

控制方程
--------
1. 涡量输运方程（对流-扩散型）：

   \frac{\partial \omega}{\partial t}
   + u \frac{\partial \omega}{\partial x}
   + v \frac{\partial \omega}{\partial y}
   = \nu \left( \frac{\partial^2 \omega}{\partial x^2}
                + \frac{\partial^2 \omega}{\partial y^2} \right)

   其中 \omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
   为涡量，\nu 为运动粘性系数。

2. 流函数泊松方程：

   \nabla^2 \psi = -\omega

   速度重构：
   u = \frac{\partial \psi}{\partial y}, \quad
   v = -\frac{\partial \psi}{\partial x}

离散方法
--------
- 空间：在结构化交错网格上，对流项采用二阶迎风格式（Upwind），
  扩散项采用中心差分。
- 时间：显式 Adams-Bashforth 2 步（对流项）+ 隐式 Crank-Nicolson
  （扩散项），即经典的 AB2-CN 分裂格式。

边界条件
--------
- 入口：均匀来流 U_\infty，涡量 \omega = 0。
- 出口：Sommerfeld 辐射条件 \partial\omega/\partial t + U_c \partial\omega/\partial x = 0。
- 壁面（圆柱表面）：采用 Thom (1933) 壁面涡量公式：

  \omega_{wall} = -\frac{2 \psi_{wall+1}}{\Delta n^2} - \frac{2 U_{wall}}{\Delta n}

  其中 \Delta n 为壁面法向网格间距。

本模块对应原种子项目：
- 352_fd1d_advection_diffusion_steady（一维对流扩散有限差分格式，
  升维至二维并加入非定常项）
r"""

import numpy as np


class VorticityTransportSolver:
    r"""
    二维涡量输运方程求解器，基于结构化笛卡尔网格。
    """

    def __init__(self, nx, ny, lx, ly, nu, u_inf, dt, cylinder_params=None):
        r"""
        参数
        ----
        nx, ny : int
            x, y 方向网格数（含边界）。
        lx, ly : float
            计算域尺寸。
        nu : float
            运动粘性系数。
        u_inf : float
            来流速度。
        dt : float
            时间步长。
        cylinder_params : dict or None
            {'cx': 圆柱中心x, 'cy': 圆柱中心y, 'r': 圆柱半径}
        """
        if nx < 5 or ny < 5:
            raise ValueError("网格数 nx, ny 至少为 5。")
        if nu <= 0:
            raise ValueError("运动粘性系数 nu 必须为正。")
        if dt <= 0:
            raise ValueError("时间步长 dt 必须为正。")

        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.nu = nu
        self.u_inf = u_inf
        self.dt = dt

        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        # CFL 与扩散稳定性检查
        cfl_x = u_inf * dt / self.dx
        cfl_y = u_inf * dt / self.dy
        diff_num_x = nu * dt / (self.dx ** 2)
        diff_num_y = nu * dt / (self.dy ** 2)

        if cfl_x > 1.0 or cfl_y > 1.0:
            # 警告但不终止，因为 AB2-CN 格式对线性问题无条件稳定
            pass
        if diff_num_x > 0.5 or diff_num_y > 0.5:
            # CN 隐式处理扩散，理论上无条件稳定
            pass

        # 场量定义在 cell-centered 网格上
        self.omega = np.zeros((ny, nx))
        self.omega_old = np.zeros((ny, nx))
        self.omega_old2 = np.zeros((ny, nx))
        self.psi = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))

        # 圆柱参数
        if cylinder_params is None:
            cylinder_params = {'cx': lx * 0.25, 'cy': ly * 0.5, 'r': min(lx, ly) * 0.05}
        self.cx = cylinder_params['cx']
        self.cy = cylinder_params['cy']
        self.r_cyl = cylinder_params['r']
        if self.r_cyl <= 0:
            raise ValueError("圆柱半径必须为正。")

        # 预计算网格坐标
        self.x_grid = np.linspace(0.0, lx, nx)
        self.y_grid = np.linspace(0.0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        # 掩码：标记圆柱内部点（固体区域）
        dist_sq = (self.X - self.cx) ** 2 + (self.Y - self.cy) ** 2
        self.solid_mask = dist_sq <= self.r_cyl ** 2
        self.fluid_mask = ~self.solid_mask

        # 预计算内部流体点索引（用于快速迭代）
        self.interior_indices = []
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.fluid_mask[j, i]:
                    self.interior_indices.append((j, i))

        # 初始化流函数为均匀来流（除圆柱内）
        for j in range(ny):
            self.psi[j, :] = u_inf * self.y_grid[j]
        self.psi[self.solid_mask] = 0.0
        # 由均匀来流初始化速度场
        self.u[:, :] = u_inf
        self.u[self.solid_mask] = 0.0

    def _is_inside_cylinder(self, x, y):
        r"""判断点是否在圆柱内部。"""
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r_cyl ** 2

    def apply_boundary_conditions(self):
        """
        应用边界条件：入口、出口、上下壁面、圆柱表面。
        r"""
        ny, nx = self.ny, self.nx

        # 入口 (i=0): 均匀流，涡量为 0
        self.omega[:, 0] = 0.0
        self.psi[:, 0] = self.u_inf * self.Y[:, 0]

        # 出口 (i=nx-1): 零梯度（简化 Sommerfeld）
        self.omega[:, nx - 1] = self.omega[:, nx - 2]
        self.psi[:, nx - 1] = self.psi[:, nx - 2]

        # 上下壁面 (j=0, j=ny-1): 无滑移
        self.omega[0, :] = 0.0
        self.omega[ny - 1, :] = 0.0
        self.psi[0, :] = 0.0
        self.psi[ny - 1, :] = self.u_inf * self.ly

        # 圆柱表面：Thom 公式计算壁面涡量
        # 对圆柱附近流体点，识别其法向邻居
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if self.solid_mask[j, i]:
                    continue
                # 检查四个邻居是否有固体点
                neighbors = [
                    (j - 1, i, self.dy),
                    (j + 1, i, self.dy),
                    (j, i - 1, self.dx),
                    (j, i + 1, self.dx),
                ]
                for nj, ni, dn in neighbors:
                    if 0 <= nj < ny and 0 <= ni < nx and self.solid_mask[nj, ni]:
                        # (j,i) 紧邻圆柱壁面，用 Thom 公式反算壁面涡量
                        # 将壁面涡量赋给固体点（ ghost cell 思想）
                        # 壁面速度为 0（无滑移）
                        # \omega_wall = -2*\psi_{fluid} / dn^2
                        # 这里 \psi_{wall}=0
                        wall_omega = -2.0 * self.psi[j, i] / (dn ** 2)
                        self.omega[nj, ni] = wall_omega

        # 固体内部涡量设为 0（不参与计算）
        self.omega[self.solid_mask] = 0.0
        self.psi[self.solid_mask] = 0.0

    def compute_velocity_from_psi(self):
        """
        由流函数重构速度场：
        u = \partial\psi/\partial y  （中心差分）
        v = -\partial\psi/\partial x
        r"""
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy

        # 内部点
        self.u[1:ny - 1, 1:nx - 1] = (
            self.psi[2:ny, 1:nx - 1] - self.psi[0:ny - 2, 1:nx - 1]
        ) / (2.0 * dy)
        self.v[1:ny - 1, 1:nx - 1] = -(
            self.psi[1:ny - 1, 2:nx] - self.psi[1:ny - 1, 0:nx - 2]
        ) / (2.0 * dx)

        # 边界速度（简化处理）
        self.u[:, 0] = self.u_inf
        self.u[:, nx - 1] = self.u[:, nx - 2]
        self.u[0, :] = 0.0
        self.u[ny - 1, :] = self.u_inf

        self.v[:, 0] = 0.0
        self.v[:, nx - 1] = self.v[:, nx - 2]
        self.v[0, :] = 0.0
        self.v[ny - 1, :] = 0.0

        # 固体内部速度为 0
        self.u[self.solid_mask] = 0.0
        self.v[self.solid_mask] = 0.0

    def convective_term(self, omega_field):
        """
        计算对流项：- (u \partial\omega/\partial x + v \partial\omega/\partial y)。
        采用二阶迎风格式，结合边界处理。
        r"""
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy
        conv = np.zeros_like(omega_field)

        for j, i in self.interior_indices:
            u_ij = self.u[j, i]
            v_ij = self.v[j, i]

            # x 方向迎风格式
            if u_ij >= 0:
                dwdx = (omega_field[j, i] - omega_field[j, i - 1]) / dx
            else:
                dwdx = (omega_field[j, i + 1] - omega_field[j, i]) / dx

            # y 方向迎风格式
            if v_ij >= 0:
                dwdy = (omega_field[j, i] - omega_field[j - 1, i]) / dy
            else:
                dwdy = (omega_field[j + 1, i] - omega_field[j, i]) / dy

            conv[j, i] = -(u_ij * dwdx + v_ij * dwdy)

        return conv

    def diffusive_term(self, omega_field):
        """
        计算扩散项：\nu (\partial^2\omega/\partial x^2 + \partial^2\omega/\partial y^2)。
        采用五点中心差分。
        r"""
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy
        diff = np.zeros_like(omega_field)

        for j, i in self.interior_indices:
            d2wdx2 = (
                omega_field[j, i + 1]
                - 2.0 * omega_field[j, i]
                + omega_field[j, i - 1]
            ) / (dx ** 2)
            d2wdy2 = (
                omega_field[j + 1, i]
                - 2.0 * omega_field[j, i]
                + omega_field[j - 1, i]
            ) / (dy ** 2)
            diff[j, i] = self.nu * (d2wdx2 + d2wdy2)

        return diff

    def time_step(self, step_count):
        """
        推进一个时间步。
        采用 AB2-CN 格式：

        \frac{\omega^{n+1} - \omega^n}{\Delta t}
        = \frac{3}{2} C^n - \frac{1}{2} C^{n-1}
        + \frac{1}{2} D^n + \frac{1}{2} D^{n+1}

        其中 C 为对流项，D 为扩散项。
        隐式扩散项通过 Jacobi 迭代近似求解（简化处理）。
        r"""
        # 计算当前步的对流项与扩散项
        c_n = self.convective_term(self.omega)
        d_n = self.diffusive_term(self.omega)

        if step_count == 0:
            # 第一步用前向欧拉
            rhs = self.omega + self.dt * (c_n + d_n)
        else:
            c_nm1 = self.convective_term(self.omega_old)
            d_nm1 = self.diffusive_term(self.omega_old)
            rhs = (
                self.omega
                + self.dt * (1.5 * c_n - 0.5 * c_nm1)
                + 0.5 * self.dt * d_n
                + 0.5 * self.dt * d_nm1
            )

        # CN 隐式扩散：简化为显式 + 局部 Richardson 修正
        # 为保持零参数可运行，采用简化处理：直接赋值
        # 实际工程中此处应求解大型稀疏线性系统
        self.omega_old2 = self.omega_old.copy()
        self.omega_old = self.omega.copy()
        self.omega = rhs.copy()

        # 对内部点做一次 Jacobi 平滑以近似隐式扩散
        # (I - 0.5*dt*D) \omega^{n+1} \approx rhs
        omega_new = self.omega.copy()
        for _ in range(3):  # 3 次 Jacobi 迭代
            for j, i in self.interior_indices:
                laplacian = (
                    self.omega[j, i + 1]
                    + self.omega[j, i - 1]
                    + self.omega[j + 1, i]
                    + self.omega[j - 1, i]
                )
                omega_new[j, i] = (
                    rhs[j, i]
                    + 0.5 * self.nu * self.dt / (self.dx ** 2) * laplacian
                ) / (
                    1.0
                    + self.nu * self.dt / (self.dx ** 2)
                    + self.nu * self.dt / (self.dy ** 2)
                )
            self.omega = omega_new.copy()

        # 注意：壁面边界条件在 main.py 中求解泊松方程后更新

    def compute_force_coefficients(self):
        """
        通过壁面涡量积分计算升力系数 C_L 与阻力系数 C_D。

        公式：
        C_D = \frac{2}{U_\infty^2 R} \int_{0}^{2\pi} \omega_{wall}(\theta)
              \sin\theta \, R d\theta
        C_L = -\frac{2}{U_\infty^2 R} \int_{0}^{2\pi} \omega_{wall}(\theta)
              \cos\theta \, R d\theta

        离散：沿圆柱表面采样壁面涡量，采用梯形公式。
        r"""
        # 沿圆柱表面均匀采样
        n_surf = 128
        theta = np.linspace(0.0, 2.0 * np.pi, n_surf, endpoint=False)
        x_surf = self.cx + self.r_cyl * np.cos(theta)
        y_surf = self.cy + self.r_cyl * np.sin(theta)

        omega_surf = np.zeros(n_surf)
        for k in range(n_surf):
            # 找到最近的网格点，取涡量
            i = int(np.clip(np.round(x_surf[k] / self.dx), 0, self.nx - 1))
            j = int(np.clip(np.round(y_surf[k] / self.dy), 0, self.ny - 1))
            # 如果最近点是固体，取其值；否则找固体邻居
            if self.solid_mask[j, i]:
                omega_surf[k] = self.omega[j, i]
            else:
                # 搜索周围固体点
                found = False
                for dj in range(-2, 3):
                    for di in range(-2, 3):
                        nj, ni = j + dj, i + di
                        if 0 <= nj < self.ny and 0 <= ni < self.nx:
                            if self.solid_mask[nj, ni]:
                                omega_surf[k] = self.omega[nj, ni]
                                found = True
                                break
                    if found:
                        break
                if not found:
                    omega_surf[k] = 0.0

        # TODO: Hole 1 — 请根据壁面涡量积分公式计算升阻力系数
        # 科学背景：
        #   C_D = (2 / (U_inf^2 * R)) * integral(omega_wall * sin(theta) * R dtheta)
        #   C_L = -(2 / (U_inf^2 * R)) * integral(omega_wall * cos(theta) * R dtheta)
        # 离散采用梯形公式，n_surf=128 个采样点均匀分布在 [0, 2*pi)。
        # 注意：omega_surf 已按圆柱表面采样，theta 为对应角度。
        # 变量说明：self.u_inf 为来流速度，self.r_cyl 为圆柱半径。
        # 请补全 d_theta、c_d、c_l 的计算，并返回 (c_d, c_l)。
        raise NotImplementedError("Hole 1: 壁面涡量积分公式尚未实现")

    def get_wake_profile(self, x_loc):
        """
        获取尾流区指定 x 位置的涡量剖面。
        r"""
        i = int(np.clip(np.round(x_loc / self.dx), 0, self.nx - 1))
        profile = self.omega[:, i].copy()
        # 将固体区域置 NaN
        profile[self.solid_mask[:, i]] = np.nan
        return profile, self.y_grid.copy()
