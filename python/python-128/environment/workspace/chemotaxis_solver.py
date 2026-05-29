"""
chemotaxis_solver.py
====================
三维趋化因子反应-扩散-对流方程求解器

融合原始项目：
  - 357_fd1d_burgers_leap： leapfrog 格式求解无粘性 Burgers 方程（非线性对流）
  - 058_atkinson/heat2：向后差分格式求解热方程（扩散项）

数学物理模型：
  趋化因子浓度 c(x,t) 满足反应-扩散-对流 (RDA) 方程：

      ∂c/∂t = D ∇²c - v · ∇c + R(c) - λ c

  其中：
    - D：扩散系数（单位：μm²/s）
    - v：对流速度场（由组织液流动或细胞运动引起）
    - R(c)：非线性产生项（Michaelis-Menten 饱和动力学）
        R(c) = V_max · c / (K_m + c)
    - λ：一级降解速率常数

  在离散层面，对每个空间维度分别采用隐式-显式 (IMEX) 分裂：
    - 扩散项：向后 Euler（三对角系统，无条件稳定）
    - 对流项：leapfrog / 中心差分（显式）
    - 反应项：半隐式处理

  稳定性条件（CFL 与扩散）：
      Δt ≤ min( Δx² / (6D),  Δx / |v|_max )
"""

import numpy as np
from special_math import tridiag_solve


class ChemotaxisSolver:
    """
    三维矩形区域上的趋化因子浓度求解器。
    采用方向分裂 (Directional Splitting) 策略，
    将三维问题分解为三个一维问题的序列求解。
    """

    def __init__(self,
                 nx: int = 32, ny: int = 32, nz: int = 16,
                 xlim: tuple = (-1.0, 1.0),
                 ylim: tuple = (-1.0, 1.0),
                 zlim: tuple = (-0.5, 0.5),
                 D: float = 0.01,
                 lambda_deg: float = 0.05,
                 Vmax: float = 1.0,
                 Km: float = 0.5):
        self.nx = max(3, int(nx))
        self.ny = max(3, int(ny))
        self.nz = max(3, int(nz))
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.D = float(D)
        self.lambda_deg = float(lambda_deg)
        self.Vmax = float(Vmax)
        self.Km = float(Km)

        self.x = np.linspace(xlim[0], xlim[1], self.nx)
        self.y = np.linspace(ylim[0], ylim[1], self.ny)
        self.z = np.linspace(zlim[0], zlim[1], self.nz)
        self.dx = (xlim[1] - xlim[0]) / (self.nx - 1)
        self.dy = (ylim[1] - ylim[0]) / (self.ny - 1)
        self.dz = (zlim[1] - zlim[0]) / (self.nz - 1)

        # 浓度场
        self.c = np.zeros((self.nx, self.ny, self.nz), dtype=float)

    def set_initial_condition(self, c0_func):
        """
        设置初始浓度分布。

        参数
        ----
        c0_func : callable
            c0_func(x, y, z) -> float
        """
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    self.c[i, j, k] = c0_func(self.x[i], self.y[j], self.z[k])

    def _reaction_source(self, c):
        """
        非线性产生项 R(c) = V_max * c / (K_m + c)。
        """
        return self.Vmax * c / (self.Km + c + 1e-12)

    def _safe_dt(self, vx, vy, vz):
        """
        根据 CFL 和扩散稳定性自动选取安全时间步长。
        """
        dt_diff = min(self.dx ** 2, self.dy ** 2, self.dz ** 2) / (6.0 * self.D + 1e-15)
        vmax = max(np.max(np.abs(vx)), np.max(np.abs(vy)), np.max(np.abs(vz)))
        dt_adv = min(self.dx, self.dy, self.dz) / (vmax + 1e-15)
        return 0.3 * min(dt_diff, dt_adv)

    def _solve_1d_diffusion_implicit(self, u, dx, dt, dir_axis):
        """
        对指定维度使用隐式向后 Euler 求解一维扩散方程：
            u* - u = r · (u*_{i-1} - 2 u*_i + u*_{i+1})
        其中 r = D Δt / Δx²。
        形成三对角系统并调用 tridiag_solve 求解。

        TODO (Hole 1): 实现隐式向后 Euler 一维扩散求解。
        需要构造三对角矩阵 (a, b, c) 并处理零 Neumann 边界条件，
        然后沿指定维度对每个空间切片调用 tridiag_solve 求解。
        """
        # === HOLE 1 BEGIN ===
        raise NotImplementedError("Hole 1: _solve_1d_diffusion_implicit 尚未实现")
        # === HOLE 1 END ===

    def _advection_step_leapfrog_1d(self, u, v, dx, dt, dir_axis):
        """
        使用中心差分处理一维对流项（Burgers/leapfrog 思想退化到线性对流）：
            u_i^{n+1} = u_i^{n-1} - (v_i Δt / Δx) (u_{i+1}^n - u_{i-1}^n)
        为避免启动问题，首步采用向前 Euler。
        """
        # 为简化，这里使用一阶迎风格式（更稳定，且无需多层存储）
        u_new = np.copy(u)
        if dir_axis == 0:
            for i in range(1, u.shape[0] - 1):
                coeff = v[i] * dt / dx
                if coeff >= 0:
                    u_new[i, :, :] = u[i, :, :] - coeff * (u[i, :, :] - u[i - 1, :, :])
                else:
                    u_new[i, :, :] = u[i, :, :] - coeff * (u[i + 1, :, :] - u[i, :, :])
        elif dir_axis == 1:
            for j in range(1, u.shape[1] - 1):
                coeff = v[j] * dt / dx
                if coeff >= 0:
                    u_new[:, j, :] = u[:, j, :] - coeff * (u[:, j, :] - u[:, j - 1, :])
                else:
                    u_new[:, j, :] = u[:, j, :] - coeff * (u[:, j + 1, :] - u[:, j, :])
        else:
            for k in range(1, u.shape[2] - 1):
                coeff = v[k] * dt / dx
                if coeff >= 0:
                    u_new[:, :, k] = u[:, :, k] - coeff * (u[:, :, k] - u[:, :, k - 1])
                else:
                    u_new[:, :, k] = u[:, :, k] - coeff * (u[:, :, k + 1] - u[:, :, k])
        return u_new

    def step(self, vx, vy, vz, dt: float = None):
        """
        推进一个时间步。

        参数
        ----
        vx, vy, vz : np.ndarray 或 float
            对流速度场（可随空间变化或常数）
        dt : float, optional
            时间步长，None 时自动计算

        返回
        ----
        dt_used : float
            实际使用的时间步长
        """
        if np.isscalar(vx):
            vx_arr = np.full(self.nx, float(vx))
        else:
            vx_arr = np.asarray(vx, dtype=float).reshape(self.nx)
        if np.isscalar(vy):
            vy_arr = np.full(self.ny, float(vy))
        else:
            vy_arr = np.asarray(vy, dtype=float).reshape(self.ny)
        if np.isscalar(vz):
            vz_arr = np.full(self.nz, float(vz))
        else:
            vz_arr = np.asarray(vz, dtype=float).reshape(self.nz)

        if dt is None:
            dt = self._safe_dt(vx_arr, vy_arr, vz_arr)
        dt = float(dt)
        if dt <= 1e-15:
            raise ValueError("chemotaxis_solver.step: dt 过小")

        c = self.c

        # 方向分裂：依次处理 x, y, z 方向的扩散+对流
        # Step 1: x 方向
        c = self._solve_1d_diffusion_implicit(c, self.dx, dt, 0)
        c = self._advection_step_leapfrog_1d(c, vx_arr, self.dx, dt, 0)

        # Step 2: y 方向
        c = self._solve_1d_diffusion_implicit(c, self.dy, dt, 1)
        c = self._advection_step_leapfrog_1d(c, vy_arr, self.dy, dt, 1)

        # Step 3: z 方向
        c = self._solve_1d_diffusion_implicit(c, self.dz, dt, 2)
        c = self._advection_step_leapfrog_1d(c, vz_arr, self.dz, dt, 2)

        # 反应源项与降解（半隐式）
        source = self._reaction_source(c)
        # 对降解项隐式：c_new = (c + dt * source) / (1 + dt * lambda)
        c = (c + dt * source) / (1.0 + dt * self.lambda_deg)

        # 边界条件：Neumann 零通量（已在扩散隐式中体现，这里裁剪负值）
        c = np.maximum(c, 0.0)

        self.c = c
        return dt

    def gradient(self):
        """
        计算浓度场的梯度 ∇c = [∂c/∂x, ∂c/∂y, ∂c/∂z]。

        采用中心差分：
            (∂c/∂x)_i ≈ (c_{i+1} - c_{i-1}) / (2 Δx)
        边界处采用向前/向后差分。
        """
        gx = np.zeros_like(self.c)
        gy = np.zeros_like(self.c)
        gz = np.zeros_like(self.c)

        # x 方向
        gx[1:-1, :, :] = (self.c[2:, :, :] - self.c[:-2, :, :]) / (2.0 * self.dx)
        gx[0, :, :] = (self.c[1, :, :] - self.c[0, :, :]) / self.dx
        gx[-1, :, :] = (self.c[-1, :, :] - self.c[-2, :, :]) / self.dx

        # y 方向
        gy[:, 1:-1, :] = (self.c[:, 2:, :] - self.c[:, :-2, :]) / (2.0 * self.dy)
        gy[:, 0, :] = (self.c[:, 1, :] - self.c[:, 0, :]) / self.dy
        gy[:, -1, :] = (self.c[:, -1, :] - self.c[:, -2, :]) / self.dy

        # z 方向
        gz[:, :, 1:-1] = (self.c[:, :, 2:] - self.c[:, :, :-2]) / (2.0 * self.dz)
        gz[:, :, 0] = (self.c[:, :, 1] - self.c[:, :, 0]) / self.dz
        gz[:, :, -1] = (self.c[:, :, -1] - self.c[:, :, -2]) / self.dz

        return gx, gy, gz

    def total_mass(self):
        """计算总质量（积分近似）。"""
        return np.sum(self.c) * self.dx * self.dy * self.dz
