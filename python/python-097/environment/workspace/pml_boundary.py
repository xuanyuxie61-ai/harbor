"""
pml_boundary.py

完美匹配层(Perfectly Matched Layer, PML)吸收边界条件模块。

PML是FDTD仿真中处理开放边界问题的核心方法。
通过在计算域边界引入人工各向异性吸收层，
使向外传播的电磁波在层内无反射地衰减。

核心物理模型:
--------------
1. PML中的麦克斯韦方程（拉伸坐标形式）:
   ∂E_x/∂t + σ_y E_x = (1/ε) · ∂H_z/∂y
   其中σ为电导率型吸收参数。

2. 复坐标拉伸:
   x̃ = x + (i/ω) ∫₀ˣ σ_x(s) ds
   这等效于在PML中引入复介电常数和复磁导率。

3. 多项式电导率分布:
   σ(d) = σ_max · (d / d_pml)^m
   其中m通常为2~4，d为到PML内边界的距离。

4. 反射系数估计:
   R(θ) ≈ exp(-2η₀ cosθ ∫₀^{d_pml} σ(s) ds)
"""

import numpy as np
from physics_constants import EPSILON_0, MU_0, ETA_0


class PMLBoundary3D:
    """
    三维PML吸收边界条件。
    """

    def __init__(self, grid, pml_thickness=8, sigma_max=None, order=3, reflection_coeff=1e-6):
        """
        Parameters
        ----------
        grid : YeeGrid3D
            Yee网格
        pml_thickness : int
            PML层厚度（网格数）
        sigma_max : float or None
            最大电导率，None时自动计算
        order : int
            电导率多项式阶数
        reflection_coeff : float
            目标反射系数
        """
        self.grid = grid
        self.pml_thickness = pml_thickness
        self.order = order

        nx, ny, nz = grid.nx, grid.ny, grid.nz

        # 计算sigma_max（基于理论反射系数）
        if sigma_max is None:
            d_pml = pml_thickness * min(grid.dx, grid.dy, grid.dz)
            sigma_max = -(order + 1) * np.log(reflection_coeff) / (2.0 * ETA_0 * d_pml)

        self.sigma_max = sigma_max

        # 生成PML电导率分布
        self.sigma_x = np.zeros((nx, ny, nz))
        self.sigma_y = np.zeros((nx, ny, nz))
        self.sigma_z = np.zeros((nx, ny, nz))

        self._build_pml_profile(nx, ny, nz, grid.dx, grid.dy, grid.dz)

        # 分裂场变量（用于PML更新）
        self.Ex_y = np.zeros((nx, ny, nz))
        self.Ex_z = np.zeros((nx, ny, nz))
        self.Ey_x = np.zeros((nx, ny, nz))
        self.Ey_z = np.zeros((nx, ny, nz))
        self.Ez_x = np.zeros((nx, ny, nz))
        self.Ez_y = np.zeros((nx, ny, nz))

        self.Hx_y = np.zeros((nx, ny, nz))
        self.Hx_z = np.zeros((nx, ny, nz))
        self.Hy_x = np.zeros((nx, ny, nz))
        self.Hy_z = np.zeros((nx, ny, nz))
        self.Hz_x = np.zeros((nx, ny, nz))
        self.Hz_y = np.zeros((nx, ny, nz))

    def _build_pml_profile(self, nx, ny, nz, dx, dy, dz):
        """构建PML电导率分布。"""
        d = self.pml_thickness
        m = self.order
        sigma_m = self.sigma_max

        # x方向PML
        for i in range(nx):
            if i < d:
                dist = (d - i) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_x[i, :, :] = sigma_val
            elif i >= nx - d:
                dist = (i - (nx - 1 - d)) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_x[i, :, :] = sigma_val

        # y方向PML
        for j in range(ny):
            if j < d:
                dist = (d - j) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_y[:, j, :] = sigma_val
            elif j >= ny - d:
                dist = (j - (ny - 1 - d)) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_y[:, j, :] = sigma_val

        # z方向PML
        for k in range(nz):
            if k < d:
                dist = (d - k) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_z[:, :, k] = sigma_val
            elif k >= nz - d:
                dist = (k - (nz - 1 - d)) / d
                sigma_val = sigma_m * (dist ** m)
                self.sigma_z[:, :, k] = sigma_val

    def update_electric_pml(self, Ex, Ey, Ez, Hx, Hy, Hz, dt, epsilon):
        """
        在PML区域内更新电场（分裂场形式）。
        非PML区域保持输入场不变。
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz
        nx, ny, nz = Ex.shape

        # PML区域掩码
        pml_mask = (self.sigma_x > 1e-15) | (self.sigma_y > 1e-15) | (self.sigma_z > 1e-15)

        if not np.any(pml_mask):
            return Ex, Ey, Ez

        # 复制输入场（非PML区域将保持不变）
        Ex_out = Ex.copy()
        Ey_out = Ey.copy()
        Ez_out = Ez.copy()

        eps_safe = np.where(np.abs(epsilon) < 1e-30, 1e-30, epsilon)

        # --- Ex更新 ---
        dHz_dy = np.zeros_like(Ex)
        dHy_dz = np.zeros_like(Ex)
        dHz_dy[:, :-1, :] = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
        dHy_dz[:, :, :-1] = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz

        denom_y = 1.0 + 0.5 * dt * self.sigma_y / eps_safe
        denom_z = 1.0 + 0.5 * dt * self.sigma_z / eps_safe
        denom_y = np.where(np.abs(denom_y) < 1e-15, 1e-15, denom_y)
        denom_z = np.where(np.abs(denom_z) < 1e-15, 1e-15, denom_z)

        self.Ex_y = ((1.0 - 0.5 * dt * self.sigma_y / eps_safe) * self.Ex_y +
                     (dt / eps_safe) * dHz_dy) / denom_y
        self.Ex_z = ((1.0 - 0.5 * dt * self.sigma_z / eps_safe) * self.Ex_z -
                     (dt / eps_safe) * dHy_dz) / denom_z
        Ex_new = self.Ex_y + self.Ex_z
        Ex_out = np.where(pml_mask, Ex_new, Ex_out)

        # --- Ey更新 ---
        dHz_dx = np.zeros_like(Ey)
        dHx_dz = np.zeros_like(Ey)
        dHz_dx[:-1, :, :] = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
        dHx_dz[:, :, :-1] = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz

        denom_x = 1.0 + 0.5 * dt * self.sigma_x / eps_safe
        denom_x = np.where(np.abs(denom_x) < 1e-15, 1e-15, denom_x)

        self.Ey_x = ((1.0 - 0.5 * dt * self.sigma_x / eps_safe) * self.Ey_x -
                     (dt / eps_safe) * dHz_dx) / denom_x
        self.Ey_z = ((1.0 - 0.5 * dt * self.sigma_z / eps_safe) * self.Ey_z +
                     (dt / eps_safe) * dHx_dz) / denom_z
        Ey_new = self.Ey_x + self.Ey_z
        Ey_out = np.where(pml_mask, Ey_new, Ey_out)

        # --- Ez更新 ---
        dHy_dx = np.zeros_like(Ez)
        dHx_dy = np.zeros_like(Ez)
        dHy_dx[:-1, :, :] = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx
        dHx_dy[:, :-1, :] = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy

        self.Ez_x = ((1.0 - 0.5 * dt * self.sigma_x / eps_safe) * self.Ez_x +
                     (dt / eps_safe) * dHy_dx) / denom_x
        self.Ez_y = ((1.0 - 0.5 * dt * self.sigma_y / eps_safe) * self.Ez_y -
                     (dt / eps_safe) * dHx_dy) / denom_y
        Ez_new = self.Ez_x + self.Ez_y
        Ez_out = np.where(pml_mask, Ez_new, Ez_out)

        return Ex_out, Ey_out, Ez_out

    def update_magnetic_pml(self, Hx, Hy, Hz, Ex, Ey, Ez, dt, mu):
        """
        在PML区域内更新磁场（分裂场形式）。
        非PML区域保持输入场不变。
        """
        dx, dy, dz = self.grid.dx, self.grid.dy, self.grid.dz

        pml_mask = (self.sigma_x > 1e-15) | (self.sigma_y > 1e-15) | (self.sigma_z > 1e-15)

        if not np.any(pml_mask):
            return Hx, Hy, Hz

        Hx_out = Hx.copy()
        Hy_out = Hy.copy()
        Hz_out = Hz.copy()

        mu_safe = np.where(np.abs(mu) < 1e-30, 1e-30, mu)

        denom_x = 1.0 + 0.5 * dt * self.sigma_x / mu_safe
        denom_y = 1.0 + 0.5 * dt * self.sigma_y / mu_safe
        denom_z = 1.0 + 0.5 * dt * self.sigma_z / mu_safe
        denom_x = np.where(np.abs(denom_x) < 1e-15, 1e-15, denom_x)
        denom_y = np.where(np.abs(denom_y) < 1e-15, 1e-15, denom_y)
        denom_z = np.where(np.abs(denom_z) < 1e-15, 1e-15, denom_z)

        # --- Hx更新 ---
        dEy_dz = np.zeros_like(Hx)
        dEz_dy = np.zeros_like(Hx)
        dEy_dz[:, :, :-1] = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        dEz_dy[:, :-1, :] = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy

        self.Hx_y = ((1.0 - 0.5 * dt * self.sigma_y / mu_safe) * self.Hx_y -
                     (dt / mu_safe) * dEz_dy) / denom_y
        self.Hx_z = ((1.0 - 0.5 * dt * self.sigma_z / mu_safe) * self.Hx_z +
                     (dt / mu_safe) * dEy_dz) / denom_z
        Hx_new = self.Hx_y + self.Hx_z
        Hx_out = np.where(pml_mask, Hx_new, Hx_out)

        # --- Hy更新 ---
        dEz_dx = np.zeros_like(Hy)
        dEx_dz = np.zeros_like(Hy)
        dEz_dx[:-1, :, :] = (Ez[1:, :, :] - Ez[:-1, :, :]) / dx
        dEx_dz[:, :, :-1] = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz

        self.Hy_x = ((1.0 - 0.5 * dt * self.sigma_x / mu_safe) * self.Hy_x +
                     (dt / mu_safe) * dEz_dx) / denom_x
        self.Hy_z = ((1.0 - 0.5 * dt * self.sigma_z / mu_safe) * self.Hy_z -
                     (dt / mu_safe) * dEx_dz) / denom_z
        Hy_new = self.Hy_x + self.Hy_z
        Hy_out = np.where(pml_mask, Hy_new, Hy_out)

        # --- Hz更新 ---
        dEy_dx = np.zeros_like(Hz)
        dEx_dy = np.zeros_like(Hz)
        dEy_dx[:-1, :, :] = (Ey[1:, :, :] - Ey[:-1, :, :]) / dx
        dEx_dy[:, :-1, :] = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy

        self.Hz_x = ((1.0 - 0.5 * dt * self.sigma_x / mu_safe) * self.Hz_x -
                     (dt / mu_safe) * dEy_dx) / denom_x
        self.Hz_y = ((1.0 - 0.5 * dt * self.sigma_y / mu_safe) * self.Hz_y +
                     (dt / mu_safe) * dEx_dy) / denom_y
        Hz_new = self.Hz_x + self.Hz_y
        Hz_out = np.where(pml_mask, Hz_new, Hz_out)

        return Hx_out, Hy_out, Hz_out

    def compute_reflection_estimate(self):
        """
        估计PML的反射系数。

        R ≈ exp(-2 η₀ σ_max d_pml / (m+1))
        """
        d_pml = self.pml_thickness * min(self.grid.dx, self.grid.dy, self.grid.dz)
        R = np.exp(-2.0 * ETA_0 * self.sigma_max * d_pml / (self.order + 1))
        return R
