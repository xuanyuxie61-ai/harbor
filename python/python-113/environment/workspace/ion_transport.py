"""
ion_transport.py
离子输运方程求解器

基于种子项目 127_burgers_time_viscous 的核心算法：
- burgers_time_viscous: 粘性 Burgers 方程的有限差分求解

在离子通道问题中的应用：
将 Burgers 方程推广为 Nernst-Planck 方程（对流-扩散方程）：

    ∂c_i/∂t = ∇·[ D_i ∇c_i + (D_i z_i e / k_B T) c_i ∇φ ]

即：
    ∂c_i/∂t = D_i ∇^2 c_i + μ_i ∇·(c_i ∇φ)

其中 μ_i = D_i z_i e / k_B T 为离子迁移率。

结合连续性方程和 Poisson 方程，构成 Poisson-Nernst-Planck (PNP) 方程组。
"""

import numpy as np
from finite_difference import apply_laplacian_3d


class NernstPlanckSolver:
    """
    三维 Nernst-Planck 方程的时间推进求解器。

    采用算子分裂法（Operator Splitting）：
        1. 扩散步：∂c/∂t = D ∇^2 c
        2. 迁移步：∂c/∂t = -μ ∇·(c ∇φ)
    """
    def __init__(self, shape, dx, dy, dz, D_k=1.96e-9, D_na=1.33e-9,
                 z_k=1.0, z_na=1.0, T=300.0):
        """
        Parameters
        ----------
        D_k, D_na : float
            扩散系数（m^2/s），K+ 和 Na+
        z_k, z_na : float
            离子电荷数
        """
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.D = {'K': D_k, 'Na': D_na}
        self.z = {'K': z_k, 'Na': z_na}
        kB = 1.380649e-23
        e_charge = 1.602176634e-19
        self.mu = {
            'K': D_k * z_k * e_charge / (kB * T),
            'Na': D_na * z_na * e_charge / (kB * T)
        }
        self.T = T

    def _diffusion_step(self, c, D, dt):
        """
        纯扩散半步：c^{*} = c^n + (dt/2) D ∇^2 c^n
        """
        lap_c = apply_laplacian_3d(c, self.dx, self.dy, self.dz)
        c_star = c + 0.5 * dt * D * lap_c
        # 非负截断
        c_star = np.maximum(c_star, 0.0)
        return c_star

    def _migration_step(self, c, mu, phi, dt):
        """
        迁移半步：c^{n+1} = c^{*} - (dt/2) μ ∇·(c^{*} ∇φ)

        其中 ∇·(c ∇φ) = ∇c · ∇φ + c ∇^2 φ
        """
        dphi_dx = np.gradient(phi, self.dx, axis=0)
        dphi_dy = np.gradient(phi, self.dy, axis=1)
        dphi_dz = np.gradient(phi, self.dz, axis=2)

        dc_dx = np.gradient(c, self.dx, axis=0)
        dc_dy = np.gradient(c, self.dy, axis=1)
        dc_dz = np.gradient(c, self.dz, axis=2)

        lap_phi = apply_laplacian_3d(phi, self.dx, self.dy, self.dz)

        div_c_grad_phi = dc_dx * dphi_dx + dc_dy * dphi_dy + dc_dz * dphi_dz + c * lap_phi
        c_new = c - 0.5 * dt * mu * div_c_grad_phi
        c_new = np.maximum(c_new, 0.0)
        return c_new

    def solve_step(self, c_k, c_na, phi, dt):
        """
        一个完整的 PNP 时间步（Strang 分裂，二阶精度）。

        Step 1: c^{*} = c^n + (dt/2) D ∇^2 c^n
        Step 2: c^{**} = c^{*} - (dt/2) μ ∇·(c^{*} ∇φ)
        Step 3: c^{***} = c^{**} + (dt/2) D ∇^2 c^{**}
        """
        # TODO: Hole 3 — 实现 Strang 算子分裂 PNP 时间推进
        # 科学背景：对每个离子（K+, Na+）执行 Diffusion-Migration-Diffusion 三步
        # 提示：
        #   - 先对 c_k 执行扩散半步 → 迁移半步 → 扩散半步
        #   - 再对 c_na 执行同样的三步
        #   - 使用 self._diffusion_step 和 self._migration_step
        #   - 注意区分 K+ 和 Na+ 的 D 和 μ（self.D['K']/self.D['Na'], self.mu['K']/self.mu['Na']）
        raise NotImplementedError("Hole 3: 请实现 Strang 算子分裂 PNP 时间步")

    def steady_state_flux(self, c, phi, ion='K'):
        """
        计算稳态离子通量密度（单位：mol/(m^2·s)）。

        Nernst-Planck 通量：
            J = -D ∇c - (D z e / k_B T) c ∇φ
        """
        D = self.D[ion]
        mu = self.mu[ion]

        dc_dx = np.gradient(c, self.dx, axis=0)
        dc_dy = np.gradient(c, self.dy, axis=1)
        dc_dz = np.gradient(c, self.dz, axis=2)

        dphi_dx = np.gradient(phi, self.dx, axis=0)
        dphi_dy = np.gradient(phi, self.dy, axis=1)
        dphi_dz = np.gradient(phi, self.dz, axis=2)

        Jx = -D * dc_dx - mu * c * dphi_dx
        Jy = -D * dc_dy - mu * c * dphi_dy
        Jz = -D * dc_dz - mu * c * dphi_dz

        return Jx, Jy, Jz

    def permeability_coefficient(self, c_k, c_na, phi, channel_area=1.0e-18):
        """
        计算离子通透系数 P（单位：m/s）。

        根据 Goldman-Hodgkin-Katz (GHK) 方程：
            J = P * (c_in - c_out * exp(-zFΔφ/RT)) / (1 - exp(-zFΔφ/RT))

        简化估算：P ≈ |<J_z>| / Δc
        """
        Jx_k, Jy_k, Jz_k = self.steady_state_flux(c_k, phi, ion='K')
        Jx_na, Jy_na, Jz_na = self.steady_state_flux(c_na, phi, ion='Na')

        # 取通道中部的平均轴向通量
        nz = self.shape[2]
        mid = nz // 2
        avg_Jz_k = np.mean(np.abs(Jz_k[:, :, mid]))
        avg_Jz_na = np.mean(np.abs(Jz_na[:, :, mid]))

        # 简化：假设浓度差约 100 mol/m^3
        delta_c = 100.0
        P_k = avg_Jz_k / delta_c
        P_na = avg_Jz_na / delta_c

        selectivity = P_k / (P_na + 1e-30)
        return P_k, P_na, selectivity


def pnp_steady_state_iterator(shape, dx, dy, dz, phi_solver, np_solver,
                               c_k_init, c_na_init, max_iter=50, dt=1e-12):
    """
    PNP 方程组的自洽迭代求解器。

    迭代格式：
        1. 用当前浓度求解 Poisson 方程得到 φ
        2. 用 φ 推进 Nernst-Planck 方程一个时间步
        3. 检查浓度变化是否收敛
    """
    c_k = c_k_init.copy()
    c_na = c_na_init.copy()

    for it in range(max_iter):
        phi = phi_solver.solve(conc_k_bulk=np.mean(c_k), conc_na_bulk=np.mean(c_na))
        c_k_new, c_na_new = np_solver.solve_step(c_k, c_na, phi, dt)

        err_k = np.max(np.abs(c_k_new - c_k))
        err_na = np.max(np.abs(c_na_new - c_na))

        c_k = c_k_new
        c_na = c_na_new

        if max(err_k, err_na) < 1e-6:
            break

    return c_k, c_na, phi
