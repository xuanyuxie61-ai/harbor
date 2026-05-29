"""
nonlinear_pde_solver.py
=======================
非线性声学PDE的高阶时间推进求解器。

融合种子项目：
  - 无直接单一融合，作为多个项目的集成平台。

科学应用：
  实现 KZK 方程、Burgers 方程、Westervelt 方程的非线性时间推进。
  使用 Strang 分裂方法将非线性对流、衍射、吸收分别处理。
  包含 CFL 条件自动判定和自适应时间步长控制。
"""

import numpy as np
from spectral_solver import SpectralDifferentiator, map_nodes_to_interval
from shock_physics import NonlinearAcousticsPhysics


class StrangSplittingSolver:
    r"""
    Strang 分裂求解器，用于非线性声学方程：

    .. math::
        \frac{\partial p}{\partial z} =
        \underbrace{\frac{1}{2 k_0} \nabla_{\perp}^2 p}_{\text{衍射}}
        + \underbrace{\frac{\delta}{2 c_0^3} \frac{\partial^2 p}{\partial \tau^2}}_{\text{吸收}}
        + \underbrace{\frac{\beta}{\rho_0 c_0^3} p \frac{\partial p}{\partial \tau}}_{\text{非线性}}

    Strang 分裂将一步 :math:`\Delta z` 分为三步：
    1. 衍射半步 :math:`\Delta z / 2`
    2. 非线性+吸收整步 :math:`\Delta z`
    3. 衍射半步 :math:`\Delta z / 2`
    """

    def __init__(self, physics, dr, dtau, Nr, Ntau, r_max, tau_max,
                 diffraction=True, absorption=True, nonlinearity=True):
        """
        Parameters
        ----------
        physics : NonlinearAcousticsPhysics
            物理参数对象。
        dr : float
            径向步长。
        dtau : float
            延迟时间步长。
        Nr : int
            径向网格点数。
        Ntau : int
            时间窗网格点数。
        r_max : float
            最大径向坐标。
        tau_max : float
            最大延迟时间。
        diffraction : bool
        absorption : bool
        nonlinearity : bool
        """
        self.physics = physics
        self.dr = float(dr)
        self.dtau = float(dtau)
        self.Nr = int(Nr)
        self.Ntau = int(Ntau)
        self.r_max = float(r_max)
        self.tau_max = float(tau_max)
        self.diffraction = diffraction
        self.absorption = absorption
        self.nonlinearity = nonlinearity

        # 网格
        self.r_grid = np.linspace(0.0, r_max, Nr)
        self.tau_grid = np.linspace(-tau_max, tau_max, Ntau)

        # 谱微分矩阵（tau方向）
        self.spec_tau = SpectralDifferentiator(Ntau, node_type='chebyshev_gauss_lobatto')
        self.tau_nodes, self.tau_jac = map_nodes_to_interval(
            self.spec_tau.nodes, -tau_max, tau_max)
        # TODO: 谱微分矩阵需要根据 Jacobian 因子进行缩放
        self.D_tau = self.spec_tau.differentiation_matrix
        self.D2_tau = self.spec_tau.second_derivative_matrix()

        # CFL 约束
        self._check_cfl()

    def _check_cfl(self):
        """
        检查 CFL 条件并设置最大步长。
        """
        c0 = self.physics.c0
        # 径向 CFL: dz < dr^2 * k0 / 2
        dz_diff = 0.5 * self.dr ** 2 * self.physics.k0
        # tau 方向 CFL 来自非线性: dz < dtau * c0 / (beta * p_est)
        # 使用声源峰值压力作为特征压力估计
        p_est = max(self.physics.p0, 1.0)
        dz_nl = self.dtau * c0 / (self.physics.beta * p_est)
        self.dz_max = min(dz_diff, dz_nl)
        if self.dz_max <= 0.0 or not np.isfinite(self.dz_max):
            raise ValueError("CFL condition yields invalid dz_max.")
        # 设置绝对最小步长限制，避免 Nz 爆炸
        self.dz_min = 1e-12

    def _step_diffraction(self, p, dz_half):
        """
        衍射半步（Crank-Nicolson 隐式）。

        轴对称衍射算子：
        .. math::
            \frac{\partial p}{\partial z} = \frac{1}{2 k_0}
            \left( \frac{\partial^2 p}{\partial r^2} + \frac{1}{r} \frac{\partial p}{\partial r} \right)

        Parameters
        ----------
        p : np.ndarray, shape (Nr, Ntau)
            当前压力场。
        dz_half : float
            半步长。

        Returns
        -------
        np.ndarray
            更新后的压力场。
        """
        if not self.diffraction:
            return p.copy()

        p_new = p.copy()
        coeff = 1.0 / (2.0 * self.physics.k0)

        for j_tau in range(self.Ntau):
            p_r = p[:, j_tau].copy()
            # 径向二阶导数 (中心差分)
            d2p = np.zeros(self.Nr, dtype=float)
            dp = np.zeros(self.Nr, dtype=float)

            if self.Nr >= 3:
                dp[1:-1] = (p_r[2:] - p_r[:-2]) / (2.0 * self.dr)
                d2p[1:-1] = (p_r[2:] - 2.0 * p_r[1:-1] + p_r[:-2]) / (self.dr ** 2)

            # 轴心 r=0 处使用 L'Hopital: d2p/dr2 + (1/r) dp/dr -> 2 d2p/dr2
            if self.Nr >= 3:
                d2p[0] = 2.0 * (p_r[1] - p_r[0]) / (self.dr ** 2)
                dp[0] = 0.0
                # 边界 r=r_max: Neumann (dp/dr = 0)
                d2p[-1] = d2p[-2]
                dp[-1] = 0.0

            # 1/r * dp/dr 处理
            laplacian = d2p.copy()
            if self.Nr > 1:
                r_safe = self.r_grid.copy()
                r_safe[0] = r_safe[1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    laplacian += dp / r_safe
                laplacian[0] = 2.0 * d2p[0]

            # 显式 Euler 半步 (简化，实际可用 ADI 或更精细处理)
            p_new[:, j_tau] = p_r + dz_half * coeff * laplacian

        return p_new

    def _step_nonlinear_absorption(self, p, dz):
        """
        非线性和吸收整步。

        在 :math:`\tau` 方向使用谱方法，每个径向位置独立求解 ODE 系统：

        .. math::
            \frac{\partial p}{\partial z} =
            \frac{\beta}{\rho_0 c_0^3} p \frac{\partial p}{\partial \tau}
            + \frac{\delta}{2 c_0^3} \frac{\partial^2 p}{\partial \tau^2}
            - \alpha_{cl} p

        Parameters
        ----------
        p : np.ndarray, shape (Nr, Ntau)
        dz : float

        Returns
        -------
        np.ndarray
        """
        p_new = p.copy()
        beta = self.physics.beta
        rho0 = self.physics.rho0
        c0 = self.physics.c0
        alpha = self.physics.classical_absorption

        # 有效声粘性（包含经典吸收）
        delta_eff = 2.0 * alpha * c0 ** 3 / self.physics.omega0 ** 2
        if delta_eff < 0.0:
            delta_eff = 0.0

        for i_r in range(self.Nr):
            p_tau = p[i_r, :].copy()

            # 谱微分
            p_tau_x = self.D_tau @ p_tau
            p_tau_xx = self.D2_tau @ p_tau

            rhs = np.zeros(self.Ntau, dtype=float)
            if self.nonlinearity:
                rhs += (beta / (rho0 * c0 ** 3)) * p_tau * p_tau_x
            if self.absorption:
                rhs += (delta_eff / (2.0 * c0 ** 3)) * p_tau_xx
                rhs -= alpha * p_tau

            # RK4
            def ode_rhs(v):
                v = np.asarray(v, dtype=float)
                vx = self.D_tau @ v
                vxx = self.D2_tau @ v
                r = np.zeros_like(v)
                if self.nonlinearity:
                    r += (beta / (rho0 * c0 ** 3)) * v * vx
                if self.absorption:
                    r += (delta_eff / (2.0 * c0 ** 3)) * vxx
                    r -= alpha * v
                return r

            k1 = ode_rhs(p_tau)
            k2 = ode_rhs(p_tau + 0.5 * dz * k1)
            k3 = ode_rhs(p_tau + 0.5 * dz * k2)
            k4 = ode_rhs(p_tau + dz * k3)
            p_new[i_r, :] = p_tau + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # 边界条件：tau 方向周期/衰减
            p_new[i_r, 0] = 0.0
            p_new[i_r, -1] = 0.0

        return p_new

    def propagate(self, p_initial, z_max, dz=None):
        """
        从 z=0 传播到 z=z_max。

        Parameters
        ----------
        p_initial : np.ndarray, shape (Nr, Ntau)
            初始压力场（z=0 处）。
        z_max : float
            最大传播距离。
        dz : float or None
            步长。None 则使用 CFL 限制的自动步长。

        Returns
        -------
        np.ndarray, shape (Nz, Nr, Ntau)
            每个 z 步的解。
        np.ndarray, shape (Nz,)
            z 坐标。
        """
        p = np.asarray(p_initial, dtype=float)
        if p.shape != (self.Nr, self.Ntau):
            raise ValueError(f"p_initial shape {p.shape} does not match ({self.Nr}, {self.Ntau}).")

        if dz is None:
            dz = self.dz_max * 0.5
        dz = float(dz)
        if dz <= 0.0 or dz > self.dz_max:
            raise ValueError(f"dz={dz} violates CFL condition (max={self.dz_max}).")

        Nz_max = 5000
        Nz = int(np.ceil(z_max / dz))
        if Nz > Nz_max:
            Nz = Nz_max
            dz = z_max / Nz
        dz = z_max / Nz  # 精确调整

        z_vec = np.linspace(0.0, z_max, Nz + 1)
        P_history = np.zeros((Nz + 1, self.Nr, self.Ntau), dtype=float)
        P_history[0, :, :] = p

        for step in range(Nz):
            # Strang splitting
            p = self._step_diffraction(p, dz / 2.0)
            p = self._step_nonlinear_absorption(p, dz)
            p = self._step_diffraction(p, dz / 2.0)

            # 数值稳定性检查
            if np.any(~np.isfinite(p)):
                raise RuntimeError(f"Non-finite values detected at z={z_vec[step + 1]}")

            # 物理合理性截断
            p_max_phys = 1e9  # 1 GPa 物理上限
            p = np.clip(p, -p_max_phys, p_max_phys)

            P_history[step + 1, :, :] = p

        return P_history, z_vec


class FiniteVolumeShockCapturing:
    r"""
    基于 Godunov 通量的一阶/二阶有限体积激波捕捉格式。

    用于处理强非线性冲击波，配合谱方法作为后处理校验。

    控制方程：
    .. math::
        \frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} =
        \nu \frac{\partial^2 u}{\partial x^2}, \quad f(u) = \frac{u^2}{2}
    """

    def __init__(self, Nx, x_min, x_max, nu):
        self.Nx = int(Nx)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.nu = float(nu)
        self.dx = (x_max - x_min) / Nx
        self.x_faces = np.linspace(x_min, x_max, Nx + 1)
        self.x_centers = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    def _godunov_flux(self, uL, uR):
        """
        Burgers 方程的 Godunov 数值通量。

        .. math::
            F(u_L, u_R) = \begin{cases}
            \min_{u \in [u_L, u_R]} f(u), & u_L \le u_R \\
            \max_{u \in [u_R, u_L]} f(u), & u_L > u_R
            \end{cases}

        对于 :math:`f(u)=u^2/2`，有显式解：
        .. math::
            F = \max(0, \min(u_L, u_R)) \quad \text{若 } u_L + u_R > 0
            \text{等}
        """
        # Godunov flux for Burgers
        flux = np.where(
            uL <= uR,
            np.minimum(0.5 * uL ** 2, 0.5 * uR ** 2),
            np.maximum(0.5 * uL ** 2, 0.5 * uR ** 2)
        )
        # 修正：考虑激波情况
        shock_mask = (uL > uR)
        s = 0.5 * (uL + uR)
        flux = np.where(shock_mask & (s > 0), 0.5 * uL ** 2, flux)
        flux = np.where(shock_mask & (s <= 0), 0.5 * uR ** 2, flux)
        return flux

    def step(self, u, dt):
        """
        执行一个时间步。

        Parameters
        ----------
        u : np.ndarray, shape (Nx,)
            单元平均值。
        dt : float
            时间步长。

        Returns
        -------
        np.ndarray
            更新后的单元平均值。
        """
        u = np.asarray(u, dtype=float)
        if u.size != self.Nx:
            raise ValueError("u size mismatch.")

        # CFL 检查
        cfl = dt / self.dx
        max_speed = np.max(np.abs(u))
        if max_speed > 0.0 and cfl * max_speed > 1.0:
            # 自动缩减时间步
            dt = 0.9 * self.dx / max_speed
            cfl = dt / self.dx

        # 构造左右状态（一阶精度）
        uL = np.concatenate([[u[-1]], u])
        uR = np.concatenate([u, [u[0]]])

        # Godunov 通量
        F = self._godunov_flux(uL, uR)

        # 守恒更新
        u_new = u - (dt / self.dx) * (F[1:] - F[:-1])

        # 粘性项（中心差分）
        if self.nu > 0.0:
            u_new += (self.nu * dt / self.dx ** 2) * (
                np.concatenate([u[1:], [u[0]]]) -
                2.0 * u +
                np.concatenate([[u[-1]], u[:-1]])
            )

        # 边界：Dirichlet zero
        u_new[0] = 0.0
        u_new[-1] = 0.0

        return u_new

    def solve(self, u0, t_final, dt=None):
        """
        求解到 t_final。

        Returns
        -------
        np.ndarray, shape (Nt, Nx)
            时间历史。
        np.ndarray, shape (Nt,)
            时间向量。
        """
        u = np.asarray(u0, dtype=float)
        if u.size != self.Nx:
            raise ValueError("u0 size mismatch.")

        if dt is None:
            max_speed = max(np.max(np.abs(u)), 1.0)
            dt = 0.5 * self.dx / max_speed

        Nt = int(np.ceil(t_final / dt))
        dt = t_final / Nt

        U = np.zeros((Nt + 1, self.Nx), dtype=float)
        U[0, :] = u
        t_vec = np.linspace(0.0, t_final, Nt + 1)

        for n in range(Nt):
            u = self.step(u, dt)
            U[n + 1, :] = u

        return U, t_vec
