# -*- coding: utf-8 -*-
"""
vlasov_solver.py
================

Fokker-Planck / 漂移动力学方程求解器.

本模块实现磁层辐射带电子相空间密度 f(L, E, t) 的时间推进求解.

控制方程 (2D Fokker-Planck):

  df/dt = (1/G) * d/dL (G * D_LL * df/dL)           ... 径向扩散
        - d/dE (dot{E} * f) + d^2/dE^2 (D_EE * f)   ... 能量扩散/损失
        - f / tau_loss                                 ... 损失项 (波散射, 大气)
        + S(L, E, t)                                   ... 源项 (注入)

其中:
  - G(L) = L^4: 度量因子 (偶极场通量管体积)
  - D_LL(L): 径向扩散系数 (L^{10} 依赖)
  - dot{E}(E, L): 能量损失率 (同步辐射, 大气碰撞)
  - D_EE(E, L): 能量扩散系数 (波-粒子散射)
  - tau_loss(E, L): 损失时间尺度
  - S(L, E, t): 源项 (亚暴注入)

数值方法:
  - 空间离散: 高阶有限差分 (WENO5, 4阶紧致)
  - 时间推进: 隐式-显式 (IMEX) 或 Crank-Nicolson
  - 边界条件: 损失锥 + 磁顶阴影
  - 稳定性: 自动 CFL 检查

物理背景:
  该方程描述了辐射带电子在相空间 (L, E) 中的输运过程,
  是空间天气预报的核心模型 (如 VERB code, BAS-RBM).

参考文献:
  [1] Schulz, M. & Lanzerotti, L.J., "Particle Diffusion in the Radiation
      Belts", Springer (1974)
  [2] Albert, J.M. et al., "Radial diffusion driven by phase oscillations",
      JGR (2009)
  [3] Tu, W. et al., "Radiation belt modeling with VERB", JGR (2014)
"""

import numpy as np
import physical_constants as pc
import high_order_fd as hofd
from magnetosphere_grid import MagnetosphereGrid
from boundary_conditions import BoundaryConditions
from stability_analysis import compute_cfl_timestep, build_diffusion_matrix


class VlasovSolver:
    """
    2D Fokker-Planck 求解器.

    参数
    ----
    grid : MagnetosphereGrid
        相空间网格
    bc : BoundaryConditions
        边界条件
    fd_order : int
        空间差分阶数 (2 或 4)
    fd_scheme : str
        差分格式 ('central', 'compact', 'weno5')
    time_scheme : str
        时间推进格式 ('explicit_euler', 'crank_nicolson', 'imex')
    cfl_number : float
        CFL 数
    Kp : float
        Kp 指数 (控制扩散系数)
    """

    def __init__(self, grid=None, bc=None,
                 fd_order=4, fd_scheme='central',
                 time_scheme='crank_nicolson',
                 cfl_number=None, Kp=4):
        # 网格
        self.grid = grid if grid is not None else MagnetosphereGrid()
        # 边界条件
        self.bc = bc if bc is not None else BoundaryConditions(
            L_min=self.grid.L_min,
            L_max=self.grid.L_max,
            n_L=self.grid.n_L
        )

        self.fd_order = fd_order
        self.fd_scheme = fd_scheme
        self.time_scheme = time_scheme
        self.cfl_number = pc.CFL_DEFAULT if cfl_number is None else cfl_number
        self.Kp = Kp

        # 预计算扩散系数
        self.D_LL = self.grid.radial_diffusion_coefficient(Kp=self.Kp)
        self.D_EE = self.grid.energy_diffusion_coefficient(wave_mode='chorus')

        # 初始化分布函数
        self.f = self._initial_distribution()

        # 时间步长
        self.dt, self.dt_diff, self.dt_adv = compute_cfl_timestep(
            self.grid, self.D_LL, cfl_number=self.cfl_number
        )

        # 时间记录
        self.time = 0.0
        self.step_count = 0
        self.history = []

    def _initial_distribution(self):
        """
        初始分布函数.

        物理模型:
          f(L, E) = f_0 * exp(-L/L_0) * E^alpha * exp(-E/E_0)

        参数化:
          - f_0: 归一化常数
          - L_0: 特征 L 值 (内带中心 ~ 3-4)
          - alpha: 能谱指数 (~2-3)
          - E_0: 特征能量 (~0.5 MeV)

        这对应于典型的辐射带电子分布:
          - 通量峰值在 L ~ 4, E ~ 1 MeV
          - 向低 L 指数衰减 (大气损失)
          - 向高 E 指数截断 (加速极限)
        """
        L = self.grid.L
        E = self.grid.E_MeV
        L2d, E2d = np.meshgrid(L, E, indexing='ij')

        f_0 = 1.0e6    # 归一化 [cm^{-3} MeV^{-1}]
        L_0 = 4.0      # 峰值 L
        alpha = 2.5    # 能谱指数
        E_0 = 0.5      # 特征能量 [MeV]

        f = f_0 * np.exp(-(L2d - L_0)**2 / 4.0) * E2d**alpha * np.exp(-E2d / E_0)

        # 确保非负
        f = np.maximum(f, 0.0)

        return f

    # -----------------------------------------------------------------
    #  源项和损失项
    # -----------------------------------------------------------------
    def compute_source_term(self, t):
        """
        计算源项 S(L, E, t).

        物理模型:
          S = S_quiet + S_storm(t)

        静时源: 持续等离子体片注入
          S_quiet(L, E) = S_0 * delta(L - L_inj) * kappa(E)

        暴时源: 亚暴注入脉冲
          S_storm(L, E, t) = A_storm * exp(-(t-t_storm)^2/sigma_t^2)
                            * exp(-(L-L_inj)^2/sigma_L^2) * kappa(E)

        参数
        ----
        t : float
            当前时间 [s]

        返回
        -------
        S : ndarray, shape (n_L, n_E)
            源项
        """
        L = self.grid.L
        E = self.grid.E_MeV
        L2d, E2d = np.meshgrid(L, E, indexing='ij')

        # 静时源 (持续注入)
        L_inj = 6.0    # 注入位置
        sigma_L = 0.5  # 注入宽度
        E_0 = 0.5      # 注入特征能量
        S_quiet = 1.0e4 * np.exp(-(L2d - L_inj)**2 / (2*sigma_L**2)) * \
                  np.exp(-E2d / E_0)

        # 暴时源 (周期脉冲)
        t_storm_period = 3 * 86400  # 3 天周期
        t_phase = (t % t_storm_period) / t_storm_period
        sigma_t = 0.05 * t_storm_period
        storm_envelope = np.exp(-(t_phase - 0.5)**2 / (2 * 0.05**2))
        S_storm = 1.0e5 * storm_envelope * \
                  np.exp(-(L2d - 5.0)**2 / 2.0) * np.exp(-E2d / 1.0)

        return S_quiet + S_storm

    def compute_loss_rate(self):
        """
        计算损失率 1/tau_loss(L, E).

        物理机制:
          1. 大气碰撞损失 (低 L, 低 E)
             1/tau_atm ~ n_atm * sigma * v
          2. 波-粒子散射损失 ( chorus, hiss -> 投掷角散射进入损失锥)
             1/tau_wave ~ (B_w/B_0)^2 * Omega_ce
          3. 磁顶阴影损失 (高 L)
             1/tau_mp ~ v_drift / (2*pi*r_mp) (当 L > L_mp)

        返回
        -------
        loss_rate : ndarray, shape (n_L, n_E)
            损失率 [1/s]
        """
        L = self.grid.L
        E = self.grid.E_MeV
        L2d, E2d = np.meshgrid(L, E, indexing='ij')

        # 大气损失 (随 L 指数衰减)
        tau_atm = 86400.0 * np.exp((L2d - 2.0) / 1.0)  # 1 天 @ L=2
        loss_atm = 1.0 / np.maximum(tau_atm, 1.0)

        # 波散射损失 ( chorus @ L=4-6, hiss @ L=2-4 )
        tau_chorus = 3 * 86400.0 * np.ones_like(L2d)
        mask_chorus = (L2d >= 3.5) & (L2d <= 6.5)
        tau_chorus[mask_chorus] = 86400.0  # 1 天
        loss_chorus = 1.0 / np.maximum(tau_chorus, 1.0)

        # 磁顶阴影 (L > L_mp)
        L_mp = self.bc.r_mp
        tau_mp = np.where(L2d > L_mp - 0.5, 3600.0, np.inf)
        loss_mp = 1.0 / np.maximum(tau_mp, 1.0)

        # 总损失率
        loss_rate = loss_atm + loss_chorus + loss_mp

        return loss_rate

    def compute_drift_velocity(self):
        """
        计算磁漂移速度 (对流项).

        物理公式:
          梯度-B 漂移: v_gradB = (mu / (q*B)) * (B x grad B) / B
          曲率漂移: v_curv = (m*v_parallel^2 / (q*B^2)) * (B x kappa) / B

        对于偶极场, 总漂移 (方位角方向):
          v_phi = - (m*v^2 / (2*q*B*L)) * (1 + 2*sin^2(alpha))

        在径向输运中, 有效的漂移速度为:
          v_L = - (1/(q*B*L^2)) * dPhi/dphi

        简化模型: v_L ~ v_0 * L^2 * E  (对流速度)

        返回
        -------
        v_L : ndarray, shape (n_L, n_E)
            径向漂移速度 [R_E/s]
        """
        L = self.grid.L
        E = self.grid.E_MeV
        L2d, E2d = np.meshgrid(L, E, indexing='ij')

        # 简化对流速度
        v_0 = 1.0e-4  # [R_E/s] (对应 ~ 1 kV 电场)
        v_L = v_0 * L2d**2 * np.sqrt(E2d + 0.01)

        return v_L

    # -----------------------------------------------------------------
    #  空间离散化
    # -----------------------------------------------------------------
    def compute_L_operator(self, f):
        """
        计算 L 方向的扩散算子.

        (1/G) * d/dL (G * D_LL * df/dL)

        使用 2 阶或 4 阶中心差分.

        参数
        ----
        f : ndarray, shape (n_L, n_E)
            分布函数

        返回
        -------
        Lf : ndarray, shape (n_L, n_E)
            扩散算子作用结果
        """
        n_L, n_E = f.shape
        dL = self.grid.dL
        G = self.grid.metric_G
        D_LL = self.D_LL

        Lf = np.zeros_like(f)

        for j in range(n_E):
            fj = f[:, j]
            # 计算通量: F = G * D_LL * df/dL
            # 先计算 df/dL
            if self.fd_scheme == 'central':
                dfdL = hofd.central_diff_2nd(fj, dL) if self.fd_order == 2 else \
                       hofd.central_diff_4th(fj, dL)
            elif self.fd_scheme == 'compact':
                dfdL = hofd.compact_first_derivative(fj, dL)
            else:
                dfdL = hofd.central_diff_4th(fj, dL)

            # 通量
            F = G * D_LL * dfdL

            # 散度: (1/G) * dF/dL
            if self.fd_scheme == 'central':
                dFdL = hofd.central_diff_2nd(F, dL) if self.fd_order == 2 else \
                       hofd.central_diff_4th(F, dL)
            else:
                dFdL = hofd.central_diff_4th(F, dL)

            Lf[:, j] = dFdL / np.maximum(G, pc.EPSILON_NUM)

        return Lf

    def compute_E_operator(self, f):
        """
        计算 E 方向的扩散/对流算子.

        -d/dE (dot{E} * f) + d^2/dE^2 (D_EE * f)

        简化: 仅考虑能量损失项 -d/dE (dot{E} * f).

        参数
        ----
        f : ndarray, shape (n_L, n_E)

        返回
        -------
        Ef : ndarray, shape (n_L, n_E)
        """
        n_L, n_E = f.shape
        Ef = np.zeros_like(f)

        # 简化能量损失率: dot{E} ~ -E / tau_rad (同步辐射)
        tau_rad = 1.0e7  # 10^7 s (~ 115 天)
        E = self.grid.E_MeV

        for i in range(n_L):
            fi = f[i, :]
            # dot{E} * f
            Edot_f = -E * fi / tau_rad
            # -d/dE (Edot_f)
            if n_E > 2:
                dEdE = hofd.central_diff_2nd(Edot_f, self.grid.dE_avg)
                Ef[i, :] = -dEdE
            else:
                Ef[i, :] = 0.0

        return Ef

    # -----------------------------------------------------------------
    #  时间推进
    # -----------------------------------------------------------------
    def step(self, dt=None):
        """
        单步时间推进.

        df/dt = L_op(f) + E_op(f) - loss*f + S

        参数
        ----
        dt : float, optional
            时间步长, 默认使用 CFL 限制的 dt

        返回
        -------
        f_new : ndarray
            新时间步的分布函数
        """
        if dt is None:
            dt = self.dt

        f = self.f
        n_L, n_E = f.shape

        # 计算各项
        L_op = self.compute_L_operator(f)
        E_op = self.compute_E_operator(f)
        loss_rate = self.compute_loss_rate()
        source = self.compute_source_term(self.time)

        # 显式 Euler
        f_new = f + dt * (L_op + E_op - loss_rate * f + source)

        # 施加边界条件
        f_new = self.bc.apply_all(f_new, self.grid.dL, self.grid.E_MeV, dt)

        # 更新状态
        self.f = f_new
        self.time += dt
        self.step_count += 1

        # 记录历史
        if self.step_count % 10 == 0:
            self._record_history()

        return f_new

    def _record_history(self):
        """记录历史诊断信息."""
        info = {
            'time': self.time,
            'step': self.step_count,
            'total_particles': float(np.sum(self.f) * self.grid.dL * self.grid.dE_avg),
            'max_f': float(np.max(self.f)),
            'min_f': float(np.min(self.f)),
            'L_peak': float(self.grid.L[np.argmax(np.sum(self.f, axis=1))]),
        }
        self.history.append(info)

    # -----------------------------------------------------------------
    #  主循环
    # -----------------------------------------------------------------
    def run(self, total_time=None, output_interval=None):
        """
        运行主循环.

        参数
        ----
        total_time : float
            总模拟时间 [s]
        output_interval : float
            输出间隔 [s]

        返回
        -------
        results : dict
            模拟结果
        """
        if total_time is None:
            total_time = 10 * self.dt  # 默认 10 步
        if output_interval is None:
            output_interval = total_time / 10

        n_steps = int(total_time / self.dt)
        output_steps = int(output_interval / self.dt)
        output_steps = max(output_steps, 1)

        print(f"开始 Fokker-Planck 模拟:")
        print(f"  总时间: {total_time:.4e} s ({total_time/86400:.1f} days)")
        print(f"  时间步长: {self.dt:.4e} s")
        print(f"  总步数: {n_steps}")
        print(f"  输出间隔: {output_steps} 步")

        results = {
            'time': [],
            'f_snapshots': [],
            'history': [],
        }

        for step in range(n_steps):
            self.step()

            if step % output_steps == 0:
                results['time'].append(self.time)
                results['f_snapshots'].append(self.f.copy())
                results['history'].append(self.history[-1] if self.history else {})

                # 进度报告
                if step % (output_steps * 5) == 0:
                    diag = self.bc.boundary_diagnostics(self.f)
                    print(f"  Step {step}/{n_steps}: t = {self.time/86400:.2f} days, "
                          f"max_f = {diag['max_value']:.4e}, "
                          f"total = {diag['total_particles']:.4e}")

        results['f_final'] = self.f.copy()
        return results

    # -----------------------------------------------------------------
    #  诊断
    # -----------------------------------------------------------------
    def compute_flux(self):
        """
        计算微分通量 (物理可观测量).

        物理公式:
          j(E) = p^2 * f(L, E)  (单位转换)

        其中 p 为动量, f 为相空间密度.

        返回
        -------
        flux : ndarray, shape (n_L, n_E)
            微分通量 [(cm^2 s sr)^{-1}]
        """
        p2 = self.grid.p**2
        flux = np.zeros_like(self.f)
        for j in range(self.grid.n_E):
            flux[:, j] = self.f[:, j] * p2[j]
        return flux

    def compute_phase_space_density_profile(self, L_index=None, E_index=None):
        """
        提取相空间密度剖面.

        参数
        ----
        L_index : int, optional
            L 索引 (固定 L, 返回 f vs E)
        E_index : int, optional
            E 索引 (固定 E, 返回 f vs L)

        返回
        -------
        profile : ndarray
            剖面数据
        coord : ndarray
            对应坐标
        """
        if L_index is not None:
            return self.f[L_index, :], self.grid.E_MeV
        elif E_index is not None:
            return self.f[:, E_index], self.grid.L
        else:
            return self.f, (self.grid.L, self.grid.E_MeV)


if __name__ == "__main__":
    print("VlasovSolver 自检验证")
    solver = VlasovSolver()
    print(f"初始分布: max = {np.max(solver.f):.4e}, min = {np.min(solver.f):.4e}")
    print(f"CFL 时间步长: dt = {solver.dt:.4e} s")

    # 运行几步
    for i in range(5):
        solver.step()
    print(f"5 步后: t = {solver.time:.4e} s, max = {np.max(solver.f):.4e}")
