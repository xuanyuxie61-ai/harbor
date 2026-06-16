"""
qgp_time_integration.py — TVD-RK3 时间推进与自适应时间步
==========================================================

融合种子项目: 018_arenstorf_ode (高精度 ODE 积分器)

本模块实现 QGP 流体力学方程的时间推进, 采用三阶
总变差递减 Runge-Kutta (TVD-RK3) 方法.

TVD-RK3 (Shu & Osher 1988):
----------------------------

对于半离散系统 du/dt = L(u):

阶段 1 (Euler 步):
    u^(1) = u^n + dt * L(u^n)

阶段 2 (凸组合):
    u^(2) = (3/4)*u^n + (1/4)*u^(1) + (1/4)*dt*L(u^(1))

阶段 3 (凸组合):
    u^{n+1} = (1/3)*u^n + (2/3)*u^(2) + (2/3)*dt*L(u^(2))

TVD 性质:
    若 Euler 步 u + dt*L(u) 满足 TVD, 则 TVD-RK3 也满足 TVD.
    TVD 条件: dt <= dt_CFL (与空间离散化相关的 CFL 条件)

精度: 三阶时间精度 O(dt^3)
强稳定保持 (SSP): SSP 系数 C = 1 (最优)

自适应时间步:
    基于 CFL 条件动态调整:
        dt = CFL * min(dx, dy) / (v_max + c_s)
    同时监测守恒量偏差, 若偏差过大则减小 dt.

守恒量监测 (源自 208_conservation_ode):
    类似 Arenstorf 轨道的 Jacobi 常数监测,
    跟踪总能量的相对变化:
        delta_E / E_0 = |E(t) - E(0)| / |E(0)|
    要求 delta_E / E_0 < tolerance
"""

import numpy as np
from typing import Callable, Tuple, Dict
from qgp_config import NumericalParams
from qgp_grid import QGPGrid
from qgp_eos import QGPEquationOfState
from qgp_operators import WENO5Operator
from qgp_conservation import ConservedVariables


class TVDRK3Integrator:
    """
    TVD-RK3 时间积分器

    将守恒律 dU/dt = L(U) + S(U) 推进一个时间步.

    U = (D, Sx, Sy) 为守恒变量
    L(U) 为 WENO5 空间离散化
    S(U) 为几何源项

    Attributes:
        grid: 计算网格
        eos: 状态方程
        operator: WENO5 差分算子
        conserved: 守恒变量管理器
        conservation_history: 守恒量历史
    """

    def __init__(self, grid: QGPGrid, eos: QGPEquationOfState,
                 operator: WENO5Operator, conserved: ConservedVariables):
        """
        Args:
            grid: 计算网格
            eos: 状态方程
            operator: WENO5 算子
            conserved: 守恒变量管理器
        """
        self.grid = grid
        self.eos = eos
        self.operator = operator
        self.conserved = conserved

        # 守恒量监测
        self.conservation_history = []
        self.initial_energy = None

    def compute_rhs(self, D: np.ndarray, Sx: np.ndarray,
                     Sy: np.ndarray, tau: float) -> tuple:
        """
        计算右端项 L(U)

        流程:
        1. 守恒量 -> 原始变量
        2. 计算物理通量 F(U), G(U)
        3. WENO5 重构 + LF 分裂
        4. 几何源项

        dD/dt = -dF_D/dx - dG_D/dy + S_D
        dSx/dt = -dF_Sx/dx - dG_Sx/dy + S_Sx
        dSy/dt = -dF_Sy/dx - dG_Sy/dy + S_Sy

        Args:
            D, Sx, Sy: 守恒变量 (含鬼单元)
            tau: 当前固有时间

        Returns:
            (dD_dt, dSx_dt, dSy_dt) 右端项 (内部区域)
        """
        # 守恒量 -> 原始变量
        e, vx, vy, nB = self.conserved.conserved_to_primitive(D, Sx, Sy, tau)

        # 最大特征速度 (用于 LF 分裂)
        alpha = self.conserved.compute_max_characteristic_speed(e, vx, vy)
        alpha = max(alpha, 0.1)  # 下限保护

        # x 方向通量和导数
        F_D, F_Sx, F_Sy = self.conserved.compute_flux_x(D, Sx, Sy, tau)
        F_D = self.grid.apply_boundary_conditions(F_D)
        F_Sx = self.grid.apply_boundary_conditions(F_Sx)
        F_Sy = self.grid.apply_boundary_conditions(F_Sy)

        dFDdx = self.operator.spatial_derivative_x(F_D, alpha)
        dFSxdx = self.operator.spatial_derivative_x(F_Sx, alpha)
        dFSydx = self.operator.spatial_derivative_x(F_Sy, alpha)

        # y 方向通量和导数
        G_D, G_Sx, G_Sy = self.conserved.compute_flux_y(D, Sx, Sy, tau)
        G_D = self.grid.apply_boundary_conditions(G_D)
        G_Sx = self.grid.apply_boundary_conditions(G_Sx)
        G_Sy = self.grid.apply_boundary_conditions(G_Sy)

        dGDdy = self.operator.spatial_derivative_y(G_D, alpha)
        dGSxdy = self.operator.spatial_derivative_y(G_Sx, alpha)
        dGSydy = self.operator.spatial_derivative_y(G_Sy, alpha)

        # 几何源项
        e_int = self.grid.interior_slice(e)
        vx_int = self.grid.interior_slice(vx)
        vy_int = self.grid.interior_slice(vy)
        D_int = self.grid.interior_slice(D)
        Sx_int = self.grid.interior_slice(Sx)
        Sy_int = self.grid.interior_slice(Sy)

        src_D, src_Sx, src_Sy, _ = self.conserved.geometric_source(
            D_int, Sx_int, Sy_int, e_int, vx_int, vy_int, tau
        )

        # 人工粘滞 (简化版: 直接对守恒量加 Laplacian 扩散)
        ng = self.grid.ng
        D_full = self.grid.apply_boundary_conditions(D)
        Sx_full = self.grid.apply_boundary_conditions(Sx)
        Sy_full = self.grid.apply_boundary_conditions(Sy)

        visc_D = self.operator.artificial_viscosity(D_full)
        visc_Sx = self.operator.artificial_viscosity(Sx_full)
        visc_Sy = self.operator.artificial_viscosity(Sy_full)

        # 汇总右端项
        dD_dt = -(dFDdx + dGDdy) + src_D + visc_D
        dSx_dt = -(dFSxdx + dGSxdy) + src_Sx + visc_Sx
        dSy_dt = -(dFSydx + dGSydy) + src_Sy + visc_Sy

        return dD_dt, dSx_dt, dSy_dt

    def step(self, D: np.ndarray, Sx: np.ndarray, Sy: np.ndarray,
              tau: float, dt: float) -> tuple:
        """
        TVD-RK3 单步推进

        u^(1) = u^n + dt * L(u^n)
        u^(2) = 3/4 * u^n + 1/4 * u^(1) + 1/4 * dt * L(u^(1))
        u^{n+1} = 1/3 * u^n + 2/3 * u^(2) + 2/3 * dt * L(u^(2))

        每一步都需要:
        1. 重新计算右端项 L(u)
        2. 施加边界条件
        3. 物理约束 (正能量, 因果速度)

        Args:
            D, Sx, Sy: 守恒变量 (含鬼单元)
            tau: 当前固有时间
            dt: 时间步长

        Returns:
            (D_new, Sx_new, Sy_new) 新时间步守恒变量
        """
        # 阶段 1: 前向 Euler
        rhs1_D, rhs1_Sx, rhs1_Sy = self.compute_rhs(D, Sx, Sy, tau)
        ng = self.grid.ng
        ny, nx = self.grid.ny, self.grid.nx

        D1 = D.copy()
        Sx1 = Sx.copy()
        Sy1 = Sy.copy()
        D1[ng:ng+ny, ng:ng+nx] += dt * rhs1_D
        Sx1[ng:ng+ny, ng:ng+nx] += dt * rhs1_Sx
        Sy1[ng:ng+ny, ng:ng+nx] += dt * rhs1_Sy

        # 边界条件和物理约束
        D1 = self.grid.apply_boundary_conditions(D1)
        Sx1 = self.grid.apply_boundary_conditions(Sx1)
        Sy1 = self.grid.apply_boundary_conditions(Sy1)

        # 阶段 2: 凸组合
        rhs2_D, rhs2_Sx, rhs2_Sy = self.compute_rhs(D1, Sx1, Sy1, tau + dt)

        D2 = 0.75 * D + 0.25 * D1
        Sx2 = 0.75 * Sx + 0.25 * Sx1
        Sy2 = 0.75 * Sy + 0.25 * Sy1
        D2[ng:ng+ny, ng:ng+nx] += 0.25 * dt * rhs2_D
        Sx2[ng:ng+ny, ng:ng+nx] += 0.25 * dt * rhs2_Sx
        Sy2[ng:ng+ny, ng:ng+nx] += 0.25 * dt * rhs2_Sy

        D2 = self.grid.apply_boundary_conditions(D2)
        Sx2 = self.grid.apply_boundary_conditions(Sx2)
        Sy2 = self.grid.apply_boundary_conditions(Sy2)

        # 阶段 3: 凸组合
        rhs3_D, rhs3_Sx, rhs3_Sy = self.compute_rhs(D2, Sx2, Sy2, tau + 0.5*dt)

        D_new = (1.0/3.0) * D + (2.0/3.0) * D2
        Sx_new = (1.0/3.0) * Sx + (2.0/3.0) * Sx2
        Sy_new = (1.0/3.0) * Sy + (2.0/3.0) * Sy2
        D_new[ng:ng+ny, ng:ng+nx] += (2.0/3.0) * dt * rhs3_D
        Sx_new[ng:ng+ny, ng:ng+nx] += (2.0/3.0) * dt * rhs3_Sx
        Sy_new[ng:ng+ny, ng:ng+nx] += (2.0/3.0) * dt * rhs3_Sy

        D_new = self.grid.apply_boundary_conditions(D_new)
        Sx_new = self.grid.apply_boundary_conditions(Sx_new)
        Sy_new = self.grid.apply_boundary_conditions(Sy_new)

        return D_new, Sx_new, Sy_new

    def adaptive_dt(self, D: np.ndarray, Sx: np.ndarray,
                     Sy: np.ndarray, tau: float) -> float:
        """
        自适应时间步长计算

        CFL 条件:
            dt_CFL = CFL * min(dx, dy) / max(v + c_s)

        守恒量监测:
            若 delta_E/E_0 > tolerance, 减小 dt

        Args:
            D, Sx, Sy: 守恒变量
            tau: 当前时间

        Returns:
            自适应时间步长 dt
        """
        # 转换为原始变量
        e, vx, vy, _ = self.conserved.conserved_to_primitive(D, Sx, Sy, tau)

        # 声速
        cs2 = self.eos.sound_speed_squared(self.grid.interior_slice(e))
        cs_max = float(np.sqrt(np.max(cs2)))

        # 最大流速
        vx_int = self.grid.interior_slice(vx)
        vy_int = self.grid.interior_slice(vy)
        v_max = float(np.max(np.sqrt(vx_int**2 + vy_int**2)))

        # CFL 时间步
        dt = self.grid.cfl_time_step(v_max, cs_max)

        # 守恒量检查
        if self.initial_energy is not None:
            e_int = self.grid.interior_slice(e)
            E_current = self.conserved.total_energy(e_int, vx_int, vy_int,
                                                     self.grid)
            rel_change = abs(E_current - self.initial_energy) / \
                         (abs(self.initial_energy) + 1.0e-15)

            # 如果守恒量偏差过大, 减小 dt
            if rel_change > NumericalParams.CONSERVATION_TOL * 10:
                dt *= 0.5

        return dt

    def monitor_conservation(self, D: np.ndarray, Sx: np.ndarray,
                              Sy: np.ndarray, tau: float, step_num: int):
        """
        监测守恒量 (源自 208_conservation_ode 的 conserved quantity)

        跟踪:
        1. 总能量 E = integral T^{00} dx dy
        2. 总重子数 B = integral gamma*nB dx dy
        3. 总熵 S = integral s*gamma dx dy (应单调不减)

        Args:
            D, Sx, Sy: 守恒变量
            tau: 当前时间
            step_num: 当前步数
        """
        e, vx, vy, nB = self.conserved.conserved_to_primitive(D, Sx, Sy, tau)
        e_int = self.grid.interior_slice(e)
        vx_int = self.grid.interior_slice(vx)
        vy_int = self.grid.interior_slice(vy)
        nB_int = self.grid.interior_slice(nB)

        E_total = self.conserved.total_energy(e_int, vx_int, vy_int, self.grid)
        B_total = self.conserved.total_baryon_number(nB_int, vx_int, vy_int,
                                                      self.grid)
        s_lab = self.conserved.entropy_density_current(e_int, vx_int, vy_int)
        S_total = self.grid.integrate_2d(s_lab)

        if self.initial_energy is None:
            self.initial_energy = E_total

        record = {
            'step': step_num,
            'tau': tau,
            'E_total': E_total,
            'B_total': B_total,
            'S_total': S_total,
            'E_rel_change': abs(E_total - self.initial_energy) / \
                            (abs(self.initial_energy) + 1.0e-15),
        }
        self.conservation_history.append(record)

    def evolve(self, D_init: np.ndarray, Sx_init: np.ndarray,
                Sy_init: np.ndarray, tau_init: float,
                tau_final: float, monitor_interval: int = 10) -> tuple:
        """
        从 tau_init 演化到 tau_final

        主循环:
            while tau < tau_final:
                1. 计算自适应 dt
                2. TVD-RK3 单步推进
                3. 更新 tau
                4. 监测守恒量

        Args:
            D_init, Sx_init, Sy_init: 初始守恒变量
            tau_init: 初始时间 (fm/c)
            tau_final: 终止时间 (fm/c)
            monitor_interval: 监测间隔 (步数)

        Returns:
            (D_final, Sx_final, Sy_final, tau_final, history)
        """
        D = D_init.copy()
        Sx = Sx_init.copy()
        Sy = Sy_init.copy()
        tau = tau_init
        step = 0

        print(f"  [TVD-RK3] 开始演化: tau = {tau:.3f} -> {tau_final:.3f} fm/c")

        while tau < tau_final:
            # 自适应时间步
            dt = self.adaptive_dt(D, Sx, Sy, tau)
            dt = min(dt, tau_final - tau)  # 最后一步精确到终点

            # TVD-RK3 步
            D, Sx, Sy = self.step(D, Sx, Sy, tau, dt)
            tau += dt
            step += 1

            # 监测
            if step % monitor_interval == 0:
                self.monitor_conservation(D, Sx, Sy, tau, step)

            # 进度输出
            if step % 50 == 0:
                e, vx, vy, _ = self.conserved.conserved_to_primitive(
                    D, Sx, Sy, tau)
                e_int = self.grid.interior_slice(e)
                e_max = np.max(e_int)
                e_mean = np.mean(e_int)
                print(f"    step={step:4d}, tau={tau:.4f}, "
                      f"dt={dt:.5f}, e_max={e_max:.4f}, e_mean={e_mean:.4f}")

        # 最终监测
        self.monitor_conservation(D, Sx, Sy, tau, step)

        print(f"  [TVD-RK3] 演化完成: {step} 步, tau = {tau:.4f} fm/c")
        return D, Sx, Sy, tau, self.conservation_history
