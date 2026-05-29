"""
hydrodynamics.py
================
拉格朗日球对称流体力学求解模块。

融合原项目 393_fem1d_lagrange（一维 Lagrange 有限元刚度矩阵组装）与
967_r83v（三对角线性系统求解，含共轭梯度法）的核心算法，
求解球坐标一维拉格朗日流体方程组：

    dr/dt = u                                    (速度定义)
    dm/dt = 0                                    (质量守恒)
    du/dt = -V0 * dP/dm - (2/r) * P/rho          (动量方程)
    de/dt = -P * dV/dt + S_e                     (能量方程)

其中 V = 1/rho 为比容，m 为拉格朗日质量坐标，
S_e 包含激光加热、热传导、聚变能量沉积等源项。

数值方法：
- 空间离散：一维有限体积 / Lagrange 有限元
- 人工粘性：von Neumann-Richtmyer 型
- 时间积分：显式 predictor-corrector
"""

import numpy as np
from typing import Tuple
from utils import safe_divide, clamp_array
from icf_parameters import NP, PC, TP
from mesh_generator import RadialMesh
from state_equation import total_pressure, sound_speed, electron_number_density


class LagrangeHydro:
    """一维球对称拉格朗日流体动力学求解器。"""

    def __init__(self, mesh: RadialMesh):
        self.mesh = mesh
        n = mesh.n_cells

        # 基本变量
        self.rho = np.zeros(n)
        self.u = np.zeros(n + 1)      # 节点速度（n+1个节点）
        self.P = np.zeros(n)          # 单元压强
        self.e = np.zeros(n)          # 比内能 [J/kg]
        self.T_e = np.zeros(n)        # 电子温度 [K]
        self.T_i = np.zeros(n)        # 离子温度 [K]

        # 辅助量
        self.mass = np.zeros(n)       # 单元质量
        self.vol = np.zeros(n)        # 单元体积
        self.q_art = np.zeros(n)      # 人工粘性压
        self.c_s = np.zeros(n)        # 声速

        self._initialize()

    def _initialize(self):
        """初始化流体变量。"""
        n = self.mesh.n_cells
        self.vol = self.mesh.cell_volumes()

        for i in range(n):
            self.rho[i] = self.mesh.get_density_by_zone(i)
            self.mass[i] = self.rho[i] * self.vol[i]
            # 初始温度: 均匀低温 (20 K)，避免初始巨大压强梯度
            self.T_e[i] = 20.0
            self.T_i[i] = 20.0

            # 初始内能（简化）
            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                mass_per_atom = TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO
            else:
                mass_per_atom = 2.5 * 1.0e-3 / PC.AVOGADRO
            self.e[i] = 1.5 * PC.BOLTZMANN * self.T_e[i] / mass_per_atom

        # 初始压强与声速
        self._update_pressure()

    def _update_pressure(self):
        """根据状态方程更新压强与声速，使用 Saha 电离度。"""
        from state_equation import ionization_state_Saha
        n = self.mesh.n_cells
        for i in range(n):
            zone = self.mesh.get_material_zone(i)

            # TODO: 根据材料区确定正确的核电荷数 Z_nuc、平均原子质量 A_avg
            # 和电离能 ion_E。当前硬编码的参数可能不正确，需要与热传导
            # 模块中的 Z_eff 计算保持一致。
            # 涉及材料: ablator (CH 塑料), dt_ice (固态 DT), gas (气态 DT)
            if zone == "ablator":
                Z_nuc = 6.0
                A_avg = TP.ablator_average_atomic_mass
                ion_E = 11.3 * PC.ELEMENTARY_CHARGE
            elif zone == "dt_ice":
                Z_nuc = 1.0
                A_avg = 2.5
                ion_E = 13.6 * PC.ELEMENTARY_CHARGE
            else:
                Z_nuc = 1.0
                A_avg = 2.5
                ion_E = 13.6 * PC.ELEMENTARY_CHARGE

            # TODO: 调用 Saha 方程计算电离度，并处理边界条件。
            # 注意：Z_eff 的计算结果需要与 main.py 中热传导模块使用的
            # Z_eff 模型保持一致。
            Z_eff = ionization_state_Saha(self.rho[i], self.T_e[i], Z_nuc, ion_E)
            Z_eff = max(Z_eff, 1.0e-6)

            self.P[i] = total_pressure(self.rho[i], self.T_e[i], self.T_i[i], Z_eff, A_avg)
            self.c_s[i] = sound_speed(self.rho[i], self.T_e[i], self.T_i[i], Z_eff, A_avg)

    def _compute_artificial_viscosity(self) -> np.ndarray:
        """
        von Neumann-Richtmyer 人工粘性:
            q = C0^2 * rho * (du)^2   当 du < 0 (压缩)
            q = 0                     当 du >= 0 (膨胀)
        线性项: q_lin = C1 * rho * c_s * |du|
        """
        n = self.mesh.n_cells
        q = np.zeros(n)
        C0 = 2.0   # 二次项系数
        C1 = 0.5   # 线性项系数

        for i in range(n):
            du = self.u[i + 1] - self.u[i]
            if du < 0.0:
                dr = self.mesh.r[i + 1] - self.mesh.r[i]
                strain_rate = du / max(dr, 1.0e-15)
                q_quad = C0**2 * self.rho[i] * dr**2 * strain_rate**2
                q_lin = C1 * self.rho[i] * self.c_s[i] * dr * abs(strain_rate)
                q[i] = q_quad + q_lin
            else:
                q[i] = 0.0

        return q

    def compute_time_step(self) -> float:
        """
        CFL 时间步长限制:
            dt = CFL * min( dr / (c_s + |u|) )
        """
        n = self.mesh.n_cells
        dt_min = NP.MAX_DT

        for i in range(n):
            dr = self.mesh.r[i + 1] - self.mesh.r[i]
            if dr <= 1.0e-15:
                continue
            speed = self.c_s[i] + abs(self.u[i]) + abs(self.u[i + 1])
            if speed < 1.0e-10:
                dt_local = NP.MAX_DT
            else:
                dt_local = NP.CFL * dr / speed
            dt_min = min(dt_min, dt_local)

        return clamp_array(np.array([dt_min]), NP.MIN_DT, NP.MAX_DT)[0]

    def momentum_equation_rhs(self) -> np.ndarray:
        """
        动量方程右端项（节点加速度）。
        球坐标拉格朗日形式:
            du/dt = - (1/rho) * dP/dr - 2*P/(rho*r)
        在节点 i 处离散:
            du_i/dt = (A_{i-1/2}*(P+q)_{i-1} - A_{i+1/2}*(P+q)_i) / m_node
                      - 2 * P_avg / (rho_avg * r_i)
        其中 m_node = 0.5*(m_{i-1}+m_i)。
        """
        n_nodes = self.mesh.n_nodes
        rhs = np.zeros(n_nodes)
        q = self.q_art

        for i in range(1, n_nodes - 1):
            i_left = i - 1
            r_i = self.mesh.r[i]
            A_i = 4.0 * np.pi * r_i**2
            m_left = self.mass[i_left]
            m_right = self.mass[i]
            P_left = self.P[i_left] + q[i_left]
            P_right = self.P[i] + q[i]

            m_node = 0.5 * (m_left + m_right)
            if m_node < 1.0e-30:
                rhs[i] = 0.0
                continue

            # 压强梯度项 (m/s^2)
            grad_term = A_i * (P_left - P_right) / m_node

            # 球坐标几何源项: -2*P/(rho*r)
            rho_avg = 0.5 * (self.rho[i_left] + self.rho[i])
            P_avg = 0.5 * (P_left + P_right)
            if r_i < 1.0e-15 or rho_avg < 1.0e-30:
                geom_term = 0.0
            else:
                geom_term = -2.0 * P_avg / (rho_avg * r_i)

            rhs[i] = grad_term + geom_term

        # 球心对称边界
        rhs[0] = 0.0
        # 外边界: 自由面条件 (P=0 外侧)
        if n_nodes > 1:
            A_outer = 4.0 * np.pi * self.mesh.r[-1]**2
            m_last = 0.5 * (self.mass[-1] + self.mass[-1])
            if m_last > 1.0e-30:
                rhs[-1] = A_outer * (self.P[-1] + q[-1]) / m_last
            else:
                rhs[-1] = 0.0

        return rhs

    def energy_equation_rhs(self, laser_heating: np.ndarray,
                            fusion_heating: np.ndarray,
                            conduction_work: np.ndarray) -> np.ndarray:
        """
        能量方程右端项:
            de/dt = -(P+q) * d(1/rho)/dt + S_laser + S_fusion + S_cond
        其中 d(1/rho)/dt = (1/rho) * div(u)  （连续性方程）
        球坐标散度:
            div(u) = (1/r^2) * d/dr(r^2 * u)
                   ≈ (r_{i+1}^2 * u_{i+1} - r_i^2 * u_i) / (r_c^2 * dr)
        """
        n = self.mesh.n_cells
        rhs = np.zeros(n)

        for i in range(n):
            r1, r2 = self.mesh.r[i], self.mesh.r[i + 1]
            rc = 0.5 * (r1 + r2)
            dr = r2 - r1

            # 球坐标散度，添加 rc 保护避免球心奇异性
            if rc < 1.0e-12:
                div_u = 0.0
            else:
                div_u = (r2**2 * self.u[i + 1] - r1**2 * self.u[i]) / (rc**2 * max(dr, 1.0e-15))

            # 限制散度幅值，防止数值爆炸
            div_u = np.clip(div_u, -1.0e15, 1.0e15)

            # pdV 功 [J/kg/s] = [Pa] * [1/s] / [kg/m^3] = Pa * m^3 / (kg * s) = J/(kg*s)
            pdV = -(self.P[i] + self.q_art[i]) * div_u / max(self.rho[i], 1.0e-30)

            rhs[i] = pdV + laser_heating[i] + fusion_heating[i] + conduction_work[i]

        return rhs

    def advance(self, dt: float,
                laser_heating: np.ndarray,
                fusion_heating: np.ndarray,
                conduction_work: np.ndarray):
        """
        显式 predictor-corrector 时间推进一个步长。
        """
        n_cells = self.mesh.n_cells
        n_nodes = self.mesh.n_nodes

        # 1. 人工粘性
        self.q_art = self._compute_artificial_viscosity()

        # 2. 保存旧状态
        u_old = self.u.copy()
        e_old = self.e.copy()
        r_old = self.mesh.r.copy()

        # 3. Predictor: 计算右端项
        rhs_u = self.momentum_equation_rhs()
        rhs_e = self.energy_equation_rhs(laser_heating, fusion_heating, conduction_work)

        # 4. 预测步
        u_pred = u_old + dt * rhs_u
        e_pred = e_old + dt * rhs_e

        # 更新网格位置（拉格朗日）
        r_pred = r_old.copy()
        for i in range(n_nodes):
            r_pred[i] = r_old[i] + dt * u_pred[i]

        # 保证球心固定且外半径非负
        r_pred[0] = 0.0
        r_pred = np.maximum(r_pred, 0.0)
        # 保证单调性
        for i in range(1, n_nodes):
            if r_pred[i] < r_pred[i - 1] + 1.0e-15:
                r_pred[i] = r_pred[i - 1] + 1.0e-15

        # 5. 更新预测状态的物理量
        self.mesh.r = r_pred
        self.u = u_pred
        self.e = e_pred

        # 更新密度
        for i in range(n_cells):
            new_vol = 4.0 * np.pi / 3.0 * (r_pred[i + 1]**3 - r_pred[i]**3)
            self.vol[i] = max(new_vol, 1.0e-30)
            self.rho[i] = self.mass[i] / self.vol[i]

        # 温度更新（假设定容），限制变化幅度
        for i in range(n_cells):
            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                cv = 1.5 * PC.BOLTZMANN / (TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO)
            else:
                cv = 1.5 * PC.BOLTZMANN / (2.5 * 1.0e-3 / PC.AVOGADRO)
            dT = self.e[i] - e_old[i]
            # 限制单步温度变化不超过 50%
            dT_clamped = np.clip(dT, -0.5 * self.T_e[i] * cv, 0.5 * self.T_e[i] * cv)
            self.T_e[i] = max(self.T_e[i] + dT_clamped / max(cv, 1.0e-30), 1.0)
            self.T_i[i] = self.T_e[i]

        self._update_pressure()

        # 6. Corrector: 用预测状态重新计算右端项
        self.q_art = self._compute_artificial_viscosity()
        rhs_u2 = self.momentum_equation_rhs()
        rhs_e2 = self.energy_equation_rhs(laser_heating, fusion_heating, conduction_work)

        # 7. 修正步（梯形法则），限制速度变化
        du = 0.5 * dt * (rhs_u + rhs_u2)
        du_clamped = np.clip(du, -1.0e6, 1.0e6)
        self.u = u_old + du_clamped
        self.e = e_old + 0.5 * dt * (rhs_e + rhs_e2)

        # 最终网格更新
        for i in range(n_nodes):
            self.mesh.r[i] = r_old[i] + dt * self.u[i]
        self.mesh.r[0] = 0.0
        self.mesh.r = np.maximum(self.mesh.r, 0.0)
        for i in range(1, n_nodes):
            if self.mesh.r[i] < self.mesh.r[i - 1] + 1.0e-15:
                self.mesh.r[i] = self.mesh.r[i - 1] + 1.0e-15

        # 最终密度与温度
        for i in range(n_cells):
            new_vol = 4.0 * np.pi / 3.0 * (self.mesh.r[i + 1]**3 - self.mesh.r[i]**3)
            self.vol[i] = max(new_vol, 1.0e-30)
            self.rho[i] = self.mass[i] / self.vol[i]

            zone = self.mesh.get_material_zone(i)
            if zone == "ablator":
                cv = 1.5 * PC.BOLTZMANN / (TP.ablator_average_atomic_mass * 1.0e-3 / PC.AVOGADRO)
            else:
                cv = 1.5 * PC.BOLTZMANN / (2.5 * 1.0e-3 / PC.AVOGADRO)
            dT = self.e[i] - e_old[i]
            dT_clamped = np.clip(dT, -0.5 * self.T_e[i] * cv, 0.5 * self.T_e[i] * cv)
            self.T_e[i] = max(self.T_e[i] + dT_clamped / max(cv, 1.0e-30), 1.0)
            self.T_i[i] = self.T_e[i]

        self._update_pressure()

    def get_kinetic_energy(self) -> float:
        """计算总动能。"""
        n_nodes = self.mesh.n_nodes
        ke = 0.0
        for i in range(n_nodes - 1):
            # 节点质量近似
            m_node = 0.5 * (self.mass[max(i - 1, 0)] + self.mass[min(i, self.mesh.n_cells - 1)])
            m_node = max(m_node, 0.0)
            ke += 0.5 * m_node * self.u[i]**2
        return ke

    def get_internal_energy(self) -> float:
        """计算总内能。"""
        return float(np.sum(self.mass * self.e))
