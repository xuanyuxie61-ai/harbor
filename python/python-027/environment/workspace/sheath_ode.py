# -*- coding: utf-8 -*-
"""
sheath_ode.py
等离子体鞘层离子密度演化ODE模块
基于种子项目 702_logistic_ode 重构

本模块求解修正的logistic方程描述鞘层中离子密度 n_i(x) 的空间演化，
同时耦合离子速度 v_i(x) 的动量方程，构成一维鞘层流体模型。
"""

import numpy as np
from parameters import get_parameters


class SheathODE:
    """
    一维等离子体鞘层ODE模型
    
    控制方程组:
        (1) 连续性方程:  d/dx (n_i * v_i) = S_ion - S_rec
        (2) 动量方程:    m_i * v_i * dv_i/dx = -Z_i*e * dphi/dx - k_B*T_i * (1/n_i)*dn_i/dx
        (3) 修正logistic密度演化:  dn_i/dx = (n_i/lambda_D)*(1 - n_i/n_0) - alpha_rec*n_i^2
    
    其中:
        - S_ion: 电离源项
        - S_rec: 复合汇项
        - alpha_rec: 复合系数 [m^3/s]
    """

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self._setup_coefficients()

    def _setup_coefficients(self):
        """预计算常用系数"""
        p = self.params
        self.lambda_D = p.debye_length()
        self.c_s = p.ion_sound_speed()
        self.n0 = p.get('n_0')
        self.Te = p.get('T_e')
        self.Ti = p.get('T_i')
        self.mi_amu = p.get('m_i')
        self.Zi = p.get('Z_i')
        # 德uterium 的三体复合系数（典型值，m^3/s）
        self.alpha_rec = 1.0e-20 * (self.Te / 10.0)**(-0.5)
        if self.alpha_rec < 1.0e-25:
            self.alpha_rec = 1.0e-25
        # 电离率系数（简化模型）
        self.k_ion = 1.0e-14 * np.exp(-13.6 / self.Te) if self.Te > 0 else 0.0

    def density_derivative(self, x, n_i, v_i):
        """
        计算 dn_i/dx
        
        修正logistic方程:
            dn_i/dx = (n_i/lambda_D)*(1 - n_i/n_0) - alpha_rec*n_i^2/v_i
        
        第二项为复合损失项，与速度 v_i 成反比（因为连续性）。
        
        Parameters:
            x:   空间位置 [m]
            n_i: 离子密度 [m^-3]
            v_i: 离子速度 [m/s]
        
        Returns:
            dn_i/dx [m^-4]
        """
        # 边界处理：密度和速度必须为正
        if n_i <= 0:
            n_i = 1.0e10
        if v_i <= 0:
            v_i = self.c_s

        # logistic增长项（鞘层压缩效应）
        logistic_term = (n_i / self.lambda_D) * (1.0 - n_i / self.n0)

        # 复合损失项
        if v_i > 1.0e-10:
            rec_term = self.alpha_rec * n_i**2 / v_i
        else:
            rec_term = 0.0

        dnidx = logistic_term - rec_term

        # 数值稳定性：限制增长速率
        max_rate = self.n0 / self.lambda_D
        if abs(dnidx) > max_rate:
            dnidx = np.sign(dnidx) * max_rate

        return dnidx

    def velocity_derivative(self, x, n_i, v_i, e_field):
        """
        计算 dv_i/dx (离子动量方程)
        
        动量方程:
            m_i * v_i * dv_i/dx = Z_i*e*E - (k_B*T_i/n_i)*dn_i/dx
        
        简化为:
            dv_i/dx = (Z_i*e*E)/(m_i*v_i) - (k_B*T_i)/(m_i*v_i*n_i)*dn_i/dx
        
        Parameters:
            x:       空间位置 [m]
            n_i:     离子密度 [m^-3]
            v_i:     离子速度 [m/s]
            e_field: 电场 [V/m]
        
        Returns:
            dv_i/dx [s^-1]
        """
        # TODO: 实现离子动量方程的数值求解
        # 提示: 需考虑电场加速项与热压力梯度项，并做数值稳定性限制
        raise NotImplementedError("Hole_1: 请实现 velocity_derivative 方法")


    def exact_density_solution(self, x_arr):
        """
        无复合(alpha_rec=0)时的精确解（基于原始logistic_exact）:
            n_i(x) = n_0 * n_s * exp(x/lambda_D) / (n_0 + n_s*(exp(x/lambda_D) - 1))
        
        其中 n_s 为鞘层边缘密度（Bohm点），取 n_s = n_0/2
        
        Parameters:
            x_arr: 空间位置数组 [m]
        
        Returns:
            n_i(x) 数组 [m^-3]
        """
        x_arr = np.asarray(x_arr, dtype=float)
        n_s = self.n0 * 0.5  # Bohm点密度约为上游密度的一半
        lam = self.lambda_D
        if lam <= 0:
            return np.full_like(x_arr, self.n0)

        exp_term = np.exp(x_arr / lam)
        numerator = self.n0 * n_s * exp_term
        denominator = self.n0 + n_s * (exp_term - 1.0)

        # 避免除零
        denominator = np.where(denominator < 1.0, 1.0, denominator)

        n_i = numerator / denominator
        return n_i

    def solve_sheath_profile(self, nx=None, x_max=None):
        """
        数值求解鞘层密度和速度剖面
        
        使用四阶Runge-Kutta方法积分ODE系统
        
        Returns:
            x:      空间网格 [m]
            n_i:    离子密度剖面 [m^-3]
            v_i:    离子速度剖面 [m/s]
            phi:    电势剖面 [V]（简化模型）
            e_field: 电场剖面 [V/m]
        """
        if nx is None:
            nx = self.params.get('nx')
        if x_max is None:
            x_max = self.params.get('x_max')

        x = np.linspace(0.0, x_max, nx)
        dx = x[1] - x[0]

        n_i = np.zeros(nx)
        v_i = np.zeros(nx)
        phi = np.zeros(nx)
        e_field = np.zeros(nx)

        # 初始条件（鞘层边缘，Bohm点）
        n_i[0] = self.n0 * 0.5
        v_i[0] = self.c_s
        phi[0] = 0.0

        # 简化电场模型: E(x) ~ (T_e/e) / (lambda_D + x)
        for idx in range(nx):
            denom = self.lambda_D + x[idx]
            if denom > 0:
                e_field[idx] = self.Te / denom
            else:
                e_field[idx] = 0.0

        # RK4积分
        for idx in range(nx - 1):
            xi = x[idx]
            ni = n_i[idx]
            vi = v_i[idx]
            Ei = e_field[idx]

            # k1
            k1_n = self.density_derivative(xi, ni, vi)
            k1_v = self.velocity_derivative(xi, ni, vi, Ei)

            # k2
            k2_n = self.density_derivative(xi + 0.5*dx, ni + 0.5*dx*k1_n, vi + 0.5*dx*k1_v)
            k2_v = self.velocity_derivative(xi + 0.5*dx, ni + 0.5*dx*k1_n, vi + 0.5*dx*k1_v, Ei)

            # k3
            k3_n = self.density_derivative(xi + 0.5*dx, ni + 0.5*dx*k2_n, vi + 0.5*dx*k2_v)
            k3_v = self.velocity_derivative(xi + 0.5*dx, ni + 0.5*dx*k2_n, vi + 0.5*dx*k2_v, Ei)

            # k4
            k4_n = self.density_derivative(xi + dx, ni + dx*k3_n, vi + dx*k3_v)
            k4_v = self.velocity_derivative(xi + dx, ni + dx*k3_n, vi + dx*k3_v, Ei)

            n_i[idx+1] = ni + (dx/6.0)*(k1_n + 2*k2_n + 2*k3_n + k4_n)
            v_i[idx+1] = vi + (dx/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

            # 物理约束
            if n_i[idx+1] < 1.0e10:
                n_i[idx+1] = 1.0e10
            if v_i[idx+1] < self.c_s:
                v_i[idx+1] = self.c_s

            # 电势（积分电场）
            phi[idx+1] = phi[idx] - e_field[idx] * dx

        return x, n_i, v_i, phi, e_field

    def compute_ion_flux(self, n_i, v_i):
        """
        计算离子通量: Gamma_i = n_i * v_i  [m^-2 s^-1]
        """
        return n_i * v_i

    def compute_ion_energy_at_wall(self, v_i_wall):
        """
        计算离子到达壁面的总能量:
            E_i = (1/2)*m_i*v_i^2 + Z_i*e*|phi_w| + Z_i*e*Delta_phi_sheath
        
        Parameters:
            v_i_wall: 壁面处离子速度 [m/s]
        
        Returns:
            E_i: 离子能量 [eV]
        """
        m_p = 1.67262192369e-27
        e_charge = 1.602176634e-19
        mi_kg = self.mi_amu * m_p

        kinetic_ev = 0.5 * mi_kg * v_i_wall**2 / e_charge
        sheath_potential = abs(self.params.sheath_potential())

        E_total = kinetic_ev + self.Zi * sheath_potential + self.Zi * self.Te
        return E_total

    def compute_sheath_edge_mach(self, v_i):
        """
        计算Mach数: M = v_i / c_s
        验证 Bohm 判据 (M >= 1)
        """
        if self.c_s <= 0:
            return np.zeros_like(v_i)
        M = v_i / self.c_s
        return M


def demo_sheath_ode():
    """演示鞘层ODE求解"""
    sheath = SheathODE()
    x, n_i, v_i, phi, e_field = sheath.solve_sheath_profile(nx=128, x_max=0.005)
    gamma = sheath.compute_ion_flux(n_i, v_i)
    M = sheath.compute_sheath_edge_mach(v_i)
    E_wall = sheath.compute_ion_energy_at_wall(v_i[-1])

    print("鞘层ODE求解结果:")
    print(f"  壁面离子密度     = {n_i[-1]:.3e} m^-3")
    print(f"  壁面离子速度     = {v_i[-1]:.3e} m/s")
    print(f"  壁面Mach数       = {M[-1]:.3f}")
    print(f"  壁面离子通量     = {gamma[-1]:.3e} m^-2 s^-1")
    print(f"  壁面离子能量     = {E_wall:.2f} eV")
    print(f"  Bohm判据满足     = {M[0] >= 1.0}")
    return x, n_i, v_i, phi, e_field


if __name__ == "__main__":
    demo_sheath_ode()
