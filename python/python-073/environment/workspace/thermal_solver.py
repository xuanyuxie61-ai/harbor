# -*- coding: utf-8 -*-
"""
thermal_solver.py
高超声速边界层温度场与速度剖面求解

核心算法来源：
- heated_plate: 稳态热方程迭代求解（Jacobi 松弛）

物理背景：
高超声速边界层内，粘性耗散导致显著的气动加热。
可压缩边界层的能量方程（稳态）:

    ρ u ∂T/∂x + ρ v ∂T/∂y = ∂/∂y (k ∂T/∂y) + μ (∂u/∂y)^2
        + u ∂p/∂x + 辐射/化学反应源项 (简化)

在相似坐标下，采用自相似解假设，能量方程简化为：

    d/dη [ (μ/Pr) dT/dη ] + (1/2) f dT/dη
        + (γ-1) Ma^2 μ (∂u/∂η)^2 = 0

壁面边界条件:
    T(0) = T_w   (等温壁)
    T(η→∞) = T_e

本模块采用迭代有限差分法求解自相似温度剖面，
并借鉴 heated_plate 的 Jacobi 松弛策略处理非线性耦合。
"""

import numpy as np
from utils import blasius_function, sutherland_viscosity, safe_divide, compressible_blasius_velocity


class HypersonicThermalSolver:
    """
    高超声速边界层热-速度耦合求解器。
    """

    def __init__(self, Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4,
                 Tw_over_Te=1.0, L=1.0, N_eta=200, eta_max=12.0):
        """
        参数:
            Ma (float): 自由流马赫数
            Re (float): 基于长度 L 的雷诺数
            Pr (float): 普朗特数
            gamma (float): 比热比
            Tw_over_Te (float): 壁温与边界层外缘温度比
            L (float): 特征长度 [m]
            N_eta (int): 相似变量 η 方向节点数
            eta_max (float): η 方向外边界
        """
        self.Ma = Ma
        self.Re = Re
        self.Pr = Pr
        self.gamma = gamma
        self.Tw_over_Te = Tw_over_Te
        self.L = L
        self.N_eta = N_eta
        self.eta_max = eta_max

        # 离散化
        self.eta = np.linspace(0.0, eta_max, N_eta)
        self.deta = self.eta[1] - self.eta[0]

    def solve_self_similar_energy(self, epsilon=1e-10, max_iter=50000):
        """
        求解自相似能量方程（扩展自 heated_plate 的 Jacobi 迭代）。

        能量方程离散形式（中心差分 + 源项）：
            a_j T_{j-1}^{new} + b_j T_j^{new} + c_j T_{j+1}^{new} = d_j

        其中:
            a_j = (μ_{j-1/2}/Pr) / Δη^2 - (1/4) f_j / Δη
            b_j = -2 (μ_{j}/Pr) / Δη^2
            c_j = (μ_{j+1/2}/Pr) / Δη^2 + (1/4) f_j / Δη
            d_j = - (γ-1) Ma^2 μ_j (u_j')^2

        壁面边界: T_0 = Tw,  远场: T_{N-1} = Te

        参数:
            epsilon (float): 收敛容差
            max_iter (int): 最大迭代次数

        返回:
            dict: 包含温度 T、速度 u、粘性 μ、密度 ρ 等剖面
        """
        n = self.N_eta
        deta = self.deta
        deta2 = deta ** 2

        # 初始化温度场
        Te = 1.0
        Tw = self.Tw_over_Te
        T_old = np.full(n, Tw + (Te - Tw) * self.eta / self.eta_max)
        T_new = T_old.copy()

        # Blasius 速度剖面（动量方程解耦，速度场固定）
        f, fp, fpp = blasius_function(self.eta)
        u = np.clip(fp, 0.0, 1.0)

        # 速度梯度 du/dη（中心差分）
        dup = np.zeros(n)
        dup[1:-1] = (u[2:] - u[:-2]) / (2.0 * deta)
        dup[0] = (u[1] - u[0]) / deta
        dup[-1] = (u[-1] - u[-2]) / deta

        # 粘性耗散源项系数
        diss_coeff = (self.gamma - 1.0) / 2.0 * self.Ma**2

        iterations = 0
        diff = epsilon + 1.0

        while diff >= epsilon and iterations < max_iter:
            T_old[:] = T_new[:]

            # 根据当前温度更新物性参数
            mu = sutherland_viscosity(T_old)
            # 密度（理想气体，p=const）
            rho = safe_divide(1.0, T_old, fill_value=1.0)

            # Jacobi 松弛更新内点温度
            # 借鉴 heated_plate 的迭代思想
            # 采用一阶迎风离散保证稳定性（f > 0，迎风在 j-1 方向）
            for j in range(1, n - 1):
                # 粘性在界面上的插值
                mu_m = 0.5 * (mu[j] + mu[j - 1])
                mu_p = 0.5 * (mu[j] + mu[j + 1])

                # 扩散系数
                D_m = mu_m / self.Pr
                D_p = mu_p / self.Pr

                # 对流项: 0.5 * f * dT/dη, 采用迎风 (f > 0 => 向后差分)
                # 方程: d/dη(D dT/dη) + 0.5*f*dT/dη + S = 0
                # 扩散离散: (D_p*(T_{j+1}-T_j) - D_m*(T_j-T_{j-1}))/deta^2
                # 对流离散: 0.5*f_j*(T_j - T_{j-1})/deta  (迎风)
                # => a_j T_{j-1} + b_j T_j + c_j T_{j+1} = -S_j
                # a_j = D_m/deta^2 + 0.5*f_j/deta
                # b_j = -(D_m+D_p)/deta^2 - 0.5*f_j/deta
                # c_j = D_p/deta^2
                a_j = D_m / deta2 + 0.5 * f[j] / deta
                b_j = -(D_m + D_p) / deta2 - 0.5 * f[j] / deta
                c_j = D_p / deta2

                # 粘性耗散源项: (γ-1) Ma^2 μ (du/dη)^2
                source = diss_coeff * 2.0 * mu[j] * dup[j]**2

                # Jacobi 迭代: T_j^{new} = (-a_j T_{j-1}^{old} - c_j T_{j+1}^{old} - source) / b_j
                denom = b_j
                if abs(denom) < 1e-15:
                    denom = -1e-15
                T_new[j] = (-a_j * T_old[j - 1] - c_j * T_old[j + 1] - source) / denom

            # 边界条件
            T_new[0] = Tw
            T_new[-1] = Te

            # 收敛判断
            diff = np.max(np.abs(T_new - T_old))
            iterations += 1

        # 最终物性
        mu_final = sutherland_viscosity(T_new)
        rho_final = safe_divide(1.0, T_new, fill_value=1.0)

        return {
            'eta': self.eta,
            'T': T_new,
            'u': u,
            'mu': mu_final,
            'rho': rho_final,
            'iterations': iterations,
            'diff': diff
        }

    def compute_wall_heat_flux(self, solution):
        """
        计算壁面热流。

        q_w = -k (∂T/∂y)|_{y=0} = -k_e / L * sqrt(Re_L) * (∂T/∂η)|_{η=0}

        采用一阶差分:
            (∂T/∂η)|_0 ≈ (T_1 - T_0) / Δη

        参数:
            solution (dict): solve_self_similar_energy 的返回值

        返回:
            float: 无量纲壁面热流 Stanton 数近似
        """
        T = solution['T']
        deta = self.deta
        dTdeta_w = (T[1] - T[0]) / deta

        # 斯坦顿数近似
        # St = q_w / (ρ_e u_e c_p (T_aw - T_w))
        # 这里返回无量纲热流梯度
        St_approx = -dTdeta_w / (self.Pr * np.sqrt(self.Re))
        return St_approx

    def compute_skin_friction(self, solution):
        """
        计算壁面摩擦系数。

        c_f = 2 τ_w / (ρ_e u_e^2) = 2 μ_w (∂u/∂y)|_0 / (ρ_e u_e^2)

        在相似坐标中:
            (∂u/∂y)|_0 = u_e / L * sqrt(Re_L) * (∂u/∂η)|_0

        参数:
            solution (dict): 求解结果

        返回:
            float: 壁面摩擦系数
        """
        u = solution['u']
        mu = solution['mu']
        deta = self.deta

        dudeta_w = (u[1] - u[0]) / deta
        cf = 2.0 * mu[0] * dudeta_w / np.sqrt(self.Re)
        return cf
