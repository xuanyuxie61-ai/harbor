# -*- coding: utf-8 -*-
"""
poisson_schrodinger.py — 自洽 Poisson-Schrödinger 求解器
============================================================
核心科学问题: 自洽求解耦合的薛定谔方程和泊松方程,
得到异质结的自洽能带结构和载流子分布.

融合种子项目:
  - 424_feynman_kac_3d: 自洽迭代思想
  - 757_mesh2d: 自适应网格在自洽循环中的应用

数学基础:
  薛定谔方程:
    Hψ_n = E_n ψ_n
    H = -(ℏ²/2) d/dz[1/m* d/dz] + V(z) + V_H(z) + V_xc(z)

  泊松方程:
    d/dz[ε(z) dφ/dz] = -ρ(z)/ε_0
    ρ(z) = e * (p - n + N_D^+ - N_A^-)

  载流子密度:
    n(z) = sum_n |ψ_n(z)|² * f(E_n, E_F, T) * g_2D*m*kB*T/ℏ²
    (对导带子带求和)

  自洽循环:
    1. 初始猜测 V(z)
    2. 求解薛定谔方程 → ψ_n, E_n
    3. 计算载流子密度 n(z)
    4. 求解泊松方程 → V_H(z)
    5. 更新 V = V_ext + V_H + V_xc
    6. 混合: V_new = α*V + (1-α)*V_old
    7. 检查收敛: ||V_new - V_old|| < tol
    8. 重复 2-7
"""

import numpy as np
from high_order_fd import HBAR, M0, E0, BenDanielDukeFD
from sparse_hetband import HeteroHamiltonianAssembler

# 物理常数
KB = 1.380649e-23       # J/K
KB_EV = KB / E0         # eV/K
EPSILON0 = 8.8541878128e-12  # F/m


class PoissonSchrodingerSolver:
    """
    自洽 Poisson-Schrödinger 求解器.
    """

    def __init__(self, z_grid, m_star, V_initial, epsilon_r,
                 T=300.0, doping_profile=None):
        """
        参数
        ----
        z_grid : ndarray
            空间网格 [m]
        m_star : ndarray
            有效质量 [m0]
        V_initial : ndarray
            初始势能 [eV]
        epsilon_r : ndarray or float
            相对介电常数
        T : float
            温度 [K]
        doping_profile : ndarray or None
            掺杂浓度 [1/m²] (面密度)
        """
        self.z = z_grid
        self.m_star = m_star
        self.V = V_initial.copy()
        self.N = len(z_grid)
        self.dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1e-10
        self.T = T
        self.kBT = KB_EV * T

        if isinstance(epsilon_r, (int, float)):
            self.epsilon_r = np.full(self.N, epsilon_r)
        else:
            self.epsilon_r = epsilon_r.copy()

        if doping_profile is None:
            self.doping = np.zeros(self.N)
        else:
            self.doping = doping_profile.copy()

        # 求解器参数
        self.max_iter = 50
        self.tolerance = 1e-5  # eV
        self.mixing_alpha = 0.3
        self.n_subbands = 5

        # 结果存储
        self.energies = []
        self.wavefunctions = []
        self.charge_density = None
        self.converged = False

    def solve_schrodinger(self, V):
        """
        求解薛定谔方程 (单步).

        返回子带能量和波函数.
        """
        assembler = HeteroHamiltonianAssembler(
            self.z, self.m_star, V, order=2)
        H_sparse = assembler.assemble()
        H_dense = H_sparse.to_dense()

        # 本征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(H_dense)

        # 取最低 n_subbands 个
        n_sub = min(self.n_subbands, len(eigenvalues))
        energies = eigenvalues[:n_sub]
        wavefunctions = eigenvectors[:, :n_sub]

        # 归一化波函数
        norm = np.sqrt(np.sum(wavefunctions**2, axis=0) * self.dz)
        norm = np.maximum(norm, 1e-30)
        wavefunctions /= norm

        return energies, wavefunctions

    def compute_charge_density(self, energies, wavefunctions, E_F):
        """
        计算载流子密度 (面密度/长度):
            n(z) = sum_n |ψ_n(z)|² * N_2D * f(E_n, E_F, T)

        其中:
            N_2D = m*kB*T/(πℏ²)  (2D 态密度)
            f = 1/(exp((E-E_F)/kBT) + 1)  (费米-狄拉克)
        """
        m_kg = np.mean(self.m_star) * M0
        N_2D = m_kg * KB * self.T / (np.pi * HBAR**2)  # [1/(J·m²)]
        # 转换为 [1/(eV·m²)]
        N_2D *= E0

        n_z = np.zeros(self.N)
        for n in range(len(energies)):
            # 费米-狄拉克分布
            arg = (energies[n] - E_F) / self.kBT
            arg = np.clip(arg, -50, 50)
            f_n = 1.0 / (np.exp(arg) + 1.0)

            # 子带载流子密度
            n_2d_n = N_2D * self.kBT * np.log(1 + np.exp(-arg))  # [1/m²]

            # 空间分布
            n_z += n_2d_n * wavefunctions[:, n]**2

        return n_z

    def solve_poisson(self, charge_density):
        """
        求解泊松方程:
            d/dz[ε(z) dφ/dz] = -e*n(z)/ε_0

        使用有限差分.
        """
        N = self.N
        dz = self.dz

        # 构建三对角系统
        # ε_{i+1/2} * (φ_{i+1}-φ_i)/dz² - ε_{i-1/2} * (φ_i-φ_{i-1})/dz²
        #   = -e*n_i/ε_0

        eps_half = np.zeros(N + 1)
        for i in range(N):
            eps_half[i] = self.epsilon_r[i]
        eps_half[N] = eps_half[N - 1]
        for i in range(1, N):
            eps_half[i] = 0.5 * (self.epsilon_r[i - 1] + self.epsilon_r[i])

        # 右端项
        rhs = np.zeros(N)
        for i in range(1, N - 1):
            rhs[i] = -E0 * charge_density[i] * dz**2 / (EPSILON0)

        # Thomas 算法求解三对角系统
        a = np.zeros(N)  # 下对角
        b = np.zeros(N)  # 对角
        c = np.zeros(N)  # 上对角

        for i in range(1, N - 1):
            a[i] = -eps_half[i]
            c[i] = -eps_half[i + 1]
            b[i] = eps_half[i] + eps_half[i + 1]

        # 边界条件: Dirichlet (φ=0)
        b[0] = 1.0
        b[-1] = 1.0

        phi = self._thomas_solver(a, b, c, rhs)
        return phi

    def _thomas_solver(self, a, b, c, d):
        """
        Thomas 算法 (追赶法) 求解三对角系统.
        """
        N = len(d)
        c_star = np.zeros(N)
        d_star = np.zeros(N)
        x = np.zeros(N)

        # 前向消元
        c_star[0] = c[0] / b[0] if b[0] != 0 else 0
        d_star[0] = d[0] / b[0] if b[0] != 0 else 0

        for i in range(1, N):
            denom = b[i] - a[i] * c_star[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30
            if i < N - 1:
                c_star[i] = c[i] / denom
            d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

        # 回代
        x[-1] = d_star[-1]
        for i in range(N - 2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def find_fermi_level(self, energies, target_density=1e16):
        """
        确定费米能级使得总载流子密度等于目标值.
        二分法求解.
        """
        E_min = np.min(energies) - 1.0
        E_max = np.max(energies) + 1.0

        for _ in range(100):
            E_F = 0.5 * (E_min + E_max)
            total_n = 0.0
            for E_n in energies:
                arg = (E_n - E_F) / self.kBT
                arg = np.clip(arg, -50, 50)
                total_n += self.kBT * np.log(1 + np.exp(-arg))

            if total_n * 1e-4 > target_density:  # 粗略比较
                E_max = E_F
            else:
                E_min = E_F

        return E_F

    def solve_self_consistent(self):
        """
        自洽求解主循环.
        """
        V_old = self.V.copy()
        E_F = np.min(self.V)  # 初始费米能级估计

        for iteration in range(self.max_iter):
            # 1. 求解薛定谔
            energies, wavefunctions = self.solve_schrodinger(V_old)

            # 2. 确定费米能级
            E_F = self.find_fermi_level(energies)

            # 3. 计算载流子密度
            n_z = self.compute_charge_density(energies, wavefunctions, E_F)
            self.charge_density = n_z

            # 4. 求解泊松
            phi = self.solve_poisson(n_z)

            # 5. 更新势能
            V_new = self.V.copy() + phi  # 外场 + 哈特里势

            # 6. 混合
            V_mixed = self.mixing_alpha * V_new + (1 - self.mixing_alpha) * V_old

            # 7. 检查收敛
            error = np.max(np.abs(V_mixed - V_old))
            V_old = V_mixed

            if error < self.tolerance:
                self.converged = True
                self.energies = energies
                self.wavefunctions = wavefunctions
                self.V = V_old
                return energies, wavefunctions

        # 未收敛, 返回最后结果
        self.energies = energies
        self.wavefunctions = wavefunctions
        self.V = V_old
        return energies, wavefunctions
