#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modal_analysis.py
水声传播抛物方程模型 — 波导简正波分析

本模块进行海洋波导的简正波（Normal Mode）分析，来源于：
- 899_polyomino_parity（Diophantine 方程与离散约束 → 模态数离散约束求解）
- 894_polynomial_conversion（Chebyshev/Legendre 谱元 → 本征问题谱离散）

核心物理与数学公式：
1. 简正波本征值方程（Sturm-Liouville 问题）：
   d²φ_n/dz² + [k²(z) − k_r,n²] φ_n = 0
   边界条件：
     φ_n(0) = 0                  （海面压力释放）
     dφ_n/dz + γ_b·φ_n = 0       （海底阻抗）
   其中 k_r,n 为第 n 号模态的水平波数（本征值），
   φ_n(z) 为对应的本征函数（模态深度函数）。

2. 模态频散关系：
   相速度：v_p,n = ω / Re(k_r,n)
   群速度：v_g,n = dω / d(Re(k_r,n)) ≈ [∫ φ_n² / c²(z) dz] / [k_r,n · ∫ φ_n² / c(z) dz]
   （采用一阶扰动近似）。

3. 截止频率与模态数估计（Diophantine 约束思想）：
   对于等声速波导（深度 H，声速 c），模态水平波数：
   k_r,n = √[k² − ((n+0.5)·π/H)²]
   传播模态需满足 k_r,n 为实数，即：
   n < (k·H/π) − 0.5 = N_max
   模态数 N_modes = floor(N_max) + 1。
   该离散约束可视为一个 Diophantine 不等式求解问题。

4. 模态展开解：
   p(r,z) = Σ_{n=0}^{N−1} A_n · φ_n(z_s) · φ_n(z) · H₀⁽¹⁾(k_r,n·r) / √r
   其中 A_n 为激励系数，与声源深度 z_s 和模态深度函数有关。

5. WKB 近似（高频极限）：
   对于渐变声速剖面，利用 WKB 近似估计模态相位积分：
   ∫_{z_{t1}}^{z_{t2}} √[k²(z) − k_r,n²] dz = (n + 0.5)·π
   其中 z_{t1}, z_{t2} 为反转深度（turning depths），满足 k(z_t) = k_r,n。

6. 谱离散本征问题（Chebyshev tau 方法）：
   将 φ(z) 在 Chebyshev 节点上展开，微分算子通过谱微分矩阵作用，
   边界条件通过 tau 行替换施加，求解广义特征值问题：
   A·φ = λ·B·φ
   其中 A 为离散化的 [d²/dz² + k²(z)]，B 为单位矩阵（标准特征值问题）。
"""

import numpy as np
from scipy import linalg as la
from utils import chebyshev_to_monomial_matrix, legendre_to_monomial_matrix


class NormalModeAnalyzer:
    """
    海洋波导简正波分析器。
    """

    def __init__(self, env, z_min=0.0, z_max=None, n_cheb=64):
        self.env = env
        self.z_min = z_min
        self.z_max = z_max if z_max is not None else env.depth_max
        self.n_cheb = n_cheb
        # Chebyshev 节点（Gauss-Lobatto，包含端点）
        j = np.arange(n_cheb + 1)
        self.xi = np.cos(np.pi * j / n_cheb)
        # 物理坐标
        self.z_nodes = 0.5 * (self.z_max - self.z_min) * self.xi + 0.5 * (self.z_max + self.z_min)
        # 微分矩阵
        self.D = self._chebyshev_differentiation_matrix()
        # 二阶导数
        self.D2 = self.D @ self.D

    def _chebyshev_differentiation_matrix(self):
        """Chebyshev Gauss-Lobatto 微分矩阵。"""
        N = self.n_cheb
        x = self.xi
        c = np.ones(N + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c *= ((-1.0) ** np.arange(N + 1))
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T + np.eye(N + 1)
        D = np.outer(c, 1.0 / c.T) / (dX + np.eye(N + 1))
        D = D - np.diag(np.sum(D, axis=1))
        return D

    def solve_eigenproblem(self, n_modes=None):
        """
        求解简正波本征值问题。
        方程：d²φ/dz² + [k²(z) − k_r²] φ = 0
        等价于 A·φ = λ·φ，其中 λ = −k_r²，A = d²/dz² + diag(k²(z))。
        返回: (k_r, phi, z_nodes)
        """
        J = 2.0 / (self.z_max - self.z_min)
        D2_phys = J ** 2 * self.D2
        kz = self.env.wavenumber(self.z_nodes)
        k2 = kz ** 2
        # === HOLE 3: Normal mode eigenproblem construction ===
        # TODO: Construct eigenproblem matrix A = D2_phys + diag(k^2(z))
        # TODO: Apply tau boundary conditions (Dirichlet at surface, Neumann at seabed)
        # TODO: Solve eigenproblem and filter propagating modes (kr^2 > 0)
        # TODO: Sort by kr^2 descending and slice to n_modes
        raise NotImplementedError("HOLE 3: Normal mode eigenproblem missing")
        kr = np.array([])
        eigvecs = np.zeros((len(self.z_nodes), 0), dtype=np.complex128)
        return kr, eigvecs, self.z_nodes

    def estimate_mode_count_wkb(self):
        """
        WKB 模态数估计：
        对于 Munk 声道，反转深度之间的相位积分决定模态数。
        简化估计：N_modes ≈ floor(2·H·f / c_min − 0.5)
        其中 H 为水深，c_min 为声道最小声速。
        """
        H = self.z_max
        c_min = self.env.c0  # Munk 轴处最小声速
        f = self.env.frequency
        k = 2.0 * np.pi * f / c_min
        N_max = k * H / np.pi - 0.5
        return int(max(np.floor(N_max), 0)) + 1

    def estimate_mode_count_diophantine(self):
        """
        将模态数估计视为 Diophantine 约束求解：
        求满足 n < (k·H/π) − 0.5 的最大非负整数 n。
        即求不等式的最大整数解。
        """
        return self.estimate_mode_count_wkb()

    def modal_phase_velocity(self, kr):
        """相速度：v_p = ω / Re(k_r)"""
        return self.env.omega / np.real(kr)

    def modal_group_velocity(self, phi, kr, dz=None):
        """
        群速度近似（一阶扰动公式）：
        v_g ≈ [∫ φ² / c² dz] / [k_r · ∫ φ² / c dz]
        """
        z = self.z_nodes
        c = self.env.sound_speed(z)
        phi2 = np.abs(phi) ** 2
        if dz is None:
            dz = np.diff(z)
            dz = np.concatenate([dz, [dz[-1]]])
        num = np.sum(phi2 / (c ** 2) * dz)
        den = np.real(kr) * np.sum(phi2 / c * dz)
        if abs(den) < 1e-20:
            return 0.0
        return float(num / den)

    def modal_excitation_coefficients(self, phi, z_s):
        """
        声源深度 z_s 处的模态激励系数：
        A_n = φ_n(z_s) / √(∫ |φ_n|² dz)
        """
        dz = np.diff(self.z_nodes)
        dz = np.concatenate([dz, [dz[-1]]])
        norms = np.sqrt(np.sum(np.abs(phi) ** 2 * dz, axis=0))
        norms = np.maximum(norms, 1e-20)
        # 插值到声源深度
        phi_at_zs = np.zeros(phi.shape[1], dtype=np.complex128)
        for n in range(phi.shape[1]):
            phi_at_zs[n] = np.interp(z_s, self.z_nodes, phi[:, n])
        return phi_at_zs / norms

    def propagate_modes(self, kr, phi, z_s, r_target):
        """
        简正波传播到距离 r_target：
        p(r,z) = Σ_n A_n·φ_n(z_s)·φ_n(z)·exp(i·k_r,n·r) / √(k_r,n·r)
        """
        A = self.modal_excitation_coefficients(phi, z_s)
        p = np.zeros(len(self.z_nodes), dtype=np.complex128)
        for n in range(len(kr)):
            if np.real(kr[n]) <= 0:
                continue
            phase = np.exp(1j * kr[n] * r_target)
            amp = 1.0 / np.sqrt(max(np.real(kr[n]) * r_target, 1e-6))
            p += A[n] * phi[:, n] * phase * amp
        return p

    def mode_dispersion_curve(self, phi, kr, frequencies):
        """
        计算各模态的频散曲线（群速度 vs 频率）。
        对每个频率重新求解本征问题并提取群速度。
        返回: list of (f, v_g) arrays for each mode。
        """
        original_freq = self.env.frequency
        curves = []
        for n in range(min(5, len(kr))):
            vg_list = []
            for f in frequencies:
                self.env.frequency = f
                self.env.omega = 2.0 * np.pi * f
                self.env.k0 = self.env.omega / self.env.c0
                kr_new, phi_new, _ = self.solve_eigenproblem(n_modes=n + 1)
                if len(kr_new) > n:
                    vg = self.modal_group_velocity(phi_new[:, n], kr_new[n])
                    vg_list.append((f, vg))
            self.env.frequency = original_freq
            self.env.omega = 2.0 * np.pi * original_freq
            self.env.k0 = self.env.omega / self.env.c0
            if vg_list:
                curves.append(np.array(vg_list))
        return curves


class ModalConstraintSolver:
    """
    将模态约束视为离散不等式求解（来自 899_polyomino_parity 的 Diophantine 思想）。
    """

    @staticmethod
    def solve_inequality_integer(a, b):
        """
        求解一维线性 Diophantine 不等式：a·n < b，求最大非负整数 n。
        等价于 n_max = floor((b−1)/a) 当 a>0 时。
        """
        a = float(a)
        b = float(b)
        if a <= 0:
            return []
        n_max = int(np.floor((b - 1e-12) / a))
        if n_max < 0:
            return []
        return list(range(n_max + 1))

    @staticmethod
    def backtrack_solutions(coeffs, target, max_n=100):
        """
        回溯求解多维非负整数线性组合：
        coeffs[0]·n_0 + coeffs[1]·n_1 + ... < target
        返回所有满足条件的 (n_0, n_1, ...) 组合。
        用于多声道耦合模态分析。
        """
        coeffs = np.asarray(coeffs, dtype=np.float64)
        solutions = []

        def backtrack(idx, current_sum, current_vec):
            if idx == len(coeffs):
                solutions.append(tuple(current_vec))
                return
            max_val = int((target - current_sum) / max(coeffs[idx], 1e-15))
            max_val = min(max_val, max_n)
            for v in range(max_val + 1):
                current_vec.append(v)
                backtrack(idx + 1, current_sum + v * coeffs[idx], current_vec)
                current_vec.pop()

        backtrack(0, 0.0, [])
        return solutions
