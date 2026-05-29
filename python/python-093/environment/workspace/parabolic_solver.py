#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parabolic_solver.py
水声传播抛物方程模型 — 宽角抛物方程（WAPE）求解核心

本模块实现宽角抛物方程的分裂步傅里叶（SSF）与有限差分（FD）混合求解器，
来源于：
- 1023_rigid_body_ode（ODE 步进架构、守恒量监测）
- 894_polynomial_conversion（Chebyshev/Legendre 谱元离散）
- 1307_triangle_integrals（三角形 FEM 积分用于变分离散）
- 942_quad_parfor（并行梯形积分用于能量通量计算）

核心物理模型：
1. 标准抛物方程（Standard PE，Tappert 1977）：
   从 Helmholtz 方程出发，设 p(r,z) = u(r,z)·H₀⁽¹⁾(k₀r)，
   在远场近似下（∂²u/∂r² ≪ 2ik₀·∂u/∂r），得到：
   2i·k₀·∂u/∂r = ∂²u/∂z² + k₀²·(n²(z) − 1)·u

2. 宽角抛物方程（Wide-Angle PE，Claerbout 1976）：
   使用 Padé(1,1) 近似展开伪微分算子：
   ∂/∂r = i·k₀·(√(1 + X) − 1) ≈ i·k₀·X / (2 + X)
   其中 X = (1/k₀²)·∂²/∂z² + (n² − 1)。
   对应的隐式方程为：
   (1 + a·X)·∂u/∂r = i·k₀·b·X·u
   取 a = 1/4, b = 1/2 对应 Claerbout 近似。

3. 分裂步傅里叶（SSF）算法：
   每一步 Δr 分为折射步和衍射步：
   u*(r+Δr/2, z) = exp[i·Δr·k₀·(n²(z)−1)/2] · u(r, z)
   ũ*(r+Δr/2, k_z) = FFT[u*]
   ũ(r+Δr, k_z) = exp[i·Δr·k_z²/(2k₀)] · ũ*
   u(r+Δr, z) = IFFT[ũ]
   最后再进行一次折射步。

4. 有限差分隐式格式（Crank-Nicolson）：
   [I − (Δr/4ik₀)·D² − (Δr/4ik₀)·k₀²·diag(n²−1)] u^{m+1}
   = [I + (Δr/4ik₀)·D² + (Δr/4ik₀)·k₀²·diag(n²−1)] u^{m}
   其中 D² 为二阶差分矩阵（包含边界条件）。

5. 能量守恒监测（来自 rigid_body_ode 思想）：
   计算每一步的垂直能量通量：
   E(r) = ∫₀^{h_b(r)} |u(r,z)|² dz
   理论上当吸收为零时应守恒；衰减量用于验证数值稳定性。

6. 谱元离散（Chebyshev tau 方法）：
   在深度方向采用 Chebyshev 多项式展开：
   u(z) ≈ Σ_{k=0}^{N} û_k · T_k(ζ)，ζ = 2z/z_max − 1
   微分算子通过 Chebyshev-to-monomial 矩阵映射实现。
"""

import numpy as np
from scipy import linalg as la
from scipy.fft import fft, ifft, fftfreq
from boundary_conditions import solve_tridiag, BoundaryConditionHandler
from utils import chebyshev_to_monomial_matrix, legendre_to_monomial_matrix


class ParabolicSolver:
    """
    宽角抛物方程求解器，支持 SSF 与 CN-FD 两种模式。
    """

    def __init__(self, env, mesh, bc_handler, method='cn_fd'):
        self.env = env
        self.mesh = mesh
        self.bc = bc_handler
        self.method = method
        self.k0 = env.k0
        self.energy_history = []
        self.range_history = []

    def _build_laplacian_fd(self, z, nz):
        """
        构建深度方向二阶导数差分矩阵（三对角形式）。
        使用变网格间距：
        D²u_j ≈ 2/(Δz_j + Δz_{j-1}) · [(u_{j+1}−u_j)/Δz_j − (u_j−u_{j-1})/Δz_{j-1}]
        """
        dz = np.diff(z)
        dz = np.concatenate([dz, [dz[-1]]])
        a = np.zeros(nz, dtype=np.complex128)
        b = np.zeros(nz, dtype=np.complex128)
        c = np.zeros(nz, dtype=np.complex128)
        for j in range(1, nz - 1):
            dz_j = dz[j]
            dz_jm1 = dz[j - 1]
            denom = dz_j + dz_jm1
            if denom < 1e-12:
                denom = 1e-12
            a[j] = 2.0 / (denom * dz_jm1)
            c[j] = 2.0 / (denom * dz_j)
            b[j] = -a[j] - c[j]
        # 边界：Dirichlet 在 j=0，Robin 在 j=nz-1（由 bc_handler 后续修改）
        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
        return a, b, c

    def _crank_nicolson_step(self, u, z, dr, m):
        """
        Crank-Nicolson 单步推进。
        [I − (Δr/4ik₀)·L] u^{m+1} = [I + (Δr/4ik₀)·L] u^{m}
        其中 L = D² + k₀²·diag(n²−1)。
        """
        nz = len(z)
        a, b, c = self._build_laplacian_fd(z, nz)
        n2_dev = self.env.refractive_index_squared_deviation(z)
        # PML 修正
        k_pml = self.bc.pml_modified_wavenumber(z)
        # 将 n2_dev 修正为 PML 影响
        n2_eff = (k_pml / self.k0) ** 2 - 1.0

        # === HOLE 2: Crank-Nicolson discretization ===
        # TODO: Implement CN coefficient and left/right tridiagonal matrices
        # coeff = dr / (4.0 * 1j * self.k0)
        # aL = -coeff * a.copy()
        # bL = 1.0 - coeff * (b + self.k0 ** 2 * n2_eff)
        # cL = -coeff * c.copy()
        # aR = coeff * a.copy()
        # bR = 1.0 + coeff * (b + self.k0 ** 2 * n2_eff)
        # cR = coeff * c.copy()
        raise NotImplementedError("HOLE 2: CN discretization missing")

        # 应用边界条件
        aL, bL, cL = self.bc.apply_seabed_bc_tridiagonal(aL, bL, cL, u, m)
        aR, bR, cR = self.bc.apply_seabed_bc_tridiagonal(aR, bR, cR, u, m)

        # 海面 Dirichlet
        bL[0] = 1.0
        cL[0] = 0.0
        bR[0] = 1.0
        cR[0] = 0.0
        u[0] = 0.0

        # 右端项
        d = np.zeros(nz, dtype=np.complex128)
        d[0] = 0.0
        for j in range(1, nz - 1):
            d[j] = aR[j] * u[j - 1] + bR[j] * u[j] + cR[j] * u[j + 1]
        d[-1] = bR[-1] * u[-1]
        if nz > 1:
            d[-1] += aR[-1] * u[-2]

        u_new = solve_tridiag(aL, bL, cL, d)
        # 再次海面约束
        u_new[0] = 0.0
        return u_new

    def _ssf_step(self, u, z, dr, m):
        """
        分裂步傅里叶单步推进。
        要求 z 为均匀网格。
        """
        nz = len(z)
        dz = z[1] - z[0]
        # 折射半步
        n2_dev = self.env.refractive_index_squared_deviation(z)
        u_half = u * np.exp(1j * dr * self.k0 * n2_dev / 2.0)
        # 衍射步（FFT）
        kz = 2.0 * np.pi * fftfreq(nz, dz)
        u_fft = fft(u_half)
        u_fft *= np.exp(1j * dr * kz ** 2 / (2.0 * self.k0))
        u_new = ifft(u_fft)
        # 再折射半步
        u_new *= np.exp(1j * dr * self.k0 * n2_dev / 2.0)
        # 海面约束
        u_new[0] = 0.0
        # 海底掩码
        u_new = self.bc.mask_field_by_bathymetry(u_new, m)
        return u_new

    def compute_energy_flux(self, u, z, m):
        """
        计算垂直能量通量（来自 quad_parfor 的并行积分思想）：
        E(r_m) = ∫₀^{h_b(r_m)} |u(z)|² dz
        使用梯形法则。
        """
        mask = self.mesh.node_mask[m, :]
        z_valid = z[mask]
        u_valid = u[mask]
        if len(z_valid) < 2:
            return 0.0
        intensity = np.abs(u_valid) ** 2
        return np.trapezoid(intensity, z_valid)

    def solve(self, u0, z_s, progress_interval=None):
        """
        主求解循环：从 r=0 步进到 r=r_max。
        参数:
            u0: 初始场 (nz,)
            z_s: 声源深度
            progress_interval: 每多少步打印一次能量信息
        返回:
            U: (nr, nz) 复数场矩阵
        """
        nr = self.mesh.nr
        nz = self.mesh.nz
        U = np.zeros((nr, nz), dtype=np.complex128)
        U[0, :] = u0.copy()
        u = u0.copy()
        z = self.mesh.z_grid

        # 记录初始能量
        e0 = self.compute_energy_flux(u, z, 0)
        self.energy_history.append(e0)
        self.range_history.append(0.0)

        for m in range(1, nr):
            dr = self.mesh.dr
            if self.method == 'cn_fd':
                u = self._crank_nicolson_step(u, z, dr, m)
            elif self.method == 'ssf':
                # SSF 要求均匀网格，检查是否可用
                dzs = np.diff(z)
                if np.max(np.abs(dzs - dzs[0])) < 1e-6:
                    u = self._ssf_step(u, z, dr, m)
                else:
                    u = self._crank_nicolson_step(u, z, dr, m)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # 应用掩码
            u = self.bc.mask_field_by_bathymetry(u, m)
            U[m, :] = u.copy()

            # 能量监测
            e = self.compute_energy_flux(u, z, m)
            self.energy_history.append(e)
            self.range_history.append(self.mesh.r_grid[m])

            if progress_interval and m % progress_interval == 0:
                rel_err = abs(e - e0) / max(abs(e0), 1e-15)
                print(f"  Range {self.mesh.r_grid[m]/1000:.1f} km: "
                      f"energy flux = {e:.6e}, rel. change = {rel_err:.6e}")

        return U

    def energy_conservation_error(self):
        """
        计算全局能量守恒误差指标：
        ε_E = max_m |E(r_m) − E(0)| / E(0)
        """
        if len(self.energy_history) < 2:
            return 0.0
        e0 = self.energy_history[0]
        if abs(e0) < 1e-20:
            return 0.0
        return max(abs(e - e0) for e in self.energy_history) / abs(e0)


class SpectralElementSolver:
    """
    Chebyshev 谱元离散求解器（一维深度方向）。
    将 u(z) 在 Chebyshev 节点上展开，利用 Chebyshev-to-monomial 矩阵
    构造微分算子，用于 WAPE 的谱精度求解。
    """

    def __init__(self, env, n_cheb=32):
        self.env = env
        self.n = n_cheb
        # Chebyshev 节点（第一类，在 [-1,1] 上）
        self.xi = np.cos(np.pi * (np.arange(n_cheb + 1) + 0.5) / (n_cheb + 1))
        # 转换矩阵 T_k(xi_j)
        self.T = np.zeros((n_cheb + 1, n_cheb + 1))
        for k in range(n_cheb + 1):
            self.T[:, k] = np.cos(k * np.arccos(self.xi))
        # 单项式系数矩阵
        self.M_cheb_mono = chebyshev_to_monomial_matrix(n_cheb)
        # 微分矩阵（在 Chebyshev 节点上）
        self.D = self._chebyshev_differentiation_matrix()

    def _chebyshev_differentiation_matrix(self):
        """
        构建 Chebyshev 节点上的微分矩阵 D，满足 u'(xi) ≈ D @ u(xi)。
        使用重心插值公式（Barycentric）。
        """
        n = self.n
        x = self.xi
        c = np.ones(n + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c *= ((-1.0) ** np.arange(n + 1))
        X = np.tile(x, (n + 1, 1))
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
        D -= np.diag(np.sum(D, axis=1))
        return D

    def map_to_physical(self, z_min, z_max):
        """将参考坐标 xi ∈ [-1,1] 映射到物理坐标 z ∈ [z_min, z_max]。"""
        return 0.5 * (z_max - z_min) * self.xi + 0.5 * (z_max + z_min)

    def solve_eigenproblem(self, z_min, z_max):
        """
        求解简正波本征值问题（简化模型）：
        [d²/dz² + k₀²·n²(z)] φ = λ φ
        在 Chebyshev 节点上离散，用于模态分析。
        返回特征值和特征向量。
        """
        z = self.map_to_physical(z_min, z_max)
        J = 2.0 / (z_max - z_min)  # 坐标缩放因子
        # 二阶导数矩阵（包含坐标变换）
        D2 = J ** 2 * self.D @ self.D
        # 折射率平方矩阵
        n2 = self.env.refractive_index(z) ** 2
        A = D2 + np.diag(self.env.k0 ** 2 * n2)
        # 边界条件：Dirichlet 在两端（简化）
        A[0, :] = 0.0
        A[0, 0] = 1.0
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        # 求解
        eigvals, eigvecs = la.eig(A)
        # 排序
        idx = np.argsort(np.real(eigvals))
        return eigvals[idx], eigvecs[:, idx]
