"""
potential_field.py
离子通道内静电势场计算

基于种子项目 612_julia_set 的核心思想：
- Julia 集的非线性复迭代 z -> z^2 + c 用于描述非线性介电响应
- 迭代收敛/发散判定映射到介电函数的自洽求解

结合 Poisson-Boltzmann 方程求解通道内的电势分布：
    ∇·[ε(r)∇φ(r)] = -ρ_ion(r) - ρ_fix(r)

其中介电常数 ε(r) 在蛋白区域（~2-4）和水区域（~78）之间非线性过渡，
采用类似 Julia 集边界的平滑过渡函数：
    ε(r) = ε_protein + (ε_water - ε_protein) * σ(|r - r_boundary|)
"""

import numpy as np
from finite_difference import apply_laplacian_3d


class DielectricProfile:
    """
    空间变化的介电函数，模拟蛋白-水界面的非线性过渡。
    """
    def __init__(self, shape, dx, dy, dz, eps_water=78.5, eps_protein=4.0,
                 transition_width=0.05):
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.eps_water = eps_water
        self.eps_protein = eps_protein
        self.transition_width = transition_width
        self.eps = self._build_profile()

    def _smooth_step(self, x):
        """
        平滑阶跃函数（类似 Julia 集边界的平滑版本）：
            σ(x) = 1 / (1 + exp(-x/δ))
        """
        return 1.0 / (1.0 + np.exp(-x / self.transition_width))

    def _build_profile(self):
        """
        构建简化的 KcsA 通道介电剖面。
        假设通道中心沿 z 轴，孔道内部介电常数较低。
        """
        Nx, Ny, Nz = self.shape
        x = np.linspace(-0.6, 0.6, Nx)
        y = np.linspace(-0.6, 0.6, Ny)
        z = np.linspace(0.0, 4.5, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 径向距离
        r = np.sqrt(X ** 2 + Y ** 2)

        # 孔道半径剖面（简化 KcsA）
        # 滤器区域：z ∈ [1.5, 2.7], r_c = 0.15 nm
        # 腔体：z ∈ [0.5, 1.5], r_c = 0.5 nm
        # 门：z < 0.5, r_c = 0.2 nm
        r_channel = np.zeros_like(Z)
        mask_filter = (Z >= 1.5) & (Z <= 2.7)
        mask_cavity = (Z >= 0.5) & (Z < 1.5)
        mask_gate = Z < 0.5

        r_channel[mask_filter] = 0.15
        r_channel[mask_cavity] = 0.5
        r_channel[mask_gate] = 0.2 + 0.4 * Z[mask_gate]  # 线性扩张

        # 孔道外部 r_c 增大
        r_channel[~mask_filter & ~mask_cavity & ~mask_gate] = 0.6

        # 距离孔道中心的归一化距离
        dist = (r - r_channel)
        sigma = self._smooth_step(dist)

        eps = self.eps_protein + (self.eps_water - self.eps_protein) * sigma
        return eps

    def gradient_eps(self):
        """
        数值计算介电常数的梯度（用于 Nernst-Planck 中的修正项）。
        """
        grad = np.gradient(self.eps, self.dx, self.dy, self.dz)
        return grad


class PotentialSolver:
    """
    求解非线性 Poisson-Boltzmann 方程。

    方程形式：
        ∇·[ε(r)∇φ] = -4πρ

    采用自洽迭代（类似 Julia 集的定点迭代）：
        φ^{k+1} = φ^k + ω * (∇^2)^{-1} [ -ρ/ε - ∇lnε·∇φ^k ]

    其中 ω 为松弛因子。
    """
    def __init__(self, dielectric, max_iter=200, tol=1e-8, omega=1.2):
        self.dielectric = dielectric
        self.max_iter = max_iter
        self.tol = tol
        self.omega = omega
        self.shape = dielectric.shape

    def _fixed_charge_density(self):
        """
        固定电荷密度（来自蛋白骨架的带电残基）。
        KcsA 选择性滤器包含保守的 TVGYG 序列，每个亚基有 4 个羰基氧
        作为 K+ 的配位点，整体带 -4e（四聚体）。
        """
        Nx, Ny, Nz = self.shape
        rho_fix = np.zeros(self.shape)
        # 在滤器区域放置固定负电荷
        z_index = np.linspace(0.0, 4.5, Nz)
        filter_z_mask = (z_index >= 1.5) & (z_index <= 2.7)
        # 中心轴附近
        cx, cy = Nx // 2, Ny // 2
        sigma_r = 1.5  # 格点数
        for iz in np.where(filter_z_mask)[0]:
            for ix in range(Nx):
                for iy in range(Ny):
                    dr2 = ((ix - cx) ** 2 + (iy - cy) ** 2) * (self.dielectric.dx ** 2)
                    rho_fix[ix, iy, iz] += -1.0e25 * np.exp(-dr2 / (2.0 * sigma_r ** 2 * self.dielectric.dx ** 2))
        return rho_fix

    def _mobile_charge_density(self, conc_k, conc_na, phi, T=300.0):
        """
        移动离子电荷密度（Boltzmann 分布，线性化近似）：
            ρ_ion ≈ e Σ z_i c_i^bulk (1 - z_i e φ / k_B T)
        线性化处理避免 φ 较大时指数爆炸导致 NaN。
        """
        kB = 1.380649e-23
        e_charge = 1.602176634e-19
        factor = e_charge / (kB * T)
        # 线性化 Debye-Hückel 近似
        rho = e_charge * (conc_k * (1.0 - factor * phi) +
                          conc_na * (1.0 - factor * phi))
        return rho

    def solve(self, conc_k_bulk=150.0, conc_na_bulk=150.0,
              boundary_potential=None):
        """
        自洽求解电势场。

        Parameters
        ----------
        conc_k_bulk : float
            体相 K+ 浓度 (mol/m^3)
        conc_na_bulk : float
            体相 Na+ 浓度 (mol/m^3)
        boundary_potential : float, optional
            边界电势 (V)

        Returns
        -------
        phi : ndarray
            电势分布 (V)
        """
        Nx, Ny, Nz = self.shape
        phi = np.zeros(self.shape)
        if boundary_potential is not None:
            phi[:, :, 0] = boundary_potential
            phi[:, :, -1] = boundary_potential
        else:
            phi[:, :, 0] = 0.0
            phi[:, :, -1] = 0.0

        rho_fix = self._fixed_charge_density()
        eps = self.dielectric.eps
        grad_eps = self.dielectric.gradient_eps()

        dx, dy, dz = self.dielectric.dx, self.dielectric.dy, self.dielectric.dz

        for iteration in range(self.max_iter):
            # TODO: Hole 2 — 实现 Poisson-Boltzmann 方程的自洽迭代核心
            # 科学背景：求解 ∇·[ε(r)∇φ] = -ρ_total，其中 ρ_total = ρ_fix + ρ_mobile
            # 提示步骤：
            #   1. 调用 _mobile_charge_density 计算移动电荷密度
            #   2. 调用 apply_laplacian_3d 计算 ∇²φ
            #   3. 数值计算 ∇ε·∇φ 作为修正项
            #   4. 残差 residual = lap_phi + corr + rho_total / eps
            #   5. Jacobi 更新 phi，注意稳定化 clip 和边界条件
            #   6. 收敛判定 err < tol
            raise NotImplementedError("Hole 2: 请实现 Poisson-Boltzmann 自洽迭代核心")

        return phi
