# -*- coding: utf-8 -*-
"""
ozone_depletion_solver.py
平流层臭氧空洞三维化学-输运耦合主求解器。

耦合方程组：

    \frac{\partial \mathbf{c}}{\partial t}
    = \mathbf{f}_{\mathrm{chem}}(\mathbf{c}; J_{\mathrm{O}_2}, J_{\mathrm{O}_3}, T)
    + \mathbf{S}_{\mathrm{trans}}(\mathbf{c})

其中化学源项由 StratosphericChemistry 提供，输运源项由 TransportOperator 提供。
时间离散采用算子分裂（Strang 分裂）：

    \mathbf{c}^{n+1}
    = \mathcal{T}_{\Delta t/2} \circ \mathcal{C}_{\Delta t} \circ
      \mathcal{T}_{\Delta t/2} \, \mathbf{c}^{n}

化学步使用半隐式 Rosenbrock 方法；
输运步使用隐式欧拉 + GMRES 求解。

融合来源：
  - 760_mgmres: 稀疏线性系统 GMRES 求解
  - 149_cg: 共轭梯度法
  - 572_ill_bvp: 边界条件处理
"""

import numpy as np
from chemistry_kinetics import StratosphericChemistry
from transport_operator import TransportOperator, temperature_profile_bvp
from photolysis_rates import PhotolysisRateCalculator
from linear_solvers import gmres_restart, conjugate_gradient
from utils import clip_positive, safe_divide

# 阿伏伽德罗常数与单位转换
NA = 6.02214076e23


class OzoneDepletionSolver:
    r"""
    平流层臭氧损耗三维耦合求解器。
    """

    def __init__(self, mesh, solar_zenith_deg=60.0):
        r"""
        Parameters
        ----------
        mesh : StratosphericMesh
        solar_zenith_deg : float
            太阳天顶角 [°]。
        """
        from stratospheric_mesh import StratosphericMesh
        self.mesh = mesh
        self.solar_zenith = solar_zenith_deg
        self.n_cells = mesh.n_cells
        self.photo = PhotolysisRateCalculator(degree=7)
        self.transport = TransportOperator(mesh)
        self._initialize_temperature()
        self._initialize_concentrations()
        self._compute_photolysis()

    def _initialize_temperature(self):
        r"""初始化各单元温度廓线。"""
        z_km_unique = np.linspace(self.mesh.alt_range[0], self.mesh.alt_range[1],
                                   self.mesh.n_alt)
        z_m = z_km_unique * 1000.0
        T_prof = temperature_profile_bvp(z_m)
        self.cell_temperature = np.zeros(self.n_cells)
        self.cell_altitude = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            z_km = self.mesh.cell_centroids[i, 2]
            self.cell_altitude[i] = z_km
            # 线性插值温度
            idx = np.searchsorted(z_km_unique, z_km)
            idx = np.clip(idx, 1, len(z_km_unique) - 1)
            z0, z1 = z_km_unique[idx - 1], z_km_unique[idx]
            T0, T1 = T_prof[idx - 1], T_prof[idx]
            if abs(z1 - z0) > 1e-10:
                frac = (z_km - z0) / (z1 - z0)
            else:
                frac = 0.0
            self.cell_temperature[i] = T0 + frac * (T1 - T0)

    def _initialize_concentrations(self):
        r"""
        初始化各物种浓度 [molecules cm^{-3}]。

        参考值（基于标准大气）：
          - O2: 0.21 * N(z)
          - N2: 0.78 * N(z)
          - M = O2 + N2
          - O3: 从标准臭氧廓线插值
          - 催化物种：ppbv 量级
        """
        nsp = StratosphericChemistry.N_SPECIES
        self.c_cell = np.zeros((self.n_cells, nsp))
        for i in range(self.n_cells):
            z_km = self.cell_altitude[i]
            T = self.cell_temperature[i]
            # 数密度 from ideal gas law: n = P/(k_B T)
            # 简化：使用标高近似
            n_dens = self._number_density(z_km, T)
            self.c_cell[i, StratosphericChemistry.IDX_O2] = 0.21 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_N2] = 0.78 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_M] = n_dens
            # O3 廓线（简化的Chapman分布）
            n_o3 = 5.0e12 * np.exp(-0.5 * ((z_km - 25.0) / 5.0) ** 2)
            self.c_cell[i, StratosphericChemistry.IDX_O3] = n_o3
            # 催化物种（ppbv级别）
            ppbv = 1e-9 * n_dens
            self.c_cell[i, StratosphericChemistry.IDX_NO] = 1.0 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_NO2] = 0.5 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_Cl] = 0.05 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_ClO] = 0.03 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_OH] = 0.01 * ppbv
            self.c_cell[i, StratosphericChemistry.IDX_HO2] = 0.02 * ppbv
            # O 和 O(1D) 初始化为化学平衡近似
            self.c_cell[i, StratosphericChemistry.IDX_O] = 1e6
            self.c_cell[i, StratosphericChemistry.IDX_O1D] = 1e2

    def _number_density(self, z_km, T):
        r"""
        计算高度 z 处的空气数密度 [molecules cm^{-3}]。

            n(z) = n_0 \exp\!\left(-\frac{z}{H}\right)

        其中 H = k_B T / (m g) 为标高，约 7 km。

        Parameters
        ----------
        z_km : float
        T : float

        Returns
        -------
        n : float
        """
        n0 = 2.55e19  # 海平面数密度 [cm^{-3}]
        H = 7.0  # km
        return n0 * np.exp(-z_km / H) * (288.0 / T)

    def _compute_photolysis(self):
        r"""
        计算每个单元的光解速率。
        """
        self.J_o2_cell = np.zeros(self.n_cells)
        self.J_o3_cell = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            z_km = self.cell_altitude[i]
            T = self.cell_temperature[i]
            # 计算 O3 和 O2 柱浓度（简化：从当前高度向上积分）
            col_o3 = self._column_density(i, StratosphericChemistry.IDX_O3)
            col_o2 = self._column_density(i, StratosphericChemistry.IDX_O2)
            self.J_o2_cell[i] = self.photo.photolysis_rate_o2(
                z_km, self.solar_zenith, col_o3, col_o2
            )
            self.J_o3_cell[i] = self.photo.photolysis_rate_o3(
                z_km, self.solar_zenith, col_o3, col_o2, T
            )

    def _column_density(self, cell_idx, species_idx):
        r"""
        计算某单元上方物种的柱浓度 [molecules cm^{-2}]。

            N = \int_{z}^{z_{\mathrm{top}}} n_{\mathrm{sp}}(z') \, dz'

        这里使用同高度层所有单元的浓度平均近似。
        """
        z_km = self.cell_altitude[cell_idx]
        mask = self.cell_altitude >= z_km
        if not np.any(mask):
            return 0.0
        # 简化的垂直积分
        conc = self.c_cell[mask, species_idx]
        dz = (self.mesh.alt_range[1] - self.mesh.alt_range[0]) / (self.mesh.n_alt - 1)
        return float(np.mean(conc) * dz * 1e5)  # cm 转换

    def _chemistry_step(self, c_in, dt, cell_idx):
        r"""
        对单个单元执行化学步（Rosenbrock）。

        Parameters
        ----------
        c_in : ndarray, shape (N_SPECIES,)
        dt : float
        cell_idx : int

        Returns
        -------
        c_out : ndarray
        """
        # TODO: 创建 StratosphericChemistry 实例，设置光解速率，调用 step_rosenbrock
        raise NotImplementedError("Hole 2: 请实现单个单元的化学步调用逻辑")

    def _transport_step_implicit(self, c_in, dt):
        r"""
        执行输运步（简化隐式处理）。

        求解 (I - dt * T) c^{n+1} = c^n，其中 T 为输运算子。
        使用 GMRES 迭代求解。

        Parameters
        ----------
        c_in : ndarray, shape (n_cells, N_SPECIES)
        dt : float

        Returns
        -------
        c_out : ndarray
        """
        nsp = StratosphericChemistry.N_SPECIES
        x0 = c_in.ravel().copy()

        def Ax(x):
            x_reshaped = x.reshape((self.n_cells, nsp))
            S = self.transport.transport_source(x_reshaped)
            return (x_reshaped - dt * S).ravel()

        b = c_in.ravel()
        x, res, it = gmres_restart(Ax, b, x0=x0, max_iter=20, restart=15,
                                    tol_abs=1e-8, tol_rel=1e-6)
        c_out = x.reshape((self.n_cells, nsp))
        c_out = np.maximum(c_out, 1e-30)
        return c_out

    def step(self, dt):
        r"""
        单步算子分裂（Strang 分裂）：

            c^{n+1/2} = \mathcal{C}_{dt/2}(c^n)
            c^{*}    = \mathcal{T}_{dt}(c^{n+1/2})
            c^{n+1}  = \mathcal{C}_{dt/2}(c^{*})

        Parameters
        ----------
        dt : float
            时间步长 [s]。

        Returns
        -------
        c_new : ndarray
        """
        # 半步化学
        c_half = np.zeros_like(self.c_cell)
        for i in range(self.n_cells):
            c_half[i] = self._chemistry_step(self.c_cell[i], dt * 0.5, i)

        # 整步输运
        c_star = self._transport_step_implicit(c_half, dt)

        # 半步化学
        c_new = np.zeros_like(self.c_cell)
        for i in range(self.n_cells):
            c_new[i] = self._chemistry_step(c_star[i], dt * 0.5, i)

        self.c_cell = c_new
        return c_new

    def integrate(self, t_total_hours=24.0, dt_max_minutes=10.0):
        r"""
        时间积分。

        Parameters
        ----------
        t_total_hours : float
            总积分时间 [h]。
        dt_max_minutes : float
            最大时间步长 [min]。

        Returns
        -------
        history : dict
            包含时间序列、臭氧柱总量、各物种分布等。
        """
        t_total = t_total_hours * 3600.0
        dt_max = dt_max_minutes * 60.0
        t = 0.0
        history = {
            'time_hours': [0.0],
            'total_ozone_du': [self._total_ozone_dobson()],
            'o3_min': [np.min(self.c_cell[:, StratosphericChemistry.IDX_O3])],
            'o3_max': [np.max(self.c_cell[:, StratosphericChemistry.IDX_O3])],
        }

        while t < t_total:
            # 自适应步长
            L_max = 0.0
            for i in range(self.n_cells):
                chem = StratosphericChemistry(T_k=self.cell_temperature[i],
                                               M_cm3=self.c_cell[i, StratosphericChemistry.IDX_M])
                chem.set_photolysis_rates(self.J_o2_cell[i], self.J_o3_cell[i])
                _, L = chem.production_loss(self.c_cell[i])
                L_max = max(L_max, np.max(L))
            dt = min(dt_max, safe_divide(0.05, L_max, 60.0))
            if t + dt > t_total:
                dt = t_total - t

            self.step(dt)
            t += dt

            if len(history['time_hours']) == 0 or t / 3600.0 - history['time_hours'][-1] >= 0.5:
                history['time_hours'].append(t / 3600.0)
                history['total_ozone_du'].append(self._total_ozone_dobson())
                history['o3_min'].append(np.min(self.c_cell[:, StratosphericChemistry.IDX_O3]))
                history['o3_max'].append(np.max(self.c_cell[:, StratosphericChemistry.IDX_O3]))

        # 确保最后一时刻被记录
        if abs(t / 3600.0 - history['time_hours'][-1]) > 1e-6:
            history['time_hours'].append(t / 3600.0)
            history['total_ozone_du'].append(self._total_ozone_dobson())
            history['o3_min'].append(np.min(self.c_cell[:, StratosphericChemistry.IDX_O3]))
            history['o3_max'].append(np.max(self.c_cell[:, StratosphericChemistry.IDX_O3]))

        return history

    def _total_ozone_dobson(self):
        r"""
        计算总臭氧柱浓度 [Dobson Unit]。

            1 DU = 2.687e16 molecules cm^{-2}

            \Omega = \int_0^{z_{\mathrm{top}}} [\mathrm{O}_3](z) \, dz

        Returns
        -------
        du : float
        """
        # 体积加权平均
        o3 = self.c_cell[:, StratosphericChemistry.IDX_O3]
        vol = self.mesh.cell_volumes
        total_o3 = np.sum(o3 * vol) / np.sum(vol)
        # 近似柱浓度（简化为平均浓度 × 平流层厚度）
        thickness_cm = (self.mesh.alt_range[1] - self.mesh.alt_range[0]) * 1e5
        col = total_o3 * thickness_cm
        du = col / 2.687e16
        return float(du)

    def get_ozone_distribution(self):
        r"""
        返回当前臭氧空间分布。

        Returns
        -------
        o3 : ndarray, shape (n_cells,)
        """
        return self.c_cell[:, StratosphericChemistry.IDX_O3].copy()

    def perturb_catalytic_species(self, factor_no=1.0, factor_cl=1.0):
        r"""
        扰动催化物种初始浓度以模拟人类活动影响。

        Parameters
        ----------
        factor_no : float
            NOx 放大因子。
        factor_cl : float
            ClOx 放大因子。
        """
        self.c_cell[:, StratosphericChemistry.IDX_NO] *= factor_no
        self.c_cell[:, StratosphericChemistry.IDX_NO2] *= factor_no
        self.c_cell[:, StratosphericChemistry.IDX_Cl] *= factor_cl
        self.c_cell[:, StratosphericChemistry.IDX_ClO] *= factor_cl
