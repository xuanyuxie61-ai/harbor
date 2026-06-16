"""
hetero_potential.py — 异质结势能剖面构建
==========================================
核心科学问题: 构建异质结界面处自洽势能 V(z), 包含:
  - 带偏移 (Anderson 规则)
  - 应变修正 (形变势理论)
  - 极化电荷 (自发极化 + 压电极化)
  - 外电场修正 (Stark 效应)

融合种子项目:
  - 539_histogram_discrete: 组分起伏的统计建模
  - 1059_Glyphosate_crystallization: 分子层拓扑排列思想
"""

import numpy as np
from material_parameters import (
    Material2D, anderson_band_offset, strain_energy_shift,
    lattice_mismatch, HBAR, M0, E0, EPSILON0, KB_EV
)


class HeteroPotential:
    """
    异质结一维势能剖面.

    物理模型
    --------
    沿生长方向 z 的势能由以下贡献构成:
        V(z) = V_band_offset(z) + V_strain(z) + V_polarization(z)
             + V_external(z) + V_image_charge(z)

    带偏移采用 Anderson 规则 + 实验修正因子.
    应变修正采用连续介质弹性理论.
    """

    def __init__(self, mat_left, mat_right, z_array,
                 layer_thickness=None, T=300.0,
                 external_field=0.0, alloy_fraction=None):
        """
        参数
        ----
        mat_left, mat_right : Material2D
            左右两侧材料
        z_array : ndarray
            空间网格坐标 [m]
        layer_thickness : float or None
            界面过渡层厚度 [m], 默认自动计算
        T : float
            温度 [K]
        external_field : float
            外加电场 [V/m] (垂直于界面)
        alloy_fraction : float or None
            合金组分 x (A_{1-x}B_x), 用于组分渐变
        """
        self.mat_left = mat_left
        self.mat_right = mat_right
        self.z = np.asarray(z_array, dtype=np.float64)
        self.T = T
        self.external_field = external_field
        self.alloy_fraction = alloy_fraction

        # 界面位置 (中点)
        self.z_interface = (self.z[0] + self.z[-1]) / 2.0

        # 界面过渡宽度
        if layer_thickness is None:
            self.delta_z = (self.z[-1] - self.z[0]) * 0.05
        else:
            self.delta_z = layer_thickness

        # 计算带偏移
        self.delta_Ec, self.delta_Ev = anderson_band_offset(
            mat_left, mat_right, T)

        # 晶格失配和应变
        self.mismatch = lattice_mismatch(mat_left, mat_right)
        self.eps_xx, self.eps_zz = self._compute_strain()

        # 构建势能剖面
        self._build_profile()

    def _smooth_step(self, z, z0, delta):
        """
        平滑阶跃函数 (Tanh 过渡):
            f(z) = 0.5 * (1 + tanh((z - z0) / delta))

        这比 Heaviside 阶跃更适合数值计算,
        因为它提供了连续可微的势能剖面.
        """
        delta = max(delta, 1e-15)
        arg = (z - z0) / delta
        # 数值截断防止溢出
        arg = np.clip(arg, -50, 50)
        return 0.5 * (1.0 + np.tanh(arg))

    def _derivative_smooth_step(self, z, z0, delta):
        """
        平滑阶跃的导数:
            f'(z) = 1/(2*delta) * sech^2((z-z0)/delta)
        """
        delta = max(delta, 1e-15)
        arg = (z - z0) / delta
        arg = np.clip(arg, -50, 50)
        sech = 1.0 / np.cosh(arg)
        return sech**2 / (2.0 * delta)

    def _compute_strain(self):
        """计算界面处的双轴应变"""
        eps_xx = self.mismatch
        # 泊松效应: eps_zz = -2*C12/(C11) * eps_xx ≈ -0.86 * eps_xx
        eps_zz = -2 * 0.43 * eps_xx / (1 - 0.43)
        return eps_xx, eps_zz

    def _band_offset_profile(self):
        """
        导带偏移势能剖面 [eV]:
            V_c(z) = Delta_Ec * smooth_step(z, z_interface, delta_z)

        价带偏移:
            V_v(z) = -Delta_Ev * smooth_step(z, z_interface, delta_z)
                     - E_g_left  (使价带顶为参考)
        """
        step = self._smooth_step(self.z, self.z_interface, self.delta_z)

        # 导带偏移
        V_c = self.delta_Ec * step

        # 价带偏移 (相对于左侧价带顶)
        eg_left = self.mat_left.bandgap_temperature(self.T)
        V_v = -(self.delta_Ev * step + eg_left)

        return V_c, V_v

    def _strain_profile(self):
        """
        应变修正势能 [eV]:
            Delta_E_c^strain(z) = a_c * eps_hydro * (1-step(z)) +
                                   a_c' * eps_hydro' * step(z)

        这里采用简化模型: 应变集中在界面附近.
        """
        step = self._smooth_step(self.z, self.z_interface, self.delta_z)

        # 左侧应变修正 (相对于无应变)
        delta_Ec_left, delta_Ev_left = strain_energy_shift(
            self.mat_left, self.eps_xx, self.eps_xx)
        delta_Ec_right, delta_Ev_right = strain_energy_shift(
            self.mat_right, self.eps_xx, self.eps_xx)

        # 应变在界面处连续过渡
        dE_c = delta_Ec_left * (1 - step) + delta_Ec_right * step
        dE_v = delta_Ev_left * (1 - step) + delta_Ev_right * step

        return dE_c, dE_v

    def _polarization_charge(self):
        """
        极化诱导电荷密度 [C/m²]:
            sigma_pol = P_sp + P_pz
            P_sp = 自发极化
            P_pz = 2*e31*eps_xx + e33*eps_zz  (压电极化)

        对于 TMDC, 自发极化较小, 主要贡献来自压电极化.
        """
        # TMDC 压电常数 (简化值) [C/m²]
        # 真实值约为 10^{-11} 量级, 这里放大以便观察效应
        e31_approx = -0.35e-10  # 典型 TMDC 值 [C/m²]
        e33_approx = 0.70e-10

        P_pz_left = 2 * e31_approx * self.eps_xx + e33_approx * self.eps_zz
        # 右侧材料应变较小 (弛豫)
        P_pz_right = P_pz_left * 0.3

        # 极化电荷集中在界面
        sigma_pol = P_pz_right - P_pz_left

        # 界面处的极化电场 (高斯定理)
        eps_r_avg = (self.mat_left.epsilon_r_z +
                     self.mat_right.epsilon_r_z) / 2.0
        E_pol = sigma_pol / (EPSILON0 * eps_r_avg)

        # 极化电场对应的势能 [eV]
        # V_pol = -e*E_pol*z / e = -E_pol*z [V] = [eV]
        V_pol = -E_pol * (self.z - self.z_interface)

        return V_pol

    def _external_field_profile(self):
        """
        外电场导致的势能 (量子限制 Stark 效应):
            V_ext(z) = -e * F_ext * z
        """
        if abs(self.external_field) < 1e-10:
            return np.zeros_like(self.z)
        # V = -eFz, 转为 eV: V[eV] = -F[V/m]*z[m] (因为 1eV = e*1V)
        return -self.external_field * (self.z - self.z_interface)

    def _alloy_potential(self):
        """
        合金组分渐变势能 (如果指定 alloy_fraction):
            E_g(x) = (1-x)*E_g_A + x*E_g_B - b*x*(1-x)
        其中 b 是弯曲参数 (bowing parameter).
        """
        if self.alloy_fraction is None:
            return 0.0, 0.0

        x = self.alloy_fraction
        x = np.clip(x, 0.0, 1.0)

        # 线性插值 + 弯曲修正
        eg_A = self.mat_left.eg_0
        eg_B = self.mat_right.eg_0
        bowing = 0.5  # 典型弯曲参数 [eV]

        eg_alloy = (1 - x) * eg_A + x * eg_B - bowing * x * (1 - x)

        # 合金带隙变化
        delta_eg = eg_alloy - eg_A
        V_alloy_c = delta_eg * 0.6  # 60% 分配给导带
        V_alloy_v = -delta_eg * 0.4  # 40% 分配给价带

        return V_alloy_c, V_alloy_v

    def _build_profile(self):
        """构建总势能剖面"""
        # 带偏移
        self.Vc_offset, self.Vv_offset = self._band_offset_profile()

        # 应变修正
        self.Vc_strain, self.Vv_strain = self._strain_profile()

        # 极化
        self.V_pol = self._polarization_charge()

        # 外电场
        self.V_ext = self._external_field_profile()

        # 合金
        self.Vc_alloy, self.Vv_alloy = self._alloy_potential()

        # 总导带势能
        self.V_conduction = (self.Vc_offset + self.Vc_strain +
                             self.V_pol + self.V_ext + self.Vc_alloy)

        # 总价带势能
        eg_left = self.mat_left.bandgap_temperature(self.T)
        self.V_valence = (self.Vv_offset + self.Vv_strain +
                          self.V_pol + self.V_ext + self.Vv_alloy)

    def get_conduction_band(self):
        """返回导带势能剖面 [eV]"""
        return self.z.copy(), self.V_conduction.copy()

    def get_valence_band(self):
        """返回价带势能剖面 [eV]"""
        return self.z.copy(), self.V_valence.copy()

    def barrier_height(self, carrier='electron'):
        """
        异质结势垒高度 [eV]:
            对电子: V_B = max(V_c) - min(V_c)
            对空穴: V_B = max(V_v) - min(V_v)
        """
        if carrier == 'electron':
            V = self.V_conduction
        else:
            V = self.V_valence
        return np.max(V) - np.min(V)

    def effective_field_at_interface(self):
        """
        界面处有效电场 [V/m]:
            F_eff = -dV/dz |_{z=z_interface}

        使用中心差分近似.
        """
        dz = self.z[1] - self.z[0]
        idx_interface = np.argmin(np.abs(self.z - self.z_interface))

        if idx_interface <= 0 or idx_interface >= len(self.z) - 1:
            return 0.0

        # 中心差分
        dVdz = (self.V_conduction[idx_interface + 1] -
                self.V_conduction[idx_interface - 1]) / (2 * dz)

        return -dVdz  # [eV/m]

    def depletion_width(self, doping_left=1e18, doping_right=1e18):
        """
        耗尽层宽度 (突变结近似):
            W = sqrt(2*eps*V_bi*(N_A+N_D)/(e*N_A*N_D))

        对于二维材料, 体掺杂概念需要修正,
        这里用面载流子密度代替.
        """
        eps_avg = EPSILON0 * (self.mat_left.epsilon_r_z +
                              self.mat_right.epsilon_r_z) / 2.0
        V_bi = abs(self.delta_Ec)  # 内建电势近似

        # 避免物理不合理值
        doping_left = max(doping_left, 1e10)
        doping_right = max(doping_right, 1e10)

        W = np.sqrt(2 * eps_avg * V_bi / E0 *
                    (doping_left + doping_right) /
                    (doping_left * doping_right))
        return W

    def __repr__(self):
        return (f"HeteroPotential({self.mat_left.name}/{self.mat_right.name}, "
                f"Delta_Ec={self.delta_Ec:.3f}eV, "
                f"Delta_Ev={self.delta_Ev:.3f}eV)")
