#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parameter_manager.py
物理参数管理与频散模型

融合种子项目 060_axon_ode (Hodgkin-Huxley门控动力学) 的思想，
将离子通道门控方程映射到地下介质电导率的频散特性建模。

核心物理：
  - 大地电磁测深(MT)中，地下岩石电导率 σ 随频率 ω 变化
  - Cole-Cole 频散模型：σ(ω) = σ_0 [1 + m * (iωτ)^c / (1 + (iωτ)^c)]
  - 该模型与 HH 模型的门控动力学具有数学同构性
"""

import numpy as np


class PhysicalConstants:
    """物理常数管理类"""

    MU_0 = 4.0 * np.pi * 1e-7          # 真空磁导率 [H/m]
    EPSILON_0 = 8.854187817e-12        # 真空介电常数 [F/m]
    MU_0_INV = 1.0 / MU_0              # 磁导率倒数

    @classmethod
    def angular_frequency(cls, freq_hz):
        """由频率 [Hz] 计算角频率 ω = 2πf [rad/s]"""
        if freq_hz <= 0.0:
            raise ValueError("频率必须为正数")
        return 2.0 * np.pi * freq_hz

    @classmethod
    def skin_depth(cls, freq_hz, conductivity_s_m):
        """
        计算趋肤深度 δ = √(2 / (ωμ₀σ))
        这是电磁波在导电介质中的特征穿透深度。
        """
        if conductivity_s_m <= 0.0:
            raise ValueError("电导率必须为正数")
        omega = cls.angular_frequency(freq_hz)
        delta = np.sqrt(2.0 / (omega * cls.MU_0 * conductivity_s_m))
        return delta


class ColeColeDispersion:
    """
    Cole-Cole 频散模型

    基于 Hodgkin-Huxley 门控动力学思想，将电导率随频率的变化
    建模为类似于离子通道门控的弛豫过程。

    复电导率公式：
        σ*(ω) = σ_0 [1 + m * (iωτ)^c / (1 + (iωτ)^c)]

    其中：
        σ_0   — 直流电导率 [S/m]
        m     — 充电率 (chargeability), 0 ≤ m ≤ 1
        τ     — 时间常数 [s]
        c     — 频率相关系数, 0 < c ≤ 1
    """

    def __init__(self, sigma_0, m_charge=0.0, tau=1e-3, c_freq=1.0):
        self.sigma_0 = float(sigma_0)
        self.m_charge = float(m_charge)
        self.tau = float(tau)
        self.c_freq = float(c_freq)
        self._validate()

    def _validate(self):
        if self.sigma_0 <= 0.0:
            raise ValueError("直流电导率 σ_0 必须为正")
        if not (0.0 <= self.m_charge <= 1.0):
            raise ValueError("充电率 m 必须在 [0, 1] 范围内")
        if self.tau <= 0.0:
            raise ValueError("时间常数 τ 必须为正")
        if not (0.0 < self.c_freq <= 1.0):
            raise ValueError("频率相关系数 c 必须在 (0, 1] 范围内")

    def complex_conductivity(self, omega):
        """
        计算复电导率 σ*(ω)

        公式：σ*(ω) = σ_0 [1 + m * (iωτ)^c / (1 + (iωτ)^c)]
        """
        omega = np.asarray(omega, dtype=np.float64)
        if np.any(omega <= 0.0):
            raise ValueError("角频率必须为正")

        iwt = (1j * omega * self.tau) ** self.c_freq
        sigma_star = self.sigma_0 * (1.0 + self.m_charge * iwt / (1.0 + iwt))
        return sigma_star

    def complex_resistivity(self, omega):
        """复电阻率 ρ*(ω) = 1 / σ*(ω)"""
        return 1.0 / self.complex_conductivity(omega)

    @staticmethod
    def hh_gating_analogy(V, alpha_0, beta_0, V_shift, k_T):
        """
        Hodgkin-Huxley 门控速率常数的类比实现

        原始 HH 方程中：
            α_n(V) = 0.01 * (10 - V) / (exp((10 - V)/10) - 1)
            β_n(V) = 0.125 * exp(-V/80)

        将其映射为介质响应的"激活"与"失活"速率：
            α(V) = α_0 * (V_shift - V) / (exp((V_shift - V)/k_T) - 1)
            β(V) = β_0 * exp(-V/k_T)

        这里 V 对应于外加电磁场的等效"驱动电势"。
        """
        eps = 1e-12
        denom = np.exp((V_shift - V) / k_T) - 1.0
        if np.abs(denom) < eps:
            denom = eps
        alpha = alpha_0 * (V_shift - V) / denom
        beta = beta_0 * np.exp(-V / k_T)
        return alpha, beta


class LayeredEarthModel:
    """
    层状大地电性模型

    描述 N 层水平层状介质的电阻率、厚度及频散特性。
    """

    def __init__(self, resistivities, thicknesses, dispersion_list=None):
        """
        Parameters
        ----------
        resistivities : array_like, shape (n_layers,)
            各层电阻率 [Ω·m]
        thicknesses : array_like, shape (n_layers - 1,)
            各层厚度 [m]，最后一层为半无限空间
        dispersion_list : list of ColeColeDispersion or None
            各层的频散模型，None 表示无频散
        """
        self.resistivities = np.asarray(resistivities, dtype=np.float64)
        self.thicknesses = np.asarray(thicknesses, dtype=np.float64)
        self.n_layers = len(self.resistivities)

        if len(self.thicknesses) != self.n_layers - 1:
            raise ValueError("厚度数组长度必须等于层数减一")
        if np.any(self.resistivities <= 0.0):
            raise ValueError("所有电阻率必须为正")
        if np.any(self.thicknesses <= 0.0):
            raise ValueError("所有厚度必须为正")

        if dispersion_list is None:
            self.dispersion_list = [None] * self.n_layers
        else:
            if len(dispersion_list) != self.n_layers:
                raise ValueError("频散模型列表长度必须等于层数")
            self.dispersion_list = dispersion_list

    def get_conductivity(self, layer_idx, omega=None):
        """
        获取指定层的电导率

        若 omega 为 None，返回直流电导率；否则返回复电导率。
        """
        sigma_dc = 1.0 / self.resistivities[layer_idx]
        disp = self.dispersion_list[layer_idx]
        if omega is not None and disp is not None:
            return disp.complex_conductivity(omega)
        return sigma_dc

    def get_resistivity(self, layer_idx, omega=None):
        """获取指定层的电阻率"""
        sigma = self.get_conductivity(layer_idx, omega)
        return 1.0 / sigma

    def depth_to_layer(self, depth):
        """
        根据深度确定所在层索引

        depth > 0 表示向下深度。
        """
        if depth < 0.0:
            return 0
        cum_depth = 0.0
        for i, h in enumerate(self.thicknesses):
            cum_depth += h
            if depth <= cum_depth:
                return i
        return self.n_layers - 1

    def summary(self):
        """输出模型摘要"""
        lines = ["=" * 60, "层状大地电性模型", "=" * 60]
        lines.append(f"{'层号':>4} {'电阻率(Ω·m)':>14} {'厚度(m)':>12} {'频散':>8}")
        for i in range(self.n_layers):
            rho = self.resistivities[i]
            h = self.thicknesses[i] if i < self.n_layers - 1 else np.inf
            disp = "有" if self.dispersion_list[i] is not None else "无"
            h_str = f"{h:.2f}" if np.isfinite(h) else "∞"
            lines.append(f"{i + 1:>4} {rho:>14.2f} {h_str:>12} {disp:>8}")
        return "\n".join(lines)


class MTDataContainer:
    """
    MT 观测数据容器

    存储各频率下的阻抗张量、视电阻率、相位等观测数据。
    """

    def __init__(self, frequencies, zxx=None, zxy=None, zyx=None, zyy=None,
                 rho_a_xy=None, rho_a_yx=None, phi_xy=None, phi_yx=None,
                 errors=None):
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        self.n_freq = len(self.frequencies)

        def _init(val, dtype=np.complex128):
            if val is None:
                return np.zeros(self.n_freq, dtype=dtype)
            v = np.asarray(val, dtype=dtype)
            if len(v) != self.n_freq:
                raise ValueError("数据长度必须与频率数一致")
            return v

        self.Zxx = _init(zxx)
        self.Zxy = _init(zxy)
        self.Zyx = _init(zyx)
        self.Zyy = _init(zyy)
        self.rho_a_xy = _init(rho_a_xy, np.float64)
        self.rho_a_yx = _init(rho_a_yx, np.float64)
        self.phi_xy = _init(phi_xy, np.float64)
        self.phi_yx = _init(phi_yx, np.float64)

        if errors is None:
            self.errors = np.ones(self.n_freq, dtype=np.float64) * 0.05
        else:
            self.errors = np.asarray(errors, dtype=np.float64)
            if len(self.errors) != self.n_freq:
                raise ValueError("误差数组长度必须与频率数一致")

    def compute_apparent_resistivity_phase(self):
        """
        由阻抗张量计算视电阻率和相位

        公式：
            ρ_a = |Z|² / (ω μ_0)
            φ = arg(Z)  [度]
        """
        mu0 = PhysicalConstants.MU_0
        for comp, Z in [("xy", self.Zxy), ("yx", self.Zyx)]:
            omega = 2.0 * np.pi * self.frequencies
            rho_a = np.abs(Z) ** 2 / (omega * mu0)
            phi = np.angle(Z, deg=True)
            if comp == "xy":
                self.rho_a_xy = rho_a
                self.phi_xy = phi
            else:
                self.rho_a_yx = rho_a
                self.phi_yx = phi

    @staticmethod
    def impedance_from_rhophi(rho_a, phi_deg, freq_hz):
        """
        由视电阻率和相位反推阻抗

        Z = √(ω μ_0 ρ_a) * exp(i φ)
        """
        omega = 2.0 * np.pi * freq_hz
        mu0 = PhysicalConstants.MU_0
        Z = np.sqrt(omega * mu0 * rho_a) * np.exp(1j * np.deg2rad(phi_deg))
        return Z


if __name__ == "__main__":
    # 简单自检
    pc = PhysicalConstants
    print(f"μ₀ = {pc.MU_0:.6e} H/m")
    print(f"趋肤深度(100 Hz, 0.01 S/m) = {pc.skin_depth(100.0, 0.01):.2f} m")

    cc = ColeColeDispersion(sigma_0=0.01, m_charge=0.5, tau=1e-2, c_freq=0.8)
    omega = 2.0 * np.pi * np.array([0.1, 1.0, 10.0, 100.0])
    sigma_star = cc.complex_conductivity(omega)
    print(f"复电导率: {sigma_star}")

    model = LayeredEarthModel(
        resistivities=[100.0, 50.0, 10.0, 200.0],
        thicknesses=[500.0, 1000.0, 2000.0]
    )
    print(model.summary())
