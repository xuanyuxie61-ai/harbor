r"""
surface_reaction.py
===================
催化剂表面反应动力学模型与热力学计算。

本模块提供多种工业催化反应的动力学表达式，包括：
1. Langmuir-Hinshelwood (L-H) 表面反应动力学；
2. Eley-Rideal (E-R) 机理；
3. 幂律动力学（Power-law）；
4. 温度依赖的 Arrhenius 参数。

核心公式：

**Langmuir-Hinshelwood 速率（以双分子反应 A + B → C 为例）：**
    R = \frac{k_r K_A K_B C_A C_B}
             {(1 + K_A C_A + K_B C_B + K_C C_C)^2}

其中：
    k_r = k_0 \exp(-E_{a,r} / RT)
    K_i = K_{i,0} \exp(-\Delta H_{ads,i} / RT)

**Thiele 模数（一级近似）：**
    \phi = R_p \sqrt{\frac{k_{obs}(C_{surf}, T_{surf})}{D_e}}

**Weisz-Prater 准则：**
    C_{WP} = \eta \phi^2 = \frac{R_{obs}}{D_e C_{surf} / R_p^2}
    若 C_{WP} \ll 1，则颗粒内扩散限制可忽略。
"""

import numpy as np
from special_functions import arrhenius_rate


class SurfaceReactionError(Exception):
    """表面反应模型异常。"""
    pass


class LangmuirHinshelwoodKinetics:
    r"""
    Langmuir-Hinshelwood 表面反应动力学。

    对于 CO 氧化反应（典型 L-H 机理）：
        CO + 1/2 O_2 → CO_2

    速率表达式：
        R = \frac{k_{r} K_{CO} C_{CO} \sqrt{K_{O_2} C_{O_2}}}
                 {(1 + K_{CO} C_{CO} + \sqrt{K_{O_2} C_{O_2}})^2}

    为简化数值求解，使用双分子形式：
        R = \frac{k_{r} K_A C_A K_B C_B}
                 {(1 + K_A C_A + K_B C_B)^2}
    """

    def __init__(self, k0, Ea, KA0, dH_ads_A, KB0, dH_ads_B,
                 reaction_order_A=1.0, reaction_order_B=1.0):
        """
        Parameters
        ----------
        k0 : float
            表面反应指前因子 [单位取决于反应级数]。
        Ea : float
            表面反应活化能 [J/mol]。
        KA0, KB0 : float
            吸附平衡常数指前因子 [m³/mol]。
        dH_ads_A, dH_ads_B : float
            吸附焓 [J/mol]（通常为负值）。
        reaction_order_A, reaction_order_B : float
            反应级数。
        """
        self.k0 = k0
        self.Ea = Ea
        self.KA0 = KA0
        self.dH_ads_A = dH_ads_A
        self.KB0 = KB0
        self.dH_ads_B = dH_ads_B
        self.reaction_order_A = reaction_order_A
        self.reaction_order_B = reaction_order_B

    def rate(self, CA, CB, temperature):
        """
        计算反应速率 [mol/(m³·s)]。

        Parameters
        ----------
        CA, CB : float
            反应物 A, B 的浓度 [mol/m³]。
        temperature : float
            温度 [K]。

        Returns
        -------
        rate : float
        """
        if CA < 0 or CB < 0:
            raise SurfaceReactionError("浓度必须非负")
        if temperature <= 0:
            raise SurfaceReactionError("温度必须为正")

        # TODO: Hole 3 — 实现 Langmuir-Hinshelwood 表面反应速率公式
        # 要求：
        # 1. 计算 Arrhenius 速率常数：kr = k0 * exp(-Ea / (R * T))
        #    其中 R = 8.314462618 J/(mol·K)
        # 2. 计算温度依赖的吸附平衡常数：
        #    KA = KA0 * exp(-dH_ads_A / (R * T))
        #    KB = KB0 * exp(-dH_ads_B / (R * T))
        #    并 clip 到 [0, 1e20] 防止溢出
        # 3. 计算 L-H 速率：
        #    R = kr * (KA*CA)^order_A * (KB*CB)^order_B / (1 + KA*CA + KB*CB)^2
        # 4. 处理 denominator 接近零的情况，返回 0.0
        # 5. 返回 max(0.0, rate) 保证非负
        # 注意：此 rate 被 Hole 1 (pore_diffusion.py) 和 Hole 2 (nonlinear_solver.py)
        #       中的求解器作为反应源项调用，公式错误将导致两个求解器都失败。
        raise NotImplementedError("Hole 3: 请实现 Langmuir-Hinshelwood 反应速率公式")

    def jacobian_entries(self, CA, CB, temperature):
        """
        计算速率对浓度的偏导数（用于牛顿迭代）。

        返回 (∂R/∂CA, ∂R/∂CB) 的近似值。
        """
        eps = np.sqrt(np.finfo(float).eps) * max(CA, 1e-12)
        R0 = self.rate(CA, CB, temperature)
        RA = self.rate(CA + eps, CB, temperature)
        RB = self.rate(CA, CB + eps, temperature)
        dRdCA = (RA - R0) / eps
        dRdCB = (RB - R0) / eps
        return dRdCA, dRdCB


class PowerLawKinetics:
    """
    幂律反应动力学：
        R = k C_A^{n_A} C_B^{n_B}

    常用于初步设计和动力学参数估算。
    """

    def __init__(self, k0, Ea, nA, nB):
        self.k0 = k0
        self.Ea = Ea
        self.nA = nA
        self.nB = nB

    def rate(self, CA, CB, temperature):
        if CA < 0 or CB < 0:
            raise SurfaceReactionError("浓度必须非负")
        k = arrhenius_rate(self.k0, self.Ea, temperature)
        CA_eff = max(CA, 0.0)
        CB_eff = max(CB, 0.0)
        return k * (CA_eff ** self.nA) * (CB_eff ** self.nB)


class CatalyticParticleModel:
    r"""
    完整的催化剂颗粒多物理场模型。

    耦合方程：
        质量守恒：D_e \nabla^2 C - R(C, T) = 0
        能量守恒：\lambda_e \nabla^2 T + (-\Delta H) R(C, T) = 0

    有效导热系数：
        \lambda_e = \lambda_{solid} (1-\varepsilon) + \lambda_{gas} \varepsilon
    """

    def __init__(self, kinetics, particle_radius, porosity, tortuosity,
                 lambda_solid, lambda_gas, heat_of_reaction,
                 T_surface, C_surface_A, C_surface_B):
        """
        Parameters
        ----------
        kinetics : object
            具有 rate(CA, CB, T) 方法的动力学对象。
        particle_radius : float
            颗粒半径 [m]。
        porosity, tortuosity : float
        lambda_solid, lambda_gas : float
            固相与气相导热系数 [W/(m·K)]。
        heat_of_reaction : float
            反应焓 [J/mol]（放热为负）。
        T_surface : float
            表面温度 [K]。
        C_surface_A, C_surface_B : float
            表面浓度 [mol/m³]。
        """
        self.kinetics = kinetics
        self.Rp = particle_radius
        self.porosity = porosity
        self.tortuosity = tortuosity
        self.lambda_eff = lambda_solid * (1.0 - porosity) + lambda_gas * porosity
        self.heat_of_reaction = heat_of_reaction
        self.T_surface = T_surface
        self.C_surface_A = C_surface_A
        self.C_surface_B = C_surface_B

    def effective_diffusivity(self, pore_diameter, temperature, molecular_weight,
                              bulk_diffusivity):
        """
        计算有效扩散系数。
        """
        from special_functions import effective_diffusivity
        return effective_diffusivity(
            pore_diameter, temperature, molecular_weight,
            bulk_diffusivity, self.tortuosity, self.porosity
        )

    def reaction_rate_local(self, CA, CB, T):
        """局部反应速率 [mol/(m³·s)]。"""
        return self.kinetics.rate(CA, CB, T)

    def heat_source(self, CA, CB, T):
        """局部热源 [W/m³] = R * (-ΔH)。"""
        return self.reaction_rate_local(CA, CB, T) * (-self.heat_of_reaction)

    def thiele_modulus(self, D_e, T_surf):
        r"""
        基于表面条件的 Thiele 模数。

        对于 L-H 动力学，使用一级近似：
            k_{obs} = R(C_surf, T_surf) / C_surf
            \phi = R_p \sqrt{k_{obs} / D_e}
        """
        R_surf = self.reaction_rate_local(self.C_surface_A,
                                          self.C_surface_B, T_surf)
        if self.C_surface_A < np.finfo(float).eps:
            return 0.0
        k_obs = R_surf / self.C_surface_A
        phi = self.Rp * np.sqrt(max(k_obs, 0.0) / max(D_e, 1e-20))
        return phi

    def weisz_prater_criterion(self, eta, D_e, T_surf):
        """
        Weisz-Prater 准则。

        C_WP = η φ²
        """
        phi = self.thiele_modulus(D_e, T_surf)
        return eta * (phi ** 2)
