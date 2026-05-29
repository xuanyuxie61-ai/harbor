"""
binary_black_hole.py
双黑洞系统物理参数与轨道力学模块。

核心公式:
1. 啁啾质量 (Chirp Mass):
   M_c = (m1 * m2)^{3/5} / (m1 + m2)^{1/5} = M * η^{3/5}

2. 对称质量比:
   η = m1 * m2 / (m1 + m2)^2

3. 后牛顿展开参数:
   x = (G M Ω / c^3)^{2/3} = v^2/c^2

4. 旋进时间 (leading order):
   t_{inspiral} = (5/256) * (c^3 / G M_c)^{5/3} * (1 / (π f_{ISCO})^{8/3})

5. 最内稳定圆轨道 (ISCO, Schwarzschild):
   r_{ISCO} = 6 G M / c^2
   f_{ISCO} = (1/π) * √(G M / r_{ISCO}^3) = c^3 / (6^{3/2} π G M)

6. 引力波频率与轨道频率关系:
   f_{GW} = 2 * f_{orbit} = Ω / π
"""

import numpy as np


class BinaryBlackHole:
    """
    双黑洞系统物理模型。
    
    所有计算使用几何单位制 (G = c = 1)，但在接口层提供物理单位转换。
    """
    
    # 物理常数 (SI)
    G_SI = 6.67430e-11      # m^3 kg^-1 s^-2
    C_SI = 2.99792458e8     # m/s
    MSUN_SI = 1.98847e30    # kg
    MPC_SI = 3.08567758e22  # m
    
    def __init__(self, m1_msun=30.0, m2_msun=25.0, a1=0.0, a2=0.0,
                 D_L_mpc=400.0, inclination=0.0, phi_c=0.0, psi=0.0, t_c=0.0):
        """
        初始化双黑洞系统。
        
        参数:
            m1_msun, m2_msun: 主/次级黑洞质量 [太阳质量]
            a1, a2: 无量纲自旋参数 [-1, 1]
            D_L_mpc: 光度距离 [Mpc]
            inclination: 轨道倾角 [rad]
            phi_c: 并合相位 [rad]
            psi: 偏振角 [rad]
            t_c: 并合时间 [s]
        """
        self.m1_msun = float(m1_msun)
        self.m2_msun = float(m2_msun)
        self.a1 = float(np.clip(a1, -0.999, 0.999))
        self.a2 = float(np.clip(a2, -0.999, 0.999))
        self.D_L_mpc = float(D_L_mpc)
        self.inclination = float(inclination)
        self.phi_c = float(phi_c)
        self.psi = float(psi)
        self.t_c = float(t_c)
        
        # 单位转换到几何单位制
        self.m1 = self.m1_msun * self.MSUN_SI * self.G_SI / self.C_SI**3  # [s]
        self.m2 = self.m2_msun * self.MSUN_SI * self.G_SI / self.C_SI**3
        self.M = self.m1 + self.m2
        self.eta = self.m1 * self.m2 / self.M**2
        self.eta = np.clip(self.eta, 1e-6, 0.25)
        
        self.M_c = self.M * self.eta**(3.0 / 5.0)
        self.D_L = self.D_L_mpc * self.MPC_SI / self.C_SI  # [s]
        
        # ISCO 频率
        self.f_isco = self._compute_isco_frequency()
        
        # 旋进时间估算
        self.t_inspiral = self._compute_inspiral_time()
    
    def _compute_isco_frequency(self):
        """
        计算最内稳定圆轨道对应的引力波频率。
        
        公式 (Schwarzschild):
            f_{ISCO} = c^3 / (6^{3/2} π G M)
                     = 1 / (6^{3/2} π M_geom)
        """
        return 1.0 / (6.0**1.5 * np.pi * self.M)
    
    def _compute_inspiral_time(self):
        """
        计算从初始频率到 ISCO 的旋进时间。
        
        公式 (leading order):
            t_{inspiral} = (5/256) * M_c / η * (1/x_{ISCO})^4
            x_{ISCO} = (M Ω_{ISCO})^{2/3} = 1/6
        """
        # TODO: 请补全旋进时间计算公式
        # 关键物理: x_ISCO = 1/6 (Schwarzschild ISCO 的后牛顿展开参数)
        #          t_inspiral = (5/256) * M_c / eta * (1/x_ISCO)^4
        raise NotImplementedError("Hole 2: _compute_inspiral_time 核心公式待补全")
    
    def orbital_separation(self, f_gw):
        """
        由引力波频率计算轨道分离 (Newtonian)。
        
        公式:
            f_{GW} = (1/π) * √(M / r^3)
            r = (M / (π f_{GW})^2)^{1/3}
        """
        f_gw = np.asarray(f_gw, dtype=np.float64)
        f_gw = np.where(f_gw <= 0, 1e-10, f_gw)
        return (self.M / (np.pi * f_gw)**2)**(1.0 / 3.0)
    
    def orbital_velocity(self, f_gw):
        """
        由引力波频率计算轨道速度 (Newtonian)。
        
        公式:
            v = (π M f_{GW})^{1/3}
        """
        f_gw = np.asarray(f_gw, dtype=np.float64)
        return (np.pi * self.M * f_gw)**(1.0 / 3.0)
    
    def energy_flux(self, f_gw):
        """
        引力波能量通量 (quadrupole formula)。
        
        公式:
            dE/dt = (32/5) * η^2 * (π M f_{GW})^{10/3}
        """
        f_gw = np.asarray(f_gw, dtype=np.float64)
        x = np.pi * self.M * f_gw
        return (32.0 / 5.0) * self.eta**2 * x**(10.0 / 3.0)
    
    def strain_amplitude(self, f_gw):
        """
        引力波特征应变振幅。
        
        公式:
            h_c = (1/D_L) * (G M_c / c^2)^{5/3} * (π f_{GW} D_L / c)^{2/3}
            简化: h_c ≈ (M_c^{5/3} / D_L) * (π f_{GW})^{2/3}
        """
        f_gw = np.asarray(f_gw, dtype=np.float64)
        return (self.M_c**(5.0 / 3.0) / self.D_L) * (np.pi * f_gw)**(2.0 / 3.0)
    
    def to_parameter_dict(self):
        """导出为参数字典。"""
        return {
            'm1': self.m1_msun,
            'm2': self.m2_msun,
            'a1': self.a1,
            'a2': self.a2,
            'D_L': self.D_L_mpc,
            'inclination': self.inclination,
            'phi_c': self.phi_c,
            'psi': self.psi,
            't_c': self.t_c
        }
    
    @classmethod
    def from_parameter_dict(cls, params):
        """从参数字典构建。"""
        return cls(
            m1_msun=params.get('m1', 30.0),
            m2_msun=params.get('m2', 25.0),
            a1=params.get('a1', 0.0),
            a2=params.get('a2', 0.0),
            D_L_mpc=params.get('D_L', 400.0),
            inclination=params.get('inclination', 0.0),
            phi_c=params.get('phi_c', 0.0),
            psi=params.get('psi', 0.0),
            t_c=params.get('t_c', 0.0)
        )
    
    def info(self):
        """返回系统信息摘要。"""
        return {
            'm1_msun': self.m1_msun,
            'm2_msun': self.m2_msun,
            'total_mass_msun': self.m1_msun + self.m2_msun,
            'chirp_mass_msun': (self.m1_msun * self.m2_msun)**(3.0/5.0) / (self.m1_msun + self.m2_msun)**(1.0/5.0),
            'symmetric_mass_ratio': self.eta,
            'spin1': self.a1,
            'spin2': self.a2,
            'distance_mpc': self.D_L_mpc,
            'f_isco_hz': self.f_isco,
            't_inspiral_s': self.t_inspiral,
        }


def mass_ratio_to_components(q, M_total):
    """
    由质量比 q = m1/m2 (q >= 1) 和总质量计算分量质量。
    
    公式:
        m1 = q * M_total / (1 + q)
        m2 = M_total / (1 + q)
    """
    if q < 1.0:
        q = 1.0 / q
    m1 = q * M_total / (1.0 + q)
    m2 = M_total / (1.0 + q)
    return m1, m2


def effective_spin(m1, m2, a1, a2):
    """
    计算有效自旋参数 χ_eff。
    
    公式:
        χ_eff = (m1*a1 + m2*a2) / (m1 + m2)
    """
    return (m1 * a1 + m2 * a2) / (m1 + m2)


def precession_spin(m1, m2, a1, a2, theta1, theta2):
    """
    计算进动有效自旋 χ_p。
    
    公式 (简化):
        χ_p = max( a1*sin(θ1), q*(4q+3)/(4+3q)*a2*sin(θ2) )
        q = m1/m2
    """
    q = m1 / m2
    term1 = np.abs(a1 * np.sin(theta1))
    term2 = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q) * np.abs(a2 * np.sin(theta2))
    return max(term1, term2)
