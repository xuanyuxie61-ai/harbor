# -*- coding: utf-8 -*-
"""
parameters.py
全局物理参数管理模块
基于种子项目 702_logistic_ode 的参数管理思想重构

本模块管理等离子体鞘层与壁材料侵蚀模拟所需的全局物理参数，
包括等离子体参数、鞘层参数、壁材料参数和数值算法参数。
"""

import numpy as np


class PlasmaParameters:
    """
    等离子体物理参数类
    
    统一管理托卡马克偏滤器靶板附近等离子体-壁相互作用所需的全部物理参数。
    参数默认值基于典型托卡马克边缘等离子体条件（如ITER-like装置）。
    """

    # 默认参数值（类变量，persistent 语义）
    _defaults = {
        # 等离子体基本参数
        'n_0': 1.0e19,          # 上游等离子体密度 [m^-3]
        'T_e': 50.0,            # 电子温度 [eV]
        'T_i': 50.0,            # 离子温度 [eV]
        'B_t': 5.3,             # 环向磁场 [T]
        'm_i': 2.0,             # 离子质量数（氘）
        'Z_i': 1,               # 离子电荷数

        # 鞘层参数
        'L_sheath': 5.0e-3,     # 鞘层特征长度 [m]
        'gamma_sheath': 7.0,    # 鞘层传热系数

        # 壁材料参数（钨）
        'wall_material': 'W',
        'wall_Z': 74,           # 钨原子序数
        'wall_M': 183.84,       # 钨原子质量 [amu]
        'wall_density': 19300.0, # 钨质量密度 [kg/m^3]
        'E_bind': 8.68,         # 钨表面结合能 [eV]
        'E_threshold': 200.0,   # 溅射阈值能量 [eV]

        # 数值参数
        'nx': 256,              # 空间网格数
        'x_max': 0.01,          # 计算域长度 [m]
        'max_iter': 10000,      # 最大迭代次数
        'tol': 1.0e-10,         # 收敛容差
        'dt': 1.0e-12,          # 时间步长 [s]
        't_stop': 1.0e-6,       # 终止时间 [s]

        # 蒙特卡洛参数
        'n_mc_samples': 10000,  # MC采样数
        'rand_seed': 314159265, # 随机数种子
    }

    def __init__(self, **kwargs):
        """
        初始化参数，支持用户覆盖默认值
        """
        self._params = dict(self._defaults)
        for key, value in kwargs.items():
            if key in self._params:
                self._params[key] = value
            else:
                raise ValueError(f"未知参数: {key}")
        self._validate()

    def _validate(self):
        """
        参数边界校验与物理合理性检查
        """
        p = self._params
        # 密度必须为正
        if p['n_0'] <= 0:
            raise ValueError("等离子体密度 n_0 必须为正")
        # 温度必须为正
        if p['T_e'] <= 0 or p['T_i'] <= 0:
            raise ValueError("温度必须为正")
        # 鞘层长度必须为正
        if p['L_sheath'] <= 0:
            raise ValueError("鞘层长度必须为正")
        # 网格数必须合理
        if p['nx'] < 4:
            raise ValueError("空间网格数 nx 至少为 4")
        if p['nx'] > 10000:
            raise ValueError("空间网格数 nx 过大，可能导致内存问题")
        # 时间参数
        if p['dt'] <= 0 or p['t_stop'] <= 0:
            raise ValueError("时间参数必须为正")
        if p['dt'] > p['t_stop']:
            raise ValueError("时间步长 dt 不能大于终止时间 t_stop")
        # 收敛容差
        if p['tol'] <= 0 or p['tol'] >= 1.0:
            raise ValueError("收敛容差 tol 必须在 (0, 1) 区间")

    def get(self, key):
        """获取参数值"""
        if key not in self._params:
            raise KeyError(f"参数 {key} 不存在")
        return self._params[key]

    def set(self, key, value):
        """设置参数值并重新校验"""
        if key not in self._params:
            raise KeyError(f"参数 {key} 不存在")
        self._params[key] = value
        self._validate()

    def get_all(self):
        """返回全部参数字典"""
        return dict(self._params)

    # ---- 派生物理量计算 ----

    def debye_length(self):
        """
        计算德拜长度:
            lambda_D = sqrt(epsilon_0 * k_B * T_e / (e^2 * n_0))
        其中 T_e 以 eV 为单位时，k_B*T_e = e*T_e，故:
            lambda_D = sqrt(epsilon_0 * T_e / (e * n_0))  [m]
        """
        epsilon_0 = 8.854187817e-12  # F/m
        e_charge = 1.602176634e-19   # C
        n0 = self._params['n_0']
        Te = self._params['T_e']
        if n0 <= 0 or Te <= 0:
            return np.nan
        return np.sqrt(epsilon_0 * Te / (e_charge * n0))

    def ion_sound_speed(self):
        """
        计算离子声速（Bohm速度）:
            c_s = sqrt((Z_i * T_e + T_i) / m_i)  [m/s]
        其中 m_i 以 amu 为单位，需转换为 kg:
            m_i[kg] = m_i[amu] * m_p
        """
        m_p = 1.67262192369e-27  # kg
        Zi = self._params['Z_i']
        Te = self._params['T_e']
        Ti = self._params['T_i']
        mi_amu = self._params['m_i']
        mi_kg = mi_amu * m_p
        e_charge = 1.602176634e-19  # J/eV
        if mi_kg <= 0:
            return np.nan
        return np.sqrt((Zi * Te + Ti) * e_charge / mi_kg)

    def bohm_criterion(self):
        """
        Bohm判据: 离子进入鞘层的马赫数 M >= 1
        返回 Bohm 速度对应的能量 [eV]
        """
        cs = self.ion_sound_speed()
        m_p = 1.67262192369e-27
        mi_kg = self._params['m_i'] * m_p
        e_charge = 1.602176634e-19
        if cs <= 0 or mi_kg <= 0:
            return np.nan
        E_bohm = 0.5 * mi_kg * cs**2 / e_charge
        return E_bohm

    def sheath_potential(self):
        """
        计算鞘层电势降（简化模型）:
            Delta_phi = (T_e / 2) * ln(2*pi * m_e / m_i * (1 + T_i/T_e))
        """
        m_e = 9.10938356e-31  # kg
        m_p = 1.67262192369e-27
        Te = self._params['T_e']
        Ti = self._params['T_i']
        mi_kg = self._params['m_i'] * m_p
        if mi_kg <= 0 or Te <= 0:
            return np.nan
        ratio = 2.0 * np.pi * m_e / mi_kg * (1.0 + Ti / Te)
        if ratio <= 0:
            return np.nan
        return 0.5 * Te * np.log(ratio)

    def ion_thermal_velocity(self):
        """
        离子热速度:
            v_th,i = sqrt(2 * k_B * T_i / m_i)
        """
        m_p = 1.67262192369e-27
        Ti = self._params['T_i']
        mi_kg = self._params['m_i'] * m_p
        e_charge = 1.602176634e-19
        if mi_kg <= 0 or Ti <= 0:
            return np.nan
        return np.sqrt(2.0 * Ti * e_charge / mi_kg)

    def plasma_frequency(self):
        """
        等离子体频率:
            omega_pe = sqrt(n_0 * e^2 / (epsilon_0 * m_e))
        """
        e_charge = 1.602176634e-19
        epsilon_0 = 8.854187817e-12
        m_e = 9.10938356e-31
        n0 = self._params['n_0']
        if n0 <= 0:
            return np.nan
        return np.sqrt(n0 * e_charge**2 / (epsilon_0 * m_e))

    def print_summary(self):
        """打印参数摘要"""
        print("=" * 60)
        print("等离子体鞘层与壁材料侵蚀模拟参数摘要")
        print("=" * 60)
        print(f"等离子体密度 n_0      = {self._params['n_0']:.3e} m^-3")
        print(f"电子温度 T_e         = {self._params['T_e']:.2f} eV")
        print(f"离子温度 T_i         = {self._params['T_i']:.2f} eV")
        print(f"德拜长度 lambda_D    = {self.debye_length():.3e} m")
        print(f"离子声速 c_s         = {self.ion_sound_speed():.3e} m/s")
        print(f"Bohm能量             = {self.bohm_criterion():.3f} eV")
        print(f"鞘层电势降 Delta_phi = {self.sheath_potential():.3f} V")
        print(f"等离子体频率 omega_pe= {self.plasma_frequency():.3e} rad/s")
        print(f"壁材料               = {self._params['wall_material']} (Z={self._params['wall_Z']})")
        print(f"溅射阈值能量         = {self._params['E_threshold']:.1f} eV")
        print("=" * 60)


# 模块级默认实例
_default_instance = None


def get_parameters(**kwargs):
    """
    获取全局参数实例（单例模式）
    基于 logistic_parameters 的 persistent 思想
    """
    global _default_instance
    if _default_instance is None or kwargs:
        _default_instance = PlasmaParameters(**kwargs)
    return _default_instance


def reset_parameters(**kwargs):
    """重置全局参数"""
    global _default_instance
    _default_instance = PlasmaParameters(**kwargs)
    return _default_instance
