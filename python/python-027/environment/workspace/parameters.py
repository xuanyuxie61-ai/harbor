# -*- coding: utf-8 -*-

import numpy as np


class PlasmaParameters:


    _defaults = {

        'n_0': 1.0e19,
        'T_e': 50.0,
        'T_i': 50.0,
        'B_t': 5.3,
        'm_i': 2.0,
        'Z_i': 1,


        'L_sheath': 5.0e-3,
        'gamma_sheath': 7.0,


        'wall_material': 'W',
        'wall_Z': 74,
        'wall_M': 183.84,
        'wall_density': 19300.0,
        'E_bind': 8.68,
        'E_threshold': 200.0,


        'nx': 256,
        'x_max': 0.01,
        'max_iter': 10000,
        'tol': 1.0e-10,
        'dt': 1.0e-12,
        't_stop': 1.0e-6,


        'n_mc_samples': 10000,
        'rand_seed': 314159265,
    }

    def __init__(self, **kwargs):
        self._params = dict(self._defaults)
        for key, value in kwargs.items():
            if key in self._params:
                self._params[key] = value
            else:
                raise ValueError(f"未知参数: {key}")
        self._validate()

    def _validate(self):
        p = self._params

        if p['n_0'] <= 0:
            raise ValueError("等离子体密度 n_0 必须为正")

        if p['T_e'] <= 0 or p['T_i'] <= 0:
            raise ValueError("温度必须为正")

        if p['L_sheath'] <= 0:
            raise ValueError("鞘层长度必须为正")

        if p['nx'] < 4:
            raise ValueError("空间网格数 nx 至少为 4")
        if p['nx'] > 10000:
            raise ValueError("空间网格数 nx 过大，可能导致内存问题")

        if p['dt'] <= 0 or p['t_stop'] <= 0:
            raise ValueError("时间参数必须为正")
        if p['dt'] > p['t_stop']:
            raise ValueError("时间步长 dt 不能大于终止时间 t_stop")

        if p['tol'] <= 0 or p['tol'] >= 1.0:
            raise ValueError("收敛容差 tol 必须在 (0, 1) 区间")

    def get(self, key):
        if key not in self._params:
            raise KeyError(f"参数 {key} 不存在")
        return self._params[key]

    def set(self, key, value):
        if key not in self._params:
            raise KeyError(f"参数 {key} 不存在")
        self._params[key] = value
        self._validate()

    def get_all(self):
        return dict(self._params)



    def debye_length(self):
        epsilon_0 = 8.854187817e-12
        e_charge = 1.602176634e-19
        n0 = self._params['n_0']
        Te = self._params['T_e']
        if n0 <= 0 or Te <= 0:
            return np.nan
        return np.sqrt(epsilon_0 * Te / (e_charge * n0))

    def ion_sound_speed(self):
        m_p = 1.67262192369e-27
        Zi = self._params['Z_i']
        Te = self._params['T_e']
        Ti = self._params['T_i']
        mi_amu = self._params['m_i']
        mi_kg = mi_amu * m_p
        e_charge = 1.602176634e-19
        if mi_kg <= 0:
            return np.nan
        return np.sqrt((Zi * Te + Ti) * e_charge / mi_kg)

    def bohm_criterion(self):
        cs = self.ion_sound_speed()
        m_p = 1.67262192369e-27
        mi_kg = self._params['m_i'] * m_p
        e_charge = 1.602176634e-19
        if cs <= 0 or mi_kg <= 0:
            return np.nan
        E_bohm = 0.5 * mi_kg * cs**2 / e_charge
        return E_bohm

    def sheath_potential(self):
        m_e = 9.10938356e-31
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
        m_p = 1.67262192369e-27
        Ti = self._params['T_i']
        mi_kg = self._params['m_i'] * m_p
        e_charge = 1.602176634e-19
        if mi_kg <= 0 or Ti <= 0:
            return np.nan
        return np.sqrt(2.0 * Ti * e_charge / mi_kg)

    def plasma_frequency(self):
        e_charge = 1.602176634e-19
        epsilon_0 = 8.854187817e-12
        m_e = 9.10938356e-31
        n0 = self._params['n_0']
        if n0 <= 0:
            return np.nan
        return np.sqrt(n0 * e_charge**2 / (epsilon_0 * m_e))

    def print_summary(self):
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



_default_instance = None


def get_parameters(**kwargs):
    global _default_instance
    if _default_instance is None or kwargs:
        _default_instance = PlasmaParameters(**kwargs)
    return _default_instance


def reset_parameters(**kwargs):
    global _default_instance
    _default_instance = PlasmaParameters(**kwargs)
    return _default_instance
