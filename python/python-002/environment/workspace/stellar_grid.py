# -*- coding: utf-8 -*-
"""
stellar_grid.py
基于 1345_triangulation_plot 与 1282_tortoise 合成
恒星拉格朗日质量坐标网格的离散化、壳层边界追踪与拓扑管理。
使用离散几何编码描述壳层边界（类似 tortoise 边界词思想）。
"""

import numpy as np
from typing import List, Tuple, Optional


class StellarGrid:
    """
    恒星一维拉格朗日质量坐标网格。
    将恒星沿质量坐标 M(r) 划分为 N 个壳层（shell）。
    每个壳层对应一个质量元 Δm_i = M_{i+1/2} - M_{i-1/2}。
    
    边界词编码（基于 1282_tortoise 的离散几何思想）：
      用符号序列表示壳层边界在参数空间中的拓扑路径。
      每个符号对应一个方向增量：
        'A': +Δm (外边界)
        'a': -Δm (内边界)
        'B': +Δr (半径增大)
        'b': -Δr (半径减小)
    """

    def __init__(self, M_total: float, N_shells: int = 200,
                 core_fraction: float = 0.05, envelope_fraction: float = 0.95):
        """
        初始化恒星网格。
        
        Parameters
        ----------
        M_total : float
            恒星总质量 [g]
        N_shells : int
            壳层数量
        core_fraction : float
            核心区域质量占比（需要更密网格）
        envelope_fraction : float
            包层质量占比（需要更密网格）
        """
        if M_total <= 0:
            raise ValueError("恒星总质量必须为正")
        if N_shells < 10:
            raise ValueError("壳层数至少为10")
        self.M_total = M_total
        self.N_shells = N_shells
        self.core_fraction = max(0.0, min(core_fraction, 1.0))
        self.envelope_fraction = max(0.0, min(envelope_fraction, 1.0))

        # 质量坐标：使用双曲正切映射在核心和包层加密
        # 标准坐标 s ∈ [0,1]，映射到质量坐标 m = M_total * s
        # 加密映射: s = 0.5 * (tanh(α*(u-uc))/tanh(α*uc) + 1)
        self.mass = self._generate_mass_grid()
        self.dm = np.diff(self.mass)
        self.dm = np.append(self.dm, self.dm[-1])  # 边界外推
        # 确保质量守恒
        self.mass = np.clip(self.mass, 0.0, M_total)

        # 网格变量（将在演化中更新）
        self.radius = np.zeros(N_shells, dtype=np.float64)
        self.density = np.zeros(N_shells, dtype=np.float64)
        self.temperature = np.zeros(N_shells, dtype=np.float64)
        self.pressure = np.zeros(N_shells, dtype=np.float64)
        self.luminosity = np.zeros(N_shells, dtype=np.float64)

        # 壳层边界词（拓扑描述）
        self.boundary_words: List[str] = []
        self._generate_boundary_words()

    def _generate_mass_grid(self) -> np.ndarray:
        """
        生成非均匀质量网格。
        使用坐标变换在核心（m/M ≈ 0）和包层（m/M ≈ 1）加密。
        """
        u = np.linspace(0.0, 1.0, self.N_shells)
        # 双曲正切加密参数
        alpha = 3.0
        uc = 0.5
        # 对称加密映射
        tanh_alpha = np.tanh(alpha * uc)
        if tanh_alpha > 1e-10:
            s = 0.5 * (np.tanh(alpha * (u - uc)) / tanh_alpha + 1.0)
        else:
            s = u
        s = np.clip(s, 0.0, 1.0)
        return s * self.M_total

    def _generate_boundary_words(self):
        """
        基于 tortoise 边界词思想，为每个壳层生成拓扑边界描述。
        符号含义：
          'A'/'a' : 质量边界外移/内移
          'B'/'b' : 半径边界外扩/内缩
          'C'/'c' : 温度边界升高/降低
        """
        self.boundary_words = []
        for i in range(self.N_shells):
            word = []
            if i == 0:
                word.append('A')  # 中心边界
            else:
                dm = self.mass[i] - self.mass[i - 1]
                if dm > 0:
                    word.append('A')
                else:
                    word.append('a')
            if i == self.N_shells - 1:
                word.append('B')  # 表面边界
            else:
                word.append('b')
            self.boundary_words.append(''.join(word))

    def get_shell_index(self, mass_coord: float) -> int:
        """给定质量坐标，返回所属壳层索引。"""
        if mass_coord <= 0:
            return 0
        if mass_coord >= self.M_total:
            return self.N_shells - 1
        idx = np.searchsorted(self.mass, mass_coord)
        return min(idx, self.N_shells - 1)

    def get_core_shells(self) -> slice:
        """返回核心区域壳层索引。"""
        m_core = self.core_fraction * self.M_total
        end = np.searchsorted(self.mass, m_core)
        return slice(0, min(end + 1, self.N_shells))

    def get_envelope_shells(self) -> slice:
        """返回包层区域壳层索引。"""
        m_env = (1.0 - self.envelope_fraction) * self.M_total
        start = np.searchsorted(self.mass, m_env)
        return slice(max(start, 0), self.N_shells)

    def get_radiative_zone(self, convection_mask: np.ndarray) -> np.ndarray:
        """
        根据对流掩码确定辐射区索引。
        radiation_zone = ~convection_mask
        """
        conv = np.asarray(convection_mask, dtype=bool)
        return np.where(~conv)[0]

    def shell_mass(self, i: int) -> float:
        """第 i 个壳层的质量。"""
        if i < 0 or i >= self.N_shells:
            raise IndexError("壳层索引越界")
        if i == 0:
            return self.mass[1] - self.mass[0] if self.N_shells > 1 else self.mass[0]
        elif i == self.N_shells - 1:
            return self.mass[-1] - self.mass[-2] if self.N_shells > 1 else self.mass[-1]
        else:
            return 0.5 * (self.mass[i + 1] - self.mass[i - 1])

    def remap_grid(self, new_mass: np.ndarray):
        """
        重新映射到新质量坐标（用于自适应网格细化 AMR）。
        使用线性插值将所有场量映射到新网格。
        """
        new_mass = np.asarray(new_mass, dtype=np.float64)
        new_mass = np.clip(new_mass, 0.0, self.M_total)
        new_N = len(new_mass)

        old_mass = self.mass.copy()
        old_radius = self.radius.copy()
        old_density = self.density.copy()
        old_temperature = self.temperature.copy()
        old_pressure = self.pressure.copy()
        old_luminosity = self.luminosity.copy()

        self.N_shells = new_N
        self.mass = new_mass
        self.dm = np.diff(new_mass)
        self.dm = np.append(self.dm, self.dm[-1])

        # 线性插值（对数域插值更稳定）
        def safe_interp(x_old, y_old, x_new):
            y_new = np.interp(x_new, x_old, y_old)
            return y_new

        self.radius = safe_interp(old_mass, old_radius, new_mass)
        self.density = safe_interp(old_mass, old_density, new_mass)
        self.temperature = safe_interp(old_mass, old_temperature, new_mass)
        self.pressure = safe_interp(old_mass, old_pressure, new_mass)
        self.luminosity = safe_interp(old_mass, old_luminosity, new_mass)

        # 确保物理边界条件
        self.radius = np.maximum(self.radius, 0.0)
        self.density = np.maximum(self.density, 1e-10)
        self.temperature = np.maximum(self.temperature, 1e3)
        self.pressure = np.maximum(self.pressure, 1e-5)
        self.luminosity = np.maximum(self.luminosity, 0.0)

        self._generate_boundary_words()

    def compute_volumes(self) -> np.ndarray:
        """
        计算每个壳层的体积 [cm^3]。
        假设球对称：V_i = 4/3 π (r_{i+1/2}^3 - r_{i-1/2}^3)
        """
        r = self.radius
        # 界面半径
        r_interface = np.zeros(self.N_shells + 1, dtype=np.float64)
        r_interface[0] = 0.0
        for i in range(1, self.N_shells):
            r_interface[i] = 0.5 * (r[i - 1] + r[i])
        r_interface[-1] = r[-1] if self.N_shells > 1 else r[0]
        volumes = (4.0 / 3.0) * np.pi * (r_interface[1:] ** 3 - r_interface[:-1] ** 3)
        return np.maximum(volumes, 1e-30)

    def to_mass_coordinates(self, f_r: np.ndarray) -> np.ndarray:
        """
        将半径坐标的场转换到质量坐标。
        这里简单用线性插值（因为内部存储已经是按质量坐标排列的）。
        """
        return np.asarray(f_r, dtype=np.float64)
