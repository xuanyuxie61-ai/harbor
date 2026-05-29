# -*- coding: utf-8 -*-
"""
transport_solver.py
裂隙介质示踪剂对流-弥散方程求解模块

基于种子项目 1061_schroedinger_nonlinear_pde 的守恒量监测思想，
用于求解裂隙网络中的示踪剂迁移方程，并严格监测质量守恒。

控制方程（二维对流-弥散方程，ADE）：
    R ∂C/∂t + v·∇C = ∇·(D·∇C) - λ C + S

其中：
    C = C(x, y, t)      示踪剂浓度 [kg/m³]
    R                  滞留因子（retardation factor）[-]
    v = (v_x, v_y)      达西流速 [m/s]
    D = D_m + α_L |v|   水动力弥散系数 [m²/s]
    λ                  一阶衰减速率 [1/s]
    S                  源汇项 [kg/(m³·s)]

质量守恒监测（基于 schroedinger_nonlinear_conserved）：
    M(t) = ∫_Ω C(x, y, t) dΩ
    
    对于衰变示踪剂：
        dM/dt = -λ M  =>  M(t) = M(0) exp(-λ t)
    
    对于保守示踪剂（λ = 0）：
        M(t) = M(0) = const

数值离散（隐式迎风-中心差分，Crank-Nicolson）：
    (C^{n+1} - C^n)/Δt = 0.5 * (L(C^{n+1}) + L(C^n))
    
    其中 L 为空间算子：
        L(C) = -v·∇C + ∇·(D·∇C) - λ C + S/R

稳定性条件（CFL + 弥散）：
    CFL = max(|v_x|/Δx + |v_y|/Δy) * Δt < 1
    D_num = D Δt / min(Δx², Δy²) < 0.5
"""

import numpy as np
from typing import Tuple, Optional, Callable


class TransportSolver:
    """
    裂隙介质示踪剂迁移方程求解器

    采用隐式有限差分格式求解二维 ADE 方程，
    并提供严格的质量守恒监测功能。
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 dt: float, R: float = 1.0, lambda_decay: float = 0.0):
        """
        Parameters
        ----------
        nx, ny : int
            网格数
        dx, dy : float
            网格间距 [m]
        dt : float
            时间步长 [s]
        R : float
            滞留因子 [-]
        lambda_decay : float
            一阶衰减速率 [1/s]
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("nx 和 ny 必须为正")
        if dx <= 0 or dy <= 0 or dt <= 0:
            raise ValueError("dx, dy, dt 必须为正")
        if R <= 0:
            raise ValueError("R 必须为正")
        if lambda_decay < 0:
            raise ValueError("lambda_decay 必须为非负")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.R = R
        self.lambda_decay = lambda_decay

        self.concentration = np.zeros((ny, nx))
        self.source = np.zeros((ny, nx))

        # 质量守恒监测历史
        self.mass_history = []
        self.time_history = []

    def set_velocity_field(self, vx: np.ndarray, vy: np.ndarray):
        """
        设置流速场

        Parameters
        ----------
        vx, vy : np.ndarray
            流速分量 (ny, nx)
        """
        if vx.shape != (self.ny, self.nx) or vy.shape != (self.ny, self.nx):
            raise ValueError("流速场形状必须与网格匹配")
        self.vx = vx
        self.vy = vy

    def set_dispersivity(self, alpha_L: float, alpha_T: float,
                         D_m: float = 1.0e-9):
        """
        设置弥散参数

        Parameters
        ----------
        alpha_L : float
            纵向弥散度 [m]
        alpha_T : float
            横向弥散度 [m]
        D_m : float
            分子扩散系数 [m²/s]
        """
        if alpha_L < 0 or alpha_T < 0 or D_m < 0:
            raise ValueError("弥散参数必须为非负")

        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.D_m = D_m

        # 计算弥散系数场
        v_mag = np.sqrt(self.vx ** 2 + self.vy ** 2)
        self.Dxx = D_m + alpha_L * v_mag
        self.Dyy = D_m + alpha_T * v_mag
        self.Dxy = np.zeros_like(v_mag)  # 简化：各向异性主轴与坐标轴对齐

    def _compute_flux_x(self, C: np.ndarray, i: int, j: int) -> float:
        """计算 x 方向净通量"""
        if j <= 0 or j >= self.nx - 1:
            return 0.0

        # 对流项（迎风差分）
        vx_ij = self.vx[i, j]
        if vx_ij > 0:
            adv = vx_ij * (C[i, j] - C[i, j-1]) / self.dx
        else:
            adv = vx_ij * (C[i, j+1] - C[i, j]) / self.dx

        # 弥散项（中心差分）
        D_face = 0.5 * (self.Dxx[i, j] + self.Dxx[i, j+1])
        diff_p = D_face * (C[i, j+1] - C[i, j]) / self.dx**2

        D_face_m = 0.5 * (self.Dxx[i, j] + self.Dxx[i, j-1])
        diff_m = D_face_m * (C[i, j] - C[i, j-1]) / self.dx**2

        return -adv + diff_p - diff_m

    def _compute_flux_y(self, C: np.ndarray, i: int, j: int) -> float:
        """计算 y 方向净通量"""
        if i <= 0 or i >= self.ny - 1:
            return 0.0

        vy_ij = self.vy[i, j]
        if vy_ij > 0:
            adv = vy_ij * (C[i, j] - C[i-1, j]) / self.dy
        else:
            adv = vy_ij * (C[i+1, j] - C[i, j]) / self.dy

        D_face = 0.5 * (self.Dyy[i, j] + self.Dyy[i+1, j])
        diff_p = D_face * (C[i+1, j] - C[i, j]) / self.dy**2

        D_face_m = 0.5 * (self.Dyy[i, j] + self.Dyy[i-1, j])
        diff_m = D_face_m * (C[i, j] - C[i-1, j]) / self.dy**2

        return -adv + diff_p - diff_m

    def explicit_step(self) -> np.ndarray:
        """
        显式时间推进一步

        Returns
        -------
        np.ndarray
            更新后的浓度场
        """
        # TODO: 实现显式时间推进
        pass

    def solve(self, n_steps: int, injection_zone: Optional[Tuple] = None,
              C_inject: float = 1.0, check_mass: bool = True) -> dict:
        """
        求解示踪剂迁移方程

        Parameters
        ----------
        n_steps : int
            时间步数
        injection_zone : tuple, optional
            注入区域 (i_min, i_max, j_min, j_max)
        C_inject : float
            注入浓度
        check_mass : bool
            是否监测质量守恒

        Returns
        -------
        dict
            求解结果
        """
        if n_steps <= 0:
            raise ValueError("n_steps 必须为正")

        if injection_zone is not None:
            i_min, i_max, j_min, j_max = injection_zone
            self.concentration[i_min:i_max, j_min:j_max] = C_inject

        self.mass_history = []
        self.time_history = []

        for step in range(n_steps):
            self.explicit_step()

            if check_mass:
                mass = self.compute_total_mass()
                self.mass_history.append(mass)
                self.time_history.append((step + 1) * self.dt)

        result = {
            'concentration': self.concentration.copy(),
            'final_mass': self.compute_total_mass(),
            'mass_history': np.array(self.mass_history),
            'time_history': np.array(self.time_history)
        }

        if check_mass and len(self.mass_history) > 0:
            result['mass_conservation_error'] = self._mass_conservation_error()

        return result

    def compute_total_mass(self) -> float:
        """
        计算域内总示踪剂质量

        公式：
            M = ∫_Ω C(x, y) dΩ ≈ Σ C_{i,j} Δx Δy
        """
        return float(np.sum(self.concentration) * self.dx * self.dy)

    def _mass_conservation_error(self) -> float:
        """
        计算质量守恒相对误差

        对于衰变示踪剂：
            M(t) = M(0) exp(-λ t)
        
        对于保守示踪剂：
            M(t) = M(0)
        """
        if len(self.mass_history) == 0:
            return 0.0

        M0 = self.mass_history[0] if len(self.mass_history) > 0 else 1.0
        if abs(M0) < 1e-20:
            return 0.0

        t_final = self.time_history[-1]
        if self.lambda_decay > 0:
            M_expected = M0 * np.exp(-self.lambda_decay * t_final)
        else:
            M_expected = M0

        M_actual = self.mass_history[-1]
        rel_error = abs(M_actual - M_expected) / abs(M_expected) if abs(M_expected) > 1e-20 else 0.0
        return float(rel_error)

    def breakthrough_curve(self, outlet_zone: Tuple,
                            n_steps: int, injection_zone: Tuple,
                            C_inject: float = 1.0) -> dict:
        """
        计算出口突破曲线

        Parameters
        ----------
        outlet_zone : tuple
            出口监测区域 (i_min, i_max, j_min, j_max)
        n_steps : int
            时间步数
        injection_zone : tuple
            注入区域
        C_inject : float
            注入浓度

        Returns
        -------
        dict
            突破曲线数据
        """
        times = []
        concentrations = []
        masses = []

        # 初始注入
        self.concentration = np.zeros((self.ny, self.nx))
        i_min, i_max, j_min, j_max = injection_zone
        self.concentration[i_min:i_max, j_min:j_max] = C_inject

        oi_min, oi_max, oj_min, oj_max = outlet_zone

        for step in range(n_steps):
            self.explicit_step()

            # 记录出口平均浓度
            C_out = np.mean(self.concentration[oi_min:oi_max, oj_min:oj_max])
            times.append((step + 1) * self.dt)
            concentrations.append(C_out)
            masses.append(self.compute_total_mass())

        return {
            'times': np.array(times),
            'concentrations': np.array(concentrations),
            'masses': np.array(masses)
        }

    def stability_check(self) -> dict:
        """
        检查数值稳定性条件

        Returns
        -------
        dict
            CFL 数和弥散数
        """
        vx_max = np.max(np.abs(self.vx))
        vy_max = np.max(np.abs(self.vy))

        cfl_x = vx_max * self.dt / self.dx
        cfl_y = vy_max * self.dt / self.dy
        cfl = cfl_x + cfl_y

        D_max = np.max(self.Dxx)
        diff_num = D_max * self.dt / (self.dx ** 2)

        return {
            'CFL': float(cfl),
            'CFL_x': float(cfl_x),
            'CFL_y': float(cfl_y),
            'diffusion_number': float(diff_num),
            'stable': cfl < 1.0 and diff_num < 0.5
        }
