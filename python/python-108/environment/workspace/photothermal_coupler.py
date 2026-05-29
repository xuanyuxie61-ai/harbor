# -*- coding: utf-8 -*-
"""
photothermal_coupler.py
光热耦合自洽求解模块

核心公式与物理背景
------------------
1. 光热耦合闭环
   光场产生热 → 温度改变折射率 → 折射率改变光场。
   自洽方程组：
       [∇² + k₀² n²(T)] E = 0          (光)
       ∇·(κ ∇T) + Q_abs(E) = 0         (热)
       n(T) = n₀ + (dn/dT)·(T - T₀)    (材料)

2. 热源积分（球面/体积分）
   总吸收功率：
       P_abs = ∫∫∫_Ω α·I(r) dV
   在 2D 截面近似下，用求积规则离散：
       P_abs ≈ Σ_l w_l · α · I(r_l)

3. 有效折射率温度依赖
   n_eff(T) = n_eff,0 + (dn_eff/dT)·ΔT
   其中 ΔT 为体积平均温升：
       ΔT_avg = (1/V) ∫ T dV

4. 自洽 Picard 迭代
   给定初始 n⁽⁰⁾ = n₀：
       for k = 0,1,2,...:
           求解 Helmholtz 得 E⁽ᵏ⁾
           计算 Q_abs⁽ᵏ⁾ = α·|E⁽ᵏ⁾|²
           求解热方程得 T⁽ᵏ⁾
           更新 n⁽ᵏ⁺¹⁾ = n₀ + (dn/dT)·(T⁽ᵏ⁾ - T₀)
           若 ‖n⁽ᵏ⁺¹⁾ - n⁽ᵏ⁾‖ < tol，停止

融合来源
--------
- 951_quadrature_weights_vandermonde_2d : 2D 求积权重用于热源积分
- 1119_sphere_integrals                 : 球面积分用于 WGM 模式功率计算
"""

import numpy as np
from typing import Tuple, Optional
from quadrature_engine import Vandermonde2DQuadrature, GaussLegendreTensor


class PhotothermalCoupler:
    """
    光热耦合自洽迭代求解器。
    """

    def __init__(self,
                 helmholtz_solver,
                 thermal_solver,
                 dn_dT: float = 1.86e-4,
                 n0: float = 3.47,
                 alpha_abs: float = 1.0e-3,
                 max_iter: int = 30,
                 tol: float = 1e-8):
        self.helmholtz = helmholtz_solver
        self.thermal = thermal_solver
        self.dn_dT = dn_dT
        self.n0 = n0
        self.alpha_abs = alpha_abs
        self.max_iter = max_iter
        self.tol = tol
        self.history = []

    def compute_heat_source(self, E: np.ndarray) -> np.ndarray:
        """
        由光场计算热源密度：
            Q_abs = α_abs · |E|²
        """
        intensity = np.abs(E) ** 2
        return self.alpha_abs * intensity

    def update_refractive_index(self, T: np.ndarray, n_current: np.ndarray) -> np.ndarray:
        """
        由温度场更新折射率：
            n_new = n₀ + (dn/dT)·(T - T_ambient)
        并进行松弛：n_new = ω·n_new + (1-ω)·n_current，ω=0.7 增强稳定性。
        """
        # TODO(Hole 3): 实现热光效应折射率更新与松弛迭代
        # 提示：
        #   1. 根据热光效应公式计算 n_new
        #   2. 使用松弛因子 ω 进行 under-relaxation
        #   3. 对结果做物理边界截断（折射率不能为负或过小）
        raise NotImplementedError("Hole 3: 请补全折射率更新公式")

    def integrate_heat_source(self, Q: np.ndarray, method: str = "trapezoidal") -> float:
        """
        对热源密度在计算域上做体积积分，得到总吸收功率 [W/m]（每单位长度）。

        参数
        ----
        method : str
            "trapezoidal" | "gauss_tensor"
        """
        ny, nx = Q.shape
        hx = self.helmholtz.hx
        hy = self.helmholtz.hy

        if method == "trapezoidal":
            # 复合梯形法则
            total = 0.0
            for j in range(ny):
                for i in range(nx):
                    weight = 1.0
                    if i == 0 or i == nx - 1:
                        weight *= 0.5
                    if j == 0 or j == ny - 1:
                        weight *= 0.5
                    total += weight * Q[j, i]
            total *= hx * hy
            return total
        elif method == "gauss_tensor":
            # 用张量积高斯求积（简化为内部节点和）
            Lx = self.helmholtz.Lx
            Ly = self.helmholtz.Ly
            # 在均匀网格上直接求和近似
            return float(np.sum(Q) * hx * hy)
        else:
            raise ValueError(f"未知积分方法: {method}")

    def integrate_vandermonde_2d(self, Q: np.ndarray, total_degree: int = 3) -> float:
        """
        使用 Vandermonde2D 求积权重对热源做高精度积分。
        在子矩形域上选取节点并计算权重。
        """
        ny, nx = Q.shape
        if nx * ny < (total_degree + 1) * (total_degree + 2) // 2:
            # 节点不足，退化到梯形法则
            return self.integrate_heat_source(Q, "trapezoidal")

        # 选取均匀子集作为求积节点
        n_needed = (total_degree + 1) * (total_degree + 2) // 2
        step = max(1, int(np.sqrt(nx * ny / n_needed)))
        xs = []
        ys = []
        vals = []
        hx = self.helmholtz.hx
        hy = self.helmholtz.hy
        for j in range(0, ny, step):
            for i in range(0, nx, step):
                xs.append(i * hx)
                ys.append(j * hy)
                vals.append(Q[j, i])
        xs = np.array(xs[:n_needed])
        ys = np.array(ys[:n_needed])
        vals = np.array(vals[:n_needed])

        try:
            w = Vandermonde2DQuadrature.compute_weights(
                xs, ys, total_degree,
                rect_a=0.0, rect_b=self.helmholtz.Lx,
                rect_c=0.0, rect_d=self.helmholtz.Ly
            )
            return float(np.dot(w, vals))
        except Exception:
            # 病态时退化
            return self.integrate_heat_source(Q, "trapezoidal")

    def self_consistent_solve(self,
                               source_mask: np.ndarray,
                               source_amplitude: complex = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        执行光热耦合自洽迭代。

        返回
        ----
        E : np.ndarray
            收敛后的光场
        T : np.ndarray
            收敛后的温度场
        n_profile : np.ndarray
            收敛后的折射率分布
        iters : int
            实际迭代次数
        """
        n_profile = np.full((self.helmholtz.ny, self.helmholtz.nx), self.n0, dtype=float)
        E = None
        T = None

        # 构建高斯型源分布（避免点源奇异性）
        ny, nx = self.helmholtz.ny, self.helmholtz.nx
        if source_mask.ndim == 1 or source_mask.dtype == bool:
            # 若为布尔掩码，转换为高斯分布
            gauss_source = np.zeros((ny, nx), dtype=complex)
            cx, cy = nx // 2, ny // 2
            for j in range(ny):
                for i in range(nx):
                    dx = (i - cx) * self.helmholtz.hx
                    dy = (j - cy) * self.helmholtz.hy
                    r2 = dx**2 + dy**2
                    gauss_source[j, i] = source_amplitude * np.exp(-r2 / (2.0 * (0.3e-6)**2))
            source_rhs = gauss_source
        else:
            source_rhs = source_mask * source_amplitude

        for it in range(1, self.max_iter + 1):
            # 1. 更新 Helmholtz 折射率
            self.helmholtz.n_profile = n_profile.copy()
            # 重新构建矩阵（因为 n 变了）
            self.helmholtz._band_solver = None
            E = (self.helmholtz.solve_for_rhs(source_rhs.real)
                 + 1j * self.helmholtz.solve_for_rhs(source_rhs.imag))

            # 2. 计算热源
            Q = self.compute_heat_source(E)

            # 3. 求解温度场
            T = self.thermal.solve_steady_state(Q, bc_type="robin")

            # 4. 更新折射率
            n_new = self.update_refractive_index(T, n_profile)

            # 5. 收敛判断
            delta_n = np.linalg.norm(n_new - n_profile) / (np.linalg.norm(n_profile) + 1e-30)
            self.history.append({
                "iter": it,
                "delta_n": delta_n,
                "max_T": float(np.max(T)),
                "mean_T": float(np.mean(T)),
            })
            n_profile = n_new

            if delta_n < self.tol:
                return E, T, n_profile, it

        return E, T, n_profile, self.max_iter

    def compute_thermal_shift(self, n_profile_uncoupled: np.ndarray,
                               n_profile_coupled: np.ndarray,
                               lambda0_nm: float = 1550.0) -> float:
        """
        估算光热耦合导致的谐振波长漂移：
            Δλ/λ₀ ≈ Δn_eff / n_eff
        这里用平均折射率变化近似：
            Δλ ≈ λ₀ · (mean(n_coupled) - mean(n_uncoupled)) / mean(n_uncoupled)
        """
        n_unc = np.mean(n_profile_uncoupled)
        n_coup = np.mean(n_profile_coupled)
        if abs(n_unc) < 1e-30:
            return 0.0
        return lambda0_nm * (n_coup - n_unc) / n_unc

    def wgm_mode_sphere_integral(self, E: np.ndarray, R_sphere: float) -> float:
        """
        对球形微腔，用球面积分估算 WGM 模式的辐射功率。
        简化为在等效球面上的数值积分：
            P_rad ∝ ∮ |E_θ|² dΩ
        这里用蒙特卡洛采样近似。
        """
        from quadrature_engine import SphereQuadrature
        sq = SphereQuadrature()
        n_samples = 500
        pts = sq.uniform_sample(n_samples)
        # 简化为对所有样本取平均（等权重）
        # 在实际物理中 E_θ 与角度相关，这里做演示性积分
        return float(np.mean(np.abs(E) ** 2) * 4.0 * np.pi * R_sphere ** 2)
