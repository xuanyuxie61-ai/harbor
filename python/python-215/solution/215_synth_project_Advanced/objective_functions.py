#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
objective_functions.py — 多目标优化目标函数定义

科学问题:
  催化反应器拓扑优化, 同时优化:
    f1: 结构柔度 (compliance) — 最小化
    f2: 热应力方差 — 最小化
    f3: 生化转化效率 (负值, 最小化)

核心公式:
  f1(ρ) = u^T K(ρ) u  (柔度)
  f2(ρ) = Var[σ_th(ρ, T(ρ))]  (热应力方差)
  f3(ρ) = -η(ρ) = -(1 - c_out/c_in)  (负转化率)

  约束:
    V(ρ) = Σ ρ_i V_i / V_total ≤ V_frac_max
    K(ρ) u = F(ρ, T)
    -∇·(κ(ρ) ∇T) = Q(ρ, T)
"""

import numpy as np
from typing import Dict, Tuple
from coupled_pde_system import (
    MaterialModel, BiochemicalKinetics,
    fem1d_assemble, solve_thermal_field,
    compute_compliance, compute_thermal_stress_variance,
    coupled_solve
)
from quadrature_engine import (
    pyramid_unit_nodes_weights, pyramid_monomial_integral,
    disk_monomial_integral, disk_sample_uniform,
    gauss_legendre_2d, integrate_2d
)
from stochastic_robustness import UncertaintyQuantifier, fidelity_witness


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------
class MultiObjectiveEvaluator:
    """
    多目标评估器.

    三个目标:
      f1: 结构柔度 C(ρ) = u^T K u
      f2: 热应力方差 Var[σ_th]
      f3: 负转化率 -(1 - c_out/c_in)
      f4: 体积分数惩罚
      f5: Fresnel 波场品质因子 (附加科学目标)

    所有目标均转为最小化.
    """

    def __init__(self,
                 n_elements: int = 20,
                 L: float = 1.0,
                 volume_fraction_max: float = 0.5,
                 fresnel_weight: float = 0.1,
                 use_robust: bool = True,
                 n_robust_samples: int = 10,
                 seed: int = 42):
        self.n_elements = n_elements
        self.L = L
        self.volume_fraction_max = volume_fraction_max
        self.fresnel_weight = fresnel_weight
        self.material = MaterialModel()
        self.kinetics = BiochemicalKinetics()
        self.use_robust = use_robust
        self.n_robust_samples = n_robust_samples

        if use_robust:
            self.uq = UncertaintyQuantifier(
                sigma_ou=0.03, theta_ou=2.0,
                beta_risk=1.5, n_mc=n_robust_samples, seed=seed
            )
        else:
            self.uq = None

        self._eval_count = 0
        self._fresnel_x_target = np.linspace(0, 2, 10)

    def evaluate_deterministic(self, rho: np.ndarray
                               ) -> Dict[str, float]:
        """
        确定性评估 — 单次 PDE 求解.

        Returns
        -------
        dict with keys: 'f1_compliance', 'f2_thermal_var',
                        'f3_neg_yield', 'f4_volume_penalty',
                        'f5_fresnel', 'volume_fraction'
        """
        rho = np.clip(rho, 0.01, 1.0)
        n = self.n_elements

        # 确保 rho 长度正确
        if len(rho) < n:
            rho = np.pad(rho, (0, n - len(rho)), constant_values=0.5)
        elif len(rho) > n:
            rho = rho[:n]

        # 1. 耦合求解
        result = coupled_solve(
            n_elements=n, L=self.L,
            material=self.material,
            kinetics=self.kinetics,
            rho=rho
        )

        # 2. 目标 f1: 结构柔度
        K, F, _ = fem1d_assemble(n, self.L, self.material, rho)
        u = np.linalg.solve(K, F)
        f1 = compute_compliance(K, u)
        # 归一化 (除以实心梁的柔度)
        K_solid, _, _ = fem1d_assemble(n, self.L, self.material,
                                       np.ones(n))
        u_solid = np.linalg.solve(K_solid, F)
        c_solid = compute_compliance(K_solid, u_solid)
        f1_normalized = f1 / max(c_solid, 1e-10)

        # 3. 目标 f2: 热应力方差
        f2 = compute_thermal_stress_variance(result['T'])
        # 归一化
        f2_ref = compute_thermal_stress_variance(
            np.linspace(300, 500, n + 1)
        )
        f2_normalized = f2 / max(f2_ref, 1e-10)

        # 4. 目标 f3: 负转化率
        c_in = result['c'][0]
        c_out = result['c'][-1]
        if c_in > 1e-15:
            yield_val = 1.0 - c_out / c_in
        else:
            yield_val = 0.0
        f3 = -yield_val  # 最小化负转化率 = 最大化转化率

        # 5. 目标 f4: 体积分数惩罚
        vol_frac = np.mean(rho)
        f4 = max(0.0, vol_frac - self.volume_fraction_max) ** 2

        # 6. 目标 f5: Fresnel 品质因子
        # 将密度分布映射为相位分布, 评估聚焦质量
        f5 = self._fresnel_objective(rho)

        self._eval_count += 1

        return {
            'f1_compliance': float(f1_normalized),
            'f2_thermal_var': float(f2_normalized),
            'f3_neg_yield': float(f3),
            'f4_volume_penalty': float(f4),
            'f5_fresnel': float(f5),
            'volume_fraction': float(vol_frac),
            'yield': float(yield_val)
        }

    def _fresnel_objective(self, rho: np.ndarray) -> float:
        """
        基于 Fresnel 积分的波场品质目标.

        将密度分布视为相位光栅:
            φ(x) = π · ρ(x)
        计算透过率场的 Fresnel 衍射,
        目标是最小化焦点处的强度偏差.

        品质因子:
            Q = 1 - |I_focus - I_target| / I_target
            f5 = -Q (最小化)
        """
        from cvt_design_sampler import fresnel_cos, fresnel_sin

        n = len(rho)
        x = np.linspace(0, 1, n)
        # 相位分布
        phase = np.pi * rho

        # 简化 Fresnel 衍射 (1D)
        # U(x') = ∫ A(x) exp(i φ(x)) exp(i π (x'-x)² / (λ z)) dx
        # 取 λz = 1 简化
        x_prime = self._fresnel_x_target
        intensity = np.zeros(len(x_prime))

        for j, xp in enumerate(x_prime):
            # 数值积分
            integrand = np.exp(1j * phase) * np.exp(1j * np.pi * (xp - x) ** 2)
            U = np.trapz(integrand, x)
            intensity[j] = np.abs(U) ** 2

        # 目标: 聚焦到中心 (x' = 1.0)
        center_idx = np.argmin(np.abs(x_prime - 1.0))
        I_focus = intensity[center_idx]
        I_total = np.sum(intensity) + 1e-15
        Q = I_focus / I_total  # Strehl ratio
        return float(-Q)

    def evaluate_robust(self, rho: np.ndarray
                        ) -> Dict[str, float]:
        """
        鲁棒评估 — 考虑制造不确定性的目标.

        对每个目标, 使用 Monte Carlo 采样计算:
            F_robust = μ_F + β · σ_F
        """
        if not self.use_robust or self.uq is None:
            return self.evaluate_deterministic(rho)

        rho = np.clip(rho, 0.01, 1.0)
        rho_perturbations = self.uq.generate_perturbations(
            rho, n_realizations=self.n_robust_samples
        )

        f1_samples = []
        f2_samples = []
        f3_samples = []
        for rho_p in rho_perturbations:
            res = self.evaluate_deterministic(rho_p)
            f1_samples.append(res['f1_compliance'])
            f2_samples.append(res['f2_thermal_var'])
            f3_samples.append(res['f3_neg_yield'])

        f1_rob, f1_std = self.uq.robust_objective(np.array(f1_samples))
        f2_rob, f2_std = self.uq.robust_objective(np.array(f2_samples))
        f3_rob, f3_std = self.uq.robust_objective(np.array(f3_samples))

        res_det = self.evaluate_deterministic(rho)
        res_det['f1_compliance'] = f1_rob
        res_det['f2_thermal_var'] = f2_rob
        res_det['f3_neg_yield'] = f3_rob
        res_det['f1_std'] = f1_std
        res_det['f2_std'] = f2_std
        res_det['f3_std'] = f3_std

        return res_det

    def get_objective_vector(self, rho: np.ndarray,
                             use_robust: bool = False
                             ) -> np.ndarray:
        """
        返回目标向量 (用于 NSGA-II).

        默认使用主要三目标:
            F = [f1_compliance, f2_thermal_var, f3_neg_yield]

        若 use_robust, 使用鲁棒版本.
        """
        if use_robust:
            res = self.evaluate_robust(rho)
        else:
            res = self.evaluate_deterministic(rho)

        return np.array([
            res['f1_compliance'],
            res['f2_thermal_var'],
            res['f3_neg_yield']
        ])

    @property
    def eval_count(self) -> int:
        return self._eval_count


# ---------------------------------------------------------------------------
# 分析工具 — 积分验证
# ---------------------------------------------------------------------------
def verify_quadrature_accuracy() -> Dict[str, float]:
    """
    验证数值积分的精度 — 与解析解比较.

    测试:
      1. 金字塔域单项式积分
      2. 圆盘域单项式积分
      3. 2D Gauss-Legendre
    """
    from quadrature_engine import (
        pyramid_monomial_integral, disk_monomial_integral,
        integrate_2d
    )

    results = {}

    # 测试 1: 金字塔域 ∫ x^2 y^2 z^2 dV
    exact_pyramid = pyramid_monomial_integral(2, 2, 2)
    # 数值积分
    pts, wts = pyramid_unit_nodes_weights(6)
    num_pyramid = np.sum(wts * pts[:, 0] ** 2 * pts[:, 1] ** 2 * pts[:, 2] ** 2)
    results['pyramid_x2y2z2_exact'] = float(exact_pyramid)
    results['pyramid_x2y2z2_numeric'] = float(num_pyramid)
    results['pyramid_rel_error'] = float(abs(num_pyramid - exact_pyramid)
                                         / max(abs(exact_pyramid), 1e-15))

    # 测试 2: 圆盘域 ∫ x^2 y^2 dA (R=1)
    exact_disk = disk_monomial_integral(2, 2, R=1.0)
    num_disk = integrate_2d(lambda x, y: x ** 2 * y ** 2,
                            domain='disk', order=10, R=1.0)
    results['disk_x2y2_exact'] = float(exact_disk)
    results['disk_x2y2_numeric'] = float(num_disk)
    results['disk_rel_error'] = float(abs(num_disk - exact_disk)
                                      / max(abs(exact_disk), 1e-15))

    # 测试 3: 矩形域 ∫∫ exp(x+y) dx dy, [-1,1]²
    # 精确: (e - 1/e)² ≈ 5.3142...
    exact_rect = (np.e - 1 / np.e) ** 2
    num_rect = integrate_2d(lambda x, y: np.exp(x + y),
                            domain='rect', order=8, R=1.0)
    results['rect_expy_exact'] = float(exact_rect)
    results['rect_exp_numeric'] = float(num_rect)
    results['rect_rel_error'] = float(abs(num_rect - exact_rect)
                                      / abs(exact_rect))

    return results
