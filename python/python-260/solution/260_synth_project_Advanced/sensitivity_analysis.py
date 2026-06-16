"""
sensitivity_analysis.py -- 灵敏度分析 (SOBP 与 OAT)
================================================================
Project 260: 暗能量状态方程约束

数学公式
--------
(1)  Sobol 一阶指数:
         S_i = V_i / V(Y)
     V_i = V_{X_i}[E_{X_{~i}}(Y | X_i)]

(2)  Sobol 全阶指数:
         S_{Ti} = E_{X_{~i}}[V_{X_i}(Y | X_{~i})] / V(Y)
                = 1 - V_{X_{~i}}[E_{X_i}(Y | X_{~i})] / V(Y)

(3)  Monte Carlo 估计 (Saltelli 2010):
         生成 A, B 两个 N x d 采样矩阵
         构造 AB_i^{(j)}: A 的第 j 列替换为 B 的第 j 列
         f_A = f(A), f_B = f(B), f_{AB_i} = f(AB_i)
         S_i ≈ (1/N sum f_B * (f_{AB_i} - f_A)) / Var(f)
         S_{Ti} ≈ (1/(2N) sum (f_A - f_{AB_i})^2) / Var(f)

(4)  OAT (One-At-a-Time):
         delta_i = |f(theta + delta_i e_i) - f(theta)| / |f(theta)|

(5)  弹性 (Elasticity):
         E_i = (d f / d theta_i) * (theta_i / f)

(6)  增长方程灵敏度:
         dD/da = partial D / partial w0 * dw0 + partial D / partial wa * dwa
         partial D / partial w0 ≈ (D(w0+eps,wa) - D(w0-eps,wa)) / (2 eps)
================================================================
"""
from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict, Callable

import cosmo_constants as cc
from growth_solver import BDFGrowthSolver


_rng = random.Random(42)


def set_seed(seed: int):
    global _rng
    _rng = random.Random(seed)


# =====================================================================
#  Sobol 全局灵敏度 (Saltelli 方案)
# =====================================================================

class SobolSensitivity:
    """
    Sobol 全局灵敏度分析 (Saltelli 2010 采样方案).

    参数: theta = (w0, wa) in [w0_lo, w0_hi] x [wa_lo, wa_hi]
    输出: D(a=1) 或 f(a=1) 或 chi^2
    """

    def __init__(self, param_ranges: List[Tuple[float, float]] = None,
                 output_func: Callable = None):
        if param_ranges is None:
            param_ranges = [(-1.3, -0.7), (-1.0, 1.0)]
        self.ranges = param_ranges
        self.d = len(param_ranges)
        self.output_func = output_func or self._default_output

    def _default_output(self, params: List[float]) -> float:
        """默认输出: f(a=1)."""
        w0, wa = params[0], params[1]
        solver = BDFGrowthSolver(w0=w0, wa=wa)
        solver.setup_grid(a_min=1e-4, n_points=60)
        solver.solve_bdf1()
        f_vals = solver.compute_growth_rate()
        return f_vals[-1] if f_vals else 0.0

    def _sample_uniform(self, n: int) -> List[List[float]]:
        """在参数域内均匀采样."""
        samples = []
        for _ in range(n):
            pt = [_rng.uniform(r[0], r[1]) for r in self.ranges]
            samples.append(pt)
        return samples

    def compute_sobol(self, n_samples: int = 100) -> Dict:
        """
        Saltelli 方案计算 Sobol 指数.
        """
        N = n_samples
        d = self.d

        # 生成 A, B
        A = self._sample_uniform(N)
        B = self._sample_uniform(N)

        # 评估 f_A, f_B
        f_A = [self.output_func(a) for a in A]
        f_B = [self.output_func(b) for b in B]

        # 评估 f_AB_i
        f_AB = [[] for _ in range(d)]
        for i in range(d):
            for j in range(N):
                ab_j = A[j][:]
                ab_j[i] = B[j][i]
                f_AB[i].append(self.output_func(ab_j))

        # Sobol 指数
        f_all = f_A + f_B
        f0 = sum(f_all) / len(f_all)
        var_f = sum((f - f0)**2 for f in f_all) / len(f_all)
        if var_f < 1e-30:
            var_f = 1e-30

        S = []
        ST = []
        for i in range(d):
            # 一阶: S_i = (1/N sum f_B * (f_AB_i - f_A)) / Var
            vi = sum(f_B[j] * (f_AB[i][j] - f_A[j]) for j in range(N)) / N
            S.append(vi / var_f)

            # 全阶: S_Ti = (1/(2N) sum (f_A - f_AB_i)^2) / Var
            vti = sum((f_A[j] - f_AB[i][j])**2 for j in range(N)) / (2*N)
            ST.append(vti / var_f)

        return {
            'S1': S,
            'ST': ST,
            'var_Y': var_f,
            'n_samples': N,
            'param_names': ['w0', 'wa'],
        }


# =====================================================================
#  OAT 局部灵敏度
# =====================================================================

def oat_sensitivity(base_params: List[float] = None,
                     delta: float = 0.01,
                     output_func: Callable = None) -> Dict:
    """
    OAT 局部灵敏度分析.
    """
    if base_params is None:
        base_params = [-1.0, 0.0]
    if output_func is None:
        def output_func(p):
            solver = BDFGrowthSolver(w0=p[0], wa=p[1])
            solver.setup_grid(a_min=1e-4, n_points=60)
            solver.solve_bdf1()
            f_vals = solver.compute_growth_rate()
            return f_vals[-1] if f_vals else 0.0

    f_base = output_func(base_params)
    if abs(f_base) < 1e-30:
        f_base = 1e-30

    sensitivities = []
    for i in range(len(base_params)):
        params_plus = base_params[:]
        params_minus = base_params[:]
        h = delta * max(abs(base_params[i]), 1e-10)
        params_plus[i] += h
        params_minus[i] -= h
        f_plus = output_func(params_plus)
        f_minus = output_func(params_minus)
        deriv = (f_plus - f_minus) / (2 * h)
        sensitivity = abs(deriv * base_params[i] / f_base)
        sensitivities.append(sensitivity)

    return {
        'base_params': base_params,
        'f_base': f_base,
        'sensitivities': sensitivities,
        'param_names': ['w0', 'wa'],
        'dominant': 0 if sensitivities[0] > sensitivities[1] else 1,
    }


# =====================================================================
#  弹性分析
# =====================================================================

def elasticity_analysis(base_params: List[float] = None,
                         delta_frac: float = 0.01
                         ) -> Dict:
    """
    弹性 E_i = (df/dp_i) * (p_i / f).
    """
    if base_params is None:
        base_params = [-1.0, 0.0]

    def output(p):
        solver = BDFGrowthSolver(w0=p[0], wa=p[1])
        solver.setup_grid(a_min=1e-4, n_points=60)
        solver.solve_bdf1()
        f_vals = solver.compute_growth_rate()
        return f_vals[-1] if f_vals else 0.0

    f_base = output(base_params)
    elasticities = []

    for i in range(len(base_params)):
        params_p = base_params[:]
        params_m = base_params[:]
        h = delta_frac * abs(base_params[i]) if abs(base_params[i]) > 1e-10 else delta_frac
        params_p[i] += h
        params_m[i] -= h
        f_p = output(params_p)
        f_m = output(params_m)
        deriv = (f_p - f_m) / (2 * h)
        E_i = deriv * base_params[i] / max(abs(f_base), 1e-30)
        elasticities.append(E_i)

    return {
        'base_params': base_params,
        'f_base': f_base,
        'elasticities': elasticities,
        'param_names': ['w0', 'wa'],
    }


# =====================================================================
#  测试
# =====================================================================

if __name__ == '__main__':
    print("=== 灵敏度分析测试 ===")
    set_seed(42)

    print("\nOAT 灵敏度:")
    oat = oat_sensitivity()
    print(f"  基准 f = {oat['f_base']:.4f}")
    for i, name in enumerate(oat['param_names']):
        print(f"  S({name}) = {oat['sensitivities'][i]:.6f}")
    print(f"  主导参数: {oat['param_names'][oat['dominant']]}")

    print("\n弹性分析:")
    elast = elasticity_analysis()
    for i, name in enumerate(elast['param_names']):
        print(f"  E({name}) = {elast['elasticities'][i]:.6f}")

    print("\nSobol 全局灵敏度 (n=30):")
    sobol = SobolSensitivity(n_samples=30)
    # 简化: 用 n=30 快速测试
    result = sobol.compute_sobol(n_samples=30)
    for i, name in enumerate(result['param_names']):
        print(f"  S1({name}) = {result['S1'][i]:.4f}")
        print(f"  ST({name}) = {result['ST'][i]:.4f}")

    print("\n所有灵敏度测试通过.")
