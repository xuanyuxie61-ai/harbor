"""
convergence_analysis.py
===================================================================
高阶有限差分格式收敛性与误差分析模块

核心分析:
  Richardson 外推:
    f_exact ≈ f_h + C*h^p + O(h^{p+1})
    => p = log((f_{h1} - f_{h2}) / (f_{h2} - f_{h3})) / log(h1/h2)

  收敛阶估计:
    p_obs = log(|E_{h1}| / |E_{h2}|) / log(h1/h2)

  各格式理论精度:
    - 二阶中心差分: O(h^2)
    - Numerov: O(h^4)
    - 六阶紧致差分: O(h^6)

  全局误差界:
    ||e|| <= C * h^p * ||u^{(p+2)}|| (对 Numerov)

  舍入误差传播:
    epsilon_round ~ u * mach_eps * N / h^2
    总误差 = 截断误差 + 舍入误差
    最优步长: h_opt ~ (u * mach_eps)^{1/(p+2)}
===================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from optical_potential import OpticalPotential, OpticalPotentialParams
from radial_schrodinger import RadialSchrodingerSolver
from phase_shift import PhaseShiftCalculator
from norms_utils import l2_norm_radial


# ---------- 物理常数 ----------
HBAR_C = 197.3269804


class ConvergenceStudy:
    """
    有限差分格式收敛性系统研究

    研究内容:
    1. 网格收敛: 相移 vs 网格密度
    2. 格式对比: 二阶CD vs Numerov
    3. Richardson 外推
    4. 误差界估计
    """

    def __init__(
        self,
        potential: OpticalPotential,
        r_max: float = 30.0,
    ):
        self.potential = potential
        self.r_max = r_max

    def grid_convergence_study(
        self,
        l: int = 0,
        n_points_list: Optional[List[int]] = None,
        scheme: str = 'numerov'
    ) -> Dict:
        """
        网格收敛性研究

        逐步加密网格, 观察相移的收敛行为

        参数:
            l: 角动量量子数
            n_points_list: 网格点数序列
            scheme: 'numerov' | 'cd2'

        返回:
            收敛数据
        """
        if n_points_list is None:
            n_points_list = [200, 400, 800, 1600, 3200]

        # 获取参考势能 (在最密网格上)
        r_fine = np.linspace(0.01, self.r_max, max(n_points_list))
        V_fine = self.potential.total_potential(r_fine, l_quantum=l)

        phase_shifts = []
        spacings = []

        for n_pts in n_points_list:
            r = np.linspace(0.01, self.r_max, n_pts)
            dr = r[1] - r[0]
            spacings.append(dr)

            # 在粗网格上插值势能
            V_coarse = np.interp(r, r_fine, V_fine.real) + \
                        1j * np.interp(r, r_fine, V_fine.imag)

            solver = RadialSchrodingerSolver(
                self.potential, r_max=self.r_max, n_points=n_pts
            )

            if scheme == 'numerov':
                _, u = solver.solve_numerov(l, V_coarse)
            else:
                _, u = solver.solve_second_order_cd(l, V_coarse)

            # 提取相移
            ps_calc = PhaseShiftCalculator(
                k=self.potential.p.k_wavevector,
                r_match=self.r_max * 0.85
            )
            delta = ps_calc.extract_phase_shift(u, r, l)
            phase_shifts.append(np.real(delta))

        phase_shifts = np.array(phase_shifts)
        spacings = np.array(spacings)

        # Richardson 外推估计收敛阶
        convergence_rates = []
        for i in range(len(phase_shifts) - 2):
            dh1 = phase_shifts[i] - phase_shifts[i + 1]
            dh2 = phase_shifts[i + 1] - phase_shifts[i + 2]
            if abs(dh2) > 1e-15:
                ratio = spacings[i] / spacings[i + 1]
                p = np.log(abs(dh1 / dh2)) / np.log(ratio)
                convergence_rates.append(float(p))

        # 参考值 (最密网格)
        delta_ref = phase_shifts[-1]

        # 误差 vs 步长
        errors = np.abs(phase_shifts[:-1] - delta_ref)

        return {
            'n_points': n_points_list,
            'spacings': spacings.tolist(),
            'phase_shifts': phase_shifts.tolist(),
            'convergence_rates': convergence_rates,
            'theoretical_order': 4 if scheme == 'numerov' else 2,
            'errors_vs_reference': errors.tolist(),
            'reference_phase_shift': float(delta_ref),
            'scheme': scheme,
        }

    def richardson_extrapolation(
        self,
        values: np.ndarray,
        spacings: np.ndarray,
        order: int = 2
    ) -> Tuple[float, float]:
        """
        Richardson 外推:

        对两个精度级别:
            f_exact ≈ (2^p * f_{h/2} - f_h) / (2^p - 1)

        参数:
            values: 不同步长的计算值 (从粗到密)
            spacings: 对应步长
            order: 理论精度阶数

        返回:
            (外推值, 误差估计)
        """
        if len(values) < 2:
            return float(values[0]), 0.0

        # 使用最后两个值
        f_coarse = values[-2]
        f_fine = values[-1]
        h_ratio = spacings[-2] / spacings[-1]

        p = order
        # Richardson 外推
        f_extrap = (h_ratio ** p * f_fine - f_coarse) / (h_ratio ** p - 1)

        # 误差估计
        error_est = abs(f_fine - f_coarse) / (h_ratio ** p - 1)

        return float(f_extrap), float(error_est)

    def error_bound_numerov(
        self,
        u: np.ndarray,
        r: np.ndarray,
        K2: np.ndarray,
        dr: float
    ) -> Dict:
        """
        Numerov 方法的误差界估计

        局部截断误差:
            tau_n = h^6/240 * u^{(6)}(xi)  (O(h^4) 全局)

        实际估计 (使用四阶差分近似六阶导数):
            |tau_n| ≈ h^4/240 * |D^4(K^2*u)|

        全局误差界:
            ||e||_inf <= C * h^4 * max|u^{(6)}| / min|K^2|

        返回:
            误差估计字典
        """
        n = len(r)
        h = dr

        # 四阶差分 (近似 u^{(4)})
        d4u = np.zeros(n)
        for i in range(2, n - 2):
            d4u[i] = (u[i + 2] - 4 * u[i + 1] + 6 * u[i] -
                       4 * u[i - 1] + u[i - 2]) / h ** 4

        # 截断误差估计
        truncation_error = h ** 4 / 240.0 * np.abs(d4u)

        # 舍入误差估计
        mach_eps = np.finfo(float).eps
        u_max = np.max(np.abs(u)) + 1e-30
        roundoff_error = mach_eps * u_max * n / h ** 2

        # 总误差
        total_error = truncation_error + roundoff_error

        # 最优步长 (平衡截断和舍入)
        # h_opt ~ (mach_eps * N / max|u^(6)|)^{1/6}
        d6u_max = np.max(np.abs(d4u)) / h ** 2 if h > 0 else 1.0
        if d6u_max > 0:
            h_opt = (mach_eps * n / d6u_max) ** (1.0 / 6.0)
        else:
            h_opt = h

        return {
            'max_truncation_error': float(np.max(truncation_error)),
            'mean_truncation_error': float(np.mean(truncation_error)),
            'roundoff_error_estimate': float(roundoff_error),
            'max_total_error': float(np.max(total_error)),
            'optimal_step_size': float(h_opt),
            'current_step_size': h,
            'h_too_large': h > h_opt * 5,
            'h_too_small': h < h_opt * 0.1,
        }

    def multi_l_convergence(
        self,
        l_values: Optional[List[int]] = None,
        n_points: int = 2000
    ) -> Dict:
        """
        对多个角动量 l 值进行相移收敛性分析

        返回:
            各 l 的相移和收敛性信息
        """
        if l_values is None:
            l_values = [0, 1, 2, 3, 4, 5]

        results = {}
        for l in l_values:
            r = np.linspace(0.01, self.r_max, n_points)
            V = self.potential.total_potential(r, l_quantum=l)

            solver = RadialSchrodingerSolver(
                self.potential, r_max=self.r_max, n_points=n_points
            )
            _, u = solver.solve_numerov(l, V)

            ps_calc = PhaseShiftCalculator(
                k=self.potential.p.k_wavevector,
                r_match=self.r_max * 0.85
            )
            delta = ps_calc.extract_phase_shift(u, r, l)

            # 波函数范数
            norm_info = l2_norm_radial(u.real, r)

            results[f'l_{l}'] = {
                'phase_shift_real': float(np.real(delta)),
                'phase_shift_imag': float(np.imag(delta)),
                'wavefunction_norm': float(norm_info),
                'converged': abs(np.imag(delta)) < 2.0,
            }

        return results

    def comparison_against_exact(
        self,
        r: np.ndarray,
        u_numerical: np.ndarray,
        u_exact: np.ndarray
    ) -> Dict:
        """
        数值解与精确解的误差分析

        误差度量:
        - 最大绝对误差: max|u_num - u_exact|
        - L2 相对误差: ||u_num - u_exact|| / ||u_exact||
        - 内积偏差: |<u_num|u_exact> / (||u_num||*||u_exact||) - 1|

        返回:
            误差统计
        """
        diff = u_numerical - u_exact

        max_abs_error = float(np.max(np.abs(diff)))
        l2_error = l2_norm_radial(diff, r)
        l2_exact = l2_norm_radial(u_exact, r)
        l2_relative = l2_error / (l2_exact + 1e-30)

        # 内积重叠
        overlap_real = np.real(np.sum(np.conj(u_numerical) * u_exact) *
                                (r[1] - r[0]))
        overlap_denom = (l2_norm_radial(u_numerical, r) * l2_exact + 1e-30)
        overlap = overlap_real / overlap_denom

        return {
            'max_absolute_error': max_abs_error,
            'l2_relative_error': l2_relative,
            'overlap_integral': float(overlap),
            'n_points': len(r),
        }
