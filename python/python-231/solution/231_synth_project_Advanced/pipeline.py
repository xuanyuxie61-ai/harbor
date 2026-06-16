"""
pipeline.py — PDF 全局拟合流水线编排 (映射自 SLAI)
=============================================================
本模块编排整个 PDF 全局拟合与误差传播流程.
映射自 SLAI (Sarathi) 的 LLM 服务流水线架构:
  - Timer 系统: 跟踪每个阶段的计算时间
  - Metrics: 收集并报告各阶段的指标
  - Pipeline stages: 按序执行各计算阶段

流水线阶段:
  Stage 1: 初始化 (参数、网格、数据)
  Stage 2: DGLAP 演化
  Stage 3: χ² 拟合
  Stage 4: Hessian 误差分析
  Stage 5: MC 副本分析
  Stage 6: 稳定性分析
  Stage 7: 退化度分析
  Stage 8: 结果汇总

核心公式 (χ²/dof):
    χ²/dof = χ²_min / (N_data − N_params)

核心公式 (p-value):
    p = 1 − Γ(χ²/2, dof/2) / Γ(dof/2)
    (简化: 使用正态近似 p ≈ exp(−(χ²/dof − 1)² × dof / 2))
"""
from __future__ import annotations
import time
import math
from typing import Dict, List, Tuple, Optional, Any

from phys_consts import (
    NX_DEFAULT, NQ_DEFAULT, X_MIN_DEFAULT,
    Q0_SQ_DEFAULT, QMAX_SQ_DEFAULT, LAMBDA_QCD_LO,
    alpha_s_lo, EPS_NUMERICAL
)
from dglap_evolution import (
    generate_x_grid, generate_q2_grid, evolve_dglap, x_to_v,
    track_perturbation_propagation, DGLAPEvolutionResult
)
from pdf_param import (
    default_params, xuv_x, xdv_x, xg_x, xs_x, normalize_parameters,
    momentum_sum_rule, valence_number_rule, evaluate_all_flavors
)
from experimental_data import (
    generate_dis_data, generate_dy_data, combine_datasets,
    mc_probability_estimate
)
from pdf_fit import (
    fit_pdf_params, compute_chi_squared, compute_chi2_data_only,
    soleflip_sensitivity_analysis, sparse_outlier_detection,
    robust_chi_squared
)
from hessian_error import (
    compute_hessian, hessian_eigendecomposition,
    hessian_pdf_uncertainty, mellin_modular_evolution,
    caustic_envelope_density, hessian_correlation_display
)
from root_finding import chandrupatla_root, invert_alpha_s, chi2_parabola_minimum
from pdf_interp import pwl_interp_1d, interpolate_pdf_at_point
from comb_canal import (
    count_degenerate_directions, canalizing_depth,
    pdf_degeneracy_report, nchoosek, stirling_second_kind
)
from fd_stability import (
    stability_check, cfl_max_timestep, print_stability_report,
    circulant_eigenvalues, second_diff_circulant
)
from mc_replicas import (
    fit_mc_replicas, mc_pdf_uncertainty, cluster_pdf_replicas
)


# ============================================================
# 1. Timer 与 Metrics (映射自 SLAI)
# ============================================================
class StageTimer:
    """阶段计时器 (映射自 SLAI 的 CudaTimer / CpuTimer)"""
    def __init__(self):
        self.records: Dict[str, float] = {}
        self._start: Dict[str, float] = {}

    def start(self, name: str):
        self._start[name] = time.time()

    def stop(self, name: str) -> float:
        if name in self._start:
            elapsed = time.time() - self._start[name]
            self.records[name] = elapsed
            del self._start[name]
            return elapsed
        return 0.0

    def report(self) -> str:
        lines = ["  流水线计时报告:"]
        total = 0.0
        for name, elapsed in self.records.items():
            lines.append(f"    {name:30s}: {elapsed:8.4f} s")
            total += elapsed
        lines.append(f"    {'总计':30s}: {total:8.4f} s")
        return '\n'.join(lines)


class MetricsStore:
    """指标收集器 (映射自 SLAI 的 MetricsStore)"""
    def __init__(self):
        self.metrics: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        self.metrics[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def report(self) -> str:
        lines = ["  关键指标:"]
        for key, val in self.metrics.items():
            if isinstance(val, float):
                lines.append(f"    {key:30s}: {val:12.6f}")
            else:
                lines.append(f"    {key:30s}: {val}")
        return '\n'.join(lines)


# ============================================================
# 2. 完整流水线
# ============================================================
class PDFGlobalFitPipeline:
    """PDF 全局拟合流水线"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.timer = StageTimer()
        self.metrics = MetricsStore()
        self.results: Dict[str, Any] = {}

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """运行完整流水线"""
        if verbose:
            print("\n" + "=" * 70)
            print("  PDF 全局拟合与误差传播流水线")
            print("  计算高能物理：高阶有限差分与稳定性分析")
            print("=" * 70)

        # Stage 1: 初始化
        self._stage1_init(verbose)
        # Stage 2: DGLAP 演化
        self._stage2_evolution(verbose)
        # Stage 3: χ² 拟合
        self._stage3_fit(verbose)
        # Stage 4: Hessian 误差分析
        self._stage4_hessian(verbose)
        # Stage 5: MC 副本
        self._stage5_mc_replicas(verbose)
        # Stage 6: 稳定性分析
        self._stage6_stability(verbose)
        # Stage 7: 退化度分析
        self._stage7_degeneracy(verbose)
        # Stage 8: 结果汇总
        self._stage8_summary(verbose)

        if verbose:
            print("\n" + self.timer.report())
            print("\n" + self.metrics.report())

        return self.results

    def _stage1_init(self, verbose: bool):
        """Stage 1: 初始化"""
        self.timer.start('stage1_init')
        if verbose:
            print("\n[Stage 1] 初始化参数、网格与合成数据...")

        # 参数与网格
        nx = self.config.get('nx', NX_DEFAULT)
        nq = self.config.get('nq', NQ_DEFAULT)
        q02 = self.config.get('q02', Q0_SQ_DEFAULT)
        qmax2 = self.config.get('qmax2', QMAX_SQ_DEFAULT)

        x_grid = generate_x_grid(nx, method='log')
        q2_grid = generate_q2_grid(nq, q02, qmax2)

        params = default_params()
        params = normalize_parameters(params, x_grid)

        # 生成合成数据
        dis_data = generate_dis_data(params, seed=42)
        dy_data = generate_dy_data(params, seed=123)
        data = combine_datasets([dis_data, dy_data])

        self.results['x_grid'] = x_grid
        self.results['q2_grid'] = q2_grid
        self.results['params_init'] = params
        self.results['data'] = data

        self.metrics.set('n_data', len(data))
        self.metrics.set('n_x', nx)
        self.metrics.set('n_q2', nq)
        self.metrics.set('Q0²', q02)
        self.metrics.set('Qmax²', qmax2)
        self.metrics.set('Λ_QCD (LO)', LAMBDA_QCD_LO)

        if verbose:
            print(f"  x 网格: {nx} 点, x ∈ [{x_grid[0]:.6f}, {x_grid[-1]:.6f}]")
            print(f"  Q² 网格: {nq} 点, Q² ∈ [{q2_grid[0]:.2f}, {q2_grid[-1]:.2f}] GeV²")
            print(f"  数据点: DIS={len(dis_data)}, DY={len(dy_data)}, 总计={len(data)}")
            print(f"  Λ_QCD (LO, nf=5) = {LAMBDA_QCD_LO:.6f} GeV")

        self.timer.stop('stage1_init')

    def _stage2_evolution(self, verbose: bool):
        """Stage 2: DGLAP 演化"""
        self.timer.start('stage2_evolution')
        if verbose:
            print("\n[Stage 2] DGLAP 演化 (IMEX 二阶)...")

        x_grid = self.results['x_grid']
        q2_grid = self.results['q2_grid']
        params = self.results['params_init']

        # 计算初始 PDF 在 x_grid 上的值
        init_uv = [xuv_x(x, params) for x in x_grid]
        init_dv = [xdv_x(x, params) for x in x_grid]
        init_g = [xg_x(x, params) for x in x_grid]
        init_s = [xs_x(x, params) for x in x_grid]

        evo = evolve_dglap(x_grid, q2_grid, init_uv, init_dv, init_g, init_s,
                           nf=3, order=2, verbose=verbose)

        self.results['evolution'] = evo

        # 动量守恒检查
        iq_final = len(q2_grid) - 1
        f_uv, f_dv, f_g, f_s = evo.pdf_history[iq_final]
        mom_init = sum(0.5 * (x_grid[i] * (init_uv[i] + init_dv[i] + init_g[i] + init_s[i])
                              + x_grid[i + 1] * (init_uv[i + 1] + init_dv[i + 1]
                                                 + init_g[i + 1] + init_s[i + 1]))
                       * (x_grid[i + 1] - x_grid[i]) for i in range(len(x_grid) - 1))
        mom_final = sum(0.5 * (x_grid[i] * (f_uv[i] + f_dv[i] + f_g[i] + f_s[i])
                               + x_grid[i + 1] * (f_uv[i + 1] + f_dv[i + 1]
                                                  + f_g[i + 1] + f_s[i + 1]))
                        * (x_grid[i + 1] - x_grid[i]) for i in range(len(x_grid) - 1))

        self.metrics.set('momentum_initial', mom_init)
        self.metrics.set('momentum_final', mom_final)
        self.metrics.set('momentum_conservation', abs(mom_final - mom_init))

        if verbose:
            print(f"  初始动量 = {mom_init:.6f}")
            print(f"  最终动量 = {mom_final:.6f}")
            print(f"  动量守恒偏差 = {abs(mom_final - mom_init):.2e}")

        self.timer.stop('stage2_evolution')

    def _stage3_fit(self, verbose: bool):
        """Stage 3: χ² 拟合"""
        self.timer.start('stage3_fit')
        if verbose:
            print("\n[Stage 3] χ² 全局拟合...")

        data = self.results['data']
        x_grid = self.results['x_grid']

        p_fit, chi2_hist, n_iter = fit_pdf_params(
            data, max_iter=15, lr=0.001, x_grid=x_grid, verbose=verbose)

        chi2_final = chi2_hist[-1] if chi2_hist else float('nan')
        n_data = len(data)
        n_params = len(p_fit)
        dof = max(n_data - n_params, 1)
        chi2_per_dof = chi2_final / dof

        # 单参数翻转敏感度
        if verbose:
            print("\n  SoleFlip 敏感度分析:")
        sensitivity = soleflip_sensitivity_analysis(p_fit, data, x_grid)
        sorted_params = sorted(sensitivity.keys(),
                               key=lambda k: sensitivity[k]['delta_chi2'],
                               reverse=True)
        if verbose:
            for rank, k in enumerate(sorted_params[:5]):
                s = sensitivity[k]
                print(f"    #{rank + 1} {k:8s}: Δχ² = {s['delta_chi2']:.6f}, "
                      f"Δmom = {s['momentum_shift']:.2e}")

        # 稀疏异常检测
        outlier_idx, outlier_res = sparse_outlier_detection(p_fit, data)
        if verbose:
            print(f"\n  异常数据点: {len(outlier_idx)} 个 (|r| > 3σ)")
            if outlier_idx:
                print(f"    前3个: {outlier_idx[:3]}, 残差: {[f'{r:.2f}' for r in outlier_res[:3]]}")

        # 鲁棒 χ²
        chi2_robust = robust_chi_squared(p_fit, data)

        self.results['params_fit'] = p_fit
        self.results['chi2_history'] = chi2_hist
        self.results['sensitivity'] = sensitivity

        self.metrics.set('chi2_min', chi2_final)
        self.metrics.set('chi2_per_dof', chi2_per_dof)
        self.metrics.set('n_iterations', n_iter)
        self.metrics.set('n_outliers', len(outlier_idx))
        self.metrics.set('chi2_robust', chi2_robust)

        if verbose:
            print(f"\n  拟合结果: χ²_min = {chi2_final:.4f}, χ²/dof = {chi2_per_dof:.4f}")
            print(f"  鲁棒 χ² = {chi2_robust:.4f}")

        self.timer.stop('stage3_fit')

    def _stage4_hessian(self, verbose: bool):
        """Stage 4: Hessian 误差分析"""
        self.timer.start('stage4_hessian')
        if verbose:
            print("\n[Stage 4] Hessian 误差分析...")

        p_fit = self.results['params_fit']
        data = self.results['data']
        x_grid = self.results['x_grid']

        # Hessian 矩阵
        hessian = compute_hessian(p_fit, data, x_grid=x_grid)
        eigs, vecs = hessian_eigendecomposition(hessian)

        if verbose:
            print(f"  Hessian 特征值 (前5): {[f'{e:.4f}' for e in sorted(eigs, reverse=True)[:5]]}")
            print(f"  特征值范围: [{min(eigs):.4f}, {max(eigs):.4f}]")

        # Mellin 模块化演化 (映射自 caustic)
        N_max = 10
        m_mult = 3
        q2_ratio = 100.0
        as_val = alpha_s_lo(100.0)
        mellin_factors = mellin_modular_evolution(N_max, m_mult, q2_ratio, as_val)
        envelope = caustic_envelope_density(N_max, m_mult)

        if verbose:
            print(f"\n  Mellin 模块化演化 (N_max={N_max}, m={m_mult}):")
            print(f"    |E(N)| 范围: [{min(abs(e) for e in mellin_factors):.4f}, "
                  f"{max(abs(e) for e in mellin_factors):.4f}]")
            print(f"    焦散包络密度最大值: {max(envelope):.4f}")

        # Hessian 相关矩阵显示
        keys = list(p_fit.keys())
        corr_display = hessian_correlation_display(hessian, keys)
        if verbose:
            print(f"\n{corr_display}")

        # α_s 反求
        q2_analytic, q2_chandra = invert_alpha_s(0.3)
        if verbose:
            print(f"\n  α_s = 0.3 对应的 Q²:")
            print(f"    解析: {q2_analytic:.4f} GeV²")
            print(f"    Chandrupatla: {q2_chandra:.4f} GeV²")

        self.results['hessian'] = hessian
        self.results['eigenvalues'] = eigs
        self.results['mellin_factors'] = mellin_factors

        self.metrics.set('hessian_max_eigenvalue', max(eigs))
        self.metrics.set('hessian_min_eigenvalue', min(eigs))
        self.metrics.set('hessian_condition', max(eigs) / max(abs(min(eigs)), EPS_NUMERICAL))

        self.timer.stop('stage4_hessian')

    def _stage5_mc_replicas(self, verbose: bool):
        """Stage 5: MC 副本分析"""
        self.timer.start('stage5_mc_replicas')
        if verbose:
            print("\n[Stage 5] Monte Carlo 副本分析...")

        data = self.results['data']
        p_fit = self.results['params_fit']
        x_grid = self.results['x_grid']
        n_rep = self.config.get('n_replicas', 3)

        # MC 副本拟合
        fitted_mc = fit_mc_replicas(data, n_replicas=n_rep,
                                    init_params=p_fit, fit_max_iter=15,
                                    x_grid=x_grid, verbose=verbose)

        # MC 不确定度
        f_mean, f_upper, f_lower = mc_pdf_uncertainty(fitted_mc, x_grid, 'g')
        if verbose:
            print(f"\n  MC 胶子 PDF 不确定度 (在 x = {x_grid[len(x_grid) // 2]:.4f}):")
            mid = len(x_grid) // 2
            print(f"    中心值: {f_mean[mid]:.6f}")
            print(f"    上限:   {f_upper[mid]:.6f}")
            print(f"    下限:   {f_lower[mid]:.6f}")
            print(f"    相对误差: {(f_upper[mid] - f_lower[mid]) / max(f_mean[mid], 1e-10):.4f}")

        # k-means 聚类
        centers, assignments, inertia, stats = cluster_pdf_replicas(fitted_mc, k=2)
        if verbose:
            print(f"\n  k-means 聚类 (k=2):")
            print(f"    聚类大小: {stats['cluster_sizes']}")
            print(f"    惯性: {inertia:.6f}")

        # 蒙特卡洛概率估计
        def is_momentum_ok(rng):
            p_test = dict(p_fit)
            for k in p_test:
                p_test[k] += 0.01 * rng.gauss(0, 1)
            mom = momentum_sum_rule(x_grid, p_test)
            return 0.8 < mom < 1.2

        p_ok, sigma_p = mc_probability_estimate(200, is_momentum_ok, seed=42)
        if verbose:
            print(f"\n  动量求和规则满足概率: {p_ok:.4f} ± {sigma_p:.4f}")

        self.results['mc_params'] = fitted_mc
        self.results['mc_pdf_mean'] = f_mean
        self.results['mc_pdf_upper'] = f_upper
        self.results['mc_pdf_lower'] = f_lower
        self.results['mc_cluster_stats'] = stats

        self.metrics.set('mc_inertia', inertia)
        self.metrics.set('mc_momentum_prob', p_ok)

        self.timer.stop('stage5_mc_replicas')

    def _stage6_stability(self, verbose: bool):
        """Stage 6: 稳定性分析"""
        self.timer.start('stage6_stability')
        if verbose:
            print("\n[Stage 6] 有限差分格式稳定性分析...")

        q2_mid = self.results['q2_grid'][len(self.results['q2_grid']) // 2]
        h_values = [0.1, 0.2, 0.5, 1.0, 2.0]

        report = print_stability_report(h_values, q2_mid, nf=3)
        if verbose:
            print(report)

        # CFL 时间步长
        h_typical = 0.5
        dt_max = cfl_max_timestep(h_typical, q2_mid, nf=3)
        if verbose:
            print(f"\n  CFL 最大步长 (h={h_typical}): dt_max = {dt_max:.6f}")

        # 循环矩阵特征值
        n_circ = 8
        d2_row = second_diff_circulant(n_circ, h=1.0)
        circ_eigs = circulant_eigenvalues(d2_row)
        if verbose:
            print(f"\n  二阶差分循环矩阵 ({n_circ}×{n_circ}) 特征值:")
            print(f"    {[f'{e.real:.4f}' for e in circ_eigs]}")

        self.results['stability_report'] = report
        self.results['cfl_dt_max'] = dt_max
        self.metrics.set('dt_max_cfl', dt_max)

        self.timer.stop('stage6_stability')

    def _stage7_degeneracy(self, verbose: bool):
        """Stage 7: 退化度分析"""
        self.timer.start('stage7_degeneracy')
        if verbose:
            print("\n[Stage 7] 特征向量退化度分析...")

        eigs = self.results.get('eigenvalues', [])
        n_params = len(self.results.get('params_fit', {}))

        report = pdf_degeneracy_report(eigs, n_params)
        if verbose:
            print(report)

        self.results['degeneracy_report'] = report
        self.timer.stop('stage7_degeneracy')

    def _stage8_summary(self, verbose: bool):
        """Stage 8: 结果汇总"""
        self.timer.start('stage8_summary')
        if verbose:
            print("\n[Stage 8] 结果汇总...")
            print("\n" + "=" * 70)
            print("  PDF 全局拟合最终结果")
            print("=" * 70)

            p_fit = self.results.get('params_fit', {})
            print("\n  拟合参数:")
            for k, v in p_fit.items():
                print(f"    {k:8s} = {v:12.6f}")

            print(f"\n  χ²_min = {self.metrics.get('chi2_min', float('nan')):.4f}")
            print(f"  χ²/dof = {self.metrics.get('chi2_per_dof', float('nan')):.4f}")
            print(f"  数据点数 = {self.metrics.get('n_data', 0)}")
            print(f"  拟合迭代 = {self.metrics.get('n_iterations', 0)}")
            print(f"  Λ_QCD (LO) = {self.metrics.get('Λ_QCD (LO)', 0):.6f} GeV")
            print(f"  动量守恒偏差 = {self.metrics.get('momentum_conservation', 0):.2e}")

            print("\n  流水线完成!")
            print("=" * 70)

        self.timer.stop('stage8_summary')
