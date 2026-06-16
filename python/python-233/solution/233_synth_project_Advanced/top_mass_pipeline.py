"""
top_mass_pipeline.py — 完整顶夸克质量分析流水线
===============================================
本模块整合所有子模块, 实现完整的顶夸克质量测量流程:

  数据生成 → 信号/背景模板 → 探测器卷积 → 似然构建 →
  质量拟合 → 系统误差评估 → 稳定性分析 → 结果输出

完整流水线:
  Step 1: 物理参数初始化
  Step 2: 信号模板计算 (光谱函数 + PDF)
  Step 3: 背景模型构建
  Step 4: 探测器响应卷积
  Step 5: 伪数据生成 (或加载观测数据)
  Step 6: 轮廓似然扫描
  Step 7: 质量拟合与误差提取
  Step 8: 高阶有限差分灵敏度分析
  Step 9: 非线性动力学稳定性分析
  Step 10: 组合优化喷注配对

映射种子项目:
  所有 15 个种子项目的核心算法均在此流水线中协同工作。
"""

import numpy as np
import time as time_module

# 导入所有子模块
from topmass_constants import (
    M_TOP_POLE_DEFAULT, M_W_BOSON, G_FERMI, ALPHA_S_MZ,
    alpha_s_running, top_width_lo, SQRT_S_LHC
)
from spectral_function import (
    ttbar_invariant_mass_spectrum, ttbar_threshold_cross_section,
    greens_function_threshold, sommerfeld_enhancement
)
from finite_diff_operators import (
    central_diff_4th, richardson_extrapolation,
    numerical_gradient, numerical_hessian, adaptive_step_size
)
from hammersley_phase_space import (
    hammersley_sequence, hammersley_to_ttbar_phase_space,
    star_discrepancy, qmc_integration
)
from detector_response import (
    detector_resolution_function, convolve_with_detector,
    gaussian_response, navier_stokes_diffusion_kernel
)
from polygon_phase_grid import (
    hexagonal_brillouin_grid, polygon_grid_points,
    nuisance_parameter_hex_scan
)
from hankel_momentum_transform import (
    hankel_transform_log_grid, top_pt_spectrum,
    hankel_matrix_construction
)
from monomial_symmetry import (
    elementary_symmetric_polynomials, ttbar_event_symmetrize,
    symmetrize_coefficients
)
from time_delay_stability import (
    floquet_multipliers_1d, convergence_monitor,
    iterative_mass_solver, border_collision_detection
)
from knapsack_combinatoric import (
    brute_force_jet_assignment, efficient_jet_matching,
    combinatorial_background_estimate
)
from profile_likelihood import (
    profile_likelihood, scan_profile_likelihood,
    confidence_interval, template_signal, template_background,
    poisson_log_likelihood
)
from nonlinear_ode_solvers import (
    duffing_fit_dynamics, anishchenko_fit_switch,
    integrate_fit_trajectory, phase_space_analysis
)


class TopMassAnalysisPipeline:
    """
    顶夸克质量测量完整分析流水线。

    封装从数据生成到物理结果提取的全部步骤。
    """

    def __init__(self, config=None):
        """
        初始化流水线配置。

        参数:
            config: 配置字典 (可选)
        """
        self.config = config or {}

        # 物理参数
        self.m_top_true = self.config.get('m_top_true', 172.5)
        self.sqrt_s = self.config.get('sqrt_s', SQRT_S_LHC)
        self.luminosity = self.config.get('luminosity', 139.0)  # fb⁻¹
        self.n_events = self.config.get('n_events', 10000)

        # 分析参数
        self.m_scan_min = self.config.get('m_scan_min', 165.0)
        self.m_scan_max = self.config.get('m_scan_max', 180.0)
        self.m_scan_npoints = self.config.get('m_scan_npoints', 30)
        self.mass_bin_edges = self.config.get('mass_bin_edges', None)

        # 系统误差
        self.systematic_uncertainties = self.config.get(
            'systematic_uncertainties',
            {'JES': 0.01, 'b_energy': 0.005, 'luminosity': 0.02,
             'PDF': 0.015, 'alpha_s': 0.01}
        )

        # 结果存储
        self.results = {}

    def run(self, verbose=True):
        """
        运行完整分析流水线。

        参数:
            verbose: 是否打印进度信息

        返回:
            results: 完整结果字典
        """
        t_start = time_module.time()

        if verbose:
            print("=" * 70)
            print("  顶夸克质量测量 — 完整分析流水线")
            print("  Computational High Energy Physics")
            print("  Top Quark Mass Statistical Analysis")
            print("=" * 70)

        # Step 1: 物理参数初始化
        if verbose:
            print("\n[Step 1] 物理参数初始化...")
        self._step1_physics_setup(verbose)

        # Step 2: 信号模板
        if verbose:
            print("\n[Step 2] 计算信号模板 (NRQCD 格林函数 + PDF)...")
        self._step2_signal_template(verbose)

        # Step 3: 背景模型
        if verbose:
            print("\n[Step 3] 构建背景模型...")
        self._step3_background(verbose)

        # Step 4: 探测器卷积
        if verbose:
            print("\n[Step 4] 探测器响应卷积 (Navier-Stokes 扩散核)...")
        self._step4_detector_convolution(verbose)

        # Step 5: 伪数据生成
        if verbose:
            print("\n[Step 5] 生成伪数据 (Hammersley QMC 采样)...")
        self._step5_generate_pseudodata(verbose)

        # Step 6: 轮廓似然扫描
        if verbose:
            print("\n[Step 6] 轮廓似然比扫描...")
        self._step6_likelihood_scan(verbose)

        # Step 7: 质量拟合
        if verbose:
            print("\n[Step 7] 质量拟合与误差提取...")
        self._step7_mass_fit(verbose)

        # Step 8: 有限差分灵敏度
        if verbose:
            print("\n[Step 8] 高阶有限差分灵敏度分析...")
        self._step8_finite_diff_sensitivity(verbose)

        # Step 9: 稳定性分析
        if verbose:
            print("\n[Step 9] 非线性动力学稳定性分析 (Duffing/Floquet)...")
        self._step9_stability_analysis(verbose)

        # Step 10: 组合优化
        if verbose:
            print("\n[Step 10] 喷注组合优化 (knapsack 暴力枚举)...")
        self._step10_combinatorial(verbose)

        t_end = time_module.time()
        self.results['total_time'] = t_end - t_start

        if verbose:
            self._print_summary()

        return self.results

    def _step1_physics_setup(self, verbose):
        """Step 1: 物理参数设置。"""
        # 顶夸克宽度
        gamma_t = top_width_lo(self.m_top_true)
        tau_t = 6.582e-25 / gamma_t

        # 强耦合常数
        alpha_s_mt = alpha_s_running(self.m_top_true)
        alpha_s_mz = ALPHA_S_MZ

        # Sommerfeld 增强 (阈值)
        beta_threshold = 0.1
        S_somm = sommerfeld_enhancement(beta_threshold, self.m_top_true, alpha_s_mt)

        self.results['physics'] = {
            'm_top_true': self.m_top_true,
            'gamma_top': gamma_t,
            'tau_top': tau_t,
            'alpha_s_mt': alpha_s_mt,
            'alpha_s_mz': alpha_s_mz,
            'sommerfeld_at_threshold': S_somm,
        }

        if verbose:
            print(f"  m_t (true) = {self.m_top_true} GeV")
            print(f"  Γ_t = {gamma_t:.4f} GeV")
            print(f"  τ_t = {tau_t:.3e} s")
            print(f"  α_s(m_t) = {alpha_s_mt:.5f}")
            print(f"  S(β=0.1) = {S_somm:.4f}")

    def _step2_signal_template(self, verbose):
        """Step 2: 计算信号模板。"""
        if self.mass_bin_edges is None:
            self.mass_bin_edges = np.linspace(300.0, 500.0, 21)

        m_centers = 0.5 * (self.mass_bin_edges[:-1] + self.mass_bin_edges[1:])

        # 计算各质量点的微分截面
        signal_raw = np.zeros(len(m_centers))
        for i, M in enumerate(m_centers):
            signal_raw[i] = ttbar_invariant_mass_spectrum(M, self.m_top_true)

        # 乘以光度得到事件数
        signal_events = signal_raw * self.luminosity * 1000 * \
            np.diff(self.mass_bin_edges)

        self.results['signal'] = {
            'mass_bins': self.mass_bin_edges,
            'mass_centers': m_centers,
            'dsigma_dM': signal_raw,
            'signal_events': signal_events,
        }

        if verbose:
            print(f"  质量范围: [{self.mass_bin_edges[0]:.0f}, "
                  f"{self.mass_bin_edges[-1]:.0f}] GeV")
            print(f"  Bin 数: {len(m_centers)}")
            print(f"  总信号事件数: {np.sum(signal_events):.1f}")
            print(f"  峰值截面: {np.max(signal_raw):.4f} pb/GeV")

    def _step3_background(self, verbose):
        """Step 3: 背景模型。"""
        bg = template_background(self.mass_bin_edges)

        self.results['background'] = {
            'background_events': bg,
        }

        if verbose:
            print(f"  总背景事件数: {np.sum(bg):.1f}")

    def _step4_detector_convolution(self, verbose):
        """Step 4: 探测器响应卷积。"""
        m_centers = self.results['signal']['mass_centers']
        signal_raw = self.results['signal']['dsigma_dM']

        # 卷积
        observed_signal = convolve_with_detector(
            m_centers, signal_raw, m_centers, channel='semileptonic'
        )

        # Navier-Stokes 扩散核验证
        sigma_test = detector_resolution_function(350.0)
        ns_kernel = navier_stokes_diffusion_kernel(
            350.0 + sigma_test, 350.0, sigma_test**2 / 2.0
        )

        self.results['detector'] = {
            'observed_signal': observed_signal,
            'resolution_at_350': sigma_test,
            'ns_kernel_test': ns_kernel,
        }

        if verbose:
            print(f"  σ_M(350 GeV) = {sigma_test:.2f} GeV")
            print(f"  卷积后信号峰: {np.max(observed_signal):.4f} pb/GeV")

    def _step5_generate_pseudodata(self, verbose):
        """Step 5: 生成伪数据。"""
        signal = self.results['signal']['signal_events']
        bg = self.results['background']['background_events']

        # 探测器展宽后的信号
        obs_signal = self.results['detector']['observed_signal']
        obs_signal_events = obs_signal * self.luminosity * 1000 * \
            np.diff(self.mass_bin_edges)

        # 期望值
        expected = obs_signal_events + bg
        expected = np.maximum(expected, 0.01)

        # Poisson 涨落
        rng = np.random.default_rng(seed=42)
        n_observed = rng.poisson(expected)

        # Hammersley QMC 采样验证
        qmc_points = hammersley_sequence(500, 6)
        D_star = star_discrepancy(qmc_points)

        self.results['data'] = {
            'n_observed': n_observed,
            'expected': expected,
            'qmc_discrepancy': D_star,
            'total_observed': np.sum(n_observed),
            'total_expected': np.sum(expected),
        }

        if verbose:
            print(f"  总观测事件: {np.sum(n_observed)}")
            print(f"  总期望事件: {np.sum(expected):.1f}")
            print(f"  Hammersley D*_500 = {D_star:.6f}")

    def _step6_likelihood_scan(self, verbose):
        """Step 6: 轮廓似然扫描。"""
        m_scan = np.linspace(self.m_scan_min, self.m_scan_max,
                             self.m_scan_npoints)
        n_obs = self.results['data']['n_observed']
        bg = self.results['background']['background_events']

        lnL_values, m_best, delta_lnL = scan_profile_likelihood(
            m_scan, self.mass_bin_edges, n_obs, bg,
            systematic_uncertainties=[0.01, 0.005, 0.02]
        )

        # 置信区间
        m_lower_68, m_upper_68, _ = confidence_interval(m_scan, delta_lnL, 0.68)
        m_lower_95, m_upper_95, _ = confidence_interval(m_scan, delta_lnL, 0.95)

        self.results['likelihood'] = {
            'm_scan': m_scan,
            'lnL': lnL_values,
            'delta_lnL': delta_lnL,
            'm_best_fit': m_best,
            'm_lower_68': m_lower_68,
            'm_upper_68': m_upper_68,
            'm_lower_95': m_lower_95,
            'm_upper_95': m_upper_95,
        }

        if verbose:
            print(f"  最佳拟合 m_t = {m_best:.2f} GeV")
            print(f"  68% CL: [{m_lower_68:.2f}, {m_upper_68:.2f}] GeV")
            print(f"  95% CL: [{m_lower_95:.2f}, {m_upper_95:.2f}] GeV")

    def _step7_mass_fit(self, verbose):
        """Step 7: 迭代质量拟合。"""
        def neg_lnL(m_t):
            lnL, _ = profile_likelihood(
                m_t, self.mass_bin_edges,
                self.results['data']['n_observed'],
                self.results['background']['background_events']
            )
            return lnL

        result = iterative_mass_solver(
            neg_lnL,
            m_init=170.0,
            m_range=(self.m_scan_min, self.m_scan_max),
            learning_rate=0.05,
            max_iter=100,
            tol=1e-5
        )

        self.results['fit'] = result

        if verbose:
            print(f"  迭代拟合 m_t = {result['m_opt']:.4f} GeV")
            print(f"  收敛: {result['converged']}")
            print(f"  迭代次数: {result['n_iterations']}")
            print(f"  Floquet 乘子: {result['floquet_multiplier']:.6f}")

    def _step8_finite_diff_sensitivity(self, verbose):
        """Step 8: 有限差分灵敏度分析。"""
        def cross_section_at_m(m_t):
            return ttbar_threshold_cross_section(
                2.0 * m_t + 10.0, m_t
            )

        # 4 阶中心差分
        h = 0.5  # GeV
        d1, err1 = central_diff_4th(cross_section_at_m, self.m_top_true, h)

        # Richardson 外推
        d_R, err_R, T_table = richardson_extrapolation(
            cross_section_at_m, self.m_top_true, h
        )

        # 自适应步长
        d_adapt, h_opt, err_adapt = adaptive_step_size(
            cross_section_at_m, self.m_top_true
        )

        # Hessian (二阶导数)
        def neg_lnL_scalar(m_arr):
            m_t = m_arr[0]
            lnL, _ = profile_likelihood(
                m_t, self.mass_bin_edges,
                self.results['data']['n_observed'],
                self.results['background']['background_events']
            )
            return lnL

        H = numerical_hessian(neg_lnL_scalar, np.array([self.m_top_true]))

        self.results['finite_diff'] = {
            'first_derivative': d1,
            'fd_error': err1,
            'richardson_deriv': d_R,
            'richardson_error': err_R,
            'adaptive_deriv': d_adapt,
            'optimal_step': h_opt,
            'hessian': H,
            'richardson_table': T_table,
        }

        if verbose:
            print(f"  dσ/dm (4th order) = {d1:.6f} pb/GeV")
            print(f"  dσ/dm (Richardson) = {d_R:.6f} pb/GeV")
            print(f"  误差 (4th) = {err1:.2e}")
            print(f"  误差 (Richardson) = {err_R:.2e}")
            print(f"  Hessian H₁₁ = {H[0, 0]:.6f}")

    def _step9_stability_analysis(self, verbose):
        """Step 9: 非线性动力学稳定性。"""
        # Duffing 拟合动力学
        duffing_params = {
            'damping': 0.5,
            'linear_stiffness': 2.0,
            'nonlinear_stiffness': 0.01,
            'forcing_amplitude': 0.1,
            'forcing_frequency': 0.5,
            'target_mass': self.m_top_true,
        }

        y0_duff = np.array([170.0, 0.0])  # [m, m']
        duff_result = integrate_fit_trajectory(
            duffing_fit_dynamics, y0_duff, (0, 50.0), duffing_params
        )

        # Anishchenko 动力学
        anish_params = {'mu': 1.2, 'eta': 0.5}
        y0_anish = np.array([-0.1, 0.5, -0.6])
        anish_result = integrate_fit_trajectory(
            anishchenko_fit_switch, y0_anish, (0, 50.0), anish_params
        )

        # 相空间分析
        phase_result = phase_space_analysis(
            duffing_fit_dynamics, y0_duff, duffing_params, t_max=30.0
        )

        # 六边形参数扫描
        hex_grid = hexagonal_brillouin_grid(3)

        self.results['stability'] = {
            'duffing_final': duff_result['final_state'],
            'duffing_n_steps': duff_result['n_steps'],
            'anishchenko_final': anish_result['final_state'],
            'phase_converged': phase_result['converged'],
            'lyapunov': phase_result['lyapunov_exponent'],
            'hex_grid_points': len(hex_grid),
        }

        if verbose:
            print(f"  Duffing 最终状态: m = {duff_result['final_state'][0]:.4f}")
            print(f"  Duffing 步数: {duff_result['n_steps']}")
            print(f"  相空间收敛: {phase_result['converged']}")
            print(f"  Lyapunov 指数: {phase_result['lyapunov_exponent']:.6f}")
            print(f"  六边形网格点数: {len(hex_grid)}")

    def _step10_combinatorial(self, verbose):
        """Step 10: 喷注组合优化。"""
        # 模拟喷注事件
        rng = np.random.default_rng(seed=123)
        n_jets = 6
        jets_4vec = np.zeros((n_jets, 4))
        for j in range(n_jets):
            E = rng.uniform(30, 200)
            pt = E * rng.uniform(0.3, 0.9)
            phi = rng.uniform(0, 2 * np.pi)
            eta = rng.uniform(-2.5, 2.5)
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            jets_4vec[j] = [E, px, py, pz]

        b_indices = [0, 1]  # 前两个标记为 b
        light_indices = list(range(2, n_jets))

        # 代价矩阵
        cost = np.zeros((4, n_jets))
        for slot in range(4):
            for j in range(n_jets):
                E_j = jets_4vec[j, 0]
                if slot < 2:
                    cost[slot, j] = (E_j - 80)**2 / 100 if j in b_indices else 1e6
                else:
                    cost[slot, j] = (E_j - 50)**2 / 100 if j in light_indices else 1e6

        # 暴力枚举
        best_asgn, best_cost = brute_force_jet_assignment(cost, n_jets, 4)

        # 高效匹配
        eff_asgn, eff_chi2 = efficient_jet_matching(
            jets_4vec, b_indices, light_indices
        )

        # 组合背景估计
        n_total, n_sig, n_bg = combinatorial_background_estimate(n_jets, 2)

        # 事件对称化
        b_energies = [jets_4vec[i, 0] for i in b_indices]
        sym_obs = ttbar_event_symmetrize(b_energies)

        # Hankel 矩阵构造
        mass_samples = rng.normal(self.m_top_true, 15.0, 100)
        H_mat, H_inv = hankel_matrix_construction(mass_samples, 5)

        # pT 谱
        qT_grid = np.logspace(-1, 2, 30)
        pt_spectrum = top_pt_spectrum(qT_grid, self.m_top_true)

        self.results['combinatorial'] = {
            'best_assignment': best_asgn,
            'best_cost': best_cost,
            'efficient_assignment': eff_asgn,
            'efficient_chi2': eff_chi2,
            'n_total_combinations': n_total,
            'n_signal': n_sig,
            'n_background': n_bg,
            'symmetric_observables': sym_obs,
            'hankel_matrix_cond': np.linalg.cond(H_mat),
            'pt_spectrum_integral': np.trapz(pt_spectrum, qT_grid),
        }

        if verbose:
            print(f"  最佳分配: {best_asgn}")
            print(f"  最小代价: {best_cost:.4f}")
            print(f"  总组合数: {n_total}")
            print(f"  对称可观测量 ΣE_b = {sym_obs['sum_Eb']:.1f} GeV")
            print(f"  Hankel 矩阵条件数: {np.linalg.cond(H_mat):.2e}")
            print(f"  pT 谱积分: {np.trapz(pt_spectrum, qT_grid):.6f}")

    def _print_summary(self):
        """打印结果摘要。"""
        print("\n" + "=" * 70)
        print("  分析结果摘要")
        print("=" * 70)

        physics = self.results.get('physics', {})
        fit = self.results.get('fit', {})
        like = self.results.get('likelihood', {})
        fd = self.results.get('finite_diff', {})
        stab = self.results.get('stability', {})

        print(f"\n  ★ 输入参数:")
        print(f"    m_t (true)     = {physics.get('m_top_true', 'N/A')} GeV")
        print(f"    Γ_t            = {physics.get('gamma_top', 0):.4f} GeV")
        print(f"    α_s(m_t)       = {physics.get('alpha_s_mt', 0):.5f}")

        print(f"\n  ★ 拟合结果:")
        print(f"    m_t (fit)      = {fit.get('m_opt', 'N/A'):.4f} GeV")
        print(f"    m_t (PLR best) = {like.get('m_best_fit', 'N/A'):.2f} GeV")
        m_lo68 = like.get('m_lower_68', 0)
        m_hi68 = like.get('m_upper_68', 0)
        print(f"    68% CL         = [{m_lo68:.2f}, {m_hi68:.2f}] GeV")

        print(f"\n  ★ 有限差分:")
        print(f"    dσ/dm (4th)    = {fd.get('first_derivative', 0):.6f}")
        print(f"    Richardson     = {fd.get('richardson_deriv', 0):.6f}")

        print(f"\n  ★ 稳定性:")
        print(f"    Duffing m_fin  = {stab.get('duffing_final', [0])[0]:.4f}")
        print(f"    Lyapunov       = {stab.get('lyapunov', 0):.6f}")
        print(f"    相空间收敛     = {stab.get('phase_converged', False)}")

        print(f"\n  ★ 组合优化:")
        comb = self.results.get('combinatorial', {})
        print(f"    总配对数       = {comb.get('n_total_combinations', 0)}")
        print(f"    最佳 χ²        = {comb.get('efficient_chi2', 0):.4f}")

        print(f"\n  ★ 计算耗时: {self.results.get('total_time', 0):.2f} s")
        print("=" * 70)
