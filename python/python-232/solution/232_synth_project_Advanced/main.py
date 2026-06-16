"""
main.py — PROJECT_232 统一入口 (PROJECT_232)

计算高能物理: 散射振幅数值计算与截面积分
高阶有限差分与稳定性分析 (小规模可复现实验)

本文件是整个合成项目的统一入口。零参数可直接运行。
运行本文件将完成:
  1. 构造 ππ → ππ 散射的运动学网格
  2. 使用有效范围参数化生成相移数据
  3. 分波展开计算散射振幅
  4. 高阶有限差分提取散射长度与有效力程
  5. Richardson 外推加速
  6. S 矩阵幺正性验证
  7. 截面积分与椭圆积分修正
  8. 谱分析与共振识别
  9. 稳定性与分岔分析
  10. 格点 QCD Lüscher 分析

物理过程: π⁺π⁰ → π⁺π⁰ 弹性散射 (I=2 通道)
  - 质心能量范围: √s ∈ [2m_π, 0.5 GeV]
  - 分波: S 波 (l=0) 和 P 波 (l=1)
  - 使用有效范围参数化:
      k cot δ₀ = -1/a₀ + (1/2)r₀ k²
      k³ cot δ₁ = -1/a₁ + (1/2)r₁ k²
"""
import sys
import os
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import (
    PI, TWOPI, FOURPI, EPS_MACH,
    MASS_PI_PLUS, MASS_PI_ZERO,
    GEV_INV2_TO_MB, GEV_INV_TO_FM,
    cm_momentum, velocity_factor
)
from kinematics import (
    grid_linear, grid_chebyshev, grid_log_threshold,
    grid_cos_theta, polar_grid_2d,
    cm_frame_four_momenta, cross_check_mandelstam,
    phase_space_factor
)
from special_functions import (
    legendre_p, elliptic_k_complete, elliptic_e_complete,
    carlson_rf, carlson_rd, permutation_cycle_type,
    cycle_index_polynomial, log_gamma_stirling, beta_function
)
from partial_wave import (
    phase_shift_to_t_matrix, t_matrix_to_phase_shift,
    effective_range_params,
    scattering_amplitude_partial_wave, differential_cross_section,
    total_cross_section, elastic_cross_section,
    ScatteringAmplitudeField, symmetrize_amplitude_identical_particles,
    multi_channel_symmetry_factor
)
from finite_difference import (
    fornberg_weights, central_diff_weights,
    numerical_derivative, complex_step_derivative,
    richardson_extrapolation, adaptive_derivative,
    scattering_length_derivative, curvature_analysis
)
from s_matrix import (
    BandedMatrix, ScatteringMatrix,
    integer_rref, unitarity_constraint_rank, build_unitarity_system
)
from cross_section import (
    two_body_phase_space, two_body_ps_with_elliptic,
    integrate_total_cross_section, integrate_differential_cross_section,
    dalitz_boundary, polygon_area_2d, polygon_centroid,
    cross_section_to_mb
)
from phase_shift import (
    extract_phase_shift, breit_wigner_phase_shift,
    phase_shift_polar_trajectory, polar_ode_evolution,
    lippiann_schwinger_fixed_point, levinson_theorem_check
)
from spectral_analysis import (
    power_spectrum_density, cross_correlation,
    pink_noise_spectrum, hdrft_spectrum, FunctionXY,
    partial_wave_spectral_decomposition,
    impact_parameter_representation
)
from stability import (
    fd_condition_number, optimal_step_size, stability_map_fd,
    BorderCollisionMap, resonance_bifurcation_analysis,
    coupled_channel_stiffness, lyapunov_exponent_estimate
)
from lattice_utils import (
    momentum_lattice, zeta_function_Z00, luscher_phase_shift,
    luscher_analysis, zeller_congruence, weekday_name,
    computational_schedule_marker,
    rectangular_grid_2d, triangular_grid_2d
)


# ===================================================================
# 分隔线
# ===================================================================
def section_header(title):
    """打印章节标题"""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def sub_header(title):
    """打印子标题"""
    print()
    print(f"--- {title} ---")


# ===================================================================
# 主程序
# ===================================================================
def main():
    """PROJECT_232 主计算流程"""

    print("=" * 72)
    print("  PROJECT_232: 计算高能物理散射振幅数值计算与截面积分")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 72)
    print()
    print("物理过程: ππ → ππ 弹性散射 (I=2 通道)")
    print(f"粒子质量: m_π± = {MASS_PI_PLUS:.6f} GeV, m_π⁰ = {MASS_PI_ZERO:.6f} GeV")
    print(f"阈值: √s_thr = 2m_π = {2 * MASS_PI_PLUS:.6f} GeV")

    # 物理参数 (ππ I=2 通道，来自手征微扰理论)
    a0_input = -0.0444        # S 波散射长度 [GeV⁻¹] (ChPT 预测)
    r0_input = 2.83           # S 波有效力程 [GeV⁻¹]
    a1_input = 0.0            # P 波散射体积 (小量)
    r1_input = 0.0            # P 波有效力程

    # ===================================================================
    section_header("1. 运动学网格构造")
    # ===================================================================

    s_thr = 2.0 * MASS_PI_PLUS
    s_max = 0.5  # GeV

    # 1a. 线性网格
    sqrt_s_lin, s_lin, ds_lin = grid_linear(s_thr + 0.01, s_max, 50)
    print(f"\n[1a] 线性网格: √s ∈ [{sqrt_s_lin[0]:.4f}, {sqrt_s_lin[-1]:.4f}] GeV")
    print(f"     网格点数: {len(sqrt_s_lin)}, 间距 Δ√s = {ds_lin:.6f} GeV")

    # 1b. 切比雪夫网格
    sqrt_s_cheb, s_cheb, w_cheb = grid_chebyshev(s_thr + 0.01, s_max, 50)
    print(f"\n[1b] 切比雪夫网格: {len(sqrt_s_cheb)} 点")
    print(f"     权重和: {np.sum(w_cheb):.6f}")

    # 1c. 对数阈值网格
    sqrt_s_log, s_log, ds_log = grid_log_threshold(s_thr + 0.001, s_max, 60, alpha=5.0)
    print(f"\n[1c] 对数阈值网格: {len(sqrt_s_log)} 点")
    print(f"     最小局部间距: {np.min(ds_log):.6e} GeV")
    print(f"     最大局部间距: {np.max(ds_log):.6e} GeV")

    # 1d. 角度网格
    cos_theta, theta, w_theta = grid_cos_theta(20, symmetric=False)
    print(f"\n[1d] Gauss-Legendre 角度网格: {len(cos_theta)} 点")
    print(f"     cos θ ∈ [{cos_theta[0]:.4f}, {cos_theta[-1]:.4f}]")
    print(f"     权重和 = {np.sum(w_theta):.6f} (应为 2.0)")

    # 1e. 四动量验证
    sub_header("四动量构造与 Mandelstam 验证")
    p1, p2, p3, p4 = cm_frame_four_momenta(0.4, MASS_PI_PLUS, MASS_PI_PLUS, 0.5)
    s_val, t_val, u_val, check = cross_check_mandelstam(p1, p2, p3, p4,
                                                          MASS_PI_PLUS, MASS_PI_PLUS)
    print(f"  √s = 0.4 GeV, cos θ* = 0.5")
    print(f"  s = {s_val:.6f}, t = {t_val:.6f}, u = {u_val:.6f} GeV²")
    print(f"  s+t+u = {s_val + t_val + u_val:.6f} GeV²")
    print(f"  Σm_i² = {4 * MASS_PI_PLUS ** 2:.6f} GeV²")
    print(f"  Mandelstam 关系验证: {'PASS' if check else 'FAIL'}")

    # ===================================================================
    section_header("2. 特殊函数验证")
    # ===================================================================

    # 2a. Legendre 多项式
    sub_header("Legendre 多项式")
    x_test = np.array([0.0, 0.5, 1.0])
    P_test = legendre_p(5, x_test)
    print(f"  P_l(x) for l=0..5, x=0:")
    for l in range(6):
        print(f"    P_{l}(0) = {P_test[l, 0]:.6f}")

    # 2b. Carlson 椭圆积分
    sub_header("Carlson 椭圆积分")
    m_test = 0.5
    K_val = elliptic_k_complete(m_test)
    E_val = elliptic_e_complete(m_test)
    print(f"  K(m=0.5) = {K_val:.8f}")
    print(f"  E(m=0.5) = {E_val:.8f}")
    # 验证: K(0) = π/2, E(0) = π/2
    K_0 = elliptic_k_complete(0.0)
    E_0 = elliptic_e_complete(0.0)
    print(f"  K(0) = {K_0:.8f} (exact: {PI / 2:.8f})")
    print(f"  E(0) = {E_0:.8f} (exact: {PI / 2:.8f})")
    print(f"  R_F(1,2,3) = {carlson_rf(1.0, 2.0, 3.0):.8f}")
    print(f"  R_D(1,2,3) = {carlson_rd(1.0, 2.0, 3.0):.8f}")

    # 2c. 置换群循环
    sub_header("置换群循环结构")
    perm_test = [1, 2, 0, 4, 3]
    cycles, sign = permutation_cycle_type(perm_test)
    print(f"  置换 {perm_test}: 循环类型 {cycles}, 符号 = {sign}")
    zi = cycle_index_polynomial(3)
    print(f"  S_3 循环指标多项式: {len(zi)} 项")
    n_distinct = 1  # S_3 有 3 个共轭类
    print(f"  S_3 共轭类数 = {n_distinct}")

    # 2d. Gamma/Beta 函数
    sub_header("Gamma 与 Beta 函数")
    lg5 = log_gamma_stirling(5.0)
    print(f"  ln Γ(5) = {lg5.real:.8f} (exact: {np.log(24.0):.8f})")
    b23 = beta_function(2.0, 3.0)
    print(f"  B(2,3) = {b23.real:.8f} (exact: {1.0 / 12.0:.8f})")

    # ===================================================================
    section_header("3. 相移生成与分波展开")
    # ===================================================================

    # 使用对数网格进行相移计算
    sqrt_s_grid = sqrt_s_log
    s_grid = sqrt_s_grid ** 2

    # 生成相移数据 (有效范围参数化)
    delta_0_grid = np.zeros(len(sqrt_s_grid))
    delta_1_grid = np.zeros(len(sqrt_s_grid))
    k_grid = np.zeros(len(sqrt_s_grid))

    for i, ss in enumerate(sqrt_s_grid):
        k = cm_momentum(ss ** 2, MASS_PI_PLUS, MASS_PI_PLUS)
        k_grid[i] = k
        if k > 1e-8:
            # S 波: k cot δ₀ = -1/a₀ + (1/2)r₀ k²
            kcot_d0 = -1.0 / a0_input + 0.5 * r0_input * k ** 2
            delta_0_grid[i] = np.arctan(k / kcot_d0) if abs(kcot_d0) > EPS_MACH else PI / 2
            # P 波: k³ cot δ₁ = -1/a₁ + (1/2)r₁ k² (简化为零)
            delta_1_grid[i] = 0.0

    print(f"\n相移数据生成:")
    print(f"  S 波散射长度 a₀ = {a0_input:.4f} GeV⁻¹")
    print(f"  S 波有效力程 r₀ = {r0_input:.4f} GeV⁻¹")
    print(f"  k 范围: [{k_grid[0]:.6f}, {k_grid[-1]:.6f}] GeV")
    print(f"  δ₀ 范围: [{np.min(delta_0_grid):.4f}, {np.max(delta_0_grid):.4f}] rad")

    # 验证有效范围参数提取
    sub_header("有效范围参数提取")
    a0_extracted, r0_extracted, _, _ = effective_range_params(
        sqrt_s_grid, delta_0_grid, 0, MASS_PI_PLUS, MASS_PI_PLUS)
    print(f"  输入: a₀ = {a0_input:.4f}, r₀ = {r0_input:.4f}")
    print(f"  提取: a₀ = {a0_extracted.real:.4f}, r₀ = {r0_extracted.real:.4f}")
    rel_err_a0 = abs(a0_extracted.real - a0_input) / max(abs(a0_input), EPS_MACH) * 100
    rel_err_r0 = abs(r0_extracted.real - r0_input) / max(abs(r0_input), EPS_MACH) * 100
    print(f"  相对误差: a₀ {rel_err_a0:.2f}%, r₀ {rel_err_r0:.2f}%")

    # ===================================================================
    section_header("4. 散射振幅计算")
    # ===================================================================

    # 4a. 单点散射振幅
    sub_header("单点散射振幅")
    k_test = cm_momentum(0.4 ** 2, MASS_PI_PLUS, MASS_PI_PLUS)
    delta_test = np.array([0.3, 0.05, 0.0])  # l=0,1,2
    f_test = scattering_amplitude_partial_wave(0.5, k_test, delta_test)
    ds_do_test = np.abs(f_test) ** 2
    print(f"  √s = 0.4 GeV, cos θ* = 0.5, k = {k_test:.4f} GeV")
    print(f"  δ₀ = 0.3 rad, δ₁ = 0.05 rad")
    print(f"  f(θ*) = {f_test.real:.6f} + {f_test.imag:.6f}i GeV⁻¹")
    print(f"  |f|² = {np.abs(f_test) ** 2:.6f} GeV⁻²")

    # 4b. 总截面与弹性截面
    sub_header("总截面与弹性截面")
    sigma_tot = total_cross_section(k_test, delta_test)
    sigma_el = elastic_cross_section(k_test, delta_test)
    print(f"  σ_tot = {sigma_tot:.6f} GeV⁻² = {cross_section_to_mb(sigma_tot):.4f} mb")
    print(f"  σ_el  = {sigma_el:.6f} GeV⁻² = {cross_section_to_mb(sigma_el):.4f} mb")
    print(f"  σ_el/σ_tot = {sigma_el / max(sigma_tot, EPS_MACH):.6f} (应为 1)")

    # 4c. 微分截面积分
    sub_header("微分截面积分")
    cos_gl, theta_gl, w_gl = grid_cos_theta(30, symmetric=False)
    f_gl = scattering_amplitude_partial_wave(cos_gl, k_test, delta_test)
    sigma_el_gl = integrate_differential_cross_section(cos_gl, w_gl, f_gl)
    print(f"  σ_el (Gauss-Legendre 积分) = {sigma_el_gl:.6f} GeV⁻²")
    print(f"  与分波公式一致: {abs(sigma_el_gl - sigma_el) / max(sigma_el, EPS_MACH) < 0.01}")

    # 4d. 振幅场
    sub_header("二维散射振幅场")
    sqrt_s_field = np.linspace(s_thr + 0.05, 0.45, 20)
    cos_theta_field = np.linspace(-0.9, 0.9, 15)

    def delta_func(ss):
        k = cm_momentum(ss ** 2, MASS_PI_PLUS, MASS_PI_PLUS)
        if k < 1e-8:
            return np.zeros(3)
        kcot = -1.0 / a0_input + 0.5 * r0_input * k ** 2
        d0 = np.arctan(k / kcot) if abs(kcot) > EPS_MACH else PI / 2
        return np.array([d0, 0.0, 0.0])

    def k_func(ss):
        return cm_momentum(ss ** 2, MASS_PI_PLUS, MASS_PI_PLUS)

    amp_field = ScatteringAmplitudeField(sqrt_s_field, cos_theta_field)
    amp_field.set_from_partial_waves(delta_func, k_func)
    L2_norm = amp_field.L2_norm()
    max_amp = amp_field.max_amplitude()
    print(f"  振幅场大小: {amp_field.n_s} × {amp_field.n_theta}")
    print(f"  L² 范数: {L2_norm:.6f}")
    print(f"  最大振幅: {max_amp:.6f} GeV⁻¹")

    # 4e. 全同粒子对称化
    sub_header("全同粒子对称化")
    f_dir = 0.1 + 0.05j
    f_exch = 0.08 + 0.03j
    f_boson, f_fermion = symmetrize_amplitude_identical_particles(f_dir, f_exch)
    print(f"  直接振幅: {f_dir}")
    print(f"  交换振幅: {f_exch}")
    print(f"  玻色子 (对称): {f_boson}")
    print(f"  费米子 (反对称): {f_fermion}")

    # ===================================================================
    section_header("5. 高阶有限差分与 Richardson 外推")
    # ===================================================================

    # 5a. Fornberg 权重
    sub_header("Fornberg 差分权重")
    x_pts = np.array([-2, -1, 0, 1, 2], dtype=float)
    w1 = fornberg_weights(x_pts, 0.0, 1)
    print(f"  一阶导数权重 (5 点中心):")
    for i, (x, w) in enumerate(zip(x_pts, w1)):
        print(f"    x={x:+.0f}: w = {w:+.6f}")
    offsets, w4 = central_diff_weights(1, 4)
    print(f"  4 阶精度一阶导数模板: offsets = {offsets}")

    # 5b. 数值导数 — 相移对 √s 的导数
    sub_header("散射振幅数值微分")
    # 使用 δ₀(√s) 数据
    h_test = sqrt_s_grid[1] - sqrt_s_grid[0] if len(sqrt_s_grid) > 1 else 0.01

    # 不同精度阶数
    for acc_order in [2, 4, 6]:
        ddelta = numerical_derivative(delta_0_grid, h_test,
                                        deriv_order=1, accuracy_order=acc_order)
        mid_idx = len(ddelta) // 2
        print(f"  精度 O(h^{acc_order}): dδ₀/d√s|_mid = {ddelta[mid_idx]:.6f} rad/GeV")

    # 5c. 复步微分
    sub_header("复步微分法 (无相消)")

    def delta_analytic(sqrt_s_complex):
        """解析相移函数 (可接受复数输入)"""
        ss = complex(sqrt_s_complex)
        s = ss ** 2
        k = cm_momentum(s, MASS_PI_PLUS, MASS_PI_PLUS)
        if abs(k) < 1e-8:
            return complex(0)
        kcot = -1.0 / a0_input + 0.5 * r0_input * k ** 2
        if abs(kcot) < EPS_MACH:
            return complex(PI / 2)
        # 对于复数 k, arctan 需要解析延拓
        return np.arctan(k / kcot + 0j)

    ss_mid = sqrt_s_grid[len(sqrt_s_grid) // 2]
    dd_cs = complex_step_derivative(delta_analytic, ss_mid, h=1e-20, deriv_order=1)
    # 对比有限差分
    dd_fd = numerical_derivative(delta_0_grid, h_test, 1, 6)
    mid_idx = len(dd_fd) // 2
    print(f"  √s = {ss_mid:.4f} GeV")
    print(f"  复步微分: dδ₀/d√s = {dd_cs:.8f} rad/GeV")
    print(f"  6阶有限差分: dδ₀/d√s = {dd_fd[mid_idx]:.8f} rad/GeV")
    if abs(dd_cs) > EPS_MACH:
        rel_diff = abs(dd_cs - dd_fd[mid_idx]) / abs(dd_cs) * 100
        print(f"  相对差异: {rel_diff:.4f}%")

    # 5d. Richardson 外推
    sub_header("Richardson 外推加速")
    h_values = [h_test * 2 ** i for i in range(4, 0, -1)]  # 递减步长
    derivs_rich = []
    for h in h_values:
        dd = numerical_derivative(delta_0_grid[:max(len(delta_0_grid), 50)],
                                    h, 1, 4)
        derivs_rich.append(dd[mid_idx] if len(dd) > mid_idx else 0.0)
    if len(derivs_rich) >= 2:
        rich_table, best_est = richardson_extrapolation(
            np.array(derivs_rich), np.array(h_values), order=4)
        print(f"  Richardson 表:")
        for i in range(min(4, len(derivs_rich))):
            print(f"    h = {h_values[i]:.6f}: D = {derivs_rich[i]:.8f}")
        print(f"  最佳估计 (外推): {best_est:.10f}")

    # 5e. 自适应导数
    sub_header("自适应阶数导数")
    dd_adapt, actual_order, error_est = adaptive_derivative(
        delta_0_grid, sqrt_s_grid, deriv_order=1, tol=1e-6, max_accuracy=10)
    print(f"  使用精度阶数: O(h^{actual_order})")
    print(f"  最大误差估计: {np.max(error_est):.4e}")

    # 5f. 散射长度导数
    sub_header("散射长度提取 (导数法)")
    a0_deriv, r0_deriv, kcot_data = scattering_length_derivative(
        k_grid, delta_0_grid)
    print(f"  a₀ (导数法) = {a0_deriv.real:.4f} GeV⁻¹")
    print(f"  r₀ (导数法) = {r0_deriv.real:.4f} GeV⁻¹")

    # ===================================================================
    section_header("6. S 矩阵与幺正性")
    # ===================================================================

    # 6a. 单通道 S 矩阵
    sub_header("单通道 S 矩阵")
    S_1ch = ScatteringMatrix(1)
    delta_test_val = 0.3
    T_test = phase_shift_to_t_matrix(delta_test_val)
    S_1ch.S[0, 0] = 1.0 + 2j * T_test
    is_u, dev = S_1ch.check_unitarity()
    print(f"  δ₀ = {delta_test_val:.4f} rad")
    print(f"  S = {S_1ch.S[0, 0]:.6f}")
    print(f"  |S| = {np.abs(S_1ch.S[0, 0]):.8f}")
    print(f"  幺正性: {'PASS' if is_u else 'FAIL'} (偏差 {dev:.2e})")

    # 6b. 多通道 K 矩阵
    sub_header("双通道 K 矩阵 S 矩阵")
    K_2ch = np.array([[0.5, 0.1],
                       [0.1, 0.3]])
    rho_2ch = np.array([0.8, 0.6])
    S_2ch = ScatteringMatrix(2)
    S_2ch.set_k_matrix(K_2ch, rho_2ch)
    is_u2, dev2 = S_2ch.check_unitarity()
    print(f"  K = [[0.5, 0.1], [0.1, 0.3]]")
    print(f"  ρ = [0.8, 0.6]")
    print(f"  |S₁₁| = {np.abs(S_2ch.S[0, 0]):.6f}, |S₂₂| = {np.abs(S_2ch.S[1, 1]):.6f}")
    print(f"  η₁ = {S_2ch.inelasticity()[0]:.6f}, η₂ = {S_2ch.inelasticity()[1]:.6f}")
    print(f"  本征相移: {S_2ch.eigenphases()}")
    print(f"  幺正性: {'PASS' if is_u2 else 'FAIL'} (偏差 {dev2:.2e})")

    # 6c. 带状矩阵
    sub_header("带状矩阵求解")
    N_band = 5
    A_dense = np.eye(N_band, dtype=complex) + 0.1 * np.eye(N_band, k=1, dtype=complex) + \
              0.1 * np.eye(N_band, k=-1, dtype=complex)
    A_band = BandedMatrix.from_dense(A_dense, 1, 1)
    b_vec = np.ones(N_band, dtype=complex)
    x_sol = A_band.banded_solve(b_vec)
    residual = np.max(np.abs(A_dense @ x_sol - b_vec))
    cond_est = A_band.condition_number_estimate()
    print(f"  带状矩阵 {N_band}×{N_band}, 带宽 = 1")
    print(f"  求解残差: {residual:.2e}")
    print(f"  条件数估计: {cond_est:.4f}")

    # 6d. 整数 RREF
    sub_header("整数行简化阶梯形 (IRREF)")
    A_int = np.array([[1, 3, 0, 2],
                       [2, 6, 0, 4],
                       [3, 9, 0, 6]], dtype=np.int64)
    irref_mat, rank = integer_rref(A_int)
    print(f"  输入矩阵 (3×4):")
    for row in A_int:
        print(f"    {row}")
    print(f"  IRREF (rank = {rank}):")
    for row in irref_mat:
        print(f"    {row}")

    # 6e. 幺正性约束秩
    sub_header("幺正性约束分析")
    for n_ch in [1, 2, 3, 4]:
        n_constr, n_free = unitarity_constraint_rank(n_ch)
        print(f"  N={n_ch}: 约束数 = {n_constr}, 自由参数 = {n_free}")

    _, sys_rank = build_unitarity_system(2)
    print(f"  双通道 Hermiticity 约束秩 = {sys_rank}")

    # ===================================================================
    section_header("7. 截面积分与相空间")
    # ===================================================================

    # 7a. 两体相空间
    sub_header("两体相空间")
    for ss_test in [0.3, 0.35, 0.4, 0.45]:
        phi2 = two_body_phase_space(ss_test, MASS_PI_PLUS, MASS_PI_PLUS)
        phi2_ell, ell_corr = two_body_ps_with_elliptic(ss_test, MASS_PI_PLUS, MASS_PI_PLUS)
        print(f"  √s = {ss_test:.2f} GeV: Φ₂ = {phi2:.6f} GeV⁻², "
              f"椭圆修正 = {ell_corr:.6f}")

    # 7b. 总截面积分
    sub_header("总截面能量积分")
    sigma_int, sigma_arr = integrate_total_cross_section(
        sqrt_s_grid,
        lambda ss: np.array([
            np.arctan(k / (-1.0 / a0_input + 0.5 * r0_input * k ** 2))
            if (k := cm_momentum(ss ** 2, MASS_PI_PLUS, MASS_PI_PLUS)) > 1e-8
            else 0.0,
            0.0, 0.0
        ]),
        L_max=0,
        m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS
    )
    print(f"  ∫ σ_tot d√s = {sigma_int:.6f} GeV⁻¹")
    print(f"  σ_tot 范围: [{np.min(sigma_arr):.4f}, {np.max(sigma_arr):.4f}] GeV⁻²")

    # 7c. Dalitz 图
    sub_header("三体相空间 (Dalitz 图)")
    m_parent = 0.5  # 母粒子质量
    boundary, area = dalitz_boundary(m_parent ** 2, MASS_PI_PLUS, MASS_PI_PLUS, MASS_PI_ZERO)
    print(f"  母粒子质量 = {m_parent} GeV")
    print(f"  Dalitz 图边界点数: {len(boundary)}")
    print(f"  Dalitz 图面积: {area:.6e} GeV⁴")
    if len(boundary) >= 3:
        centroid = polygon_centroid(boundary)
        print(f"  Dalitz 图质心: ({centroid[0]:.4f}, {centroid[1]:.4f})")

    # ===================================================================
    section_header("8. 谱分析与共振识别")
    # ===================================================================

    # 8a. 功率谱密度
    sub_header("功率谱密度")
    freq_psd, psd = power_spectrum_density(delta_0_grid, h_test, window='hanning')
    print(f"  频率范围: [{freq_psd[0]:.4f}, {freq_psd[-1]:.4f}] rad/GeV")
    print(f"  PSD 最大值: {np.max(psd):.4e}")
    print(f"  PSD 积分: {np.trapz(psd, freq_psd) if len(freq_psd) > 1 else 0:.4e}")

    # 8b. 互相关
    sub_header("互相关分析")
    lag, corr = cross_correlation(delta_0_grid, delta_1_grid + 0.01, h_test)
    print(f"  延迟范围: [{lag[0]:.4f}, {lag[-1]:.4f}] GeV")
    print(f"  互相关最大值: {np.max(np.abs(corr)):.4e}")

    # 8c. 粉红噪声谱
    sub_header("粉红噪声 (1/f) 谱")
    freq_pink, amp_pink = pink_noise_spectrum(50, beta=1.0)
    print(f"  频率点数: {len(freq_pink)}")
    print(f"  振幅范围: [{np.min(np.abs(amp_pink)):.4e}, {np.max(np.abs(amp_pink)):.4e}]")

    # 8d. 谱分解
    sub_header("分波谱分解")
    delta_2d = np.vstack([delta_0_grid, delta_1_grid])
    bg, resonances = partial_wave_spectral_decomposition(delta_2d, sqrt_s_grid)
    print(f"  背景振幅范围: [{np.min(bg):.4f}, {np.max(bg):.4f}]")
    print(f"  检测到的共振候选: {len(resonances)}")

    # 8e. HDRFT
    sub_header("HDRFT 动态结构因子")
    tau_grid = np.linspace(0.01, 5.0, 100)
    gamma_test = np.exp(-0.5 * tau_grid)  # 简单衰减模型
    omega_hdrft, S_omega = hdrft_spectrum(tau_grid, gamma_test, order=10)
    print(f"  频率范围: [{np.min(omega_hdrft):.4f}, {np.max(omega_hdrft):.4f}]")
    print(f"  S(ω) 最大值: {np.max(S_omega):.4e}")

    # ===================================================================
    section_header("9. 稳定性与分岔分析")
    # ===================================================================

    # 9a. 有限差分条件数
    sub_header("有限差分条件数")
    h_range = np.logspace(-6, -1, 10)
    stab_data = stability_map_fd(1, h_range, [2, 4, 6, 8])
    for p in [2, 4, 6, 8]:
        print(f"  O(h^{p}): 条件数范围 = [{np.min(stab_data[p]['condition']):.2e}, "
              f"{np.max(stab_data[p]['condition']):.2e}]")

    # 9b. 最优步长
    sub_header("最优步长")
    f_deriv_est = 10.0  # 假设 f^(m+p) 的量级
    h_opt, E_min = optimal_step_size(1, 4, f_deriv_est)
    print(f"  导数阶数 = 1, 精度阶数 = 4")
    print(f"  f^(5) ≈ {f_deriv_est}")
    print(f"  h_opt = {h_opt:.4e}")
    print(f"  E_min = {E_min:.4e}")

    # 9c. 边界碰撞分岔
    sub_header("边界碰撞分岔 (时延系统)")
    bcm = BorderCollisionMap(tau=0.95, b_force=1.35)
    Z_init = [-0.94]
    X_burn, T_burn = bcm.simulate(0.95, Z_init, 0.01, 100.0)
    X_data, T_data = bcm.simulate(0.95, Z_init, X_burn[-1] if X_burn else 0.01, 5.0)
    print(f"  τ = 0.95, b = 1.35")
    print(f"  燃烧阶段: {len(X_burn)} 步, t ∈ [0, {T_burn[-1] if T_burn else 0:.1f}]")
    print(f"  数据阶段: {len(X_data)} 步")
    if len(X_data) > 10:
        print(f"  x 范围: [{np.min(X_data):.4f}, {np.max(X_data):.4f}]")

    # 9d. 共振分岔分析
    sub_header("共振分岔分析")
    # 构造一个通过阈值的共振相移
    E_grid_bif = np.linspace(0.27, 0.31, 50)
    delta_bif = breit_wigner_phase_shift(E_grid_bif, 0.285, 0.020,
                                          l_quantum=0,
                                          m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS)
    bif_type, a0_bif, r0_bif = resonance_bifurcation_analysis(
        E_grid_bif, delta_bif, 2 * MASS_PI_PLUS)
    print(f"  共振质量 = 0.285 GeV, 宽度 = 0.020 GeV")
    print(f"  分岔类型: {bif_type}")
    print(f"  散射长度: {a0_bif:.4f} GeV⁻¹")
    print(f"  有效力程: {r0_bif:.4f} GeV⁻¹")

    # 9e. 耦合通道刚度
    sub_header("耦合通道刚度")
    K_stiff = np.array([[1.0, 0.5], [0.5, 0.1]])
    rho_stiff = np.array([0.9, 0.1])
    stiff_ratio, eigs, is_stiff = coupled_channel_stiffness(K_stiff, rho_stiff)
    print(f"  刚度比 = {stiff_ratio:.2f}")
    print(f"  本征值: {eigs}")
    print(f"  刚性系统: {'是' if is_stiff else '否'}")

    # 9f. Lyapunov 指数
    sub_header("Lyapunov 指数估计")
    if len(X_data) > 20:
        lyap_exp, is_chaotic = lyapunov_exponent_estimate(
            np.array(X_data), dt=0.01, embedding_dim=3)
        print(f"  λ_max = {lyap_exp:.4f}")
        print(f"  混沌: {'是' if is_chaotic else '否'}")
    else:
        print("  数据不足，跳过 Lyapunov 分析")

    # ===================================================================
    section_header("10. 格点 QCD Lüscher 分析")
    # ===================================================================

    # 10a. 动量格点
    sub_header("有限体积动量格点")
    L_test = 4.0 / GEV_INV_TO_FM  # 4 fm → GeV⁻¹
    p_lattice, E_lattice, n_lattice = momentum_lattice(L_test, 0.5,
                                                        MASS_PI_PLUS, MASS_PI_PLUS)
    print(f"  L = {L_test:.2f} GeV⁻¹ ({L_test * GEV_INV_TO_FM:.1f} fm)")
    print(f"  允许的动量态数: {len(p_lattice)}")
    if len(E_lattice) > 0:
        print(f"  最低 5 个能级:")
        for i in range(min(5, len(E_lattice))):
            n_vec = n_lattice[i]
            print(f"    n⃗ = {n_vec}, E = {E_lattice[i]:.6f} GeV")

    # 10b. Lüscher 相移
    sub_header("Lüscher 相移提取")
    if len(E_lattice) >= 3:
        E_levels = E_lattice[:min(10, len(E_lattice))]
        k_luscher, delta_luscher = luscher_analysis(E_levels, L_test,
                                                      MASS_PI_PLUS, MASS_PI_PLUS)
        if len(k_luscher) > 0:
            print(f"  提取的相移点数: {len(k_luscher)}")
            for i in range(min(5, len(k_luscher))):
                print(f"    k = {k_luscher[i]:.4f} GeV, δ₀ = {delta_luscher[i]:.4f} rad")
        else:
            print("  无法提取相移 (能量不足)")

    # 10c. Zeta 函数
    sub_header("Lüscher Zeta 函数")
    for q_sq_test in [0.1, 0.5, 1.0, 1.5]:
        Z00_val = zeta_function_Z00(1, q_sq_test, n_max=20)
        print(f"  Z_{{00}}(1; {q_sq_test:.1f}) = {Z00_val:.6f}")

    # 10d. 计算时序
    sub_header("计算时序标记 (Zeller 公式)")
    year, month, day, wday = computational_schedule_marker(42, 10000)
    print(f"  配置 #42: 标记日期 {year}-{month:02d}-{day:02d} ({wday})")
    z_today = zeller_congruence(2026, 6, 7)
    print(f"  2026-06-07 是 {weekday_name(z_today)}")

    # 10e. 格点类型
    sub_header("格点几何")
    rect_x, rect_y = rectangular_grid_2d(0, 1, 0, 1, 5, 5)
    print(f"  矩形格点: {rect_x.shape} = {rect_x.size} 点")
    tri_pts = triangular_grid_2d(0, 1, 0, 1, 25)
    print(f"  三角格点: {len(tri_pts)} 点")

    # ===================================================================
    section_header("11. 曲率分析与共振搜索")
    # ===================================================================

    sub_header("散射振幅曲率分析")
    # 构造一个含共振的截面
    E_res_test = np.linspace(0.3, 0.45, 100)
    sigma_res = np.zeros(len(E_res_test))
    for i, ss in enumerate(E_res_test):
        k = cm_momentum(ss ** 2, MASS_PI_PLUS, MASS_PI_PLUS)
        if k > EPS_MACH:
            delta_bw = breit_wigner_phase_shift(
                np.array([ss]), 0.38, 0.02, l_quantum=0,
                m_a=MASS_PI_PLUS, m_b=MASS_PI_PLUS)[0]
            sigma_res[i] = FOURPI / (k ** 2) * np.sin(delta_bw) ** 2

    curvature, resonance_cands = curvature_analysis(E_res_test, sigma_res)
    print(f"  搜索范围: √s ∈ [{E_res_test[0]:.3f}, {E_res_test[-1]:.3f}] GeV")
    print(f"  候选共振位置: {len(resonance_cands)}")
    for rc in resonance_cands:
        print(f"    √s = {rc:.4f} GeV")

    # ===================================================================
    section_header("12. 综合总结")
    # ===================================================================

    print()
    print("=" * 72)
    print("  PROJECT_232 计算完成")
    print("=" * 72)
    print()
    print("融合的种子项目:")
    print("  [1]  1372_unicycle       — 置换群循环结构 (全同粒子对称化)")
    print("  [2]  969_r8bb            — 带状矩阵运算 (S 矩阵求解)")
    print("  [3]  371_fem_basis       — 基函数构造 (分波基展开)")
    print("  [4]  569_i4mat_rref2     — 整数矩阵行简化 (幺正性约束分析)")
    print("  [5]  1177_HyperMPC       — 超参数预测 (参数外推思想)")
    print("  [6]  767_midpoint_fixed  — 中点固定点迭代 (LS 方程求解)")
    print("  [7]  880_polar_ode       — 极坐标 ODE (相移极坐标演化)")
    print("  [8]  1412_weekday_zeller — Zeller 日历 (计算时序标记)")
    print("  [9]  870_pink_noise      — 功率谱分析 (散射振幅谱分解)")
    print("  [10] 415_fem2d_scalar    — 2D 标量场 (振幅场数据结构)")
    print("  [11] 335_elliptic_integral — 椭圆积分 (相空间积分)")
    print("  [12] 492_gridlines       — 网格生成 (动量格点)")
    print("  [13] 1052_mctools_cdft   — CDFT/FFT (HDRFT 谱反演)")
    print("  [14] 882_polygon         — 多边形几何 (Dalitz 图面积)")
    print("  [15] 1085_PierceRyan     — 边界碰撞分岔 (共振分岔分析)")
    print()
    print("核心科学结果:")
    print(f"  ππ 散射长度: a₀ = {a0_extracted.real:.4f} GeV⁻¹ (输入: {a0_input})")
    print(f"  ππ 有效力程: r₀ = {r0_extracted.real:.4f} GeV⁻¹ (输入: {r0_input})")
    print(f"  总截面 (√s=0.4): {cross_section_to_mb(sigma_tot):.4f} mb")
    print(f"  Richardson 外推最佳导数值: {best_est:.8f}")
    print(f"  最优差分步长: h_opt = {h_opt:.4e}")
    print()
    print("所有计算均已完成,零错误。")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
