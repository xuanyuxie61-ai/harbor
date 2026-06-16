"""
main.py
-------
统一入口: 博士级 B 物理衰变链重建与 CP 破坏分析的高阶有限差分及稳定性研究。
零参数可运行。

完整流程:
  [1] 加载 B 物理常数与 CKM 矩阵参数;
  [2] 构造 B0 → D∓ π± 三体 Dalitz 图的几何与 FEM 网格;
  [3] 用对称求积规则计算 Dalitz 图上的积分 (衰变宽度);
  [4] 构造 isobar 模型振幅 (含共振道 + Legendre 角分布);
  [5] 求解 B0-B0bar 混合时间演化, 提取 sin(2β);
  [6] 对 Dalitz 平面 PDE 做高阶有限差分离散, 进行稳定性分析;
  [7] 构造衰变链转移矩阵 (CRS), 计算级联稳态;
  [8] 枚举量子数守恒的末态 (Diophantine);
  [9] 蒙特卡罗相空间采样, 计算衰变宽度;
 [10] 分形扰动扫描共振参数系统误差;
 [11] 多通道 CVT 接受度优化;
 [12] 输出综合报告 (无可视化)。
"""

from __future__ import annotations
import math
import time
import sys
import os

# 确保本目录在 sys.path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np

from b_physics_constants import (
    M_B0, M_DP, M_D0, M_PI, M_K, M_KS, M_JPSI, M_PHI, M_RHO,
    TAU_B0, TAU_BS, DELTA_M_D, DELTA_M_S, DELTA_GAMMA_D, DELTA_GAMMA_S,
    LAMBDA_W, A_W, RHOBAR_W, ETABAR_W,
    wolfenstein_to_ckm, unitarity_triangle_angles,
    kallen, blatt_weisskopf, breit_wigner_running,
)
# D^- 与 D^+ 质量相同 (CPT), 用同一常量
M_DM = M_DP
from dalitz_geometry import (
    dalitz_boundary, dalitz_extent, DalitzAffineMap,
    build_dalitz_fem_mesh, mesh_statistics, boundary_normal_segments,
)
from high_order_finite_difference import (
    fd_first_deriv_4th, fd_first_deriv_6th, fd_second_deriv_4th,
    fd_2d_laplacian_4th, fd_2d_mixed_4th,
    build_effective_hamiltonian, propagate_rk4, propagate_crank_nicolson,
    von_neumann_advection, von_neumann_diffusion,
    rk4_stability_limit, cfl_time_step,
    solve_dalitz_pde_cn, convergence_order,
)
from isobar_amplitude import (
    legendre_polynomial, legendre_quadrature,
    legendre_exponential_product_table, legendre_linear_product_table,
    IsobarResonance, IsobarAmplitude,
    minimal_surface_catenoid_amplitude, minimal_surface_residual,
    angular_distribution_legendre_coefficients,
)
from dalitz_quadrature import (
    get_symq_rule, integrate_on_dalitz, integrate_dalitz_decay_width,
    triangle_histogram, histogram_chi_squared,
)
from time_evolution_cp import (
    ymdf_to_jed_gregorian, jed_to_b_proper_time,
    time_evolution_untagged, time_evolution_tagged,
    cp_asymmetry_time_dependent,
    lambda_f_from_weak_phases, extract_sin2beta_from_time_data,
    simulate_decay_times,
)
from sparse_decay_io import (
    SparseCRS, build_decay_transition_matrix,
    leontief_inverse, decay_chain_cascade, cascade_steady_state,
    sensitivity_to_branch_ratio, total_particle_yield,
)
from multichannel_cvt import (
    CellArray, build_multichannel_resonance_cells,
    metric_matrix_acceptance, metric_distance,
    cvt_lloyd_iteration, cvt_energy, integrate_channel_cells,
)
from monte_carlo_phase import (
    hypercube_sample, hypercube_monomial_integral, monomial_value,
    dalitz_sample_rejection, monte_carlo_integral,
    decay_width_monte_carlo, monte_carlo_convergence,
)
from diophantine_quantum import (
    diophantine_1d_nonnegative, diophantine_nd_nonnegative,
    enumerate_charge_conserving_final_states, filter_by_mass,
    frobenius_number_estimate, count_decay_modes_by_invariant_mass,
)
from fractal_perturbation import (
    fractal_perturb_closed_curve, fractal_iteration,
    fractal_dimension_estimate, build_resonance_parameter_curve,
    scan_systematic_uncertainty,
    fd_robustness_under_fractal_perturbation,
)


# ========================================================================== #
#                       分隔符工具                                          #
# ========================================================================== #
def _section(title: str) -> None:
    bar = "=" * 76
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def _subsection(title: str) -> None:
    print()
    print(f"-- {title} " + "-" * max(0, 70 - len(title)))


# ========================================================================== #
#                       演示 1: CKM 与么正三角形                          #
# ========================================================================== #
def demo_ckm_unitarity() -> None:
    _section("[1] CKM 矩阵与么正三角形")
    V = wolfenstein_to_ckm()
    print(f"Wolfenstein 参数: lambda={LAMBDA_W:.5f}, A={A_W:.4f}, "
          f"rhobar={RHOBAR_W:.4f}, etabar={ETABAR_W:.4f}")
    print("CKM 矩阵 (近似到 O(lambda^4)):")
    for i in range(3):
        row_str = "  |V_{" + "ud,cd,td"[i] + "}|:  "
        row_str += "  ".join(
            f"{abs(V[i][j]):.5f}  exp(i {math.atan2(V[i][j].imag, V[i][j].real + 1e-30):.4f})"
            for j in range(3)
        )
        print(row_str)
    # 么正性检验: V V^dagger = I
    Varr = np.array(V, dtype=complex)
    VVdag = Varr @ Varr.conj().T
    unitarity_deviation = float(np.linalg.norm(VVdag - np.eye(3)))
    print(f"么正性偏差 ||V V^dagger - I|| = {unitarity_deviation:.3e}")
    alpha, beta, gamma = unitarity_triangle_angles()
    print(f"么正三角形内角: alpha = {alpha:.4f} rad ({math.degrees(alpha):.2f} deg)")
    print(f"                beta  = {beta:.4f} rad ({math.degrees(beta):.2f} deg)")
    print(f"                gamma = {gamma:.4f} rad ({math.degrees(gamma):.2f} deg)")
    print(f"                sum   = {alpha+beta+gamma:.4f} (应为 π = {math.pi:.4f})")


# ========================================================================== #
#                   演示 2: Dalitz 几何与 FEM 网格                          #
# ========================================================================== #
def demo_dalitz_geometry() -> None:
    _section("[2] Dalitz 图几何与 FEM 网格")
    m_parent = M_B0
    m1, m2, m3 = M_DP, M_DM, M_PI   # B0 → D+ D- pi0 (虚拟示例)
    # 实际 B0 → D+ D- pi0 阈值为 M_DP + M_DM + M_PI ≈ 3879, 小于 M_B0 ≈ 5280, OK
    print(f"衰变: B0 → D+ D- pi0  (m_B = {m_parent:.2f} MeV)")
    print(f"     m_1 = {m1:.2f} MeV (D+), m_2 = {m2:.2f} MeV (D-), m_3 = {m3:.2f} MeV (pi0)")
    ext = dalitz_extent(m_parent, m1, m2, m3)
    print(f"  Dalitz 外接矩形: s12 ∈ [{ext[0]:.1f}, {ext[1]:.1f}] MeV^2")
    print(f"                  s13 ∈ [{ext[2]:.1f}, {ext[3]:.1f}] MeV^2")
    boundary = dalitz_boundary(m_parent, m1, m2, m3, n_boundary=60)
    print(f"  边界点数目: {len(boundary)}")
    segs = boundary_normal_segments(boundary)
    print(f"  边界线段数目: {len(segs)}")
    # FEM 网格
    nodes, elements = build_dalitz_fem_mesh(m_parent, m1, m2, m3, n_s12=20, n_s13=20)
    stats = mesh_statistics(nodes, elements)
    print(f"  FEM 网格: {stats['n_node']} 节点, {stats['n_elem']} 三角形单元")
    print(f"    单元面积: min = {stats['min_area']:.3e}, "
          f"mean = {stats['mean_area']:.3e}, max = {stats['max_area']:.3e} MeV^4")
    # 仿射映射检查
    affine = DalitzAffineMap(m_parent, m1, m2, m3)
    s12_test, s13_test = affine.ref_to_dalitz(0.3, 0.4)
    xi_back, eta_back = affine.dalitz_to_ref(s12_test, s13_test)
    print(f"  仿射映射往返测试: (0.3, 0.4) → ({s12_test:.1f}, {s13_test:.1f}) "
          f"→ ({xi_back:.3f}, {eta_back:.3f})")


# ========================================================================== #
#             演示 3: 对称求积规则与 Dalitz 积分                            #
# ========================================================================== #
def demo_dalitz_quadrature() -> None:
    _section("[3] Dalitz 图上的对称求积规则")
    m_parent = M_B0
    m1, m2, m3 = M_DP, M_DM, M_PI
    for deg in range(6):
        x, y, w = get_symq_rule(deg)
        sum_w = w.sum()
        print(f"  degree {deg}: {len(w):2d} 点, Σw = {sum_w:.6f} "
              f"(应为 0.5 = 参考三角形面积)")
    # 常数函数积分: 应返回物理区域面积 × 0.5
    def integrand_unit(s12, s13):
        return 1.0
    for deg in [2, 4, 5]:
        area = integrate_on_dalitz(integrand_unit, m_parent, m1, m2, m3, degree=deg)
        print(f"    degree {deg} 积分 1.0 → 物理面积 ≈ {area:.3e} MeV^4")


# ========================================================================== #
#             演示 4: Isobar 振幅与 Legendre 展开                           #
# ========================================================================== #
def demo_isobar_amplitude() -> None:
    _section("[4] Isobar 振幅与 Legendre 角分布")
    # 构造一个共振道 B0 → rho0(→ pi+ pi-) Ks
    resonances = [
        {
            "name": "rho(770)",
            "m_r": M_RHO,
            "gamma_r": 149.1,
            "J": 1,
            "L_parent": 1,   # B0 → rho Ks, L=1 (P-wave)
            "daug1": M_PI,
            "daug2": M_PI,
            "spectator": M_KS,
            "m_parent": M_B0,
            "a": 1.0,
            "delta": 0.3,
        },
    ]
    res_objs = [IsobarResonance(**resonances[0])]
    iso = IsobarAmplitude(res_objs, A_NR=0.1 + 0.0j, gamma_weak=1.218)
    # 在 s_res = m_rho^2 处计算振幅
    s_test = M_RHO ** 2
    for cos_theta in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        amp_minus = iso.evaluate(s_test, cos_theta, charge_sign=+1)
        amp_plus  = iso.evaluate(s_test, cos_theta, charge_sign=-1)
        cp_asym = iso.cp_asymmetry_direct(s_test, cos_theta)
        print(f"  cosθ = {cos_theta:+.2f}: |A^-|^2 = {abs(amp_minus)**2:.4e}, "
              f"|A^+|^2 = {abs(amp_plus)**2:.4e}, A_CP = {cp_asym:+.4f}")
    # Legendre 乘积表
    print("  Legendre 指数乘积表 (p=3, b=0.1j, 对应弱相位):")
    table = legendre_exponential_product_table(3, 0.1j)
    for i in range(4):
        row_str = "    " + "  ".join(
            f"{table[i,j].real:+.4f}{table[i,j].imag:+.4f}j" for j in range(4)
        )
        print(row_str)
    # 最小曲面参考
    u = np.linspace(-0.5, 0.5, 21)
    v = np.linspace(-0.5, 0.5, 21)
    U, V = np.meshgrid(u, v)
    A_ref, A_u, A_v = minimal_surface_catenoid_amplitude(U, V, a_param=3.0)
    print(f"  最小曲面参考解范围: A ∈ [{A_ref.min():.4f}, {A_ref.max():.4f}]")


# ========================================================================== #
#             演示 5: 时间演化与 CP 不对称                                  #
# ========================================================================== #
def demo_time_evolution() -> None:
    _section("[5] B0-B0bar 混合与时间依赖 CP 不对称")
    # JED 转换测试
    jed0 = ymdf_to_jed_gregorian(2024, 4, 1, 0.0)
    jed1 = ymdf_to_jed_gregorian(2024, 4, 1, 1.0 / 86400.0)   # +1 秒
    delta_t = jed_to_b_proper_time(jed0, jed1)
    print(f"  JED(2024-04-01 00:00:00) = {jed0:.3f}")
    print(f"  +1 秒 → proper_time = {delta_t:.3e} ps (应为 1e12 ps)")
    # 时间演化
    gamma_total = 1.0 / TAU_B0
    lam_f = lambda_f_from_weak_phases(
        phi_mix=-2.0 * math.atan2(ETABAR_W, RHOBAR_W),
        phi_decay=0.0,
    )
    print(f"  lambda_f = {lam_f.real:+.4f} {lam_f.imag:+.4f}j, "
          f"|lambda_f| = {abs(lam_f):.4f}")
    times_plot = np.linspace(0.0, 5.0 * TAU_B0, 25)
    print("  t(ps)  P(B0 unmixed)  P(B0 mixed)  A_CP(t)")
    for t in times_plot[::5]:
        P_unmix, P_mix = time_evolution_untagged(
            t, DELTA_M_D, DELTA_GAMMA_D, gamma_total
        )
        S_f = math.sin(2.0 * math.atan2(ETABAR_W, RHOBAR_W))
        A_cp = cp_asymmetry_time_dependent(t, S_f, 0.0, DELTA_M_D)
        print(f"  {t:5.2f}  {P_unmix:13.4e}  {P_mix:13.4e}  {A_cp:+.4f}")
    # 拟合提取 sin(2β)
    t_data = np.linspace(0.1, 4.0 * TAU_B0, 40).tolist()
    S_true = math.sin(2.0 * math.atan2(ETABAR_W, RHOBAR_W))
    a_data = [cp_asymmetry_time_dependent(t, S_true, 0.0, DELTA_M_D) for t in t_data]
    S_fit, C_fit = extract_sin2beta_from_time_data(t_data, a_data, DELTA_M_D)
    print(f"  拟合: S = {S_fit:+.4f} (真值 {S_true:+.4f}), C = {C_fit:+.4f}")
    # RK4 vs Crank-Nicolson 对比
    H = build_effective_hamiltonian(DELTA_M_D, DELTA_GAMMA_D, phi_mix=0.0)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    t_final = 3.0 * TAU_B0
    n_step = 200
    times_rk4, hist_rk4 = propagate_rk4(H, psi0, t_final, n_step)
    times_cn, hist_cn = propagate_crank_nicolson(H, psi0, t_final, n_step)
    norm_rk4 = float(np.linalg.norm(hist_rk4[-1]))
    norm_cn = float(np.linalg.norm(hist_cn[-1]))
    print(f"  时间传播 {n_step} 步后 ||psi||: RK4 = {norm_rk4:.6f}, "
          f"CN = {norm_cn:.6f} (应接近 1)")
    # 稳定性步长
    dt_max = cfl_time_step(DELTA_M_D, safety=0.8)
    y_max = rk4_stability_limit()
    print(f"  RK4 稳定轴截距 y_max = {y_max:.4f}, "
          f"建议 dt_max = {dt_max:.4f} ps (for Delta m_d = {DELTA_M_D} ps^-1)")


# ========================================================================== #
#              演示 6: 高阶有限差分与稳定性                                #
# ========================================================================== #
def demo_finite_difference_stability() -> None:
    _section("[6] 高阶有限差分格式与 von Neumann 稳定性")
    # 1D 差分: 对 f(x) = sin(k x) 计算导数
    N = 128
    h = 2.0 * math.pi / N
    x = np.arange(N) * h
    k_wave = 3.0
    f = np.sin(k_wave * x)
    df_exact = k_wave * np.cos(k_wave * x)
    ddf_exact = -k_wave ** 2 * np.sin(k_wave * x)
    df4 = fd_first_deriv_4th(f, h)
    df6 = fd_first_deriv_6th(f, h)
    ddf4 = fd_second_deriv_4th(f, h)
    err_4th_1 = float(np.max(np.abs(df4 - df_exact)))
    err_6th_1 = float(np.max(np.abs(df6 - df_exact)))
    err_4th_2 = float(np.max(np.abs(ddf4 - ddf_exact)))
    print(f"  对 f(x) = sin({k_wave}x) 在 N={N} 点:")
    print(f"    4阶一阶导最大误差 = {err_4th_1:.3e}")
    print(f"    6阶一阶导最大误差 = {err_6th_1:.3e}")
    print(f"    4阶二阶导最大误差 = {err_4th_2:.3e}")
    # von Neumann 分析: 平流方程
    c_adv = 1.0
    dt_test = 0.5 * h / c_adv
    G_adv_rk4 = von_neumann_advection(c_adv, h, dt_test, scheme="rk4_fd4")
    G_adv_cn  = von_neumann_advection(c_adv, h, dt_test, scheme="cn_fd2")
    print(f"  von Neumann 平流方程 (c={c_adv}, dt={dt_test:.4f}):")
    print(f"    RK4+FD4:  |G|_max = {G_adv_rk4.max():.4f}, |G|_min = {G_adv_rk4.min():.4f}")
    print(f"    CN+FD2:   |G|_max = {G_adv_cn.max():.4f}, |G|_min = {G_adv_cn.min():.4f}")
    # von Neumann: 扩散方程
    D_diff = 0.1
    dt_diff = 0.4 * h * h / D_diff
    G_diff = von_neumann_diffusion(D_diff, h, dt_diff)
    print(f"  von Neumann 扩散方程 (D={D_diff}, dt={dt_diff:.4f}):")
    print(f"    FE: |G|_max = {np.abs(1 - 4 * D_diff * dt_diff / h**2):.4f}")
    print(f"    CN: |G|_max = {G_diff.max():.4f}")
    # 2D Dalitz PDE: Helmholtz 求解
    print("  2D Dalitz PDE (Helmholtz) Jacobi 迭代求解:")
    N_grid = 30
    source = np.zeros((N_grid, N_grid))
    source[15, 15] = 1.0
    h12 = h13 = 0.1
    D12 = D13 = 1.0
    kappa = 5.0
    A_sol, res_hist = solve_dalitz_pde_cn(
        source, h12, h13, D12, D13, kappa, n_iter=500, tol=1.0e-8
    )
    print(f"    迭代 {len(res_hist)} 次, 最终残差 = {res_hist[-1]:.3e}")
    order_est = convergence_order(res_hist)
    print(f"    收敛速率 ≈ {order_est:.4f}")
    print(f"    解的范围: [{A_sol.min():.4e}, {A_sol.max():.4e}]")


# ========================================================================== #
#            演示 7: 稀疏衰变链与 IO 分析                                  #
# ========================================================================== #
def demo_sparse_decay_io() -> None:
    _section("[7] 稀疏衰变链转移矩阵与 Leontief 逆分析")
    B_mat = build_decay_transition_matrix()
    print(f"  衰变转移矩阵维度: {B_mat.n} × {B_mat.n}")
    print(f"  非零元数目: {B_mat.nnz}, 稀疏度: {B_mat.density():.4f}")
    # 源: 注入 1000 个 B0
    source = np.zeros(B_mat.n)
    source[0] = 1000.0   # B0 index = 0
    # 级联 3 代
    history = decay_chain_cascade(source, B_mat, n_generation=3)
    for k, n_k in enumerate(history):
        nonzero = (n_k > 0.1).sum()
        total = n_k.sum()
        print(f"    第 {k} 代: 总粒子数 = {total:.2f}, 非零种类 = {nonzero}")
    # 稳态
    n_star = cascade_steady_state(source, B_mat)
    print(f"  稳态总粒子数: {n_star.sum():.2f}")
    print(f"  稳态 pip 产额: {total_particle_yield(source, B_mat, 9):.2f}")
    print(f"  稳态 KS 产额: {total_particle_yield(source, B_mat, 14):.2f}")
    # 灵敏度分析
    if B_mat.nnz > 0:
        t0, sens = sensitivity_to_branch_ratio(B_mat, source, channel_idx=0, delta_br=0.001)
        print(f"  分支比灵敏度 (channel 0, δBR=0.001): 基线={t0:.2f}, 灵敏度={sens:+.4f}")


# ========================================================================== #
#            演示 8: 量子数守恒的 Diophantine 枚举                         #
# ========================================================================== #
def demo_diophantine_quantum() -> None:
    _section("[8] 量子数守恒的 Diophantine 末态枚举")
    # McNuggets 问题 (仿 743)
    frob = frobenius_number_estimate([6, 9, 20], max_b=100)
    print(f"  McNuggets Frobenius 数 g(6,9,20) = {frob} (应为 43)")
    # B0 (Q=0, S=0) 的末态枚举
    final_states = enumerate_charge_conserving_final_states(initial_Q=0, initial_S=0)
    print(f"  B0 (Q=0, S=0) 允许末态数 (质量未筛): {len(final_states)}")
    filtered = filter_by_mass(final_states, M_B0, threshold=2000.0)
    print(f"  质量约束后 (Σm ∈ [m_B-2000, m_B] MeV): {len(filtered)} 种")
    for i, state in enumerate(filtered[:5]):
        state_str = " + ".join(f"{n} {p}" for p, n in state.items())
        total_m = sum(
            n * {"pip": M_PI, "pim": M_PI, "pi0": M_PI,
                 "Kp": M_K, "Km": M_K, "KS": M_KS}[p]
            for p, n in state.items()
        )
        print(f"    [{i}] {state_str}  (Σm = {total_m:.1f} MeV)")
    if len(filtered) > 5:
        print(f"    ... 还有 {len(filtered) - 5} 个末态")
    # 多维 Diophantine
    A = [[1, 1, 1], [2, 3, 5]]
    b = [10, 30]
    sols = diophantine_nd_nonnegative(A, b, max_per_var=10)
    print(f"  多维 Diophantine: x_1 + x_2 + x_3 = 10, 2 x_1 + 3 x_2 + 5 x_3 = 30")
    print(f"    解数 = {len(sols)}")
    for s in sols[:5]:
        print(f"      {s}")


# ========================================================================== #
#           演示 9: 蒙特卡罗相空间采样                                      #
# ========================================================================== #
def demo_monte_carlo_phase() -> None:
    _section("[9] 高维相空间蒙特卡罗采样")
    # 单项式积分检查
    for d, exps in [(2, [1, 2]), (3, [1, 1, 1]), (4, [2, 0, 1, 3])]:
        est = hypercube_monomial_integral(d, np.array(exps))
        exact = 1.0
        for e in exps:
            exact /= (e + 1.0)
        print(f"  [0,1]^{d} 单项式 x^{exps}: 解析 = {exact:.6f}")
    # Dalitz 采样
    m_parent = M_B0
    m1, m2, m3 = M_DP, M_DM, M_PI
    pts = dalitz_sample_rejection(m_parent, m1, m2, m3, n_target=1000, seed=42)
    print(f"  Dalitz 采样: 得到 {len(pts)} 个物理点")
    if len(pts) > 0:
        s12_mean = pts[:, 0].mean()
        s13_mean = pts[:, 1].mean()
        print(f"    <s12> = {s12_mean:.1f} MeV^2, <s13> = {s13_mean:.1f} MeV^2")
    # 直方图
    if len(pts) > 0:
        # 转换到参考三角形
        affine = DalitzAffineMap(m_parent, m1, m2, m3)
        ref_pts = np.array([
            affine.dalitz_to_ref(p[0], p[1]) for p in pts
        ])
        _, counts = triangle_histogram(ref_pts, n_subdiv=6)
        n_bins = len(counts)
        n_nonempty = (counts > 0).sum()
        print(f"  直方图 (N=6, 共 {n_bins} bins): 非空 bins = {n_nonempty}")
    # 蒙特卡罗积分: ∫_{[0,1]^2} (x + y) dx dy = 1
    def integrand_2d(xy):
        return xy[:, 0] + xy[:, 1]
    est, se, ess = monte_carlo_integral(integrand_2d, 2, 50000, seed=42)
    print(f"  MC 积分 ∫_[0,1]^2 (x+y) = {est:.4f} ± {se:.4f} (应为 1.0)")


# ========================================================================== #
#          演示 10: 多通道 CVT 接受度优化                                  #
# ========================================================================== #
def demo_multichannel_cvt() -> None:
    _section("[10] 多通道共振与 CVT 接受度优化")
    # 多通道 Cell 数组
    resonances = [
        {"name": "rho(770)", "m_r": M_RHO, "gamma_r": 149.1,
         "s_min": 0.3e6, "s_max": 1.0e6},
        {"name": "phi(1020)", "m_r": M_PHI, "gamma_r": 4.266,
         "s_min": 1.0e6, "s_max": 1.1e6},
        {"name": "J/psi", "m_r": M_JPSI, "gamma_r": 0.0929,
         "s_min": 9.5e6, "s_max": 9.7e6},
    ]
    cells = build_multichannel_resonance_cells(resonances, n_points=30)
    print(cells.print_summary("共振通道 |BW|^2 线形"))
    total = integrate_channel_cells(cells)
    print(f"  总 |BW|^2 积分 = {total:.4f}")
    # CVT: 在 Dalitz 图上优化采样
    m_parent = M_B0
    m1, m2, m3 = M_DP, M_DM, M_PI
    metric_func = lambda s12, s13: metric_matrix_acceptance(
        s12, s13, m_parent, m1, m2, m3
    )
    rng = np.random.default_rng(42)
    n_gen = 15
    n_sample = 80
    ext = dalitz_extent(m_parent, m1, m2, m3)
    gens = np.column_stack([
        ext[0] + (ext[1] - ext[0]) * rng.random(n_gen),
        ext[2] + (ext[3] - ext[2]) * rng.random(n_gen),
    ])
    samples = np.column_stack([
        ext[0] + (ext[1] - ext[0]) * rng.random(n_sample),
        ext[2] + (ext[3] - ext[2]) * rng.random(n_sample),
    ])
    E_before = cvt_energy(gens, samples, metric_func)
    gens_opt = cvt_lloyd_iteration(gens, samples, metric_func, n_iter=5)
    E_after = cvt_energy(gens_opt, samples, metric_func)
    print(f"  CVT 能量: 优化前 = {E_before:.3e}, 优化后 = {E_after:.3e}")
    print(f"  改善比例: {(E_before - E_after) / max(E_before, 1.0e-30) * 100:.2f} %")


# ========================================================================== #
#          演示 11: 分形扰动与鲁棒性                                        #
# ========================================================================== #
def demo_fractal_perturbation() -> None:
    _section("[11] 共振参数分形扰动扫描与 FD 鲁棒性")
    # 构造共振参数曲线
    resonances_param = [
        {"m_r": M_RHO, "gamma_r": 149.1, "a": 1.0, "delta": 0.3},
        {"m_r": M_PHI, "gamma_r": 4.266, "a": 0.5, "delta": 0.1},
    ]
    initial_curve = build_resonance_parameter_curve(resonances_param)
    history = fractal_iteration(initial_curve, mu=0.08, n_iter=3, seed=42)
    print(f"  初始曲线顶点数: {len(history[0])}")
    for k, curve in enumerate(history):
        D = fractal_dimension_estimate(curve)
        print(f"    迭代 {k}: {len(curve)} 顶点, 分形维数 ≈ {D:.4f}")
    # 系统误差扫描
    def obs_amp(params):
        # 简化: 返回所有共振的 a 之和
        return sum(p["a"] for p in params)
    scan_result = scan_systematic_uncertainty(
        resonances_param, obs_amp,
        mu_scan=0.08, n_iter=3, n_sample_obs=50,
    )
    print(f"  分形扫描结果: 均值 = {scan_result['mean']:.4f}, "
          f"标准差 = {scan_result['std']:.4f}")
    print(f"                 min = {scan_result['min']:.4f}, "
          f"max = {scan_result['max']:.4f}")
    print(f"                 点数 = {scan_result['n_points']}")
    # FD 鲁棒性
    def laplacian_op(F):
        h = 0.1
        return fd_2d_laplacian_4th(F, h, h, 1.0, 1.0)
    exact = np.zeros((30, 30))
    exact[15, 15] = 1.0
    mu_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    robustness = fd_robustness_under_fractal_perturbation(
        laplacian_op, exact, mu_values, noise_level=0.01
    )
    print("  有限差分鲁棒性 (相对残差 vs 扰动强度 mu):")
    for mu, rel_res, pert_norm in robustness:
        print(f"    mu = {mu:.3f}: ||r||/||p|| = {rel_res:.3e}, ||p|| = {pert_norm:.3e}")


# ========================================================================== #
#                            main()                                         #
# ========================================================================== #
def main() -> None:
    t0 = time.time()
    print("=" * 76)
    print("  博士级 B 物理衰变链重建与 CP 破坏分析")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 76)
    print()
    print("  统一入口: 零参数可运行")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy:  {np.__version__}")

    demo_ckm_unitarity()
    demo_dalitz_geometry()
    demo_dalitz_quadrature()
    demo_isobar_amplitude()
    demo_time_evolution()
    demo_finite_difference_stability()
    demo_sparse_decay_io()
    demo_diophantine_quantum()
    demo_monte_carlo_phase()
    demo_multichannel_cvt()
    demo_fractal_perturbation()

    _section("综合结论")
    elapsed = time.time() - t0
    print(f"  全流程完成, 耗时 {elapsed:.3f} 秒。")
    print()
    print("  本项目完整实现了 B 物理衰变链重建与 CP 破坏分析的计算框架,")
    print("  包括:")
    print("    - CKM 矩阵与么正三角形参数化;")
    print("    - Dalitz 图几何、FEM 网格、对称求积规则;")
    print("    - Isobar 模型振幅 (含 Legendre 角分布、最小曲面参考);")
    print("    - B0-B0bar 混合时间演化与 CP 不对称提取;")
    print("    - 高阶有限差分格式与 von Neumann 稳定性分析;")
    print("    - 稀疏衰变链转移矩阵与 Leontief 逆稳态分析;")
    print("    - 量子数守恒 Diophantine 末态枚举;")
    print("    - 蒙特卡罗相空间采样;")
    print("    - 多通道共振 CVT 接受度优化;")
    print("    - 共振参数分形扰动与 FD 鲁棒性分析。")
    print()
    print("  所有数值结果均在小规模实验设置下可复现, 适合作为博士级")
    print("  计算高能物理方法论教学与研究的参考实现。")
    print()
    print("=" * 76)


if __name__ == "__main__":
    main()
