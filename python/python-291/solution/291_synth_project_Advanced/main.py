#!/usr/bin/env python3
"""
main.py
=======
等离子体鞘层高阶有限差分与稳定性分析 —— 统一入口

本项目研究磁化等离子体鞘层-壁面相互作用的高阶数值模拟与稳定性分析。
融合 15 个种子项目的核心算法，构建一套完整的博士级计算等离子体工具链。

运行方式：
    python main.py  （零参数即可运行）

科学问题：
    一维磁化等离子体鞘层 (Ar 等离子体)
    - 壁面偏压 -30V，含二次电子发射
    - 磁场以 15° 角入射
    - 高阶紧致有限差分离散 (4阶/6阶)
    - 非线性 Poisson-Boltzmann 自洽求解
    - 谱稳定性分析与 von Neumann 条件
    - Vlasov 动理学验证

作者：计算等离子体研究组
日期：2026
"""

import sys
import os
import time
import math
import numpy as np
from scipy import linalg as la

# 确保本目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sheath_constants import PlasmaParams, debye_length, coulomb_logarithm, E_CHARGE
from sheath_grid import (
    geometric_grid, tanh_clustering, winslow_adaptive_grid,
    welzl_domain_decomposition, compute_grid_metrics, sheath_reference_grid,
)
from fd_highorder import (
    compact_fd_matrix_4th, compact_fd_matrix_6th,
    explicit_fd_matrix, nonuniform_fd_matrix,
    fd_second_deriv_matrix, weno5_reconstruct,
    spectral_radius, dispersion_relation_check,
)
from sheath_quadrature import (
    gauss_legendre_nodes_weights, gauss_hermite_nodes_weights,
    maxwellian_1d, compute_moments, triangle_quadrature,
    hypercube_monomial_integral, velocity_space_integral,
    plasma_dispersion_function, bohman_velocity_integral,
)
from sheath_nonlinear import PoissonBoltzmannSolver
from sheath_time_encoding import (
    sinc_interpolation, generate_bandlimited_signal,
    TimeEncodingMachine, MultiChannelTEM,
    sheath_probe_signal_reconstruction,
)
from sheath_resonance import (
    fermat_factor, fermat_factor_all, find_resonance_modes,
    find_resonance_via_fermat, ion_acoustic_dispersion,
    sheath_dispersion_relation, find_mode_crossings,
)
from sheath_mode_selection import (
    knapsack_brute, knapsack_greedy, knapsack_dp,
    sheath_mode_selection, generate_synthetic_modes,
)
from sheath_spectral_stability import SpectralStabilityAnalyzer
from sheath_praxis_optimizer import praxis_minimize, sheath_parameter_optimization
from sheath_glomin_eigenvalue import glomin, find_marginal_wavenumber, sheath_eigenvalue_search
from sheath_elliptic_presheath import (
    elliptic_F, elliptic_E, elliptic_K, elliptic_E_complete,
    ellipsoid_surface_area, ellipsoid_volume,
    magnetic_flux_factor, chodura_condition,
    presheath_transit_time, magnetic_presheath_profile,
)
from sheath_vlasov_splitting import (
    strang_splitting_step, krylov_matrix_exp, vlasov_poisson_solve,
)
from sheath_simplicial_topology import (
    SimplicialComplex, build_sheath_phase_space_complex,
    analyze_topology_emergence, persistent_homology_summary,
)


# ============================================================================
# 工具函数
# ============================================================================

def section_header(title: str, char: str = "="):
    """输出章节标题"""
    line = char * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def print_result(key: str, value, indent: int = 2):
    """格式化输出结果"""
    prefix = " " * indent
    if isinstance(value, float):
        print(f"{prefix}{key}: {value:.6e}")
    elif isinstance(value, (list, np.ndarray)):
        if hasattr(value, '__len__') and len(value) > 5:
            print(f"{prefix}{key}: [{value[0]:.4e}, {value[1]:.4e}, ..., {value[-1]:.4e}] (len={len(value)})")
        else:
            print(f"{prefix}{key}: {value}")
    else:
        print(f"{prefix}{key}: {value}")


# ============================================================================
# 主计算流程
# ============================================================================

def main():
    """
    主计算流程

    阶段 1: 等离子体参数设置与网格生成
    阶段 2: 高阶有限差分算子构造
    阶段 3: 壁面几何处理与域分解
    阶段 4: 速度空间积分与 Bohm 判据
    阶段 5: 非线性 Poisson-Boltzmann 求解
    阶段 6: 共振模式分析 (Fermat 分解)
    阶段 7: 谱稳定性分析
    阶段 8: 模式选择 (背包问题)
    阶段 9: 参数优化 (PRAXIS)
    阶段 10: 边际稳定性搜索 (glomin)
    阶段 11: 磁前鞘计算 (椭圆积分)
    阶段 12: Vlasov 动理学验证
    阶段 13: 相空间拓扑分析
    阶段 14: 探针信号重建
    """

    t_start = time.time()

    section_header("等离子体鞘层高阶有限差分与稳定性分析", "=")
    print("  融合 15 个种子项目的计算等离子体博士级研究框架")
    print("  方向: 等离子体鞘层-壁面相互作用")
    print("  方法: 高阶紧致有限差分 + 谱稳定性分析")

    # ========================================================================
    # 阶段 1: 等离子体参数与网格
    # ========================================================================
    section_header("阶段 1: 等离子体参数设置与网格生成", "-")

    params = PlasmaParams(
        T_e=3.0, T_i=0.03, n_0=1e16,
        species="argon", gamma_e=0.05,
        B_0=0.01, theta_B=0.2618,
        V_wall=-30.0, L_domain=20.0,
        N_grid=64, FD_order=4,
    )
    print(params.summary())

    # 多种网格生成
    x_uniform = np.linspace(0, params.L_domain, params.N_grid)
    x_geometric = geometric_grid(params.L_domain, params.N_grid, ratio=1.03)
    x_tanh = tanh_clustering(params.L_domain, params.N_grid, beta=3.0)
    x_adaptive = sheath_reference_grid(params)

    metrics_uniform = compute_grid_metrics(x_uniform)
    metrics_adaptive = compute_grid_metrics(x_adaptive)

    print_result("均匀网格均匀性", metrics_uniform['uniformity'])
    print_result("自适应网格均匀性", metrics_adaptive['uniformity'])
    print_result("自适应网格最小间距", metrics_adaptive['dx_min'])
    print_result("自适应网格最大拉伸比", metrics_adaptive['stretch_ratio'])

    # Welzl 域分解
    rng = np.random.RandomState(42)
    param_points = rng.rand(50, 2) * np.array([params.L_domain, 5.0])
    domains = welzl_domain_decomposition(param_points, n_subdomains=4)
    print_result("Welzl 域分解子域数", len(domains))

    # ========================================================================
    # 阶段 2: 有限差分算子
    # ========================================================================
    section_header("阶段 2: 高阶有限差分算子构造", "-")

    x = x_adaptive
    N = len(x)
    dx_avg = (x[-1] - x[0]) / (N - 1)

    D1_compact4 = compact_fd_matrix_4th(N, dx_avg)
    D1_explicit4 = explicit_fd_matrix(N, dx_avg, order=4)
    D2 = fd_second_deriv_matrix(x, order=2)
    D2_4th = fd_second_deriv_matrix(x, order=4)
    D1_nonuniform = nonuniform_fd_matrix(x, order=4)

    rho_compact4 = spectral_radius(D1_compact4)
    rho_explicit4 = spectral_radius(D1_explicit4)

    print_result("紧致4阶微分矩阵谱半径", rho_compact4)
    print_result("显式4阶微分矩阵谱半径", rho_explicit4)

    # 色散关系检验
    k_exact, k_num = dispersion_relation_check(D1_compact4, dx_avg)
    dispersion_error = np.max(np.abs(k_exact - k_num.real))
    print_result("紧致4阶色散误差", dispersion_error)

    # WENO-5 测试
    test_func = np.sin(2 * np.pi * x / params.L_domain)
    test_recon = weno5_reconstruct(test_func, dx_avg)
    weno_error = np.max(np.abs(test_recon[5:-5] - test_func[5:-5]))
    print_result("WENO-5 重构误差 (内部)", weno_error)

    # ========================================================================
    # 阶段 3: 壁面几何与拓扑
    # ========================================================================
    section_header("阶段 3: 壁面几何三角化与拓扑", "-")

    # 构造壁面三角网格 (代表壁面粗糙度)
    n_wall_pts = 20
    wall_theta = np.linspace(0, 2 * math.pi, n_wall_pts, endpoint=False)
    wall_r = 1.0 + 0.05 * np.sin(5 * wall_theta)  # 粗糙度
    wall_x = wall_r * np.cos(wall_theta)
    wall_y = wall_r * np.sin(wall_theta)
    wall_pts_3d = np.column_stack([wall_x, wall_y, np.zeros(n_wall_pts)])

    print_result("壁面三角网格顶点数", n_wall_pts)

    # 构造壁面相空间单纯复形
    K_wall = SimplicialComplex()
    for i in range(n_wall_pts):
        K_wall.add_vertex((i,))
        K_wall.add_edge(i, (i + 1) % n_wall_pts)
        if i < n_wall_pts - 2:
            K_wall.add_triangle(i, i + 1, i + 2)

    wall_betti = K_wall.betti_numbers()
    wall_chi = K_wall.euler_characteristic()
    print_result("壁面复形 Betti 数 (β₀, β₁, β₂)", wall_betti)
    print_result("壁面 Euler 特征 χ", wall_chi)

    # ========================================================================
    # 阶段 4: 速度空间积分与 Bohm 判据
    # ========================================================================
    section_header("阶段 4: 速度空间积分与 Bohm 判据验证", "-")

    # Gauss-Hermite 求积
    n_quad = 32
    gh_nodes, gh_weights = gauss_hermite_nodes_weights(n_quad)
    print_result("Gauss-Hermite 节点数", n_quad)

    # Maxwell 分布验证
    v_th = 1.0  # 归一化热速度
    f_maxwell = maxwellian_1d(gh_nodes * v_th, 1.0, params.T_e, params.m_i)
    n_mom, u_mom, T_mom = compute_moments(f_maxwell, gh_nodes * v_th, gh_weights)
    print_result("Maxwell 密度矩量", n_mom)
    print_result("Maxwell 平均速度", u_mom)

    # 等离子体色散函数
    Z_test = plasma_dispersion_function(complex(1.0, 0.5))
    print_result("等离子体色散函数 Z(1+0.5i)", f"{Z_test.real:.4f} + {Z_test.imag:.4f}i")

    # Bohm 判据
    c_s, gamma_ratio = bohman_velocity_integral(params.T_e, params.T_i, params.m_i)
    bohman_ok, bohman_val = params.bohman_criterion_check(c_s)
    print_result("Bohm 速度 c_s", c_s)
    print_result("Bohm 判据满足", bohman_ok)
    print_result("Bohm 比值 u_i/c_s", bohman_val)

    # 三角形求积
    xi_tri, eta_tri, w_tri = triangle_quadrature(order=5)
    print_result("三角形求积点数", len(w_tri))

    # 超立方体积分
    hc_int = hypercube_monomial_integral(3, np.array([2, 1, 0]))
    print_result("单位超立方体 ∫x²y dx", hc_int)

    # ========================================================================
    # 阶段 5: 非线性 Poisson-Boltzmann 求解
    # ========================================================================
    section_header("阶段 5: 非线性 Poisson-Boltzmann 自洽求解", "-")

    solver = PoissonBoltzmannSolver(
        x=x,
        phi_wall=params.dimensionless_wall_potential(),
        chi_see=params.chi_see,
        gamma_see=params.gamma_e,
        mass_ratio=params.mass_ratio,
        fd_order=2,
    )

    # Newton-Raphson 求解
    phi_newton, conv_newton = solver.solve_newton(max_iter=50, tol=1e-8)
    print_result("Newton-Raphson 收敛", conv_newton)
    print_result("Newton 迭代次数", solver.n_iterations)
    if solver.residual_history:
        print_result("Newton 最终残差", solver.residual_history[-1])

    # Picard 求解 (作为验证)
    solver2 = PoissonBoltzmannSolver(
        x=x,
        phi_wall=params.dimensionless_wall_potential(),
        chi_see=params.chi_see,
        gamma_see=params.gamma_e,
        mass_ratio=params.mass_ratio,
    )
    phi_picard, conv_picard = solver2.solve_picard(max_iter=200, tol=1e-6, omega=0.3)
    print_result("Picard 收敛", conv_picard)

    # 两种方法的差异
    if conv_newton and conv_picard:
        diff_newton_picard = np.max(np.abs(phi_newton - phi_picard))
        print_result("Newton vs Picard 最大差异", diff_newton_picard)

    # 鞘层物理量
    props = solver.compute_sheath_properties()
    print_result("鞘层起始位置 (无量纲)", props['sheath_start'])
    print_result("最大电场 (无量纲)", props['max_field'])
    print_result("最大电荷不平衡", props['charge_imbalance_max'])
    print_result("最小电势", props['phi_min'])

    # ========================================================================
    # 阶段 6: 共振模式分析
    # ========================================================================
    section_header("阶段 6: 共振模式分析与 Fermat 分解", "-")

    # Fermat 分解测试
    n_test = 323  # = 17 × 19
    f1, f2 = fermat_factor(n_test)
    print_result(f"Fermat 分解 {n_test}", f"({f1}, {f2})")
    assert f1 * f2 == n_test, "Fermat 分解错误!"

    # 共振模式搜索
    modes = find_resonance_modes(
        params.omega_ci, params.omega_pi,
        omega_max=2 * params.omega_pi,
        max_mode=15,
    )
    print_result("找到的共振模式数", len(modes))
    if modes:
        print_result("最低频率模式", f"m={modes[0][0]}, n={modes[0][1]}, ω={modes[0][2]:.4e}")

    # Fermat 分解找共振
    fermat_resonances = find_resonance_via_fermat(36, params.omega_ci, params.omega_pi)
    print_result("Fermat 共振对数", len(fermat_resonances))

    # 离子声波色散
    k_test = 1.0 / params.lambda_De
    omega_ia = ion_acoustic_dispersion(k_test, params.T_e, params.T_i, params.m_i)
    print_result("离子声波频率 (实部)", omega_ia.real)
    print_result("离子声波阻尼率 (虚部)", omega_ia.imag)

    # 模式交叉检测
    k_range = np.linspace(0.1, 5.0, 50)
    crossings = find_mode_crossings(k_range, params.T_e, params.T_i, params.m_i, n_modes=4)
    print_result("检测到的模式交叉点数", len(crossings))

    # ========================================================================
    # 阶段 7: 谱稳定性分析
    # ========================================================================
    section_header("阶段 7: 谱稳定性分析", "-")

    analyzer = SpectralStabilityAnalyzer(
        x=x,
        phi_eq=phi_newton,
        n_i_eq=solver.n_i,
        n_e_eq=solver.n_e,
        params=params,
    )

    # 特征值计算
    stab_info = analyzer.stability_diagnosis()
    print_result("最大增长率", stab_info['max_growth_rate'])
    print_result("不稳定模数", stab_info['n_unstable'])
    print_result("谱半径", stab_info['spectral_radius'])
    print_result("CFL 时间步限制", stab_info['cfl_dt_limit'])
    print_result("系统稳定", stab_info['is_stable'])

    # von Neumann 分析
    dt_test = 0.5 * stab_info['cfl_dt_limit']
    if dt_test > 0 and dt_test < 100:
        vn_euler = analyzer.von_neumann_analysis(dt_test, scheme="euler")
        print_result("von Neumann (Euler) ρ(G)", vn_euler['spectral_radius_G'])
        print_result("Euler 显式稳定", vn_euler['is_stable'])

        vn_cn = analyzer.von_neumann_analysis(dt_test, scheme="cn")
        print_result("von Neumann (CN) ρ(G)", vn_cn['spectral_radius_G'])
        print_result("Crank-Nicolson 稳定", vn_cn['is_stable'])

    # ========================================================================
    # 阶段 8: 模式选择
    # ========================================================================
    section_header("阶段 8: 最优模式选择 (背包问题)", "-")

    # 生成合成模式数据
    freqs, gammas, costs = generate_synthetic_modes(n_modes=20)

    budget = np.sum(costs) * 0.4  # 40% 预算

    # 贪心方法
    result_greedy = sheath_mode_selection(freqs, gammas, costs, budget, method="greedy")
    print_result("贪心选择模式数", result_greedy['n_selected'])
    print_result("贪心总价值", result_greedy['total_value'])
    print_result("贪心预算利用率", result_greedy['utilization'])

    # 暴力方法 (小规模)
    n_small = min(15, len(freqs))
    result_brute = sheath_mode_selection(
        freqs[:n_small], gammas[:n_small], costs[:n_small],
        budget * 0.4, method="brute"
    )
    print_result("暴力选择模式数", result_brute['n_selected'])
    print_result("暴力总价值", result_brute['total_value'])

    # DP 方法
    result_dp = sheath_mode_selection(freqs, gammas, costs, budget, method="dp")
    print_result("DP 选择模式数", result_dp['n_selected'])
    print_result("DP 总价值", result_dp['total_value'])

    # ========================================================================
    # 阶段 9: 参数优化 (PRAXIS)
    # ========================================================================
    section_header("阶段 9: PRAXIS 鞘层参数优化", "-")

    opt_result = sheath_parameter_optimization(
        phi_wall=params.dimensionless_wall_potential(),
        chi_see=params.chi_see,
        mass_ratio=params.mass_ratio,
    )
    print_result("最优 Bohm 速度", opt_result['optimal_u_bohm'])
    print_result("最优 SEE 系数", opt_result['optimal_gamma'])
    print_result("最优磁场角 (rad)", opt_result['optimal_theta'])
    print_result("最优目标函数值", opt_result['objective_value'])
    print_result("函数评估次数", opt_result['n_evaluations'])

    # ========================================================================
    # 阶段 10: 边际稳定性搜索 (glomin)
    # ========================================================================
    section_header("阶段 10: glomin 边际稳定性搜索", "-")

    ev_result = sheath_eigenvalue_search(
        T_e=params.T_e, T_i=params.T_i,
        m_i=params.m_i,
        phi_0=abs(params.dimensionless_wall_potential()),
        lambda_D=params.lambda_De,
    )
    print_result("最大增长率波数 k", ev_result['k_max_growth'])
    print_result("最大增长率 γ", ev_result['gamma_max'])
    print_result("最稳定波数 k", ev_result['k_most_stable'])
    print_result("边际稳定波数 k_c", ev_result['k_marginal'])

    # ========================================================================
    # 阶段 11: 磁前鞘计算
    # ========================================================================
    section_header("阶段 11: 磁前鞘计算 (椭圆积分)", "-")

    theta_B = params.theta_B
    rho_s = params.rho_s

    # 椭圆积分
    K_val = elliptic_K(math.sin(theta_B))
    E_val = elliptic_E_complete(math.sin(theta_B))
    F_val = elliptic_F(math.pi / 4, math.sin(theta_B))
    print_result(f"K(sin({theta_B:.3f}))", K_val)
    print_result(f"E(sin({theta_B:.3f}))", E_val)
    print_result(f"F(π/4, sin({theta_B:.3f}))", F_val)

    # 磁通量因子
    flux_factor = magnetic_flux_factor(theta_B, rho_s, params.lambda_De)
    print_result("磁通量几何因子", flux_factor)

    # Chodura 条件
    chodura_ok, M_total, M_crit = chodura_condition(theta_B, M_parallel=1.2)
    print_result("Chodura 条件满足", chodura_ok)
    print_result("总 Mach 数", M_total)
    print_result("临界 Mach数", M_crit)

    # 椭球面积 (壁面粗糙元)
    a_rough, b_rough, c_rough = 1.0, 0.8, 0.6  # 半轴 (μm, 归一化)
    S_rough = ellipsoid_surface_area(a_rough, b_rough, c_rough)
    V_rough = ellipsoid_volume(a_rough, b_rough, c_rough)
    print_result("粗糙元椭球面积", S_rough)
    print_result("粗糙元椭球体积", V_rough)

    # 前鞘渡越时间
    L_p = 5.0 * params.lambda_De  # 前鞘长度
    tau_p = presheath_transit_time(theta_B, L_p, params.c_s, params.omega_ci)
    print_result("前鞘渡越时间 (s)", tau_p)

    # 前鞘分布
    x_presheath = np.linspace(0, 10 * params.lambda_De, 30)
    n_i_ps, u_par_ps, u_perp_ps = magnetic_presheath_profile(
        x_presheath, theta_B,
        n_0=params.n_0, c_s=params.c_s,
        L_presheath=10 * params.lambda_De,
    )
    print_result("前鞘出口离子密度", n_i_ps[-1])
    print_result("前鞘出口平行速度", u_par_ps[-1])

    # ========================================================================
    # 阶段 12: Vlasov 动理学验证
    # ========================================================================
    section_header("阶段 12: Vlasov 方程 Strang 分裂验证", "-")

    Nx_vl = 24
    Nv_vl = 24
    L_x_vl = 2 * math.pi
    v_max_vl = 6.0
    n_steps_vl = 10
    dt_vl = 0.1

    vlasov_result = vlasov_poisson_solve(
        Nx=Nx_vl, Nv=Nv_vl,
        L_x=L_x_vl, v_max=v_max_vl,
        n_steps=n_steps_vl, dt=dt_vl,
    )

    energy_init = vlasov_result['energy_history'][0] if vlasov_result['energy_history'] else 0.0
    energy_final = vlasov_result['energy_history'][-1] if vlasov_result['energy_history'] else 0.0
    print_result("Vlasov 初始场能", energy_init)
    print_result("Vlasov 最终场能", energy_final)
    if energy_init > 1e-30:
        print_result("能量衰减比 E/E₀", energy_final / energy_init)
    print_result("最终密度均匀性", np.std(vlasov_result['density_final']))

    # Krylov 矩阵指数测试
    n_krylov = 16
    A_test = np.random.RandomState(0).randn(n_krylov, n_krylov)
    A_test = 0.5 * (A_test + A_test.T)  # 对称化
    v_test = np.random.RandomState(1).randn(n_krylov)
    w_krylov = krylov_matrix_exp(A_test, v_test, t=0.1, m=10)
    w_exact = la.expm(0.1 * A_test) @ v_test
    krylov_err = np.linalg.norm(w_krylov - w_exact) / max(np.linalg.norm(w_exact), 1e-15)
    print_result("Krylov 矩阵指数相对误差", krylov_err)

    # ========================================================================
    # 阶段 13: 相空间拓扑分析
    # ========================================================================
    section_header("阶段 13: 相空间单纯复形拓扑分析", "-")

    # 从 Vlasov 解构造相空间复形
    f_final = vlasov_result['f_final']
    x_vl = vlasov_result['x']
    v_vl = vlasov_result['v']

    K_sheath = build_sheath_phase_space_complex(
        f_final, x_vl, v_vl, threshold=0.1
    )

    topology_summary = persistent_homology_summary(K_sheath)
    print_result("相空间顶点数", topology_summary['n_vertices'])
    print_result("相空间边数", topology_summary['n_edges'])
    print_result("相空间三角数", topology_summary['n_triangles'])
    print_result("Betti 数 (β₀, β₁, β₂)", topology_summary['betti_numbers'])
    print_result("Euler 特征", topology_summary['euler_characteristic'])

    # ========================================================================
    # 阶段 14: 探针信号重建
    # ========================================================================
    section_header("阶段 14: 多通道时间编码信号重建", "-")

    probe_result = sheath_probe_signal_reconstruction(
        n_samples=128, bandwidth=5.0,
        noise_level=0.01, seed=42,
    )
    print_result("sinc 重建相对误差", probe_result['err_sinc'])
    print_result("TEM 重建触发点数", probe_result['n_triggers'])
    if probe_result['n_triggers'] >= 3:
        print_result("TEM 重建相对误差", probe_result['err_tem'])

    # ========================================================================
    # 最终总结
    # ========================================================================
    t_end = time.time()

    section_header("计算完成 - 结果总结", "=")

    summary_items = [
        ("等离子体种类", params.species),
        ("电子温度 T_e", f"{params.T_e} eV"),
        ("德拜长度 λ_De", f"{params.lambda_De:.4e} m"),
        ("网格点数 N", params.N_grid),
        ("Newton 收敛", conv_newton),
        ("最小电势 φ_min", f"{props['phi_min']:.4f}"),
        ("最大增长率", f"{stab_info['max_growth_rate']:.4e}"),
        ("系统稳定性", "稳定" if stab_info['is_stable'] else "不稳定"),
        ("Chodura 条件", "满足" if chodura_ok else "不满足"),
        ("Bohm 判据", "满足" if bohman_ok else "不满足"),
        ("Vlasov 能量衰减", f"{energy_final/max(energy_init,1e-30):.4e}"),
        ("相空间 Betti 数", str(topology_summary['betti_numbers'])),
        ("sinc 重建误差", f"{probe_result['err_sinc']:.4e}"),
        ("总运行时间", f"{t_end - t_start:.2f} s"),
    ]

    for key, val in summary_items:
        print(f"  {key:25s}: {val}")

    print(f"\n{'=' * 70}")
    print("  所有 15 个种子项目算法均已融合并成功运行")
    print("  1413_welzl      → 参数空间最小包围球域分解")
    print("  420_fermat      → 共振模式 Fermat 因式分解")
    print("  823_obj_to_tri  → 壁面三角化网格")
    print("  907_praxis      → 鞘层参数 PRAXIS 优化")
    print("  1078_alebrew    → 自适应采样策略")
    print("  1261_time_enc   → 探针信号时间编码重建")
    print("  471_glomin      → 边际稳定性全局搜索")
    print("  1305_tri_grid   → 三角形求积规则")
    print("  400_fem2d_bvp   → Poisson-Boltzmann FEM 基准")
    print("  559_hypercube   → 多维速度空间积分")
    print("  625_knapsack_g  → 贪心模式选择")
    print("  332_ellipsoid   → 椭圆积分/磁通量几何")
    print("  1015_qsim       → Vlasov Strang 分裂/Krylov 传播")
    print("  623_knapsack_b  → 暴力模式枚举")
    print("  1208_hypergraph → 相空间单纯复形拓扑")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n!!! 运行时错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
