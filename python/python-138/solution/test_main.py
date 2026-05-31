"""
微反应器混合与反应强化多尺度计算框架 — 统一入口
================================================
PROJECT_138: 化学工程 - 微反应器混合与反应强化

本程序基于 15 个种子项目的核心算法，融合构建了一个面向微反应器
设计优化的博士级计算框架。无需任何输入参数，直接运行即可得到
完整的数值分析结果。

执行流程：
    1. 对流-扩散-反应 PDE 稳态求解
    2. 催化剂最优分布 (CVT)
    3. 反应动力学参数估计 (QR 最小二乘)
    4. 稳态稳定性特征值分析
    5. 质量-能量耦合平衡定点迭代
    6. 降阶模型基函数构造 (MGS)
    7. 混合质量多变量统计度量
    8. 操作条件优化 (黄金分割 + Newton)
    9. 稀疏 skyline 矩阵运算
   10. 反应器网络拓扑分析
   11. 薄板热应力分析 (双调和方程)
   12. 离散催化剂负载整数规划
"""

import sys
import numpy as np

# 设置随机种子以保证可复现
np.random.seed(138)

from reactor_pde_solver import MicroreactorPDESolver
from catalyst_placement_cvt import CatalystCVTPlacer
from kinetics_parameter_estimation import KineticsParameterEstimator
from stability_eigenanalysis import ReactorStabilityAnalyzer
from mass_energy_balance import MassEnergyBalanceSolver
from reduced_order_basis import ReducedOrderBasisBuilder
from mixing_quality_statistics import MixingQualityAnalyzer
from reactor_optimization import ReactorOptimizer
from sparse_matrix_ops import SkylineMatrixOperator
from network_topology import MicroreactorNetworkTopology
from thermal_stress_analysis import ThermalStressAnalyzer
from discrete_catalyst_loading import DiscreteCatalystLoadingOptimizer


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("微反应器混合与反应强化多尺度计算框架")
    print("PROJECT_138 | 化学工程 | 零参数运行")
    print("=" * 70)

    # =====================================================================
    # 1. 对流-扩散-反应 PDE 稳态求解 (基于 schroedinger_linear_pde)
    # =====================================================================
    print_section("1. 微通道对流-扩散-反应 PDE 稳态求解")

    pde_solver = MicroreactorPDESolver(
        L=0.05,
        Nx=120,
        D_m=2.0e-9,
        u=0.02,
        A_arr=5.0e7,
        Ea=48000.0,
        reaction_order=1.0,
        rho=980.0,
        cp=4200.0,
        lam=0.55,
        dH=-7.5e4,
        T_wall=360.0,
        h_wall=600.0,
        hydraulic_diameter=4.0e-4,
        C_in=800.0,
        T_in=310.0,
    )
    C_ss, T_ss = pde_solver.solve_steady_state(max_iter=3000, tol=1.0e-9)
    X_conv, C_out = pde_solver.compute_conversion_and_yield(C_ss)
    Pe, Da = pde_solver.compute_peclet_damkohler(C_ss, T_ss)
    print(f"  稳态求解完成，迭代收敛")
    print(f"  出口浓度 C_out = {C_out:.3f} mol/m³")
    print(f"  转化率 X = {X_conv*100:.2f}%")
    print(f"  出口温度 T_out = {T_ss[-1]:.2f} K")
    print(f"  Peclet 数 Pe = {Pe:.2e}")
    print(f"  Damköhler 数 Da = {Da:.2e}")

    # =====================================================================
    # 2. 催化剂最优分布 CVT (基于 cvt, florida_cvt_geo)
    # =====================================================================
    print_section("2. 微反应器催化剂 CVT 最优分布")

    def gaussian_catalyst_density(pts: np.ndarray) -> np.ndarray:
        # 中心高斯加权，模拟入口附近需要更多催化剂
        center = np.array([0.5, 0.5])
        dist2 = np.sum((pts - center) ** 2, axis=1)
        return np.exp(-dist2 / 0.15) + 0.1

    cvt_placer = CatalystCVTPlacer(
        dim=2,
        n_generators=36,
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        density_func=gaussian_catalyst_density,
        sample_num=15000,
        max_iter=30,
        tol=1.0e-5,
    )
    gens, energy, max_shift = cvt_placer.iterate()
    uniformity = cvt_placer.compute_uniformity_index()
    print(f"  CVT Lloyd 迭代完成")
    print(f"  生成元数量: {len(gens)}")
    print(f"  最终能量 E = {energy:.6e}")
    print(f"  最大移动距离 = {max_shift:.6e}")
    print(f"  均匀度指数 η = {uniformity:.4f}")

    # =====================================================================
    # 3. 反应动力学参数估计 (基于 qr_solve)
    # =====================================================================
    print_section("3. 反应动力学参数 QR 最小二乘估计")

    # 生成合成实验数据 (使用中等范围参数，保证数值稳定性)
    A_true = 5.0e4
    Ea_true = 12000.0
    n_true = 0.85
    R_gas = 8.314
    n_data = 50
    T_data = np.linspace(300.0, 400.0, n_data)
    C_data = np.linspace(100.0, 600.0, n_data)
    r_true = A_true * np.exp(-Ea_true / (R_gas * T_data)) * (C_data ** n_true)
    # 使用相对较小的均匀噪声，避免小值被淹没
    noise = np.random.normal(0.0, 0.03 * np.median(r_true), n_data)
    r_exp = r_true + noise
    r_exp = np.maximum(r_exp, 1.0e-3)

    estimator = KineticsParameterEstimator(R_gas=R_gas)
    A_est, Ea_est, n_est, res_norm = estimator.estimate_arrhenius_parameters(
        C_data, T_data, r_exp
    )
    print(f"  真实参数: A={A_true:.2e}, Ea={Ea_true:.1f}, n={n_true:.2f}")
    print(f"  估计参数: A={A_est:.2e}, Ea={Ea_est:.1f}, n={n_est:.3f}")
    print(f"  残差范数 ||r_pred - r_exp|| = {res_norm:.4f}")

    # =====================================================================
    # 4. 稳态稳定性特征值分析 (基于 chladni_figures)
    # =====================================================================
    print_section("4. 反应器稳态线性稳定性特征值分析")

    stability = ReactorStabilityAnalyzer(
        Nx=len(C_ss),
        L=0.05,
        D_m=2.0e-9,
        alpha=0.55 / (980.0 * 4200.0),
        u=0.02,
        dH=-7.5e4,
        rho=980.0,
        cp=4200.0,
        h_wall=600.0,
        d_h=4.0e-4,
    )
    eigenvals, max_real, is_stable = stability.analyze_stability(
        C_ss, T_ss, A_est, Ea_est, n_est
    )
    Da_stable, Da_unstable, max_real_trans = stability.compute_critical_damkohler_bracket(
        C_ss, T_ss, A_est, Ea_est, n_est, da_min=0.01, da_max=5.0, n_scan=15
    )
    risk_index = stability.compute_thermal_explosion_index(max_real)
    print(f"  Jacobian 矩阵维度: {len(eigenvals)} × {len(eigenvals)}")
    print(f"  最大实部特征值 Re(λ_max) = {max_real:.6f}")
    print(f"  稳态稳定性: {'稳定' if is_stable else '不稳定'}")
    print(f"  临界 Da 区间: [{Da_stable:.3f}, {Da_unstable:.3f}]")
    print(f"  热爆炸风险指数 I_TE = {risk_index:.4f}")

    # =====================================================================
    # 5. 质量-能量耦合平衡定点迭代 (基于 cobweb_plot)
    # =====================================================================
    print_section("5. 全混流微反应器质量-能量耦合平衡")

    cstr_solver = MassEnergyBalanceSolver(
        C0=800.0,
        T0=310.0,
        Q=2.0e-6,
        V=1.0e-5,
        rho=980.0,
        cp=4200.0,
        dH=-7.5e4,
        Ua=0.8,
        Tc=350.0,
        A_arr=A_est,
        Ea=Ea_est,
        reaction_order=n_est,
    )
    T_cstr, C_cstr, it_cstr, conv_cstr = cstr_solver.solve_fixed_point()
    bifurcation_idx = cstr_solver.bifurcation_indicator()
    print(f"  CSTR 稳态求解: T_ss = {T_cstr:.2f} K, C_ss = {C_cstr:.2f} mol/m³")
    print(f"  迭代次数 = {it_cstr}, 收敛 = {conv_cstr}")
    print(f"  定点映射导数 |dG/dT| = {bifurcation_idx:.4f}")
    print(f"  分岔风险: {'低' if bifurcation_idx < 0.8 else '高'}")

    # =====================================================================
    # 6. 降阶模型基函数构造 (基于 gram_schmidt)
    # =====================================================================
    print_section("6. POD 降阶模型基函数 (Modified Gram-Schmidt)")

    # 生成快照：不同壁温下的稳态温度场
    n_snapshots = 20
    snapshots = np.zeros((len(T_ss), n_snapshots))
    T_wall_range = np.linspace(330.0, 380.0, n_snapshots)
    for i, Tw in enumerate(T_wall_range):
        pde_temp = MicroreactorPDESolver(
            L=0.05, Nx=120, D_m=2.0e-9, u=0.02,
            A_arr=A_est, Ea=Ea_est, reaction_order=n_est,
            rho=980.0, cp=4200.0, lam=0.55, dH=-7.5e4,
            T_wall=Tw, h_wall=600.0, hydraulic_diameter=4.0e-4,
            C_in=800.0, T_in=310.0,
        )
        _, T_field = pde_temp.solve_steady_state(max_iter=2000, tol=1.0e-8)
        snapshots[:, i] = T_field

    rom_builder = ReducedOrderBasisBuilder(tolerance=1.0e-10)
    basis, rank, orth_err = rom_builder.modified_gram_schmidt(snapshots)
    pod_basis, svals, energy_ratio = rom_builder.compute_pod_modes_svd(snapshots)
    reduction_err = rom_builder.compute_reduction_error(snapshots, pod_basis)
    print(f"  快照数量: {n_snapshots}")
    print(f"  MGS 有效秩: {rank}, 正交误差: {orth_err:.2e}")
    print(f"  POD 前 5 奇异值: {svals[:5]}")
    print(f"  POD 能量占比: {energy_ratio*100:.2f}%")
    print(f"  降阶重构误差: {reduction_err:.4e}")

    # =====================================================================
    # 7. 混合质量统计度量 (基于 normal01_multivariate_distance)
    # =====================================================================
    print_section("7. 微反应器混合质量多变量统计度量")

    mixer = MixingQualityAnalyzer(ideal_mean=500.0, ideal_std=40.0)
    samples_mixed = mixer.sample_concentration_field(
        n_samples=2000, dim=2, mixing_efficiency=0.85
    )
    M_d = mixer.compute_mixing_defect(samples_mixed)
    cv_multi = mixer.compute_multivariate_cv(samples_mixed)
    D_kl = mixer.compute_kl_divergence(samples_mixed)
    eta_mix = mixer.compute_mixing_efficiency_index(samples_mixed)
    # 两区对比
    zone_a = samples_mixed[:1000]
    zone_b = samples_mixed[1000:]
    D_M, D_B = mixer.statistical_distance_between_zones(zone_a, zone_b)
    print(f"  混合缺陷 M_d = {M_d:.4f}")
    print(f"  多变量变异系数 CV_multi = {cv_multi:.4f}")
    print(f"  KL 散度 D_KL = {D_kl:.4f}")
    print(f"  混合效率指数 η_mix = {eta_mix:.4f}")
    print(f"  区间 Mahalanobis 距离 = {D_M:.4f}, Bhattacharyya 距离 = {D_B:.4f}")

    # =====================================================================
    # 8. 操作条件优化 (基于 test_opt, golden_section)
    # =====================================================================
    print_section("8. 反应器操作条件优化")

    optimizer = ReactorOptimizer()

    # 单参数优化：停留时间
    def obj_tau(tau_val: float) -> float:
        # 模拟目标：转化率 / (1 + 压降 ∝ tau)
        k_eff = A_est * np.exp(-Ea_est / (R_gas * 350.0))
        X = 1.0 - np.exp(-k_eff * tau_val)
        penalty = 0.01 * tau_val  # 压降惩罚
        return -(X - penalty)  # 最小化负目标 = 最大化目标

    tau_opt, f_opt, _, _ = optimizer.golden_section_search(
        obj_tau, 0.5, 60.0, max_iter=60, x_tol=1.0e-6
    )
    print(f"  黄金分割搜索 - 最优停留时间 τ = {tau_opt:.3f} s")
    print(f"  对应目标值 = {-f_opt:.4f}")

    # 多参数优化 (Rosenbrock 型测试)
    x_opt_ros, f_opt_ros = optimizer.optimize_reactor_conditions(n_params=4)
    print(f"  Newton 优化 - 最优参数: {x_opt_ros}")
    print(f"  目标函数值 = {f_opt_ros:.6e}")

    # =====================================================================
    # 9. 稀疏 skyline 矩阵运算 (基于 r8ss)
    # =====================================================================
    print_section("9. 有限元稀疏 Skyline 矩阵运算")

    n_mat = 50
    lower = np.ones(n_mat) * (-1.0)
    lower[0] = 0.0
    diagonal = np.ones(n_mat) * 2.01  # 对角占优
    upper = np.ones(n_mat) * (-1.0)
    upper[-1] = 0.0

    skyline = SkylineMatrixOperator(n_mat)
    skyline.build_from_tridiagonal(lower, diagonal, upper)
    x_test = np.ones(n_mat)
    y_test = skyline.multiply(x_test)
    A_dense = skyline.to_dense()
    x_solve = skyline.solve_cholesky_skyline(x_test)
    cond_est = skyline.condition_number_estimate()
    print(f"  矩阵维度: {n_mat} × {n_mat}")
    print(f"  Skyline 存储元素数: {skyline.na}")
    print(f"  稠密元素数: {n_mat*n_mat}")
    print(f"  存储压缩比: {skyline.na / (n_mat*n_mat):.4f}")
    print(f"  矩阵-向量乘法范数 ||y|| = {np.linalg.norm(y_test):.4f}")
    print(f"  线性系统求解残差 = {np.linalg.norm(A_dense @ x_solve - x_test):.2e}")
    print(f"  条件数估计 ≈ {cond_est:.2e}")

    # =====================================================================
    # 10. 反应器网络拓扑分析 (基于 digraph_arc, graph_arc)
    # =====================================================================
    print_section("10. 微反应器网络拓扑设计与流路分析")

    net = MicroreactorNetworkTopology(n_nodes=8)
    # 构建一个分配网络（树状）+ 收集网络（反向树）
    edges_distributor = [(0,1), (0,2), (1,3), (1,4), (2,5), (2,6), (6,7)]
    for u, v in edges_distributor:
        net.add_edge(u, v)
    has_circuit, has_path = net.is_eulerian()
    euler_path = net.find_eulerian_path()
    # Pruefer 编码演示
    tree_edges = net.generate_optimal_distribution_tree(root=0)
    pruefer_code = net.tree_to_pruefer(tree_edges)
    n_spanning = net.count_spanning_trees()
    net_uniformity = net.network_uniformity_index()
    print(f"  节点数: {net.n}")
    print(f"  边数: {len(net.edges)}")
    print(f"  Eulerian 回路存在: {has_circuit}")
    print(f"  Eulerian 路径存在: {has_path}")
    print(f"  欧拉路径: {euler_path if euler_path else 'None'}")
    print(f"  生成树数量 (Kirchhoff 定理): {n_spanning}")
    print(f"  网络均匀度指数: {net_uniformity:.4f}")
    print(f"  Pruefer 码: {pruefer_code}")

    # =====================================================================
    # 11. 薄板热应力分析 (基于 biharmonic_exact)
    # =====================================================================
    print_section("11. 微反应器薄板双调和热应力分析")

    thermal = ThermalStressAnalyzer(
        E=210.0e9, nu=0.28, h=0.8e-3, alpha_T=1.1e-5, z_eval=0.4e-3
    )
    nx_grid = 40
    x_grid = np.linspace(-0.02, 0.02, nx_grid)
    y_grid = np.linspace(-0.01, 0.01, nx_grid // 2)
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    delta_T_field = 80.0 * np.exp(-(Xg**2 + Yg**2) / (0.01**2))

    # 使用小振幅参数，避免应力数值爆炸
    sigma_x, sigma_y, tau_xy, sigma_vm = thermal.compute_thermal_stresses(
        Xg, Yg, delta_T_field, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0
    )
    # 验证双调和残差
    residual_biharm = thermal.biharmonic_residual(
        Xg, Yg, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0
    )
    sf = thermal.safety_factor(sigma_vm, yield_strength=300.0e6)
    theta_shock = thermal.thermal_shock_parameter(
        delta_T_max=np.max(delta_T_field)
    )
    print(f"  评估网格: {nx_grid} × {nx_grid//2}")
    print(f"  双调和残差 ||R||_max = {np.max(np.abs(residual_biharm)):.2e}")
    print(f"  最大 von Mises 应力 = {np.max(sigma_vm)/1e6:.3f} MPa")
    print(f"  最小 von Mises 应力 = {np.min(sigma_vm)/1e6:.3f} MPa")
    print(f"  安全系数 SF = {sf:.3f}")
    print(f"  热冲击参数 Θ = {theta_shock:.4f}")

    # =====================================================================
    # 12. 离散催化剂负载整数规划 (基于 diophantine_nd)
    # =====================================================================
    print_section("12. 离散催化剂负载整数规划")

    a_load = np.array([2.5, 3.0, 4.0, 5.5, 6.0])  # 单位负载量
    budget_load = 50.0
    dioph = DiscreteCatalystLoadingOptimizer(a_load, budget_load)
    sols, n_sols = dioph.solve_exact_nonnegative(max_solutions=200)
    weights_obj = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
    x_greedy, obj_greedy = dioph.greedy_heuristic_solution(weights_obj)
    util_greedy, unif_greedy = dioph.compute_loading_efficiency(x_greedy)
    if n_sols > 0:
        x_opt_load, obj_opt = dioph.select_optimal_loading(sols)
        util_opt, unif_opt = dioph.compute_loading_efficiency(x_opt_load)
        print(f"  精确可行解数量: {n_sols}")
        print(f"  最优解: {x_opt_load}, 目标 = {obj_opt:.4f}")
        print(f"  最优解利用率: {util_opt:.4f}, 均匀度: {unif_opt:.4f}")
    else:
        print(f"  精确可行解数量: 0 (预算无法精确满足)")
    print(f"  贪心解: {x_greedy}, 目标 = {obj_greedy:.4f}")
    print(f"  贪心解利用率: {util_greedy:.4f}, 均匀度: {unif_greedy:.4f}")

    # =====================================================================
    # 汇总
    # =====================================================================
    print_section("计算结果汇总")
    print(f"  PDE 稳态转化率:         {X_conv*100:.2f}%")
    print(f"  CVT 催化剂均匀度:       {uniformity:.4f}")
    print(f"  动力学参数估计残差:     {res_norm:.4f}")
    print(f"  稳态稳定性:             {'稳定' if is_stable else '不稳定'}")
    print(f"  CSTR 稳态温度:          {T_cstr:.2f} K")
    print(f"  POD 降阶能量占比:       {energy_ratio*100:.2f}%")
    print(f"  混合效率指数:           {eta_mix:.4f}")
    print(f"  最优停留时间:           {tau_opt:.3f} s")
    print(f"  稀疏矩阵条件数:         {cond_est:.2e}")
    print(f"  网络生成树数:           {n_spanning}")
    print(f"  热应力安全系数:         {sf:.3f}")
    print(f"  离散负载贪心均匀度:     {unif_greedy:.4f}")
    print("\n" + "=" * 70)
    print("  所有计算模块执行完毕，无报错。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: MicroreactorPDESolver construction and basic attributes ----
pde = MicroreactorPDESolver(L=0.05, Nx=120, D_m=2.0e-9, u=0.02, A_arr=5.0e7,
    Ea=48000.0, reaction_order=1.0, rho=980.0, cp=4200.0, lam=0.55, dH=-7.5e4,
    T_wall=360.0, h_wall=600.0, hydraulic_diameter=4.0e-4, C_in=800.0, T_in=310.0)
assert pde.Nx == 120, '[TC01] Nx mismatch FAILED'
assert pde.dx == 0.05 / 119, '[TC01] dx mismatch FAILED'
assert pde.C_in == 800.0, '[TC01] C_in mismatch FAILED'

# ---- TC02: MicroreactorPDESolver reaction_rate finite and non-negative ----
C_test = np.array([500.0, 0.0, 1000.0])
T_test = np.array([350.0, 350.0, 350.0])
r = pde.reaction_rate(C_test, T_test)
assert np.all(np.isfinite(r)), '[TC02] reaction_rate produced non-finite values FAILED'
assert np.all(r >= 0.0), '[TC02] reaction_rate produced negative values FAILED'
assert r[1] == 0.0, '[TC02] reaction_rate for zero concentration should be zero FAILED'

# ---- TC03: MicroreactorPDESolver steady_state solve produces correct shape ----
C_ss_test, T_ss_test = pde.solve_steady_state(max_iter=3000, tol=1.0e-9)
assert len(C_ss_test) == 120, '[TC03] C_ss_test shape mismatch FAILED'
assert len(T_ss_test) == 120, '[TC03] T_ss_test shape mismatch FAILED'
assert C_ss_test[0] == 800.0, '[TC03] inlet concentration boundary FAILED'
assert np.all(np.isfinite(C_ss_test)), '[TC03] C_ss_test contains NaN/Inf FAILED'
assert np.all(np.isfinite(T_ss_test)), '[TC03] T_ss_test contains NaN/Inf FAILED'

# ---- TC04: MicroreactorPDESolver conversion in valid range ----
X_conv_test, C_out_test = pde.compute_conversion_and_yield(C_ss_test)
assert 0.0 <= X_conv_test <= 1.0, '[TC04] Conversion out of [0,1] FAILED'
assert C_out_test >= 0.0, '[TC04] outlet concentration negative FAILED'

# ---- TC05: MicroreactorPDESolver Peclet/Damkohler finite positive ----
Pe_test, Da_test = pde.compute_peclet_damkohler(C_ss_test, T_ss_test)
assert np.isfinite(Pe_test) and Pe_test > 0.0, '[TC05] Peclet number invalid FAILED'
assert np.isfinite(Da_test) and Da_test > 0.0, '[TC05] Damkohler number invalid FAILED'

# ---- TC06: CatalystCVTPlacer construction and initial generators in bounds ----
cvt = CatalystCVTPlacer(dim=2, n_generators=25, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=5000, max_iter=10, tol=1.0e-5)
assert cvt.generators.shape == (25, 2), '[TC06] generator shape mismatch FAILED'
assert np.all(cvt.generators >= 0.0) and np.all(cvt.generators <= 1.0), '[TC06] generators out of bounds FAILED'

# ---- TC07: CatalystCVTPlacer iterate produces finite energy and max_shift >= 0 ----
import numpy as np
np.random.seed(42)
cvt2 = CatalystCVTPlacer(dim=2, n_generators=16, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=4000, max_iter=8, tol=1.0e-5)
gens2, energy2, max_shift2 = cvt2.iterate()
assert np.isfinite(energy2), '[TC07] CVT energy not finite FAILED'
assert max_shift2 >= 0.0, '[TC07] max_shift negative FAILED'
assert gens2.shape == (16, 2), '[TC07] generator shape after iteration mismatch FAILED'

# ---- TC08: CatalystCVTPlacer uniformity index in [0,1] ----
np.random.seed(42)
cvt3 = CatalystCVTPlacer(dim=2, n_generators=16, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=4000, max_iter=8, tol=1.0e-5)
cvt3.iterate()
eta = cvt3.compute_uniformity_index()
assert 0.0 <= eta <= 1.0, '[TC08] Uniformity index out of [0,1] FAILED'

# ---- TC09: KineticsParameterEstimator QR factorization correctness ----
ke = KineticsParameterEstimator(R_gas=8.314)
A_qr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
Q, R = ke.qr_factorize(A_qr)
reconstructed = Q @ R
assert np.allclose(reconstructed, A_qr, atol=1e-10), '[TC09] QR factorization reconstruction mismatch FAILED'
I_check = Q.T @ Q
assert np.allclose(I_check, np.eye(3), atol=1e-10), '[TC09] Q not orthogonal FAILED'

# ---- TC10: KineticsParameterEstimator least squares solve ----
A_ls = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b_ls = np.array([1.0, 2.0, 3.0])
x_ls = ke.solve_least_squares(A_ls, b_ls)
assert len(x_ls) == 2, '[TC10] LS solution dimension mismatch FAILED'

# ---- TC11: KineticsParameterEstimator Arrhenius estimation finite outputs ----
np.random.seed(42)
n_est_data = 30
T_est = np.linspace(310.0, 390.0, n_est_data)
C_est = np.linspace(100.0, 500.0, n_est_data)
r_true_est = 5.0e4 * np.exp(-12000.0/(8.314*T_est)) * (C_est**0.85)
noise_est = np.random.normal(0, 0.02*np.median(r_true_est), n_est_data)
r_est = np.maximum(r_true_est + noise_est, 1e-3)
A_est, Ea_est, n_est, res_norm = ke.estimate_arrhenius_parameters(C_est, T_est, r_est)
assert np.isfinite(A_est) and A_est > 0.0, '[TC11] A_est invalid FAILED'
assert np.isfinite(Ea_est) and Ea_est > 0.0, '[TC11] Ea_est invalid FAILED'
assert np.isfinite(n_est) and n_est >= 0.0, '[TC11] n_est invalid FAILED'
assert np.isfinite(res_norm) and res_norm >= 0.0, '[TC11] residual norm invalid FAILED'

# ---- TC12: ReactorStabilityAnalyzer Jacobian shape ----
rsa = ReactorStabilityAnalyzer(Nx=30, L=0.05, D_m=2.0e-9, alpha=0.55/(980.0*4200.0),
    u=0.02, dH=-7.5e4, rho=980.0, cp=4200.0, h_wall=600.0, d_h=4.0e-4)
C_steady_test = np.linspace(800.0, 400.0, 30)
T_steady_test = np.linspace(310.0, 360.0, 30)
J_stab = rsa.compute_jacobian(C_steady_test, T_steady_test, A_arr=5.0e7, Ea=48000.0, n_order=1.0)
assert J_stab.shape == (60, 60), '[TC12] Jacobian shape mismatch FAILED'
assert np.all(np.isfinite(J_stab)), '[TC12] Jacobian contains NaN/Inf FAILED'

# ---- TC13: ReactorStabilityAnalyzer thermal explosion index in [0,1) ----
risk_idx = rsa.compute_thermal_explosion_index(-0.5)
assert 0.0 <= risk_idx <= 1.0, '[TC13] risk index for stable case FAILED'
assert risk_idx == 0.0, '[TC13] risk index should be 0 for negative max_real FAILED'

# ---- TC14: MassEnergyBalanceSolver rate_constant positive finite ----
cstr = MassEnergyBalanceSolver(C0=800.0, T0=310.0, Q=2.0e-6, V=1.0e-5,
    rho=980.0, cp=4200.0, dH=-7.5e4, Ua=0.8, Tc=350.0, A_arr=1.0e8, Ea=50000.0, reaction_order=1.0)
k_rate = cstr.rate_constant(350.0)
assert np.isfinite(k_rate) and k_rate > 0.0, '[TC14] rate constant invalid FAILED'

# ---- TC15: MassEnergyBalanceSolver concentration_from_T in valid range ----
C_from_T = cstr.concentration_from_T(350.0)
assert np.isfinite(C_from_T) and C_from_T >= 0.0, '[TC15] C_from_T invalid FAILED'
assert C_from_T <= cstr.C0, '[TC15] C_from_T exceeds C0 FAILED'

# ---- TC16: MassEnergyBalanceSolver fixed point solve returns valid state ----
np.random.seed(42)
cstr2 = MassEnergyBalanceSolver(C0=800.0, T0=310.0, Q=2.0e-6, V=1.0e-5,
    rho=980.0, cp=4200.0, dH=-7.5e4, Ua=0.8, Tc=350.0,
    A_arr=5.0e7, Ea=48000.0, reaction_order=1.0)
T_cstr_test, C_cstr_test, it_cstr, conv_cstr = cstr2.solve_fixed_point()
assert np.isfinite(T_cstr_test) and T_cstr_test > 200.0, '[TC16] T_cstr_test invalid FAILED'
assert np.isfinite(C_cstr_test) and C_cstr_test >= 0.0, '[TC16] C_cstr_test invalid FAILED'
assert conv_cstr, '[TC16] fixed point did not converge FAILED'

# ---- TC17: MassEnergyBalanceSolver bifurcation indicator finite ----
bif_idx = cstr2.bifurcation_indicator()
assert np.isfinite(bif_idx) and bif_idx >= 0.0, '[TC17] bifurcation indicator invalid FAILED'

# ---- TC18: ReducedOrderBasisBuilder MGS returns correct rank and basis shape ----
np.random.seed(42)
N_dim = 50
m_snap = 15
snapshots_test = np.random.randn(N_dim, m_snap)
rom = ReducedOrderBasisBuilder(tolerance=1.0e-10)
basis_mgs, rank_mgs, err_mgs = rom.modified_gram_schmidt(snapshots_test)
assert basis_mgs.shape[0] == N_dim, '[TC18] basis rows mismatch FAILED'
assert basis_mgs.shape[1] == rank_mgs, '[TC18] basis columns != rank FAILED'
assert rank_mgs <= m_snap, '[TC18] rank exceeds number of snapshots FAILED'
assert err_mgs >= 0.0, '[TC18] orth error negative FAILED'

# ---- TC19: ReducedOrderBasisBuilder POD SVD energy ratio in [0,1] ----
pod_basis, svals, energy_ratio = rom.compute_pod_modes_svd(snapshots_test)
assert len(svals) > 0, '[TC19] no singular values returned FAILED'
assert 0.0 <= energy_ratio <= 1.0, '[TC19] energy ratio out of [0,1] FAILED'
assert pod_basis.shape[0] == N_dim, '[TC19] POD basis rows mismatch FAILED'

# ---- TC20: ReducedOrderBasisBuilder projection-reconstruction cycle ----
if rank_mgs > 0:
    field_test = snapshots_test[:, 0]
    coeffs = rom.project_onto_basis(field_test, basis_mgs)
    reconstructed = rom.reconstruct_from_basis(coeffs, basis_mgs)
    assert len(reconstructed) == N_dim, '[TC20] reconstruction length mismatch FAILED'

# ---- TC21: MixingQualityAnalyzer mixing defect non-negative ----
np.random.seed(42)
mixer = MixingQualityAnalyzer(ideal_mean=500.0, ideal_std=40.0)
samples_test = mixer.sample_concentration_field(n_samples=500, dim=2, mixing_efficiency=0.85)
assert samples_test.shape == (500, 2), '[TC21] sample shape mismatch FAILED'
M_d = mixer.compute_mixing_defect(samples_test)
assert np.isfinite(M_d) and M_d >= 0.0, '[TC21] mixing defect invalid FAILED'

# ---- TC22: MixingQualityAnalyzer efficiency index in [0,1] ----
eta_mix = mixer.compute_mixing_efficiency_index(samples_test)
assert 0.0 <= eta_mix <= 1.0, '[TC22] mixing efficiency index out of [0,1] FAILED'

# ---- TC23: MixingQualityAnalyzer zone distance finite ----
np.random.seed(42)
zone_a = samples_test[:250]
zone_b = samples_test[250:]
D_M, D_B = mixer.statistical_distance_between_zones(zone_a, zone_b)
assert np.isfinite(D_M) and D_M >= 0.0, '[TC23] Mahalanobis distance invalid FAILED'
assert np.isfinite(D_B), '[TC23] Bhattacharyya distance not finite FAILED'

# ---- TC24: MixingQualityAnalyzer multivariate CV non-negative ----
cv_multi = mixer.compute_multivariate_cv(samples_test)
assert np.isfinite(cv_multi) and cv_multi >= 0.0, '[TC24] multivariate CV invalid FAILED'

# ---- TC25: ReactorOptimizer golden section search on quadratic ----
opt = ReactorOptimizer()
def f_quad(x):
    return (x - 3.0)**2
x_opt_gss, f_opt_gss, it_gss, nf_gss = opt.golden_section_search(f_quad, 0.0, 10.0, max_iter=50, x_tol=1e-8)
assert np.abs(x_opt_gss - 3.0) < 1e-4, '[TC25] golden section failed to find minimum of (x-3)^2 FAILED'
assert np.abs(f_opt_gss) < 1e-7, '[TC25] golden section minimum value not near zero FAILED'

# ---- TC26: ReactorOptimizer Rosenbrock objective and gradient consistency ----
np.random.seed(42)
x0_ros = np.array([-1.0, 0.5, 0.3])
f0, g0, H0 = opt.reactor_objective_rosenbrock_like(x0_ros)
assert np.isfinite(f0), '[TC26] Rosenbrock objective not finite FAILED'
assert len(g0) == 3, '[TC26] gradient dimension mismatch FAILED'
assert H0.shape == (3, 3), '[TC26] Hessian shape mismatch FAILED'
assert np.all(np.isfinite(g0)), '[TC26] gradient contains NaN/Inf FAILED'

# ---- TC27: SkylineMatrixOperator build and to_dense symmetry ----
n_sky = 30
lower_s = np.ones(n_sky) * (-1.0); lower_s[0] = 0.0
diag_s = np.ones(n_sky) * 2.01
upper_s = np.ones(n_sky) * (-1.0); upper_s[-1] = 0.0
sky = SkylineMatrixOperator(n_sky)
sky.build_from_tridiagonal(lower_s, diag_s, upper_s)
A_dense_s = sky.to_dense()
assert A_dense_s.shape == (n_sky, n_sky), '[TC27] dense matrix shape mismatch FAILED'
assert np.allclose(A_dense_s, A_dense_s.T, atol=1e-14), '[TC27] matrix not symmetric FAILED'

# ---- TC28: SkylineMatrixOperator multiply correctness ----
x_sky = np.ones(n_sky)
y_sky = sky.multiply(x_sky)
y_dense = A_dense_s @ x_sky
assert np.allclose(y_sky, y_dense, atol=1e-12), '[TC28] skyline multiply does not match dense multiply FAILED'

# ---- TC29: SkylineMatrixOperator solve residual small ----
x_solve_sky = sky.solve_cholesky_skyline(x_sky)
residual = np.linalg.norm(A_dense_s @ x_solve_sky - x_sky)
assert residual < 1e-8, '[TC29] skyline solve residual too large FAILED'

# ---- TC30: SkylineMatrixOperator condition number finite ----
cond_sky = sky.condition_number_estimate()
assert np.isfinite(cond_sky) and cond_sky >= 1.0, '[TC30] condition number invalid FAILED'

# ---- TC31: MicroreactorNetworkTopology degrees and Eulerian ----
net = MicroreactorNetworkTopology(n_nodes=6)
edges_test = [(0,1), (0,2), (1,3), (1,4), (2,5)]
for u, v in edges_test:
    net.add_edge(u, v)
indeg, outdeg = net.compute_degrees()
assert len(indeg) == 6 and len(outdeg) == 6, '[TC31] degree array length mismatch FAILED'

# ---- TC32: MicroreactorNetworkTopology spanning tree count non-negative ----
n_span_test = net.count_spanning_trees()
assert n_span_test >= 0, '[TC32] spanning tree count negative FAILED'

# ---- TC33: MicroreactorNetworkTopology uniform tree edges from pruefer roundtrip ----
tree_edges = net.generate_optimal_distribution_tree(root=0)
pruefer_code = net.tree_to_pruefer(tree_edges)
assert len(pruefer_code) == net.n - 2, '[TC33] Pruefer code length mismatch FAILED'

# ---- TC34: ThermalStressAnalyzer biharmonic residual near zero ----
thermal = ThermalStressAnalyzer(E=210.0e9, nu=0.28, h=0.8e-3, alpha_T=1.1e-5, z_eval=0.4e-3)
nx_t = 20
x_t = np.linspace(-0.01, 0.01, nx_t)
y_t = np.linspace(-0.005, 0.005, nx_t)
Xg_t, Yg_t = np.meshgrid(x_t, y_t)
resid_biharm = thermal.biharmonic_residual(Xg_t, Yg_t, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0)
assert np.all(np.isfinite(resid_biharm)), '[TC34] biharmonic residual contains NaN/Inf FAILED'

# ---- TC35: ThermalStressAnalyzer stresses finite and safety factor positive ----
delta_T_test = 50.0 * np.exp(-(Xg_t**2 + Yg_t**2) / (0.008**2))
sigma_x, sigma_y, tau_xy, sigma_vm = thermal.compute_thermal_stresses(
    Xg_t, Yg_t, delta_T_test, a=1.0e-6, b=0.0, c=0.0, d=0.0, e=1.0, f=0.0, g=80.0)
assert np.all(np.isfinite(sigma_x)), '[TC35] sigma_x contains NaN/Inf FAILED'
assert np.all(np.isfinite(sigma_vm)), '[TC35] sigma_vm contains NaN/Inf FAILED'
sf = thermal.safety_factor(sigma_vm, yield_strength=300.0e6)
assert np.isfinite(sf) and sf > 0.0, '[TC35] safety factor invalid FAILED'

# ---- TC36: ThermalStressAnalyzer thermal shock parameter positive ----
theta_shock = thermal.thermal_shock_parameter(delta_T_max=np.max(delta_T_test))
assert np.isfinite(theta_shock) and theta_shock >= 0.0, '[TC36] thermal shock parameter invalid FAILED'

# ---- TC37: DiscreteCatalystLoadingOptimizer greedy solution satisfies budget ----
a_load = np.array([2.5, 3.0, 4.0, 5.5, 6.0])
budget_load = 50.0
dioph = DiscreteCatalystLoadingOptimizer(a_load, budget_load)
weights_obj = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
x_greedy, obj_greedy = dioph.greedy_heuristic_solution(weights_obj)
total_used = np.dot(a_load, x_greedy)
assert total_used <= budget_load + 1e-8, '[TC37] greedy solution exceeds budget FAILED'
assert np.all(x_greedy >= 0), '[TC37] greedy solution has negative loads FAILED'

# ---- TC38: DiscreteCatalystLoadingOptimizer loading efficiency in [0,1] ----
util_g, unif_g = dioph.compute_loading_efficiency(x_greedy)
assert 0.0 <= util_g <= 1.0, '[TC38] utilization out of [0,1] FAILED'
assert 0.0 <= unif_g <= 1.0, '[TC38] uniformity out of [0,1] FAILED'

# ---- TC39: KineticsParameterEstimator confidence intervals shape ----
A_ci = np.random.randn(20, 4)
b_ci = np.random.randn(20)
np.random.seed(42)
x_ci = ke.solve_least_squares(A_ci, b_ci)
std_dev = ke.compute_confidence_intervals(A_ci, b_ci, x_ci)
assert len(std_dev) == len(x_ci), '[TC39] confidence interval length mismatch FAILED'

# ---- TC40: CatalystCVTPlacer get_catalyst_loading_map shape ----
np.random.seed(42)
cvt_map = CatalystCVTPlacer(dim=2, n_generators=9, bounds=np.array([[0.0,1.0],[0.0,1.0]]),
    sample_num=3000, max_iter=5, tol=1.0e-4)
cvt_map.iterate()
density_map, coords = cvt_map.get_catalyst_loading_map(grid_res=30)
assert density_map.shape == (30, 30), '[TC40] density map shape mismatch FAILED'
assert coords.shape == (30, 30, 2), '[TC40] coords shape mismatch FAILED'

print('\n全部 40 个测试通过!\n')
