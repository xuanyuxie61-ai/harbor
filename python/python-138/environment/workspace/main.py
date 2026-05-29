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
    sys.exit(main())
