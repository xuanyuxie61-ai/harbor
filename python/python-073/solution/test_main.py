#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
高超声速边界层转捩预测统一入口

科学问题：
计算流体力学 —— 高超声速边界层转捩的谱稳定性分析与热-结构耦合预测

本程序执行完整计算流程：
    1. 生成高超声速平板/前缘边界层计算网格
    2. 求解可压缩边界层自相似温度场与速度剖面
    3. 基于 Chebyshev 谱方法进行线性稳定性分析 (LST)
    4. 计算 e^N 扰动放大因子并预测转捩位置
    5. 多展向站位的转捩前沿优化
    6. 蒙特卡洛不确定性量化
    7. 输出计算报告与数据文件

所有参数已内嵌，无需命令行输入，直接运行即可。
"""

import os
import sys
import numpy as np
from math import sqrt, pi

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from mesh_generator import BoundaryLayerMesh, sphere_wavevector_grid, save_xy_data
from fem_basis import tet4_basis, tetrahedron_volume, build_fem_mass_matrix
from thermal_solver import HypersonicThermalSolver
from stability_analysis import CompressibleLST, track_eigenvalue_mode
from spectral_integrator import (
    chebyshev1_exactness_test,
    sphere_triangle_quad_icos1c,
    integrate_boundary_layer_growth,
    amplification_factor_integral
)
from transition_predictor import (
    e_n_method,
    compute_growth_rate_profile,
    multi_station_transition_prediction,
    optimize_transition_front,
    receptivity_coefficient
)
from monte_carlo_sampler import HypersonicParameterSampler, random_transition_model
from data_io import (
    write_transition_report,
    write_eigenvalue_spectrum,
    write_xy_data as io_write_xy
)
from utils import (
    optimal_chebyshev_order,
    blasius_function,
    compressible_blasius_velocity,
    normalize_array
)


def main():
    """
    主程序入口，零参数运行。
    """
    print("=" * 72)
    print("  高超声速边界层转捩预测 — 博士级科研代码合成项目")
    print("  领域: 计算流体力学 (CFD) — 高超声速边界层转捩")
    print("=" * 72)

    # ========================================================================
    # 步骤 1: 全局参数设定
    # ========================================================================
    Ma = 6.0               # 马赫数
    Re_L = 1.0e6           # 基于长度 L 的雷诺数
    Pr = 0.72              # 普朗特数
    gamma = 1.4            # 比热比
    Tw_Te = 0.6            # 冷壁条件 (T_w / T_e)
    L = 1.0                # 特征长度 [m]
    N_eta = 200            # 法向节点数
    eta_max = 12.0         # 相似变量外边界

    print("\n【全局参数】")
    print(f"  Ma = {Ma}, Re_L = {Re_L:.2e}, Pr = {Pr}, γ = {gamma}")
    print(f"  Tw/Te = {Tw_Te}, L = {L} m")

    # ========================================================================
    # 步骤 2: 边界层网格生成 (mesh_generator)
    # ========================================================================
    print("\n【步骤 2】边界层网格生成 ...")
    mesh = BoundaryLayerMesh(L=L, H=0.1, Nx=60, Ny=50, Re=Re_L, Ma=Ma)
    nodes, nx, ny = mesh.generate_flat_plate_mesh()
    triangles = mesh.generate_triangles_from_structured(nx, ny)
    neighbors = mesh.triangle_neighbors(triangles.shape[0], triangles)
    boundaries = mesh.boundary_nodes(nx, ny)

    print(f"  生成节点数: {len(nodes)}")
    print(f"  生成三角形数: {len(triangles)}")
    print(f"  壁面节点数: {len(boundaries['wall'])}")

    # 球面波矢方向网格 (sphere_llt_grid)
    wavevectors = sphere_wavevector_grid(lat_num=8, long_num=16)
    print(f"  波矢方向离散点数: {len(wavevectors)}")

    # ========================================================================
    # 步骤 3: 基流求解 — 可压缩边界层温度场与速度剖面 (thermal_solver)
    # ========================================================================
    print("\n【步骤 3】可压缩边界层基流求解 ...")
    thermal = HypersonicThermalSolver(
        Ma=Ma, Re=Re_L, Pr=Pr, gamma=gamma,
        Tw_over_Te=Tw_Te, L=L, N_eta=N_eta, eta_max=eta_max
    )
    solution = thermal.solve_self_similar_energy(epsilon=1e-8, max_iter=20000)

    print(f"  迭代收敛: {solution['iterations']} 步")
    print(f"  最终残差: {solution['diff']:.4e}")

    # 壁面热流与摩阻
    St = thermal.compute_wall_heat_flux(solution)
    cf = thermal.compute_skin_friction(solution)
    print(f"  壁面斯坦顿数近似: {St:.4e}")
    print(f"  壁面摩擦系数: {cf:.4e}")

    # 提取剖面
    eta = solution['eta']
    T_prof = solution['T']
    u_prof = solution['u']
    mu_prof = solution['mu']
    rho_prof = solution['rho']

    # ========================================================================
    # 步骤 4: 有限元基函数验证 (fem_basis)
    # ========================================================================
    print("\n【步骤 4】有限元基函数验证 ...")
    # 构造一个参考四面体并映射到物理空间
    t_ref = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=float)
    p_test = np.array([[0.25], [0.25], [0.25]])
    phi = tet4_basis(t_ref, p_test)
    vol = tetrahedron_volume(t_ref)
    print(f"  参考四面体体积: {vol:.6f}")
    print(f"  重心处基函数值: {phi.flatten()}")
    print(f"  基函数和: {np.sum(phi):.6f} (应为 1.0)")

    # 构造 FEM 质量矩阵与刚度矩阵（小规模测试）
    nodes_3d = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=float)
    tets = np.array([
        [0, 1, 2, 3], [1, 4, 2, 5], [2, 4, 6, 5],
        [3, 2, 5, 6], [1, 5, 3, 2], [4, 5, 6, 7]
    ], dtype=int)
    M_fem = build_fem_mass_matrix(nodes_3d, tets)
    print(f"  FEM 质量矩阵条件数: {np.linalg.cond(M_fem):.4e}")

    # ========================================================================
    # 步骤 5: 线性稳定性分析 (stability_analysis)
    # ========================================================================
    print("\n【步骤 5】线性稳定性分析 (LST) ...")

    # 选取最优 Chebyshev 阶数
    N_cheb = optimal_chebyshev_order(80, max_prime=5)
    print(f"  最优 Chebyshev 阶数: {N_cheb}")

    lst = CompressibleLST(Ma=Ma, Re=Re_L, Pr=Pr, gamma=gamma, N=N_cheb)
    lst.set_baseflow(eta, u_prof, T_prof, mu_prof)

    # 时间模式特征值（固定波数 α）
    alpha_test = 0.3
    eigvals = lst.temporal_eigenvalues(alpha=alpha_test, beta=0.0)
    print(f"  波数 α={alpha_test} 时，最不稳定特征值:")
    print(f"    ω = {eigvals[0]:.6f}")
    print(f"    时间增长率 Im(ω) = {eigvals[0].imag:.6e}")

    # Jordan 分析（非模态增长）
    jordan_info = lst.jordan_analysis(alpha=alpha_test, beta=0.0)
    print(f"  模态矩阵条件数: {jordan_info['condition_number']:.4e}")
    print(f"  最大 Jordan 块大小: {jordan_info['max_jordan_block']}")
    print(f"  瞬态增长上界估计: {jordan_info['transient_growth_bound']:.4e}")

    # 模态追踪（随波数扫描）
    alpha_list = np.linspace(0.05, 0.8, 30)
    tracked = track_eigenvalue_mode(alpha_list, lst, beta=0.0)
    growth_rates = [np.imag(om) if not np.isnan(om) else -1e9 for om in tracked]
    max_growth_idx = int(np.argmax(growth_rates))
    print(f"  最大不稳定波数: α={alpha_list[max_growth_idx]:.4f}")
    print(f"  对应最大增长率: {growth_rates[max_growth_idx]:.6e}")

    # ========================================================================
    # 步骤 6: 谱积分与精确度验证 (spectral_integrator)
    # ========================================================================
    print("\n【步骤 6】谱方法与积分验证 ...")

    # Chebyshev 求积精确度测试
    exactness = chebyshev1_exactness_test(n=16, degree_max=20)
    max_err = max(err for _, err in exactness)
    print(f"  Gauss-Chebyshev (n=16) 最大误差: {max_err:.4e}")

    # 球面三角形积分测试（验证立体角）
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    area_est, npts = sphere_triangle_quad_icos1c(a, b, c, factor=4,
                                                  func=lambda p: 1.0)
    exact_area = pi / 2.0  # 八分之一个球面 = π/2
    print(f"  球面三角形积分测试 (factor=4):")
    print(f"    估计面积: {area_est:.6f}, 精确值: {exact_area:.6f}, 误差: {abs(area_est-exact_area):.4e}")

    # e^N 放大因子积分
    Re_x_range = np.linspace(2e5, 8e6, 300)
    ai_profile = compute_growth_rate_profile(Re_x_range, Ma, Re_L, Tw_Te)
    Re_N, N_prof = amplification_factor_integral(Re_x_range, ai_profile, method='trapz')
    print(f"  e^N 积分完成，N(Re=8e6) = {N_prof[-1]:.3f}")

    # ========================================================================
    # 步骤 7: 转捩位置预测 (transition_predictor)
    # ========================================================================
    print("\n【步骤 7】转捩位置预测 ...")

    # 单站位 e^N 方法
    Re_xt_single, N_single = e_n_method(Re_x_range, ai_profile, N_cr=9.0)
    print(f"  单站位转捩雷诺数 Re_xt = {Re_xt_single:.4e}")

    # 多展向站位预测
    z_stations = np.linspace(-0.5, 0.5, 11)
    np.random.seed(42)
    roughness = 0.001 + 0.003 * np.random.rand(len(z_stations))
    multi_result = multi_station_transition_prediction(
        Ma, Re_L, Tw_Te, Tu=0.005,
        z_stations=z_stations,
        roughness_array=roughness,
        N_cr=9.0
    )
    print(f"  多站位平均转捩雷诺数: {multi_result['mean_Re_xt']:.4e}")
    print(f"  展向标准差: {multi_result['std_Re_xt']:.4e}")

    # 转捩前沿优化（TSP 思想）
    penalties = roughness * 1e5
    optimized_xt, cost_hist = optimize_transition_front(
        z_stations, multi_result['Re_xt'], penalties,
        max_iter=3000, lambda_penalty=0.3
    )
    print(f"  优化后转捩前沿光滑性: {np.sum(np.diff(optimized_xt)**2):.4e}")
    print(f"  优化成本下降: {cost_hist[0]:.4e} → {cost_hist[-1]:.4e}")

    # 感受性系数
    C_rec = receptivity_coefficient(Ma, Tw_Te, Tu=0.005)
    print(f"  感受性系数估计: {C_rec:.4e}")

    # ========================================================================
    # 步骤 8: 蒙特卡洛不确定性量化 (monte_carlo_sampler)
    # ========================================================================
    print("\n【步骤 8】蒙特卡洛不确定性量化 ...")
    sampler = HypersonicParameterSampler(
        Ma_range=(5.0, 8.0),
        Re_range=(5e5, 2e7),
        Tw_Te_range=(0.4, 1.5),
        Tu_range=(0.001, 0.02)
    )

    # LHS 采样均匀性统计
    lhs_samples = sampler.lhs_sampling(n_samples=200)
    mu_dist, var_dist = sampler.parameter_distance_stats(lhs_samples)
    print(f"  LHS 样本对平均距离: {mu_dist:.4f}")
    print(f"  距离方差: {var_dist:.4e}")

    # 序贯最优采样
    opt_sample = sampler.sequential_optimal_sampling(n_total=100)
    print(f"  序贯采样策略: {opt_sample['strategy']}")
    print(f"  最优样本价值: {opt_sample['best_value']:.4f}")

    # 不确定性传播
    mc_result = sampler.uncertainty_propagation(
        random_transition_model, n_samples=300
    )
    print(f"  转捩雷诺数均值: {mc_result['mean']:.4e}")
    print(f"  转捩雷诺数标准差: {mc_result['std']:.4e}")
    ci_low, ci_high = mc_result['ci95']
    print(f"  95% 置信区间: [{ci_low:.4e}, {ci_high:.4e}]")

    # ========================================================================
    # 步骤 9: 结果输出与报告生成 (data_io)
    # ========================================================================
    print("\n【步骤 9】结果输出 ...")
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    # 输出基流剖面
    io_write_xy(os.path.join(out_dir, "baseflow_profile.xy"),
                eta, u_prof, header="Eta U_velocity")
    io_write_xy(os.path.join(out_dir, "temperature_profile.xy"),
                eta, T_prof, header="Eta Temperature_ratio")

    # 输出特征值谱
    write_eigenvalue_spectrum(
        os.path.join(out_dir, "eigenvalue_spectrum.dat"),
        alpha_list,
        np.array(tracked),
        labels=[f"mode_{i}" for i in range(len(alpha_list))]
    )

    # 输出转捩报告
    results = {
        'Ma': Ma,
        'Re': Re_L,
        'Pr': Pr,
        'Tw_Te': Tw_Te,
        'thermal': {
            'iterations': solution['iterations'],
            'diff': solution['diff'],
            'T': T_prof
        },
        'stability': {
            'max_temporal_growth_rate': eigvals[0].imag,
            'condition_number': jordan_info['condition_number'],
            'max_jordan_block': jordan_info['max_jordan_block']
        },
        'transition': {
            'mean_Re_xt': multi_result['mean_Re_xt'],
            'std_Re_xt': multi_result['std_Re_xt'],
            'smoothness': multi_result['smoothness']
        },
        'monte_carlo': {
            'mean': mc_result['mean'],
            'std': mc_result['std'],
            'ci95': mc_result['ci95']
        }
    }
    write_transition_report(os.path.join(out_dir, "transition_report.txt"), results)

    # 保存球面波矢方向
    np.savetxt(os.path.join(out_dir, "wavevectors.dat"), wavevectors,
               fmt='%.6f', header='kx ky kz')

    print("\n  输出文件:")
    print(f"    {out_dir}/baseflow_profile.xy")
    print(f"    {out_dir}/temperature_profile.xy")
    print(f"    {out_dir}/eigenvalue_spectrum.dat")
    print(f"    {out_dir}/transition_report.txt")
    print(f"    {out_dir}/wavevectors.dat")

    # ========================================================================
    # 步骤 10: 边界条件与数值鲁棒性验证
    # ========================================================================
    print("\n【步骤 10】数值鲁棒性验证 ...")

    # 验证基函数 partition of unity
    phi_sum = np.sum(phi)
    assert abs(phi_sum - 1.0) < 1e-10, "基函数不满足 partition of unity"
    print("  [通过] FEM 基函数 partition of unity")

    # 验证温度边界条件
    assert abs(T_prof[0] - Tw_Te) < 1e-3, "壁面温度边界条件不满足"
    assert abs(T_prof[-1] - 1.0) < 1e-3, "远场温度边界条件不满足"
    print("  [通过] 温度边界条件")

    # 验证速度边界条件
    assert abs(u_prof[0]) < 1e-3, "壁面无滑移条件不满足"
    assert abs(u_prof[-1] - 1.0) < 1e-3, "远场速度边界条件不满足"
    print("  [通过] 速度边界条件")

    # 验证质量守恒（积分连续性）
    mass_flux = np.trapezoid(rho_prof * u_prof, eta)
    assert mass_flux > 0, "质量通量非正"
    print(f"  [通过] 质量通量积分: {mass_flux:.6f}")

    # 验证 Chebyshev 求积对低次多项式精确
    assert max_err < 1e-12, "Chebyshev 求积精度不足"
    print("  [通过] Chebyshev 求积精确度")

    print("\n" + "=" * 72)
    print("  计算流程全部完成，无报错。")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: prime_factors 分解 60 为 [2,2,3,5] ----
from utils import prime_factors
factors = prime_factors(60)
assert factors == [2, 2, 3, 5], '[TC01] prime_factors(60) FAILED'

# ---- TC02: 最优 Chebyshev 阶数 80 满足 81=3^4 ----
from utils import optimal_chebyshev_order
N_opt = optimal_chebyshev_order(80, max_prime=5)
assert N_opt == 80, '[TC02] optimal_chebyshev_order(80) FAILED'

# ---- TC03: normalize_array minmax 归一化到 [0,1] ----
from utils import normalize_array
arr = np.array([10.0, 20.0, 30.0])
norm = normalize_array(arr, method="minmax")
assert np.allclose(norm, [0.0, 0.5, 1.0]), '[TC03] normalize_array minmax FAILED'

# ---- TC04: safe_divide 零除返回 fill_value ----
from utils import safe_divide
result = safe_divide(5.0, 0.0, fill_value=999.0)
assert result == 999.0, '[TC04] safe_divide zero division FAILED'

# ---- TC05: Blasius 壁面无滑移与远场渐近 ----
from utils import blasius_function
eta_test = np.array([0.0, 12.0])
f, fp, fpp = blasius_function(eta_test)
assert abs(fp[0]) < 1e-3, '[TC05] Blasius fp(0) FAILED'
assert abs(fp[-1] - 1.0) < 1e-3, '[TC05] Blasius fp(inf) FAILED'

# ---- TC06: Sutherland 粘度在 T=T_ref 时等于 mu_ref ----
from utils import sutherland_viscosity
mu_ref = 1.7894e-5
mu = sutherland_viscosity(np.array([300.0]))
assert abs(mu[0] - mu_ref) < 1e-10, '[TC06] sutherland_viscosity T_ref FAILED'

# ---- TC07: Chebyshev 节点数量与范围 ----
from utils import chebyshev_nodes
nodes = chebyshev_nodes(8, a=0.0, b=1.0)
assert len(nodes) == 9, '[TC07] chebyshev_nodes length FAILED'
assert np.all((nodes >= 0.0) & (nodes <= 1.0)), '[TC07] chebyshev_nodes range FAILED'

# ---- TC08: Chebyshev 微分矩阵尺寸为 (n+1, n+1) ----
from utils import chebyshev_diff_matrix
D = chebyshev_diff_matrix(6)
assert D.shape == (7, 7), '[TC08] chebyshev_diff_matrix shape FAILED'

# ---- TC09: 单位四面体体积精确为 1/6 ----
from fem_basis import tetrahedron_volume
t_unit = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
vol = tetrahedron_volume(t_unit)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC09] tetrahedron_volume unit tet FAILED'

# ---- TC10: TET4 基函数在重心处满足 partition of unity ----
from fem_basis import tet4_basis
phi = tet4_basis(t_unit, np.array([[0.25], [0.25], [0.25]]))
assert abs(np.sum(phi) - 1.0) < 1e-10, '[TC10] tet4_basis partition of unity FAILED'

# ---- TC11: 参考坐标与物理坐标互逆映射 ----
from fem_basis import reference_to_physical_tet4, physical_to_reference_tet4
xi_test = np.array([0.2, 0.3, 0.1])
x_phys = reference_to_physical_tet4(t_unit, xi_test)
xi_back = physical_to_reference_tet4(t_unit, x_phys)
assert np.allclose(xi_test, xi_back, atol=1e-10), '[TC11] reference/physical inverse FAILED'

# ---- TC12: Chebyshev 求积对不超过 2n-1 次多项式精确 ----
from spectral_integrator import chebyshev1_exactness_test
exactness = chebyshev1_exactness_test(n=8, degree_max=10)
low_degree_err = [err for deg, err in exactness if deg <= 7]
assert all(err < 1e-14 for err in low_degree_err), '[TC12] chebyshev1_exactness FAILED'

# ---- TC13: 球面直角三角形面积为 pi/2 ----
from spectral_integrator import sphere01_triangle_area
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
c = np.array([0.0, 0.0, 1.0])
area = sphere01_triangle_area(a, b, c)
assert abs(area - pi/2.0) < 1e-3, '[TC13] sphere01_triangle_area FAILED'

# ---- TC14: 零增长率时 N 积分保持为零 ----
from spectral_integrator import integrate_boundary_layer_growth
x_test = np.linspace(0, 1, 11)
alpha_zero = np.zeros_like(x_test)
N_prof = integrate_boundary_layer_growth(x_test, alpha_zero)
assert np.allclose(N_prof, 0.0), '[TC14] integrate_boundary_layer_growth zero FAILED'

# ---- TC15: trapz 与 simpson 积分方法一致性 ----
from spectral_integrator import amplification_factor_integral
Re_test = np.linspace(1e5, 2e6, 301)
ai_test = -0.001 * np.ones_like(Re_test)
_, N_trapz = amplification_factor_integral(Re_test, ai_test, method='trapz')
_, N_simp = amplification_factor_integral(Re_test, ai_test, method='simpson')
assert abs(N_trapz[-1] - N_simp[-1]) < 1e-3, '[TC15] amplification methods consistency FAILED'

# ---- TC16: e_n_method 输出 N 长度与输入一致 ----
from transition_predictor import e_n_method, compute_growth_rate_profile
Re_x = np.linspace(1e5, 5e6, 100)
ai = compute_growth_rate_profile(Re_x, Ma=6.0, Re_unit=1e6, Tw_Te=1.0)
Re_xt, N_prof = e_n_method(Re_x, ai, N_cr=9.0)
assert len(N_prof) == len(Re_x), '[TC16] e_n_method output length FAILED'

# ---- TC17: 空间增长率非正（负值表示扰动增长） ----
from transition_predictor import compute_growth_rate_profile
Re_x = np.linspace(1e5, 5e6, 50)
ai = compute_growth_rate_profile(Re_x, Ma=6.0, Tw_Te=1.0)
assert np.all(ai <= 0.0), '[TC17] growth_rate_profile sign FAILED'

# ---- TC18: 转捩前沿成本平移不变性 ----
from transition_predictor import transition_front_cost
pos1 = np.array([1.0, 2.0, 3.0])
pen = np.array([0.1, 0.1, 0.1])
c1 = transition_front_cost(pos1, pen)
c2 = transition_front_cost(pos1 + 5.0, pen)
assert c1 == c2, '[TC18] transition_front_cost translation invariance FAILED'

# ---- TC19: 感受性系数非负 ----
from transition_predictor import receptivity_coefficient
C_rec = receptivity_coefficient(Ma=6.0, Tw_Te=1.0, Tu=0.005)
assert C_rec >= 0.0, '[TC19] receptivity_coefficient sign FAILED'

# ---- TC20: 平板网格节点数匹配 Nx*Ny ----
from mesh_generator import BoundaryLayerMesh
mesh = BoundaryLayerMesh(L=1.0, H=0.1, Nx=10, Ny=8, Re=1e6, Ma=6.0)
nodes, nx, ny = mesh.generate_flat_plate_mesh()
assert len(nodes) == nx * ny, '[TC20] flat_plate mesh size FAILED'

# ---- TC21: 三角形邻居边界标记为 -1 ----
tri_small = np.array([[0, 1, 2], [1, 3, 2]])
neighbors = mesh.triangle_neighbors(2, tri_small)
assert np.any(neighbors == -1), '[TC21] triangle_neighbors boundary FAILED'

# ---- TC22: 边界节点集合非空 ----
bnds = mesh.boundary_nodes(nx, ny)
total_boundary = len(bnds['wall']) + len(bnds['inlet']) + len(bnds['outlet']) + len(bnds['farfield'])
assert total_boundary > 0, '[TC22] boundary_nodes empty FAILED'

# ---- TC23: 球面波矢方向均为单位向量 ----
from mesh_generator import sphere_wavevector_grid
wv = sphere_wavevector_grid(lat_num=4, long_num=8)
norms = np.linalg.norm(wv, axis=1)
assert np.allclose(norms, 1.0), '[TC23] sphere_wavevector_grid unit norm FAILED'

# ---- TC24: 热求解器壁温与远场温度边界条件 ----
from thermal_solver import HypersonicThermalSolver
thermal = HypersonicThermalSolver(Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4,
                                   Tw_over_Te=0.6, L=1.0, N_eta=50, eta_max=12.0)
sol = thermal.solve_self_similar_energy(epsilon=1e-6, max_iter=10000)
assert abs(sol['T'][0] - 0.6) < 1e-2, '[TC24] thermal wall temp BC FAILED'
assert abs(sol['T'][-1] - 1.0) < 1e-2, '[TC24] thermal farfield temp BC FAILED'

# ---- TC25: LST 时间特征值为复数数组 ----
from stability_analysis import CompressibleLST
from utils import blasius_function, sutherland_viscosity
lst = CompressibleLST(Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4, N=40)
eta_c = np.linspace(0, 12, 200)
f_c, fp_c, _ = blasius_function(eta_c)
u_c = np.clip(fp_c, 0, 1)
T_c = np.ones_like(eta_c)
mu_c = sutherland_viscosity(T_c)
lst.set_baseflow(eta_c, u_c, T_c, mu_c)
eigvals = lst.temporal_eigenvalues(alpha=0.3, beta=0.0)
assert np.iscomplexobj(eigvals), '[TC25] LST eigenvalue type FAILED'

# ---- TC26: 模态追踪输出长度与输入波数列表一致 ----
from stability_analysis import track_eigenvalue_mode
alpha_list = np.linspace(0.05, 0.3, 5)
tracked = track_eigenvalue_mode(alpha_list, lst, beta=0.0)
assert len(tracked) == len(alpha_list), '[TC26] track_eigenvalue_mode length FAILED'

# ---- TC27: LHS 采样马赫数在指定范围内 ----
from monte_carlo_sampler import HypersonicParameterSampler
np.random.seed(42)
sampler = HypersonicParameterSampler(Ma_range=(5.0, 8.0), Re_range=(5e5, 2e7),
                                      Tw_Te_range=(0.4, 1.5), Tu_range=(0.001, 0.02))
samples = sampler.lhs_sampling(n_samples=20)
assert np.all((samples[:, 0] >= 5.0) & (samples[:, 0] <= 8.0)), '[TC27] LHS Ma range FAILED'

# ---- TC28: 随机转捩模型返回值不低于下限 1e4 ----
from monte_carlo_sampler import random_transition_model
np.random.seed(42)
Re_t = random_transition_model(Ma=6.0, Re=1e6, Tw_Te=1.0, Tu=0.005)
assert Re_t >= 1e4, '[TC28] random_transition_model lower bound FAILED'

# ---- TC29: XY 数据文件读写一致性 ----
import tempfile
from data_io import write_xy_data, read_xy_data
x_test = np.array([1.0, 2.0, 3.0])
y_test = np.array([4.0, 5.0, 6.0])
with tempfile.NamedTemporaryFile(mode='w', suffix='.xy', delete=False) as tf:
    tmpname = tf.name
write_xy_data(tmpname, x_test, y_test)
x_read, y_read = read_xy_data(tmpname)
os.remove(tmpname)
assert np.allclose(x_read, x_test) and np.allclose(y_read, y_test), '[TC29] XY data IO consistency FAILED'

# ---- TC30: 主程序输出文件已生成 ----
assert os.path.exists("baseflow_profile.xy"), '[TC30] baseflow_profile.xy missing FAILED'
assert os.path.exists("transition_report.txt"), '[TC30] transition_report.txt missing FAILED'

print('\n全部 30 个测试通过!\n')
