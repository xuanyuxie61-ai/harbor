#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from math import sqrt, pi




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
    print("=" * 72)
    print("  高超声速边界层转捩预测 — 博士级科研代码合成项目")
    print("  领域: 计算流体力学 (CFD) — 高超声速边界层转捩")
    print("=" * 72)




    Ma = 6.0
    Re_L = 1.0e6
    Pr = 0.72
    gamma = 1.4
    Tw_Te = 0.6
    L = 1.0
    N_eta = 200
    eta_max = 12.0

    print("\n【全局参数】")
    print(f"  Ma = {Ma}, Re_L = {Re_L:.2e}, Pr = {Pr}, γ = {gamma}")
    print(f"  Tw/Te = {Tw_Te}, L = {L} m")




    print("\n【步骤 2】边界层网格生成 ...")
    mesh = BoundaryLayerMesh(L=L, H=0.1, Nx=60, Ny=50, Re=Re_L, Ma=Ma)
    nodes, nx, ny = mesh.generate_flat_plate_mesh()
    triangles = mesh.generate_triangles_from_structured(nx, ny)
    neighbors = mesh.triangle_neighbors(triangles.shape[0], triangles)
    boundaries = mesh.boundary_nodes(nx, ny)

    print(f"  生成节点数: {len(nodes)}")
    print(f"  生成三角形数: {len(triangles)}")
    print(f"  壁面节点数: {len(boundaries['wall'])}")


    wavevectors = sphere_wavevector_grid(lat_num=8, long_num=16)
    print(f"  波矢方向离散点数: {len(wavevectors)}")




    print("\n【步骤 3】可压缩边界层基流求解 ...")
    thermal = HypersonicThermalSolver(
        Ma=Ma, Re=Re_L, Pr=Pr, gamma=gamma,
        Tw_over_Te=Tw_Te, L=L, N_eta=N_eta, eta_max=eta_max
    )
    solution = thermal.solve_self_similar_energy(epsilon=1e-8, max_iter=20000)

    print(f"  迭代收敛: {solution['iterations']} 步")
    print(f"  最终残差: {solution['diff']:.4e}")


    St = thermal.compute_wall_heat_flux(solution)
    cf = thermal.compute_skin_friction(solution)
    print(f"  壁面斯坦顿数近似: {St:.4e}")
    print(f"  壁面摩擦系数: {cf:.4e}")


    eta = solution['eta']
    T_prof = solution['T']
    u_prof = solution['u']
    mu_prof = solution['mu']
    rho_prof = solution['rho']




    print("\n【步骤 4】有限元基函数验证 ...")

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




    print("\n【步骤 5】线性稳定性分析 (LST) ...")


    N_cheb = optimal_chebyshev_order(80, max_prime=5)
    print(f"  最优 Chebyshev 阶数: {N_cheb}")

    lst = CompressibleLST(Ma=Ma, Re=Re_L, Pr=Pr, gamma=gamma, N=N_cheb)
    lst.set_baseflow(eta, u_prof, T_prof, mu_prof)


    alpha_test = 0.3
    eigvals = lst.temporal_eigenvalues(alpha=alpha_test, beta=0.0)
    print(f"  波数 α={alpha_test} 时，最不稳定特征值:")
    print(f"    ω = {eigvals[0]:.6f}")
    print(f"    时间增长率 Im(ω) = {eigvals[0].imag:.6e}")


    jordan_info = lst.jordan_analysis(alpha=alpha_test, beta=0.0)
    print(f"  模态矩阵条件数: {jordan_info['condition_number']:.4e}")
    print(f"  最大 Jordan 块大小: {jordan_info['max_jordan_block']}")
    print(f"  瞬态增长上界估计: {jordan_info['transient_growth_bound']:.4e}")


    alpha_list = np.linspace(0.05, 0.8, 30)
    tracked = track_eigenvalue_mode(alpha_list, lst, beta=0.0)
    growth_rates = [np.imag(om) if not np.isnan(om) else -1e9 for om in tracked]
    max_growth_idx = int(np.argmax(growth_rates))
    print(f"  最大不稳定波数: α={alpha_list[max_growth_idx]:.4f}")
    print(f"  对应最大增长率: {growth_rates[max_growth_idx]:.6e}")




    print("\n【步骤 6】谱方法与积分验证 ...")


    exactness = chebyshev1_exactness_test(n=16, degree_max=20)
    max_err = max(err for _, err in exactness)
    print(f"  Gauss-Chebyshev (n=16) 最大误差: {max_err:.4e}")


    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    area_est, npts = sphere_triangle_quad_icos1c(a, b, c, factor=4,
                                                  func=lambda p: 1.0)
    exact_area = pi / 2.0
    print(f"  球面三角形积分测试 (factor=4):")
    print(f"    估计面积: {area_est:.6f}, 精确值: {exact_area:.6f}, 误差: {abs(area_est-exact_area):.4e}")


    Re_x_range = np.linspace(2e5, 8e6, 300)
    ai_profile = compute_growth_rate_profile(Re_x_range, Ma, Re_L, Tw_Te)
    Re_N, N_prof = amplification_factor_integral(Re_x_range, ai_profile, method='trapz')
    print(f"  e^N 积分完成，N(Re=8e6) = {N_prof[-1]:.3f}")




    print("\n【步骤 7】转捩位置预测 ...")


    Re_xt_single, N_single = e_n_method(Re_x_range, ai_profile, N_cr=9.0)
    print(f"  单站位转捩雷诺数 Re_xt = {Re_xt_single:.4e}")


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


    penalties = roughness * 1e5
    optimized_xt, cost_hist = optimize_transition_front(
        z_stations, multi_result['Re_xt'], penalties,
        max_iter=3000, lambda_penalty=0.3
    )
    print(f"  优化后转捩前沿光滑性: {np.sum(np.diff(optimized_xt)**2):.4e}")
    print(f"  优化成本下降: {cost_hist[0]:.4e} → {cost_hist[-1]:.4e}")


    C_rec = receptivity_coefficient(Ma, Tw_Te, Tu=0.005)
    print(f"  感受性系数估计: {C_rec:.4e}")




    print("\n【步骤 8】蒙特卡洛不确定性量化 ...")
    sampler = HypersonicParameterSampler(
        Ma_range=(5.0, 8.0),
        Re_range=(5e5, 2e7),
        Tw_Te_range=(0.4, 1.5),
        Tu_range=(0.001, 0.02)
    )


    lhs_samples = sampler.lhs_sampling(n_samples=200)
    mu_dist, var_dist = sampler.parameter_distance_stats(lhs_samples)
    print(f"  LHS 样本对平均距离: {mu_dist:.4f}")
    print(f"  距离方差: {var_dist:.4e}")


    opt_sample = sampler.sequential_optimal_sampling(n_total=100)
    print(f"  序贯采样策略: {opt_sample['strategy']}")
    print(f"  最优样本价值: {opt_sample['best_value']:.4f}")


    mc_result = sampler.uncertainty_propagation(
        random_transition_model, n_samples=300
    )
    print(f"  转捩雷诺数均值: {mc_result['mean']:.4e}")
    print(f"  转捩雷诺数标准差: {mc_result['std']:.4e}")
    ci_low, ci_high = mc_result['ci95']
    print(f"  95% 置信区间: [{ci_low:.4e}, {ci_high:.4e}]")




    print("\n【步骤 9】结果输出 ...")
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)


    io_write_xy(os.path.join(out_dir, "baseflow_profile.xy"),
                eta, u_prof, header="Eta U_velocity")
    io_write_xy(os.path.join(out_dir, "temperature_profile.xy"),
                eta, T_prof, header="Eta Temperature_ratio")


    write_eigenvalue_spectrum(
        os.path.join(out_dir, "eigenvalue_spectrum.dat"),
        alpha_list,
        np.array(tracked),
        labels=[f"mode_{i}" for i in range(len(alpha_list))]
    )


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


    np.savetxt(os.path.join(out_dir, "wavevectors.dat"), wavevectors,
               fmt='%.6f', header='kx ky kz')

    print("\n  输出文件:")
    print(f"    {out_dir}/baseflow_profile.xy")
    print(f"    {out_dir}/temperature_profile.xy")
    print(f"    {out_dir}/eigenvalue_spectrum.dat")
    print(f"    {out_dir}/transition_report.txt")
    print(f"    {out_dir}/wavevectors.dat")




    print("\n【步骤 10】数值鲁棒性验证 ...")


    phi_sum = np.sum(phi)
    assert abs(phi_sum - 1.0) < 1e-10, "基函数不满足 partition of unity"
    print("  [通过] FEM 基函数 partition of unity")


    assert abs(T_prof[0] - Tw_Te) < 1e-3, "壁面温度边界条件不满足"
    assert abs(T_prof[-1] - 1.0) < 1e-3, "远场温度边界条件不满足"
    print("  [通过] 温度边界条件")


    assert abs(u_prof[0]) < 1e-3, "壁面无滑移条件不满足"
    assert abs(u_prof[-1] - 1.0) < 1e-3, "远场速度边界条件不满足"
    print("  [通过] 速度边界条件")


    mass_flux = np.trapezoid(rho_prof * u_prof, eta)
    assert mass_flux > 0, "质量通量非正"
    print(f"  [通过] 质量通量积分: {mass_flux:.6f}")


    assert max_err < 1e-12, "Chebyshev 求积精度不足"
    print("  [通过] Chebyshev 求积精确度")

    print("\n" + "=" * 72)
    print("  计算流程全部完成，无报错。")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
