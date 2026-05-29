"""
结构力学：大变形非线性有限元分析
=================================
统一入口模块 (零参数可运行)

本项目综合应用了以下15个种子项目的核心算法：
  - 743_mcnuggets_diophantine  -> 载荷步整数分解规划
  - 981_r8ge                  -> 稠密矩阵PLU分解与共轭梯度求解
  - 1247_tetrahedron_grid     -> 四面体均匀网格生成
  - 1229_test_zero            -> Brent/Newton非线性求根
  - 1238_tet_mesh_refine      -> 四面体网格自适应细化
  - 1059_sawtooth_ode         -> 锯齿波周期冲击载荷
  - 833_ode_trapezoidal       -> 隐式梯形时间积分
  - 350_fd_predator_prey      -> 损伤演化显式差分
  - 854_pce_ode_hermite       -> 多项式混沌展开UQ
  - 984_r8lt                  -> Cholesky分解与前代法
  - 1360_truncated_normal     -> 截断正态随机参数建模
  - 586_image_threshold       -> 应力场阈值分割
  - 779_monty_hall_simulation -> 蒙特卡洛统计验证
  - 1179_subset_sum_backtrack -> 自适应载荷步回溯控制
  - 1344_triangulation_orient -> 网格方向一致性修正

科学问题:
  考虑材料参数不确定性的超弹性体大变形非线性有限元分析。
  求解三维可压缩Neo-Hookean立方体在准静态压缩载荷下的变形响应，
  并采用多项式混沌展开(PCE)量化剪切模量不确定性对顶部位移的影响。
  同时引入连续损伤力学(CDM)模型描述大变形下的材料退化。
"""

import numpy as np
import time
import sys

# 本地模块导入
from tetrahedral_mesh import generate_cube_tetrahedral_mesh, check_mesh_quality, refine_tetrahedral_mesh, get_surface_triangles
from mesh_orientation import orient_tetrahedra, orient_surface_triangles
from hyperelastic_constitutive import solve_effective_shear_modulus
from stiffness_solver import plu_decompose, solve_plu, apply_dirichlet_to_system
from nonlinear_fem_core import assemble_global_system, compute_external_force
from boundary_conditions import apply_dirichlet_bc, apply_pressure_load
from load_stepping import AdaptiveLoadStepping
from damage_evolution import update_element_damage, compute_equivalent_strain_rate
from stress_field_segmentation import otsu_threshold, single_threshold_segmentation, compute_damage_zone_statistics, identify_critical_elements
from uncertainty_quantification import (
    hermite_polynomial, hermite_basis_vector, hermite_double_product, hermite_triple_product,
    truncated_normal_sample, generate_hermite_quadrature_points,
    pce_mean, pce_variance, pce_standard_deviation
)
from monte_carlo_sampler import monte_carlo_simulation, convergence_analysis


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def solve_static_nonlinear_fem(nodes, elements, mu, lam, pressure,
                                use_damage=False, alpha_damage=0.0,
                                max_nr_iter=15, nr_tol=1e-8):
    """
    求解准静态大变形非线性有限元问题。
    使用Newton-Raphson迭代 + 自适应载荷步。
    """
    n_nodes = nodes.shape[0]
    n_elements = elements.shape[0]
    n_dof = 3 * n_nodes

    # 边界条件: 底面 z=0 固定
    bc_dofs, bc_values = apply_dirichlet_bc(nodes, [(2, 0.0)])

    # 表面三角形与载荷
    surface_tris = get_surface_triangles(elements)
    # 顶面压力 (z=1)
    top_tris = []
    for tri in surface_tris:
        z_avg = np.mean(nodes[tri, 2])
        if abs(z_avg - 1.0) < 0.05:
            top_tris.append(tri)
    top_tris = np.array(top_tris, dtype=np.int32)

    # 初始化
    u = np.zeros(n_dof, dtype=np.float64)
    D_elements = np.zeros(n_elements, dtype=np.float64)
    eps_p_elements = np.zeros(n_elements, dtype=np.float64)

    # 自适应载荷步
    load_stepper = AdaptiveLoadStepping(lambda_max=1.0, initial_step=0.1,
                                         min_step=0.01, max_step=0.3,
                                         desired_iterations=5, max_total_steps=200)

    converged_steps = 0
    all_stress_data = []

    while True:
        lam, finished = load_stepper.next_lambda()
        if finished:
            break

        # 当前步目标载荷
        p_current = lam * pressure
        F_ext = apply_pressure_load(nodes, top_tris, p_current, direction=np.array([0.0, 0.0, -1.0]))

        # Newton-Raphson迭代
        u_step = u.copy()
        nr_converged = False
        nr_iters = 0

        for nr_it in range(max_nr_iter):
            # TODO: Hole 3 - 实现Newton-Raphson迭代的核心步骤:
            #   1. 组装全局刚度矩阵和内部力向量
            #   2. 计算残差并施加Dirichlet边界条件
            #   3. 使用PLU分解求解线性系统
            #   4. 更新位移并检查收敛
            raise NotImplementedError("Hole 3: 请实现Newton-Raphson迭代核心步骤")

        # 自适应步长调整
        load_stepper.adjust_step(nr_iters, nr_converged)

        if nr_converged:
            u = u_step
            converged_steps += 1
            all_stress_data = stress_list

            # 更新损伤
            if use_damage:
                # 简化的等效应变率估计 (准静态假设小应变率)
                eps_dot = np.ones(n_elements) * 1e-3
                sigma_vm_arr = np.array([s["sigma_vm"] for s in stress_list])
                D_elements, eps_p_elements = update_element_damage(
                    n_elements, D_elements, eps_p_elements,
                    dt=1.0, eps_dot_elements=eps_dot,
                    sigma_vm_elements=sigma_vm_arr,
                    sigma_y=1e7, A=alpha_damage, B=0.5, eps_f=0.5
                )
        else:
            # 不收敛则回溯
            load_stepper.reset_to_previous()
            if load_stepper.step < load_stepper.min_step * 1.01:
                print(f"    [LoadStep] 步长已达最小值，终止 at lambda={lam:.3f}")
                break

    return u, all_stress_data, D_elements, converged_steps, load_stepper.n_backtracks


def run_pce_uq(nodes, elements, pressure, mu_mean, mu_std, lam, degree=3):
    """
    使用多项式混沌展开量化剪切模量不确定性对顶部位移的影响。
    采用Gauss-Hermite数值积分计算PCE系数。
    """
    print_section("不确定性量化 (PCE)")
    print(f"  剪切模量分布: N({mu_mean}, {mu_std}^2)")
    print(f"  PCE阶数: {degree}")

    xi_pts, w_pts = generate_hermite_quadrature_points(n_points=max(degree + 1, 5))
    n_quad = len(xi_pts)
    print(f"  Gauss-Hermite积分点数: {n_quad}")

    # 获取顶面中心节点参考位移 (作为QoI)
    top_nodes = np.where(np.abs(nodes[:, 2] - 1.0) < 0.05)[0]
    if len(top_nodes) == 0:
        top_nodes = [nodes.shape[0] - 1]
    qoi_node = top_nodes[len(top_nodes) // 2]
    qoi_dof = 3 * qoi_node + 2  # z向位移

    # 计算各积分点处的响应
    responses = np.zeros(n_quad, dtype=np.float64)
    for i in range(n_quad):
        xi = xi_pts[i]
        mu_sample = mu_mean + mu_std * xi
        # 确保正值
        mu_sample = max(mu_sample, mu_mean * 0.1)
        u_sol, _, _, _, _ = solve_static_nonlinear_fem(
            nodes, elements, mu_sample, lam, pressure,
            use_damage=False, max_nr_iter=10, nr_tol=1e-6
        )
        responses[i] = u_sol[qoi_dof]

    # Galerkin投影计算PCE系数
    coeffs = np.zeros(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        num = 0.0
        den = 0.0
        for i in range(n_quad):
            he_k = hermite_polynomial(k, xi_pts[i])
            num += responses[i] * he_k * w_pts[i]
            den += he_k * he_k * w_pts[i]
        if abs(den) > 1e-14:
            coeffs[k] = num / den
        else:
            coeffs[k] = 0.0

    mean_est = pce_mean(coeffs)
    std_est = pce_standard_deviation(coeffs)

    print(f"  PCE均值 (顶部位移): {mean_est:.6e} m")
    print(f"  PCE标准差:          {std_est:.6e} m")
    print(f"  PCE变异系数:        {abs(std_est/mean_est)*100:.2f}%" if abs(mean_est) > 1e-12 else "  PCE变异系数: N/A")
    return coeffs, responses, xi_pts, w_pts


def run_monte_carlo_validation(nodes, elements, pressure, mu_mean, mu_std, lam, n_samples=200):
    """
    蒙特卡洛验证PCE结果。
    """
    print_section("蒙特卡洛验证")
    print(f"  MC采样数: {n_samples}")

    top_nodes = np.where(np.abs(nodes[:, 2] - 1.0) < 0.05)[0]
    if len(top_nodes) == 0:
        top_nodes = [nodes.shape[0] - 1]
    qoi_node = top_nodes[len(top_nodes) // 2]
    qoi_dof = 3 * qoi_node + 2

    def model(sample):
        mu_s = sample[0]
        u_s, _, _, _, _ = solve_static_nonlinear_fem(
            nodes, elements, mu_s, lam, pressure,
            use_damage=False, max_nr_iter=8, nr_tol=1e-6
        )
        return float(u_s[qoi_dof])

    results = monte_carlo_simulation(
        model,
        mu_params=np.array([mu_mean]),
        sigma_params=np.array([mu_std]),
        bounds=np.array([[mu_mean * 0.3, mu_mean * 2.0]]),
        n_samples=n_samples,
        seed=42
    )

    print(f"  MC均值:   {results['mean']:.6e} m")
    print(f"  MC标准差: {results['std']:.6e} m")
    print(f"  95% CI:   [{results['ci_95'][0]:.6e}, {results['ci_95'][1]:.6e}]")
    print(f"  有效样本: {results['n_valid']}/{results['n_samples']}")
    return results


def run_stress_segmentation(stress_data):
    """
    对应力场进行阈值分割，识别危险区域。
    """
    print_section("应力场分割与损伤区域识别")
    sigma_vm_arr = np.array([s["sigma_vm"] for s in stress_data])

    # Otsu自适应阈值
    theta = otsu_threshold(sigma_vm_arr, n_bins=128)
    labels = single_threshold_segmentation(sigma_vm_arr, theta)
    stats = compute_damage_zone_statistics(sigma_vm_arr, labels)

    print(f"  Otsu自适应阈值: {theta:.4e} Pa")
    for key, val in stats.items():
        print(f"  {key}: count={val['count']}, mean_stress={val['mean_stress']:.4e}, max_stress={val['max_stress']:.4e}")

    critical = identify_critical_elements(sigma_vm_arr, np.arange(len(sigma_vm_arr))[:, None], top_percentile=5.0)
    print(f"  关键单元数 (前5%): {len(critical)}")
    return labels, stats, critical


def main():
    print("\n" + "#" * 60)
    print("#  结构力学: 大变形非线性有限元分析")
    print("#  含不确定性量化与连续损伤力学")
    print("#" * 60)

    t_start = time.time()

    # =====================================================================
    # 1. 网格生成
    # =====================================================================
    print_section("1. 四面体网格生成")
    nx, ny, nz = 2, 2, 2
    nodes, elements = generate_cube_tetrahedral_mesh(nx, ny, nz)
    print(f"  初始网格: {nodes.shape[0]} 节点, {elements.shape[0]} 单元")

    # 网格方向修正
    elements = orient_tetrahedra(nodes, elements)
    quality = check_mesh_quality(nodes, elements)
    print(f"  网格质量: min_vol={quality['min_volume']:.4e}, neg_count={quality['negative_count']}")

    # =====================================================================
    # 2. 材料参数与边界条件
    # =====================================================================
    print_section("2. 材料参数与载荷定义")
    # 超弹性Neo-Hookean参数
    E_mod = 1.0e9      # Young模量 [Pa]
    nu = 0.35          # Poisson比
    mu = E_mod / (2.0 * (1.0 + nu))
    lam = E_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    print(f"  Young模量 E = {E_mod:.3e} Pa")
    print(f"  Poisson比 nu = {nu:.3f}")
    print(f"  剪切模量 mu = {mu:.3e} Pa")
    print(f"  Lamé参数 lambda = {lam:.3e} Pa")

    # 压缩压力载荷
    pressure = 5.0e7   # [Pa]
    print(f"  顶面压力 p = {pressure:.3e} Pa")

    # =====================================================================
    # 3. 确定性非线性有限元求解
    # =====================================================================
    print_section("3. 确定性大变形非线性FEM求解")
    u_det, stress_data, D_elems, n_conv, n_back = solve_static_nonlinear_fem(
        nodes, elements, mu, lam, pressure,
        use_damage=True, alpha_damage=0.05,
        max_nr_iter=15, nr_tol=1e-8
    )
    print(f"  收敛载荷步: {n_conv}")
    print(f"  回溯次数: {n_back}")

    # 顶面最大位移
    top_nodes = np.where(np.abs(nodes[:, 2] - 1.0) < 0.05)[0]
    max_disp = np.max(np.abs(u_det[3 * top_nodes + 2]))
    print(f"  顶面最大z向位移: {max_disp:.6e} m")

    # 最大von Mises应力
    sigma_vm_max = max(s["sigma_vm"] for s in stress_data)
    print(f"  最大von Mises应力: {sigma_vm_max:.4e} Pa")

    # =====================================================================
    # 4. 应力场分割
    # =====================================================================
    labels, stats, critical = run_stress_segmentation(stress_data)

    # =====================================================================
    # 5. 不确定性量化 (PCE)
    # =====================================================================
    mu_mean = mu
    mu_std = mu * 0.15  # 15%变异系数
    coeffs, responses_pce, xi_pts, w_pts = run_pce_uq(nodes, elements, pressure, mu_mean, mu_std, lam, degree=2)

    # =====================================================================
    # 6. 蒙特卡洛验证
    # =====================================================================
    mc_results = run_monte_carlo_validation(nodes, elements, pressure, mu_mean, mu_std, lam, n_samples=30)

    # =====================================================================
    # 7. 损伤演化结果
    # =====================================================================
    print_section("7. 连续损伤力学结果")
    D_max = np.max(D_elems)
    D_mean = np.mean(D_elems)
    n_damaged = np.sum(D_elems > 0.01)
    print(f"  最大单元损伤: {D_max:.4f}")
    print(f"  平均单元损伤: {D_mean:.4f}")
    print(f"  损伤单元数 (>0.01): {n_damaged}/{len(D_elems)}")

    # =====================================================================
    # 8. 等效剪切模量非线性求解演示
    # =====================================================================
    print_section("8. 损伤耦合等效模量 (Brent求根)")
    gamma_demo = 0.3
    mu_eff = solve_effective_shear_modulus(mu, gamma_demo, alpha=0.05)
    print(f"  初始模量 mu0 = {mu:.4e} Pa")
    print(f"  等效剪应变 gamma = {gamma_demo}")
    print(f"  等效模量 mu_eff = {mu_eff:.4e} Pa")
    print(f"  模量退化率 = {(1 - mu_eff/mu)*100:.2f}%")

    # =====================================================================
    # 总结
    # =====================================================================
    t_elapsed = time.time() - t_start
    print_section("分析完成")
    print(f"  总耗时: {t_elapsed:.2f} 秒")
    print(f"  所有计算模块运行正常，无报错。")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
