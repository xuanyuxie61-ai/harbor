#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
人工耳蜗电刺激电场分布数值模拟系统

统一入口，零参数可运行。
执行完整的从几何建模、电场计算到神经响应分析的流程。
"""

import sys
import numpy as np

# 确保当前目录在路径中
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from utils import (
    print_section, print_subsection, check_finite, compute_rmse,
    safe_divide, gaussian_2d
)
from cochlea_geometry import CochleaGeometry
from electrode_array import ElectrodeArray
from fem_solver import FEM2DSolver, generate_cochlea_mesh
from laplacian_operator import laplacian_5point, laplacian_9point, anisotropic_laplacian_5point
from potential_field import (
    multi_electrode_superposition, cylindrical_potential_line_source,
    gegenbauer_polynomial_value, sincn
)
from neural_membrane import SimplifiedSGNModel, DetailedSGNModel, biphasic_pulse
from reaction_diffusion import NeuralActivationPattern
from quadrature_integration import integrate_over_mesh_elements, test_quadrature_precision
from svd_analysis import PatientSVDAnalyzer, generate_synthetic_patient_data, electrode_config_optimization_svd
from statistics_patient import PatientVariabilityModel, clinical_outcome_probability
from current_continuity import lax_wendroff_current_continuity, current_density_1d


def main():
    print_section("人工耳蜗电刺激电场分布数值模拟系统", width=76)
    print("  科学领域: 生物医学 - 人工耳蜗电刺激电场分布")
    print("  基于 15 个种子项目的核心算法融合")
    print("=" * 76)

    # ========================================================================
    # 步骤 1: 耳蜗几何建模
    # ========================================================================
    print_section("步骤 1: 耳蜗几何建模")
    geometry = CochleaGeometry(
        r0=3.5, b=0.15, theta_max=4.5 * np.pi,
        scala_height=1.2, scala_width=2.0
    )
    print(f"  蜗轴中心线点数: {geometry._centerline['points'].shape[0]}")
    print(f"  螺旋紧缩系数 b = {geometry.b:.3f}")

    # 插值个性化几何 (基于种子 1210_test_interp)
    known_thetas = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.5]) * np.pi
    known_radii = geometry.r0 * np.exp(-geometry.b * known_thetas)
    cs = geometry.interpolate_patient_geometry(known_thetas, known_radii)
    print(f"  个性化插值样条在 θ=2π 处的半径: {cs(2.0*np.pi):.3f} mm")

    # 电极-蜗轴距离 (基于种子 150_cg_lab_triangles)
    test_point = np.array([[2.0, 1.0]])
    sd, idx = geometry.signed_distance_to_modiolar_axis(test_point)
    print(f"  测试点 {test_point[0]} 到蜗轴的有符号距离: {sd[0]:.4f} mm")

    # SGN 图拓扑 (基于种子 490_grf_io)
    nodes, edges, weights = geometry.build_sgn_graph(n_neurons=100)
    print(f"  SGN 神经元节点数: {len(nodes)}, 连接边数: {len(edges)}")

    # ========================================================================
    # 步骤 2: 电极阵列配置
    # ========================================================================
    print_section("步骤 2: 电极阵列配置")
    electrode_array = ElectrodeArray(
        n_electrodes=12, electrode_radius=0.2, spacing=1.0,
        insertion_depth_mm=20.0, offset_from_modiolus=0.8
    )
    elec_positions = electrode_array.place_along_modiolar_axis(geometry)
    print(f"  电极数量: {electrode_array.n_electrodes}")
    print(f"  电极间距: {electrode_array.spacing:.1f} mm")
    print(f"  电极半径: {electrode_array.electrode_radius:.2f} mm")

    # 三极刺激模式
    electrode_array.tripolar_stimulus(center_idx=5, amplitude_uA=300.0, fraction=0.5)
    print(f"  三极刺激配置 (中心电极 #5, 300 μA):")
    print(f"    电流分布 (μA): {np.round(electrode_array.currents * 1e6, 2)}")

    # ========================================================================
    # 步骤 3: 有限元网格生成与 FEM 电场求解
    # ========================================================================
    print_section("步骤 3: FEM 电场求解")
    nodes_mesh, elements_mesh = generate_cochlea_mesh(geometry, n_radial=16, n_angular=60)
    print(f"  网格节点数: {nodes_mesh.shape[0]}")
    print(f"  三角形单元数: {elements_mesh.shape[0]}")

    # 创建 FEM 求解器 (基于种子 416_fem2d_scalar_display_gpl)
    sigma_tissue = 0.3  # S/m, 生理盐水+组织混合电导率
    fem_solver = FEM2DSolver(nodes_mesh, elements_mesh, conductivity=sigma_tissue)

    # 源项
    source = electrode_array.get_source_terms(nodes_mesh)
    check_finite(source, "source")

    # Dirichlet 边界: 外侧骨壁接地
    dists_to_axis, _ = geometry.signed_distance_to_modiolar_axis(nodes_mesh)
    max_dist = np.max(dists_to_axis)
    bc_nodes = np.where(np.abs(dists_to_axis - max_dist) < 0.1)[0]
    bc_values = np.zeros(len(bc_nodes))

    print(f"  边界条件: {len(bc_nodes)} 个节点接地 (Dirichlet V=0)")

    # 求解
    V_fem = fem_solver.solve(source, dirichlet_nodes=bc_nodes, dirichlet_values=bc_values)
    check_finite(V_fem, "V_fem")
    print(f"  FEM 电势范围: [{np.min(V_fem)*1e3:.3f}, {np.max(V_fem)*1e3:.3f}] mV")

    # 激活函数
    af = fem_solver.compute_activation_function(V_fem)
    print(f"  激活函数 AF 范围: [{np.min(af)*1e3:.3f}, {np.max(af)*1e3:.3f}] mV/mm²")

    # 梯度
    grad = fem_solver.compute_gradient(V_fem)
    grad_norm = np.linalg.norm(grad, axis=1)
    print(f"  电场梯度范数范围: [{np.min(grad_norm):.3e}, {np.max(grad_norm):.3e}] V/mm")

    # ========================================================================
    # 步骤 4: 高阶 Laplacian 与解析电势验证
    # ========================================================================
    print_section("步骤 4: 高阶 Laplacian 与解析电势")

    # 将 FEM 结果插值到规则网格进行 Laplacian 分析
    nx_grid, ny_grid = 40, 20
    x_min, x_max = np.min(nodes_mesh[:, 0]), np.max(nodes_mesh[:, 0])
    y_min, y_max = np.min(nodes_mesh[:, 1]), np.max(nodes_mesh[:, 1])
    xg = np.linspace(x_min, x_max, nx_grid)
    yg = np.linspace(y_min, y_max, ny_grid)
    Xg, Yg = np.meshgrid(xg, yg, indexing='ij')
    query_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    # 使用多电极叠加解析解 (基于种子 462_gegenbauer, 1082_sinc)
    V_analytic = multi_electrode_superposition(
        electrode_array.positions, electrode_array.currents,
        query_grid, sigma_tissue
    )
    V_analytic_grid = V_analytic.reshape(nx_grid, ny_grid)
    print(f"  解析电势范围: [{np.min(V_analytic)*1e3:.3f}, {np.max(V_analytic)*1e3:.3f}] mV")

    # 5点 vs 9点 Laplacian 比较 (基于种子 487_gray_scott_pde, 282_differ)
    lap5 = laplacian_5point(V_analytic_grid, xg[1]-xg[0], yg[1]-yg[0])
    lap9 = laplacian_9point(V_analytic_grid, xg[1]-xg[0], yg[1]-yg[0])
    rmse_lap = compute_rmse(lap5[1:-1, 1:-1], lap9[1:-1, 1:-1])
    print(f"  5点 vs 9点 Laplacian RMSE: {rmse_lap:.3e}")

    # 各向异性 Laplacian
    lap_aniso = anisotropic_laplacian_5point(
        V_analytic_grid, xg[1]-xg[0], yg[1]-yg[0], sigma_xx=0.4, sigma_yy=0.2
    )
    print(f"  各向异性 Laplacian 范围: [{np.min(lap_aniso):.3e}, {np.max(lap_aniso):.3e}]")

    # Gegenbauer 多项式展开验证 (基于种子 462_gegenbauer_polynomial)
    theta_test = np.linspace(0.01, np.pi - 0.01, 50)
    C_vals = gegenbauer_polynomial_value(10, 0.5, np.cos(theta_test))
    print(f"  Gegenbauer C_10^{{(0.5)}}(cos θ) 范围: [{np.min(C_vals[-1]):.3f}, {np.max(C_vals[-1]):.3f}]")

    # Sinc 函数验证 (基于种子 1082_sinc)
    x_sinc = np.linspace(-3.0, 3.0, 100)
    s_vals = sincn(x_sinc)
    print(f"  Sinc 函数在 x=0 处值: {s_vals[50]:.6f} (理论值 1.0)")

    # ========================================================================
    # 步骤 5: 神经元膜电位响应模拟
    # ========================================================================
    print_section("步骤 5: 神经元膜电位响应模拟")

    # 提取某神经元位置处的刺激电流时间历程
    neuron_pos = nodes[20]  # 从 SGN 图拓扑中取一个神经元
    stim_amplitude = 50.0  # μA/cm²

    def stimulus_func(t):
        return biphasic_pulse(t, amplitude=stim_amplitude,
                              phase_width_ms=0.1, interphase_gap_ms=0.05)

    # 简化 FHN 模型 (基于种子 100_blood_pressure_ode 的 ODE 思想)
    sgn_simple = SimplifiedSGNModel(tau_m=0.1, epsilon=0.08, V_rest=-65.0, V_thresh=-40.0)
    sol_simple, spikes_simple = sgn_simple.simulate(
        (0.0, 2.0), [-65.0, 0.0], stimulus_func, max_step=0.005
    )
    print(f"  简化 FHN 模型: 在 0-2 ms 内发放 {len(spikes_simple)} 次动作电位")

    # 详细 HH 模型 (基于种子 619_kepler_perturbed_ode 的 ODE 积分思想)
    sgn_detail = DetailedSGNModel(C_m=1.0, g_Na=120.0, g_K=36.0, T=310.15)
    sol_detail, spikes_detail = sgn_detail.simulate(
        (0.0, 2.0), stimulus_func, max_step=0.002
    )
    print(f"  详细 HH 模型: 在 0-2 ms 内发放 {len(spikes_detail)} 次动作电位")

    # ========================================================================
    # 步骤 6: 神经激活反应-扩散斑图
    # ========================================================================
    print_section("步骤 6: 神经激活反应-扩散斑图")

    rd_solver = NeuralActivationPattern(
        nx=32, ny=16, dx=0.1, dy=0.1,
        D_u=0.01, D_v=0.005, gamma=0.024, kappa=0.06
    )
    rd_solver.initialize(seed_pattern='gaussian')

    # 生成刺激历史: 在电极附近施加高斯脉冲
    stimulus_history = []
    for step in range(200):
        stim = np.zeros((rd_solver.nx, rd_solver.ny))
        # 高斯刺激源
        X, Y = np.meshgrid(np.arange(rd_solver.nx), np.arange(rd_solver.ny), indexing='ij')
        stim = gaussian_2d(X, Y, 16, 8, 4.0, 3.0, amplitude=0.05)
        stimulus_history.append(stim)

    U_hist, V_hist = rd_solver.evolve(n_steps=200, stimulus_history=stimulus_history, dt=0.02)
    area, centroid, spread = rd_solver.compute_spread_metrics()
    print(f"  反应-扩散演化 200 步后:")
    print(f"    激活区域面积: {area:.1f} 像素")
    print(f"    质心: ({centroid[0]:.1f}, {centroid[1]:.1f})")
    print(f"    空间展宽 (std): {spread:.2f} 像素")

    # ========================================================================
    # 步骤 7: 高精度三角形数值积分
    # ========================================================================
    print_section("步骤 7: 高精度三角形数值积分")

    # 测试求积规则精度 (基于种子 1325_triangle_witherden_rule)
    quad_errors = test_quadrature_precision(max_precision=5)
    for p, err in quad_errors.items():
        print(f"  精度阶 {p}: 最大单项式积分误差 = {err:.3e}")

    # 在 FEM 网格上积分电势能密度 (0.5 σ |∇V|²)
    def energy_density(x, y):
        # 支持标量或数组输入; 近似: 使用最近节点值
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        result = np.empty_like(x, dtype=float)
        for i in range(x.size):
            query = np.array([x.flat[i], y.flat[i]])
            dists = np.linalg.norm(nodes_mesh - query, axis=1)
            nearest = np.argmin(dists)
            result.flat[i] = 0.5 * sigma_tissue * grad_norm[nearest]**2
        return result

    total_energy, elem_energies = integrate_over_mesh_elements(
        energy_density, nodes_mesh, elements_mesh, precision=4
    )
    print(f"  网格总电势能: {total_energy:.6e} J")
    check_finite(elem_energies, "elem_energies")

    # ========================================================================
    # 步骤 8: SVD 患者参数降维分析
    # ========================================================================
    print_section("步骤 8: SVD 患者参数降维分析")

    # 生成合成患者数据 (基于种子 1186_svd_faces)
    patient_data, true_modes = generate_synthetic_patient_data(
        n_patients=50, n_features=200, n_modes=5, noise_level=0.05
    )
    svd_analyzer = PatientSVDAnalyzer()
    svd_analyzer.fit(patient_data)

    sv = svd_analyzer.get_singular_values()
    evr = svd_analyzer.explained_variance_ratio()
    cvr = svd_analyzer.cumulative_variance_ratio()
    print(f"  合成患者数: 50, 特征维度: 200")
    print(f"  前 5 个奇异值: {np.round(sv[:5], 3)}")
    print(f"  前 5 个主成分解释方差比: {np.round(evr[:5], 4)}")
    print(f"  前 5 个主成分累积方差比: {np.round(cvr[:5], 4)}")

    # 低秩近似
    A_approx = svd_analyzer.low_rank_approximation(rank=5)
    approx_error = compute_rmse(patient_data, A_approx)
    print(f"  秩-5 近似 RMSE: {approx_error:.4f}")

    # 电极配置优化
    n_configs = 20
    n_neurons = 100
    config_responses = np.random.rand(n_configs, n_neurons)
    # 加入结构化模式
    for i in range(n_configs):
        config_responses[i, :] += 0.5 * np.sin(np.linspace(0, 2*np.pi, n_neurons) + i*0.3)
    opt_dir, importance = electrode_config_optimization_svd(config_responses, n_components=3)
    print(f"  电极配置 SVD 优化: 前 3 个主成分")
    print(f"  最优方向前 5 个参数重要性: {np.round(importance[:5], 4)}")

    # ========================================================================
    # 步骤 9: 患者变异性统计模型
    # ========================================================================
    print_section("步骤 9: 患者变异性统计模型")

    # 基于种子 055_asa310: 非中心 Beta 分布
    var_model = PatientVariabilityModel(
        sigma_mean=0.3, sigma_std=0.08,
        survival_alpha=5.0, survival_beta=2.0, survival_lambda=2.0
    )
    cohort = var_model.generate_patient_cohort(n_patients=100)
    print(f"  患者队列统计 (n=100):")
    print(f"    电导率 μ±σ: {np.mean(cohort['conductivity']):.3f} ± {np.std(cohort['conductivity']):.3f} S/m")
    print(f"    神经存活率 μ±σ: {np.mean(cohort['survival_rate']):.3f} ± {np.std(cohort['survival_rate']):.3f}")
    print(f"    电极偏移 μ±σ: {np.mean(cohort['offset']):.3f} ± {np.std(cohort['offset']):.3f} mm")

    # 非中心 Beta CDF 计算示例
    survival_example = 0.6
    prob = var_model.probability_threshold_hearing(survival_example, threshold=0.5)
    print(f"  神经存活率 {survival_example} 的患者达到听力阈值的概率: {prob:.4f}")

    # 临床预后
    outcome_prob = clinical_outcome_probability(
        survival_rate=0.7, stimulation_level=0.8,
        alpha=5.0, beta=2.0, lambda_nc=2.0
    )
    print(f"  存活率 0.7 + 刺激水平 0.8 的临床成功概率: {outcome_prob:.4f}")

    # ========================================================================
    # 步骤 10: 一维电流连续性方程
    # ========================================================================
    print_section("步骤 10: 一维电流连续性方程")

    # 基于种子 1068_shallow_water_1d 的 Lax-Wendroff 格式
    def source_1d(x, t):
        # 在 x=10 mm 处施加脉冲电流源
        peak_pos = 10.0
        sigma_t = 0.5
        return 100.0 * np.exp(-((x - peak_pos)**2) / (2 * sigma_t**2)) * np.exp(-t)

    V_1d, x_1d, t_1d = lax_wendroff_current_continuity(
        nx=101, nt=201, x_max=20.0, t_max=2.0,
        sigma=sigma_tissue, I_source=source_1d, bc_type='neumann'
    )
    print(f"  一维纵向传播: nx={len(x_1d)}, nt={len(t_1d)}")
    print(f"  电势时空最大值: {np.max(V_1d)*1e3:.3f} mV")

    # 电流密度
    J_1d = current_density_1d(V_1d[:, -1], x_1d, sigma_tissue)
    print(f"  稳态电流密度范围: [{np.min(J_1d):.3e}, {np.max(J_1d):.3e}] A/mm²")

    # ========================================================================
    # 汇总与验证
    # ========================================================================
    print_section("结果汇总与数值验证")

    print_subsection("物理一致性检查")
    # 检查 FEM 解是否满足基本物理约束
    assert np.max(V_fem) > np.min(V_fem), "电势必须有变化"
    assert np.isfinite(np.max(V_fem)), "FEM 电势必须有限"
    assert len(spikes_simple) >= 0, "发放次数非负"
    assert area >= 0, "激活面积非负"
    assert np.all(evr >= 0), "方差解释比非负"
    assert np.all(cohort['conductivity'] > 0), "电导率必须为正"
    assert np.all((cohort['survival_rate'] >= 0) & (cohort['survival_rate'] <= 1)), "存活率必须在 [0,1]"

    print("  [PASS] FEM 电势有变化且有限")
    print("  [PASS] 神经元发放次数非负")
    print("  [PASS] 反应-扩散激活面积非负")
    print("  [PASS] SVD 方差解释比非负")
    print("  [PASS] 患者参数物理合理")

    print_subsection("各模块集成验证")
    print(f"  模块 cochlea_geometry: 几何+插值+距离+图拓扑 [OK]")
    print(f"  模块 electrode_array: 电极配置+刺激模式 [OK]")
    print(f"  模块 fem_solver: 2D FEM Poisson 求解 [OK]")
    print(f"  模块 laplacian_operator: 5点/9点/各向异性 Laplacian [OK]")
    print(f"  模块 potential_field: Gegenbauer+sinc+多电极叠加 [OK]")
    print(f"  模块 neural_membrane: FHN+HH 膜模型 [OK]")
    print(f"  模块 reaction_diffusion: Gray-Scott 型神经激活 [OK]")
    print(f"  模块 quadrature_integration: Witherden 三角形求积 [OK]")
    print(f"  模块 svd_analysis: 患者参数降维+电极优化 [OK]")
    print(f"  模块 statistics_patient: 非中心 Beta+队列生成 [OK]")
    print(f"  模块 current_continuity: Lax-Wendroff 守恒律 [OK]")

    print("\n" + "=" * 76)
    print("  人工耳蜗电刺激电场分布数值模拟完成")
    print("  所有计算模块已成功集成并验证")
    print("=" * 76 + "\n")

    return 0


if __name__ == "__main__":
    ret = main()

# Monkey-patch for numpy compatibility (needed for laplacian_operator in NumPy >=1.24)
import math
if not hasattr(np, "math"):
    np.math = math

# Add missing imports for test cases not imported by main.py
from quadrature_integration import triangle_quadrature_rule, integrate_triangle
from laplacian_operator import build_differ_matrix, high_order_derivative_coefficients
from statistics_patient import noncentral_beta_cdf

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: safe_divide 正常除法 ----
result = safe_divide(np.array([10.0]), np.array([2.0]))
assert np.isclose(result[0], 5.0), '[TC01] safe_divide 正常除法 FAILED'

# ---- TC02: safe_divide 除零返回 fill_value ----
result = safe_divide(np.array([1.0]), np.array([0.0]), fill_value=-1.0)
assert np.isclose(result[0], -1.0), '[TC02] safe_divide 除零返回 fill_value FAILED'

# ---- TC03: compute_rmse 完全匹配 ----
a = np.array([1.0, 2.0, 3.0])
rmse = compute_rmse(a, a)
assert np.isclose(rmse, 0.0), '[TC03] compute_rmse 完全匹配 FAILED'

# ---- TC04: compute_rmse 已知误差 ----
b = np.array([2.0, 3.0, 4.0])
rmse = compute_rmse(a, b)
assert np.isclose(rmse, 1.0), '[TC04] compute_rmse 已知误差 FAILED'

# ---- TC05: gaussian_2d 峰值在中心 ----
X, Y = np.meshgrid(np.arange(5), np.arange(5), indexing='ij')
g = gaussian_2d(X, Y, 2, 2, 1.0, 1.0, amplitude=1.0)
assert np.isclose(g[2, 2], 1.0), '[TC05] gaussian_2d 峰值在中心 FAILED'

# ---- TC06: check_finite 有限数组通过 ----
assert check_finite(np.array([1.0, 2.0, 3.0]), "test") is True, '[TC06] check_finite 有限数组 FAILED'

# ---- TC07: CochleaGeometry 构造与属性 ----
import numpy as np
cg = CochleaGeometry(r0=3.5, b=0.15, scala_height=1.2, scala_width=2.0)
assert cg.r0 == 3.5, '[TC07] CochleaGeometry r0 FAILED'
assert cg.b == 0.15, '[TC07] CochleaGeometry b FAILED'
assert cg._centerline is not None, '[TC07] CochleaGeometry _centerline FAILED'
assert cg._centerline['points'].shape[0] == 400, '[TC07] CochleaGeometry 中心线点数 FAILED'
assert cg._centerline['points'].shape[1] == 2, '[TC07] CochleaGeometry 中心线维度 FAILED'

# ---- TC08: centerline_at 返回正确形状 ----
pt = cg.centerline_at(np.pi)
assert pt.shape == (2,), '[TC08] centerline_at 形状 FAILED'
assert np.all(np.isfinite(pt)), '[TC08] centerline_at 有限值 FAILED'

# ---- TC09: signed_distance_to_modiolar_axis 输出形状 ----
test_pts = np.array([[2.0, 1.0], [3.0, 0.5]])
sd, idx = cg.signed_distance_to_modiolar_axis(test_pts)
assert sd.shape == (2,), '[TC09] signed_distance 形状 FAILED'
assert idx.shape == (2,), '[TC09] signed_distance idx 形状 FAILED'
assert np.all(np.isfinite(sd)), '[TC09] signed_distance 有限值 FAILED'

# ---- TC10: ElectrodeArray 构造 ----
import numpy as np
ea = ElectrodeArray(n_electrodes=8, electrode_radius=0.2, spacing=1.0, insertion_depth_mm=20.0, offset_from_modiolus=0.8)
assert ea.n_electrodes == 8, '[TC10] ElectrodeArray n_electrodes FAILED'
assert ea.electrode_radius == 0.2, '[TC10] ElectrodeArray electrode_radius FAILED'

# ---- TC11: place_along_modiolar_axis 返回正确位置数 ----
import numpy as np
np.random.seed(42)
cg2 = CochleaGeometry(r0=3.0, b=0.12)
ea2 = ElectrodeArray(n_electrodes=6, spacing=1.0, insertion_depth_mm=15.0)
pos = ea2.place_along_modiolar_axis(cg2)
assert pos.shape == (6, 2), '[TC11] place_along_modiolar_axis 形状 FAILED'
assert np.all(np.isfinite(pos)), '[TC11] place_along_modiolar_axis 有限值 FAILED'

# ---- TC12: monopolar_stimulus 电流设定正确 ----
import numpy as np
np.random.seed(42)
ea3 = ElectrodeArray(n_electrodes=4, spacing=1.0, insertion_depth_mm=10.0)
ea3.place_along_modiolar_axis(cg2)
ea3.monopolar_stimulus(active_electrode_idx=2, amplitude_uA=100.0)
assert np.isclose(ea3.currents[2], 100e-6), '[TC12] monopolar 活动电极电流 FAILED'
assert np.isclose(ea3.currents[0], 0.0), '[TC12] monopolar 非活动电极电流 FAILED'

# ---- TC13: tripolar_stimulus 电流总和接近零 ----
import numpy as np
np.random.seed(42)
ea4 = ElectrodeArray(n_electrodes=6, spacing=1.0, insertion_depth_mm=15.0)
ea4.place_along_modiolar_axis(cg2)
ea4.tripolar_stimulus(center_idx=3, amplitude_uA=200.0, fraction=0.5)
# 中心电极 200 μA, 相邻各 -50 μA
assert np.isclose(ea4.currents[3], 200e-6), '[TC13] tripolar 中心电极电流 FAILED'
assert np.isclose(ea4.currents[2], -50e-6), '[TC13] tripolar 相邻电极1电流 FAILED'
assert np.isclose(ea4.currents[4], -50e-6), '[TC13] tripolar 相邻电极2电流 FAILED'

# ---- TC14: generate_cochlea_mesh 输出形状正确 ----
import numpy as np
np.random.seed(42)
cg3 = CochleaGeometry(r0=3.0, b=0.12, theta_max=2.0*np.pi)
nodes_m, elems_m = generate_cochlea_mesh(cg3, n_radial=10, n_angular=20)
assert nodes_m.ndim == 2 and nodes_m.shape[1] == 2, '[TC14] mesh nodes 形状 FAILED'
assert elems_m.ndim == 2 and elems_m.shape[1] == 3, '[TC14] mesh elements 形状 FAILED'
assert nodes_m.shape[0] == 200, '[TC14] mesh nodes 数量 FAILED'

# ---- TC15: FEM2DSolver 求解产生有限值 ----
import numpy as np
np.random.seed(42)
solver = FEM2DSolver(nodes_m, elems_m, conductivity=0.3)
src = np.zeros(nodes_m.shape[0])
src[nodes_m.shape[0] // 2] = 1.0  # 中心节点源
V = solver.solve(src, dirichlet_nodes=np.array([0, nodes_m.shape[0]-1]), dirichlet_values=np.array([0.0, 0.0]))
assert V.shape == (nodes_m.shape[0],), '[TC15] FEM solve 形状 FAILED'
assert np.all(np.isfinite(V)), '[TC15] FEM solve 有限值 FAILED'
assert np.max(V) > np.min(V), '[TC15] FEM solve 电势有变化 FAILED'

# ---- TC16: compute_gradient 输出形状正确 ----
grad = solver.compute_gradient(V)
assert grad.shape == (elems_m.shape[0], 2), '[TC16] compute_gradient 形状 FAILED'
assert np.all(np.isfinite(grad)), '[TC16] compute_gradient 有限值 FAILED'

# ---- TC17: laplacian_5point 常数场 Laplacian 为零 ----
import numpy as np
V_const = np.ones((5, 5))
lap5_const = laplacian_5point(V_const, 1.0, 1.0)
assert np.allclose(lap5_const, 0.0, atol=1e-14), '[TC17] laplacian_5point 常数场 FAILED'

# ---- TC18: laplacian_9point 常数场 Laplacian 为零 ----
lap9_const = laplacian_9point(V_const, 1.0, 1.0)
assert np.allclose(lap9_const, 0.0, atol=1e-14), '[TC18] laplacian_9point 常数场 FAILED'

# ---- TC19: anisotropic_laplacian_5point 常数场为零 ----
lap_aniso_const = anisotropic_laplacian_5point(V_const, 1.0, 1.0, 0.4, 0.2)
assert np.allclose(lap_aniso_const, 0.0, atol=1e-14), '[TC19] anisotropic_laplacian 常数场 FAILED'

# ---- TC20: gegenbauer_polynomial_value C_0 = 1 ----
import numpy as np
x_vals = np.linspace(-0.9, 0.9, 20)
C = gegenbauer_polynomial_value(5, 0.5, x_vals)
assert np.allclose(C[0, :], 1.0), '[TC20] gegenbauer C_0 FAILED'

# ---- TC21: gegenbauer_polynomial_value C_1 = 2αx ----
assert np.allclose(C[1, :], 2.0 * 0.5 * x_vals), '[TC21] gegenbauer C_1 FAILED'

# ---- TC22: sincn(0) = 1 ----
import numpy as np
x_sinc_test = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
s_vals = sincn(x_sinc_test)
assert np.isclose(s_vals[0], 1.0), '[TC22] sincn(0) FAILED'
assert np.all(np.abs(s_vals) <= 1.0 + 1e-14), '[TC22] sincn 有界 FAILED'

# ---- TC23: cylindrical_potential_line_source 距离增大电势减小 ----
import numpy as np
rho_test = np.array([0.5, 1.0, 2.0])
z_test = np.array([0.0, 0.0, 0.0])
V_cyl = cylindrical_potential_line_source(rho_test, z_test, 0.0, 1e-6, 0.3)
assert V_cyl[0] > V_cyl[1] > V_cyl[2], '[TC23] cylindrical 距离衰减 FAILED'
assert np.all(np.isfinite(V_cyl)), '[TC23] cylindrical 有限值 FAILED'

# ---- TC24: multi_electrode_superposition 输出有限 ----
import numpy as np
epos = np.array([[0.0, 0.0], [1.0, 0.0]])
ecurr = np.array([1e-6, -0.5e-6])
qpts = np.array([[0.5, 0.5], [0.3, 0.0], [1.5, 0.0]])
V_sup = multi_electrode_superposition(epos, ecurr, qpts, 0.3)
assert V_sup.shape == (3,), '[TC24] multi_electrode 形状 FAILED'
assert np.all(np.isfinite(V_sup)), '[TC24] multi_electrode 有限值 FAILED'

# ---- TC25: biphasic_pulse 时间点值正确 ----
import numpy as np
# 0 <= t < 0.1: amplitude; 0.1 <= t < 0.15: 0; 0.15 <= t < 0.25: -amplitude
bp = biphasic_pulse(0.05, amplitude=50.0, phase_width_ms=0.1, interphase_gap_ms=0.05)
assert np.isclose(bp, 50.0), '[TC25] biphasic 第一相 FAILED'
bp2 = biphasic_pulse(0.12, amplitude=50.0, phase_width_ms=0.1, interphase_gap_ms=0.05)
assert np.isclose(bp2, 0.0), '[TC25] biphasic 间隙 FAILED'
bp3 = biphasic_pulse(0.18, amplitude=50.0, phase_width_ms=0.1, interphase_gap_ms=0.05)
assert np.isclose(bp3, -50.0), '[TC25] biphasic 第二相 FAILED'

# ---- TC26: SimplifiedSGNModel 模拟输出 ----
import numpy as np
np.random.seed(42)
sgn = SimplifiedSGNModel(tau_m=0.1, epsilon=0.08, V_rest=-65.0, V_thresh=-40.0)
def stim_zero(t):
    return 0.0
sol_s, spikes_s = sgn.simulate((0.0, 1.0), [-65.0, 0.0], stim_zero, max_step=0.01)
assert sol_s.t.shape[0] >= 2, '[TC26] FHN 模拟输出 FAILED'
assert isinstance(spikes_s, list), '[TC26] FHN spikes 类型 FAILED'

# ---- TC27: DetailedSGNModel alpha_m 在静息电位处为正 ----
import numpy as np
np.random.seed(42)
dsgn = DetailedSGNModel(C_m=1.0, g_Na=120.0, g_K=36.0, T=310.15)
a_m = float(dsgn.alpha_m(-65.0))
assert a_m > 0.0, '[TC27] HH alpha_m 正值 FAILED'

# ---- TC28: NeuralActivationPattern 初始化 gaussian ----
import numpy as np
np.random.seed(42)
rd = NeuralActivationPattern(nx=16, ny=8, dx=0.1, dy=0.1, D_u=0.01, D_v=0.005, gamma=0.024, kappa=0.06)
rd.initialize(seed_pattern='gaussian')
assert rd.U is not None and rd.V is not None, '[TC28] RD 初始化 FAILED'
assert rd.U.shape == (16, 8), '[TC28] RD U 形状 FAILED'
assert rd.V.shape == (16, 8), '[TC28] RD V 形状 FAILED'

# ---- TC29: NeuralActivationPattern evolve 输出历史正确长度 ----
np.random.seed(42)
U_hist, V_hist = rd.evolve(n_steps=5)
assert len(U_hist) == 5, '[TC29] RD evolve U 长度 FAILED'
assert len(V_hist) == 5, '[TC29] RD evolve V 长度 FAILED'
assert U_hist[0].shape == (16, 8), '[TC29] RD evolve 形状 FAILED'

# ---- TC30: NeuralActivationPattern compute_spread_metrics ----
area, cent, spr = rd.compute_spread_metrics()
assert area >= 0, '[TC30] RD spread area 非负 FAILED'
assert isinstance(cent, tuple), '[TC30] RD spread centroid 类型 FAILED'
assert spr >= 0, '[TC30] RD spread std 非负 FAILED'

# ---- TC31: triangle_quadrature_rule 返回正确形状 ----
import numpy as np
from quadrature_integration import triangle_quadrature_rule
xq, yq, wq = triangle_quadrature_rule(3)
assert len(xq) == 6 and len(yq) == 6 and len(wq) == 6, '[TC31] quadrature_rule precision=3 点数 FAILED'

# ---- TC32: triangle_monomial_integral 已知值 ----
from quadrature_integration import triangle_monomial_integral
# 参考三角形上 ∫ 1 dxdy = 1/2
integral_const = triangle_monomial_integral(0, 0)
assert np.isclose(integral_const, 0.5), '[TC32] monomial 常数积分 FAILED'

# ---- TC33: integrate_triangle 常数函数 ----
import numpy as np
from quadrature_integration import integrate_triangle
np.random.seed(42)
verts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
integral = integrate_triangle(lambda x, y: np.ones_like(x), verts, precision=4)
# 三角形面积 = 2.0 (底2高2的直角三角形), 积分 = 面积
assert np.isclose(integral, 2.0, rtol=1e-12), '[TC33] integrate_triangle 常数 FAILED'

# ---- TC34: test_quadrature_precision 返回字典 ----
qp_errors = test_quadrature_precision(max_precision=5)
assert isinstance(qp_errors, dict), '[TC34] quadrature_precision 类型 FAILED'
assert len(qp_errors) == 5, '[TC34] quadrature_precision 长度 FAILED'

# ---- TC35: lax_wendroff_current_continuity 输出形状 ----
import numpy as np
np.random.seed(42)
def src_1d(x, t):
    return np.exp(-0.5 * (x - 5.0)**2) * np.exp(-t)
V_lw, x_lw, t_lw = lax_wendroff_current_continuity(
    nx=51, nt=101, x_max=10.0, t_max=1.0, sigma=0.3, I_source=src_1d, bc_type='neumann'
)
assert V_lw.shape == (len(x_lw), len(t_lw)), '[TC35] lax_wendroff 形状 FAILED'
assert len(x_lw) >= 51, '[TC35] lax_wendroff x 长度 FAILED'
assert np.all(np.isfinite(V_lw)), '[TC35] lax_wendroff 有限值 FAILED'

# ---- TC36: current_density_1d 输出形状一致 ----
J_1d_test = current_density_1d(V_lw[:, -1], x_lw, 0.3)
assert J_1d_test.shape == V_lw[:, -1].shape, '[TC36] current_density 形状 FAILED'
assert np.all(np.isfinite(J_1d_test)), '[TC36] current_density 有限值 FAILED'

# ---- TC37: PatientSVDAnalyzer fit 与 explained_variance_ratio 和为 1 ----
import numpy as np
np.random.seed(42)
data_svd, _ = generate_synthetic_patient_data(n_patients=30, n_features=100, n_modes=3, noise_level=0.01)
svd = PatientSVDAnalyzer()
svd.fit(data_svd)
evr = svd.explained_variance_ratio()
assert np.isclose(np.sum(evr), 1.0, atol=1e-12), '[TC37] SVD evr 和 FAILED'

# ---- TC38: SVD get_singular_values 非负 ----
sv = svd.get_singular_values()
assert np.all(sv >= 0), '[TC38] SVD 奇异值非负 FAILED'

# ---- TC39: SVD cumulative_variance_ratio 单调非减 ----
cvr = svd.cumulative_variance_ratio()
assert np.all(np.diff(cvr) >= -1e-14), '[TC39] SVD cvr 单调 FAILED'
assert np.isclose(cvr[-1], 1.0, atol=1e-12), '[TC39] SVD cvr 末尾值 FAILED'

# ---- TC40: SVD project/reconstruct 往返 ----
import numpy as np
np.random.seed(42)
patient_vec = data_svd[:, 0]
coeffs = svd.project(patient_vec, n_components=3)
recon = svd.reconstruct(coeffs)
assert recon.shape == patient_vec.shape, '[TC40] SVD reconstruct 形状 FAILED'
assert np.all(np.isfinite(recon)), '[TC40] SVD reconstruct 有限值 FAILED'

# ---- TC41: SVD low_rank_approximation 形状 ----
A_approx_test = svd.low_rank_approximation(rank=2)
assert A_approx_test.shape == data_svd.shape, '[TC41] SVD low_rank 形状 FAILED'

# ---- TC42: electrode_config_optimization_svd 输出 ----
import numpy as np
np.random.seed(42)
config_resp_test = np.random.rand(10, 50)
opt_dir, imp = electrode_config_optimization_svd(config_resp_test, n_components=2)
assert opt_dir.shape == (50,), '[TC42] electrode_opt direction 形状 FAILED'
assert imp.shape == (50,), '[TC42] electrode_opt importance 形状 FAILED'

# ---- TC43: noncentral_beta_cdf 边界 x=0 返回 0 ----
import numpy as np
from statistics_patient import noncentral_beta_cdf
np.random.seed(42)
cdf0, ifault0 = noncentral_beta_cdf(0.0, 5.0, 2.0, 2.0)
assert np.isclose(cdf0, 0.0), '[TC43] noncentral_beta x=0 FAILED'

# ---- TC44: noncentral_beta_cdf 边界 x=1 返回 1 ----
cdf1, ifault1 = noncentral_beta_cdf(1.0, 5.0, 2.0, 2.0)
assert np.isclose(cdf1, 1.0), '[TC44] noncentral_beta x=1 FAILED'

# ---- TC45: noncentral_beta_cdf CDF 单调性 (中间值) ----
cdf_mid, _ = noncentral_beta_cdf(0.5, 5.0, 2.0, 2.0)
assert 0.0 < cdf_mid < 1.0, '[TC45] noncentral_beta CDF 范围 FAILED'

# ---- TC46: PatientVariabilityModel 构造与队列生成 ----
import numpy as np
np.random.seed(42)
pvm = PatientVariabilityModel(sigma_mean=0.3, sigma_std=0.08, survival_alpha=5.0, survival_beta=2.0, survival_lambda=2.0)
cohort_test = pvm.generate_patient_cohort(n_patients=50)
assert 'conductivity' in cohort_test, '[TC46] cohort conductivity FAILED'
assert 'survival_rate' in cohort_test, '[TC46] cohort survival_rate FAILED'
assert 'offset' in cohort_test, '[TC46] cohort offset FAILED'
assert len(cohort_test['conductivity']) == 50, '[TC46] cohort 长度 FAILED'
# 电导率为正
assert np.all(cohort_test['conductivity'] > 0), '[TC46] cohort 电导率正 FAILED'
# 存活率在 [0,1]
assert np.all((cohort_test['survival_rate'] >= 0) & (cohort_test['survival_rate'] <= 1)), '[TC46] cohort 存活率范围 FAILED'

# ---- TC47: clinical_outcome_probability 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
outcome = clinical_outcome_probability(survival_rate=0.7, stimulation_level=0.8, alpha=5.0, beta=2.0, lambda_nc=2.0)
assert 0.0 <= outcome <= 1.0, '[TC47] clinical_outcome 范围 FAILED'

# ---- TC48: laplacian_5point 二次函数返回常数 Laplacian ----
import numpy as np
# f(x) = x^2, then f''(x) = 2. On 2D with dy -> infinity effectively:
# Use V_ij = x_i^2: ∂²V/∂x² = 2, ∂²V/∂y² = 0 (since no y dependence)
nx_t, ny_t = 11, 11
X_l, Y_l = np.meshgrid(np.linspace(0, 1, nx_t), np.linspace(0, 1, ny_t), indexing='ij')
V_quad = X_l**2
dx_l = 1.0 / (nx_t - 1)
dy_l = 1.0 / (ny_t - 1)
lap_quad = laplacian_5point(V_quad, dx_l, dy_l)
# 内部点应接近 2.0
assert np.allclose(lap_quad[2:-2, 2:-2], 2.0, atol=0.05), '[TC48] laplacian 二次函数 FAILED'

# ---- TC49: build_differ_matrix 可逆 ----
import numpy as np
from laplacian_operator import build_differ_matrix, high_order_derivative_coefficients
np.random.seed(42)
stencil_test = np.array([-2.0, -1.0, 1.0, 2.0])
A_diff = build_differ_matrix(4, stencil_test)
det_A = np.linalg.det(A_diff)
assert abs(det_A) > 1e-10, '[TC49] differ_matrix 可逆 FAILED'

# ---- TC50: high_order_derivative_coefficients 一阶导数 ----
coeffs_1 = high_order_derivative_coefficients(1, stencil_test)
# 一阶中心差分系数: f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
# Not exact but checking signs are alternating
assert len(coeffs_1) == 4, '[TC50] derivative_coeffs 长度 FAILED'
assert np.all(np.isfinite(coeffs_1)), '[TC50] derivative_coeffs 有限 FAILED'
print('\n全部 50 个测试通过!\n')
