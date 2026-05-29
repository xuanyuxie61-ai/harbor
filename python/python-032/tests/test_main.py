"""
核裂变动力学与碎片质量分布 —— 统一入口
========================================

本项目基于 15 个种子科研代码项目的核心算法，
融合构建了一个面向核物理前沿问题的博士级计算框架。

运行方式:
  python main.py

无需任何参数，自动执行完整的裂变动力学模拟流程。
"""

import numpy as np
import time

# 导入各模块
from collective_coordinates import (
    nuclear_radius,
    mass_asymmetry_to_fragment_mass,
    fragment_mass_to_asymmetry,
    closest_point_brute,
    triangulate_configuration_space_1d,
)
from potential_energy_surface import (
    liquid_drop_energy,
    shell_correction_energy,
    pairing_correction_energy,
    fission_barrier_height,
    potential_energy,
    zero_laguerre,
    zero_muller,
    find_saddle_point_1d,
    find_scission_point_1d,
)
from chebyshev_pes import (
    chebyshev_coefficients,
    chebyshev_evaluate,
    chebyshev_derivative,
    build_chebyshev_pes_approximation,
)
from diffusion_coefficient import (
    nuclear_temperature,
    wall_formula_viscosity,
    one_body_dissipation,
    diffusion_tensor,
    diffusion_deriv_1d,
    alnorm,
    gammad,
)
from fokker_planck_fem import (
    assemble_fem_tridiagonal,
    solve_tridiagonal,
    fokker_planck_steady_state,
    fokker_planck_time_stepping,
    sparse_matrix_vector_product,
)
from langevin_ode import (
    euler_maruyama_step,
    langevin_dynamics_1d,
    langevin_ensemble_mass_distribution,
    robertson_like_stiff_test,
    rk4_step,
    fission_decay_dynamics,
)
from mass_yield_mc import (
    triangle_monte_carlo_integral,
    gaussian_mass_distribution,
    bimodal_mass_distribution,
    importance_sampling_mc_yield,
    scission_point_yield_model,
)
from multi_dimensional_integral import (
    legendre_gauss_nodes_weights,
    five_dimensional_gauss_quadrature,
    partition_function_integral,
)
from fission_pathway import (
    FissionPathwayGraph,
    digraph_is_eulerian,
    dijkstra_min_energy_path,
    build_fission_network_from_pes,
    pathway_entropy,
)
from cvt_partition import (
    cvt_iterate_2d,
    cvt_partition_fission_space,
)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(42)
    start_time = time.time()
    
    # ============================================================
    # 物理参数设定：U-235 热中子诱发裂变
    # ============================================================
    mass_number = 235
    charge_number = 92
    excitation_energy = 6.5  # MeV, 热中子俘获后激发能
    
    print_section("核裂变动力学与碎片质量分布模拟")
    print(f"母核: A={mass_number}, Z={charge_number}")
    print(f"激发能: E* = {excitation_energy} MeV")
    
    # ============================================================
    # 1. 核几何与集体坐标
    # ============================================================
    print_section("1. 核几何参数与集体坐标空间")
    R0 = nuclear_radius(mass_number)
    print(f"等效核半径 R0 = {R0:.3f} fm")
    
    beta3_test = 0.3
    A_L, A_H = mass_asymmetry_to_fragment_mass(beta3_test, mass_number)
    print(f"β₃ = {beta3_test} => 轻碎片 A_L={A_L:.1f}, 重碎片 A_H={A_H:.1f}")
    
    beta3_back = fragment_mass_to_asymmetry(A_L, mass_number)
    print(f"反推 β₃ = {beta3_back:.4f} (验证一致性)")
    
    # 最近邻测试
    ref_points = np.array([
        [0.2, 0.0], [0.5, 0.1], [1.0, -0.2], [1.5, 0.3]
    ])
    target = np.array([0.9, -0.1])
    idx, dist = closest_point_brute(ref_points, target)
    print(f"最近邻参考构型: 索引={idx}, 距离={dist:.4f}")
    
    # 网格剖分
    nodes_1d = triangulate_configuration_space_1d(50, -0.3, 2.5)
    print(f"一维 β₂ 网格节点数: {len(nodes_1d)}")
    
    # ============================================================
    # 2. 裂变势能面 (PES) 分析
    # ============================================================
    print_section("2. 裂变势能面分析")
    
    E_ldm_gs = liquid_drop_energy(mass_number, charge_number, 0.0)
    E_ldm_def = liquid_drop_energy(mass_number, charge_number, 1.0)
    print(f"液滴模型: E_gs(β₂=0) = {E_ldm_gs:.2f} MeV")
    print(f"液滴模型: E(β₂=1.0) = {E_ldm_def:.2f} MeV")
    
    delta_shell = shell_correction_energy(0.0, 0.0, mass_number)
    print(f"球形壳修正: δE_shell = {delta_shell:.2f} MeV")
    
    E_barrier = fission_barrier_height(mass_number, charge_number)
    print(f"裂变势垒高度(经验): E_B = {E_barrier:.2f} MeV")
    
    # 鞍点与断裂点搜索
    beta2_saddle, V_saddle = find_saddle_point_1d(mass_number, charge_number)
    print(f"鞍点位置: β₂^saddle = {beta2_saddle:.4f}, V_saddle = {V_saddle:.2f} MeV")
    
    beta2_scission, V_scission = find_scission_point_1d(
        mass_number, charge_number, beta2_saddle
    )
    print(f"断裂点位置: β₂^scission = {beta2_scission:.4f}, V_scission = {V_scission:.2f} MeV")
    
    # Laguerre 求根验证：搜索 dV/dβ₂ = 0
    def dV_numeric(b):
        h = 1e-5
        return (potential_energy(np.array([b+h,0,0,0,0]), mass_number, charge_number) -
                potential_energy(np.array([b-h,0,0,0,0]), mass_number, charge_number)) / (2*h)
    
    root_lag, ierr, ksteps = zero_laguerre(dV_numeric, beta2_saddle, degree=6, abserr=1e-8)
    print(f"Laguerre求根: β₂ = {root_lag:.4f}, 迭代={ksteps}, 误差码={ierr}")
    
    # Muller 求根（复平面）
    def V_complex(z):
        if abs(z.imag) < 1e-14:
            return complex(potential_energy(np.array([z.real,0,0,0,0]), mass_number, charge_number))
        return complex(1e10, 0)
    
    root_muller, fx = zero_muller(
        V_complex, complex(0.5), complex(1.0), complex(1.5),
        fatol=1e-8, xatol=1e-8, itmax=50
    )
    print(f"Muller求根: β₂ = {root_muller.real:.4f} + {root_muller.imag:.4e}j, |f|={abs(fx):.4e}")
    
    # ============================================================
    # 3. Chebyshev 势能面逼近
    # ============================================================
    print_section("3. Chebyshev 级数逼近")
    c_cheb, b_min, b_max = build_chebyshev_pes_approximation(
        mass_number, charge_number, beta2_min=-0.3, beta2_max=2.5, n_terms=24
    )
    print(f"Chebyshev 系数非零项数: {np.sum(np.abs(c_cheb) > 1e-3)}")
    print(f"首项系数 c0 = {c_cheb[0]:.4f}, 尾项 |c23| = {abs(c_cheb[-1]):.4e}")
    
    # 验证逼近精度
    test_points = np.linspace(-0.2, 2.3, 10)
    max_err = 0.0
    for b in test_points:
        V_exact = potential_energy(np.array([b,0,0,0,0]), mass_number, charge_number)
        V_cheb = chebyshev_evaluate(np.array([b]), c_cheb, b_min, b_max)[0]
        err = abs(V_exact - V_cheb)
        if err > max_err:
            max_err = err
    print(f"Chebyshev 最大逼近误差: {max_err:.4f} MeV")
    
    # ============================================================
    # 4. 扩散系数与温度
    # ============================================================
    print_section("4. 扩散张量与粘滞系数")
    T_nucleus = nuclear_temperature(excitation_energy, mass_number)
    print(f"核温度: T = {T_nucleus:.3f} MeV")
    
    gamma_wall = wall_formula_viscosity(mass_number, beta2=0.5)
    print(f"Wall-Formula 粘滞: γ = {gamma_wall:.3f} (自然单位)")
    
    gamma_tensor = one_body_dissipation(mass_number, charge_number, beta2=0.5, beta3=0.2)
    print(f"OBD 张量 γ_22 = {gamma_tensor[0,0]:.3f}, γ_33 = {gamma_tensor[1,1]:.3f}")
    
    D_tensor = diffusion_tensor(excitation_energy, mass_number, charge_number, beta2=0.5, beta3=0.2)
    print(f"扩散张量 D_22 = {D_tensor[0,0]:.6f}, D_33 = {D_tensor[1,1]:.6f}")
    
    # alnorm 测试
    p_trans = alnorm(np.sqrt(2.0 * E_barrier / T_nucleus), upper=True)
    print(f"势垒透射系数(近似): P = {p_trans:.4e}")
    
    # gammad 测试
    gamma_val = gammad(3.0, 2.0)
    print(f"不完全伽马函数 P(3,2) = {gamma_val:.6f}")
    
    # ============================================================
    # 5. Fokker-Planck FEM 求解
    # ============================================================
    print_section("5. Fokker-Planck 方程 FEM 求解")
    x_nodes = np.linspace(-1.0, 1.0, 101)
    
    def V_1d(x_arr):
        return np.array([potential_energy(np.array([xi if abs(xi)<2.0 else 2.0,0,0,0,0]), mass_number, charge_number) for xi in x_arr])
    
    def dV_1d(x_arr):
        h = 1e-5
        return np.array([
            (potential_energy(np.array([xi+h,0,0,0,0]), mass_number, charge_number) -
             potential_energy(np.array([xi-h,0,0,0,0]), mass_number, charge_number)) / (2*h)
            for xi in x_arr
        ])
    
    def D_const(x_arr):
        return np.full_like(x_arr, D_tensor[1,1])
    
    P_steady = fokker_planck_steady_state(x_nodes, V_1d, T_nucleus, D_const=D_const)
    print(f"稳态分布积分归一化: {np.trapezoid(P_steady, x_nodes):.6f}")
    print(f"稳态分布峰值位置: β₃ = {x_nodes[np.argmax(P_steady)]:.3f}")
    
    # 时间演化
    P0 = np.exp(-0.5 * ((x_nodes - 0.3) / 0.1) ** 2)
    P0 = P0 / np.trapezoid(P0, x_nodes)
    P_evolved = fokker_planck_time_stepping(
        x_nodes, P0, D_const, dV_1d, T_nucleus, dt=0.001, n_steps=100
    )
    print(f"演化后分布峰值: β₃ = {x_nodes[np.argmax(P_evolved)]:.3f}")
    print(f"演化后分布宽度(FWHM): {np.std(P_evolved):.4f}")
    
    # 稀疏矩阵向量乘法测试
    col_ptr = np.array([0, 2, 4, 6])
    row_ind = np.array([0, 1, 0, 1, 0, 1])
    values = np.array([1.0, 0.5, 0.5, 2.0, 0.3, 0.8])
    vec = np.array([1.0, 2.0])
    result = sparse_matrix_vector_product(col_ptr, row_ind, values, vec)
    print(f"稀疏矩阵向量乘法测试: result = [{result[0]:.2f}, {result[1]:.2f}]")
    
    # ============================================================
    # 6. Langevin 动力学与质量分布
    # ============================================================
    print_section("6. Langevin 系综模拟")
    
    # 定义 β₃ 方向的简化势能
    def V_beta3(b3):
        return potential_energy(np.array([1.0, b3, 0.0, 0.0, 0.0]), mass_number, charge_number)
    
    # 单条轨迹
    t_arr, x_traj = langevin_dynamics_1d(
        V_beta3, gamma=gamma_tensor[1,1], T=T_nucleus,
        x0=0.1, t_max=5.0, dt=0.01, x_bounds=(-1.5, 1.5)
    )
    print(f"单轨迹终点 β₃ = {x_traj[-1]:.4f}")
    
    # 系综模拟（减少轨迹数以加速演示）
    n_traj = 2000
    mass_centers_mc, yield_mc = langevin_ensemble_mass_distribution(
        V_beta3, gamma=gamma_tensor[1,1], T=T_nucleus,
        x0=0.0, t_max=3.0, dt=0.01, n_trajectories=n_traj,
        mass_number=mass_number, n_bins=40, x_bounds=(-1.2, 1.2)
    )
    print(f"Langevin MC 产额: 峰值 A ≈ {mass_centers_mc[np.argmax(yield_mc)]:.1f}")
    print(f"Langevin MC 产额: 最大产额 ≈ {np.max(yield_mc):.4e}")
    
    # Robertson 型刚性 ODE 测试
    y0 = np.array([1.0, 0.0, 0.0])
    t_rob = 0.0
    dt_rob = 0.001
    y_rob = y0.copy()
    for _ in range(1000):
        y_rob = rk4_step(robertson_like_stiff_test, t_rob, y_rob, dt_rob)
        t_rob += dt_rob
    print(f"Robertson ODE 终点: y=[{y_rob[0]:.4e}, {y_rob[1]:.4e}, {y_rob[2]:.4f}]")
    
    # 裂变竞争动力学
    t_comp, N_c, N_f, N_n, N_g = fission_decay_dynamics(
        lambda_fission=0.3, lambda_neutron=0.1, lambda_gamma=0.05,
        t_max=10.0, dt=0.01
    )
    print(f"裂变竞争: t=10 zs 时 P_fission={N_f[-1]:.4f}, P_neutron={N_n[-1]:.4f}, P_gamma={N_g[-1]:.4f}")
    
    # ============================================================
    # 7. 蒙特卡洛质量产额
    # ============================================================
    print_section("7. 蒙特卡洛质量产额计算")
    
    # 三角形积分测试
    def f_triangle(xy):
        return xy[:,0] ** 2 + xy[:,1] ** 2
    
    I_tri = triangle_monte_carlo_integral(f_triangle, n_samples=50000)
    print(f"三角形区域积分 ∫∫(x²+y²)dxdy ≈ {I_tri:.6f} (理论值=1/6≈0.1667)")
    
    # 重要性采样 MC
    mass_centers_imp, yield_imp = importance_sampling_mc_yield(
        mass_number, charge_number, excitation_energy, n_samples=5000
    )
    print(f"重要性采样: 产额峰值 A ≈ {mass_centers_imp[np.argmax(yield_imp)]:.1f}")
    
    # 断裂点模型
    mass_centers_sci, yield_sci = scission_point_yield_model(
        mass_number, charge_number, excitation_energy
    )
    print(f"断裂点模型: 轻峰 A ≈ {mass_centers_sci[np.argmax(yield_sci[:50])]:.1f}")
    print(f"断裂点模型: 重峰 A ≈ {mass_centers_sci[50+np.argmax(yield_sci[50:])]:.1f}")
    
    # ============================================================
    # 8. 多维高斯积分
    # ============================================================
    print_section("8. 多维相空间积分")
    
    def test_integrand(q_array):
        vals = np.zeros(len(q_array))
        for i in range(len(q_array)):
            V = potential_energy(q_array[i], mass_number, charge_number)
            vals[i] = np.exp(-V / max(T_nucleus, 0.1))
        return vals
    
    Z_partition = five_dimensional_gauss_quadrature(
        test_integrand, n_per_dim=4, mass_number=mass_number
    )
    print(f"五维配分函数 Z = {Z_partition:.4e}")
    
    Z_direct = partition_function_integral(
        mass_number, charge_number, excitation_energy, n_per_dim=4
    )
    print(f"配分函数(专用接口) Z = {Z_direct:.4e}")
    
    # ============================================================
    # 9. 裂变路径网络
    # ============================================================
    print_section("9. 裂变路径网络分析")
    
    graph = build_fission_network_from_pes(
        mass_number, charge_number, excitation_energy,
        n_grid_beta2=12, n_grid_beta3=12
    )
    print(f"网络节点数: {graph.n_nodes}")
    
    euler_status = digraph_is_eulerian(graph)
    euler_text = {0: "非Eulerian", 1: "开放Euler路径", 2: "闭合Euler回路"}
    print(f"Eulerian 性质: {euler_text.get(euler_status, '未知')}")
    
    if graph.n_nodes > 1:
        path, cost = dijkstra_min_energy_path(graph, source=0, target=graph.n_nodes - 1)
        print(f"最小能量路径: 节点序列长度={len(path)}, 总代价={cost:.4f}")
    
    ent = pathway_entropy(graph, source=0)
    print(f"路径熵: 平均={np.mean(ent):.4f}, 最大={np.max(ent):.4f}")
    
    # ============================================================
    # 10. CVT 构型空间划分
    # ============================================================
    print_section("10. CVT 自适应构型空间划分")
    
    z_opt, diff, energy = cvt_partition_fission_space(
        mass_number, charge_number, excitation_energy,
        n_generators=12, n_samples=2000
    )
    print(f"CVT 收敛位移: {diff:.6f}")
    print(f"CVT 离散能量: {energy:.6f}")
    print(f"生成点 β₂ 范围: [{z_opt[:,0].min():.3f}, {z_opt[:,0].max():.3f}]")
    print(f"生成点 β₃ 范围: [{z_opt[:,1].min():.3f}, {z_opt[:,1].max():.3f}]")
    
    # ============================================================
    # 总结
    # ============================================================
    print_section("模拟总结")
    elapsed = time.time() - start_time
    print(f"总运行时间: {elapsed:.2f} 秒")
    print("所有模块执行完毕，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（36个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: nuclear_radius 对正质量数返回有限正值 ----
r = nuclear_radius(235)
assert r > 0 and np.isfinite(r), '[TC01] nuclear_radius 返回非正值或无穷 FAILED'

# ---- TC02: mass_asymmetry_to_fragment_mass 对称裂变时两碎片质量相等 ----
A_L, A_H = mass_asymmetry_to_fragment_mass(0.0, 235)
assert abs(A_L - A_H) < 1e-10 and abs(A_L - 117.5) < 1e-10, '[TC02] 对称裂变碎片质量不等 FAILED'

# ---- TC03: mass_asymmetry_to_fragment_mass 与 fragment_mass_to_asymmetry 互逆 ----
beta3_orig = 0.3
A_L, A_H = mass_asymmetry_to_fragment_mass(beta3_orig, 235)
beta3_back = fragment_mass_to_asymmetry(A_L, 235)
assert abs(beta3_back - beta3_orig) < 1e-6, '[TC03] 质量-不对称度转换不互逆 FAILED'

# ---- TC04: closest_point_brute 最近邻索引与距离正确 ----
pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
tgt = np.array([0.9, 0.1])
idx, dist = closest_point_brute(pts, tgt)
assert idx == 1 and abs(dist - np.sqrt(0.02)) < 1e-10, '[TC04] 最近邻搜索错误 FAILED'

# ---- TC05: triangulate_configuration_space_1d 输出节点数与端点正确 ----
nodes = triangulate_configuration_space_1d(50, -0.3, 2.5)
assert len(nodes) == 51 and nodes[0] == -0.3 and nodes[-1] == 2.5, '[TC05] 1D网格节点数或端点错误 FAILED'

# ---- TC06: liquid_drop_energy 球形与变形构型能量均有限且不同 ----
E_gs = liquid_drop_energy(235, 92, 0.0)
E_def = liquid_drop_energy(235, 92, 1.0)
assert np.isfinite(E_gs) and np.isfinite(E_def) and abs(E_def - E_gs) > 1e-6, '[TC06] 液滴模型能量异常或球形与变形相等 FAILED'

# ---- TC07: shell_correction_energy 大形变下绝对值衰减 ----
delta_shell_small = shell_correction_energy(0.0, 0.0, 235)
delta_shell_large = shell_correction_energy(2.0, 2.0, 235)
assert abs(delta_shell_large) < abs(delta_shell_small), '[TC07] 壳修正未随大形变衰减 FAILED'

# ---- TC08: fission_barrier_height 非负且有限 ----
Eb = fission_barrier_height(235, 92)
assert Eb >= 0 and np.isfinite(Eb), '[TC08] 裂变势垒高度为负或无穷 FAILED'

# ---- TC09: potential_energy 输入长度不足时抛出 ValueError ----
try:
    potential_energy(np.array([0.0, 0.0]), 235, 92)
    assert False, '[TC09] 短输入未触发异常 FAILED'
except ValueError:
    pass

# ---- TC10: pairing_correction_energy 返回值非正 ----
E_pair = pairing_correction_energy(1.0, 1.5)
assert E_pair <= 0, '[TC10] 配对修正能为正 FAILED'

# ---- TC11: zero_laguerre 对线性函数精确收敛到根 ----
root_lag, ierr, _ = zero_laguerre(lambda x: x - 2.0, 1.0, degree=4, abserr=1e-10)
assert abs(root_lag - 2.0) < 1e-6 and ierr == 0, '[TC11] Laguerre求根失败 FAILED'

# ---- TC12: chebyshev_evaluate 常数函数估值精确 ----
c_const = np.array([5.0, 0.0, 0.0])
val = chebyshev_evaluate(np.array([0.0, 0.5, 1.0]), c_const, -1.0, 1.0)
assert np.allclose(val, 5.0), '[TC12] Chebyshev常数函数估值错误 FAILED'

# ---- TC13: chebyshev_derivative 常数函数导数为零 ----
c_const2 = np.array([5.0, 0.0, 0.0])
dV = chebyshev_derivative(np.array([0.0, 0.5, 1.0]), c_const2, -1.0, 1.0)
assert np.allclose(dV, 0.0), '[TC13] Chebyshev常数函数导数不为零 FAILED'

# ---- TC14: build_chebyshev_pes_approximation 返回参数正确 ----
c_cheb, b_min, b_max = build_chebyshev_pes_approximation(235, 92, beta2_min=-0.3, beta2_max=2.5, n_terms=8)
assert len(c_cheb) == 9 and b_min == -0.3 and b_max == 2.5, '[TC14] Chebyshev逼近返回参数错误 FAILED'

# ---- TC15: nuclear_temperature 非负且随激发能单调不减 ----
T1 = nuclear_temperature(0.0, 235)
T2 = nuclear_temperature(6.5, 235)
assert T1 >= 0 and T2 > T1, '[TC15] 核温度非正或随激发能递减 FAILED'

# ---- TC16: alnorm 满足对称性 P(-x) = 1 - P(x) ----
p_pos = alnorm(1.0, upper=True)
p_neg = alnorm(-1.0, upper=True)
assert abs(p_pos + p_neg - 1.0) < 1e-10, '[TC16] alnorm对称性不满足 FAILED'

# ---- TC17: gammad 在 x=0 时返回 0 ----
g0 = gammad(0.0, 2.0)
assert g0 == 0.0, '[TC17] gammad(0,p)不为0 FAILED'

# ---- TC18: solve_tridiagonal 对角系统精确解 ----
u = solve_tridiagonal(np.array([2.0, 3.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([4.0, 9.0]))
assert np.allclose(u, [2.0, 3.0]), '[TC18] 三对角对角系统求解错误 FAILED'

# ---- TC19: sparse_matrix_vector_product 对角稀疏矩阵结果正确 ----
cp = np.array([0, 1, 2])
ri = np.array([0, 1])
va = np.array([3.0, 4.0])
v = np.array([1.0, 2.0])
res = sparse_matrix_vector_product(cp, ri, va, v)
assert np.allclose(res, [3.0, 8.0]), '[TC19] 稀疏对角矩阵乘法错误 FAILED'

# ---- TC20: rk4_step 对指数增长 ODE 单步精度足够 ----
def dydt_exp(t, y): return y
y_next = rk4_step(dydt_exp, 0.0, np.array([1.0]), 0.1)
assert abs(y_next[0] - np.exp(0.1)) < 1e-6, '[TC20] RK4单步精度不足 FAILED'

# ---- TC21: fission_decay_dynamics 总概率守恒 ----
t_arr, N_c, N_f, N_n, N_g = fission_decay_dynamics(0.3, 0.1, 0.05, 1.0, 0.01)
total_prob = N_c[-1] + N_f[-1] + N_n[-1] + N_g[-1]
assert abs(total_prob - 1.0) < 1e-6, '[TC21] 裂变竞争概率不守恒 FAILED'

# ---- TC22: gaussian_mass_distribution 在宽区间积分近似归一 ----
mass_centers = np.linspace(80, 160, 1000)
y_dist = gaussian_mass_distribution(mass_centers, 117.5, 5.0)
integral = np.trapezoid(y_dist, mass_centers)
assert abs(integral - 1.0) < 0.05, '[TC22] 高斯分布积分不归一 FAILED'

# ---- TC23: bimodal_mass_distribution 双峰峰值位置正确 ----
mc = np.linspace(60, 180, 1000)
y_bi = bimodal_mass_distribution(mc, 90.0, 145.0, 3.0, 4.0, 1.0)
peak_light = mc[np.argmax(y_bi[:500])]
peak_heavy = mc[500 + np.argmax(y_bi[500:])]
assert abs(peak_light - 90.0) < 2.0 and abs(peak_heavy - 145.0) < 2.0, '[TC23] 双峰位置偏差过大 FAILED'

# ---- TC24: triangle_monte_carlo_integral 常数函数等于面积 ----
np.random.seed(42)
I_const = triangle_monte_carlo_integral(lambda xy: np.ones(len(xy)), n_samples=1000)
assert abs(I_const - 0.5) < 0.05, '[TC24] 三角形常数积分不接近理论值 FAILED'

# ---- TC25: legendre_gauss_nodes_weights 权重和严格为2 ----
nodes, weights = legendre_gauss_nodes_weights(8)
assert abs(np.sum(weights) - 2.0) < 1e-12, '[TC25] Legendre-Gauss权重和不为2 FAILED'

# ---- TC26: partition_function_integral 固定输入结果可复现 ----
Z1 = partition_function_integral(235, 92, 6.5, n_per_dim=3)
Z2 = partition_function_integral(235, 92, 6.5, n_per_dim=3)
assert abs(Z1 - Z2) < 1e-12, '[TC26] 配分函数两次调用结果不一致 FAILED'

# ---- TC27: euler_maruyama_step 零扩散退化为确定性漂移 ----
np.random.seed(42)
x_det = euler_maruyama_step(np.array([1.0, 2.0]), np.array([0.5, -0.3]), np.zeros((2, 2)), 0.1)
assert np.allclose(x_det, np.array([1.05, 1.97])), '[TC27] Euler-Maruyama零扩散步错误 FAILED'

# ---- TC28: diffusion_deriv_1d 均匀场时间导数为零 ----
x_grid = np.linspace(0.0, 1.0, 11)
u_uniform = np.ones(11)
dudt = diffusion_deriv_1d(0.0, u_uniform, x_grid, 1.0)
assert np.allclose(dudt, 0.0), '[TC28] 均匀场扩散导数不为零 FAILED'

# ---- TC29: FissionPathwayGraph 出入度序列正确 ----
g = FissionPathwayGraph(3)
g.add_edge(0, 1, 0.5)
g.add_edge(1, 2, 0.3)
indeg, outdeg = g.degree_sequence()
assert outdeg[0] == 1 and indeg[2] == 1 and outdeg[2] == 0, '[TC29] 图度序列错误 FAILED'

# ---- TC30: digraph_is_eulerian 检测闭合 Euler 回路 ----
g_euler = FissionPathwayGraph(3)
g_euler.add_edge(0, 1, 0.5)
g_euler.add_edge(1, 2, 0.5)
g_euler.add_edge(2, 0, 0.5)
status = digraph_is_eulerian(g_euler)
assert status == 2, '[TC30] 闭合Euler回路检测失败 FAILED'

# ---- TC31: pathway_entropy 非负性 ----
g_ent = FissionPathwayGraph(2)
g_ent.add_edge(0, 1, 0.5)
g_ent.add_edge(0, 1, 0.3)
ent = pathway_entropy(g_ent, source=0)
assert np.all(ent >= 0), '[TC31] 路径熵为负 FAILED'

# ---- TC32: langevin_dynamics_1d 输出数组长度一致 ----
np.random.seed(42)
def V_harm(x): return 0.5 * x * x
t_arr, x_arr = langevin_dynamics_1d(V_harm, 1.0, 0.1, 0.0, 1.0, 0.01, (-2.0, 2.0))
assert len(t_arr) == len(x_arr) and len(t_arr) == 101, '[TC32] Langevin轨迹长度错误 FAILED'

# ---- TC33: build_fission_network_from_pes 返回非空图 ----
np.random.seed(42)
graph = build_fission_network_from_pes(235, 92, 6.5, n_grid_beta2=12, n_grid_beta3=12)
assert graph.n_nodes > 0, '[TC33] 裂变网络节点数为0 FAILED'

# ---- TC34: fokker_planck_steady_state 积分严格归一 ----
x_nodes_fp = np.linspace(-1.0, 1.0, 51)
P_fp = fokker_planck_steady_state(x_nodes_fp, lambda x: 0.5 * x * x, 1.0)
Z_fp = np.trapezoid(P_fp, x_nodes_fp)
assert abs(Z_fp - 1.0) < 1e-6, '[TC34] Fokker-Planck稳态分布不归一 FAILED'

# ---- TC35: partition_function_integral 返回有限正值 ----
np.random.seed(42)
Z_val = partition_function_integral(235, 92, 6.5, n_per_dim=3)
assert Z_val > 0 and np.isfinite(Z_val), '[TC35] 配分函数非正或无穷 FAILED'

# ---- TC36: cvt_partition_fission_space 生成点落在物理范围内 ----
np.random.seed(42)
z_opt, diff_val, energy_val = cvt_partition_fission_space(235, 92, 6.5, n_generators=4, n_samples=100)
assert np.all(z_opt[:, 0] >= -0.5) and np.all(z_opt[:, 0] <= 2.5), '[TC36] CVT生成点β₂超出范围 FAILED'

print('\n全部 36 个测试通过!\n')
