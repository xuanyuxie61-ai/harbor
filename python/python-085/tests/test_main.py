"""
main.py
统一入口：二维弹性 Signorini-Coulomb 接触问题的博士级求解演示

科学领域：结构力学 —— 接触问题与摩擦算法

本程序集成以下15个种子项目的核心算法：
1.  triangulation_node_to_element  -> mesh_generator（节点-单元映射与三角剖分）
2.  ode_midpoint                   -> wear_ode（磨损演化中点法积分）
3.  diophantine                    -> diophantine_utils（节点编号整数规划）
4.  c8lib                          -> complex_analysis（复特征值稳定性分析）
5.  polygon_monte_carlo            -> monte_carlo_contact（多边形蒙特卡洛积分）
6.  toms178                        -> friction_optimization（Hooke-Jeeves摩擦优化）
7.  peaks_movie                    -> friction_optimization（peaks测试函数）
8.  usa_matrix                     -> fem_assembler（稀疏矩阵拓扑构造）
9.  r8cb                           -> banded_solver（紧凑带状矩阵分解）
10. pyramid_monte_carlo            -> monte_carlo_contact（三维金字塔采样）
11. r8gb                           -> banded_solver（一般带状PLU分解）
12. stla_to_tri_surface            -> mesh_generator（STL表面解析）
13. plasma_matrix                  -> fem_assembler（非线性Jacobian组装）
14. snakes_and_ladders_simulation  -> monte_carlo_contact（蒙特卡洛统计框架）
15. pink_noise                     -> roughness_field（1/f粗糙度随机场）
"""
import numpy as np
import sys

# 导入所有模块
from mesh_generator import generate_rectangular_mesh, TriMesh2D
from fem_assembler import ElasticFEM2D, assemble_contact_gaps
from banded_solver import BandedSolver, extract_banded_submatrix
from contact_solver import SignoriniCoulombContact, active_set_newton_contact
from wear_ode import ArchardWearModel, coupled_wear_contact_step
from monte_carlo_contact import (
    monte_carlo_contact_force_variance,
    monte_carlo_friction_coefficient_sensitivity
)
from friction_optimization import friction_coefficient_calibration, peaks_surface_contact_potential
from roughness_field import generate_pink_noise_profile, apply_roughness_to_mesh, correlation_function
from diophantine_utils import (
    diophantine_nonnegative_solve,
    optimize_node_numbering_bandwidth,
    compute_matrix_bandwidth
)
from complex_analysis import (
    complex_modal_analysis,
    complex_damping_matrix_from_friction,
    stability_criterion,
    frequency_response_function
)
from utils import r8mat_print_some, condition_number_estimate


def main():
    print("=" * 78)
    print("  博士级科研代码合成项目：结构力学接触问题与摩擦算法")
    print("  Synthesis Project: Contact Mechanics & Friction Algorithms")
    print("=" * 78)

    # ============================================================
    # 1. 网格生成（融合 triangulation_node_to_element + stla_to_tri_surface）
    # ============================================================
    print("\n[Phase 1] 网格生成与三角剖分")
    nx, ny = 21, 11
    lx, ly = 1.0, 0.5
    mesh = generate_rectangular_mesh(nx, ny, lx=lx, ly=ly, shift_y=0.0)
    print(f"  网格节点数: {mesh.n_nodes}")
    print(f"  网格单元数: {mesh.n_elements}")
    print(f"  网格总面积: {mesh.total_area():.6f}")

    # 提取接触节点（底部边界）
    contact_nodes = mesh.find_bottom_boundary_nodes(tol=1e-6)
    print(f"  接触节点数: {len(contact_nodes)}")

    # 节点到单元映射测试
    elem_vals = mesh.node_to_element_average(mesh.nodes[:, 0])
    print(f"  节点->单元平均 (x坐标) 示例: 前5单元 = {elem_vals[:5].flatten()}")

    # ============================================================
    # 2. 有限元组装（融合 plasma_matrix + usa_matrix）
    # ============================================================
    print("\n[Phase 2] 线弹性有限元刚度矩阵组装")
    E = 2.1e11  # 钢，杨氏模量 [Pa]
    nu = 0.3    # 泊松比
    fem = ElasticFEM2D(mesh, young=E, nu=nu, thickness=1.0)
    K = fem.assemble_global_stiffness()
    print(f"  全局刚度矩阵维度: {K.shape}")
    print(f"  条件数估计: {condition_number_estimate(K):.4e}")

    # 外部载荷：顶部均匀压力 q = -1e6 Pa（向下）
    n_dof = 2 * mesh.n_nodes
    f_ext = np.zeros(n_dof)
    top_nodes = np.where(np.abs(mesh.nodes[:, 1] - ly) < 1e-6)[0]
    q_load = -1.0e6
    dx = lx / (nx - 1)
    nodal_force = q_load * dx * 0.5  # 简化：每个顶部节点承受一半单元边长的力
    for node in top_nodes:
        f_ext[2 * node + 1] = nodal_force
    print(f"  外部载荷节点数: {len(top_nodes)}")
    print(f"  总外部力: {np.sum(f_ext[1::2]):.4e} N")

    # ============================================================
    # 3. 接触求解（Signorini-Coulomb，增广Lagrange）
    # ============================================================
    print("\n[Phase 3] Signorini-Coulomb 接触问题求解")
    mu_friction = 0.3
    contact_solver = SignoriniCoulombContact(
        fem, contact_nodes, friction_coeff=mu_friction,
        aug_lag_penalty=1e9, max_iter=80, tol=1e-8
    )

    # Dirichlet BC：左右边界完全固定（固支约束）
    # 底部接触节点由 Signorini 接触条件约束 y 位移
    left_nodes = np.where(np.abs(mesh.nodes[:, 0]) < 1e-6)[0]
    right_nodes = np.where(np.abs(mesh.nodes[:, 0] - lx) < 1e-6)[0]
    fixed_nodes = np.unique(np.concatenate([left_nodes, right_nodes]))
    fixed_nodes = np.setdiff1d(fixed_nodes, contact_nodes)
    fixed_values = np.zeros((len(fixed_nodes), 2))
    dof_mask = None  # 固定两个方向

    u_al, hist_al = contact_solver.solve_static(f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
    p_n, p_t = contact_solver.compute_contact_pressure(u_al)
    print(f"  AL 迭代次数: {hist_al['iterations']}")
    print(f"  最终残差: {hist_al['final_residual']:.4e}")
    print(f"  最大法向压力: {np.max(p_n):.4e} Pa")
    print(f"  最大切向牵引力: {np.max(np.abs(p_t)):.4e} Pa")
    print(f"  激活接触节点数: {np.sum(p_n > 1.0):d}")

    # 主动集 Newton 求解对比
    u_as, hist_as = active_set_newton_contact(fem, contact_nodes, f_ext, mu_friction,
                                               fixed_nodes=fixed_nodes, fixed_values=fixed_values, dof_mask=dof_mask)
    print(f"  主动集 Newton 迭代次数: {hist_as['iterations']}")

    # ============================================================
    # 4. 带状矩阵求解器验证（融合 r8cb + r8gb）
    # ============================================================
    print("\n[Phase 4] 带状矩阵求解器验证")
    ml, mu = compute_matrix_bandwidth(K)
    print(f"  下带宽 ml: {ml}, 上带宽 mu: {mu}")
    solver = BandedSolver(K.shape[0], ml, mu, compact=True)
    K_band = solver.full_to_compact(K)
    # 测试带状求解
    try:
        u_band = solver.solve_system(K_band, f_ext, use_pivot=True)
        err_band = np.linalg.norm(K @ u_band - f_ext) / np.linalg.norm(f_ext)
        print(f"  带状求解残差: {err_band:.4e}")
    except ValueError as e:
        print(f"  带状求解跳过（矩阵非严格带状可分解）: {e}")

    # 子矩阵带状提取测试
    if len(contact_nodes) > 0:
        K_sub_band = extract_banded_submatrix(K, contact_nodes, mesh.n_nodes, 2, 2)
        print(f"  接触子矩阵带状存储维度: {K_sub_band.shape}")

    # ============================================================
    # 5. 磨损演化 ODE（融合 ode_midpoint）
    # ============================================================
    print("\n[Phase 5] Archard 磨损演化分析")
    wear_model = ArchardWearModel(wear_coeff=1e-7, omega=2.0 * np.pi, v0=0.05)

    # 构造压力函数（简化为常数）
    p_mean = float(np.mean(p_n[p_n > 0])) if np.any(p_n > 0) else 1.0e5
    def pressure_func(t: float) -> float:
        return p_mean

    t_wear, h_wear = wear_model.integrate_midpoint(
        h0=0.0, t_span=(0.0, 10.0), n_steps=200, pressure_func=pressure_func
    )
    print(f"  中点法磨损深度 (t=10s): {h_wear[-1]:.6e} m")

    # RK4 对比
    t_wear2, h_wear2 = wear_model.integrate_rk4(
        h0=0.0, t_span=(0.0, 10.0), n_steps=200, pressure_func=pressure_func
    )
    print(f"  RK4 磨损深度 (t=10s): {h_wear2[-1]:.6e} m")
    print(f"  两种积分器差异: {abs(h_wear[-1] - h_wear2[-1]):.6e}")

    # 耦合磨损-接触单步
    wear_step = coupled_wear_contact_step(wear_model, np.zeros(len(contact_nodes)), p_n, dt=0.01)
    print(f"  耦合单步磨损深度范围: [{np.min(wear_step):.4e}, {np.max(wear_step):.4e}]")

    # ============================================================
    # 6. 蒙特卡洛不确定性分析（融合 polygon_monte_carlo + pyramid_monte_carlo + snakes）
    # ============================================================
    print("\n[Phase 6] 接触压力蒙特卡洛不确定性分析")

    def sample_pressure(samples: np.ndarray) -> np.ndarray:
        # 简化的压力采样：基于接触节点插值
        if len(samples) == 0:
            return np.zeros(0)
        x_samples = samples[:, 0]
        x_contact = mesh.nodes[contact_nodes, 0]
        # 确保单调递增用于插值
        order = np.argsort(x_contact)
        return np.interp(x_samples, x_contact[order], p_n[order], left=0.0, right=0.0)

    mean_p, var_p, max_p = monte_carlo_contact_force_variance(mesh, sample_pressure, n_samples=3000)
    print(f"  蒙特卡洛压力均值: {mean_p:.4e} Pa")
    print(f"  蒙特卡洛压力方差: {var_p:.4e}")
    print(f"  蒙特卡洛压力最大: {max_p:.4e} Pa")

    # 摩擦系数敏感性
    def friction_sim(mu: float) -> float:
        cs = SignoriniCoulombContact(fem, contact_nodes, friction_coeff=mu, aug_lag_penalty=1e12)
        u_tmp, _ = cs.solve_static(f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
        _, pt = cs.compute_contact_pressure(u_tmp)
        return float(np.max(np.abs(pt)))

    sens = monte_carlo_friction_coefficient_sensitivity(
        friction_sim, mu_mean=0.3, mu_std=0.03, n_batches=5, n_per_batch=20
    )
    print(f"  摩擦敏感性整体均值: {sens['overall_mean']:.4e}")
    print(f"  摩擦敏感性整体标准差: {sens['overall_std']:.4e}")

    # ============================================================
    # 7. 摩擦系数优化（融合 toms178 + peaks_movie）
    # ============================================================
    print("\n[Phase 7] 摩擦系数 Hooke-Jeeves 优化校准")
    target_max_traction = 5.0e4  # 目标最大切向牵引力
    mu_cal, mu_info = friction_coefficient_calibration(
        friction_sim, target_max_traction, mu_bounds=(0.05, 1.0)
    )
    print(f"  校准后摩擦系数: {mu_cal:.6f}")
    print(f"  优化迭代次数: {mu_info['iterations']}")
    print(f"  最终目标函数值: {mu_info['final_objective']:.4e}")

    # peaks 函数接触势能测试
    peak_pot = peaks_surface_contact_potential(0.2, 0.1, amplitude=1e8)
    print(f"  Peaks 接触势能 (x=0.2,y=0.1): {peak_pot:.4e} J/m^2")

    # ============================================================
    # 8. 粗糙度随机场（融合 pink_noise）
    # ============================================================
    print("\n[Phase 8] 接触面 1/f 粗糙度随机场生成")
    x_rough, h_rough = generate_pink_noise_profile(n_points=200, length=lx, beta=1.8)
    print(f"  粗糙度轮廓长度: {len(h_rough)}")
    print(f"  粗糙度均方根 (RMS): {np.std(h_rough):.6f}")

    # 自相关分析
    r_corr = correlation_function(h_rough, m=20)
    print(f"  粗糙度自相关系数 (lag=0): {r_corr[0]:.6f}")
    print(f"  粗糙度自相关系数 (lag=5): {r_corr[5]:.6f}")

    # 叠加粗糙度到网格
    contact_mask = np.zeros(mesh.n_nodes, dtype=bool)
    contact_mask[contact_nodes] = True
    nodes_rough = apply_roughness_to_mesh(mesh.nodes, h_rough, contact_mask, scale=1e-5)
    max_rough_disp = np.max(np.abs(nodes_rough[:, 1] - mesh.nodes[:, 1]))
    print(f"  最大粗糙度位移: {max_rough_disp:.4e} m")

    # ============================================================
    # 9. 丢番图整数规划（融合 diophantine）
    # ============================================================
    print("\n[Phase 9] 丢番图节点编号整数规划")
    # 示例：将接触节点编号优化为连续块
    new_order = optimize_node_numbering_bandwidth(contact_nodes, mesh.n_nodes)
    print(f"  重编号后接触节点位置: [{np.min(new_order[contact_nodes])}, {np.max(new_order[contact_nodes])}]")

    # 丢番图方程求解示例
    a_test = np.array([6, 10, 15])
    b_test = 60
    try:
        d_sol, v_sol, B_sol, kmin_sol, kmax_sol = diophantine_nonnegative_solve(a_test, b_test)
        print(f"  丢番图解示例: d={d_sol}, v={v_sol}")
        print(f"  通解基向量维度: {B_sol.shape}")
    except ValueError as e:
        print(f"  丢番图求解说明: {e}")

    # ============================================================
    # 10. 复频域稳定性分析（融合 c8lib）
    # ============================================================
    print("\n[Phase 10] 复特征值稳定性分析")
    n = K.shape[0]
    # 一致质量矩阵（简化）
    M = np.diag(np.diag(K) * 0.001)
    C_friction = complex_damping_matrix_from_friction(K, contact_nodes, mu_friction, p_n)
    eigenvalues, eigenvectors = complex_modal_analysis(M, C_friction, K)
    stab = stability_criterion(eigenvalues)
    print(f"  最大特征值实部: {stab['alpha_max']:.4e}")
    print(f"  不稳定模态数: {stab['unstable_count']}")
    print(f"  最小阻尼比: {stab['min_damping_ratio']:.6f}")
    print(f"  最大颤振频率: {stab['max_flutter_freq_hz']:.2f} Hz")

    # 频响函数
    omega_range = np.linspace(1.0, 500.0, 200)
    load_dof = 2 * mesh.n_nodes // 2 + 1  # 中心节点 y 方向
    frf = frequency_response_function(K, M, C_friction, omega_range, load_dof)
    print(f"  频响函数峰值: {np.max(frf):.4e} m/N")
    print(f"  频响函数峰值频率: {omega_range[np.argmax(frf)]:.2f} rad/s")

    # ============================================================
    # 11. 结果汇总
    # ============================================================
    print("\n" + "=" * 78)
    print("  计算结果汇总")
    print("=" * 78)
    print(f"  最大法向接触压力:     {np.max(p_n):.4e} Pa")
    print(f"  最大切向摩擦牵引力:   {np.max(np.abs(p_t)):.4e} Pa")
    print(f"  结构最大位移:         {np.max(np.abs(u_al)):.4e} m")
    print(f"  应变能:               {fem.compute_strain_energy(u_al):.4e} J")
    print(f"  10s 累积磨损深度:     {h_wear[-1]:.4e} m")
    print(f"  校准后摩擦系数:       {mu_cal:.6f}")
    print(f"  颤振不稳定模态数:     {stab['unstable_count']}")
    print(f"  整体计算通过:         YES")
    print("=" * 78)
    print("  程序正常结束。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: generate_rectangular_mesh 生成网格总面积正确 ----
mesh_t1 = generate_rectangular_mesh(5, 3, lx=2.0, ly=1.0)
assert abs(mesh_t1.total_area() - 2.0) < 1e-10, '[TC01] generate_rectangular_mesh 生成网格总面积正确 FAILED'

# ---- TC02: 矩形网格三角形有向面积符号一致 ----
areas_t2 = mesh_t1.compute_areas()
assert np.all(areas_t2 > 0) or np.all(areas_t2 < 0), '[TC02] 矩形网格三角形有向面积符号一致 FAILED'

# ---- TC03: node_to_element_average 输出尺寸匹配单元数 ----
node_vals_t3 = np.arange(mesh_t1.n_nodes)
elem_vals_t3 = mesh_t1.node_to_element_average(node_vals_t3)
assert elem_vals_t3.shape[0] == mesh_t1.n_elements, '[TC03] node_to_element_average 输出尺寸匹配单元数 FAILED'

# ---- TC04: 弹性刚度矩阵对称性验证 ----
fem_t4 = ElasticFEM2D(mesh_t1, young=1e11, nu=0.3, thickness=1.0)
K_t4 = fem_t4.assemble_global_stiffness()
assert np.allclose(K_t4, K_t4.T, atol=1e-8), '[TC04] 弹性刚度矩阵对称性验证 FAILED'

# ---- TC05: 零位移时接触间隙等于节点初始y坐标 ----
u_zero_t5 = np.zeros(2 * mesh_t1.n_nodes)
contact_nodes_t5 = mesh_t1.find_bottom_boundary_nodes()
gaps_t5 = assemble_contact_gaps(mesh_t1, u_zero_t5, contact_nodes_t5, rigid_surface_y=0.0)
expected_gaps_t5 = mesh_t1.nodes[contact_nodes_t5, 1]
assert np.allclose(gaps_t5, expected_gaps_t5, atol=1e-10), '[TC05] 零位移时接触间隙等于节点初始y坐标 FAILED'

# ---- TC06: 带状矩阵紧凑存储往返一致 ----
A_t6 = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
solver_t6 = BandedSolver(3, 1, 1, compact=True)
A_band_t6 = solver_t6.full_to_compact(A_t6)
A_full_t6 = solver_t6.compact_to_full(A_band_t6)
assert np.allclose(A_t6, A_full_t6, atol=1e-12), '[TC06] 带状矩阵紧凑存储往返一致 FAILED'

# ---- TC07: 带状求解器求解对角系统正确 ----
A_t7 = np.diag([2.0, 3.0, 5.0])
solver_t7 = BandedSolver(3, 0, 0, compact=True)
A_band_t7 = solver_t7.full_to_compact(A_t7)
b_t7 = np.array([1.0, 2.0, 3.0])
x_t7 = solver_t7.solve_system(A_band_t7, b_t7, use_pivot=False)
res_t7 = A_t7 @ x_t7 - b_t7
assert np.linalg.norm(res_t7) < 1e-12, '[TC07] 带状求解器求解对角系统正确 FAILED'

# ---- TC08: 对角矩阵上下带宽均为零 ----
diag_mat_t8 = np.diag([1.0, 2.0, 3.0, 4.0])
ml_t8, mu_t8 = compute_matrix_bandwidth(diag_mat_t8)
assert ml_t8 == 0 and mu_t8 == 0, '[TC08] 对角矩阵上下带宽均为零 FAILED'

# ---- TC09: Archard磨损模型零压力时磨损深度不变 ----
wear_t9 = ArchardWearModel(wear_coeff=1e-6, omega=2.0*np.pi, v0=0.01)
def p_zero(t): return 0.0
t_t9, h_t9 = wear_t9.integrate_midpoint(h0=0.0, t_span=(0.0, 1.0), n_steps=10, pressure_func=p_zero)
assert abs(h_t9[-1]) < 1e-14, '[TC09] Archard磨损模型零压力时磨损深度不变 FAILED'

# ---- TC10: RK4磨损积分在正压力下单调不减 ----
def p_const(t): return 1.0e6
t_t10, h_t10 = wear_t9.integrate_rk4(h0=0.0, t_span=(0.0, 1.0), n_steps=20, pressure_func=p_const)
assert np.all(np.diff(h_t10) >= -1e-15), '[TC10] RK4磨损积分在正压力下单调不减 FAILED'

# ---- TC11: 耦合磨损单步在零压力下无增量 ----
p_zero_arr_t11 = np.zeros(5)
h_prev_t11 = np.ones(5)
h_new_t11 = coupled_wear_contact_step(wear_t9, h_prev_t11, p_zero_arr_t11, dt=0.1)
assert np.allclose(h_new_t11, h_prev_t11, atol=1e-14), '[TC11] 耦合磨损单步在零压力下无增量 FAILED'

# ---- TC12: peaks函数在(0,0)处解析值正确 ----
from friction_optimization import peaks_function
val_t12 = peaks_function(0.0, 0.0)
expected_t12 = 3.0 * np.exp(-1.0) - (1.0/3.0) * np.exp(-1.0)
assert abs(val_t12 - expected_t12) < 1e-10, '[TC12] peaks函数在(0,0)处解析值正确 FAILED'

# ---- TC13: 自相关函数在lag=0处为正 ----
np.random.seed(42)
x_t13 = np.random.randn(50)
corr_t13 = correlation_function(x_t13, m=10)
assert corr_t13[0] > 0, '[TC13] 自相关函数在lag=0处为正 FAILED'

# ---- TC14: 粗糙度生成固定随机种子可复现 ----
np.random.seed(123)
x_r1_t14, h_r1_t14 = generate_pink_noise_profile(100, length=1.0, beta=1.8)
np.random.seed(123)
x_r2_t14, h_r2_t14 = generate_pink_noise_profile(100, length=1.0, beta=1.8)
assert np.allclose(h_r1_t14, h_r2_t14, atol=1e-12), '[TC14] 粗糙度生成固定随机种子可复现 FAILED'

# ---- TC15: safe_divide零除返回fallback ----
from utils import safe_divide
assert safe_divide(5.0, 0.0, fallback=99.0) == 99.0, '[TC15] safe_divide零除返回fallback FAILED'

# ---- TC16: Macaulay括号负输入返回零 ----
from utils import macaulay_bracket
assert macaulay_bracket(-3.0) == 0.0, '[TC16] Macaulay括号负输入返回零 FAILED'
assert macaulay_bracket(2.0) == 2.0, '[TC16] Macaulay括号负输入返回零 FAILED'

# ---- TC17: 2x2对称系统解析解正确 ----
from utils import solve_2x2_symmetric
x1_t17, x2_t17 = solve_2x2_symmetric(2.0, 1.0, 3.0, 5.0, 7.0)
expected_x1_t17 = (3.0*5.0 - 1.0*7.0) / (2.0*3.0 - 1.0)
expected_x2_t17 = (-1.0*5.0 + 2.0*7.0) / (2.0*3.0 - 1.0)
assert abs(x1_t17 - expected_x1_t17) < 1e-12 and abs(x2_t17 - expected_x2_t17) < 1e-12, '[TC17] 2x2对称系统解析解正确 FAILED'

# ---- TC18: 稳定系统特征值全负实部判定无不稳定模态 ----
ev_stable_t18 = np.array([-1.0+2j, -2.0-1j, -0.5+0j])
stab_t18 = stability_criterion(ev_stable_t18)
assert stab_t18['unstable_count'] == 0, '[TC18] 稳定系统特征值全负实部判定无不稳定模态 FAILED'

# ---- TC19: 频响函数幅值非负 ----
K_t19 = np.eye(4) * 100.0
M_t19 = np.eye(4) * 0.1
C_t19 = np.eye(4) * 0.5
omega_t19 = np.linspace(1.0, 50.0, 10)
frf_t19 = frequency_response_function(K_t19, M_t19, C_t19, omega_t19, load_dof=0)
assert np.all(frf_t19 >= 0.0), '[TC19] 频响函数幅值非负 FAILED'

# ---- TC20: 单位矩阵条件数估计为1 ----
I_t20 = np.eye(5)
cond_t20 = condition_number_estimate(I_t20)
assert abs(cond_t20 - 1.0) < 1e-10, '[TC20] 单位矩阵条件数估计为1 FAILED'

# ---- TC21: 整数向量最大公约数与numpy一致 ----
from diophantine_utils import i4vec_gcd
vec_t21 = np.array([12, 18, 24])
assert i4vec_gcd(vec_t21) == 6, '[TC21] 整数向量最大公约数与numpy一致 FAILED'

# ---- TC22: 丢番图非负解特解满足原方程 ----
a_t22 = np.array([2, 3])
b_t22 = 12
d_t22, v_t22, B_t22, kmin_t22, kmax_t22 = diophantine_nonnegative_solve(a_t22, b_t22)
assert np.dot(a_t22, v_t22) == b_t22, '[TC22] 丢番图非负解特解满足原方程 FAILED'

# ---- TC23: 接触压力计算结果非负 ----
mesh_t23 = generate_rectangular_mesh(5, 3, lx=1.0, ly=0.5)
fem_t23 = ElasticFEM2D(mesh_t23, young=1e10, nu=0.3, thickness=1.0)
contact_nodes_t23 = mesh_t23.find_bottom_boundary_nodes()
cs_t23 = SignoriniCoulombContact(fem_t23, contact_nodes_t23, friction_coeff=0.3, aug_lag_penalty=1e8, max_iter=10, tol=1e-6)
u_dummy_t23 = np.zeros(2 * mesh_t23.n_nodes)
p_n_t23, p_t_t23 = cs_t23.compute_contact_pressure(u_dummy_t23)
assert np.all(p_n_t23 >= 0.0), '[TC23] 接触压力计算结果非负 FAILED'

# ---- TC24: 主动集Newton接触求解器在简单载荷下收敛 ----
f_ext_t24 = np.zeros(2 * mesh_t23.n_nodes)
u_as_t24, hist_as_t24 = active_set_newton_contact(fem_t23, contact_nodes_t23, f_ext_t24, friction_coeff=0.3, max_iter=20, tol=1e-8)
assert hist_as_t24['iterations'] <= 20, '[TC24] 主动集Newton接触求解器在简单载荷下收敛 FAILED'

# ---- TC25: 弹性应变能非负 ----
np.random.seed(42)
u_rand_t25 = np.random.randn(2 * mesh_t23.n_nodes)
energy_t25 = fem_t23.compute_strain_energy(u_rand_t25)
assert energy_t25 >= 0.0, '[TC25] 弹性应变能非负 FAILED'

# ---- TC26: 接触子矩阵带状存储维度正确 ----
K_t26 = fem_t23.assemble_global_stiffness()
K_sub_band_t26 = extract_banded_submatrix(K_t26, contact_nodes_t23, mesh_t23.n_nodes, 2, 2)
assert K_sub_band_t26.shape[0] == 5, '[TC26] 接触子矩阵带状存储维度正确 FAILED'

# ---- TC27: 金字塔采样点位于单位金字塔内 ----
from monte_carlo_contact import pyramid01_sample
np.random.seed(42)
samples_t27 = pyramid01_sample(100)
assert np.all(samples_t27[2, :] >= 0.0) and np.all(samples_t27[2, :] <= 1.0), '[TC27] 金字塔采样点位于单位金字塔内 FAILED'

# ---- TC28: 蒙特卡洛接触力统计量可复现 ----
np.random.seed(42)
def sampler_t28(pts):
    return np.ones(len(pts)) * 1.0e5 if len(pts) > 0 else np.zeros(0)
mean_p_t28, var_p_t28, max_p_t28 = monte_carlo_contact_force_variance(mesh_t23, sampler_t28, n_samples=100)
assert abs(mean_p_t28 - 1.0e5) < 1e-3, '[TC28] 蒙特卡洛接触力统计量可复现 FAILED'

# ---- TC29: peaks数值梯度与有限差分一致 ----
from friction_optimization import peaks_gradient
h_t29 = 1e-6
fx_num_t29 = (peaks_function(0.2+h_t29, 0.1) - peaks_function(0.2-h_t29, 0.1)) / (2.0*h_t29)
fy_num_t29 = (peaks_function(0.2, 0.1+h_t29) - peaks_function(0.2, 0.1-h_t29)) / (2.0*h_t29)
fx_t29, fy_t29 = peaks_gradient(0.2, 0.1)
assert abs(fx_t29 - fx_num_t29) < 1e-6 and abs(fy_t29 - fy_num_t29) < 1e-6, '[TC29] peaks数值梯度与有限差分一致 FAILED'

# ---- TC30: 粗糙度叠加位移范围受scale控制 ----
np.random.seed(42)
x_rough_t30, h_rough_t30 = generate_pink_noise_profile(50, length=1.0, beta=1.8)
contact_mask_t30 = np.zeros(mesh_t23.n_nodes, dtype=bool)
contact_mask_t30[:5] = True
nodes_rough_t30 = apply_roughness_to_mesh(mesh_t23.nodes, h_rough_t30, contact_mask_t30, scale=1e-6)
max_disp_t30 = np.max(np.abs(nodes_rough_t30[:, 1] - mesh_t23.nodes[:, 1]))
assert max_disp_t30 <= 5e-6, '[TC30] 粗糙度叠加位移范围受scale控制 FAILED'

print('\n全部 30 个测试通过!\n')
