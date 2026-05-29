"""
main.py
混凝土结构长期徐变-收缩耦合分析的统一入口

科学问题:
  基于粘弹性有限元的混凝土长期徐变-收缩耦合分析：
  非线性老化本构、随机材料场与不确定性量化。

  本程序模拟一个受载混凝土板在 50 年（18250 天）内的时变力学响应，
  考虑材料老化、环境湿度、随机弹性模量场以及徐变-收缩耦合效应。

运行方式:
  python main.py
"""

import numpy as np
import time

# 导入所有模块
from numeric_utils import (
    machine_epsilon, safe_divide, compute_moments,
    gershgorin_discs, is_diagonally_dominant, relative_residual
)
from mesh_utils import (
    generate_triangular_mesh, mesh_quality_metrics,
    build_adjacency_from_elements, reverse_cuthill_mckee,
    triangle_area
)
from sampling_sequences import (
    hammersley_sequence, halton_sequence,
    quasi_monte_carlo_integral, transform_to_gaussian
)
from spectral_methods import (
    shifted_legendre_polynomial, gauss_legendre_nodes_weights,
    spectral_projection, spectral_reconstruct
)
from stochastic_field import (
    generate_random_field, lognormal_random_field,
    brownian_motion, ornstein_uhlenbeck_process,
    karhunen_loeve_expansion
)
from banded_solver import (
    thomas_algorithm, conjugate_gradient_band,
    gauss_seidel_band, banded_lower_triangular_solve,
    solve_sparse_symmetric_positive_definite
)
from nonlinear_solver import (
    wdk_roots, newton_raphson_scalar, newton_raphson_system,
    companion_matrix_eigenvalues, complex_iterative_refine
)
from concrete_creep_model import (
    b3_compliance_function, b3_creep_coefficient, mc2010_creep_coefficient,
    mc2010_shrinkage_strain, aging_elastic_modulus,
    kelvin_chain_compliance, maxwell_chain_relaxation,
    complex_modulus_maxwell, degree_of_hydration,
    effective_creep_modulus, stress_strain_creep_integral
)
from fem_core import (
    assemble_stiffness_matrix_t6, apply_dirichlet_boundary,
    compute_nodal_forces_uniform, compute_equivalent_creep_load,
    compute_strain_stress_at_nodes,
    plane_stress_constitutive_matrix
)
from viscoelastic_time_integration import (
    backward_euler_viscoelastic, hereditary_integral_discrete,
    viscoelastic_relaxation_spectrum, power_law_creep_kernel,
    effective_time_for_aging_creep, adaptive_time_stepping
)


def setup_problem():
    """
    设置分析问题的几何、材料参数和边界条件。
    """
    print("=" * 70)
    print("混凝土结构长期徐变-收缩耦合分析")
    print("Creep-Shrinkage Coupled Analysis of Concrete Structures")
    print("=" * 70)

    # 几何参数: 2m x 1m 的混凝土板，厚度 0.3m
    Lx, Ly = 2.0, 1.0  # [m]
    thickness = 0.3    # [m]

    # 网格参数
    nx, ny = 9, 5  # 节点数
    print(f"\n[1] 几何与网格")
    print(f"    域尺寸: {Lx}m x {Ly}m, 厚度: {thickness}m")
    print(f"    网格: {nx}x{ny} 节点")

    nodes, elements, boundary_nodes = generate_triangular_mesh(
        nx, ny, domain=(0.0, Lx, 0.0, Ly)
    )

    n_nodes = len(nodes)
    n_elements = len(elements)
    print(f"    节点数: {n_nodes}, 单元数: {n_elements}")

    # 网格质量检查
    quality = mesh_quality_metrics(nodes, elements)
    print(f"    网格质量: 最小角 {quality['min_angle_deg']:.2f}°, "
          f"平均面积 {quality['mean_area']:.4f} m²")

    # 材料参数
    print(f"\n[2] 材料参数")
    E28 = 30000.0       # 28天弹性模量 [MPa]
    nu = 0.2            # 泊松比
    fcm = 35.0          # 抗压强度 [MPa]
    RH = 65.0           # 相对湿度 [%]
    h0 = 200.0          # 名义厚度 [mm]
    cement_type = "N"   # 普通水泥

    print(f"    E28 = {E28} MPa, nu = {nu}")
    print(f"    fcm = {fcm} MPa, RH = {RH}%, h0 = {h0} mm")

    # B3 模型参数 (单位: 1/MPa)
    q1 = 1.0 / E28
    q2 = 20.0e-6
    q3 = 5.0e-6
    q4 = 1.5e-6

    # 随机材料场: 对数正态随机场模拟弹性模量空间变异
    print(f"\n[3] 随机材料场生成 (Karhunen-Loève 展开)")
    correlation_length = 0.5  # [m]
    cov = 0.15  # 变异系数 15%
    median_E = E28

    E_field = lognormal_random_field(
        nodes, median=median_E, cov=cov,
        correlation_length=correlation_length, n_modes=15
    )
    print(f"    弹性模量场: 均值={np.mean(E_field):.1f}, "
          f"std={np.std(E_field):.1f}, COV={np.std(E_field)/np.mean(E_field):.3f}")

    # 边界条件: 底部固定 (y=0), 右侧受均布荷载
    bc_nodes = []
    bc_values = []

    # 底部固定: u_x = 0, u_y = 0
    for i, node in enumerate(nodes):
        if abs(node[1] - 0.0) < 1e-6:
            bc_nodes.extend([2 * i, 2 * i + 1])
            bc_values.extend([0.0, 0.0])

    # 左侧约束水平位移 (对称边界)
    for i, node in enumerate(nodes):
        if abs(node[0] - 0.0) < 1e-6:
            bc_nodes.append(2 * i)
            bc_values.append(0.0)

    bc_nodes = np.array(bc_nodes, dtype=int)
    bc_values = np.array(bc_values)
    print(f"\n[4] 边界条件")
    print(f"    约束自由度: {len(bc_nodes)} 个")

    # 载荷: 自重 + 顶部均布压力
    rho = 2500.0  # 密度 [kg/m³]
    g = 9.81      # 重力加速度 [m/s²]
    qy = -rho * g * 1e-6  # 体力 [MN/m³] -> [MPa/m]
    qx = 0.0

    # 顶部均布压力 0.5 MPa (向下)
    top_pressure = 0.5  # [MPa]

    print(f"\n[5] 载荷条件")
    print(f"    自重: rho*g = {rho*g*1e-6:.4f} MPa/m")
    print(f"    顶部均布压力: {top_pressure} MPa")

    return {
        "nodes": nodes,
        "elements": elements,
        "boundary_nodes": boundary_nodes,
        "E28": E28,
        "E_field": E_field,
        "nu": nu,
        "fcm": fcm,
        "RH": RH,
        "h0": h0,
        "cement_type": cement_type,
        "q1": q1, "q2": q2, "q3": q3, "q4": q4,
        "bc_nodes": bc_nodes,
        "bc_values": bc_values,
        "qx": qx, "qy": qy,
        "top_pressure": top_pressure,
        "thickness": thickness,
        "Lx": Lx, "Ly": Ly,
    }


def solve_instantaneous_elasticity(props):
    """
    求解瞬时弹性问题。
    """
    print(f"\n[6] 瞬时弹性分析")
    nodes = props["nodes"]
    elements = props["elements"]
    E_field = props["E_field"]
    nu = props["nu"]
    thickness = props["thickness"]
    bc_nodes = props["bc_nodes"]
    bc_values = props["bc_values"]

    # 使用平均弹性模量组装刚度矩阵
    E_avg = np.mean(E_field)
    K = assemble_stiffness_matrix_t6(
        nodes, elements, E_avg, nu, thickness, plane_stress=True
    )

    # 体力
    F_body = compute_nodal_forces_uniform(
        nodes, elements, props["qx"], props["qy"], thickness
    )

    # 顶部均布压力 -> 等效节点力
    F_pressure = np.zeros(2 * len(nodes))
    top_nodes = [i for i, node in enumerate(nodes) if abs(node[1] - props["Ly"]) < 1e-6]
    # 简化: 将压力均匀分配到顶部节点
    if len(top_nodes) > 0:
        node_force = -props["top_pressure"] * props["Lx"] * thickness / len(top_nodes)
        for idx in top_nodes:
            F_pressure[2 * idx + 1] = node_force

    F_total = F_body + F_pressure

    # 施加边界条件
    K_bc, F_bc = apply_dirichlet_boundary(K, F_total, bc_nodes, bc_values)

    # 求解
    n_dof = K_bc.shape[0]
    print(f"    自由度: {n_dof}")

    # 检查对角占优
    if is_diagonally_dominant(K_bc, strict=False):
        print(f"    刚度矩阵弱对角占优: True")

    # 使用直接法求解
    u = solve_sparse_symmetric_positive_definite(K_bc, F_bc, method="direct")

    # 检查残差
    res = relative_residual(K_bc, u, F_bc)
    print(f"    相对残差: {res:.2e}")

    # 计算应变和应力
    strains, stresses = compute_strain_stress_at_nodes(
        nodes, elements, u, E_avg, nu, plane_stress=True
    )

    max_stress = np.max(np.abs(stresses))
    print(f"    最大应力: {max_stress:.4f} MPa")
    max_disp = np.max(np.abs(u))
    print(f"    最大位移: {max_disp:.4f} m")

    return u, strains, stresses, K


def solve_time_dependent_creep(props, u0, K_elastic):
    """
    时间相关徐变分析。
    """
    print(f"\n[7] 时间相关徐变分析")
    nodes = props["nodes"]
    elements = props["elements"]
    E28 = props["E28"]
    nu = props["nu"]
    fcm = props["fcm"]
    RH = props["RH"]
    h0 = props["h0"]
    cement_type = props["cement_type"]
    q1, q2, q3, q4 = props["q1"], props["q2"], props["q3"], props["q4"]
    thickness = props["thickness"]
    bc_nodes = props["bc_nodes"]
    bc_values = props["bc_values"]

    # 时间设置: 0 -> 18250 天 (50年)
    t0 = 28.0  # 加载龄期 [day]
    tf = 18250.0  # 50年
    n_steps = 50
    time_points = np.linspace(t0, tf, n_steps)
    print(f"    分析时长: {t0} -> {tf} days ({(tf-t0)/365:.1f} 年)")
    print(f"    时间步数: {n_steps}")

    # 收缩应变历史（每个时间点）
    shrinkage_strains = np.array([
        mc2010_shrinkage_strain(t, 3.0, fcm, RH, h0, cement_type)
        for t in time_points
    ])

    print(f"    最终收缩应变: {shrinkage_strains[-1]:.6f}")

    # 有效刚度法：在每个时间点更新有效模量
    displacements = np.zeros((n_steps, len(u0)))
    displacements[0] = u0

    stresses_history = []
    strains_history = []

    # 初始弹性应力
    E_avg = np.mean(props["E_field"])
    strains_0, stresses_0 = compute_strain_stress_at_nodes(
        nodes, elements, u0, E_avg, nu, plane_stress=True
    )
    stresses_history.append(stresses_0)
    strains_history.append(strains_0)

    # 使用 B3 模型计算各时间点的有效模量
    for step in range(1, n_steps):
        t = time_points[step]
        phi = b3_creep_coefficient(t, t0, E28, q1, q2, q3, q4)
        E_eff = effective_creep_modulus(t, t0, E28, phi)

        # 考虑老化的弹性模量
        E_t = aging_elastic_modulus(t, E28)
        E_combined = E_eff * (E_t / E28)

        # 重新组装刚度矩阵（有效模量）
        K_eff = assemble_stiffness_matrix_t6(
            nodes, elements, E_combined, nu, thickness, plane_stress=True
        )

        # 收缩等效载荷（简化：将收缩视为均匀应变）
        eps_sh = np.array([shrinkage_strains[step], shrinkage_strains[step], 0.0])
        D_eff = plane_stress_constitutive_matrix(E_combined, nu)
        sigma_sh = D_eff @ eps_sh

        # 将收缩应力转换为等效节点力
        F_shrink = np.zeros(2 * len(nodes))
        from fem_core import gauss_points_triangle_t6
        gp_points, gp_weights = gauss_points_triangle_t6(3)

        for e, elem in enumerate(elements):
            elem_nodes = nodes[elem]
            fe = np.zeros(12)
            for gp, w in zip(gp_points, gp_weights):
                from fem_core import compute_B_matrix_t6, t6_jacobian
                r, s = gp
                B = compute_B_matrix_t6(elem_nodes, r, s)
                _, det_J = t6_jacobian(elem_nodes, r, s)
                if det_J <= 0:
                    det_J = abs(det_J) + 1e-14
                fe += w * det_J * thickness * (B.T @ sigma_sh)

            dof_map = []
            for node_idx in elem:
                dof_map.extend([2 * node_idx, 2 * node_idx + 1])
            for i in range(12):
                F_shrink[dof_map[i]] += fe[i]

        # 体力 + 压力 + 收缩
        F_body = compute_nodal_forces_uniform(
            nodes, elements, props["qx"], props["qy"], thickness
        )
        F_total = F_body + F_shrink

        # 顶部压力
        top_nodes = [i for i, node in enumerate(nodes) if abs(node[1] - props["Ly"]) < 1e-6]
        if len(top_nodes) > 0:
            node_force = -props["top_pressure"] * props["Lx"] * thickness / len(top_nodes)
            for idx in top_nodes:
                F_total[2 * idx + 1] += node_force

        # 边界条件
        K_bc, F_bc = apply_dirichlet_boundary(K_eff, F_total, bc_nodes, bc_values)

        # 求解
        u_t = solve_sparse_symmetric_positive_definite(K_bc, F_bc, method="direct")
        displacements[step] = u_t

        # 计算应变和应力
        strains_t, stresses_t = compute_strain_stress_at_nodes(
            nodes, elements, u_t, E_combined, nu, plane_stress=True
        )
        strains_history.append(strains_t)
        stresses_history.append(stresses_t)

    print(f"    徐变分析完成")

    # 位移增长分析
    max_disp_history = np.max(np.abs(displacements), axis=1)
    creep_ratio = max_disp_history[-1] / max_disp_history[0]
    print(f"    徐变位移放大系数: {creep_ratio:.3f}")

    return time_points, displacements, stresses_history, shrinkage_strains


def perform_uncertainty_quantification(props):
    """
    基于准蒙特卡洛的不确定性量化。
    """
    print(f"\n[8] 不确定性量化 (Quasi-Monte Carlo)")
    nodes = props["nodes"]
    E28 = props["E28"]
    nu = props["nu"]
    fcm = props["fcm"]

    n_samples = 64
    dim = 3  # E28, fcm, RH

    # 使用 Hammersley 序列采样
    samples = hammersley_sequence(0, n_samples - 1, dim, n_base=n_samples).T

    # 参数分布
    # E28 ~ N(30000, 3000^2)
    # fcm ~ N(35, 3.5^2)
    # RH ~ Uniform(50, 80)
    E28_samples = 30000.0 + 3000.0 * transform_to_gaussian(samples[:, :1])[:, 0]
    fcm_samples = 35.0 + 3.5 * transform_to_gaussian(samples[:, 1:2])[:, 0]
    RH_samples = 50.0 + 30.0 * samples[:, 2]

    max_displacements = []
    max_stresses = []

    # 简化的弹性分析（用于统计）
    thickness = props["thickness"]
    elements = props["elements"]
    bc_nodes = props["bc_nodes"]
    bc_values = props["bc_values"]

    for i in range(n_samples):
        E_s = max(E28_samples[i], 10000.0)
        K = assemble_stiffness_matrix_t6(
            nodes, elements, E_s, nu, thickness, plane_stress=True
        )
        F = compute_nodal_forces_uniform(
            nodes, elements, 0.0, -2500.0 * 9.81 * 1e-6, thickness
        )
        # 顶部压力
        top_nodes = [j for j, node in enumerate(nodes) if abs(node[1] - props["Ly"]) < 1e-6]
        if len(top_nodes) > 0:
            nf = -0.5 * props["Lx"] * thickness / len(top_nodes)
            for idx in top_nodes:
                F[2 * idx + 1] += nf

        K_bc, F_bc = apply_dirichlet_boundary(K, F, bc_nodes, bc_values)
        u = solve_sparse_symmetric_positive_definite(K_bc, F_bc, method="direct")
        _, stresses = compute_strain_stress_at_nodes(
            nodes, elements, u, E_s, nu, plane_stress=True
        )
        max_displacements.append(np.max(np.abs(u)))
        max_stresses.append(np.max(np.abs(stresses)))

    mu_d, var_d, skew_d, kurt_d = compute_moments(np.array(max_displacements))
    mu_s, var_s, skew_s, kurt_s = compute_moments(np.array(max_stresses))

    print(f"    样本数: {n_samples}")
    print(f"    最大位移统计: 均值={mu_d:.4f}, 标准差={np.sqrt(var_d):.4f}")
    print(f"    最大应力统计: 均值={mu_s:.2f}, 标准差={np.sqrt(var_s):.2f}")

    return {
        "max_disp_mean": mu_d,
        "max_disp_std": np.sqrt(var_d),
        "max_stress_mean": mu_s,
        "max_stress_std": np.sqrt(var_s),
    }


def perform_spectral_analysis():
    """
    谱方法验证：移位勒让德多项式展开。
    """
    print(f"\n[9] 谱方法验证")
    # 测试函数: f(x) = exp(x)
    def test_func(x):
        return np.exp(x)

    n_modes = 8
    coeffs = spectral_projection(test_func, n_modes, n_quad=16)
    x_test = np.linspace(0, 1, 100)
    f_approx = spectral_reconstruct(coeffs, x_test)
    f_exact = test_func(x_test)
    error = np.max(np.abs(f_approx - f_exact))
    print(f"    exp(x) 的 {n_modes} 阶谱逼近误差: {error:.2e}")

    # 积分验证
    integral_approx = coeffs[0]
    integral_exact = np.exp(1.0) - 1.0
    print(f"    积分验证: 近似={integral_approx:.6f}, 精确={integral_exact:.6f}, "
          f"误差={abs(integral_approx - integral_exact):.2e}")

    return error


def perform_polynomial_root_finding():
    """
    多项式求根验证（用于特征值分析）。
    """
    print(f"\n[10] 多项式求根验证 (WDK 算法)")
    # 测试多项式: (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
    coeffs = np.array([1.0, -10.0, 35.0, -50.0, 24.0])
    roots = wdk_roots(coeffs, tol=1e-12, max_iter=100)
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    errors = np.min(np.abs(roots[:, None] - expected[None, :]), axis=1)
    max_error = np.max(errors)
    print(f"    多项式 x^4 - 10x^3 + 35x^2 - 50x + 24 = 0")
    print(f"    计算根: {np.sort(np.real(roots))}")
    print(f"    最大误差: {max_error:.2e}")
    return max_error


def main():
    """
    主函数：零参数运行，执行完整分析流程。
    """
    start_time = time.time()

    # 步骤 1: 问题设置
    props = setup_problem()

    # 步骤 2: 瞬时弹性分析
    u0, strains_0, stresses_0, K_elastic = solve_instantaneous_elasticity(props)

    # 步骤 3: 时间相关徐变分析
    time_points, displacements, stresses_history, shrinkage_strains = \
        solve_time_dependent_creep(props, u0, K_elastic)

    # 步骤 4: 不确定性量化
    uq_results = perform_uncertainty_quantification(props)

    # 步骤 5: 谱方法验证
    spectral_error = perform_spectral_analysis()

    # 步骤 6: 多项式求根验证
    root_error = perform_polynomial_root_finding()

    # 汇总
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("分析完成摘要")
    print("=" * 70)
    print(f"总计算时间: {elapsed:.2f} 秒")
    print(f"节点数: {len(props['nodes'])}, 单元数: {len(props['elements'])}")
    print(f"徐变位移放大系数: {np.max(np.abs(displacements[-1])) / np.max(np.abs(u0)):.3f}")
    print(f"50年收缩应变: {shrinkage_strains[-1]:.6f}")
    print(f"谱逼近误差: {spectral_error:.2e}")
    print(f"多项式求根误差: {root_error:.2e}")
    print(f"位移不确定性: 均值={uq_results['max_disp_mean']:.4f}, "
          f"COV={uq_results['max_disp_std']/uq_results['max_disp_mean']:.3f}")
    print("=" * 70)
    print("所有计算成功完成，无报错。")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: machine_epsilon 返回标准双精度值 ----
eps = machine_epsilon()
assert isinstance(eps, float), '[TC01] machine_epsilon 返回值类型 FAILED'
assert 1e-16 < eps < 1e-15, '[TC01] machine_epsilon 数值范围 FAILED'

# ---- TC02: safe_divide 避免除以零 ----
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 0.0, 3.0])
result = safe_divide(a, b)
assert np.all(np.isfinite(result)), '[TC02] safe_divide 有限值 FAILED'
assert result[0] == 1.0, '[TC02] safe_divide 正常除法 FAILED'

# ---- TC03: compute_moments 已知分布统计矩 ----
samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
mu, var, skew, kurt = compute_moments(samples)
assert abs(mu - 3.0) < 1e-10, '[TC03] compute_moments 均值 FAILED'
assert abs(var - 2.5) < 1e-10, '[TC03] compute_moments 方差 FAILED'

# ---- TC04: triangle_area 直角三角形 ----
p1 = np.array([0.0, 0.0])
p2 = np.array([3.0, 0.0])
p3 = np.array([0.0, 4.0])
area = triangle_area(p1, p2, p3)
assert abs(area - 6.0) < 1e-10, '[TC04] triangle_area 直角三角形 FAILED'

# ---- TC05: generate_triangular_mesh 输出维度 ----
nodes_mesh, elements_mesh, boundary_nodes_mesh = generate_triangular_mesh(3, 3, domain=(0.0, 1.0, 0.0, 1.0))
assert nodes_mesh.shape[1] == 2, '[TC05] generate_triangular_mesh 节点维度 FAILED'
assert elements_mesh.shape[1] == 6, '[TC05] generate_triangular_mesh 单元节点数 FAILED'

# ---- TC06: mesh_quality_metrics 返回预期键 ----
metrics = mesh_quality_metrics(nodes_mesh, elements_mesh)
assert 'min_area' in metrics, '[TC06] mesh_quality_metrics 键 FAILED'
assert metrics['num_elements'] == len(elements_mesh), '[TC06] mesh_quality_metrics 单元数 FAILED'

# ---- TC07: hammersley_sequence 输出范围 ----
seq = hammersley_sequence(0, 9, 3, n_base=10)
assert np.all(seq >= 0.0), '[TC07] hammersley_sequence 下界 FAILED'
assert np.all(seq <= 1.0), '[TC07] hammersley_sequence 上界 FAILED'

# ---- TC08: halton_sequence 输出范围 ----
seq = halton_sequence(10, 3, skip=0)
assert np.all(seq >= 0.0), '[TC08] halton_sequence 下界 FAILED'
assert np.all(seq <= 1.0), '[TC08] halton_sequence 上界 FAILED'

# ---- TC09: transform_to_gaussian Box-Muller 变换 ----
np.random.seed(42)
uniform = np.random.rand(100, 2)
gaussian = transform_to_gaussian(uniform)
assert gaussian.shape[1] == 2, '[TC09] transform_to_gaussian 输出形状 FAILED'
assert abs(np.mean(gaussian)) < 0.5, '[TC09] transform_to_gaussian 均值范围 FAILED'

# ---- TC10: shifted_legendre_polynomial P0 恒为1 ----
x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
vals = shifted_legendre_polynomial(x, 0)
assert np.allclose(vals[:, 0], 1.0), '[TC10] shifted_legendre_polynomial P0 FAILED'

# ---- TC11: gauss_legendre_nodes_weights 权重和 ----
nodes_gl, weights_gl = gauss_legendre_nodes_weights(5, domain=(0.0, 1.0))
assert abs(np.sum(weights_gl) - 1.0) < 1e-14, '[TC11] gauss_legendre_nodes_weights 权重和 FAILED'

# ---- TC12: spectral_projection 常数函数积分 ----
coeffs = spectral_projection(lambda x: np.ones_like(x), 0, n_quad=4)
assert abs(coeffs[0] - 1.0) < 1e-12, '[TC12] spectral_projection 常数函数 FAILED'

# ---- TC13: thomas_algorithm 精确求解三对角系统 ----
n = 5
diag = np.ones(n) * 2.0
lower = np.ones(n - 1) * (-1.0)
upper = np.ones(n - 1) * (-1.0)
b_vec = np.ones(n)
x = thomas_algorithm(lower, diag, upper, b_vec)
Ax = diag * x
Ax[:-1] += upper * x[1:]
Ax[1:] += lower * x[:-1]
assert np.allclose(Ax, b_vec, atol=1e-12), '[TC13] thomas_algorithm 精度 FAILED'

# ---- TC14: conjugate_gradient_band 收敛到精确解 ----
x_cg = conjugate_gradient_band(lower, diag, upper, b_vec, tol=1e-12)
assert np.allclose(x_cg, x, atol=1e-8), '[TC14] conjugate_gradient_band 收敛 FAILED'

# ---- TC15: wdk_roots 已知多项式求根 ----
coeffs_poly = np.array([1.0, -10.0, 35.0, -50.0, 24.0])
roots = wdk_roots(coeffs_poly, tol=1e-12, max_iter=100)
expected = np.array([1.0, 2.0, 3.0, 4.0])
for r in expected:
    assert np.any(np.abs(roots - r) < 1e-6), f'[TC15] wdk_roots 缺少根 {r} FAILED'

# ---- TC16: newton_raphson_scalar 标量方程求根 ----
f = lambda x: x**2 - 4.0
df = lambda x: 2.0 * x
root = newton_raphson_scalar(f, df, 3.0, tol=1e-12)
assert abs(root - 2.0) < 1e-10, '[TC16] newton_raphson_scalar FAILED'

# ---- TC17: b3_compliance_function t=t_prime 时等于 q1 ----
J = b3_compliance_function(28.0, 28.0, 1.0/30000, 20e-6, 5e-6, 1.5e-6)
assert abs(J - 1.0/30000) < 1e-12, '[TC17] b3_compliance_function 初始柔量 FAILED'

# ---- TC18: mc2010_creep_coefficient t<=t0 时为零 ----
phi = mc2010_creep_coefficient(10.0, 28.0, 35.0, 65.0, 200.0)
assert phi == 0.0, '[TC18] mc2010_creep_coefficient 零值 FAILED'

# ---- TC19: mc2010_shrinkage_strain t<=ts 时为零 ----
eps_sh = mc2010_shrinkage_strain(1.0, 3.0, 35.0, 65.0, 200.0)
assert eps_sh == 0.0, '[TC19] mc2010_shrinkage_strain 零值 FAILED'

# ---- TC20: aging_elastic_modulus 正值和单调性 ----
E28 = 30000.0
E_7 = aging_elastic_modulus(7.0, E28)
E_28 = aging_elastic_modulus(28.0, E28)
E_100 = aging_elastic_modulus(100.0, E28)
assert E_7 > 0, '[TC20] aging_elastic_modulus 正值 FAILED'
assert E_7 < E_28 < E_100, '[TC20] aging_elastic_modulus 单调性 FAILED'

# ---- TC21: effective_creep_modulus 非负且小于E_t0 ----
E_eff = effective_creep_modulus(100.0, 28.0, 30000.0, 1.5)
E_t0 = aging_elastic_modulus(28.0, 30000.0)
assert E_eff > 0, '[TC21] effective_creep_modulus 正值 FAILED'
assert E_eff < E_t0, '[TC21] effective_creep_modulus 范围 FAILED'

# ---- TC22: plane_stress_constitutive_matrix 对称性 ----
D = plane_stress_constitutive_matrix(30000.0, 0.2)
assert np.allclose(D, D.T, atol=1e-12), '[TC22] plane_stress_constitutive_matrix 对称 FAILED'

# ---- TC23: apply_dirichlet_boundary 边界值正确 ----
K = np.eye(4)
F = np.ones(4)
bc_nodes = np.array([0, 2])
bc_values = np.array([0.0, 5.0])
K_mod, F_mod = apply_dirichlet_boundary(K, F, bc_nodes, bc_values)
assert K_mod[0, 0] > 1e19, '[TC23] apply_dirichlet_boundary 大数置位 FAILED'

# ---- TC24: b3_creep_coefficient 非负性 ----
phi = b3_creep_coefficient(100.0, 28.0, 30000.0, 1.0/30000, 20e-6, 5e-6, 1.5e-6)
assert phi >= 0.0, '[TC24] b3_creep_coefficient 非负 FAILED'

# ---- TC25: power_law_creep_kernel tau<=0 时为零 ----
K_val = power_law_creep_kernel(-1.0, 30000.0, 0.2, 1.0)
assert K_val == 0.0, '[TC25] power_law_creep_kernel 零值 FAILED'

# ---- TC26: effective_time_for_aging_creep 基本计算 ----
t_eff = effective_time_for_aging_creep(100.0, 28.0, alpha_h=0.5)
assert abs(t_eff - 36.0) < 1e-10, '[TC26] effective_time_for_aging_creep 计算 FAILED'

# ---- TC27: is_diagonally_dominant 严格对角占优判断 ----
A_dd = np.array([[4.0, 1.0], [1.0, 4.0]])
assert is_diagonally_dominant(A_dd, strict=True), '[TC27] is_diagonally_dominant 严格占优 FAILED'

# ---- TC28: relative_residual 精确解为零残差 ----
A = np.eye(3)
x_exact = np.array([1.0, 2.0, 3.0])
b = A @ x_exact
res = relative_residual(A, x_exact, b)
assert abs(res) < 1e-14, '[TC28] relative_residual 精确解 FAILED'

# ---- TC29: banded_lower_triangular_solve 前向替换 ----
A_band = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 0.0]])
b_vec = np.array([4.0, 6.0, 4.0])
x = banded_lower_triangular_solve(A_band, b_vec, ml=1)
assert abs(x[0] - 2.0) < 1e-12, '[TC29] banded_lower_triangular_solve x0 FAILED'
assert abs(x[1] - 2.0) < 1e-12, '[TC29] banded_lower_triangular_solve x1 FAILED'
assert abs(x[2] - 1.0) < 1e-12, '[TC29] banded_lower_triangular_solve x2 FAILED'

# ---- TC30: solve_sparse_symmetric_positive_definite 直接法 ----
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b_vec = np.array([5.0, 4.0])
x = solve_sparse_symmetric_positive_definite(A, b_vec, method="direct")
assert np.allclose(A @ x, b_vec, atol=1e-12), '[TC30] solve_sparse_symmetric_positive_definite FAILED'

print('\n全部 30 个测试通过!\n')
