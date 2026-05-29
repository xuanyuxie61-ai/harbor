"""
main.py
基于多尺度有限元-优化耦合的骨质疏松骨重建力学预测与骨密度分布反演模型

统一入口，零参数可运行。

执行流程：
1. 生成骨骼截面几何与T6有限元网格
2. 构建骨小梁微观结构模型，计算有效材料属性
3. 使用切比雪夫级数参数化骨密度场
4. 组装并求解线弹性有限元系统
5. 计算应变能密度场
6. 求解骨重建ODE，预测骨密度演化
7. 进行参数反演优化（非线性最小二乘）
8. 数值精度诊断与误差分析
9. 输出综合报告
"""

import numpy as np
import time
import sys

from bone_geometry import BoneGeometry, point_line_distance_signed
from microstructure_model import TrabecularMicrostructure, build_trabecular_field
from density_field import BoneDensityField, csevl, inits, dot_l2
from quadrature_engine import (gegenbauer_rule, triangle_gauss_rule,
                               tetrahedron_unit_monomial_integral,
                               monomial_value, comp_next,
                               l2_error_estimate, h1_seminorm_error_estimate)
from fem_core import (ElasticFEM2D, elastic_matrix_plane_stress,
                      r8blt_sl, r8blt_mv, t6_basis_functions,
                      t6_physical_derivatives)
from bone_remodeling_ode import BoneRemodelingODE, CoupledBoneRemodelingODE
from parameter_optimization import (golden_section_search, gradient_descent,
                                    ParameterIdentification,
                                    bone_remodeling_forward_model)
from numerical_diagnostics import (matrix_exponential_pade, matrix_exponential_taylor,
                                   compare_polynomial_evaluation,
                                   quadratic_roots_stable, quadratic_roots_standard,
                                   condition_number_analysis, NumericalDiagnostics)


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.set_printoptions(precision=6, suppress=True)
    start_time = time.time()

    # =================================================================
    # 1. 骨骼几何建模与网格生成
    # =================================================================
    print_section("STEP 1: Bone Geometry & Mesh Generation")

    geom = BoneGeometry(width=20.0, height=30.0,
                        cortical_thickness=2.5, nx=9, ny=9)
    print(f"  Nodes: {geom.node_num}")
    print(f"  Elements: {geom.element_num}")
    print(f"  Half-bandwidth: {geom.compute_half_bandwidth()}")
    print(f"  Total mesh area: {np.sum(geom.element_area):.4f} mm²")

    cortical_nodes, trabecular_nodes = geom.classify_nodes()
    print(f"  Cortical nodes: {len(cortical_nodes)}")
    print(f"  Trabecular nodes: {len(trabecular_nodes)}")

    # 测试点到直线距离
    p1 = np.array([0.0, 0.0])
    p2 = np.array([20.0, 0.0])
    p = np.array([10.0, 5.0])
    dist = point_line_distance_signed(p1, p2, p)
    print(f"  Sample signed distance to bottom edge: {dist:.4f} mm")

    # =================================================================
    # 2. 骨小梁微观结构模型
    # =================================================================
    print_section("STEP 2: Trabecular Microstructure Model")

    micro = TrabecularMicrostructure(grid_size=15, pattern_seed=42)
    report = micro.generate_microstructure_report()
    for key, val in report.items():
        print(f"  {key}: {val:.6f}" if isinstance(val, float) else f"  {key}: {val}")

    # 在整个截面上构建骨密度场
    density_field = build_trabecular_field(
        geom.node_num, 1,
        np.array([geom.is_cortical(i) for i in range(geom.node_num)]),
        seed_offset=0
    )
    print(f"  Density range: [{density_field.min():.4f}, {density_field.max():.4f}] g/cm³")

    # =================================================================
    # 3. 骨密度场切比雪夫级数表示
    # =================================================================
    print_section("STEP 3: Bone Density Field (Chebyshev Expansion)")

    cheb_field = BoneDensityField(nx_cheb=8, ny_cheb=8)
    rho_sample = cheb_field.evaluate_physical(10.0, 15.0,
                                               xlim=(0.0, 20.0), ylim=(0.0, 30.0))
    print(f"  Chebyshev density at center (10,15): {rho_sample:.6f}")

    # 测试切比雪夫级数求值
    test_coeffs = np.array([1.0, 0.5, -0.25, 0.125])
    val_csevl = csevl(0.3, test_coeffs)
    print(f"  csevl test at x=0.3: {val_csevl:.6f}")

    n_eff = inits(test_coeffs, eta=1e-12)
    print(f"  Effective Chebyshev terms: {n_eff}")

    # L2内积测试
    def f_test(x):
        return x ** 2

    def g_test(x):
        return np.sin(x)

    l2_prod = dot_l2(f_test, g_test, 0.0, 1.0)
    print(f"  L2 inner product <x², sin(x)> on [0,1]: {l2_prod:.8f}")

    # =================================================================
    # 4. 高精度求积规则
    # =================================================================
    print_section("STEP 4: High-Order Quadrature Rules")

    # Gegenbauer-Gauss 求积
    x_q, w_q = gegenbauer_rule(n=5, lambda_param=0.5, a=0.0, b=1.0)
    integral_approx = np.sum(w_q * (x_q ** 3))
    print(f"  Gegenbauer-Gauss ∫₀¹ x³ dx ≈ {integral_approx:.8f} (exact: 0.25)")

    # 三角形求积
    x_tri, y_tri, w_tri = triangle_gauss_rule(order=7)
    print(f"  Triangle quadrature nodes: {len(x_tri)}, sum weights: {np.sum(w_tri):.6f}")

    # 四面体单项式积分
    val_tet = tetrahedron_unit_monomial_integral(1, 1, 0, 0)
    print(f"  Tetrahedron ∫ x*y dV = {val_tet:.8f} (exact: 1/120 = 0.008333...)")

    # 组合生成器测试
    a = np.zeros(3, dtype=int)
    more, h, t = False, 0, 0
    combos = []
    for _ in range(10):
        a, more, h, t = comp_next(4, 3, a, more, h, t)
        combos.append(a.copy())
        if not more:
            break
    print(f"  Compositions of 4 into 3 parts: {len(combos)} generated")

    # =================================================================
    # 5. 线弹性有限元求解
    # =================================================================
    print_section("STEP 5: Linear Elastic FEM Solver")

    # 材料属性：由骨密度计算弹性模量
    E_nodes = np.zeros(geom.node_num)
    for i in range(geom.node_num):
        rho_norm = density_field[i] / 1.8  # 归一化到 [0,1]
        E_nodes[i] = cheb_field.elastic_modulus_from_density(
            rho_norm, E0=17.0e3, power=2.0)

    print(f"  Elastic modulus range: [{E_nodes.min():.2f}, {E_nodes.max():.2f}] MPa")

    fem = ElasticFEM2D(
        node_xy=geom.node_xy,
        element_node=geom.element_node,
        element_area=geom.element_area,
        E_field=E_nodes,
        nu=0.3,
        thickness=1.0
    )

    # 边界条件：底部固定，顶部施加压缩载荷
    # 找到底部和顶部边界节点
    y_coords = geom.node_xy[1, :]
    tol = 1e-6
    bottom_nodes = np.where(y_coords < tol)[0]
    top_nodes = np.where(y_coords > geom.height - tol)[0]
    bc_nodes = np.concatenate([bottom_nodes, top_nodes])

    # 顶部施加 -50 MPa 压缩应力（y方向）
    top_elements = np.array([e for e in range(geom.element_num)
                             if np.any(np.isin(geom.element_node[:, e], top_nodes))])
    traction = np.array([0.0, -50.0])  # MPa

    print(f"  Dirichlet BC nodes: {len(bc_nodes)}")
    print(f"  Top boundary elements: {len(top_elements)}")

    U = fem.solve(bc_nodes=bc_nodes,
                  traction=(top_elements, traction))

    # 提取位移
    u_x = U[0::2]
    u_y = U[1::2]
    print(f"  Max displacement u_x: {np.max(np.abs(u_x)):.6f} mm")
    print(f"  Max displacement u_y: {np.max(np.abs(u_y)):.6f} mm")

    # 应变能密度
    sed = fem.compute_strain_energy_density(U)
    print(f"  Strain energy density range: [{sed.min():.6f}, {sed.max():.6f}] MPa")

    # 节点应力
    stress = fem.compute_nodal_stress(U)
    print(f"  Max von Mises stress: {np.max(np.abs(stress)):.2f} MPa")

    # =================================================================
    # 6. 骨重建ODE求解
    # =================================================================
    print_section("STEP 6: Bone Remodeling ODE")

    # 将单元SED映射到节点（简单平均）
    node_sed = np.zeros(geom.node_num)
    node_count = np.zeros(geom.node_num)
    for e in range(geom.element_num):
        for n in geom.element_node[:, e]:
            node_sed[n] += sed[e]
            node_count[n] += 1
    node_sed /= np.maximum(node_count, 1)

    # 归一化SED
    U_max = np.max(node_sed)
    if U_max > 1e-10:
        node_sed_norm = node_sed / U_max * 2.0  # 缩放至合理范围
    else:
        node_sed_norm = np.ones_like(node_sed) * 0.5

    ode_model = BoneRemodelingODE(
        k_form=0.05, k_res=0.03, U_ref=0.8,
        rho_min=0.01, rho_max=1.8
    )

    rho0 = density_field.copy()
    t_history, rho_history = ode_model.solve_time_dependent(
        rho0=rho0,
        strain_energy_field=node_sed_norm,
        t_span=(0.0, 365.0),
        t_eval=np.linspace(0.0, 365.0, 20)
    )

    print(f"  Time steps: {len(t_history)}")
    print(f"  Initial avg density: {np.mean(rho_history[:, 0]):.4f} g/cm³")
    print(f"  Final avg density: {np.mean(rho_history[:, -1]):.4f} g/cm³")

    # 质量守恒检查
    # 节点体积（简化：用周围单元面积平均近似）
    node_volumes = np.zeros(geom.node_num)
    for e in range(geom.element_num):
        for n in geom.element_node[:, e]:
            node_volumes[n] += geom.element_area[e] / 3.0
    conserved = ode_model.check_mass_conservation(rho_history, node_volumes, tolerance=0.5)
    print(f"  Mass conservation check: {'PASS' if conserved else 'WARNING'}")

    # 精确解验证（线性简化模型）
    A_test = 0.5
    B_test = 0.1
    t_exact = np.linspace(0.0, 100.0, 50)
    rho_exact = ode_model.exact_solution_linear(t_exact, rho0=1.0, A=A_test, B=B_test)
    print(f"  Exact linear ODE test: rho(0)={rho_exact[0]:.4f}, rho(100)={rho_exact[-1]:.4f}")

    # 耦合ODE测试
    coupled = CoupledBoneRemodelingODE()
    y0 = np.array([1.0, 0.5, 0.3])
    t_coupled, y_coupled = coupled.solve(y0, lambda t: 1.0 + 0.5 * np.sin(t / 100.0))
    print(f"  Coupled ODE final state: rho={y_coupled[0, -1]:.4f}, "
          f"c_oc={y_coupled[1, -1]:.4f}, c_ob={y_coupled[2, -1]:.4f}")

    # =================================================================
    # 7. 参数反演优化
    # =================================================================
    print_section("STEP 7: Parameter Identification via Nonlinear Least Squares")

    # TODO HOLE_3: 构造参数反演流程
    # 提示：需要设置 true_params、生成测量数据、创建 ParameterIdentification 并调用 optimize
    # 注意：params 的顺序和含义必须与 bone_remodeling_ode.py 和 parameter_optimization.py 保持一致
    raise NotImplementedError("Hole 3: 请实现参数反演优化流程")


    # 黄金分割搜索测试
    def f_quad(x):
        return (x - 2.0) ** 2 + 3.0

    a, b, it, nf = golden_section_search(f_quad, 0.0, 5.0, max_iter=50)
    x_opt = (a + b) / 2.0
    print(f"  Golden section: min at x={x_opt:.6f}, f={f_quad(x_opt):.6f}, iter={it}, nfev={nf}")

    # 梯度下降测试
    def fp_quad(x):
        return 2.0 * (x - 2.0)

    x_gd, it_gd = gradient_descent(fp_quad, x0=0.0, gamma=0.1)
    print(f"  Gradient descent: min at x={x_gd:.6f}, iter={it_gd}")

    # =================================================================
    # 8. 数值精度诊断
    # =================================================================
    print_section("STEP 8: Numerical Diagnostics")

    diag = NumericalDiagnostics()

    # 矩阵指数测试
    A_test = np.array([[-0.1, 0.05],
                       [0.08, -0.12]], dtype=float)
    diag.test_matrix_exponential(A_test)

    # 多项式求值测试
    diag.test_polynomial_evaluation(degree=15)

    # 二次方程求根测试
    diag.test_quadratic_roots()

    # 刚度矩阵条件数
    K_dense = fem._assemble_stiffness().toarray()
    # 移除边界条件对应的行/列以评估内部矩阵
    internal_dofs = [i for i in range(fem.n_dofs)
                     if not (i // 2 in bc_nodes)]
    if len(internal_dofs) > 0:
        K_internal = K_dense[np.ix_(internal_dofs, internal_dofs)]
        diag.test_matrix_condition(K_internal)

    report_text = diag.generate_report()
    print(report_text)

    # =================================================================
    # 9. 综合报告输出
    # =================================================================
    print_section("STEP 9: Final Summary")

    elapsed = time.time() - start_time
    print(f"  Total execution time: {elapsed:.4f} seconds")
    print(f"  Mesh: {geom.node_num} nodes, {geom.element_num} T6 elements")
    print(f"  Density range: [{density_field.min():.4f}, {density_field.max():.4f}] g/cm³")
    print(f"  Max displacement: {np.max(np.abs(u_y)):.6f} mm")
    print(f"  Parameter estimation success: {result['success']}")
    print(f"  All modules executed successfully.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
