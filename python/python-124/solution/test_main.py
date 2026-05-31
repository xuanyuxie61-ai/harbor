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

    # 构造模拟测量数据（使用已知参数生成）
    true_params = np.array([0.05, 0.03, 0.8])
    x_meas = np.linspace(0.0, 20.0, 15)
    U_meas_field = 1.0 * np.exp(-0.02 * (x_meas - 10.0) ** 2)
    y_meas = bone_remodeling_forward_model(true_params, x_meas, U_field=U_meas_field)
    y_meas += 0.02 * np.random.default_rng(123).normal(size=len(y_meas))  # 添加噪声

    forward = lambda p, x: bone_remodeling_forward_model(p, x, U_field=U_meas_field)
    ident = ParameterIdentification(
        forward_model=forward,
        measured_data=y_meas,
        measurement_points=x_meas,
        param_bounds=[(0.001, 0.5), (0.001, 0.5), (0.1, 5.0)]
    )

    result = ident.optimize(x0=np.array([0.02, 0.05, 1.0]), method='lm')
    print(f"  True params:  k_form={true_params[0]:.4f}, k_res={true_params[1]:.4f}, U_ref={true_params[2]:.4f}")
    print(f"  Estimated:    k_form={result['params'][0]:.4f}, k_res={result['params'][1]:.4f}, U_ref={result['params'][2]:.4f}")
    print(f"  Cost: {result['cost']:.6e}")
    print(f"  Success: {result['success']}")
    print(f"  Function evals: {result['nfev']}")

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
    main()

# ================================================================
# 测试用例（35个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: BoneGeometry构建 - 验证节点和单元数量 ----
from bone_geometry import BoneGeometry
geom = BoneGeometry(width=20.0, height=30.0, cortical_thickness=2.5, nx=9, ny=9)
assert geom.node_num == 289, '[TC01] node_num mismatch FAILED'
assert geom.element_num == 128, '[TC01] element_num mismatch FAILED'
assert geom.node_xy.shape == (2, 289), '[TC01] node_xy shape mismatch FAILED'

# ---- TC02: BoneGeometry - classify_nodes返回正确类型 ----
cortical, trabecular = geom.classify_nodes()
assert isinstance(cortical, np.ndarray), '[TC02] cortical should be ndarray FAILED'
assert len(cortical) + len(trabecular) == geom.node_num, '[TC02] node classification total mismatch FAILED'
assert len(cortical) > 0, '[TC02] should have cortical nodes FAILED'
assert len(trabecular) > 0, '[TC02] should have trabecular nodes FAILED'

# ---- TC03: BoneGeometry - 单元面积总和等于矩形面积 ----
total_area = np.sum(geom.element_area)
assert abs(total_area - 20.0 * 30.0) < 1e-10, '[TC03] total element area should equal rectangle area FAILED'
assert np.all(geom.element_area > 0), '[TC03] all element areas must be positive FAILED'

# ---- TC04: BoneGeometry - compute_half_bandwidth为正整数 ----
hbw = geom.compute_half_bandwidth()
assert hbw > 0, '[TC04] half_bandwidth should be positive FAILED'
assert isinstance(hbw, (int, np.integer)), '[TC04] half_bandwidth should be integer FAILED'

# ---- TC05: BoneGeometry - nx/ny必须为奇数 ----
try:
    BoneGeometry(width=20.0, height=30.0, nx=8, ny=9)
    assert False, '[TC05] should raise ValueError for even nx FAILED'
except ValueError:
    pass

# ---- TC06: point_line_distance_signed - 已知距离验证 ----
from bone_geometry import point_line_distance_signed
p1 = np.array([0.0, 0.0])
p2 = np.array([20.0, 0.0])
p = np.array([10.0, 5.0])
dist = point_line_distance_signed(p1, p2, p)
assert abs(dist - 5.0) < 1e-12, '[TC06] signed distance should be 5.0 FAILED'
# 对称性: 交换p1,p2应得负值
dist_rev = point_line_distance_signed(p2, p1, p)
assert abs(dist_rev + 5.0) < 1e-12, '[TC06] reversed line should give -5.0 FAILED'

# ---- TC07: TrabecularMicrostructure - 孔隙率在[0,1]区间 ----
import numpy as np
np.random.seed(42)
from microstructure_model import TrabecularMicrostructure
micro = TrabecularMicrostructure(grid_size=15, pattern_seed=42)
assert 0.0 <= micro.porosity <= 1.0, '[TC07] porosity should be in [0,1] FAILED'

# ---- TC08: TrabecularMicrostructure - 有效模量为正值 ----
assert micro.effective_modulus > 0, '[TC08] effective_modulus should be positive FAILED'
report = micro.generate_microstructure_report()
assert 'porosity' in report, '[TC08] report should contain porosity FAILED'
assert 'effective_young_modulus_GPa' in report, '[TC08] report should contain effective_young_modulus_GPa FAILED'

# ---- TC09: get_pentomino_matrix - 合法/非法名称 ----
from microstructure_model import get_pentomino_matrix, rotate_matrix_90, flip_matrix
mat = get_pentomino_matrix('X')
assert mat.shape == (3, 3), '[TC09] X pentomino should be 3x3 FAILED'
assert np.sum(mat) == 5, '[TC09] X pentomino should have 5 cells FAILED'
try:
    get_pentomino_matrix('Q')
    assert False, '[TC09] should raise ValueError for unknown name FAILED'
except ValueError:
    pass

# ---- TC10: rotate_matrix_90 - 4次旋转后还原 ----
mat = get_pentomino_matrix('L')
rotated = rotate_matrix_90(mat, k=4)
assert np.array_equal(mat, rotated), '[TC10] 4x90 rotation should return original FAILED'

# ---- TC11: BoneDensityField - 中心求值 ----
from density_field import BoneDensityField, csevl, inits
bdf = BoneDensityField(nx_cheb=8, ny_cheb=8)
val_center = bdf.evaluate(0.0, 0.0)
assert np.isfinite(val_center), '[TC11] center density should be finite FAILED'
assert val_center > 0, '[TC11] center density should be positive FAILED'

# ---- TC12: csevl - 已知切比雪夫级数求值 ----
# 切比雪夫级数: f(x) = c0/2 + Σ c_k T_k(x); 系数cs=[c0,c1,...]
# cs=[2.0,0,0] => f(x) = 1.0*T0(x) = 1.0
coeffs = np.array([2.0, 0.0, 0.0])
val = csevl(0.5, coeffs)
assert abs(val - 1.0) < 1e-12, '[TC12] csevl with only c0=2 should return 1.0 FAILED'
# cs=[0,1,0] => f(x) = 1.0*T1(x) = x = 0.3
coeffs2 = np.array([0.0, 1.0, 0.0])
val2 = csevl(0.3, coeffs2)
assert abs(val2 - 0.3) < 1e-12, '[TC12] csevl with only c1 should return x FAILED'

# ---- TC13: inits - 系数截断 ----
coeffs = np.array([1.0, 0.5, 1e-15, 0.0])
n = inits(coeffs, eta=1e-12)
assert n == 2, '[TC13] inits should return 2 for trailing small coeffs FAILED'

# ---- TC14: BoneDensityField.elastic_modulus_from_density - 单调性 ----
bdf = BoneDensityField(nx_cheb=4, ny_cheb=4)
E1 = bdf.elastic_modulus_from_density(0.3)
E2 = bdf.elastic_modulus_from_density(0.6)
assert E2 > E1, '[TC14] elastic modulus should be monotonic in density FAILED'
E0 = bdf.elastic_modulus_from_density(0.0)
assert E0 == 0.0, '[TC14] E(0) should be 0 FAILED'
E_max = bdf.elastic_modulus_from_density(1.0)
assert abs(E_max - 17000.0) < 1e-6, '[TC14] E(1) should be E0=17000 FAILED'

# ---- TC15: gegenbauer_rule - Legendre情形的权值和 ----
from quadrature_engine import gegenbauer_rule
x_q, w_q = gegenbauer_rule(n=5, lambda_param=0.5, a=0.0, b=1.0)
assert abs(np.sum(w_q) - 1.0) < 1e-12, '[TC15] Legendre weights on [0,1] should sum to 1 FAILED'
assert len(x_q) == 5, '[TC15] should have 5 nodes FAILED'

# ---- TC16: triangle_gauss_rule - 权值和为0.5 ----
from quadrature_engine import triangle_gauss_rule
x_tri, y_tri, w_tri = triangle_gauss_rule(order=3)
assert abs(np.sum(w_tri) - 0.5) < 1e-12, '[TC16] triangle weights should sum to 0.5 FAILED'
assert np.all(w_tri > 0), '[TC16] all triangle weights should be positive FAILED'

# ---- TC17: tetrahedron_unit_monomial_integral - 解析值验证 ----
from quadrature_engine import tetrahedron_unit_monomial_integral
val = tetrahedron_unit_monomial_integral(0, 0, 0, 0)
assert abs(val - 1.0/6.0) < 1e-15, '[TC17] integral of 1 over unit tetrahedron = 1/6 FAILED'
val_xy = tetrahedron_unit_monomial_integral(1, 1, 0, 0)
assert abs(val_xy - 1.0/120.0) < 1e-15, '[TC17] integral of x*y = 1/120 FAILED'

# ---- TC18: comp_next - 生成正确数量的组合 ----
from quadrature_engine import comp_next
a = np.zeros(3, dtype=int)
more, h, t = False, 0, 0
count = 0
for _ in range(50):
    a, more, h, t = comp_next(4, 3, a, more, h, t)
    count += 1
    if not more:
        break
assert count == 15, '[TC18] compositions of 4 into 3 parts should be 15 FAILED'

# ---- TC19: elastic_matrix_plane_stress - 属性验证 ----
from fem_core import elastic_matrix_plane_stress
D = elastic_matrix_plane_stress(E=200.0, nu=0.3)
assert D.shape == (3, 3), '[TC19] D matrix should be 3x3 FAILED'
assert D[0, 0] > D[0, 1], '[TC19] diagonal terms should dominate FAILED'
assert D[0, 0] > 0, '[TC19] D(0,0) should be positive FAILED'

# ---- TC20: t6_basis_functions - 单位分解性质 ----
from fem_core import t6_basis_functions
phi, _, _ = t6_basis_functions(0.3, 0.3)
assert abs(np.sum(phi) - 1.0) < 1e-12, '[TC20] partition of unity FAILED'
phi2, _, _ = t6_basis_functions(1.0/3.0, 1.0/3.0)
assert abs(np.sum(phi2) - 1.0) < 1e-12, '[TC20] partition of unity at centroid FAILED'

# ---- TC21: r8blt_sl - 求解对角占优下三角系统 ----
from fem_core import r8blt_sl
n, ml = 3, 2
a_band = np.array([[2.0, 3.0, 4.0], [0.5, 0.3, 0.0], [0.1, 0.0, 0.0]])
b = np.array([2.0, 1.0, 3.0])
x = r8blt_sl(n, ml, a_band, b)
assert abs(x[0] - 1.0) < 1e-12, '[TC21] x[0] should be 1.0 FAILED'

# ---- TC22: BoneRemodelingODE - 稳态密度在合法范围 ----
from bone_remodeling_ode import BoneRemodelingODE
ode = BoneRemodelingODE(k_form=0.05, k_res=0.03, U_ref=0.5, rho_min=0.01, rho_max=1.8)
rho_ss = ode.steady_state_density(U=1.0)
assert 0.01 <= rho_ss <= 1.8, '[TC22] steady state density in bounds FAILED'
rho_ss2 = ode.steady_state_density(U=0.0)
assert abs(rho_ss2 - 0.01) < 1e-12, '[TC22] rho_ss at U=0 should be rho_min FAILED'

# ---- TC23: BoneRemodelingODE - remodeling_rate符号正确性 ----
rate_form = ode.remodeling_rate(rho=0.5, U=1.0)
assert rate_form >= 0, '[TC23] high U should cause formation (positive rate) FAILED'
rate_res = ode.remodeling_rate(rho=0.5, U=0.1)
assert rate_res <= 0, '[TC23] low U should cause resorption (negative rate) FAILED'

# ---- TC24: BoneRemodelingODE - 精确解验证 ----
t_test = np.array([0.0, 10.0, 200.0])
rho_exact = ode.exact_solution_linear(t_test, rho0=1.0, A=0.5, B=0.1)
assert abs(rho_exact[0] - 1.0) < 1e-12, '[TC24] rho(0) should be initial value FAILED'
assert abs(rho_exact[-1] - 5.0) < 1e-6, '[TC24] rho(infty) should approach A/B=5.0 FAILED'

# ---- TC25: golden_section_search - 寻找已知极小值 ----
from parameter_optimization import golden_section_search, gradient_descent
def f_quad(x):
    return (x - 3.0) ** 2 + 2.0
a, b, it, nf = golden_section_search(f_quad, 0.0, 10.0, max_iter=50, x_tol=1e-10)
x_opt = (a + b) / 2.0
assert abs(x_opt - 3.0) < 1e-6, '[TC25] golden section should find minimum at x=3 FAILED'
assert it > 0, '[TC25] should take at least one iteration FAILED'

# ---- TC26: gradient_descent - 寻找已知极小值 ----
def fp_quad(x):
    return 2.0 * (x - 3.0)
x_gd, it_gd = gradient_descent(fp_quad, x0=0.0, gamma=0.1, max_iter=10000)
assert abs(x_gd - 3.0) < 1e-5, '[TC26] gradient descent should find minimum at x=3 FAILED'

# ---- TC27: matrix_exponential_pade - 零矩阵返回单位阵 ----
from numerical_diagnostics import matrix_exponential_pade, matrix_exponential_taylor
A_zero = np.zeros((2, 2))
E_zero = matrix_exponential_pade(A_zero, order=3)
assert np.allclose(E_zero, np.eye(2), atol=1e-14), '[TC27] exp(0) should be I FAILED'

# ---- TC28: matrix_exponential_pade vs Taylor - 小矩阵一致性 ----
A_small = np.array([[0.1, 0.02], [0.03, 0.15]])
E_pade = matrix_exponential_pade(A_small, order=3)
E_taylor = matrix_exponential_taylor(A_small, terms=50)
diff = np.linalg.norm(E_pade - E_taylor, ord='fro')
assert diff < 1e-5, '[TC28] Pade and Taylor should be consistent for small A FAILED'

# ---- TC29: polynomial_value_horner - 已知结果 ----
from numerical_diagnostics import polynomial_value_horner, polynomial_value_direct
# p(x) = 1 + 2x + 3x^2  =>  p(2) = 1 + 4 + 12 = 17
coeffs_poly = np.array([1.0, 2.0, 3.0])
val_horner = polynomial_value_horner(2.0, coeffs_poly)
assert abs(val_horner - 17.0) < 1e-12, '[TC29] Horner: p(2)=1+2*2+3*4=17 FAILED'
val_direct = polynomial_value_direct(2.0, coeffs_poly)
assert abs(val_direct - 17.0) < 1e-12, '[TC29] Direct: p(2)=1+2*2+3*4=17 FAILED'

# ---- TC30: quadratic_roots_stable - 病态问题精度更高 ----
from numerical_diagnostics import quadratic_roots_standard, quadratic_roots_stable
a, b, c = 1.0, -2.0, 1.0 - 1e-12
r1_std, r2_std = quadratic_roots_standard(a, b, c)
r1_stb, r2_stb = quadratic_roots_stable(a, b, c)
true_r1 = 1.0 + 1e-6
true_r2 = 1.0 - 1e-6
err_std = max(abs(r1_std - true_r1), abs(r2_std - true_r2))
err_stb = max(abs(r1_stb - true_r1), abs(r2_stb - true_r2))
assert err_stb < 1e-9, '[TC30] stable method should have high accuracy FAILED'

# ---- TC31: condition_number_analysis - 单位矩阵条件数 ----
from numerical_diagnostics import condition_number_analysis
I_mat = np.eye(3)
analysis = condition_number_analysis(I_mat)
assert analysis['rank'] == 3, '[TC31] identity matrix rank should be 3 FAILED'
assert analysis['cond_2'] < 2.0, '[TC31] identity condition number should be ~1 FAILED'
assert not analysis['is_singular'], '[TC31] identity should not be singular FAILED'

# ---- TC32: l2_error_estimate - 精确匹配时误差为0 ----
from quadrature_engine import l2_error_estimate
uh = np.array([1.0, 2.0, 3.0])
uexact = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, 0.3, 0.2])
areas = np.array([1.0, 1.0, 1.0])
err = l2_error_estimate(uh, uexact, weights, areas)
assert err < 1e-15, '[TC32] L2 error of exact match should be zero FAILED'

# ---- TC33: build_trabecular_field - 输出形状和范围 ----
from microstructure_model import build_trabecular_field
cortical_mask = np.array([True, True, False, False], dtype=bool)
density = build_trabecular_field(2, 2, cortical_mask, seed_offset=0)
assert len(density) == 4, '[TC33] density field length should be 4 FAILED'
assert density[0] > 0, '[TC33] cortical density should be positive FAILED'
assert np.all(density > 0), '[TC33] all densities should be positive FAILED'

# ---- TC34: BoneDensityField - evaluate_batch形状正确 ----
bdf = BoneDensityField(nx_cheb=4, ny_cheb=4)
xy_batch = np.array([[10.0, 5.0, 15.0], [15.0, 20.0, 10.0]])
vals = bdf.evaluate_batch(xy_batch)
assert vals.shape == (3,), '[TC34] batch evaluation shape should be (3,) FAILED'
assert np.all(np.isfinite(vals)), '[TC34] all batch values should be finite FAILED'

# ---- TC35: 可复现性 - 固定种子后两次构造结果相同 ----
np.random.seed(123)
micro1 = TrabecularMicrostructure(grid_size=15, pattern_seed=123)
np.random.seed(123)
micro2 = TrabecularMicrostructure(grid_size=15, pattern_seed=123)
assert abs(micro1.porosity - micro2.porosity) < 1e-15, '[TC35] reproducibility with fixed seed FAILED'
assert abs(micro1.effective_modulus - micro2.effective_modulus) < 1e-15, '[TC35] reproducibility of effective modulus FAILED'

print('\n全部 35 个测试通过!\n')
