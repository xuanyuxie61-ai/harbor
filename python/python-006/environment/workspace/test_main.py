"""
main.py
中子星物态方程与致密物质多物理场数值模拟统一入口

本项目围绕天体物理前沿问题——中子星物态方程(EOS)与致密物质结构——
融合15个种子项目的核心算法，构建博士级科学计算平台。

运行方式:
    python main.py

无需任何命令行参数。
"""

import numpy as np
import math
import time

from utils_physics import (
    init_physics_system, generate_prime_seed, rot13_encode,
    geometric_units, safe_divide, fermi_momentum_to_density
)
from eos_legendre import (
    associated_legendre_polynomial_value,
    SkyrmeEOS, PolytropicEOS, build_composite_eos,
    legendre_angular_expansion
)
from tov_solver import (
    TOVIntegrator, compute_mass_radius_relation, compute_tidal_deformability,
    build_eps_of_P_simplified
)
from crust_integrals import (
    triangle_area, triangle_unit_monomial_integral,
    quadrilateral_witherden_rule, quadrilateral_unit_monomial_integral,
    wedge01_monomial_integral, pasta_free_energy_triangle_lattice,
    pasta_free_energy_quadrilateral_sheet, pasta_free_energy_wedge_cylinder
)
from lattice_grid import (
    polygon_grid_count, polygon_grid_points,
    generate_crust_lattice_hexagonal,
    compute_crust_shear_modulus,
    r8mat_solve
)
from neutrino_diffusion import (
    solve_neutrino_diffusion_1d,
    compute_neutrino_luminosity,
    compute_deleptonization_timescale
)
from matrix_stability import (
    r8lt_det, r8lt_inverse,
    hilbert_matrix, hilbert_inverse,
    matrix_condition_number_1d,
    test_numerical_stability_on_hilbert,
    analyze_eigenvalue_stability
)
from quadrature_verify import (
    hermite_integral, hermite_quadrature_exactness,
    gauss_hermite_nodes_weights, gauss_laguerre_nodes_weights,
    verify_eos_integrals
)


def run_phase_1_eos_and_legendre():
    """
    阶段1: 物态方程与Legendre多项式展开
    测试 Skyrme EOS 和 Legendre 角向展开。
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Equation of State & Legendre Polynomial Expansion")
    print("=" * 70)

    eos = SkyrmeEOS()

    # 测试不同密度下的状态方程
    rho_vals = np.linspace(0.05, 0.8, 10)  # fm^-3
    print(f"\n{'rho (fm^-3)':>12} {'eps (MeV/fm^3)':>18} {'P (MeV/fm^3)':>18} {'cs^2/c^2':>12}")
    print("-" * 65)
    for rho in rho_vals:
        eps = eos.energy_density(rho)
        P = eos.pressure(rho)
        cs2 = eos.sound_speed_squared(rho)
        print(f"{rho:12.4f} {eps:18.4f} {P:18.4f} {cs2:12.6f}")

    # Legendre角向展开测试
    print("\n--- Legendre Angular Expansion Test ---")
    coeffs = np.array([1.0, 0.5, -0.3, 0.1])
    cos_theta = np.linspace(-1.0, 1.0, 21)
    V_vals = legendre_angular_expansion(coeffs, cos_theta)
    print(f"Potential V(cosθ) at θ=0: {V_vals[0]:.6f} MeV")
    print(f"Potential V(cosθ) at θ=π: {V_vals[-1]:.6f} MeV")

    # 化学势
    mu_n, mu_p = eos.chemical_potential(0.16, delta=0.2)
    print(f"\nChemical potentials at saturation (δ=0.2):")
    print(f"  μ_n = {mu_n:.4f} MeV")
    print(f"  μ_p = {mu_p:.4f} MeV")

    return eos


def run_phase_2_tov_structure(eos):
    """
    阶段2: TOV方程求解与质量-半径关系
    """
    print("\n" + "=" * 70)
    print("PHASE 2: TOV Equation & Mass-Radius Relation")
    print("=" * 70)

    # 构建物态方程 ε(P) 关系
    eps_of_P = build_eps_of_P_simplified(Gamma=2.5)

    # 参数扫描
    print("\nComputing mass-radius relation via parameter sweep...")
    Pc_min = 50.0    # MeV/fm^3
    Pc_max = 800.0   # MeV/fm^3
    mr_data = compute_mass_radius_relation(eps_of_P, Pc_min_MeV=Pc_min,
                                            Pc_max_MeV=Pc_max, n_points=12)

    print(f"\n{'Pc (MeV/fm^3)':>16} {'R (km)':>12} {'M (M_sun)':>12} {'M/R (geom)':>14}")
    print("-" * 58)
    for i in range(len(mr_data['Pc'])):
        Pc = mr_data['Pc'][i]
        R_km = mr_data['R_km'][i]
        M_sun = mr_data['M_sun'][i]
        MR = mr_data['M_over_R'][i]
        if not (np.isnan(R_km) or np.isnan(M_sun)):
            print(f"{Pc:16.2f} {R_km:12.3f} {M_sun:12.4f} {MR:14.6f}")

    # 潮汐形变参数
    print("\n--- Tidal Deformability ---")
    Pc_test = 200.0  # MeV/fm^3
    Lambda = compute_tidal_deformability(eps_of_P, Pc_test)
    if not np.isnan(Lambda):
        print(f"Tidal deformability Λ at Pc={Pc_test:.2f} MeV/fm^3: {Lambda:.4e}")
    else:
        print("Tidal deformability computation failed for this central pressure.")

    return mr_data


def run_phase_3_crust_integrals():
    """
    阶段3: 地壳核pasta相的多区域积分
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Crust Nuclear Pasta Phase Multi-Region Integration")
    print("=" * 70)

    # 三角形区域积分验证
    print("\n--- Triangle Monomial Integral Exactness ---")
    test_expons = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for expon in test_expons:
        exact = triangle_unit_monomial_integral(expon)
        print(f"  x^{expon[0]} y^{expon[1]}: exact = {exact:.10e}")

    # 四边形积分规则
    print("\n--- Quadrilateral Witherden Rule Verification ---")
    for p in [1, 3, 5]:
        n, x, y, w = quadrilateral_witherden_rule(p)
        # 验证 ∫∫ x^p y^0 dx dy 的精度
        max_err = 0.0
        for px in range(p + 1):
            for py in range(p + 1 - px):
                exact = quadrilateral_unit_monomial_integral((px, py))
                quad = np.sum(w * (x**px) * (y**py))
                err = abs(quad - exact)
                max_err = max(max_err, err)
        print(f"  Order p={p:2d}: n={n:2d} points, max monomial error = {max_err:.2e}")

    # 楔形积分
    print("\n--- Wedge Monomial Integral ---")
    test_wedge = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 2), (1, 1, 2)]
    for e in test_wedge:
        val = wedge01_monomial_integral(e)
        print(f"  x^{e[0]} y^{e[1]} z^{e[2]}: integral = {val:.10e}")

    # 核pasta相自由能
    print("\n--- Nuclear Pasta Free Energy Estimates ---")
    F_gnocchi = pasta_free_energy_triangle_lattice(
        lattice_spacing=50.0, surface_tension=1.0, nuclear_radius=8.0
    )
    F_lasagna = pasta_free_energy_quadrilateral_sheet(
        sheet_width=10.0, sheet_spacing=50.0, surface_tension=1.0
    )
    F_spaghetti = pasta_free_energy_wedge_cylinder(
        cylinder_radius=8.0, cell_size=50.0, surface_tension=1.0
    )
    print(f"  Gnocchi phase (triangle lattice):  {F_gnocchi:.4f} MeV/fm^2")
    print(f"  Lasagna phase (quad sheet):        {F_lasagna:.4f} MeV/fm^2")
    print(f"  Spaghetti phase (wedge cylinder):  {F_spaghetti:.4f} MeV/fm^3")


def run_phase_4_lattice_grid():
    """
    阶段4: 地壳晶格网格生成与弹性分析
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Crust Lattice Grid Generation & Elasticity")
    print("=" * 70)

    # 六边形晶格生成
    print("\n--- Hexagonal Lattice Generation ---")
    nodes, elements = generate_crust_lattice_hexagonal(
        lattice_constant=30.0, n_layers=3
    )
    print(f"  Generated {nodes.shape[0]} nodes and {len(elements)} triangular elements.")

    # 多边形网格
    print("\n--- Polygon Grid (Wigner-Seitz Cell) ---")
    n_sub = 4
    nv = 6
    v = np.array([
        [0.0, 0.0],
        [30.0, 0.0],
        [45.0, 25.98],
        [30.0, 51.96],
        [0.0, 51.96],
        [-15.0, 25.98]
    ])
    ng = polygon_grid_count(n_sub, nv)
    xg = polygon_grid_points(n_sub, nv, v, ng)
    print(f"  Polygon grid: {n_sub} subdivisions -> {ng} points")
    print(f"  Centroid at: ({xg[0, 0]:.2f}, {xg[0, 1]:.2f}) fm")

    # 剪切模量
    shear = compute_crust_shear_modulus(nodes, elements, young_modulus=1.0e35)
    print(f"\n  Estimated crust shear modulus: {shear:.4e} Pa")

    # Gauss-Jordan求解测试
    print("\n--- Linear System Test (Gauss-Jordan) ---")
    n_test = 5
    A_test = np.eye(n_test) + 0.1 * np.random.rand(n_test, n_test)
    b_test = np.ones(n_test)
    aug = np.hstack([A_test, b_test.reshape(-1, 1)])
    sol, info = r8mat_solve(n_test, 1, aug)
    if info == 0:
        x_sol = sol[:, -1]
        residual = np.linalg.norm(A_test @ x_sol - b_test)
        print(f"  Gauss-Jordan solve residual: {residual:.2e}")
    else:
        print(f"  Gauss-Jordan failed at step {info}")


def run_phase_5_neutrino_diffusion():
    """
    阶段5: 中微子扩散与冷却
    """
    print("\n" + "=" * 70)
    print("PHASE 5: Neutrino Diffusion & Cooling")
    print("=" * 70)

    print("\nSolving 1D neutrino diffusion-convection-reaction equations...")
    x, t, sol = solve_neutrino_diffusion_1d(
        nx=100, nt=2000, t_final=5.0, L=5.0e4
    )

    # 初始和最终状态
    Y_e_initial = sol[0, :, 0]
    Y_e_final = sol[-1, :, 0]
    T_initial = sol[0, :, 1]
    T_final = sol[-1, :, 1]

    print(f"\n  Initial Y_e range: [{Y_e_initial.min():.4f}, {Y_e_initial.max():.4f}]")
    print(f"  Final Y_e range:   [{Y_e_final.min():.4f}, {Y_e_final.max():.4f}]")
    print(f"  Initial T range:   [{T_initial.min():.4e}, {T_initial.max():.4e}] K")
    print(f"  Final T range:     [{T_final.min():.4e}, {T_final.max():.4e}] K")

    # 中微子光度
    dx = x[1] - x[0]
    L_nu = compute_neutrino_luminosity(T_final, dx, R_star=1.0e6)
    print(f"\n  Neutrino luminosity: {L_nu:.4e} erg/s")

    # 电子丰度弛豫时标
    tau = compute_deleptonization_timescale(
        Y_e_initial=0.3, Y_e_final=0.1, weak_rate=1.0e-2
    )
    print(f"  Deleptonization timescale: {tau:.2f} s")


def run_phase_6_matrix_stability():
    """
    阶段6: 矩阵稳定性分析
    """
    print("\n" + "=" * 70)
    print("PHASE 6: Matrix Stability & Ill-Conditioned System Analysis")
    print("=" * 70)

    # 下三角矩阵行列式
    print("\n--- Lower Triangular Matrix Determinant ---")
    n_lt = 8
    L_mat = np.tril(np.random.rand(n_lt, n_lt) + 0.5)
    det_lt = r8lt_det(n_lt, L_mat)
    det_np = np.linalg.det(L_mat)
    print(f"  Custom r8lt_det:   {det_lt:.10e}")
    print(f"  NumPy det:         {det_np:.10e}")
    print(f"  Relative diff:     {abs(det_lt - det_np) / abs(det_np):.2e}")

    # 下三角矩阵求逆
    L_inv = r8lt_inverse(n_lt, L_mat)
    identity_check = L_mat @ L_inv
    off_diag_max = np.max(np.abs(identity_check - np.eye(n_lt)))
    print(f"  Inverse accuracy (off-diag max): {off_diag_max:.2e}")

    # Hilbert矩阵稳定性测试
    print("\n--- Hilbert Matrix Numerical Stability ---")
    stability_results = test_numerical_stability_on_hilbert(n_max=8)
    print(f"{'n':>4} {'cond(H)':>16} {'||x_num - x_ex||':>20}")
    print("-" * 45)
    for n, res in stability_results.items():
        print(f"{n:4d} {res['condition_number']:16.4e} {res['error']:20.4e}")

    # 特征值稳定性分析
    print("\n--- Eigenvalue Stability Test ---")
    A_stable = np.array([[-2.0, 1.0], [0.5, -1.5]])
    eigs, is_stable = analyze_eigenvalue_stability(A_stable)
    print(f"  Test matrix eigenvalues: {eigs[0]:.4f}, {eigs[1]:.4f}")
    print(f"  System is {'STABLE' if is_stable else 'UNSTABLE'}")


def run_phase_7_quadrature_verification():
    """
    阶段7: 积分规则精度验证
    """
    print("\n" + "=" * 70)
    print("PHASE 7: Quadrature Rule Exactness Verification")
    print("=" * 70)

    # Gauss-Hermite节点生成与验证
    print("\n--- Gauss-Hermite Quadrature Verification ---")
    for n in [4, 8, 16]:
        x, w = gauss_hermite_nodes_weights(n)
        errors = hermite_quadrature_exactness(n, x, w, p_max=2 * n - 1)
        max_err = np.max(errors)
        print(f"  n={n:2d}: max relative error for polynomials up to degree {2*n-1} = {max_err:.2e}")

    # Gauss-Laguerre
    print("\n--- Gauss-Laguerre Quadrature Verification ---")
    for n in [4, 8]:
        x, w = gauss_laguerre_nodes_weights(n)
        # 快速验证: p=0 时 ∫ e^{-x} dx = 1
        quad_0 = np.sum(w)
        print(f"  n={n:2d}: ∫e^{{-x}}dx via quadrature = {quad_0:.15f} (exact=1.0)")

    # 物态方程关键积分验证
    print("\n--- EOS Critical Integral Verification ---")
    verify_res = verify_eos_integrals()
    for key, val in verify_res.items():
        print(f"  {key}: {val:.4e}")


def run_phase_8_synthesis_summary():
    """
    阶段8: 合成总结与编码输出
    """
    print("\n" + "=" * 70)
    print("PHASE 8: Synthesis Summary & Encoded Output")
    print("=" * 70)

    seed = generate_prime_seed(lower=1000, upper=5000)
    print(f"\n  Generated prime random seed: {seed}")

    summary = (
        "NeutronStarEOSDenseMatterSynthesisCompletedSuccessfully"
    )
    encoded = rot13_encode(summary)
    print(f"  ROT13 encoded status: {encoded}")
    print(f"  Decoded verification: {rot13_encode(encoded)}")

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED SUCCESSFULLY")
    print("=" * 70)


def main():
    """
    统一入口函数：零参数可运行。
    依次执行8个阶段的科学计算流程。
    """
    t_start = time.time()
    init_physics_system()

    # 阶段1: 物态方程
    eos = run_phase_1_eos_and_legendre()

    # 阶段2: TOV结构
    mr_data = run_phase_2_tov_structure(eos)

    # 阶段3: 地壳积分
    run_phase_3_crust_integrals()

    # 阶段4: 晶格网格
    run_phase_4_lattice_grid()

    # 阶段5: 中微子扩散
    run_phase_5_neutrino_diffusion()

    # 阶段6: 矩阵稳定性
    run_phase_6_matrix_stability()

    # 阶段7: 积分验证
    run_phase_7_quadrature_verification()

    # 阶段8: 总结
    run_phase_8_synthesis_summary()

    t_elapsed = time.time() - t_start
    print(f"\nTotal execution time: {t_elapsed:.3f} seconds\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: rot13_encode 自反性验证 ----
encoded = rot13_encode("HelloWorld")
decoded = rot13_encode(encoded)
assert decoded == "HelloWorld", '[TC01] rot13_encode 自反性验证 FAILED'

# ---- TC02: safe_divide 零除保护返回默认值 ----
result = safe_divide(5.0, 0.0, default=42.0)
assert result == 42.0, '[TC02] safe_divide 零除保护 FAILED'

# ---- TC03: fermi_momentum_to_density kf=0 返回 0 ----
n_val = fermi_momentum_to_density(0.0)
assert abs(n_val) < 1e-14, '[TC03] fermi_momentum_to_density 边界 FAILED'

# ---- TC04: geometric_units 质量为零时返回 0 ----
m_geom = geometric_units(0.0)
assert abs(m_geom) < 1e-14, '[TC04] geometric_units 零质量边界 FAILED'

# ---- TC05: associated_legendre_polynomial_value P_0^0(1) = 1 ----
val = associated_legendre_polynomial_value(1, 0, 0, np.array([1.0]))
assert abs(val[0, 0] - 1.0) < 1e-14, '[TC05] Legendre P_0^0 FAILED'

# ---- TC06: SkyrmeEOS energy_density 饱和密度处为有限正值 ----
eos = SkyrmeEOS()
eps = eos.energy_density(0.16)
assert eps > 0.0 and np.isfinite(eps), '[TC06] SkyrmeEOS energy_density FAILED'

# ---- TC07: SkyrmeEOS pressure 零密度返回 0 ----
eos = SkyrmeEOS()
P = eos.pressure(0.0)
assert abs(P) < 1e-14, '[TC07] SkyrmeEOS pressure 边界 FAILED'

# ---- TC08: legendre_angular_expansion 常数系数输出维度匹配 ----
coeffs = np.array([2.0])
cos_theta = np.linspace(-1.0, 1.0, 5)
V = legendre_angular_expansion(coeffs, cos_theta)
assert V.shape == cos_theta.shape and np.allclose(V, 2.0), '[TC08] Legendre 常数展开 FAILED'

# ---- TC09: build_eps_of_P_simplified 单调性验证 ----
eps_of_P = build_eps_of_P_simplified(Gamma=2.5, eps0_MeV=50.0)
eps_low = eps_of_P(1.0)
eps_high = eps_of_P(10.0)
assert eps_high > eps_low, '[TC09] eps_of_P 单调性 FAILED'

# ---- TC10: TOVIntegrator integrate 输出包含必要键且半径为正 ----
eps_of_P = build_eps_of_P_simplified(Gamma=2.5, eps0_MeV=50.0)
integrator = TOVIntegrator(eps_of_P, n_steps=20000)
result = integrator.integrate(100.0)
assert 'radius_km' in result and 'mass_Msun' in result and result['radius_km'] > 0.0, '[TC10] TOVIntegrator 结构 FAILED'

# ---- TC11: compute_tidal_deformability 返回非负有限值 ----
eps_of_P = build_eps_of_P_simplified(Gamma=2.5, eps0_MeV=50.0)
Lambda = compute_tidal_deformability(eps_of_P, 100.0)
assert Lambda >= 0.0 and np.isfinite(Lambda), '[TC11] tidal_deformability 范围 FAILED'

# ---- TC12: triangle_unit_monomial_integral (0,0) 精确值 0.5 ----
val = triangle_unit_monomial_integral((0, 0))
assert abs(val - 0.5) < 1e-14, '[TC12] 三角形单项式积分 FAILED'

# ---- TC13: quadrilateral_unit_monomial_integral (1,1) 精确值 0.25 ----
val = quadrilateral_unit_monomial_integral((1, 1))
assert abs(val - 0.25) < 1e-14, '[TC13] 四边形单项式积分 FAILED'

# ---- TC14: wedge01_monomial_integral 奇数 ez 返回 0 ----
val = wedge01_monomial_integral((1, 1, 1))
assert abs(val) < 1e-14, '[TC14] 楔形积分奇偶性 FAILED'

# ---- TC15: triangle_area 直角三角形面积验证 ----
t = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
area = triangle_area(t)
assert abs(area - 6.0) < 1e-14, '[TC15] 三角形面积 FAILED'

# ---- TC16: polygon_grid_count 公式验证 n=2,nv=4 ----
ng = polygon_grid_count(2, 4)
assert ng == 1 + 4 * 2 * 3 // 2, '[TC16] 多边形网格计数 FAILED'

# ---- TC17: r8mat_solve 单位矩阵精确求解 ----
n = 3
aug = np.hstack([np.eye(n), np.array([[1.0], [2.0], [3.0]])])
sol, info = r8mat_solve(n, 1, aug)
assert info == 0 and np.allclose(sol[:, -1], [1.0, 2.0, 3.0]), '[TC17] Gauss-Jordan 求解 FAILED'

# ---- TC18: compute_deleptonization_timescale 解析公式验证 ----
tau = compute_deleptonization_timescale(0.3, 0.1, weak_rate=1.0e-2)
expected = abs(0.1 - 0.3) / (1.0e-2 * 0.3)
assert abs(tau - expected) < 1e-10, '[TC18] 退轻子时标 FAILED'

# ---- TC19: r8lt_det 下三角矩阵行列式等于对角线乘积 ----
np.random.seed(42)
L = np.tril(np.random.rand(4, 4) + 0.5)
det_custom = r8lt_det(4, L)
det_np = np.linalg.det(L)
assert abs(det_custom - det_np) < 1e-10, '[TC19] 下三角行列式 FAILED'

# ---- TC20: hilbert_matrix 与 hilbert_inverse 乘积近似单位矩阵 ----
H = hilbert_matrix(4)
H_inv = hilbert_inverse(4)
product = H @ H_inv
assert np.allclose(product, np.eye(4), atol=1e-10), '[TC20] Hilbert 逆矩阵 FAILED'

# ---- TC21: analyze_eigenvalue_stability 负定对角矩阵判定稳定 ----
A = np.diag([-1.0, -2.0, -3.0])
eigs, is_stable = analyze_eigenvalue_stability(A)
assert is_stable and np.all(np.real(eigs) < 0), '[TC21] 特征值稳定性 FAILED'

# ---- TC22: hermite_integral p=0 精确值 sqrt(pi) ----
val = hermite_integral(0)
assert abs(val - math.sqrt(math.pi)) < 1e-14, '[TC22] Hermite 积分 FAILED'

# ---- TC23: gauss_hermite_nodes_weights 权重和等于 sqrt(pi) ----
x, w = gauss_hermite_nodes_weights(8)
assert abs(np.sum(w) - math.sqrt(math.pi)) < 1e-14, '[TC23] Gauss-Hermite 权重和 FAILED'

# ---- TC24: gauss_laguerre_nodes_weights 对 p=0 积分精确为 1 ----
x, w = gauss_laguerre_nodes_weights(6)
quad_0 = np.sum(w)
assert abs(quad_0 - 1.0) < 1e-14, '[TC24] Gauss-Laguerre 零阶积分 FAILED'

# ---- TC25: PolytropicEOS pressure_from_density 解析公式验证 ----
poly = PolytropicEOS(K=0.05, Gamma=2.5)
P = poly.pressure_from_density(2.0)
expected = 0.05 * 2.0**2.5
assert abs(P - expected) < 1e-14, '[TC25] 多方物态方程 FAILED'

# ---- TC26: SkyrmeEOS chemical_potential 返回二元组 ----
eos = SkyrmeEOS()
mu_n, mu_p = eos.chemical_potential(0.16, delta=0.2)
assert isinstance(mu_n, float) and isinstance(mu_p, float) and mu_n != mu_p, '[TC26] 化学势返回值 FAILED'

# ---- TC27: compute_crust_shear_modulus 返回正值 ----
nodes, elements = generate_crust_lattice_hexagonal(30.0, 2)
shear = compute_crust_shear_modulus(nodes, elements, young_modulus=1.0e35)
assert shear > 0.0 and np.isfinite(shear), '[TC27] 剪切模量正定性 FAILED'

# ---- TC28: matrix_condition_number_1d 单位矩阵条件数为 1 ----
cond = matrix_condition_number_1d(np.eye(5))
assert abs(cond - 1.0) < 1e-10, '[TC28] 单位矩阵条件数 FAILED'

# ---- TC29: legendre_angular_expansion 奇次项在 x=0 处对称性 ----
coeffs = np.array([0.0, 1.0])
cos_theta = np.array([-1.0, 0.0, 1.0])
V = legendre_angular_expansion(coeffs, cos_theta)
assert abs(V[0] + V[2]) < 1e-14, '[TC29] Legendre 奇函数对称性 FAILED'

# ---- TC30: pasta_free_energy_triangle_lattice 正值参数返回有限值 ----
F = pasta_free_energy_triangle_lattice(50.0, 1.0, 8.0)
assert np.isfinite(F), '[TC30] pasta自由能 鲁棒性 FAILED'

print('\n全部 30 个测试通过!\n')
