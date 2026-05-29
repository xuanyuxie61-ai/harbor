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
