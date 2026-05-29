# ---- TC01: saturation_vapor_pressure 输出为正的有限值 ----
es = saturation_vapor_pressure(300.0)
assert es > 0.0 and np.isfinite(es), '[TC01] saturation_vapor_pressure FAILED'

# ---- TC02: lambert_w(1) 满足 W*exp(W)=1 ----
w1 = lambert_w(1.0)
assert abs(w1 * np.exp(w1) - 1.0) < 1e-10, '[TC02] lambert_w FAILED'

# ---- TC03: log_gamma(5) 等于 ln(24) ----
lg5 = log_gamma(5.0)
assert abs(lg5 - np.log(24.0)) < 1e-10, '[TC03] log_gamma FAILED'

# ---- TC04: potential_temperature 随气压降低而增大 ----
theta1 = potential_temperature(300.0, 100000.0)
theta2 = potential_temperature(300.0, 50000.0)
assert theta2 > theta1, '[TC04] potential_temperature FAILED'

# ---- TC05: virtual_temperature 非负 ----
tv = virtual_temperature(280.0, np.array([0.0, 0.01, 0.02]))
assert np.all(tv >= 0.0), '[TC05] virtual_temperature FAILED'

# ---- TC06: specific_humidity_from_t_rh 边界截断 ----
q = specific_humidity_from_t_rh(350.0, 1.0, 100000.0)
assert 0.0 <= q <= 0.05, '[TC06] specific_humidity_from_t_rh FAILED'

# ---- TC07: dewpoint_from_vapor_pressure 逆一致性 ----
es = saturation_vapor_pressure(295.0)
td = dewpoint_from_vapor_pressure(es)
assert abs(td - 295.0) < 0.5, '[TC07] dewpoint_from_vapor_pressure FAILED'

# ---- TC08: saturation_adjustment 未饱和返回原值 ----
T_adj, qv_adj = saturation_adjustment(300.0, 0.005, 100000.0)
assert abs(T_adj - 300.0) < 1e-6 and abs(qv_adj - 0.005) < 1e-6, '[TC08] saturation_adjustment FAILED'

# ---- TC09: StochasticMicrophysics precipitation_rate 非负 ----
micro = StochasticMicrophysics(chaos_order=2, n_quad=8)
rate = micro.precipitation_rate(0.5e-3)
assert rate >= 0.0 and np.isfinite(rate), '[TC09] StochasticMicrophysics precipitation_rate FAILED'

# ---- TC10: EnsembleSparseGridUQ 标准差非负 ----
uq = EnsembleSparseGridUQ(dim=2, level=2)
mean, std = uq.compute_statistics(lambda p: p[0] + p[1], [(-1.0, 1.0), (-1.0, 1.0)])
assert std >= 0.0 and np.isfinite(std), '[TC10] EnsembleSparseGridUQ FAILED'

# ---- TC11: gauss_legendre_quadrature 精确积分三次多项式 ----
result = gauss_legendre_quadrature(lambda x: x**3, 0.0, 1.0, n=4)
assert abs(result - 0.25) < 1e-14, '[TC11] gauss_legendre_quadrature FAILED'

# ---- TC12: legendre_gauss_nodes_weights 权重和为 2 ----
x, w = legendre_gauss_nodes_weights(8)
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC12] legendre_gauss_nodes_weights FAILED'

# ---- TC13: gradient_2d_centered 常数场梯度为零 ----
const_field = np.ones((8, 8))
dfdx, dfdy = gradient_2d_centered(const_field, 1000.0, 1000.0)
assert np.allclose(dfdx, 0.0) and np.allclose(dfdy, 0.0), '[TC13] gradient_2d_centered FAILED'

# ---- TC14: divergence_2d 均匀场散度为零 ----
u = np.ones((8, 8)) * 5.0
v = np.ones((8, 8)) * 3.0
div = divergence_2d(u, v, 1000.0, 1000.0)
assert np.allclose(div, 0.0, atol=1e-12), '[TC14] divergence_2d FAILED'

# ---- TC15: laplacian_9point_torus 常数场 Laplacian 为零 ----
const_field = np.ones((8, 8))
lap = laplacian_9point_torus(const_field, 1000.0, 1000.0)
assert np.allclose(lap, 0.0, atol=1e-12), '[TC15] laplacian_9point_torus FAILED'

# ---- TC16: SparseMatrixCOO 矩阵向量乘法正确性 ----
A = SparseMatrixCOO(3, 3)
A.append(0, 0, 2.0)
A.append(1, 1, 3.0)
A.append(2, 2, 4.0)
x_vec = np.array([1.0, 2.0, 3.0])
y_vec = A.mv(x_vec)
assert np.allclose(y_vec, np.array([2.0, 6.0, 12.0])), '[TC16] SparseMatrixCOO.mv FAILED'

# ---- TC17: conjugate_gradient 单位矩阵精确解 ----
A = SparseMatrixCOO(3, 3)
for i in range(3): A.append(i, i, 1.0)
b = np.array([1.0, 2.0, 3.0])
x_sol, iters, res = conjugate_gradient(A, b)
assert np.allclose(x_sol, b) and res < 1e-8, '[TC17] conjugate_gradient FAILED'

# ---- TC18: solve_anelastic_pressure 输出形状正确 ----
nx, nz = 4, 3
rho = np.ones((nz, nx))
rhs = np.zeros((nz, nx))
p = solve_anelastic_pressure(nx, nz, 1000.0, 500.0, rho, rhs)
assert p.shape == (nz, nx), '[TC18] solve_anelastic_pressure FAILED'

# ---- TC19: ConvectionDynamics 积分后 U,V 在 [0,1] 内 ----
dyn = ConvectionDynamics(nx=8, ny=8, dx=2000.0)
U, V = dyn.integrate(dt=0.5, nsteps=3)
assert np.all(U >= 0.0) and np.all(U <= 1.0), '[TC19] ConvectionDynamics U range FAILED'
assert np.all(V >= 0.0) and np.all(V <= 1.0), '[TC19] ConvectionDynamics V range FAILED'

# ---- TC20: ConvectionDynamics 对流能量非负有限 ----
dyn2 = ConvectionDynamics(nx=8, ny=8, dx=2000.0)
dyn2.integrate(dt=0.5, nsteps=3)
ece = dyn2.total_convective_energy()
assert ece >= 0.0 and np.isfinite(ece), '[TC20] ConvectionDynamics energy FAILED'

# ---- TC21: hypercube_grid 输出尺寸与张量积一致 ----
grid = hypercube_grid(2, [3, 2], [(0.0, 1.0), (-1.0, 1.0)])
assert grid.shape == (6, 2), '[TC21] hypercube_grid FAILED'

# ---- TC22: nearest_interp_1d 精确命中数据点 ----
xd = np.array([1.0, 2.0, 3.0])
yd = np.array([10.0, 20.0, 30.0])
yi = nearest_interp_1d(xd, yd, np.array([2.0]))
assert abs(yi[0] - 20.0) < 1e-12, '[TC22] nearest_interp_1d FAILED'

# ---- TC23: gamma_distribution_pdf 输出非负有限 ----
D = np.linspace(0.1, 5.0, 10)
pdf = gamma_distribution_pdf(D, N0=8e6, mu=2.0, lam=3.0)
assert np.all(pdf >= 0.0) and np.all(np.isfinite(pdf)), '[TC23] gamma_distribution_pdf FAILED'

# ---- TC24: gamma_moment k=0 解析验证 ----
M0 = gamma_moment(0, N0=8e6, mu=2.0, lam=3.0)
expected = 8e6 * 2.0 / 27.0
assert abs(M0 - expected) / expected < 1e-6, '[TC24] gamma_moment FAILED'

# ---- TC25: laguerre_polynomials L_0 恒为 1 ----
x_test = np.array([0.0, 1.0, 2.0])
L = laguerre_polynomials(0, x_test)
assert np.allclose(L[0, :], 1.0), '[TC25] laguerre_polynomials FAILED'

# ---- TC26: scale_to_physical 端点映射正确 ----
pts = np.array([[-1.0], [0.0], [1.0]])
scaled = scale_to_physical(pts, [(0.0, 10.0)])
assert abs(scaled[0, 0] - 0.0) < 1e-12 and abs(scaled[2, 0] - 10.0) < 1e-12, '[TC26] scale_to_physical FAILED'

# ---- TC27: triangle_area 平面直角三角形面积 ----
v0 = np.array([0.0, 0.0, 0.0])
v1 = np.array([3.0, 0.0, 0.0])
v2 = np.array([0.0, 4.0, 0.0])
A = triangle_area(v0, v1, v2)
assert abs(A - 6.0) < 1e-12, '[TC27] triangle_area FAILED'

# ---- TC28: tetrahedron01_volume 解析值 1/6 ----
vol = tetrahedron01_volume()
assert abs(vol - 1.0 / 6.0) < 1e-12, '[TC28] tetrahedron01_volume FAILED'

# ---- TC29: surface_sensible_heat_flux 暖面向上为正 ----
H = surface_sensible_heat_flux(305.0, 298.0, 5.0)
assert H > 0.0, '[TC29] surface_sensible_heat_flux FAILED'

# ---- TC30: main 集成测试返回 0 ----
ret = main()
assert ret == 0, '[TC30] main FAILED'
