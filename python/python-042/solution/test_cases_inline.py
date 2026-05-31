# ---- TC01: PiSpigot 计算 π 值在合理区间 ----
spigot = PiSpigot(digits=30)
pi_val = spigot.compute()
assert 3.1 < pi_val < 3.2, '[TC01] PiSpigot 计算 π 值不在合理区间 FAILED'

# ---- TC02: SphericalGeometry 表面积公式正确 ----
geo = SphericalGeometry(R_surf=1.0, R_cmb=0.5)
expected_area = 4.0 * pi_val * 1.0 ** 2
assert abs(geo.surface_area() - expected_area) < 1e-10, '[TC02] 表面积公式不正确 FAILED'

# ---- TC03: SphericalGeometry 球壳体积公式正确 ----
expected_vol = (4.0 / 3.0) * pi_val * (1.0 ** 3 - 0.5 ** 3)
assert abs(geo.shell_volume() - expected_vol) < 1e-10, '[TC03] 球壳体积公式不正确 FAILED'

# ---- TC04: SphericalGeometry 球坐标与直角坐标转换一致性 ----
r_test = np.array([1.0, 0.8])
theta_test = np.array([np.pi / 4, np.pi / 3])
phi_test = np.array([np.pi / 6, np.pi / 2])
x, y, z = geo.spherical_to_cartesian(r_test, theta_test, phi_test)
r_back, theta_back, phi_back = geo.cartesian_to_spherical(x, y, z)
assert np.max(np.abs(r_back - r_test)) < 1e-10, '[TC04] 坐标转换回代误差过大 FAILED'

# ---- TC05: PolygonTriangulator 简单多边形剖分数量正确 ----
triangulator = PolygonTriangulator()
angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
x_poly = 1.0 + 0.3 * np.cos(angles)
y_poly = 1.0 + 0.3 * np.sin(angles)
triangles = triangulator.triangulate(x_poly, y_poly)
assert triangles.shape == (6, 3), '[TC05] 三角剖分数量不正确 FAILED'

# ---- TC06: VoronoiTessellator 各 cell 面积和等于总区域面积 ----
voronoi = VoronoiTessellator(bounds=(0.0, 10.0, 0.0, 10.0))
generators = np.array([[3.0, 4.0], [7.0, 8.0], [7.0, 2.0], [5.0, 5.0]])
labels, areas = voronoi.compute_cells(generators, grid_res=200)
assert abs(np.sum(areas) - 100.0) < 5.0, '[TC06] Voronoi cell 面积和不等于总区域面积 FAILED'

# ---- TC07: UnicycleIndexer 循环索引产生完整序列 ----
indexer = UnicycleIndexer()
n = 12
shift = 3
u_index = indexer.create_cycle(n, shift)
sequence = indexer.index_to_sequence(n, u_index)
assert len(np.unique(sequence)) == n, '[TC07] 循环索引序列不完整 FAILED'

# ---- TC08: GramSchmidt 经典正交化 U^T U 近似单位矩阵 ----
A = np.array([[1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]], dtype=float)
U = GramSchmidt.classical(A)
UTU = U.T @ U
I = np.eye(3)
assert np.max(np.abs(UTU - I)) < 1e-10, '[TC08] Gram-Schmidt 正交化不精确 FAILED'

# ---- TC09: TrigonometricBasis 在 x=0 处取值为 1 ----
trig = TrigonometricBasis()
for k in [1, 2, 3, 4]:
    val = trig.basis(np.array([0.0]), k)
    assert abs(val[0] - 1.0) < 1e-10, '[TC09] 三角基函数在 x=0 处不为 1 FAILED'

# ---- TC10: LagrangeInterpolation 对常数函数精确重构 ----
lagrange = LagrangeInterpolation()
xd = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
yd = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
xi = np.linspace(0.0, 1.0, 21)
yi = lagrange.interpolate(xd, yd, xi)
assert np.max(np.abs(yi - 3.0)) < 1e-10, '[TC10] Lagrange 插值对常数函数不精确 FAILED'

# ---- TC11: GaussLegendre 1D 积分多项式精确 ----
gl = GaussLegendre()
integral = gl.integrate_1d(lambda t: t ** 2, -1.0, 1.0, n=16)
assert abs(integral - 2.0 / 3.0) < 1e-12, '[TC11] Gauss-Legendre 积分 x^2 不精确 FAILED'

# ---- TC12: GaussLaguerre 积分 x^2 exp(-x) 等于 2 ----
laguerre = GaussLaguerre()
integral_l = laguerre.integrate(lambda t: t ** 2, n=12, alpha=0.0, a=0.0, b=1.0)
assert abs(integral_l - 2.0) < 1e-8, '[TC12] Gauss-Laguerre 积分 x^2 exp(-x) 不精确 FAILED'

# ---- TC13: Quadrature2D 矩形积分常数函数面积正确 ----
quad2d = Quadrature2D()
val2d = quad2d.integrate_rectangle(lambda x, y: np.ones_like(x), (0.0, 2.0), (0.0, 3.0), nx=4, ny=4)
assert abs(val2d - 6.0) < 1e-12, '[TC13] 2D 矩形积分常数函数不正确 FAILED'

# ---- TC14: HypercubeSampler 固定种子可复现 ----
sampler = HypercubeSampler()
s1 = sampler.sample(m=3, n=100, seed=42)
s2 = sampler.sample(m=3, n=100, seed=42)
assert np.array_equal(s1, s2), '[TC14] 固定种子采样结果不可复现 FAILED'

# ---- TC15: ViscosityModel Arrhenius 在 T_ref 处等于 eta0 ----
viscosity = ViscosityModel()
T_ref = 1600.0
eta = viscosity.arrhenius(np.array([T_ref]))
assert abs(eta[0] - viscosity.eta0) < 1e-10, '[TC15] Arrhenius 粘度在 T_ref 处不为 eta0 FAILED'

# ---- TC16: DensityModel 浮力在 T_ref 处为零 ----
density = DensityModel()
buoy = density.buoyancy(np.array([density.T_ref]))
assert abs(buoy[0]) < 1e-10, '[TC16] 浮力在 T_ref 处不为零 FAILED'

# ---- TC17: DimensionlessNumbers Rayleigh 数正参数返回正值 ----
Ra = DimensionlessNumbers.rayleigh_number(1000.0, 1000.0)
assert Ra > 0.0, '[TC17] Rayleigh 数不为正 FAILED'

# ---- TC18: DimensionlessNumbers Nusselt 数边界处理正确 ----
Nu = DimensionlessNumbers.nusselt_number(1.2, 0.0)
assert abs(Nu - 1.0) < 1e-10, '[TC18] Nusselt 数在 q_cond=0 时边界处理不正确 FAILED'

# ---- TC19: ThermalPhysics 热产生项为有限正数 ----
physics = ThermalPhysics()
Q = physics.heat_production_term()
assert np.isfinite(Q) and Q > 0.0, '[TC19] 热产生项不是有限正数 FAILED'

# ---- TC20: GrazingChemicalExchange RK4 积分结果非负 ----
chem = GrazingChemicalExchange(a=1.1, c1=1.2, c2=1.5, d1=0.001, d2=0.001, K=3000.0, r1=0.8)
t_chem, y_chem = chem.integrate_rk4(y0=np.array([3000.0, 5.0]), t_span=(0.0, 10.0), n_steps=500)
assert np.all(y_chem >= 0.0), '[TC20] 化学交换 ODE 积分结果出现负值 FAILED'

# ---- TC21: ThermalSolver 初始温度线性模式边界条件 ----
thermal = ThermalSolver()
r_grid = np.linspace(0.5, 1.0, 10).reshape(-1, 1)
theta_grid = np.zeros_like(r_grid)
T = thermal.initial_temperature(r_grid, theta_grid, mode="linear")
assert abs(T[0, 0] - thermal.T_cmb) < 1e-10, '[TC21] 初始温度内边界不为 T_cmb FAILED'
assert abs(T[-1, 0] - thermal.T_surf) < 1e-10, '[TC21] 初始温度外边界不为 T_surf FAILED'

# ---- TC22: ChandrupatlaRootFinder 能找到 x^2-2=0 的根 ----
root_finder = ChandrupatlaRootFinder()
xm, fm, calls = root_finder.find_root(lambda x: x ** 2 - 2.0, 1.0, 2.0)
assert abs(xm - np.sqrt(2.0)) < 1e-5, '[TC22] Chandrupatla 根查找不精确 FAILED'

# ---- TC23: LissajousForcing 热流调制在合理范围内 ----
forcing = LissajousForcing(a1=2.0, b1=np.pi / 4.0, a2=3.0, b2=0.0)
q_mod = forcing.boundary_heat_flux_modulation(t=1.0, q0=1.0, amplitude=0.1)
assert 0.9 <= q_mod <= 1.1, '[TC23] Lissajous 热流调制超出合理范围 FAILED'

# ---- TC24: MantleDiagnostics 温度场统计量范围正确 ----
np.random.seed(42)
T_test = np.random.normal(loc=1600.0, scale=400.0, size=(10, 10))
T_test = np.clip(T_test, 300.0, 3000.0)
diag = MantleDiagnostics(T_min=300.0, T_max=3000.0)
stats = diag.analyze_temperature_field(T_test)
assert stats['mean'] >= stats['min'] and stats['mean'] <= stats['max'], '[TC24] 温度场均值不在 [min, max] 范围内 FAILED'

# ---- TC25: StokesSolver 速度场输出尺寸与输入温度场一致 ----
stokes = StokesSolver(R_inner=0.5, R_outer=1.0, n_radial=6, n_angular=8)
r = np.linspace(0.5, 1.0, 8)
theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
T_field = np.ones((8, 12)) * 1600.0
u_r, u_theta = stokes.compute_velocity_from_streamfunction(np.zeros((6, 8)), T_field, r_grid, theta_grid)
assert u_r.shape == T_field.shape, '[TC25] 速度场 u_r 尺寸与温度场不匹配 FAILED'
assert u_theta.shape == T_field.shape, '[TC25] 速度场 u_theta 尺寸与温度场不匹配 FAILED'
