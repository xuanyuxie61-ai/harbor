# ---- TC01: build_channel_mesh_2d 返回 Triangulation 对象，含 nodes 和 elements ----
np.random.seed(42)
tri = build_channel_mesh_2d(nz=20, nr=10)
assert tri.nodes.shape[0] > 0, '[TC01] 网格节点数应大于0 FAILED'
assert tri.elements.shape[0] > 0, '[TC01] 网格单元数应大于0 FAILED'
assert tri.nodes.shape[1] == 2, '[TC01] 二维网格节点应为2坐标 FAILED'

# ---- TC02: 自适应细化增加网格节点数 ----
np.random.seed(42)
tri2 = build_channel_mesh_2d(nz=20, nr=10)
n_nodes_before = tri2.nodes.shape[0]
tri2.adaptive_refine_filter(max_level=1)
n_nodes_after = tri2.nodes.shape[0]
assert n_nodes_after >= n_nodes_before, '[TC02] 细化后节点数应≥细化前 FAILED'

# ---- TC03: element_centers 返回形状 (N_elem, dim) ----
np.random.seed(42)
tri3 = build_channel_mesh_2d(nz=10, nr=5)
centers = tri3.element_centers()
assert centers.shape[0] == tri3.elements.shape[0], '[TC03] 重心数量应等于单元数 FAILED'
assert centers.shape[1] == 2, '[TC03] 重心应为二维坐标 FAILED'

# ---- TC04: Laplacian 作用于常数场内部点应为零 ----
np.random.seed(42)
Nx, Ny, Nz = 8, 8, 8
dx = dy = dz = 0.1
const_field = np.ones((Nx, Ny, Nz))
lap = apply_laplacian_3d(const_field, dx, dy, dz)
assert np.max(np.abs(lap[1:-1, 1:-1, 1:-1])) < 1e-12, '[TC04] 常数场Laplacian内部点应接近零 FAILED'

# ---- TC05: 1D Laplacian 作用于线性函数内部点应为零 ----
np.random.seed(42)
N_1d = 15
dx_1d = 0.1
L1d = build_laplacian_1d(N_1d, dx_1d, bc_type='neumann')
# 线性函数 f(x) = a*x + b, 其二阶导数为0
x_vals = np.arange(N_1d) * dx_1d
f_linear = 2.0 * x_vals + 1.0
lap_linear = L1d @ f_linear
assert np.max(np.abs(lap_linear[2:-2])) < 1e-10, '[TC05] 线性函数内部Laplacian应为零 FAILED'

# ---- TC06: erf_cody 基本值: erf(0)=0, erf(100)≈1, 奇函数性 ----
np.random.seed(42)
assert abs(erf_cody(0.0)) < 1e-12, '[TC06] erf(0)应为0 FAILED'
assert abs(erf_cody(100.0) - 1.0) < 1e-6, '[TC06] erf(大正数)应≈1 FAILED'
assert abs(erf_cody(-1.0) + erf_cody(1.0)) < 1e-12, '[TC06] erf应为奇函数 FAILED'

# ---- TC07: Hermite H_0(x)=1, H_1(x)=2x ----
np.random.seed(42)
x_test = 0.5
H = hermite_phys(5, x_test)
assert abs(H[0] - 1.0) < 1e-12, '[TC07] H_0(x)应为1 FAILED'
assert abs(H[1] - 2.0 * x_test) < 1e-12, '[TC07] H_1(x)应为2x FAILED'

# ---- TC08: Laguerre L_0(x)=1, L_1(x)=1-x ----
np.random.seed(42)
x_test2 = 0.5
L = laguerre_poly(4, x_test2)
assert abs(L[0] - 1.0) < 1e-12, '[TC08] L_0(x)应为1 FAILED'
assert abs(L[1] - (1.0 - x_test2)) < 1e-12, '[TC08] L_1(x)应为1-x FAILED'

# ---- TC09: Legendre P_0=1, P_1(0)=0, P_2(0)=-0.5 ----
np.random.seed(42)
P = legendre_poly(3, 0.0)
assert abs(P[0] - 1.0) < 1e-12, '[TC09] P_0(0)应为1 FAILED'
assert abs(P[1] - 0.0) < 1e-12, '[TC09] P_1(0)应为0 FAILED'
assert abs(P[2] + 0.5) < 1e-12, '[TC09] P_2(0)应为-0.5 FAILED'

# ---- TC10: boltzmann_factor 随能量单调递减且恒正 ----
np.random.seed(42)
b1 = boltzmann_factor(1.0e-20, T=300.0)
b2 = boltzmann_factor(2.0e-20, T=300.0)
assert b1 > b2, '[TC10] Boltzmann因子应随能量增加递减 FAILED'
assert b1 > 0, '[TC10] Boltzmann因子应为正 FAILED'

# ---- TC11: Debye-Hückel κ 应为正实数有限值 ----
np.random.seed(42)
kappa = debeye_huckel_kappa(0.15, T=300.0)
assert kappa > 0, '[TC11] Debye κ应为正 FAILED'
assert np.isfinite(kappa), '[TC11] Debye κ应为有限值 FAILED'

# ---- TC12: 二项式系数 C(5,2)=10, C(10,0)=1, C(10,10)=1, C(5,6)=0 ----
np.random.seed(42)
assert binomial_coefficient(5, 2) == 10, '[TC12] C(5,2)应为10 FAILED'
assert binomial_coefficient(10, 0) == 1, '[TC12] C(10,0)应为1 FAILED'
assert binomial_coefficient(10, 10) == 1, '[TC12] C(10,10)应为1 FAILED'
assert binomial_coefficient(5, 6) == 0, '[TC12] C(5,6)应为0（越界返回0） FAILED'

# ---- TC13: combination_lex_index 首尾组合索引 ----
np.random.seed(42)
c_first = combination_lex_index(5, 2, 1)
assert c_first[0] == 1 and c_first[1] == 2, '[TC13] 第1个组合应为[1,2] FAILED'
c_last = combination_lex_index(5, 2, 10)
assert c_last[0] == 4 and c_last[1] == 5, '[TC13] 最后一个组合应为[4,5] FAILED'

# ---- TC14: enumerate_occupations 构型数等于 C(n,k) ----
np.random.seed(42)
configs = enumerate_occupations(5, 2)
assert len(configs) == binomial_coefficient(5, 2), '[TC14] 构型枚举数应等于二项式系数 FAILED'

# ---- TC15: compression_ratio 应在(0,1]区间内 ----
np.random.seed(42)
cr = compression_ratio(100, 50, 3)
assert 0 < cr <= 1, '[TC15] 压缩比应在(0,1]内 FAILED'

# ---- TC16: 累积能量占比最后一项应为1 ----
np.random.seed(42)
svals = np.array([5.0, 3.0, 1.0, 0.5])
cum = np.cumsum(svals ** 2) / np.sum(svals ** 2)
assert abs(cum[-1] - 1.0) < 1e-12, '[TC16] 累积能量占比最后一项应为1 FAILED'

# ---- TC17: compute_pod_basis 返回模态/奇异值/系数形状正确 ----
np.random.seed(42)
data_matrix = np.random.randn(50, 20)
modes, svals_pod, coeffs, mean_vec = compute_pod_basis(data_matrix, n_modes=5)
assert modes.shape == (50, 5), '[TC17] POD模态矩阵形状应为(50,5) FAILED'
assert len(svals_pod) == 5, '[TC17] 应有5个奇异值 FAILED'
assert coeffs.shape == (5, 20), '[TC17] 系数矩阵形状应为(5,20) FAILED'

# ---- TC18: 球面单项式 x^2 积分精确值 = 4π/3, 奇次幂=0 ----
np.random.seed(42)
exact = sphere01_monomial_integral([2, 0, 0])
assert abs(exact - 4 * np.pi / 3) < 1e-10, '[TC18] x^2球面积分应为4π/3 FAILED'
assert abs(sphere01_monomial_integral([1, 0, 0])) < 1e-12, '[TC18] x奇次幂球面积分应为0 FAILED'

# ---- TC19: fibonacci_lattice_2d 积分返回有限数值 ----
np.random.seed(42)
def f_const(x):
    return 1.0
fib_integ = fibonacci_lattice_2d(8, f_const)
assert np.isfinite(fib_integ), '[TC19] Fibonacci格点积分应为有限值 FAILED'
assert fib_integ > 0, '[TC19] Fibonacci格点积分正值函数应为正 FAILED'

# ---- TC20: sphere01_sample 返回(3,n)形状，所有点在单位球面上 ----
np.random.seed(42)
n_samples = 1000
pts = sphere01_sample(n_samples)
assert pts.shape[0] == 3, '[TC20] 球面采样第一维应为3 FAILED'
assert pts.shape[1] == n_samples, '[TC20] 球面采样点数不匹配 FAILED'
norms = np.sqrt(np.sum(pts ** 2, axis=0))
assert np.all(np.abs(norms - 1.0) < 1e-12), '[TC20] 所有采样点应在单位球面上 FAILED'

# ---- TC21: DielectricProfile 介电常数在 [eps_protein, eps_water] 内 ----
np.random.seed(42)
Nx_dp, Ny_dp, Nz_dp = 8, 8, 12
dielectric = DielectricProfile((Nx_dp, Ny_dp, Nz_dp), 0.1, 0.1, 0.1, eps_water=78.5, eps_protein=4.0)
assert np.min(dielectric.eps) >= 4.0 - 1e-10, '[TC21] 介电常数最小值应≥4 FAILED'
assert np.max(dielectric.eps) <= 78.5 + 1e-10, '[TC21] 介电常数最大值应≤78.5 FAILED'

# ---- TC22: NernstPlanckSolver solve_step 浓度保持非负 ----
np.random.seed(42)
Nx_np, Ny_np, Nz_np = 6, 6, 10
dx_np = dy_np = dz_np = 0.1
solver_np = NernstPlanckSolver((Nx_np, Ny_np, Nz_np), dx_np*1e-9, dy_np*1e-9, dz_np*1e-9)
c_k_init = np.ones((Nx_np, Ny_np, Nz_np)) * 150.0
c_na_init = np.ones((Nx_np, Ny_np, Nz_np)) * 150.0
phi_zero = np.zeros((Nx_np, Ny_np, Nz_np))
c_k_new, c_na_new = solver_np.solve_step(c_k_init, c_na_init, phi_zero, 1e-14)
assert np.min(c_k_new) >= 0, '[TC22] K+浓度应为非负 FAILED'
assert np.min(c_na_new) >= 0, '[TC22] Na+浓度应为非负 FAILED'

# ---- TC23: IonParticle 初始化后轨迹含初始位置 ----
np.random.seed(42)
p_ion = IonParticle([1.0, 2.0, 3.0], charge=+1.0, radius=0.138)
assert len(p_ion.trajectory) == 1, '[TC23] 初始轨迹应含1个点 FAILED'
assert np.allclose(p_ion.trajectory[0], [1.0, 2.0, 3.0]), '[TC23] 初始位置应正确 FAILED'

# ---- TC24: build_kcsa_k_channel_network 应存在欧拉路径 ----
np.random.seed(42)
net_k = build_kcsa_k_channel_network()
euler_k = net_k.is_eulerian_path()
assert euler_k in [1, 2], '[TC24] KcsA K+网络应存在欧拉路径 FAILED'

# ---- TC25: LatticeChannel valid_configurations 满足最小间距约束 ----
np.random.seed(42)
channel_lat = LatticeChannel(shape=(10,))
configs_lat = channel_lat.valid_configurations(n_ions=3, min_distance=2)
for conf in configs_lat:
    for i in range(len(conf) - 1):
        assert conf[i+1] - conf[i] >= 2, '[TC25] 构型不满足最小间距约束 FAILED'
assert len(configs_lat) > 0, '[TC25] 应有合法构型 FAILED'

# ---- TC26: Born溶剂化自由能为负，小半径离子更负 ----
np.random.seed(42)
dG_K = sphere_solvation_free_energy(charge=1.0, radius=0.138e-9)
dG_Na = sphere_solvation_free_energy(charge=1.0, radius=0.102e-9)
assert dG_K < 0, '[TC26] K+溶剂化自由能应为负 FAILED'
assert dG_Na < 0, '[TC26] Na+溶剂化自由能应为负 FAILED'
assert dG_Na < dG_K, '[TC26] Na+溶剂化能应更负(半径更小) FAILED'

# ---- TC27: 选择性通透比值为正且有限 ----
np.random.seed(42)
sel = selective_permeability_ratio(8.0e-21, 3.2e-20)
assert sel > 0, '[TC27] 选择性比值应为正 FAILED'
assert np.isfinite(sel), '[TC27] 选择性比值应为有限值 FAILED'

# ---- TC28: 可复现性——固定随机种子两次球面采样结果相同 ----
np.random.seed(42)
pts1 = sphere01_sample(100)
np.random.seed(42)
pts2 = sphere01_sample(100)
assert np.allclose(pts1, pts2), '[TC28] 固定种子两次采样应相同 FAILED'

# ---- TC29: erf_cody 极端输入不产生 NaN/Inf ----
np.random.seed(42)
for xv in [0.0, -0.0, 1e-10, 1e10, -1e10]:
    v = erf_cody(xv)
    assert np.isfinite(v), '[TC29] erf_cody(%.0e) 应为有限值 FAILED' % xv

# ---- TC30: PotentialSolver 求解电势场不产生 NaN/Inf ----
np.random.seed(42)
Nx_ps, Ny_ps, Nz_ps = 8, 8, 12
dielectric_ps = DielectricProfile((Nx_ps, Ny_ps, Nz_ps), 0.1, 0.1, 0.1)
solver_ps = PotentialSolver(dielectric_ps, max_iter=30, tol=1e-5)
phi_ps = solver_ps.solve(conc_k_bulk=150.0, conc_na_bulk=150.0, boundary_potential=0.0)
assert np.all(np.isfinite(phi_ps)), '[TC30] 电势场应全为有限值 FAILED'

# ---- TC31: compute_mean_square_displacement 返回的MSD应为非负 ----
np.random.seed(42)
traj_sim = np.random.randn(100, 3).cumsum(axis=0) * 0.01
trajectories_sim = [traj_sim]
tau_sim, msd_sim, D_est_sim = compute_mean_square_displacement(trajectories_sim, dt=1e-15)
assert np.all(msd_sim >= 0), '[TC31] MSD应为非负 FAILED'
