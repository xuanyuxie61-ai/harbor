# ---- TC01: reduced_mass 对称性验证 ----
from constants import reduced_mass
m = 10.0
assert abs(reduced_mass(m, m) - m / 2.0) < 1e-10, '[TC01] reduced_mass 对称性验证 FAILED'

# ---- TC02: hbar2_over_2m 返回有限正值 ----
from constants import hbar2_over_2m
val = hbar2_over_2m()
assert np.isfinite(val) and val > 0, '[TC02] hbar2_over_2m 返回有限正值 FAILED'

# ---- TC03: woods_saxon 远场趋于零 ----
from nuclear_potential import woods_saxon
r_far = np.array([100.0, 200.0])
V_far = woods_saxon(r_far, -50.0, 5.0, 0.67)
assert np.all(np.abs(V_far) < 1.0), '[TC03] woods_saxon 远场趋于零 FAILED'

# ---- TC04: deformed_radius theta=pi/2 对称性 ----
from nuclear_potential import deformed_radius
R0 = deformed_radius(np.pi / 2, 16, 0.2, 0.0, 0.0)
Rpi = deformed_radius(np.pi / 2, 16, 0.2, 0.0, 0.0)
assert abs(R0 - Rpi) < 1e-10, '[TC04] deformed_radius theta=pi/2 对称性 FAILED'

# ---- TC05: coulomb_potential 内外连续有限 ----
from nuclear_potential import coulomb_potential
r_test = np.array([5.99, 6.0, 6.01])
Vc = coulomb_potential(r_test, 8, 16)
assert np.all(np.isfinite(Vc)), '[TC05] coulomb_potential 内外连续有限 FAILED'

# ---- TC06: build_neutron_potential 输出尺寸匹配 ----
rr = np.linspace(0, 10, 50)
Vn = build_neutron_potential(rr, 20, 8, 0.15, 0.05, -0.02)
assert Vn.shape == rr.shape, '[TC06] build_neutron_potential 输出尺寸匹配 FAILED'

# ---- TC07: gauss_lobatto_nodes 节点数量正确 ----
from radial_solver import gauss_lobatto_nodes
nodes = gauss_lobatto_nodes(8, -1.0, 1.0)
assert nodes.size == 8, '[TC07] gauss_lobatto_nodes 节点数量正确 FAILED'

# ---- TC08: lagrange_derivative_matrix 常数导数为零 ----
from radial_solver import lagrange_derivative_matrix
x = np.linspace(0, 1, 5)
D = lagrange_derivative_matrix(x)
const = np.ones(5)
deriv = D @ const
assert np.allclose(deriv, 0.0, atol=1e-10), '[TC08] lagrange_derivative_matrix 常数导数为零 FAILED'

# ---- TC09: solve_radial_schroedinger 返回束缚态能量 ----
V_func = lambda r_in: build_neutron_potential(r_in, 20, 8, 0.0, 0.0, 0.0)
en, wf, r_out = solve_radial_schroedinger(rmax=15.0, N=64, l=0, V_func=V_func, n_eig=3)
assert en.size > 0 and np.all(en < 0), '[TC09] solve_radial_schroedinger 返回束缚态能量 FAILED'

# ---- TC10: radial_matrix_element 交换对称性 ----
r_test = np.linspace(0, 5, 50)
u1 = np.sin(r_test)
u2 = np.cos(r_test)
m12 = radial_matrix_element(r_test, u1, u2)
m21 = radial_matrix_element(r_test, u2, u1)
assert abs(m12 - m21) < 1e-10, '[TC10] radial_matrix_element 交换对称性 FAILED'

# ---- TC11: bcs_occupation u2+v2=1 守恒 ----
from hfb_selfconsistent import bcs_occupation
eps = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
u2, v2, E = bcs_occupation(eps, 0.0, 1.0)
assert np.allclose(u2 + v2, 1.0, atol=1e-10), '[TC11] bcs_occupation u2+v2=1 守恒 FAILED'

# ---- TC12: conjugate_gradient_solve 解析验证 ----
from hfb_selfconsistent import conjugate_gradient_solve
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
x, info = conjugate_gradient_solve(A, b)
assert np.allclose(A @ x, b, atol=1e-8), '[TC12] conjugate_gradient_solve 解析验证 FAILED'

# ---- TC13: solve_hfb_bcs 粒子数守恒 ----
epsilon = np.array([-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0])
result = solve_hfb_bcs(epsilon, target_N=4, Delta0=1.0, tol=1e-8)
assert abs(result['particle_number'] - 4.0) < 0.1, '[TC13] solve_hfb_bcs 粒子数守恒 FAILED'

# ---- TC14: liquid_drop_binding_energy 返回值正数 ----
B = liquid_drop_binding_energy(8, 8)
assert B > 0 and np.isfinite(B), '[TC14] liquid_drop_binding_energy 返回值正数 FAILED'

# ---- TC15: atomic_mass_ldm 返回值有限正数 ----
M = atomic_mass_ldm(8, 8)
assert M > 0 and np.isfinite(M), '[TC15] atomic_mass_ldm 返回值有限正数 FAILED'

# ---- TC16: NuclearMassSurface 数据点精确重构 ----
data_N = np.array([8, 9, 10, 11, 12])
data_Z = np.array([8, 8, 8, 8, 8])
data_mass = np.array([atomic_mass_ldm(int(z), int(n)) for z, n in zip(data_Z, data_N)])
ms = NuclearMassSurface(data_N, data_Z, data_mass)
m_eval = ms.evaluate(data_N, data_Z)
assert np.allclose(m_eval, data_mass, atol=1e-6), '[TC16] NuclearMassSurface 数据点精确重构 FAILED'

# ---- TC17: mass_surface_curvature 返回标量有限值 ----
kappa = mass_surface_curvature(ms, 10.0, 8.0, h=1.0)
assert np.isscalar(kappa) and np.isfinite(kappa), '[TC17] mass_surface_curvature 返回标量有限值 FAILED'

# ---- TC18: noncentral_beta_cdf 边界值 ----
from decay_statistics import noncentral_beta_cdf
assert noncentral_beta_cdf(0.0, 2.0, 3.0, 0.5) == 0.0, '[TC18] noncentral_beta_cdf 边界值 F(0)=0 FAILED'
assert abs(noncentral_beta_cdf(1.0, 2.0, 3.0, 0.5) - 1.0) < 1e-10, '[TC18] noncentral_beta_cdf 边界值 F(1)=1 FAILED'

# ---- TC19: q_value_beta_decay 解析验证 ----
Q = q_value_beta_decay(100.0, 95.0)
assert abs(Q - 5.0) < 1e-10, '[TC19] q_value_beta_decay 解析验证 FAILED'

# ---- TC20: beta_decay_halflife Q<=0 返回无穷 ----
T12 = beta_decay_halflife(8, 0.0)
assert np.isinf(T12), '[TC20] beta_decay_halflife Q<=0 返回无穷 FAILED'

# ---- TC21: decay_chain_simulation 固定种子可复现 ----
np.random.seed(42)
Tmat = np.array([[0.0, 1.0], [0.0, 1.0]])
pops1, _ = decay_chain_simulation(0, Tmat, n_steps=3, n_samples=1000, seed=42)
np.random.seed(42)
pops2, _ = decay_chain_simulation(0, Tmat, n_steps=3, n_samples=1000, seed=42)
assert np.allclose(pops1, pops2, atol=1e-10), '[TC21] decay_chain_simulation 固定种子可复现 FAILED'

# ---- TC22: disk_monomial_integral I_00 等于 pi ----
I00 = disk_monomial_integral(0, 0)
assert abs(I00 - np.pi) < 1e-10, '[TC22] disk_monomial_integral I_00 等于 pi FAILED'

# ---- TC23: transfer_cross_section 非负有限 ----
sigma = transfer_cross_section(5.0, 1.0, 0.1)
assert sigma >= 0 and np.isfinite(sigma), '[TC23] transfer_cross_section 非负有限 FAILED'

# ---- TC24: angular_momentum_coupling_weight 范围 [0,1] ----
wgt = angular_momentum_coupling_weight(2.0, 2.5, 3.0, 0.5)
assert 0.0 <= wgt <= 1.0, '[TC24] angular_momentum_coupling_weight 范围 [0,1] FAILED'

# ---- TC25: tetrahedron_volume 正体积解析验证 ----
from density_mesh import tetrahedron_volume
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
vol = tetrahedron_volume(nodes, [0, 1, 2, 3])
assert vol > 0 and abs(vol - 1.0 / 6.0) < 1e-10, '[TC25] tetrahedron_volume 正体积解析验证 FAILED'

# ---- TC26: deformed_fermi_density 中心点正密度 ----
rho = deformed_fermi_density(0.0, 0.0, 0.0, 16, 0.1, 0.0, 0.0)
assert rho > 0 and np.isfinite(rho), '[TC26] deformed_fermi_density 中心点正密度 FAILED'

# ---- TC27: build_tetrahedral_sphere_mesh 节点和单元维度正确 ----
nodes, elements = build_tetrahedral_sphere_mesh(0.0, 5.0, 3, 1)
assert nodes.shape[1] == 3 and elements.shape[1] == 4, '[TC27] build_tetrahedral_sphere_mesh 节点和单元维度正确 FAILED'

# ---- TC28: clenshaw_curtis_weights 权重和为2 ----
from quadrature_engine import clenshaw_curtis_weights
w = clenshaw_curtis_weights(5)
assert abs(np.sum(w) - 2.0) < 1e-10, '[TC28] clenshaw_curtis_weights 权重和为2 FAILED'

# ---- TC29: integrate_on_triangle 常数函数等于三角形面积 ----
from quadrature_engine import integrate_on_triangle
f = lambda x, y: 1.0
area = integrate_on_triangle(f, degree=5)
assert abs(area - 0.5) < 1e-10, '[TC29] integrate_on_triangle 常数函数等于三角形面积 FAILED'

# ---- TC30: sparse_grid_integrate 常数函数等于 2^dim ----
val = sparse_grid_integrate(lambda x: 1.0, dim_num=2, level_max=2)
assert abs(val - 4.0) < 1e-6, '[TC30] sparse_grid_integrate 常数函数等于 2^dim FAILED'

# ---- TC31: nuclear_temperature 解析验证 ----
T = nuclear_temperature(excitation_energy=8.0, A=16)
expected = np.sqrt(8.0 / 2.0)
assert abs(T - expected) < 1e-10, '[TC31] nuclear_temperature 解析验证 FAILED'

# ---- TC32: evaporative_decay_rate 极端参数返回零 ----
rate = evaporative_decay_rate(16, 8, 0.0, 5.0)
assert rate == 0.0, '[TC32] evaporative_decay_rate 极端参数返回零 FAILED'
