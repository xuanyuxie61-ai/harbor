
# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: prime_factors 分解 60 为 [2,2,3,5] ----
from utils import prime_factors
factors = prime_factors(60)
assert factors == [2, 2, 3, 5], '[TC01] prime_factors(60) FAILED'

# ---- TC02: 最优 Chebyshev 阶数 80 满足 81=3^4 ----
from utils import optimal_chebyshev_order
N_opt = optimal_chebyshev_order(80, max_prime=5)
assert N_opt == 80, '[TC02] optimal_chebyshev_order(80) FAILED'

# ---- TC03: normalize_array minmax 归一化到 [0,1] ----
from utils import normalize_array
arr = np.array([10.0, 20.0, 30.0])
norm = normalize_array(arr, method="minmax")
assert np.allclose(norm, [0.0, 0.5, 1.0]), '[TC03] normalize_array minmax FAILED'

# ---- TC04: safe_divide 零除返回 fill_value ----
from utils import safe_divide
result = safe_divide(5.0, 0.0, fill_value=999.0)
assert result == 999.0, '[TC04] safe_divide zero division FAILED'

# ---- TC05: Blasius 壁面无滑移与远场渐近 ----
from utils import blasius_function
eta_test = np.array([0.0, 12.0])
f, fp, fpp = blasius_function(eta_test)
assert abs(fp[0]) < 1e-3, '[TC05] Blasius fp(0) FAILED'
assert abs(fp[-1] - 1.0) < 1e-3, '[TC05] Blasius fp(inf) FAILED'

# ---- TC06: Sutherland 粘度在 T=T_ref 时等于 mu_ref ----
from utils import sutherland_viscosity
mu_ref = 1.7894e-5
mu = sutherland_viscosity(np.array([300.0]))
assert abs(mu[0] - mu_ref) < 1e-10, '[TC06] sutherland_viscosity T_ref FAILED'

# ---- TC07: Chebyshev 节点数量与范围 ----
from utils import chebyshev_nodes
nodes = chebyshev_nodes(8, a=0.0, b=1.0)
assert len(nodes) == 9, '[TC07] chebyshev_nodes length FAILED'
assert np.all((nodes >= 0.0) & (nodes <= 1.0)), '[TC07] chebyshev_nodes range FAILED'

# ---- TC08: Chebyshev 微分矩阵尺寸为 (n+1, n+1) ----
from utils import chebyshev_diff_matrix
D = chebyshev_diff_matrix(6)
assert D.shape == (7, 7), '[TC08] chebyshev_diff_matrix shape FAILED'

# ---- TC09: 单位四面体体积精确为 1/6 ----
from fem_basis import tetrahedron_volume
t_unit = np.array([[0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])
vol = tetrahedron_volume(t_unit)
assert abs(vol - 1.0/6.0) < 1e-12, '[TC09] tetrahedron_volume unit tet FAILED'

# ---- TC10: TET4 基函数在重心处满足 partition of unity ----
from fem_basis import tet4_basis
phi = tet4_basis(t_unit, np.array([[0.25], [0.25], [0.25]]))
assert abs(np.sum(phi) - 1.0) < 1e-10, '[TC10] tet4_basis partition of unity FAILED'

# ---- TC11: 参考坐标与物理坐标互逆映射 ----
from fem_basis import reference_to_physical_tet4, physical_to_reference_tet4
xi_test = np.array([0.2, 0.3, 0.1])
x_phys = reference_to_physical_tet4(t_unit, xi_test)
xi_back = physical_to_reference_tet4(t_unit, x_phys)
assert np.allclose(xi_test, xi_back, atol=1e-10), '[TC11] reference/physical inverse FAILED'

# ---- TC12: Chebyshev 求积对不超过 2n-1 次多项式精确 ----
from spectral_integrator import chebyshev1_exactness_test
exactness = chebyshev1_exactness_test(n=8, degree_max=10)
low_degree_err = [err for deg, err in exactness if deg <= 7]
assert all(err < 1e-14 for err in low_degree_err), '[TC12] chebyshev1_exactness FAILED'

# ---- TC13: 球面直角三角形面积为 pi/2 ----
from spectral_integrator import sphere01_triangle_area
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
c = np.array([0.0, 0.0, 1.0])
area = sphere01_triangle_area(a, b, c)
assert abs(area - pi/2.0) < 1e-3, '[TC13] sphere01_triangle_area FAILED'

# ---- TC14: 零增长率时 N 积分保持为零 ----
from spectral_integrator import integrate_boundary_layer_growth
x_test = np.linspace(0, 1, 11)
alpha_zero = np.zeros_like(x_test)
N_prof = integrate_boundary_layer_growth(x_test, alpha_zero)
assert np.allclose(N_prof, 0.0), '[TC14] integrate_boundary_layer_growth zero FAILED'

# ---- TC15: trapz 与 simpson 积分方法一致性 ----
from spectral_integrator import amplification_factor_integral
Re_test = np.linspace(1e5, 2e6, 301)
ai_test = -0.001 * np.ones_like(Re_test)
_, N_trapz = amplification_factor_integral(Re_test, ai_test, method='trapz')
_, N_simp = amplification_factor_integral(Re_test, ai_test, method='simpson')
assert abs(N_trapz[-1] - N_simp[-1]) < 1e-3, '[TC15] amplification methods consistency FAILED'

# ---- TC16: e_n_method 输出 N 长度与输入一致 ----
from transition_predictor import e_n_method, compute_growth_rate_profile
Re_x = np.linspace(1e5, 5e6, 100)
ai = compute_growth_rate_profile(Re_x, Ma=6.0, Re_unit=1e6, Tw_Te=1.0)
Re_xt, N_prof = e_n_method(Re_x, ai, N_cr=9.0)
assert len(N_prof) == len(Re_x), '[TC16] e_n_method output length FAILED'

# ---- TC17: 空间增长率非正（负值表示扰动增长） ----
from transition_predictor import compute_growth_rate_profile
Re_x = np.linspace(1e5, 5e6, 50)
ai = compute_growth_rate_profile(Re_x, Ma=6.0, Tw_Te=1.0)
assert np.all(ai <= 0.0), '[TC17] growth_rate_profile sign FAILED'

# ---- TC18: 转捩前沿成本平移不变性 ----
from transition_predictor import transition_front_cost
pos1 = np.array([1.0, 2.0, 3.0])
pen = np.array([0.1, 0.1, 0.1])
c1 = transition_front_cost(pos1, pen)
c2 = transition_front_cost(pos1 + 5.0, pen)
assert c1 == c2, '[TC18] transition_front_cost translation invariance FAILED'

# ---- TC19: 感受性系数非负 ----
from transition_predictor import receptivity_coefficient
C_rec = receptivity_coefficient(Ma=6.0, Tw_Te=1.0, Tu=0.005)
assert C_rec >= 0.0, '[TC19] receptivity_coefficient sign FAILED'

# ---- TC20: 平板网格节点数匹配 Nx*Ny ----
from mesh_generator import BoundaryLayerMesh
mesh = BoundaryLayerMesh(L=1.0, H=0.1, Nx=10, Ny=8, Re=1e6, Ma=6.0)
nodes, nx, ny = mesh.generate_flat_plate_mesh()
assert len(nodes) == nx * ny, '[TC20] flat_plate mesh size FAILED'

# ---- TC21: 三角形邻居边界标记为 -1 ----
tri_small = np.array([[0, 1, 2], [1, 3, 2]])
neighbors = mesh.triangle_neighbors(2, tri_small)
assert np.any(neighbors == -1), '[TC21] triangle_neighbors boundary FAILED'

# ---- TC22: 边界节点集合非空 ----
bnds = mesh.boundary_nodes(nx, ny)
total_boundary = len(bnds['wall']) + len(bnds['inlet']) + len(bnds['outlet']) + len(bnds['farfield'])
assert total_boundary > 0, '[TC22] boundary_nodes empty FAILED'

# ---- TC23: 球面波矢方向均为单位向量 ----
from mesh_generator import sphere_wavevector_grid
wv = sphere_wavevector_grid(lat_num=4, long_num=8)
norms = np.linalg.norm(wv, axis=1)
assert np.allclose(norms, 1.0), '[TC23] sphere_wavevector_grid unit norm FAILED'

# ---- TC24: 热求解器壁温与远场温度边界条件 ----
from thermal_solver import HypersonicThermalSolver
thermal = HypersonicThermalSolver(Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4,
                                   Tw_over_Te=0.6, L=1.0, N_eta=50, eta_max=12.0)
sol = thermal.solve_self_similar_energy(epsilon=1e-6, max_iter=10000)
assert abs(sol['T'][0] - 0.6) < 1e-2, '[TC24] thermal wall temp BC FAILED'
assert abs(sol['T'][-1] - 1.0) < 1e-2, '[TC24] thermal farfield temp BC FAILED'

# ---- TC25: LST 时间特征值为复数数组 ----
from stability_analysis import CompressibleLST
from utils import blasius_function, sutherland_viscosity
lst = CompressibleLST(Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4, N=40)
eta_c = np.linspace(0, 12, 200)
f_c, fp_c, _ = blasius_function(eta_c)
u_c = np.clip(fp_c, 0, 1)
T_c = np.ones_like(eta_c)
mu_c = sutherland_viscosity(T_c)
lst.set_baseflow(eta_c, u_c, T_c, mu_c)
eigvals = lst.temporal_eigenvalues(alpha=0.3, beta=0.0)
assert np.iscomplexobj(eigvals), '[TC25] LST eigenvalue type FAILED'

# ---- TC26: 模态追踪输出长度与输入波数列表一致 ----
from stability_analysis import track_eigenvalue_mode
alpha_list = np.linspace(0.05, 0.3, 5)
tracked = track_eigenvalue_mode(alpha_list, lst, beta=0.0)
assert len(tracked) == len(alpha_list), '[TC26] track_eigenvalue_mode length FAILED'

# ---- TC27: LHS 采样马赫数在指定范围内 ----
from monte_carlo_sampler import HypersonicParameterSampler
np.random.seed(42)
sampler = HypersonicParameterSampler(Ma_range=(5.0, 8.0), Re_range=(5e5, 2e7),
                                      Tw_Te_range=(0.4, 1.5), Tu_range=(0.001, 0.02))
samples = sampler.lhs_sampling(n_samples=20)
assert np.all((samples[:, 0] >= 5.0) & (samples[:, 0] <= 8.0)), '[TC27] LHS Ma range FAILED'

# ---- TC28: 随机转捩模型返回值不低于下限 1e4 ----
from monte_carlo_sampler import random_transition_model
np.random.seed(42)
Re_t = random_transition_model(Ma=6.0, Re=1e6, Tw_Te=1.0, Tu=0.005)
assert Re_t >= 1e4, '[TC28] random_transition_model lower bound FAILED'

# ---- TC29: XY 数据文件读写一致性 ----
import tempfile
from data_io import write_xy_data, read_xy_data
x_test = np.array([1.0, 2.0, 3.0])
y_test = np.array([4.0, 5.0, 6.0])
with tempfile.NamedTemporaryFile(mode='w', suffix='.xy', delete=False) as tf:
    tmpname = tf.name
write_xy_data(tmpname, x_test, y_test)
x_read, y_read = read_xy_data(tmpname)
os.remove(tmpname)
assert np.allclose(x_read, x_test) and np.allclose(y_read, y_test), '[TC29] XY data IO consistency FAILED'

# ---- TC30: 主程序输出文件已生成 ----
assert os.path.exists("baseflow_profile.xy"), '[TC30] baseflow_profile.xy missing FAILED'
assert os.path.exists("transition_report.txt"), '[TC30] transition_report.txt missing FAILED'

print('\n全部 30 个测试通过!\n')
