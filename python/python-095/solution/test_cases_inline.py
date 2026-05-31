# ---- TC01: tridiagonal_solver求解有限且非NaN ----
from tridiagonal_acoustics import tridiagonal_solver
a = np.array([0.0, -1.0, -1.0])
b = np.array([2.0, 2.0, 2.0])
c = np.array([-1.0, -1.0, 0.0])
d = np.array([1.0, 0.0, 0.0])
x = tridiagonal_solver(a, b, c, d)
assert np.all(np.isfinite(x)), '[TC01] tridiagonal_solver产生非有限值 FAILED'

# ---- TC02: tridiagonal_mv与solver一致性 ----
from tridiagonal_acoustics import tridiagonal_solver, tridiagonal_mv
a = np.array([0.0, -1.0, -1.0])
b = np.array([2.0, 2.0, 2.0])
c = np.array([-1.0, -1.0, 0.0])
d = np.array([1.0, 0.0, 0.0])
x = tridiagonal_solver(a.copy(), b.copy(), c.copy(), d.copy())
rhs = tridiagonal_mv(a, b, c, x)
assert np.max(np.abs(rhs - d)) < 1e-10, '[TC02] tridiagonal_mv与solver不一致 FAILED'

# ---- TC03: pipe_helmholtz_solver声压有限 ----
from tridiagonal_acoustics import pipe_helmholtz_solver
L = 1.0
N = 50
k = 2.0 * math.pi * 500.0 / 343.0
source = np.zeros(N, dtype=complex)
source[N // 2] = 1.0e-3
x_pipe, p_pipe = pipe_helmholtz_solver(L, N, k, source)
assert np.all(np.isfinite(p_pipe)), '[TC03] pipe_helmholtz_solver产生非有限声压 FAILED'

# ---- TC04: lindberg_exact_solution残差接近零 ----
from tridiagonal_acoustics import lindberg_exact_solution, lindberg_residual
t_test = np.linspace(0, 1.0, 11)
y_exact, dydt_exact = lindberg_exact_solution(t_test)
res = lindberg_residual(t_test, y_exact, dydt_exact)
max_res = np.max(np.abs(res))
assert max_res < 1e-10, '[TC04] Lindberg精确解残差过大 FAILED'

# ---- TC05: sphere_fibonacci_grid_points点在球面上 ----
from spherical_array_geometry import sphere_fibonacci_grid_points
N_sensors = 32
radius = 0.5
sensors = sphere_fibonacci_grid_points(N_sensors, radius)
distances = np.linalg.norm(sensors, axis=1)
assert np.max(np.abs(distances - radius)) < 1e-10, '[TC05] Fibonacci网格点不在球面上 FAILED'

# ---- TC06: sphere_fibonacci_grid_points输出形状 ----
from spherical_array_geometry import sphere_fibonacci_grid_points
sensors = sphere_fibonacci_grid_points(16, 1.0)
assert sensors.shape == (16, 3), '[TC06] Fibonacci网格输出形状错误 FAILED'

# ---- TC07: SparseAcousticMatrix ST与CCS乘法一致 ----
from sparse_acoustics import SparseAcousticMatrix
S = SparseAcousticMatrix(5, 5)
for i in range(5):
    S.add_entry(i, i, 2.0)
    if i < 4:
        S.add_entry(i, i + 1, -1.0)
S.st_to_ccs()
x_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_st = S.st_mv(x_vec)
y_ccs = S.ccs_mv(x_vec)
assert np.max(np.abs(y_st - y_ccs)) < 1e-12, '[TC07] ST与CCS乘法不一致 FAILED'

# ---- TC08: generate_room_coupling_graph输出有限 ----
from sparse_acoustics import generate_room_coupling_graph
graph = generate_room_coupling_graph(10, connection_prob=0.15, seed=42)
graph.st_to_ccs()
y = graph.ccs_mv(np.ones(10))
assert np.all(np.isfinite(y)), '[TC08] 房间耦合图输出非有限 FAILED'

# ---- TC09: zero_chandrupatla求根精度 ----
from source_phase_optimizer import zero_chandrupatla
f = lambda x: x**3 - 2.0
xm, fm, calls = zero_chandrupatla(f, 1.0, 2.0)
assert abs(fm) < 1e-6, '[TC09] Chandrupatla求根精度不足 FAILED'

# ---- TC10: optimize_source_phase能量不增 ----
from source_phase_optimizer import optimize_source_phase
np.random.seed(42)
H_col = (np.random.randn(8) + 1j * np.random.randn(8)) * 0.5
d_noise = (np.random.randn(8) + 1j * np.random.randn(8)) * 0.3
phi_opt, min_energy = optimize_source_phase(H_col, d_noise, 0.5)
s0 = 0.5 * np.exp(1j * 0.0)
p0 = d_noise + H_col * s0
energy0 = np.vdot(p0, p0).real
assert min_energy <= energy0 + 1e-12, '[TC10] 相位优化未降低能量 FAILED'

# ---- TC11: qr_rank_revealing_ls残差小 ----
from adaptive_filter import qr_rank_revealing_ls
np.random.seed(42)
A_test = np.random.randn(20, 5)
w_true = np.random.randn(5)
b_test = A_test @ w_true
w_qr, rank = qr_rank_revealing_ls(A_test, b_test)
residual_norm = np.linalg.norm(A_test @ w_qr - b_test)
assert residual_norm < 1e-8, '[TC11] QR秩揭示最小二乘残差过大 FAILED'

# ---- TC12: MultichannelFxLMS系数更新形状不变 ----
from adaptive_filter import MultichannelFxLMS
np.random.seed(42)
sec_model = np.random.randn(2, 2, 4) * 0.1
fxlms = MultichannelFxLMS(2, 8, sec_model, mu=0.001)
x_ref = np.random.randn(2) * 0.5
target = np.random.randn(2) * 0.3
fxlms.update(x_ref, target)
assert fxlms.w.shape == (2, 8), '[TC12] FxLMS更新后系数形状改变 FAILED'

# ---- TC13: subset_sum_swap_anc功率不超预算 ----
from optimal_source_selection import subset_sum_swap_anc
powers = np.array([50, 80, 120, 40, 90, 60, 110, 30], dtype=float)
budget = 250.0
selected, achieved = subset_sum_swap_anc(powers, budget)
assert achieved <= budget + 1e-6, '[TC13] 子集和功率超过预算 FAILED'

# ---- TC14: greedy_source_selection选中数不超上限 ----
from optimal_source_selection import greedy_source_selection
np.random.seed(42)
M_sens = 4
N_sources = 8
H_sel = np.random.randn(M_sens, N_sources) + 1j * np.random.randn(M_sens, N_sources)
d_sel = np.random.randn(M_sens) + 1j * np.random.randn(M_sens)
max_src = 3
src_powers = np.array([30, 45, 60, 35, 50, 70, 40, 55], dtype=float)
sel, filters = greedy_source_selection(H_sel, d_sel, max_src, 200.0, src_powers)
assert np.sum(sel) <= max_src, '[TC14] 贪心选择声源数超过上限 FAILED'

# ---- TC15: dirichlet_estimate_mle估计有限 ----
from statistical_noise_model import dirichlet_estimate_mle
rng = np.random.default_rng(42)
alpha_true = np.array([3.0, 2.0, 4.0, 2.5])
x_data = rng.dirichlet(alpha_true, 500)
alpha_est, niter, loglik = dirichlet_estimate_mle(x_data)
assert np.all(np.isfinite(alpha_est)) and niter >= 0, '[TC15] Dirichlet MLE估计非有限或迭代异常 FAILED'

# ---- TC16: adaptive_step_size_from_dirichlet范围约束 ----
from statistical_noise_model import adaptive_step_size_from_dirichlet
rng = np.random.default_rng(42)
alpha_true = np.array([3.0, 2.0, 4.0, 2.5])
x_data = rng.dirichlet(alpha_true, 200)
mu = adaptive_step_size_from_dirichlet(x_data, base_mu=0.001)
assert np.all((mu >= 0.0001) & (mu <= 0.005)), '[TC16] 自适应步长超出合理范围 FAILED'

# ---- TC17: noise_stationarity_test平稳信号判稳 ----
from statistical_noise_model import noise_stationarity_test
np.random.seed(42)
err_hist = np.random.normal(0, 0.1, 200)
is_stat, f_stat = noise_stationarity_test(err_hist, window=40)
assert is_stat, '[TC17] 平稳信号误判为非平稳 FAILED'

# ---- TC18: cos_power_int解析验证 ----
from special_functions import cos_power_int
cpi = cos_power_int(0.0, math.pi / 2, 4)
expected = 3.0 * math.pi / 16.0
assert abs(cpi - expected) < 1e-10, '[TC18] cos^4积分与理论值不符 FAILED'

# ---- TC19: digamma已知值验证 ----
from special_functions import digamma
psi_val, ierr = digamma(1.0)
expected = -0.5772156649
assert abs(psi_val - expected) < 1e-6 and ierr == 0, '[TC19] digamma(1)值错误 FAILED'

# ---- TC20: trigamma已知值验证 ----
from special_functions import trigamma
psi_prime, ierr = trigamma(1.0)
expected = math.pi**2 / 6.0
assert abs(psi_prime - expected) < 1e-6 and ierr == 0, '[TC20] trigamma(1)值错误 FAILED'

# ---- TC21: piston_radiation_resistance小ka近似 ----
from integrals_radiation import piston_radiation_resistance
ka = 1e-4
R_ratio = piston_radiation_resistance(ka)
approx = ka**2 / 2.0
assert abs(R_ratio - approx) < 1e-6, '[TC21] 小ka辐射阻力近似不符 FAILED'

# ---- TC22: rk4_integrate常数ODE精确 ----
from nonlinear_ode_dynamics import rk4_integrate
traj = rk4_integrate(lambda t, y: np.array([0.0, 0.0]), 0.0, [1.0, -1.0], 1.0, h=0.1)
final = traj[-1][1]
assert np.max(np.abs(final - np.array([1.0, -1.0]))) < 1e-12, '[TC22] RK4常数ODE积分不精确 FAILED'

# ---- TC23: stability_boundary_anishchenko稳定区域验证 ----
from nonlinear_ode_dynamics import stability_boundary_anishchenko
mu_grid, gamma_grid, stable = stability_boundary_anishchenko((0.1, 2.0), (0.1, 2.0), n_grid=30)
i_mu = np.argmin(np.abs(mu_grid - 0.5))
j_ga = np.argmin(np.abs(gamma_grid - 1.0))
assert stable[i_mu, j_ga], '[TC23] 已知稳定点被判为不稳定 FAILED'

# ---- TC24: AcousticRoomFEM RCM重排序带宽不增 ----
from acoustic_room_model import generate_box_mesh, AcousticRoomFEM
nodes, elements = generate_box_mesh(1.0, 1.0, 1.0, nx=4, ny=4, nz=4)
fem = AcousticRoomFEM(nodes, elements)
bw_before = fem.compute_bandwidth()
fem.rcm_reorder()
bw_after = fem.compute_bandwidth()
assert bw_after <= bw_before, '[TC24] RCM重排序后带宽增加 FAILED'

# ---- TC25: generate_box_mesh节点数正确 ----
from acoustic_room_model import generate_box_mesh
nodes, elements = generate_box_mesh(1.0, 1.0, 1.0, nx=3, ny=3, nz=3)
expected_nodes = 3 * 3 * 3
assert nodes.shape[0] == expected_nodes, '[TC25] 盒状网格节点数错误 FAILED'

# ---- TC26: rayleigh_integral_piston输出有限 ----
from integrals_radiation import disk_unit_sample, rayleigh_integral_piston
disk_pts = disk_unit_sample(100, radius=0.1)
observer = np.array([0.0, 0.0, 1.0])
p_rayleigh = rayleigh_integral_piston(observer, disk_pts, u_n=0.01, k=2.0 * math.pi * 1000.0 / 343.0)
assert np.isfinite(p_rayleigh), '[TC26] Rayleigh积分产生非有限值 FAILED'

# ---- TC27: piston_directivity_factor小ka理论值 ----
from integrals_radiation import piston_directivity_factor
di = piston_directivity_factor(0.001)
assert abs(di - 3.0103) < 0.1, '[TC27] 小ka指向性因子与理论值不符 FAILED'

# ---- TC28: acoustic_transfer_matrix_sparse维度正确 ----
from sparse_acoustics import acoustic_transfer_matrix_sparse
from spherical_array_geometry import sphere_fibonacci_grid_points
sensors = sphere_fibonacci_grid_points(8, 0.5)
sources = sphere_fibonacci_grid_points(4, 0.3)
k = 2.0 * math.pi * 500.0 / 343.0
H = acoustic_transfer_matrix_sparse(sensors, sources, k)
assert H.m == 8 and H.n == 4, '[TC28] 稀疏传递矩阵维度错误 FAILED'

# ---- TC29: log_beta对称性 ----
from special_functions import log_beta
lb1 = log_beta(2.0, 3.0)
lb2 = log_beta(3.0, 2.0)
assert abs(lb1 - lb2) < 1e-12, '[TC29] log_beta不对称 FAILED'

# ---- TC30: betain边界值 ----
from special_functions import betain, log_beta
lb = log_beta(2.0, 3.0)
val0, ierr0 = betain(0.0, 2.0, 3.0, lb)
val1, ierr1 = betain(1.0, 2.0, 3.0, lb)
assert val0 == 0.0 and ierr0 == 0, '[TC30] betain(0)边界错误 FAILED'
assert val1 == 1.0 and ierr1 == 0, '[TC30] betain(1)边界错误 FAILED'
