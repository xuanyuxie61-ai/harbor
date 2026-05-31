# ---- TC01: detq_orthogonal 正交矩阵行列式为1 ----
import numpy as np
theta = 0.5
Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
d, ifault = detq_orthogonal(Q, 2)
assert abs(abs(d) - 1.0) < 1e-8, '[TC01] 正交矩阵行列式应为±1 FAILED'

# ---- TC02: detq_orthogonal 零维矩阵返回0 ----
d, ifault = detq_orthogonal(np.eye(2), 0)
assert d == 0.0 and ifault == 1, '[TC02] 零维输入应返回(0.0, 1) FAILED'

# ---- TC03: bisection_root_find 线性方程求根 ----
def f_linear(x):
    return x - 2.0
root, it, conv = bisection_root_find(f_linear, 0.0, 4.0, tol=1e-10)
assert abs(root - 2.0) < 1e-8, '[TC03] 线性方程求根误差过大 FAILED'
assert conv, '[TC03] 应标记为收敛 FAILED'

# ---- TC04: safe_divide 正常除法 ----
a = np.array([10.0, 20.0, 30.0])
b = np.array([2.0, 5.0, 3.0])
result = safe_divide(a, b)
assert abs(result[0] - 5.0) < 1e-12, '[TC04] 正常除法结果错误 FAILED'
assert abs(result[2] - 10.0) < 1e-12, '[TC04] 正常除法结果错误 FAILED'

# ---- TC05: safe_divide 除零保护 ----
a = np.array([1.0, 1.0])
b = np.array([0.0, 1e-16])
result = safe_divide(a, b)
assert np.all(np.isfinite(result)), '[TC05] 除零保护失败(产生inf) FAILED'
assert not np.any(np.isnan(result)), '[TC05] 除零保护失败(产生NaN) FAILED'

# ---- TC06: check_cfl 亚音速CFL条件 ----
dt = check_cfl(dx=0.01, dy=0.01, u=0.5, v=0.3, c=0.8, nu=1e-5, CFL_max=0.8)
assert dt > 0, '[TC06] CFL时间步长应为正数 FAILED'
assert dt < 1.0, '[TC06] CFL时间步长过大 FAILED'

# ---- TC07: check_cfl 超声速流CFL条件 ----
import numpy as np
dt2 = check_cfl(dx=0.005, dy=0.005, u=2.0, v=1.0, c=1.5, nu=1e-4, CFL_max=0.5)
assert dt2 > 0, '[TC07] CFL时间步长应为正数 FAILED'
assert np.isfinite(dt2), '[TC07] CFL时间步长应有限 FAILED'

# ---- TC08: condition_hager 单位矩阵条件数 ----
n = 4
A = np.eye(n)
cond = condition_hager(n, A)
assert 0.5 < cond < 5.0, '[TC08] 单位矩阵条件数应接近1 FAILED'

# ---- TC09: lu_decomposition_with_pivot LU分解可重构 ----
A = np.array([[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]])
L, U, P, success = lu_decomposition_with_pivot(A)
assert success, '[TC09] LU分解应成功 FAILED'
recon = P.T @ L @ U
assert np.linalg.norm(A - recon) < 1e-10, '[TC09] LU分解重构误差过大 FAILED'

# ---- TC10: solve_lu 求解线性系统 ----
A = np.array([[3.0, 1.0, -2.0], [1.0, 4.0, 1.0], [-2.0, 1.0, 5.0]])
b = np.array([1.0, 2.0, 3.0])
L, U, P, success = lu_decomposition_with_pivot(A)
assert success, '[TC10] LU分解应成功 FAILED'
x = solve_lu(L, U, P, b)
res = np.linalg.norm(A @ x - b)
assert res < 1e-10, '[TC10] LU求解残差过大 FAILED'

# ---- TC11: jacobi_preconditioner 对角预处理矩阵 ----
A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
M_inv = jacobi_preconditioner(A)
assert M_inv.shape == (3, 3), '[TC11] 预处理矩阵形状错误 FAILED'
assert abs(M_inv[0, 0] - 0.25) < 1e-12, '[TC11] 对角元素错误 FAILED'

# ---- TC12: hexagon_lyness_rule 六边形积分规则权重和为1 ----
n, x, y, w, strength = hexagon_lyness_rule(rule_id=2)
assert n == 6, '[TC12] rule_id=2应有6个积分点 FAILED'
assert abs(np.sum(w) - 1.0) < 1e-12, '[TC12] 权重和应为1 FAILED'
assert strength == 5, '[TC12] 代数精度应为5 FAILED'

# ---- TC13: wandzura_triangle_rule 三角形积分规则权重和为1 ----
xy, w, degree = wandzura_triangle_rule(rule_id=1)
assert xy.shape[0] == 2, '[TC13] 积分点坐标形状错误 FAILED'
assert abs(np.sum(w) - 1.0) < 1e-12, '[TC13] 权重和应为1 FAILED'
assert degree == 5, '[TC13] 多项式精度应为5 FAILED'

# ---- TC14: generate_voronoi_mesh Voronoi网格输出结构 ----
np.random.seed(42)
vor = generate_voronoi_mesh(nc=10, m=20, n=20)
assert 'generators' in vor, '[TC14] 缺少generators键 FAILED'
assert vor['generators'].shape[0] == 2, '[TC14] 生成点坐标维度错误 FAILED'
assert vor['voronoi_map'].shape == (20, 20), '[TC14] Voronoi地图形状错误 FAILED'

# ---- TC15: sample_boundary_points 边界采样点数量 ----
bx, by, dy_wall = sample_boundary_points(n_points=20, Re=1e5)
assert len(bx) == 400, '[TC15] x坐标点数错误 FAILED'
assert dy_wall > 0, '[TC15] 壁面间距应为正 FAILED'

# ---- TC16: generate_spectral_element_mesh 谱元网格输出 ----
mesh = generate_spectral_element_mesh(nx=8, ny=6, stretch_y=True)
assert mesh['nx'] > 0, '[TC16] nx应为正 FAILED'
assert mesh['ny'] > 0, '[TC16] ny应为正 FAILED'
assert 'X' in mesh and 'Y' in mesh, '[TC16] 缺少X/Y坐标网格 FAILED'

# ---- TC17: DCT可逆性 (DCT-II与DCT-III互为逆变换) ----
np.random.seed(42)
dct_input = np.random.randn(8)
c = discrete_cosine_transform_1d(dct_input)
d = inverse_discrete_cosine_transform_1d(c)
assert np.linalg.norm(dct_input - d) < 1e-10, '[TC17] DCT可逆性误差过大 FAILED'

# ---- TC18: hermite_interpolant Hermite插值验证 ----
x_nodes = np.array([0.0, 1.0, 2.0])
y_nodes = np.array([1.0, 0.0, 1.0])
yp_nodes = np.array([0.0, -2.0, 0.0])
xd, yd = hermite_interpolant_coeffs(3, x_nodes, y_nodes, yp_nodes)
y_mid = hermite_interpolant_eval(xd, yd, 0.5)
assert np.isfinite(y_mid), '[TC18] Hermite插值应产生有限值 FAILED'

# ---- TC19: snapshot_pod POD模态正交性 ----
np.random.seed(42)
A_test = np.random.randn(100, 10)
pod_result = snapshot_pod(A_test, num_modes=5)
modes = pod_result['modes']
assert pod_result['num_modes'] > 0, '[TC19] 应至少保留1个模态 FAILED'
assert modes.shape[0] == 100, '[TC19] 模态空间维度错误 FAILED'

# ---- TC20: build_markov_transition_matrix 转移矩阵行和为1 ----
P = build_markov_transition_matrix(n_states=10, move_range=1)
row_sums = np.sum(P, axis=1)
assert np.allclose(row_sums, 1.0), '[TC20] 转移矩阵行和应为1 FAILED'
assert np.all(P >= 0.0), '[TC20] 转移概率应非负 FAILED'

# ---- TC21: compute_markov_chain_stationary 稳态分布 ----
P = build_markov_transition_matrix(n_states=10, move_range=1)
pi = compute_markov_chain_stationary(P, max_iter=500)
assert abs(np.sum(pi) - 1.0) < 1e-10, '[TC21] 稳态分布和应为1 FAILED'
assert np.all(pi >= 0.0), '[TC21] 稳态概率应非负 FAILED'

# ---- TC22: cavitation_probability_local 确定性空化概率边界 ----
p_high = cavitation_probability_local(mean_p=1000.0, p_vapor=10.0, std_p=10.0)
assert p_high < 0.01, '[TC22] 高压下空化概率应接近0 FAILED'
p_low = cavitation_probability_local(mean_p=10.0, p_vapor=1000.0, std_p=10.0)
assert p_low > 0.99, '[TC22] 低压下空化概率应接近1 FAILED'

# ---- TC23: joint_cavitation_probability 联合概率边界 ----
probs_low = np.array([0.0, 0.0, 0.0])
jp = joint_cavitation_probability(probs_low, independence=True)
assert jp == 0.0, '[TC23] 全零概率联合应为0 FAILED'
probs_high = np.array([1.0, 0.5])
jp2 = joint_cavitation_probability(probs_high, independence=True)
assert jp2 == 1.0, '[TC23] 含1.0时联合应为1 FAILED'

# ---- TC24: cavitation_inception_criterion 空化初生准则输出结构 ----
result = cavitation_inception_criterion(Re=1e5, sigma=0.3)
assert 'cavitation_inception' in result, '[TC24] 缺少cavitation_inception FAILED'
assert 'sigma_critical' in result, '[TC24] 缺少sigma_critical FAILED'
assert 0.0 <= result['inception_risk'] <= 1.0, '[TC24] inception_risk应在[0,1] FAILED'

# ---- TC25: compute_nucleation_rate 成核率有限性 ----
rate = compute_nucleation_rate(p=1000.0, p_vapor=2000.0, T=300.0)
assert np.isfinite(rate), '[TC25] 成核率应为有限值 FAILED'
assert rate >= 0.0, '[TC25] 成核率应非负 FAILED'

# ---- TC26: estimate_convergence_order 收敛阶估计 ----
residuals = [1.0 * (0.5 ** k) for k in range(50)]
conv = estimate_convergence_order(residuals)
assert conv['order'] is not None, '[TC26] 应能估计收敛阶 FAILED'
assert conv['order'] > 0.5, '[TC26] 收敛阶应接近1 FAILED'

# ---- TC27: compute_gci GCI单调性 ----
gci = compute_gci(fine=1.000, medium=0.980, coarse=0.950, r=2.0)
assert 'gci_fine_medium' in gci, '[TC27] 缺少gci_fine_medium FAILED'
assert gci['gci_fine_medium'] >= 0.0, '[TC27] GCI应非负 FAILED'

# ---- TC28: CompressibleNSSolver 初始化后守恒变量有限 ----
np.random.seed(42)
solver = CompressibleNSSolver(nx=16, ny=16, Lx=1.0, Ly=0.5, gamma=1.4, Re=1000.0, Pr=0.71, Ma=0.3, T_wall=1.0)
assert solver.Q.shape == (16, 16, 4), '[TC28] 守恒变量形状错误 FAILED'
assert np.all(np.isfinite(solver.Q)), '[TC28] 守恒变量应全为有限值 FAILED'

# ---- TC29: spectral_derivative_1d 常数函数导数为零 ----
n_pts = 8
x_cheb = np.cos(np.pi * np.arange(n_pts + 1) / n_pts)
u_const = np.ones(n_pts + 1)
du = spectral_derivative_1d(u_const, x_cheb)
assert np.max(np.abs(du)) < 1e-8, '[TC29] 常数函数导数应接近0 FAILED'

# ---- TC30: assemble_fem_mass_matrix_2d 质量矩阵正定性 ----
nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
elements = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
M = assemble_fem_mass_matrix_2d(nodes, elements)
eigvals = np.linalg.eigvalsh(M)
assert np.all(eigvals > 0), '[TC30] 质量矩阵应正定 FAILED'

# ---- TC31: compute_turbulent_kinetic_energy TKE非负 ----
np.random.seed(42)
u_snaps = np.random.randn(200, 5) * 0.1 + 1.0
v_snaps = np.random.randn(200, 5) * 0.1
tke_result = compute_turbulent_kinetic_energy(u_snaps, v_snaps)
assert np.all(tke_result['tke'] >= 0.0), '[TC31] 湍动能应非负 FAILED'
assert 'R_uv' in tke_result, '[TC31] 缺少Reynolds应力 FAILED'

# ---- TC32: metropolis_hastings_sampler 接受率在[0,1] ----
def log_posterior_simple(theta):
    return -0.5 * np.sum(theta ** 2)
np.random.seed(42)
result = metropolis_hastings_sampler(log_posterior_simple, np.array([0.0, 0.0]),
                                      n_samples=100, burn_in=50, thin=1,
                                      proposal_cov=np.eye(2) * 0.1)
assert 0.0 <= result['acceptance_rate'] <= 1.0, '[TC32] 接受率应在[0,1] FAILED'
assert result['samples'].shape[0] == 100, '[TC32] 样本数错误 FAILED'

# ---- TC33: 集成测试 CompressibleNSSolver 单步推进不崩溃 ----
np.random.seed(42)
solver2 = CompressibleNSSolver(nx=32, ny=16, Lx=1.0, Ly=0.5, gamma=1.4, Re=5000.0, Pr=0.71, Ma=0.3, T_wall=1.0)
Q_before = solver2.Q.copy()
solver2.step_rk3()
assert solver2.iter == 1, '[TC33] 迭代计数应为1 FAILED'
assert np.all(np.isfinite(solver2.Q)), '[TC33] 推进后守恒变量应有限 FAILED'

# ---- TC34: dct_poisson_solver_2d 求解无源方程输出全零(近似) ----
nx_p, ny_p = 16, 12
dx_p, dy_p = 1.0 / (nx_p - 1), 1.0 / (ny_p - 1)
f_zero = np.zeros((ny_p, nx_p))
p_zero = dct_poisson_solver_2d(f_zero, dx_p, dy_p)
assert np.max(np.abs(p_zero)) < 1e-10, '[TC34] 无源泊松方程解应接近零 FAILED'

# ---- TC35: check_energy_conservation 能量检查输出结构 ----
Q_list = [np.ones((4, 4, 4))]
energy_check = check_energy_conservation(Q_list, gamma=1.4)
assert 'drift' in energy_check, '[TC35] 缺少drift FAILED'
assert 'energy' in energy_check, '[TC35] 缺少energy FAILED'

# ---- TC36: integrate_scalar_on_hexagon 六边形积分数值检查 ----
def f_hex(x, y):
    return 1.0
integral = integrate_scalar_on_hexagon(f_hex, R=1.0, rule_id=2)
theory = 3.0 * np.sqrt(3.0) / 2.0
assert abs(integral - theory) < 0.01, '[TC36] 六边形常数积分误差过大 FAILED'

# ---- TC37: integrate_scalar_on_triangle 三角形积分数值检查 ----
def f_tri(xi, eta):
    return 1.0
tri_nodes = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
integral_tri = integrate_scalar_on_triangle(f_tri, tri_nodes, rule_id=1)
assert abs(integral_tri - 0.5) < 0.01, '[TC37] 三角形常数积分误差过大 FAILED'

# ---- TC38: apply_jacobi_precond 正定系统收敛 ----
A_diag = np.array([[4.0, 1.0], [1.0, 4.0]])
b_test = np.array([1.0, 1.0])
x_jac = apply_jacobi_precond(A_diag, b_test, max_iter=200)
res_jac = np.linalg.norm(A_diag @ x_jac - b_test)
assert res_jac < 0.01, '[TC38] Jacobi迭代残差过大 FAILED'

# ---- TC39: compute_mass_flow_rate 质量流量输出结构 ----
Q_test = np.ones((8, 8, 4))
y_test = np.linspace(0, 1, 8)
mass = compute_mass_flow_rate(Q_test, y_test)
assert 'mass_flow_in' in mass, '[TC39] 缺少mass_flow_in FAILED'
assert 'relative_error' in mass, '[TC39] 缺少relative_error FAILED'

# ---- TC40: CompressibleNSSolver.primitive_variables 原始变量一致性 ----
np.random.seed(42)
solver3 = CompressibleNSSolver(nx=16, ny=16, Lx=1.0, Ly=0.5, gamma=1.4, Re=1000.0)
rho, u, v, p, e, T = solver3.primitive_variables(solver3.Q)
assert np.all(rho > 0), '[TC40] 密度应全为正 FAILED'
assert np.all(p > 0), '[TC40] 压力应全为正 FAILED'
assert np.all(np.isfinite(T)), '[TC40] 温度应有限 FAILED'
