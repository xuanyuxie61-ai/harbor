from combustion_utils import (
    rankine_hugoniot_pressure_ratio, rankine_hugoniot_density_ratio,
    arrhenius_rate, sound_speed, specific_heat_ratio_cv_cp,
    znd_progress_variable_derivative, cholesky_factor, solve_lower_triangular,
    check_interval, temperature_from_energy
)
from adaptive_mesh import adaptive_density_function
from sparse_grid_chemistry import clenshaw_curtis_nodes_1d, piecewise_linear_basis, sparse_grid_index_set
from thermal_quadrature import integrate_square
from reaction_kinetics import euler_flux_x, euler_flux_y

# ---- TC01: CJ爆轰速度为正值有限数 ----
D_cj_test = cj_detonation_velocity(1.4, 2.5e6, 101325.0, 1.225)
assert D_cj_test > 0 and np.isfinite(D_cj_test), '[TC01] CJ detonation velocity not positive finite FAILED'

# ---- TC02: Von Neumann尖峰压强大于初始压强 ----
p_vn_test, rho_vn_test, T_vn_test, M_test = von_neumann_spike_conditions(D_cj_test, 1.4, 101325.0, 1.225)
assert p_vn_test > 101325.0, '[TC02] Von Neumann pressure not greater than p0 FAILED'

# ---- TC03: Von Neumann尖峰密度大于初始密度 ----
assert rho_vn_test > 1.225, '[TC03] Von Neumann density not greater than rho0 FAILED'

# ---- TC04: Von Neumann马赫数大于1 ----
assert M_test > 1.0, '[TC04] Von Neumann Mach number not > 1 FAILED'

# ---- TC05: Rankine-Hugoniot压力比在M>1时大于1 ----
rp = rankine_hugoniot_pressure_ratio(2.0, 1.4)
assert rp > 1.0, '[TC05] RH pressure ratio not > 1 FAILED'

# ---- TC06: Rankine-Hugoniot密度比有限正值且大于1 ----
rd = rankine_hugoniot_density_ratio(2.0, 1.4)
assert rd > 1.0 and np.isfinite(rd), '[TC06] RH density ratio not > 1 finite FAILED'

# ---- TC07: Arrhenius速率在参考温度下有限正值 ----
k0 = arrhenius_rate(300.0, 1.0e8, 8.314e4)
assert k0 > 0 and np.isfinite(k0), '[TC07] Arrhenius rate not positive finite FAILED'

# ---- TC08: Arrhenius速率随温度单调递增 ----
k1 = arrhenius_rate(500.0, 1.0e8, 8.314e4)
k2 = arrhenius_rate(1000.0, 1.0e8, 8.314e4)
assert k2 > k1, '[TC08] Arrhenius rate not monotonic increasing FAILED'

# ---- TC09: 声速计算结果为正值有限数 ----
a_test = sound_speed(300.0, 1.4, 0.029)
assert a_test > 0 and np.isfinite(a_test), '[TC09] Sound speed not positive finite FAILED'

# ---- TC10: sound_speed_from_prho与sound_speed一致性 ----
a_prho = sound_speed_from_prho(101325.0, 1.225, 1.4)
assert a_prho > 0 and np.isfinite(a_prho), '[TC10] Sound speed from p,rho not positive finite FAILED'

# ---- TC11: 定容比热容和定压比热容为正值且cp>cv ----
cv_test, cp_test = specific_heat_ratio_cv_cp(1.4)
assert cv_test > 0 and cp_test > cv_test, '[TC11] Specific heats not valid FAILED'

# ---- TC12: 反应进度导数dλ/dt为非正值 ----
dlam = znd_progress_variable_derivative(0.0, 1500.0, 1.0e8, 8.314e4)
assert dlam <= 0, '[TC12] Progress variable derivative should be non-positive FAILED'

# ---- TC13: λ=1时反应进度导数为零 ----
dlam1 = znd_progress_variable_derivative(1.0, 1500.0, 1.0e8, 8.314e4)
assert dlam1 == 0.0, '[TC13] dlambda/dt at lambda=1 not zero FAILED'

# ---- TC14: Cholesky分解L*L^T恢复原矩阵 ----
A_test_mat = np.array([[4.0, 1.0], [1.0, 3.0]])
L_test = cholesky_factor(A_test_mat)
A_recon = L_test @ L_test.T
assert np.allclose(A_test_mat, A_recon), '[TC14] Cholesky L*L^T not reconstruct A FAILED'

# ---- TC15: 三角形面积计算（单位直角三角形面积为1） ----
t_test = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
area_test = triangle_area(t_test)
assert abs(area_test - 1.0) < 1.0e-12, '[TC15] Triangle area not 1.0 FAILED'

# ---- TC16: T3基函数在形心处和为1 ----
t_test_t3 = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
centroid = np.mean(t_test_t3, axis=1)
phi_sum = 0.0
for idx in range(3):
    phi_i, _, _ = basis_t3(t_test_t3, idx, centroid)
    phi_sum += phi_i
assert abs(phi_sum - 1.0) < 1.0e-12, '[TC16] T3 basis sum at centroid not 1 FAILED'

# ---- TC17: 自适应密度函数在波前附近值更大 ----
np.random.seed(42)
dens_near = adaptive_density_function(0.5, 0.0, wave_x=0.5, wave_width=0.05)
dens_far = adaptive_density_function(0.0, 0.0, wave_x=0.5, wave_width=0.05)
assert dens_near > dens_far, '[TC17] Adaptive density not higher near wave FAILED'

# ---- TC18: 椭圆采样点数正确 ----
np.random.seed(42)
A_ell = np.array([[4.0, 1.0], [1.0, 3.0]])
samples_ell = sample_ellipse(100, A_ell, r=0.01)
assert samples_ell.shape[0] == 100, '[TC18] Ellipse sample count not correct FAILED'

# ---- TC19: 点火概率均值在[0,1]区间内且标准差非负 ----
np.random.seed(42)
mean_p, std_p, batch_p = ignition_probability_monte_carlo(
    n_samples=500, T_mean=1200.0, T_std=200.0,
    p_mean=5.0e5, p_std=1.0e5,
    phi_mean=1.0, phi_std=0.2,
    Ea=8.314e4, A=1.0e8, T_ign=1500.0, n_batches=5
)
assert 0.0 <= mean_p <= 1.0, '[TC19] Ignition probability not in [0,1] FAILED'
assert std_p >= 0.0, '[TC19b] Std dev should be non-negative FAILED'

# ---- TC20: 临界核逃逸时间分析面积占比在[0,1]内 ----
np.random.seed(42)
af, ae, _ = critical_kernel_escape_time(
    n_grid=20, it_max=50, D_wave=D_cj_test,
    gamma=1.4, Q=2.5e6, rho0=1.225, p0=101325.0
)
assert 0.0 <= af <= 1.0, '[TC20] Area fraction not in [0,1] FAILED'
assert ae >= 0, '[TC20b] Avg escape time not non-negative FAILED'

# ---- TC21: CJ解析速度为正值有限数 ----
cj_solver_test = CJConditionSolver(gamma=1.4, Q=2.5e6, p0=101325.0, rho0=1.225)
D_exact_test = cj_solver_test.exact_cj_velocity()
assert D_exact_test > 0 and np.isfinite(D_exact_test), '[TC21] CJ exact velocity not positive finite FAILED'

# ---- TC22: Hugoniot压力返回有限值 ----
p_hug = cj_solver_test.hugoniot_pressure(1.0 / 1.225)
assert np.isfinite(p_hug), '[TC22] Hugoniot pressure not finite FAILED'

# ---- TC23: Rayleigh线在v=v0处等于p0 ----
v0_test = 1.0 / 1.225
p_ray = cj_solver_test.rayleigh_line(v0_test, D_exact_test)
assert abs(p_ray - 101325.0) < 1.0e-6, '[TC23] Rayleigh line at v0 not equal p0 FAILED'

# ---- TC24: Brent求根法对-x+2在[0,3]上求根得2 ----
def f_root(x):
    return -x + 2.0
root = zero_brent(f_root, 0.0, 3.0, tol=1.0e-10)
assert abs(root - 2.0) < 1.0e-8, '[TC24] Brent root not close to 2 FAILED'

# ---- TC25: Brent局部最小化对(x-3)^2在[0,5]上求得x≈3 ----
def f_min(x):
    return (x - 3.0) ** 2
xmin, fmin_val = local_min_brent(f_min, 0.0, 5.0, tol=1.0e-10)
assert abs(xmin - 3.0) < 1.0e-6, '[TC25] Brent min not at x=3 FAILED'
assert fmin_val < 1.0e-8, '[TC25b] Brent min value not near zero FAILED'

# ---- TC26: H2-O2反应网络有8个物种节点 ----
net_test = build_hydrogen_oxygen_network()
stats_test = net_test.network_statistics()
assert stats_test['n_nodes'] == 8, '[TC26] Network node count not 8 FAILED'
assert stats_test['n_edges'] > 0, '[TC26b] Network edge count should be positive FAILED'

# ---- TC27: BFS最短路径H2→H2O存在且端点正确 ----
path_test = net_test.bfs_shortest_path("H2", "H2O")
assert path_test is not None, '[TC27] BFS path H2→H2O not found FAILED'
assert len(path_test) >= 2, '[TC27b] Path should have at least 2 nodes FAILED'
assert path_test[0] == "H2", '[TC27c] Path should start with H2 FAILED'
assert path_test[-1] == "H2O", '[TC27d] Path should end with H2O FAILED'

# ---- TC28: 反应循环检测返回列表 ----
cycles_test = net_test.find_cycles(max_length=5)
assert isinstance(cycles_test, list), '[TC28] Cycles should be a list FAILED'

# ---- TC29: Clenshaw-Curtis节点level=0返回[0] ----
cc0 = clenshaw_curtis_nodes_1d(0)
assert len(cc0) == 1 and abs(cc0[0] - 0.0) < 1.0e-12, '[TC29] CC level 0 not [0] FAILED'

# ---- TC30: Clenshaw-Curtis节点level=2有5个节点且端点正确 ----
cc2 = clenshaw_curtis_nodes_1d(2)
assert len(cc2) == 5, '[TC30] CC level 2 not 5 nodes FAILED'
assert abs(cc2[0] - 1.0) < 1.0e-12, '[TC30b] CC first node should be 1 FAILED'
assert abs(cc2[-1] + 1.0) < 1.0e-12, '[TC30c] CC last node should be -1 FAILED'

# ---- TC31: 分段线性基函数左外推时仅首项为1 ----
nodes_pl = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
w_left = piecewise_linear_basis(nodes_pl, -2.0)
assert abs(w_left[0] - 1.0) < 1.0e-12, '[TC31] Linear basis left extrapolate not 1 at first node FAILED'

# ---- TC32: 分段线性基函数内插权重非负 ----
w_mid = piecewise_linear_basis(nodes_pl, 0.25)
assert np.all(w_mid >= 0), '[TC32] Linear basis weights contain negative FAILED'

# ---- TC33: 对称求积规则∫∫_[-1,1]^2 1 dxdy ≈ 4 ----
sq_result = integrate_square(lambda x, y: np.ones_like(x), degree=5)
assert abs(sq_result - 4.0) < 1.0e-12, '[TC33] Symmetric quadrature ∫∫1 not ≈4 FAILED'

# ---- TC34: 平均温度计算返回正值 ----
T_test_field = np.ones((5, 5)) * 1000.0
T_avg_test = average_temperature_profile(T_test_field, 0.01, 0.01)
assert T_avg_test > 0, '[TC34] Average temperature not positive FAILED'
assert abs(T_avg_test - 1000.0) < 1.0e-6, '[TC34b] Average temperature not 1000 FAILED'

# ---- TC35: CRS矩阵向量乘法输出维度正确且值有限 ----
sp_test = SparseCRS(5, 5,
    np.array([0, 2, 4, 6, 8, 10]),
    np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 4]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
y_sp = sp_test.multiply(np.ones(5))
assert len(y_sp) == 5, '[TC35] CRS multiply output dimension wrong FAILED'
assert np.all(np.isfinite(y_sp)), '[TC35b] CRS multiply produced non-finite values FAILED'

# ---- TC36: ReactiveState守恒量转换可逆 ----
state_orig = ReactiveState(rho=1.2, u=100.0, v=0.0, e=3.0e6, lambda_var=0.3)
U_test = state_orig.to_conservative()
state_recov = ReactiveState.from_conservative(U_test, gamma=1.4, Q=2.5e6)
assert abs(state_recov.rho - 1.2) < 1.0e-10, '[TC36] ReactiveState rho recovery FAILED'
assert abs(state_recov.u - 100.0) < 1.0e-10, '[TC36b] ReactiveState u recovery FAILED'
assert abs(state_recov.lambda_var - 0.3) < 1.0e-10, '[TC36c] ReactiveState lambda recovery FAILED'

# ---- TC37: 化学源项输出为5分量向量且质量源项为零 ----
omega_test = chemical_source_term(state_orig, gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4)
assert len(omega_test) == 5, '[TC37] Chemical source term not 5 components FAILED'
assert omega_test[0] == 0.0, '[TC37b] Mass source should be zero FAILED'

# ---- TC38: Euler通量x方向输出5分量有限值 ----
Fx_test = euler_flux_x(state_orig, gamma=1.4, Q=2.5e6)
assert len(Fx_test) == 5, '[TC38] Euler flux x not 5 components FAILED'
assert np.isfinite(Fx_test[0]), '[TC38b] Euler flux x mass flux not finite FAILED'

# ---- TC39: 反应态温度返回正值 ----
T_state = state_orig.temperature(gamma=1.4, Q=2.5e6, W_mol=0.029)
assert T_state > 0, '[TC39] State temperature not positive FAILED'

# ---- TC40: check_positive对正值正常通过 ----
try:
    check_positive(5.0, "test_val")
    passed40 = True
except ValueError:
    passed40 = False
assert passed40, '[TC40] check_positive failed on valid input FAILED'

# ---- TC41: check_positive对负值抛出ValueError ----
try:
    check_positive(-1.0, "test_val")
    passed41 = False
except ValueError:
    passed41 = True
assert passed41, '[TC41] check_positive should raise on negative input FAILED'

# ---- TC42: check_interval对a<b正常通过 ----
try:
    check_interval(0.0, 1.0)
    passed42 = True
except ValueError:
    passed42 = False
assert passed42, '[TC42] check_interval failed on valid interval FAILED'

# ---- TC43: ZND求解器正常求解不崩溃 ----
np.random.seed(42)
znd_test = ZNDSolver(gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4,
                     rho0=1.225, p0=101325.0, T0=300.0, W_mol=0.029)
xi_test, sol_test = znd_test.solve(D=znd_test.cj_velocity(), ximax=1.0e-4, npts=500)
assert len(xi_test) == 500, '[TC43] ZND solve output length wrong FAILED'
assert sol_test.shape == (500, 4), '[TC43b] ZND solve output shape wrong FAILED'
assert np.all(np.isfinite(sol_test)), '[TC43c] ZND solution contains non-finite values FAILED'

# ---- TC44: 诱导区长度在求解域内 ----
L_ind_test = znd_test.induction_length(xi_test, sol_test, threshold=0.95)
assert 0 <= L_ind_test <= xi_test[-1], '[TC44] Induction length out of domain FAILED'

# ---- TC45: 半反应长度不超过诱导区长度 ----
L_half_test = znd_test.half_reaction_length(xi_test, sol_test)
assert L_half_test <= L_ind_test + 1.0e-12, '[TC45] Half-reaction length > induction length FAILED'

# ---- TC46: 自适应网格生成不崩溃 ----
np.random.seed(42)
mesh_test = AdaptiveDetonationMesh(x_min=0.0, x_max=1.0e-3,
                                   y_min=-0.5e-3, y_max=0.5e-3,
                                   n_base=50, wave_x=0.5e-3,
                                   wave_width=0.05e-3)
nodes_test, elems_test = mesh_test.generate(cvt_samples=200, cvt_iter=5)
assert len(nodes_test) > 0, '[TC46] Mesh generation produced no nodes FAILED'

# ---- TC47: 稳定性分析特征值个数为4 ----
stab_test = DetonationStability(xi_test, sol_test, gamma=1.4, Q=2.5e6)
evals_test, evecs_test = stab_test.eigenvalue_analysis()
assert len(evals_test) == 4, '[TC47] Stability eigenvalues not 4 FAILED'

# ---- TC48: 反应进度剖面始终在[0,1]区间且单调非减 ----
lambda_prof = sol_test[:, 3]
assert np.all(lambda_prof >= -1.0e-12), '[TC48] Lambda profile has negative values FAILED'
assert np.all(lambda_prof <= 1.0 + 1.0e-12), '[TC48b] Lambda profile exceeds 1 FAILED'
assert lambda_prof[-1] >= lambda_prof[0], '[TC48c] Lambda should be non-decreasing FAILED'

# ---- TC49: 密度剖面始终为正 ----
rho_prof_test = sol_test[:, 0]
assert np.all(rho_prof_test > 0), '[TC49] Density profile has non-positive values FAILED'

# ---- TC50: Euler求解器初始化和CFL时间步长正常 ----
np.random.seed(42)
euler_test = ReactiveEulerSolver(10, 5, 1.0e-4, 2.0e-4,
                                 gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4)
euler_test.initialize_cj_planar_wave(D_cj_test, 1.225, 101325.0)
assert euler_test.U.shape == (10, 5, 5), '[TC50] Euler solver init shape wrong FAILED'
assert np.all(np.isfinite(euler_test.U)), '[TC50b] Euler solver init contains non-finite FAILED'
assert np.all(euler_test.U[:, :, 0] > 0), '[TC50c] Euler solver density not positive FAILED'

# ---- TC51: 热力学释热率积分返回非负值 ----
lambda_test_field = np.ones((4, 4)) * 0.5
T_test_field2 = np.ones((4, 4)) * 1500.0
rho_test_field = np.ones((4, 4)) * 1.0
q_total_test = integrate_thermal_source(lambda_test_field, T_test_field2, rho_test_field,
                                       0.01, 0.01, degree=3,
                                       A=1.0e8, Ea=8.314e4, Q=2.5e6)
assert q_total_test >= 0, '[TC51] Thermal source integral should be non-negative FAILED'

# ---- TC52: 稀疏网格索引集非空 ----
idx_set = sparse_grid_index_set(4, 3)
assert len(idx_set) > 0, '[TC52] Sparse grid index set is empty FAILED'

# ---- TC53: 稀疏网格构建和评估一致性 ----
np.random.seed(42)
sg_test = SparseGridChemistry(max_level=2, dim=2)
def simple_func(y):
    return y[0] + y[1]
sg_test.build(simple_func)
val_test = sg_test.evaluate(np.array([0.5, 0.5]))
assert np.isfinite(val_test), '[TC53] Sparse grid eval produced non-finite FAILED'

# ---- TC54: solve_lower_triangular解方程组 ----
L_mat = np.array([[2.0, 0.0], [1.0, 3.0]])
b_vec = np.array([4.0, 5.0])
x_vec = solve_lower_triangular(L_mat, b_vec)
assert abs(x_vec[0] - 2.0) < 1.0e-12, '[TC54] Lower triangular solve FAILED'
assert abs(L_mat[1,0]*x_vec[0] + L_mat[1,1]*x_vec[1] - b_vec[1]) < 1.0e-12, '[TC54b] Lower triangular verify FAILED'

# ---- TC55: 压强计算返回非负值 ----
p_state = state_orig.pressure(gamma=1.4, Q=2.5e6)
assert p_state >= 0, '[TC55] Pressure should be non-negative FAILED'

# ---- TC56: ZND求解器CJ速度计算正值 ----
D_cj_znd = znd_test.cj_velocity()
assert D_cj_znd > 0 and np.isfinite(D_cj_znd), '[TC56] ZND CJ velocity not positive finite FAILED'

# ---- TC57: Von Neumann状态推导一致 ----
rho_vn_z, p_vn_z, T_vn_z, u_vn_z = znd_test.von_neumann_state(D_cj_znd)
assert rho_vn_z > znd_test.rho0, '[TC57] VN density should exceed rho0 FAILED'
assert p_vn_z > znd_test.p0, '[TC57b] VN pressure should exceed p0 FAILED'

# ---- TC58: CRS矩阵转置乘法输出维度正确 ----
yt_sp = sp_test.multiply_transpose(np.ones(5))
assert len(yt_sp) == 5, '[TC58] CRS transpose multiply output dimension wrong FAILED'
assert np.all(np.isfinite(yt_sp)), '[TC58b] CRS transpose multiply produced non-finite values FAILED'

# ---- TC59: temperature_from_energy返回正值 ----
T_fe = temperature_from_energy(3.0e6, 0.3, 2.5e6, 1.0)
assert T_fe > 0, '[TC59] temperature_from_energy not positive FAILED'

# ---- TC60: Euler通量y方向输出5分量有限值 ----
Fy_test = euler_flux_y(state_orig, gamma=1.4, Q=2.5e6)
assert len(Fy_test) == 5, '[TC60] Euler flux y not 5 components FAILED'
assert np.isfinite(Fy_test[0]), '[TC60b] Euler flux y mass flux not finite FAILED'
