# ---- TC01: PINNNetwork (gaussian_rbf) forward 输出形状正确 ----
net_rbf = PINNNetwork(input_dim=2, hidden_dims=[16, 16], output_dim=1, activation='gaussian_rbf', rbf_scale=1.0, seed=42)
X_test = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
Y = net_rbf.forward(X_test)
assert Y.shape == (3, 1), '[TC01] gaussian_rbf forward shape FAILED'
assert np.all(np.isfinite(Y)), '[TC01] gaussian_rbf forward non-finite FAILED'

# ---- TC02: PINNNetwork (tanh) forward 输出形状正确 ----
net_tanh = PINNNetwork(input_dim=2, hidden_dims=[24, 24], output_dim=1, activation='tanh', seed=123)
X_test2 = np.array([[0.0, 0.0], [1.0, 2.0]])
Y2 = net_tanh.forward(X_test2)
assert Y2.shape == (2, 1), '[TC02] tanh forward shape FAILED'
assert np.all(np.isfinite(Y2)), '[TC02] tanh forward non-finite FAILED'

# ---- TC03: PINNNetwork parameter_count 返回正整数 ----
P_rbf = net_rbf.parameter_count()
P_tanh = net_tanh.parameter_count()
assert P_rbf > 0, '[TC03] rbf param count non-positive FAILED'
assert P_tanh > 0, '[TC03] tanh param count non-positive FAILED'
assert isinstance(P_rbf, int), '[TC03] param count not int FAILED'

# ---- TC04: PINNNetwork get/set params 往返一致性 ----
params_orig = net_rbf.get_params_flat()
P = len(params_orig)
net_rbf.set_params_flat(params_orig)
params_roundtrip = net_rbf.get_params_flat()
assert np.allclose(params_orig, params_roundtrip), '[TC04] get/set roundtrip FAILED'

# ---- TC05: PINNNetwork finite_difference_derivatives 输出形状 ----
du_dt = net_rbf.finite_difference_derivatives(X_test, var_idx=0)
du_dx = net_rbf.finite_difference_derivatives(X_test, var_idx=1)
assert du_dt.shape == (3, 1), '[TC05] dt derivative shape FAILED'
assert du_dx.shape == (3, 1), '[TC05] dx derivative shape FAILED'
assert np.all(np.isfinite(du_dt)), '[TC05] dt derivative non-finite FAILED'

# ---- TC06: PINNNetwork second_derivative 输出形状 ----
d2u_dx2 = net_rbf.second_derivative(X_test, var_idx=1)
assert d2u_dx2.shape == (3, 1), '[TC06] second derivative shape FAILED'
assert np.all(np.isfinite(d2u_dx2)), '[TC06] second derivative non-finite FAILED'

# ---- TC07: PINNNetwork fourth_derivative 输出形状 ----
d4u_dx4 = net_rbf.fourth_derivative(X_test, var_idx=1)
assert d4u_dx4.shape == (3, 1), '[TC07] fourth derivative shape FAILED'
assert np.all(np.isfinite(d4u_dx4)), '[TC07] fourth derivative non-finite FAILED'

# ---- TC08: ETDRK4 solver 返回正确形状 ----
x_sol, t_sol, u_sol, k_sol, L_op = solve_ks_etdrk4(nx=32, tmax=2.0, dt=0.25, n_snapshots=9)
assert u_sol.shape == (32, 9), '[TC08] u_sol shape FAILED'
assert len(x_sol) == 32, '[TC08] x_sol length FAILED'
assert len(t_sol) == 9, '[TC08] t_sol length FAILED'
assert np.all(np.isfinite(u_sol)), '[TC08] u_sol non-finite FAILED'

# ---- TC09: ks_reference_residual 输出形状 ----
u_test = u_sol
res = ks_reference_residual(u_test, x_sol, t_sol, k_sol)
assert res.shape == u_test.shape, '[TC09] residual shape mismatch FAILED'
assert np.all(np.isfinite(res)), '[TC09] residual non-finite FAILED'

# ---- TC10: compute_total_loss 返回浮点数与正确键值 ----
net_p = PINNNetwork(input_dim=2, hidden_dims=[8, 8], output_dim=1, activation='tanh', seed=99)
X_f = np.random.default_rng(42).uniform(0, 1, size=(20, 2))
X_ic = np.random.default_rng(42).uniform(0, 1, size=(10, 2))
u_ic_target = np.zeros(10)
X_bc_0 = np.random.default_rng(42).uniform(0, 1, size=(5, 2))
X_bc_L = np.random.default_rng(42).uniform(0, 1, size=(5, 2))
loss_total, loss_dict = compute_total_loss(
    net_p, X_f, X_ic, u_ic_target, X_bc_0, X_bc_L,
    lambda_pde=1.0, lambda_ic=10.0, lambda_bc=5.0
)
assert np.isscalar(loss_total), '[TC10] total loss not scalar FAILED'
assert loss_total >= 0, '[TC10] total loss negative FAILED'
assert np.isfinite(loss_total), '[TC10] total loss non-finite FAILED'
assert 'pde' in loss_dict, '[TC10] pde key missing FAILED'
assert 'ic' in loss_dict, '[TC10] ic key missing FAILED'
assert 'bc' in loss_dict, '[TC10] bc key missing FAILED'

# ---- TC11: generate_collocation_grid 输出形状 ----
tmax_p, L_p = 2.0, 32.0 * np.pi
X_grid, t_grid, x_grid = generate_collocation_grid(tmax_p, L_p, nt=8, nx=16)
assert X_grid.shape == (8 * 16, 2), '[TC11] collocation grid shape FAILED'
assert np.all(np.isfinite(X_grid)), '[TC11] collocation grid non-finite FAILED'

# ---- TC12: generate_boundary_points 输出形状与配对 ----
X_b0, X_bL = generate_boundary_points(tmax_p, L_p, 10)
assert X_b0.shape == (10, 2), '[TC12] X_bc_0 shape FAILED'
assert X_bL.shape == (10, 2), '[TC12] X_bc_L shape FAILED'
assert np.allclose(X_b0[:, 1], 0.0), '[TC12] bc_0 x not zero FAILED'
assert np.allclose(X_bL[:, 1], L_p), '[TC12] bc_L x not L FAILED'

# ---- TC13: triangulation_boundary_edges 提取正方形边界 ----
nodes = np.array([[0,0],[1,0],[1,1],[0,1]])
tris = np.array([[0,1,2],[0,2,3]])
b_edges = triangulation_boundary_edges(tris)
assert len(b_edges) == 4, '[TC13] boundary edge count FAILED'

# ---- TC14: find_nearest_neighbors 输出形状与单调性 ----
ref_pts = np.random.default_rng(42).uniform(0, 1, size=(30, 2))
qry_pts = np.random.default_rng(42).uniform(0, 1, size=(8, 2))
idx_nn, dist_nn = find_nearest_neighbors(ref_pts, qry_pts)
assert len(idx_nn) == 8, '[TC14] nearest idx length FAILED'
assert len(dist_nn) == 8, '[TC14] nearest dist length FAILED'
assert np.all(dist_nn >= 0), '[TC14] nearest dist negative FAILED'

# ---- TC15: Gauss-Legendre 积分 sin(x) 于 [0,pi] 精度 ----
x_gl, w_gl = gauss_legendre_1d(n=7, a=0.0, b=np.pi)
integral_gl = np.sum(w_gl * np.sin(x_gl))
assert abs(integral_gl - 2.0) < 1e-10, '[TC15] Gauss-Legendre sin integral FAILED'

# ---- TC16: Kronrod nodes/weights 输出 ----
x_kr, w_kr, w_g_kr = kronrod_nodes_weights(n=7)
assert len(x_kr) == 15, '[TC16] Kronrod nodes count FAILED'
assert len(w_kr) == 15, '[TC16] Kronrod weights count FAILED'
assert len(w_g_kr) == 15, '[TC16] Gauss embedded weights count FAILED'
assert abs(np.sum(w_kr) - 2.0) < 1e-12, '[TC16] Kronrod weights sum FAILED'

# ---- TC17: compute_wavenumbers 输出范围 ----
k_wave = compute_wavenumbers(nx=32, L_domain=32.0 * np.pi)
assert len(k_wave) == 32, '[TC17] wavenumber count FAILED'
assert k_wave[0] == 0.0, '[TC17] wavenumber k0 non-zero FAILED'

# ---- TC18: spectral_derivative 对 sin(x) 求导得 cos(x) ----
from spectral_ops import spectral_derivative
x_sp = np.linspace(0, 32.0 * np.pi, 64, endpoint=False)
u_sin = np.sin(x_sp)
k_sp = compute_wavenumbers(64, 32.0 * np.pi)
du_dx_spec = spectral_derivative(u_sin, k_sp, order=1)
cos_expected = np.cos(x_sp)
err_cos = np.max(np.abs(du_dx_spec.real - cos_expected))
assert err_cos < 1e-6, '[TC18] spectral derivative sin->cos FAILED'

# ---- TC19: compute_energy_spectrum 输出非负 ----
k_espec, E_spec = compute_energy_spectrum(u_sin, 32.0 * np.pi)
assert np.all(E_spec >= 0), '[TC19] energy spectrum negative FAILED'
assert len(E_spec) == 64, '[TC19] energy spectrum length FAILED'

# ---- TC20: kolmogorov_length_scale 输出正值 ----
eta = kolmogorov_length_scale(u_sin, 32.0 * np.pi)
assert eta > 0, '[TC20] Kolmogorov scale non-positive FAILED'
assert np.isfinite(eta), '[TC20] Kolmogorov scale non-finite FAILED'

# ---- TC21: squircle_trajectory 输出有限 ----
from chaos_utils import squircle_trajectory
t_sq, xy_sq = squircle_trajectory(s=4.0, t0=0.0, y0=np.array([1.0, 0.0]), tstop=10.0, n_points=200)
assert np.all(np.isfinite(xy_sq)), '[TC21] squircle trajectory non-finite FAILED'
assert xy_sq.shape == (200, 2), '[TC21] squircle shape FAILED'

# ---- TC22: cross_chaos_ifs 输出范围在有限区间 ----
from chaos_utils import cross_chaos_ifs
xy_ifs = cross_chaos_ifs(n_points=500, seed=42)
assert xy_ifs.shape == (500, 2), '[TC22] cross_ifs shape FAILED'
assert np.all(np.isfinite(xy_ifs)), '[TC22] cross_ifs non-finite FAILED'

# ---- TC23: cellular_automaton_rule30 输出形状 ----
from chaos_utils import cellular_automaton_rule30
ca = cellular_automaton_rule30(cell_num=32, step_num=16, seed_center=10)
assert ca.shape == (16, 32), '[TC23] CA rule30 shape FAILED'
assert np.all((ca == 0) | (ca == 1)), '[TC23] CA not binary FAILED'

# ---- TC24: manufactured_solution_1 输出有限 ----
u_ms1 = manufactured_solution_1(t=1.0, x=10.0)
assert np.isfinite(u_ms1), '[TC24] MS1 non-finite FAILED'
u_ms1_arr = manufactured_solution_1(t=np.array([0.0, 0.5, 1.0]), x=np.array([0.0, 1.0, 2.0]))
assert np.all(np.isfinite(u_ms1_arr)), '[TC24] MS1 array non-finite FAILED'

# ---- TC25: compute_pin_error 返回所有必需键 ----
u_a = np.array([1.0, 2.0, 3.0])
u_b = np.array([1.1, 1.9, 3.2])
err_pin = compute_pin_error(u_a, u_b)
for k in ['l2_abs', 'l2_rel', 'linf_abs', 'linf_rel', 'mse']:
    assert k in err_pin, f'[TC25] error key {k} missing FAILED'
assert err_pin['mse'] > 0, '[TC25] mse zero for mismatched arrays FAILED'

# ---- TC26: compute_pairwise_distance 对称性 ----
from rbf_kernel import compute_pairwise_distance
X_rbf1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
D = compute_pairwise_distance(X_rbf1, X_rbf1)
assert D.shape == (3, 3), '[TC26] pairwise distance shape FAILED'
assert np.allclose(D, D.T), '[TC26] pairwise distance not symmetric FAILED'
assert np.all(np.diag(D) == 0.0), '[TC26] self-distance non-zero FAILED'
assert np.all(D >= 0), '[TC26] distance negative FAILED'

# ---- TC27: RBF interpolation 精确重构 ----
X_data = np.linspace(-1, 1, 10).reshape(-1, 1)
f_data = np.sin(X_data.ravel())
w_rbf, cond_rbf = rbf_interpolation_weights(X_data, f_data, r0=0.5, phi_type='gaussian')
X_qry = X_data.copy()
f_rbf = rbf_interpolate(X_data, w_rbf, r0=0.5, X_query=X_qry, phi_type='gaussian')
err_rbf = np.max(np.abs(f_rbf - f_data))
assert err_rbf < 1e-6, '[TC27] RBF exact reconstruction FAILED'

# ---- TC28: CosineAnnealingScheduler 输出在 [eta_min, eta_max] ----
sched = CosineAnnealingScheduler(eta_max=0.1, eta_min=1e-5, T_max=100)
lr_0 = sched.get_lr(0)
lr_50 = sched.get_lr(50)
lr_99 = sched.get_lr(99)
lr_100 = sched.get_lr(100)
assert abs(lr_0 - 0.1) < 1e-12, '[TC28] LR not max at t=0 FAILED'
assert lr_50 < 0.1 and lr_50 > 1e-5, '[TC28] LR not decreasing at mid FAILED'
assert abs(lr_99 - 1e-5) > 1e-9, '[TC28] LR hit min too early FAILED'
assert abs(lr_100 - 1e-5) < 1e-12, '[TC28] LR not min at T_max FAILED'

# ---- TC29: SGDWithMomentum 单步更新改变参数 ----
from stochastic_optimizer import SGDWithMomentum
sgd = SGDWithMomentum(params_dim=4, lr=0.1, momentum=0.0, lr_decay=1.0, min_lr=0.0)
p0 = np.array([1.0, 2.0, 3.0, 4.0])
grad = np.array([0.1, 0.2, 0.3, 0.4])
p1 = sgd.step(p0, grad)
assert not np.allclose(p0, p1), '[TC29] SGD step did not change params FAILED'

# ---- TC30: checkpoint save/load 往返一致性 ----
from data_io import checkpoint_save, checkpoint_load
net_ck = PINNNetwork(input_dim=2, hidden_dims=[4, 4], output_dim=1, activation='tanh', seed=99)
params_before = net_ck.get_params_flat().copy()
import tempfile, os
tmpdir = tempfile.mkdtemp()
ck_path = os.path.join(tmpdir, 'test_checkpoint.txt')
try:
    checkpoint_save(net_ck, ck_path)
    net_ck_loaded = PINNNetwork(input_dim=2, hidden_dims=[4, 4], output_dim=1, activation='tanh', seed=999)
    checkpoint_load(net_ck_loaded, ck_path)
    params_after = net_ck_loaded.get_params_flat()
    assert np.allclose(params_before, params_after), '[TC30] checkpoint roundtrip FAILED'
finally:
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

# ---- TC31: parse_variable_line 解析正确 ----
from data_io import parse_variable_line
parsed = parse_variable_line("t=0.5 x=3.14 residual=0.001")
assert 't' in parsed and parsed['t'] == 0.5, '[TC31] parse t FAILED'
assert 'x' in parsed and parsed['x'] == 3.14, '[TC31] parse x FAILED'
assert 'residual' in parsed and parsed['residual'] == 0.001, '[TC31] parse residual FAILED'

# ---- TC32: adaptive_refinement_sample 输出形状 ----
X_curr = np.random.default_rng(42).uniform(0, 1, size=(50, 2))
r_vals = np.random.default_rng(42).random(50)
X_new = adaptive_refinement_sample(net_p, X_curr, r_vals, n_add=20, threshold_percentile=70)
assert X_new.shape[0] == 20, '[TC32] refinement sample count FAILED'
assert X_new.shape[1] == 2, '[TC32] refinement sample dim FAILED'

# ---- TC33: multi_level_grid_refinement 各层网格形状 ----
from adaptive_sampler import multi_level_grid_refinement
grids = multi_level_grid_refinement(tmax=1.0, L_domain=10.0, base_nt=4, base_nx=8, levels=3)
assert len(grids) == 3, '[TC33] grid level count FAILED'
for lv, grid in enumerate(grids):
    expected_rows = (4 * (2**lv)) * (8 * (2**lv))
    assert grid.shape[0] == expected_rows, f'[TC33] level {lv} grid rows FAILED'
    assert grid.shape[1] == 2, f'[TC33] level {lv} grid cols FAILED'

# ---- TC34: RBFKernelLayer forward 输出形状 ----
from rbf_kernel import RBFKernelLayer
X_rbf_in = np.random.default_rng(42).normal(size=(10, 2))
rbf_layer = RBFKernelLayer(n_centers=5, input_dim=2, r0=1.0, phi_type='gaussian', learnable_centers=False, seed=42)
Y_rbf = rbf_layer.forward(X_rbf_in)
assert Y_rbf.shape == (10, 1), '[TC34] RBFKernelLayer forward shape FAILED'
assert np.all(np.isfinite(Y_rbf)), '[TC34] RBFKernelLayer forward non-finite FAILED'

# ---- TC35: 积分测试 — 完整 Manufactured Solution 小规模训练验证 ----
net_ms_test = PINNNetwork(input_dim=2, hidden_dims=[8, 8], output_dim=1, activation='gaussian_rbf', rbf_scale=1.0, seed=42)
for i in range(net_ms_test.n_layers):
    net_ms_test.weights[i] *= 0.2
params_ms = net_ms_test.get_params_flat()
P_ms = len(params_ms)
t_train = np.linspace(0.0, 1.0, 4)
x_train = np.linspace(0.0, 10.0, 8, endpoint=False)
Tg_ms, Xg_ms = np.meshgrid(t_train, x_train, indexing='ij')
X_tr = np.column_stack([Tg_ms.ravel(), Xg_ms.ravel()])
u_ex = manufactured_solution_1(X_tr[:, 0], X_tr[:, 1])
prev_loss = None
for it in range(5):
    h = 1e-4
    u_center = net_ms_test.predict(X_tr).ravel()
    loss_center = np.mean((u_center - u_ex) ** 2)
    for pidx in range(P_ms):
        params_plus = params_ms.copy()
        params_plus[pidx] += h
        net_ms_test.set_params_flat(params_plus)
        u_plus = net_ms_test.predict(X_tr).ravel()
        loss_plus = np.mean((u_plus - u_ex) ** 2)
        grad_val = (loss_plus - loss_center) / h
        params_ms[pidx] -= 0.1 * grad_val
        net_ms_test.set_params_flat(params_ms)
    if prev_loss is not None:
        pass  # loss should generally decrease
    prev_loss = loss_center
u_pred_ms = net_ms_test.predict(X_tr).ravel()
err_ms = compute_pin_error(u_pred_ms, u_ex)
assert err_ms['mse'] >= 0, '[TC35] integration MSE negative FAILED'
assert np.isfinite(err_ms['l2_abs']), '[TC35] integration L2 non-finite FAILED'
