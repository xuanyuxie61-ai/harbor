import numpy as np

# ---- TC01: safe_divide 正常除法返回正确值 ----
result = safe_divide(10.0, 2.0, fallback=0.0)
assert abs(result - 5.0) < 1e-12, '[TC01] safe_divide normal division FAILED'

# ---- TC02: safe_divide 除零时返回 fallback ----
result = safe_divide(5.0, 0.0, fallback=999.0)
assert result == 999.0, '[TC02] safe_divide zero fallback FAILED'

# ---- TC03: clip_spin_norm 退化零矢量对齐 z 轴 ----
from utils import clip_spin_norm
s_zero = np.array([0.0, 0.0, 0.0])
s_clip = clip_spin_norm(s_zero, target_norm=1.0)
assert np.allclose(s_clip, np.array([0.0, 0.0, 1.0])), '[TC03] clip_spin_norm degenerate FAILED'

# ---- TC04: rms_norm 空数组返回零 ----
assert rms_norm(np.array([])) == 0.0, '[TC04] rms_norm empty array FAILED'

# ---- TC05: skyline_mv 与稠密矩阵乘法一致 ----
np.random.seed(42)
L5 = exchange_laplacian_1d(n=8, h=0.1, bc="dirichlet")
na, diag_idx, a_sky = build_skyline_from_tridiagonal(np.diag(L5, -1), np.diag(L5), np.diag(L5, 1))
x_vec = np.random.randn(8)
y_sky = skyline_mv(8, diag_idx, a_sky, x_vec)
y_dense = L5 @ x_vec
assert np.allclose(y_sky, y_dense), '[TC05] skyline_mv consistency FAILED'

# ---- TC06: is_prime 正确检测素数与非素数 ----
from utils import is_prime
assert is_prime(17) == True, '[TC06] is_prime prime check FAILED'
assert is_prime(1) == False, '[TC06] is_prime non-prime check FAILED'

# ---- TC07: exchange_laplacian_1d Dirichlet 对称正定 ----
L_dir = exchange_laplacian_1d(n=5, h=1.0, bc="dirichlet")
assert np.allclose(L_dir, L_dir.T), '[TC07] Laplacian symmetry FAILED'
assert np.all(np.diag(L_dir) > 0), '[TC07] Laplacian positive diagonal FAILED'

# ---- TC08: exchange_laplacian_1d periodic 行和为零 ----
L_per = exchange_laplacian_1d(n=6, h=0.5, bc="periodic")
assert np.allclose(np.sum(L_per, axis=1), 0.0), '[TC08] periodic Laplacian row sum FAILED'

# ---- TC09: exchange_energy Ising 标量情形 ----
J_ising = np.array([[0.0, 1.0], [1.0, 0.0]])
spins_ising = np.array([1.0, -1.0])
e_ising = exchange_energy(J_ising, spins_ising)
assert abs(e_ising - (-1.0)) < 1e-10, '[TC09] exchange energy Ising FAILED'

# ---- TC10: apply_exchange_operator Heisenberg 输出形状与值 ----
J_id = np.eye(4)
spins_heis = np.ones((4, 3))
H_eff = apply_exchange_operator(J_id, spins_heis)
assert H_eff.shape == (4, 3), '[TC10] apply_exchange shape FAILED'
assert np.allclose(H_eff, spins_heis), '[TC10] apply_exchange identity FAILED'

# ---- TC11: q_multiply 单位四元数保持右操作数 ----
from spin_quaternion import q_multiply, q_normalize
q_i = np.array([1.0, 0.0, 0.0, 0.0])
q_j = np.array([0.0, 1.0, 0.0, 0.0])
q_res = q_multiply(q_i, q_j)
assert np.allclose(q_res, q_j), '[TC11] q_multiply identity FAILED'

# ---- TC12: q_to_rotation_matrix 与 rotation_matrix_to_q 互逆 ----
from spin_quaternion import q_to_rotation_matrix, rotation_matrix_to_q, random_spin_quaternion
np.random.seed(42)
q_orig = random_spin_quaternion(seed=123)
R_mat = q_to_rotation_matrix(q_orig)
q_back = rotation_matrix_to_q(R_mat)
dot_abs = abs(np.dot(q_normalize(q_orig), q_normalize(q_back)))
assert dot_abs > 0.999, '[TC12] quat-rotmat roundtrip FAILED'

# ---- TC13: q_rotate_vector 保持矢量长度不变 ----
from spin_quaternion import q_rotate_vector, axis_angle_to_q
q_rot = axis_angle_to_q(np.array([0.0, 0.0, 1.0]), np.pi / 4.0)
v_in = np.array([2.0, 3.0, 4.0])
v_out = q_rotate_vector(q_rot, v_in)
assert np.allclose(np.linalg.norm(v_out), np.linalg.norm(v_in)), '[TC13] rotation preserves norm FAILED'

# ---- TC14: random_spin_quaternion 固定种子结果可复现 ----
np.random.seed(42)
q_a = random_spin_quaternion(seed=77)
q_b = random_spin_quaternion(seed=77)
assert np.allclose(q_a, q_b), '[TC14] random quaternion reproducibility FAILED'

# ---- TC15: local_min_brent 精确求 x^2 极小 ----
from energy_landscape import local_min_brent
theta_opt, e_min, calls = local_min_brent(lambda x: x * x, -2.0, 2.0)
assert abs(theta_opt) < 0.01, '[TC15] Brent min x^2 position FAILED'
assert abs(e_min) < 1e-4, '[TC15] Brent min x^2 value FAILED'

# ---- TC16: spin_wave_dispersion_1d k=0 频率严格为零 ----
k_vals = np.array([0.0, np.pi / 4, np.pi / 2])
omega_sw = spin_wave_dispersion_1d(J=1.0, S=1.0, a=1.0, k_points=k_vals)
assert abs(omega_sw[0]) < 1e-12, '[TC16] spin wave omega(0) FAILED'
assert omega_sw[-1] > omega_sw[0], '[TC16] spin wave monotonicity FAILED'

# ---- TC17: correlation_length_from_gap 零间隙发散为 inf ----
xi_inf = correlation_length_from_gap(0.0, J=1.0, a=1.0)
assert np.isinf(xi_inf), '[TC17] correlation length inf FAILED'

# ---- TC18: power_method 对角矩阵主导特征值精确 ----
np.random.seed(42)
A_diag = np.diag([1.0, 3.0, 7.0])
lam_dom, v_dom, iters_pm = power_method(A_diag, max_iter=300, tol=1e-10)
assert abs(lam_dom - 7.0) < 0.01, '[TC18] power method dominant eigenvalue FAILED'

# ---- TC19: spectral_gap_and_soft_modes 谱隙非负 ----
np.random.seed(42)
J_sym = np.random.randn(5, 5)
J_sym = (J_sym + J_sym.T) * 0.5
lam_min, gap_val, eigs_all, eigvecs = spectral_gap_and_soft_modes(J_sym, n_soft=2)
assert gap_val >= 0.0, '[TC19] spectral gap non-negative FAILED'

# ---- TC20: euler_integrate_llg 保持自旋单位范数 ----
np.random.seed(42)
J_llg = np.eye(3) * 0.05
spins_init = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
t_e, traj_e = euler_integrate_llg(J_llg, spins_init, t_span=(0.0, 0.2), n_steps=20, gamma=1.0, alpha=0.1)
norms_final = np.linalg.norm(traj_e[-1], axis=1)
assert np.allclose(norms_final, 1.0, atol=1e-5), '[TC20] Euler spin norm conservation FAILED'

# ---- TC21: fisher_kpp_domain_wall_exact 左右边界渐进行为 ----
from spin_dynamics import fisher_kpp_domain_wall_exact
x_bound = np.array([-15.0, 15.0])
u_kpp, ut_kpp, ux_kpp, uxx_kpp = fisher_kpp_domain_wall_exact(t=0.0, x=x_bound)
assert u_kpp[0] > 0.99, '[TC21] KPP left boundary FAILED'
assert u_kpp[-1] < 0.01, '[TC21] KPP right boundary FAILED'

# ---- TC22: connected_components_2d_spin_map 正确计数连通区域 ----
spin_map_test = np.array([[0.8, 0.8, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.9]])
labels_map = connected_components_2d_spin_map(spin_map_test, threshold=0.5)
assert int(labels_map.max()) == 2, '[TC22] connected components count FAILED'

# ---- TC23: solve_tridiagonal 与稠密求解器结果一致 ----
from adaptive_fem_solver import solve_tridiagonal
n_td = 5
aD_td = np.ones(n_td) * 2.0
aL_td = np.zeros(n_td)
aR_td = np.zeros(n_td)
aL_td[1:] = -1.0
aR_td[:-1] = -1.0
rhs_td = np.ones(n_td)
sol_td = solve_tridiagonal(aL_td, aD_td, aR_td, rhs_td)
A_full_td = np.diag(aD_td) + np.diag(aR_td[:-1], 1) + np.diag(aL_td[1:], -1)
sol_np_td = np.linalg.solve(A_full_td, rhs_td)
assert np.allclose(sol_td, sol_np_td), '[TC23] tridiagonal solve FAILED'

# ---- TC24: adaptive_fem_order_parameter 输出结构正确 ----
nodes_fem, sol_fem, edens_fem, hist_fem = adaptive_fem_order_parameter(
    A_func=lambda x: 1.0,
    B_func=lambda x: 0.5,
    F_func=lambda x: x,
    m_left=0.0,
    m_right=1.0,
    n_initial=4,
    max_refinements=2,
    error_threshold=0.1,
    max_nodes=50,
)
assert nodes_fem.size == sol_fem.size, '[TC24] FEM nodes-solution size FAILED'
assert len(hist_fem) > 0, '[TC24] FEM history non-empty FAILED'

# ---- TC25: integrate_brusselator_pump 输出时序形状正确 ----
t_p, y_p = integrate_brusselator_pump(a=1.0, b=3.0, n_steps=100)
assert y_p.shape == (101, 2), '[TC25] Brusselator output shape FAILED'
assert t_p.size == 101, '[TC25] Brusselator time size FAILED'

# ---- TC26: extract_domain_statistics 空图返回零域 ----
empty_spin_map = np.zeros((4, 4))
dom_stats = extract_domain_statistics(empty_spin_map, threshold=0.5)
assert dom_stats["n_domains"] == 0, '[TC26] empty domain stats FAILED'

# ---- TC27: histogram_stats_1d 空数组安全回退 ----
counts_h, edges_h, stats_h = histogram_stats_1d(np.array([]), bins=10)
assert stats_h["mean"] == 0.0, '[TC27] histogram empty mean FAILED'

# ---- TC28: triangle_area_histogram_2d 子区域数量正确 ----
pts_tri = np.array([[0.1, 0.1], [0.3, 0.2], [0.15, 0.05]])
histo_tri, info_tri = triangle_area_histogram_2d(pts_tri, n_sub=3)
assert histo_tri.size == 9, '[TC28] triangle histogram size FAILED'
