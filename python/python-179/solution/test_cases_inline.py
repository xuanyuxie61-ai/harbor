# ---- TC01: safe_inv 正常标量逆 ----
from system_utils import safe_inv
y = safe_inv(np.array([2.0, 4.0]))
assert np.allclose(y, [0.5, 0.25]), '[TC01] safe_inv normal inverse FAILED'

# ---- TC02: safe_inv 零值保护 ----
z = safe_inv(np.array([0.0, 1e-20]))
assert z[0] == 0.0, '[TC02] safe_inv zero protection FAILED'
assert np.isfinite(z).all(), '[TC02] safe_inv finite check FAILED'

# ---- TC03: robust_sqrt 负值保护(返回0而非NaN) ----
from system_utils import robust_sqrt
r = robust_sqrt(np.array([4.0, -1.0, 0.0]))
assert np.isfinite(r).all(), '[TC03] robust_sqrt finite FAILED'
assert r[0] == 2.0, '[TC03] robust_sqrt positive FAILED'
assert r[1] == 0.0, '[TC03] robust_sqrt negative protection FAILED'

# ---- TC04: chebyshev_nodes_1d 范围与排序 ----
xc = chebyshev_nodes_1d(-1.0, 1.0, 16)
assert len(xc) == 16, '[TC04] chebyshev_nodes_1d count FAILED'
assert xc.min() >= -1.0 and xc.max() <= 1.0, '[TC04] chebyshev_nodes_1d range FAILED'

# ---- TC05: hand_outline_polygon 输出形状 ----
poly = hand_outline_polygon(100)
assert poly.shape == (100, 2), '[TC05] hand_outline_polygon shape FAILED'
assert np.isfinite(poly).all(), '[TC05] hand_outline_polygon finite FAILED'

# ---- TC06: FEM 质量矩阵对称性 ----
M_fem, K_fem = assemble_fem_matrices_1d(np.linspace(-1.0, 1.0, 33))
assert np.allclose(M_fem, M_fem.T), '[TC06] FEM mass matrix symmetry FAILED'
assert np.allclose(K_fem, K_fem.T), '[TC06] FEM stiffness matrix symmetry FAILED'

# ---- TC07: 反应动力学精确解端点值 ----
t_test2 = np.linspace(0.0, 1.0, 20)
u_exact2 = twoway_exact_solution(t_test2, u0=0.1, k1=1.0, k2=10.0)
assert abs(u_exact2[0] - 0.1) < 1e-12, '[TC07] exact solution u(0) FAILED'
u_star = 10.0 / 11.0
assert abs(u_exact2[-1] - u_star) < 1e-3, '[TC07] exact solution steady state FAILED'

# ---- TC08: 三对角 CG 求解器残差精度 ----
A_r83_tc = r83_dif2(64)
b_tc = np.ones(64)
x_cg_tc = r83_cg(A_r83_tc, b_tc, tol=1e-12)
res_cg_tc = np.linalg.norm(r83_res(A_r83_tc, x_cg_tc, b_tc))
assert res_cg_tc < 1e-8, '[TC08] CG solver residual FAILED'

# ---- TC09: Hilbert 矩阵随机化 SVD 可复现性 ----
import numpy as np
H_tc = hilbert_matrix(30, 30)
U1, s1, Vt1 = randomized_svd(H_tc, k=5, p=3, seed=42)
U2, s2, Vt2 = randomized_svd(H_tc, k=5, p=3, seed=42)
assert np.allclose(s1, s2), '[TC09] randomized SVD reproducibility FAILED'

# ---- TC10: adaptive_rank_threshold 衰减奇异值 ----
s_decay = np.array([100.0, 0.5, 0.01, 0.001])
r = adaptive_rank_threshold(s_decay, tol=1e-2)
assert r == 2, '[TC10] adaptive_rank_threshold FAILED'

# ---- TC11: annulus_sample 半径范围 ----
import numpy as np
np.random.seed(42)
pts = annulus_sample(500, pc=np.array([0.0, 0.0]), r1=0.5, r2=1.0)
dists = np.linalg.norm(pts, axis=1)
assert dists.min() >= 0.5 - 1e-12, '[TC11] annulus_sample min radius FAILED'
assert dists.max() <= 1.0 + 1e-12, '[TC11] annulus_sample max radius FAILED'

# ---- TC12: van_der_corput_sequence 范围 ----
seq = van_der_corput_sequence(100, base=2)
assert seq.min() >= 0.0 and seq.max() < 1.0, '[TC12] van_der_corput range FAILED'

# ---- TC13: grid_integrate_1d 解析验证 (∫₀¹ x² dx = 1/3) ----
def f_sq(x):
    return x * x
I_grid = grid_integrate_1d(f_sq, 0.0, 1.0, 100)
assert abs(I_grid - 1.0/3.0) < 1e-4, '[TC13] grid_integrate_1d FAILED'

# ---- TC14: Monte Carlo 积分可复现性 ----
def f_mc(x):
    return np.exp(-np.sum(x**2))
est1, err1 = monte_carlo_integrate(f_mc, dim=3, n=5000, seed=42)
est2, err2 = monte_carlo_integrate(f_mc, dim=3, n=5000, seed=42)
assert est1 == est2, '[TC14] MC reproducibility FAILED'

# ---- TC15: rref_rank 单位矩阵 ----
I_mat = np.eye(20)
assert rref_rank(I_mat) == 20, '[TC15] rref_rank identity FAILED'

# ---- TC16: rref_rank 秩亏矩阵 ----
A_deficient = np.outer(np.ones(10), np.ones(10))
assert rref_rank(A_deficient) == 1, '[TC16] rref_rank rank-deficient FAILED'

# ---- TC17: collatz_polynomial_next 确定性 ----
p0_tc = np.array([1, 1, 0, 1], dtype=int)
p1 = collatz_polynomial_next(p0_tc)
p2 = collatz_polynomial_next(p0_tc)
assert np.array_equal(p1, p2), '[TC17] collatz_polynomial_next determinism FAILED'

# ---- TC18: hankel_matrix_from_sequence 结构 ----
s_hankel = np.arange(1, 15, dtype=float)
H_h = hankel_matrix_from_sequence(s_hankel, 5, 5)
assert H_h[0, 0] == 1.0, '[TC18] Hankel H[0,0] FAILED'
assert H_h[0, 1] == 2.0, '[TC18] Hankel H[0,1] FAILED'
assert H_h[1, 0] == 2.0, '[TC18] Hankel anti-diagonal constancy FAILED'

# ---- TC19: NMF 初始化非负性 ----
W, H_nmf = nmf_init_random(20, 15, 4, seed=42)
assert W.min() >= 0.0, '[TC19] NMF W nonnegative FAILED'
assert H_nmf.min() >= 0.0, '[TC19] NMF H nonnegative FAILED'

# ---- TC20: nonnegative_projection ----
X_np = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
X_proj = nonnegative_projection(X_np)
assert X_proj.min() >= 0.0, '[TC20] nonnegative_projection FAILED'
assert np.allclose(X_proj, [0.0, 0.0, 0.0, 1.0, 3.0]), '[TC20] nonnegative_projection values FAILED'

# ---- TC21: soft_threshold 解析验证 ----
X_st = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
X_th = soft_threshold(X_st, tau=2.0)
assert np.allclose(X_th, [-1.0, 0.0, 0.0, 0.0, 1.0]), '[TC21] soft_threshold FAILED'

# ---- TC22: parametric_reaction_source 有限性 ----
u_tc = np.array([0.0, 0.5, 1.0])
R_tc = parametric_reaction_source(u_tc, k1=1.0, k2=5.0, mu=0.5, mix_ratio=0.5)
assert np.isfinite(R_tc).all(), '[TC22] parametric_reaction_source finite FAILED'

# ---- TC23: triangle_area 已知三角形 ----
v1 = np.array([0.0, 0.0])
v2 = np.array([3.0, 0.0])
v3 = np.array([0.0, 4.0])
assert abs(triangle_area(v1, v2, v3) - 6.0) < 1e-12, '[TC23] triangle_area FAILED'

# ---- TC24: tensor_to_coordinate 往返 ----
tensor_small = np.array([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [0.0, 0.0]]])
idx, vals = tensor_to_coordinate(tensor_small, tol=0.0)
tensor_back = coordinate_to_tensor(idx, vals, tensor_small.shape)
assert np.allclose(tensor_small, tensor_back), '[TC24] tensor_to_coordinate round-trip FAILED'

# ---- TC25: low_rank_test_matrix 秩验证 ----
A_lr_tc = low_rank_test_matrix(30, 30, rank=4, seed=99)
s_lr = np.linalg.svd(A_lr_tc, compute_uv=False)
rank_est = adaptive_rank_threshold(s_lr, tol=1e-10)
assert rank_est >= 3, '[TC25] low_rank_test_matrix rank FAILED'

# ---- TC26: extract_tridiagonal 结构 ----
A_dense = np.eye(5) * 2.0 + np.eye(5, k=-1) * (-1.0) + np.eye(5, k=1) * (-1.0)
A83 = extract_tridiagonal(A_dense)
assert A83.shape == (3, 5), '[TC26] extract_tridiagonal shape FAILED'

# ---- TC27: vanderpol_reaction_term 符号 ----
u_vdp = np.array([0.0, 0.5, 1.0, 1.5])
R_vdp = vanderpol_reaction_term(u_vdp, mu=2.0)
assert R_vdp[0] == 0.0, '[TC27] vanderpol u=0 => R=0 FAILED'
assert R_vdp[1] > 0.0, '[TC27] vanderpol |u|<1 => R>0 FAILED'
assert R_vdp[2] == 0.0, '[TC27] vanderpol |u|=1 => R=0 FAILED'

# ---- TC28: reaction_jacobian_diagonal 有限 ----
J_diag = reaction_jacobian_diagonal(np.array([0.5, 1.0]), k1=1.0, k2=2.0, mu=0.5, mix_ratio=0.3)
assert np.isfinite(J_diag).all(), '[TC28] reaction_jacobian_diagonal finite FAILED'

# ---- TC29: qmc_integrate 与 MC 一致性检查 ----
def f_qmc(x):
    return 1.0
est_qmc = qmc_integrate(f_qmc, dim=3, n=2000)
assert abs(est_qmc - 1.0) < 0.1, '[TC29] QMC constant integrand FAILED'

# ---- TC30: r83_mv 与 r83_mtv 对称性 (A=A^T时) ----
A_sym = r83_dif2(10)
x_a = np.random.randn(10)
y_a = np.random.randn(10)
assert abs(np.dot(y_a, r83_mv(A_sym, x_a)) - np.dot(x_a, r83_mtv(A_sym, y_a))) < 1e-10, '[TC30] r83 mv/mtv symmetry FAILED'
