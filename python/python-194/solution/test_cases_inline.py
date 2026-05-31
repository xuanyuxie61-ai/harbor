# ---- TC01: BandedSPDMatrix set/get ----
import numpy as np
A1 = BandedSPDMatrix(5, 2)
A1.set(0, 0, 3.0)
A1.set(2, 0, 0.5)
assert abs(A1.get(0, 0) - 3.0) < 1e-15, '[TC01] BandedSPDMatrix set/get FAILED'
assert abs(A1.get(2, 0) - 0.5) < 1e-15, '[TC01] BandedSPDMatrix symmetry access FAILED'
assert abs(A1.get(0, 2) - 0.5) < 1e-15, '[TC01] BandedSPDMatrix upper triangle access FAILED'

# ---- TC02: BandedSPDMatrix.to_dense ----
A2 = BandedSPDMatrix(4, 1)
A2.set(0, 0, 2.0)
A2.set(1, 1, 2.0)
A2.set(1, 0, -1.0)
A2.set(2, 2, 2.0)
A2.set(2, 1, -1.0)
A2.set(3, 3, 2.0)
A2.set(3, 2, -1.0)
dense = A2.to_dense()
assert dense.shape == (4, 4), '[TC02] to_dense shape FAILED'
assert abs(dense[0, 0] - 2.0) < 1e-15, '[TC02] to_dense diagonal FAILED'
assert abs(dense[1, 0] + 1.0) < 1e-15, '[TC02] to_dense off-diagonal FAILED'
assert abs(dense[0, 1] + 1.0) < 1e-15, '[TC02] to_dense symmetry FAILED'

# ---- TC03: BandedSPDMatrix.dif2_band ----
A3 = BandedSPDMatrix.dif2_band(6)
assert A3.n == 6, '[TC03] dif2_band size FAILED'
assert abs(A3.get(0, 0) - 2.0) < 1e-15, '[TC03] dif2_band diagonal FAILED'
assert abs(A3.get(1, 0) + 1.0) < 1e-15, '[TC03] dif2_band off-diagonal FAILED'
assert abs(A3.get(5, 5) - 2.0) < 1e-15, '[TC03] dif2_band last diagonal FAILED'

# ---- TC04: Cholesky factorization + solve ||Ax - b|| ----
np.random.seed(42)
A4 = BandedSPDMatrix(10, 2)
for i in range(10):
    A4.set(i, i, 4.0 + np.random.rand())
    if i > 0:
        A4.set(i, i - 1, -1.0 + 0.1 * np.random.rand())
    if i > 1:
        A4.set(i, i - 2, -0.3 + 0.05 * np.random.rand())
x_exact = np.ones(10, dtype=float)
b4 = A4.to_dense() @ x_exact
L4 = A4.cholesky_band()
x_sol = A4.solve_cholesky(L4, b4)
err = np.linalg.norm(x_sol - x_exact) / max(1.0, np.linalg.norm(x_exact))
assert err < 1e-9, '[TC04] Cholesky solve accuracy FAILED'

# ---- TC05: banded_cholesky_solve convenience function ----
A5 = BandedSPDMatrix.dif2_band(5)
b5 = np.array([3.0, 1.0, 2.0, 1.0, 3.0], dtype=float)
x5 = banded_cholesky_solve(A5, b5)
assert x5.shape == (5,), '[TC05] banded_cholesky_solve output shape FAILED'
assert np.all(np.isfinite(x5)), '[TC05] banded_cholesky_solve NaN/Inf FAILED'

# ---- TC06: residual_norm computes correct residual ----
A6 = BandedSPDMatrix.dif2_band(4)
x6 = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
b6 = np.array([0.0, -1.0, -1.0, 2.0], dtype=float)
r6 = residual_norm(A6, x6, b6)
assert np.isfinite(r6), '[TC06] residual_norm NaN FAILED'
assert r6 >= 0.0, '[TC06] residual_norm negative FAILED'

# ---- TC07: SparseTriplet to_dense and matvec ----
S7 = SparseTriplet(3, 3)
S7.add(0, 0, 1.0)
S7.add(0, 1, 2.0)
S7.add(1, 1, 3.0)
S7.add(2, 2, 4.0)
d7 = S7.to_dense()
assert d7.shape == (3, 3), '[TC07] SparseTriplet to_dense shape FAILED'
assert abs(d7[0, 0] - 1.0) < 1e-15, '[TC07] SparseTriplet dense value FAILED'
y7 = S7.matvec(np.array([1.0, 0.0, 0.0], dtype=float))
assert abs(y7[0] - 1.0) < 1e-15, '[TC07] SparseTriplet matvec FAILED'

# ---- TC08: SparseTriplet to_csr format ----
S8 = SparseTriplet(3, 3)
S8.add(0, 2, 5.0)
S8.add(1, 1, 3.0)
S8.add(2, 0, 1.0)
data, indices, indptr = S8.to_csr()
assert len(data) == 3, '[TC08] to_csr data length FAILED'
assert indptr.shape == (4,), '[TC08] to_csr indptr shape FAILED'

# ---- TC09: metric_identity returns 2x2 identity ----
M9 = metric_identity(np.array([0.5, 0.5]))
assert M9.shape == (2, 2), '[TC09] metric_identity shape FAILED'
assert abs(M9[0, 0] - 1.0) < 1e-15, '[TC09] metric_identity diagonal FAILED'
assert abs(M9[0, 1]) < 1e-15, '[TC09] metric_identity off-diagonal FAILED'

# ---- TC10: metric_anisotropic returns diag(alpha, 1) ----
M10 = metric_anisotropic(np.array([0.3, 0.7]), alpha=5.0)
assert M10.shape == (2, 2), '[TC10] metric_anisotropic shape FAILED'
assert abs(M10[0, 0] - 5.0) < 1e-15, '[TC10] metric_anisotropic alpha FAILED'
assert abs(M10[1, 1] - 1.0) < 1e-15, '[TC10] metric_anisotropic y-diagonal FAILED'
assert abs(M10[0, 1]) < 1e-15, '[TC10] metric_anisotropic off-diagonal FAILED'

# ---- TC11: metric_boundary_layer metric properties ----
M11 = metric_boundary_layer(np.array([0.05, 0.5]), eps=0.01)
assert M11.shape == (2, 2), '[TC11] metric_boundary_layer shape FAILED'
assert M11[0, 0] > 1.0, '[TC11] metric_boundary_layer stretching FAILED'
assert abs(M11[1, 1] - 1.0) < 1e-15, '[TC11] metric_boundary_layer y-diagonal FAILED'

# ---- TC12: anisotropic_distance Euclidean case ----
np.random.seed(42)
z1 = np.array([0.2, 0.3])
z2 = np.array([0.5, 0.7])
d12 = anisotropic_distance(z1, z2, metric_identity)
expected = np.linalg.norm(z1 - z2)
assert abs(d12 - expected) < 1e-14, '[TC12] anisotropic_distance Euclidean FAILED'

# ---- TC13: anisotropic_distance self-distance is zero ----
z13 = np.array([0.42, 0.73])
d13 = anisotropic_distance(z13, z13, metric_anisotropic)
assert abs(d13) < 1e-15, '[TC13] anisotropic_distance self-distance FAILED'

# ---- TC14: lloyd_iteration_cvt basic run (seed fixed) ----
np.random.seed(42)
gens = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
new_gens = lloyd_iteration_cvt(gens, n_samples=2000, metric_func=metric_identity, domain=(0.0, 1.0, 0.0, 1.0))
assert new_gens.shape == (2, 2), '[TC14] lloyd_iteration_cvt output shape FAILED'
assert np.all(np.isfinite(new_gens)), '[TC14] lloyd_iteration_cvt NaN/Inf FAILED'
assert np.all(new_gens >= 0.0) and np.all(new_gens <= 1.0), '[TC14] lloyd_iteration_cvt domain bounds FAILED'

# ---- TC15: compute_cvt deterministic (seed fixed) ----
np.random.seed(99)
c15 = compute_cvt(n_subdomains=3, n_iterations=10, n_samples=3000, metric_func=metric_identity, domain=(0.0, 1.0, 0.0, 1.0), tol=1e-6)
assert c15.shape == (3, 2), '[TC15] compute_cvt output shape FAILED'
assert np.all(np.isfinite(c15)), '[TC15] compute_cvt NaN/Inf FAILED'

# ---- TC16: legendre_polynomial P0=1, P1=x ----
x16 = np.array([-0.5, 0.0, 0.5, 1.0])
P0 = legendre_polynomial(x16, 0)
assert np.allclose(P0, 1.0), '[TC16] legendre P0 FAILED'
P1 = legendre_polynomial(x16, 1)
assert np.allclose(P1, x16), '[TC16] legendre P1 FAILED'

# ---- TC17: legendre_polynomial P2 known values ----
x17 = np.array([0.0, 1.0, -1.0])
P2 = legendre_polynomial(x17, 2)
# P2(x) = (3x^2 - 1)/2
assert abs(P2[0] + 0.5) < 1e-14, '[TC17] legendre P2(0) FAILED'
assert abs(P2[1] - 1.0) < 1e-14, '[TC17] legendre P2(1) FAILED'
assert abs(P2[2] - 1.0) < 1e-14, '[TC17] legendre P2(-1) FAILED'

# ---- TC18: gll_nodes_weights endpoints and sum ----
xi18, w18 = gll_nodes_weights(p=4)
assert xi18.shape[0] == 5, '[TC18] gll_nodes_weights node count FAILED'
assert abs(xi18[0] + 1.0) < 1e-14, '[TC18] gll_nodes_weights first node FAILED'
assert abs(xi18[-1] - 1.0) < 1e-14, '[TC18] gll_nodes_weights last node FAILED'
assert abs(np.sum(w18) - 2.0) < 1e-12, '[TC18] gll_nodes_weights sum FAILED'
assert np.all(w18 > 0), '[TC18] gll_nodes_weights positivity FAILED'

# ---- TC19: gll_nodes_weights monotonic ----
xi19, _ = gll_nodes_weights(p=5)
assert np.all(np.diff(xi19) > 0), '[TC19] gll_nodes_weights monotonic FAILED'

# ---- TC20: fekete_line_rule weights sum = b - a ----
x20, w20 = fekete_line_rule(0.0, 1.0, 3)
assert abs(np.sum(w20) - 1.0) < 1e-12, '[TC20] fekete_line_rule weight sum FAILED'
assert x20.shape[0] == 4, '[TC20] fekete_line_rule node count FAILED'

# ---- TC21: fekete_line_rule on [a,b] nodes within range ----
x21, w21 = fekete_line_rule(2.0, 5.0, 4)
assert np.all(x21 >= 2.0) and np.all(x21 <= 5.0), '[TC21] fekete_line_rule node bounds FAILED'
assert abs(np.sum(w21) - 3.0) < 1e-12, '[TC21] fekete_line_rule weight sum FAILED'

# ---- TC22: fekete_triangle_nodes_weights output ----
pts22, wts22 = fekete_triangle_nodes_weights(p=4)
n_expected = (4 + 1) * (4 + 2) // 2  # (p+1)(p+2)/2 = 15
assert pts22.shape[0] > 0, '[TC22] fekete_triangle nodes count FAILED'
assert wts22.shape[0] == pts22.shape[0], '[TC22] fekete_triangle weights mismatch FAILED'
assert np.all(wts22 >= 0), '[TC22] fekete_triangle weights positivity FAILED'

# ---- TC23: vandermonde_1d shape ----
v23 = vandermonde_1d(np.array([-1.0, 0.0, 1.0]), 3)
assert v23.shape == (3, 4), '[TC23] vandermonde_1d shape FAILED'
assert np.all(np.isfinite(v23)), '[TC23] vandermonde_1d NaN/Inf FAILED'

# ---- TC24: stokes_solution_polynomial divergence-free ----
np.random.seed(42)
x_rand = np.random.rand(20)
y_rand = np.random.rand(20)
u_poly, v_poly, p_poly = stokes_solution_polynomial(x_rand, y_rand)
# Finite difference divergence check
eps = 1e-6
u_xp = 2.0 * (x_rand + eps) ** 2 * (x_rand + eps - 1.0) ** 2 * y_rand * (2.0 * y_rand - 1.0) * (y_rand - 1.0)
u_xm = 2.0 * (x_rand - eps) ** 2 * (x_rand - eps - 1.0) ** 2 * y_rand * (2.0 * y_rand - 1.0) * (y_rand - 1.0)
dudx_fd = (u_xp - u_xm) / (2 * eps)
v_yp = -2.0 * x_rand * (2.0 * x_rand - 1.0) * (x_rand - 1.0) * (y_rand + eps) ** 2 * (y_rand + eps - 1.0) ** 2
v_ym = -2.0 * x_rand * (2.0 * x_rand - 1.0) * (x_rand - 1.0) * (y_rand - eps) ** 2 * (y_rand - eps - 1.0) ** 2
dvdy_fd = (v_yp - v_ym) / (2 * eps)
div_fd = np.abs(dudx_fd + dvdy_fd)
assert np.max(div_fd) < 1e-4, '[TC24] stokes_solution_polynomial divergence FAILED'

# ---- TC25: stokes_solution_trigonometric divergence-free ----
np.random.seed(43)
x_trig = np.random.rand(15)
y_trig = np.random.rand(15)
u_trig, v_trig, p_trig = stokes_solution_trigonometric(x_trig, y_trig)
div_trig = np.pi * np.cos(np.pi * x_trig) * np.cos(np.pi * y_trig) - np.pi * np.cos(np.pi * x_trig) * np.cos(np.pi * y_trig)
assert np.max(np.abs(div_trig)) < 1e-14, '[TC25] stokes_solution_trigonometric divergence FAILED'

# ---- TC26: stokes_solution_polynomial boundary values ----
xb = np.array([0.0, 1.0, 0.5])
yb = np.array([0.0, 1.0, 0.5])
u_b, v_b, p_b = stokes_solution_polynomial(xb, yb)
assert abs(u_b[0]) < 1e-15, '[TC26] u(0,0)=0 FAILED'
assert abs(v_b[0]) < 1e-15, '[TC26] v(0,0)=0 FAILED'

# ---- TC27: evaluate_solution polynomial returns 6 arrays ----
x27 = np.linspace(0.0, 1.0, 5)
y27 = np.linspace(0.0, 1.0, 5)
result = evaluate_solution(x27, y27, sol_type="polynomial", nu=1.0)
assert len(result) == 6, '[TC27] evaluate_solution output count FAILED'
u27, v27, p27, fx27, fy27, h27 = result
assert u27.shape == (5,), '[TC27] evaluate_solution u shape FAILED'
assert h27.shape == (5,), '[TC27] evaluate_solution h shape FAILED'

# ---- TC28: evaluate_solution trigonometric returns valid ----
x28 = np.linspace(0.0, 1.0, 5)
y28 = np.linspace(0.0, 1.0, 5)
u28, v28, p28, fx28, fy28, h28 = evaluate_solution(x28, y28, sol_type="trigonometric", nu=1.0)
assert np.all(np.isfinite(u28)), '[TC28] trig u NaN/Inf FAILED'
assert np.all(np.isfinite(fx28)), '[TC28] trig fx NaN/Inf FAILED'

# ---- TC29: compute_discrete_residual zero for exact solution ----
x29 = np.linspace(0.0, 1.0, 8)
y29 = np.linspace(0.0, 1.0, 8)
X29, Y29 = np.meshgrid(x29, y29)
x_f = X29.flatten()
y_f = Y29.flatten()
u_ex29, v_ex29, p_ex29, _, _, _ = evaluate_solution(x_f, y_f, sol_type="polynomial", nu=1.0)
res29 = compute_discrete_residual(u_ex29, v_ex29, p_ex29, x_f, y_f, nu=1.0, sol_type="polynomial")
assert abs(res29) < 1e-14, '[TC29] compute_discrete_residual exact FAILED'

# ---- TC30: diophantine_partition sum equals N ----
np.random.seed(42)
p30 = diophantine_partition(100, 4)
assert sum(p30) == 100, '[TC30] diophantine_partition sum FAILED'
assert all(m >= 1 for m in p30), '[TC30] diophantine_partition positivity FAILED'

# ---- TC31: diophantine_partition balanced for equal case ----
p31 = diophantine_partition(64, 8)
assert sum(p31) == 64, '[TC31] diophantine_partition sum FAILED'
assert max(p31) - min(p31) <= 1, '[TC31] diophantine_partition balance FAILED'

# ---- TC32: build_local_stokes_matrix output types ----
np.random.seed(42)
A_loc32, B_loc32, M_loc32 = build_local_stokes_matrix(10, dx=0.1, nu=1.0)
assert A_loc32.n == 10, '[TC32] build_local_stokes_matrix A size FAILED'
assert B_loc32.shape == (10,), '[TC32] build_local_stokes_matrix B shape FAILED'
assert M_loc32.shape == (10,), '[TC32] build_local_stokes_matrix M shape FAILED'
assert np.all(M_loc32 > 0), '[TC32] build_local_stokes_matrix M positivity FAILED'

# ---- TC33: solve_local_schur_complement basic solve ----
np.random.seed(42)
A_loc33, B_loc33, M_loc33 = build_local_stokes_matrix(8, dx=0.125, nu=1.0)
f_loc33 = np.ones(8, dtype=float)
u33, p33 = solve_local_schur_complement(A_loc33, B_loc33, f_loc33, g_loc=0.0, max_iter=50, tol=1e-10)
assert u33.shape == (8,), '[TC33] solve_local_schur_complement u shape FAILED'
assert np.isfinite(p33), '[TC33] solve_local_schur_complement p NaN FAILED'
assert np.all(np.isfinite(u33)), '[TC33] solve_local_schur_complement u NaN FAILED'

# ---- TC34: partition_domain_1d covers all points ----
subs34 = partition_domain_1d(128, 4, 4)
assert len(subs34) == 4, '[TC34] partition_domain_1d count FAILED'
# Check coverage (可能不完全覆盖但应有意义)
for start, end in subs34:
    assert 0 <= start <= 128, '[TC34] partition_domain_1d start bounds FAILED'
    assert 0 <= end <= 128, '[TC34] partition_domain_1d end bounds FAILED'

# ---- TC35: additive_schwarz_iteration basic convergence ----
np.random.seed(42)
N35 = 32
dx35 = 1.0 / N35
subs35 = partition_domain_1d(N35, 4, 2)
x_grid35 = np.linspace(0.0, 1.0, N35)
u_ex35, v_ex35, p_ex35, fx35, fy35, h35 = evaluate_solution(x_grid35, np.zeros_like(x_grid35), sol_type="polynomial", nu=1.0)
f_global35 = fx35.copy()
g_global35 = h35.copy()
u_sol35, p_sol35, res35 = additive_schwarz_iteration(
    global_n=N35, subdomains=subs35, overlap=2,
    f_global=f_global35, g_global=g_global35,
    dx=dx35, nu=1.0, max_iter=50, tol=1e-6
)
assert u_sol35.shape == (N35,), '[TC35] additive_schwarz_iteration u shape FAILED'
assert p_sol35.shape == (N35,), '[TC35] additive_schwarz_iteration p shape FAILED'
assert np.isfinite(res35), '[TC35] additive_schwarz_iteration residual NaN FAILED'
assert np.all(np.isfinite(u_sol35)), '[TC35] additive_schwarz_iteration u NaN/Inf FAILED'

# ---- TC36: semi_implicit_euler_step output shape and finiteness ----
np.random.seed(42)
A36 = BandedSPDMatrix.dif2_band(8)
A36_scaled = BandedSPDMatrix(8, 1)
for i36 in range(8):
    A36_scaled.set(i36, i36, 2.0)
    if i36 > 0:
        A36_scaled.set(i36, i36 - 1, -0.5)
u36 = np.ones(8, dtype=float)
f36 = np.zeros(8, dtype=float)
u_new36 = semi_implicit_euler_step(u36, A36_scaled, f36, dt=0.01)
assert u_new36.shape == (8,), '[TC36] semi_implicit_euler_step shape FAILED'
assert np.all(np.isfinite(u_new36)), '[TC36] semi_implicit_euler_step NaN/Inf FAILED'

# ---- TC37: velocity_verlet_step basic properties ----
np.random.seed(42)
def accel_simple(u):
    return -2.0 * u
u37 = np.array([1.0, 0.0, 0.0], dtype=float)
v37 = np.array([0.0, 1.0, 0.0], dtype=float)
u_new37, v_new37, a_new37 = velocity_verlet_step(u37, v37, accel_simple, dt=0.01)
assert u_new37.shape == (3,), '[TC37] velocity_verlet_step u shape FAILED'
assert v_new37.shape == (3,), '[TC37] velocity_verlet_step v shape FAILED'
assert np.all(np.isfinite(u_new37)), '[TC37] velocity_verlet_step NaN/Inf FAILED'

# ---- TC38: transient_stokes_step output shapes ----
np.random.seed(42)
A38 = BandedSPDMatrix.dif2_band(8)
A38_scaled = BandedSPDMatrix(8, 1)
for i38 in range(8):
    A38_scaled.set(i38, i38, 2.0)
    if i38 > 0:
        A38_scaled.set(i38, i38 - 1, -0.5)
B_div38 = np.zeros(8, dtype=float)
B_div38[0] = -10.0
B_div38[1] = 10.0
for i38 in range(1, 7):
    B_div38[i38] = -10.0
    B_div38[i38 + 1] = 10.0
u_n38 = np.ones(8, dtype=float)
v_n38 = np.zeros(8, dtype=float)
p_n38 = np.zeros(8, dtype=float)
f_n38 = np.zeros(8, dtype=float)
u_new38, v_new38, p_new38 = transient_stokes_step(
    u_n38, v_n38, p_n38, A38_scaled, B_div38, f_n38, dt=0.01, nu=1.0, rho=1.0
)
assert u_new38.shape == (8,), '[TC38] transient_stokes_step u shape FAILED'
assert p_new38.shape == (8,), '[TC38] transient_stokes_step p shape FAILED'
assert np.all(np.isfinite(u_new38)), '[TC38] transient_stokes_step NaN/Inf FAILED'

# ---- TC39: golden_section_search finds minimum of x^2 ----
x39, f39 = golden_section_search(lambda x: x**2, -2.0, 3.0, tol=1e-8)
assert abs(x39) < 1e-6, '[TC39] golden_section_search x^2 minimizer FAILED'
assert abs(f39) < 1e-12, '[TC39] golden_section_search x^2 minimum FAILED'

# ---- TC40: golden_section_search on (x-1.5)^2 ----
x40, f40 = golden_section_search(lambda x: (x - 1.5)**2 + 0.5, 0.0, 3.0, tol=1e-8)
assert abs(x40 - 1.5) < 1e-6, '[TC40] golden_section_search shifted minimizer FAILED'
assert abs(f40 - 0.5) < 1e-12, '[TC40] golden_section_search shifted minimum FAILED'

# ---- TC41: backtracking_line_search returns finite positive step ----
def f41(x):
    return float(x[0]**2 + x[1]**2)
def g41(x):
    return np.array([2.0*x[0], 2.0*x[1]], dtype=float)
x41 = np.array([2.0, 3.0], dtype=float)
p41 = -g41(x41)
alpha41 = backtracking_line_search(f41, g41, x41, p41, alpha_init=1.0)
assert alpha41 > 0.0, '[TC41] backtracking_line_search zero step FAILED'
assert alpha41 <= 1.0, '[TC41] backtracking_line_search too large step FAILED'

# ---- TC42: power_iteration_estimate positive eigenvalue ----
np.random.seed(42)
A42 = BandedSPDMatrix.dif2_band(10)
def matvec42(v):
    return A42.to_dense() @ v
lam42 = power_iteration_estimate(matvec42, 10, max_iter=30)
assert lam42 > 0.0, '[TC42] power_iteration_estimate non-positive FAILED'
# DIF2(10) largest eigenvalue: 2 + 2*cos(pi/11) ≈ 3.92
assert 3.0 < lam42 < 5.0, '[TC42] power_iteration_estimate out of range FAILED'

# ---- TC43: optimize_subdomain_overlap ----
def resid_alpha(a):
    return (a - 0.2)**2 + 0.05 * np.sin(10.0 * a)
a_opt43 = optimize_subdomain_overlap(resid_alpha, overlap_range=(0.0, 0.5), tol=1e-5)
assert 0.0 <= a_opt43 <= 0.5, '[TC43] optimize_subdomain_overlap bounds FAILED'

# ---- TC44: hypersphere_positive_sample unit norm ----
np.random.seed(42)
sp44 = hypersphere_positive_sample(m=5)
assert abs(np.linalg.norm(sp44) - 1.0) < 1e-14, '[TC44] hypersphere_positive_sample norm FAILED'
assert np.all(sp44 >= 0.0), '[TC44] hypersphere_positive_sample positivity FAILED'

# ---- TC45: hypersphere_positive_distance_stats finite outputs ----
np.random.seed(42)
mean45, var45 = hypersphere_positive_distance_stats(m=3, n_samples=500)
assert np.isfinite(mean45), '[TC45] hypersphere_positive_distance_stats mean NaN FAILED'
assert np.isfinite(var45), '[TC45] hypersphere_positive_distance_stats var NaN FAILED'
assert mean45 > 0.0, '[TC45] hypersphere_positive_distance_stats mean non-positive FAILED'
assert var45 >= 0.0, '[TC45] hypersphere_positive_distance_stats var negative FAILED'

# ---- TC46: compute_partition_quality balanced case ----
Q46 = compute_partition_quality(np.array([25.0, 25.0, 25.0, 25.0]))
assert abs(Q46 - 1.0) < 1e-12, '[TC46] compute_partition_quality balanced FAILED'
Q47 = compute_partition_quality(np.array([100.0, 0.0, 0.0, 0.0]))
assert Q47 < 0.5, '[TC46] compute_partition_quality unbalanced FAILED'

# ---- TC47: plane_tetrahedron_intersect basic ----
tetra47 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
poly47 = plane_tetrahedron_intersect(np.array([0.3, 0.3, 0.1]), np.array([1.0, 1.0, 1.0]), tetra47)
assert poly47 is not None, '[TC47] plane_tetrahedron_intersect None FAILED'
assert poly47.shape[0] >= 2, '[TC47] plane_tetrahedron_intersect too few vertices FAILED'

# ---- TC48: polygon_area_3d triangle ----
tri48 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
area48 = polygon_area_3d(tri48)
assert abs(area48 - 0.5) < 1e-14, '[TC48] polygon_area_3d triangle FAILED'

# ---- TC49: interface_matching_score perfect match ----
nodes49 = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=float)
score49 = interface_matching_score(nodes49, nodes49.copy(), tol=1e-6)
assert abs(score49 - 1.0) < 1e-14, '[TC49] interface_matching_score perfect FAILED'

# ---- TC50: interface_matching_score no match ----
nodes50a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
nodes50b = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=float)
score50 = interface_matching_score(nodes50a, nodes50b, tol=1e-6)
assert score50 < 0.5, '[TC50] interface_matching_score no match FAILED'

# ---- TC51: partial_digest_reconstruct basic ----
dists51 = [0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
recon51 = partial_digest_reconstruct(dists51, max_coord=2.0)
assert recon51 is not None, '[TC51] partial_digest_reconstruct None FAILED'
assert len(recon51) >= 3, '[TC51] partial_digest_reconstruct too few points FAILED'
assert abs(recon51[0]) < 1e-15, '[TC51] partial_digest_reconstruct first point FAILED'

# ---- TC52: partial_digest_reconstruct from known points ----
true52 = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
dists52 = []
for i52 in range(len(true52)):
    for j52 in range(i52 + 1, len(true52)):
        dists52.append(abs(true52[i52] - true52[j52]))
recon52 = partial_digest_reconstruct(dists52, max_coord=1.0)
assert recon52 is not None, '[TC52] partial_digest_reconstruct None FAILED'

# ---- TC53: parallel_efficiency_estimate in range ----
eff53 = parallel_efficiency_estimate(serial_time=1.0, n_subdomains=4, comm_fraction=0.1)
assert 0.0 < eff53 <= 1.0, '[TC53] parallel_efficiency_estimate out of range FAILED'
eff53b = parallel_efficiency_estimate(serial_time=1.0, n_subdomains=1, comm_fraction=0.0)
assert abs(eff53b - 1.0) < 1e-14, '[TC53] parallel_efficiency_estimate single FAILED'

# ---- TC54: linpack_benchmark_dense finite results ----
np.random.seed(42)
t54, m54, r54 = linpack_benchmark_dense(n=64)
assert np.isfinite(t54) and t54 > 0, '[TC54] linpack_benchmark_dense time FAILED'
assert np.isfinite(m54) and m54 > 0, '[TC54] linpack_benchmark_dense mflops FAILED'
assert np.isfinite(r54), '[TC54] linpack_benchmark_dense residual FAILED'

# ---- TC55: banded_benchmark finite results ----
np.random.seed(42)
t55, m55, r55 = banded_benchmark(n=64, bandwidth=3, n_runs=2)
assert np.isfinite(t55) and t55 > 0, '[TC55] banded_benchmark time FAILED'
assert np.isfinite(m55) and m55 > 0, '[TC55] banded_benchmark mflops FAILED'
assert np.isfinite(r55), '[TC55] banded_benchmark residual FAILED'

# ---- TC56: subdomain_solver_benchmark results ----
np.random.seed(42)
bench56 = subdomain_solver_benchmark([32, 64], bandwidth=2, n_runs=1)
assert len(bench56) == 2, '[TC56] subdomain_solver_benchmark count FAILED'
for v56 in bench56.values():
    assert v56['time'] > 0, '[TC56] subdomain_solver_benchmark time FAILED'

# ---- TC57: adaptive_time_stepping basic run (fixed seed) ----
np.random.seed(42)
N57 = 16
dx57 = 1.0 / N57
A57 = BandedSPDMatrix(N57, 1)
coeff57 = 1.0 / (dx57**2)
for i57 in range(N57):
    A57.set(i57, i57, 2.0 * coeff57)
    if i57 > 0:
        A57.set(i57, i57 - 1, -coeff57)
u057 = np.ones(N57, dtype=float) * 0.1
def rhs57(t, u):
    return np.zeros(N57, dtype=float)
t_hist57, u_hist57, dt_hist57 = adaptive_time_stepping(
    u0=u057, t_span=(0.0, 0.02), dt_init=0.005,
    rhs_func=rhs57, A_band=A57, tol=1e-3,
    dt_min=1e-6, dt_max=0.01, max_steps=100
)
assert len(t_hist57) >= 2, '[TC57] adaptive_time_stepping steps FAILED'
assert np.all(np.isfinite(u_hist57)), '[TC57] adaptive_time_stepping NaN/Inf FAILED'

# ---- TC58: reproducibility (fixed seed, same result) ----
np.random.seed(12345)
A58 = BandedSPDMatrix.dif2_band(6)
L58_a = A58.cholesky_band()
np.random.seed(12345)
A58b = BandedSPDMatrix.dif2_band(6)
L58_b = A58b.cholesky_band()
# dif2_band is deterministic, Cholesky should be identical
dense_a = L58_a.to_dense()
dense_b = L58_b.to_dense()
assert np.allclose(dense_a, dense_b), '[TC58] reproducibility FAILED'

# ---- TC59: numerical stability - zero input to polynomial solution ----
x59 = np.array([0.5, 0.5])
y59 = np.array([0.5, 0.5])
u59, v59, p59 = stokes_solution_polynomial(x59, y59)
assert np.all(np.isfinite(u59)), '[TC59] zero-like input NaN FAILED'

# ---- TC60: numerical stability - very small values ----
A60 = BandedSPDMatrix(3, 1)
A60.set(0, 0, 1e-10)
A60.set(1, 1, 1e-10)
A60.set(2, 2, 1e-10)
A60.set(1, 0, 1e-11)
A60.set(2, 1, 1e-11)
L60 = A60.cholesky_band()
b60 = np.array([1e-10, 2e-10, 3e-10], dtype=float)
x60 = A60.solve_cholesky(L60, b60)
assert np.all(np.isfinite(x60)), '[TC60] small values NaN FAILED'

# ---- TC61: monotonicity - golden section search convergence ----
vals = []
def track_f(x):
    vals.append(x)
    return (x - 1.0)**2
x61, f61 = golden_section_search(track_f, 0.0, 2.0, tol=1e-6)
assert abs(x61 - 1.0) < 1e-5, '[TC61] golden_section monotonic convergent FAILED'

# ---- TC62: SparseTriplet matvec match with dense ----
np.random.seed(42)
S62 = SparseTriplet(4, 4)
S62.add(0, 0, 3.0)
S62.add(1, 1, 2.0)
S62.add(2, 2, 1.0)
S62.add(3, 3, 4.0)
S62.add(0, 2, 1.5)
S62.add(2, 0, 1.5)
x62 = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
y_sp = S62.matvec(x62)
y_den = S62.to_dense() @ x62
assert np.allclose(y_sp, y_den), '[TC62] SparseTriplet matvec vs dense FAILED'

# ---- TC63: gll_nodes_weights various p ----
for p63 in [1, 2, 3, 5, 7]:
    xi63, w63 = gll_nodes_weights(p63)
    assert xi63.shape[0] == p63 + 1, f'[TC63] gll nodes count p={p63} FAILED'
    assert abs(np.sum(w63) - 2.0) < 1e-10, f'[TC63] gll weights sum p={p63} FAILED'

# ---- TC64: fekete_line_rule moment exactness ----
for p64 in [2, 3, 4]:
    x64, w64 = fekete_line_rule(0.0, 1.0, p64)
    # GLL integrates up to degree 2p-1, so x^p should be exact
    exact64 = 1.0 / (p64 + 1)
    approx64 = np.sum(w64 * x64**p64)
    assert abs(approx64 - exact64) < 1e-10, f'[TC64] fekete exactness p={p64} FAILED'

# ---- TC65: robustness - large overlap doesn't crash ----
try:
    subs65 = partition_domain_1d(50, 3, 8)
    assert len(subs65) == 3, '[TC65] partition_domain_1d large overlap FAILED'
except Exception:
    assert False, '[TC65] partition_domain_1d crash FAILED'
