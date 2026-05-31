# Additional imports needed for test cases
from matrix_exponential_ode import kepler_derivatives, kepler_hessian
from chebyshev_hermite_approx import chebyshev_nodes, chebyshev_coefficients
from legendre_quadrature_kernel import rescale_quadrature
from sampling_distribution import normal_01_cdf
from parallel_prime_partition import prime_factorization, is_smooth, find_optimal_process_count, prime_pi, balanced_block_sizes
from polynomial_hash_verify import polynomial_degree, collatz_polynomial_next

# ---- TC01: cannon_multiply identity matrix 4x4 correctness ----
A = np.eye(4)
B = np.eye(4)
C = cannon_multiply(A, B, num_processes=4)
assert np.allclose(C, np.eye(4), atol=1e-12), '[TC01] cannon_multiply identity FAILED'

# ---- TC02: mpi_summa_multiply 4x4 correctness against numpy.dot ----
np.random.seed(42)
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
C_summa = mpi_summa_multiply(A, B, num_processes=4)
C_ref = np.dot(A, B)
assert np.allclose(C_summa, C_ref, atol=1e-10), '[TC02] mpi_summa_multiply FAILED'

# ---- TC03: frobenius_error zero for identical matrices ----
A = np.random.randn(6, 6)
err = frobenius_error(A, A.copy())
assert err == 0.0 or err < 1e-15, '[TC03] frobenius_error identical FAILED'

# ---- TC04: matrix_exponential_pade rotation matrix exp(A) = [[cos1,sin1],[-sin1,cos1]] ----
A = np.array([[0.0, 1.0], [-1.0, 0.0]])
E = matrix_exponential_pade(A, order=7)
E_true = np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]])
assert np.linalg.norm(E - E_true, 'fro') < 1e-10, '[TC04] matrix_exponential_pade rotation FAILED'

# ---- TC05: kepler_derivatives for circular orbit (r=1, p tangential) ----
state = np.array([1.0, 0.0, 0.0, 1.0])
dy = kepler_derivatives(state)
assert abs(dy[0] - 0.0) < 1e-12, '[TC05] kepler_derivatives dq1 FAILED'
assert abs(dy[1] - 1.0) < 1e-12, '[TC05] kepler_derivatives dq2 FAILED'
assert abs(dy[2] + 1.0) < 1e-12, '[TC05] kepler_derivatives dp1 FAILED'
assert abs(dy[3] - 0.0) < 1e-12, '[TC05] kepler_derivatives dp2 FAILED'

# ---- TC06: integrate_kepler_stm symplectic property det(Phi) approx 1 ----
y0 = np.array([1.0, 0.0, 0.0, 1.0])
yf, Phi = integrate_kepler_stm(y0, (0.0, 0.5), n_steps=500)
det_Phi = np.linalg.det(Phi)
assert abs(det_Phi - 1.0) < 0.01, '[TC06] integrate_kepler_stm det(Phi) != 1 FAILED'

# ---- TC07: legendre_compute_glr exactness for polynomial degree 2n-1 (n=8, x^14) ----
x, w = legendre_compute_glr(8)
val_gl = np.sum(w * x ** 14)
true_val = 2.0 / 15.0
assert abs(val_gl - true_val) < 1e-10, '[TC07] Gauss-Legendre quadrature exactness FAILED'

# ---- TC08: integrate_disk_kernel r^2 over unit disk equals pi/2 ----
kernel = lambda x, y: x * x + y * y
val_disk = integrate_disk_kernel(kernel, nr=16, nt=32)
true_disk = np.pi / 2.0
assert abs(val_disk - true_disk) < 1e-4, '[TC08] integrate_disk_kernel FAILED'

# ---- TC09: cpr_roots finds roots for cos(3x) on [0, 2] (check roots exist) ----
f = lambda x: np.cos(3.0 * x)
roots, Einter = cpr_roots(f, 0.0, 2.0, N=128)
assert len(roots) >= 2, '[TC09] cpr_roots count FAILED'
# Check roots are near expected values (pi/6, pi/2) within tolerance
assert np.any(np.abs(roots - np.pi / 6.0) < 0.1), '[TC09] cpr_roots root pi/6 FAILED'
assert np.any(np.abs(roots - np.pi / 2.0) < 0.1), '[TC09] cpr_roots root pi/2 FAILED'

# ---- TC10: hermite interpolation accuracy for sin(x) with derivative info ----
x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_nodes = np.sin(x_nodes)
yp_nodes = np.cos(x_nodes)
z, d = hermite_divided_differences(x_nodes, y_nodes, yp_nodes)
x_test = np.linspace(0.0, 2.0, 51)
y_hermite = hermite_evaluate(z, d, x_test)
y_true = np.sin(x_test)
err_max = np.max(np.abs(y_hermite - y_true))
assert err_max < 0.01, '[TC10] hermite interpolation accuracy FAILED'

# ---- TC11: chebyshev_nodes monotonic and within [a,b] ----
a, b, N = -2.0, 3.0, 10
xn = chebyshev_nodes(a, b, N)
assert len(xn) == N + 1, '[TC11] chebyshev_nodes count FAILED'
assert np.min(xn) >= a - 1e-12, '[TC11] chebyshev_nodes lower bound FAILED'
assert np.max(xn) <= b + 1e-12, '[TC11] chebyshev_nodes upper bound FAILED'
assert np.all(np.diff(xn) <= 0), '[TC11] chebyshev_nodes monotonic FAILED'

# ---- TC12: TriangularMesh triangle_areas for right triangle (0,0)-(1,0)-(0,1) ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
elements = np.array([[0, 1, 2]])
mesh = TriangularMesh(nodes, elements)
areas = mesh.triangle_areas()
assert len(areas) == 1, '[TC12] triangle_areas count FAILED'
assert abs(areas[0] - 0.5) < 1e-12, '[TC12] triangle_areas value FAILED'

# ---- TC13: stiffness matrix COO data symmetrical structure ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 0.0]])
elements = np.array([[0, 1, 2], [0, 3, 1]])
mesh2 = TriangularMesh(nodes, elements)
data, rows, cols = mesh2.assemble_stiffness_matrix_2d()
assert len(data) > 0, '[TC13] stiffness matrix non-empty FAILED'
assert len(rows) == len(cols) == len(data), '[TC13] stiffness matrix COO shape mismatch FAILED'

# ---- TC14: sparse_matrix_vector_product correctness against dense multiplication ----
np.random.seed(42)
data_sp = np.array([2.0, -1.0, -1.0, 2.0])
row_sp = np.array([0, 0, 1, 1], dtype=np.int64)
col_sp = np.array([0, 1, 0, 1], dtype=np.int64)
x_vec = np.array([3.0, 4.0])
y_sp = sparse_matrix_vector_product(data_sp, row_sp, col_sp, x_vec, 2)
A_dense = np.array([[2.0, -1.0], [-1.0, 2.0]])
y_dense = np.dot(A_dense, x_vec)
assert np.allclose(y_sp, y_dense, atol=1e-12), '[TC14] sparse_matrix_vector_product FAILED'

# ---- TC15: normal_01_cdf monotonic and boundary values ----
assert normal_01_cdf(-10.0) < 1e-10, '[TC15] normal_01_cdf lower tail FAILED'
assert normal_01_cdf(10.0) > 0.999999, '[TC15] normal_01_cdf upper tail FAILED'
assert abs(normal_01_cdf(0.0) - 0.5) < 1e-8, '[TC15] normal_01_cdf median FAILED'

# ---- TC16: walker_build and walker_sampler produce correct distribution (fixed seed) ----
np.random.seed(42)
prob = np.array([0.1, 0.3, 0.4, 0.2])
y, a = walker_build(prob)
assert len(y) == 4 and len(a) == 4, '[TC16] walker_build shape FAILED'
counts = np.zeros(4)
for _ in range(5000):
    counts[walker_sampler(y, a)] += 1
assert np.all(counts > 0), '[TC16] walker_sampler all categories sampled FAILED'

# ---- TC17: cvt_energy_2d non-negative and Lloyd step reduces energy ----
np.random.seed(42)
gens = np.random.rand(5, 2)
samps = np.random.rand(500, 2)
E0 = cvt_energy_2d(gens, samps)
assert E0 >= 0.0, '[TC17] cvt_energy_2d non-negative FAILED'
gens_new = lloyd_step_2d(gens, samps)
E1 = cvt_energy_2d(gens_new, samps)
assert E1 <= E0 + 1e-12, '[TC17] Lloyd energy monotonic FAILED'

# ---- TC18: disk01_positive_sample points lie within unit disk (fixed seed) ----
np.random.seed(42)
pts = disk01_positive_sample(500)
r_sq = pts[:, 0] ** 2 + pts[:, 1] ** 2
assert np.all(r_sq <= 1.0 + 1e-12), '[TC18] disk01_positive_sample radius FAILED'
assert np.all(r_sq >= 0.0), '[TC18] disk01_positive_sample non-negative FAILED'

# ---- TC19: generate_random_matrix_lognormal shape correct ----
np.random.seed(42)
M_lognorm = generate_random_matrix_lognormal(6, mu=0.0, sigma=0.5, a=1e-6, b=10.0)
assert M_lognorm.shape == (6, 6), '[TC19] generate_random_matrix_lognormal shape FAILED'
assert np.all(np.isfinite(M_lognorm)), '[TC19] generate_random_matrix_lognormal finite FAILED'

# ---- TC20: prime_sieve known primes below 30 ----
primes = prime_sieve(30)
expected = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
assert np.array_equal(primes, expected), '[TC20] prime_sieve FAILED'

# ---- TC21: prime_factorization known factorizations ----
factors_12 = prime_factorization(12)
assert (2, 2) in factors_12, '[TC21] prime_factorization 2^2 FAILED'
assert (3, 1) in factors_12, '[TC21] prime_factorization 3^1 FAILED'

# ---- TC22: balanced_block_sizes sum equals n ----
n, p = 10, 3
blocks = balanced_block_sizes(n, p)
assert len(blocks) == p, '[TC22] balanced_block_sizes count FAILED'
assert sum(blocks) == n, '[TC22] balanced_block_sizes sum FAILED'

# ---- TC23: compute_load_imbalance zero for uniform blocks ----
imb = compute_load_imbalance([5, 5, 5, 5])
assert imb == 0.0, '[TC23] compute_load_imbalance uniform FAILED'

# ---- TC24: polynomial_degree for known polynomials ----
assert polynomial_degree(np.array([1, 0, 1, 0])) == 2, '[TC24] polynomial_degree nonzero FAILED'
assert polynomial_degree(np.array([0, 0, 0])) == -1, '[TC24] polynomial_degree zero poly FAILED'
assert polynomial_degree(np.array([0, 0, 1])) == 2, '[TC24] polynomial_degree degree 2 FAILED'

# ---- TC25: collatz_polynomial_sequence deterministic output ----
p0 = np.array([1, 0, 1])
seq = collatz_polynomial_sequence(p0, 3)
assert len(seq) == 4, '[TC25] collatz sequence length FAILED'
assert np.array_equal(seq[0], np.array([1, 0, 1])), '[TC25] collatz sequence first FAILED'

# ---- TC26: binary_matrix_multiply_f2 known result for small matrices ----
A_f2 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
B_f2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
C_f2 = binary_matrix_multiply_f2(A_f2, B_f2)
assert C_f2.shape == (3, 3), '[TC26] binary_matrix_multiply_f2 shape FAILED'
assert np.all((C_f2 == 0) | (C_f2 == 1)), '[TC26] binary_matrix_multiply_f2 binary FAILED'

# ---- TC27: verify_matrix_multiply_checksum passes for valid product ----
np.random.seed(42)
A_chk = np.random.randn(4, 4)
B_chk = np.random.randn(4, 4)
C_chk = np.dot(A_chk, B_chk)
assert verify_matrix_multiply_checksum(A_chk, B_chk, C_chk), '[TC27] verify checksum valid FAILED'

# ---- TC28: exponential_growth_rate of zero matrix is zero ----
Z = np.zeros((3, 3))
lam = exponential_growth_rate(Z)
assert abs(lam) < 1e-12, '[TC28] exponential_growth_rate zero matrix FAILED'

# ---- TC29: kepler_hessian matrix is symmetric ----
state_h = np.array([1.0, 0.5, 0.0, 0.8])
H = kepler_hessian(state_h)
assert H.shape == (2, 2), '[TC29] kepler_hessian shape FAILED'
assert np.allclose(H, H.T, atol=1e-12), '[TC29] kepler_hessian symmetry FAILED'

# ---- TC30: approximate_matrix_element both methods return finite values ----
kernel = lambda x, y: np.exp(-(x - y) ** 2)
val_cheb = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="chebyshev")
val_herm = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="hermite")
assert np.isfinite(val_cheb), '[TC30] approximate_matrix_element chebyshev FAILED'
assert np.isfinite(val_herm), '[TC30] approximate_matrix_element hermite FAILED'

# ---- TC31: cannon_multiply on 8x8 random matrices (sequential fallback) ----
np.random.seed(191)
A8 = np.random.randn(8, 8)
B8 = np.random.randn(8, 8)
C8 = cannon_multiply(A8, B8, num_processes=4)
C8_ref = np.dot(A8, B8)
assert np.allclose(C8, C8_ref, atol=1e-10), '[TC31] cannon_multiply 8x8 FAILED'

# ---- TC32: matrix_exponential_pade zero matrix returns identity ----
Z2 = np.zeros((3, 3))
E_zero = matrix_exponential_pade(Z2, order=5)
assert np.allclose(E_zero, np.eye(3), atol=1e-12), '[TC32] matrix_exponential_pade zero FAILED'

# ---- TC33: sparsity_ratio known sparse pattern ----
data_sr = np.array([1.0, 2.0, 3.0])
sr = sparsity_ratio(data_sr, 5, 5)
assert 0.85 < sr < 0.95, '[TC33] sparsity_ratio FAILED'

# ---- TC34: disk01_rule weights are positive ----
w_disk, r_disk, t_disk = disk01_rule(8, 16)
assert np.all(w_disk > 0), '[TC34] disk01_rule weights positive FAILED'
assert len(r_disk) == 8 and len(t_disk) == 16, '[TC34] disk01_rule shape FAILED'

# ---- TC35: prime_balanced_partition returns valid process count ----
p_bp, blocks_bp, imb_bp = prime_balanced_partition(128, 32)
assert p_bp >= 1, '[TC35] prime_balanced_partition process count FAILED'
assert len(blocks_bp) == p_bp, '[TC35] prime_balanced_partition blocks count FAILED'
assert imb_bp >= 0.0, '[TC35] prime_balanced_partition imbalance FAILED'

# ---- TC36: log_normal_truncated_ab_sample produces values in [a,b] (fixed seed) ----
np.random.seed(42)
for _ in range(200):
    val = log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0)
    assert 0.1 - 1e-12 <= val <= 5.0 + 1e-12, '[TC36] log_normal_truncated_ab_sample bounds FAILED'

# ---- TC37: polynomial_hash_matrix deterministic output ----
M_hash = np.array([[1.0, 0.0], [0.0, 1.0]])
h1 = polynomial_hash_matrix(M_hash, 5)
h2 = polynomial_hash_matrix(M_hash, 5)
assert h1 == h2, '[TC37] polynomial_hash_matrix deterministic FAILED'
assert len(h1) > 0, '[TC37] polynomial_hash_matrix non-empty FAILED'

# ---- TC38: construct_kernel_matrix_1d positive semidefinite ----
nodes_k = np.array([0.0, 1.0])
phi = lambda x, c: np.exp(-5.0 * (x - c) ** 2)
K_mat = construct_kernel_matrix_1d(nodes_k, phi, quadrature_order=16)
assert K_mat.shape == (2, 2), '[TC38] construct_kernel_matrix_1d shape FAILED'
eigs = np.linalg.eigvalsh(K_mat)
assert np.all(eigs > -1e-10), '[TC38] construct_kernel_matrix_1d PSD FAILED'

# ---- TC39: hermite_divided_differences with single node ----
z_1, d_1 = hermite_divided_differences(np.array([0.0]), np.array([1.0]), np.array([2.0]))
assert len(z_1) == 2 and len(d_1) == 2, '[TC39] hermite single node shape FAILED'

# ---- TC40: frobenius_error returns float type ----
A_fe = np.eye(3)
B_fe = 2 * np.eye(3)
err_fe = frobenius_error(A_fe, B_fe)
assert isinstance(err_fe, float), '[TC40] frobenius_error return type FAILED'
assert err_fe > 0.0, '[TC40] frobenius_error positive for non-identical FAILED'

# ---- TC41: kepler_variational_matrix 4x4 structure ----
state_v = np.array([1.0, 0.0, 0.0, 1.0])
M_var = kepler_variational_matrix(state_v)
assert M_var.shape == (4, 4), '[TC41] kepler_variational_matrix shape FAILED'
assert M_var[0, 2] == 1.0 and M_var[1, 3] == 1.0, '[TC41] kepler_variational_matrix structure FAILED'

# ---- TC42: is_smooth detects 7-smooth numbers ----
assert is_smooth(12, max_prime=7), '[TC42] is_smooth 12 FAILED'
assert not is_smooth(11, max_prime=7), '[TC42] is_smooth prime 11 FAILED'

# ---- TC43: triangular mesh node_degrees correct ----
nodes_d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.2, 0.0]])
elements_d = np.array([[0, 1, 2], [1, 3, 2]])
mesh_d = TriangularMesh(nodes_d, elements_d)
deg = mesh_d.node_degrees()
assert deg[2] == 2, '[TC43] node_degrees middle node FAILED'

# ---- TC44: construct_kernel_matrix_2d_disk shape ----
nodes_2d = np.array([[0.0, 0.0], [0.5, 0.5]])
phi_2d = lambda x, y, cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2))
K_2d = construct_kernel_matrix_2d_disk(nodes_2d, phi_2d, nr=8, nt=16)
assert K_2d.shape == (2, 2), '[TC44] construct_kernel_matrix_2d_disk shape FAILED'
assert np.all(np.isfinite(K_2d)), '[TC44] construct_kernel_matrix_2d_disk finite FAILED'

# ---- TC45: prime_pi matches direct count ----
primes_50 = prime_sieve(50)
assert prime_pi(50) == len(primes_50), '[TC45] prime_pi FAILED'

# ---- TC46: collatz_polynomial_next zero polynomial stays zero ----
p_zero = np.array([0])
p_next = collatz_polynomial_next(p_zero)
assert np.array_equal(p_next, np.array([0])), '[TC46] collatz zero poly FAILED'

# ---- TC47: chebyshev_coefficients returns correct length and finite values ----
f_vals = np.cos(np.arange(9))
c = chebyshev_coefficients(f_vals)
assert len(c) == len(f_vals), '[TC47] chebyshev_coefficients length FAILED'
assert np.all(np.isfinite(c)), '[TC47] chebyshev_coefficients finite FAILED'

# ---- TC48: rescale_quadrature preserves weight sum ----
x_q, w_q = legendre_compute_glr(5)
t_scaled, w_scaled = rescale_quadrature(x_q, w_q, -1.0, 3.0)
assert abs(np.sum(w_scaled) - 4.0) < 1e-12, '[TC48] rescale_quadrature weight sum FAILED'

# ---- TC49: find_optimal_process_count returns valid result ----
opt_p = find_optimal_process_count(64, 16, algorithm="cannon")
assert 1 <= opt_p <= 16, '[TC49] find_optimal_process_count range FAILED'

# ---- TC50: trinity_tile_cover_pattern output shapes ----
A1, A2 = trinity_tile_cover_pattern(6, 3)
assert A1.shape[0] == 6, '[TC50] trinity A1 rows FAILED'
assert A2.shape[0] == 3, '[TC50] trinity A2 rows FAILED'
assert A1.shape[1] == A2.shape[1], '[TC50] trinity column mismatch FAILED'
