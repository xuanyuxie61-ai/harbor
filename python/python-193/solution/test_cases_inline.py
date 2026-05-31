# ---- TC01: safe_divide scalar basic division ----
assert abs(safe_divide(10.0, 2.0) - 5.0) < 1e-15, '[TC01] safe_divide scalar 10/2 FAILED'

# ---- TC02: safe_divide scalar zero denominator returns default ----
assert safe_divide(10.0, 0.0, default=99.0) == 99.0, '[TC02] safe_divide zero denom returns default FAILED'

# ---- TC03: safe_divide vectorized with zero entries ----
result = safe_divide(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 2.0]), default=-1.0)
assert result[0] == 1.0 and result[1] == -1.0 and abs(result[2] - 1.5) < 1e-15, '[TC03] safe_divide vectorized FAILED'

# ---- TC04: i4_div_rounded basic rounding ----
from utils import i4_div_rounded
assert i4_div_rounded(7, 3) == 2, '[TC04] i4_div_rounded 7/3 FAILED'
assert i4_div_rounded(8, 3) == 3, '[TC04b] i4_div_rounded 8/3 FAILED'

# ---- TC05: double_factorial2 known values ----
from utils import double_factorial2
assert double_factorial2(0) == 1.0, '[TC05] double_factorial2(0) FAILED'
assert double_factorial2(1) == 1.0, '[TC05b] double_factorial2(1) FAILED'
assert double_factorial2(5) == 15.0, '[TC05c] double_factorial2(5)=15 FAILED'
assert double_factorial2(6) == 48.0, '[TC05d] double_factorial2(6)=48 FAILED'

# ---- TC06: legendre_monomial_integral odd returns 0 ----
for p in [1, 3, 5, 7]:
    assert legendre_monomial_integral(p) == 0.0, f'[TC06] legendre odd p={p} FAILED'

# ---- TC07: legendre_monomial_integral even matches analytic 2/(p+1) ----
for p in [0, 2, 4, 6]:
    expected = 2.0 / (p + 1.0)
    assert abs(legendre_monomial_integral(p) - expected) < 1e-15, f'[TC07] legendre even p={p} FAILED'

# ---- TC08: chebyshev1_monomial_integral odd returns 0 ----
for p in [1, 3, 5]:
    assert chebyshev1_monomial_integral(p) == 0.0, f'[TC08] chebyshev1 odd p={p} FAILED'

# ---- TC09: hermite_monomial_integral even p=0 matches sqrt(pi) ----
assert abs(hermite_monomial_integral(0) - math.sqrt(math.pi)) < 1e-15, '[TC09] hermite p=0 FAILED'
assert abs(hermite_monomial_integral(2) - math.sqrt(math.pi) / 2.0) < 1e-15, '[TC09b] hermite p=2 FAILED'

# ---- TC10: laguerre_monomial_integral matches p! ----
for p in [0, 1, 2, 3, 4]:
    assert abs(laguerre_monomial_integral(p) - float(math.factorial(p))) < 1e-15, f'[TC10] laguerre p={p} FAILED'

# ---- TC11: r8vec_bracket basic search ----
from utils import r8vec_bracket
x_sorted = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
assert r8vec_bracket(x_sorted, 0.3) == 0, '[TC11] r8vec_bracket 0.3 FAILED'
assert r8vec_bracket(x_sorted, 0.7) == 1, '[TC11b] r8vec_bracket 0.7 FAILED'
assert r8vec_bracket(x_sorted, 1.2) == 2, '[TC11c] r8vec_bracket 1.2 FAILED'

# ---- TC12: density_transform output range [0,1] ----
from mesh_cvt import density_transform
import numpy as np
np.random.seed(42)
s_test = np.random.random(200)
d_test = density_transform(s_test)
assert np.all(d_test >= 0.0) and np.all(d_test <= 1.0), '[TC12] density_transform range FAILED'

# ---- TC13: cvt_generate reproducibility with fixed seed ----
import numpy as np
np.random.seed(42)
g1 = cvt_generate(n_generators=16, n_samples=1000, n_iterations=10, seed=123)
np.random.seed(42)
g2 = cvt_generate(n_generators=16, n_samples=1000, n_iterations=10, seed=123)
assert np.allclose(g1, g2), '[TC13] cvt_generate reproducibility FAILED'

# ---- TC14: cvt_generate output dimensions ----
g3 = cvt_generate(n_generators=25, n_samples=2000, n_iterations=5, seed=77)
assert g3.shape == (25, 2), '[TC14] cvt_generate shape FAILED'
assert np.all(g3 >= 0.0) and np.all(g3 <= 1.0), '[TC14b] cvt_generate bounds FAILED'

# ---- TC15: element_area known right triangle ----
from mesh_cvt import element_area
nodes_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
elems_tri = np.array([[0, 1, 2]])
areas = element_area(nodes_tri, elems_tri)
assert abs(areas[0] - 0.5) < 1e-15, '[TC15] element_area FAILED'

# ---- TC16: compute_mesh_quality equilateral triangle returns 1 ----
nodes_eq = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]])
elems_eq = np.array([[0, 1, 2]])
qual = compute_mesh_quality(nodes_eq, elems_eq)
assert abs(qual[0] - 1.0) < 1e-12, '[TC16] compute_mesh_quality equilateral FAILED'

# ---- TC17: gauss_legendre_nodes_weights sum of weights equals 2 ----
from fem_assembler import gauss_legendre_nodes_weights
for n in [1, 2, 3, 4, 5]:
    _, w = gauss_legendre_nodes_weights(n)
    assert abs(np.sum(w) - 2.0) < 1e-14, f'[TC17] GL weights sum n={n} FAILED'

# ---- TC18: triangle_quadrature_3 weights sum to 0.5 ----
from fem_assembler import triangle_quadrature_3
_, w3_q = triangle_quadrature_3()
assert abs(np.sum(w3_q) - 0.5) < 1e-15, '[TC18] triangle_quadrature_3 weight sum FAILED'

# ---- TC19: triangle_quadrature_7 weights sum to 0.5 ----
from fem_assembler import triangle_quadrature_7
_, w7_q = triangle_quadrature_7()
assert abs(np.sum(w7_q) - 0.5) < 1e-15, '[TC19] triangle_quadrature_7 weight sum FAILED'

# ---- TC20: exactness_test_fem_quadrature passes for max_degree=5 ----
assert exactness_test_fem_quadrature(max_degree=5) == True, '[TC20] exactness_test_fem_quadrature FAILED'

# ---- TC21: clenshaw_curtis_nodes endpoints ----
from sparse_grid import clenshaw_curtis_nodes
for n in [2, 3, 5, 9]:
    cc = clenshaw_curtis_nodes(n)
    assert abs(cc[0] - 1.0) < 1e-15, f'[TC21] CC left endpoint n={n} FAILED'
    assert abs(cc[-1] + 1.0) < 1e-15, f'[TC21b] CC right endpoint n={n} FAILED'

# ---- TC22: clenshaw_curtis_weights are positive and finite ----
from sparse_grid import clenshaw_curtis_weights
for n in [3, 5, 9, 17]:
    w_cc = clenshaw_curtis_weights(n)
    assert np.all(w_cc > 0), f'[TC22] CC weights positive n={n} FAILED'
    assert np.all(np.isfinite(w_cc)), f'[TC22b] CC weights finite n={n} FAILED'

# ---- TC23: sparse_grid_integrate on constant function returns finite result ----
import numpy as np
np.random.seed(42)
def const_func(x):
    return 1.0
for d in [1, 2, 3]:
    I_const = sparse_grid_integrate(const_func, d, L=3)
    assert np.isfinite(I_const), f'[TC23] sparse_grid_integrate const d={d} finite FAILED'
    assert abs(I_const) > 0, f'[TC23b] sparse_grid_integrate const d={d} positive FAILED'

# ---- TC24: sparse_grid_integrate returns finite result for Genz function ----
import numpy as np
np.random.seed(42)
d24 = 2
a24 = np.ones(d24) * 0.5
c24 = 2.0 * math.pi * 0.5
def genz_2d(x):
    return np.cos(c24 + np.dot(a24, x))
I24 = sparse_grid_integrate(genz_2d, d24, L=4)
assert np.isfinite(I24), '[TC24] sparse_grid Genz finite FAILED'

# ---- TC25: jacobi_preconditioner shape and finite ----
A25 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
m25 = jacobi_preconditioner(A25)
assert len(m25) == 3, '[TC25] jacobi_preconditioner shape FAILED'
assert np.all(np.isfinite(m25)), '[TC25b] jacobi_preconditioner finite FAILED'

# ---- TC26: pcg_solve identity matrix converges in 1 iteration ----
A26 = np.eye(10)
b26 = np.ones(10)
x26, info26 = pcg_solve(A26, b26, tol=1e-12)
assert info26['converged'] == True, '[TC26] pcg_solve identity converged FAILED'
assert info26['iterations'] <= 1, '[TC26b] pcg_solve identity iterations FAILED'

# ---- TC27: task_division covers all tasks ----
divs = task_division(100, 0, 3)
total_tasks = sum(e - s + 1 for _, s, e in divs)
assert total_tasks == 100, '[TC27] task_division coverage FAILED'

# ---- TC28: cycle_decomposition known permutation ----
perm28 = np.array([1, 2, 0, 4, 3])
cycles = cycle_decomposition(perm28)
assert len(cycles) == 2, '[TC28] cycle_decomposition count FAILED'
cycle_lens = sorted([len(c) for c in cycles])
assert cycle_lens == [2, 3], f'[TC28b] cycle_decomposition lengths FAILED got {cycle_lens}'

# ---- TC29: random_permutation reproducibility ----
import numpy as np
np.random.seed(42)
p1 = random_permutation(20, seed=99)
np.random.seed(42)
p2 = random_permutation(20, seed=99)
assert np.array_equal(p1, p2), '[TC29] random_permutation reproducibility FAILED'

# ---- TC30: bandwidth of known diagonal matrix equals 1 ----
A30 = np.diag(np.arange(1, 11, dtype=float))
assert bandwidth(A30) == 1, '[TC30] bandwidth diagonal FAILED'

# ---- TC31: reverse_cuthill_mckee returns valid permutation ----
A31 = np.array([[5, 1, 0, 0], [1, 4, 1, 0], [0, 1, 3, 1], [0, 0, 1, 2]], dtype=float)
order31 = reverse_cuthill_mckee(A31)
assert len(order31) == 4, '[TC31] RCM length FAILED'
assert set(order31) == set(range(4)), '[TC31b] RCM valid permutation FAILED'

# ---- TC32: apply_reordering preserves matrix shape ----
A32 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
perm32 = np.array([2, 0, 1])
A_perm = apply_reordering(A32, perm32)
assert A_perm.shape == (3, 3), '[TC32] apply_reordering shape FAILED'

# ---- TC33: logistic_reaction_diffusion_matrix is tridiagonal and SPD ----
A33 = logistic_reaction_diffusion_matrix(20, D=0.5, r=1.0, K=1.0)
assert A33.shape == (20, 20), '[TC33] logistic matrix shape FAILED'
eig33 = np.linalg.eigvalsh(A33)
assert np.all(eig33 > 0), '[TC33b] logistic matrix SPD FAILED'

# ---- TC34: feynman_kac_exact at boundary a ----
a34 = 2.0
u_exact_a = feynman_kac_exact(a34, np.array([a34]))
assert abs(u_exact_a[0] - 1.0) < 1e-15, '[TC34] feynman_kac_exact at boundary a FAILED'

# ---- TC35: feynman_kac_potential non-negative ----
from benchmark_suite import feynman_kac_potential
x_test_35 = np.linspace(-2.0, 2.0, 50)
V35 = feynman_kac_potential(2.0, x_test_35)
assert np.all(V35 >= 0.0), '[TC35] feynman_kac_potential non-negative FAILED'

# ---- TC36: menger_sponge_hierarchical_matrix SPD ----
from benchmark_suite import menger_sponge_hierarchical_matrix
A36 = menger_sponge_hierarchical_matrix(level=3, epsilon=0.1)
assert A36.shape == (8, 8), '[TC36] menger shape FAILED'
eig36 = np.linalg.eigvalsh(A36)
assert np.all(eig36 > 0), '[TC36b] menger SPD FAILED'

# ---- TC37: candy_circulant_matrix dimensions ----
from benchmark_suite import candy_circulant_matrix
A37 = candy_circulant_matrix(n=32, block_size=5, value=1.0)
assert A37.shape == (32, 32), '[TC37] candy shape FAILED'
eig37 = np.linalg.eigvalsh(A37)
assert np.all(eig37 > 0), '[TC37b] candy SPD FAILED'

# ---- TC38: generate_all_benchmark_matrices returns 5 entries ----
suite = generate_all_benchmark_matrices()
assert len(suite) == 5, '[TC38] benchmark suite count FAILED'
for name in ['logistic_reaction_diffusion', 'feynman_kac_fd', 'chirikov_tiled', 'menger_hierarchical', 'candy_circulant']:
    assert name in suite, f'[TC38b] benchmark suite missing {name} FAILED'

# ---- TC39: csr_matvec matches dense matvec ----
import numpy as np
np.random.seed(42)
row_ptr39 = np.array([0, 2, 4, 5], dtype=int)
col_idx39 = np.array([0, 2, 1, 2, 1], dtype=int)
vals39 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
x39 = np.array([1.0, -1.0, 2.0])
y_csr = csr_matvec(row_ptr39, col_idx39, vals39, x39)
A_dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 0.0]])
y_dense = A_dense @ x39
assert np.allclose(y_csr, y_dense), '[TC39] csr_matvec FAILED'

# ---- TC40: gmres_solve on small SPD system ----
A40 = np.array([[4.0, 1.0], [1.0, 3.0]])
b40 = np.array([1.0, 1.0])
x40, info40 = gmres_solve(A40, b40, tol=1e-10, max_iter=10)
assert info40['converged'] == True, '[TC40] gmres_solve converged FAILED'
res40 = np.linalg.norm(b40 - A40 @ x40) / np.linalg.norm(b40)
assert res40 < 1e-8, '[TC40b] gmres_solve residual FAILED'

# ---- TC41: parallel_matvec matches direct matvec ----
import numpy as np
np.random.seed(42)
A41 = np.random.rand(12, 12)
x41 = np.random.rand(12)
y_par = parallel_matvec(A41, x41, nproc=3)
y_seq = A41 @ x41
assert np.allclose(y_par, y_seq), '[TC41] parallel_matvec FAILED'

# ---- TC42: analyze_permutation_cycles returns expected keys ----
import numpy as np
np.random.seed(42)
stats = analyze_permutation_cycles(n=20, n_trials=100, seed=42)
for key in ['n', 'n_trials', 'expected_total_cycles', 'mean_total_cycles', 'mean_max_cycle_length']:
    assert key in stats, f'[TC42] analyze_permutation_cycles missing {key} FAILED'
assert stats['mean_total_cycles'] > 0, '[TC42b] mean_total_cycles positive FAILED'

# ---- TC43: construct_prolongation_1d fine nodes within coarse range ----
from preconditioner import construct_prolongation_1d
fine43 = np.linspace(0, 1, 10)
coarse43 = np.linspace(0, 1, 4)
P43 = construct_prolongation_1d(fine43, coarse43)
assert P43.shape == (10, 4), '[TC43] prolongation shape FAILED'
assert np.allclose(P43.sum(axis=1), 1.0), '[TC43b] prolongation row sum = 1 FAILED'

# ---- TC44: level_to_n_cc known mapping ----
from sparse_grid import level_to_n_cc
assert level_to_n_cc(1) == 1, '[TC44] level_to_n_cc(1) FAILED'
assert level_to_n_cc(2) == 3, '[TC44b] level_to_n_cc(2) FAILED'
assert level_to_n_cc(3) == 5, '[TC44c] level_to_n_cc(3) FAILED'
assert level_to_n_cc(4) == 9, '[TC44d] level_to_n_cc(4) FAILED'

# ---- TC45: spgetseq d=2, n=2 returns correct 3 combinations ----
from sparse_grid import spgetseq
seqs = spgetseq(2, 2)
assert len(seqs) == 3, '[TC45] spgetseq count FAILED'
assert (0, 2) in seqs, '[TC45b] spgetseq (0,2) FAILED'
assert (1, 1) in seqs, '[TC45c] spgetseq (1,1) FAILED'
assert (2, 0) in seqs, '[TC45d] spgetseq (2,0) FAILED'

# ---- TC46: is_symmetric on symmetric matrix returns True ----
from utils import is_symmetric
A46_sym = np.array([[2.0, 1.0], [1.0, 3.0]])
assert is_symmetric(A46_sym) == True, '[TC46] is_symmetric True FAILED'

# ---- TC47: is_symmetric on non-symmetric matrix returns False ----
A47_nonsym = np.array([[1.0, 2.0], [3.0, 4.0]])
assert is_symmetric(A47_nonsym) == False, '[TC47] is_symmetric False FAILED'

# ---- TC48: node_to_element_average simple triangle ----
from mesh_cvt import node_to_element_average
nodes48 = np.array([0.0, 5.0, 10.0])
elems48 = np.array([[0, 1, 2]])
avg48 = node_to_element_average(nodes48, elems48)
assert abs(avg48[0] - 5.0) < 1e-15, '[TC48] node_to_element_average FAILED'

# ---- TC49: mm_write / mm_read round-trip ----
import os, tempfile
A49 = np.array([[1.0, 2.0], [3.0, 4.0]])
tmpfile = os.path.join(tempfile.gettempdir(), 'test_mm_roundtrip.mtx')
mm_write(tmpfile, A49, title="Test", field="real", symm="general")
A49_read = mm_read(tmpfile)
if hasattr(A49_read, 'toarray'):
    A49_read = A49_read.toarray()
assert np.allclose(A49, A49_read), '[TC49] mm_write/read round-trip FAILED'
os.remove(tmpfile)

# ---- TC50: _dense_to_csc output lengths consistent ----
from sparse_formats import _dense_to_csc
A50 = np.array([[1.0, 0.0], [2.0, 3.0]])
data50, row50, col50 = _dense_to_csc(A50)
assert len(data50) == 3, '[TC50] _dense_to_csc nnz FAILED'
assert len(col50) == 3, '[TC50b] _dense_to_csc col_ptr len FAILED'
assert col50[0] == 0 and col50[-1] == len(data50), '[TC50c] _dense_to_csc col_ptr bounds FAILED'

# ---- TC51: ssor_preconditioner applied on small SPD ----
A51 = np.array([[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]])
M51 = ssor_preconditioner(A51, omega=1.0)
r51 = np.array([1.0, 2.0, 3.0])
z51 = M51(r51)
assert len(z51) == 3, '[TC51] ssor_preconditioner shape FAILED'
assert np.all(np.isfinite(z51)), '[TC51b] ssor_preconditioner finite FAILED'

# ---- TC52: delaunay_triangulation output is 2D int array ----
import numpy as np
np.random.seed(42)
nodes52 = np.random.rand(20, 2)
elems52 = delaunay_triangulation(nodes52)
assert elems52.ndim == 2, '[TC52] delaunay ndim FAILED'
assert elems52.shape[1] == 3, '[TC52b] delaunay 3 columns FAILED'
assert elems52.dtype == np.int64 or elems52.dtype == np.int32, '[TC52c] delaunay dtype int FAILED'

# ---- TC53: sparse_grid_points_weights returns finite positive weights ----
for d53 in [1, 2]:
    pts53, w53 = sparse_grid_points_weights(d53, L=3)
    assert np.all(np.isfinite(w53)), f'[TC53] sparse_grid weights finite d={d53} FAILED'
    assert len(w53) > 0, f'[TC53b] sparse_grid has points d={d53} FAILED'

# ---- TC54: adaptive_sparse_grid_refine on smooth function converges ----
import numpy as np
np.random.seed(42)
def smooth_gauss(x):
    return np.exp(-0.5 * np.sum(x ** 2))
I_adapt, pts, w, vals, L_final = adaptive_sparse_grid_refine(smooth_gauss, d=2, L_max=4, abs_tol=1e-4, rel_tol=1e-3)
assert I_adapt > 0, '[TC54] adaptive sparse grid integral positive FAILED'
assert len(w) > 0, '[TC54b] adaptive sparse grid has points FAILED'

# ---- TC55: multigrid_preconditioner function returns correct shape ----
A55 = np.array([[4.0, 1.0], [1.0, 3.0]])
M55 = multigrid_preconditioner(A55, P=None, smoother_sweeps=2, omega=0.8, max_levels=2)
r55 = np.array([1.0, -1.0])
z55 = M55(r55)
assert len(z55) == 2, '[TC55] multigrid preconditioner shape FAILED'
assert np.all(np.isfinite(z55)), '[TC55b] multigrid preconditioner finite FAILED'
