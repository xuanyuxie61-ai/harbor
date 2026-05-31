# ---- TC01: SemanticEmbeddingBases正交基数量为正整数 ----
bases_tc01 = SemanticEmbeddingBases(radius=1.0, max_mode_m=4, max_mode_n=3)
assert bases_tc01.num_bases > 0, '[TC01] num_bases should be positive FAILED'
assert isinstance(bases_tc01.num_bases, int), '[TC01] num_bases should be int FAILED'

# ---- TC02: 语义场重构误差为有限值 ----
import numpy as np
np.random.seed(42)
bases_tc02 = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
r_grid = np.linspace(0.01, 1.0, 20)
theta_grid = np.linspace(0, 2 * np.pi, 30)
R_m, T_m = np.meshgrid(r_grid, theta_grid)
r_flat = R_m.flatten()
theta_flat = T_m.flatten()
semantic_field = np.exp(-((r_flat - 0.5) ** 2 + (np.sin(theta_flat)) ** 2) / 0.1)
coeffs = bases_tc02.project_semantic_vector(semantic_field, r_flat, theta_flat)
reconstructed = bases_tc02.reconstruct_semantic_field(coeffs, r_flat, theta_flat)
error = np.linalg.norm(semantic_field - reconstructed) / np.linalg.norm(semantic_field)
assert np.isfinite(error), '[TC02] reconstruction error should be finite FAILED'
assert error >= 0.0, '[TC02] reconstruction error should be non-negative FAILED'
assert error < 1.0, '[TC02] reconstruction error should be < 1.0 FAILED'

# ---- TC03: 正交基Gram矩阵对角元接近1 ----
import numpy as np
bases_tc03 = SemanticEmbeddingBases(radius=1.0, max_mode_m=3, max_mode_n=2)
r_grid2 = np.linspace(0.01, 1.0, 15)
theta_grid2 = np.linspace(0, 2 * np.pi, 30)
R2, T2 = np.meshgrid(r_grid2, theta_grid2)
r_flat2 = R2.flatten()
theta_flat2 = T2.flatten()
gram = bases_tc03.basis_orthogonality_check(r_flat2, theta_flat2)
assert gram.shape == (bases_tc03.num_bases, bases_tc03.num_bases), '[TC03] Gram matrix shape wrong FAILED'
assert np.all(np.abs(np.diag(gram) - 1.0) < 0.1), '[TC03] Gram diagonal should be close to 1 FAILED'

# ---- TC04: FEM投影解形状与节点数一致 ----
import numpy as np
fem_tc04 = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
def sem_density(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + x
U = fem_tc04.project(sem_density)
assert len(U) == fem_tc04.node_num, '[TC04] FEM solution length should equal node_num FAILED'
assert np.all(np.isfinite(U)), '[TC04] FEM solution should be all finite FAILED'

# ---- TC05: FEM L2误差为非负有限值 ----
import numpy as np
fem_tc05 = FEM2DSemanticProjection(xl=0.0, xr=1.0, yb=0.0, yt=1.0, nx=9, ny=9)
def sem_density2(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y) + x
U2 = fem_tc05.project(sem_density2)
errors = fem_tc05.compute_l2_error(U2, sem_density2)
assert errors['relative_error'] >= 0.0, '[TC05] relative_error should be non-negative FAILED'
assert np.isfinite(errors['relative_error']), '[TC05] relative_error should be finite FAILED'

# ---- TC06: 热扩散解长度正确 ----
import numpy as np
diff_tc06 = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
def cond(x):
    return 1.0 + 0.5 * np.sin(np.pi * x) ** 2
def src(x):
    return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)
U_diff = diff_tc06.solve(1.0, 0.0, cond, src)
assert len(U_diff) == 41, '[TC06] solution length should be 41 FAILED'
assert np.all(np.isfinite(U_diff)), '[TC06] solution should be finite FAILED'

# ---- TC07: 热扩散守恒性（总热源≈净热流） ----
import numpy as np
diff_tc07 = SemanticHeatDiffusion(n=41, a=0.0, b=1.0)
def cond2(x):
    return 1.0 + 0.5 * np.sin(np.pi * x) ** 2
def src2(x):
    return 2.0 * np.exp(-((x - 0.3) ** 2) / 0.01)
U2_diff = diff_tc07.solve(1.0, 0.0, cond2, src2)
flux = diff_tc07.compute_flux(U2_diff, cond2)
total_source = np.trapezoid([src2(xi) for xi in diff_tc07.x], diff_tc07.x)
net_flux = flux[-1] - flux[0]
conservation_error = abs(total_source - net_flux) / (abs(total_source) + 1e-15)
assert conservation_error < 0.05, '[TC07] heat conservation error too large FAILED'

# ---- TC08: 反应动力学精确解守恒 ----
import numpy as np
rxn_tc08 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0, t0=0.0, tstop=50.0)
t_test = np.linspace(0.0, 50.0, 100)
w1_ex, w2_ex = rxn_tc08.exact_solution(t_test)
conserved = rxn_tc08.conserved_quantity(w1_ex, w2_ex)
assert np.all(np.abs(conserved - conserved[0]) < 1e-12), '[TC08] exact solution should conserve w1+w2 FAILED'

# ---- TC09: 数值解与精确解偏差可控 ----
import numpy as np
rxn_tc09 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0, t0=0.0, tstop=50.0)
t_num, w1_num, w2_num = rxn_tc09.solve_numerical(num_points=200)
w1_exact, w2_exact = rxn_tc09.exact_solution(t_num)
error1 = np.max(np.abs(w1_num - w1_exact))
error2 = np.max(np.abs(w2_num - w2_exact))
assert error1 < 1e-6, '[TC09] w1 numerical error too large FAILED'
assert error2 < 1e-6, '[TC09] w2 numerical error too large FAILED'

# ---- TC10: 稳态平衡值满足w1_eq + w2_eq = w10 + w20 ----
rxn_tc10 = SemanticReactionDynamics(k1=0.3, k2=0.1, w10=1.0, w20=0.0)
w1_eq, w2_eq = rxn_tc10.equilibrium()
assert abs(w1_eq + w2_eq - (rxn_tc10.w10 + rxn_tc10.w20)) < 1e-12, '[TC10] equilibrium must conserve total mass FAILED'

# ---- TC11: Van der Pol周期估计为正有限值 ----
vdp_tc11 = VanDerPolSemanticOscillator(mu=1.0)
p_est = vdp_tc11.period_estimate()
assert p_est > 0.0, '[TC11] Van der Pol period estimate should be positive FAILED'
assert np.isfinite(p_est), '[TC11] Van der Pol period estimate should be finite FAILED'

# ---- TC12: 捕食者-猎物守恒能量为有限值 ----
pp_tc12 = PredatorPreySemanticCycle(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075, u0=10.0, v0=5.0)
E = pp_tc12.conserved_energy()
assert np.isfinite(E), '[TC12] conserved energy should be finite FAILED'

# ---- TC13: CVT能量单调不增 ----
import numpy as np
np.random.seed(42)
cvt_tc13 = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
result_tc13 = cvt_tc13.quantize(init_mode='random', seed=42)
energies = result_tc13['energy_history']
is_decreasing = all(energies[i] >= energies[i + 1] - 1e-14 for i in range(len(energies) - 1))
assert is_decreasing, '[TC13] CVT energy should be monotonically non-increasing FAILED'

# ---- TC14: CVT量化误差小于均匀量化误差 ----
import numpy as np
np.random.seed(42)
cvt_tc14 = SemanticSpaceQuantization(n_generators=10, max_iter=200, tol=1e-12)
result_tc14 = cvt_tc14.quantize(init_mode='random', seed=42)
uniform = np.linspace(0.0, 1.0, cvt_tc14.n + 2)[1:-1]
test_pts = np.linspace(0.0, 1.0, 1000)
error_cvt = cvt_tc14.quantization_error(result_tc14['generators'], test_pts)
error_uniform = cvt_tc14.quantization_error(uniform, test_pts)
assert error_cvt <= error_uniform, '[TC14] CVT error should not exceed uniform error FAILED'

# ---- TC15: TSP优化后代价不大于顺序路径代价 ----
import numpy as np
np.random.seed(42)
D_tc15 = generate_semantic_distance_matrix(10, dim=8, seed=42)
opt_tc15 = SemanticPathOptimizer(D_tc15, seed=42)
random_path = np.arange(10)
random_cost = opt_tc15.path_cost(random_path)
result_tc15 = opt_tc15.multi_start_optimize(num_starts=5, max_variations=3000)
assert result_tc15['cost'] <= random_cost, '[TC15] optimized cost should not exceed sequential cost FAILED'

# ---- TC16: 子集和暴力搜索精确解 ----
import numpy as np
sel_tc16 = SemanticSubsetSelector(max_brute_force_n=20)
weights = np.array([1, 2, 4, 8, 16, 32])
target = 22.0
result_tc16 = sel_tc16.brute_force_search(weights, target)
assert result_tc16['found_exact'], '[TC16] should find exact solution for 2+4+16=22 FAILED'
assert abs(result_tc16['subset_sum'] - target) < 1e-8, '[TC16] subset sum should match target FAILED'

# ---- TC17: Mandelbrot逃逸时间矩阵形状正确 ----
import numpy as np
mbd_tc17 = MandelbrotSemanticBoundary(escape_radius=2.0)
result_tc17 = mbd_tc17.compute_region(x_min=-1.0, x_max=-0.6, y_min=0.0, y_max=0.4, nx=41, ny=41, max_iter=30)
assert result_tc17.shape == (41, 41), '[TC17] Mandelbrot result shape should be (41, 41) FAILED'
assert np.all(result_tc17 >= 0), '[TC17] all escape times should be non-negative FAILED'
assert np.all(result_tc17 <= 30), '[TC17] all escape times should not exceed max_iter FAILED'

# ---- TC18: IFS Lyapunov指数为有限值 ----
import numpy as np
np.random.seed(42)
ifs_tc18 = IFSSemanticTransformer(seed=42)
lyap = ifs_tc18.lyapunov_exponent(num_iterations=5000)
assert np.isfinite(lyap), '[TC18] Lyapunov exponent should be finite FAILED'

# ---- TC19: Fermat因式分解验证 ----
decomp_tc19 = SemanticStructuredDecomposition()
f1, f2 = decomp_tc19.fermat_factor(91)
assert f1 * f2 == 91, '[TC19] Fermat: 91 = f1 * f2 FAILED'

# ---- TC20: LHS样本在[0,1]范围内 ----
import numpy as np
np.random.seed(42)
sampler_tc20 = SemanticSpaceSampler(seed=42)
samples = sampler_tc20.latin_hypercube_sampling(n_samples=50, n_dims=5)
assert samples.shape == (50, 5), '[TC20] LHS sample shape should be (50, 5) FAILED'
assert np.all(samples >= 0.0), '[TC20] LHS samples should be >= 0 FAILED'
assert np.all(samples <= 1.0), '[TC20] LHS samples should be <= 1 FAILED'

# ---- TC21: Steinerberger精确积分公式 ----
verif_tc21 = SteinerbergerVerifier()
exact_n5 = verif_tc21.exact_integral(5)
H5 = float(np.sum(1.0 / np.arange(1, 6)))
expected = (2.0 / np.pi) * H5
assert abs(exact_n5 - expected) < 1e-14, '[TC21] exact integral formula FAILED'

# ---- TC22: safe_divide除零返回默认值 ----
utils_tc22 = RobustNumericUtils()
assert utils_tc22.safe_divide(1.0, 0.0) == 0.0, '[TC22] safe_divide(1, 0) should return default 0.0 FAILED'
assert utils_tc22.safe_divide(6.0, 3.0) == 2.0, '[TC22] safe_divide(6, 3) should be 2.0 FAILED'
assert utils_tc22.safe_log(0.0) == -700.0, '[TC22] safe_log(0) should return default -700.0 FAILED'
assert utils_tc22.safe_sqrt(-1.0) == 0.0, '[TC22] safe_sqrt(-1) should return default 0.0 FAILED'

# ---- TC23: 语义相似度对称且自身相似度为1 ----
import numpy as np
utils_tc23 = RobustNumericUtils()
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.array([0.0, 1.0, 0.0])
sim12 = utils_tc23.semantic_similarity_safe(e1, e2)
sim21 = utils_tc23.semantic_similarity_safe(e2, e1)
sim11 = utils_tc23.semantic_similarity_safe(e1, e1)
assert abs(sim12 - sim21) < 1e-14, '[TC23] cosine similarity should be symmetric FAILED'
assert abs(sim11 - 1.0) < 1e-14, '[TC23] self similarity should be 1.0 FAILED'

# ---- TC24: 零向量语义相似度为0 ----
import numpy as np
utils_tc24 = RobustNumericUtils()
e_zero = np.array([0.0, 0.0, 0.0])
e_norm = np.array([1.0, 2.0, 3.0])
sim_zero = utils_tc24.semantic_similarity_safe(e_zero, e_norm)
assert sim_zero == 0.0, '[TC24] similarity with zero vector should be 0 FAILED'

# ---- TC25: 可复现性：固定种子两次CVT结果相同 ----
import numpy as np
np.random.seed(42)
cvt_tc25a = SemanticSpaceQuantization(n_generators=6, max_iter=100, tol=1e-10)
res_a = cvt_tc25a.quantize(init_mode='random', seed=123)
np.random.seed(42)
cvt_tc25b = SemanticSpaceQuantization(n_generators=6, max_iter=100, tol=1e-10)
res_b = cvt_tc25b.quantize(init_mode='random', seed=123)
assert np.allclose(res_a['generators'], res_b['generators']), '[TC25] CVT results should be reproducible with same seed FAILED'

# ---- TC26: 病态矩阵安全求解不崩溃 ----
import numpy as np
utils_tc26 = RobustNumericUtils()
A = np.array([[1.0, 1.0], [1.0, 1.0000001]])
b = np.array([2.0, 2.0])
x = utils_tc26.solve_linear_system_safe(A, b)
assert len(x) == 2, '[TC26] solution should have length 2 FAILED'
assert np.all(np.isfinite(x)), '[TC26] solution should be finite FAILED'

# ---- TC27: 质因数分解正确性 ----
decomp_tc27 = SemanticStructuredDecomposition()
primes_12 = decomp_tc27.prime_factors(12)
assert sorted(primes_12) == [2, 2, 3], '[TC27] prime factors of 12 should be [2,2,3] FAILED'
primes_768 = decomp_tc27.prime_factors(768)
prod_768 = 1
for p in primes_768:
    prod_768 *= p
assert prod_768 == 768, '[TC27] product of prime factors should equal 768 FAILED'

# ---- TC28: LHS方向采样范数接近1 ----
import numpy as np
np.random.seed(42)
sampler_tc28 = SemanticSpaceSampler(seed=42)
directions = sampler_tc28.uniform_direction_sampling(200, 8)
norms = np.linalg.norm(directions, axis=1)
assert np.all(np.abs(norms - 1.0) < 1e-12), '[TC28] direction vector norms should be 1.0 FAILED'

# ---- TC29: 多概念网络稳态守恒 ----
import numpy as np
n = 4
K = np.zeros((n, n))
for i in range(n):
    j_next = (i + 1) % n
    K[j_next, i] = 0.2
    K[i, i] -= 0.2
    j_prev = (i - 1) % n
    K[j_prev, i] = 0.1
    K[i, i] -= 0.1
y0 = np.array([1.0, 0.0, 0.0, 0.0])
network_tc29 = MultiConceptReactionNetwork(n, K, y0)
y_eq = network_tc29.equilibrium_state()
assert abs(np.sum(y_eq) - np.sum(y0)) < 1e-10, '[TC29] equilibrium state should conserve total mass FAILED'

# ---- TC30: 弛豫时间为正值 ----
rxn_tc30 = SemanticReactionDynamics(k1=0.3, k2=0.1)
tau = rxn_tc30.relaxation_time()
assert tau > 0.0, '[TC30] relaxation time should be positive FAILED'
assert np.isfinite(tau), '[TC30] relaxation time should be finite FAILED'

print('\n全部 30 个测试通过!\n')
