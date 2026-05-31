# ---- TC01: validate_array_1d 输入一维数组应返回展平结果 ----
import numpy as np
result = validate_array_1d(np.array([1.0, 2.0, 3.0]))
assert result.shape == (3,) and np.allclose(result, [1.0, 2.0, 3.0]), '[TC01] validate_array_1d basic FAILED'

# ---- TC02: validate_array_1d 空数组应抛出 ValueError ----
try:
    validate_array_1d(np.array([]))
    assert False, '[TC02] validate_array_1d empty should raise FAILED'
except ValueError:
    pass

# ---- TC03: safe_inverse 正常求逆 ----
x = np.array([0.5, 2.0, -1.0])
inv = safe_inverse(x)
assert np.allclose(inv, [2.0, 0.5, -1.0]), '[TC03] safe_inverse basic FAILED'

# ---- TC04: safe_inverse 近零值安全处理（不产生 NaN/Inf） ----
x = np.array([0.0, 1e-16, 1e-13])
inv = safe_inverse(x)
assert np.all(np.isfinite(inv)), '[TC04] safe_inverse near-zero safety FAILED'

# ---- TC05: build_sparse_hamiltonian_indices 结构验证 ----
rows, cols, data = build_sparse_hamiltonian_indices(5)
assert rows.size == cols.size == data.size, '[TC05] sparse indices size mismatch FAILED'
n_diag = np.sum(rows == cols)
assert n_diag == 5, '[TC05] diagonal count FAILED'
assert np.all(data[rows == cols] == 2.0), '[TC05] diagonal elements not 2.0 FAILED'

# ---- TC06: spmatvec 空稀疏矩阵 × 向量 = 零向量 ----
rows = np.array([], dtype=int); cols = np.array([], dtype=int); data = np.array([], dtype=float)
vec = np.array([1.0, 2.0, 3.0])
result = spmatvec(rows, cols, data, vec)
assert np.allclose(result, [0.0, 0.0, 0.0]), '[TC06] spmatvec zero FAILED'

# ---- TC07: tridiagonal_solve 已知三对角系统精确求解 ----
n = 5
a = np.ones(n - 1); b = np.full(n, 2.0); c = np.ones(n - 1); d = np.arange(1, n + 1, dtype=float)
x = tridiagonal_solve(a, b, c, d)
T = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
assert np.allclose(T @ x, d, atol=1e-10), '[TC07] tridiagonal_solve FAILED'

# ---- TC08: estimate_condition_number_dense 单位矩阵条件数近似为 1 ----
I = np.eye(10)
cond = estimate_condition_number_dense(I)
assert abs(cond - 1.0) < 1e-6, '[TC08] condition number of identity FAILED'

# ---- TC09: effective_mass 已知材料值查表 ----
assert effective_mass("InAs") == 0.023, '[TC09] effective_mass InAs FAILED'
assert effective_mass("GaAs") == 0.067, '[TC09] effective_mass GaAs FAILED'
assert effective_mass("InP") == 0.077, '[TC09] effective_mass InP FAILED'

# ---- TC10: spherical_confinement_potential 内部零外部 V0 ----
r = np.array([1.0e-9, 3.0e-9, 5.0e-9, 7.0e-9, 10.0e-9])
V0 = 1.0e-19
V = spherical_confinement_potential(r, 5.0e-9, V0)
assert V[0] == 0.0 and V[1] == 0.0 and V[2] == 0.0, '[TC10] spherical potential inside FAILED'
assert V[3] == V0 and V[4] == V0, '[TC10] spherical potential outside FAILED'

# ---- TC11: reduced_mass 约化质量公式验证 ----
mu = reduced_mass(0.023, 0.067)
expected = (0.023 * 0.067) / (0.023 + 0.067)
assert abs(mu - expected) < 1e-10, '[TC11] reduced_mass FAILED'

# ---- TC12: solve_eigenvalues_1d 本征值递增且基态波函数归一化 ----
x_grid = np.linspace(-1.5e-8, 1.5e-8, 100)
result = solve_eigenvalues_1d(x_grid, 0.023, potential_type="spherical", R_dot=5.0e-9, V0=0.5 * 1.602176634e-19)
energies = result["energies_eV"]
assert np.all(np.diff(energies) >= -1e-12), '[TC12] eigenvalues not sorted ascending FAILED'
wf = result["wavefunctions"][:, 0]
assert abs(np.sum(wf ** 2) - 1.0) < 1e-6, '[TC12] wavefunction discrete normalization FAILED'

# ---- TC13: triangle_area 已知三角形面积 (0,0)-(1,0)-(0,1) = 0.5 ----
p1 = np.array([0.0, 0.0]); p2 = np.array([1.0, 0.0]); p3 = np.array([0.0, 1.0])
area = triangle_area(p1, p2, p3)
assert abs(area - 0.5) < 1e-10, '[TC13] triangle_area FAILED'

# ---- TC14: quadrilateral_area 单位正方形面积为 1 ----
square = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
area = quadrilateral_area(square)
assert abs(area - 1.0) < 1e-10, '[TC14] quadrilateral_area unit square FAILED'

# ---- TC15: compute_mesh_areas 所有单元面积为正且总和不超过圆面积 ----
R = 1.0e-6
nodes, elements = generate_circular_domain_nodes(R, n_r=5, n_theta=8)
elem_areas, mesh_area = compute_mesh_areas(nodes, elements)
assert np.all(elem_areas > 0), '[TC15] element areas not all positive FAILED'
assert mesh_area > 0, '[TC15] total area not positive FAILED'
assert mesh_area < np.pi * R ** 2 * 1.1, '[TC15] total area exceeds circle FAILED'

# ---- TC16: points_in_polygon 射线法点在多边形内/外判定 ----
poly = np.array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
inside = points_in_polygon(np.array([1.0, 0.5, 3.0]), np.array([1.0, 0.5, 3.0]), poly)
assert inside[0] == True, '[TC16] point (1,1) inside FAILED'
assert inside[1] == True, '[TC16] point (0.5,0.5) inside FAILED'
assert inside[2] == False, '[TC16] point (3,3) outside FAILED'

# ---- TC17: gaussian_mode_profile 峰值正确且单调衰减 ----
x = np.array([0.0, 1.0, 2.0])
y = np.array([0.0, 0.0, 0.0])
E = gaussian_mode_profile(x, y, 0.0, 0.0, 1.0, amplitude=2.0)
assert abs(E[0] - 2.0) < 1e-10, '[TC17] gaussian peak FAILED'
assert E[2] < E[1] < E[0], '[TC17] gaussian monotonic decay FAILED'

# ---- TC18: purcell_factor 正值 ----
Fp = purcell_factor(1e4, 1e-20, 930e-9, n_eff=3.5)
assert Fp > 0, '[TC18] purcell_factor not positive FAILED'

# ---- TC19: jaynes_cummings_hamiltonian Hermitian 性 ----
H = jaynes_cummings_hamiltonian(1e15, 1e15, 1e11, n_photon_cutoff=3)
assert np.allclose(H, H.conj().T), '[TC19] JC Hamiltonian not Hermitian FAILED'

# ---- TC20: vectorize/unvectorize_density_matrix 往返恒等 ----
rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex)
vec = vectorize_density_matrix(rho)
rho2 = unvectorize_density_matrix(vec, 2)
assert np.allclose(rho, rho2), '[TC20] vectorize roundtrip FAILED'

# ---- TC21: lindblad_dissipator 零跳跃算符应返回全零 ----
rho = np.eye(2, dtype=complex) / 2.0
L = np.zeros((2, 2), dtype=complex)
D_val = lindblad_dissipator(rho, L)
assert np.allclose(D_val, 0.0), '[TC21] zero jump operator FAILED'

# ---- TC22: disk_unit_sample 所有点在单位圆盘内（固定种子） ----
np.random.seed(42)
pts = disk_unit_sample(100)
r_sq = pts[0, :] ** 2 + pts[1, :] ** 2
assert np.all(r_sq <= 1.0 + 1e-12), '[TC22] disk samples outside unit circle FAILED'

# ---- TC23: second_order_correlation_weak_coupling g2(0) < 1（反聚束） ----
tau = np.array([0.0])
g2 = second_order_correlation_weak_coupling(tau, 1e9, 2e9, 5e8)
assert g2[0] < 1.0, '[TC23] g2(0) not below 1 FAILED'

# ---- TC24: antibunching_parameter 分类正确 ----
assert antibunching_parameter(0.1) == "strong_antibunching", '[TC24] strong antibunching FAILED'
assert antibunching_parameter(0.7) == "weak_antibunching", '[TC24] weak antibunching FAILED'
assert antibunching_parameter(1.5) == "bunching_or_poissonian", '[TC24] bunching FAILED'

# ---- TC25: photon_indistinguishability_homodyne 理想情况 V=1 ----
V = photon_indistinguishability_homodyne(1.0, 0.0, 1e-10)
assert abs(V - 1.0) < 1e-10, '[TC25] perfect indistinguishability FAILED'

# ---- TC26: photon_indistinguishability_homodyne 输出范围 [0,1] ----
V2 = photon_indistinguishability_homodyne(0.5, 1e10, 1e-9)
assert 0.0 <= V2 <= 1.0, '[TC26] indistinguishability range FAILED'

# ---- TC27: detection_area_efficiency 输出范围 [0,1] ----
eta = detection_area_efficiency(1e-3, 1e-3, 0.0)
assert 0.0 <= eta <= 1.0, '[TC27] detection efficiency range FAILED'

# ---- TC28: criterion_variance 非负且同类为零 ----
data = np.random.randn(2, 50)
np.random.seed(42)
labels = np.zeros(50, dtype=int)
labels[25:] = 1
crit = criterion_variance(data, labels, 2)
assert crit >= 0, '[TC28] criterion_variance negative FAILED'

# ---- TC29: classify_quantum_dot_ensemble 输出结构完整 ----
np.random.seed(42)
energies = np.concatenate([np.random.normal(1.32, 0.02, 10), np.random.normal(1.35, 0.015, 10)])
linewidths = np.concatenate([np.random.normal(0.05, 0.01, 10), np.random.normal(0.03, 0.008, 10)])
result = classify_quantum_dot_ensemble(energies, linewidths, n_clusters=2)
assert "labels" in result and "cluster_centers" in result, '[TC29] classify output structure FAILED'
assert len(result["labels"]) == 20, '[TC29] labels count FAILED'

# ---- TC30: roughness_induced_broadening 正值 ----
Gamma = roughness_induced_broadening(0.2, 5.0, m_star_ratio=0.023)
assert Gamma > 0, '[TC30] roughness broadening not positive FAILED'

# ---- TC31: int_to_binary_vector 已知值 6(10) = 0110(2) ----
bv = int_to_binary_vector(6, 4)
assert np.array_equal(bv, [0, 1, 1, 0]), '[TC31] int_to_binary_vector FAILED'

# ---- TC32: single_photon_figure_of_merit 理想情况正值 ----
fom = single_photon_figure_of_merit(0.01, 100, 0.9, 1e8)
assert fom > 0, '[TC32] FoM not positive FAILED'

# ---- TC33: solve_steady_state 返回 Hermitian 矩阵且输出的物理量非负 ----
H_test = jaynes_cummings_hamiltonian(1e15, 1e15, 1e11, n_photon_cutoff=3)
sigma_m = np.zeros((6, 6), dtype=complex)
for n in range(3):
    sigma_m[2 * n, 2 * n + 1] = 1.0
a_ann = np.zeros((6, 6), dtype=complex)
for n in range(1, 3):
    a_ann[2 * (n - 1), 2 * n] = np.sqrt(n)
    a_ann[2 * (n - 1) + 1, 2 * n + 1] = np.sqrt(n)
jump_ops = [sigma_m, a_ann]
gamma_rates = np.array([1e8, 1e9])
rho_ss = solve_steady_state(H_test, jump_ops, gamma_rates)
assert np.allclose(rho_ss, rho_ss.conj().T), '[TC33] steady state not Hermitian FAILED'
p_e = excited_state_population(rho_ss)
n_ph = cavity_photon_number(rho_ss, 3)
assert 0.0 <= p_e <= 1.0, '[TC33] excited population out of range FAILED'
assert n_ph >= 0, '[TC33] photon number negative FAILED'

# ---- TC34: generate_rough_quantum_dot_boundary 输出结构与 D_f 范围 ----
np.random.seed(42)
boundary, D_f = generate_rough_quantum_dot_boundary(5.0e-9, n_vertices=16, mu_perturb=0.01, n_iter=2)
assert boundary.shape[0] == 2, '[TC34] boundary shape FAILED'
assert 0.0 < D_f <= 2.0, '[TC34] fractal dimension range FAILED'

# ---- TC35: 可复现性 - 固定种子两次调用 disk_distance_stats 结果一致 ----
np.random.seed(42)
mu1, var1 = disk_distance_stats(n_samples=500)
np.random.seed(42)
mu2, var2 = disk_distance_stats(n_samples=500)
assert abs(mu1 - mu2) < 1e-12, '[TC35] reproducibility mean FAILED'
assert abs(var1 - var2) < 1e-12, '[TC35] reproducibility variance FAILED'
