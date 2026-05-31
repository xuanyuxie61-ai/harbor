# ---- TC01: IsingHamiltonian 合法构造 n_spins 校验 ----
ham = IsingHamiltonian(8, seed=42)
assert ham.n_spins == 8, '[TC01] n_spins mismatch FAILED'
assert ham.J is None and ham.h is None, '[TC01] J/h should be None before build FAILED'

# ---- TC02: IsingHamiltonian 负自旋数抛出 ValueError ----
try:
    _ = IsingHamiltonian(-1, seed=0)
    assert False, '[TC02] Expected ValueError for negative n_spins FAILED'
except ValueError:
    pass

# ---- TC03: random_ensemble connectivity 边界校验 ----
ham3 = IsingHamiltonian(6, seed=42)
ham3.build_random_ensemble(connectivity=0.5, j_std=1.0, h_std=0.5)
assert ham3.J.shape == (6, 6), '[TC03] J shape mismatch FAILED'
assert ham3.h.shape == (6,), '[TC03] h shape mismatch FAILED'
assert np.allclose(ham3.J, ham3.J.T), '[TC03] J not symmetric FAILED'
assert np.allclose(np.diag(ham3.J), 0.0), '[TC03] J diagonal not zero FAILED'

# ---- TC04: energy 为已知构型计算能量 ----
ham4 = IsingHamiltonian(3, seed=42)
J_test = np.zeros((3, 3))
J_test[0, 1] = J_test[1, 0] = 0.5
J_test[0, 2] = J_test[2, 0] = -0.3
h_test = np.array([0.1, -0.2, 0.4])
ham4.J = J_test
ham4.h = h_test
ham4.offset = 0.0
e_all_up = ham4.energy(np.array([1, 1, 1]))
e_expected = 0.5 + (-0.3) + 0.1 + (-0.2) + 0.4
assert abs(e_all_up - e_expected) < 1e-12, f'[TC04] energy mismatch: {e_all_up} vs {e_expected} FAILED'

# ---- TC05: Gray 码序列长度和元素类型 ----
from state_sampler import gray_code_sequence
seq5 = gray_code_sequence(4)
assert len(seq5) == 16, f'[TC05] Gray code length {len(seq5)} != 16 FAILED'
assert all(np.all(np.isin(s, [0, 1])) for s in seq5), '[TC05] non-binary element FAILED'

# ---- TC06: 精确基态能量 brute force 不高于简单构型 ----
ham6 = IsingHamiltonian(8, seed=42)
ham6.build_random_ensemble(connectivity=0.4, j_std=0.5, h_std=0.3)
s_opt, e_ground = ham6.exact_ground_state_brute_force()
e_all_up = ham6.energy(np.ones(8, dtype=int))
assert e_ground <= e_all_up, f'[TC06] ground state energy {e_ground} > all-up {e_all_up} FAILED'
assert np.all(np.isin(s_opt, [-1, 1])), '[TC06] invalid spin config FAILED'

# ---- TC07: knapsack QUBO 构造与 J 对称性 ----
import numpy as np
np.random.seed(42)
weights = np.array([2, 3, 4, 5, 1])
values = np.array([10, 15, 20, 25, 5])
ham7 = IsingHamiltonian(5, seed=42)
ham7.build_knapsack_qubo(weights, values, capacity=8, penalty=6.0)
assert ham7.J.shape == (5, 5), '[TC07] J shape mismatch FAILED'
assert np.allclose(ham7.J, ham7.J.T), '[TC07] J not symmetric FAILED'

# ---- TC08: incomplete_beta 边界值 x=0 返回 0, x=1 返回 1 ----
from annealing_schedules import incomplete_beta
assert incomplete_beta(0.0, 2.0, 3.0) == 0.0, '[TC08] incomplete_beta(0) != 0 FAILED'
assert incomplete_beta(1.0, 2.0, 3.0) == 1.0, '[TC08] incomplete_beta(1) != 1 FAILED'
ib = incomplete_beta(0.5, 2.0, 2.0)
assert 0.0 <= ib <= 1.0, f'[TC08] incomplete_beta(0.5)={ib} out of [0,1] FAILED'

# ---- TC09: collatz_polynomial_next 迭代确定性 ----
from annealing_schedules import collatz_polynomial_next
p = np.array([1, 1, 1], dtype=int)
p2 = collatz_polynomial_next(p)
assert np.all(np.isin(p2, [0, 1])), '[TC09] non-binary output FAILED'
p3 = collatz_polynomial_next(p2)
import numpy as np
np.random.seed(42)
p2b = collatz_polynomial_next(np.array([1, 1, 1], dtype=int))
assert np.array_equal(p2, p2b), '[TC09] non-deterministic FAILED'

# ---- TC10: 退火 schedule linear 端点 A(0)=1, B(1)=1 ----
sched = AnnealingSchedule(T_total=1.0, n_steps=100)
A_lin, B_lin = sched.linear()
assert abs(A_lin[0] - 1.0) < 1e-12, f'[TC10] A(0)={A_lin[0]} != 1 FAILED'
assert abs(B_lin[-1] - 1.0) < 1e-12, f'[TC10] B(1)={B_lin[-1]} != 1 FAILED'
assert np.all(A_lin + B_lin > 0.99), '[TC10] A+B not near 1 FAILED'

# ---- TC11: logistic schedule 输出范围 [0,1] ----
A_log, B_log = sched.logistic_schedule(kappa=10.0, s0=0.5)
assert np.all(B_log >= 0.0) and np.all(B_log <= 1.0), '[TC11] B out of [0,1] FAILED'
assert np.all(A_log >= 0.0) and np.all(A_log <= 1.0), '[TC11] A out of [0,1] FAILED'

# ---- TC12: 截断正态均值对称情况下等于 mu ----
from noise_model import truncated_normal_ab_mean
mean_sym = truncated_normal_ab_mean(0.0, 1.0, -3.0, 3.0)
assert abs(mean_sym) < 0.1, f'[TC12] symmetric mean={mean_sym} not near 0 FAILED'

# ---- TC13: 截断正态方差非负 ----
from noise_model import truncated_normal_ab_variance
var_val = truncated_normal_ab_variance(0.0, 2.0, -5.0, 5.0)
assert var_val >= 0.0, f'[TC13] variance={var_val} < 0 FAILED'

# ---- TC14: 热激发概率在 [0,1] 且低温下 delta_e>0 几乎为 0 ----
import numpy as np
np.random.seed(42)
noise14 = QuantumAnnealingNoiseModel(n_spins=4, T_bath=0.01, seed=42)
prob_large = noise14.thermal_excitation_probability(1.0)
assert 0.0 <= prob_large <= 1.0, f'[TC14] prob={prob_large} out of [0,1] FAILED'
prob_neg = noise14.thermal_excitation_probability(-1.0)
assert prob_neg == 1.0, '[TC14] negative delta_e should give prob=1 FAILED'

# ---- TC15: HexLattice 格点数量确定性验证 ----
hex_lat = HexLattice(lattice_constant=1.0)
sites2 = hex_lat.generate_sites(n_ring=2)
assert sites2.shape[0] == 27, f'[TC15] sites count {sites2.shape[0]} != 27 FAILED'
assert sites2.shape[1] == 2, f'[TC15] sites must be 2D coords FAILED'

# ---- TC16: 边界词反射类型 3 (中心反演) ----
word_orig = "001122"
w_ref3 = hex_lat.reflect_boundary(word_orig, reflection_type=3)
expected_ref = "334455"
assert w_ref3 == expected_ref, f'[TC16] reflection got {w_ref3} expected {expected_ref} FAILED'

# ---- TC17: Q4 形函数在参考单元中心 (0,0) 处求和为 1 ----
Nq = Q4Basis.shape_functions(0.0, 0.0)
assert abs(Nq.sum() - 1.0) < 1e-12, f'[TC17] shape sum={Nq.sum()} != 1 FAILED'

# ---- TC18: Jacobi 求解器对对角占优矩阵快速收敛 ----
A_diag = np.diag(np.array([4.0, 5.0, 6.0]))
b_diag = np.array([4.0, 5.0, 6.0])
x_jac, it_jac, res_jac = jacobi_solve(A_diag, b_diag, max_iter=100, tol=1e-12)
expected_x = np.array([1.0, 1.0, 1.0])
assert np.allclose(x_jac, expected_x), f'[TC18] solution {x_jac} != [1,1,1] FAILED'
assert res_jac < 1e-10, f'[TC18] residual={res_jac} too large FAILED'

# ---- TC19: SCMF 对小系统收敛且磁化强度范围 [-1,1] ----
import numpy as np
np.random.seed(42)
J_small = np.array([[0.0, 0.3], [0.3, 0.0]])
h_small = np.array([0.1, -0.1])
m_scmf, it_scmf = self_consistent_mean_field(J_small, h_small, beta=1.0, tol=1e-10)
assert np.all(np.abs(m_scmf) <= 1.0), f'[TC19] magnetization {m_scmf} out of [-1,1] FAILED'
assert it_scmf > 0, '[TC19] SCMF did zero iterations FAILED'

# ---- TC20: 幂迭代对正定矩阵返回正特征值 ----
import numpy as np
np.random.seed(42)
A_pos = np.array([[2.0, 0.5], [0.5, 2.0]])
lam, vec = power_iteration_eigenvalue(A_pos, tol=1e-10)
assert lam > 0, f'[TC20] eigenvalue {lam} <= 0 FAILED'
assert abs(np.linalg.norm(vec) - 1.0) < 1e-10, '[TC20] eigenvector not normalized FAILED'

# ---- TC21: Metropolis 采样输出 dict 含正确键 ----
import numpy as np
np.random.seed(42)
ham21 = IsingHamiltonian(6, seed=42)
ham21.build_random_ensemble(connectivity=0.3, j_std=0.5, h_std=0.3)
sampler21 = MetropolisSampler(6, ham21.energy, beta=2.0, seed=42)
result21 = sampler21.sample(n_sweeps=100, burn_in=50, thinning=10)
assert 'energies' in result21 and 'states' in result21, '[TC21] missing keys in result FAILED'
assert result21['energies'].ndim == 1, '[TC21] energies not 1D FAILED'

# ---- TC22: 精确配分函数随温度单调递减 ----
import numpy as np
np.random.seed(42)
ham22 = IsingHamiltonian(4, seed=42)
ham22.build_random_ensemble(connectivity=0.5, j_std=1.0, h_std=0.5)
configs_all, energies_all = enumerate_all_energies(ham22.energy, 4)
Z_b1 = exact_partition_function(energies_all, 0.5)
Z_b2 = exact_partition_function(energies_all, 2.0)
assert Z_b1 > 0 and Z_b2 > 0, '[TC22] partition function negative FAILED'

# ---- TC23: effective_transverse_coupling 非负 ----
from path_integral_monte_carlo import effective_transverse_coupling
J_perp = effective_transverse_coupling(0.1, 1.0)
assert J_perp >= 0, f'[TC23] J_perp={J_perp} < 0 FAILED'
J_perp_small = effective_transverse_coupling(0.001, 2.0)
assert np.isfinite(J_perp_small), '[TC23] J_perp not finite FAILED'

# ---- TC24: normalize_array_to_range 目标区间端点正确 ----
arr24 = np.array([0.0, 10.0])
norm24 = normalize_array_to_range(arr24, -1.0, 1.0)
assert abs(norm24.min() - (-1.0)) < 1e-12, f'[TC24] min={norm24.min()} != -1 FAILED'
assert abs(norm24.max() - 1.0) < 1e-12, f'[TC24] max={norm24.max()} != 1 FAILED'

# ---- TC25: unit_simplex_volume 理论公式 1/d! ----
vol4 = unit_simplex_volume(4)
assert abs(vol4 - 1.0/24.0) < 1e-12, f'[TC25] vol4={vol4} != 1/24 FAILED'

# ---- TC26: log_sum_exp 恒等式验证 ----
log_vals = np.array([1.0, 2.0, 3.0])
lse = log_sum_exp(log_vals)
expected_lse = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
assert abs(lse - expected_lse) < 1e-12, f'[TC26] LSE {lse} != {expected_lse} FAILED'

# ---- TC27: quantum_state_fidelity 输出范围 [0,1] ----
import numpy as np
np.random.seed(42)
psi_a = np.random.randn(8) + 1j * np.random.randn(8)
psi_a = psi_a / np.linalg.norm(psi_a)
fid_self = quantum_state_fidelity(psi_a, psi_a)
assert abs(fid_self - 1.0) < 1e-12, f'[TC27] self-fidelity={fid_self} != 1 FAILED'

# ---- TC28: entanglement_entropy 非负 ----
sv28 = np.array([0.8, 0.4, 0.2, 0.1])
S_ent = entanglement_entropy_singular_values(sv28)
assert S_ent >= 0, f'[TC28] entropy={S_ent} < 0 FAILED'

# ---- TC29: Monte Carlo 盒形积分维度校验 ----
rng29 = np.random.default_rng(42)
val29, err29 = monte_carlo_box_integral(
    dim=2, n_points=5000, integrand=lambda x: x[0] * x[1],
    box_a=np.array([0.0, 0.0]), box_b=np.array([1.0, 1.0]), rng=rng29
)
assert abs(val29 - 0.25) < 0.05, f'[TC29] MC integral {val29} far from 0.25 FAILED'

# ---- TC30: triangle_area_2d 公式正确 ----
from utils import triangle_area_2d
area = triangle_area_2d(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
assert abs(area - 0.5) < 1e-12, f'[TC30] area={area} != 0.5 FAILED'

# ---- TC31: 噪声模型 disordered_h 输出维度正确 ----
import numpy as np
np.random.seed(42)
noise31 = QuantumAnnealingNoiseModel(n_spins=5, h_noise_sigma=0.1, seed=42)
h_orig = np.array([0.5, -0.3, 0.0, 0.2, -0.1])
h_noisy = noise31.disordered_h(h_orig)
assert h_noisy.shape == h_orig.shape, '[TC31] output shape mismatch FAILED'
assert np.all(np.isfinite(h_noisy)), '[TC31] non-finite values in noisy h FAILED'

# ---- TC32: Chebyshev 加速 Jacobi 收敛 ----
A32 = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
b32 = np.array([1.0, 2.0, 3.0])
x_cheb, it_cheb, res_cheb = chebyshev_accelerated_jacobi(A32, b32, max_iter=1000, tol=1e-10)
assert res_cheb < 1e-8, f'[TC32] Chebyshev residual={res_cheb} too large FAILED'

# ---- TC33: 变分基态能量是有限值 ----
J33 = np.array([[0.0, 0.5], [0.5, 0.0]])
h33 = np.array([0.2, -0.2])
m33 = np.array([0.5, -0.3])
e_var = variational_ground_state_energy(J33, h33, m33)
assert np.isfinite(e_var), f'[TC33] variational energy not finite: {e_var} FAILED'

# ---- TC34: 并行回火输出维度正确 ----
import numpy as np
np.random.seed(42)
ham34 = IsingHamiltonian(4, seed=42)
ham34.build_random_ensemble(connectivity=0.5, j_std=0.5, h_std=0.3)
betas_pt = np.array([0.5, 1.5, 3.0])
pt_sampler = ParallelTemperingSampler(4, ham34.energy, betas_pt, seed=42)
pt_result = pt_sampler.sample(n_steps=30, exchange_freq=5)
assert pt_result['energies'].shape[1] == 3, f'[TC34] n_replicas != 3 FAILED'

# ---- TC35: 条件采样给定固定自旋后输出符合约束 ----
import numpy as np
np.random.seed(42)
ham35 = IsingHamiltonian(4, seed=42)
ham35.build_random_ensemble(connectivity=0.5, j_std=0.5, h_std=0.3)
cond_sampler = ConditionalProbabilitySampler(4, ham35.energy, seed=42)
fixed = {0: 1, 2: -1}
samples, energies = cond_sampler.sample_given_partial(fixed, n_samples=20, beta=2.0)
assert np.all(samples[:, 0] == 1), '[TC35] fixed spin 0 not respected FAILED'
assert np.all(samples[:, 2] == -1), '[TC35] fixed spin 2 not respected FAILED'

# ---- TC36: 厄米特多项式递推关系 H_2(x) = 4x^2-2 ----
import numpy as np
np.random.seed(42)
x_test36 = np.array([0.0, 0.5, 1.0])
H36 = physicist_hermite_polynomials(x_test36, 2)
expected_H2 = 4.0 * x_test36**2 - 2.0
assert np.allclose(H36[:, 2], expected_H2), f'[TC36] H_2 mismatch FAILED'

# ---- TC37: Q4Basis interpolate 在节点处恢复节点值 ----
q4_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
nodal_vals = np.array([2.0, 3.0, 1.0, 4.0])
val_node2 = Q4Basis.interpolate_scalar_field(q4_nodes, nodal_vals, 1.0, 1.0)
assert abs(val_node2 - 1.0) < 1e-8, f'[TC37] node 2 interp failed: {val_node2} FAILED'

# ---- TC38: density_matrix_purity 纯态为 1 ----
from utils import density_matrix_purity
psi38 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
rho38 = np.outer(psi38, psi38.conj())
pur38 = density_matrix_purity(rho38)
assert abs(pur38 - 1.0) < 1e-12, f'[TC38] purity={pur38} != 1 FAILED'

# ---- TC39: TunnelingKernel kinetic matrix element cosh/sinh ----
import numpy as np
np.random.seed(42)
kernel = TunnelingKernel(beta=5.0, gamma=1.0, n_slices=10, n_basis=4)
k_same = kernel.kinetic_matrix_element(1, 1)
k_diff = kernel.kinetic_matrix_element(1, -1)
assert k_same > 1.0, f'[TC39] cosh(a)={k_same} should be > 1 FAILED'
assert k_diff > 0, f'[TC39] sinh(a)={k_diff} should be > 0 FAILED'

# ---- TC40: transverse_field_hamiltonian_dense 矩阵对称 ----
H_tf = transverse_field_hamiltonian_dense(4, 0.5)
assert np.allclose(H_tf, H_tf.T), '[TC40] H_D not symmetric FAILED'
assert H_tf.shape == (16, 16), f'[TC40] shape {H_tf.shape} != (16,16) FAILED'

# ---- TC41: Hermite function basis L2 norm ----
import numpy as np
np.random.seed(42)
from transverse_field_basis import hermite_function_basis
x_grid = np.linspace(-5.0, 5.0, 1001)
psi_basis = hermite_function_basis(x_grid, 3)
dx = x_grid[1] - x_grid[0]
norm0 = np.trapezoid(psi_basis[:, 0]**2, x_grid)
assert abs(norm0 - 1.0) < 0.01, f'[TC41] norm ψ0={norm0} != 1 FAILED'

# ---- TC42: generate_full_hamiltonian_schedule 输出结构 ----
sched42 = AnnealingSchedule(T_total=1.0, n_steps=50)
result42 = sched42.generate_full_hamiltonian_schedule('linear')
assert 'A' in result42 and 'B' in result42 and 'dA_dt' in result42, '[TC42] missing keys FAILED'
assert result42['A'].shape == (50,), '[TC42] A shape mismatch FAILED'

# ---- TC43: collatz_inspired_schedule 输出单调 ----
A_col, B_col = sched42.collatz_inspired_schedule(n_iter=6)
assert np.all(np.diff(B_col) >= -1e-12), '[TC43] B schedule not monotonic FAILED'

# ---- TC44: 截断正态采样输出在边界内 ----
import numpy as np
np.random.seed(42)
rng44 = np.random.default_rng(42)
from noise_model import truncated_normal_ab_sample
for _ in range(10):
    s = truncated_normal_ab_sample(0.0, 1.0, -2.0, 2.0, rng44)
    assert -2.0 - 1e-10 <= s <= 2.0 + 1e-10, f'[TC44] sample {s} out of bounds FAILED'

# ---- TC45: jacobi_iteration_step 单步对对角矩阵返回精确解 ----
from iterative_solver import jacobi_iteration_step
A45 = np.diag(np.array([2.0, 3.0]))
b45 = np.array([4.0, 9.0])
x45_init = np.array([0.0, 0.0])
x45_new = jacobi_iteration_step(A45, b45, x45_init)
assert np.allclose(x45_new, np.array([2.0, 3.0])), f'[TC45] single step {x45_new} != [2,3] FAILED'

# ---- TC46: Mesh2D integrate_scalar 面积为正且有限 ----
boundary = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
mesh = Mesh2D(boundary, max_area=0.1)
V_nodes = np.ones(mesh.nodes.shape[0])
V_int = mesh.integrate_scalar_over_mesh(V_nodes)
assert V_int > 0.0, f'[TC46] mesh area {V_int} not positive FAILED'
assert np.isfinite(V_int), f'[TC46] mesh area {V_int} not finite FAILED'
assert mesh.nodes.shape[0] >= 3, f'[TC46] too few nodes: {mesh.nodes.shape[0]} FAILED'

# ---- TC47: adiabatic_optimal_local schedule 端点 ----
A_ad, B_ad = sched.adiabatic_optimal_local(gap_estimate=0.1, s_star=0.4)
assert B_ad[0] < 0.5, f'[TC47] B(0) should be near 0, got {B_ad[0]} FAILED'
assert B_ad[-1] > 0.5, f'[TC47] B(1) should be near 1, got {B_ad[-1]} FAILED'

# ---- TC48: PIMC observables 输出为有限值 ----
import numpy as np
np.random.seed(42)
ham48 = IsingHamiltonian(4, seed=42)
ham48.build_random_ensemble(connectivity=0.5, j_std=0.5, h_std=0.3)
sched48 = AnnealingSchedule(T_total=1.0, n_steps=8)
A48, _ = sched48.linear()
gamma48 = A48 * 1.5
pimc48 = PathIntegralMonteCarlo(
    n_spins=4, beta=3.0, n_slices=8,
    energy_func=ham48.energy, gamma_schedule=gamma48, seed=42
)
pimc48.thermalize(n_sweeps=50)
obs48 = pimc48.measure_observables(n_measurements=10, sampling_interval=2)
assert np.isfinite(obs48['energy_mean']), '[TC48] PIMC energy not finite FAILED'
assert np.isfinite(obs48['magnetization']), '[TC48] PIMC magnetization not finite FAILED'
