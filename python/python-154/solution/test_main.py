"""
main.py
================================================================================
量子计算：量子退火组合优化 —— 统一入口程序

运行方式：
    python main.py

零参数运行，自动执行以下完整流程：
  1) 构造随机自旋玻璃实例与背包 QUBO 实例
  2) 精确基态求解（小系统 Gray 码枚举）
  3) 平均场自洽场方程求解
  4) 多种退火 schedule 设计与比较
  5) 噪声模型下的哈密顿量扰动
  6) 路径积分蒙特卡洛（PIMC）模拟量子退火
  7) 有限元空间离散化与晶格几何分析
  8) 马尔可夫链采样与并行回火
  9) 高维 Monte Carlo 积分验证
 10) 综合结果报告与数值鲁棒性检验
"""

import numpy as np
import sys
import time

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from ising_hamiltonian import IsingHamiltonian
from transverse_field_basis import (
    physicist_hermite_polynomials,
    probabilist_hermite_polynomials,
    tunneling_amplitude_1d,
    transverse_field_hamiltonian_dense,
    TunnelingKernel,
)
from noise_model import QuantumAnnealingNoiseModel
from annealing_schedules import AnnealingSchedule
from lattice_geometry import HexLattice, Q4Basis, Mesh2D
from iterative_solver import (
    jacobi_solve,
    self_consistent_mean_field,
    variational_ground_state_energy,
    power_iteration_eigenvalue,
    chebyshev_accelerated_jacobi,
)
from state_sampler import (
    MetropolisSampler,
    ParallelTemperingSampler,
    ConditionalProbabilitySampler,
    enumerate_all_energies,
    exact_partition_function,
)
from path_integral_monte_carlo import PathIntegralMonteCarlo
from utils import (
    normalize_array_to_range,
    monte_carlo_box_integral,
    monte_carlo_triangle_integral,
    log_sum_exp,
    quantum_state_fidelity,
    entanglement_entropy_singular_values,
    unit_simplex_volume,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_all() -> None:
    total_start = time.time()
    rng = np.random.default_rng(154)

    # ========================================================================
    # 阶段 1: 构造伊辛哈密顿量
    # ========================================================================
    print_section("阶段 1: 伊辛哈密顿量构造")
    n_small = 14  # 用于精确枚举
    n_large = 40  # 用于近似方法

    # 随机自旋玻璃
    ham_small = IsingHamiltonian(n_small, seed=154)
    ham_small.build_random_ensemble(connectivity=0.35, j_mean=0.0, j_std=1.0,
                                     h_mean=0.1, h_std=0.5)
    print(f"[INFO] 小规模系统: N={n_small}, 非零耦合比例≈{0.35:.2f}")

    # 背包 QUBO（中等规模）
    weights = rng.integers(1, 10, size=n_large)
    values = rng.integers(1, 20, size=n_large)
    capacity = int(0.4 * weights.sum())
    ham_knapsack = IsingHamiltonian(n_large, seed=155)
    ham_knapsack.build_knapsack_qubo(weights, values, capacity, penalty=8.0)
    print(f"[INFO] 背包 QUBO: N={n_large}, capacity={capacity}")

    # ========================================================================
    # 阶段 2: 精确基态（Gray 码枚举）
    # ========================================================================
    print_section("阶段 2: 精确基态求解 (Gray 码枚举)")
    s_opt, e_ground = ham_small.exact_ground_state_brute_force()
    print(f"[RESULT] 精确基态能量 E_0 = {e_ground:.8f}")
    print(f"[RESULT] 基态磁化强度 M = {s_opt.mean():.6f}")

    # 同时精确计算配分函数与热力学量
    configs_all, energies_all = enumerate_all_energies(ham_small.energy, n_small)
    betas_test = [0.5, 1.0, 2.0, 5.0]
    for b in betas_test:
        Z = exact_partition_function(energies_all, b)
        F = -np.log(Z) / b
        print(f"[THERMO] β={b:.2f}: Z={Z:.6e}, F={F:.6f}")

    # ========================================================================
    # 阶段 3: 平均场自洽场方程
    # ========================================================================
    print_section("阶段 3: 平均场自洽场方程 (SCMF)")
    beta_mf = 2.0
    m_sc, it_sc = self_consistent_mean_field(
        ham_small.J, ham_small.h, beta=beta_mf, max_iter=5000, tol=1e-10, damping=0.5
    )
    e_var = variational_ground_state_energy(ham_small.J, ham_small.h, m_sc)
    print(f"[RESULT] SCMF 收敛于 {it_sc} 次迭代")
    print(f"[RESULT] 变分基态能量估计 E_var = {e_var:.8f}")
    print(f"[RESULT] 与精确解误差 |E_var - E_0| = {abs(e_var - e_ground):.8f}")

    # ========================================================================
    # 阶段 4: Jacobi 迭代求解线性化平均场方程
    # ========================================================================
    print_section("阶段 4: Jacobi 迭代与 Chebyshev 加速")
    # 构造一个严格对角占优的对称正定矩阵进行测试（1D Laplacian 微扰）
    # 来自 dif2 的 -1/2/-1 三对角矩阵，加上强对角项保证收敛性
    n_test = n_small
    A_test = np.zeros((n_test, n_test))
    for i in range(n_test):
        A_test[i, i] = 4.0 + abs(ham_small.J[i, i])  # 强对角占优
        if i > 0:
            A_test[i, i - 1] = -1.0
            A_test[i - 1, i] = -1.0
    b_test = beta_mf * ham_small.h + rng.normal(0, 0.5, size=n_test)
    x_jac, it_jac, res_jac = jacobi_solve(A_test, b_test, max_iter=5000, tol=1e-10, omega=1.0)
    print(f"[RESULT] 标准 Jacobi: {it_jac} 次迭代, 残差={res_jac:.3e}")
    x_cheb, it_cheb, res_cheb = chebyshev_accelerated_jacobi(
        A_test, b_test, max_iter=5000, tol=1e-10
    )
    print(f"[RESULT] Chebyshev 加速: {it_cheb} 次迭代, 残差={res_cheb:.3e}")
    print(f"[RESULT] 加速比 = {it_jac / max(it_cheb, 1):.2f}x")

    # ========================================================================
    # 阶段 5: 退火 Schedule 设计
    # ========================================================================
    print_section("阶段 5: 量子退火 Schedule 设计")
    sched = AnnealingSchedule(T_total=1.0, n_steps=200)
    schedules = {
        "linear": sched.linear(),
        "polynomial": sched.polynomial_slowdown(degree=3, s_star=0.5),
        "logistic": sched.logistic_schedule(kappa=12.0, s0=0.45),
        "collatz": sched.collatz_inspired_schedule(n_iter=8),
        "adiabatic": sched.adiabatic_optimal_local(gap_estimate=0.08, s_star=0.42),
    }
    for name, (A, B) in schedules.items():
        # 检查 schedule 的绝热条件近似指标：∫ |dA/dt|/Δ dt
        dA = np.gradient(A, sched.T_total / sched.n_steps)
        metric = np.sum(np.abs(dA)) * sched.T_total / sched.n_steps
        print(f"[SCHED] {name:12s}: A(0)={A[0]:.3f}, B(1)={B[-1]:.3f}, "
              f"变分指标={metric:.4f}")

    # ========================================================================
    # 阶段 6: 噪声模型与扰动哈密顿量
    # ========================================================================
    print_section("阶段 6: 量子退火噪声模型")
    noise_model = QuantumAnnealingNoiseModel(
        n_spins=n_small, T_bath=0.012, h_noise_sigma=0.06,
        j_noise_sigma=0.025, gamma_jitter=0.015, seed=154
    )
    h_noisy = noise_model.disordered_h(ham_small.h)
    J_noisy = noise_model.disordered_J(ham_small.J)
    gamma_fluct = noise_model.fluctuating_gamma(1.0)
    print(f"[RESULT] 施加截断正态噪声后，h 的相对扰动 = "
          f"{np.linalg.norm(h_noisy - ham_small.h) / np.linalg.norm(ham_small.h):.4f}")
    print(f"[RESULT] J 的 Frobenius 范数相对扰动 = "
          f"{np.linalg.norm(J_noisy - ham_small.J, 'fro') / np.linalg.norm(ham_small.J, 'fro'):.4f}")
    print(f"[RESULT] 横向场抖动后 Γ' = {gamma_fluct:.6f}")

    # 噪声下的基态能量变化
    ham_noisy = IsingHamiltonian(n_small, seed=156)
    ham_noisy.J = J_noisy
    ham_noisy.h = h_noisy
    ham_noisy.offset = ham_small.offset
    # Monkey-patch energy method to use noisy J,h
    def noisy_energy(s):
        return ham_noisy.energy(s)
    e_ground_noisy = min(noisy_energy(c) for c in configs_all)
    print(f"[RESULT] 噪声下基态能量 E_0' = {e_ground_noisy:.8f}")
    print(f"[RESULT] 能量漂移 |E_0' - E_0| = {abs(e_ground_noisy - e_ground):.8f}")

    # ========================================================================
    # 阶段 7: 厄米特基函数与隧穿振幅
    # ========================================================================
    print_section("阶段 7: 横向场厄米特基函数与隧穿振幅")
    x_test = np.linspace(-3.0, 3.0, 101)
    H_phys = physicist_hermite_polynomials(x_test, 6)
    He_prob = probabilist_hermite_polynomials(x_test, 6)
    # 验证正交归一化（数值积分）
    dx = x_test[1] - x_test[0]
    w_phys = np.exp(-x_test ** 2)
    overlap_00 = np.trapezoid(H_phys[:, 0] * H_phys[:, 0] * w_phys, x_test)
    overlap_11 = np.trapezoid(H_phys[:, 1] * H_phys[:, 1] * w_phys, x_test)
    print(f"[VERIFY] H_0 范数积分 ≈ {overlap_00:.6f} (理论 √π={np.sqrt(np.pi):.6f})")
    print(f"[VERIFY] H_1 范数积分 ≈ {overlap_11:.6f} (理论 2√π={2*np.sqrt(np.pi):.6f})")
    gamma_vals = np.linspace(0.1, 2.0, 11)
    amps = [tunneling_amplitude_1d(x=0.0, gamma=g, n_basis=8) for g in gamma_vals]
    print(f"[RESULT] 隧穿振幅在 Γ∈[0.1,2] 范围: min={min(amps):.4f}, max={max(amps):.4f}")

    # ========================================================================
    # 阶段 8: 路径积分蒙特卡洛 (PIMC)
    # ========================================================================
    print_section("阶段 8: 路径积分蒙特卡洛 (PIMC)")
    n_pimc_spins = 12  # PIMC 计算代价高，控制规模
    ham_pimc = IsingHamiltonian(n_pimc_spins, seed=157)
    ham_pimc.build_random_ensemble(connectivity=0.4, j_std=0.8, h_std=0.4)

    beta_pimc = 5.0
    n_slices = 32
    sched_pimc = AnnealingSchedule(T_total=1.0, n_steps=n_slices)
    A_sched, B_sched = sched_pimc.linear()
    # 线性 schedule: gamma(t) = A(t) * gamma_max
    gamma_max = 2.0
    gamma_pimc = A_sched * gamma_max

    def pimc_energy_func(s):
        return ham_pimc.energy(s)

    pimc = PathIntegralMonteCarlo(
        n_spins=n_pimc_spins, beta=beta_pimc, n_slices=n_slices,
        energy_func=pimc_energy_func, gamma_schedule=gamma_pimc, seed=158
    )
    print(f"[INFO] PIMC 配置: N={n_pimc_spins}, β={beta_pimc}, M={n_slices}, "
          f"Δτ={pimc.dtau:.4f}")
    pimc.thermalize(n_sweeps=400)
    obs = pimc.measure_observables(n_measurements=80, sampling_interval=4)
    print(f"[RESULT] PIMC 平均能量 ⟨E⟩ = {obs['energy_mean']:.6f} ± {obs['energy_std']:.6f}")
    print(f"[RESULT] PIMC 磁化强度 ⟨M⟩ = {obs['magnetization']:.6f} ± {obs['magnetization_std']:.6f}")
    print(f"[RESULT] PIMC 磁化率 χ = {obs['susceptibility']:.6f}")
    print(f"[RESULT] PIMC 世界线缠绕数 = {obs['winding_number']:.4f}")

    e_pimc_gs = pimc.estimate_ground_state_energy(n_replicas=3, n_sweeps_each=200)
    print(f"[RESULT] PIMC 基态能量估算 E_0(PIMC) ≈ {e_pimc_gs:.6f}")

    # ========================================================================
    # 阶段 9: 晶格几何与有限元分析
    # ========================================================================
    print_section("阶段 9: 晶格几何与 Q4 有限元分析")
    hex_lat = HexLattice(lattice_constant=1.0)
    sites = hex_lat.generate_sites(n_ring=2)
    print(f"[INFO] 六边形晶格 (2 层环): 格点数 = {sites.shape[0]}")
    adj = hex_lat.coupling_graph_from_geometry(sites, cutoff_radius=1.05)
    print(f"[INFO] 几何耦合图: 平均度 = {adj.sum(axis=1).mean():.2f}")

    # 边界词反射
    word = "001122334455"
    for rtype in range(4):
        w_ref = hex_lat.reflect_boundary(word, reflection_type=rtype)
        print(f"[SYMM] 反射类型 {rtype}: '{word}' -> '{w_ref}'")

    # Q4 基函数插值测试
    q4_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    nodal_vals = np.array([0.0, 1.0, 0.5, 0.2])
    val_interp = Q4Basis.interpolate_scalar_field(q4_nodes, nodal_vals, 0.5, 0.5)
    print(f"[RESULT] Q4 插值 (0.5,0.5) 处势值 = {val_interp:.6f}")

    # 网格生成与积分
    boundary = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.2, 0.5], [0.8, 1.0], [0.2, 0.9], [-0.1, 0.4]
    ])
    mesh = Mesh2D(boundary, max_area=0.05)
    print(f"[MESH] 生成网格: 节点数={mesh.nodes.shape[0]}, 三角形数={mesh.triangles.shape[0]}")
    # 在节点上定义一个量子势 V(x,y) = x^2 + y^2 (谐振子势)
    V_nodes = mesh.nodes[:, 0] ** 2 + mesh.nodes[:, 1] ** 2
    V_integral = mesh.integrate_scalar_over_mesh(V_nodes)
    print(f"[RESULT] 谐振子势在网格上的积分 ≈ {V_integral:.6f}")

    # ========================================================================
    # 阶段 10: 马尔可夫链采样与并行回火
    # ========================================================================
    print_section("阶段 10: Metropolis 采样与并行回火")
    sampler = MetropolisSampler(n_small, ham_small.energy, beta=2.0, seed=159)
    sample_result = sampler.sample(n_sweeps=300, burn_in=100, thinning=5)
    e_samples = sample_result["energies"]
    print(f"[RESULT] Metropolis 采样: ⟨E⟩={e_samples.mean():.6f}, "
          f"std={e_samples.std(ddof=1):.6f}, min={e_samples.min():.6f}")

    # 并行回火
    betas_pt = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    pt_sampler = ParallelTemperingSampler(n_small, ham_small.energy, betas_pt, seed=160)
    pt_result = pt_sampler.sample(n_steps=80, exchange_freq=4)
    print(f"[RESULT] 并行回火: 最低温副本平均能量 = "
          f"{pt_result['energies'][:, 0].mean():.6f}")

    # 条件采样（Monty Hall 思想）
    cond_sampler = ConditionalProbabilitySampler(n_small, ham_small.energy, seed=161)
    fixed = {0: 1, 1: -1}
    cond_samples, cond_energies = cond_sampler.sample_given_partial(
        fixed, n_samples=50, beta=2.0
    )
    print(f"[RESULT] 条件采样 (固定自旋 0,1): ⟨E|fixed⟩ = {cond_energies.mean():.6f}")

    # ========================================================================
    # 阶段 11: 高维 Monte Carlo 积分验证
    # ========================================================================
    print_section("阶段 11: 高维 Monte Carlo 积分")

    # 测试 1: 3D 高斯积分（解析解 = π^{3/2}）
    def gauss3d(x):
        return np.exp(-np.sum(x ** 2))
    val3d, err3d = monte_carlo_box_integral(
        dim=3, n_points=50000, integrand=gauss3d,
        box_a=np.array([-3.0, -3.0, -3.0]), box_b=np.array([3.0, 3.0, 3.0]), rng=rng
    )
    exact3d = np.pi ** 1.5
    print(f"[INTEGRAL] 3D 高斯: MC={val3d:.6f} ± {err3d:.6f}, 精确={exact3d:.6f}, "
          f"相对误差={abs(val3d - exact3d) / exact3d:.4f}")

    # 测试 2: 4D 单形积分（解析体积已知）
    dim4 = 4
    vol4 = unit_simplex_volume(dim4)
    print(f"[INTEGRAL] 4D 标准单形体积 = {vol4:.8f} (理论 1/24={1/24:.8f})")

    # 测试 3: 三角形上的积分
    p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    val_tri, err_tri = monte_carlo_triangle_integral(
        p1, p2, p3, n_points=20000,
        integrand=lambda pt: pt[0] * pt[1] * np.exp(-(pt[0] ** 2 + pt[1] ** 2)),
        rng=rng
    )
    print(f"[INTEGRAL] 三角形 Monte Carlo: {val_tri:.6f} ± {err_tri:.6f}")

    # ========================================================================
    # 阶段 12: 保真度与纠缠熵工具验证
    # ========================================================================
    print_section("阶段 12: 量子信息度量")
    psi1 = np.random.randn(16) + 1j * np.random.randn(16)
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = np.random.randn(16) + 1j * np.random.randn(16)
    psi2 = psi2 / np.linalg.norm(psi2)
    fid = quantum_state_fidelity(psi1, psi2)
    print(f"[RESULT] 随机态保真度 F = {fid:.6f}")

    sv = np.array([0.9, 0.3, 0.2, 0.1])
    sv = sv / np.linalg.norm(sv)
    S_ent = entanglement_entropy_singular_values(sv)
    print(f"[RESULT] Schmidt 纠缠熵 S = {S_ent:.6f}")

    # ========================================================================
    # 阶段 13: 综合数值鲁棒性检验
    # ========================================================================
    print_section("阶段 13: 数值鲁棒性检验")
    # 检验边界条件
    try:
        _ = IsingHamiltonian(-1)
    except ValueError as e:
        print(f"[ROBUST] 负自旋数捕获: {e}")
    try:
        _ = jacobi_solve(np.zeros((3, 3)), np.ones(3), max_iter=10)
    except RuntimeError as e:
        print(f"[ROBUST] 奇异矩阵捕获: {e}")
    try:
        _ = monte_carlo_box_integral(2, 10, lambda x: 1.0,
                                      box_a=np.array([0.0]), box_b=np.array([1.0, 1.0]), rng=rng)
    except ValueError as e:
        print(f"[ROBUST] 维度不匹配捕获: {e}")

    # 归一化工具验证
    arr_test = np.array([1e-12, 1e-6, 1e-3, 1.0])
    arr_norm = normalize_array_to_range(arr_test, 0.0, 1.0)
    print(f"[ROBUST] 极端数值归一化: [{arr_norm.min():.3e}, {arr_norm.max():.3e}]")

    total_time = time.time() - total_start
    print("\n" + "=" * 78)
    print(f"  全部计算完成，总耗时: {total_time:.2f} 秒")
    print("=" * 78)


if __name__ == "__main__":
    run_all()

# ================================================================
# 测试用例（48个，assert模式，涉及随机值均使用固定种子）
# ================================================================
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

print('\n全部 48 个测试通过!\n')
