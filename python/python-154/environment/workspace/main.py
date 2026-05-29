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
