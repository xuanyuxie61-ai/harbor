"""
main.py
量子计算：量子机器学习核方法 — 统一入口

本项目基于 15 个种子科研代码项目，在量子机器学习核方法领域进行
博士级科学计算合成。项目涵盖量子特征映射、量子核估计、核矩阵分析、
变分优化、并行电路模拟、蒙特卡洛采样、几何分析与数值稳定性等
前沿研究方向。

核心科学问题:
基于反应扩散动力学与谱方法的参数化量子特征映射，在量子机器学习核方法中
的表达能力与数值稳定性研究。

数学模型总览:
1. 量子特征映射: |phi(x)> = (bigotimes_j R_Y(x_j) R_Z(x_j^2)) |0^n>
2. 量子核函数: k(x, x') = |<0^n|U^dagger(x)U(x')|0^n>|^2
3. Gray-Scott 反应扩散方程驱动的参数生成
4. Chebyshev 谱微分用于量子动力学离散
5. Stroud 多维求积用于期望值积分
6. Broyden 拟牛顿法用于变分优化
7. Hager/LINPACK 条件数估计算法用于核矩阵稳定性分析
8. Feynman-Kac 路径积分用于量子核的蒙特卡洛估计
"""

import numpy as np
import sys
import os

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    QuantumGateLibrary, print_section, print_subsection,
    normalize_vector, clip_probability
)
from randomness_engine import (
    QuantumRandomnessEngine, quantum_random_hermitian,
    quantum_random_unitary
)
from reaction_diffusion_kernel import (
    ReactionDiffusionFeatureMap,
    gray_scott_simulation,
    advection_ftcs_step,
    pattern_to_quantum_parameters
)
from stroud_integrator import (
    StroudIntegrator,
    en_r2_monomial_integral,
    cn_leg_monomial_integral,
    gaussian_quadrature_kernel_expectation
)
from quantum_monte_carlo import (
    feynman_kac_2d_estimator,
    quantum_walk_kernel_estimate,
    markov_chain_hit_time_stats,
    quantum_kernel_monte_carlo
)
from kernel_matrix_analysis import (
    QuantumKernelMatrix,
    chebyshev_grid,
    chebyshev_differentiation_matrix,
    hager_condition_number_estimate,
    sample_condition_estimate,
    vandermonde_determinant,
    quantum_kernel_with_vandermonde
)
from variational_optimizer import (
    VariationalQuantumOptimizer,
    unstable_ode_system,
    unstable_exact_solution,
    broyden_quasi_newton
)
from parallel_circuit_simulator import (
    ParallelTaskScheduler,
    single_qubit_gate_tensor,
    two_qubit_gate_tensor,
    apply_quantum_circuit,
    sparse_pagerank_matrix,
    power_iteration_pagerank,
    quantum_circuit_pagerank_spectrum
)
from geometric_feature_map import (
    minimal_surface_catenoid,
    minimal_surface_scherk,
    minimal_surface_residual,
    point_in_polygon,
    geometric_quantum_kernel,
    quantum_feature_space_volume
)
from stability_analysis import (
    von_neumann_amplification_ftcs,
    cfl_condition_hyperbolic,
    diffusion_stability_limit,
    matrix_spectral_radius,
    analyze_kernel_matrix_stability,
    trotter_error_bound,
    quantum_kernel_robustness_score
)


def demo_randomness_engine():
    """演示量子随机数引擎 (基于 1373_uniform 与 1374_unstable_ode)。"""
    print_subsection("1. 量子随机数引擎 (基于 Park-Miller LCRG)")

    engine = QuantumRandomnessEngine(seed=42)

    # 生成均匀随机序列
    seq = engine.generate_sequence(10)
    print(f"  均匀随机序列 (前10个): {seq[:5]}")

    # 复数圆盘采样
    z = engine.uniform_disk()
    print(f"  单位圆盘复数采样: {z:.6f}, |z| = {abs(z):.6f}")

    # 高维球面采样
    sphere_vec = engine.uniform_sphere_nd(5)
    print(f"  5维单位球面向量范数: {np.linalg.norm(sphere_vec):.10f}")

    # 跳跃测试
    engine2 = QuantumRandomnessEngine(seed=42)
    engine2.jump_ahead(100)
    val_after_jump = engine2.uniform_01()

    engine3 = QuantumRandomnessEngine(seed=42)
    for _ in range(100):
        engine3._advance()
    val_after_iter = engine3.uniform_01()
    print(f"  跳跃一致性检查: {val_after_jump:.10f} == {val_after_iter:.10f} ? {np.isclose(val_after_jump, val_after_iter)}")

    # 随机厄米矩阵
    H = quantum_random_hermitian(4, engine)
    is_hermitian = np.allclose(H, H.conj().T)
    print(f"  4x4 随机厄米矩阵检查: {is_hermitian}")


def demo_reaction_diffusion():
    """演示反应扩散驱动的量子特征映射 (基于 487_gray_scott_pde 与 353_fd1d_advection_ftcs)。"""
    print_subsection("2. 反应扩散驱动的量子特征映射")

    # Gray-Scott 模拟
    U, V = gray_scott_simulation(
        nx=32, ny=32, n_steps=2000,
        D_u=8.0e-5, D_v=4.0e-5, gamma=0.024, kappa=0.06
    )
    print(f"  Gray-Scott U 场范围: [{U.min():.4f}, {U.max():.4f}]")
    print(f"  Gray-Scott V 场范围: [{V.min():.4f}, {V.max():.4f}]")

    # 将模式映射为量子参数
    params = pattern_to_quantum_parameters(U, n_qubits=4, n_layers=3)
    print(f"  量子参数形状: {params.shape}")
    print(f"  参数范围: [{params.min():.4f}, {params.max():.4f}]")

    # 特征映射器
    feature_map = ReactionDiffusionFeatureMap(n_qubits=4, n_layers=3)
    data_point = np.array([0.1, -0.2, 0.3, -0.1])
    mapped_params = feature_map.get_parameters(data_point)
    print(f"  特征映射后参数形状: {mapped_params.shape}")

    # 对流方程 FTCS (用于稳定性分析演示)
    u0 = np.zeros(64)
    x = np.linspace(0, 1, 64)
    mask = (x >= 0.4) & (x <= 0.6)
    u0[mask] = ((10 * x[mask] - 4) ** 2) * ((6 - 10 * x[mask]) ** 2)
    u1 = advection_ftcs_step(u0, c=1.0, dt=0.001, dx=1.0 / 63)
    print(f"  FTCS 单步后 u 范数变化: {np.linalg.norm(u0):.4f} -> {np.linalg.norm(u1):.4f}")


def demo_stroud_integration():
    """演示 Stroud 多维求积 (基于 1174_stroud_rule)。"""
    print_subsection("3. Stroud 多维求积规则")

    # 验证 EN_R2 下单项式积分
    val = en_r2_monomial_integral((2, 2))
    expected = np.sqrt(np.pi) / 2 * np.sqrt(np.pi) / 2
    print(f"  EN_R2 (2,2) 单项式积分: {val:.6f}, 期望: {expected:.6f}, 误差: {abs(val - expected):.2e}")

    # 3 维高斯权重积分
    integrator = StroudIntegrator(n_dim=3, rule_type="en_r2_03")
    print(f"  EN_R2 3维规则节点数: {len(integrator.nodes)}")

    # 积分常数函数 1
    f_const = lambda x: 1.0
    result = integrator.integrate(f_const)
    expected_vol = np.pi ** 1.5
    print(f"  常数积分结果: {result:.6f}, 期望: {expected_vol:.6f}")

    # 积分二次函数
    f_quad = lambda x: np.sum(x ** 2)
    result_quad = integrator.integrate(f_quad)
    print(f"  x^2 积分结果: {result_quad:.6f}")


def demo_quantum_monte_carlo():
    """演示量子蒙特卡洛方法 (基于 423_feynman_kac_2d 与 1092_snakes_and_ladders_simulation)。"""
    print_subsection("4. 量子蒙特卡洛与路径积分")

    # Feynman-Kac 估计
    estimate, exact = feynman_kac_2d_estimator(
        x0=0.5, y0=0.3, a=2.0, b=1.0,
        h=0.001, n_trajectories=500, max_steps=50000
    )
    print(f"  Feynman-Kac 估计: {estimate:.6f}, 精确值: {exact:.6f}, 相对误差: {abs(estimate - exact) / exact:.4f}")

    # 量子行走核估计
    state_a = np.array([1.0, 0.0, 0.0, 0.0])
    state_b = np.array([0.5, 0.5, 0.5, 0.5]) / np.linalg.norm([0.5, 0.5, 0.5, 0.5])
    kernel_val = quantum_walk_kernel_estimate(state_a, state_b, n_samples=200)
    exact_overlap = abs(np.vdot(state_a / np.linalg.norm(state_a), state_b)) ** 2
    print(f"  量子行走核估计: {kernel_val:.6f}, 精确重叠: {exact_overlap:.6f}")

    # 吸收态马尔可夫链命中时间
    # 构造简单的转移矩阵
    n = 5
    P = np.eye(n) * 0.5 + np.roll(np.eye(n), 1, axis=1) * 0.3 + np.roll(np.eye(n), -1, axis=1) * 0.2
    P = P / P.sum(axis=1, keepdims=True)
    stats = markov_chain_hit_time_stats(P, start_state=0, absorbing_state=4, n_games=500)
    print(f"  马尔可夫链命中时间: min={stats['min']:.0f}, mean={stats['mean']:.2f}, max={stats['max']:.0f}, std={stats['std']:.2f}")


def demo_kernel_matrix():
    """演示核矩阵分析 (基于 1004_r8vm, 161_chebyshev_matrix, 207_condition)。"""
    print_subsection("5. 量子核矩阵分析")

    # Chebyshev 网格与微分矩阵
    n = 8
    x_cheb = chebyshev_grid(n)
    D = chebyshev_differentiation_matrix(n)
    # 验证: D @ 常数向量 = 0
    const_vec = np.ones(n + 1)
    deriv_const = D @ const_vec
    print(f"  Chebyshev D @ 1 最大误差: {np.max(np.abs(deriv_const)):.2e}")

    # Vandermonde 行列式
    v_points = np.array([1.0, 2.0, 3.0, 4.0])
    v_det = vandermonde_determinant(v_points)
    V = np.vander(v_points, increasing=True)
    np_det = np.linalg.det(V)
    print(f"  Vandermonde 行列式: 公式={v_det:.2f}, numpy={np_det:.2f}")

    # 构造量子核矩阵
    np.random.seed(123)
    n_samples = 20
    data = np.random.randn(n_samples, 4)

    def quantum_kernel(x, xp):
        # 简单的量子核模拟
        return np.exp(-0.5 * np.sum((x - xp) ** 2))

    qkm = QuantumKernelMatrix(quantum_kernel, data)
    K = qkm.compute_kernel_matrix()
    cond = qkm.condition_number()
    hager_cond = qkm.hager_cond_estimate()
    print(f"  核矩阵条件数 (谱): {cond:.4e}")
    print(f"  核矩阵条件数 (Hager L1): {hager_cond:.4e}")

    # 核目标对齐度
    labels = np.sign(data[:, 0] + data[:, 1])
    kta = qkm.kernel_target_alignment(labels)
    print(f"  核目标对齐度 (KTA): {kta:.4f}")

    # 求解核岭回归
    alpha = qkm.solve_kernel_system(labels, reg=1e-4)
    print(f"  核岭回归解范数: {np.linalg.norm(alpha):.4f}")


def demo_variational_optimizer():
    """演示变分优化 (基于 120_broyden 与 1374_unstable_ode)。"""
    print_subsection("6. 变分量子优化 (Broyden 拟牛顿法)")

    # 不稳定 ODE 精确解验证
    t = 1.0
    mu = 0.1
    y_exact = unstable_exact_solution(t, mu)
    print(f"  不稳定 ODE 精确解 t=1.0: y1={y_exact[0]:.4f}, y2={y_exact[1]:.4f}")

    # Broyden 求解简单非线性系统
    def simple_nonlinear(x):
        """简单的二维非线性系统 (有解析解 x=[1, -1])。"""
        f1 = x[0]**2 + x[1]**2 - 2.0
        f2 = x[0] - x[1] - 2.0
        return np.array([f1, f2])

    x0 = np.array([1.5, 0.0])
    x_opt, ierr = broyden_quasi_newton(simple_nonlinear, x0, atol=1e-10, rtol=1e-8, maxit=200, maxdim=15)
    residual = np.linalg.norm(simple_nonlinear(x_opt))
    print(f"  Broyden 求解结果: x=[{x_opt[0]:.6f}, {x_opt[1]:.6f}], 残差={residual:.2e}, 标志={ierr}")

    # VQE 能量最小化
    H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    def ansatz(theta):
        # 单参数 ansatz: RY(theta) |0>
        return np.array([np.cos(theta[0] / 2.0), np.sin(theta[0] / 2.0)])

    optimizer = VariationalQuantumOptimizer()
    theta_opt, E_opt = optimizer.vqe_minimize(H, ansatz, theta0=np.array([0.5]))
    print(f"  VQE 最优参数: theta={theta_opt[0]:.6f}, 能量={E_opt:.6f}")
    print(f"  VQE 期望基态能量: -1.0")


def demo_parallel_circuit():
    """演示并行量子电路模拟 (基于 237_cuda_loop 与 845_pagerank2)。"""
    print_subsection("7. 并行量子电路模拟与 PageRank 谱分析")

    # 并行调度器
    scheduler = ParallelTaskScheduler(blocks=(2, 1, 1), threads=(4, 1, 1))
    assignments = scheduler.get_task_assignments(20)
    total_assigned = sum(len(a) for a in assignments)
    print(f"  20 个任务分配给 {scheduler.chunk} 个线程, 总分配数: {total_assigned}")

    # 单量子门张量
    Rx = QuantumGateLibrary.Rx(np.pi / 4.0)
    U_full = single_qubit_gate_tensor(Rx, target=1, n_qubits=3)
    print(f"  RX(pi/4) 在 qubit 1 上的张量积维度: {U_full.shape}")

    # 应用量子电路
    state = np.zeros(8, dtype=np.complex128)
    state[0] = 1.0
    H = QuantumGateLibrary.H()
    gates = [
        single_qubit_gate_tensor(H, 0, 3),
        single_qubit_gate_tensor(H, 1, 3),
        single_qubit_gate_tensor(H, 2, 3),
    ]
    final_state = apply_quantum_circuit(state, gates)
    probs = np.abs(final_state) ** 2
    print(f"  3-Hadamard 电路输出概率和: {np.sum(probs):.6f}")

    # PageRank 谱分析
    adj_list = [
        [1, 2], [2, 3], [0, 3], [1, 4], [0, 2]
    ]
    ranks = quantum_circuit_pagerank_spectrum(n_qubits=3, adjacency_list=adj_list, damping=0.85)
    print(f"  电路门 PageRank 排名: {ranks}")


def demo_geometric_analysis():
    """演示几何特征映射 (基于 768_minimal_surface_exact 与 1265_toms112)。"""
    print_subsection("8. 极小曲面几何与量子态空间分析")

    # 悬链面
    X = np.linspace(1.1, 3.0, 5)
    Y = np.linspace(0.0, 2.0, 5)
    X_grid, Y_grid = np.meshgrid(X, Y)
    U, Ux, Uy, Uxx, Uxy, Uyy = minimal_surface_catenoid(X_grid, Y_grid, a=1.0)
    R = minimal_surface_residual(Uxx, Uxy, Uyy, Ux, Uy)
    print(f"  悬链面残差最大绝对值: {np.max(np.abs(R)):.2e}")

    # Scherk 曲面
    X2 = np.linspace(-0.5, 0.5, 5)
    Y2 = np.linspace(-0.5, 0.5, 5)
    X2_grid, Y2_grid = np.meshgrid(X2, Y2)
    U2, Ux2, Uy2, Uxx2, Uxy2, Uyy2 = minimal_surface_scherk(X2_grid, Y2_grid, a=1.0)
    R2 = minimal_surface_residual(Uxx2, Uxy2, Uyy2, Ux2, Uy2)
    print(f"  Scherk 曲面残差最大绝对值: {np.max(np.abs(R2)):.2e}")

    # 点在多边形内判定
    poly_x = np.array([0.0, 2.0, 2.0, 0.0])
    poly_y = np.array([0.0, 0.0, 2.0, 2.0])
    inside1 = point_in_polygon(1.0, 1.0, poly_x, poly_y)
    inside2 = point_in_polygon(3.0, 3.0, poly_x, poly_y)
    print(f"  点 (1,1) 在正方形内: {inside1}")
    print(f"  点 (3,3) 在正方形内: {inside2}")

    # 几何量子核
    x1 = np.array([0.5, 0.5])
    x2 = np.array([0.6, 0.4])
    gk = geometric_quantum_kernel(x1, x2, surface_type="catenoid")
    print(f"  几何量子核 (catenoid): {gk:.6f}")

    # 量子特征空间体积
    vol = quantum_feature_space_volume(n_qubits=3, n_samples=500)
    print(f"  3-qubit 特征空间有效体积估计: {vol:.4f}")


def demo_stability_analysis():
    """演示数值稳定性分析 (基于 1374_unstable_ode 与 353_fd1d_advection_ftcs)。"""
    print_subsection("9. 数值稳定性与误差分析")

    # von Neumann 放大因子
    k_vals = np.linspace(0, np.pi, 50)
    G_abs = von_neumann_amplification_ftcs(c=1.0, dt=0.001, dx=0.01, k_values=k_vals)
    print(f"  FTCS 放大因子范围: [{np.min(G_abs):.6f}, {np.max(G_abs):.6f}]")
    print(f"  FTCS 无条件不稳定确认: max|G| > 1 ? {np.max(G_abs) > 1.0}")

    # CFL 条件
    dt_max = cfl_condition_hyperbolic(wave_speed=2.0, dx=0.01)
    print(f"  CFL 最大 dt (c=2, dx=0.01): {dt_max:.6f}")

    # 扩散稳定性
    dt_diff = diffusion_stability_limit(D=1.0e-4, dx=0.01, dimension=2)
    print(f"  2D 扩散稳定性极限: {dt_diff:.6f}")

    # 核矩阵稳定性
    np.random.seed(456)
    n = 15
    K = np.random.randn(n, n)
    K = K @ K.T + 0.1 * np.eye(n)
    K = K / np.max(K)
    stab_info = analyze_kernel_matrix_stability(K)
    print(f"  核矩阵条件数: {stab_info['condition_number']:.4e}")
    print(f"  最小特征值: {stab_info['smallest_eigenvalue']:.4e}")
    print(f"  推荐正则化参数: {stab_info['recommended_reg']:.4e}")
    print(f"  是否良态: {stab_info['is_well_conditioned']}")

    # Trotter 误差
    H_test = np.random.randn(4, 4)
    H_test = (H_test + H_test.T) / 2.0
    trotter_err = trotter_error_bound(H_test, dt=0.01, order=1)
    print(f"  一阶 Trotter 误差上界 (dt=0.01): {trotter_err:.4e}")

    # 量子核鲁棒性
    robust_score = quantum_kernel_robustness_score(K, noise_level=0.01)
    print(f"  量子核鲁棒性分数: {robust_score:.4f}")


def run_full_pipeline():
    """运行完整的量子机器学习核方法计算流程。"""
    print_section("量子计算: 量子机器学习核方法 — 博士级合成项目")
    print("=" * 70)
    print("  科学领域: 量子计算 — 量子机器学习核方法")
    print("  核心问题: 基于反应扩散与谱方法的量子特征映射及其数值稳定性")
    print("=" * 70)

    demo_randomness_engine()
    demo_reaction_diffusion()
    demo_stroud_integration()
    demo_quantum_monte_carlo()
    demo_kernel_matrix()
    demo_variational_optimizer()
    demo_parallel_circuit()
    demo_geometric_analysis()
    demo_stability_analysis()

    print_section("计算流程完成")
    print("  所有模块运行完毕，数值结果已输出至控制台。")
    print("  详细数学模型与算法说明请参阅 README_博士级合成说明.md")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()

# ================================================================
# 测试用例（64个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# 补充导入：以下函数在 main.py 中未显式导入，测试用例需要直接引用
from utils import extended_gcd, mod_inverse, power_mod, is_power_of_two, log2_int, safe_divide
from randomness_engine import box_muller_transform
from reaction_diffusion_kernel import laplacian9_torus
from quantum_monte_carlo import potential_elliptic
from kernel_matrix_analysis import plu_decomposition

# ---- TC01: extended_gcd 基本正确性与 Bezout 恒等式 ----
g, x, y = extended_gcd(48, 18)
assert g == 6, '[TC01] extended_gcd(48,18) gcd 应为 6 FAILED'
assert 48 * x + 18 * y == g, '[TC01] Bezout 恒等式 48*x+18*y=g 不成立 FAILED'

# ---- TC02: extended_gcd 互质情况 ----
g2, x2, y2 = extended_gcd(17, 13)
assert g2 == 1, '[TC02] extended_gcd(17,13) gcd 应为 1 FAILED'

# ---- TC03: mod_inverse 正确性 ----
inv = mod_inverse(3, 11)
assert (3 * inv) % 11 == 1, '[TC03] 3 在模 11 下的逆元错误 FAILED'

# ---- TC04: mod_inverse 不存在时应抛异常 ----
try:
    mod_inverse(4, 8)
    assert False, '[TC04] gcd(4,8)!=1 应抛 ValueError FAILED'
except ValueError:
    pass

# ---- TC05: power_mod 正确性 (小值验算) ----
result = power_mod(2, 10, 1000)
assert result == 24, '[TC05] 2^10 mod 1000 应为 24 FAILED'

# ---- TC06: power_mod 边界: n=0 应返回 1 ----
result0 = power_mod(7, 0, 13)
assert result0 == 1, '[TC06] 7^0 mod 13 应为 1 FAILED'

# ---- TC07: is_power_of_two 正确性 ----
assert is_power_of_two(16) == True, '[TC07] 16 是 2 的幂 FAILED'
assert is_power_of_two(15) == False, '[TC07] 15 不是 2 的幂 FAILED'
assert is_power_of_two(1) == True, '[TC07] 1 是 2 的幂 (2^0) FAILED'

# ---- TC08: log2_int 正确性 ----
assert log2_int(8) == 3, '[TC08] log2(8) 应为 3 FAILED'
assert log2_int(1) == 0, '[TC08] log2(1) 应为 0 FAILED'

# ---- TC09: clip_probability 范围约束 ----
assert clip_probability(0.5) == 0.5, '[TC09] 0.5 裁剪后应为 0.5 FAILED'
assert clip_probability(1.5) == 1.0, '[TC09] 1.5 裁剪后应为 1.0 FAILED'
assert clip_probability(-0.5) == 0.0, '[TC09] -0.5 裁剪后应为 0.0 FAILED'

# ---- TC10: normalize_vector 范数验证 ----
import numpy as np
v = np.array([3.0, 4.0], dtype=np.float64)
nv = normalize_vector(v)
assert abs(np.linalg.norm(nv) - 1.0) < 1e-12, '[TC10] 归一化后范数应接近 1 FAILED'

# ---- TC11: normalize_vector 零向量处理 ----
v_zero = np.zeros(5)
nv_zero = normalize_vector(v_zero)
assert np.allclose(nv_zero, np.zeros(5)), '[TC11] 零向量归一化应返回零向量 FAILED'

# ---- TC12: safe_divide 正常与边界情况 ----
assert safe_divide(10.0, 2.0) == 5.0, '[TC12] 10/2 应为 5 FAILED'
assert safe_divide(10.0, 0.0, default=-1.0) == -1.0, '[TC12] 除零应返回默认值 -1 FAILED'

# ---- TC13: QuantumRandomnessEngine 可复现性 ----
import numpy as np
engine1 = QuantumRandomnessEngine(seed=42)
engine2 = QuantumRandomnessEngine(seed=42)
seq1 = engine1.generate_sequence(5)
seq2 = engine2.generate_sequence(5)
assert np.allclose(seq1, seq2), '[TC13] 相同种子应产生相同序列 FAILED'

# ---- TC14: QuantumRandomnessEngine uniform_01 返回 [0,1] 值 ----
engine = QuantumRandomnessEngine(seed=123)
vals = [engine.uniform_01() for _ in range(100)]
assert all(0.0 <= v <= 1.0 for v in vals), '[TC14] uniform_01 应返回 [0,1] 内的值 FAILED'

# ---- TC15: QuantumRandomnessEngine jump_ahead 一致性 ----
import numpy as np
e1 = QuantumRandomnessEngine(seed=42)
e1.jump_ahead(100)
v1 = e1.uniform_01()
e2 = QuantumRandomnessEngine(seed=42)
for _ in range(100):
    e2._advance()
v2 = e2.uniform_01()
assert np.isclose(v1, v2), '[TC15] jump_ahead(100) 应与 100 次 _advance 结果一致 FAILED'

# ---- TC16: box_muller_transform 输出有限且非零 (典型输入) ----
z1, z2 = box_muller_transform(0.5, 0.3)
assert np.isfinite(z1) and np.isfinite(z2), '[TC16] Box-Muller 输出应为有限值 FAILED'

# ---- TC17: box_muller_transform 边界输入安全性 ----
z1b, z2b = box_muller_transform(0.0, 0.5)
assert np.isfinite(z1b) and np.isfinite(z2b), '[TC17] Box-Muller u1=0 不应产生 NaN FAILED'

# ---- TC18: laplacian9_torus 常数场应为零 ----
field = np.ones((8, 8), dtype=np.float64)
lap = laplacian9_torus(field, dx=0.1)
assert np.max(np.abs(lap)) < 1e-12, '[TC18] 常数场的 Laplacian 应为零 FAILED'

# ---- TC19: pattern_to_quantum_parameters 输出形状与范围 ----
import numpy as np
pattern = np.random.rand(10, 10)
params = pattern_to_quantum_parameters(pattern, n_qubits=4, n_layers=3)
assert params.shape == (3, 4), '[TC19] 参数形状应为 (3,4) FAILED'
assert np.all(params >= -np.pi / 2 - 1e-10) and np.all(params <= np.pi / 2 + 1e-10), '[TC19] 参数应在 [-pi/2, pi/2] 范围内 FAILED'

# ---- TC20: en_r2_monomial_integral 奇次幂为零 ----
result = en_r2_monomial_integral((1, 2))
assert result == 0.0, '[TC20] 含奇次指数的高斯积分应为 0 FAILED'

# ---- TC21: en_r2_monomial_integral 全偶次幂非零 ----
result = en_r2_monomial_integral((2, 0))
assert result > 0.0, '[TC21] 全偶次指数的高斯积分应为正值 FAILED'

# ---- TC22: cn_leg_monomial_integral 奇次积分为零 ----
result = cn_leg_monomial_integral((1, 0))
assert abs(result) < 1e-12, '[TC22] (-1,1) 上 x^1 积分应为 0 FAILED'

# ---- TC23: cn_leg_monomial_integral 1D x^2 积分 ----
result = cn_leg_monomial_integral((2,))
assert abs(result - 2.0 / 3.0) < 1e-12, '[TC23] (-1,1) 上 x^2 积分应为 2/3 FAILED'

# ---- TC24: StroudIntegrator 3 维高斯常数积分 ----
import numpy as np
integrator = StroudIntegrator(n_dim=3, rule_type="en_r2_03")
f_const = lambda x: 1.0
result = integrator.integrate(f_const)
expected = np.pi ** 1.5
assert abs(result - expected) / expected < 0.3, '[TC24] 3 维高斯常数积分应接近 pi^{3/2} FAILED'

# ---- TC25: StroudIntegrator CN_LEG 规则节点/权重长度一致 ----
integrator_cn = StroudIntegrator(n_dim=4, rule_type="cn_leg_03")
assert len(integrator_cn.nodes) == len(integrator_cn.weights), '[TC25] 节点与权重长度应一致 FAILED'
assert len(integrator_cn.nodes) == 8, '[TC25] 4 维 CN_LEG 应有 8 个节点 FAILED'

# ---- TC26: chebyshev_grid 首尾节点与单调性 ----
grid = chebyshev_grid(8)
assert abs(grid[0] - 1.0) < 1e-12, '[TC26] Chebyshev 网格首节点应为 1 FAILED'
assert abs(grid[-1] - (-1.0)) < 1e-12, '[TC26] Chebyshev 网格末节点应为 -1 FAILED'
assert np.all(np.diff(grid) < 0), '[TC26] Chebyshev 网格应单调递减 FAILED'

# ---- TC27: chebyshev_differentiation_matrix D@1=0 ----
import numpy as np
n = 8
D = chebyshev_differentiation_matrix(n)
const_vec = np.ones(n + 1)
deriv = D @ const_vec
assert np.max(np.abs(deriv)) < 1e-10, '[TC27] Chebyshev D 作用于常数向量应为零 FAILED'

# ---- TC28: vandermonde_determinant 与 numpy 一致 ----
import numpy as np
x = np.array([1.0, 2.0, 3.0])
det_formula = vandermonde_determinant(x)
det_numpy = np.linalg.det(np.vander(x, increasing=True))
assert abs(det_formula - det_numpy) < 1e-10, '[TC28] Vandermonde 行列式公式应与 numpy 一致 FAILED'

# ---- TC29: hager_condition_number_estimate 正定矩阵正值 ----
import numpy as np
A = np.array([[4.0, 1.0], [1.0, 3.0]])
cond_est = hager_condition_number_estimate(A)
assert cond_est > 0, '[TC29] Hager 条件数估计应为正 FAILED'

# ---- TC30: QuantumKernelMatrix KTA 对称性/等价性 ----
import numpy as np
np.random.seed(42)
data = np.random.randn(10, 3)
def rbf_kernel(x, xp):
    return np.exp(-np.sum((x - xp) ** 2))
qkm = QuantumKernelMatrix(rbf_kernel, data)
K = qkm.compute_kernel_matrix()
assert np.allclose(K, K.T), '[TC30] 核矩阵应为对称矩阵 FAILED'
assert K.shape == (10, 10), '[TC30] 核矩阵形状应为 (10,10) FAILED'

# ---- TC31: QuantumKernelMatrix condition_number 有限正值 ----
cond = qkm.condition_number()
assert np.isfinite(cond) and cond > 0, '[TC31] 条件数应为有限正值 FAILED'

# ---- TC32: unstable_exact_solution t=0 初始值 ----
y0 = unstable_exact_solution(0.0, mu=0.1)
assert abs(y0[0] - 1.0) < 1e-12, '[TC32] t=0 时 y1 应为 1.0 FAILED'
assert abs(y0[1] - 0.0) < 1e-12, '[TC32] t=0 时 y2 应为 0.0 FAILED'

# ---- TC33: unstable_ode_system 维度检查 ----
import numpy as np
dy = unstable_ode_system(0.5, np.array([1.0, 0.0]), mu=0.5)
assert len(dy) == 2, '[TC33] ODE 导数向量应为二维 FAILED'
assert np.all(np.isfinite(dy)), '[TC33] ODE 导数应为有限值 FAILED'

# ---- TC34: broyden_quasi_newton 解可分离线性系统 ----
def separable_system(x):
    return np.array([x[0] - 1.0, x[1] + 1.0])
x0 = np.array([0.0, 0.0], dtype=np.float64)
x_opt, ierr = broyden_quasi_newton(separable_system, x0, maxit=100)
residual = np.linalg.norm(separable_system(x_opt))
assert residual < 1e-8, '[TC34] Broyden 应能求解可分离线性系统 FAILED'
assert abs(x_opt[0] - 1.0) < 1e-6, '[TC34] Broyden 解 x[0] 应为 1.0 FAILED'
assert abs(x_opt[1] - (-1.0)) < 1e-6, '[TC34] Broyden 解 x[1] 应为 -1.0 FAILED'

# ---- TC35: VariationalQuantumOptimizer VQE 运行不崩溃且返回有限值 ----
import numpy as np
H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
def ansatz(theta):
    return np.array([np.cos(theta[0] / 2.0), np.sin(theta[0] / 2.0)])
optimizer = VariationalQuantumOptimizer()
theta_opt, E_opt = optimizer.vqe_minimize(H, ansatz, np.array([0.5]))
assert np.isfinite(E_opt), '[TC35] VQE 优化能量应为有限值 FAILED'
assert np.isfinite(theta_opt[0]), '[TC35] VQE 最优参数应为有限值 FAILED'

# ---- TC36: single_qubit_gate_tensor 形状与幺正性 ----
import numpy as np
Hgate = QuantumGateLibrary.H()
U = single_qubit_gate_tensor(Hgate, target=0, n_qubits=2)
assert U.shape == (4, 4), '[TC36] 2-qubit 门张量积应为 4x4 FAILED'
assert np.allclose(U @ U.conj().T, np.eye(4)), '[TC36] 门张量积应保持幺正性 FAILED'

# ---- TC37: ParallelTaskScheduler 任务分配完整性 ----
scheduler = ParallelTaskScheduler(blocks=(2, 1, 1), threads=(4, 1, 1))
assignments = scheduler.get_task_assignments(20)
total = sum(len(a) for a in assignments)
assert total == 20, '[TC37] 总分配任务数应为 20 FAILED'

# ---- TC38: sparse_pagerank_matrix 列和为 1 ----
import numpy as np
adj = [[1], [0, 2], [0]]
G = sparse_pagerank_matrix(adj, n_nodes=3, damping=0.85)
col_sums = G.sum(axis=0)
assert np.allclose(col_sums, 1.0), '[TC38] PageRank 矩阵各列和应为 1 FAILED'

# ---- TC39: power_iteration_pagerank 收敛性 ----
ranks = power_iteration_pagerank(G)
assert abs(np.sum(ranks) - 1.0) < 1e-10, '[TC39] PageRank 向量和应为 1 FAILED'
assert np.all(ranks >= 0), '[TC39] PageRank 值应非负 FAILED'

# ---- TC40: minimal_surface_catenoid 返回有限值且导数符号合理 ----
import numpy as np
X = np.array([[2.0]])
Y = np.array([[0.5]])
U, Ux, Uy, Uxx, Uxy, Uyy = minimal_surface_catenoid(X, Y, a=1.0)
assert np.all(np.isfinite(U)), '[TC40] 悬链面 U 应为有限值 FAILED'
assert np.all(np.isfinite(Ux)), '[TC40] 悬链面 Ux 应为有限值 FAILED'
assert np.all(np.isfinite(Uy)), '[TC40] 悬链面 Uy 应为有限值 FAILED'
assert U[0,0] > 0, '[TC40] 悬链面在 (2.0,0.5) 处 U 应为正 FAILED'

# ---- TC41: minimal_surface_scherk 极小曲面残差 ----
import numpy as np
X2 = np.array([[0.3]])
Y2 = np.array([[0.4]])
U2, Ux2, Uy2, Uxx2, Uxy2, Uyy2 = minimal_surface_scherk(X2, Y2, a=1.0)
R2 = minimal_surface_residual(Uxx2, Uxy2, Uyy2, Ux2, Uy2)
assert np.max(np.abs(R2)) < 1e-6, '[TC41] Scherk 曲面极小曲面方程残差应接近零 FAILED'

# ---- TC42: point_in_polygon 正方形内外判定 ----
import numpy as np
poly_x = np.array([0.0, 2.0, 2.0, 0.0])
poly_y = np.array([0.0, 0.0, 2.0, 2.0])
assert point_in_polygon(1.0, 1.0, poly_x, poly_y) == True, '[TC42] (1,1) 应在正方形内 FAILED'
assert point_in_polygon(3.0, 3.0, poly_x, poly_y) == False, '[TC42] (3,3) 应在正方形外 FAILED'
assert point_in_polygon(0.5, 0.0, poly_x, poly_y) == True, '[TC42] (0.5,0) 在正方形边界上应为内 FAILED'

# ---- TC43: geometric_quantum_kernel 输出在 [0,1] 内 ----
import numpy as np
x = np.array([0.5, 0.5])
xp = np.array([0.6, 0.4])
gk = geometric_quantum_kernel(x, xp, surface_type="catenoid")
assert 0.0 <= gk <= 1.0, '[TC43] 几何量子核值应在 [0,1] 内 FAILED'

# ---- TC44: geometric_quantum_kernel 自核接近 1 ----
gk_self = geometric_quantum_kernel(x, x, surface_type="catenoid")
assert abs(gk_self - 1.0) < 1e-10, '[TC44] 几何量子核 k(x,x) 应接近 1 FAILED'

# ---- TC45: von_neumann_amplification_ftcs 不稳定性确认 ----
import numpy as np
k_vals = np.linspace(0, np.pi, 20)
G_abs = von_neumann_amplification_ftcs(c=1.0, dt=0.01, dx=0.1, k_values=k_vals)
assert np.any(G_abs > 1.0), '[TC45] FTCS 应表现出无条件不稳定 (max|G| > 1) FAILED'

# ---- TC46: cfl_condition_hyperbolic 正值 ----
dt_max = cfl_condition_hyperbolic(wave_speed=2.0, dx=0.01)
assert dt_max > 0, '[TC46] CFL 最大时间步长应为正 FAILED'

# ---- TC47: diffusion_stability_limit 高维更严格 ----
dt_1d = diffusion_stability_limit(D=1.0, dx=0.1, dimension=1)
dt_2d = diffusion_stability_limit(D=1.0, dx=0.1, dimension=2)
assert dt_2d < dt_1d, '[TC47] 高维扩散稳定性限制应更严格 FAILED'

# ---- TC48: matrix_spectral_radius 单位阵 ----
I = np.eye(5)
rho = matrix_spectral_radius(I)
assert abs(rho - 1.0) < 1e-12, '[TC48] 单位阵谱半径应为 1 FAILED'

# ---- TC49: analyze_kernel_matrix_stability 正定矩阵良态 ----
import numpy as np
K = np.array([[2.0, 1.0], [1.0, 2.0]])
info = analyze_kernel_matrix_stability(K)
assert info["is_well_conditioned"] == True, '[TC49] 良好条件矩阵应判定为良态 FAILED'
assert info["condition_number"] > 1.0, '[TC49] 非标量矩阵条件数应大于 1 FAILED'

# ---- TC50: trotter_error_bound 非负 ----
H = np.eye(4)
err = trotter_error_bound(H, dt=0.1, order=1)
assert err >= 0, '[TC50] Trotter 误差上界应非负 FAILED'

# ---- TC51: trotter_error_bound 高阶更小 ----
err1 = trotter_error_bound(H, dt=0.1, order=1)
err2 = trotter_error_bound(H, dt=0.1, order=2)
assert err2 < err1, '[TC51] 二阶 Trotter 误差应小于一阶 FAILED'

# ---- TC52: quantum_kernel_robustness_score 非负有限 ----
import numpy as np
np.random.seed(42)
Krob = np.array([[2.0, 1.0], [1.0, 2.0]])
score = quantum_kernel_robustness_score(Krob, noise_level=0.01)
assert score >= 0.0, '[TC52] 鲁棒性分数应非负 FAILED'
assert np.isfinite(score), '[TC52] 鲁棒性分数应为有限值 FAILED'

# ---- TC53: potential_elliptic 正值 ----
v = potential_elliptic(a=2.0, b=1.0, x=0.5, y=0.3)
assert v > 0, '[TC53] 椭圆势函数应返回正值 FAILED'

# ---- TC54: QuantumGateLibrary 各门厄米/幺正性 ----
import numpy as np
for gate_name in ['I', 'X', 'Y', 'Z', 'H', 'S', 'T']:
    gate = getattr(QuantumGateLibrary, gate_name)()
    assert gate.shape == (2, 2), f'[TC54] {gate_name} 门形状应为 2x2 FAILED'
    # 量子门应为幺正矩阵
    prod = gate @ gate.conj().T
    assert np.allclose(prod, np.eye(2)), f'[TC54] {gate_name} 门应为幺正 FAILED'

# ---- TC55: quantum_feature_space_volume 在 [0,1] 内 ----
import numpy as np
np.random.seed(42)
vol = quantum_feature_space_volume(n_qubits=2, n_samples=200)
assert 0.0 <= vol <= 1.0, '[TC55] 特征空间有效体积应在 [0,1] 内 FAILED'

# ---- TC56: two_qubit_gate_tensor 形状正确 ----
import numpy as np
U_cnot = QuantumGateLibrary.X()
CU = two_qubit_gate_tensor(U_cnot, control=0, target=1, n_qubits=2)
assert CU.shape == (4, 4), '[TC56] 受控门张量积应为 4x4 FAILED'
# CNOT 应为幺正
assert np.allclose(CU @ CU.conj().T, np.eye(4), atol=1e-10), '[TC56] 受控门应为幺正 FAILED'

# ---- TC57: apply_quantum_circuit 概率守恒 ----
import numpy as np
state = np.zeros(8, dtype=np.complex128)
state[0] = 1.0
H = QuantumGateLibrary.H()
gates = [
    single_qubit_gate_tensor(H, 0, 3),
    single_qubit_gate_tensor(H, 1, 3),
    single_qubit_gate_tensor(H, 2, 3),
]
final_state = apply_quantum_circuit(state, gates)
probs = np.abs(final_state) ** 2
assert abs(np.sum(probs) - 1.0) < 1e-10, '[TC57] 电路应用后概率和应为 1 FAILED'

# ---- TC58: markov_chain_hit_time_stats 输出结构 ----
import numpy as np
n = 5
P = np.eye(n) * 0.5 + np.roll(np.eye(n), 1, axis=1) * 0.3 + np.roll(np.eye(n), -1, axis=1) * 0.2
P = P / P.sum(axis=1, keepdims=True)
stats = markov_chain_hit_time_stats(P, start_state=0, absorbing_state=4, n_games=100)
for key in ['min', 'mean', 'max', 'std']:
    assert key in stats, f'[TC58] stats 应包含键 {key} FAILED'
assert stats['min'] >= 0, '[TC58] 最小命中时间应非负 FAILED'
assert stats['mean'] >= stats['min'], '[TC58] 平均命中时间应 >= 最小值 FAILED'

# ---- TC59: quantum_kernel_monte_carlo 输出在 [0,1] 内 ----
import numpy as np
np.random.seed(42)
def feature_map(x):
    return np.array([np.cos(x[0]), np.sin(x[0]), np.cos(x[1]), np.sin(x[1])])
x1 = np.array([0.2, 0.3])
x2 = np.array([0.4, 0.5])
k_mc = quantum_kernel_monte_carlo(feature_map, x1, x2, n_shots=200)
assert 0.0 <= k_mc <= 1.0, '[TC59] 蒙特卡洛量子核估计应在 [0,1] 内 FAILED'

# ---- TC60: quantum_kernel_with_vandermonde 自核为 1 ----
import numpy as np
x = np.array([0.5, 0.5, 0.5, 0.5])
k_self = quantum_kernel_with_vandermonde(x, x, n_qubits=4)
assert abs(k_self - 1.0) < 1e-10, '[TC60] Vandermonde 量子自核 k(x,x) 应为 1 FAILED'

# ---- TC61: gray_scott_simulation 小规模集成测试 ----
import numpy as np
U_small, V_small = gray_scott_simulation(nx=8, ny=8, n_steps=50, D_u=0.2, D_v=0.1, gamma=0.024, kappa=0.06)
assert U_small.shape == (8, 8) and V_small.shape == (8, 8), '[TC61] Gray-Scott 输出形状应为 (8,8) FAILED'
assert np.all(U_small >= 0) and np.all(U_small <= 1), '[TC61] U 场应在 [0,1] 内 FAILED'
assert np.all(V_small >= 0) and np.all(V_small <= 1), '[TC61] V 场应在 [0,1] 内 FAILED'

# ---- TC62: plu_decomposition 重构验证 ----
import numpy as np
A = np.array([[4.0, 3.0], [6.0, 3.0]])
P, L, U = plu_decomposition(A)
PA = P @ A
LU = L @ U
assert np.allclose(PA, LU, atol=1e-10), '[TC62] PLU 分解应满足 P*A = L*U FAILED'

# ---- TC63: ReactionDiffusionFeatureMap get_parameters 形状 ----
import numpy as np
fm = ReactionDiffusionFeatureMap(n_qubits=4, n_layers=3)
fm.generate_pattern(grid_size=8, n_steps=30)
data_point = np.array([0.1, -0.2, 0.3, -0.1])
mapped = fm.get_parameters(data_point)
assert mapped.shape == (3, 4), '[TC63] 特征映射输出形状应为 (3,4) FAILED'
assert np.all(np.isfinite(mapped)), '[TC63] 特征映射输出应为有限值 FAILED'

# ---- TC64: advection_ftcs_step 输出有限 ----
import numpy as np
x = np.linspace(0, 1, 64)
u0 = np.zeros(64)
mask = (x >= 0.4) & (x <= 0.6)
u0[mask] = ((10 * x[mask] - 4) ** 2) * ((6 - 10 * x[mask]) ** 2)
u1 = advection_ftcs_step(u0, c=1.0, dt=0.001, dx=1.0/63)
assert np.all(np.isfinite(u1)), '[TC64] FTCS 单步输出应为有限值 FAILED'
assert len(u1) == 64, '[TC64] FTCS 输出长度应为 64 FAILED'

print('\n全部 64 个测试通过!\n')
