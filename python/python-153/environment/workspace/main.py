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

    # TODO: Complete the FTCS advection demonstration.
    # Set up the initial condition, call advection_ftcs_step with appropriate
    # parameters, and print the norm change after one step.
    pass


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

    # TODO: Complete the von Neumann amplification factor demonstration.
    # Compute the amplification factor for FTCS and verify unconditional instability.
    pass

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
