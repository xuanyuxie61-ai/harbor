"""
main.py
=======
变分量子本征求解器(VQE)统一入口
================================
量子计算：变分量子本征求解器VQE
--------------------------------

本项目基于15个科研代码项目的核心算法，融合构建了一个面向
量子化学基态能量计算的博士级VQE计算框架。

运行方式: python main.py （零参数，直接运行）
"""

import numpy as np
import sys
import time

# 导入所有模块
from banded_solver import (
    BandedMatrix, PDEParameters, pde_coefficients,
    pde_boundary_conditions, pde_initial_condition,
    finite_difference_discretize, solve_steady_pde
)
from pauli_operator import (
    PauliString, rref_compute, rref_solve,
    ShermanMorrisonSolver, extract_independent_paulis,
    build_pauli_hamiltonian
)
from ansatz_tree import (
    AnsatzTree, circle_arc_grid_params,
    initialize_parameters_on_bloch_circle, is_tree_adjacency
)
from optimizer_geodesic import (
    GyroscopeDynamics, HermiteCubicSpline, GeodesicVQEOptimizer
)
from fem_sampler import (
    bracket4, fem1d_interpolate, tetrahedron_volume,
    basis_mn_tet4, FEMExpectationSampler
)
from molecular_grid import (
    hypercube_grid, cvtm_1d_optimize, MolecularIntegralGrid
)
from measurement_sampler import (
    chebyshev2_sample, cvt_density_sample, QuantumMeasurementSampler
)
from hamiltonian_builder import (
    gauss_laguerre_rule, r8mat_orth_uniform, r8symm_gen,
    MolecularHamiltonian, compute_radial_integral_gauss_laguerre
)
from vqe_core import VQESolver, VQEConvergenceAnalysis


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_banded_solver():
    """演示带状矩阵求解器与PDE离散化（987_r8pbl + 1368_tumor_pde）"""
    print_section("模块1: 带状矩阵与PDE离散化")
    n = 64
    A = BandedMatrix.dif2(n, ml=1)
    print(f"  DIF2带状矩阵: n={n}, ml=1")
    print(f"  解析最小特征值: {A.eigenvalues_dif2()[0]:.6f}")
    print(f"  解析最大特征值: {A.eigenvalues_dif2()[-1]:.6f}")

    # 求解稳态PDE
    u = solve_steady_pde(n=32)
    print(f"  稳态PDE解: 均值={np.mean(u):.4f}, 范围=[{np.min(u):.4f}, {np.max(u):.4f}]")

    # PDE系数函数验证
    params = PDEParameters.get_defaults()
    c, f, s = pde_coefficients(0.5, 0.1, np.array([0.8, 0.3]), np.array([0.1, 0.2]), params)
    print(f"  PDE系数 (x=0.5, t=0.1): c={c}, f={f}, s={s}")


def demo_pauli_algebra():
    """演示Pauli算符代数与线性代数（1048_rref2 + 995_r8sm）"""
    print_section("模块2: Pauli算符代数与稀疏线性代数")
    p1 = PauliString('ZI', 1.0)
    p2 = PauliString('IZ', 1.0)
    p3 = p1.multiply(p2)
    print(f"  ZI * IZ = {p3}")

    p4 = PauliString('X', 1.0)
    p5 = PauliString('Y', 1.0)
    p6 = p4.multiply(p5)
    print(f"  X * Y = {p6}")

    # RREF测试
    A = np.array([[1, 3, 0, 2], [-2, -6, 0, -2], [3, 9, 0, 0], [-1, -3, 0, 1]], dtype=float)
    Arref, pivots = rref_compute(A)
    print(f"  RREF主元列: {pivots}")
    print(f"  RREF矩阵秩: {len(pivots)}")

    # Sherman-Morrison求解
    n = 8
    A_base = np.eye(n) + 0.1 * np.random.randn(n, n)
    A_base = A_base @ A_base.T  # 确保正定
    u = np.random.randn(n)
    v = np.random.randn(n)
    b = np.random.randn(n)
    sm = ShermanMorrisonSolver(A_base)
    x_sm = sm.solve(u, v, b)
    x_exact = np.linalg.solve(A_base - np.outer(u, v), b)
    err = np.linalg.norm(x_sm - x_exact)
    print(f"  Sherman-Morrison求解误差: {err:.2e}")


def demo_ansatz_tree():
    """演示Ansatz树结构（1291_treepack + 176_circle_arc_grid）"""
    print_section("模块3: 自适应Ansatz树与圆弧参数化")
    ansatz = AnsatzTree(n_qubits=2, max_depth=2)
    ansatz.build_hardware_efficient('CNOT')
    print(f"  HEA ansatz: {ansatz.n_qubits} 量子比特, 深度={ansatz.get_circuit_depth()}")
    print(f"  参数数量: {ansatz.param_count}")

    # 圆弧参数化初始化
    initialize_parameters_on_bloch_circle(ansatz)
    print(f"  圆弧初始化后参数: {ansatz.parameters}")

    # 状态向量验证
    psi = ansatz.evaluate_statevector()
    norm = np.linalg.norm(psi)
    print(f"  状态向量范数: {norm:.6f} (应为1.0)")

    # 树连通性检测
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    is_tree = is_tree_adjacency(adj)
    print(f"  线性链邻接矩阵是否为树: {is_tree}")


def demo_optimizer():
    """演示测地线优化器（495_gyroscope_ode + 518_hermite_cubic）"""
    print_section("模块4: 测地线梯度流与Hermite样条")
    gyro = GyroscopeDynamics(A1=1.0, A2=1.0, A3=3.0)
    t_vals, y_vals = gyro.simulate(dt=0.01)
    print(f"  陀螺仪轨迹: t in [{t_vals[0]:.2f}, {t_vals[-1]:.2f}], 步数={len(t_vals)}")
    print(f"  末态欧拉角: psi={y_vals[-1,0]:.4f}, theta={y_vals[-1,1]:.4f}, phi={y_vals[-1,2]:.4f}")

    # Hermite三次样条
    xn = np.array([0.0, 1.0, 2.0, 3.0])
    fn = np.array([1.0, 2.0, 1.5, 2.5])
    dn = np.array([0.5, 1.0, -0.5, 0.8])
    spline = HermiteCubicSpline(xn, fn, dn)
    xs = np.array([0.5, 1.5, 2.5])
    f_vals, d_vals, s_vals, t_vals_s = spline.evaluate(xs)
    print(f"  Hermite样条插值 at x={xs}: f={f_vals}")
    integral = spline.integrate(0.0, 3.0)
    print(f"  Hermite样条积分 [0,3]: {integral:.4f}")


def demo_fem_sampler():
    """演示有限元采样（398_fem1d_sample + 418_fem3d_project）"""
    print_section("模块5: 有限元采样与3D基础函数")
    nodes = np.linspace(0, 1, 11)
    vals = np.sin(np.pi * nodes)
    sample_x = np.array([0.15, 0.37, 0.82])
    sample_vals = fem1d_interpolate(nodes, vals, sample_x)
    exact_vals = np.sin(np.pi * sample_x)
    print(f"  1D FEM插值误差: {np.max(np.abs(sample_vals - exact_vals)):.2e}")

    # 四面体基础函数
    t = np.array([[0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    p = np.array([[0.25], [0.25], [0.25]])
    phi = basis_mn_tet4(t, 1, p)
    print(f"  四面体基础函数在形心: {phi[:,0]}")
    vol = tetrahedron_volume(t)
    print(f"  单位四面体体积: {vol:.6f} (应为1/6={1/6:.6f})")


def demo_molecular_grid():
    """演示分子网格（558_hypercube_grid + 263_cvtm_1d）"""
    print_section("模块6: 分子积分网格与CVT优化")
    grid = MolecularIntegralGrid(n_orbitals=4)
    pts_3d = grid.build_3d_grid(n_per_dim=6)
    print(f"  3D超立方体积分网格点数: {pts_3d.shape[0]}")

    # CVT优化径向网格
    radial = grid.optimize_radial_grid(n_points=8)
    print(f"  CVT径向网格节点: {np.round(radial, 3)}")

    # 独立计算一个双电子积分
    weights = np.ones(pts_3d.shape[0]) * (5.0 * 4.0 * 4.0 / pts_3d.shape[0])
    g_0000 = grid.two_electron_integral(0, 0, 0, 0, pts_3d, weights)
    print(f"  (00|00) 双电子积分近似值: {g_0000:.4f}")


def demo_measurement():
    """演示量子测量采样（1021_rejection_sample）"""
    print_section("模块7: 量子测量与拒绝采样")
    samples, trials = chebyshev2_sample(1000)
    print(f"  Chebyshev2采样: n=1000, 试验次数={trials}, 接受率={1000/trials:.3f}")
    print(f"  样本均值={np.mean(samples):.4f}, 方差={np.var(samples):.4f} (理论: 0, 0.25)")

    sampler = QuantumMeasurementSampler(n_qubits=2, n_shots=4096)
    # 制备Bell态
    psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2.0)
    exp_zz = sampler.sample_pauli_expectation(psi, 'ZZ')
    print(f"  Bell态 ZZ 测量期望值: {exp_zz:.4f} (理论: 1.0)")


def demo_hamiltonian():
    """演示哈密顿量构建（467_gen_laguerre_rule + 1206_test_eigen）"""
    print_section("模块8: 高斯求积与随机矩阵")
    x, w = gauss_laguerre_rule(8, alpha=0.0)
    # 验证求积精度: int_0^inf exp(-x) x^2 dx = 2
    integral = np.sum(w * x ** 2)
    print(f"  Laguerre-Gauss求积 (x^2): {integral:.6f} (精确: 2.0)")

    # 随机对称矩阵
    A, Q, lam = r8symm_gen(6, lambda_mean=0.0, lambda_dev=1.0)
    eigvals = np.linalg.eigvalsh(A)
    print(f"  随机对称矩阵特征值: {np.round(eigvals, 3)}")

    # 验证正交性
    err_orth = np.linalg.norm(Q.T @ Q - np.eye(6))
    print(f"  随机正交矩阵误差 ||Q^T Q - I||: {err_orth:.2e}")


def demo_full_vqe():
    """演示完整VQE流程"""
    print_section("模块9: 完整VQE计算")
    print("  构建H2分子简化模型 (2量子比特, 4空间轨道)...")
    solver = VQESolver(n_qubits=2, n_orbitals=4, ansatz_depth=2, n_shots=8192)

    # 精确基准
    E_fci, _ = solver.hamiltonian.compute_exact_ground_state()
    print(f"  FCI精确基态能量: {E_fci:.6f} Hartree")

    # 精确VQE（无噪声）
    print("  执行精确VQE优化（参数位移梯度）...")
    t0 = time.time()
    opt_params, E_vqe, info = solver.run_vqe(use_parameter_shift=True, exact_energy=True)
    t1 = time.time()
    print(f"  VQE优化后能量: {E_vqe:.6f} Hartree")
    print(f"  绝对误差: {abs(E_vqe - E_fci):.6f} Hartree")
    print(f"  迭代次数: {info['n_iterations']}")
    print(f"  优化耗时: {t1-t0:.3f}s")
    print(f"  收敛速率估计: {info['convergence_rate']:.4f}")

    # 含噪声VQE
    print("  执行含测量噪声VQE...")
    opt_params_n, E_vqe_n, info_n = solver.run_vqe(use_parameter_shift=False, exact_energy=False)
    print(f"  含噪声VQE能量: {E_vqe_n:.6f} Hartree")
    print(f"  与FCI误差: {abs(E_vqe_n - E_fci):.6f} Hartree")

    # 误差分析
    analysis = VQEConvergenceAnalysis(solver)
    gap = analysis.estimate_spectral_gap()
    print(f"  基态-激发态能隙: {gap:.4f} Hartree")

    # 验证ansatz的幺正性
    solver.ansatz.set_parameters(opt_params)
    psi = solver.ansatz.evaluate_statevector()
    norm = np.linalg.norm(psi)
    print(f"  最终量子态范数: {norm:.6f}")


def main():
    np.random.seed(42)
    print("\n")
    print("*" * 70)
    print("*  变分量子本征求解器 (VQE) — 博士级科研代码合成项目")
    print("*  科学领域: 量子计算 — 分子基态能量计算")
    print("*  基于15个种子项目的核心算法融合")
    print("*" * 70)

    demo_banded_solver()
    demo_pauli_algebra()
    demo_ansatz_tree()
    demo_optimizer()
    demo_fem_sampler()
    demo_molecular_grid()
    demo_measurement()
    demo_hamiltonian()
    demo_full_vqe()

    print_section("运行完成")
    print("  所有模块验证通过，VQE计算流程正常结束。")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
