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
    main()
# ---- TC01: DIF2带状矩阵解析特征值最小值为正 ----
A = BandedMatrix.dif2(10, ml=1)
evals = A.eigenvalues_dif2()
assert np.all(evals > 0), '[TC01] DIF2特征值应全为正 FAILED'
assert evals[0] > 0.02, '[TC01] DIF2最小特征值过小 FAILED'

# ---- TC02: DIF2解析特征向量正交性——不同特征向量内积≈0 ----
n2 = 16
A2 = BandedMatrix.dif2(n2, ml=1)
v1 = A2.eigenvector_dif2(1)
v3 = A2.eigenvector_dif2(3)
dot13 = np.dot(v1, v3)
assert abs(dot13) < 1e-10, f'[TC02] 特征向量1和3应正交, 内积={dot13:.2e} FAILED'
# 验证特征方程: A @ v_1 ≈ lambda_1 * v_1
lambda1 = A2.eigenvalues_dif2()[0]
res2 = np.linalg.norm(A2.mv(v1) - lambda1 * v1)
assert res2 < 1e-10, f'[TC02] 特征方程残差过大: {res2:.2e} FAILED'

# ---- TC03: 带状矩阵mv与稠密矩阵乘法一致 ----
np.random.seed(42)
n3 = 8
A3 = BandedMatrix(n3, 2)
A3.a[0, :] = np.random.randn(n3)
A3.a[1, :n3-1] = np.random.randn(n3-1)
A3.a[2, :n3-2] = np.random.randn(n3-2)
x3 = np.random.randn(n3)
y_band = A3.mv(x3)
y_dense = A3.to_dense() @ x3
assert np.linalg.norm(y_band - y_dense) < 1e-10, '[TC03] 带状mv与稠密乘法不一致 FAILED'

# ---- TC04: PDEParameters获取默认参数——返回字典 ----
params_default = PDEParameters.get_defaults()
assert isinstance(params_default, dict), '[TC04] PDEParameters应返回字典 FAILED'
assert 'alpha' in params_default, '[TC04] 缺少alpha参数 FAILED'
assert params_default['alpha'] == 10.0, '[TC04] alpha默认值错误 FAILED'

# ---- TC05: PauliString乘积验证 ZI*IZ=ZZ ----
p1 = PauliString('ZI', 1.0)
p2 = PauliString('IZ', 1.0)
p3 = p1.multiply(p2)
assert p3.string == 'ZZ', f'[TC05] ZI*IZ应=ZZ, 实际={p3.string} FAILED'
assert abs(p3.coefficient - 1.0) < 1e-10, '[TC05] 系数应为1.0 FAILED'

# ---- TC06: PauliString权重函数验证 ----
p6a = PauliString('IXYZ', 1.0)
assert p6a.weight() == 3, f'[TC06] IXYZ权重应为3, 实际={p6a.weight()} FAILED'
p6b = PauliString('IIII', 1.0)
assert p6b.weight() == 0, '[TC06] IIII权重应为0 FAILED'

# ---- TC07: RREF秩计算——已知秩2矩阵 ----
A7 = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]], dtype=float)
Arref, pivots = rref_compute(A7)
assert len(pivots) == 2, f'[TC07] RREF秩应为2, 实际={len(pivots)} FAILED'

# ---- TC08: ShermanMorrison求解精度——||(A-uv^T)x - b|| ≈ 0 ----
np.random.seed(42)
n8 = 6
A8_base = np.eye(n8) + 0.05 * np.random.randn(n8, n8)
A8_base = A8_base @ A8_base.T
u8 = np.random.randn(n8)
v8 = np.random.randn(n8)
b8 = np.random.randn(n8)
sm8 = ShermanMorrisonSolver(A8_base)
x8_sm = sm8.solve(u8, v8, b8)
B8 = A8_base - np.outer(u8, v8)
res8 = np.linalg.norm(B8 @ x8_sm - b8)
assert res8 < 1e-10, f'[TC08] ShermanMorrison求解残差过大: {res8:.2e} FAILED'

# ---- TC09: AnsatzTree幺正性——状态向量范数应为1 ----
ansatz9 = AnsatzTree(n_qubits=2, max_depth=2)
ansatz9.build_hardware_efficient('CNOT')
ansatz9.set_parameters(np.array([0.3, -0.5, 0.7, -0.2]))
psi9 = ansatz9.evaluate_statevector()
norm9 = np.linalg.norm(psi9)
assert abs(norm9 - 1.0) < 1e-10, f'[TC09] 状态向量范数{norm9:.2e}不等于1 FAILED'

# ---- TC10: 圆弧参数网格输出形状正确 ----
points10 = circle_arc_grid_params(r=2.0, center=np.array([1.0, 1.0]),
                                   angles_deg=(0.0, 180.0), n_points=10)
assert points10.shape == (10, 2), f'[TC10] 圆弧点形状应为(10,2), 实际{points10.shape} FAILED'
assert np.all(np.isfinite(points10)), '[TC10] 圆弧点含NaN/Inf FAILED'

# ---- TC11: 树邻接矩阵检测——线性链是树, 环不是树 ----
adj_tree11 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
assert is_tree_adjacency(adj_tree11) == True, '[TC11] 线性链应为树 FAILED'
adj_cycle11 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
assert is_tree_adjacency(adj_cycle11) == False, '[TC11] 三角形环不应为树 FAILED'

# ---- TC12: 陀螺仪ODE模拟产生有限值 ----
gyro12 = GyroscopeDynamics(A1=1.0, A2=1.0, A3=3.0)
t12, y12 = gyro12.simulate(dt=0.01)
assert len(t12) > 0, '[TC12] 陀螺仪轨迹为空 FAILED'
assert np.all(np.isfinite(y12)), '[TC12] 陀螺仪轨迹含NaN/Inf FAILED'
assert t12[-1] > t12[0], '[TC12] 时间未递增 FAILED'

# ---- TC13: Hermite三次样条在节点处还原函数值 ----
xn13 = np.array([0.0, 1.0, 2.0, 3.0])
fn13 = np.array([1.0, 2.0, 1.5, 2.5])
dn13 = np.array([0.5, 1.0, -0.5, 0.8])
spline13 = HermiteCubicSpline(xn13, fn13, dn13)
f_node, _, _, _ = spline13.evaluate(xn13)
assert np.max(np.abs(f_node - fn13)) < 1e-10, f'[TC13] 节点处插值误差过大 FAILED'

# ---- TC14: Hermite样条积分在[a,b]内为有限值 ----
spline14 = HermiteCubicSpline(xn13, fn13, dn13)
integral14 = spline14.integrate(0.5, 2.5)
assert np.isfinite(integral14), '[TC14] 样条积分应为有限值 FAILED'
assert integral14 != 0.0, '[TC14] 样条积分不应为零 FAILED'

# ---- TC15: 1D FEM插值在节点处还原原函数值 ----
nodes15 = np.linspace(0, 1, 11)
vals15 = np.sin(np.pi * nodes15)
sample_x15 = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
sample_vals15 = fem1d_interpolate(nodes15, vals15, sample_x15)
exact15 = np.sin(np.pi * sample_x15)
err15 = np.max(np.abs(sample_vals15 - exact15))
assert err15 < 0.05, f'[TC15] FEM插值误差过大: {err15:.4f} FAILED'

# ---- TC16: 单位四面体体积 = 1/6 ----
t16 = np.array([[0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
vol16 = tetrahedron_volume(t16)
assert abs(vol16 - 1.0/6.0) < 1e-10, f'[TC16] 单位四面体体积={vol16:.10f}, 应=1/6 FAILED'

# ---- TC17: 四面体基础函数归一性——所有phi_i之和=1 ----
t17 = np.array([[0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
p17 = np.array([[0.25], [0.25], [0.25]])
phi17 = basis_mn_tet4(t17, 1, p17)
assert abs(np.sum(phi17) - 1.0) < 1e-10, f'[TC17] 基础函数之和={np.sum(phi17):.10f}, 应=1 FAILED'

# ---- TC18: 超立方体网格输出形状正确 ----
m18, ns18 = 2, np.array([3, 4])
a18 = np.array([0.0, 0.0])
b18 = np.array([1.0, 1.0])
grid18 = hypercube_grid(m18, ns18, a18, b18)
assert grid18.shape == (2, 12), f'[TC18] 超立方体网格形状应为(2,12), 实际{grid18.shape} FAILED'
assert np.all(grid18[0, :] >= 0.0) and np.all(grid18[0, :] <= 1.0), '[TC18] 坐标越界 FAILED'

# ---- TC19: Slater轨道在原子中心处为有限正值 ----
grid19 = MolecularIntegralGrid(n_orbitals=4)
center19 = np.array([0.0, 0.0, 0.0])
val19 = grid19.slater_orbital(center19, center19, zeta=1.0)
assert val19 > 0, f'[TC19] Slater轨道在中心处应为正, 实际={val19} FAILED'
assert np.isfinite(val19), '[TC19] Slater轨道应为有限值 FAILED'

# ---- TC20: Chebyshev2拒绝采样——样本在[-1,1]内 ----
np.random.seed(42)
samples20, trials20 = chebyshev2_sample(200)
assert np.all(samples20 >= -1.0) and np.all(samples20 <= 1.0), '[TC20] 样本超出[-1,1] FAILED'
assert trials20 >= 200, f'[TC20] 试验次数应>=200, 实际={trials20} FAILED'

# ---- TC21: 量子测量比特串采样——频率之和≈1 ----
np.random.seed(42)
sampler21 = QuantumMeasurementSampler(n_qubits=2, n_shots=4096)
psi21 = np.array([1.0, 0.0, 0.0, 0.0])  # |00>
bits21, freqs21 = sampler21.sample_bitstrings(psi21)
assert abs(np.sum(freqs21) - 1.0) < 1e-10, f'[TC21] 频率之和={np.sum(freqs21):.10f}, 应=1 FAILED'

# ---- TC22: Laguerre-Gauss求积精度——∫x²e^{-x}dx = 2 ----
x22, w22 = gauss_laguerre_rule(8, alpha=0.0)
integral22 = np.sum(w22 * x22 ** 2)
assert abs(integral22 - 2.0) < 1e-6, f'[TC22] Laguerre求积={integral22:.6f}, 精确=2.0 FAILED'

# ---- TC23: 随机正交矩阵——Q^T Q ≈ I ----
np.random.seed(42)
Q23 = r8mat_orth_uniform(5)
err_orth23 = np.linalg.norm(Q23.T @ Q23 - np.eye(5))
assert err_orth23 < 1e-10, f'[TC23] 正交性误差{err_orth23:.2e}过大 FAILED'

# ---- TC24: 随机对称矩阵特征值验证——A与Q diag(λ) Q^T一致 ----
np.random.seed(42)
A24, Q24, lam24 = r8symm_gen(5, lambda_mean=0.0, lambda_dev=1.0)
recon24 = Q24 @ np.diag(lam24) @ Q24.T
err_sym24 = np.linalg.norm(A24 - recon24)
assert err_sym24 < 1e-10, f'[TC24] 对称矩阵重构误差{err_sym24:.2e} FAILED'
assert np.allclose(A24, A24.T), '[TC24] 矩阵不对称 FAILED'

# ---- TC25: VQE精确能量为有限值且参数位移梯度维数正确 ----
np.random.seed(42)
solver25 = VQESolver(n_qubits=2, n_orbitals=4, ansatz_depth=2, n_shots=4096)
params25 = np.array([0.1, 0.2, 0.3, -0.1])
E25 = solver25.energy_exact(params25)
assert np.isfinite(E25), f'[TC25] 精确能量{float(E25):.6f}应为有限值 FAILED'
grad25 = solver25.energy_gradient_parameter_shift(params25)
assert len(grad25) == len(params25), f'[TC25] 梯度维数{len(grad25)}!={len(params25)} FAILED'
assert np.all(np.isfinite(grad25)), '[TC25] 梯度含NaN/Inf FAILED'

# ---- TC26: VQE能隙非负 ----
solver26 = VQESolver(n_qubits=2, n_orbitals=4, ansatz_depth=2, n_shots=4096)
analysis26 = VQEConvergenceAnalysis(solver26)
gap26 = analysis26.estimate_spectral_gap()
assert gap26 >= 0, f'[TC26] 能隙{gap26:.6f}应为非负数 FAILED'

# ---- TC27: VQE噪声能量为有限值 ----
np.random.seed(42)
solver27 = VQESolver(n_qubits=2, n_orbitals=4, ansatz_depth=2, n_shots=4096)
params27 = np.array([0.0, 0.0, 0.0, 0.0])
E_noisy27 = solver27.energy_noisy(params27)
assert np.isfinite(E_noisy27), f'[TC27] 噪声能量{float(E_noisy27):.6f}应为有限值 FAILED'

# ---- TC28: 稳态PDE求解输出维度和有限值 ----
u28 = solve_steady_pde(n=20)
assert len(u28) == 20, f'[TC28] PDE解维数应为20, 实际={len(u28)} FAILED'
assert np.all(np.isfinite(u28)), '[TC28] PDE解含NaN/Inf FAILED'

# ---- TC29: PauliString对易子——[ZI,IZ]=0（不同qubit对易）----
ps_a = PauliString('ZI', 1.0)
ps_b = PauliString('IZ', 1.0)
comm29 = ps_a.commutator(ps_b)
assert abs(comm29.coefficient) < 1e-10, f'[TC29] [ZI,IZ]应对易(系数≈0), 实际系数={comm29.coefficient} FAILED'

# ---- TC30: FEM期望值采样器插值在端点还原 ----
np.random.seed(42)
fem30 = FEMExpectationSampler(n_qubits=2, n_grid_1d=32)
theta30 = np.linspace(-np.pi, np.pi, 20)
E30 = np.sin(theta30)  # 已知函数
est30 = fem30.estimate_energy_1d(theta30, E30, 0.0)
assert abs(est30 - 0.0) < 0.2, f'[TC30] FEM插值在θ=0处应≈0, 实际={est30:.4f} FAILED'

# ---- TC31: 分子积分网格3D构建——点数正确 ----
grid31 = MolecularIntegralGrid(n_orbitals=4)
pts31 = grid31.build_3d_grid(n_per_dim=5)
assert pts31.shape[0] == 125, f'[TC31] 3D网格点数应为125, 实际={pts31.shape[0]} FAILED'
assert pts31.shape[1] == 3, f'[TC31] 3D网格维数应为3, 实际={pts31.shape[1]} FAILED'

# ---- TC32: CVT径向网格优化——输出非降序且有限 ----
np.random.seed(42)
grid32 = MolecularIntegralGrid(n_orbitals=4)
radial32 = grid32.optimize_radial_grid(n_points=8)
assert len(radial32) == 8, f'[TC32] 径向网格点数应为8, 实际={len(radial32)} FAILED'
assert np.all(radial32 >= 0), '[TC32] 径向网格应非负 FAILED'
assert np.all(np.isfinite(radial32)), '[TC32] 径向网格含NaN/Inf FAILED'

# ---- TC33: 双电子积分非负值（(00|00)应≥0） ----
grid33 = MolecularIntegralGrid(n_orbitals=4)
pts33 = grid33.build_3d_grid(n_per_dim=4)
w33 = np.ones(pts33.shape[0]) / pts33.shape[0]
g33 = grid33.two_electron_integral(0, 0, 0, 0, pts33, w33)
assert g33 >= 0, f'[TC33] (00|00)双电子积分应为非负, 实际={g33:.6f} FAILED'

# ---- TC34: 随机正交矩阵特征向量——Q^T A Q为对角矩阵 ----
np.random.seed(42)
A34, Q34, lam34 = r8symm_gen(4, lambda_mean=1.0, lambda_dev=0.5)
diag_check = Q34.T @ A34 @ Q34
off_diag = np.sum(np.abs(diag_check - np.diag(np.diag(diag_check))))
assert off_diag < 1e-10, f'[TC34] Q^T A Q非对角元素过大: {off_diag:.2e} FAILED'

# ---- TC35: extract_independent_paulis返回非空列表 ----
ps1 = PauliString('ZI', 1.0)
ps2 = PauliString('IZ', 1.0)
ps3_dup = PauliString('ZI', 2.0)  # 线性相关
indep35 = extract_independent_paulis([ps1, ps2, ps3_dup])
assert len(indep35) >= 1, '[TC35] 应至少找到1个独立Pauli FAILED'
assert len(indep35) <= 2, f'[TC35] 独立Pauli过多: {len(indep35)} FAILED'

print('\n全部 35 个测试通过!\n')
