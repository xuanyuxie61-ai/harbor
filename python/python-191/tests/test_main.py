"""
main.py

统一入口：MPI并行矩阵乘法在科学计算中的综合验证平台。

本项目围绕"高性能计算：MPI并行矩阵乘法"领域，融合15个种子项目的核心算法，
构建一个面向天体力学、有限元分析与高阶数值积分的博士级科学计算代码库。

运行方式:
    python main.py

无需任何命令行参数。程序自动执行以下实验序列：
    1. 并行矩阵乘法基准测试（Cannon & SUMMA算法）
    2. 矩阵指数与开普勒变分方程传播
    3. Chebyshev-Hermite插值用于核矩阵近似
    4. Gauss-Legendre快速求积与圆盘积分
    5. FEM稀疏刚度矩阵组装
    6. 概率采样与蒙特卡洛验证
    7. 素数分布负载均衡分析
    8. 多项式哈希结果验证
"""

import numpy as np
import time
import sys

# Import all modules
from mpi_cannon_multiply import cannon_multiply, mpi_summa_multiply, frobenius_error
from matrix_exponential_ode import (
    matrix_exponential_pade,
    integrate_kepler_stm,
    kepler_variational_matrix,
    exponential_growth_rate
)
from chebyshev_hermite_approx import (
    cpr_roots,
    hermite_divided_differences,
    hermite_evaluate,
    approximate_matrix_element
)
from legendre_quadrature_kernel import (
    legendre_compute_glr,
    disk01_rule,
    integrate_disk_kernel,
    construct_kernel_matrix_1d,
    construct_kernel_matrix_2d_disk
)
from sparse_fem_topology import (
    TriangularMesh,
    trinity_tile_cover_pattern,
    sparse_matrix_vector_product,
    sparsity_ratio
)
from sampling_distribution import (
    log_normal_truncated_ab_sample,
    walker_build,
    walker_sampler,
    cvt_energy_2d,
    lloyd_step_2d,
    disk01_positive_sample,
    generate_random_matrix_lognormal
)
from parallel_prime_partition import (
    prime_sieve,
    prime_balanced_partition,
    compute_load_imbalance
)
from polynomial_hash_verify import (
    polynomial_hash_matrix,
    verify_matrix_multiply_checksum,
    binary_matrix_multiply_f2,
    collatz_polynomial_sequence
)


def experiment_1_parallel_matrix_multiply():
    """
    实验1: MPI风格并行稠密矩阵乘法基准测试。
    
    测试Cannon算法和SUMMA算法在多种矩阵尺寸下的精度。
    对比基准为numpy.dot的浮点结果。
    """
    print("\n" + "=" * 70)
    print("实验 1: MPI并行矩阵乘法基准测试")
    print("=" * 70)
    
    sizes = [32, 64, 128]
    
    for n in sizes:
        print(f"\n--- 矩阵尺寸: {n} x {n} ---")
        np.random.seed(191)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_ref = np.dot(A, B)
        
        t0 = time.time()
        C_cannon = cannon_multiply(A, B, num_processes=4)
        t_cannon = time.time() - t0
        
        t0 = time.time()
        C_summa = mpi_summa_multiply(A, B, num_processes=4)
        t_summa = time.time() - t0
        
        err_cannon = frobenius_error(C_cannon, C_ref)
        err_summa = frobenius_error(C_summa, C_ref)
        
        print(f"  Cannon 误差: {err_cannon:.3e}, 耗时: {t_cannon:.4f}s")
        print(f"  SUMMA   误差: {err_summa:.3e}, 耗时: {t_summa:.4f}s")
        
        # Verify with checksum
        ok_c = verify_matrix_multiply_checksum(A, B, C_cannon)
        ok_s = verify_matrix_multiply_checksum(A, B, C_summa)
        print(f"  Cannon 校验: {'通过' if ok_c else '失败'}")
        print(f"  SUMMA  校验: {'通过' if ok_s else '失败'}")


def experiment_2_matrix_exponential_kepler():
    """
    实验2: 矩阵指数与开普勒变分方程。
    
    计算旋转矩阵的指数，验证 exp([[0,1],[-1,0]]) = [[cos(1),sin(1)],[-sin(1),cos(1)]]。
    积分开普勒问题状态转移矩阵，验证辛性质 det(Phi) = 1。
    """
    print("\n" + "=" * 70)
    print("实验 2: 矩阵指数与开普勒变分方程")
    print("=" * 70)
    
    # Matrix exponential test
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    E = matrix_exponential_pade(A, order=7)
    E_true = np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]])
    err_exp = np.linalg.norm(E - E_true, 'fro')
    print(f"\n矩阵指数误差: {err_exp:.3e}")
    
    # Kepler STM test
    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    yf, Phi = integrate_kepler_stm(y0, (0.0, 1.0), n_steps=1000)
    det_Phi = np.linalg.det(Phi)
    print(f"开普勒终态: q=({yf[0]:.4f}, {yf[1]:.4f}), p=({yf[2]:.4f}, {yf[3]:.4f})")
    print(f"STM行列式 (应为~1): {det_Phi:.6f}")
    
    # Growth rate of variational matrix at initial state
    M0 = kepler_variational_matrix(y0)
    lam = exponential_growth_rate(M0)
    print(f"变分矩阵最大实特征值: {lam:.4f}")


def experiment_3_chebyshev_hermite():
    """
    实验3: Chebyshev代理根查找与Hermite插值。
    
    使用CPR求解非线性方程的根，Hermite插值近似矩阵核函数。
    """
    print("\n" + "=" * 70)
    print("实验 3: Chebyshev-Hermite插值与根查找")
    print("=" * 70)
    
    # CPR test with higher resolution
    f = lambda x: np.cos(3.0 * x)
    roots, Einter = cpr_roots(f, 0.0, 2.0, N=64)
    true_roots = np.array([np.pi / 6.0, np.pi / 2.0])
    print(f"\nCPR找到的根: {roots}")
    print(f"真实根: {true_roots}")
    print(f"插值残差 Einter: {Einter:.3e}")
    
    # Hermite interpolation for kernel
    x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_nodes = np.sin(x_nodes)
    yp_nodes = np.cos(x_nodes)
    z, d = hermite_divided_differences(x_nodes, y_nodes, yp_nodes)
    
    x_test = np.linspace(0.0, 2.0, 21)
    y_hermite = hermite_evaluate(z, d, x_test)
    y_true = np.sin(x_test)
    err_hermite = np.max(np.abs(y_hermite - y_true))
    print(f"Hermite插值最大误差: {err_hermite:.3e}")
    
    # Matrix element approximation
    kernel = lambda x, y: np.exp(-(x - y) ** 2)
    val = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="hermite")
    val_true = kernel(1.0, 2.0)
    print(f"核矩阵元素近似 (Hermite): {val:.6f} (真实值: {val_true:.6f})")


def experiment_4_legendre_quadrature():
    """
    实验4: Gauss-Legendre快速求积与圆盘积分。
    
    验证Gauss-Legendre规则对多项式的精确性。
    计算圆盘上的核函数积分。
    """
    print("\n" + "=" * 70)
    print("实验 4: Gauss-Legendre求积与圆盘积分")
    print("=" * 70)
    
    # GL exactness test
    x, w = legendre_compute_glr(8)
    # Integral of x^14 from -1 to 1 = 2/15
    val_gl = np.sum(w * x ** 14)
    true_val = 2.0 / 15.0
    print(f"\nGL积分 x^14: {val_gl:.10f} (真实值: {true_val:.10f})")
    
    # Disk integration
    kernel = lambda x, y: x * x + y * y
    val_disk = integrate_disk_kernel(kernel, nr=16, nt=32)
    true_disk = np.pi / 2.0  # integral of r^2 over unit disk = pi/2
    print(f"圆盘积分: {val_disk:.6f} (真实值: {true_disk:.6f})")
    
    # 1D kernel matrix
    nodes = np.linspace(-1.0, 1.0, 5)
    phi = lambda x, c: np.exp(-5.0 * (x - c) ** 2)
    K = construct_kernel_matrix_1d(nodes, phi, quadrature_order=16)
    print(f"核矩阵条件数: {np.linalg.cond(K):.3e}")


def experiment_5_fem_sparse():
    """
    实验5: FEM网格拓扑与稀疏刚度矩阵组装。
    
    构建简单三角网格，组装Laplacian刚度矩阵，分析稀疏模式。
    """
    print("\n" + "=" * 70)
    print("实验 5: FEM稀疏刚度矩阵组装")
    print("=" * 70)
    
    # Create a simple planar mesh
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [1.0, 1.0, 0.0]
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 3, 1],
        [1, 4, 5],
        [1, 5, 2]
    ])
    
    mesh = TriangularMesh(nodes, elements)
    areas = mesh.triangle_areas()
    degrees = mesh.node_degrees()
    print(f"\n三角形数量: {len(elements)}")
    print(f"节点数量: {len(nodes)}")
    print(f"三角形面积: {areas}")
    print(f"节点度数: {degrees}")
    
    data, rows, cols = mesh.assemble_stiffness_matrix_2d()
    sparsity = sparsity_ratio(data, len(nodes), len(nodes))
    print(f"刚度矩阵非零元: {len(data)}, 稀疏度: {sparsity:.4f}")
    
    # Sparse mat-vec
    v = np.random.randn(len(nodes))
    y = sparse_matrix_vector_product(data, rows, cols, v, len(nodes))
    print(f"稀疏矩阵-向量乘积范数: {np.linalg.norm(y):.4f}")
    
    # Trinity pattern
    A1, A2 = trinity_tile_cover_pattern(6, 3)
    print(f"Trinity覆盖模式 A1: {A1.shape}, A2: {A2.shape}")


def experiment_6_sampling_monte_carlo():
    """
    实验6: 概率采样、CVT优化与蒙特卡洛验证。
    
    截断对数正态采样用于生成随机测试矩阵。
    Walker别名方法用于重要性采样。
    CVT优化采样点分布。
    """
    print("\n" + "=" * 70)
    print("实验 6: 概率采样与蒙特卡洛验证")
    print("=" * 70)
    
    # Log-normal sampling
    samples = [log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0) for _ in range(500)]
    print(f"\n截断对数正态样本均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
    
    # Random matrix from log-normal
    M = generate_random_matrix_lognormal(8, mu=0.0, sigma=1.0, a=0.1, b=5.0)
    print(f"随机矩阵范数: {np.linalg.norm(M):.4f}")
    
    # Walker alias sampling
    prob = np.array([0.1, 0.3, 0.4, 0.2])
    y, a = walker_build(prob)
    counts = np.zeros(4)
    for _ in range(10000):
        counts[walker_sampler(y, a)] += 1
    print(f"Walker经验频率: {counts / 10000}")
    
    # CVT optimization
    gens = np.random.rand(5, 2)
    samps = np.random.rand(2000, 2)
    E0 = cvt_energy_2d(gens, samps)
    for _ in range(20):
        gens = lloyd_step_2d(gens, samps)
    E1 = cvt_energy_2d(gens, samps)
    print(f"CVT能量 (Lloyd前/后): {E0:.6f} / {E1:.6f}")
    
    # Disk sampling
    pts = disk01_positive_sample(100)
    r_mean = np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    print(f"圆盘采样平均半径: {r_mean:.4f} (理论: 2/3 ~ 0.667)")


def experiment_7_prime_partition():
    """
    实验7: 素数分布负载均衡分析。
    
    使用素数定理指导进程数量选择，实现矩阵分块的负载均衡。
    """
    print("\n" + "=" * 70)
    print("实验 7: 素数分布与并行负载均衡")
    print("=" * 70)
    
    primes = prime_sieve(100)
    print(f"\n100以内素数个数: {len(primes)}")
    print(f"素数定理预测: {100.0 / np.log(100):.1f}")
    
    for n in [64, 128, 256]:
        p, blocks, imb = prime_balanced_partition(n, 16)
        print(f"\n矩阵尺寸 {n}x{n}:")
        print(f"  选择进程数: {p}")
        print(f"  分块大小: {blocks[:min(4, len(blocks))]}...")
        print(f"  负载不平衡度: {imb:.4f}")


def experiment_8_polynomial_verify():
    """
    实验8: 多项式哈希与模2验证。
    
    Collatz多项式序列生成，矩阵哈希校验，F_2矩阵乘法验证。
    """
    print("\n" + "=" * 70)
    print("实验 8: 多项式哈希与模2验证")
    print("=" * 70)
    
    # Collatz sequence
    p0 = np.array([1, 0, 1, 1])
    seq = collatz_polynomial_sequence(p0, 6)
    print("\nCollatz多项式序列:")
    for i, p in enumerate(seq):
        deg = len(p) - 1 if np.any(p != 0) else -1
        print(f"  步骤 {i}: 度={deg}, 系数={p}")
    
    # Matrix hash
    M = np.random.randn(8, 8)
    h = polynomial_hash_matrix(M, 8)
    print(f"\n随机矩阵哈希: {h[:32]}...")
    
    # F2 multiply
    A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    B = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
    C = binary_matrix_multiply_f2(A, B)
    print(f"\nF_2矩阵乘法:")
    print(f"  A =\n{A}")
    print(f"  B =\n{B}")
    print(f"  A@B (mod 2) =\n{C}")


def run_all_experiments():
    """
    执行所有实验并汇总结果。
    """
    print("\n" + "#" * 70)
    print("#  MPI并行科学计算矩阵乘法综合验证平台")
    print("#  领域: 高性能计算 - MPI并行矩阵乘法")
    print("#  融合15个种子项目的博士级科学计算代码库")
    print("#" * 70)
    
    np.set_printoptions(precision=4, suppress=True)
    
    experiment_1_parallel_matrix_multiply()
    experiment_2_matrix_exponential_kepler()
    experiment_3_chebyshev_hermite()
    experiment_4_legendre_quadrature()
    experiment_5_fem_sparse()
    experiment_6_sampling_monte_carlo()
    experiment_7_prime_partition()
    experiment_8_polynomial_verify()
    
    print("\n" + "#" * 70)
    print("#  所有实验执行完毕，系统正常退出。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    run_all_experiments()

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# Additional imports needed for test cases
from matrix_exponential_ode import kepler_derivatives, kepler_hessian
from chebyshev_hermite_approx import chebyshev_nodes, chebyshev_coefficients
from legendre_quadrature_kernel import rescale_quadrature
from sampling_distribution import normal_01_cdf
from parallel_prime_partition import prime_factorization, is_smooth, find_optimal_process_count, prime_pi, balanced_block_sizes
from polynomial_hash_verify import polynomial_degree, collatz_polynomial_next

# ---- TC01: cannon_multiply identity matrix 4x4 correctness ----
A = np.eye(4)
B = np.eye(4)
C = cannon_multiply(A, B, num_processes=4)
assert np.allclose(C, np.eye(4), atol=1e-12), '[TC01] cannon_multiply identity FAILED'

# ---- TC02: mpi_summa_multiply 4x4 correctness against numpy.dot ----
np.random.seed(42)
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
C_summa = mpi_summa_multiply(A, B, num_processes=4)
C_ref = np.dot(A, B)
assert np.allclose(C_summa, C_ref, atol=1e-10), '[TC02] mpi_summa_multiply FAILED'

# ---- TC03: frobenius_error zero for identical matrices ----
A = np.random.randn(6, 6)
err = frobenius_error(A, A.copy())
assert err == 0.0 or err < 1e-15, '[TC03] frobenius_error identical FAILED'

# ---- TC04: matrix_exponential_pade rotation matrix exp(A) = [[cos1,sin1],[-sin1,cos1]] ----
A = np.array([[0.0, 1.0], [-1.0, 0.0]])
E = matrix_exponential_pade(A, order=7)
E_true = np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]])
assert np.linalg.norm(E - E_true, 'fro') < 1e-10, '[TC04] matrix_exponential_pade rotation FAILED'

# ---- TC05: kepler_derivatives for circular orbit (r=1, p tangential) ----
state = np.array([1.0, 0.0, 0.0, 1.0])
dy = kepler_derivatives(state)
assert abs(dy[0] - 0.0) < 1e-12, '[TC05] kepler_derivatives dq1 FAILED'
assert abs(dy[1] - 1.0) < 1e-12, '[TC05] kepler_derivatives dq2 FAILED'
assert abs(dy[2] + 1.0) < 1e-12, '[TC05] kepler_derivatives dp1 FAILED'
assert abs(dy[3] - 0.0) < 1e-12, '[TC05] kepler_derivatives dp2 FAILED'

# ---- TC06: integrate_kepler_stm symplectic property det(Phi) approx 1 ----
y0 = np.array([1.0, 0.0, 0.0, 1.0])
yf, Phi = integrate_kepler_stm(y0, (0.0, 0.5), n_steps=500)
det_Phi = np.linalg.det(Phi)
assert abs(det_Phi - 1.0) < 0.01, '[TC06] integrate_kepler_stm det(Phi) != 1 FAILED'

# ---- TC07: legendre_compute_glr exactness for polynomial degree 2n-1 (n=8, x^14) ----
x, w = legendre_compute_glr(8)
val_gl = np.sum(w * x ** 14)
true_val = 2.0 / 15.0
assert abs(val_gl - true_val) < 1e-10, '[TC07] Gauss-Legendre quadrature exactness FAILED'

# ---- TC08: integrate_disk_kernel r^2 over unit disk equals pi/2 ----
kernel = lambda x, y: x * x + y * y
val_disk = integrate_disk_kernel(kernel, nr=16, nt=32)
true_disk = np.pi / 2.0
assert abs(val_disk - true_disk) < 1e-4, '[TC08] integrate_disk_kernel FAILED'

# ---- TC09: cpr_roots finds roots for cos(3x) on [0, 2] (check roots exist) ----
f = lambda x: np.cos(3.0 * x)
roots, Einter = cpr_roots(f, 0.0, 2.0, N=128)
assert len(roots) >= 2, '[TC09] cpr_roots count FAILED'
# Check roots are near expected values (pi/6, pi/2) within tolerance
assert np.any(np.abs(roots - np.pi / 6.0) < 0.1), '[TC09] cpr_roots root pi/6 FAILED'
assert np.any(np.abs(roots - np.pi / 2.0) < 0.1), '[TC09] cpr_roots root pi/2 FAILED'

# ---- TC10: hermite interpolation accuracy for sin(x) with derivative info ----
x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_nodes = np.sin(x_nodes)
yp_nodes = np.cos(x_nodes)
z, d = hermite_divided_differences(x_nodes, y_nodes, yp_nodes)
x_test = np.linspace(0.0, 2.0, 51)
y_hermite = hermite_evaluate(z, d, x_test)
y_true = np.sin(x_test)
err_max = np.max(np.abs(y_hermite - y_true))
assert err_max < 0.01, '[TC10] hermite interpolation accuracy FAILED'

# ---- TC11: chebyshev_nodes monotonic and within [a,b] ----
a, b, N = -2.0, 3.0, 10
xn = chebyshev_nodes(a, b, N)
assert len(xn) == N + 1, '[TC11] chebyshev_nodes count FAILED'
assert np.min(xn) >= a - 1e-12, '[TC11] chebyshev_nodes lower bound FAILED'
assert np.max(xn) <= b + 1e-12, '[TC11] chebyshev_nodes upper bound FAILED'
assert np.all(np.diff(xn) <= 0), '[TC11] chebyshev_nodes monotonic FAILED'

# ---- TC12: TriangularMesh triangle_areas for right triangle (0,0)-(1,0)-(0,1) ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
elements = np.array([[0, 1, 2]])
mesh = TriangularMesh(nodes, elements)
areas = mesh.triangle_areas()
assert len(areas) == 1, '[TC12] triangle_areas count FAILED'
assert abs(areas[0] - 0.5) < 1e-12, '[TC12] triangle_areas value FAILED'

# ---- TC13: stiffness matrix COO data symmetrical structure ----
nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 0.0]])
elements = np.array([[0, 1, 2], [0, 3, 1]])
mesh2 = TriangularMesh(nodes, elements)
data, rows, cols = mesh2.assemble_stiffness_matrix_2d()
assert len(data) > 0, '[TC13] stiffness matrix non-empty FAILED'
assert len(rows) == len(cols) == len(data), '[TC13] stiffness matrix COO shape mismatch FAILED'

# ---- TC14: sparse_matrix_vector_product correctness against dense multiplication ----
np.random.seed(42)
data_sp = np.array([2.0, -1.0, -1.0, 2.0])
row_sp = np.array([0, 0, 1, 1], dtype=np.int64)
col_sp = np.array([0, 1, 0, 1], dtype=np.int64)
x_vec = np.array([3.0, 4.0])
y_sp = sparse_matrix_vector_product(data_sp, row_sp, col_sp, x_vec, 2)
A_dense = np.array([[2.0, -1.0], [-1.0, 2.0]])
y_dense = np.dot(A_dense, x_vec)
assert np.allclose(y_sp, y_dense, atol=1e-12), '[TC14] sparse_matrix_vector_product FAILED'

# ---- TC15: normal_01_cdf monotonic and boundary values ----
assert normal_01_cdf(-10.0) < 1e-10, '[TC15] normal_01_cdf lower tail FAILED'
assert normal_01_cdf(10.0) > 0.999999, '[TC15] normal_01_cdf upper tail FAILED'
assert abs(normal_01_cdf(0.0) - 0.5) < 1e-8, '[TC15] normal_01_cdf median FAILED'

# ---- TC16: walker_build and walker_sampler produce correct distribution (fixed seed) ----
np.random.seed(42)
prob = np.array([0.1, 0.3, 0.4, 0.2])
y, a = walker_build(prob)
assert len(y) == 4 and len(a) == 4, '[TC16] walker_build shape FAILED'
counts = np.zeros(4)
for _ in range(5000):
    counts[walker_sampler(y, a)] += 1
assert np.all(counts > 0), '[TC16] walker_sampler all categories sampled FAILED'

# ---- TC17: cvt_energy_2d non-negative and Lloyd step reduces energy ----
np.random.seed(42)
gens = np.random.rand(5, 2)
samps = np.random.rand(500, 2)
E0 = cvt_energy_2d(gens, samps)
assert E0 >= 0.0, '[TC17] cvt_energy_2d non-negative FAILED'
gens_new = lloyd_step_2d(gens, samps)
E1 = cvt_energy_2d(gens_new, samps)
assert E1 <= E0 + 1e-12, '[TC17] Lloyd energy monotonic FAILED'

# ---- TC18: disk01_positive_sample points lie within unit disk (fixed seed) ----
np.random.seed(42)
pts = disk01_positive_sample(500)
r_sq = pts[:, 0] ** 2 + pts[:, 1] ** 2
assert np.all(r_sq <= 1.0 + 1e-12), '[TC18] disk01_positive_sample radius FAILED'
assert np.all(r_sq >= 0.0), '[TC18] disk01_positive_sample non-negative FAILED'

# ---- TC19: generate_random_matrix_lognormal shape correct ----
np.random.seed(42)
M_lognorm = generate_random_matrix_lognormal(6, mu=0.0, sigma=0.5, a=1e-6, b=10.0)
assert M_lognorm.shape == (6, 6), '[TC19] generate_random_matrix_lognormal shape FAILED'
assert np.all(np.isfinite(M_lognorm)), '[TC19] generate_random_matrix_lognormal finite FAILED'

# ---- TC20: prime_sieve known primes below 30 ----
primes = prime_sieve(30)
expected = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
assert np.array_equal(primes, expected), '[TC20] prime_sieve FAILED'

# ---- TC21: prime_factorization known factorizations ----
factors_12 = prime_factorization(12)
assert (2, 2) in factors_12, '[TC21] prime_factorization 2^2 FAILED'
assert (3, 1) in factors_12, '[TC21] prime_factorization 3^1 FAILED'

# ---- TC22: balanced_block_sizes sum equals n ----
n, p = 10, 3
blocks = balanced_block_sizes(n, p)
assert len(blocks) == p, '[TC22] balanced_block_sizes count FAILED'
assert sum(blocks) == n, '[TC22] balanced_block_sizes sum FAILED'

# ---- TC23: compute_load_imbalance zero for uniform blocks ----
imb = compute_load_imbalance([5, 5, 5, 5])
assert imb == 0.0, '[TC23] compute_load_imbalance uniform FAILED'

# ---- TC24: polynomial_degree for known polynomials ----
assert polynomial_degree(np.array([1, 0, 1, 0])) == 2, '[TC24] polynomial_degree nonzero FAILED'
assert polynomial_degree(np.array([0, 0, 0])) == -1, '[TC24] polynomial_degree zero poly FAILED'
assert polynomial_degree(np.array([0, 0, 1])) == 2, '[TC24] polynomial_degree degree 2 FAILED'

# ---- TC25: collatz_polynomial_sequence deterministic output ----
p0 = np.array([1, 0, 1])
seq = collatz_polynomial_sequence(p0, 3)
assert len(seq) == 4, '[TC25] collatz sequence length FAILED'
assert np.array_equal(seq[0], np.array([1, 0, 1])), '[TC25] collatz sequence first FAILED'

# ---- TC26: binary_matrix_multiply_f2 known result for small matrices ----
A_f2 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
B_f2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
C_f2 = binary_matrix_multiply_f2(A_f2, B_f2)
assert C_f2.shape == (3, 3), '[TC26] binary_matrix_multiply_f2 shape FAILED'
assert np.all((C_f2 == 0) | (C_f2 == 1)), '[TC26] binary_matrix_multiply_f2 binary FAILED'

# ---- TC27: verify_matrix_multiply_checksum passes for valid product ----
np.random.seed(42)
A_chk = np.random.randn(4, 4)
B_chk = np.random.randn(4, 4)
C_chk = np.dot(A_chk, B_chk)
assert verify_matrix_multiply_checksum(A_chk, B_chk, C_chk), '[TC27] verify checksum valid FAILED'

# ---- TC28: exponential_growth_rate of zero matrix is zero ----
Z = np.zeros((3, 3))
lam = exponential_growth_rate(Z)
assert abs(lam) < 1e-12, '[TC28] exponential_growth_rate zero matrix FAILED'

# ---- TC29: kepler_hessian matrix is symmetric ----
state_h = np.array([1.0, 0.5, 0.0, 0.8])
H = kepler_hessian(state_h)
assert H.shape == (2, 2), '[TC29] kepler_hessian shape FAILED'
assert np.allclose(H, H.T, atol=1e-12), '[TC29] kepler_hessian symmetry FAILED'

# ---- TC30: approximate_matrix_element both methods return finite values ----
kernel = lambda x, y: np.exp(-(x - y) ** 2)
val_cheb = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="chebyshev")
val_herm = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="hermite")
assert np.isfinite(val_cheb), '[TC30] approximate_matrix_element chebyshev FAILED'
assert np.isfinite(val_herm), '[TC30] approximate_matrix_element hermite FAILED'

# ---- TC31: cannon_multiply on 8x8 random matrices (sequential fallback) ----
np.random.seed(191)
A8 = np.random.randn(8, 8)
B8 = np.random.randn(8, 8)
C8 = cannon_multiply(A8, B8, num_processes=4)
C8_ref = np.dot(A8, B8)
assert np.allclose(C8, C8_ref, atol=1e-10), '[TC31] cannon_multiply 8x8 FAILED'

# ---- TC32: matrix_exponential_pade zero matrix returns identity ----
Z2 = np.zeros((3, 3))
E_zero = matrix_exponential_pade(Z2, order=5)
assert np.allclose(E_zero, np.eye(3), atol=1e-12), '[TC32] matrix_exponential_pade zero FAILED'

# ---- TC33: sparsity_ratio known sparse pattern ----
data_sr = np.array([1.0, 2.0, 3.0])
sr = sparsity_ratio(data_sr, 5, 5)
assert 0.85 < sr < 0.95, '[TC33] sparsity_ratio FAILED'

# ---- TC34: disk01_rule weights are positive ----
w_disk, r_disk, t_disk = disk01_rule(8, 16)
assert np.all(w_disk > 0), '[TC34] disk01_rule weights positive FAILED'
assert len(r_disk) == 8 and len(t_disk) == 16, '[TC34] disk01_rule shape FAILED'

# ---- TC35: prime_balanced_partition returns valid process count ----
p_bp, blocks_bp, imb_bp = prime_balanced_partition(128, 32)
assert p_bp >= 1, '[TC35] prime_balanced_partition process count FAILED'
assert len(blocks_bp) == p_bp, '[TC35] prime_balanced_partition blocks count FAILED'
assert imb_bp >= 0.0, '[TC35] prime_balanced_partition imbalance FAILED'

# ---- TC36: log_normal_truncated_ab_sample produces values in [a,b] (fixed seed) ----
np.random.seed(42)
for _ in range(200):
    val = log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0)
    assert 0.1 - 1e-12 <= val <= 5.0 + 1e-12, '[TC36] log_normal_truncated_ab_sample bounds FAILED'

# ---- TC37: polynomial_hash_matrix deterministic output ----
M_hash = np.array([[1.0, 0.0], [0.0, 1.0]])
h1 = polynomial_hash_matrix(M_hash, 5)
h2 = polynomial_hash_matrix(M_hash, 5)
assert h1 == h2, '[TC37] polynomial_hash_matrix deterministic FAILED'
assert len(h1) > 0, '[TC37] polynomial_hash_matrix non-empty FAILED'

# ---- TC38: construct_kernel_matrix_1d positive semidefinite ----
nodes_k = np.array([0.0, 1.0])
phi = lambda x, c: np.exp(-5.0 * (x - c) ** 2)
K_mat = construct_kernel_matrix_1d(nodes_k, phi, quadrature_order=16)
assert K_mat.shape == (2, 2), '[TC38] construct_kernel_matrix_1d shape FAILED'
eigs = np.linalg.eigvalsh(K_mat)
assert np.all(eigs > -1e-10), '[TC38] construct_kernel_matrix_1d PSD FAILED'

# ---- TC39: hermite_divided_differences with single node ----
z_1, d_1 = hermite_divided_differences(np.array([0.0]), np.array([1.0]), np.array([2.0]))
assert len(z_1) == 2 and len(d_1) == 2, '[TC39] hermite single node shape FAILED'

# ---- TC40: frobenius_error returns float type ----
A_fe = np.eye(3)
B_fe = 2 * np.eye(3)
err_fe = frobenius_error(A_fe, B_fe)
assert isinstance(err_fe, float), '[TC40] frobenius_error return type FAILED'
assert err_fe > 0.0, '[TC40] frobenius_error positive for non-identical FAILED'

# ---- TC41: kepler_variational_matrix 4x4 structure ----
state_v = np.array([1.0, 0.0, 0.0, 1.0])
M_var = kepler_variational_matrix(state_v)
assert M_var.shape == (4, 4), '[TC41] kepler_variational_matrix shape FAILED'
assert M_var[0, 2] == 1.0 and M_var[1, 3] == 1.0, '[TC41] kepler_variational_matrix structure FAILED'

# ---- TC42: is_smooth detects 7-smooth numbers ----
assert is_smooth(12, max_prime=7), '[TC42] is_smooth 12 FAILED'
assert not is_smooth(11, max_prime=7), '[TC42] is_smooth prime 11 FAILED'

# ---- TC43: triangular mesh node_degrees correct ----
nodes_d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.2, 0.0]])
elements_d = np.array([[0, 1, 2], [1, 3, 2]])
mesh_d = TriangularMesh(nodes_d, elements_d)
deg = mesh_d.node_degrees()
assert deg[2] == 2, '[TC43] node_degrees middle node FAILED'

# ---- TC44: construct_kernel_matrix_2d_disk shape ----
nodes_2d = np.array([[0.0, 0.0], [0.5, 0.5]])
phi_2d = lambda x, y, cx, cy: np.exp(-((x - cx) ** 2 + (y - cy) ** 2))
K_2d = construct_kernel_matrix_2d_disk(nodes_2d, phi_2d, nr=8, nt=16)
assert K_2d.shape == (2, 2), '[TC44] construct_kernel_matrix_2d_disk shape FAILED'
assert np.all(np.isfinite(K_2d)), '[TC44] construct_kernel_matrix_2d_disk finite FAILED'

# ---- TC45: prime_pi matches direct count ----
primes_50 = prime_sieve(50)
assert prime_pi(50) == len(primes_50), '[TC45] prime_pi FAILED'

# ---- TC46: collatz_polynomial_next zero polynomial stays zero ----
p_zero = np.array([0])
p_next = collatz_polynomial_next(p_zero)
assert np.array_equal(p_next, np.array([0])), '[TC46] collatz zero poly FAILED'

# ---- TC47: chebyshev_coefficients returns correct length and finite values ----
f_vals = np.cos(np.arange(9))
c = chebyshev_coefficients(f_vals)
assert len(c) == len(f_vals), '[TC47] chebyshev_coefficients length FAILED'
assert np.all(np.isfinite(c)), '[TC47] chebyshev_coefficients finite FAILED'

# ---- TC48: rescale_quadrature preserves weight sum ----
x_q, w_q = legendre_compute_glr(5)
t_scaled, w_scaled = rescale_quadrature(x_q, w_q, -1.0, 3.0)
assert abs(np.sum(w_scaled) - 4.0) < 1e-12, '[TC48] rescale_quadrature weight sum FAILED'

# ---- TC49: find_optimal_process_count returns valid result ----
opt_p = find_optimal_process_count(64, 16, algorithm="cannon")
assert 1 <= opt_p <= 16, '[TC49] find_optimal_process_count range FAILED'

# ---- TC50: trinity_tile_cover_pattern output shapes ----
A1, A2 = trinity_tile_cover_pattern(6, 3)
assert A1.shape[0] == 6, '[TC50] trinity A1 rows FAILED'
assert A2.shape[0] == 3, '[TC50] trinity A2 rows FAILED'
assert A1.shape[1] == A2.shape[1], '[TC50] trinity column mismatch FAILED'
print('\n全部 50 个测试通过!\n')
