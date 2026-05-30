
import numpy as np
import time
import sys


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
        

        ok_c = verify_matrix_multiply_checksum(A, B, C_cannon)
        ok_s = verify_matrix_multiply_checksum(A, B, C_summa)
        print(f"  Cannon 校验: {'通过' if ok_c else '失败'}")
        print(f"  SUMMA  校验: {'通过' if ok_s else '失败'}")


def experiment_2_matrix_exponential_kepler():
    print("\n" + "=" * 70)
    print("实验 2: 矩阵指数与开普勒变分方程")
    print("=" * 70)
    

    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    E = matrix_exponential_pade(A, order=7)
    E_true = np.array([[np.cos(1.0), np.sin(1.0)], [-np.sin(1.0), np.cos(1.0)]])
    err_exp = np.linalg.norm(E - E_true, 'fro')
    print(f"\n矩阵指数误差: {err_exp:.3e}")
    

    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    yf, Phi = integrate_kepler_stm(y0, (0.0, 1.0), n_steps=1000)
    det_Phi = np.linalg.det(Phi)
    print(f"开普勒终态: q=({yf[0]:.4f}, {yf[1]:.4f}), p=({yf[2]:.4f}, {yf[3]:.4f})")
    print(f"STM行列式 (应为~1): {det_Phi:.6f}")
    

    M0 = kepler_variational_matrix(y0)
    lam = exponential_growth_rate(M0)
    print(f"变分矩阵最大实特征值: {lam:.4f}")


def experiment_3_chebyshev_hermite():
    print("\n" + "=" * 70)
    print("实验 3: Chebyshev-Hermite插值与根查找")
    print("=" * 70)
    

    f = lambda x: np.cos(3.0 * x)
    roots, Einter = cpr_roots(f, 0.0, 2.0, N=64)
    true_roots = np.array([np.pi / 6.0, np.pi / 2.0])
    print(f"\nCPR找到的根: {roots}")
    print(f"真实根: {true_roots}")
    print(f"插值残差 Einter: {Einter:.3e}")
    

    x_nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_nodes = np.sin(x_nodes)
    yp_nodes = np.cos(x_nodes)
    z, d = hermite_divided_differences(x_nodes, y_nodes, yp_nodes)
    
    x_test = np.linspace(0.0, 2.0, 21)
    y_hermite = hermite_evaluate(z, d, x_test)
    y_true = np.sin(x_test)
    err_hermite = np.max(np.abs(y_hermite - y_true))
    print(f"Hermite插值最大误差: {err_hermite:.3e}")
    

    kernel = lambda x, y: np.exp(-(x - y) ** 2)
    val = approximate_matrix_element(kernel, 1.0, 2.0, order=8, method="hermite")
    val_true = kernel(1.0, 2.0)
    print(f"核矩阵元素近似 (Hermite): {val:.6f} (真实值: {val_true:.6f})")


def experiment_4_legendre_quadrature():
    print("\n" + "=" * 70)
    print("实验 4: Gauss-Legendre求积与圆盘积分")
    print("=" * 70)
    

    x, w = legendre_compute_glr(8)

    val_gl = np.sum(w * x ** 14)
    true_val = 2.0 / 15.0
    print(f"\nGL积分 x^14: {val_gl:.10f} (真实值: {true_val:.10f})")
    

    kernel = lambda x, y: x * x + y * y
    val_disk = integrate_disk_kernel(kernel, nr=16, nt=32)
    true_disk = np.pi / 2.0
    print(f"圆盘积分: {val_disk:.6f} (真实值: {true_disk:.6f})")
    

    nodes = np.linspace(-1.0, 1.0, 5)
    phi = lambda x, c: np.exp(-5.0 * (x - c) ** 2)
    K = construct_kernel_matrix_1d(nodes, phi, quadrature_order=16)
    print(f"核矩阵条件数: {np.linalg.cond(K):.3e}")


def experiment_5_fem_sparse():
    print("\n" + "=" * 70)
    print("实验 5: FEM稀疏刚度矩阵组装")
    print("=" * 70)
    

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
    

    v = np.random.randn(len(nodes))
    y = sparse_matrix_vector_product(data, rows, cols, v, len(nodes))
    print(f"稀疏矩阵-向量乘积范数: {np.linalg.norm(y):.4f}")
    

    A1, A2 = trinity_tile_cover_pattern(6, 3)
    print(f"Trinity覆盖模式 A1: {A1.shape}, A2: {A2.shape}")


def experiment_6_sampling_monte_carlo():
    print("\n" + "=" * 70)
    print("实验 6: 概率采样与蒙特卡洛验证")
    print("=" * 70)
    

    samples = [log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0) for _ in range(500)]
    print(f"\n截断对数正态样本均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")
    

    M = generate_random_matrix_lognormal(8, mu=0.0, sigma=1.0, a=0.1, b=5.0)
    print(f"随机矩阵范数: {np.linalg.norm(M):.4f}")
    

    prob = np.array([0.1, 0.3, 0.4, 0.2])
    y, a = walker_build(prob)
    counts = np.zeros(4)
    for _ in range(10000):
        counts[walker_sampler(y, a)] += 1
    print(f"Walker经验频率: {counts / 10000}")
    

    gens = np.random.rand(5, 2)
    samps = np.random.rand(2000, 2)
    E0 = cvt_energy_2d(gens, samps)
    for _ in range(20):
        gens = lloyd_step_2d(gens, samps)
    E1 = cvt_energy_2d(gens, samps)
    print(f"CVT能量 (Lloyd前/后): {E0:.6f} / {E1:.6f}")
    

    pts = disk01_positive_sample(100)
    r_mean = np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    print(f"圆盘采样平均半径: {r_mean:.4f} (理论: 2/3 ~ 0.667)")


def experiment_7_prime_partition():
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
    print("\n" + "=" * 70)
    print("实验 8: 多项式哈希与模2验证")
    print("=" * 70)
    

    p0 = np.array([1, 0, 1, 1])
    seq = collatz_polynomial_sequence(p0, 6)
    print("\nCollatz多项式序列:")
    for i, p in enumerate(seq):
        deg = len(p) - 1 if np.any(p != 0) else -1
        print(f"  步骤 {i}: 度={deg}, 系数={p}")
    

    M = np.random.randn(8, 8)
    h = polynomial_hash_matrix(M, 8)
    print(f"\n随机矩阵哈希: {h[:32]}...")
    

    A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    B = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1]])
    C = binary_matrix_multiply_f2(A, B)
    print(f"\nF_2矩阵乘法:")
    print(f"  A =\n{A}")
    print(f"  B =\n{B}")
    print(f"  A@B (mod 2) =\n{C}")


def run_all_experiments():
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
