# -*- coding: utf-8 -*-
"""
main.py
=======
统一入口：基于谱预处理与随机SVD的高维各向异性稀疏线性系统
         自适应迭代求解器综合验证平台。

科学问题：
---------
本项目围绕"计算数学：稀疏矩阵迭代求解与预处理"，综合 15 个种子项目的
核心算法，构造并求解以下前沿博士级问题：

  考虑由二维各向异性扩散方程、Helmholtz 方程、Darcy 流模型及
  Hankel 型矩问题离散化得到的大型稀疏线性系统 Ax = b。
  由于系数矩阵常具有极大条件数（κ ~ 10^4–10^8）、特征值谱聚类不良、
  或来自随机介质的剧烈系数振荡，标准 CG 方法收敛极慢。

  本项目实现并对比以下高级技术：
    1) 多种稀疏存储格式（R83, R8PBU, R8SD, COO）上的 CG/PCG；
    2) 基于 Laguerre/Hermite 正交多项式零点的谱等价预处理；
    3) 不完全 Cholesky (IC0) 与 SSOR 预处理；
    4) 随机 SVD 与 Hutchinson 迹估计用于谱分布分析；
    5) Halton 准随机序列用于随机探测与蒙特卡洛验证；
    6) CVT 自适应网格生成用于构造局部加密测试问题；
    7) Lambert W 函数精细化理论收敛速率估计；
    8) 基于 Steinerberger 函数、Genz 测试包、多元多项式（Rosenbrock等）
       的复杂右端项构造；
    9) 多层网格粗化（mesh refinement）与块对角预处理；
    10) 校验和机制（改编自 ISBN 思想）用于解向量一致性验证。

运行方式：
---------
    python main.py
无需任何命令行参数。
"""

import numpy as np
import math
import sys

# 导入所有模块
from utils import (
    print_header, print_vec, checksum_vector, verify_checksum,
    condition_number_estimate, is_spd, vec_norm
)
from special_functions import (
    lambert_w_fast, lambert_w_convergence_rate,
    steinerberger_function, steinerberger_integral_01, harmonic_number
)
from random_tools import (
    halton_value, halton_sequence, random_orthogonal_matrix,
    random_spd_matrix, random_spd_with_clustered_spectrum,
    randomized_svd_approx, hutchinson_trace_estimator
)
from grid_generation import (
    line_grid, cvt_1d_lloyd, chebyshev_zero_density,
    mesh_refinement_1d, multi_level_grid
)
from orthogonal_polynomials import (
    laguerre_polynomial, generalized_laguerre_function,
    gauss_laguerre_rule, gauss_generalized_laguerre_rule,
    gauss_hermite_rule
)
from quadrature_rules import (
    spherical_to_cartesian, lebedev_rule_14, integrate_on_sphere,
    genz_evaluate, genz_integral_exact, qmc_integral_halton,
    multidimensional_gauss_legendre_simple
)
from sparse_matrix import (
    dif2_r8ge, dif2_r83, dif2_r83s, dif2_r83t,
    dif2_r8pbu, dif2_r8sd, dif2_r8sp,
    r8ge_mv, r83_mv, r83s_mv, r83t_mv,
    r8pbu_mv, r8sd_mv, r8sp_mv,
    SparseMatrixOperator
)
from test_problems import (
    hankel_spd_cholesky_lower, hankel_spd_from_moments,
    anisotropic_diffusion_2d, helmholtz_2d,
    cavity_flow_stokes_matrix, cavity_flow_rhs,
    rosenbrock_rhs, camel_rhs,
    build_test_problem
)
from preconditioner import (
    jacobi_preconditioner, ssor_preconditioner,
    incomplete_cholesky_ic0, polynomial_spectral_preconditioner,
    block_diagonal_preconditioner
)
from conjugate_gradient import (
    conjugate_gradient, preconditioned_cg, flexible_cg,
    restarted_cg, solve_with_format
)
from convergence_analysis import (
    theoretical_cg_error_bound, theoretical_cg_iteration_count,
    lambert_w_refined_convergence_bound,
    preconditioner_quality, compare_solvers,
    estimate_condition_number_randomized
)


def demo_sparse_formats_cg():
    """演示多种稀疏存储格式下的 CG 求解。"""
    print_header("Demo 1: Sparse Format CG Solvers on DIF2 Matrix")
    n = 64
    b = np.ones(n, dtype=float)
    formats = {
        'R8GE (Dense)': ('ge', dif2_r8ge(n), {}),
        'R83 (Tridiag vertical)': ('r83', dif2_r83(n), {}),
        'R83S (Tridiag scalar)': ('r83s', dif2_r83s(), {}),
        'R83T (Tridiag horizontal)': ('r83t', dif2_r83t(n), {}),
        'R8PBU (Symmetric banded)': ('pbu', dif2_r8pbu(n, mu=1), {'mu': 1}),
    }
    results = {}
    for name, (fmt, data, extra) in formats.items():
        op = SparseMatrixOperator(fmt, data, n, extra)
        x, info = conjugate_gradient(op.matvec, b, tol=1e-12, max_iter=n)
        results[name] = info
        print(f"  {name:30s} | Iter: {info['iterations']:3d} | Res: {info['final_residual']:.3e}")
    print()


def demo_pcg_preconditioners():
    """演示不同预处理子对 PCG 的加速效果。"""
    print_header("Demo 2: Preconditioned CG Comparison")
    n = 256
    A, b, x_exact = build_test_problem('aniso2d', n, extra={'nx': 16, 'ny': 16, 'eps_x': 1.0, 'eps_y': 0.001})
    matvec = lambda v: A @ v

    # 标准 CG
    x_cg, info_cg = conjugate_gradient(matvec, b, tol=1e-8, max_iter=300)

    # Jacobi-PCG
    prec_jac = jacobi_preconditioner(A)
    x_jac, info_jac = preconditioned_cg(matvec, prec_jac, b, tol=1e-8, max_iter=300)

    # SSOR-PCG
    prec_ssor = ssor_preconditioner(A, omega=1.5)
    x_ssor, info_ssor = preconditioned_cg(matvec, prec_ssor, b, tol=1e-8, max_iter=300)

    # IC0-PCG
    prec_ic0 = incomplete_cholesky_ic0(A)
    x_ic0, info_ic0 = preconditioned_cg(matvec, prec_ic0, b, tol=1e-8, max_iter=300)

    # Spectral-PCG (Laguerre)
    prec_spec = polynomial_spectral_preconditioner(A, poly_type='laguerre', n_nodes=8)
    x_spec, info_spec = preconditioned_cg(matvec, prec_spec, b, tol=1e-8, max_iter=300)

    results = {
        'CG (no prec)': info_cg,
        'PCG-Jacobi': info_jac,
        'PCG-SSOR': info_ssor,
        'PCG-IC0': info_ic0,
        'PCG-Spectral(Laguerre)': info_spec,
    }
    print(compare_solvers(results))
    print()


def demo_orthogonal_polynomials():
    """演示正交多项式与求积规则。"""
    print_header("Demo 3: Orthogonal Polynomials & Quadrature Rules")
    n = 8
    x_eval = np.array([0.5, 1.5, 3.0])

    # Laguerre
    v = laguerre_polynomial(len(x_eval), n, x_eval)
    print(f"  Laguerre L_{n}(x) at x={x_eval}: {v[:, n]}")

    # Generalized Laguerre
    v2 = generalized_laguerre_function(len(x_eval), n, alpha=0.5, x=x_eval)
    print(f"  Gen.Laguerre Lf_{n}^{{(0.5)}}(x) at x={x_eval}: {v2[:, n]}")

    # Gauss-Laguerre quadrature: ∫_0^∞ exp(-x) x^2 dx = 2! = 2
    nodes, weights = gauss_laguerre_rule(n)
    approx = float(np.dot(weights, nodes ** 2))
    print(f"  Gauss-Laguerre (n={n}) ∫ exp(-x) x^2 dx ≈ {approx:.6f} (exact: 2.0)")

    # Gauss-Hermite quadrature: ∫_{-∞}^{∞} exp(-x^2) x^2 dx = sqrt(pi)/2
    nodes_h, weights_h = gauss_hermite_rule(n)
    approx_h = float(np.dot(weights_h, nodes_h ** 2))
    exact_h = math.sqrt(math.pi) / 2.0
    print(f"  Gauss-Hermite (n={n}) ∫ exp(-x^2) x^2 dx ≈ {approx_h:.6f} (exact: {exact_h:.6f})")
    print()


def demo_halton_and_random_tools():
    """演示 Halton 序列与随机矩阵工具。"""
    print_header("Demo 4: Halton Sequence & Random Matrix Generation")
    pts = halton_sequence(0, 9, 2)
    print(f"  Halton 2D points (0..9):\n{pts.T}")

    Q = random_orthogonal_matrix(5, seed=42)
    orth_err = np.linalg.norm(Q.T @ Q - np.eye(5))
    print(f"  Random orthogonal matrix 5x5 orthogonality error: {orth_err:.3e}")

    A, lam, Q2 = random_spd_matrix(10, seed=42)
    print(f"  Random SPD 10x10 eigenvalue range: [{lam.min():.4f}, {lam.max():.4f}]")

    # Hutchinson trace estimate
    matvec = lambda v: A @ v
    tr_est = hutchinson_trace_estimator(matvec, 10, num_samples=50, seed=42)
    tr_exact = float(np.trace(A))
    print(f"  Hutchinson trace estimate: {tr_est:.4f} (exact: {tr_exact:.4f})")
    print()


def demo_cvt_grid():
    """演示 CVT 自适应网格生成。"""
    print_header("Demo 5: CVT Adaptive Grid Generation")
    g, energy, motion = cvt_1d_lloyd(
        n=16, it_num=20, s_num=2000,
        density_func=chebyshev_zero_density, init_type=2
    )
    print(f"  CVT final generator range: [{g.min():.4f}, {g.max():.4f}]")
    print(f"  Final energy: {energy[-1]:.6e}")
    print()


def demo_special_functions():
    """演示 Lambert W 与 Steinerberger 函数。"""
    print_header("Demo 6: Special Functions (Lambert W & Steinerberger)")
    for x in [0.5, 1.0, 2.0, 5.0]:
        w, en = lambert_w_fast(x)
        print(f"  W({x}) ≈ {w:.6f} (last correction: {en:.3e})")

    # 收敛速率估计
    kappa = 1000.0
    rho, refined = lambert_w_convergence_rate(kappa)
    print(f"  CG convergence rate for κ={kappa}: ρ={rho:.6f}, refined log≈{refined:.6e}")

    # Steinerberger
    n_sb = 10
    val_int = steinerberger_integral_01(n_sb)
    val_hn = 2.0 * harmonic_number(n_sb) / math.pi
    print(f"  Steinerberger integral I({n_sb}) = {val_int:.6f} (via H_n: {val_hn:.6f})")
    print()


def demo_hankel_and_test_problems():
    """演示 Hankel SPD 矩阵与综合测试问题。"""
    print_header("Demo 7: Hankel SPD & Test Problems")
    n = 16
    lii = np.linspace(1.0, 0.5, n)
    liim1 = 0.3 * np.ones(n - 1)
    L, H = hankel_spd_cholesky_lower(n, lii, liim1)
    spd_ok = is_spd(H)
    print(f"  Hankel SPD {n}x{n} is SPD: {spd_ok}")

    # 测试问题对比
    problems = ['dif2', 'hankel', 'random_spd', 'clustered', 'steinerberger']
    for pid in problems:
        A, b, x_exact = build_test_problem(pid, n, extra={'seed': 42})
        kappa, lmax, lmin = condition_number_estimate(A)
        print(f"  Problem {pid:15s} | cond(A) ≈ {kappa:.3e} | λ_max={lmax:.3e} | λ_min={lmin:.3e}")
    print()


def demo_sphere_quadrature():
    """演示 Lebedev 球面积分。"""
    print_header("Demo 8: Lebedev Spherical Quadrature")
    nodes, w = lebedev_rule_14()
    # 积分 f(x,y,z)=1 应得 4π
    val_const = integrate_on_sphere(lambda pts: np.ones(pts.shape[0]), rule='14')
    print(f"  ∫_{'{S^2}'} 1 dΩ ≈ {val_const:.6f} (exact: {4*math.pi:.6f})")

    # 积分 f(x,y,z)=x^2 应得 4π/3
    val_x2 = integrate_on_sphere(lambda pts: pts[:, 0] ** 2, rule='14')
    print(f"  ∫ x^2 dΩ ≈ {val_x2:.6f} (exact: {4*math.pi/3:.6f})")
    print()


def demo_genz_functions():
    """演示 Genz 多维测试函数。"""
    print_header("Demo 9: Genz Multi-dimensional Test Functions")
    m = 3
    c = np.ones(m) / m
    w = np.full(m, 0.5)
    rng = np.random.default_rng(42)
    pts = rng.random((m, 100000))
    for prob in range(1, 7):
        vals = genz_evaluate(prob, m, c, w, pts)
        mc_est = np.mean(vals)
        print(f"  Genz prob {prob} Monte Carlo estimate [0,1]^3: {mc_est:.6f}")
    print()


def demo_checksum_verification():
    """演示基于 ISBN 思想的校验和机制。"""
    print_header("Demo 10: Checksum Verification (ISBN-inspired)")
    n = 32
    A, b, x_exact = build_test_problem('dif2', n)
    x_comp, _ = conjugate_gradient(lambda v: A @ v, b, tol=1e-12)

    cs_exact = checksum_vector(x_exact)
    cs_comp = checksum_vector(x_comp)
    match = verify_checksum(x_comp, cs_exact)
    print(f"  Exact solution checksum: {cs_exact}")
    print(f"  Computed solution checksum: {cs_comp}")
    print(f"  Checksum match: {match}")

    rel_err = np.linalg.norm(x_comp - x_exact) / np.linalg.norm(x_exact)
    print(f"  Relative solution error: {rel_err:.3e}")
    print()


def demo_full_pipeline():
    """综合演示：从问题构造到求解、分析、验证的完整流程。"""
    print_header("Demo 11: Full Pipeline — Adaptive Preconditioned CG")
    nx, ny = 32, 32
    n = nx * ny
    print(f"  Problem: 2D anisotropic diffusion on {nx}x{ny} grid (N={n})")

    # 构造问题
    A = anisotropic_diffusion_2d(nx, ny, epsilon_x=1.0, epsilon_y=0.001)
    rng = np.random.default_rng(123)
    x_exact = rng.random(n)
    b = A @ x_exact

    kappa_est, lmax, lmin = condition_number_estimate(A)
    print(f"  Estimated condition number: {kappa_est:.3e}")
    print(f"  Theoretical CG iterations to 1e-8: {theoretical_cg_iteration_count(kappa_est, 1e-8)}")

    matvec = lambda v: A @ v

    # 标准 CG
    x_cg, info_cg = conjugate_gradient(matvec, b, tol=1e-8, max_iter=500)

    # 随机 SVD 估计谱
    U, lam = randomized_svd_approx(matvec, n, rank=20, power_iterations=2, seed=42)
    print(f"  Randomized SVD top-5 eigenvalues: {lam[:5]}")
    print(f"  Randomized SVD bottom-5 eigenvalues: {lam[-5:]}")

    # 谱预处理 PCG
    prec_spec = polynomial_spectral_preconditioner(A, poly_type='laguerre', n_nodes=12)
    x_pcg, info_pcg = preconditioned_cg(matvec, prec_spec, b, tol=1e-8, max_iter=500)

    # SSOR-PCG
    prec_ssor = ssor_preconditioner(A, omega=1.2)
    x_ssor, info_ssor = preconditioned_cg(matvec, prec_ssor, b, tol=1e-8, max_iter=500)

    results = {
        'CG': info_cg,
        'PCG-Spectral': info_pcg,
        'PCG-SSOR': info_ssor,
    }
    print(compare_solvers(results))

    # 解校验
    cs_exact = checksum_vector(x_exact)
    cs_pcg = checksum_vector(x_pcg)
    print(f"  Exact checksum: {cs_exact} | PCG checksum: {cs_pcg} | Match: {verify_checksum(x_pcg, cs_exact)}")
    print()


def main():
    """统一入口，依次运行所有演示。"""
    print("=" * 70)
    print("  Sparse Matrix Iterative Solvers & Preconditioning Suite")
    print("  Domain: Computational Mathematics — Sparse Linear Systems")
    print("=" * 70)
    print()

    demo_sparse_formats_cg()
    demo_orthogonal_polynomials()
    demo_halton_and_random_tools()
    demo_cvt_grid()
    demo_special_functions()
    demo_hankel_and_test_problems()
    demo_sphere_quadrature()
    demo_genz_functions()
    demo_pcg_preconditioners()
    demo_checksum_verification()
    demo_full_pipeline()

    print("=" * 70)
    print("  All demonstrations completed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
