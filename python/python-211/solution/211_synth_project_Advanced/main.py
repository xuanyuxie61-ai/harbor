"""
main.py
=======
PROJECT_211 —— 多尺度无约束非线性优化: 量子-经典混合能量景观全局寻优

统一入口, 零参数可运行.

本合成项目融合 15 个种子项目的核心算法, 围绕"数学优化: 无约束非线性优化"领域,
构造一个前沿博士级科学计算问题:

  多尺度势能面全局优化:
  给定由量子光学态密度、关联随机场、Chebyshev 代理模型共同定义的
  复杂多模态目标函数 V(x), 使用一系列高级优化方法
  (BFGS, L-BFGS, 延拓法, 梯度流, 模拟退火, 代理优化)
  找到全局极小点.

项目结构 (12 个 .py 文件):
  - special_functions.py: 特殊函数库 (Bernoulli, Chebyshev, Beta, Lerch, AGM, ...)
  - mesh_basis.py: 有限元网格与重心基函数
  - correlation_field.py: 相关函数与随机场生成
  - linalg_solver.py: 高斯消元与线性系统求解
  - quasi_newton.py: BFGS, L-BFGS, Newton, Wolfe 线搜索
  - continuation.py: 延拓法与同伦全局优化
  - ode_optimizers.py: ODE 梯度流优化器 (隐式中点, Heavy-Ball, Nesterov)
  - quantum_objective.py: 量子光学目标函数 (光子态密度, 介电函数)
  - global_search.py: 全局搜索 (排列枚举, 拉丁超立方, 模拟退火)
  - chebyshev_surrogate.py: Chebyshev 代理模型优化
  - test_problems.py: 标准优化测试问题集
  - main.py: 本文件, 统一入口

种子项目映射:
  1. 1286_Shukti042_dreamrelation: 余弦相似度 → 特征匹配优化目标
  2. 548_human_mesh2d: 2D 三角网格 → 优化域离散化
  3. 210_continuation: 延拓法 Newton → 同伦路径跟踪
  4. 1242_ce335805_PhotonDosReference: 光子态密度 → 量子光学势
  5. 382_fem_to_xml: 网格序列化 → 优化域数据表示
  6. 220_correlation: 相关函数 → 随机场构造
  7. 163_chebyshev_series: Chebyshev 级数 → 代理模型
  8. 881_polpak: 特殊函数 → 数学基础库
  9. 572_ill_bvp: 病态问题 → 收敛诊断
  10. 764_midpoint: 隐式中点 → 辛优化积分器
  11. 1363_tsp_brute: 排列枚举 → 全局搜索
  12. 234_cube_integrals: 单位立方体积分 → Monte Carlo
  13. 337_eros: 高斯消元 → 线性代数核心
  14. 1434_zombie_ode: 守恒量 ODE → 种群动力学优化
  15. 371_fem_basis: 重心基函数 → 有限元基

运行方式:
  python main.py

作者: 自动合成
日期: 2026-06-07
"""

import sys
import os
import time
import warnings
import numpy as np

# 抑制 RuntimeWarning (如溢出, 数值噪声, 在优化中常见)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 确保当前目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 模块导入
# ===========================================================================
from special_functions import (
    chebyshev_T, chebyshev_U, chebyshev_series_eval, chebyshev_coefficients,
    chebyshev_series_deriv, bernoulli_numbers, beta_function, agm,
    elliptic_K, elliptic_E, factorial, log_factorial, double_factorial,
    lerch_transcendent, pochhammer, hermite_phys, moebius_mu, sigma_divisors
)
from mesh_basis import (
    generate_boundary_points, delaunay_triangulation_2d,
    fem_basis_1d, fem_basis_2d, fem_basis_3d, fem_basis_md,
    mesh_to_dict, mesh_to_json, gauss_triangle, refine_mesh_uniform,
    barycentric_coords_2d
)
from correlation_field import (
    correlation_spherical, correlation_linear, correlation_exponential,
    correlation_gaussian, correlation_matern, correlation_power, correlation_bessel,
    sample_paths_cholesky, sample_paths_fft, CorrelatedRandomField
)
from linalg_solver import (
    GaussianElimination, plu_decomposition, condition_number,
    conjugate_gradient, cholesky_factorization, hessian_modification,
    symmetric_eigenvalues
)
from quasi_newton import (
    armijo_backtrack, wolfe_line_search, bfgs, lbfgs,
    gradient_descent, newton_method, cosine_similarity_loss,
    check_convergence, estimate_convergence_rate
)
from continuation import (
    compute_tangent, continuation_step, homotopy_optimization,
    pseudo_arclength_continuation, newton_corrector
)
from ode_optimizers import (
    implicit_midpoint, explicit_midpoint, rk4,
    heavy_ball_optimization, nesterov_accelerated,
    ode_optimizer, PopulationDynamics
)
from quantum_objective import (
    DielectricModel, photon_dos_bulk, photon_dos_surface,
    QuantumOpticalObjective, rosenbrock, rosenbrock_grad, rosenbrock_hessian,
    rastrigin, rastrigin_grad, ackley, ackley_grad,
    cube01_monomial_integral, monte_carlo_integral
)
from global_search import (
    latin_hypercube_sample, multi_start_optimization,
    simulated_annealing, random_search_with_refinement,
    grid_search_coarse_to_fine, brute_force_tsp, permutation_search_optimizer
)
from chebyshev_surrogate import (
    ChebyshevSurrogate, chebyshev_surrogate_optimization,
    chebyshev_adaptive_degree, chebyshev_error_estimate
)
from test_problems import (
    get_test_problems, run_test_suite,
    SphereProblem, RosenbrockProblem, TridProblem,
    HimmelblauProblem, BealeProblem
)


# ===========================================================================
# 输出工具
# ===========================================================================

def print_header(title: str):
    """打印格式化标题."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str):
    """打印子标题."""
    print(f"\n--- {title} ---")


def print_result(label: str, value, fmt: str = ".6e"):
    """打印结果."""
    if isinstance(value, float):
        print(f"  {label}: {value:{fmt}}")
    elif isinstance(value, np.ndarray):
        print(f"  {label}: [{', '.join(f'{v:.4f}' for v in value[:5])}{'...' if len(value) > 5 else ''}]")
    else:
        print(f"  {label}: {value}")


# ===========================================================================
# 测试模块 1: 特殊函数库验证
# ===========================================================================

def demo_special_functions():
    """演示特殊函数的正确性."""
    print_header("模块 1: 特殊函数库验证")

    # Chebyshev 多项式
    print_subheader("Chebyshev 多项式 T_n(cos θ) = cos(nθ)")
    theta = np.pi / 4
    for n in range(6):
        T_val = chebyshev_T(n, np.cos(theta))
        T_exact = np.cos(n * theta)
        print(f"  T_{n}(cos(π/4)) = {T_val:.10f}, cos({n}π/4) = {T_exact:.10f}, err = {abs(T_val - T_exact):.2e}")

    # Chebyshev 级数: 展开 exp(x)
    print_subheader("Chebyshev 级数展开 exp(x) on [-1,1]")
    coef = chebyshev_coefficients(lambda x: np.exp(x), -1, 1, 20)
    x_test = 0.5
    approx = chebyshev_series_eval(x_test, coef)
    exact = np.exp(x_test)
    print(f"  exp(0.5) 精确 = {exact:.10f}")
    print(f"  exp(0.5) Chebyshev 近似 = {approx:.10f}")
    print(f"  误差 = {abs(approx - exact):.2e}")

    # 导数验证
    fx, dfx = chebyshev_series_deriv(x_test, coef)
    print(f"  exp'(0.5) 精确 = {exact:.10f}")
    print(f"  exp'(0.5) Chebyshev 导数 = {dfx:.10f}")
    print(f"  导数误差 = {abs(dfx - exact):.2e}")

    # Bernoulli 数
    print_subheader("Bernoulli 数 B_0..B_10")
    B = bernoulli_numbers(10)
    exact_B = [1, -0.5, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    for k in range(11):
        print(f"  B_{k} = {B[k]:.8f} (精确: {exact_B[k]:.8f})")

    # Beta 函数
    print_subheader("Beta 函数 B(a,b) = Γ(a)Γ(b)/Γ(a+b)")
    print(f"  B(2,3) = {beta_function(2,3):.8f} (精确: {1/12:.8f})")
    print(f"  B(0.5,0.5) = {beta_function(0.5,0.5):.8f} (精确: π = {np.pi:.8f})")

    # AGM 与椭圆积分
    print_subheader("AGM 与完全椭圆积分")
    k = 0.5
    K_val = elliptic_K(k)
    E_val = elliptic_E(k)
    print(f"  K(0.5) = {K_val:.10f}")
    print(f"  E(0.5) = {E_val:.10f}")
    print(f"  AGM(1, √(1-0.25)) = {agm(1.0, np.sqrt(1-0.25)):.10f}")

    # Lerch 超越函数
    print_subheader("Lerch 超越函数 Φ(z,s,a)")
    z, s, a = 0.5, 2.0, 1.0
    phi = lerch_transcendent(z, s, a, n_terms=100)
    # Φ(0.5, 2, 1) = Σ 0.5^n / (n+1)² ≈ 0.5572...
    print(f"  Φ(0.5, 2, 1) = {phi.real:.8f}")


# ===========================================================================
# 模块 2: 网格与基函数
# ===========================================================================

def demo_mesh_basis():
    """演示网格生成与基函数."""
    print_header("模块 2: 有限元网格与基函数")

    # 边界点生成
    print_subheader("2D 边界点生成 (人体轮廓)")
    boundary_pts = generate_boundary_points(50, shape="human_like")
    print(f"  生成 {len(boundary_pts)} 个边界点")
    print(f"  x 范围: [{boundary_pts[:,0].min():.4f}, {boundary_pts[:,0].max():.4f}]")
    print(f"  y 范围: [{boundary_pts[:,1].min():.4f}, {boundary_pts[:,1].max():.4f}]")

    # Delaunay 三角剖分
    print_subheader("Delaunay 三角剖分")
    # 添加内部点
    rng = np.random.default_rng(42)
    internal_pts = rng.uniform(-0.3, 0.3, (20, 2))
    all_pts = np.vstack([boundary_pts, internal_pts])
    triangles = delaunay_triangulation_2d(all_pts)
    print(f"  总顶点: {len(all_pts)}, 三角形: {len(triangles)}")

    # 网格序列化
    mesh_data = mesh_to_dict(all_pts, triangles)
    json_str = mesh_to_json(mesh_data)
    print(f"  网格 JSON 大小: {len(json_str)} 字节")

    # 基函数验证
    print_subheader("2D 三角形基函数 (D=2, 6 节点)")
    # 参考三角形 (0,0)-(1,0)-(0,1)
    # D=2 时有 6 个基函数
    basis_indices = [(2,0,0), (1,1,0), (0,2,0), (1,0,1), (0,1,1), (0,0,2)]
    test_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
    for idx, (i, j, k) in enumerate(basis_indices):
        vals = [fem_basis_2d(i, j, k, p[0], p[1]) for p in test_points]
        print(f"  L_{i}{j}{k}: 节点值 = [{', '.join(f'{v:.4f}' for v in vals)}]")

    # 重心坐标
    print_subheader("重心坐标计算")
    v1, v2, v3 = np.array([0, 0.0]), np.array([1, 0.0]), np.array([0, 1.0])
    p = np.array([0.25, 0.25])
    lam = barycentric_coords_2d(p, v1, v2, v3)
    print(f"  点 (0.25, 0.25) 的重心坐标: λ = ({lam[0]:.4f}, {lam[1]:.4f}, {lam[2]:.4f})")
    print(f"  和: {sum(lam):.4f}")

    # Gauss 求积
    print_subheader("三角形 Gauss 求积")
    for deg in [1, 2, 4, 5]:
        pts, wts = gauss_triangle(deg)
        print(f"  degree={deg}: {len(pts)} 点, 权重和 = {sum(wts):.6f} (精确: 0.5)")

    # 网格细化
    print_subheader("均匀网格细化")
    vertices = np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    for level in range(3):
        v_ref, e_ref = refine_mesh_uniform(vertices, elements, n_refine=1)
        print(f"  细化 {level+1} 次: 顶点 {len(v_ref)}, 单元 {len(e_ref)}")
        vertices, elements = v_ref, e_ref


# ===========================================================================
# 模块 3: 相关函数与随机场
# ===========================================================================

def demo_correlation_field():
    """演示相关函数与随机场采样."""
    print_header("模块 3: 相关函数与高斯随机场")

    n = 50
    rho_max = 5.0
    rho0 = 1.0
    rho = np.linspace(0, rho_max, n)

    correlations = {
        "球面": correlation_spherical,
        "线性": correlation_linear,
        "指数": correlation_exponential,
        "高斯": correlation_gaussian,
        "幂律": lambda r, r0: correlation_power(r, r0, 2.0),
        "Bessel": correlation_bessel,
    }

    print_subheader("相关函数在原点附近的值")
    for name, func in correlations.items():
        c_vals = func(rho[:5], rho0)
        print(f"  {name}: C(0..4Δ) = [{', '.join(f'{v:.4f}' for v in c_vals)}]")

    # Cholesky 采样
    print_subheader("Cholesky 采样 (5 条路径, 高斯相关)")
    rng = np.random.default_rng(42)
    rho_vec, X = sample_paths_cholesky(n, 5, rho_max, rho0, correlation_gaussian, rng)
    print(f"  路径形状: {X.shape}")
    print(f"  各路径均值: [{', '.join(f'{np.mean(X[:,i]):.4f}' for i in range(5))}]")
    print(f"  各路径标准差: [{', '.join(f'{np.std(X[:,i]):.4f}' for i in range(5))}]")

    # FFT 采样
    print_subheader("FFT 采样 (5 条路径)")
    rho_vec2, X2 = sample_paths_fft(n, 5, rho_max, rho0, correlation_gaussian, rng)
    print(f"  路径形状: {X2.shape}")
    print(f"  各路径均值: [{', '.join(f'{np.mean(X2[:,i]):.4f}' for i in range(5))}]")

    # 随机场对象
    print_subheader("CorrelatedRandomField 对象")
    field = CorrelatedRandomField(dim=3, n_centers=20, rho0=0.5, rng=rng)
    test_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    for pt in test_pts:
        val = field.evaluate(pt)
        print(f"  f({pt}) = {val:.6f}")


# ===========================================================================
# 模块 4: 线性代数
# ===========================================================================

def demo_linalg():
    """演示线性代数核心."""
    print_header("模块 4: 高斯消元与线性系统")

    # 小规模系统求解
    print_subheader("Gauss 消元: 3×3 系统")
    A = np.array([[4, 1, -1], [2, 7, 1], [1, -3, 12.0]])
    b = np.array([3, 19, 31.0])
    ge = GaussianElimination(A, b)
    x = ge.solve()
    print(f"  A = {A.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  x = {x.flatten().tolist()}")
    print(f"  残差 ||Ax-b|| = {np.linalg.norm(A @ x.flatten() - b):.2e}")

    # 行列式
    print_subheader("行列式计算")
    ge2 = GaussianElimination(A)
    det_val = ge2.determinant()
    det_np = np.linalg.det(A)
    print(f"  高斯消元 det(A) = {det_val:.8f}")
    print(f"  NumPy det(A) = {det_np:.8f}")
    print(f"  误差 = {abs(det_val - det_np):.2e}")

    # 逆矩阵
    print_subheader("逆矩阵")
    ge3 = GaussianElimination(A)
    A_inv = ge3.inverse()
    I_check = A @ A_inv
    print(f"  ||A·A⁻¹ - I|| = {np.linalg.norm(I_check - np.eye(3)):.2e}")

    # PLU 分解
    print_subheader("PLU 分解")
    P, L, U, swaps = plu_decomposition(A)
    plu_err = np.linalg.norm(P @ A - L @ U)
    print(f"  ||PA - LU|| = {plu_err:.2e}")
    print(f"  交换次数 = {swaps}")

    # 条件数
    print_subheader("条件数分析")
    kappa = condition_number(A)
    kappa2 = np.linalg.cond(A)
    print(f"  κ_∞(A) = {kappa:.4f}")
    print(f"  NumPy κ(A) = {kappa2:.4f}")

    # 病态系统
    print_subheader("病态 Hilbert 矩阵")
    n = 8
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    kappa_H = condition_number(H)
    print(f"  {n}×{n} Hilbert 矩阵条件数 = {kappa_H:.4e}")
    print(f"  (Hilbert 矩阵是典型病态矩阵)")

    # 共轭梯度法
    print_subheader("共轭梯度法 (SPD 系统)")
    n = 20
    A_spd = np.random.default_rng(42).standard_normal((n, n))
    A_spd = A_spd.T @ A_spd + n * np.eye(n)  # SPD
    b_spd = np.ones(n)
    x_cg, iters, res = conjugate_gradient(A_spd, b_spd, tol=1e-10)
    print(f"  {n}×{n} SPD 系统, CG 迭代 {iters} 次, 残差 = {res:.2e}")

    # Cholesky
    print_subheader("Cholesky 分解")
    L_chol = cholesky_factorization(A_spd)
    if L_chol is not None:
        chol_err = np.linalg.norm(L_chol @ L_chol.T - A_spd)
        print(f"  ||LL^T - A|| = {chol_err:.2e}")


# ===========================================================================
# 模块 5: 标准测试问题基准
# ===========================================================================

def demo_benchmark():
    """在标准测试问题上运行 BFGS."""
    print_header("模块 5: 标准测试问题基准 (BFGS)")

    results = run_test_suite(bfgs, tol=1e-10)

    print(f"\n{'问题':<15} {'f(x*)':>12} {'f_opt':>12} {'误差':>12} {'迭代':>6} {'收敛':>6}")
    print("-" * 70)
    for name, res in results.items():
        if "f_val" in res:
            print(f"{name:<15} {res['f_val']:>12.6e} {res['f_opt']:>12.6e} "
                  f"{res['error']:>12.2e} {res['iterations']:>6} "
                  f"{'是' if res['converged'] else '否':>6}")
        else:
            print(f"{name:<15} {'错误':>12} {res.get('error', 'unknown'):>30}")


# ===========================================================================
# 模块 6: L-BFGS 与梯度下降比较
# ===========================================================================

def demo_optimizer_comparison():
    """比较不同优化器在 Rosenbrock 上的性能."""
    print_header("模块 6: 优化器性能比较 (Rosenbrock 5D)")

    dim = 5
    x0 = np.ones(dim) * (-1.0)

    optimizers = {
        "梯度下降": lambda f, g, x: gradient_descent(f, g, x, max_iter=5000),
        "BFGS": lambda f, g, x: bfgs(f, g, x, max_iter=500),
        "L-BFGS(m=10)": lambda f, g, x: lbfgs(f, g, x, m=10, max_iter=500),
        "Heavy-Ball": lambda f, g, x: heavy_ball_optimization(f, g, x, lr=0.001, momentum=0.9),
        "Nesterov": lambda f, g, x: nesterov_accelerated(f, g, x, L=1000.0),
    }

    print(f"\n{'优化器':<18} {'f(x*)':>12} {'||∇f||':>12} {'迭代':>8} {'收敛':>6}")
    print("-" * 60)
    for name, opt_func in optimizers.items():
        t0 = time.time()
        res = opt_func(rosenbrock, rosenbrock_grad, x0.copy())
        dt = time.time() - t0
        print(f"{name:<18} {res['f_val']:>12.4e} {res['grad_norm']:>12.4e} "
              f"{res['iterations']:>8} {'是' if res['converged'] else '否':>6} ({dt:.3f}s)")


# ===========================================================================
# 模块 7: 延拓法与同伦优化
# ===========================================================================

def demo_continuation():
    """演示延拓法与同伦优化."""
    print_header("模块 7: 延拓法与同伦全局优化")

    # 同伦优化: Rosenbrock
    print_subheader("同伦法优化 Rosenbrock (5D)")

    # 简单起始函数: 二次
    def f_start(x):
        return np.sum((x - 0) ** 2)

    def grad_start(x):
        return 2.0 * x

    x0 = np.ones(5) * (-2.0)

    result = homotopy_optimization(
        rosenbrock, rosenbrock_grad,
        f_start, grad_start,
        x0, n_stages=10, tol=1e-8, optimizer=bfgs
    )

    print(f"  同伦阶段数: {len(result['path'])}")
    print(f"  最终 f(x) = {result['f_val']:.6e}")
    print(f"  最终 x = [{', '.join(f'{v:.4f}' for v in result['x'])}]")
    print(f"  收敛: {'是' if result['converged'] else '否'}")
    print(f"\n  同伦路径:")
    for stage in result['path']:
        print(f"    t={stage['t']:.2f}: f_target={stage['f_target']:.6e}, gnorm={stage['gnorm']:.2e}")


# ===========================================================================
# 模块 8: ODE 梯度流优化
# ===========================================================================

def demo_ode_optimizers():
    """演示 ODE 梯度流优化."""
    print_header("模块 8: ODE 梯度流优化")

    # Sphere 函数的梯度流
    print_subheader("Sphere 函数梯度流 (解析解)")

    def sphere_grad(x):
        return 2.0 * x

    x0 = np.array([3.0, 2.0, 1.0])

    # RK4 积分
    t_arr, y_arr = rk4(lambda t, x: -sphere_grad(x), (0, 5), x0, n_steps=200)
    print(f"  解析解: x(t) = x₀ exp(-2t)")
    print(f"  t=5 时:")
    print(f"    解析: x = {x0 * np.exp(-10)}")
    print(f"    RK4:  x = {y_arr[-1]}")
    print(f"    误差: {np.linalg.norm(y_arr[-1] - x0 * np.exp(-10)):.2e}")

    # 隐式中点 (辛积分)
    t_arr2, y_arr2 = implicit_midpoint(lambda t, x: -sphere_grad(x), (0, 5), x0, n_steps=200)
    print(f"    隐式中点: x = {y_arr2[-1]}")
    print(f"    误差: {np.linalg.norm(y_arr2[-1] - x0 * np.exp(-10)):.2e}")

    # Rosenbrock 梯度流
    print_subheader("Rosenbrock 梯度流 (5D)")
    x0_rb = np.ones(5) * (-0.5)
    result = ode_optimizer(rosenbrock, rosenbrock_grad, x0_rb, t_final=20.0, n_steps=2000)
    print(f"  终态 f(x) = {result['f_val']:.6e}")
    print(f"  终态 ||∇f|| = {result['grad_norm']:.6e}")
    print(f"  能量单调递减: {np.all(np.diff(result['energy']) <= 1e-10)}")

    # 守恒量 (Hamilton)
    print_subheader("守恒量监控")
    energy = result['energy']
    abs_err, rel_err = conservation_error(energy)
    print(f"  能量最大变化: {abs(energy[-1] - energy[0]):.6e}")
    print(f"  (梯度流中能量应单调递减)")


def conservation_error(H):
    H0 = H[0]
    abs_err = np.max(np.abs(H - H0))
    rel_err = abs_err / max(abs(H0), 1e-300)
    return float(abs_err), float(rel_err)


# ===========================================================================
# 模块 9: 量子光学目标优化
# ===========================================================================

def demo_quantum_objective():
    """演示量子光学目标函数优化."""
    print_header("模块 9: 量子光学目标函数")

    # 介电模型
    print_subheader("STO 介电函数")
    eps = DielectricModel()
    w_SPhP = eps.surface_mode_frequency()
    print(f"  ω_TO = {eps.wTO/1e12:.2f} THz")
    print(f"  ω_LO = {eps.wLO/1e12:.2f} THz")
    print(f"  ω_SPhP = {w_SPhP/1e12:.2f} THz")

    # 量子目标函数优化
    print_subheader("量子光学目标函数 (5D)")
    obj = QuantumOpticalObjective(dim=5, seed=42)
    x0 = np.zeros(5)

    # BFGS 优化
    res = bfgs(obj.evaluate, obj.gradient, x0, tol=1e-8, max_iter=300)
    print(f"  BFGS: f(x*) = {res['f_val']:.6e}, ||∇f|| = {res['grad_norm']:.2e}")
    print(f"  x* = [{', '.join(f'{v:.4f}' for v in res['x'])}]")

    # Monte Carlo 积分 (单位立方体上的期望)
    print_subheader("Monte Carlo 积分验证")
    e = np.array([2, 1, 0, 3, 1])
    exact = cube01_monomial_integral(e)
    est, se = monte_carlo_integral(
        lambda x: np.prod([x[i] ** e[i] for i in range(5) if e[i] > 0]),
        dim=5, n_samples=50000
    )
    print(f"  单项式指数 e = {e}")
    print(f"  精确积分 = {exact:.8f}")
    print(f"  MC 估计 = {est:.8f} ± {se:.2e}")


# ===========================================================================
# 模块 10: 全局搜索
# ===========================================================================

def demo_global_search():
    """演示全局搜索方法."""
    print_header("模块 10: 全局搜索方法")

    # Rastrigin (2D) — 多极小
    print_subheader("Rastrigin 2D (多极小, 全局极小 = 0)")

    def rastrigin_2d(x):
        return 20 + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    def rastrigin_2d_grad(x):
        return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

    # 多起始点
    print("\n  多起始点 BFGS (30 个起点):")
    ms_result = multi_start_optimization(
        rastrigin_2d, rastrigin_2d_grad, dim=2,
        n_starts=30, bounds=(-3, 3), local_optimizer=bfgs
    )
    print(f"    最优 f = {ms_result['best_f_val']:.6e}")
    print(f"    最优 x = {ms_result['best_x']}")
    print(f"    成功率 = {ms_result['success_rate']:.0%}")

    # 模拟退火
    print("\n  模拟退火:")
    x0 = np.array([2.5, 2.5])
    sa_result = simulated_annealing(rastrigin_2d, x0, T_init=50.0, cooling_rate=0.95)
    print(f"    f = {sa_result['f_val']:.6e}")
    print(f"    x = {sa_result['x']}")

    # 粗到细网格搜索
    print("\n  粗到细网格搜索:")
    gs_result = grid_search_coarse_to_fine(rastrigin_2d, dim=2, bounds=(-3, 3),
                                           n_coarse=15, n_levels=4)
    print(f"    f = {gs_result['f_val']:.6e}")
    print(f"    x = {gs_result['x']}")

    # 暴力 TSP (小规模)
    print_subheader("暴力 TSP (5 城市)")
    rng = np.random.default_rng(42)
    cities = rng.uniform(0, 100, (5, 2))
    dist_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])
    min_cost, best_perm = brute_force_tsp(dist_matrix)
    print(f"  5 城市 TSP 最短路径代价 = {min_cost:.2f}")
    print(f"  最优排列 = {best_perm}")


# ===========================================================================
# 模块 11: Chebyshev 代理优化
# ===========================================================================

def demo_chebyshev_surrogate():
    """演示 Chebyshev 代理模型."""
    print_header("模块 11: Chebyshev 代理模型优化")

    # 一维代理
    print_subheader("一维 Chebyshev 代理: f(x) = x⁴ - 3x³ + 2 on [-2, 3]")
    f_test = lambda x: x ** 4 - 3 * x ** 3 + 2
    surrogate = ChebyshevSurrogate(f_test, -2, 3, n_terms=30)

    # 验证精度
    x_test_pts = np.linspace(-2, 3, 10)
    max_err = max(abs(surrogate.evaluate(x) - f_test(x)) for x in x_test_pts)
    print(f"  代理最大误差 (10 测试点) = {max_err:.2e}")

    # 找最小值
    x_min, f_min = surrogate.find_minimum()
    print(f"  代理预测极小点: x = {x_min:.6f}, f = {f_min:.6f}")

    # 与精确解比较 (f'(x) = 4x³ - 9x² = 0 => x=0 or x=9/4)
    x_exact = 9.0 / 4.0  # 局部极小
    f_exact = f_test(x_exact)
    print(f"  精确极小点: x = {x_exact:.6f}, f = {f_exact:.6f}")
    print(f"  误差: Δx = {abs(x_min - x_exact):.2e}, Δf = {abs(f_min - f_exact):.2e}")

    # 自适应阶数
    print_subheader("自适应 Chebyshev 阶数选择")
    coef, degree = chebyshev_adaptive_degree(f_test, -2, 3, tol=1e-12)
    print(f"  达到 1e-12 精度所需阶数: {degree}")
    err = chebyshev_error_estimate(coef)
    print(f"  截断误差估计: {err:.2e}")

    # 代理优化 (减少昂贵评估)
    print_subheader("自适应代理优化")
    n_calls = 0
    def expensive_f(x):
        nonlocal n_calls
        n_calls += 1
        return f_test(x)

    res = chebyshev_surrogate_optimization(expensive_f, -2, 3, n_initial=15, n_iterations=30)
    print(f"  代理优化结果: x = {res['x']:.6f}, f = {res['f_val']:.6f}")
    print(f"  总评估次数: {res['n_evaluations']}")
    print(f"  直接网格搜索需 ~1000 次")


# ===========================================================================
# 模块 12: 种群动力学优化
# ===========================================================================

def demo_population_dynamics():
    """演示种群动力学参数优化."""
    print_header("模块 12: SZR 种群动力学参数优化")

    # 模拟 SZR 模型 (使用温和的参数)
    print_subheader("SZR 模型模拟")
    dyn = PopulationDynamics(alpha=0.1, beta=0.005, gamma=0.05, delta=0.005)
    y0 = np.array([500.0, 10.0, 0.0])  # S, Z, R
    t_arr, y_arr = dyn.simulate(y0, t_final=50.0, n_steps=10000)
    print(f"  初始: S={y0[0]:.0f}, Z={y0[1]:.0f}, R={y0[2]:.0f}")
    if not np.any(np.isnan(y_arr)):
        print(f"  终态: S={y_arr[-1,0]:.2f}, Z={y_arr[-1,1]:.2f}, R={y_arr[-1,2]:.2f}")
        print(f"  守恒量: 初始 = {dyn.conserved(y0):.2f}, 终态 = {dyn.conserved(y_arr[-1]):.2f}")
        print(f"  守恒误差 = {abs(dyn.conserved(y_arr[-1]) - dyn.conserved(y0)):.2e}")
    else:
        print(f"  模拟发散 (刚性系统), 使用更小步长或隐式方法")

    # 参数优化 (用 BFGS)
    print_subheader("参数优化 (最小化终态 Zombies)")
    params0 = np.array([0.1, 0.005, 0.05, 0.005])

    # 数值梯度
    def grad_params(params):
        eps = 1e-3
        g = np.zeros(4)
        f0 = dyn.objective(params)
        for i in range(4):
            p_plus = params.copy()
            p_plus[i] += eps
            g[i] = (dyn.objective(p_plus) - f0) / eps
        return g

    res = bfgs(dyn.objective, grad_params, params0, tol=1e-4, max_iter=20)
    print(f"  初始参数: α={params0[0]:.4f}, β={params0[1]:.4f}, γ={params0[2]:.4f}, δ={params0[3]:.4f}")
    if not np.any(np.isnan(res['x'])):
        print(f"  优化参数: α={res['x'][0]:.4f}, β={res['x'][1]:.4f}, γ={res['x'][2]:.4f}, δ={res['x'][3]:.4f}")
        print(f"  目标值: {res['f_val']:.4f}")
    print(f"  迭代: {res['iterations']}, 收敛: {'是' if res['converged'] else '否'}")


# ===========================================================================
# 模块 13: 余弦相似度特征匹配优化
# ===========================================================================

def demo_similarity_optimization():
    """演示余弦相似度优化."""
    print_header("模块 13: 余弦相似度特征匹配优化")

    # 问题: 找向量 a 使 cos(a, b_target) 最大, 同时 ||a|| 受约束
    print_subheader("特征匹配: min 1 - cos(a, b)")

    rng = np.random.default_rng(42)
    dim = 10
    b_target = rng.standard_normal(dim)
    b_target /= np.linalg.norm(b_target)

    # 目标: cosine_similarity_loss(a, b_target)
    def f_match(a):
        return cosine_similarity_loss(a, b_target) + 0.01 * np.sum(a ** 2)

    def grad_match(a):
        from quasi_newton import cosine_similarity_loss_grad
        return cosine_similarity_loss_grad(a, b_target) + 0.02 * a

    # BFGS 优化
    a0 = rng.standard_normal(dim)
    res = bfgs(f_match, grad_match, a0, tol=1e-10, max_iter=200)

    cos_sim = np.dot(res['x'], b_target) / (np.linalg.norm(res['x']) * np.linalg.norm(b_target))
    print(f"  目标维度: {dim}")
    print(f"  优化后余弦相似度: {cos_sim:.8f}")
    print(f"  损失值: {res['f_val']:.6e}")
    print(f"  收敛: {'是' if res['converged'] else '否'}, 迭代: {res['iterations']}")


# ===========================================================================
# 模块 14: 收敛分析与病态问题
# ===========================================================================

def demo_convergence_analysis():
    """演示收敛分析与病态问题处理."""
    print_header("模块 14: 收敛分析与病态问题")

    # 收敛阶估计
    print_subheader("收敛阶估计")
    # 生成 BFGS 在 Rosenbrock 上的历史
    x0 = np.ones(5) * (-1.0)
    res = bfgs(rosenbrock, rosenbrock_grad, x0, tol=1e-12, max_iter=200)
    rate = estimate_convergence_rate(res['history'])
    print(f"  BFGS on Rosenbrock 5D:")
    print(f"    收敛率 (最后 5 步平均) = {rate:.4f}")
    if rate < 0.1:
        print(f"    判定: 超线性收敛 (BFGS 理论保证)")
    elif rate < 0.5:
        print(f"    判定: 较快线性收敛")
    else:
        print(f"    判定: 较慢收敛")

    # 病态问题
    print_subheader("病态问题: 条件数对收敛的影响")
    for kappa_target in [10, 100, 1000, 10000]:
        # 构造条件数为 kappa_target 的二次函数
        n = 10
        eigvals = np.logspace(0, np.log10(kappa_target), n)
        Q = np.diag(eigvals)
        def f_ill(x, Q=Q):
            return 0.5 * x @ Q @ x
        def g_ill(x, Q=Q):
            return Q @ x

        x0 = np.ones(n)
        t0 = time.time()
        res = bfgs(f_ill, g_ill, x0, tol=1e-10, max_iter=2000)
        dt = time.time() - t0
        print(f"  κ = {kappa_target:>6}: 迭代 {res['iterations']:>5}, "
              f"f = {res['f_val']:.2e}, 时间 {dt:.3f}s")


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """PROJECT_211 统一入口.
    多尺度无约束非线性优化: 量子-经典混合能量景观全局寻优.
    """
    print("=" * 72)
    print("  PROJECT_211: 多尺度无约束非线性优化")
    print("  量子-经典混合能量景观全局寻优")
    print("  博士级科学计算项目 (15 种子项目融合)")
    print("=" * 72)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t_start = time.time()

    # 运行所有演示模块
    demos = [
        demo_special_functions,
        demo_mesh_basis,
        demo_correlation_field,
        demo_linalg,
        demo_benchmark,
        demo_optimizer_comparison,
        demo_continuation,
        demo_ode_optimizers,
        demo_quantum_objective,
        demo_global_search,
        demo_chebyshev_surrogate,
        demo_population_dynamics,
        demo_similarity_optimization,
        demo_convergence_analysis,
    ]

    success_count = 0
    fail_count = 0
    for demo_func in demos:
        try:
            demo_func()
            success_count += 1
        except Exception as e:
            print(f"\n[错误] {demo_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    t_total = time.time() - t_start

    print_header("运行总结")
    print(f"  总模块数: {len(demos)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总耗时: {t_total:.2f} 秒")
    print("\n  融合种子项目: 15 个")
    print("  核心优化方法: BFGS, L-BFGS, Newton, 梯度下降, 延拓法,")
    print("    ODE 梯度流, Heavy-Ball, Nesterov, 模拟退火, 代理优化")
    print("  应用科学领域: 量子光学, 种群动力学, 统计物理")
    print("=" * 72)


if __name__ == "__main__":
    main()
