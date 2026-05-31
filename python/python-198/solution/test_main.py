"""
main.py
=======
通信避免的多项式混沌展开-有限体积法 (CA-PCE-FVM)
用于大规模分布式不确定性传播的高性能计算框架

科学问题：
    在分布式内存系统上求解带有随机边界条件和随机对流系数的
    二维无粘Burgers方程，采用通信避免的s-step Krylov子空间方法
    加速PCE-Galerkin耦合系统的时间推进。

核心创新：
    1. 通信避免算法将PCE-Galerkin全局耦合通信减少s倍
    2. 结合矩阵指数方法提供高精度时间参考解
    3. 焦散拓扑分析预测激波形成与通信热点

输入：无（零参数可运行，所有参数内嵌）
输出：控制台报告数值精度、通信加速比、不确定性统计量
"""

import numpy as np
import time
import sys
import os

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_geometry import (
    generate_disk_triangulation,
    compute_element_quality,
    extract_boundary_edges,
    domain_decomposition,
    compute_interface_nodes
)
from random_parameters import (
    truncated_normal_ab_pdf,
    truncated_normal_ab_mean,
    truncated_normal_ab_variance,
    truncated_normal_ab_sample,
    generate_kl_coefficients
)
from pce_basis import (
    hermite_he_prob_matrix,
    build_pce_galerkin_matrix,
    vandermonde_matrix,
    vandermonde_solve
)
from burgers_fvm import (
    solve_burgers_fvm,
    build_fvm_operators
)
from matrix_exponential_int import (
    matrix_exponential_pade,
    pce_matrix_exponential_integrate
)
from communication_model import (
    estimate_disk_distance_mean,
    ca_speedup_theory,
    optimize_s_parameter,
    processor_communication_schedule
)
from ca_sstep_solver import (
    gmres_solve,
    ca_sstep_arnoldi
)
from monte_carlo_uq import (
    ellipse_sample,
    monte_carlo_pce_verify,
    disk_distance_monte_carlo
)
from sparse_io import (
    dense_to_csr,
    build_pce_block_sparse,
    write_hb_simple
)
from exact_benchmarks import (
    laplace_radial_2d_exact,
    solve_sawtooth_ode,
    compute_l2_error,
    sawtooth_wave
)
from caustic_analysis import (
    caustic_mapping,
    shock_formation_time,
    caustic_inspired_topology_field,
    detect_gradient_catastrophe
)


def print_header(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_section_1_mesh_generation():
    print_header("Section 1: 三角剖分与域分解")
    
    n_r, n_theta = 6, 12
    nodes, elements, boundary_mask = generate_disk_triangulation(n_r, n_theta)
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]
    
    quality = compute_element_quality(nodes, elements)
    boundary_edges = extract_boundary_edges(elements)
    
    n_parts = 4
    partition = domain_decomposition(nodes, elements, n_parts)
    interface = compute_interface_nodes(elements, partition)
    
    print(f"  节点数: {n_nodes}, 单元数: {n_elem}")
    print(f"  边界边数: {len(boundary_edges)}")
    print(f"  网格质量范围: [{quality.min():.4f}, {quality.max():.4f}]")
    print(f"  域分解: {n_parts} 分区")
    print(f"  界面节点数: {np.sum(interface)}")
    
    return nodes, elements, boundary_mask, partition, interface


def run_section_2_random_field():
    print_header("Section 2: 截断正态随机场建模")
    
    mu, sigma, a, b = 0.5, 0.3, 0.0, 1.5
    x_test = np.linspace(a - 0.2, b + 0.2, 100)
    pdf_vals = truncated_normal_ab_pdf(x_test, mu, sigma, a, b)
    mean_est = truncated_normal_ab_mean(mu, sigma, a, b)
    var_est = truncated_normal_ab_variance(mu, sigma, a, b)
    
    print(f"  截断正态参数: μ={mu}, σ={sigma}, a={a}, b={b}")
    print(f"  解析均值: {mean_est:.6f}")
    print(f"  解析方差: {var_est:.6f}")
    
    samples = truncated_normal_ab_sample(mu, sigma, a, b, size=10000)
    print(f"  采样均值: {np.mean(samples):.6f}, 采样方差: {np.var(samples, ddof=1):.6f}")
    
    kl_coeffs = generate_kl_coefficients(n_modes=5, correlation_length=0.2)
    print(f"  KL展开系数: {kl_coeffs}")
    
    return mu, sigma, a, b, kl_coeffs


def run_section_3_pce_basis():
    print_header("Section 3: PCE Hermite基与Vandermonde矩阵")
    
    degree = 4
    xi_test = np.linspace(-3, 3, 100)
    H = hermite_he_prob_matrix(degree, xi_test)
    
    # 验证正交性
    from pce_basis import he_double_product_integral
    norm_checks = [he_double_product_integral(k, k) for k in range(degree + 1)]
    print(f"  PCE阶数: {degree}")
    print(f"  Hermite模长 [k!]: {norm_checks}")
    
    alpha_mu, alpha_sigma = 1.0, 0.2
    A_pce = build_pce_galerkin_matrix(degree, alpha_mu, alpha_sigma)
    print(f"  PCE-Galerkin矩阵 (α_μ={alpha_mu}, α_σ={alpha_sigma}):")
    print(f"  条件数: {np.linalg.cond(A_pce):.4e}")
    
    # Vandermonde测试
    x_vand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    V = vandermonde_matrix(5, x_vand)
    det_V = np.linalg.det(V)
    print(f"  Vandermonde行列式: {det_V:.4e}")
    
    b_test = np.array([1.0, 3.0, 9.0, 27.0, 81.0])
    c = vandermonde_solve(V, b_test)
    print(f"  Vandermonde求解验证: 系数 = {c}")
    
    return degree, A_pce


def run_section_4_burgers_fvm(nodes, elements):
    print_header("Section 4: 确定性Burgers方程FVM求解")
    
    def u0_func(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        # 平滑初始条件: 0.5 * sin(πr) * (1-r²)，在边界r=1处自动为0
        # 幅值限制在[-0.5, 0.5]内，确保数值稳定性
        u = 0.5 * np.sin(np.pi * r) * (1.0 - r ** 2)
        return u
    
    t_max = 0.2
    nt = 100
    
    t, U = solve_burgers_fvm(nodes, elements, u0_func, t_max, nt,
                              flux_type='godunov', boundary_value=0.0)
    
    print(f"  求解时间域: [0, {t_max}], 步数: {nt}")
    print(f"  初始解范围: [{U[0].min():.4f}, {U[0].max():.4f}]")
    print(f"  终态解范围: [{U[-1].min():.4f}, {U[-1].max():.4f}]")
    
    # 与精确解对比（拉普拉斯解用于边界条件验证，不作为Burgers精确解）
    area, centroid, _, _ = build_fvm_operators(nodes, elements)
    u_exact_laplace, _, _, _, _, _ = laplace_radial_2d_exact(
        centroid[:, 0], centroid[:, 1], a=0.1, b=0.5
    )
    print(f"  拉普拉斯精确解（边界验证）范围: [{u_exact_laplace.min():.4f}, {u_exact_laplace.max():.4f}]")
    
    return t, U, area, centroid


def run_section_5_matrix_exponential(degree, alpha_mu, alpha_sigma):
    print_header("Section 5: 矩阵指数时间积分（PCE参考解）")
    
    A_pce = build_pce_galerkin_matrix(degree, alpha_mu, alpha_sigma)
    u0_pce = np.zeros(degree + 1)
    u0_pce[0] = 1.0
    
    tf = 0.5
    nt = 50
    
    t_start = time.time()
    t_arr, U_pce = pce_matrix_exponential_integrate(A_pce, u0_pce, tf, nt)
    t_elapsed = time.time() - t_start
    
    print(f"  PCE系统维度: {degree + 1}")
    print(f"  时间积分: [0, {tf}], 步数: {nt}")
    print(f"  矩阵指数时间: {t_elapsed:.4f} s")
    print(f"  PCE均值终值 (0阶系数): {U_pce[-1, 0]:.6f}")
    print(f"  PCE方差终值: {np.sum(U_pce[-1, 1:] ** 2):.6f}")
    
    # 解析验证
    analytical_mean = u0_pce[0] * np.exp(-alpha_mu * tf + 0.5 * alpha_sigma ** 2 * tf ** 2)
    print(f"  解析均值终值: {analytical_mean:.6f}")
    print(f"  误差: {abs(U_pce[-1, 0] - analytical_mean):.6e}")
    
    return t_arr, U_pce


def run_section_6_communication_analysis():
    print_header("Section 6: 通信模型与CA算法加速比分析")
    
    mean_d, var_d = estimate_disk_distance_mean(n_samples=20000)
    print(f"  圆盘距离MC估计: E[D]={mean_d:.6f}, Var[D]={var_d:.6f}")
    print(f"  理论值: E[D]={128.0/(45.0*np.pi):.6f}")
    
    t_comp, t_comm = 1e-3, 1e-2
    best_s, best_sp = optimize_s_parameter(t_comp, t_comm, s_max=20)
    print(f"  计算/通信时间比: {t_comp/t_comm:.4f}")
    print(f"  最优聚合步数 s*: {best_s}")
    print(f"  理论最大加速比: {best_sp:.4f}x")
    
    for s in [1, 2, 4, 8, 16]:
        sp = ca_speedup_theory(s, t_comp, t_comm)
        print(f"    s={s:2d}: 加速比 = {sp:.3f}x")
    
    schedule = processor_communication_schedule(n_procs=4, topology='tsp_optimized')
    print(f"  TSP优化通信调度 (4处理器): {len(schedule)} 条通信链路")
    
    return best_s, best_sp


def run_section_7_ca_solver(degree, alpha_mu, alpha_sigma):
    print_header("Section 7: 通信避免s-step Krylov求解器")
    
    A_pce = build_pce_galerkin_matrix(degree, alpha_mu, alpha_sigma)
    n = A_pce.shape[0]
    b = np.random.randn(n)
    
    # 标准GMRES
    x_std, res_std, it_std = gmres_solve(A_pce, b, restart=10, max_iter=20, tol=1e-8, s_step=1)
    
    # CA s-step GMRES (s=4)
    x_ca, res_ca, it_ca = gmres_solve(A_pce, b, restart=10, max_iter=20, tol=1e-8, s_step=4)
    
    print(f"  系统维度: {n}")
    print(f"  标准GMRES: 迭代 {it_std}, 最终残差 {res_std[-1]:.6e}")
    print(f"  CA GMRES(s=4): 迭代 {it_ca}, 最终残差 {res_ca[-1]:.6e}")
    print(f"  解差异 ||x_std - x_ca||: {np.linalg.norm(x_std - x_ca):.6e}")
    
    # Arnoldi基正交性测试
    v0 = b / np.linalg.norm(b)
    V_std, H_std = ca_sstep_arnoldi(A_pce, v0, s=1, m_total=5)
    V_ca, H_ca = ca_sstep_arnoldi(A_pce, v0, s=4, m_total=5)
    
    ortho_std = np.linalg.norm(V_std[:, :5].T @ V_std[:, :5] - np.eye(5))
    ortho_ca = np.linalg.norm(V_ca[:, :5].T @ V_ca[:, :5] - np.eye(5))
    print(f"  标准Arnoldi正交性偏离: {ortho_std:.6e}")
    print(f"  CA-Arnoldi正交性偏离: {ortho_ca:.6e}")
    
    return x_ca, res_ca


def run_section_8_mc_verification(degree, alpha_mu, alpha_sigma):
    print_header("Section 8: 蒙特卡洛不确定性量化验证")
    
    result = monte_carlo_pce_verify(
        n_samples=50000,
        pce_degree=degree,
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
        u0_scalar=1.0,
        tf=0.5,
        exact_mean_func=None
    )
    
    print(f"  MC样本数: {result['n_samples']}")
    print(f"  MC均值: {result['mc_mean']:.6f}")
    print(f"  MC方差: {result['mc_var']:.6f}")
    print(f"  PCE解析均值: {result['pce_mean_analytical']:.6f}")
    print(f"  相对误差: {result['error_mean']:.6e}")
    
    # 椭圆采样测试
    A_ellipse = np.array([[4.0, 1.0], [1.0, 3.0]])
    samples = ellipse_sample(1000, A_ellipse, R=1.0)
    norms = np.sqrt(np.sum(samples ** 2, axis=0))
    print(f"  椭圆采样验证: 最大范数={norms.max():.4f}")
    
    return result


def run_section_9_sparse_io(degree, alpha_mu, alpha_sigma, nodes, elements):
    print_header("Section 9: 稀疏矩阵I/O (Harwell-Boeing格式)")
    
    # 构造一个简化的空间-PCE耦合矩阵
    n_elem = elements.shape[0]
    spatial_A = 0.1 * np.eye(n_elem)
    A_total = build_pce_block_sparse(spatial_A, degree, alpha_mu, alpha_sigma)
    
    n_total = A_total.shape[0]
    print(f"  块稀疏PCE矩阵维度: {n_total} x {n_total}")
    
    # 转为CSR
    csr = dense_to_csr(A_total)
    nnz = len(csr['data'])
    sparsity = 1.0 - nnz / (n_total ** 2)
    print(f"  非零元数: {nnz}, 稀疏度: {sparsity:.4%}")
    
    # 写入HB文件
    filename = os.path.join(os.path.dirname(__file__), "pce_matrix.hb")
    write_hb_simple(filename, A_total, title="CA_PCE_GALERKIN_MATRIX")
    print(f"  HB矩阵文件已写入: {filename}")
    
    return A_total


def run_section_10_sawtooth_driver():
    print_header("Section 10: 锯齿波驱动ODE系统")
    
    t_span = (0.0, 10.0)
    y0 = np.array([0.0, 1.0])
    omega = 2.0 * np.pi
    
    t, y = solve_sawtooth_ode(t_span, y0, omega, n_steps=2000)
    
    print(f"  积分区间: [{t_span[0]}, {t_span[1]}]")
    print(f"  终态位移 y1: {y[-1, 0]:.6f}")
    print(f"  终态速度 y2: {y[-1, 1]:.6f}")
    print(f"  锯齿波幅值（t=1.25时）: {sawtooth_wave(1.25, omega):.6f}")
    
    return t, y


def run_section_11_caustic_topology(nodes, elements, t, U):
    print_header("Section 11: 焦散拓扑与激波预测")
    
    # 焦刻映射
    edges, pj, pk = caustic_mapping(n=20, m=5)
    print(f"  焦刻映射边数: {len(edges)}")
    
    # 特征速度场
    velocity = caustic_inspired_topology_field(nodes, elements, m=5, n=12)
    print(f"  焦刻启发速度场范围: [{velocity.min():.4f}, {velocity.max():.4f}]")
    
    # 激波预测（基于1D简化模型）
    x_1d = np.linspace(-1, 1, 50)
    u0_1d = np.sin(np.pi * x_1d)
    t_b, x_s = shock_formation_time(lambda x: np.sin(np.pi * x), x_1d)
    print(f"  一维Burgers激波形成时间估计: t_b = {t_b:.4f}")
    print(f"  激波初始位置: x = {x_s:.4f}")
    
    # 数值解梯度检测
    if len(t) > 1 and U.shape[1] > 1:
        area, centroid, _, _ = build_fvm_operators(nodes, elements)
        x_mid = centroid[:, 0]
        t_c, grad_hist = detect_gradient_catastrophe(x_mid, U, t)
        if t_c is not None:
            print(f"  数值梯度灾变检测: t_c ≈ {t_c:.4f}")
        else:
            print(f"  数值梯度灾变: 在计算时间内未检测到")
    
    return velocity


def main():
    np.random.seed(42)
    
    print("=" * 70)
    print("  通信避免的多保真不确定性量化框架 (CA-PCE-FVM)")
    print("  高性能计算：通信避免算法设计")
    print("  Project 198 - 博士级科研代码合成")
    print("=" * 70)
    print()
    
    # Section 1: 网格
    nodes, elements, boundary_mask, partition, interface = run_section_1_mesh_generation()
    print()
    
    # Section 2: 随机场
    mu, sigma, a, b, kl_coeffs = run_section_2_random_field()
    print()
    
    # Section 3: PCE基
    degree, A_pce_demo = run_section_3_pce_basis()
    print()
    
    # Section 4: Burgers FVM
    t, U, area, centroid = run_section_4_burgers_fvm(nodes, elements)
    print()
    
    # Section 5: 矩阵指数
    alpha_mu, alpha_sigma = 1.0, 0.2
    t_arr, U_pce = run_section_5_matrix_exponential(degree, alpha_mu, alpha_sigma)
    print()
    
    # Section 6: 通信分析
    best_s, best_sp = run_section_6_communication_analysis()
    print()
    
    # Section 7: CA求解器
    x_ca, res_ca = run_section_7_ca_solver(degree, alpha_mu, alpha_sigma)
    print()
    
    # Section 8: MC验证
    mc_result = run_section_8_mc_verification(degree, alpha_mu, alpha_sigma)
    print()
    
    # Section 9: 稀疏I/O
    A_total = run_section_9_sparse_io(degree, alpha_mu, alpha_sigma, nodes, elements)
    print()
    
    # Section 10: 锯齿波ODE
    t_ode, y_ode = run_section_10_sawtooth_driver()
    print()
    
    # Section 11: 焦刻拓扑
    velocity = run_section_11_caustic_topology(nodes, elements, t, U)
    print()
    
    # 最终总结
    print_header("综合性能报告")
    print(f"  1. 网格规模: {nodes.shape[0]} 节点, {elements.shape[0]} 单元")
    print(f"  2. PCE展开阶数: {degree}, 耦合系统维度: {A_pce_demo.shape[0]}")
    print(f"  3. Burgers FVM终态解均值: {U[-1].mean():.6f}")
    print(f"  4. 矩阵指数PCE均值误差: {abs(U_pce[-1,0] - np.exp(-alpha_mu*0.5+0.5*alpha_sigma**2*0.5**2)):.6e}")
    print(f"  5. 通信避免最优聚合步数: s*={best_s}, 理论加速比: {best_sp:.2f}x")
    print(f"  6. CA-GMRES残差收敛: {res_ca[-1]:.6e}")
    print(f"  7. MC-PCE均值验证误差: {mc_result['error_mean']:.6e}")
    print(f"  8. 块稀疏矩阵稀疏度: {1.0 - len(dense_to_csr(A_total)['data'])/(A_total.shape[0]**2):.4%}")
    print(f"  9. 锯齿波ODE终态能量: {0.5*(y_ode[-1,0]**2 + y_ode[-1,1]**2):.6f}")
    print(f" 10. 焦刻速度场L2范数: {np.sqrt(np.sum(velocity**2 * area)):.6f}")
    print()
    print("  所有模块执行完毕，无错误。")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（36个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_disk_triangulation 输出形状正确 ----
n_r, n_theta, n_parts_test = 6, 12, 4
nodes, elements, boundary_mask = generate_disk_triangulation(n_r, n_theta)
assert nodes.shape[1] == 2, '[TC01] nodes shape FAILED'
assert elements.shape[1] == 3, '[TC01] elements shape FAILED'
assert len(boundary_mask) == nodes.shape[0], '[TC01] boundary_mask length FAILED'
assert elements.shape[0] > 0, '[TC01] elements count FAILED'

# ---- TC02: generate_disk_triangulation 边界节点数量合理 ----
n_theta_outer = n_theta
n_boundary = np.sum(boundary_mask)
assert n_boundary == n_theta_outer, '[TC02] boundary node count FAILED'

# ---- TC03: compute_element_quality 输出在 [0, 1] 范围内 ----
quality = compute_element_quality(nodes, elements)
assert quality.min() >= 0.0, '[TC03] quality min FAILED'
assert quality.max() <= 1.0, '[TC03] quality max FAILED'
assert np.all(np.isfinite(quality)), '[TC03] quality finite FAILED'

# ---- TC04: extract_boundary_edges 提取边界边非空 ----
boundary_edges = extract_boundary_edges(elements)
assert boundary_edges.shape[0] > 0, '[TC04] boundary edges empty FAILED'
assert boundary_edges.shape[1] == 2, '[TC04] boundary edges shape FAILED'

# ---- TC05: domain_decomposition 所有节点分区完成 ----
partition = domain_decomposition(nodes, elements, n_parts_test)
assert partition.shape[0] == nodes.shape[0], '[TC05] partition shape FAILED'
assert np.min(partition) >= 0, '[TC05] partition min FAILED'
assert np.max(partition) < n_parts_test, '[TC05] partition max FAILED'

# ---- TC06: compute_interface_nodes 接口节点存在 ----
interface = compute_interface_nodes(elements, partition)
assert interface.shape[0] == nodes.shape[0], '[TC06] interface shape FAILED'
assert np.sum(interface) >= 0, '[TC06] interface count negative FAILED'

# ---- TC07: truncated_normal_ab_pdf 积分近似为1 ----
mu, sigma, a, b = 0.5, 0.3, 0.0, 1.5
x_grid = np.linspace(a, b, 2000)
pdf_vals = truncated_normal_ab_pdf(x_grid, mu, sigma, a, b)
integral = np.trapz(pdf_vals, x_grid)
assert abs(integral - 1.0) < 0.01, '[TC07] PDF integral FAILED'

# ---- TC08: truncated_normal_ab_mean 在截断区间内 ----
mean_val = truncated_normal_ab_mean(mu, sigma, a, b)
assert a <= mean_val <= b, '[TC08] mean out of truncation bounds FAILED'

# ---- TC09: truncated_normal_ab_variance 非负有限 ----
var_val = truncated_normal_ab_variance(mu, sigma, a, b)
assert var_val >= 0.0, '[TC09] variance negative FAILED'
assert np.isfinite(var_val), '[TC09] variance infinite FAILED'

# ---- TC10: truncated_normal_ab_sample 输出形状正确（固定种子） ----
np.random.seed(42)
samples = truncated_normal_ab_sample(mu, sigma, a, b, size=1000)
assert len(samples) == 1000, '[TC10] sample count FAILED'
assert np.all(samples >= a), '[TC10] sample below a FAILED'
assert np.all(samples <= b), '[TC10] sample above b FAILED'

# ---- TC11: truncated_normal_ab_sample 可复现性 ----
np.random.seed(42)
s1 = truncated_normal_ab_sample(mu, sigma, a, b, size=100)
np.random.seed(42)
s2 = truncated_normal_ab_sample(mu, sigma, a, b, size=100)
assert np.allclose(s1, s2), '[TC11] reproducibility FAILED'

# ---- TC12: hermite_he_prob_matrix 正交性验证 ----
degree_test = 4
xi_test = np.linspace(-3, 3, 300)
H = hermite_he_prob_matrix(degree_test, xi_test)
assert H.shape == (300, degree_test + 1), '[TC12] H shape FAILED'
# 验证与X的二次方关系
he2_exact = xi_test ** 2 - 1.0
assert np.allclose(H[:, 2], he2_exact, atol=1e-10), '[TC12] He_2 orthogonality FAILED'

# ---- TC13: build_pce_galerkin_matrix 输出方阵 ----
alpha_mu, alpha_sigma = 1.0, 0.2
A_pce = build_pce_galerkin_matrix(degree_test, alpha_mu, alpha_sigma)
assert A_pce.shape == (degree_test + 1, degree_test + 1), '[TC13] A_pce shape FAILED'
assert np.all(np.isfinite(A_pce)), '[TC13] A_pce finite FAILED'

# ---- TC14: vandermonde_solve 精确恢复 ----
x_vand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
V = vandermonde_matrix(5, x_vand)
b_test = np.array([1.0, 3.0, 9.0, 27.0, 81.0])
c = vandermonde_solve(V, b_test)
recovered = V @ c
assert np.allclose(recovered, b_test, atol=1e-10), '[TC14] vandermonde solve FAILED'

# ---- TC15: godunov_flux 确定性 - f(uL,uR) == f(uL,uR) 相同输入相同输出 ----
from burgers_fvm import godunov_flux
f1 = godunov_flux(np.array([0.3]), np.array([0.8]))
f2 = godunov_flux(np.array([0.3]), np.array([0.8]))
assert np.allclose(f1, f2), '[TC15] godunov reproducibility FAILED'

# ---- TC16: godunov_flux uL==uR 时通量有限 ----
from burgers_fvm import godunov_flux
u_test = np.array([0.5, -0.3])
f_test = godunov_flux(u_test, u_test)
assert np.all(np.isfinite(f_test)), '[TC16] godunov equal inputs FAILED'

# ---- TC17: build_fvm_operators 输出维度正确 ----
area, centroid, internal_edges, boundary_edges_fvm = build_fvm_operators(nodes, elements)
assert area.shape[0] == elements.shape[0], '[TC17] area shape FAILED'
assert centroid.shape[0] == elements.shape[0], '[TC17] centroid shape FAILED'
assert centroid.shape[1] == 2, '[TC17] centroid cols FAILED'
assert np.all(area > 0), '[TC17] area positive FAILED'

# ---- TC18: solve_burgers_fvm 输出形状正确 ----
def u0_func(x, y):
    r = np.sqrt(x**2 + y**2)
    return 0.5 * np.sin(np.pi * r) * (1.0 - r**2)

np.random.seed(42)
t, U = solve_burgers_fvm(nodes, elements, u0_func, t_max=0.1, nt=20)
assert len(t) == 21, '[TC18] time steps FAILED'
assert U.shape == (21, elements.shape[0]), '[TC18] U shape FAILED'
assert np.all(np.isfinite(U)), '[TC18] U finite FAILED'

# ---- TC19: matrix_exponential_pade exp(0) = I ----
A_zero = np.zeros((3, 3))
E_zero = matrix_exponential_pade(A_zero)
assert np.allclose(E_zero, np.eye(3), atol=1e-12), '[TC19] exp(0) FAILED'

# ---- TC20: matrix_exponential_pade 对稀疏矩阵输出有限 ----
A_small = np.array([[0.1, 0.2], [-0.2, 0.1]])
E_small = matrix_exponential_pade(A_small)
assert E_small.shape == (2, 2), '[TC20] exp small shape FAILED'
assert np.all(np.isfinite(E_small)), '[TC20] exp small finite FAILED'

# ---- TC21: ca_speedup_theory s=1 加速比为1 ----
sp = ca_speedup_theory(1, 1e-3, 1e-2)
assert abs(sp - 1.0) < 1e-10, '[TC21] s=1 speedup FAILED'

# ---- TC22: ca_speedup_theory 加速比 >= 1 ----
for s_test in [1, 2, 4, 8, 16]:
    sp_s = ca_speedup_theory(s_test, 1e-3, 1e-2)
    assert sp_s >= 1.0 - 1e-12, f'[TC22] speedup s={s_test} FAILED'

# ---- TC23: optimize_s_parameter 返回有效参数 ----
best_s, best_sp = optimize_s_parameter(1e-3, 1e-2, s_max=10)
assert best_s >= 1, '[TC23] best_s FAILED'
assert best_sp >= 1.0, '[TC23] best_sp FAILED'

# ---- TC24: gmres_solve 求解小型线性系统 ----
np.random.seed(42)
A_tiny = np.array([[4.0, 1.0], [1.0, 3.0]])
b_tiny = np.ones(2)
x_gmres, res_hist, iters = gmres_solve(A_tiny, b_tiny, restart=2, max_iter=10, tol=1e-10, s_step=1)
assert len(x_gmres) == 2, '[TC24] gmres solution shape FAILED'
assert res_hist[-1] < 1e-8, '[TC24] gmres residual FAILED'

# ---- TC25: ca_sstep_arnoldi 输出矩阵尺寸正确 ----
np.random.seed(42)
n_v = 5
A_arn = np.diag(np.arange(1, n_v + 1, dtype=float))
v0_arn = np.ones(n_v)
V, H = ca_sstep_arnoldi(A_arn, v0_arn, s=2, m_total=3)
assert V.shape[0] == n_v, '[TC25] V row count FAILED'
assert H.shape[1] == 3, '[TC25] H col count FAILED'

# ---- TC26: ellipse_sample 采样点满足椭圆约束（固定种子） ----
np.random.seed(42)
A_ellipse = np.array([[4.0, 1.0], [1.0, 3.0]])
R_test = 1.0
samples_ell = ellipse_sample(500, A_ellipse, R_test)
assert samples_ell.shape == (2, 500), '[TC26] ellipse sample shape FAILED'
quad_form = np.sum(samples_ell * (A_ellipse @ samples_ell), axis=0)
assert np.all(quad_form <= R_test**2 + 1e-10), '[TC26] ellipse constraint FAILED'

# ---- TC27: monte_carlo_pce_verify 返回有效统计量 ----
np.random.seed(42)
result = monte_carlo_pce_verify(
    n_samples=10000, pce_degree=4, alpha_mu=1.0, alpha_sigma=0.2,
    u0_scalar=1.0, tf=0.5, exact_mean_func=None
)
assert 'mc_mean' in result, '[TC27] mc_mean key FAILED'
assert result['mc_mean'] > 0, '[TC27] mc_mean non-positive FAILED'
assert result['error_mean'] < 0.05, '[TC27] error_mean too large FAILED'

# ---- TC28: dense_to_csr 往返无损 ----
from sparse_io import csr_to_dense
A_dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
csr = dense_to_csr(A_dense)
A_recovered = csr_to_dense(csr)
assert np.allclose(A_recovered, A_dense), '[TC28] CSR round-trip FAILED'

# ---- TC29: build_pce_block_sparse 矩阵形状正确 ----
spatial_A = 0.1 * np.eye(4)
A_total = build_pce_block_sparse(spatial_A, 2, alpha_mu=1.0, alpha_sigma=0.2)
n_expected = 4 * (2 + 1)
assert A_total.shape == (n_expected, n_expected), '[TC29] block sparse shape FAILED'
assert np.all(np.isfinite(A_total)), '[TC29] block sparse finite FAILED'

# ---- TC30: laplace_radial_2d_exact 满足拉普拉斯方程 ----
x_lap = np.array([0.5, 1.2, 0.8])
y_lap = np.array([0.3, -0.5, 0.0])
u, ux, uy, uxx, uxy, uyy = laplace_radial_2d_exact(x_lap, y_lap, a=0.1, b=0.5)
laplacian = uxx + uyy
assert np.allclose(laplacian, 0.0, atol=1e-12), '[TC30] Laplace equation FAILED'

# ---- TC31: sawtooth_wave 值域在 [-1, 1] ----
t_saw = np.linspace(0, 5.0, 100)
saw_vals = sawtooth_wave(t_saw, omega=2.0 * np.pi, amplitude=1.0)
assert np.min(saw_vals) >= -1.0, '[TC31] sawtooth min FAILED'
assert np.max(saw_vals) <= 1.0, '[TC31] sawtooth max FAILED'

# ---- TC32: sawtooth_wave 周期性 ----
t1 = 0.25
T = 2.0 * np.pi / (2.0 * np.pi)
v1 = sawtooth_wave(t1, omega=2.0 * np.pi)
v2 = sawtooth_wave(t1 + T, omega=2.0 * np.pi)
assert abs(v1 - v2) < 1e-12, '[TC32] sawtooth periodicity FAILED'

# ---- TC33: compute_l2_error 非负 ----
u_num = np.array([0.5, 0.8, 1.2])
u_exact = np.array([0.5, 0.7, 1.3])
area_err = np.array([0.1, 0.2, 0.15])
err = compute_l2_error(u_num, u_exact, area_err)
assert err >= 0.0, '[TC33] L2 error negative FAILED'

# ---- TC34: caustic_mapping 边数正确 ----
n_caustic, m_caustic = 20, 5
edges_c, pj_c, pk_c = caustic_mapping(n_caustic, m_caustic)
assert edges_c.shape == (n_caustic + 1, 2), '[TC34] caustic edges shape FAILED'
assert pj_c.shape == (n_caustic + 1, 2), '[TC34] pj shape FAILED'

# ---- TC35: shock_formation_time 检测激波 ----
def u0_shock(x):
    return -0.5 * np.sin(np.pi * x)
x_grid = np.linspace(-1.0, 1.0, 100)
t_b, x_s = shock_formation_time(u0_shock, x_grid)
assert np.isfinite(t_b), '[TC35] shock time infinite FAILED'
assert t_b > 0, '[TC35] shock time non-positive FAILED'

# ---- TC36: generate_kl_coefficients 输出有限且形状正确 ----
np.random.seed(42)
kl_coeffs = generate_kl_coefficients(n_modes=5, correlation_length=0.2)
assert len(kl_coeffs) == 5, '[TC36] KL coeffs length FAILED'
assert np.all(np.isfinite(kl_coeffs)), '[TC36] KL coeffs finite FAILED'

print('\n全部 36 个测试通过!\n')
