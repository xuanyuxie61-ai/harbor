"""
Weyl 半金属 Berry Curvature 计算：高阶有限差分与稳定性分析
===============================================================

主入口文件

科学问题：
Weyl 半金属是一类具有非平庸拓扑性质的新型量子材料。其低能激发
由 Weyl 费米子描述，在动量空间中存在成对的 Weyl 节点，每个节点
具有确定的手性 (Chern number ±1)。

Berry curvature Ω(k) 是刻画拓扑性质的核心物理量，满足：
1. 在 Weyl 点附近呈现奇异性：Ω(k) ∝ ±1/(2|k - k_W|²)
2. 对整个 Brillouin 区的积分给出 Chern 数 (整数拓扑不变量)
3. 决定反常 Hall 电导等物理响应

本程序实现：
- Weyl Hamiltonian 的构建与对角化
- 多种高阶有限差分格式计算 Berry curvature
- Fukui-Hatsugai-Suzuki 离散方法计算 Chern 数
- 系统化的数值稳定性与收敛性分析
- 拓扑相变的统计分析

使用方法：
    python main.py

作者：PROJECT_272 合成项目
日期：2026-06
"""

import numpy as np
import sys
import time
from typing import Dict, List

# 导入自定义模块
from weyl_hamiltonian import WeylHamiltonian, find_weyl_nodes
from brillouin_mesh import BrillouinMesh
from berry_curvature import BerryCurvatureCalculator
from chern_number import ChernNumberCalculator
from finite_difference import HighOrderFDScheme, FDMatrixOperator, richardson_extrapolation
from stability_analysis import StabilityAnalyzer
from spectral_solver import SpectralSolver, golub_welsch_nodes_weights
from polynomial_basis import PolynomialBasisConverter
from ode_integrator import ODEIntegrator, compute_zygv_phase
from statistics_analyzer import StatisticalAnalyzer


def print_header():
    """打印程序头"""
    print("=" * 70)
    print("  Weyl 半金属 Berry Curvature 计算系统")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)
    print()


def section_1_hamiltonian_construction():
    """
    第 1 部分：Weyl Hamiltonian 构建

    物理背景：
    Weyl Hamiltonian H(k) = χ·v_F (k·σ) + m(k)·σ₀

    其中：
    - χ = ±1 为手性 (chirality)
    - v_F 为费米速度
    - σ 为 Pauli 矩阵矢量
    - m(k) 为质量项 (时间反演对称性破缺)
    """
    print("\n" + "=" * 70)
    print("第 1 部分：Weyl Hamiltonian 构建与能带结构")
    print("=" * 70)

    # 创建两个手性相反的 Weyl 点
    ham_positive = WeylHamiltonian(v_f=1.0, chirality=+1,
                                    weyl_node=np.array([0.3, 0.0, 0.0]))
    ham_negative = WeylHamiltonian(v_f=1.0, chirality=-1,
                                    weyl_node=np.array([-0.3, 0.0, 0.0]))

    # 在 Γ 点计算 Hamiltonian
    k_gamma = np.array([0.0, 0.0, 0.0])
    H_gamma = ham_positive.hamiltonian_matrix(k_gamma, m0=0.1, alpha=0.5)

    print(f"\n[Hamiltonian] Γ 点 Hamiltonian 矩阵:")
    print(f"  H(Γ) =")
    print(f"  [[{H_gamma[0,0].real:+.4f} {H_gamma[0,1].real:+.4f} + {H_gamma[0,1].imag:+.4f}i],")
    print(f"   [{H_gamma[1,0].real:+.4f} {H_gamma[1,1].real:+.4f} + {H_gamma[1,1].imag:+.4f}i]]")

    # 计算能量本征值
    eigenvalues = np.linalg.eigvalsh(H_gamma)
    gap = np.abs(eigenvalues[1] - eigenvalues[0])
    print(f"\n[Hamiltonian] Γ 点能隙: Δ = {gap:.4f}")
    print(f"  本征值: E_± = {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")

    # 沿高对称路径计算能带
    mesh = BrillouinMesh()
    k_path, labels = mesh.high_symmetry_path()

    print(f"\n[Hamiltonian] 高对称路径: {' → '.join(labels)}")
    print(f"  路径点数: {len(k_path)}")

    # 沿路径计算能量色散
    energies_plus = []
    energies_minus = []
    for k in k_path[::10]:  # 采样
        E_plus, E_minus = ham_positive.energy_dispersion(k, 0.1, 0.5)
        energies_plus.append(E_plus)
        energies_minus.append(E_minus)

    print(f"  导带能量范围: [{min(energies_plus):.4f}, {max(energies_plus):.4f}]")
    print(f"  价带能量范围: [{min(energies_minus):.4f}, {max(energies_minus):.4f}]")

    # 速度算符
    v_x = ham_positive.velocity_operator(k_gamma, 'x')
    print(f"\n[Hamiltonian] 速度算符 v_x:")
    print(f"  v_x = v_F · σ_x = [[0, {v_x[0,1].real:.4f}], [{v_x[1,0].real:.4f}, 0]]")

    return ham_positive, ham_negative, mesh


def section_2_brillouin_mesh(ham_positive, mesh):
    """
    第 2 部分：Brillouin 区网格生成

    实现：
    1. 均匀矩形网格 (用于有限差分)
    2. 三角剖分网格 (用于 FHS 方法)
    3. 自适应细化
    """
    print("\n" + "=" * 70)
    print("第 2 部分：Brillouin 区网格生成")
    print("=" * 70)

    # 均匀网格
    n_grid = 10
    k_points, mesh_info = mesh.generate_uniform_mesh(n_grid, n_grid, n_grid)
    print(f"\n[Mesh] 均匀网格:")
    print(f"  网格密度: {n_grid} × {n_grid} × {n_grid}")
    print(f"  总点数: {mesh_info['total_points']}")
    print(f"  微分体积: {mesh_info['differential_volume']:.6f}")

    # 三角网格 (2D 切片)
    vertices, triangles = mesh.generate_triangular_mesh_2d('kz_const', 0.0, 15, 15)
    print(f"\n[Mesh] 三角剖分网格 (kz=0 切片):")
    print(f"  顶点数: {len(vertices)}")
    print(f"  三角形数: {len(triangles)}")

    # 网格质量
    quality = mesh.compute_mesh_quality(vertices, triangles)
    print(f"  最小距离: {quality['min_distance']:.6f}")
    print(f"  最小面积: {quality['min_area']:.6f}")

    # 自适应细化
    weyl_position = np.array([0.3, 0.0, 0.0])
    k_adaptive, adaptive_info = mesh.adaptive_refine_around_weyl(
        weyl_position, radius=0.3, base_n=5, refine_factor=3
    )
    print(f"\n[Mesh] 自适应细化:")
    print(f"  Weyl 点位置: {weyl_position}")
    print(f"  细化半径: 0.3")
    print(f"  总点数: {adaptive_info['total_points']} (粗: {adaptive_info['n_coarse']}, 细: {adaptive_info['n_fine']})")

    return k_points


def section_3_berry_curvature(ham_positive, mesh, k_points):
    """
    第 3 部分：Berry Curvature 计算

    核心公式：
    Ω_n^{μν}(k) = -2·Im Σ_{m≠n} <u_n|v_μ|u_m><u_m|v_ν|u_n> / (E_m - E_n)²

    数值方法：
    1. Kubo 公式 (连续极限)
    2. 高阶有限差分 (O(h²), O(h⁴))
    """
    print("\n" + "=" * 70)
    print("第 3 部分：Berry Curvature 计算")
    print("=" * 70)

    berry_calc = BerryCurvatureCalculator(ham_positive, mesh)

    # 在几个测试点计算 Berry curvature
    test_points = [
        np.array([0.1, 0.1, 0.1]),
        np.array([0.2, 0.0, 0.0]),
        np.array([0.3, 0.0, 0.0]),  # 接近 Weyl 点
        np.array([0.0, 0.0, 0.0]),
    ]

    print(f"\n[Berry] 使用 Kubo 公式计算 Berry curvature:")
    print(f"  {'k 点':20s} {'|Ω|':>12s} {'Ω_xy':>12s} {'Ω_yz':>12s} {'Ω_zx':>12s}")
    print(f"  {'-'*70}")

    for k in test_points:
        omega = berry_calc.berry_curvature_kubo(k, band_idx=0, eta=1e-3)
        omega_norm = np.linalg.norm(omega)
        print(f"  [{k[0]:.2f},{k[1]:.2f},{k[2]:.2f}]   {omega_norm:12.6f} "
              f"{omega[0]:12.6f} {omega[1]:12.6f} {omega[2]:12.6f}")

    # 高阶有限差分
    print(f"\n[Berry] 高阶有限差分方法:")
    k_test = np.array([0.15, 0.1, 0.1])
    for order in [2, 4]:
        omega_fd = berry_calc.berry_curvature_high_order_fd(k_test, 0, order, dk=0.01)
        print(f"  O(h^{order}) 方法: Ω = [{omega_fd[0]:.6f}, {omega_fd[1]:.6f}, {omega_fd[2]:.6f}]")

    # Berry connection 与 Wilson loop
    print(f"\n[Berry] Berry connection 与 Wilson loop:")
    dk_test = np.array([0.01, 0.0, 0.0])
    A_dk = berry_calc.berry_connection(k_test, dk_test, 0)
    print(f"  A·dk = {A_dk:.6f}")

    # Wilson loop 沿小回路
    k_loop = np.array([
        [0.1, 0.1, 0.1],
        [0.15, 0.1, 0.1],
        [0.15, 0.15, 0.1],
        [0.1, 0.15, 0.1],
        [0.1, 0.1, 0.1],
    ])
    berry_phase = berry_calc.wilson_loop(k_loop, 0)
    print(f"  Wilson loop Berry phase: γ = {berry_phase[0]:.6f}")

    # Berry curvature 场 (小网格)
    print(f"\n[Berry] 计算 Berry curvature 场 (10×10×10 网格)...")
    k_small, _ = mesh.generate_uniform_mesh(5, 5, 5)
    omega_field = berry_calc.compute_berry_curvature_field(k_small, 0, method='kubo')
    print(f"  总计算点数: {len(k_small)}")
    print(f"  |Ω| 范围: [{np.min(np.linalg.norm(omega_field, axis=1)):.6f}, "
          f"{np.max(np.linalg.norm(omega_field, axis=1)):.6f}]")

    return berry_calc, omega_field


def section_4_chern_number(berry_calc, mesh):
    """
    第 4 部分：Chern 数计算

    Chern 数 C = (1/2π) ∫∫_{BZ} Ω(k) d²k

    方法：
    1. 直接积分
    2. FHS 离散方法 (保证整数化)
    3. Chern 数随 kz 的变化
    """
    print("\n" + "=" * 70)
    print("第 4 部分：Chern 数计算与拓扑分类")
    print("=" * 70)

    chern_calc = ChernNumberCalculator(berry_calc)

    # FHS 方法计算 Chern 数
    print(f"\n[Chern] FHS 方法计算 kz=0 切片的 Chern 数:")
    result = chern_calc.fhs_method('kz_const', 0.0, 0, n_grid=15)
    print(f"  Chern 数 C = {result['chern_number']}")
    print(f"  总 Berry flux = {result['total_flux']:.6f}")

    # Chern 数随 kz 的变化
    print(f"\n[Chern] Chern 数随 kz 的变化 (搜索 Weyl 点投影):")
    kz_values = np.linspace(0, 0.5, 10)
    kz_vals, chern_nums = chern_calc.chern_number_vs_kz(kz_values, 0, 10, 'fhs')
    print(f"  {'kz':>8s} {'C':>6s}")
    for kz, c in zip(kz_vals, chern_nums):
        print(f"  {kz:8.4f} {c:6d}")

    # 搜索 Chern 跳变点
    transitions = chern_calc.find_chern_transitions((0, 0.5), 0, 20, 10)
    print(f"\n[Chern] 发现 {len(transitions)} 个 Chern 跳变点:")
    for i, trans in enumerate(transitions[:3]):  # 最多显示 3 个
        print(f"  跳变 {i+1}: kz ∈ [{trans['kz_left']:.4f}, {trans['kz_right']:.4f}], "
              f"ΔC = {trans['delta_chern']}")

    return chern_calc


def section_5_finite_difference_analysis():
    """
    第 5 部分：有限差分格式分析

    分析不同阶数差分格式的精度和稳定性
    """
    print("\n" + "=" * 70)
    print("第 5 部分：高阶有限差分格式分析")
    print("=" * 70)

    # 差分系数展示
    print(f"\n[FD] 各阶差分格式系数:")
    for order in [2, 4, 6, 8]:
        scheme = HighOrderFDScheme(order)
        print(f"  O(h^{order}): stencil = {scheme.scheme['stencil']}")
        weights_str = ', '.join([f'{w:+.4f}' for w in scheme.scheme['weights']])
        print(f"          weights = [{weights_str}]")

    # 测试一维导数
    print(f"\n[FD] 一维函数导数测试:")
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)
    f = np.sin(x)  # f(x) = sin(x), f'(x) = cos(x)
    df_exact = np.cos(x)

    for order in [2, 4, 6]:
        scheme = HighOrderFDScheme(order)
        df_fd = scheme.derivative_1d(f, 2*np.pi/100)
        error = np.max(np.abs(df_fd - df_exact))
        print(f"  O(h^{order}) 方法: 最大误差 = {error:.6e}")

    # 截断误差估计
    print(f"\n[FD] 截断误差分析:")
    def test_func(k):
        return np.sin(k[0]) * np.cos(k[1]) * np.exp(-k[2]**2)

    k_test = np.array([0.5, 0.3, 0.2])
    scheme4 = HighOrderFDScheme(4)

    error_est = scheme4.truncation_error_estimate(test_func, k_test, 0, 0.1)
    print(f"  四阶差分截断误差估计: {error_est:.6e}")

    # Richardson 外推
    print(f"\n[FD] Richardson 外推:")
    values = [1.01, 1.001, 1.0001]
    ratios = [2.0]
    extrap = richardson_extrapolation(values, ratios, order=2)
    print(f"  原始值: {values}")
    print(f"  外推值: {extrap:.6f}")

    # 有限差分矩阵算子
    print(f"\n[FD] 有限差分矩阵算子:")
    fd_op = FDMatrixOperator(n_grid=20)
    k_mat = fd_op.momentum_operator(0)
    print(f"  动量算符维度: {k_mat.shape}")
    print(f"  动量算符 Hermiticity 检查: ||K - K^†|| = {np.linalg.norm(k_mat - k_mat.conj().T):.6e}")

    return fd_op


def section_6_stability_analysis(berry_calc, mesh):
    """
    第 6 部分：数值稳定性分析

    分析维度：
    1. 网格收敛性
    2. 条件数
    3. 规范依赖性
    4. 奇异性接近
    """
    print("\n" + "=" * 70)
    print("第 6 部分：数值稳定性分析")
    print("=" * 70)

    analyzer = StabilityAnalyzer(berry_calc)

    k_test = np.array([0.1, 0.1, 0.1])
    weyl_pos = np.array([0.3, 0.0, 0.0])

    # 收敛性分析
    print(f"\n[Stability] 收敛性分析 (k = {k_test}):")
    conv_results = analyzer.convergence_analysis(k_test, 0)
    for order in [2, 4]:
        if f'order_{order}' in conv_results:
            final_val = conv_results[f'order_{order}']['final_value']
            final_norm = np.linalg.norm(final_val)
            print(f"  O(h^{order}) 方法最终值: |Ω| = {final_norm:.6f}")

    # 条件数分析
    print(f"\n[Stability] 条件数分析:")
    cond_results = analyzer.condition_number_analysis(k_test)
    print(f"  最大条件数: {cond_results['max_condition']:.4e}")
    print(f"  最小能隙: {cond_results['min_gap']:.6f}")

    # 规范依赖性
    print(f"\n[Stability] 规范依赖性测试:")
    gauge_results = analyzer.gauge_dependence_test(k_test, 0)
    print(f"  最大规范偏差: {gauge_results['max_deviation']:.6e}")
    print(f"  平均规范偏差: {gauge_results['mean_deviation']:.6e}")

    # 奇异性测试
    print(f"\n[Stability] 奇异性接近测试 (Weyl 点 = {weyl_pos}):")
    sing_results = analyzer.singularity_proximity_test(weyl_pos, 0)
    print(f"  幂律指数 α = {sing_results['power_law_exponent']:.4f} (理论值 2.0)")

    # 综合报告
    print(f"\n[Stability] 综合稳定性评估:")
    report = analyzer.stability_report(k_test, weyl_pos, 0)
    stability = report['overall_stability']
    print(f"  综合评分: {stability['overall_score']:.2f}/10")
    print(f"  等级: {stability['grade']}")
    for key, score in stability['individual_scores'].items():
        print(f"    {key}: {score:.2f}")

    return report


def section_7_spectral_methods():
    """
    第 7 部分：谱方法

    实现多种本征值求解算法
    """
    print("\n" + "=" * 70)
    print("第 7 部分：谱方法求解器")
    print("=" * 70)

    # 测试矩阵
    n = 10
    np.random.seed(42)
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (A + A.conj().T) / 2  # Hermitian

    # 精确解
    e_exact, _ = np.linalg.eigh(H)

    # 不同方法
    print(f"\n[Spectral] 本征值求解器对比:")
    print(f"  矩阵维度: {n}×{n}")
    print(f"  {'方法':20s} {'误差':>12s}")
    print(f"  {'-'*35}")

    for method in ['qr', 'divide_conquer', 'lanczos', 'golub_welsch']:
        solver = SpectralSolver(method)
        e_calc, _ = solver.solve_eigenproblem(H)
        error = np.max(np.abs(e_calc[:n] - e_exact))
        print(f"  {method:20s} {error:12.4e}")

    # Golub-Welsch 节点权重
    print(f"\n[Spectral] Golub-Welsch 高斯求积:")
    for weight_func in ['hermite', 'legendre', 'chebyshev']:
        nodes, weights = golub_welsch_nodes_weights(5, weight_func)
        print(f"  {weight_func:12s}: nodes = [{nodes[0]:.4f}, ..., {nodes[-1]:.4f}]")
        print(f"                weights sum = {np.sum(weights):.4f}")

    # 态密度
    solver = SpectralSolver('qr')
    energies = e_exact
    E_grid, dos = solver.compute_dos(energies, (-3, 3), 50, 0.1)
    print(f"\n[Spectral] 态密度计算:")
    print(f"  能量范围: [{E_grid[0]:.2f}, {E_grid[-1]:.2f}]")
    print(f"  DOS 最大值: {np.max(dos):.4f}")

    # Green 函数
    G = solver.compute_green_function(H, 0.0, 0.01)
    print(f"\n[Spectral] Green 函数 G(E=0+iη):")
    print(f"  维度: {G.shape}")
    print(f"  谱权重 Tr[Im G]/π = {-np.imag(np.trace(G))/np.pi:.4f}")

    return solver


def section_8_polynomial_basis():
    """
    第 8 部分：正交多项式基变换

    用于 Berry curvature 的高精度逼近
    """
    print("\n" + "=" * 70)
    print("第 8 部分：正交多项式基变换")
    print("=" * 70)

    converter = PolynomialBasisConverter(max_degree=8)

    # Legendre 多项式展开
    print(f"\n[Polynomial] Legendre 多项式展开系数:")
    for n in range(6):
        coeffs = converter.legendre_to_monomial(n)
        coeffs_str = ', '.join([f'{c:+.4f}' for c in coeffs])
        print(f"  P_{n}(x) = [{coeffs_str}]")

    # Chebyshev 多项式
    print(f"\n[Polynomial] Chebyshev 多项式展开系数:")
    for n in range(5):
        coeffs = converter.chebyshev_to_monomial(n)
        coeffs_str = ', '.join([f'{c:+.4f}' for c in coeffs])
        print(f"  T_{n}(x) = [{coeffs_str}]")

    # Gegenbauer 多项式
    print(f"\n[Polynomial] Gegenbauer 多项式 C_{{3}}^{{λ}}(x):")
    for lam in [0.5, 1.0, 1.5, 2.0]:
        coeffs = converter.gegenbauer_to_monomial(3, lam)
        coeffs_str = ', '.join([f'{c:+.4f}' for c in coeffs])
        print(f"  lambda={lam:.1f}: [{coeffs_str}]")

    # Hermite 多项式
    print(f"\n[Polynomial] Hermite 多项式 H_n(x):")
    for n in range(5):
        coeffs = converter.hermite_to_monomial(n)
        coeffs_str = ', '.join([f'{c:+.4f}' for c in coeffs])
        print(f"  H_{n}(x) = [{coeffs_str}]")

    # 多项式计算测试
    print(f"\n[Polynomial] 多项式计算测试:")
    x = np.linspace(-1, 1, 5)
    for basis in ['legendre', 'chebyshev']:
        coeffs = np.array([1.0, 0.5, 0.25])  # 前 3 项
        y = converter.evaluate_polynomial(coeffs, x, basis)
        print(f"  {basis} 基: f({x[2]:.2f}) = {y[2]:.6f}")

    # Berry curvature 拟合
    print(f"\n[Polynomial] Berry curvature 多项式拟合:")
    np.random.seed(42)
    k_data = np.random.uniform(-0.5, 0.5, (50, 3))
    # 构造测试数据: Ω ≈ sin(kx) * cos(ky) * exp(-kz²)
    omega_data = np.sin(k_data[:, 0]) * np.cos(k_data[:, 1]) * np.exp(-k_data[:, 2]**2)

    coeffs = converter.fit_berry_curvature(k_data, omega_data, 'legendre', max_degree=3)
    print(f"  拟合系数数量: {len(coeffs)}")
    print(f"  前 5 个系数: {coeffs[:5]}")

    return converter


def section_9_ode_integration(berry_calc):
    """
    第 9 部分：ODE 积分与 Berry phase 计算

    实现 Schrödinger 方程在参数空间的绝热演化
    """
    print("\n" + "=" * 70)
    print("第 9 部分：ODE 积分与 Berry phase")
    print("=" * 70)

    # 不同积分方法
    print(f"\n[ODE] ODE 积分方法:")
    integrator_rk4 = ODEIntegrator('rk4')
    integrator_magnus = ODEIntegrator('magnus')

    # 简单测试 ODE: dy/dt = -y
    def f_decay(t, y):
        return -y

    y0 = np.array([1.0])
    t_rk4, y_rk4 = integrator_rk4.integrate(f_decay, y0, (0, 2), 50)
    y_exact = np.exp(-t_rk4)
    error_rk4 = np.max(np.abs(y_rk4[:, 0] - y_exact))
    print(f"  RK4 方法: 最大误差 = {error_rk4:.6e}")

    # Berry phase 计算
    print(f"\n[ODE] Berry phase 沿闭合路径:")
    ham = berry_calc.ham
    mesh = berry_calc.mesh

    # 构造圆形路径
    n_path = 30
    theta = np.linspace(0, 2*np.pi, n_path, endpoint=False)
    radius = 0.1
    k_center = np.array([0.0, 0.0, 0.0])

    k_loop = np.array([
        k_center + radius * np.array([np.cos(t), np.sin(t), 0.0])
        for t in theta
    ])

    # 使用 Zak 相位公式
    def H_func(k):
        k_cart = k[0] * mesh.b1 + k[1] * mesh.b2 + k[2] * mesh.b3
        return ham.hamiltonian_matrix(k_cart)

    berry_phase = compute_zygv_phase(H_func, k_loop, band_idx=0)
    print(f"  路径半径: {radius}")
    print(f"  Berry phase: γ = {berry_phase:.6f}")
    print(f"  γ/π = {berry_phase/np.pi:.6f}")

    # Schrödinger 方程演化
    print(f"\n[ODE] Schrödinger 方程绝热演化:")
    _, u0 = berry_calc.solve_eigenproblem(k_loop[0])
    psi0 = u0[:, 0]

    result = integrator_rk4.compute_berry_phase_evolution(H_func, psi0, k_loop, 50)
    print(f"  最终 Berry phase: {result['final_berry_phase']:.6f}")
    print(f"  态保真度: {result['fidelity']:.6f}")

    # 自适应积分
    print(f"\n[ODE] 自适应步长积分:")
    t_adapt, y_adapt = integrator_rk4.adaptive_integration(f_decay, y0, (0, 2), tol=1e-10)
    error_adapt = np.max(np.abs(y_adapt[:, 0] - np.exp(-t_adapt)))
    print(f"  步数: {len(t_adapt) - 1}")
    print(f"  最大误差: {error_adapt:.6e}")

    return integrator_rk4


def section_10_statistical_analysis(omega_field, k_points):
    """
    第 10 部分：统计分析

    分析 Berry curvature 场的统计特性
    """
    print("\n" + "=" * 70)
    print("第 10 部分：Berry Curvature 统计分析")
    print("=" * 70)

    analyzer = StatisticalAnalyzer()

    # 基本统计
    print(f"\n[Statistics] 基本统计量:")
    stats = analyzer.compute_statistics(omega_field)
    for key, value in stats.items():
        print(f"  {key:12s}: {value:.6f}")

    # 直方图分析
    print(f"\n[Statistics] 分布分析:")
    hist_result = analyzer.histogram_analysis(omega_field, n_bins=20)
    if 'power_law_exponent' in hist_result:
        print(f"  幂律指数 α = {hist_result['power_law_exponent']:.4f}")
        print(f"  (理论值：三维 Weyl 半金属 α = 2.5)")

    # Bootstrap 置信区间
    print(f"\n[Statistics] Bootstrap 置信区间:")
    data_1d = np.linalg.norm(omega_field, axis=1)
    pe, lb, ub = analyzer.bootstrap_confidence_interval(data_1d, 'mean')
    print(f"  均值估计: {pe:.6f}")
    print(f"  95% 置信区间: [{lb:.6f}, {ub:.6f}]")

    # 生成完整报告
    print(f"\n[Statistics] 完整统计报告:")
    report = analyzer.generate_report(omega_field, k_points)
    print(f"  基本统计: mean={report['basic_stats']['mean']:.6f}, "
          f"std={report['basic_stats']['std']:.6f}")
    print(f"  Bootstrap: {report['bootstrap']['point_estimate']:.6f} "
          f"[{report['bootstrap']['confidence_interval'][0]:.6f}, "
          f"{report['bootstrap']['confidence_interval'][1]:.6f}]")

    return report


def print_summary(all_results: Dict):
    """打印总结"""
    print("\n" + "=" * 70)
    print("  计算总结")
    print("=" * 70)

    print("""
本项目实现了 Weyl 半金属 Berry curvature 的完整计算框架：

1. Weyl Hamiltonian 构建
   - 2×2 和 4×4 模型
   - 手性和质量项
   - 速度算符

2. Brillouin 区网格
   - 均匀网格、三角剖分
   - 自适应细化
   - 高对称路径

3. Berry Curvature 计算
   - Kubo 公式
   - 高阶有限差分 (O(h²), O(h⁴))
   - Wilson loop 方法

4. Chern 数计算
   - FHS 离散方法
   - Chern 数随 kz 变化
   - 拓扑相变搜索

5. 稳定性分析
   - 收敛性分析
   - 条件数测试
   - 规范依赖性
   - 奇异性接近

6. 谱方法
   - QR、分治、Lanczos 算法
   - Golub-Welsch 求积
   - 态密度与 Green 函数

7. 多项式基
   - Legendre、Chebyshev、Gegenbauer
   - Hermite、Laguerre
   - Berry curvature 拟合

8. ODE 积分
   - Euler、RK4、RK45、Magnus
   - Berry phase 演化
   - 自适应步长

9. 统计分析
   - 基本统计量
   - 幂律分布拟合
   - Bootstrap 置信区间
   - 关联函数

物理意义：
Weyl 半金属的拓扑性质由 Berry curvature 的积分 (Chern 数) 刻画。
本程序通过多种数值方法的交叉验证，确保结果的可靠性。

运行环境：Python 3.x + NumPy
运行时间：约 30-60 秒 (取决于网格密度)
""")


def main():
    """
    主函数：协调所有计算模块

    计算流程：
    1. Hamiltonian 构建
    2. 网格生成
    3. Berry curvature 计算
    4. Chern 数计算
    5. 有限差分分析
    6. 稳定性分析
    7. 谱方法
    8. 多项式基
    9. ODE 积分
    10. 统计分析
    """
    print_header()
    start_time = time.time()

    all_results = {}

    try:
        # 第 1 部分：Hamiltonian 构建
        ham_positive, ham_negative, mesh = section_1_hamiltonian_construction()
        all_results['hamiltonian'] = {'positive': ham_positive, 'negative': ham_negative}

        # 第 2 部分：Brillouin 区网格
        k_points = section_2_brillouin_mesh(ham_positive, mesh)
        all_results['mesh'] = k_points

        # 第 3 部分：Berry curvature 计算
        berry_calc, omega_field = section_3_berry_curvature(ham_positive, mesh, k_points)
        all_results['berry_curvature'] = omega_field

        # 第 4 部分：Chern 数
        chern_calc = section_4_chern_number(berry_calc, mesh)
        all_results['chern'] = chern_calc

        # 第 5 部分：有限差分
        fd_op = section_5_finite_difference_analysis()
        all_results['fd'] = fd_op

        # 第 6 部分：稳定性分析
        report = section_6_stability_analysis(berry_calc, mesh)
        all_results['stability'] = report

        # 第 7 部分：谱方法
        solver = section_7_spectral_methods()
        all_results['spectral'] = solver

        # 第 8 部分：多项式基
        converter = section_8_polynomial_basis()
        all_results['polynomial'] = converter

        # 第 9 部分：ODE 积分
        integrator = section_9_ode_integration(berry_calc)
        all_results['ode'] = integrator

        # 第 10 部分：统计分析
        stats_report = section_10_statistical_analysis(omega_field, k_points)
        all_results['statistics'] = stats_report

    except Exception as e:
        print(f"\n[错误] 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印总结
    print_summary(all_results)

    elapsed = time.time() - start_time
    print(f"总运行时间: {elapsed:.2f} 秒")
    print("=" * 70)
    print("计算完成！所有模块成功运行。")
    print("=" * 70)


if __name__ == "__main__":
    main()
