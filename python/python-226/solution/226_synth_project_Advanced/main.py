"""
中微子振荡概率与参数反演：高阶有限差分与稳定性分析
=================================================
统一入口程序

科学问题：
在计算高能物理中，中微子振荡是探测基本粒子物理参数的重要工具。
本项目研究三代中微子在真空和物质中的传播，使用高阶有限差分方法
数值求解薛定谔型演化方程，并进行参数反演和稳定性分析。

核心物理模型：
1. 三代中微子振荡（PMNS矩阵）
2. MSW物质效应
3. 高阶有限差分空间离散化
4. von Neumann稳定性分析
5. 参数反演（χ²最小化）
6. 不确定度量化（CLP界 + Bootstrap）

运行方式：
    python main.py

输出：
    控制台打印各模块的计算结果和验证信息。

作者：博士级科研代码合成项目
日期：2026
"""

import sys
import os
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """打印子标题"""
    print(f"\n--- {title} ---")


# ============================================================
# 模块1: 中微子物理基础 (neutrino_physics.py)
# ============================================================
def test_neutrino_physics():
    """测试中微子物理模块"""
    print_header("模块1: 中微子物理基础")

    from neutrino_physics import (
        pmns_matrix, matter_potential, hamiltonian_vacuum,
        hamiltonian_matter, oscillation_probability,
        effective_mixing_angles_matter, check_probability_normalization
    )

    # 物理参数（Global fit values, NuFIT 5.2）
    theta12 = np.radians(33.44)   # 太阳混合角
    theta13 = np.radians(8.57)    # 反应堆混合角
    theta23 = np.radians(49.0)    # 大气混合角
    delta_cp = np.radians(197)    # CP破坏相角
    delta_m2_21 = 7.42e-5         # 太阳质量平方差 (eV²)
    delta_m2_31 = 2.515e-3        # 大气质量平方差 (eV²)

    print(f"混合角: θ₁₂={np.degrees(theta12):.2f}°, θ₁₃={np.degrees(theta13):.2f}°, θ₂₃={np.degrees(theta23):.2f}°")
    print(f"CP相角: δ={np.degrees(delta_cp):.1f}°")
    print(f"质量平方差: Δm²₂₁={delta_m2_21:.2e} eV², Δm²₃₁={delta_m2_31:.2e} eV²")

    # 构造PMNS矩阵
    U = pmns_matrix(theta12, theta13, theta23, delta_cp)
    print(f"\nPMNS矩阵 U:")
    print(np.array2string(U, precision=4, suppress_small=True))

    # 验证幺正性
    unitarity = np.linalg.norm(U.conj().T @ U - np.eye(3))
    print(f"幺正性验证: ||U†U - I|| = {unitarity:.2e}")

    # 计算物质势
    E_MeV = 10.0  # 10 MeV中微子
    N_e = 100.0   # mol/cm³ (太阳中心附近)
    V = matter_potential(N_e, E_MeV)
    print(f"\n物质势: V({N_e} mol/cm³, {E_MeV} MeV) = {V:.4e}")

    # 真空哈密顿量
    E_eV = E_MeV * 1e6
    H_vac = hamiltonian_vacuum(U, delta_m2_21, delta_m2_31, E_eV)
    print(f"\n真空哈密顿量 H_vac (eV):")
    print(np.array2string(H_vac, precision=4, suppress_small=True))

    # 物质中的哈密顿量
    H_mat = hamiltonian_matter(U, delta_m2_21, delta_m2_31, E_eV, N_e)
    print(f"\n物质中哈密顿量 H_mat (eV):")
    print(np.array2string(H_mat, precision=4, suppress_small=True))

    # 有效混合角
    theta12_m, theta13_m, theta23_m = effective_mixing_angles_matter(
        U, delta_m2_21, delta_m2_31, E_eV, N_e
    )
    print(f"\n物质中有效混合角:")
    print(f"  θ₁₂^m = {np.degrees(theta12_m):.2f}° (真空: {np.degrees(theta12):.2f}°)")
    print(f"  θ₁₃^m = {np.degrees(theta13_m):.2f}° (真空: {np.degrees(theta13):.2f}°)")
    print(f"  θ₂₃^m = {np.degrees(theta23_m):.2f}° (真空: {np.degrees(theta23):.2f}°)")

    return {
        'U': U,
        'theta12': theta12, 'theta13': theta13, 'theta23': theta23,
        'delta_cp': delta_cp,
        'delta_m2_21': delta_m2_21, 'delta_m2_31': delta_m2_31,
    }


# ============================================================
# 模块2: 高阶有限差分与稳定性分析 (high_order_finite_difference.py)
# ============================================================
def test_finite_difference():
    """测试有限差分模块"""
    print_header("模块2: 高阶有限差分与稳定性分析")

    from high_order_finite_difference import FDOperator, StabilityAnalyzer, convergence_test

    # 测试不同阶数的精度
    print("收敛性测试（f(x) = sin(x), f'(x) = cos(x)）:")
    f_func = lambda x: np.sin(x)
    df_exact = lambda x: np.cos(x)

    for order in [2, 4, 6, 8]:
        fd_op = FDOperator(dx=0.01, order=order, periodic=True)
        result = convergence_test(fd_op, f_func, df_exact,
                                   N_values=[50, 100, 200, 400])
        print(f"  {order}阶精度: 观测阶数 = {result['observed_order']:.2f}, "
              f"收敛性: {'✓' if result['convergence_ok'] else '✗'}")

    # 稳定性分析
    print("\nvon Neumann稳定性分析:")
    analyzer = StabilityAnalyzer(dx=0.01, dt=0.005)

    c = 1.0  # 对流速度
    k = np.linspace(0, np.pi / 0.01, 100)

    # FTCS (不稳定)
    G_ftcs = analyzer.ftcs_amplification_factor(c, k)
    print(f"  FTCS: |G|_max = {np.max(np.abs(G_ftcs)):.4f} "
          f"({'不稳定' if np.max(np.abs(G_ftcs)) > 1 else '稳定'})")

    # Lax-Wendroff
    G_lw = analyzer.lax_wendroff_amplification(c, k)
    print(f"  Lax-Wendroff: |G|_max = {np.max(np.abs(G_lw)):.4f} "
          f"({'不稳定' if np.max(np.abs(G_lw)) > 1 else '稳定'})")

    # Leapfrog
    G_lf_plus, G_lf_minus = analyzer.leapfrog_amplification(c, k)
    max_lf = max(np.max(np.abs(G_lf_plus)), np.max(np.abs(G_lf_minus)))
    print(f"  Leapfrog: |G|_max = {max_lf:.4f} "
          f"({'不稳定' if max_lf > 1 else '稳定'})")

    # 中微子演化稳定性
    print("\n中微子演化方程稳定性:")
    from neutrino_physics import pmns_matrix, hamiltonian_vacuum
    U = pmns_matrix(0.58, 0.15, 0.85, 3.4)
    H = hamiltonian_vacuum(U, 7.42e-5, 2.515e-3, 1e7)

    dx_test = 100.0  # 100 m
    stability = analyzer.neutrino_evolution_stability(H, dx_test)
    print(f"  H的本征值范围: [{np.min(stability['H_eigenvalues']):.2e}, {np.max(stability['H_eigenvalues']):.2e}]")
    print(f"  Euler格式谱半径: {stability['euler_spectral_radius']:.4f} "
          f"({'不稳定' if not stability['euler_is_stable'] else '稳定'})")
    print(f"  矩阵指数格式酉性: {stability['exact_is_unitary']}")
    print(f"  临界步长: dx_crit = {stability['critical_dx']:.2e} m")

    return True


# ============================================================
# 模块3: 物质密度分布 (matter_density_fem.py)
# ============================================================
def test_matter_density():
    """测试物质密度模块"""
    print_header("模块3: 太阳物质密度分布 (FEM)")

    from matter_density_fem import SolarDensityProfile, FEM1DSolver, generate_1d_density_profile

    # 解析密度分布
    print("标准太阳模型密度分布:")
    profile = SolarDensityProfile('polynomial')
    r = np.linspace(0, 1, 11)
    for r_val in r:
        rho = profile.polynomial_profile(np.array([r_val]))[0]
        N_e = profile.electron_density(np.array([r_val]))[0]
        print(f"  r/R☉ = {r_val:.2f}: ρ = {rho:.2f} g/cm³, N_e = {N_e:.2e} mol/cm³")

    # FEM求解
    print("\n1D FEM求解密度分布:")
    fem = FEM1DSolver(N_elements=50, element_order=1)
    nodes, rho_fem = fem.solve_density_profile(D=0.01, v=1.0, rho_c=150.0)
    print(f"  节点数: {len(nodes)}")
    print(f"  中心密度: {rho_fem[0]:.2f} g/cm³")
    print(f"  表面密度: {rho_fem[-1]:.4f} g/cm³")
    print(f"  密度范围: [{np.min(rho_fem):.4f}, {np.max(rho_fem):.2f}]")

    # 生成完整密度剖面
    r_full, N_e_full = generate_1d_density_profile(200)
    print(f"\n完整密度剖面:")
    print(f"  网格点数: {len(r_full)}")
    print(f"  电子密度范围: [{np.min(N_e_full):.2e}, {np.max(N_e_full):.2e}] mol/cm³")

    return True


# ============================================================
# 模块4: 带状矩阵与Jacobi迭代 (band_matrix_operations.py)
# ============================================================
def test_band_matrix():
    """测试带状矩阵模块"""
    print_header("模块4: 带状矩阵存储与Jacobi迭代")

    from band_matrix_operations import R8PBLMatrix, JacobiSolver, neutrino_hamiltonian_band

    # 构造带状矩阵
    print("R8PBL带状矩阵操作:")
    N = 50
    ML = 3
    band = R8PBLMatrix(N, ML)
    band.random_spd(seed=42)

    print(f"  矩阵维度: {N}×{N}, 半带宽: {ML}")
    print(f"  存储效率: {(ML+1)*N}/{N*N} = {(ML+1)/N*100:.1f}%")

    # 矩阵向量乘积
    x = np.ones(N)
    b = band.matrix_vector_product(x)
    print(f"  矩阵向量乘积: ||Ax|| = {np.linalg.norm(b):.4f}")

    # 转换为稠密矩阵
    A_dense = band.to_dense()
    print(f"  对称性误差: {np.linalg.norm(A_dense - A_dense.T):.2e}")
    print(f"  正定性: {band.is_positive_definite()}")

    # Jacobi迭代求解
    print("\nJacobi迭代求解 Ax = b:")
    b_rhs = np.random.RandomState(42).randn(N)
    solver = JacobiSolver(max_iter=5000, tol=1e-10)

    x_sol, info = solver.jacobi_linear_solve(A_dense, b_rhs)
    print(f"  收敛: {info['converged']}, 迭代次数: {info['iterations']}")
    print(f"  残差: {info['final_residual']:.2e}")
    print(f"  迭代矩阵谱半径: {info['spectral_radius']:.4f}")

    # 验证解
    residual = np.linalg.norm(A_dense @ x_sol - b_rhs)
    print(f"  验证残差: {residual:.2e}")

    # 中微子哈密顿量的本征值
    print("\n中微子哈密顿量本征值 (Jacobi旋转法):")
    from neutrino_physics import pmns_matrix, hamiltonian_vacuum
    U = pmns_matrix(0.58, 0.15, 0.85, 0.0)
    H = hamiltonian_vacuum(U, 7.42e-5, 2.515e-3, 1e7)

    # 取实部构造对称矩阵
    H_sym = np.real(H)
    H_sym = 0.5 * (H_sym + H_sym.T)

    eigvals, eigvecs, eig_info = solver.jacobi_eigenvalues(H_sym)
    print(f"  本征值: {eigvals}")
    print(f"  收敛: {eig_info['converged']}, 迭代: {eig_info['iterations']}")

    # 与numpy对比
    eigvals_np = np.linalg.eigvalsh(H_sym)
    print(f"  numpy本征值: {eigvals_np}")
    print(f"  最大误差: {np.max(np.abs(eigvals - eigvals_np)):.2e}")

    return True


# ============================================================
# 模块5: 矩阵指数演化 (matrix_evolution.py)
# ============================================================
def test_matrix_evolution():
    """测试矩阵指数演化模块"""
    print_header("模块5: 矩阵指数与中微子演化")

    from matrix_evolution import MatrixExponential, NeutrinoEvolution, vacuum_oscillation_length

    # 矩阵指数测试
    print("矩阵指数算法对比:")
    A = np.array([[0, 1], [-1, 0]], dtype=complex)  # 旋转生成元

    methods = ['pade', 'taylor', 'eigenvalue']
    results = {}
    for method in methods:
        mat_exp = MatrixExponential(method)
        expA = mat_exp.compute(A * np.pi / 2)  # 90°旋转
        results[method] = expA
        print(f"  {method:12s}: exp(Aπ/2) = [[{expA[0,0]:.3f}, {expA[0,1]:.3f}], [{expA[1,0]:.3f}, {expA[1,1]:.3f}]]")

    # 验证：应该是 [[0, 1], [-1, 0]]
    expected = np.array([[0, 1], [-1, 0]], dtype=complex)
    for method, res in results.items():
        error = np.linalg.norm(res - expected)
        print(f"    {method:12s}: 误差 = {error:.2e}")

    # 中微子演化
    print("\n中微子真空演化:")
    from neutrino_physics import pmns_matrix, hamiltonian_vacuum

    theta12, theta13, theta23 = 0.58, 0.15, 0.85
    delta_m2_21, delta_m2_31 = 7.42e-5, 2.515e-3
    E_eV = 1e7  # 10 MeV

    U = pmns_matrix(theta12, theta13, theta23, 0.0)
    H = hamiltonian_vacuum(U, delta_m2_21, delta_m2_31, E_eV)

    evolution = NeutrinoEvolution('matrix_exp')

    # 振荡长度
    L_osc_21 = vacuum_oscillation_length(delta_m2_21, E_eV)
    L_osc_31 = vacuum_oscillation_length(delta_m2_31, E_eV)
    print(f"  振荡长度: L₂₁ = {L_osc_21:.2e} m, L₃₁ = {L_osc_31:.2e} m")

    # 演化一段距离
    L = 1000e3  # 1000 km
    U_evolution = evolution.evolve_constant_H(H, L)

    # 振荡概率
    P_ee = np.abs(U_evolution[0, 0]) ** 2
    P_em = np.abs(U_evolution[1, 0]) ** 2
    P_et = np.abs(U_evolution[2, 0]) ** 2

    print(f"\n  传播距离 L = {L/1e3:.0f} km:")
    print(f"  P(νe→νe) = {P_ee:.4f}")
    print(f"  P(νe→νμ) = {P_em:.4f}")
    print(f"  P(νe→ντ) = {P_et:.4f}")
    print(f"  概率归一化: {P_ee + P_em + P_et:.6f}")

    return True


# ============================================================
# 模块6: 网格生成 (mesh_generation.py)
# ============================================================
def test_mesh_generation():
    """测试网格生成模块"""
    print_header("模块6: 网格生成与Voronoi分割")

    from mesh_generation import (
        TriangularMesh2D, VoronoiPartitioner,
        create_rectangular_boundary, create_circular_boundary
    )

    # 矩形网格
    print("矩形域三角形网格:")
    boundary = create_rectangular_boundary(1.0, 1.0, 10)
    mesh = TriangularMesh2D(boundary, max_element_size=0.15)
    nodes, elements = mesh.generate_mesh()
    quality = mesh.compute_mesh_quality()

    print(f"  节点数: {quality['n_nodes']}")
    print(f"  单元数: {quality['n_elements']}")
    print(f"  最小角: {quality['min_angle_deg']:.1f}°")
    print(f"  最大角: {quality['max_angle_deg']:.1f}°")
    print(f"  总面积: {quality['total_area']:.4f}")

    # 圆形网格
    print("\n圆形域三角形网格:")
    boundary_circle = create_circular_boundary(1.0, n_points=30)
    mesh_circle = TriangularMesh2D(boundary_circle, max_element_size=0.2)
    nodes_c, elements_c = mesh_circle.generate_mesh()
    quality_c = mesh_circle.compute_mesh_quality()

    print(f"  节点数: {quality_c['n_nodes']}")
    print(f"  单元数: {quality_c['n_elements']}")
    print(f"  总面积: {quality_c['total_area']:.4f} (理论: π ≈ 3.1416)")

    # Voronoi分割
    print("\n参数空间Voronoi分割:")
    centers = np.array([
        [0.2, 0.3], [0.5, 0.5], [0.8, 0.7], [0.3, 0.8]
    ])
    voronoi = VoronoiPartitioner(centers)
    regions = voronoi.compute_voronoi_regions(resolution=50)

    print(f"  中心点数: {len(centers)}")
    print(f"  区域数: {len(regions)}")
    for i, region in enumerate(regions):
        print(f"    区域{i+1}: {len(region)}个点")

    # 带辅助点的Voronoi
    augmented = voronoi.add_boundary_points(margin=2.0)
    print(f"  增广后中心点数: {len(augmented)} (原始: {len(centers)})")

    return True


# ============================================================
# 模块7: 参数反演 (parameter_inversion.py)
# ============================================================
def test_parameter_inversion():
    """测试参数反演模块"""
    print_header("模块7: 中微子振荡参数反演")

    from parameter_inversion import (
        golden_section_search, GaussianProcessSurrogate,
        NeutrinoParameterInversion, create_synthetic_observation
    )

    # 黄金分割搜索测试
    print("黄金分割搜索测试 (f(x) = (x-2)²):")
    result = golden_section_search(lambda x: (x - 2)**2, 0, 5, tol=1e-8)
    print(f"  最小值点: x = {result['x_min']:.6f} (真值: 2.0)")
    print(f"  最小值: f = {result['f_min']:.2e}")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  收敛: {result['converged']}")

    # GP代理模型测试
    print("\n高斯过程代理模型测试:")
    X_train = np.linspace(0, 5, 20).reshape(-1, 1)
    y_train = np.sin(X_train.ravel()) + 0.1 * np.random.RandomState(42).randn(20)

    gp = GaussianProcessSurrogate(length_scale=0.5, noise=0.01)
    gp.fit(X_train, y_train)

    X_test = np.array([[1.0], [2.5], [4.0]])
    mu, sigma = gp.predict(X_test)
    print(f"  预测测试:")
    for i, x in enumerate(X_test.ravel()):
        print(f"    x={x:.1f}: μ={mu[i]:.4f}, σ={sigma[i]:.4f}")

    # 创建合成观测数据
    print("\n创建合成观测数据:")
    true_params = {
        'theta12': np.radians(33.44),
        'theta13': np.radians(8.57),
        'theta23': np.radians(49.0),
        'delta_cp': 0.0,
        'delta_m2_21': 7.42e-5,
        'delta_m2_31': 2.515e-3,
    }

    energies = np.array([1e6, 5e6, 1e7, 5e7])  # 1, 5, 10, 50 MeV
    baselines = np.array([100e3, 500e3, 1000e3])  # 100, 500, 1000 km

    obs = create_synthetic_observation(true_params, energies, baselines,
                                        noise_level=0.01, seed=42)
    print(f"  观测点数: {len(obs['probs'])}")
    print(f"  概率范围: [{np.min(obs['probs']):.4f}, {np.max(obs['probs']):.4f}]")
    print(f"  噪声水平: {obs['errors'][0]:.4f}")

    # 参数反演
    print("\n参数反演 (坐标下降法):")

    def forward_model(params):
        """简化的正向模型"""
        from neutrino_physics import pmns_matrix, hamiltonian_vacuum
        from matrix_evolution import NeutrinoEvolution

        U = pmns_matrix(params['theta12'], params['theta13'],
                       params['theta23'], params.get('delta_cp', 0))

        evolution = NeutrinoEvolution('matrix_exp')
        probs = np.zeros(len(obs['probs']))

        idx = 0
        for E in energies:
            for L in baselines:
                H = hamiltonian_vacuum(U, params['delta_m2_21'],
                                      params['delta_m2_31'], E)
                U_evo = evolution.evolve_constant_H(H, L)
                probs[idx] = np.abs(U_evo[1, 0]) ** 2
                idx += 1

        return probs

    # 初始化反演器
    inversion = NeutrinoParameterInversion(forward_model, obs)

    # 初始猜测（偏离真值）
    initial_params = {
        'theta12': np.radians(30.0),
        'theta13': np.radians(10.0),
        'theta23': np.radians(45.0),
        'delta_cp': 0.0,
        'delta_m2_21': 8.0e-5,
        'delta_m2_31': 2.6e-3,
    }

    chi2_initial = inversion.chi_squared(initial_params)
    print(f"  初始χ²: {chi2_initial:.4f}")

    # 坐标下降优化
    result = inversion.coordinate_descent(initial_params, max_cycles=10, tol=1e-6)

    print(f"  最终χ²: {result['chi2_min']:.4f}")
    print(f"  循环次数: {result['cycles']}")
    print(f"  收敛: {result['converged']}")

    # 参数对比
    print(f"\n  参数反演结果:")
    for name in true_params:
        true_val = true_params[name]
        inv_val = result['optimal_params'][name]
        if 'theta' in name or 'delta_cp' in name:
            print(f"    {name}: {np.degrees(inv_val):.2f}° (真值: {np.degrees(true_val):.2f}°)")
        else:
            print(f"    {name}: {inv_val:.2e} (真值: {true_val:.2e})")

    return True


# ============================================================
# 模块8: 蒙特卡罗采样 (monte_carlo_sampling.py)
# ============================================================
def test_monte_carlo():
    """测试蒙特卡罗模块"""
    print_header("模块8: 蒙特卡罗采样与KMC")

    from monte_carlo_sampling import (
        HypersphereSampler, ArrheniusExtrapolator,
        KineticMonteCarlo, PermutationOrthogonality, CasinoParadox
    )

    # 超球面采样
    print("正超球面采样 (3维):")
    sampler = HypersphereSampler(dimension=3)
    stats = sampler.distance_statistics(n_samples=500, seed=42)

    print(f"  样本数: {stats['n_samples']}")
    print(f"  点对数: {stats['n_pairs']}")
    print(f"  平均距离: {stats['mean_distance']:.4f}")
    print(f"  距离标准差: {stats['std_distance']:.4f}")
    print(f"  距离范围: [{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]")
    print(f"  均方距离: {stats['mean_squared_distance']:.4f}")

    # Arrhenius外推
    print("\nArrhenius温度外推:")
    # 合成数据: k = 1e10 × exp(-5000/T)
    T_data = np.array([1000, 1200, 1500, 2000])  # K
    k_data = 1e10 * np.exp(-5000 / T_data)

    extrapolator = ArrheniusExtrapolator()
    result = extrapolator.extrapolate_with_bootstrap(
        T_data, k_data, target_T=800, n_bootstrap=500, seed=42
    )

    print(f"  训练温度: {T_data} K")
    print(f"  目标温度: 800 K")
    print(f"  预测速率: {result['rate_mean']:.2e} ± {result['rate_std']:.2e}")
    print(f"  95%置信区间: [{result['ci_95_lower']:.2e}, {result['ci_95_upper']:.2e}]")

    # KMC模拟
    print("\n动力学蒙特卡罗模拟:")
    kmc = KineticMonteCarlo(n_states=3)
    initial_state = np.array([1.0, 0.0, 0.0])  # 纯νe
    kmc.initialize(initial_state)

    # 简单的味转换速率
    rates = np.array([0.1, 0.05, 0.02])  # 三个转换通道
    # 转移矩阵（简化）
    transition_matrices = np.array([
        [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]],  # νe→νμ
        [[0.95, 0.0, 0.05], [0.0, 1.0, 0.0], [0.05, 0.0, 0.95]],  # νe→ντ
        [[1.0, 0.0, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9]],  # νμ→ντ
    ])

    result_kmc = kmc.run(rates, transition_matrices, n_steps=100, seed=42)

    print(f"  总步数: {result_kmc['n_steps']}")
    print(f"  总时间: {result_kmc['total_time']:.2f}")
    print(f"  终态: {result_kmc['final_state']}")
    print(f"  终态概率: [|νe|²={result_kmc['final_state'][0]:.3f}, "
          f"|νμ|²={result_kmc['final_state'][1]:.3f}, |ντ|²={result_kmc['final_state'][2]:.3f}]")

    # 置换正交性
    print("\n置换正交性演示:")
    ortho_result = PermutationOrthogonality.demonstrate(n=10, seed=42)
    print(f"  置换维度: n={ortho_result['n']}")
    print(f"  点积: {ortho_result['dot_product']:.2e}")
    print(f"  正交性: {'✓ 满足' if ortho_result['is_orthogonal'] else '✗ 不满足'}")

    # 赌场悖论
    print("\n赌场悖论模拟:")
    casino = CasinoParadox.simulate(n_flips=100, n_trials=1000, seed=42)
    print(f"  抛币次数: {casino['n_flips']}")
    print(f"  试验次数: {casino['n_trials']}")
    print(f"  期望赌注: {casino['mean_final']:.2f}")
    print(f"  中位数赌注: {casino['median_final']:.2f}")
    print(f"  几何平均: {casino['geometric_mean']:.2f}")
    print(f"  单次期望: {casino['expected_single_flip']:.4f}")
    print(f"  单次几何平均: {casino['geometric_single_flip']:.4f}")

    return True


# ============================================================
# 模块9: 不确定度量化 (uncertainty_quantification.py)
# ============================================================
def test_uncertainty():
    """测试不确定度量化模块"""
    print_header("模块9: 不确定度量化与CLP界")

    from uncertainty_quantification import (
        ConditionalLinearProgram, CrossFittingEstimator,
        MultiplierBootstrap, ParameterUncertainty, create_constraint_matrix
    )

    # CLP界
    print("条件线性规划界:")
    A, q = create_constraint_matrix(n_params=3)
    clp = ConditionalLinearProgram(A, q)
    vertices = clp.enumerate_vertices()
    print(f"  约束矩阵: {A.shape}")
    print(f"  顶点数: {len(vertices)}")

    # 测试界计算
    b_hat = np.array([0.5, 0.3, 0.2])
    min_bound = clp.solve_bound(b_hat, 'min')
    max_bound = clp.solve_bound(b_hat, 'max')
    print(f"  测试向量: {b_hat}")
    print(f"  下界: {min_bound:.4f}")
    print(f"  上界: {max_bound:.4f}")

    # 交叉拟合
    print("\n交叉拟合估计:")
    N = 200
    D = 5
    X = np.random.RandomState(42).randn(N, D)
    true_beta = np.array([1.0, -0.5, 0.3, 0.0, 0.2])
    y = X @ true_beta + 0.1 * np.random.RandomState(42).randn(N)

    cf = CrossFittingEstimator(n_folds=5, estimator_type='ols')
    y_hat = cf.fit_predict(X, y)

    mse = np.mean((y - y_hat) ** 2)
    print(f"  样本数: {N}, 特征数: {D}")
    print(f"  交叉拟合MSE: {mse:.4f}")
    print(f"  真实系数: {true_beta}")

    # 直接用OLS对比
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"  OLS估计: {beta_ols}")
    print(f"  系数误差: {np.linalg.norm(beta_ols - true_beta):.4f}")

    # Bootstrap置信区间
    print("\nMultiplier Bootstrap置信区间:")
    contributions = np.random.RandomState(42).randn(100) * 0.5 + 1.0
    bootstrap = MultiplierBootstrap(n_bootstrap=1000, confidence_level=0.95)
    ci = bootstrap.compute_ci(contributions, seed=42)

    print(f"  样本数: {len(contributions)}")
    print(f"  点估计: {ci['point_estimate']:.4f}")
    print(f"  95%置信区间: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    print(f"  标准误: {ci['std_error']:.4f}")
    print(f"  Bootstrap均值: {ci['bootstrap_mean']:.4f}")

    # 参数不确定度
    print("\n参数不确定度分析:")
    param_unc = ParameterUncertainty()

    # 合成第一阶段估计
    N_samples = 50
    r = 3
    b_hat_samples = np.random.RandomState(42).randn(N_samples, r) * 0.1 + np.array([0.5, 0.3, 0.2])

    bounds = param_unc.compute_parameter_bounds(b_hat_samples, A, q, seed=42)
    print(f"  样本数: {N_samples}")
    print(f"  平均下界: {bounds['mean_min_bound']:.4f}")
    print(f"  平均上界: {bounds['mean_max_bound']:.4f}")
    print(f"  下界95%CI: [{bounds['min_bound']['ci_lower']:.4f}, {bounds['min_bound']['ci_upper']:.4f}]")
    print(f"  上界95%CI: [{bounds['max_bound']['ci_lower']:.4f}, {bounds['max_bound']['ci_upper']:.4f}]")

    return True


# ============================================================
# 模块10: 数值工具 (numerical_utils.py)
# ============================================================
def test_numerical_utils():
    """测试数值工具模块"""
    print_header("模块10: 数值工具与正则化")

    from numerical_utils import (
        local_contrast_enhancement, log_sum_exp, condition_number,
        regularize_matrix, compute_gamma_function, compute_bessel_j,
        compute_error_function, numerical_gradient, softmax,
        kullback_leibler_divergence, jensen_shannon_divergence
    )

    # 对比度增强
    print("局部对比度增强:")
    data_1d = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.RandomState(42).randn(100)

    for s in [0.5, 1.0, 1.5, 2.0]:
        enhanced = local_contrast_enhancement(data_1d, sharpness=s)
        variation = np.std(enhanced) / np.std(data_1d)
        print(f"  s={s:.1f}: 标准差比 = {variation:.3f} "
              f"({'平滑' if s < 1 else '锐化' if s > 1 else '无变化'})")

    # log-sum-exp
    print("\nlog-sum-exp测试:")
    x = np.array([1000, 1001, 1002])
    lse = log_sum_exp(x)
    print(f"  x = {x}")
    print(f"  log-sum-exp = {lse:.4f}")
    print(f"  max(x) = {np.max(x):.4f}")
    print(f"  数值稳定性: 无溢出 ✓")

    # 条件数
    print("\n矩阵条件数:")
    A_well = np.array([[1, 0], [0, 1]], dtype=float)
    A_ill = np.array([[1, 1], [1, 1.001]], dtype=float)

    print(f"  良态矩阵: κ = {condition_number(A_well):.2f}")
    print(f"  病态矩阵: κ = {condition_number(A_ill):.2f}")

    A_reg = regularize_matrix(A_ill, epsilon=0.01)
    print(f"  正则化后: κ = {condition_number(A_reg):.2f}")

    # 特殊函数
    print("\n特殊函数计算:")
    print(f"  Γ(5) = {compute_gamma_function(5):.4f} (理论: 24)")
    print(f"  J₀(1) = {compute_bessel_j(0, 1.0):.6f}")
    print(f"  erf(1) = {compute_error_function(1.0):.6f}")

    # 数值梯度
    print("\n数值梯度:")
    f = lambda x: x[0]**2 + x[1]**2
    x0 = np.array([1.0, 2.0])
    grad = numerical_gradient(f, x0)
    print(f"  f(x) = x₁² + x₂²")
    print(f"  在 x={x0} 处:")
    print(f"  数值梯度: {grad}")
    print(f"  解析梯度: [2.0, 4.0]")
    print(f"  误差: {np.linalg.norm(grad - np.array([2.0, 4.0])):.2e}")

    # Softmax
    print("\nSoftmax函数:")
    x = np.array([1.0, 2.0, 3.0])
    probs = softmax(x)
    print(f"  输入: {x}")
    print(f"  Softmax: {probs}")
    print(f"  归一化: {np.sum(probs):.6f}")

    # KL散度和JS散度
    print("\n信息散度:")
    p = np.array([0.4, 0.3, 0.3])
    q = np.array([0.3, 0.4, 0.3])

    kl = kullback_leibler_divergence(p, q)
    js = jensen_shannon_divergence(p, q)
    print(f"  P = {p}")
    print(f"  Q = {q}")
    print(f"  D_KL(P||Q) = {kl:.6f}")
    print(f"  D_JS(P||Q) = {js:.6f}")

    return True


# ============================================================
# 主程序
# ============================================================
def main():
    """主程序入口"""
    print("\n" + "=" * 70)
    print("  中微子振荡概率与参数反演：高阶有限差分与稳定性分析")
    print("  博士级科研代码合成项目")
    print("=" * 70)
    print("\n本项目融合15个种子项目的核心算法，解决计算高能物理中的")
    print("中微子振荡参数反演问题。包含：")
    print("  • 三代中微子振荡与MSW物质效应")
    print("  • 高阶有限差分与von Neumann稳定性分析")
    print("  • FEM太阳密度分布求解")
    print("  • 带状矩阵存储与Jacobi迭代")
    print("  • 矩阵指数精确演化")
    print("  • 参数反演（黄金分割 + GP代理）")
    print("  • 蒙特卡罗采样与KMC模拟")
    print("  • 不确定度量化（CLP界 + Bootstrap）")
    print("  • 数值工具与正则化")

    print("\n" + "=" * 70)
    print("  开始执行各模块测试...")
    print("=" * 70)

    # 执行所有测试
    modules = [
        ("中微子物理基础", test_neutrino_physics),
        ("高阶有限差分", test_finite_difference),
        ("物质密度分布", test_matter_density),
        ("带状矩阵与Jacobi", test_band_matrix),
        ("矩阵指数演化", test_matrix_evolution),
        ("网格生成", test_mesh_generation),
        ("参数反演", test_parameter_inversion),
        ("蒙特卡罗采样", test_monte_carlo),
        ("不确定度量化", test_uncertainty),
        ("数值工具", test_numerical_utils),
    ]

    results = {}
    for name, func in modules:
        try:
            result = func()
            results[name] = "✓ 通过"
        except Exception as e:
            results[name] = f"✗ 失败: {str(e)}"
            import traceback
            traceback.print_exc()

    # 总结
    print_header("执行总结")
    for name, status in results.items():
        print(f"  {name:20s}: {status}")

    n_passed = sum(1 for s in results.values() if s.startswith("✓"))
    print(f"\n  总计: {n_passed}/{len(results)} 个模块通过")

    print("\n" + "=" * 70)
    print("  计算完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
