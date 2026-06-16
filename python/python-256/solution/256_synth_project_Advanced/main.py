"""
main.py
=======
PROJECT_256 统一入口: 计算天体物理——恒星振动模式与星震学反演
高阶有限差分与稳定性分析 (小规模可复现实验)

本程序执行完整的星震学计算流程:
  1. 构建恒星平衡模型 (多方球)
  2. 计算振荡方程系数
  3. 高阶有限差分离散化
  4. 稳定性分析 (非绝热)
  5. 域分解 (有限元网格)
  6. 模式分类与频率匹配
  7. 反演推断内部结构
  8. 蒙特卡罗不确定性量化
  9. 超网络参数预测

所有计算不依赖可视化, 结果输出到标准输出.

运行方式:
  python main.py

作者: 合成项目自动生成
"""

import numpy as np
import time
import sys

# ============================================================
# 导入项目模块
# ============================================================
from stellar_structure import (
    StellarStructureModel,
    HypernetworkParameterMapper,
    compute_frequency_spacings,
)
from oscillation_equations import (
    LagrangeBasisHighOrder,
    OscillationEquationCoefficients,
    UniformSphereAnalytic,
)
from high_order_finite_diff import (
    FornbergWeights,
    CompactFiniteDifference,
    SpectralDifferentiation,
    JacobiEigenSolver,
    stability_analysis_fd,
)
from stability_analysis import (
    NonAdiabaticStabilityMatrix,
    BorderCollisionBifurcationDetector,
    PulsationAmplitudeEquation,
)
from inverse_problem import (
    FrequencySensitivityKernels,
    SOLAInversion,
    AdjointGradientComputer,
    HankelMatrixInversion,
)
from monte_carlo_sampler import (
    ZigguratGenerator,
    EllipseParameterSampler,
    MCMCSampler,
    monte_carlo_uncertainty_quantification,
)
from mode_classifier import (
    ModeIdentifierMap,
    EigenfunctionTopologyEncoder,
    ModePatternRecognizer,
)
from boundary_encoding import (
    BoundaryConditionEncoder,
    boundary_is_legal,
    boundary_sort,
    boundary_to_edge_xy,
    compute_boundary_enclosure_area,
)
from domain_decomposition import (
    SphericalShellMesh,
    QuadraticShapeFunctions,
    EarClippingTriangulator,
    assemble_stiffness_matrix,
)
from hypernetwork_params import (
    TimeSeriesEncoder,
    HyperNetworkParams,
    FAdjointTrainer,
    generate_training_data,
)


def print_header(title: str) -> None:
    """打印章节标题."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """打印子节标题."""
    print(f"\n--- {title} ---")


def phase1_stellar_structure() -> StellarStructureModel:
    """
    阶段 1: 构建恒星平衡模型.

    使用 n=3 多方球 (Eddington 标准模型),
    模拟太阳型恒星.
    """
    print_header("阶段 1: 恒星平衡模型构建")

    model = StellarStructureModel(
        mass_solar=1.0,
        radius_solar=1.0,
        polytropic_index=3.0,
        X_H=0.70,
        Z_metal=0.02,
        n_radial=256,
    )

    print(f"  恒星质量: {model.M_star / 1.989e33:.4f} M_sun")
    print(f"  恒星半径: {model.R_star / 6.957e10:.4f} R_sun")
    print(f"  多方指数: n = {model.n_poly}")
    print(f"  平均分子量: μ = {model.mu:.4f}")
    print(f"  中心密度: ρ_c = {model.rho_c:.4e} g/cm³")
    print(f"  中心压强: P_c = {model.P_c:.4e} dyn/cm²")
    print(f"  中心温度: T_c = {model.T_c:.4e} K")

    # 声速和频率标度
    c_s = model.get_sound_speed_profile()
    omega_ac = model.get_acoustic_cutoff_frequency()
    print(f"  中心声速: c_s(0) = {c_s[0]:.4e} cm/s")
    print(f"  表面声速: c_s(R) = {c_s[-1]:.4e} cm/s")
    print(f"  声学截止频率: ω_ac = {omega_ac:.4e} rad/s")

    # N² 剖面统计
    N2 = model.N2
    print(f"  Brunt-Väisälä 频率²: max = {np.max(N2):.4e}, min = {np.min(N2):.4e}")

    # Lamb 频率
    L2 = model.get_lamb_frequency(1)
    print(f"  l=1 Lamb 频率 (表面): L_1(R) = {L2[-1]:.4e} rad/s")

    return model


def phase2_oscillation_equations(model: StellarStructureModel) -> dict:
    """
    阶段 2: 振荡方程构造与高阶差分.
    """
    print_header("阶段 2: 振荡方程与高阶有限差分")

    # Lagrange 基函数
    section_title = "6 阶 Lagrange 插值 (Chebyshev 节点)"
    print_section(section_title)
    lagrange = LagrangeBasisHighOrder(order=6, a=0.0, b=1.0)
    print(f"  节点: {lagrange.nodes}")

    # 微分矩阵
    D = lagrange.differentiation_matrix()
    print(f"  微分矩阵 D (前 3×3):")
    print(f"    {D[:3, :3]}")

    # 振荡方程系数
    print_section("振荡方程系数矩阵")
    osc = OscillationEquationCoefficients(model)
    omega_test = 2.0 * np.pi * 100e-6  # 100 μHz
    M_osc = osc.build_oscillation_matrix(omega_test, l=1)
    print(f"  测试频率: ω = {omega_test:.4e} rad/s ({omega_test/(2*np.pi*1e-6):.1f} μHz)")
    print(f"  系数矩阵 M 形状: {M_osc.shape}")
    print(f"  M[0,0,0] = {M_osc[0, 0, 0]:.4e}")

    # Richardson 数
    Ri = osc.compute_richardson_number()
    pos_mask = Ri > 0
    if np.any(pos_mask):
        print(f"  Richardson 数: max = {np.max(Ri):.4e}, min(>0) = {np.min(Ri[pos_mask]):.4e}")
    else:
        print(f"  Richardson 数: max = {np.max(Ri):.4e}, 无正值")

    # 紧致差分
    print_section("4 阶紧致差分格式")
    cfd = CompactFiniteDifference(scheme_order=4, n_points=64, h=0.01)
    A_cfd, B_cfd = cfd.first_derivative_matrix()
    print(f"  隐式矩阵 A: 非零元素 = {np.count_nonzero(A_cfd)}")
    print(f"  显式矩阵 B: 非零元素 = {np.count_nonzero(B_cfd)}")

    # Von Neumann 分析
    kh, k_prime = cfd.von_neumann_analysis()
    resolution_error = np.max(np.abs(k_prime - kh)) / np.pi
    print(f"  修正波数最大偏差: Δk/k_max = {resolution_error:.6f}")

    # 谱方法
    print_section("Chebyshev 谱方法 (N=16)")
    spectral = SpectralDifferentiation(N=16)
    print(f"  Chebyshev 节点数: {spectral.N + 1}")
    print(f"  微分矩阵 D 的谱半径: {np.max(np.abs(np.linalg.eigvals(spectral.D))):.4e}")

    # 均匀球基准
    print_section("均匀密度球解析基准")
    analytic = UniformSphereAnalytic(rho0=1.41, R=6.957e10)
    print(f"  特征频率: ω₀ = {analytic.omega0:.4e} rad/s")
    for n in range(1, 4):
        freq = analytic.radial_frequency(n)
        print(f"  径向模 n={n}: ν = {freq/(2*np.pi*1e-6):.2f} μHz")
    for l in range(1, 4):
        freq = analytic.f_mode_frequency(l)
        print(f"  f 模 l={l}: ν = {freq/(2*np.pi*1e-6):.2f} μHz")

    return {"lagrange": lagrange, "osc_coeff": osc, "analytic": analytic}


def phase3_stability(model: StellarStructureModel) -> dict:
    """
    阶段 3: 稳定性分析与分岔检测.
    """
    print_header("阶段 3: 稳定性分析与分岔检测")

    # 非绝热稳定性
    print_section("非绝热稳定性矩阵")
    n_modes = min(32, model.n_r)
    stability = NonAdiabaticStabilityMatrix(n_modes, model)
    omega_complex, eigenvalues = stability.compute_complex_frequencies()

    # 增长率谱
    result = stability.growth_rate_spectrum()
    n_unstable = np.sum(~result["is_stable"])
    print(f"  模式数量: {len(result['omega_real'])}")
    print(f"  不稳定模式数: {n_unstable}")
    if len(result["gamma"]) > 0:
        print(f"  最大增长率: γ_max = {np.max(result['gamma']):.4e} s⁻¹")
        print(f"  最高频率: ν_max = {np.max(result['omega_real'])/(2*np.pi*1e-6):.2f} μHz")

    # 分岔检测
    print_section("Border-collision 分岔检测")
    bcd = BorderCollisionBifurcationDetector(n_steps=20)

    # p-g 模反交叉
    freq_p = np.array([100.0, 150.0, 200.0, 250.0])  # μHz
    freq_g = np.array([95.0, 148.0, 205.0])
    anticross = bcd.detect_anticrossing(freq_p, freq_g, coupling_strength=2.0)
    print(f"  p-g 反交叉事件数: {anticross['n_crossings']}")
    if anticross['n_crossings'] > 0:
        print(f"  最小频率间距: {anticross['min_spacing']:.2f} μHz")

    # 分岔扫描 (简化)
    bifurcation = bcd.scan_bifurcation(mu_range=(0.5, 1.5), tau=0.95)
    print(f"  分岔扫描: {len(bifurcation['mu_values'])} 个参数点")

    # 振幅方程
    print_section("非线性脉动振幅方程")
    amp_eq = PulsationAmplitudeEquation(n_modes=2)
    sigma = np.array([0.01 + 0.1j, -0.005 + 0.15j])
    l_coeff = np.array([0.1, 0.1])
    A, T = amp_eq.evolve_amplitude(sigma, l_coeff, dt=0.1, n_steps=500)
    print(f"  模式 1 最终振幅: |A₁| = {np.abs(A[-1, 0]):.6f}")
    print(f"  模式 2 最终振幅: |A₂| = {np.abs(A[-1, 1]):.6f}")

    # FD 稳定性
    print_section("有限差分稳定性分析")
    for order in [2, 4, 6]:
        sa = stability_analysis_fd(order=order, n_points=64)
        print(f"  {order} 阶格式: 谱半径 = {sa['spectral_radius']:.4e}, "
              f"稳定 CFL = {sa['stable_cfl']:.4f}")

    return {"stability": stability, "bifurcation": bifurcation}


def phase4_mode_classification(model: StellarStructureModel) -> dict:
    """
    阶段 4: 模式分类与频率匹配.
    """
    print_header("阶段 4: 模式分类与频率匹配")

    # 合成观测频率 (从解析模型)
    analytic = UniformSphereAnalytic(rho0=1.41, R=6.957e10)
    freqs_analytic = analytic.get_all_frequencies(n_max=5, l_max=2)
    # 转换为 μHz
    freqs_muHz = freqs_analytic / (2.0 * np.pi * 1e-6)

    print_section("合成振荡频率")
    print(f"  频率数量: {len(freqs_muHz)}")
    print(f"  频率范围: [{freqs_muHz[0]:.2f}, {freqs_muHz[-1]:.2f}] μHz")

    # 频率间距
    spacings = compute_frequency_spacings(freqs_muHz)
    print(f"  平均大间距: Δν = {spacings['delta_nu_mean']:.2f} μHz")

    # 模式分类
    print_section("模式标识符映射")
    mapper = ModeIdentifierMap()
    assignments = mapper.assign_quantum_numbers(
        freqs_muHz, delta_nu=spacings['delta_nu_mean']
    )
    print(f"  已分类模式数: {len(assignments)}")
    for i, a in enumerate(assignments[:5]):
        print(f"    模式 {i}: ν={a['frequency']:.2f} μHz, "
              f"n={a['n']}, l={a['l']}, id={a['identifier']}")

    # 加密/解密测试
    print_section("单表替换密码模式标识")
    test_plain = "ABCDEF"
    encrypted = mapper.encrypt(test_plain)
    decrypted = mapper.decrypt(encrypted)
    print(f"  原始: {test_plain}")
    print(f"  加密: {encrypted}")
    print(f"  解密: {decrypted}")
    print(f"  正确性: {decrypted == test_plain}")

    # 拓扑编码
    print_section("特征函数拓扑编码")
    encoder = EigenfunctionTopologyEncoder()
    xi_r_test = np.sin(np.linspace(0, 3 * np.pi, 100))
    xi_h_test = np.cos(np.linspace(0, 3 * np.pi, 100))
    boundary_word = encoder.encode_eigenfunction(xi_r_test, xi_h_test, l=2)
    parity = encoder.compute_parity(boundary_word)
    print(f"  边界字: {boundary_word[:30]}...")
    print(f"  奇偶性: {parity}")

    symmetry = encoder.mode_symmetry_check(xi_r_test, l=2)
    print(f"  对称性检查: 理论=(-1)^l={symmetry['expected_parity']}, "
          f"计算={symmetry['computed_parity']}, "
          f"一致={symmetry['is_consistent']}")

    # echelle 图分析
    print_section("echelle 图模式图样")
    recognizer = ModePatternRecognizer(delta_nu=spacings['delta_nu_mean'])
    x_ech, y_ech = recognizer.compute_echelle_coordinates(freqs_muHz)
    print(f"  echelle X 范围: [{np.min(x_ech):.2f}, {np.max(x_ech):.2f}] μHz")
    print(f"  echelle Y 范围: [{np.min(y_ech):.0f}, {np.max(y_ech):.0f}]")

    ridges = recognizer.identify_ridges(freqs_muHz, assignments)
    for l_val, ridge_freqs in ridges.items():
        if len(ridge_freqs) >= 3:
            fit = recognizer.fit_ridge_curvature(ridge_freqs)
            print(f"  l={l_val} ridge: {len(ridge_freqs)} 个模式, "
                  f"Δν={fit['delta_nu']:.2f}, "
                  f"曲率={fit['curvature']:.4f}")

    return {"assignments": assignments, "freqs": freqs_muHz}


def phase5_domain_decomposition(model: StellarStructureModel) -> dict:
    """
    阶段 5: 域分解与有限元网格.
    """
    print_header("阶段 5: 域分解与有限元网格")

    # 球壳网格
    print_section("球壳域三角形网格")
    mesh = SphericalShellMesh(n_r=12, n_theta=15, r_min=0.01, r_max=1.0)
    print(f"  径向节点: {mesh.n_r}")
    print(f"  角度节点: {mesh.n_theta}")
    print(f"  3 节点单元数: {len(mesh.elements_3)}")
    print(f"  6 节点单元数: {len(mesh.elements_6)}")
    print(f"  总节点数 (3 节点): {len(mesh.nodes)}")
    print(f"  总节点数 (6 节点): {len(mesh.nodes_6)}")

    # 面积
    area = mesh.compute_element_area(mesh.elements_3[0])
    print(f"  第一个三角形面积: {area:.6f}")

    # 6 节点形函数
    print_section("6 节点二次形函数")
    elem_coords = mesh.nodes_6[mesh.elements_6[0] - 1]
    shape = QuadraticShapeFunctions(elem_coords)

    # 在重心处测试
    lam_test = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    N_test = shape.shape_values(lam_test)
    print(f"  重心处形函数值: N = {N_test}")
    print(f"  和 = {np.sum(N_test):.6f} (应为 1)")

    # 导数
    dN_test = shape.shape_derivatives(lam_test)
    print(f"  形函数导数 (重心处, N₁): {dN_test[0]}")

    # 刚度矩阵组装
    print_section("有限元刚度/质量矩阵组装")
    K, M = assemble_stiffness_matrix(mesh, stiffness_func=None)
    print(f"  刚度矩阵 K: {K.shape}, 非零元素 = {np.count_nonzero(np.abs(K) > 1e-10)}")
    print(f"  质量矩阵 M: {M.shape}, 非零元素 = {np.count_nonzero(np.abs(M) > 1e-10)}")
    print(f"  K 的谱半径: {np.max(np.abs(np.linalg.eigvals(K + 1e-10 * np.eye(len(K))))):.4e}")

    # 耳切法三角化
    print_section("耳切法多边形三角化")
    # 创建一个凸多边形
    n_verts = 8
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    triangulator = EarClippingTriangulator()
    triangles = triangulator.triangulate(vertices)
    print(f"  多边形顶点数: {n_verts}")
    print(f"  三角形数: {len(triangles)}")
    print(f"  应为: {n_verts - 2}")

    return {"mesh": mesh, "K": K, "M": M}


def phase6_boundary_encoding(model: StellarStructureModel) -> dict:
    """
    阶段 6: 边界条件编码与验证.
    """
    print_header("阶段 6: 边界条件编码与验证")

    # 边界条件编码
    print_section("六方向边界字编码")
    encoder = BoundaryConditionEncoder(model)
    word = encoder.encode_boundary_conditions(omega=2e-4, l=1)
    print(f"  边界字: {word}")
    print(f"  长度: {len(word)}")

    # 合法性检查
    is_legal = boundary_is_legal(word)
    print(f"  合法性: {is_legal}")

    # 测试合法性函数
    legal_word = "123456123456"  # 合法 (各对匹配)
    illegal_word = "112233"       # 非法 (1/4 不匹配)
    print(f"  测试合法字 '{legal_word}': {boundary_is_legal(legal_word)}")
    print(f"  测试非法字 '{illegal_word}': {boundary_is_legal(illegal_word)}")

    # 排序
    print_section("边界字规范化排序")
    if len(word) > 0 and boundary_is_legal(word):
        word_sorted, p_sorted = boundary_sort(word)
        print(f"  排序后: {word_sorted}")
        print(f"  新起点: {p_sorted}")

    # 边界字到坐标
    print_section("边界字几何映射")
    test_word = "ABCDEFGHIJ"
    coords = boundary_to_edge_xy(test_word)
    print(f"  测试字: {test_word}")
    print(f"  路径起点: ({coords[0, 0]:.4f}, {coords[0, 1]:.4f})")
    print(f"  路径终点: ({coords[-1, 0]:.4f}, {coords[-1, 1]:.4f})")
    area = compute_boundary_enclosure_area(coords)
    print(f"  围成面积: {area:.6f}")

    return {"boundary_word": word}


def phase7_inverse_problem(model: StellarStructureModel, freqs: np.ndarray) -> dict:
    """
    阶段 7: 星震学反演.
    """
    print_header("阶段 7: 星震学反演")

    n_modes_test = min(10, len(freqs))

    # 灵敏度核
    print_section("频率灵敏度核")
    kernel_comp = FrequencySensitivityKernels(model)
    xi_r_test = np.sin(np.linspace(0, 2 * np.pi, model.n_r))
    xi_h_test = 0.1 * np.cos(np.linspace(0, 2 * np.pi, model.n_r))

    k_c2 = kernel_comp.compute_kernel_c2(xi_r_test, xi_h_test, l=1)
    k_rho = kernel_comp.compute_kernel_rho(xi_r_test, xi_h_test, l=1)
    print(f"  声速核 K_c²: 范围 [{np.min(k_c2):.4e}, {np.max(k_c2):.4e}]")
    print(f"  密度核 K_ρ: 范围 [{np.min(k_rho):.4e}, {np.max(k_rho):.4e}]")

    # SOLA 反演
    print_section("SOLA 反演")
    n_kernels = n_modes_test
    kernels = np.random.randn(n_kernels, model.n_r) * 0.01
    kernels[0] = k_c2  # 第一个核使用真实核

    # 合成频率偏差
    delta_nu = np.random.randn(n_kernels) * 1e-4

    # 目标位置
    r_targets = np.linspace(0.1, 0.9, 10) * model.R_star

    sola = SOLAInversion(n_targets=10, theta=1e-3, target_width=0.05)
    result = sola.invert(kernels, model.r[:model.n_r], delta_nu, r_targets)
    print(f"  目标位置数: {len(r_targets)}")
    print(f"  推断 δq/q 范围: [{np.min(result['delta_q_over_q']):.4e}, "
          f"{np.max(result['delta_q_over_q']):.4e}]")

    # 伴随梯度
    print_section("伴随梯度反演")
    n_adjoint_modes = min(5, n_modes_test)
    adjoint = AdjointGradientComputer(n_params=5, n_modes=n_adjoint_modes, learning_rate=0.05)

    # 合成观测
    params_true = np.array([1.0, 1.0, 0.70, 0.02, 1.8])
    nu_obs = freqs[:n_adjoint_modes]
    sigma_obs = nu_obs * 0.01  # 1% 误差

    params_init = np.array([0.9, 1.1, 0.65, 0.03, 1.5])
    params_opt, loss_hist, _ = adjoint.gradient_descent(
        nu_obs, params_init, n_iter=30
    )
    print(f"  初始参数: {params_init}")
    print(f"  优化参数: {params_opt}")
    print(f"  初始损失: {loss_hist[0]:.6f}")
    print(f"  最终损失: {loss_hist[-1]:.6f}")
    print(f"  损失下降: {(1 - loss_hist[-1]/max(loss_hist[0], 1e-10)) * 100:.1f}%")

    # Hankel 反演
    print_section("Hankel 矩阵反演")
    n_hankel = 5
    hankel_inv = HankelMatrixInversion(n=n_hankel)
    x_vec = np.arange(1, 2 * n_hankel, dtype=float)
    H = hankel_inv.build_hankel(x_vec)
    print(f"  Hankel 矩阵 ({n_hankel}×{n_hankel}):")
    print(f"    {H}")
    H_inv = hankel_inv.invert_hankel(x_vec)
    # 验证
    product = H @ H_inv
    identity_error = np.max(np.abs(product - np.eye(n_hankel)))
    print(f"  H × H⁻¹ - I 的最大误差: {identity_error:.4e}")

    return {"sola_result": result}


def phase8_monte_carlo(model: StellarStructureModel, freqs: np.ndarray) -> dict:
    """
    阶段 8: 蒙特卡罗不确定性量化.
    """
    print_header("阶段 8: 蒙特卡罗不确定性量化")

    # Ziggurat 采样器
    print_section("Ziggurat 随机数发生器")
    rng = ZigguratGenerator(seed=42)
    samples_norm = rng.sample_normal(1000)
    samples_exp = rng.sample_exponential(1000)
    print(f"  正态样本: mean = {np.mean(samples_norm):.4f}, std = {np.std(samples_norm):.4f}")
    print(f"  指数样本: mean = {np.mean(samples_exp):.4f} (理论 1.0)")

    # 椭圆采样
    print_section("椭球参数空间采样")
    center = np.array([1.0, 1.0, 0.70])
    cov = np.diag([0.01, 0.01, 0.001])
    ell_sampler = EllipseParameterSampler(3, center, cov)

    samples_ell = ell_sampler.sample_on_ellipsoid(200, chi2_level=2.3, rng=rng)
    print(f"  样本数: {len(samples_ell)}")
    print(f"  样本均值: {np.mean(samples_ell, axis=0)}")
    print(f"  样本标准差: {np.std(samples_ell, axis=0)}")

    dist_stats = ell_sampler.compute_distance_statistics(200, rng=rng)
    print(f"  平均距离: {dist_stats['mean_distance']:.6f}")
    print(f"  距离方差: {dist_stats['var_distance']:.6f}")

    # MCMC
    print_section("MCMC 贝叶斯采样")
    freqs_obs = freqs[:10]
    sigma_obs = freqs_obs * 0.01

    def log_lik(p):
        nu_scale = np.sqrt(np.clip(p[0], 0.1, 10) / np.clip(p[1], 0.1, 10)**3)
        nu_model = freqs_obs * nu_scale
        return -0.5 * np.sum(((freqs_obs - nu_model) / sigma_obs)**2)

    def log_pri(p):
        if 0.1 <= p[0] <= 10 and 0.1 <= p[1] <= 10:
            return 0.0
        return -np.inf

    mcmc = MCMCSampler(log_lik, log_pri, proposal_scale=0.02)
    mcmc_result = mcmc.sample(np.array([1.0, 1.0]), n_steps=300, burn_in=50, rng=rng)
    print(f"  接受率: {mcmc_result['acceptance_rate']:.3f}")
    print(f"  后验均值: M={mcmc_result['mean'][0]:.4f}, R={mcmc_result['mean'][1]:.4f}")
    print(f"  后验标准差: σ_M={mcmc_result['std'][0]:.4f}, σ_R={mcmc_result['std'][1]:.4f}")

    return {"mcmc_result": mcmc_result}


def phase9_hypernetwork(model: StellarStructureModel, freqs: np.ndarray) -> dict:
    """
    阶段 9: 超网络参数预测与 F-adjoint 训练.
    """
    print_header("阶段 9: 超网络参数预测")

    # 超网络参数映射
    print_section("超网络参数推断")
    hyper_mapper = HypernetworkParameterMapper(n_modes=20, n_params=5)
    freq_test = freqs[:20] if len(freqs) >= 20 else np.pad(
        freqs, (0, 20 - len(freqs)), mode='edge'
    )
    params_pred = hyper_mapper.predict_parameters(freq_test)
    print(f"  预测参数:")
    print(f"    质量: {params_pred[0]:.4f} M_sun")
    print(f"    半径: {params_pred[1]:.4f} R_sun")
    print(f"    氢丰度: X = {params_pred[2]:.4f}")
    print(f"    金属丰度: Z = {params_pred[3]:.4f}")
    print(f"    混合长: α = {params_pred[4]:.4f}")

    # F-adjoint 训练
    print_section("F-adjoint 训练")
    X_train, Y_train = generate_training_data(n_samples=50, n_modes=20, n_params=7)
    trainer = FAdjointTrainer(layers=(20, 32, 32, 7), learning_rate=0.005, n_inner_iter=20)

    print(f"  训练数据: {X_train.shape[1]} 样本, {X_train.shape[0]} 维输入")
    for epoch in range(5):
        loss = trainer.train_epoch(X_train, Y_train, batch_size=16)
        if epoch == 0 or epoch == 4:
            print(f"    Epoch {epoch + 1}: loss = {loss:.6f}")

    # 时间序列编码器
    print_section("时间序列编码器")
    enc = TimeSeriesEncoder(input_dim=20, hidden_dim=32, latent_dim=8)
    h = enc.encode(freq_test)
    print(f"  输入维度: {enc.input_dim}")
    print(f"  隐表示维度: {enc.latent_dim}")
    print(f"  编码结果 (前 4): {h[:4]}")

    return {"params_pred": params_pred}


def print_summary(results: dict) -> None:
    """打印总结."""
    print_header("计算总结")

    print("  完成的计算阶段:")
    print("    1. ✓ 恒星平衡模型 (多方球 n=3)")
    print("    2. ✓ 振荡方程构造 (高阶有限差分)")
    print("    3. ✓ 稳定性分析 (非绝热 + 分岔)")
    print("    4. ✓ 模式分类 (量子数赋值)")
    print("    5. ✓ 域分解 (有限元网格)")
    print("    6. ✓ 边界编码 (六方向符号)")
    print("    7. ✓ 星震学反演 (SOLA + 伴随)")
    print("    8. ✓ 不确定性量化 (蒙特卡罗)")
    print("    9. ✓ 超网络参数预测 (F-adjoint)")

    print("\n  核心科学结果:")
    if 'freqs' in results:
        print(f"    - 频率计算完成: {len(results['freqs'])} 个模式")
    if 'boundary_word' in results:
        print(f"    - 边界编码: {results['boundary_word'][:20]}")

    print("\n  方法学创新:")
    print("    - 15 个种子项目算法的深度融合")
    print("    - 高阶有限差分 + 紧致格式 + 谱方法")
    print("    - 非绝热稳定性 + 分岔检测")
    print("    - 伴随反演 + Hankel 矩阵加速")
    print("    - Ziggurat 采样 + 椭球几何")
    print("    - F-adjoint 局部学习训练")

    print("\n" + "=" * 70)
    print("  PROJECT_256 计算完成!")
    print("=" * 70)


def main():
    """主函数: 执行完整的星震学计算流程."""
    print("=" * 70)
    print("  PROJECT_256: 计算天体物理——恒星振动模式与星震学反演")
    print("  高阶有限差分与稳定性分析 (小规模可复现实验)")
    print("=" * 70)

    start_time = time.time()
    results = {}

    try:
        # 阶段 1: 恒星结构
        model = phase1_stellar_structure()
        results["model"] = model

        # 阶段 2: 振荡方程
        osc_results = phase2_oscillation_equations(model)
        results.update(osc_results)

        # 阶段 3: 稳定性
        stab_results = phase3_stability(model)
        results.update(stab_results)

        # 阶段 4: 模式分类
        mode_results = phase4_mode_classification(model)
        results.update(mode_results)

        # 阶段 5: 域分解
        mesh_results = phase5_domain_decomposition(model)
        results.update(mesh_results)

        # 阶段 6: 边界编码
        bound_results = phase6_boundary_encoding(model)
        results.update(bound_results)

        # 阶段 7: 反演
        inv_results = phase7_inverse_problem(model, mode_results["freqs"])
        results.update(inv_results)

        # 阶段 8: 蒙特卡罗
        mc_results = phase8_monte_carlo(model, mode_results["freqs"])
        results.update(mc_results)

        # 阶段 9: 超网络
        hyper_results = phase9_hypernetwork(model, mode_results["freqs"])
        results.update(hyper_results)

        # 总结
        print_summary(results)

    except Exception as e:
        print(f"\n[ERROR] 计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n  总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
