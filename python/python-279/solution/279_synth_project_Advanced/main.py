"""
main.py - 统一入口
==================
高阶有限差分与稳定性分析的材料基因组高通量筛选框架。

科学问题:
  在掺杂 LLZO (Li7La3Zr2O12) 固态电解质中, 通过高通量计算筛选
  最优的掺杂浓度和应变状态, 使得离子电导率最大化的同时保持
  机械稳定性和热稳定性。

方法:
  1. 晶体结构编码与描述符生成
  2. CVT 自适应非均匀网格
  3. 高阶有限差分离散 (O(h^4), O(h^6))
  4. 修正 PNP 离子输运求解
  5. 2D 热传导分析 (稀疏矩阵)
  6. 谱稳定性分析 (Lorenz-96 + Lindberg)
  7. ML 代理模型加速
  8. 多目标 Pareto 筛选

运行:
  python main.py
  (零参数可运行, 完整流程 ~30-60 秒)
"""

import sys
import os
import time
import numpy as np

# 确保模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from material_constants import (
    KB, ELEMENTARY_CHARGE, LLZO_IONIC_CONDUCTIVITY, LLZO_ACTIVATION_ENERGY,
    LLZO_DIFFUSION_COEFF, LLZO_RELATIVE_PERMITTIVITY, SMALL_NUMBER,
    DOPANT_CONCENTRATIONS, TEMPERATURE_RANGE, STRAIN_RANGE,
    arrhenius_conductivity, nernst_einstein_diffusion, debye_length,
)
from crystal_descriptor import (
    llzo_unit_cell, coulomb_matrix, sine_matrix, composition_descriptor,
    lattice_volume, metric_tensor, reciprocal_lattice,
)
from grid_generator import (
    cvt_1d_lloyd, circle_arc_grid, structured_triangle_mesh,
    triangulation_triangle_neighbors, grid_quality_metrics,
)
from high_order_fd import (
    central_fd_1d_uniform, nonuniform_fd_matrix, compact_pade_1d,
    fornberg_weights, truncation_error_analysis,
)
from quadrature_rules import (
    clenshaw_curtis_1d, gauss_legendre_1d, wedge_quadrature,
    pyramid_witherden_rule, clenshaw_curtis_sparse_grid, sparse_grid_integrate,
    quadrature_exactness_test,
)
from ion_transport import ModifiedPNPSolver
from thermal_diffusion import ThermalDiffusion2D, thermal_conductivity_mixture
from stability_analysis import (
    lorenz96_rhs, lorenz96_integrate_rk4, lyapunov_exponent,
    lorenz96_bifurcation_scan, lindberg_ring_oscillator, lindberg_exact_response,
    spectral_stability, von_neumann_stability_heat, cahn_hilliard_dispersion,
)
from mesh_topology import (
    mesh_etoe, fem_basis_1d_p2, fem_basis_triangle_p1, fem_basis_triangle_p2,
    fem_basis_tet_p1, triangle_jacobian, grain_boundary_mesh, assign_grain_ids,
)
from time_integrator import (
    rk4_step, rk4_integrate, adaptive_integrate, backward_euler_step,
    phase_field_rhs, double_well_derivative, cfl_timestep_diffusion,
)
from ml_potential import (
    GaussianKernelRegressor, generate_llzo_training_data,
    arrhenius_fit, predict_optimal_doping, cross_validate,
)
from high_throughput_screen import (
    evaluate_candidate, screening_parameter_grid, run_high_throughput_screening,
    screening_summary, pareto_front, elastic_constants_llzo, born_stability_check,
    vrh_moduli, debye_temperature, screening_sparse_grid,
)


def print_section(title):
    """打印章节标题"""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def phase1_crystal_descriptor():
    """Phase 1: 晶体结构编码与描述符"""
    print_section("Phase 1: 晶体结构编码与描述符生成")

    positions, atomic_numbers, a_lat = llzo_unit_cell()
    print(f"  LLZO 立方晶格常数: a = {a_lat*1e10:.3f} Å")
    print(f"  代表性原子数: {len(positions)}")

    # 晶格体积
    V = lattice_volume(a_lat, a_lat, a_lat, 90.0, 90.0, 90.0)
    print(f"  晶胞体积: V = {V*1e30:.2f} Å³")

    # 度量张量
    G = metric_tensor(a_lat, a_lat, a_lat, 90.0, 90.0, 90.0)
    print(f"  度量张量 det(G) = {np.linalg.det(G):.4e}")

    # 倒格矢
    G_star = reciprocal_lattice(a_lat, a_lat, a_lat, 90.0, 90.0, 90.0)
    print(f"  倒格矢度量张量 G* (0,0) = {G_star[0,0]:.4e}")

    # Coulomb matrix
    cm = coulomb_matrix(positions, atomic_numbers, max_atoms=20)
    print(f"  Coulomb matrix 前 5 特征值: {cm[:5]}")

    # 组成描述符
    formula = {'Li': 7, 'La': 3, 'Zr': 2, 'O': 12}
    desc = composition_descriptor(formula)
    print(f"  组成描述符 (8D): {desc}")
    return desc


def phase2_grid_generation():
    """Phase 2: 自适应非均匀网格生成"""
    print_section("Phase 2: 自适应网格生成 (CVT + 圆弧 + 三角)")

    # CVT 1D 非均匀网格
    z_cvt, energy_hist = cvt_1d_lloyd(
        n_generators=20, n_samples=2000, n_steps=50,
        density_mode='gradient', seed=42,
    )
    print(f"  CVT 生成点数: {len(z_cvt)}")
    print(f"  CVT 能量 (初→终): {energy_hist[0]:.4e} → {energy_hist[-1]:.4e}")
    print(f"  网格间距范围: [{np.min(np.diff(z_cvt)):.4f}, {np.max(np.diff(z_cvt)):.4f}]")

    # 圆弧网格
    R = 5e-6  # 5 微米
    x_arc, y_arc, theta_arc = circle_arc_grid(
        R, theta_start=0.0, theta_end=2*np.pi, n_points=40,
        clustering='tanh', concentration_param=1.5,
    )
    print(f"  圆弧网格点数: {len(x_arc)}")
    print(f"  圆弧半径: R = {R*1e6:.1f} μm")

    # 三角形网格
    node_xy, tri_node = structured_triangle_mesh(15, 15, Lx=1e-5, Ly=1e-5)
    n_node = node_xy.shape[1]
    n_tri = tri_node.shape[1]
    print(f"  三角网格: {n_node} 节点, {n_tri} 单元")

    # 三角形邻接
    neighbors = triangulation_triangle_neighbors(tri_node)
    n_boundary_edges = np.sum(neighbors == 0)
    print(f"  邻接表构建: 边界边数 = {n_boundary_edges // 2}")

    # 网格质量
    quality = grid_quality_metrics(node_xy, tri_node)
    print(f"  网格质量: 最小角 = {quality['min_angle_deg']:.1f}°, "
          f"最大长宽比 = {quality['max_aspect']:.2f}")

    return z_cvt, node_xy, tri_node


def phase3_high_order_fd(z_cvt):
    """Phase 3: 高阶有限差分算子"""
    print_section("Phase 3: 高阶有限差分构造与精度验证")

    N = len(z_cvt) - 2
    h = 1.0 / (N + 1)

    # 各阶差分矩阵
    for order in [2, 4, 6]:
        D1 = central_fd_1d_uniform(N, h, order=order, deriv=1)
        D2 = central_fd_1d_uniform(N, h, order=order, deriv=2)
        nnz = D1.nnz + D2.nnz
        print(f"  O(h^{order}) 差分: D1 nnz={D1.nnz}, D2 nnz={D2.nnz}")

    # 紧致 Pade 差分
    A_pade, B_pade, D_eff = compact_pade_1d(N, h, alpha=0.25)
    print(f"  紧致 Pade (alpha=1/4): 有效算子谱半径 = {np.max(np.abs(np.linalg.eigvals(D_eff))):.4f}")

    # 非均匀网格差分
    D1_nonuni = nonuniform_fd_matrix(z_cvt, deriv=1, bc_type='dirichlet')
    D2_nonuni = nonuniform_fd_matrix(z_cvt, deriv=2, bc_type='dirichlet')
    print(f"  非均匀网格差分: D1 nnz={D1_nonuni.nnz}, D2 nnz={D2_nonuni.nnz}")

    # Fornberg 权重 (5点模板, 一阶导数)
    # 标准 O(h^4) 中心差分: [1, -8, 0, 8, -1] / (12h)
    x_nodes = np.array([-2, -1, 0, 1, 2], dtype=np.float64) * h
    w = fornberg_weights(0.0, x_nodes, deriv_order=1)
    # 验证权重和 (应接近 0 对常数函数)
    print(f"  Fornberg 5点一阶导数权重: {w}")
    print(f"    权重和 (应为 0): {np.sum(w):.4e}")

    # 截断误差分析
    test_func = lambda x: np.sin(2*np.pi*x)
    h_values = [1.0/20, 1.0/40, 1.0/80]
    err_4, obs_order_4 = truncation_error_analysis(test_func, [0, 1], h_values, 4, deriv=1)
    print(f"  O(h^4) 精度验证: 观测阶数 = {obs_order_4:.2f}")

    return D1_nonuni, D2_nonuni


def phase4_quadrature_rules():
    """Phase 4: 高阶求积规则"""
    print_section("Phase 4: 高阶求积规则 (楔形/金字塔/稀疏网格)")

    # Gauss-Legendre
    x_gl, w_gl = gauss_legendre_1d(5)
    exactness_gl = quadrature_exactness_test(x_gl, w_gl, max_degree=10)
    print(f"  5点 Gauss-Legendre 精确度: {exactness_gl} 次多项式")

    # Clenshaw-Curtis
    x_cc, w_cc = clenshaw_curtis_1d(7)
    exactness_cc = quadrature_exactness_test(x_cc, w_cc, max_degree=10)
    print(f"  7点 Clenshaw-Curtis 精确度: {exactness_cc} 次多项式")

    # 楔形求积
    w_pts, w_wts = wedge_quadrature(tri_order=3, z_order=3)
    vol_wedge = np.sum(w_wts)
    print(f"  楔形求积: {len(w_pts)} 点, 体积 = {vol_wedge:.6f} (精确: 0.5)")

    # 金字塔求积
    p_pts, p_wts = pyramid_witherden_rule(order=3)
    vol_pyramid = np.sum(p_wts)
    print(f"  金字塔求积 (Witherden): {len(p_pts)} 点, 体积 = {vol_pyramid:.6f} (精确: 1.333333)")

    # 稀疏网格积分
    def test_func_3d(x):
        return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2])
    integral, n_pts = sparse_grid_integrate(test_func_3d, d=3, q=4)
    print(f"  3D 稀疏网格积分 (q=4): {n_pts} 点, I = {integral:.6f}")

    # 稀疏网格维度扩展
    for d in [2, 3, 4, 5]:
        pts, wts = clenshaw_curtis_sparse_grid(d, q=4)
        print(f"    d={d}, q=4: {len(pts)} 点 (全张量需要 ~{7**d} 点)")


def phase5_ion_transport(z_cvt):
    """Phase 5: 修正 PNP 离子输运"""
    print_section("Phase 5: 修正 Poisson-Nernst-Planck 离子输运")

    N = 50
    L = 1e-6  # 1 微米电解质
    T = 300.0

    solver = ModifiedPNPSolver(N, L, T_kelvin=T, c_ref=1000.0, fd_order=4)

    # 初始条件: 线性浓度分布 + 线性电势
    x = solver.x
    phi_init = -0.5 + x / L  # 0 到 0.5 V
    c_init = 1000.0 * (1.0 + 0.5 * np.sin(np.pi * x / L))
    solver.set_initial_conditions(phi_init[1:-1], c_init[1:-1])

    # 时间演化
    dt = 1e-6
    n_steps = 5
    print(f"  电解质厚度: L = {L*1e6:.1f} μm")
    print(f"  网格点数: N = {N}")
    print(f"  Debye 长度: λ_D = {solver.lambda_D*1e9:.2f} nm")
    print(f"  热电压: V_T = {solver.V_T*1e3:.2f} mV")
    print(f"  时间步长: dt = {dt*1e6:.2f} μs")

    for step in range(n_steps):
        res = solver.step_backward_euler(dt, c_max=5000.0)
    print(f"  Picard 收敛残差: {res:.4e}")

    sigma = solver.ionic_conductivity()
    print(f"  计算离子电导率: σ = {sigma:.4e} S/m")
    print(f"  参考 LLZO 电导率: σ_ref = {LLZO_IONIC_CONDUCTIVITY:.4e} S/m")

    mass = solver.mass_conservation_check()
    print(f"  质量守恒: ∫c dx = {mass:.4e}")

    return solver


def phase6_thermal_diffusion(node_xy, tri_node):
    """Phase 6: 2D 热传导"""
    print_section("Phase 6: 2D 有限元热传导")

    thermal = ThermalDiffusion2D(node_xy, tri_node)
    print(f"  节点数: {thermal.n_node}, 单元数: {thermal.n_tri}")
    print(f"  热扩散率: α = {thermal.alpha:.4e} m²/s")

    # 边界条件: 四个边都施加 Dirichlet T=300K
    nx = 15
    ny = 15
    boundary_nodes = []
    for j in range(ny):
        boundary_nodes.append(j)                  # 左边界 x=0
        boundary_nodes.append((nx-1)*ny + j)      # 右边界 x=Lx
    for i in range(1, nx - 1):
        boundary_nodes.append(i*ny)               # 下边界 y=0
        boundary_nodes.append(i*ny + (ny-1))      # 上边界 y=Ly
    boundary_nodes = list(set(boundary_nodes))
    thermal.set_boundary_dirichlet(boundary_nodes, 300.0)

    # 稳态 (有热源) - 焦耳热源 1 MW/m^3 (适中)
    Q_source = np.ones(thermal.n_node) * 1e6
    T_steady = thermal.solve_steady_state(Q_source)
    print(f"  稳态温度范围: [{np.min(T_steady):.1f}, {np.max(T_steady):.1f}] K")

    # 时间步进
    dt = 1e-4
    T_prev = T_steady.copy()
    for _ in range(3):
        T_new = thermal.step_backward_euler(dt, Q_source)
    print(f"  瞬态温度 (3步后): [{np.min(T_new):.1f}, {np.max(T_new):.1f}] K")

    # 复合材料热导率
    kappa_mix = thermal_conductivity_mixture(1.8, 30.0, 0.3, model='maxwell')
    print(f"  Maxwell 有效热导率 (30% 填充): κ_eff = {kappa_mix:.3f} W/(m·K)")

    # Biot 数
    Bi = thermal.biot_number(10.0, 1e-5)
    tau = thermal.thermal_diffusion_time(1e-5)
    print(f"  Biot 数: Bi = {Bi:.4f}")
    print(f"  热扩散特征时间: τ = {tau:.4e} s")


def phase7_stability_analysis():
    """Phase 7: 线性稳定性与混沌分析"""
    print_section("Phase 7: 稳定性分析与混沌诊断")

    # Lorenz-96 (N=8)
    N = 8
    F_values = [2.0, 4.0, 6.0, 8.0]
    print(f"  Lorenz-96 系统维度: N = {N}")
    for F in F_values:
        y0 = np.random.RandomState(int(F * 1000)).randn(N) * 0.1
        traj = lorenz96_integrate_rk4(y0, F, N, dt=0.05, n_steps=2000)
        lam, hist = lyapunov_exponent(traj[-1], F, N, dt=0.05, n_steps=2000, renorm_interval=10)
        print(f"    F={F:.1f}: λ_max = {lam:+.4f}, 状态={'混沌' if lam > 0 else '规则'}")

    # Lindberg 环形振荡器
    n_stages = 5
    omega_0, A_crit, poles = lindberg_ring_oscillator(n_stages, gain_per_stage=1.0)
    print(f"  Lindberg 环形振荡器 (N={n_stages}):")
    print(f"    振荡频率 ω₀ = {omega_0:.4f} rad/s")
    print(f"    临界增益 A_crit = {A_crit:.4f}")
    stable_poles = np.sum(np.real(poles) < 0)
    print(f"    稳定极点数: {stable_poles}/{n_stages}")

    # 精确响应
    t_vals = np.linspace(0, 10, 50)
    v_resp = lindberg_exact_response(t_vals, n_stages, 1.0)
    print(f"    阶跃响应范围: [{np.min(v_resp):.4f}, {np.max(v_resp):.4f}]")

    # von Neumann 热方程稳定性
    alpha = 1e-5
    h = 1e-4
    dt = 1e-3
    vns = von_neumann_stability_heat(alpha, dt, h)
    print(f"  热方程 FTCS von Neumann 分析:")
    print(f"    扩散数 r = {vns['r']:.4f}, 稳定 = {vns['is_stable']}")
    print(f"    临界时间步 dt_crit = {vns['dt_critical']:.4e} s")

    # Cahn-Hilliard 色散
    k_array = np.linspace(0, 100, 200)
    f_pp = -1.0  # spinodal
    omega_ch, k_max, k_c = cahn_hilliard_dispersion(f_pp, 1e-18, 1e-15, k_array)
    print(f"  Cahn-Hilliard 色散 (spinodal):")
    print(f"    最快生长波数 k_max = {k_max:.2e}")
    print(f"    临界波数 k_c = {k_c:.2e}")


def phase8_mesh_and_basis(node_xy, tri_node):
    """Phase 8: 网格拓扑与 FEM 基函数"""
    print_section("Phase 8: 网格拓扑与 FEM 基函数")

    # ETOE
    etoe = mesh_etoe(3, tri_node.shape[1], tri_node)
    print(f"  ETOE 表: {etoe.shape[1]} 个单元")
    n_boundary = np.sum(etoe == 0)
    print(f"  边界边数: {n_boundary}")

    # 基函数测试
    xi_test = 0.3
    N_p2, dN_p2 = fem_basis_1d_p2(xi_test)
    print(f"  1D P2 基函数 (xi={xi_test}): N = {N_p2}, sum(N) = {np.sum(N_p2):.6f}")

    L1, L2 = 0.3, 0.4
    N_tri, dN_tri = fem_basis_triangle_p1(L1, L2)
    print(f"  2D P1 三角基 (L1={L1}, L2={L2}): N = {N_tri}, sum = {np.sum(N_tri):.6f}")

    N_p2_2d, dN_p2_2d = fem_basis_triangle_p2(L1, L2)
    print(f"  2D P2 三角基 (6节点): sum(N) = {np.sum(N_p2_2d):.6f}")

    N_tet, dN_tet = fem_basis_tet_p1(0.2, 0.3, 0.25)
    print(f"  3D P1 四面体基: sum(N) = {np.sum(N_tet):.6f}")

    # Jacobian
    J, detJ = triangle_jacobian(node_xy, tri_node[:, 0])
    print(f"  三角形 #0 Jacobian det = {detJ:.4e}")

    # 多晶网格
    node_gb, tri_gb = grain_boundary_mesh(16, 1e-6, 1e-5, 1e-5, seed=42)
    grain_ids = assign_grain_ids(node_gb, 16, 1e-5, 1e-5, seed=42)
    n_grains = len(np.unique(grain_ids))
    print(f"  多晶网格: {node_gb.shape[1]} 节点, {n_grains} 晶粒")


def phase9_time_integration():
    """Phase 9: 时间积分与相场动力学"""
    print_section("Phase 9: 时间积分器与相场动力学")

    # 相场 Allen-Cahn 动力学 (spinodal decomposition)
    N = 64
    L = 1.0
    h = L / N
    x = np.linspace(0, L, N, endpoint=False)
    phi0 = 0.5 + 0.1 * np.cos(2*np.pi*x/L) + 0.05 * np.random.RandomState(42).randn(N)

    from high_order_fd import central_fd_1d_uniform
    D2 = central_fd_1d_uniform(N, h, order=4, deriv=2)

    M_mob = 1.0
    kappa_g = 0.01
    def f_prime(phi):
        return double_well_derivative(phi, A=1.0)

    def rhs(t, phi):
        return phase_field_rhs(phi, M_mob, kappa_g, f_prime, D2)

    dt = cfl_timestep_diffusion(kappa_g, h, dim=1, safety=0.3)
    print(f"  相场网格: N={N}, h={h:.4f}")
    print(f"  CFL 时间步: dt = {dt:.6f}")

    # RK4 积分
    t_arr, phi_arr = rk4_integrate(rhs, (0.0, 0.1), phi0, n_steps=100)
    print(f"  RK4 积分 100 步:")
    print(f"    相场范围: [{np.min(phi_arr[-1]):.4f}, {np.max(phi_arr[-1]):.4f}]")
    print(f"    平均相场: {np.mean(phi_arr[-1]):.4f}")

    # 自适应积分 (小系统)
    def simple_ode(t, y):
        return np.array([-0.5 * y[0] + np.sin(t)])
    t_adapt, y_adapt, n_acc, n_rej = adaptive_integrate(simple_ode, (0, 5), np.array([1.0]), tol=1e-8)
    print(f"  自适应 RK45: {n_acc} 步接受, {n_rej} 步拒绝")


def phase10_ml_and_screening():
    """Phase 10: ML 代理模型 + 高通量筛选"""
    print_section("Phase 10: ML 代理模型与高通量筛选")

    # 训练数据生成
    X_train, y_train, data_dict = generate_llzo_training_data(n_samples=100, seed=42)
    print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

    # ML 模型训练
    model = GaussianKernelRegressor(length_scale=1.0, sigma_f=1.0, lambda_reg=1e-2)
    model.fit(X_train, y_train)
    R2_train = model.score(X_train, y_train)
    print(f"  KRR 训练 R² = {R2_train:.4f}")

    # 交叉验证
    rmse_cv, std_cv, r2_cv = cross_validate(X_train, y_train, n_folds=5)
    print(f"  5-fold CV: RMSE = {rmse_cv:.4f} ± {std_cv:.4f}, R² = {r2_cv:.4f}")

    # Arrhenius 拟合
    T_arr = np.array([250, 275, 300, 325, 350], dtype=np.float64)
    kB_eV = KB / ELEMENTARY_CHARGE
    E_a = 0.37  # eV
    sigma_arr = LLZO_IONIC_CONDUCTIVITY * np.exp(-E_a / (kB_eV * T_arr))
    E_a_fit, sigma_0_fit, R2_arr = arrhenius_fit(T_arr, sigma_arr)
    print(f"  Arrhenius 拟合: E_a = {E_a_fit:.4f} eV, R² = {R2_arr:.6f}")

    # 最优掺杂预测
    x_opt, sigma_opt, all_sigma = predict_optimal_doping(model, strain=0.0, temperature=300.0)
    print(f"  ML 预测最优掺杂: x = {x_opt:.3f}, σ = {sigma_opt:.4e} S/m")

    # 高通量筛选
    # 粗网格快速筛选
    x_coarse = np.linspace(0.0, 0.3, 5)
    strain_coarse = np.array([-0.01, 0.0, 0.01])
    T_coarse = np.array([300.0])
    grid = screening_parameter_grid(x_coarse, strain_coarse, T_coarse)
    print(f"\n  高通量筛选: {len(grid)} 个候选材料")

    results = run_high_throughput_screening(grid)
    summary = screening_summary(results)
    print(f"  机械稳定候选: {summary['n_mech_stable']}/{summary['n_candidates']}")
    print(f"  电导率范围: [{summary['sigma_min']:.4e}, {summary['sigma_max']:.4e}] S/m")
    print(f"  最优候选: x={summary['best_candidate']['x_dopant']:.3f}, "
          f"ε={summary['best_candidate']['strain']:.3f}")
    print(f"  最优 σ = {summary['best_candidate']['sigma_S_m']:.4e} S/m, "
          f"score = {summary['best_candidate']['score']:.4f}")

    # Pareto 前沿
    objectives = np.array([[r['sigma_S_m'], r['E_GPa'], r['Theta_D']] for r in results])
    # 归一化
    obj_norm = objectives / np.max(np.abs(objectives), axis=0)
    pareto_mask = pareto_front(obj_norm)
    n_pareto = np.sum(pareto_mask)
    print(f"  Pareto 前沿解数: {n_pareto}")

    # 弹性与 Debye 温度
    C11, C12, C44 = elastic_constants_llzo(0.15, 0.0)
    stable, viol = born_stability_check(C11, C12, C44)
    K, G, E_mod, nu = vrh_moduli(C11, C12, C44)
    Theta_D = debye_temperature(K, G, 5120.0, 1.8e29, 50.0)
    print(f"  掺杂 x=0.15 弹性: C11={C11/1e9:.1f} GPa, K={K/1e9:.1f} GPa, G={G/1e9:.1f} GPa")
    print(f"  Debye 温度: Θ_D = {Theta_D:.1f} K")

    return model, results


def main():
    """主执行流程"""
    print("=" * 70)
    print("  PROJECT 279: 计算材料基因组高通量筛选框架")
    print("  高阶有限差分与稳定性分析 (LLZO 固态电解质)")
    print("=" * 70)
    t_start = time.time()

    # Phase 1
    desc = phase1_crystal_descriptor()
    # Phase 2
    z_cvt, node_xy, tri_node = phase2_grid_generation()
    # Phase 3
    D1_nonuni, D2_nonuni = phase3_high_order_fd(z_cvt)
    # Phase 4
    phase4_quadrature_rules()
    # Phase 5
    pnp_solver = phase5_ion_transport(z_cvt)
    # Phase 6
    phase6_thermal_diffusion(node_xy, tri_node)
    # Phase 7
    phase7_stability_analysis()
    # Phase 8
    phase8_mesh_and_basis(node_xy, tri_node)
    # Phase 9
    phase9_time_integration()
    # Phase 10
    ml_model, screen_results = phase10_ml_and_screening()

    # 总结
    t_end = time.time()
    print_section("计算完成")
    print(f"  总运行时间: {t_end - t_start:.2f} 秒")
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  NumPy 版本: {np.__version__}")
    print(f"  总模块数: 12 个 Python 模块")
    print(f"  覆盖源项目: 15 个种子项目全部融入")

    print("\n  科学贡献:")
    print("    - 基于 CVT 自适应网格的高阶有限差分法")
    print("    - 修正 PNP 离子输运 (含 steric 效应)")
    print("    - 2D 稀疏矩阵热传导 (含焦耳热耦合)")
    print("    - Lorenz-96 混沌诊断 + Lindberg 精确解基准")
    print("    - KRR 机器学习代理模型加速筛选")
    print("    - 多目标 Pareto 高通量筛选 LLZO 固态电解质")

    print("\n" + "=" * 70)
    print("  计算成功完成, 无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()
